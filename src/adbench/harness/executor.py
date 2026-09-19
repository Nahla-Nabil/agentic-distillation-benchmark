"""The execution loop: call tool -> read result -> decide next step ->
handle errors, repeated for a task's configured chain length (1, 3, or 5
steps per configs/experiment.yaml).

The model under test is injected rather than imported, so the exact same
loop drives the base student, the SFT-only student, the distilled student,
and (for layer analysis) the teacher:

    ModelFn = Callable[[list[dict]], str]
        Takes the running chat-message history, returns the model's raw
        text completion for the next turn.

Tool-call protocol: the model is expected to emit a tool call wrapped in
<tool_call>...</tool_call> tags containing a JSON object with "name" and
"arguments" keys — this matches Qwen's own tool-calling chat-template
convention, so no format translation is needed when real Qwen3 models are
wired in later. Anything else is treated as a (premature, for an
intermediate step) final answer.

A task's chain_length is the number of sequential tool-call steps it
requires; task success means completing all of them (with retries as
configured), not necessarily ending with a final natural-language answer —
this matches the research question's framing of "multi-step tool-use task
success", not open-ended conversation.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from adbench.harness.errors import (
    HarnessError,
    MalformedCallError,
    ToolExecutionError,
    UnknownToolError,
    WrongToolError,
)
from adbench.harness.tools import ToolRegistry

ModelFn = Callable[[list[dict[str, Any]]], str]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass
class StepOutcome:
    """error_type is the exception CLASS NAME (e.g. "WrongToolError",
    "MalformedCallError") of the step's *last* attempt — None if it
    succeeded outright. A failed step's error_type/error reflect only its
    final (retries-exhausted) attempt, even if earlier attempts on the same
    step failed differently. Used by evaluation/metrics.py's
    error_breakdown() to categorize failures by harness.errors' exception
    hierarchy (protocol vs tool-execution) without re-parsing free-text
    error messages.
    """

    step_index: int
    tool_name: str | None
    succeeded: bool
    recovered_from_error: bool = False
    error: str | None = None
    error_type: str | None = None
    # Did the FIRST attempt call the expected tool with exactly the expected
    # arguments? None when the task carries no ground-truth arguments. Logged
    # only: it does not affect success, retries or the messages the model sees.
    arguments_match: bool | None = None


@dataclass
class TaskState:
    task_id: str
    chain_length: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[StepOutcome] = field(default_factory=list)
    final_success: bool = False


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) <= 1e-6 * max(1.0, abs(expected))
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Same argument names, and equal values (numbers within a tiny tolerance,
    strings ignoring case and surrounding whitespace, JSON types otherwise strict)."""
    return set(actual) == set(expected) and all(_values_match(actual[k], expected[k]) for k in expected)


def parse_model_output(text: str) -> ToolCall | FinalAnswer:
    """Parse one model completion into either a ToolCall or a FinalAnswer.

    Raises MalformedCallError if a <tool_call> block is present but isn't
    valid JSON, or is missing the required "name" field.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return FinalAnswer(text=text.strip())

    raw_json = match.group(1)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise MalformedCallError(f"Tool call is not valid JSON: {e}") from None

    if not isinstance(payload, dict) or "name" not in payload:
        raise MalformedCallError('Tool call JSON must be an object with a "name" field.')

    name = payload["name"]
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise MalformedCallError('Tool call "arguments" must be a JSON object.')

    return ToolCall(name=name, arguments=arguments)


def build_system_prompt(registry: ToolRegistry) -> str:
    """Human-readable tool listing + protocol instructions, for the first
    message of a task's conversation."""
    tool_lines = []
    for spec in registry.describe_all():
        tool_lines.append(
            f"- {spec['name']}: {spec['description']} "
            f"(parameters schema: {json.dumps(spec['parameters'])})"
        )
    tools_block = "\n".join(tool_lines)
    return (
        "You are an assistant that can call tools to complete a task.\n"
        "Available tools:\n"
        f"{tools_block}\n\n"
        "To call a tool, respond with exactly one block of the form:\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>\n'
        "You will then be given the tool's result and asked to decide the "
        "next step. If a tool call fails, read the error and decide whether "
        "to retry, adjust your arguments, or try a different tool."
    )


def _should_inject_error(task: Any, step_index: int, attempt: int) -> bool:
    """Force the first attempt at `injected_error_at_step` to fail, so the
    harness can score whether the model recovers on a later attempt."""
    return (
        getattr(task, "injected_error_at_step", None) == step_index
        and attempt == 0
    )


def _run_step(
    model_fn: ModelFn,
    state: TaskState,
    task: Any,
    registry: ToolRegistry,
    step_index: int,
    max_retries: int,
) -> StepOutcome:
    expected_sequence = getattr(task, "expected_tool_sequence", None) or []
    expected_name = expected_sequence[step_index] if step_index < len(expected_sequence) else None
    expected_arguments = getattr(task, "expected_arguments", None)
    expected_args = (
        expected_arguments[step_index]
        if expected_arguments and step_index < len(expected_arguments) else None
    )
    first_attempt_args_match: bool | None = None if expected_args is None else False

    last_error: str | None = None
    last_error_type: str | None = None
    tool_name_attempted: str | None = None

    for attempt in range(max_retries + 1):
        raw = model_fn(state.messages)
        state.messages.append({"role": "assistant", "content": raw})

        try:
            parsed = parse_model_output(raw)

            if isinstance(parsed, FinalAnswer):
                raise MalformedCallError(
                    "Model produced a final answer instead of the expected tool call."
                )

            tool_name_attempted = parsed.name
            if attempt == 0 and expected_args is not None and parsed.name == expected_name:
                first_attempt_args_match = arguments_match(parsed.arguments, expected_args)

            if expected_name is not None and parsed.name != expected_name:
                raise WrongToolError(
                    f"Expected tool {expected_name!r} at step {step_index}, got {parsed.name!r}."
                )
            if not registry.has(parsed.name):
                raise UnknownToolError(f"Unknown tool {parsed.name!r}.")
            if _should_inject_error(task, step_index, attempt):
                raise ToolExecutionError(
                    f"Simulated injected failure at step {step_index} (attempt {attempt})."
                )

            result = registry.call(parsed.name, parsed.arguments)

        except HarnessError as e:
            last_error = str(e)
            last_error_type = type(e).__name__
            state.messages.append({"role": "tool", "content": f"ERROR: {last_error}"})
            continue

        # Success.
        state.messages.append({"role": "tool", "content": json.dumps(result)})
        return StepOutcome(
            step_index=step_index,
            tool_name=parsed.name,
            succeeded=True,
            recovered_from_error=(attempt > 0),
            error=None,
            error_type=None,
            arguments_match=first_attempt_args_match,
        )

    return StepOutcome(
        step_index=step_index,
        tool_name=tool_name_attempted,
        succeeded=False,
        recovered_from_error=False,
        error=last_error,
        error_type=last_error_type,
        arguments_match=first_attempt_args_match,
    )


def run_task(
    model_fn: ModelFn,
    task: Any,
    registry: ToolRegistry,
    max_retries_per_step: int = 2,
) -> TaskState:
    """Run one task's full chain: call -> read -> decide -> handle errors,
    for `task.chain_length` steps. Stops early (final_success=False) on the
    first step that fails to succeed within max_retries_per_step."""
    state = TaskState(
        task_id=task.task_id,
        chain_length=task.chain_length,
        messages=[
            {"role": "system", "content": build_system_prompt(registry)},
            {"role": "user", "content": task.user_goal},
        ],
    )

    for step_index in range(task.chain_length):
        outcome = _run_step(model_fn, state, task, registry, step_index, max_retries_per_step)
        state.steps.append(outcome)
        if not outcome.succeeded:
            state.final_success = False
            return state

    state.final_success = len(state.steps) == task.chain_length and all(
        s.succeeded for s in state.steps
    )
    return state
