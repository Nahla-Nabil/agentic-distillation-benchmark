"""End-to-end check tying tools.py -> executor.py -> tasks.py together: every
synthetic task, run with a correctly-scripted model, should complete
successfully — including the tasks with an injected error, which must show
up as a recovered (not failed) step. This is the same run_task() call
evaluation/run_eval.py will make against real models later.

The harness's grading criterion is the *tool-name* sequence (executor.py
checks parsed.name against task.expected_tool_sequence step by step) — it
does not verify that a step's arguments derive from an earlier step's
result (see tasks.py's module docstring, "KNOWN LIMITATION"). So a single
set of plausible, always-valid arguments per tool name is enough to
exhaustively test every one of the 121 synthetic tasks (across all three
chain lengths) end to end, without hand-tuning arguments per task.
"""

import json

import pytest

from adbench.harness.executor import run_task
from adbench.harness.tasks import build_synthetic_tasks
from adbench.harness.tools import build_demo_registry

# One always-valid argument set per demo tool — used regardless of what a
# task's user_goal text actually asks for, since the executor never checks
# argument values, only tool names (see module docstring above).
DEFAULT_ARGS_BY_TOOL = {
    "get_weather": {"city": "Paris"},
    "calculator": {"expression": "1 + 1"},
    "search_knowledge_base": {"query": "lora"},
    "get_current_time": {"timezone": "utc"},
    "convert_temperature": {"value": 100, "from_unit": "F", "to_unit": "C"},
    "compare_numbers": {"a": 5, "b": 3},
}


def _make_model_fn(expected_sequence):
    def model_fn(messages):
        success_count = sum(
            1 for m in messages
            if m["role"] == "tool" and not m["content"].startswith("ERROR:")
        )
        name = expected_sequence[success_count]
        args = DEFAULT_ARGS_BY_TOOL[name]
        return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'

    return model_fn


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_every_synthetic_task_succeeds_with_a_correct_model(chain_length):
    registry = build_demo_registry()
    tasks = build_synthetic_tasks(chain_length, registry)
    assert len(tasks) >= 30  # exhaustively covering the full generated set, not a sample

    for task in tasks:
        model_fn = _make_model_fn(task.expected_tool_sequence)

        state = run_task(model_fn, task, registry)

        assert state.final_success is True, f"{task.task_id} did not succeed: {state.steps}"
        assert len(state.steps) == task.chain_length

        if task.injected_error_at_step is not None:
            injected_step = state.steps[task.injected_error_at_step]
            assert injected_step.succeeded is True
            assert injected_step.recovered_from_error is True
            # no other step should have been forced to fail
            for i, step in enumerate(state.steps):
                if i != task.injected_error_at_step:
                    assert step.recovered_from_error is False


def test_default_args_cover_every_tool_the_registry_has():
    """If tools.py grows a new tool, this fails loudly instead of the
    integration test above silently skipping tasks that use it."""
    registry = build_demo_registry()
    assert set(DEFAULT_ARGS_BY_TOOL) == set(registry.list_names())


# --- dependency-feasibility check: a model that actually reads and computes
# from prior results (not just a fixed default per tool) can also complete
# the chains — demonstrating the designed dependencies are real and
# completable, even though the grader above doesn't independently enforce
# argument-level correctness (see tasks.py's "KNOWN LIMITATION"). ---

def _extract_number(tool_result_json: str, *keys: str) -> float:
    result = json.loads(tool_result_json)
    for key in keys:
        if key in result:
            return result[key]
    raise KeyError(f"None of {keys} found in {result}")


def _dependency_aware_model_fn(expected_sequence):
    """Walks the actual message history and, for convert_temperature /
    compare_numbers steps, derives arguments FROM the most recent relevant
    prior tool result instead of a fixed default — proving a step really
    can consume an earlier step's output through this harness."""

    def model_fn(messages):
        tool_messages = [
            m for m in messages
            if m["role"] == "tool" and not m["content"].startswith("ERROR:")
        ]
        success_count = len(tool_messages)
        name = expected_sequence[success_count]

        if name == "convert_temperature" and tool_messages:
            # Use the most recent get_weather result if there is one.
            for m in reversed(tool_messages):
                payload = json.loads(m["content"])
                if "temperature_f" in payload:
                    args = {"value": payload["temperature_f"], "from_unit": "F", "to_unit": "C"}
                    break
            else:
                args = DEFAULT_ARGS_BY_TOOL[name]
        elif name == "compare_numbers" and len(tool_messages) >= 2:
            # Compare the two most recent numeric-bearing results.
            numbers = []
            for m in reversed(tool_messages):
                payload = json.loads(m["content"])
                for key in ("converted_value", "temperature_f", "result"):
                    if key in payload:
                        numbers.append(payload[key])
                        break
                if len(numbers) == 2:
                    break
            args = {"a": numbers[0], "b": numbers[-1]} if len(numbers) == 2 else DEFAULT_ARGS_BY_TOOL[name]
        else:
            args = DEFAULT_ARGS_BY_TOOL[name]

        return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'

    return model_fn


@pytest.mark.parametrize("chain_length", [3, 5])
def test_dependency_chains_are_completable_by_a_model_that_actually_chains_values(chain_length):
    """Not exhaustive (only tasks whose sequence includes convert_temperature
    or compare_numbers exercise the value-threading branches above) — this
    is evidence the designed dependencies are real, not a replacement for
    the exhaustive default-args check above."""
    registry = build_demo_registry()
    tasks = build_synthetic_tasks(chain_length, registry)
    exercised = 0
    for task in tasks:
        model_fn = _dependency_aware_model_fn(task.expected_tool_sequence)
        state = run_task(model_fn, task, registry)
        assert state.final_success is True, f"{task.task_id} did not succeed: {state.steps}"
        if "convert_temperature" in task.expected_tool_sequence or "compare_numbers" in task.expected_tool_sequence:
            exercised += 1
    assert exercised > 0
