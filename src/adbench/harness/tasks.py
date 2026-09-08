"""Task/scenario definitions: the multi-step agentic tasks the harness runs,
grouped by chain length (1, 3, 5 — configs/experiment.yaml).

A task is a scripted scenario, not a free-form prompt: it specifies the user
goal, the sequence of tool calls a correct agent would make
(expected_tool_sequence — checked step-by-step by executor.run_task, not
just left implicit), and, for tasks with inject_errors=True in
configs/experiment.yaml, which step should have its first attempt forced to
fail so we can score recovery behavior specifically.

Two task sources, kept distinct on purpose:
  - build_synthetic_tasks() — a small, hand-written scenario set built on
    harness.tools.DEMO_REGISTRY. Used by this package's own unit tests and
    for harness development before the real dataset is wired in. NOT the
    real eval set.
  - load_tasks() — the real, data-driven loader that will build tasks out of
    the held-out test split (data/splits/test.jsonl) once
    adbench.data.prepare has run. This is what evaluation/run_eval.py uses.
    Chain-length-3 and -5 tasks likely need to be *composed* from several
    single-step glaive examples (since that dataset is single-turn),
    stitched into one scenario — that composition logic belongs here, once
    the subset exists to compose from.
"""

from dataclasses import dataclass, field

from adbench.harness.tools import ToolRegistry, build_demo_registry


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    chain_length: int
    user_goal: str
    expected_tool_sequence: list[str] = field(default_factory=list)
    injected_error_at_step: int | None = None


def _make_synthetic_scenarios() -> dict[int, list[TaskSpec]]:
    return {
        1: [
            TaskSpec(
                task_id="s1-weather",
                chain_length=1,
                user_goal="What's the weather in Paris?",
                expected_tool_sequence=["get_weather"],
            ),
            TaskSpec(
                task_id="s1-calc",
                chain_length=1,
                user_goal="What is 12 * (3 + 4)?",
                expected_tool_sequence=["calculator"],
            ),
        ],
        3: [
            TaskSpec(
                task_id="s3-weather-calc-time",
                chain_length=3,
                user_goal=(
                    "Check the weather in Tokyo, then compute 71 * 2, "
                    "then tell me the current time in JST."
                ),
                expected_tool_sequence=["get_weather", "calculator", "get_current_time"],
            ),
            TaskSpec(
                task_id="s3-with-injected-error",
                chain_length=3,
                user_goal=(
                    "Check the weather in Cairo, then compute 95 - 10, "
                    "then look up 'lora' in the knowledge base."
                ),
                expected_tool_sequence=["get_weather", "calculator", "search_knowledge_base"],
                injected_error_at_step=1,
            ),
        ],
        5: [
            TaskSpec(
                task_id="s5-full-chain",
                chain_length=5,
                user_goal=(
                    "Check the weather in San Francisco, compute 62 - 32, "
                    "look up 'distillation', get the current time in UTC, "
                    "then compute 100 / 4."
                ),
                expected_tool_sequence=[
                    "get_weather", "calculator", "search_knowledge_base",
                    "get_current_time", "calculator",
                ],
            ),
            TaskSpec(
                task_id="s5-with-injected-error",
                chain_length=5,
                user_goal=(
                    "Check the weather in Tokyo, compute 71 + 9, "
                    "look up 'unsloth', get the current time in EST, "
                    "then compute 80 / 2."
                ),
                expected_tool_sequence=[
                    "get_weather", "calculator", "search_knowledge_base",
                    "get_current_time", "calculator",
                ],
                injected_error_at_step=3,
            ),
        ],
    }


def build_synthetic_tasks(
    chain_length: int, registry: ToolRegistry | None = None
) -> list[TaskSpec]:
    """Hand-written scenarios for harness dev/testing (see module docstring —
    not the real eval set). Validates every expected tool name actually
    exists in `registry` (defaults to the demo registry these scenarios were
    written against), so a typo here fails fast instead of surfacing as a
    confusing UnknownToolError deep in a test run.
    """
    registry = registry or build_demo_registry()
    scenarios = _make_synthetic_scenarios()
    if chain_length not in scenarios:
        raise ValueError(
            f"No synthetic scenarios defined for chain_length={chain_length}; "
            f"supported: {sorted(scenarios)}"
        )

    tasks = scenarios[chain_length]
    for task in tasks:
        for name in task.expected_tool_sequence:
            if not registry.has(name):
                raise ValueError(
                    f"Task {task.task_id!r} expects unknown tool {name!r}."
                )
    return tasks


def load_tasks(chain_length: int) -> list[TaskSpec]:
    """TODO: build TaskSpecs for the given chain length from the held-out
    test split (data/splits/test.jsonl) + the data-derived tool registry,
    once adbench.data.prepare has been run. This is the real eval task
    loader used by evaluation/run_eval.py — build_synthetic_tasks() above is
    for harness development/testing only."""
    raise NotImplementedError(
        "Real data-driven task loading requires data/splits/test.jsonl "
        "(run `python -m adbench.data.prepare` first). Use "
        "build_synthetic_tasks() for harness development/testing."
    )
