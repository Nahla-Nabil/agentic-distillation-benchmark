"""End-to-end check tying tools.py -> executor.py -> tasks.py together: every
synthetic task, run with a correctly-scripted model, should complete
successfully — including the tasks with an injected error, which must show
up as a recovered (not failed) step. This is the same run_task() call
evaluation/run_eval.py will make against real models later.
"""

import json

import pytest

from adbench.harness.executor import run_task
from adbench.harness.tasks import build_synthetic_tasks
from adbench.harness.tools import build_demo_registry

# Concrete, valid arguments for each synthetic task's expected tool sequence,
# by task_id -> list of argument dicts (one per step). Not otherwise checked
# by the harness (tool-name correctness is what's scored, not argument
# semantics) — just needs to be accepted by the demo tools without error.
TASK_ARGS: dict[str, list[dict]] = {
    "s1-weather": [{"city": "Paris"}],
    "s1-calc": [{"expression": "12 * (3 + 4)"}],
    "s3-weather-calc-time": [
        {"city": "Tokyo"}, {"expression": "71 * 2"}, {"timezone": "jst"},
    ],
    "s3-with-injected-error": [
        {"city": "Cairo"}, {"expression": "95 - 10"}, {"query": "lora"},
    ],
    "s5-full-chain": [
        {"city": "San Francisco"}, {"expression": "62 - 32"}, {"query": "distillation"},
        {"timezone": "utc"}, {"expression": "100 / 4"},
    ],
    "s5-with-injected-error": [
        {"city": "Tokyo"}, {"expression": "71 + 9"}, {"query": "unsloth"},
        {"timezone": "est"}, {"expression": "80 / 2"},
    ],
}


def _make_model_fn(expected_sequence, arguments_by_position):
    def model_fn(messages):
        success_count = sum(
            1 for m in messages
            if m["role"] == "tool" and not m["content"].startswith("ERROR:")
        )
        name = expected_sequence[success_count]
        args = arguments_by_position[success_count]
        return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'

    return model_fn


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_all_synthetic_tasks_succeed_with_a_correct_model(chain_length):
    registry = build_demo_registry()
    tasks = build_synthetic_tasks(chain_length, registry)

    for task in tasks:
        args = TASK_ARGS[task.task_id]
        model_fn = _make_model_fn(task.expected_tool_sequence, args)

        state = run_task(model_fn, task, registry)

        assert state.final_success is True, f"{task.task_id} did not succeed: {state.steps}"
        assert len(state.steps) == task.chain_length

        if task.injected_error_at_step is not None:
            injected_step = state.steps[task.injected_error_at_step]
            assert injected_step.succeeded is True
            assert injected_step.recovered_from_error is True
