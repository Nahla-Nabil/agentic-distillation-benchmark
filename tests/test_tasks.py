import pytest

from adbench.harness.tasks import TaskSpec, build_synthetic_tasks, load_tasks
from adbench.harness.tools import ToolRegistry, build_demo_registry


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_build_synthetic_tasks_shapes_match_chain_length(chain_length):
    tasks = build_synthetic_tasks(chain_length)
    assert len(tasks) > 0
    for task in tasks:
        assert isinstance(task, TaskSpec)
        assert task.chain_length == chain_length
        assert len(task.expected_tool_sequence) == chain_length
        assert task.user_goal


def test_build_synthetic_tasks_unsupported_chain_length_raises():
    with pytest.raises(ValueError):
        build_synthetic_tasks(2)


def test_build_synthetic_tasks_validates_tools_exist_in_registry():
    empty_registry = ToolRegistry()
    with pytest.raises(ValueError):
        build_synthetic_tasks(1, registry=empty_registry)


def test_build_synthetic_tasks_uses_demo_registry_by_default():
    # should not raise, since default registry has every tool these
    # scenarios reference
    for chain_length in (1, 3, 5):
        build_synthetic_tasks(chain_length, registry=build_demo_registry())


def test_at_least_one_task_per_chain_length_has_an_injected_error():
    """Sanity check that error-injection scenarios (configs/experiment.yaml:
    harness.inject_errors) actually exist for the multi-step chains."""
    for chain_length in (3, 5):
        tasks = build_synthetic_tasks(chain_length)
        assert any(t.injected_error_at_step is not None for t in tasks)


def test_load_tasks_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        load_tasks(1)
