import json

import pytest
import yaml

from adbench.harness.tasks import TaskSpec, build_synthetic_tasks, load_tasks
from adbench.harness.tools import (
    ToolRegistry,
    build_demo_registry,
    build_glaive_registry,
)


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_build_synthetic_tasks_shapes_match_chain_length(chain_length):
    tasks = build_synthetic_tasks(chain_length)
    assert len(tasks) > 0
    for task in tasks:
        assert isinstance(task, TaskSpec)
        assert task.chain_length == chain_length
        assert len(task.expected_tool_sequence) == chain_length
        assert task.user_goal


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_build_synthetic_tasks_count_within_requested_range(chain_length):
    """30-50 tasks per chain length, per spec."""
    tasks = build_synthetic_tasks(chain_length)
    assert 30 <= len(tasks) <= 50


@pytest.mark.parametrize("chain_length", [1, 3, 5])
def test_build_synthetic_tasks_ids_are_unique(chain_length):
    tasks = build_synthetic_tasks(chain_length)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("chain_length", [3, 5])
def test_build_synthetic_tasks_have_real_sequence_variety(chain_length):
    """Multiple distinct tool-sequence templates, not one pattern repeated
    with only argument values changed."""
    tasks = build_synthetic_tasks(chain_length)
    distinct_sequences = {tuple(t.expected_tool_sequence) for t in tasks}
    assert len(distinct_sequences) >= 5


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
    harness.inject_errors) actually exist for every chain length."""
    for chain_length in (1, 3, 5):
        tasks = build_synthetic_tasks(chain_length)
        assert any(t.injected_error_at_step is not None for t in tasks)


def test_injected_error_step_is_always_a_valid_index():
    for chain_length in (1, 3, 5):
        for task in build_synthetic_tasks(chain_length):
            if task.injected_error_at_step is not None:
                assert 0 <= task.injected_error_at_step < task.chain_length


# --- load_tasks: source="synthetic" (default) ---

def test_load_tasks_default_source_is_synthetic():
    assert load_tasks(3) == build_synthetic_tasks(3)


def test_load_tasks_synthetic_explicit():
    assert load_tasks(1, source="synthetic") == build_synthetic_tasks(1)


def test_load_tasks_unknown_source_raises():
    with pytest.raises(ValueError):
        load_tasks(1, source="nonsense")


# --- load_tasks: source="glaive_train" / "glaive_test" ---
#
# Tested against fixture data (a temp config + temp JSONL), not the real
# data/splits/*.jsonl — so this doesn't depend on `python -m
# adbench.data.prepare` having been run on the machine running the tests.

def _write_fixture_config(tmp_path, train_path, test_path):
    config = {
        "source": {"hf_dataset": "glaiveai/glaive-function-calling-v2", "license": "Apache-2.0"},
        "subset": {
            "n_examples_min": 1, "n_examples_max": 100,
            "min_distinct_tools": 1, "max_distinct_tools": 10,
            "selected_tools": [{"name": "calculate_bmi", "canonical_args": ["height", "weight"]}],
            "per_tool_target": 10,
        },
        "split": {"train_fraction": 0.8, "test_fraction": 0.2, "seed": 1, "test_split_frozen": True},
        "output": {
            "train_path": str(train_path), "test_path": str(test_path),
            "stats_path": str(tmp_path / "stats.json"),
        },
    }
    config_path = tmp_path / "data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def _write_fixture_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


_FIXTURE_RECORD = {
    "task_id": "glaive-123",
    "chain_length": 1,
    "user_goal": "What's my BMI at 70kg and 1.75m?",
    "expected_tool_sequence": ["calculate_bmi"],
    "injected_error_at_step": None,
    "arguments": {"height": 1.75, "weight": 70},
    "arguments_style": "string_single_quoted",
    "source_index": 123,
}


def test_load_tasks_glaive_train_reads_real_jsonl_fields(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_fixture_jsonl(train_path, [_FIXTURE_RECORD])
    _write_fixture_jsonl(test_path, [])
    config_path = _write_fixture_config(tmp_path, train_path, test_path)

    tasks = load_tasks(1, source="glaive_train", config_path=config_path)

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, TaskSpec)
    assert task.task_id == "glaive-123"
    assert task.chain_length == 1
    assert task.user_goal == "What's my BMI at 70kg and 1.75m?"
    assert task.expected_tool_sequence == ["calculate_bmi"]
    assert task.injected_error_at_step is None


def test_load_tasks_glaive_test_reads_the_test_file_not_train(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_fixture_jsonl(train_path, [])
    _write_fixture_jsonl(test_path, [{**_FIXTURE_RECORD, "task_id": "glaive-999"}])
    config_path = _write_fixture_config(tmp_path, train_path, test_path)

    tasks = load_tasks(1, source="glaive_test", config_path=config_path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "glaive-999"


def test_load_tasks_glaive_tools_exist_in_glaive_registry(tmp_path):
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_fixture_jsonl(train_path, [_FIXTURE_RECORD])
    _write_fixture_jsonl(test_path, [])
    config_path = _write_fixture_config(tmp_path, train_path, test_path)

    tasks = load_tasks(1, source="glaive_train", config_path=config_path)

    registry = build_glaive_registry()
    for task in tasks:
        for tool_name in task.expected_tool_sequence:
            assert registry.has(tool_name)


def test_load_tasks_glaive_unsupported_chain_length_raises(tmp_path):
    config_path = _write_fixture_config(tmp_path, tmp_path / "train.jsonl", tmp_path / "test.jsonl")
    with pytest.raises(ValueError):
        load_tasks(3, source="glaive_train", config_path=config_path)


def test_load_tasks_glaive_missing_file_raises_file_not_found(tmp_path):
    # config points at jsonl files that were never written
    config_path = _write_fixture_config(
        tmp_path, tmp_path / "does_not_exist_train.jsonl", tmp_path / "does_not_exist_test.jsonl"
    )
    with pytest.raises(FileNotFoundError):
        load_tasks(1, source="glaive_train", config_path=config_path)
