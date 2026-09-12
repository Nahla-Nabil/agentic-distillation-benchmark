"""Unit tests for evaluation/metrics.py — pure functions over the row-dict
schema (see that module's docstring), no GPU/network needed. Most tests use
hand-built row fixtures for precise control; a few run the real harness
(executor.run_task) to prove task_state_to_row() and the metrics compose
correctly with live TaskState objects, not just fabricated dicts.
"""

import json

from adbench.evaluation.metrics import (
    clean_step_success_rate,
    error_breakdown,
    error_category_breakdown,
    full_chain_success_rate,
    per_step_success_rate,
    read_results_csv,
    read_results_jsonl,
    recovery_rate,
    step_success_rate_by_position,
    success_rate_by_chain_length,
    summarize,
    summarize_by_condition_and_chain_length,
    task_state_to_row,
    write_results_csv,
    write_results_jsonl,
)
from adbench.harness.executor import StepOutcome, TaskState, run_task
from adbench.harness.tasks import TaskSpec
from adbench.harness.tools import build_demo_registry


def _step(step_index, tool_name="get_weather", succeeded=True, recovered=False, error_type=None):
    return {
        "step_index": step_index, "tool_name": tool_name, "succeeded": succeeded,
        "recovered_from_error": recovered, "error": None if error_type is None else "boom",
        "error_type": error_type,
    }


def _row(condition="sft_only", chain_length=1, task_id="t-0", success=True, steps=None,
         injected_error_at_step=None, failure_step_index=None, failure_error_type=None):
    return {
        "condition": condition, "chain_length": chain_length, "task_id": task_id,
        "success": success, "num_steps_attempted": len(steps or []),
        "injected_error_at_step": injected_error_at_step,
        "failure_step_index": failure_step_index, "failure_error_type": failure_error_type,
        "steps": steps or [],
    }


# --- task_state_to_row ---

def test_task_state_to_row_success_shape():
    task = TaskSpec(task_id="t1", chain_length=1, user_goal="g", expected_tool_sequence=["get_weather"])
    state = TaskState(
        task_id="t1", chain_length=1,
        steps=[StepOutcome(step_index=0, tool_name="get_weather", succeeded=True)],
        final_success=True,
    )
    row = task_state_to_row("sft_only", task, state)

    assert row == {
        "condition": "sft_only", "chain_length": 1, "task_id": "t1", "success": True,
        "num_steps_attempted": 1, "injected_error_at_step": None,
        "failure_step_index": None, "failure_error_type": None,
        "steps": [{"step_index": 0, "tool_name": "get_weather", "succeeded": True,
                   "recovered_from_error": False, "error": None, "error_type": None}],
    }


def test_task_state_to_row_failure_shape():
    task = TaskSpec(task_id="t2", chain_length=3, user_goal="g",
                     expected_tool_sequence=["get_weather", "calculator", "get_current_time"],
                     injected_error_at_step=1)
    state = TaskState(
        task_id="t2", chain_length=3,
        steps=[
            StepOutcome(step_index=0, tool_name="get_weather", succeeded=True),
            StepOutcome(step_index=1, tool_name="calculator", succeeded=False,
                        error="bad tool", error_type="WrongToolError"),
        ],
        final_success=False,
    )
    row = task_state_to_row("distilled", task, state)

    assert row["success"] is False
    assert row["num_steps_attempted"] == 2
    assert row["injected_error_at_step"] == 1
    assert row["failure_step_index"] == 1
    assert row["failure_error_type"] == "WrongToolError"


def test_task_state_to_row_integrates_with_real_run_task():
    """Not a fabricated row — an actual harness run, converted, and scored."""
    registry = build_demo_registry()
    task = TaskSpec(task_id="real-1", chain_length=1, user_goal="weather?",
                     expected_tool_sequence=["get_weather"])

    def model_fn(messages):
        return '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'

    state = run_task(model_fn, task, registry)
    row = task_state_to_row("base", task, state)

    assert row["success"] is True
    assert full_chain_success_rate([row]) == 1.0
    assert per_step_success_rate([row]) == 1.0


# --- full_chain_success_rate ---

def test_full_chain_success_rate_basic():
    rows = [_row(success=True), _row(success=True), _row(success=False), _row(success=False)]
    assert full_chain_success_rate(rows) == 0.5


def test_full_chain_success_rate_empty_is_none():
    assert full_chain_success_rate([]) is None


# --- per_step_success_rate / clean_step_success_rate ---

def test_per_step_success_rate_gives_full_credit_to_recovered_steps():
    rows = [_row(steps=[_step(0, succeeded=True, recovered=True)])]
    assert per_step_success_rate(rows) == 1.0


def test_clean_step_success_rate_excludes_recovered_steps():
    rows = [_row(steps=[
        _step(0, succeeded=True, recovered=False),
        _step(1, succeeded=True, recovered=True),
    ])]
    assert per_step_success_rate(rows) == 1.0     # both count fully
    assert clean_step_success_rate(rows) == 0.5   # only the non-recovered one


def test_per_step_success_rate_mixed():
    rows = [_row(steps=[
        _step(0, succeeded=True),
        _step(1, succeeded=False, error_type="WrongToolError"),
    ])]
    assert per_step_success_rate(rows) == 0.5


def test_step_metrics_empty_is_none():
    assert per_step_success_rate([]) is None
    assert clean_step_success_rate([]) is None
    assert per_step_success_rate([_row(steps=[])]) is None


# --- recovery_rate ---

def test_recovery_rate_counts_only_reached_injected_steps():
    rows = [
        # injected at step 1, reached and recovered -> counts as a recovery
        _row(task_id="a", injected_error_at_step=1,
             steps=[_step(0), _step(1, succeeded=True, recovered=True)]),
        # injected at step 2, but chain stopped at step 0 (never reached) -> excluded
        _row(task_id="b", injected_error_at_step=2, success=False,
             steps=[_step(0, succeeded=False, error_type="MalformedCallError")]),
        # injected at step 0, reached but NOT recovered (retries exhausted)
        _row(task_id="c", injected_error_at_step=0, success=False,
             steps=[_step(0, succeeded=False, error_type="ToolExecutionError")]),
    ]
    # reached: task a's step 1 (success) and task c's step 0 (failure) = 1/2
    assert recovery_rate(rows) == 0.5


def test_recovery_rate_none_when_nothing_reached():
    rows = [_row(injected_error_at_step=5, steps=[_step(0)])]
    assert recovery_rate(rows) is None


def test_recovery_rate_none_when_no_injected_errors():
    rows = [_row(injected_error_at_step=None, steps=[_step(0)])]
    assert recovery_rate(rows) is None


# --- step_success_rate_by_position ---

def test_step_success_rate_by_position():
    rows = [
        _row(steps=[_step(0, succeeded=True), _step(1, succeeded=True)]),
        _row(steps=[_step(0, succeeded=True), _step(1, succeeded=False, error_type="WrongToolError")]),
        _row(steps=[_step(0, succeeded=False, error_type="MalformedCallError")]),
    ]
    result = step_success_rate_by_position(rows)
    # position 0: 2 successes / 3 attempts
    assert result[0] == 2 / 3
    # position 1: only reached by the first two tasks, 1 success / 2 attempts
    assert result[1] == 0.5


# --- error_breakdown / error_category_breakdown ---

def test_error_breakdown_counts_by_exact_type():
    rows = [
        _row(success=False, failure_error_type="WrongToolError"),
        _row(success=False, failure_error_type="WrongToolError"),
        _row(success=False, failure_error_type="ToolArgumentError"),
        _row(success=True, failure_error_type=None),  # successful task, excluded
    ]
    assert error_breakdown(rows) == {"WrongToolError": 2, "ToolArgumentError": 1}


def test_error_category_breakdown_groups_protocol_vs_execution():
    rows = [
        _row(success=False, failure_error_type="WrongToolError"),      # protocol
        _row(success=False, failure_error_type="MalformedCallError"),  # protocol
        _row(success=False, failure_error_type="ToolArgumentError"),   # tool_execution
        _row(success=False, failure_error_type="ToolTimeoutError"),    # tool_execution
    ]
    assert error_category_breakdown(rows) == {"protocol": 2, "tool_execution": 2}


def test_error_category_breakdown_unknown_type_falls_back():
    rows = [_row(success=False, failure_error_type="SomeFutureErrorClass")]
    assert error_category_breakdown(rows) == {"unknown": 1}


def test_error_breakdown_empty_when_all_succeed():
    rows = [_row(success=True), _row(success=True)]
    assert error_breakdown(rows) == {}
    assert error_category_breakdown(rows) == {}


# --- success_rate_by_chain_length ---

def test_success_rate_by_chain_length_groups_correctly():
    rows = [
        _row(chain_length=1, success=True), _row(chain_length=1, success=False),
        _row(chain_length=3, success=True), _row(chain_length=3, success=True),
    ]
    result = success_rate_by_chain_length(rows)
    assert result == {1: 0.5, 3: 1.0}


# --- summarize / summarize_by_condition_and_chain_length ---

def test_summarize_bundles_every_metric():
    rows = [_row(steps=[_step(0)])]
    summary = summarize(rows)
    assert set(summary) == {
        "n_tasks", "full_chain_success_rate", "per_step_success_rate",
        "clean_step_success_rate", "recovery_rate", "error_breakdown",
        "error_category_breakdown",
    }
    assert summary["n_tasks"] == 1


def test_summarize_by_condition_and_chain_length_groups_and_sorts():
    rows = [
        _row(condition="distilled", chain_length=3, success=True, steps=[_step(0)]),
        _row(condition="base", chain_length=1, success=False, steps=[_step(0, succeeded=False, error_type="UnknownToolError")]),
        _row(condition="base", chain_length=1, success=True, steps=[_step(0)]),
    ]
    result = summarize_by_condition_and_chain_length(rows)

    assert [(r["condition"], r["chain_length"]) for r in result] == [
        ("base", 1), ("distilled", 3),
    ]
    base_1 = result[0]
    assert base_1["n_tasks"] == 2
    assert base_1["full_chain_success_rate"] == 0.5


# --- I/O round trips ---

def test_jsonl_round_trip_preserves_nested_steps(tmp_path):
    rows = [_row(steps=[_step(0, succeeded=True), _step(1, succeeded=False, error_type="WrongToolError")])]
    path = tmp_path / "results.jsonl"
    write_results_jsonl(rows, path)
    assert read_results_jsonl(path) == rows


def test_csv_round_trip_preserves_summary_and_nested_steps(tmp_path):
    rows = [
        _row(task_id="t1", success=True, steps=[_step(0, succeeded=True)]),
        _row(task_id="t2", success=False, injected_error_at_step=0, failure_step_index=0,
             failure_error_type="ToolTimeoutError",
             steps=[_step(0, succeeded=False, error_type="ToolTimeoutError")]),
    ]
    path = tmp_path / "results.csv"
    write_results_csv(rows, path)
    loaded = read_results_csv(path)

    assert loaded == rows


def test_csv_output_has_readable_summary_columns(tmp_path):
    rows = [_row(task_id="t1", success=True, steps=[_step(0)])]
    path = tmp_path / "results.csv"
    write_results_csv(rows, path)

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert "condition" in header
    assert "task_id" in header
    assert "steps_json" in header
    # and steps_json is genuinely parseable JSON, not a raw Python repr
    with open(path, encoding="utf-8") as f:
        import csv as _csv
        row = next(_csv.DictReader(f))
    json.loads(row["steps_json"])
