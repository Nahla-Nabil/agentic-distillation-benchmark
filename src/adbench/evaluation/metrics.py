"""Scoring for harness runs.

ROW SCHEMA — the unit everything here operates on is a plain dict, one per
task run (produced by task_state_to_row() right after executor.run_task()
returns, and exactly what gets written to results/eval_results.jsonl by
evaluation/run_eval.py):

    {
        "condition": "distilled",              # which of base/sft_only/distilled
        "chain_length": 3,
        "task_id": "c3-weather_convert_compare-0",
        "success": false,                      # TaskState.final_success
        "num_steps_attempted": 2,               # len(TaskState.steps) — the
                                                 # chain stops at the first
                                                 # unrecoverable step, so a
                                                 # failed task's steps list is
                                                 # SHORTER than chain_length
        "injected_error_at_step": 1,            # from the TaskSpec, or null
        "failure_step_index": 1,                # last step's index, or null if success
        "failure_error_type": "WrongToolError", # last step's error_type, or null
        "steps": [
            {"step_index": 0, "tool_name": "get_weather", "succeeded": true,
             "recovered_from_error": false, "error": null, "error_type": null},
            {"step_index": 1, "tool_name": "calculator", "succeeded": false,
             "recovered_from_error": false, "error": "...", "error_type": "WrongToolError"}
        ]
    }

Metrics functions below all take `rows: list[dict]` in this shape (usually
pre-filtered to one (condition, chain_length) group by the caller, or by
summarize_by_condition_and_chain_length() below) — deliberately NOT
`list[TaskState]`, so these functions have no dependency on harness
internals and work identically on freshly-run results or on results
reloaded from a saved JSONL file days later.

DESIGN DECISION — crediting a step that succeeds after a retry (flagged
per the eval spec, since this affects what the paper can claim):

Three options considered for a step that fails on attempt 0 (e.g. an
injected error) but succeeds on attempt 1:
  (a) full credit — counts the same as a step that succeeded immediately.
  (b) partial credit — some fraction < 1, e.g. 0.5.
  (c) don't blend it into one number — report clean-vs-recovered separately.

Chosen: (a) for the PRIMARY metric (per_step_success_rate), because it
matches the eval spec's own wording literally ("of all individual tool
calls attempted, how many were valid" — a call that eventually returns a
valid result IS valid; there's no principled fraction to assign for (b)
without inventing an arbitrary weight that would itself need justifying).
BUT this is combined with (c): clean_step_success_rate (succeeded AND
never needed a retry) and recovery_rate (of steps that were forced to fail
by an injected error, what fraction still succeeded) are reported as
SEPARATE metrics, not blended into per_step_success_rate. This matters for
the research question specifically: distillation could plausibly degrade
"gets it right immediately" while leaving "eventually gets it right after
retries" roughly intact (or vice versa) — collapsing those into one number
would hide exactly the kind of reliability difference this study is about.
per_step_success_rate - clean_step_success_rate is, by construction, the
fraction of attempted steps that needed at least one retry to succeed.

A SEPARATE caveat, not a design choice but a property of the data worth
knowing before reading per-step numbers: because a chain stops at its first
unrecoverable step, later step *positions* are only reached by tasks that
survived that far — pooling all steps together (as per_step_success_rate
does) is survivorship-biased toward whatever tasks didn't fail early. A
per-position breakdown (step index 0 vs 1 vs 2...) would show this
directly; not computed here since none of the three deliverables above
needed it, but see step_success_rate_by_position() if that granularity
becomes useful for the analysis stage.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Error-class-name -> category, derived from harness/errors.py's hierarchy.
# Kept as a static name lookup (not issubclass checks against imported
# classes) because these functions operate on saved *strings*
# (StepOutcome.error_type is `type(e).__name__`, already just a name by the
# time it reaches a row) — reloading a results file days later shouldn't
# require re-importing harness.errors just to categorize a string.
_TOOL_EXECUTION_ERROR_NAMES = {"ToolExecutionError", "ToolArgumentError", "ToolTimeoutError"}
_PROTOCOL_ERROR_NAMES = {"ProtocolError", "MalformedCallError", "UnknownToolError", "WrongToolError"}


# Rows written before the seen/unseen split have no `task_set` field; they are
# all from the synthetic (unseen-tool) set.
DEFAULT_TASK_SET = "unseen_tools"


def task_state_to_row(condition: str, task: Any, state: Any, task_set: str = DEFAULT_TASK_SET) -> dict[str, Any]:
    """Convert one executor.run_task() result into the row schema documented
    above. `task` is the TaskSpec that was run (for injected_error_at_step);
    `state` is the TaskState run_task() returned. `task_set` names which
    evaluation set the task came from ("unseen_tools": the synthetic six-tool
    vocabulary the students never trained on; "seen_tools": held-out Glaive
    tasks over the training tools)."""
    steps = [
        {
            "step_index": s.step_index,
            "tool_name": s.tool_name,
            "succeeded": s.succeeded,
            "recovered_from_error": s.recovered_from_error,
            "error": s.error,
            "error_type": s.error_type,
            "arguments_match": getattr(s, "arguments_match", None),
        }
        for s in state.steps
    ]

    failure_step_index = None
    failure_error_type = None
    if not state.final_success and state.steps:
        last = state.steps[-1]
        failure_step_index = last.step_index
        failure_error_type = last.error_type

    return {
        "condition": condition,
        "task_set": task_set,
        "chain_length": state.chain_length,
        "task_id": state.task_id,
        "success": state.final_success,
        "num_steps_attempted": len(state.steps),
        "injected_error_at_step": getattr(task, "injected_error_at_step", None),
        "failure_step_index": failure_step_index,
        "failure_error_type": failure_error_type,
        "steps": steps,
    }


def _all_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [step for row in rows for step in row["steps"]]


def full_chain_success_rate(rows: list[dict[str, Any]]) -> float | None:
    """Fraction of tasks that completed their entire chain successfully.
    None (not 0.0 or nan) if `rows` is empty — there's no rate to report,
    and 0.0 would misleadingly read as "every task failed"."""
    if not rows:
        return None
    return sum(1 for r in rows if r["success"]) / len(rows)


def per_step_success_rate(rows: list[dict[str, Any]]) -> float | None:
    """Fraction of *attempted* steps (pooled across all given tasks) that
    ultimately succeeded — full credit regardless of retries. See the
    module docstring's "DESIGN DECISION" for why, and why this is meant to
    be read alongside clean_step_success_rate/recovery_rate, not alone."""
    steps = _all_steps(rows)
    if not steps:
        return None
    return sum(1 for s in steps if s["succeeded"]) / len(steps)


def clean_step_success_rate(rows: list[dict[str, Any]]) -> float | None:
    """Fraction of attempted steps that succeeded WITHOUT needing any
    retry. Always <= per_step_success_rate; the gap between the two is the
    fraction of steps that needed a retry to eventually succeed."""
    steps = _all_steps(rows)
    if not steps:
        return None
    return sum(1 for s in steps if s["succeeded"] and not s["recovered_from_error"]) / len(steps)


def recovery_rate(rows: list[dict[str, Any]]) -> float | None:
    """Of tasks with an injected error whose targeted step was actually
    reached (i.e. no earlier, unrelated step failed the chain first), the
    fraction where that specific step ultimately succeeded. Isolates
    error-handling/recovery skill from general task-following skill — a
    task where the injected step was never reached reflects a DIFFERENT
    failure (an earlier step), so it's excluded rather than counted as a
    failed recovery. None if no task's injected step was reached."""
    reached_steps = []
    for row in rows:
        idx = row["injected_error_at_step"]
        if idx is not None and idx < len(row["steps"]):
            reached_steps.append(row["steps"][idx])
    if not reached_steps:
        return None
    return sum(1 for s in reached_steps if s["succeeded"]) / len(reached_steps)


def step_success_rate_by_position(rows: list[dict[str, Any]]) -> dict[int, float]:
    """per_step_success_rate broken out by step *position* (0, 1, 2, ...)
    instead of pooled — makes the survivorship-bias caveat in the module
    docstring visible directly: a late position's rate only reflects tasks
    that survived to reach it."""
    by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for step in _all_steps(rows):
        by_position[step["step_index"]].append(step)
    return {
        pos: sum(1 for s in steps if s["succeeded"]) / len(steps)
        for pos, steps in sorted(by_position.items())
    }


def _error_category(error_type_name: str) -> str:
    if error_type_name in _TOOL_EXECUTION_ERROR_NAMES:
        return "tool_execution"
    if error_type_name in _PROTOCOL_ERROR_NAMES:
        return "protocol"
    return "unknown"  # forward-compatible: a future error class not yet listed above


def error_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Counter of failing-step error_type (exact exception class name)
    among failed tasks — malformed calls vs wrong tool vs argument errors
    vs timeouts, per the eval spec."""
    counter: Counter[str] = Counter()
    for row in rows:
        if not row["success"] and row["failure_error_type"] is not None:
            counter[row["failure_error_type"]] += 1
    return dict(counter)


def error_category_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """error_breakdown(), grouped into harness.errors' two top-level
    categories (protocol vs tool_execution) instead of exact class name."""
    counter: Counter[str] = Counter()
    for error_type, count in error_breakdown(rows).items():
        counter[_error_category(error_type)] += count
    return dict(counter)


def success_rate_by_chain_length(rows: list[dict[str, Any]]) -> dict[int, float]:
    """full_chain_success_rate(), grouped by chain_length — the central
    plot: does the gap between conditions widen as chain length grows?"""
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[row["chain_length"]].append(row)
    return {length: full_chain_success_rate(group) for length, group in sorted(by_length.items())}


def argument_accuracy(rows: list[dict[str, Any]]) -> float | None:
    """Share of steps whose first attempt used exactly the expected tool
    arguments, over the steps that carry ground truth (real Glaive tasks).
    None when no step has ground truth, e.g. for the synthetic tasks."""
    values = [
        s["arguments_match"] for row in rows for s in row["steps"] if s.get("arguments_match") is not None
    ]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every metric above, bundled for one slice of rows (typically already
    filtered to one (condition, chain_length) pair by the caller)."""
    return {
        "n_tasks": len(rows),
        "full_chain_success_rate": full_chain_success_rate(rows),
        "per_step_success_rate": per_step_success_rate(rows),
        "clean_step_success_rate": clean_step_success_rate(rows),
        "recovery_rate": recovery_rate(rows),
        "argument_accuracy": argument_accuracy(rows),
        "error_breakdown": error_breakdown(rows),
        "error_category_breakdown": error_category_breakdown(rows),
    }


def summarize_by_condition_and_chain_length(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group `rows` by (condition, task_set, chain_length) and summarize() each
    group — this is the table evaluation/run_eval.py writes to
    results/eval_summary.json, and what the analysis notebook plots. Rows
    without a `task_set` count as the default (unseen-tool) set."""
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row.get("task_set", DEFAULT_TASK_SET), row["chain_length"])].append(row)

    result = []
    for (condition, task_set, chain_length), group in sorted(groups.items()):
        entry = {"condition": condition, "task_set": task_set, "chain_length": chain_length}
        entry.update(summarize(group))
        result.append(entry)
    return result


# --------------------------------------------------------------------------
# I/O — JSONL is the primary format (preserves the full nested per-step
# detail; consistent with the rest of the repo's *.jsonl conventions). CSV
# is also written for spreadsheet tools that can't do nested JSON: the
# per-task summary columns are proper CSV columns, and the full per-step
# detail is preserved (not dropped) as a JSON string in one `steps_json`
# column — no information is lost, just re-nested.
# --------------------------------------------------------------------------

def write_results_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    from adbench.data.prepare import write_jsonl
    write_jsonl(rows, path)


def read_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    from adbench.data.prepare import read_jsonl
    return read_jsonl(path)


_CSV_SUMMARY_FIELDS = [
    "condition", "task_set", "chain_length", "task_id", "success", "num_steps_attempted",
    "injected_error_at_step", "failure_step_index", "failure_error_type",
]


def write_results_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*_CSV_SUMMARY_FIELDS, "steps_json"])
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field, DEFAULT_TASK_SET) if field == "task_set" else row[field]
                       for field in _CSV_SUMMARY_FIELDS}
            csv_row["steps_json"] = json.dumps(row["steps"])
            writer.writerow(csv_row)


def read_results_csv(path: str | Path) -> list[dict[str, Any]]:
    """Inverse of write_results_csv() — mainly for tests/round-trip checks;
    read_results_jsonl() is the format run_eval.py's own downstream
    consumers (metrics functions above) should prefer."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for csv_row in csv.DictReader(f):
            row = dict(csv_row)
            row["chain_length"] = int(row["chain_length"])
            row["success"] = row["success"] == "True"
            row["num_steps_attempted"] = int(row["num_steps_attempted"])
            row["injected_error_at_step"] = (
                int(row["injected_error_at_step"]) if row["injected_error_at_step"] else None
            )
            row["failure_step_index"] = (
                int(row["failure_step_index"]) if row["failure_step_index"] else None
            )
            row["failure_error_type"] = row["failure_error_type"] or None
            row["steps"] = json.loads(row.pop("steps_json"))
            rows.append(row)
    return rows
