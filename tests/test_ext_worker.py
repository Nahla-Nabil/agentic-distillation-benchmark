"""Job bookkeeping of evaluation/ext_worker.py with fakes — no GPU, no network."""

import pytest

from adbench.evaluation.ext_worker import parse_jobs, result_path, run_jobs, split_jobs


def test_parse_jobs_and_bad_input():
    assert parse_jobs("0:base, 3:distilled ,") == [(0, "base"), (3, "distilled")]
    with pytest.raises(ValueError):
        parse_jobs("3")


def test_split_jobs_interleaves_and_covers_everything_once():
    jobs = [(s, c) for s in range(3) for c in ("a", "b")]
    parts = [split_jobs(jobs, 2, i) for i in range(2)]
    assert sorted(parts[0] + parts[1]) == sorted(jobs)
    assert not set(parts[0]) & set(parts[1])
    assert parts[0][0] == jobs[0] and parts[1][0] == jobs[1]


def test_result_path_matches_the_tag_layout_used_by_the_pipeline():
    assert result_path(3, "distilled") == "runs/v2-seed3/results/stages/ext_eval_distilled.json"


def _row(success, task_set="unseen_tools_ext"):
    return {"task_set": task_set, "success": success}


def test_run_jobs_skips_finished_saves_results_and_tags_the_seed():
    saved = {}
    result = run_jobs(
        [(0, "a"), (1, "a"), (2, "b")],
        evaluate=lambda seed, cond: [_row(True), _row(False)],
        already_done=lambda seed, cond: (seed, cond) == (1, "a"),
        save=lambda seed, cond, rows: saved.update({(seed, cond): rows}),
        log=lambda _: None,
    )
    assert result == {"done": [(0, "a"), (2, "b")], "skipped": [(1, "a")], "deferred": []}
    assert set(saved) == {(0, "a"), (2, "b")}
    assert all(r["seed"] == s for (s, _), rows in saved.items() for r in rows)


def test_run_jobs_defers_new_jobs_once_the_time_budget_is_used_up():
    now = [0.0]

    def evaluate(seed, cond):
        now[0] += 600.0      # every job takes 10 minutes
        return [_row(True)]

    result = run_jobs(
        [(0, "a"), (1, "a"), (2, "a"), (3, "a")], evaluate, lambda s, c: False, lambda s, c, r: None,
        max_minutes=15, clock=lambda: now[0], log=lambda _: None,
    )
    assert result["done"] == [(0, "a"), (1, "a")]         # the second one started at 10 min < 15
    assert result["deferred"] == [(2, "a"), (3, "a")]


def test_run_jobs_propagates_an_evaluation_failure():
    def boom(seed, cond):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run_jobs([(0, "a")], boom, lambda s, c: False, lambda s, c, r: None, log=lambda _: None)
