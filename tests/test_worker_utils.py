"""evaluation/worker_utils.py — shared job parsing/splitting/running, used by ext_worker.py and
bfcl_worker.py. No GPU, no network."""

import pytest

from adbench.evaluation.worker_utils import parse_jobs, run_jobs, split_jobs


def test_parse_jobs_and_bad_input():
    assert parse_jobs("0:base, 3:distilled ,") == [(0, "base"), (3, "distilled")]
    with pytest.raises(ValueError):
        parse_jobs("3")


def test_split_jobs_interleaves_and_covers_everything_once():
    jobs = [(s, c) for s in range(3) for c in ("a", "b")]
    parts = [split_jobs(jobs, 2, i) for i in range(2)]
    assert sorted(parts[0] + parts[1]) == sorted(jobs)
    assert not set(parts[0]) & set(parts[1])


def _row(success):
    return {"success": success}


def test_run_jobs_skips_finished_saves_results_tags_the_seed_and_uses_describe():
    saved, logged = {}, []
    result = run_jobs(
        [(0, "a"), (1, "a"), (2, "b")],
        evaluate=lambda seed, cond: [_row(True), _row(False)],
        already_done=lambda seed, cond: (seed, cond) == (1, "a"),
        save=lambda seed, cond, rows: saved.update({(seed, cond): rows}),
        describe=lambda rows: f"custom:{len(rows)}",
        log=logged.append,
    )
    assert result == {"done": [(0, "a"), (2, "b")], "skipped": [(1, "a")], "deferred": []}
    assert set(saved) == {(0, "a"), (2, "b")}
    assert all(r["seed"] == s for (s, _), rows in saved.items() for r in rows)
    assert any("custom:2" in line for line in logged)


def test_run_jobs_default_describe_and_time_budget():
    now = [0.0]

    def evaluate(seed, cond):
        now[0] += 600.0
        return [_row(True)]

    result = run_jobs(
        [(0, "a"), (1, "a"), (2, "a")], evaluate, lambda s, c: False, lambda s, c, r: None,
        max_minutes=15, clock=lambda: now[0], log=lambda _: None,
    )
    assert result["done"] == [(0, "a"), (1, "a")]
    assert result["deferred"] == [(2, "a")]


def test_run_jobs_propagates_an_evaluation_failure():
    def boom(seed, cond):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run_jobs([(0, "a")], boom, lambda s, c: False, lambda s, c, r: None, log=lambda _: None)
