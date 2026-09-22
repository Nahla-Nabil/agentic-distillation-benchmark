"""Shared machinery for resumable, two-GPU-worker Kaggle evaluation scripts: parsing a
comma-separated <seed>:<condition> job list, splitting it between workers, and running jobs
under a per-worker time budget with upload-based resumability. Used by evaluation/ext_worker.py
(the extended synthetic set) and evaluation/bfcl_worker.py (the external BFCL benchmark) —
factored out once a second worker needed the exact same job bookkeeping, so it's tested once
here instead of twice per caller.
"""

import time
from collections.abc import Callable
from typing import Any


def parse_jobs(spec: str) -> list[tuple[int, str]]:
    """'0:base,1:distilled' -> [(0, 'base'), (1, 'distilled')]."""
    jobs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        seed, _, condition = item.partition(":")
        if not condition:
            raise ValueError(f"job {item!r} must look like <seed>:<condition>")
        jobs.append((int(seed), condition))
    return jobs


def split_jobs(jobs: list[tuple[int, str]], n_workers: int, index: int) -> list[tuple[int, str]]:
    """Every n_workers-th job starting at `index`, so each worker gets a similar mix."""
    return jobs[index::n_workers]


def default_describe(rows: list[dict[str, Any]]) -> str:
    return f"{len(rows)} rows"


def run_jobs(
    jobs: list[tuple[int, str]],
    evaluate: Callable[[int, str], list[dict[str, Any]]],
    already_done: Callable[[int, str], bool],
    save: Callable[[int, str, list[dict[str, Any]]], None],
    describe: Callable[[list[dict[str, Any]]], str] = default_describe,
    max_minutes: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> dict[str, list[tuple[int, str]]]:
    """Run `jobs` in order. Returns which were done, skipped (already finished) and deferred
    (not started because the time budget ran out). `describe(rows)` renders each finished job's
    result into the tail of its log line (e.g. "158 tasks, success 0.930")."""
    started = clock()
    result: dict[str, list[tuple[int, str]]] = {"done": [], "skipped": [], "deferred": []}
    for i, (seed, condition) in enumerate(jobs):
        if already_done(seed, condition):
            log(f"[{i + 1}/{len(jobs)}] seed {seed} {condition}: already on Hugging Face, skipping")
            result["skipped"].append((seed, condition))
            continue
        if max_minutes is not None and (clock() - started) / 60 >= max_minutes:
            log(f"[{i + 1}/{len(jobs)}] seed {seed} {condition}: time budget of {max_minutes} min used up, deferring")
            result["deferred"].append((seed, condition))
            continue
        t0 = clock()
        rows = evaluate(seed, condition)
        for row in rows:
            row["seed"] = seed
        save(seed, condition, rows)
        log(f"[{i + 1}/{len(jobs)}] seed {seed} {condition}: {describe(rows)}, {(clock() - t0) / 60:.1f} min")
        result["done"].append((seed, condition))
    return result
