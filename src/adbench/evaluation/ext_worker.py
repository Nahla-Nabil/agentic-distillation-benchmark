"""Evaluate finished checkpoints on the extended synthetic tasks, one (seed, condition) job at a time.

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.ext_worker --jobs 0:base,0:distilled,1:sft_only --work-dir /kaggle/working/w0

Meant to be started twice, once per GPU (batched generation must not span both T4s, see
scripts/check_batched_eval.py), each with its own share of the jobs. A worker never touches the
repo's checkpoints/ or results/ folders, so two workers cannot collide: for each job it downloads
only that one checkpoint from Hugging Face, evaluates it, and uploads a single result file

    runs/v2-seed<k>/results/stages/ext_eval_<condition>.json

A job whose result file already exists on Hugging Face is skipped, so re-running a stopped session
resumes where it stopped. `--max-minutes` stops the worker from starting new jobs after that long,
which protects the weekly GPU quota; finished jobs stay saved.

Needs HF_TOKEN (write access to ADBENCH_HF_REPO) in the environment.
"""

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_REPO = "NahlaNabil/adbench-run"


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


def result_path(seed: int, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/ext_eval_{condition}.json"


def run_jobs(
    jobs: list[tuple[int, str]],
    evaluate: Callable[[int, str], list[dict[str, Any]]],
    already_done: Callable[[int, str], bool],
    save: Callable[[int, str, list[dict[str, Any]]], None],
    max_minutes: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = print,
) -> dict[str, list[tuple[int, str]]]:
    """Run `jobs` in order. Returns which were done, skipped (already finished) and deferred
    (not started because the time budget ran out)."""
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
        ext = [r for r in rows if r["task_set"] == "unseen_tools_ext"]
        rate = sum(r["success"] for r in ext) / max(len(ext), 1)
        log(f"[{i + 1}/{len(jobs)}] seed {seed} {condition}: {len(ext)} tasks, success {rate:.3f}, "
            f"{(clock() - t0) / 60:.1f} min")
        result["done"].append((seed, condition))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<condition> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to read checkpoints and upload results.")
    os.environ.setdefault("ADBENCH_EVAL_BATCH", "16")
    os.environ["ADBENCH_STORE_TRANSCRIPTS"] = "1"

    from huggingface_hub import HfApi, snapshot_download

    from adbench.evaluation.batching import BatchedModelFn
    from adbench.evaluation.run_eval import (
        build_model_fn,
        load_condition_model,
        run_ext_eval_for_condition,
    )
    from adbench.training.train import load_experiment_config

    api = HfApi(token=token)
    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    harness = experiment_config["harness"]
    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def already_done(seed: int, condition: str) -> bool:
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, condition), repo_type="model")

    def evaluate(seed: int, condition: str) -> list[dict[str, Any]]:
        tag = f"v2-seed{seed}"
        snapshot_download(
            repo_id=args.repo, repo_type="model", token=token, local_dir=str(work_dir / "dl"),
            allow_patterns=[f"runs/{tag}/checkpoints/{condition}/**"],
        )
        checkpoint = work_dir / "dl" / "runs" / tag / "checkpoints" / condition
        model, tokenizer = load_condition_model(
            condition, experiment_config, models_config, checkpoint_dir=checkpoint
        )
        model_fn = build_model_fn(model, tokenizer, experiment_config)
        try:
            return run_ext_eval_for_condition(
                condition, model_fn, harness["chain_lengths"], harness["max_retries_per_step"]
            )
        finally:
            if isinstance(model_fn, BatchedModelFn):
                model_fn.close()
            del model
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(work_dir / "dl", ignore_errors=True)

    def save(seed: int, condition: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")   # kept as a Kaggle output too
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=result_path(seed, condition),
                    repo_id=args.repo, repo_type="model", commit_message=f"ext eval seed {seed} {condition}",
                )
                return
            except Exception as e:  # noqa: BLE001 — two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {condition}; the result is kept in {local}")

    summary = run_jobs(parse_jobs(args.jobs), evaluate, already_done, save, max_minutes=args.max_minutes)
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
