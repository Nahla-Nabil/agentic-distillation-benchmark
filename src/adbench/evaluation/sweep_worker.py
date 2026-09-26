"""Hyperparameter-sweep worker: train + evaluate one (seed, pair, sweep, condition) job at a time.

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.sweep_worker --jobs 0:main:lower_lr:sft_only --work-dir /kaggle/working/sw0

Two questions it exists to answer (see notebooks/16_sweeps.ipynb):
  1. Is sft_only's seed-to-seed instability just an untuned baseline? Re-train it with a gentler
     recipe (configs/experiment.yaml's `lower_lr`, `lower_lr_1ep` sweeps) and compare.
  2. Is KD's benefit a dial of "anchoring strength"? Vary the KD weight (`sft_vheavy` 0.05,
     `sft_heavy` 0.2, baseline 0.5, `kd_heavy` 0.8) with a teacher-free anchor (self_distill /
     self_distill_small) on the strong (main) and weak (small) student.

A `sweep` may carry a data-scale suffix, `<sweep>@n<k>` (e.g. `baseline@n20`): train on a seeded, nested
subset of the default train split with at most k examples per tool (data-scale ablation; the default
split has 80 per tool). `baseline` means no hyperparameter overrides.

`pair` is "main" (Qwen3-4B student) or "small" (Qwen3-1.7B student); everything else - the training
data (configs/data.yaml's default split), the 121-task unseen_tools eval set, chains 1/3/5 - is
identical to the main pipeline, so numbers are directly comparable. `sweep` is a name from
configs/experiment.yaml's training.sweep list ("baseline" = no overrides).

Same shape as second_pair_worker.py / ablation_worker.py: job bookkeeping from worker_utils.py,
resumable via a result file already on Hugging Face, checkpoints deleted after eval. Uploads

    runs/v2-seed<k>/results/stages/sweep_<pair>_<sweep>_<condition>.json
    runs/v2-seed<k>/results/training_logs/sweep_<pair>_<sweep>_<condition>.jsonl

Needs HF_TOKEN (write access to ADBENCH_HF_REPO) in the environment.
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from adbench.evaluation.second_pair_worker import summarize_loss_log
from adbench.evaluation.worker_utils import run_jobs, split_jobs  # noqa: F401 - re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"
TASK_SET = "unseen_tools"
STUDENT_KEYS = {"main": "student", "small": "student_small"}
EVAL_MAX_SEQ_LENGTH = 4096  # same reason as second_pair_worker: prose-answering models overflow 2048


def parse_sweep_jobs(spec: str) -> list[tuple[int, str, str, str]]:
    """'0:main:lower_lr:sft_only' -> [(0, 'main', 'lower_lr', 'sft_only')]."""
    jobs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"job {item!r} must look like <seed>:<pair>:<sweep>:<condition>")
        seed, pair, sweep, condition = parts
        if pair not in STUDENT_KEYS:
            raise ValueError(f"job {item!r}: pair must be one of {sorted(STUDENT_KEYS)}, got {pair!r}")
        jobs.append((int(seed), pair, sweep, condition))
    return jobs


def split_sweep_spec(sweep: str) -> tuple[str, int | None]:
    """'lower_lr' -> ('lower_lr', None); 'baseline@n20' -> ('baseline', 20)."""
    name, sep, scale = sweep.partition("@n")
    if not sep:
        return sweep, None
    if not scale.isdigit() or int(scale) < 1:
        raise ValueError(f"sweep {sweep!r}: the data-scale suffix must look like @n<positive integer>")
    return name, int(scale)


def result_path(seed: int, pair: str, sweep: str, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/sweep_{pair}_{sweep}_{condition}.json"


def loss_log_repo_path(seed: int, pair: str, sweep: str, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/training_logs/sweep_{pair}_{sweep}_{condition}.jsonl"


def describe_sweep_result(rows: list[dict[str, Any]]) -> str:
    rate = sum(r["success"] for r in rows) / max(len(rows), 1)
    steps = rows[0].get("train_steps", "?") if rows else "?"
    return f"{steps} train steps, {len(rows)} eval tasks, success {rate:.3f}"


def fold_key(pair: str, sweep: str, condition: str) -> str:
    return f"{pair}:{sweep}:{condition}"


def split_key(key: str) -> tuple[str, str, str]:
    pair, sweep, condition = key.split(":", 2)
    return pair, sweep, condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<pair>:<sweep>:<condition> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to upload results.")

    from huggingface_hub import HfApi

    from adbench.data.prepare import read_jsonl, subsample_per_tool, write_jsonl
    from adbench.evaluation.run_eval import load_condition_model, make_harness_model_fn, run_harness_eval_for_condition
    from adbench.harness.tools import build_demo_registry
    from adbench.training.train import (
        REPO_ROOT, free_gpu_memory, load_experiment_config, loss_log_path, train_condition,
    )

    api = HfApi(token=token)
    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    harness = experiment_config["harness"]
    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_demo_registry()
    glaive_train_path = REPO_ROOT / "data" / "splits" / "train.jsonl"
    os.environ["ADBENCH_TRAINING_LOG_DIR"] = str(work_dir / "training_logs")  # per worker, never shared

    def already_done(seed: int, key: str) -> bool:
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, *split_key(key)), repo_type="model")

    def evaluate(seed: int, key: str) -> list[dict[str, Any]]:
        pair, sweep, condition = split_key(key)
        if not glaive_train_path.exists():
            raise FileNotFoundError(
                f"{glaive_train_path} does not exist - run `python -m adbench.data.prepare` first "
                "(once, before starting workers)."
            )
        student_key = STUDENT_KEYS[pair]
        base_sweep, per_tool = split_sweep_spec(sweep)
        sweep_name = None if base_sweep == "baseline" else base_sweep
        train_path = glaive_train_path
        if per_tool is not None:
            train_path = work_dir / "data" / f"train_{per_tool}_per_tool_seed{seed}.jsonl"
            write_jsonl(subsample_per_tool(read_jsonl(glaive_train_path), per_tool, seed), train_path)
        checkpoint_dir = work_dir / "checkpoints" / f"{pair}_{sweep}" / condition
        os.environ["ADBENCH_SEED"] = str(seed)
        train_summary = train_condition(
            condition, experiment_config, models_config,
            sweep_name=sweep_name,
            glaive_train_path=train_path, checkpoint_dir_override=checkpoint_dir,
            student_key=student_key,
        )
        log_path = loss_log_path(condition, sweep_name)
        loss_summary = None
        if log_path.exists():
            loss_summary = summarize_loss_log(read_jsonl(log_path))
            try:
                api.upload_file(
                    path_or_fileobj=str(log_path), path_in_repo=loss_log_repo_path(seed, pair, sweep, condition),
                    repo_id=args.repo, repo_type="model",
                    commit_message=f"sweep loss log {pair} {sweep} {condition} seed {seed}",
                )
            except Exception as e:  # noqa: BLE001 - a diagnostic; never lose the eval result over it
                print(f"  WARNING: could not upload loss log: {type(e).__name__}: {e}")
        model, tokenizer = load_condition_model(
            condition, experiment_config, models_config, checkpoint_dir=checkpoint_dir,
            max_seq_length=EVAL_MAX_SEQ_LENGTH, student_key=student_key,
        )
        try:
            model_fn = make_harness_model_fn(model, tokenizer)
            eval_rows = run_harness_eval_for_condition(
                condition, model_fn, harness["chain_lengths"], registry, harness["max_retries_per_step"],
                task_set=TASK_SET,
            )
        finally:
            del model
            gc.collect()
            free_gpu_memory()
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        for row in eval_rows:
            row["pair"], row["sweep"] = pair, sweep
            row["train_steps"] = train_summary.get("steps")
            row["train_n_examples"] = train_summary.get("n_examples")
            row["train_loss_summary"] = loss_summary
        return eval_rows

    def save(seed: int, key: str, rows: list[dict[str, Any]]) -> None:
        pair, sweep, condition = split_key(key)
        local = out_dir / f"seed{seed}_{pair}_{sweep}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=result_path(seed, pair, sweep, condition),
                    repo_id=args.repo, repo_type="model",
                    commit_message=f"sweep {pair} {sweep} {condition} seed {seed}",
                )
                return
            except Exception as e:  # noqa: BLE001 - two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {key}; kept in {local}")

    keyed_jobs = [(seed, fold_key(pair, sweep, condition)) for seed, pair, sweep, condition in parse_sweep_jobs(args.jobs)]
    summary = run_jobs(
        keyed_jobs, evaluate, already_done, save,
        describe=describe_sweep_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{key}" for s, key in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
