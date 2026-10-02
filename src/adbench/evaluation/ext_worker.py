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

The extended set's 12-tool system prompt plus a 5-step conversation history can exceed the 2048
tokens the checkpoints were trained/originally evaluated with (a "distilled" condition, whose
answers run longer, hit this first: shape [8, 2113] > 2048, then a mask-size crash). Loading the
model here uses `--max-seq-length` (default 4096), not the value in configs/models.yaml, so this
only widens the context window at load time — it never touches the trained adapter weights.

`--task-set primary` (added 2026-10-02) instead re-runs the ORIGINAL 121-task primary set at the
same 4096 context, uploading `primary4k_eval_<condition>.json` with task_set
"unseen_tools_ctx4096": the main pair's primary set was first evaluated at 2048, while the sweeps
and every other pair used 4096, so the main-pair references they are compared against need the
same context. The default (`--task-set ext`) is unchanged.

Needs HF_TOKEN (write access to ADBENCH_HF_REPO) in the environment.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from adbench.evaluation.worker_utils import parse_jobs, run_jobs, split_jobs  # noqa: F401 — re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"


# --task-set value -> (result-file prefix, task_set string carried by the rows)
TASK_SETS = {
    "ext": ("ext_eval", "unseen_tools_ext"),
    "primary": ("primary4k_eval", "unseen_tools_ctx4096"),
}


def result_path(seed: int, condition: str, task_set: str = "ext") -> str:
    return f"runs/v2-seed{seed}/results/stages/{TASK_SETS[task_set][0]}_{condition}.json"


def describe_ext_result(rows: list[dict[str, Any]]) -> str:
    sets = {name for _, name in TASK_SETS.values()}
    ext = [r for r in rows if r["task_set"] in sets]
    rate = sum(r["success"] for r in ext) / max(len(ext), 1)
    return f"{len(ext)} tasks, success {rate:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<condition> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument(
        "--max-seq-length", type=int, default=4096,
        help="Context window to load the model with (see load_condition_model's docstring): "
             "the extended set's 12-tool prompt + a 5-step history can exceed the 2048 the "
             "checkpoints were trained/originally evaluated with.",
    )
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="ext")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to read checkpoints and upload results.")
    os.environ.setdefault("ADBENCH_EVAL_BATCH", str(args.eval_batch))
    os.environ["ADBENCH_STORE_TRANSCRIPTS"] = "1"

    from huggingface_hub import HfApi, snapshot_download

    from adbench.evaluation.batching import BatchedModelFn
    from adbench.evaluation.run_eval import (
        build_model_fn,
        load_condition_model,
        run_ext_eval_for_condition,
        run_primary_ctx4096_eval_for_condition,
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
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, condition, args.task_set), repo_type="model")

    def evaluate(seed: int, condition: str) -> list[dict[str, Any]]:
        tag = f"v2-seed{seed}"
        snapshot_download(
            repo_id=args.repo, repo_type="model", token=token, local_dir=str(work_dir / "dl"),
            allow_patterns=[f"runs/{tag}/checkpoints/{condition}/**"],
        )
        checkpoint = work_dir / "dl" / "runs" / tag / "checkpoints" / condition
        model, tokenizer = load_condition_model(
            condition, experiment_config, models_config, checkpoint_dir=checkpoint,
            max_seq_length=args.max_seq_length,
        )
        model_fn = build_model_fn(model, tokenizer, experiment_config)
        try:
            run = run_primary_ctx4096_eval_for_condition if args.task_set == "primary" else run_ext_eval_for_condition
            return run(condition, model_fn, harness["chain_lengths"], harness["max_retries_per_step"])
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
        local = out_dir / f"{args.task_set}_seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")   # kept as a Kaggle output too
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=result_path(seed, condition, args.task_set),
                    repo_id=args.repo, repo_type="model",
                    commit_message=f"{args.task_set} eval seed {seed} {condition}",
                )
                return
            except Exception as e:  # noqa: BLE001 — two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {condition}; the result is kept in {local}")

    summary = run_jobs(
        parse_jobs(args.jobs), evaluate, already_done, save,
        describe=describe_ext_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
