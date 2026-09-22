"""Evaluate finished checkpoints on an external benchmark subset (BFCL "simple" — see
evaluation/bfcl.py's module docstring for why), one (seed, condition) job at a time.

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.bfcl_worker --jobs 0:base,0:distilled --work-dir /kaggle/working/b0

Same shape as evaluation/ext_worker.py (meant to be started twice, once per GPU; a job whose
result file already exists on Hugging Face is skipped, so a stopped session resumes; job order
should be condition-major across the two `--jobs` shares so neither worker gets stuck with every
slow condition — see notebooks/11_ext_eval.ipynb's job-list comment). Downloads only the one
checkpoint each job needs, evaluates it on a deterministic `--n`-example BFCL subset, and
uploads a single result file:

    runs/v2-seed<k>/results/stages/bfcl_eval_<category>_<condition>.json

BFCL "simple" prompts are short single turns (no multi-step history to build up), so unlike
ext_worker.py this does not need a larger --max-seq-length than configs/models.yaml's default.

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

from adbench.evaluation.worker_utils import parse_jobs, run_jobs, split_jobs  # noqa: F401 — re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"


def result_path(seed: int, condition: str, category: str = "simple") -> str:
    return f"runs/v2-seed{seed}/results/stages/bfcl_eval_{category}_{condition}.json"


def describe_bfcl_result(rows: list[dict[str, Any]]) -> str:
    rate = sum(r["success"] for r in rows) / max(len(rows), 1)
    return f"{len(rows)} examples, success {rate:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<condition> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--n", type=int, default=100, help="deterministic BFCL example subset size")
    parser.add_argument("--category", default="simple")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=8)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to read checkpoints and upload results.")
    os.environ.setdefault("ADBENCH_EVAL_BATCH", str(args.eval_batch))

    from huggingface_hub import HfApi, snapshot_download

    from adbench.evaluation.batching import BatchedModelFn
    from adbench.evaluation.bfcl import download_bfcl_examples, run_bfcl_eval
    from adbench.evaluation.run_eval import build_model_fn, load_condition_model
    from adbench.training.train import load_experiment_config

    api = HfApi(token=token)
    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n} BFCL {args.category!r} examples ...")
    examples = download_bfcl_examples(n=args.n, category=args.category, cache_dir=str(work_dir / "bfcl_cache"))
    print(f"{len(examples)} examples loaded.")

    def already_done(seed: int, condition: str) -> bool:
        return api.file_exists(
            repo_id=args.repo, filename=result_path(seed, condition, args.category), repo_type="model"
        )

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
            return run_bfcl_eval(condition, model_fn, examples, category=args.category)
        finally:
            if isinstance(model_fn, BatchedModelFn):
                model_fn.close()
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(work_dir / "dl", ignore_errors=True)

    def save(seed: int, condition: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"{args.category}_seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=result_path(seed, condition, args.category),
                    repo_id=args.repo, repo_type="model",
                    commit_message=f"bfcl {args.category} eval seed {seed} {condition}",
                )
                return
            except Exception as e:  # noqa: BLE001 — two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {condition}; the result is kept in {local}")

    summary = run_jobs(
        parse_jobs(args.jobs), evaluate, already_done, save,
        describe=describe_bfcl_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
