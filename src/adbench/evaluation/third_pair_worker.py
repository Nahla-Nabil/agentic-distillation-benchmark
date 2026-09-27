"""Third model pair (does the finding generalise beyond one model FAMILY, not just one size?):
train + evaluate one (seed, condition) job at a time for student_olmo (OLMo-2-0425-1B-Instruct)
+ teacher_olmo (OLMo-2-1124-7B-Instruct) — a different lab/architecture family from the Qwen3
pairs used elsewhere in this project, ~7x apart (main pair ~3.7x, second pair ~4.8x).

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.third_pair_worker --jobs 0:sft_only,0:distilled_olmo --work-dir /kaggle/working/p0

Conditions: "base", "sft_only", "distilled_olmo" (KD from teacher_olmo), "self_distill_olmo" (KD
from student_olmo's own frozen base — a SEPARATE condition id from the other two pairs', because
self-distillation needs a teacher pointed at THIS pair's student). Uses the exact same training
data as the main pipeline (configs/data.yaml's default 8-tool split) and the exact same 121-task
unseen_tools eval set (chains 1/3/5), so results are directly comparable to the other two pairs'
numbers. Evaluates at max_seq_length=4096 from the start (second_pair_worker.py's first run used
2048 and had to be re-run after a weak student's prose-instead-of-tool-call answers overflowed
it — see that module's history) and uploads the training loss log for every job, also from the
start, for the same reason.

Same shape as second_pair_worker.py / ablation_worker.py: job bookkeeping from worker_utils.py,
resumable via a result file already on Hugging Face, checkpoints deleted after eval. Uploads:

    runs/v2-seed<k>/results/stages/pair3_<condition>.json
    runs/v2-seed<k>/results/training_logs/pair3_<condition>.jsonl

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
from adbench.evaluation.worker_utils import parse_jobs, run_jobs, split_jobs  # noqa: F401 - re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"
TASK_SET = "unseen_tools"
STUDENT_KEY = "student_olmo"
EVAL_MAX_SEQ_LENGTH = 4096


def result_path(seed: int, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/pair3_{condition}.json"


def loss_log_repo_path(seed: int, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/training_logs/pair3_{condition}.jsonl"


def describe_pair3_result(rows: list[dict[str, Any]]) -> str:
    rate = sum(r["success"] for r in rows) / max(len(rows), 1)
    steps = rows[0].get("train_steps", "?") if rows else "?"
    return f"{steps} train steps, {len(rows)} eval tasks, success {rate:.3f}"


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
        raise SystemExit("HF_TOKEN is not set; a worker needs it to upload results.")

    from huggingface_hub import HfApi

    from adbench.data.prepare import read_jsonl
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
    glaive_train_path = REPO_ROOT / "data" / "splits" / "train.jsonl"  # the main pipeline's own split, unmodified

    def already_done(seed: int, condition: str) -> bool:
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, condition), repo_type="model")

    def upload_loss_log(seed: int, condition: str, log_path: Path) -> None:
        try:
            api.upload_file(
                path_or_fileobj=str(log_path), path_in_repo=loss_log_repo_path(seed, condition),
                repo_id=args.repo, repo_type="model", commit_message=f"third-pair loss log seed {seed} {condition}",
            )
        except Exception as e:  # noqa: BLE001 - a diagnostic; never lose the eval result over it
            print(f"  WARNING: could not upload loss log: {type(e).__name__}: {e}")

    def evaluate(seed: int, condition: str) -> list[dict[str, Any]]:
        if not glaive_train_path.exists():
            raise FileNotFoundError(
                f"{glaive_train_path} does not exist - run `python -m adbench.data.prepare` first "
                "(once, before starting workers)."
            )
        checkpoint_dir = work_dir / "checkpoints" / condition
        os.environ["ADBENCH_SEED"] = str(seed)
        train_summary = train_condition(
            condition, experiment_config, models_config,
            glaive_train_path=glaive_train_path, checkpoint_dir_override=checkpoint_dir,
            student_key=STUDENT_KEY,
        )
        model, tokenizer = load_condition_model(
            condition, experiment_config, models_config, checkpoint_dir=checkpoint_dir,
            max_seq_length=EVAL_MAX_SEQ_LENGTH, student_key=STUDENT_KEY,
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
        log_path = loss_log_path(condition)
        loss_summary = summarize_loss_log(read_jsonl(log_path)) if log_path.exists() else None
        if log_path.exists():
            upload_loss_log(seed, condition, log_path)
        for row in eval_rows:
            row["train_steps"] = train_summary.get("steps")
            row["train_n_examples"] = train_summary.get("n_examples")
            row["train_loss_summary"] = loss_summary
        return eval_rows

    def save(seed: int, condition: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        path = result_path(seed, condition)
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=path, repo_id=args.repo, repo_type="model",
                    commit_message=f"third-pair eval seed {seed} {condition}",
                )
                return
            except Exception as e:  # noqa: BLE001 - two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {condition}; kept in {local}")

    summary = run_jobs(
        parse_jobs(args.jobs), evaluate, already_done, save,
        describe=describe_pair3_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
