"""Second model pair (does the finding generalise beyond one model size?): train + evaluate
one (seed, condition) job at a time for student_small (Qwen3-1.7B) + teacher_8b (Qwen3-8B) —
~4.7x apart, vs the main pair's ~3.5x (Qwen3-14B teacher / Qwen3-4B student).

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.second_pair_worker --jobs 0:sft_only,0:distilled_8b --work-dir /kaggle/working/p0

Conditions: "base", "sft_only", "distilled_8b" (KD from teacher_8b — the same condition id the
main pipeline defines for its own distilled_8b control, reused here with student_small instead;
this worker is what actually points it at the small student, via train_condition's
`student_key`), and "self_distill_small" (KD from student_small's own frozen base — a SEPARATE
condition id from the main pipeline's "self_distill", because self-distillation needs a teacher
pointed at THIS pair's student, not the main pair's 4B one). Uses the exact same training data as
the main pipeline (configs/data.yaml's default 8-tool split — this experiment varies model size,
not tool diversity) and the exact same 121-task unseen_tools eval set (chains 1/3/5), so results
are directly comparable to the main pair's numbers.

Same shape as evaluation/ablation_worker.py (also trains, not just evaluates) and
ext_worker.py/bfcl_worker.py (two workers, resumable, per-worker time budget) — job bookkeeping
reused from worker_utils.py. Checkpoints are trained locally and deleted after eval, not
uploaded (deterministic seed makes them cheap to reproduce). Uploads:

    runs/v2-seed<k>/results/stages/pair2_<condition>.json

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
TASK_SET = "unseen_tools"
STUDENT_KEY = "student_small"


def result_path(seed: int, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/pair2_{condition}.json"


def describe_pair2_result(rows: list[dict[str, Any]]) -> str:
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

    from adbench.evaluation.run_eval import load_condition_model, make_harness_model_fn, run_harness_eval_for_condition
    from adbench.harness.tools import build_demo_registry
    from adbench.training.train import REPO_ROOT, free_gpu_memory, load_experiment_config, train_condition

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

    def evaluate(seed: int, condition: str) -> list[dict[str, Any]]:
        if not glaive_train_path.exists():
            raise FileNotFoundError(
                f"{glaive_train_path} does not exist — run `python -m adbench.data.prepare` first "
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
            condition, experiment_config, models_config, checkpoint_dir=checkpoint_dir, student_key=STUDENT_KEY
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
            row["train_steps"] = train_summary.get("steps")
            row["train_n_examples"] = train_summary.get("n_examples")
        return eval_rows

    def save(seed: int, condition: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        path = result_path(seed, condition)
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=path, repo_id=args.repo, repo_type="model",
                    commit_message=f"second-pair eval seed {seed} {condition}",
                )
                return
            except Exception as e:  # noqa: BLE001 — two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} {condition}; kept in {local}")

    summary = run_jobs(
        parse_jobs(args.jobs), evaluate, already_done, save,
        describe=describe_pair2_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
