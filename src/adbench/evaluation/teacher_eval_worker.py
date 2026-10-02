"""Evaluate a TEACHER model itself on the 121-task primary set (eval only, no training).

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.teacher_eval_worker --jobs 0:teacher,0:teacher_olmo13 --work-dir /kaggle/working/t0

Why (added 2026-10-03): the paper's teacher increment (Distilled - Self-distilled) could not be
related to how well each teacher performs the task, because no teacher had been run through the
harness. A job's "condition" is a configs/models.yaml key (`teacher`, `teacher_8b`, `teacher_olmo`,
`teacher_olmo13`, ...); the raw Hugging Face model is loaded in 4-bit exactly as KD loads it, with a
4,096-token context, and run through the same harness as every student (greedy, 256 new tokens,
thinking mode off via run_eval.make_harness_model_fn). The seed is only a bookkeeping slot: an
untrained, greedy-decoded model has no seed dependence, so use seed 0. Uploads

    runs/v2-seed<k>/results/stages/teacher_eval_<models.yaml key>.json

Needs HF_TOKEN (write access to ADBENCH_HF_REPO). Loading real weights needs a GPU, so main() is
reviewed by reading; result_path() is unit-tested.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

from adbench.evaluation.worker_utils import parse_jobs, run_jobs, split_jobs  # noqa: F401 - re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"
EVAL_MAX_SEQ_LENGTH = 4096


def result_path(seed: int, model_key: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/teacher_eval_{model_key}.json"


def describe_teacher_result(rows: list[dict[str, Any]]) -> str:
    rate = sum(r["success"] for r in rows) / max(len(rows), 1)
    return f"{len(rows)} eval tasks, success {rate:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<models.yaml key> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to upload results.")
    os.environ["ADBENCH_STORE_TRANSCRIPTS"] = "1"

    from huggingface_hub import HfApi

    from adbench.evaluation.batching import BatchedModelFn
    from adbench.evaluation.run_eval import build_model_fn, run_primary_ctx4096_eval_for_condition
    from adbench.training.train import free_gpu_memory, load_experiment_config

    api = HfApi(token=token)
    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    harness = experiment_config["harness"]
    out_dir = Path(args.work_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def already_done(seed: int, key: str) -> bool:
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, key), repo_type="model")

    def evaluate(seed: int, key: str) -> list[dict[str, Any]]:
        from unsloth import FastLanguageModel

        cfg = models_config[key]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["hf_id"], max_seq_length=EVAL_MAX_SEQ_LENGTH,
            load_in_4bit=cfg.get("load_in_4bit", True), device_map={"": 0},
        )
        FastLanguageModel.for_inference(model)
        model_fn = build_model_fn(model, tokenizer, experiment_config)
        try:
            rows = run_primary_ctx4096_eval_for_condition(
                f"teacher_{key}", model_fn, harness["chain_lengths"], harness["max_retries_per_step"],
            )
        finally:
            if isinstance(model_fn, BatchedModelFn):
                model_fn.close()
            del model
            gc.collect()
            free_gpu_memory()
        for row in rows:
            row["model_key"], row["hf_id"] = key, cfg["hf_id"]
        return rows

    def save(seed: int, key: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"teacher_eval_seed{seed}_{key}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=result_path(seed, key),
                    repo_id=args.repo, repo_type="model", commit_message=f"teacher eval {key}",
                )
                return
            except Exception as e:  # noqa: BLE001 - two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload {key}; kept in {local}")

    summary = run_jobs(
        parse_jobs(args.jobs), evaluate, already_done, save,
        describe=describe_teacher_result, max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{c}" for s, c in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
