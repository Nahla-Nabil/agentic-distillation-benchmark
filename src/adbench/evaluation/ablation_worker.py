"""Tool-diversity ablation: train + evaluate one (seed, n_tools, condition) job at a time.

    CUDA_VISIBLE_DEVICES=0 python -m adbench.evaluation.ablation_worker --jobs 0:2:sft_only,0:2:distilled --work-dir /kaggle/working/ab0

Research question (see project memory / README's positioning writeup): does a richer
supervision signal (KD from an external teacher) substitute for tool diversity when the
training data covers few distinct tools? The main pipeline always trains on 8 tools; this holds
the TOTAL example count fixed at 800 (matching the main pipeline) while varying how many
distinct tools those 800 examples are drawn from (2, 4, 8 — see
adbench.data.prepare.select_top_n_tools/ablation_paths), then evaluates on the exact same
121-task unseen-tool set (build_demo_registry, chains 1/3/5) every other condition in this
project is evaluated on, so ablation numbers are directly comparable to the existing v2 results.

**n_tools=8 is NOT retrained here** — with the same 8 curated tools, the same per_tool_target
(100), and the same split seed as the main pipeline's default configs/data.yaml, --n-tools 8
would reproduce the exact same train.jsonl the main pipeline already used; its sft_only/distilled
numbers are simply the already-published seed-0..4 results (task_set "unseen_tools"). Only
n_tools in {2, 4} need new training runs.

The ablation's data split (train_ntoolsN.jsonl / test_ntoolsN.jsonl) must be prepared ONCE,
before starting workers, with `python -m adbench.data.prepare --n-tools N` for each N in use —
this needs only network + CPU, not a GPU, and both GPU worker subprocesses (spawned from the
same notebook, sharing the same repo checkout) then read the same local files. It is not
downloaded from Hugging Face and not re-derived per job.

Same shape as evaluation/ext_worker.py and bfcl_worker.py (two workers, resumable via a
per-(seed,n_tools,condition) result file already on Hugging Face, a per-worker time budget) — job
bookkeeping reused from worker_utils.py. Trained checkpoints are NOT uploaded (kept only for the
duration of one job, then deleted): the split is deterministic (fixed seeds), so a checkpoint is
cheap to reproduce if ever needed again, and this keeps the ablation's Hugging Face footprint to
just its (much smaller) result rows. Uploads:

    runs/v2-seed<k>/results/stages/ablation_n<N>_<condition>.json

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

from adbench.evaluation.worker_utils import run_jobs, split_jobs  # noqa: F401 — re-exported

DEFAULT_REPO = "NahlaNabil/adbench-run"
TASK_SET = "unseen_tools"


def parse_ablation_jobs(spec: str) -> list[tuple[int, int, str]]:
    """'0:2:sft_only,0:4:distilled' -> [(0, 2, 'sft_only'), (0, 4, 'distilled')]."""
    jobs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"job {item!r} must look like <seed>:<n_tools>:<condition>")
        seed, n_tools, condition = parts
        jobs.append((int(seed), int(n_tools), condition))
    return jobs


def result_path(seed: int, n_tools: int, condition: str) -> str:
    return f"runs/v2-seed{seed}/results/stages/ablation_n{n_tools}_{condition}.json"


def describe_ablation_result(rows: list[dict[str, Any]]) -> str:
    rate = sum(r["success"] for r in rows) / max(len(rows), 1)
    steps = rows[0].get("train_steps", "?") if rows else "?"
    return f"{steps} train steps, {len(rows)} eval tasks, success {rate:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True, help="comma-separated <seed>:<n_tools>:<condition> list")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--repo", default=os.environ.get("ADBENCH_HF_REPO", DEFAULT_REPO))
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; a worker needs it to upload results.")

    from huggingface_hub import HfApi

    from adbench.data.prepare import ablation_paths, load_config
    from adbench.evaluation.run_eval import load_condition_model, make_harness_model_fn, run_harness_eval_for_condition
    from adbench.harness.tools import build_demo_registry
    from adbench.training.train import free_gpu_memory, load_experiment_config, train_condition

    api = HfApi(token=token)
    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    data_config = load_config(args.data_config)
    harness = experiment_config["harness"]
    work_dir = Path(args.work_dir)
    out_dir = work_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_demo_registry()

    def already_done(seed: int, n_tools: int, condition: str) -> bool:
        return api.file_exists(repo_id=args.repo, filename=result_path(seed, n_tools, condition), repo_type="model")

    def evaluate(seed: int, n_tools: int, condition: str) -> list[dict[str, Any]]:
        train_path = ablation_paths(data_config, n_tools)["train_path"]
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} does not exist — run "
                f"`python -m adbench.data.prepare --n-tools {n_tools}` first (once, before starting workers)."
            )
        checkpoint_dir = work_dir / "checkpoints" / f"n{n_tools}" / condition
        os.environ["ADBENCH_SEED"] = str(seed)
        train_summary = train_condition(
            condition, experiment_config, models_config,
            glaive_train_path=train_path, checkpoint_dir_override=checkpoint_dir,
        )
        from adbench.evaluation.run_eval import load_condition_model

        model, tokenizer = load_condition_model(condition, experiment_config, models_config, checkpoint_dir=checkpoint_dir)
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
            row["n_tools"] = n_tools
            row["train_steps"] = train_summary.get("steps")
            row["train_n_examples"] = train_summary.get("n_examples")
        return eval_rows

    def save(seed: int, n_tools: int, condition: str, rows: list[dict[str, Any]]) -> None:
        local = out_dir / f"n{n_tools}_seed{seed}_{condition}.json"
        local.write_text(json.dumps(rows), encoding="utf-8")
        path = result_path(seed, n_tools, condition)
        for attempt in range(1, 4):
            try:
                api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=path, repo_id=args.repo, repo_type="model",
                    commit_message=f"ablation n{n_tools} {condition} seed {seed}",
                )
                return
            except Exception as e:  # noqa: BLE001 — two workers may commit at the same moment
                print(f"  upload attempt {attempt} failed: {type(e).__name__}: {e}")
                time.sleep(5 * attempt)
        print(f"  WARNING: could not upload seed {seed} n{n_tools} {condition}; kept in {local}")

    jobs = parse_ablation_jobs(args.jobs)
    # worker_utils.run_jobs works over (seed, key) pairs; fold n_tools into the key string and
    # split it back apart in each callback, rather than duplicating run_jobs's loop/time-budget
    # logic here for a 3-tuple.
    keyed_jobs = [(seed, f"{n_tools}:{condition}") for seed, n_tools, condition in jobs]

    def _split(key: str) -> tuple[int, str]:
        n_tools_str, condition = key.split(":", 1)
        return int(n_tools_str), condition

    summary = run_jobs(
        keyed_jobs,
        evaluate=lambda seed, key: evaluate(seed, *_split(key)),
        already_done=lambda seed, key: already_done(seed, *_split(key)),
        save=lambda seed, key, result: save(seed, *_split(key), result),
        describe=describe_ablation_result,
        max_minutes=args.max_minutes,
    )
    print(json.dumps({k: [f"{s}:{key}" for s, key in v] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
