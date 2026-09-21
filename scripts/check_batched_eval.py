"""GPU check: does batched evaluation give the same results as one-at-a-time, and how much faster is it?

    python scripts/check_batched_eval.py --condition base --n 40 --batch 16

Runs the same evenly spaced subset of the synthetic tasks twice on one loaded
checkpoint, once sequentially and once through BatchedModelFn, and prints wall
time, per-task success agreement and any task whose step-by-step tool names
differ. Greedy decoding makes them identical up to padding-related
floating-point noise, so a tiny number of disagreements is expected on some
GPUs; a large number means batched evaluation must not be mixed into the same
table as the sequential results.

Exit status is 1 if success agreement falls below --min-agreement.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Batched generation pads prompts, which makes Unsloth build an attention bias on the first GPU
# while a model split over both T4s keeps later layers on the second one ("Expected all tensors
# to be on the same device"). One GPU holds the 4-bit 4B model easily, so pin the run to it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adbench.evaluation.batching import BatchedModelFn, make_batch_generate_fn  # noqa: E402
from adbench.evaluation.metrics import task_state_to_row  # noqa: E402
from adbench.evaluation.run_eval import load_condition_model, make_harness_model_fn  # noqa: E402
from adbench.harness.executor import run_task  # noqa: E402
from adbench.harness.tasks import load_tasks  # noqa: E402
from adbench.harness.tools import build_demo_registry  # noqa: E402
from adbench.training.train import ALL_CONDITIONS, load_experiment_config  # noqa: E402


def pick_tasks(n: int):
    tasks = [t for chain_length in (1, 3, 5) for t in load_tasks(chain_length)]
    if n >= len(tasks):
        return tasks
    step = len(tasks) / n
    return [tasks[int(i * step)] for i in range(n)]


def print_generation_stats(generate_batch, wall_seconds: float) -> None:
    """Where the batched pass spent its time: per generate() call, the batch size, seconds and
    how many tokens each row produced. Rows that use the whole token budget (never emit their
    end-of-turn token) stretch the whole batch to the budget."""
    stats = generate_batch.stats
    print(f"batched pass: {wall_seconds:.0f}s in {len(stats)} generate() calls")
    for i, c in enumerate(stats):
        n = sorted(c["new_tokens"])
        print(f"  call {i:2d}: batch={c['batch']:2d} prompt={c['prompt_tokens']:4d} tok  {c['seconds']:6.1f}s  "
              f"new tokens min/median/max = {n[0]}/{n[len(n) // 2]}/{n[-1]}  rows at budget: {c['hit_budget']}")
    total_tokens = sum(sum(c["new_tokens"]) for c in stats)
    steps = sum(max(c["new_tokens"]) for c in stats)
    seconds = sum(c["seconds"] for c in stats)
    print(f"  decode steps (sum of per-call max): {steps}  ->  {1000 * seconds / max(steps, 1):.0f} ms per decode step")
    print(f"  useful tokens / decode-step slots: {total_tokens} / {sum(c['batch'] * max(c['new_tokens']) for c in stats)}")
    for text in generate_batch.capped_samples:
        print("  --- a row that used the whole budget (last 600 chars) ---")
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="base")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--min-agreement", type=float, default=0.95)
    parser.add_argument(
        "--skip-sequential", action="store_true",
        help="Only run the batched pass and print where its time goes (no comparison, much faster).",
    )
    args = parser.parse_args()

    experiment_config = load_experiment_config("configs/experiment.yaml")
    models_config = load_experiment_config("configs/models.yaml")
    max_retries = experiment_config["harness"]["max_retries_per_step"]
    registry = build_demo_registry()
    tasks = pick_tasks(args.n)
    model, tokenizer = load_condition_model(args.condition, experiment_config, models_config)

    sequential, t_seq = [], 0.0
    if not args.skip_sequential:
        sequential_fn = make_harness_model_fn(model, tokenizer)
        start = time.monotonic()
        sequential = [task_state_to_row(args.condition, t, run_task(sequential_fn, t, registry, max_retries)) for t in tasks]
        t_seq = time.monotonic() - start

    generate_batch = make_batch_generate_fn(model, tokenizer)
    batched_fn = BatchedModelFn(generate_batch, max_batch=args.batch)
    try:
        start = time.monotonic()
        states = batched_fn.run_concurrently(lambda t: run_task(batched_fn, t, registry, max_retries), tasks)
        t_batch = time.monotonic() - start
    finally:
        batched_fn.close()
    batched = [task_state_to_row(args.condition, t, s) for t, s in zip(tasks, states, strict=True)]

    print_generation_stats(generate_batch, t_batch)
    if args.skip_sequential:
        return

    agree = sum(a["success"] == b["success"] for a, b in zip(sequential, batched, strict=True))
    identical_steps = sum(
        [s["tool_name"] for s in a["steps"]] == [s["tool_name"] for s in b["steps"]]
        for a, b in zip(sequential, batched, strict=True)
    )
    print(f"tasks: {len(tasks)}  condition: {args.condition}")
    print(f"sequential: {t_seq:.0f}s   batched (batch {args.batch}): {t_batch:.0f}s   speed-up: {t_seq / t_batch:.1f}x")
    print(f"success agreement: {agree}/{len(tasks)}   identical step-by-step tool names: {identical_steps}/{len(tasks)}")
    print(f"batch sizes used: {sorted(set(batched_fn.batch_sizes))}, {len(batched_fn.batch_sizes)} generate() calls")
    for a, b in zip(sequential, batched, strict=True):
        if [s["tool_name"] for s in a["steps"]] != [s["tool_name"] for s in b["steps"]]:
            print(f"  differs: {a['task_id']}  seq={[s['tool_name'] for s in a['steps']]}  batch={[s['tool_name'] for s in b['steps']]}")
    if agree / len(tasks) < args.min_agreement:
        raise SystemExit(f"agreement {agree / len(tasks):.2f} is below {args.min_agreement}")


if __name__ == "__main__":
    main()
