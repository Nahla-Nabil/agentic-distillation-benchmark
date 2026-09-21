"""Drives all three conditions (base / sft_only / distilled) through the
harness at every configured chain length, plus the perplexity baseline, and
writes results to results/ for analysis/plotting later.

Run in Colab (needs GPU to load a model), after training has produced each
condition's checkpoint (src/adbench/training/train.py) and
data/general_eval/wikitext2_sample.jsonl exists (adbench.data.general_eval):

    python -m adbench.evaluation.run_eval                    # all 3 conditions
    python -m adbench.evaluation.run_eval --condition sft_only # just one

STRUCTURE — same split as training/train.py: config parsing, the harness-run
loop (run_harness_eval_for_condition), and the model_fn wiring
(make_harness_model_fn) are plain Python + CPU torch, unit-tested locally
with fake/scripted models (tests/test_run_eval.py) — no GPU needed. Only
load_condition_model (real Unsloth checkpoint loading) needs Colab; every
other function here is built from and tested against that boundary.

Task sources: harness.tasks.load_tasks(chain_length, source="synthetic") —
the fixed 6-tool vocabulary held constant across chain lengths 1/3/5 (see
harness/tasks.py's module docstring and README's "Why synthetic tasks for
multi-step eval" for why this, not real glaive data, is the eval axis).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from adbench.evaluation.batching import BatchedModelFn, configure_greedy, make_batch_generate_fn
from adbench.evaluation.metrics import (
    DEFAULT_TASK_SET,
    summarize_by_condition_and_chain_length,
    task_state_to_row,
    write_results_csv,
    write_results_jsonl,
)
from adbench.evaluation.perplexity import (
    compute_perplexity,
    general_eval_config,
    load_general_eval_texts,
)
from adbench.harness.executor import ModelFn, run_task
from adbench.harness.tasks import load_tasks
from adbench.harness.tools import (
    ToolRegistry,
    build_demo_registry,
    build_extended_registry,
    build_glaive_registry,
)
from adbench.training.train import (
    ALL_CONDITIONS,
    CONDITIONS,
    REPO_ROOT,
    load_experiment_config,
    resolve_checkpoint_dir,
)


def make_harness_model_fn(model, tokenizer, max_new_tokens: int = 256) -> ModelFn:
    """Wrap a loaded (model, tokenizer) pair as the ModelFn
    harness.executor.run_task() expects: takes the running chat-message
    history, returns the model's raw text completion for the next turn.

    Builds the prompt via tokenizer.apply_chat_template (same convention
    training/train.py::format_training_example uses, so eval prompts look
    like what the model was fine-tuned on), generates, and decodes ONLY the
    newly-generated tokens — not the echoed-back prompt, which
    model.generate() includes in its output by HF convention.
    """

    # Greedy decoding and no default max_length, set once on the generation
    # config (see batching.configure_greedy for why).
    configure_greedy(model)

    def model_fn(messages: list[dict[str, Any]]) -> str:
        import torch

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        new_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens)

    return model_fn


def run_harness_eval_for_condition(
    condition: str,
    model_fn: ModelFn,
    chain_lengths: list[int],
    registry: ToolRegistry | None = None,
    max_retries_per_step: int = 2,
    task_source: str = "synthetic",
    task_set: str = DEFAULT_TASK_SET,
    max_tasks: int | None = None,
    keep_transcript: bool | None = None,
) -> list[dict[str, Any]]:
    """(If `model_fn` is a BatchedModelFn the tasks run concurrently, and the
    rows come back in the same order as a sequential run.)
    Run every task of `task_source` at every given chain length through
    model_fn, converting each result to a metrics.py row tagged with
    `task_set`. Model-agnostic — model_fn can be a real loaded model wrapped by
    make_harness_model_fn(), or a scripted fake for testing (see
    tests/test_run_eval.py). `max_tasks` keeps only the first N tasks per chain
    length (for the seen-tool set, which is large). `keep_transcript` stores each
    task's full conversation in its row (default: on when the environment variable
    ADBENCH_STORE_TRANSCRIPTS=1), for failure-mode analysis."""
    if keep_transcript is None:
        keep_transcript = os.environ.get("ADBENCH_STORE_TRANSCRIPTS") == "1"
    registry = registry or build_demo_registry()
    rows = []
    for chain_length in chain_lengths:
        tasks = load_tasks(chain_length, source=task_source)
        if max_tasks is not None:
            tasks = tasks[:max_tasks]
        def run_one(task, model_fn=model_fn):
            return run_task(model_fn, task, registry, max_retries_per_step=max_retries_per_step)

        if isinstance(model_fn, BatchedModelFn):
            states = model_fn.run_concurrently(run_one, tasks)
        else:
            states = [run_one(task) for task in tasks]
        for task, state in zip(tasks, states, strict=True):
            rows.append(task_state_to_row(condition, task, state, task_set=task_set, keep_transcript=keep_transcript))
    return rows


EXT_TASK_SET = "unseen_tools_ext"


def ext_eval_enabled(experiment_config: dict[str, Any]) -> bool:
    """Also run the extended synthetic set (new templates, 12 tools)? Off unless
    ADBENCH_EXT_EVAL=1 or harness.ext_eval is true, so runs that predate it keep
    their exact task list."""
    default = "1" if experiment_config["harness"].get("ext_eval", False) else "0"
    return os.environ.get("ADBENCH_EXT_EVAL", default) == "1"


def run_ext_eval_for_condition(
    condition: str, model_fn: ModelFn, chain_lengths: list[int], max_retries_per_step: int = 2
) -> list[dict[str, Any]]:
    """The extended synthetic tasks (new templates over the 12-tool vocabulary),
    tagged task_set="unseen_tools_ext" so they never mix into the original
    unseen-tool numbers."""
    return run_harness_eval_for_condition(
        condition, model_fn, chain_lengths, build_extended_registry(), max_retries_per_step,
        task_source="synthetic_ext", task_set=EXT_TASK_SET,
    )


def run_seen_tool_eval_for_condition(
    condition: str, model_fn: ModelFn, n_tasks: int, max_retries_per_step: int = 2
) -> list[dict[str, Any]]:
    """Held-out single-step Glaive tasks over the tools the students were
    TRAINED on. Compared with the unseen-tool set this shows whether a
    condition is strong only on tools it memorised. Note the eval prompt lists
    all eight Glaive tools, while training prompts listed only the one relevant
    tool, so this is a harder prompt than the training one, not a replay of it."""
    if n_tasks <= 0:
        return []
    return run_harness_eval_for_condition(
        condition, model_fn, [1], build_glaive_registry(), max_retries_per_step,
        task_source="glaive_test", task_set="seen_tools", max_tasks=n_tasks,
    )


def run_perplexity_for_condition(model, tokenizer, experiment_config: dict[str, Any]) -> float:
    """Perplexity on the general (non-agentic) wikitext sample — see
    evaluation/perplexity.py. Raises FileNotFoundError with a clear message
    if the sample hasn't been generated yet."""
    eval_config = general_eval_config(experiment_config)
    eval_set_path = REPO_ROOT / eval_config["eval_set"]
    if not eval_set_path.exists():
        raise FileNotFoundError(
            f"{eval_set_path} does not exist — run `python -m adbench.data.general_eval` first."
        )
    texts = load_general_eval_texts(eval_set_path)
    return compute_perplexity(model, tokenizer, texts)


# --------------------------------------------------------------------------
# Orchestration — needs Unsloth/torch/a GPU to load a real checkpoint. Not
# locally testable end-to-end; built from the tested functions above.
# --------------------------------------------------------------------------

def load_condition_model(
    condition: str,
    experiment_config: dict[str, Any],
    models_config: dict[str, Any],
    use_fast_inference: bool = True,
):
    """Load a condition's saved checkpoint (training/train.py::save_checkpoint)
    for inference. Every condition — including "base" — has a checkpoint to
    load from (a freshly-initialized, untrained adapter for "base"), so this
    is the one loading path for all three conditions; see train.py's
    save_checkpoint() docstring for why.

    use_fast_inference=True (the default, used by evaluate_condition() above)
    switches the model into Unsloth's fast-generation mode via
    FastLanguageModel.for_inference() — a real speed difference for
    generation, but NOT what determines whether torch's
    register_forward_hook fires on self_attn.o_proj/mlp.down_proj: loading
    a LoRA checkpoint through Unsloth patches every layer's attention/MLP
    into a fused kernel at FastLanguageModel.get_peft_model() time — i.e.
    at LOAD time, independent of this flag — so those submodules are never
    called as plain nn.Module.forward() either way, and
    register_forward_hook on them never fires.
    analysis/layer_analysis.py's activation extraction depends on exactly
    those hooks firing, so it does NOT use this function at all — see
    layer_analysis.load_student_checkpoint_for_extraction(), which loads
    via plain transformers + peft (no Unsloth) instead.
    """
    from unsloth import FastLanguageModel

    checkpoint_dir = resolve_checkpoint_dir(experiment_config, condition)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint for condition {condition!r} at {checkpoint_dir} — "
            "run `python -m adbench.training.train --condition ...` first."
        )
    student_cfg = models_config["student"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=student_cfg["max_seq_length"],
        load_in_4bit=student_cfg["load_in_4bit"],
    )
    if use_fast_inference:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def eval_batch_size(experiment_config: dict[str, Any]) -> int:
    """Concurrent tasks per generate() call: ADBENCH_EVAL_BATCH, else the
    config's harness.eval_batch_size, else 1 (the original one-at-a-time path)."""
    return int(os.environ.get("ADBENCH_EVAL_BATCH", experiment_config["harness"].get("eval_batch_size", 1)))


def build_model_fn(model, tokenizer, experiment_config: dict[str, Any]):
    batch = eval_batch_size(experiment_config)
    if batch > 1:
        return BatchedModelFn(make_batch_generate_fn(model, tokenizer), max_batch=batch)
    return make_harness_model_fn(model, tokenizer)


def evaluate_condition(
    condition: str,
    experiment_config: dict[str, Any],
    models_config: dict[str, Any],
    chain_lengths: list[int],
) -> tuple[list[dict[str, Any]], float]:
    """Load one condition's checkpoint once, run both the harness eval and
    the perplexity eval against it. Returns (rows, perplexity)."""
    registry = build_demo_registry()
    model, tokenizer = load_condition_model(condition, experiment_config, models_config)
    model_fn = build_model_fn(model, tokenizer, experiment_config)

    max_retries = experiment_config["harness"]["max_retries_per_step"]
    try:
        rows = run_harness_eval_for_condition(
            condition, model_fn, chain_lengths, registry, max_retries_per_step=max_retries
        )
        rows += run_seen_tool_eval_for_condition(
            condition, model_fn, experiment_config["harness"].get("seen_tool_eval_n", 0), max_retries
        )
        if ext_eval_enabled(experiment_config):
            rows += run_ext_eval_for_condition(condition, model_fn, chain_lengths, max_retries)
    finally:
        if isinstance(model_fn, BatchedModelFn):
            model_fn.close()
    perplexity = run_perplexity_for_condition(model, tokenizer, experiment_config)
    return rows, perplexity


def evaluate_ext_only(
    condition: str, experiment_config: dict[str, Any], models_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Just the extended synthetic tasks for one condition (no original tasks, no
    perplexity) — for adding the new task set to checkpoints that were already
    evaluated on the original one."""
    model, tokenizer = load_condition_model(condition, experiment_config, models_config)
    model_fn = build_model_fn(model, tokenizer, experiment_config)
    try:
        return run_ext_eval_for_condition(
            condition, model_fn, experiment_config["harness"]["chain_lengths"],
            experiment_config["harness"]["max_retries_per_step"],
        )
    finally:
        if isinstance(model_fn, BatchedModelFn):
            model_fn.close()


def evaluate_seen_only(
    condition: str, experiment_config: dict[str, Any], models_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Just the held-out seen-tool tasks for one condition (no synthetic tasks,
    no perplexity) — a cheap top-up (~10 minutes per condition) for runs whose
    main evaluation predates argument logging."""
    model, tokenizer = load_condition_model(condition, experiment_config, models_config)
    model_fn = build_model_fn(model, tokenizer, experiment_config)
    try:
        return run_seen_tool_eval_for_condition(
            condition, model_fn, experiment_config["harness"].get("seen_tool_eval_n", 0),
            experiment_config["harness"]["max_retries_per_step"],
        )
    finally:
        if isinstance(model_fn, BatchedModelFn):
            model_fn.close()


def write_eval_outputs(
    all_rows: list[dict[str, Any]],
    perplexities: dict[str, float],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write eval_results.jsonl / .csv (all rows) and eval_summary.json
    (per condition x chain length metrics + perplexity). Returns the
    summary dict that was written, for main() to also print."""
    output_dir = Path(output_dir)
    write_results_jsonl(all_rows, output_dir / "eval_results.jsonl")
    write_results_csv(all_rows, output_dir / "eval_results.csv")

    summary_rows = summarize_by_condition_and_chain_length(all_rows)
    for entry in summary_rows:
        entry["perplexity"] = perplexities.get(entry["condition"])

    output = {"summary": summary_rows, "perplexity_by_condition": perplexities}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", choices=ALL_CONDITIONS, default=None,
        help="Evaluate just this condition; omit to evaluate the three core conditions in one pass.",
    )
    parser.add_argument(
        "--conditions", default=None,
        help="Comma-separated list of conditions to evaluate in one pass (overrides the default three).",
    )
    parser.add_argument(
        "--seen-only", action="store_true",
        help="Evaluate only the held-out seen-tool tasks (with argument accuracy); write to --output-dir.",
    )
    parser.add_argument(
        "--ext-only", action="store_true",
        help="Evaluate only the extended synthetic tasks (new templates, 12 tools); write to --output-dir.",
    )
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    chain_lengths = experiment_config["harness"]["chain_lengths"]
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in conditions if c not in ALL_CONDITIONS]
        if unknown:
            parser.error(f"unknown condition(s) {unknown}; choose from {list(ALL_CONDITIONS)}")
    else:
        conditions = [args.condition] if args.condition else list(CONDITIONS)

    if args.ext_only:
        ext_rows: list[dict[str, Any]] = []
        for condition in conditions:
            print(f"=== Extended-task evaluation: {condition} ===")
            rows = evaluate_ext_only(condition, experiment_config, models_config)
            ext_rows.extend(rows)
            print(f"{condition}: {len(rows)} extended tasks")
        output = write_eval_outputs(ext_rows, {}, REPO_ROOT / args.output_dir)
        print(json.dumps(output["summary"], indent=2))
        return

    if args.seen_only:
        seen_rows: list[dict[str, Any]] = []
        for condition in conditions:
            print(f"=== Seen-tool evaluation: {condition} ===")
            rows = evaluate_seen_only(condition, experiment_config, models_config)
            seen_rows.extend(rows)
            print(f"{condition}: {len(rows)} seen-tool tasks")
        output = write_eval_outputs(seen_rows, {}, REPO_ROOT / args.output_dir)
        print(json.dumps(output["summary"], indent=2))
        return

    all_rows: list[dict[str, Any]] = []
    perplexities: dict[str, float] = {}
    for condition in conditions:
        print(f"=== Evaluating {condition} ===")
        rows, perplexity = evaluate_condition(condition, experiment_config, models_config, chain_lengths)
        all_rows.extend(rows)
        perplexities[condition] = perplexity
        print(f"{condition}: {len(rows)} tasks run, perplexity={perplexity:.2f}")

    output = write_eval_outputs(all_rows, perplexities, REPO_ROOT / args.output_dir)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
