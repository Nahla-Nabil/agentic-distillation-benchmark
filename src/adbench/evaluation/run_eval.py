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
from pathlib import Path
from typing import Any

from adbench.evaluation.metrics import (
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
from adbench.harness.tools import ToolRegistry, build_demo_registry
from adbench.training.train import (
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

    def model_fn(messages: list[dict[str, Any]]) -> str:
        import torch

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = torch.tensor([tokenizer(prompt_text)["input_ids"]])
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens)

    return model_fn


def run_harness_eval_for_condition(
    condition: str,
    model_fn: ModelFn,
    chain_lengths: list[int],
    registry: ToolRegistry | None = None,
    max_retries_per_step: int = 2,
) -> list[dict[str, Any]]:
    """Run every synthetic task at every given chain length through
    model_fn, converting each result to a metrics.py row. Model-agnostic —
    model_fn can be a real loaded model wrapped by make_harness_model_fn(),
    or a scripted fake for testing (see tests/test_run_eval.py)."""
    registry = registry or build_demo_registry()
    rows = []
    for chain_length in chain_lengths:
        tasks = load_tasks(chain_length, source="synthetic")
        for task in tasks:
            state = run_task(model_fn, task, registry, max_retries_per_step=max_retries_per_step)
            rows.append(task_state_to_row(condition, task, state))
    return rows


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

def load_condition_model(condition: str, experiment_config: dict[str, Any], models_config: dict[str, Any]):
    """Load a condition's saved checkpoint (training/train.py::save_checkpoint)
    for inference. Every condition — including "base" — has a checkpoint to
    load from (a freshly-initialized, untrained adapter for "base"), so this
    is the one loading path for all three conditions; see train.py's
    save_checkpoint() docstring for why."""
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
    FastLanguageModel.for_inference(model)
    return model, tokenizer


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
    model_fn = make_harness_model_fn(model, tokenizer)

    max_retries = experiment_config["harness"]["max_retries_per_step"]
    rows = run_harness_eval_for_condition(
        condition, model_fn, chain_lengths, registry, max_retries_per_step=max_retries
    )
    perplexity = run_perplexity_for_condition(model, tokenizer, experiment_config)
    return rows, perplexity


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
        "--condition", choices=CONDITIONS, default=None,
        help="Evaluate just this condition; omit to evaluate all three in one pass.",
    )
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    chain_lengths = experiment_config["harness"]["chain_lengths"]
    conditions = [args.condition] if args.condition else list(CONDITIONS)

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
