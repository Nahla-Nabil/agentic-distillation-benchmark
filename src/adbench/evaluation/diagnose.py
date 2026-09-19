"""Print what a condition's model actually generates for a few eval prompts,
next to the exact completion it was trained to produce — for debugging a
condition whose harness success rate is implausibly low.

    python -m adbench.evaluation.diagnose --condition sft_only

Needs a GPU (loads the real checkpoint through the same path as
run_eval.py). The eval loop only stores parsed outcomes (tool name, error
type), never the raw text, so a parse failure like "Model produced a final
answer instead of the expected tool call." can't be diagnosed from
eval_results.jsonl alone — this prints the raw text.
"""

import argparse
import json

from adbench.evaluation.run_eval import load_condition_model, make_harness_model_fn
from adbench.harness.executor import build_system_prompt, parse_model_output
from adbench.harness.tasks import load_tasks
from adbench.harness.tools import build_demo_registry, build_glaive_registry
from adbench.training.train import (
    ALL_CONDITIONS,
    REPO_ROOT,
    format_training_example,
    load_experiment_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="sft_only")
    parser.add_argument(
        "--loader", choices=["unsloth", "unsloth_nofast", "plain"], default="unsloth",
        help="unsloth: what run_eval.py uses (FastLanguageModel + for_inference). "
             "unsloth_nofast: same but without for_inference(). "
             "plain: transformers + peft, Unsloth never imported (run in a fresh process).",
    )
    parser.add_argument("--skip-training-target", action="store_true")
    parser.add_argument(
        "--require-tool-call", action="store_true",
        help="Exit non-zero unless every generation parses as a tool call — a cheap guard "
             "to run before a multi-hour eval.",
    )
    parser.add_argument(
        "--max-failures", type=int, default=0,
        help="With --require-tool-call, how many non-tool-call generations to tolerate "
             "(generation is sampled, so one odd sample should not abort a long run).",
    )
    parser.add_argument("--n", type=int, default=3, help="Eval prompts to generate for.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    if args.loader == "plain":
        from adbench.analysis.layer_analysis import load_student_checkpoint_for_extraction
        model, tokenizer = load_student_checkpoint_for_extraction(
            args.condition, experiment_config, models_config
        )
    else:
        model, tokenizer = load_condition_model(
            args.condition, experiment_config, models_config,
            use_fast_inference=(args.loader == "unsloth"),
        )
    print(f"=== LOADER: {args.loader}, CONDITION: {args.condition} ===")

    # 1. What the model was trained to output (first training record).
    if not args.skip_training_target:
        data_config = load_experiment_config(REPO_ROOT / "configs" / "data.yaml")
        train_path = REPO_ROOT / data_config["output"]["train_path"]
        with open(train_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        example = format_training_example(record, tokenizer, build_glaive_registry())
        trained_on = [t for t, label in zip(example["input_ids"], example["labels"]) if label != -100]
        print("=== TRAINING TARGET (tokens with label != -100) ===")
        print(repr(tokenizer.decode(trained_on)))

    # 2. What the model generates on eval prompts.
    registry = build_demo_registry()
    model_fn = make_harness_model_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)
    n_failed = 0
    for task in load_tasks(1, source="synthetic")[: args.n]:
        messages = [
            {"role": "system", "content": build_system_prompt(registry)},
            {"role": "user", "content": task.user_goal},
        ]
        raw = model_fn(messages)
        print(f"\n=== EVAL {task.task_id}: {task.user_goal!r} (expects {task.expected_tool_sequence}) ===")
        print("RAW:", repr(raw[:300]))
        try:
            parsed = parse_model_output(raw)
            print("PARSED:", type(parsed).__name__, str(parsed)[:200])
            if type(parsed).__name__ != "ToolCall":
                n_failed += 1
        except Exception as e:  # noqa: BLE001 — diagnostic printout only
            print("PARSE ERROR:", type(e).__name__, e)
            n_failed += 1

    if args.require_tool_call and n_failed > args.max_failures:
        raise SystemExit(f"{n_failed}/{args.n} generations for {args.condition!r} were not a valid tool call.")


if __name__ == "__main__":
    main()
