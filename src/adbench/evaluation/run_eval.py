"""Drives all three conditions (base / sft_only / distilled) through the
harness at every configured chain length, plus the perplexity baseline, and
writes results to results/ for the README's results section and the
layer-analysis notebook to pick up.

Run in Colab (needs GPU to load the models) after both training notebooks
have produced their checkpoints:
    python -m adbench.evaluation.run_eval --config configs/experiment.yaml

TODO:
  - For each condition in configs/experiment.yaml:
      - load the model (base weights, or base + LoRA adapter from checkpoint)
      - wrap it as a harness.executor.ModelFn
      - for each chain_length in [1, 3, 5]:
          tasks = harness.tasks.load_tasks(chain_length)
          states = [harness.executor.run_task(model_fn, t, registry) for t in tasks]
          log step-by-step outcomes, not just pass/fail, to results/<condition>/chain_<n>.jsonl
      - run perplexity.compute_perplexity on the general LM eval set
  - Write a summary CSV/JSON (per condition x chain length: success rate,
    error-recovery rate, perplexity) to results/summary.json.
  - IMPORTANT: this is the *only* script that should ever read
    data/splits/test.jsonl-derived tasks — keep it that way.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.parse_args()
    raise NotImplementedError("Evaluation loop not yet implemented.")


if __name__ == "__main__":
    main()
