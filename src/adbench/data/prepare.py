"""Build the fixed 500-1000 example subset of glaiveai/glaive-function-calling-v2
used across all three conditions, and freeze its 80/20 train/test split.

Run once, locally (CPU-only, no GPU needed):
    python -m adbench.data.prepare --config configs/data.yaml

TODO:
  - Load the HF dataset, parse out each example's function/tool name(s).
  - Filter down to examples whose tool falls within `selected_tools`
    (configs/data.yaml), enforcing min/max distinct tool count.
  - Subsample to n_examples within [n_examples_min, n_examples_max].
  - Split 80/20 with the configured seed; write data/splits/train.jsonl and
    data/splits/test.jsonl.
  - Safety rail: refuse to overwrite an existing test.jsonl once
    `test_split_frozen: true` — the whole point of the held-out set is that
    it's touched exactly once, at final eval.
  - Record basic subset stats (tool distribution, example count) to
    data/README.md or a sidecar JSON, for the paper's dataset section.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.parse_args()
    raise NotImplementedError("Dataset preparation not yet implemented.")


if __name__ == "__main__":
    main()
