"""Condition 2 — SFT-only student (control): standard supervised fine-tuning
of Qwen3-4B-Instruct-2507 on data/splits/train.jsonl via Unsloth, no teacher
involved. Intended to run inside notebooks/02_train_sft.ipynb on a Colab T4.

TODO:
  - Load student per configs/models.yaml via Unsloth's FastLanguageModel
    (4-bit, LoRA config as specified there).
  - Standard causal-LM SFT loop (Unsloth + TRL's SFTTrainer, or a plain
    HF Trainer) over the tool-calling examples in train.jsonl.
  - Save adapter to checkpoints/sft_only/ (path from configs/experiment.yaml).
  - Log training curves (loss) so they can be compared against
    distill_train.py's curves in the README's results section.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.parse_args()
    raise NotImplementedError("SFT training not yet implemented.")


if __name__ == "__main__":
    main()
