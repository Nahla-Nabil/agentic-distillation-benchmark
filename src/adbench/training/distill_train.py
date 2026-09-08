"""Condition 3 — distillation-guided student (treatment): same data and LoRA
setup as sft_train.py, but the loss adds a KD term against teacher
(Qwen3-14B) logits, via losses.combined_loss. Intended to run inside
notebooks/03_train_distill.ipynb on a Colab T4.

TODO:
  - Load teacher (4-bit, inference-only) and student (4-bit, LoRA) per
    configs/models.yaml.
  - Per training step: forward both models on the same batch, compute
    losses.combined_loss(student_logits, teacher_logits, labels, cfg).
  - Memory budget on a single T4 is the main constraint of holding two
    14B/4B-class models resident — see the "on-the-fly vs precomputed
    teacher logits" note in losses.py; likely need the precomputed route.
  - Save adapter to checkpoints/distilled/ (path from configs/experiment.yaml).
  - Everything else (data, LoRA rank, epochs, seed) must match sft_train.py
    exactly so the two conditions are only different in the loss function —
    that's the whole point of the three-way comparison.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.parse_args()
    raise NotImplementedError("Distillation training not yet implemented.")


if __name__ == "__main__":
    main()
