"""Combined KD + SFT loss.

    total_loss = sft_weight * CE(student_logits, labels)
               + kd_weight  * KD(student_logits, teacher_logits, temperature)

Keeping sft_weight/kd_weight as explicit, independently-set config values
(rather than a single interpolation alpha) is what lets sft_train.py and
distill_train.py share this one function: SFT-only sets kd_weight=0.0.

TODO:
  - Implement kd_divergence(student_logits, teacher_logits, temperature):
    standard softened-softmax KL divergence, scaled by temperature**2.
  - Implement combined_loss(student_logits, teacher_logits, labels, cfg).
  - Decide on-the-fly teacher inference vs precomputed teacher logits —
    given the T4's single GPU, precomputing/caching teacher logits (or
    top-k logits) once over the train split likely fits memory better than
    holding both models resident during the student's backward pass.
  - Vocab mismatch: confirm Qwen3-14B and Qwen3-4B-Instruct-2507 share a
    tokenizer/vocab before assuming logits are directly comparable — if not,
    KD needs to operate on a shared subword space or teacher-forced
    token-level alignment.
"""

from dataclasses import dataclass


@dataclass
class KDLossConfig:
    temperature: float = 2.0
    kd_weight: float = 0.5
    sft_weight: float = 0.5


def kd_divergence(student_logits, teacher_logits, temperature: float):
    raise NotImplementedError


def combined_loss(student_logits, teacher_logits, labels, cfg: KDLossConfig):
    raise NotImplementedError
