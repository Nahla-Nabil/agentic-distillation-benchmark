"""Combined KD + SFT loss.

    total_loss = sft_weight * CE(student_logits, labels)
               + kd_weight  * KD(student_logits, teacher_logits, temperature)

Keeping sft_weight/kd_weight as explicit, independently-set config values
(rather than a single interpolation alpha) is what lets sft_train.py and
distill_train.py share this one function: SFT-only sets kd_weight=0.0.

TOKENIZER COMPATIBILITY — VERIFIED (see scripts/verify_tokenizer_compatibility.py):
    Qwen/Qwen3-14B and Qwen/Qwen3-4B-Instruct-2507 share the identical
    tokenizer (Qwen2Tokenizer, vocab_size=151643, same eos/pad tokens,
    identical token-id sequences across code/JSON-tool-call/CJK samples).
    Logits are therefore directly comparable position-for-position and
    logit-level KD (kd_divergence/combined_loss below) is safe to use as the
    primary path.

    SEQUENCE-LEVEL DISTILLATION FALLBACK — kept in reserve, not currently
    used: if a future teacher/student pairing does NOT share a tokenizer,
    logit-level KD isn't valid (the two vocabs don't index the same
    distribution) and training should fall back to sequence-level
    distillation instead — i.e. train the student with plain SFT
    cross-entropy against the teacher's *generated text* (re-tokenized in
    the student's own vocab), rather than against the teacher's token-level
    probability distribution. It's simpler and tokenizer-agnostic by
    construction, at the cost of a softer training signal (only the
    teacher's argmax path, not its full distribution, transfers). See
    sequence_level_distillation_loss() below — left unimplemented since it
    is not needed for the current model pair, but the interface is fixed so
    switching later is a config change, not a rewrite.

TODO:
  - Implement kd_divergence(student_logits, teacher_logits, temperature):
    standard softened-softmax KL divergence, scaled by temperature**2.
  - Implement combined_loss(student_logits, teacher_logits, labels, cfg).
  - Decide on-the-fly teacher inference vs precomputed teacher logits —
    given the T4's single GPU, precomputing/caching teacher logits (or
    top-k logits) once over the train split likely fits memory better than
    holding both models resident during the student's backward pass.
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


def sequence_level_distillation_loss(student_logits, teacher_generated_ids, labels):
    """Fallback path, not currently used (see module docstring) — plain CE
    of the student against the teacher's generated token sequence instead of
    its logits. Only needed if a future teacher/student pair fails the
    tokenizer check in scripts/verify_tokenizer_compatibility.py."""
    raise NotImplementedError
