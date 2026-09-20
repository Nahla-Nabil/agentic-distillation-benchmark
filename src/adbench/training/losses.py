"""Combined KD + SFT loss.

    total_loss = sft_weight * CE(student_logits, labels)
               + kd_weight  * KD(student_logits, teacher_logits, temperature)

Keeping sft_weight/kd_weight as explicit, independently-set config values
(rather than a single interpolation alpha) is what lets the "sft_only" and
"distilled" conditions in training/train.py share this one function:
sft_only runs with kd_weight forced to 0.0 (see train.py's
resolve_training_config), so the two conditions differ *only* in these
weights, nothing else about the run.

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

TEACHER INFERENCE STRATEGY: on-the-fly (teacher forward pass alongside the
student's, both under the same batch, teacher under torch.no_grad()) rather
than precomputed/cached teacher logits. Precomputing would save the
teacher's forward-pass cost per step, but a full (batch, seq_len, vocab)
logit tensor per training example is large to cache to disk at this
vocab size (151,643) — on-the-fly recomputation is simpler to get right
first and is what train.py's training_step() implements; switching to a
cached-logits path later only requires replacing training_step()'s teacher
forward call with a lookup, the loss math here is unaffected either way.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KDLossConfig:
    temperature: float = 2.0
    kd_weight: float = 0.5
    sft_weight: float = 0.5
    # Label smoothing on the SFT cross-entropy term only (the "regularised SFT"
    # control: it discourages the collapse onto hard labels without any teacher).
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing!r}.")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature!r}.")
        if self.kd_weight < 0:
            raise ValueError(f"kd_weight must be >= 0, got {self.kd_weight!r}.")
        if self.sft_weight < 0:
            raise ValueError(f"sft_weight must be >= 0, got {self.sft_weight!r}.")
        if self.kd_weight == 0 and self.sft_weight == 0:
            raise ValueError("kd_weight and sft_weight can't both be 0 — the loss would always be 0.")


def kd_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softened-softmax KL divergence KL(teacher || student), scaled by
    temperature**2 (Hinton et al. 2015's convention — keeps gradient
    magnitude comparable across temperature choices, since softening with a
    higher T shrinks the raw KL by roughly 1/T**2).

    student_logits, teacher_logits: (..., vocab) raw (pre-softmax) logits,
        same shape, teacher's typically produced under torch.no_grad() by
        the caller (this function doesn't care either way — it never
        touches .requires_grad itself, just computes a differentiable
        function of both tensors as given).
    mask: optional boolean/float tensor broadcastable to
        student_logits.shape[:-1] (e.g. (batch, seq_len)) — positions where
        mask is falsy/0 are excluded from the mean (typically prompt/padding
        tokens outside the labeled completion span). If omitted, every
        position contributes equally.

    Returns a scalar tensor (0-dim).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student_logits shape {tuple(student_logits.shape)} != "
            f"teacher_logits shape {tuple(teacher_logits.shape)}."
        )

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # F.kl_div(reduction="none") computes target * (log(target) - input)
    # elementwise; summing over the vocab dim gives the per-position KL.
    kl_per_position = F.kl_div(
        student_log_probs, teacher_probs, reduction="none", log_target=False
    ).sum(dim=-1)
    kl_per_position = kl_per_position * (temperature ** 2)

    if mask is None:
        return kl_per_position.mean()

    mask = mask.to(dtype=kl_per_position.dtype)
    if mask.shape != kl_per_position.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} doesn't match the per-position "
            f"KL shape {tuple(kl_per_position.shape)} (student/teacher logits "
            "minus the vocab dim)."
        )
    denom = mask.sum().clamp(min=1.0)
    return (kl_per_position * mask).sum() / denom


def combined_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    labels: torch.Tensor,
    cfg: KDLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """total_loss = cfg.sft_weight * CE(student_logits, labels)
                  + cfg.kd_weight  * kd_divergence(student_logits, teacher_logits, cfg.temperature)

    labels: (..., ) int64 token ids aligned with student_logits' leading
        dims (caller is responsible for any causal-LM shift-by-one before
        calling this — this function just computes CE/KD over whatever
        positions it's given). Positions equal to -100 (the standard HF
        "ignore this position" convention) are excluded from BOTH the SFT
        cross-entropy and, via the same mask, the KD term — so the two loss
        terms are always computed over the exact same positions (typically
        the completion span, not the prompt).

    teacher_logits: required (and must match student_logits' shape) when
        cfg.kd_weight > 0; ignored (may be None) when cfg.kd_weight == 0 —
        the KD term is skipped entirely rather than computed-and-multiplied-
        by-zero, so a caller with kd_weight=0 (the "sft_only" condition)
        never needs a teacher model loaded at all.

    Returns (total_loss, components) where total_loss is the scalar tensor
    to call .backward() on, and components is a plain dict of detached
    Python floats — {"sft_loss", "kd_loss", "total_loss"} — for logging
    each term separately (train.py logs these to the per-condition loss
    curve), without holding onto the computation graph.
    """
    if student_logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"student_logits shape {tuple(student_logits.shape)} (minus the vocab "
            f"dim) must match labels shape {tuple(labels.shape)}."
        )

    mask = labels != -100
    vocab_size = student_logits.size(-1)

    # Label smoothing averages -log p over the WHOLE vocabulary, and in fp16 (T4) the
    # log-softmax of unlikely tokens underflows to -inf, making the loss inf on every
    # step. Only that variant needs the float32 cast; leaving the plain CE path
    # untouched keeps every other condition numerically identical to earlier runs.
    ce_logits = student_logits.float() if cfg.label_smoothing > 0 else student_logits
    sft_loss = F.cross_entropy(
        ce_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=cfg.label_smoothing,
    )

    if cfg.kd_weight > 0:
        if teacher_logits is None:
            raise ValueError("cfg.kd_weight > 0 but teacher_logits is None.")
        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                f"teacher_logits shape {tuple(teacher_logits.shape)} != "
                f"student_logits shape {tuple(student_logits.shape)}."
            )
        kd_loss = kd_divergence(student_logits, teacher_logits, cfg.temperature, mask=mask)
    else:
        kd_loss = torch.zeros((), dtype=sft_loss.dtype, device=sft_loss.device)

    total_loss = cfg.sft_weight * sft_loss + cfg.kd_weight * kd_loss

    components = {
        "sft_loss": sft_loss.detach().item(),
        "kd_loss": kd_loss.detach().item(),
        "total_loss": total_loss.detach().item(),
    }
    return total_loss, components


def sequence_level_distillation_loss(student_logits, teacher_generated_ids, labels):
    """Fallback path, not currently used (see module docstring) — plain CE
    of the student against the teacher's generated token sequence instead of
    its logits. Only needed if a future teacher/student pair fails the
    tokenizer check in scripts/verify_tokenizer_compatibility.py."""
    raise NotImplementedError(
        "Not needed for the current model pair — tokenizer compatibility is "
        "verified (see scripts/verify_tokenizer_compatibility.py), so "
        "logit-level KD (combined_loss above) is the active path."
    )
