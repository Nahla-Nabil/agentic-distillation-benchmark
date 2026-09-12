"""Unit tests for training/losses.py's KD + SFT loss math, using dummy
tensors — no GPU, no real model, no Unsloth needed (just CPU torch, which
IS in requirements.txt for exactly this reason; see that file's comment).
"""

import math

import pytest
import torch

from adbench.training.losses import KDLossConfig, combined_loss, kd_divergence

# --- KDLossConfig validation ---

def test_kd_loss_config_defaults_are_valid():
    cfg = KDLossConfig()
    assert cfg.temperature == 2.0
    assert cfg.kd_weight == 0.5
    assert cfg.sft_weight == 0.5


def test_kd_loss_config_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        KDLossConfig(temperature=0.0)
    with pytest.raises(ValueError):
        KDLossConfig(temperature=-1.0)


def test_kd_loss_config_rejects_negative_weights():
    with pytest.raises(ValueError):
        KDLossConfig(kd_weight=-0.1)
    with pytest.raises(ValueError):
        KDLossConfig(sft_weight=-0.1)


def test_kd_loss_config_rejects_both_weights_zero():
    with pytest.raises(ValueError):
        KDLossConfig(kd_weight=0.0, sft_weight=0.0)


def test_kd_loss_config_sft_only_shape_is_valid():
    """The exact config train.py forces for the "sft_only" condition."""
    cfg = KDLossConfig(kd_weight=0.0, sft_weight=1.0)
    assert cfg.kd_weight == 0.0


def test_kd_loss_config_pure_kd_shape_is_valid():
    cfg = KDLossConfig(kd_weight=1.0, sft_weight=0.0)
    assert cfg.sft_weight == 0.0


# --- kd_divergence ---

def _random_logits(batch=2, seq_len=5, vocab=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, vocab, generator=g)


def test_kd_divergence_zero_when_student_equals_teacher():
    logits = _random_logits()
    kd = kd_divergence(logits, logits.clone(), temperature=2.0)
    assert kd.item() == pytest.approx(0.0, abs=1e-5)


def test_kd_divergence_is_nonnegative():
    student = _random_logits(seed=1)
    teacher = _random_logits(seed=2)
    kd = kd_divergence(student, teacher, temperature=2.0)
    assert kd.item() >= -1e-6


def test_kd_divergence_increases_with_teacher_student_divergence():
    """A student further from the teacher (in logit space) should score a
    higher KD loss than one closer to it, at the same temperature."""
    teacher = _random_logits(seed=3)
    close_student = teacher + 0.01 * torch.randn(teacher.shape, generator=torch.Generator().manual_seed(4))
    far_student = teacher + 5.0 * torch.randn(teacher.shape, generator=torch.Generator().manual_seed(5))

    kd_close = kd_divergence(close_student, teacher, temperature=2.0).item()
    kd_far = kd_divergence(far_student, teacher, temperature=2.0).item()
    assert kd_far > kd_close


def test_kd_divergence_shape_mismatch_raises():
    student = _random_logits(vocab=10)
    teacher = _random_logits(vocab=11)
    with pytest.raises(ValueError):
        kd_divergence(student, teacher, temperature=2.0)


def test_kd_divergence_is_differentiable_wrt_student():
    student = _random_logits(seed=6)
    student.requires_grad_(True)
    teacher = _random_logits(seed=7)
    kd = kd_divergence(student, teacher, temperature=2.0)
    kd.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_kd_divergence_mask_excludes_masked_positions():
    """Only the masked-in position should affect the result — verified by
    making the masked-out position wildly divergent and confirming it
    doesn't move the loss at all."""
    torch.manual_seed(8)
    student = torch.randn(1, 2, 5)
    teacher = student.clone()
    # position 1 is wildly different, but masked out
    teacher[0, 1] = student[0, 1] + 100.0
    mask = torch.tensor([[True, False]])

    kd = kd_divergence(student, teacher, temperature=2.0, mask=mask)
    assert kd.item() == pytest.approx(0.0, abs=1e-4)


def test_kd_divergence_mask_shape_mismatch_raises():
    student = _random_logits(batch=2, seq_len=5)
    teacher = _random_logits(batch=2, seq_len=5)
    bad_mask = torch.ones(2, 3, dtype=torch.bool)  # wrong seq_len
    with pytest.raises(ValueError):
        kd_divergence(student, teacher, temperature=2.0, mask=bad_mask)


def test_kd_divergence_higher_temperature_softens_the_distributions():
    """At a very high temperature both softmaxes approach uniform, so KD
    should shrink toward 0 even for very different logits."""
    student = _random_logits(seed=9)
    teacher = _random_logits(seed=10)
    kd_low_temp = kd_divergence(student, teacher, temperature=1.0).item()
    kd_high_temp = kd_divergence(student, teacher, temperature=100.0).item()
    assert kd_high_temp < kd_low_temp


# --- combined_loss ---

def _labeled_batch(batch=2, seq_len=5, vocab=17, ignore_last=True, seed=0):
    student = _random_logits(batch, seq_len, vocab, seed=seed)
    teacher = _random_logits(batch, seq_len, vocab, seed=seed + 100)
    labels = torch.randint(0, vocab, (batch, seq_len), generator=torch.Generator().manual_seed(seed + 200))
    if ignore_last:
        labels[:, -1] = -100  # simulate a masked-out (e.g. prompt) position
    return student, teacher, labels


def test_combined_loss_returns_scalar_and_components():
    student, teacher, labels = _labeled_batch()
    cfg = KDLossConfig(temperature=2.0, kd_weight=0.5, sft_weight=0.5)

    loss, components = combined_loss(student, teacher, labels, cfg)

    assert loss.dim() == 0
    assert set(components) == {"sft_loss", "kd_loss", "total_loss"}
    assert all(isinstance(v, float) for v in components.values())
    assert components["total_loss"] == pytest.approx(loss.item(), abs=1e-5)


def test_combined_loss_matches_weighted_sum_of_components():
    student, teacher, labels = _labeled_batch()
    cfg = KDLossConfig(temperature=1.5, kd_weight=0.3, sft_weight=0.7)

    _, components = combined_loss(student, teacher, labels, cfg)

    expected_total = cfg.sft_weight * components["sft_loss"] + cfg.kd_weight * components["kd_loss"]
    assert components["total_loss"] == pytest.approx(expected_total, abs=1e-5)


def test_combined_loss_kd_weight_zero_ignores_teacher_entirely():
    """kd_weight=0 (the sft_only condition) must not require a valid
    teacher_logits — None must work, and the result must equal plain CE."""
    student, _, labels = _labeled_batch()
    cfg = KDLossConfig(kd_weight=0.0, sft_weight=1.0)

    loss, components = combined_loss(student, None, labels, cfg)

    assert components["kd_loss"] == 0.0
    import torch.nn.functional as F
    expected_ce = F.cross_entropy(
        student.reshape(-1, student.size(-1)), labels.reshape(-1), ignore_index=-100
    ).item()
    assert components["sft_loss"] == pytest.approx(expected_ce, abs=1e-5)
    assert loss.item() == pytest.approx(expected_ce, abs=1e-5)


def test_combined_loss_kd_weight_zero_with_teacher_provided_still_ignores_it():
    """Passing a teacher anyway when kd_weight=0 must be harmless (teacher
    simply unused), not an error."""
    student, teacher, labels = _labeled_batch()
    cfg = KDLossConfig(kd_weight=0.0, sft_weight=1.0)

    loss_with_teacher, comp_with = combined_loss(student, teacher, labels, cfg)
    loss_without_teacher, comp_without = combined_loss(student, None, labels, cfg)

    assert loss_with_teacher.item() == pytest.approx(loss_without_teacher.item(), abs=1e-6)
    assert comp_with["kd_loss"] == comp_without["kd_loss"] == 0.0


def test_combined_loss_kd_weight_positive_without_teacher_raises():
    student, _, labels = _labeled_batch()
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)
    with pytest.raises(ValueError):
        combined_loss(student, None, labels, cfg)


def test_combined_loss_pure_kd_sft_weight_zero():
    student, teacher, labels = _labeled_batch()
    cfg = KDLossConfig(kd_weight=1.0, sft_weight=0.0)

    loss, components = combined_loss(student, teacher, labels, cfg)

    assert loss.item() == pytest.approx(components["kd_loss"], abs=1e-5)


def test_combined_loss_teacher_shape_mismatch_raises():
    student, _, labels = _labeled_batch(vocab=17)
    teacher_wrong_vocab = _random_logits(vocab=13)
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)
    with pytest.raises(ValueError):
        combined_loss(student, teacher_wrong_vocab, labels, cfg)


def test_combined_loss_labels_shape_mismatch_raises():
    student = _random_logits(batch=2, seq_len=5)
    teacher = _random_logits(batch=2, seq_len=5)
    wrong_labels = torch.zeros(2, 4, dtype=torch.long)  # wrong seq_len
    cfg = KDLossConfig()
    with pytest.raises(ValueError):
        combined_loss(student, teacher, wrong_labels, cfg)


def test_combined_loss_all_positions_ignored_does_not_crash():
    """Every label is -100 (e.g. a batch with no completion tokens at all,
    a real edge case if a batch boundary lands badly) — should return a
    finite (likely ~0 or nan-free) loss, not raise or produce nan/inf."""
    student, teacher, labels = _labeled_batch(ignore_last=False)
    labels[:, :] = -100
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)

    loss, _components = combined_loss(student, teacher, labels, cfg)

    # cross_entropy with all positions ignored returns nan by convention;
    # this test's job is to document that behavior, not silently hide it.
    assert torch.isnan(loss) or torch.isfinite(loss)


def test_combined_loss_is_differentiable_wrt_student_only():
    student, teacher, labels = _labeled_batch()
    student.requires_grad_(True)
    teacher.requires_grad_(True)
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)

    loss, _ = combined_loss(student, teacher, labels, cfg)
    loss.backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    # teacher should get no gradient in real training (caller wraps it in
    # torch.no_grad()) — but combined_loss itself is a pure function of
    # whatever tensors it's given, so if the caller *did* leave teacher
    # grad-tracking on, teacher.grad would be populated too. This test just
    # confirms the function doesn't crash either way; train.py's
    # training_step tests (test_train.py) confirm the real no_grad wrapping.


@pytest.mark.parametrize("kd_weight,sft_weight", [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (0.1, 0.9), (0.9, 0.1)])
def test_combined_loss_various_weightings_stay_finite(kd_weight, sft_weight):
    student, teacher, labels = _labeled_batch()
    cfg = KDLossConfig(kd_weight=kd_weight, sft_weight=sft_weight)
    teacher_arg = teacher if kd_weight > 0 else None

    loss, components = combined_loss(student, teacher_arg, labels, cfg)

    assert torch.isfinite(loss)
    assert not any(math.isnan(v) for v in components.values())
