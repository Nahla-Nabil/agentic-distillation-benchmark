"""Unit tests for training/train.py's config parsing, per-step loss wiring,
and training-example formatting — the parts of the module that don't need
Unsloth/a GPU/real Qwen weights (see that module's docstring for the
testable/not-testable split). Uses CPU torch (already a dependency for
losses.py's tests) and tiny fake nn.Module "models" standing in for the
real student/teacher.
"""

import pytest
import torch
import yaml
from torch import nn

from adbench.training.losses import KDLossConfig
from adbench.training.train import (
    REPO_ROOT,
    TrainingConfig,
    _pad_batch,
    _set_by_dotted_path,
    compute_total_optimizer_steps,
    format_training_example,
    load_experiment_config,
    loss_log_path,
    resolve_checkpoint_dir,
    resolve_training_config,
    training_step,
    write_loss_log,
)

REAL_EXPERIMENT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment.yaml"


# --- load_experiment_config / _set_by_dotted_path ---

def test_load_experiment_config_reads_the_real_config():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    assert "training" in config
    assert "conditions" in config


def test_set_by_dotted_path_single_level():
    d = {"a": 1}
    _set_by_dotted_path(d, "a", 2)
    assert d["a"] == 2


def test_set_by_dotted_path_nested():
    d = {"kd": {"temperature": 2.0}}
    _set_by_dotted_path(d, "kd.temperature", 4.0)
    assert d["kd"]["temperature"] == 4.0


def test_set_by_dotted_path_unknown_key_raises():
    d = {"kd": {"temperature": 2.0}}
    with pytest.raises(KeyError):
        _set_by_dotted_path(d, "kd.nonexistent", 1.0)
    with pytest.raises(KeyError):
        _set_by_dotted_path(d, "nonexistent.temperature", 1.0)


# --- resolve_checkpoint_dir ---

@pytest.mark.parametrize("condition,expected_suffix", [
    ("base", "checkpoints/base"),
    ("sft_only", "checkpoints/sft_only"),
    ("distilled", "checkpoints/distilled"),
])
def test_resolve_checkpoint_dir_real_config(condition, expected_suffix):
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    path = resolve_checkpoint_dir(config, condition)
    assert path == REPO_ROOT / expected_suffix


def test_resolve_checkpoint_dir_unknown_condition_raises():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    with pytest.raises(ValueError):
        resolve_checkpoint_dir(config, "nonexistent")


# --- resolve_training_config ---

def test_resolve_training_config_sft_only_forces_kd_weight_zero():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    # Sanity: the real config's training.kd.kd_weight is NOT 0 — proves the
    # forcing below is resolve_training_config's own doing, not a config
    # coincidence.
    assert config["training"]["kd"]["kd_weight"] != 0.0

    cfg = resolve_training_config(config, "sft_only")

    assert isinstance(cfg, TrainingConfig)
    assert cfg.kd.kd_weight == 0.0
    assert cfg.kd.sft_weight == 1.0


def test_resolve_training_config_distilled_uses_configured_kd_weights():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "distilled")

    assert cfg.kd.kd_weight == config["training"]["kd"]["kd_weight"]
    assert cfg.kd.sft_weight == config["training"]["kd"]["sft_weight"]
    assert cfg.kd.temperature == config["training"]["kd"]["temperature"]


def test_resolve_training_config_base_does_not_raise():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "base")
    assert cfg.condition == "base"


def test_resolve_training_config_shares_non_kd_hyperparameters():
    """sft_only and distilled must differ ONLY in the kd weighting."""
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    sft_cfg = resolve_training_config(config, "sft_only")
    distill_cfg = resolve_training_config(config, "distilled")

    for field in ("learning_rate", "num_train_epochs", "per_device_train_batch_size",
                  "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
                  "lr_scheduler_type", "logging_steps", "save_steps", "seed"):
        assert getattr(sft_cfg, field) == getattr(distill_cfg, field), field


def test_resolve_training_config_unknown_condition_raises():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    with pytest.raises(ValueError):
        resolve_training_config(config, "nonexistent")


def test_resolve_training_config_sweep_overrides_kd_temperature():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "distilled", sweep_name="higher_kd_temp")
    assert cfg.kd.temperature == 4.0


def test_resolve_training_config_sweep_overrides_two_keys():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "distilled", sweep_name="kd_heavy")
    assert cfg.kd.kd_weight == 0.8
    assert cfg.kd.sft_weight == 0.2


def test_resolve_training_config_sweep_overriding_non_kd_field():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "distilled", sweep_name="lower_lr")
    assert cfg.learning_rate == 5.0e-5


def test_resolve_training_config_sweep_does_not_affect_sft_only():
    """A sweep entry overriding kd.* must not leak into sft_only's forced
    kd_weight=0 — sft_only's KD weighting isn't swept, by design."""
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    cfg = resolve_training_config(config, "sft_only", sweep_name="kd_heavy")
    assert cfg.kd.kd_weight == 0.0
    assert cfg.kd.sft_weight == 1.0


def test_resolve_training_config_unknown_sweep_raises():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    with pytest.raises(ValueError):
        resolve_training_config(config, "distilled", sweep_name="nonexistent_sweep")


def test_resolve_training_config_baseline_sweep_is_a_no_op():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    plain = resolve_training_config(config, "distilled")
    swept = resolve_training_config(config, "distilled", sweep_name="baseline")
    assert plain == swept


def test_every_configured_sweep_name_resolves_without_error():
    config = load_experiment_config(REAL_EXPERIMENT_CONFIG_PATH)
    for entry in config["training"]["sweep"]:
        resolve_training_config(config, "distilled", sweep_name=entry["name"])


# --- training_step: fake models, real CPU torch autograd ---

class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeCausalLM(nn.Module):
    """A tiny real nn.Module — forward pass depends on actual parameters,
    so .backward() populates real .grad tensors we can assert on."""

    def __init__(self, vocab_size=11, hidden=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        return _FakeOutput(self.proj(self.embed(input_ids)))


class _FakeCausalLMRawTensor(_FakeCausalLM):
    """Some model wrappers return the logits tensor directly rather than a
    `.logits`-bearing object — training_step must accept both."""

    def forward(self, input_ids, attention_mask=None):
        return self.proj(self.embed(input_ids))


def _dummy_batch(vocab_size=11, batch=2, seq_len=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), generator=g)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, 0] = -100  # simulate one masked (prompt) position
    return input_ids, attention_mask, labels


def test_training_step_sft_only_needs_no_teacher():
    student = _FakeCausalLM()
    input_ids, attention_mask, labels = _dummy_batch()
    cfg = KDLossConfig(kd_weight=0.0, sft_weight=1.0)

    loss, components = training_step(student, None, input_ids, attention_mask, labels, cfg)
    loss.backward()

    assert components["kd_loss"] == 0.0
    assert student.embed.weight.grad is not None
    assert torch.isfinite(student.embed.weight.grad).all()


def test_training_step_distilled_updates_student_not_teacher():
    student = _FakeCausalLM()
    teacher = _FakeCausalLM()
    input_ids, attention_mask, labels = _dummy_batch()
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)

    loss, components = training_step(student, teacher, input_ids, attention_mask, labels, cfg)
    loss.backward()

    assert student.embed.weight.grad is not None
    assert torch.isfinite(student.embed.weight.grad).all()
    # teacher's forward ran under torch.no_grad() inside training_step, so
    # its parameters must never receive a gradient — the whole point of KD.
    assert teacher.embed.weight.grad is None
    assert teacher.proj.weight.grad is None
    assert components["kd_loss"] >= 0.0


def test_training_step_kd_weight_positive_without_teacher_raises():
    student = _FakeCausalLM()
    input_ids, attention_mask, labels = _dummy_batch()
    cfg = KDLossConfig(kd_weight=0.5, sft_weight=0.5)
    with pytest.raises(ValueError):
        training_step(student, None, input_ids, attention_mask, labels, cfg)


def test_training_step_accepts_raw_tensor_output_not_just_logits_attribute():
    student = _FakeCausalLMRawTensor()
    input_ids, attention_mask, labels = _dummy_batch()
    cfg = KDLossConfig(kd_weight=0.0, sft_weight=1.0)

    loss, _ = training_step(student, None, input_ids, attention_mask, labels, cfg)
    assert torch.isfinite(loss)


# --- compute_total_optimizer_steps ---
#
# Pulled out specifically because it's easy to get wrong by forgetting the
# gradient-accumulation division, which would silently misconfigure the LR
# scheduler for a real run (see the function's docstring) — this is exactly
# the kind of orchestration arithmetic that's cheap to verify locally even
# though the training loop it feeds isn't runnable without a GPU.

def test_compute_total_optimizer_steps_accounts_for_grad_accumulation():
    # 100 examples, batch_size=2 -> 50 batches/epoch; accum=4 -> 12 optimizer
    # steps/epoch (50 // 4 = 12); 3 epochs -> 36.
    assert compute_total_optimizer_steps(
        n_examples=100, batch_size=2, gradient_accumulation_steps=4, num_train_epochs=3
    ) == 36


def test_compute_total_optimizer_steps_no_accumulation_equals_batch_count():
    assert compute_total_optimizer_steps(
        n_examples=100, batch_size=10, gradient_accumulation_steps=1, num_train_epochs=1
    ) == 10


def test_compute_total_optimizer_steps_fractional_epochs():
    # 10 batches/epoch (no accumulation), 0.5 epochs -> 5 steps.
    assert compute_total_optimizer_steps(
        n_examples=100, batch_size=10, gradient_accumulation_steps=1, num_train_epochs=0.5
    ) == 5


def test_compute_total_optimizer_steps_never_returns_zero():
    """A tiny dataset (fewer examples than one batch, or fewer batches than
    one accumulation window) must still yield at least 1 step, not 0 (which
    would make transformers.get_scheduler divide by zero)."""
    assert compute_total_optimizer_steps(
        n_examples=1, batch_size=8, gradient_accumulation_steps=4, num_train_epochs=1
    ) >= 1
    assert compute_total_optimizer_steps(
        n_examples=100, batch_size=8, gradient_accumulation_steps=100, num_train_epochs=1
    ) >= 1


# --- _pad_batch ---

def test_pad_batch_pads_to_longest_sequence():
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    batch = _pad_batch(examples, pad_token_id=0)

    assert batch["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"] == [[-100, 2, 3], [-100, 5, -100]]


def test_pad_batch_no_padding_needed_when_equal_length():
    examples = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
        {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [3, 4]},
    ]
    batch = _pad_batch(examples, pad_token_id=99)
    assert batch["input_ids"] == [[1, 2], [3, 4]]


# --- format_training_example: stub tokenizer (no network) ---

class _StubTokenizer:
    """Deterministic word-level pseudo-tokenizer with a persistent vocab, so
    a prompt's tokenization is a genuine prefix of the full example's
    tokenization — the same structural property a real chat template has
    (appending a turn adds tokens, it doesn't rewrite earlier ones)."""

    def __init__(self):
        self.pad_token_id = 0
        self._vocab: dict[str, int] = {}

    def _token_id(self, word: str) -> int:
        return self._vocab.setdefault(word, len(self._vocab) + 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages]
        text = " ".join(parts)
        if add_generation_prompt:
            text += " <assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._token_id(w) for w in text.split()]}


def _sample_record():
    return {
        "task_id": "glaive-1",
        "chain_length": 1,
        "user_goal": "What is my BMI at 70kg and 1.75m?",
        "expected_tool_sequence": ["calculate_bmi"],
        "injected_error_at_step": None,
        "arguments": {"height": 1.75, "weight": 70},
    }


def test_format_training_example_prompt_is_a_prefix_of_full():
    from adbench.harness.tools import build_glaive_registry
    tokenizer = _StubTokenizer()
    example = format_training_example(_sample_record(), tokenizer, build_glaive_registry())

    assert example["input_ids"]
    assert len(example["labels"]) == len(example["input_ids"])
    assert len(example["attention_mask"]) == len(example["input_ids"])


def test_format_training_example_masks_exactly_the_prompt_prefix():
    from adbench.harness.tools import build_glaive_registry
    tokenizer = _StubTokenizer()
    example = format_training_example(_sample_record(), tokenizer, build_glaive_registry())

    # find where masking ends
    first_unmasked = next(i for i, v in enumerate(example["labels"]) if v != -100)
    # everything before first_unmasked is masked, everything from there on
    # matches input_ids exactly (labels = input_ids on the completion span)
    assert all(v == -100 for v in example["labels"][:first_unmasked])
    assert example["labels"][first_unmasked:] == example["input_ids"][first_unmasked:]
    assert first_unmasked > 0  # the prompt is genuinely non-empty


def test_format_training_example_completion_contains_tool_call():
    from adbench.harness.tools import build_glaive_registry
    tokenizer = _StubTokenizer()
    record = _sample_record()
    format_training_example(record, tokenizer, build_glaive_registry())

    # Confirm the tool_call completion's distinctive content actually
    # reached the tokenizer (i.e. shows up among the "words" it saw).
    assert any("tool_call" in w for w in tokenizer._vocab)
    assert any("calculate_bmi" in w for w in tokenizer._vocab)
    assert any("arguments" in w for w in tokenizer._vocab)


def test_format_training_example_attention_mask_is_all_ones():
    from adbench.harness.tools import build_glaive_registry
    tokenizer = _StubTokenizer()
    example = format_training_example(_sample_record(), tokenizer, build_glaive_registry())
    assert set(example["attention_mask"]) == {1}


def test_format_training_example_different_tools_use_only_their_own_tool_in_system_prompt():
    """The system prompt should describe just this record's tool, not the
    whole 8-tool registry — matching the real dataset's per-example system
    prompts (see the function's docstring)."""
    from adbench.harness.tools import build_glaive_registry

    class _RecordingTokenizer(_StubTokenizer):
        def __init__(self):
            super().__init__()
            self.last_prompt_text = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            if add_generation_prompt:
                self.last_prompt_text = text
            return text

    tokenizer = _RecordingTokenizer()
    format_training_example(_sample_record(), tokenizer, build_glaive_registry())

    assert "calculate_bmi" in tokenizer.last_prompt_text
    # none of the other 7 glaive tools should be described
    for other_tool in ("convert_currency", "get_stock_price", "calculate_discount",
                       "calculate_tip", "calculate_age", "generate_random_number",
                       "calculate_distance"):
        assert other_tool not in tokenizer.last_prompt_text


# --- format_training_example: real Qwen tokenizer (network, no GPU) ---
#
# The stub-tokenizer tests above verify format_training_example's masking
# LOGIC in isolation; this confirms that logic's core assumption — that a
# chat template appends new turns rather than rewriting earlier tokens, so
# the prompt's tokenization really is a prefix of the full example's — also
# holds for the REAL Qwen tokenizer/chat template, not just the stub's
# simplified one. Skipped (not failed) if the tokenizer can't be fetched —
# same reasoning as scripts/verify_tokenizer_compatibility.py: no GPU
# needed, but this does need network once (then it's cached).

@pytest.fixture(scope="module")
def real_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    except Exception as e:  # noqa: BLE001 — network/HF-hub failures raise many
        # different exception types (requests errors, OSError, HTTPError, ...);
        # any of them here means "skip this test", not "the test failed".
        pytest.skip(f"could not fetch the real Qwen tokenizer: {e}")


def test_format_training_example_real_tokenizer_prompt_is_a_true_prefix(real_qwen_tokenizer):
    from adbench.harness.tools import build_glaive_registry
    example = format_training_example(_sample_record(), real_qwen_tokenizer, build_glaive_registry())

    first_unmasked = next(i for i, v in enumerate(example["labels"]) if v != -100)
    assert first_unmasked > 0
    assert example["labels"][first_unmasked:] == example["input_ids"][first_unmasked:]
    assert all(v == -100 for v in example["labels"][:first_unmasked])


def test_format_training_example_real_tokenizer_decodes_to_valid_tool_call(real_qwen_tokenizer):
    """The unmasked (completion) span, decoded, should be exactly the
    <tool_call>...</tool_call> JSON block harness/executor.py's
    parse_model_output() expects — proving training data and the harness's
    own eval-time parser agree on format."""
    import json
    import re

    from adbench.harness.executor import parse_model_output
    from adbench.harness.tools import build_glaive_registry

    record = _sample_record()
    example = format_training_example(record, real_qwen_tokenizer, build_glaive_registry())

    first_unmasked = next(i for i, v in enumerate(example["labels"]) if v != -100)
    completion_text = real_qwen_tokenizer.decode(example["input_ids"][first_unmasked:])

    match = re.search(r"<tool_call>.*?</tool_call>", completion_text, re.DOTALL)
    assert match is not None, f"no <tool_call> block in decoded completion: {completion_text!r}"

    parsed = parse_model_output(match.group(0))
    assert parsed.name == record["expected_tool_sequence"][0]
    assert parsed.arguments == record["arguments"]
    # also confirm json.dumps round-trips exactly what we expect, independent
    # of parse_model_output's own leniency
    assert json.loads(match.group(0)[len("<tool_call>"):-len("</tool_call>")]) == {
        "name": record["expected_tool_sequence"][0], "arguments": record["arguments"],
    }


# --- loss log path / round-trip ---

def test_loss_log_path_without_sweep():
    path = loss_log_path("sft_only")
    assert path.name == "sft_only.jsonl"


def test_loss_log_path_with_sweep():
    path = loss_log_path("distilled", "higher_kd_temp")
    assert path.name == "distilled-higher_kd_temp.jsonl"


def test_write_loss_log_round_trip(tmp_path):
    from adbench.data.prepare import read_jsonl
    entries = [
        {"step": 1, "epoch": 0, "sft_loss": 1.2, "kd_loss": 0.5, "total_loss": 1.7, "learning_rate": 2e-4},
        {"step": 2, "epoch": 0, "sft_loss": 1.1, "kd_loss": 0.4, "total_loss": 1.5, "learning_rate": 2e-4},
    ]
    path = tmp_path / "log.jsonl"
    write_loss_log(entries, path)
    assert read_jsonl(path) == entries


# --- config file sanity (the real configs/experiment.yaml is well-formed) ---

def test_real_experiment_config_training_block_has_required_keys():
    with open(REAL_EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {
        "seed", "learning_rate", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
        "lr_scheduler_type", "logging_steps", "save_steps", "kd", "sweep",
    }
    assert required <= set(config["training"])
    assert {"temperature", "kd_weight", "sft_weight"} <= set(config["training"]["kd"])
