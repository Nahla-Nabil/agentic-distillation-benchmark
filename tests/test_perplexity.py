"""Unit tests for evaluation/perplexity.py — CPU torch + tiny fake models,
no GPU, no real Qwen weights (same philosophy as tests/test_train.py)."""

import math

import pytest
import torch

from adbench.evaluation.perplexity import (
    compute_perplexity,
    general_eval_config,
    load_general_eval_texts,
)


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _UniformModel:
    """Always predicts a uniform distribution — cross-entropy of a uniform
    distribution over `vocab_size` classes is exactly log(vocab_size)
    regardless of which token is "correct", so perplexity should come out
    to exactly vocab_size (up to floating point)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        return _FakeOutput(torch.zeros(batch, seq_len, self.vocab_size))


class _PerfectPredictorModel:
    """Cheats by reading the actual next token straight out of input_ids and
    putting all its probability mass there — isolates whether
    compute_perplexity's formula is right, independent of any real model's
    quality. Returns a raw tensor (not a `.logits`-bearing object) to also
    cover that code path."""

    def __init__(self, vocab_size: int, confidence: float = 30.0):
        self.vocab_size = vocab_size
        self.confidence = confidence

    def __call__(self, input_ids):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -10.0)
        for b in range(batch):
            for t in range(seq_len - 1):
                logits[b, t, input_ids[b, t + 1].item()] = self.confidence
        return logits


class _WordTokenizer:
    """Deterministic word-level pseudo-tokenizer with a persistent vocab."""

    def __init__(self, vocab_size: int | None = None):
        self._vocab: dict[str, int] = {}
        self._fixed_vocab_size = vocab_size

    def __call__(self, text):
        return {"input_ids": [self._vocab.setdefault(w, len(self._vocab)) for w in text.split()]}

    @property
    def vocab_size(self) -> int:
        return self._fixed_vocab_size if self._fixed_vocab_size is not None else len(self._vocab)


# --- compute_perplexity: correctness of the formula ---

def test_compute_perplexity_uniform_model_equals_vocab_size():
    tokenizer = _WordTokenizer()
    texts = ["the quick brown fox jumps over the lazy dog", "a second sentence with different words entirely"]
    # tokenize once first so we know the vocab size the model should match
    for t in texts:
        tokenizer(t)
    model = _UniformModel(vocab_size=tokenizer.vocab_size)

    ppl = compute_perplexity(model, tokenizer, texts)

    assert ppl == pytest.approx(tokenizer.vocab_size, rel=1e-4)


def test_compute_perplexity_confident_correct_model_is_near_one():
    tokenizer = _WordTokenizer()
    texts = ["the quick brown fox jumps over the lazy dog"]
    for t in texts:
        tokenizer(t)
    model = _PerfectPredictorModel(vocab_size=max(tokenizer.vocab_size, 20), confidence=30.0)

    ppl = compute_perplexity(model, tokenizer, texts)

    assert ppl == pytest.approx(1.0, abs=1e-3)


def test_compute_perplexity_worse_model_scores_higher_than_better_model():
    tokenizer = _WordTokenizer()
    texts = ["the quick brown fox jumps over the lazy dog and then keeps running"]
    for t in texts:
        tokenizer(t)
    vocab_size = max(tokenizer.vocab_size, 20)

    confident_model = _PerfectPredictorModel(vocab_size, confidence=30.0)
    uniform_model = _UniformModel(vocab_size)

    ppl_confident = compute_perplexity(confident_model, tokenizer, texts)
    ppl_uniform = compute_perplexity(uniform_model, tokenizer, texts)

    assert ppl_confident < ppl_uniform


# --- edge cases ---

def test_compute_perplexity_skips_texts_too_short_to_score():
    tokenizer = _WordTokenizer()
    texts = ["onlyoneword", "this sentence has more than one word in it definitely"]
    for t in texts:
        tokenizer(t)
    model = _UniformModel(vocab_size=tokenizer.vocab_size)

    # should not raise — the 1-token text is silently skipped, not counted
    ppl = compute_perplexity(model, tokenizer, texts)
    assert math.isfinite(ppl)


def test_compute_perplexity_all_texts_too_short_raises():
    tokenizer = _WordTokenizer()
    model = _UniformModel(vocab_size=10)
    with pytest.raises(ValueError):
        compute_perplexity(model, tokenizer, ["oneword", ""])


def test_compute_perplexity_empty_text_list_raises():
    tokenizer = _WordTokenizer()
    model = _UniformModel(vocab_size=10)
    with pytest.raises(ValueError):
        compute_perplexity(model, tokenizer, [])


def test_compute_perplexity_truncates_to_max_length():
    """A text longer than max_length shouldn't crash or silently include
    tokens beyond the cap — just checking it runs and returns something
    sane; the truncation itself is exercised by not raising on a long input."""
    tokenizer = _WordTokenizer()
    long_text = " ".join(f"word{i}" for i in range(1000))
    tokenizer(long_text)
    model = _UniformModel(vocab_size=tokenizer.vocab_size)

    ppl = compute_perplexity(model, tokenizer, [long_text], max_length=50)
    assert math.isfinite(ppl)


def test_compute_perplexity_accepts_object_with_logits_attribute():
    """_UniformModel returns a `.logits`-bearing object; confirm that path
    (as opposed to a raw tensor, covered by _PerfectPredictorModel above)
    works too."""
    tokenizer = _WordTokenizer()
    texts = ["a short test sentence with several words"]
    for t in texts:
        tokenizer(t)
    model = _UniformModel(vocab_size=tokenizer.vocab_size)
    assert math.isfinite(compute_perplexity(model, tokenizer, texts))


# --- load_general_eval_texts ---

def test_load_general_eval_texts_round_trip(tmp_path):
    from adbench.data.prepare import write_jsonl
    records = [
        {"text": "first chunk of text", "source_index": 0, "n_words": 4},
        {"text": "second chunk of text", "source_index": 1, "n_words": 4},
    ]
    path = tmp_path / "sample.jsonl"
    write_jsonl(records, path)

    texts = load_general_eval_texts(path)
    assert texts == ["first chunk of text", "second chunk of text"]


# --- general_eval_config ---

def test_general_eval_config_extracts_the_block():
    experiment_config = {
        "general_lm_eval": {"metric": "perplexity", "n_samples": 200},
        "training": {},
    }
    assert general_eval_config(experiment_config) == {"metric": "perplexity", "n_samples": 200}


def test_general_eval_config_matches_the_real_config():
    from adbench.training.train import REPO_ROOT, load_experiment_config
    config = load_experiment_config(REPO_ROOT / "configs" / "experiment.yaml")
    block = general_eval_config(config)
    assert block["metric"] == "perplexity"
    assert "eval_set" in block
