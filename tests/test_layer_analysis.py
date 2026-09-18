"""Unit tests for analysis/layer_analysis.py.

Divergence metrics (cka, cosine_distance_rsa) are pure numpy math, tested
with synthetic tensors where the expected answer is known analytically —
no GPU, no model. find_decoder_layers/extract_activations are tested
against a tiny fake nn.Module standing in for a real HF model (same
philosophy as tests/test_train.py's _FakeCausalLM), including one wrapped
in an extra layer of nesting to mimic PEFT's attribute structure.
"""

import numpy as np
import pytest
import torch
from torch import nn

from adbench.analysis.layer_analysis import (
    DIVERGENCE_METRICS,
    align_layers,
    cka,
    compare_cached_activations,
    compare_layers,
    cosine_distance_rsa,
    count_layers,
    divergence,
    extract_activations,
    extract_and_cache_activations,
    find_decoder_layers,
    load_activation_cache,
    load_general_probe_texts,
    load_tool_use_probe_texts,
    read_layer_analysis_results,
    save_activation_cache,
    write_layer_analysis_results,
)


def _rng(seed):
    return np.random.default_rng(seed)


def _random_orthogonal(n, seed):
    """A random n x n orthogonal matrix, via QR decomposition."""
    a = _rng(seed).normal(size=(n, n))
    q, _ = np.linalg.qr(a)
    return q


# --- cka ---

def test_cka_identical_matrices_is_one():
    X = _rng(0).normal(size=(20, 10))
    assert cka(X, X) == pytest.approx(1.0, abs=1e-8)


def test_cka_rotation_invariant_same_width():
    X = _rng(1).normal(size=(20, 10))
    Q = _random_orthogonal(10, seed=2)
    assert cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)


def test_cka_scale_invariant():
    X = _rng(3).normal(size=(20, 10))
    assert cka(X, 5.0 * X) == pytest.approx(1.0, abs=1e-6)


def test_cka_negation_invariant():
    X = _rng(4).normal(size=(20, 10))
    assert cka(X, -X) == pytest.approx(1.0, abs=1e-6)


def test_cka_independent_random_matrices_is_low():
    X = _rng(5).normal(size=(200, 50))
    Y = _rng(6).normal(size=(200, 50))
    assert cka(X, Y) < 0.3


def test_cka_invariant_to_independent_rotation_of_each_side_different_widths():
    """The property that actually matters for comparing 5120-dim vs
    2560-dim activations: rotating EITHER side within its own space leaves
    CKA unchanged, even when the two widths differ."""
    X = _rng(7).normal(size=(30, 50))
    Y = _rng(8).normal(size=(30, 20))
    Qx = _random_orthogonal(50, seed=9)
    Qy = _random_orthogonal(20, seed=10)

    assert cka(X, Y) == pytest.approx(cka(X @ Qx, Y @ Qy), abs=1e-6)


def test_cka_different_widths_does_not_raise():
    X = _rng(11).normal(size=(30, 50))
    Y = _rng(12).normal(size=(30, 20))
    result = cka(X, Y)
    assert 0.0 <= result <= 1.0 + 1e-6


def test_cka_zero_variance_input_returns_zero_not_nan():
    X = np.ones((10, 5))  # every row identical -> zero variance
    Y = _rng(13).normal(size=(10, 5))
    assert cka(X, Y) == 0.0


def test_cka_sample_count_mismatch_raises():
    X = _rng(14).normal(size=(10, 5))
    Y = _rng(15).normal(size=(12, 5))
    with pytest.raises(ValueError):
        cka(X, Y)


# --- cosine_distance_rsa ---

def test_cosine_distance_identical_matrices_is_zero():
    X = _rng(20).normal(size=(20, 10))
    assert cosine_distance_rsa(X, X) == pytest.approx(0.0, abs=1e-8)


def test_cosine_distance_rotation_invariant_same_width():
    X = _rng(21).normal(size=(20, 10))
    Q = _random_orthogonal(10, seed=22)
    assert cosine_distance_rsa(X, X @ Q) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_negation_invariant():
    X = _rng(23).normal(size=(20, 10))
    assert cosine_distance_rsa(X, -X) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_invariant_to_independent_rotation_different_widths():
    X = _rng(24).normal(size=(30, 50))
    Y = _rng(25).normal(size=(30, 20))
    Qx = _random_orthogonal(50, seed=26)
    Qy = _random_orthogonal(20, seed=27)

    assert cosine_distance_rsa(X, Y) == pytest.approx(
        cosine_distance_rsa(X @ Qx, Y @ Qy), abs=1e-6
    )


def test_cosine_distance_independent_random_is_higher_than_identical():
    X = _rng(28).normal(size=(100, 30))
    Y = _rng(29).normal(size=(100, 30))
    assert cosine_distance_rsa(X, Y) > cosine_distance_rsa(X, X)


def test_cosine_distance_too_few_samples_raises():
    X = _rng(30).normal(size=(1, 5))
    Y = _rng(31).normal(size=(1, 5))
    with pytest.raises(ValueError):
        cosine_distance_rsa(X, Y)


def test_cosine_distance_sample_count_mismatch_raises():
    X = _rng(32).normal(size=(10, 5))
    Y = _rng(33).normal(size=(12, 5))
    with pytest.raises(ValueError):
        cosine_distance_rsa(X, Y)


def test_cosine_distance_zero_vector_row_does_not_crash():
    X = _rng(34).normal(size=(10, 5))
    X[0] = 0.0  # a degenerate all-zero row
    Y = _rng(35).normal(size=(10, 5))
    result = cosine_distance_rsa(X, Y)
    assert np.isfinite(result)


# --- divergence() dispatcher ---

def test_divergence_dispatches_to_cka():
    X = _rng(40).normal(size=(20, 10))
    assert divergence(X, X, metric="cka") == pytest.approx(cka(X, X))


def test_divergence_dispatches_to_cosine_distance():
    X = _rng(41).normal(size=(20, 10))
    assert divergence(X, X, metric="cosine_distance") == pytest.approx(cosine_distance_rsa(X, X))


def test_divergence_unknown_metric_raises():
    X = _rng(42).normal(size=(20, 10))
    with pytest.raises(ValueError):
        divergence(X, X, metric="nonsense")


def test_divergence_metrics_matches_configured_names():
    """configs/experiment.yaml:layer_analysis.divergence_metrics names must
    exactly match this module's registry, or run_eval-style config-driven
    dispatch would silently do nothing for a misspelled metric."""
    import yaml

    from adbench.training.train import REPO_ROOT
    with open(REPO_ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert set(config["layer_analysis"]["divergence_metrics"]) == set(DIVERGENCE_METRICS)


# --- align_layers ---

def test_align_layers_real_teacher_student_depths():
    mapping = align_layers(36, 40)
    assert mapping[0] == 0
    assert mapping[35] == 39  # the actual last-layer alignment this project needs
    assert len(mapping) == 36


def test_align_layers_monotonic_nondecreasing():
    mapping = align_layers(36, 40)
    values = [mapping[i] for i in range(36)]
    assert values == sorted(values)


def test_align_layers_equal_depths_is_identity():
    mapping = align_layers(36, 36)
    assert mapping == {i: i for i in range(36)}


def test_align_layers_all_targets_in_range():
    mapping = align_layers(36, 40)
    assert all(0 <= v < 40 for v in mapping.values())


def test_align_layers_single_source_layer():
    assert align_layers(1, 5) == {0: 0}


def test_align_layers_nonpositive_raises():
    with pytest.raises(ValueError):
        align_layers(0, 5)
    with pytest.raises(ValueError):
        align_layers(5, 0)


# --- find_decoder_layers / extract_activations: fake model ---

class _FakeAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class _FakeMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.self_attn = _FakeAttn(hidden)
        self.mlp = _FakeMLP(hidden, inter)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class _FakeInnerModel(nn.Module):
    def __init__(self, hidden, inter, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer(hidden, inter) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _FakeCausalLM(nn.Module):
    """Mirrors Qwen3ForCausalLM's `.model.layers` structure."""

    def __init__(self, vocab_size=20, hidden=8, inter=16, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.model = _FakeInnerModel(hidden, inter, n_layers)

    def forward(self, input_ids, attention_mask=None):
        return self.model(self.embed(input_ids))


class _PeftLikeWrapper(nn.Module):
    """Mimics PEFT's extra `base_model.model` nesting around the real
    model, to confirm find_decoder_layers() doesn't depend on a fixed
    attribute depth."""

    def __init__(self, inner):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = inner

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


class _WordTokenizer:
    def __init__(self, vocab_size):
        self._vocab: dict[str, int] = {}
        self._vocab_size = vocab_size

    def __call__(self, text):
        ids = [self._vocab.setdefault(w, len(self._vocab) % self._vocab_size) for w in text.split()]
        return {"input_ids": ids}


def test_find_decoder_layers_bare_model():
    model = _FakeCausalLM(n_layers=4)
    found = find_decoder_layers(model)
    assert [idx for idx, _ in found] == [0, 1, 2, 3]
    assert all(isinstance(layer, _FakeDecoderLayer) for _, layer in found)


def test_find_decoder_layers_peft_wrapped():
    model = _PeftLikeWrapper(_FakeCausalLM(n_layers=4))
    found = find_decoder_layers(model)
    assert [idx for idx, _ in found] == [0, 1, 2, 3]


def test_find_decoder_layers_no_match_raises():
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        find_decoder_layers(model)


def test_extract_activations_shapes():
    torch.manual_seed(0)
    model = _FakeCausalLM(vocab_size=20, hidden=8, n_layers=4)
    tokenizer = _WordTokenizer(vocab_size=20)
    texts = ["the quick brown fox", "a sentence", "one more probe text here"]

    result = extract_activations(model, tokenizer, texts, max_length=16)

    assert set(result) == {0, 1, 2, 3}
    for layer_result in result.values():
        assert set(layer_result) == {"attention", "ffn"}
        assert layer_result["attention"].shape == (3, 8)
        assert layer_result["ffn"].shape == (3, 8)


def test_extract_activations_layer0_attention_matches_manual_computation():
    torch.manual_seed(1)
    model = _FakeCausalLM(vocab_size=20, hidden=8, n_layers=2)
    tokenizer = _WordTokenizer(vocab_size=20)
    text = "hello"

    result = extract_activations(model, tokenizer, [text], max_length=16)

    input_ids = torch.tensor([tokenizer(text)["input_ids"]])
    with torch.no_grad():
        expected = model.model.layers[0].self_attn.o_proj(model.embed(input_ids)).mean(dim=1)
    np.testing.assert_allclose(result[0]["attention"][0], expected.numpy().reshape(-1), atol=1e-6)


def test_extract_activations_subset_of_layers():
    torch.manual_seed(2)
    model = _FakeCausalLM(n_layers=4)
    tokenizer = _WordTokenizer(vocab_size=20)

    result = extract_activations(model, tokenizer, ["a b c"], layer_indices=[1, 3], max_length=16)

    assert set(result) == {1, 3}


def test_extract_activations_unknown_layer_index_raises():
    model = _FakeCausalLM(n_layers=4)
    tokenizer = _WordTokenizer(vocab_size=20)
    with pytest.raises(ValueError):
        extract_activations(model, tokenizer, ["a"], layer_indices=[99], max_length=16)


def test_extract_activations_hooks_are_removed_after():
    """Calling extract_activations twice shouldn't accumulate duplicate
    hooks (which would double-count rows on the second call)."""
    torch.manual_seed(3)
    model = _FakeCausalLM(n_layers=2)
    tokenizer = _WordTokenizer(vocab_size=20)

    extract_activations(model, tokenizer, ["a", "b"], max_length=16)
    result2 = extract_activations(model, tokenizer, ["a", "b", "c"], max_length=16)

    assert result2[0]["attention"].shape == (3, 8)  # not (6, 8) from leaked hooks


def test_extract_activations_different_texts_vary():
    torch.manual_seed(4)
    model = _FakeCausalLM(n_layers=2)
    tokenizer = _WordTokenizer(vocab_size=20)

    result = extract_activations(model, tokenizer, ["completely different words here", "other"], max_length=16)

    assert not np.allclose(result[0]["attention"][0], result[0]["attention"][1])


# --- compare_layers ---

def _acts(n_layers, n_samples, hidden):
    rng = _rng(50)
    return {
        i: {"attention": rng.normal(size=(n_samples, hidden)), "ffn": rng.normal(size=(n_samples, hidden))}
        for i in range(n_layers)
    }


def test_compare_layers_row_shape_and_count():
    source = _acts(2, 10, 8)
    target = _acts(2, 10, 8)
    rows = compare_layers(source, target, {0: 0, 1: 1})

    assert len(rows) == 4  # 2 layers x 2 streams
    for row in rows:
        assert {"source_layer", "target_layer", "stream", "cka", "cosine_distance"} <= set(row)


def test_compare_layers_tags_input_set():
    source = _acts(1, 10, 8)
    target = _acts(1, 10, 8)
    rows = compare_layers(source, target, {0: 0}, input_set="tool_use")
    assert all(r["input_set"] == "tool_use" for r in rows)


def test_compare_layers_identical_activations_score_perfectly():
    acts = _acts(1, 10, 8)
    rows = compare_layers(acts, acts, {0: 0})
    for row in rows:
        assert row["cka"] == pytest.approx(1.0, abs=1e-6)
        assert row["cosine_distance"] == pytest.approx(0.0, abs=1e-6)


def test_compare_layers_handles_mismatched_widths():
    source = _acts(1, 10, 50)   # e.g. student
    target = _acts(1, 10, 20)   # e.g. teacher, different hidden size
    rows = compare_layers(source, target, {0: 0})
    assert len(rows) == 2
    assert all(np.isfinite(r["cka"]) and np.isfinite(r["cosine_distance"]) for r in rows)


def test_compare_layers_skips_alignment_entries_missing_from_either_side():
    source = _acts(2, 10, 8)
    target = _acts(1, 10, 8)  # only layer 0 exists
    rows = compare_layers(source, target, {0: 0, 1: 1})  # target layer 1 doesn't exist
    assert len(rows) == 2  # only layer 0's pair produced rows


def test_compare_layers_respects_requested_metrics_subset():
    acts = _acts(1, 10, 8)
    rows = compare_layers(acts, acts, {0: 0}, metrics=("cka",))
    assert "cka" in rows[0]
    assert "cosine_distance" not in rows[0]


# --- I/O ---

def test_activation_cache_round_trip(tmp_path):
    acts = _acts(2, 5, 8)
    path = save_activation_cache(acts, tmp_path / "cache")  # no .npz suffix given
    assert path.suffix == ".npz"

    loaded = load_activation_cache(path)
    assert set(loaded) == {0, 1}
    for i in (0, 1):
        np.testing.assert_allclose(loaded[i]["attention"], acts[i]["attention"])
        np.testing.assert_allclose(loaded[i]["ffn"], acts[i]["ffn"])


def test_load_activation_cache_accepts_the_extensionless_path_used_to_save(tmp_path):
    acts = _acts(2, 5, 8)
    save_activation_cache(acts, tmp_path / "cache")
    loaded = load_activation_cache(tmp_path / "cache")  # same extension-less path, not the returned one
    assert set(loaded) == {0, 1}


def test_activation_cache_round_trip_with_explicit_npz_suffix(tmp_path):
    acts = _acts(1, 5, 8)
    path = save_activation_cache(acts, tmp_path / "cache.npz")
    assert path == tmp_path / "cache.npz"
    loaded = load_activation_cache(path)
    np.testing.assert_allclose(loaded[0]["ffn"], acts[0]["ffn"])


def test_compare_cached_activations_end_to_end(tmp_path):
    source_acts = _acts(1, 10, 8)
    target_acts = _acts(1, 10, 8)
    source_path = save_activation_cache(source_acts, tmp_path / "source")
    target_path = save_activation_cache(target_acts, tmp_path / "target")

    rows = compare_cached_activations(source_path, target_path, {0: 0}, input_set="general")

    assert len(rows) == 2
    assert all(r["input_set"] == "general" for r in rows)


def test_layer_analysis_results_jsonl_round_trip(tmp_path):
    rows = [
        {"source_layer": 0, "target_layer": 0, "stream": "attention", "cka": 0.9, "cosine_distance": 0.1},
        {"source_layer": 0, "target_layer": 0, "stream": "ffn", "cka": 0.8, "cosine_distance": 0.2},
    ]
    path = tmp_path / "results.jsonl"
    write_layer_analysis_results(rows, path)
    assert read_layer_analysis_results(path) == rows


# --- extract_and_cache_activations: fake model, still exercises the Colab-facing wrapper ---

def test_extract_and_cache_activations_writes_a_loadable_file(tmp_path):
    torch.manual_seed(5)
    model = _FakeCausalLM(n_layers=2)
    tokenizer = _WordTokenizer(vocab_size=20)

    path = extract_and_cache_activations(model, tokenizer, ["a b", "c d"], tmp_path / "cache", max_length=16)

    loaded = load_activation_cache(path)
    assert loaded[0]["attention"].shape == (2, 8)


# --- count_layers ---

def test_count_layers_matches_find_decoder_layers():
    model = _FakeCausalLM(n_layers=5)
    assert count_layers(model) == 5


# --- load_tool_use_probe_texts: fixture Glaive test split, not the real one ---

def _write_fixture_data_config(tmp_path, test_path):
    import yaml
    config = {
        "output": {"train_path": str(tmp_path / "train.jsonl"), "test_path": str(test_path)},
    }
    config_path = tmp_path / "data.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def _write_fixture_jsonl(path, records):
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(_json.dumps(r) + "\n" for r in records)


def test_load_tool_use_probe_texts_samples_user_goals(tmp_path):
    test_path = tmp_path / "test.jsonl"
    records = [
        {"task_id": f"glaive-{i}", "chain_length": 1, "user_goal": f"goal number {i}",
         "expected_tool_sequence": ["calculate_bmi"], "injected_error_at_step": None,
         "arguments": {}, "arguments_style": "raw_object", "source_index": i}
        for i in range(20)
    ]
    _write_fixture_jsonl(test_path, records)
    config_path = _write_fixture_data_config(tmp_path, test_path)

    texts = load_tool_use_probe_texts(config_path, n_samples=5, seed=1)

    assert len(texts) == 5
    assert all(t.startswith("goal number") for t in texts)


def test_load_tool_use_probe_texts_deterministic_for_fixed_seed(tmp_path):
    test_path = tmp_path / "test.jsonl"
    records = [
        {"task_id": f"glaive-{i}", "chain_length": 1, "user_goal": f"goal number {i}",
         "expected_tool_sequence": ["calculate_bmi"], "injected_error_at_step": None,
         "arguments": {}, "arguments_style": "raw_object", "source_index": i}
        for i in range(20)
    ]
    _write_fixture_jsonl(test_path, records)
    config_path = _write_fixture_data_config(tmp_path, test_path)

    a = load_tool_use_probe_texts(config_path, n_samples=5, seed=42)
    b = load_tool_use_probe_texts(config_path, n_samples=5, seed=42)
    assert a == b


# --- load_general_probe_texts: fixture wikitext-like sample, not the real one ---

def test_load_general_probe_texts_returns_first_n(tmp_path):
    from adbench.data.prepare import write_jsonl

    sample_path = tmp_path / "wikitext2_sample.jsonl"
    write_jsonl(
        [{"text": f"chunk {i}", "source_index": i, "n_words": 2} for i in range(30)],
        sample_path,
    )
    experiment_config = {"general_lm_eval": {"eval_set": str(sample_path)}}

    texts = load_general_probe_texts(experiment_config, n_samples=10)

    assert texts == [f"chunk {i}" for i in range(10)]


def test_compare_all_conditions_tags_rows_and_skips_missing_caches(tmp_path, monkeypatch):
    from adbench.analysis import layer_analysis
    from adbench.analysis.layer_analysis import compare_all_conditions, read_layer_analysis_results

    monkeypatch.setattr(layer_analysis, "REPO_ROOT", tmp_path)
    experiment_config = {"layer_analysis": {
        "cache_dir": "cache/", "results_path": "out/divergence.jsonl",
        "divergence_metrics": ["cka", "cosine_distance"],
    }}
    cache = tmp_path / "cache"
    for input_set in ("tool_use", "general"):
        save_activation_cache(_acts(4, 10, 12), cache / f"teacher_{input_set}")
        for condition in ("base", "distilled"):  # sft_only deliberately absent
            save_activation_cache(_acts(3, 10, 8), cache / f"student_{condition}_{input_set}")

    rows = compare_all_conditions(experiment_config, ("base", "sft_only", "distilled"))

    assert {r["condition"] for r in rows} == {"base", "distilled"}
    assert {r["input_set"] for r in rows} == {"tool_use", "general"}
    assert len(rows) == 2 * 2 * 3 * 2  # conditions x input sets x student layers x streams
    assert read_layer_analysis_results(tmp_path / "out" / "divergence.jsonl") == rows
