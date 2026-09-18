"""Layer-wise activation comparison between teacher and student on matched
inputs, to localize where representational divergence originates (attention
vs FFN, early vs late layers) — the paper's key mechanistic result.

ARCHITECTURE (verified, not assumed — see configs/models.yaml's
layer_alignment comment): Qwen3-14B has 40 layers x hidden_size 5120;
Qwen3-4B-Instruct-2507 has 36 layers x hidden_size 2560. Both depth AND
width differ. Each transformer block (Qwen3DecoderLayer, standard
Llama-family layout) exposes `.self_attn.o_proj` (attention output
projection) and `.mlp.down_proj` (FFN output projection) — hooked
separately per the eval spec ("not just the combined residual stream").
find_decoder_layers() locates these by NAME PATTERN (`...layers.<N>`) via
named_modules(), not a hardcoded attribute chain, so it keeps working
whether `model` is a bare transformers model or a PEFT/Unsloth-wrapped one
(different wrapper depths, same underlying architecture).

WHY BOTH CKA AND COSINE DISTANCE, AND WHY COSINE ISN'T WHAT IT SOUNDS LIKE
HERE (flagged design decision): plain cosine similarity between two
activation VECTORS requires the same dimensionality — teacher's are
5120-dim, student's are 2560-dim, so raw-vector cosine distance is simply
not defined between them (no valid way to pad/truncate one to match the
other without fabricating information). CKA (Kornblith et al. 2019) sidesteps
this by comparing (n x n) Gram matrices X@X.T built from n samples, which
have the SAME shape regardless of the original per-sample dimensionality —
this is what actually makes a teacher/student comparison possible here at
all, and is the metric to trust as primary. cosine_distance_rsa() below
gets the same dimension-independence by applying the same trick with a
different inner product: instead of raw activation vectors, it compares
each model's own (n x n) PAIRWISE-COSINE-SIMILARITY matrix (representational
similarity analysis / RSA) — i.e. "how similar are pairs of probe inputs to
EACH OTHER, from teacher's point of view, vs from student's point of view",
which is a comparison of similarity *structure*, not of raw geometry, and
is exactly what stays well-defined across differing widths. Reported
alongside CKA as a robustness check (they can disagree; neither is known to
be strictly more correct for this comparison), not as a literal
"cosine(teacher_vector, student_vector)".

SAMPLE UNIT: each probe text is reduced to ONE vector per layer per stream
by mean-pooling over its token positions (extract_activations processes one
text at a time — no padding/attention-mask bookkeeping needed for CKA/RSA's
purposes). This trades away token-level granularity for a fixed, known
sample count (= n_probe_examples) per layer, which is what both CKA and RSA
need as their "n" dimension; per-token analysis is a possible future
extension, not implemented here.

MEMORY (flagged per the eval spec): a free-tier T4 has 16GB VRAM. Qwen3-14B
in 4-bit is roughly 7-8GB; Qwen3-4B in 4-bit roughly 2-2.5GB — around
9.5-10.5GB for base weights alone before activation memory, KV-cache, and
Unsloth/bitsandbytes overhead. Loading both simultaneously is tight and a
real risk of OOM, especially since this workload additionally holds every
layer's activations in memory during extraction. The design here avoids the
question entirely rather than hoping it fits: extract_and_cache_activations()
loads ONE model, runs the probe sets, writes activations to disk, and
returns — the caller then frees that model (del + torch.cuda.empty_cache())
before loading the other. compare_cached_activations() only needs the two
resulting cache FILES, never both models loaded at once. See
notebooks/04_layer_analysis.ipynb, which follows exactly this sequence.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

_LAYER_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")


def find_decoder_layers(model: Any) -> list[tuple[int, Any]]:
    """Locate a model's numbered transformer-block submodules by NAME
    PATTERN (any module whose qualified name ends in `layers.<N>`), sorted
    by layer index — works for a bare transformers model
    (`model.layers.0`) or one wrapped by PEFT/Unsloth
    (`base_model.model.model.layers.0`, etc.) without needing to know the
    wrapper depth in advance. Raises ValueError if nothing matches (wrong
    architecture, or an unrecognized wrapping scheme worth investigating
    rather than silently hooking nothing)."""
    found = []
    for name, module in model.named_modules():
        m = _LAYER_NAME_RE.search(name)
        if m:
            found.append((int(m.group(1)), module))
    if not found:
        raise ValueError(
            "No submodules matching '...layers.<N>' found — this model's "
            "architecture or wrapping isn't what find_decoder_layers() "
            "expects (Llama-family: model.layers.<N>, optionally wrapped "
            "by PEFT/Unsloth). Inspect model.named_modules() directly."
        )
    found.sort(key=lambda pair: pair[0])
    return found


def count_layers(model: Any) -> int:
    """Number of decoder layers find_decoder_layers() finds — used to build
    align_layers()'s mapping from the ACTUALLY LOADED models rather than a
    hardcoded layer count in a config file, so it's always right even if a
    checkpoint's architecture ever changes."""
    return len(find_decoder_layers(model))


class _ActivationRecorder:
    """Registers forward hooks on every layer's self_attn.o_proj and
    mlp.down_proj, mean-pooling each captured output over the sequence
    dimension (dim=1, i.e. tokens) as it arrives — one call at a time, one
    pooled vector appended per layer per stream per call. remove() detaches
    all hooks; always call it (a context manager would work too, but the
    extraction loop needs the hooks live across many forward() calls, so
    this is used as an explicit start/stop pair instead)."""

    def __init__(self, layer_modules: list[tuple[int, Any]]):
        self.pooled: dict[int, dict[str, list[np.ndarray]]] = {
            idx: {"attention": [], "ffn": []} for idx, _ in layer_modules
        }
        self._handles = []
        for idx, layer in layer_modules:
            self._handles.append(
                layer.self_attn.o_proj.register_forward_hook(self._make_hook(idx, "attention"))
            )
            self._handles.append(
                layer.mlp.down_proj.register_forward_hook(self._make_hook(idx, "ffn"))
            )

    def _make_hook(self, layer_idx: int, stream: str):
        def hook(_module, _inputs, output):
            # output: (batch, seq_len, hidden). Cast to float32 before
            # pooling (activations may be bf16/fp16 under 4-bit inference;
            # pooling and the numpy math downstream want full precision).
            # Mean-pool over tokens -> (batch, hidden); batch is always 1
            # here (extract_activations processes one text at a time) but
            # reshape defensively rather than assuming.
            pooled = output.detach().float().mean(dim=1)
            self.pooled[layer_idx][stream].append(pooled.cpu().numpy().reshape(-1))
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def stacked(self) -> dict[int, dict[str, np.ndarray]]:
        """(n_calls, hidden) array per layer per stream, in call order."""
        return {
            idx: {stream: np.stack(vectors) for stream, vectors in streams.items()}
            for idx, streams in self.pooled.items()
        }


def extract_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer_indices: list[int] | None = None,
    max_length: int = 128,
) -> dict[int, dict[str, np.ndarray]]:
    """Run every text through `model` one at a time (no padding needed),
    returning {layer_index: {"attention": (n_texts, hidden), "ffn": (n_texts, hidden)}}
    — each row is one text's mean-pooled activation at that layer/stream.

    layer_indices=None (default) captures every layer find_decoder_layers()
    finds; pass a subset to save memory/time if a full sweep isn't needed.
    """
    import torch

    all_layers = find_decoder_layers(model)
    if layer_indices is not None:
        wanted = set(layer_indices)
        all_layers = [(idx, layer) for idx, layer in all_layers if idx in wanted]
        if len(all_layers) != len(wanted):
            missing = wanted - {idx for idx, _ in all_layers}
            raise ValueError(f"layer_indices {sorted(missing)} not found in this model.")

    recorder = _ActivationRecorder(all_layers)
    try:
        model.eval()
        device = next(model.parameters()).device
        for text in texts:
            ids = tokenizer(text)["input_ids"][:max_length]
            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                model(input_ids=input_ids)
    finally:
        recorder.remove()

    return recorder.stacked()


def load_student_checkpoint_for_extraction(
    condition: str, experiment_config: dict[str, Any], models_config: dict[str, Any]
):
    """Load a trained student checkpoint for activation extraction, via
    plain transformers + peft — deliberately NOT Unsloth's
    FastLanguageModel (which evaluation/run_eval.py's load_condition_model()
    uses for eval/generation).

    Loading a LoRA checkpoint through Unsloth patches every decoder layer's
    attention/MLP forward into a fused kernel at FastLanguageModel.
    get_peft_model() time (the "Unsloth patched N layers with N QKV layers,
    N O layers and N MLP layers" message) — this happens at LOAD time, not
    per-call, so it happens regardless of for_inference()/for_training()
    mode. That fused path computes attention/MLP directly rather than by
    calling self_attn.o_proj/mlp.down_proj as plain nn.Module.forward(), so
    extract_activations()'s register_forward_hook on those submodules never
    fires — every pooled list stays empty, and np.stack([]) raises "need at
    least one array to stack". Loading through plain
    transformers.AutoModelForCausalLM + peft.PeftModel keeps the standard
    per-submodule forward() calls the hooks depend on.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from adbench.training.train import resolve_checkpoint_dir

    checkpoint_dir = resolve_checkpoint_dir(experiment_config, condition)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint for condition {condition!r} at {checkpoint_dir} — "
            "run `python -m adbench.training.train --condition ...` first."
        )
    student_cfg = models_config["student"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=student_cfg["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        student_cfg["hf_id"], quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(student_cfg["hf_id"])
    return model, tokenizer


# --------------------------------------------------------------------------
# Layer alignment — teacher and student have different depths.
# --------------------------------------------------------------------------

def align_layers(n_source_layers: int, n_target_layers: int) -> dict[int, int]:
    """{source_layer_index: target_layer_index}, uniform_span strategy
    (configs/models.yaml: layer_alignment.strategy) — source layer i maps to
    round(i * n_target_layers / n_source_layers), clamped to a valid target
    index. Identity when the two depths are equal (e.g. sft_only vs
    distilled, both 36 layers) — no special-casing needed for that case."""
    if n_source_layers <= 0 or n_target_layers <= 0:
        raise ValueError("n_source_layers and n_target_layers must both be positive.")
    return {
        i: min(round(i * n_target_layers / n_source_layers), n_target_layers - 1)
        for i in range(n_source_layers)
    }


# --------------------------------------------------------------------------
# Divergence metrics — pure numpy, fully unit-tested with synthetic tensors.
# --------------------------------------------------------------------------

_EPS = 1e-8


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two representation
    matrices X (n, p1) and Y (n, p2) — n must match (same probe inputs);
    p1/p2 need NOT match (this is exactly why CKA, not raw cosine, is the
    primary metric for a 5120-dim vs 2560-dim comparison). Rotation- and
    reflection-invariant, in [0, 1] for real inputs (1 = identical
    representational geometry up to rotation/isotropic scaling).

    Returns 0.0 (not NaN) if either X or Y has zero variance across samples
    (a degenerate, fully-constant representation) — a documented edge-case
    convention so a single degenerate layer doesn't propagate NaN through
    an entire results table/plot.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples: {X.shape[0]} != {Y.shape[0]}.")

    Kx = _center_gram(X @ X.T)
    Ky = _center_gram(Y @ Y.T)

    hsic_xy = float(np.sum(Kx * Ky))
    hsic_xx = float(np.sum(Kx * Kx))
    hsic_yy = float(np.sum(Ky * Ky))

    denom = np.sqrt(hsic_xx) * np.sqrt(hsic_yy)
    if denom < _EPS:
        return 0.0
    return hsic_xy / denom


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / np.clip(norms, _EPS, None)
    return normalized @ normalized.T


def cosine_distance_rsa(X: np.ndarray, Y: np.ndarray) -> float:
    """Representational-similarity-analysis-style cosine distance: 1 minus
    the cosine similarity between X's and Y's own (n, n) pairwise-cosine-
    similarity matrices — see the module docstring's "WHY BOTH CKA AND
    COSINE DISTANCE" section for why this, not raw-vector cosine, is what's
    computed. In [0, 2]; 0 means the two representations induce the exact
    same relative-similarity structure over the probe inputs (also
    rotation/reflection-invariant, like CKA).
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples: {X.shape[0]} != {Y.shape[0]}.")

    n = X.shape[0]
    if n < 2:
        raise ValueError("cosine_distance_rsa needs at least 2 samples to form a similarity structure.")

    iu = np.triu_indices(n, k=1)
    a = _cosine_similarity_matrix(X)[iu]
    b = _cosine_similarity_matrix(Y)[iu]

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < _EPS:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


DIVERGENCE_METRICS = {"cka": cka, "cosine_distance": cosine_distance_rsa}


def divergence(source_acts: np.ndarray, target_acts: np.ndarray, metric: str = "cka") -> float:
    """Dispatch to one of DIVERGENCE_METRICS by name. Note cka() returns a
    SIMILARITY (1 = identical) while cosine_distance_rsa() returns a
    DISTANCE (0 = identical) — deliberately not normalized to a common
    "higher = more different" convention, since flipping CKA's well-known
    published scale would make it less recognizable; compare_layers()
    below labels each column by its metric name so this isn't ambiguous
    downstream."""
    if metric not in DIVERGENCE_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(DIVERGENCE_METRICS)}.")
    return DIVERGENCE_METRICS[metric](source_acts, target_acts)


# --------------------------------------------------------------------------
# Comparison across an aligned layer pair, both streams, both metrics.
# --------------------------------------------------------------------------

def compare_layers(
    source_activations: dict[int, dict[str, np.ndarray]],
    target_activations: dict[int, dict[str, np.ndarray]],
    layer_alignment: dict[int, int],
    metrics: tuple[str, ...] = ("cka", "cosine_distance"),
    input_set: str | None = None,
) -> list[dict[str, Any]]:
    """One row per (source layer, stream), each row's aligned target layer
    from `layer_alignment` — exactly the shape needed to plot "divergence
    vs layer index, one line per stream" (and, across two calls with
    input_set="tool_use"/"general", per input set). `source` is
    conventionally the student and `target` the teacher (or any other
    model pair — align_layers()/compare_layers() don't care which is
    "bigger"), matching how a paper figure reads "student layer i is like
    teacher layer j".
    """
    rows = []
    for source_layer, target_layer in sorted(layer_alignment.items()):
        if source_layer not in source_activations or target_layer not in target_activations:
            continue
        for stream in ("attention", "ffn"):
            row = {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "stream": stream,
            }
            if input_set is not None:
                row["input_set"] = input_set
            for metric in metrics:
                row[metric] = divergence(
                    source_activations[source_layer][stream],
                    target_activations[target_layer][stream],
                    metric=metric,
                )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# I/O — cache activations to disk (see module docstring's MEMORY section)
# and write the final comparison table.
# --------------------------------------------------------------------------

def save_activation_cache(activations: dict[int, dict[str, np.ndarray]], path: str | Path) -> Path:
    """Returns the path actually written — `np.savez` silently appends
    ".npz" to a path that doesn't already end in it, so a caller passing an
    extension-less path and later reading it back with the *original* path
    would get a FileNotFoundError; normalizing the suffix here first and
    returning it avoids that footgun."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        f"layer{idx}__{stream}": array
        for idx, streams in activations.items()
        for stream, array in streams.items()
    }
    np.savez_compressed(path, **flat)
    return path


def load_activation_cache(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    loaded: dict[int, dict[str, np.ndarray]] = {}
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    with np.load(path) as data:
        for key in data.files:
            layer_part, stream = key.split("__")
            idx = int(layer_part.removeprefix("layer"))
            loaded.setdefault(idx, {})[stream] = data[key]
    return loaded


def write_layer_analysis_results(rows: list[dict[str, Any]], path: str | Path) -> None:
    from adbench.data.prepare import write_jsonl
    write_jsonl(rows, path)


def read_layer_analysis_results(path: str | Path) -> list[dict[str, Any]]:
    from adbench.data.prepare import read_jsonl
    return read_jsonl(path)


# --------------------------------------------------------------------------
# Probe input sets — the two the eval spec asks for, side by side:
#   tool_use: real Glaive TEST-split text (held out from training, same
#     discipline as evaluation/run_eval.py).
#   general: the same wikitext sample evaluation/perplexity.py uses.
# Both pure I/O + sampling — no model needed, unit-tested against fixture
# files rather than the real generated data/splits/data/general_eval output.
# --------------------------------------------------------------------------

def load_tool_use_probe_texts(
    data_config_path: str | Path, n_samples: int, seed: int
) -> list[str]:
    """Seeded sample of `user_goal` strings from the Glaive test split —
    read-only, same held-out file evaluation/run_eval.py's
    source="glaive_test" tasks use."""
    from adbench.data.general_eval import sample_texts
    from adbench.data.prepare import load_config, read_jsonl

    config = load_config(data_config_path)
    records = read_jsonl(REPO_ROOT / config["output"]["test_path"])
    sampled = sample_texts(records, n_samples, seed)
    return [r["user_goal"] for r in sampled]


def load_general_probe_texts(experiment_config: dict[str, Any], n_samples: int) -> list[str]:
    """The first `n_samples` wikitext chunks from
    data/general_eval/wikitext2_sample.jsonl (already seeded-sampled once
    at prep time by adbench.data.general_eval — no need to re-sample)."""
    from adbench.evaluation.perplexity import (
        general_eval_config,
        load_general_eval_texts,
    )

    eval_config = general_eval_config(experiment_config)
    texts = load_general_eval_texts(REPO_ROOT / eval_config["eval_set"])
    return texts[:n_samples]


# --------------------------------------------------------------------------
# Orchestration — needs a real model (Unsloth/GPU) to extract activations
# from; compare_cached_activations() itself only needs two cache FILES (see
# module docstring's MEMORY section — this split is what lets the two
# models never be loaded at the same time).
# --------------------------------------------------------------------------

def extract_and_cache_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    cache_path: str | Path,
    layer_indices: list[int] | None = None,
    max_length: int = 128,
) -> Path:
    """Run extract_activations() and immediately persist the result — call
    this once per model (teacher, then free it; student, then free it),
    never with both models resident. Returns the path written."""
    activations = extract_activations(model, tokenizer, texts, layer_indices, max_length)
    return save_activation_cache(activations, cache_path)


def compare_cached_activations(
    source_cache_path: str | Path,
    target_cache_path: str | Path,
    layer_alignment: dict[int, int],
    metrics: tuple[str, ...] = ("cka", "cosine_distance"),
    input_set: str | None = None,
) -> list[dict[str, Any]]:
    """compare_layers(), reading both sides from disk — no model objects
    needed at all, so this can run long after (or on a different machine
    from) whichever session extracted the activations."""
    source_activations = load_activation_cache(source_cache_path)
    target_activations = load_activation_cache(target_cache_path)
    return compare_layers(source_activations, target_activations, layer_alignment, metrics, input_set)


# --------------------------------------------------------------------------
# CLI — one model per PROCESS. Unsloth monkeypatches transformers'/peft's
# forward classes globally the moment it is imported, so once ANY earlier
# cell in the same kernel has loaded a model through Unsloth (training,
# eval, or the teacher below), even a model loaded through plain
# transformers + peft is routed through Unsloth's fused forward — which
# both bypasses the o_proj/down_proj hooks and raises NotImplementedError on
# a model Unsloth did not load itself. A fresh interpreter that never
# imports Unsloth avoids all of that, so each role is extracted in its own
# subprocess (see notebooks/07_final.ipynb, section 4).
# --------------------------------------------------------------------------

def _extract_role(role: str, condition: str, experiment_config: dict[str, Any], models_config: dict[str, Any]) -> int:
    """Extract + cache tool_use and general activations for one role
    ("teacher" or "student"); returns that model's layer count."""
    import gc
    import sys

    import torch
    from transformers import AutoTokenizer

    la_config = experiment_config["layer_analysis"]
    cache_dir = REPO_ROOT / la_config["cache_dir"]
    tool_use_texts = load_tool_use_probe_texts("configs/data.yaml", la_config["n_probe_examples"], la_config["seed"])
    general_texts = load_general_probe_texts(experiment_config, la_config["n_probe_examples"])
    # Teacher and student share the Qwen3 tokenizer vocabulary.
    tokenizer = AutoTokenizer.from_pretrained(models_config["student"]["hf_id"])

    if role == "student":
        if "unsloth" in sys.modules:
            raise RuntimeError(
                "unsloth is already imported in this process — student activation "
                "extraction must run in a fresh process without it."
            )
        model, _ = load_student_checkpoint_for_extraction(condition, experiment_config, models_config)
    else:
        from adbench.training.train import load_teacher
        model = load_teacher(models_config)

    n_layers = count_layers(model)
    print(f"{role}: {n_layers} layers")
    for name, texts in (("tool_use", tool_use_texts), ("general", general_texts)):
        extract_and_cache_activations(
            model, tokenizer, texts, cache_dir / f"{role}_{name}",
            max_length=la_config["probe_max_length"],
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return n_layers


def main() -> None:
    import argparse
    import os
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.chdir(REPO_ROOT)

    from adbench.training.train import CONDITIONS, load_experiment_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["teacher", "student"], required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="distilled",
                        help="Which trained student checkpoint to extract (role=student only).")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    models_config = load_experiment_config(args.models_config)
    _extract_role(args.role, args.condition, experiment_config, models_config)


if __name__ == "__main__":
    main()
