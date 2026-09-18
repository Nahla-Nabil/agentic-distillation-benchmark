"""General language-modeling comparison baseline: perplexity of each
condition (base / sft_only / distilled student) on a general text sample,
to distinguish "this model got worse at everything" from "this model got
worse specifically at agentic tool-use" (configs/experiment.yaml:
general_lm_eval).

Eval set is wikitext-2-raw-v1 (not the Glaive test split, which is entirely
tool-calling text) — see src/adbench/data/general_eval.py for how it's
fetched/cached, and configs/experiment.yaml:general_lm_eval for the source
config. Run `python -m adbench.data.general_eval` first if
data/general_eval/wikitext2_sample.jsonl doesn't exist yet.

compute_perplexity() itself needs only a model + tokenizer object (CPU
torch is enough to unit-test it against a tiny fake model — see
tests/test_perplexity.py — though a real run against Qwen3 needs a GPU for
any reasonable speed). Each text is scored independently in a single
forward pass (no sliding window) — the wikitext sample is pre-chunked to
short spans at prep time specifically so this is valid; a sliding-window
stride would only be needed for documents longer than the model's context.
"""

import math
from pathlib import Path
from typing import Any


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    """Standard exp(mean NLL per token) perplexity, pooling every token
    across all `texts` into one mean (not one perplexity per text then
    averaged — pooling first is the standard convention, and avoids
    over-weighting short texts relative to long ones).

    model: called as model(input_ids=...), returning either a `.logits`-
        bearing object (HF's CausalLMOutput convention) or a raw logits
        tensor directly.
    tokenizer: called as tokenizer(text) -> {"input_ids": list[int]}
        (no padding needed — texts are processed one at a time).
    max_length: texts longer than this are truncated (wikitext2_sample.jsonl's
        chunks are short by construction, so this is a safety cap, not the
        normal case).

    Raises ValueError if every text is too short to score (fewer than 2
    tokens — need at least one token pair to form a next-token prediction).
    """
    import torch
    import torch.nn.functional as F

    total_nll = 0.0
    total_tokens = 0

    # A real nn.Module lives on a specific device and needs its inputs
    # there too (without a multi-GPU dispatch hook to move them for us);
    # a plain callable — the contract also allows one — has no parameters,
    # so leave the tensor on the default device.
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = None

    for text in texts:
        ids = tokenizer(text)["input_ids"][:max_length]
        if len(ids) < 2:
            continue

        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            output = model(input_ids=input_ids)
        logits = output.logits if hasattr(output, "logits") else output

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    if total_tokens == 0:
        raise ValueError(
            f"No text had >= 2 tokens to score (out of {len(texts)} given) — "
            "nothing to compute perplexity over."
        )
    return math.exp(total_nll / total_tokens)


def load_general_eval_texts(path: str | Path) -> list[str]:
    """Read data/general_eval/wikitext2_sample.jsonl (or wherever
    configs/experiment.yaml:general_lm_eval.eval_set points) back into a
    plain list of strings for compute_perplexity()."""
    from adbench.data.prepare import read_jsonl
    return [record["text"] for record in read_jsonl(path)]


def general_eval_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """The general_lm_eval block of an already-loaded configs/experiment.yaml."""
    return experiment_config["general_lm_eval"]
