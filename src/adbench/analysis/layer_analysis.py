"""Extract and compare per-layer activations between teacher and student on
the same probe inputs (configs/experiment.yaml: layer_analysis).

TODO:
  - extract_activations(model, tokenizer, texts, layers) -> dict mapping
    layer index -> {"attention": tensor, "ffn": tensor} using forward hooks
    on the attention output projection and the FFN down-projection of each
    transformer block (module names differ between Qwen3-14B and
    Qwen3-4B-Instruct-2507 — confirm exact module paths before hooking).
  - Layer alignment: teacher and student have different depths, so compare
    via configs/models.yaml's `layer_alignment.strategy` (uniform_span by
    default) rather than assuming a 1:1 layer index match.
  - divergence(student_acts, teacher_acts, metric) -> float per aligned
    layer pair, for metric in {cka, cosine_distance} (experiment.yaml).
  - Probe inputs: use a fixed sample of n_probe_examples from
    data/splits/test.jsonl (read-only use, consistent with run_eval.py) so
    activation and behavioral results are drawn from the same held-out data.
  - Output shape should be easy to plot as "divergence vs layer depth,
    attention vs FFN" — that plot is the deliverable.
"""


def extract_activations(model, tokenizer, texts: list[str], layers: list[int]):
    raise NotImplementedError


def divergence(student_acts, teacher_acts, metric: str = "cka"):
    raise NotImplementedError
