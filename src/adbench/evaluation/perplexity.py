"""General language-modeling comparison baseline: perplexity of each
condition (base / sft_only / distilled student) on a general text sample,
to distinguish "this model got worse at everything" from "this model got
worse specifically at agentic tool-use" (configs/experiment.yaml:
general_lm_eval).

Eval set is wikitext-2-raw-v1 (not the Glaive test split, which is entirely
tool-calling text) — see src/adbench/data/general_eval.py for how it's
fetched/cached, and configs/experiment.yaml:general_lm_eval for the source
config.

TODO:
  - Load data/general_eval/wikitext2_sample.jsonl (built by
    adbench.data.general_eval; run it first if the file doesn't exist yet).
  - Implement compute_perplexity(model, tokenizer, texts) -> float using
    standard sliding-window / stride perplexity computation.
"""


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    raise NotImplementedError
