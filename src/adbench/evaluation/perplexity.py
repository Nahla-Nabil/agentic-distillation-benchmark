"""General language-modeling comparison baseline: perplexity of each
condition (base / sft_only / distilled student) on a general text sample,
to distinguish "this model got worse at everything" from "this model got
worse specifically at agentic tool-use" (configs/experiment.yaml:
general_lm_eval).

TODO:
  - Pick the eval_set: could be a held-out slice of a general corpus
    (e.g. wikitext) rather than data/splits/test.jsonl, since the latter is
    entirely tool-calling text and wouldn't isolate "general" degradation.
  - Implement compute_perplexity(model, tokenizer, texts) -> float using
    standard sliding-window / stride perplexity computation.
"""


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    raise NotImplementedError
