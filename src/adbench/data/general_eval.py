"""Fetch and cache the general-language-modeling eval sample (wikitext-2-raw-v1)
used as the non-agentic comparison baseline (configs/experiment.yaml:
general_lm_eval). Kept separate from prepare.py because it's a different
dataset serving a different purpose: isolating "the student got worse at
everything" from "the student got worse specifically at agentic tool-use".

Run once, locally (CPU-only, no GPU needed):
    python -m adbench.data.general_eval --config configs/experiment.yaml

TODO:
  - Load `wikitext`/`wikitext-2-raw-v1` (HF datasets) test split.
  - Sample/chunk `n_samples` non-empty text spans (wikitext-2 has a lot of
    blank/heading-only lines — filter those out).
  - Write to data/general_eval/wikitext2_sample.jsonl (gitignored, like
    data/splits/ — regenerate locally rather than committing).
  - This sample is read-only from evaluation/perplexity.py once written;
    same discipline as the Glaive test split, though wikitext isn't a
    "don't peek during training" concern since nothing is trained on it.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.parse_args()
    raise NotImplementedError("General LM eval-set preparation not yet implemented.")


if __name__ == "__main__":
    main()
