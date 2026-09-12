"""Fetch and cache the general-language-modeling eval sample (wikitext-2-raw-v1)
used as the non-agentic comparison baseline (configs/experiment.yaml:
general_lm_eval). Kept separate from prepare.py because it's a different
dataset serving a different purpose: isolating "the student got worse at
everything" from "the student got worse specifically at agentic tool-use".

Run once, locally (CPU-only, no GPU needed):
    python -m adbench.data.general_eval --config configs/experiment.yaml

NOTE ON THE SOURCE REPO: configs/experiment.yaml's hf_dataset is
"Salesforce/wikitext", not the classic "wikitext" — the original repo's
loading script is no longer supported by `datasets` >= 4 (a
HfUriError/config-resolution error at load time; the underlying breaking
change was `datasets` dropping support for script-based dataset repos).
"Salesforce/wikitext" is the maintained, script-free mirror with the exact
same configs/splits/content.

FILTERING: about a third of wikitext-2-raw-v1's rows are blank (blank lines
between articles/sections) and many more are just a heading like
" = = Career = = " with no real prose — both are useless for a perplexity
eval and would silently deflate the sample's average text quality. Filtered
by a minimum word count (`min_word_count` in configs/experiment.yaml,
checked empirically: headings are typically 2-6 "words" including the "="
markers, ordinary sentences are usually 60+), not by pattern-matching the
heading syntax, since a word-count floor also naturally excludes blank rows
in the same pass.
"""

import argparse
import random
from pathlib import Path
from typing import Any


def filter_and_index_wikitext(raw_texts: list[str], min_word_count: int) -> list[dict[str, Any]]:
    """Keep only rows with at least `min_word_count` whitespace-separated
    words (drops blank lines and heading-only rows — see module docstring),
    tagging each with its position in the original (unfiltered) sequence
    for provenance. Pure function — no dataset/network access — so it's
    unit-testable against fabricated `raw_texts`."""
    kept = []
    for i, text in enumerate(raw_texts):
        stripped = text.strip()
        n_words = len(stripped.split())
        if n_words >= min_word_count:
            kept.append({"text": stripped, "source_index": i, "n_words": n_words})
    return kept


def sample_texts(records: list[dict[str, Any]], n_samples: int, seed: int) -> list[dict[str, Any]]:
    """Seeded sample of up to `n_samples` records — deterministic for a
    fixed seed, so perplexity results are reproducible across runs. Returns
    all records unchanged (not an error) if there are fewer than
    n_samples available."""
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n_samples]


def _resolve(repo_root: Path, config_relative_path: str) -> Path:
    return repo_root / config_relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing sample instead of refusing to run.",
    )
    args = parser.parse_args()

    import yaml

    repo_root = Path(__file__).resolve().parents[3]
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)["general_lm_eval"]

    output_path = _resolve(repo_root, config["eval_set"])
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to overwrite an existing sample ({output_path}). Pass "
            "--force if you deliberately want to regenerate it (this will "
            "change perplexity numbers for any condition re-evaluated afterward)."
        )

    from datasets import load_dataset  # deferred: only main() needs network/this dep

    source = config["source"]
    print(f"Loading {source['hf_dataset']} ({source['hf_config']}, split={source['split']}) ...")
    ds = load_dataset(source["hf_dataset"], source["hf_config"], split=source["split"])
    raw_texts = [row["text"] for row in ds]
    print(f"Loaded {len(raw_texts)} raw rows.")

    min_word_count = config.get("min_word_count", 10)
    filtered = filter_and_index_wikitext(raw_texts, min_word_count)
    print(f"{len(filtered)} rows have >= {min_word_count} words (kept).")

    sampled = sample_texts(filtered, config["n_samples"], seed=config.get("seed", 42))
    print(f"Sampled {len(sampled)} of them (target: {config['n_samples']}).")

    from adbench.data.prepare import write_jsonl
    write_jsonl(sampled, output_path)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
