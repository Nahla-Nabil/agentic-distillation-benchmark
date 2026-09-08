# Data

This project uses a filtered subset of
[glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
(Apache-2.0), built by `python -m adbench.data.prepare` (see
[configs/data.yaml](../configs/data.yaml) for the exact filtering/split
parameters).

Nothing under `data/raw/`, `data/processed/`, or `data/splits/` is checked
into the repo (see `.gitignore`) — regenerate it locally with the command
above. This keeps the repo light and avoids redistributing a filtered cut of
someone else's dataset.

## Split discipline

`data/splits/test.jsonl` (20%) is the held-out evaluation set. It is written
once by `prepare.py` and must never be touched again until final evaluation
(`adbench.evaluation.run_eval`) — no peeking during training or harness
development. `prepare.py` refuses to overwrite an existing test split for
this reason.

## Format notes from the raw dataset

Worth knowing before touching `prepare.py` or `_glaive_parsing.py`:

- The raw `chat` field is free text (`USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:`
  markers), not structured turns, and a `<functioncall>`'s `arguments` value
  is a **JSON-encoded string** wrapped in single quotes
  (`'arguments': '{"a": 1}'`) — not a nested object like Qwen's `<tool_call>`
  convention. `_glaive_parsing.py` reformats this at prep time (brace-count +
  string-boundary scanning; a naive regex fails on ~74% of real calls because
  the arguments string itself contains nested braces).
- Genuine **same-turn multi-step chains are vanishingly rare**: only 8 of
  112,960 raw examples have 2+ sequential tool calls with no user message in
  between. This dataset is essentially single-step. `prepare.py` keeps only
  examples with exactly one function call and labels every one
  `chain_length=1`; chain-length-3/5 eval tasks must be **composed** from
  several of these records — that belongs in `tasks.py`'s real
  `load_tasks()`, not here.
- The same tool name is called with **inconsistent argument key names**
  across different raw examples (synthetic per-example generation, not a
  fixed real API) — e.g. `calculate_distance` appears with
  `{origin,destination}`, `{start_location,end_location}`,
  `{point1,point2}`, etc. `prepare.py` keeps only the single most common
  ("canonical") key signature per tool (see `configs/data.yaml`) and drops
  the rest, rather than guessing a rename mapping.
- Argument *values*, not just keys, can be inconsistent even within a
  canonical signature — e.g. `convert_currency` examples use `"Euros"` and
  `"US dollars"` as often as `"EUR"`/`"USD"`. Found by
  `tests/test_prepare_real_output.py` actually *calling* every referenced
  tool against the real prepared data, not just checking the name exists —
  `build_glaive_registry()`'s `convert_currency` normalizes common
  natural-language names before falling back to a deterministic rate.
- ~0.9% of `<functioncall>` blocks fail to parse even with proper
  brace-counting — genuine data artifacts (an unescaped apostrophe inside the
  single-quoted arguments string, e.g. a message body containing
  `"[User's Name]"`, breaks the string's own closing-quote boundary). Skipped
  and counted (`drop_reasons.parse_error`), not repaired.

## Subset stats

From the most recent `python -m adbench.data.prepare` run against the full
112,960-row raw train split:

| | |
|---|---|
| Total kept | 800 (640 train / 160 test) |
| Distinct tools | 8 |
| Per tool | 100 (80 train / 20 test) |

Drop reasons (of 112,960 raw rows): `zero_calls` 49,742 · `tool_not_selected`
29,073 · `multi_call` 19,583 · `non_canonical_args` 3,437 · `parse_error` 999
· kept (`ok`) 10,126 (further capped to 800 by `per_tool_target`).

Selected tools (all deterministically mockable — see
`harness/tools.py::build_glaive_registry()`) and how many canonical-signature
examples were available for each before the 100/tool cap:

| Tool | Available | Kept |
|---|---|---|
| calculate_bmi | 2,946 | 100 |
| calculate_tip | 1,913 | 100 |
| calculate_discount | 1,852 | 100 |
| calculate_age | 1,722 | 100 |
| convert_currency | 1,081 | 100 |
| generate_random_number | 240 | 100 |
| calculate_distance | 189 | 100 |
| get_stock_price | 183 | 100 |

Full stats (including the top-40 raw tool-name frequency table across all
966 distinct tool names seen in the raw data) are written to
`data/splits/prepare_stats.json` by every `prepare.py` run — also gitignored,
regenerate locally to see the current numbers.

## General LM eval sample

`data/general_eval/wikitext2_sample.jsonl` — a small sample of
[wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) used as the
*non-agentic* comparison baseline (perplexity), kept deliberately separate
from the Glaive split above since that's entirely tool-calling text. Built
by `python -m adbench.data.general_eval` (see
[configs/experiment.yaml](../configs/experiment.yaml):`general_lm_eval`).
Also gitignored — regenerate locally.
