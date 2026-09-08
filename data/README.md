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

## Subset stats

_Filled in once `prepare.py` is run — final example count, the selected tool
types, and their distribution._

## General LM eval sample

`data/general_eval/wikitext2_sample.jsonl` — a small sample of
[wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) used as the
*non-agentic* comparison baseline (perplexity), kept deliberately separate
from the Glaive split above since that's entirely tool-calling text. Built
by `python -m adbench.data.general_eval` (see
[configs/experiment.yaml](../configs/experiment.yaml):`general_lm_eval`).
Also gitignored — regenerate locally.
