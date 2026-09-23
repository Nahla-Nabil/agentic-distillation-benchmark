# CLAUDE.md — orientation for working in this repo

This file is for whoever (or whichever Claude session) picks up this repo next.
It is a map, not a duplicate of the README/paper — read those for the research
argument; read this for "where things are and what's in flight."

**Owner:** Nahla Nabil. **Target:** SCIBT 2027 (IEEE, 4-8 pages), deadline
**2026-10-17**. Self-imposed cutoff for new GPU experiments: **2026-10-05**,
to leave ~12 days for writing.

## What this project is

Qwen3-14B teacher -> Qwen3-4B-Instruct-2507 student, Unsloth 4-bit QLoRA.
Question: does the benefit of knowledge distillation over plain SFT come from
the teacher's actual knowledge, or mostly from KD's regularizing effect
(anchoring against overfitting a small dataset) — and does that benefit scale
with chain length / tool novelty? See README.md's "Why this matters" for the
full framing.

## Repo layout

- `src/adbench/data/` — Glaive dataset prep (`prepare.py`, supports
  `--n-tools N` for the tool-diversity ablation), general-eval text sampling.
- `src/adbench/harness/` — the multi-step tool-use harness: `tools.py` (demo
  6-tool + extended 12-tool + Glaive 8-tool registries), `tasks.py` (synthetic
  task generation, `source="synthetic"|"synthetic_ext"|"glaive_train"|"glaive_test"`),
  `executor.py` (the run-a-task loop, tool-call parsing, retries).
- `src/adbench/training/` — `train.py` (all conditions go through
  `train_condition()`, selected by `--condition`; `student_key`/
  `checkpoint_dir_override`/`glaive_train_path` params make it reusable for
  the ablation and second model pair without touching the main pipeline),
  `losses.py` (SFT/KD/label-smoothing).
- `src/adbench/evaluation/` — `run_eval.py` (core eval loop + model loading),
  `batching.py` (BatchedModelFn — see "batching" below), `metrics.py`,
  `bfcl.py` (external BFCL benchmark loader/checker), and the **resumable
  two-GPU workers**: `ext_worker.py`, `bfcl_worker.py`, `ablation_worker.py`,
  `second_pair_worker.py` — all share job bookkeeping from `worker_utils.py`
  (`parse_jobs`/`split_jobs`/`run_jobs`; a worker's own `result_path()` and
  `describe_*_result()` are the only per-worker pieces).
- `src/adbench/analysis/layer_analysis.py` — CKA/cosine-distance layer probe.
  Runs teacher and student in **separate subprocesses** — Unsloth patches
  `transformers` globally on import, which breaks the other model's hooks in
  the same process.
- `src/adbench/pipeline_state.py` — resumable-stage + Hugging Face persistence
  (`restore()`/`push()`/`run_stage()`), used by every notebook.
- `configs/` — `data.yaml`, `models.yaml` (student/teacher pairs, LoRA), `experiment.yaml`
  (conditions list, training hyperparameters, harness settings).
- `notebooks/` — Kaggle-run pipelines, see "Notebooks" below.
- `tests/` — everything not needing a GPU is unit-tested (currently 449+
  tests). Anything touching Unsloth/real model weights is "reviewed by
  reading," not tested locally — flagged as such in the relevant module's
  docstring.

## Running tests

```
PYTHONPATH=src python3 -m pytest tests -q
```
No GPU/network needed. `datasets`, `unsloth`, `torch`-with-CUDA are NOT
installed in the local dev environment — anything importing them is deferred
inside functions, not at module level, specifically so this test suite stays
runnable without them.

## GPU execution model (Kaggle, T4 x2)

Every GPU-needing notebook: clones the repo, restores prior progress from a
private HF model repo (`NahlaNabil/adbench-run`), runs stages, pushes results
back. Needs `HF_TOKEN` (write access) as a Kaggle Secret; `GH_TOKEN` is
optional (repo is public). **Never link Kaggle's GitHub sync** — it has
overwritten notebooks in this repo before by pushing Kaggle's own execution
artifacts back to `master`.

**Batching**: `BatchedModelFn` exists and is correct (unit-tested, and
verified byte-identical to sequential eval on GPU), but on a T4 with Unsloth
it gives only ~1.0-1.4x speedup, not the expected ~10x — a batch's wall time
is set by its slowest row, and T4 decode throughput doesn't scale much with
batch size here. So the two-GPU-worker pattern (each worker its own
subprocess, `CUDA_VISIBLE_DEVICES` pinned) is what actually parallelizes
work, not batching within one GPU. `ADBENCH_EVAL_BATCH`/`ADBENCH_FAST_INFERENCE`
env vars still exist and are used at a modest batch size (8) since it's not
harmful and helps a little.

**Job-list balance**: when splitting jobs across two GPU workers by
`jobs[gpu::2]`, order the job list so expensive and cheap conditions
interleave evenly (condition-major, not seed-major) — a seed-major list with
an even number of conditions per seed silently gives one worker every slow
condition every time (this happened once in `ext_worker`'s job list; see
project memory for the exact failure). Check the resulting split's balance
before trusting a time estimate.

**Context length**: eval checkpoints load with `configs/models.yaml`'s
`max_seq_length` (2048) by default. The extended 12-tool task set's longer
system prompt + 5-step history can exceed that and crash Unsloth's attention
mask — `ext_worker.py` loads with `--max-seq-length 4096` for this reason;
new eval paths over longer contexts should do the same
(`load_condition_model(..., max_seq_length=...)`).

## Notebooks (chronological, only the ones still relevant)

- `09_v2_controls.ipynb` — main pipeline: base/sft_only/sft_early/self_distill/
  distilled(/sft_ls, /distilled_8b optionally) for one seed
  (`ADBENCH_SEED`/`ADBENCH_RUN_TAG=v2-seed<k>`). Run once per seed (0-4 done).
- `11_ext_eval.ipynb` — extended synthetic task set (158 tasks, 12 tools) on
  finished checkpoints, two GPU workers, resumable.
- `13_bfcl_eval.ipynb` — external BFCL benchmark ("simple" + "multiple"
  categories) on finished checkpoints, two GPU workers, resumable.
- `14_tool_diversity_ablation.ipynb` — trains+evaluates sft_only/distilled at
  2/4 distinct training tools (8 reuses the main pipeline's existing results
  — see `ablation_worker.py`'s docstring for why it's not retrained).
- `15_second_model_pair.ipynb` — trains+evaluates the second model pair
  (student_small=Qwen3-1.7B, teacher_8b=Qwen3-8B) — tests whether findings
  generalize beyond one model size.

`10_check_batching.ipynb` is superseded (folded into 11/13/14/15's setup
cells) — don't run it standalone.

## Current status / what's in flight

See the session's own project memory for exact numbers and dates (this file
doesn't duplicate those, since they change every run) — but as of writing:
seeds 0-4 done on the main pipeline + extended set + BFCL; tool-diversity
ablation and second model pair are implemented and either running or about
to run. If you're picking this repo up cold, check
`results/` and the HF repo's `runs/v2-seed*/results/stages/*.done` markers
for what's actually finished before assuming anything above is current.

## Conventions worth preserving

- A worker script's `result_path()` always includes every dimension that
  varies (condition, and n_tools/category when relevant) so two kinds of run
  can never silently collide on Hugging Face.
- New eval task sets get their own `task_set` string (`unseen_tools`,
  `unseen_tools_ext`, `bfcl_simple`, `bfcl_multiple`, ...) — never reuse or
  overload an existing one, so old and new results can be told apart in the
  same `eval_results.jsonl`/DataFrame without extra bookkeeping.
- Additive-only changes to shared config/code when possible: new conditions
  go in `configs/experiment.yaml` as new entries, new model pairs as new
  `configs/models.yaml` keys, new script params default to the old behavior
  — so nothing already-computed (checkpoints, eval results) needs to be
  distrusted or rerun because of a later change.
