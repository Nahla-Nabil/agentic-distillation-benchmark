# CLAUDE.md — orientation for working in this repo

This file is for whoever (or whichever Claude session) picks up this repo next.
It is a map, not a duplicate of the README/paper — read those for the research
argument; read this for "where things are and what's in flight."

**Owner:** Nahla Nabil. **Target:** SCIBT 2027 (IEEE, 4-8 pages), deadline
**2026-10-17**. Self-imposed cutoff for new GPU experiments: **2026-10-05**,
to leave ~12 days for writing. **The plan of record is `docs/EXPERIMENT_PLAN.md`**
(thesis, claim/evidence table, weaknesses -> experiments, round schedule, and a
full crossed-bootstrap statistics pass) — read it first. **All GPU experiments
are DONE as of 2026-10-01** (three model pairs incl. OLMo-2); the project is in
the writing phase. **`docs/PAPER_OUTLINE.md`** has the section-by-section
outline, figure/table inventory, and drafting schedule.

## Writing the paper — read `docs/WRITING_GUIDE.md` before touching `paper/`

`docs/WRITING_GUIDE.md` is Nahla's own instruction file for AI models helping
with papers (copied from her Downloads, 2026-10-02). It is binding for every
edit under `paper/`. The rules that matter most for this paper:

- **Role:** editorial assistant, not ghostwriter. Nahla owns the ideas and the
  voice. Work **one section at a time**: explain the section to her in
  Levantine Arabic (what it says, why, which number comes from where), get her
  understanding/changes, THEN write or revise. She explicitly wants to
  understand every part — never batch-write several sections unreviewed.
- **Track B (IMRAD)**: Title → Abstract → Introduction → Methods → Results →
  Discussion → (Limitations/Conclusion) → References. Draft order: Methods →
  Results → Discussion → Introduction → Conclusion → Abstract → Title.
- **Methods:** simple past tense; primary and secondary endpoints stated
  precisely (primary = multi-step success on chains 3+5 of the 121-task
  unseen-tool set); software/hardware named; statistics paragraph last.
- **Results:** report only — NO interpretation, no "surprisingly/interestingly",
  simple past, same order and matching subheadings as Methods; don't restate in
  prose what a table shows. Interpretation ("anchoring, not knowledge") belongs
  in Discussion.
- **Discussion order:** findings recap → interpretation within the data →
  comparison with literature → what is new → negative results plainly →
  implications → limitations → future work → closing.
- **Conclusion:** what we found / why it matters / what next; no new results.
- **Abstract:** no citations, no interpretation; written last.
- **Voice:** follow `docs/VOICE_PROFILE.md` (derived from two of her own IEEE
  papers; also resolves where her style and the guide disagree, e.g. present
  tense for describing the method) and avoid the AI-cliché list in §3 of the
  guide. No em-dashes in paper prose. Never claim "first"/"novel"; scope novelty against
  [Furlanello18]/[Zhang19]/[Mobahi20] as in `docs/PAPER_OUTLINE.md` §2.
- **Numbers:** every number in `paper/` must trace to `docs/EXPERIMENT_PLAN.md`
  (or the HF results it summarizes); never round in a way that changes meaning.

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
  `second_pair_worker.py`, `third_pair_worker.py`, `sweep_worker.py` — all share job bookkeeping from `worker_utils.py`
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
- `tests/` — everything not needing a GPU is unit-tested (currently 478
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

**Teacher conditions need BOTH GPUs visible.** `train.load_teacher` puts the
teacher on GPU 1 only when `torch.cuda.device_count() > 1`; pinning a worker to
one GPU (right for `sft_only`/`base`) makes teacher + student share one T4 ->
CUDA OOM. So notebooks 14/15/16 run in two phases: Phase A = teacher-free jobs
as two pinned parallel workers; Phase B = teacher jobs (KD weight > 0, detected
via `resolve_training_config(...).kd.kd_weight > 0`) sequentially in ONE process
with both GPUs visible. Teacher jobs cost ~52 min (14B teacher), ~85-110 min
(1.7B student, 8B teacher or self-teacher).

**Loss logs are per Kaggle session and get lost** unless a worker uploads them:
`second_pair_worker` and `sweep_worker` upload each job's loss log and store a
`train_loss_summary` in every result row. Do that for any new training worker.

**Result-path versions**: `pair2_<cond>.json` = second pair, first run (2048-ctx
eval; base + sft_only live here); `pair2b_<cond>.json` = re-run of the teacher
conditions (4096-ctx, with loss logs). The first run's distilled_8b files are
kept as a trace. `sweep_<pair>_<sweep>_<cond>.json` = notebook 16.

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
  generalize beyond one model size. Finding: with the 1.7B student, KD (from
  8B or from itself) stays ~base (0.32-0.44) while SFT reaches 0.83-0.88 —
  KD's anchoring hurts a student whose base can't do the task.
- `16_sweeps.ipynb` — one `EXPERIMENT` per run (`sft_fair`, `dial_main`,
  `dial_small_probe`, `dial_small_full`, `dial_main_vheavy`,
  `dial_small_threshold`, `data_scale`): fair-SFT baseline and the KD-weight
  "anchoring dial", via `sweep_worker.py` and the additive sweeps in
  `configs/experiment.yaml` (`lower_lr`, `lower_lr_1ep`, `sft_heavy`,
  `sft_vheavy`, `kd_heavy`, `kd_0p1`). A sweep name may end in `@n<k>` = train
  on k examples per tool (seeded nested subset of the train split;
  `data/prepare.py::subsample_per_tool`). Different accounts run DIFFERENT
  experiments. `ADBENCH_TRAINING_LOG_DIR` redirects loss logs per worker.
  **Complete** — see `docs/EXPERIMENT_PLAN.md`'s "Statistics pass" for the
  full bootstrap results.
- `17_third_model_pair.ipynb` — trains+evaluates the THIRD model pair
  (student_olmo=OLMo-2-0425-1B-Instruct, teacher_olmo=OLMo-2-1124-7B-Instruct)
  — a different model FAMILY from Qwen3 (AI2's fully-open OLMo-2), not just a
  different size. Llama/Gemma were considered first and ruled out: both are
  gated on Hugging Face and block on a manual per-account license-acceptance
  step; OLMo-2 is ungated, tokenizer-verified compatible (GPT-2-style BPE,
  vocab 100278), and Unsloth-supported. Same two-phase pattern, same
  `third_pair_worker.py` (evaluates at 4096 ctx and uploads loss logs from
  the start — lessons from `second_pair_worker.py`'s history already baked
  in, so this pair should not need the same 2048->4096 re-run). **Complete**
  (2026-10-01); results + floor-effect caveat in `docs/EXPERIMENT_PLAN.md`
  "Round 3". The repo is now PUBLIC (made public 2026-10-01 so new Kaggle
  accounts can clone without a `GH_TOKEN`).
- `18_olmo7b_pair.ipynb` — FOURTH pair, OLMo-2 again but larger
  (student_olmo7=OLMo-2-1124-7B-Instruct, teacher_olmo13=OLMo-2-1124-13B-Instruct;
  conditions base/sft_only/distilled_olmo7/self_distill_olmo7), added because
  the 1B OLMo student sat at the chains-3/5 floor. Same worker:
  `third_pair_worker.py --pair olmo7b` (default `--pair olmo1b` unchanged),
  results under the `pair4_` prefix. Split across two accounts via the first
  cell's `CONDITIONS`/`INCLUDE_BASE` (see the notebook's intro). Built
  2026-10-02, not yet run.
- `19_ctx4096_reeval.ipynb` — re-evaluates every primary-set number that was
  measured at a 2048 context (main pair: eval only via `ext_worker --task-set
  primary` → `primary4k_eval_*`, task_set `unseen_tools_ctx4096`; second pair
  base/sft_only: retrain via `second_pair_worker --rerun-4k` → `pair2c_*`).
  After it runs, paper numbers for those conditions come from the 4096 files.
  Built 2026-10-02, not yet run. See `docs/EXPERIMENT_PLAN.md` "Round 5".

`10_check_batching.ipynb` is superseded (folded into 11/13/14/15's setup
cells) — don't run it standalone.

## Current status / what's in flight

See the session's own project memory for exact numbers and dates (this file
doesn't duplicate those, since they change every run) — but as of writing:
seeds 0-4 done on the main pipeline + extended set + BFCL; tool-diversity
ablation complete (12/12 jobs, KD flat across 2/4/8 tools); second model pair
complete (KD ~ base for the small student); notebook 16's sweep rounds 1-2
complete with a full crossed-bootstrap pass (23 contrasts); notebook 17 (third
model pair, OLMo-2) complete — ordering replicates (base 0 < distilled_olmo 4.1
< self_distill_olmo 10.7 < sft_only 31.7 overall %) but chains 3+5 are at floor
(~0%) for every condition, so that pair supports only the overall/chain-1
claim. **The project is in the writing phase** — `paper/03_method.md` and
`paper/04_results.md` are first drafts written BEFORE `docs/WRITING_GUIDE.md`
was adopted and need a revision pass (Methods to past tense + explicit
endpoints + software; Results stripped of interpretation, which moves to
Discussion). `docs/PAPER_OUTLINE.md` has the plan. If you're picking this repo up cold,
check `results/` and the HF repo's `runs/v2-seed*/results/stages/*.done`
markers for what's actually finished before assuming anything above is
current.

## Conventions worth preserving

- A worker script's `result_path()` always includes every dimension that
  varies (condition, and n_tools/category/pair/sweep when relevant) so two kinds of run
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
