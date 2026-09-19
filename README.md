# Agentic Distillation Benchmark

**Research question:** when a smaller student model is distilled from a larger
teacher, does its reliability on multi-step agentic tool-use degrade faster
than its general language modeling performance — and if so, at which layers
does that gap originate?

## Why this matters

Distillation is usually validated against general benchmarks (perplexity,
MMLU-style accuracy) or single-turn task accuracy. Multi-step agentic
tool-use — call a tool, read the result, decide the next step, recover from
errors, repeat — compounds small per-step error rates across a chain, so it
may be a much more sensitive probe of what a student model actually lost
during distillation. This project measures that gap directly, and tries to
localize it to specific transformer layers.

## Models

| Role    | Model                              | Notes                                   |
|---------|-------------------------------------|------------------------------------------|
| Teacher | `Qwen/Qwen3-14B`                    | inference-only, produces KD targets      |
| Student | `Qwen/Qwen3-4B-Instruct-2507`       | fine-tuned under 3 conditions below      |

Both are loaded via [Unsloth](https://github.com/unslothai/unsloth) (4-bit
QLoRA) so training fits on a free Colab T4. See
[configs/models.yaml](configs/models.yaml).

## The three-way comparison

The same student model, evaluated identically, under three conditions —
isolating whether any degradation comes from *distillation specifically*,
rather than from fine-tuning in general:

1. **Base student** — no fine-tuning. Baseline.
2. **SFT-only student** — standard fine-tuning on the tool-calling data, no
   teacher involved. Control.
3. **Distillation-guided student** — combined KD (from teacher) + SFT loss
   on the same data, same LoRA config. Treatment.

Conditions 2 and 3 differ **only** in the loss function (see
[configs/experiment.yaml](configs/experiment.yaml) and
[src/adbench/training/losses.py](src/adbench/training/losses.py)) — same
data, same split, same LoRA rank/hyperparameters, same seed.

## Dataset

[`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
(Apache-2.0), filtered to a 500-1000 example subset covering 5-10 distinct
tool types, split 80% train / 20% held-out test. **The test split is frozen
at creation time and never touched until final evaluation.** See
[data/README.md](data/README.md) and [configs/data.yaml](configs/data.yaml).

## What gets measured

1. **Multi-step task success rate** via a mini agentic harness that runs
   3-5 sequential tool calls (call → read result → decide next step →
   handle injected errors) — not just single-turn function-calling accuracy.
2. **Success rate vs chain length** (1 vs 3 vs 5 steps), per condition —
   the central plot for the research question.
3. **General language modeling performance** (perplexity) per condition, as
   the comparison baseline that isolates agentic-specific degradation from
   general capability loss.
4. **Layer-wise activation divergence** between teacher and distilled
   student on matched inputs (attention vs FFN, by layer depth), to
   localize where the gap originates.

## Why synthetic tasks for multi-step eval

The harness's multi-step eval (chain lengths 1, 3, 5) runs against a
**hand-designed synthetic task set** (`harness/tasks.py::build_synthetic_tasks()`,
121 tasks: 46 at length 1, 40 at length 3, 35 at length 5), not real
glaive-function-calling-v2 data — deliberately, and only for the multi-step
eval axis. Training still uses real glaive data throughout.

**Why not real data for eval too:** `data/prepare.py` found that genuine
multi-step tool chains are essentially absent from
glaive-function-calling-v2 — only 8 of 112,960 raw examples have a second
tool call before the next user message (see `data/README.md`'s "Format
notes"). There's nothing to build a chain-length-3-or-5 real-data eval set
*from*.

**Why not mix real data (length 1) with synthetic data (length 3, 5):** the
research question is whether success rate degrades *as chain length
increases* — chain length needs to be the only thing that varies across
that comparison. Glaive's tool vocabulary (8 tools: `calculate_bmi`,
`convert_currency`, ...) and phrasing style are both systematically
different from anything synthetic. If chain_length=1 eval used real glaive
data while chain_length=3/5 used a synthetic tool set, any measured
"degradation" would be confounded with a simultaneous tool-domain shift —
a model could fail at length 3 partly (or entirely) because the tools are
unfamiliar, not because the chain got longer. Holding one fixed 6-tool
vocabulary (`harness/tools.py::build_demo_registry()`) constant across all
three chain lengths removes that confound.

This means the eval tool vocabulary is intentionally *not* the training
distribution — eval measures generalization to a fixed, unfamiliar-to-the-model
tool set, consistently across all three conditions and all three chain
lengths. `harness/tasks.py::load_tasks(1, source="glaive_test")` is kept
available as a secondary, single-step-only, in-distribution data point (e.g.
to sanity-check a fine-tuned model didn't also regress on data resembling
what it trained on) — it is not part of the chain-length comparison itself.

**A known scope boundary, not an oversight:** the synthetic tasks are
designed so later steps genuinely follow from earlier ones (e.g. "check the
weather, then convert that reading to Celsius, then compare it to a
threshold" — `convert_temperature` and `compare_numbers` specifically exist
to take a *number* as input so they can consume a prior step's result). The
harness's grader, however, currently checks only that the *tool name*
sequence matches, not that a step's arguments actually derive from an
earlier result — so a model that happened to guess the right tool order
without reading intermediate results would currently score the same as one
that followed the chain properly. `tests/test_integration.py`'s
dependency-feasibility tests demonstrate the designed dependencies are real
and completable via a model that actually threads values through, but
argument-level dependency isn't yet independently *enforced*. Closing that
gap belongs to `evaluation/` (numeric-value extraction or an LLM-judge), not
`harness/`.

## Repo structure

```
configs/                  Model, data, and experiment configuration (YAML)
src/adbench/
  harness/                Tool registry, execution loop, error handling,
                           state tracking — pure Python, no GPU needed
  data/                   Dataset filtering + train/test split
  training/               train.py (all 3 conditions, one script) + losses.py
                           (KD+SFT loss) — config/loss/data-formatting logic
                           unit-tested locally; the training loop itself needs
                           Colab/GPU
  evaluation/             run_eval.py (all 3 conditions x all chain lengths +
                           perplexity, one pass) + metrics.py (scoring) +
                           perplexity.py — same unit-tested-locally split as
                           training/; only checkpoint loading needs Colab/GPU
  analysis/               Teacher/student activation comparison utilities
notebooks/                Colab notebooks that drive the GPU-heavy steps
tests/                    Unit tests (harness, losses, training config/data
                           formatting) — runs locally, no GPU
data/                     Data card; generated splits are gitignored
results/                  Eval outputs (gitignored; regenerated by run_eval.py)
scripts/                  One-off diagnostic scripts (e.g. tokenizer compatibility check)
```

The harness and loss code live in one importable package (`adbench`, via
`pip install -e .`) so local tests, Colab notebooks, and eval scripts all run
the exact same code — no drift between what's tested locally and what runs
on the GPU.

## Reproducing

**Local (VS Code, CPU-only)** — harness development and unit tests:
```bash
pip install -r requirements.txt
pytest
```

**One notebook, resumable (recommended)** — `notebooks/07_final.ipynb` runs data
prep, all three trainings, evaluation and layer analysis in one go on Kaggle
(GPU T4 x2, Internet on). Each finished stage is saved to a private Hugging
Face repo (Kaggle secret `HF_TOKEN`), so a stopped run continues instead of
restarting; see the notebook's first cell. `notebooks/08_results.ipynb` turns
the files in `results/` into the tables and figures below, with no GPU.

**Colab (GPU), stage by stage** — data prep, training, evaluation, layer analysis, in order:
1. `notebooks/00_setup_colab.ipynb` — clone repo, install GPU deps
2. `notebooks/01_data_prep.ipynb` — build the filtered subset + split
3. `notebooks/02_training.ipynb` — all three conditions (base/sft_only/distilled)
   via `training/train.py --condition ...`; see "Training" below
4. `notebooks/03_evaluate.ipynb` — run all 3 conditions through the harness
5. `notebooks/04_layer_analysis.ipynb` — teacher/student activation comparison

## Training

One script, `src/adbench/training/train.py`, runs all three conditions via
`--condition {base,sft_only,distilled}` — this is what guarantees sft_only
and distilled differ *only* in the loss function (see `resolve_training_config`'s
docstring: sft_only's `kd_weight` is forced to 0 regardless of config, not
just defaulted). `base` still goes through the same load-and-save path
(a freshly-initialized, untrained LoRA adapter) rather than being
special-cased, so `evaluation/run_eval.py` loads all three conditions
identically.

Hyperparameters (learning rate, epochs, KD temperature, KD/SFT loss
weighting, ...) are named values in `configs/experiment.yaml`'s `training:`
block, not hardcoded — including a `training.sweep` list of named
overrides for comparing a few settings without editing the file:
```bash
python -m adbench.training.train --condition distilled --sweep higher_kd_temp
```
Training logs `sft_loss` and `kd_loss` separately (`results/training_logs/
<condition>.jsonl`) so convergence can be sanity-checked before spending
eval time on a run that didn't converge — see `notebooks/02_training.ipynb`'s
plotting cell.

The heavy pieces (config resolution, the per-step KD+SFT loss wiring,
glaive-record-to-training-example formatting) are unit-tested locally with
CPU torch and fake/stub models — no GPU, no Unsloth, no real Qwen weights
(`tests/test_train.py`, `tests/test_losses.py`). Only the actual multi-epoch
loop over the real 14B/4B models needs Colab; see `train.py`'s module
docstring for exactly where that boundary is.

## Evaluation

`src/adbench/evaluation/run_eval.py` runs all three conditions x all three
chain lengths + perplexity in one pass:
```bash
python -m adbench.evaluation.run_eval              # all 3 conditions
python -m adbench.evaluation.run_eval --condition sft_only   # just one
```
For each condition: load its checkpoint (same convention as `train.py` —
`base` has one too, an untrained adapter, so all three load identically),
run every `harness.tasks.load_tasks(chain_length, source="synthetic")` task
through `executor.run_task()`, and compute perplexity on the wikitext
sample (`adbench.data.general_eval` — run once first). Writes
`results/eval_results.jsonl` (full per-task, per-step detail),
`results/eval_results.csv` (the same, flattened — summary columns plus a
`steps_json` column so nothing is lost), and `results/eval_summary.json`
(metrics per condition x chain length, `evaluation/metrics.py`).

**Design decision, flagged during scoping:** how to credit a step that
succeeds after a retry (e.g. recovering from an injected error). Full
credit in the primary metric (`per_step_success_rate`) — matches the literal
"of all calls attempted, how many were valid" framing, since a partial-credit
fraction would need its own arbitrary justification — but `clean_step_success_rate`
(succeeded with *no* retry) and `recovery_rate` (of steps forced to fail by
an injected error, how many still succeeded) are reported as *separate*
metrics rather than blended in, specifically so a difference between
"eventually gets it right" and "gets it right immediately" isn't hidden by
one number. Full reasoning, plus a survivorship-bias caveat on pooling
per-step rates across a chain, in `metrics.py`'s module docstring.

Metrics, the model_fn wiring (prompt → generate → decode-new-tokens-only),
and the harness-run loop are unit-tested locally with fake/scripted models —
no GPU (`tests/test_run_eval.py`, `tests/test_metrics.py`,
`tests/test_perplexity.py`). Only checkpoint loading needs Colab.

Also fixed along the way: the classic `wikitext` HF repo's loading script is
no longer supported by `datasets` >= 4 (raises `HfUriError`) — switched to
the maintained `Salesforce/wikitext` mirror (`configs/experiment.yaml`).

## Layer analysis

`src/adbench/analysis/layer_analysis.py` compares teacher (Qwen3-14B) vs a
trained student checkpoint's activations layer by layer — the paper's key
mechanistic figure ("which layers does the gap originate from"). Run via
`notebooks/04_layer_analysis.ipynb`.

**Architecture, verified before writing any hooking code** (not assumed —
see `configs/models.yaml`'s `layer_alignment` comment): teacher has 40
layers, hidden_size 5120; student has 36 layers, hidden_size 2560. Both
depth *and* width differ. Each transformer block's `.self_attn.o_proj`
(attention output) and `.mlp.down_proj` (FFN output) are hooked separately
— confirmed exact module names by reading `transformers`' own
`modeling_qwen3.py` source, not by guessing from a different model family.
`find_decoder_layers()` locates them by name pattern rather than a
hardcoded attribute chain, so it keeps working under PEFT/Unsloth's extra
wrapper layers.

**Flagged design decision — why cosine distance isn't literally "cosine
between two activation vectors" here:** with hidden sizes 5120 vs 2560,
raw-vector cosine similarity is undefined (no valid way to compare vectors
of different length). Both metrics instead operate on (n x n)
sample-pairwise structure, which stays valid across differing widths — CKA
(Kornblith et al.) via Gram matrices, and cosine distance via each model's
*own* pairwise-cosine-similarity matrix (representational similarity
analysis) compared to the other's. Reported side by side as each other's
robustness check, not because either is known to be more correct here.
Both are verified against synthetic tensors with known analytic answers
(identical inputs → 0 divergence; independent random inputs → high
divergence; invariant to rotating either side independently, even at
different widths — the exact property this comparison depends on).

**Memory, flagged per the eval spec:** loading a 14B and a 4B model
simultaneously on a free T4 (16GB) is tight and risky once activation
memory is added on top of ~9.5-10.5GB of 4-bit base weights. Resolved by
never doing it: `extract_and_cache_activations()` loads one model at a
time, writes its activations to disk, and the caller frees it before
loading the other; `compare_cached_activations()` only needs the two
resulting files. Two probe sets — Glaive test-split text (`tool_use`) and
the wikitext sample (`general`, same file `evaluation/perplexity.py` uses)
— so tool-use-specific divergence can be told apart from generic
distillation drift.

51 tests: the divergence metrics against synthetic tensors with known
answers, and the hooking/extraction pipeline against a fake nn.Module
(mirroring Qwen3's real module structure, including one test wrapped in
extra nesting to mimic PEFT) — no GPU, no real model weights needed for any
of it. Only the actual extraction from a real checkpoint needs Colab.

## Status

All six deliverables — harness, dataset prep, training, evaluation, and
layer analysis — are implemented, tested, and have been run end to end once.

### First full run (seed 42, single run)

Full-chain success on the 121 synthetic tasks (fixed six-tool vocabulary the
students were not trained on):

| condition | chain 1 | chain 3 | chain 5 | perplexity |
|---|---|---|---|---|
| base | 93.5% | 70.0% | 74.3% | 18.31 |
| SFT only | 95.7% | 70.0% | 68.6% | 17.35 |
| distilled (KD + SFT) | 100% | 87.5% | 91.4% | 18.46 |

The distilled student is ahead of both other conditions on the multi-step
chains (pooled over chains 3+5: +17 points over base, +20 over SFT-only, with
template-level bootstrap intervals that exclude zero); SFT-only is
indistinguishable from the untouched base. Layer-wise CKA/RSA similarity to the
teacher does **not** separate the conditions on the 50+50 probe texts used
(mean CKA 0.845 for all three), so this run does not support "the student's
layers move toward the teacher's" as the mechanism. Limitations: one seed, 121
tasks from a small set of templates, 640 training examples, mean-pooled probes.
Everything is in `notebooks/08_results.ipynb`; figures in `results/figures/`.

Resolved design questions:
- **Tokenizer compatibility (teacher/student)** — verified shared
  (`scripts/verify_tokenizer_compatibility.py`); logit-level KD is safe.
  Sequence-level distillation kept as a documented fallback in
  `training/losses.py` in case a future model pair doesn't share a vocab.
- **General-LM eval set** — uses a wikitext-2-raw-v1 sample
  (`data/general_eval/`), not the Glaive test split, so it actually measures
  general capability rather than more tool-calling ability.
- **Dataset prepared** — `data/prepare.py` builds the 800-example
  (640/160) subset across 8 deterministically-mockable tools; see
  `data/README.md`'s "Format notes from the raw dataset" for what needed
  reformatting (not just filtering) along the way, and
  `harness/tools.py::build_glaive_registry()` for the tool implementations.
- **Harness complete** — `harness/tasks.py::load_tasks()` implemented
  (`source="synthetic"` for eval, `"glaive_train"`/`"glaive_test"` as a thin
  reader over `data/prepare.py`'s output); see "Why synthetic tasks for
  multi-step eval" above for the eval-vs-training data split rationale.
  6-tool demo vocabulary (up from 4 — see `harness/tools.py`'s module
  docstring for why), 121 hand-designed synthetic tasks across chain lengths
  1/3/5.
- **Training complete** — `training/train.py` runs all three conditions
  from one script via `--condition`; `training/losses.py`'s combined KD+SFT
  loss is implemented and thoroughly unit-tested (dummy tensors, every
  weighting edge case). See "Training" above for what's unit-tested locally
  vs what needs Colab.
- **Evaluation complete** — `evaluation/run_eval.py` runs all three
  conditions x all three chain lengths + perplexity in one pass;
  `evaluation/metrics.py` implements full-chain/per-step success rate,
  clean-vs-recovered step crediting, recovery rate, and error-type
  breakdown. See "Evaluation" above for the per-step-crediting design
  decision and what's unit-tested locally vs what needs Colab.
- **Layer analysis complete** — `analysis/layer_analysis.py` hooks
  attention/FFN outputs at every layer (module names verified against
  `transformers`' own Qwen3 source, not assumed), compares teacher vs any
  student checkpoint via CKA and an RSA-style cosine distance (both needed
  since hidden sizes differ — see "Layer analysis" above), on both a
  tool-use and a general probe set, without ever holding both models in
  GPU memory at once.
- 336 tests passing (`pytest`), ruff-clean, no GPU needed (2 tests touch
  network once, to check against the real Qwen tokenizer, and skip cleanly
  if it's unreachable).

## License

Code is MIT-licensed (see [LICENSE](LICENSE)). The dataset
(`glaiveai/glaive-function-calling-v2`) is Apache-2.0 and is not
redistributed here — only fetched at prep time. Model weights are subject to
their own licenses on Hugging Face.
