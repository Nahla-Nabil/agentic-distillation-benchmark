# III. Methodology

<!-- Revision status (2026-10-02): §A and §B revised with Nahla and approved. §3.2 onward are the OLD
first draft (written before docs/WRITING_GUIDE.md and docs/VOICE_PROFILE.md) and are being
revised one section at a time. -->

## A. Teacher–Student Pairs

This study uses four teacher–student pairs from two model families (Table I). The main pair
distills Qwen3-14B into Qwen3-4B-Instruct-2507 [Yang25]. The second pair keeps the same family and
reduces the student to Qwen3-1.7B, with Qwen3-8B as teacher, to examine whether the findings depend
on the student's starting ability. The third and fourth pairs use OLMo-2 [OLMo25], a family with a
different architecture and training pipeline, so that the findings can be checked outside Qwen.
OLMo-2 was selected because its weights, training data and code are fully open. Llama and Gemma
models were not used because they require a per-account license request on Hugging Face, which
prevents reproduction from public artifacts alone.

TABLE I: TEACHER–STUDENT PAIRS

| Pair | Teacher | Student | Ratio |
|---|---|---|---|
| Main | Qwen3-14B | Qwen3-4B-Instruct-2507 | 3.7× |
| Second | Qwen3-8B | Qwen3-1.7B | 4.8× |
| Third | OLMo-2-7B-Instruct | OLMo-2-1B-Instruct | 7× |
| Fourth | OLMo-2-13B-Instruct | OLMo-2-7B-Instruct | 1.9× |

<!-- TODO (Round 4, notebook 18): keep the Fourth row only if that run completes; otherwise
change "four pairs" to "three pairs" above and drop the row. -->

Every student is fine-tuned with QLoRA [Dettmers23]: the base weights stay frozen in 4-bit
precision and only low-rank adapters [Hu22] are trained.

**Adapter configuration:**
- Rank 16, alpha 16, dropout 0
- Applied to all seven linear projections in each block (q, k, v, o, gate, up, down)
- Identical in every condition and every pair

Logit-level distillation compares the teacher's and the student's next-token distributions token by
token, which is meaningful only when both models share one vocabulary. For each pair, the
tokenizers were checked for identical vocabulary size, special tokens and token ids on code, JSON,
tool-call and non-Latin samples (Qwen3: 151,643 tokens; OLMo-2: 100,278 tokens), so no vocabulary
alignment step is needed.

## B. Training Conditions

All conditions train the same student on the same data with the same adapter configuration and
differ only in the loss function, so that any difference in behaviour can be attributed to the loss
rather than to the training recipe.

- **Base:** no fine-tuning. A freshly initialised adapter is attached and saved without training,
  so this condition is loaded through the same code path as every trained condition.
- **SFT:** standard supervised fine-tuning. Cross-entropy is computed only on the assistant's
  tool-call completion; the system prompt and the user turn are masked.
- **Distilled:** cross-entropy combined with a distillation term,

  L = w_sft · L_CE + w_kd · L_KD,   (1)

  where L_KD is the temperature-softened Kullback–Leibler divergence between the teacher's and the
  student's next-token distributions [Hinton15], computed on the same completion tokens as L_CE.
  The default setting is T = 2 and w_sft = w_kd = 0.5.
- **Self-distilled:** the same loss as Distilled, but the teacher is a frozen copy of the student
  itself before fine-tuning. At the first step its distribution equals the student's, so this term
  carries no information from a stronger model; it can only hold the student close to where it
  started. This condition separates the anchoring effect of the distillation loss from the
  knowledge of a larger teacher, following earlier self-distillation work [Furlanello18],
  [Zhang19], [Mobahi20].

To check that the SFT baseline is not simply under-tuned, three further SFT recipes are trained on
the main pair: one epoch instead of three, a lower learning rate (5×10⁻⁵ instead of 2×10⁻⁴), and
the lower rate with one epoch. A label-smoothing control was also run but is not reported, because
only two seeds completed and its training was unstable.

To measure how the strength of the anchor affects the outcome, Self-distilled is retrained with
w_kd ∈ {0.05, 0.1, 0.2, 0.5, 0.8} and w_sft = 1 − w_kd on the main and second pairs; w_kd = 0 is SFT.

**Shared hyperparameters:**
- Learning rate 2×10⁻⁴, cosine schedule, 3% warm-up, no weight decay
- 3 epochs, batch size 2 with 4 accumulation steps (effective batch 8), 240 optimiser steps
- Five seeds (0–4) for the main comparison, three seeds for every other experiment

<!-- Code-name mapping (not for the paper): SFT = sft_only; Distilled = distilled / distilled_8b /
distilled_olmo / distilled_olmo7; Self-distilled = self_distill / self_distill_small /
self_distill_olmo / self_distill_olmo7. The one-epoch SFT recipe is sft_early; the lower-rate
recipes are sweeps lower_lr / lower_lr_1ep; label smoothing is sft_ls. Main-pair dial points:
0.05, 0.2, 0.5, 0.8; second-pair dial points: 0.05, 0.1, 0.2, 0.5. -->

## 3.3 Training data

Training examples are drawn from the Glaive function-calling v2 corpus [Qin23]-lineage synthetic
data (single-tool, single-turn function-call examples; we did not use ToolLLM/ToolBench directly,
noting the difference in scale and provenance). From the raw corpus we keep only examples whose
tool name and argument-key signature match one of eight curated, deterministically-mockable tools
(`calculate_bmi`, `calculate_tip`, `calculate_discount`, `calculate_age`, `calculate_distance`,
`convert_currency`, `generate_random_number`, `get_stock_price`), capped at 100 examples per tool
and split 80/20 train/test (640 train / 160 test), with the test split frozen at creation and
never used for anything but final held-out checks. Two ablations vary this default: a
**tool-diversity ablation** re-derives the same total 800-example subset from only the 2 or 4
most-available of the eight tools (per-tool cap scaled up to hold total example count constant),
and a **training-set-size ablation** takes seeded, nested subsamples of the default 640-example
train split at 20 and 40 examples per tool (160 and 320 total), against the default 80/tool (640)
as the upper end.

## 3.4 Evaluation harness

We evaluate with a synthetic multi-step agentic harness, not single-turn function-calling
accuracy: a task specifies a chain of 1, 3, or 5 sequential tool calls, and the model must, at
each step, read the running conversation (including prior tool results), emit a well-formed
`<tool_call>` matching the expected next tool and arguments, and recover from up to 2 retries on
an injected malformed-call or wrong-argument error before the step is scored a failure. This
design follows the broader argument in agentic-evaluation work [Liu23] that single-turn accuracy
under-measures the compounding error of a multi-step trajectory. All evaluation happens on tools
*unseen* during training — the primary set is 121 held-out tasks (30 per chain length, plus
single-step held-out-tool checks) over the same eight-tool registry; an extended set adds six
further tools and longer system prompts (158 tasks total across 38 templates) to check that
findings are not an artifact of the eight-tool vocabulary; and the external Berkeley Function
Calling Leaderboard [Patil25] ("simple" and "multiple" categories, 500 examples each, scored with
a simplified name-plus-acceptable-value checker rather than BFCL's own AST-based scorer) checks
generalization to a benchmark we did not design.

## 3.5 Statistics

Every paired contrast (e.g., `distilled` vs. `sft_only`) is reported as a point estimate with a
95% confidence interval and a two-sided p-value from a **crossed bootstrap over seeds and task
templates**: for each of 6000 resamples, we independently resample (with replacement) the set of
training seeds and the set of task templates, average the paired per-template, per-seed success
difference over that resample, and take the 2.5th/97.5th percentiles of the resulting
distribution as the CI (p is twice the smaller tail proportion on either side of zero). We use
this crossed design, rather than a per-seed-only bootstrap, because both seed (training-run
randomness) and template (task-design randomness) are independent sources of variance in our
setup, and a per-seed-only interval would understate the true uncertainty. All statistics in
Sec. 4 are computed on chains of length 3 and 5 pooled (chain-1 tasks are near-ceiling for every
condition and are reported separately, Fig. 2, rather than folded into significance tests), except
for the third (OLMo-2) pair, where chains 3 and 5 are at floor (~0%) for every condition including
`sft_only` — a capability ceiling at this model size and training budget, not a KD-specific
result — so that pair's contribution is reported on the overall/chain-1 metric only (Sec. 4.5),
with no bootstrap test run where there is no variance to test.
