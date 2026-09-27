# 3. Method

## 3.1 Models

We study two teacher-student pairs, both from the Qwen3 family [Yang25], to check whether our
findings hold across model scale. The **main pair** distills Qwen3-14B (14.8B parameters) into
Qwen3-4B-Instruct-2507 (4.0B parameters), a 3.7x compression. The **second pair** distills
Qwen3-8B (8.2B parameters) into Qwen3-1.7B (1.7B parameters), a 4.8x compression, and tests
whether the main pair's findings generalize to a weaker student. Both students are fine-tuned
with QLoRA [Dettmers23]: 4-bit-quantized frozen base weights with trainable low-rank adapters
[Hu22] (rank 16, alpha 16, dropout 0, applied to all seven linear projections in each transformer
block — query/key/value/output and gate/up/down). All four models are loaded through Unsloth for
4-bit inference and training on a single NVIDIA T4 GPU. Teacher and student in each pair share an
identical tokenizer (Qwen2Tokenizer, vocabulary size 151,643), verified by exact token-id
agreement on code, JSON, and CJK text samples — this guarantee is what makes logit-level KD (Sec.
3.2) valid without any vocabulary alignment step.

## 3.2 Conditions

Every condition trains the same student model on the same data with the same LoRA configuration
and differs *only* in its loss function, so that any measured difference in downstream behavior
is attributable to the loss, not to a confound in the training recipe:

- **base** — no fine-tuning (a freshly-initialized, mathematically-inert LoRA adapter is attached
  and saved immediately, so it loads through the identical code path as every trained condition).
- **sft_only** — standard supervised fine-tuning: cross-entropy loss on the assistant's
  `<tool_call>{...}</tool_call>` completion only (the system prompt and user turn are masked out
  of the loss).
- **distilled** — combined loss, $L = w_{\text{sft}} L_{\text{CE}} + w_{\text{kd}} L_{\text{KD}}$,
  where $L_{\text{KD}}$ is the standard temperature-softened KL divergence between student and
  teacher next-token distributions [Hinton15] at temperature $T=2$, computed only over the
  completion span (the same mask as sft_only). Default weights $w_{\text{sft}}{=}w_{\text{kd}}{=}0.5$.
- **self_distill** (main pair) / **self_distill_small** (second pair) — identical loss to
  `distilled`, but the "teacher" is a frozen copy of the student's *own* pre-fine-tuning
  checkpoint, never updated during training. At step 0 the self-teacher's logits equal the
  student's own, so this condition isolates the KD loss term's regularizing/anchoring effect from
  any information the teacher's own extra training could contribute — cf. Born-Again Networks
  [Furlanello18] and self-distillation-as-regularization [Zhang19], [Mobahi20].
- **distilled_8b** — for the second pair only: KD from the larger external teacher (Qwen3-8B),
  the second-pair analogue of `distilled`.
- Two secondary controls on the main pair reported alongside the primary comparison:
  **sft_early** (SFT with a single epoch instead of three, to check whether SFT's behavior is an
  overfitting artifact of training length) and **sft_ls** (SFT with label smoothing 0.1, a loss-
  level regularizer that does not require a teacher, as an alternative anchoring mechanism).

For the anchoring-strength experiments (Sec. 4.3), the same self-distillation loss is retrained
at five KD weights — 0, 0.05, 0.1 (second pair only), 0.2, 0.5 (default), 0.8 — holding every
other hyperparameter fixed, so that weight is the only variable across that sweep.

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
condition and are reported separately, Fig. 2, rather than folded into significance tests).
