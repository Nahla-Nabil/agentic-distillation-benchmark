# Experiment plan — closing the paper's gaps before the 2026-10-05 GPU cutoff

Status of the evidence as of the last update: see "Where we stand". Paper deadline 2026-10-17
(SCIBT 2027, IEEE, 4-8 pages); no new GPU experiments after **2026-10-05**.

## Thesis the paper defends

> In agentic (tool-use) fine-tuning on a small dataset, most of knowledge distillation's advantage
> over plain SFT comes from **anchoring** the student to a reference distribution (regularisation),
> not from the teacher's extra knowledge — and the benefit is bounded by the student's own competence.

Claims must be worded to what the data supports ("the teacher-specific increment is small and not
significant in our setting"), never "KD transfers no knowledge".

## Where we stand (claim -> evidence -> status)

| Claim | Evidence | Status |
|---|---|---|
| KD >> SFT on unseen-tool chains, and is far more stable across seeds | 5 seeds main pair; chains 3+5: distilled ~90 (SD ~1-2) vs SFT variants 62-66 (SD 16-27) | solid |
| Most of the gain is anchoring, not the teacher | self_distill ~ distilled; teacher-specific extra +2.7, n.s. (crossed bootstrap seeds x templates) | solid for this pair |
| Holds on other benchmarks | extended synthetic set (158 tasks/12 tools); BFCL simple+multiple (SFT degrades robustness) | solid (BFCL uses our simplified checker) |
| Benefit is NOT a tool-scarcity effect | ablation complete: distilled 93.7/91.7/93.9 at 2/4/8 tools; SFT 78.5/75.8/75.5; interaction -4 [-51,+44] | solid negative |
| Benefit depends on student competence | second pair (Qwen3-1.7B / 8B): distilled_8b ~ self_distill_small ~ base (0.32-0.44), SFT 0.83-0.88; failures are prose answers instead of tool calls | consistent, **mechanism unproven** |

## Known weaknesses and the experiment that answers each

| # | Weakness | Experiment | Decision it drives |
|---|---|---|---|
| W1 | SFT baseline may simply be untuned | `sft_fair`: sft_only with lr 5e-5 (`lower_lr`) and lr 5e-5 + 1 epoch (`lower_lr_1ep`), main pair, 3 seeds | If SFT stays unstable (SD >> KD's) the claim gets stronger. If it stabilises near KD, reframe: "KD ~ well-tuned SFT" and report it. |
| W2 | "anchoring" is a hypothesis, not shown | `dial_main`: self_distill at KD weight 0.2 / 0.8 (0.5 exists), main pair, 3 seeds — expect a monotone stability/accuracy trade-off | Gives a dose-response curve for the anchoring claim. |
| W3 | second-pair failure mechanism unproven | `dial_small_probe`: self_distill_small at KD weight 0.2 / 0.05, seed 0; then `dial_small_full` (seeds 1-2) for the setting that works | If lowering the anchor rescues the weak student -> "anchor strength must match student competence" is demonstrated. If not -> report as a boundary of the method. |
| W4 | small data is the assumed regime | `data_scale` (to build): sft_only vs self_distill at 160 / 320 / 640 training examples (per-tool target 20 / 40 / 100), main pair | Tests the original regularisation-against-overfitting story directly. Upward scaling is capped by the dataset (scarcest tool has only 183 kept examples), so we scale DOWN. |
| W5 | one model family | `second_family` (to build, stretch): Llama-3.2-3B-Instruct student / Llama-3.1-8B-Instruct teacher (same tokenizer, vocab 128256 both, so KD logits align; gated repos -> needs HF access). Fallback: Qwen2.5-3B/7B needs vocab slicing (151936 vs 152064). | Only launched if Rounds 1-2 finish by 2026-10-01. |
| W6 | BFCL checker is simplified | not fixable in time (needs re-evaluation with the official scorer); state as a limitation | — |

Never run the same setting on two accounts: the two accounts always run DIFFERENT jobs.

## Schedule (two Kaggle accounts in parallel; both have ~30 h/week)

Each run = open notebooks/16_sweeps.ipynb (Import from local file), edit the first cell's
`EXPERIMENT`, Run All. Results land on Hugging Face automatically; send the last 15 log lines back.

| Round | Account A | Account B | ~time | Then |
|---|---|---|---|---|
| 1 | `sft_fair` | `dial_small_probe` | 2 h / 3.3 h | analyse W1 and W3; choose `SMALL_SWEEP` |
| 2 | `dial_main` | `dial_small_full` (only if the probe rescued the student; else the seed-1/2 repeat of the better setting) | 3.5 h / 3.3 h | analyse W2 |
| 3 | `data_scale` (after it is built) | `second_family` (stretch) | 4-6 h / 8-10 h | analyse W4/W5 |
| — | **2026-10-05: GPU cutoff** | | | writing only |

Rules: (1) each round's results are analysed before the next is chosen; (2) a job that fails is
diagnosed from the loss log before re-running; (3) all numbers go into project memory the day they
arrive; (4) confirm with Nahla before every `git push`.

## Analysis to run once the data is in (all CPU, local)

- Crossed (seeds x templates) bootstrap for every paired contrast; report CIs and p, and say plainly
  which contrasts are not significant.
- Failure-mode counts (malformed call vs wrong tool vs prose answer) per condition.
- One figure per claim: (a) chains 3+5 by condition with per-seed dots; (b) tool-diversity;
  (c) anchoring dial (KD weight vs accuracy and vs seed SD, both students); (d) second pair.
- Move the scratchpad analysis scripts into `scripts/` once numbers are final.

## Paper outline (4-8 pages, IEEE)

1. Introduction — the question and the thesis above.
2. Setup — pairs, data, conditions, harness, metrics, statistics.
3. Results — (i) KD vs SFT and stability, (ii) anchoring vs teacher knowledge, (iii) external
   benchmarks, (iv) tool diversity and data scale, (v) when anchoring fails (second pair, dial).
4. Analysis — failure modes; (optional) layer-wise CKA if there is time.
5. Limitations — simplified BFCL checker, synthetic harness, one family unless W5 lands, small data.
6. Conclusion.

Writing starts 2026-10-05 at the latest; a full draft by 2026-10-12; final polish and references
by 2026-10-16. A literature check (self-distillation / anchoring / KD for tool use) is done in
parallel to the GPU rounds — novelty is claimed only for what that check supports.
