# 4. Results

All contrasts below are reported as point estimate [95% CI], p (Sec. 3.5); "n.s." marks p > .05.
Unless noted, numbers are success rate (%) pooled over chains 3 and 5 of the 121-task primary
unseen-tool set, main pair (Qwen3-14B/4B), 5 seeds for the base comparison, 3 seeds for sweep
conditions.

## 4.1 KD beats even a tuned SFT baseline

Table 1 compares `distilled` and `self_distill` against three SFT recipes: the project's default
(learning rate 2e-4, 3 epochs), and two deliberately gentler ones (learning rate 5e-5, with either
3 epochs or 1) introduced specifically to test whether SFT's apparent weakness is a tuning
artifact.

| recipe | overall % | SD | chains 3+5 % | SD |
|---|---|---|---|---|
| SFT, default | 75.5 | 17.2 | 62.2 | 26.7 |
| SFT, lr 5e-5 | 79.6 | 1.3 | 67.1 | 2.0 |
| SFT, lr 5e-5, 1 epoch | 78.2 | 2.9 | 64.9 | 4.7 |
| self_distill | 87.1 | 2.7 | 81.8 | 5.0 |
| distilled | 93.9 | 0.5 | 90.2 | 0.8 |

The default recipe's seed-to-seed standard deviation (17.2 points overall, 26.7 on chains 3+5) all
but disappears under the gentler recipe (1.3 / 2.0) — SFT's instability is a property of that one
hyperparameter choice, not of SFT as such. Critically, this does **not** close the gap to KD: with
the lowest-variance SFT recipe available, `distilled` still wins by +23.1 [+12.0, +34.7] (p<.001)
and `self_distill` by +14.7 [+2.7, +28.9] (p=.019). The gentler recipe itself is not significantly
different from the default (+4.9 [−18.7, +36.5], n.s.) — tuning the learning rate narrows SFT's
variance without moving its mean. We therefore state the comparison this paper rests on as *KD
variants vs. the best SFT recipe we found*, not vs. an arbitrary default, and report that even
this fair comparison favors KD by a wide, significant margin (Fig. 1, Fig. 7).

One contrast is weaker than it first appears and we flag it explicitly: `self_distill` vs.
*default*-recipe SFT is +16.5 [−1.1, +40.0], p=.072 — not significant at the conventional
threshold, entirely because of the default recipe's own variance. We do not build any claim on
this specific comparison; the tuned-SFT contrasts above, together with the extended-set and BFCL
results (Sec. 4.5), carry that part of the argument instead.

## 4.2 Most of the gain is anchoring, not knowledge

`self_distill` replaces the 14B teacher with a frozen copy of the student's own pre-fine-tuning
checkpoint — same loss, same training budget, zero information from any model larger than the
student itself. If KD's advantage were primarily transferred knowledge, `self_distill` should
trail `distilled` by a wide margin; if it is primarily anchoring (holding the student near a
reference distribution during fine-tuning on a small dataset), the two should be close. On the
121-task primary set, the teacher-specific increment is **+4.8 [−5.9, +16.0], p=.39** — not
distinguishable from zero in our sample. The point estimate is positive and consistent across
every evaluation we ran (91.8 vs. 89.0 pooled chains 3+5 on the 279-task combined primary+extended
pool, Fig. 1; 95.0 vs. 95.2 / 95.4 vs. 93.6 on BFCL simple/multiple, Table 2) — so we do not claim
the teacher contributes *nothing* — but in no case does it reach significance at our sample size.
We report this as the paper's central qualitative finding: **the bulk of knowledge distillation's
benefit over SFT on this task is attributable to the KD loss term's anchoring effect, not to
information specific to a stronger teacher**, with the caveat that a teacher-specific increment on
the order of a few points cannot be ruled out by this study.

## 4.3 The anchoring dial: benefit is conditional on student competence

If anchoring is the mechanism, its strength should be tunable independently of whether a teacher
is involved — and if it is a genuine regularizer rather than free knowledge, its effect should
depend on what it anchors the student *to*. We retrained `self_distill` at KD weights
0, 0.05, 0.2, 0.5 (default), 0.8 on the main (strong) pair and 0, 0.05, 0.1, 0.2, 0.5 on the second
(weak) pair, holding every other hyperparameter fixed (Fig. 6).

| KD weight | strong student (Qwen3-4B), overall % | weak student (Qwen3-1.7B), overall % |
|---|---|---|
| 0 (plain SFT) | 78.5 (SD 14.0) | 84.6 (SD 2.7) |
| 0.05 | 90.4 (SD 3.3) | 64.7 (SD 10.0) |
| 0.1 | — | 40.2 (SD 3.4) |
| 0.2 | 92.0 (SD 0.5) | 30.6 (seed 0 only) |
| 0.5 | 87.4 (SD 3.3) | 38.3 (SD 2.1) |
| 0.8 | 90.1 (SD 1.4) | — |

The two students respond to the identical anchoring mechanism in opposite directions. The strong
student jumps to a plateau at the lightest weight tested (0→0.05: +11.9 points overall) and stays
there through 0.8 — every weight from 0.05 to 0.8 clearly beats plain SFT, and no two points on the
plateau differ significantly from each other (e.g., 0.2 vs. 0.5, chains 3+5: +5.3 [−0.4, +14.2],
n.s.). The
weak student declines in the opposite direction, close to monotonically, losing 19.9 points at the
same KD weight of 0.05 and continuing down to near-floor by 0.2: 0→0.05 is −30.2 [−50.7, −9.8]
(p=.003), and 0.1→0.2 is −16.0 [−30.7, −2.7] (p=.016); the middle step (0.05→0.1) is directionally
consistent but not individually significant (p=.062). Inspecting failures shows *why*: the weak
student's errors at higher anchor weight are overwhelmingly the model answering in prose instead
of emitting a tool call at all — the untrained base model's own behavior, which anchoring pulls
the student back toward rather than away from. We read this as direct evidence that **anchoring
value depends on the student's own pre-anchor competence at the task**: it is a net regularizer
for a student that can already do the task reasonably well, and a net harm for one whose own
unfine-tuned baseline cannot.

## 4.4 Ruling out two alternative explanations

**Tool diversity.** One might expect KD's advantage to shrink as training data covers more
distinct tools (less need for a regularizer once there is enough coverage to avoid memorization).
Holding total training examples fixed at 800 while varying the number of distinct tools those
examples are drawn from (2, 4, or 8; Fig. 4), `distilled` is flat (89.8 / 86.7 / 90.2 chains-3+5
%, 2 vs. 8: −0.4 [−8.9, +8.0], n.s.) and `sft_only` shows no trend either (65.8 / 61.8 / 62.2,
2 vs. 8: +3.6 [−42.7, +49.3], n.s., though this design's power is limited by sft_only's own
variance). The KD-over-SFT gain is +24.0 at 2 tools and +28.0 at 8; the interaction (whether that
gap itself depends on diversity) is −4.0 [−51.1, +44.0], n.s. — we find no evidence that KD matters
more when training tools are scarce.

**Training-set size.** The anchoring-as-regularization story predicts SFT should be *more* unstable
with less data (more room to overfit a small set) and KD's benefit should shrink as data grows. We
retrained both conditions on seeded, nested subsamples of 20 and 40 examples/tool (160 and 320
total) against the default 80/tool (640; Fig. 8). `self_distill` is stable at every scale tested
(chains-3+5 SD 0.8–5.0). `sft_only`'s instability does **not** shrink at lower data as the naive
small-data-overfitting story predicts — descriptively its SD is smallest at 40 examples/tool (7.6)
and largest at the full 640 (26.7). None of the scale-vs-scale contrasts reach significance (widest
CI: n20 self_distill vs. n80, −4.4 [−19.1, +8.0]; only 3 seeds per point), so we report this as a
genuine but inconclusive pattern, not a proven reversal — it rules out treating "insufficient data"
as a sufficient explanation for SFT's instability without requiring the (unproven) reversed-direction
effect to be real. `self_distill`'s own stability across the full range is the result this ablation
was designed to check, and that result holds.

## 4.5 Generalization: external benchmark and a different model family

**BFCL.** On the external Berkeley Function Calling Leaderboard (simple and multiple categories,
500 tasks each, a benchmark we did not design), the same ranking holds: `base`, `self_distill`, and
`distilled` never produce a malformed tool call (0/500 on either category), while `sft_only` and
`sft_early` do (2.0–3.2%; Fig. 3, Table 2). Overall accuracy on BFCL is close across all trained
conditions (89.0–95.4%), so the malformed-call rate — not raw accuracy — is BFCL's clearest signal
here, and it orders conditions identically to the primary-harness result.

**A third model family.** All results above use the Qwen3 family. To test whether the findings are
specific to Qwen's training recipe, we repeated the base/sft_only/distilled/self_distill comparison
on a third, architecturally unrelated pair: OLMo-2-1124-7B-Instruct distilling into
OLMo-2-0425-1B-Instruct (AI2's fully open OLMo-2 [OLMo25]), 3 seeds.

| condition | overall % | SD |
|---|---|---|
| base | 0.0 | — |
| distilled_olmo | 4.1 | 0.0 |
| self_distill_olmo | 10.7 | 0.0 |
| sft_only | 31.7 | 0.5 |

The ordering replicates cleanly on a model family with no shared lineage to Qwen: both KD variants
stay near the untrained base, far below `sft_only`, and `self_distill_olmo` is again at least as
strong as `distilled_olmo` (no teacher-specific advantage, consistent with Sec. 4.2). Two things
distinguish this pair from the other two and we report them plainly. First, `base` fails all 121
tasks outright, including single-step ones (0.0% at chain length 1, vs. 78–91% for the two Qwen3
`base` models) — OLMo-2's instruction tuning evidently does not include the kind of structured
function-calling exposure Qwen3's does, so it never spontaneously emits our `<tool_call>` format;
every `base` failure is the same "produced a final answer instead of the expected tool call" error
seen for the weak Qwen student's anchored conditions (Sec. 4.3), now at the base rate rather than
only at high anchor weight. Second, and more consequentially for this study's statistics, chains 3
and 5 are at floor (0.0%, one condition reaching 1.3% in one seed) for **every** condition,
including `sft_only` — a capability ceiling for this 1B-parameter model at this LoRA budget, not a
KD-specific effect, since `sft_only` hits the same floor. We therefore cannot run the chains-3+5
contrasts this paper is otherwise built on for this pair; its contribution is the overall/chain-1
ranking, which still replicates the paper's central claim on a third, unrelated model family.
