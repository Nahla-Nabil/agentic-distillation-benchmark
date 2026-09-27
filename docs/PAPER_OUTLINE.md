# Paper outline — SCIBT 2027 (IEEE, 4-8 pages)

Status: outline, ready to draft from. Built from `docs/EXPERIMENT_PLAN.md`'s claim/evidence table
and the final crossed-bootstrap pass (both complete as of 2026-09-28). Page budget assumes IEEE
two-column format (~700-800 words/page including one figure or table).

## Working title

**"Anchoring, Not Knowledge: What Distillation Actually Buys Agentic Tool-Use Fine-Tuning"**

Alternatives to consider while drafting: "When Does Knowledge Distillation Help Tool-Use Agents?
A Study of Anchoring vs. Teacher Knowledge"; "The Anchoring Dial: Isolating Regularization from
Knowledge Transfer in Agentic Distillation".

## Abstract (150-200 words, draft last)

One sentence each: problem, method, the three headline numbers (tuned-SFT gap +23.1pt p<.001;
teacher-specific increment n.s. p=.39; symmetric dial result +14.9/-19.9pt), the limitation
(small data, one family), the takeaway (anchor strength must match student competence).

## 1. Introduction (~0.75 page)

- Open on the practical question: teams distilling a small agentic tool-use model from a big
  teacher assume the win comes from transferred knowledge. Does it?
- State the two candidate mechanisms plainly: (a) knowledge transfer — the teacher's logits carry
  information the student lacks; (b) anchoring/regularization — KD's loss term keeps the student
  close to a reference distribution, protecting it from a small dataset's noise, regardless of
  whether that reference is a stronger model.
- Contributions, as a numbered list (reviewers scan this):
  1. A controlled three-way comparison (base / SFT / KD) isolating the loss function as the only
     variable, on a multi-step agentic tool-use harness (not single-turn function calling).
  2. Evidence that most of KD's advantage survives when the "teacher" is the student's own frozen
     base (self-distillation) — the teacher-specific increment is small and not significant.
  3. An anchoring-strength dial (5 KD weights) showing the effect is not fixed: it is a plateau of
     benefit for a competent student and a monotone harm for a weak one, at the *same* weights.
  4. A tool-diversity and training-set-size ablation ruling out two alternative explanations
     (tool scarcity, small-data overfitting) for why SFT is unstable.
- Close with the one-sentence thesis and a forward pointer to Section 3's results.

## 2. Related work (~0.5-0.75 page — needs the literature pass before drafting)

Buckets to fill (search before writing, do not cite from memory):
- Knowledge distillation surveys / foundational KD (Hinton et al. framing already used for the
  loss — cite properly).
- Self-distillation as regularization (Born-Again Networks and successors) — this is the closest
  prior art to the anchoring claim; the contribution here is applying it to *agentic tool-use*
  with controlled ablations, not the regularization idea itself. Scope the novelty claim to that.
- LLM function-calling / tool-use fine-tuning and benchmarks (Glaive, BFCL, ToolBench-style work).
- Agentic multi-step evaluation harnesses (why single-turn accuracy under-measures compounding
  error).
- Small-model fine-tuning instability / LoRA fine-tuning variance across seeds, if literature
  exists — relevant to the SFT-instability finding.

## 3. Method (~1-1.25 pages)

### 3.1 Models and setup
Table: main pair (Qwen3-14B teacher, Qwen3-4B-Instruct-2507 student) and second pair (Qwen3-8B,
Qwen3-1.7B), both Unsloth 4-bit QLoRA. State why a second pair: generalization check across
student competence.

### 3.2 Conditions
The three-way design (base / SFT-only / distilled), identical everywhere except the loss (cite
`losses.py`'s `combined_loss`). Self-distillation condition: KD loss against the student's own
frozen pre-fine-tuning checkpoint — isolates anchoring from knowledge transfer by construction.

### 3.3 Data
Glaive function-calling subset, 640 train / 160 test, 8 curated tools, frozen split. State the
per-tool cap and why (balance).

### 3.4 Evaluation harness
Multi-step agentic loop (call → read result → next step → error recovery), chains of length
1/3/5, on held-out *unseen* tools (121-task primary set) plus an extended 158-task/12-tool set and
the external BFCL benchmark (simple + multiple categories) for generalization beyond the authors'
own task design.

### 3.5 Statistics
Crossed (seed × template) bootstrap for every paired contrast — explain briefly why (seeds and
templates are both sources of variance; naive per-seed CIs understate it). 6000 resamples, 95% CI,
two-sided p from the resample distribution.

## 4. Results (~2-2.5 pages — the paper's core, cite report figures)

Order matters here — lead with the strongest, cleanest result, not chronological experiment order.

### 4.1 KD beats even a *tuned* SFT baseline
Figure: chains-3+5 bar chart (report section 1) + the fair-SFT table (report section 7a). Numbers:
distilled vs tuned SFT +23.1 [+12.0,+34.7] p<.001; self_distill vs tuned SFT +14.7 [+2.7,+28.9]
p=.019. State explicitly: default-recipe SFT's instability (SD 17.2) is an artifact of that one
recipe, not evidence on its own — a gentler recipe removes it (SD→1.3) without closing the gap.

### 4.2 Most of the gain is anchoring, not knowledge
Teacher-specific increment (distilled − self_distill): +4.8pt [-5.9,+16.0], p=.39, not significant
on the 121-task set; consistent pattern on the extended set and BFCL (report table, section 3).
State this as the paper's central negative result, worded to what the data supports.

### 4.3 The anchoring dial: benefit is conditional on student competence
The strongest single figure (report section 6). Same KD weight (0.05): strong student +14.9pt,
weak student -19.9pt. Full dial both directions. This section carries the paper's mechanistic
claim — anchoring value scales with (or against) the student's own pre-anchor competence at the
task, not with anchor strength alone.

### 4.4 Ruling out two alternative explanations
- Tool diversity (2/4/8 training tools): KD flat, SFT unstable at every level — rules out "KD
  matters more when tools are scarce."
- Training-set size (20/40/80 examples/tool): self_distill stable at every scale; SFT's
  instability does not shrink with less data (report section 7b) — report as descriptive,
  flag non-significance honestly, and state what it rules out (a naive small-data-overfitting
  story) without overclaiming a reversal.

### 4.5 External validity
BFCL (external benchmark, external task design) reproduces the ranking; malformed-call rate is
the cleanest single failure-mode signal and orders conditions identically to the primary metric.

## 5. Discussion (~0.5-0.75 page)

- Practical takeaway for practitioners: before reaching for a bigger teacher, try self-distillation
  (train against your own model's frozen outputs) — cheaper, no teacher-hosting cost, and this
  paper's evidence says it captures most of the benefit for a competent student.
- The dial result reframes what "KD hyperparameter tuning" should target: not accuracy directly,
  but matching anchor strength to the student's own baseline competence at the task.
- Why the weak-student failure looks the way it does: anchoring pulls the student toward a
  reference (its own base) that cannot do the task; the failure mode (prose instead of a tool
  call) is literally the base model's own behavior, not a new failure mode introduced by training.

## 6. Limitations (~0.25-0.5 page — be complete, this is a strength if done honestly)

- Single model family (Qwen3); one additional size pair, not an additional family — the second-
  family experiment was scoped as a stretch goal and may or may not have run by submission.
- Small dataset regime (640 training examples) — the data-scale ablation only tests down to 160,
  not up, since the curated tool pool caps the upward direction; conclusions about the mechanism
  are bounded to this regime.
- BFCL evaluated with a simplified checker (name + acceptable-value matching), not the official
  scorer.
- Harness tasks are synthetic (template-generated), not human-authored; mitigated by chain-length
  variation, an extended 12-tool set, and cross-validation against BFCL, but not eliminated.
- Anchoring-dial experiments use self-distillation only (no external-teacher dial); the strong-vs-
  weak-student asymmetry is demonstrated for one anchor source.
- Per-point sample sizes in the dial (some at 1 seed, most at 3) are smaller than the main
  pipeline's 5 seeds — stated per-contrast throughout via the bootstrap CIs, not hidden.

## 7. Conclusion (~0.25 page)

Restate the thesis in one paragraph, plus one forward-looking sentence (e.g., testing the dial
against additional model families, or an adaptive anchor-strength schedule keyed to a cheap
pre-training competence probe).

## References

To be built during the literature pass (Section 2 above). Target 15-25 references for an 8-page
IEEE paper — KD/self-distillation foundations, function-calling/tool-use benchmarks, agentic
eval methodology, and any directly relevant fine-tuning-stability work.

## Figures/tables inventory (map 1:1 to the Artifact report's sections)

| # | Content | Report section | Paper section |
|---|---|---|---|
| 1 | Chains-3+5 bar, main pipeline, 5 conditions | 1 | 4.1 |
| 2 | Success by chain length, line chart | 2 | 4.1 (or appendix) |
| 3 | Malformed-call rate, 3 eval sets | 3 | 4.5 |
| 4 | Tool-diversity ablation bars | 4 | 4.4 |
| 5 | Second-pair summary table | 5 | 4.3 (setup) |
| 6 | Anchoring-dial line chart (both students) | 6 | 4.3 (headline figure) |
| 7 | Fair-SFT table | 7a | 4.1 |
| 8 | Data-scale bars | 7b | 4.4 |
| — | Full bootstrap contrast table | (plan doc) | appendix or supplementary |

Reuse the report's SVG charts as a starting point for camera-ready figures — IEEE wants
vector-friendly, greyscale-safe figures, so re-render with a print-safe palette (not the report's
dark/light theme tokens) before submission.

## Drafting schedule (fits inside 2026-10-05 → 2026-10-17)

| Date | Task |
|---|---|
| 10-05 → 10-07 | Literature pass (Section 2); finalize figures for print |
| 10-07 → 10-10 | Draft Sections 3-4 (method, results) — the parts fully determined by data in hand |
| 10-10 → 10-12 | Draft Sections 1, 5, 6, 7 (intro, discussion, limitations, conclusion) |
| 10-12 → 10-14 | Full read-through, abstract, trim to page limit |
| 10-14 → 10-16 | Nahla's pass + revisions |
| 10-16 → 10-17 | Final formatting, submission |
