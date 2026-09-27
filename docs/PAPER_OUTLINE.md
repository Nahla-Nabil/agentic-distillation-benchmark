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

## 2. Related work (~0.5-0.75 page)

Literature pass done 2026-09-28 (WebSearch, not from memory — see References for exact citations).
Five buckets, each with the one-sentence framing to use when citing it:

- **Foundational KD** [Hinton15]. Standard soft-target KD loss and temperature — cite for the loss
  formulation (`losses.py::kd_divergence`), not as a claim about mechanism.
- **Self-distillation as regularization — the closest prior art to the anchoring claim.**
  [Furlanello18] (Born-Again Networks: a student identically parameterized to its teacher, trained
  by KD from it, outperforms the teacher) and [Zhang19] (splits one network into sections and
  distills the deep section into shallow ones, no external teacher) established empirically that a
  "teacher" is not required for KD to help. [Mobahi20] gives the only theoretical account we found:
  self-distillation provably contracts the effective hypothesis space (fewer usable basis
  functions per round), a regularization effect, and further rounds move from under- to over-
  regularizing. **Scope the novelty claim precisely against this bucket**: the regularization-via-
  self-distillation *idea* is established (2019-2020, vision/tabular settings); this paper's
  contribution is (a) testing it in agentic multi-step tool-use, an interactive, compounding-error
  setting distinct from single-forward-pass classification, and (b) the anchoring-strength dial
  showing the same mechanism helps a competent model and actively harms a weak one at the same
  weight — a symmetry [Mobahi20]'s theory does not by itself predict (their "too many rounds hurt"
  finding is about anchor *iteration count* holding student capability fixed, not about anchor
  *weight* varying with the student's own baseline competence). Say this plainly in Section 2, not
  just in Discussion.
- **LLM function-calling fine-tuning and data.** [Patil23] (Gorilla) fine-tunes LLaMA for accurate
  API calls and introduces APIBench; [Qin23] (ToolLLM/ToolBench) scales this to 16k+ real APIs with
  auto-generated multi-step instructions. The training data here (Glaive function-calling v2) is a
  smaller, simpler single-tool-per-example corpus in the same lineage — cite both for the
  fine-tuning-for-tool-use framing, note the scale difference honestly.
- **Evaluation: single-turn vs. multi-step/agentic.** [Patil25] (BFCL) is the external benchmark
  used here (simple + multiple categories); [Liu23] (AgentBench) established multi-turn,
  multi-environment agent evaluation as necessary because single-turn accuracy does not capture
  compounding error over a trajectory — cite for the methodological argument behind the
  chain-length design (Section 3.4), even though this paper's own harness is a separate,
  purpose-built synthetic one, not AgentBench's environments.
- **Fine-tuning instability across random seeds.** [Dodge20], [Zhou25], and [Chen23] each show
  pretrained-LM fine-tuning varies substantially across seeds, worse on small datasets — cite as
  the general phenomenon behind Section 4.1/7a's SFT seed-variance finding; note none of these are
  agentic/tool-use settings, so this paper extends the observation to that setting and adds the "a
  gentler recipe removes the instability, but the KD gap doesn't close" result on top of it.

Base-model and method citations to include but not build a Related Work paragraph around: [Yang25]
(Qwen3 technical report, the teacher/student model family), [Hu22] (LoRA), [Dettmers23] (QLoRA) —
cite in Section 3.1 (Method) where the models and training method are introduced, not here.

## 3. Method (~1-1.25 pages)

### 3.1 Models and setup
Table: main pair (Qwen3-14B teacher, Qwen3-4B-Instruct-2507 student), second pair (Qwen3-8B,
Qwen3-1.7B), and third pair (OLMo-2-1124-7B-Instruct, OLMo-2-0425-1B-Instruct — a different model
family from Qwen), all Unsloth 4-bit QLoRA. State why a second pair: generalization check across
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

- Model-family coverage: two Qwen3 size pairs plus one different-family pair (OLMo-2, notebook
  17/Round 3, running) — still only two families total; update this bullet with the OLMo-2 result
  once it lands (either it replicates the main finding, strengthening the generality claim, or it
  doesn't, which is itself reportable and goes in Sec. 4/5, not just here).
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

Literature pass done 2026-09-28. Citation keys match Section 2's bracketed references. Verify each
against the venue's final (non-arXiv) publication before submission where one exists (several of
these are ICML/ICLR/NeurIPS papers with arXiv preprints); IEEE reference formatting is applied at
draft time, not here.

- **[Hinton15]** G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural
  Network," arXiv:1503.02531, 2015.
- **[Furlanello18]** T. Furlanello, Z. C. Lipton, M. Tschannen, L. Itti, and A. Anandkumar,
  "Born-Again Neural Networks," in Proc. ICML, 2018, pp. 1607-1616. arXiv:1805.04770.
- **[Zhang19]** L. Zhang, J. Song, A. Gao, J. Chen, C. Bao, and K. Ma, "Be Your Own Teacher: Improve
  the Performance of Convolutional Neural Networks via Self Distillation," in Proc. ICCV, 2019,
  pp. 3713-3722. arXiv:1905.08094.
- **[Mobahi20]** H. Mobahi, M. Farajtabar, and P. L. Bartlett, "Self-Distillation Amplifies
  Regularization in Hilbert Space," in Proc. NeurIPS, 2020. arXiv:2002.05715.
- **[Patil23]** S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large Language Model
  Connected with Massive APIs," arXiv:2305.15334, 2023.
- **[Qin23]** Y. Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+
  Real-world APIs," arXiv:2307.16789, 2023.
- **[Patil25]** S. G. Patil et al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool
  Use to Agentic Evaluation of Large Language Models," in Proc. ICML, 2025.
  https://proceedings.mlr.press/v267/patil25a.html
- **[Liu23]** X. Liu et al., "AgentBench: Evaluating LLMs as Agents," in Proc. ICLR, 2024.
  arXiv:2308.03688.
- **[Dodge20]** J. Dodge, G. Ilharco, R. Schwartz, A. Farhadi, H. Hajishirzi, and N. Smith,
  "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early
  Stopping," arXiv:2002.06305, 2020.
- **[Zhou25]** [author list to verify from the paper], "Assessing the Macro and Micro Effects of
  Random Seeds on Fine-Tuning Large Language Models," arXiv:2503.07329, 2025.
- **[Chen23]** [author list to verify from the paper], "Measuring the Instability of
  Fine-Tuning," arXiv:2302.07778, 2023.
- **[Yang25]** A. Yang et al. (Qwen Team), "Qwen3 Technical Report," arXiv:2505.09388, 2025.
- **[Hu22]** E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in Proc.
  ICLR, 2022. arXiv:2106.09685.
- **[Dettmers23]** T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient
  Finetuning of Quantized LLMs," in Proc. NeurIPS, 2023. arXiv:2305.14314.
- **[OLMo25]** Team OLMo et al. (P. Walsh, L. Soldaini, D. Groeneveld, K. Lo, et al.), "2 OLMo 2
  Furious," in Proc. COLM, 2025. arXiv:2501.00656. (Third-pair model family — chosen for being
  fully open: weights, training data, and code all public, unlike Llama/Gemma which are gated.)

Two entries ([Zhou25], [Chen23]) need their full author lists confirmed from the paper itself
before the reference list is finalized — the search gave the paper and arXiv id but not a
complete author list. 14 references now; add 2-4 more if the drafted Related Work needs finer
distinctions (e.g. a second agentic-eval benchmark alongside AgentBench, or a KD-for-LLMs
survey) once Section 4 is drafted and it's clear which claims need another citation.

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
