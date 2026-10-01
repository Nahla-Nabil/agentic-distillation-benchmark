# Reference checklist — for Nahla to verify by hand before citing

Rule (from `docs/WRITING_GUIDE.md` §2.7): every reference is checked manually. For each entry:
open the paper (PDF or official page), confirm the bibliographic details, and confirm that the
specific claim we attribute to it is actually in the paper. Tick the box only when both are true.
A reference stays out of the paper until it is ticked.

Priority order: the first block carries the novelty argument, so check it first.

## Block 1 — the closest prior work (check these first)

- [ ] **[MasonWilliams25]** arXiv:2510.12615 — I. Mason-Williams, G. Mason-Williams,
  H. Yannakoudakis, "A Functional Perspective on Knowledge Distillation in Neural Networks".
  Confirm: (a) they conclude KD acts mainly as a "data-dependent regulariser"; (b) the phrase
  "asymmetric transfer of negative knowledge"; (c) their setups are image/text classification,
  NOT LLM agents or tool use; (d) whether it has since been published at a venue (else cite as
  arXiv preprint).
- [ ] **[Yuan20]** CVPR 2020 — "Revisiting Knowledge Distillation via Label Smoothing
  Regularization". Confirm: a poorly trained teacher still improves the student; soft-target
  regularization matters as much as inter-class information; image classification only.
- [ ] **[Stanton21]** NeurIPS 2021, arXiv:2106.05945 — "Does Knowledge Distillation Really Work?".
  Confirm: full author list; self-distillation improves generalization while fidelity stays low.
- [ ] **[Sultan23]** EMNLP 2023, arXiv:2301.12609 — "Knowledge Distillation ≈ Label Smoothing:
  Fact or Fallacy?". Confirm: text classification; KD and label smoothing move confidence in
  opposite directions; the author argues against KD-as-regularization.
- [ ] **[Cho19]** ICCV 2019, pp. 4793-4801 — "On the Efficacy of Knowledge Distillation".
  Confirm: larger teachers are often not better teachers (capacity mismatch); early-stopped
  teacher helps.
- [ ] **[Busbridge25]** ICML 2025, arXiv:2502.08606 — "Distillation Scaling Laws". Confirm: the
  capacity-gap regime for language-model students; author list.

## Block 2 — LLM fine-tuning and agents

- [ ] **[YangSDFT24]** ACL 2024, pp. 1028-1043 — "Self-Distillation Bridges Distribution Gap in
  Language Model Fine-Tuning". Confirm: self-generated data used to reduce catastrophic forgetting.
- [ ] **[Biderman24]** TMLR 2024, arXiv:2405.09673 — "LoRA Learns Less and Forgets Less".
  Confirm: LoRA underperforms full fine-tuning on the target domain but forgets less.
- [ ] **[Kang25]** NeurIPS 2025, arXiv:2505.17612 — "Distilling LLM Agent into Small Models with
  Retrieval and Code Tools". Confirm: author list; trajectory distillation into 0.5-3B students;
  no self-distillation control.

## Block 3 — from the first literature pass (2026-09-28)

- [ ] [Hinton15] arXiv:1503.02531 — soft-target KD with temperature.
- [ ] [Furlanello18] ICML 2018 — Born-Again Networks.
- [ ] [Zhang19] ICCV 2019 — Be Your Own Teacher.
- [ ] [Mobahi20] NeurIPS 2020 — Self-Distillation Amplifies Regularization in Hilbert Space.
- [ ] [Patil23] arXiv:2305.15334 — Gorilla.
- [ ] [Qin23] arXiv:2307.16789 — ToolLLM.
- [ ] [Patil25] ICML 2025 — BFCL.
- [ ] [Liu23] ICLR 2024 — AgentBench.
- [ ] [Dodge20] arXiv:2002.06305 — fine-tuning variance across seeds.
- [ ] [Zhou25] arXiv:2503.07329 — random-seed effects (author list missing).
- [ ] [Chen23] arXiv:2302.07778 — Measuring the Instability of Fine-Tuning (author list missing).
- [ ] [Yang25] arXiv:2505.09388 — Qwen3 Technical Report.
- [ ] [Hu22] ICLR 2022 — LoRA.
- [ ] [Dettmers23] NeurIPS 2023 — QLoRA.
- [ ] [OLMo25] COLM 2025, arXiv:2501.00656 — 2 OLMo 2 Furious.
