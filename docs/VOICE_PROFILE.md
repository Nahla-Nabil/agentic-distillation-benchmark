# Nahla's writing voice — reference for drafting `paper/`

Derived 2026-10-02 from two of her own published-style IEEE papers (both hers, per Nahla):
"Random Forest Weight Auditing for Gulf Sustainability Indices" (a104) and "Hybrid Edge-Cloud
Federated Learning Framework for Predictive Maintenance in Industrial IoT" (a72). Use together with
`docs/WRITING_GUIDE.md` (binding rules); where the two disagree, see "Resolved conflicts" below.

## Signature patterns to reproduce

- **Abstract:** opens with the problem in one or two plain sentences, then "In this study, ...",
  then a dense run of concrete numbers (her a104 abstract carries six numbers). No citations.
- **Introduction:** concrete context first (a fact about the domain), then the gap stated plainly
  ("Neither method inquires as to whether ..."), then the method in one sentence, then explicit
  numbered contributions — either inline "This study differs from previous research in three ways:
  (1) ...; (2) ...; and (3) ..." or a bulleted "Main contributions:" list.
- **Related Work:** lettered subsections (A., B., ...) and a **summary table**
  (Study | Method | Key Finding | Limitation) with "This study" as the last row. A closing
  "Research Gap" paragraph that lists what no prior work does simultaneously.
- **Methods:** lettered subsections; hyperparameters as short bold-labelled lists; numbered
  equations; **explicit justification of what was not done and why** ("We did not use SHAP values
  because ..."; "Permutation importance was omitted because ..."); a subsection that confronts a
  known weakness of the method head-on ("MDI: Rationale and Known Bias").
- **Results:** subsections that mirror Methods; every subsection anchored on a numbered table or
  figure; short sentences that state the number, then (at most) one sentence on what it shows.
- **Conclusion:** restates the main numbers, the practical conclusion, then a limitation paragraph
  and concrete future work in the same section; a separate "Code Availability" section with the
  GitHub URL.
- **Register:** medium-length sentences joined with commas and colons; a mix of passive voice
  and "this study"; "we" used sparingly; plain verbs. Em-dashes are essentially absent from her
  prose: do not use them.

## Resolved conflicts with `docs/WRITING_GUIDE.md`

- **Methods tense.** The guide (drawn from medical-writing sources) asks for simple past; her
  IEEE papers describe the method in present tense ("Each indicator is mapped ...", "Linear
  interpolation is used ..."). Use her convention, which is standard for IEEE computing papers:
  present tense to describe the method/design, simple past for what was run and observed.
- **Interpretation in Results.** The guide says none; her papers add one short "what this shows"
  sentence after a number. Keep that light touch, but every substantive interpretive claim (e.g.
  "anchoring, not knowledge") belongs in Discussion.
- **Novelty claims.** a104 uses "For the first time ...". Do NOT do that in this paper: the
  self-distillation-as-regularization idea has prior art ([Furlanello18], [Zhang19], [Mobahi20]);
  state precisely how this study differs instead.
