# Routing, Fusion, and Answer Selection


<span id="back-to-research-themes" class="legacy-fragment-alias"
aria-hidden="true"></span>
<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Routing,
Fusion, and Answer Selection</span>
</nav>

<nav class="theme-local-nav" aria-label="On this theme page">

<a href="#current-working-answer">Working answer</a>
<a href="#what-has-been-tried">Tried</a>
<a href="#known-failures-and-cases">Failures</a>
<a href="#cheapest-valid-next-experiments">Next tests</a>
</nav>

<span id="why-this-theme-matters" class="legacy-fragment-alias"
aria-hidden="true"></span>

## New to this theme?

More evidence is useful only if the system requests the right evidence
and then uses it correctly. A route can fail because it did not retrieve
the needed moment, transcript span, sound, or detail. It can also fail
after that evidence was present, because the model selected the wrong
fact or phrased the wrong answer. Those are different research problems.

For example, when a transcript contains the needed word but the answer
stays wrong, retrieving more text may not help. The next diagnostic
should hold the text fixed and test grounding or answer selection. A
**router** chooses an evidence path; **fusion** combines evidence;
**answer election** selects among candidate answers.

<span id="technical-summary" class="legacy-fragment-alias"
aria-hidden="true"></span>

## Current working answer

Use the visual baseline unless a declared, quality-gated specialist
condition applies. Existing taxonomy tags are **Diagnostic signal** for
experiment design, not a validated deployment policy. A future router
must abstain to the baseline when uncertain and pass route-level
non-regression before it becomes a default decision rule. The first
local multiple-choice fusion diagnostic found that blind equal-family
Borda underperformed the strongest direct path; keep Gemini native AV as
its provisional control and test Qwen transcript only as a predeclared,
abstaining speech route on fresh data. A follow-up one-window adaptive
inspection stayed at 15/20 while repairing one first-stage error and
regressing one, so do not treat a single higher-FPS or higher-resolution
crop as a sufficient agent loop. A later single-model deliberative loop
fell to 14/20: its duplicate deterministic solvers agreed, the planner
opened no inspection, and the judge repeated that consensus. Keep direct
Gemini as the control; the next agent must create adversarial diversity
and make stopping a claim that another pass can challenge.

<div class="theme-decision-strip" aria-label="Routing decisions">

- **Use now:** visual baseline fallback and selective accepted Whisper
  for declared lexical speech.
- **Do not default:** taxonomy tags as if they were a validated router,
  an ensemble before evidence paths are measured, a one-window adaptive
  crop, or duplicate same-seed solvers as if agreement were independent
  evidence.
- **Still possible:** abstaining multi-label routing, grounded fusion,
  and an adversarial evidence agent that preserves global context,
  challenges the stop decision, and can backtrack.
- **Next test:** use the existing zero-compute six-change screen as a
  provisional mechanism hypothesis, not a completed audit; then measure
  selector/evidence recall, oracle-evidence answering, and end-to-end
  answering separately. For MC, test a full-video evidence ledger plus
  up to two distinct local inspections and final arbitration against the
  unchanged Gemini control, but first force genuine hypothesis diversity
  and an adversarial stop review on a fresh label-isolated panel.

</div>

## Evidence at a glance

- **Route seeds:** lexical question counts and operational tags help
  form strata; they do not measure routing prevalence.
- **Observed mechanism:** some transcripts contain needed words while
  the frozen answer remains wrong.
- **Audit gaps:** there is no versioned evaluated router or local
  required- evidence label set.
- **First local MC fusion result:** on a selected balanced 20-row panel,
  Gemini native AV scored 15/20, Qwen transcript 13/20, Qwen native
  11/20, within-Qwen Borda 12/20, and equal-family Borda 13/20. This is
  a mechanism diagnostic, not a full-panel promotion benchmark.
- **First adaptive MC result:** on the same frozen panel, a
  complete-video planning pass followed by one model-selected native-AV
  inspection also scored 15/20. It repaired one stage-1 error and
  regressed one; none of its requested windows covered more than 75% of
  a clip.
- **First deliberative MC result:** a bounded single-model loop on the
  same panel scored 14/20, with 0 repairs and 1 regression versus direct
  Gemini. All 18 valid planners stopped immediately and requested 0
  evidence views; this diagnoses deterministic self-confirmation, not
  the value of higher FPS or resolution after inspection.
- **Condition boundary:** an unvalidated router does not imply routing,
  fusion, or answer election cannot help under a controlled test.

## What has been tried

<a name="taxonomy-audit"></a>

### Taxonomy audit and route seeds

- **Reason for testing:** identify recurring question/evidence types and
  prior routing ideas before assigning specialist experiments.
- **Tested condition and panel:** a conservative lexical audit of 481
  training questions plus primary documentation for prior TRECVID
  systems.
- **Observed result:** the audit found lower-bound prompt counts of 71
  counting, 22 spoken-language, 20 sound/music, and 15 visible-text
  questions. It proposes operational tags such as speech content,
  non-speech sound/music, prosody, temporal order, counting, small
  spatial detail, global visual, and multi-hop.
- **Current condition-bound status:** **Diagnostic signal** and **Open
  hypothesis** for a future router, not measured routing prevalence.
- **What this rules out now:** treating lexical lower bounds as
  exhaustive, mutually exclusive labels or claiming that the repository
  has a validated question router.
- **What it does not rule out:** a versioned multi-label taxonomy or a
  label-blind classifier that is validated on held-out questions.
- **Possible defect, audit gap, or uncertainty:**
  **Uncertainty**—lexical counts are lower bounds and tags may overlap
  or miss paraphrases.
- **Reconsider when:** tags, routing thresholds, abstention, and
  baseline fallback are fixed before a held-out route evaluation.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The official taxonomy is useful for reporting, while operational tags
describe which evidence path may help. Prompt wording is only a
conservative lower bound: paraphrases can be missed and tags can
overlap. Prior 2025 systems used related ingredients, but no paper
documents a complete conditional router that chooses model, audio
treatment, and frame density per question. See the [taxonomy audit ·
Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/taxonomy_prior_art_and_open_omni_models_2026-07-24.md)
and [closed issue \#4 · Issue/PR history ·
GitHub](https://github.com/npatta01/trec_video_2026/issues/4).
</details>

<span id="transcript-residuals" class="legacy-fragment-alias"
aria-hidden="true"></span>

### Transcript residuals: evidence present, answer still wrong

- **Reason for testing:** determine whether transcript improvements
  solve the downstream answer problem or only reveal the spoken
  evidence.
- **Tested condition and panel:** targeted speech and multilingual
  diagnostics, including two non-English examples and a selected
  automatic-transcript panel.
- **Observed result:** language-aware ASR produced coherent
  native-language text for two diagnostic examples, but the downstream
  Qwen answers were still wrong or incomplete; two residuals had needed
  words but the VLM selected or used the wrong evidence, and one ASR
  output missed a non-lexical vocalization. In the selected 10-row
  Whisper ablation, accepted transcripts improved manual accuracy from
  4/10 to 7/10, leaving residual errors.
- **Current condition-bound status:** **Diagnostic signal** that speech
  recovery and answer grounding are separable; current Whisper remains a
  **Routed candidate**, not a universal route.
- **What this rules out now:** assuming that a fluent transcript, or a
  better transcription family, automatically yields the correct final
  answer.
- **What it does not rule out:** timestamped evidence-span prompts, a
  better answer-grounding stage, or a matched multilingual route.
- **Possible defect, audit gap, or uncertainty:**
  **Uncertainty**—selected residuals identify a mechanism but not its
  frequency.
- **Reconsider when:** a frozen panel compares the same accepted
  evidence with and without explicit evidence-span grounding and records
  null/rejection controls.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The two-language diagnostic is too small to estimate multilingual
performance; the 10-row Whisper ablation is selected lexical-speech
evidence. The [transcript diagnostic · Evidence of record · Evidence
report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md)
supports the two evidence-use residuals and the missed non-lexical
vocalization. The [failure-panel report · Evidence of record · Evidence
report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md)
explains the separation between speech recovery and answer selection.
See also [Speech and Transcript
Evidence](research-speech-transcripts.md).
</details>

<a name="retrieval-versus-fusion"></a>

### Retrieval versus fusion diagnostic

- **Reason for testing:** prevent a temporal or multimodal failure from
  being misdiagnosed as only a model-quality failure.
- **Tested condition and panel:** a tracked literature synthesis and
  experiment design; no local required-evidence labels or oracle-frame
  evaluation exist yet.
- **Observed result:** the project has a concrete measurement
  distinction but no empirical local retrieval-versus-fusion
  decomposition. The existing A4 Qwen3-Omni screen is a motivating local
  case: five target changes are consistent with acoustic sensitivity in
  its declared dialogue/speech and sound/music target strata, while one
  control `C→W` regression is consistent with cross-modal interference
  or downstream answer selection. The frozen result cannot distinguish
  those mechanisms.
- **Current condition-bound status:** **Open hypothesis** and **Audit
  gap** in the current evidence record.
- **What this rules out now:** claiming that a wrong answer proves
  either poor evidence selection or poor reasoning without measuring
  both.
- **What it does not rule out:** a small grounded development set that
  measures selection, oracle evidence, and end-to-end answers
  separately.
- **Possible defect, audit gap, or uncertainty:** **Audit gap**—no local
  required-evidence labels or oracle-evidence evaluation exist yet;
  audio may also suppress visual grounding rather than merely alter
  retrieval.
- **Reconsider when:** required moments or evidence spans are labeled
  and the three metrics are reported under one frozen answer protocol.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

**Retrieval failure** means the needed frame, transcript span, sound, or
crop was not selected. **Fusion/reasoning failure** means the needed
evidence was available but the model still combined it incorrectly or
chose the wrong answer. The literature map recommends selector recall,
oracle-evidence answer quality, and end-to-end answer quality precisely
to tell these apart. See the [research-thread source · Evidence of
record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/queries/literature_review.sql)
and [closed issue \#4 · Issue/PR history ·
GitHub](https://github.com/npatta01/trec_video_2026/issues/4).
</details>

<a name="answer-election"></a>

### Evidence-aware answer election

- **Reason for testing:** explore whether multiple plausible generations
  or evidence views need a separate selection step.
- **Tested condition and panel:** this record began as a
  literature-backed proposal with no local answer-election result. The
  later 20-row MC diagnostic below supplies a first matched blind-Borda
  test but does not test grounded evidence-aware election.
- **Observed result:** prior systems motivate probability pooling,
  reranking, and evidence-aware election. The first local blind fusion
  result was negative, while a grounded election remains untested.
- **Current condition-bound status:** blind fusion is **Not promoted
  under the tested condition**; grounded answer election remains an
  **Open hypothesis**.
- **What this rules out now:** presenting voting, reranking, or
  ensembles as a current project recommendation.
- **What it does not rule out:** a targeted election experiment once
  candidate evidence paths and answer formats are frozen.
- **Possible defect, audit gap, or uncertainty:** **Audit gap**—no
  matched local grounded election experiment establishes benefit,
  evidence support, or cost.
- **Reconsider when:** a controlled comparison uses the same generated
  candidates, evidence citations, costs, and a baseline single-answer
  path.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

An election mechanism cannot repair missing evidence, and exact-string
voting can split equivalent open answers. The [literature map · Evidence
of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/queries/literature_review.sql)
recommends semantic deduplication and evidence-aware ranking only as a
testable future direction.
</details>

<a name="mc-poc20"></a>

### Twenty-video direct-path and Borda diagnostic

- **Reason for testing:** determine whether Gemini native AV, Qwen
  native video, Qwen video plus transcript, or a family-balanced fusion
  deserved a larger multiple-choice run.
- **Tested condition and panel:** 20 VQA-2025 training videos, balanced
  at four each across temporal, causal, OCR/detail, counting, and speech
  strata. Answer positions were balanced; labels were isolated until all
  predictions froze.
- **Observed result:** every canonical ranking was valid. Gemini scored
  15/20 (75%), Qwen transcript 13/20 (65%), Qwen native 11/20 (55%),
  within-Qwen Borda 12/20 (60%), and equal-family Borda 13/20 (65%).
  Qwen views agreed on 18/20 top choices but often shared errors. Four
  of Gemini’s five missed correct choices were ranked second.
- **Current condition-bound status:** **Verified** completion and
  scoring; **Not promoted under the tested condition** for blind fusion,
  Qwen native, or Qwen dual-view. Gemini is the provisional larger-panel
  control; Qwen transcript is only a fresh speech-route hypothesis.
- **What this rules out now:** treating model-family independence, view
  agreement, or blind Borda as sufficient evidence for a third
  submission.
- **What it does not rule out:** Qwen on a representative panel, a
  predeclared speech route with abstention, exact-spelling OCR crops,
  dense event counting, or a grounded top-two verifier.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  small balanced panel is selected and nonrepresentative. The post-hoc
  speech route scored 16/20 but is not promotion evidence.
- **Reconsider when:** a fresh label-blind panel predeclares routing and
  non-regression rules, or the intended full panel completes under
  frozen conditions.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Gemini received complete native video/audio. The pinned Qwen3.6-27B
revision received matched native-video and video-plus-Whisper views.
Strict four-choice rankings and family-balanced Borda were frozen before
scoring. Two initial operational failures were excluded from quality
comparison: Gemini’s high-thinking output budget was too small, and
Qwen’s private media prefix was empty before model startup. Bounded
repaired runs completed without exceeding the authorized cost cap. See
the [dated evidence record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-08-21-vqa-mc-poc20.md)
and its [aggregate evidence bundle · Evidence of record ·
GitHub](https://github.com/npatta01/trec_video_2026/tree/main/docs/experiments/artifacts/vqa-mc-poc20-20260821/).
</details>

<a name="mc-adaptive-poc20"></a>

### Two-stage adaptive-inspection diagnostic

- **Reason for testing:** determine whether native video with a
  question-selected second pass at higher temporal or spatial detail
  could repair the direct Gemini control’s multiple-choice errors.
- **Tested condition and panel:** the same frozen 20-row balanced MC
  panel and frozen Gemini direct predictions. Stage 1 received complete
  native AV at 4 FPS/low resolution, ranked all choices, and requested
  one arbitrary interval at 1/4/8 FPS and low/high resolution. Stage 2
  received that interval and adjudicated the stage-1 top two. Labels
  stayed isolated until final predictions froze.
- **Observed result:** frozen control, adaptive stage 1, and adaptive
  final all scored 15/20. The final MRR was 0.8583 versus 0.8667 for the
  control. Stage 2 repaired one stage-1 error and regressed one;
  relative to the control it repaired two and regressed two. Nineteen
  stage-2 responses were valid, and the predeclared stage-1 fallback
  preserved 20/20 valid finals.
- **Current condition-bound status:** **Verified** completion, label
  isolation, freeze linkage, cost, and scoring; **Not promoted under the
  tested condition** because the final exceeded neither control nor
  stage 1 and repairs did not exceed regressions.
- **What this rules out now:** one narrowed inspection plus one top-two
  reconsideration as a sufficient MC agent topology on this selected
  panel.
- **What it does not rule out:** an agent that preserves a full-video
  evidence ledger, compares competing hypotheses, requests more than one
  distinct inspection, backtracks, and lets the final arbiter retain the
  control.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  model requested no interval covering more than 75% of its clip, stayed
  at low resolution on 13/20 rows, and used 8 FPS on only 4/20; the
  selected panel is too small to estimate full-panel gains. The result
  diagnoses the tested flow, not all adaptive video agents.
- **Reconsider when:** a predeclared bounded multi-inspection flow
  improves a fresh label-blind panel with more repairs than regressions
  and no control degradation.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The run made exactly 40 Gemini 3.7 Flash calls. It used 116,098 prompt
tokens and 28,645 output tokens and cost `$0.09724613`, below the
`$0.75` cap. Every inspection plan was schema-valid; one second-stage
answer was invalid JSON and used the frozen stage-1 fallback. The
mechanism counts and hashes of all ignored private freezes are in the
[adaptive aggregate · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/artifacts/vqa-mc-poc20-20260821/adaptive-agent-summary.json),
with interpretation in the [dated evidence record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-08-21-vqa-mc-poc20.md#two-stage-adaptive-inspection-follow-up).
</details>

<a name="mc-deliberative-poc20"></a>

### Bounded single-model deliberative-agent diagnostic

- **Reason for testing:** determine whether two complete-video solvers,
  an evidence notebook, up to two blinded inspections, and a fresh judge
  could repair the direct Gemini control without the one-window agent’s
  regressions.
- **Tested condition and panel:** the same frozen 20-row mechanism panel
  and 15/20 direct Gemini control. One Gemini model produced two
  option-by-option hypotheses, planned or stopped, could request two
  model-selected native-AV views, and judged the frozen notebook. Labels
  remained isolated until all finals and notebook hashes froze.
- **Observed result:** the final scored **14/20** with 20/20 valid
  rankings and MRR 0.8292, versus 15/20 and 0.8667 for the control. It
  repaired 0 control misses and regressed 1 control hit. The two valid
  solvers agreed on top-1 in 19/19 rows; every valid planner stopped, no
  inspection ran, and the judge repeated the blind solver’s top choice
  in 20/20 rows.
- **Current condition-bound status:** **Verified negative result; not
  promoted.** The tested loop collapsed into deterministic
  self-confirmation rather than question-adaptive evidence collection.
- **What this rules out now:** treating same-model, same-prompt,
  same-seed agreement plus an unconstrained stop decision as useful
  deliberative diversity on this panel.
- **What it does not rule out:** different solver roles or seeds, an
  explicit counterexample seeker, an adversarial stop reviewer, and
  genuinely invoked local evidence on a fresh label-isolated panel.
- **Possible defect, audit gap, or uncertainty:** **Confirmed design
  defect**— the controller interpreted duplicated confidence as evidence
  even when the solver records named unresolved checks. The
  now-heavily-inspected panel can support regression diagnosis only, not
  promotion.
- **Reconsider when:** a fresh panel uses genuinely diverse hypotheses
  and a stop reviewer that can require inspection whenever uncertainty
  or an untested counterargument remains.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

All 20 final notebooks, citations, provenance records, and declared
hashes verified. Across the invalid output-budget attempt, the
interrupted parser correction, and the completed reused-response run,
observed cost was `$0.26213850`; the final conservative combined ceiling
was `$0.66699825`, below the authorized `$0.75`. Post-label native-media
review found no miss for which all four options were wrong: four
references were clearly supported and two rows had a defensible
alternative option. Official scoring remains unchanged. See the
[deliberative aggregate · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/artifacts/vqa-mc-poc20-20260821/deliberative-agent-summary.json)
and the [dated evidence record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-08-21-vqa-mc-poc20.md#deliberative-agent-poc20).
</details>

## Known failures and cases

- [CASE-001 — Later-span answer selection](failure-library.md#case-001)
  separates retrieved evidence from the fact ultimately selected.
- [CASE-003 — Retrieved token unused by
  answerer](failure-library.md#case-003) motivates grounded fusion or
  extraction tests instead of more retrieval.

### What the evidence supports

- The current taxonomy is useful to form strata and candidate routes,
  but it is not a measured router.
- Retrieval and fusion are different failure modes and need different
  metrics.
- The A4 aggregate provides a condition-bound acoustic-sensitivity
  signal with a control-safety loss; it does not identify whether the
  mechanism is missing evidence, fusion, or answer selection.
- Transcript residuals demonstrate that recovering words does not
  guarantee the VLM will ground the final answer in those words.
- Specialist policies need abstention, a baseline fallback, and
  route-level non-regression—not just a target-slice improvement.
- Blind Borda can erase a stronger direct answer: on the selected MC
  panel it lost two correct rows versus Gemini native AV.
- A single question-selected inspection can also move rather than reduce
  errors: it repaired one stage-1 answer and regressed one on the same
  panel.
- Same-model agreement is not automatically independent evidence: the
  tested duplicate solvers agreed on every comparable top choice, opened
  no local evidence, and produced a final one point below the direct
  control.

### What it does not rule out

- The absence of an evaluated router does not imply a simple route
  cannot help.
- A wrong answer does not reveal whether the missing piece was frame
  selection, audio/transcript selection, reasoning, or answer
  normalization.
- A modality-on regression does not by itself prove answer-election
  failure; first check whether the required visual evidence remained
  available and sufficient.
- A literature-supported technique is not a local result until it runs
  under the project’s integrity and scoring boundaries.
- A future ensemble may help after evidence paths stabilize; it should
  not be used to mask an unmeasured retrieval problem.
- The negative blind-fusion result does not rule out a grounded top-two
  verifier or a predeclared specialist route with abstention.
- The negative one-window result does not rule out a bounded agent that
  retains global evidence, compares multiple local views, backtracks,
  and explicitly preserves the control when evidence is inconclusive.
- The negative deliberative result does not measure the benefit of
  higher FPS or resolution after inspection because no inspection ran.
  It rules out the tested deterministic consensus-and-stop protocol, not
  an adversarial agent that creates distinct hypotheses and challenges
  stopping.

<span id="possible-bugs-confounds-and-audit-gaps"
class="legacy-fragment-alias" aria-hidden="true"></span>

### Confirmed defects, audit gaps, and uncertainties

- **Audit gap:** route seeds exist, but no versioned router records its
  tag model, thresholds, abstentions, fallback, held-out evaluation, and
  route-level metrics.
- **Uncertainty:** lexical tags are lower bounds and may miss
  paraphrases or assign overlapping evidence needs.
- **Audit gap:** no local required-moment or evidence-span labels exist
  to distinguish selector recall from fusion/reasoning accuracy.
- **Uncertainty:** selected transcript residuals show a mechanism but do
  not estimate its frequency across the competition distribution.
- **Audit gap:** the A4 six-change screen is provisional and does not
  yet have frozen required-evidence labels, selector traces, or
  oracle-answer results.
- **Uncertainty:** the 20-row MC panel identifies candidate error
  mechanisms but is too small and selected to estimate full-panel or
  hidden-test accuracy.
- **Uncertainty:** the adaptive result diagnoses one forced-inspection
  topology; it does not estimate the value of additional inspection
  rounds or a global evidence ledger.

## Open hypotheses and ideas

- A label-blind multi-label router can reduce harmful global modality
  use by abstaining to the visual baseline.
- A union of broad visual coverage and targeted evidence can improve
  retrieval without forcing a model to reason over only question-matched
  fragments.
- Grounding an answer in cited timestamps, frames, transcript spans, or
  sound events can expose whether an answer-selection stage is using
  relevant evidence.
- Grounded top-two election or a targeted ensemble may help only after
  the evidence paths themselves are controlled and frozen; blind Borda
  is not the current candidate.

## Cheapest valid next experiments

### Versioned label-blind router

- **Hypothesis:** a multi-label router with abstention selects
  specialist evidence more safely than a global modality switch.
- **Smallest valid panel or work item:** a held-out, label-blind
  development slice with declared tags for speech, sound, prosody,
  temporal, spatial, and visual controls.
- **Matched controls and integrity safeguards:** version tag rules/model
  and thresholds; hide references from routing and inference; record
  every route, abstention, fallback, and downstream configuration;
  freeze answers before scoring.
- **Decision or promotion rule:** advance only if routed end-to-end
  quality meets a predeclared target gain with no route-level or
  aggregate regression beyond the declared limits versus baseline
  fallback.
- **Boundary on what a positive result would establish:** the router’s
  value on its held-out slice, not universal classification accuracy.

### Required-evidence labels and three-way measurement

- **Hypothesis:** separating selector recall, oracle-evidence answer
  quality, and end-to-end answer quality identifies whether a route
  needs retrieval or fusion work.
- **Smallest valid panel or work item:** first freeze a private
  six-change mechanism sheet with required frames, transcript spans,
  sound events, or visible-text evidence; then use a small manually
  grounded development slice if the static screen remains ambiguous.
- **Matched controls and integrity safeguards:** label evidence
  independently of model answers; preserve timestamps/IDs; evaluate
  uniform and proposed selectors at the same budget; use the same
  model/prompt/decoding and frozen scoring boundary.
- **Decision or promotion rule:** pursue retrieval only when selector
  recall is limiting and oracle evidence improves answers; pursue
  fusion/grounding or answer election when the necessary evidence is
  present but answers remain wrong; pursue modality-competition
  safeguards when audio-on reduces visual evidence sufficiency or
  control quality.
- **Boundary on what a positive result would establish:** a diagnosis
  for the labeled slice, not a universal causal explanation.

### Modality-competition / visual-grounding guard

- **Hypothesis:** native audio can improve an audio-relevant answer
  while suppressing or outweighing other sufficient evidence on control
  rows.
- **Smallest valid panel or work item:** the A4 control regression plus
  predeclared unchanged controls; use the frozen outputs first and add
  an oracle-evidence comparison only if the required-evidence labels
  show that the other evidence was available.
- **Matched controls and integrity safeguards:** hold model, frames,
  audio, prompt, decoding, and scoring fixed; record evidence
  sufficiency before revealing grades; compare audio-off fallback,
  audio-on, and any grounded answer-election condition after prediction
  freeze.
- **Decision or promotion rule:** route audio only when audio-relevant
  gains survive the visual-control gate; otherwise abstain to audio-off
  or add a grounded citation/selection guard. Do not interpret a
  corrected control as general native-audio benefit without a held-out
  panel.
- **Boundary on what a positive result would establish:** a cross-modal
  safeguard for the tested model and slice, not a universal attention or
  fusion explanation.

<a name="bounded-multi-inspection-evidence-agent"></a>

### Bounded multi-inspection evidence agent

- **Prior result:** the first bounded implementation scored 14/20 versus
  the 15/20 direct control because duplicate deterministic solvers
  agreed, all valid planners stopped, and no inspection ran. The
  remaining hypothesis is narrower: adversarially diverse roles plus a
  challengeable stop decision may activate useful evidence collection.
- **Hypothesis:** preserving a global evidence ledger, assigning one
  solver to seek counterevidence, challenging any proposed stop while
  unresolved checks remain, and testing competing hypotheses with up to
  two distinct inspections repairs more errors without losing the direct
  control’s correct answers.
- **Smallest valid panel or work item:** a fresh label-blind MC panel
  with an unchanged full-video Gemini control and a bounded agent:
  role-diverse global hypotheses, adversarial stop review, up to two
  non-duplicate inspections, and final arbitration. The existing 20 rows
  are only a regression suite.
- **Matched controls and integrity safeguards:** freeze the control
  first; isolate labels; record every evidence claim, timestamp,
  inspection setting, stop/backtrack choice, and fallback; let the
  arbiter see global and local evidence and retain the control; use
  different declared solver roles or seeds; freeze all predictions
  before scoring.
- **Decision or promotion rule:** advance only if the agent exceeds the
  control and its own first stage, repairs exceed regressions, all
  finals are valid, and observed cost stays under the predeclared cap.
- **Boundary on what a positive result would establish:** the bounded
  flow on the tested panel, not that every MC question needs an agent or
  that more passes are universally better.

## Method, result, validity risks, and evidence

<details>

<summary>

How to audit this theme’s records
</summary>

The visible summaries distinguish route selection, evidence retrieval,
fusion, and answer election. Open each record’s native details block for
source links and validity risks. A later election experiment must hold
generated candidates and evidence fixed; it cannot compensate for
unmeasured missing evidence.
</details>

## Related themes

- [Visual and Temporal Evidence](research-visual-evidence.md) covers the
  retrieval side of frame and temporal selection.
- [Speech and Transcript Evidence](research-speech-transcripts.md)
  covers provenance and accepted transcript evidence.
- [Native Audio and Sound](research-native-audio.md) covers
  category-routed sound hypotheses.
- [Models and Adaptation](research-models.md) explains why a
  specialist-model decision needs matched evidence and gates.
