# Experiment Ledger


<div class="lead">

The experiments narrowed the project toward a simple visual baseline
with specialist evidence routes. The strongest general comparison is
frozen-30; the smaller panels explain failures, compatibility, or
routing and must not be read as 2026 accuracy estimates.

</div>

Use [Start Here](index.md) for the current recommendation and the [Code
Map](code-map.md) to find the implementation behind each workflow.

## Research questions

Read this page by research question to understand why the project ran
each experiment and what decision followed. Use the chronology index at
the end only when you need the date-ordered audit trail.

``` mermaid
flowchart TD
  accTitle: Research-question reading path
  accDescr: A top-to-bottom path through four research questions. Reproducible execution does not establish quality. Qwen3.5-4B with 12 frames is the accuracy baseline and MiniCPM-o audio-off is the fast alternate. Whisper is routed while native audio remains experimental. No model is promoted and Qwen3-Omni remains ineligible.
  P[Research questions] --> T[1. Trust experiments?<br/>Reproducible execution;<br/>quality remains separate]
  T --> B[2. Global default?<br/>Qwen3.5-4B + 12 frames;<br/>MiniCPM audio-off for speed]
  B --> S[3. Add specialist evidence?<br/>Route Whisper; native audio<br/>remains experimental]
  S --> M[4. Promote another model?<br/>No; Qwen3-Omni is ineligible]
```

**Figure caption:** Research-question reading path: four thematic
questions and the current decision from each, ordered for a newcomer
rather than as a chronology.

The path is a decision index, not a chronology or evidence ranking. The
four compact records below are its text equivalent and state the
strongest evidence each path can support.

**1. Can we run trustworthy experiments?**

- **Why it was tested:** Model comparisons require private,
  reproducible, label-isolated execution.
- **Current answer:** The repository can run controlled work; readiness
  and compatibility do not establish answer quality.
- **Evidence strength:** Operational checks and a three-row
  compatibility panel.

**2. What should every question start with?**

- **Why it was tested:** The project needed a simple default before
  adding specialists.
- **Current answer:** Qwen3.5-4B with 12 frames is the accuracy
  baseline; MiniCPM-o audio-off is the fast alternate.
- **Evidence strength:** Matched frozen-30 plus narrower temporal
  diagnostics.

**3. When is specialist evidence worth adding?**

- **Why it was tested:** Speech and sound may help selected questions
  but can distract globally.
- **Current answer:** Route quality-gated Whisper to lexical speech;
  keep native audio experimental and the human-oracle tier blocked.
- **Evidence strength:** Selected diagnostics, including a ten-row
  transcript ablation.

**4. Should another multimodal model replace the baseline?**

- **Why it was tested:** Larger and native-audio models might recover
  baseline failures.
- **Current answer:** None is promoted; Qwen3-Omni is promising but
  formally ineligible.
- **Evidence strength:** Failure-selected and three-/ten-row quality
  screens.

<a name="trustworthy-experiments"></a>

## Can we run trustworthy experiments?

**Why this mattered.** Before comparing models, the project needed
private, reproducible acquisition, label isolation, runnable adapters,
and a clear line between compatibility and quality.

**Current answer.** The repository can prepare and execute controlled
local work, but operational readiness and compatibility do not establish
answer quality.

<a name="field-guide"></a>

### July 12 field guide

- **Why tested:** The repository began with candidate architectures and
  prior work but lacked local executed evidence to choose among them.
- **Test and panel:** Synthesized the prior TRECVID landscape,
  open-model options, scoring constraints, and a staged evidence-first
  architecture from literature and prior-team evidence, not a local
  benchmark.
- **Result → decision:** The project needed observable, matched evidence
  before complex routing or training. Keep the [field
  guide](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/report.html)
  as background orientation, not current experiment status.
- **Boundary:** It predates every local result in this ledger and cannot
  pick today’s baseline.
- **Evidence:** [field guide source
  notes](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/source_notes.md)
  and the later [repository recovery PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

### Data readiness

- **Why tested:** Controlled inference required resumable acquisition,
  source validation, shared-data linking, and label isolation without
  committing private artifacts.
- **Test and panel:** Built resumable acquisition, source validation,
  shared-data linking, and label-isolated preparation seams as
  operational readiness, not answer-quality evaluation.
- **Result → decision:** The repository can prepare and audit local
  evidence without committing the competition dataset, media,
  transcripts, predictions, or logs. Keep one ignored shared dataset and
  run preflight before inference; do not start a large acquisition or
  benchmark merely to inspect the repository.
- **Boundary:** Readiness proves neither video availability for every
  future row nor 2026 answer quality.
- **Evidence:** [root operating
  boundary](https://github.com/npatta01/trec_video_2026/blob/main/README.md),
  [download
  module](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_download.py),
  and [recovery PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

<a name="compatibility-gate"></a>

### Three-video compatibility gate

- **Why tested:** Before paying for larger panels, the project needed to
  learn which model-family adapters could execute reproducibly on the
  local AMD/ROCm host.
- **Test and panel:** Ran Qwen3.5, Molmo2, InternVL, MiniCPM-o, and
  Gemma through a common gate on three deliberately different labeled
  2025 examples.
- **Result → decision:** The family adapters could be made reproducible,
  but the panel was too small for model ranking; MiniCPM’s initial
  selected-frame native-audio path also exposed a speech-row
  tensor-shape failure. Retain Qwen3.5-4B with 12 uniform frames as the
  practical baseline and expand only through controlled workflows.
- **Boundary:** This is **compatibility-only**, not a statistical
  accuracy sample.
- **Evidence:** [three-video gate
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/three_video_gate_results_2026-07-20.md)
  and [recovered benchmark work in PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

<a name="global-default"></a>

## What should every question start with?

**Why this mattered.** The project needed a simple global baseline
before adding routing, more frames, or specialist models.

**Current answer.** Qwen3.5-4B with 12 uniform frames remains the global
accuracy baseline; MiniCPM-o 4.5 audio-off is the fast alternate. Global
24 frames and the tested temporal heuristics did not earn default
status.

### Frozen-30 protocol v2

- **Why tested:** The three-video gate established runnable adapters but
  was too small to choose a default; the project needed a matched
  comparison of model, frame-budget, and transcript conditions.
- **Test and panel:** Four matched VQA arms across 30 labeled 2025
  training examples balanced across five declared strata; 120 valid
  predictions and 30 valid Whisper executions.
- **Result → decision:** Qwen12+Whisper led exact match at 11/30 and
  token F1 at 0.506795; MiniCPM audio-off led METEOR at 0.374089 and
  outer-wall speed; Qwen24 added no exact answers over Qwen12. Keep
  Qwen12 as the accuracy baseline, MiniCPM audio-off as the fast
  alternate, and Whisper as a separately reported route.
- **Boundary:** This is the strongest verified repository comparison,
  but it is descriptive model selection on 2025 training data—not a 2026
  estimate.
- **Evidence:** [protocol-v2
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md)
  and [recovered benchmark work in PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

### Directed temporal sampling

- **Why tested:** Global frame increases were costly and inconclusive,
  so the project tested whether reallocating a fixed 12-frame budget
  toward question-relevant time spans could help.
- **Test and panel:** Replaced part of a 12-frame uniform sequence with
  question-directed temporal evidence while holding model and pixel
  bounds fixed, across ten temporal rows.
- **Result → decision:** Directed 12 tied uniform 12 and trailed uniform
  24 in blinded C/P/W adjudication. Stop the heuristic; it did not pass
  either promotion condition.
- **Boundary:** A small temporal diagnostic can reject this
  implementation, not every possible learned or question-directed
  selector.
- **Evidence:** [temporal
  report](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/temporal-sampling-issue-8.md),
  [closed issue
  \#8](https://github.com/npatta01/trec_video_2026/issues/8), its
  [conclusion audit in closed issue
  \#13](https://github.com/npatta01/trec_video_2026/issues/13), and [PR
  \#10](https://github.com/npatta01/trec_video_2026/pull/10).

### Dense-burst sampling

- **Why tested:** A narrower temporal hypothesis asked whether a dense
  local burst could beat a broader uniform-24 view on rows requiring
  short-lived evidence.
- **Test and panel:** Compared uniform-24 against dense-burst-12 under
  that narrow temporal hypothesis on two rows only.
- **Result → decision:** Uniform-24 was 2C/0P/0W while dense-burst-12
  was 0C/1P/1W, comprising one C→P and one C→W regression. Stop
  dense-burst-12; do not generalize from or expand the losing heuristic
  without a new design.
- **Boundary:** The panel is too small for prevalence or effect-size
  claims. No row identifiers, answers, predictions, paths, or raw
  artifacts are part of this report.
- **Evidence:** Aggregate source digest
  `d4825b4530b181f1b4414552a80136e2f0e5a7ee64def23131d8c163acc458f1` and
  [closed issue
  \#16](https://github.com/npatta01/trec_video_2026/issues/16).

<a name="specialist-evidence"></a>

## When is specialist evidence worth adding?

**Why this mattered.** Some questions depend on speech, sound, spatial
detail, or evidence provenance, but adding every modality globally can
distract the model or waste compute.

**Current answer.** Retain quality-gated Whisper for lexical speech;
study native audio only as a category-routed specialist; keep the
independent human-oracle tier blocked until its provenance records
exist. The [failure-selected native-audio result](#failure-panel)
supplies the strongest current caution against a global switch.

<a name="audio-spatial-seamless"></a>

### Audio, spatial, and Seamless diagnostics

- **Why tested:** Early baseline failures suggested missing speech or
  spatial detail, so the project tested the simplest transcript, crop,
  and translation escalations before designing a router.
- **Test and panel:** Added full Whisper large-v3 translation,
  question-guided spatial crops, and a SeamlessM4T-v2 translation on two
  audio/spatial failures, then one speech failure.
- **Result → decision:** The transcript could change Qwen’s answer and
  the crop could add pixels without making either answer correct;
  Seamless output was incomplete and the downstream answer remained
  wrong. Stop these particular escalations, keep 12 frames as the simple
  baseline, and require better routing evidence.
- **Boundary:** These are targeted **diagnostics**, not accuracy
  estimates; manual crops were not scale-ready.
- **Evidence:** [audio/spatial
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/audio_spatial_followup_results_2026-07-21.md),
  [Seamless
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/seamless_audio_followup_results_2026-07-21.md),
  and [recovery PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

### Automatic-transcript ablation

- **Why tested:** A transcript had changed an answer in an early
  diagnostic, but the project needed a matched, label-blind test to
  learn whether accepted ASR reliably helps lexical-speech rows.
- **Test and panel:** Compared identical 12-frame Qwen requests with and
  without a label-blind, quality-accepted Whisper large-v3 transcript
  across ten selected lexical-speech rows; all 20 predictions were
  frozen before scoring.
- **Result → decision:** Manual accuracy rose from 4/10 to 7/10, with
  three rescues and no observed degradation. Retain quality-gated
  Whisper on lexical-speech routes and work next on transcript grounding
  plus non-lexical-vocalization rejection.
- **Boundary:** A selected ten-row diagnostic does not validate routing
  prevalence or global transcript use.
- **Evidence:** [automatic-transcript
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md),
  [closed issue
  \#7](https://github.com/npatta01/trec_video_2026/issues/7), and [PR
  \#9](https://github.com/npatta01/trec_video_2026/pull/9).

### Taxonomy audit

- **Why tested:** Specialist routing needed evidence about which
  question types exist before the project could justify route categories
  or stratified follow-ups.
- **Test and panel:** Audited prior systems and prompt wording across
  481 training questions plus primary prior-system documentation.
- **Result → decision:** Conservative lexical lower bounds found 71
  counting, 22 spoken-language, 20 sound/music, and 15 visible-text
  prompts. Keep these as router-design seeds and stratification hints.
- **Boundary:** Lexical lower bounds overlap, omit paraphrases, and do
  not validate routing prevalence.
- **Evidence:** [taxonomy
  audit](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/taxonomy_prior_art_and_open_omni_models_2026-07-24.md),
  [closed issue
  \#4](https://github.com/npatta01/trec_video_2026/issues/4), and its
  restored [evidence-routing
  plan](https://github.com/npatta01/trec_video_2026/blob/main/docs/superpowers/plans/2026-07-24-evidence-routing-benchmark.md).

### Human-oracle tier

- **Why tested:** Automatic ASR gains do not establish the value of an
  independently produced transcript, so the project specified a
  provenance-clean upper-bound tier.
- **Test and panel:** Specified a separate tier for independently
  created and reviewed human transcripts; automatic ASR, uploader
  captions, and dataset prompts were explicitly kept out; none—the
  strict tier did not run.
- **Result → decision:** The required independent transcriber/reviewer
  records do not yet exist. Keep the human-oracle tier **blocked** in
  [open issue
  \#19](https://github.com/npatta01/trec_video_2026/issues/19).
- **Boundary:** There is no oracle result to compare or cite.
- **Evidence:** [transcript-evidence
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md),
  [open issue
  \#19](https://github.com/npatta01/trec_video_2026/issues/19), and the
  reconciled [evidence-routing
  plan](https://github.com/npatta01/trec_video_2026/blob/main/docs/superpowers/plans/2026-07-24-evidence-routing-benchmark.md).

<a name="model-promotion"></a>

## Should another multimodal model replace the baseline?

**Why this mattered.** Larger or native-audio-capable models could
potentially recover baseline failures, but runtime success is not
evidence of semantic quality.

**Current answer.** None is promoted. Qwen3.6, Phi, and Gemma failed
their relevant quality gates; Qwen3-Omni remains promising but formally
ineligible after only 19 of 20 strict outputs completed.

<a name="failure-panel"></a>

### Failure-selected Qwen3.6 and audio panel

- **Why tested:** Known baseline failures created a diagnostic
  opportunity to test whether a larger Qwen model or native audio could
  recover cases that the global baseline missed.
- **Test and panel:** Compared Qwen12, Qwen24, Qwen+Whisper,
  Qwen3.6-35B-A3B, and MiniCPM audio-off/audio-on paths on 12
  deliberately selected failures, explicitly not an accuracy sample.
- **Result → decision:** Qwen24 and MiniCPM audio-off each scored
  4C/4P/4W; Qwen3.6 scored 3C/2P/7W. Native audio helped some prosody
  and sound rows but harmed some temporal reasoning. Do not promote
  Qwen3.6 or a global audio switch; retain a small, category-routed
  ensemble idea.
- **Boundary:** Failure selection inflates the value of recovery
  behavior and cannot rank full-test systems.
- **Evidence:** [failure-panel
  report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md)
  and [recovery PR
  \#21](https://github.com/npatta01/trec_video_2026/pull/21).

<a name="phi-gemma-semantic-stop"></a>

### Phi and Gemma: compatibility, then semantic stop

- **Why tested:** Joint audio/vision runtime compatibility left open the
  separate question of whether Phi and Gemma produced semantically
  useful answers.
- **Test and panel:** Ran joint vision/audio contracts on
  Phi-4-multimodal-instruct and Gemma 3n, then corrected platform/output
  confounds and applied a predeclared answer-quality screen across three
  selected audio roles.
- **Result → decision:** Compatibility and semantic quality separated
  cleanly. On the semantic gate, Gemma scored 1C/1P/1W and Phi 0C/2P/1W.
  Neither model was promoted beyond three rows.
- **Boundary:** The counts are a semantic screen, not a general
  leaderboard; the earlier Gemma compatibility promotion is superseded.
- **Evidence:** [compatibility
  report](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-24-nonqwen-omni-compatibility.md),
  [semantic
  correction](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-24-omni-semantic-quality.md),
  [closed issues
  \#6](https://github.com/npatta01/trec_video_2026/issues/6) and
  [\#12](https://github.com/npatta01/trec_video_2026/issues/12), plus
  [PR \#11](https://github.com/npatta01/trec_video_2026/pull/11) and [PR
  \#15](https://github.com/npatta01/trec_video_2026/pull/15).

<a name="qwen3-omni-quality"></a>

### Qwen3-Omni compatibility and quality-10

- **Why tested:** Qwen3-Omni’s repaired joint runtime made it a
  candidate, but promotion required a neutral matched quality panel and
  a complete strict-output contract.
- **Test and panel:** Repaired the joint runtime/output contract, then
  compared 12-frame frames-only and native-audio conditions on a
  three-row compatibility screen and ten matched neutral-quality pairs.
- **Result → decision:** The final compatibility run passed the
  runtime/output contract and scored 1C/1P/1W; that made it a candidate,
  not a baseline. In neutral A2, a post-hoc answer-only diagnostic
  changed frames-only 3C/3P/4W to audio-on 5C/2P/3W. It remains
  promising but formally ineligible: the unchanged strict A2 run
  completed only 19/20 outputs.
- **Boundary:** The recovered answer is diagnostic only, one reviewer
  was not blinded, and 2026 answer quality remains unknown.
- **Evidence:** [compatibility
  report](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-24-qwen3-omni-dstack-compatibility.md),
  [neutral A2
  report](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-25-qwen3-omni-quality-10-neutral-a2.md),
  [closed issues
  \#5](https://github.com/npatta01/trec_video_2026/issues/5) and
  [\#17](https://github.com/npatta01/trec_video_2026/issues/17), [PR
  \#14](https://github.com/npatta01/trec_video_2026/pull/14), [PR
  \#18](https://github.com/npatta01/trec_video_2026/pull/18), and
  recovery [PR
  \#20](https://github.com/npatta01/trec_video_2026/pull/20).

## What superseded what

| Earlier statement or result | Superseding evidence | Current interpretation |
|----|----|----|
| July 12 architecture priorities | Executed panels through July 25 | Background field guide only; use this ledger for current status. |
| Failure panel favored global Qwen24 | Frozen-30 protocol v2 | Qwen12 is the global default; 24 frames added no exact answers. |
| Whisper as a broad additional modality | Frozen-30 plus automatic-transcript ablation | Route only to quality-gated lexical speech. |
| Initial Gemma compatibility promotion | Phi/Gemma semantic gate | Stop both at three rows; compatibility is not quality. |
| Qwen3-Omni compatibility success | Neutral quality-10 A2 | Promising but ineligible after 19/20 strict outputs. |
| Directed and dense temporal heuristics | Their matched diagnostic panels | Stop both implementations. |

The next work should close the gaps rather than repeat a superseded
default; see [Code Map and Remaining Gaps](code-map.md#remaining-gaps).

## Chronology index

| Date | Milestone | Research questions |
|----|----|----|
| July 12 | Field guide | [Trustworthy experiments](#trustworthy-experiments) |
| July 20 | Data readiness and compatibility gate | [Trustworthy experiments](#trustworthy-experiments) |
| July 21 | Audio/spatial diagnostics and frozen-30 | [Global default](#global-default); [specialist evidence](#specialist-evidence) |
| July 24 | Routing, temporal, and model-promotion diagnostics | [Global default](#global-default); [specialist evidence](#specialist-evidence); [model promotion](#model-promotion) |
| July 25 | Strict Qwen3-Omni quality result | [Model promotion](#model-promotion) |
