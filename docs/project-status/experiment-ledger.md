# Research Themes and Experiment Ledger


<div class="lead">

The project currently uses **Qwen3.5-4B with 12 uniform frames** as its
visual baseline and adds specialist evidence only through declared,
quality-gated routes. Frozen-30 protocol v2 is the strongest matched
comparison; the smaller panels are diagnostics. A result that missed one
promotion gate is not a permanent verdict on every model, prompt,
selector, budget, or panel.

</div>

Use [Start Here · Report page](index.md) for the current operational
recommendation and the [Code Map · Report page](code-map.md) to find the
implementation boundary. The six themes below are independent research
areas, not a sequence.

For record-level provenance, browse the [dated experiment evidence
index](../../docs/experiments/README.md). It routes to the canonical
records without duplicating this hub’s current conclusions or next-test
queue.

> [!NOTE]
>
> ### Contributors: place evidence, claims, and decisions consistently
>
> Before adding or revising a result, follow the [Adding Experiments and
> Results guide · Report page](adding-experiments.md). It defines the
> evidence bundle, canonical claim, decision disposition, and
> theme-routing rules used by this hub.

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="research-questions"></span><span id="fig-experiment-timeline"></span></div>

<a name="six-theme-research-map"></a>

## Current research-theme map

<div class="theme-grid">

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="trustworthy-experiments"></span><span id="field-guide"></span><span id="data-readiness"></span><span id="compatibility-gate"></span><span id="frozen-30-protocol-v2"></span></div>

### [Experiment integrity and evaluation](research-integrity.md)

**Working answer:** frozen-30 protocol v2 is interpretable; successful
execution alone does not establish answer quality.

**Strongest evidence:** [verified 30-row, four-arm protocol v2 plus
integrity audits](research-integrity.md#frozen-30-protocol-v2).

**Open question:** can one reusable gate reject label, identity,
mutation, and scoring-boundary faults before an expensive run?

</div>

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="global-default"></span><span id="directed-temporal-sampling"></span><span id="dense-burst-sampling"></span></div>

### [Visual and temporal evidence](research-visual-evidence.md)

**Working answer:** uniform-12 remains the default; global 24 frames and
the tested directed/burst heuristics were not promoted under their
tested conditions. A verified r4 audit represented the preferred
evidence in `10/10` rows, but the completed matched answer test found
oracle-hybrid-12 slightly worse than uniform-12 and did not promote it.
A frozen-timestamp audit found limited selection headroom: uniform-12
already covered all required intervals in `6/10` rows and at least one
in `9/10`.

**Strongest evidence:** [matched frozen-30 frame-budget comparison;
smaller 10-row and 2-row temporal
diagnostics](research-visual-evidence.md#frozen-30-protocol-v2).

**Open question:** does relevant-scene selection improve answers on a
prospectively frozen panel where uniform-12 actually misses required
evidence?

</div>

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="automatic-transcript-ablation"></span><span id="human-oracle-tier"></span></div>

### [Speech and transcript evidence](research-speech-transcripts.md)

**Working answer:** route quality-accepted Whisper to lexical speech;
keep transcript provenance explicit and the independent human tier
blocked.

**Strongest evidence:** [matched full-panel 475-row frame/audio audit
plus the selected 10-row automatic-transcript
ablation](research-speech-transcripts.md#matched-full-panel-frame-audio-audit).

**Open question:** can a semantic question/transcript relevance route,
tested against a shuffled-transcript negative control, retain the
Whisper gain without repeating the clear audio regression?

</div>

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="audio-spatial-seamless"></span></div>

### [Native audio and sound](research-native-audio.md)

**Working answer:** native audio is not a global switch; prosody and
non-speech sound remain routed candidates.

**Strongest evidence:** [matched audio-on/off and failure-selected
diagnostics, not a representative
benchmark](research-native-audio.md#minicpm-audio-on-off).

**Open question:** which declared audio strata improve without
regressing visual controls?

</div>

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="model-promotion"></span><span id="failure-panel"></span><span id="phi-gemma-semantic-stop"></span><span id="qwen3-omni-quality"></span></div>

### [Models and adaptation](research-models.md)

**Working answer:** no later candidate has displaced Qwen3.5-4B;
MiniCPM-o audio-off is the fast alternate.

**Strongest evidence:** [frozen-30 model comparison; later model screens
are selected diagnostics or compatibility
checks](research-models.md#minicpm-fast-alternate).

**Open question:** can a repaired, complete candidate pass a matched
quality and integrity gate?

</div>

<div class="theme-card">

<div class="legacy-fragment-alias-group" aria-hidden="true"><span id="specialist-evidence"></span><span id="taxonomy-audit"></span></div>

### [Routing, fusion, and answer selection](research-routing.md)

**Working answer:** use the baseline unless a label-blind specialist
rule applies; the repository has route seeds, not a validated router.

**Strongest evidence:** [taxonomy and residual-error
diagnostics](research-routing.md#taxonomy-audit).

**Open question:** is each failure missing evidence, unused evidence, or
wrong answer selection—and can a router abstain without regressions?

</div>

</div>

## Theme status matrix

This is a navigation layer, not a second results ledger. **Use**, **Do
not default**, **Possible**, **Blocked**, and **Next** are short status
tokens; each link opens the visible theme record with the tested
condition and limits.

<div class="research-matrix" role="region"
aria-label="Scrollable research theme status matrix" tabindex="0">

| Theme | Why it matters | Tried | Learned now | Still open | Next falsifying test |
|----|----|----|----|----|----|
| [Integrity](research-integrity.md#current-working-answer) | Makes comparisons interpretable | [v2 freeze](research-integrity.md#frozen-30-protocol-v2) | **Use:** v2; compatibility is not quality | **Possible:** reusable gate | [Invalid-fixture gate](research-integrity.md#reusable-integrity-gate) |
| [Visual](research-visual-evidence.md#current-working-answer) | Finds the useful moment within a frame budget | [Uniform/directed/burst](research-visual-evidence.md#directed-temporal-sampling); [r4 representation, answer, and headroom audits](research-visual-evidence.md#matched-r4-oracle-hybrid-12-inference-comparison) | **Use:** uniform-12; **Do not default:** tested oracle-hybrid-12 | **Possible:** learned retrieval or improved evidence use | [Selection-sensitive panel](research-visual-evidence.md#cheapest-valid-next-experiments) |
| [Speech](research-speech-transcripts.md#current-working-answer) | Adds spoken words with explicit provenance | [Whisper ablation](research-speech-transcripts.md#automatic-transcript-ablation) | **Use selectively:** accepted Whisper | **Blocked:** human tier; **Possible:** alternate ASR | [Selective Whisper route](research-speech-transcripts.md#distribution-level-selective-whisper) |
| [Native audio](research-native-audio.md#current-working-answer) | Preserves prosody and non-speech sound | [Audio on/off](research-native-audio.md#minicpm-audio-on-off) | **Do not default:** global audio on | **Possible:** routed sound/prosody | [Specialist panel](research-native-audio.md#matched-native-audio-specialist-panel) |
| [Models](research-models.md#current-working-answer) | Separates compatibility, quality, and adaptation | [Candidate screens](research-models.md#failure-panel) | **Use:** Qwen3.5-4B; **Fast:** MiniCPM-o | **Possible:** repaired/specialist model or adaptation | [Bottleneck-first decision](research-models.md#bottleneck-first-adaptation-decision) |
| [Routing](research-routing.md#current-working-answer) | Chooses evidence and separates retrieval from answering | [Route seeds](research-routing.md#taxonomy-audit) | **Use:** baseline fallback; router unvalidated | **Possible:** fusion and answer election | [Three-way measurement](research-routing.md#required-evidence-labels-and-three-way-measurement) |

</div>

## Experiment index

<details>

<summary>

Expand named experiments by theme
</summary>

### Experiment integrity and evaluation

- [July 12 field guide](research-integrity.md#field-guide)
- [Data readiness](research-integrity.md#data-readiness)
- [Three-video compatibility
  gate](research-integrity.md#compatibility-gate)
- [Frozen-30 protocol transition: v1 to
  v2](research-integrity.md#frozen-30-protocol-v2)
- [Scorer-reuse audit](research-integrity.md#scorer-reuse-audit)

### Visual and temporal evidence

- [Frozen-30 protocol v2: global frame
  budget](research-visual-evidence.md#frozen-30-protocol-v2)
- [Directed temporal
  sampling](research-visual-evidence.md#directed-temporal-sampling)
- [r4 required-moment Stage 1 representation
  audit](research-visual-evidence.md#temporal-required-moment-oracle)
- [Historical primary annotation-agreement
  stop](research-visual-evidence.md#historical-annotation-agreement-stop)
- [Dense-burst
  sampling](research-visual-evidence.md#dense-burst-sampling)
- [Spatial-crop
  diagnostic](research-visual-evidence.md#spatial-crop-diagnostic)

### Speech and transcript evidence

- [Transcript provenance
  tiers](research-speech-transcripts.md#transcript-provenance-tiers)
- [Automatic transcript
  ablation](research-speech-transcripts.md#automatic-transcript-ablation)
- [Matched full-panel frame/audio
  audit](research-speech-transcripts.md#matched-full-panel-frame-audio-audit)
- [Distribution-level selective Whisper
  route](research-speech-transcripts.md#distribution-level-selective-whisper)
- [Multilingual and translation
  follow-ups](research-speech-transcripts.md#multilingual-translation-followups)
- [Seamless
  diagnostic](research-speech-transcripts.md#seamless-diagnostic)
- [External dataset-reference
  research](research-speech-transcripts.md#external-dataset-reference-research)
- [Human-oracle tier](research-speech-transcripts.md#human-oracle-tier)

### Native audio and sound

- [Matched MiniCPM audio-on/off
  screen](research-native-audio.md#minicpm-audio-on-off)
- [Failure-selected native-audio
  diagnostic](research-native-audio.md#failure-panel-native-audio)
- [Native audio with
  Qwen3-Omni](research-native-audio.md#qwen3-omni-native-audio)
- [Sound and music diagnostic
  boundaries](research-native-audio.md#sound-music-boundary)

### Models and adaptation

- [Three-video compatibility
  gate](research-models.md#compatibility-gate-models)
- [Failure-selected Qwen3.6 screen](research-models.md#failure-panel)
- [VideoChat3 Temporal/Action specialist
  screen](research-models.md#videochat3-temporal-specialist)
- [Phi and Gemma semantic
  screen](research-models.md#phi-gemma-semantic-stop)
- [MiniCPM-o 4.5 fast
  alternate](research-models.md#minicpm-fast-alternate)
- [Qwen3-Omni quality and completion
  gate](research-models.md#qwen3-omni-quality)

### Routing, fusion, and answer selection

- [Taxonomy audit and route seeds](research-routing.md#taxonomy-audit)
- [Transcript residuals: evidence present, answer still
  wrong](research-routing.md#transcript-residuals)
- [Retrieval versus fusion
  diagnostic](research-routing.md#retrieval-versus-fusion)
- [Evidence-aware answer election](research-routing.md#answer-election)
- [Twenty-video direct-path and Borda
  diagnostic](research-routing.md#mc-poc20)
- [Two-stage adaptive-inspection
  diagnostic](research-routing.md#mc-adaptive-poc20)
- [Bounded single-model deliberative-agent
  diagnostic](research-routing.md#mc-deliberative-poc20)

</details>

## Three failure mechanisms to preview

These selected development cases explain mechanisms, not prevalence.
Open the [Failure Library · Report page](failure-library.md) for full
evidence traces and the disclosure boundary.

<div class="failure-preview-grid">

<div class="failure-preview">

### [Later-span answer selection](failure-library.md#case-001)

Needed speech was retrieved, but later numeric speculation was selected.

</div>

<div class="failure-preview">

### [Non-lexical event missed](failure-library.md#case-002)

An unrelated transcript passed the tested generic gate while the
requested vocal sound remained unrepresented.

</div>

<div class="failure-preview">

### [Unsupported audio specificity](failure-library.md#case-004)

Native audio produced a more specific music interpretation that the
evidence did not support.

</div>

</div>

<span id="what-to-investigate-next" class="legacy-fragment-alias"
aria-hidden="true"></span>

Before turning a branch into a run or result, use the [Adding
Experiments and Results guide · Report page](adding-experiments.md) to
declare its boundary and choose its primary theme.

## Branch-complete next-test queue

These branches are independent choices; their numbering is for
reference, not a prescribed research sequence.

1.  **Smarter visual sampling/retrieval:** [freeze a selection-sensitive
    panel where uniform-12 misses required
    evidence](research-visual-evidence.md#cheapest-valid-next-experiments);
    the original matched Stage 2 panel had full uniform interval
    coverage in `6/10` rows.
2.  **Alternative ASR:** [compare an alternative or multilingual ASR on
    one frozen, source-language-aware
    panel](research-speech-transcripts.md#alternative-or-multilingual-asr-matched-panel).
3.  **Human/reference transcript controls:** [produce an independently
    reviewed, label-blind upper
    bound](research-speech-transcripts.md#label-blind-human-transcript-upper-bound).
4.  **Routed native audio/sound:** [test prosody and non-speech strata
    with abstention and matched
    controls](research-native-audio.md#matched-native-audio-specialist-panel).
5.  **Repaired/specialist models or bottleneck-first adaptation:**
    [repair the strict Omni
    path](research-models.md#repaired-complete-neutral-qwen3-omni-run),
    or [measure the bottleneck before
    adapting](research-models.md#bottleneck-first-adaptation-decision).
6.  **Routing, fusion, and answer election:** [measure retrieval,
    oracle-evidence answering, and end-to-end answering
    separately](research-routing.md#required-evidence-labels-and-three-way-measurement),
    then test a fresh abstaining speech route and a [bounded
    multi-inspection evidence
    agent](research-routing.md#bounded-multi-inspection-evidence-agent);
    [blind Borda](research-routing.md#mc-poc20) and [one-window adaptive
    inspection](research-routing.md#mc-adaptive-poc20) are not promoted.
    The [deterministic single-model deliberative
    loop](research-routing.md#mc-deliberative-poc20) is also not
    promoted; its next valid form needs adversarial solver diversity and
    a challengeable stop decision on fresh data.

<span id="how-to-interpret-results" class="legacy-fragment-alias"
aria-hidden="true"></span>

## How to interpret evidence and decisions

Evidence validity describes whether and how evidence can be used.
Decision disposition describes what the project does now; the axes are
independent.

### Evidence validity

| Status | Meaning |
|----|----|
| **Verified** | The declared protocol completed and supports the stated condition-bound result. |
| **Diagnostic** | Deliberately limited evidence that selects a next test but does not estimate prevalence or general quality. |
| **Compatibility-only** | Execution or interface success without quality evidence. |
| **Incomplete** | The declared evidence bundle did not complete; partial observations cannot satisfy the planned comparison. |
| **Invalid** | A demonstrated integrity failure makes the affected comparison unusable for its intended claim. |
| **Blocked** | A named prerequisite is absent, so the planned evidence cannot yet be produced or interpreted. |
| **Unknown** | No admissible evidence establishes the claim. |

### Decision disposition

These labels describe what the project does now, not permanent
properties of a technique:

| Status | Meaning |
|----|----|
| **Current default** | Strongest available evidence for present use. |
| **Fast alternate** | Explicit cost or latency trade-off, not an accuracy claim on every metric. |
| **Routed candidate** | Consider only for a declared question or evidence type. |
| **Diagnostic signal** | Chooses the next test; does not estimate prevalence or general quality. |
| **Not promoted under the tested condition** | This implementation, panel, budget, or gate did not justify promotion; another controlled condition remains possible. |
| **Superseded** | Later valid evidence replaces the earlier conclusion. |
| **Open hypothesis** | Proposed but not yet tested. |

Validity concerns are separate from evidence validity or decision
disposition:

- **Confirmed defect — demonstrated fault.** Frozen-30 v1 could not
  demonstrate label isolation, so v1 cannot support the comparative
  conclusion.
- **Audit gap — exact path is not proven by tracked evidence.** The
  directed selector has source and tests but no tracked general-workflow
  caller; that is not proof that the diagnostic run was buggy.
- **Uncertainty — limited generalization without bug evidence.** Small
  or selected diagnostic panels constrain what can be claimed, even when
  the run itself is clean.

## History and provenance

### Issue and PR trail

<details>

<summary>

Expand the historical issue and PR trail
</summary>

These links preserve the implementation and review trail behind the
theme records. **Evidence report · GitHub** links are immutable evidence
of record; **Issue/PR history · GitHub** links preserve discussion and
review. GitHub requires repository access until public release. They are
provenance, not substitutes for the current condition-bound conclusions.

- **Foundation and repository recovery — [Experiment integrity and
  evaluation](research-integrity.md):** the [rendered July 12 field
  guide · Evidence of record · Evidence report ·
  GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/report.html)
  and [repository recovery PR \#21 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/21) record
  the background synthesis and restoration of the benchmark evidence.
- **Directed temporal sampling — [Visual and temporal
  evidence](research-visual-evidence.md#directed-temporal-sampling):**
  [closed issue \#8 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/8), its
  [conclusion audit in issue \#13 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/13), and
  [implementation PR \#10 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/10) preserve
  the tested heuristic’s decision trail.
- **Automatic and human transcripts — [Speech and transcript
  evidence](research-speech-transcripts.md):** the automatic-ablation
  work is tracked in [closed issue \#7 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/7) and [PR
  \#9 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/9); the
  independently produced human-transcript tier remains blocked in [open
  issue \#19 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/19).
- **Evidence taxonomy and routing — [Routing, fusion, and answer
  selection](research-routing.md#taxonomy-audit):** [closed issue \#4 ·
  Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/4) records
  the taxonomy and evidence-routing research thread.
- **Phi and Gemma quality screen — [Models and
  adaptation](research-models.md#phi-gemma-semantic-stop):**
  compatibility and semantic correction are tracked in [closed issue \#6
  · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/6), [closed
  issue \#12 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/12), [PR
  \#11 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/11), and [PR
  \#15 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/15).
- **Qwen3-Omni compatibility and quality — [Models and
  adaptation](research-models.md#qwen3-omni-quality):** the [tracked
  compatibility report · Evidence of record · Evidence report ·
  GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/docs/experiments/2026-07-24-qwen3-omni-dstack-compatibility.md),
  [closed issue \#5 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/5), [closed
  issue \#17 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/issues/17), [PR
  \#14 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/14), and
  recovery [PR \#20 · Issue/PR history ·
  GitHub](https://github.com/npatta01/trec_video_2026/pull/20) preserve
  the gate-specific trail.
- **VideoChat3 Temporal/Action specialist screen — [Models and
  adaptation](research-models.md#videochat3-temporal-specialist):** the
  [sanitized terminal retired record · Evidence of record ·
  GitHub](https://github.com/npatta01/trec_video_2026/blob/b8f80f00fa97fbdaddd08a143b4809a38809e456/dstack/videochat3-24frame-dev-20260801-a1.retired.yml)
  preserves both fixed-frame conclusions, terminal completion and cost,
  evidence validity, and the condition-bound negative decision without
  private rows, answers, or artifact locators.

</details>

### What superseded what

Six records explain which later evidence replaced an earlier project
statement.

<details>

<summary>

Expand six supersession records
</summary>

| Earlier statement or result | Superseding evidence | Current interpretation |
|----|----|----|
| July 12 architecture priorities | Executed panels through July 25 | Background field guide only; use this hub and its theme pages for current status. |
| Failure panel favored global Qwen24 | Frozen-30 protocol v2 | Qwen12 is the global default; global 24 frames added no exact answers under the tested condition. |
| Whisper as a broad additional modality | Frozen-30 plus automatic-transcript ablation | Route quality-gated Whisper only to lexical-speech work. |
| Initial Gemma compatibility promotion | Phi/Gemma semantic gate | Gemma and Phi were not promoted beyond the tested three-row semantic screen; compatibility is not quality. |
| Qwen3-Omni compatibility success | Neutral quality-10 A2 | The strict run is ineligible for its predeclared promotion decision after 19/20 outputs; the model remains eligible for a repaired fresh comparison. |
| Directed and dense temporal heuristics | Their matched diagnostic panels | The tested implementations were not promoted; smarter fixed-budget sampling remains an open hypothesis. |

</details>

### Chronology index

Seven milestones preserve the date-ordered audit trail without making
chronology the primary reading path.

<details>

<summary>

Expand the seven-milestone chronology
</summary>

| Date | Milestone | Thematic home |
|----|----|----|
| July 12 | Field guide | [Experiment integrity and evaluation](research-integrity.md#field-guide) |
| July 20 | Data readiness and compatibility gate | [Experiment integrity and evaluation](research-integrity.md#data-readiness) |
| July 21 | Audio/spatial diagnostics and frozen-30 | [Visual evidence](research-visual-evidence.md); [native audio](research-native-audio.md) |
| July 24 | Routing, temporal, and model diagnostics | [Routing](research-routing.md); [visual evidence](research-visual-evidence.md); [models](research-models.md) |
| July 25 | Strict Qwen3-Omni quality result | [Models and adaptation](research-models.md#qwen3-omni-quality) |
| July 30 | Historical required-moment annotation agreement stop; separate r4 Stage 1 representation selection | [Visual evidence](research-visual-evidence.md#temporal-required-moment-oracle) |
| August 1 | VideoChat3 Temporal/Action specialist screen | [Models and adaptation](research-models.md#videochat3-temporal-specialist) |

</details>
