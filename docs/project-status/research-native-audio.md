# Native Audio and Sound


<span id="back-to-research-themes" class="legacy-fragment-alias"
aria-hidden="true"></span>
<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Native Audio
and Sound</span>
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

An automatic transcript can preserve spoken words but loses tone, music,
alarms, overlapping voices, and timing. A native-audio model can consume
that signal, but it may also distract from a visual or temporal
question. Treating all audio questions alike would hide that trade-off.

For example, “what word was spoken?” may be served by ASR, while “did
the voice sound angry?” needs prosody and “what made the popping sound?”
needs non-speech audio. Exact music-title questions are different again:
they may need fingerprint retrieval with abstention, not confident
generation.

<span id="technical-summary" class="legacy-fragment-alias"
aria-hidden="true"></span>

## Current working answer

Keep the 12-frame visual baseline as the default and the MiniCPM-o
audio-off path as the **Fast alternate**. Native audio is a **Routed
candidate** for a declared prosody or non-speech-sound question, not a
global switch. Exact music identification should abstain or use
fingerprint/retrieval evidence rather than ask a general VLM to invent a
title.

<div class="theme-decision-strip" aria-label="Native-audio decisions">

- **Use now:** the visual baseline; MiniCPM-o audio-off remains the fast
  alternate.
- **Do not default:** native audio on every question or general-VLM
  music-title guesses.
- **Still possible:** routed prosody, non-speech sound, fingerprinting,
  and repaired Omni audio.
- **Next test:** a matched specialist panel with audio-off fallback and
  route-level non-regression.

</div>

## Evidence at a glance

- **Global result:** MiniCPM audio on tied exact match and reduced
  F1/METEOR on its ten-row pre-v2 screen.
- **Localized signal:** selected diagnostics improved some prosody and
  sound cases while regressing temporal cases.
- **Confirmed defect:** Qwen3-Omni’s strict run completed only 19/20
  outputs.
- **Condition boundary:** these panels do not permanently reject routed
  native audio or another audio-capable model.

## What has been tried

### Matched MiniCPM audio-on/off screen

- **Reason for testing:** learn whether adding the same clip’s native
  audio to MiniCPM-o improves a matched visual path.
- **Tested condition and panel:** a 10-row pre-v2 gate compared
  MiniCPM-o 4.5 with 12-frame native-video input, audio off versus audio
  on.
- **Observed result:** audio off was `0.300 / 0.586 / 0.414` (exact
  match / token F1 / METEOR), while audio on was
  `0.300 / 0.471 / 0.283`: the exact match tied and F1/METEOR changed by
  `-0.115 / -0.131`.
- **Current condition-bound status:** **Not promoted under the tested
  condition** as a global audio-on policy.
- **What this rules out now:** enabling native audio by default for this
  MiniCPM version and 10-row gate.
- **What it does not rule out:** a matched specialist route for prosody,
  non-speech sound, or another model/prompt/panel.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—this
  pre-v2 ten-row screen cannot estimate category-specific or global
  value.
- **Reconsider when:** a stratified, audio-off/audio-on comparison meets
  a predeclared target-stratum gain and visual/temporal-control
  non-regression.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

This was pre-v2 screening evidence, not part of frozen-30’s
four-condition ranking. Its promotion rule also required non-speech
wins, multilingual wins, and no temporal/control loss; it did not pass.
The [protocol-v2 report · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md)
records the gate values and explicitly calls for a new controlled
native-audio test rather than assuming it helps.
</details>

<a name="failure-panel-native-audio"></a>

### Failure-selected native-audio diagnostic

- **Reason for testing:** determine whether native audio recovers known
  failures involving voice delivery, sound, music, and brief actions.
- **Tested condition and panel:** MiniCPM-o 4.5 with the same 12 frames,
  audio off versus full native audio, on 12 deliberately selected
  failures: two dialogue/prosody, four sound/music, four
  temporal-action, and two visual controls.
- **Observed result:** audio on was `4C/3P/5W` versus audio off
  `4C/4P/4W`. It improved emotional prosody, a popping sound, and
  instrument-family plausibility, but lost a bell and two temporal rows;
  its blind ordinal score rose from 4 to 6 on the first six audio-heavy
  rows and fell from 12 to 11 overall.
- **Current condition-bound status:** **Diagnostic signal** for
  category-gated native audio; **Not promoted under the tested
  condition** as a global switch.
- **What this rules out now:** treating this MiniCPM audio-on
  configuration as a better all-question policy on the failure-selected
  panel.
- **What it does not rule out:** native audio for declared prosody or
  non-speech-sound strata, a different audio-capable model, or an audio
  route with abstention.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  panel was selected from known failures and is not a prevalence sample.
- **Reconsider when:** a representative matched panel shows a
  target-stratum gain while preserving visual-control and temporal
  performance.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The first launch had a container-mount error before inference; the
corrected run completed all 12 requests. That operational repair is not
evidence of quality. This panel was selected from known failures, so it
can diagnose recovery behavior but cannot estimate full-test prevalence
or rank systems. See the [failure-panel report · Evidence of record ·
Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md).
</details>

### Native audio with Qwen3-Omni

- **Reason for testing:** check whether a larger joint audio/vision
  model can use native audio while holding the 12-frame visual evidence
  fixed.
- **Tested condition and panel:** a 10-row difficult, failure-selected
  panel compared Qwen3-Omni frames-only with the same frames plus native
  audio under a modality-neutral prompt.
- **Observed result:** the semantic diagnostic changed from `3C/3P/4W`
  to `5C/2P/3W`: prosody and non-speech sound improved from partial to
  correct, and one action improved from wrong to partial; both visual
  controls stayed correct. The unchanged strict run completed only 19 of
  20 required outputs.
- **Current condition-bound status:** **Diagnostic signal**; it is
  formally ineligible for promotion under the predeclared 20/20
  completion gate.
- **What this rules out now:** promoting that incomplete strict run or
  calling native audio a new overall default from this ten-row
  diagnostic.
- **What it does not rule out:** a repaired, complete, blinded
  Qwen3-Omni comparison, or a category route whose gate is met on a
  representative panel.
- **Possible defect, audit gap, or uncertainty:** **Confirmed
  defect**—one strict-parser output failed the predeclared completion
  contract.
- **Reconsider when:** terminal-output handling passes a sentinel check
  and a fresh full paired run completes 20/20 under one frozen contract.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

One complete answer was wrapped in a Markdown JSON fence and failed the
unchanged strict parser; a post-hoc semantic diagnostic cannot change
that eligibility outcome. The panel is selected and frame-based, so it
is neither a global ranking nor a complete native-video test.
Model-specific gate details are on [Models and
Adaptation](research-models.md#qwen3-omni-quality). See the [quality-10
report · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/docs/experiments/2026-07-25-qwen3-omni-quality-10.md).
</details>

### Sound and music diagnostic boundaries

- **Reason for testing:** distinguish non-speech sound recognition from
  speech transcription and exact music-title retrieval.
- **Tested condition and panel:** frozen-30 includes six non-speech
  sound/music rows; the failure panel additionally includes four
  selected sound/music failures.
- **Observed result:** no frozen-30 condition produced an exact match in
  the non-speech sound/music stratum. In the failure panel, native audio
  could recognize some sound character but produced plausible wrong
  music and bell answers; general VLMs did not establish exact-title
  retrieval.
- **Current condition-bound status:** **Open hypothesis** for a
  dedicated sound route; exact music-title retrieval is **Not promoted
  under the tested general-VLM conditions**.
- **What this rules out now:** using ASR text as a substitute for
  non-speech audio or asking the current general VLMs to guess an exact
  composition title.
- **What it does not rule out:** an audio-event classifier, fingerprint
  lookup, or a provenance-safe music retrieval system that can abstain.
- **Possible defect, audit gap, or uncertainty:**
  **Uncertainty**—six-row and selected failure panels cannot establish
  catalog coverage or prevalence.
- **Reconsider when:** a route names its evidence source, abstention
  behavior, and target-stratum control before a matched evaluation.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Frozen-30 is the stronger panel but has only six rows per stratum; the
failure-selected rows are diagnostic only. The [protocol-v2 report ·
Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md)
and [failure-panel report · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md)
support this separation.
</details>

## Known failures and cases

- [CASE-002 — Non-lexical event missed by transcript
  gate](failure-library.md#case-002) motivates a sound route rather than
  forcing non-lexical evidence through ASR.
- [CASE-004 — Unsupported native-audio
  specificity](failure-library.md#case-004) shows why native sound needs
  grounded labels or abstention.

### What the evidence supports

- Speech text and native sound are complementary evidence types, not
  interchangeable routes.
- No global audio-on condition has earned promotion over its audio-off
  control under a complete eligible gate; the encouraging Qwen3-Omni A2
  semantic signal remains ineligible because that run completed only
  19/20 strict outputs.
- Prosody and non-speech sound are plausible specialist strata because
  two separate selected diagnostics show localized native-audio gains.
- Exact music-title questions require a retrieval-style hypothesis; no
  current general-VLM result validates confident title generation.

### What it does not rule out

- A negative global switch does not reject a quality-gated native-audio
  route.
- The selected panels do not estimate how often prosody, sound, or music
  occur in the 2026 evaluation set.
- The Qwen3-Omni diagnostic does not settle the quality of a complete
  run, a different prompt, or native temporal video input.
- Audio-event recognition is not established merely because a model
  generated a plausible sound label.

<span id="possible-bugs-confounds-and-audit-gaps"
class="legacy-fragment-alias" aria-hidden="true"></span>

### Confirmed defects, audit gaps, and uncertainties

- **Uncertainty:** the MiniCPM and Omni audio findings are on pre-v2 or
  deliberately selected panels, not representative prevalence samples.
- **Confirmed defect:** the Qwen3-Omni strict run had one
  parser-contract failure. It is a completion-gate defect, not proof
  that its semantic output is useless.
- **Uncertainty:** changing native audio can alter how a model allocates
  attention; a temporal regression need not mean the sound evidence was
  absent.
- **Audit gap:** the repository has route ideas and diagnostics, not a
  versioned evaluated audio router with abstention and route-level
  non-regression.
- **Audit gap:** the failure-panel report names localized audio gains
  and losses but does not retain a complete transition matrix for every
  audio-on/off row; that missing detail is not evidence of an execution
  bug.

## Open hypotheses and ideas

- Native audio improves prosody and non-speech sound when the question
  is explicitly routed, while it can hurt unrelated temporal or visual
  answers.
- A lightweight non-speech audio classifier can decide when to invoke a
  native audio model and otherwise abstain to an audio-off control.
- Music fingerprinting or retrieval can answer exact-title questions
  more reliably than a general VLM, provided it can abstain on weak
  matches.

## Cheapest valid next experiments

### Matched native-audio specialist panel

- **Hypothesis:** audio on improves prosody and non-speech sound more
  than audio off without degrading visual or temporal controls.
- **Smallest valid panel or work item:** a frozen, label-blind panel
  with speech, prosody, non-speech, temporal, and visual-control strata.
- **Matched controls and integrity safeguards:** identical clips,
  frames, model, prompt, decoding, ordering, scoring, audio-off control,
  and frozen answers before labels load.
- **Decision or promotion rule:** advance a route only with a
  predeclared paired target-stratum gain and no worse than the declared
  temporal/control regression limit.
- **Boundary on what a positive result would establish:** a category
  route for the tested model and strata, not universal native-audio
  benefit.

### Non-speech classifier with abstention

- **Hypothesis:** a non-speech classifier can safely select native audio
  only when its signal is relevant.
- **Smallest valid panel or work item:** a hand-reviewed, label-blind
  sound versus non-sound development slice with an audio-off/native-VLM
  control.
- **Matched controls and integrity safeguards:** version the classifier
  and threshold; hold visual evidence and downstream prompt fixed;
  record route, abstention, and fallback; freeze predictions before
  scoring.
- **Decision or promotion rule:** adopt only if routed end-to-end
  quality beats audio-off without route-level or aggregate
  non-regression failures.
- **Boundary on what a positive result would establish:** selection
  quality for that classifier and slice, not sound recognition in every
  genre.

### Exact-music fingerprint or retrieval route

- **Hypothesis:** a fingerprint/retrieval path with abstention answers
  exact music-title questions more reliably than a general VLM guess.
- **Smallest valid panel or work item:** a provenance-approved
  music-title diagnostic with known-answer controls and no-match cases.
- **Matched controls and integrity safeguards:** freeze the
  fingerprint/index version and matching threshold; include audio-off
  and general-VLM controls; retain abstentions; score only after
  predictions freeze.
- **Decision or promotion rule:** advance only when exact retrieval
  improves the declared title cases, weak matches abstain rather than
  fabricate, and a prevalence estimate shows enough routed cases to
  provide useful expected aggregate value for the added cost.
- **Boundary on what a positive result would establish:** the retrieval
  route’s performance on its indexed catalog and declared prevalence
  slice, not general music knowledge or value on a distribution with a
  different title frequency.

## Method, result, validity risks, and evidence

<details>

<summary>

How to audit this theme’s records
</summary>

The visible summaries separate lexical speech, prosody, non-speech
sound, and music retrieval. Open each record’s native details block for
model/panel conditions, source links, and validity risks. Selected gains
never override a failed completion gate or establish a global audio-on
policy.
</details>

## Related themes

- [Speech and Transcript Evidence](research-speech-transcripts.md)
  covers lexical speech and transcript provenance.
- [Models and Adaptation](research-models.md) records the model-specific
  Qwen3-Omni completion and promotion boundary.
- [Routing, Fusion, and Answer Selection](research-routing.md) explains
  how a specialist route must abstain and fall back.
