# Failure Library


<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Failure
Library</span>
</nav>

<div class="lead">

These cases preserve concrete failure mechanisms and the next test that
could disprove each diagnosis. They are working research records, not
examples of the full benchmark distribution.

</div>

> [!WARNING]
>
> ### Selected development evidence only
>
> This is a selected development-case library, not a prevalence estimate
> or leaderboard. Repeatedly inspected cases are development evidence,
> never held-out validation. No 2026 evaluation row is admissible.
> Expected answers and media appear only under the recorded disclosure
> and rights policy.

## How to read the records

**Status key:** “Selected development case” means a deliberately chosen
2025 training row that has been repeatedly inspected. It is useful for
debugging a mechanism, but cannot validate a policy or estimate how
often the mechanism occurs.

**Diagnosis key:** “Post-hoc diagnosis” means an answer-informed
explanation written after grading. It is a testable project
interpretation, not a recorded model observation or established cause.
Each card therefore includes a competing explanation and a next
falsifying test.

The frozen predictions and grades below are project-generated outputs
and project adjudications. The source questions, expected answers,
transcripts, media, and source identifiers remain withheld. No case has
a public-safe media attestation, so this page is text-only.

## Cases

<div class="failure-case">

<span id="case-001" class="legacy-fragment-alias"
aria-hidden="true"></span>

### CASE-001 — Later-span answer selection

<p class="case-meta">

<strong>CASE-001</strong> · Selected development case · Owner: TRECVID
VQA project · Last verified: 2026-07-26 · Source/split role: 2025
training row · Theme:
<a href="research-routing.md#transcript-residuals">Routing, fusion, and
answer selection</a>
</p>

<div class="case-fields">

<div class="case-field">

#### What failed

In a spoken-answer case, a brief response was followed by longer numeric
speculation. The system selected the later material instead of the
needed span.

</div>

<div class="case-field">

#### Change tried

Automatic transcript evidence was added to the frames-only condition.

</div>

<div class="case-field">

#### Result

Frames only predicted `2012`; frames plus automatic transcript predicted
`The horse is 22 years old.` Both were graded **wrong**. Manual
accuracy, exact match, token F1, and METEOR were all `0.0`.

</div>

<div class="case-field">

#### Post-hoc diagnosis plus competing explanation

**Post-hoc diagnosis:** the needed span was retrieved, but a later span
was selected. A competing explanation is that the answerer preferred the
longer, more explicit numeric statement rather than later position
itself.

</div>

<div class="case-field">

#### Limit

This is one repeatedly inspected development case (`n=1`), not held-out
validation or evidence of prevalence.

</div>

<div class="case-field">

#### Next falsifying test

Keep the evidence and answerer fixed; compare the current prompt with a
condition that masks later speculation and one that preserves it but
changes its position. The diagnosis is weakened if selection does not
track the later span.

</div>

</div>

<details>

<summary>

Evidence received, run details, grading, and provenance
</summary>

| Field | Recorded value |
|----|----|
| Evidence configuration | `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; 12 frozen chronological frames (256²–512²/frame); greedy decoding; 32-token cap; seed `20260724`. Conditions: frames only versus the same frames plus an automatic transcript from `openai/whisper-large-v3` revision `06f233fe06e710322aca913c1bc4249a0d71fce1`; native audio absent. |
| Selected timestamps, spans, or events | not recorded |
| Model observation | Not recorded |
| Project-generated frozen prediction | Frames only: `2012`; frames plus automatic transcript: `The horse is 22 years old.` |
| Expected answer | `withheld` |
| Project grade | Both `wrong`; manual accuracy, exact match, token F1, and METEOR all `0.0` |
| Freeze and grading provenance | Paired predictions were frozen before references were loaded; manual adjudication occurred after the freeze |
| Rater provenance | Rater count not recorded |
| Blinding provenance | Condition blinding not recorded |
| Adjudication provenance | Separate adjudicator not recorded |
| Uncertainty | Post-hoc diagnosis; competing explanation remains viable; `n=1` selected development evidence |
| Next comparison | Position-versus-explicitness comparison with evidence and answerer held fixed |
| Immutable evidence links | [Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md) |
| Supersession | No row-level supersession; the broader transcript policy is quality-gated and routed, not global |

</details>

</div>

<div class="failure-case">

<span id="case-002" class="legacy-fragment-alias"
aria-hidden="true"></span>

### CASE-002 — Non-lexical event missed by transcript gate

<p class="case-meta">

<strong>CASE-002</strong> · Selected development case · Owner: TRECVID
VQA project · Last verified: 2026-07-26 · Source/split role: 2025
training row · Theme: <a href="research-speech-transcripts.md">Speech
and transcript evidence</a>
</p>

<div class="case-fields">

<div class="case-field">

#### What failed

The case asked the system to identify the clip’s only non-lexical vocal
sound. The automatic transcript did not represent that event.

</div>

<div class="case-field">

#### Change tried

Automatic transcript evidence was added to the frames-only condition and
passed through the old generic gate.

</div>

<div class="case-field">

#### Result

Frames only predicted `No`; frames plus automatic transcript predicted
`Thank you for watching!` Both were graded **wrong**. Manual accuracy,
exact match, token F1, and METEOR were all `0.0`.

</div>

<div class="case-field">

#### Post-hoc diagnosis plus competing explanation

**Post-hoc diagnosis:** the ASR missed the non-lexical event and an
unrelated phrase passed the old generic gate. A competing explanation is
that the gate was not the decisive failure because text-only ASR cannot
reliably encode the requested sound in the first place.

</div>

<div class="case-field">

#### Limit

This is one repeatedly inspected development case (`n=1`), not held-out
validation or evidence of prevalence.

</div>

<div class="case-field">

#### Next falsifying test

On a declared non-lexical development panel, compare the old generic
text gate with a label-blind non-lexical route and a frames-only
control. The diagnosis is weakened if rejecting unrelated transcript
text does not reduce this failure pattern.

</div>

</div>

<details>

<summary>

Evidence received, run details, grading, and provenance
</summary>

| Field | Recorded value |
|----|----|
| Evidence configuration | `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; 12 frozen chronological frames (256²–512²/frame); greedy decoding; 32-token cap; seed `20260724`. Conditions: frames only versus the same frames plus an automatic transcript from `openai/whisper-large-v3` revision `06f233fe06e710322aca913c1bc4249a0d71fce1` under the old generic gate; native audio absent. |
| Selected timestamps, spans, or events | not recorded |
| Model observation | Not recorded |
| Project-generated frozen prediction | Frames only: `No`; frames plus automatic transcript: `Thank you for watching!` |
| Expected answer | `withheld` |
| Project grade | Both `wrong`; manual accuracy, exact match, token F1, and METEOR all `0.0` |
| Freeze and grading provenance | Paired predictions were frozen before references were loaded; manual adjudication occurred after the freeze |
| Rater provenance | Rater count not recorded |
| Blinding provenance | Condition blinding not recorded |
| Adjudication provenance | Separate adjudicator not recorded |
| Uncertainty | Post-hoc diagnosis; competing explanation remains viable; `n=1` selected development evidence |
| Next comparison | Old generic text gate versus label-blind non-lexical route and frames-only control |
| Immutable evidence links | [Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md) |
| Supersession | Old generic acceptance behavior is superseded by repaired quality gates; a dedicated non-lexical check remains open |

</details>

</div>

<div class="failure-case">

<span id="case-003" class="legacy-fragment-alias"
aria-hidden="true"></span>

### CASE-003 — Retrieved token unused by answerer

<p class="case-meta">

<strong>CASE-003</strong> · Selected development case · Owner: TRECVID
VQA project · Last verified: 2026-07-26 · Source/split role: 2025
training row · Theme:
<a href="research-routing.md#transcript-residuals">Routing, fusion, and
answer selection</a>
</p>

<div class="case-fields">

<div class="case-field">

#### What failed

In a final-spoken-token extraction case, two text evidence sources
contained the needed token, but the frozen answer did not change.

</div>

<div class="case-field">

#### Change tried

The same answerer was compared across frames-only, automatic-transcript,
and uploader-caption conditions.

</div>

<div class="case-field">

#### Result

All three conditions predicted `Justice`. All were graded **wrong**,
with manual accuracy `0.0`.

</div>

<div class="case-field">

#### Post-hoc diagnosis plus competing explanation

**Post-hoc diagnosis:** two text sources contained the reference token,
but the frozen answer did not change. A competing explanation is that
the answerer did not recognize the task as final-token extraction,
rather than failing to use text evidence generally.

</div>

<div class="case-field">

#### Limit

This is one repeatedly inspected development case (`n=1`), not held-out
validation or evidence of prevalence.

</div>

<div class="case-field">

#### Next falsifying test

Hold the two text sources and answerer fixed; compare the current prompt
with a label-blind final-token extraction instruction and a matched
non-final-token control. The diagnosis is weakened if explicit
extraction does not change the answer while the control behaves
normally.

</div>

</div>

<details>

<summary>

Evidence received, run details, grading, and provenance
</summary>

| Field | Recorded value |
|----|----|
| Evidence configuration | `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; 12 frozen chronological frames (256²–512²/frame); greedy decoding; 32-token cap; seed `20260724`. Three conditions: frames only; the same frames plus an automatic transcript from `openai/whisper-large-v3` revision `06f233fe06e710322aca913c1bc4249a0d71fce1`; and the same frames plus the verified uploader-caption reference. Native audio was absent. |
| Selected timestamps, spans, or events | not recorded |
| Model observation | Not recorded |
| Project-generated frozen prediction | `Justice` in frames-only, automatic-transcript, and uploader-caption conditions |
| Expected answer | `withheld` |
| Project grade | All `wrong`; manual accuracy `0.0` |
| Freeze, scoring, and caption provenance | All three predictions were frozen before label access; scoring occurred after the freeze; caption coverage was verified |
| Rater provenance | Rater count not recorded |
| Blinding provenance | Condition blinding not recorded |
| Adjudication provenance | Separate adjudication not recorded |
| Uncertainty | Post-hoc diagnosis; competing explanation remains viable; `n=1` selected development evidence |
| Next comparison | Current prompt versus label-blind final-token instruction and matched control |
| Immutable evidence links | [Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md) |
| Supersession | No row-level supersession; uploader captions remain distinct provenance, not a human oracle |

</details>

</div>

<div class="failure-case">

<span id="case-004" class="legacy-fragment-alias"
aria-hidden="true"></span>

### CASE-004 — Unsupported native-audio specificity

<p class="case-meta">

<strong>CASE-004</strong> · Selected development case · Owner: TRECVID
VQA project · Last verified: 2026-07-26 · Source/split role: 2025
training row · Theme: <a href="research-native-audio.md">Native audio
and omni models</a>
</p>

<div class="case-fields">

<div class="case-field">

#### What failed

In a brief non-speech-event case near the end of a clip, native audio
produced an unsupported, overly specific music interpretation.

</div>

<div class="case-field">

#### Change tried

Native audio was enabled and compared with an audio-off condition.

</div>

<div class="case-field">

#### Result

Audio off predicted `Wedding bells`; native audio predicted
`Wedding march music.` The native-audio result was graded **Wrong**:
exact match `0.0`, token F1 `0.0`, and METEOR `0.125`.

</div>

<div class="case-field">

#### Post-hoc diagnosis plus competing explanation

**Post-hoc diagnosis:** native audio produced an unsupported, overly
specific music interpretation. A competing explanation is that answer
wording became too specific even if the underlying sound representation
improved.

</div>

<div class="case-field">

#### Limit

This is one repeatedly inspected development case (`n=1`), not held-out
validation or evidence of prevalence.

</div>

<div class="case-field">

#### Next falsifying test

Hold native audio and decoding fixed; compare the current prompt with a
prompt that requires the least-specific supported event label, plus an
audio-off control. The diagnosis is weakened if native audio remains
wrong without the specific music interpretation.

</div>

</div>

<details>

<summary>

Evidence received, run details, grading, and provenance
</summary>

| Field | Recorded value |
|----|----|
| Evidence configuration | `openbmb/MiniCPM-o-4_5` revision `1f761131fa83f5ed3cd6f2f22b225c4501d154fa`; 12 frames. Conditions: audio off versus full 16-kHz native audio. |
| Selected timestamps, spans, or events | not recorded |
| Model observation | Not recorded |
| Project-generated frozen prediction | Audio off: `Wedding bells`; native audio: `Wedding march music.` |
| Expected answer | `withheld` |
| Project grade | Native audio `Wrong`; exact match `0.0`, token F1 `0.0`, METEOR `0.125` |
| Rater provenance | One independent advisor reviewed answers; no multi-rater count recorded |
| Blinding provenance | The independent advisor reviewed answers without model identities |
| Adjudication provenance | No adjudication process recorded |
| Uncertainty | Post-hoc diagnosis; competing explanation remains viable; `n=1` selected development evidence |
| Next comparison | Current prompt versus least-specific supported-label prompt, with audio-off control |
| Immutable evidence links | [Evidence of record · Evidence report · GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md) |
| Supersession | The old panel’s global 24-frame default is superseded by frozen-30’s global 12-frame default; this row still supports category-gated, not global, native audio |

</details>

</div>

## Library boundary

These four selected cases are a mechanism-oriented development set.
Their count is neither a frequency estimate nor a model ranking. A
future case may be added only after its public fields are independently
verified, its expected answer remains withheld unless redistribution is
attested, and any media has a complete rights record.
