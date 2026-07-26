# Start Here


<div class="lead">

This repository prepares video evidence, runs reproducible local VQA
experiments, and records sanitized findings for TRECVID 2026. The answer
today: start with **Qwen3.5-4B and 12 uniformly sampled frames**, then
add modalities only through narrow, quality-gated routes.

</div>

> [!NOTE]
>
> ### What is known—and what is not
>
> The recommendation comes from labeled 2025 training examples and
> controlled diagnostics. It is useful for choosing the next run; it is
> **not** an estimate of answer quality on the unlabeled 2026 evaluation
> set. That performance is unknown.

## Current technical default

- **Accuracy baseline:** Qwen3.5-4B with 12 uniform frames.
- **Fast alternate:** MiniCPM-o 4.5 with native audio disabled. It had
  the best METEOR and fastest measured outer wall in the matched
  frozen-30 run.
- **Lexical-speech route:** in a declared label-blind routed experiment,
  add Whisper large-v3 only when the question is classified as lexical
  speech and transcript quality passes. No versioned evaluated router
  exists yet.
- **Prosody or non-speech candidate:** test native audio as a specialist
  branch, never as a global switch on the current evidence.

The matched [frozen-30 protocol-v2
report](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md)
is the strongest current global comparison. The later
[automatic-transcript
ablation](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md)
strengthens the lexical-speech route, while the [failure-selected audio
panel](https://github.com/npatta01/trec_video_2026/blob/main/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md)
shows why native audio remains a specialist.

## Recommended router

``` mermaid
flowchart TD
  accTitle: Recommended evidence router
  accDescr: Start with Qwen and 12 uniform frames, add accepted Whisper text only for lexical speech, and reserve native audio for specialist experiments.
  Q[Question] --> B[Qwen3.5-4B<br/>12 uniform frames]
  Q --> R{Special evidence route?}
  R -->|Lexical speech| G{Whisper quality passes?}
  G -->|Yes| T[Add transcript<br/>report separately]
  G -->|No| B
  R -->|Prosody or non-speech| A[Native-audio specialist<br/>candidate only]
  R -->|No| B
```

**Figure caption:** Recommended evidence router: keep the 12-frame Qwen
baseline, use Whisper only for quality-gated lexical speech, and reserve
native audio for specialist evaluation.

Plain-text equivalent:

1.  Run Qwen3.5-4B on 12 uniform frames for every question.
2.  For lexical-speech questions, add Whisper large-v3 only after its
    output passes the label-blind quality gate; retain the frame-only
    result as the matched control.
3.  For prosody or non-speech sound, native audio is a candidate
    specialist experiment, not a promoted production route.
4.  Otherwise, continue with the 12-frame baseline.

## Current decision map

The table is the primary decision interface. It states the current use
of each system without requiring the expanded visual.

| Status | Systems or routes | Meaning now |
|----|----|----|
| Current default | Qwen3.5-4B, 12 uniform frames | Evidence-backed visual baseline. |
| Fast alternate | MiniCPM-o 4.5, audio off | Explicit speed/METEOR trade-off, not the accuracy winner on every metric. |
| Routed candidate | Quality-gated Whisper | Lexical-speech evidence only; not a global default. |
| Routed candidate | Native audio | Prosody and non-speech candidate; global use can distract visual reasoning. |
| Gate-specific ineligible run | Qwen3-Omni neutral A2 | Encouraging diagnostic, but this strict run produced 19/20 accepted outputs and cannot support its predeclared promotion decision. |
| Not promoted as current defaults under the tested conditions | Global 24 frames, directed temporal sampling, dense bursts, Qwen3.6, Gemma, Phi | Each tested implementation missed its applicable efficiency, semantic, or promotion gate; different controlled conditions remain possible. |

The Qwen3-Omni neutral A2 diagnostic changed frames-only **3C/3P/4W** to
audio-on **5C/2P/3W** after a narrow post-hoc answer recovery. The
immutable strict run still completed only **19/20** accepted outputs, so
that run is formally ineligible under its predeclared 20/20 promotion
gate. The model may be reconsidered in a repaired fresh comparison. See
the [tracked neutral A2
result](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-25-qwen3-omni-quality-10-neutral-a2.md)
and [PR \#18](https://github.com/npatta01/trec_video_2026/pull/18).

<details class="diagram-details">

<summary>

Expand the full decision diagram
</summary>

<div class="diagram-scroll" tabindex="0"
aria-label="Scrollable full current decision diagram">

``` mermaid
flowchart TD
  accTitle: Current project decisions
  accDescr: Qwen with 12 frames is the default, MiniCPM audio-off is the fast alternate, speech and native audio are routed candidates, several tested conditions were not promoted, and one Qwen3-Omni run missed its completion gate.
  E[Current evidence] --> D[Use now]
  E --> ROUTE[Route or study]
  E --> S[Do not default]
  D --> Q12[Qwen3.5-4B<br/>12 uniform frames]
  D --> M[MiniCPM-o 4.5<br/>audio off: fast alternate]
  ROUTE --> W[Quality-gated Whisper<br/>lexical speech]
  ROUTE --> NA[Native audio<br/>prosody / non-speech]
  ROUTE --> O[Qwen3-Omni A2 run<br/>19/20 gate-specific ineligibility]
  S --> X[Not promoted under tested conditions<br/>Global 24f; directed; bursts;<br/>Qwen3.6; Gemma; Phi]
```

**Figure caption:** Current decision map separating defaults, routed
candidates, condition-bound non-promotions, and Qwen3-Omni’s
gate-specific ineligible run.

</div>

</details>

## What not to infer

- **More frames are not automatically better.** Global Qwen 24 frames
  added no exact answers over 12 in frozen-30; a directed 12-frame
  heuristic tied uniform 12 and trailed uniform 24; a two-row
  dense-burst probe regressed.
- **A larger model is not automatically better.** Qwen3.6 scored
  3C/2P/7W on a deliberately failure-selected diagnostic and did not
  displace the smaller baseline.
- **Runtime compatibility is not answer quality.** Gemma passed an
  initial joint-input compatibility gate and Phi needed output handling
  correction, but neither was promoted beyond the later three-row
  semantic gate.
- **Native audio is not universally helpful.** It improved some prosody
  and sound cases while harming some temporal cases.

Follow the evidence, limitations, and next experiments in [Research
Themes](experiment-ledger.md).

## Choose a reading path

### Five minutes: decide what to run

1.  Read **Current technical default** and **Recommended router** above.
2.  Scan the four supported commands in the [Code
    Map](code-map.md#supported-commands).
3.  Before any substantial benchmark, read the repository [privacy and
    shared-data
    boundary](https://github.com/npatta01/trec_video_2026/blob/main/README.md).

### Fifteen minutes: understand the decision

1.  Complete the five-minute path.
2.  Read the [frozen-30 integrity
    record](research-integrity.md#frozen-30-protocol-v2) and
    [automatic-transcript
    record](research-speech-transcripts.md#automatic-transcript-ablation).
3.  Scan the [supersession
    table](experiment-ledger.md#what-superseded-what), then review
    [Remaining gaps](code-map.md#remaining-gaps).

### Thirty minutes: trace evidence to code

1.  Complete the fifteen-minute path.
2.  Choose relevant pages from [Research Themes](experiment-ledger.md).
3.  Walk the acquisition-to-report [system
    pipeline](code-map.md#system-pipeline) and [repository
    map](code-map.md#repository-map).
4.  Open the linked tracked reports or source modules for the decision
    you plan to revisit.

## Evidence vocabulary

| Term | Meaning in this report |
|----|----|
| **Verified** | The declared protocol completed, its required outputs and provenance checks passed, and the stated result was scored or reviewed within that protocol. |
| **Diagnostic** | A deliberately small, selected, or post-hoc comparison that can explain a mechanism but cannot estimate population accuracy. |
| **Compatibility-only** | The runtime, modality-input, and output contract ran; answer quality was not established. |
| **Not promoted under the tested condition** | This implementation, panel, budget, or gate did not justify promotion; it is not a permanent verdict. |
| **Blocked** | A required input, independent record, budget, or authorization is absent, so the planned tier cannot run validly yet. |
| **Unknown** | No valid evidence answers the question. In particular, 2026 evaluation answer quality is unknown. |

These categories prevent a clean runtime probe, an interesting failure
panel, or a post-hoc diagnostic from being reported as a baseline win.
For exact panels and decisions, continue to [Research
Themes](experiment-ledger.md).
