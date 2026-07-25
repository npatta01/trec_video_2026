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
- **Lexical-speech route:** add Whisper large-v3 only when a label-blind
  question/speech router selects the route and transcript quality
  passes.
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
4.  Otherwise, stop at the 12-frame baseline.

## Current decision map

``` mermaid
flowchart TD
  accTitle: Current project decisions
  accDescr: Qwen with 12 frames is the default, MiniCPM audio-off is the fast alternate, speech and native audio are routed candidates, and stopped approaches are not defaults.
  E[Current evidence] --> D[Use now]
  E --> R[Route or study]
  E --> S[Do not default]
  D --> Q12[Qwen3.5-4B<br/>12 uniform frames]
  D --> M[MiniCPM-o 4.5<br/>audio off: fast alternate]
  R --> W[Quality-gated Whisper<br/>lexical speech]
  R --> NA[Native audio<br/>prosody / non-speech]
  R --> O[Qwen3-Omni<br/>promising, ineligible]
  S --> X[Global 24f; directed; bursts;<br/>Qwen3.6; Gemma; Phi]
```

**Figure caption:** Current decision map separating defaults, routed
candidates, stopped approaches, and an ineligible but promising model.

| Status | Systems or routes | Meaning now |
|----|----|----|
| Use now | Qwen3.5-4B, 12 uniform frames | Evidence-backed default. |
| Use when speed dominates | MiniCPM-o 4.5, audio off | Fast alternate, not the accuracy winner on every metric. |
| Route | Quality-gated Whisper | Lexical-speech evidence only; not a global default. |
| Study as specialist | Native audio | Prosody and non-speech candidate; global use can distract visual reasoning. |
| Ineligible candidate | Qwen3-Omni | Encouraging answer-only diagnostic, but the strict A2 run passed only 19/20 outputs. |
| Stop as current defaults | Global 24 frames, directed temporal sampling, dense bursts, Qwen3.6, Gemma, Phi | Each failed its applicable efficiency, semantic, or promotion gate. |

The Qwen3-Omni neutral A2 diagnostic changed frames-only **3C/3P/4W** to
audio-on **5C/2P/3W** after a narrow post-hoc answer recovery, but the
immutable strict run still completed only **19/20** outputs. It is
therefore formally ineligible for promotion. See the [tracked neutral A2
result](https://github.com/npatta01/trec_video_2026/blob/main/docs/experiments/2026-07-25-qwen3-omni-quality-10-neutral-a2.md)
and [PR \#18](https://github.com/npatta01/trec_video_2026/pull/18).

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
  correction, but the later semantic gate stopped both.
- **Native audio is not universally helpful.** It improved some prosody
  and sound cases while harming some temporal cases.

Follow the experiments and their limitations in the [Experiment
Ledger](experiment-ledger.qmd).

## Choose a reading path

### Five minutes: decide what to run

1.  Read **Current technical default** and **Recommended router** above.
2.  Scan the four supported commands in the [Code
    Map](code-map.qmd#supported-commands).
3.  Before any substantial benchmark, read the repository [privacy and
    shared-data
    boundary](https://github.com/npatta01/trec_video_2026/blob/main/README.md).

### Fifteen minutes: understand the decision

1.  Complete the five-minute path.
2.  Read the [frozen-30
    entry](experiment-ledger.qmd#frozen-30-protocol-v2) and
    [automatic-transcript
    entry](experiment-ledger.qmd#automatic-transcript-ablation).
3.  Scan the [supersession
    table](experiment-ledger.qmd#what-superseded-what), then review
    [Remaining gaps](code-map.qmd#remaining-gaps).

### Thirty minutes: trace evidence to code

1.  Complete the fifteen-minute path.
2.  Read the full [Experiment Ledger](experiment-ledger.qmd).
3.  Walk the acquisition-to-report [system
    pipeline](code-map.qmd#system-pipeline) and [repository
    map](code-map.qmd#repository-map).
4.  Open the linked tracked reports or source modules for the decision
    you plan to revisit.

## Evidence vocabulary

| Term | Meaning in this report |
|----|----|
| **Verified** | The declared protocol completed, its required outputs and provenance checks passed, and the stated result was scored or reviewed within that protocol. |
| **Diagnostic** | A deliberately small, selected, or post-hoc comparison that can explain a mechanism but cannot estimate population accuracy. |
| **Compatibility-only** | The runtime, modality-input, and output contract ran; answer quality was not established. |
| **Stopped** | The work ran far enough to miss a declared promotion condition or to show no reason for expansion. |
| **Blocked** | A required input, independent record, budget, or authorization is absent, so the planned tier cannot run validly yet. |
| **Unknown** | No valid evidence answers the question. In particular, 2026 evaluation answer quality is unknown. |

These categories prevent a clean runtime probe, an interesting failure
panel, or a post-hoc diagnostic from being reported as a baseline win.
For exact panels and decisions, continue to the [Experiment
Ledger](experiment-ledger.qmd).
