# Start Here


<div class="lifecycle-banner" role="status"
aria-label="Project lifecycle">

**Lifecycle banner — current working recommendation.** Evidence is drawn
from labeled 2025 training examples and controlled diagnostics through
2026-07-25. Answer quality on the unlabeled 2026 evaluation set remains
**unknown**.

</div>

## Project in one minute

**Video question answering (VQA)** means answering a natural-language
question about a video. This repository prepares evidence, runs
reproducible local VQA experiments, and keeps a sanitized memory of what
the team has learned for TRECVID 2026.

The practical answer today is simple: begin with **Qwen3.5-4B and 12
uniformly sampled frames**, then add speech or audio evidence only
through narrow, quality-gated routes. This is a choice for the next
experiment, not a claim about 2026 evaluation accuracy.

## What the system sees

The baseline does not watch a video continuously. Its **frame budget**
is the maximum number of still images it may receive: currently 12.
**Uniform sampling** places those frames at evenly spaced times across
the clip so the whole duration is represented without looking at the
expected answer.

Some questions require words that were spoken. **Automatic speech
recognition (ASR)** converts audio into timestamped text; this project
currently tests Whisper for that job. A **lexical-speech** question asks
about the words spoken, rather than tone of voice, music, or another
non-speech sound.

The system records a **frozen prediction**: an answer saved before the
expected answer is revealed or used for scoring. Evidence selection is
**label-blind** when the expected answer and correctness label cannot
influence which route, frames, transcript, or model the system receives.

## Current recommendation

<div class="decision-strip" aria-label="Current project decisions">

<div class="decision-item">

**Use now**

Qwen3.5-4B, 12 uniform frames

</div>

<div class="decision-item">

**Use for speed**

MiniCPM-o 4.5, audio off

</div>

<div class="decision-item">

**Route selectively**

accepted Whisper for spoken-word questions

</div>

<div class="decision-item">

**MC submission baseline**

frozen fusion rank-one answer → fresh revised-option Gemini ranking

</div>

<div class="decision-item">

**MC review queue**

798 revised blind-audit flags → frozen V2 targeted-video candidate

</div>

<div class="decision-item">

**MC direct-video candidate**

2,423 fresh revised rankings + 20 explicit fresh-V1 fallbacks → frozen
V3

</div>

<div class="decision-item">

**Still testing**

category-routed native audio and repaired Qwen3-Omni

</div>

<div class="decision-item">

**Do not default**

global 24 frames and tested directed/burst heuristics

</div>

</div>

These labels are condition-bound. “Do not default” means the tested
version did not justify promotion; it is not a permanent verdict on
every future version.

## See one answer from start to finish

The [worked current-approach example · Report page](worked-example.md)
follows one question through frame sampling, the route decision,
constructed ASR evidence, a frozen answer, grading, and a post-hoc next
test. It is a fully disclosed, project-authored synthetic teaching
scenario—**not benchmark evidence and not an estimate of system
accuracy**.

## Learn from concrete failures

The [Failure Library · Report page](failure-library.md) preserves four
selected development cases as compact “what failed / what changed / what
happened / what to test next” records. Use it to study failure
mechanisms and competing explanations—not to estimate how often a
failure occurs across the benchmark.

## Evidence behind the recommendation

- **Accuracy baseline:** Qwen3.5-4B with 12 uniform frames.
- **Fast alternate:** MiniCPM-o 4.5 with native audio disabled. It had
  the best **METEOR** (a token-overlap score that gives partial credit
  for aligned words and some linguistic variants) and fastest measured
  outer wall in the matched frozen-30 run.
- **Lexical-speech route:** add Whisper large-v3 only when a label-blind
  router classifies the question as spoken-word evidence and transcript
  quality passes. No versioned evaluated router exists yet.
- **Prosody or non-speech candidate:** test native audio as a specialist
  branch, never as a global switch on the current evidence.
- **Multiple-choice operational baseline:** reuse the submitted fusion
  rank-one answer and rank all released options with discounted
  text-only Gemini. The full 2,443-row run completed for `$0.66126844`
  with no new video calls. It is unscored and therefore establishes
  format, provenance, and cost, not hidden-test quality. A separate
  ranking-blind text audit agreed with the submitted Top-1 on
  2,382/2,443 rows, but only 1,625 rows were both closest and
  exact/paraphrastically supported; use its 818-row private queue for
  targeted video review. The audit did not use video or hidden labels
  and therefore does not measure accuracy. That follow-up is now frozen
  as V2 (805 direct rows on the 818-row queue) and V3 (2,424 direct rows
  over the full panel), with explicit V1 fallback elsewhere or on
  terminal failures. Both exact-option audits pass, but neither can be
  called better before organizer scoring. The shared full batch cost
  `$12.28247138`, `$0.28247138` above its `$12.00` cap. A later bounded
  `$0.05033026` recovery produced valid direct rankings for seven of
  nine invalid outputs. Final V2b has 808/818 direct weak-queue rows;
  final V3b has 2,431/2,443 direct rows and twelve explicit fallbacks.
  The two persistent failures returned no video tokens across three
  representations; further paid retries stopped.

The matched [frozen-30 protocol-v2 report · Evidence of record ·
Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md)
is the strongest current global comparison. A **protocol version** is
the named, fixed revision of the rules, inputs, and scoring used for a
run. The later [automatic-transcript ablation · Evidence of record ·
Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/oracle_transcript_diagnostic_issue_7.md)
strengthens the lexical-speech route, while the [failure-selected audio
panel · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md)
shows why native audio remains a specialist. A **panel** is the declared
set of examples in a comparison; a **stratum** is a predeclared
subgroup, such as lexical speech, within that panel. These GitHub links
require repository access until public release.

For a file-level view of the sanitized dated records behind these
decisions, use the [experiment evidence
index](../../docs/experiments/README.md). The project-status pages
remain the source for current decisions and next tests.

<details class="diagram-details">

<summary>

Expand the full evidence router
</summary>

<div class="diagram-scroll" tabindex="0"
aria-label="Scrollable recommended evidence router">

<span id="fig-recommended-router" class="legacy-fragment-alias"
aria-hidden="true"></span>

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

</div>

Plain-text equivalent:

1.  Run Qwen3.5-4B on 12 uniform frames for every question.
2.  For lexical-speech questions, add Whisper large-v3 only after its
    output passes the label-blind quality gate; retain the frame-only
    result as the matched control.
3.  For prosody or non-speech sound, native audio is a candidate
    specialist experiment, not a promoted production route.
4.  Otherwise, continue with the 12-frame baseline.

</details>

## How to read current results

Results sometimes use **C/P/W**: counts graded correct, partially
correct, and wrong. Qwen3-Omni’s neutral A2 diagnostic changed
frames-only **3C/3P/4W** to audio-on **5C/2P/3W** after narrow post-hoc
answer recovery. The immutable strict run completed only **19/20**
accepted outputs, so it missed its predeclared **promotion gate**—the
pass/fail rule a result must satisfy before the project adopts it. That
run is ineligible for its planned promotion decision; the repaired model
remains open to a fresh comparison. See the [tracked neutral A2 result ·
Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/docs/experiments/2026-07-25-qwen3-omni-quality-10-neutral-a2.md)
and [PR \#18 · Issue/PR history ·
GitHub](https://github.com/npatta01/trec_video_2026/pull/18).

**Exact match** is 1 only when a normalized prediction equals the
normalized reference. **Token F1** balances word-level precision and
recall, so it can give partial credit for overlapping words. Neither
metric turns a small diagnostic into a population estimate.

### Evidence validity

Read every result with its evidence-validity label. These labels keep a
working runtime probe or a selected diagnostic from being mistaken for
promotion evidence.

| Status | What it means here |
|----|----|
| **Verified** | The declared protocol completed, required outputs and provenance checks passed, and the stated result was scored or reviewed within that protocol. |
| **Diagnostic** | A deliberately small, selected, or post-hoc comparison that may explain a mechanism but cannot estimate population accuracy. |
| **Compatibility-only** | The runtime, modality input, and output contract worked; answer quality was not established. |
| **Incomplete** | The declared evidence bundle did not complete; partial observations cannot satisfy the planned comparison. |
| **Invalid** | A demonstrated integrity failure makes the affected comparison unusable for its intended claim. |
| **Blocked** | A required input, independent record, budget, or authorization is absent, so the planned evidence tier cannot run validly yet. |
| **Unknown** | No valid evidence answers the question. In particular, answer quality on the 2026 evaluation set is **Unknown**. |

Evidence validity and answer grade are different: a prediction can be
correct inside a tiny diagnostic without making the tested method a
verified default. For what the project does now, read the hub’s separate
[decision-disposition
table](experiment-ledger.md#how-to-interpret-evidence-and-decisions).

### What not to infer

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

## Choose a reading path

### Five minutes: decide what to run

1.  Read **Current recommendation** above.
2.  Open [How One Answer Is Made · Report page](worked-example.md).
3.  Scan the four supported commands in the [Code Map · Report
    page](code-map.md#supported-commands).

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
2.  Choose relevant pages from [Research Themes · Report
    page](experiment-ledger.md).
3.  Walk the acquisition-to-report [system
    pipeline](code-map.md#system-pipeline) and [repository
    map](code-map.md#repository-map).

### Contributors: add a result without creating report drift

Read [Adding Experiments and Results · Report
page](adding-experiments.md) before turning a research idea, run, or
finding into the living report.

## Newcomer glossary

| Term | Plain-language meaning |
|----|----|
| **VQA** | Video question answering: answering a natural-language question about a video. |
| **Frame budget** | The maximum number of still images supplied to the model. |
| **Uniform sampling** | Selecting frames at evenly spaced times across a clip. |
| **ASR** | Automatic speech recognition: converting speech audio into timestamped text. |
| **Lexical speech** | The words spoken, as distinct from tone, music, or non-speech sound. |
| **Frozen prediction** | An answer saved before the expected answer is revealed or used for scoring. |
| **Label-blind** | Hidden expected answers and grades cannot influence evidence or route selection. |
| **C/P/W** | Counts graded correct / partially correct / wrong. |
| **Exact match** | A strict normalized-answer score: 1 for a match and 0 otherwise. |
| **Token F1** | A word-overlap score balancing precision and recall. |
| **METEOR** | A token-alignment score that can give partial credit for linguistic overlap. |
| **Panel** | The declared set of examples in a comparison. |
| **Stratum** | A predeclared subgroup within a panel. |
| **Promotion gate** | A rule a result must pass before the project adopts the tested change. |
| **Protocol version** | The named, fixed revision of a run’s rules, inputs, and scoring. |
