# How One Answer Is Made


<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<span aria-current="page">How One Answer Is Made</span>
</nav>

<div class="trace-label">

**one traced example, not the policy**

</div>

## Why this example

> [!WARNING]
>
> ### Constructed teaching content—not benchmark evidence
>
> Every question, scene note, timestamp, transcript line, prediction,
> expected answer, grade, and provenance field on this page was invented
> by the TRECVID VQA project for teaching. No real TRECVID annotation,
> frame, transcript, audio, answer, or model run appears here. No pixels
> or audio are published.

The scenario makes the current approach inspectable without
redistributing source annotations or media. It illustrates what a
lexical-speech route is supposed to record; it does not demonstrate that
the router or model works.

## 1. Question and difficulty

**Constructed teaching question:** “What drink does the speaker say they
packed for lunch?”

This fictional question is difficult for a frame-only system because the
drink is named in speech and the imagined thermos has no visible label.
It is a **lexical-speech** question: the requested evidence is a spoken
word or phrase, not tone of voice or a non-speech sound.

## 2. What the baseline receives

The illustrated baseline receives the constructed question and a
**12-frame budget** sampled uniformly across an invented 22-second clip.
Uniform sampling means selecting evenly spaced timestamps, here every
two seconds. The system does not receive the constructed expected answer
or grade.

<div class="timestamp-storyboard"
aria-label="Text-only timeline of twelve uniformly sampled timestamps">

<div class="timestamp-card">

**00:00** Fictional scene note: lunch bag on a kitchen table.

</div>

<div class="timestamp-card">

**00:02** Fictional scene note: speaker enters.

</div>

<div class="timestamp-card">

**00:04** Fictional scene note: unlabelled thermos appears.

</div>

<div class="timestamp-card">

**00:06** Fictional scene note: thermos is lifted.

</div>

<div class="timestamp-card">

**00:08** Fictional scene note: speaker is talking.

</div>

<div class="timestamp-card">

**00:10** Fictional scene note: thermos enters bag.

</div>

<div class="timestamp-card">

**00:12** Fictional scene note: fruit is added.

</div>

<div class="timestamp-card">

**00:14** Fictional scene note: bag closes.

</div>

<div class="timestamp-card">

**00:16** Fictional scene note: speaker lifts bag.

</div>

<div class="timestamp-card">

**00:18** Fictional scene note: speaker walks away.

</div>

<div class="timestamp-card">

**00:20** Fictional scene note: table is empty.

</div>

<div class="timestamp-card">

**00:22** Fictional scene note: end of constructed clip.

</div>

</div>

**Text equivalent:** At 00:00 a lunch bag is on a kitchen table; a
speaker enters, handles an unlabelled thermos, speaks, places it in the
bag, adds fruit, closes and carries the bag away, leaving the table
empty by 00:20. The invented clip ends at 00:22. None of the twelve
constructed scene notes visually names the drink.

## 3. Route decision

**Constructed route record:**
`lexical_speech → request ASR → quality accepted`.

The route is **label-blind** in the scenario: it uses only the question
wording and ASR quality fields, never the constructed expected answer or
grade. The phrase “does the speaker say” requests spoken words, so the
scenario follows the selective Whisper branch. See the complete
[evidence router · Report page](index.md#fig-recommended-router).

## 4. Evidence added

**Constructed ASR evidence:**
`[00:07.200–00:09.400] “I packed green tea for lunch.”`

This line is project-authored fiction, not an excerpt from a source
video or a real Whisper run. In the scenario it passes the declared
checks: non-empty text, timestamps within the 22-second duration, no
repeated failure phrase, and direct relevance to a spoken-word question.
The original 12-frame result would still be retained as the matched
control in a real experiment.

## 5. Recorded model observation

**Not recorded.** The teaching record contains an answer field, not
private reasoning, chain of thought, an evidence quote, or a
model-authored rationale. The route and diagnosis on this page are
project-authored descriptions; they must not be presented as hidden
model reasoning.

## 6. Frozen prediction

**Constructed frozen prediction:** `green tea`.

“Frozen” means that, within this narrated protocol, the answer record is
saved before the constructed expected answer is disclosed or scored.
This value is illustrative teaching content, not output recovered from
an actual model run.

## 7. Expected-answer disclosure and grade

After the constructed prediction is frozen, the teaching reference is
opened.

- **Constructed expected answer:** `green tea`
- **Constructed manual grade:** `C` (correct)
- **Constructed normalized exact match:** `1.0`
- **Constructed token F1:** `1.0`

Exact match gives full credit because the normalized strings are
identical; token F1 gives full credit because both tokens overlap. These
scores explain the mechanics only and are not benchmark measurements.

## 8. Post-hoc diagnosis and next test

**Post-hoc diagnosis:** In this constructed trace, the added text
contains the answer phrase and the constructed prediction matches it.
That is an answer-informed description written after the teaching
reference is opened, not a model observation and not evidence that
transcripts usually help.

**Next falsifying test:** On a declared real development panel, freeze
the question classifier, Whisper quality gate, Qwen model, prompt, 12
timestamps, and decoding. Compare frames only with frames plus accepted
transcript within the predeclared lexical-speech stratum. Reject the
route if it misses its predeclared completion and answer-quality
promotion gate.

<details>

<summary>

Evidence received, run details, grading, and provenance
</summary>

| Field | Fully disclosed synthetic value |
|----|----|
| Content status | Project-owned synthetic teaching scenario; not benchmark evidence |
| Authored | 2026-07-26 by the TRECVID VQA project |
| Source / split | None; no TRECVID row or source video |
| Media rights | No media used; timestamp storyboard and text equivalent only |
| Imagined duration | 22 seconds |
| Constructed question | What drink does the speaker say they packed for lunch? |
| Baseline evidence | Twelve constructed scene notes at 00:00 through 00:22, every two seconds |
| Route inputs | Constructed question wording and constructed ASR quality fields only |
| Constructed transcript | `[00:07.200–00:09.400] I packed green tea for lunch.` |
| Recorded model observation | Not recorded |
| Constructed frozen prediction | `green tea` |
| Constructed expected answer | `green tea` |
| Constructed grade | C; normalized exact match 1.0; token F1 1.0 |
| Label boundary | Expected answer disclosed only after the narrated freeze |
| Reproducibility boundary | No run identifier, model seed, artifact hash, or executable record exists because no run occurred |
| Supersession | None; the page teaches the current approach and does not define policy |

</details>

## What this example cannot establish

- It cannot estimate 2026 evaluation accuracy, route precision, ASR
  quality, or the prevalence of lexical-speech questions.
- It cannot show that Qwen3.5-4B, Whisper, 12 frames, or their
  combination would produce this answer on a real clip.
- It cannot support a model comparison or promotion decision; there is
  one constructed row and no actual inference.
- It cannot establish that transcripts always help. Real evidence
  includes cases where accepted text is incomplete, distracting, or
  unused.
- It is **one traced example, not the policy**. The current
  recommendation and its limits remain on [Start Here · Report
  page](index.md).
