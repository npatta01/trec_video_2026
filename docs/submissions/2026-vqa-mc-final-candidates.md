# TRECVID 2026 VQA Multiple-Choice Final Candidates

## Purpose and task context

This document is the complete handoff record for three candidate submissions
to the **TRECVID 2026 Video Question Answering Multiple-Choice task**.

It explains what each run does.

It identifies the exact files to upload.

It also explains unavailable videos, fallbacks, and the remaining submission
steps.

The Multiple-Choice task contains the same 2,443 questions and videos as the
earlier Answer Generation task.

It adds between 4 and 11 answer options to every question.

In Answer Generation, a system writes an answer in its own words.

In Multiple Choice, a submission ranks all released options. The first-ranked
option is the system's selected answer.

The three runs deliberately test increasingly expensive ways to use the prior
Answer Generation work:

1. **V1 converts an existing free-text answer into an option ranking.** It does
   not process the video again.
2. **V2 rereads the video only for 818 questions where the text conversion was
   judged uncertain.** It keeps V1 for the other questions.
3. **V3 rereads the video for every question whose video can be processed.**

**Direct video** or **native audio-video** means that Gemini receives:

- the complete video and its audio;
- the question; and
- every released option.

A **V1 fallback** means that direct video could not produce a valid ranking.

That row therefore retains V1's text-derived ranking.

## Current state

- **Task:** TRECVID 2026 Video Question Answering, Multiple Choice
- **State:** Three upload artifacts prepared and locally validated; **not yet
  uploaded to Evalbase**
- **Official allowance:** At most five runs per registered team
- **Selected runs:** Three
- **Submission rows:** 2,443/2,443 official Task 2 queries in every run
- **Organizer schema:** Each row contains only `Q_ID`, `Video_ID`, and a ranked
  `Answers` list. Each answer contains only integer `Rank` and an exact released
  option string in `Answer`.
- **Official format source:** [TRECVID 2026 VQA Run Submission
  Format](https://www-nlpir.nist.gov/projects/tv2026/vqa.html#Runs)

Local validation checks the published schema and released Task 2 file. It does
not establish organizer acceptance. Change a run to **submitted** only after
Evalbase returns a successful server-side validation result.

## Approaches at a glance

### V1 — `g37q36_mc_text`

- **Approach:** Map the prior rank-one free-text answer to all released options.
- **Video processed again:** No.
- **Direct-video rows:** 0.
- **Fallback rows:** Not applicable.

### V2b — `g37_mc_target`

- **Approach:** Keep V1 except on a frozen 818-row uncertainty queue.
- **Video processed again:** Only for those 818 rows.
- **Direct-video rows:** 808.
- **V1 fallback rows:** 10.

### V3b — `g37_mc_direct`

- **Approach:** Rank options directly from video for every processable row.
- **Video processed again:** Yes.
- **Direct-video rows:** 2,431.
- **V1 fallback rows:** 12.

These are complementary submissions, not successive versions of one upload.

V1 measures how far the existing Answer Generation answer can carry the MC
task.

V2 spends video inference only where the text evidence is weak.

V3 is the full direct-video comparison.

## Files to upload

| Evalbase run tag | Candidate | Rows | Size | SHA-256 |
|---|---|---:|---:|---|
| `g37q36_mc_text` | V1 text conversion | 2,443 | 1,685,335 bytes | `3dcd6cd75e97a13da0ccdf209bb6612a335ec43dc54f1c194a2145b9dfcb47ca` |
| `g37_mc_target` | V2b targeted direct-video repair | 2,443 | 1,013,095 bytes | `e7bcf6fa2c766b4633b1d2abb7058919dd8e736aba88ca161eb9056d5f17afaf` |
| `g37_mc_direct` | V3b full direct-video candidate | 2,443 | 1,013,095 bytes | `02d80093d19b051ce1c85ada5f5c571addffe09b7922056917900f6ee91f3faf` |

- [`g37q36_mc_text/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37q36_mc_text/submission.json)
- [`g37_mc_target/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37_mc_target/submission.json)
- [`g37_mc_direct/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37_mc_direct/submission.json)

The names V2b and V3b identify the repaired final artifacts.

Do not upload the earlier V2/V3 working files as additional runs.

Later V2c/V3c checks produced files that are byte-identical to V2b/V3b. They are
not distinct submissions.

## Detailed approach descriptions

### `g37q36_mc_text` — V1 text conversion

#### How it works

For each question, V1 starts with the rank-one free-text answer from the
previously submitted `g37_q36_fusion` Answer Generation run.

That source run fused answers from Gemini 3.7 Flash native audio-video
inference and Qwen3.6 inference.

A text-only Gemini 3.7 Flash call receives:

- the released question;
- that existing rank-one free-text answer; and
- every released multiple-choice option.

Gemini ranks only the released options from best to worst.

It cannot invent a new answer for the submission.

It does not receive the video in this MC conversion step.

#### Coverage and limitation

- **Rows ranked:** 2,443/2,443.
- **Video-related fallback gap:** None during the MC stage.
- **Main limitation:** V1 depends on the earlier free-text answer.

If that answer is wrong, incomplete, or poorly represented by the options,
text-only matching cannot recover the missing video evidence.

### `g37_mc_target` — V2b targeted video

#### How it works

V2 begins with the complete V1 submission.

Before any direct-video replacement, a separate text-only audit examines:

- the question;
- the prior free-text answer; and
- every released option.

The audit does not see V1's current ranking.

It places 818 rows in a frozen review queue because the prior answer is only
partially represented, unsupported by the options, ambiguous between options,
or otherwise requires video review.

No evaluation labels or reference answers were used to create this queue.

For those 818 rows only, Gemini 3.7 Flash receives the complete native
audio-video, the question, and all options.

The video condition uses:

- 4 FPS;
- low media resolution; and
- high reasoning effort.

A valid direct-video ranking replaces V1 on that row.

The other 1,625 rows remain exactly as V1 ranked them.

#### Coverage and limitation

- **Queued rows:** 818.
- **Valid direct-video rankings:** 808.
- **V1 fallbacks:** 10.
  - 8 unavailable videos.
  - 2 persistent invalid provider responses.
- **Main limitation:** The text-only audit must correctly identify which rows
  need another video pass.

V2 is cheaper than processing every video.

### `g37_mc_direct` — V3b full video

#### How it works

V3 does not use the text audit to decide when to inspect video.

For every question, Gemini 3.7 Flash is asked to rank all released options from:

- the complete native audio-video;
- the question; and
- every released option.

It uses the same 4 FPS, low-resolution, high-reasoning condition as V2.

#### Coverage and limitation

- **Questions:** 2,443.
- **Valid direct-video rankings:** 2,431.
- **V1 fallbacks:** 12.
  - 10 unavailable videos.
  - 2 persistent invalid provider responses.
- **Main limitation:** This is the most expensive of the three approaches.

V3 uses the most direct evidence and provides the cleanest comparison with V1.

## Retry and fallback handling

In the full-video V3 run, nine direct-video calls initially returned invalid
outputs.

- A bounded retry with a fresh upload recovered 6.
- The final 3 videos were normalized to H.264/AAC MP4 and tried once more.
- One additional call recovered, but its ranking matched the existing V1
  ranking exactly.
- Q_IDs 310 and 1563 still returned no candidates and counted no video tokens.

Further paid retries stopped for Q_IDs 310 and 1563.

Those rows retained V1.

V2 has only 8 unavailable-video fallbacks.

The other 2 unavailable videos were outside its 818-row queue.

V3 attempts every row, so all 10 unavailable videos appear in its fallback
count.

## Exact local validation

All three artifacts were validated against the frozen released Task 2 JSON at
Git commit `34f1d1e`. For every run, the validator confirmed:

- exactly 2,443 rows in official order;
- exact official `Q_ID` and `Video_ID` identity;
- only the published MC fields, with no AG-only `Time` field;
- consecutive integer ranks beginning at 1;
- every released option exactly once and no generated answer text;
- the real option-count distribution: 472 four-option, 1,952 eight-option,
  10 nine-option, 5 ten-option, and 4 eleven-option rows.

The private machine-readable catalog is
`data/vqa/submissions/trecvid-2026-vqa-mc-final-20260823/catalog.json`. The
private exception tracker is the sibling `exceptions.json`; it records Q_ID,
Video_ID, status, affected runs, and fallback source for all ten unavailable
videos and both persistent provider failures.

## Evalbase copy-paste descriptions

### `g37q36_mc_text`

> We reused the frozen rank-one answer from our previously submitted
> Gemini–Qwen Answer Generation fusion run and used off-the-shelf Gemini 3.7
> Flash as a text-only selector to rank every released multiple-choice option.
> No reference answers, evaluation labels, or additional video inference were
> used in this MC stage.

### `g37_mc_target`

> This hybrid run retains our text-conversion control except on a frozen
> 818-query review queue identified without labels. On those rows,
> off-the-shelf Gemini 3.7 Flash ranks the released options from complete
> native audio-video at 4 FPS and low media resolution. It uses 808 direct
> video rankings and ten declared control fallbacks.

### `g37_mc_direct`

> Off-the-shelf Gemini 3.7 Flash directly ranks every released option from the
> complete native audio-video, question, and choices at 4 FPS, low media
> resolution, and high thinking. The final run contains 2,431 direct rankings
> and twelve declared text-control fallbacks for ten unavailable videos and
> two persistent provider failures.

For all three runs select **off-the-shelf / no fine-tuning**. No model was
trained for this task, and no hidden labels or reference answers entered
inference, routing, repair, or ranking.

## Submission-day checklist

1. Upload exactly the `submission.json` linked for the matching run tag.
2. Confirm the local SHA-256 shown above immediately before upload.
3. Record the Evalbase run ID, upload time, and complete validator response.
4. Do not label the run submitted or accepted until server-side validation
   succeeds.
5. If Evalbase reports any schema difference from the published page, preserve
   the rejected bytes and validator response before making a new release
   candidate.
