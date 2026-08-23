# TRECVID 2026 VQA Multiple-Choice Final Candidates

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

## Upload files

| Evalbase run tag | Candidate | Rows | Size | SHA-256 |
|---|---|---:|---:|---|
| `g37q36_mc_text` | V1 text conversion | 2,443 | 1,685,335 bytes | `3dcd6cd75e97a13da0ccdf209bb6612a335ec43dc54f1c194a2145b9dfcb47ca` |
| `g37_mc_target` | V2b targeted direct-video repair | 2,443 | 1,013,095 bytes | `e7bcf6fa2c766b4633b1d2abb7058919dd8e736aba88ca161eb9056d5f17afaf` |
| `g37_mc_direct` | V3b full direct-video candidate | 2,443 | 1,013,095 bytes | `02d80093d19b051ce1c85ada5f5c571addffe09b7922056917900f6ee91f3faf` |

- [`g37q36_mc_text/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37q36_mc_text/submission.json)
- [`g37_mc_target/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37_mc_target/submission.json)
- [`g37_mc_direct/submission.json`](../../submissions/trecvid-2026/vqa-mc/g37_mc_direct/submission.json)

Do not upload the historical V2/V3 files as additional runs. V2c/V3c are
byte-identical to V2b/V3b and therefore are not distinct submissions.

## Run methods

### `g37q36_mc_text` — V1

The already-submitted `g37_q36_fusion` Answer Generation rank-one answer was
given to a discounted text-only Gemini 3.7 Flash selector together with the
released question and all released MC options. Gemini returned a complete
option ranking. No video was processed again during this MC selection stage.

This is the inexpensive control and has no media-unavailable gap at the MC
stage. Its upstream AG answer may nevertheless be wrong or only partially
represented by the options.

### `g37_mc_target` — repaired V2b

V1 is retained outside a frozen 818-row text-support review queue. Within that
queue, Gemini 3.7 Flash received complete native audio-video, the question,
and all released options at 4 FPS, low media resolution, and high thinking.
It supplies 808 direct rankings; ten queue rows use the V1 fallback—eight
unavailable videos and two persistent provider-invalid rows.

### `g37_mc_direct` — repaired V3b

Gemini 3.7 Flash directly ranked the released options from complete native
audio-video for 2,431/2,443 rows using the same 4 FPS, low-resolution,
high-thinking condition. Twelve rows use V1 fallback: ten unavailable videos
and two persistent provider-invalid rows.

The nine initial invalid outputs received a bounded fresh-upload retry; six
recovered. The final three were normalized to H.264/AAC MP4; one more direct
ranking recovered but exactly matched its existing V1 ranking. Q_IDs 310 and
1563 still returned no candidates and counted no video tokens, so further paid
retries stopped.

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
