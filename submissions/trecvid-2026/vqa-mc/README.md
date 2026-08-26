# Historical TRECVID 2026 VQA Multiple-Choice Candidates

> **Superseded:** these files target NIST's first Task 2 query release. Do not
> upload them. Use the [revised Task 2 replacement
> candidates](../vqa-mc-task2-v2/README.md).

This directory preserves three superseded candidates for the first release's
2,443 Multiple-Choice questions. They are audit artifacts, not upload
candidates.

Each file ranks every released option for every question.

The runs differ in how they obtain that ranking.

## V1 text conversion

- **Run tag:** `g37q36_mc_text`
- **File:** [`g37q36_mc_text/submission.json`](g37q36_mc_text/submission.json)
- **Approach:** Map the existing rank-one Answer Generation response to the
  released options.
- **Video processed again:** No.

## V2 targeted video

- **Run tag:** `g37_mc_target`
- **File:** [`g37_mc_target/submission.json`](g37_mc_target/submission.json)
- **Approach:** Keep V1 except on a frozen 818-question uncertainty queue.
- **Direct-video rankings:** 808.
- **V1 fallbacks:** 10.

## V3 full video

- **Run tag:** `g37_mc_direct`
- **File:** [`g37_mc_direct/submission.json`](g37_mc_direct/submission.json)
- **Approach:** Rank options from native audio-video wherever possible.
- **Direct-video rankings:** 2,431.
- **V1 fallbacks:** 12.

A fallback means direct video could not return a valid ranking.

That row retains V1's text-derived option ranking.

See the standalone [submission
record](../../../docs/submissions/2026-vqa-mc-final-candidates.md) for the full
methods, hashes, unavailable-video handling, validation evidence, and
historical Evalbase descriptions.

These files were locally validated only against the superseded first query
release and must not be submitted. Historical V2/V3 and byte-identical V2c/V3c
files are intentionally not packaged as separate runs.
