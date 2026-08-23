# TRECVID 2026 VQA Multiple-Choice Uploads

This directory contains the three selected organizer-upload candidates for all
2,443 released Multiple-Choice questions.

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

Use the directory name as the Evalbase run tag.

A fallback means direct video could not return a valid ranking.

That row retains V1's text-derived option ranking.

See the standalone [submission
record](../../../docs/submissions/2026-vqa-mc-final-candidates.md) for the full
methods, hashes, unavailable-video handling, validation evidence, and
copy-paste Evalbase descriptions.

These files are locally validated but have not yet received organizer-side
validation. Historical V2/V3 and byte-identical V2c/V3c files are intentionally
not packaged as separate runs.

The Evalbase Multiple-Choice submission window opens 11 September 2026 and
closes 18 September 2026. The portal correctly showed submissions as closed
when checked on 23 August.
