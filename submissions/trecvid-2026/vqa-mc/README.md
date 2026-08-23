# TRECVID 2026 VQA Multiple-Choice Uploads

This directory contains the three selected organizer-upload candidates for all
2,443 released Multiple-Choice questions. Each file ranks every released option
for every question. The runs differ in how they obtain that ranking:

| Run tag and file | Standalone description |
|---|---|
| [`g37q36_mc_text/submission.json`](g37q36_mc_text/submission.json) | **V1 text conversion:** maps the existing rank-one Answer Generation response to the released options; no video is processed again. |
| [`g37_mc_target/submission.json`](g37_mc_target/submission.json) | **V2 targeted video:** keeps V1 except on a frozen 818-question uncertainty queue, where Gemini directly ranks options from native audio-video; 808 direct rankings and 10 V1 fallbacks. |
| [`g37_mc_direct/submission.json`](g37_mc_direct/submission.json) | **V3 full video:** Gemini directly ranks options from native audio-video wherever possible; 2,431 direct rankings and 12 V1 fallbacks. |

Use the directory name as the Evalbase run tag. A fallback means direct video
could not return a valid ranking, so that row retains V1's text-derived option
ranking. See the standalone [submission
record](../../../docs/submissions/2026-vqa-mc-final-candidates.md) for the full
methods, hashes, unavailable-video handling, validation evidence, and
copy-paste Evalbase descriptions.

These files are locally validated but have not yet received organizer-side
validation. Historical V2/V3 and byte-identical V2c/V3c files are intentionally
not packaged as separate runs.
