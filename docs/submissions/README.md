# TRECVID Submission Ledger

This ledger preserves what was prepared or sent to an organizer. An explicitly
approved final upload file may be tracked under `submissions/` in this private
repository; questions, media, credentials, raw predictions, and logs remain
outside Git.

| Task | Run tag | Method | State | Record |
|---|---|---|---|---|
| TRECVID 2026 VQA Answer Generation | `q35_4b_u12` | Qwen3.5-4B, uniform 12-frame evidence | Submitted 2026-08-11; organizer validation succeeded | [Run record](2026-q35-4b-u12.md) |
| TRECVID 2026 VQA Answer Generation | `g37_q36_fusion` | Gemini 3.7 Flash native AV + Qwen3.6 transcript/native fusion | Submitted 2026-08-17; organizer validation succeeded | [Run record](2026-g37-q36-fusion.md) |
| TRECVID 2026 VQA Answer Generation | `q36_dualview` | Qwen3.6 transcript-conditioned + native-video dual view | Submitted 2026-08-17; organizer validation succeeded | [Run record](2026-q36-dualview.md) |
| TRECVID 2026 VQA Answer Generation | `g37_grouped_unique` | Gemini-grouped grounded rewrites over Gemini/Qwen rank-one answers | Compact RC2 submitted 2026-08-17; organizer validation succeeded after the truncated RC1 upload was replaced | [Run record](2026-g37-grouped-unique.md) |
| TRECVID 2026 VQA Multiple Choice | `g37q36_mc_text` | Frozen AG fusion rank-one answer mapped to released options by text-only Gemini | Prepared and locally validated; not uploaded | [Three-run MC record](2026-vqa-mc-final-candidates.md) |
| TRECVID 2026 VQA Multiple Choice | `g37_mc_target` | Targeted Gemini direct-video ranking on the frozen weak queue with text fallback | Prepared and locally validated; not uploaded | [Three-run MC record](2026-vqa-mc-final-candidates.md) |
| TRECVID 2026 VQA Multiple Choice | `g37_mc_direct` | Full Gemini direct-video option ranking with declared text fallback | Prepared and locally validated; not uploaded | [Three-run MC record](2026-vqa-mc-final-candidates.md) |

An upload is recorded as accepted only after Evalbase returns its own validator
result. Local validation alone does not change a run to submitted.

## 2026 Answer Generation closure

All four Answer Generation runs are submitted and organizer-validated. PR
[#43](https://github.com/npatta01/trec_video_2026/pull/43) is the integration
record for the deadline Qwen3.6, Gemini 3.7 Flash, fusion, rescue, grouped
rewrite, validation, and submission work. It was integrated through merge
commit `215060ae06eeb0f00f611807b65aace35de92f82`.

The experiment records intentionally retain historical executed-source commit
hashes from PR #43. Those hashes identify the code that actually produced
frozen evidence and must not be rewritten to the eventual squash-merge commit.
The PR remains the reviewable history that connects those source identities to
the integrated repository state; tracked submission hashes and the individual
run records remain the upload identities.

Nonessential refactoring, retirement of deadline-only runners/configurations,
and unrelated repository-wide test-environment repairs are follow-up work and
are not part of the submission closure gate.

See [post-deadline maintenance](2026-vqa-post-deadline-maintenance.md) for the
immutable upload boundary, retained reproducibility tooling, test-health
context, and the checklist for any later archival cleanup.
