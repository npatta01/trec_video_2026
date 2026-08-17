# TRECVID Submission Ledger

This ledger preserves what was prepared or sent to an organizer. An explicitly
approved final upload file may be tracked under `submissions/` in this private
repository; questions, media, credentials, raw predictions, and logs remain
outside Git.

| Task | Run tag | Method | State | Record |
|---|---|---|---|---|
| TRECVID 2026 VQA Answer Generation | `q35_4b_u12` | Qwen3.5-4B, uniform 12-frame evidence | Upload artifact tracked and locally validated; not uploaded | [Run record](2026-q35-4b-u12.md) |
| TRECVID 2026 VQA Answer Generation | `g37_q36_fusion` | Gemini 3.7 Flash native AV + Qwen3.6 transcript/native fusion | Upload artifact tracked and locally validated; not uploaded | [Run record](2026-g37-q36-fusion.md) |
| TRECVID 2026 VQA Answer Generation | `q36_dualview` | Qwen3.6 transcript-conditioned + native-video dual view | Upload artifact tracked and locally validated; not uploaded | [Run record](2026-q36-dualview.md) |
| TRECVID 2026 VQA Answer Generation | `g37_grouped_unique` | Gemini-grouped grounded rewrites over Gemini/Qwen rank-one answers | Upload artifact tracked and locally validated; not uploaded | [Run record](2026-g37-grouped-unique.md) |

An upload is recorded as accepted only after Evalbase returns its own validator
result. Local validation alone does not change a run to submitted.
