# TRECVID 2026 VQA-MC revised Task 2 candidates

These are the three locally validated replacement candidates generated from
NIST's revised `vqa.2026.testingDataset.Task2_v2.json` release. They have not
been uploaded to Evalbase.

| Run tag | Method | Direct rows | Fresh V1 fallbacks | SHA-256 |
|---|---|---:|---:|---|
| `g37q36_mc_text` | Fresh text selector over revised options | 0 | Not applicable | `14822f18a86aab72b766a0ca3a936194a0338ac3b6278ded536bda8a95c1f6fd` |
| `g37_mc_target` | Fresh direct video on the new 798-row audit queue | 786 | 1,657 total: 1,645 non-queue rows plus 12 queue fallbacks | `da513f71d843d2af288cfd78b49d2c64491c6c3e038f7a859f0530856cf8ebbe` |
| `g37_mc_direct` | Fresh direct video wherever processable | 2,423 | 20 | `0048a2ae9212f21974335f2d50173a1605e2a45b7e69dd029e701e01304f3198` |

Upload paths:

- [`g37q36_mc_text/submission.json`](g37q36_mc_text/submission.json)
- [`g37_mc_target/submission.json`](g37_mc_target/submission.json)
- [`g37_mc_direct/submission.json`](g37_mc_direct/submission.json)

Every file has 2,443 rows, preserves revised organizer order and identities,
uses consecutive ranks, and contains every exact revised option once. The
revised option distribution is 472 four-option rows and 1,971 eight-option
rows.

All three also pass NIST's published `validate_vqa_mc_2026.py` at commit
`acb28557f74e9e5a90cce79bfdc66df84af80a9b` with
`--warnings-as-errors`: zero errors, zero warnings, and `STATUS: VALID`.

V2's 12 queue fallbacks comprise eight unavailable-media rows and four invalid
provider outputs. V3's 20 fallbacks comprise ten unavailable-media rows and ten
invalid provider outputs. All fallbacks use the newly generated revised V1;
no first-release MC ranking was remapped or reused.

See the [revised submission handoff](../../../docs/submissions/2026-vqa-mc-task2-v2-final-candidates.md)
and [sanitized experiment evidence](../../../docs/experiments/artifacts/vqa-mc-task2-v2-refresh-20260826/README.md).
