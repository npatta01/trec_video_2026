# TRECVID 2026 VQA-MC revised Task 2 candidates

## Current state

- **State:** all three replacement candidates were submitted to Evalbase on
  12 September 2026; organizer validation succeeded for every run.
- **Organizer input:**
  `vqa.2026.testingDataset.Task2_v2.json`, SHA-256
  `12042650e1bee6842543b39204fc483d6e46669a861b351435b6ececa3d1aabc`.
- **Coverage:** 2,443/2,443 rows in every run.
- **Validation:** exact revised `Q_ID`, `Video_ID`, row order, option strings,
  complete consecutive ranks, and option multiplicity.
- **Organizer validator:** all three returned exit code 0, `VALID`, zero errors,
  and zero warnings under `--warnings-as-errors`.
- **First-release artifacts:** the old query JSON is removed; the old candidate
  bundle remains only for audit and is explicitly superseded.
- **Additional candidate:** `g38_mc_direct` was submitted as one upload file on
  13 September 2026; Evalbase organizer validation succeeded.

NIST's revision retains every question and video identity but changes 2,435 of
2,443 option arrays: 1,166 are exact option-set shuffles, 1,250 differ only
after punctuation/format normalization, and 19 change option membership. The
replacement candidates therefore use fresh inference throughout; no prior MC
ranking was remapped.

## Submitted files

| Evalbase run tag | Candidate | Rows | Size | SHA-256 |
|---|---|---:|---:|---|
| `g37q36_mc_text` | V1 fresh text selector | 2,443 | 1,676,499 bytes | `14822f18a86aab72b766a0ca3a936194a0338ac3b6278ded536bda8a95c1f6fd` |
| `g37_mc_target` | V2 fresh targeted direct video | 2,443 | 1,005,347 bytes | `da513f71d843d2af288cfd78b49d2c64491c6c3e038f7a859f0530856cf8ebbe` |
| `g37_mc_direct` | V3 fresh full direct video | 2,443 | 1,005,347 bytes | `0048a2ae9212f21974335f2d50173a1605e2a45b7e69dd029e701e01304f3198` |

- [`g37q36_mc_text/submission.json`](../../submissions/trecvid-2026/vqa-mc-task2-v2/g37q36_mc_text/submission.json)
- [`g37_mc_target/submission.json`](../../submissions/trecvid-2026/vqa-mc-task2-v2/g37_mc_target/submission.json)
- [`g37_mc_direct/submission.json`](../../submissions/trecvid-2026/vqa-mc-task2-v2/g37_mc_direct/submission.json)

Use the directory name as the Evalbase run tag. Do not upload the files under
the superseded `submissions/trecvid-2026/vqa-mc/` directory.

## Submitted fourth candidate

| Evalbase run tag | Candidate | Rows | Size | SHA-256 | State |
|---|---|---:|---:|---|---|
| `g38_mc_direct` | Gemini 3.8 full direct video with one invalid-row retry | 2,443 | 1,005,347 bytes | `13327801b9569a18cacfb16307ece774c251842e2f5a331268e4e679ed02068d` | Submitted 2026-09-13; organizer validation succeeded |

- [`g38_mc_direct/submission.json`](../../submissions/trecvid-2026/vqa-mc-task2-v2/g38_mc_direct/submission.json)

This is one submission, not separate base and rescue runs. The
initial full pass and 13-row retry remain private provenance only. The final
file contains 2,428 direct Gemini 3.8 rankings and 15 revised-V1 fallbacks:
ten unavailable-media rows and five persistent invalid responses. The retry
recovered eight valid rankings, six of which changed the assembled submission
because two matched the existing V1 ranking exactly.

The Gemini 3.8 canary cost `$0.03853463`, the full Batch cost `$14.67042263`,
and the retry cost `$0.11541488`, for `$14.82437214` total. The retry's actual
cost exceeded its `$0.10` projected launch gate by `$0.01541488` due to higher
observed reasoning usage; the total remained below the authorized `$16`
envelope.

The packaged file passes exact revised-option validation. NIST's pinned
validator at commit `acb28557f74e9e5a90cce79bfdc66df84af80a9b` also reports
2,443 submitted items, 2,443 unique Q_IDs, zero errors, zero warnings, and
`STATUS: VALID` under `--warnings-as-errors`. These are local validator results;
the separate Evalbase acceptance is recorded below.

## Evalbase result

Evalbase lists all three runs under the TREC 2026 Video Question Answering
Multiple Choice Task for NP Labs, submitted by `npatta01` on 12 September
2026. The organizer page reports **validation succeeded** for
`g37q36_mc_text`, `g37_mc_target`, and `g37_mc_direct`.

The fourth run, [`g38_mc_direct`](https://ir.nist.gov/evalbase/run/trec-2026/trec2026-vqa-mcq/g38_mc_direct),
was submitted from local Chrome by `npatta01` for NP Labs on 13 September
2026. Its run page reports **Validated? validation succeeded**. The recorded
fine-tuning choice is **Off the shelf**. Before upload, the file matched the
2,443 rows, 1,005,347 bytes, and SHA-256 above, and the conference run list
contained only the three Gemini 3.7 runs. Only one Gemini 3.8 run was uploaded.

This establishes organizer acceptance, not accuracy: no organizer scores are
available, and completion or disagreement counts do not show that Gemini 3.8
is more accurate than Gemini 3.7.

## Approaches

### V1 — `g37q36_mc_text`

A fresh Gemini 3.7 Flash text-only Batch pass receives the question, the frozen
rank-one answer from the previously submitted `g37_q36_fusion` Answer
Generation run, and every revised option. It ranks only those revised options.
It does not reread video and does not reuse a prior MC ranking.

- Complete fresh rankings: 2,443/2,443.
- Full-run cost: `$0.64891763`; canary: `$0.00076819`.

### V2 — `g37_mc_target`

A new blind text audit receives the question, prior Answer Generation answer,
and revised options, but not V1's ranking. It freezes a new 798-row review
queue. V2 retains fresh V1 outside that queue and uses the shared fresh direct
video result on queue rows when valid.

- New queue: 798 rows.
- Fresh direct rankings: 786.
- Queue fallbacks: 12: eight unavailable-media rows and four invalid provider
  outputs.
- Non-queue fresh V1 rows: 1,645.
- Audit full-run cost: `$1.00464169`; canary: `$0.00139275`.

### V3 — `g37_mc_direct`

One fresh Gemini 3.7 Flash Batch pass receives complete native audio-video at
4 FPS and low media resolution, the question, and every revised option. High
thinking is enabled. The direct request is blind to the Answer Generation
answer, V1, the audit queue, prior MC rankings, and hidden labels.

- Fresh direct rankings: 2,423.
- Fresh V1 fallbacks: 20: ten unavailable-media rows and ten invalid provider
  outputs.
- The same fresh direct prediction freeze supplies the identical V2 treatment
  rows; this avoids duplicate paid calls without remapping old predictions.

## Cost and provider exception

The full direct Batch returned usage for 2,432/2,433 requests: 24,942,588 prompt
tokens and 1,574,390 billed output tokens including reasoning, accounting for
`$12.30545175`. One provider-internal-error response had no usage metadata and
became a declared fresh V1 fallback. The direct canary cost `$0.03732337`.

Accounted cumulative refresh spend, including all three canaries, is
**`$13.99849538`**. The missing-usage response means this is not a complete
provider-observed total, but both the accounted value and the predeclared
`$5 + $5 + $15` stage ceilings remain within the user's `$25` authorization.
No paid retry or second full Batch was launched.

## Validation and interpretation

All three files independently pass the exact revised-release validator. They
also pass NIST's published
[`validate_vqa_mc_2026.py`](https://github.com/gawad-nist/VQA/blob/acb28557f74e9e5a90cce79bfdc66df84af80a9b/validate_vqa_mc_2026.py)
at commit `acb28557f74e9e5a90cce79bfdc66df84af80a9b` with
`--warnings-as-errors`: each reports 2,443 submitted items, 2,443 unique
Q_IDs, zero errors, zero warnings, and `STATUS: VALID`. This is a local run of
the organizer's script, not an Evalbase acceptance result. The revised
distribution is 472 four-option rows and 1,971 eight-option rows. Repeated
direct freeze is immutable, and every normal provider response ID is unique;
the single provider error is bound by its unique request metadata.

Compared with V1, V2 changes 229 Top-1 choices and 654 complete rankings; V3
changes 363 Top-1 choices and 1,976 complete rankings. These are disagreement
counts, not corrections: hidden labels remain unavailable, so the refresh does
not establish relative Top-1 accuracy or MRR.

The compact evidence record is
[`vqa-mc-task2-v2-refresh-20260826/`](../experiments/artifacts/vqa-mc-task2-v2-refresh-20260826/README.md).
