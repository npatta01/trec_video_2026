# VQA MC revised Task 2 full refresh

## Objective

Regenerate the three selected TRECVID 2026 VQA Multiple-Choice candidates from
the organizer's revised Task 2 JSON. No prior MC ranking may be remapped into a
revised candidate.

## Frozen input and panel

- Revised Task 2 attachment SHA-256:
  `12042650e1bee6842543b39204fc483d6e46669a861b351435b6ececa3d1aabc`.
- Panel: all 2,443 revised rows in organizer order.
- Identity: the revised file retains every Q_ID, Video_ID, and question from
  the original release. It changes option presentation or membership on 2,435
  rows.
- Data boundary: private competition media and provider artifacts stay under
  the ignored shared `data/` tree. Only compact sanitized evidence may be
  tracked.

## Treatments and controls

1. **V1 text conversion:** fresh Gemini 3.7 Flash text-only ranking from the
   frozen rank-one `g37_q36_fusion` Answer Generation response and the revised
   released options.
2. **V2 targeted video:** fresh V1 except on a newly frozen blind-audit review
   queue derived from the revised options; use fresh direct native-AV rankings
   on queue rows when available.
3. **V3 full video:** fresh direct native-AV ranking on every processable row;
   fall back only to the fresh revised V1 when media is unavailable or the
   provider cannot return a valid complete ranking.

One fresh direct-video prediction freeze supplies identical V2 and V3
treatment rows. This avoids duplicate paid calls; it does not reuse or remap
any old prediction.

## Held fixed and integrity safeguards

- Models: Gemini 3.7 Flash for all three stages.
- Direct-video condition: complete native audio-video, 4 FPS, low media
  resolution, high thinking, seed 42, complete ranking schema.
- Direct requests receive only video, question, and revised options. They are
  blind to Task 1 answers, V1 rankings, audit assessments, and hidden labels.
- The blind audit receives the Task 1 answer and revised options but never the
  current V1 ranking.
- Every provider-mutating stage uses a new immutable experiment identity and
  refuses duplicate creation or overwrite.
- Final validation requires 2,443 rows, exact revised option coverage,
  consecutive ranks, and frozen hashes.

## Cost authorization and gates

- User authorized a full refresh under a cumulative `$25` ceiling.
- Prior observed spend: V1 `$0.66126844`; audit `$1.03602844`; direct-video
  batch plus canary `$12.40184251`.
- Current Batch pricing was reverified on 2026-08-26.
- Allocation: V1 hard cap `$5`; audit hard cap `$5`; direct-video hard cap
  `$15`; cumulative paid spend must remain `<= $25`.
- Stop before any later paid stage if observed prior-stage spend plus the next
  stage's projected ceiling would exceed the cumulative cap.

## Scientific question and decision rule

- **Question:** Can the same three complementary methods produce complete,
  schema-valid candidates against the revised organizer query release?
- **Falsifier:** a stage cannot complete under its integrity and cost gate, or
  a final candidate cannot validate exactly against the revised Task 2 file.
- **Decision rule:** retain all three refreshed artifacts as distinct
  organizer-upload candidates only if each passes revised-file validation.
  Do not claim relative quality without organizer scoring.

## Status

- [x] User approved the three-run refresh strategy.
- [x] Revised file diff and SHA-256 verified.
- [x] Shared dataset symlink restored.
- [x] V1 provider-free dry run passed.
- [x] Fresh V1 inference and validation: 2,443/2,443 rows, `$0.64891763`,
  submission SHA-256
  `14822f18a86aab72b766a0ca3a936194a0338ac3b6278ded536bda8a95c1f6fd`.
- [x] Fresh blind audit and V2 queue freeze: 2,443/2,443 rows,
  `$1.00464169`, 798 review rows, queue SHA-256
  `a82ae414e6fe28dc3a9542d4a0d2464ab4c2df2bafacb24a89dc9a5fb723ba93`.
- [x] Fresh direct-video canary: 2/2 rows, `$0.03732337`; both revised-file
  submission variants passed exact-option validation.
- [x] Full direct-video protocol prepared for 2,433 processable rows and 1,946
  unique videos; projected `$11.28992288` under its `$15` hard cap.
- [x] All 1,946 unique videos uploaded with no upload failures.
- [x] Sole 2,433-request full Batch succeeded; its private provider resource
  name is represented only by SHA-256
  `a671e48eecc7102bab0beec3c34b040cd9b6806b2ef27cd83caf1c29880a7e18`.
  All expected responses were frozen without a second full Batch or paid retry.
- [x] Fresh direct-video freeze: 2,423 valid rankings, ten media-unavailable
  rows, and ten provider-invalid rows. V2 uses 786 direct rankings on the new
  798-row queue; V3 uses all 2,423 valid direct rankings.
- [x] Three revised candidates packaged separately and validated exactly
  against the revised file. Accounted cumulative spend is `$13.99849538`;
  one provider-internal-error row returned no usage metadata.
- [x] Experiment evidence and submission handoff updated; first-release
  candidates marked superseded.
- [x] Superseded first-release Task 2 query JSON removed; its hash and aggregate
  diff remain in the audit record, and the bytes remain recoverable from Git.
- [x] Final repeat-freeze reproduced byte-identical hashes; QMD regeneration,
  Ruff, JSON/link/privacy checks, `git diff --check`, and the 48-test focused
  regression suite all passed.
- [x] All three candidates passed NIST's published MC validator at commit
  `acb28557f74e9e5a90cce79bfdc66df84af80a9b` under
  `--warnings-as-errors`: exit 0, zero errors, zero warnings, and `VALID`.

## Next action

No local refresh work remains. Do not upload to Evalbase without explicit user
authorization; when upload is authorized, use only the three files under
`submissions/trecvid-2026/vqa-mc-task2-v2/` and record organizer validation.

The broader repository suite was also sampled: 2,656 tests passed and one was
skipped, while 21 failures remained in unrelated dstack archive-size,
optional-NLTK, and YouTube download-runtime tests. None of the failing paths is
modified by this refresh; the focused VQA-MC suite is green.
