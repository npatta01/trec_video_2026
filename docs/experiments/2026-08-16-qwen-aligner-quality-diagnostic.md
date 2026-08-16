# Qwen ASR plus Forced-Aligner Quality Diagnostics


Date: 2026-08-16

Primary theme: **Speech and Transcript Evidence**

Historical Stage 0.5d disposition: **Not ready for comparison**

Stage 0.5e mechanism disposition: **Ready for a fresh aligned-English
quality panel**

## Plain-English result

The original Stage 0.5d run was one fresh eight-video English speech
test of Qwen3-ASR plus the Qwen forced aligner. The container itself
worked, but Qwen returned no transcript for two videos. Six videos
produced valid aligned output. Because a fair comparison required all
eight rows, that test remains recorded as `not_ready`.

This does **not** show that Qwen is better or worse than Whisper. The
paired quality score was intentionally not created when two rows failed.

## Frozen condition

- Eight fresh English exact-speech WAV rows from the preserved oracle
  panel.
- Prior Qwen rows were excluded before preparation.
- Requests contained only run/condition/video identity, an absolute WAV
  path, and `mode=transcription`.
- ASR revision: `7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Forced-aligner revision: `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- Digest-pinned offline ROCm container, read-only mounts, and one
  attempt.
- Paid cost: `$0.00`.

## Observed result

| Measure                  |              Result |
|--------------------------|--------------------:|
| Required rows            |                   8 |
| Completed rows           |                   6 |
| Valid aligned rows       |                   6 |
| Empty-output failures    |                   2 |
| Container exit code      |                   0 |
| Maximum allocated memory | 6,161,528,320 bytes |
| Container elapsed time   |      64.105 seconds |
| Frozen decision          |         `not_ready` |

The two failures were deterministic `Qwen3-ASR returned empty text`
runner errors. The raw output, terminal record, validation, and
prediction freeze were preserved without retry or relabeling.

## Interpretation boundary

This is a zero-cost incomplete-output diagnostic. It establishes that
the corrected harness can process multiple rows, but not that the model
has better transcript fidelity, better timestamps, downstream VQA value,
or replacement value relative to Whisper. A future attempt would need a
separately frozen panel and an explicit resolution for empty Qwen
output.

## Evidence

- Private run root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/`
- Sanitized aggregate:
  [`qwen-aligner-quality-20260816/`](artifacts/qwen-aligner-quality-20260816/)
- Harness:
  [`vqa_qwen_aligner_canary.py`](../../src/trec_video_2026/vqa_qwen_aligner_canary.py)
- Plan:
  [`2026-08-16-qwen-aligner-quality-test.md`](../superpowers/plans/2026-08-16-qwen-aligner-quality-test.md)

## Stage 0.5e empty-output mechanism diagnostic

Date: 2026-08-16

### Question and decision rule

The follow-up tested whether the two preserved empty outputs were
reproduced by the aligned automatic-language path and whether forcing
`language="English"` removed them under the same pinned Qwen models and
offline runtime. The predeclared decision rule was: if the aligned
automatic-language baseline reproduces both failures and the
aligned-English treatment completes all four rows with valid alignment,
advance only a fresh quality panel using that English-language
condition. The four diagnostic rows are never scored.

### Frozen method and panel

- Four rows selected only from the frozen Stage 0.5d terminal status:
  the two preserved empty-output failures and the first two preserved
  passing controls in frozen request order.
- Four process-isolated arms ran once, in order, with read-only mounts,
  `--network none`, pinned model revisions, and `$0.00` paid compute.
- Qwen3-ASR revision: `7278e1e70fe206f11671096ffdd38061171dd6e5`;
  forced-aligner revision: `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- The diagnostic froze before any labels, annotations, Whisper text, or
  scoring inputs were opened. Row-level Qwen outputs necessarily existed
  for validation, but remained private and were neither inspected nor
  scored.

### Observed result

| Arm | Condition | Successes | Failures | Valid aligned | Validation | Wall seconds |
|----|----|---:|---:|---:|----|---:|
| A `aligned_auto` | aligner, automatic language, timestamps on | 2/4 | 2 | 2/4 | `not_ready` | 46.722 |
| B `base_auto` | no aligner, automatic language, timestamps off | 2/4 | 2 | 0/4 | `not_ready` | 46.691 |
| C `aligner_plain_auto` | aligner, automatic language, timestamps off | 2/4 | 2 | 0/4 | `not_ready` | 46.294 |
| D `aligned_english` | aligner, English, timestamps on | 4/4 | 0 | 4/4 | `ready` | 47.787 |

All four arm terminals exited `0` and were contract-valid. Maximum peak
memory was 6,175,102,976 bytes for A, 4,331,441,664 for B, 6,175,102,464
for C, and 6,174,173,184 for D. The frozen status is
`diagnostic_frozen`; baseline reproducibility is `true`; the advancing
arm is `aligned_english`; and the decision is
`ready_for_fresh_aligned_quality_panel`.

### Interpretation boundary

On this four-row frozen mechanism panel, forcing `language="English"`
removed the two preserved empty-output failures. Each automatic-language
explanatory arm remained at `2` successes and `2` failures, and neither
restored all four rows. The bounded evidence therefore points to the
automatic-language path association rather than merely loading the
aligner or requesting timestamps; it is not an internal causal bug
claim. This is condition-bound mechanism evidence, not a
transcript-quality result, Whisper comparison, or downstream VQA result.
The four rows remain unscored, and the original Stage 0.5d summary
remains preserved in the evidence bundle.

### Authorized next action

Run a new label-blind aligned-English Qwen quality panel using no reused
Qwen rows. Hold the comparator, media, model revisions, prompts,
decoding, and all-row readiness gate fixed; load labels only after every
prediction freezes. This is the authorized next action, not a run
completed by this record.

### Evidence and immutable bindings

- Private diagnostic root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05e-qwen-empty-output-diagnostic-v1/`
- Sanitized aggregate:
  [`qwen-aligner-quality-20260816/`](artifacts/qwen-aligner-quality-20260816/)
- New aggregate record:
  [`diagnostic-summary.json`](artifacts/qwen-aligner-quality-20260816/diagnostic-summary.json)
- Input-freeze SHA-256:
  `aed463e5c6f206e346c29b41b56b7199f07874981ffadc908d41974cf0466083`
- Diagnostic-freeze payload SHA-256:
  `d157e6dc2187ea14d7ead4f42f8ee2d8c14dddc8707acd314d1e7c4a02fe57ec`

## Separate proposed Gemini 3.7 Flash native-AV smoke

No Gemini call was issued by this diagnostic. Google documents stable
`gemini-3.7-flash` as a native-multimodal model with text, image, video,
audio, and PDF inputs, text output, structured outputs, a
1,048,576-token input limit, and a 65,536-token output limit in the
[official model
page](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash).
The [official
changelog](https://ai.google.dev/gemini-api/docs/changelog) records GA
release on 2026-08-13. The [official pricing
page](https://ai.google.dev/gemini-api/docs/pricing) lists introductory
standard paid pricing through 2026-12-31 of `$0.75/M` input and
`$3.75/M` output, then `$1.50/M` and `$7.50/M`.

This is a separate proposed compatibility smoke, not an arm of the Qwen
mechanism test and not a superiority claim. If authorized, compare it
against the pinned Gemini 3.1 Pro native-AV condition with the same
media bytes and media resolution, API route and request surface,
timestamp and FPS settings, structured-output schema, post-freeze
evaluation, and declared panel. The [video-understanding
guide](https://ai.google.dev/gemini-api/docs/video-understanding)
documents timestamp prompts and default FPS behavior; the [audio
guide](https://ai.google.dev/gemini-api/docs/audio) documents audio
input and structured-output use. Any change in media delivery,
timestamps/FPS, schema, or evaluation would make the smoke a
compatibility test rather than a fair model comparison.
