# Qwen ASR plus Forced-Aligner Quality Diagnostic


Date: 2026-08-16

Primary theme: **Speech and Transcript Evidence**

Decision disposition: **Not ready for comparison**

## Plain-English result

I ran one fresh eight-video English speech test of Qwen3-ASR plus the
Qwen forced aligner. The container itself worked, but Qwen returned no
transcript for two videos. Six videos produced valid aligned output.
Because a fair comparison required all eight rows, the test is recorded
as `not_ready`.

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
