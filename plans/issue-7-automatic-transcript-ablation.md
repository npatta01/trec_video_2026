# Issue #7 automatic-transcript ablation

> **Historical status:** The referenced PR is merged and this diagnostic is
> complete. Whisper is routed through a quality gate rather than used globally;
> the strict human-oracle tier remains deferred in open issue #19. See
> [Project Status](../docs/project-status/index.md).

## Objective

Measure the paired change from frozen frames-only evidence to the same frames
plus an automatic transcript on the ten-row lexical-speech panel. Keep this
diagnostic distinct from both a human-caption reference and a genuine
gold-transcript benchmark.

## Scope and invariants

- Freeze the ten row IDs, source-media hashes, questions, frames, model revision,
  runtime image, prompt, decoding, and seed before inference.
- Generate label-blind Whisper large-v3 transcripts with explicit language
  detection and quality gates.
- Run exactly two matched TREC conditions: `frames_only` and
  `frames_automatic_transcript`.
- Freeze all predictions before loading correct answers or distractors.
- Treat uploader subtitles as a `human-caption reference` only when full-clip
  coverage is verified. Never call them oracle or gold.
- Treat multi-ASR agreement as consensus evidence only, never gold or oracle.
- Run a separate, small public benchmark with genuine reference transcripts for
  a frames/automatic/gold sanity comparison when primary-source licensing and
  access permit it.
- Keep media, transcripts, requests, predictions, labels, and raw logs beneath
  ignored `data/`; commit only code, tests, and aggregate reporting.

## Decisions

- TREC model: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- VLM runtime:
  `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
- ASR model: `openai/whisper-large-v3` at revision
  `06f233fe06e710322aca913c1bc4249a0d71fce1`.
- Visual evidence: 12 ordered frames per clip, with the same paths, hashes, and
  timestamps in both conditions.
- One French clip uses the accepted English translation selected by the
  automatic-transcript contract; nine clips use English output.

## Current status

- [x] Frozen a label-free ten-row lexical panel.
- [x] Added and tested automatic-transcript and human-caption contracts.
- [x] Extracted 120 frames and ten 16 kHz mono WAV files.
- [x] Generated and validated ten automatic transcripts.
- [x] Built and pair-checked 20 matched inference requests.
- [x] Pass the three-row GPU inference probe.
- [x] Run all 20 requests and freeze predictions.
- [x] Load labels only after freeze, score, and adjudicate paired transitions.
- [x] Complete and run the public dataset-reference sanity benchmark.
- [x] Verify exactly one full-coverage uploader-caption reference row.
- [x] Update issue #7 and draft PR #9 with hashes and aggregate evidence.

## Verification evidence

- Focused contract and paired-analysis tests: 38 passed.
- Automatic transcripts: 10/10 passed selected-English quality and language
  detection gates.
- Matched requests: 20 unique pairs; only condition/transcript content differs
  within each video pair.
- Request artifact SHA-256:
  `dfa03029293cefda74c104f81154fae266792dddb9ec6c48d9201d4648f4447b`.
- Automatic-transcript artifact SHA-256:
  `b017368822fd93528446356945db21cd7ca18b400a396a03b13fc355fed4ef01`.
- TREC manual accuracy: 0.400 frames-only and 0.700 with automatic
  transcripts; three rescues, zero degradations.
- Residual attribution: two VLM answer-selection/evidence-use failures and one
  ASR non-lexical-vocalization miss.
- Human-caption control: one row at 99.68% verified coverage; all three
  conditions remain wrong, corroborating a VLM evidence-use failure.
- External Lombard GRID control: 0/4 frames-only, 4/4 automatic transcript, and
  4/4 dataset recording-prompt reference.
- Complete prediction freezes were written before either TREC or external
  answer scoring.

## Next action

Review draft PR #9. The strict human-oracle tier remains intentionally blocked
until independent transcriber/reviewer records exist; automatic ASR, the one
verified uploader-caption reference, and the external dataset reference are
complete.
