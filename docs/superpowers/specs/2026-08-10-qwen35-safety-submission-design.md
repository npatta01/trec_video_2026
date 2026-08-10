# Qwen3.5 Safety Submission Design

**Date:** 2026-08-10
**Status:** Approved for execution by the user's instruction to freeze and prepare
**Primary theme:** Submission operations and fallback safety

## Objective

Prepare a private, validator-ready TRECVID 2026 Answer Generation safety run
without uploading it to Evalbase. Preserve the verified Qwen3.5-4B,
uniform-12 development baseline as the rank-1 path while producing the exact
ten-answer ranked contract required for every one of the 2,443 test queries.

## Frozen system

- Model: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime: the existing pinned CUDA image and Transformers revision used by
  the verified 475-row baseline.
- Rank 1: the existing `current` answer policy over twelve uniform
  bin-midpoint frames. This path must not be changed by ranked-tail work.
- Ranks 2-10: one additional label-blind model request over the same twelve
  frames asks for nine concise, independently plausible alternatives ordered
  by likelihood. The rank-1 string is supplied only as an exclusion so it is
  not repeated.
- Deduplication: answers are normalized exactly as the submission serializer
  normalizes them. A bounded repair request may fill a short or duplicate
  tail. If it still cannot produce ten unique answers, deterministic,
  question-type-aware fallback strings complete the schema and are recorded
  as fallbacks rather than model predictions.
- Timing: each answer records finite, non-negative generation time. Rank 1
  uses its measured request time; tail candidates share the measured tail
  request time because they were generated together.

## Input and evidence boundary

The authoritative input is
`data/vqa/2026/testing/vqa.2026.testingDataset.Task1.json`. A frozen manifest
binds every query ID, video ID, question, resolved local media path, source
hash, and availability status without labels or references. Shared frame
evidence is extracted once per unique available video and reused for all
questions about that video.

The ten locally unavailable videos remain in the run. Their affected queries
use a question-only Qwen request with an explicit no-video instruction; if
that request fails, the deterministic fallback fills all ten ranks. They may
not disappear from the submission.

## Artifact and integrity contract

The private run root lives under
`data/vqa/submissions/qwen35-safety-20260810-a1/` and is ignored by Git. It
contains the immutable manifest, evidence records, raw model outputs, ranked
predictions, freeze record, top-1/top-5/top-10 JSON files, validator checkout
identity and reports, SHA-256 manifest, runtime identity, timing, and terminal
status.

Predictions are frozen before serialization. Serialization must verify the
freeze, preserve all 2,443 query IDs and source order, require exactly ten
unique non-empty answers with contiguous ranks, and refuse overwrite. The
official NIST validator is pinned to commit
`e06cb7854d7defe53e86412c2bba0ca5f636ea40`; its current README says it is the
available 2026 AG format validator even though scoring remains based on the
2025 implementation.

No test question, answer, media, raw output, or private path is committed.
No submission is uploaded to Evalbase without a separate explicit request.

## Execution gates

1. Local unit and contract tests pass.
2. A label-free one-query fixture proves manifest, ranked-tail parsing,
   freezing, serialization, and validator invocation.
3. A small live GPU canary completes before the full task is submitted.
4. The full run freezes all 2,443 queries and records explicit handling for
   every unavailable-media row.
5. The official validator exits zero on the top-10 file; top-1 and top-5 are
   retained as analysis views and are not expected to satisfy the required
   ten-answer upload contract.

