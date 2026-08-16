# Local Gemma Structured-Frame Counting Canary

**Goal:** Before the 2026-08-17 submission date, determine whether cached
Gemma 3n E4B can generate complete frame-presence states for the narrow visible
appearance reducer, or stop counting work without changing the validated
fallback submission.

**Decision boundary:** This is a mechanistic development canary, not a generic
counting solution. The frozen census contains exactly one compatible confirmed
failure (`RZrkA52TGBA`) and one plausible compatible correct control
(`PgKOzRgYMAI`). Even a complete pass cannot authorize replacing the primary
submission because generalization and control preservation are underpowered.

## Immutable fallback

The existing fallback remains untouched at
`data/vqa/submissions/qwen35-safety-20260810-a1/final/`. Its `SHA256SUMS`
verifies and its preserved validator report says `valid: true`, with no errors
or warnings.

## Condition

- Model: cached pinned `google/gemma-3n-E4B-it` snapshot
  `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10`.
- Runtime: cached pinned ROCm container
  `localhost/trec-vqa-omni-rocm@sha256:b1f97c47769a8a2f32a5ac9a18cc62d2aab57ed32c66720775407f6213ebf96f`.
- Network: disabled; no provider, model download, remote GPU, or paid call.
- Sampling: decoded indices `0,4,8,...` plus the final decoded frame when not
  already selected; actual decoded PTS govern coverage.
- Primary: numbered contact sheets, at most 11 frames per model call.
- Challenge: the same frames as full-resolution individual images, at most 11
  images per model call.
- Generation: greedy, thinking disabled, one predeclared prompt, no repair of
  semantic states. Malformed or incomplete output abstains.
- Repeats: two complete clean executions with identical frozen inputs.
- Labels/references/panel roles: unavailable inside preparation, prompts,
  inference, parsing, reconciliation, and reduction. They are joined only
  after both prediction freezes.

## Panels

- Failure: original question “How many times balls appear in the video?”
- Control: original question “how many times we see a colorful screen with
  lines in the video?”
- Router-only sentinels: at least one action repetition, physical entity, part
  cardinality, and edited-shot question; all must be unsupported without model
  inference.

## Gates

1. Preparation hashes source media, decoder configuration, every frame, every
   presentation, both prompts, implementation sources, and the frozen request.
2. Each pass returns exactly one allowed state for every requested frame.
3. Primary/challenge disagreement, uncertainty, omission, malformed JSON,
   excessive gap, or endpoint failure forces abstention.
4. The two complete repeats must produce the same reduced outcome and value on
   each row. State differences are allowed only if they reconcile to the same
   conservative ledger without uncertainty.
5. Post-freeze mechanism pass requires failure answer `4`, preservation of the
   control reference, and all router sentinels unsupported.

Any failed gate or runtime over two hours stops the canary. No prompt tuning,
router expansion, third pass, model swap, or human state repair follows.

## Decision

- **Negative or incomplete:** preserve it and stop counting work for this
  submission.
- **Positive:** preserve it as narrow mechanistic evidence only. Do not modify
  the primary fallback. A shadow candidate may demonstrate the allowlisted
  routing seam, but lack of a second compatible failure and additional controls
  prevents primary promotion before the deadline.

## Verification before inference

- RED/GREEN tests for the two exact census phrasings.
- Router audit returns exactly the one failure and one control above.
- Synthetic zero-network runner test proves strict parsing, batching,
  write-once freezing, label isolation, and deterministic reduction.
- Focused counting tests, Ruff, format check, and `git diff --check` pass.
