# Gemini Failure Rescue Design

## Objective

Run one additive, bounded Gemini 3.7 Flash rescue over exactly the 27 rows from
the completed full-panel freeze whose terminal reason is `max_tokens` (11),
`empty_text` (9), or `provider_upload_failed` (7). Preserve the original run
and the five validated submissions unchanged until a successful-only overlay
has independently passed validation.

## Frozen inputs

- Original Gemini predictions freeze SHA-256:
  `e3f0e4cf6d9dd65986f328f3834e96b753b180b3b6123a646418ec1424078cb0`.
- Selected query IDs:
  `116, 139, 175, 310, 647, 886, 932, 963, 995, 1013, 1022, 1055,
  1063, 1084, 1090, 1127, 1140, 1168, 1173, 1190, 1462, 1563, 1771,
  2279, 2385, 2404, 2421`.
- The selector must re-derive this set from the frozen reasons and fail if the
  freeze hash, counts, IDs, questions, videos, media hashes, or source request
  binding changes.
- No annotations, references, labels, metrics, or prior answer text may enter
  selection or inference.

## Provider protocol

- Model and prompt remain `gemini-3.7-flash` and the completed run's direct
  answer prompt.
- Native AV remains 4 FPS, low media resolution, and high thinking.
- Increase `max_output_tokens` from 2,048 to 8,192 for all 27 rescue rows.
- Use one upload worker and at most one exact-operation retry for HTTP 503.
- Use a new private output root; never overwrite or reinterpret the original
  raw responses, attempts, freezes, or submission packet.
- Hard caps: estimated and observed cost each `$2.00`. The frozen conservative
  estimate is about `$1.19`.

## Terminal evidence and overlay

Every attempted upload and inference must have an immutable terminal artifact.
The rescue freezes raw and parsed outputs before assembly. Only rows parsed as
`succeeded` with nonempty text may replace a fallback. Any repeated truncation,
empty response, refusal, upload failure, request failure, or cap block leaves
the existing validated answer unchanged.

Create new revisioned Gemini-only and Gemini/Qwen-fusion files. Do not overwrite
the current packet files. After both candidates pass the official NIST
validator, promote them atomically into the packet, preserve the superseded
files in an additive archive, refresh provenance/manifest/checksums, and keep
the same five-run priority order.

## Verification

Before paid calls, tests must prove exact selector identity, the 8,192-token
request, `$2` caps, one-worker behavior, immutable source bindings, successful-
only overlay, and fallback preservation. Dry-run must report 27 rows and 26
unique videos. After the run, verify all internal hashes, zero unmatched
attempts, rescue counts/cost, manual samples, all five official NIST validators,
packet checksums, credential scan, Ruff, and focused tests.
