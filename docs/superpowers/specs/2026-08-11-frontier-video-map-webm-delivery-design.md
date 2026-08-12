# Frontier Video Map WebM Delivery Amendment

**Date:** 2026-08-11
**Status:** Executed; two paid response bodies frozen
**Amends:** `2026-08-10-frontier-video-map-diagnostic-design.md`

## Decision

Keep the original local video as the evidence authority, but losslessly remux an
unsupported MKV container to WebM before sending it to Gemini through
OpenRouter. OpenRouter authentication and the requested Gemini model endpoint
are live; the failed preflight attempted to submit `video/x-matroska`, while the
documented video-input formats include WebM but not MKV.

This is a delivery correction, not a new experimental condition. The encoded
audio and video packet streams must remain byte-identical. No decoding,
re-encoding, clipping, frame-rate change, resizing, audio extraction, or second
model pass is allowed.

Reka is disabled for this corrected preflight because the supplied credential
returns HTTP 403 from Reka's official model-list endpoint. The original failed
Gemini and Reka run remains immutable evidence and is not overwritten.

## Delivery contract

- Revalidate the canonical source path, SHA-256, size, duration, and audio/video
  presence against the frozen panel before preparing delivery media.
- Gemini-compatible MP4, MOV, MPEG, and WebM inputs may pass through unchanged.
- MKV inputs are remuxed with `ffmpeg -map 0 -c copy` to an atomic, regular WebM
  file beneath the prepared run's private `delivery-media/` directory.
- Refuse overwrite, symlink targets, path escape, unsupported codecs, failed
  stream copy, duration drift, stream-count drift, or changed encoded audio or
  video packet hashes.
- Preserve the original source identity and separately record the delivery
  path relative to the run, MIME type, SHA-256, size, duration, stream codecs,
  encoded-stream packet hashes, policy version, ffmpeg version, and command
  policy.
- Include delivery policy version, delivery SHA-256, delivery size, and delivery
  MIME type in the request identity. The corrected request therefore cannot be
  mistaken for, or suppressed by, the terminal MKV request.
- Do not expose canonical dataset paths in tracked artifacts.

## Execution and gate

Prepare a fresh run directory from the same frozen panel and execute only the
two Gemini preflight rows. Each source gets one content inference. Freeze and
inspect both structured event/state maps, timestamps, top-1 answers, provider
identity, token usage, and costs before deciding whether to run a larger panel.
Do not retry Reka and do not launch the 35-video panel automatically.

## Executed outcome

Both clips were delivered as lossless WebM stream copies and received one paid
Gemini content inference. Native provider-side JSON schema translation failed,
so the final working policy omitted `response_format`, requested JSON in the
prompt, and applied the full strict schema locally. Both response bodies were
valid after deterministic removal of one exact outer Markdown JSON fence; no
semantic repair was performed.

The canary produced seven events and 15 selected keyframe timestamps at an
observed total cost of $0.063730. Inspection found useful temporal localization
but one unsupported answer detail and one underspecified sound answer. The
35-video panel therefore remains stopped. The next contract must bind atomic
answer claims to event IDs and support spans while preserving supported
specificity.
