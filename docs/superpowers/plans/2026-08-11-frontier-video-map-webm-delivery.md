# Frontier Video Map WebM Delivery Implementation Plan

**Goal:** Correct the Gemini preflight's unsupported MKV delivery without
changing its audio/video evidence, then obtain two reviewable event/state maps.

**Architecture:** Add one focused delivery-media module and integrate its
immutable record into the frontier runner request identity. Preserve the old
failed run and use a fresh prepared run for corrected calls.

## Task 1: Lossless delivery-media boundary

**Files:**

- Create `src/trec_video_2026/vqa_frontier_map_media.py`
- Create `tests/test_vqa_frontier_map_media.py`
- Modify `src/trec_video_2026/vqa_frontier_map_runner.py`
- Modify `tests/test_vqa_frontier_map_runner.py`

- [x] Write failing tests for MKV-to-WebM stream-copy provenance, compatible
  pass-through, request-identity separation, tamper rejection, and atomic
  overwrite/symlink refusal.
- [x] Implement a frozen delivery record and preparation function. Use ffmpeg
  stream copy only and verify packet hashes, duration, stream counts, codecs,
  and file identity.
- [x] Integrate delivery identity into prepared manifests, request hashes, and
  uploads while retaining the canonical source identity.
- [x] Run focused pytest and Ruff checks.

## Task 2: Fresh two-video Gemini preflight

- [x] Verify the shared `data` symlink and frozen-panel hash.
- [x] Prepare fresh run directories; never mutate a terminal attempt.
- [x] Verify both delivery records before dispatch.
- [x] Inject the OpenRouter credential through hidden process input and run
  exactly the two Gemini preflight requests.
- [x] Freeze and inspect raw responses, validated maps, timestamps, provider
  identity, usage, cost, and artifact hashes.
- [x] Report a no-go for the unchanged 35-video contract. Do not launch Reka or
  the 35-video panel.

## Outcome

- Gemini/Google route: live and authenticated through OpenRouter.
- Paid content calls: 2/2 response bodies preserved; 2/2 strict-schema-valid
  after exact JSON-fence parsing.
- Structure: seven events and 15 model-selected timestamps.
- Usage: 5,471 input tokens, 4,399 output tokens, 9,870 total.
- Observed cost: $0.063730.
- Private audit: valid two-row packet with two overview contact sheets and all
  15 timestamped event frames; reviewer verdict files intentionally remain
  empty/unfrozen.
- Quality: temporal localization was useful, but one answer clause was not
  supported by inspected frames; the sound answer lost reference-critical
  specificity.
- Decision: keep the one-pass event-map architecture as a diagnostic candidate,
  tighten clause-level answer grounding, and test unused categories before any
  broader panel.
