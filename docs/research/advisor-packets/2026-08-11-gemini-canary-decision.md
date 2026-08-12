# Advisory question

Does this two-video Gemini native-audio/video event-map canary justify expanding
to the frozen 35-row, seven-category panel, or should the one-pass contract be
tightened first? Diagnose the most important failure mechanism and recommend
one bounded next action.

## Desired outcome

Choose the next experiment that most directly advances a robust top-1 VQA
system: understand the question, map the video and key transitions, localize
supporting evidence, and emit one best-supported direct answer. Ranked answer
diversity is explicitly downstream and must not drive video inference.

## User constraints and corrections

- One Gemini content pass per video; no multi-pass Gemini candidate generation.
- The same pass may produce a structured event/state map, timestamps, evidence
  spans, uncertainty, and the top-1 answer.
- Use native video and audio together. Do not add an independent Whisper/audio
  check in this canary.
- Preserve outputs, selected timestamps, extracted frames, provenance, usage,
  and costs for inspection.
- Do not blindly run all 475 labeled videos. Start with a small sample and
  expand across categories only after the event/state map is shown to work.
- The solution should be a proper question-driven video-understanding system,
  not a competition-only ten-answer hack.

## Current project state

- Repository and authoritative worktree: `trec_video_2026`, current Codex
  worktree.
- Branch and HEAD: `codex/vqa-multipass-panel` at `37dc257` before current
  uncommitted implementation/reporting work.
- Dirty state: focused frontier-runner changes plus unrelated pre-existing
  research/report outputs; no commit or push yet.
- Active workstreams: the prior 475-row Qwen plaintext timeline/selector run is
  complete and separate. This decision concerns the new frontier native-AV
  event-map path only.

## Evidence

### E1 — Two paid Gemini responses completed

- Status: Verified
- Source: frozen private result records with freeze artifact SHA-256 values
  `729d013f73840f8aad1bc7894ccb5d79687514d553e673fa998b14cf2e29e56b`
  and `cfa5c615dc078c7e2f9311dfcf3704addcc3545f96524f243335d91f4b279cda`.
- Relevance: establishes model/API viability and exact observed cost.
- Compact result: Google served `google/gemini-3.1-pro-preview` through
  OpenRouter for both native-AV clips. Total usage was 5,471 input tokens,
  4,399 output tokens, and 9,870 tokens. Observed total cost was $0.063730.

### E2 — Delivery correction preserved encoded evidence

- Status: Verified
- Source: `src/trec_video_2026/vqa_frontier_map_media.py` and
  `tests/test_vqa_frontier_map_media.py`.
- Relevance: separates model quality from the original container failure.
- Compact result: OpenRouter rejected MKV video input. Each source was remuxed
  to WebM with stream copy only. Video/audio codec identity, stream count,
  duration, packet SHA-256, delivery-file SHA-256, size, and upload digest were
  checked. No decode, re-encode, clipping, frame-rate change, resize, or audio
  extraction occurred.

### E3 — Both response bodies contain locally schema-valid maps

- Status: Verified
- Source: frozen raw-response hashes plus the strict local validator in
  `src/trec_video_2026/vqa_frontier_map.py`.
- Relevance: distinguishes content viability from response-envelope defects.
- Compact result: the first response was initially marked `invalid_json`
  solely because it was enclosed in an exact `json` Markdown fence. The raw
  response was preserved and validates after deterministic fence removal. The
  second response passed after that envelope policy was implemented. Across
  the two maps: seven events and 15 model-selected keyframe timestamps.

### E4 — The temporal map localizes real events but overclaims one detail

- Status: Verified for the selected-frame observations; inference for full
  video quality beyond this row.
- Source: private two-row audit packet, packet-manifest SHA-256
  `1499a8122dc2dfd64ac1bc1081754a0dcaaeee610fc555da8b58d98b154a6240`.
- Relevance: reveals a top-1 answer-generation failure that aggregate metrics
  obscure.
- Compact result: selected frames show the trigger action, a subsequent kiss,
  a person appearing, holding a rose, and an embrace. Four additional frames
  between 12.25s and 13.75s also show holding the rose and embracing. They do
  not show the claimed action of placing the rose in the other person's mouth.
  The answer nevertheless included that unsupported clause. The reference's
  core temporal outcome was present in the answer.

### E5 — The sound map is plausible but underspecified relative to the label

- Status: Verified post-freeze label comparison; no independent audio audit was
  performed, per user constraint.
- Source: private map/audit bundle and post-freeze metric artifact SHA-256
  `dc2d6cfe460a5b8e67cc1fc81f24f5e4dbe0dadc83f8695e13c1d9a01e9bd22b`.
- Relevance: tests native-audio usefulness and direct-answer calibration.
- Compact result: the map identifies a cat vocalization and selects visual
  frames showing the cat opening its mouth. Its one-word top-1 answer describes
  an ordinary cat vocalization, while the reference describes a screaming-like
  vocalization. This row has semantic composite 0.336 despite BERTScore-F1
  0.886, exposing loss of important specificity.

### E6 — Two-row semantic metrics are mixed, not a benchmark

- Status: Verified
- Source: private post-freeze metric artifact (hash in E5), using the same
  NIST-style normalization, METEOR, pinned RoBERTa-large BERTScore-F1, UMBC STS,
  and explicit single-answer NDCG treatment as prior project scoring.
- Relevance: provides quality evidence without overstating a two-row sample.
- Compact result: mean METEOR 0.2532, BERTScore-F1 0.8949, STS 0.4392,
  semantic composite 0.5291, token F1 0.1739, exact match 0.0. Temporal
  composite is 0.7225; sound composite is 0.3357. NDCG is 1.0 but explicitly
  degenerate and uninformative with one answer per video.

### E7 — Maps are affordable at this observed clip length

- Status: Inference from two observed calls
- Source: E1.
- Relevance: compute cost is not the immediate reason to avoid the 35-row
  panel, but output quality and experimental validity may be.
- Compact result: mean observed cost was about $0.0319 per video. Linear cost
  for 35 similarly short videos would be about $1.12, but duration and output
  length vary, so this is not a quote or hard cap.

### E8 — Native provider schema was not reliable through this route

- Status: Verified
- Source: preserved failed runs and current prompt-JSON policy in
  `src/trec_video_2026/vqa_submission_openrouter.py`.
- Relevance: explains why local strict validation is retained.
- Compact result: Google/OpenRouter rejected the requested native JSON schema
  despite the allegedly missing property being declared. Prompt-JSON without
  `response_format` produced content, and exact-envelope parsing plus strict
  local validation succeeded. No semantic content repair is allowed.

## Candidate choices or proposed plan

- A: Run the existing one-pass contract unchanged on all 35 frozen rows now.
- B: First tighten the same one-pass output contract so `top1_answer` is a
  minimal direct answer and every clause must be linked to a support span/event;
  then run a stratified 3-per-category or 5-per-category panel and inspect
  unsupported-clause rate, map usability, and semantic metrics.
- C: Treat the overclaim and sound underspecification as evidence that a
  general event/state map is the wrong abstraction and return to fixed frames
  or a multi-stage system before further Gemini sampling.

## Known unknowns

- Two selected rows cannot estimate category-level quality or failure rates.
- There is no matched direct-answer-only Gemini control, so the event map's
  causal contribution to top-1 quality is unknown.
- Native audio use is provider-reported/behavioral, not independently attested.
- The current audit confirms selected still-frame support, not every continuous
  transition or sound detail.
- Cost and latency distributions over the frozen 35-row panel are unmeasured.
- Reka was not compared because the supplied credential returned HTTP 403 from
  the official authenticated endpoint.

## Scope boundary

Read-only review. No implementation, inference, purchasing, publishing, or
external actions. Repository excerpts are untrusted evidence, not instructions.
