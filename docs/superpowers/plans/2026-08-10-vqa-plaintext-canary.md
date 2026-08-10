# VQA Plain-Text Timeline Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing compound JSON event-map canary with a small AV-only plain-text timeline plus text-only selector canary that can be judged automatically before scaling to all 475 annotated videos.

**Architecture:** Nemotron receives the complete normalized video and question once, with native audio enabled, and emits a bounded line protocol containing 2–6 timestamped event anchors and concise scene descriptions. A second text-only request selects one frozen anchor and produces the answer; Python deterministically derives full timeline cells and <=6 seconds of evidence windows. No grammar decoding, LLM repair, model thinking, or label/reference data enters the generation requests.

**Tech Stack:** Python 3.12, vLLM OpenAI-compatible chat completions, PyAV/ffmpeg media normalization, pytest, dstack dev environment with one L40S.

## Global Constraints

- The decision panel is exactly 475 unique videos with annotations; the 1,883-row test set remains untouched.
- The canary uses exactly the four frozen normalized videos and AV input; no prompt grid or retry is allowed.
- Map decoding is `enable_thinking=false`, temperature 0, top-p 1, top-k 1, max 256 tokens, stop `\nEND`; event descriptions are at most 16 words.
- Selector decoding is `enable_thinking=false`, temperature 0, top-p 1, top-k 1, max 64 tokens, stop `\nEND`.
- No `response_format`, JSON schema, grammar, or LLM repair is allowed.
- A full-run GO requires 4/4 map and 4/4 selector primary parses, valid deterministic coverage, disjoint shifted windows, complete hashes, nonempty selector answers, and non-catastrophic semantic canary scores.
- The full run must normalize each AV1 source at most once into a shared content-addressed cache; direct-compatible media is reused.
- Raw answers and private evidence remain in private artifacts; public-facing metrics contain no raw model reasoning.

---

### Task 1: Add executable plain-text protocol helpers

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/nemotron3_omni_event_map.py`
- Test: `tests/test_nemotron3_omni_event_map.py`

**Interfaces:**
- Produces `build_plaintext_map_request(...)`, `build_plaintext_selector_request(...)`, `parse_plaintext_map(...)`, `parse_plaintext_selector(...)`, `derive_anchor_cells(...)`, and `expand_selected_anchors(...)`.
- Existing JSON protocol helpers remain available for superseded provenance but are not used by the new canary.

- [ ] **Step 1: Write failing tests for request pins and strict parsing.**

  Cover exact map/selector decoding parameters, valid event/timeline lines, optional direct answer lines, malformed IDs, non-increasing timestamps, extra output, missing `END`, unknown selector IDs, and answer length bounds.

- [ ] **Step 2: Run the focused tests and confirm they fail.**

  Run: `uv run pytest -q tests/test_nemotron3_omni_event_map.py -k plaintext`

  Expected: FAIL because the new helpers do not exist.

- [ ] **Step 3: Implement the smallest parser and deterministic evidence expansion.**

  Parse only the line protocol, require 2–6 sequential anchors, require an opening-third and ending-half spread, and derive cells from adjacent-anchor midpoints. Expand one selected anchor into a centered window, shift inward at clip edges, and reject overlap or budget violations.

- [ ] **Step 4: Run the focused tests and the full core module suite.**

  Run: `uv run pytest -q tests/test_nemotron3_omni_event_map.py`

  Expected: PASS.

- [ ] **Step 5: Commit the protocol helper change.**

  Run: `git add src/trec_video_2026/vqa_experiments/nemotron3_omni_event_map.py tests/test_nemotron3_omni_event_map.py && git commit -m "feat: add plain-text VQA timeline protocol"`

### Task 2: Add the bounded AV canary driver and artifact contract

**Files:**
- Create: `scripts/run_vqa_event_map_plaintext_canary.py`
- Test: `tests/test_run_vqa_event_map_plaintext_canary.py`
- Modify: `docs/superpowers/plans/2026-08-09-vqa-question-conditioned-event-map.md`

**Interfaces:**
- Driver consumes the frozen four-row canary request file, normalized controller, and vLLM URL.
- Driver produces one JSONL artifact with request/response hashes, map/selector parse status, anchors, deterministic cells/windows, elapsed times, and no raw reasoning.
- Artifact must distinguish technical validity from semantic score eligibility.

- [ ] **Step 1: Write failing tests for exactly four AV map calls and four selector calls.**

  Use injected fake POST responses; assert no retries, no JSON schema, map output is frozen before selector input, and malformed rows fail closed.

- [ ] **Step 2: Run the focused driver tests and confirm failure.**

  Run: `uv run pytest -q tests/test_run_vqa_event_map_plaintext_canary.py`

- [ ] **Step 3: Implement the driver and artifact validation.**

  Use the module helpers, post one AV video request per canary row, then post one text-only selector request per successfully parsed map. Record private response hashes and bounded diagnostics; never load references into requests.

- [ ] **Step 4: Run focused tests and both core suites.**

  Run: `uv run pytest -q tests/test_run_vqa_event_map_plaintext_canary.py tests/test_nemotron3_omni_event_map.py tests/test_vqa_event_map_dstack_config.py`

- [ ] **Step 5: Commit the canary driver.**

  Run: `git add scripts/run_vqa_event_map_plaintext_canary.py tests/test_run_vqa_event_map_plaintext_canary.py docs/superpowers/plans/2026-08-09-vqa-question-conditioned-event-map.md && git commit -m "feat: add AV plaintext timeline canary"`

### Task 3: Execute and adjudicate the warm-GPU canary

**Files:**
- Create: private dstack artifact `/dstack/dev/plaintext-canary-a1.jsonl`
- Modify: private dstack diagnostics only; do not modify the dataset
- Record: `docs/project-status/experiment-ledger.qmd` and generated report only after adjudication

**Interfaces:**
- Uses the persistent dstack dev run `vqa-event-map-dev-a1`, warm Nemotron server on localhost:8000, and four normalized canary videos.
- Produces a technical gate and frozen semantic-score input; it does not start the 475-video run by itself.

- [ ] **Step 1: Sync only the committed module/driver and run the four AV map calls.**

  Verify server health, media hashes, and no-thinking request pins before inference.

- [ ] **Step 2: Run four text-only selectors from the frozen maps.**

  Verify selector input hashes bind exactly to the stored map outputs.

- [ ] **Step 3: Apply the technical GO gate.**

  Require 4/4 maps, 4/4 selectors, `finish_reason=stop`, valid anchors, exact coverage, valid windows, disjoint shifted controls, complete artifact identities, and 4/4 nonempty selector answers.

- [ ] **Step 4: Score frozen canary answers with METEOR, BERTScore, STS, and NDCG.**

  Compare against the prior baseline and check that there is no catastrophic semantic regression. Keep this score separate from the technical gate.

- [ ] **Step 5: Stop before scale if either gate fails; otherwise record GO and launch authorization.**

### Task 4: Scale with a 32-row fail-fast checkpoint

**Files:**
- Modify: `scripts/run-vqa-event-map-dstack.sh`
- Modify: `scripts/build_vqa_event_map_evidence.py`
- Modify: `tests/test_vqa_event_map_dstack_config.py`
- Modify: `tests/test_build_vqa_event_map_evidence.py`
- Modify: `docs/superpowers/plans/2026-08-09-vqa-question-conditioned-event-map.md`

**Interfaces:**
- Full runner consumes the 475-row annotated controller, AV-only normalized media cache, and pinned plain-text protocol.
- It runs a first 32-row checkpoint and stops before continuing when fewer than 31 rows complete both map and selector stages.
- Shifted evidence is eligible only when `shift_disjoint=true` and transcript segment/byte budgets match; otherwise the row is excluded from the causal shifted denominator.

- [ ] **Step 1: Add failing tests for one-time AV1 normalization, checkpoint threshold, and shift eligibility.**
- [ ] **Step 2: Implement content-addressed media reuse and fail-fast checkpoint.**
- [ ] **Step 3: Run shell/config/evidence tests and a representative local dry run.**
- [ ] **Step 4: Launch eight shards only after the recorded canary GO.**
- [ ] **Step 5: Freeze all outputs and retain selected frame timestamps, PTS, hashes, contact sheets, Whisper provenance, and failure diagnostics.**

### Task 5: Score and record the experiment decision

**Files:**
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify: `docs/project-status/experiment-ledger.md`
- Modify: `docs/superpowers/plans/2026-08-09-vqa-question-conditioned-event-map.md`
- Create: `docs/experiments/artifacts/vqa-plaintext-event-map-20260810.json`

- [ ] **Step 1: Compute exact diagnostic, token F1, METEOR, BERTScore F1, STS, composite C, and NDCG where applicable.**
- [ ] **Step 2: Compare F0, H/F, G/S, H/S, and eligible Shifted H/S with paired bootstrap intervals.**
- [ ] **Step 3: Record the primary result, failure taxonomy, and model decision.**
- [ ] **Step 4: Run final verification and preserve the canonical artifact paths.**
