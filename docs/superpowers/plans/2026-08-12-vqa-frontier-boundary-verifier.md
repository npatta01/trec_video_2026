# Frontier VQA Boundary Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether a second question-blind Gemini native-AV pass can detect and repair the known-bad v5 scene-map coverage and temporal boundaries before any seven-video screen.

**Architecture:** Make OpenRouter request seeds depend on stable part identities instead of ephemeral upload IDs. Add one narrowly scoped verifier condition that receives the complete native AV plus an immutable draft neutral map and returns a change ledger and a fully corrected neutral map. Run it first only on the frozen v5 pilot, bind all inputs/outputs immutably, and dense-audit the result against the already-preserved native evidence.

**Tech Stack:** Python 3.13, pytest, existing OpenRouter/Gemini transport, existing strict neutral-map contract, ffmpeg/ffprobe evidence audit.

## Global Constraints

- Both annotation passes are question-blind; no question, reference answer, prior answer, or metric may enter a request.
- The verifier receives full native video/audio and the exact frozen draft map.
- Preserve `run-v4`, `run-v5`, and all audits unchanged.
- Make exactly one paid verifier call on the v5 pilot before deciding whether to test v4 or additional videos.
- Stop if the verifier does not restore the omitted extended foot-kiss state and correct the displaced 6–10 second boundary.
- Total spend remains capped at USD 25; this call should cost well below USD 1.
- Provider output is strict structured JSON; full local validation remains stricter than the Gemini-compatible projected schema.
- Use TDD and commit each independently reviewable task.

---

### Task 1: Stable content-addressed request seed

**Files:**
- Modify: `src/trec_video_2026/vqa_submission_openrouter.py:420-456`
- Modify: `tests/test_vqa_submission_openrouter.py`

**Interfaces:**
- Consumes: each rendered content part plus its stable identity from `_part_content`.
- Produces: `_stable_seed_payload(...) -> Mapping[str, object]`, excluding provider upload IDs while binding model, prompt text, stable media identity, schema, and policy.

- [ ] **Step 1: Write failing tests**

Add tests proving that two file parts with different provider `file_id` values but the same source/delivery identity produce the same outbound `seed`, while different media SHA-256 or prompt text changes the seed. Assert the actual request still contains the live upload ID.

- [ ] **Step 2: Verify red**

Run: `uv run pytest tests/test_vqa_submission_openrouter.py -q`

Expected: the equal-content/different-upload seed assertion fails.

- [ ] **Step 3: Implement the minimal seed projection**

Build the seed input from the request fields that affect inference, replacing rendered file-upload references with their stable `identity` records. Never include credentials or provider upload IDs. Record the integer seed in the outbound request as before.

- [ ] **Step 4: Verify green and regressions**

Run: `uv run pytest tests/test_vqa_submission_openrouter.py tests/test_vqa_frontier_phase0_runner.py -q`

Expected: all tests pass and upload deletion/request behavior is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/trec_video_2026/vqa_submission_openrouter.py tests/test_vqa_submission_openrouter.py
git commit -m "fix: stabilize frontier request seeds"
```

### Task 2: Question-blind boundary verifier contract and immutable runner

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_boundary_verifier.py`
- Create: `scripts/run_vqa_frontier_boundary_verifier.py`
- Create: `tests/test_vqa_frontier_boundary_verifier.py`
- Create: `tests/test_run_vqa_frontier_boundary_verifier.py`

**Interfaces:**
- Consumes: a frozen Phase 0 pilot result, its exact validated neutral map, canonical source media, source duration/hash, and the frozen `revise_contract` audit.
- Produces: `prepare_verifier(...)`, `run_verifier(...)`, `freeze_verifier(...)`, `status_verifier(...)` and immutable `manifest.json`, `request.json`, `attempts.jsonl`, `raw-response.json`, `result.json`, and `freeze.json`.
- Provider output keys: `schema_version`, `draft_map_sha256`, `verdict`, `event_reviews`, `corrected_map`, `global_notes`.
- Each event review contains exact keys `draft_event_ids`, `action`, `observed_start_seconds`, `observed_end_seconds`, `evidence_timestamps`, `rationale`, and `uncertainty`; action is one of `confirmed`, `adjusted`, `split`, `merged`, `removed`, or `added`.

- [ ] **Step 1: Write failing contract tests**

Test exact keys, integer schema version 1, exact draft hash binding, chronological corrected events, corrected-map duration, declared event-review references, nonempty evidence timestamps inside the source duration, and rejection of question/reference/answer fields anywhere in input or output.

- [ ] **Step 2: Write failing runner tests**

Test refusal unless the source pilot prediction and `revise_contract` audit are frozen; one request only; full native AV plus canonical draft in the prompt; stable request/source/draft hashes; hard USD 25 cap; provider/model validation; raw response preservation; no retry after a terminal schema result; and immutable freeze verification.

- [ ] **Step 3: Verify red**

Run: `uv run pytest tests/test_vqa_frontier_boundary_verifier.py tests/test_run_vqa_frontier_boundary_verifier.py -q`

Expected: collection fails because the verifier module/script do not exist.

- [ ] **Step 4: Implement the minimal contract**

The prompt must explicitly state that the draft is untrusted, the verifier must re-watch the complete AV, preserve supported events, add omitted states, correct onset/offset boundaries, and use uncertainty instead of schematic rounded timing. It must not see the question. Validate `corrected_map` with the existing `validate_neutral_map` and bind `draft_map_sha256` with `canonical_sha256`.

- [ ] **Step 5: Implement the minimal immutable runner and CLI**

Reuse existing delivery-media preparation, `OpenRouterTransport`, Gemini-compatible schema projection, provider-cost evidence, and atomic artifact helpers. The runner handles only this one pilot condition and refuses a nonempty output directory.

- [ ] **Step 6: Verify green and focused regressions**

Run: `uv run pytest tests/test_vqa_frontier_boundary_verifier.py tests/test_run_vqa_frontier_boundary_verifier.py tests/test_vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_runner.py tests/test_vqa_submission_openrouter.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/trec_video_2026/vqa_frontier_boundary_verifier.py scripts/run_vqa_frontier_boundary_verifier.py tests/test_vqa_frontier_boundary_verifier.py tests/test_run_vqa_frontier_boundary_verifier.py
git commit -m "feat: add frontier boundary verifier gate"
```

### Task 3: One paid verifier call, dense audit, and decision

**Files:**
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5/`
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5-audit/`
- Modify: `.superpowers/sdd/2026-08-12-vqa-frontier-phase0/progress.md`

**Interfaces:**
- Consumes: frozen `run-v5` pilot, frozen `pilot-audit-v5`, canonical source video, verifier code.
- Produces: exactly one paid raw verifier response, validated corrected map, immutable freeze, dense review artifacts, and `go_two_stage` or `stop_two_stage` decision.

- [ ] **Step 1: Run preflight and project cost without media inference**

```bash
uv run python scripts/run_vqa_frontier_boundary_verifier.py prepare \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run-v5 \
  --pilot-audit-freeze data/vqa/benchmarks/vqa-frontier-phase0-20260812/pilot-audit-v5/freeze.json \
  --video-id Dc4pETr9vGA \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5
uv run python scripts/run_vqa_frontier_boundary_verifier.py preflight \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5
```

Expected: exact source/draft/audit hashes and projected cost below USD 1; no provider media request.

- [ ] **Step 2: Run exactly one paid verifier call and freeze it**

```bash
uv run python scripts/run_vqa_frontier_boundary_verifier.py run \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5
uv run python scripts/run_vqa_frontier_boundary_verifier.py freeze \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/verifier-v5
```

Expected: one terminal result and one raw response with known cost; no retries or overwrites.

- [ ] **Step 3: Dense-audit against preserved native frames**

Use the existing 4-fps contact sheets plus corrected event keyframes. Confirm whether the corrected map restores the foot-kiss through approximately 10 seconds, moves the woman's transition to approximately 10 seconds, preserves the rose/embrace, and reports uncertainty for coarse boundaries.

- [ ] **Step 4: Apply the decision rule**

`go_two_stage` only if the verifier catches both known v5 boundary contradictions, restores the omitted salient state, and introduces no material false correction. Otherwise record `stop_two_stage` and do not call v4 or fresh videos.

- [ ] **Step 5: Update progress and verify artifacts**

Run: `git diff --check && uv run pytest tests/test_vqa_frontier_boundary_verifier.py tests/test_run_vqa_frontier_boundary_verifier.py -q`

Record cost, hashes, audit result, and the next authorized action in the canonical progress file. Do not add private media or raw provider responses to Git.

