# VQA Frontier Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the bounded Gemini Phase 0 that tests question-blind full-video annotation, direct native-AV answering, and map-guided native-AV answering on one spent pilot video and seven fresh labeled-development videos.

**Architecture:** Add a focused Phase 0 path beside the old question-aware frontier-map runner. Reuse the existing OpenRouter transport, lossless MKV-to-WebM delivery, immutable freezing, source provenance, and semantic metric helpers, while introducing separate neutral-map and matched-answer contracts. A private, label-isolated runner produces one neutral map plus two answer conditions per fresh video; a two-view audit freezes before labels; an offline scorer then joins references and writes a sanitized evidence projection.

**Tech Stack:** Python 3.12+, standard-library JSON/dataclasses/pathlib/urllib, existing `OpenRouterTransport`, ffmpeg/ffprobe, pytest, NLTK METEOR, BERTScore, UMBC STS, Quarto.

**Approved design:** `docs/superpowers/specs/2026-08-12-vqa-annotator-boundary-selector-design.md`. This plan implements only its Phase 0 frontier-model screen; it does not begin the later local production architecture.

## Global Constraints

- The annotation call receives complete native video/audio but no VQA question, reference, label, answer hypothesis, question-relevance field, or answer field.
- The model is pinned to `google/gemini-3.1-pro-preview` through the already verified OpenRouter path.
- The pilot and each fresh row use exactly one question-blind annotation pass; do not add a refinement pass.
- The screen contains exactly seven fresh rows: one each from causal, counting, OCR/detail, other/global, sound, speech, and temporal.
- Direct and map-guided answering both receive the same question and complete native video/audio under the same schema, instructions, reasoning level, temperature, and token ceiling.
- The direct condition emits an empty `map_event_ids` array; the guided condition must cite valid event IDs from its frozen neutral map.
- Do not add question-type routing, Whisper, OCR, tracking, dense frames, crops, super-resolution, specialist models, candidate diversity, or a local-model production pipeline.
- The current baseline is the already-frozen Qwen3.5-4B uniform-12 frames-only condition; do not run Qwen again.
- References remain unopened until provider outputs, native-evidence audits, and their freezes are complete.
- Preserve every request identity, raw provider response, parsed result, attempt, source/delivery hash, timestamp, token count, observed cost, and terminal state.
- Permit one identical retry only for an explicit transport, rate-limit, empty-response, or truncated-response failure. Invalid JSON, schema failure, bad content, or an unwanted answer is terminal for that condition.
- Cumulative new paid inference must stop before USD 25. Project remaining cost after the pilot and after the first complete fresh row.
- The one-row pilot is a contract/transport gate only. The seven-row result is a mechanism screen, not a population benchmark or permanent model election.
- Keep private questions, references, predictions, raw responses, credentials, and licensed media under the ignored canonical data root. Never stage dataset contents.
- QMD is the editable reporting source. Generate sibling Markdown and HTML; never hand-edit generated report surfaces.

---

## File Structure

### New focused modules

- `src/trec_video_2026/vqa_frontier_phase0_contract.py` — neutral-map and matched-answer schemas, prompts, immutable records, and strict local validators.
- `src/trec_video_2026/vqa_frontier_phase0_panel.py` — deterministic spent-pilot and seven-category selection plus label-free manifest validation.
- `src/trec_video_2026/vqa_frontier_response_store.py` — exact JSON-envelope parsing and conflict-safe raw-response preservation shared by old and new runners.
- `src/trec_video_2026/vqa_frontier_phase0_runner.py` — prepared-run state, condition dispatch, retry/cost gates, result freezing, and compact status.
- `src/trec_video_2026/vqa_frontier_phase0_audit.py` — private audit packet construction, verdict validation, pilot authorization, and screen audit freezing.

### New command-line entry points

- `scripts/build_vqa_frontier_phase0_panel.py` — create and validate the private frozen Phase 0 panel.
- `scripts/run_vqa_frontier_phase0.py` — prepare, preflight, run pilot/screen, project cost, authorize, freeze, and show safe status.
- `scripts/build_vqa_frontier_phase0_audit.py` — build, validate, and freeze pilot/screen audit packets.
- `scripts/score_vqa_frontier_phase0.py` — post-freeze reference join, three-condition metrics, map/citation diagnostics, and sanitized projection.

### Existing files modified

- `src/trec_video_2026/vqa_frontier_map_runner.py` — delegate exact-envelope parsing and raw-response preservation to the shared store without changing old-run bindings.
- `docs/experiments/2026-08-12-vqa-frontier-phase0.qmd` — dated condition-bound result, created only after execution.
- `docs/project-status/research-routing.qmd` — primary-theme current claim and next decision.
- `docs/project-status/research-visual-evidence.qmd` — cross-link to the routing-owned claim.
- `docs/project-status/experiment-ledger.qmd` — queue/status update only if the result changes the next action.
- `AGENTS.md` — durable shared-checkout pointers to the private and sanitized Phase 0 artifacts.

### New tests

- `tests/test_vqa_frontier_phase0_contract.py`
- `tests/test_vqa_frontier_phase0_panel.py`
- `tests/test_vqa_frontier_response_store.py`
- `tests/test_vqa_frontier_phase0_runner.py`
- `tests/test_vqa_frontier_phase0_audit.py`
- `tests/test_build_vqa_frontier_phase0_panel.py`
- `tests/test_run_vqa_frontier_phase0_script.py`
- `tests/test_build_vqa_frontier_phase0_audit.py`
- `tests/test_score_vqa_frontier_phase0.py`

---

### Task 1: Freeze neutral-map and matched-answer contracts

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_phase0_contract.py`
- Create: `tests/test_vqa_frontier_phase0_contract.py`

**Interfaces:**
- Produces: `NeutralVideoMap`, `GroundedAnswer`, `NEUTRAL_MAP_SCHEMA`, `PROVIDER_NEUTRAL_MAP_SCHEMA`, `GROUNDED_ANSWER_SCHEMA`, `PROVIDER_GROUNDED_ANSWER_SCHEMA`.
- Produces: `build_neutral_map_prompt(duration_seconds: float) -> str`.
- Produces: `build_grounded_answer_prompt(question: str, duration_seconds: float, neutral_map: Mapping[str, object] | None) -> str`.
- Produces: `validate_neutral_map(payload: object, duration_seconds: float) -> NeutralVideoMap`.
- Produces: `validate_grounded_answer(payload: object, duration_seconds: float, *, allowed_map_event_ids: Collection[str] | None) -> GroundedAnswer`.
- Depends on: `trec_video_2026.vqa_submission.canonical_sha256` only.

- [ ] **Step 1: Write contract tests that enforce question blindness and stable identifiers**

```python
def test_neutral_prompt_has_no_question_channel() -> None:
    prompt = build_neutral_map_prompt(18.25)
    assert "18.25" in prompt
    assert list(inspect.signature(build_neutral_map_prompt).parameters) == [
        "duration_seconds"
    ]
    assert "sentinel private question" not in prompt.lower()
    assert "Question:" not in prompt


def test_neutral_map_rejects_question_and_answer_fields() -> None:
    payload = neutral_payload()
    payload["question"] = "sentinel private question"
    with pytest.raises(ValueError, match="forbidden field"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_requires_unique_event_and_entity_ids() -> None:
    payload = neutral_payload()
    payload["events"][1]["event_id"] = payload["events"][0]["event_id"]
    with pytest.raises(ValueError, match="event IDs must be unique"):
        validate_neutral_map(payload, 20.0)
```

- [ ] **Step 2: Write matched-answer tests before implementation**

```python
def test_direct_and_guided_use_the_same_schema() -> None:
    assert set(GROUNDED_ANSWER_SCHEMA["properties"]) == {
        "schema_version", "top1_answer", "claims", "could_be_wrong_because"
    }


def test_direct_answer_requires_empty_map_event_ids() -> None:
    payload = answer_payload(map_event_ids=["E1"])
    with pytest.raises(ValueError, match="direct answer map_event_ids must be empty"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_guided_answer_rejects_unknown_map_event_id() -> None:
    payload = answer_payload(map_event_ids=["E9"])
    with pytest.raises(ValueError, match="unknown map event"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids={"E1"})
```

- [ ] **Step 3: Run the focused tests and verify the missing module fails**

Run: `uv run pytest tests/test_vqa_frontier_phase0_contract.py -q`

Expected: FAIL during import because `vqa_frontier_phase0_contract` does not exist.

- [ ] **Step 4: Implement the exact neutral-map contract**

Use these root fields and no others:

```python
NEUTRAL_MAP_PROMPT_VERSION = "frontier-neutral-map-v1"
NEUTRAL_MAP_ROOT_FIELDS = frozenset({
    "schema_version",
    "perceived_duration_seconds",
    "global_description",
    "entities",
    "events",
    "global_uncertainties",
})
NEUTRAL_ENTITY_FIELDS = frozenset({
    "entity_id", "description", "aliases", "confidence",
})
NEUTRAL_EVENT_FIELDS = frozenset({
    "event_id", "start_seconds", "end_seconds", "keyframe_seconds",
    "description", "entity_ids", "actions", "relations",
    "state_before", "state_change", "state_after",
    "visual_observations", "speech_observations", "sound_observations",
    "visible_text", "confidence", "uncertainties",
})
FORBIDDEN_NEUTRAL_KEY_TOKENS = frozenset({
    "answer", "annotation", "correct", "gold", "label", "oracle",
    "question", "reference", "relevance", "score",
})
```

The validator must reject unknown fields recursively, non-finite numbers,
out-of-duration timestamps, duplicate IDs, event references to missing entity
IDs, unordered events, and `perceived_duration_seconds` outside 0.5 seconds of
the supplied source duration. Empty observation arrays are valid. The prompt
must embed the complete local JSON schema and explicitly request chronological,
question-blind observation. It may explain that the map must not answer or use
a question, but its only runtime input is `duration_seconds`; it must have no
question argument, question slot, or question content.

- [ ] **Step 5: Implement the exact shared answer contract**

```python
ANSWER_PROMPT_VERSION = "frontier-grounded-answer-v1"
ANSWER_ROOT_FIELDS = frozenset({
    "schema_version", "top1_answer", "claims", "could_be_wrong_because",
})
ANSWER_CLAIM_FIELDS = frozenset({
    "claim_id", "text", "supporting_spans", "supporting_modalities",
    "map_event_ids",
})
ANSWER_SPAN_FIELDS = frozenset({"start_seconds", "end_seconds"})
ANSWER_MODALITIES = frozenset({
    "video", "audio", "speech", "visible_text", "neutral_map",
})
```

`build_grounded_answer_prompt` must render one shared instruction body. When
`neutral_map is None`, render `Neutral map: not supplied` and require empty
`map_event_ids`. Otherwise embed canonical JSON for the frozen map and require
every material claim to cite one or more declared event IDs. Both forms require
native timestamp spans and prohibit unsupported details from entering
`top1_answer`.

- [ ] **Step 6: Run contract tests and the old map-contract tests**

Run: `uv run pytest tests/test_vqa_frontier_phase0_contract.py tests/test_vqa_frontier_map.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the contracts**

```bash
git add src/trec_video_2026/vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_contract.py
git commit -m "feat: add frontier phase zero contracts"
```

---

### Task 2: Freeze the spent pilot and seven fresh rows without labels

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_phase0_panel.py`
- Create: `scripts/build_vqa_frontier_phase0_panel.py`
- Create: `tests/test_vqa_frontier_phase0_panel.py`
- Create: `tests/test_build_vqa_frontier_phase0_panel.py`

**Interfaces:**
- Consumes: the existing 37-row `frontier-video-map-20260811/frozen-panel.json`, the successful spent-canary result/freeze, and the 475-row frozen Qwen frames-only prediction file.
- Produces: `build_phase0_panel(source_panel: Mapping[str, object], successful_pilot_video_id: str, qwen_rows: Sequence[Mapping[str, object]]) -> dict[str, object]`.
- Produces: `validate_phase0_panel(payload: object, *, source_panel: Mapping[str, object], qwen_rows: Sequence[Mapping[str, object]]) -> Phase0Panel`.
- Produces private manifest phases `pilot` and `screen`; pilot has only `neutral_map`, screen rows have all three condition IDs.

- [ ] **Step 1: Write selection tests with a synthetic five-per-category source panel**

```python
def test_build_panel_selects_one_median_duration_row_per_category() -> None:
    source = source_panel_fixture(durations=(5.0, 10.0, 15.0, 20.0, 25.0))
    panel = build_phase0_panel(source, "spent-pilot", qwen_fixture(source))
    screen = [row for row in panel["examples"] if row["phase"] == "screen"]
    assert len(screen) == 7
    assert {row["category"] for row in screen} == set(CATEGORIES)
    assert {row["duration_seconds"] for row in screen} == {15.0}


def test_build_panel_rejects_missing_qwen_baseline() -> None:
    source = source_panel_fixture()
    qwen = [row for row in qwen_fixture(source) if row["video_id"] != "causal-3"]
    with pytest.raises(ValueError, match="matched Qwen baseline"):
        build_phase0_panel(source, "spent-pilot", qwen)
```

- [ ] **Step 2: Run the tests and verify they fail on the missing module**

Run: `uv run pytest tests/test_vqa_frontier_phase0_panel.py tests/test_build_vqa_frontier_phase0_panel.py -q`

Expected: FAIL during import.

- [ ] **Step 3: Implement deterministic selection and immutable bindings**

Use this policy exactly:

```python
PHASE0_RUN_ID = "vqa-frontier-phase0-20260812"
SELECTION_POLICY = "spent-successful-pilot-plus-category-median-duration-v1"
CONDITIONS = ("neutral_map", "direct_answer", "map_guided_answer")

def _screen_choice(rows: Sequence[PanelRow]) -> PanelRow:
    ordered = sorted(rows, key=lambda row: (row.duration_seconds, row.ordinal))
    return ordered[len(ordered) // 2]
```

The pilot must be one of the source panel's two `preflight` rows and must match
the verified successful canary result. Each screen row must come from the 35
unused `evaluation` rows, use the median-duration row of its category, and have
a passed Qwen frames-only prediction. Bind the source panel file SHA-256, Qwen
prediction SHA-256, selected ordered-ID digest, source-media digest, and the
selection policy. Recursively reject label/reference keys.

- [ ] **Step 4: Implement the safe CLI**

The CLI accepts paths, never a question/reference argument:

```text
--source-panel
--successful-pilot-results
--successful-pilot-freeze
--qwen-predictions
--output
```

It loads the pilot row's sibling `results/gemini.jsonl`, verifies that successful
aggregate result against `preflight-gemini-results.json` and the prediction
freeze, writes once atomically, and
prints only run ID, pilot count, screen count, categories, duration summary,
and manifest SHA-256.

- [ ] **Step 5: Run panel tests**

Run: `uv run pytest tests/test_vqa_frontier_phase0_panel.py tests/test_build_vqa_frontier_phase0_panel.py tests/test_vqa_frontier_map_panel.py -q`

Expected: PASS.

- [ ] **Step 6: Commit panel selection**

```bash
git add src/trec_video_2026/vqa_frontier_phase0_panel.py scripts/build_vqa_frontier_phase0_panel.py tests/test_vqa_frontier_phase0_panel.py tests/test_build_vqa_frontier_phase0_panel.py
git commit -m "feat: freeze frontier phase zero panel"
```

---

### Task 3: Extract shared exact-response preservation

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_response_store.py`
- Create: `tests/test_vqa_frontier_response_store.py`
- Modify: `src/trec_video_2026/vqa_frontier_map_runner.py`
- Modify: `tests/test_vqa_frontier_map_runner.py`

**Interfaces:**
- Produces: `parse_prompt_json_response(raw_response: bytes | str) -> object`.
- Produces: `preserve_raw_response(root: Path, *, run_id: str, condition_id: str, video_id: str, raw_response: bytes | str) -> Mapping[str, object]`.
- Produces: `verify_raw_response(path: Path, *, run_id: str, condition_id: str, video_id: str, response_sha256: str) -> None`.
- Existing runner retains its public API and artifact semantics.

- [ ] **Step 1: Write shared-store tests for the exact envelope and conflicts**

```python
def test_parse_removes_only_one_exact_json_fence() -> None:
    assert parse_prompt_json_response('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        parse_prompt_json_response('```JSON\n{"ok": true}\n```')


def test_preserve_raw_response_never_overwrites_conflicting_billable_body(tmp_path: Path) -> None:
    preserve_raw_response(tmp_path, run_id="r", condition_id="neutral_map", video_id="v", raw_response='{"a":1}')
    with pytest.raises(RawResponseConflictError):
        preserve_raw_response(tmp_path, run_id="r", condition_id="neutral_map", video_id="v", raw_response='{"a":2}')
    assert (tmp_path / "raw-conflicts" / "neutral_map" / "v").is_dir()
```

- [ ] **Step 2: Verify the new tests fail**

Run: `uv run pytest tests/test_vqa_frontier_response_store.py -q`

Expected: FAIL during import.

- [ ] **Step 3: Move—not duplicate—the exact parser and raw-record rules**

Raw records use this stable shape:

```python
record = {
    "schema_version": 1,
    "run_id": run_id,
    "condition_id": condition_id,
    "video_id": video_id,
    "raw_response_sha256": canonical_sha256(raw_text),
    "raw_response": raw_text,
}
```

Reject unsafe condition/video path segments and symlinks. Repeated identical
writes are idempotent. A different body is written under a no-follow,
unique conflict directory before raising `RawResponseConflictError`; it never
replaces the first body.

- [ ] **Step 4: Make the old runner delegate through compatibility wrappers**

Keep `_parse_raw_response`, `_write_raw_response`, and
`_verify_raw_response_record` callable for existing tests, but make them thin
wrappers over the shared module with `condition_id=provider`. Do not alter old
request hashes, raw-file paths, or freeze bindings.

- [ ] **Step 5: Run shared and old runner regression tests**

Run: `uv run pytest tests/test_vqa_frontier_response_store.py tests/test_vqa_frontier_map_runner.py -q`

Expected: PASS with the existing frontier-map artifact behavior unchanged.

- [ ] **Step 6: Commit response preservation**

```bash
git add src/trec_video_2026/vqa_frontier_response_store.py src/trec_video_2026/vqa_frontier_map_runner.py tests/test_vqa_frontier_response_store.py tests/test_vqa_frontier_map_runner.py
git commit -m "refactor: share frontier response preservation"
```

---

### Task 4: Build the immutable Phase 0 runner and cost gates

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_phase0_runner.py`
- Create: `scripts/run_vqa_frontier_phase0.py`
- Create: `tests/test_vqa_frontier_phase0_runner.py`
- Create: `tests/test_run_vqa_frontier_phase0_script.py`

**Interfaces:**
- Consumes: `Phase0Panel`, Task 1 contracts, `OpenRouterTransport`, `prepare_delivery_media`, Task 3 response store, and `freeze_json_artifact`.
- Produces: `PreparedPhase0Run` and `Phase0RunSummary`.
- Produces: `prepare_run(panel: Path, output: Path) -> PreparedPhase0Run`.
- Produces: `run_pilot(prepared, transport) -> Phase0RunSummary`.
- Produces: `authorize_screen(prepared, pilot_audit_freeze: Path) -> Mapping[str, object]`.
- Produces: `run_screen_row(prepared, *, ordinal: int, transport_factory: Callable[[str], FrontierTransport]) -> Phase0RunSummary`.
- Produces: `project_remaining_cost(prepared) -> Mapping[str, object]`, `freeze_pilot(prepared)`, `freeze_screen(prepared)`, and `status(prepared)`.

- [ ] **Step 1: Write runner tests for condition order and information boundaries**

```python
def test_screen_row_dispatches_exactly_three_calls_with_one_neutral_map(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=True)
    transport = RecordingTransport()
    summary = run_screen_row(prepared, ordinal=1, transport_factory=lambda _: transport)
    assert summary.passed == 3
    assert [call.condition_id for call in transport.calls] == [
        "neutral_map", "direct_answer", "map_guided_answer"
    ]
    assert SENTINEL_QUESTION not in transport.calls[0].prompt
    assert SENTINEL_QUESTION in transport.calls[1].prompt
    assert SENTINEL_QUESTION in transport.calls[2].prompt
    assert '"event_id":"E1"' in canonicalize(transport.calls[2].prompt)


def test_screen_cannot_run_without_usable_pilot_audit(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, pilot_authorized=False)
    with pytest.raises(PilotGateError, match="pilot audit authorization"):
        run_screen_row(prepared, ordinal=1, transport_factory=lambda _: RecordingTransport())
```

- [ ] **Step 2: Write cost and retry tests**

```python
def test_cost_gate_stops_before_twenty_five_dollars(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path, cumulative_cost=24.90, pilot_authorized=True)
    transport = RecordingTransport(next_cost=0.11)
    with pytest.raises(CostGateError, match="USD 25"):
        run_screen_row(prepared, ordinal=1, transport_factory=lambda _: transport)
    assert transport.call_count == 0


@pytest.mark.parametrize("failure", ["transport", "rate_limit", "empty", "truncated"])
def test_one_identical_retry_for_only_approved_failures(tmp_path: Path, failure: str) -> None:
    transport = FailOnceTransport(failure)
    run_pilot(prepared_fixture(tmp_path), transport)
    assert transport.request_hashes[0] == transport.request_hashes[1]
    assert len(transport.request_hashes) == 2
```

- [ ] **Step 3: Verify runner tests fail**

Run: `uv run pytest tests/test_vqa_frontier_phase0_runner.py tests/test_run_vqa_frontier_phase0_script.py -q`

Expected: FAIL during import.

- [ ] **Step 4: Implement prepared-run and request identities**

The private run layout is fixed:

```text
panel.json
run-manifest.json
requests.jsonl
attempts.jsonl
results/neutral_map.jsonl
results/direct_answer.jsonl
results/map_guided_answer.jsonl
raw/<condition>/<video_id>.json
delivery/<provider>/<video_id>/...
pilot-results.json
pilot-freeze.json
pilot-authorization.json
cost-gate.json
screen-results.json
screen-freeze.json
terminal-status.json
```

Every request hash binds run/panel/source/delivery/model/condition/prompt/schema,
plus the neutral-map SHA-256 for the guided condition. A completed request is
resumable only if every binding matches. A started attempt lacking a terminal
record blocks automatic resume.

- [ ] **Step 5: Implement condition-specific generation with held-fixed answer settings**

Use `OpenRouterRequestPolicy` as follows:

```python
MAP_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=8192,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=False,
)
ANSWER_POLICY = OpenRouterRequestPolicy(
    max_output_tokens=4096,
    temperature=0.0,
    allow_fallbacks=False,
    use_structured_outputs=False,
)
```

Both answer conditions use the same `ANSWER_POLICY` and prompt builder. The
neutral condition uses the neutral prompt and schema. Upload the verified
lossless delivery media for every call, delete the transport-local upload in a
`finally` block, preserve the raw body before validation, then apply strict
local validation without content repair.

- [ ] **Step 6: Implement pilot, screen, freeze, and cost state transitions**

The runner enforces:

```text
prepared -> pilot complete -> pilot frozen -> pilot audit authorized
-> first screen row complete -> projected total < USD 25
-> remaining screen rows -> screen frozen
```

`project_remaining_cost` uses provider-reported cost when available, includes
failed billable attempts, reports observed cost by condition, and projects all
remaining calls from observed per-condition means. The first-row cost gate file
must bind all three first-row result hashes and declare
`projected_total_usd < 25.0` before ordinals 2–7 may dispatch.

- [ ] **Step 7: Implement safe CLI commands**

```text
prepare --panel PATH --output PATH
preflight
run-pilot --prepared PATH
freeze-pilot --prepared PATH
authorize-screen --prepared PATH --pilot-audit-freeze PATH
run-screen-row --prepared PATH --ordinal 1..7
project-cost --prepared PATH
authorize-remaining --prepared PATH
run-screen-remaining --prepared PATH
freeze-screen --prepared PATH
status --prepared PATH
```

CLI stdout contains only counts, safe status names, hashes, and cost. It must
never print questions, answers, prompts, raw responses, credentials, media
paths, or provider error bodies.

- [ ] **Step 8: Run focused and regression tests**

Run: `uv run pytest tests/test_vqa_frontier_phase0_runner.py tests/test_run_vqa_frontier_phase0_script.py tests/test_vqa_frontier_map_runner.py tests/test_vqa_frontier_map_media.py tests/test_vqa_submission_openrouter.py -q`

Expected: PASS.

- [ ] **Step 9: Commit runner and CLI**

```bash
git add src/trec_video_2026/vqa_frontier_phase0_runner.py scripts/run_vqa_frontier_phase0.py tests/test_vqa_frontier_phase0_runner.py tests/test_run_vqa_frontier_phase0_script.py
git commit -m "feat: add bounded frontier phase zero runner"
```

---

### Task 5: Build the two-view native-evidence audit and pilot authorization

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_phase0_audit.py`
- Create: `scripts/build_vqa_frontier_phase0_audit.py`
- Create: `tests/test_vqa_frontier_phase0_audit.py`
- Create: `tests/test_build_vqa_frontier_phase0_audit.py`

**Interfaces:**
- Consumes: pilot or screen result freeze, Phase 0 panel, canonical source media, neutral maps, and answer citations.
- Produces: `build_audit_packet(prepared: Path, *, phase: Literal["pilot", "screen"], output: Path) -> AuditPacket`.
- Produces: `validate_audit_packet(root: Path) -> Mapping[str, object]`.
- Produces: `freeze_pilot_audit(root: Path) -> Mapping[str, object]` and `freeze_screen_audit(root: Path) -> Mapping[str, object]`.
- Pilot freeze contains `neutral_map_decision: usable_for_screen | revise_contract | stop`.

- [ ] **Step 1: Write audit leakage and completeness tests**

```python
def test_blind_manifest_contains_map_but_no_question_or_answer(tmp_path: Path) -> None:
    packet = build_audit_packet(prepared_fixture(tmp_path), phase="screen", output=tmp_path / "audit")
    payload = json.loads(packet.blind_manifest.read_text())
    rendered = json.dumps(payload).lower()
    assert "sentinel private question" not in rendered
    assert "top1_answer" not in rendered
    assert "reference" not in rendered
    assert "events" in rendered


def test_screen_audit_requires_every_map_event_and_answer_claim(tmp_path: Path) -> None:
    packet = complete_audit_fixture(tmp_path)
    remove_one_event_verdict(packet.map_verdicts)
    with pytest.raises(ValueError, match="every neutral-map event"):
        freeze_screen_audit(packet.root)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_audit.py -q`

Expected: FAIL during import.

- [ ] **Step 3: Implement private media materialization and manifests**

For each row create:

```text
rows/<video_id>/overview.jpg
rows/<video_id>/event-keyframes/<event_id>-<index>.jpg
rows/<video_id>/answer-citations/direct_answer/<claim_id>-<index>.jpg
rows/<video_id>/answer-citations/map_guided_answer/<claim_id>-<index>.jpg
```

Use six uniform overview frames, every neutral-map keyframe, and each cited
span's start/mid/end source timestamps clamped to decoded duration. Bind desired
time, actual decoded PTS, frame SHA-256, source SHA-256, and ffmpeg version.
Keep the canonical native-video path only in the private packet so the primary
agent can inspect motion/audio directly. Do not run ASR or an independent audio
model; mark audio-only claims `unverifiable` if native playback cannot establish
them.

- [ ] **Step 4: Implement exact verdict schemas**

`map-verdicts.jsonl` contains one row per video:

```json
{
  "video_id": "safe-id",
  "event_checks": [
    {"event_id": "E1", "status": "confirmed", "boundary_status": "confirmed", "notes": "concise audit note"}
  ],
  "coverage": "complete",
  "entity_identity": "confirmed",
  "ordering": "confirmed",
  "visible_text": "unverifiable",
  "speech_and_sound": "unverifiable",
  "answer_critical_evidence": "not_assessed",
  "overall_decision": "usable_for_screen",
  "notes": "concise row note"
}
```

Allowed observation states are `confirmed`, `contradicted`, and
`unverifiable`; coverage is `complete`, `partial`, or `insufficient`. Pilot
`answer_critical_evidence` must be `not_assessed`. Screen audit changes it to
`present`, `partial`, or `absent` after the question is revealed.

`answer-verdicts.jsonl` contains one row per video and answer condition with
claim-level native-span support, guided map-event citation validity,
unsupported-detail flag, failure causes, and a supported-top1 boolean. It
contains questions/answers privately but never references.

- [ ] **Step 5: Implement freeze and authorization validation**

Pilot authorization is possible only when all map events have verdicts, no
answer-changing contradiction exists, timestamps are inspectable, and the row
has `overall_decision == "usable_for_screen"`. Screen audit freeze requires
seven complete map verdicts plus fourteen complete answer verdicts and binds
the screen prediction freeze SHA-256. Both freezes reject reference fields.

- [ ] **Step 6: Run audit tests and old audit regression tests**

Run: `uv run pytest tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_map_audit.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the audit path**

```bash
git add src/trec_video_2026/vqa_frontier_phase0_audit.py scripts/build_vqa_frontier_phase0_audit.py tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_audit.py
git commit -m "feat: add frontier phase zero evidence audit"
```

---

### Task 6: Verify the complete workflow offline before spending

**Files:**
- Modify: `tests/test_vqa_frontier_phase0_runner.py`
- Modify: `tests/test_run_vqa_frontier_phase0_script.py`
- Modify: `tests/test_build_vqa_frontier_phase0_audit.py`

**Interfaces:**
- Consumes all Tasks 1–5.
- Produces one synthetic, artifact-complete pilot and seven-row screen using `FixtureTransport`; no network or credentials.

- [ ] **Step 1: Add one end-to-end fixture test**

```python
def test_fixture_end_to_end_freezes_pilot_screen_and_audits(tmp_path: Path) -> None:
    panel = write_phase0_fixture_panel(tmp_path)
    prepared = prepare_run(panel, tmp_path / "run")
    run_pilot(prepared, FixtureTransport())
    freeze_pilot(prepared)
    pilot_audit = build_complete_fixture_audit(prepared, phase="pilot")
    freeze_pilot_audit(pilot_audit.root)
    authorize_screen(prepared, pilot_audit.root / "freeze.json")
    for ordinal in range(1, 8):
        run_screen_row(prepared, ordinal=ordinal, transport_factory=fixture_factory)
    freeze_screen(prepared)
    screen_audit = build_complete_fixture_audit(prepared, phase="screen")
    freeze_screen_audit(screen_audit.root)
    assert status(prepared)["terminal_conditions"] == 22
```

- [ ] **Step 2: Run the end-to-end test alone**

Run: `uv run pytest tests/test_vqa_frontier_phase0_runner.py::test_fixture_end_to_end_freezes_pilot_screen_and_audits -q`

Expected: PASS with 1 pilot result and 21 screen results.

- [ ] **Step 3: Run the complete focused suite**

Run: `uv run pytest tests/test_vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_panel.py tests/test_vqa_frontier_response_store.py tests/test_vqa_frontier_phase0_runner.py tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_panel.py tests/test_run_vqa_frontier_phase0_script.py tests/test_build_vqa_frontier_phase0_audit.py -q`

Expected: PASS.

- [ ] **Step 4: Run all frontier and OpenRouter regressions**

Run: `uv run pytest tests/test_vqa_frontier_map.py tests/test_vqa_frontier_map_media.py tests/test_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_runner.py tests/test_build_vqa_frontier_map_audit.py tests/test_vqa_submission_openrouter.py -q`

Expected: PASS.

- [ ] **Step 5: Run static and secret checks**

```bash
uv run ruff check src/trec_video_2026/vqa_frontier_phase0_*.py src/trec_video_2026/vqa_frontier_response_store.py scripts/*vqa_frontier_phase0*.py tests/test_*vqa_frontier_phase0*.py
git diff --check
git grep -nE 'sk-or-v1-|bdecc551|OPENROUTER_API_KEY=' -- ':!docs/superpowers/plans/2026-08-12-vqa-frontier-phase0.md'
```

Expected: ruff and diff checks pass; secret scan prints nothing.

- [ ] **Step 6: Commit offline workflow verification**

```bash
git add tests/test_vqa_frontier_phase0_runner.py tests/test_run_vqa_frontier_phase0_script.py tests/test_build_vqa_frontier_phase0_audit.py
git commit -m "test: verify frontier phase zero workflow"
```

---

### Task 7: Run and audit the one-video paid neutral-map pilot

**Files:**
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/panel.json`
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/run/`
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/pilot-audit/`
- Modify only if the pilot exposes a contract defect: Task 1 contract/tests and this plan's status notes.

**Interfaces:**
- Consumes: verified OpenRouter credential at runtime, successful spent canary artifacts, frozen source panel, and Qwen frames-only predictions.
- Produces: one paid question-blind map, immutable pilot freeze, native-media audit, pilot audit freeze, and screen authorization or a stop decision.

- [ ] **Step 1: Verify shared data and build the private panel**

```bash
test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
uv run python scripts/build_vqa_frontier_phase0_panel.py \
  --source-panel data/vqa/benchmarks/frontier-video-map-20260811/frozen-panel.json \
  --successful-pilot-results data/vqa/benchmarks/frontier-video-map-20260811/run-webm-v6-second/preflight-gemini-results.json \
  --successful-pilot-freeze data/vqa/benchmarks/frontier-video-map-20260811/run-webm-v6-second/preflight-gemini-freeze.json \
  --qwen-predictions data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-only.json \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/panel.json
```

Expected: safe JSON summary with `pilot_count: 1`, `screen_count: 7`, and all seven categories.

- [ ] **Step 2: Prepare locally and run authenticated no-media preflight**

```bash
uv run python scripts/run_vqa_frontier_phase0.py prepare \
  --panel data/vqa/benchmarks/vqa-frontier-phase0-20260812/panel.json \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py preflight \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
```

Expected: authenticated model identity is `google/gemini-3.1-pro-preview`; no media call or charge.

- [ ] **Step 3: Run exactly one paid pilot call and freeze it**

```bash
uv run python scripts/run_vqa_frontier_phase0.py run-pilot \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py freeze-pilot \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py project-cost \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
```

Expected: one terminal neutral-map result, one raw response, observed cost recorded, projected total below USD 25.

- [ ] **Step 4: Build and complete the blind pilot audit**

```bash
uv run python scripts/build_vqa_frontier_phase0_audit.py build \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run \
  --phase pilot \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/pilot-audit
```

Inspect the complete native video and every overview/event frame. Fill the
private map verdict with `confirmed`, `contradicted`, or `unverifiable` for
every event and boundary. Do not reveal or use the question/reference. Then:

```bash
uv run python scripts/build_vqa_frontier_phase0_audit.py freeze-pilot \
  --audit data/vqa/benchmarks/vqa-frontier-phase0-20260812/pilot-audit
```

- [ ] **Step 5: Apply the pilot decision rule**

If the freeze says `usable_for_screen`, authorize the screen:

```bash
uv run python scripts/run_vqa_frontier_phase0.py authorize-screen \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run \
  --pilot-audit-freeze data/vqa/benchmarks/vqa-frontier-phase0-20260812/pilot-audit/freeze.json
```

If it says `revise_contract` or `stop`, do not call a fresh row. Preserve the
pilot, record the exact gap, update contract tests, and return to Task 1 as a
new condition revision.

---

### Task 8: Run, audit, and freeze the seven-video mechanism screen

**Files:**
- Modify privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/run/`
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/screen-audit/`

**Interfaces:**
- Requires: usable pilot authorization.
- Produces: 7 neutral maps, 7 direct answers, 7 map-guided answers, 21 raw responses/terminal records, a screen prediction freeze, and a complete pre-reference audit freeze.

- [ ] **Step 1: Run the first fresh row's three conditions**

```bash
uv run python scripts/run_vqa_frontier_phase0.py run-screen-row \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run \
  --ordinal 1
```

Expected: three terminal condition records in neutral/direct/guided order.

- [ ] **Step 2: Project cost and authorize only if below the hard ceiling**

```bash
uv run python scripts/run_vqa_frontier_phase0.py project-cost \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py authorize-remaining \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
```

Expected: `projected_total_usd < 25.0` and immutable `cost-gate.json`. If not,
stop with the first-row artifacts preserved.

- [ ] **Step 3: Run the remaining six rows resumably and freeze predictions**

```bash
uv run python scripts/run_vqa_frontier_phase0.py run-screen-remaining \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py status \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
uv run python scripts/run_vqa_frontier_phase0.py freeze-screen \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run
```

Expected: 7/7 terminal rows per condition, no ambiguous attempts, cumulative cost below USD 25, and `screen-freeze.json` binding all 21 results/raw-response records.

- [ ] **Step 4: Build the two-view screen audit packet**

```bash
uv run python scripts/build_vqa_frontier_phase0_audit.py build \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run \
  --phase screen \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/screen-audit
```

Expected: seven blind map rows, seven aware rows, every map keyframe, and all direct/guided citation frames.

- [ ] **Step 5: Complete the blind map audit before opening questions**

Inspect each native video and the full neutral map. Fill all seven private map
verdicts, including every event ID and boundaries. Keep the question and both
answers hidden during this view.

- [ ] **Step 6: Complete the question-aware evidence audit without references**

Reveal each question and both frozen Gemini answers, but not the reference.
For all 14 answers, verify each native span, guided event citation, top-1
support, unsupported details, and failure mechanism. Sound/speech claims not
established from native playback are `unverifiable`; do not add Whisper.

- [ ] **Step 7: Freeze the complete audit before label access**

```bash
uv run python scripts/build_vqa_frontier_phase0_audit.py validate \
  --audit data/vqa/benchmarks/vqa-frontier-phase0-20260812/screen-audit
uv run python scripts/build_vqa_frontier_phase0_audit.py freeze-screen \
  --audit data/vqa/benchmarks/vqa-frontier-phase0-20260812/screen-audit
```

Expected: freeze binds seven map verdicts, fourteen answer verdicts, all audit media, and the screen prediction freeze; no reference fields are present.

---

### Task 9: Score only after freezes and diagnose the result

**Files:**
- Create: `scripts/score_vqa_frontier_phase0.py`
- Create: `tests/test_score_vqa_frontier_phase0.py`
- Create privately: `data/vqa/benchmarks/vqa-frontier-phase0-20260812/private-score.json`

**Interfaces:**
- Consumes: panel, screen freeze/results, screen audit freeze, Qwen frames-only predictions, annotation root, and pinned BERT model snapshot.
- Produces: private row-level score plus sanitized projection containing no questions, answers, references, raw responses, credentials, or private paths.
- Reuses: `vqa_evaluation.normalized_exact_match`, `token_f1`, `semantic_composite`, and pinned metric helpers from `scripts/score_vqa_semantic_panel.py`.

- [ ] **Step 1: Write scorer integrity and metric tests**

```python
def test_scorer_refuses_labels_before_prediction_and_audit_freezes(tmp_path: Path) -> None:
    inputs = score_fixture(tmp_path, screen_frozen=False)
    with pytest.raises(ValueError, match="screen prediction freeze"):
        score_phase0(**inputs)


def test_scorer_reports_three_conditions_and_degenerate_ndcg(tmp_path: Path) -> None:
    result = score_phase0(**score_fixture(tmp_path))
    assert set(result["conditions"]) == {
        "gemini_direct", "gemini_map_guided", "qwen35_uniform12"
    }
    assert result["ndcg"] == {"value": 1.0, "degenerate": True, "informative": False}
    assert result["panel"]["rows"] == 7
```

- [ ] **Step 2: Verify scorer tests fail**

Run: `uv run pytest tests/test_score_vqa_frontier_phase0.py -q`

Expected: FAIL because the scorer does not exist.

- [ ] **Step 3: Implement strict input binding before annotation access**

Verify in this order: panel hash/order/source digest; complete screen freeze;
every raw response hash; complete audit freeze; exact selected Qwen rows and
source prediction hash. Only then read the seven annotation JSON files from the
canonical annotation root.

- [ ] **Step 4: Implement official-style descriptive scoring**

For all three conditions compute per-row and aggregate normalized exact match,
token F1, METEOR, BERTScore-F1, STS, and semantic composite. Report direct vs
guided semantic wins/losses/ties and each Gemini arm vs Qwen. Do not report a
confidence interval or population significance claim for seven rows. Record
single-answer NDCG as exactly degenerate.

Add audited mechanism summaries: neutral-map confirmed/contradicted/
unverifiable events, answer-critical evidence coverage, native-span support,
guided event-citation validity, unsupported-detail count, and the seven failure
categories from the spec.

- [ ] **Step 5: Run scorer tests**

Run: `uv run pytest tests/test_score_vqa_frontier_phase0.py tests/test_score_vqa_frontier_map.py tests/test_score_vqa_semantic_panel.py -q`

Expected: PASS.

- [ ] **Step 6: Run the post-freeze private scorer**

```bash
uv run python scripts/score_vqa_frontier_phase0.py \
  --prepared data/vqa/benchmarks/vqa-frontier-phase0-20260812/run \
  --audit-freeze data/vqa/benchmarks/vqa-frontier-phase0-20260812/screen-audit/freeze.json \
  --annotations data/vqa/2025/training/annotations \
  --qwen-predictions data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-only.json \
  --bert-model "/home/npatta01/.cache/huggingface/hub/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59" \
  --output data/vqa/benchmarks/vqa-frontier-phase0-20260812/private-score.json
```

Expected: seven rows per condition, all metric values finite, labels joined only after freezes, and a private score hash printed.

- [ ] **Step 7: Commit the scorer**

```bash
git add scripts/score_vqa_frontier_phase0.py tests/test_score_vqa_frontier_phase0.py
git commit -m "feat: score frontier phase zero screen"
```

---

### Task 10: Publish the condition-bound report and durable artifact pointers

**Files:**
- Create: `docs/experiments/2026-08-12-vqa-frontier-phase0.qmd`
- Create: `docs/experiments/artifacts/vqa-frontier-phase0-20260812/README.md`
- Create sanitized JSON under: `docs/experiments/artifacts/vqa-frontier-phase0-20260812/`
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Modify when the queue changes: `docs/project-status/experiment-ledger.qmd`
- Modify: `AGENTS.md`
- Generate, never hand-edit: sibling Markdown and HTML report files.

**Interfaces:**
- Consumes: private score, run/audit freezes, attempt ledger, and immutable hashes.
- Produces: one routing-owned claim, cross-theme links, compact sanitized evidence, and canonical private paths accessible from every shared checkout.

- [ ] **Step 1: Build a sanitized evidence projection**

Track only:

```text
README.md
summary.json
metrics.json
map-audit-summary.json
cost.json
manifest.json
SHA256SUMS
```

Exclude questions, answers, references, raw responses, prompts containing
questions, media, private paths, and credentials. Include model/config hashes,
panel category/duration aggregates, completion denominators, metric aggregates,
row-level safe IDs/metrics only if already allowed by the existing disclosure
boundary, audit counts, costs, decisions, and hashes of private freezes.

- [ ] **Step 2: Write the dated QMD result with the predeclared decision table**

The report must separate observed results from interpretation and state one of:

```text
faithful maps + guided better -> promote Gemini/map to larger gate
faithful maps + tie -> carry direct and guided to larger gate; no answer-lift claim
unfaithful maps + tie -> revise neutral annotation
faithful maps + guided weak -> diagnose evidence-to-answer interface
direct consistently stronger -> retain direct Gemini candidate; redesign map interface
missing/mistimed events -> consider one targeted question-blind refinement-pass test
both answer arms fail with evidence present -> block Gemini as sole foundation
```

State that seven rows are diagnostic and NDCG is degenerate.

- [ ] **Step 3: Update the living research memory**

Make Routing, Fusion, and Answer Selection the primary theme. Add only a link
from Visual and Temporal Evidence. Update the hub queue/current decision only
if the Phase 0 outcome changes what should run next. Preserve the older
question-conditioned two-row canary as historical evidence rather than
rewriting it.

- [ ] **Step 4: Add a precise `AGENTS.md` artifact reference**

Record both:

```text
/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/vqa-frontier-phase0-20260812/
/home/npatta01/data/competitions/trec_video_2026/docs/experiments/artifacts/vqa-frontier-phase0-20260812/
```

Explain which directories are private, which files are tracked, that merging
the branch exposes tracked artifacts from the shared checkout, and that no
provider rerun is needed to inspect the preserved run.

- [ ] **Step 5: Render QMD sources and verify report surfaces**

```bash
phase0_render_cache=$(mktemp -d /tmp/trecvid-phase0-render.XXXXXX)
for source in docs/project-status/*.qmd; do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$phase0_render_cache/cache" quarto render "$source" --to gfm --output-dir docs/project-status --output "$page.md"
done
env XDG_CACHE_HOME="$phase0_render_cache/cache" quarto render docs/project-status --to html
env XDG_CACHE_HOME="$phase0_render_cache/cache" quarto render docs/experiments/2026-08-12-vqa-frontier-phase0.qmd --to gfm --output-dir docs/experiments --output 2026-08-12-vqa-frontier-phase0.md
env XDG_CACHE_HOME="$phase0_render_cache/cache" quarto render docs/experiments/2026-08-12-vqa-frontier-phase0.qmd --to html --output-dir reports --output vqa-frontier-phase0-20260812.html
```

Expected: all renders succeed without remote runtime dependencies.

- [ ] **Step 6: Run privacy, link, checksum, and full focused verification**

```bash
git diff --check
git grep -nE 'sk-or-v1-|bdecc551|"raw_response"[[:space:]]*:|"reference_answer"[[:space:]]*:|"correct_answer"[[:space:]]*:' -- docs/experiments/artifacts/vqa-frontier-phase0-20260812 docs/experiments/2026-08-12-vqa-frontier-phase0.qmd docs/project-status AGENTS.md
sha256sum -c docs/experiments/artifacts/vqa-frontier-phase0-20260812/SHA256SUMS
uv run pytest tests/test_vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_panel.py tests/test_vqa_frontier_response_store.py tests/test_vqa_frontier_phase0_runner.py tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_panel.py tests/test_run_vqa_frontier_phase0_script.py tests/test_build_vqa_frontier_phase0_audit.py tests/test_score_vqa_frontier_phase0.py -q
```

Expected: no credential/private-content matches, checksums valid, and focused tests pass.

- [ ] **Step 7: Commit the report and tracked evidence**

```bash
git add \
  AGENTS.md \
  docs/experiments/2026-08-12-vqa-frontier-phase0.qmd \
  docs/experiments/2026-08-12-vqa-frontier-phase0.md \
  docs/experiments/artifacts/vqa-frontier-phase0-20260812 \
  docs/project-status/research-routing.qmd \
  docs/project-status/research-routing.md \
  docs/project-status/research-visual-evidence.qmd \
  docs/project-status/research-visual-evidence.md \
  reports/project-status/research-routing.html \
  reports/project-status/research-visual-evidence.html \
  reports/vqa-frontier-phase0-20260812.html

# Add the experiment-ledger QMD/Markdown/HTML only if Task 10 actually changes
# the queue or current decision recorded there.
git commit -m "docs: record frontier phase zero result"
```

- [ ] **Step 8: Final repository verification**

```bash
git status --short --branch
git log --oneline --decorate -10
```

Expected: only the user's pre-existing unrelated untracked files remain; no private dataset content is staged or committed.
