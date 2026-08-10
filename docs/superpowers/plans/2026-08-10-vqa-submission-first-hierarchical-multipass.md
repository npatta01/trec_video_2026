# Submission-First Hierarchical Multi-Pass VQA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, label-blind VideoQA pipeline that maps the whole video, materializes global and dense local evidence, generates and ranks ten answers, freezes predictions, scores development data, and emits validator-ready 2026 AG JSON portfolios.

**Architecture:** A host-side submission layer sits between the existing manifest/evidence utilities and model runners. It constructs immutable timelines, evidence plans, ten answer requests, and verifier requests; a transport-injected Gemini adapter executes paid multimodal calls; a thin CLI persists every phase and freezes predictions before a separate scorer can access labels. The current Qwen artifacts remain the fail-closed operational fallback.

**Tech Stack:** Python 3.12, dataclasses, JSON/JSONL, pytest, ffprobe/ffmpeg through existing helpers, `google-genai==2.12.1` as an optional `gemini` extra, existing TREC VQA semantic/NIST evaluation utilities.

## Global Constraints

- Only the 475 VQA development rows with local media are quality-bearing; the six missing-media rows remain explicit exclusions.
- The primary provider model ID is exactly `gemini-3.1-pro-preview`; the paid SDK is optional and must never expose credential values.
- Inference requests are recursively label-blind: reject keys or path components containing `answer`, `reference`, `annotation`, `label`, `metric`, `score`, `gold`, or `oracle`, except the candidate field `answer` after generation and never inside model-input provenance.
- Stage 2 produces exactly eight global frames and at most 24 local frames; local sampling is capped at four source frames per second and preserves requested time, actual decoded PTS, source hash, image hash, and fallback reason.
- Candidate pass IDs are exactly `direct-av`, `direct-av-alternate`, `global-transcript`, `global-local-timeline`, `question-specialist`, `local-only`, `audio-first`, `scene-boundary`, `detail-temporal-route`, and `conservative-synthesis`.
- The verifier is reference-blind, may only rank candidate IDs, and cannot generate a new semantic answer.
- Never pad duplicate answers. A top-10 run requires ten normalized-unique answers; top-1 and top-5 use prefixes of the same frozen ranking.
- Submission JSON uses official 2026 fields `Q_ID`, `Video_ID`, and `Answers`, with each answer containing `Rank`, `Answer`, and `Time`.
- Predictions freeze before references load. Hard failures remain in denominators. Report exact match, token F1, METEOR, BERTScore-F1, STS, semantic composite, and rank cutoffs; mark one-answer NDCG degenerate.
- Keep raw media, transcripts, model outputs, labels, private paths, and private URLs out of tracked projections.
- Existing baseline failures are recorded and out of scope: three failures, 2,035 passes, one skip on 2026-08-10.
- The official AG submission deadline is 2026-08-17; preserve a validator-passing fallback before any quality expansion.

---

## File Structure

- Create `src/trec_video_2026/vqa_submission.py`: immutable run, timeline, candidate, ranking, freeze, and AG serialization contracts.
- Create `src/trec_video_2026/vqa_submission_evidence.py`: timeline validation, deterministic global/local frame planning, transcript overlap, and evidence manifests.
- Create `src/trec_video_2026/vqa_submission_gemini.py`: transport-injected Gemini request execution, structured output, usage accounting, upload deletion, and retries.
- Create `scripts/run_vqa_submission.py`: resumable `prepare`, `run`, `freeze`, `status`, and `preflight-live` orchestration.
- Create `scripts/score_vqa_submission.py`: post-freeze label join, semantic/ranked metrics, sanitized projection, and top-1/top-5/top-10 AG JSON.
- Create `tests/test_vqa_submission.py`: contracts, leakage, candidates, rankings, freeze, and serializer.
- Create `tests/test_vqa_submission_evidence.py`: timeline bounds, global/local frame allocation, transcript spans, provenance, and fallbacks.
- Create `tests/test_vqa_submission_gemini.py`: fake-transport request shapes, schema repair, usage, deletion, and secret-safe failures.
- Create `tests/test_run_vqa_submission_script.py`: phase transitions, resume, atomic writes, failure recording, and credential-free preflight.
- Create `tests/test_score_vqa_submission.py`: freeze-before-label enforcement, metrics, safe projection, portfolio generation, and validator invocation.
- Modify `pyproject.toml` and `uv.lock`: optional `gemini` dependency group pinned to `google-genai==2.12.1`.
- Create `configs/vqa/vqa_submission_canary_6.json`: frozen label-free six-row operational canary.
- Create `configs/vqa/vqa_submission_gate_60.json`: frozen label-free 40 representative plus 20 diagnostic gate with cohort provenance but no labels or references.

---

### Task 1: Immutable submission contracts, ranking, freeze, and AG serialization

**Files:**
- Create: `src/trec_video_2026/vqa_submission.py`
- Create: `tests/test_vqa_submission.py`

**Interfaces:**
- Consumes: `canonical_request_sha256` from `trec_video_2026.vqa_gate`; `freeze_json_artifact` and `verify_json_artifact` from `trec_video_2026.vqa_experiments.temporal_bottleneck`.
- Produces: `TimelineSpan`, `RunManifest`, `CandidateRecord`, `VerifierDecision`, `RankedPrediction`, `audit_label_free_payload`, `canonical_sha256`, `deduplicate_candidates`, `apply_verifier_ranking`, `freeze_predictions`, `verify_prediction_freeze`, and `serialize_ag_submission`.

- [ ] **Step 1: Write failing contract and leakage tests**

```python
def test_label_free_audit_rejects_nested_reference_paths() -> None:
    payload = {"request": {"media_path": "/tmp/video.mp4", "nested": {"annotation_path": "/tmp/gold.json"}}}
    with pytest.raises(ValueError, match="annotation"):
        audit_label_free_payload(payload)

def test_run_manifest_hash_changes_when_evidence_policy_changes() -> None:
    first = _manifest(evidence_policy_sha256="a" * 64)
    second = _manifest(evidence_policy_sha256="b" * 64)
    assert first.sha256() != second.sha256()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/test_vqa_submission.py -q`
Expected: collection fails because `trec_video_2026.vqa_submission` does not exist.

- [ ] **Step 3: Implement immutable dataclasses and canonical auditing**

Implement frozen dataclasses with `to_dict()` methods:

```python
@dataclass(frozen=True)
class TimelineSpan:
    span_id: str
    start_seconds: float
    end_seconds: float
    modalities: tuple[str, ...]
    description: str

@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    protocol_version: str
    run_id: str
    ordered_video_ids: tuple[str, ...]
    panel_sha256: str
    model_id: str
    runtime_identity: str
    evidence_policy_sha256: str
    pass_ids: tuple[str, ...]

@dataclass(frozen=True)
class CandidateRecord:
    video_id: str
    pass_id: str
    answer: str
    elapsed_seconds: float
    confidence: float
    support_span_ids: tuple[str, ...]
    request_sha256: str
    runtime_provenance: Mapping[str, object]

@dataclass(frozen=True)
class VerifierDecision:
    ranked_pass_ids: tuple[str, ...]
    support_by_pass: Mapping[str, str]

@dataclass(frozen=True)
class RankedPrediction:
    query_id: int
    video_id: str
    candidates: tuple[CandidateRecord, ...]
    ranked_pass_ids: tuple[str, ...]
    ranking_source: str
    fallback_status: str | None
```

Require `schema_version == 1`, model ID `gemini-3.1-pro-preview`, exact ten pass IDs, unique non-empty video IDs, 64-hex hashes, and canonical UTF-8 JSON hashing. The recursive audit checks both mapping keys and string path components before any model request is persisted.

- [ ] **Step 4: Write failing candidate/ranking tests**

```python
def test_deduplication_keeps_first_candidate_identity() -> None:
    candidates = [_candidate("direct-av", "Four times"), _candidate("global-transcript", "four times.")]
    assert [row.pass_id for row in deduplicate_candidates(candidates)] == ["direct-av"]

def test_verifier_must_rank_every_unique_candidate_once() -> None:
    with pytest.raises(ValueError, match="complete permutation"):
        apply_verifier_ranking(_ten_candidates(), VerifierDecision(("direct-av",), {}))
```

- [ ] **Step 5: Run the candidate tests and verify RED**

Run: `uv run pytest tests/test_vqa_submission.py -q`
Expected: failures name missing candidate and ranking functions.

- [ ] **Step 6: Implement candidate normalization, ranking, and fallback**

`CandidateRecord` carries `video_id`, `pass_id`, non-empty `answer`, `elapsed_seconds`, `confidence`, `support_span_ids`, `request_sha256`, and runtime provenance. Normalize only Unicode whitespace, case, and terminal punctuation for deduplication. `VerifierDecision.ranked_pass_ids` must be a complete permutation. Deterministic fallback precedence is `global-local-timeline`, `direct-av`, `question-specialist`, `detail-temporal-route`, `global-transcript`, `local-only`, `scene-boundary`, `audio-first`, `direct-av-alternate`, `conservative-synthesis`.

- [ ] **Step 7: Write failing freeze and portfolio tests**

```python
def test_tampering_after_freeze_is_detected(tmp_path: Path) -> None:
    predictions = tmp_path / "ranked-predictions.json"
    predictions.write_text(json.dumps(_ranked_payload()), encoding="utf-8")
    freeze = freeze_predictions(predictions, bindings={"run_id": "r1", "row_count": 1})
    predictions.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_prediction_freeze(predictions, freeze)

@pytest.mark.parametrize("cutoff", [1, 5, 10])
def test_ag_serialization_uses_official_json_schema(cutoff: int) -> None:
    payload = serialize_ag_submission([_ranked_prediction()], cutoff=cutoff)
    assert payload[0]["Q_ID"] == 7
    assert len(payload[0]["Answers"]) == cutoff
    assert list(payload[0]["Answers"][0]) == ["Rank", "Answer", "Time"]
```

- [ ] **Step 8: Implement write-once freeze and AG serializer**

Reject existing output paths, non-finite or negative times, duplicate normalized answers, missing ranks, non-contiguous ranks, and cutoffs outside `{1, 5, 10}`. `RankedPrediction` binds integer query ID, video ID, exactly ten candidates, ranking source, and fallback status.

- [ ] **Step 9: Run Task 1 tests and commit**

Run: `uv run pytest tests/test_vqa_submission.py -q`
Expected: all pass.
Commit: `git add src/trec_video_2026/vqa_submission.py tests/test_vqa_submission.py && git commit -m "feat: add immutable VQA submission contracts"`

---

### Task 2: Timeline validation and global/local evidence planning

**Files:**
- Create: `src/trec_video_2026/vqa_submission_evidence.py`
- Create: `tests/test_vqa_submission_evidence.py`

**Interfaces:**
- Consumes: `TimelineSpan`; `bin_midpoint_timestamps`, `probe_media`, and `source_fingerprint` from `vqa_benchmark`; `parse_timestamped_whisper` from `vqa_experiments.silver_oracle`; decoded frame inventories shaped like `probe_frame_inventory` output from `vqa_experiments.temporal_bottleneck`.
- Produces: `ValidatedTimeline`, `FrameSelection`, `EvidencePlan`, `validate_timeline`, `build_global_local_frame_plan`, `select_overlapping_transcript_segments`, and `write_evidence_manifest`.

- [ ] **Step 1: Write failing timeline-bound tests**

```python
def test_timeline_rejects_overlapping_or_out_of_range_spans() -> None:
    with pytest.raises(ValueError, match="duration"):
        validate_timeline(_timeline(end_seconds=31.0), duration_seconds=30.0)

def test_invalid_timeline_uses_declared_global_fallback() -> None:
    result = validate_timeline(_malformed_timeline(), duration_seconds=15.0, allow_fallback=True)
    assert result.fallback_reason == "invalid_timeline"
    assert result.support_spans == ()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_vqa_submission_evidence.py -q`
Expected: module import fails.

- [ ] **Step 3: Implement schema-constrained timeline validation**

Accept chronological events and at most three support spans. Require finite `0 <= start < end <= duration`, unique span IDs, sorted events, declared modality values from `{visual, speech, non_speech_audio, visible_text}`, and bounded descriptions. Expand a support span shorter than one second around its midpoint to exactly one second while clamping to `[0, duration]`; reject media shorter than one second. One fallback record contains no free-form unvalidated fields.

- [ ] **Step 4: Write failing frame-allocation tests**

```python
def test_frame_plan_keeps_eight_global_and_allocates_twenty_four_local() -> None:
    plan = build_global_local_frame_plan(15.0, _inventory(450), _three_spans())
    assert len(plan.global_frames) == 8
    assert len(plan.local_frames) == 24
    assert all(frame.actual_pts_seconds >= 0 for frame in plan.all_frames)
    assert len({frame.decoded_identity for frame in plan.all_frames}) == 32

def test_local_plan_caps_each_span_at_four_frames_per_second() -> None:
    plan = build_global_local_frame_plan(15.0, _inventory(450), [_span(2.0, 3.0)])
    assert len(plan.local_frames) <= 4
```

- [ ] **Step 5: Implement deterministic nearest-unused allocation**

Use eight `bin_midpoint_timestamps(duration, count=8)` targets. Allocate local slots by Hamilton allocation over support-span durations with a minimum of four slots per non-empty span, then cap each span at `floor(duration * 4)` with a minimum of one. Choose the nearest unused decoded frame by `(absolute PTS delta, actual PTS, stream index, frame index)`. If a span cannot fill its allocation, record `local_collision_fallback` and use the nearest unused global frame only after exhausting in-span frames.

- [ ] **Step 6: Write failing transcript/provenance tests**

```python
def test_selected_transcript_is_verbatim_and_overlap_only() -> None:
    selected = select_overlapping_transcript_segments(_timestamped_text(), [_span(1.0, 2.0)], duration_seconds=4.0)
    assert selected.text == "[00:01.00] second line"
    assert selected.source_sha256 == hashlib.sha256(_timestamped_text().encode()).hexdigest()

def test_manifest_records_requested_pts_and_hashes(tmp_path: Path) -> None:
    record = write_evidence_manifest(tmp_path / "evidence.json", _evidence_plan())
    assert record["frames"][0]["actual_pts_seconds"] == 0.48
    assert len(record["frames"][0]["image_sha256"]) == 64
```

- [ ] **Step 7: Implement transcript overlap and atomic manifest writing**

Persist requested time, actual PTS, decoded identity, frame path, image/source hashes, slot kind, support span ID, and fallback reason. Reject labels/references using `audit_label_free_payload`. Refuse overwrite.

- [ ] **Step 8: Run Task 2 tests and commit**

Run: `uv run pytest tests/test_vqa_submission_evidence.py tests/test_vqa_submission.py -q`
Expected: all pass.
Commit: `git add src/trec_video_2026/vqa_submission_evidence.py tests/test_vqa_submission_evidence.py && git commit -m "feat: plan global and local VQA evidence"`

---

### Task 3: Ten pass request construction and verifier contracts

**Files:**
- Modify: `src/trec_video_2026/vqa_submission.py`
- Modify: `tests/test_vqa_submission.py`

**Interfaces:**
- Consumes: `EvidencePlan`, `ValidatedTimeline`, and existing label-free `RunnerRequest` field semantics.
- Produces: `AnswerRequest`, `VerifierRequest`, `build_answer_requests`, `build_backfill_request`, and `build_verifier_request`.

- [ ] **Step 1: Write failing exact-pass and evidence-view tests**

```python
def test_answer_matrix_contains_exact_ten_passes() -> None:
    requests = build_answer_requests(_row(), _timeline(), _evidence_plan())
    assert tuple(requests) == PASS_IDS
    assert requests["direct-av"].media_path == Path("/data/video.mp4")
    assert len(requests["global-local-timeline"].frame_paths) == 32
    assert requests["local-only"].frame_paths == _evidence_plan().local_paths

def test_no_answer_request_contains_reference_or_scores() -> None:
    for request in build_answer_requests(_row(), _timeline(), _evidence_plan()).values():
        audit_label_free_payload(request.to_model_payload())
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_vqa_submission.py -q`
Expected: failures name missing request builders.

- [ ] **Step 3: Implement the ten deterministic request views**

Use exact pass IDs and evidence policies from the design specification. `conservative-synthesis` accepts only the first nine frozen candidate strings plus union evidence. Every request hashes model-visible bytes and records source span IDs. Question routing uses deterministic text rules for counting, temporal, OCR/detail, speech, causal, and other; it never inspects references or historical scores.

- [ ] **Step 4: Write failing verifier/backfill tests**

```python
def test_verifier_randomization_is_seeded_and_hides_pass_names() -> None:
    request = build_verifier_request(_row(), _timeline(), _evidence_plan(), _ten_candidates(), seed=20260810)
    assert [item.public_id for item in request.candidates] != list(PASS_IDS)
    assert "direct-av" not in json.dumps(request.to_model_payload())

def test_backfill_excludes_existing_normalized_answers() -> None:
    request = build_backfill_request(_row(), _evidence_plan(), _nine_candidates(), attempt=1)
    assert request.excluded_normalized_answers == tuple(sorted(set(request.excluded_normalized_answers)))
```

- [ ] **Step 5: Implement public candidate IDs and two bounded backfills**

Public IDs are deterministic SHA-256 prefixes derived from `(run_id, video_id, pass_id, seed)` and do not reveal the pass. Verifier output maps back to private pass IDs only after validation. Backfill attempts are exactly 1 and 2; attempt 3 raises `ValueError`.

- [ ] **Step 6: Run Task 3 tests and commit**

Run: `uv run pytest tests/test_vqa_submission.py -q`
Expected: all pass.
Commit: `git add src/trec_video_2026/vqa_submission.py tests/test_vqa_submission.py && git commit -m "feat: construct ten grounded VQA answer passes"`

---

### Task 4: Transport-injected Gemini adapter and optional dependency

**Files:**
- Create: `src/trec_video_2026/vqa_submission_gemini.py`
- Create: `tests/test_vqa_submission_gemini.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `AnswerRequest`, `VerifierRequest`, `CandidateRecord`, `VerifierDecision`, and `canonical_request_sha256`.
- Produces: `GeminiTransport` protocol, `GeminiSubmissionClient`, `create_google_transport`, `GenerationResult`, and `UploadedFileRecord`.

- [ ] **Step 1: Write failing fake-transport lifecycle tests**

```python
def test_client_uploads_generates_records_usage_and_deletes() -> None:
    transport = FakeTransport(answer_payload=_valid_answer_payload())
    result = GeminiSubmissionClient(transport, model_id="gemini-3.1-pro-preview").answer(_answer_request())
    assert result.answer == "four times"
    assert transport.deleted_ids == transport.uploaded_ids
    assert result.usage.input_tokens == 321

def test_client_deletes_upload_after_generation_failure() -> None:
    transport = FakeTransport(error=RuntimeError("provider failed"))
    with pytest.raises(ProviderError, match="provider failed"):
        GeminiSubmissionClient(transport, model_id="gemini-3.1-pro-preview").answer(_answer_request())
    assert transport.deleted_ids == transport.uploaded_ids
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_vqa_submission_gemini.py -q`
Expected: module import fails.

- [ ] **Step 3: Implement transport protocol and secret-safe errors**

```python
class GeminiTransport(Protocol):
    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord: ...
    def generate(self, *, model: str, parts: Sequence[object], schema: Mapping[str, object]) -> GenerationResult: ...
    def delete(self, file_id: str) -> None: ...
```

The client owns all uploads, deletes in `finally`, records deletion status, and bounds errors to exception class plus sanitized message. It never serializes environment variables, headers, tokens, or client reprs.

- [ ] **Step 4: Write failing schema-repair and verifier tests**

```python
def test_invalid_primary_output_gets_one_fixed_schema_repair() -> None:
    transport = FakeTransport(sequence=[{"bad": True}, _valid_timeline_payload()])
    result = GeminiSubmissionClient(transport, model_id=MODEL_ID).timeline(_timeline_request())
    assert result.attempt_count == 2
    assert transport.generate_count == 2

def test_second_invalid_output_fails_without_third_call() -> None:
    transport = FakeTransport(sequence=[{"bad": True}, {"still_bad": True}])
    with pytest.raises(ProviderProtocolError):
        GeminiSubmissionClient(transport, model_id=MODEL_ID).timeline(_timeline_request())
    assert transport.generate_count == 2
```

- [ ] **Step 5: Implement timeline, answer, and verifier JSON schemas**

Use one fixed repair prompt that contains no model output text beyond the parsed invalid object. Validate all output again with local dataclasses. Record raw response SHA-256, model ID, elapsed time, token usage, attempt count, and uploaded-file deletion status.

- [ ] **Step 6: Add the optional pinned Gemini SDK**

Add:

```toml
[project.optional-dependencies]
gemini = ["google-genai==2.12.1"]
```

Run `uv lock`. `create_google_transport()` lazily imports `google.genai`, calls the SDK's standard client constructor, and raises `ProviderAuthenticationError("Gemini API or Vertex authentication is not configured")` without inspecting or printing credential values.

- [ ] **Step 7: Run Task 4 tests and commit**

Run: `uv run pytest tests/test_vqa_submission_gemini.py tests/test_vqa_submission.py -q`
Expected: all pass without network access.
Commit: `git add pyproject.toml uv.lock src/trec_video_2026/vqa_submission_gemini.py tests/test_vqa_submission_gemini.py && git commit -m "feat: add Gemini VQA submission adapter"`

---

### Task 5: Resumable prepare/run/freeze orchestration

**Files:**
- Create: `scripts/run_vqa_submission.py`
- Create: `tests/test_run_vqa_submission_script.py`

**Interfaces:**
- Consumes: all Task 1-4 interfaces; `load_manifest`, `write_resolved_manifest`, `probe_media`, and existing frame inventory/evidence helpers.
- Produces CLI subcommands `prepare`, `run`, `freeze`, `status`, and `preflight-live`; artifacts `run-manifest.json`, `timeline.jsonl`, `evidence/manifest.json`, `candidates.jsonl`, `ranked-predictions.json`, `failures.jsonl`, `freeze.json`, and `terminal-status.json`.

- [ ] **Step 1: Write failing prepare/resume state-machine tests**

```python
def test_prepare_writes_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    assert main(["prepare", "--config", str(_manifest()), "--output", str(tmp_path / "run")]) == 0
    with pytest.raises(FileExistsError):
        main(["prepare", "--config", str(_manifest()), "--output", str(tmp_path / "run")])

def test_run_resumes_only_hash_matching_completed_rows(tmp_path: Path) -> None:
    transport = FixtureTransport()
    first = run_prepared(_prepared(tmp_path), transport=transport, limit=1)
    second = run_prepared(_prepared(tmp_path), transport=transport)
    assert first.completed_rows == 1
    assert second.reused_rows == 1
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_run_vqa_submission_script.py -q`
Expected: script/module loader fails.

- [ ] **Step 3: Implement atomic prepare and row-level checkpoints**

Prepare verifies the canonical data symlink, media existence, unique ordered IDs, panel digest, model ID, ten pass IDs, and label-free input schema. Write to a sibling temporary path, fsync, then rename. Each JSONL attempt includes request bytes and hash, response hash, status, timing, and bounded failure code. Resume reuses only exact request-hash matches.

- [ ] **Step 4: Write failing full fixture-transport orchestration test**

```python
def test_fixture_run_completes_timeline_ten_candidates_verifier_and_freeze(tmp_path: Path) -> None:
    run_dir = _prepare_one_row(tmp_path)
    assert main(["run", "--prepared", str(run_dir), "--transport", "fixture"]) == 0
    assert main(["freeze", "--prepared", str(run_dir)]) == 0
    assert len(_jsonl(run_dir / "candidates.jsonl")) == 10
    verify_prediction_freeze(run_dir / "ranked-predictions.json", _json(run_dir / "freeze.json"))
```

- [ ] **Step 5: Implement phase orchestration and deterministic fallbacks**

Order per row: timeline → evidence → passes 1-9 → pass 10 → up to two backfills → verifier → ranked prediction. On invalid timeline use global-only evidence. On invalid verifier use deterministic pass precedence. On fewer than ten unique candidates, record `candidate_shortfall` and bind the configured baseline ranked-answer fallback. Never load references.

- [ ] **Step 6: Implement credential-free status and live preflight**

`status` reads artifacts only. `preflight-live` checks that the optional SDK imports and client construction succeeds, but uploads no media and emits only `{model_id, sdk_version, authentication_configured}`. Missing authentication exits 2 with the fixed safe message from Task 4.

- [ ] **Step 7: Run Task 5 tests and commit**

Run: `uv run pytest tests/test_run_vqa_submission_script.py tests/test_vqa_submission*.py -q`
Expected: all pass.
Commit: `git add scripts/run_vqa_submission.py tests/test_run_vqa_submission_script.py && git commit -m "feat: orchestrate resumable VQA submissions"`

---

### Task 6: Post-freeze scoring, safe projection, and submission portfolio

**Files:**
- Create: `scripts/score_vqa_submission.py`
- Create: `tests/test_score_vqa_submission.py`

**Interfaces:**
- Consumes: `verify_prediction_freeze`, `serialize_ag_submission`, `evaluate_saved_predictions`, `ranked_cutoff_metrics`, `prepare_nist_invocation`, and `execute_nist_invocation`.
- Produces: `score_frozen_predictions`, `build_safe_projection`, `write_submission_portfolio`, and CLI commands `score` and `validate`.

- [ ] **Step 1: Write failing freeze-before-label and alignment tests**

```python
def test_labels_cannot_load_before_valid_freeze(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="freeze"):
        score_frozen_predictions(tmp_path / "ranked.json", tmp_path / "missing-freeze.json", _references())

def test_misaligned_video_ids_fail_before_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ordered video IDs"):
        score_frozen_predictions(*_frozen_paths(tmp_path, ids=["a"]), references={"b": "answer"})
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_score_vqa_submission.py -q`
Expected: script/module loader fails.

- [ ] **Step 3: Implement failure-zero top-1 and ranked metrics**

Join references only after freeze verification. Emit per-row exact match, token F1, METEOR, BERTScore-F1, STS, composite, completion/fallback status, and rank-1/best/mean/NDCG at 1, 5, and 10 where semantic scores exist. Reuse the pinned semantic scorer inputs rather than silently substituting new models. Hard failures receive zero for aggregate metrics.

- [ ] **Step 4: Write failing privacy-projection and portfolio tests**

```python
def test_safe_projection_rejects_private_fields() -> None:
    with pytest.raises(ValueError, match="transcript"):
        build_safe_projection({"metrics": {}, "transcript": "private"})

def test_portfolio_writes_three_official_json_files(tmp_path: Path) -> None:
    paths = write_submission_portfolio(_ranked_rows(), tmp_path)
    assert {path.name for path in paths} == {"submission-top1.json", "submission-top5.json", "submission-top10.json"}
    assert len(json.loads((tmp_path / "submission-top10.json").read_text())[0]["Answers"]) == 10
```

- [ ] **Step 5: Implement recursive allowlist projection and JSON portfolio**

Tracked projection allows run/model/protocol IDs, video IDs, strata, statuses, counts, timing aggregates, artifact hashes, aggregate/per-stratum metrics, paired deltas, intervals, and bounded failure codes. It rejects raw answers, references, transcripts, media paths, frame paths, audio paths, provider file IDs, and private URLs. Submission files remain private runtime artifacts.

- [ ] **Step 6: Implement official validator invocation**

Use `prepare_nist_invocation(NIST_2026_VALIDATION_MODE, ...)` and `execute_nist_invocation` with argv-only, credential-free environment, explicit timeout, and isolated directories. Retain each validation report and output SHA-256. A portfolio file is publishable only when its validator exit code is zero.

- [ ] **Step 7: Run Task 6 tests and commit**

Run: `uv run pytest tests/test_score_vqa_submission.py tests/test_vqa_evaluation.py tests/test_vqa_submission.py -q`
Expected: all pass.
Commit: `git add scripts/score_vqa_submission.py tests/test_score_vqa_submission.py && git commit -m "feat: score and serialize VQA submission portfolios"`

---

### Task 7: Freeze canary/gate manifests and run the operational preflight

**Files:**
- Create: `configs/vqa/vqa_submission_canary_6.json`
- Create: `configs/vqa/vqa_submission_gate_60.json`
- Modify: `tests/test_run_vqa_submission_script.py`

**Interfaces:**
- Consumes: canonical 475-row manifest, question-derived strata, source metadata, historical diagnostic artifacts only for the 20 diagnostic rows, and Task 5 preflight.
- Produces: immutable six-row and 60-row label-free manifests with ordered-ID and source hashes.

- [ ] **Step 1: Write failing manifest-integrity tests**

```python
def test_canary_has_exact_six_declared_strata_and_no_labels() -> None:
    payload = _load_config(CANARY_CONFIG)
    assert len(payload["examples"]) == 6
    assert {row["selection_stratum"] for row in payload["examples"]} == {"temporal", "counting", "audio_text", "ocr_detail", "causal", "other"}
    audit_label_free_payload(payload)

def test_gate_separates_representative_and_diagnostic_rows() -> None:
    payload = _load_config(GATE_CONFIG)
    assert sum(row["analysis_population"] == "representative" for row in payload["examples"]) == 40
    assert sum(row["analysis_population"] == "diagnostic" for row in payload["examples"]) == 20
    assert len({row["video_id"] for row in payload["examples"]}) == 60
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_run_vqa_submission_script.py -q`
Expected: config files are missing.

- [ ] **Step 3: Generate and freeze both label-free manifests**

Representative quotas are exactly audio/text 5, causal 6, counting 7, other 13, temporal 9. Within each input-derived pool sort by `SHA256("vqa-submission-60-v1|representative|" + video_id)`. Diagnostic quotas are prompt/schema 2, reference/metric ambiguity 3, transition/localization 5, audio 4, FPS/resolution 3, reasoning/model 3; resolve overlaps with that precedence and sort by `SHA256("vqa-submission-60-v1|diagnostic|" + video_id)`. Store only the diagnostic category name, not prior answers, references, metrics, or rationales.

- [ ] **Step 4: Run config tests and local prepare**

Run:

```bash
uv run pytest tests/test_run_vqa_submission_script.py -q
uv run python scripts/run_vqa_submission.py prepare \
  --config configs/vqa/vqa_submission_canary_6.json \
  --dataset-root data \
  --output data/vqa/benchmarks/vqa-submission-canary-20260810-a1
uv run python scripts/run_vqa_submission.py status \
  --prepared data/vqa/benchmarks/vqa-submission-canary-20260810-a1
```

Expected: tests pass; status reports six prepared, zero inferred, zero labels loaded.

- [ ] **Step 5: Run paid-provider preflight without uploading media**

Run: `uv run --extra gemini python scripts/run_vqa_submission.py preflight-live --model-id gemini-3.1-pro-preview`
Expected with configured authentication: exit 0 and safe JSON reporting authentication configured.
Expected without configured authentication: exit 2 and the single fixed authentication message; no media upload and no credential output.

- [ ] **Step 6: Commit frozen manifests and tests**

Commit: `git add configs/vqa/vqa_submission_canary_6.json configs/vqa/vqa_submission_gate_60.json tests/test_run_vqa_submission_script.py && git commit -m "test: freeze VQA submission canary and gate"`

---

### Task 11: Add the OpenRouter transport for the paid Gemini canary

**Files:**
- Create: `src/trec_video_2026/vqa_submission_openrouter.py`
- Create: `tests/test_vqa_submission_openrouter.py`
- Modify: `scripts/run_vqa_submission.py`
- Modify: `tests/test_run_vqa_submission_script.py`

**Interfaces:**
- Consumes: `GeminiTransport`, `UploadedFileRecord`, `GenerationResult`, `TokenUsage`, and `GeminiSubmissionClient` from `vqa_submission_gemini`.
- Produces: `OpenRouterTransport`, `create_openrouter_transport`, and `preflight_openrouter_transport`; CLI support for `run --transport openrouter` and `preflight-live --provider openrouter`.

- [ ] **Step 1: Write failing transport and CLI tests**

Use an injected fake URL opener and no network. Tests must cover model mapping, authenticated preflight, base64 multimodal content conversion, strict structured-output requests, deterministic seed, privacy routing, token usage parsing, in-memory deletion, missing authentication, malformed provider output, and secret-safe HTTP failures. CLI tests cover the new provider choices without reading a real credential.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_vqa_submission_openrouter.py tests/test_run_vqa_submission_script.py -q`
Expected: failures name the missing OpenRouter module and CLI choices.

- [ ] **Step 3: Implement the thin standard-library transport**

Map `gemini-3.1-pro-preview` to `google/gemini-3.1-pro-preview`. The frozen source clips are Matroska, which OpenRouter does not document as a supported video-input MIME type. Before the first live model request for a row, stream-copy its visual stream to a durable `provider-video.webm` under that row's ignored evidence directory, excluding audio because the complete extracted WAV is sent separately. Never relabel Matroska bytes as WebM. Atomically preserve the provider-video artifact and bind its SHA-256 plus the original source SHA-256 into request/evidence provenance; validate and reuse the exact artifact on resume. `upload` validates an allowed regular media file and retains bytes only in memory under an opaque hash-derived ID; it never persists a data URI. Convert uploaded parts to OpenRouter chat content: video as `video_url`, images as `image_url`, audio as `input_audio`, and prompts as `text`. POST non-streaming chat completions with strict JSON Schema output, high reasoning excluded from the response, a deterministic request-derived seed, bounded output tokens, and provider routing `{zdr: true, data_collection: "deny", require_parameters: true}`. Parse exactly one successful JSON message plus token usage into `GenerationResult`. Errors never expose credentials, headers, provider bodies, data URIs, or client representations.

`create_openrouter_transport()` reads only `OPENROUTER_API_KEY` from the process environment. `preflight_openrouter_transport()` performs an authenticated model-catalog GET without prompts or media and returns only authentication status, model ID, input modalities, and structured-output support.

- [ ] **Step 4: Wire the existing reviewed client and CLI**

`run --transport openrouter` wraps `OpenRouterTransport` in the existing `GeminiSubmissionClient`; direct `gemini` and `fixture` behavior remains unchanged. `preflight-live --provider openrouter` runs only the no-media catalog check and emits bounded safe JSON.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_vqa_submission_openrouter.py tests/test_run_vqa_submission_script.py tests/test_vqa_submission_gemini.py -q`
Expected: all pass without network access.
Commit: `git add src/trec_video_2026/vqa_submission_openrouter.py tests/test_vqa_submission_openrouter.py scripts/run_vqa_submission.py tests/test_run_vqa_submission_script.py && git commit -m "feat: add OpenRouter VQA transport"`

---

### Task 8: Execute and preserve the six-row canary

**Files:**
- Runtime only: `data/vqa/benchmarks/vqa-submission-canary-20260810-a1/`
- Create after successful inference: `docs/experiments/2026-08-10-vqa-submission-canary.qmd`
- Create after successful inference: `docs/experiments/artifacts/vqa-submission-canary-20260810/`

**Interfaces:**
- Consumes: configured paid OpenRouter authentication, prepared canary bundle, and all earlier tasks.
- Produces: six-row timelines, evidence, ten candidates per row, rankings, frozen predictions, terminal status, sanitized artifact projection, and QMD evidence record.

- [ ] **Step 1: Run one row with an explicit spend stop**

Run:

```bash
uv run python scripts/run_vqa_submission.py run \
  --prepared data/vqa/benchmarks/vqa-submission-canary-20260810-a1 \
  --transport openrouter --limit 1 --max-provider-usd 2
```

Expected: one row completes, every uploaded provider file records deletion, ten unique candidates and one ranking persist, and no labels are loaded.

- [ ] **Step 2: Inspect the one-row operational artifacts**

Run: `uv run python scripts/run_vqa_submission.py status --prepared data/vqa/benchmarks/vqa-submission-canary-20260810-a1`
Expected: one complete row, five pending rows, zero integrity failures, finite usage/cost.

- [ ] **Step 3: Resume the remaining five rows**

Run:

```bash
uv run python scripts/run_vqa_submission.py run \
  --prepared data/vqa/benchmarks/vqa-submission-canary-20260810-a1 \
  --transport openrouter --max-provider-usd 10
uv run python scripts/run_vqa_submission.py freeze \
  --prepared data/vqa/benchmarks/vqa-submission-canary-20260810-a1
```

Expected: six final answers; at least five without Qwen fallback; valid freeze; no undeleted provider file; all requested/actual timestamps and hashes present.

- [ ] **Step 4: Build the post-freeze audit and sanitized evidence record**

Run the scorer against the six development references only after freeze. Create the dated QMD under the routing theme and a tracked artifact containing protocol/model identity, completion, aggregate operational counts, safe metrics, hashes, and private evidence marker. Do not track raw answers, references, transcripts, media, or private paths.

- [ ] **Step 5: Verify report and commit the evidence record**

Run:

```bash
uv run pytest tests/test_vqa_submission*.py tests/test_run_vqa_submission_script.py tests/test_score_vqa_submission.py -q
quarto render docs/experiments/2026-08-10-vqa-submission-canary.qmd
git diff --check
git status --short
```

Expected: all new tests pass; QMD renders; no private artifact is staged. Commit only the QMD source and sanitized tracked bundle with message `docs: record hierarchical VQA submission canary`.

---

### Task 9: Run the frozen 60-row quality gate and make the promotion decision

**Files:**
- Runtime only: `data/vqa/benchmarks/vqa-submission-gate60-20260810-a1/`
- Create: `docs/experiments/2026-08-10-vqa-submission-gate60.qmd`
- Create: `docs/experiments/artifacts/vqa-submission-gate60-20260810/`

**Interfaces:**
- Consumes: validator-passing canary, `configs/vqa/vqa_submission_gate_60.json`, frozen Qwen3.5-4B plus Whisper baseline, and Task 6 scorer.
- Produces: 60-row frozen ranking, representative/diagnostic metrics kept separate, deep failure classifications, and one predeclared `promote`, `expand-120`, or `stop` decision.

- [ ] **Step 1: Prepare the immutable 60-row run**

Run:

```bash
uv run python scripts/run_vqa_submission.py prepare \
  --config configs/vqa/vqa_submission_gate_60.json \
  --dataset-root data \
  --output data/vqa/benchmarks/vqa-submission-gate60-20260810-a1
```

Expected: 40 representative and 20 diagnostic rows, unique IDs, canonical source/evidence hashes, zero label-bearing fields.

- [ ] **Step 2: Execute resumably under the measured canary cost projection**

Run:

```bash
uv run --extra gemini python scripts/run_vqa_submission.py run \
  --prepared data/vqa/benchmarks/vqa-submission-gate60-20260810-a1 \
  --transport gemini --max-provider-usd 75
uv run python scripts/run_vqa_submission.py freeze \
  --prepared data/vqa/benchmarks/vqa-submission-gate60-20260810-a1
```

Expected: 60 ranked predictions and a valid freeze; at least 59 frontier-complete rows; all 60 have a declared final fallback; no undeleted provider files.

- [ ] **Step 3: Score only after freeze and keep populations separate**

Run `scripts/score_vqa_submission.py score` with the frozen predictions, freeze record, local VQA references, baseline artifact, and output directory under the ignored run root. Report the 40-row representative effect separately from all diagnostic failure tables. Generate seeded paired bootstrap intervals and component metrics.

- [ ] **Step 4: Deep-classify every required failure row**

Run the post-freeze Qwen3-Omni adjudicator over all 20 diagnostic rows and every representative hard failure, rescue, regression, large semantic delta, metric disagreement, transcript issue, PTS/FPS anomaly, or suspected localization miss. Preserve uncertain classifications and inspect only adjudicator disagreements manually.

- [ ] **Step 5: Apply the frozen decision table**

- `promote`: representative `ΔC >= +0.020`, positive METEOR and BERTScore-F1, STS `>= -0.005`, at least 59 frontier-complete rows, and no integrity failure.
- `expand-120`: `+0.010 <= ΔC < +0.020` with no integrity failure; freeze 80 additional representative rows before rerunning the decision.
- `stop`: `ΔC < +0.010`, integrity failure, or missing validator-safe fallback.

Write the machine-readable decision into the private audit and a sanitized decision record into the tracked bundle.

- [ ] **Step 6: Render, verify, and commit the evidence record**

Run the new targeted tests, render the QMD, verify all artifact hashes, scan recursively for private fields/paths, run `git diff --check`, and commit only the QMD plus sanitized artifact projection with message `docs: record hierarchical VQA gate results`.

---

### Task 10: Scale the promoted stack and build deadline-safe submission files

**Files:**
- Runtime only: full development and unlabeled 2026 run roots under `data/vqa/benchmarks/`.
- Create after completion: dated full-development and submission-operation QMD evidence records.
- Create after completion: sanitized tracked artifact projections under `docs/experiments/artifacts/`.

**Interfaces:**
- Consumes: `promote` decision from Task 9, complete 475-row development manifest, unlabeled 2026 AG topics/media, and official 2026 validator checkout.
- Produces: full-development confirmation, frozen production protocol, validator-passing top-1/top-5/top-10 submission portfolio, hashes, and receipts.

- [ ] **Step 1: Run unchanged on all 475 development videos**

Prepare, execute, and freeze the 475-row run with identical model ID, prompts, evidence budgets, pass IDs, verifier schema, fallback, and scorer revisions. Any protocol change creates a new run ID and invalidates direct confirmation.

- [ ] **Step 2: Confirm the full-development result**

Score all 475 rows with failure-zero accounting. Require a positive full-panel semantic-composite delta, positive METEOR and BERTScore-F1 point deltas, STS no worse than `-0.005`, complete fallback coverage, and no integrity failure. If confirmation fails, retain the frozen baseline submission and do not use test topics to tune the protocol.

- [ ] **Step 3: Freeze the production protocol before test inference**

Write a production manifest binding the development confirmation hashes, exact prompts/schemas, provider model ID, optional SDK version, evidence policy, verifier seed, pass order, fallback hashes, and official topics/media manifest. The 2026 test run remains label-free.

- [ ] **Step 4: Run 2026 AG inference resumably and freeze**

Use measured development throughput to choose bounded parallel workers and provider cost. Checkpoint each row, delete uploaded provider files after preservation, retain the Qwen fallback, and freeze every ranked prediction before serialization.

- [ ] **Step 5: Generate and validate the submission portfolio**

Generate `submission-top1.json`, `submission-top5.json`, and `submission-top10.json` from the same frozen ranking. Run the official validator in a clean isolated directory. Reject files with missing IDs, duplicate normalized answers, non-contiguous ranks, non-finite time, wrong field names, or nonzero validator exit status.

- [ ] **Step 6: Preserve exact submission artifacts and operational evidence**

Record each file SHA-256, validator report, protocol/freeze hash, generation timing/cost, completion/fallback counts, and the private immutable run root. Commit only sanitized hashes and operational summaries. Do not upload the run to Evalbase without the user's explicit submission authorization.
