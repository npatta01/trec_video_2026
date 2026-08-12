# Frontier Video Event/State Map Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the frozen 35-video, two-provider diagnostic that tests whether one question-aware native-AV pass can produce a faithful timestamped event/state map and a well-supported top-1 VQA answer.

**Architecture:** Add a small provider-neutral event-map contract, isolated OpenRouter and Reka adapters, and a resumable immutable runner rather than extending the old ten-pass submission orchestrator. Keep inference and references in separate phases: freeze all provider outputs, construct and freeze a two-view visual audit, then reveal annotations for semantic scoring and failure diagnosis. Raw media, responses, answers, and audits stay under the canonical ignored data root; only a sanitized evidence projection and QMD result enter Git.

**Tech Stack:** Python 3.12, standard-library `urllib`, dataclasses and JSON Schema, ffprobe/ffmpeg, pytest, OpenRouter Gemini 3.1 Pro Preview, Reka Core Chat API, existing NLTK/BERTScore/UMBC STS scoring, Quarto.

## Global Constraints

- The governing design is `docs/superpowers/specs/2026-08-10-frontier-video-map-diagnostic-design.md`.
- Use exactly 35 evaluation videos: five each for temporal, counting, causal, speech, non-speech sound, OCR/detail, and other/global questions; use two additional out-of-panel videos for preflight.
- Require every selected clip, including preflight, to be at most 30.0 seconds so both providers use their documented short-video path.
- Send the original local video with native audio and the question. Do not add extracted frames, Whisper, OCR, tracking, or independent audio evidence to provider inference.
- Make exactly one content inference per provider per video. Invalid JSON is an operational failure; there is no prompt repair, JSON repair, best-of-N, candidate voting, or content retry.
- An explicit identical transport retry is allowed only after a terminal transport or rate-limit failure, and every attempt remains in the immutable ledger.
- Gemini `google/gemini-3.1-pro-preview` through OpenRouter is primary. Reka `reka-core` is an independent comparator, never a repair, ensemble member, fallback, or selector.
- Freeze the prompt, response schema, model IDs, route, reasoning/sampling settings, token limit, timeouts, source hashes, and request hashes.
- Labels and reference answers remain inaccessible until provider outputs and both audit views are frozen.
- Gemini passes the architecture gate only at 28/35 usable maps or better with no category below 3/5. A correct answer cannot rescue an unusable map.
- Report normalized exact match, token F1, METEOR, BERTScore-F1, STS, their three-metric semantic composite, paired wins/losses/ties, and degenerate single-answer NDCG.
- Use failure-zero scoring across the complete 35-row denominator. Also report completed-case metrics when a provider has operational failures; never hide failures by scoring only successful rows.
- Treat provider audio claims as provider-observed. Do not claim independent verification of speech or non-speech sound.
- Stop for review if preflight projects more than $15 total. Never spend more than $30 without new authorization.
- Credentials are read only from `OPENROUTER_API_KEY` and `REKA_API_KEY`. Never put credentials in CLI arguments, files, source, logs, request artifacts, process listings, shell history, Git, or delegated context.
- Verify that `data` resolves to `/home/npatta01/data/competitions/trec_video_2026/data` before any dataset or live-run command. Never copy, move, delete, stage, or commit dataset contents.
- Keep private run artifacts under `data/vqa/benchmarks/frontier-video-map-20260811/`. Tracked artifacts must exclude media, references, raw answers, raw responses, transcripts, credentials, private paths, and private URLs.
- Reuse and hash-check the existing Qwen baseline at `data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-whisper.json`; do not rerun it.
- Do not call a paid provider from pytest, fixture tests, import-time code, or a delegated worker. Live preflight and panel execution remain primary-agent actions.

## File Structure

- `src/trec_video_2026/vqa_frontier_map.py` — immutable event/state-map types, prompt, JSON Schema, local validation, masking, and usable-map gate calculations.
- `src/trec_video_2026/vqa_frontier_map_panel.py` — safe question-only category rules, panel manifest validation, media identity binding, and category/topology checks.
- `src/trec_video_2026/vqa_frontier_map_reka.py` — credential-injected, bounded Reka Chat video transport with no content retry.
- `src/trec_video_2026/vqa_frontier_map_runner.py` — prepared-run contract, request hashing, attempt ledger, cost accounting, resume/freeze enforcement, and provider-neutral orchestration.
- `scripts/build_vqa_frontier_map_panel.py` — candidate sheet, reviewed selection, and frozen label-free panel CLI.
- `scripts/run_vqa_frontier_map.py` — thin `prepare`, `preflight`, `project-cost`, `run`, `status`, and `freeze` CLI.
- `scripts/build_vqa_frontier_map_audit.py` — post-inference visual packet, masked/question-aware views, verdict validation, and audit freeze CLI.
- `scripts/score_vqa_frontier_map.py` — post-audit reference reveal, baseline join, three-condition metrics, gate/failure summary, and sanitized projection.
- `tests/test_vqa_frontier_map.py`, `tests/test_vqa_frontier_map_panel.py`, `tests/test_vqa_frontier_map_reka.py`, `tests/test_vqa_frontier_map_runner.py`, `tests/test_build_vqa_frontier_map_audit.py`, and `tests/test_score_vqa_frontier_map.py` — offline contract and end-to-end coverage.
- `configs/vqa/frontier_video_map_panel_35.json` — frozen safe 35-row panel plus two preflight rows.
- `docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd` and `docs/experiments/artifacts/frontier-video-map-20260811/` — canonical report and sanitized evidence-of-record.

---

### Task 1: Define the one-pass event/state-map contract

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_map.py`
- Create: `tests/test_vqa_frontier_map.py`

**Interfaces:**
- Consumes: `trec_video_2026.vqa_submission.canonical_sha256` and the approved response fields in the design spec.
- Produces: `EVENT_MAP_SCHEMA`, `EVENT_MAP_PROMPT_VERSION`, `EventRecord`, `SupportSpan`, `FrontierMap`, `build_event_map_prompt(question, duration_seconds)`, `validate_event_map(payload, duration_seconds)`, `mask_question_fields(map_payload)`, `audit_pre_reference_payload(payload)`, and `summarize_map_gate(verdict_rows)`.

- [ ] **Step 1: Write failing schema and validation tests**

Add a complete valid fixture and tests that require chronological bounded events, event-linked support, one top-1 answer, uncertainty, and no reference-like fields:

```python
def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_interpretation": {
            "meaning": "Identify what changes after the person opens the box.",
            "required_modalities": ["visual"],
            "required_evidence": "The box before and after opening.",
        },
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "keyframe_seconds": [1.0, 3.5],
                "scene": "A closed box is on a table.",
                "entities": ["person", "box", "table"],
                "actions": ["person approaches box"],
                "relations": ["box on table"],
                "state_before": "box closed",
                "state_change": "person opens box",
                "state_after": "box open",
                "visual_observations": ["lid moves from closed to open"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "question_relevance": "direct",
                "confidence": 0.95,
                "gaps": [],
            }
        ],
        "support_spans": [
            {
                "event_ids": ["E1"],
                "start_seconds": 2.5,
                "end_seconds": 4.0,
                "modalities": ["visual"],
                "claim": "The box is open after the action.",
            }
        ],
        "top1_answer": "The box is open.",
        "support_explanation": "E1 shows the lid opening and the final open state.",
        "could_be_wrong_because": ["The object inside the box is not clearly visible."],
    }


def test_validate_event_map_accepts_complete_bounded_map() -> None:
    result = validate_event_map(valid_payload(), duration_seconds=4.0)
    assert result.top1_answer == "The box is open."
    assert result.support_spans[0].event_ids == ("E1",)


@pytest.mark.parametrize("mutation", ["overlap", "out_of_bounds", "unknown_event", "reference_key"])
def test_validate_event_map_rejects_contract_violations(mutation: str) -> None:
    payload = valid_payload()
    mutate_invalid(payload, mutation)
    with pytest.raises((TypeError, ValueError)):
        validate_event_map(payload, duration_seconds=4.0)
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
uv run pytest tests/test_vqa_frontier_map.py -q
```

Expected: collection fails because `trec_video_2026.vqa_frontier_map` does not exist.

- [ ] **Step 3: Implement immutable contracts, prompt, schema, and masking**

Implement frozen dataclasses and these exact public signatures:

```python
EVENT_MAP_PROMPT_VERSION = "frontier-event-map-v1"
MAX_EVENTS = 12
MAX_SUPPORT_SPANS = 3


@dataclass(frozen=True, slots=True)
class FrontierMap:
    schema_version: int
    question_interpretation: Mapping[str, object]
    events: tuple[EventRecord, ...]
    support_spans: tuple[SupportSpan, ...]
    top1_answer: str
    support_explanation: str
    could_be_wrong_because: tuple[str, ...]
```

`EVENT_MAP_SCHEMA` must set `additionalProperties: false` at every object level, cap arrays, require every listed field, and use numeric bounds of zero or greater. Local validation must additionally enforce event order, no overlapping event IDs, timestamps within the probed duration, support spans contained within their cited events, unique event IDs, and non-empty evidence/answer text. `mask_question_fields` must retain only event chronology and observational fields; it must remove question interpretation, relevance, support spans, answer, explanation, and uncertainty alternatives.

- [ ] **Step 4: Run tests and static checks**

Run:

```bash
uv run pytest tests/test_vqa_frontier_map.py -q
uv run ruff check src/trec_video_2026/vqa_frontier_map.py tests/test_vqa_frontier_map.py
```

Expected: all focused tests pass and Ruff reports no findings.

- [ ] **Step 5: Commit the contract**

```bash
git add src/trec_video_2026/vqa_frontier_map.py tests/test_vqa_frontier_map.py
git commit -m "feat: define frontier video map contract"
```

### Task 2: Build and freeze the label-free 35+2 panel

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_map_panel.py`
- Create: `scripts/build_vqa_frontier_map_panel.py`
- Create: `tests/test_vqa_frontier_map_panel.py`
- Create after question-only review: `configs/vqa/frontier_video_map_panel_35.json`

**Interfaces:**
- Consumes: local VQA-2025 annotation filenames and `question` only, `vqa_benchmark.probe_media`, `vqa_benchmark.source_fingerprint`, and original media under `data/vqa/2025/training/videos_original/`.
- Produces: `classify_question(question)`, `build_candidate_sheet(data_root)`, `validate_panel_manifest(payload, data_root)`, a private candidate sheet, and one tracked frozen label-free manifest with hashes and no references.

- [ ] **Step 1: Write failing panel tests**

Cover word-boundary classification, exact quotas, distinct preflight rows, duration/media hashes, topology diversity, and recursive label-field rejection:

```python
def test_sound_rule_does_not_match_heart_substring() -> None:
    assert classify_question("What is the heart made of?") == "other_global"
    assert classify_question("What sound is heard after the door closes?") == "sound"


def test_panel_requires_five_rows_per_category_and_two_preflight(tmp_path: Path) -> None:
    payload = make_panel_fixture(tmp_path)
    panel = validate_panel_manifest(payload, tmp_path)
    assert len(panel.evaluation_rows) == 35
    assert len(panel.preflight_rows) == 2
    assert Counter(row.category for row in panel.evaluation_rows) == {
        "temporal": 5,
        "counting": 5,
        "causal": 5,
        "speech": 5,
        "sound": 5,
        "ocr_detail": 5,
        "other_global": 5,
    }
```

- [ ] **Step 2: Run the tests and confirm failure**

```bash
uv run pytest tests/test_vqa_frontier_map_panel.py -q
```

Expected: import failure for the missing panel module.

- [ ] **Step 3: Implement the safe candidate and manifest boundary**

Implement:

```python
CATEGORIES = (
    "temporal", "counting", "causal", "speech",
    "sound", "ocr_detail", "other_global",
)
EVALUATION_PER_CATEGORY = 5
PREFLIGHT_ROWS = 2


@dataclass(frozen=True, slots=True)
class PanelRow:
    ordinal: int
    phase: str
    video_id: str
    question: str
    category: str
    evidence_topology: str
    selection_rationale: str
    media_relative_path: str
    media_sha256: str
    size_bytes: int
    duration_seconds: float
    has_audio: bool
    has_video: bool
```

The annotation loader must immediately project each file to `(video_id, question)` and must never return or serialize the remaining annotation object. Classification uses word-boundary regular expressions and precedence `counting`, `causal`, `temporal`, `speech`, `sound`, `ocr_detail`, `other_global`. The frozen manifest records a manually reviewed category and evidence topology; validation requires three distinct topology values in each category and short/typical/long duration coverage relative to that category's candidate distribution.

- [ ] **Step 4: Add the candidate and freeze CLI**

Provide commands:

```bash
uv run python scripts/build_vqa_frontier_map_panel.py candidates \
  --data-root data \
  --output data/vqa/benchmarks/frontier-video-map-20260811/panel-candidates.json

uv run python scripts/build_vqa_frontier_map_panel.py freeze \
  --data-root data \
  --selection configs/vqa/frontier_video_map_panel_35.json \
  --output data/vqa/benchmarks/frontier-video-map-20260811/frozen-panel.json
```

`candidates` writes questions, categories, durations, topological hints, and media identities but no answer/reference field. `freeze` refuses a symlinked output, validates source hashes against current bytes, checks the 35+2 contract, writes canonical JSON atomically, and refuses overwrite.

- [ ] **Step 5: Run tests and commit code**

```bash
uv run pytest tests/test_vqa_frontier_map_panel.py -q
uv run ruff check src/trec_video_2026/vqa_frontier_map_panel.py scripts/build_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_panel.py
git add src/trec_video_2026/vqa_frontier_map_panel.py scripts/build_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_panel.py
git commit -m "feat: add frontier map panel builder"
```

Do not commit the final panel config until Task 7 completes the question-only review and freeze check.

### Task 3: Harden the OpenRouter path for a single pinned Gemini call

**Files:**
- Modify: `src/trec_video_2026/vqa_submission_gemini.py`
- Modify: `src/trec_video_2026/vqa_submission_openrouter.py`
- Modify: `tests/test_vqa_submission_gemini.py`
- Modify: `tests/test_vqa_submission_openrouter.py`

**Interfaces:**
- Consumes: existing `GeminiTransport`, `GenerationResult`, bounded upload/data-URI logic, and `OPENROUTER_API_KEY`.
- Produces: optional safe provider metadata on `GenerationResult` and an injectable `OpenRouterRequestPolicy` that pins route/fallback/sampling/token behavior without changing legacy defaults.

- [ ] **Step 1: Write failing policy and metadata tests**

```python
def test_frontier_policy_pins_route_and_disables_fallback(monkeypatch, opener) -> None:
    transport = OpenRouterTransport(
        "test-secret",
        opener=opener,
        policy=OpenRouterRequestPolicy(
            max_output_tokens=16384,
            temperature=0.1,
            provider_order=("Google AI Studio",),
            allow_fallbacks=False,
        ),
    )
    result = generate_fixture_video(transport)
    body = json.loads(opener.requests[-1].data)
    assert body["max_tokens"] == 16384
    assert body["temperature"] == 0.1
    assert body["provider"]["order"] == ["Google AI Studio"]
    assert body["provider"]["allow_fallbacks"] is False
    assert result.provider_metadata["provider"] == "Google AI Studio"
```

Also retain the existing exact legacy-request tests to prove their default body is unchanged.

- [ ] **Step 2: Run tests and confirm the new test fails**

```bash
uv run pytest tests/test_vqa_submission_gemini.py tests/test_vqa_submission_openrouter.py -q
```

Expected: the new policy/metadata assertions fail while existing tests remain green.

- [ ] **Step 3: Add safe optional response metadata and request policy**

Add:

```python
@dataclass(frozen=True, slots=True)
class OpenRouterRequestPolicy:
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature: float | None = None
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: bool | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    payload: object
    raw_response: bytes | str
    usage: TokenUsage
    provider_metadata: Mapping[str, object] = field(default_factory=dict)
```

Only allow safe metadata keys `provider`, `model`, `request_id`, and `reported_cost_usd`, validate finite nonnegative cost, and freeze the mapping. Parse those fields from the OpenRouter response envelope. The frontier policy adds `temperature`, `provider.order`, and `provider.allow_fallbacks`; the default policy must serialize exactly as before. Do not add retry behavior.

- [ ] **Step 4: Verify the full existing adapter surface**

```bash
uv run pytest tests/test_vqa_submission_gemini.py tests/test_vqa_submission_openrouter.py tests/test_run_vqa_submission_script.py -q
uv run ruff check src/trec_video_2026/vqa_submission_gemini.py src/trec_video_2026/vqa_submission_openrouter.py tests/test_vqa_submission_gemini.py tests/test_vqa_submission_openrouter.py
```

Expected: legacy and new tests pass.

- [ ] **Step 5: Commit the OpenRouter extension**

```bash
git add src/trec_video_2026/vqa_submission_gemini.py src/trec_video_2026/vqa_submission_openrouter.py tests/test_vqa_submission_gemini.py tests/test_vqa_submission_openrouter.py
git commit -m "feat: pin frontier OpenRouter requests"
```

### Task 4: Add the isolated Reka Core short-video transport

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_map_reka.py`
- Create: `tests/test_vqa_frontier_map_reka.py`

**Interfaces:**
- Consumes: `GeminiTransport`, `GenerationResult`, `UploadedFileRecord`, `EVENT_MAP_SCHEMA`, `REKA_API_KEY`, and original local video bytes.
- Produces: `RekaTransport`, `create_reka_transport()`, and `preflight_reka_transport()` with one bounded HTTP request per generation and no automatic retry.

- [ ] **Step 1: Write failing transport tests with a fake opener**

Test exact endpoint, header redaction, local data URI, prompt/schema identity, `reka-core`, greedy decoding, token usage, bounded response, safe errors, and no network call when credentials are absent:

```python
def test_reka_posts_one_native_video_request_and_parses_usage(tmp_path, opener) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-bytes")
    transport = RekaTransport("test-secret", opener=opener)
    upload = transport.upload(video, mime_type="video/mp4")
    result = transport.generate(
        model="reka-core",
        parts=[{"file_id": upload.file_id, "mime_type": "video/mp4"}, "frozen prompt"],
        schema=EVENT_MAP_SCHEMA,
    )
    body = json.loads(opener.requests[0].data)
    assert body["model"] == "reka-core"
    assert body["temperature"] == 0.0
    assert body["seed"] == deterministic_seed(body, exclude=("seed",))
    assert body["messages"][0]["content"][0]["type"] == "video_url"
    assert body["messages"][0]["content"][0]["video_url"].startswith("data:video/mp4;base64,")
    assert result.usage.output_tokens == 321
    assert len(opener.requests) == 1
```

- [ ] **Step 2: Run the tests and confirm import failure**

```bash
uv run pytest tests/test_vqa_frontier_map_reka.py -q
```

- [ ] **Step 3: Implement the bounded standard-library adapter**

Use:

```python
REKA_MODEL_ID = "reka-core"
REKA_CHAT_URL = "https://api.reka.ai/v1/chat/completions"
REKA_MODELS_URL = "https://api.reka.ai/v1/models"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_TOKENS = 16384
HTTP_TIMEOUT_SECONDS = 180.0
```

The request content order is video then text. Because Reka Chat does not declare native JSON-Schema enforcement, append the canonical compact `EVENT_MAP_SCHEMA` to the frozen prompt and validate the returned JSON locally; do not switch to Reka Research or add an assistant repair pass. Use `temperature: 0.0`, deterministic seed, `stream: false`, and one choice with `finish_reason: stop`. Read only `REKA_API_KEY`; `__repr__` and all raised errors must be credential-free and bounded.

- [ ] **Step 4: Run focused tests and commit**

```bash
uv run pytest tests/test_vqa_frontier_map_reka.py -q
uv run ruff check src/trec_video_2026/vqa_frontier_map_reka.py tests/test_vqa_frontier_map_reka.py
git add src/trec_video_2026/vqa_frontier_map_reka.py tests/test_vqa_frontier_map_reka.py
git commit -m "feat: add Reka video map transport"
```

### Task 5: Implement the immutable resumable runner and cost gate

**Files:**
- Create: `src/trec_video_2026/vqa_frontier_map_runner.py`
- Create: `scripts/run_vqa_frontier_map.py`
- Create: `tests/test_vqa_frontier_map_runner.py`

**Interfaces:**
- Consumes: frozen panel, event-map contract, provider transports, `freeze_json_artifact`, media hashes, and provider usage.
- Produces: `PreparedFrontierRun`, `prepare_run`, `run_phase`, `project_cost`, `freeze_provider_outputs`, `status`, a request/attempt/result ledger, and the operational CLI.

- [ ] **Step 1: Write failing fixture-run and resume tests**

Cover exact request count, no repair, request/hash binding, started-at checkpoints, completed-row reuse, fail-closed ambiguous attempts, provider separation, route pinning, cost thresholds, freeze prerequisites, and label scans:

```python
def test_fixture_run_makes_one_call_per_provider_row_and_freezes(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=2, preflight_rows=1)
    gemini = FixtureTransport(provider="openrouter")
    reka = FixtureTransport(provider="reka")
    run_phase(prepared, phase="preflight", provider="gemini", transport=gemini)
    run_phase(prepared, phase="preflight", provider="reka", transport=reka)
    run_phase(prepared, phase="panel", provider="gemini", transport=gemini)
    run_phase(prepared, phase="panel", provider="reka", transport=reka)
    freeze = freeze_provider_outputs(prepared)
    assert gemini.call_count == 3
    assert reka.call_count == 3
    assert freeze["bindings"]["provider_result_count"] == 4


def test_invalid_json_is_terminal_without_content_retry(tmp_path: Path) -> None:
    prepared = prepare_fixture_run(tmp_path, evaluation_rows=1, preflight_rows=0)
    transport = FixtureTransport(raw_response="not-json")
    summary = run_phase(prepared, phase="panel", provider="gemini", transport=transport)
    assert summary.failed == 1
    assert transport.call_count == 1
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
uv run pytest tests/test_vqa_frontier_map_runner.py -q
```

- [ ] **Step 3: Implement prepared-run, attempt, and result contracts**

Use these public types:

```python
@dataclass(frozen=True, slots=True)
class PreparedFrontierRun:
    path: Path
    run_id: str
    panel_sha256: str
    prompt_sha256: str
    schema_sha256: str
    examples: tuple[PanelRow, ...]
    provider_configs: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class FrontierRunSummary:
    provider: str
    phase: str
    expected: int
    passed: int
    failed: int
    skipped: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
```

Write `run-manifest.json`, `requests.jsonl`, `attempts.jsonl`, `results/{provider}.jsonl`, `raw/{provider}/{video_id}.json`, `terminal-status.json`, and `freeze.json`. Request artifacts store only canonical prompt/schema hashes and media identity, never data URIs or local absolute paths. Append and `fsync` a `started` attempt before network I/O and a terminal attempt after it. Resume only a completed result whose request, source, model, route, prompt, and schema hashes all match. A lone `started` record fails closed until an explicit operator classification; it must not rebill automatically.

- [ ] **Step 4: Implement deterministic cost accounting**

Pin the price snapshot dated `2026-08-11`:

```python
PRICE_SNAPSHOT = {
    "gemini_input_per_million": 2.0,
    "gemini_output_per_million": 12.0,
    "reka_input_per_million": 2.0,
    "reka_output_per_million": 6.0,
    "reka_video_per_minute": 0.08,
    "review_threshold_usd": 15.0,
    "absolute_cap_usd": 30.0,
}
```

Prefer provider-reported OpenRouter cost when present; otherwise calculate from tokens. Reka adds source video minutes to token charges. `project_cost` scales observed preflight cost by provider-specific video duration and row count, reports low/observed/high assumptions, and emits `authorized_to_continue: false` above $15. `run --phase panel` refuses to start without a complete four-call preflight and a projection at or below $15. Every call checks cumulative estimated spend against $30 before dispatch.

- [ ] **Step 5: Add the thin CLI**

The CLI must expose:

```text
prepare --panel PATH --output PATH
preflight --prepared PATH --provider {gemini,reka,fixture}
project-cost --prepared PATH
run --prepared PATH --provider {gemini,reka,fixture}
status --prepared PATH
freeze --prepared PATH
```

`gemini` constructs `OpenRouterTransport` with the frontier policy and pins the provider route observed in the successful preflight result. `reka` constructs `RekaTransport`. `fixture` is permitted only for fixture manifests. CLI output is a compact safe JSON summary and never includes prompts, questions, answers, media paths, raw responses, or provider error bodies.

- [ ] **Step 6: Verify runner behavior and commit**

```bash
uv run pytest tests/test_vqa_frontier_map_runner.py tests/test_vqa_frontier_map.py tests/test_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_reka.py -q
uv run ruff check src/trec_video_2026/vqa_frontier_map_runner.py scripts/run_vqa_frontier_map.py tests/test_vqa_frontier_map_runner.py
git add src/trec_video_2026/vqa_frontier_map_runner.py scripts/run_vqa_frontier_map.py tests/test_vqa_frontier_map_runner.py
git commit -m "feat: add resumable frontier map runner"
```

### Task 6: Build the blinded visual-audit and post-freeze scoring tools

**Files:**
- Create: `scripts/build_vqa_frontier_map_audit.py`
- Create: `scripts/score_vqa_frontier_map.py`
- Create: `tests/test_build_vqa_frontier_map_audit.py`
- Create: `tests/test_score_vqa_frontier_map.py`

**Interfaces:**
- Consumes: frozen provider results, original media, panel manifest, audit verdicts, VQA annotations only after audit freeze, the frozen Qwen baseline, audit/freeze patterns from `vqa_frozen30.py` and `relevant_scene_oracle.py`, and the unchanged `_score_unique_pairs`, `_aggregate_rows`, `_bootstrap_intervals`, and baseline-binding patterns in `scripts/score_vqa_factorial.py`.
- Produces: private visual packets and verdict freeze, three-condition metrics, map gate, seven-way failure counts, pairwise comparisons, and a sanitized projection.

- [ ] **Step 1: Write failing audit-boundary tests**

```python
def test_blind_packet_masks_question_answer_support_and_relevance(tmp_path: Path) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")
    blind = json.loads(packet.blind_manifest.read_text())
    text = json.dumps(blind).lower()
    for forbidden in ("question", "top1_answer", "support_span", "relevance", "could_be_wrong"):
        assert forbidden not in text


def test_audit_freeze_requires_two_views_for_every_provider_row(tmp_path: Path) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=35, providers=("gemini", "reka"))
    freeze = freeze_audit(audit)
    assert freeze["bindings"]["blind_verdicts"] == 70
    assert freeze["bindings"]["aware_verdicts"] == 70
```

- [ ] **Step 2: Implement private visual packets and verdict validation**

For each provider/video, create:

```text
audit/rows/{provider}/{video_id}/blind.json
audit/rows/{provider}/{video_id}/aware.json
audit/rows/{provider}/{video_id}/overview.jpg
audit/rows/{provider}/{video_id}/event-keyframes/*.jpg
audit/verdicts.blind.jsonl
audit/verdicts.aware.jsonl
audit/freeze.json
```

The blind manifest uses `mask_question_fields`; the aware manifest adds question, support spans, answer, and explanation but no reference. ffmpeg-derived images are audit-only and never enter provider requests. Reject symlinks and stale source hashes. Blind verdicts cover event coverage, order, entities, actions, state changes, visible text, timestamp boundary error, answer-changing false claims, and notes. Aware verdicts cover question understanding, modality choice, evidence presence/localization, support citation, usable-map decision, and confidence. Audio fields permit only `provider_reported`, `not_claimed`, or `unverified`.

For an operationally failed provider row, still create the overview and two manifests, mark the map payload absent with the frozen failure code, and require an `unusable` verdict. Do not fabricate events, support, or an answer.

- [ ] **Step 3: Write failing scoring and baseline-integrity tests**

```python
def test_score_requires_prediction_and_audit_freezes_before_annotations(tmp_path: Path) -> None:
    fixture = make_scoring_fixture(tmp_path)
    fixture.audit_freeze.unlink()
    with pytest.raises(ValueError, match="audit freeze"):
        score_frontier_map(**fixture.kwargs)


def test_sanitized_projection_has_all_metrics_and_no_private_text(tmp_path: Path) -> None:
    result = score_frontier_map(**make_scoring_fixture(tmp_path).kwargs)
    assert set(result["conditions"]) == {"gemini", "reka", "qwen_baseline"}
    for metric in ("normalized_exact_match", "token_f1", "meteor", "bertscore_f1", "sts", "semantic_composite"):
        assert metric in result["conditions"]["gemini"]
    serialized = json.dumps(result).lower()
    for forbidden in ("correct_answer", "reference_answer", "raw_response", "media_path"):
        assert forbidden not in serialized
```

- [ ] **Step 4: Implement post-freeze scoring and failure classification**

Before reading annotations, verify provider freeze, audit freeze, panel digest, output order, source hashes, and Qwen `SHA256SUMS`. Adapt the factorial scorer's baseline-binding checks to require the exact baseline source hash, exact ordered 35-ID subset, annotation-root hash, scorer identity, and pinned BERT snapshot. Filter the existing 475-row Qwen file to those exact IDs without changing answers.

Call the unchanged factorial `_score_unique_pairs` helper once with all Qwen rows and each provider's completed rows so repeated prediction/reference pairs share METEOR, BERTScore, and STS work. Insert explicit zero values for every operationally failed provider row when calculating complete-panel metrics. Use the helper's normalized exact match, token F1, METEOR, BERTScore-F1, STS, semantic composite, and degenerate NDCG values. Build provider-versus-Qwen paired deltas, wins/losses/ties, and bootstrap intervals over all 35 rows. Report both failure-zero metrics and completed-case metrics; merge only numeric/safe row fields.

Validate one or more failure labels from this exact set:

```python
FAILURE_CAUSES = (
    "question_misunderstood",
    "wrong_modality",
    "wrong_temporal_or_local_evidence",
    "perception_failure",
    "reasoning_failure_with_correct_evidence",
    "answer_generation_or_normalization",
    "ambiguous_or_insufficient_evidence",
)
```

Report Gemini's 28/35 and category >=3/5 map gate, stricter 4/5 category counts, Reka diagnostics, answer metrics, paired wins/losses/ties, completion denominator, cost, and degenerate NDCG. Keep row-level answers/references and free-text audit notes private. The tracked projection may contain video IDs, categories, numeric metrics, booleans, failure labels, hashes, and counts.

- [ ] **Step 5: Run audit/scoring tests and commit**

```bash
uv run pytest tests/test_build_vqa_frontier_map_audit.py tests/test_score_vqa_frontier_map.py tests/test_score_vqa_factorial.py -q
uv run ruff check scripts/build_vqa_frontier_map_audit.py scripts/score_vqa_frontier_map.py tests/test_build_vqa_frontier_map_audit.py tests/test_score_vqa_frontier_map.py
git add scripts/build_vqa_frontier_map_audit.py scripts/score_vqa_frontier_map.py tests/test_build_vqa_frontier_map_audit.py tests/test_score_vqa_frontier_map.py
git commit -m "feat: add frontier map audit and scoring"
```

### Task 7: Freeze the panel and pass the offline review gate

**Files:**
- Create: `configs/vqa/frontier_video_map_panel_35.json`
- Modify only for review findings: files created or modified in Tasks 1–6

**Interfaces:**
- Consumes: candidate sheet, question-only selection criteria, all offline tests, and the approved design.
- Produces: one frozen, reviewed, source-bound panel and independently reviewed code before any paid call.

- [ ] **Step 1: Verify the canonical data link and build candidates**

```bash
test -L data
test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
uv run python scripts/build_vqa_frontier_map_panel.py candidates \
  --data-root data \
  --output data/vqa/benchmarks/frontier-video-map-20260811/panel-candidates.json
```

Expected: 475 candidates, no answer/reference fields, all media present, durations and hashes recorded.

- [ ] **Step 2: Perform and record the question-only panel review**

Select five rows per category from the candidate sheet without opening any reference or historical prediction. Within each category, include short, median, and long duration bands and at least three declared evidence topologies. Review all 35 questions for semantic category purity. If an automatic category is wrong, take the next candidate in canonical SHA-256 order that satisfies the missing duration/topology cell. Select two additional typical-duration rows not in the 35 as preflight.

Write `configs/vqa/frontier_video_map_panel_35.json` with `schema_version: 1`,
`run_id: frontier-video-map-20260811`, selection policy
`question-only-category-purity-duration-topology-v1`, reference policy
`not-read-not-serialized`, `evaluation_count: 35`, `preflight_count: 2`, and
exactly 37 ordered `examples`. Each example contains every `PanelRow` field
defined in Task 2. The final file must contain no annotation path or
answer/reference field.

- [ ] **Step 3: Freeze and verify the exact panel**

```bash
uv run python scripts/build_vqa_frontier_map_panel.py freeze \
  --data-root data \
  --selection configs/vqa/frontier_video_map_panel_35.json \
  --output data/vqa/benchmarks/frontier-video-map-20260811/frozen-panel.json
uv run python scripts/run_vqa_frontier_map.py prepare \
  --panel data/vqa/benchmarks/frontier-video-map-20260811/frozen-panel.json \
  --output data/vqa/benchmarks/frontier-video-map-20260811/run
```

Expected: exact 35+2 counts, five per category, two disjoint preflight IDs, native audio/video for every source, matching source hashes, and immutable prepared manifest.

- [ ] **Step 4: Run the complete offline verification suite and secret scan**

```bash
uv run pytest -q
uv run ruff check src scripts tests
git diff --check
rg -n -g '!data/**' -g '!uv.lock' -g '!docs/superpowers/plans/**' 'sk-or-v1-|OPENROUTER_API_KEY[[:space:]]*=|REKA_API_KEY[[:space:]]*=' .
```

Expected: all tests and Ruff pass; diff check is clean; secret scan prints nothing.

- [ ] **Step 5: Obtain an independent consequential-change review**

Use `sol_reviewer` on the branch diff with the design spec and this plan. Require review of one-call enforcement, label isolation, resume/no-rebilling behavior, provider route/model identity, media/hash integrity, secret handling, cost stops, freeze ordering, audit masking, scoring validity, and private/tracked artifact separation. Resolve all Critical and Important findings, rerun affected tests, and record the final verdict in the implementation log.

- [ ] **Step 6: Commit the frozen panel and review fixes**

```bash
git add \
  configs/vqa/frontier_video_map_panel_35.json \
  src/trec_video_2026/vqa_frontier_map.py \
  src/trec_video_2026/vqa_frontier_map_panel.py \
  src/trec_video_2026/vqa_frontier_map_reka.py \
  src/trec_video_2026/vqa_frontier_map_runner.py \
  src/trec_video_2026/vqa_submission_gemini.py \
  src/trec_video_2026/vqa_submission_openrouter.py \
  scripts/build_vqa_frontier_map_panel.py \
  scripts/run_vqa_frontier_map.py \
  scripts/build_vqa_frontier_map_audit.py \
  scripts/score_vqa_frontier_map.py \
  tests/test_vqa_frontier_map.py \
  tests/test_vqa_frontier_map_panel.py \
  tests/test_vqa_frontier_map_reka.py \
  tests/test_vqa_frontier_map_runner.py \
  tests/test_build_vqa_frontier_map_audit.py \
  tests/test_score_vqa_frontier_map.py \
  tests/test_vqa_submission_gemini.py \
  tests/test_vqa_submission_openrouter.py
git diff --cached --check
git commit -m "test: freeze frontier video map panel"
```

Confirm only intended source, tests, and the safe panel config are staged; no file under `data/` may be staged.

### Task 8: Run the two-video preflight and full 35-video panel

**Files:**
- Create outside Git: `data/vqa/benchmarks/frontier-video-map-20260811/run/**`
- Modify: no tracked files

**Interfaces:**
- Consumes: reviewed prepared run and runtime-only provider credentials.
- Produces: four preflight results, a cost projection, 70 panel results or explicit failures, immutable raw/private artifacts, and provider-output freeze.

- [ ] **Step 1: Perform credential-presence and clean-state checks without printing values**

```bash
test -n "${OPENROUTER_API_KEY:-}"
test -n "${REKA_API_KEY:-}"
test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
git status --short --branch
```

Expected: both variables are present, the data link is canonical, and only known unrelated user files are untracked or modified.

- [ ] **Step 2: Run exactly four content preflight calls**

```bash
uv run python scripts/run_vqa_frontier_map.py preflight \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --provider gemini
uv run python scripts/run_vqa_frontier_map.py preflight \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --provider reka
uv run python scripts/run_vqa_frontier_map.py status \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run
```

Expected: two terminal preflight rows per provider; exact native-video request hashes; valid or explicitly failed JSON; provider/model/route identity; token usage, latency, and cost; no content retry.

- [ ] **Step 3: Apply the cost and compatibility stop**

```bash
uv run python scripts/run_vqa_frontier_map.py project-cost \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  > data/vqa/benchmarks/frontier-video-map-20260811/run/cost-projection.json
```

Continue only if both providers ingested native video/audio, the four calls are terminal, artifacts are complete, and `authorized_to_continue` is true. Stop and report if projected cost exceeds $15, if a provider silently changes model/route, if source media differs, or if systematic audio/video ingestion is absent.

- [ ] **Step 4: Run the unchanged 35 rows once per provider**

```bash
uv run python scripts/run_vqa_frontier_map.py run \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --provider gemini
uv run python scripts/run_vqa_frontier_map.py run \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --provider reka
```

Run sequentially so cost and provider identity remain easy to audit. If the process is interrupted, inspect `status` and the attempt ledger before resuming. Do not rerun a completed request. An explicit transport retry must reuse the exact frozen request hash and preserve both attempts.

- [ ] **Step 5: Validate completion and freeze provider outputs**

```bash
uv run python scripts/run_vqa_frontier_map.py status \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run
uv run python scripts/run_vqa_frontier_map.py freeze \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run
```

Expected: 35 terminal panel records per provider, 70 total; completion/failure denominators, cost, and model identity recorded; prediction freeze verifies. Do not open annotations before this succeeds.

### Task 9: Audit every map, reveal references, score, and publish the result

**Files:**
- Create outside Git: `data/vqa/benchmarks/frontier-video-map-20260811/audit/**`
- Create outside Git: `data/vqa/benchmarks/frontier-video-map-20260811/scoring/**`
- Create: `docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd`
- Create: `docs/experiments/2026-08-11-frontier-video-map-diagnostic.md`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/README.md`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/summary.json`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/panel.json`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/cost.json`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/metrics.json`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/failure-counts.json`
- Create: `docs/experiments/artifacts/frontier-video-map-20260811/SHA256SUMS`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Modify: `docs/project-status/research-visual-evidence.md`
- Modify: `reports/project-status/research-visual-evidence.html`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify: `docs/project-status/experiment-ledger.md`
- Modify: `reports/project-status/experiment-ledger.html`
- Modify: `docs/project-status/code-map.qmd`
- Modify: `docs/project-status/code-map.md`
- Modify: `reports/project-status/code-map.html`

**Interfaces:**
- Consumes: frozen provider outputs, private visual packets, complete two-view verdicts, annotations after audit freeze, baseline predictions, and advisor feedback.
- Produces: a valid condition-bound claim, full private audit/scoring evidence, sanitized tracked evidence, updated living report, and a clear next architecture decision.

- [ ] **Step 1: Build the audit packet without labels**

```bash
uv run python scripts/build_vqa_frontier_map_audit.py build \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --output data/vqa/benchmarks/frontier-video-map-20260811/audit
```

Verify 70 blind manifests, 70 aware manifests, 70 overview sheets, event-keyframe images, source/media hashes, and no question/answer leakage in blind manifests.

- [ ] **Step 2: Complete all 140 ordered audit verdicts**

For each of 70 provider rows, inspect the original video and blind packet first and record the complete fidelity verdict. Then open the aware packet and record question/evidence/answer sufficiency. Inspect every material visual event, transition, identity, OCR claim, and cited timestamp; do not limit review to wrong answers. An operationally failed row is unusable and retains its failure code. Mark all audio observations `provider_reported` or `unverified` rather than independently correct/incorrect.

Validate and freeze:

```bash
uv run python scripts/build_vqa_frontier_map_audit.py validate \
  --audit data/vqa/benchmarks/frontier-video-map-20260811/audit
uv run python scripts/build_vqa_frontier_map_audit.py freeze \
  --audit data/vqa/benchmarks/frontier-video-map-20260811/audit
```

Expected: 70 complete blind plus 70 complete aware verdicts, immutable hashes, and no references loaded.

- [ ] **Step 3: Reveal references and score all three conditions**

```bash
uv run python scripts/score_vqa_frontier_map.py \
  --prepared data/vqa/benchmarks/frontier-video-map-20260811/run \
  --audit data/vqa/benchmarks/frontier-video-map-20260811/audit \
  --annotations data/vqa/2025/training/annotations \
  --qwen-baseline data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/predictions.frames-whisper.json \
  --bert-model /home/npatta01/.cache/huggingface/hub/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59 \
  --private-output data/vqa/benchmarks/frontier-video-map-20260811/scoring \
  --sanitized-output data/vqa/benchmarks/frontier-video-map-20260811/scoring/sanitized.json
```

Expected: exact 35-row joins for Gemini, Reka, and Qwen; all requested metrics; map gates; per-category counts; paired comparisons; failure taxonomy; completion and cost. The scorer must verify the pinned BERT snapshot directory name and content hash; do not download or silently change the model.

- [ ] **Step 4: Obtain independent interpretation criticism**

Give `sol_advisor` only the approved design, sanitized metrics/gates/failure counts, completion denominator, and cost. Ask whether the evidence supports using one-pass maps as the system foundation, which failure category dominates, what the next adaptive component should be, and what the 35-row panel cannot establish. Do not send raw media, questions, answers, references, credentials, private paths, or raw audit notes. Reconcile the advice against the row-level private evidence rather than copying it into the report uncritically.

- [ ] **Step 5: Write the canonical report and sanitized bundle**

The QMD must separate observed results from interpretation and include:

- reason and predeclared question/hypothesis/falsifier;
- 35+2 panel and selection boundary;
- exact provider/model/route/prompt/schema identities;
- completion and cost;
- Gemini primary and Reka comparator map gates by category;
- exact/token/semantic metrics versus the frozen Qwen baseline;
- all seven failure-category counts and representative condition-bound examples without raw private content;
- what the evidence rules out, what it does not rule out, decision disposition, and reconsideration condition;
- explicit unverified-audio limitation;
- links to the private evidence location by stable repo/data-relative description and to the tracked sanitized bundle.

The tracked bundle contains only the six declared JSON/README/checksum files. Generate `SHA256SUMS` from those five non-checksum files after privacy checks; never include QMD-generated HTML, raw model text, references, free-text audit notes, or media.

- [ ] **Step 6: Update the living report in one primary theme**

Add the canonical claim under Visual and temporal evidence. Update the experiment-ledger matrix/queue only if the decision changes the current next step. Add the reusable provider/map/audit/scoring boundaries to Code Map. Related themes link to the canonical visual claim rather than copying it. Preserve the superseded ten-pass design as historical provenance.

- [ ] **Step 7: Render and verify source, privacy, and links**

```bash
render_cache=$(mktemp -d /tmp/trecvid-frontier-map-render.XXXXXX)
env XDG_CACHE_HOME="$render_cache/cache" quarto render docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd --to gfm --output-dir docs/experiments --output 2026-08-11-frontier-video-map-diagnostic.md
env XDG_CACHE_HOME="$render_cache/cache" quarto render docs/project-status --to html
uv run pytest -q
uv run ruff check src scripts tests
git diff --check
rg -n 'sk-or-v1-|OPENROUTER_API_KEY[[:space:]]*=|REKA_API_KEY[[:space:]]*=|correct_answer|reference_answer|raw_response|data:video/' \
  docs/experiments/artifacts/frontier-video-map-20260811 \
  docs/project-status \
  reports/project-status \
  docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd \
  docs/experiments/2026-08-11-frontier-video-map-diagnostic.md
```

Expected: renders and tests pass; no secret/private-content scan match in new publication surfaces; repository-relative links resolve; tracked hashes verify.

- [ ] **Step 8: Commit the report and tracked evidence**

```bash
git add \
  docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd \
  docs/experiments/2026-08-11-frontier-video-map-diagnostic.md \
  docs/experiments/artifacts/frontier-video-map-20260811/README.md \
  docs/experiments/artifacts/frontier-video-map-20260811/summary.json \
  docs/experiments/artifacts/frontier-video-map-20260811/panel.json \
  docs/experiments/artifacts/frontier-video-map-20260811/cost.json \
  docs/experiments/artifacts/frontier-video-map-20260811/metrics.json \
  docs/experiments/artifacts/frontier-video-map-20260811/failure-counts.json \
  docs/experiments/artifacts/frontier-video-map-20260811/SHA256SUMS \
  docs/project-status/research-visual-evidence.qmd \
  docs/project-status/research-visual-evidence.md \
  docs/project-status/experiment-ledger.qmd \
  docs/project-status/experiment-ledger.md \
  docs/project-status/code-map.qmd \
  docs/project-status/code-map.md \
  reports/project-status/research-visual-evidence.html \
  reports/project-status/experiment-ledger.html \
  reports/project-status/code-map.html
git diff --cached --check
git commit -m "docs: report frontier video map diagnostic"
```

Confirm no path under `data/`, no raw provider output, no reference, no media, and no credential is staged. The final handoff must state outcome, map/answer gates, dominant failures, exact spend, artifact paths, verification evidence, and the next recommended architecture action.
