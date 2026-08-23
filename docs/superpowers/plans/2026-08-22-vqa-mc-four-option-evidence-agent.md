# VQA MC Four-Option Evidence Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an inference-only Gemini evidence agent that keeps all four MC choices explicit, performs at least one blinded native-video inspection per item, adversarially reviews stopping, and advances from the five known direct-control misses to the frozen 20-row regression panel only when its declared gates pass.

**Architecture:** Preserve the frozen direct Gemini ranking as a protected fallback. A fresh complete-video initializer builds an A-D ledger; a text planner chooses a discriminating native-video view; a blinded observer records literal evidence; a text updater maps evidence back to A-D; and an adversarial critic either accepts stopping or requires another round, up to three observations. A fresh complete-video judge may replace the protected answer only with cited contradictory and positive observer evidence. The runner and scorer are additive so the completed negative deliberative experiment remains immutable.

**Tech Stack:** Python 3.12+, `google-genai` 2.x, JSON Schema structured outputs, pytest 9.x, Quarto QMD reports.

**Spec:** `docs/superpowers/specs/2026-08-22-vqa-mc-four-option-evidence-agent-design.md`

## Global Constraints

- Use only `gemini-3.7-flash`; do not call Qwen and do not train or fine-tune any model.
- Use fresh Gemini contexts for initializer, planner, observer, updater, critic, and judge.
- Configure high thinking and the live model metadata's `output_token_limit` for every semantic call; never use a smaller role-specific output cap.
- Keep A, B, C, and D in every ledger and decision contract.
- Hide the question, choices, option IDs, protected ranking, provisional rankings, and option-conditioned expectations from the observer.
- Require at least one valid blinded observation and permit at most three observations per row.
- Permit one same-contract recovery for `MAX_TOKENS`, incomplete JSON, or structurally invalid output; never score such an attempt as an answer.
- Count canary, recoveries, and full-run spend under one `$1.50` cap.
- Price the formal worst case at the provider output maximum; if the combined formal bound exceeds `$1.50`, make no paid generation call.
- Keep questions, options, video IDs, labels, predictions, and raw model text in ignored private artifacts only.
- Treat the existing 20 rows as a regression/mechanism suite, never unbiased promotion evidence.
- Preserve the shared dataset at `/home/npatta01/data/competitions/trec_video_2026/data`; do not copy, move, delete, stage, or commit dataset contents.
- Edit QMD sources only; regenerate Markdown and HTML rather than editing generated report files.

## File Structure

| File | Responsibility |
|---|---|
| `src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py` | Provider-independent schemas, parsers, prompts, A-D ledger transitions, stop and replacement invariants, and notebook integrity. |
| `scripts/run_vqa_mc_gemini_four_option_agent_poc20.py` | Gemini request construction, model-capability lookup, formal cost preflight, recovery, private artifact freezing, canary/full execution, and reuse verification. |
| `scripts/score_vqa_mc_four_option_agent_poc20.py` | Label-dependent canary preparation, canary/full scoring, transition metrics, decision gates, private audit, and sanitized aggregate generation. |
| `tests/test_vqa_mc_four_option_agent.py` | Unit tests for contracts, prompts, ledger state, mandatory inspection, critic stopping, and protected-answer replacement. |
| `tests/test_vqa_mc_four_option_agent_runner.py` | Unit and fake-client integration tests for information barriers, provider-maximum output, formal cost gates, recovery, bounded rounds, artifact hashes, and canary reuse. |
| `tests/test_vqa_mc_four_option_agent_scoring.py` | Tests for private canary derivation, score integrity, continuation gates, regression metrics, and privacy-safe aggregation. |
| `docs/experiments/2026-08-21-vqa-mc-poc20.qmd` | Existing evidence-of-record report extended with the new planned condition and observed disposition. |
| `docs/experiments/2026-08-21-vqa-mc-poc20.md` | Generated GFM rendering of the evidence-of-record report. |
| `docs/project-status/research-routing.qmd` | Primary routing-theme working answer and next-test queue after the result. |
| `docs/project-status/research-routing.md` | Generated GFM rendering of the routing-theme source. |
| `reports/project-status/research-routing.html` | Generated HTML rendering of the routing-theme source. |
| `docs/experiments/artifacts/vqa-mc-poc20-20260821/README.md` | Evidence-bundle index linking the new aggregate. |
| `docs/experiments/artifacts/vqa-mc-poc20-20260821/four-option-agent-summary.json` | Tracked aggregate containing only counts, hashes, costs, metrics, integrity outcomes, and gates. |

---

### Task 1: Provider-independent four-option contracts

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py`
- Create: `tests/test_vqa_mc_four_option_agent.py`

**Interfaces:**
- Consumes: `OPTION_IDS`, `parse_ranking`, and `canonical_json` from `trec_video_2026.vqa_experiments.vqa_mc_common`.
- Produces: `MediaAction`, `INITIALIZER_SCHEMA`, `PLANNER_SCHEMA`, `OBSERVER_SCHEMA`, `UPDATER_SCHEMA`, `CRITIC_SCHEMA`, `JUDGE_SCHEMA`, `parse_initializer()`, `parse_planner()`, `parse_observer()`, `parse_update()`, `parse_critic()`, `parse_judge()`, `initializer_prompt()`, `planner_prompt()`, `observer_prompt()`, `updater_prompt()`, `critic_prompt()`, `judge_prompt()`, `initialize_notebook()`, `append_round_stage()`, `notebook_observation_ids()`, `live_option_ids()`, and `choose_final_ranking()`.

- [ ] **Step 1: Write failing tests for exact A-D ledger membership**

```python
def test_initializer_requires_exactly_one_ledger_row_for_each_option() -> None:
    parsed = parse_initializer(
        _initializer_payload(),
        duration_seconds=6.0,
        protected_ranking=list("BACD"),
    )
    assert [row["option_id"] for row in parsed["ledger"]] == list("ABCD")

    duplicate = _initializer_payload()
    duplicate["ledger"][3]["option_id"] = "A"
    with pytest.raises(ValueError, match="each option exactly once"):
        parse_initializer(
            duplicate,
            duration_seconds=6.0,
            protected_ranking=list("BACD"),
        )
```

Each fixture ledger row contains these exact keys:

```python
{
    "option_id": "A",
    "observable_claim": "Exactly one completed jump occurs.",
    "supporting_evidence_ids": [],
    "contradicting_evidence_ids": [],
    "untested_claims": ["Whether another jump occurs later."],
    "status": "viable",
    "interpretation": "One jump is currently plausible.",
}
```

- [ ] **Step 2: Run the contract test and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py::test_initializer_requires_exactly_one_ledger_row_for_each_option`

Expected: FAIL because `vqa_mc_four_option_agent` does not exist.

- [ ] **Step 3: Implement the immutable action and ledger parsers**

Create these public types and constants:

```python
OPTION_STATUSES = frozenset({"viable", "disfavored", "inconsistent"})
FOCUS_RESOLUTIONS = frozenset({"present", "absent", "unresolved"})
STOP_DECISIONS = frozenset({"continue", "accept_stop"})
MAX_OBSERVATIONS = 3

@dataclass(frozen=True)
class MediaAction:
    kind: str
    start_seconds: float
    end_seconds: float
    fps: float
    resolution: str
    neutral_request: str
    tested_option_ids: tuple[str, ...]
    expected_outcomes: tuple[dict[str, str], ...]
```

Implement strict object parsing with exact keys, finite timestamps, supported
actions (`rewatch_full`, `inspect_interval`), complete ranking permutations,
unique A-D ledger rows normalized into A-D order, valid evidence references,
and literal observations separated from interpretations.

- [ ] **Step 4: Write failing tests for mandatory inspection and adversarial stop rules**

```python
def test_critic_cannot_accept_stop_before_a_valid_blinded_observation() -> None:
    notebook = initialize_notebook("video-a", list("BACD"), _initializer())
    with pytest.raises(ValueError, match="before a valid blinded observation"):
        parse_critic(
            _critic_payload("accept_stop"),
            notebook=notebook,
            observation_count=0,
        )

def test_critic_accepts_stop_only_after_testing_every_viable_option() -> None:
    notebook = _notebook_with_observation()
    payload = _critic_payload("accept_stop")
    payload["tested_viable_option_ids"] = ["A", "B"]
    with pytest.raises(ValueError, match="all viable options"):
        parse_critic(payload, notebook=notebook, observation_count=1)

def test_third_rejected_stop_records_forced_stop() -> None:
    assert next_search_state(
        decision="continue", observation_count=3
    ) == "forced_stop"
```

- [ ] **Step 5: Run the stop-rule tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py -k 'critic or forced_stop'`

Expected: FAIL because critic parsing and bounded state transitions are missing.

- [ ] **Step 6: Implement planner, observer, updater, and critic contracts**

Use these decision shapes:

```python
planner = {
    "provisional_ranking": ["B", "A", "C", "D"],
    "unresolved_distinction": "Whether one or two completed jumps occur.",
    "action": asdict(media_action),
    "why_discriminating": "A gapless count separates A from B.",
}

critic = {
    "decision": "continue",
    "tested_viable_option_ids": ["A", "B", "C", "D"],
    "leader_evidence_ids": ["round-1-o1"],
    "unexplained_counterevidence_ids": [],
    "next_unresolved_distinction": "Whether the landing completes before the cut.",
    "direct_replacement_supported": False,
    "reason": "A wider view could still change the ranking.",
}
```

The observer schema contains `observations`, `exact_transcription`,
`focus_resolution`, and `view_limitations`; it contains no ranking or option
field. `append_round_stage()` deep-copies the notebook, refuses stage rewrites,
and assigns stable `round-{n}-o{k}` observation IDs.

- [ ] **Step 7: Write failing tests for observer blindness and protected-answer replacement**

```python
def test_observer_prompt_contains_only_neutral_request_and_timeline() -> None:
    action = _action()
    prompt = observer_prompt(action, original_time_origin_seconds=0.0)
    prohibited = [
        _row()["question"], "A.", "B.", "C.", "D.",
        "protected", "ranking", "expected_outcomes", "option_id",
    ]
    assert all(value.casefold() not in prompt.casefold() for value in prohibited)

def test_judge_replacement_requires_new_contradiction_and_positive_support() -> None:
    notebook = _notebook_with_observation(protected_ranking=list("BACD"))
    payload = _judge_payload(ranking="ABCD")
    payload["direct_contradiction_evidence_ids"] = []
    with pytest.raises(ValueError, match="contradiction evidence"):
        parse_judge(payload, notebook=notebook, protected_ranking=list("BACD"))

    payload["direct_contradiction_evidence_ids"] = ["round-1-o1"]
    payload["replacement_support_evidence_ids"] = ["round-1-o1"]
    parsed = parse_judge(
        payload, notebook=notebook, protected_ranking=list("BACD")
    )
    assert parsed["protected_answer_action"] == "replaced"
```

- [ ] **Step 8: Run the blindness/replacement tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py -k 'observer_prompt or replacement'`

Expected: FAIL until prompts and judge invariants exist.

- [ ] **Step 9: Implement role prompts, judge parsing, and final fallback**

The initializer and planner see the exact question/options. The observer sees
only `neutral_request` and the original timeline origin. The updater and critic
see the ledger and immutable observer records. The judge sees complete native
AV, question/options, protected ranking, and a notebook rendering that lists
literal observer evidence before interpretations.

`choose_final_ranking(judge, protected_ranking)` returns `(judge["ranking"],
"judge")` only for a valid judge; otherwise it returns `(protected_ranking,
"protected_direct_fallback")` and the caller marks the row operationally
invalid.

- [ ] **Step 10: Run the complete contract suite**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py`

Expected: PASS.

- [ ] **Step 11: Commit the contracts**

```bash
git add src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent.py
git commit -m "feat: add four-option VQA evidence contracts"
```

---

### Task 2: Gemini requests, information barriers, and formal cost preflight

**Files:**
- Create: `scripts/run_vqa_mc_gemini_four_option_agent_poc20.py`
- Create: `tests/test_vqa_mc_four_option_agent_runner.py`

**Interfaces:**
- Consumes: all public schemas, prompt builders, parsers, and notebook helpers from Task 1; existing request bundle and preserved WebM deliveries.
- Produces: `ModelCapabilities`, `load_requests()`, `load_protected_control()`, `get_model_capabilities()`, `build_initializer_request()`, `build_planner_request()`, `build_observer_request()`, `build_updater_request()`, `build_critic_request()`, `build_judge_request()`, `audit_observer_request()`, `estimate_formal_cost()`, `preflight()`, `run_phase()`, and `verify_output()`.

- [ ] **Step 1: Write failing request-builder tests**

```python
def test_every_role_uses_live_provider_output_maximum_and_high_thinking() -> None:
    capabilities = runner.ModelCapabilities(output_token_limit=65536)
    requests = _all_role_requests(capabilities)
    assert {request.config.max_output_tokens for request in requests} == {65536}
    assert all(
        request.config.thinking_config.thinking_level.name.endswith("HIGH")
        for request in requests
    )

def test_live_capabilities_use_model_output_token_limit() -> None:
    client = _MetadataClient(output_token_limit=65536)
    assert runner.get_model_capabilities(client).output_token_limit == 65536
    assert client.models.requested_model == runner.MODEL
```

- [ ] **Step 2: Run request tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'provider_output or live_capabilities'`

Expected: FAIL because the new runner does not exist.

- [ ] **Step 3: Implement model metadata and request construction**

Use these constants and capability type:

```python
MODEL = "gemini-3.7-flash"
MODEL_ARM = "four-option-evidence-native-av-v1"
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
MAX_OBSERVATIONS = 3
MAX_SEMANTIC_CALLS_PER_ROW = 14
MAX_PROVIDER_CALLS_PER_ROW = 28
MAX_COST_USD = 1.50
BASELINE_FPS = 4.0

@dataclass(frozen=True)
class ModelCapabilities:
    output_token_limit: int
    input_token_limit: int | None = None
    model_version: str | None = None
    supported_actions: tuple[str, ...] = ()
```

`get_model_capabilities(client)` calls `client.models.get(model=MODEL)`, rejects
a missing/non-positive `output_token_limit`, and records the model name,
version, input limit, output limit, and supported actions in the private
preflight artifact. `_config(schema, capabilities)` sets exactly that output
limit, JSON schema response mode, seed 42, and high thinking.

- [ ] **Step 4: Write failing information-barrier tests**

```python
def test_observer_request_passes_host_information_barrier() -> None:
    action = _action()
    request = runner.build_observer_request(
        upload=_upload(), action=action, capabilities=_capabilities()
    )
    audit = runner.audit_observer_request(
        request,
        prohibited_values=[
            _row()["question"],
            *[option["text"] for option in _row()["options"]],
            *list("ABCD"),
            "BACD",
        ],
    )
    assert audit == {"passed": True, "matched_value_hashes": []}

def test_observer_schema_has_no_answer_or_ranking_fields() -> None:
    serialized = json.dumps(runner.OBSERVER_SCHEMA).casefold()
    assert "ranking" not in serialized
    assert "option_id" not in serialized
```

The host audit checks exact question and option strings plus serialized
rankings and protected-choice text. For single-letter option IDs, inspect JSON
field names and values rather than substring-searching arbitrary prose.

- [ ] **Step 5: Run information-barrier tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'information_barrier or observer_schema'`

Expected: FAIL until request auditing exists.

- [ ] **Step 6: Implement native-video request builders and barrier auditing**

Reuse the proven Google SDK shape:

```python
types.Part(
    file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
    video_metadata=types.VideoMetadata(
        fps=action.fps,
        start_offset=f"{action.start_seconds:.3f}s",
        end_offset=f"{action.end_seconds:.3f}s",
    ),
    media_resolution=types.PartMediaResolution(
        level=RESOLUTION_LEVELS[action.resolution]
    ),
)
```

Initializer and judge receive the complete video with audio at 4 FPS. Observer
requests use the model-selected complete or interval view at 1/4/8 FPS and
low/high resolution. Planner, updater, and critic are text-only fresh calls.
Freeze the observer audit beside the request before calling Gemini; refuse the
call if the audit fails.

- [ ] **Step 7: Write failing formal-cost tests**

```python
def test_formal_cost_prices_every_recovery_at_provider_maximum() -> None:
    estimate = runner.estimate_formal_cost(
        _request_payload(), capabilities=runner.ModelCapabilities(65536)
    )
    assert estimate["provider_call_ceiling"] == 20 * 28
    assert estimate["output_token_ceiling"] == 20 * 28 * 65536

def test_preflight_blocks_before_upload_when_formal_bound_exceeds_cap() -> None:
    uploads: list[Path] = []
    with pytest.raises(ValueError, match="formal worst-case"):
        runner.preflight(
            requests_path=_requests_path(),
            protected_control_path=_control_path(),
            historical_root=_historical_root(),
            capabilities=runner.ModelCapabilities(65536),
            upload_probe=lambda path: uploads.append(path),
        )
    assert uploads == []
```

- [ ] **Step 8: Run cost tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'formal_cost or preflight_blocks'`

Expected: FAIL until formal reservation and fail-before-upload exist.

- [ ] **Step 9: Implement label isolation, control loading, and preflight**

Recursively reject the existing forbidden label-key set from requests,
canary-gate artifacts, notebooks, and model requests. Require the protected
control to contain exactly one valid canonical ranking per request row and pin
its SHA-256. Compute media input from actual durations and the maximum
three-observation path. Compute output cost as
`provider_call_ceiling * output_token_limit * OUTPUT_USD_PER_MILLION / 1e6`.
Write a private `preflight.json` with both the formal and empirical estimates,
but gate only on the formal estimate.

- [ ] **Step 10: Run the focused runner tests**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py`

Expected: request-builder, barrier, and cost-preflight tests PASS; orchestration
tests added in Task 3 are not present yet.

- [ ] **Step 11: Commit request construction and preflight**

```bash
git add scripts/run_vqa_mc_gemini_four_option_agent_poc20.py tests/test_vqa_mc_four_option_agent_runner.py
git commit -m "feat: preflight Gemini four-option evidence runs"
```

---

### Task 3: Bounded trajectory, recovery, and immutable artifacts

**Files:**
- Modify: `scripts/run_vqa_mc_gemini_four_option_agent_poc20.py`
- Modify: `tests/test_vqa_mc_four_option_agent_runner.py`

**Interfaces:**
- Consumes: Task 1 contracts and Task 2 request/preflight functions.
- Produces: `call_structured_with_recovery()`, `run_row()`, `run_phase()`, stage fallbacks, per-call cost reservation, exclusive artifact writes, and full-output verification.

- [ ] **Step 1: Write failing recovery tests**

```python
def test_max_tokens_is_recovered_and_never_becomes_an_answer(tmp_path: Path) -> None:
    client = _Client([
        _provider_response(_initializer_payload(), finish_reason="MAX_TOKENS"),
        _provider_response(_initializer_payload(), finish_reason="STOP"),
        *_one_observation_stop_path(),
    ])
    result = runner.run_phase(
        client=client,
        output_root=tmp_path / "run",
        **_single_row_inputs(tmp_path),
    )
    assert result["recovery_call_count"] == 1
    assert result["operational_invalid_count"] == 0
    first = _read(tmp_path / "run/rows/00/initializer/attempt-1.json")
    assert first["finish_reason"] == "MAX_TOKENS"

def test_second_invalid_attempt_uses_declared_stage_fallback(tmp_path: Path) -> None:
    client = _Client([
        _provider_response({"broken": 1}),
        _provider_response({"still_broken": 1}),
        *_path_after_invalid_initializer(),
    ])
    result = runner.run_phase(client=client, output_root=tmp_path / "run", **_single_row_inputs(tmp_path))
    assert result["operational_invalid_count"] == 1
    assert _final_row(tmp_path)["initializer_status"] == "fallback_unknown_ledger"
```

- [ ] **Step 2: Run recovery tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'max_tokens or second_invalid'`

Expected: FAIL because structured recovery and stage fallbacks are missing.

- [ ] **Step 3: Implement structured-call recovery and dollar reservation**

`call_structured_with_recovery()` receives a request factory and parser. For
each attempt it freezes the safe response object, finish reason, usage, and
hash. Accept only `STOP` plus a complete parsed object. Retry exactly once for
`MAX_TOKENS`, incomplete JSON, or parser failure. Before each call, reserve the
provider maximum output cost plus estimated prompt/media input; refuse the call
when observed spend plus reserve would exceed `$1.50`.

Return this exact result shape:

```python
{
    "status": "passed" | "failed" | "budget_blocked",
    "record": parsed_or_none,
    "attempt_count": 1 | 2,
    "recovered": bool,
    "usage": usage_totals,
    "attempt_paths": ["rows/00/initializer/attempt-1.json"],
    "error_type": safe_error_or_none,
}
```

- [ ] **Step 4: Write failing end-to-end state-machine tests**

```python
def test_one_observation_path_cannot_stop_before_observer(tmp_path: Path) -> None:
    result = _run_fake(tmp_path, responses=_one_observation_stop_path())
    row = _final_row(tmp_path)
    assert row["observer_action_count"] == 1
    assert row["stop_status"] == "evidence_certified"
    assert result["provider_call_counts_by_role"] == {
        "initializer": 1, "planner": 1, "observer": 1,
        "updater": 1, "critic": 1, "judge": 1,
    }

def test_three_continue_decisions_force_stop_and_still_judge(tmp_path: Path) -> None:
    result = _run_fake(tmp_path, responses=_three_observation_path())
    row = _final_row(tmp_path)
    assert row["observer_action_count"] == 3
    assert row["stop_status"] == "forced_stop"
    assert row["judge_status"] == "passed"
    assert result["provider_request_count"] == 14

def test_invalid_final_judge_falls_back_to_protected_ranking_but_is_invalid(tmp_path: Path) -> None:
    _run_fake(tmp_path, responses=_invalid_judge_path())
    row = _final_row(tmp_path)
    assert row["ranking"] == list("BACD")
    assert row["final_source"] == "protected_direct_fallback"
    assert row["status"] == "operationally_invalid"
```

- [ ] **Step 5: Run state-machine tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'one_observation_path or force_stop or protected_ranking'`

Expected: FAIL because phase orchestration is missing.

- [ ] **Step 6: Implement row and phase orchestration**

`run_row()` performs:

```text
initializer
for round_index in 1..3:
    planner -> observer -> updater -> critic
    break only when critic accepts stop after a valid observer
judge
```

Use the declared fallback behavior from the spec at each stage. Freeze each
request audit, raw attempt, parsed result, view provenance, ledger snapshot,
and notebook snapshot before the next call. Use exclusive creation (`"x"`
mode) and refuse an existing output root. The final freeze declares hashes for
every stage source and each row notebook.

- [ ] **Step 7: Write failing canary-reuse and tamper tests**

```python
def test_full_phase_reuses_canary_rows_only_when_protocol_hashes_match(tmp_path: Path) -> None:
    canary = _completed_canary(tmp_path)
    result = runner.verify_canary_reuse(
        canary_root=canary,
        full_requests_path=_full_requests(tmp_path),
        canary_gate_path=_passing_gate(tmp_path, canary),
    )
    assert result["reusable_row_count"] == 5

    manifest = _read(canary / "run-manifest.json")
    manifest["prompt_contract_sha256"] = "0" * 64
    _write(canary / "run-manifest.json", manifest)
    with pytest.raises(ValueError, match="protocol identity"):
        runner.verify_canary_reuse(
            canary_root=canary,
            full_requests_path=_full_requests(tmp_path),
            canary_gate_path=_passing_gate(tmp_path, canary),
        )

def test_verify_output_rejects_tampered_observer_audit(tmp_path: Path) -> None:
    root = _completed_run(tmp_path)
    _write(root / "rows/00/round-1/observer/request-audit.json", {"passed": False})
    with pytest.raises(ValueError, match="observer audit"):
        runner.verify_output(root)
```

- [ ] **Step 8: Run reuse/tamper tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_runner.py -k 'canary_reuse or tampered_observer'`

Expected: FAIL until provenance verification exists.

- [ ] **Step 9: Implement canary reuse and output verification**

Require matching model/version, model arm, output limit, prompt-contract hash,
schema hash, pricing hash, recovery-policy hash, request-row hash, media hash,
protected-control hash, and final notebook hash. Verify every observer audit,
append-only observation ID, ledger citation, direct replacement condition,
call usage, and cost total. Link or copy no dataset media; reuse immutable JSON
records by declaring their original private paths and hashes.

- [ ] **Step 10: Run new and prior focused suites**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent_runner.py`

Expected: PASS with the frozen prior runner behavior unchanged.

- [ ] **Step 11: Commit orchestration**

```bash
git add scripts/run_vqa_mc_gemini_four_option_agent_poc20.py tests/test_vqa_mc_four_option_agent_runner.py
git commit -m "feat: run bounded four-option evidence trajectories"
```

---

### Task 4: Label-isolated canary preparation and scoring

**Files:**
- Create: `scripts/score_vqa_mc_four_option_agent_poc20.py`
- Create: `tests/test_vqa_mc_four_option_agent_scoring.py`

**Interfaces:**
- Consumes: private full requests, private labels, frozen direct-control predictions, phase outputs, and verified notebook hashes.
- Produces: `prepare_canary()`, `score_phase()`, `build_canary_gate()`, `sanitized_aggregate()`, and `build_private_audit()`.

- [ ] **Step 1: Write failing tests for private canary preparation**

```python
def test_prepare_canary_selects_exactly_the_five_control_misses_without_labels(tmp_path: Path) -> None:
    result = scorer.prepare_canary(
        requests_path=_requests_20(tmp_path),
        labels_path=_labels_20(tmp_path),
        control_path=_control_15_of_20(tmp_path),
        output_path=tmp_path / "canary-requests.json",
    )
    payload = _read(tmp_path / "canary-requests.json")
    assert result["row_count"] == 5
    assert len(payload["rows"]) == 5
    serialized = json.dumps(payload).casefold()
    assert "correct_option" not in serialized
    assert "reference_answer" not in serialized
    assert "label" not in serialized

def test_prepare_canary_rejects_a_control_other_than_15_of_20(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen 15/20 control"):
        scorer.prepare_canary(
            requests_path=_requests_20(tmp_path),
            labels_path=_labels_20(tmp_path),
            control_path=_control_14_of_20(tmp_path),
            output_path=tmp_path / "canary-requests.json",
        )
```

- [ ] **Step 2: Run canary-preparation tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_scoring.py -k prepare_canary`

Expected: FAIL because the scorer does not exist.

- [ ] **Step 3: Implement canary preparation as a separate label-bearing command**

Read labels only in `--prepare-canary` mode. Match request, label, and control
identities exactly; require the frozen control to score 15/20; select its five
misses in original request order; write only the selected request rows and
canonical variants plus source hashes and `selection_kind` equal to
`frozen_control_misses`. Use exclusive output creation and recursively assert
that no forbidden label key appears in the emitted bundle.

- [ ] **Step 4: Write failing canary/full scoring and gate tests**

```python
def test_canary_gate_requires_one_repair_and_all_behavioral_checks(tmp_path: Path) -> None:
    scored = scorer.score_phase(**_five_row_scoring_inputs(tmp_path, repaired=1))
    gate = scorer.build_canary_gate(scored)
    assert gate["continue_to_full"] is True

    scored["integrity"]["observer_blind_count"] = 4
    assert scorer.build_canary_gate(scored)["continue_to_full"] is False

def test_full_gate_requires_more_than_15_and_repairs_exceed_regressions(tmp_path: Path) -> None:
    scored = scorer.score_phase(**_full_scoring_inputs(tmp_path, top1=16, repaired=2, regressed=1))
    assert scored["decision_gate"]["candidate_for_fresh_evaluation"] is True

def test_operational_fallback_is_invalid_not_wrong(tmp_path: Path) -> None:
    scored = scorer.score_phase(**_inputs_with_invalid_judge_fallback(tmp_path))
    assert scored["metrics"]["final"]["valid_count"] == 19
    assert scored["integrity"]["operational_invalid_count"] == 1
```

- [ ] **Step 5: Run scoring tests and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_scoring.py -k 'canary_gate or full_gate or operational_fallback'`

Expected: FAIL until scoring and gates exist.

- [ ] **Step 6: Implement scoring, integrity, and gates**

For canary and full phases, compute top-1, MRR, valid denominator, per-stratum
metrics, repairs/regressions relative to the frozen control, observations per
row, forced/budget stops, replacements/retentions, cited replacement evidence,
recovery counts, role call counts, and observed/formal cost. Validate declared
source hashes and row notebooks before reading rankings.

The canary gate is the conjunction of:

```python
{
    "five_valid": valid_count == 5,
    "mandatory_observation": valid_observation_count == 5,
    "observer_blind": observer_blind_count == 5,
    "complete_ledgers": complete_ledger_count == 5,
    "citations_valid": citation_valid_count == 5,
    "no_truncation_as_answer": truncation_answer_count == 0,
    "repairs_at_least_one": transitions["repaired"] >= 1,
    "within_cost_cap": combined_reserved_cost_usd <= 1.50,
    "protocol_frozen": protocol_identity_valid,
}
```

The full gate additionally requires 20 valid rankings, top-1 greater than 15,
repairs greater than regressions, one valid blinded observation per row, full
integrity, unchanged protocol, and cost compliance.

- [ ] **Step 7: Write failing sanitized-aggregate privacy test**

```python
def test_sanitized_aggregate_excludes_private_rows_and_raw_text(tmp_path: Path) -> None:
    scored = scorer.score_phase(**_full_scoring_inputs(tmp_path))
    public = scorer.sanitized_aggregate(scored)
    serialized = json.dumps(public)
    assert "rows" not in public
    assert "video-a" not in serialized
    assert _question_text() not in serialized
    assert _option_text() not in serialized
    assert "raw_response" not in serialized
```

- [ ] **Step 8: Run privacy test and verify RED**

Run: `pytest -q tests/test_vqa_mc_four_option_agent_scoring.py -k sanitized_aggregate`

Expected: FAIL until the allowlisted aggregate exists.

- [ ] **Step 9: Implement private audit and allowlisted aggregate**

Private audit rows may contain row IDs, labels, rankings, decisive evidence,
and transition classifications; they remain under `data/`. The tracked
aggregate allowlist contains only schema/status, source hashes, aggregate
metrics, per-stratum counts, transitions, path/action distributions, usage,
cost, integrity counts, and gate booleans.

- [ ] **Step 10: Run all focused tests**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_four_option_agent_scoring.py`

Expected: PASS.

- [ ] **Step 11: Commit the scorer**

```bash
git add scripts/score_vqa_mc_four_option_agent_poc20.py tests/test_vqa_mc_four_option_agent_scoring.py
git commit -m "feat: score four-option VQA evidence agent"
```

---

### Task 5: Dry verification, cost gate, and authorized experiment execution

**Files:**
- Private create if preflight is permitted: `data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/inputs/canary-requests.json`
- Private create: `data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/preflight.json`
- Private create only if formal cost passes: `data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary/`
- Private create only if canary passes: `data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full/`

**Interfaces:**
- Consumes: completed Tasks 1-4, canonical shared data symlink, private Gemini credential, frozen requests/labels/control, and preserved WebM deliveries.
- Produces: a verified formal preflight; then, only if authorized gates pass, immutable canary and full-run artifacts.

- [ ] **Step 1: Verify the dataset link and clean tracked scope**

Run: `test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"`

Expected: exit 0.

Run: `git status --short`

Expected: the new implementation changes plus pre-existing unrelated untracked
files; no dataset file staged or modified.

- [ ] **Step 2: Run the full focused regression suite**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_four_option_agent_scoring.py tests/test_vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent_runner.py tests/test_vqa_mc_deliberative_agent_scoring.py`

Expected: PASS.

- [ ] **Step 3: Run static checks**

Run: `ruff check src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py scripts/run_vqa_mc_gemini_four_option_agent_poc20.py scripts/score_vqa_mc_four_option_agent_poc20.py tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_four_option_agent_scoring.py`

Expected: exit 0 with no findings.

Run: `python -m py_compile src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py scripts/run_vqa_mc_gemini_four_option_agent_poc20.py scripts/score_vqa_mc_four_option_agent_poc20.py`

Expected: exit 0.

- [ ] **Step 4: Prepare and verify the private five-miss canary bundle**

Run:

```bash
python scripts/score_vqa_mc_four_option_agent_poc20.py --prepare-canary --requests data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json --labels data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/labels.json --control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/inputs/canary-requests.json
```

Expected: `status=canary_prepared`, `row_count=5`, and a label-free output
bundle with frozen source hashes.

- [ ] **Step 5: Fetch live model metadata and run the formal preflight**

Run:

```bash
python scripts/run_vqa_mc_gemini_four_option_agent_poc20.py --preflight --canary-requests data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/inputs/canary-requests.json --full-requests data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json --protected-control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --historical-root data/vqa/benchmarks/direct-gemini-batch-475-v1 --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/preflight.json
```

Expected: the artifact records the live model output limit, model version,
formal canary/full/recovery cost, active pinned prices, and `paid_run_allowed`.
This metadata/preflight operation must not upload media or call
`generate_content`.

- [ ] **Step 6: Enforce the authorized cost decision**

If `paid_run_allowed` is false, stop here. Preserve `preflight.json`, report
the formal estimate, and request a new budget or a changed output/trajectory
contract; do not call Gemini generation.

If `paid_run_allowed` is true, proceed without another approval because the
user already authorized this exact protocol up to `$1.50`.

- [ ] **Step 7: Run and verify the paid canary when permitted**

Run:

```bash
python scripts/run_vqa_mc_gemini_four_option_agent_poc20.py --run-canary --requests data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/inputs/canary-requests.json --protected-control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --historical-root data/vqa/benchmarks/direct-gemini-batch-475-v1 --preflight data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/preflight.json --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary
```

Run:

```bash
python scripts/run_vqa_mc_gemini_four_option_agent_poc20.py --verify-output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary
```

Expected: five final rows, at least one observer record per row, valid hashes,
and observed plus reserved cost at or below `$1.50`.

- [ ] **Step 8: Score the canary and freeze its continuation gate**

Run:

```bash
python scripts/score_vqa_mc_four_option_agent_poc20.py --score-canary --requests data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/inputs/canary-requests.json --labels data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/labels.json --control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --run-root data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary-score.json --gate-output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary-gate.json
```

Expected: the private score records five rows; the gate contains only aggregate
booleans, counts, source hashes, protocol identity, and remaining formal cost.

- [ ] **Step 9: Enforce the canary decision**

If `continue_to_full` is false, stop without changing prompts, schemas, or live
artifacts. Preserve and report the negative/incomplete canary.

If `continue_to_full` is true, proceed automatically to the full phase.

- [ ] **Step 10: Run the remaining 15 rows and verify five-row reuse**

Run:

```bash
python scripts/run_vqa_mc_gemini_four_option_agent_poc20.py --run-full --requests data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json --protected-control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --historical-root data/vqa/benchmarks/direct-gemini-batch-475-v1 --preflight data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/preflight.json --reuse-canary data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary --canary-gate data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/canary-gate.json --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full
```

Run:

```bash
python scripts/run_vqa_mc_gemini_four_option_agent_poc20.py --verify-output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full
```

Expected: 20 final rows, exactly five verified reusable canary rows, no label
fields, valid notebooks/citations/audits, and reconciled cost no greater than
`$1.50`.

---

### Task 6: Score, record, render, and verify the result

**Files:**
- Modify: `docs/experiments/2026-08-21-vqa-mc-poc20.qmd`
- Regenerate: `docs/experiments/2026-08-21-vqa-mc-poc20.md`
- Modify: `docs/project-status/research-routing.qmd`
- Regenerate: `docs/project-status/research-routing.md`
- Regenerate: `reports/project-status/research-routing.html`
- Modify: `docs/experiments/artifacts/vqa-mc-poc20-20260821/README.md`
- Create: `docs/experiments/artifacts/vqa-mc-poc20-20260821/four-option-agent-summary.json`

**Interfaces:**
- Consumes: whichever terminal state Task 5 reached—formal preflight block, canary stop, or completed 20-row run.
- Produces: private score/audit when inference ran, one sanitized tracked aggregate, and updated canonical QMD evidence/decision records.

- [ ] **Step 1: Score the full result when it exists**

Run:

```bash
python scripts/score_vqa_mc_four_option_agent_poc20.py --score-full --requests data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json --labels data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/labels.json --control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a2-canonical/predictions-freeze.json --run-root data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full --output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full-score.json --audit-output data/vqa/benchmarks/vqa-mc-four-option-agent-poc20-20260822-v1/full-audit.json --sanitized-output docs/experiments/artifacts/vqa-mc-poc20-20260821/four-option-agent-summary.json
```

Expected: 20-row private score, private audit, and a sanitized aggregate. If
Task 5 stopped at preflight or canary, use the scorer's corresponding
`--sanitize-preflight` or `--sanitize-canary` mode to create the aggregate with
that terminal status and no invented full metrics.

- [ ] **Step 2: Validate the aggregate privacy boundary**

Run: `python -m json.tool docs/experiments/artifacts/vqa-mc-poc20-20260821/four-option-agent-summary.json >/dev/null`

Expected: exit 0.

Run: `pytest -q tests/test_vqa_mc_four_option_agent_scoring.py -k sanitized_aggregate`

Expected: PASS.

- [ ] **Step 3: Update the evidence-of-record QMD**

Add a `Four-option evidence-agent follow-up` subsection to
`docs/experiments/2026-08-21-vqa-mc-poc20.qmd` containing:

- the predeclared question, hypothesis, falsifier, treatment, controls, held
  fixed variables, five-row canary boundary, 20-row regression boundary,
  integrity safeguards, `$1.50` authorization, and decision rule;
- the observed terminal state and completion denominator;
- observed metrics/costs only when their artifacts exist;
- evidence validity and decision disposition using the project's established
  vocabulary; and
- a relative link to `four-option-agent-summary.json`.

Do not claim an accuracy result when formal preflight blocks or the canary does
not complete.

- [ ] **Step 4: Update the routing-theme QMD**

Update only the routing theme's working answer, what-has-been-tried entry, and
cheapest-valid-next-experiment queue as justified by the observed terminal
state. Link to the canonical experiment subsection instead of duplicating its
method and tables.

- [ ] **Step 5: Render QMD sources**

Run:

```bash
guide_render_cache=$(mktemp -d /tmp/trecvid-guide-render.XXXXXX)
env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render docs/experiments/2026-08-21-vqa-mc-poc20.qmd --to gfm --output-dir docs/experiments --output 2026-08-21-vqa-mc-poc20.md
```

Run:

```bash
for source in docs/project-status/*.qmd; do
  page=$(basename "$source" .qmd)
  env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render "$source" --to gfm --output-dir docs/project-status --output "$page.md"
done
env XDG_CACHE_HOME="$guide_render_cache/cache" quarto render docs/project-status --to html
```

Expected: every render exits 0.

- [ ] **Step 6: Run final verification**

Run: `pytest -q tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_four_option_agent_scoring.py tests/test_vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent_runner.py tests/test_vqa_mc_deliberative_agent_scoring.py`

Expected: PASS.

Run: `ruff check src/trec_video_2026/vqa_experiments/vqa_mc_four_option_agent.py scripts/run_vqa_mc_gemini_four_option_agent_poc20.py scripts/score_vqa_mc_four_option_agent_poc20.py tests/test_vqa_mc_four_option_agent.py tests/test_vqa_mc_four_option_agent_runner.py tests/test_vqa_mc_four_option_agent_scoring.py`

Expected: exit 0 with no findings.

Run: `git diff --check`

Expected: exit 0.

Run: `git status --short`

Expected: only intended tracked implementation/report changes and preserved
pre-existing unrelated untracked files; no private `data/` artifact staged.

- [ ] **Step 7: Commit the evidence record**

```bash
git add docs/experiments/2026-08-21-vqa-mc-poc20.qmd docs/experiments/2026-08-21-vqa-mc-poc20.md docs/project-status/research-routing.qmd docs/project-status/research-routing.md reports/project-status/research-routing.html docs/experiments/artifacts/vqa-mc-poc20-20260821/README.md docs/experiments/artifacts/vqa-mc-poc20-20260821/four-option-agent-summary.json
git commit -m "docs: record four-option evidence agent result"
```
