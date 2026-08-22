# VQA MC Single-Model Deliberative Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, run, score, and record a bounded single-Gemini deliberative video-MC agent on the frozen 20-video development panel.

**Architecture:** Keep the frozen direct and two-call experiments unchanged. A new provider-independent module owns strict response contracts, prompt visibility, media-action validation, append-only notebook transitions, and fallbacks; a new Gemini runner owns uploads, fresh-context calls, adaptive orchestration, cost reservation, and immutable private artifacts; a label-only scorer evaluates frozen outputs after inference. Gemini makes every semantic judgment, while Python enforces only structure, provenance, visibility, and operational budgets.

**Tech Stack:** Python 3.12, `google-genai`, dataclasses, JSON Schema response contracts, `pytest`, `ruff`, Quarto QMD reporting.

**Spec:** `docs/superpowers/specs/2026-08-22-vqa-mc-single-model-deliberative-agent-design.md`

## Global Constraints

- Use only `gemini-3.7-flash` with high thinking for every solver, planner, observer, and judge call.
- Preserve frozen direct and two-call code and artifacts; create new files and output roots.
- Use the existing 20 canonical rows and canonical A-D option order.
- Reject labels recursively before any inference operation.
- The observer request must not receive option structures, solver rankings, hypothesis tables, or notebook state.
- Controller code may validate structure and provenance but may not classify question semantics or decide whether evidence changes an answer.
- Maximum seven Gemini calls per row; a correction call consumes that ceiling and may remove an inspection.
- Reserve one complete-video final-judge call before authorizing any correction or observer call.
- Maximum two observer views per row.
- Preflight must conservatively remain at or below `$0.75` for all 20 rows before credentials are loaded.
- Private questions, options, predictions, media, notebooks, and provider responses stay under ignored `data/`; tracked evidence is sanitized and aggregate-only.
- QMD is authoritative; generated Markdown and HTML are never hand-edited.

---

## File map

- Create `src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py`: schemas, parsers, prompts, media actions, notebook transitions, visibility renderers, and final fallback.
- Create `scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py`: label-isolated preflight, uploads, request builders, adaptive state machine, budget accounting, and immutable freezes.
- Create `scripts/score_vqa_mc_deliberative_agent_poc20.py`: post-freeze scoring, transitions, path distributions, integrity metrics, and decision gate.
- Create `tests/test_vqa_mc_deliberative_agent.py`: contracts, prompt visibility, action parsing, notebook immutability, and fallback tests.
- Create `tests/test_vqa_mc_deliberative_agent_runner.py`: request construction, preflight/cost, state-machine paths, visibility, freezes, and failure handling.
- Create `tests/test_vqa_mc_deliberative_agent_scoring.py`: label joins, metrics, transition counts, action distributions, integrity checks, and gate behavior.
- Modify `docs/experiments/2026-08-21-vqa-mc-poc20.qmd`: append the planned condition and, after scoring, the sanitized observed result.
- Modify `docs/experiments/README.md`, `docs/project-status/research-routing.qmd`, and `docs/project-status/experiment-ledger.qmd` only after a completed result changes current evidence or the next-test queue.
- Create `docs/experiments/artifacts/vqa-mc-poc20-20260821/deliberative-agent-summary.json`: sanitized aggregate evidence after the private score is frozen.

---

### Task 1: Provider-independent response contracts and media actions

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py`
- Create: `tests/test_vqa_mc_deliberative_agent.py`

**Interfaces:**
- Consumes: `OPTION_IDS` and `parse_ranking` from `vqa_mc_poc20.py`.
- Produces: `SOLVER_SCHEMA`, `PLANNER_SCHEMA`, `OBSERVER_SCHEMA`, `JUDGE_SCHEMA`, `MediaAction`, `parse_solver`, `parse_planner`, `parse_observer`, `parse_judge`, and `validate_action`.

- [ ] **Step 1: Write failing schema and parser tests**

```python
def test_solver_requires_all_four_hypotheses_and_separates_observation() -> None:
    parsed = parse_solver(_solver_payload(), duration_seconds=6.0, solver_id="solver_1")
    assert parsed["solver_id"] == "solver_1"
    assert [row["option_id"] for row in parsed["hypotheses"]] == list("ABCD")
    assert parsed["observations"][0]["literal_observation"] == "The hand touches the ball."
    assert parsed["observations"][0]["interpretation"] == "This may be a release."


def test_planner_finish_has_no_action_and_inspect_requires_action() -> None:
    finished = parse_planner(_planner_finish_payload(), duration_seconds=6.0)
    assert finished["decision"] == "finish_search"
    assert finished["action"] is None
    with pytest.raises(ValueError, match="action"):
        parse_planner({**_planner_inspect_payload(), "action": None}, duration_seconds=6.0)


def test_observer_timestamps_must_be_inside_realized_view() -> None:
    action = MediaAction("inspect_interval", 2.0, 4.0, 8.0, "high", "Inspect contact.", ())
    with pytest.raises(ValueError, match="view"):
        parse_observer(_observer_payload(timestamp_seconds=4.5), action=action, duration_seconds=6.0, view_id="view-1")


def test_judge_citations_must_exist() -> None:
    with pytest.raises(ValueError, match="observation ID"):
        parse_judge(_judge_payload(citations=["missing"]), observation_ids={"solver-1-o1"})
```

- [ ] **Step 2: Run the focused test and confirm the new module is missing**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent.py`

Expected: collection fails with `ModuleNotFoundError` for `vqa_mc_deliberative_agent`.

- [ ] **Step 3: Implement bounded schemas and exact parsers**

```python
@dataclass(frozen=True)
class MediaAction:
    kind: str
    start_seconds: float
    end_seconds: float
    fps: float
    resolution: str
    investigative_focus: str
    expectations: tuple[dict[str, str], ...]


def validate_action(value: object, *, duration_seconds: float) -> MediaAction:
    payload = _strict_object(value, required=ACTION_KEYS)
    kind = _enum(payload["kind"], "kind", {"rewatch_full", "inspect_interval"})
    start = _finite(payload["start_seconds"], "start_seconds")
    end = _finite(payload["end_seconds"], "end_seconds")
    if start < 0 or end > duration_seconds or end <= start:
        raise ValueError("action interval is outside the video")
    if kind == "rewatch_full" and (start != 0.0 or end != duration_seconds):
        raise ValueError("rewatch_full must cover the complete video")
    return MediaAction(
        kind=kind,
        start_seconds=start,
        end_seconds=end,
        fps=_enum_number(payload["fps"], "fps", {1.0, 4.0, 8.0}),
        resolution=_enum(payload["resolution"], "resolution", {"low", "high"}),
        investigative_focus=_bounded_text(payload["investigative_focus"], 500),
        expectations=tuple(_parse_expectations(payload["expectations"])),
    )
```

Implement complete JSON schemas with `additionalProperties: false`; complete A-D permutations; one hypothesis per option; bounded strings and arrays; finite timestamps; observer timestamps inside the realized action; and judge citations limited to existing IDs.

- [ ] **Step 4: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the contract layer**

```bash
git add src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent.py
git commit -m "feat: add deliberative VQA agent contracts"
```

---

### Task 2: Prompt visibility and append-only notebook

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py`
- Modify: `tests/test_vqa_mc_deliberative_agent.py`

**Interfaces:**
- Consumes: parsed solver/planner/observer records and `MediaAction` from Task 1.
- Produces: `solver_prompt`, `planner_prompt`, `observer_prompt`, `judge_prompt`, `initialize_notebook`, `append_planner`, `append_observer`, `notebook_observation_ids`, and `choose_final_ranking`.

- [ ] **Step 1: Add failing visibility and immutability tests**

```python
def test_blind_solver_prompt_has_no_anchor_state() -> None:
    prompt = solver_prompt(_row())
    assert "anchor" not in prompt.casefold()
    assert "A. once" in prompt
    assert "D. four times" in prompt


def test_observer_prompt_contains_focus_but_no_question_options_or_rankings() -> None:
    action = _action()
    prompt = observer_prompt(action)
    assert action.investigative_focus in prompt
    assert _row()["question"] not in prompt
    assert "A. once" not in prompt
    assert "ranking" not in prompt.casefold()


def test_append_observer_does_not_mutate_prior_notebook() -> None:
    original = initialize_notebook("video-a", [_solver_record("solver_1"), _solver_record("solver_2")])
    before = canonical_json(original)
    updated = append_observer(original, round_index=1, action=_action(), observer=_observer_record(), provenance=_provenance())
    assert canonical_json(original) == before
    assert updated["rounds"][0]["observer"]["observations"][0]["observation_id"] == "round-1-o1"


def test_judge_prompt_renders_literal_observations_before_interpretations() -> None:
    prompt = judge_prompt(_row(), _notebook())
    assert prompt.index("Literal observations") < prompt.index("Interpretations")
```

- [ ] **Step 2: Run the tests and verify the new functions fail**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent.py`

Expected: failures name the missing prompt/notebook functions.

- [ ] **Step 3: Implement the four fresh-context prompts and notebook transitions**

```python
def observer_prompt(action: MediaAction) -> str:
    return (
        "Observe only the supplied native audio-video view. Do not choose an answer "
        "or infer unseen events. Record literal visible or audible facts first, then "
        "put any interpretation in its separate field. Use original-video seconds.\n\n"
        f"Investigative focus: {action.investigative_focus}"
    )


def append_observer(notebook: Mapping[str, Any], *, round_index: int,
                    action: MediaAction, observer: Mapping[str, Any],
                    provenance: Mapping[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(notebook))
    entry = _round_entry(updated, round_index)
    if "observer" in entry:
        raise ValueError("observer entry already exists")
    entry["action"] = asdict(action)
    entry["provenance"] = dict(provenance)
    entry["observer"] = _assign_observation_ids(observer, prefix=f"round-{round_index}")
    return updated
```

Render solver records neutrally as `solver_1` and `solver_2`. Keep observer prompts constructed only from `MediaAction`; do not accept `row` or `notebook` arguments. Keep every transition copy-on-write and reject duplicate rounds or observer entries.

- [ ] **Step 4: Run all contract tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent.py`

Expected: pass.

- [ ] **Step 5: Commit prompt and notebook behavior**

```bash
git add src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent.py
git commit -m "feat: add deliberative prompts and notebook"
```

---

### Task 3: Gemini request builders and conservative no-credential preflight

**Files:**
- Create: `scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py`
- Create: `tests/test_vqa_mc_deliberative_agent_runner.py`

**Interfaces:**
- Consumes: schemas, prompts, parsers, `MediaAction`, and notebook functions from Tasks 1–2; frozen request and delivery artifacts.
- Produces: `load_requests`, `canonical_variants`, `estimate_worst_case_cost`, `preflight`, `build_solver_request`, `build_planner_request`, `build_observer_request`, and `build_judge_request`.

- [ ] **Step 1: Write failing request-visibility and cost tests**

```python
def test_solver_and_judge_receive_complete_low_4fps_video() -> None:
    solver = runner.build_solver_request(row=_row(), upload=_upload(), stage="anchor")
    judge = runner.build_judge_request(row=_row(), upload=_upload(), notebook=_notebook())
    for request in (solver, judge):
        video = request.contents[0].parts[0]
        assert video.video_metadata.fps == 4.0
        assert video.video_metadata.start_offset is None
        assert video.video_metadata.end_offset is None
        assert video.media_resolution.level.name.endswith("LOW")


def test_planner_is_text_only_and_observer_is_blinded() -> None:
    planner = runner.build_planner_request(row=_row(), notebook=_notebook(), budget=_budget())
    observer = runner.build_observer_request(upload=_upload(), action=_action())
    assert all(getattr(part, "file_data", None) is None for part in planner.contents[0].parts)
    rendered = "\n".join(part.text or "" for part in observer.contents[0].parts)
    assert _row()["question"] not in rendered
    assert "A. once" not in rendered


def test_preflight_rejects_labels_and_cost_above_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label"):
        runner.load_requests(_request_payload(tmp_path, include_label=True), expected_rows=None, expected_variants=None)
    with pytest.raises(ValueError, match="cap"):
        runner.preflight(requests_path=_request_payload(tmp_path), historical_root=_historical_root(tmp_path), max_cost_usd=0.000001, expected_rows=None, expected_variants=None)
```

- [ ] **Step 2: Run runner tests and confirm import failure**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent_runner.py`

Expected: import fails because the runner does not exist.

- [ ] **Step 3: Implement request builders with role-specific schemas and visibility**

```python
def build_observer_request(*, upload: Mapping[str, Any], action: MediaAction) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[
            _video_part(upload=upload, fps=action.fps, resolution=action.resolution,
                        start_offset=f"{action.start_seconds:.3f}s",
                        end_offset=f"{action.end_seconds:.3f}s"),
            types.Part(text=observer_prompt(action)),
        ])],
        metadata={"arm": MODEL_ARM, "stage": "observer"},
        config=_config(OBSERVER_SCHEMA, max_output_tokens=OBSERVER_MAX_OUTPUT_TOKENS),
    )
```

Use `solver_prompt(row)` for both initial calls, `planner_prompt(row, notebook, budget)` for text-only calls, and `judge_prompt(row, notebook)` with complete baseline media. Every role uses model `gemini-3.7-flash`, seed `42`, JSON output, high thinking, and `include_thoughts=False`.

- [ ] **Step 4: Implement conservative pricing and media-token reservation**

```python
ROLE_OUTPUT_LIMITS = {"solver": 2048, "planner": 1536, "observer": 1536, "judge": 2048}
MEDIA_TOKENS_PER_SECOND = {
    (1.0, "low"): 78,
    (4.0, "low"): 312,
    (8.0, "low"): 624,
    (1.0, "high"): 284,
    (4.0, "high"): 1136,
    (8.0, "high"): 2272,
}
MAX_OBSERVER_INPUT_TOKENS_PER_ROW = 10_000
MAX_TEXT_INPUT_TOKENS_PER_ROW = 4_000
MAX_COST_USD = 0.75
```

With 246.9 total panel seconds, the preflight formula is exactly:

```python
base_media = math.ceil(total_seconds * 3 * MEDIA_TOKENS_PER_SECOND[(4.0, "low")])
observer_media = row_count * MAX_OBSERVER_INPUT_TOKENS_PER_ROW
text_input = row_count * MAX_TEXT_INPUT_TOKENS_PER_ROW
output_tokens = row_count * (
    2 * ROLE_OUTPUT_LIMITS["solver"]
    + 2 * ROLE_OUTPUT_LIMITS["planner"]
    + 2 * ROLE_OUTPUT_LIMITS["observer"]
    + ROLE_OUTPUT_LIMITS["judge"]
)
```

After the bounded output-truncation recovery, the repaired-run ceiling is
approximately `$0.6525`; including the invalid `$0.09529950` `a1` run, the
combined conservative ceiling is `$0.74776162`, below `$0.75`. Return every
assumption and reject any configuration whose recomputed combined total
exceeds the cap. `authorize_action` rejects an action whose estimated media
tokens exceed the row's remaining 10,000-token observer allowance or would
consume the reserved judge budget.

- [ ] **Step 5: Run runner tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent_runner.py`

Expected: request construction, label rejection, cost cap, and media reservation tests pass.

- [ ] **Step 6: Commit request and preflight support**

```bash
git add scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py tests/test_vqa_mc_deliberative_agent_runner.py
git commit -m "feat: add deliberative Gemini preflight"
```

---

### Task 4: Adaptive runner and immutable evidence lifecycle

**Files:**
- Modify: `scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py`
- Modify: `tests/test_vqa_mc_deliberative_agent_runner.py`

**Interfaces:**
- Consumes: Task 3 request builders/preflight and Task 2 notebook transitions.
- Produces: `run`, `verify_output`, CLI `--preflight`/`--run`/`--verify-output`, private stage freezes, `final-predictions-freeze.json`, and `summary.json`.

- [ ] **Step 1: Add mocked zero-, one-, and two-observer end-to-end tests**

```python
@pytest.mark.parametrize(
    ("responses", "expected_calls", "expected_observers"),
    [
        (_zero_observer_responses(), 4, 0),
        (_one_observer_responses(), 6, 1),
        (_two_observer_responses(), 7, 2),
    ],
)
def test_run_supports_all_bounded_paths(tmp_path: Path, responses, expected_calls, expected_observers) -> None:
    client = _Client(responses)
    result = runner.run(
        requests_path=_request_payload(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=tmp_path / "output",
        client=client,
        upload_file=lambda _client, _path: _upload(),
        expected_rows=None,
        expected_variants=None,
    )
    assert len(client.models.requests) == expected_calls
    assert result["observer_call_count"] == expected_observers
    assert result["final_valid_count"] == 1
```

Add tests that an invalid action consumes a call slot and removes a later action, a malformed observer never erases prior evidence, an invalid judge falls back explicitly, all artifacts use exclusive creation, and one upload is reused across every call for a row.

- [ ] **Step 2: Run state-machine tests and confirm failure**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent_runner.py -k 'bounded_paths or invalid_action or invalid_judge'`

Expected: failures show that `run` and freezes are incomplete.

- [ ] **Step 3: Implement stage-wide initial calls and per-round adaptive orchestration**

```python
for stage, solver_id in (("anchor", "solver_1"), ("blind", "solver_2")):
    raw_rows = _call_solver_stage(
        client=client, stage=stage, variants=variants,
        indexed=indexed, uploads=uploads,
    )
    _freeze_raw_stage(output_root=output_root, stage=stage, rows=raw_rows)
    parsed_rows = _parse_solver_stage(
        output_root=output_root, stage=stage, solver_id=solver_id,
        variants=variants, indexed=indexed,
    )
    _freeze_parsed_stage(output_root=output_root, stage=stage, rows=parsed_rows)

notebooks = _initialize_all_notebooks(anchor_rows, blind_rows)
for round_index in (1, 2):
    planner_rows = _call_eligible_planners(notebooks, budgets, round_index)
    _freeze_round(
        output_root=output_root, kind="planner",
        round_index=round_index, rows=planner_rows,
    )
    observer_rows = _call_authorized_observers(planner_rows, uploads, budgets)
    _freeze_round(
        output_root=output_root, kind="observer",
        round_index=round_index, rows=observer_rows,
    )
    notebooks = _append_round_results(notebooks, planner_rows, observer_rows)
    _freeze_notebooks(
        output_root=output_root, round_index=round_index,
        notebooks=notebooks,
    )
```

Rows choosing `finish_search` skip later planner/observer calls. Rows denied by media or call budget record the tool error and proceed to final judge. No correctness-derived choice enters this path.

- [ ] **Step 4: Implement final judge, fallback, verification, and summary**

```python
final_ranking, final_source = choose_final_ranking(
    judge_record=parsed_judge,
    planner_records=notebook["rounds"],
    solver_records=notebook["solvers"],
)
```

Freeze judge raw responses before parsing. Final rows include ranking, source, cited observation IDs, ambiguity flag, call path, action count, notebook hash, usage, and errors. `verify_output` recomputes every declared freeze hash and validates counts, notebook hashes, final citation references, and usage records without labels.

- [ ] **Step 5: Run runner and contract tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent_runner.py`

Expected: pass.

- [ ] **Step 6: Commit the runnable agent**

```bash
git add scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py tests/test_vqa_mc_deliberative_agent_runner.py
git commit -m "feat: run bounded deliberative Gemini agent"
```

---

### Task 5: Label-isolated scorer and decision gate

**Files:**
- Create: `scripts/score_vqa_mc_deliberative_agent_poc20.py`
- Create: `tests/test_vqa_mc_deliberative_agent_scoring.py`

**Interfaces:**
- Consumes: frozen requests, separate labels, frozen direct predictions, two-call final predictions, deliberative solver predictions, final predictions, notebooks, and run summary.
- Produces: private `score.json` with row-level audit detail and a sanitized aggregate object for tracked evidence.

- [ ] **Step 1: Write failing scorer tests**

```python
def test_score_reports_repairs_regressions_paths_and_integrity(tmp_path: Path) -> None:
    result = scorer.score(**_artifacts(tmp_path))
    assert result["metrics"]["final"]["top1_correct"] == 2
    assert result["transitions_control_to_final"]["repaired"] == 1
    assert result["transitions_control_to_final"]["regressed"] == 0
    assert result["path_distribution"] == {"zero_actions": 1, "one_action": 1, "two_actions": 0}
    assert result["integrity"]["citation_valid_count"] == 2


def test_gate_requires_improvement_more_repairs_and_known_hit_preservation(tmp_path: Path) -> None:
    result = scorer.score(**_artifacts(tmp_path))
    assert result["decision_gate"]["final_exceeds_control"] is True
    assert result["decision_gate"]["repairs_exceed_regressions"] is True
    assert result["decision_gate"]["known_regression_hits_preserved"] is True
```

Add identity-drift, duplicate-row, invalid-ranking, missing-notebook-hash, missing-citation, over-cap, and nonadaptive-path tests.

- [ ] **Step 2: Run scorer tests and verify import failure**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent_scoring.py`

Expected: import fails because the scorer does not exist.

- [ ] **Step 3: Implement post-freeze joins, metrics, and gate**

```python
decision_gate = {
    "requires_20_valid": metrics["final"]["valid_count"] == len(ids),
    "final_exceeds_control": metrics["final"]["top1_correct"] > metrics["frozen_control"]["top1_correct"],
    "repairs_exceed_regressions": transitions["repaired"] > transitions["regressed"],
    "known_regression_hits_preserved": all(final_predictions[row_id][0] == labels[row_id] for row_id in known_regression_ids),
    "structural_integrity": all(integrity.values()),
    "adaptive_paths": zero_action_count >= 1 and distinct_action_configurations >= 2,
    "within_cost_cap": observed_cost is not None and observed_cost <= 0.75,
}
decision_gate["promising_development_result"] = all(decision_gate.values())
```

Rows include labels and private traces only in `score.json`. The sanitized aggregate excludes questions, options, labels, row IDs, rankings, notebook contents, and provider artifact locations.

- [ ] **Step 4: Run scorer tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_deliberative_agent_scoring.py`

Expected: pass.

- [ ] **Step 5: Commit scoring**

```bash
git add scripts/score_vqa_mc_deliberative_agent_poc20.py tests/test_vqa_mc_deliberative_agent_scoring.py
git commit -m "feat: score deliberative VQA agent"
```

---

### Task 6: Full local verification and no-credential preflight

**Files:**
- Modify only files from Tasks 1–5 if verification reveals defects.

**Interfaces:**
- Consumes: completed implementation and canonical shared dataset link.
- Produces: passing test/lint evidence and a sanitized preflight result.

- [ ] **Step 1: Run focused and regression tests**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/test_vqa_mc_poc20.py \
  tests/test_vqa_mc_poc20_runners.py \
  tests/test_vqa_mc_poc20_scoring.py \
  tests/test_vqa_mc_adaptive_agent.py \
  tests/test_vqa_mc_adaptive_agent_runner.py \
  tests/test_vqa_mc_adaptive_agent_scoring.py \
  tests/test_vqa_mc_deliberative_agent.py \
  tests/test_vqa_mc_deliberative_agent_runner.py \
  tests/test_vqa_mc_deliberative_agent_scoring.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run lint and patch checks**

Run:

```bash
ruff check \
  src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py \
  scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py \
  scripts/score_vqa_mc_deliberative_agent_poc20.py \
  tests/test_vqa_mc_deliberative_agent.py \
  tests/test_vqa_mc_deliberative_agent_runner.py \
  tests/test_vqa_mc_deliberative_agent_scoring.py
git diff --check
```

Expected: no findings.

- [ ] **Step 3: Verify canonical data and unused output root**

Run:

```bash
test "$(readlink -f data)" = "/home/npatta01/data/competitions/trec_video_2026/data"
test ! -e data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2
```

Expected: both commands exit zero.

- [ ] **Step 4: Run no-credential preflight**

Run:

```bash
PYTHONPATH=src python scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py \
  --preflight --canonical-only
```

Expected: JSON reports `status: preflight_passed`, `row_count: 20`, `max_calls_per_row: 7`, `final_call_reserved: true`, and `estimated_cost_usd <= 0.75`. It must not load or report credential material.

- [ ] **Step 5: Commit verification fixes if any**

```bash
git add src/trec_video_2026/vqa_experiments/vqa_mc_deliberative_agent.py scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py scripts/score_vqa_mc_deliberative_agent_poc20.py tests/test_vqa_mc_deliberative_agent.py tests/test_vqa_mc_deliberative_agent_runner.py tests/test_vqa_mc_deliberative_agent_scoring.py
git commit -m "test: verify deliberative VQA agent"
```

Skip this commit when verification required no changes.

---

### Task 7: Authorized 20-video run and post-freeze score

**Files:**
- Private create: `data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2/`

**Interfaces:**
- Consumes: verified runner, canonical requests, preserved WebM deliveries, private Gemini credential, frozen labels, and frozen control predictions.
- Produces: immutable private inference artifacts, `score.json`, and private audit summary.

- [ ] **Step 1: Launch paid run only after Task 6 passes**

Run:

```bash
PYTHONPATH=src python scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py \
  --run --canonical-only \
  --output data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2
```

Expected: `status: completed`, 20 final rows, no output overwrite, and observed cost at or below `$0.75`.

- [ ] **Step 2: Verify freezes before labels are loaded**

Run:

```bash
PYTHONPATH=src python scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py \
  --verify-output data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2
```

Expected: raw freezes, notebook snapshots, final predictions, counts, hashes, citation references, and usage records validate.

- [ ] **Step 3: Score only frozen predictions**

Run:

```bash
PYTHONPATH=src python scripts/score_vqa_mc_deliberative_agent_poc20.py \
  --requests data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json \
  --labels data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/labels.json \
  --control data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a1/predictions-freeze.json \
  --adaptive data/vqa/benchmarks/vqa-mc-adaptive-agent-poc20-20260821-a1/final-predictions-freeze.json \
  --run-root data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2 \
  --output data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a2/score.json
```

Expected: `status: scored`; labels appear only in the private score artifact created after inference freeze.

- [ ] **Step 4: Audit every final miss and answer change**

Inspect private traces for every final miss, repair, regression, budget-rejected request, missing citation, judge departure from two agreeing solvers, and ambiguity flag. Write a private audit JSON under the run root. Do not copy questions, options, row IDs, or trace text into tracked files.

---

### Task 8: Sanitize and record the result

**Files:**
- Create: `docs/experiments/artifacts/vqa-mc-poc20-20260821/deliberative-agent-summary.json`
- Modify: `docs/experiments/2026-08-21-vqa-mc-poc20.qmd`
- Modify: `docs/experiments/README.md`
- Conditionally modify: `docs/project-status/research-routing.qmd`
- Conditionally modify: `docs/project-status/experiment-ledger.qmd`
- Generate from QMD: sibling Markdown and affected HTML reports.

**Interfaces:**
- Consumes: verified private summary, score, notebook audit, and hashes.
- Produces: aggregate evidence of record and condition-bound project decision.

- [ ] **Step 1: Create sanitized aggregate artifact**

Include only:

```json
{
  "schema_version": "vqa-mc-deliberative-agent-poc20-aggregate-v1",
  "tested_condition": {},
  "completion": {},
  "metrics": {},
  "paired_transitions": {},
  "path_distribution": {},
  "inspection_distribution": {},
  "integrity": {},
  "decision_gate": {},
  "cost_usd": {},
  "usage": {},
  "private_evidence_sha256": {},
  "privacy": "aggregate-only"
}
```

Reject the tracked artifact if it contains a question, answer option, label, video/row identifier, prediction, observation text, notebook text, credential, or provider artifact locator.

- [ ] **Step 2: Update owning QMD result record**

Append reason, treatment/control/panel, held-fixed variables, hypothesis/falsifier/gate, observed completion and metrics, evidence validity, contamination boundary, decision disposition, reconsideration condition, and evidence links to `docs/experiments/2026-08-21-vqa-mc-poc20.qmd`.

- [ ] **Step 3: Update routing only if decision changes**

If the gate passes, require a fresh label-isolated confirmation in routing and queue surfaces. If it fails, preserve the negative condition and leave direct Gemini as the operational baseline.

- [ ] **Step 4: Render and verify documentation**

Run the project guide's Quarto regeneration commands for affected QMD files, then:

```bash
git diff --check
rg -n 'GEMINI_API_KEY|provider.*uri|video_id|correct_option_id|question' \
  docs/experiments/artifacts/vqa-mc-poc20-20260821/deliberative-agent-summary.json
```

Expected: render succeeds; patch check is clean; privacy scan finds no row-level fields.

- [ ] **Step 5: Commit evidence record**

```bash
git add docs/experiments/2026-08-21-vqa-mc-poc20.qmd docs/experiments/2026-08-21-vqa-mc-poc20.md docs/experiments/README.md docs/experiments/artifacts/vqa-mc-poc20-20260821/deliberative-agent-summary.json
git add docs/project-status/research-routing.qmd docs/project-status/research-routing.md docs/project-status/experiment-ledger.qmd docs/project-status/experiment-ledger.md reports/project-status/research-routing.html reports/project-status/experiment-ledger.html
git commit -m "docs: record deliberative VQA agent result"
```

Stage only files actually changed. Do not stage private `data/` artifacts or unrelated dirty files.

---

## Final verification

- [ ] Run focused and existing MC tests again.
- [ ] Run `ruff check` on every new Python file.
- [ ] Run `git diff --check`.
- [ ] Verify private run hashes and score source hashes.
- [ ] Verify tracked aggregate privacy.
- [ ] Confirm result disposition matches the predeclared gate.
- [ ] Report exact top-1/MRR, repairs/regressions, path counts, observed cost, failures, and whether a fresh unseen panel is justified.
