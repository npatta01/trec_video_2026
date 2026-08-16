# Dense Question-Localized Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether 12-FPS sampling inside frozen question-relevant clips recovers brief visual facts that the existing 4-FPS question-blind map omitted.

**Architecture:** A treatment-first runner owns the exact Gemini probe, schema, validator, one-attempt artifact chain, and prediction freeze. A separate preparation script classifies the completed Issue #35 insufficiency rows, freezes paired local clips and provider-safe inputs, and enforces the 24-call cap. A post-freeze scorer performs reference-aware paired adjudication and produces a sanitized mechanism result without final-answer metrics.

**Tech Stack:** Python 3.11, pytest, Google GenAI SDK, ffmpeg/ffprobe, SHA-256-bound JSON artifacts, Quarto QMD/GFM/HTML reporting.

## Global Constraints

- Treat this as selected, oracle-screened evidence-recovery evidence, not a prevalence estimate or deployable localization evaluation.
- Keep all 475 source rows eligible; prior inspection is an exposure flag, never an exclusion.
- Freeze and commit the prompt, schema, validator, runner, model settings, call cap, and decision rule before label-aware panel formation.
- Provider requests contain only the question, frozen scene-map lines, clip offset, and local clip; never references, direct predictions, prior answers, scores, or audit verdicts.
- Use `gemini-3.1-pro-preview`, temperature `0.0`, thinking level `HIGH`, high media resolution, maximum output `2048`, one attempt per arm, and no retry.
- Compare one byte-identical local clip at `4.0` FPS and `12.0` FPS; requested FPS is the only arm difference.
- Run at most eight brief-visual targets and four supported controls, for at most 24 calls. Require at least two targets and two controls before authorization.
- Do not emit competition answers or multiple candidate facts. This stage measures evidence recovery only.
- Do not implement counting issue #34, shared router issue #36, AutoGaze, a global map rerun, or a full-panel native-video rerun.
- Store private media, questions, references, responses, and row-level evidence only under `data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1/`.
- Preserve zero-call, incomplete, negative, and failed-gate outcomes; never overwrite a frozen artifact.

---

### Task 1: Freeze the paired probe treatment

**Files:**
- Create: `configs/vqa/dense_question_localized_evidence_issue35.json`
- Create: `scripts/run_dense_question_localized_evidence.py`
- Create: `tests/test_run_dense_question_localized_evidence.py`

**Interfaces:**
- Produces `PROBE_SCHEMA`, `build_prompt(question: str, map_lines: Sequence[Mapping[str, Any]], clip_start_seconds: float, clip_end_seconds: float) -> str`, `validate_probe(payload: Any, clip_duration_seconds: float) -> dict[str, Any]`, `run(root: Path, call: CallFunction | None = None) -> None`, and `verify_predictions_freeze(root: Path) -> None`.
- `CallFunction` consumes `(clip_bytes: bytes, mime_type: str, prompt: str, fps: float)` and returns `(raw_text: str, metadata: Mapping[str, Any])`.
- Consumes later: `treatment-freeze.json`, `selection-freeze.json`, `provider-rows.json`, `media-manifest.json`, and `provider-authorization.json` from Task 2.

- [ ] **Step 1: Write failing prompt and validator tests**

```python
def test_probe_requires_supported_observation_or_explicit_more_evidence():
    observed = runner.validate_probe(
        {
            "question_interpretation": {
                "requested_entity": "the robot",
                "requested_visual_fact": "final action",
                "temporal_qualifier": "at the end",
            },
            "fact_observed": True,
            "observed_fact": "The robot clicks its feet.",
            "evidence_timestamps": [2.75],
            "conflicts_or_uncertainty": [],
            "needs_more_evidence": False,
        },
        clip_duration_seconds=4.0,
    )
    assert observed["evidence_timestamps"] == [2.75]
    with pytest.raises(ValueError, match="timestamp"):
        runner.validate_probe(
            {**observed, "evidence_timestamps": [4.5]},
            clip_duration_seconds=4.0,
        )
```

Also assert that `fact_observed=false` requires an empty `observed_fact`, no timestamps, nonempty uncertainty, and `needs_more_evidence=true`; reject extra fields and competition-answer fields.

- [ ] **Step 2: Run the focused test and confirm red**

Run: `pytest -q tests/test_run_dense_question_localized_evidence.py`

Expected: import or collection failure because the runner does not exist.

- [ ] **Step 3: Implement the exact probe and arm contract**

The runner must build a strict JSON-schema request with the video part first and prompt second. The prompt identifies the clip's source offset, asks only whether the requested visual fact is observable, requires clip-relative timestamps, and prohibits references, previous answers, guessing, answer candidates, purpose, exact speech, OCR inference, and counts. `EXPECTED_FPS = (4.0, 12.0)` and `MAX_CALLS = 24` are constants bound by the treatment freeze.

- [ ] **Step 4: Write failing artifact-chain and authorization tests**

```python
def test_run_writes_one_terminal_chain_per_pair_without_retry(tmp_path):
    root = write_authorized_fixture(tmp_path, rows=2)
    calls = []
    runner.run(root, call=lambda *args: calls.append(args) or valid_probe_call())
    assert len(calls) == 4
    assert len(list((root / "rows").glob("*/terminal.json"))) == 4
    runner.verify_predictions_freeze(root)


def test_run_rejects_label_bearing_provider_rows_before_call(tmp_path):
    root = write_authorized_fixture(tmp_path, rows=2)
    add_forbidden_field(root / "provider-rows.json", "reference", "hidden")
    calls = []
    with pytest.raises(ValueError, match="provider-safe"):
        runner.run(root, call=lambda *args: calls.append(args))
    assert calls == []
```

Cover malformed JSON preservation, schema failure, missing raw response, one-attempt enforcement, call-cap drift, FPS drift, selection/treatment/authorization hash drift, mismatched clip bytes, and write-once refusal.

- [ ] **Step 5: Implement raw-first storage and prediction freezing**

Each `rows/{audit_index:02d}-{role}-{fps}/` directory contains exactly `request.json`, `raw-response.txt`, `metadata.json`, optional `parsed.json`, and `terminal.json`. The prediction freeze binds all five possible artifacts per arm plus the treatment, selection, provider-row, media-manifest, authorization, runner, config, prompt, schema, and validator hashes.

- [ ] **Step 6: Verify and commit the treatment implementation**

Run:

```bash
pytest -q tests/test_run_dense_question_localized_evidence.py
ruff check scripts/run_dense_question_localized_evidence.py tests/test_run_dense_question_localized_evidence.py
python -m py_compile scripts/run_dense_question_localized_evidence.py
git diff --check
```

Expected: all commands exit 0.

Commit:

```bash
git add configs/vqa/dense_question_localized_evidence_issue35.json scripts/run_dense_question_localized_evidence.py tests/test_run_dense_question_localized_evidence.py
git commit -m "feat: freeze dense localized evidence probe"
```

### Task 2: Form the modality panel and freeze identical local clips

**Files:**
- Create: `scripts/prepare_dense_question_localized_evidence.py`
- Create: `tests/test_prepare_dense_question_localized_evidence.py`
- Create private: `data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1/**`

**Interfaces:**
- Produces CLI phases `freeze-treatment`, `prepare-modality-audit`, `freeze-panel`, and `authorize`.
- Produces private `treatment-freeze.json`, `modality-audit-template.json`, `manual-modality-audit.json`, `selection-freeze.json`, `provider-rows.json`, `media-manifest.json`, `clips/*.mp4`, and `provider-authorization.json`.
- `freeze-panel` consumes the completed `manual-modality-audit.json`, full-panel `protocol.json`, corrected Issue #35 audit, and exact Task 1 treatment hash.

- [ ] **Step 1: Write failing treatment-order and modality-completeness tests**

```python
def test_modality_template_requires_committed_treatment_freeze(tmp_path):
    fixture = write_source_fixture(tmp_path)
    with pytest.raises(ValueError, match="treatment freeze"):
        prep.prepare_modality_audit(**fixture)


def test_freeze_panel_requires_all_17_insufficiency_rows_classified(tmp_path):
    fixture = write_source_fixture(tmp_path, insufficient_rows=17)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    fill_only_first_n_rows(fixture["manual_audit"], 16)
    with pytest.raises(ValueError, match="17"):
        prep.freeze_panel(**fixture)
```

- [ ] **Step 2: Run the preparation tests and confirm red**

Run: `pytest -q tests/test_prepare_dense_question_localized_evidence.py`

Expected: import or collection failure because the preparation script does not exist.

- [ ] **Step 3: Implement treatment freeze and complete modality template**

`freeze-treatment` binds the committed Task 1 runner/config/prompt/schema/validator bytes, model settings, FPS arms, call cap, source protocol/maps, corrected Issue #35 audit, and decision gate. `prepare-modality-audit` emits all 17 insufficiency rows and all 8 controls privately with blank `modality`, `rationale`, `map_line_numbers`, `window_start_seconds`, `window_end_seconds`, and `include` fields.

- [ ] **Step 4: Complete and validate the label-aware modality audit**

Classify each insufficiency row as exactly one of `brief_visual_action`, `speech_or_audio`, `onscreen_text`, `fine_spatial_detail`, or `genuinely_absent_or_ambiguous`. Mark controls visually eligible or ineligible. Record a 2–6 second window and map-line rationale for every included row. Do not stop early and do not exclude on exposure.

- [ ] **Step 5: Write failing deterministic selection and media tests**

```python
def test_selection_uses_first_eight_visual_targets_and_first_four_controls(tmp_path):
    fixture = write_completed_modality_fixture(tmp_path, visual_targets=9, controls=5)
    prep.freeze_panel(**fixture)
    selection = load(fixture["root"] / "selection-freeze.json")
    assert selection["target_count"] == 8
    assert selection["control_count"] == 4
    assert selection["planned_call_count"] == 24


def test_both_arms_bind_the_same_clip_bytes(tmp_path):
    fixture = write_completed_modality_fixture(tmp_path, visual_targets=2, controls=2)
    prep.freeze_panel(**fixture)
    rows = load(fixture["root"] / "provider-rows.json")["rows"]
    assert all(row["arms"]["4fps"]["clip_sha256"] == row["arms"]["12fps"]["clip_sha256"] for row in rows)
```

Also test the two-target/two-control minimum, 24-call maximum, exact audit ordering, invalid windows, source SHA drift, map-line drift, missing media, output clips over 100 MB, exposure reporting, and private/provider projection separation.

- [ ] **Step 6: Implement accurate MP4 clip generation**

Use ffmpeg with input seeking, exact duration, video decode to H.264 lossless (`libx264`, `-crf 0`, `yuv420p`), AAC audio at 192 kbps when present, and `+faststart`. Run ffprobe afterward and bind duration, codecs, source hash, clip hash, and command policy. Create one clip per row; the arms reference that same clip and differ only in requested FPS metadata.

- [ ] **Step 7: Verify and commit preparation code before real selection**

Run:

```bash
pytest -q tests/test_prepare_dense_question_localized_evidence.py tests/test_run_dense_question_localized_evidence.py
ruff check scripts/prepare_dense_question_localized_evidence.py tests/test_prepare_dense_question_localized_evidence.py
python -m py_compile scripts/prepare_dense_question_localized_evidence.py
git diff --check
```

Expected: all commands exit 0. Never stage private artifacts.

Commit:

```bash
git add scripts/prepare_dense_question_localized_evidence.py tests/test_prepare_dense_question_localized_evidence.py
git commit -m "feat: prepare dense localized evidence panel"
```

- [ ] **Step 8: Freeze the real panel or preserve a zero-call result**

Run the four preparation phases against:

- source protocol: `data/vqa/benchmarks/direct-gemini-batch-475-v1/protocol.json`;
- frozen maps: `data/vqa/benchmarks/direct-gemini-batch-475-v1/maps.question-blind.json`;
- corrected audit: `data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/manual-audit-corrected-v3.json`.

If fewer than two credible visual targets or two controls remain, write a frozen feasibility failure and skip `authorize`. Otherwise generate `provider-authorization.json` binding the approved 24-call maximum and every current freeze/hash.

### Task 3: Execute and freeze the paired Gemini probe

**Files:**
- Create private: additional immutable files under `data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1/`
- Modify: `docs/superpowers/plans/2026-08-16-dense-question-localized-evidence.md`

**Interfaces:**
- Consumes Task 2's frozen treatment, selection, clips, provider rows, and authorization.
- Produces one terminal artifact chain per row-arm and `predictions-freeze.json` before any reference-aware scoring.

- [ ] **Step 1: Perform the cheapest pre-call validation**

Verify the dataset symlink resolves to `/home/npatta01/data/competitions/trec_video_2026/data`; all source/clip hashes match; requests contain none of `reference`, `direct_answer`, `verdict`, `score`, or prior candidate text; every pair shares its clip hash; planned calls are at most 24; and the environment contains `GEMINI_API_KEY` without printing it.

- [ ] **Step 2: Run one representative mocked end-to-end test**

Run the real frozen protocol through the runner with the injected fake provider and a temporary output root. Verify request ordering, complete artifact chains, raw-first malformed-response handling, and prediction-freeze verification.

- [ ] **Step 3: Issue the authorized paired calls**

Run:

```bash
python scripts/run_dense_question_localized_evidence.py run \
  --root data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1
```

Do not retry any terminal failure. Record progress by completed pairs and total calls, not by repeating unchanged status.

- [ ] **Step 4: Freeze predictions before references**

Run the runner's `freeze` and `verify-freeze` phases. Confirm the exact completion denominator, usage/cost metadata, model version, requested FPS values, raw-response hashes, and zero duplicate attempts. Only after this succeeds may Task 4 load references.

- [ ] **Step 5: Record the execution boundary**

Add exact call count, terminal statuses, cost, prediction-freeze SHA-256, and whether Task 4 is authorized to this plan. Commit only the plan update; never stage private rows or clips.

#### Task 3 execution boundary — 2026-08-16

- Status: execution complete with a frozen provider failure; the only execution root was `data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1/run-v2`.
- Pre-call validation: PASS. The dataset symlink resolved to `/home/npatta01/data/competitions/trec_video_2026/data`; Task 1 input validation passed for 12 rows (8 targets, 4 controls); all 12 source hashes and 12 clip hashes matched; all 12 pairs bound the same clip hash; selected map-line projections matched the frozen audit for 12/12 rows; provider-safe request structure was label-free for 24/24 planned arms with video first and prompt second; planned calls were 24/24 under the cap; `GEMINI_API_KEY` was present without being printed; existing attempts and provider calls were zero.
- Mocked end-to-end validation: PASS. A temporary-root run exercised all 24 arms in exact 4-FPS/12-FPS pair order, preserved one malformed response raw-first, produced complete terminal chains, and passed prediction-freeze verification (mock freeze SHA-256 `3dab5531dd0ba873e8d12dee25b8c1f7117ab446d6bcd35dd492d659e84d7b05`).
- Authorized provider execution: exactly 24 terminal calls, one per arm, with no retry; terminal statuses were `provider_error=24`.
- Provider boundary concern: all calls returned HTTP 400 `INVALID_ARGUMENT` before a model response because Gemini rejected `additional_properties` in the submitted response schema. No row content was inspected or recorded, and no retry was made.
- Model/settings: requested model `gemini-3.1-pro-preview`; requested FPS values `[4.0, 12.0]`; frozen settings were temperature `0.0`, seed `42`, HIGH thinking, HIGH media resolution, candidate count `1`, maximum output tokens `2048`, retry count `0`. Provider model version and usage metadata were unavailable because every call failed at request validation; cost was unavailable.
- Prediction freeze: `status=frozen`, `row_count=24`, `completed_count=24`, `call_count=24`, `passed_count=0`, `failed_count=24`, terminal status counts `provider_error=24`; SHA-256 `feaf6f1fab8925e798b647b2a5926a0c3026057ed85a1eb31ec494ea0ffe3528`. `freeze` and `verify-freeze` both passed; no references or score outputs were loaded.
- Task 4 authorization: NOT AUTHORIZED. The frozen prediction set is complete but unusable for the recovery gate because no provider arm produced a parsed observation; Task 4 must not load references or score this run.

#### Task 3A schema-compatibility repair and additive v3 boundary — 2026-08-16

- Status: treatment repair and zero-call additive v3 preparation complete. The v2 evidence remains immutable and independently verifies with its original runner/config bytes.
- Aggregate v2 failure fact: 24/24 authorized attempts terminated `provider_error` before a model response with uniform HTTP 400 `INVALID_ARGUMENT`; prediction freeze SHA-256 remains `feaf6f1fab8925e798b647b2a5926a0c3026057ed85a1eb31ec494ea0ffe3528`.
- Repair: the strict local response contract remains unchanged for `validate_probe`; provider requests recursively omit only `additionalProperties`. The committed provider-schema SHA-256 is `26322157cd6879c7576b7e1ad0099afbc17be6fa2c024c7fcb18052cb7cf8498`; compatibility canary policy is exactly `1` pair.
- Additive v3 authorization: panel aggregate is 8 target rows and 4 control rows, 24 planned arms, 24-call maximum, 4-FPS then 12-FPS pair ordering, first pair as the compatibility canary, no retries, and `provider_calls_made=0`. The provider authorization SHA-256 is `eecb3bc5bc9b5c41a36076f06215d07eee816cd363bcb527c2f9510672874a43`; treatment, selection, provider-row, and media-manifest SHA-256 values are `116732c004cd385947d255c28c9749b56c0366a4b5c5c9fa59b0cd1695f8589a`, `f004ea05bfcbfea2fcfd17021b69ffc188c36fbfc8d4600ff619ed0fc1896f29`, `007fd6a66fd403d88782438c655028c0381a714ef84f70a812ed511d44eed293`, and `00dc65c6d9e96d564ebfbfae056ddcc57ce50b3c1da021514f269d68635d7b6b` respectively.
- Zero-call boundary: Task 1 real input loading passed; all paired clip hashes, source hashes, selected map projections, windows, role counts, label isolation, and FPS arm bindings passed; no provider rows, terminals, raw responses, or prediction freeze were created and no Gemini/provider call was issued. Task 4 is NOT AUTHORIZED until an explicitly authorized provider execution produces a verified prediction freeze.
- Round-1 integrity hardening: freeze verification now derives the complete/failed-canary branch from ordered request and terminal artifacts, requiring a passed first pair for complete execution or exactly two completed canary terminals plus 22 incomplete terminals for canary failure; authorization now requires `provider_calls_made` present and exactly zero. The existing zero-call authorization remains preserved and no provider execution was issued during this repair.

#### Task 3A round-2 additive freeze boundary — 2026-08-16

- V3 status: superseded and non-executable under the hardened runner because its immutable treatment freeze binds the pre-hardening runner bytes. Its private bytes and zero-call authorization remain unchanged.
- V4 status: executable zero-call candidate freshly frozen through the existing preparation pipeline under the hardened runner, using the same frozen upstream panel decisions and media. Aggregate panel is 8 targets and 4 controls, 24 planned arms, 24-call maximum, canary policy `1`, and `provider_calls_made=0`; the existing user authorization covers this same max-24 treatment.
- V4 hashes: treatment freeze `0e418e8a3f25ae055ba64d865a7eddd4a5912ba0b58a71d63acaf9fbfc2592b5`, selection freeze `05eb1f91c5404a7481875b6c3e29c0a9b5f098d4fd9d1488428eab4a54d713c5`, provider-row artifact `007fd6a66fd403d88782438c655028c0381a714ef84f70a812ed511d44eed293`, media manifest `0b0875fec5c83f5567d0ff3b6d1051a3026b938a0ee5c7dbaa90a07023be5b66`, and authorization `c240010ca312e7b3c86c7eecc0f532ed7e47955209e145e2167cafbc75434c09`.
- Hardened binding: current runner `a6c0aaecbdd3d4843033f492b2bc593eaa5e2d7782392f181cf5c5fd426eed34`, config `0d6843d51695251c0fdb4cb75483d61cfd20a2e3496eab6c0a06f5b24e51caab`, and provider schema `26322157cd6879c7576b7e1ad0099afbc17be6fa2c024c7fcb18052cb7cf8498` are bound by v4 treatment. Task 1 input validation passes; no provider execution or output artifacts were created.

### Task 4: Audit paired fact recovery and apply the gate

**Files:**
- Create: `scripts/score_dense_question_localized_evidence.py`
- Create: `tests/test_score_dense_question_localized_evidence.py`
- Create private: `postfreeze-audit-template.json`, `manual-postfreeze-audit.json`, `postfreeze-score.json`, and integrity sidecars under the benchmark root.

**Interfaces:**
- Produces `validate_manual_audit(selection: Mapping[str, Any], predictions: Mapping[str, Any], audit: Mapping[str, Any]) -> list[dict[str, Any]]`, `score(root: Path) -> dict[str, Any]`, and `verify_score(root: Path) -> None`.
- Consumes only a verified prediction freeze, the frozen selection/media chain, corrected references, and a complete manual audit.

- [ ] **Step 1: Write failing audit-consistency and metric tests**

```python
def test_incremental_recovery_requires_4fps_false_and_12fps_true():
    rows = paired_fixture(targets=[(False, True), (True, True)], controls=[(True, True)])
    result = scorer.aggregate(rows)
    assert result["target_4fps_recovered"] == 1
    assert result["target_12fps_recovered"] == 2
    assert result["incremental_recoveries"] == 1


def test_gate_rejects_unsupported_observation_even_with_two_recoveries():
    result = passing_gate_fixture()
    result["unsupported_observations_12fps"] = 1
    assert scorer.classify_gate(result) == "FAIL"
```

Cover incomplete pairs, uninspected rows, ambiguous references excluded from the auditable-target denominator but retained in completion, control regressions, timestamp support, exposure splits, and score-input hash drift.

- [ ] **Step 2: Run the score tests and confirm red**

Run: `pytest -q tests/test_score_dense_question_localized_evidence.py`

Expected: import or collection failure because the scorer does not exist.

- [ ] **Step 3: Implement post-freeze audit generation and validation**

Each row-arm audit records clip fact presence, model fact support, recovery, timestamp support, unsupported observation, ambiguity, and reviewer note. Require every selected row and both arms exactly once. Reject a claimed recovery without a supported observed fact and timestamp.

- [ ] **Step 4: Inspect every real pair and freeze the audit**

Use the private clips, raw responses, parsed probes, question, and reference. Preserve disagreements and questionable references explicitly. Freeze the completed audit before aggregating results.

- [ ] **Step 5: Compute the predeclared gate**

Pass only with complete terminal pairs and inspection, at least two supported `4 FPS false -> 12 FPS true` target recoveries, 12-FPS recovery on at least half of auditable targets, zero unsupported observations in either arm, and zero 12-FPS control regressions. Report 4-FPS and 12-FPS recovery, incremental recoveries, controls, completion, exposure splits, and cost. Do not compute answer metrics.

- [ ] **Step 6: Verify and commit the scorer**

Run:

```bash
pytest -q tests/test_score_dense_question_localized_evidence.py tests/test_prepare_dense_question_localized_evidence.py tests/test_run_dense_question_localized_evidence.py
ruff check scripts/score_dense_question_localized_evidence.py tests/test_score_dense_question_localized_evidence.py
python -m py_compile scripts/score_dense_question_localized_evidence.py
git diff --check
```

Expected: all commands exit 0.

Commit:

```bash
git add scripts/score_dense_question_localized_evidence.py tests/test_score_dense_question_localized_evidence.py
git commit -m "feat: score dense localized evidence recovery"
```

### Task 5: Publish the durable negative-or-positive result privately and verify

**Files:**
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify generated: sibling Markdown and HTML outputs
- Create: `docs/experiments/artifacts/vqa-dense-localized-evidence-20260816/README.md`
- Create: `docs/experiments/artifacts/vqa-dense-localized-evidence-20260816/summary.json`
- Create: `docs/experiments/artifacts/vqa-dense-localized-evidence-20260816/SHA256SUMS`

**Interfaces:**
- Consumes only verified aggregate results and immutable private hashes.
- Produces a sanitized evidence index and updates the Routing theme's next action without exposing row-level competition material.

- [ ] **Step 1: Write the condition-bound result**

Report reason, paired condition, target/control denominator, completion, recovery counts by arm, incremental dense-only recoveries, unsupported observations, control regressions, exposure split, cost, gate, evidence validity, decision disposition, limitations, and reconsideration condition. State clearly that oracle-screened localization cannot establish router performance or prevalence.

- [ ] **Step 2: Generate the sanitized bundle and checksums**

Include only aggregate counts, schema descriptions, hashes, model/settings, call count, cost, and gate. Exclude row IDs, questions, references, clips, paths outside the repository, evidence text, raw responses, and predictions.

- [ ] **Step 3: Render QMD-owned outputs**

Render the dated experiment QMD to GFM and HTML. Render all project-status QMD files using the repository guide's isolated cache commands. Do not hand-edit generated Markdown or HTML.

- [ ] **Step 4: Run complete verification**

Run focused and neighboring tests; Ruff; Python compilation; JSON parsing; private freeze verification; score recalculation; bundle checksums; privacy scans; `git diff --check`; generated-source consistency; link checks; and tailnet-only portal HTTP checks. Then run an independent review of the full branch diff before push or issue update.

- [ ] **Step 5: Commit the result without publishing externally**

Stage only code, tests, canonical QMD, generated outputs, the plan, and sanitized artifacts. Confirm no dataset paths are staged. Commit with a result-specific message. Do not push, update GitHub Issue #35, or alter sharing permissions without explicit user authorization.
