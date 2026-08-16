# Task-support Sufficiency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a single-answer evidence-citing verifier before labels influence panel formation, audit 32 deterministic candidates from all 475 eligible rows, and run a 12-row Gemini diagnostic only if the offline role gate passes.

**Architecture:** A new preparation script owns treatment, candidate, audit, and selection freezes without changing historical issue #35 code. A separate runner owns the exact provider prompt/schema/validator and one-attempt artifact chain. A separate scorer loads references only after predictions freeze, produces role/exposure-aware metrics, and drives sanitized reporting.

**Tech Stack:** Python 3.11, pytest, Google GenAI SDK already used by the repository, JSON write-once artifacts, Quarto QMD/GFM/HTML reporting.

## Global Constraints

- All 475 completed rows remain eligible; never reject a row merely because it was previously inspected.
- Record separate exposure flags for the 30-row hard audit, grounded-answering v2/v3, multi-election, seven-row canary, scope control, and synthesis follow-up.
- Freeze and commit the treatment implementation before loading references or sufficiency verdicts for panel formation.
- Provider inputs contain only video ID, question, and cited frozen-map evidence; never labels, references, direct predictions, scores, or audit verdicts.
- Use `gemini-3.1-pro-preview`, temperature `0.0`, thinking level `HIGH`, maximum output `2048`, one attempt per row, and no multiple candidate answers.
- Answer only when one cited map interpretation uniquely entails the requested entity–fact–qualifier tuple; otherwise emit exactly `NEEDS VERIFICATION`.
- Do not rerun native video, acquire evidence, implement counting issue #34, or add router issue #36.
- Historical grounded, premise, and multi-election runners/configs/tests are immutable evidence and must not be modified.
- Private row content remains under `data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/`; tracked artifacts contain only sanitized counts, hashes, and aggregate results.
- Preserve a failed offline or provider gate as a valid negative result.

---

### Task 1: Frozen single-answer verifier implementation

**Files:**
- Create: `scripts/run_task_support_verifier.py`
- Create: `tests/test_run_task_support_verifier.py`

**Interfaces:**
- Produces: `ANSWER_SCHEMA`, `build_prompt(question: str, evidence: Sequence[Mapping[str, str]]) -> str`, `validate_answer(payload: Any, evidence_ids: set[str]) -> dict[str, Any]`, `run(protocol_root: Path, output_root: Path, call: CallFunction | None = None) -> None`, and `verify_predictions_freeze(root: Path) -> None`.
- Consumes later: `protocol.json`, `treatment-freeze.json`, and `selection-freeze.json` from Task 2.

- [ ] **Step 1: Write failing schema, prompt, and validator tests**

```python
def test_validator_requires_unique_supported_answer_or_verification():
    answered = verifier.validate_answer(
        {
            "question_interpretation": {
                "requested_entity": "the man",
                "requested_fact": "action after sitting",
                "qualifier": "after",
                "answer_unit": "action",
            },
            "cited_evidence": [{"evidence_id": "E01", "claim": "He waves."}],
            "conflicts_or_missing_qualifiers": [],
            "needs_verification": False,
            "answer": "He waves.",
        },
        {"E01"},
    )
    assert answered["answer"] == "He waves."
    with pytest.raises(ValueError):
        verifier.validate_answer({**answered, "cited_evidence": []}, {"E01"})
    with pytest.raises(ValueError):
        verifier.validate_answer(
            {**answered, "needs_verification": True, "answer": "He waves."}, {"E01"}
        )
```

- [ ] **Step 2: Run the focused test and confirm red**

Run: `pytest -q tests/test_run_task_support_verifier.py`
Expected: collection/import failure because the runner does not exist.

- [ ] **Step 3: Implement the exact contract and deterministic validator**

The schema requires exactly the five top-level fields from the design. Reject unknown/missing evidence IDs, empty answered citations, nonempty answer text that does not exactly equal one cited claim, verification without the exact sentinel, verification with answer claims, and answered rows whose conflicts say the requested fact is absent or ambiguous. The prompt must explicitly prohibit labels, reference guessing, purpose/cause/count/speech inference, and multiple candidate answers.

- [ ] **Step 4: Add write-once provider artifact and freeze tests**

```python
def test_run_writes_complete_one_attempt_chain_and_freeze(tmp_path):
    protocol_root = write_protocol_fixture(tmp_path)
    verifier.run(protocol_root, tmp_path / "run", call=fake_valid_call)
    verifier.verify_predictions_freeze(tmp_path / "run")
    assert len(list((tmp_path / "run" / "rows").glob("*/terminal.json"))) == 12
    assert json.loads((tmp_path / "run" / "predictions-freeze.json").read_text())["completed_count"] == 12

def test_run_preserves_malformed_raw_response(tmp_path):
    protocol_root = write_protocol_fixture(tmp_path)
    verifier.run(protocol_root, tmp_path / "run", call=fake_malformed_call)
    row = next((tmp_path / "run" / "rows").iterdir())
    assert (row / "raw-response.txt").read_text() == "{broken"
    assert json.loads((row / "terminal.json").read_text())["status"] == "invalid_json"
```

- [ ] **Step 5: Run tests, Ruff, and compile**

Run: `pytest -q tests/test_run_task_support_verifier.py && ruff check scripts/run_task_support_verifier.py tests/test_run_task_support_verifier.py && python -m py_compile scripts/run_task_support_verifier.py`
Expected: all pass.

- [ ] **Step 6: Commit the frozen runner bytes**

```bash
git add scripts/run_task_support_verifier.py tests/test_run_task_support_verifier.py
git commit -m "feat: add frozen task-support verifier"
```

### Task 2: Treatment-first candidate and audit preparation

**Files:**
- Create: `configs/vqa/task_support_sufficiency_audit_issue35.json`
- Create: `scripts/prepare_task_support_sufficiency_audit.py`
- Create: `tests/test_prepare_task_support_sufficiency_audit.py`

**Interfaces:**
- Produces CLI phases `freeze-treatment`, `prepare-candidates`, and `freeze-selection`.
- Produces private `treatment-freeze.json`, `candidate-sheet.json`, `candidate-sheet-freeze.json`, `audit-template.json`, `manual-audit.json`, `selection-freeze.json`, `protocol.json`, and `selection-integrity.json`.
- Consumes the exact Task 1 runner SHA-256 and source/exposure artifact hashes declared in the config.

- [ ] **Step 1: Write failing treatment-order, all-row, exposure, and label-isolation tests**

```python
def test_prepare_candidates_requires_existing_treatment_freeze(tmp_path):
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    with pytest.raises(ValueError, match="treatment freeze"):
        prep.prepare_candidates(**fixture)

def test_candidate_sheet_keeps_exposed_rows_and_hides_labels(tmp_path):
    fixture = write_full_panel_fixture(tmp_path, row_count=5, exposed={"v01"})
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    rows = load(fixture["output"] / "candidate-sheet.json")["rows"]
    assert len(rows) == 5
    assert next(row for row in rows if row["video_id"] == "v01")["exposure_flags"]
    assert "correct_answer" not in json.dumps(rows)
    assert "direct_prediction" not in json.dumps(rows)
```

- [ ] **Step 2: Run the focused test and confirm red**

Run: `pytest -q tests/test_prepare_task_support_sufficiency_audit.py`
Expected: collection/import failure because the preparation script does not exist.

- [ ] **Step 3: Implement write-once phases and source bindings**

`freeze-treatment` binds runner, prompt, schema, validator, model/settings, gate, full-panel protocol/maps hashes, and config hash. `prepare-candidates` loads all 475 rows, retains exposed rows, writes exposure booleans, hashes annotation files without copying references, and creates deterministic low/control queues. `audit-template.json` contains exactly the first 24 low-score and first 8 control rows with complete frozen maps, references, direct answers, and blank adjudication fields; it is private and never a provider input.

- [ ] **Step 4: Add deterministic queue and selection validation tests**

```python
def test_queue_and_selection_are_deterministic(tmp_path):
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    template = load(fixture["output"] / "audit-template.json")
    assert [row["video_id"] for row in template["rows"][:3]] == ["low-a", "low-b", "low-c"]
    fill_audit_with_four_per_role(fixture["output"] / "manual-audit.json")
    prep.freeze_selection(**fixture)
    selection = load(fixture["output"] / "selection-freeze.json")
    assert selection["role_counts"] == {
        "answer_sufficient_failure": 4,
        "insufficient_or_ambiguous": 4,
        "supported_control": 4,
    }
```

Also test source hash drift, treatment tampering, duplicate IDs, invalid line numbers, fewer-than-four role failure, write-once refusal, and exposure-stratified counts.

- [ ] **Step 5: Run tests, Ruff, and compile**

Run: `pytest -q tests/test_prepare_task_support_sufficiency_audit.py tests/test_run_task_support_verifier.py && ruff check scripts/prepare_task_support_sufficiency_audit.py tests/test_prepare_task_support_sufficiency_audit.py && python -m py_compile scripts/prepare_task_support_sufficiency_audit.py`
Expected: all pass.

- [ ] **Step 6: Commit preparation code and exact config before any label-aware audit**

```bash
git add configs/vqa/task_support_sufficiency_audit_issue35.json scripts/prepare_task_support_sufficiency_audit.py tests/test_prepare_task_support_sufficiency_audit.py
git commit -m "feat: freeze task-support sufficiency audit"
```

### Task 3: Execute the zero-call 32-row sufficiency audit

**Files:**
- Create private: `data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/**`
- Modify: `docs/superpowers/plans/2026-08-15-task-support-sufficiency-audit.md`

**Interfaces:**
- Consumes Task 1 runner and Task 2 config/preparation script.
- Produces a frozen 12-row provider-safe protocol only when all three role quotas reach four.

- [ ] **Step 1: Freeze treatment before candidate adjudication**

Run the `freeze-treatment` phase against the canonical 475-row source and record the treatment hash in this plan's status section. Verify it before opening `audit-template.json`.

- [ ] **Step 2: Generate all-475 candidate and 32-row audit artifacts**

Run `prepare-candidates`. Verify row count `475`, low audit count `24`, control audit count `8`, and that every exposure source is represented as a flag rather than an exclusion.

- [ ] **Step 3: Adjudicate all 32 rows**

For every row, fill requested entity/fact/qualifier/unit, exact evidence line numbers, conflicts, exposure flags, verdict, and note. Compare question, reference, complete map, and direct answer. Never inspect new native-video evidence. Do not stop when quotas are met.

- [ ] **Step 4: Freeze or reject the panel**

Run `freeze-selection`. If fewer than four rows qualify for any role, write `feasibility-result.json` with the exact denominator/counts and stop before provider work. If the gate passes, verify `protocol.json` contains only `video_id`, `question`, and cited evidence while `selection-freeze.json` binds all 32 verdicts and the selected 12 roles.

- [ ] **Step 5: Commit only the updated canonical plan status**

Never stage private artifacts. Record hashes, counts, exposure composition, gate outcome, and next action in the plan; commit that plan update.

#### Superseded v1 Task 3 execution status — 2026-08-15

- Treatment freeze verified before opening the label-aware template: `treatment-freeze.json` SHA-256 `fc850cc44e4108959d58ff1bf31a6b737d5e1c4b9a37e083a9f050f387d642ce`; declared source hashes and settings matched the committed config.
- Candidate preparation retained all `475/475` eligible rows and produced the deterministic `24` low-score plus `8` control audit rows. Exposure remained flag-only: hard audit `30`, grounded v2 `24`, grounded v3 `11`, multi-interpretation `12`, seven-row canary `7`, scope control `7`, synthesis follow-up `7`.
- All `32/32` audit rows were adjudicated from the frozen question/reference/map/direct-answer bundle. Audited role counts: `answer_sufficient_failure=8`, `insufficient_or_ambiguous=9`, `reference_or_question_issue=6` (nonselectable), `supported_control=9`. Manual audit SHA-256: `9ad6cd3e6290df8a5da74ea8bc6c27788e962dbfb8093f5e8010c468fc6858cf`.
- Offline feasibility gate: **PASS**. Frozen selected roles are exactly `4/4/4`; `selection-freeze.json` SHA-256 `1e8ed1bf26f0b7ebe527fb4bd1bf673a3200e682a532ef1119957471576e8c3e` and `protocol.json` SHA-256 `f456883192cf8a231ef32cfaa988492d8284be32bd5fc4d4ddf711bd2a0fadbf`. Provider-row digest: `43810cbafdae377c3ffcf46de381e1289475b8202e6dbfffa3f51c62efd0a42a`.
- Selected exposure composition (audited/selected): grounded v2 `5/1`, grounded v3 `3/1`, hard audit `1/0`, multi-interpretation `5/5`, scope control `0/0`, seven-row canary `0/0`, synthesis follow-up `0/0`. These are reporting flags, not exclusions or fresh generalization strata.
- No provider/API/native-video calls were made. The provider promotion gate is therefore unevaluated; next action is the separately scoped Task 4 run over the frozen 12-row provider-safe protocol.

#### Superseded Task 3 correction round 1 status — exact-definition re-adjudication

- The original `manual-audit.json`, `selection-freeze.json`, `protocol.json`, and `selection-integrity.json` remain byte-for-byte unchanged. The original selection is invalid for provider authorization.
- Additive corrected artifacts: `manual-audit-corrected-v2.json` SHA-256 `d7e092ed4986cc1c63c5ed4cb9be53214268f1cedbb9ed028a4323343672c7a9`, `audit-correction-v2.json` SHA-256 `bf081c0201074b4302c9dde45b24df3f717c84760bc4ad0750b1f29b821e2956`, `corrected-audit-freeze-v2.json` SHA-256 `90d579f77bf153945229cb120300e2260bcab066a1f7e905601323290b2dd473`, and `feasibility-result-v2.json` SHA-256 `b36e8ac1a502bece0dcc1edc37bb64997df555ca7722e44c6ecc99d916ded878`.
- All `32/32` rows were re-adjudicated against the exact predeclared role definitions. Corrected counts: `answer_sufficient_failure=2`, `insufficient_or_ambiguous=16`, `reference_or_question_issue=6` (nonselectable), `supported_control=8`.
- Corrected feasibility gate: **FAIL**. Target quota is `2/4` (deficit `2`); safeguard and control quotas pass at `16/4` and `8/4`. No replacement selection or protocol was created.
- No provider/API/native-video calls were made. Provider authorization is blocked; Task 4 must not run. The reviewer’s MM19feoJGmM long-line impossibility finding is retracted; the row remains a supported control because Task 1 compares the final answer to a cited model-output claim, not the full input evidence line.

#### Task 3 correction round 2 status — current corrected feasibility

- Correction round 2 preserves every v1/v2 private artifact byte-for-byte and changes only `LxZuar32o1o` from `answer_sufficient_failure` to `insufficient_or_ambiguous` because the requested two-hula-hoop count qualifier is absent from the frozen map.
- Current additive artifacts: `manual-audit-corrected-v3.json` SHA-256 `db68cec9be89f4f89305b70578fbd7cb67893e0196c92f93143b5d154db6d3d2`, `audit-correction-v3.json` SHA-256 `4fb41decdca6d2c1df09ac8be2507d28eb02ca5427e607fde2e15ed8e29dcc4a`, `corrected-audit-freeze-v3.json` SHA-256 `45e796683c75cc7f698cbac286e0ba32e006aeae55537e33c6ac6e863f9ae55c`, and `feasibility-result-v3.json` SHA-256 `b639de5dae550f3ba0a5003545b5c35d9b7f9e470d117924d21edd9db8440b1a`.
- All `32/32` rows remain adjudicated. Current counts: `answer_sufficient_failure=1`, `insufficient_or_ambiguous=17`, `reference_or_question_issue=6` (nonselectable), `supported_control=8`.
- Current feasibility gate: **FAIL**. Target quota is `1/4` (deficit `3`); safeguard and control quotas pass at `17/4` and `8/4`. No replacement selection or protocol was created, provider authorization is blocked, and Task 4 must not run.
- The MM19feoJGmM long-line impossibility finding is retracted. No provider/API/native-video calls were made.

### Task 4: Provider run, scoring, and changed-row audit (conditional)

**Files:**
- Create: `scripts/score_task_support_verifier.py`
- Create: `tests/test_score_task_support_verifier.py`
- Create private: additional files under `data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/`

**Interfaces:**
- Runs only if Task 3's feasibility gate passes.
- Produces frozen predictions, `manual-result-audit.json`, and `postfreeze-score.json` with role and exposure aggregates.

- [ ] **Step 1: Write failing scorer tests before provider calls**

```python
def test_metrics_exclude_verification_sentinel_and_split_roles_exposure():
    result = scorer.score_fixture(answered=6, verified=6)
    assert result["metric_row_count"] == 6
    assert result["verification_row_count"] == 6
    assert set(result["roles"]) == {
        "answer_sufficient_failure",
        "insufficient_or_ambiguous",
        "supported_control",
    }
    assert set(result["prior_exposure"]) == {"exposed", "not_exposed"}
```

- [ ] **Step 2: Implement scorer integrity and gate classification**

Internally load and verify treatment, selection, evidence, prediction, request/raw/metadata/parsed/terminal hashes. Load annotations only after prediction freeze verification. Compute METEOR/BERTScore-F1 and comparable cached STS where available only for emitted answers; compute verification accuracy, rescue, control preservation, unsupported answers, role results, exposure results, and the exact predeclared gate.

- [ ] **Step 3: Run scorer tests and static checks**

Run: `pytest -q tests/test_score_task_support_verifier.py tests/test_run_task_support_verifier.py tests/test_prepare_task_support_sufficiency_audit.py && ruff check scripts/run_task_support_verifier.py scripts/prepare_task_support_sufficiency_audit.py scripts/score_task_support_verifier.py tests/test_run_task_support_verifier.py tests/test_prepare_task_support_sufficiency_audit.py tests/test_score_task_support_verifier.py`
Expected: all pass before calls.

- [ ] **Step 4: Run exactly 12 provider calls and freeze predictions**

Use the existing `GEMINI_API_KEY` environment variable without printing it. Confirm one terminal record per row, no retries, and freeze predictions before scoring.

- [ ] **Step 5: Score and inspect every changed row**

Fill manual verdicts for all rows whose final route or answer differs from direct Gemini. Freeze the manual audit, then write the score artifact without overwriting any earlier artifact.

- [ ] **Step 6: Commit scorer code only**

Never stage private outputs. Commit the scorer and tests after focused and neighboring tests pass.

### Task 5: Sanitized artifacts, report, verification, and PR update

**Files:**
- Create: `docs/experiments/artifacts/vqa-task-support-sufficiency-20260815/README.md`
- Create: `docs/experiments/artifacts/vqa-task-support-sufficiency-20260815/summary.json`
- Create: `docs/experiments/artifacts/vqa-task-support-sufficiency-20260815/SHA256SUMS`
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify derived sibling Markdown and HTML reports only by rendering their QMD sources.

**Interfaces:**
- Consumes Task 3 feasibility result and, only if run, Task 4 predictions/score/audit.
- Produces the durable decision record and updated issue #35 PR evidence.

- [x] **Step 1: Write sanitized result bundle**

Include all-475 eligibility, 32-row audit denominator, role counts, exposure counts, gate result, private hashes, provider completion/cost if any, metrics if any, evidence validity, decision disposition, and limitations. Privacy scan must find no row IDs, questions, references, answers, or evidence text.

- [x] **Step 2: Update the owning dated report and routing surfaces**

State observed result separately from interpretation. Preserve a negative result. Explicitly say whether provider calls were authorized, whether the promotion gate passed, and that exposed-row results are not a fresh generalization test.

- [x] **Step 3: Render canonical QMD outputs**

Use the exact render commands from `docs/project-status/adding-experiments.qmd`; do not hand-edit generated Markdown or HTML. Refresh the tailnet-only portal artifact and verify HTTPS 200 plus private Serve status.

- [x] **Step 4: Run complete verification**

Run focused and neighboring pytest suites, Ruff, `python -m py_compile`, `jq empty`, `sha256sum -c`, `git diff --check`, privacy scans, source/derived consistency checks, and an independent consequential review. Record exact commands and outcomes in the plan.

- [x] **Step 5: Commit locally; external push/post not performed**

Commit locally after verification. External push, PR update, and issue post were
not performed because publication was not authorized in this session.

#### Task 5 completion record — 2026-08-15

- **Steps 1–2:** sanitized v3 bundle and the owning dated/routing/ledger QMD
  sources were updated; no row-level content was added.
- **Step 3:** the dated experiment was rendered to GFM and embedded HTML, and
  both project-status QMD sources were rendered to GFM and HTML using the
  commands in `docs/project-status/adding-experiments.qmd`. Derived files match
  their QMD sources.
- **Step 4:** exact verification outcomes were:
  - `pytest -q tests/test_prepare_task_support_sufficiency_audit.py tests/test_run_task_support_verifier.py` — **32 passed**.
  - `ruff check scripts/prepare_task_support_sufficiency_audit.py scripts/run_task_support_verifier.py tests/test_prepare_task_support_sufficiency_audit.py tests/test_run_task_support_verifier.py` — **All checks passed**.
  - `python -m py_compile scripts/prepare_task_support_sufficiency_audit.py scripts/run_task_support_verifier.py` — **passed**.
  - `jq empty` over the sanitized JSON and `sha256sum -c docs/experiments/artifacts/vqa-task-support-sufficiency-20260815/SHA256SUMS` — **passed**.
  - `git diff --check` — **passed**; privacy scan of new tracked additions — **zero** row-ID, question, reference, answer, evidence, or private-payload hits.
  - isolated QMD-to-derived-GFM consistency checks — **passed**.
  - tailnet portal artifact hashes matched the tracked ledger/routing HTML,
    HTTPS returned **200**, and `tailscale serve status` plus
    `tailscale funnel status` remained **tailnet only**.
  - The broader `.venv/bin/pytest -q` check recorded **2,784 passed, 1
    skipped, 29 unrelated baseline/environment failures**; these failures did
    not involve Task 5 files and were not changed.
- **Step 5 commit/status:** the first verified reporting commit was
  `9f5c040` (`docs: record task-support sufficiency gate`); the review-fix
  commit is recorded below after the rerender and narrow recheck. No push or
  external post was made.
