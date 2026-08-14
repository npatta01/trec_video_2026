# Premise-Tolerant Grounded Answering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether separating requested-answer support from harmless question-premise variance repairs the frozen issue #35 false abstentions without creating unsupported answers.

**Architecture:** Add an isolated 11-row runner and scorer that consume only the completed v2 label-free protocol. Preparation writes a provider-safe protocol plus a separate pre-inference gate freeze; inference persists one immutable structured response per row; prediction and evidence freezes precede all annotation, prior-prediction, and manual-audit loads. The existing issue #35 v1/v2 artifacts and code remain unchanged except for reusable read-only helper imports.

**Tech Stack:** Python 3.12, `google-genai`, pytest, JSON, existing METEOR/BERTScore utilities, Quarto.

## Global Constraints

- Use only questions and exact evidence lines already frozen in `grounded-map-answering-issue35-v2/protocol.json`.
- Use Gemini 3.1 Pro Preview, temperature `0`, seed `42`, high thinking, one candidate, one attempt, and no retry.
- Do not use native video/audio, acquire evidence, retrieve new lines, generate multiple answers, implement #34, or create #36.
- Keep the label-aware gate outside every provider request and freeze it before inference.
- Freeze predictions and all per-row artifacts before opening annotations, prior predictions, references, or manual scoring rubrics.
- Preserve provider and contract failures in the denominator and preserve malformed paid responses before classification.
- The final non-verification answer must exactly equal one cited claim and contain at most 24 words on one line.
- Keep complete row content private under the canonical dataset; track only sanitized hashes, counts, metrics, and decisions.
- Preserve a positive or negative result without rerunning failed rows.

---

### Task 1: Freeze the exact label-isolated 11-row protocol and gate

**Files:**
- Create: `configs/vqa/premise_tolerant_grounded_answering_issue35.json`
- Create: `scripts/run_premise_tolerant_grounded_answering.py`
- Create: `tests/test_run_premise_tolerant_grounded_answering.py`

**Interfaces:**
- Consumes: `prepare(source_protocol: Path, selection_path: Path, output: Path) -> None`.
- Produces: private `protocol.json` with only IDs, questions, evidence, model settings, prompt/schema hashes, and source hashes; private `preinference-gate-freeze.json` with roles and thresholds.
- Exact roles: five `fix_false_abstention`, two `detectable_insufficiency`, one `known_limit`, and three `stability_control` rows.

```python
def prepare(source_protocol: Path, selection_path: Path, output: Path) -> None:
    """Freeze provider-safe rows and a separate label-bearing gate before calls."""

PANEL_IDS = {
    "fix_false_abstention": [
        "D_HTq7IiSPo", "KBb0DfsH0eg", "LNb5uNQXkwg",
        "PWP17NlIgs8", "BDKkVYmWqdM",
    ],
    "detectable_insufficiency": ["G2w7PrdwYhM", "LkdfiZcQ7lU"],
    "known_limit": ["HcvOhX3n784"],
    "stability_control": ["Bvv6lMUTlEw", "GWCGF1WpMPc", "JTXpUxVg144"],
}
```

- [x] **Step 1: Write failing preparation tests**

  Add tests that assert the exact 11 IDs from the approved design, reject an incorrect source-protocol SHA, reject missing/duplicate IDs, reject a selection whose roles do not total `5/2/1/3`, and prove `protocol.json` contains no role, expected-route, threshold, reference, baseline, assessment, or label fields.

- [x] **Step 2: Run preparation tests and verify RED**

  Run: `uv run pytest -q tests/test_run_premise_tolerant_grounded_answering.py -k prepare`

  Expected: collection failure because the runner does not exist.

- [x] **Step 3: Implement the minimum selection and preparation path**

  Define `MODEL`, the exact structured schema, `answer_prompt`, source verification, and write-once preparation. Copy exact question/evidence objects from the v2 protocol; never load v2 audits, metrics, annotations, or predictions. Write the gate freeze from the tracked selection in the same preparation command before a `rows/` directory exists.

- [x] **Step 4: Run preparation tests and verify GREEN**

  Run: `uv run pytest -q tests/test_run_premise_tolerant_grounded_answering.py -k prepare`

  Expected: all preparation tests pass.

### Task 2: Enforce the premise-tolerant one-answer contract and immutable calls

**Files:**
- Modify: `scripts/run_premise_tolerant_grounded_answering.py`
- Modify: `tests/test_run_premise_tolerant_grounded_answering.py`

**Interfaces:**
- Produces: `validate_answer(payload, evidence_ids) -> dict`, `run_row(root, video_id, call_fn=None) -> None`, `freeze(root) -> None`.
- Response fields: `question_interpretation`, `premise_status`, `answer_claims`, `conflicts_or_missing_qualifiers`, `needs_verification`, `answer`.

```python
PREMISE_STATUSES = {"supported", "harmless_locator_variance", "answer_blocking"}

def validate_answer(payload: Mapping[str, Any], evidence_ids: set[str]) -> dict[str, Any]:
    """Enforce citations, premise status, verification sentinel, and exact answer binding."""

def run_row(
    root: Path,
    video_id: str,
    *,
    call_fn: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
) -> None:
    """Issue at most one immutable provider call for a prepared row."""
```

- [x] **Step 1: Write failing contract and artifact tests**

  Cover all three `premise_status` values; reject an answer not exactly repeated by a cited claim, unknown citations, claims on verification rows, missing verification reasons, multiline/overlong answers, and unsupported fields. Test that malformed provider JSON writes raw response and metadata before a `contract_error`. Test one-attempt immutability and source/prompt/schema drift rejection.

- [x] **Step 2: Run contract tests and verify RED**

  Run: `uv run pytest -q tests/test_run_premise_tolerant_grounded_answering.py -k 'validate or run_row'`

  Expected: failures for absent validation and execution behavior.

- [x] **Step 3: Implement the minimal provider boundary and row runner**

  Persist `request.json`, `raw-response.txt`, `metadata.json`, `parsed.json`, and `terminal.json`. The prompt must explicitly separate locator/premise assessment from requested-answer support, state the allowed/forbidden entailments from the design, and instruct that `harmless_locator_variance` does not itself require verification. Use exactly one call and no retry.

- [x] **Step 4: Write failing freeze-drift tests**

  Complete all fixture rows with an injected provider, call `freeze`, mutate each of request/raw/metadata/parsed/terminal in parametrized tests, and assert freeze verification rejects every mutation.

- [x] **Step 5: Run freeze tests and verify RED**

  Run: `uv run pytest -q tests/test_run_premise_tolerant_grounded_answering.py -k freeze`

  Expected: failure because prediction/evidence freeze creation or verification is incomplete.

- [x] **Step 6: Implement prediction and evidence freezes**

  Write `predictions-freeze.json` over all 11 terminal hashes, then reuse or call the existing issue #35 evidence-freeze helper to bind every persisted row artifact. Refuse incomplete panels and post-freeze calls.

- [x] **Step 7: Run all runner tests and verify GREEN**

  Run: `uv run pytest -q tests/test_run_premise_tolerant_grounded_answering.py tests/test_run_grounded_map_answering.py`

  Expected: all tests pass.

### Task 3: Score the four panel roles after freeze

**Files:**
- Create: `scripts/score_premise_tolerant_grounded_answering.py`
- Create: `tests/test_score_premise_tolerant_grounded_answering.py`

**Interfaces:**
- Consumes: verified v3 protocol/freeze, v2 terminal predictions, direct baseline predictions, training annotations, BERT model, and an optional post-freeze manual audit.
- Produces: `load_frozen_predictions(root)`, `classify_gate(changed_rows, audit, gate)`, and `score(...)` with automatic condition metrics, role-level outcomes, changed rows, known-limit observation, and decision gate.

```python
def load_frozen_predictions(root: Path) -> list[dict[str, Any]]:
    """Verify protocol, gate, prediction, and evidence freezes before label load."""

def classify_gate(
    treatment_rows: Sequence[Mapping[str, Any]],
    manual_audit: Mapping[str, Any],
    gate_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply 4/5 fixes, 2/2 insufficiencies, 3/3 controls, and zero unsupported."""
```

- [x] **Step 1: Write failing frozen-load and label-isolation tests**

  Assert that scorer rejects changed row artifacts, protocol/gate/prediction hash drift, question mismatch after trailing-whitespace normalization, and missing v2/direct source hashes. Prove references and prior predictions are loaded only after both freezes verify.

- [x] **Step 2: Run frozen-load tests and verify RED**

  Run: `uv run pytest -q tests/test_score_premise_tolerant_grounded_answering.py -k frozen`

  Expected: collection failure because the scorer does not exist.

- [x] **Step 3: Implement frozen loading and descriptive metrics**

  Score direct baseline, v2 grounded answer, and v3 premise-tolerant answer on the same 11 rows. Compute METEOR and BERTScore-F1; record STS/composite as `null` with the already verified service-unavailable reason. Preserve operational/contract failures as zero automatic metrics. Report per-role and all-row aggregates without using them as the decision gate.

- [x] **Step 4: Write failing manual-gate tests**

  Use literal rows to prove: `4/5` fix rescues passes that component; `1/2` insufficiency verifications fails; a known-limit outcome never changes pass/fail; any stability regression fails; any unsupported final answer fails; and the audit must cover exactly every normalized v2-to-v3 changed row plus all role rows requiring adjudication.

- [x] **Step 5: Run gate tests and verify RED**

  Run: `uv run pytest -q tests/test_score_premise_tolerant_grounded_answering.py -k gate`

  Expected: failures for absent role-aware classification.

- [x] **Step 6: Implement the role-aware gate**

  Accept exact audit enums per role: `rescued/not_rescued`, `correct_verification/unsafe_answer`, `observed_answer/observed_verification`, and `preserved/regressed`. Require citation support for every answered row and a nonempty note for every audit row. Apply the exact `4/5`, `2/2`, `3/3`, zero-unsupported, full-inspection gate.

- [x] **Step 7: Run scorer and neighboring tests and verify GREEN**

  Run: `uv run pytest -q tests/test_score_premise_tolerant_grounded_answering.py tests/test_score_grounded_map_answering.py tests/test_score_vqa_semantic_panel.py`

  Expected: all tests pass.

### Task 4: Execute and freeze the 11 one-attempt calls

**Files:**
- Private create: `data/vqa/benchmarks/grounded-map-answering-issue35-premise-v3/**`

**Interfaces:**
- Produces the complete private protocol, pre-inference gate, 11 immutable terminal rows, prediction/evidence freezes, automatic score, manual audit, and final post-freeze metrics.

- [x] **Step 1: Prepare and inspect the provider-safe protocol**

  Run preparation. Verify exact source hashes, 11 unique IDs, no label-bearing protocol fields, role totals `5/2/1/3` only in the gate freeze, and no row artifacts before the first call.

- [x] **Step 2: Run one operational canary and inspect its contract**

  Run one false-abstention row. Verify one candidate, expected model/settings, raw/metadata preservation, exact answer-to-claim binding or valid verification route, and no reference/prior-answer content in the request.

- [x] **Step 3: Complete the remaining ten rows without retry**

  Preserve provider/contract failures as terminal. Do not edit the prompt/schema, acquire evidence, or rerun any row.

- [x] **Step 4: Freeze before labels and verify every artifact hash**

  Write prediction/evidence freezes, record their SHA-256 values, and prove a second provider call is rejected.

- [x] **Step 5: Generate automatic metrics, audit every required row, and finalize the gate**

  Score once without manual labels, author the exact private audit after prediction freeze, then score again. Preserve the result regardless of outcome.

### Task 5: Preserve the result and update issue #35 surfaces

**Files:**
- Create: `docs/experiments/artifacts/vqa-premise-tolerant-grounded-answering-20260814/README.md`
- Create: `docs/experiments/artifacts/vqa-premise-tolerant-grounded-answering-20260814/summary.json`
- Create: `docs/experiments/artifacts/vqa-premise-tolerant-grounded-answering-20260814/SHA256SUMS`
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify: `docs/project-status/research-routing.qmd`
- Modify if the queue changes: `docs/project-status/experiment-ledger.qmd`
- Generated by Quarto only: sibling Markdown and HTML reports.

**Interfaces:**
- The dated QMD remains the detailed record; routing remains the primary-theme claim; the new sanitized index contains no questions, references, evidence text, predictions, raw responses, or media.

- [x] **Step 1: Write the sanitized result bundle**

  Record panel roles, completion denominator, source/prompt/schema/freeze/audit/metric hashes, automatic metrics, exact gate counts, known-limit observation, validity boundary, and disposition. Keep full content private.

- [x] **Step 2: Update QMD sources with observed result and interpretation**

  State the exact tested condition, distinguish frozen-gate observations from interpretation, correct the pencil-control audit without overwriting v2, and declare the next action conditioned on pass/failure. Preserve a negative result.

- [x] **Step 3: Regenerate all affected Markdown and HTML**

  Render the dated experiment and project-status surfaces from QMD using an isolated Quarto cache. Remove only newly generated duplicate output directories, trim generator-added trailing whitespace mechanically when required by `git diff --check`, and never hand-edit generated content.

### Task 6: Verify, independently review, and update the existing PR/issue

**Files:**
- Verify every tracked change on `codex/issue-35-grounded-map-answering`.

**Interfaces:**
- Updates draft PR #37 and issue #35 only after the result and review are durable.

- [x] **Step 1: Run proportional verification**

  Run focused/neighboring tests, Ruff, `py_compile`, JSON/checksums, freeze verification, privacy/credential scans, `git diff --check`, QMD regeneration checks, and the full test suite. Distinguish baseline failures from regressions with an exact base-commit reproduction when needed.

- [x] **Step 2: Obtain independent review and fix Critical/Important findings**

  Review code, protocol isolation, private/tracked artifact hashes, every manual row verdict, report claims, and scope boundaries. Rerun affected checks after fixes.

- [x] **Step 3: Commit and push the complete follow-up**

  Stage only the focused code, tests, config, spec/plan, sanitized artifacts, QMD, and generated reports. Never stage `data` or private artifacts. Push to the existing issue #35 branch.

- [x] **Step 4: Update draft PR #37 and issue #35**

  Post completion, corrected panel interpretation, automatic metrics, exact gate outcome, known-limit result, durable links, verification, and next action. Keep the issue open unless the durable handoff fully resolves its scope.
