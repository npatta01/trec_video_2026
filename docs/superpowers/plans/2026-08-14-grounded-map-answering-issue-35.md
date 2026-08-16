# Grounded Map Answering Issue 35 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether one constrained text-only Gemini answer can repair frozen direct-native-AV failures when exact evidence from the frozen question-blind map is supplied and cited.

**Architecture:** Treat GitHub issue #35 as the approved experiment specification. A deterministic preparation step validates the 12 audited map-sufficient failures, pairs them with 12 same-stratum supported controls, and freezes exact map lines before inference. A one-attempt structured-output runner sends only question plus those lines to Gemini 3.1 Pro; a separate post-freeze scorer joins references, computes the existing semantic metrics, applies manual rescue/regression verdicts, and emits private evidence plus a sanitized tracked summary.

**Tech Stack:** Python 3.12, `google-genai`, pytest, existing NLTK/BERTScore/UMBC STS scorer, JSON, Quarto.

## Completed outcome

- The invalid schema preflight is preserved under
  `data/vqa/benchmarks/grounded-map-answering-issue35-v1/`; it returned no
  answer candidate and was not retried or relabeled.
- The completed 24/24 one-attempt run is preserved under
  `data/vqa/benchmarks/grounded-map-answering-issue35-v2/`, with immutable
  `terminal.json` rows, the original prediction freeze, and an additive
  evidence freeze binding requests, raw responses, metadata, parsed responses,
  and terminals.
- The predeclared gate failed with 5/12 target rescues, two control regressions,
  and seven unnecessary verification routes. Independent post-freeze review
  corrected two source map-sufficiency labels, yielding 5/10 rescues
  descriptively among confirmed answer-sufficient targets; the frozen 5/12
  gate is unchanged.
- The original and corrected audit/metric artifacts are both preserved. STS
  and composite remain explicitly unavailable rather than substituted.
- Decision disposition: **Not promoted under the tested condition**. The next
  possible test is one explicit question-driven temporal or speech
  verification operation over frozen context, with answering evaluated
  separately—not #34, #36, or another generic prompt rewrite.

## Global Constraints

- Use only the existing 475-row question-blind maps, direct predictions, hard-row manual audit, and frozen annotations/metrics.
- No native-video calls, new evidence acquisition, multiple candidate answers, reranking, counting specialist (#34), or shared router framework (#36).
- Use one deterministic text-only call per row with no retry; preserve operational failures in the denominator.
- Freeze the panel, exact evidence lines, prompt/schema hashes, and then predictions before any post-treatment label/reference load.
- A non-verification answer must cite supplied evidence IDs; unsupported or conflicting rows must return `needs_verification: true` and `NEEDS VERIFICATION` rather than guess.
- Report METEOR, BERTScore-F1, STS, their mean composite, target rescues, control regressions, and every changed row.
- Promotion requires meaningful rescue on the 12 target rows, no more than one control regression, valid citations for every answered row, and no invented answer on a row routed to verification.
- Keep complete evidence private under the canonical shared dataset; commit only sanitized summaries, hashes, code, tests, and QMD/report outputs.
- Preserve a valid negative result with disposition **Not promoted under the tested condition**.

---

### Task 1: Freeze the 24-row protocol and oracle evidence selections

**Files:**
- Create: `configs/vqa/grounded_map_answering_issue35.json`
- Create: `tests/test_run_grounded_map_answering.py`
- Create: `scripts/run_grounded_map_answering.py`

**Interfaces:**
- Consumes: full-run `protocol.json`, `maps.question-blind.json`, `predictions.direct-answer.json`, and `failure-audit-v1/manual-review.json`.
- Produces: `prepare(source_root: Path, selection_path: Path, output: Path) -> None` and a write-once private `protocol.json` with 12 `target` and 12 `control` rows, one-to-one pair IDs, exact evidence text, source hashes, and no references.

- [x] **Step 1: Write failing preparation tests**

  Add literal fixtures proving preparation rejects a target not marked incorrect/partial and map-sufficient, rejects duplicate/missing controls, rejects evidence selectors that do not resolve to exact frozen map lines, and writes a 24-row protocol with no label-bearing keys.

- [x] **Step 2: Run the preparation tests and confirm RED**

  Run: `uv run pytest -q tests/test_run_grounded_map_answering.py`

  Expected: collection/assertion failure because `scripts/run_grounded_map_answering.py` does not exist.

- [x] **Step 3: Implement the minimum preparation path**

  Use exact 1-based nonblank map-line indexes from the tracked selection file. Re-read and hash all four source artifacts, verify the audit-derived target set exactly, verify role/pair uniqueness, assign stable `E01...` IDs, and write the protocol once. Do not load annotations or score files.

- [x] **Step 4: Run the preparation tests and confirm GREEN**

  Run: `uv run pytest -q tests/test_run_grounded_map_answering.py`

  Expected: all preparation tests pass.

### Task 2: Enforce one evidence-citing top-1 answer and immutable prediction freeze

**Files:**
- Modify: `tests/test_run_grounded_map_answering.py`
- Modify: `scripts/run_grounded_map_answering.py`

**Interfaces:**
- Produces: `answer_prompt(question: str, evidence: Sequence[Mapping[str, str]]) -> str`, `validate_answer(payload: Mapping[str, Any], evidence_ids: set[str]) -> dict[str, Any]`, `run_row(root: Path, video_id: str) -> None`, and `freeze(root: Path) -> None`.
- Structured answer fields: `question_interpretation`, `answer_claims[{claim,evidence_ids}]`, `conflicts_or_missing_qualifiers[]`, `needs_verification`, and `answer`.

- [x] **Step 1: Write failing contract and freeze tests**

  Cover unknown citations, uncited non-verification answers, verification rows with invented answers, multiline/overlong top-1 answers, changed protocol inputs, incomplete panels, and changed post-freeze row content. The fake provider boundary returns one complete structured Gemini-shaped response but no test asserts on mock call counts.

- [x] **Step 2: Run the focused tests and confirm RED**

  Run: `uv run pytest -q tests/test_run_grounded_map_answering.py`

  Expected: failures for the absent validation, row execution, and freeze behavior.

- [x] **Step 3: Implement minimal structured inference and freeze behavior**

  Call `gemini-3.1-pro-preview` with temperature `0.0`, seed `42`, high thinking, JSON MIME/schema, and only the frozen question/evidence prompt. Record request, raw response, provider metadata, parsed response, and one `terminal.json`; never retry or include direct predictions/references. Freeze all 24 terminal hashes plus protocol hash in `predictions-freeze.json`, then bind all row artifacts in `evidence-freeze.json`.

- [x] **Step 4: Run focused and neighboring runner tests**

  Run: `uv run pytest -q tests/test_run_grounded_map_answering.py tests/test_run_direct_gemini_scope_control.py tests/test_run_direct_gemini_synthesis_treatment.py`

  Expected: all pass.

### Task 3: Score only after freeze and encode the predeclared gate

**Files:**
- Create: `tests/test_score_grounded_map_answering.py`
- Create: `scripts/score_grounded_map_answering.py`

**Interfaces:**
- Consumes: verified private freeze, frozen direct predictions, annotations, existing frozen score metadata, manual changed-row audit JSON, and pinned BERT model.
- Produces: `load_frozen_predictions(root: Path)`, `classify_gate(rows, manual_audit)`, and `score(...) -> dict[str, Any]` with target/control aggregates, paired deltas, rescue/regression counts, verification routing, and changed-row coverage.

- [x] **Step 1: Write failing scorer tests**

  Prove that a mutated frozen row is rejected, treatment questions must match annotations, every changed row needs exactly one manual verdict, unchanged rows cannot be silently added to changed-row inspection, a target rescue and a control regression are counted by role, and promotion fails above one control regression or with invalid citations/unsupported guesses.

- [x] **Step 2: Run the scorer tests and confirm RED**

  Run: `uv run pytest -q tests/test_score_grounded_map_answering.py`

  Expected: collection/assertion failure because the scorer is absent.

- [x] **Step 3: Implement the minimum post-freeze scorer**

  Reuse the pinned METEOR/BERTScore/STS implementations and semantic composite definition. Score direct baseline and grounded treatment on the same 24 rows, split target/control results, report paired metric deltas, and keep manual rescue/regression assessment separate from automatic metrics. Write outputs only after freeze verification.

- [x] **Step 4: Run scorer tests and the semantic-scoring regression tests**

  Run: `uv run pytest -q tests/test_score_grounded_map_answering.py tests/test_score_direct_gemini_scope_control.py tests/test_score_vqa_semantic_panel.py`

  Expected: all pass.

### Task 4: Execute and freeze the smallest valid experiment

**Files:**
- Private preserve: invalid preflight
  `data/vqa/benchmarks/grounded-map-answering-issue35-v1/**`
- Private create: completed run
  `data/vqa/benchmarks/grounded-map-answering-issue35-v2/**`

**Interfaces:**
- Produces immutable private `protocol.json`, per-row
  `request.json`/`raw-response.txt`/`metadata.json`/`parsed.json`/`terminal.json`,
  `predictions-freeze.json`, additive `evidence-freeze.json`, and the preserved
  original plus corrected `changed-row-audit*.json` and
  `postfreeze-metrics*.json` records.

- [x] **Step 1: Prepare and inspect the label-free protocol**

  Run `prepare`, verify 24 unique rows, 12 targets/12 controls, 24 exact pair memberships, no label-bearing keys, all source hashes, and a prompt/schema hash before any provider call.

- [x] **Step 2: Run one cheap provider call as an operational canary**

  Run one target row, validate its structured result and citations, then continue without changing the frozen prompt/protocol.

- [x] **Step 3: Complete all remaining one-attempt text-only rows**

  Preserve any provider/contract failure as terminal. Do not retry or acquire media.

- [x] **Step 4: Freeze predictions before opening references/scoring**

  Run `freeze`; record the freeze SHA-256 and verify every per-row hash.

- [x] **Step 5: Score, inspect every changed row, and rerun scoring**

  Generate automatic metrics, author one manual verdict for every normalized answer change, then regenerate the final score/gate artifact. Preserve the result even when the gate fails.

### Task 5: Preserve durable evidence and update the canonical Routing claim

**Files:**
- Create: `docs/experiments/artifacts/vqa-grounded-map-answering-20260814/README.md`
- Create: `docs/experiments/artifacts/vqa-grounded-map-answering-20260814/summary.json`
- Create: `docs/experiments/artifacts/vqa-grounded-map-answering-20260814/SHA256SUMS`
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify: `docs/project-status/research-routing.qmd`
- Modify if queue/status changed: `docs/project-status/experiment-ledger.qmd`
- Generated by Quarto only: sibling Markdown and HTML report surfaces.

**Interfaces:**
- The dated QMD remains the detailed evidence-of-record; `research-routing.qmd#frontier-boundary-verifier` remains the canonical primary-theme claim.

- [x] **Step 1: Add sanitized evidence index and checksums**

  Include panel boundaries, source/freeze hashes, completion denominator, automatic metrics, manual rescue/control counts, gate result, evidence validity, decision disposition, and private canonical path. Exclude questions, references, raw map text, raw responses, and dataset/media content.

- [x] **Step 2: Update the dated experiment and primary theme**

  Separate observed result from interpretation; state what the selected label-aware mechanism panel can and cannot establish; preserve a negative as **Not promoted under the tested condition**; update the next decision without implementing #34 or #36.

- [x] **Step 3: Regenerate QMD-derived outputs**

  Render the dated QMD to GFM and HTML and regenerate all project-status GFM/HTML surfaces using an isolated Quarto cache. Never hand-edit generated Markdown/HTML.

### Task 6: Verify, commit, report to issue #35, and publish a focused PR

**Files:**
- Verify every tracked change on `codex/issue-35-grounded-map-answering`.

**Interfaces:**
- Produces one focused commit/branch, concise issue result comment, and draft PR only if the evidence bundle/report is valid and reviewable.

- [x] **Step 1: Run full proportional verification**

  Run focused tests, the full test suite if practical, `git diff --check`, JSON parsing/checksums, privacy scans, report link checks, Quarto regeneration diff checks, and responsive/accessibility report checks appropriate to the changed pages.

- [x] **Step 2: Review the complete diff and evidence contract**

  Confirm no dataset file is staged, no secrets/private rows/raw logs leaked, all 24 changed rows are inspected, and the report matches the frozen artifacts.

- [x] **Step 3: Commit and push intentionally**

  Commit the implementation, tests, plan, sanitized artifacts, QMD, and generated outputs with a focused issue-35 message; push the branch.

- [x] **Step 4: Post issue progress/result and open a draft PR**

  Post the completion denominator, metric/gate outcome, condition-bound decision, evidence links, and PR link to issue #35. Do not close the issue unless the result and durable handoff fully satisfy it.
