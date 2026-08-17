# Grouped-Rewrite Fallback Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mechanically avoidable `g37_grouped_unique` fusion fallbacks with provenance-bound grouped rewrites while preserving all 2,443 rank-one answers and official validity.

**Architecture:** A preparation script deterministically repairs the 320 current-schema failures and writes six filtered source inputs for the 290 legacy failures. The existing grouped runner gains subset and per-run-cap CLI support and executes the 290 questions in 58 five-question batches. A finalizer merges fresh results, deterministic repairs, and the existing immutable salvage in that precedence order into a new evidence root and tracked submission.

**Tech Stack:** Python 3.11, pytest, Google GenAI SDK, repository SHA-256/JSON helpers, official TRECVID VQA validator.

## Global Constraints

- Do not overwrite the original A1, A2, A3, salvage, or tracked submission artifacts.
- Preserve the existing fusion rank-one answer byte-for-byte for all 2,443 questions.
- Deterministic repair may use only frozen generated text, exact frozen source answers, and validated fusion answers.
- Fresh inference covers exactly the 290 legacy query IDs, uses five questions per call, permits one attempt per batch, and has a hard $5 new-run cap.
- Reference answers, labels, annotations, video, audio, and transcripts must never enter repair inputs.
- Any unprovable repair uses the exact validated fusion row.
- `.claude/` is unrelated and must remain untracked and unstaged.

---

### Task 1: Deterministic repair and filtered-input preparation

**Files:**
- Create: `scripts/prepare_vqa_grouped_rewrite_fallback_repair.py`
- Create: `tests/test_vqa_grouped_rewrite_fallback_repair.py`

**Interfaces:**
- Consumes: immutable A1/A2/A3/salvage roots and the six source files used by `run_vqa_grouped_rewrite_full.load_rows`.
- Produces: `repair_current_query(query, row) -> dict`, `prepare_repair(...) -> dict`, `deterministic-repairs.json`, `legacy-query-ids.json`, six filtered JSON source files, `manifest.json`, and `summary.json` under a new private preparation root.

- [ ] **Step 1: Write failing tests for the 320 deterministic repairs**

Create literal fixtures showing duplicate removal, exact-source insertion, fusion filling, rank-one preservation, conflict/source drift rejection, and immutable-output refusal. The core assertion is:

```python
repaired = prepare.repair_current_query(raw_query, frozen_row)
assert repaired["answers"][0]["answer"] == frozen_row["rank1_answer"]
assert len(repaired["answers"]) == 10
assert len({" ".join(a["answer"].split()).casefold() for a in repaired["answers"]}) == 10
assert repaired["repair_reason"] == "deduplicated_with_fusion_fill"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_fallback_repair.py`

Expected: collection fails because `prepare_vqa_grouped_rewrite_fallback_repair` does not exist.

- [ ] **Step 3: Implement the minimal deterministic repair**

Implement normalization, source/group validation, rank-one validation, ordered deduplication, insertion of missing exact source answers, removal of the lowest-ranked replaceable answer, and unused-fusion filling. Return `None` instead of weakening an invariant.

```python
def repair_current_query(query: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any] | None:
    groups = validate_group_bindings(query, row)
    answers = ordered_unique_answers(query["answers"])
    answers = ensure_exact_group_hedges(answers, groups, row)
    answers = fill_from_fusion(answers, row["fallback"]["Answers"])
    return validate_repaired_query(answers, groups, row)
```

- [ ] **Step 4: Add failing integration tests for exact real-panel counts and filtered source shape**

Assert the preparation dry-run reports literal counts `320` deterministic candidates, `290` legacy query IDs, and six filtered sources each covering exactly the required query IDs without labels or extra ranked-tail inputs.

- [ ] **Step 5: Run the new integration test and confirm RED**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_fallback_repair.py -k real_panel`

Expected: failure because `prepare_repair` and its manifest do not exist.

- [ ] **Step 6: Implement preparation and immutable evidence writes**

Bind source hashes, old-root inventories, selected query IDs, deterministic decision hashes, and filtered source hashes. Dry-run must not create the output root; `--run` writes once into an absent root.

- [ ] **Step 7: Run focused tests and commit**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_fallback_repair.py && uv run ruff check scripts/prepare_vqa_grouped_rewrite_fallback_repair.py tests/test_vqa_grouped_rewrite_fallback_repair.py`

Commit: `git commit -m "feat: prepare grouped rewrite fallback repair"`

### Task 2: Subset execution and hard per-run cost cap

**Files:**
- Modify: `scripts/run_vqa_grouped_rewrite_full.py`
- Modify: `tests/test_vqa_grouped_rewrite_full.py`

**Interfaces:**
- Consumes: existing `run_full(...)` arguments plus `max_cost_usd: float = 50.0`; CLI adds `--expected-query-count` and `--max-cost-usd`.
- Produces: an immutable 290-query runner root using existing batch/request/response/terminal/final formats.

- [ ] **Step 1: Write failing behavior tests**

Test that a two-row subset can run with `expected_query_count=2`, that `max_cost_usd` is present in the manifest, and that a reservation above the supplied cap fails before any fake-client call. Reject bool, nonfinite, nonpositive, and over-authorized caps.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_full.py -k 'expected_query_count or max_cost'`

Expected: failure because the CLI and manifest do not bind the new cap/count arguments.

- [ ] **Step 3: Implement the minimal runner change**

Thread `max_cost_usd` through dry-run estimation, manifest binding, pre-wave reservation checks, post-call accounting, recovery validation, CLI parsing, and `main`. Keep defaults byte-compatible with the completed A3 contract.

- [ ] **Step 4: Run the complete grouped-runner test file and commit**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_full.py && uv run ruff check scripts/run_vqa_grouped_rewrite_full.py tests/test_vqa_grouped_rewrite_full.py`

Commit: `git commit -m "fix: support bounded grouped rewrite subsets"`

### Task 3: Final repair merger

**Files:**
- Create: `scripts/finalize_vqa_grouped_rewrite_fallback_repair.py`
- Modify: `tests/test_vqa_grouped_rewrite_fallback_repair.py`

**Interfaces:**
- Consumes: original salvage root, preparation root, completed fresh-run root, and the six canonical source files.
- Produces: `finalize_repair(...) -> dict`, a new immutable full-panel root with `final-organizer.json`, `final-provenance.json`, `manifest.json`, and `summary.json`.

- [ ] **Step 1: Write failing merge-precedence and integrity tests**

Use literal fixtures proving fresh-generated beats deterministic, deterministic beats old fallback, existing 1,833 grouped rows remain byte-exact, failed fresh questions fall back safely, and overlapping source query IDs fail closed.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_fallback_repair.py -k finalize`

Expected: import failure because the finalizer does not exist.

- [ ] **Step 3: Implement merge, provenance, and summary**

```python
for row in canonical_rows:
    qid = int(row["query_id"])
    chosen = fresh_generated.get(qid) or deterministic.get(qid) or existing_grouped.get(qid)
    final = chosen if chosen is not None else row["fallback"]
    assert final["Answers"][0]["Answer"] == row["rank1_answer"]
```

Bind all input inventories/hashes, answer hashes, status/reason, source batch IDs, total cost, and fallback count. Validate the complete organizer shape before any final freeze.

- [ ] **Step 4: Run focused tests and commit**

Run: `uv run pytest -q tests/test_vqa_grouped_rewrite_fallback_repair.py && uv run ruff check scripts/finalize_vqa_grouped_rewrite_fallback_repair.py tests/test_vqa_grouped_rewrite_fallback_repair.py`

Commit: `git commit -m "feat: finalize grouped rewrite fallback repair"`

### Task 4: Prepare, canary, and complete the 290-question run

**Files:**
- Private only: `data/vqa/benchmarks/grouped-rewrite-fallback-repair-20260817/`

**Interfaces:**
- Consumes: scripts from Tasks 1–3 and the configured Google API credential.
- Produces: preparation, fresh-run, and final immutable private evidence roots.

- [ ] **Step 1: Run preparation dry-run and live freeze**

Run the preparation script without `--run`, confirm `320/290`, the source hashes, and no output root; then run it once with `--run`.

- [ ] **Step 2: Run the fresh subset dry-run**

Invoke `run_vqa_grouped_rewrite_full.py` with the six filtered source paths, `--expected-query-count 290`, `--max-cost-usd 5`, `--exposed-batch-count 0`, and an empty immutable prior root. Confirm 58 batches, 58 calls, five questions per batch, and an estimate below $5.

- [ ] **Step 3: Execute one fresh batch as a canary**

Run with `--run --stop-after-fresh-batches 1`. Require one terminal batch, five covered query IDs, valid usage/cost, ten answers per query, preserved rank one, no second batch attempt, and no final freeze.

- [ ] **Step 4: Resume the same root to completion**

Run the identical command without `--stop-after-fresh-batches`. Do not retry any terminal or ambiguous batch. Require all 58 terminal batches, balanced attempts/raw evidence, and a complete 290-row final freeze.

- [ ] **Step 5: Finalize the full panel**

Run the finalizer once into an absent repair root. Record deterministic/fresh/fallback counts and verify all cross-hashes.

### Task 5: Release validation, analysis, and tracked artifact update

**Files:**
- Modify: `submissions/trecvid-2026/vqa-ag/g37_grouped_unique/submission.json`
- Modify: `docs/submissions/2026-g37-grouped-unique.md`
- Modify if required by the submission ledger: `docs/submissions/README.md`

**Interfaces:**
- Consumes: final repair root from Task 4.
- Produces: corrected tracked upload artifact, updated ledger/hashes/counts/cost, verification evidence, and PR update.

- [ ] **Step 1: Run official and independent structural validation**

Require 2,443 rows, 10 answers each, exact question/video bindings, ranks 1–10, nonempty answers, finite nonnegative times, official normalized uniqueness, and zero official errors/warnings.

- [ ] **Step 2: Run provenance, privacy, and quality comparison**

Require all hashes and inventories to resolve; labels/reference fields and credentials to be absent; rank one equal on 2,443/2,443; fallback strictly below 610; no regression in stricter duplicate or non-answer rates; and a label-free stratified manual inspection of new fresh and deterministic rows.

- [ ] **Step 3: Update the tracked submission and ledger**

Copy only the validated final organizer file into the existing tracked run-tag path. Update method, counts, cost, SHA-256, provenance, and supersession wording in the ledger without claiming upload or organizer acceptance.

- [ ] **Step 4: Run repository verification**

Run focused tests, Ruff, `git diff --check`, tracked SHA checks, official validator, and a credential/privacy scan. Confirm `.claude/` is unstaged.

- [ ] **Step 5: Commit, push, and verify the draft PR**

Commit: `git commit -m "fix: reduce grouped rewrite fallbacks"`

Push `codex/qwen-aligner-quality`, verify the PR head and checks, and report the new fallback count, submission hash, observed repair cost, and any residual limitation.
