# Video Count Query Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a pure, typed video-count query engine that
returns an exact count only when question interpretations, evidence coverage,
identity, and temporal boundaries support one singleton result.

**Architecture:** `CountQuery` and `EvidenceGraph` are strict label-blind
values. The deep `count_video(query, graph) -> ProofState` module hides a small
algebra for temporal filtering, identity quotienting, cardinality, maximum
concurrency, and event-transition counting. Existing atomic records enter
through a separate adapter; a three-phase private workflow selects a held-out
slice, executes frozen reference-blind queries/graphs, then joins references
only during scoring.

**Tech Stack:** Python 3.12 standard library, frozen dataclasses, pytest,
existing canonical JSON hashing, Ruff, Quarto.

## Global Constraints

- Do not issue provider calls or spend experimental budget.
- Do not rerun or modify the frozen 475-row Gemini panel.
- Reject labels, references, prior answers, correctness, expected counts,
  predictions, metrics, and panel roles recursively from queries and graphs.
- Keep the existing issue #34 negative result unchanged.
- Keep issue #35 and issue #36 out of scope.
- Preserve private row-level artifacts only under the canonical ignored
  benchmark root.
- Add no row-specific query field or executor operator after unblinding.
- Use strict test-first RED/GREEN cycles for every production behavior.

---

### Task 1: Typed query and graph validation

**Files:**

- Create: `src/trec_video_2026/vqa_count_query.py`
- Create: `tests/test_vqa_count_query.py`

**Interfaces:**

- Produces:
  - `parse_count_query(payload: Mapping[str, object]) -> CountQuery`
  - `parse_evidence_graph(payload: Mapping[str, object]) -> EvidenceGraph`
  - immutable `CountQuery`, `QueryInterpretation`, `EvidenceGraph`, node,
    relation, coverage, and provenance values
- Consumes: `validate_label_blind` and `canonical_sha256` from existing modules.

- [x] **Step 1: Write the failing strict-schema test**

  Add a literal one-interpretation query and graph fixture. Assert that parsing
  succeeds and that unknown root fields, duplicate node IDs, dangling part
  owners, invalid intervals, overlapping shots, and a nested
  `reference_answer` key each raise a precise `ValueError`.

- [x] **Step 2: Run the schema test and observe RED**

  Run:

  ```bash
  uv run --extra gemini pytest tests/test_vqa_count_query.py -q
  ```

  Expected: collection fails because `trec_video_2026.vqa_count_query` does not
  exist.

- [x] **Step 3: Implement the immutable domain values and parsers**

  Use frozen dataclasses and explicit enum-value sets. Expose no filesystem or
  network behavior. Require exact fields and finite times. Use payload field
  names `relation_state`, `coverage_state`, and `outcome` so the existing
  label-blind validator continues rejecting generic `status` keys.

- [x] **Step 4: Run the schema tests and observe GREEN**

  Run the Task 1 test file. Expected: all schema tests pass.

- [x] **Step 5: Add canonical serialization behavior**

  Write a failing test that two logically identical inputs produce identical
  `complete_sha256` values and that mutation changes the hash. Add
  `query_to_dict` and `graph_to_dict`, using existing canonical hashing.

- [x] **Step 6: Run the canonicalization test RED, implement, then run GREEN**

  Expected final result: all Task 1 tests pass and `git diff --check` is clean.

---

### Task 2: Deterministic count algebra and proof state

**Files:**

- Modify: `src/trec_video_2026/vqa_count_query.py`
- Modify: `tests/test_vqa_count_query.py`

**Interfaces:**

- Consumes: parsed `CountQuery` and `EvidenceGraph` from Task 1.
- Produces:
  - `count_video(query: CountQuery, graph: EvidenceGraph) -> ProofState`
  - `proof_to_dict(proof: ProofState) -> dict[str, object]`

- [ ] **Step 1: Write the failing maximum-concurrency test**

  Use four entity intervals split across two edits where at most two overlap.
  Assert a `visible_instance + max_concurrent` query returns outcome `exact`,
  bounds `2..2`, exact count `2`, and the IDs forming the peak overlap.

- [ ] **Step 2: Run the single test and observe RED**

  Expected: import or missing-symbol failure for `count_video`.

- [ ] **Step 3: Implement filtering, temporal relations, and interval sweep**

  Implement scope matching for `starts_within`, `overlaps`, and
  `completes_within`; then implement a deterministic endpoint sweep where ends
  at a timestamp are processed before starts at that same timestamp.

- [ ] **Step 4: Run the maximum-concurrency test and observe GREEN**

- [ ] **Step 5: Write failing identity-bound tests**

  Cover:

  - `confirmed_same` reduces distinct physical entities;
  - `confirmed_different` preserves separate classes;
  - unresolved relevant identity yields a lower/upper interval and abstention;
  - cross-shot policy `separate` permits an exact appearance count;
  - a contradictory identity relation is rejected during graph parsing.

- [ ] **Step 6: Run RED, implement union-find and unresolved bounds, then GREEN**

  The lower bound unions both confirmed-same and unresolved candidate pairs;
  the upper bound unions confirmed-same only. Exact requires equal bounds.

- [ ] **Step 7: Write failing event-boundary tests**

  Assert that an event marked `left_censored` is excluded from
  `event_onset + transition_count + starts_within`, included by
  `active_event + cardinality + overlaps`, and forces abstention when onset is
  `unknown` and the result could change.

- [ ] **Step 8: Run RED, implement event projection, then GREEN**

- [ ] **Step 9: Write failing coverage and ambiguity tests**

  Assert:

  - complete matching coverage permits exact zero;
  - missing or partial coverage yields upper bound `null` and
    `incomplete_coverage`;
  - two exact interpretations that disagree yield
    `ambiguous_interpretation`;
  - two exact interpretations that agree return the shared count;
  - exact proof implies a singleton bound and no failed obligations.

- [ ] **Step 10: Run RED, implement proof composition, then GREEN**

- [ ] **Step 11: Add invariant tests and refactor**

  Add literal fixtures proving lower never exceeds upper, adding a confirmed
  observation cannot reduce cardinality, adding confirmed-same cannot increase
  distinct count, and max concurrency cannot exceed interval count. Refactor
  only after all tests are green.

- [ ] **Step 12: Run the complete Task 1-2 suite**

  Run:

  ```bash
  uv run --extra gemini pytest tests/test_vqa_count_query.py -q
  uvx ruff check src/trec_video_2026/vqa_count_query.py tests/test_vqa_count_query.py
  uvx ruff format --check src/trec_video_2026/vqa_count_query.py tests/test_vqa_count_query.py
  ```

---

### Task 3: Existing atomic-evidence adapter

**Files:**

- Create: `src/trec_video_2026/vqa_count_graph_adapter.py`
- Create: `tests/test_vqa_count_graph_adapter.py`

**Interfaces:**

- Consumes:
  - existing `AtomicEvidence`
  - `AtomicGraphPlan` containing graph ID, duration, target kind/predicate,
    visibility, node interpretation, and optional shot list
- Produces:
  - `graph_from_atomic_evidence(
      evidence: AtomicEvidence, plan: AtomicGraphPlan
    ) -> EvidenceGraph`

- [ ] **Step 1: Write the failing adapter test**

  Parse the existing literal atomic fixture and assert its two records become
  two graph nodes with unchanged intervals and provenance source IDs `R1` and
  `R2`.

- [ ] **Step 2: Run the adapter test and observe RED**

- [ ] **Step 3: Implement the minimal adapter**

  The adapter maps records only. It emits no identity relation and no complete
  coverage claim. Event plans map records to event nodes; entity and part plans
  map to the corresponding node types.

- [ ] **Step 4: Run the adapter test and observe GREEN**

- [ ] **Step 5: Write failing safety tests**

  Assert the adapter rejects duration mismatch, invalid target-kind/count-unit
  combinations, and any attempt to declare complete coverage from atomic
  output alone.

- [ ] **Step 6: Run RED, implement validation, then GREEN**

- [ ] **Step 7: Run both new test files plus existing issue #34 tests**

  ```bash
  uv run --extra gemini pytest \
    tests/test_vqa_count_query.py \
    tests/test_vqa_count_graph_adapter.py \
    tests/test_vqa_counting_evidence.py \
    tests/test_build_gemini_counting_panel.py -q
  ```

---

### Task 4: Frozen held-out oracle-slice preparation

**Files:**

- Create: `scripts/build_count_query_oracle_slice.py`
- Create: `tests/test_build_count_query_oracle_slice.py`

**Interfaces:**

- Consumes:
  - private `panel.freeze.json`
  - private label-blind `requests.label-blind.json`
  - frozen design-set IDs from the six-row canary
- Produces under one private output root:
  - `selection.freeze.json`: private 8-failure/4-control selection with bindings
  - `authoring.requests.json`: recursively label-blind 12-row authoring packet
  - `authoring.scaffolds.json`: empty query/graph slots with media bindings

- [ ] **Step 1: Write the failing selection test**

  Use a controlled 26-failure/12-control fixture. Assert selection excludes all
  design-set IDs, chooses exactly eight failures and four controls, balances
  declared question-semantic strata where available, hash-ranks ties, and is
  invariant to input ordering.

- [ ] **Step 2: Run the selection test and observe RED**

- [ ] **Step 3: Implement deterministic selection and self-hashed freezes**

  Bind the panel and request-manifest complete hashes. Require all selected IDs
  to exist exactly once in both inputs. Use write-once atomic files.

- [ ] **Step 4: Run the selection test and observe GREEN**

- [ ] **Step 5: Write the failing leakage test**

  Recursively scan both authoring outputs. Assert they contain no panel role,
  reference, prior answer, prediction, correctness, metric, or selection
  rationale. Assert the private selection freeze contains those fields only
  outside the authoring packet.

- [ ] **Step 6: Run RED, implement the projection, then GREEN**

- [ ] **Step 7: Run the preparation CLI against the canonical frozen panel**

  Write only under:

  `data/vqa/benchmarks/direct-gemini-count-query-engine-v1/`

  Verify 12 rows, input hashes, design-set exclusion, write-once behavior, and
  recursive label blindness. Do not inspect references in the authoring phase.

---

### Task 5: Label-blind execution freeze and post-freeze scorer

**Files:**

- Create: `scripts/run_count_query_oracle_slice.py`
- Create: `scripts/score_count_query_oracle_slice.py`
- Create: `tests/test_run_count_query_oracle_slice.py`
- Create: `tests/test_score_count_query_oracle_slice.py`

**Interfaces:**

- Runner consumes only `authoring.requests.json` plus completed label-blind
  queries/graphs and produces `proofs.freeze.json`.
- Scorer consumes the immutable proof freeze plus `selection.freeze.json` and
  reports exact answered count, exact correctness, abstentions by reason,
  failure rescue, control preservation, and gate decision.

- [ ] **Step 1: Write the failing execution-chain test**

  Assert the runner rejects label-bearing authored inputs, changed request or
  media hashes, missing rows, duplicate rows, query/graph ID mismatch, and
  conflicting proof-freeze overwrite.

- [ ] **Step 2: Run the execution test and observe RED**

- [ ] **Step 3: Implement pure execution and immutable freeze binding**

  Call `count_video` for every row, serialize proof traces canonically, and bind
  query, graph, authoring-request, and source-media hashes.

- [ ] **Step 4: Run the execution test and observe GREEN**

- [ ] **Step 5: Write the failing post-freeze scoring test**

  Use a 12-row literal fixture. Assert the scorer:

  - authenticates every input self-hash and row binding;
  - never exposes questions, IDs, references, or proof text in sanitized output;
  - counts an abstention as unanswered rather than correct;
  - calculates rescue and control preservation on their frozen denominators;
  - authorizes the architecture only with no incorrect non-abstained answer,
    at least 9/12 exact singleton results, and 4/4 controls exact and correct.

- [ ] **Step 6: Run RED, implement scorer, then GREEN**

- [ ] **Step 7: Add mutation tests for the complete chain**

  Mutate one result, graph, query, request, selection, and freeze hash in turn.
  Assert every mutation is rejected before metrics are calculated.

- [ ] **Step 8: Run the Task 5 suite and observe GREEN**

---

### Task 6: Reference-blind authoring and zero-cost architecture gate

**Files:**

- Private create:
  `data/vqa/benchmarks/direct-gemini-count-query-engine-v1/authored/`
- Private create:
  `data/vqa/benchmarks/direct-gemini-count-query-engine-v1/proofs.freeze.json`
- Private create:
  `data/vqa/benchmarks/direct-gemini-count-query-engine-v1/score.json`
- Private create:
  `data/vqa/benchmarks/direct-gemini-count-query-engine-v1/inspection.md`

**Interfaces:**

- Consumes only the 12-row label-blind authoring packet, source media, frozen
  navigation maps, and the versioned schemas.
- Produces frozen queries, evidence graphs, proof states, and a later
  reference-joined private score.

- [ ] **Step 1: Complete all 12 `CountQuery` values from question text only**

  Use no reference, prior answer, correctness, or panel-role file. Record every
  ambiguity as multiple interpretations instead of silently choosing.

- [ ] **Step 2: Complete all 12 evidence graphs from video only**

  Use timestamped observations, shots, relations, and conservative coverage
  claims. Mark incomplete searches partial. Do not infer absence from a missing
  navigation-map entry.

- [ ] **Step 3: Validate and freeze authored inputs before scoring**

  Run the label-blind validator, schema validators, media-hash checks, and the
  execution runner. Record the freeze hash and confirm the scorer has not yet
  loaded references.

- [ ] **Step 4: Run the post-freeze scorer**

  Report exact singleton rate, incorrect non-abstained count, abstention
  reasons, rescue rate, control preservation, and the predeclared architecture
  gate.

- [ ] **Step 5: Inspect every miss or abstention**

  Classify only after freeze as representation failure, evidence failure,
  unresolved benchmark convention, or correct conservative abstention. Do not
  modify schema, query, or graph after unblinding.

---

### Task 7: Evidence-of-record, verification, and stacked PR

**Files:**

- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify generated sibling Markdown and HTML through Quarto only
- Create:
  `docs/experiments/artifacts/vqa-count-query-engine-20260814/README.md`
- Create:
  `docs/experiments/artifacts/vqa-count-query-engine-20260814/summary.json`
- Create:
  `docs/experiments/artifacts/vqa-count-query-engine-20260814/SHA256SUMS`

**Interfaces:**

- Sanitized tracked artifacts expose only aggregates, method, limitations,
  disposition, and private hashes.
- Complete private evidence stays in the canonical benchmark root.

- [ ] **Step 1: Record the result under the existing routing theme**

  Preserve the earlier atomic canary result. Add the query-engine outcome as a
  distinct additive result with canonical evidence-validity and decision axes.

- [ ] **Step 2: Render QMD-derived Markdown and HTML**

  Do not hand-edit generated outputs. Check links, private-data leakage,
  authored anchors, and `git diff --check`.

- [ ] **Step 3: Run final verification**

  Run all new tests, the relevant issue #34/submission regressions, Ruff check
  and format check, artifact hash verification, chain mutation tests, and a
  recursive secret/private-row scan.

- [ ] **Step 4: Request independent substantive review**

  Require review of schema depth, algebra correctness, label isolation,
  provenance authentication, and whether any operator is row-specific. Resolve
  every blocker and rerun verification.

- [ ] **Step 5: Publish a focused stacked draft PR**

  Push `codex/issue-34-count-query-engine` and open it against
  `codex/issue-34-counting-evidence` so PR #38 remains the focused negative
  result. Link both PRs and post an issue #34 update only after the user-authorized
  implementation is verified.
