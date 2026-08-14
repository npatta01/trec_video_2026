# Question-Driven Atomic Counting Evidence Implementation Plan

> **Execution:** Implement inline under the delegated issue #34 authorization,
> using test-driven development and a cheapest-representative canary gate.

**Goal:** Reproduce and correct the 66-row count census, freeze 26 confirmed
Gemini failures plus one ambiguity and 12 correct controls, test one label-blind
atomic-evidence recovery call, and publish a fully provenance-bound result.

**Architecture:** Pure code owns count parsing, census/panel validation, atomic
record validation, deterministic reduction, and scoring joins. A thin Gemini
runner owns upload reuse, requests, raw responses, usage/cost accounting, and
immutable freezes. Preparation and inference consume a recursively label-free
manifest; references are loaded only by the post-freeze scorer.

**Tech stack:** Python 3.12, pytest, google-genai 2.12.1, native Gemini video,
existing semantic scorers (NLTK METEOR, BERTScore, UMBC STS), Quarto, GitHub.

**Current status (2026-08-14):** Implementation and the six-row canary are
complete. The tested treatment is a valid negative (`1/5` rescues, `1/1`
control, `$0.253924`) and did not authorize scale. Final review hardening is
complete. Implementation commit `907e3b1` is pushed, draft PR #38 is open,
and the final result is posted to issue #34.

## Global constraints

- Never rerun or modify the frozen 475-row Gemini panel.
- Never send labels, references, prior predictions, correctness, expected
  counts, or panel roles to Gemini.
- Use one identical request per row; no answer-diversity or repair call.
- Run the six-row canary before remaining paid inference.
- Stop before cumulative projected or observed cost reaches USD 3.
- Preserve full private provenance under the canonical shared data root and
  only sanitized aggregates in Git.
- Freeze predictions before reference access and scoring.
- Keep issue #35 and #36 out of scope.

## Task 1: Count semantics and corrected census

**Files:**

- Create `src/trec_video_2026/vqa_counting_evidence.py`
- Create `tests/test_vqa_counting_evidence.py`
- Create `scripts/build_gemini_counting_panel.py`
- Create `tests/test_build_gemini_counting_panel.py`

- [x] Write failing tests for timestamp stripping, anchored reference parsing,
  compound number words including `forty`, non-committal `N or M` answers,
  the lexical-but-not-count question, and count-unit ambiguity.
- [x] Implement the minimal parser and census classifier.
- [x] Write failing tests for deterministic panel selection, 26/1/12 role
  counts, immutable ID/hash binding, and recursive label-key rejection.
- [x] Implement the panel builder and safe CLI.
- [x] Run focused tests and reproduce the 38/26/1/1 census from frozen files.
- [x] Freeze census, adjudication, panel, request manifest, and hashes privately.

## Task 2: Atomic evidence contract and reduction

**Files:**

- Extend `src/trec_video_2026/vqa_counting_evidence.py`
- Extend `tests/test_vqa_counting_evidence.py`

- [x] Write failing tests for the seven count units, exact root/record fields,
  duplicate record IDs, excluded records, non-finite/out-of-range timestamps,
  inverted spans, and deterministic record-count reduction.
- [x] Implement the strict validator and prompt builder with no answer/count
  output field.
- [x] Add a test that recursively rejects label-bearing runtime inputs.
- [x] Run focused contract tests.

## Task 3: Provenance-bound Gemini runner and cost gate

**Files:**

- Create `scripts/run_gemini_counting_evidence.py`
- Create `tests/test_run_gemini_counting_evidence.py`

- [x] Write failing tests using a fake provider at the transport boundary for
  request identity, upload reuse validation, raw-response preservation,
  identical canary/full configuration, one-call limit, prediction freezing,
  and pre-request USD 3 hard stop.
- [x] Implement prepare, canary, project, continue, freeze, and status commands.
- [x] Verify current official standard Gemini pricing and freeze it with source
  URL/access time before any paid call.
- [x] Run the six-row canary and freeze its observed usage/cost.
- [x] Continue only if schema, rescue/control, and projected-cost gates pass.

## Task 4: Post-freeze scoring and audit

**Files:**

- Create `scripts/score_gemini_counting_evidence.py`
- Create `tests/test_score_gemini_counting_evidence.py`

- [x] Write failing tests for freeze/hash authentication, reference isolation,
  26-row rescue and 12-row control denominators, ambiguous exclusion, semantic
  metric aggregation, and sanitized output.
- [x] Implement the scorer using existing semantic metric interfaces.
- [x] Freeze METEOR, BERTScore-F1, STS, composite, rescue/control rates,
  count-unit strata, and failure taxonomy.
- [x] Inspect every unresolved canary failure and record whether evidence was omitted,
  duplicated, mis-unitized, or correctly enumerated but misreduced.

## Task 5: Evidence-of-record and handoff

**Files:**

- Create `docs/experiments/artifacts/vqa-gemini-counting-evidence-20260814/`
- Extend `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Update the primary research-theme status/ledger only if the conclusion or
  next action changes
- Update `AGENTS.md` with durable private/tracked pointers

- [x] Project only sanitized aggregates and hashes into the tracked bundle.
- [x] Add the result, limitations, disposition, and next decision to QMD.
- [x] Render/validate generated report surfaces according to project tooling.
- [x] Run focused and relevant regression tests plus artifact validators.
- [x] Review the diff for secrets, private data, and scope drift.
- [x] Commit, push `codex/issue-34-counting-evidence`, open a focused PR, and
  post concise progress/result comments to issue #34.
