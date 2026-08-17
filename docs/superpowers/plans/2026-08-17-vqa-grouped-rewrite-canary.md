# VQA Grouped-Rewrite Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one bounded five-query Gemini grouping-and-rewrite canary without changing the ready submission packet.

**Architecture:** A single runner derives a label-free frozen input from the final submission artifacts, validates one structured provider response, and freezes auditable evidence. Provider interaction is opt-in with `--run`; dry-run performs selection, prompt construction, and cost validation only.

**Tech Stack:** Python 3.12, existing `google-genai` client/runtime, pytest, Ruff.

## Global Constraints

- Exact query IDs: `1, 310, 1563, 2404, 2421`.
- Model: `gemini-3.7-flash`; one text-only batched call per fixed strategy; seed 42; JSON response; maximum 8,192 output tokens.
- Fixed strategies: `semantic_rewrite_a` and `exact_group_hedge_b`; separate immutable roots.
- Hard estimated and observed cost caps: `$0.10` per strategy.
- Rank 1 must equal the frozen current fusion rank-1 answer byte-for-byte.
- Every genuine source candidate is grouped exactly once; no labels, references, annotations, metrics, media, transcripts, or baseline ranks 2–10.
- The existing ready packet and all prior evidence are read-only.
- Do not commit, push, upload, or issue a provider call during implementation/review.

---

### Task 1: Five-query grouped-rewrite canary runner

**Files:**
- Create: `scripts/run_vqa_grouped_rewrite_canary.py`
- Create: `tests/test_vqa_grouped_rewrite_canary.py`

**Interfaces:**
- Consumes: final testing questions, Gemini overlay, Qwen aggregate freeze, Qwen3.5 control, final fusion submission and provenance.
- Produces: `select_rows(...) -> list[dict]`, `build_request(rows)`, `validate_response(payload, rows) -> dict`, and `run_canary(..., run=False) -> dict`.

- [ ] **Step 1: Write failing tests** proving the exact panel/source selection, original-rank marker, exclusion of speculative tails/forbidden fields, one batched request per strategy, exact rank-1 preservation, semantic grouping of differently worded equivalents, source/group answer coverage, strategy-B exact-source retention, ten distinct answers, cross-query rejection, real SDK usage accounting, and `$0.10` caps.
- [ ] **Step 2: Run `uv run pytest -q tests/test_vqa_grouped_rewrite_canary.py`** and confirm failure because the runner does not exist.
- [ ] **Step 3: Implement the minimal runner** with immutable JSON artifacts, dry-run default, explicit `--run`, one provider call, strict schema checks, usage-derived cost, and fail-closed terminal evidence.
- [ ] **Step 4: Run `uv run pytest -q tests/test_vqa_grouped_rewrite_canary.py` and `uv run ruff check scripts/run_vqa_grouped_rewrite_canary.py tests/test_vqa_grouped_rewrite_canary.py`** and require clean output.
- [ ] **Step 5: Run both real dry-runs** and require five queries, one request per strategy, ten requested answers per query, and estimated cost below `$0.10` per strategy while both private output roots remain absent.
- [ ] **Step 6: Independent read-only prelaunch review** must return GO before the paid call.
- [ ] **Step 7: Run both strategies once, freeze evidence, and manually inspect all 100 answers** without labels. Compare validity, grouping, exact-source retention, introduced facts, diversity, cost, and latency; do not scale automatically.
