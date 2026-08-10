# Qwen3.5 Safety Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze and prepare a private, validator-ready 2,443-query TRECVID 2026 AG safety submission with the verified Qwen3.5-4B uniform-12 path at rank 1.

**Architecture:** A local preparation module creates a label-free query/media contract and deduplicated uniform-frame evidence plan. A pinned CUDA dstack task runs one baseline answer and one nine-answer tail request per query, freezes the complete ranking, and writes immutable private outputs. A host-side finalizer verifies the freeze, serializes the official JSON schema, runs the pinned NIST validator, and records checksums without uploading to Evalbase.

**Tech Stack:** Python 3.12, pytest, PyAV/ffmpeg, Qwen3.5-4B, Transformers 5 pinned source revision, dstack 0.20.29, Hugging Face Buckets CLI 1.26.0, NIST VQA validator commit `e06cb7854d7defe53e86412c2bba0ca5f636ea40`.

## Global Constraints

- The source topics contain exactly 2,443 unique integer query IDs and 1,956 unique video IDs.
- Keep all competition questions, media, predictions, and raw logs under ignored `data/`; never commit them.
- Rank 1 must use the pinned Qwen model and unchanged `current` policy over twelve uniform bin-midpoint frames.
- Every query must have exactly ten unique, non-empty ranked answers with finite non-negative timing.
- Labels and references are forbidden from every inference request.
- Freeze predictions before serialization and refuse overwrite of immutable outputs.
- Do not upload to Evalbase or change sharing permissions.

---

### Task 1: Label-free 2026 manifest and media resolution

**Files:**
- Create: `src/trec_video_2026/vqa_safety_submission.py`
- Create: `scripts/prepare_qwen35_safety_submission.py`
- Create: `tests/test_vqa_safety_submission.py`

**Interfaces:**
- Consumes: official topic JSON and the two local 2026 media directories.
- Produces: `build_test_manifest(topics_path, dataset_root) -> dict`, with ordered query records and source hashes.

- [ ] **Step 1: Write failing manifest tests**

  Cover duplicate query IDs, two queries sharing one video, supplied WebM and
  downloaded YouTube resolution, unavailable media, forbidden label fields,
  and exact source-order preservation.

- [ ] **Step 2: Run tests and verify RED**

  Run: `uv run pytest tests/test_vqa_safety_submission.py -q`

- [ ] **Step 3: Implement the minimal manifest builder and CLI**

  Store `query_id`, `video_id`, `question`, dataset-relative `media`,
  availability, byte size, and SHA-256. Refuse ambiguous media matches and
  write atomically to a new run root.

- [ ] **Step 4: Run tests and verify GREEN**

  Run: `uv run pytest tests/test_vqa_safety_submission.py -q`

### Task 2: Uniform-12 evidence and ranked-tail contracts

**Files:**
- Modify: `src/trec_video_2026/vqa_safety_submission.py`
- Modify: `src/trec_video_2026/vqa_models.py`
- Modify: `tests/test_vqa_safety_submission.py`
- Modify: `tests/test_vqa_models.py`

**Interfaces:**
- Produces: `build_evidence_plan`, `build_rank1_request`,
  `build_ranked_tail_request`, `parse_ranked_tail`, and
  `complete_ranked_candidates`.

- [ ] **Step 1: Write failing request and parsing tests**

  Assert exact twelve bin-midpoint frames, reuse across duplicate video IDs,
  unchanged rank-1 `current` prompt behavior, exclusion of the frozen rank-1
  answer from the tail prompt, strict ordered JSON parsing, normalized
  deduplication, and bounded fallback completion.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `uv run pytest tests/test_vqa_safety_submission.py tests/test_vqa_models.py -q`

- [ ] **Step 3: Implement evidence and candidate contracts**

  Keep unavailable-media requests explicit and question-only. Record every
  parse/repair/fallback disposition in the private prediction row.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run: `uv run pytest tests/test_vqa_safety_submission.py tests/test_vqa_models.py -q`

### Task 3: Freeze, serialization, and validator wrapper

**Files:**
- Modify: `src/trec_video_2026/vqa_safety_submission.py`
- Create: `scripts/finalize_qwen35_safety_submission.py`
- Modify: `tests/test_vqa_safety_submission.py`

**Interfaces:**
- Consumes: ordered manifest and complete private prediction rows.
- Produces: `ranked-predictions.json`, `freeze.json`,
  `submission-top1.json`, `submission-top5.json`,
  `submission-top10.json`, validator report, and `SHA256SUMS`.

- [ ] **Step 1: Write failing finalization tests**

  Test freeze-before-write, tamper rejection, exact 2,443-row/order checks on
  a parameterized fixture, ten-answer uniqueness, contiguous ranks, finite
  times, overwrite refusal, and nonzero validator propagation.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `uv run pytest tests/test_vqa_safety_submission.py -q`

- [ ] **Step 3: Implement the finalizer**

  Reuse `freeze_predictions`, `verify_prediction_freeze`, and
  `serialize_ag_submission`. Pin and record the validator revision and invoke
  it in an isolated temporary directory.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run: `uv run pytest tests/test_vqa_safety_submission.py tests/test_vqa_submission.py -q`

### Task 4: Pinned remote task and resumable inference

**Files:**
- Create: `scripts/run-qwen35-safety-submission-dstack.sh`
- Create: `dstack/qwen35-safety-submission.dstack.yml`
- Create: `tests/test_qwen35_safety_submission_dstack.py`

**Interfaces:**
- Consumes: immutable private HF Bucket input bundle and repository code.
- Produces: resumable query-level rank-1/tail results, runtime attestation,
  terminal status, and an immutable private HF output bundle.

- [ ] **Step 1: Write failing executable contract tests**

  Exercise the shell entry point in preflight mode and assert fail-closed run
  name, input/output URI, manifest hash, model/runtime revisions, expected
  query count, no labels, and resumable checkpoint rules.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `uv run pytest tests/test_qwen35_safety_submission_dstack.py -q`

- [ ] **Step 3: Implement the pinned dstack task**

  Reuse the verified CUDA image and model revision. Extract frames once per
  available video, run rank 1 before the tail, checkpoint each query, sync
  only private artifacts, and enforce terminal completion before freeze.

- [ ] **Step 4: Run focused tests and verify GREEN**

  Run: `uv run pytest tests/test_qwen35_safety_submission_dstack.py -q`

### Task 5: Prepare, canary, execute, freeze, and validate

**Files:**
- Runtime only: `data/vqa/submissions/qwen35-safety-20260810-a1/`

**Interfaces:**
- Produces: the final private top-10 submission and audit bundle.

- [ ] **Step 1: Run the full local test suite relevant to submission**

  Run: `uv run pytest tests/test_vqa_safety_submission.py tests/test_qwen35_safety_submission_dstack.py tests/test_vqa_submission.py tests/test_vqa_models.py -q`

- [ ] **Step 2: Prepare and audit the local immutable input bundle**

  Confirm 2,443 queries, 1,956 videos, 1,946 available videos, ten
  unavailable videos, exact source hashes, no labels, and private write-once
  HF prefix emptiness before upload.

- [ ] **Step 3: Preview dstack offers without launching**

  Run: `echo "n" | dstack apply -f dstack/qwen35-safety-submission.dstack.yml`
  Inspect the full offer and cost output. Because the user already authorized
  the run, submit detached only if the preview has a valid offer at or below
  the configured price and duration cap.

- [ ] **Step 4: Run a bounded live canary and inspect artifacts**

  Require successful rank-1 and ranked-tail output, unique ten-answer
  completion, correct timing, runtime attestation, and remote sync before the
  full workload proceeds.

- [ ] **Step 5: Execute the full run and monitor to terminal status**

  Use detached dstack execution, normal non-diagnostic logs, and resumable
  checkpoints. Do not claim completion until all 2,443 query rows exist.

- [ ] **Step 6: Download, freeze, serialize, and validate**

  Run the finalizer, verify the prediction freeze and all SHA-256 entries,
  and require the pinned NIST validator to exit zero on
  `submission-top10.json`.

- [ ] **Step 7: Final privacy and repository checks**

  Run: `git diff --check && git status --short`
  Confirm no private run artifact is tracked or staged and report the local
  submission path, hash, validator result, fallback counts, and remaining
  upload action.

