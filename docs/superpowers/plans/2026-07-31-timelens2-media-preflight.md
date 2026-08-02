# TimeLens2 Media Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the frozen three-video decoder failure before downloading TimeLens2 weights, force the proven torchvision backend, and explain why dstack development-environment planning rejects catalog-visible A40-class offers.

**Architecture:** Add a model-free `media-preflight` mode to the TimeLens2 adapter. The Bash runner invokes it after runtime bootstrap and immutable-prefix validation but before model download; compatibility then uses the same forced torchvision backend. Diagnose dstack independently by replaying one fixed dev configuration through the CLI and tracing the installed planner's fleet/resource intersection without changing server or provider state.

**Tech Stack:** Python 3.12, qwen-vl-utils 0.0.14, torchvision, pytest, Bash, rootless Podman/ROCm, dstack 0.20.29.

## Global Constraints

- Work only on `codex/timelens2-selector-pilot` in the existing linked worktree.
- Keep model revision `90dd30b3eeff3520a30c830eff2209f4ad788ca9` and image digest `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755` unchanged.
- Set `FORCE_QWENVL_VIDEO_READER=torchvision` before importing qwen-vl-utils.
- Validate exactly the frozen three-video compatibility manifest and mounted media hashes; do not read oracle labels.
- Run preflight before model download and preserve its completion denominator on failure.
- Keep private media, manifests, and raw logs outside Git.
- Use dstack diagnosis read-only; do not submit or retry a development environment in this plan.

---

### Task 1: Strict model-free media preflight

**Files:**
- Modify: `tests/test_timelens2_selector.py`
- Modify: `src/trec_video_2026/vqa_experiments/timelens2_selector.py`

**Interfaces:**
- Consumes: `validate_compatibility_requests()`, `build_messages()`, qwen-vl-utils `process_vision_info()`, and an output directory.
- Produces: `run_media_preflight(requests_path: Path, output_dir: Path) -> int` and `media-preflight.json` with status, required/attempted/accepted counts, backend, frame shape, duration, latency, and peak RSS.

- [x] **Step 1: Write a failing backend-contract test**

Add a test that passes three real temporary media paths through a deterministic fake vision processor returning one tensor-like video and metadata. Require all rows to report `backend == "torchvision"`, `accepted == 3`, and a frozen JSON record. Add a second test whose metadata says `decord`; require failure after one attempted row and a persisted `0/1/3` record.

- [x] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest -q \
  tests/test_timelens2_selector.py::test_media_preflight_accepts_exact_three_torchvision_rows \
  tests/test_timelens2_selector.py::test_media_preflight_rejects_non_torchvision_backend
```

Expected: both tests fail because the preflight API does not exist.

- [x] **Step 3: Implement the minimal preflight**

Validate the request manifest first. Decode each row sequentially with the official prompt parameters, require one returned video, require `video_backend == "torchvision"`, require a non-empty four-dimensional shape, record timing/RSS, delete decoded tensors, and persist the summary after every attempt. Re-raise decoder/backend errors after persisting failure evidence.

- [x] **Step 4: Verify GREEN**

Run the two targeted tests and then:

```bash
uv run --no-sync pytest -q tests/test_timelens2_selector.py
```

Expected: all TimeLens2 selector tests pass.

### Task 2: Preflight-before-download runner contract

**Files:**
- Modify: `tests/test_timelens2_dstack_config.py`
- Modify: `scripts/run-timelens2-selector-dstack.sh`
- Modify: `dstack/timelens2-selector-pilot.dstack.yml`

**Interfaces:**
- Consumes: Python `media-preflight` mode from Task 1.
- Produces: compatibility orchestration that runs the model-free preflight before `download_model`, syncs partial evidence on failure, and forces torchvision in the dstack environment.

- [x] **Step 1: Write failing runner/config tests**

Require `FORCE_QWENVL_VIDEO_READER=torchvision` in the compatibility configuration. Require the wrapper's compatibility branch to invoke `--mode media-preflight` before the `download_model` call and to finalize/sync immediately when preflight returns nonzero.

- [x] **Step 2: Verify RED**

Run:

```bash
uv run --no-sync pytest -q tests/test_timelens2_dstack_config.py
```

Expected: failures for the missing forced backend and pre-download preflight.

- [x] **Step 3: Implement minimal orchestration**

Split preparation from model download in the Bash runner. For compatibility, make mounted input read-only, invoke the foreground Python preflight, finalize/sync on failure, then download the model and execute the existing compatibility workload. Keep canary behavior unchanged.

- [x] **Step 4: Verify GREEN**

Run:

```bash
uv run --no-sync pytest -q tests/test_timelens2_dstack_config.py tests/test_timelens2_selector.py
bash -n scripts/run-timelens2-selector-dstack.sh
```

Expected: all focused tests and shell syntax pass.

### Task 3: Local ROCm container regression gate

**Files:**
- Read-only input: the three frozen media files under the ignored `data` symlink.
- Private output: `.timelens2-private/timelens2-selector-pilot-20260730-a1/`.

**Interfaces:**
- Consumes: the committed adapter source and qwen-vl-utils 0.0.14 inside the existing ROCm PyTorch image.
- Produces: a deterministic local verdict that strict decord reproduces `avcodec EAGAIN` while the production media preflight passes all three rows through torchvision.

- [x] **Step 1: Run the strict decord reproducer twice**

Use the existing network-disabled Podman harness and require the exact `-11 / Error sending packet` symptom on both attempts.

- [x] **Step 2: Run the production preflight**

Mount only the three frozen videos, the validated label-free manifest, and repository source read-only. Execute `--mode media-preflight`; require exit zero, `3/3/3`, and `backend=torchvision` for every row.

### Task 4: Source-backed dstack diagnosis

**Files:**
- Read-only: installed dstack 0.20.29 Python sources and ignored declined-preview records.
- Modify: `docs/experiments/2026-07-30-timelens2-selector-pilot.md`

**Interfaces:**
- Consumes: one reproducible pair—`dstack offer` lists eligible machines while `dstack apply` for the fixed dev environment returns no offers.
- Produces: a sanitized explanation naming the exact planner filter and why task and dev-environment behavior differ.

- [x] **Step 1: Trace configuration-specific filters**

Follow the CLI plan request into server offer selection and identify the first predicate that removes the catalog offers. Confirm the exact-range conflict with the installed dstack resource combiner without printing secrets or private payloads.

- [x] **Step 2: Record the diagnosis**

Update only the uniquely owned TimeLens2 evidence record with the local decoder reproduction, tested mitigation, declined dev previews, zero added cost, and the proven dstack planner cause. Do not edit shared project-status QMD or generated reports.

### Task 5: Verification and commit

**Files:**
- All files changed by Tasks 1–4.

**Interfaces:**
- Produces: one clean follow-up commit on `codex/timelens2-selector-pilot`.

- [x] **Step 1: Run fresh verification**

Run:

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check \
  src/trec_video_2026/vqa_experiments/timelens2_selector.py \
  tests/test_timelens2_selector.py \
  tests/test_timelens2_dstack_config.py
bash -n scripts/run-timelens2-selector-dstack.sh
git diff --check
```

- [x] **Step 2: Verify privacy and commit**

Require the `data` symlink to resolve to the canonical dataset, `.timelens2-private/` to remain ignored, no private path to be tracked, and no literal credential to appear in the staged diff. Stage only model-specific code/config/tests, this plan, and the TimeLens2 evidence record; then commit the complete follow-up.
