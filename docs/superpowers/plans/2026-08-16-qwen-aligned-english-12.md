# Qwen Aligned-English Fresh-12 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one immutable, label-isolated 12-row Qwen3-ASR plus forced-aligner quality panel with forced English, then compare answer-phrase containment against frozen Whisper only after a complete prediction freeze.

**Architecture:** Extend the existing canary controller with one new selection kind while preserving every old frozen contract. A private exposure index proves Qwen3-ASR freshness, a deterministic question-only 6/3/3 selector freezes WAV identities, and the unchanged dedicated mode runner executes forced English plus timestamps in one offline process. Existing freeze/status/scoring primitives remain the authority, with additive paired-count output.

**Tech Stack:** Python 3.12, pytest, rootless Podman, ROCm/BF16, local Hugging Face snapshots, JSON/SHA-256 freezes, and Quarto.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-16-qwen-aligned-english-12-design.md` exactly.
- Preserve Stage 0.5d and Stage 0.5e byte-for-byte; never rerun or mutate them.
- Keep `src/trec_video_2026/vqa_qwen_empty_output_runner.py` and `src/trec_video_2026/vqa_audio_runner.py` unchanged; bind their current hashes.
- Use exactly one local `$0.00` inference attempt and no retry.
- Never open annotations or Whisper content before a verified `12/12` prediction freeze.
- Never stage `data/` or the unrelated `.claude/` directory.
- Gemini is independently planned and executed; it cannot share this run or decision.

---

### Task 1: Freshness index and deterministic 6/3/3 selection

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `scripts/run_qwen_aligner_canary.py`
- Create: `tests/test_vqa_qwen_fresh_quality_panel.py`

**Interfaces:**
- Produces `build_qwen_asr_exposure_index(source_paths, oracle_manifest, built_at) -> dict[str, Any]`.
- Produces `select_fresh_english_quality_panel(manifest, exposure_index, audio_root, selected_at, source_hashes, panel_size=12) -> dict[str, Any]`.
- Adds CLI commands `fresh-exposure-index` and `fresh-quality-select`.

- [ ] **Step 1: Write failing exposure-index and selection tests**

  Add synthetic canonical-data fixtures proving exact marker detection,
  conservative identity union, oracle exclusion, source hash/path binding,
  visible-text exclusion, fixed stratum priority, exact `6/3/3` quotas,
  global manifest order, no question in output rows, and rejection of missing
  audio, insufficient strata, duplicate IDs, mutated sources, or noncanonical
  paths.

  ```python
  index = build_qwen_asr_exposure_index(
      source_paths=[prior_path],
      oracle_manifest=oracle_path,
      built_at="2026-08-16T00:00:00Z",
  )
  selection = select_fresh_english_quality_panel(
      manifest=manifest,
      exposure_index=index,
      audio_root=audio_root,
      selected_at="2026-08-16T00:00:01Z",
      source_hashes={"manifest_sha256": sha256(manifest_path)},
  )
  assert Counter(row["speech_stratum"] for row in selection["rows"]) == {
      "reported_speech": 6,
      "conversation": 3,
      "vocalization": 3,
  }
  ```

- [ ] **Step 2: Run the new tests and confirm RED**

  Run:
  `uv run pytest -q tests/test_vqa_qwen_fresh_quality_panel.py -k 'exposure or select'`

  Expected: import/attribute failures because both functions and selection kind are absent.

- [ ] **Step 3: Implement the minimal index and selector**

  Add these exact public constants and function interfaces:

  ```python
  FRESH_ENGLISH_SELECTION_KIND = "qwen_aligned_english_fresh_speech_panel"
  FRESH_PANEL_SIZE = 12
  FRESH_STRATUM_QUOTAS = {
      "reported_speech": 6,
      "conversation": 3,
      "vocalization": 3,
  }

  ```

  `build_qwen_asr_exposure_index` takes keyword-only `source_paths`,
  `oracle_manifest`, and `built_at` arguments and returns a JSON-compatible
  dictionary. `select_fresh_english_quality_panel` takes keyword-only
  `manifest`, `exposure_index`, `audio_root`, `selected_at`, `source_hashes`,
  and an optional `panel_size` defaulting to `FRESH_PANEL_SIZE`; it also
  returns a JSON-compatible dictionary.

  The index records only source path/hash, markers, and excluded identities.
  The selector reads only manifest questions, index metadata, WAV bytes, and
  WAV duration; it emits no question or comparator field.

- [ ] **Step 4: Add exclusive CLI output paths and pass focused tests**

  `fresh-exposure-index` accepts repeated `--source`, one
  `--oracle-manifest`, `--output`, and `--built-at`. `fresh-quality-select`
  accepts `--manifest`, `--exposure-index`, `--audio-root`, `--output`,
  `--selected-at`, and `--experiment-id`. Both use `_exclusive_json`.

  Run:
  `uv run pytest -q tests/test_vqa_qwen_fresh_quality_panel.py -k 'exposure or select'`

### Task 2: Bind the forced-English execution surface without changing old freezes

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `tests/test_vqa_qwen_fresh_quality_panel.py`
- Verify unchanged: `src/trec_video_2026/vqa_qwen_empty_output_runner.py`
- Verify unchanged: `src/trec_video_2026/vqa_audio_runner.py`

**Interfaces:**
- `prepare_canary()` recognizes `FRESH_ENGLISH_SELECTION_KIND` and requires the unchanged dedicated runner path.
- `build_podman_command()` emits the dedicated module with `--language English --timestamps on` only for the new kind.
- `_verify_runner_binding()` validates dedicated-runner, helper, and controller hashes for the new kind while retaining the old default behavior.

- [ ] **Step 1: Write failing preparation/command/mutation tests**

  Assert the private request schema remains exactly run/condition/video/audio/
  mode, the protocol condition is
  `qwen3-asr-1.7b-forced-aligner-0.6b-english`, and the command contains:

  ```text
  -m trec_video_2026.vqa_qwen_empty_output_runner
  --forced-aligner-id Qwen/Qwen3-ForcedAligner-0.6B
  --language English
  --timestamps on
  ```

  Add runner/helper/controller mutation tests that fail before command or
  record construction. Retain a regression asserting the old command still
  invokes `trec_video_2026.vqa_audio_runner` with no language flag.

- [ ] **Step 2: Run the command tests and confirm RED**

  Run:
  `uv run pytest -q tests/test_vqa_qwen_fresh_quality_panel.py -k 'prepare or command or mutation'`

- [ ] **Step 3: Implement conditional protocol and command binding**

  Use the frozen selection kind to choose runner/condition/decode controls.
  Record absolute runner/helper/controller paths and SHA-256 values in the new
  protocol. Reject any unexpected module, duplicate decode flag, network mode,
  image, revision, mount, or credential-bearing argument before subprocess.

- [ ] **Step 4: Run old and new focused suites**

  Run:

  ```bash
  uv run pytest -q \
    tests/test_vqa_qwen_fresh_quality_panel.py \
    tests/test_vqa_qwen_aligner_canary.py \
    tests/test_run_qwen_aligner_canary.py \
    tests/test_vqa_qwen_empty_output_runner.py \
    tests/test_vqa_audio_runner.py
  uv run ruff check \
    src/trec_video_2026/vqa_qwen_aligner_canary.py \
    scripts/run_qwen_aligner_canary.py \
    tests/test_vqa_qwen_fresh_quality_panel.py
  ```

### Task 3: Add paired post-freeze outcome counts

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `tests/test_vqa_qwen_fresh_quality_panel.py`

**Interfaces:**
- `score_postfreeze()` retains existing row records and adds aggregate
  `both_present_count`, `qwen_only_present_count`,
  `whisper_only_present_count`, `neither_present_count`, and
  `directional_decision`.

- [ ] **Step 1: Write a failing paired-count test**

  Build a ready synthetic 12-row freeze with both-hit, Qwen-only,
  Whisper-only, and neither-hit examples. Assert the four cells sum to 12 and
  the decision is derived only from the two aggregate containment totals.

- [ ] **Step 2: Run the score test and confirm RED**

  Run:
  `uv run pytest -q tests/test_vqa_qwen_fresh_quality_panel.py -k score`

- [ ] **Step 3: Implement additive paired counts and decision**

  ```python
  qwen_total = sum(row["official_phrase_present"] for row in rows)
  whisper_total = sum(row["whisper_official_phrase_present"] for row in rows)
  decision = (
      "directional_qwen_signal" if qwen_total > whisper_total
      else "whisper_directional_signal" if qwen_total < whisper_total
      else "no_directional_difference"
  )
  ```

  Missing/null Whisper text remains a miss. `require_quality_scoring_ready()`
  must execute before either private scoring source is loaded.

- [ ] **Step 4: Pass the complete focused suite and commit code**

  Run the Task 2 focused commands, `git diff --check`, then commit only source,
  tests, this plan, and the design with:
  `git commit -m "feat: add fresh aligned-English Qwen panel"`.

### Task 4: Freeze inputs and execute the sole local batch

**Files:**
- Private create: `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-selection-v1/`
- Private create: `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/`

**Interfaces:**
- Exposure index: private `exposure-index.json`.
- Selection: private `selection.json`.
- Run root: immutable input, command, terminal, validation, and prediction freezes.

- [ ] **Step 1: Preflight canonical paths and immutable runtime**

  Verify the physical canonical data root, fresh selection/run roots, exact
  local image digest, both model snapshots, unchanged runner/helper hashes,
  ROCm device access, BF16 smoke, free disk, and `$0.00` external spend.

- [ ] **Step 2: Build the conservative exposure index and exact 12-row selection**

  Enumerate private JSON/JSONL sources under `data/vqa` that contain an exact
  Qwen3-ASR marker, pass every resolved source explicitly to
  `fresh-exposure-index`, then run `fresh-quality-select`. Inspect only
  aggregate source/exclusion counts, panel count, stratum counts, duplicate
  count, missing-audio count, and hashes.

- [ ] **Step 3: Prepare and audit the frozen command**

  Run `prepare` with experiment ID
  `qwen-aligner-stage05f-20260816-aligned-english-12-v1` and the unchanged
  dedicated runner. Verify the printed command has exact pinned flags, no
  credentials, read-only canonical mounts, network disabled, and one input
  batch. Do not execute the printed command.

- [ ] **Step 4: Record exactly one attempt and freeze predictions**

  Invoke the `record` subcommand once, then `freeze` and `status`. Never invoke
  the raw Podman command, never replace a failed row, and never retry.

### Task 5: Score only after readiness and record the result

**Files:**
- Private conditional create: `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/postfreeze-score.json`
- Modify: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.qmd`
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Modify: `docs/experiments/artifacts/qwen-aligner-quality-20260816/`
- Regenerate: affected QMD-derived Markdown and HTML.

**Interfaces:**
- Consumes only aggregate terminal/validation/freeze data before readiness.
- Produces a private score only for `12/12` ready and a sanitized aggregate record.

- [ ] **Step 1: Apply the all-row gate**

  If status is not `predictions_frozen/ready_for_quality_test`, record the
  completion denominator and stop without running `score`. If ready, invoke
  `score` with the private annotation directory and frozen full-panel
  `whisper.json`.

- [ ] **Step 2: Create a sanitized aggregate evidence record**

  Include only row count, stratum counts, completion/alignment counts, Qwen and
  Whisper containment totals, four paired cells, decision, timing/memory,
  runtime/model bindings, cost, and immutable hashes. Exclude row identities,
  questions, answers, transcripts, raw logs, and media paths.

- [ ] **Step 3: Update QMD sources and regenerate outputs**

  Separate observation from interpretation, state the evidence validity and
  limits, preserve earlier Stage 0.5d/0.5e claims, update the next action, and
  render GFM/HTML only through Quarto.

- [ ] **Step 4: Verify, review, and commit the record**

  Run the focused suite, Ruff, JSON parsing, tracked `SHA256SUMS`, link checks,
  privacy/credential scans, both old frozen statuses plus Stage 0.5f status,
  `git diff --check`, and an independent read-only review. Commit only intended
  tracked files with `git commit -m "docs: record aligned-English Qwen panel"`.
