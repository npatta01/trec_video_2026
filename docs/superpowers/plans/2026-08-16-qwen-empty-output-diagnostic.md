# Qwen Empty-Output Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent incomplete Qwen quality runs from reaching labels or scoring, then execute one immutable four-arm local diagnostic for the two preserved empty outputs.

**Architecture:** Add a shared ready-for-quality guard to the existing canary, but leave its bound audio runner unchanged. Build a separate Qwen-only diagnostic runner plus an immutable host harness that selects four rows using frozen terminal status only, executes four process-isolated Podman arms once, validates and freezes every arm, and emits a deterministic mechanism decision.

**Tech Stack:** Python 3.12, pytest, rootless Podman, ROCm/BF16, pinned local Hugging Face snapshots, JSON/SHA-256 freeze artifacts, and Quarto QMD reporting.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-16-qwen-empty-output-diagnostic-design.md` exactly.
- Preserve `src/trec_video_2026/vqa_audio_runner.py` byte-for-byte so the Stage 0.5d runner binding remains valid.
- Bind and re-verify both the new diagnostic runner hash and its imported `vqa_audio_runner.py` helper-source hash throughout the new diagnostic lifecycle.
- Never retry, relabel, overwrite, score, or reopen the frozen Stage 0.5d attempt.
- Keep questions, answers, annotations, Whisper text, transcript text, grades, and metrics out of selection and inference.
- Use exactly two preserved empty-output rows plus the first two preserved passing rows in frozen request order.
- Run arms A, B, C, and D once each, in order, under `--network none`; paid provider cost stays `$0.00`.
- Keep row-level identities and outputs under the ignored canonical data root; track only aggregate sanitized evidence.
- Do not run Gemini as part of this plan. Gemini 3.7 Flash gets a separate native-AV compatibility test.

---

### Task 1: Enforce the all-row scoring gate

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `scripts/run_qwen_aligner_canary.py`
- Test: `tests/test_vqa_qwen_aligner_canary.py`
- Create: `tests/test_run_qwen_aligner_canary.py`

**Interfaces:**
- Produce: `require_quality_scoring_ready(root: Path) -> dict[str, Any]`
- Consume: verified `predictions-freeze.json` and `validated-results.json`

- [x] **Step 1: Write the core failing regression test**

  Extend `test_nonzero_exit_cannot_freeze_ready_decision` after freezing:

  ```python
  with pytest.raises(ValueError, match="not ready|ready_for_quality_test"):
      score_postfreeze(
          root,
          references={"target": "phrase", "control": "phrase"},
          whisper={"target": {"text": "phrase"}, "control": {"text": "phrase"}},
      )
  assert not (root / "postfreeze-score.json").exists()
  ```

- [x] **Step 2: Write the CLI privacy-boundary failing test**

  Import `scripts.run_qwen_aligner_canary`, monkeypatch `status_canary()` to
  return `{"status": "predictions_frozen", "decision": "not_ready"}`, and
  make `_load_json()` raise `AssertionError("labels were opened")`. Assert
  `_score_inputs()` raises the readiness error before `_load_json()` is called.

- [x] **Step 3: Run both tests and verify RED**

  Run:

  ```bash
  uv run pytest -q \
    tests/test_vqa_qwen_aligner_canary.py::test_nonzero_exit_cannot_freeze_ready_decision \
    tests/test_run_qwen_aligner_canary.py
  ```

  Expected: direct scoring succeeds unexpectedly and the CLI calls
  `_load_json()`.

- [x] **Step 4: Implement the minimal shared guard**

  In the core module:

  ```python
  def require_quality_scoring_ready(root: Path) -> dict[str, Any]:
      freeze = _verify_prediction_freeze(root)
      validation = _read_object(root / "validated-results.json")
      if freeze.get("decision") != validation.get("status"):
          raise ValueError("prediction decision does not match validation")
      if freeze.get("decision") != "ready_for_quality_test":
          raise ValueError("predictions are not ready for quality scoring")
      return freeze
  ```

  Call it from `score_postfreeze()` and export it. In `_score_inputs()`, call it
  before the first `_load_json()` so private labels cannot be opened first.

- [x] **Step 5: Verify GREEN and the focused canary suite**

  Run:

  ```bash
  uv run pytest -q tests/test_vqa_qwen_aligner_canary.py tests/test_run_qwen_aligner_canary.py
  uv run ruff check src/trec_video_2026/vqa_qwen_aligner_canary.py scripts/run_qwen_aligner_canary.py tests/test_vqa_qwen_aligner_canary.py tests/test_run_qwen_aligner_canary.py
  ```

- [x] **Step 6: Commit the scoring repair**

  Commit only the two source files and their tests with message
  `fix: block incomplete Qwen quality scoring`.

### Task 2: Add the isolated four-mode Qwen runner

**Files:**
- Create: `src/trec_video_2026/vqa_qwen_empty_output_runner.py`
- Create: `tests/test_vqa_qwen_empty_output_runner.py`
- Reuse without modifying: `src/trec_video_2026/vqa_audio_runner.py`

**Interfaces:**
- Produce: `transcribe_qwen_mode(model, audio_path, *, language, timestamps) -> dict[str, Any]`
- Produce: `parse_args(argv) -> argparse.Namespace` with `--language auto|English` and `--timestamps on|off`
- Produce: `main(argv=None) -> int`, stdin ordered request batch to stdout ordered result batch

- [x] **Step 1: Write failing exact-call tests for all decode modes**

  Use a fake model whose `transcribe()` records keyword arguments. Assert:

  ```python
  transcribe_qwen_mode(model, "/a.wav", language=None, timestamps=False)
  assert model.kwargs == {"audio": "/a.wav", "language": None}

  transcribe_qwen_mode(model, "/a.wav", language=None, timestamps=True)
  assert model.kwargs == {
      "audio": "/a.wav", "language": None, "return_time_stamps": True
  }

  transcribe_qwen_mode(model, "/a.wav", language="English", timestamps=True)
  assert model.kwargs["language"] == "English"
  ```

  Also assert timestamps `on` without forced-aligner arguments is rejected.

- [x] **Step 2: Run the runner tests and verify RED**

  Run `uv run pytest -q tests/test_vqa_qwen_empty_output_runner.py`.
  Expected: import failure because the dedicated runner does not exist.

- [x] **Step 3: Implement the smallest Qwen-only runner**

  Reuse `load_qwen`, `load_qwen_with_aligner`, `normalize_qwen_asr_result`,
  `validate_request`, `validate_rocm_bfloat16`, `audio_evidence_quality`, and
  `finalize_result_metrics` from `vqa_audio_runner`. The configured call must
  omit `return_time_stamps` when timestamps are off. Preserve the existing
  structured result keys, per-row error capture, source hash, duration,
  runtime, BF16/device check, and stdout/stderr separation.

- [x] **Step 4: Add request-batch and failure-preservation tests**

  Use fake dependencies to assert ordered four-row output, nonempty success
  normalization, exact empty-text `ValueError` preservation, and no label keys
  in output identity.

- [x] **Step 5: Verify GREEN without changing the original runner**

  Run:

  ```bash
  uv run pytest -q tests/test_vqa_audio_runner.py tests/test_vqa_qwen_empty_output_runner.py
  git diff --exit-code -- src/trec_video_2026/vqa_audio_runner.py
  uv run ruff check src/trec_video_2026/vqa_qwen_empty_output_runner.py tests/test_vqa_qwen_empty_output_runner.py
  ```

- [x] **Step 6: Commit the isolated runner**

  Commit the new runner and tests with message
  `feat: add isolated Qwen empty-output runner`.

### Task 3: Build the immutable diagnostic harness

**Files:**
- Create: `src/trec_video_2026/vqa_qwen_empty_output_diagnostic.py`
- Create: `scripts/run_qwen_empty_output_diagnostic.py`
- Create: `tests/test_vqa_qwen_empty_output_diagnostic.py`

**Interfaces:**
- Produce: `prepare_diagnostic(root: Path, source_root: Path, *, runner_path: Path) -> dict[str, Any]`
- Produce: `build_arm_command(root: Path, arm_id: str, *, hub_cache: Path, data_root: Path) -> list[str]`
- Produce: `record_arm(root: Path, arm_id: str, command: Sequence[str], *, runner=subprocess.run) -> dict[str, Any]`
- Produce: `freeze_diagnostic(root: Path) -> dict[str, Any]`
- Produce: `status_diagnostic(root: Path) -> dict[str, Any]`

- [x] **Step 1: Write failing preparation tests**

  Build a synthetic verified Stage 0.5d source with two exact empty failures
  and at least two successes. Assert preparation selects the two failures plus
  the first two successes in source request order; freezes audio hashes,
  durations, source freeze hash, diagnostic runner hash, imported helper-source
  hash, four arm definitions, and four ordered request files; and rejects
  label-bearing keys, a ready source,
  wrong failure count, missing audio, or a noncanonical data root.

- [x] **Step 2: Run preparation tests and verify RED**

  Run `uv run pytest -q tests/test_vqa_qwen_empty_output_diagnostic.py -k prepare`.
  Expected: import failure because the harness does not exist.

- [x] **Step 3: Implement exclusive preparation and freeze verification**

  Use schema version `qwen-empty-output-diagnostic.v1`, experiment ID
  `qwen-aligner-stage05e-20260816-empty-output-v1`, and the private root from
  the design. Bind the exact source prediction-freeze SHA-256, source runner
  evidence, selected WAV hashes/durations, new runner path/hash, arms, model
  revisions, image digest, row order, one-attempt rule, and `$0.00` provider
  cost. Re-verify both executed source hashes before command construction,
  recording, freezing, and status. Write with exclusive creation only.

- [x] **Step 4: Write failing command and one-attempt tests**

  Assert each arm's command contains `-i`, `--network none`, `--read-only`,
  read-only canonical data/cache mounts, pinned image/revisions, no credential
  environment variables, the dedicated module name, and the exact mode:

  ```text
  A: aligner args, --language auto,    --timestamps on
  B: no aligner,   --language auto,    --timestamps off
  C: aligner args, --language auto,    --timestamps off
  D: aligner args, --language English, --timestamps on
  ```

  A second record for the same arm must fail before subprocess execution.

- [x] **Step 5: Implement arm recording and deterministic validation**

  Preserve `command.json`, `raw-results.json`, sanitized `stderr.txt`,
  `terminal.json`, and `validated-results.json` under `arms/<arm_id>/` even on
  nonzero exit or launch exception. Validate count, order, condition ID, audio
  hash, success fields, and aligned chunk bounds exactly as declared.

- [x] **Step 6: Write failing decision-matrix and mutation tests**

  Cover at least these literal outcomes:

  ```text
  A reproduces + D passes all four -> ready_for_fresh_aligned_quality_panel
  A does not reproduce             -> unstable_baseline
  A reproduces + B/C only pass     -> not_ready
  any arm terminal invalid         -> not_ready
  ```

  Mutating the runner, requests, raw output, terminal record, validation, or
  top-level decision after freezing must make status verification fail.

- [x] **Step 7: Implement diagnostic freeze, decision, status, and CLI**

  CLI subcommands are `prepare`, `command --arm`, `record --arm`, `freeze`, and
  `status`. `freeze` requires all four arm attempts, freezes their artifacts,
  records per-arm aggregate success/failure counts, baseline reproducibility,
  explanatory localization, advancing arm or null, and the predeclared final
  decision. It must never expose row IDs or transcript text in CLI output.

- [x] **Step 8: Verify GREEN and commit the harness**

  Run:

  ```bash
  uv run pytest -q tests/test_vqa_qwen_empty_output_diagnostic.py
  uv run ruff check src/trec_video_2026/vqa_qwen_empty_output_diagnostic.py scripts/run_qwen_empty_output_diagnostic.py tests/test_vqa_qwen_empty_output_diagnostic.py
  ```

  Commit with message `feat: add frozen Qwen empty-output diagnostic`.

### Task 4: Preflight and execute the one diagnostic

**Files:**
- Private create: `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05e-qwen-empty-output-diagnostic-v1/`

**Interfaces:**
- Consume: Task 3 CLI and the frozen Stage 0.5d root
- Produce: one immutable four-arm terminal evidence bundle and decision

- [x] **Step 1: Run the cheapest full preflight**

  Verify the canonical `data` symlink, host, ROCm visibility, free disk,
  digest-pinned local image, both exact model snapshots, the new runner and
  imported helper-source hashes, clean private target root, and frozen Stage
  0.5d `predictions_frozen/not_ready`
  status. Run the dedicated runner's no-model stdin/argument smoke before
  preparation.

- [x] **Step 2: Prepare and inspect aggregate-only inputs**

  Run `prepare`; inspect only status, row count, role counts, arm IDs, hashes,
  and command flags. Confirm no question, answer, annotation, Whisper,
  transcript text, grade, metric, or credential key appears.

- [x] **Step 3: Execute arms A through D exactly once**

  Invoke only the harness recorder for each arm in immutable order. Do not run
  a printed Podman command separately. Preserve failures and continue to later
  arms unless host safety or immutable-input verification fails.

- [x] **Step 4: Freeze and verify the diagnostic**

  Run `freeze`, then `status`. Record only aggregate counts, baseline
  reproducibility, advancing arm, final decision, elapsed time, peak memory,
  artifact hashes, and `$0.00` provider cost outside the private root.

### Task 5: Record the result and place Gemini 3.7 Flash

**Files:**
- Modify: `docs/experiments/2026-08-16-qwen-aligner-quality-diagnostic.qmd`
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Create or update: `docs/experiments/artifacts/qwen-aligner-quality-20260816/`
- Regenerate: QMD-derived Markdown/HTML only through Quarto

**Interfaces:**
- Consume: sanitized aggregate diagnostic evidence
- Produce: condition-bound claim and next-test decision

- [ ] **Step 1: Write the observed result separately from interpretation**

  Append the four-arm method, completion denominator, per-arm aggregate
  success counts, baseline reproducibility, decision rule outcome, evidence
  validity, limits, private root, and immutable hashes. Do not include row IDs,
  transcript text, or raw logs.

- [ ] **Step 2: Record the next action from the frozen decision**

  If D advances, specify a new-label-blind-pool quality panel with no reused
  Qwen rows. Otherwise preserve `not_ready` or `unstable_baseline` and name the
  exact reconsideration condition. Never score the four-row diagnostic.

- [ ] **Step 3: Record Gemini 3.7 Flash as a separate proposed smoke**

  Note official stable model ID `gemini-3.7-flash`, 2026-08-13 GA status,
  native video/audio input, structured output, and the need for a separately
  authorized compatibility smoke against the pinned Gemini 3.1 Pro condition.
  Do not claim superiority and do not issue an API call in this task.

- [ ] **Step 4: Render and perform proportional verification**

  Regenerate affected QMD-derived Markdown/HTML. Run the complete focused
  Qwen suite, Ruff, JSON parsing, tracked SHA256SUMS checks, privacy/secret and
  private-ID scans, `git diff --check`, and status verification for both frozen
  roots. Confirm `data/` and `.claude/` are not staged.

- [ ] **Step 5: Request independent review and commit the record**

  Review source, tests, frozen aggregate evidence, and report wording. Resolve
  all actionable findings, rerun verification, and commit only intended
  tracked project files.
