# Qwen ASR + Forced Aligner Eight-Video Quality Test Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fresh, label-isolated eight-video English exact-speech comparison of Qwen3-ASR plus its forced aligner against the frozen Whisper transcript condition.

**Architecture:** Extend the existing immutable canary harness from its fixed two-row operational probe to an explicit quality-panel mode without changing or reopening the three frozen Stage 0.5 roots. The new mode selects eight fresh rows from the preserved oracle-speech-10 media boundary, freezes only audio requests and model/runtime bindings, executes one offline batch, freezes raw predictions, and only then joins private answer references and Whisper text for a mechanism diagnostic.

**Tech Stack:** Python 3.12, pytest, rootless Podman, the already verified local Hugging Face snapshots, JSON/SHA-256 freeze artifacts, and Quarto reporting.

## Global Constraints

- Use a fresh root `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/` and experiment ID `qwen-aligner-stage05d-20260816-quality-v1`.
- Select exactly eight English exact-speech rows from `data/vqa/benchmarks/oracle-speech-10/`, excluding all IDs in the prior Qwen3-ASR attempts and both Stage 0.5c rows; do not add rows to pad the panel.
- Requests contain only run/condition/video identity, absolute read-only WAV path, and `mode=transcription`; no question, answer, annotation, reference, map, prior transcript, grade, or metric enters inference.
- Match byte-identical source WAVs to the frozen Whisper evidence; bind every audio SHA-256, duration, model revision, runtime image, runner hash, row order, and one-attempt limit before execution.
- Use the already verified ASR revision `7278e1e70fe206f11671096ffdd38061171dd6e5`, forced-aligner revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, runtime image `localhost/trec-vqa-audio-rocm@sha256:faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234`, ROCm `cuda:0`, BF16, SDPA, and `--network none`.
- Run one local Podman batch only. Paid compute and external API spend must remain `$0.00`; preserve any failure terminally and never retry or relabel it.
- Do not claim WER, human-transcript fidelity, Whisper replacement, downstream VQA improvement, or global multilingual robustness. Phrase containment against dataset answers is a task-specific mechanism diagnostic; bounded aligned timestamps are operational validity, not alignment accuracy.
- Keep all row-level artifacts private under the ignored canonical data root. Tracked documentation may contain only aggregate counts, hashes, and sanitized paths.

---

### Task 1: Generalize the frozen canary contract to a quality panel

**Files:**
- Modify: `src/trec_video_2026/vqa_qwen_aligner_canary.py`
- Modify: `scripts/run_qwen_aligner_canary.py`
- Test: `tests/test_vqa_qwen_aligner_canary.py`

**Interfaces:**
- Preserve `select_panel()` and all existing two-row behavior unchanged.
- Add a deterministic quality-panel selector and preparation path that validate an exact eight-row label-free selection.
- Make result validation, recording, freeze verification, status, and postfreeze scoring derive their denominator from frozen request/protocol row count rather than the old two-row constant.

- [x] **Step 1: Add failing tests for eight-row selection, validation, and score comparison**

  Use synthetic eight-row audio fixtures and assert that previously used IDs are excluded, source hashes/durations are frozen, requests have no label-bearing keys, validation requires all eight ordered rows, and postfreeze scoring reports both Qwen and Whisper phrase containment.

- [x] **Step 2: Run the new tests and confirm RED**

  Run `uv run pytest -q tests/test_vqa_qwen_aligner_canary.py -k quality_panel`; expect failures because the current harness requires exactly two rows.

- [x] **Step 3: Implement the minimal generic denominator and quality selection path**

  Add a quality selection schema with `selection_kind`, `panel_size`, and ordered rows; preserve the existing target/control schema for the old canary. Derive `required_count` from `protocol.row_count` or the request list, and reject any mismatch, duplicate ID, missing audio, hash mismatch, or label-bearing field.

- [x] **Step 4: Add the quality CLI flow and run focused tests**

  Add a quality-panel selection command that consumes only the frozen oracle manifest/transcript metadata and prior-Qwen ID files. Generalize score input loading to accept all frozen result IDs while reading annotation answers only after prediction freeze. Run the canary/audio/model-verifier focused suite and Ruff.

- [x] **Step 5: Commit the harness extension**

  Commit only source and tests with `git commit -m "feat: support Qwen aligner quality panels"` after the focused suite is green.

### Task 2: Freeze the eight-video panel and run one offline batch

**Files:**
- Private create: `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/`
- Use: `data/vqa/benchmarks/oracle-speech-10/automatic-transcripts.whisper-large-v3.json`
- Use: `data/vqa/benchmarks/oracle-speech-10/manifest.json`
- Use: `data/vqa/benchmarks/oracle-speech-10/audio/`

- [x] **Step 1: Verify the canonical data link, cache, image, GPU, and fresh root**

  Confirm the data symlink, host, exact local image, verified model caches, ROCm/BF16 smoke test, free disk, and absence of the new root before creating it.

- [x] **Step 2: Select and prepare exactly eight fresh English speech rows**

  Use the frozen oracle metadata and an explicit prior-Qwen exclusion set; inspect only aggregate selection metadata before preparation. Require target IDs to be absent from Stage 0.5c and the preserved task5/failure-panel Qwen ASR attempts.

- [x] **Step 3: Inspect the generated command and run the no-model stdin smoke**

  Require `-i`, `--network none`, `--read-only`, pinned image/revisions, read-only canonical data/cache mounts, explicit `HF_HOME` and `HUGGINGFACE_HUB_CACHE`, no credentials, and the exact runner hash. Run the existing no-model stdin probe before the sole inference batch.

- [x] **Step 4: Record the one batch attempt**

  Invoke only the recorder; do not execute the printed Podman command separately. Preserve stdout, sanitized stderr, terminal status, validation, and command hash even on failure.

- [x] **Step 5: Freeze predictions and verify status**

  Require a verified prediction freeze before any annotation or Whisper text is opened. The operational gate is `8/8` completed rows with eight valid ordered aligned sequences; otherwise the quality comparison is unusable.

### Task 3: Score the frozen diagnostic and record the result

**Files:**
- Modify: `docs/experiments/2026-08-14-vqa-model-verifier-canaries.qmd`
- Modify: `docs/experiments/artifacts/vqa-model-verifier-canaries-20260814/README.md`
- Modify: `docs/experiments/artifacts/vqa-model-verifier-canaries-20260814/stage0-summary.json`
- Modify: `docs/experiments/artifacts/vqa-model-verifier-canaries-20260814/SHA256SUMS`
- Modify: `docs/project-status/research-speech-transcripts.qmd`
- Regenerate: affected Markdown/HTML report surfaces.

- [ ] **Step 1: Score only after prediction freeze**

  Join private `data/vqa/2025/training/annotations/<video_id>.json` answers and the frozen Whisper text only after prediction-freeze verification. Report completion, valid alignments, Qwen/Whisper acceptance, and phrase containment as aggregate metrics; keep row-level values private.

- [x] **Step 2: Apply the predeclared interpretation boundary**

  Treat `8/8` as an operationally usable mechanism diagnostic. A positive signal supports only a follow-up downstream matched test; it does not promote Qwen, establish transcript fidelity, or justify full-panel reruns.

- [x] **Step 3: Update sanitized evidence and render reports**

  Add the private root and aggregate hashes, regenerate QMD-derived Markdown/HTML, verify JSON and `SHA256SUMS`, scan for private IDs/paths/secrets, and confirm no dataset content is staged.

- [x] **Step 4: Run proportional final verification and commit**

  Run the focused suite, Ruff, `git diff --check`, status for all frozen roots, and the evidence hash check. Commit source, tests, plan, and sanitized report changes without staging `data/`.

## Execution outcome (2026-08-16)

The fresh root was prepared and executed once. The container exited normally at
zero paid cost, but only `6/8` rows completed with valid alignment. The two
remaining rows returned `ValueError: Qwen3-ASR returned empty text`. The frozen
decision is therefore `not_ready`; no private answers or Whisper text were
opened, and no post-freeze comparison score was created. The private terminal,
raw-results, validation, and prediction-freeze artifacts remain under the
declared root for audit. This is a Qwen empty-output diagnostic, not a Whisper
replacement or transcript-quality result.
