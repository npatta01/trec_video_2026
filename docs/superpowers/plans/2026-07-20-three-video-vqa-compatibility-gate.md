# Three-Video VQA Compatibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task by task,
> with `superpowers:test-driven-development` for every behavior change.

**Goal:** Execute a reproducible three-video VQA portability gate on the local
AMD/ROCm host and produce independently reviewed code and findings without
starting the 100-example benchmark.

**Architecture:** A deep benchmark module accepts a resolved manifest and run
configuration, creates deterministic cached evidence, invokes model-specific
runner adapters through a JSON subprocess seam, evaluates normalized outputs,
and renders a provenance-rich report. Model dependencies stay in a pinned
rootless ROCm container; generated evidence and results stay under ignored
`data/` paths.

**Tech stack:** Python 3.12, pytest, FFmpeg/ffprobe, rootless Podman, ROCm
PyTorch, Hugging Face Hub, Transformers/model-specific processors, Whisper,
NIST VQA scorer submodule.

## Fixed experiment scope

- Examples: `BDKkVYmWqdM`, `O60MtMuSpf8`, `Fk_j6re_Uiw` from labeled 2025
  training data.
- Controlled evidence: 12 bin-midpoint frames and one frozen timestamped
  Whisper transcript.
- Baseline first: Qwen3.5-4B without and with transcript.
- Then one-example load probes and, on success, three-example runs for
  Qwen3.5-9B, Molmo2-4B, Gemma 4 12B, InternVL3.5-4B, and MiniCPM-o 4.5.
- Hard limit: 12 wall-clock hours; never launch the 100-example run.
- Current immutable Hub revisions and dry-run sizes are recorded in the run
  configuration: 9.3 GB, 19.3 GB, 19.4 GB, 23.9 GB, 9.5 GB, and 20.0 GB,
  respectively.

## Task 1: Add the fixed manifest and domain records

**Files:**

- Create: `configs/vqa/three_video_gate.json`
- Create: `src/trec_video_2026/vqa_benchmark.py`
- Create: `tests/test_vqa_benchmark.py`

- [ ] Write failing tests for annotation/media resolution, stream validation,
  label separation, and deterministic manifest serialization.
- [ ] Implement immutable example, evidence, attempt, and result records plus a
  manifest loader that resolves paths only beneath the shared dataset root.
- [ ] Verify the three real examples resolve and contain both video and audio.

## Task 2: Build deterministic evidence caching

**Files:**

- Modify: `src/trec_video_2026/vqa_benchmark.py`
- Modify: `tests/test_vqa_benchmark.py`

- [ ] Write failing tests for 12 bin-midpoint timestamps, source fingerprints,
  cache keys, checksum verification, and stale-cache rejection.
- [ ] Implement ffprobe-backed media metadata and FFmpeg frame extraction with
  atomic cache publication beneath `data/vqa/benchmarks/evidence/`.
- [ ] Create a tiny generated-video integration fixture and verify all 12 image
  files and timestamps.

## Task 3: Add the runner protocol and resumable orchestrator

**Files:**

- Create: `src/trec_video_2026/vqa_gate.py`
- Create: `tests/test_vqa_gate.py`
- Modify: `pyproject.toml`

- [ ] Write failing tests for JSON request/result validation, timeout/failure
  classification, append-only attempts, successful-result preservation, and
  resume behavior using fake runner commands.
- [ ] Implement `vqa-gate preflight`, `prepare`, `run`, and `report` commands.
- [ ] Ensure references cannot appear in runner requests or prompt payloads.
- [ ] Run a fake three-example end-to-end gate and regenerate its report from
  saved results.

## Task 4: Add Whisper transcript generation

**Files:**

- Create: `containers/vqa-gate/Containerfile`
- Create: `scripts/vqa_gate_whisper.py`
- Create: `tests/test_vqa_gate_whisper.py`
- Modify: `src/trec_video_2026/vqa_gate.py`

- [ ] Write failing tests around transcript-result schema and timestamp/prompt
  formatting with a fake transcriber.
- [ ] Pin the ROCm image by digest and install locked Python dependencies in a
  reproducible local image without exposing a service.
- [ ] Implement batch-size-one Whisper transcription and record model revision,
  language, timestamps, elapsed time, and device.
- [ ] Generate and manually inspect transcripts for the three gate clips.

## Task 5: Add the controlled-frame model runner

**Files:**

- Create: `scripts/vqa_gate_transformers.py`
- Create: `tests/test_vqa_gate_transformers.py`
- Create: `configs/vqa/model_runs.json`

- [ ] Write failing tests for prompt assembly, timestamped frame ordering,
  transcript inclusion/omission, answer normalization, deterministic decoding,
  and provenance output using fake processor/model objects.
- [ ] Implement adapter branches only where the processors genuinely differ,
  behind the common JSON runner protocol.
- [ ] Execute a Qwen3.5-4B one-example ROCm probe before acquiring later model
  weights; then run both baseline conditions on all three examples.
- [ ] Probe Qwen3.5-9B, Molmo2-4B, Gemma 4 12B, and InternVL3.5-4B sequentially,
  advancing only successful probes to all three examples.

## Task 6: Add the native MiniCPM audio/video probe

**Files:**

- Create: `scripts/vqa_gate_minicpm.py`
- Create: `tests/test_vqa_gate_minicpm.py`
- Modify: `configs/vqa/model_runs.json`

- [ ] Write a fake-model test proving original video/audio is passed and a
  transcript is not substituted into the native condition.
- [ ] Implement offline, non-streaming inference with bounded generation and the
  common result schema.
- [ ] Run one ROCm probe and, only if it passes, all three examples. Record any
  Nvidia-only or kernel incompatibility precisely.

## Task 7: Add safe evaluation and pinned NIST parity scoring

**Files:**

- Add submodule: `external/nist-vqa`
- Create: `src/trec_video_2026/vqa_evaluation.py`
- Create: `tests/test_vqa_evaluation.py`
- Modify: `src/trec_video_2026/vqa_gate.py`
- Modify: `.gitignore`

- [ ] Add `https://github.com/gawad-nist/VQA` as a submodule pinned to
  `e06cb7854d7defe53e86412c2bba0ca5f636ea40`.
- [ ] Write failing tests for normalized exact match, token F1, missing metrics,
  scorer isolation, CSV discovery hygiene, and mode labels.
- [ ] Implement local diagnostics and an isolated `nist-2025-parity` adapter
  that never interpolates answers into a shell command.
- [ ] Score the gate outputs; treat unavailable remote STS as unavailable and do
  not interpret single-answer NDCG.

## Task 8: Produce and independently review findings

**Files:**

- Create: `reports/trecvid_vqa_2026/three_video_gate_results_2026-07-20.md`
- Modify files identified by review as necessary.

- [ ] Generate the report from raw JSONL, including every pass/failure, answer,
  reference, metric, latency, peak memory, revision, runtime digest, and caveat.
- [ ] Run all tests and targeted integration checks; verify no dataset, weight,
  transcript, prediction, or raw log is staged.
- [ ] Ask a fresh advisor to review the code diff and findings for leakage,
  evidence mismatch, scoring defects, ROCm fallback, and overclaiming.
- [ ] Address actionable findings and rerun every affected check.
- [ ] Stop and report to the user. Do not create or run a 100-example manifest.

## Verification commands

```bash
uv run pytest -q
uv run vqa-gate preflight --config configs/vqa/three_video_gate.json
uv run vqa-gate report --run-dir data/vqa/benchmarks/runs/three-video-gate
git diff --check
git status --short
```

The final handoff must distinguish verified results from unsupported candidates
and unknowns, link the canonical spec/plan/report, and explicitly confirm that
the 100-example run was not started.
