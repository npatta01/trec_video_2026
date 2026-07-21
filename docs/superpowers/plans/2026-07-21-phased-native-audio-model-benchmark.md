# Phased Native-Audio Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` task by task, with
> `superpowers:test-driven-development` for behavior changes,
> `superpowers:systematic-debugging` for compatibility failures, and
> `superpowers:verification-before-completion` before claims or handoff.

**Goal:** Recommend an open visual/video VQA base model and audio-evidence
recipe using a reproducible ten-video gate and promoted 30-video benchmark on
labeled 2025 training data.

**Architecture:** Extend the existing benchmark module rather than create a
second harness. A frozen manifest resolves shared media and private evaluation
labels, evidence builders cache deterministic frames/audio/transcripts, and
isolated model runners communicate through the existing JSON protocol. Every
native-audio family exposes a matched audio-off/audio-on switch. Append-only
attempts feed one scorer and a provenance-rich report.

**Tech stack:** Python 3.12, pytest, FFmpeg/ffprobe, rootless Podman, ROCm
PyTorch, Hugging Face Transformers/model remote code, existing NIST scorer
submodule.

## Fixed experimental scope

- Manifest version: `trecvid-native-audio-2026-07-21-v1`.
- Eligible pool: 464 IDs common to annotation JSON, parsed CSV, and downloaded
  media after excluding the original three gate videos.
- Frozen panel: 30 examples, six in each of dialogue/speech, non-speech audio,
  temporal/action, spatial/OCR/color, and visual-control strata.
- Compatibility gate: the exact ten IDs marked in the design, two per stratum.
- Labels remain outside model requests and prompts.
- Newly integrated models receive a one-video probe before the ten-video gate.
- Conditions advance to 30 only under the frozen promotion rules.
- Runs are serial and resumable; model caches are retained and never pruned.

## Task 1: Add and validate the frozen manifest

**Files:**

- Create: `configs/vqa/native_audio_benchmark_30.json`
- Modify: `src/trec_video_2026/vqa_benchmark.py`
- Modify: `tests/test_vqa_benchmark.py`

- [ ] Write failing tests for manifest version, exact 30 IDs, five strata,
  exact ten gate IDs, shared-root resolution, media-extension resolution, and
  reference separation.
- [ ] Add the frozen questions/categories and resolve references only from the
  authoritative annotation JSON at evaluation time.
- [ ] Validate that all 30 files contain video and audio streams and record
  source fingerprints, durations, and stream summaries.
- [ ] Verify the deterministic gate-selection hash and emit a resolved manifest
  beneath ignored benchmark data.

## Task 2: Generalize evidence and paired scoring

**Files:**

- Modify: `src/trec_video_2026/vqa_benchmark.py`
- Modify: `src/trec_video_2026/vqa_evaluation.py`
- Modify: `tests/test_vqa_benchmark.py`
- Modify: `tests/test_vqa_evaluation.py`

- [ ] Write failing tests for configurable 12/24 frame policies, cache-key
  separation, stratum aggregates, paired audio deltas, wins/losses, and
  deterministic bootstrap intervals.
- [ ] Generalize the existing frame cache without changing the established
  12-frame timestamps.
- [ ] Add paired and per-stratum summaries while keeping exact match, token F1,
  METEOR, and NIST parity behavior backward compatible.
- [ ] Verify score regeneration from saved fake predictions and prove reference
  fields never enter a runner request.

## Task 3: Repair and control MiniCPM-o 4.5

**Files:**

- Modify: `src/trec_video_2026/vqa_specialist_runner.py`
- Modify: `tests/test_vqa_specialist_runner.py`
- Modify: `configs/vqa/model_runs.json`

- [ ] Reproduce the `[15,4096]` target versus `[14,4096]` source failure in the
  smallest available unit/integration boundary before changing code.
- [ ] Add failing tests for explicit audio-off/audio-on modes, identical visual
  evidence, no transcript substitution, processor-native input, and bounded
  structured failures.
- [ ] Compare the current hand-interleaved chunks against MiniCPM's native video
  helper and `merge_audio_from_same_content=False`; select the documented path
  that avoids boundary rounding without truncating meaningful audio.
- [ ] Re-run the original failed speech clip plus two frozen-gate clips in both
  modes. Record completion, answers, latency, memory, revision, and image digest.
- [ ] Run the matched ten-video pair only if the three-video compatibility check
  has no silent evidence loss or unexplained shape failure.

## Task 4: Run cached visual and transcript baselines on ten

**Files:**

- Modify: `configs/vqa/model_runs.json`
- Modify: `src/trec_video_2026/vqa_transformers_runner.py`
- Modify: `src/trec_video_2026/vqa_whisper.py` only if required
- Modify: `src/trec_video_2026/vqa_seamless.py` only if required
- Modify corresponding tests when behavior changes

- [ ] Add configurations for Qwen3.5 4B at 12 and 24 frames and Qwen3.5 9B at
  12 frames.
- [ ] Run Molmo2 4B, InternVL3.5 4B, and Gemma 4 12B using their already-pinned
  revisions and runtime images.
- [ ] Generate/reuse Whisper large-v3, Whisper large-v3-turbo, and
  SeamlessM4T-v2 evidence on the four audio gate examples; run the same frozen
  downstream model/prompt for each transcript condition.
- [ ] Check every attempt for device, precision, evidence checksum, terminal
  completion, and answer shape before scoring.

## Task 5: Add Voxtral and Qwen3-ASR transcript paths

**Files:**

- Create: `src/trec_video_2026/vqa_audio_runner.py`
- Create: `tests/test_vqa_audio_runner.py`
- Create or modify: `containers/vqa-gate/Containerfile.audio`
- Modify: `configs/vqa/model_runs.json`

- [ ] Write fake-processor/model tests for isolated audio input, question-aware
  versus pure transcription modes, language metadata, timestamp/text output,
  deterministic decoding, and error classification.
- [ ] Add `mistralai/Voxtral-Mini-3B-2507` and
  `Qwen/Qwen3-ASR-1.7B` at immutable revisions in isolated runtimes.
- [ ] Perform one-audio probes before processing the four gate audio examples.
- [ ] If Qwen3-ASR output is not English, pass it through one frozen,
  independently recorded translation step before downstream VQA.
- [ ] Retain raw transcript output for a label-blind semantic audit; never use
  reference answers to clean or select a transcript.

## Task 6: Add Qwen2.5-Omni native AV

**Files:**

- Create: `src/trec_video_2026/vqa_omni_runner.py`
- Create: `tests/test_vqa_omni_runner.py`
- Create: `containers/vqa-gate/Containerfile.qwen-omni`
- Modify: `configs/vqa/model_runs.json`

- [ ] Write tests proving `use_audio_in_video` is passed consistently through
  preprocessing and generation, audio-off preserves video evidence, talker is
  disabled, output is text-only, and references cannot leak.
- [ ] Resolve and record immutable revisions and dry-run sizes for
  Qwen2.5-Omni 3B and 7B before downloading.
- [ ] Build a ROCm/SDPA-first isolated runtime and verify accelerator execution;
  do not silently fall back to CPU or an unrecorded checkpoint.
- [ ] Probe 3B on one gate video in both modes, then run the matched ten if it
  completes reliably.
- [ ] Probe 7B in both modes; run ten only if it fits with headroom and provides
  useful capacity evidence or the 3B result warrants it.

## Task 7: Probe optional family and heavyweight candidates

**Files:**

- Modify: `configs/vqa/model_runs.json`
- Reuse or extend the native-AV runner only behind tested adapter seams
- Modify corresponding tests

- [ ] Review Baichuan-Omni's additional license terms before acquiring weights;
  if acceptable locally, run only a one-to-three-video diversity probe.
- [ ] Record Qwen3-Omni 30B-A3B-Thinking as a larger-machine condition. Attempt
  one local probe only if measured free unified memory, runtime compatibility,
  and expected headroom make it safe after the smaller gates.
- [ ] Do not promote a heavyweight condition merely because it loads; require
  useful answer-quality evidence relative to its latency and memory.

## Task 8: Apply promotion rules and run the frozen 30

**Files:**

- Create: `data/vqa/benchmarks/runs/native-audio-benchmark-v1/` (ignored)
- Modify: `docs/superpowers/plans/2026-07-21-phased-native-audio-model-benchmark.md`

- [ ] Generate a gate table with completion, three primary metrics, per-stratum
  results, paired audio changes, latency, memory, and unique correct answers.
- [ ] Record the promotion decision and rationale before launching each
  30-video condition.
- [ ] Run promoted conditions serially and resumably on the exact frozen panel.
- [ ] Verify all expected rows, immutable revisions, evidence checksums, device
  records, and absence of label fields in saved requests.

## Task 9: Report and independently review the recommendation

**Files:**

- Create: `reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md`
- Modify implementation/docs identified by review

- [ ] Regenerate metrics and the report from append-only raw predictions.
- [ ] Report every condition and stopped candidate, not only winners; distinguish
  verified facts, inferences, and unknowns.
- [ ] Recommend a visual base, frame/spatial policy, transcript option, native
  audio option, and larger-machine shortlist with measured local costs.
- [ ] Ask a fresh advisor to review code and experimental findings for leakage,
  evidence mismatch, unfair ablations, scoring defects, fallback, and
  overclaiming.
- [ ] Address findings and rerun affected tests, inference rows, and scores.
- [ ] Run full verification and confirm no media, transcripts, weights, raw
  logs, credentials, or unrelated user files are staged.

## Verification commands

```bash
uv run pytest -q
uv run vqa-gate preflight \
  --config configs/vqa/native_audio_benchmark_30.json \
  --dataset-root data
git diff --check
git status --short
```

Real-run verification additionally checks the expected condition/example row
counts, runner-request schema, device/precision provenance, source/evidence
fingerprints, and reproducible score regeneration. A successful process exit
alone is not sufficient.
