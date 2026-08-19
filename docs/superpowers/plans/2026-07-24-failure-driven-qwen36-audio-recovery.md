# Failure-driven Qwen3.6 and Audio Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify a stronger TRECVID VQA base model by repairing the Whisper failure path and testing Qwen3.6 plus genuinely audio-capable alternatives on a frozen, advisor-audited 12-video failure panel.

**Architecture:** Preserve the completed protocol-v2 benchmark as immutable evidence. Build label-free inference requests for one frozen diagnostic panel, run controlled visual-only, transcript, and native-audio comparisons, then score in a separate label-bearing phase. Start with deterministic local compatibility probes; use a bounded dstack task with durable private Hugging Face Bucket artifacts only when the runtime and evidence contract pass locally.

**Tech Stack:** Python 3.12, pytest, Hugging Face Transformers, FFmpeg, rootless Podman/ROCm, dstack, private Hugging Face Buckets, existing VQA runner and evaluator.

## Frozen advisor panel

Keep this exact order:

1. `NN7G28sBOlY` — multilingual dialogue and answer selection.
2. `NUmb2P3Bzu8` — multilingual dialogue plus prosody/gesture.
3. `IRTTtnSRTyY` — non-speech popping; false ASR boilerplate.
4. `IWSPAfCKtV8` — end-of-video bell; irrelevant speech.
5. `MdMKHhFFqJQ` — instrument identification; irrelevant transcript.
6. `OzNdzgQi1CA` — music identity; speech distractor.
7. `CH8Dp7EgW4I` — temporal coverage.
8. `EtNOkUJd5w8` — before/after reasoning plus accepted ASR boilerplate.
9. `Gp4LVTn3YbY` — temporal action plus unreliable spoken reaction.
10. `KgOvCPOX-Ms` — fine-grained terminal action.
11. `EMI_anpHII4` — visual control and deterministic no-speech ASR loop.
12. `LusRurRrmf0` — visual count control plus irrelevant Hindi speech.

Panel roles are 2 multilingual dialogue, 4 non-speech/music, 4 temporal/action, and 2 visual/ASR-distractor controls.

## Global constraints and promotion gates

- Never include references or label-derived hints in inference requests, prompts, filenames, dstack inputs, or model artifacts.
- Freeze inference requests and their hashes before label-bearing scoring.
- Use two external phases: label-free prediction freezing, then read-only prediction evaluation with labels. Do not use the rejected single-container `score()` boundary.
- Preserve the completed v2 run and all attempts; write follow-up artifacts append-only under an ignored run directory.
- Pin model revisions, container digests, package lockfiles, prompts, evidence hashes, and generation settings.
- Do not promote on panel aggregate alone. Require at least 3 net blinded human-adjudicated true-error wins across the 10 substantive failures, no loss on either visual control, no new hallucinated audio claims, and no aggregate token-F1 or METEOR regression.
- Whisper/transcript promotion additionally requires zero accepted repetition/boilerplate on `IRTTtnSRTyY`, `EtNOkUJd5w8`, and `EMI_anpHII4`; retained detected language and confidence; no regression on `EtNOkUJd5w8`, `Gp4LVTn3YbY`, or `LusRurRrmf0`; and at least one improvement on `NN7G28sBOlY` or `NUmb2P3Bzu8`.
- Native-audio promotion requires at least 2/4 non-speech wins and at least 1/2 multilingual-dialogue wins with no temporal/control loss.

---

### Task 1: Freeze the label-free failure panel

**Files:**
- Create: `configs/vqa/failure_panel_12.json`
- Create: `configs/vqa/failure_panel_12_inference.json`
- Modify: `tests/test_vqa_benchmark.py`

- [x] Write a failing test asserting the exact IDs/order, absence of annotation/reference fields in the inference manifest, canonical dataset media resolution, and stable SHA-256 identity.
- [x] Generate the panel manifest from the completed frozen-30 manifest without consulting labels.
- [x] Generate the inference-only projection and record both hashes in this plan:
  - `failure_panel_12.json`: `ad76ec57b964bf391f763e7f82b1671475cddaa90814727d9c40ee7c6310ccbd`
  - `failure_panel_12_inference.json`: `d2c5515b8c7398f326931e2f993f3fd6eac8e70d9ef01d1a1bd4afcf3f6792e8`
- [x] Run targeted tests and the full unit suite (`25 passed`; `444 passed`).

### Task 2: Diagnose and repair Whisper at the ASR seam

**Files:**
- Modify: `src/trec_video_2026/vqa_whisper.py`
- Modify: `tests/test_vqa_whisper.py`
- Create ignored:
  `data/vqa/benchmarks/followups/failure-panel-2026-07-24/whisper-fixed-probes/`

- [x] Preserve the deterministic red reproduction for `EMI_anpHII4`: 30.011813 seconds of valid PCM currently yields 284 words, 9.462941 words/sec, repeated chunks, and timestamps extending to about 59.98 seconds.
- [x] Add regression tests for duration-bounded timestamps, repeated-window/phrase rejection, known boilerplate rejection, and persisted source-language/no-speech metadata.
- [x] Test decoding repetition/VAD safeguards and language-aware transcription versus forced translation with Whisper large-v3 and large-v3-turbo.
- [x] Implement the smallest validated fix. Retain both original-language transcript and English translation rather than silently replacing one with the other.
- [x] Retain rejected attempts and route transcript text downstream only when quality is accepted. The repaired large-v3 probe rejected timestamp overrun; large-v3-turbo returned empty and was rejected.
- [x] Run targeted and full tests; root cause is no-speech/music hallucination plus missing timestamp/repetition/empty-output gates, not corrupt audio.

### Task 3: Add a Qwen3.6 compatibility condition

**Files:**
- Modify: `src/trec_video_2026/vqa_transformers_runner.py`
- Modify: `tests/test_vqa_transformers_runner.py`
- Create: `configs/vqa/qwen36_a4_runtime_requirements.txt`
- Create: `configs/vqa/qwen36_a4_runtime_lock.txt`

- [x] Add runner tests for Qwen3.6 label-free requests and strict live BF16
  accelerator attestation.
- [x] Verify `Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  and its 55.6 GB dry-run footprint.
- [x] Build a digest-pinned runtime that supports Qwen3.6 and verify exact
  interpreter/package versions and BF16 accelerator placement.
- [x] Replace the local 27B probe with the stronger bounded dstack
  `Qwen/Qwen3.6-35B-A3B` comparison after the exact runtime preflight passed.
- [x] Run the frozen 12-frame, 12-video visual-only comparison. The larger model
  passed technically but did not earn promotion, so no 24-frame large-model
  expansion was justified.

### Task 4: Run the larger Qwen3.6 comparison on dstack

**Files:**
- Create: `dstack/qwen36-failure-panel.dstack.yml`
- Create: `scripts/run_failure_panel_dstack.sh`
- Create: `tests/test_failure_panel_dstack.py`

- [x] Pin an image containing an exact Python/uv runtime and the Qwen3.6-compatible inference dependencies.
- [x] Add a wrapper test that exercises the real nested child argv, validates the label-free manifest/hash, preserves workload and artifact-sync return codes, and rejects an existing immutable run prefix.
- [x] Use the private path `hf://buckets/Npatta01/trec_video_2026/experiments/<immutable-run-name>/`.
- [x] Query a closed compatible GPU/provider pool with a hard hourly price cap and bounded native no-capacity retry. Nine main-run offers and twelve preflight offers were available.
- [x] Preview with `echo "n" | dstack apply -f dstack/qwen36-failure-panel.dstack.yml` and preserve the plan output.
- [x] Run bounded preflights through A10. A10 completed with exit code zero
  and attested Python 3.12.3, PyTorch 2.10.0+cu128, CUDA 12.8,
  Transformers 5.15.0.dev0 at `cbf4d720...`, live H100-80GB access, BF16
  support, Qwen3.6 MoE architecture support, and 96 compatible packages.
- [x] Submit one main run as immutable attempt `qwen36-failure-panel-20260724-a10`.
- [x] Run `Qwen/Qwen3.6-35B-A3B` visual-only on the same 12 examples.
- [x] Verify remote object listing, download, manifest hashes, model/runtime
  attestation, and terminal workload/sync return codes. Main A10 completed with
  workload and sync return code zero.

### Task 5: Test targeted non-Qwen/audio alternatives

**Files:**
- Create ignored:
  `data/vqa/benchmarks/followups/failure-panel-2026-07-24/asr-attempts/`
- Create ignored:
  `data/vqa/benchmarks/followups/failure-panel-2026-07-24/native-audio-diagnostic/`

- [x] Run MiniCPM-o 4.5 audio-on versus audio-off on all 12. Native audio
  improved the audio subset but regressed temporal answers and was not promoted
  globally.
- [x] Test Qwen3-ASR-1.7B as a genuinely different ASR condition on ROCm: it avoided Whisper's 284-word loop and produced coherent French/Malay native transcripts, but the current downstream pair did not improve the two multilingual answers.
- [x] Do not broaden to additional general VLMs: residuals now point to
  category-specific ASR, audio recognition, music retrieval, and temporal
  sampling rather than another undirected model-family sweep.
- [x] Keep model-specific failures and ROCm/runtime incompatibilities as
  evidence. No inference silently fell back to CPU or another precision.

### Task 6: Score, adjudicate, review, and render

**Files:**
- Create ignored:
  `data/vqa/benchmarks/followups/failure-panel-2026-07-24/scoring/`
- Create: `reports/trecvid_vqa_2026/failure_panel_qwen36_audio_results_2026-07-24.md`
- Update derived:
  `/home/npatta01/codex-rendered/plans/trec-video-failure-panel-qwen36-audio-20260724.html`

- [x] Have an independent advisor inspect phase-one and phase-two scorer
  commands before scoring.
- [x] Freeze predictions in a label-free container from staged, hashed source;
  evaluate only after the freeze boundary.
- [x] Evaluate normalized exact match, token F1, and METEOR in the separate
  label-bearing phase.
- [x] Blindly adjudicate semantic ambiguities and report automatic metrics and
  adjudication separately.
- [x] Obtain independent audits of failures, regressions, ASR routing, runtime
  attestations, hashes, and promotion decisions.
- [x] Render and verify the private tailnet-only HTML artifact.

## Current status

- [x] Independent advisor audited all raw protocol-v2 failures and approved the frozen 12-video panel.
- [x] Whisper anomaly reproduced twice with identical output; audio corruption and extraction duration are ruled out.
- [x] Local resources verified: 116 GiB available RAM and 2.6 TiB free disk.
- [x] Qwen3.6-27B model revision and 55.6 GB download footprint verified.
- [x] Failure-panel manifest and tests implemented.
- [x] Whisper root cause fixed and regression-tested.
- [x] Qwen3-ASR ROCm probe and two-video downstream VQA comparison completed.
- [x] Qwen3.6 exact runtime preflight completed successfully on dstack H100-80GB.
- [x] Qwen3.6 and native-audio experiments completed.
- [x] Independent failure audit and canonical Markdown report completed.
- [x] Responsive HTML report rendered and verified on the private portal.
