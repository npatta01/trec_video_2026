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

### Frozen promotion rules

These rules and the decisions below were frozen at
`2026-07-24T00:28:27Z`, before
`data/vqa/benchmarks/runs/native-audio-benchmark-v1/` existed and before any
30-video attempt was launched.

A condition is eligible only when:

1. at least 90% of its applicable gate rows are terminal, scoreable,
   exact-request matches with a pinned revision and image, observed ROCm/BF16,
   non-null timing/memory provenance, and no silent evidence loss;
2. its gate quality satisfies at least one predeclared path:
   - **leader:** it is on the leading quality frontier across normalized exact
     match, token F1, and NIST-style METEOR;
   - **complement:** it has at least one normalized-exact correct answer missed
     by every leader;
   - **paired evidence:** a transcript/native-audio treatment has no negative
     exact-match delta, has token-F1 delta at least `+0.10` or METEOR delta at
     least `+0.05` on the four audio rows, and does not materially regress the
     visual controls;
   - **predeclared diagnostic:** it is the minimum matched control needed to
     answer the 12-vs-24-frame or transcript-evidence question;
3. projected local cost is at most 45 minutes and observed peak allocation is
   below 32 GiB per 30-row condition. A same-family condition within `0.01`
   token F1 and `0.03` METEOR of a cheaper condition is redundant unless it
   supplies a complement.

METEOR below was regenerated over the frozen saved gate predictions in the
already pinned Qwen-Omni image using NLTK 3.9.2 and the recorded WordNet and
punkt resources. Latency is mean per-row inference; memory is maximum allocated
accelerator memory. Strata cells are `EM / token F1 / METEOR` in manifest order
`dialogue; sound; temporal; spatial; control`.

| Condition | Scoreable | EM | Token F1 | METEOR | Strata | Mean s | Peak GiB | Exclusive exact wins | Decision |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| MiniCPM-o 4.5 audio-off | 10/10 | .300 | .586 | .414 | `.500/.800/.392; .000/.250/.066; .000/.382/.343; 1.00/1.00/.820; .000/.500/.451` | 7.82 | 19.07 | 0 | **Promote: quality leader** |
| Qwen3.5-4B 24f | 10/10 | .300 | .496 | .357 | `.500/.500/.132; .000/.000/.000; .000/.481/.380; 1.00/1.00/.820; .000/.500/.451` | 44.49 | 15.30 | 0 | **Promote: predeclared 12-vs-24 diagnostic** |
| Qwen3.5-4B 12f | 10/10 | .300 | .475 | .333 | `.500/.500/.132; .000/.000/.046; .000/.624/.541; 1.00/1.00/.820; .000/.250/.128` | 43.82 | 10.56 | 0 | **Promote: matched control** |
| Qwen3.5-4B 12f + Whisper large-v3 | 4/4 audio | .250 | .500 | .271 | `.500/.500/.167; .000/.500/.374; —; —; —` | 8.83 downstream | 10.82 | 0 | **Promote: paired transcript evidence** |
| Qwen3.5-9B 12f | 10/10 | .300 | .493 | .333 | `.500/.500/.132; .000/.250/.066; .000/.467/.464; 1.00/1.00/.938; .000/.250/.066` | 48.18 | 19.68 | 0 | Stop: redundant with 4B/24f |
| Gemma4-12B 12f | 10/10 | .200 | .405 | .372 | `.500/.618/.223; .000/.250/.319; .000/.208/.266; .500/.750/.740; .000/.200/.312` | 16.09 | 25.66 | 0 | Stop: below leaders, no complement |
| InternVL3.5-4B | 10/10 | .200 | .315 | .241 | `.500/.500/.167; .000/.250/.066; .000/.125/.245; .500/.700/.726; .000/.000/.000` | 14.53 | 22.44 | 0 | Stop: below leaders, no complement |
| Molmo2-4B | 10/10 | .100 | .149 | .184 | `.000/.000/.036; .000/.000/.190; .000/.154/.218; .500/.591/.410; .000/.000/.066` | .87 | 9.88 | 0 | Stop: below leaders, no complement |
| MiniCPM-o 4.5 audio-on | 10/10 | .300 | .471 | .283 | `.500/.500/.132; .000/.250/.103; .000/.356/.296; 1.00/1.00/.820; .000/.250/.066` | 10.35 | 19.82 | 0 | Stop: paired F1 `-0.115`, METEOR `-0.131` |
| Qwen2.5-Omni-3B audio-off | 10/10 | .300 | .467 | .287 | see Task 6 evaluation | 398.73 | 75.68 | 0 | Stop: exceeds cost/memory bounds |
| Qwen2.5-Omni-3B audio-on | 10/10 | .300 | .488 | .273 | see Task 6 evaluation | 415.18 | 80.35 | 0 | Stop: METEOR `-0.015`, exceeds bounds |
| Qwen2.5-Omni-7B off/on | 2/2 probe | .000 | .154 | `.132/.045` | one temporal probe only | `74.33/29.07` | `20.57/21.10` | 0 | Stop: no probe quality signal |
| New Voxtral/Qwen3-ASR paths | applicable audio rows only | .250 downstream | .250 downstream | not run | audio strata only | about 10 downstream | about 10.7 | 0 | Stop: no downstream gain; incomplete ASR row |

Whisper large-v3 changes the four-audio subset relative to Qwen3.5-4B/12f by
`0.000` exact match, `+0.250` token F1, and `+0.182` METEOR. MiniCPM audio-on
changes all ten by `0.000`, `-0.115`, and `-0.131`; Qwen2.5-Omni-3B audio-on
changes all ten by `0.000`, `+0.021`, and `-0.015`, with no exact-match
wins/losses. No full-gate condition has an exclusive normalized-exact win.

### Frozen 30 launch decisions

The promoted execution order is serial:

1. `qwen35-4b-12f-frozen30` — required visual-only control;
2. `qwen35-4b-24f-frozen30` — controlled temporal-coverage treatment;
3. `minicpmo45-native-video-audio-off-frozen30` — gate quality leader;
4. `qwen35-4b-12f-whisper-large-v3-frozen30` — exact 12-frame control plus
   quality-gated, independently generated Whisper large-v3 evidence.

Each launch additionally requires an append+fsync promotion record binding the
condition, this manifest/version/hash, exact ordered 30-request SHA-256, pinned
model/revision/image, and the rationale above. The runner refuses a condition
without that pre-launch record.

At checkpoints after rows 3, 10, and 20, stop the active condition if any saved
pass fails exact-request/evidence/provenance validation; if projected wall time
exceeds 45 minutes; if peak allocation reaches 32 GiB; or, after ten attempted
rows, if more than 10% are unscoreable. Stop immediately on CPU fallback,
unpinned identity, label leakage, silent audio/frame/transcript loss, an
unverified timed-out container cleanup, or a surviving competing benchmark
process. Never replace a pass with a retry; later failures remain append-only.

- [x] Generate a gate table with completion, three primary metrics, per-stratum
  results, paired audio changes, latency, memory, and unique correct answers.
- [x] Record the promotion decision and rationale before launching each
  30-video condition.
- [x] Run promoted conditions serially and resumably on the exact frozen panel.
- [x] Verify all expected rows, immutable revisions, evidence checksums, device
  records, and absence of label fields in saved requests.

## Task 9: Report and independently review the recommendation

**Files:**

- Create: `reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md`
- Modify implementation/docs identified by review

- [x] Regenerate metrics and the report from append-only raw predictions.
- [x] Report every condition and stopped candidate, not only winners; distinguish
  verified facts, inferences, and unknowns.
  The final report separates the four clean v2 frozen-30 conditions from the
  complete pre-v2 visual/native-AV candidate inventory, pre-launch safety
  stops, and source/quality/downstream ASR diagnostics.
- [x] Recommend a visual base, frame/spatial policy, transcript option, native
  audio option, and larger-machine shortlist with measured local costs.
- [x] Ask a fresh advisor to review code and experimental findings for leakage,
  evidence mismatch, unfair ablations, scoring defects, fallback, and
  overclaiming.
- [x] Address findings and rerun affected tests, inference rows, and scores.
- [x] Run full verification and confirm no media, transcripts, weights, raw
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

## Task 10: Remediate the fresh post-execution protocol review

The first frozen-30 output remains immutable historical audit evidence.
Protocol-v2 remediation, clean inference, two-phase scoring, deterministic
score regeneration, and the canonical results report are complete. The next
action is a fresh final advisor review of the v2 evidence and report.

- [x] Add an inference-manifest loader that resolves only frozen questions,
  categories, media, and annotation paths without opening annotation content.
- [x] Restrict inference container mounts to exact media, frame-evidence,
  run-audio, repository, and model-cache roots; do not mount label-bearing
  annotation directories.
- [x] Live-attest Qwen ROCm, actual parameter device, and actual parameter
  dtype instead of returning configured strings.
- [x] Recompute and compare the full current evidence manifest before every
  authorization and run; bind each selected attempt to the current manifest.
- [x] Reject nonzero Whisper runner exits before parsing a possible success
  payload.
- [x] Audit competing benchmark containers/processes before launch and at
  checkpoints without killing unrelated work.
- [x] Persist outer chunk/condition wall time, including model/container
  overhead, and enforce the 45-minute stop against that measured value.
- [x] Pin the scoring image, NLTK 3.9.2, punkt-tab SHA-256
  `6073f4561d752e90f3b02bbb80a955100eb90b7f62c5fcfe17c86e8030909522`,
  and WordNet SHA-256
  `6c4365b1018d44f39b284f9a5e44766548b18db9c436c9cfc1b7116230e35711`
  in generated evaluation metadata.
- [x] Mark the first Whisper frozen-30 promotion exploratory/protocol-deviating:
  its four-audio gate could not test the required visual-control guard, and
  the later panel measured visual-control METEOR `-0.1032` and temporal F1
  `-0.0552`.
- [x] Expand stopped ASR reporting into separate Voxtral transcription,
  Voxtral translation, Qwen3-ASR downstream, and Voxtral direct-QA rows with
  completeness, cost, metrics, and unavailable METEOR.
- [x] Obtain independent code review, run the full suite/preflight, then launch
  a new immutable v2 run rather than reusing v1 passes.
- [x] Freeze v2 predictions before any reference load and regenerate scoring
  and the canonical report.
- [ ] Obtain a fresh final advisor approval of the protocol-v2 code, artifacts,
  conclusions, and report.

Protocol-v2 evidence:

- 120/120 VQA rows and 30/30 Whisper rows passed, with zero ROCm failures;
- selected prediction SHA-256
  `8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879`;
- scored prediction SHA-256
  `c95669135b29551ef407e2889ac5e203053e944b00a0968cbe20da1fdfd0be7a`;
- final evaluation SHA-256
  `f610dbd3d468f3b72074c7ace38dec95176c7f47fa213b5ce374d0a9429ad9c0`;
- canonical report:
  `reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md`.
- report-only advisor follow-up added every pre-v2 stopped candidate, separated
  ASR source completion from transcript acceptance and downstream completion,
  and narrowed v2-only claims to frozen-30 comparisons. Final advisor approval
  remains pending.
