# Nemotron 3 Nano Omni frozen-30 baseline replacement

> **For Codex:** Use `superpowers:subagent-driven-development` to execute
> Tasks 1-4 with strict RED-GREEN-REFACTOR cycles. The primary agent owns the
> sequential cloud gates in Tasks 5-7.

**Goal:** Determine whether native audio-video Nemotron should replace the
current Qwen3.5-4B 12-frame VQA baseline on the complete frozen 30-video panel.

**Architecture:** Extend the existing pinned Nemotron vLLM adapter with one
explicit 30-row paired mode, while preserving all earlier run identities and
contracts. Freeze 60 adjacent audio-off/audio-on predictions before any label
access. A separate fail-closed evaluator validates both the original Qwen v2
freeze and the new Nemotron freeze from independent SHA-256 anchors, then
scores and blindly grades the three conditions.

**Tech stack:** Python 3.12, pytest, Bash, dstack 0.20.27, CUDA on one L40S
48 GB, official vLLM 0.20.0 image, Hugging Face Buckets.

## Global constraints

- Branch: `codex/nemotron3-omni-frozen30` in the current Codex-managed linked
  worktree. The ignored `data` link must continue to resolve exactly to
  `/home/npatta01/data/competitions/trec_video_2026/data`.
- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` at revision
  `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`.
- Image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`.
- CUDA/dstack only for Nemotron. Do not download the model locally and do not
  attempt ROCm. Keep `HF_TOKEN` in dstack secrets only; never print it or use
  secret-bearing `diagnose` output.
- Inference is non-thinking, greedy, and answer-only: temperature 0, top-k 1,
  at most 32 output tokens and 12 answer words. Never request, persist, grade,
  or expose chain-of-thought.
- Private media, raw rows, references, predictions, grades, and logs remain out
  of Git. Durable private artifacts live under the immutable run's HF Bucket
  prefix and must be listed, downloaded, and checksum-verified.
- Preview every dstack config with `echo "n" | dstack apply -f ...`, retain the
  complete output, use the scalar `L40S:1:48GB..` selector, and require multiple
  physically distinct compatible offers when practical. Apply detached; keep
  the workload foreground. No blind retry loops and no rerun for poor quality.
- The first technical failure permits at most one separately diagnosed `-a2`
  corrective attempt. Do not create or allowlist `-a2` unless that condition
  occurs. A valid negative result ends the track.
- The canary is capped at 30 minutes and `$2/hour` (worst case `$1`). The full
  run is capped at 60 minutes and `$2/hour` (worst case `$2`). Together with
  `$0.9972` already spent, the cumulative worst case is `$3.9972`, within the
  original `$8` allocation. Do not consume the central reserve.
- Own only Nemotron-specific adapter/config/script/test files, this plan, and
  one uniquely named sanitized evidence record. Do not edit shared
  project-status QMD or generated Markdown/HTML.

## Preregistered question and design

The treatment is Nemotron with native video and audio enabled. Its matched
diagnostic control is the same normalized native video with audio disabled.
The operational baseline is the already frozen Qwen3.5-4B 12-frame condition.
This is an end-to-end pipeline replacement test, not a same-evidence
model-only comparison: Qwen receives 12 sampled frames, whereas Nemotron sees
continuous normalized video and optional native audio.

The panel is the exact ordered, label-free 30 rows in
`configs/vqa/native_audio_benchmark_30_inference.json`, six rows in each of
dialogue/speech, non-speech sound/music, temporal/action,
spatial/OCR/color, and visual controls. Its SHA-256 is
`12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`.
The manifest's missing `QFljXc4WpFA.mkv` resolves only to the existing
`QFljXc4WpFA.mp4`; the frozen contract records that unique extension without
modifying the source manifest. All 30 inputs are normalized inside the exact
image to metadata-stripped H.264/yuv420p plus AAC, with source and derived
hashes, duration drift at most 0.5 seconds, and independent video/audio decode
checks before model load.

The reusable comparator is the original 120-row, label-free Qwen v2 freeze at
`data/vqa/benchmarks/runs/native-audio-benchmark-v2/selected-predictions.jsonl`.
Evaluation must independently require its full-file SHA-256
`8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879`,
validate the canonical condition/order/provenance, and only then extract the
first 30 Qwen12 rows. A caller-supplied boolean or mutable sidecar is not a
trust anchor.

## Predeclared decision rule

The run is scoreable only if all 60 Nemotron requests complete in exact
adjacent audio-off/audio-on order, the prediction file and manifest match the
frozen `SHA256SUMS`, and the original Qwen comparator independently passes its
pinned full-file hash and order checks. Any incomplete or mutated freeze is
**Incomplete** or **Invalid**, never a quality result.

After both freezes verify, two independent graders assign C/P/W grades from
separately randomized, condition-blind packets. A third independent grader
adjudicates only disagreements. Use ordinal ranks W=0, P=1, C=2. Nemotron
audio-on passes the semantic replacement gate only when, versus Qwen12:

- at least 6 of 30 rows improve and no more than 2 regress;
- no category has more regressions than improvements;
- none of the six visual controls regress; and
- the automatic guardrails show no loss greater than 1 exact-match row and no
  loss greater than 0.02 in either token F1 or METEOR.

The operational gate additionally requires a complete run within 60 minutes,
actual compute cost at most `$1.25`, verified private artifacts, and no
load-bearing privacy or audit gap. Passing both gates changes the disposition
to **Current default** for this frozen-30 VQA pipeline. Passing the semantic
gate but not the operational gate yields **Routed candidate**. Otherwise the
result is **Not promoted under the tested condition**. Independently report
audio-on versus audio-off by category; a sound-specific gain can retain the
existing **Routed candidate** status but cannot override a failed replacement
gate.

## Immutable identities and sequential gates

- Public exact-image canary evidence ID:
  `nemotron3-omni-frozen30-canary-20260801-a1`; scheduler alias:
  `nemotron3-f30-canary-20260801-a1`.
- Private paired run and scheduler ID:
  `nemotron3-omni-frozen30-20260801-a1`.
- Both store under
  `hf://buckets/Npatta01/trec_video_2026/experiments/<run-id>/`.

The public canary must run the exact image, actual foreground wrapper and
native audio-video request seam on three synthetic H.264/AAC videos and pass
3/3 before private upload. The private input bundle then contains exactly one
label-free contract, 30 media objects, and `INPUTS_COMPLETE.json` (32 remote
objects); upload, list, download, and checksum-verify it before the full run.

### Task 1: Add the frozen-30 contract and probe mode

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/nemotron3_omni_pilot.py`
- Modify: `tests/test_nemotron3_omni_pilot.py`

1. Add focused failing tests that name these breaks: wrong 30-row identity or
   order, label-bearing input, ambiguous/missing source media, failure to use
   the unique QFl MP4, non-adjacent pairs, wrong 60-request denominator,
   unverified normalization/decode, and reasoning-bearing output.
2. Run the focused tests and retain the expected RED failures.
3. Add an explicit `frozen30-paired` contract/CLI path. Derive the exact rows
   from the pinned label-free manifest, resolve exactly one `.mkv` or `.mp4`
   source per row, and freeze source hashes in the private contract.
4. Extend normalized validation and the real probe only through this explicit
   mode. Preserve all earlier modes and identities unchanged.
5. Run the focused tests to GREEN and refactor only duplication exposed by
   the new path.

### Task 2: Add independent dual-freeze evaluation

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/nemotron3_omni_frozen30.py`
- Create: `scripts/score-nemotron3-omni-frozen30.py`
- Create: `tests/test_nemotron3_omni_frozen30.py`
- Create: `tests/test_score_nemotron3_omni_frozen30_script.py`

1. Add failing tests for original Qwen full-file hash mutation, wrong Qwen
   order/provenance, candidate prediction/manifest/SHA256SUMS mutation,
   missing/reordered/duplicate candidate pairs, early reference access,
   label leakage, condition-name leakage in blind packets, stratum mismatch,
   and predeclared decision-rule boundaries.
2. Run the focused tests and retain the expected RED failures.
3. Implement a strict verifier that accepts paths, recomputes all hashes and
   structure itself, and loads references only after both freezes pass. Reuse
   existing scoring primitives where their contracts match; do not weaken or
   mutate protocol v2.
4. Produce two independently seeded, condition-blind grading packets and a
   disagreement-only adjudication packet. Produce EM, token F1, METEOR,
   C/P/W comparisons, category summaries, completion, and the predeclared
   disposition in private outputs.
5. Run the focused tests to GREEN.

### Task 3: Add upload, runner, collector, and dstack boundaries

**Files:**
- Modify: `scripts/upload-nemotron3-omni-pilot-inputs.sh`
- Modify: `scripts/run-nemotron3-omni-dstack.sh`
- Modify: `scripts/collect-nemotron3-omni-dstack.sh`
- Create: `dstack/nemotron3-omni-frozen30-canary-a1.dstack.yml`
- Create: `dstack/nemotron3-omni-frozen30-a1.dstack.yml`
- Modify: `tests/test_nemotron3_omni_dstack_config.py`

1. Add behavioral RED tests for exact identities, config/runtime pins, scalar
   L40S selector, on-demand provisioning, hard caps, detached-compatible
   foreground execution, exact 32-object upload bundle, contract-derived media
   selection, and rejection of every unrecognized or uncreated a2 identity.
2. Run the focused tests and retain RED failures.
3. Implement the smallest explicit frozen30 branches. Normalize and validate
   all 30 media before model load; never hardcode a second 30-ID list in Bash.
4. Ensure collection preserves checksummed workload artifacts, exact preview
   text, and sanitized scheduler/region/duration/cost metadata without
   `diagnose`, then recursively re-downloads and verifies the bundle.
5. Run the focused tests, shell syntax checks, and YAML parsing to GREEN.

### Task 4: Preregister the sanitized evidence record and verify locally

**Files:**
- Create: `docs/experiments/2026-08-01-nemotron3-omni-frozen30.md`
- Modify only model-specific tests if verification exposes a real defect.

1. Record the question, exact panel and runtime identities, modality/evidence
   distinction, integrity gates, predeclared rule, budget, limitations,
   rules-out/does-not-rule-out, reconsideration, and initially pending outcome.
2. Use the repository vocabulary exactly: Evidence validity and Decision
   disposition. Do not claim an outcome before verified evidence exists.
3. Run model-specific tests, the full suite, Ruff/format checks where
   configured, shell syntax, privacy scans, and `git diff --check`.
4. Obtain a pre-cloud review and resolve all load-bearing findings.

### Task 5: Execute the exact-image public canary

1. Capture a fresh offer listing and require multiple distinct compatible
   on-demand L40S offers where practical.
2. Run and retain the required declined preview. Verify its complete output
   and hash before detached submission.
3. Submit the a1 canary once, monitor the foreground workload without secret
   diagnostics, and require 3/3 valid native audio-video short answers.
4. Collect, list, download, and checksum-verify its workload and sanitized
   audit bundle. Stop on a valid gate failure; diagnose one technical failure
   before deciding whether the separately bounded a2 is justified.

### Task 6: Upload and execute the paired frozen-30 run

1. Build the label-free contract and upload only the exact 30 media, contract,
   and marker. Recursively list, download, and checksum-verify all 32 objects.
2. After fleet release, capture fresh offers and the full declined preview.
3. Submit the a1 run once. Require normalization/decode validation for 30/30
   media before model load and exactly 60/60 frozen outputs.
4. Collect and independently verify workload and sanitized audit artifacts,
   actual region, duration, price, and cost. Never repeat a technically valid
   run because quality is poor.

### Task 7: Freeze, grade, decide, and commit

1. Download the private output bundle and independently verify every checksum,
   manifest binding, exact pair order, run/model/image identity, and 60/60
   completion before making references accessible.
2. Validate the pinned original Qwen freeze. Only then create two randomized
   blind grading packets; obtain two independent grades and third-party
   adjudication only for disagreements.
3. Compute automatic metrics and apply the predeclared semantic and operational
   gates without post-hoc threshold changes.
4. Update only the sanitized evidence record with aggregate results, validity,
   disposition, limitations, rules-out/does-not-rule-out, audit gaps, and
   reconsideration. Keep raw rows, predictions, references, grades, and logs
   private.
5. Run fresh focused and full verification, privacy scans, `git diff --check`,
   and final independent review. Commit the complete branch and report the
   branch, commits, run names, exact revisions, offer/cost evidence,
   completion denominator, quality result, artifact URI/checksums, files,
   tests, and persistent errors.
