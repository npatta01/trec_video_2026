# Qwen Empty-Output Mechanism Diagnostic Design

Date: 2026-08-16

## Objective

Determine, without opening labels or scoring transcript quality, whether the
two empty outputs in the frozen eight-row Qwen3-ASR plus forced-aligner run are
caused by automatic language selection, timestamp generation, loading the
forced aligner, or an unstable baseline. Repair the scoring boundary before
any new experiment so a frozen `not_ready` decision cannot reach label loading
or post-freeze scoring.

## Existing evidence and root cause

The immutable Stage 0.5d run completed one local batch with container exit code
zero. Six of eight rows produced valid aligned output; two produced
`ValueError: Qwen3-ASR returned empty text`. Its frozen decision is
`not_ready`, and no answer or Whisper comparison was created.

The scoring defect is separate from the model failure. Both score entry points
verify that a prediction freeze exists, but neither requires that the freeze's
decision is `ready_for_quality_test`. Because `status_canary()` reports the
phase as `predictions_frozen` for both ready and not-ready freezes, a caller can
load private labels and the core scorer can create a survivor-only score from
an incomplete run. The repair must reject the run before label files are read
and again inside the core scorer.

## Diagnostic question

On the two Stage 0.5d rows that returned empty text, which single Qwen inference
surface restores a nonempty result while preserving output on two passing
controls under the same model revisions, audio bytes, runtime image, precision,
and offline environment?

This is an outcome-selected mechanism diagnostic. It cannot estimate failure
prevalence, transcript quality, Whisper replacement value, or downstream VQA
quality.

## Panel and data boundary

Use a fresh private root:

`data/vqa/benchmarks/model-verifier-canaries-20260814/stage05e-qwen-empty-output-diagnostic-v1/`

The panel contains exactly four WAVs selected only from frozen Stage 0.5d
terminal results and request order:

- both rows whose preserved result is `failed`, whose error type is
  `ValueError`, and whose error message is exactly
  `Qwen3-ASR returned empty text`;
- the first two rows in frozen request order whose preserved result is
  `passed` with nonempty text and valid alignment.

The selector may read identity, status, error type/message, request order,
audio path, duration, and audio hashes. It must not copy or inspect transcript
text, questions, correct answers, annotations, Whisper text, metrics, or grades.
Row identities and all row-level results remain private under the ignored data
root.

## Frozen arms

Run each arm once, in this immutable order, in a fresh rootless Podman process:

| Arm | Aligner loaded | Timestamps requested | Language | Purpose |
|---|---:|---:|---|---|
| A `aligned_auto` | yes | yes | automatic | Reproduction control for the failed condition |
| B `base_auto` | no | no | automatic | Tests the base ASR path without aligner state |
| C `aligner_plain_auto` | yes | no | automatic | Isolates timestamp invocation while retaining aligner load |
| D `aligned_english` | yes | yes | `English` | Tests automatic-language selection while retaining alignment |

Held fixed:

- ASR revision `7278e1e70fe206f11671096ffdd38061171dd6e5`;
- forced-aligner revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`;
- runtime image
  `localhost/trec-vqa-audio-rocm@sha256:faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234`;
- ROCm `cuda:0`, BF16, SDPA, batch size one, `max_new_tokens=256`;
- byte-identical WAV inputs and order;
- read-only mounts, `--network none`, no credentials, and `$0.00` provider
  spend.

Each arm receives only run identity, condition identity, video identity,
absolute WAV path, and `mode=transcription`. The immutable container command
binds the arm-wide language and timestamp switches. No label-bearing field
enters inference.

## Result validity

Every arm must preserve four ordered results and source-audio hashes. A valid
plain-ASR success requires `status=passed`, nonempty text, and nonempty language.
A valid aligned success additionally requires at least one nonempty,
monotonic, nonnegative chunk whose end is within the frozen audio duration.
Nonzero container exits, launch errors, result-count/order mismatches, hash
mismatches, malformed output, or runner-source mutation make that arm invalid.

The harness writes an exclusive command, raw stdout, sanitized stderr,
terminal record, deterministic validation, and cryptographic freeze for every
arm. It never overwrites an attempt and never retries a failed arm.

## Predeclared decision rule

The baseline is reproducible only if arm A again returns the exact empty-text
failure on both prior-empty rows and valid aligned output on both controls.

- If arm A is not reproducible, the decision is `unstable_baseline`; no
  mechanism is attributed and no quality panel advances.
- If arm A is reproducible and arm D produces valid aligned success on all four
  rows, the decision is `ready_for_fresh_aligned_quality_panel` and D is the
  only advancing treatment.
- Arms B and C are explanatory only. Their success can localize aligner-load or
  timestamp-path behavior, but cannot advance the original aligned pipeline.
- Otherwise the decision is `not_ready`.

A later quality panel must use rows never exposed to Qwen or this diagnostic,
must predeclare its comparator and missingness rule, and must score only after
an all-row ready freeze.

## Implementation boundaries

1. Add one shared core guard that verifies the prediction freeze and requires
   `decision == "ready_for_quality_test"`; call it before CLI label loading and
   from `score_postfreeze()`. Verification must also require the top-level
   frozen decision to equal the status inside the already verified
   `validated-results.json` artifact.
2. Keep `vqa_audio_runner.py` byte-for-byte unchanged so the Stage 0.5d runner
   binding remains verifiable. Add a dedicated Qwen-only diagnostic runner
   with arm-wide language and timestamp command switches, reusing the existing
   load, normalization, quality, and metric helpers.
3. Add a dedicated diagnostic harness and CLI rather than weakening the
   quality canary's one-condition contract.
4. Test selection, input freeze, command construction, result validation,
   one-attempt preservation, decision logic, mutation detection, and status
   using synthetic files and fake subprocess results before real inference.

## Gemini 3.7 Flash boundary

Gemini 3.7 Flash (`gemini-3.7-flash`) is a stable native-multimodal model
released on 2026-08-13 with text, image, video, audio, and PDF input and text
output. It is not an arm in this diagnostic because it cannot identify the
cause of a local Qwen empty-output path. The valid follow-up is a separate,
small native-AV compatibility smoke that freezes API surface, media delivery,
structured output, timestamp behavior, media resolution, custom-FPS behavior,
cost, and a comparison boundary against the prior Gemini 3.1 Pro condition.
No Gemini call is authorized by this design.
