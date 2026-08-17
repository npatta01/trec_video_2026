# Qwen Aligned-English Fresh-12 Quality Design

Date: 2026-08-16

Status: approved by the user before implementation or row selection

Primary theme: Speech and Transcript Evidence

## Decision question

On a fresh, question-indicated speech panel, does the pinned Qwen3-ASR plus
forced-aligner path complete reliably when `language="English"` is fixed, and
does its transcript contain the task's official answer phrase more often than
the already frozen Whisper comparator?

This is the smallest follow-up to the incomplete Stage 0.5d quality attempt
and the Stage 0.5e mechanism diagnostic. It is not an ASR benchmark, WER test,
timestamp-accuracy test, or model promotion benchmark.

## Hypothesis and falsifier

- **Hypothesis:** forced English removes the previously observed automatic-
  language empty-output mechanism on unseen Qwen3-ASR rows, producing `12/12`
  valid aligned results and a directional answer-phrase containment advantage
  over frozen Whisper.
- **Operational falsifier:** any row fails, is missing, is out of order, has a
  source-hash mismatch, lacks a nonempty transcript, or lacks valid bounded
  alignment. The run freezes as `not_ready` and no label, answer, Whisper text,
  or score input may be opened.
- **Quality falsifier:** after a `12/12` ready freeze, Qwen answer-phrase
  containment is tied with or below Whisper. Preserve the score and do not
  advance Qwen to a larger confirmatory or downstream test.

## Freshness boundary

`Fresh` means never previously submitted to **Qwen3-ASR**, not never submitted
to any Qwen-family visual-language model. The selector must exclude:

1. every video identity found in any preserved private JSON/JSONL artifact
   containing one of the exact Qwen3-ASR markers
   `Qwen/Qwen3-ASR-1.7B`, `qwen3_asr`, or `qwen3-asr`; and
2. every row in the complete `oracle-speech-10` manifest, even if another
   preserved artifact does not expose its identity.

Before selection, freeze a private exposure index containing only source
paths, source SHA-256 values, the exact markers, and excluded video identities.
It must not contain questions, answers, transcripts, labels, grades, or model
outputs. The selector consumes that index and binds its SHA-256.

### Pre-inference freshness-proxy erratum (2026-08-16)

Actual submission of a row's audio to Qwen3-ASR remains the authoritative
freshness event. Exact-marker scanning is a conservative discovery proxy, and
every discovered marker-bearing JSON/JSONL file is exposure-bearing by
default. During pre-inference review, the controller created but did not
execute two preparation bundles, preserved at:

- `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-selection-v1-pre-review-455e7ac/`; and
- `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1-pre-review-455e7ac/`.

These exact roots are the sole `preparation_only_never_submitted`
disposition. They remain visible in the complete marker census. A private
ledger must bind both root paths, every file path and SHA-256, the prepared
identity set and fingerprint, the controller revision, and an explicit
operator attestation that neither the controller record transition nor an
out-of-band Qwen3-ASR invocation occurred. The disposition is valid only when
the inventories contain input/preparation artifacts and no command, raw
result, stderr, terminal, validation, prediction-freeze, or score artifact.

The exposure index must prove the disjoint and exhaustive classification:
every marker match is either a genuine exposure source or belongs to the
exact preparation-only roots. Any changed inventory, unexpected file,
attempt/output artifact, symlink escape, unclassified marker match, or
prepared identity also found in a genuine exposure source invalidates the
disposition and stops the experiment. The canonical selector must replay to
the same private ordered identity/stratum/source-index/audio-hash/duration
fingerprint as the preserved pre-review selection; mismatch stops without
inspection, replacement, or reselection. A complete marker-census delta guard
runs again immediately before the sole record transition, treating only the
current canonical index/selection/input-freeze files as active preparation.

This erratum changes only the proxy treatment of self-generated,
never-submitted preparation metadata. It does not change quotas, panel order,
the model condition, scoring, attempts, or decision rules.

## Label-blind panel selection

The sole source pool is the ordered 475-row manifest at:

`data/vqa/benchmarks/vqa-factorial-full-evidence-20260809-f1/payload/evidence/manifest.json`

That manifest contains only `video_id` and `question`. Selection must not open
annotations, answers, Whisper results/status, Qwen results, model maps, grades,
or score files. It may open the frozen exposure index and WAV metadata.

Classify eligible questions in this fixed priority order, using case-folded
whole-word regular expressions:

1. `reported_speech` (quota 6): forms of `say`, `tell`, `ask`, `answer`,
   `reply`, `respond`, `announce`, or `mention`;
2. `conversation` (quota 3): forms of `speak` or `talk`;
3. `vocalization` (quota 3): forms of `shout`, `yell`, `scream`, `sing`, or
   `song`.

Exclude a question from all strata when it contains a whole-word visible-text
marker: `sign`, `caption`, `subtitle`, `screen`, `label`, `shirt`, `board`,
`paper`, `written`, `writing`, or `text`. Assign overlaps to the first matching
stratum, take the first quota rows within each stratum in manifest order, then
restore global manifest order for inference. Do not filter on Whisper
acceptance, media outcome, transcript contents, reference-answer contents, or
prior Qwen3-ASR success/failure.

The selected private artifact contains only role/stratum, identity, absolute
WAV path, audio SHA-256, duration, source-order index, and selection
provenance. Questions do not enter inference requests.

## Frozen treatment and runtime

- Experiment ID: `qwen-aligner-stage05f-20260816-aligned-english-12-v1`
- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/`
- Panel size: exactly 12, ordered once.
- ASR: `Qwen/Qwen3-ASR-1.7B` at
  `7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B` at
  `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- Decode surface: forced aligner loaded, timestamps on,
  `language="English"` on every row.
- Image:
  `localhost/trec-vqa-audio-rocm@sha256:faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234`.
- Runtime: rootless Podman, ROCm `cuda:0`, BF16, SDPA, batch size 1,
  `--network none`, read-only repository/data/cache mounts.
- Cost: local only, `$0.00` paid compute.
- Attempts: one batch process, no retry, no row replacement, no relabeling.

The input freeze and every command/status check bind the dedicated English-
capable runner, its imported audio helper, the host controller, model
revisions, image, decode controls, media hashes, durations, and row order.

## Integrity and scoring boundary

Inference requests contain only run/condition/video identity, absolute WAV
path, and `mode="transcription"`. The run must freeze terminal output and
deterministic validation before any scoring source is opened.

Only if the verified prediction decision is `ready_for_quality_test` with
`12/12` completed and `12/12` valid aligned rows may scoring open:

- `data/vqa/2025/training/annotations/<video_id>.json`; and
- the already frozen full-panel Whisper artifact
  `data/vqa/benchmarks/vqa-factorial-full-evidence-20260809-f1/payload/evidence/whisper.json`.

Normalize answer and transcript strings with case-folded alphanumeric tokens.
The task-specific primary metric is exact contiguous containment of the full
normalized official answer phrase. A missing, rejected, null, or empty Whisper
transcript counts as a Whisper miss. Report aggregate Qwen count, Whisper
count, both-hit, Qwen-only, Whisper-only, and neither-hit counts. Row-level
answers, transcripts, identities, and scores remain private.

## Predeclared decision rule

1. If the one attempt is not `12/12` ready, freeze `not_ready`, do not score,
   and stop.
2. If ready and Qwen containment is strictly greater than Whisper containment,
   record `directional_qwen_signal` and permit only a larger confirmatory or
   matched downstream VQA test.
3. If ready and Qwen is tied with Whisper, record `no_directional_difference`.
4. If ready and Qwen is below Whisper, record `whisper_directional_signal`.

No outcome promotes Qwen as a Whisper replacement or supports WER, transcript
fidelity, alignment accuracy, global English robustness, or downstream VQA
improvement.

## Gemini boundary

Gemini 3.7 Flash is a separate native-audio/video system question. It must not
enter this panel, selector, inference process, missingness rule, or decision.
Any Gemini call requires its own frozen same-media compatibility design,
credential/cost preflight, and evidence root.
