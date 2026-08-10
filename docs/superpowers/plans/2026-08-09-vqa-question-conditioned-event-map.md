# Full-panel question-conditioned event-map experiment

Date: 2026-08-09

Status: **Approved direction; plain-text AV canary implementation in progress**

## Objective

Test a deployable multistage VQA system on every annotated video with local
media: understand the complete clip, create a compact timestamped event map,
select question-relevant evidence, answer directly, and reconstruct raw
frames plus verbatim ASR for matched downstream answers.

This supersedes automatic launch of the full reference-assisted 3x3
silver-oracle factorial. The B4 oracle run ends after its 11-row operational
canary. A later oracle is allowed only if deployable localization remains
unresolved, and then only for G/F, H/F, G/S, and H/S with the same tight span
budget.

## Verified starting point

- Relevant panel: 481 annotated rows, 475 unique videos with local media.
  The six missing-media rows stay excluded and the 1,883-row labeled test set
  stays untouched.
- Best frozen fixed answer is approximately `C=0.498`; the reference-aware
  row maximum over four frozen candidates is `C=0.551`. This proves candidate
  complementarity, not deployable selection.
- Full Whisper already improves `C` by approximately `0.025`, so blanket
  no-transcript reruns are not part of this experiment.
- Four prior 4B/8B answer arms used only 0.801 aggregate GPU-hours and
  completed in 13.6 minutes parallel. Whole-video understanding, not the 4B
  answer matrix, is the runtime bottleneck.
- The prior verified Nemotron frozen-30 experiment produced 60 matched native
  audio-off/on answers in 34.832 L40S minutes at $0.5747. Audio-on improved
  automatic metrics but failed the preregistered semantic replacement gate;
  it remains evidence for testing audio, not for assuming it wins.

## Frozen model and runtime

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Revision: `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
- Runtime image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`
- Greedy, non-thinking decoding. No chain-of-thought is requested or retained.
- Full runs use eight private L40S tasks: four shards for each of two
  conditions.

## Two-pass holistic core

Both passes receive the same question and the same complete normalized video.
The original strict-JSON event-map interface is superseded after only 2/24
valid outputs across interval and point tests. The approved replacement uses a
plain-text timeline and a separate text-only selector; no grammar decoding or
LLM repair is permitted.

1. **V-map:** native video; native audio disabled.
2. **AV-map:** native video plus native audio.

Whisper text is not supplied to either map model, preserving a clean native
audio intervention. After the map is frozen, selected timestamps are joined
deterministically to the original Whisper segments by temporal overlap.

The direct V-map/AV-map answers are useful full-video model baselines, but do
not by themselves establish that the selected intervals caused the answer.
That causal claim is reserved for the second-stage H/S answer and its matched
shifted-interval control, both reconstructed from the frozen raw media at the
selected timestamps. The complete experiment is therefore map generation
followed by selected-evidence answering, not a claim that the first call alone
implements the whole pipeline.

A question-agnostic map is intentionally excluded: there is one question per
video, so it has no reuse benefit and adds irrelevant text and timestamp
surface area.

## Superseded strict-JSON response contract

The following contract is retained only for rejected provenance and must not be
used for the next canary:

- schema/run/condition/video identity;
- a nonempty short answer, abstention boolean, and bounded confidence;
- one to six chronological, non-overlapping events covering the clip, each
  with an opaque event ID, start/end seconds, and private visual/audio summary;
- one or two selected intervals referencing event IDs, with chronological
  start/end seconds and at most six seconds total selected duration;
- first-pass/retry status, request hash, response hash, model/runtime identity,
  elapsed time, and memory telemetry.

Generated summaries and rationales remain private and are never passed to a
downstream answerer. Downstream requests are reconstructed only from media
bytes, selected timestamps, actual decoded frame PTS, and verbatim Whisper
segments.

## Canary and full launch

Run one AV-only four-video canary on the warm single L40S. The model map phase
produces four plain-text timeline outputs; after all four maps are frozen, a
text-only selector phase produces four selector outputs. The label-free panel
covers speech, non-speech sound, OCR/detail, and temporal/action questions.

The canary has a 45-minute hard cap and is expected to finish within that cap;
the 2.3--3.5-hour estimate applies only to the later eight-shard 475-video
full-panel run. A canary that reaches 45 minutes is an operational NO-GO and
does not justify increasing the cap automatically.

Canary gates for the replacement protocol:

- 4/4 map outputs and 4/4 selector outputs pass on the primary request with no
  retry;
- map decoding uses `enable_thinking=false`, temperature 0, top-p 1, top-k 1,
  max 256 tokens, and stop `\nEND`; selector decoding uses the same settings
  with max 64 tokens;
- no `response_format`, JSON schema, grammar, or LLM repair;
- exact model/image/revision, native-audio toggle, and panel identity;
- all anchors, deterministic cells, and selected-window constraints valid
  against measured duration;
- 4/4 nonempty map answers, 4/4 nonempty selector answers, and frozen
  request/response hashes;
- no reference, label, metric, or held-out-data leakage;
- frozen canary scores show no catastrophic semantic regression on METEOR,
  BERTScore, STS, and NDCG;
- projected full aggregate time at most 28 GPU-hours, longest shard at most
  3.5 hours, and projected provider cost at most $30.

The canary projection is frozen before any successful canary payload. Scale
the measured normalization duration per unique canary medium over 950 full-run
transcodes. Scale bootstrap/download, model startup, finalization, any positive
non-row inference overhead, and any positive scheduler-duration residual over
all eight independently loading tasks. Scale the condition-specific mean row
inference time over 475 V-map and 475 AV-map calls with a conservative 10%
inference margin. The longest-shard
estimate uses one task's fixed phases, 119 normalizations, and 119 calls at the
slower condition mean with the same margin; because all eight shards launch in
parallel, that estimate is also the projected wall time. Producer phase totals
must reconcile with scheduler elapsed time within the larger of 120 seconds or
10%. Sanitized scheduler cost must reconcile with elapsed hours times observed
price within the larger of $0.05 or 5%, and projected cost uses that observed
price.

If operational gates pass, launch all eight 119/119/119/118-row tasks in
parallel. Hard core limits are 960 primary calls, 960 retries only on malformed
rows, 32 L40S-hours, $32 provider cost, and 5 hours wall time after full launch.

## Evidence reconstruction and matched 4B tests

Use the frozen AV-map selections for downstream evidence. Preserve V-map as
the clean native-audio control. Build exactly these full-panel conditions:

- **F0:** reuse frozen G/F: 12 global frames plus full Whisper.
- **H/F:** six global plus six map-local frames, full Whisper.
- **G/S:** global frames, map-selected verbatim Whisper.
- **H/S:** hybrid frames, selected Whisper.
- **Shifted H/S:** the same frame/transcript quantity and interval lengths,
  deterministically shifted away from selected events.

Every materialized frame records requested timestamp, actual decoded PTS,
stream/frame identity, decoder/runtime version, byte hash, and contact sheet.
Every transcript selection records original segment IDs, source-line bytes,
and hashes. No manual full-panel grading is required.

## Metrics and decisions

Score direct V-map and AV-map answers plus all 4B conditions with exact match
(diagnostic), token F1, METEOR, BERTScore F1, STS, and
`C=(METEOR+BERTScore_F1+STS)/3`. Use 10,000 paired media-level bootstrap
replicates and report the official-style metrics separately.

Primary gates:

- Main promotion: `delta C >= 0.015`, lower 95% bound above zero, and no
  component regression worse than `0.002`.
- Native audio: AV-map minus V-map; target sound/speech stratum
  `delta C >= 0.030`, visual/temporal controls no worse than `-0.005`.
- Visual localization: H/F minus F0 and H/S minus Shifted H/S; the first must
  be positive and timestamp specificity must be at least `0.010 C` with lower
  bound above zero.
- Transcript selection: G/S minus F0.
- Combined deployable evidence: H/S minus F0 at least `0.015 C`, lower bound
  above zero.

Run a question-only high-resolution/OCR policy on routed rows plus 30 seeded
controls after the core result. Run a matched stronger answerer on identical
H/S evidence only if H/S beats Shifted H/S but does not materially beat F0;
`delta C >= 0.015` with a positive lower bound identifies the 4B answerer as a
bottleneck.

Reference-blind election is not on the core critical path. Pool A may run on
the four already frozen candidates after the evidence policy is frozen. Pools
B/C remain stopped unless Pool A passes its own preregistered gate.

## Artifact boundary

Private only:

- label-free requests, media, raw model outputs, event maps, selected
  intervals, transcripts, frames, answers, per-row metrics, logs, and freeze
  locators;
- immutable Hugging Face Bucket payloads and local ignored mirrors.

Tracked:

- protocol/model/runtime/panel/source hashes;
- aggregate metrics, paired intervals, strata counts, coverage, timing/cost,
  gate booleans, and private artifact URIs/checksums.

## Current status and next action

- [x] Advisor review selected the two-pass question-conditioned design.
- [x] Full reference-assisted 3x3 launch stopped before submission.
- [x] Stop the 11-row B4 operational canary after it exhausted the 45-minute
  active-runtime cap without a terminal payload. The exact dstack run
  `39f2dbfb-c058-4934-839d-e765beae7a91` failed after costing `$2.5953`;
  this is an operational NO-GO for the Qwen selector, not a quality result.
- [x] Reuse and harden the verified Nemotron runner with a strict event-map
  schema, shard-only normalization, durable failure artifacts, and
  terminal-marker-last publication.
- [x] Pin the 475-row label-free controller, source, model, runtime, paired
  canary, and eight full execution attempts. The incomplete metadata-only
  private `a1` transport prefix and the pre-final-retry-fix `a2` bundle are
  superseded; only the verified `a3` bundle may be consumed.
- [x] Preserve the first paired execution attempt as a pre-inference
  operational failure. Run `vqa-event-map-canary` / attempt
  `4baaf48c-b690-45f0-b6bf-065b9dc1e527` failed at the final preflight write
  after `$0.3266`; all preceding identity and remote media-hash checks passed.
  The root cause was a string-valued `preflight_path`, now covered by a
  behavioral regression test and converted to `Path` before writing. No
  output bucket was reserved and no model call ran.
- [x] Preserve the second paired execution attempt as another pre-inference
  operational NO-GO. Run `vqa-event-map-canary-a2` / attempt
  `d6d3a33e-84c5-4067-bcea-fe82cdd10e2f` failed during media normalization and
  produced no payload. The root cause was the missing `-1` value after
  `-map_metadata` in the deterministic ffmpeg command; the corrected command
  is covered by a behavioral regression test. No A2 evidence may satisfy a
  launch gate.
- [x] Switch from the strict-JSON interface to the approved plain-text AV
  timeline plus text-only selector after the interval/point canaries achieved
  only 2/24 valid outputs; thinking was slower and did not improve validity.
- [ ] Run the newly pinned plain-text AV canary on the persistent dstack dev
  environment. Launch a 32-row fail-fast full-panel checkpoint only after
  technical and semantic canary GO; require at least 31/32 complete map and
  selector pipelines before continuing.
- [ ] Freeze V-map/AV-map outputs before loading references.
- [ ] Materialize H/F, G/S, H/S, and Shifted H/S evidence and answers.
- [ ] Score, record one primary experiment result, and choose the next system
  architecture from the frozen gates.

The immediate critical path is therefore the Nemotron event-map runner and
paired four-row canary. No full-panel job is submitted before that canary
passes. On a pass, the expected full-panel wall time is 2.3--3.5 hours because
the eight V-map/AV-map shards run concurrently.
