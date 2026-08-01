# Nemotron 3 Nano Omni frozen-30 baseline-replacement preregistration

Date: 2026-08-01

Evidence validity: **Unknown**

Decision disposition: **Open hypothesis**

## Pending evidence record

This is a sanitized preregistration, not an experiment result. No cloud
preview, provider offer, model download, private-panel upload, workload,
prediction, reference lookup, grade, or score has been performed for either
identity below. Accordingly, the completion denominator, outcome, actual
region, duration, cost, artifact checksums, and gate outcomes are **not
recorded**.

- **Primary theme:** model comparison and specialist candidates.
- **Canonical claim page and anchor:** not yet assigned; this pending record
  creates no claim until an admissible completed evidence bundle exists.
- **Question:** on the exact frozen-30 panel, does native audio-video
  Nemotron meet the preregistered semantic and operational gates to replace
  the current Qwen3.5-4B 12-frame VQA baseline?
- **Shared evidence bundle / immutable evidence link:** pending private
  workload and sanitized audit bundles under
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/<run-id>/`.
  No object at either prospective prefix has been inspected or created by this
  record.
- **Supersedes / superseded by:** neither. The current Qwen baseline remains
  unchanged pending valid evidence.

## Frozen identities, panel, and modality boundary

The only preregistered a1 identities are:

- public exact-image canary evidence ID:
  `nemotron3-omni-frozen30-canary-20260801-a1`; scheduler alias:
  `nemotron3-f30-canary-20260801-a1`;
- private paired run and scheduler ID:
  `nemotron3-omni-frozen30-20260801-a1`.

The candidate runtime is
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` at revision
`dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`, using
`docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`
on one on-demand CUDA L40S 48 GB worker. Requests are non-thinking, greedy,
answer-only (temperature 0, top-k 1, at most 32 output tokens and 12 answer
words). Chain-of-thought is neither requested nor retained.

The label-free panel is the exact ordered 30 rows in
`configs/vqa/native_audio_benchmark_30_inference.json`, SHA-256
`12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`:
six rows each of dialogue/speech, non-speech sound/music, temporal/action,
spatial/OCR/color, and visual controls. The existing source manifest's one
unique MP4 extension is bound by the private contract rather than changing the
source manifest. The Qwen control is the first 30 Qwen12 rows extracted only
after independent validation of the full 120-row label-free v2 freeze,
`native-audio-benchmark-v2`, whose full-file SHA-256 is
`8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879`.

This is an end-to-end pipeline replacement comparison, **not** a same-evidence
model-only comparison. The operational control receives 12 sampled frames.
Nemotron audio-on receives continuous normalized video plus native audio; its
matched diagnostic control receives the same normalized video bytes with audio
disabled. Audio-on versus audio-off is reported by category, but a
sound-specific difference cannot override the replacement gate.

## Integrity gates and evidence boundary

1. The public canary must use the exact image, foreground wrapper, and native
   audio-video request seam on three synthetic H.264/AAC videos and complete
   3/3 valid short answers before any private upload.
2. The private input bundle must contain exactly one label-free contract, 30
   contract-derived media objects, and `INPUTS_COMPLETE.json` (32 objects).
   It is uploaded, recursively listed, downloaded, and SHA-256 verified before
   the paired run.
3. Before model load, each private source is hash-bound and normalized inside
   the exact image to metadata-stripped H.264/yuv420p plus AAC. The private
   contract records source and derived hashes; duration drift is at most 0.5
   seconds, and independent video and audio decode checks must pass.
4. The candidate freeze is scoreable only with exactly 60/60 adjacent
   audio-off/audio-on predictions, a manifest and prediction file bound by
   `SHA256SUMS`, and the expected model, revision, image, contract, and
   normalized-media identities. An incomplete or mutated freeze is
   **Incomplete** or **Invalid**, never a quality result.
5. The original Qwen comparator must independently pass its pinned complete
   file hash, canonical order, and provenance checks before its first 30 rows
   are used. References remain inaccessible until both freezes verify.
6. Two independently randomized, condition-blind C/P/W grade packets are
   required; a distinct third grader sees only disagreements. Raw predictions,
   references, grades, private media, and raw logs remain outside Git and this
   sanitized record.
7. The collector must preserve the declined preview and allowlisted sanitized
   scheduler/region/duration/price/cost metadata, then recursively download and
   checksum-verify the private workload and audit bundles. Secret-bearing
   diagnostics are prohibited.

## Predeclared decision rule

After both freezes verify, graders use ordinal ranks W=0, P=1, and C=2.
Relative to Qwen12, Nemotron audio-on passes the semantic replacement gate
only if all of the following hold:

- at least 6 of 30 rows improve and no more than 2 regress;
- no category has more regressions than improvements;
- none of the six visual controls regress; and
- automatic guardrails lose no more than 1 exact-match row and no more than
  0.02 in either token F1 or METEOR.

The operational gate additionally requires a 60/60 completed run within 60
minutes, actual compute cost at most `$1.25`, verified private artifacts, and
no load-bearing privacy or audit gap. The rule was predeclared before cloud
execution:

- semantic and operational gates pass: **Current default** for this
  frozen-30 VQA pipeline;
- semantic gate passes but operational gate does not: **Routed candidate**;
- otherwise: **Not promoted under the tested condition**.

The currently recorded gate outcome is **not recorded (not executed)**. This
does not alter the current **Open hypothesis** disposition.

## Budget and execution limits

The canary is capped at 30 minutes and `$2/hour` (maximum `$1`); the paired
run is capped at 60 minutes and `$2/hour` (maximum `$2`). With `$0.9972`
already spent, the cumulative worst case is `$3.9972`, within the original
`$8` allocation; the central reserve is not authorized. The actual-cost
criterion is stricter than the run price cap: it is at most `$1.25` for
operational promotion.

The a1 identities may be submitted once through the sequential gates. A first
technical failure permits at most one separately diagnosed and explicitly
authorized `-a2` corrective attempt; no `-a2` identity is pre-allowlisted.
A technically valid negative quality result ends this track and must not cause
a rerun.

## Limitations, audit gap, and interpretation boundary

- **Limitations:** the result, if completed, is condition-bound to this exact
  30-row panel, runtime, normalization path, prompt/decoding contract,
  hardware envelope, budget, and Qwen v2 control. It compares end-to-end
  pipelines with different evidence modalities, so it cannot isolate a
  model-family effect from continuous-video or native-audio evidence.
- **What this rules out now:** treating an absent result, partial completion,
  unverified artifacts, a canary, or an audio-only diagnostic as evidence for
  baseline replacement; changing the baseline before the declared gates are
  met.
- **What this does not rule out:** Nemotron compatibility, quality under this
  or another condition, a sound-specific routed use, a different controlled
  model/evidence comparison, or a future separately authorized technical
  correction after demonstrated failure.
- **Audit gap:** no a1 preview text, offer evidence, scheduler selection,
  actual runtime metadata, private artifact bundle, checksum, completion,
  prediction, reference access, grade, or score exists yet. This is absence of
  execution evidence, not a demonstrated defect.
- **Reconsider when:** the sequential canary and paired run produce an
  independently verified evidence bundle and blinded grades, then the
  preregistered rule is applied without threshold changes. A new identity,
  runtime, panel, or corrective attempt requires separate explicit
  authorization and a new or amended preregistration.

<details>
<summary>Sanitized method and future evidence disclosure</summary>

The private run will preserve the frozen contract, source/derived media hashes,
normalized artifacts, candidate predictions, manifest, `SHA256SUMS`, complete
declined preview text, and a sanitized audit projection. The audit projection
may state scheduler identity, provider/region, elapsed duration, hourly price,
and cost, but not private media, raw questions, answers, references,
item-level grades, raw logs, credentials, or chain-of-thought. The evidence
record may be updated only after independent checksum, structure, provenance,
and label-access gates have passed.

</details>
