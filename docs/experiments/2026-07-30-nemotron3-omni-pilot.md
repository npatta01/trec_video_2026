# Nemotron 3 Nano Omni native audio-video pilot

Date: 2026-07-30

Evidence validity: **Incomplete**

Decision disposition: **Not promoted under the tested condition**

## Scope and predeclared gate

This track tested NVIDIA Nemotron 3 Nano Omni with native video plus optional
audio on a six-row paired panel. The private panel contained four non-speech
sound/music targets and two visual controls. Inference used adjacent audio-off
and audio-on requests, yielding a denominator of 12. Promotion required all 12
requests to complete, audio-on to improve at least two targets, and neither
control to regress.

The model was served non-thinking with greedy, short-answer decoding. No
chain-of-thought was requested or stored. Reference answers were withheld
during inference and were not revealed after the incomplete result, so no
C/P/W grading or quality comparison was performed.

## Exact runtime identity

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Model revision:
  `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
- Runtime image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`
- Verified runtime: vLLM `0.20.0`, PyTorch `2.11.0+cu130`, CUDA `13.0`,
  NVIDIA L40S 48 GB

## Infrastructure and compatibility gate

The first canary preview, `nemotron3-omni-runner-canary-20260730-a1`, found
no schedulable instance because the shared one-node fleet was occupied. Its
locally retained preview SHA-256 is
`4ebfb73f074757b113e230e6fe10fcc68311b2081dd5c0e9b1169d264ab9cf2c`;
no GPU run was submitted and it incurred no cost.

After that fleet node was released, the single authorized corrective preview
for `nemotron3-omni-runner-canary-20260730-a2` showed three physically
distinct RunPod L40S offers in EU-NL-1, US-MO-1, and US-TX-4, each at
`$0.99/hour`. The locally retained preview SHA-256 is
`399ef80fce9fb63ec8f869804c8cf75f0f2c53ae64480b671118e5c8d64a4d96`.

The a2 canary ran in EU-NL-1 at `$0.99/hour`, completed 3/3 public synthetic
native audio-video requests, exited successfully, and cost `$0.1213` over
`441.055116` seconds. This passed the compatibility gate.

## Focused pilot result

The `nemotron3-omni-pilot-20260730-a1` preview again exposed the same three
distinct compatible regions at `$0.99/hour`. Its locally retained preview
SHA-256 is
`9259c1f6f9a513c58f96e29d8bcf9733eb7dba204c2b6a4d8859215d42452219`.

The pilot ran on an on-demand RunPod L40S in US-MO-1. Input preflight verified
the exact label-free 12-request contract and the six private media hashes
before model loading. The run then completed only 6/12 requests, exited with
status `failed`, and cost `$0.1222` over `444.544765` seconds.

Three source media pairs decoded successfully. For the other three pairs,
vLLM's OpenCV path returned empty sampled-frame arrays after container codec
errors. `NanoNemotronVLProcessor` then raised an index error and the API
returned HTTP 400 for both paired conditions. This is an incomplete technical
result, not a negative quality result.

The track's one corrective attempt had already been consumed by the a2
canary. A pilot a2 was therefore not submitted. The final completion
denominator is **6/12**, target improvements and control regressions are
**not graded**. Its evidence validity is **Incomplete**, and its decision
disposition is **Not promoted under the tested condition**.

## Cost and durable evidence

Actual GPU spend was `$0.1213 + $0.1222 = $0.2435`, within the `$8` base
allocation. No central reserve was used.

- Canary URI:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-runner-canary-20260730-a2/outputs/`
- Canary remote `SHA256SUMS` file SHA-256:
  `0990e5e0f017b75ad9efac6e4705e1d6b17381ca5eca244fdb489b5db716a179`
- Canary frozen predictions SHA-256:
  `c52c4c4e60a60e0482b1708052403893f7a766d34d979ad0dd96932a0de5dc31`
- Pilot URI:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-a1/outputs/`
- Pilot remote `SHA256SUMS` file SHA-256:
  `41dfff5ff5ddd6b763003357006e4359dc21450c791d24e05141643e113d29e8`
- Pilot frozen partial predictions SHA-256:
  `4bf3330aa2e5b4f24ae15130a4af376a681ccdb22aaef8ce4356aec2183c7340`

Both remote prefixes were recursively listed, downloaded, and verified against
their remote `SHA256SUMS`. Private media, model answers, reference answers, and
raw logs remain only in ignored storage and the private bucket; none are
committed here.

## Evidence validity, limitations, and decision

- **Evidence validity:** **Incomplete**. The compatibility gate completed, but
  the declared paired pilot produced only 6/12 required outputs. The partial
  outputs cannot support the planned audio-on quality comparison.
- **Decision disposition:** **Not promoted under the tested condition**. The
  incomplete run is stopped and cannot be graded or promoted.
- **Confirmed defect:** three source media pairs produced empty sampled-frame
  arrays in the tested vLLM/OpenCV codec path, causing processor failures for
  both paired conditions.
- **Audit gap:** the linked private HF prefixes preserve and checksum the
  workload contracts, manifests, predictions, summaries, and server logs, but
  they do not preserve the preview text or the collector-added sanitized
  `dstack-run.json`. The exact offer regions, selected region, elapsed duration,
  and actual cost above were observed and checksum-verified in ignored local
  records, but cannot be reconstructed from the durably linked evidence bundle
  alone.
- **Limitations:** the public canary used synthetic H.264/AAC media and did not
  represent every codec found in the private MKV panel. The panel contains only
  six selected rows, half of its pairs failed before a valid answer, and no
  reference-backed quality judgment was made.
- **What this rules out now:** promoting or quality-grading this exact
  Nemotron/vLLM/container/input-path run, or using its partial answers to claim
  an audio benefit.
- **What it does not rule out:** the model family on a codec-compatible,
  preflighted media path; a separately authorized fresh experiment with a new
  immutable identity; or native-audio value under a complete blinded panel.
- **Reconsider when:** a separately authorized experiment first passes an
  exact-media codec sentinel, durably links sanitized preview and scheduler/cost
  metadata, completes all 12 paired outputs, and cryptographically verifies the
  frozen manifest and prediction file against `SHA256SUMS`, including the
  manifest-recorded prediction hash, before loading references.
