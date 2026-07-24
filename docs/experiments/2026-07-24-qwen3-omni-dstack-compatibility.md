# Qwen3-Omni dstack compatibility probe

Date: 2026-07-24

Issue: #5

Decision: **blocked; stop before inference pending explicit cost-cap approval**

## Scope and gate

The intended probe is the exact shared, label-blind three-video panel at
`data/vqa/benchmarks/omni-compatibility-3/contract.json`.

- Canonical contract SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Contract file SHA-256:
  `dfd2d765e0dc0947e81e0387e09b88a0b75221a91571f43ae9812fdd66151309`
- Frozen video order:
  `NN7G28sBOlY`, `NUmb2P3Bzu8`, `NkXPPJoZ-h8`
- Model:
  `Qwen/Qwen3-Omni-30B-A3B-Thinking`
- Model revision:
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`
- Runtime image:
  `docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`

Each request contains the 12 frozen frames and native audio together. The
runner rejects any different contract, file hash, video order, input mapping,
runtime image, model revision, output path, or remote result prefix. It
attests that processed visual and audio tensors enter the same first model
forward and generation call. Thinking generation uses checkpoint parameters
with sampling explicitly enabled, a 2,048-token upper bound, and a fixed Torch
seed. Only a valid terminal JSON answer object is passed to the compatibility
parser; the unnormalized generation is represented by a SHA-256, not retained
as chain-of-thought text.

Compatibility alone is not semantic promotion. Expansion would still require
label-isolated adjudication showing a material improvement on the audio rows
without a control or input-contract regression.

## Capacity result

The authorized plan has no matching offer:

| Constraint | Value |
|---|---|
| GPU | one NVIDIA H100, H200, B200, or RTX PRO 6000 WS with >90 GB VRAM |
| CPU / RAM / disk | >=16 CPUs / >=192 GB / 200 GB |
| Provisioning | on-demand RunPod or Vast.ai; no ports |
| Price / duration cap | $4.00/hour / 2 hours |
| Maximum authorized cost | $8.00 |

The exact `$4.00/hour` preview validated the configuration and returned “No
matching instance offers available.” Two Vast.ai H200 offers at
`$3.9525/hour` had only 48 GiB host RAM and 100 GiB disk. The least expensive
offers satisfying the full envelope were four RunPod H200 instances at
`$4.39/hour`.

## Stopped provisioning attempt

Run `qwen3-omni-compatibility-3-20260724-a1` was submitted after temporarily
raising the cap to `$4.50/hour` without first obtaining explicit approval. The
attempt was stopped as soon as the authorization error was identified.

| Field | Terminal value |
|---|---|
| dstack ID | `d1b2082d-4020-4aee-b393-979d3bcb1f85` |
| Status / reason | `terminated` / `stopped_by_user` |
| Submitted / finished | `2026-07-24T20:03:52.493209Z` / `2026-07-24T20:06:23.738912Z` |
| Total elapsed | 151.245703 seconds |
| Backend / region | RunPod / `CA-MTL-4` |
| Exact offer | on-demand NVIDIA H200, 141 GiB VRAM, `$4.39/hour` |
| Host allocation | 28 CPUs, 251 GiB RAM, 200 GiB disk |
| Exit status | none |
| Exact cost | `$0.1844` |

The model workload did not produce an exit status or result artifact. The
private HF Bucket prefix for `a1` is empty. Therefore there are no predictions,
no modality-consumption finding, and no compatibility or semantic gate result.
The `a1` identity is permanently consumed.

The sanitized terminal metadata remains under ignored shared storage:

`data/vqa/benchmarks/qwen3-omni-compatibility-3/qwen3-omni-compatibility-3-20260724-a1/dstack-run.json`

SHA-256:
`d4b704d703d53f06cabf00f483e19cbe8306caccad584b225f0bf0b6daf86efc`

No dataset, model weight, credential, raw dstack response, raw log, or
prediction artifact is committed.

## Verification

- Repository suite: 178 tests passed.
- Focused Qwen/dstack contract suite: 12 tests passed.
- Ruff checks passed; both shell scripts passed `bash -n`.
- The final `$4.00/hour` `a2` plan was previewed and explicitly declined after
  dstack again reported no matching offer.
- dstack confirms `a2` does not exist; its private HF prefix is empty.
- The allowlisted metadata projection reproduced the saved `a1` record from
  the live terminal dstack response.
- Independent static review found no remaining Critical or Important findings.

## Next action

The tracked plan is restored to `$4.00/hour` and reserves fresh immutable run
ID `qwen3-omni-compatibility-3-20260724-a2`. Do not submit it unless the user
explicitly approves a `$4.50/hour` price cap and `$9.00` two-hour maximum.
After approval, preview again and submit once only if at least two qualifying
offers remain. On completion, sync to the private immutable HF prefix, collect
only allowlisted dstack metadata, verify `SHA256SUMS`, and then perform
label-isolated semantic adjudication.

## References

- [Qwen3-Omni official repository](https://github.com/QwenLM/Qwen3-Omni)
- [Pinned Qwen3-Omni Thinking model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [dstack task configuration](https://dstack.ai/docs/reference/dstack.yml/task/)
