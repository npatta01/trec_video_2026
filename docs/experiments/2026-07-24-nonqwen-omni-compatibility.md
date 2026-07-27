# Non-Qwen omni compatibility probe

Date: 2026-07-24

Issue: #6

Decision: **promote Gemma 3n E4B-it; stop Phi-4-multimodal-instruct**

> **Superseded decision:** Compatibility alone did not establish quality. On a
> later three-row semantic screen, neither Gemma nor Phi earned promotion;
> broader comparisons were not ruled out. See the current
> [Project Status](../project-status/index.md).

## Scope and gate

The probe used the shared, label-blind three-video panel at
`data/vqa/benchmarks/omni-compatibility-3/contract.json`.

- Canonical contract SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Contract file SHA-256:
  `dfd2d765e0dc0947e81e0387e09b88a0b75221a91571f43ae9812fdd66151309`
- Frozen roles and video IDs:
  - lexical speech: `NN7G28sBOlY`
  - prosody: `NUmb2P3Bzu8`
  - non-speech sound: `NkXPPJoZ-h8`

Promotion requires exactly one result for each role, successful output parsing,
12 vision elements and one audio element in the same request and first model
forward, one joint generation call, processed-input hashes, and no
input-contract error. Semantic answer quality was not scored in this
compatibility probe.

## Results

| Model | Revision | Joint first-forward input | Parsed rows | Decision |
|---|---|---:|---:|---|
| Phi-4-multimodal-instruct | `93f923e1a7727d1c4f446756212d9d3e8fcc5d81` | 3/3 | 0/3 | **stop** |
| Gemma 3n E4B-it | `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10` | 3/3 | 3/3 | **promote** |

Phi consumed both modalities correctly on every row, but returned plain text
instead of the requested JSON object. The fail-closed output parser therefore
stopped promotion. Gemma returned parsed JSON on all three rows and satisfied
every promotion requirement.

### Per-row runtime evidence

| Model | Role | Status | Time (s) | Peak GPU bytes |
|---|---|---|---:|---:|
| Phi-4 | lexical speech | output parse failed | 53.453 | 32,685,783,040 |
| Phi-4 | prosody | output parse failed | 214.968 | 59,415,433,728 |
| Phi-4 | non-speech sound | output parse failed | 114.037 | 59,429,237,760 |
| Gemma 3n | lexical speech | passed | 245.919 | 18,128,105,984 |
| Gemma 3n | prosody | passed | 28.195 | 18,127,977,984 |
| Gemma 3n | non-speech sound | passed | 12.456 | 18,127,442,432 |

For Phi, the first-forward key set included `input_image_embeds` and
`input_audio_embeds`. For Gemma, it included `pixel_values` and
`input_features`. Each saved row contains the complete first-forward key set,
the processed vision/audio hashes, and derived element counts of 12 and 1.

## Reproducibility

Both runs used rootless Podman, ROCm BF16, no network, no published ports, and
read-only model-cache, repository, and dataset mounts. Only the run output
directory was writable.

### Phi-4

- Runtime image:
  `localhost/trec-vqa-phi4mm-rocm@sha256:17275d29326529d0f0275d0466311f380b623a62d965ba372f931cae34c6d6f5`
- Image size: 30,415,604,729 bytes
- Runtime: Python 3.12.3, PyTorch
  `2.10.0+rocm7.2.4.git3d3aa833`, ROCm `7.2.53211`, Transformers `4.48.2`
- Cached snapshot size: 12,855,250,627 bytes
- Run ID: `phi4mm-rocm-bf16-aotriton-20260724`
- Runner SHA-256:
  `569e5c22e10ddbb533c94b3368665546d5fa9c6be79909be2cf5f19c74fca295`
- Predictions SHA-256:
  `6a1ee0d5ea29d150f871f7942c6377643ad6a0d86b4df0953a339d7e6c890478`
- Summary SHA-256:
  `2081db9f03a1b10cf1fa49296604dc58a3583aa2f7507bdfafa38e83a9e73326`

Phi's pinned remote code is incompatible with Transformers 5 meta-device
initialization (`Tensor.item()` on a meta tensor), so its model-declared
Transformers 4.48.2 runtime is required. Eager attention and ROCm's default
SDPA fallback exceeded the 100 GiB device limit at this 12-frame sequence
length. SDPA with
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` completed all three joint
generations. These are verified ROCm/runtime constraints, not input-contract
failures.

### Gemma 3n

- Runtime image:
  `localhost/trec-vqa-omni-rocm@sha256:b1f97c47769a8a2f32a5ac9a18cc62d2aab57ed32c66720775407f6213ebf96f`
- Image size: 30,321,490,851 bytes
- Runtime: Python 3.12.3, PyTorch
  `2.10.0+rocm7.2.4.git3d3aa833`, ROCm `7.2.53211`, Transformers
  `5.15.0.dev0`, timm `1.0.28`
- Cached snapshot size: 15,743,651,698 bytes
- Run ID: `gemma3n-rocm-bf16-timm-20260724`
- Runner SHA-256:
  `569e5c22e10ddbb533c94b3368665546d5fa9c6be79909be2cf5f19c74fca295`
- Predictions SHA-256:
  `9797c14694bf99a438a76eba27f64fd336cae06526e11d8d2faa99021423ef00`
- Summary SHA-256:
  `4c3bbf2cf0a70105bbf3eeb7f253855283c170c72d3908ae663af8e9428a731d`

The Gemma vision tower requires `timm`; the final digest pins version 1.0.28.
With that dependency present, the model loaded and all three rows promoted.

## Commands

```bash
scripts/run-omni-compatibility.sh \
  phi4mm phi4mm-rocm-bf16-aotriton-20260724

scripts/run-omni-compatibility.sh \
  gemma3n gemma3n-rocm-bf16-timm-20260724
```

The contract and result artifacts remain under ignored shared `data/` storage;
no dataset, model weight, raw log, or prediction artifact is committed.
