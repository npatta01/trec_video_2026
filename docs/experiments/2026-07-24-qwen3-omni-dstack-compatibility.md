# Qwen3-Omni dstack compatibility probe

Date: 2026-07-24

Issue: #5

Decision: **runtime-compatible; retain for a broader, question-shaped quality
benchmark, but do not claim semantic promotion from this three-row panel**

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

The final authorized plan used:

| Constraint | Value |
|---|---|
| GPU | one NVIDIA H100, H200, B200, or RTX PRO 6000 WS with >90 GB VRAM |
| CPU / RAM / disk | >=16 CPUs / >=192 GB / 200 GB |
| Provisioning | on-demand RunPod or Vast.ai; no ports |
| Price / duration cap | $4.50/hour / 2 hours |
| Maximum authorized cost | $9.00 |

The original `$4.00/hour` preview validated the configuration and returned “No
matching instance offers available.” Two Vast.ai H200 offers at
`$3.9525/hour` had only 48 GiB host RAM and 100 GiB disk. The least expensive
offers satisfying the full envelope were four RunPod H200 instances at
`$4.39/hour`. After explicit authorization of the higher envelope, the
declined `$4.50/hour` preview showed those four qualifying offers and immutable
run `a2` was submitted exactly once.

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

## Completed `a2` result

Run `qwen3-omni-compatibility-3-20260724-a2` completed its wrapper and private
artifact sync, but all three inference rows failed in `generation` before
joint first-forward attestation could complete:

`Input type (float) and bias type (c10::BFloat16) should be the same`

| Role | Video ID | Status | Generation time | Peak GPU allocated | Peak GPU reserved |
|---|---|---|---:|---:|---:|
| lexical speech | `NN7G28sBOlY` | failed | 9.978 s | 63,759,737,344 B | 63,883,444,224 B |
| prosody | `NUmb2P3Bzu8` | failed | 2.936 s | 64,471,490,560 B | 64,588,087,296 B |
| non-speech sound | `NkXPPJoZ-h8` | failed | 1.556 s | 64,864,802,304 B | 64,990,740,480 B |

The exact contract, all uploaded evidence hashes, BF16 capability, model
revision, image digest, and >80 GiB GPU-memory gate passed preflight. The
executed runner then moved processor inputs to the GPU without applying the
model dtype. That left floating processor tensors in FP32 while the model used
BF16. The official pinned Qwen usage applies
`inputs.to(model.device).to(model.dtype)`; Hugging Face `BatchFeature.to`
casts floating tensors while preserving integer token and index tensors.

The verified result is therefore a **probe input-dtype failure**, not evidence
that Qwen3-Omni is incompatible with the pinned runtime. The automatic
run-level compatibility decision remains **stop** for all three roles because
no row passed. Because no answer was generated, label-isolated semantic
adjudication was not possible and the semantic promotion gate remains **not
evaluated**. No expansion is permitted.

The runner was corrected after artifact collection to follow the official
device-plus-dtype transfer. That correction was not rerun and is not validated
by `a2`; the immutable `a2` summary records the executed runner SHA-256
`f602367ccc6859118ea9f9a57bfa6fd8d5c138d03a322dc2863cac3179141d27`.
The corrected, unexecuted runner SHA-256 is
`5ca8b1a1e81c6bd662890d2dc9160d1d7426d2967ccf132d4726ce1bfae0016d`.

### Runtime and cost

| Field | Terminal value |
|---|---|
| dstack ID | `7e8852bd-d601-46ba-9fe1-c8df85077452` |
| Status / reason | `done` / `all_jobs_done` |
| Submitted / finished | `2026-07-24T20:55:43.691745Z` / `2026-07-24T20:59:48.919137Z` |
| Total elapsed | 245.227392 seconds |
| Backend / region | RunPod / `CA-MTL-4` |
| Exact offer | on-demand NVIDIA H200, 141 GiB VRAM, `$4.39/hour` |
| Host allocation | 28 CPUs, 251 GiB RAM, 200 GiB disk |
| Runtime | Python 3.10.12, PyTorch 2.7.0+cu126, CUDA 12.6, Transformers 4.57.0.dev0 |
| Peak allocated / reserved | 64,864,802,304 B / 64,990,740,480 B |
| Exit status | 0 |
| Exact cost | `$0.2990` |
| Combined `a1` + `a2` spend | `$0.4834` |

The wrapper exit status is zero because the probe intentionally records
per-row compatibility failures as structured results and successfully synced
them; it does not mean the model passed compatibility.

### Artifact provenance

The private immutable HF prefix contains exactly eight synced files. They were
downloaded into:

`data/vqa/benchmarks/qwen3-omni-compatibility-3/qwen3-omni-compatibility-3-20260724-a2`

Remote `SHA256SUMS` verification and the local collector checksum both passed.
Key hashes:

- predictions:
  `21238eb8fbb2eae3644f47bc00508ed938c3d2e720587a8dfd9d756d53238920`
- summary:
  `e88c88596e7e4f61473de80b27c32ffdfd363f6158f3214068ca71b041546b7e`
- sanitized dstack record:
  `938206d727dc47db1a8d92fa129dbb6528a4eb37e0f07d9cb9560b9ae2e644d0`
- local artifact manifest:
  `5985c59361ede7dc98239adc3c460e9f634207decf0bcf68f9365a8fddf2fec4`

## Corrected `a3` result

The corrected runner was submitted exactly once as immutable run
`qwen3-omni-compatibility-3-20260724-a3`. All three rows passed the joint input
contract:

- 12 visual elements and one audio element were present in the same request.
- Visual and audio tensors entered the same first model forward.
- Each row used exactly one joint generation call.
- Processed visual and audio SHA-256 attestations were recorded.
- No input-contract error occurred.

All three rows nevertheless failed strict output parsing. Each 2,048-token
Thinking generation produced a valid-looking JSON answer and then continued
with unsolicited or repeated trailing text. Because the answer object was not
terminal, the fail-closed normalizer returned the full output and the parser
correctly rejected it as not being a single JSON object.

| Role | Input contract | Parse | Generation time | Peak GPU allocated | Peak GPU reserved |
|---|---|---|---:|---:|---:|
| lexical speech | passed | failed | 195.537 s | 68,312,297,472 B | 70,086,819,840 B |
| prosody | passed | failed | 194.275 s | 74,436,415,488 B | 78,502,690,816 B |
| non-speech sound | passed | failed | 188.075 s | 74,346,811,392 B | 78,429,290,496 B |

Compatibility decision: **stop** on all three roles. The raw generations are
not successful contract outputs, so label-isolated semantic adjudication was
not performed. The semantic promotion gate remains **not evaluated**, and no
expansion is permitted.

### `a3` runtime, cost, and provenance

| Field | Terminal value |
|---|---|
| dstack ID | `52477784-dfe9-476e-9d10-b1d0c77cbc3a` |
| Status / reason | `done` / `all_jobs_done` |
| Submitted / finished | `2026-07-24T22:14:41.749480Z` / `2026-07-24T22:36:55.194926Z` |
| Total elapsed | 1,333.445446 seconds |
| Backend / region | RunPod / `CA-MTL-4` |
| Exact offer | on-demand NVIDIA H200, 141 GiB VRAM, `$4.39/hour` |
| Host allocation | 28 CPUs, 251 GiB RAM, 200 GiB disk |
| Corrected runner SHA-256 | `5ca8b1a1e81c6bd662890d2dc9160d1d7426d2967ccf132d4726ce1bfae0016d` |
| Exit status | 0 |
| Exact `a3` cost | `$1.6261` |
| Combined `a1` + `a2` + `a3` spend | `$2.1095` |

The private immutable `a3` HF prefix contains exactly eight files. They were
downloaded into ignored canonical storage at:

`data/vqa/benchmarks/qwen3-omni-compatibility-3/qwen3-omni-compatibility-3-20260724-a3`

Both checksum layers passed. Key hashes:

- predictions:
  `b5584e20f26e479fd3a2a5d18dcaad50f04c8e6ece083a279b145e68093f3d53`
- summary:
  `afdc2934b2f4315cb7bb8f156a554084925fa015ed95740962161b8bd2608aac`
- sanitized dstack record:
  `fff9d4507fb480be316e83f1d712389a43055afb602601cdd54ccc3e44024308`
- local artifact manifest:
  `38ed055e9232d7c261a5424b0982856e3b951a54b18f40e8e7625b690971bd47`

## Verification

- Repository suite: 178 tests passed.
- Focused Qwen/dstack contract suite: 12 tests passed.
- Ruff checks passed; both shell scripts passed `bash -n`.
- The final `$4.50/hour` `a2` preview showed four qualifying H200 offers.
- dstack confirms `a2` is terminal with exact cost `$0.2990`.
- The private HF prefix contains exactly the eight expected remote files.
- Both the remote and local artifact checksums passed.
- The corrected `a3` runner hash matched the pre-submit hash.
- `a3` attested joint visual/audio input consumption on all three rows.
- `a3` terminal state, `$1.6261` cost, eight-file remote prefix, and both
  checksum layers were independently re-read after completion.
- The allowlisted metadata projection reproduced the saved `a1` record from
  the live terminal dstack response.
- Independent static review found no remaining Critical or Important findings.
- After collecting `a4`, the repository suite passed with 187 tests; Ruff,
  shell syntax, and `git diff --check` also passed.
- `a4` ended on the configured EOS and passed strict JSON parsing on all three
  rows; the downloaded eight-file artifact set passed both checksum layers.

## Authorized `a4` termination correction

The user authorized one additional immutable attempt after review of the `a3`
output-conformance failure. The narrow hypothesis is that the tokenizer
defines `<|im_end|>` as EOS ID `151645` and `<|endoftext|>` as padding ID
`151643`, but neither ID is populated in the pinned checkpoint model or
generation configuration and `a3` did not pass them explicitly.

The `a4` runner therefore passes those exact tokenizer IDs to `generate`,
verifies both token identities, and fails closed unless the generated sequence
actually ends on EOS. Its artifacts record the token IDs and strings, generated
token count, final token ID, and EOS-termination result. The strict
terminal-JSON parser is unchanged.

The duration cap is reduced to 45 minutes. At the unchanged `$4.50/hour` price
cap, the worst-case new cost is `$3.375`; with `$2.1095` already spent, the
worst-case cumulative Qwen spend is `$5.4845`, below the authorized hard total
of `$10`.

## Completed `a4` result

Run `qwen3-omni-compatibility-3-20260724-a4` completed on a RunPod H200. The
explicit tokenizer termination IDs corrected the `a3` output-conformance
failure without relaxing the parser:

- all three rows supplied 12 frames and one native-audio input in the same
  request and same first model forward;
- all three used exactly one generation call;
- all three ended on `<|im_end|>` (token ID `151645`);
- all three returned one terminal JSON object accepted by the unchanged strict
  parser.

| Role | New tokens | EOS | Answer |
|---|---:|---:|---|
| lexical speech | 295 | yes | `True` |
| prosody | 507 | yes | Intense expression, forceful/raised voice, emphatic gestures |
| non-speech sound | 274 | yes | A cat meowing or yowling |

The automatic compatibility decision is therefore **promote (3/3)**. This
means the pinned model/runtime can jointly consume the frozen visual and audio
evidence and produce contract-valid outputs. It is not by itself a semantic
quality result.

### Label-isolated inference and post-run scoring

Reference labels were excluded from inference requests, and predictions were
frozen before the post-run label join and scoring. The same references had
already been used in earlier failure-panel and Gemma/Phi analysis, so this is
not a newly blinded evaluation. Exact match and token F1 use the repository's
frozen `vqa_evaluation.py` implementation at commit
`44d59446bcb26df292fb2094aadaf5b24b0b0208` (source SHA-256
`0cdd46df8347f3ba6e0986c9da349f6d89b41aa2ecfb15f637570a15f7480712`).
The exact answers, reference hashes and ignored annotation paths, metrics, and
scorer provenance are preserved in
`docs/experiments/artifacts/qwen3-omni-a4-semantic-score.json`. Running
`uv run python scripts/verify_qwen3_omni_a4_semantic_score.py` joins the
references from the ignored shared dataset, verifies their hashes, and
reproduces the scores. Manual judgments use the existing `C` (correct), `P`
(partial), and `W` (wrong) rubric from the matched Gemma/Phi comparison.

| Role | Reference meaning | Qwen3-Omni answer | Judgment | Token F1 |
|---|---|---|---:|---:|
| lexical speech | The bystander says she does not know | `True` | W | 0.0000 |
| prosody | Faster, louder, emphatic speech and forceful gestures | Raised/forceful voice, intense expression, emphatic gestures | C | 0.2619 |
| non-speech sound | A cat screaming | A cat meowing or yowling | P | 0.3333 |

Aggregate: `1C/1P/1W`, normalized exact match `0.0000`, mean token F1
`0.1984`. Applying issue #12's predeclared Gemma/Phi screen for comparability
(zero wrong and at least two correct), Qwen would fail that screen. This
threshold was not separately preregistered for Qwen issue #5.

Issue #5's own promotion gate remains **incomplete**, not failed: it requires a
material improvement on an audio-focused row without a visual-control
regression, but this compatibility panel has no visual-control row. The
prosody answer is a material audio-focused success; a matched visual control
is still required before Qwen can pass or fail its actual promotion gate.

The lexical row is also a benchmark-design confound. Its text is a declarative
description—“A woman ... asks a bystander how old the horse is.”—rather than
an explicit question such as “What does the bystander answer?” Qwen's `True`
is a natural truth-verification reading of that wording, but it still does not
match the official answer and is scored wrong. A future quality run should use
question-shaped prompts and keep any prompt repair explicitly separated from
the frozen benchmark result.

On the same tiny panel, Qwen3-Omni's `1C/1P/1W` ties Gemma CUDA and
Qwen3.5-4B+Whisper by manual category count, while its mean token F1 (`0.1984`)
is below the latter (`0.2333`). Those three examples are a compatibility
screen, not enough evidence to rank systems. Qwen should therefore be retained
for a broader stratified native-audio quality benchmark that includes matched
visual controls, not declared the new best baseline yet.

### `a4` runtime, cost, and provenance

| Field | Terminal value |
|---|---|
| dstack ID | `0f68deb4-6f5a-43ce-8d29-803dc5eda2c4` |
| Status / reason | `done` / `all_jobs_done` |
| Submitted / finished | `2026-07-25T00:48:11.465201Z` / `2026-07-25T00:55:15.762653Z` |
| Total elapsed | 424.297452 seconds |
| Backend / region | RunPod / `CA-MTL-3` |
| Exact offer | on-demand NVIDIA H200, 141 GiB VRAM, `$4.39/hour` |
| Host allocation | 22 CPUs, 221 GiB RAM, 200 GiB disk |
| Exit status | 0 |
| Exact `a4` cost | `$0.5174` |
| Cumulative `a1`–`a4` spend | `$2.6269` |

The private immutable HF prefix contains exactly eight files. Both remote and
local checksum layers passed. Key hashes:

- predictions:
  `3deaa7296fb737de1f832fba4cb187c0112ea2a62f45aafcf1198dfe37b19451`
- summary:
  `98797e4075edba97f09d912b362ebaac38d41f106a8e7f8fa862120341cfbe2a`
- sanitized dstack record:
  `8588b5442f0ea708f80e94bea066f6c195938cf02e2c91c4618dc1753361bd5e`
- local artifact manifest:
  `aeacc5e0ca2c06cf143ce19e54af5853846f560a1a38e4a6f80a8a45148c5b70`

No raw logs, raw inference predictions, media, model weights, credentials, or
dataset files are committed. The tracked scored artifact contains only the
three exact model answers, annotation lookup paths and reference hashes,
metrics, judgments, and provenance required to reproduce this report.

## References

- [Qwen3-Omni official repository](https://github.com/QwenLM/Qwen3-Omni)
- [Pinned Qwen3-Omni Thinking model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
- [dstack task configuration](https://dstack.ai/docs/reference/dstack.yml/task/)
