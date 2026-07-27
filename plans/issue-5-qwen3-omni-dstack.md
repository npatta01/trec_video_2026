# Issue #5: Qwen3-Omni dstack compatibility

> **Historical status:** The referenced PR is merged and this compatibility
> work is complete. Compatibility did not establish semantic quality; the
> later strict quality run failed its output contract, so Qwen3-Omni is not
> promoted. See [Project Status](../docs/project-status/index.md).

## Objective

Run the exact frozen three-video joint visual/audio compatibility contract with
`Qwen/Qwen3-Omni-30B-A3B-Thinking` on a detached dstack task, preserve
reproducibility and provenance, and make an honest expand/stop decision.

## Scope

- GitHub issue: `npatta01/trec_video_2026#5`
- Branch: `codex/issue-5-qwen3-omni-dstack`
- Base dependency: `codex/issue-6-nonqwen-omni` / draft PR #11
- Canonical contract:
  `data/vqa/benchmarks/omni-compatibility-3/contract.json`
- Canonical contract SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Video order:
  `NN7G28sBOlY`, `NUmb2P3Bzu8`, `NkXPPJoZ-h8`

## Decisions

- Use the official Qwen runtime image pinned at
  `sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`.
- Pin the model at revision
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
- Supply the 12 frozen frames and native audio together in one request. This
  tests joint sampled-visual/audio consumption, not native temporal-video
  decoding.
- Upload only the exact contract, 36 frames, and three audio files through
  dstack. Verify hashes before and after inference.
- Sync small outputs to the existing private HF Bucket at the immutable
  run-specific prefix, then download them into the ignored canonical benchmark
  directory.
- Keep ports absent. Use on-demand capacity from only RunPod and Vast.ai with
  a bounded 30-minute `no-capacity` retry.
- Preserve the checkpoint's Thinking generation parameters, explicitly enable
  sampling, bound generation at 2,048 new tokens, and seed Torch with
  `20260724`. Explicitly pass the checkpoint tokenizer's `<|im_end|>` EOS and
  `<|endoftext|>` padding IDs, fail closed unless generation ends on EOS, and
  attest the configured IDs, token strings, generated-token count, and final
  token. Store only the normalized terminal answer object and a SHA-256 of the
  unnormalized generation.
- Compatibility is not semantic scoring. Expansion still requires
  label-isolated adjudication showing an audio-row material improvement without
  a control or input-contract regression.

## Resource and cost envelope

- GPU: one NVIDIA H100, H200, B200, or RTX PRO 6000 WS with at least 90 GB VRAM
- CPU: at least 16 cores
- Host RAM: at least 192 GB
- Disk: 200 GB
- Price cap: $4.50/hour
- Duration cap for the authorized `a4` correction: 45 minutes
- Worst-case `a4` cost cap: $3.375
- Prior cumulative spend: $2.1095
- Worst-case cumulative spend through `a4`: $5.4845

## Status

- [x] Isolated worktree created and shared `data` symlink verified.
- [x] Baseline test suite: 166 passed.
- [x] Frozen contract file, canonical digest, IDs, 36 frames, and three audio
  files verified.
- [x] Static dstack config and Qwen runner implemented test-first.
- [x] Private HF Bucket and empty immutable run prefix verified.
- [x] Exact dstack plan previewed.
- [x] Stopped `a1` provisioning attempt recorded under ignored canonical data.
- [x] Future immutable run identity advanced to `a2`.
- [x] Authorized detached compatibility run completed.
- [x] Remote artifacts downloaded and hash-verified.
- [x] Corrected immutable `a3` run completed and evaluated.
- [x] Static findings independently reviewed.
- [x] Issue updated and stacked draft PR #14 opened.
- [x] User authorized an EOS-corrected `a4` rerun under the existing hard
  cumulative cap below $10.
- [x] EOS/padding wiring and fail-closed termination checks implemented
  test-first.
- [x] `a4` private artifact prefix verified empty and qualifying H200 offers
  previewed.
- [x] Corrected immutable `a4` run completed and checksum-verified.
- [x] `a4` label-free inference output frozen before post-run scoring.
- [x] Final `a4` findings independently reviewed and prepared for PR #14.

## Current evidence and blocker

The original $4.00/hour plan validated but found no offer inside the full resource
envelope. Two Vast.ai H200 offers at $3.9525/hour have only 48 GB host RAM and
100 GB disk. Four physically distinct RunPod H200 offers satisfy the full
resource envelope at $4.39/hour.

An `a1` run was submitted after temporarily raising the cap to $4.50/hour
without first obtaining explicit approval. It was stopped during provisioning;
no model exit status or remote result artifact exists. The terminal run record
is:

- dstack ID: `d1b2082d-4020-4aee-b393-979d3bcb1f85`
- status/reason: `terminated` / `stopped_by_user`
- elapsed provisioning time: 151.245703 seconds
- exact cost: $0.1844
- selected offer: RunPod `CA-MTL-4`, on-demand H200 141 GiB at $4.39/hour
- host allocation: 28 CPUs, 251 GiB RAM, 200 GiB disk

The sanitized allowlisted record is stored at
`data/vqa/benchmarks/qwen3-omni-compatibility-3/qwen3-omni-compatibility-3-20260724-a1/dstack-run.json`
with SHA-256
`d4b704d703d53f06cabf00f483e19cbe8306caccad584b225f0bf0b6daf86efc`.
The private HF `a1` prefix remains empty. The user subsequently authorized a
$4.50/hour price cap with a hard total envelope below $10. The tracked
configuration reserves `a2` for that single approved attempt.

Run `a2` completed on a RunPod H200 at $4.39/hour:

- dstack ID: `7e8852bd-d601-46ba-9fe1-c8df85077452`
- status/reason: `done` / `all_jobs_done`
- elapsed: 245.227392 seconds
- exact cost: $0.2990
- combined `a1` + `a2` spend: $0.4834
- remote artifacts: eight files, downloaded and checksum-verified
- automatic compatibility gate: **stop** on all three roles
- failure: `Input type (float) and bias type (c10::BFloat16) should be the same`
- semantic gate: not evaluated because no answer was generated

The preflight passed, but the executed runner transferred inputs to the device
without applying the BF16 model dtype. All rows failed in `generation` before
joint visual/audio first-forward attestation could complete. This is an
inconclusive probe input-dtype failure, not evidence of Qwen incompatibility.
The runner is corrected to use the official device-plus-dtype transfer. The
user authorized exactly one fresh immutable `a3` attempt under the existing
$4.50/hour and two-hour envelope. Combined prior spend is $0.4834, so the
native $9.00 run cap keeps cumulative Qwen spend below $10.

Run `a3` completed on the same qualifying RunPod H200 envelope:

- dstack ID: `52477784-dfe9-476e-9d10-b1d0c77cbc3a`
- status/reason: `done` / `all_jobs_done`
- elapsed: 1,333.445446 seconds
- exact `a3` cost: $1.6261
- cumulative `a1` + `a2` + `a3` spend: $2.1095
- remote artifacts: eight files, downloaded and checksum-verified
- joint visual/audio input contract: passed on all three rows
- output parsing: failed on all three rows because valid-looking JSON was
  followed by unsolicited or repeated trailing generation
- automatic compatibility gate: **stop**
- semantic gate: not evaluated because no contract output succeeded

The user subsequently authorized testing Qwen3-Omni again because native
audio-video question answering remains strategically important. The `a4`
hypothesis is narrow: the pinned tokenizer defines `<|im_end|>` as EOS ID
`151645` and `<|endoftext|>` as padding ID `151643`, while the checkpoint model
and generation configs leave their corresponding IDs null. The `a3` runner did
not pass tokenizer termination IDs, and every row exhausted the 2,048-token
limit after generating at least one valid-looking JSON candidate.

Run `a4` completed on a RunPod H200:

- dstack ID: `0f68deb4-6f5a-43ce-8d29-803dc5eda2c4`
- status/reason: `done` / `all_jobs_done`
- elapsed: 424.297452 seconds
- exact `a4` cost: $0.5174
- cumulative `a1`–`a4` spend: $2.6269
- remote artifacts: eight files, downloaded and checksum-verified
- joint visual/audio input contract: passed on all three rows
- EOS termination and strict JSON parsing: passed on all three rows
- automatic compatibility gate: **promote**
- semantic result: `1C/1P/1W`, exact match `0.0000`, mean token F1 `0.1984`
- existing issue-12 screen applied for comparability: **would fail**
- issue #5 promotion gate: **incomplete** because the panel has no visual
  control

The lexical row is confounded by a declarative benchmark prompt, which Qwen
interpreted as a truth-verification statement and answered `True`. It remains
wrong against the official answer. The prosody answer is correct and the cat
sound answer partial. This panel establishes working native audio/vision
inference, but it is too small, prompt-confounded, and lacks the visual control
required by issue #5 to establish a new best baseline.

## Next action

Preserve this compatibility result in draft PR #14 and define a separate,
broader stratified native-audio quality benchmark with explicit
question-shaped prompts and matched visual controls before spending more of
the Qwen cap.
