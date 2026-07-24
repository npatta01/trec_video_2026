# Issue #5: Qwen3-Omni dstack compatibility

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
  `20260724`. Store only the normalized terminal answer object and a SHA-256 of
  the unnormalized generation.
- Compatibility is not semantic scoring. Expansion still requires
  label-isolated adjudication showing an audio-row material improvement without
  a control or input-contract regression.

## Resource and cost envelope

- GPU: one NVIDIA H100, H200, B200, or RTX PRO 6000 WS with at least 90 GB VRAM
- CPU: at least 16 cores
- Host RAM: at least 192 GB
- Disk: 200 GB
- Price cap: $4.00/hour
- Duration cap: 2 hours
- Worst-case cost cap: $8.00

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
- [ ] Authorized detached compatibility run completed.
- [ ] Remote artifacts downloaded and hash-verified.
- [x] Static findings independently reviewed.
- [ ] Issue updated and draft PR opened.

## Current evidence and blocker

The $4.00/hour plan validated but found no offer inside the full resource
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
The private HF `a1` prefix remains empty. The tracked configuration is back at
the authorized $4.00/hour envelope and reserves `a2` for any future approved
attempt.

## Next action

Obtain explicit approval before raising the cap to $4.50/hour (2-hour hard
maximum: $9.00). If approved, re-preview immutable run `a2` and submit once
only if at least two full-resource offers remain. Otherwise stop without
evaluating compatibility or the semantic promotion gate.
