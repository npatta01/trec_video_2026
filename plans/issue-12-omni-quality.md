# Issue #12: omni semantic quality correction

## Objective

Separate joint audio/vision runtime compatibility from semantic answer quality
for Gemma 3n E4B-it and Phi-4-multimodal-instruct. Correct the strict-JSON
overinterpretation from issue #6 without weakening provenance or accepting
empty/malformed structured output.

## Fixed scope

- Base dependency: draft PR #11 / `codex/issue-6-nonqwen-omni`
- Contract:
  `data/vqa/benchmarks/omni-compatibility-3/contract.json`
- Canonical contract SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Existing Gemma prediction SHA-256:
  `9797c14694bf99a438a76eba27f64fd336cae06526e11d8d2faa99021423ef00`
- Existing Phi ROCm prediction SHA-256:
  `6a1ee0d5ea29d150f871f7942c6377643ad6a0d86b4df0953a339d7e6c890478`
- Labels remain unavailable until model outputs are frozen.
- No dataset contents, model weights, predictions, or raw logs are tracked.

## Frozen quality gate

This gate was recorded before revealing the new CUDA predictions. A model
passes the three-row semantic gate only if:

1. all three rows retain exact-request and joint first-forward audio/vision
   attestation;
2. all three normalized answers are non-empty and scoreable;
3. blinded adjudication yields zero `wrong` rows and at least two `correct`
   rows.

The 10–30 row expansion occurs only if both Gemma and Phi pass. This is a
screening rule for a tiny diagnostic panel, not a leaderboard estimate.

## Runtime and budget

- Image:
  `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`
- Model revision:
  `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- CUDA 12.4 / PyTorch 2.5.1 / Transformers 4.48.2 / BF16
- One on-demand NVIDIA A100, H100, or H200 with at least 80 GB VRAM
- RunPod or Vast.ai only; no ports
- Price cap: $2.00/hour
- Duration cap: one hour; worst-case cost cap: $2.00 per submitted run
- Additional issue #12 CUDA authorization: `$10.00` combined cap for Phi and
  Gemma after the first three attempts; stop and ask before exceeding it.

## Status

- [x] Isolated worktree and canonical shared-data link verified
- [x] Baseline suite: 166 passed
- [x] Existing Gemma and Phi outputs normalized label-blind
- [x] Existing Gemma and Phi outputs adjudicated after the label reveal
- [x] Fail-closed plain-text normalization implemented test-first
- [x] Pinned CUDA/dstack task implemented and statically verified
- [x] CUDA Phi artifacts downloaded and checksum-verified
- [x] Final matched comparison and promote/stop decision recorded
- [x] Fresh verification and independent review
- [x] Draft PR and issue update
- [x] Explicit authorization received for one corrected Phi `a4` and an exact
  three-row Gemma CUDA comparison
- [x] Phi `a4` reviewer blockers fixed and independent re-review passed
- [x] Corrected Phi preview: five eligible RunPod A100 offers,
  `$1.39`–`$1.49`/hour
- [ ] Submit and verify Phi `a4`
- [ ] Submit and verify Gemma CUDA three-row comparison

## Publication

- Blocked draft PR:
  <https://github.com/npatta01/trec_video_2026/pull/15>
- Issue #12 update:
  <https://github.com/npatta01/trec_video_2026/issues/12#issuecomment-5074164932>

## Execution history

- `phi4mm-cuda-quality-3-20260724-a1`: preflight stopped before model download
  because the generic contract validator required an unused source-video file;
  cost `$0.0489`.
- `phi4mm-cuda-quality-3-20260724-a2`: the input-only preflight passed and the
  pinned model downloaded, but the same source-video requirement remained at
  the probe boundary; cost `$0.0598`.
- `phi4mm-cuda-quality-3-20260724-a3`: the pinned model downloaded and the
  NVIDIA A100/CUDA 12.4 runtime attested, but model loading stopped before
  inference because the runtime omitted `scipy`; cost `$0.0484`.

All three stopped runs synced immutable, checksummed failure artifacts to their
private run-specific HF Bucket prefixes. The price cap, provider set, spot
policy, GPU envelope, and model/image revisions were never broadened. Total
billed cost was `$0.1571`.

## Next action

Submit the reviewed immutable Phi `a4` once under the same `$2.00/hour` and
one-hour caps. Download and verify its private artifacts before constructing
the separate Gemma CUDA three-row run. Expansion remains stopped unless new
coordinator evidence explicitly revises the frozen gate.
