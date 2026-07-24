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
- Phi model revision:
  `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; Transformers 4.48.2
- Gemma model revision:
  `c1221e9c62e34a43ab7ffacd1be0ea71f126ef10`; Transformers 5.14.1,
  timm 1.0.28, librosa 0.11.0
- CUDA 12.4 / PyTorch 2.5.1 / BF16
- One on-demand NVIDIA GPU from the model-specific closed pool
- RunPod or Vast.ai only; no ports
- Price cap: Phi $2.00/hour; Gemma $1.50/hour
- Duration cap: one hour per submitted run
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
- [x] Submit and verify Phi `a4`
- [x] Freeze and independently review the corrected Gemma CUDA source
- [x] Submit and checksum-verify Gemma `a1` failure evidence
- [ ] Submit and verify corrected Gemma CUDA three-row comparison

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
- `phi4mm-cuda-quality-3-20260724-a4`: succeeded on a RunPod
  A100-SXM4-80GB at `$1.49`/hour; cost `$0.0951`. All three rows retained
  joint first-forward audio/vision attestation. Blinded adjudication was
  `0 correct / 2 partial / 1 wrong`, so Phi failed the frozen quality gate.
  Prediction SHA-256:
  `f935265e4c15572d36e500ea2097ec83eebe88bc811a9d919cd805c27982b27d`.
- `gemma3n-cuda-quality-3-20260724-a1`: all three joint CUDA forwards ran
  successfully on a RunPod L40S 48 GB at `$0.99`/hour; cost `$0.0625`.
  Lexical and prosody generation each reached the exact 160-token ceiling and
  stopped mid-JSON, so the fail-closed parser correctly rejected both rows.
  Non-speech sound stopped naturally at 39 tokens and passed. The checksummed
  evidence isolates output length as the failure; no prompt, model, media, or
  modality-attestation failure occurred.

All four runs synced immutable, checksummed artifacts to their private
run-specific HF Bucket prefixes. The price cap, provider set, spot policy, GPU
envelope, and model/image revisions were never broadened. Total Phi billed
cost was `$0.2522`.

## Next action

After independent review, submit immutable Gemma `a2` once under the
`$1.50/hour` and one-hour caps. The only inference change is increasing
Gemma's maximum generated tokens from 160 to 320; decoding remains greedy and
the prompt, model revision, and exact media remain frozen. Download and
checksum-verify its private artifacts, adjudicate the three rows, and record
the matched comparison. Expansion remains stopped unless new coordinator
evidence explicitly revises the frozen gate.
