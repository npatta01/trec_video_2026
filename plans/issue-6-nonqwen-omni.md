# Issue #6: non-Qwen omni compatibility

## Objective

Probe Phi-4-multimodal-instruct and Gemma 3n E4B-it on the same frozen
three-video lexical/prosody/non-speech panel, requiring a single combined
vision/audio request and first forward for each row.

## Scope

- Freeze and validate a shared ignored three-row contract.
- Implement fail-closed, model-specific ROCm BF16 runners.
- Record model/runtime revisions, processed-input hashes, first-forward keys,
  runtime, peak memory, parse status, and promotion decision.
- Keep datasets, model weights, predictions, and raw logs out of Git.
- Publish only a branch, draft PR, sanitized findings, and artifact hashes.

## Decisions

- Contract canonical SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Phi uses its declared Transformers 4.48.2 runtime and ROCm AOTriton SDPA.
- Gemma uses Transformers `5.15.0.dev0` with pinned `timm==1.0.28`.
- Promotion is fail-closed and requires exactly three unique passed rows.
- Phi stops because all three outputs failed the requested JSON parse contract.
- Gemma promotes because all three rows satisfied every compatibility gate.

## Status

- [x] Isolated worktree and baseline verification
- [x] Frozen label-blind three-row contract
- [x] Joint-input runner and attestation gate
- [x] Pinned rootless Podman ROCm runtimes
- [x] Phi three-row run and verified stop decision
- [x] Gemma three-row run and verified promote decision
- [x] Sanitized findings and artifact hashes
- [x] Final verification and independent review
- [x] Push, issue comment, and draft PR

## Verification evidence

- Independent post-run review: ready to merge with no Critical or Important
  findings at `413d13b`.
- Phi predictions:
  `6a1ee0d5ea29d150f871f7942c6377643ad6a0d86b4df0953a339d7e6c890478`
- Gemma predictions:
  `9797c14694bf99a438a76eba27f64fd336cae06526e11d8d2faa99021423ef00`
- Fresh full verification: 166 tests passed; Ruff, relevant format checks,
  shell syntax, diff checks, deterministic contract regeneration, and
  recomputed promotion decisions passed.

## Next action

Await review and merge of draft PR #11.
