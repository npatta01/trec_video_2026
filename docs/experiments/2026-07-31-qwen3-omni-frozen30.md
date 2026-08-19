# Qwen3-Omni frozen-30 A4 result

## Decision

The A4 comparison is **complete and scientifically eligible**, but neither
predeclared promotion gate passed.

Native audio produced a strong target-stratum signal: five of the twelve
dialogue/speech and sound/music rows improved, no target row changed from
correct to wrong, and the target ordinal delta was `+9`. The specialist gate
still failed because the eighteen temporal, spatial, and visual-control rows
had a `-1` ordinal delta and included one `C→W` regression. Qwen3-Omni therefore
remains a diagnostic native-audio candidate rather than an enabled route.

The frames-only condition also failed the general-model gate. Its exact match
and token F1 were both below the existing Qwen3.5-4B 12-frame baseline, and its
spatial/OCR/color stratum lost two exact answers. The current visual baseline
does not change.

## Frozen design and execution

- Scientific run: `qwen3-omni-frozen30-audio-20260731-a4`.
- Dstack development environment:
  `qwen3-omni-frozen30-audio-20260731-a4-dev`.
- Full-bundle non-development canary:
  `qwen3-omni-f30-a4-bundle-canary`.
- Model: `Qwen/Qwen3-Omni-30B-A3B-Thinking` at revision
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
- Runtime image digest:
  `sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`.
- Contract SHA-256:
  `9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5`.
- Neutral instruction SHA-256:
  `44b83ed11a2b9b8a62584302b126fb8eec13b03c55acb57d08b146f41c5e09a6`.
- Evidence bundle: 390 label-free files, comprising 360 PNG frames and 30 WAV
  clips, in a deterministic 318,177,280-byte private archive with SHA-256
  `9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943`.
- Bundle manifest SHA-256:
  `3f88bb004c9d95c433e544ddf0cf6b314deffd0d5a22e5adb2d3e1d09613ca74`.
- Conditions: the same twelve frozen frames, first without audio and then with
  the native WAV, for each of thirty balanced labeled-2025 examples.
- Decoding: unchanged deterministic frozen configuration with seed `20260724`.
- Integrity boundary: private bundle verification and reconstruction, exact
  preflight, label-blind inference, prediction freeze, randomized answer-only
  adjudication, and only then reference-backed scoring.

The canary downloaded and verified the complete bundle through the exact
runtime and secret path, reconstructed all 390 files, passed the frozen
preflight, and terminated with `all_jobs_done`. The dev workload then completed
`60/60` strict outputs in expected order with workload return code `0`; all nine
remote artifacts and the collector's exhaustive local hashes verified.

## Metrics

| Condition | Exact match | Token F1 | METEOR |
|---|---:|---:|---:|
| Qwen3-Omni frames only | 0.233333 | 0.418762 | 0.302982 |
| Qwen3-Omni frames + native audio | 0.266667 | 0.445415 | 0.345087 |
| Audio minus frames | +0.033334 | +0.026653 | +0.042105 |
| Qwen3.5-4B 12-frame baseline | 0.300000 | 0.430725 | 0.286064 |

Audio-on improved exact match from seven to eight answers. Its strongest
stratum was dialogue and speech: `4/6` exact, token F1 `0.715447`, and METEOR
`0.532531`, versus `2/6`, `0.457317`, and `0.311880` for frames only. Neither
condition produced an exact answer in non-speech sound/music or temporal/action.

Frames only produced `4/6` exact answers in spatial/OCR/color, versus `6/6` for
the existing baseline. That two-answer deficit fails the general gate's
per-stratum floor in addition to the aggregate exact and F1 thresholds.

## Condition-blind adjudication

A fresh reviewer received only randomized `{question, reference, answer}`
items. The packet excluded condition names, audio flags, pair indexes, model
evidence, and the blind key. The 60 judgments were `34 C / 8 P / 18 W`.

After unblinding, paired transitions were:

| Transition | Rows |
|---|---:|
| `C→C` | 12 |
| `C→P` | 2 |
| `C→W` | 1 |
| `P→C` | 2 |
| `P→P` | 2 |
| `W→C` | 5 |
| `W→W` | 6 |

The twelve target rows had five improvements and ordinal delta `+9`. The
eighteen control rows had ordinal delta `-1` and one `C→W`. The native-audio
specialist gate therefore failed only its two control-safety clauses. The
audio-on exact-match non-regression and every target-stratum clause passed.

## Scoring repair and reproducibility

The first label-free prediction freeze succeeded. The first evaluator attempt
then exposed two scorer-schema omissions: the pinned Transformers first
forward legitimately added `return_dict` and `use_audio_in_video`, and the
frozen baseline contains separate exact field sets for frame-based and
native-video rows. Both repairs were test-first, limited to exact allowlists,
and retained rejection of unknown fields.

Because the second repair occurred after the first evaluator had opened the
baseline schema, the scorer-drift guard correctly rejected silent reuse. The
entire first scoring attempt was preserved as an ignored audit artifact. A new
label-isolated container froze the same selected prediction bytes under the
committed repaired scorer. The selected-predictions SHA-256 remained
`819a63aba6066a5b79616b7c4aee7a745f2489284172d978fe4ae833c16f5d87`.
A newly shuffled packet was adjudicated again and reproduced the same aggregate
`34 C / 8 P / 18 W`. Two independent evaluator executions produced identical
sanitized bytes with SHA-256
`2344e96361112e06142fdf588bad448a2cdcc3314576313f18bc7cbce1c0be73`.

## Cost, validity, and limits

- A4 full-bundle canary: `$0.2958`.
- A4 development-environment inference: `$2.9943`.
- A4 operational total: `$3.2901`.
- Earlier A3 bulk-transfer diagnostic: `$1.2035`, stopped before inference.

Evidence validity is **Verified** for the exact frozen A4 conditions and panel.
The result does not estimate performance on the unlabeled 2026 evaluation set,
does not prove that audio is globally harmful or helpful, and does not identify
whether the single control regression came from audio fusion, evidence
selection, or answer selection. The next useful work is a label-blind error
audit of the five target wins and the one control `C→W`, followed by a narrower
route only if a predeclared selector can retain the wins without control loss.

The tracked sanitized score is
[`qwen3-omni-frozen30-a4-score.json`](artifacts/qwen3-omni-frozen30-a4-score.json).
Historical infrastructure attempts remain in
[`2026-07-30-qwen3-omni-frozen30-infrastructure-attempts.md`](2026-07-30-qwen3-omni-frozen30-infrastructure-attempts.md).

## Seeded termination-repair follow-up

The A4 quality decision above remains unchanged. A later development-mode
follow-up isolated and repaired a completion-boundary failure: one sampled
frames-only generation had reached the `2,048`-token cap without emitting EOS.
The follow-up changes the seed only through a new sealed profile identity; it
does not add semantic labels, change the evidence bytes, or claim new quality
evidence.

- **Reason for testing:** determine whether the prior non-EOS failure could be
  reproduced and contained before spending on another quality comparison.
- **Tested condition and panel:** the same thirty frames-only frozen examples,
  visual-concise instruction, model revision, runtime image, twelve frames,
  and explicit sampled decoding (`temperature=0.6`, `top_p=0.95`, `top_k=20`),
  with the sealed retry seed changed from `20260724` to `42`.
- **Observed result:** run `qwen3-omni-f30-calibrated-20260801-s5-seed42`
  completed `30/30` rows in expected order. Every row parsed, used seed `42`,
  and ended with EOS; token counts ranged from `87` to `845`. The prior
  `Hw0bRvn1ZIo` row (`pair_index=18`) completed at `845` tokens with EOS.
- **Current condition-bound status:** **Verified execution repair for the
  sealed seed-42 profile; no promotion weight.** The dev run cost `$1.4805`
  and its canary cost `$0.1273`; the environment was terminated after its
  ten checksum-bound artifacts synced to the private HF prefix.
- **What this rules out now:** treating the original failure as an unresolved
  blocker for running this sealed profile, or claiming that the completed
  A4 quality comparison was invalid solely because of the earlier retry.
- **What it does not rule out:** non-EOS behavior under another seed, a larger
  workload, or a different decoding profile; it also does not establish that
  Qwen3-Omni is more accurate than Qwen3.5-4B.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  successful retry is one explicitly selected seed and the prior stopped-run
  record was not retained as a complete prediction artifact. The termination
  gate now records seed, final token, and EOS status and fails closed on a
  future cap-without-EOS row.
- **Reconsider when:** a future matched quality experiment has a new question
  (for example, a predeclared specialist route that addresses A4's control
  regression) or a recurrence under a different sealed seed justifies a
  bounded multi-seed reliability probe.

**Evidence of record:**
[`summary.json`](../../data/vqa/benchmarks/qwen3-omni-frozen30/qwen3-omni-f30-calibrated-20260801-s5-seed42/summary.json),
[`predictions.jsonl`](../../data/vqa/benchmarks/qwen3-omni-frozen30/qwen3-omni-f30-calibrated-20260801-s5-seed42/predictions.jsonl),
and the private HF prefix
`hf://buckets/Npatta01/trec_video_2026/experiments/qwen3-omni-f30-calibrated-20260801-s5-seed42`.
