# Dense question-localized evidence diagnostic

This tracked directory is the sanitized evidence-of-record index for the
2026-08-16 paired dense-sampling diagnostic. It contains aggregate counts,
condition metadata, immutable artifact hashes, and the predeclared gate only.

The test compared a byte-identical local video window requested at 4 FPS and
12 FPS for eight selected visual targets and four supported controls. Two
targets remained explicitly ambiguous and were retained in completion but
excluded from the auditable-target recovery denominator. All 24 arms completed
and all 12 clips/24 arms were inspected.

The result is negative under the tested condition: neither sampling arm
recovered an auditable target fact, so denser frames produced zero incremental
rescues. Controls were preserved at 3/4 in both arms, with zero 12-FPS
regressions and zero unsupported observations. The predeclared gate failed.

Attempt accounting is separate from the final score denominator. Across
authorized stages there were 48 provider attempts: the initial authorization
issued 24 v2 attempts, all HTTP 400 `INVALID_ARGUMENT` schema rejects before a
model response; after schema repair, the user separately reauthorized the
24-arm treatment and v4 plus v22 issued 24 unique model-bearing arms. The
final score denominator is 24 model-bearing arms, not 48, with no retry or
duplicate arm.

The original v22 prediction freeze preserved 16 passed arms and 8 contract
errors. Correction-v2 was a post-hoc, output-informed, zero-call parser repair
that accepted those eight responses; it is not a clean completion of the
predeclared acceptance protocol. This is **Diagnostic/post-hoc corrected**
evidence. The only repaired target pair remained unrecovered, so the negative
target conclusion is robust (`0/6` corrected, `0/5` if that pair is excluded);
control recovery is acceptance-boundary-sensitive (only `1/4` control pairs
had both arms accepted before correction versus `3/4` recovered after it) and
does not upgrade the gate.

Evidence validity is **Diagnostic** and the decision disposition is **Not
promoted under the tested condition**. This was an oracle-screened,
label-aware mechanism diagnostic, not a prevalence estimate or router
evaluation. It rules out this dense-FPS recovery mechanism for the tested
localized windows and provider condition; it does not rule out
question-conditioned evidence selection/verification or modality-specific
acquisition under a newly frozen and authorized condition.

`summary.json` records the sanitized aggregate result and immutable hashes.
`SHA256SUMS` binds both tracked files. No row identifiers, questions,
references, evidence text, predictions, raw responses, private filesystem
paths, media, or credentials are included.
