# Dense question-localized evidence diagnostic

This tracked directory is the sanitized evidence-of-record index for the
2026-08-16 paired dense-sampling diagnostic. It contains aggregate counts,
condition metadata, immutable artifact hashes, and the predeclared gate only.

The test compared a byte-identical local video window requested at 4 FPS and
12 FPS for eight selected visual targets and four supported controls. Two
targets remained explicitly ambiguous and were retained in completion but
excluded from the auditable-target recovery denominator. All 24 arms completed
and all 12 clips/24 arms were inspected.

The frozen score is negative under the tested condition: neither sampling arm
recovered a scored target fact, so denser frames produced zero incremental
rescues. Controls were preserved at 3/4 in both arms, with zero 12-FPS
regressions and zero unsupported observations. The predeclared gate failed.

A later additive manual failure decomposition materially narrows that result.
Five of the six targets counted in the frozen recovery denominator did not
contain the reference-level answer inside the selected clip. Across all eight
targets, the primary gaps were three reference/premise mismatches, two
truncated windows, two ambiguous question/reference pairs, and one genuine
model entity-focus error. Only that last row was a clean opportunity for the
FPS treatment; neither 4 FPS nor 12 FPS recovered it (`0/1` in both arms).
The original `0/6` score remains preserved, but it is not a valid
six-opportunity estimate of dense-frame recovery.

Control outputs and control clip sufficiency tell different stories. Three of
four controls produced the expected observation, but only one control clip
contained both the answer fact and its temporal anchor. The other three showed
the answer while placing the anchor outside or at the clip boundary; two were
answered and one routed to verification. That inconsistency is an answer-use
gap, not evidence that the other control windows were fully sufficient.

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

Evidence validity is **Diagnostic**; the mechanism interpretation is
**inconclusive after post-hoc failure decomposition** and the decision
disposition is **Not promoted under the tested condition**. This was a selected mechanism diagnostic, not a
prevalence estimate or router evaluation. It does not support another FPS
increase, but the panel also cannot establish a broad dense-FPS null effect.
The next valid test must independently verify, before provider authorization,
that each target's exact reference-level fact and temporal anchor are visible
inside the frozen clip.

`summary.json` records the sanitized aggregate result, failure decomposition,
and immutable hashes.
`SHA256SUMS` binds both tracked files. No row identifiers, questions,
references, evidence text, predictions, raw responses, private filesystem
paths, media, or credentials are included.
