# Bounded multi-pass Gemini counting canary

This sanitized bundle indexes the completed 2026-08-16 twelve-row counting
canary. The v4 replicate completed both frozen arms, observed 1/8 known
failures rescued and 1/4 controls preserved, and was not promoted. The
refinement arm made no additional calls because all coverage responses encoded
no eligible dispute. An additive review then found material deviations from
the frozen condition, so the intended comparison is unusable.

- Evidence validity: **Invalid** (supersedes the original Diagnostic score)
- Decision disposition: **Not promoted under the tested condition**
- v4 provider cost: **$2.048074**
- cumulative v1-v4 exposure: **$3.358862**
- private score hash:
  `86575144f497c7176088ecf7f3fc31afd1cd83dc2b4ff669140c882b8d0d82f7`

`summary.json` contains only aggregate completion, decision, cost, and metric
fields from the original post-freeze score. `validity-correction.json` binds
that score, records the superseding validity state, and adds sanitized
count-unit and panel-stratum breakdowns. Questions, answers, row IDs, raw model
responses, media, credentials, and private paths are excluded. Complete
evidence remains in the canonical private benchmark root documented in the
experiment report.
