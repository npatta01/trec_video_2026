# Typed video count-query engine index

This tracked directory contains only the sanitized aggregate record for the
2026-08-14 zero-provider-cost count-query engine oracle slice. It contains no
video IDs, questions, labels, answers, row-level proofs, or media.

Complete private evidence is retained at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-count-query-engine-v1/`

Start with the private `inspection.md` when authorized. The canonical report is
`docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`.

The frozen architecture expressed all 12 held-out questions, produced 10 exact
proofs and two bounded abstentions, but four exact answers were wrong. It
rescued 3/8 prior failures and preserved 3/4 controls, so it failed the
predeclared gate. The negative isolates the next problem to mechanically
verifiable spatial/temporal coverage rather than more row-specific reducers.

Independent review after unblinding found label-independent v1 algebra and
proof-audit defects outside the tested rows. The immutable v1 result is
preserved, and an additive hardened proof-schema-v2 replay over the unchanged
authored inputs produced identical row outcomes, bounds, reasons, and aggregate
metrics. The private provenance manifest binds both source revisions and both
artifact chains; this correction does not change the negative disposition.

- **Evidence validity:** Diagnostic.
- **Decision disposition:** Not promoted under the tested condition.
- **Decision code:** `stop_count_query_engine_v1`.
