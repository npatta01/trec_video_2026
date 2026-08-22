# Twenty-video MC POC aggregate evidence

This tracked directory contains only sanitized aggregate evidence for the
2026-08-21 multiple-choice diagnostic. Private questions, choices, transcripts,
labels, predictions, media, and HF Bucket locations are intentionally omitted.

- `summary.json`: frozen aggregate scores, completion, cost, run identity, and
  hashes of the ignored private evidence of record.
- `adaptive-agent-summary.json`: frozen aggregate scores, inspection choices,
  completion, cost, gate result, and private-evidence hashes for the two-stage
  adaptive-inspection follow-up.
- `deliberative-agent-summary.json`: three-attempt validity record, frozen
  aggregate scores, deliberation/path diagnostics, completion, cost, gate
  result, manual-review counts, and private-evidence hashes for the bounded
  single-model deliberative follow-up.
- `four-option-agent-summary.json`: verified live-model capability and formal
  cost preflight for the mandatory-inspection, four-option evidence agent. The
  `$1.50` gate blocked paid inference before media upload, so it contains no
  accuracy result.

The canonical interpretation is the
[dated experiment report](../../2026-08-21-vqa-mc-poc20.md).
