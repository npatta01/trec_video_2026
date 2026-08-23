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
- `gemini-direct-v2-v3-full-summary.json`: label-blind full-panel completion,
  usage, cost, fallback counts, submission hashes, and integrity results for
  the shared direct-video pass that produced V2 and V3.
- `gemini-direct-retry-summary.json`: bounded recovery of the nine invalid
  provider outputs, including the fresh-upload and normalized-MP4 attempts,
  final fallback counts, costs, and V2b/V3b hashes.

The canonical interpretation is the
[dated experiment report](../../2026-08-21-vqa-mc-poc20.md).
