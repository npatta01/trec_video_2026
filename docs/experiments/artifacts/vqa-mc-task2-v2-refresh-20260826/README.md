# Revised Task 2 VQA-MC refresh evidence

This compact bundle records the 2026-08-26 full refresh against NIST's revised
Task 2 query file.

- [`task2-diff-summary.json`](task2-diff-summary.json) records the organizer
  file delta without copying row text or answer options.
- [`run-summary.json`](run-summary.json) records coverage, fallbacks, hashes,
  disagreement counts, and cost evidence for the three refreshed candidates.

Private prompts, provider responses, row-level predictions, audit decisions,
and media remain under the ignored `data/` tree. The three complete candidate
files are tracked separately under
[`submissions/trecvid-2026/vqa-mc-task2-v2/`](../../../../submissions/trecvid-2026/vqa-mc-task2-v2/README.md).
