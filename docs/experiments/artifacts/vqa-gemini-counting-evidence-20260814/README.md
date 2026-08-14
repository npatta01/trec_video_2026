# Question-driven atomic counting evidence index

This tracked directory contains only a sanitized summary of the 2026-08-14
six-row counting-evidence canary. It contains no video IDs, questions, labels,
answers, raw model responses, credentials, or media.

The complete private evidence is retained at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-counting-evidence-v1/`

Start with the private `inspection.md` when authorized to inspect the data.
The canonical report is
`docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`.
`advisor-review.md` records the sanitized three-lane architecture review and
its bounded zero-cost next action.

The original write-once metrics file lacked embedded runtime versions. A
separately hashed offline CPU replay reproduced every frozen metric value and
now binds the exact dependency lock, scorer, model snapshot, and tokenizer
resources. `summary.json` records the environment and the explicitly
post-freeze provenance limitation without exposing row data.

The result is a valid negative for the tested OpenRouter-delivered atomic
counting treatment: 1/5 frozen failures were rescued and 1/1 control was
preserved, below the predeclared 3/5 rescue gate. No scale-up was submitted.

- **Evidence validity:** Diagnostic.
- **Decision disposition:** Not promoted under the tested condition.
- **Decision code:** `stop_atomic_counting_v1`.
