# Gemini 3.7 Flash two-video native-AV smoke

This sanitized bundle records the smallest matched check requested on
2026-08-16. The same two WebM deliveries and direct-answer prompt were sent by
Batch to `gemini-3.1-pro-preview` and stable `gemini-3.7-flash` at four FPS.

Both models returned two non-empty answers and both arms landed inside the
frozen custom-FPS token band. The immutable strict freeze remains `not_ready`
because the Batch responses returned each requested model alias, while the
harness required the separate dated catalog version string. That strict result
was not rewritten.

A bounded no-new-inference diagnostic over the already returned answers found:

- both models were semantically correct on both rows in an unblinded manual
  check;
- exact match was 0/2 for both;
- Gemini 3.7 Flash won one token-F1 comparison and lost one;
- mean token F1 was 0.2496 for 3.7 Flash and 0.4365 for 3.1 Pro, largely
  reflecting the candidate's more verbose answers;
- observed Batch cost was $0.067445.

The result establishes that Gemini 3.7 Flash can complete this native
audio/video path. It does **not** show that Flash is better than the 3.1 Pro
control, and it does not justify a larger run.

`summary.json` contains only sanitized aggregates and hashes. Questions,
references, answers, provider identifiers, and row-level judgments remain in
the private evidence root
`data/vqa/benchmarks/gemini37-flash-two-video-smoke-v2/`. The earlier
`data/vqa/benchmarks/gemini37-flash-two-video-smoke-v1/` setup-only root is
also preserved: one file upload reached Google, a local digest-format parser
rejected its metadata, and no Batch or inference call occurred. V2 is a
separate replacement, not a continuation of v1.
