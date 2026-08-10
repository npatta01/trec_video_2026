# OpenRouter transport report

## RED

- Added offline transport, preflight, CLI-selection, provenance, lifecycle, malformed-response, and secret-redaction tests before the transport implementation.
- The first focused run failed at collection with `ModuleNotFoundError` for `trec_video_2026.vqa_submission_openrouter`, confirming the feature was absent.

## GREEN

- Added the stdlib/injected-opener `OpenRouterTransport` with local streamed hashing, bounded in-memory data-URI storage, multimodal chat conversion, strict JSON-schema requests, deterministic seeds, ZDR/provider constraints, safe response parsing, usage accounting, cleanup, and bounded redacted errors.
- Added authenticated no-media model preflight and runner `openrouter` transport/provider choices.
- Preserved direct frozen model provenance and added durable OpenRouter WebM preparation for MKV source media with exact resume reuse.
- Verification: `202 passed` for the requested pytest command; Ruff reports `All checks passed!`; `git diff --check` is clean.

## Commit

- Target commit message: `feat: add OpenRouter Gemini VQA transport`
- Commit message: `feat: add OpenRouter Gemini VQA transport` (created in this task).

## Concerns

- All verification is offline by design. No credentials were read and no network request was made, so live provider behavior remains covered by the injected-opener contract only.
