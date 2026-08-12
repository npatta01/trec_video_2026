# Task 5 report

Status: complete.

Commit: `dd69da496eb8b1967a75370c91250d7104c90e2f` — `feat: add frontier phase zero evidence audit`

Tests: `PYTHONPATH=src uv run pytest tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_audit.py -q` — 6 passed in 8.27s; Ruff and `git diff --check` passed.

Concerns: broader regression suites were not run in this handoff; no provider, network, credentials, labels, references, ASR, or audio-model calls were used.

## Fix round 1/5 — native evidence binding

Status: complete; commit subject `fix: bind phase zero audit evidence`.

RED/GREEN evidence: regressions first reproduced deleted/replaced/symlinked JPEG acceptance, missing event/citation records, incomplete/contradicted usable pilot rulings, JSON-only runner authorization, symlinked source ancestry, and rebased nonzero ffmpeg PTS. Each targeted regression passed after its minimal implementation change; focused audit verification finished at 21 passed.

Technical rulings:

- The immutable audit artifact now binds a sorted evidence-media file count and digest; validation and pilot authorization rehash all overview sheets, six preserved overview frames, event keyframes, and answer-citation frames from disk.
- Media must be regular files with no symlink component, use audit-root-relative `rows/<video_id>/...` paths, retain source-media and ffmpeg provenance, and exactly cover map keyframes and answer citation points.
- ffmpeg extraction preserves source-timeline PTS with explicit requested, clamped, actual, seek, and fallback provenance; decoded PTS must remain within 0.5 seconds of the intended clamped timestamp.
- `usable_for_screen` is limited to complete pilot coverage and complete event checks without contradictions. `revise_contract`/`stop`, screen contradictions, and manually non-critical `unverifiable` observations remain recordable.

Verification: `PYTHONPATH=src uv run pytest -q tests/test_vqa_frontier_phase0_audit.py tests/test_build_vqa_frontier_phase0_audit.py tests/test_vqa_frontier_phase0_runner.py tests/test_run_vqa_frontier_phase0_script.py tests/test_build_vqa_frontier_map_audit.py` — 114 passed in 27.45s. Ruff and `git diff --check` passed.

Concerns: none within this fix-round scope. No provider, network, credentials, labels, references, ASR, or audio-model calls were used.
