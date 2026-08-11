# Task 2 report — label-free 35+2 panel boundary and freeze CLI

Date: 2026-08-11

Scope completed:
- Implemented `src/trec_video_2026/vqa_frontier_map_panel.py`
- Implemented `scripts/build_vqa_frontier_map_panel.py`
- Corrected and finalized `tests/test_vqa_frontier_map_panel.py`

RED evidence:
- Ran `uv run pytest tests/test_vqa_frontier_map_panel.py -q`
- Observed the expected collection failure:
  - `ModuleNotFoundError: No module named 'trec_video_2026.vqa_frontier_map_panel'`
- To honor the brief and TDD, I treated the unexpected untracked preexisting panel module/script as non-authoritative leftovers and moved them out of the worktree before the RED run.

Test fidelity corrections before implementation:
- Added an explicit distinct-preflight test.
- Added a duration-coverage test for short/typical/long selection against the category candidate distribution.
- Expanded the candidate-sheet test to assert topology/media identity fields while keeping the label-free boundary checks.
- Narrowed an over-broad `"reference"` substring assertion so `reference_policy` remains legal.
- Fixed the duration-coverage fixture so it preserves topology diversity and isolates the intended failure.

Implementation summary:
- `classify_question(question)` now uses precedence-ordered, word-boundary classification without falsely matching `heart` as `sound` or generic `after` as `temporal`.
- `build_candidate_sheet(data_root)` projects each annotation immediately to `(video_id, question)`, never serializes annotation labels, and emits only label-free candidate metadata plus media/source identities.
- `validate_panel_manifest(payload, data_root)` enforces:
  - the frozen schema/run/policy header,
  - exact 35 evaluation rows and 2 preflight rows,
  - five evaluation rows per category,
  - distinct preflight rows,
  - recursive rejection of forbidden label/reference fields,
  - current-byte validation for media hashes, sizes, durations, and audio/video flags,
  - minimum topology diversity per category,
  - short/typical/long duration coverage per category relative to the candidate distribution.
- `scripts/build_vqa_frontier_map_panel.py` provides `candidates` and `freeze` subcommands, with `freeze` refusing symlinked output and overwrite.
- Canonical JSON writing is atomic and refuses overwrite.

Verification:
- `uv run pytest tests/test_vqa_frontier_map_panel.py -q`
  - `11 passed in 0.09s`
- `uv run ruff check src/trec_video_2026/vqa_frontier_map_panel.py scripts/build_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_panel.py`
  - `All checks passed!`

Deferred by brief:
- Did not create `configs/vqa/frontier_video_map_panel_35.json`.
- Did not inspect or run against real annotations, answers/references, media beyond synthetic temporary fixtures, credentials, or APIs.

## Fix round 1 — review findings

Date: 2026-08-11

Added regression coverage for:
- exact `<= 30.0` second enforcement on selected evaluation and preflight rows,
- rejection of selected rows above `30.0` seconds,
- preflight `video_id` distinctness from other preflight rows and all evaluation rows regardless of question,
- canonical dataset-root verification via a synthetic, testable resolver boundary,
- exact allowed candidate row key surface with all `annotation_*` fields removed.

RED evidence:
- Ran `uv run pytest tests/test_vqa_frontier_map_panel.py -q`
- Observed the expected failure before implementation:
  - `ImportError: cannot import name 'verify_canonical_data_root' from 'trec_video_2026.vqa_frontier_map_panel'`

Implementation changes:
- Added `verify_canonical_data_root(...)` and called it before both `candidates` and `freeze` CLI paths.
- Enforced `MAX_SELECTED_DURATION_SECONDS = 30.0` for every selected row.
- Strengthened preflight distinctness to operate on `video_id`, not `(video_id, question)`.
- Removed `annotation_filename`, `annotation_sha256`, `annotation_size_bytes`, `has_audio`, and `has_video` from candidate rows so the surface stays question/category/duration/topology/media-identity only.
- Kept manifest validation on selected rows’ `has_audio` and `has_video` unchanged.

GREEN evidence:
- Ran `uv run pytest tests/test_vqa_frontier_map_panel.py -q`
  - `17 passed in 0.13s`
- Ran `uv run ruff check src/trec_video_2026/vqa_frontier_map_panel.py scripts/build_vqa_frontier_map_panel.py tests/test_vqa_frontier_map_panel.py`
  - `All checks passed!`
