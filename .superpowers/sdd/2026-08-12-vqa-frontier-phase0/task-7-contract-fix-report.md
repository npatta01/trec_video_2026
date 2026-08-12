# Phase 0 contract fix

## Outcome

- Bumped `NEUTRAL_MAP_PROMPT_VERSION` and `ANSWER_PROMPT_VERSION` to v2.
- Replaced serialized local JSON Schema text with concise root-key and field-semantic guidance.
- Both prompts now require a JSON object instance only, forbid Markdown/schema/meta output, state that the provider schema is separately enforced, and end with the task-specific AV imperative.
- Full schemas, validators, provider projections, runner/policy/provider files, and concurrent edits were left unchanged.

## TDD and verification

- RED: `PYTHONPATH=src pytest -q tests/test_vqa_frontier_phase0_contract.py` failed 3 prompt-contract assertions against the v1/schema prompts.
- GREEN: `PYTHONPATH=src pytest -q tests/test_vqa_frontier_phase0_contract.py` — 25 passed.
- Regression: `PYTHONPATH=src pytest -q tests/test_vqa_frontier_phase0_contract.py tests/test_vqa_frontier_map.py` — 40 passed.
- `ruff check src/trec_video_2026/vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_contract.py` — passed.
- `git diff --check -- src/trec_video_2026/vqa_frontier_phase0_contract.py tests/test_vqa_frontier_phase0_contract.py` — passed.

## Concern

Plain `pytest` does not find the source package in this checkout; verification used `PYTHONPATH=src`. No network, credential, or dataset reads were performed.
