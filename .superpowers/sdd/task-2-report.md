# Task 2 report: generalized evidence and paired scoring

## Outcome

Implemented Task 2 on top of commit `8ca7e90` while preserving the existing
three-video and NIST/Seamless behavior.

## Implementation

- Made the bin-midpoint timestamp policy an explicit method of
  `EvidenceParameters` and used it for both cache creation and validation.
  The default remains 12 frames, and the established 12-frame timestamps are
  unchanged. Frame count remains part of the serialized cache identity, so
  12-frame and 24-frame evidence cannot collide.
- Added deterministic regeneration of scoring reports from saved JSON arrays
  and append-only JSONL attempts.
- Retained the existing successful-attempt rule: a later failure cannot replace
  a successful prediction.
- Added aggregate, per-condition, and per-stratum completion and mean metrics
  for normalized exact match, token F1, and externally supplied METEOR.
- Added paired treatment-minus-baseline metric deltas, normalized-exact-match
  wins/losses, per-stratum paired summaries, and deterministic seeded bootstrap
  intervals explicitly labeled exploratory.
- Kept existing `evaluate_local_answer`, NIST semantic parity, NIST invocation,
  and scorer isolation interfaces unchanged.
- Validated every saved runner request, including superseded retry attempts,
  before references are joined. Reference, correct/incorrect-answer, and
  scoring fields cause regeneration to fail.

## TDD evidence

### Initial RED

Command:

```text
uv run pytest -q tests/test_vqa_benchmark.py tests/test_vqa_evaluation.py
```

Exact result:

```text
==================================== ERRORS ====================================
________________ ERROR collecting tests/test_vqa_evaluation.py _________________
ImportError while importing test module '/home/npatta01/.codex/worktrees/c32dfd7e-252f-44e8-b46c-854369072c00/trec_video_2026/tests/test_vqa_evaluation.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
../../../../.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_vqa_evaluation.py:10: in <module>
    from trec_video_2026.vqa_evaluation import (
E   ImportError: cannot import name 'evaluate_saved_predictions' from 'trec_video_2026.vqa_evaluation' (/home/npatta01/.codex/worktrees/c32dfd7e-252f-44e8-b46c-854369072c00/trec_video_2026/src/trec_video_2026/vqa_evaluation.py)
=========================== short test summary info ============================
ERROR tests/test_vqa_evaluation.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.10s
```

This failed because the required saved-prediction and paired-summary APIs did
not exist.

### First implementation check

Relevant failure excerpt:

```text
...............................F............                             [100%]
=================================== FAILURES ===================================
________ test_saved_predictions_regenerate_aggregate_and_stratum_scores ________

E       AssertionError: assert 'violin' == 'dog'
E         - dog
E         + violin

tests/test_vqa_evaluation.py:170: AssertionError
=========================== short test summary info ============================
FAILED tests/test_vqa_evaluation.py::test_saved_predictions_regenerate_aggregate_and_stratum_scores
1 failed, 43 passed in 1.58s
```

The implementation was sorting scored examples. It was corrected to preserve
the saved artifact's order for traceability.

### First GREEN

```text
............................................                             [100%]
44 passed in 1.71s
```

### Self-review RED for append-only artifacts

Command:

```text
uv run pytest -q tests/test_vqa_evaluation.py -k saved_jsonl
```

Relevant failure excerpt:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
________ test_saved_jsonl_attempts_keep_a_success_over_a_later_failure _________

E           ValueError: Saved predictions are missing or invalid JSON

src/trec_video_2026/vqa_evaluation.py:148: ValueError
=========================== short test summary info ============================
FAILED tests/test_vqa_evaluation.py::test_saved_jsonl_attempts_keep_a_success_over_a_later_failure
1 failed, 21 deselected in 0.05s
```

This proved that the first version did not yet accept the repository's
append-only JSONL attempt format.

### Final targeted GREEN

```text
.............................................                            [100%]
45 passed in 1.44s
```

## Final verification

Command:

```text
uv run pytest -q
```

Exact output:

```text
........................................................................ [ 28%]
........................................................................ [ 56%]
........................................................................ [ 84%]
.......................................                                  [100%]
255 passed in 2.24s
```

`git diff --check` also completed with no output.

## Files

- `src/trec_video_2026/vqa_benchmark.py`
- `src/trec_video_2026/vqa_evaluation.py`
- `tests/test_vqa_benchmark.py`
- `tests/test_vqa_evaluation.py`
- `.superpowers/sdd/task-2-report.md` (tracked Task 2 handoff artifact)

## Self-review

- Verified default 12-frame timestamps remain exactly the old values.
- Verified 12/24 policies produce distinct cache identities.
- Verified scoring preserves exact-match/token-F1 behavior and consumes METEOR
  values without changing the pinned NIST parity path.
- Verified scoring is deterministic across repeated regeneration.
- Verified paired metrics align by video ID rather than input position.
- Verified bootstrap resampling uses a local seeded RNG and records seed,
  confidence, iterations, and the exploratory label.
- Verified labels are separate inputs and cannot be present in nested saved
  runner requests.
- Verified unrelated untracked files and dataset contents were not modified.

## Concerns

- Bootstrap intervals are descriptive percentile intervals over a small panel,
  as required; they must not be presented as precise confidence bounds.
- METEOR regeneration expects the pinned parity path to have saved per-example
  METEOR values. Missing METEOR remains `None` and is never coerced to zero.

## Review fixes

Commit `524f466` received a focused correctness review. All four findings were
addressed test-first:

- The request-field denylist was replaced with an allowlist derived directly
  from the fields of the established `vqa_gate.RunnerRequest` dataclass.
- Every value is recursively traversed through sequences; any mapping below
  the request root is rejected because the canonical schema has no nested
  mapping field. This catches mappings inside nested lists as well as direct
  nested mappings.
- Unknown fields such as `ground_truth` and `evaluation_metrics` are rejected,
  while a complete request emitted by the established schema is accepted.
- `_is_valid_success` now defines success once as `status == "passed"` plus a
  nonblank string answer. Both retry selection and scoring use it, so a later
  malformed passed record cannot displace an earlier valid success.
- Raw/superseded attempts are each leakage-validated once; the redundant second
  validation of selected requests was removed.
- Paired summaries reject identical baseline and treatment condition IDs.

### Review RED

Command:

```text
uv run pytest -q tests/test_vqa_evaluation.py -k 'canonical_schema_recursively or established_schema or saved_jsonl or identical_condition'
```

Exact captured output summary:

```text
FFF.FF                                                                   [100%]
=================================== FAILURES ===================================
_ test_saved_runner_requests_enforce_canonical_schema_recursively[request_update0] _
E       Failed: DID NOT RAISE ValueError
_ test_saved_runner_requests_enforce_canonical_schema_recursively[request_update1] _
E       Failed: DID NOT RAISE ValueError
_ test_saved_runner_requests_enforce_canonical_schema_recursively[request_update2] _
E       Failed: DID NOT RAISE ValueError
________ test_saved_jsonl_attempts_keep_a_success_over_a_later_failure _________
E       assert None == 1.0
_____________ test_paired_summary_rejects_identical_condition_ids ______________
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'must be different'
E         Actual message: 'Paired conditions have no aligned completed examples'
=========================== short test summary info ============================
FAILED tests/test_vqa_evaluation.py::test_saved_runner_requests_enforce_canonical_schema_recursively[request_update0]
FAILED tests/test_vqa_evaluation.py::test_saved_runner_requests_enforce_canonical_schema_recursively[request_update1]
FAILED tests/test_vqa_evaluation.py::test_saved_runner_requests_enforce_canonical_schema_recursively[request_update2]
FAILED tests/test_vqa_evaluation.py::test_saved_jsonl_attempts_keep_a_success_over_a_later_failure
FAILED tests/test_vqa_evaluation.py::test_paired_summary_rejects_identical_condition_ids
5 failed, 1 passed, 21 deselected in 0.05s
```

The failures directly demonstrated the three leakage bypasses, malformed-pass
selection bug, and missing identical-condition guard.

### Review GREEN

Same focused command, exact output:

```text
......                                                                   [100%]
6 passed, 21 deselected in 0.03s
```

Complete Task 2 files:

```text
..................................................                       [100%]
50 passed in 1.72s
```

Full repository suite:

```text
........................................................................ [ 27%]
........................................................................ [ 55%]
........................................................................ [ 83%]
............................................                             [100%]
260 passed in 2.27s
```

### Review-fix self-review

- The allowlist is not duplicated: it is generated from the actual runner
  dataclass, so additions to the canonical top-level schema are visible here.
- The established full serialized request and the earlier minimal stored
  request fixtures both remain accepted because only unknown fields and nested
  mappings are rejected; evaluation fields cannot hide in either location.
- Sequence traversal excludes strings and bytes and descends through arbitrarily
  nested JSON lists/tuples.
- A valid success remains sticky over later failure, timeout, null-answer pass,
  or blank-answer pass records.
- No NIST parity, evidence-cache, three-video, or Seamless behavior changed.

## Second review fix: request/attempt identity

The second review identified that a label-free request could still carry a
`condition_id` or `video_id` different from its enclosing saved attempt. The
fix validates both identities on every raw attempt before retry selection.

Full evidence-bearing requests now also receive structural validation against
the established `RunnerRequest` contract: all dataclass fields are required,
JSON value types are checked, paths are converted to `Path`, and construction
applies the existing nonempty identity/question, evidence-length, and
nonnegative-timestamp invariants. Deliberately supported minimal legacy
requests may contain only `run_id`, `condition_id`, `video_id`, and `question`;
their present fields are type/value checked, and present identity fields must
match the enclosing attempt.

### Second-review RED

Command:

```text
uv run pytest -q tests/test_vqa_evaluation.py -k 'identity_must_match or superseded_jsonl_attempt_request_identity or canonical_runner_requests_validate'
```

Exact captured output summary:

```text
FFFFFF                                                                   [100%]
=================================== FAILURES ===================================
_ test_saved_runner_request_identity_must_match_enclosing_attempt[condition_id-other-condition] _
E       Failed: DID NOT RAISE ValueError
_ test_saved_runner_request_identity_must_match_enclosing_attempt[video_id-other-video] _
E       Failed: DID NOT RAISE ValueError
______ test_superseded_jsonl_attempt_request_identity_is_still_validated _______
E       Failed: DID NOT RAISE ValueError
_ test_full_canonical_runner_requests_validate_required_fields_and_types[missing_audio_path] _
E       Failed: DID NOT RAISE ValueError
_ test_full_canonical_runner_requests_validate_required_fields_and_types[string_timestamp] _
E       Failed: DID NOT RAISE ValueError
_ test_full_canonical_runner_requests_validate_required_fields_and_types[non_string_question] _
E       Failed: DID NOT RAISE ValueError
=========================== short test summary info ============================
FAILED tests/test_vqa_evaluation.py::test_saved_runner_request_identity_must_match_enclosing_attempt[condition_id-other-condition]
FAILED tests/test_vqa_evaluation.py::test_saved_runner_request_identity_must_match_enclosing_attempt[video_id-other-video]
FAILED tests/test_vqa_evaluation.py::test_superseded_jsonl_attempt_request_identity_is_still_validated
FAILED tests/test_vqa_evaluation.py::test_full_canonical_runner_requests_validate_required_fields_and_types[missing_audio_path]
FAILED tests/test_vqa_evaluation.py::test_full_canonical_runner_requests_validate_required_fields_and_types[string_timestamp]
FAILED tests/test_vqa_evaluation.py::test_full_canonical_runner_requests_validate_required_fields_and_types[non_string_question]
6 failed, 27 deselected in 0.07s
```

### Second-review GREEN

Same focused command:

```text
......                                                                   [100%]
6 passed, 27 deselected in 0.02s
```

Complete Task 2 files:

```text
........................................................                 [100%]
56 passed in 1.74s
```

Final full repository suite:

```text
........................................................................ [ 27%]
........................................................................ [ 54%]
........................................................................ [ 81%]
..................................................                       [100%]
266 passed in 2.39s
```

### Second-review self-review

- Identity validation occurs inside the raw-attempt loop, before selection or
  scoring, so a mismatched failed/superseded attempt cannot evade validation.
- Both `condition_id` and `video_id` mismatches have independent regressions.
- JSON array and nested request-in-JSONL artifact shapes are covered.
- Full canonical requests are validated with the actual `RunnerRequest`
  dataclass field set and constructor rather than an independent model schema.
- The earlier schema allowlist, recursive nested-mapping rejection,
  valid-success selection, paired-ID guard, and metric behavior remain intact.
