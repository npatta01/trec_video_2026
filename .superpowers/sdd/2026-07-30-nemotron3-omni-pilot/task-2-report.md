# Task 2 report: expanded freeze verification and blind grading

## Outcome

Implemented the explicit `expanded-pilot` grading path while retaining the
existing six-row pilot API and behavior. The expanded path derives its exact
12-row identity, category, and role order from `EXPANDED_ROWS` in
`nemotron3_omni_pilot.py`.

## Changed files

- `src/trec_video_2026/vqa_experiments/nemotron3_omni_grading.py`
  - Added `ExpandedPilotScore`, expanded freeze verification, expanded
    label-free blind-packet construction, and expanded scoring.
  - Validates files, SHA256SUMS, manifest identity, frozen status, 24/24
    completion, manifest-to-predictions SHA-256 binding, and all 24 records in
    the exact 12 adjacent `audio-off`/`audio-on` pairs.
  - Loads references only after those checks complete. Packets retain only
    question, candidate answer, reference answer, and opaque item IDs.
  - Reports the two approved gates: sound-target improvements,
    non-control improvements/regressions, visual-control regressions,
    `sound_replication_passed`, and `broader_expansion_passed`.
- `tests/test_nemotron3_omni_grading.py`
  - Added expanded synthetic freeze fixtures and tests for exact ordering,
    wrong denominator, reordered pairs, changed predictions, changed manifest
    bindings, reference isolation, independent shuffle, packet leakage, and
    both scoring gates.

## RED/GREEN evidence

RED (with `PYTHONPATH=src`, because the shell-level `pytest` command did not
include the repository source path):

```
4 failed, 12 passed in 0.06s
```

The four failures were the intended missing expanded builder/scorer API.
The unprefixed `pytest -q tests/test_nemotron3_omni_grading.py` first failed
at collection with `ModuleNotFoundError: trec_video_2026`; that is an
environment-path issue, not a test result.

GREEN / final verification:

```
.venv/bin/pytest -q tests/test_nemotron3_omni_grading.py
16 passed in 0.05s
```

Also passed:

```
git diff --check
.venv/bin/python -m compileall -q src/trec_video_2026/vqa_experiments/nemotron3_omni_grading.py
```

## Commit

Implementation commit: `ae8f897c7f04c7ff322593b091ea5dac8c4c22a6`
(`Add expanded Nemotron blind grading`).

## Self-review

- Exact expanded membership is not copied into grading code; the verifier and
  panel validator consume the pilot module's ordered `EXPANDED_ROWS`.
- The verifier returns before invoking the reference loader for bad files,
  SHA256SUMS, manifest values, completion denominator, manifest binding, or
  pair identity/order.
- Both independent packets contain the same 24 records in different orders;
  no row/video/category/role/condition linkage or CoT field is present in the
  public packet item schema.
- Existing six-row grading tests remain green, including the original 12/12
  denominator and adjudication behavior.

## Concerns

- The expanded blind-packet assembly intentionally parallels the existing
  six-row builder to avoid changing its public API/default behavior. A future
  shared internal helper could reduce that duplication, but should be done
  with explicit compatibility coverage rather than folded into this scoped
  change.
