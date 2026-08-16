# Blind Structured Frame Counting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a zero-network interface through which a future model can label explicitly supplied frames and receive either a deterministic visible-episode count or a conservative abstention.

**Architecture:** Add one focused domain module beside the existing frame-presence reducer. It strictly parses the supported count unit, binds decoded frames to presentation artifacts, validates exact model-output coverage, chooses a mechanically complete challenge set, reconciles decisions, and emits a hash-bound coverage certificate. A small CLI exercises the same public interfaces using synthetic JSON only.

**Tech Stack:** Python 3.12, frozen dataclasses, standard-library JSON/hash/path handling, existing `vqa_submission.canonical_sha256`, existing `vqa_counting_evidence.validate_label_blind`, existing `vqa_count_frame_presence` reducer, pytest, Ruff.

## Global Constraints

- No model, provider, GPU, network, or paid call.
- Support only `visible_appearance_episode`; every other unit returns typed `unsupported`.
- No reference, label, prior prediction, correctness, expected count, or panel-role field may enter a pre-inference payload.
- Code owns decoded-frame coverage, PTS, hashes, counting, and abstention.
- The model emits only one enum state per required frame; it never emits ranges or a final count.
- The unblinded four-appearance row is not scored or used as a model-quality fixture.
- Frozen issue #34 artifacts remain untouched.
- No issue #35 or #36 scope.

---

## File structure

- Create `src/trec_video_2026/vqa_blind_frame_counting.py`: all pure schemas, parsers, request construction, decision validation, challenge selection, reconciliation, and certificate construction.
- Create `tests/test_vqa_blind_frame_counting.py`: public-interface unit and integration tests using synthetic frame inventories only.
- Create `scripts/prepare_blind_frame_counting.py`: zero-network JSON dry-run CLI with authenticated, write-once output.
- Create `tests/test_prepare_blind_frame_counting.py`: CLI and byte-identical replay tests.
- Modify `docs/superpowers/specs/2026-08-16-blind-structured-frame-counting-design.md` only if implementation reveals a real contradiction; do not broaden the design.

### Task 1: Supported CountSpec and frame request

**Files:**
- Create: `src/trec_video_2026/vqa_blind_frame_counting.py`
- Create: `tests/test_vqa_blind_frame_counting.py`

**Interfaces:**
- Consumes: JSON-like mappings plus existing `canonical_sha256()` and `validate_label_blind()`.
- Produces: `parse_visible_episode_count_spec(payload)`, `build_frame_decision_request(spec, inventory, presentations)`, `VisibleEpisodeCountSpec`, `UnsupportedCountSpec`, and `FrameDecisionRequest`.

- [ ] **Step 1: Write a failing public-seam test for the one supported unit**

```python
def test_supported_count_spec_and_endpoint_complete_request_are_hash_bound() -> None:
    parsed = parse_visible_episode_count_spec(visible_episode_spec())
    assert isinstance(parsed, VisibleEpisodeCountSpec)

    request = build_frame_decision_request(
        parsed,
        inventory=three_frame_inventory(),
        presentations=one_sheet_presentations(),
    )

    assert [frame.frame_id for frame in request.frames] == ["F0", "F1", "F2"]
    assert request.frames[0].is_first_decoded_frame
    assert request.frames[-1].is_last_decoded_frame
    assert request.maximum_observed_gap_seconds == 0.5
    assert len(request.complete_sha256) == 64
```

- [ ] **Step 2: Run the exact test and verify RED**

Run:

```bash
uv run pytest -q tests/test_vqa_blind_frame_counting.py::test_supported_count_spec_and_endpoint_complete_request_are_hash_bound
```

Expected: collection failure because `vqa_blind_frame_counting` does not exist.

- [ ] **Step 3: Implement the strict public schemas and builder**

Implement frozen dataclasses with these exact fields:

```python
@dataclass(frozen=True)
class VisibleEpisodeCountSpec:
    spec_id: str
    target: str
    unit: str
    scope: str
    episode_rule: str
    maximum_sample_gap_seconds: float
    every_decoded_frame: bool
    cut_policy: str
    occluded_identity: str
    complete_sha256: str

@dataclass(frozen=True)
class UnsupportedCountSpec:
    spec_id: str
    unit: str
    outcome: str
    reason_codes: tuple[str, ...]
    complete_sha256: str

@dataclass(frozen=True)
class RequestFrame:
    frame_id: str
    decoded_frame_index: int
    timestamp_seconds: float
    artifact_sha256: str
    is_first_decoded_frame: bool
    is_last_decoded_frame: bool

@dataclass(frozen=True)
class PresentationBinding:
    presentation_id: str
    presentation_sha256: str
    frame_ids: tuple[str, ...]
    mode: str

@dataclass(frozen=True)
class FrameDecisionRequest:
    spec: VisibleEpisodeCountSpec
    video_id: str
    source_sha256: str
    duration_seconds: float
    decoded_frame_count: int
    frames: tuple[RequestFrame, ...]
    presentations: tuple[PresentationBinding, ...]
    maximum_observed_gap_seconds: float
    complete_sha256: str
```

`parse_visible_episode_count_spec()` must recursively call
`validate_label_blind()`, reject unknown/missing fields, validate the two cut
policies, require `occluded_identity == "unsupported"`, and return
`UnsupportedCountSpec` for any well-formed unit other than
`visible_appearance_episode`.

`build_frame_decision_request()` must reject unordered/duplicate frame IDs and
indices, missing decoded index 0 or `decoded_frame_count - 1`, false endpoint
timestamps, out-of-range PTS, gaps above the spec maximum, presentation modes
outside `numbered_contact_sheet`, `individual_frame`, and `centered_triplet`,
unbound frames, duplicate presentation membership, and mismatched hashes.

- [ ] **Step 4: Run the tracer test and verify GREEN**

Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Add one red-green slice per request invariant**

Add parametrized public tests that independently mutate:

```python
[
    "reference_answer",
    "missing_first_frame",
    "missing_last_frame",
    "duplicate_frame_id",
    "duplicate_decoded_index",
    "non_monotonic_pts",
    "gap_above_limit",
    "unbound_frame",
    "duplicate_presentation_membership",
    "unknown_presentation_mode",
]
```

For each case, first run the isolated test and observe the intended failure,
then add only the corresponding validation and rerun it to green.

- [ ] **Step 6: Verify Task 1**

```bash
uv run pytest -q tests/test_vqa_blind_frame_counting.py
uv run ruff check src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
uv run ruff format --check src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
```

- [ ] **Step 7: Commit Task 1**

```bash
git add src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
git commit -m "feat: bind blind frame counting requests"
```

### Task 2: Exact decisions, challenge selection, and reconciliation

**Files:**
- Modify: `src/trec_video_2026/vqa_blind_frame_counting.py`
- Modify: `tests/test_vqa_blind_frame_counting.py`

**Interfaces:**
- Consumes: `FrameDecisionRequest` from Task 1 and JSON-like model decision payloads.
- Produces: `parse_frame_decisions(request, payload)`, `select_challenge_frames(request, decisions)`, `reconcile_frame_decisions(request, primary, challenge)`, `FrameDecisionSet`, and an existing `FramePresenceLedger`.

- [ ] **Step 1: Write and run a failing exact-coverage response test**

```python
def test_decisions_must_cover_every_required_frame_exactly_once() -> None:
    request = valid_request()
    decisions = parse_frame_decisions(
        request,
        decision_payload([("F0", "present"), ("F1", "absent"), ("F2", "present")]),
    )
    assert decisions.request_complete_sha256 == request.complete_sha256
    assert [item.frame_id for item in decisions.decisions] == ["F0", "F1", "F2"]
```

Run the isolated test. Expected: import failure for `parse_frame_decisions`.

- [ ] **Step 2: Implement minimal strict decision parsing and verify GREEN**

Add:

```python
@dataclass(frozen=True)
class FrameDecision:
    frame_id: str
    target_state: str

@dataclass(frozen=True)
class FrameDecisionSet:
    pass_id: str
    request_complete_sha256: str
    presentation_mode: str
    decisions: tuple[FrameDecision, ...]
    complete_sha256: str
```

Require strict top-level and row fields, exact request hash, allowed states from
`vqa_count_frame_presence.TARGET_STATES`, and exact set/order equality with the
request frame IDs. Recursively reject label-bearing keys.

- [ ] **Step 3: Add red-green decision rejection slices**

Add isolated tests for missing, duplicate, extra, reordered, and unknown-state
decisions; a mutated request hash; a label-bearing nested field; and prose or a
final integer appearing as an unknown field. Each must be rejected before
reconciliation.

- [ ] **Step 4: Write and run the failing challenger-selection test**

```python
def test_small_canary_challenges_every_frame() -> None:
    request = valid_request(frame_count=100)
    decisions = alternating_decisions(request)
    assert select_challenge_frames(request, decisions) == tuple(
        frame.frame_id for frame in request.frames
    )
```

Expected: failure because the selector does not exist.

- [ ] **Step 5: Implement deterministic challenge selection and verify GREEN**

For `len(request.frames) <= 100`, return every frame ID. For larger requests,
return ordered unique IDs consisting of endpoints, both sides of every state
transition, one available neighbor on each side, and exactly three evenly
spaced interior indices for every maximal state run. If a run has fewer than
three interior frames, include all of them.

- [ ] **Step 6: Write and run failing reconciliation tests**

```python
def test_reconciliation_turns_disagreement_into_uncertain_and_reducer_abstains() -> None:
    request = valid_request()
    primary = decisions_for(request, ["present", "absent", "present"])
    challenge = decisions_for(request, ["present", "present", "present"], pass_id="challenge")
    ledger = reconcile_frame_decisions(request, primary, challenge)
    result = reduce_presence_episodes(ledger)
    assert ledger.frames[1].target_state == "uncertain"
    assert result.outcome == "abstain"
```

Expected: failure because reconciliation does not exist.

- [ ] **Step 7: Implement conservative reconciliation and verify GREEN**

Require `primary.pass_id == "primary"`, `challenge.pass_id == "challenge"`,
different presentation modes, exact request bindings, and challenge coverage of
every selected ID. Preserve a primary state only when the challenge agrees;
otherwise emit `uncertain`. For unchallenged IDs allowed only in requests above
100 frames, preserve the primary state. Construct a `FramePresenceLedger` from
the authenticated request fields and reconciled states.

- [ ] **Step 8: Add cut-policy and unsupported semantics tests**

The initial reducer has no cut inventory. A spec with `split_at_cuts` must
therefore return an abstaining reconciliation reason until a future request
schema explicitly binds cut boundaries. Do not silently apply ordinary
contiguous-run semantics.

- [ ] **Step 9: Verify and commit Task 2**

```bash
uv run pytest -q tests/test_vqa_blind_frame_counting.py
uv run ruff check src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
uv run ruff format --check src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
git add src/trec_video_2026/vqa_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py
git commit -m "feat: reconcile blind frame decisions"
```

### Task 3: Coverage certificate and authenticated dry-run

**Files:**
- Modify: `src/trec_video_2026/vqa_blind_frame_counting.py`
- Modify: `tests/test_vqa_blind_frame_counting.py`
- Create: `scripts/prepare_blind_frame_counting.py`
- Create: `tests/test_prepare_blind_frame_counting.py`

**Interfaces:**
- Consumes: authenticated spec/request/primary/challenge/ledger plus `PresenceCountResult`.
- Produces: `certify_frame_presence_result(...) -> FramePresenceCoverageCertificate` and CLI files `request.freeze.json`, `ledger.freeze.json`, `result.freeze.json`, and `certificate.freeze.json`.

- [ ] **Step 1: Write and run a failing certificate test**

```python
def test_certificate_binds_complete_chain_and_exact_result() -> None:
    chain = valid_reconciled_chain()
    result = reduce_presence_episodes(chain.ledger)
    certificate = certify_frame_presence_result(
        spec=chain.spec,
        request=chain.request,
        primary=chain.primary,
        challenge=chain.challenge,
        ledger=chain.ledger,
        result=result,
        reducer_source_sha256="f" * 64,
        source_revision="abc1234",
    )
    assert certificate.outcome == "exact"
    assert certificate.value == 2
    assert certificate.reason_codes == ()
    assert len(certificate.complete_sha256) == 64
```

Expected: failure because the certificate function does not exist.

- [ ] **Step 2: Implement the certificate and verify GREEN**

Add a frozen `FramePresenceCoverageCertificate` binding the canonical hashes
of the spec, request, both decision passes, serialized ledger, serialized
reducer result, reducer source, and source revision. Recompute all hashes from
the supplied objects. Emit exact only when the reducer is exact and all chain
bindings validate; otherwise emit abstain with stable reason codes. Integrity
mutation raises rather than becoming an ordinary abstention.

- [ ] **Step 3: Add red-green mutation and abstention tests**

Cover request mutation, response hash mutation, missing challenge decision,
same presentation mode across passes, uncertain reconciliation, excessive
sampling gap, and unsupported cut policy. Assert that integrity errors raise
and valid insufficiency returns abstention.

- [ ] **Step 4: Write a failing zero-network CLI integration test**

```python
def test_cli_freezes_byte_identical_authenticated_chain(tmp_path: Path) -> None:
    inputs = write_synthetic_inputs(tmp_path)
    first = run_cli(inputs, tmp_path / "out")
    before = snapshot_bytes(tmp_path / "out")
    second = run_cli(inputs, tmp_path / "out")
    assert first.returncode == second.returncode == 0
    assert snapshot_bytes(tmp_path / "out") == before
    assert json.loads((tmp_path / "out/certificate.freeze.json").read_text())["outcome"] == "exact"
```

Expected: failure because the script does not exist.

- [ ] **Step 5: Implement the CLI and verify GREEN**

The CLI accepts exactly:

```text
--spec SPEC.json
--inventory INVENTORY.json
--presentations PRESENTATIONS.json
--primary PRIMARY.json
--challenge CHALLENGE.json
--output-dir DIR
```

It performs no imports from `google`, no network access, and no environment
credential reads. It calls only the public module interfaces, serializes
dataclasses with explicit `*_to_dict()` functions, hashes the current reducer
source, records `git rev-parse HEAD`, and writes each canonical artifact with a
temporary file plus `os.replace`. Existing identical files are accepted;
conflicting files raise `FileExistsError`.

- [ ] **Step 6: Add CLI failure tests**

Assert nonzero exit and no certificate for missing decisions, label-bearing
input, conflicting existing output, and hash mutation. Assert the output JSON
contains no keys matching reference, label, answer, correctness, score,
metric, expected, or panel role.

- [ ] **Step 7: Verify and commit Task 3**

```bash
uv run pytest -q tests/test_vqa_blind_frame_counting.py tests/test_prepare_blind_frame_counting.py
uv run ruff check src/trec_video_2026/vqa_blind_frame_counting.py scripts/prepare_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py tests/test_prepare_blind_frame_counting.py
uv run ruff format --check src/trec_video_2026/vqa_blind_frame_counting.py scripts/prepare_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py tests/test_prepare_blind_frame_counting.py
git add src/trec_video_2026/vqa_blind_frame_counting.py scripts/prepare_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py tests/test_prepare_blind_frame_counting.py
git commit -m "feat: certify blind frame counting dry runs"
```

### Task 4: Focused verification and handoff

**Files:**
- Modify only if needed: `docs/superpowers/specs/2026-08-16-blind-structured-frame-counting-design.md`
- Modify: PR #39 description and issue #34 comment after local verification.

**Interfaces:**
- Consumes: complete Task 1–3 implementation.
- Produces: verified branch state and concise, condition-bound handoff; no experiment result claim.

- [ ] **Step 1: Run focused and adjacent tests**

```bash
uv run pytest -q \
  tests/test_vqa_blind_frame_counting.py \
  tests/test_prepare_blind_frame_counting.py \
  tests/test_vqa_count_frame_presence.py \
  tests/test_vqa_count_query.py \
  tests/test_run_gemini_counting_recovery.py \
  tests/test_score_gemini_counting_recovery.py
```

- [ ] **Step 2: Run static and formatting checks**

```bash
uv run ruff check \
  src/trec_video_2026/vqa_blind_frame_counting.py \
  scripts/prepare_blind_frame_counting.py \
  tests/test_vqa_blind_frame_counting.py \
  tests/test_prepare_blind_frame_counting.py
uv run ruff format --check \
  src/trec_video_2026/vqa_blind_frame_counting.py \
  scripts/prepare_blind_frame_counting.py \
  tests/test_vqa_blind_frame_counting.py \
  tests/test_prepare_blind_frame_counting.py
git diff --check
```

- [ ] **Step 3: Run the full suite with the repository's secure umask**

```bash
umask 0022
uv run pytest -q
```

Any failure must be reproduced on the pre-feature commit before it can be
classified as pre-existing.

- [ ] **Step 4: Independently review the diff**

Request one consequential code review focused on label isolation, exact
one-to-one coverage, abstention, hash authentication, write-once behavior, and
accidental provider/network paths. Resolve every P1/P2 finding and rerun the
focused checks.

- [ ] **Step 5: Commit any review corrections**

```bash
git add src/trec_video_2026/vqa_blind_frame_counting.py scripts/prepare_blind_frame_counting.py tests/test_vqa_blind_frame_counting.py tests/test_prepare_blind_frame_counting.py docs/superpowers/specs/2026-08-16-blind-structured-frame-counting-design.md
git commit -m "fix: harden blind frame counting preparation"
```

Skip this commit if review requires no changes.

- [ ] **Step 6: Push and update the existing focused PR and issue**

Report this exactly as zero-cost infrastructure/protocol completion. Do not
claim model rescue, counting accuracy, or promotion. State that fresh inference
still requires authorization and the 5-failure/3-control/2-sentinel blind gate.

## Plan self-review

- Spec coverage: every supported interface, failure boundary, certificate
  binding, synthetic verification requirement, and future gate maps to a task.
- Placeholder scan: no deferred behavior or unspecified implementation step
  remains.
- Type consistency: Tasks 2 and 3 consume the exact dataclasses and function
  names produced by Tasks 1 and 2.
- Scope check: the plan implements one model-facing visible-episode seam and
  does not include inference, detectors, tracking, or other count units.
