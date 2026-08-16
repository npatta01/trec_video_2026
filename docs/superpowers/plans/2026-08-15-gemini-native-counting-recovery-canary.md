# Gemini Native-Video Counting Recovery Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a provenance-bound, three-row Gemini native-video
canary that locates counting evidence, verifies a timestamp-clipped high-FPS
subset, challenges the ledger over the complete video, and deterministically
returns an exact count or abstention.

**Architecture:** A focused pure module owns schemas, prompts, interval
normalization, atomic-ledger validation, challenge reconciliation, and exact or
abstaining result construction. A thin direct-Google runner owns file upload,
native-video metadata, exclusive paid-call claims, immutable artifacts, and a
conservative $1 budget ledger. A separate post-freeze scorer is the only code
allowed to load references or earlier predictions.

**Tech Stack:** Python 3.12, pytest, google-genai 2.12.1, Gemini 3.1 Pro Preview,
native WebM video, NLTK METEOR, BERTScore, frozen UMBC STS cache, Quarto, GitHub.

## Global Constraints

- Use exactly three developmental rows: two known temporal failures and one
  correct temporal control.
- Never expose references, expected counts, previous predictions, correctness,
  or panel roles before the final-result freeze.
- Use direct Google Gemini only; never silently fall back to OpenRouter.
- Pass 1 uses complete native video at 4 FPS; Pass 2 uses the canonical
  timestamp envelope at 12 FPS; Pass 3 uses complete native video at 12 FPS.
- Allow at most one additive verification request per row and no automatic
  retry of a paid provider error.
- Stop after the first row if cumulative cost exceeds $0.34, after two rows if
  it exceeds $0.67, and always before projected or observed total cost could
  exceed $1.00.
- Preserve complete private artifacts under the canonical benchmark root and
  only sanitized aggregates in Git.
- Do not modify or replay the frozen 475-row, generic enumerator, or count-query
  v1/v2 artifacts.
- Keep issue #35 and issue #36 out of scope.

---

### Task 1: Pure multi-pass counting contract

**Files:**

- Create: `src/trec_video_2026/vqa_gemini_counting_recovery.py`
- Create: `tests/test_vqa_gemini_counting_recovery.py`

**Interfaces:**

- Consumes: label-blind row mappings with `video_id`, `question`, and
  `duration_seconds`.
- Produces: `build_locator_prompt(row) -> str`,
  `parse_locator(payload, duration_seconds) -> LocatorResult`,
  `verification_envelope(locator, duration_seconds, padding_seconds=1.5) ->
  TimeRange`, `build_verifier_prompt(...) -> str`,
  `parse_atomic_ledger(payload, ...) -> AtomicLedger`,
  `build_challenger_prompt(...) -> str`,
  `parse_challenge(payload, ...) -> ChallengeResult`, and
  `reduce_recovery_result(...) -> RecoveryResult`.

- [ ] **Step 1: Write strict locator and interval RED tests**

```python
def test_verification_envelope_unions_candidates_and_clips_padding():
    locator = parse_locator(
        {
            "schema_version": 1,
            "count_unit": "appearance_episode",
            "target": "ball",
            "candidates": [
                {"candidate_id": "C1", "start_seconds": 0.2, "end_seconds": 0.8,
                 "reason": "first appearance"},
                {"candidate_id": "C2", "start_seconds": 6.5, "end_seconds": 7.0,
                 "reason": "last appearance"},
            ],
            "reviewed_spans": [{"start_seconds": 0.0, "end_seconds": 7.268}],
            "unresolved_spans": [],
        },
        duration_seconds=7.268,
    )
    assert verification_envelope(locator, 7.268) == TimeRange(0.0, 7.268)
```

- [ ] **Step 2: Run locator tests and verify RED**

Run: `uv run pytest -q tests/test_vqa_gemini_counting_recovery.py -k 'locator or envelope'`

Expected: import or symbol failure because the module does not exist.

- [ ] **Step 3: Implement immutable types, strict parsing, and prompts**

```python
@dataclass(frozen=True)
class TimeRange:
    start_seconds: float
    end_seconds: float

def verification_envelope(locator: LocatorResult, duration_seconds: float,
                          padding_seconds: float = 1.5) -> TimeRange:
    if not locator.candidates:
        return TimeRange(0.0, duration_seconds)
    return TimeRange(
        max(0.0, min(item.start_seconds for item in locator.candidates) - padding_seconds),
        min(duration_seconds, max(item.end_seconds for item in locator.candidates) + padding_seconds),
    )
```

Reject unknown fields, duplicate IDs, booleans-as-numbers, non-finite values,
out-of-range spans, inverted spans, label-bearing keys, and incomplete
full-duration partitions.

- [ ] **Step 4: Write atomic-ledger and challenge RED tests**

Cover stable instance IDs; ordered start/peak/end timestamps;
before/during/after observations; predicate support; ambiguity; duplicate,
split, mismatch, accept, and additional-range challenge dispositions; and
absolute source timestamps inside the verifier envelope.

- [ ] **Step 5: Run the new tests and verify RED**

Run: `uv run pytest -q tests/test_vqa_gemini_counting_recovery.py -k 'ledger or challenge or reduce'`

Expected: failures for missing parsers and reducer.

- [ ] **Step 6: Implement deterministic reconciliation**

```python
def reduce_recovery_result(ledger: AtomicLedger, challenge: ChallengeResult,
                           additive: AtomicLedger | None) -> RecoveryResult:
    unresolved = challenge.unresolved_ids or challenge.requested_ranges
    accepted = reconcile_instances(ledger, challenge, additive)
    if unresolved:
        return RecoveryResult.abstain(accepted, reason="unresolved_challenge")
    return RecoveryResult.exact(accepted, value=len(accepted))
```

The reducer must never trust a model-supplied final integer and must abstain on
unresolved coverage, ambiguous instances, identity conflicts, or incomplete
additive verification.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
uv run pytest -q tests/test_vqa_gemini_counting_recovery.py
uv run ruff check src/trec_video_2026/vqa_gemini_counting_recovery.py tests/test_vqa_gemini_counting_recovery.py
uv run ruff format --check src/trec_video_2026/vqa_gemini_counting_recovery.py tests/test_vqa_gemini_counting_recovery.py
```

Commit: `feat: add Gemini counting recovery contract`

### Task 2: Direct-Google runner, media metadata, freezes, and budget

**Files:**

- Create: `scripts/run_gemini_counting_recovery.py`
- Create: `tests/test_run_gemini_counting_recovery.py`
- Reuse without modification: `src/trec_video_2026/vqa_submission.py`
- Reuse patterns from: `scripts/run_gemini_counting_evidence.py`

**Interfaces:**

- Consumes: the pure Task 1 functions, a private label-blind request manifest,
  verified WebM delivery files, frozen direct-Google pricing, and
  `GEMINI_API_KEY`.
- Produces: `prepare`, `run-row`, `run-canary`, and `status` CLI commands plus
  immutable per-stage `request.json`, `raw-response.json`, `parsed.json`,
  `complete.json`, and final `predictions-freeze.json` artifacts.

- [ ] **Step 1: Write native-video metadata RED tests**

```python
def test_pass_video_parts_use_frozen_fps_and_offsets():
    locator = build_video_part(file_uri="files/v1", mime_type="video/webm",
                               fps=4.0, time_range=None)
    verifier = build_video_part(file_uri="files/v1", mime_type="video/webm",
                                fps=12.0, time_range=TimeRange(1.5, 6.25))
    assert locator.video_metadata.fps == 4.0
    assert verifier.video_metadata.fps == 12.0
    assert verifier.video_metadata.start_offset == "1.500s"
    assert verifier.video_metadata.end_offset == "6.250s"
```

Also assert high media resolution, one native-video part per request, prompt
placement, model/temperature/seed/thinking configuration, and exact source-time
bounds in verifier prompts.

- [ ] **Step 2: Run metadata tests and verify RED**

Run: `uv run --extra gemini pytest -q tests/test_run_gemini_counting_recovery.py -k metadata`

Expected: script import failure.

- [ ] **Step 3: Implement provider request construction and preparation**

Preparation must select the frozen private row IDs without serializing panel
roles or references into inference manifests, bind original-source and delivery
hashes, validate the existing WebM stream-copy provenance, and write a
self-hashed request manifest under
`data/vqa/benchmarks/direct-gemini-counting-recovery-v1/`.

- [ ] **Step 4: Write claim, freeze, and no-retry RED tests**

Use a fake client to assert that concurrent or repeated invocations cannot
claim the same stage, raw provider output is written before parsing, a provider
exception becomes a terminal failed artifact, and the runner makes no second
provider call.

- [ ] **Step 5: Implement exclusive stage claims and write-once artifacts**

```python
def reserve_paid_stage(stage_root: Path, request_identity: Mapping[str, object]) -> Path:
    claim = stage_root / "paid-call-claim.json"
    descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(self_hashed(request_identity), handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return claim
```

Every `complete.json` must bind the request, raw response, parsed payload,
provider model/version, usage, estimated cost, and preceding-stage complete
hash. Rerunning an identical completed command is read-only; a conflicting
artifact raises.

- [ ] **Step 6: Write hard-budget RED tests**

Test atomic cross-process reservations; maximum current-or-observed per-call
projection; `$0.34`, `$0.67`, and `$1.00` stage gates; no release of a paid
claim after an attempted call; and refusal before provider access when the
worst-case remaining path cannot fit.

- [ ] **Step 7: Implement sequential orchestration and budget ledger**

The runner performs locator → verifier → challenger → optional additive
verifier. The challenger may cause at most one additive call. `run-canary`
runs the appearance row first, checks its terminal cost/integrity gate, then
runs the action row and control only while their proportional and global gates
remain satisfied.

- [ ] **Step 8: Run fake-provider end-to-end tests and commit**

Run:

```bash
uv run --extra gemini pytest -q tests/test_run_gemini_counting_recovery.py
uv run --extra gemini ruff check scripts/run_gemini_counting_recovery.py tests/test_run_gemini_counting_recovery.py
uv run --extra gemini ruff format --check scripts/run_gemini_counting_recovery.py tests/test_run_gemini_counting_recovery.py
```

Commit: `feat: add cost-gated Gemini counting recovery runner`

### Task 3: Post-freeze scorer and inspection packet

**Files:**

- Create: `scripts/score_gemini_counting_recovery.py`
- Create: `tests/test_score_gemini_counting_recovery.py`
- Reuse metrics from: `scripts/score_gemini_counting_evidence.py`

**Interfaces:**

- Consumes: authenticated final prediction freeze, private panel with
  references, frozen 475-row baseline scores, local BERT model snapshot, and
  frozen STS cache.
- Produces: private `score.json`, `metrics.json`, `inspection.md`, and a
  sanitized summary projection.

- [ ] **Step 1: Write authentication and isolation RED tests**

Construct valid and mutated fixture chains. Assert the scorer recomputes every
self-hash and file hash, binds all three stage completions per row, refuses a
post-freeze result edit, verifies exact panel identity, and is the first path
that can accept `reference_answer`, `panel_role`, or `status`.

- [ ] **Step 2: Run scorer tests and verify RED**

Run: `uv run --extra metrics pytest -q tests/test_score_gemini_counting_recovery.py`

Expected: script import failure.

- [ ] **Step 3: Implement exact-count and semantic scoring**

```python
rescue_rate = rescued_failures / 2
control_rate = preserved_controls / 1
gate_passed = (
    rescued_failures == 2
    and preserved_controls == 1
    and invalid_ledgers == 0
    and unresolved_exact_rows == 0
    and total_cost_usd <= 1.0
)
```

Record METEOR, BERTScore-F1, STS, semantic composite, treatment-minus-baseline
deltas, exact/abstain outcomes, rescue/control denominators, timestamp-ledger
validity, challenge additions, failure reasons, model version, cost, metric
package versions, BERT snapshot hash, and STS provenance.

- [ ] **Step 4: Implement private inspection and sanitized projection**

The private packet links every row's source media, locator candidates,
verification envelope, atomic ledger, challenge, optional additive ledger,
final result, and post-freeze reference. The sanitized projection contains no
video IDs, questions, answers, predictions, row-level raw text, or media paths.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
uv run --extra metrics pytest -q tests/test_score_gemini_counting_recovery.py
uv run --extra metrics ruff check scripts/score_gemini_counting_recovery.py tests/test_score_gemini_counting_recovery.py
uv run --extra metrics ruff format --check scripts/score_gemini_counting_recovery.py tests/test_score_gemini_counting_recovery.py
```

Commit: `feat: score Gemini counting recovery canary`

### Task 4: Preflight and execute the paid canary

**Files:**

- Create privately: `data/vqa/benchmarks/direct-gemini-counting-recovery-v1/`
- Do not modify: existing issue #34 private roots.

**Interfaces:**

- Consumes: Tasks 1–3, authorized $1 ceiling, local credential, frozen pricing,
  and three existing verified delivery videos.
- Produces: immutable private provider and score artifacts.

- [ ] **Step 1: Run all zero-cost verification before provider access**

Run focused tests, `ruff`, formatting, `git diff --check`, dataset-symlink
validation, source/delivery hash checks, recursive secret scan, request-manifest
label scan, credential-presence check, model-name check, and a complete fake
provider dry run.

- [ ] **Step 2: Freeze pricing and preparation**

Record official direct-Google pricing URL, access time, model ID, per-token
rates, estimate formula, source hashes, request hashes, git revision, Python,
SDK, FFmpeg, and platform versions. Verify the private root is new or contains
only byte-identical preparation artifacts.

- [ ] **Step 3: Run the first-row canary**

Execute the appearance-episode row only. Preserve any provider/schema failure
as terminal. Authenticate the artifact chain and confirm cumulative cost is at
most `$0.34` before proceeding.

- [ ] **Step 4: Run remaining rows only if authorized gates remain open**

Execute the repeated-action failure, enforce the `$0.67` cumulative gate, then
the control. Stop before any next call whose reserved worst-case cost could
cross `$1.00`.

- [ ] **Step 5: Freeze predictions, unlock references, score, and inspect**

Authenticate all artifacts before loading references. Freeze metrics and
inspect every accepted/rejected atomic instance against timestamped media.
Preserve a valid negative, incomplete, or budget-stopped result without repair
calls.

### Task 5: Durable report, verification, and GitHub handoff

**Files:**

- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Create: `docs/experiments/artifacts/vqa-gemini-counting-recovery-20260815/`
- Modify generated instruction source and render: `AGENTS.md`
- Modify generated QMD outputs only through Quarto rendering.

**Interfaces:**

- Consumes: authenticated Task 4 results and sanitized projection.
- Produces: canonical claim update, tracked evidence index, rendered report,
  issue #34 comment, and focused PR update.

- [ ] **Step 1: Record the condition-bound result**

Report reason, panel, treatment/control, completion denominator, observed
rescue/control rates, exact/abstain counts, evidence validity, decision
disposition, METEOR/BERTScore-F1/STS/composite, cost, limitations, what the
result rules out, and the next decision. State explicitly that the rows were
developmental and known to the experiment designer.

- [ ] **Step 2: Build and authenticate the sanitized bundle**

Include `README.md`, `summary.json`, schema documentation, source/private-root
hash pointers, and `SHA256SUMS`. Run the repository privacy scanner and verify
that no private row identifiers, questions, references, raw responses, media,
credentials, or absolute private paths are tracked.

- [ ] **Step 3: Render and verify report surfaces**

Render from QMD, verify links, responsive/self-contained HTML, tailnet-only
portal mapping if updated, and tracked hashes. Do not hand-edit generated HTML
or Markdown.

- [ ] **Step 4: Run final verification**

Run focused tests, relevant regression tests, full pytest, ruff, format check,
`git diff --check`, artifact validators, source/provenance hash checks, and
secret/privacy scans. Distinguish any base-reproducing failures from new
regressions with executable evidence.

- [ ] **Step 5: Review, commit, push, and update issue #34**

Request an independent substantive review, address blockers, commit only
focused files, push the existing issue #34 branch, update its draft PR, and
post a concise issue comment with the result, cost, validity, disposition, and
evidence links.
