# Bounded Multi-Pass Video Counting Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, execute, authenticate, and score the twelve-row three-arm bounded multi-pass Gemini counting canary under a hard $30 provider-cost ceiling.

**Final status (2026-08-16):** Implementation and the v4 twelve-row execution
completed, but additive independent review classified the intended comparison
**Invalid**. Provider inputs were label-free, yet the runner process loaded the
private panel; the required small-target static crop condition did not execute;
and per-call returned model revision was not preserved. The observed 1/8
failure rescues, 1/4 controls preserved, semantic metrics, and `$2.048074` v4
cost remain factual outputs only. `validity-correction.json` in the tracked and
private bundles supersedes the original Diagnostic score. Post-run code
hardening is not experimental evidence, and no unchanged rerun is authorized.

**Architecture:** A small pure contract module owns typed CountSpec, evidence, coverage, refinement, and deterministic reduction. Separate preparation, provider runner, and scorer scripts reuse the repository's existing label firewall, canonical hashing, media binding, exclusive claims, budget reservations, write-once publication, and metric environment. The live runner executes a one-row smoke before the fixed arm and runs the bounded-refinement arm only after the fixed arm freezes.

**Tech Stack:** Python 3.12, frozen dataclasses, standard-library JSON/hash/path/file-lock handling, `google-genai==2.12.1`, ffmpeg/ffprobe, existing counting and submission modules, pytest 9, Ruff, existing metrics extra.

## Global Constraints

- Preserve every completed issue #34 artifact byte-for-byte; write only to a new canonical private benchmark root and a new sanitized tracked bundle.
- Keep references, baseline answers, correctness, failure/control roles, and post-label annotations unavailable until both treatment prediction freezes are terminal.
- Use only direct Google `gemini-3.1-pro-preview`; refuse silent model or provider substitution and automatic retries.
- Give every pass different evidence or a different role; never re-ask the same scalar-answer question over the same input.
- Maximum ten paid calls per row, one refinement per row, $3 smoke gate, $15 pre-refinement gate, and $30 total provider-cost ceiling.
- Exact results require complete authenticated evidence; malformed, incomplete, disputed, unsupported, or over-budget work abstains or freezes incomplete.
- QMD is the editable report source. Never hand-edit generated Markdown or HTML.

---

## File structure

- Create `src/trec_video_2026/vqa_bounded_multipass_counting.py`: pure schemas, parsers, coverage state machine, reducers, prompts, and serialization.
- Create `scripts/prepare_bounded_multipass_counting.py`: private deterministic panel selection, label-blind projection, media/frame/window preparation, and immutable preparation freeze.
- Create `scripts/run_bounded_multipass_counting.py`: direct-Google execution, exclusive claims, conservative budget, stage authentication, Arm B/Arm C freezes, and status CLI.
- Create `scripts/score_bounded_multipass_counting.py`: full-chain authentication, post-freeze reference join, exact/semantic metrics, gates, and sanitized summary.
- Create `tests/test_vqa_bounded_multipass_counting.py`: pure-contract behavior tests.
- Create `tests/test_prepare_bounded_multipass_counting.py`: selection, isolation, decoding, and preparation integration tests.
- Create `tests/test_run_bounded_multipass_counting.py`: request, concurrency, cost, freeze, and synthetic runner tests.
- Create `tests/test_score_bounded_multipass_counting.py`: mutation authentication, metrics, and decision tests.
- Modify `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`: pre-run or completed condition-bound claim.
- Create `docs/experiments/artifacts/vqa-bounded-multipass-counting-20260816/`: sanitized README, summary, and checksums after the run reaches a terminal state.
- Modify the generated instruction manifest rather than hand-editing `AGENTS.md` when the result changes the durable next-test pointer.

---

### Task 1: Pure CountSpec, evidence, coverage, and reducer contract

**Files:**
- Create: `src/trec_video_2026/vqa_bounded_multipass_counting.py`
- Create: `tests/test_vqa_bounded_multipass_counting.py`

**Interfaces:**
- Produces: `parse_count_spec(payload, *, question, duration_seconds) -> CountSpec | UnsupportedCountSpec`
- Produces: `parse_evidence(payload, *, spec, requested_views) -> EvidenceLedger`
- Produces: `parse_coverage_review(payload, *, spec, ledger) -> CoverageReview`
- Produces: `build_refinement_request(review, *, requested_views) -> RefinementRequest | None`
- Produces: `merge_refinement(ledger, refinement, review) -> EvidenceLedger`
- Produces: `reduce_count(spec, ledger, review) -> CountResult`
- Produces: prompt builders and `*_to_dict` serializers for those public types.

- [ ] **Step 1: Write CountSpec parser tests.** Add literal fixtures for action repetition, appearance episode, static cardinality, maximum simultaneous, and per-scene sum. Assert strict keys, question hash binding, duration bounds, nonempty observable target, permitted units, and typed `unsupported` for distinct cross-shot identity or visible-part ownership. Add recursive label-like key/value rejection.
- [ ] **Step 2: Run the parser tests and confirm RED.** Run `uv run pytest -q tests/test_vqa_bounded_multipass_counting.py -k count_spec`; expect import failure because the module does not exist.
- [ ] **Step 3: Implement the minimal CountSpec dataclasses/parser/serializer.** Reuse `validate_label_blind` and `canonical_sha256`; reject unknown fields and model scalar-answer fields.
- [ ] **Step 4: Run the parser tests and confirm GREEN.** Run the command from Step 2; expect all selected tests to pass.
- [ ] **Step 5: Write evidence parser tests.** Hand-construct endpoint-inclusive temporal views and static shot/frame views. Assert exact requested-view coverage, source-time bounds, stable IDs, nonempty provenance, normalized boxes, unique records, and rejection of overlaps that cannot be distinguished under the unit.
- [ ] **Step 6: Run the evidence tests and confirm RED.** Run `uv run pytest -q tests/test_vqa_bounded_multipass_counting.py -k evidence`; expect missing symbols or behavior failures.
- [ ] **Step 7: Implement the evidence types and parser.** Keep event and object observations separate and bind every record to one or more requested view IDs.
- [ ] **Step 8: Run the evidence tests and confirm GREEN.** Repeat Step 6.
- [ ] **Step 9: Write coverage/refinement state-machine tests.** Assert complete review, missing endpoint, uncovered window, disputed record, exactly one targeted refinement, no second refinement, and no scalar provisional answer in the coverage prompt.
- [ ] **Step 10: Run the coverage tests and confirm RED.** Run `uv run pytest -q tests/test_vqa_bounded_multipass_counting.py -k 'coverage or refinement'`.
- [ ] **Step 11: Implement coverage parsing and the one-refinement transition.** A review may reference only frozen view/record IDs and request one bounded interval, shot, or crop already inside source bounds.
- [ ] **Step 12: Run the coverage tests and confirm GREEN.** Repeat Step 10.
- [ ] **Step 13: Write reducer tests with literal answers.** Cover two separated action intervals, four appearance runs including both endpoints, maximum simultaneous two, per-scene `2+3=5`, static frame cardinality three, duplicate overlap abstention, unresolved challenge abstention, unsupported spec abstention, and empty evidence abstention.
- [ ] **Step 14: Run reducer tests and confirm RED.** Run `uv run pytest -q tests/test_vqa_bounded_multipass_counting.py -k reduce`.
- [ ] **Step 15: Implement deterministic reducers and traces.** Return exact only when coverage is complete, selected record/relation IDs are traceable, and the unit-specific invariants hold.
- [ ] **Step 16: Run the complete contract suite and quality checks.** Run `uv run pytest -q tests/test_vqa_bounded_multipass_counting.py`, `uv run ruff check src/trec_video_2026/vqa_bounded_multipass_counting.py tests/test_vqa_bounded_multipass_counting.py`, and `uv run ruff format --check src/trec_video_2026/vqa_bounded_multipass_counting.py tests/test_vqa_bounded_multipass_counting.py`.
- [ ] **Step 17: Commit Task 1.** Stage only the two Task 1 files and commit `feat: add bounded multi-pass counting contract`.

### Task 2: Deterministic private panel and evidence preparation

**Files:**
- Create: `scripts/prepare_bounded_multipass_counting.py`
- Create: `tests/test_prepare_bounded_multipass_counting.py`

**Interfaces:**
- Consumes: corrected census/panel freezes, frozen direct-answer result, canonical source videos, CountSpec route eligibility.
- Produces: `prepare_experiment(*, census_path: Path, routing_path: Path, excluded_ids_path: Path, source_root: Path, delivery_root: Path, output_root: Path) -> dict[str, object]`, private `panel.freeze.json`, `requests.label-blind.json`, and `preparation-complete.json`; per-row provider views are frozen later because they depend on the model-authored CountSpec.

- [ ] **Step 1: Write deterministic selection tests.** Use a synthetic census with more candidates than required and assert exactly four action failures, three static failures, one other supported temporal failure, two temporal controls, and two static controls selected by literal SHA ordering after exclusions. Assert duplicate IDs, missing strata, ambiguity, prior-canary reuse, and zero-row input fail closed.
- [ ] **Step 2: Run selection tests and confirm RED.** Run `uv run pytest -q tests/test_prepare_bounded_multipass_counting.py -k selection`; expect import failure.
- [ ] **Step 3: Implement selection and private/label-blind projections.** The inference projection must omit roles, references, prior predictions, correctness, expected counts, and annotations and pass recursive leakage validation.
- [ ] **Step 4: Run selection tests and confirm GREEN.** Repeat Step 2.
- [ ] **Step 5: Write media/view preparation tests.** With a generated short video fixture, assert source/delivery hashes, ffprobe duration/time base, endpoint-inclusive 7.5-fps windows using actual decoded PTS, deterministic overlap, per-shot native-resolution frames, crop membership, and no idealized timestamp substitution.
- [ ] **Step 6: Run media tests and confirm RED.** Run `uv run pytest -q tests/test_prepare_bounded_multipass_counting.py -k 'media or view or pts'`.
- [ ] **Step 7: Implement ffprobe/ffmpeg preparation and manifests.** Use lossless or explicitly recorded delivery conversion, write images privately, bind all bytes and decoder commands, and reject maximum observed gaps beyond the frozen limit.
- [ ] **Step 8: Run media tests and confirm GREEN.** Repeat Step 6.
- [ ] **Step 9: Write publication tests.** Assert duplicate-key rejection, exclusive preparation lock, self-hashes, full source closure, write-once conflict refusal, and byte-identical replay.
- [ ] **Step 10: Run publication tests and confirm RED.** Run `uv run pytest -q tests/test_prepare_bounded_multipass_counting.py -k 'lock or hash or replay or write_once'`.
- [ ] **Step 11: Implement immutable preparation publication.** Stage all files, verify them, and publish `preparation-complete.json` last.
- [ ] **Step 12: Run all Task 2 tests and quality checks.** Run `uv run pytest -q tests/test_prepare_bounded_multipass_counting.py`, `uv run ruff check scripts/prepare_bounded_multipass_counting.py tests/test_prepare_bounded_multipass_counting.py`, `uv run ruff format --check scripts/prepare_bounded_multipass_counting.py tests/test_prepare_bounded_multipass_counting.py`, and `git diff --check`.
- [ ] **Step 13: Commit Task 2.** Commit `feat: prepare bounded counting canary`.

### Task 3: Direct-Google runner, claims, budget, and arm freezes

**Files:**
- Create: `scripts/run_bounded_multipass_counting.py`
- Create: `tests/test_run_bounded_multipass_counting.py`

**Interfaces:**
- Consumes: Task 2 preparation terminal and Task 1 prompts/parsers/reducers.
- Produces: `preflight`, `run-smoke`, `run-fixed`, `run-refinement`, `freeze`, and `status` CLI commands with write-once row/stage terminals and arm prediction freezes.

- [ ] **Step 1: Write request-construction tests.** Assert exact model ID, temperature, thinking, structured schema, media resolution, full-video 4-fps CountSpec input, 7.5-fps clipped temporal views, image batches for static views, source-time offsets, and absence of label/baseline/scalar-answer fields.
- [ ] **Step 2: Run request tests and confirm RED.** Run `uv run --extra gemini pytest -q tests/test_run_bounded_multipass_counting.py -k request`.
- [ ] **Step 3: Implement request construction and response normalization.** Reuse hardened media/upload bindings from `run_gemini_counting_recovery.py`; keep the provider call itself injected for offline testing.
- [ ] **Step 4: Run request tests and confirm GREEN.** Repeat Step 2.
- [ ] **Step 5: Write exclusive-claim and budget tests.** Use concurrent subprocesses to prove one upload and one paid stage, exact request/media binding, valid self-hashes, settled usage, no retry, $3 smoke refusal, $15 pre-refinement refusal, and $30 hard refusal using worst-case remaining exposure.
- [ ] **Step 6: Run claim/budget tests and confirm RED.** Run `uv run --extra gemini pytest -q tests/test_run_bounded_multipass_counting.py -k 'claim or concurrent or budget or retry'`.
- [ ] **Step 7: Implement claims and conservative budget reservations.** Claims use `O_EXCL`, remove inherited self-hash fields before rehashing, bind upload records, and reserve before any provider call.
- [ ] **Step 8: Run claim/budget tests and confirm GREEN.** Repeat Step 6.
- [ ] **Step 9: Write synthetic end-to-end state tests.** Assert valid exact, valid abstain, malformed JSON, provider failure, CountSpec unsupported, missing stage, Arm B complete, Arm C no-op, Arm C one refinement, forbidden second refinement, and arm freeze identity set.
- [ ] **Step 10: Run state tests and confirm RED.** Run `uv run --extra gemini pytest -q tests/test_run_bounded_multipass_counting.py -k 'end_to_end or freeze or refinement or stage'`.
- [ ] **Step 11: Implement sequential runner and freezes.** Authenticate every predecessor before continuing; require the exact twelve-row identity set for final arm freezes; publish prediction freezes only after all reachable terminals exist.
- [ ] **Step 12: Run all Task 3 tests and quality checks.** Run `uv run --extra gemini pytest -q tests/test_run_bounded_multipass_counting.py`, `uv run --extra gemini ruff check scripts/run_bounded_multipass_counting.py tests/test_run_bounded_multipass_counting.py`, `uv run --extra gemini ruff format --check scripts/run_bounded_multipass_counting.py tests/test_run_bounded_multipass_counting.py`, and `git diff --check`.
- [ ] **Step 13: Commit Task 3.** Commit `feat: run bounded multi-pass counting canary`.

### Task 4: Full-chain scorer and predeclared decisions

**Files:**
- Create: `scripts/score_bounded_multipass_counting.py`
- Create: `tests/test_score_bounded_multipass_counting.py`

**Interfaces:**
- Consumes: authenticated preparation, Arm B/Arm C freezes, original baseline freeze, and post-freeze references.
- Produces: private score, inspection queue, decision, metric provenance, and sanitized aggregate summary.

- [ ] **Step 1: Write authentication mutation tests.** Start with a complete synthetic chain, then independently mutate preparation, upload, request, raw response, parsed response, view, stage terminal, row terminal, arm freeze, baseline freeze, source hash, model ID, and result. Every mutation must fail before scoring.
- [ ] **Step 2: Run authentication tests and confirm RED.** Run `uv run --extra metrics pytest -q tests/test_score_bounded_multipass_counting.py -k authenticate`.
- [ ] **Step 3: Implement full-chain authentication.** Require every stage dictated by the arm state, recompute every self/file hash, exact row identity, video ID, condition, result hash, source closure, and terminal reference.
- [ ] **Step 4: Run authentication tests and confirm GREEN.** Repeat Step 2.
- [ ] **Step 5: Write literal metric/gate tests.** Cover `5/8 + 4/4 + three-row gain` pass, each fixed-arm failure condition, two-row refinement gain, refinement regression, twice-cost rejection, incomplete denominator, and route-stratified summaries. Assert METEOR/BERTScore/STS/composite inputs are frozen concise answers only.
- [ ] **Step 6: Run metric tests and confirm RED.** Run `uv run --extra metrics pytest -q tests/test_score_bounded_multipass_counting.py -k 'metric or gate or decision'`.
- [ ] **Step 7: Implement scoring, decisions, and provenance.** Keep evidence validity and decision disposition separate and produce no quality metrics for incomplete or invalid comparisons.
- [ ] **Step 8: Run all Task 4 tests and quality checks.** Run `uv run --extra metrics pytest -q tests/test_score_bounded_multipass_counting.py`, `uv run --extra metrics ruff check scripts/score_bounded_multipass_counting.py tests/test_score_bounded_multipass_counting.py`, `uv run --extra metrics ruff format --check scripts/score_bounded_multipass_counting.py tests/test_score_bounded_multipass_counting.py`, and `git diff --check`.
- [ ] **Step 9: Commit Task 4.** Commit `feat: score bounded multi-pass counting canary`.

### Task 5: Offline preparation and cheapest live smoke

**Files:**
- Create privately: `data/vqa/benchmarks/direct-gemini-bounded-multipass-counting-v1/`
- Modify only if a pre-run record is needed: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`

**Interfaces:**
- Consumes: Tasks 1–4 and immutable prior artifacts.
- Produces: terminal preparation and, after provider authorization checks, the representative smoke terminal.

- [ ] **Step 1: Run all offline focused and regression tests.** Run `uv run --extra gemini --extra metrics pytest -q tests/test_vqa_bounded_multipass_counting.py tests/test_prepare_bounded_multipass_counting.py tests/test_run_bounded_multipass_counting.py tests/test_score_bounded_multipass_counting.py tests/test_vqa_counting_evidence.py tests/test_vqa_count_query.py tests/test_vqa_blind_frame_counting.py tests/test_vqa_gemini_counting_recovery.py tests/test_run_gemini_counting_recovery.py tests/test_score_gemini_counting_recovery.py tests/test_prepare_local_gemma_frame_counting_canary.py tests/test_vqa_local_gemma_canary_freeze.py tests/test_vqa_submission.py`, then run `uv run ruff check src/trec_video_2026/vqa_bounded_multipass_counting.py scripts/prepare_bounded_multipass_counting.py scripts/run_bounded_multipass_counting.py scripts/score_bounded_multipass_counting.py tests/test_vqa_bounded_multipass_counting.py tests/test_prepare_bounded_multipass_counting.py tests/test_run_bounded_multipass_counting.py tests/test_score_bounded_multipass_counting.py` and the matching `uv run ruff format --check` command.
- [ ] **Step 2: Prepare the private twelve-row panel.** Verify `data` resolves to the canonical dataset, run the preparation CLI once, then independently verify all media, view, source, and self-hashes.
- [ ] **Step 3: Inspect the label-blind request projection.** Search recursively for references, answers, correctness, roles, expected counts, prior predictions, annotations, and leaked numeric labels; stop on any match that is not the original question text or media metadata.
- [ ] **Step 4: Run provider preflight without inference.** Verify credential availability, exact model ID, pricing input, API schema compatibility, and estimated maximum exposure without printing credentials.
- [ ] **Step 5: Run the one-row Arm B smoke.** Enforce the $3 smoke ceiling and no retry. Preserve exact success, abstention, provider failure, invalid, incomplete, or budget-stop terminal.
- [ ] **Step 6: Authenticate the smoke independently.** Do not score against the reference. Continue only if the evidence chain is structurally valid, the terminal is exact-or-abstain, and cost is at most $3.
- [ ] **Step 7: Commit any label-independent code correction only after a failing regression test.** Never alter the frozen smoke. If a correction changes the condition, invalidate the smoke and require new explicit run authorization.

### Task 6: Complete the blind arms, score, inspect, and report

**Files:**
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Create: `docs/experiments/artifacts/vqa-bounded-multipass-counting-20260816/README.md`
- Create: `docs/experiments/artifacts/vqa-bounded-multipass-counting-20260816/summary.json`
- Create: `docs/experiments/artifacts/vqa-bounded-multipass-counting-20260816/SHA256SUMS`
- Modify: the applicable generated instruction manifest and render `AGENTS.md`

**Interfaces:**
- Consumes: authenticated preparation and smoke.
- Produces: terminal Arm B/Arm C freezes, private scores/inspection, sanitized evidence, rendered report, and issue #34 update.

- [ ] **Step 1: Complete Arm B.** Run sequentially, stop on a structural failure or if settled/projected exposure crosses the gates, and freeze the exact twelve-row identity set or explicit incomplete denominator.
- [ ] **Step 2: Complete Arm C when authorized by the Arm B terminal and budget.** Reuse fixed evidence, permit only one targeted refinement, and freeze all reachable rows.
- [ ] **Step 3: Score post-freeze.** Authenticate the entire chain before loading references or baseline answers; compute exact, rescue/control, semantic, cost, and gate results.
- [ ] **Step 4: Inspect every rescue, regression, and abstention manually.** Record perception omission, split, merge, scope, CountSpec, coverage, and convention mechanisms without changing predictions or scores.
- [ ] **Step 5: Write the condition-bound report and sanitized bundle.** Separate observed result, validity, limitations, and decision; preserve any negative, incomplete, invalid, or corrected outcome.
- [ ] **Step 6: Render QMD-derived Markdown/HTML and generated instructions.** Use isolated Quarto cache/output checks, never hand-edit derived files, and verify no relative report assets are missing.
- [ ] **Step 7: Verify privacy and integrity.** Run checksum verification, secret/private-data scans, link checks, `git diff --check`, focused/full tests proportional to changes, and independent provenance replay.
- [ ] **Step 8: Commit the terminal result.** Stage only source, tests, sanitized artifacts, QMD, generated reports/instructions, and checksums; never stage private benchmark data.
- [ ] **Step 9: Post a concise issue #34 result.** Include completion denominator, exact rescue/control rates, Arm B/C comparison, validity, metrics, cost, decision, and evidence link; do not publish private rows or raw provider data.

---

## Plan self-review

- Every design requirement maps to a task: label isolation and panel boundary
  (Task 2), different evidence and bounded refinement (Tasks 1 and 3), cost and
  exclusivity (Task 3), complete authentication and gates (Task 4), cheapest
  smoke (Task 5), and durable negative-or-positive reporting (Task 6).
- The pure contract is independent of provider execution; the provider runner
  can be tested with complete response doubles while keeping filesystem,
  hashing, parsing, state transitions, and publication real.
- No task requires changing an immutable prior artifact or exposing labels
  before treatment freezes.
- The planned route set deliberately excludes distinct cross-shot identity and
  part ownership rather than pretending orchestration solves them.
- The live-run steps occur only after all test-first implementation and
  offline integrity checks pass.
