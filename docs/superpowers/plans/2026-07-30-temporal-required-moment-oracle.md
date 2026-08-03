# Temporal Required-Moment Oracle Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Production behavior follows `superpowers:test-driven-development`; every task
> uses checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether missing visual moments materially limit the pinned
Qwen3.5-4B temporal baseline on the frozen ten-row 2025 development panel,
without running or starting any transcript experiment.

**Architecture:** Add one tracked, pure-Python temporal-bottleneck module for
annotation validation, reviewer agreement, decoded-frame planning, selector
recall, request identity, prediction freezing, and paired decision logic.
Panel questions, media, annotations, extracted frames, predictions, references,
grades, and logs remain in a new ignored private run directory. Execute a
strict CPU-only coverage gate before any local ROCm inference; if the gate
fails, preserve and report that result and stop.

**Tech Stack:** Python 3.12, pytest, stdlib JSON/hash/dataclasses/statistics,
`ffprobe`/`ffmpeg`, rootless Podman, pinned ROCm image, Qwen3.5-4B, Quarto.

## Completion Record — 2026-07-30

- **Outcome:** stopped at the preregistered primary annotation-agreement gate.
  Structural agreement was `8/10` and passed its threshold; median tIoU was
  `0.4791666666666665` and failed the required `0.50`. The frozen decision is
  `stop_before_inference`.
- **Downstream status:** adjudication, decoded-frame extraction, selector
  recall, GPU inference, grading, and transcript work were not run.
- **Evidence roots:** agreement SHA-256
  `d4d94bb2f149ed3314130d7bd6f6bbab60583ab9e43a10c294227a998d87a1ea`;
  stop-decision SHA-256
  `0ab6bdaa76a8b433d5d240983e7f97ff80636885488477db82421b8d5e4450cf`;
  freeze-record SHA-256
  `83e78f15afa38cf84cba0ccb11cb5363424ebc4b5dc88b353d6e8b09d881f416`.
- **Verification:** `809` tests pass; changed Python files pass Ruff check and
  format; QMD-derived Markdown and the full HTML site match clean isolated
  regeneration; privacy/dependency scans pass; independent final review
  passes with no remaining finding; live HTTPS content matches the rendered
  artifact and Tailscale reports `tailnet only`.
- **Next action:** share this visual checkpoint and wait for explicit user
  direction. A fresh visual retry should calibrate the rubric and collect new
  blind annotations before reapplying the same CPU agreement gate.

## Global Constraints

- The primary theme is **Visual and Temporal Evidence**; Routing is related.
- Use only the frozen ten-row temporal panel from labeled 2025 training data.
  Make no claim about prevalence, 2026 accuracy, or general temporal accuracy.
- Keep `/home/npatta01/data/competitions/trec_video_2026/data` as the only
  physical dataset copy. Access it only through the ignored `data` symlink.
- Never copy, move, delete, stage, or commit dataset contents, private media,
  questions, annotations, answers, predictions, transcripts, frames, or logs.
- Create a new ignored run directory
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/`; never overwrite or
  modify `data/vqa/benchmarks/temporal-diagnostic-10/`.
- Do not run or prepare a transcript-grounding experiment. Stop and share the
  visual results with the user before any transcript work.
- Do not publish, expose a service, pull a large model/image, or use cloud GPU.
  Inference may use the existing local rootless Podman image and cached model
  only after the CPU gate passes.
- Pin model `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, runtime image
  `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`,
  seed `0`, greedy decoding, `max_new_tokens=32`, and per-frame processor bounds
  `256*256` through `512*512`.
- Both answer conditions use exactly 12 unique decoded frames and the same
  maximum-pixel proxy. The treatment is an oracle diagnostic, never a
  deployable selector or baseline candidate.
- Actual decoded-frame PTS, time base, frame index, requested time, frame hash,
  extraction command identity, prompt/source identity, request hash, and
  runtime identity must be recorded before selector recall or inference.
- References, prior predictions/grades, and new predictions/grades must remain
  unavailable to evidence annotators and inference. Required-moment metadata
  controls offline frame construction only and never enters the model prompt.
- QMD is the editable report source. Generated Markdown and HTML must not be
  hand-edited.

## Frozen Scientific Contract

- Annotation unit: one frozen video-question row.
- Eligibility values: `labelable_visual`, `not_visually_answerable`,
  `ambiguous_question`, or `insufficient_media`.
- Each labelable row has one or more alternative minimal sufficient evidence
  sets. A set contains one to six conjunctive closed intervals
  `[start_seconds, end_seconds]`, recorded to 0.1 seconds. Every interval in a
  set is required; any complete alternative set is sufficient.
- For temporal-order questions, intervals include every moment needed to
  establish the named anchor and the answer event.
- Two primary reviewers independently annotate all ten rows from the full
  video and question only. A third reviewer first annotates all rows de novo,
  freezes that file, then adjudicates all rows using the three frozen files.
- Before individual freeze, reviewers may not see references, answers,
  predictions, grades, condition names, selector timestamps, selected-frame
  contact sheets, other annotations, or Issue-8 row outcomes.
- Record reviewer pseudonym, role, rubric revision, packet SHA-256, start/end
  timestamps, and annotation SHA-256. The adjudication binds all three hashes.
- Primary agreement passes only if at least 8/10 rows have equal eligibility
  and the same preferred-set interval count, and median temporal IoU among
  eligible structurally matched rows is at least 0.50. Flag any eligibility or
  topology difference, tIoU below 0.50, or matched boundary difference above
  1.0 second. If agreement fails, stop and report; do not average labels.
- At least eight adjudicated rows must be `labelable_visual`.
- A decoded frame hits `[s,e]` when actual PTS lies in
  `[max(0,s-tolerance), min(duration,e+tolerance)]`. Report strict tolerance
  `0.0` and primary tolerance `0.5` seconds. A set is recalled only when every
  conjunctive interval has a hit; a row is recalled when any alternative set
  is recalled. Primary selector recall is macro row recall over eligible rows;
  also report component recall and exact missed-row count.
- Oracle-hybrid-12 uses six global and six oracle-local slots. Global desired
  times are six equal-bin midpoints. For the shortest-duration sufficient set
  (earliest-onset tie break), assign one local slot per interval, allocate the
  remaining local slots proportionally by Hamilton largest remainder, and use
  equal-sub-bin midpoints within each interval. Bind each desired time to the
  nearest unused decoded frame PTS, breaking equal-distance ties toward the
  earlier frame. Local slots must remain inside their interval. Sort all 12
  frames by actual PTS before inference. Any row without 12 unique frames is
  unconstructible and stops before inference.
- The CPU gate passes only if: at least 8/10 rows are eligible; agreement
  passes; both conditions are constructible for every eligible row;
  oracle-hybrid recall is 100% at tolerances 0.0 and 0.5; and relative to fresh
  uniform-12 it gains at least two recalled rows and loses zero at both
  tolerances. All upstream artifacts must be hashed and frozen first.
- If the CPU gate passes, inference requires complete paired outputs for every
  eligible row. Predictions freeze before references are loaded.
- Two independent answer graders receive an opaque-row, cryptographically
  randomized, condition-blind packet. A third grader adjudicates disagreements
  after both grade files freeze. Condition mapping stays separately frozen and
  withheld until adjudication freezes.
- Advance the **retrieval research branch**, not the oracle, only if paired
  `C=2/P=1/W=0` ordinal gain is at least 2, at least two rows improve, zero rows
  regress, and every CPU gate remains satisfied. Otherwise record **Not
  promoted under the tested condition**.

---

### Task 1: Annotation, agreement, and recall contract

**Status:** Complete.

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/temporal_bottleneck.py`
- Create: `tests/test_temporal_bottleneck.py`

**Interfaces:**
- Produces `validate_annotation(payload, panel, *, expected_role) -> dict`.
- Produces `annotation_agreement(left, right, panel) -> dict`.
- Produces `score_selector_recall(annotation, frames_by_video, *, tolerance_seconds) -> dict`.
- Produces constants for schema version, roles, eligibility values, forbidden
  keys, agreement thresholds, recall tolerances, and minimum eligible rows.

- [x] **Step 1: Write failing annotation-validation tests**

  Add literal fixtures for a two-row synthetic panel. Test acceptance of
  duration-bounded closed intervals and rejection of: missing panel rows,
  reordered IDs, duplicate IDs, empty sets, more than six intervals, negative
  or reversed bounds, values beyond duration, invalid eligibility, intervals
  on ineligible rows, mismatched packet hash/role, and forbidden keys including
  `answer`, `reference`, `prediction`, `grade`, `condition`, and
  `frame_timestamps` at any nesting depth. Name each test for the production
  break it catches.

- [x] **Step 2: Run the focused tests and verify RED**

  Run `uv run pytest tests/test_temporal_bottleneck.py -q` and confirm failures
  are caused by the missing module/interfaces.

- [x] **Step 3: Implement the minimal annotation validator**

  Canonicalize to sorted-key compact JSON for hashing. Return normalized rows
  in frozen panel order. Preserve reviewer/rubric/timing metadata, but do not
  synthesize missing historical fields.

- [x] **Step 4: Write failing agreement tests**

  Hand-check literal cases for identical spans (`tIoU=1.0`), disjoint spans
  (`0.0`), partial overlap, topology mismatch, eligibility mismatch, boundary
  drift above one second, the 8/10 structural threshold, and median-tIoU gate.

- [x] **Step 5: Run the focused agreement tests and verify RED**

  Confirm the new tests fail for missing behavior rather than malformed
  fixtures.

- [x] **Step 6: Implement agreement calculation**

  Compare the preferred sufficient set in each primary annotation. Calculate
  union duration without double-counting overlaps, intersection duration,
  per-row tIoU, structural agreement, flags, median/IQR, counts, and the exact
  pass/fail decision.

- [x] **Step 7: Write failing selector-recall tests**

  Use literal PTS lists to cover strict/0.5-second boundary behavior,
  conjunctive intervals, alternative sets, ineligible rows, component recall,
  row recall, missed IDs, and rejection of requested timestamps presented as
  actual PTS.

- [x] **Step 8: Implement selector recall and verify GREEN**

  Require frame records to contain `actual_pts_seconds`, `frame_index`,
  `time_base`, and `sha256`. Run the focused file and then `uv run pytest -q`.

- [x] **Step 9: Self-review and commit**

  Run `git diff --check`, inspect the diff for private data, and commit only the
  tracked module and tests with message
  `feat: add temporal evidence recall contract`.

### Task 2: Exact decoded-frame and matched-request planning

**Status:** Complete.

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/temporal_bottleneck.py`
- Modify: `tests/test_temporal_bottleneck.py`

**Interfaces:**
- Consumes normalized adjudicated annotations from Task 1.
- Produces `probe_frame_inventory(ffprobe_payload, *, media_sha256) -> list[dict]`.
- Produces `uniform_decoded_frame_plan(duration, inventory, count=12) -> list[dict]`.
- Produces `oracle_hybrid_decoded_frame_plan(duration, sufficient_sets, inventory) -> list[dict]`.
- Produces `build_temporal_requests(panel, condition_frames, inference_identity) -> list[dict]`.
- Produces `freeze_json_artifact(path, *, bindings) -> dict` and
  `verify_json_artifact(path, freeze_record) -> None`.

- [x] **Step 1: Write failing frame-inventory tests**

  Use literal ffprobe JSON with `best_effort_timestamp_time`, stream index,
  frame index, and time base. Reject missing/nonfinite/nonmonotonic PTS,
  duplicates, missing time base, and media-hash mismatch. Verify stable frame
  indices and six-decimal canonical PTS.

- [x] **Step 2: Verify RED, then implement the inventory parser**

  Run the focused tests before and after the minimal implementation.

- [x] **Step 3: Write failing uniform/oracle planning tests**

  Hand-check uniform midpoint binding, earlier-frame tie break, local frames
  staying inside intervals, 6/6 allocation, Hamilton allocation for intervals
  of unequal duration, shortest-set/earliest-onset selection, exact uniqueness,
  chronological output, global/local collision recovery, and unconstructible
  rows. Include a literal twelve-frame expected plan.

- [x] **Step 4: Verify RED, then implement both planners**

  Each plan record contains slot kind, desired/requested time, frame index,
  actual PTS, time base, and source media hash. Do not accept answer-bearing
  annotations or emit required spans into model requests.

- [x] **Step 5: Write failing request/freeze tests**

  Assert exact condition set `{uniform-12, oracle-hybrid-12}`, panel identity,
  12 unique frame hashes/PTS per request, identical inference configuration,
  null transcript/media fields, absence of annotation spans/answers, stable
  request hashes, no overwrite, binding verification, and tamper detection.

- [x] **Step 6: Verify RED, implement, then verify GREEN**

  Run the focused tests after each behavior, followed by `uv run pytest -q` and
  `git diff --check`.

- [x] **Step 7: Self-review and commit**

  Confirm no private paths or values entered tracked fixtures. Commit with
  message `feat: plan exact temporal oracle evidence`.

### Task 3: Freeze protocol, collect blinded annotations, and run CPU gate

**Status:** Stopped at the predeclared primary agreement gate. Steps 4–7 were
not run.

**Files:**
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/protocol.json`
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/annotation-packet.json`
- Create, ignored/private: three `annotations/reviewer-*.json` files and their
  SHA-256 sidecars
- Create, ignored/private: `annotations/adjudicated.json`, agreement/recall
  outputs, frame inventories/plans/images, request manifest, and freeze records
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/run_temporal_oracle.py`

**Interfaces:**
- Consumes Task 1/2 tracked APIs and the immutable prior ten-row panel identity.
- If primary agreement fails, produces a frozen
  `agreement-stop-decision.json` with `stop_before_inference` and does not
  create downstream artifacts. After a passing agreement and adjudication,
  produces a frozen `cpu-gate.json` with every remaining prerequisite and one
  decision: `run_inference` or `stop_before_inference`.

- [x] **Step 1: Write and hash the private protocol and blind packet**

  Copy only panel IDs, questions, full-media paths, durations, and media hashes
  from the prior manifest. Exclude anchors, selector timestamps, contact sheets,
  references, predictions, grades, and condition outcomes. Record all frozen
  constants from this plan and source/packet hashes.

- [x] **Step 2: Independently annotate and freeze three reviewer files**

  Dispatch three fresh reviewers with only the rubric and blind packet. Each
  inspects full video, records all ten rows, validates locally, and writes one
  separate private file. Freeze each before any annotation is shared.

- [x] **Step 3: Compute primary agreement**

  Validate reviewer A/B, write and hash agreement output, and apply the exact
  structural/tIoU gate. If it fails, emit `stop_before_inference` and skip to
  Task 6.

- [ ] **Step 4: Adjudicate and freeze required moments**

  Give the third reviewer all three frozen annotation files only after its de
  novo file exists. Validate and hash final adjudication with all input hashes.
  Stop if fewer than eight rows are labelable.

- [ ] **Step 5: Build actual-PTS inventories and extract frames**

  Run `ffprobe` on each private source video, hash raw probe output, construct
  both frame plans, extract exactly the planned frame indices with `ffmpeg`,
  and bind each image hash back to frame index/PTS/time base/requested time and
  extraction/source identity. Refuse overwrite or nonmatching frame counts.

- [ ] **Step 6: Score and freeze recall**

  Score both conditions at tolerances 0.0 and 0.5, write exact row/component
  counts and missed opaque IDs, and verify oracle construction/recall.

- [ ] **Step 7: Apply and freeze the CPU gate**

  Record each boolean prerequisite, artifact hash, eligible denominator,
  recall delta, loss count, and final decision. If the decision is
  `stop_before_inference`, do not inspect references or predictions and skip to
  Task 6.

### Task 4: Conditional local-ROCm paired inference and prediction freeze

**Status:** Skipped because Task 3's agreement prerequisite failed. No GPU
inference ran.

**Files:**
- Create, ignored/private: compatibility and full paired prediction artifacts,
  SHA-256 sidecars, completeness record, and prediction freeze under the new
  run directory.

**Interfaces:**
- Consumes only a passing frozen CPU gate and label-free requests.
- Produces complete frozen paired predictions without loading references.

- [ ] **Step 1: Verify the no-pull local runtime preflight**

  Confirm GPU visibility, image digest presence with `podman image exists`,
  cached model/revision availability in offline mode, frame/request hashes,
  source hashes, and absence of label/reference mounts. Stop and report if any
  prerequisite is missing; do not pull or install.

- [ ] **Step 2: Run a declared compatibility subset**

  Run both conditions on three predeclared opaque panel rows. Require six of six
  `passed` results, bound request/frame hashes, correct pinned identities,
  ROCm/BF16 attestation, and nonempty normalized answers. Freeze the artifact.

- [ ] **Step 3: Run the complete paired panel**

  Invoke the pinned offline runner once with every eligible paired request.
  Refuse output overwrite. Require exactly one successful result per request
  and preserve elapsed wall time, row time, peak memory, and attestation.

- [ ] **Step 4: Freeze and verify predictions**

  Bind protocol, annotations/adjudication, CPU gate, frame plans, requests,
  runtime sources, predictions, completion denominator, and timestamps. Only a
  complete paired freeze may proceed to grading.

### Task 5: Conditional blind grading and paired decision

**Status:** Skipped because no predictions were generated. No references,
grading packet, or grades were loaded.

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/temporal_bottleneck.py`
- Modify: `tests/test_temporal_bottleneck.py`
- Create, ignored/private: references/metrics, opaque grading packet, secret
  condition map, two grader files, adjudication, unblinded evaluation, and
  hashes under the new run directory.

**Interfaces:**
- Produces `build_blind_grade_packet(predictions, references, *, random_bytes) -> tuple[dict, dict]`.
- Produces `evaluate_paired_grades(adjudicated_grades, condition_map, cpu_gate) -> dict`.

- [ ] **Step 1: Write failing blind-packet and decision tests**

  Verify opaque row IDs, CSPRNG-driven permutations, separated secret map,
  stable bindings, no condition/panel/video IDs in grader material, two-grade
  completeness, third-party disagreement adjudication, exact C/P/W ordinal
  transitions, improvement/regression counts, and the frozen promotion rule.

- [ ] **Step 2: Verify RED, implement, and verify GREEN**

  Run focused tests, full `uv run pytest -q`, and `git diff --check`. Commit the
  tracked implementation/tests with message
  `feat: add blinded temporal oracle evaluation`.

- [ ] **Step 3: Load references only after prediction freeze**

  Verify the prediction-freeze hash first. Create post-freeze metrics and an
  opaque answer-only packet whose mapping is separately frozen and withheld.

- [ ] **Step 4: Collect and freeze two independent grade files**

  Each grader sees only the rubric, reference, and randomized candidates.
  Validate `C/P/W` completeness and freeze both before sharing.

- [ ] **Step 5: Adjudicate disagreements and unblind once**

  A third grader adjudicates only disagreement rows without condition names.
  Freeze adjudication, then verify and open the secret map once. Generate paired
  transitions, aggregate results, and the exact branch decision.

### Task 6: Record, render, verify, and stop before transcript work

**Status:** Complete. The visual result is ready for user handoff; transcript
work remains unstarted.

**Files:**
- Create: `docs/experiments/2026-07-30-temporal-required-moment-oracle.md`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Regenerate only: sibling Markdown pages and `reports/project-status/*.html`
- Refresh only if report content changes materially: sanitized derived HTML
  under `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`

**Interfaces:**
- Consumes either the frozen primary-agreement stop or the later CPU gate and,
  conditionally, the frozen paired evaluation.
- Produces one sanitized condition-bound claim and a user-facing checkpoint.

- [x] **Step 1: Write the dated evidence record**

  Record the question, hypothesis/falsifier, panel and selection boundary,
  treatment/control/held-fixed variables, annotation agreement, actual-PTS
  method, recall results, CPU-gate outcome, conditional answer results,
  completion denominator, evidence validity, decision disposition, immutable
  private artifact hashes, limitations, and next decision. Include no private
  questions, answers, frames, media, predictions, labels, or raw logs.

- [x] **Step 2: Update the primary theme and ledger**

  Update only QMD sources. Keep one canonical claim in Visual Evidence and link
  from the ledger. Do not alter the operational baseline unless supported by a
  later deployable selector experiment.

- [x] **Step 3: Regenerate and verify report surfaces**

  Render GFM and HTML from QMD; run focused/full tests, link/fragment/privacy
  scans, `git diff --check`, a clean isolated report regeneration comparison,
  and responsive/accessibility checks proportional to the change.

- [x] **Step 4: Review the whole branch**

  Dispatch the final most-capable reviewer against the complete diff and this
  plan. Fix all Critical/Important findings through the defined review loop and
  rerun fresh verification.

- [x] **Step 5: Stop and share results**

  Report the annotation agreement, both selector recalls, exact missed-row
  counts, CPU-gate decision, conditional paired C/P/W transitions, runtime,
  validity limits, and recommendation. Do not begin transcript work. Wait for
  explicit user direction.
