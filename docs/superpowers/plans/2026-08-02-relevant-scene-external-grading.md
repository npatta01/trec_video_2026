# Relevant-Scene External Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the frozen relevant-scene answer test without changing the sealed Task 2 run, by verifying it through an identity-preserving namespace and performing all new grading work in a private external workspace.

**Architecture:** A private external controller owns one narrow `verify_frozen_task2()` barrier. It launches Bubblewrap and mounts a tmpfs exact view over the original frozen Task 2 pathname inside that namespace, recursively read-only binds the original `phase-bundles/` tree, read-only binds the other legitimate Task 2 evidence, and read-write binds only the original lock file. The child imports the original frozen controller from the same frozen logical pathname used during Task 2, verifies its byte hash, reassigns only `RUN` to that identical pathname, and calls the unchanged `verify_action()`. This preserves every path string in the frozen runner/protocol semantics while hiding the seven historical grading files from `RUN`. Reference files are unavailable in this namespace. Only a fresh external process that revalidates the published verification barrier may open the predeclared ten-file reference allowlist, construct a condition-blind packet, collect independent C/P/W grades, adjudicate disagreements, and unblind. All new controllers, tests, state, and grading artifacts remain outside the shared dataset under `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/`.

**Tech Stack:** Python 3.12, pytest, Bubblewrap 0.11.1, Linux bind mounts and `flock`, SHA-256 canonical JSON records, Quarto.

## Frozen Inputs and Safety Contract

- Read and follow `docs/project-status/adding-experiments.qmd` before any experiment action. The editable result belongs in `docs/project-status/research-visual-evidence.qmd`; generated Markdown and HTML are never hand-edited.
- The shared `data` symlink must resolve to `/home/npatta01/data/competitions/trec_video_2026/data` before every real-tree action.
- Do not create, copy, move, delete, rename, repair, or overwrite anything under `data/`. The sealed Task 2 run is read-only historical evidence.
- Bind the original Task 2 controller path `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/run_relevant_scene_stage2.py` and byte SHA-256 `e432f4b8c642321f46a92481e79a3e415185a4a5d357d833d0a92e7e22cb22fb`.
- Bind the complete Task 2 prediction freeze SHA-256 `111e1215da648bd7fb9e02cb412ff45ce174dc7d0a2dc5019cf8ca11c301f393`, row count `20`, pair count `10`, condition counts `10/10`, and completed phase sequence `preflight, build, compatibility, run, freeze`.
- The original Task 2 top level must contain exactly the six legitimate entries `.controller.lock`, `.quarantine`, `.transactions`, `phase-bundles`, `run_relevant_scene_stage2.py`, and `runtime-src`, plus exactly these seven known historical grading-attempt files: `grade_relevant_scene_stage2.py`, `grading-phase-state.json`, `grading-phase-state.json.sha256`, `grading-step4-state.json`, `grading-step4-state.json.sha256`, `grading-step4-state-v2.json`, and `grading-step4-state-v2.json.sha256`. Any unknown entry fails closed.
- The exact view exposes only the six legitimate entries. Bind `phase-bundles/`, `runtime-src/`, `.transactions/`, `.quarantine/`, and the controller read-only; bind the original `.controller.lock` read-write for cross-process exclusion. Do not reconstruct or filter anything below `phase-bundles/`.
- The verification child imports from the original frozen logical controller pathname under this worktree, leaves `SCRIPT`, `ROOT`, `DATA`, `WRAPPER_SOURCE`, and every other module global unchanged, reassigns only `module.RUN` to its identical original value, and calls the existing decorated `module.verify_action()` once. The mount namespace—not a changed path string—provides the filtered exact view.
- Bubblewrap uses `--unshare-all`, `--unshare-net`, `--die-with-parent`, `--new-session`, and `--clearenv`. It has no external grading output mount, no reference mount, no transcript mount, and no network. A bounded single JSON value on stdout is its only result channel.
- The structural capability probe never invokes `verify_action()` and never opens prediction or reference values. Its sole purpose is to validate namespace policy, exact top-level names, four-field wrapper identities, read-only mounts, lock behavior, canonical import path, single `RUN` override, reference masking, and teardown.
- Before the verification barrier publishes, all failures must record `references_loaded: false` only in the external workspace and must not start the reference process.
- After the barrier, resolve only the predeclared exact ten paths `data/vqa/2025/training/annotations/{video_id}.json`, require no symlinks/path escape and a nonempty string at `/correct_answer`, and open each through no-follow descriptors exactly once. Do not explore for alternatives.
- Use the frozen answer-only rubric: `C` is factually and temporally correct and directly answers the question; `P` has correct relevant content but misses or ambiguously states an essential temporal relation/detail, or mixes correct and incorrect claims; `W` is incorrect, temporally contradictory, unrelated, or nonresponsive.
- Score `C=2`, `P=1`, `W=0`. The answer gate requires at least two improved rows, zero regressed rows, and treatment ordinal gain at least two.
- Grader material contains only opaque row ID, reference answer, randomized candidate A/B answers, and the rubric. Condition/video identity, panel order, frame/runtime provenance, filesystem paths, and the secret map remain withheld until both grade files and the agreement-or-adjudication record are frozen.
- No transcript is created, loaded, mounted, inspected, or run. Share the visual-only result first.
- Keep questions, references, model answers, video IDs, grades, paths, raw logs, and secret mappings out of tracked files and rendered artifacts. Refuse overwrite and publish every external evidence phase atomically with SHA-256 binding and explicit predecessor state.

---

### Task 1: Build the external controller seam with synthetic TDD

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/external_grade_relevant_scene_stage2.py`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/namespace_worker.py`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/test_external_grading.py`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/progress.md`

**Interfaces:**
- `inspect_task2_top_level(root: Path) -> Task2SourceManifest`
- `build_exact_view_spec(manifest: Task2SourceManifest) -> ExactViewSpec`
- `run_structural_probe(spec: ExactViewSpec) -> StructuralProbeResult`
- `verify_frozen_task2(spec: ExactViewSpec) -> FrozenTask2Verification`
- CLI actions `probe`, `verify-task2`, `prepare-packet`, `validate-grades`, and `finalize`.

- [x] **Step 1: Write failing synthetic contract tests**

  Test that the exact normal-plus-historical source set is accepted; an unknown, missing, symlinked, or wrong-kind entry rejects; the exact view contains only the six legitimate names; a recreated wrapper parent fails identity matching; a direct-bind fixture passes; controller or freeze hash drift rejects before import; an attempt to override any global other than `RUN` rejects; references cannot be opened before a verified barrier; malformed/extra verifier output cannot publish; and every incomplete state records `references_loaded: false`.

- [x] **Step 2: Verify RED**

  Run:

  ```bash
  uv run pytest -q .superpowers/sdd/2026-08-02-relevant-scene-external-grading/test_external_grading.py
  ```

  Expected: failure because the external controller and worker contracts do not exist.

- [x] **Step 3: Implement the minimal synthetic seam**

  Use no-follow `lstat`/descriptor checks, exact name and kind allowlists, canonical JSON, semantic/file SHA-256 helpers, no-replace writes, fsync, and predecessor-bound phase records. Separate mount-spec construction from execution so tests can validate observable safety properties without a real namespace. The worker accepts one frozen JSON request, validates exact keys and hashes, performs only the requested `probe` or `verify` operation, and emits one bounded JSON result.

- [x] **Step 4: Verify GREEN and negative boundaries**

  Run the private test and repository checks:

  ```bash
  uv run pytest -q .superpowers/sdd/2026-08-02-relevant-scene-external-grading/test_external_grading.py
  uv run pytest -q
  uv run ruff check src tests
  git diff --check
  ```

  Expected: all available checks pass; any pre-existing unrelated failure is recorded exactly and separated from this controller's tests.

### Task 2: Prove the real Bubblewrap exact view without protected-value access

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/structural-probe.json`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-2-structural-report.md`

- [x] **Step 1: Freeze controller/worker hashes and recheck the real source manifest**

  Require the canonical dataset symlink, exact Task 2 controller hash, exact top-level name/kind set, empty legitimate management directories, and frozen wrapper snapshot identities. Open mount sources with `O_PATH|O_NOFOLLOW` when supported, retain descriptors through the child, and recheck the named objects after exit.

- [x] **Step 2: Run the structural-only Bubblewrap probe**

  Mount a tmpfs exact-view root, bind only the legitimate entries, mask the annotation root, and run the probe worker. Assert the controller remains imported from its canonical original path; only `RUN` changes; the wrapper snapshot and its parent retain exact `(st_dev, st_ino, st_size, st_mtime_ns)` tuples; every evidence bind is read-only in `/proc/self/mountinfo`; the original lock opens and flocks; all seven historical files are absent; references and network are unavailable; stdout is one valid JSON value; and no Task 2 value file is opened.

- [x] **Step 3: Prove teardown and source immutability**

  Require clean child exit, no residual namespace process/mount, identical pre/post four-field metadata for all source entries, unchanged controller hash, unchanged source top-level set, and no new Task 2 artifact. Publish the external structural record only after these checks pass.

- [x] **Step 4: Gate failure handling**

  If Bubblewrap is denied or direct binds change identities, run the same structural-only probe using direct rootless Podman binds without `:O`, `:U`, `:z`, `:Z`, idmap, named volumes, pulls, or network. If both adapters fail, stop before prediction/reference access and require a new user decision; do not substitute a new semantic verifier silently.

### Task 3: Run unchanged Task 2 verification in the reference-denied namespace

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0001-task2-verification/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-3-verification-report.md`

- [x] **Step 1: Revalidate the structural record in a fresh process**

  Verify controller/worker hashes, exact source name/kind set, held identities, source hashes, namespace manifest, and reference masking. Reject drift or a pre-existing verification destination.

- [x] **Step 2: Invoke the unchanged verifier exactly once**

  Inside the same reference-denied exact view, hash and import the original controller at the frozen worktree pathname used during Task 2, reassign only `module.RUN` to that identical run pathname, and call `module.verify_action()`. Accept only one bounded JSON result with `valid: true`, `terminal_incomplete: false`, exact completed phases `preflight, build, compatibility, run, freeze`, row count `20`, pair count `10`, and the frozen prediction hash.

- [x] **Step 3: Publish the external barrier and recheck immutability**

  Atomically publish a phase bundle binding controller/worker hashes, the exact source/mount manifest, structural-probe record, sanitized verifier result, Task 2 freeze hash, and pre/post source observations. It contains no answers. Recheck that the sealed Task 2 top level, hashes, and metadata are unchanged and `references_loaded: false`.

- [x] **Step 4: Independent isolation review**

  A fresh Sol reviewer reads the controller, worker, tests, exact mount manifest, probe record, and verification record. It must independently confirm canonical import, the single `RUN` assignment, recursive original `phase-bundles` bind, read-only evidence, original shared lock, reference absence, strict result predicate, teardown, and zero Task 2 writes. Fix every finding, rerun Tasks 1–3 as applicable, and do not access references until the review is PASS.

### Task 4: Build the blind packet in a fresh external process

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0002-reference-binding/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0003-blind-packet/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-4-packet-report.md`

**Interfaces:**
- `build_blind_grade_packet(predictions, references, *, random_bytes) -> tuple[dict, dict]`
- `validate_grades(packet, grades) -> dict`
- `evaluate_paired_grades(adjudicated, condition_map) -> dict`

- [x] **Step 1: Write and pass synthetic packet/evaluation tests before real access**

  Test fixed-vector domain-separated randomness, pair and candidate permutations, opaque-ID bijection, exactly-once prediction coverage, absence of condition/video/path identity, rubric binding, grade schema, duplicate/missing rows, distinct grader identities, disagreement-only adjudication, early-unblind rejection, all nine C/P/W transitions, per-condition arithmetic, and each independent answer-gate failure. Freeze the reviewed controller and synthetic-test result.

- [x] **Step 2: Revalidate the Task 2 barrier in a fresh process**

  Verify the external predecessor chain and rerun `verify_frozen_task2()` reference-denied. Only then resolve the exact predeclared reference allowlist. Reject symlinks, path escape, wrong kinds, missing/extra IDs, malformed JSON, or an absent/empty/non-string `correct_answer` before publication.

- [x] **Step 3: Read frozen predictions and references once and publish blindly**

  Open the frozen prediction file and each allowlisted reference with held no-follow descriptors, verify all bindings and the 20-row/10-pair join, construct a fresh cryptographically randomized packet and separate secret map, and atomically publish reference-binding and blind-packet phases. Grader handoffs contain no workspace path. No transcript path is accepted.

### Task 5: Collect independent grades, adjudicate, and unblind once

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0004-grader-left/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0005-grader-right/`
- Create, ignored/private as needed: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0006-adjudication/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/phase-bundles/0007-final-evaluation/`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-5-final-private-report.md`

- [x] **Step 1: Freeze two independent grader handoffs**

  Give each fresh no-context grader only the rubric and identical blind packet content, tell it not to use tools or inspect the workspace, and freeze the exact material manifest plus structured C/P/W output. Require distinct grader identities and complete one-grade-per-candidate coverage.

- [x] **Step 2: Adjudicate only disagreements**

  If the two graders disagree, give a fresh adjudicator only the rubric and disputed blind rows; otherwise publish an explicit no-disagreement record. Reject adjudication of agreement rows or incomplete/non-C/P/W output.

- [x] **Step 3: Freeze grades, unblind once, and evaluate**

  Revalidate every predecessor and material hash, then join the frozen final blind grades to the secret map exactly once. Compute condition counts, ordinal totals, all transition counts, improved/regressed/tied rows, treatment gain, and the frozen answer gate. Publish only the private final evaluation; never retry grades after unblinding.

- [x] **Step 4: Independently validate arithmetic and lineage**

  A fresh reviewer recomputes the blind-packet bijection, grader/adjudicator coverage, unblind join, C/P/W totals, transitions, and decision from frozen artifacts, without changing them. Any mismatch marks the experiment incomplete rather than repairing history.

### Task 6: Record and share the sanitized visual-only finding

**Files:**
- Modify, tracked QMD: `docs/project-status/research-visual-evidence.qmd`
- Update by rendering only: generated project-status Markdown/HTML and private portal artifact
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-6-report.md`

- [x] **Step 1: Update the canonical experiment record**

  Add one concise result entry containing the predeclared question, treatment/control, denominator, frozen hashes, verification method, aggregate C/P/W counts, transition summary, gate decision, integrity caveats, and next action. Label incomplete branches honestly. Do not duplicate the result into another theme; link to Visual Evidence instead.

- [x] **Step 2: Render and verify tracked documentation**

  Run the project-prescribed Quarto render and drift checks from `docs/project-status/adding-experiments.qmd`, then `git diff --check`. Inspect the diff for protected content and ensure generated files were changed only by rendering.

- [x] **Step 3: Refresh the private tailnet-only viewing artifact**

  Put only sanitized derived HTML/assets under `/home/npatta01/codex-rendered/plans/`, update `/home/npatta01/codex-rendered/index.html`, verify the Tailscale Serve URL returns the intended artifact and remains tailnet-only, and confirm no dataset, raw log, secret, or unrelated repository path is exposed.

- [x] **Step 4: Share the result before transcript work**

  Report the visual-only outcome, denominator, aggregate score/gate, verified limitations, canonical QMD path, and private portal URL. Do not begin transcript experiments in this plan.

### Task 7: Audit uniform-12 selection headroom without inference

**Files:**
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/audit_uniform_target_overlap.py`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/test_uniform_target_overlap_audit.py`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/uniform-target-overlap-audit.json`
- Create, ignored/private: `.superpowers/sdd/2026-08-02-relevant-scene-external-grading/task-7-selection-headroom-audit.md`

- [x] **Step 1: Bind the exact frozen timestamp inputs**

  Verify the r4 frame-plan and extraction sidecars, the Stage 2 uniform-manifest
  sidecar, exact cross-manifest bindings, ten-row/12-frame matrices, and
  preservation of every uniform plan and extraction record into Stage 2.

- [x] **Step 2: Implement the overlap calculator with synthetic TDD**

  Count inclusive interval contact per row, distinguishing all, partial, and
  zero required-interval coverage. Refuse drift, inconsistent row matrices,
  malformed intervals, and overwrite. Keep row identities opaque and output
  fixed flags proving no model, prediction, reference, or transcript access.

- [x] **Step 3: Run once and independently recompute**

  Freeze one sanitized aggregate outside `data/`, validate its sidecar, and
  reproduce the aggregate through an independent jq calculation over the exact
  frame plan.

- [x] **Step 4: Correct the report and next-test queue**

  Record the limited-headroom result in the Visual Evidence primary theme,
  link it concisely from the ledger, render QMD-derived outputs, refresh the
  tailnet-only report, and retain uniform-12 as the operational baseline.
