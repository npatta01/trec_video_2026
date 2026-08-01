# Relevant-Scene Stage 2 Answer Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Qwen3.5-4B answers the frozen ten temporal questions better when given the r4-audited `oracle-hybrid-12` frames than when given matched `uniform-12` frames.

**Architecture:** Keep inference and grading in separate ignored/private controllers. The inference controller can read questions and selected frame provenance but has no reference-answer input; it runs a six-output compatibility gate, then a complete 20-output paired run, and freezes predictions. Only after that freeze verifies may a new grading controller load references, create a cryptographically randomized condition-blind packet, collect two independent C/P/W grades plus third-party disagreement adjudication, and unblind exactly once. The existing Visual Evidence claim owns the sanitized result.

**Tech Stack:** Python 3.12, pytest, SHA-256 JSON sidecars, rootless Podman, pinned ROCm Transformers runtime, Qwen3.5-4B, Quarto.

## Global Constraints

- Read `docs/project-status/adding-experiments.qmd` before proposing, running, interpreting, or recording this experiment.
- Treat `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-r4/` as immutable Stage 1 evidence; never change it.
- Use a new ignored/private run directory: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/`.
- Bind Stage 1 protocol SHA-256 `514c17d4cd56f05f07d6e0b80219d6edd5f7e53675ce3dfe4838618cf892758c`, extraction manifest file SHA-256 `212a41bfff3f35af3c7fc571c9b7e269b84da11f20821c8ad8ae7234aa7558a5`, selection file SHA-256 `27465d50b80833bf9a488408371fa33908d6c55b7068b1c8050795d3d11097b9`, and sanitized scoreboard file SHA-256 `355b65aab8f5b316b04a623dbce3baabdd300700414986cdf5e643251f656524`.
- Treatment is `oracle-hybrid-12`; control is `uniform-12`; use all ten rows in frozen panel order.
- Model is `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime is `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
- Use seed `0`, greedy decoding, `max_new_tokens=32`, and processor bounds `256*256` through `512*512` pixels per frame.
- Use the existing temporal question-plus-images prompt with `transcript: null`; do not create, load, mount, or run transcript material.
- Use only cached local model/runtime assets and rootless ROCm. Set offline variables, disable container networking, do not pull, do not install, and do not expose a service.
- Compatibility uses the first three paired rows in frozen panel order: exactly six outputs. Full inference is allowed only if all six pass runtime, provenance, completeness, and nonempty-answer checks.
- Predictions must be complete and frozen before any reference-answer path is discovered, loaded, mounted, or passed to code.
- Grader packets must contain only an opaque row ID, reference answer, and randomized candidate A/B answers. Keep the condition map separate and withheld until two grade files and any third-party adjudication are frozen.
- Score `C=2`, `P=1`, `W=0`. The answer gate passes only with at least two improved rows, zero regressed rows, and treatment ordinal gain at least two.
- Passing is exploratory support for retrieval research only. It does not promote an oracle, change uniform-12 as the baseline, or establish generalization.
- Refuse overwrite for every evidence artifact. Give every JSON artifact a verified SHA-256 sidecar or a freeze record that binds upstream artifacts.
- Keep questions, references, answers, predictions, grades, video IDs, frame paths, media, and raw logs out of tracked reports and rendered portal artifacts.
- Share the final visual-test result before any transcript work; this plan ends without transcript work.

---

### Task 1: Build and review the label-isolated Stage 2 inference controller

**Files:**
- Create, ignored/private: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/run_relevant_scene_stage2.py`
- Create, private test: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-1-private-test.py`
- Create, private coordination: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/progress.md`
- Create, tracked plan: `docs/superpowers/plans/2026-08-01-relevant-scene-stage2.md`

**Interfaces:**
- Consumes immutable r4 `protocol.json`, `selection.json`, `stage1-scoreboard-sanitized.json`, `extracted-frame-manifest.json`, and the prior label-free `annotation-packet.json` question order.
- Produces `build_protocol() -> dict[str, object]`, `build_requests() -> list[dict[str, object]]`, `validate_runner_results(requests, results) -> list[dict[str, object]]`, and CLI actions `preflight`, `build`, `compatibility`, `run`, `freeze`, and `verify`.
- Writes no reference, grading, or transcript artifact.

- [ ] **Step 1: Write the failing private contract tests**

  Test exact Stage 1 hashes and decision, frozen order, two requests per row, 12 actual extracted frames per request, matching PNG hashes and actual PTS values, within-row unique `(stream_index, frame_index)` identities, `transcript is None`, pinned model/runtime/decoding, and absence of keys containing `reference`, `answer`, `grade`, or `transcript` except the required null `transcript` request field.

  ```python
  def test_build_requests_is_complete_paired_and_label_isolated(controller):
      rows = controller.build_requests()
      assert len(rows) == 20
      assert [row["condition_id"] for row in rows[:4]] == [
          "uniform-12", "oracle-hybrid-12", "uniform-12", "oracle-hybrid-12"
      ]
      assert all(len(row["frame_paths"]) == 12 for row in rows)
      assert all(len(set(row["frame_sha256s"])) == 12 for row in rows)
      assert all(row["transcript"] is None for row in rows)
      assert all(row["inference"]["seed"] == 0 for row in rows)
      assert all(row["inference"]["do_sample"] is False for row in rows)
      assert all(row["inference"]["max_new_tokens"] == 32 for row in rows)
  ```

- [ ] **Step 2: Run the test to verify RED**

  Run:

  ```bash
  uv run pytest -q .superpowers/sdd/2026-08-01-relevant-scene-stage2/task-1-private-test.py
  ```

  Expected: fail because the Stage 2 controller does not exist.

- [ ] **Step 3: Implement the minimal inference controller**

  Use canonical JSON (`sort_keys=True`, compact separators) for semantic hashes and byte SHA-256 for file bindings. `build_requests()` must join the frozen question order to `extracted-frame-manifest.json` by video ID, emit `uniform-12` then `oracle-hybrid-12` for each row, use `actual_pts_seconds`, and bind each request with `request_sha256`.

  `runner_command()` must mount only the staged temporal runtime source, r4 frame tree, and Hugging Face cache; include:

  ```text
  podman run --rm -i --network=none
    --device /dev/kfd --device /dev/dri --group-add keep-groups
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1
    <pinned-image>
    python -m trec_video_2026.vqa_transformers_runner
    --model-class image-text
    --model-id Qwen/Qwen3.5-4B
    --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    --max-new-tokens 32
  ```

  `validate_runner_results()` must require exact count/order, `status == "passed"`, nonempty normalized answer, pinned model/revision, `device == "cuda:0 (ROCm)"`, `precision == "bfloat16"`, a nonempty `runtime_attestation.torch_hip`, CUDA:0 parameter/buffer placement, and request/frame hashes copied from the frozen requests. It must reject any mismatch or overwrite attempt.

- [ ] **Step 4: Verify GREEN and negative boundaries**

  Add cases for altered frame hash, reordered result, empty answer, wrong revision, missing HIP attestation, incomplete pair, reference-like input, existing destination, and any non-null transcript. Run the private test, then:

  ```bash
  uv run pytest -q
  uv run ruff check src tests
  git diff --check
  ```

  Expected: all pass.

- [ ] **Step 5: Independent two-stage review**

  A fresh spec reviewer checks this task against the design and this plan. A fresh quality reviewer checks failure handling, label isolation, immutable bindings, mount boundaries, and no-pull behavior. Fix all findings and rerun Step 4.

- [ ] **Step 6: Commit tracked Task 1 artifacts**

  Stage only the plan and private coordination record; never stage the ignored run controller, data, frames, questions, or future outputs.

  ```bash
  git add docs/superpowers/plans/2026-08-01-relevant-scene-stage2.md
  git commit -m "docs: plan relevant-scene answer test"
  ```

### Task 2: Run the compatibility gate, full paired inference, and prediction freeze

**Files:**
- Create, ignored/private under the Stage 2 run: `protocol.json`, `requests.json`, `preflight.json`, `compatibility-predictions.json`, `predictions.json`, `prediction-freeze.json`, `phase-state/*.json`, and SHA-256 sidecars
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-2-report.md`

**Interfaces:**
- Consumes the reviewed Task 1 controller and the immutable r4 selection/evidence.
- Produces a verified complete 20-output prediction freeze, or a preserved incomplete/invalid stop without references.

- [ ] **Step 1: Verify the cheapest no-pull preflight**

  Verify the `data` symlink resolves exactly to `/home/npatta01/data/competitions/trec_video_2026/data`; the exact Podman digest exists; Podman is rootless; `/dev/kfd` and `/dev/dri/renderD128` are accessible; the exact model snapshot and all symlink targets exist; runtime source hashes match the protocol; every selected PNG and sidecar verifies; no Stage 2 reference/grade/transcript artifact exists.

  Run controller `preflight`, then `build`. Verify both outputs and sidecars with controller `verify`.

- [ ] **Step 2: Run and freeze the six-output compatibility subset**

  Run controller `compatibility`. It must select the first three panel rows and both conditions in pair order. Require 6/6 passed with exact request/frame provenance, nonempty answers, ROCm/BF16 attestation, correct model/revision/runtime, and no label/reference mount. Freeze the compatibility artifact and re-run controller `verify`.

  If any check fails, preserve the output and stop Task 2 before full inference.

- [ ] **Step 3: Run the complete paired panel**

  Run controller `run` once for all 20 requests. Refuse pre-existing output. Preserve wall time, per-row inference time, peak allocated memory, runtime version, device, precision, and attestation. Require exactly 20/20 passed results with ten complete pairs.

  If the run is partial, preserve it as incomplete and do not retry in place, score, or load references.

- [ ] **Step 4: Freeze and independently verify predictions**

  Run controller `freeze`, then `verify`. `prediction-freeze.json` must bind protocol, r4 selection, r4 manifest, requests, runtime source hashes, full prediction bytes, row count `20`, pair count `10`, condition counts `10/10`, and completeness `true`.

  Dispatch a fresh read-only reviewer to recompute every hash, verify request/result joins and runtime attestations, scan the inference controller and container mount list for label/reference/transcript access, and write a PASS/FAIL review. Only PASS permits Task 3.

- [ ] **Step 5: Share the pre-grading checkpoint**

  Tell the user whether compatibility passed, whether the full run completed, exact denominator, elapsed wall time, and that predictions are frozen. Do not show model answers or inspect references yet.

### Task 3: Build the post-freeze blind grading controller and adjudicate

**Files:**
- Create only after Task 2 freeze verification, ignored/private: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/grade_relevant_scene_stage2.py`
- Create only after Task 2 freeze verification, private test: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-3-private-test.py`
- Create, ignored/private: `reference-binding.json`, `blind-grade-packet.json`, `secret-condition-map.json`, `grader-left.json`, `grader-right.json`, `grade-adjudication.json`, `unblind-record.json`, `final-evaluation.json`, and SHA-256 sidecars
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-3-report.md`

**Interfaces:**
- Consumes the independently verified complete prediction freeze before discovering/loading the existing frozen development-reference source.
- Produces `build_blind_grade_packet(predictions, references, *, random_bytes) -> tuple[dict, dict]`, `validate_grades(packet, grades) -> dict`, and `evaluate_paired_grades(adjudicated, condition_map) -> dict`.

- [ ] **Step 1: Reverify freeze before reference discovery**

  Run the Task 2 controller `verify` in a clean process. Record the successful freeze hash check in phase state. Only then inspect the existing temporal diagnostic scoring workflow to bind the same frozen development-reference source; do not use prior predictions or grades.

- [ ] **Step 2: Write failing blind-packet and decision tests**

  ```python
  def test_blind_packet_separates_secret_condition_map(grader):
      packet, secret = grader.build_blind_grade_packet(
          predictions=complete_pairs(),
          references=references(),
          random_bytes=b"\x01" * 64,
      )
      assert len(packet["rows"]) == 10
      assert all(set(row) == {"opaque_id", "reference", "candidate_a", "candidate_b"}
                 for row in packet["rows"])
      assert "condition" not in json.dumps(packet).lower()
      assert len(secret["rows"]) == 10

  def test_answer_gate_requires_two_gains_no_losses_and_gain_two(grader):
      result = grader.evaluate_paired_grades(adjudicated(), condition_map())
      assert result["answer_gate"] == (
          result["improved_rows"] >= 2
          and result["regressed_rows"] == 0
          and result["treatment_ordinal_gain"] >= 2
      )
  ```

  Add negative cases for changed freeze, missing reference, duplicate opaque ID, leaking video/condition identity, non-C/P/W grade, incomplete grader file, same grader identity twice, adjudicator not fresh, adjudicating an agreement row, and unblinding before grade/adjudication freezes.

- [ ] **Step 3: Verify RED, implement minimally, and verify GREEN**

  Use `secrets.token_bytes()` for a fresh run seed, SHA-256 counter expansion for deterministic packet construction from that secret, independently permute pair order and candidate A/B order, and store seed/mappings only in `secret-condition-map.json`. Bind the packet and map to prediction-freeze and reference hashes. Never expose panel order, video ID, condition ID, runtime, or frame identity in the grader packet.

  Run the private tests, full tests, Ruff, and `git diff --check`. Review the controller for one-way phase enforcement.

- [ ] **Step 4: Build and freeze the blind packet/map**

  Load only references for the ten frozen questions, verify one-to-one coverage, and write/freeze packet and map. The controller keeps the map withheld. Compute semantic metrics only as a secondary post-freeze diagnostic; never use them to infer C/P/W grades.

- [ ] **Step 5: Collect two independent blind grades**

  Dispatch two fresh graders. Each sees only the rubric and `blind-grade-packet.json`. Grade every candidate answer independently as `C`, `P`, or `W`, with a concise answer-specific rationale. Freeze and validate both complete grade files before comparing them.

- [ ] **Step 6: Adjudicate disagreements without condition identities**

  If the two grade files disagree anywhere, dispatch a fresh third grader with only the disagreement rows, rubric, reference, randomized candidates, and the two rationales. The third grader must not see the map. Freeze adjudication and validate it cannot override agreement rows.

- [ ] **Step 7: Unblind once and freeze the decision**

  Verify all upstream hashes, open `secret-condition-map.json` once, write `unblind-record.json`, and compute per-condition C/P/W counts, ordinal scores, all paired transitions, improved/regressed/tied counts, treatment ordinal gain, exact answer-gate booleans, runtime aggregates, and the condition-bound decision. Freeze `final-evaluation.json` and have a fresh reviewer recompute it from frozen inputs.

### Task 4: Record, render, review, and hand off the result

**Files:**
- Modify authored evidence: `docs/experiments/2026-07-30-temporal-required-moment-oracle.md`
- Modify authored primary theme: `docs/project-status/research-visual-evidence.qmd`
- Modify authored ledger: `docs/project-status/experiment-ledger.qmd`
- Regenerate only: sibling Markdown and `reports/project-status/*.html`
- Refresh only sanitized derived HTML: `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-4-report.md`

**Interfaces:**
- Consumes frozen Stage 1 scoreboard, prediction freeze, blind grades/adjudication, and independently verified final evaluation.
- Produces one sanitized answering result under the existing Visual Evidence claim and a user-facing result before any transcript work.

- [ ] **Step 1: Update authored report sources only**

  Record extraction gate, representation gate, 20/20 or actual inference denominator, compatibility outcome, runtime, control/treatment C/P/W, ordinal scores, exact paired transitions, improved/regressed rows, ordinal gain, answer-gate result, evidence validity, decision disposition, immutable private artifact hashes, limitations, and next decision. State explicitly that no transcript was used or run. Do not copy private questions, references, answers, grades, IDs, paths, frames, media, or logs.

- [ ] **Step 2: Regenerate report outputs**

  Render GFM and HTML from QMD. Never hand-edit generated Markdown or HTML. Update the tailnet-only portal with only sanitized derived HTML.

- [ ] **Step 3: Verify and independently review**

  Run focused/full tests, Ruff, `git diff --check`, link/fragment checks, privacy scans, isolated regeneration comparison, responsive/accessibility checks, and exact arithmetic/hash review. Confirm the portal URL returns the intended artifact and the Tailscale mapping remains tailnet-only.

- [ ] **Step 4: Commit the scoped report and share results**

  Commit only tracked report/QMD/generated surfaces. Share the three gates separately: whether relevant scenes were extracted, whether their relationships were represented, and whether Qwen3.5-4B answered better. Include exact C/P/W transitions and runtime, then stop. Do not begin transcript work.

## Plan Self-Review

- Spec coverage: all pinned inference values, Stage 1 prerequisite, first-three paired compatibility gate, complete prediction freeze, delayed reference access, two graders plus third adjudicator, one-time unblind, exact answer gate, reporting boundary, and no-transcript stop have explicit tasks.
- Placeholder scan: no deferred implementation placeholders are present; post-freeze reference discovery is an intentional label-isolation gate tied to the existing scoring workflow.
- Type consistency: Task 1 produces the request/prediction freeze consumed by Task 3; Task 3 function and artifact names are consistent through evaluation and reporting.
