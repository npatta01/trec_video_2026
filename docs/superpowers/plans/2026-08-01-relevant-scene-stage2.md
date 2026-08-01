# Relevant-Scene Stage 2 Answer Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Qwen3.5-4B answers the frozen ten temporal questions better when given the r4-audited `oracle-hybrid-12` frames than when given matched `uniform-12` frames.

**Architecture:** Keep inference and grading in separate ignored/private controllers. A small staged in-container wrapper applies and attests seed `0`, verifies the canonical request and every mounted PNG byte immediately before generation, and augments the pinned base runner's result with source/prompt/runtime provenance. The inference controller can read questions and selected frame provenance but has no reference-answer input; it runs a six-output compatibility gate, then a complete 20-output paired run, and freezes predictions. Only after that freeze independently verifies may a new grading controller resolve the predeclared ten-file reference allowlist, create a cryptographically randomized condition-blind packet, collect two isolated C/P/W grades plus third-party disagreement adjudication, and unblind exactly once. The existing Visual Evidence claim owns the sanitized result.

**Tech Stack:** Python 3.12, pytest, SHA-256 JSON sidecars, rootless Podman, pinned ROCm Transformers runtime, Qwen3.5-4B, Quarto.

## Global Constraints

- Read `docs/project-status/adding-experiments.qmd` before proposing, running, interpreting, or recording this experiment.
- Treat `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-r4/` as immutable Stage 1 evidence; never change it.
- Use a new ignored/private run directory: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/`.
- Bind Stage 1 protocol SHA-256 `514c17d4cd56f05f07d6e0b80219d6edd5f7e53675ce3dfe4838618cf892758c`, extraction manifest file SHA-256 `212a41bfff3f35af3c7fc571c9b7e269b84da11f20821c8ad8ae7234aa7558a5`, bounded-frame manifest file SHA-256 `7106763cfb5bbd12062f203f81021d3ae3de5903bd1442bf4a7ead3444c3fe49`, selection file SHA-256 `27465d50b80833bf9a488408371fa33908d6c55b7068b1c8050795d3d11097b9`, and sanitized scoreboard file SHA-256 `355b65aab8f5b316b04a623dbce3baabdd300700414986cdf5e643251f656524`.
- Bind the label-free question-order source `data/vqa/benchmarks/temporal-required-moment-oracle-10/annotation-packet.json` at file SHA-256 `f7e57def9f296f5c1d4804a45b802ed4abee30bf1d7c1690bc9dc5e39861a122`.
- Bind the staged base runner SHA-256 `a1e0ccbf1bc15fda04fe811110d47eebc1f91b9c23a6a89c457ced0ab98340a6` and prompt/model-message source SHA-256 `ed2b85245551dedc484df8fd2e38a560429c795df89c0a270ed2150e3423c9cb` before any run.
- Treatment is `oracle-hybrid-12`; control is `uniform-12`; use all ten rows in frozen panel order.
- Treatment requests use only immutable `bounded_png_path`/`bounded_png_sha256` records from r4 `bounded-frame-manifest.json`; full-resolution treatment PNGs and every `oracle-local-12` PNG are forbidden. Before requests freeze, derive a matched bounded uniform control under the new Stage 2 run with the exact Stage 1 rule: `scale=min(1,sqrt(262144/(width*height)))`, target dimensions `floor(width*scale)` by `floor(height*scale)`, and pinned ImageMagick `-resize WIDTHxHEIGHT! PNG24`. Freeze source hashes, dimensions, exact argv/tool identity, and output hashes in `uniform-bounded-manifest.json`.
- Model is `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime is `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
- Apply Python, NumPy, Torch CPU, and Torch accelerator seed `0` inside the container before model load. Record the applied seed and deterministic-runtime flags in every output. Use greedy decoding, `max_new_tokens=32`, and processor bounds `256*256` through `512*512` pixels per frame.
- Use the existing temporal question-plus-images prompt with `transcript: null`; do not create, load, mount, or run transcript material.
- Use only cached local model/runtime assets and rootless ROCm. Set offline variables, disable container networking, do not pull, do not install, and do not expose a service.
- Compatibility uses the first three paired rows in frozen panel order: exactly six outputs. Full inference is allowed only if all six pass runtime, provenance, completeness, and nonempty-answer checks.
- Predictions must be complete and frozen before any predeclared reference-answer file is opened, hashed, parsed, mounted, or passed to code.
- The non-content-bearing reference locator is predeclared as exactly `data/vqa/2025/training/annotations/{video_id}.json`, limited to the ten frozen video IDs, with answer JSON pointer `/correct_answer`. Knowing this locator is allowed before prediction freeze; opening, hashing, parsing, or mounting any matching file is forbidden until the freeze independently passes. The post-freeze resolver must reject symlinks, path escape, missing/extra IDs, non-object JSON, or an absent/empty/non-string `correct_answer`; no repository exploration is allowed after predictions exist.
- Freeze this answer-only rubric before inference: `C` means factually and temporally correct and directly answers the question, allowing harmless paraphrase; `P` means some correct relevant content but an essential temporal relation/detail is missing or ambiguous, or correct and incorrect claims are mixed so the reference is not fully satisfied; `W` means incorrect, temporally contradictory, unrelated, or nonresponsive. Every grader/adjudicator material manifest and grade file binds the rubric hash.
- Grader packets must contain only an opaque row ID, reference answer, and randomized candidate A/B answers. Keep the condition map outside the grader handoff and withheld until two grade files and a conditional adjudication-or-no-disagreement record are frozen. Fresh graders receive the frozen rubric and packet content directly in a no-context prompt, are told not to use tools or inspect the workspace, and receive no run-directory path. Freeze the exact material manifest/brief supplied to each grader and adjudicator.
- Score `C=2`, `P=1`, `W=0`. The answer gate passes only with at least two improved rows, zero regressed rows, and treatment ordinal gain at least two.
- Passing is exploratory support for retrieval research only. It does not promote an oracle, change uniform-12 as the baseline, or establish generalization.
- Refuse overwrite for every evidence artifact. Give every JSON artifact a verified SHA-256 sidecar or a freeze record that binds upstream artifacts.
- Keep questions, references, answers, predictions, grades, video IDs, frame paths, media, and raw logs out of tracked reports and rendered portal artifacts.
- A failed compatibility gate or partial full run is a terminal incomplete branch: preserve the denominator and failure record, never load references, never compute C/P/W, and report no completed paired comparison.
- Share the final visual-test result before any transcript work; this plan ends without transcript work.

---

### Task 1: Build and review the label-isolated Stage 2 inference controller

**Files:**
- Create, ignored/private: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/run_relevant_scene_stage2.py`
- Create, ignored/private staged runtime: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/runtime-src/stage2_runner.py`
- Create, private test: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-1-private-test.py`
- Create, private coordination: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/progress.md`
- Create, tracked plan: `docs/superpowers/plans/2026-08-01-relevant-scene-stage2.md`

**Interfaces:**
- Consumes immutable r4 `protocol.json`, `selection.json`, `stage1-scoreboard-sanitized.json`, `extracted-frame-manifest.json`, and the prior label-free `annotation-packet.json` question order.
- Produces `build_protocol() -> dict[str, object]`, `build_requests() -> list[dict[str, object]]`, `validate_runner_results(requests, results) -> list[dict[str, object]]`, and CLI actions `preflight`, `build`, `compatibility`, `run`, `freeze`, and `verify`.
- The staged `stage2_runner.py` consumes the JSON request batch on stdin, verifies canonical request hashes and PNG bytes, applies seed `0`, invokes the pinned base runner in-process, and emits augmented result JSON on stdout.
- Writes no reference, grading, or transcript artifact.

- [ ] **Step 1: Write the failing private contract tests**

  Test exact Stage 1 extraction/bounded/question/base-runner/prompt-source hashes and decision, frozen order, two requests per row, 12 bounded frames per request, immutable treatment `bounded_png_path`/`bounded_png_sha256`, synthetically derived uniform-bounded fixture provenance, matching actual PTS values, within-row unique `(stream_index, frame_index)` identities, `transcript is None`, pinned model/runtime/decoding, the frozen answer-only rubric, and absence of keys containing `reference`, `answer`, `grade`, or `transcript` except the required null `transcript` request field.

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

  Use canonical JSON (`sort_keys=True`, compact separators) for semantic hashes and byte SHA-256 for file bindings. Implement the exact Stage 1 bounded-dimension/PNG24 derivation for uniform controls and test it against hand-calculated dimensions plus a synthetic ImageMagick fixture without executing the real 120-frame build. `build_requests()` must join the frozen question order to the frozen Stage 2 `uniform-bounded-manifest.json` and r4 `bounded-frame-manifest.json` by video ID, emit `uniform-12` then `oracle-hybrid-12` for each row, use `actual_pts_seconds`, and bind each request with `request_sha256`.

  Write the staged wrapper only after a failing wrapper-contract test. The test must run it against a fake in-process base runner so that deleting seed application, changing a PNG byte, changing the canonical request, omitting source/prompt/runtime attestation, or emitting a different pixel bound fails without loading a real model.

  `runner_command()` must return the literal prefix/environment/image/module argv below and an exact read-only mount allowlist. Tests compare the argv list, not a joined substring. The only mounts are wrapper directory, pinned base runtime source, the ten sorted parent directories named by r4 `oracle-hybrid-12` bounded paths (mounted to the identical absolute paths), the new Stage 2 uniform-bounded root (mounted to its identical absolute path), and the one cached Qwen model repository: 14 mounts total. The controller must prove the ten treatment parents equal the selected hybrid set in the bound manifest and are disjoint from all `oracle-local-12` parents.

  ```python
  hybrid_parents = tuple(sorted({
      str(Path(frame["bounded_png_path"]).parent)
      for frames in bounded_manifest["conditions"]["oracle-hybrid-12"].values()
      for frame in frames
  }))
  local_parents = {
      str(Path(frame["bounded_png_path"]).parent)
      for frames in bounded_manifest["conditions"]["oracle-local-12"].values()
      for frame in frames
  }
  if len(hybrid_parents) != 10 or set(hybrid_parents) & local_parents:
      raise ValueError("selected bounded-treatment mount set is invalid")
  mounts = [
      (str(STAGE2_RUNTIME), "/opt/trec-stage2-wrapper"),
      (str(BASE_RUNTIME), "/opt/trec-base"),
      *((path, path) for path in hybrid_parents),
      (str(STAGE2_UNIFORM_BOUNDED), str(STAGE2_UNIFORM_BOUNDED)),
      (str(QWEN_CACHE), "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B"),
  ]
  argv = ["podman", "run", "--pull=never", "--rm", "-i", "--network=none",
          "--device", "/dev/kfd", "--device", "/dev/dri",
          "--group-add", "keep-groups"]
  for host, container in mounts:
      argv.extend(["-v", f"{host}:{container}:ro"])
  argv.extend([
      "-w", "/opt/trec-base",
      "-e", "PYTHONPATH=/opt/trec-stage2-wrapper:/opt/trec-base",
      "-e", "PYTHONDONTWRITEBYTECODE=1",
      "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
      "-e", "HF_DATASETS_OFFLINE=1", "-e", "PYTHONHASHSEED=0",
      "-e", f"VQA_RUNTIME_IMAGE={RUNTIME_IMAGE}", "-e", "STAGE2_SEED=0",
      RUNTIME_IMAGE, "python", "-m", "stage2_runner",
      "--model-class", "image-text", "--model-id", "Qwen/Qwen3.5-4B",
      "--revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      "--max-new-tokens", "32",
  ])
  ```

  Before importing/invoking the base runner, the wrapper must call `random.seed(0)`, `numpy.random.seed(0)`, `torch.manual_seed(0)`, and `torch.cuda.manual_seed_all(0)`, then set `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True`. It must open each frame through its mounted path immediately before base-runner invocation, SHA-256 the bytes, compare them to the request, and emit the verified hashes. It must emit the canonical request hash, all four seed applications, `PYTHONHASHSEED`, both backend flags, greedy flag, pixel bounds, runtime-image digest, wrapper/base-runner/prompt-source byte hashes, prompt contract ID, and request identity.

  `validate_runner_results()` must require exact count/order, `status == "passed"`, nonempty normalized answer, pinned model/revision, `device == "cuda:0 (ROCm)"`, `precision == "bfloat16"`, a nonempty `runtime_attestation.torch_hip`, CUDA:0 parameter/buffer placement, exact in-container verified request/frame hashes, seed/runtime/source/prompt/pixel attestations, and the literal runtime image. It must reject controller-added provenance that was not emitted by the wrapper, any mismatch, or any overwrite attempt.

- [ ] **Step 4: Verify GREEN and negative boundaries**

  Add cases for altered frame bytes/hash, full-resolution treatment path/hash, an `oracle-local-12` path, changed uniform resize rule/tool/provenance, altered canonical request, reordered result, empty answer, wrong revision, missing/wrong seed attestation, missing HIP attestation, wrong runtime/source/prompt hash, wrong pixel bound, incomplete pair, reference-like input, an argv/mount outside the exact allowlist, forbidden r4 subtrees being reachable, existing destination, and any non-null transcript. Run the private test, then:

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
- Create, ignored/private under the Stage 2 run: `grading-rubric.json`, `protocol.json`, `uniform-bounded-manifest.json`, `frames/bounded/uniform-12/**/*.png`, `requests.json`, `preflight.json`, `preflight-pass.json`, `compatibility-predictions.json`, `compatibility-pass.json`, `predictions.json`, `prediction-freeze.json`, terminal failure record when applicable, `phase-state/*.json`, and SHA-256 sidecars
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-2-report.md`

**Interfaces:**
- Consumes the reviewed Task 1 controller and the immutable r4 selection/evidence.
- Produces a verified complete 20-output prediction freeze, or a preserved incomplete/invalid stop without references.

- [ ] **Step 1: Verify the cheapest no-pull preflight**

  Verify the `data` symlink resolves exactly to `/home/npatta01/data/competitions/trec_video_2026/data`; the exact Podman digest exists without pull; Podman is rootless; `/dev/kfd` and `/dev/dri/renderD128` are accessible; the exact model snapshot and all symlink targets exist; wrapper/base-runner/prompt-source hashes match the protocol; r4 extraction and bounded manifests plus every selected treatment PNG/sidecar verify; the pinned ImageMagick executable/version identity matches the r4 resize tool; the question source matches its predeclared hash; the rubric is frozen; the exact dynamic 14-mount allowlist passes; forbidden full-resolution/unselected r4 subtrees are not mounted/reachable; and no Stage 2 reference/grade/transcript artifact exists. Freeze an explicit `preflight-pass.json` bound to these checks.

  Run controller `preflight`, then `build`. During `build`, derive all 120 uniform bounded controls once with the exact frozen Stage 1 dimension/PNG24 rule, refuse any pre-existing output, freeze `uniform-bounded-manifest.json` with source/tool/argv/output provenance, then construct requests from those outputs and immutable r4 bounded hybrid frames. Verify both conditions contain 120 distinct PNG hashes, no full-resolution treatment path/hash appears, no `oracle-local-12` path appears, and all outputs/sidecars verify with controller `verify`.

- [ ] **Step 2: Run and freeze the six-output compatibility subset**

  Run controller `compatibility`. It must select the first three panel rows and both conditions in pair order. Require 6/6 passed with exact in-container request/frame provenance, applied seed, runtime/source/prompt/pixel attestations, nonempty answers, ROCm/BF16 attestation, correct model/revision/runtime, and no label/reference mount. Freeze both the compatibility artifact and an explicit `compatibility-pass.json` bound to it, then re-run controller `verify`.

  If any check fails, preserve the output and stop Task 2 before full inference.

- [ ] **Step 3: Run the complete paired panel**

  Run controller `run` once for all 20 requests. Refuse pre-existing output. Preserve wall time, per-row inference time, peak allocated memory, runtime version, device, precision, and attestation. Require exactly 20/20 passed results with ten complete pairs.

  If the run is partial, preserve it as incomplete and do not retry in place, score, or load references.

- [ ] **Step 4: Freeze and independently verify predictions**

  Run controller `freeze`, then `verify`. `prediction-freeze.json` must bind the exact question-order source/hash, Stage 2 protocol, controller and wrapper bytes, frozen rubric, preflight result/pass, compatibility predictions/pass, phase-state predecessor, r4 protocol/selection/extraction/bounded manifest/scoreboard, frozen Stage 2 uniform-bounded manifest, requests, base-runner/prompt-source hashes, full prediction bytes, row count `20`, pair count `10`, condition counts `10/10`, and completeness `true`.

  Dispatch a fresh read-only reviewer to recompute every hash, verify the predecessor chain, request/result joins and in-container provenance/runtime attestations, scan controller/wrapper/argv/mounts for label/reference/transcript access, and write a PASS/FAIL review. Only PASS permits Task 3.

- [ ] **Step 5: Share the pre-grading checkpoint**

  Tell the user whether compatibility passed, whether the full run completed, exact denominator, elapsed wall time, and that predictions are frozen. Do not show model answers or inspect references yet.

- [ ] **Step 6: Freeze a terminal record for any incomplete branch**

  If compatibility is not 6/6 or full inference is not 20/20 with ten pairs, write a hash-bound terminal failure record containing the completed denominator, failed check, preserved artifact hashes, and `references_loaded: false`. Skip Task 3 and branch directly to Task 4's incomplete-report path; do not stop before that reporting/portal workflow finishes.

### Task 3: Build the post-freeze blind grading controller and adjudicate

**Files:**
- Create only after Task 2 freeze verification, ignored/private: `data/vqa/benchmarks/temporal-relevant-scene-oracle-10-stage2-r1/grade_relevant_scene_stage2.py`
- Create only after Task 2 freeze verification, private test: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-3-private-test.py`
- Create, ignored/private: `reference-binding.json`, `blind-grade-packet.json`, per-grader/adjudicator material manifests and briefs, `secret-condition-map.json`, `grader-left.json`, `grader-right.json`, `grade-adjudication-or-agreement.json`, `unblind-record.json`, `final-evaluation.json`, and SHA-256 sidecars
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-3-report.md`

**Interfaces:**
- Consumes the independently verified complete prediction freeze before opening/hashing/parsing the predeclared `data/vqa/2025/training/annotations/{video_id}.json` allowlist and `/correct_answer` fields.
- Produces `build_blind_grade_packet(predictions, references, *, random_bytes) -> tuple[dict, dict]`, `validate_grades(packet, grades) -> dict`, and `evaluate_paired_grades(adjudicated, condition_map) -> dict`.

- [ ] **Step 1: Reverify freeze before resolving the reference allowlist**

  Run the Task 2 controller `verify` in a clean process and record the successful freeze hash check in phase state without displaying model-answer values. Do not open any real reference or inspect model-answer content while designing, implementing, or testing the grading controller. Use only synthetic request/prediction/reference fixtures through Step 3.

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

  def test_answer_gate_arithmetic_from_literal_pairs(grader):
      # Control -> treatment: W->P, P->C, C->C, P->P.
      result = grader.evaluate_paired_grades(
          literal_adjudicated_pairs([("W", "P"), ("P", "C"), ("C", "C"), ("P", "P")]),
          literal_condition_map(),
      )
      assert result["transitions"] == {"W->P": 1, "P->C": 1, "C->C": 1, "P->P": 1}
      assert result["control"] == {"C": 1, "P": 2, "W": 1, "ordinal": 4}
      assert result["treatment"] == {"C": 2, "P": 2, "W": 0, "ordinal": 6}
      assert result["improved_rows"] == 2
      assert result["regressed_rows"] == 0
      assert result["tied_rows"] == 2
      assert result["treatment_ordinal_gain"] == 2
      assert result["answer_gate"] is True
  ```

  Add fixed-vector tests for SHA-256 domain-separated expansion, row permutation, independent A/B permutation, opaque-ID derivation, bijection, exact stable mapping for fixed random bytes, and every prediction appearing exactly once. Add literal arithmetic fixtures covering `score(C)=2`, `score(P)=1`, `score(W)=0`, every transition direction, per-condition totals, treatment-minus-control gain, and separate answer-gate failures for one improvement, one regression, and gain below two. Add negative cases for changed freeze, non-allowlisted/symlinked/malformed reference, missing reference, duplicate opaque ID, leaking video/condition identity, non-C/P/W grade, incomplete grader file, same grader identity twice, adjudicator not fresh, adjudicating an agreement row, and unblinding before grade/adjudication-or-agreement freezes.

- [ ] **Step 3: Verify RED, implement minimally, and verify GREEN**

  Use `secrets.token_bytes()` for a fresh run seed and domain-separated SHA-256 counter expansion (`row-order`, `candidate-order`, and `opaque-id` domains) for deterministic construction from that secret. Independently permute pair order and candidate A/B order, prove a one-to-one mapping, and bind packet/map commitments to prediction-freeze, reference, and rubric hashes. Never expose panel order, video ID, condition ID, runtime, or frame identity in the grader packet.

  Run the private tests, full tests, Ruff, and `git diff --check`. Review the controller for one-way phase enforcement. Freeze the grading controller byte hash, exact ten-ID reference path allowlist/schema configuration, rubric hash, and synthetic-test result before any real reference or model-answer value is opened.

- [ ] **Step 4: Build and freeze the blind packet/map**

  Start a clean process that first reverifies the complete prediction freeze, grading-controller byte hash, reference allowlist/schema configuration hash, rubric hash, and synthetic-test record. Only then resolve exactly the ten allowlisted regular files under `data/vqa/2025/training/annotations/`, reject symlinks/path escape/missing/extra IDs/schema violations, read each `/correct_answer` once, and freeze each file hash in `reference-binding.json`. In that same phase, load the frozen predictions, verify one-to-one coverage, and write/freeze the packet plus a cryptographic commitment to the withheld map. Keep the actual map outside every grader handoff; no grader prompt may name its path. Do not inspect repository scoring workflows, prior predictions, prior grades, or any other annotation key. Compute semantic metrics only as a secondary post-freeze diagnostic; never use them to infer C/P/W grades.

- [ ] **Step 5: Collect two independent blind grades**

  Build and freeze a separate material manifest/brief for each grader containing the exact rubric hash, packet hash/content, grader identity, and prohibition on tools/workspace inspection. Dispatch two fresh no-context graders by embedding only those materials in the prompt; provide no filesystem/run path. Each grader uses no tools and grades every candidate independently as `C`, `P`, or `W`, with a concise answer-specific rationale. Freeze and validate both complete grade files before comparing them.

- [ ] **Step 6: Adjudicate disagreements without condition identities**

  If the two grade files disagree anywhere, freeze an adjudicator material manifest/brief and dispatch a fresh no-context third grader with only disagreement rows, rubric, references, randomized candidates, and the two rationales; provide no path and allow no tools. Validate that the adjudicator cannot override agreement rows. If there are no disagreements, freeze an explicit no-disagreement record. In either branch, `grade-adjudication-or-agreement.json` must exist and verify before unblinding.

- [ ] **Step 7: Unblind once and freeze the decision**

  Verify all upstream hashes, open `secret-condition-map.json` once, write `unblind-record.json`, and compute per-condition C/P/W counts, ordinal scores, all paired transitions, improved/regressed/tied counts, treatment ordinal gain, exact answer-gate booleans, runtime aggregates, and the condition-bound decision. Freeze `final-evaluation.json` and have a fresh reviewer recompute it from frozen inputs.

### Task 4: Record, render, review, and hand off the result

**Files:**
- Modify authored evidence: `docs/experiments/2026-07-30-temporal-required-moment-oracle.md`
- Modify authored primary theme: `docs/project-status/research-visual-evidence.qmd`
- Modify authored ledger: `docs/project-status/experiment-ledger.qmd`
- Regenerate only: sibling Markdown and `reports/project-status/*.html`
- Refresh only sanitized derived HTML: `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`
- Modify portal index only when the sanitized report changes: `/home/npatta01/codex-rendered/index.html`
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-4-report.md`

**Interfaces:**
- Consumes either `(frozen Stage 1 scoreboard + terminal failure record)` for the incomplete path or `(frozen Stage 1 scoreboard + verified prediction freeze + blind grades/adjudication + independently verified final evaluation)` for the complete path.
- Produces one sanitized answering result under the existing Visual Evidence claim and a user-facing result before any transcript work.

- [ ] **Step 1: Update authored report sources only**

  For a complete verified run, record extraction gate, representation gate, 20/20 inference denominator, compatibility outcome, runtime, control/treatment C/P/W, ordinal scores, exact paired transitions, improved/regressed rows, ordinal gain, answer-gate result, evidence validity, decision disposition, immutable private artifact hashes, limitations, and next decision. For a compatibility failure or partial full run, record only the completed denominator, failed gate, preserved hashes, and **Incomplete** validity; state that references were not loaded and do not report C/P/W, ordinal scores, or an answer-gate conclusion. In every branch state explicitly that no transcript was used or run. Do not copy private questions, references, answers, grades, IDs, paths, frames, media, or logs.

- [ ] **Step 2: Regenerate report outputs**

  Render GFM and HTML from QMD. Never hand-edit generated Markdown or HTML. Update the tailnet-only portal with only sanitized derived HTML and refresh `/home/npatta01/codex-rendered/index.html` to the intended artifact.

- [ ] **Step 3: Verify and independently review**

  Run focused/full tests, Ruff, `git diff --check`, link/fragment checks, privacy scans of both rendered artifact and portal index, isolated regeneration comparison, responsive/accessibility checks, and exact arithmetic/hash review. Confirm the exact HTTPS artifact and portal index return the intended sanitized content and the Tailscale mapping remains tailnet-only.

- [ ] **Step 4: Commit the scoped report and share results**

  Commit only tracked report/QMD/generated surfaces. Share the three gates separately: whether relevant scenes were extracted, whether their relationships were represented, and—only for a complete graded run—whether Qwen3.5-4B answered better. Include exact C/P/W transitions and runtime only when valid. Handoff both the canonical repository source path and the working tailnet-only portal URL, then stop. Do not begin transcript work.

## Plan Self-Review

- Spec coverage: all pinned inference values including applied/attested seed, Stage 1/question/runtime/prompt prerequisites, exact no-pull mounts, in-container request/PNG provenance, first-three paired compatibility gate, complete expanded prediction freeze, predeclared post-freeze reference allowlist, frozen rubric, isolated two-graders-plus-third adjudicator, one-time unblind, literal answer arithmetic, terminal incomplete branch, portal handoff, and no-transcript stop have explicit tasks.
- Placeholder scan: no deferred implementation placeholders are present; the exact reference locator/schema is predeclared while all file contents and hashes remain post-freeze only.
- Type consistency: Task 1 produces the request/prediction freeze consumed by Task 3; Task 3 function and artifact names are consistent through evaluation and reporting.
