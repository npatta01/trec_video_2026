# Relevant-Scene Stage 2 Answer Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether Qwen3.5-4B answers the frozen ten temporal questions better when given the r4-audited `oracle-hybrid-12` frames than when given matched `uniform-12` frames.

**Architecture:** Keep inference and grading in separate ignored/private controllers. Preflight freezes a sealed provenance snapshot of the reviewed wrapper and embeds those exact bytes, base64 encoded with their SHA-256, into the private frozen runner argv. A fixed literal `python -c` bootstrap decodes, hashes, compiles, and executes only that in-argv payload; no host wrapper pathname is an execution input. The wrapper applies and attests seed `0`, reads each mounted PNG once, verifies its hash, snapshots those exact bytes to container-private paths, and makes the pinned base runner consume only those snapshots; it likewise captures the exact three allowlisted Python source byte payloads and executes them with a private byte-backed loader that never delegates to ambient import finders or reopens a pathname. It augments results with source-to-snapshot, prompt, executed-byte, and runtime provenance. The inference controller can read questions and selected frame provenance but has no reference-answer input; it uses no-replace atomically published phase bundles with explicit quarantine/recovery lineage, runs a six-output compatibility gate, then a complete 20-output paired run, and freezes predictions. Only after that freeze independently verifies may a new grading controller resolve the predeclared ten-file reference allowlist, create a cryptographically randomized condition-blind packet, collect two isolated C/P/W grades plus third-party disagreement adjudication, and unblind exactly once. The existing Visual Evidence claim owns the sanitized result.

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
- Require every model parameter to be BF16 on CUDA:0 and every registered buffer to be on CUDA:0. Record buffer dtypes without requiring integer or FP32 buffers to be BF16.
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
- Hold one controller-wide exclusive `flock` from the first phase-state read through publication and verification. Publish each mutating phase with dirfd-relative Linux `renameat2(..., RENAME_NOREPLACE)` (or an independently tested equivalent same-filesystem no-replace primitive) containing all evidence, sidecars, and its predecessor-bound phase record. Traverse, open, and hash staged/published entries only fd-relatively with no-follow/beneath semantics (`openat2` or an equivalently tested design), reject every symlink and non-regular/non-directory entry, retain and recheck `(st_dev, st_ino, size, mtime_ns)` identities through publication, fsync and seal the source tree, and prove that the published directory is the same held source inode before accepting it. Once a normal phase's first rename succeeds, any later error routes directly to recovery v2; do not attempt failed-transaction archival or ordinary terminal staging after a normal publication becomes uncertain. For terminal publication, one recovery state machine owns every remaining destination check, quarantine-root preparation, no-replace move, descriptor cleanup, identity/inventory check, parent fsync, and final readback; no `exists()` shortcut or generic exception may bypass it. Root observation must `lstat` without following, open a directory fd with no-follow/beneath semantics, enumerate and open children relative to that held fd, then recheck the root fd identity; dangling-symlink/wrong-kind roots reject ordinary verification and become bound recovery observations. Any error creates exactly one recovery lineage that binds the actual normal and terminal destinations plus complete observations of `.transactions`, `.failed-transactions`, and `.quarantine`. `verify` checks `recovery-required.json` before ordinary phase discovery and can never accept a bound uncertain normal name as a completed phase.
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
- The staged `stage2_runner.py` consumes exactly one JSON request batch on stdin, rejects any non-contract or label-like key, verifies canonical request hashes and mounted PNG bytes, snapshots those bytes into container-private temporary files, rewrites only the internal execution batch to those snapshots, applies seed `0`, invokes the pinned base runner in-process under complete stdout capture, and emits exactly one augmented result JSON value on stdout while retaining original request identities.
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

  Use canonical JSON (`sort_keys=True`, compact separators) for semantic hashes and byte SHA-256 for file bindings. Implement the exact Stage 1 bounded-dimension/PNG24 derivation for uniform controls. Each source must be read once into a private verified snapshot; ImageMagick `identify` and `resize` operate only on that snapshot. An end-to-end synthetic builder test uses a small real PNG and controlled ImageMagick, then reloads the manifest and independently proves source/order identity, exact expected dimensions, identify/resize argv/tool identities, PNG24 signature/color type, decoded output dimensions/hash, ten-by-twelve coverage, and 120 distinct hashes. `build_requests()` must revalidate every record one-to-one and in order against the frozen extraction manifest before joining the frozen question order to the frozen Stage 2 `uniform-bounded-manifest.json` and r4 `bounded-frame-manifest.json` by video ID; it emits `uniform-12` then `oracle-hybrid-12`, uses `actual_pts_seconds`, and binds each request with `request_sha256`.

  Write the staged wrapper only after failing wrapper-contract tests. Treat the broad `/opt/trec-base` mount as untrusted source input, never as an import root: the container working directory and initial `PYTHONPATH` exclude it, and the wrapper unconditionally rejects and removes every preloaded `trec_video_2026` package/module object regardless of claimed origin. Before any base-package execution, read exactly the allowlisted `/opt/trec-base/trec_video_2026/__init__.py`, `vqa_transformers_runner.py`, and `vqa_models.py` source bytes once with no-follow regular-file checks and verify their frozen hashes. A private loader owns those captured immutable byte payloads, constructs a new empty package namespace, compiles/executes the initializer, model/prompt module, and runner module directly from the captured bytes in dependency order, and sets/validates private immutable module specs/loaders/origins; it never calls ambient `sys.meta_path`, delegates to another finder, or reopens a source pathname. A private bytecode-free mirror may be retained for provenance but is not the execution authority. Compute emitted executable-source and prompt attestations from the loader's consumed byte payloads. The fake in-process base runner must mutate every original host frame and runtime-source path after verification and prove it can consume only container-private snapshots with the originally verified bytes. Separate integration tests call wrapper `main()` through the byte-backed runner and prove complete stdin/stdout capture, stream restoration, exactly one JSON output, consumed-byte hashes/loader identities, initializer binding, unconditional rejection of private-looking preloaded modules, immunity to a forged `MetaPathFinder` and snapshot swap/import/restore, and rejection of nonzero status, exceptions, malformed output, multiple JSON values, extra keys, and label-like canonical requests without loading a real model.

  `runner_command()` must return the literal prefix/environment/image/module argv below and an exact read-only mount allowlist. Tests compare the argv list, not a joined substring. Preflight copies the reviewed wrapper bytes into `phase-bundles/0001-preflight/runtime-snapshot/stage2_runner.py`, binds that provenance file/hash in `preflight-pass.json`, and builds one immutable argv value containing `STAGE2_WRAPPER_B64=<base64 of those exact bytes>`, `STAGE2_WRAPPER_SHA256=<frozen hash>`, and a literal bootstrap that removes those variables, validates strict base64 and the hash, injects the executed-wrapper hash into the new globals, and `exec(compile(bytes, "<frozen-stage2-wrapper>", "exec"))`. Later inference launches an immutable copy of `preflight-pass.json`'s exact `runner_argv`; it never calls `runner_command()` or reopens the wrapper snapshot for execution. The published wrapper snapshot directory remains the first read-only mount only for audit/provenance and may not appear in `PYTHONPATH` or module origins. The remaining mounts are pinned base runtime source, the ten sorted parent directories named by r4 `oracle-hybrid-12` bounded paths (mounted to the identical absolute paths), the new Stage 2 uniform-bounded root (mounted to its identical absolute path), and the one cached Qwen model repository: 14 mounts total. The controller must prove the ten treatment parents equal the selected hybrid set in the bound manifest and are disjoint from all `oracle-local-12` parents.

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
      (str(PREFLIGHT_WRAPPER_SNAPSHOT.parent), "/opt/trec-stage2-wrapper"),
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
      "-w", "/tmp",
      "-e", "PYTHONPATH=",
      "-e", "PYTHONDONTWRITEBYTECODE=1",
      "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
      "-e", "HF_DATASETS_OFFLINE=1", "-e", "PYTHONHASHSEED=0",
      "-e", f"VQA_RUNTIME_IMAGE={RUNTIME_IMAGE}", "-e", "STAGE2_SEED=0",
      "-e", f"STAGE2_WRAPPER_B64={wrapper_b64}",
      "-e", f"STAGE2_WRAPPER_SHA256={wrapper_sha256}",
      RUNTIME_IMAGE, "python", "-c", FROZEN_WRAPPER_BOOTSTRAP,
      "--model-class", "image-text", "--model-id", "Qwen/Qwen3.5-4B",
      "--revision", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
      "--max-new-tokens", "32",
  ])
  ```

  Before importing/invoking the base runner, the wrapper must call `random.seed(0)`, `numpy.random.seed(0)`, `torch.manual_seed(0)`, and `torch.cuda.manual_seed_all(0)`, then set `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True`. It must read each source path once, hash those bytes, compare them to the request, write them with exclusive creation to a container-private temporary directory, hash the snapshots, and pass only snapshot paths to the base runner. It must emit the canonical original request hash, source-to-snapshot hashes, all four seed applications, `PYTHONHASHSEED`, both backend flags, greedy flag, pixel bounds, runtime-image digest, wrapper/base-runner/prompt-source byte hashes, prompt contract ID, and original request identity.

  `validate_runner_results()` must require exact count/order, `status == "passed"`, nonempty normalized answer, pinned model/revision, `device == "cuda:0 (ROCm)"`, `precision == "bfloat16"`, a nonempty `runtime_attestation.torch_hip`, CUDA:0 parameter/buffer placement, exact in-container verified request/frame hashes, seed/runtime/source/prompt/pixel attestations, and the literal runtime image. It must reject controller-added provenance that was not emitted by the wrapper, any mismatch, or any overwrite attempt.

- [ ] **Step 4: Verify GREEN and negative boundaries**

  Add cases for altered frame bytes/hash, host-path mutation after wrapper verification, runtime source/initializer mutation after wrapper verification, private-looking preloaded modules, a forged ambient finder, snapshot swap/import/restore, full-resolution treatment path/hash, an `oracle-local-12` path, changed uniform resize rule/tool/per-frame provenance, invalid PNG24 signature/color type/dimensions, non-120-distinct condition hashes, altered canonical request, label-like/extra wrapper input, wrapper stdin/stdout/import failures, reordered result, empty answer, wrong revision, missing/wrong seed attestation, missing HIP attestation, wrong runtime/source/prompt hash, wrong pixel bound, legitimate mixed buffer dtypes, incomplete pair, reference-like input, an argv/mount outside the independently calculated exact allowlist, pairwise nested treatment parents, forbidden r4 subtrees or wrapper-directory extras being reachable, existing destination, no-replace publication races, timeouts/launch/partial-output failures, and any non-null transcript. Preserve all prior launch, pixel, provenance, label, and phase tests. Add selective-drift tests proving terminal/recovery tolerate only recorded preflight `preflight-pass.json`, its sidecar, `runtime-snapshot/stage2_runner.py`, and external live controller/argv drift while still exactly rejecting changed build/compatibility/run evidence, sidecars, directories, JSON semantics, phase-state/history, nested artifacts, and terminal upstream-hash mismatch. Add post-normal-rename failures during failed-root/archive/terminal staging and require direct recovery v2. Add dangling-symlink management roots plus held-root rename/swap during iteration and require no-follow rejection or an incomplete recovery observation with the original fd identity. Add faults before quarantine-root preparation, at every destination/quarantine/cleanup/fsync step, and after publish return before readback; require exactly one verifier-recognized incomplete lineage, never an accepted normal phase. Add root-as-regular/symlink/FIFO, real no-replace collision of every kind, multiple entries, persistent `lstat`/iteration failure, and restored-readable-state cases. Run the private test, then:

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
- Create, ignored/private under the Stage 2 run: `.transactions/`; `.quarantine/`; atomically published `phase-bundles/0001-preflight/` through the completed normal phase or one terminal bundle; and within those bundles the logical evidence (`grading-rubric.json`, `protocol.json`, preflight/pass, the durable preflight `historical-bindings.json`, `uniform-bounded-manifest.json`, `frames/bounded/uniform-12/**/*.png`, `requests.json`, compatibility predictions/pass, full predictions, prediction freeze, or terminal failure record), all SHA-256 sidecars, and one predecessor-bound `phase-state.json`. The sole emergency fallback is the exact root path `recovery-required.json`; no other recovery-marker path is permitted.
- Create, private: `.superpowers/sdd/2026-08-01-relevant-scene-stage2/task-2-report.md`

**Interfaces:**
- Consumes the reviewed Task 1 controller and the immutable r4 selection/evidence.
- Produces a verified complete 20-output prediction freeze, or a preserved incomplete/invalid stop without references.

- [ ] **Step 1: Verify the cheapest no-pull preflight**

  Verify the `data` symlink resolves exactly to `/home/npatta01/data/competitions/trec_video_2026/data`; the exact Podman digest exists without pull; Podman is rootless; `/dev/kfd` and `/dev/dri/renderD128` are accessible; the exact model snapshot and all symlink targets exist; wrapper/base-runner/prompt-source hashes match the protocol; r4 extraction and bounded manifests plus every selected treatment PNG/sidecar verify; the pinned ImageMagick executable/version identity matches the r4 resize tool; the question source matches its predeclared hash; the rubric is frozen; the exact dynamic 14-mount allowlist passes; and no Stage 2 reference/grade/transcript artifact exists. Freeze `preflight-pass.json` and predecessor-bound `historical-bindings.json` as specified above. Normal verification requires every accepted phase-state inventory entry, directory, sidecar, JSON semantic, predecessor hash, and terminal upstream hash to match exactly. Terminal/recovery verification applies a path-exact exception only when the terminal/recovery record declares implementation drift: `0001-preflight/preflight-pass.json`, its `.sha256` sidecar, and `0001-preflight/runtime-snapshot/stage2_runner.py`, plus external live controller/argv state. It must still verify the preflight phase-state, durable historical bindings/sidecar, rubric/protocol/preflight evidence, every non-exempt preflight file, and every build/compatibility/run/freeze/terminal artifact exactly. Broad `verify_accepted_evidence=False` behavior is forbidden. Every later action loads historical bindings before the live gate, enters its transaction, launches the historical argv, and routes any post-rename error according to the single-state-machine rule.

  Run controller `preflight`, then `build`. During `build`, snapshot each verified uniform source once and derive all 120 bounded controls from those snapshots with the exact frozen Stage 1 dimension/PNG24 rule. Atomically publish the build bundle containing output PNGs, their sidecars, `uniform-bounded-manifest.json` with per-record source/order/dimension/tool/argv/output provenance, protocol, requests, and phase record. Re-read and independently validate the complete bundle before request use. Verify both conditions contain 120 distinct valid PNG hashes, no full-resolution treatment path/hash appears, no `oracle-local-12` path appears, and all bundle bindings verify. After each action, require `.transactions/` to be empty and controller `verify` to accept exactly the published phase prefix while rejecting empty state, orphan files/sidecars/derived trees, missing bundle evidence, predecessor drift, or any later unrecorded artifact.

- [ ] **Step 2: Run and freeze the six-output compatibility subset**

  Run controller `compatibility`. It must select the first three panel rows and both conditions in pair order. Require 6/6 passed with exact in-container request/frame provenance, applied seed, runtime/source/prompt/pixel attestations, nonempty answers, ROCm/BF16 attestation, correct model/revision/runtime, and no label/reference mount. Freeze both the compatibility artifact and an explicit `compatibility-pass.json` bound to it, then re-run controller `verify`.

  If launch, timeout, return-code, parse, completeness, provenance, or attestation checks fail, atomically publish a terminal compatibility bundle containing attempted and completed denominators, captured stdout/stderr hashes and bounded diagnostic tails, failure class, all upstream hashes, and `references_loaded: false`; do not run full inference.

- [ ] **Step 3: Run the complete paired panel**

  Run controller `run` once for all 20 requests. Refuse pre-existing output. Preserve wall time, per-row inference time, peak allocated memory, runtime version, device, precision, and attestation. Require exactly 20/20 passed results with ten complete pairs.

  If launch, timeout, return-code, parse, completeness, provenance, or attestation checks fail, atomically publish a terminal full-run bundle with attempted/completed denominators and the same failure/upstream/reference fields. Preserve it as incomplete and do not retry in place, score, or load references.

- [ ] **Step 4: Freeze and independently verify predictions**

  Run controller `freeze`, then `verify`. `prediction-freeze.json` must bind the exact question-order source/hash, Stage 2 protocol, controller and wrapper bytes, frozen rubric, preflight result/pass, compatibility predictions/pass, phase-state predecessor, r4 protocol/selection/extraction/bounded manifest/scoreboard, frozen Stage 2 uniform-bounded manifest, requests, base-runner/prompt-source hashes, full prediction bytes, row count `20`, pair count `10`, condition counts `10/10`, and completeness `true`.

  Dispatch a fresh read-only reviewer to recompute every hash, verify the predecessor chain, request/result joins and in-container provenance/runtime attestations, scan controller/wrapper/argv/mounts for label/reference/transcript access, and write a PASS/FAIL review. Only PASS permits Task 3.

- [ ] **Step 5: Share the pre-grading checkpoint**

  Tell the user whether compatibility passed, whether the full run completed, exact denominator, elapsed wall time, and that predictions are frozen. Do not show model answers or inspect references yet.

- [ ] **Step 6: Freeze a terminal record for any incomplete branch**

  If compatibility is not 6/6 or full inference is not 20/20 with ten pairs, verify the atomically published terminal bundle contains exact attempted/completed denominators, failed check/class, preserved artifact hashes, captured process evidence, predecessor binding, and `references_loaded: false`. If any post-rename or recovery operation makes normal or terminal publication uncertain, require an exclusively created, fd-written/fsynced canonical `recovery-required.json` v2 with the exact top-level keys `schema_version`, `status`, `phase_index`, `intended_phase`, `predecessor_phase_state_sha256`, `failure_class`, `failure_message_sha256`, `references_loaded`, `normal_destination`, `attempted_terminal`, `management_roots`, `historical_upstream_bindings`, and `payload_sha256`; require `schema_version == "stage2-recovery-required-v2"`, `status == "incomplete-recovery-required"`, `references_loaded is false`, and `payload_sha256` to equal the semantic SHA-256 of the other exact fields. Destination objects use exact keys `relative_path`, `state`, `st_dev`, `st_ino`, and `inventory_sha256`, with state exactly `absent`, `quarantined`, `present-uncertain`, or `present-unreadable`. `management_roots` contains exactly three objects for `.transactions`, `.failed-transactions`, and `.quarantine`, each with exact keys `relative_path`, `kind`, `state`, `st_dev`, `st_ino`, `inventory_sha256`, and `entries`; `kind` is exactly `absent`, `directory`, `regular`, `symlink`, `fifo`, `other`, or `unknown`, and root state is exactly `absent`, `present-readable`, or `present-unreadable`. Every entry object has exact keys `relative_path`, `kind`, `state`, `st_dev`, `st_ino`, and `inventory_sha256`, using the same non-absent kind enum and state `present-readable` or `present-unreadable`. For a readable non-directory, `inventory_sha256` is the semantic hash of exact no-follow descriptor keys `kind`, `st_dev`, `st_ino`, `st_mode`, `st_size`, and `st_mtime_ns`; `entries` is empty. For a completely readable directory, `entries` is the complete path-sorted immediate-entry list and `inventory_sha256` is its semantic hash. For whole-root or mid-iteration failure, state is `present-unreadable`, `entries` contains only the successfully observed path-sorted prefix, and `inventory_sha256` hashes exact keys `root_descriptor`, `observed_entries`, and `observation_complete: false`; unavailable scalar values are null. Any unreadable root/entry forces verifier output `lineage_fully_verified: false`; readable observations must recompute exactly. `historical_upstream_bindings` is the durable predecessor-bound object, not live recomputation. `verify` checks the marker before phase discovery, validates its payload/historical binding, rejects unbound/restored/extra readable state, and returns only an incomplete non-resumable result with explicit `lineage_fully_verified`; it never interprets an uncertain normal name as a phase. Skip Task 3 and branch directly to Task 4's incomplete-report path with either the verified terminal bundle or verified recovery-required lineage; do not stop before that reporting/portal workflow finishes.

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
- Consumes either `(frozen Stage 1 scoreboard + verified terminal failure record or verified recovery-required lineage)` for the incomplete path or `(frozen Stage 1 scoreboard + verified prediction freeze + blind grades/adjudication + independently verified final evaluation)` for the complete path.
- Produces one sanitized answering result under the existing Visual Evidence claim and a user-facing result before any transcript work.

- [ ] **Step 1: Update authored report sources only**

  For a complete verified run, record extraction gate, representation gate, 20/20 inference denominator, compatibility outcome, runtime, control/treatment C/P/W, ordinal scores, exact paired transitions, improved/regressed rows, ordinal gain, answer-gate result, evidence validity, decision disposition, immutable private artifact hashes, limitations, and next decision. For a compatibility failure, partial full run, or recovery-required lineage, record only the completed denominator when known, failed gate/recovery state, preserved terminal-or-recovery marker hash, and **Incomplete** validity; state that references were not loaded and do not report C/P/W, ordinal scores, or an answer-gate conclusion. In every branch state explicitly that no transcript was used or run. Do not copy private questions, references, answers, grades, IDs, paths, frames, media, or logs.

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
