# TimeLens2 Selector Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a blinded, matched 10-video uniform-12 versus TimeLens2 hybrid-12 quality gate and record a valid promotion or negative disposition.

**Architecture:** A label-blind contract module binds the existing frozen temporal-diagnostic inference manifest to exactly ten media files, runs the selector, constructs and extracts two matched frame recipes, and freezes twenty Qwen answers. A separate local evaluator receives only the verified freeze and the accepted relevant-scene adjudication after inference, then computes component coverage and consumes condition-blind C/P/W judgments. A pinned dstack wrapper runs CUDA inference and incrementally persists private artifacts to Hugging Face Bucket storage.

**Tech Stack:** Python 3.11, pytest, PyTorch CUDA 2.8.0 image, Transformers 4.57.6, PyAV 16.0.1, imageio-ffmpeg 0.6.0, dstack, Hugging Face CLI 1.26.0.

## Global Constraints

- Branch: `codex/timelens2-selector-pilot` in the existing Codex-managed worktree.
- Selector: `MCG-NJU/TimeLens2-2B@90dd30b3eeff3520a30c830eff2209f4ad788ca9`.
- Answer model: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime: `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755` on NVIDIA CUDA only.
- Quality run: `timelens2-selector-quality-20260801-a1`; `-a2` is allowed only for one corrective technical attempt.
- Labels: only `temporal-relevant-scene-oracle-10/annotations/adjudicated.json` at SHA-256 `774326c801b85ac3e542c29be44040630b6ba1afadbfa94971366b425c08de98` and accepted sidecars; never use temporal-required-moment or `-r2`.
- Sampling: no padding; uniform-12 versus six global plus six normalized-interval midpoint frames using Hamilton allocation.
- Freeze selector recipes and all twenty answers before label access.
- Base allocation is $4; $0.6580 was already spent, leaving $3.3420. Do not use the central reserve.
- Private files stay ignored and durable artifacts stay under `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/<run-name>/`.
- Do not edit shared project-status QMD or generated Markdown/HTML.

---

### Task 1: Label-blind quality contract and frame recipes

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/timelens2_quality.py`
- Create: `tests/test_timelens2_quality.py`
- Modify: `src/trec_video_2026/vqa_experiments/timelens2_selector.py`

**Interfaces:**
- Consumes: `build_hybrid_timestamps()`, strict raw/normalized selector schemas, and the frozen temporal-diagnostic inference manifest.
- Produces: `prepare_quality_requests(...)`, `build_matched_recipes(...)`, `freeze_quality_inference(...)`, and `QualityEvidenceStore`.

- [ ] **Step 1: Write failing source-binding tests**

```python
def test_prepare_quality_requests_binds_exact_frozen_ten_rows(tmp_path):
    prepared = prepare_quality_requests(
        source_path=source,
        dataset_root=dataset,
        output_path=output,
        expected_source_sha256=SOURCE_SHA256,
    )
    assert prepared["schema_version"] == "timelens2-quality-requests-v1"
    assert len(prepared["requests"]) == 10
    assert "correct_answer" not in json.dumps(prepared)

@pytest.mark.parametrize("mutation", ["source_hash", "video_id", "question", "media_hash", "duration", "duplicate"])
def test_prepare_quality_requests_rejects_frozen_identity_mutation(tmp_path, mutation):
    with pytest.raises(ValueError, match="frozen ten-row source"):
        prepare_quality_requests(...)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_timelens2_quality.py`

Expected: collection fails because `timelens2_quality` does not exist.

- [ ] **Step 3: Implement the minimal request contract**

Implement an exact constant for the complete frozen source-file SHA-256. That
digest cryptographically binds all ten private identities, questions, media
hashes, and durations without tracking those values individually. Reject
nested label keys, require ten distinct identities, resolve each media file
under the canonical dataset root, verify file hashes, and write a canonical
request hash per row. The serialized request contains questions but no
answers, labels, oracle intervals, or distractors.

- [ ] **Step 4: Write failing matched-recipe and freeze tests**

```python
def test_build_matched_recipes_uses_no_padding_and_twelve_distinct_frames():
    recipes = build_matched_recipes(duration_seconds=12.0, normalized_intervals=[[3.0, 6.0]])
    assert len(recipes["uniform-12"]) == 12
    assert len(recipes["hybrid-12"]) == 12
    assert all(3.0 < value < 6.0 for value in recipes["hybrid_local"])

def test_freeze_quality_inference_requires_ten_selector_and_twenty_answer_rows(tmp_path):
    with pytest.raises(ValueError, match="20 complete answers"):
        freeze_quality_inference(requests, selectors, recipes, answers[:-1], tmp_path / "freeze.json")
```

- [ ] **Step 5: Run the new tests and verify RED**

Run: `pytest -q tests/test_timelens2_quality.py -k 'recipe or freeze'`

Expected: failures because recipe/freeze functions are absent.

- [ ] **Step 6: Implement matched recipes and generic selector freeze**

Generalize selector freezing without changing the three-row compatibility contract. Sort the combined hybrid timestamps chronologically, reject duplicates, retain separate global/local provenance, and hash requests, raw selectors, normalized selectors, recipes, frame manifests, answers, and the completion denominator in the inference freeze.

- [ ] **Step 7: Implement incremental evidence persistence**

`QualityEvidenceStore(output_dir, required_videos=10, required_answers=20)` writes `phase-status.json`, selector attempts, answer attempts, partial outputs, and final denominators after each phase and in `finally`. Setup/OOM/mid-panel failures must remain inspectable.

- [ ] **Step 8: Verify GREEN and commit**

Run: `pytest -q tests/test_timelens2_selector.py tests/test_timelens2_quality.py`

Commit: `feat: add TimeLens2 quality gate contract`

---

### Task 2: CUDA inference workload

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/timelens2_quality.py`
- Modify: `tests/test_timelens2_quality.py`

**Interfaces:**
- Consumes: ten validated requests, two local model directories, the existing TimeLens message/parser functions, and `vqa_transformers_runner.generate_answer()`.
- Produces: `run_quality_inference(...)` and private frozen selector/frame/answer artifacts.

- [ ] **Step 1: Write failing extraction and phase-order tests**

```python
def test_extract_recipe_records_exact_requested_and_decoded_identity(tmp_path):
    manifest = extract_recipe_frames(media, recipe, output, ffmpeg_executable=fake_ffmpeg)
    assert manifest["requested_timestamps_seconds"] == recipe
    assert len({row["sha256"] for row in manifest["frames"]}) == 12

def test_quality_inference_freezes_before_any_label_path_can_be_supplied():
    assert "adjudication" not in inspect.signature(run_quality_inference).parameters
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_timelens2_quality.py -k 'extract or inference'`

- [ ] **Step 3: Implement exact frame extraction**

Use the pinned `imageio-ffmpeg` executable with `-ss <timestamp> -i <media> -map 0:v:0 -frames:v 1`. Extract the union of timestamps once per video, hash every PNG, reject missing/duplicate decoded images, and produce condition manifests that reference the same cached global frames where timestamps coincide.

- [ ] **Step 4: Implement sequential model execution**

Load TimeLens once, run all ten selector requests, freeze raw/normalized results and recipes, release the model and CUDA cache, then load Qwen3.5-4B once and run twenty matched frames-only answers with greedy decoding, 32 new-token maximum, 256²–512² pixels, and bfloat16 CUDA attestation. Append attempts after every row.

- [ ] **Step 5: Implement terminal freeze/failure behavior**

Only a 10/10 selector, 20/20 answer result writes `inference-freeze.json` with `label_access: permitted_after_verified_freeze`. Every exception writes a sanitized failure record and returns nonzero after attempting durable artifact sync.

- [ ] **Step 6: Verify GREEN and commit**

Run: `pytest -q tests/test_timelens2_quality.py tests/test_vqa_transformers_runner.py`

Commit: `feat: run blinded TimeLens2 matched inference`

---

### Task 3: Pinned dstack task and durable artifact wrapper

**Files:**
- Create: `dstack/timelens2-selector-quality-10.dstack.yml`
- Create: `scripts/run-timelens2-selector-quality-dstack.sh`
- Create: `tests/test_timelens2_quality_dstack_config.py`

**Interfaces:**
- Consumes: private input prefix `<run>/input`, label-blind quality module, dstack secret `hf_token`.
- Produces: detached CUDA run and checksummed `<run>/output` payload.

- [ ] **Step 1: Write failing config tests**

```python
def test_quality_task_is_pinned_cuda_and_budget_bounded():
    text = CONFIG.read_text()
    assert "name: timelens2-selector-quality-20260801-a1" in text
    assert f"image: {IMAGE}" in text
    assert f"gpu: {GPU_POOL}:1:48GB.." in text
    assert "max_price: 1.00" in text
    assert "max_duration: 2h" in text
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert "adjudicated.json" not in text
```

- [ ] **Step 2: Run config tests and verify RED**

Run: `pytest -q tests/test_timelens2_quality_dstack_config.py`

- [ ] **Step 3: Add the pinned task**

Use A40/A6000/L40/L40S/RTX6000Ada/A100/H100/H200, one GPU with at least 48 GB VRAM, 8+ CPUs, 48+ GB host memory, open-ended 100+ GB disk, RunPod/Vast.ai on-demand, `$1.00/h` maximum, two-hour hard duration, and no external retry loop.

- [ ] **Step 4: Add the wrapper**

Validate exact image/model/run/URI values, unset `HF_TOKEN` after copying it to a non-exported shell variable, install exact binary-wheel dependencies, verify NVIDIA CUDA and no ROCm, verify the remote input checksum manifest, download both pinned models, run foreground inference, write `SHA256SUMS`, and foreground-sync output even after workload failure.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest -q tests/test_timelens2_quality_dstack_config.py tests/test_timelens2_dstack_config.py`

Run: `bash -n scripts/run-timelens2-selector-quality-dstack.sh`

Commit: `feat: configure TimeLens2 quality run`

---

### Task 4: Private input staging, preview, and CUDA execution

**Files:**
- Create ignored: `.timelens2-private/timelens2-selector-quality-20260801-a1/`
- Modify only if a corrective technical attempt is required: config/allowlist for `-a2`

**Interfaces:**
- Consumes: frozen label-free request manifest and the exact ten media files.
- Produces: private HF input/output/audit prefixes and dstack run evidence.

- [ ] **Step 1: Generate and verify private inputs**

Run the request-preparation CLI, confirm the ten media hashes, write `SHA256SUMS`, scan recursively for forbidden label keys, and sync only the label-free request file, media files, and checksum manifest to `<run>/input` using the authenticated HF CLI without printing credentials.

- [ ] **Step 2: Verify remote input immutability**

List the remote prefix, download it to a fresh `mktemp -d` directory, and run `sha256sum -c SHA256SUMS`.

- [ ] **Step 3: Preview and retain complete output**

Run: `echo "n" | dstack apply -f dstack/timelens2-selector-quality-10.dstack.yml`

Require multiple physically distinct compatible offers when practical; save the complete declined preview under the ignored private audit directory.

- [ ] **Step 4: Submit once, detached**

Run: `dstack apply -f dstack/timelens2-selector-quality-10.dstack.yml --detach`

Do not use diagnose because the run carries a secret. Monitor ordinary status/logs; do not blindly resubmit. Permit one `-a2` only for a diagnosed technical correction.

- [ ] **Step 5: Collect cost and artifacts**

Record selected backend/location/GPU/offer price, actual cost, duration, exit status, and completion denominator. List/download/verify `<run>/output`; sync the declined preview, submission transcript, sanitized status, and ordinary log to `<run>/audit` with its own checksum manifest.

---

### Task 5: Post-freeze evaluation and sanitized evidence

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/timelens2_quality.py`
- Modify: `tests/test_timelens2_quality.py`
- Modify: `docs/experiments/2026-07-30-timelens2-selector-pilot.md`

**Interfaces:**
- Consumes: verified `inference-freeze.json`, frozen answers/recipes, accepted adjudication, and private condition-blind judgments.
- Produces: private evaluation plus sanitized aggregate evidence and final quality disposition.

- [ ] **Step 1: Write failing evaluation-boundary tests**

```python
def test_evaluate_refuses_labels_before_verified_inference_freeze(tmp_path):
    with pytest.raises(ValueError, match="verified inference freeze"):
        evaluate_quality_gate(incomplete_run, adjudication, judgments)

def test_decision_requires_coverage_improvements_and_regression_limit():
    assert quality_decision(completed=10, covered=8, improved=2, regressed=1) == "pass"
    assert quality_decision(completed=10, covered=7, improved=2, regressed=1) == "valid-negative"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_timelens2_quality.py -k 'evaluate or decision'`

- [ ] **Step 3: Implement coverage and blinded grading packet**

Verify the accepted annotation checksum only after the inference freeze. Compute component coverage by testing each extracted frame timestamp against every sufficient component interval in the accepted row. Build a randomized condition-blind answer packet containing question, reference answer, and two anonymized answers; store it only under `.timelens2-private/`.

- [ ] **Step 4: Complete answer-only C/P/W adjudication**

Grade each anonymized answer as correct, partial, or wrong using the repository's answer-only rubric; freeze judgments before unblinding condition IDs. Compute paired ordinal transitions where `W < P < C`, improvement/regression counts, and both condition totals.

- [ ] **Step 5: Apply the predeclared decision**

Pass only for 10/10 completed rows, all required components covered in at least eight rows, two or more improvements, and at most one regression. Otherwise record a valid negative result; never repeat a technically valid run for quality.

- [ ] **Step 6: Fresh verification and privacy scan**

Run: `pytest -q tests/test_timelens2_selector.py tests/test_timelens2_quality.py tests/test_timelens2_dstack_config.py tests/test_timelens2_quality_dstack_config.py tests/test_vqa_transformers_runner.py`

Run: `git diff --check`

Run tracked-content scans for `.timelens2-private`, panel IDs, raw outputs, label keys, media extensions, and credential patterns. Confirm only sanitized aggregate evidence is staged.

- [ ] **Step 7: Commit final evidence**

Commit: `docs: record TimeLens2 selector quality result`

---

## Execution outcome

Status: **complete with a non-promotion disposition and a persistent technical
error in the matched-answer phase**.

- `timelens2-selector-quality-20260801-a1` selected a RunPod A40 48 GB
  node at $0.44/hour. It preserved 7/10 selector attempts before exit 137;
  the evidence points to host-memory exhaustion, not GPU-memory exhaustion.
- The sole corrective run, `timelens2-selector-quality-20260801-a2`, raised
  minimum host memory from 48 GB to 96 GB. It selected a Vast.ai RTX 6000 Ada
  48 GB node at $0.82074/hour and completed all 10 selector rows.
- The corrected run covered every required component in 7/10 rows and 14/17
  components overall. This misses the predeclared minimum of 8/10 rows, so
  TimeLens2 hybrid-12 is not promoted and no 30-video expansion is warranted.
- Matched answer grading remains incomplete: both runs produced 0/20 answers.
  The corrected run stopped during frame extraction because distinct requested
  timestamps decoded to identical frame bytes. The one allowed corrective
  attempt was already consumed, so no further infrastructure run was made.
- Actual incremental cost was $0.1113, bringing total TimeLens2 spend to
  $0.7693 of the $4 base allocation. The central reserve was not used.
- Private inputs, outputs, logs, and evaluation payloads remain excluded from
  Git. Durable Hugging Face Bucket copies were listed, downloaded, and
  checksum-verified; only sanitized aggregate evidence is committed.
