# TimeLens2 Three-Condition Frozen-30 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and grade a label-isolated frozen-30 comparison of uniform-12, hybrid-12, and combined-18 using pinned TimeLens2 and Qwen models.

**Architecture:** A new TimeLens-specific module prepares an exact 30-row label-free source, constructs slot-based three-condition recipes, resolves only colliding local slots, freezes 90 answers, and creates a condition-blind grading packet. Separate digest-pinned dstack canary and experiment tasks use immutable Hugging Face Bucket prefixes; evaluation runs locally only after the remote inference freeze verifies.

**Tech Stack:** Python 3.11, pytest, PyTorch CUDA image, corrected Transformers
`5.15.0.dev0@cbf4d720ec734edb77d25452b2790c7e4be2f8d7`, qwen-vl-utils
0.0.14, PyAV 16.0.1, imageio-ffmpeg 0.6.0, dstack, and Hugging Face CLI
1.26.0.

**Final status (2026-08-01):** A test-first A3 correction passed the exact-image
canary with 1/1 selectors and 2/2 Qwen answers. Frozen A1 preserved 26/30
selectors before exposing a 50 ms FFprobe/PyAV duration-consistency defect.
Corrective frozen A2 accepted 29/30 strict interval results; one TimeLens output
was out of duration, so frame extraction and all 90 answers were withheld.
Labels were never opened. Selector compatibility was not promoted and matched
answer quality remains incomplete. See
`docs/experiments/2026-08-01-timelens2-three-condition-frozen30.md`.

## Global Constraints

- Work only on `codex/timelens2-selector-pilot` in the existing linked worktree.
- Source is `configs/vqa/native_audio_benchmark_30_inference.json@12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`.
- Selector is `MCG-NJU/TimeLens2-2B@90dd30b3eeff3520a30c830eff2209f4ad788ca9`.
- Answer model is `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Image is `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- Durable artifact IDs reached `timelens2-three-condition-canary-20260801-a3` and `timelens2-three-condition-frozen30-20260801-a2`; the exact immutable A1 input is reused by frozen A2. Valid dstack resource names use the matching shortened `tl2-3cond-*` identities because dstack limits resource names to 41 characters.
- Use CUDA only, one A40-or-better 48 GB GPU, at least 96 GB RAM, on demand, and `max_price: 1.00`. Final TimeLens2 spend is `$1.0902 / $4.00`.
- Keep labels out of inference, keep `HF_TOKEN` only in dstack secrets, never use secret-bearing diagnose output, and keep `.timelens2-private/` ignored.
- Do not edit shared project-status QMD or generated Markdown/HTML.

---

### Task 1: Three-condition frame and freeze contracts

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/timelens2_three_condition.py`
- Create: `tests/test_timelens2_three_condition.py`

**Interfaces:**
- Consumes: `build_hybrid_timestamps`, strict TimeLens schemas, `generate_answer`, and an injected frame decoder/source-PTS provider.
- Produces: `prepare_frozen30_source`, `build_three_condition_recipes`, `extract_three_condition_frames`, `freeze_three_condition_inference`, `build_blind_grading_packet`, and `score_blind_judgments`.

- [ ] **Step 1: Write RED contract tests**

Add literal-fixture tests proving: exactly 30 label-free rows are required; source/media hashes bind; the recipes contain 12/12/18 slots; combined-18 reuses all uniform slots plus the six hybrid local slots; shared frames across conditions are allowed; a within-condition local collision moves only that local slot to the nearest unused interval-interior PTS; incomplete 30/90 freezes fail; and blind judgments must contain one C/P/W grade per row/condition.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_timelens2_three_condition.py`. Expected: collection fails because `timelens2_three_condition` does not exist.

- [ ] **Step 3: Implement minimal contracts**

Use slot objects with `slot_id`, `role`, `requested_timestamp_seconds`, and optional `interval_seconds`. Reuse a uniform slot identity between uniform-12 and combined-18. Share each local slot identity between hybrid-12 and combined-18. Decode each slot once, enforce distinct decoded hashes within each condition, and resolve only local slots.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_timelens2_three_condition.py tests/test_timelens2_quality.py tests/test_timelens2_selector.py`. Expected: all pass.

- [ ] **Step 5: Commit**

Commit the module and tests as `feat: add TimeLens three-condition contracts` after `git diff --check`.

---

### Task 2: Runtime entrypoints and dstack safety contracts

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/timelens2_three_condition.py`
- Create: `scripts/run-timelens2-three-condition-canary-dstack.sh`
- Create: `scripts/run-timelens2-three-condition-dstack.sh`
- Create: `dstack/timelens2-three-condition-canary.dstack.yml`
- Create: `dstack/timelens2-three-condition-frozen30.dstack.yml`
- Create: `tests/test_timelens2_three_condition_dstack.py`

**Interfaces:**
- Canary command: `python -m trec_video_2026.vqa_experiments.timelens2_three_condition canary --selector-model-dir PATH --answer-model-dir PATH --output-dir PATH --ffmpeg PATH`.
- Experiment command: `... run --requests PATH --selector-model-dir PATH --answer-model-dir PATH --output-dir PATH --ffmpeg PATH`.

- [ ] **Step 1: Write RED runtime/config tests**

Parse both YAML files and assert immutable names, digest image, exact revisions, 48 GB GPU floor, 96 GB RAM floor, on-demand RunPod/Vast.ai marketplace policy, `$1.00/h` cap, 30m/2h durations, native bounded no-capacity retry, code-only dstack file mounts, secret indirection, and distinct immutable HF URIs. Execute each runner's `validate` mode with a dummy token and exact expected environment, asserting zero exit without network access.

- [ ] **Step 2: Verify RED**

Run `pytest -q tests/test_timelens2_three_condition_dstack.py`. Expected: collection or file-not-found failure because the configs and runners do not exist.

- [ ] **Step 3: Implement canary and experiment entrypoints**

The canary generates local synthetic video/frames, downloads exact revisions through the secret-bearing runner, loads/unloads the selector, then loads Qwen and generates once with 12 and once with 18 images. The experiment validates 30 requests, runs and persists selector attempts incrementally, builds/extracts all recipes, unloads the selector, runs 90 Qwen answers incrementally, and finalizes failure evidence in all cases. Both foreground workloads checksum and sync outputs while preserving workload and sync return codes.

- [ ] **Step 4: Verify GREEN**

Run `pytest -q tests/test_timelens2_three_condition_dstack.py tests/test_timelens2_three_condition.py tests/test_timelens2_quality_dstack_config.py`. Expected: all pass.

- [ ] **Step 5: Commit**

Commit runtime/config changes as `feat: configure TimeLens three-condition run` after model-specific tests and `git diff --check`.

---

### Task 3: Private input bundle and exact-image canary

**Files:**
- Private only: `.timelens2-private/timelens2-three-condition-frozen30-20260801-a1/input/`
- Private only: `.timelens2-private/timelens2-three-condition-canary-20260801-a1/audit/`

**Interfaces:**
- Prepared source contains exactly 30 rows with request identity, question, category, duration, media hash, and container media path; `SHA256SUMS` excludes itself.

- [ ] **Step 1: Build and verify the private source**

Run the module's `prepare` command against the label-free manifest and canonical data root. Verify no forbidden keys, exactly 30 distinct media files, safe relative paths, member hashes, total byte count, and local checksums.

- [ ] **Step 2: Upload immutable input**

Require the run input prefix to be empty, sync only the prepared source, media, and checksum manifest, then list, download to a fresh ignored directory, and run `sha256sum -c SHA256SUMS`.

- [ ] **Step 3: Preview canary**

Run `echo "n" | dstack apply -f dstack/timelens2-three-condition-canary.dstack.yml`, retain the complete output privately, and require multiple physically distinct compatible offers when practical.

- [ ] **Step 4: Submit and verify canary**

Submit once with `-y -d`, monitor with normal logs/status, then list, download, and checksum-verify the canary artifacts. Stop on a repeated technical failure after at most one corrective `-a2`.

---

### Task 4: Frozen-30 inference and post-freeze grading

**Files:**
- Private only: `.timelens2-private/timelens2-three-condition-frozen30-20260801-a1/`
- Create after grading: `docs/experiments/2026-08-01-timelens2-three-condition-frozen30.md`

**Interfaces:**
- Durable prefix: `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-three-condition-frozen30-20260801-a1/`.
- Completion denominator: 30 selector rows and 90 answer rows.

- [ ] **Step 1: Preview and submit**

After canary success, decline-preview the real config, retain full output, verify offers and maximum exposure, then submit exactly once detached.

- [ ] **Step 2: Monitor and verify artifacts**

Use normal logs and allowlisted run status only. After termination, list and download the entire durable prefix to a fresh ignored directory and verify every checksum. Preserve incomplete evidence if any phase fails.

- [ ] **Step 3: Open labels only after freeze verification**

Require a frozen 30/90 denominator and matching source/model/runtime hashes before loading `configs/vqa/native_audio_benchmark_30.json`. Create an anonymized deterministic grading packet, freeze it, assign all 90 C/P/W judgments without condition identities, freeze judgments, then reveal the mapping and compute paired transitions.

- [ ] **Step 4: Apply the predeclared decision**

Report each condition's C/P/W counts, hybrid and combined improvements/regressions versus uniform, completion, latency/memory, cost, and whether either crosses the `4 improvements / <=2 regressions / net >=3 / correct non-decrease` rule. Do not repeat a technically valid negative run.

- [ ] **Step 5: Commit sanitized evidence**

Write one evidence record containing aggregates and immutable URIs/checksums only. Run privacy scans, model-specific and relevant VQA tests, `git diff --check`, then commit as `docs: record TimeLens three-condition result`.

---

### Task 5: Final verification

**Files:**
- Verify all changed tracked files; do not edit shared status QMD or generated outputs.

- [ ] **Step 1: Run fresh verification**

Run `pytest -q tests/test_timelens2_three_condition.py tests/test_timelens2_three_condition_dstack.py tests/test_timelens2_quality.py tests/test_timelens2_selector.py tests/test_vqa_transformers_runner.py`, `git diff --check`, a tracked/private-path scan, and a credential-pattern scan.

- [ ] **Step 2: Verify Git and privacy state**

Confirm `.timelens2-private/` is ignored, no raw media/manifest/log/answer/judgment file is tracked, and the branch is clean.

- [ ] **Step 3: Handoff**

Return branch and commit SHA; canary/run names; exact model/image revisions; offer and actual-cost evidence; completion denominator; quality disposition; private artifact URI and checksum verification; files changed; tests; and any persistent errors.
