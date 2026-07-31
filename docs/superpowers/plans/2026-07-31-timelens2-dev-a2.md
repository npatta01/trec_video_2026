# TimeLens2 CUDA Dev A2 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the separately authorized TimeLens2 corrective compatibility gate in a dstack CUDA dev environment with forced torchvision and durable private evidence.

**Architecture:** Reuse the already verified exact-image no-media canary and committed compatibility runner. Create one ignored immutable dev configuration that changes only the diagnosed admission constraint from exact `50GB` to open-ended `100GB..`, runs the three-video media preflight before model download, and continues to compatibility only when preflight succeeds. Persist workload and failure artifacts to a fresh Hugging Face Bucket prefix, then stop the dev environment after capture.

**Tech Stack:** dstack 0.20.29, RunPod/Vast.ai, pinned PyTorch CUDA image, TimeLens2-2B, qwen-vl-utils 0.0.14, torchvision, Hugging Face CLI 1.26.0.

## Global Constraints

- Branch is `codex/timelens2-selector-pilot`; private config, request manifest, media, and logs remain ignored.
- Run name is immutable `timelens2-selector-pilot-20260730-a2`; no further external submission loop is allowed.
- Model is `MCG-NJU/TimeLens2-2B@90dd30b3eeff3520a30c830eff2209f4ad788ca9`.
- Image is `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- CUDA only; compatible GPU pool is A40/A6000/L40/L40S/RTX6000Ada/A100/H100/H200 with at least 48 GB VRAM.
- Use on-demand RunPod/Vast.ai through `gpu-marketplace`, `max_price: 1.00`, `max_duration: 45m`, and bounded native `no-capacity` retry only.
- Use `disk: 100GB..` with no upper disk bound.
- `HF_TOKEN` comes only from dstack secret `hf_token`; never print it or use secret-bearing diagnose output.
- Upload exactly the frozen three-video compatibility panel and label-free manifest; never consult oracle labels.
- Durable output prefix is `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/timelens2-selector-pilot-20260730-a2/` and must be empty before submission.

---

### Task 1: Freeze and preview the dev environment

**Files:**
- Create ignored: `.timelens2-private/timelens2-selector-pilot-20260730-a2/timelens2-selector-pilot-20260730-a2.dstack.yml`
- Modify: `tests/test_timelens2_dstack_config.py`
- Modify: `scripts/run-timelens2-selector-dstack.sh`
- Read: `dstack/timelens2-selector-pilot.dstack.yml`

**Interfaces:**
- Consumes: committed runner, frozen request manifest, three exact media paths, `gpu-marketplace`.
- Produces: a full declined-preview transcript and one immutable dev configuration.

- [x] **Step 1: Verify local and private-input invariants**

Run branch/status, exact data-symlink, ignore, manifest validation, byte-count, remote-prefix-empty, and existing-run-name checks. Stop on any conflict.

- [x] **Step 2: Permit only the authorized A2 identity test-first**

Add a subprocess test for the runner's side-effect-free `validate` command. Verify RED because `timelens2-selector-pilot-20260730-a2` is not yet accepted, then minimally allow A1/A2 compatibility IDs while continuing to reject other suffixes and run the focused tests GREEN.

- [ ] **Step 3: Create the ignored configuration**

Use `type: dev-environment`, the pinned image, the same runner inputs/environment, `FORCE_QWENVL_VIDEO_READER=torchvision`, `fleets: [gpu-marketplace]`, `gpu: A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:48GB..`, and `disk: 100GB..`. Run the committed wrapper as the foreground `init` workload.

- [ ] **Step 4: Validate offers and preview**

Run a read-only `dstack offer` through `gpu-marketplace`, then `echo "n" | dstack apply -f <config>`. Save the complete preview unchanged and require at least two physically distinct compatible offers at or below `$1.00/h`.

### Task 2: Submit once and capture the workload

**Files:**
- Private output: Hugging Face prefix for `timelens2-selector-pilot-20260730-a2`.

**Interfaces:**
- Consumes: accepted Task 1 preview.
- Produces: one detached dstack run and its foreground preflight/compatibility artifacts.

- [ ] **Step 1: Submit once detached**

Run `dstack apply -f <config> -y -d` exactly once and retain the full submission output.

- [ ] **Step 2: Monitor normal status and logs**

Use `dstack ps -v`, allowlisted `dstack run get ... --json`, and ordinary `dstack logs`. Do not use diagnose output. Stop if infrastructure/runtime admission fails; never externally resubmit.

- [ ] **Step 3: Stop the dev environment after capture**

When the runner has uploaded terminal status and checksums, stop the dev environment with `dstack stop timelens2-selector-pilot-20260730-a2 -y` to cap cost.

### Task 3: Verify evidence and commit the sanitized record

**Files:**
- Modify: `docs/experiments/2026-07-30-timelens2-selector-pilot.md`
- Modify: `docs/superpowers/plans/2026-07-31-timelens2-dev-a2.md`

**Interfaces:**
- Consumes: immutable remote artifact prefix and dstack cost/status evidence.
- Produces: checksum-verified private download plus a sanitized committed result.

- [ ] **Step 1: List, download, and checksum-verify artifacts**

Use pinned HF CLI, download to the ignored A2 directory, verify `SHA256SUMS`, and record the checksum of the checksum manifest. Keep raw outputs and logs private.

- [ ] **Step 2: Classify the result**

Record preflight and compatibility completion denominators separately. Do not grade quality or run the focused oracle-dependent pilot unless all compatibility requirements independently make it eligible.

- [ ] **Step 3: Verify and commit**

Run model-specific tests, full tests if code changed, Ruff, Bash syntax, `git diff --check`, data/private tracking checks, and credential-pattern checks. Commit only this plan and the uniquely owned sanitized evidence record unless a test-first code correction was required.
