# TimeLens2 Decoder Dev Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start one persistent exact-image CUDA dstack dev environment and identify a reproducible decoder configuration that reads the frozen three-video TimeLens2 compatibility gate.

**Architecture:** Provision an interactive dev environment with no fallible initialization workload, attach to it, and run one-variable decoder experiments in the same node and container. Keep model credentials and weights out of this phase; only after all three private videos decode will a later workload be allowed to exercise model inference.

**Tech Stack:** dstack dev environments, pinned PyTorch CUDA 12.8 image, Python, PyAV, torchvision, FFmpeg, RunPod/Vast.ai GPU marketplace.

## Global Constraints

- Use branch `codex/timelens2-selector-pilot` and preserve the ignored `.timelens2-private/` boundary.
- Use image `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- Use CUDA only and GPUs with at least 48 GB VRAM from A40, A6000, L40, L40S, RTX6000Ada, A100, H100, or H200.
- Allow disk `100GB..`, memory `48GB..`, maximum price `$1/hour`, and maximum duration `2h`.
- Preview the exact config with `echo "n" | dstack apply -f ...` and retain the complete output before detached submission.
- Do not put `HF_TOKEN` on the debug node, download model weights, expose services, or use competition labels.
- Upload only the frozen three-video compatibility media panel; keep all private media and raw logs out of Git.
- Make at most one corrected infrastructure submission if the first dev environment cannot start.

---

### Task 1: Freeze and provision the interactive dev environment

**Files:**
- Create (ignored): `.timelens2-private/timelens2-decoder-dev-20260801-d1/timelens2-decoder-dev-20260801-d1.dstack.yml`
- Create (ignored, generated): `.timelens2-private/timelens2-decoder-dev-20260801-d1/declined-preview.txt`
- Create (ignored, generated): `.timelens2-private/timelens2-decoder-dev-20260801-d1/submission.txt`

**Interfaces:**
- Consumes: the frozen A1 compatibility manifest and its exact three referenced media files.
- Produces: an SSH-accessible dev environment named `timelens2-decoder-dev-20260801-d1`.

- [x] **Step 1: Verify immutable inputs and capacity**

Run the repository manifest validation tests, verify the shared data symlink, confirm the run name is unused, and request marketplace offers for the exact resource pool.

- [x] **Step 2: Create a no-init dev config**

Create a `type: dev-environment` config that maps the adapter source, manifest, and three media files into `/dstack/run`, but has no `init` command and no secrets.

- [x] **Step 3: Preview and retain the complete declined preview**

Run: `echo "n" | dstack apply -f .timelens2-private/timelens2-decoder-dev-20260801-d1/timelens2-decoder-dev-20260801-d1.dstack.yml |& tee .timelens2-private/timelens2-decoder-dev-20260801-d1/declined-preview.txt`

Expected: multiple compatible physical offers when marketplace capacity permits.

- [x] **Step 4: Submit once in detached mode and wait for running**

Run: `dstack apply -y -d -f .timelens2-private/timelens2-decoder-dev-20260801-d1/timelens2-decoder-dev-20260801-d1.dstack.yml |& tee .timelens2-private/timelens2-decoder-dev-20260801-d1/submission.txt`

Expected: the run reaches `running` and remains available for SSH debugging.

### Task 2: Reproduce and isolate the CUDA decoder failure

**Files:**
- Create (ignored, generated): `.timelens2-private/timelens2-decoder-dev-20260801-d1/diagnostics/`

**Interfaces:**
- Consumes: the live dev SSH alias and `/dstack/run/input/compatibility-requests.json`.
- Produces: command transcripts identifying the first failing layer and a three-video decoder result with an explicit denominator.

- [x] **Step 1: Capture baseline system and runtime facts**

Record `nvidia-smi`, memory, process/thread limits, FFmpeg version, Python version, and initial Python package versions without printing environment secrets.

- [x] **Step 2: Reproduce the A2 dependency state**

Install the same lightweight runtime dependency revisions used by A2, then execute the production torchvision preflight against all three frozen videos.

Expected: either 3/3 pass or the same FFmpeg/swscaler failure appears before any model download.

- [x] **Step 3: Test the lowest decoder layer**

For each frozen video, compare direct single-thread FFmpeg decode, PyAV decode, and forced torchvision decode while holding the node and media constant.

Expected: identify whether failure begins in FFmpeg, PyAV, torchvision, or their thread/resource interaction.

- [x] **Step 4: Change one variable at a time**

Test only evidence-supported corrections, beginning with decoder thread limits and PyAV `16.0.1`, and rerun the complete three-video preflight after each change.

Expected: a correction is accepted only when the same frozen panel passes 3/3 in the production preflight.

### Task 3: Preserve evidence and leave the environment usable

**Files:**
- Create (ignored, generated): `.timelens2-private/timelens2-decoder-dev-20260801-d1/diagnostics/SHA256SUMS`
- Modify only if a code correction is proved: model-specific adapter tests and implementation files.

**Interfaces:**
- Consumes: phase transcripts and the live run status.
- Produces: checksummed private diagnostics, a sanitized outcome, and either a still-running debug environment or an explicit terminal error.

- [x] **Step 1: Persist incremental phase results**

After every phase, copy its transcript to the ignored host directory and record completion as `passed/attempted/expected` so mid-session failure remains auditable.

- [x] **Step 2: Apply TDD only if repository code must change**

Write and run a failing focused test before modifying any tracked decoder or adapter behavior; then run the focused suite and full verification.

- [x] **Step 3: Checksum and privacy-check artifacts**

Generate `SHA256SUMS` for the private diagnostic files and verify that Git contains no media, private manifest, raw node logs, or credentials.

- [ ] **Step 4: Report the live handoff**

Return the run name, provider/region/GPU, actual hourly price and accrued cost, decoder matrix, completion denominator, exact working revisions, and SSH reconnect command. Leave the dev environment running unless it reaches its hard two-hour maximum or the user asks to stop it.
