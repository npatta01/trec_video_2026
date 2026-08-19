# TimeLens2 A40-Class Fleet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provision up to four A40-class CUDA nodes and execute one immutable TimeLens2 `-a2` canary followed by the three-video compatibility gate if eligible.

**Architecture:** Update the existing elastic `gpu-marketplace` template and the two TimeLens2 task configurations to share one closed GPU selector with a 24 GB VRAM floor. Keep task price and duration caps unchanged, preserve label isolation, and enforce canary-first detached execution with immutable Hugging Face Bucket artifacts.

**Tech Stack:** dstack 0.20.27, YAML task/fleet configurations, Bash runner, pytest, Hugging Face Hub CLI 1.26.0.

## Global Constraints

- Use `codex/timelens2-selector-pilot` in the existing linked worktree.
- GPU pool: `A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:24GB..`.
- Fleet size: `0..4`; on-demand; five-minute idle timeout.
- Preserve canary `max_price: 0.55`, `max_duration: 15m`.
- Preserve compatibility `max_price: 0.65`, `max_duration: 1h`.
- Use immutable `timelens2-runner-canary-20260730-a2` and `timelens2-selector-pilot-20260730-a2`.
- Preview every dstack configuration with `echo "n" | dstack apply -f ...`.
- Submit tasks once with `-y -d`; do not use external retry loops.
- Keep `HF_TOKEN` in dstack secrets and avoid diagnostic output.
- Do not run the focused pilot while the relevant-scene oracle is absent.

---

### Task 1: Configuration contract

**Files:**
- Modify: `tests/test_timelens2_dstack_config.py`
- Modify: `dstack/gpu-fleet.dstack.yml`
- Modify: `dstack/timelens2-runner-canary.dstack.yml`
- Modify: `dstack/timelens2-selector-pilot.dstack.yml`
- Modify: `scripts/run-timelens2-selector-dstack.sh`

**Interfaces:**
- Consumes: existing exact image, model revision, price caps, and private mount boundary.
- Produces: tested four-node fleet and immutable `-a2` task definitions.

- [ ] **Step 1: Write failing contract tests**

Add `FLEET = Path("dstack/gpu-fleet.dstack.yml")` and
`GPU_POOL = "A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:24GB.."`.
Assert that the fleet contains `nodes: 0..4`, the GPU pool, `idle_duration: 5m`,
and on-demand RunPod/Vast.ai backends. Replace all `-a1` and A40-only assertions
with `-a2` and `GPU_POOL`, while retaining the price, duration, secret, and
private-media assertions.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest -q tests/test_timelens2_dstack_config.py
```

Expected: failures for `nodes: 0..4`, the closed GPU pool, and `-a2` names.

- [ ] **Step 3: Implement the minimal configuration change**

Set the fleet to:

```yaml
nodes: 0..4
resources:
  gpu: A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:24GB..
  disk: 100GB..
```

Use the same GPU selector in both TimeLens2 task configurations. Change names,
environment IDs, remote prefixes, runner allowlists, and the private manifest
mount to `-a2`. Do not change price caps, durations, image, model revision,
providers, or retry policy.

- [ ] **Step 4: Stage the private `-a2` manifest**

Copy the existing validated label-free request manifest to:

```text
.timelens2-private/timelens2-selector-pilot-20260730-a2/compatibility-requests.json
```

Verify identical SHA-256 bytes and confirm both paths remain ignored.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_timelens2_dstack_config.py tests/test_timelens2_selector.py
bash -n scripts/run-timelens2-selector-dstack.sh
git diff --check
```

Expected: all tests and static checks pass.

### Task 2: Apply the fleet correction

**Files:**
- Input: `dstack/gpu-fleet.dstack.yml`
- Private output: `.timelens2-private/timelens2-selector-pilot-20260730-a2/fleet-preview.txt`

**Interfaces:**
- Consumes: tested fleet YAML from Task 1.
- Produces: an active `gpu-marketplace` fleet with maximum four nodes.

- [ ] **Step 1: Retain the declined fleet preview**

Run:

```bash
echo "n" | dstack apply -f dstack/gpu-fleet.dstack.yml
```

Retain the complete output privately. Require the update to preserve the
existing busy node and show the approved template.

- [ ] **Step 2: Apply once**

Run:

```bash
dstack apply -f dstack/gpu-fleet.dstack.yml -y
```

- [ ] **Step 3: Verify the live fleet**

Use an allowlisted projection of:

```bash
dstack fleet get --json gpu-marketplace
```

Require `nodes.max == 4`, the approved GPU selector, and active status.

- [ ] **Step 4: Check fleet-scoped offers**

Run an offer check through `gpu-marketplace` with the task price and resource
caps. Require at least two physically distinct eligible offers when practical.

### Task 3: Exact-image CUDA canary

**Files:**
- Input: `dstack/timelens2-runner-canary.dstack.yml`
- Private output: `.timelens2-private/timelens2-selector-pilot-20260730-a2/canary-preview.txt`
- Remote output: `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-runner-canary-20260730-a2/`

**Interfaces:**
- Consumes: corrected fleet and exact pinned image.
- Produces: verified CUDA/runtime/model-config canary artifacts.

- [ ] **Step 1: Preview and retain output**

Run:

```bash
echo "n" | dstack apply -f dstack/timelens2-runner-canary.dstack.yml
```

Stop if no acceptable offers are shown.

- [ ] **Step 2: Submit once detached**

Run:

```bash
dstack apply -f dstack/timelens2-runner-canary.dstack.yml -y -d
```

- [ ] **Step 3: Monitor without diagnostic logs**

Use `dstack ps -v`, an allowlisted `dstack run get ... --json` projection, and
normal `dstack logs`. Do not resubmit.

- [ ] **Step 4: Verify remote artifacts**

List the remote prefix, download it to the ignored private directory, and run:

```bash
sha256sum -c SHA256SUMS
```

Require a successful terminal status and canary payload before Task 4.

### Task 4: Three-video compatibility gate

**Files:**
- Input: `dstack/timelens2-selector-pilot.dstack.yml`
- Private input: `.timelens2-private/timelens2-selector-pilot-20260730-a2/compatibility-requests.json`
- Remote output: `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-selector-pilot-20260730-a2/`

**Interfaces:**
- Consumes: passing canary and exact frozen three-video manifest.
- Produces: raw, normalized, attempt, phase, summary, freeze, and checksum artifacts.

- [ ] **Step 1: Revalidate the private panel**

Verify the frozen source gate SHA, three distinct identities, query hashes,
request hashes, and local media hashes without printing questions or paths.

- [ ] **Step 2: Preview and retain output**

Run:

```bash
echo "n" | dstack apply -f dstack/timelens2-selector-pilot.dstack.yml
```

Stop if no acceptable offers are shown.

- [ ] **Step 3: Submit once detached**

Run:

```bash
dstack apply -f dstack/timelens2-selector-pilot.dstack.yml -y -d
```

- [ ] **Step 4: Monitor and verify artifacts**

Use non-diagnostic status/log commands. List and download the remote prefix,
verify `SHA256SUMS`, and report accepted/attempted/required completion.

- [ ] **Step 5: Assign evidence disposition**

Pass compatibility only if all three rows contain strict, duration-bounded JSON
intervals. Preserve a valid negative or incomplete result without rerunning for
quality. Keep the focused pilot incomplete because its oracle is absent.

### Task 5: Verification and handoff

**Files:**
- Modify: `docs/experiments/2026-07-30-timelens2-selector-pilot.md`
- Do not modify: `docs/project-status/*.qmd`

**Interfaces:**
- Consumes: verified live run and artifact evidence.
- Produces: sanitized durable record and committed implementation.

- [ ] **Step 1: Update sanitized evidence**

Record the `-a2` offer, fleet, run, cost, completion, artifact URI/checksum, and
evidence disposition. Include no raw media, questions, model outputs, or logs.

- [ ] **Step 2: Run fresh verification**

Run:

```bash
uv run pytest -q
uvx ruff check --select E,F,I,UP src/trec_video_2026/vqa_experiments/timelens2_selector.py tests/test_timelens2_selector.py tests/test_timelens2_dstack_config.py
uvx ruff format --check src/trec_video_2026/vqa_experiments/timelens2_selector.py tests/test_timelens2_selector.py tests/test_timelens2_dstack_config.py
bash -n scripts/run-timelens2-selector-dstack.sh
git diff --check
```

- [ ] **Step 3: Run privacy checks**

Confirm `.timelens2-private/` and `data` are ignored, no private path or
credential is tracked, and no shared project-status QMD changed.

- [ ] **Step 4: Commit**

Stage only the approved fleet/task/runner/test and sanitized evidence files,
then commit with:

```bash
git commit -m "fix: expand TimeLens2 CUDA fleet capacity"
```
