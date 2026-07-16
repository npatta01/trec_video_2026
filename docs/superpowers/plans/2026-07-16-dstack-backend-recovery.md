# dstack Backend Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a secret-safe local dstack backend configuration for RunPod, Vast.ai, DigitalOcean, and GCP, then add non-secret GPU/CPU fleet and YouTube network-test configurations.

**Architecture:** Provider credentials live only under `~/.dstack/server/` with mode `0600`; the repository contains only dstack fleet/task policy. The existing dstack Sky client project remains untouched, and no backend upload or billable apply occurs without a separate authorization.

**Tech Stack:** dstack 0.20.27, YAML, RunPod API, Vast.ai API, DigitalOcean API, GCP service-account JSON, yt-dlp 2026.7.4

## Global Constraints

- Never print, log, commit, or send provider credentials through chat.
- Do not overwrite `~/.dstack/config.yml`.
- Do not upload local credentials to dstack Sky without separate authorization.
- Do not provision an instance during implementation or validation.
- Keep fleet minimum size at zero and cap CPU offers at `$0.02/hour`.
- GCP uses a service-account JSON file referenced by filename.

---

### Task 1: Reconstruct the Local Credential Source

**Files:**
- Create: `~/.dstack/server/config.yml`
- User-provided: `~/.dstack/server/gcp-service-account.json`
- Preserve: `~/.dstack/config.yml`

**Interfaces:**
- Consumes: provider API keys and a GCP service-account JSON file supplied directly on the host.
- Produces: a local dstack server configuration with project name `main` and four backend definitions.

- [ ] **Step 1: Capture the existing Sky client configuration checksum**

Run:

```bash
sha256sum ~/.dstack/config.yml
```

Expected: one SHA-256 checksum for `~/.dstack/config.yml`; save only the checksum for the final comparison.

- [ ] **Step 2: Create the protected backend directory and configuration**

Create `~/.dstack/server/config.yml` with mode `0600` and this exact structure:

```yaml
projects:
  - name: main
    backends:
      - type: runpod
        creds:
          type: api_key
          api_key: REPLACE_WITH_RUNPOD_API_KEY
      - type: vastai
        creds:
          type: api_key
          api_key: REPLACE_WITH_VASTAI_API_KEY
      - type: digitalocean
        creds:
          type: api_key
          api_key: REPLACE_WITH_DIGITALOCEAN_API_KEY
      - type: gcp
        project_id: REPLACE_WITH_GCP_PROJECT_ID
        creds:
          type: service_account
          filename: /home/npatta01/.dstack/server/gcp-service-account.json
```

- [ ] **Step 3: Validate ownership, permissions, shape, and sentinel count**

Run:

```bash
stat -c '%U %a %n' ~/.dstack/server ~/.dstack/server/config.yml
rg -c 'REPLACE_WITH_' ~/.dstack/server/config.yml
```

Expected: the directory belongs to `npatta01`, the config mode is `600`, and the sentinel count is `4`.

- [ ] **Step 4: Confirm the Sky client configuration was preserved**

Run `sha256sum ~/.dstack/config.yml` again.

Expected: checksum exactly matches Step 1.

### Task 2: Add Cost-Bounded Fleet Policies

**Files:**
- Create: `dstack/gpu-fleet.dstack.yml`
- Create: `dstack/cpu-exit-fleet.dstack.yml`

**Interfaces:**
- Consumes: configured RunPod, Vast.ai, DigitalOcean, and GCP backends.
- Produces: zero-minimum fleet templates named `gpu-marketplace` and `cpu-exit`.

- [ ] **Step 1: Create the GPU fleet policy**

Create `dstack/gpu-fleet.dstack.yml`:

```yaml
type: fleet
name: gpu-marketplace
nodes: 0..1
idle_duration: 5m
backends: [runpod, vastai]
spot_policy: on-demand
resources:
  gpu: 16GB..
  disk: 100GB..
```

- [ ] **Step 2: Create the CPU exit fleet policy**

Create `dstack/cpu-exit-fleet.dstack.yml`:

```yaml
type: fleet
name: cpu-exit
nodes: 0..1
idle_duration: 1m
backends: [digitalocean, gcp]
spot_policy: auto
max_price: 0.02
resources:
  cpu: 1..2
  memory: 512MB..4GB
  disk: 10GB..25GB
```

- [ ] **Step 3: Parse both files using dstack without applying them**

Run:

```bash
echo n | dstack apply -f dstack/gpu-fleet.dstack.yml
echo n | dstack apply -f dstack/cpu-exit-fleet.dstack.yml
```

Expected: each command either displays an unmodified offer table and declines creation, or reports no configured backend/offers. It must not create a fleet or instance.

- [ ] **Step 4: Commit the fleet policies**

```bash
git add dstack/gpu-fleet.dstack.yml dstack/cpu-exit-fleet.dstack.yml
git commit -m "feat: add cost-bounded dstack fleets"
```

### Task 3: Add a Non-Downloading YouTube Network Test

**Files:**
- Create: `dstack/youtube-network-test.dstack.yml`

**Interfaces:**
- Consumes: the `cpu-exit` fleet and a `YOUTUBE_TEST_URL` environment variable.
- Produces: a task named `youtube-network-test` that simulates extraction without downloading media.

- [ ] **Step 1: Create the network-test task**

Create `dstack/youtube-network-test.dstack.yml`:

```yaml
type: task
name: youtube-network-test
python: "3.12"
fleets: [cpu-exit]
env:
  - YOUTUBE_TEST_URL
commands:
  - uv pip install "yt-dlp[default]==2026.7.4"
  - yt-dlp --simulate --no-playlist -- "$YOUTUBE_TEST_URL"
max_duration: 10m
```

- [ ] **Step 2: Preview task parsing without a fleet or billable apply**

Run:

```bash
YOUTUBE_TEST_URL='https://www.youtube.com/watch?v=BaW_jenozKc' \
  sh -c 'echo n | dstack apply -f dstack/youtube-network-test.dstack.yml'
```

Expected: dstack parses the task and stops before submission; with no created fleet it may report `No fleets`.

- [ ] **Step 3: Scan tracked and untracked repository files for credential material**

Run:

```bash
git grep -n -I -E 'REPLACE_WITH_|BEGIN PRIVATE KEY|private_key_id|api_key:' -- . ':!docs/superpowers/plans/2026-07-16-dstack-backend-recovery.md'
```

Expected: no matches.

- [ ] **Step 4: Commit the network test**

```bash
git add dstack/youtube-network-test.dstack.yml
git commit -m "feat: add dstack YouTube network test"
```

### Task 4: Hand Off Credential Entry and Backend Validation

**Files:**
- Modify manually: `~/.dstack/server/config.yml`
- Add manually: `~/.dstack/server/gcp-service-account.json`

**Interfaces:**
- Consumes: four user-controlled credentials.
- Produces: a locally valid backend configuration ready for control-plane selection.

- [ ] **Step 1: Give the user one local editing command**

Run:

```bash
nano ~/.dstack/server/config.yml
```

Expected: user replaces the four `REPLACE_WITH_...` sentinels directly on the host.

- [ ] **Step 2: Validate completion without displaying values**

Run:

```bash
test "$(rg -c 'REPLACE_WITH_' ~/.dstack/server/config.yml || true)" = 0
test -s ~/.dstack/server/gcp-service-account.json
test "$(stat -c '%a' ~/.dstack/server/config.yml)" = 600
test "$(stat -c '%a' ~/.dstack/server/gcp-service-account.json)" = 600
```

Expected: all commands exit zero and print nothing.

- [ ] **Step 3: Stop at the control-plane boundary**

Report whether the local config is complete. Do not start `dstack server`, upload backends to Sky, create fleets, or launch the YouTube test until the user chooses the control-plane path and approves any credential upload or billable plan.

