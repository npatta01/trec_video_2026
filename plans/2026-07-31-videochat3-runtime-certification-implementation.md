# VideoChat3 Runtime Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` for implementation tasks. The
> primary agent alone owns paid dstack preview, submission, monitoring, stop,
> artifact handling, and evidence reconciliation.

**Goal:** Build and exercise a fail-closed VideoChat3 runtime certification
that catches local packaging/adapter failures and requires one real synthetic
NVIDIA BF16 inference before another immutable canary is eligible.

**Architecture:** A pure-Python certification contract validates lane identity,
runtime pins, backend attestation, synthetic inference outcome, and certificate
hash binding. Rootless Podman drives required exact-NVIDIA-image CPU smoke and
advisory ROCm smoke without model weights. A bounded dstack dev environment
downloads the pinned model remotely, runs the same synthetic 12-frame path, and
emits a sanitized certificate consumed by a fail-closed eligibility check.

**Tech Stack:** Python 3.12, pytest, Bash, rootless Podman, PyTorch,
Transformers, YAML/dstack 0.20.29, Hugging Face CLI/Buckets.

**Status (2026-07-31): complete.** The explicitly reauthorized a3 dev
environment completed the synthetic CUDA BF16 path after test-covered
interactive adapter fixes. Its frozen output was incoherent and the load report
identified a missing, newly initialized `lm_head.weight`; the exact condition
is not promoted. A3 cost `$0.2165`, cumulative VideoChat3 cost is `$0.3535`,
the private bundle was downloaded and checksum-verified, and the consumed
identity is retired. No competition media, compatibility gate, or pilot ran.

## Global Constraints

- Work only on branch `codex/videochat3-4b-pilot` in the existing linked
  worktree.
- The ignored `data` symlink must resolve exactly to
  `/home/npatta01/data/competitions/trec_video_2026/data`; stop on conflict.
- Scientific inference is NVIDIA CUDA BF16 only. CPU and ROCm are explicit
  smoke lanes and never automatic model fallbacks.
- Do not download VideoChat3 model weights locally.
- Exact CUDA image:
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`.
- VideoChat3 model revision:
  `MCG-NJU/VideoChat3-4B@37fa901ec5913f84bc31108ebc1e60ad1903634c`.
- Transformers source revision:
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`.
- Runtime requirements SHA-256:
  `9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6`.
- Use twelve synthetic frames, chronological exact timestamps, seed
  `20260724`, greedy decoding, and `max_new_tokens=32`.
- The remote dev environment receives no competition rows, private media,
  frozen experiment manifests, reference answers, or grades.
- Keep `HF_TOKEN` in dstack secrets only. Never print/persist it and never use
  diagnose output for a secret-bearing run.
- Preview with `echo "n" | dstack apply -f ...`, retain full output privately,
  submit detached, and use a foreground workload with hard price/duration
  bounds and no blind external retry loop.
- Cumulative prior VideoChat3 cost is `$0.0797`; total cost must remain below
  the user's `$10.00` authorization.
- No existing consumed or retired dstack identity becomes runnable again.
- Do not edit shared project-status QMD or generated Markdown/HTML.

---

### Task 1: Certification contract and local smoke lanes

**Files:**

- Create: `src/trec_video_2026/vqa_experiments/videochat3_runtime_certification.py`
- Create: `scripts/run-videochat3-runtime-smoke.sh`
- Create: `containers/videochat3-cuda-smoke.Containerfile`
- Test: `tests/test_videochat3_runtime_certification.py`

**Interfaces:**

- Produces:
  `classify_torch_backend(torch_module: Any) -> Literal["cpu", "rocm", "nvidia-cuda"]`.
- Produces:
  `validate_lane(lane: str, backend: str) -> None`, where `cpu-smoke` accepts
  only `cpu`, `rocm-smoke` accepts only `rocm`, and `cuda-certification`
  accepts only `nvidia-cuda`.
- Produces:
  `synthetic_timestamps() -> tuple[float, ...]` with the literal values
  `(0.0, 1.0, ..., 11.0)`.
- Produces:
  `build_synthetic_video(torch_module: Any) -> Any` with shape
  `(12, 3, 32, 32)` and deterministic byte-valued frames.
- Produces:
  `build_certificate(runtime: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]`
  and `validate_certificate(certificate: Mapping[str, Any], expected: Mapping[str, str]) -> None`.
- Produces CLI modes `cpu-smoke`, `rocm-smoke`, and `cuda-certification`.
- Produces shell entry points `cpu`, `rocm`, and `local`, where `local` runs
  required CPU and advisory ROCm in sequence.

- [ ] **Step 1: Write RED backend/lane tests**

  Add table-driven tests with complete fake Torch version/CUDA structures.
  Assert that HIP takes precedence over the shared `torch.cuda` namespace,
  CPU is detected without an accelerator, and `cuda-certification` rejects
  both CPU and ROCm with an error naming the observed backend.

- [ ] **Step 2: Run the focused RED tests**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_runtime_certification.py -k 'backend or lane'
  ```

  Expected: collection/import failure because the module does not exist.

- [ ] **Step 3: Implement minimal backend/lane behavior**

  Add explicit backend detection using `torch.version.hip`,
  `torch.version.cuda`, and `torch.cuda.is_available()`. Add a literal
  lane-to-backend map and reject unknown lanes.

- [ ] **Step 4: Run the focused GREEN tests**

  Run the Step 2 command and require all selected tests to pass.

- [ ] **Step 5: Write RED synthetic-input and certificate tests**

  Assert the exact tensor shape/dtype/value pattern, literal timestamps,
  certificate omission of raw output/secrets, lowercase SHA-256 validation,
  pin mismatch rejection, incomplete outcome rejection, and source-snapshot
  mismatch invalidation.

- [ ] **Step 6: Run the expanded RED tests**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_runtime_certification.py
  ```

  Expected: failures for missing synthetic/certificate behavior.

- [ ] **Step 7: Implement minimal synthetic and certificate behavior**

  Build frames without Pillow or private files. Certificates contain only
  schema/version, pins, backend/device/version attestations, decoding settings,
  input shape/timestamps, completion/answer-length, remote-detail SHA-256,
  provider/region/GPU/cost, and source snapshot SHA-256. Reject unknown keys in
  the expected pin mapping and prohibit keys matching token, secret, answer,
  prompt, media, row, or reference.

- [ ] **Step 8: Add RED shell behavior tests**

  Execute the shell script with a temporary fake `podman` executable. Assert
  the CPU invocation uses the exact CUDA image digest with no GPU devices, the
  ROCm invocation uses the cached ROCm digest plus `/dev/kfd` and `/dev/dri`,
  and no command contains a model-download operation.

- [ ] **Step 9: Run shell RED tests**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_runtime_certification.py -k podman
  ```

  Expected: failures because the script/Containerfile behavior is absent.

- [ ] **Step 10: Implement shell and Containerfile behavior**

  The Containerfile uses the exact CUDA base digest, removes the conflicting
  preinstalled Spin package, installs `uv==0.11.3`, and installs the pinned
  requirements without model weights. The shell uses `--pull=never`, mounts
  only `src/`, runs the CLI, detects the cached ROCm digest, and records an
  explicit advisory skip when ROCm is unavailable.

- [ ] **Step 11: Run task verification and commit**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_runtime_certification.py tests/test_videochat3_pilot.py
  bash -n scripts/run-videochat3-runtime-smoke.sh
  git diff --check
  ```

  Commit only Task 1 files with message
  `Add VideoChat3 local runtime certification`.

---

### Task 2: Bounded dstack CUDA dev gate and fail-closed interlock

**Files:**

- Create: `dstack/videochat3-runtime-cert-20260731-a1.dstack.yml`
- Create: `scripts/prepare-videochat3-runtime-cert-dstack.sh`
- Create: `scripts/verify-videochat3-runtime-certificate.py`
- Modify: `tests/test_videochat3_dstack_config.py`
- Modify: `tests/test_videochat3_runtime_certification.py`
- Modify after execution only: `plans/videochat3-4b-pilot.md`

**Interfaces:**

- Consumes Task 1 CLI `cuda-certification` and certificate validator.
- The prep script validates exact run/image/model/requirements/source hashes,
  installs the pinned environment, downloads the pinned model remotely, and
  prints credential-safe CUDA readiness.
- The verifier takes `--certificate PATH`, `--source-root PATH`, and
  `--requirements PATH`; exit 0 means eligibility, exit 2 means fail closed.
- The dstack environment runs as
  `videochat3-runtime-cert-20260731-a1`, with
  `videochat3-runtime-cert-20260731-a2` reserved only for one diagnosed
  infrastructure correction.

- [ ] **Step 1: Write RED config and verifier tests**

  Assert the runnable config is a dev environment with the exact image,
  scalar `gpu: A40:1:24GB..`, on-demand RunPod/Vast.ai pool, `$1.00/hour`,
  `30m` maximum, `10m` inactivity, no private paths/manifests, and only the
  secret name `HF_TOKEN`. Execute the verifier against missing, malformed,
  mismatched, and valid certificate fixtures.

- [ ] **Step 2: Run RED tests**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_dstack_config.py \
    tests/test_videochat3_runtime_certification.py -k 'runtime_cert or certificate_cli'
  ```

  Expected: failures because config, prep script, and verifier do not exist.

- [ ] **Step 3: Implement minimal config/prep/verifier**

  Upload only source, requirements, prep script, and synthetic certificate
  module. Keep model download remote. Use a foreground init command and an IDE
  suitable for SSH attachment. The verifier recomputes the source snapshot and
  requirements SHA-256 and rejects non-NVIDIA or incomplete certificates.

- [ ] **Step 4: Run GREEN tests and static shell/config checks**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_dstack_config.py \
    tests/test_videochat3_runtime_certification.py
  bash -n scripts/prepare-videochat3-runtime-cert-dstack.sh
  python -c 'import yaml; yaml.safe_load(open("dstack/videochat3-runtime-cert-20260731-a1.dstack.yml"))'
  git diff --check
  ```

  Require all commands to exit 0.

- [ ] **Step 5: Commit Task 2**

  Commit only Task 2 implementation/tests with message
  `Add bounded VideoChat3 CUDA certification gate`.

---

### Task 3: Exercise certification and reconcile evidence

**Owner:** primary agent only; no subagent may submit, attach, stop, access
secrets, download artifacts, or edit the evidence record.

**Files:**

- Modify: `plans/videochat3-4b-pilot.md`
- Create only after a terminal run:
  `dstack/videochat3-runtime-cert-20260731-a1.retired.yml`
- Remove after preserving retired provenance:
  `dstack/videochat3-runtime-cert-20260731-a1.dstack.yml`

- [ ] **Step 1: Run required exact-image CPU smoke**

  Build/run with rootless Podman, `--pull=never`, no model token, no model
  weights, and no dataset mount. Retain the complete local result and source /
  image / requirements hashes.

- [ ] **Step 2: Run advisory ROCm smoke**

  Use the cached ROCm image digest and `/dev/kfd`/`/dev/dri`. Record pass,
  advisory failure, or explicit skip without presenting it as CUDA evidence.

- [ ] **Step 3: Establish durable dstack control plane**

  Verify dstack client/server `0.20.29`, live listener, successful API request
  after settling, and credential-safe logging. Do not print generated tokens.

- [ ] **Step 4: Preview and evaluate offers**

  Run exactly:

  ```bash
  echo "n" | dstack apply -f dstack/videochat3-runtime-cert-20260731-a1.dstack.yml
  ```

  Retain full output privately. Require a valid plan and multiple physically
  distinct compatible offers when practical. If the first infrastructure gate
  fails, diagnose and allow only the predeclared a2 correction.

- [ ] **Step 5: Submit detached and attach**

  Submit with `-y -d`, verify status, then attach without diagnose output.
  Run the CUDA synthetic certification command interactively on the exact
  uploaded snapshot. Any code fix must return through a local RED/GREEN cycle
  before the same environment receives the new hash-bound snapshot.

- [ ] **Step 6: Stop and verify cost**

  Stop immediately after terminal pass/persistent failure, inspect allowlisted
  run JSON fields, and record incremental/cumulative cost below `$10.00`.

- [ ] **Step 7: Preserve sanitized terminal provenance**

  Replace the runnable config with non-runnable retired YAML containing no
  `type`, `env`, `init`, `files`, commands, token, or secret references. Record
  run identity, source/config hashes, offer, actual cost, completion
  denominator, validity, disposition, error/audit gap, and what the result does
  and does not rule out.

- [ ] **Step 8: Run final verification and commit reconciliation**

  Run:

  ```bash
  uv run pytest -q tests/test_videochat3_runtime_certification.py \
    tests/test_videochat3_pilot.py tests/test_videochat3_dstack_config.py
  uv run pytest -q
  git diff --check
  git status --short
  ```

  Run repository privacy/secret scans over changed files, inspect the complete
  diff, and commit with message
  `Record VideoChat3 runtime certification result`.

## Self-review

- **Spec coverage:** Task 1 owns CPU/ROCm smoke and certificate schema; Task 2
  owns the NVIDIA dev gate/interlock; Task 3 owns real execution, budget,
  privacy, retirement, and evidence vocabulary.
- **Placeholder scan:** the plan contains no deferred implementation choices;
  environment identity, revisions, pins, limits, commands, and file ownership
  are explicit.
- **Type consistency:** Task 2 consumes the exact Task 1 validator and CLI
  names; Task 3 executes the exact Task 2 config and retires the same identity.
