# VideoChat3 runtime certification design

## Status and objective

- **Status:** design approved in conversation on 2026-07-31; implementation
  pending written-spec review.
- **Branch:** `codex/videochat3-4b-pilot`.
- **Objective:** catch packaging, adapter, processor, device, and control-plane
  failures before another immutable VideoChat3 experiment is eligible to run.
- **Scope:** synthetic, label-free runtime certification only. No competition
  rows, cached private frames, reference answers, grading, or quality claims.
- **Scientific boundary:** VideoChat3 inference remains NVIDIA CUDA BF16 only.
  CPU and ROCm are development smoke lanes and can never satisfy the CUDA gate.

## Approaches considered

### 1. Local exact-image CPU smoke plus remote CUDA certification

Run the digest-pinned NVIDIA image locally without a GPU to verify imports,
dependency resolution, manifest checks, tensor/list construction, metadata,
prompt construction, and runner control flow. Then run one synthetic forward
pass in a bounded dstack NVIDIA dev environment.

This is the minimum credible approach. It catches most software defects
locally but does not exercise the host's available AMD GPU.

### 2. Layered CPU, advisory ROCm, and required remote CUDA certification

Add an advisory ROCm lane between the exact-image CPU smoke and remote CUDA.
PyTorch on ROCm intentionally exposes the `torch.cuda` API, so this lane can
catch accidental device assumptions and tensor-placement defects. It cannot
certify NVIDIA CUDA binaries, the exact NVIDIA image, or NVIDIA-specific
kernels.

This is the selected approach because the ROCm image and `/dev/kfd` are already
available locally, while the authoritative final gate remains production-like.

### 3. Maintain equivalent CUDA and ROCm model environments

Build separate full model images and run VideoChat3 weights on both GPU stacks.
This would add large local downloads, dual dependency maintenance, and a second
runtime that is not part of the scientific condition. It is rejected for this
pilot.

## Selected architecture

The certification is a fail-closed sequence:

```text
unit tests
    -> exact NVIDIA image / CPU / synthetic frames
        -> advisory ROCm / synthetic tensors
            -> dstack client-server and offer health
                -> bounded NVIDIA dev environment
                    -> exact model revision / synthetic 12-frame inference
                        -> signed-by-hashes local certificate
                            -> immutable-run eligibility
```

The ROCm result is recorded but does not block the NVIDIA lane unless it
reveals a shared source defect. Every required gate binds the same candidate
source tree and runtime inputs by SHA-256. A later source, requirements, image,
model-revision, or dstack-version change invalidates the certificate.

## Components

### Runtime contract module

A model-specific module owns the certification schema and checks:

- exact CUDA image digest;
- exact VideoChat3 model revision and Transformers source revision;
- requirements-file SHA-256;
- source snapshot SHA-256;
- dstack client and server versions;
- 12-frame tensor/list shape and chronological exact timestamps;
- model-native `VideoMetadata` compatibility;
- deterministic greedy generation with the 32-token cap;
- CUDA availability, NVIDIA backend, BF16 support, and BF16 model parameters;
- one non-empty short synthetic answer.

It must distinguish `cpu-smoke`, `rocm-smoke`, and `cuda-certification`
explicitly. Production code has no automatic CPU or ROCm fallback.

### Local exact-image CPU lane

A terminal-friendly Podman command uses the already-cached digest-pinned
NVIDIA image. It installs or verifies the pinned dependency snapshot without
downloading model weights, generates twelve temporary synthetic frames, and
executes the real request/metadata/input-construction path with a strict test
processor and model fixture.

The lane proves Python/package compatibility and adapter control flow. Its
attestation says `backend=cpu-smoke` and cannot be consumed as CUDA evidence.

### Advisory ROCm lane

A separate Podman command uses the locally cached ROCm PyTorch image by digest,
passes `/dev/kfd` and `/dev/dri`, and verifies that PyTorch reports a HIP build
and a usable device. It runs synthetic tensor allocation, BF16 arithmetic when
supported, and the shared adapter preprocessing path without model weights.

The lane is skipped with an explicit reason if the cached image or device is
unavailable. It never installs the CUDA-pinned Torch wheel and never pretends
that HIP is NVIDIA CUDA.

### Durable dstack control-plane gate

The client and server must run the same pinned dstack version. The server is
launched as a durable user-scoped service rather than as a child of a short
shell command. Readiness requires both a live listener and a successful API
request after a settling interval. Logs are credential-safe; generated tokens
are never printed or committed.

Before provisioning, the workflow performs a declined config preview with
`echo "n" | dstack apply -f ...`, retains the full output privately, and
requires compatible offers from multiple physical locations when practical.

### Bounded NVIDIA dev environment

The dev environment uses the exact digest-pinned NVIDIA image, scalar
`A40:1:24GB..`, on-demand capacity, a hard price ceiling, a 30-minute running
limit, and a 10-minute inactivity limit. It receives only:

- the candidate source snapshot;
- pinned requirements;
- a synthetic-frame generator and certification command;
- the model identifier and pinned revision;
- `HF_TOKEN` through dstack secrets.

It receives no private media or frozen experiment manifests. The pinned model
is downloaded only inside the remote environment. Interactive SSH is used to
diagnose and rerun the synthetic command on the same provisioned machine. Any
source fix is first made with a RED/GREEN local test, then transferred as a new
hash-bound snapshot; ad-hoc remote edits are not certifiable.

One initial dev-environment identity and at most one diagnosed infrastructure
correction are permitted. A second infrastructure failure stops certification.
Poor synthetic output quality is not a retry reason; only completion and
contract validity matter.

## Certificate and immutable-run interlock

The successful sanitized certificate records:

- schema version and UTC completion time;
- Git commit and source snapshot SHA-256;
- image, requirements, model, and Transformers revisions;
- dstack client/server versions and dev-environment identity;
- provider, region, GPU model, and actual cost;
- Torch/Transformers versions and CUDA/BF16 attestations;
- synthetic input shape, exact timestamps, decoding settings, and completion;
- SHA-256 of the private detailed runtime record.

It excludes tokens, paths to private data, raw model output, and full logs. The
immutable-run preparation command fails closed unless a successful certificate
matches every bound value. Existing consumed or retired dstack identities stay
non-runnable and are never repurposed.

Passing certification permits preparation of a fresh immutable canary; it is
not scientific evidence and does not itself authorize or execute the pilot.

## Error handling and cost control

- Local CPU or ROCm failure provisions nothing and costs nothing.
- A shared-source failure found by ROCm blocks progression until fixed and
  retested; a ROCm-only incompatibility is recorded as advisory.
- A dstack control-plane or offer failure gets one diagnosed corrective attempt
  at most.
- The dev environment is stopped immediately after a conclusive pass or
  persistent failure.
- No blind retry loop, broadened provider/hardware scope, CPU model fallback,
  or quality-driven rerun is allowed.
- Cumulative VideoChat3 cloud spend starts at the recorded `$0.0797` and must
  remain below the user's `$10.00` total authorization.

## Test and verification strategy

Implementation follows RED/GREEN tests for:

- lane identity and the prohibition on production fallback;
- certificate schema, hash binding, expiry-on-change, and fail-closed use;
- synthetic 12-frame request, exact timestamp metadata, and 32-token greedy
  decoding;
- NVIDIA-vs-HIP backend discrimination;
- dstack config image/model/dependency pins, scalar selector, price/duration,
  no-private-media boundary, and secret handling;
- durable server readiness and version mismatch rejection;
- retired identity non-runnability;
- sanitized evidence and secret/privacy scans.

Verification runs model-specific tests, exact-image CPU smoke, advisory ROCm
smoke, dstack declined preview, the remote synthetic CUDA command, certificate
validation, `git diff --check`, and repository privacy checks. Full command
output is inspected before any passing claim or commit.

## Non-goals

- No Docker-in-Docker.
- No local VideoChat3 model download.
- No competition-media upload to the dev environment.
- No gate/pilot inference, grading, or result interpretation in this
  certification phase.
- No edits to shared project-status QMD or generated Markdown/HTML.

## 2026-07-31 dev-only a3 amendment

The user explicitly reopened only the synthetic dev-environment gate after the
retained generated job exposed the a1/a2 lifecycle cause: `exec bash` in the
init item replaced dstack's wrapper shell, so its SSH keep-alive could never
run. The a3 candidate removes only `exec`. It retains every runtime, hardware,
budget, privacy, and inference pin above, freezes the raw synthetic answer in
the private run prefix, and stops after one conclusive synthetic result. It
does not upload competition media or authorize a pilot, a4, or a quality claim.

### A3 outcome

A3 completed `1/1` synthetic inference in the mutable dev environment after
local RED/GREEN adapter corrections were transferred to the live machine. The
exact model loaded with CUDA BF16 parameters and the tensor-plus-metadata path
completed, but the loader reported `lm_head.weight` missing and newly
initialized and the frozen 32-token output was incoherent. Evidence validity is
**Compatibility-only** and decision disposition is **Not promoted under the
tested condition**. The identity is retired; no private-media gate or pilot was
run or authorized by this result.
