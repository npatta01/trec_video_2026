# Qwen3-Omni frozen-30 infrastructure attempts

## Decision

This file owns the historical A1–A3 infrastructure attempts. Those attempts
are **incomplete and scientifically ineligible**: A1 and A2 never reached the
runner, while A3 was stopped before inference after instrumentation exposed
bulk dstack file transport as the remaining bottleneck.

The later A4 private-bundle path completed the unchanged scientific condition
and supersedes only the pending quality unknown, not this failure history. Its
eligible result and model decisions are recorded in the
[A4 evidence of record](2026-07-31-qwen3-omni-frozen30.md).

The July 31 D1 diagnostic proved one fresh dstack 0.20.29 runner, code-map,
and task-start path in the pinned image. The later C1 non-development task
independently repeated that boundary on a fresh node and also proved benign
secret resolution, CUDA availability, the pinned Transformers environment,
and the transferred frozen-30 runner digest. C1 stopped before model download
or inference as designed and exited 0 with `all_jobs_done`. The remaining
sealed-task canary condition was satisfied, so the bounded A3 full attempt was
operationally authorized. A3 then showed that sending 445 mapped paths and
about 318 MB through dstack could not finish before runner startup. A4 mapped
only four small code/metadata paths and downloaded one immutable private bundle
after startup; its exact normal-task bundle canary and full dev workload both
completed.

## Intended scientific contract

- Primary question: whether native audio makes Qwen3-Omni useful as a routed
  specialist on the balanced frozen-30 panel, with a matched frames-only
  control.
- Model: `Qwen/Qwen3-Omni-30B-A3B-Thinking`.
- Model revision:
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
- Runtime image:
  `docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`.
- A2 canonical contract digest:
  `bdd89956b7a5b89f9690d3fa0799f439667fc94e5a233c860ca0c45daaa32480`.
- A2 transport-manifest digest:
  `d160da24ef88d32a4323d09fde2202781086ad62597ffffb2fff350df3c3b543`.
- A2 task-configuration digest:
  `a097d949ea1a944933ec79ea240bf81ccefdd35a7f8b7074797b52b2c1500ad5`.
- A3 canonical contract digest:
  `f7ca093af6fc54bd9403727613d624c6ae4cef3e2ffea17f520a6f1456913669`.
- A3 transport-manifest digest:
  `f5c38dee682da4e2c716114c736a8f7e74879fb5731f2ca3d7ba076b5ee8c4a8`.
- A3 task-configuration digest:
  `f81a8d16c36f6de6eb65cabc9b95f3fc1f91f1f68be8215f92ebf590f9b04490`.
- Integrity boundary: paired label-blind inference, prediction freeze before
  labels, blind adjudication, and scoring only after every eligibility gate.

The scientific contract never executed. These hashes preserve the attempted
condition for A2/A3; they are not evidence about model quality. A4 used a fresh
contract identity while preserving the panel, evidence bytes, prompt, model,
decoding, and scoring rules. Its contract SHA-256 is
`9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5`.

## Operational observations

| Run | Backend / offer | Terminal observation | Cost |
|---|---|---|---:|
| `qwen3-omni-frozen30-audio-20260730-a1` | RunPod AP-JP-1, on-demand H200, `$4.39/hour` | Submitted `2026-07-30T22:36:18Z`; failed before workload start at `2026-07-30T23:00:50Z` with `waiting_runner_limit_exceeded` after an observed submission age of `1467.635295` seconds. | `$1.7946` |
| `qwen3-omni-runner-canary-20260730-c3` | RunPod US-CA-2, on-demand H200, `$4.39/hour` | Minimal pinned-image canary completed with exit status 0 from `2026-07-31T00:50:24Z` to `2026-07-31T00:59:15Z`. | `$0.6478` |
| `qwen3-omni-frozen30-audio-20260730-a2` | RunPod US-CA-2, on-demand H200, `$4.39/hour` | Submitted `2026-07-31T01:41:30Z`; the instance became busy at `01:48:38Z`, but the runner never responded. The job failed at `02:04:16Z` with `waiting_runner_limit_exceeded` after an observed submission age of `1361.06722` seconds. | `$1.6660` |
| `qwen3-omni-runner-diagnostic-20260731-d1` | RunPod US-CA-2, on-demand H200, `$4.39/hour` | Submitted `2026-07-31T11:10:32Z` under dstack 0.20.29. The runner became healthy, the pinned-image task printed both task-start and task-done sentinels, and dstack recorded `all_jobs_done` at `11:18:42Z`. | `$0.5893` |
| `qwen3-omni-usefulness-dev-20260731-a1` | RunPod US-GA-2, on-demand H200, `$4.39/hour` | A development environment loaded the exact pinned model and produced 8/8 requested paired generations across four diagnostic examples before an intentional stop. This established model/runtime viability but was not a frozen-panel task result. | `$1.0221` |
| `qwen3-omni-task-canary-20260731-c1` | RunPod AP-JP-1, on-demand H200, `$4.39/hour` | A normal non-development task used the exact pinned image, resolved a nonempty Hugging Face secret without printing it, confirmed CUDA and Transformers, verified the transferred runner SHA-256, printed start/done sentinels, and exited 0 with `all_jobs_done` at `2026-07-31T14:25:05Z`. It deliberately did not download or run the model. | `$0.3767` |
| `qwen3-omni-frozen30-audio-20260731-a3` | RunPod, on-demand H200, `$4.39/hour` | Instrumentation proved a healthy runner and HTTP 200 healthcheck, then measured serial transfer of 445 mappings at roughly 3.3 MB per 10 seconds. The run was intentionally stopped before inference because the 318 MB transfer could not finish within the runner-startup window. | `$1.2035` |
| `qwen3-omni-f30-a4-bundle-canary` | RunPod, on-demand H200, `$4.39/hour` | The exact normal task downloaded the complete 318,177,280-byte private bundle, verified its pinned hashes, reconstructed all 390 evidence files, passed the frozen preflight, and ended with `all_jobs_done`. | `$0.2958` |
| `qwen3-omni-frozen30-audio-20260731-a4-dev` | RunPod, on-demand H200, `$4.39/hour` | The dev environment completed all `60/60` strict paired outputs, synced nine checksum-bound artifacts, and was intentionally stopped after workload return code `0`. | `$2.9943` |

For A2, the provider SSH endpoint returned a valid OpenSSH banner and accepted
dstack's authenticated tunnel. The local controller held established Unix
connections to that tunnel, but the remote runner service behind forwarded
port 10999 did not respond. `image_pull_progress` remained absent. This narrows
the verified boundary to the runner/bootstrap layer; it does not establish why
the remote runner service failed to answer.

The installed dstack 0.20.27 implementation allows RunPod jobs 1200 seconds
from submission for runner availability. A blocked runner check delayed A2's
observed termination past that nominal age. Increasing the timeout alone is
not justified by this evidence because the service showed no progress during
the additional interval.

Before D1, the local dstack tool and controller were upgraded from 0.20.27 to
0.20.29 after a private, integrity-checked database and configuration backup.
The live controller reported version 0.20.29; SQLite integrity was `ok` at
Alembic head `4d3cbb932bb2`. D1 then exercised a fresh fleet, the exact pinned
image, repository code transfer, runner health, and task startup without
secrets, datasets, model download, or inference.

The first remote diagnostic probe had its own portability defect:
`datetime.UTC` made the fixed probe exit 1 on the remote host. Replacing only
that expression with `datetime.timezone.utc` made the same allowlisted
interactive probe exit 0. It reported a live `dstack-runner` process and HTTP
200 from `/api/healthcheck`; the 48-byte response was represented only by its
byte count and SHA-256 digest. That successful interactive observation was not
appended before task teardown. The retained mode-0600 `remote.jsonl` therefore
contains 19 failure-only events and no process or HTTP-200 observation. Treat
the successful probe as a transcript-backed operational observation, not an
immutable machine artifact. The exact remote Python version was not captured,
so the version-specific explanation is an inference from the one-change A/B
probe, not a direct environment observation. This probe defect did not affect
the dstack task, which completed independently.

Controller-side exception hooks were implemented and reviewed but were not
active for D1 because other concurrent experiments owned the shared controller
on port 3100. No controller exception occurred in D1, so this limits only the
diagnostic evidence, not the successful task result. Raw runner logs were not
read.

A3 added runner-side and controller-side timing without reading raw runner
diagnostics. The runner was healthy and its local healthcheck returned HTTP
200, while dstack serialized 445 mapped files totaling about 318 MB at roughly
3.3 MB per ten seconds. The transfer could not complete near the runner-startup
window, so A3 was stopped before model inference. This was the missing causal
boundary: the large evidence mapping, not model startup, was consuming the
window. The A4 bundle design followed directly from that measurement.

## Evidence validity and disposition

- Historical A1–A3 evidence validity: **Incomplete** for model quality and
  valid for the stated infrastructure boundaries.
- D1 infrastructure validity: dstack's terminal `all_jobs_done`, cost, and the
  task's start/done sentinels are valid for one runner/code-transfer/task-start
  observation. The corrected live-probe success lacks a retained event, and
  the controller hook was not active; both are explicit audit gaps.
- Decision disposition: superseded for the pending scientific question by the
  complete A4 result; retain this record for failure causality and spend.
- Gate outcome: A1/A2/A3 produced 0/60 scientific outputs. A4 later produced
  60/60 and is scored in its own evidence record.
- Artifact outcome: both full-run private prefixes contained no files. The A2
  collector found no `SHA256SUMS`, published no local bundle, and removed its
  temporary directory.
- Historical spend through C1 was `$6.0965`. Adding A3 (`$1.2035`), the A4
  bundle canary (`$0.2958`), and A4 dev inference (`$2.9943`) makes cumulative
  observed program spend `$10.5901`.
- Cleanup: A2, D1, the usefulness diagnostic, A3, and A4 dev environments are
  terminated; both C1 task canaries completed normally.
- Supersession: A4 supersedes only the pending retry/quality unknown. It does
  not replace the earlier quality-10 result, failed-run history, or existing
  frozen-30 baseline.

## Resolution

A3 measured bulk mapped-file transport as the bottleneck. A4 preserved the
scientific contract, moved only label-free evidence transport to one immutable
private bundle, passed an exact normal-task canary, completed 60/60 outputs in
a dev environment, and produced a checksum-verified private artifact bundle.
No further runner retry is needed for this question. Follow the
[A4 result](2026-07-31-qwen3-omni-frozen30.md) for the quality decision and next
experiment.

## Tracked provenance

- Original design:
  [`2026-07-30-qwen3-omni-frozen30-design.md`](../superpowers/specs/2026-07-30-qwen3-omni-frozen30-design.md)
- A2 recovery design:
  [`2026-07-30-qwen3-omni-frozen30-a2-recovery-design.md`](../superpowers/specs/2026-07-30-qwen3-omni-frozen30-a2-recovery-design.md)
- A2 recovery execution plan:
  [`2026-07-30-qwen3-omni-frozen30-a2-recovery.md`](../superpowers/plans/2026-07-30-qwen3-omni-frozen30-a2-recovery.md)
- Sealed A2 implementation commit: `26b9a72`.
- D1 submitted implementation/config revision:
  `a79bacc39fdb60c3cca405d9009ed5366390729a`.
- D1 uploaded script SHA-256:
  `13d21a2e1c2ae4f5a143531ac2a6e6e3eae75434342c0efb3d94386fbcde3228`.
- D1 fleet-config SHA-256:
  `b90aa457ab30ac232f05531728cabfbc0fa52119641f7d83730d49f5e8eb9fca`.
- D1 task-config SHA-256:
  `4fcbdd03dc11732bd03ab9b1a103278b05a5981080931bb2d75a3500bd191c00`.
- Retained private failure-only remote stream: 19 mode-0600 JSONL events,
  SHA-256
  `3665f02ed78e63d168f23794a26fbbdd03964eac5cba3066b91816ca3d294f89`.
- Portable probe-fix revision:
  `1cd2f1984d187e7e42182483626764c07506f7ed`.
- Final sealed instrumentation revision:
  `f3bbf2d99e036ecdc5b5d5a7e54d7caf8dcaa24f`,
  script SHA-256
  `e960740a723cf53984489a6d38c138cc9bea7961b952981f9b82c2b4c71c84e8`.
- C1 and A3 submitted implementation/config revision:
  `b1da5e8b11cbd47ddc9f3fcfb63939c4a39e0f3d`.
- C1 task-config SHA-256:
  `5ba28a80ce2e72306fc7155b39f2aa3c6d3c1c4a193ad4f9d6f074a4e77e0595`.
- C1 transferred frozen-30 runner SHA-256:
  `691ffba8924b229f415fcf34e1c3f435a56889432d6aa05116fa2591c8c2504d`.
- A4 private-bundle implementation commits: `6b2af58` and `2c12d7b`.
- A4 contract SHA-256:
  `9cfa26eacecde5bd8c916823ce2a59e74ded7f71ae978b973e306a0525b24bb5`.
- A4 bundle manifest/archive SHA-256:
  `3f88bb004c9d95c433e544ddf0cf6b314deffd0d5a22e5adb2d3e1d09613ca74`
  and
  `9f94a991ecdf5a4722aabb6e2a8f62a761ffdca34bc3ec08724b509668ed5943`.
- A4 sanitized score commit: `771b4ae`; score SHA-256:
  `2344e96361112e06142fdf588bad448a2cdcc3314576313f18bc7cbce1c0be73`.
- Canonical sanitized D1 evidence record: this file; immutable committed
  snapshot `23be2fa25507a7afd37a66614182c3eaf1eec989`. A web link is deferred
  until authorized branch integration.
