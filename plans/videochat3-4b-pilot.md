# VideoChat3-4B dstack pilot

## Objective and ownership

- **Primary theme:** Models and adaptation.
- **Question:** On the declared temporal/action failures, does
  `MCG-NJU/VideoChat3-4B` improve semantic C/P/W grade relative to a matched
  `Qwen/Qwen3.5-4B` rerun without regressing either visual control?
- **Operational owner:** branch `codex/videochat3-4b-pilot`.
- **Evidence boundary:** private requests, frames, predictions, grader packets,
  and logs remain under the immutable Hugging Face Bucket run prefix. This file
  is the single sanitized repository evidence record.

## Frozen pre-inference declaration

- **Treatment:** `MCG-NJU/VideoChat3-4B` revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`.
- **Control:** `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Runtime:** NVIDIA CUDA only, BF16 model parameters, no CPU or ROCm fallback;
  exact digest-pinned image recorded by the dstack configs.
- **Prompt:** the existing controlled VQA instruction: the question followed by
  `Answer with one short phrase or sentence. Return only the answer; do not
  explain your reasoning.`
- **Decoding:** seed `20260724`, greedy (`do_sample=False`), 32 new-token cap.
- **Evidence:** exactly the cached 12 chronological frames already bound to
  each label-free request, retaining ordered `frame-01.png` through
  `frame-12.png`, the request timestamps, and per-frame checksums. VideoChat3
  receives one 12-frame tensor/list video plus metadata; it does not decode or
  resample the source video.
- **Compatibility gate:** the three examples declared by
  `configs/vqa/three_video_gate.json`; pass only on 3/3 completed, non-empty,
  short answers with CUDA BF16 attestation.
- **Temporal/action targets:** the four ordered target rows declared in
  `configs/vqa/failure_panel_12.json`.
- **Visual controls:** the two ordered control rows declared in that same
  manifest. The immutable six-row request manifest hash binds their identities
  and order without republishing them in this sanitized plan.
- **Completion rule:** both models must complete all six matched rows.
- **Quality rule:** pass only if at least two of four target rows improve by at
  least one C/P/W grade and neither visual control regresses.
- **Grading:** freeze both prediction sets before reference access; obtain two
  independent blind C/P/W grades and a third blind adjudication only for
  disagreements.
- **Negative result:** a technically valid failure of the quality rule ends the
  track; no quality-driven rerun is allowed.

## Immutable execution and budget

- Original canary: `videochat3-runner-canary-20260730-a1`; the single diagnosed
  corrective attempt is `videochat3-runner-canary-20260730-a2`.
- Reauthorized canary: `videochat3-runner-canary-20260730-a3`; this new
  immutable identity was explicitly authorized after a2 was retired and is now
  itself consumed and permanently retired.
- Second reauthorized correction: `videochat3-runner-canary-20260730-a4`. After
  reviewing the a3 resolver defect, the user explicitly authorized fixing it
  and trying once more. At that point no `a5` or other retry was authorized;
  the later a5 authorization below supersedes that stop.
- Third reauthorized correction: `videochat3-runner-canary-20260730-a5`. After
  reviewing the a4 preview mismatch, the user explicitly authorized fixing and
  trying again. At that point no `a6` or other retry was authorized; the later
  2026-07-31 authorization below supersedes that stop.
- Interactive reopening: on 2026-07-31 the user explicitly authorized fixing
  and trying again and asked for interactive debugging. The one bounded debug
  environment is `videochat3-debug-20260731-a1`; the only new immutable task
  identity is `videochat3-runner-canary-20260731-a6`. No `a7`, second debug
  environment, or replacement pilot identity was authorized under that path.
- Runtime-certification reopening: after the historical debug stop, the user
  explicitly authorized the layered local/remote prerequisite validation and
  asked to try it on dstack. The bounded identities were
  `videochat3-runtime-cert-20260731-a1` and one corrective
  `videochat3-runtime-cert-20260731-a2`; no `a3` was authorized.
- Dev-only a3 reopening: after reviewing the retained generated dstack job,
  the user explicitly asked to use a dev environment and see the model result.
  This authorizes only `videochat3-runtime-cert-20260731-a3`; it does not revive
  either retired runtime-cert identity or the private pilot.
- Pilot: `videochat3-pilot-20260730-a1`.
- `-a2` is reserved for at most one corrective technical attempt after a
  diagnosed first infrastructure/runtime failure.
- Original allocation: `$5.00`. On 2026-07-30 the user explicitly authorized a
  total spend up to `$10.00` to obtain the result.
- Actual a2 cost was `$0.0231`; a3 cost was `$0.0283`; a4 was not submitted and
  cost `$0.0000`; a5 cost was `$0.0283`. Cumulative VideoChat3 cost is
  `$0.0797`, below the reauthorized total. The debug environment and a6 canary
  each have hard ceilings of `$1.00/hour × 0.5h`; the unchanged pilot ceiling
  is `$1.25/hour × 3h`. Prior actual cost plus all three ceilings is `$4.8297`,
  below the `$10.00` authorization. The later runtime-certification a1 and a2
  environments cost `$0.0304` and `$0.0269`, respectively. Their `$0.0573`
  incremental cost brings cumulative actual VideoChat3 spend to `$0.1370`.
  The a3 actual cost was `$0.2165`; cumulative VideoChat3 cost is `$0.3535`,
  below the `$10.00` authorization.
  The a6 canary and pilot were not submitted and cost `$0.0000`.
- Every config is previewed with `echo "n" | dstack apply -f ...`; full preview
  output is retained privately. Submission is detached and the workload stays
  foregrounded. Native bounded `no-capacity` retry is the only capacity retry.
- Each historical persistent stop was superseded only by the later explicit
  immutable authorization recorded here. The a3 dev result is final for this
  path; its identity and the private pilot are retired.

## A4 corrective declaration frozen before execution

- **Single changed runtime condition:** replace the conflicting Click `8.3.1`
  pin with Click `8.4.2`, remove the unrelated preinstalled Spin CLI before
  dependency resolution, and require the complete uv environment check to
  pass.
- **Held fixed:** exact CUDA image, VideoChat3 and Qwen revisions, prompt,
  seed, greedy 32-token decoding, cached evidence, manifest hashes, row IDs,
  compatibility threshold, completion denominator, quality rule, grading
  protocol, providers, hardware selector, price bounds, and artifact boundary.
- **Canary identity:** `videochat3-runner-canary-20260730-a4`; a no-media,
  exact-image CUDA/runner canary. It is submitted at most once after a declined
  preview shows acceptable physically distinct offers.
- **Conditional pilot identity:** `videochat3-pilot-20260730-a1`, still unused.
  It may run only if a4 passes, and its own compatibility gate must pass `3/3`
  before the matched six-row workload.
- **Stop rule:** any a4 infrastructure/runtime failure ends this reopened path.
  A technically valid pilot result is never rerun because quality is poor.

## A5 corrective declaration frozen before execution

- **Diagnosis:** an exact full-resource `dstack offer` query and three
  preview-only no-submission probes successively covering the image/resources,
  file-transfer/working-directory layer, and complete environment/secret layer
  each showed the same two compatible RunPod A40 offers. No a4 task field
  reproduced the prior empty preview.
- **Single changed execution condition:** use the newly authorized immutable a5
  identity after the capacity probes; retain the corrected Click/Spin runtime
  unchanged. Hardware, providers, price, duration, files, model revisions, and
  all scientific conditions remain fixed.
- **Canary identity:** `videochat3-runner-canary-20260730-a5`; a no-media,
  exact-image CUDA/runner canary submitted at most once after its own declined
  preview shows acceptable physically distinct offers.
- **Conditional pilot:** the unused `videochat3-pilot-20260730-a1` remains
  conditional on a5 success and then its own `3/3` compatibility gate.
- **Stop rule:** any a5 infrastructure/runtime failure ends this reopened path.
  No `a6`, external apply loop, quality-driven rerun, price increase, provider
  expansion, or hardware broadening is authorized.

## A6 interactive-debug declaration frozen before execution

- **Root cause carried forward:** a5 proved dependency resolution, pinned model
  download, CUDA weight loading, and then failed because the adapter passed a
  custom remote-code metadata class outside the pinned processor's accepted
  Transformers `VideoMetadata`/dictionary contract. The local correction now
  subclasses `transformers.video_utils.VideoMetadata` and omits non-standard
  constructor fields.
- **Interactive gate:** submit `videochat3-debug-20260731-a1` at most once as a
  dstack dev environment using the exact digest-pinned CUDA image, scalar A40
  selector, provider pool, and corrected dependency snapshot. It receives no
  private media or frozen manifests. Initialization may download only the
  pinned VideoChat3 revision; interactive commands may exercise only the
  synthetic 12-frame canary. The environment is bounded by `$1.00/hour`, 30
  running minutes, and 10 inactive minutes and is stopped immediately after a
  conclusive pass or persistent technical failure.
- **Interactive purpose:** attach/SSH permits repeated diagnostic commands on
  the same provisioned instance without consuming additional immutable task
  identities. Code changes still require local RED/GREEN tests and an exact
  uploaded snapshot; interactive edits are diagnostic only and are never the
  evidence of record.
- **Immutable a6 canary:** only after the interactive synthetic canary passes,
  preview and submit `videochat3-runner-canary-20260731-a6` at most once. It is
  the normal foreground, no-private-media task with durable failure/success
  artifacts at its matching immutable HF Bucket prefix.
- **Conditional pilot:** only an a6 terminal success permits previewing and
  submitting the still-unused `videochat3-pilot-20260730-a1`; its existing
  frozen manifests, evidence, model/control revisions, prompt, seed, decoding,
  panel, and decision rule remain unchanged.
- **Stop rule:** a debug-environment infrastructure failure allows no second
  debug environment. A processor/model contract that cannot be made to pass in
  that environment stops before a6. Any a6 infrastructure/runtime failure ends
  the path with no a7. A technically valid negative result is never rerun for
  quality.

## Runtime-certification reopening frozen before execution

- **Purpose:** validate packaging and the real adapter path cheaply before any
  immutable canary. Exact-image CPU and advisory ROCm smoke lanes were local;
  only a bounded NVIDIA A40 environment could certify CUDA BF16 inference.
- **Frozen runtime:** image
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`,
  VideoChat3 revision `37fa901ec5913f84bc31108ebc1e60ad1903634c`,
  Transformers revision `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`,
  requirements SHA-256
  `9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6`,
  and source snapshot SHA-256
  `3039526f2ace8104fb25a1a989d3046281a5499c39a6b4ded439481c20eb9529`.
- **Synthetic contract:** twelve deterministic `32 × 32` RGB frames, exact
  timestamps `0.0` through `11.0`, seed `20260724`, greedy decoding, and a
  32-token cap. No competition row, manifest, private media, reference answer,
  or grade entered either environment.
- **Bounds:** scalar `A40:1:24GB..`, RunPod/Vast.ai on-demand, `$1.00/hour`,
  30-minute maximum, 10-minute inactivity limit, retry off. The a2 identity was
  the sole corrective provisioned-runtime attempt; no a3 was authorized.
- **Stop rule:** certification requires one complete synthetic answer and a
  hash-bound sanitized certificate. A repeated environment lifecycle failure
  stops before any immutable canary or pilot.

## A3 dev-only reopening frozen before execution

- **Diagnosed cause:** dstack generated a wrapper that runs initialization and
  then starts its SSH keep-alive. The a1/a2 init item used `exec bash`, replacing
  that wrapper shell; therefore its keep-alive was unreachable and each job
  ended immediately after successful preparation.
- **Single change:** a3 uses `bash` without `exec` for initialization. Exact
  image, source snapshot, requirements, model and Transformers revisions,
  scalar A40 selector, provider pool, price, duration, seed, timestamps,
  decoding, and no-private-media boundary remain fixed.
- **Result boundary:** after SSH attachment, run one synthetic 12-frame model
  inference. Freeze the raw synthetic answer privately, emit the sanitized
  hash-bound certificate, upload both with a checksum manifest, then list,
  download, and checksum-verify the private evidence class. This is compatibility-only evidence; it does
  not evaluate competition-row quality.
- **Identity and stop rule:** submit
  `videochat3-runtime-cert-20260731-a3` at most once after a declined preview
  shows acceptable offers. Stop immediately after a conclusive pass or
  failure. No a4, private pilot, or quality-driven retry is authorized.

## Implementation plan

- [x] Add failing tests for the VideoChat3 video tensor/metadata adapter,
  label-free six-row projection, CUDA BF16 attestation, immutable configs,
  artifact finalization, and joint cost bound.
- [x] Implement the minimal model-specific runner and foreground dstack wrapper.
- [x] Add exact-image no-media canary and private gate/pilot configs.
- [x] Freeze and hash label-free gate and six-row inputs before preview.
- [x] Run the authorized a3 canary and stop before compatibility, pilot, or
  grading when it fails.
- [x] List, download, and checksum-verify the a3 private failure bundle.
- [x] Retire the consumed a3 submission surface and prohibit resubmission.
- [x] Add the corrected `VideoMetadata` contract and verify it in the exact
  no-token/no-model/no-media container.
- [x] Predeclare the bounded interactive debug environment and a6 canary.
- [x] Stop without submission after the initial and single corrective debug
  previews both failed at the local control-plane connection boundary.
- [x] Retire the unsubmitted debug and a6 identities and prohibit submission.
- [x] Run exact-image CPU and advisory ROCm synthetic smoke lanes without local
  model weights or private media.
- [x] Preview and run the bounded runtime-certification a1 environment; after
  its terminal lifecycle failure, use only the predeclared a2 correction.
- [x] Stop after a2 repeated the terminal lifecycle failure, preserve costs and
  credential-safe logs, and retire both identities with no a3 or pilot.
- [x] Run the explicitly reauthorized a3 dev-only synthetic result, verify its
  private artifact bundle, and retire the identity without reviving the pilot.
- [x] Run model-specific tests and `git diff --check`, review, and commit.

## Results and disposition

- **Observed result:** a3 proved the corrected dstack keep-alive, loaded the
  exact VideoChat3 revision on a RunPod A40 with CUDA BF16 parameters, accepted
  the synthetic 12-frame tensor plus exact metadata, ran the model-provided
  SDPA vision path, and completed greedy generation. The frozen 32-token output
  was an incoherent multilingual token sequence rather than a meaningful short
  answer. The model load also reported `lm_head.weight` missing and newly
  initialized, which is the leading explanation for the unusable output.
- **Evidence validity:** **Compatibility-only**. Runtime certification completion was `1/1`:
  one prediction, sanitized certificate, and checksum
  manifest were frozen and independently downloaded and verified. The private
  competition compatibility remained `0/3`; matched pilot completion remained VideoChat3 `0/6`
  and Qwen control `0/6`. No competition prediction, reference
  access, grading, or adjudication occurred.
- **Decision disposition:** **Not promoted under the tested condition**. The
  exact pinned model/runtime combination can execute but did not produce a
  credible synthetic answer. The consumed a3 identity is permanently retired,
  and the private compatibility gate and pilot remain retired. No a4 or
  quality-driven rerun is authorized.
- **Decision rule and predeclaration:** the `3/3` compatibility, `6/6` matched
  completion, two-of-four target improvement, and no-control-regression rules
  remain predeclared and unevaluated; the synthetic result cannot satisfy or
  fail those competition-quality rules.
- **What this rules out now:** inability to load the pinned checkpoint in BF16,
  inability to process the declared tensor-plus-metadata video form, absence of
  a working non-FlashAttention vision path, inability to generate tokens, or
  inability to durably freeze and recover the synthetic result. It also rules
  out promotion of this exact model/runtime combination on the present evidence.
- **What this does not rule out:** meaningful output from a different official
  dependency combination, a checkpoint export whose language-model head loads
  without warning, quality on any competition row, a `3/3` compatibility-gate
  pass, or improvement over the pinned Qwen control.
- **Possible defect, audit gap, or uncertainty:** the dstack `exec` keep-alive
  defect and the processor/generation/attention adapter defects were confirmed
  and corrected test-first inside the one mutable dev environment. The missing,
  newly initialized `lm_head.weight` is a confirmed load warning; whether its
  root cause is the checkpoint export or this loader/revision pairing remains
  uncertain. **Audit gap:** the final frozen output and provenance are complete,
  but the raw interactive terminal transcript and intermediate failed output
  directories were not uploaded before the instance was stopped, so their full
  text and timing cannot be reconstructed from the immutable bundle.
- **Frozen declaration:** this file's pre-inference SHA-256 was
  `29d311cb35e6c0468e728ed484d7b4adce4c075bd084d57d624e9447c5286c75`.
  Label-free gate and pilot inputs were frozen before any preview with SHA-256
  `7160e3c4e3bd7b2641004b4bc1331841828b4d1b2e9e89b0a0dee718512c6ebf`
  and `3ab0fc1dc36373914b42c44efdb8b305cb0a1536706995377e5c4513467ce403`,
  respectively.
- **Runtime identity:** image
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`;
  Transformers source commit
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`; VideoChat3 model revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`; Qwen control revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **First gate:** the `-a1` list-selector preview returned no offers and was not
  submitted. The diagnosed selector compatibility correction used scalar
  `A40:1:24GB..` for the single allowed `-a2` attempt.
- **Corrective preview:** two physically distinct compatible on-demand RunPod
  offers were available: NVIDIA A40 48 GB in `CA-MTL-1` and `EU-SE-1`, each at
  `$0.44/hour`. The configured canary ceiling remained `$1/hour` for 30 minutes.
- **Corrective execution:** `videochat3-runner-canary-20260730-a2` ran on the
  `CA-MTL-1` A40 offer and failed before model download or media access with
  container exit status 1. The exact-image dependency check found that `spin`
  requires Click `<8.4`, while the executed resolution installed Click `8.4.2`.
  Actual dstack cost was `$0.0231`; no reserve funds were used.
- **Executed dependency snapshot:** the failed container installed `uv 0.11.3`
  and used the uncorrected declared set: Transformers source commit
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7` (reported as
  `5.15.0.dev0`), `accelerate 1.12.0`, `huggingface-hub 1.24.0`,
  `pillow 12.2.0`, `safetensors 0.8.0`, `sentencepiece 0.2.1`,
  `torch 2.10.0`, and `torchvision 0.25.0`. The resolver replaced Click
  `8.3.1` with `8.4.2`; the check then reported the installed `spin` package's
  constraint `click>=8,!=8.3.0,<8.4`. The `spin` version and the complete
  preinstalled base-image package set were not recorded.
- **Missing immutable failure bundle:** the intended private evidence bundle was
  never created.
  The failure occurred before artifact finalization/upload, so there is no
  remote artifact to download or checksum-verify; a credential-safe post-run
  Bucket listing completed successfully and returned no objects.
- **What cannot be reconstructed:** no immutable bundle preserves the
  container-produced terminal-status, frozen-plan-digest, or checksum-manifest
  payloads; consequently the exact uploaded wrapper/source/requirements/plan
  bytes cannot be bound to that failed execution through remote checksums. The
  complete transitive environment also cannot be reconstructed from the
  retained resolver excerpt. Current post-stop files must not be mistaken for
  the exact failed config/runtime snapshot.
- **A3 preview and offer evidence:** the full declined preview is retained
  privately with SHA-256
  `45d0c5825091a06dc5ede1954574e1b1520fea45f3e2867b119c260918de75b3`.
  It found physically distinct RunPod NVIDIA A40 48 GB offers in `CA-MTL-1`
  and `EU-SE-1`, each at `$0.44/hour`, within the hard `$1.00/hour` and
  30-minute canary bounds.
- **A3 execution and cost evidence:**
  `videochat3-runner-canary-20260730-a3`, run ID
  `90fb6f5d-a909-4cfd-814d-a004d69392c7`, was submitted at
  `2026-07-31T01:29:18.687606Z` and selected the RunPod `CA-MTL-1` NVIDIA A40
  48 GB offer at `$0.44/hour`. It exited 1 with
  `container_exited_with_error`. Incremental cost was `$0.0283`; cumulative
  VideoChat3 cost was `$0.0514`.
- **A3 submitted snapshot:** the exact submitted config had SHA-256
  `252afefc9a7e8c55fb647162c6a8384b780b320f249ba5e1a22a3c7f422b7aa1`
  and the declared requirements had SHA-256
  `959a2aac3e8847aff94c8f302cd135d5c76b3e1e3dcd0c71c1921f5aa58569fd`.
  Sanitized non-runnable provenance is preserved in
  `dstack/videochat3-runner-canary-20260730-a3.retired.yml`; execution fields
  that could submit work or reference credentials are intentionally absent.
- **A3 immutable failure bundle:** the four-object private evidence bundle
  was listed, downloaded, and checksum-verified. Its manifest has SHA-256
  `afc9b60771d972a3a762c08bce22653de72c69a87ce6452277a6e0e8981e02aa`;
  it verified frozen-plan
  (`6e9db2ce1802cf79a8284f10f8a5488d98ac03534b112dfa6baa0b532d2f2377`),
  terminal-phase
  (`0a3fa093ea56fcb145fbfeb52492530620001098a19ad47bba4fa74ed396f168`),
  and terminal-status
  (`94ede509e693cfc2dcaf4777122d29ccb18ef3d790bda66844c5da7745e594b0`).
  The terminal record reports pipeline return code 1 and phase
  `bootstrap-runtime`. Because this was a no-media canary, the bundle contains
  no gate/pilot manifest verification and provides no model compatibility or
  quality evidence.
- **Post-stop state:** the declared Click pin is retained only as executed
  a3 provenance. The corrected runtime removes the unrelated base-image Spin
  CLI and pins Click `8.4.2`; the exact-image local package check passed, but
  this was not a CUDA/model compatibility test. The exact gate and pilot
  manifest hashes remain config-pinned and pre-load verified, but those
  safeguards were not exercised remotely.
- **Consumed-config retirement:** the runnable a2 canary config was replaced by
  `dstack/videochat3-runner-canary-20260730-a2.retired.yml`, a sanitized
  non-task provenance document with `submission_allowed: false`. It preserves
  the executed snapshot above, contains no secret reference or commands, and
  cannot submit a replacement identity.
- **Historical persistent stop:** after reviewing the a2 audit gap, the user
  explicitly
  authorized up to `$10.00` total and the new immutable a3 canary. A3 then
  failed terminally. Both a2 and a3 identities remain permanently retired, and
  the a3 submission surface remains removed. The later direct a4 authorization
  supersedes only the no-retry decision; it does not alter the incomplete a3
  evidence or authorize any identity after a4.
- **A4 local dependency evidence:** a disposable no-token, no-media,
  no-model-download container used the exact CUDA image digest. It removed
  preinstalled `spin 0.16`, installed `uv 0.11.3`, resolved 57 packages, and
  passed `uv pip check` across 94 installed packages. Attested versions were
  Click `8.4.2`, Hugging Face Hub `1.24.0`, Transformers `5.15.0.dev0`, Torch
  `2.10.0+cu128`, and torchvision `0.25.0+cu128`. This is dependency evidence
  only; the local AMD host cannot provide the required NVIDIA CUDA execution.
- **A4 preview evidence:** the exact a4 config SHA-256 was
  `64e54e924133a2ca94c058320e285e8d8dd5f9e58946981eac7faf37b0f5d3f4`;
  corrected requirements SHA-256 was
  `9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6`.
  Both full preview outputs were retained privately and have SHA-256
  `38dc34fd45d7f97d310c6a9b5b92f0123754963157e8e2c7b97eb8ae57c684c1`.
  Each returned `No matching instance offers available`. Between them, the
  scalar `A40:1:24GB..` offer query listed two physically distinct RunPod A40
  48 GB offers, `CA-MTL-1` and `EU-SE-1`, each at `$0.44/hour`.
- **A4 submission and cost:** `videochat3-runner-canary-20260730-a4` was never
  submitted, has no run ID or remote artifact, and incurred `$0.0000`.
  Cumulative VideoChat3 cost remains `$0.0514`. Its sanitized, non-runnable
  provenance is
  `dstack/videochat3-runner-canary-20260730-a4.retired.yml`.
- **Historical a4 stop:** a4 failed the infrastructure preview gate twice, so
  no pilot, predictions, or grading occurred. The later direct a5
  authorization supersedes only the no-retry decision; it does not change the
  incomplete a4 evidence or authorize any identity after a5.
- **A5 preview and offer evidence:** exact a5 config SHA-256 was
  `8ade2881e16a70722beefe6b2734a35339bdfffecf9489c6a67a98e1dcf7f64f`.
  The retained declined preview has SHA-256
  `13a5fb70c02cc868304e9eb070efaedb3cd00933268df527fbc36eac06e21474`
  and showed physically distinct RunPod NVIDIA A40 48 GB offers in
  `CA-MTL-1` and `EU-SE-1`, each at `$0.44/hour`, within the hard
  `$1.00/hour` and 30-minute bounds.
- **A5 execution and cost evidence:**
  `videochat3-runner-canary-20260730-a5`, run ID
  `02b9d381-15a4-4092-9728-5329f759513c`, was submitted at
  `2026-07-31T02:21:47.026717Z` and selected the RunPod `CA-MTL-1` NVIDIA A40
  48 GB offer at `$0.44/hour`. It exited 1 with
  `container_exited_with_error`. Incremental cost was `$0.0283`; cumulative
  VideoChat3 cost is `$0.0797`.
- **A5 workload failure:** the corrected uv bootstrap passed the complete
  installed-package check and the pinned VideoChat3 revision downloaded. Model
  weights loaded, then the processor rejected
  `[ExactTimestampMetadata(...)]` for `video_metadata`; the pinned contract
  accepts `VideoMetadata`, `dict`, lists of either, or nested lists of either.
  No prediction was produced.
- **A5 immutable failure bundle:** the six-object private evidence bundle
  was listed, downloaded, and checksum-verified. Its manifest has SHA-256
  `4a1765a9ce2414c2186660ad4499969b7ec9155774ee574f99206a9b626fc3b7`;
  it verified frozen-plan, terminal-phase, terminal-status, and two workload
  stream payloads. The terminal
  record reports pipeline return code 1 and phase `workload`.
- **A5 retirement:** the runnable a5 config was replaced by
  `dstack/videochat3-runner-canary-20260730-a5.retired.yml`, sanitized
  non-task provenance with `submission_allowed: false`.
- **Post-stop adapter hardening:** without authorizing or submitting another
  run, the loader now subclasses the pinned processor's accepted
  `transformers.video_utils.VideoMetadata` type instead of the remote
  `VideoChat3VideoMetadata` type and passes only the standard constructor
  fields. Regression tests preserve the exact timestamps and frame indices.
  A disposable exact-image, no-token, no-model, no-media container passed the
  94-package environment check and the corrected metadata type/constructor
  contract. This hardening was not executed remotely and does not convert a5
  into compatibility evidence.
- **Historical a5 stop:** a5 failed the workload gate, so no pilot,
  predictions, or grading occurred. The later direct interactive/a6
  authorization superseded only the no-retry decision; it did not change the
  incomplete a5 evidence.
- **2026-07-31 local contract evidence:** the exact digest-pinned container,
  without a token, model, or media, resolved the fixed requirements with uv,
  passed the 94-package environment check, and instantiated the corrected
  Transformers `VideoMetadata` subclass while preserving all 12 synthetic
  indices and timestamps. This is dependency/adapter evidence only, not CUDA
  model compatibility evidence.
- **Interactive preview evidence:** frozen plan SHA-256 was
  `a519fbcd9e358c2f18d6f7209e9cef45cba95f97306449b32a78972246f0e6e5`;
  the debug source config SHA-256 was
  `c76304e7997b8c93fc021383dcdc44481a0d57c31688fb469de10655c9d3ad5e`.
  Both retained preview outputs have SHA-256
  `8ec63b7b920423b7a978cd8cdb26b045bd646f13c4411599eef78a97fe9a48ad`
  and contain the same failure to connect to `http://127.0.0.1:3100`. No offer
  table or provisioning request was produced.
- **Corrective control-plane evidence:** before the first preview no listener
  or server process existed. The one corrective server start reached HTTP 200;
  the process and listener were absent again when the corrective preview ran.
  The replacement local server token was rotated after the temporary startup
  log disclosed its generated predecessor; the secret-bearing log was deleted
  and no credential value is recorded here.
- **A6 disposition:** source config SHA-256 was
  `e69aba8632801fd075f992b65bba56b73756ece7e886dcdc6209f7e0c4176b52`.
  Because the interactive prerequisite failed, a6 was not previewed or
  submitted. Debug and a6 incremental cloud cost was `$0.0000`; cumulative
  VideoChat3 cloud cost remains `$0.0797`. No remote artifact prefix exists for
  either unsubmitted identity.
- **Retirement:** sanitized non-runnable records are
  `dstack/videochat3-debug-20260731-a1.retired.yml` and
  `dstack/videochat3-runner-canary-20260731-a6.retired.yml`; both carry
  `submission_allowed: false` and no commands or secret references.
- **Historical interactive stop:** the earlier interactive infrastructure gate
  failed twice, so no a6 was submitted. The later direct runtime-certification
  authorization superseded only that no-retry decision; it did not change the
  incomplete historical evidence.
- **Runtime-certification local lanes:** the exact CUDA image passed the CPU
  synthetic adapter smoke with checksum `4700160`, shape `12 × 3 × 32 × 32`,
  and timestamps `0.0` through `11.0`; no GPU, token, model weight, or private
  file was used. The advisory ROCm image
  `docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
  passed the same adapter seam and BF16 tensor arithmetic on the Radeon 8060S.
  Neither local lane is NVIDIA compatibility evidence.
- **Runtime-certification control plane:** dstack client/server `0.20.29` ran
  through a durable user service on loopback. During repair, a credential was
  exposed in process arguments; it was rotated immediately, local CLI mode was
  verified as `0600`, and the replacement service no longer carries the
  credential in its arguments. The first a1 declined preview failed locally on
  the unsupported `entrypoint` field and cost `$0.0000`; its retained output
  SHA-256 is
  `8c3cd72bf831b3e57708f4aeafcc581fbd208189ef8ef95d163d6de1221103ff`.
- **Runtime-certification offer evidence:** the corrected a1 preview and the a2
  preview each showed physically distinct RunPod A40 48 GB offers in
  `CA-MTL-1` and `EU-SE-1` at `$0.44/hour`, with retry off, `$1.00/hour` max,
  and 30-minute max duration. Their retained preview SHA-256 values are
  `6e87ea453a020d8365887bf575cc290af5a2fd30463809c0b03408e03f4657ed`
  and `7f1bbe4ddd291f2a381ab45fdfda0bb15732f7e9aae5cf6d88695e0c69239fa6`.
- **Runtime-certification a1 execution:** run ID
  `e7d2c5aa-dc15-4f09-aa18-ec81970781eb` used the Montreal A40 offer. The
  submitted config SHA-256 was
  `54056c9d3a61e17d4172f15ce057f4357a6dae120489f726129bd0b8185368ae`.
  Dependency checking, pinned model download, and basic CUDA/BF16 readiness
  passed before the environment ended with `done_by_runner`, exit 0. The
  synthetic runner never started. Actual cost was `$0.0304`; cumulative
  VideoChat3 cost became `$0.1101`. The retained normal log SHA-256 is
  `1054d732a80bb14b1495d0c45266c998c02b564b8d5f35844b176e36d5bc7601`.
- **Runtime-certification a2 execution:** run ID
  `1724f5fa-a758-4c0d-a74a-1c8b756c3d0f` used the same Montreal A40 class.
  The submitted config SHA-256 was
  `c73979a54e64cd4af1e3768c2365fcca790e617bb268142d75e954afed81f6f9`.
  It repeated successful dependency checking, pinned model download, and
  basic CUDA/BF16 readiness, then ended without user interruption with
  `done_by_runner`, exit 0, before interactive certification. Actual cost was
  `$0.0269`; runtime-certification incremental cost was `$0.0573`, and
  cumulative VideoChat3 cost is `$0.1370`. The retained normal log SHA-256 is
  `4bcbe29186cf19b897813c0a67221a84dd9bbdfc8de1d567c29d88caa93ec21c`.
- **Runtime-certification artifact audit:** credential-safe post-run listings
  of the intended a1 and a2 Hugging Face prefixes returned no objects. Each
  retained empty listing has SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  There is therefore no certificate or checksum manifest to download or
  verify. This is an explicit audit gap, not a successful artifact check.
- **Runtime-certification retirement:** runnable a1 and a2 configs are absent.
  Sanitized non-runnable provenance is preserved in
  `dstack/videochat3-runtime-cert-20260731-a1.retired.yml` and
  `dstack/videochat3-runtime-cert-20260731-a2.retired.yml`; both carry
  `submission_allowed: false` and contain no runnable task fields or
  credential references. At that historical stop no a3 had been authorized.
- **A3 preview and offer evidence:** the mandatory declined preview showed two
  physically distinct compatible RunPod A40 48 GB offers, `CA-MTL-1` and
  `EU-SE-1`, each at `$0.44/hour`. Its retained output SHA-256 is
  `b0bc9bb05fbea198982d4115fec2492415382a0740c53fe27ea4491f9abe9295`.
  A3 selected the Montreal offer under the unchanged `$1.00/hour`, 30-minute,
  on-demand, retry-off bounds.
- **A3 execution and cost:** `videochat3-runtime-cert-20260731-a3`, run ID
  `a4661ded-80a9-4f4f-a210-8aee2ad19d2f`, ran from
  `2026-07-31T16:55:09.509162Z` until it was explicitly stopped at
  `2026-07-31T17:24:40.773139Z`. Dstack recorded `stopped_by_user`, actual cost
  `$0.2165`, runtime-certification incremental cost `$0.2738`, and cumulative
  VideoChat3 cost `$0.3535`.
- **A3 corrected dev provenance:** the submitted config SHA-256 was
  `4191cfc0bf872081932e08b3906853b0e2537feeeb49924b9a323445f0483356`
  at commit `2f1f1616935039cfcc63cb3302562361e1681ac4`, with submitted source
  snapshot `3039526f2ace8104fb25a1a989d3046281a5499c39a6b4ded439481c20eb9529`.
  Interactive test-covered fixes produced the executed source snapshot
  `dfe460cc993fc6d9dc77e53a9a16fc1b7ab023639714bbcaaf1c03fe7a6034a0`.
  The certificate provenance was corrected to that executed hash before
  upload; the frozen prediction itself was not changed or regenerated.
- **A3 private artifact verification:** the immutable private certificate bundle
  contains prediction, certificate, and checksum-manifest payloads. Their
  SHA-256 values are respectively
  `b7abd5c772def49fd97221fa5b65630520b97a892efd3a0f8e3fe4683106b9bd`,
  `d67e66fdd1a4b61272f1c6e61592f2d16ce6431e84074a64f07ff004c6006174`,
  and
  `76bc9bdd27dced81d82c615c48a069bc7437b1f0cd418216f2b51559e09a52a2`.
  A fresh credential-safe listing and download found exactly those three files,
  and `sha256sum -c` verified both payloads. Runtime certification completion
  was `1/1`; the answer was frozen at the greedy 32-token cap and sanitized here
  as an incoherent multilingual token sequence.
- **A3 retirement:** the runnable a3 config is absent. Its submitted snapshot,
  interactive source hash, offer, actual cost, completion, result boundary, and
  artifact checksums are preserved in sanitized non-runnable
  `dstack/videochat3-runtime-cert-20260731-a3.retired.yml`, which carries
  `submission_allowed: false` and contains no runnable fields or credential
  references. No a4 runtime-cert identity is authorized.
- **Pilot retirement:** the unsubmitted `videochat3-pilot-20260730-a1`
  configuration is also absent. Its frozen hashes, bounds, and `0/3` plus
  `0/6`/`0/6` completion denominators are preserved in sanitized non-runnable
  `dstack/videochat3-pilot-20260730-a1.retired.yml` with
  `submission_allowed: false`, no private paths, and `$0.0000` cost.
- **Historical runtime-certification stop:** runtime certification completed
  `0/1` twice at the environment level. The later direct a3 authorization
  superseded only the no-more-dev-environment decision. A3 completed the
  synthetic path but did not revive the immutable canary or private pilot; no
  prediction on competition rows, grading, or a4 is authorized.
- **Reconsider when:** only if a separately scoped investigation identifies an
  official checkpoint/runtime combination that loads the language-model head
  without warning and produces a coherent synthetic answer before any private
  media is introduced.

## A4 output-head hypothesis frozen before execution

- **Authorization and boundary:** the user explicitly authorized one additional
  dstack development test under the existing `$10.00` total cap. The immutable
  identity is `videochat3-runtime-cert-20260731-a4`. This test uses synthetic
  input only: no competition manifest, cached evidence, answer reference, or
  private media may be mounted or uploaded. No a5 runtime-cert identity is
  authorized.
- **Primary hypothesis:** at exact model revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`, the tensor index omits
  `lm_head.weight`, the remote model declares that weight tied, but the outer
  config sets `tie_word_embeddings` false. A3 therefore loaded a newly
  initialized output head and produced incoherent tokens even though the video
  token count, vision features, tensor-plus-metadata input, CUDA BF16 forward
  path, and greedy generation all completed.
- **One-variable correction:** after loading the same checkpoint, explicitly
  validate matching shape, dtype, and device, then assign the output embedding
  weight to the exact trained input-embedding parameter. Record whether it was
  previously tied and verify object identity after assignment. No model tensor
  is downloaded or executed locally.
- **Frozen runtime:** model `MCG-NJU/VideoChat3-4B` at
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`; Transformers source at
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`; image
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`;
  NVIDIA CUDA BF16 only, scalar `A40:1:24GB..`, RunPod/Vast.ai on demand,
  retry off, `$1.00/hour` maximum, and 30-minute maximum duration.
- **Frozen input and decoding:** the same deterministic byte-ramp tensor of
  shape `12 × 3 × 32 × 32`, timestamps `0.0` through `11.0`, question
  `What is in the synthetic video?`, seed `20260724`, greedy decoding, and a
  32-token cap used for A3. The A3 frozen result is the baseline; it is not
  regenerated.
- **Decision rule:** complete exactly one A4 generation, freeze and checksum
  the answer before inspection, and stop the environment. A coherent non-empty
  answer supports the missing-tie diagnosis; another incoherent token sequence
  rejects that correction as sufficient. Either outcome ends this diagnostic
  test and does not authorize the 3/3 competition gate or six-row pilot.
- **Cost bound:** prior cumulative VideoChat3 cost is `$0.3535`; this config's
  hard worst-case additional cost is `$0.5000`, for a cumulative ceiling of
  `$0.8535`, below the user-authorized `$10.00` cap.

## A4 output-head diagnostic result

- **Evidence validity:** **Diagnostic**. A4 completed `1/1` frozen synthetic
  generation after the exact pinned checkpoint again reported
  `lm_head.weight` missing and newly initialized. The output embedding had not
  been tied; the adapter verified matching shape `[151936, 2560]`, BF16 dtype,
  and CUDA device, then shared the exact trained input-embedding parameter.
- **Frozen result:** with the same byte-ramp tensor, timestamps, question, seed,
  greedy decoding, and 32-token cap as A3, the checksum-frozen A4 answer was
  `A person's face.` This is a coherent short sentence rather than A3's
  incoherent multilingual token sequence. Because the source is an artificial
  byte ramp rather than a labeled semantic clip, the content may be a
  hallucination and does not establish VideoChat3 answer quality.
- **Decision disposition:** **Diagnostic signal**. The controlled contrast
  supports the missing-output-head tie as the cause of the garbled text and
  makes the repaired loader eligible for a separately authorized semantic
  compatibility test. It does not promote VideoChat3 to the project default or
  authorize private-media execution. Competition compatibility remains `0/3`;
  matched pilot completion remains VideoChat3 `0/6` and Qwen `0/6`.
- **Immutable execution:** `videochat3-runtime-cert-20260731-a4`, run ID
  `8dbb7b72-4332-4690-b451-c67879268a0d`, used the RunPod Montreal NVIDIA A40
  48 GB offer at `$0.44/hour`. It ran from
  `2026-07-31T17:56:14.495559Z` until the explicit stop at
  `2026-07-31T17:59:47.370691Z`. A4 actual cost was `$0.0260`; cumulative
  VideoChat3 cost is `$0.3795`, within the user-authorized `$10.00` cap.
- **Preview and snapshot:** the declined preview showed physically distinct
  Montreal and Sweden RunPod A40 offers at `$0.44/hour`; its full retained
  output SHA-256 is
  `34f7b219a1a1c0d6573345ed7c44da31b672839c0da57ea096a858ad4182bac9`.
  The submitted config SHA-256 is
  `a28c35685ad9ba0a2021131bf5090bd432f3103f4a7eccd09d4e1abbbebc3c1a`
  at commit `486f18681b9ceb828ea80dfba37afae5edbc76f1`, with source snapshot
  `64076fdb3a4a474a19e1561e6a44c136a4358ad7699960731ca976d9f8782261`.
- **Private artifact verification:** the three-object private certificate bundle
  was listed after upload, downloaded to a fresh temporary directory, and
  verified with its manifest. Prediction, certificate, and checksum-manifest
  SHA-256 values are respectively
  `df7b04e33a9eb7435c0614295fddb34ef90c9c67b287b0f1f8747250201c8baa`,
  `f558eb3c48f5ec428e271bb62b6708f2abf15a347dbd5b5178a63add920a1fe6`,
  and
  `920fc6b2bfbebeac6530b3c967d13fd4c2c8447ebbfd91cd6565690f9bee81c0`.
- **Interactive connection correction:** the first convenience SSH lookup did
  not reach the remote environment. The supported dstack attach tunnel then
  connected; its fresh SSH shell lacked the declared task variables, so the
  one corrective attempt inherited the task process environment in memory.
  No credential value was printed or persisted, and generation ran only once.
- **Audit gap:** the immutable bundle reconstructs the prediction, certificate,
  manifest, exact pins, source snapshot, and output-head tie attestation. It
  does not contain the complete raw interactive transcript or exact per-line
  timing, so those details cannot be independently reconstructed from the
  bundle.
- **What this rules out:** under the exact tested runtime, the frozen
  tensor-plus-metadata input path alone does not explain the garbling; replacing
  the untrained output head with the checkpointed token embeddings produced
  coherent language with every other frozen inference variable unchanged.
- **What this does not rule out:** semantic hallucination on artificial input,
  poor competition-row quality, failure of the 3/3 gate, or failure to improve
  the predeclared six-row matched panel against Qwen.
- **Retirement:** the consumed runnable config is absent. Sanitized non-runnable
  provenance is preserved in
  `dstack/videochat3-runtime-cert-20260731-a4.retired.yml` with
  `submission_allowed: false`, no runnable task fields, and no credential
  references. No later runtime-cert identity is authorized.

## 2026-08-01 semantic gate and matched-six reopening frozen before inference

- **Authorization:** after reviewing the A4 output-head diagnosis, the user
  explicitly authorized the previously described real-video gate and the
  matched comparison only if that gate passes. Historical consumed and retired
  identities remain retired. Fresh immutable identities are
  `videochat3-gate-20260801-a1` and, conditionally,
  `videochat3-pilot-20260801-a1`. An `-a2` identity is reserved only for one
  diagnosed corrective infrastructure/runtime attempt; no quality-driven rerun
  is authorized.
- **Question and hypothesis:** does the repaired VideoChat3 loader produce
  valid short answers on all three compatibility rows, and if so does
  VideoChat3 improve at least two of four selected Temporal/Action failures
  relative to the exact pinned Qwen3.5-4B control without regressing either
  Visual control?
- **Gate panel:** exactly the three ordered compatibility rows declared in
  `configs/vqa/three_video_gate.json`. Gate manifest SHA-256 is
  `1f476fb3a0f7f5cac3bd3a790348bc6a3eed91c16311e79909f48f8c9760a772`.
  The gate uploads only those three cached evidence directories. It passes only
  with `3/3` completed, nonempty, single-line short answers; its predictions are
  frozen and durably verified before the matched-six config may be previewed.
- **Matched panel:** the four ordered target rows followed by two ordered
  Visual controls declared in `configs/vqa/failure_panel_12.json`. This exact
  order was selected from
  `configs/vqa/failure_panel_12.json` before inference. Pilot manifest SHA-256
  is `8b5cf944553ac000795b837090799f07041c8b86cc3ed9a997dc52a9ed4941c4`.
  The pilot uploads only those six cached evidence directories.
- **Treatment and control:** `MCG-NJU/VideoChat3-4B` revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c` with the A4-proven explicit
  output-head/input-embedding tie, versus `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Both conditions use the same
  six requests and exact evidence; Qwen receives the same twelve ordered frames
  as image messages because that is its supported form.
- **Held fixed:** exact cached 12-frame evidence, original frame indices and
  timestamps, controlled short-answer prompt, seed `20260724`, greedy decoding,
  32-token cap, CUDA BF16, no CPU/ROCm fallback, Transformers source revision
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`, and image
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`.
- **Integrity:** each complete request manifest and every frame/evidence hash is
  verified before model loading. Inference inputs are label-free. Predictions
  are written exclusively, made read-only, uploaded, listed, downloaded, and
  checksum-verified before any reference answer or grade is accessed. Only
  then are two independent blind C/P/W grades obtained, with a third blind
  adjudication only on disagreements.
- **Decision rule:** pass only with VideoChat3 gate completion `3/3`, matched
  completion VideoChat3 `6/6` and Qwen `6/6`, at least two of four targets
  improving by at least one grade, and neither Visual control regressing. A
  complete negative result ends the track and is never rerun for quality.
- **Operational bounds:** prior cumulative VideoChat3 cost is `$0.3795`. The
  gate is bounded by `$1.00/hour × 1h`; the conditional pilot by
  `$1.25/hour × 2h`. Their combined hard ceiling makes cumulative VideoChat3
  spend at most `$3.8795`, below the user-authorized `$10.00` total. Both use
  scalar `A40:1:24GB..`, RunPod/Vast.ai on demand, retry off, detached
  submission, and foreground workloads. A4 is the exact-image repaired-model
  synthetic canary; no additional synthetic run is performed.

## 2026-08-01 completed real-video gate and matched-six result

- **Observed result:** the declared real-video gate completed `3/3` with valid
  short answers. The conditional matched run then completed VideoChat3 `6/6`
  and Qwen `6/6`. Two independent answer-only blind graders agreed on `11/12`
  candidate judgments; a third blind reviewer adjudicated only the one
  disagreement. Final C/P/W aggregates were VideoChat3 `0C/2P/4W` and Qwen
  `2C/2P/2W`.
- **Evidence validity:** **Verified**. Both declared protocols completed; the
  full manifests and all exact cached frames were verified before model
  loading, inference stayed label-free, predictions were frozen and durably
  checksum-verified before references were accessed, and every candidate
  received the predeclared blind grading protocol.
- **Decision disposition:** **Not promoted under the tested condition**. The
  predeclared quality rule failed: only `1/4` targets improved by at least one
  grade, below the required `2/4`, and both Visual controls regressed, rather
  than the required `0/2`. This is a valid negative result; the track is final
  and no quality rerun is authorized.

- **Exact executed identities:** submitted snapshot commit
  `28f02ecdcf812cd132154a3a10f8bc6748968b90`; image
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`;
  VideoChat3 revision `37fa901ec5913f84bc31108ebc1e60ad1903634c`;
  Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; and Transformers revision
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`. The VideoChat3 runtime attested
  NVIDIA A40 CUDA `12.8`, BF16 parameters, and an explicitly tied output head.
- **Offer and actual-cost evidence:** each successful declined preview showed
  three physically distinct on-demand RunPod A40 offers in `CA-MTL-1`,
  `EU-SE-1`, and `US-MO-1`, all at `$0.44/hour`. Gate run
  `videochat3-gate-20260801-a1` / dstack ID
  `a14b9052-0a3d-418b-9f1a-b1212b47a154` cost `$0.0292`. Pilot run
  `videochat3-pilot-20260801-a1` / dstack ID
  `b7eb62b3-67ab-4a89-ab27-ef95e3136070` cost `$0.0459`. Incremental cost was
  `$0.0751`; cumulative VideoChat3 cost is `$0.4546` of the authorized
  `$10.00`.
- **Operational prerequisite correction:** the first pilot preview submitted
  nothing and cost `$0.0000` because its compressed exact-panel archive was
  `18.4 MiB`, above the local control plane's `16 MiB` upload limit. The local
  control plane was restarted with the supported `64 MiB` limit; the one
  corrected preview retained the identical config and evidence and then showed
  the three offers above. This correction changed packaging capacity, not the
  scientific inputs or runtime.
- **Private immutable evidence:** the gate evidence class has downloaded
  checksum-manifest SHA-256
  `273bc721ad0d4f5c3f8fa54594f7bb8806bc066af6d013d2baa9673080e95206`.
  The pilot evidence class has downloaded checksum-manifest SHA-256
  `ef26e5671d7fcf558edb56ea4b020e459e57280508c4eb94be66d36ccb2a24a1`;
  its frozen predictions SHA-256 is
  `7f102bb31a89598a83b7bb567f554edeec78ffb1151489dace2eb344bdff77d9`.
  The private grading evidence class was independently listed, downloaded, and
  verified; its checksum-manifest SHA-256 is
  `fbb7190485831e7ba870923390b11e3c3de38396e6f4cf66c0710b7cf371c292`.
  Its context-isolated scored comparison SHA-256 is
  `42ce8737cb0c440455f70cd27f21b9083f3a99f1a6d2b93e822f65ddcef61529`.
- **Confirmed defect, audit gap, and uncertainty:** no confirmed defect or
  audit gap affects the completed gate, pilot, or grading bundles. The small,
  preselected failure panel is an **uncertainty**: it supports this
  condition-bound decision but does not estimate general competition accuracy
  or prevalence.
- **What this rules out now:** promoting this exact VideoChat3 revision,
  tied-head adapter, prompt, frozen twelve-frame evidence, and CUDA BF16 runtime
  as a temporal/action specialist over the pinned Qwen control under the
  declared six-row rule.
- **What this does not rule out:** a materially different VideoChat3 revision,
  another evidence-selection policy, a larger independently selected panel, or
  another model family under a fresh predeclared comparison.
- **Reconsider when:** a materially different revision or evidence policy has
  its own frozen panel, matched baseline, label isolation, and predeclared
  decision rule. The consumed gate and pilot identities are permanently
  retired as sanitized non-runnable provenance records.

## 2026-08-01 exact-timestamp boundary correction

- **Corrected interpretation:** this section supersedes only the evidence-validity
  and decision-disposition claims in the completed-result section above. The
  historical outputs, costs, completion denominators, grading, and immutable
  run provenance remain preserved.
- **Confirmed defect:** the treatment adapter passed a subclass of Transformers
  `VideoMetadata` with exact timestamps implemented only as a property. The
  pinned VideoChat3 processor accepted that object at strict validation, then
  reconstructed its model-native metadata from `metadata.__dict__`. That
  discarded the property and changed the timestamp sequence to `0.0` through
  `11.0`; the frozen prediction record inspected the pre-conversion object and
  therefore did not detect the change.
- **Evidence validity:** **Invalid** for the intended matched promotion claim.
  The exact cached pixels, questions, decoding settings, and Qwen control are
  intact, but VideoChat3 did not receive the held-fixed timestamp metadata.
- **Decision disposition:** the earlier **Not promoted under the tested
  condition** decision is unusable and must not be treated as a valid negative.
  A fresh isolated run is required before making a VideoChat3-versus-Qwen
  quality decision.
- **Approved correction:** construct one metadata object that is simultaneously
  an instance of Transformers `VideoMetadata` and the exact pinned processor's
  `VideoChat3VideoMetadata`. Strict input validation then accepts it and the
  processor retains, rather than reconstructs, its exact timestamp property.
  Do not encode timestamps as fabricated FPS/frame-index values and do not
  modify the pinned remote processor source.
- **Test-first gate:** a local regression test must fail on the current adapter,
  then prove the corrected object passes both type boundaries and retains the
  hand-derived irregular timestamps after the processor conversion branch. A
  digest-pinned dstack synthetic canary must next inspect the actual processor
  timestamp flow without private media or reference answers. No quality run is
  allowed unless this input-flow gate passes.
- **What this rules out now:** using the completed `20260801-a1` gate and pilot
  as evidence for or against VideoChat3 quality under the declared exact-input
  condition.
- **What this does not rule out:** the corrected adapter or VideoChat3 model on
  the same frozen panel in a fresh isolated run.
- **Supersedes / superseded by:** this corrected interpretation supersedes the
  validity and disposition wording immediately above; no replacement quality
  result exists yet. The primary theme and canonical shared-report update remain
  integration-owned and are intentionally not edited on this branch.

## 2026-08-01 timestamp-fix input-flow verification

- **Implemented correction:** `ExactTimestampMetadata` now inherits from both
  the pinned Transformers `VideoMetadata` type used by strict input validation
  and the pinned processor's `VideoChat3VideoMetadata` type used by its
  conversion branch. The exact timestamp property is therefore retained rather
  than reconstructed from `fps=1.0` and indices `0..11`.
- **Local test evidence:** the new boundary tests were observed RED before the
  implementation, then GREEN. They verify both `isinstance` contracts,
  post-conversion object identity, exact irregular timestamps, merged prompt
  timestamps, CUDA-only canary orchestration without model loading, and frozen
  attestation output. The model-specific suite passed `89/89` before the cloud
  preview gate.
- **Cloud gate identity:** `videochat3-input-flow-canary-20260801-a1` used the
  exact digest-pinned CUDA image and model/Transformers revisions, synthetic
  12-frame input only, no private media, no model weights, a `$1.00/hour` cap,
  `30m` duration, retry off, and scalar `A40:1:24GB..`.
- **Persistent infrastructure error:** the initial declined preview returned no
  offers. A live scalar query simultaneously showed three physically distinct
  RunPod A40 offers at `$0.44/hour`. The single corrective preview reduced only
  the unnecessary processor-only CPU, memory, and disk reservations; it again
  returned no offers. The persistent-error rule therefore stopped the canary.
- **Execution and cost:** no provisioning or run submission occurred, completion
  is `0/1`, incremental cost is `$0.0000`, and cumulative VideoChat3 cost remains
  `$0.4546` of the authorized `$10.00`.
- **Evidence validity / decision disposition:** **Incomplete** / **Diagnostic
  signal**. Local tests support the adapter change, but actual pinned CUDA
  processor behavior is not verified and no replacement quality decision is
  available.
- **Audit gap:** no immutable remote failure bundle exists because the task was
  never admitted or submitted. CUDA processor metadata identity, final prompt
  tokens, and tensor placement cannot be reconstructed from the declined
  previews.
- **What this rules out now:** completing this exact input-flow canary through
  the local dstack admission path at the observed time.
- **What this does not rule out:** the local correction, the pinned model, or a
  future exact CUDA input-flow result after the admission mismatch is diagnosed.
- **Retirement:** the unsubmitted runnable configuration is removed. Sanitized,
  fail-closed provenance is retained at
  `dstack/videochat3-input-flow-canary-20260801-a1.retired.yml`; no further
  submission is authorized under this identity.

## 2026-08-01 timestamp-fix dev-environment persistent stop

- **Reason for reopening:** the user authorized a fresh interactive dstack path
  to validate the corrected input flow before comparing the pinned
  VideoChat3-4B treatment with the pinned Qwen3.5-4B baseline on the already
  frozen six-row panel. The synthetic canary remained a hard prerequisite, and
  no private media could enter before it passed.
- **Pre-run freeze:** the gate manifest remained
  `1f476fb3a0f7f5cac3bd3a790348bc6a3eed91c16311e79909f48f8c9760a772`;
  the six-row manifest remained
  `8b5cf944553ac000795b837090799f07041c8b86cc3ed9a997dc52a9ed4941c4`.
  The panel retained the four ordered target roles and two ordered Visual
  controls bound by the declared manifest SHA-256.
  Held-fixed model revisions, image, seed, greedy decoding, 32-token cap, and
  exact twelve-frame evidence were unchanged. Neither dev config transferred
  a manifest, private frame, label, or reference answer.
- **Offer evidence:** both mandatory declined previews showed three physically
  distinct on-demand RunPod A40 48 GB offers in `CA-MTL-1`, `EU-SE-1`, and
  `US-MO-1`, each at `$0.44/hour`. Each config used scalar
  `A40:1:24GB..`, retry off, `$1.00/hour` maximum price, and `2h` maximum
  duration. The a2 preview output SHA-256 was
  `0a5f25a4a156f11f048ef99407ea8b4ebfc61a99c4484883bb87f93d80c82676`.
- **First technical attempt:** `videochat3-timestamp-dev-20260801-a1` / dstack
  ID `65785a53-a647-42a7-9ad0-54a039b12c21` failed closed before dependency
  installation. Its submitted requirements environment pin omitted one hex
  character, so it did not equal the complete pinned file SHA-256. Exit status
  was `2`, termination was `container_exited_with_error`, actual cost was
  `$0.0369`, and no remote artifact prefix was created.
- **Single corrective attempt:** a RED regression test reproduced the malformed
  63-character pin, and the corrected a2 config verified the full 64-character
  requirements, source, plan, and runner hashes inside the exact image.
  `videochat3-timestamp-dev-20260801-a2` / dstack ID
  `da69c2a2-f102-4f98-b98f-dd0fa1b12e4b` then attested NVIDIA A40, CUDA
  `12.8`, BF16 support, Torch `2.10.0+cu128`, Transformers revision
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`, and the other pinned
  dependencies. Initialization completed successfully, but its startup entry
  replaced dstack's generated long-lived shell. The environment therefore
  terminated `done_by_runner` with exit status `0` before an SSH replica or
  synthetic processor canary was available. Actual a2 cost was `$0.0237`.
- **Confirmed defects:** a1 had a configuration-pin transcription defect; a2
  had a dev-lifecycle configuration defect. dstack 0.20.29 appends its idle
  process after user startup entries, so replacing the generated shell prevents
  that idle process from running. Both defects are preserved in sanitized,
  non-runnable records; neither is a model-quality observation.
- **Gate and completion denominator:** dev initialization was a1 `0/1`, a2
  `1/1`; synthetic exact-timestamp canary `0/1`; real-video gate `0/3`;
  VideoChat3 matched pilot `0/6`; Qwen matched pilot `0/6`. No predictions
  exist, so freezing, blind grading, target improvements, and control
  regressions are not applicable.
- **Cost:** incremental dev cost was `$0.0606`; cumulative actual VideoChat3
  spend is `$0.5152` of the user-authorized `$10.00`. Fleet events confirm both
  VideoChat3 instances terminated with `JOB_FINISHED`, so no paid VideoChat3
  capacity remained live. A later instance under the reused fleet slot belonged
  to an unrelated TimeLens run and was not changed.
- **Evidence validity:** **Incomplete**. The a2 runtime preparation evidence is
  valid for its narrow CUDA/dependency claim, but the input-flow canary and
  scientific comparison did not execute.
- **Decision disposition:** **Persistent technical stop**. The original
  one-correction rule is exhausted. No third dev identity, pilot submission, or
  quality-driven rerun is authorized.
- **Audit gap:** neither attempt produced a remote immutable failure bundle.
  A1 stopped before runtime verification; a2 completed before SSH attachment
  or synchronization. Exact processor metadata identity, prompt timestamps and
  tokens, CUDA tensor placement, corrected VideoChat3 predictions, and the
  matched Qwen comparison cannot be reconstructed from these attempts.
- **What this rules out now:** treating the earlier timestamp-invalid pilot or
  these incomplete dev attempts as a valid VideoChat3-versus-Qwen quality
  comparison, and resubmitting either consumed dev identity.
- **What this does not rule out:** the corrected dual-compatible metadata
  adapter, the pinned VideoChat3 model, the pinned Qwen baseline, or a future
  comparison under separately authorized infrastructure after the long-lived
  dev lifecycle is validated before immutable submission.
- **Retirement:** runnable a1/a2 dev configs and the stage runner are absent.
  Sanitized, secret-free records are retained at
  `dstack/videochat3-timestamp-dev-20260801-a1.retired.yml` and
  `dstack/videochat3-timestamp-dev-20260801-a2.retired.yml`. The canonical
  shared theme update remains integration-owned and is not edited here.

## 2026-08-01 explicitly reauthorized a3 dev validation plan

- **Authorization and identity:** the user explicitly reopened the dstack dev
  validation after the a1/a2 persistent stop. The only newly authorized dstack
  identity is `videochat3-timestamp-dev-20260801-a3`. Its immutable evidence
  stages are the synthetic canary, compatibility gate, and matched pilot. No a4
  retry is authorized.
- **Lifecycle prerequisite:** a local behavioral regression must prove the dev
  startup entry returns control to dstack so its generated idle process and SSH
  replica remain available. The prior shell-replacing startup form is forbidden.
- **Synthetic prerequisite:** the exact digest-pinned A40 CUDA/BF16 environment
  must run the actual pinned VideoChat3 processor on a synthetic twelve-frame
  tensor with irregular timestamps. It must preserve metadata object identity,
  all exact frame timestamps, merged prompt timestamps, prompt labels, and CUDA
  tensor placement. No private manifest, frame, model weights, label, or
  reference answer may enter before this stage passes and its remote artifact is
  listed, downloaded, and checksum-verified.
- **Frozen real-video gate:** only after the synthetic prerequisite passes,
  transfer gate manifest SHA-256
  `1f476fb3a0f7f5cac3bd3a790348bc6a3eed91c16311e79909f48f8c9760a772`
  and exactly the three ordered compatibility rows bound by that hash. Require
  `3/3` completed, nonempty, single-line short answers.
- **Frozen matched panel:** only after the real-video gate passes and is durably
  verified, transfer pilot manifest SHA-256
  `8b5cf944553ac000795b837090799f07041c8b86cc3ed9a997dc52a9ed4941c4`.
  The four ordered targets and two ordered Visual controls remain bound by the
  declared manifest SHA-256.
- **Held fixed:** VideoChat3 revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`, Qwen revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Transformers revision
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`, exact image digest
  `b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`,
  exact cached twelve-frame pixels/indices/timestamps, the controlled prompt,
  seed `20260724`, greedy decoding, 32-token cap, CUDA BF16, and no CPU/ROCm
  fallback.
- **Integrity and decision rule:** verify the complete manifest and every
  evidence hash before model loading. Freeze predictions before accessing
  references, then use two independent blind C/P/W graders and a third only on
  disagreements. Pass only with VideoChat3 `6/6`, Qwen `6/6`, at least `2/4`
  target improvements, and `0/2` control regressions. Never rerun a technically
  valid result for quality.
- **Cost boundary:** prior cumulative VideoChat3 spend is `$0.5152`. The a3 dev
  uses scalar `A40:1:24GB..`, RunPod/Vast.ai on demand, retry off,
  `$1.00/hour` maximum price, `2h` maximum duration, and therefore a cumulative
  hard ceiling of `$2.5152`, below the user-authorized `$10.00`.

## 2026-08-01 a3 dev canary correction before private input

- The first synthetic a3 canary failed closed before private input because
  Transformers tensorization copied the exact-timestamp `Mapping` metadata.
  Live inspection proved that the copy retained the adapter's exact metadata
  type, all twelve irregular timestamps, and the merged prompt timestamps.
- The identity-only assertion was therefore a false failure and is superseded
  by the scientifically relevant invariant: the post-processor metadata must
  remain the exact adapter type and must preserve every timestamp and merged
  prompt time exactly. A stock or altered metadata object still fails closed.
- This correction is test-driven and will be revalidated in the same authorized
  a3 dev environment. It does not authorize another provisioned run, another
  model revision, different evidence, or any quality rerun.
- The consumed failed canary evidence remains immutable. The single corrected
  synthetic recheck was predeclared within the same live a3 dev instance and
  still carries no private input or model weights.

## 2026-08-01 valid a3 matched result

- **Infrastructure and lifecycle:** `videochat3-timestamp-dev-20260801-a3`
  (dstack ID `9bd81210-af7e-4b1a-a9bd-5fa20dae0302`) ran on an on-demand RunPod
  NVIDIA A40 48 GB in `CA-MTL-1` at `$0.44/hour`. Initialization completed and
  the SSH/idle process stayed live, validating the corrected dev lifecycle.
  The environment was stopped after durable verification at an actual cost of
  `$0.1420`; cumulative VideoChat3 spend is `$0.6572` of the authorized
  `$10.00`. Ephemeral private input was removed before stop.
- **Input-flow result:** the first synthetic artifact failed an identity-only
  assertion. Live inspection showed that Transformers tensorization copied the
  exact adapter `Mapping` while retaining its type and every exact timestamp.
  A RED/GREEN regression replaced object identity with exact type, timestamp,
  and merged-prompt-time invariants. The predeclared corrected canary passed on
  CUDA BF16 without model weights or private media; all twelve irregular times
  and three merged prompt labels were preserved.
- **Gate and completion:** the complete manifest and evidence hashes were
  verified before model loading. The real-video gate passed `3/3`. The matched
  panel completed VideoChat3 `6/6` and Qwen `6/6` on identical frozen evidence,
  prompt, seed `20260724`, greedy decoding, and 32-token cap. Predictions were
  frozen, uploaded, listed, downloaded, and checksum-verified before references
  or grades were loaded.
- **Blind C/P/W result:** two independent masked graders agreed on `11/12`
  candidates; one third grader adjudicated the sole disagreement. Aggregate
  grades were VideoChat3 `0C/2P/4W` and Qwen `2C/2P/2W`; row-level judgments
  remain only in the private checksum-bound grading artifact.
- **Decision rule:** VideoChat3 improved `1/4` target rows, tied `2/4`,
  regressed `1/4`, and regressed both controls (`2/2`). It therefore fails the
  required at-least-`2/4` target improvements and zero-control-regression rule.
- **Evidence validity:** **Verified** for this condition-bound six-row comparison.
  **Decision disposition:** **Not promoted under the tested condition**. This
  rules out promoting the pinned exact-timestamp VideoChat3 condition over the
  pinned Qwen baseline on this frozen panel. It does not establish general
  VideoChat3 inferiority on other samples, prompts, evidence policies, frame
  counts, or revisions. The purposive panel is the principal limitation; the
  one adjudicated grade is the recorded uncertainty, and there is no audit gap.
- **Durable private artifacts:** the corrected canary, gate, and matched pilot
  remain in private checksum-bound evidence classes. Their checksum-manifest hashes are
  `86e4e63fc3757d90f691a9e47616a7e3f364552a0261f7872d935e2ee9616a55`,
  `65d30918fc2875b32c5e9ae3e594ba530933126a38f75804cd97ee9520c47082`,
  and `38691d175a73b0ae2f7821c8b491f39946372c7068ab73abb77e25ae45aaf2f0`.
- **Retirement:** runnable a3 config and stage runner are removed. Sanitized,
  secret-free, fail-closed provenance is retained at
  `dstack/videochat3-timestamp-dev-20260801-a3.retired.yml`. No a4 identity or
  quality rerun is authorized. The shared project-status QMD remains
  integration-owned and was not edited here.

## 2026-08-01 advisor-approved terminal 24-frame follow-up

### Decision and scope

- **User authorization:** after reviewing the fixed-12-frame negative, the user
  asked the project advisors which materially different condition makes sense
  for this competition and authorized that test if one does. This section is
  frozen before implementation or inference.
- **Independent advice:** Codex `gpt-5.6-sol`, Claude `fable`, and Gemini
  `gemini-3.1-pro-high` independently selected one matched 24-frame offline
  specialist screen, followed by permanent archival on any valid negative.
  They rejected proactive streaming / Adaptive Frame Resolution because the
  competition supplies the complete clip before answering, rejected raw-video
  sampling because it would confound the evidence selector with model quality,
  and found no newer checkpoint: Hub `main` remains
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`.
- **Primary theme:** Models and Adaptation. This is a second density condition
  for the existing `#videochat3-temporal-specialist` claim, not a new theme.
- **Hypothesis:** VideoChat3's variable-length I3D-ViT can exploit 24 uniformly
  sampled frames—six complete four-frame temporal chunks instead of three—to
  recover Temporal/Action failures without regressing Visual controls.
- **Falsifier:** any valid completed comparison that misses either the target
  improvement threshold or the control non-regression threshold ends the
  VideoChat3 track. Do not escalate to 32/48 frames, raw-video sampling, AFR,
  streaming, or another prompt after a valid negative.

### Frozen condition

- **Treatment:** `MCG-NJU/VideoChat3-4B` revision
  `37fa901ec5913f84bc31108ebc1e60ad1903634c`.
- **Control:** `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Image:**
  `docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca`,
  NVIDIA CUDA BF16 only, no CPU or ROCm fallback.
- **Evidence:** exactly 24 uniform bin-midpoint frames already present in the
  canonical cache for each selected clip. Both models receive the identical
  ordered pixels and timestamps. Processor-side resampling remains disabled.
  No raw source video is transferred.
- **Panel order and roles:** four ordered targets followed by two ordered Visual
  controls, as bound by the declared six-row manifest SHA-256. The two-row
  operational gate is the first target plus the first control. Gate predictions are not
  graded or interpreted before the complete six-row prediction freeze.
- **Held fixed:** controlled short-answer prompt, seed `20260724`, greedy
  decoding, 32-token cap, model revisions, runtime image, CUDA BF16, row order,
  label isolation, output normalization, and blind C/P/W grading protocol.
- **Input inventory:** before implementation, all six source media identities
  resolved uniquely through the established `.mkv/.mp4/.webm/.mov` resolver.
  Each had one unique 24-frame cache. All 144 frame files matched their recorded
  SHA-256 values; their aggregate size is `120148701` bytes. The first ad-hoc
  inventory count is invalid and discarded because extensionless media
  identities produced empty hashes; only this fail-closed resolved inventory is
  evidence.
- **Integrity:** write a label-free 24-frame request manifest, pin its complete
  SHA-256, verify its row order, cache records, timestamps, and every frame hash
  before model loading, then freeze predictions before loading any reference or
  grade. Upload only the six selected cache directories and label-free manifest.
- **Frozen private manifest:** the exclusively written six-row label-free
  manifest has SHA-256
  `d13a1135184d92988af28df0fcda47a6d0debef268a8861cfd4ced3898a9518b`.
  It contains 24 ordered paths and original bin-midpoint timestamps for every
  declared row, binds the six verified cache keys, and contains no reference or
  grade fields.

### Immutable identity and operational bounds

- Dev environment: `videochat3-24frame-dev-20260801-a1`.
- Private evidence classes: one input bundle, one synthetic no-private-media
  canary, one two-row matched operational gate, and one six-row matched
  comparison. Relative object names remain private.
- A suffix `-a2` is reserved only for one narrowly corrected technical attempt
  after a diagnosed implementation/configuration failure. It is not authorized
  for a quality retry. No other run identity is authorized.
- Use a scalar `A40:1:24GB..` selector, on-demand capacity, multiple compatible
  physical offers when practical, digest-pinned image, detached submission,
  `$0.60/hour` maximum price, `90m` maximum duration, and retry disabled. Preview
  with `echo "n" | dstack apply -f ...` and retain the full output.
- Incremental spend must remain below `$1.00`; cumulative VideoChat3 spend must
  remain below `$1.6572`, which is stricter than the user-authorized `$10.00`
  total cap. Stop and terminate the environment immediately after durable
  verification or any terminal gate.
- `HF_TOKEN` remains a dstack secret. Never print it, persist it, pass it as a
  CLI argument, or use diagnostic output on the secret-bearing environment.

### Gates and terminal decision rule

1. **Local freeze gate:** generate the label-free 24-frame manifest from the
   checksum-verified canonical caches; verify that it contains only the six
   declared rows, 24 ordered frame paths/timestamps per row, no label fields,
   and a complete manifest SHA-256.
2. **Exact-image synthetic gate:** run one fabricated 24-frame tensor through
   the pinned processor and model on CUDA BF16 without private media. Require
   `1/1`, exact timestamp/order/metadata retention, a nonempty short answer, and
   recorded CUDA memory use.
3. **Two-row matched operational gate:** after the canary passes, run the first
   target and first control through both pinned models on identical private
   evidence. Require VideoChat3 `2/2` and Qwen `2/2` valid short answers. Do not
   expose references or grade this partial gate.
4. **Complete matched screen:** after the operational gate passes, run all six
   rows through both models. Require VideoChat3 `6/6` and Qwen `6/6`, freeze and
   checksum predictions, then obtain two independent blind C/P/W grades and a
   third adjudication only on disagreements.
5. **Promotion:** advance only if VideoChat3 improves at least `2/4`
   Temporal/Action targets versus Qwen-24 and neither Visual control regresses.
   Record Qwen-24 versus archived Qwen-12 grades as a non-gating sanity check so
   a weaker 24-frame comparator cannot be mistaken for VideoChat3 improvement.
6. **Terminal stop:** a valid negative is **Verified** / **Not promoted under
   the tested condition** and permanently archives this VideoChat3 track. A
   technical failure permits at most the reserved single `-a2` correction after
   root-cause diagnosis; a repeated technical gate failure stops as persistent.

### Implementation plan

#### Task 1 — Parameterize the existing adapter for a frozen frame count

**Files:** modify `src/trec_video_2026/vqa_experiments/videochat3_pilot.py` and
`tests/test_videochat3_pilot.py`.

**Interfaces:** add `frame_count: int = 12` to request/evidence validation and
row execution helpers; add CLI `--frame-count` restricted to `12` or `24`;
retain 12-frame defaults and existing manifest pins. Produce dynamic input-form,
frame-name, condition, and provenance fields.

- [x] Write failing tests proving 24 paths/timestamps validate, 23/25 fail,
  exact 24-frame cache hashes are checked, VideoChat3 and Qwen receive 24 frames,
  and existing 12-frame behavior remains unchanged.
- [x] Run the focused tests and confirm each fails because frame-count support
  is absent.
- [x] Implement the smallest parameterization and pass `frame_count` through
  `validate_cached_request`, `_videochat3_rows`, `_qwen_rows`, canary modes, and
  CLI dispatch.
- [x] Run the focused tests green, then the full VideoChat3 test files.

#### Task 2 — Freeze the label-free 24-frame manifest

**Files:** modify the same adapter and tests; private output stays under ignored
`data/vqa/benchmarks/videochat3-24frame-20260801-a1/`.

**Interfaces:** add a local preparation mode that consumes the already-frozen
12-frame label-free pilot request manifest only for row IDs/questions, resolves
each canonical source media identity, derives `EvidenceParameters(frame_count=24)`
cache keys, verifies the cache record and every frame, and exclusively writes a
six-row 24-frame manifest plus its SHA-256. No annotations or references enter
the output.

- [x] Write failing tests for unique media resolution, exact cache-key binding,
  label rejection, selected row order, 24 timestamps/paths, and exclusive write.
- [x] Run RED, implement minimally, then run GREEN.
- [x] Generate the private manifest, verify it independently, and record its
  SHA-256 here before preview or inference.

#### Task 3 — Add a fail-closed dev environment and stage runner

**Files:** create `dstack/videochat3-24frame-dev-20260801-a1.dstack.yml` and
`scripts/run-videochat3-24frame-dev-dstack.sh`; modify
`tests/test_videochat3_dstack_config.py`.

**Interfaces:** the config transfers only code, requirements, runner, and this
plan; the stage runner exposes `bootstrap`, `canary`, `matched-gate`, and
`matched-pilot` phases. Every phase pins image, model revisions, source snapshot,
plan, requirements, input manifest when applicable, immutable HF destination,
and output checksums. It foregrounds Python and preserves workload/sync failures.

- [x] Write failing config/runner tests for immutable names, scalar GPU, hard
  limits, retry-off, no private files, secret-only token handling, complete hash
  pins, stage ordering, and remote checksum finalization.
- [x] Run RED, implement the minimal config/runner, then run GREEN and shell
  syntax checks.
- [x] Run `git diff --check`, the model-specific suites, and the full local test
  suite before preview.

#### Task 4 — Execute and decide

- [x] Upload only the exact six cached 24-frame directories and frozen label-free
  manifest to the immutable private input URI; list, download, and checksum it.
- [x] Preview the exact config, retain full output, require multiple physical
  compatible offers when practical, and submit detached once.
- [x] Run the synthetic canary; on pass run the two-row operational gate; on pass
  run the six-row matched comparison. Stop at the first terminal failure.
- [x] List/download/checksum each durable stage artifact and stop the environment.
- [x] Freeze blind grader packets after prediction freeze, collect two independent
  grades, adjudicate disagreements only, and apply the predeclared rule.
- [x] Replace the runnable config/runner with sanitized fail-closed retired
  provenance, update the canonical Models claim and research ledger, render and
  privacy-check the hub, run the full suite and `git diff --check`, then commit.

## 2026-08-01 terminal 24-frame result

- **Infrastructure:** `videochat3-24frame-dev-20260801-a1` (dstack ID
  `d5cc3983-76e5-4468-a7b2-8c29beea27c9`) ran once on an on-demand RunPod
  NVIDIA A40 48 GB in `EU-SE-1` at `$0.44/hour`. The declined preview showed
  two physically distinct compatible A40 offers in `CA-MTL-1` and `EU-SE-1`.
  The environment was stopped immediately after durable verification at
  `$0.0763`; cumulative VideoChat3 spend is `$0.7335`.
- **Frozen input and runtime:** the private 120,320,000-byte archive SHA-256 is
  `f234f73228116e3ceeefe492341b5cc99ad0125d5bd4b944f53bf4da4cdcdca1`.
  Its six evidence records and 144 frames were verified against the complete
  request-manifest SHA-256
  `d13a1135184d92988af28df0fcda47a6d0debef268a8861cfd4ced3898a9518b`
  before model loading. The exact-image CUDA BF16 canary passed `1/1` and
  peaked at 9,722,740,736 allocated bytes on the A40.
- **Completion:** the matched target/control gate passed VideoChat3 `2/2` and
  Qwen `2/2`. The full screen completed VideoChat3 `6/6` and Qwen `6/6` on
  identical 24-frame evidence. Predictions were frozen and checksum-verified
  before the answer key or grading packet was loaded.
- **Blind quality:** two independent graders agreed on all `12/12` candidate
  judgments; no third adjudication was needed. VideoChat3-24 scored
  `0C/1P/5W`, while Qwen-24 scored `3C/2P/1W`. VideoChat3 improved `1/4`
  targets, regressed `3/4`, and regressed both controls (`2/2`). Qwen-24 versus
  archived Qwen-12 improved `1/6`, tied `5/6`, and regressed `0/6`, so the
  VideoChat3 loss is not an artifact of a weakened 24-frame comparator.
- **Decision:** **Evidence validity: Verified. Decision disposition: Not
  promoted under the tested condition.** The result falsifies the terminal
  density hypothesis. Together with the valid fixed-12-frame negative, it
  rules out this checkpoint as the project's Temporal/Action specialist under
  the tested fixed-frame input flows. Do not run 32/48 frames, raw-video, AFR,
  streaming, or another prompt for this checkpoint. This does not rule out a
  future materially different checkpoint, task-specific fine-tuning, or a
  representative benchmark result.
- **Durable artifacts:** input, canary, gate, pilot, and grading bundles live
  under the private repository experiment bucket. The canary, gate, pilot, and
  grading checksum-manifest hashes are respectively
  `97ea6536240c34a24e5e6a9c04e73ba5b315f47b9858da19bcbc22c81527efac`,
  `3ea1189285a5532ec003b1114623a4d3600f76eea987452ea287420b0f72c1d7`,
  `080247d5ffad25e0c39c32d3d5c4d68caa5283a6349b2c73af5cc55e5faa90bb`,
  and `e2512db5ee286c38fd5b2ecbd3af33c9a02b1f5fbdcbf4d26d4ede6e8dc68005`;
  every bundle was listed, downloaded, and checksum-verified.
- **Security remediation:** while locating the dev SSH alias, a local
  control-plane trace containing injected credentials was accidentally
  surfaced in the private task transcript. Inspection stopped immediately and
  the exact local trace was deleted; it was neither uploaded nor committed.
  The affected Hugging Face OAuth session and RunPod provider key must be
  rotated. This is an audit gap for the deleted trace, but it does not affect
  the independently checksum-verifiable inference or grading evidence.
- **Retirement:** the runnable config and stage runner were removed. Sanitized,
  fail-closed provenance remains in
  `dstack/videochat3-24frame-dev-20260801-a1.retired.yml`; no further
  VideoChat3 run identity is authorized.
