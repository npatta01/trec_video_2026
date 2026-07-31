# Nemotron 3 Nano Omni native audio-video pilot

Date: 2026-07-30

Evidence validity: **Verified**

Decision disposition: **Routed candidate**

## Separately authorized b1 retry design

The user separately authorized a fresh retry after the a1 track stopped. This
does not reopen the retired `nemotron3-omni-pilot-20260730-a2` identity. The new
immutable run identities are:

- codec sentinel: `nemotron3-omni-codec-sentinel-20260730-b1`
- paired pilot: `nemotron3-omni-pilot-20260730-b1`

The root-cause pattern is exact: all three AV1/Opus MKVs failed both a1
conditions, while all three VP9/Opus MKVs passed both conditions. Because Opus
is shared across the successful and failed groups, the b1 hypothesis is that
vLLM's tested OpenCV path cannot decode the panel's AV1 video streams. The
falsifier is any failed native audio-video request after normalization, or any
failure to complete the later 12-request paired contract.

### Treatment, controls, and held-fixed variables

All six original MKVs will be downloaded and hash-verified before model
loading, then normalized inside the exact pinned runtime image to H.264/AAC
MP4. Normalization will strip metadata, use one CPU encoding thread, fix the
pixel and audio formats, record every derived hash, and verify that video and
audio both decode. Normalizing all six avoids introducing a treatment-specific
codec path. Each row's audio-off and audio-on requests will use the same
normalized bytes.

The model ID and revision, vLLM image and version, L40S CUDA hardware class,
non-thinking greedy short-answer request, six-row panel, row order, questions,
and promotion rule remain unchanged from a1. No chain-of-thought will be
requested or stored.

### Sequential gates

1. Upload only the existing six label-free source media, contract, and marker
   to the new pilot b1 input prefix.
2. Preview the exact-image codec-sentinel config and retain the complete offer
   output. Submit it detached only when multiple compatible offers are
   available where practical.
3. Normalize all six files, verify their derived hashes and decodability, then
   exercise the real foreground wrapper with native audio-video requests on
   the three formerly failing AV1 rows. Require 3/3 valid short answers.
4. Stop without a pilot if the codec sentinel is not 3/3. No b2 retry is
   authorized.
5. After sentinel success and fleet release, preview and submit the pilot b1
   config. Require all 12 adjacent audio-off/audio-on outputs.
6. Freeze outputs before loading any references. Blind grading must verify
   `manifest.json` and `predictions.jsonl` against frozen `SHA256SUMS`, verify
   the manifest-recorded prediction hash and exact 12-record structure, and
   only then invoke the reference loader.
7. Obtain two independent C/P/W grades and a distinct third adjudicator only
   for disagreements. Promotion still requires at least two of four target
   improvements and no control regression.

### Evidence, privacy, and operational boundary

The sentinel and pilot will each use their own private HF experiment prefix.
In addition to workload outputs, a sanitized checksummed audit bundle will
durably preserve the exact preview output and post-run dstack
scheduler/region/duration/cost metadata. The audit bundle will be listed,
downloaded, and checksum-verified. Private media, raw answers, references, and
raw logs remain outside Git and outside sanitized reporting.

The sentinel has a 30-minute and `$2/hour` hard cap; the pilot has a 45-minute
and `$2/hour` hard cap. Their incremental worst case is `$2.50`, bringing the
existing `$0.2435` spend to at most `$2.7435`, within the original `$8` base
allocation. No central reserve and no automatic b2 correction are authorized.

### Executable b1 implementation plan

1. Add RED tests for a normalized-media contract that retains the six frozen
   row identities and questions, records source and derived SHA-256 values,
   rewrites media paths to MP4, and creates a three-row, audio-on-only codec
   sentinel for `IWSPAfCKtV8`, `OzNdzgQi1CA`, and `EMI_anpHII4`.
2. Implement only the contract builder and validation needed to make those
   tests pass. The real probe must accept the sentinel as exactly 3/3 and the
   normalized pilot as exactly 12/12 while retaining the no-reasoning,
   greedy, short-answer request contract.
3. Add RED tests for the two b1 dstack configs and shell boundaries. Both
   configs must use the exact pinned image, scalar `L40S:1:48GB..` selector,
   shared one-node fleet, on-demand CUDA, detached submission compatibility,
   foreground wrapper, and hard price/duration caps. The scripts must reject
   the retired pilot a2 and every unrecognized identity.
4. Implement deterministic in-container normalization of all six verified
   source MKVs to metadata-stripped H.264/AAC MP4 with one encoding thread.
   Before model loading, verify source hashes, derived hashes, exact stream
   codecs, and successful video and audio decode. Run the three-row sentinel
   through the same foreground interpreter and native audio-video request seam
   used by the full pilot.
5. Add RED tests for sanitized immutable audit capture. Collection must require
   the retained exact preview file for each b1 run, sanitize `dstack run get`
   output through the existing metadata adapter, checksum the preview and
   metadata, upload the audit bundle to that run's private HF prefix, then
   recursively list, download, and checksum-verify it. Never use `diagnose`.
6. Run the model-specific suite, shell syntax checks, config parsing, privacy
   scans, and `git diff --check` before any cloud action.
7. Upload only the original six label-free media plus contract and completion
   marker to the pilot b1 prefix. Preview the sentinel with
   `echo "n" | dstack apply -f ...`, retain the complete output, and require
   multiple physically distinct compatible offers where practical.
8. Submit the sentinel detached and monitor the foreground workload without
   secret-bearing diagnostics. Collect and verify both workload and audit
   artifacts. Stop the track if the sentinel is not 3/3; no b2 is authorized.
9. Only after a 3/3 sentinel and fleet release, repeat the retained preview,
   detached submission, monitored foreground execution, and artifact
   verification for the 12-output b1 pilot.
10. Only after cryptographic freeze verification, obtain two independent blind
    C/P/W grades and a third adjudication only for disagreements. Apply the
    predeclared promotion rule, update this sanitized record, run the full
    verification suite and an independent review, and commit the complete
    model-specific work.

## Approved expanded-12 follow-up design

The user requested a test on at least 10 videos and approved using the complete
12-row failure panel. This is a new, separately authorized experiment; it does
not mutate or repeat either b1 run. Its immutable identities are:

- public exact-image canary:
  `nemotron3-omni-expanded12-canary-20260730-c1`
- paired expanded pilot:
  `nemotron3-omni-expanded12-20260730-c1`

No c2 submission or external retry loop is authorized. The retired historical
pilot a2 identity remains unavailable.

### Question, panel, and decision rules

The question is whether the condition-bound b1 audio benefit replicates on the
original four non-speech-sound/music targets and extends across a broader
12-video panel. The label-free panel contains 12 videos and 24 adjacent paired
requests:

- 2 dialogue and speech targets
- 4 non-speech sound and music targets
- 4 temporal and action targets
- 2 visual controls

The treatment is native video with audio enabled; the matched control is the
same normalized video bytes with audio disabled. Model, revision, runtime
image, CUDA/L40S hardware envelope, questions, row order, prompt, greedy
short-answer decoding, and normalization remain held fixed.

The result has two predeclared interpretation gates:

1. **Sound replication:** 24/24 requests complete, audio-on improves at least
   2 of the 4 non-speech-sound/music targets, and neither visual control
   regresses.
2. **Broader expansion:** sound replication passes, audio-on improves at least
   4 of all 10 non-control rows, no more than 1 non-control row regresses, and
   neither visual control regresses.

If only sound replication passes, the decision remains **Routed candidate**
for the sound-specific condition. If both gates pass, the condition-bound
decision may expand to **Routed candidate** across the tested target mix. A
positive result cannot establish **Current default** because the panel is
small and deliberately failure-selected. Failure of either gate is a valid
negative result for that gate and must not trigger a technically unnecessary
rerun.

### Runtime and sequential integrity gates

The runtime remains:

- model:
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- model revision:
  `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
- image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`

The expanded run will:

1. Use tests first to add an exact 12-row label-free contract, a 24-request
   normalized contract, expanded grading validation, immutable identities, and
   dstack configs.
2. Upload only the 12 source media, label-free contract, and completed marker
   to the new private c1 input prefix, then recursively download and hash-check
   them.
3. Preview the public canary with
   `echo "n" | dstack apply -f ...`, retain the full output, and require
   multiple physically distinct compatible offers where practical.
4. Run the pinned image's real foreground interpreter, vLLM server, and native
   audio-video request seam on three public synthetic H.264/AAC files. Require
   3/3 valid non-thinking short answers before the private pilot.
5. After canary verification and fleet release, preview and submit the
   expanded pilot once. Inside the exact image, source-hash-verify and
   normalize all 12 files to metadata-stripped H.264/AAC MP4, record derived
   hashes, validate codecs and duration, and decode-check both streams before
   model loading.
6. Freeze all 24 predictions and verify `manifest.json` and
   `predictions.jsonl` against `SHA256SUMS`, including the manifest-recorded
   prediction hash and exact row/condition order, before references load.
7. Obtain two independently shuffled blind C/P/W grades. Use a distinct third
   grader only for disagreements, then apply both predeclared gates.
8. Preserve outputs, exact preview text, sanitized terminal dstack
   scheduler/region/duration/cost metadata, and private grading artifacts under
   the immutable run prefixes. Recursively list, download, and checksum-verify
   each bundle.

No request may ask for, save, or expose chain-of-thought. Private media,
answers, references, item-level grades, raw competition rows, and raw logs
remain outside Git and sanitized reporting. `HF_TOKEN` remains in dstack
secrets only.

### Budget and ownership boundary

The public canary is capped at 30 minutes and `$2/hour`; the expanded pilot is
capped at 45 minutes and `$2/hour`. Incremental worst-case spend is `$2.50`.
Together with the existing `$0.5781`, cumulative worst case is `$3.0781`,
inside the original `$8` base allocation. No central reserve is authorized.

This branch may change only Nemotron-specific adapters, configs, scripts,
tests, and this existing evidence record. The shared project-status QMD and
generated Markdown/HTML remain the parent integration task's responsibility.

### Executable expanded-12 implementation plan

All implementation tasks use strict RED-GREEN-REFACTOR sequencing. Existing
a1 and b1 behavior is a compatibility constraint: the expanded path must be
explicitly selected and must not re-enable any retired identity or retry.

### Task 1: Expand the label-free inference contract

Modify
`src/trec_video_2026/vqa_experiments/nemotron3_omni_pilot.py` and
`tests/test_nemotron3_omni_pilot.py`.

1. First add failing tests for the exact 12-row identity, category, role, and
   panel order; a 12-row label-free source contract; and a normalized contract
   with exactly 24 adjacent audio-off/audio-on records.
2. Add an explicit `expanded-pilot` contract mode and
   `build-expanded-contract` CLI without changing the six-row b1 defaults.
3. Generalize input validation so this mode requires exactly 12 media files,
   one contract, and a completed marker. Require exact source and normalized
   hashes, expected codecs, duration agreement, and decode validation before
   inference.
4. Extend the probe only through the explicit expanded mode. Preserve greedy,
   non-thinking, short-answer decoding and reject any output field intended to
   carry reasoning.
5. Run `pytest -q tests/test_nemotron3_omni_pilot.py` after observing the
   focused RED failure and again after the minimal implementation.

### Task 2: Expand cryptographic freeze verification and blind scoring

Modify
`src/trec_video_2026/vqa_experiments/nemotron3_omni_grading.py` and
`tests/test_nemotron3_omni_grading.py`.

1. First add failing tests for exact 12-row/24-record order, wrong denominator,
   reordered pairs, modified predictions, modified manifest, and references
   being inaccessible until all hashes and internal manifest bindings verify.
2. Add an explicit expanded freeze verifier and blind-packet builder while
   retaining the six-row API behavior.
3. Extend scoring to report: four sound-target improvements, ten non-control
   improvements and regressions, two visual-control regressions,
   `sound_replication_passed`, and `broader_expansion_passed`.
4. Keep independently shuffled blind packets label-free. Require two complete
   C/P/W grade sets and accept third-grader adjudication only where those sets
   disagree.
5. Run `pytest -q tests/test_nemotron3_omni_grading.py` after the focused RED
   failure and after implementation.

### Task 3: Add immutable c1 runner, uploader, collector, and dstack configs

Modify `scripts/run-nemotron3-omni-dstack.sh`,
`scripts/upload-nemotron3-omni-pilot-inputs.sh`,
`scripts/collect-nemotron3-omni-dstack.sh`, and
`tests/test_nemotron3_omni_dstack_config.py`. Add
`dstack/nemotron3-omni-expanded12-canary-c1.dstack.yml` and
`dstack/nemotron3-omni-expanded12-c1.dstack.yml`.

1. First add failing tests for the two exact c1 identities, absence of any c2
   or historical pilot-a2 allowlist path, 13 pilot input objects, exact pinned
   image/model revision, scalar `L40S:1:48GB..`, shared marketplace fleet,
   on-demand RunPod/Vast backends, `$2/hour` price caps, and 30/45-minute
   duration caps.
2. Route the public canary through the existing real foreground wrapper and
   three synthetic H.264/AAC native audio-video requests. Route the expanded
   pilot through the new 12-row contract and exact-image normalization path.
3. Require preview capture for both c1 identities, detached dstack submission,
   foreground workload execution, completed input marker, durable private
   workload/audit prefixes, and terminal sanitized scheduler/cost metadata.
4. Permit collection only for the two immutable c1 identities and existing
   completed historical identities. Never use secret-bearing diagnosis.
5. Run the focused config tests plus `bash -n` and YAML parsing.

### Task 4: Complete the local pre-cloud gate

Run all Nemotron-specific tests, the full test suite, Ruff check and format
check, shell syntax checks, YAML parsing, credential/private-data scans,
`git diff --check`, and the shared-data symlink check. Obtain an independent
read-only review of the diff. Resolve only high-confidence findings using new
RED tests, then commit the complete local implementation before cloud work.

### Task 5: Run and verify the public exact-image canary

Confirm live scalar L40S offers and that the shared fleet is free. Run exactly
`echo "n" | dstack apply -f dstack/nemotron3-omni-expanded12-canary-c1.dstack.yml`
and retain the entire output. Require multiple physically distinct compatible
offers where practical. Submit the immutable c1 canary detached once, monitor
the foreground workload without `diagnose`, and collect its private workload
and audit bundles. Recursively list, download, and SHA-256-verify both. Continue
only if all three public native audio-video requests produce valid short
answers.

### Task 6: Run and verify the private expanded pilot

Upload exactly the 12 label-free source media, contract, and marker to the
private c1 input prefix, then list, download, and checksum-verify the upload.
After canary success and fleet release, retain the full `echo "n"` pilot
preview, submit the immutable c1 pilot detached once, and monitor without
secret-bearing diagnostics. Collect and recursively checksum-verify the
workload and audit bundles. Stop on a technical failure; no c2 exists. Require
24/24 frozen outputs before grading.

### Task 7: Grade only after cryptographic freeze verification

Download the immutable frozen bundle and independently verify `SHA256SUMS`,
the manifest-recorded prediction SHA-256, exact 24-record order, and
denominator before references can load. Produce two independently shuffled
blind C/P/W packets and obtain two independent complete grade sets. Invoke a
third independent adjudicator only for disagreements. Score both predeclared
gates, upload the exact private grading bundle, and recursively list, download,
and checksum-verify it.

### Task 8: Record and verify the result

Update only this sanitized evidence record with the exact revisions, preview
hashes, physically distinct offer evidence, actual region/duration/cost,
completion denominator, both quality gates, Evidence validity, Decision
disposition, limitations, rules-out/does-not-rule-out, reconsideration
conditions, private artifact URIs, and checksum-verification status. Run the
full local verification and a final independent review before the result
commit. A valid negative result is complete and must not be rerun for quality.

## Expanded-12 c1 observed result

The expanded follow-up stopped at its predeclared compatibility gate. No
private panel input was uploaded, no private pilot was submitted, and no
reference answer or grade was loaded.

The first preview failed locally before provider admission because
the approved canary experiment identity is 44 characters while dstack 0.20.27
requires scheduler resource names to match
`^[a-z][a-z0-9-]{1,40}$`. Its retained output SHA-256 is
`0a632f9705b86780cab49082198bf1c2f551222b498f70da957d87591eefdf9a`.
That failed preview used code/config revision `5d08ee9`.
The single corrective configuration kept the full immutable experiment and
artifact identity
`nemotron3-omni-expanded12-canary-20260730-c1`, while using the 39-character
scheduler alias `nemotron3-expanded12-canary-20260730-c1`. Tests bind that
mapping in the collector.

The one corrective declined preview passed config validation and showed one
compatible on-demand RunPod L40S 48 GB offer in US-MO-1 at `$0.99/hour`. A
second physically distinct compatible machine was not available. The retained
preview SHA-256 is
`00fcdd67a78a4f1f2c389eec3f3169e4ec6dc29697d610ccaa77fb98fcf4b22f`.
The canary was then submitted detached exactly once. It never obtained an
instance: every provider-admission attempt returned `no offers`, and the
bounded in-run 20-minute no-capacity traversal terminated with
`retry_limit_exceeded` and
`failed_to_start_due_to_no_capacity`. The workload, exact-image interpreter,
model server, and native audio-video seam therefore never started.
The corrective preview and submitted task used code/config revision
`1dec3e2`; terminal scheduler status was `failed`.

The compatibility denominator is **0/3**. The private expanded-pilot
denominator is **0/24 (not submitted)**. Both the sound-replication and broader
expansion quality gates are **not graded**. Expanded c1 incurred **$0.00**;
cumulative model-track spend remains **$0.5781** of the original `$8` base
allocation, and no central reserve was used.

### Expanded-12 c1 durable evidence

- Sanitized failure audit:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-expanded12-canary-20260730-c1/audit/`
- Audit `SHA256SUMS` file SHA-256:
  `ffea773f734a18df126e79051c54d9374806ae0306f7fdc26e6881ca9a3dc76b`
- Exact pinned model:
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Exact model revision:
  `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
- Exact pinned image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`

The six-file private audit bundle preserves both preview outputs, the
sanitized exact-offer observation, the experiment-to-scheduler identity map,
and terminal status/cost metadata. It was recursively listed, downloaded to a
fresh directory, and verified against its remote `SHA256SUMS`. It contains no
credential, private media, competition row, model answer, reference, grade, or
raw log.

### Expanded-12 c1 validity, limits, and decision

- **Evidence validity:** **Incomplete**. The exact-image compatibility gate did
  not start and produced 0/3 required answers, so the 24-output paired pilot
  could not proceed.
- **Decision disposition:** **Not promoted under the tested condition** for the
  expanded-12 follow-up. This incomplete infrastructure result does not
  supersede the completed b1 **Routed candidate** decision for the narrower
  codec-normalized sound condition.
- **Persistent error:** provider admission returned `no offers` throughout the
  bounded internal retry window despite a contemporaneous offer query and
  declined preview observing one US-MO-1 L40S listing. The immutable run
  terminated without provisioning or cost. No c2 or external retry exists.
- **Confirmed defect and correction:** the original canary scheduler name
  exceeded dstack's 40-character limit. The one corrective preview used a
  tested short scheduler alias while preserving the full experiment/artifact
  identity. The corrected preview passed; the subsequent persistent failure
  was provider capacity, not the name validation.
- **Audit gap:** no selected region, actual instance, runtime duration, model
  revision loaded in memory, or workload output exists because no provider
  admitted the run. This is an absence of execution evidence, not missing
  metadata for an executed workload. The preview, offer observation, terminal
  reason, and zero cost are durably reconstructable from the linked bundle.
- **Limitations:** one marketplace listing is not reserved capacity and does
  not prove provider admission. Because the canary never ran, this result says
  nothing about exact-image compatibility, native audio-video correctness,
  completion, answer quality, audio benefit, control regression, or the two
  predeclared expanded quality gates.
- **What this rules out now:** treating expanded c1 as a completed experiment,
  grading or promoting it, claiming 10-or-more-video coverage, or retrying its
  quality under the same authorization.
- **What it does not rule out:** the pinned model/runtime on an admitted L40S,
  a 3/3 public canary, 24/24 expanded-pilot completion, or either quality gate
  under a separately authorized future experiment.
- **Reconsider when:** a new immutable identity is explicitly authorized after
  the shared fleet and provider can admit an exact compatible L40S, followed
  by a fresh declined preview, 3/3 public canary, verified 24/24 freeze, and
  blind grading. The current authorization permits no c2.
- **Shared-report integration:** the primary theme remains models, with native
  audio as a related theme. This isolated branch does not edit the shared
  project-status QMD or generated outputs; the parent integration task owns
  that update. The canonical claim page and anchor are not yet assigned.
- **Supersedes / superseded by:** expanded c1 neither supersedes nor is
  superseded by another claim at the time of this record. It leaves the
  completed b1 result current for its narrower codec-normalized condition.

## Approved expanded-12 d1 capacity rerun

After reviewing the c1 capacity-only failure, the user explicitly requested
that the 12-video test proceed. This separately authorizes one fresh attempt
under new immutable identities:

- public exact-image canary experiment:
  `nemotron3-omni-expanded12-canary-20260731-d1`
- canary scheduler alias:
  `nemotron3-expanded12-canary-20260731-d1`
- paired expanded pilot:
  `nemotron3-omni-expanded12-20260731-d1`

The c1 run and its evidence remain immutable. No d2, external submission loop,
hardware broadening, price increase, spot fallback, or quality-driven rerun is
authorized. The d1 attempt reuses the exact approved 12-row panel, 24 adjacent
audio-off/audio-on requests, questions, normalization, non-thinking greedy
short-answer decoding, model/revision, digest-pinned vLLM 0.20.0 image, scalar
`L40S:1:48GB..` CUDA requirement, RunPod/Vast on-demand backends, and shared
`gpu-marketplace` fleet.

The same sequential gates remain binding: submit the pilot only after a 3/3
public canary; require 24/24 frozen pilot outputs before references load or
grading begins; use two independent blind C/P/W graders and a third only on
disagreements; apply the predeclared sound-replication and broader-expansion
decision rules without rerunning a technically valid low-quality result.

The canary remains capped at 30 minutes and `$2/hour`; the pilot remains capped
at 45 minutes and `$2/hour`. Additional worst-case spend is `$2.50`, making
cumulative worst-case model-track spend `$3.0781`, within the original `$8`
base allocation. No central reserve is authorized. Private inputs and durable
outputs remain under the new d1 HF experiment prefixes, and every transferred
bundle must be recursively listed, downloaded, and checksum-verified.

Implementation is limited to new d1 configs, exact model-specific script and
grading allowlists, focused tests, and this evidence record. Tests must first
fail for the absent d1 identities and provenance acceptance, then pass before
any preview or cloud submission. Shared project-status QMD and generated
surfaces remain outside this branch's ownership.

### Task 9: Add exact d1 execution and grading identities

Modify only `scripts/run-nemotron3-omni-dstack.sh`,
`scripts/upload-nemotron3-omni-pilot-inputs.sh`,
`scripts/collect-nemotron3-omni-dstack.sh`,
`src/trec_video_2026/vqa_experiments/nemotron3_omni_grading.py`,
`tests/test_nemotron3_omni_grading.py`, and
`tests/test_nemotron3_omni_dstack_config.py`. Add
`dstack/nemotron3-omni-expanded12-canary-d1.dstack.yml` and
`dstack/nemotron3-omni-expanded12-d1.dstack.yml`.

1. Add focused failing tests for both exact d1 experiment identities, the
   valid 39-character canary scheduler alias, full immutable HF prefixes,
   absence of any d2 identity, and d1 frozen-provenance verification before
   references load.
2. Add the minimum explicit d1 allowlist branches without weakening c1 or
   historical identity rejection. Keep the existing exact image/model/runtime,
   resource, price, duration, secret, normalization, foreground, and durable
   collection behavior unchanged.
3. Generalize expanded grading only enough for a caller to select the exact
   allowed d1 run identity; retain c1 as the compatibility default and reject
   any unapproved identity or cross-run artifact before references load.
4. Run the full Nemotron suite, Ruff, formatting, shell syntax, YAML parsing,
   privacy/scope scans, and `git diff --check`, then obtain an independent
   read-only review and commit before cloud work.

### Task 10: Execute the sequential d1 cloud gates

Retain and hash the full declined canary preview, submit the d1 canary detached
once, monitor without secret-bearing diagnostics, and recursively verify its
private outputs and audit bundles. Only after 3/3 and fleet release, upload and
round-trip-verify the exact 13 pilot input objects, retain the pilot preview,
submit once, and require a verified 24/24 freeze. Grade only after freeze
verification, upload and round-trip-verify the grading bundle, apply both
predeclared decision rules, update this sanitized record, and commit. Stop on
any technical gate failure or valid negative quality result; no d2 exists.

## Codec-normalized b1 observed result

The separately authorized b1 retry passed its sequential gates. This current
result does not erase the incomplete a1 attempt below: it tests a new,
predeclared codec-normalized condition under new immutable identities.

The codec sentinel preview exposed two physically distinct on-demand RunPod
L40S offers in US-MO-1 and US-TX-4 at `$0.99/hour`. Its retained preview
SHA-256 is
`e6e7dfa3c9f362b24566a97c9b73730009c79f30dee83b11abf76ad41c349fb4`.
The sentinel ran in US-MO-1, normalized and decode-verified all six source
files, and completed 3/3 native audio-video requests on the three formerly
failing AV1 rows. It exited successfully and cost `$0.1626` over
`591.199799` seconds.

After the shared one-node fleet released, the pilot preview showed the same
two regions and price. Its retained preview SHA-256 is
`0f5324d91251b681d4b7569a27468ce193c1399afd42eea7c4f17408443848f3`.
The pilot ran in US-MO-1, independently repeated source verification and
all-six normalization, and froze **12/12** valid outputs before references
loaded. It exited successfully and cost `$0.1720` over `625.319866` seconds.

Two independently shuffled blind graders assigned C/P/W grades. They
disagreed on 10 of 12 items, so a distinct third grader adjudicated exactly
those 10 disagreements. The finalized comparison found audio-on improvement
on **2/4 targets** and regression on **0/2 visual controls**. This meets the
predeclared promotion rule exactly. Because the six-row selected panel is too
narrow to establish a general default, the condition-bound decision
disposition is **Routed candidate** rather than **Current default**.

The local dstack control-plane process was terminated by an unrelated parent
cleanup while the cloud sentinel remained active. The same persisted local
server state was restored solely to resume monitoring the existing immutable
run. No cloud run was resubmitted, altered, or retried, and the terminal cloud
metadata retained the original run identity and submission.

### b1 durable evidence and cost

The b1 runs cost `$0.1626 + $0.1720 = $0.3346`. Including the historical a1
spend, cumulative model-track spend is `$0.5781`, within the original `$8`
base allocation. No central reserve was used.

- Sentinel outputs:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-codec-sentinel-20260730-b1/outputs/`
- Sentinel output `SHA256SUMS` file SHA-256:
  `20f9ad00517c797d50b1a6e5788e9035fbb3290ca11fd10c5d5de3a52fee79cf`
- Sentinel frozen predictions SHA-256:
  `118c6212142aa714d6f3cc9b7cea4b7222b1355a52ae06a60189936b3527cc2b`
- Sentinel audit:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-codec-sentinel-20260730-b1/audit/`
- Sentinel audit `SHA256SUMS` file SHA-256:
  `7d08d7d5fac4aa9b4788aeb24f4e1f012a2e50bf60a6b31dee4e8a51202bb77a`
- Pilot outputs:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-b1/outputs/`
- Pilot output `SHA256SUMS` file SHA-256:
  `61069ca2f9565a5fb70bf792d1ab482f203c7cacb28ada6aff2c4bb3d68e2108`
- Pilot frozen predictions SHA-256:
  `13caa9add0351f31168bc06a48a4ca582f4d96378d324ce0b3feb6ec14961ae6`
- Pilot audit:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-b1/audit/`
- Pilot audit `SHA256SUMS` file SHA-256:
  `8d382e7d8584928f6ea539392aae6dc5bd635dcc66002e7f4ae858db0ce79653`
- Private blind-grading bundle:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-b1/grading/`
- Grading `SHA256SUMS` file SHA-256:
  `dfc414ee68868aa819b7b75eda5c93ad13543f65a9c3bef412dcaef33f4a1909`

All five b1 artifact subprefixes were recursively listed, downloaded, and
verified against their remote `SHA256SUMS`: sentinel outputs, sentinel audit,
pilot outputs, pilot audit, and pilot grading. The audit bundles preserve the
exact preview text and sanitized scheduler, region, duration, price, and
actual cost metadata. Private media, model answers, references, item-level
grades, and raw logs remain outside Git and this report.

### b1 validity, limits, and decision

- **Evidence validity:** **Verified**. The declared sequential protocol
  completed, the sentinel was 3/3, the paired pilot was 12/12, and references
  loaded only after the frozen manifest and prediction hashes verified.
- **Decision disposition:** **Routed candidate** for the declared
  non-speech-sound/music condition. The predeclared pass rule was satisfied,
  but this selected pilot does not establish a general model default.
- **Confirmed defect and correction:** the historical a1 OpenCV path could not
  decode three AV1 video streams. Normalizing all six rows to verified
  H.264/AAC MP4 removed that technical blocker in b1.
- **Uncertainty:** the panel contains only four selected targets and two
  controls. Ten of 12 items required third-party adjudication, showing
  substantial grader variance even though the final protocol was followed.
- **Audit gap:** none for the b1 preview, scheduler, region, duration, cost,
  output-freeze, or grading artifacts. The historical a1 audit gap remains
  documented below.
- **Shared-report integration:** the primary theme is models, with native audio
  as a related theme. The canonical claim page and anchor are **not yet
  assigned** because this isolated workstream is prohibited from editing the
  shared project-status QMD or generated surfaces. The parent integration task
  must update the currently stale Nemotron entry in
  `docs/project-status/research-models.qmd`. This reporting handoff does not
  change the b1 artifact validity or create an artifact audit gap.
- **Limitations:** normalization changes container and codecs for every row;
  decode and duration checks do not prove perceptual equivalence. The selected
  panel cannot estimate prevalence, general visual quality, or performance on
  other questions, codecs, languages, or media distributions.
- **What this rules out now:** the hypothesis that the exact pinned
  codec-normalized Nemotron/vLLM path cannot complete this native AV panel, and
  the claim that audio-on has no condition-bound benefit on any of its four
  selected targets.
- **What it does not rule out:** regressions outside the two controls, failure
  on a broader representative panel, sensitivity to normalization choices, or
  a different conclusion under independent human grading.
- **Reconsider when:** a larger matched panel with more controls, preserved
  source-to-derived perceptual checks, and lower-variance independent grading
  either confirms or reverses the two-target benefit.
- **Supersedes / superseded by:** b1 is the current result for the
  codec-normalized condition. It does not supersede the incomplete a1
  observation for the original source-codec path; no later result supersedes
  b1 at the time of this record.

## Historical a1 scope and predeclared gate

This track tested NVIDIA Nemotron 3 Nano Omni with native video plus optional
audio on a six-row paired panel. The private panel contained four non-speech
sound/music targets and two visual controls. Inference used adjacent audio-off
and audio-on requests, yielding a denominator of 12. Promotion required all 12
requests to complete, audio-on to improve at least two targets, and neither
control to regress.

The model was served non-thinking with greedy, short-answer decoding. No
chain-of-thought was requested or stored. Reference answers were withheld
during inference and were not revealed after the incomplete result, so no
C/P/W grading or quality comparison was performed.

## Exact runtime identity

- Model: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`
- Model revision:
  `dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
- Runtime image:
  `docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`
- Verified runtime: vLLM `0.20.0`, PyTorch `2.11.0+cu130`, CUDA `13.0`,
  NVIDIA L40S 48 GB

## Infrastructure and compatibility gate

The first canary preview, `nemotron3-omni-runner-canary-20260730-a1`, found
no schedulable instance because the shared one-node fleet was occupied. Its
locally retained preview SHA-256 is
`4ebfb73f074757b113e230e6fe10fcc68311b2081dd5c0e9b1169d264ab9cf2c`;
no GPU run was submitted and it incurred no cost.

After that fleet node was released, the single authorized corrective preview
for `nemotron3-omni-runner-canary-20260730-a2` showed three physically
distinct RunPod L40S offers in EU-NL-1, US-MO-1, and US-TX-4, each at
`$0.99/hour`. The locally retained preview SHA-256 is
`399ef80fce9fb63ec8f869804c8cf75f0f2c53ae64480b671118e5c8d64a4d96`.

The a2 canary ran in EU-NL-1 at `$0.99/hour`, completed 3/3 public synthetic
native audio-video requests, exited successfully, and cost `$0.1213` over
`441.055116` seconds. This passed the compatibility gate.

## Focused pilot result

The `nemotron3-omni-pilot-20260730-a1` preview again exposed the same three
distinct compatible regions at `$0.99/hour`. Its locally retained preview
SHA-256 is
`9259c1f6f9a513c58f96e29d8bcf9733eb7dba204c2b6a4d8859215d42452219`.

The pilot ran on an on-demand RunPod L40S in US-MO-1. Input preflight verified
the exact label-free 12-request contract and the six private media hashes
before model loading. The run then completed only 6/12 requests, exited with
status `failed`, and cost `$0.1222` over `444.544765` seconds.

Three source media pairs decoded successfully. For the other three pairs,
vLLM's OpenCV path returned empty sampled-frame arrays after container codec
errors. `NanoNemotronVLProcessor` then raised an index error and the API
returned HTTP 400 for both paired conditions. This is an incomplete technical
result, not a negative quality result.

The track's one corrective attempt had already been consumed by the a2
canary. A pilot a2 was therefore not submitted. The final completion
denominator is **6/12**, target improvements and control regressions are
**not graded**. Its evidence validity is **Incomplete**, and its decision
disposition is **Not promoted under the tested condition**.

## Cost and durable evidence

Actual GPU spend was `$0.1213 + $0.1222 = $0.2435`, within the `$8` base
allocation. No central reserve was used.

- Canary URI:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-runner-canary-20260730-a2/outputs/`
- Canary remote `SHA256SUMS` file SHA-256:
  `0990e5e0f017b75ad9efac6e4705e1d6b17381ca5eca244fdb489b5db716a179`
- Canary frozen predictions SHA-256:
  `c52c4c4e60a60e0482b1708052403893f7a766d34d979ad0dd96932a0de5dc31`
- Pilot URI:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-pilot-20260730-a1/outputs/`
- Pilot remote `SHA256SUMS` file SHA-256:
  `41dfff5ff5ddd6b763003357006e4359dc21450c791d24e05141643e113d29e8`
- Pilot frozen partial predictions SHA-256:
  `4bf3330aa2e5b4f24ae15130a4af376a681ccdb22aaef8ce4356aec2183c7340`

Both remote prefixes were recursively listed, downloaded, and verified against
their remote `SHA256SUMS`. Private media, model answers, reference answers, and
raw logs remain only in ignored storage and the private bucket; none are
committed here.

## Historical a1 evidence validity, limitations, and decision

- **Evidence validity:** **Incomplete**. The compatibility gate completed, but
  the declared paired pilot produced only 6/12 required outputs. The partial
  outputs cannot support the planned audio-on quality comparison.
- **Decision disposition:** **Not promoted under the tested condition**. The
  incomplete run is stopped and cannot be graded or promoted.
- **Confirmed defect:** three source media pairs produced empty sampled-frame
  arrays in the tested vLLM/OpenCV codec path, causing processor failures for
  both paired conditions.
- **Audit gap:** the linked private HF prefixes preserve and checksum the
  workload contracts, manifests, predictions, summaries, and server logs, but
  they do not preserve the preview text or the collector-added sanitized
  `dstack-run.json`. The exact offer regions, selected region, elapsed duration,
  and actual cost above were observed and checksum-verified in ignored local
  records, but cannot be reconstructed from the durably linked evidence bundle
  alone.
- **Limitations:** the public canary used synthetic H.264/AAC media and did not
  represent every codec found in the private MKV panel. The panel contains only
  six selected rows, half of its pairs failed before a valid answer, and no
  reference-backed quality judgment was made.
- **What this rules out now:** promoting or quality-grading this exact
  Nemotron/vLLM/container/input-path run, or using its partial answers to claim
  an audio benefit.
- **What it does not rule out:** the model family on a codec-compatible,
  preflighted media path; a separately authorized fresh experiment with a new
  immutable identity; or native-audio value under a complete blinded panel.
- **Reconsider when:** a separately authorized experiment first passes an
  exact-media codec sentinel, durably links sanitized preview and scheduler/cost
  metadata, completes all 12 paired outputs, and cryptographically verifies the
  frozen manifest and prediction file against `SHA256SUMS`, including the
  manifest-recorded prediction hash, before loading references.
