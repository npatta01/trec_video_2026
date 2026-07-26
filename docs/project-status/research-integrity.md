# Experiment Integrity and Evaluation


<span id="back-to-research-themes" class="legacy-fragment-alias"
aria-hidden="true"></span>
<nav class="status-breadcrumb" aria-label="Breadcrumb">

<a href="index.md">Start Here</a> <span aria-hidden="true">/</span>
<a href="experiment-ledger.md">Research Themes</a>
<span aria-hidden="true">/</span> <span aria-current="page">Experiment
Integrity and Evaluation</span>
</nav>

<nav class="theme-local-nav" aria-label="On this theme page">

<a href="#current-working-answer">Working answer</a>
<a href="#what-has-been-tried">Tried</a>
<a href="#known-failures-and-cases">Failures</a>
<a href="#cheapest-valid-next-experiments">Next tests</a>
</nav>

<span id="why-this-theme-matters" class="legacy-fragment-alias"
aria-hidden="true"></span>

## New to this theme?

An answer comparison is useful only when its input evidence, model
identity, label boundary, prediction freeze, and scoring procedure are
all known. A run that completes can still be invalid for a comparative
conclusion, and a small compatibility probe can still be valuable
without estimating accuracy.

For example, two model answers cannot be compared fairly if one was
produced after the expected answer became visible. A **frozen
prediction** is saved before that disclosure; **label-blind** inference
prevents the answer or grade from selecting evidence. These terms use
the definitions on [Start Here](index.md#newcomer-glossary).

<span id="technical-summary" class="legacy-fragment-alias"
aria-hidden="true"></span>

## Current working answer

Use frozen-30 protocol v2, not frozen-30 v1, for comparative
conclusions. The three-video gate is evidence of portability and
pipeline compatibility only. The current repository has useful tested
seams, but a reusable integrity gate is **Open work**, not an existing
supported CLI.

<div class="theme-decision-strip" aria-label="Integrity decisions">

- **Use now:** frozen-30 protocol v2 for matched comparisons.
- **Do not default:** compatibility success as evidence of answer
  quality.
- **Still possible:** a reusable end-to-end integrity gate and a later
  audited scorer.
- **Next test:** deliberately invalid fixtures that must fail before
  model cost is incurred.

</div>

## Evidence at a glance

- **Strongest evidence:** 120 frozen VQA predictions and 30 Whisper
  executions under protocol v2’s separated freeze and scoring boundary.
- **Confirmed defect:** frozen-30 v1 could not demonstrate label
  isolation.
- **Audit gap:** no supported reusable
  `prepare → run → freeze → score → report` command exists.
- **Unknown:** answer quality on the 2026 evaluation set.

## What has been tried

<a name="field-guide"></a>

### July 12 field guide

- **Reason for testing:** establish an evidence-first research direction
  before local comparisons existed.
- **Tested condition and panel:** a synthesis of prior TRECVID material,
  open model options, scoring constraints, and staged architecture; it
  was not a local benchmark.
- **Observed result:** it identified the need for observable matched
  evidence and label-safe workflows.
- **Current condition-bound status:** background orientation,
  **Superseded** as a baseline-selection source by later executed
  evidence.
- **What this rules out now:** using the July 12 synthesis alone to
  select the current baseline.
- **What it does not rule out:** the field guide’s architectural ideas
  under a later matched, label-isolated test.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—the
  synthesis predates local comparison evidence; no execution defect is
  claimed.
- **Reconsider when:** a specific idea has a declared panel, controls,
  and current execution evidence.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The field guide predates the local panels. It can explain why the
project prioritized evidence boundaries, but cannot choose a current
model, prompt, or frame policy. See the [field-guide source notes ·
Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/source_notes.md).
</details>

### Data readiness

- **Reason for testing:** prepare private media and labels without
  committing them or exposing labels to inference-side work.
- **Tested condition and panel:** resumable acquisition, shared-data
  linking, source validation, and label-isolated preparation seams; no
  answer-quality panel.
- **Observed result:** local evidence can be prepared and audited while
  media, labels, predictions, and raw logs remain ignored.
- **Current condition-bound status:** operationally useful; **not** an
  answer-quality result.
- **What this rules out now:** treating successful preparation as a
  quality or model-ranking result.
- **What it does not rule out:** a later controlled answer-quality
  comparison.
- **Possible defect, audit gap, or uncertainty:** **Audit
  gap**—preparation success does not attest a complete general
  execution-to-scoring workflow.
- **Reconsider when:** a label-isolated, frozen comparison is scored on
  a declared panel.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Readiness does not prove that every future video is available, that a
model will run, or that answers will be correct. Operational readiness
does not establish 2026 answer quality. The [benchmark construction
module · Current code · Source code ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/main/src/trec_video_2026/vqa_benchmark.py)
binds manifest identities and hashes; the shared data boundary remains
local.
</details>

<a name="compatibility-gate"></a>

### Three-video compatibility gate

- **Reason for testing:** learn which model-family adapters could
  execute on the local AMD/ROCm host before spending on a larger panel.
- **Tested condition and panel:** five model families on three
  deliberately different labeled 2025 examples.
- **Observed result:** Qwen, Molmo, InternVL, and Gemma completed all
  requested inferences; MiniCPM completed two of three and had a
  speech-row tensor-shape error.
- **Current condition-bound status:** **Diagnostic signal** for adapter
  and runtime compatibility, not a statistical model ranking.
- **What this rules out now:** promoting a family from this three-row
  portability check or scaling MiniCPM’s observed speech-row failure.
- **What it does not rule out:** a repaired adapter or a model’s quality
  on a larger, representative controlled panel.
- **Possible defect, audit gap, or uncertainty:** **Uncertainty**—three
  selected rows measure portability, not general completion or quality.
- **Reconsider when:** a complete matched quality panel passes the
  integrity gate with the repaired condition, if applicable.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

Labels were joined after inference, but three deliberately selected rows
are too small to estimate competition accuracy. The current public
`vqa-gate` CLI exposes preflight only; completed runs used tested Python
seams. The [three-video gate report · Evidence of record · Evidence
report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/three_video_gate_results_2026-07-20.md)
documents an audited execution, not a general one-command workflow.
</details>

<a name="frozen-30-protocol-v2"></a>

### Frozen-30 protocol transition: v1 to v2

- **Reason for testing:** make a matched 30-row comparison interpretable
  after v1 could not demonstrate label isolation.
- **Tested condition and panel:** protocol v2 reran four matched
  conditions on 30 labeled 2025 examples (120 VQA predictions and 30
  Whisper executions).
- **Observed result:** a network-disabled freeze container selected and
  froze predictions without references; a separate evaluation container
  loaded labels only after the freeze and rejected changed bytes.
- **Current condition-bound status:** v2 is the comparative protocol;
  frozen-30 v1 label isolation is a **Confirmed defect** and v1 is
  historical audit evidence only.
- **What this rules out now:** using v1 predictions for frozen-30
  comparative conclusions.
- **What it does not rule out:** a valid comparison under v2 or a later
  integrity-preserving protocol.
- **Possible defect, audit gap, or uncertainty:** **Confirmed
  defect**—v1 could not demonstrate label isolation; this does not
  transfer to v2.
- **Reconsider when:** only if historical v1 evidence is explicitly
  discussed as an audit record rather than reused for a comparison.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

V1 inference mounted a dataset root containing labels, preparation
opened annotation content, and Qwen device/dtype strings were configured
rather than live-attested. V2 reran rather than reusing v1 predictions,
with minimal mounts, source/model/image attestation, immutable
prediction bytes, and a two-phase scoring boundary. Reproducible
execution still does not establish answer quality; its 2025
training-data comparison is not a 2026 estimate. See the [protocol-v2
report · Evidence of record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md).
</details>

### Scorer-reuse audit

- **Reason for testing:** decide whether the published NIST tooling can
  support local development and 2026 submission checks.
- **Tested condition and panel:** static audit of pinned revision
  `e06cb7854d7defe53e86412c2bba0ca5f636ea40`, including a small scorer
  hazard check; not a VQA accuracy panel.
- **Observed result:** the 2026 AG JSON validator is suitable for format
  validation, while the CSV evaluators remain 2025 historical-parity
  tools with dependency, network, validation, and repeat-run hazards.
- **Current condition-bound status:** reuse behind a pinned, isolated
  adapter; do not call it the final official 2026 scorer.
- **What this rules out now:** labeling the current historical CSV
  evaluator an official final 2026 score.
- **What it does not rule out:** pinned historical-parity checks or a
  later audited authoritative 2026 evaluator.
- **Possible defect, audit gap, or uncertainty:** **Audit gap**—final
  2026 scoring authority and its audited interface were not available.
- **Reconsider when:** NIST publishes updated scoring code and the
  project re-audits its interface, dependencies, and fixtures.

<details>

<summary>

Method, result, validity risks, and evidence
</summary>

The audit identified shell interpolation, a skipped-query denominator
bug, directory rescanning, and unpinned dependencies among the reasons
to isolate the historical evaluator. NIST’s 2026 scoring implementation
was not final at the audit date. The [scorer-reuse audit · Evidence of
record · Evidence report ·
GitHub](https://github.com/npatta01/trec_video_2026/blob/814e55429f6dd1dc8570bc08f1dad849f74e36a1/reports/trecvid_vqa_2026/nist_scorer_reuse_audit_2026-07-20.md)
recommends a pinned external checkout/cache, local fixtures, and the
validator as a final format gate.
</details>

## Known failures and cases

The [Failure Library](failure-library.md) shows why integrity records
must distinguish frozen output from post-hoc diagnosis. Its selected
cases are development evidence, not a held-out validation set.

### What the evidence supports

- A valid comparison needs a label-free inference phase, an immutable
  prediction freeze, and separately mounted evaluation references.
- Protocol v2 replaces v1 for frozen-30 comparative conclusions.
- Reproducibility, data readiness, and adapter compatibility are
  necessary integrity evidence, but do not establish answer quality.
- A pinned validator can check current 2026 AG submission shape;
  historical parity metrics need an explicitly quarantined context.

### What it does not rule out

- V2 does not prove 2026 evaluation quality, generalization beyond its
  30 labeled 2025 rows, or superiority under another model, prompt, or
  panel.
- The compatibility gate does not rule out model differences on a
  larger, representative, controlled quality panel.
- The scorer audit does not rule out a later authoritative 2026
  evaluator; it requires re-audit when one is published.

<span id="possible-bugs-confounds-and-audit-gaps"
class="legacy-fragment-alias" aria-hidden="true"></span>

### Confirmed defects, audit gaps, and uncertainties

- **Confirmed defect:** frozen-30 v1 could not demonstrate label
  isolation. It must not supply comparative conclusions.
- **Audit gap:** there is no reusable supported
  `prepare → run → freeze → score → report` integrity CLI. Existing
  `vqa-gate` is preflight-only, even though individual seams and audited
  workflows exist.
- **Uncertainty:** frozen-30 v2 is a descriptive 2025 training-data
  comparison, not a representative 2026 evaluation estimate.
- **Audit gap:** the NIST scorer remains a pinned historical-parity and
  format validation dependency until NIST releases and the project
  audits a final 2026 evaluator.

## Open hypotheses and ideas

- A single integrity contract with deliberately invalid fixtures can
  make label, mutation, identity, and scorer-boundary failures
  detectable before an expensive run.
- A small representative panel and scorer-validation gate may prevent a
  technically reproducible but decision-unhelpful large benchmark.

## Cheapest valid next experiments

### Reusable integrity gate

- **Hypothesis:** a reusable `prepare → run → freeze → score → report`
  gate can reject boundary violations before model cost is incurred.
- **Smallest valid panel or work item:** one label-free fixture and
  deliberately invalid fixtures for label visibility, prediction
  mutation, wrong request/result identity, and malformed scorer input.
- **Matched controls and integrity safeguards:** fixed
  source/model/runtime attestation; network-disabled freeze; separate
  read-only evaluation mount; known-valid fixture; and assertions that
  every invalid fixture fails for its intended reason.
- **Decision or promotion rule:** promote the gate only if the valid
  fixture completes reproducibly and every invalid fixture is rejected
  without loading labels during inference.
- **Boundary on what a positive result would establish:** it would
  establish a reusable integrity check, not model quality or final 2026
  scoring authority.

### Representative-panel and scorer-validation gate

- **Hypothesis:** a declared representative panel plus scorer validation
  makes a subsequent expensive comparison decision-relevant and
  correctly formatted.
- **Smallest valid panel or work item:** a label-blind panel-design
  document, stratum counts, a small fixture submission, and
  validator/parity checks.
- **Matched controls and integrity safeguards:** freeze panel membership
  before inference; keep model/prompt/decoding comparable; run the
  pinned 2026 JSON validator; isolate any 2025-parity scorer in a fresh
  directory.
- **Decision or promotion rule:** authorize an expensive run only when
  strata, panel identities, and scoring mode are declared and the
  fixture passes the validator without relying on an unofficial score as
  2026 authority.
- **Boundary on what a positive result would establish:** it would
  establish a valid evaluation plan, not accuracy on the unlabeled 2026
  set.

## Method, result, validity risks, and evidence

<details>

<summary>

How to audit this theme’s records
</summary>

Each exact experiment heading above remains visible and carries the
tested condition, observed result, current status, and reconsideration
rule. Open its native details block for source links, method, and
validity risks. Do not infer answer quality from data readiness,
compatibility, or successful execution.
</details>

## Related themes

- [Visual and Temporal Evidence](research-visual-evidence.md) needs this
  page’s freeze and comparison boundaries.
- [Speech and Transcript Evidence](research-speech-transcripts.md) adds
  provenance-specific label and transcript controls.
- [Code Map and Remaining Gaps](code-map.md#remaining-gaps) records the
  missing general workflow as an engineering gap.
