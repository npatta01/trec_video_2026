# Task-support sufficiency audit — sanitized evidence index

This directory is the tracked, sanitized evidence-of-record index for the
2026-08-15 task-support sufficiency audit. The complete private audit bundle
is retained under the approved benchmark root
`data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/`; row content,
questions, references, answers, and evidence text are intentionally not
copied here.

## Current result

The treatment-first audit retained all `475/475` eligible source rows and
adjudicated a `32`-row audit denominator: `24` low-score candidates and `8`
controls. The corrected v3 role counts were:

| Role | Count | Required | Result |
|---|---:|---:|---|
| Answer-sufficient target | 1 | 4 | Fail; deficit 3 |
| Insufficiency safeguard | 17 | 4 | Pass |
| Supported control | 8 | 4 | Pass |
| Reference/question issue (nonselectable) | 6 | — | Recorded |

The offline feasibility gate therefore **failed**. Provider authorization was
blocked, with zero provider/API/native-video calls, no provider cost, and no
provider metrics. No replacement selection or protocol was created. The
original v1 selection and protocol remain preserved as historical artifacts but
are invalid for provider authorization.

## Exposure accounting

Prior exposure was retained only as a reporting flag; it was not an eligibility
exclusion or a fresh generalization stratum. Candidate-level exposure counts
were hard audit `30`, grounded v2 `24`, grounded v3 `11`, multi-interpretation
`12`, seven-row canary `7`, scope control `7`, and synthesis follow-up `7`.

## Evidence status and decision

- **Evidence validity:** valid but negative offline feasibility result; the v3
  correction is the current interpretation and supersedes v2.
- **Decision disposition:** **Not promoted under the tested condition**.
- **Interpretation:** the declared provider experiment was not feasible under
  the frozen panel because the answer-sufficient target quota was short by
  three rows. This is a panel-feasibility finding, not a provider-quality
  result.
- **Next action:** do not run the provider task from the invalidated v1
  selection. Any future verifier test requires a newly reviewed panel and
  explicit reauthorization.

## Reproducibility bindings

`summary.json` records the aggregate result, private-artifact SHA-256 values,
and the no-call boundary. `SHA256SUMS` covers the tracked files in this
directory. The relevant private benchmark root and source artifacts are bound
by hashes in the summary; no private row-level material is exposed.

## Limitations

- The audit is label-aware and uses frozen map/reference/direct-answer
  material only for sufficiency adjudication; it does not measure provider
  behavior or answer quality.
- Exposed-row flags describe provenance only. They do not turn this selected
  audit into a fresh prevalence or generalization estimate.
- Six nonselectable reference/question issues are recorded in the denominator;
  they cannot be promoted into target, safeguard, or control quotas.
- Since the gate failed before authorization, provider completion, cost,
  rescue, safeguard-verification, control-preservation, unsupported-answer,
  and metric fields are not applicable.
