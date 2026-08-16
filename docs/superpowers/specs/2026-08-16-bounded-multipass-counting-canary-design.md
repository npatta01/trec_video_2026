# Bounded Multi-Pass Video Counting Canary Design

**Date:** 2026-08-16

## Decision

Test a bounded, question-driven video-counting workflow before building an
open-ended video agent or integrating a specialist tracker. The workflow gives
each model pass materially different evidence, freezes a deterministic count
specification, and lets code aggregate timestamped events or localized object
records. It may perform one targeted refinement, but it may not loop.

The canary compares three matched arms on twelve developmental rows: the
existing frozen direct Gemini answer, a fixed multi-pass workflow, and that
same workflow with one bounded model-selected refinement. The hard new-provider
cost ceiling is **$30.00**. A one-row representative smoke must complete with a
valid evidence chain before the remaining rows or the adaptive arm may run.

This is a new condition. It does not modify or rerun any completed issue #34
artifact, the 475-row Gemini panel, the prior generic atomic enumerator, the
typed count-query engine, the incomplete recovery canary, or the invalid local
Gemma canary.

## Question and falsifier

**Question:** Does a fixed workflow that separates count-unit specification,
dense or high-resolution evidence collection, and deterministic aggregation
rescue representative counting failures while preserving correct controls;
and does one targeted refinement add value beyond that fixed workflow?

The main hypothesis is falsified if the fixed workflow rescues fewer than five
of eight failures, changes any of four controls from correct to incorrect, or
fails to beat the frozen direct-answer arm by at least three exact rows.

The refinement hypothesis is falsified if the refinement arm adds fewer than
two exact rows over the fixed arm, regresses a fixed-arm exact row, exceeds
twice the fixed arm's observed provider cost, or cannot remain within the
global cost ceiling.

## Developmental panel

The private panel contains exactly twelve rows selected from the already
corrected 66-row count census:

- four confirmed action-repetition failures;
- three confirmed static or spatial-count failures;
- one confirmed failure from a different supported temporal counting unit;
- two correct temporal controls; and
- two correct static or spatial controls.

Within each stratum, selection is deterministic from a frozen private table
using stable SHA-256 ordering after excluding:

- the four-appearance feasibility row used for post-label development;
- all rows used in the earlier six-row generic atomic canary;
- the hardened one-row Gemini recovery repeat;
- ambiguous and not-count rows; and
- questions whose required identity, part-ownership, or convention semantics
  the frozen router marks unsupported.

Panel roles, references, prior answers, correctness, and failure annotations
are available only to private preparation and post-freeze scoring. Inference
artifacts contain the original question, video/media binding, question-derived
count specification, and evidence created within the current arm. A separate
label-blind projection is frozen and recursively checked before any provider
call.

Because the designer knows the development census and chooses the strata, the
result estimates mechanism viability only. It is not a held-out generalization
estimate and cannot alone justify full-panel replacement.

## Three matched arms

### Arm A: frozen direct-answer baseline

Arm A reuses the immutable Gemini 3.1 Pro native-video 4-fps answer already
recorded for each row in the completed 475-row benchmark. It makes no new
provider call. The scorer authenticates the original prediction freeze before
loading its answers after Arms B and C are frozen.

### Arm B: fixed multi-pass evidence workflow

Arm B has a fixed execution graph.

1. **CountSpec pass.** `gemini-3.1-pro-preview` receives the complete video and original
   question at 4 fps. It returns only a typed specification: observable target,
   count unit, temporal scope, inclusion and exclusion rules, identity policy,
   route, candidate shots or intervals, and uncertainty. It does not return a
   scalar answer.
2. **Evidence pass.** The route determines the evidence, not another free-form
   planning loop:
   - action repetitions and appearance episodes receive endpoint-inclusive,
     overlapping source-time windows at 7.5 fps and return atomic event or
     presence records with source timestamps;
   - static and spatial counts receive one native-resolution frame from every
     candidate shot plus deterministic crops when the target occupies a small
     portion of the frame, and return localized instance records;
   - maximum-simultaneous and per-scene questions use the corresponding typed
     reducer over those same records.
3. **Coverage pass.** A blinded `gemini-3.1-pro-preview` call sees the specification and
   evidence records but not the provisional integer. It checks video endpoints,
   uncovered windows or shots, duplicates, merged actions, missed instances,
   cut boundaries, and scope violations. Arm B abstains if the challenger
   identifies an unresolved gap or record dispute.
4. **Reducer.** Code applies the frozen typed reducer. It never accepts a model
   scalar as the evidence result and never repairs malformed JSON or converts
   uncertainty into an exact answer.

Arm B makes no refinement call. A challenged row either freezes an exact result
with complete evidence or abstains with reason codes.

### Arm C: one bounded refinement

Arm C begins from the same frozen CountSpec and initial evidence policy as Arm
B. When the coverage pass finds a specific disputed interval, shot, crop, or
record, code permits exactly one additional evidence call containing only that
targeted material. The coverage pass is then rerun once over the additive
record. No second refinement, majority vote, recursive planning, or silent
retry is permitted.

Arm C must bind its shared inputs to Arm B so the only treatment difference is
the single refinement opportunity. If no eligible dispute exists, Arm C reuses
the fixed-arm result and incurs no additional call.

## Evidence and reducer contracts

The implementation reuses the existing label-isolation, canonical hashing,
media binding, paid-call claiming, budget, and write-once publication helpers.
It extends the existing blind-frame and typed count-query contracts rather than
creating another independent proof format.

Every exact result requires:

- a question-derived CountSpec with a supported unit;
- source media, delivery, sampling, clipping, frame, crop, and presentation
  hashes sufficient to reconstruct exactly what each provider call saw;
- complete endpoint and window or shot coverage under the frozen policy;
- one accepted timestamped event or localized-object record per counted unit;
- explicit dispositions for every coverage challenge and refinement request;
- deterministic aggregation with selected record and relation identifiers in
  the trace; and
- a terminal self-hashed freeze authenticating every required stage and file.

Missing stages, malformed provider output, incomplete coverage, disputed
records, provider errors, unsupported semantics, identity ambiguity, or budget
refusal produce a typed abstention or incomplete terminal. They never fall
through to an exact count.

## Call, cost, and stopping boundaries

The direct Google Gemini API is the only live provider route. There is no
OpenRouter or local-model fallback inside this condition. No paid call is
automatically retried.

The preflight freezes the exact provider model identifier and returned model
metadata. The runner refuses to begin or continue if the identifier is
unavailable or changes; it never silently substitutes another model.

The per-row ceiling is:

- one CountSpec call;
- at most six deterministic evidence-window or image-batch calls;
- one coverage call for Arm B;
- at most one refinement evidence call and one final coverage call for Arm C;
- no more than ten paid calls in total for a row.

The runner reserves cost before every call using the greater of current
official pricing projection and the highest observed compatible call cost with
a safety margin. Filesystem-exclusive run, upload, and paid-stage claims prevent
concurrent duplicate work. It refuses the next call if the worst-case remaining
condition could cross **$30.00**.

Execution gates are sequential:

1. Offline preparation, label-isolation, synthetic tests, dry-run, and request
   freeze must pass.
2. One action-repetition failure runs first as the representative smoke. It
   must complete Arm B with valid provenance, a valid exact-or-abstain terminal,
   and cumulative cost no greater than **$3.00**.
3. If the smoke is structurally valid, complete Arm B on the twelve rows. Stop
   before Arm C if cumulative cost exceeds **$15.00** or the projected full
   condition could exceed $30.00.
4. Run Arm C only after Arm B is completely frozen. Stop immediately on any
   integrity, authentication, schema, model-identity, or budget failure.

An invalid or incomplete smoke is preserved and ends the live experiment. A
wrong but structurally valid smoke does not by itself stop the panel because
labels remain sealed until the predeclared freeze boundary.

## Metrics and decision rule

After all reachable predictions are frozen, the scorer loads references and
the authenticated Arm A answers. The primary metric is exact numeric
correctness. It also records:

- failure rescue and control-preservation rates by route;
- Arm B versus Arm A and Arm C versus Arm B paired exact differences;
- abstention, unsupported, split, merge, omission, scope, and coverage rates;
- calls, video seconds, sampled frames, tokens, latency, and provider cost;
- METEOR, BERTScore-F1, STS, and the established semantic composite for the
  concise answer rendered from each frozen exact count; and
- manual post-freeze inspection of every rescue, regression, and abstention.

Arm B passes the developmental gate only if it:

- rescues at least five of eight failures;
- preserves all four controls;
- produces at least three more exact-correct rows than Arm A; and
- has no invalid evidence chain or incorrect non-abstained control.

Arm C is retained only if it:

- adds at least two exact-correct rows over Arm B;
- does not regress an Arm B exact-correct row;
- preserves all four controls; and
- costs no more than twice Arm B.

If Arm B passes and Arm C does not, the fixed workflow is the selected design.
If both pass, the bounded refinement is selected. If Arm B fails, neither an
open-ended agent nor scale-up is justified; inspect whether the remaining
bottleneck is perception, count-unit specification, or convention noise before
considering a specialist detector or rented-GPU worker.

## Provenance and privacy

Complete private evidence is written only beneath a new write-once directory
in the canonical shared benchmark root. It includes the private panel,
label-blind projection, media manifests, prompts, schemas, exact provider
requests and raw responses, usage, cost, claims, normalized records, stage
terminals, prediction freezes, scores, inspections, source hashes, dependency
lock, runtime, model revision, and git revision.

Tracked artifacts contain only sanitized aggregates, schemas, reason codes,
hashes, costs, and no competition questions, answers, private media, provider
URIs, or raw responses. The existing counting report remains the primary theme
record. Any valid negative, incomplete, invalid, or corrected result is
preserved under the experiment guide's standard evidence-validity and decision
disposition fields.

## Verification

Before any live call, tests must cover:

- deterministic panel selection and exact stratum cardinalities;
- recursive label-key and value leakage rejection;
- supported and unsupported CountSpec routing;
- endpoint-inclusive overlapping windows and exact decoded PTS bindings;
- native-resolution static frame and crop membership;
- duplicate, missing, overlapping, out-of-bounds, and unresolved records;
- coverage challenge and single-refinement state transitions;
- deterministic action, appearance, static, simultaneous, and per-scene
  reducers on synthetic fixtures;
- required-stage and full referenced-file authentication;
- exclusive upload and paid-call claims under concurrent processes;
- hard budget refusal before a call;
- byte-identical dry-run replay; and
- synthetic end-to-end exact, abstain, provider-failure, and budget-stop paths.

The live evidence is scientifically usable only if the exact frozen condition
executes. Any material deviation is recorded as invalid rather than repaired
after outcome inspection.

## Success boundary

A positive canary establishes that bounded, differently evidenced model passes
can solve a useful developmental slice of counting errors at acceptable cost.
It does not establish full-panel or test-set accuracy. The next step after a
positive result is a fresh held-out counting slice with the selected fixed or
bounded-refinement architecture. A negative result remains useful: it would
show that adding orchestration around the tested frontier perception path is
not enough and would justify focusing the next experiment on a specialist
perception worker rather than more prompting.
