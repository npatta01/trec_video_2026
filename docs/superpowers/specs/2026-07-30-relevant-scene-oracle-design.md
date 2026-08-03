# Relevant-Scene Oracle Exploratory Diagnostic Design

**Date:** 2026-07-30

**Primary theme:** Visual and Temporal Evidence

**Status:** Advisor-refined design and integrity corrections approved for
implementation; Stage 1 is planned separately from conditional inference.

## Objective

Determine, on all ten videos in the frozen 2025 temporal development panel,
whether an adjudicated relevant scene can be faithfully represented in a fixed
12-frame input and whether the pinned visual model answers better when shown
that input than when shown twelve uniformly sampled frames.

The diagnostic is staged. Stage 1 constructs and audits both a six-local-frame
hybrid candidate and a twelve-local-frame scene-only candidate without running
the model. A frozen rule then selects at most one candidate for a matched two-
arm Stage 2 comparison against uniform-12. The Stage 1 result, including that
selection or stop decision, is shared before inference.

This is a new exploratory diagnostic. It deliberately proceeds beyond the
earlier failed primary annotation-agreement gate because the user asked to
inspect and test the relevant scenes directly. It does not amend, rescue, or
supersede that failed-gate result.

## Questions and outputs

The diagnostic answers three questions separately:

1. **Extraction:** do the extracted oracle-local frames visibly contain every
   adjudicated scene component needed to answer the question?
2. **Representation:** do the ordered frames preserve any motion, ordering,
   duration, transition, or co-occurrence relationship required by the
   question, rather than merely showing related static states?
3. **Answering:** when those verified frames are included, does Qwen3.5-4B
   produce a better answer than it produces from uniform-12 at the same frame
   and maximum-pixel budget?

The report must not collapse these into one outcome. A scene-extraction failure
is not a representation failure, a representation failure is not a model-answer
failure, and a model-answer failure after verified extraction and representation
is evidence about answering or fusion, not proof that retrieval was already
sufficient in the baseline.

## Stage 1 hypothesis and falsifier

**Hypothesis:** at least one candidate completely extracts and faithfully
represents the frozen preferred evidence for every row in a cohort of at least
eight labelable, still-frame-representable rows.

**Falsifier:** fewer than eight rows enter that cohort, or neither candidate
passes every cohort row after the single simultaneous repair round. Either
result stops before model requests or inference.

## Scope and claim boundary

- Review all ten frozen temporal-panel videos.
- Use only labeled 2025 training data and the existing ignored shared dataset.
- Complete the offline extraction-and-representation audit for all ten rows,
  freeze the Stage 2 selection or stop decision, and share the Stage 1
  scoreboard before GPU inference.
- Run paired inference only for rows adjudicated `labelable_visual`, classified
  as faithfully still-frame-representable, and having a verified complete frame
  set under the selected treatment. Keep all ten rows visible in every
  completion denominator and report every exclusion reason.
- Treat the panel as selected, previously evaluated development evidence. Do
  not estimate prevalence, 2026 accuracy, or general temporal accuracy.
- Treat oracle-local frames as a human-label-derived diagnostic, never a
  deployable selector or operational baseline candidate.
- Do not start, prepare, or evaluate transcript work.
- Keep media, questions, annotations, frames, contact sheets, answers,
  references, grades, mappings, and raw logs in ignored private artifacts.

## Why adjudication is used

The earlier two-primary agreement gate failed narrowly: structural agreement
was `8/10`, while median temporal IoU was `0.4791666666666665` against the
required `0.50`. Therefore those annotations cannot support a confirmatory
boundary-sensitive claim.

For this exploratory run, reviewer C—whose de novo annotation was frozen while
the A/B result remained hidden—will inspect the frozen A, B, and C annotations
and adjudicate all ten rows. The adjudicator selects a final eligibility value,
one or more minimal sufficient evidence sets, and one `preferred_set_index`; it
does not average interval boundaries. Before candidate frames exist, the
adjudicator also freezes an opaque component ID and role for every interval in
the preferred set and every required `motion`, `ordering`, `duration`,
`transition`, or `cooccurrence` relationship with its component bindings. The
adjudication artifacts bind the three frozen annotation hashes and are
explicitly labeled exploratory.

Alternatives were rejected:

- The union of all reviewer intervals would artificially enlarge the relevant
  scene and make extraction success too easy.
- Three reviewer-specific oracle conditions would triple inference and answer
  grading while still avoiding the necessary judgment about which scene is
  sufficient.

## Candidate conditions

### Control: `uniform-12`

Select the nearest unused decoded frame to each midpoint of twelve equal-width
whole-video bins.

### Candidate A: `oracle-hybrid-12`

Use exactly six global and six oracle-local frames:

- Global frames are nearest unused decoded frames to six equal-bin whole-video
  midpoints.
- Local frames use exactly the sufficient set named by the adjudicated
  `preferred_set_index`; legacy shortest-set selection is not used.
- Assign one local slot to every conjunctive interval, then allocate remaining
  slots proportional to interval duration using Hamilton largest remainder.
- Place desired local times at equal-sub-bin midpoints inside each interval.
- Bind to actual decoded PTS while preserving stream index, frame index, time
  base, requested time, extraction identity, source hash, and frame hash.
- Require twelve unique decoded frame identities and sort them by actual PTS
  before model input.

### Candidate B: `oracle-local-12`

Use twelve oracle-local frames and no global frames. Use exactly the sufficient
set named by the adjudicated `preferred_set_index`, assign one slot to every
conjunctive interval, and allocate the remaining slots proportional to interval
duration with the same Hamilton largest-remainder and equal-sub-bin-midpoint
rules as Candidate A. Bind to unique actual decoded PTS with the same provenance
and chronological ordering requirements.

All three conditions use the same 12-frame count and maximum-pixel proxy. Stage
1 audits both candidates, but Stage 2 compares the control with at most one
candidate.

## Predeclared Stage 2 condition selection

Freeze the Stage 2 selection record after all Stage 1 audits and repairs freeze
but before any model request or prediction is created:

1. Define the prospective paired cohort as every row adjudicated
   `labelable_visual` and judged faithfully still-frame-representable.
2. If fewer than eight rows are in that cohort, stop after Stage 1 and report
   insufficient still-representable evidence.
3. If `oracle-hybrid-12` passes the complete extraction-and-representation audit
   for every prospective paired row, select it. This preserves global context
   while directly satisfying the evidence audit.
4. Otherwise, if `oracle-local-12` passes for every prospective paired row,
   select it. This is the denser fixed-frame oracle upper bound.
5. Otherwise, stop after Stage 1. Do not create a favorable subset by dropping
   candidate failures and do not run an under-instantiated treatment.

The selection record binds the protocol, adjudication, both candidate plans,
both audit results, every exclusion, and the exact selected condition or stop
decision. A three-arm answer comparison is not part of this smallest diagnostic.
Rows whose evidence cannot be represented by ordered timestamped stills are
reported as modality failures and may motivate a separately designed native-
clip follow-up; they are not silently treated as model failures.

## Relevant-scene visual audit

Model inference cannot begin until both candidate extractions and their
representational sufficiency are audited.

For each labelable row, build two private, opaque candidate packets: one with
the complete twelve-frame hybrid input and one with the twelve-local-frame
input. Preserve chronological order and show the exact timestamp before each
frame, matching the model prompt. Each displayed frame must be the same bounded-
pixel PNG reserved for later model input; auditors must not receive a higher-
resolution copy. Randomize candidate presentation order per auditor with a
frozen seed. Do not identify the candidate, distinguish local from global
frames, or include references, model answers, grades, uniform frames, or
condition outcomes. This is partial condition blinding: timestamp distribution
may allow an auditor to infer which candidate is denser locally, and the report
must disclose that limitation.

Each fresh auditor completes a frame-first pass before receiving the full source
video or adjudicated interval packet. For each candidate, the auditor records a
sealed short answer attempt or `not_answerable`, confidence, and which temporal
relations appear available from the frames alone. The sealed human answer is an
audit diagnostic only: it remains private, is never supplied to the model, and
cannot be opened during request construction or prediction.

After both frame-first passes freeze, each auditor receives the question, full
source video, actual PTS, opaque interval labels, and a separate timecoded sheet
containing only the candidate's local audit-target frames. The sheet does not
name the candidate. The auditor then independently records:

- a representation class containing one or more of `static_state`, `motion`,
  `ordering`, `duration`, `transition`, `cooccurrence`, or `uncertain`;
- a question-level `still_frame_representable`, `still_frame_incompatible`, or
  `uncertain` judgment based on whether ordered timestamped stills can in
  principle preserve the required evidence;
- for every conjunctive interval, one visibility grade from `visible`,
  `partial`, `not_visible`, or `uncertain`. `visible` requires at least one local
  audit-target frame to contain the required event or state with enough context
  for that interval's role; `partial` contains related but insufficient content;
  `not_visible` means the event or state is absent.

The auditor also assigns `preserved`, `not_preserved`, `not_applicable`, or
`uncertain` to every required temporal relationship, and assigns the complete
candidate input `answer_sufficient`, `answer_insufficient`, or `uncertain`.
After both independent audit records freeze, a fresh third auditor adjudicates
every disagreement without condition identities, model outputs, references, or
grades.

A candidate passes a row only when every interval in the exact frozen preferred
set is `visible` in its local audit-target frames, every relationship frozen for
that set is `preserved`, and the complete input is `answer_sufficient` after
disagreement adjudication. `not_applicable` is valid only for a relation class
that was not frozen as required; `uncertain` never passes. Alternative sets are
not used to rescue a candidate that was constructed from the preferred set. The
frame-first answer attempt is descriptive and does not override these gates.

If either candidate has a documented `partial` or `not_visible` local-frame gap
because a desired time landed between useful decoded frames, one predeclared
simultaneous repair round is allowed before condition selection. Each auditor
records one proposed event-center time to `0.1` seconds for each failed
preferred-set interval; the third auditor resolves disagreement without seeing
model output. Repair is prohibited for still-frame incompatibility, missing
global context, `uncertain`, relationship failure, or generic answer
insufficiency. The extractor chooses the nearest unused decoded frame to each
resolved time, constrained to the original interval and breaking equal-distance
ties toward the earlier PTS. Freeze proposed/resolved times, replacement
reasons, original/replacement identities, and complete frame provenance, then
rerun the complete candidate-blind packet once. A second repair round is
invalid. Rows still failing remain in the ten-row denominator and participate
in the predeclared Stage 2 stop/selection rule.

This audit distinguishes whether the relevant scene was extracted from whether
the still-frame representation preserved what made the scene answer-relevant;
interval-hit recall alone is insufficient.

## Pinned inference

Run only after adjudication, both candidate constructions, both visual audits,
repairs, the Stage 1 user-facing scoreboard, and the condition-selection record
freeze.

- Model: `Qwen/Qwen3.5-4B`
- Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Runtime image:
  `localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`
- Seed: `0`
- Decoding: greedy
- Maximum new tokens: `32`
- Per-frame processor bounds: `256*256` through `512*512`
- Prompt: the existing temporal question-plus-images, no-transcript prompt

Use cached local model/runtime assets and rootless ROCm only. Do not pull a new
image or model and do not expose an inference service.

Run a paired three-row compatibility subset first, using the first three paired
rows in frozen panel order. Full inference requires all six subset outputs to
pass runtime, provenance, completeness, and nonempty answer checks. Freeze
every complete prediction before loading references.

## Blind answer grading

After prediction freeze, create a cryptographically randomized condition-blind
packet. Each opaque row contains the reference and two answer candidates, with
the condition mapping stored separately and withheld.

Two fresh graders independently assign `C`, `P`, or `W`; a third grader
adjudicates disagreements without condition identities. Freeze all grades and
adjudication before opening the condition map once.

Report paired transitions and ordinal scores with `C=2`, `P=1`, and `W=0`.

## Interpretation rule

The primary result is descriptive and has three gates:

1. **Extraction works** when every paired row has a visually verified complete
   sufficient evidence set in the treatment's local audit-target frames.
2. **Representation works** when every paired row preserves all required
   temporal relationships and is answer-sufficient in the complete treatment
   input.
3. **Relevant scenes help answering** when, among complete paired rows, the
   selected treatment improves at least two rows, regresses zero rows, and has
   an ordinal-score gain of at least two over uniform-12.

Passing all three gates supports further label-blind retrieval research. It
does not promote the selected treatment, change uniform-12 as the baseline, or
establish a general effect. Failing the answer gate after passing extraction
and representation indicates that showing the adjudicated scene was not
sufficient to improve this pinned model under the tested prompt and budget.

Any incomplete condition, unresolved provenance mismatch, or reference access
before prediction freeze makes the answer comparison incomplete or invalid;
preserve the evidence and stop rather than filling missing values.

## Artifact and freeze order

Use a new ignored private run directory and never mutate the previous
agreement-stop bundle.

1. Freeze exploratory protocol, claim boundary, panel, code, model, runtime,
   prompt, both candidate definitions, and decision rules.
2. Verify and bind the three existing de novo annotation artifacts.
3. Freeze exploratory adjudication.
4. Freeze decoded-frame inventories and all three condition plans.
5. Extract/hash frames and freeze provenance.
6. Freeze both frame-first passes, full-video audits, adjudications, and any
   single repair round.
7. Freeze the Stage 2 condition-selection or stop record.
8. Freeze and share the sanitized Stage 1 extraction-and-representation
   scoreboard, including the selection or stop decision.
9. Freeze final requests and extraction-completeness decision.
10. Run/freeze compatibility predictions.
11. Run/freeze complete paired predictions.
12. Load references and freeze the blind grading packet/map.
13. Freeze two grade files and disagreement adjudication.
14. Unblind once and freeze the exploratory evaluation.
15. Write a sanitized report that keeps extraction, representation, and
    answering separate.

Every artifact must refuse overwrite and have a verified SHA-256 sidecar or
freeze record that binds its upstream inputs.

## Failure handling

- Adjudicated fewer than eight labelable rows, or fewer than eight labelable rows
  judged faithfully still-frame-representable: report insufficient panel and do
  not run GPU inference.
- Cannot construct twelve unique actual decoded frames: record the affected row
  and stop before inference rather than duplicate frames.
- Neither candidate passes every prospective paired row after the one
  predeclared repair round: freeze a Stage 1 stop and do not run GPU inference.
- Compatibility failure, missing GPU/runtime attestation, or output mismatch:
  stop before the full run.
- Partial paired inference: preserve outputs but do not compute a completed
  paired decision.
- Grader disagreement: use the frozen third-party adjudication; never infer a
  grade from semantic metrics alone.

## Verification and reporting

Before handoff:

- verify the shared-data symlink resolves to the canonical dataset;
- validate every artifact binding and sidecar;
- rerun focused and full tests plus Ruff and `git diff --check`;
- independently review code, protocol, aggregate calculations, privacy, and
  claim wording;
- render QMD-derived report surfaces and compare an isolated regeneration;
- scan reports for media paths, questions, IDs, frames, answers, and logs;
- refresh only sanitized HTML on the existing tailnet-only portal and verify
  the live HTTPS artifact.

The Stage 1 user-facing result must state, separately for six-local hybrid and
twelve-local scene-only inputs, how many of ten scenes were visibly extracted,
how many preserved their required temporal relationships, how many were
answer-sufficient to the auditors, the still-incompatible count, and the frozen
Stage 2 selection or stop decision. It must also report raw independent-auditor
agreement and disagreement counts for representability, interval visibility,
relationship preservation, and answer sufficiency. If Stage 2 runs, its result
must add the paired-inference denominator, grade transitions, runtime, evidence
validity, and remaining unknowns. Stop and share each completed visual result
before any transcript work.
