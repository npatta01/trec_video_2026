# Relevant-Scene Oracle Exploratory Diagnostic Design

**Date:** 2026-07-30

**Primary theme:** Visual and Temporal Evidence

**Status:** Design approved by the user; awaiting written-spec review before
implementation planning.

## Objective

Determine, on all ten videos in the frozen 2025 temporal development panel,
whether an adjudicated relevant scene can be extracted into a fixed 12-frame
input and whether the pinned visual model answers better when shown that input
than when shown twelve uniformly sampled frames.

This is a new exploratory diagnostic. It deliberately proceeds beyond the
earlier failed primary annotation-agreement gate because the user asked to
inspect and test the relevant scenes directly. It does not amend, rescue, or
supersede that failed-gate result.

## Questions and outputs

The diagnostic answers two questions separately:

1. **Extraction:** do the extracted oracle-local frames visibly contain every
   adjudicated scene component needed to answer the question?
2. **Answering:** when those verified frames are included, does Qwen3.5-4B
   produce a better answer than it produces from uniform-12 at the same frame
   and maximum-pixel budget?

The report must not collapse these into one outcome. A scene-extraction failure
is not a model-answer failure, and a model-answer failure after verified scene
extraction is evidence about answering or fusion, not proof that retrieval was
already sufficient in the baseline.

## Scope and claim boundary

- Review all ten frozen temporal-panel videos.
- Use only labeled 2025 training data and the existing ignored shared dataset.
- Run paired inference only for rows adjudicated `labelable_visual` and having a
  visually verified oracle-scene frame set. Keep all ten rows visible in the
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
and adjudicate all ten rows. The adjudicator selects a final eligibility value
and one or more minimal sufficient evidence sets; it does not average interval
boundaries. The adjudication artifact binds the three frozen annotation hashes
and is explicitly labeled exploratory.

Alternatives were rejected:

- The union of all reviewer intervals would artificially enlarge the relevant
  scene and make extraction success too easy.
- Three reviewer-specific oracle conditions would triple inference and answer
  grading while still avoiding the necessary judgment about which scene is
  sufficient.

## Frozen conditions

### Control: `uniform-12`

Select the nearest unused decoded frame to each midpoint of twelve equal-width
whole-video bins.

### Treatment: `oracle-scene-12`

Use exactly six global and six oracle-local frames:

- Global frames are nearest unused decoded frames to six equal-bin whole-video
  midpoints.
- Local frames use the adjudicated preferred sufficient set.
- Assign one local slot to every conjunctive interval, then allocate remaining
  slots proportional to interval duration using Hamilton largest remainder.
- Place desired local times at equal-sub-bin midpoints inside each interval.
- Bind to actual decoded PTS while preserving stream index, frame index, time
  base, requested time, extraction identity, source hash, and frame hash.
- Require twelve unique decoded frame identities and sort them by actual PTS
  before model input.

Both conditions use the same 12-frame count and maximum-pixel proxy.

## Relevant-scene visual audit

Model inference cannot begin until the treatment extraction is audited.

For each labelable row, build a private packet containing the question, full
source video, a timecoded contact sheet of the six oracle-local frames, actual
PTS, and opaque interval labels. Do not include references, model answers,
grades, uniform frames, or condition outcomes.

Two fresh auditors independently grade every conjunctive interval:

- `visible`: at least one selected local frame visibly contains the required
  event/state and enough context for that interval's role;
- `partial`: the frame contains related content but not sufficient evidence;
- `not_visible`: the required event/state is absent;
- `uncertain`: media or question ambiguity prevents a confident judgment.

A row passes extraction only when every interval in at least one adjudicated
sufficient set is graded `visible` after disagreement adjudication. Alternative
sufficient sets remain alternatives; components from different alternatives
cannot be mixed.

If a row fails because the desired time landed between useful decoded frames,
one predeclared repair round is allowed before inference. Each auditor records
one proposed event-center time to `0.1` seconds inside the same adjudicated
interval; the third auditor resolves disagreement without seeing model output.
The extractor then chooses the nearest unused decoded frame to the resolved
time, constrained to that interval and breaking equal-distance ties toward the
earlier PTS. Freeze the proposed/resolved time, replacement reason, and complete
frame provenance, then rerun the blind scene audit. Rows still failing remain
in the ten-row denominator but are excluded from paired answer inference.

This audit directly establishes whether the relevant scene was actually
extracted; interval-hit recall alone is insufficient.

## Pinned inference

Run only after adjudication, construction, and visual audit artifacts freeze.

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

The primary result is descriptive and has two gates:

1. **Extraction works** when every paired row has a visually verified complete
   sufficient evidence set in the treatment frames.
2. **Relevant scenes help answering** when, among complete paired rows,
   oracle-scene-12 improves at least two rows, regresses zero rows, and has an
   ordinal-score gain of at least two over uniform-12.

Passing both gates supports further label-blind retrieval research. It does not
promote oracle-scene-12, change uniform-12 as the baseline, or establish a
general effect. Failing the answer gate after passing extraction indicates that
showing the adjudicated scene was not sufficient to improve this pinned model
under the tested prompt and budget.

Any incomplete condition, unresolved provenance mismatch, or reference access
before prediction freeze makes the answer comparison incomplete or invalid;
preserve the evidence and stop rather than filling missing values.

## Artifact and freeze order

Use a new ignored private run directory and never mutate the previous
agreement-stop bundle.

1. Freeze exploratory protocol, claim boundary, panel, code, model, runtime,
   prompt, and decision rules.
2. Verify and bind the three existing de novo annotation artifacts.
3. Freeze exploratory adjudication.
4. Freeze decoded-frame inventories and both condition plans.
5. Extract/hash frames and freeze provenance.
6. Freeze independent scene audits and any single repair round.
7. Freeze final requests and extraction-completeness decision.
8. Run/freeze compatibility predictions.
9. Run/freeze complete paired predictions.
10. Load references and freeze the blind grading packet/map.
11. Freeze two grade files and disagreement adjudication.
12. Unblind once and freeze the exploratory evaluation.
13. Write a sanitized report that separates extraction from answering.

Every artifact must refuse overwrite and have a verified SHA-256 sidecar or
freeze record that binds its upstream inputs.

## Failure handling

- Adjudicated fewer than eight labelable rows: report insufficient panel and do
  not run GPU inference.
- Cannot construct twelve unique actual decoded frames: record the affected row
  and stop before inference rather than duplicate frames.
- Scene audit fails after one deterministic repair: exclude that row from
  answer inference and report the exact aggregate denominator.
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

The user-facing result must state, separately, how many of ten scenes were
visibly extracted, how many rows entered paired inference, the paired grade
transitions, runtime, evidence validity, and what remains unknown. Stop and
share this visual result before any transcript work.
