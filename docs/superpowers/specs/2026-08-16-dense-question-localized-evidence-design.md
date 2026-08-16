# Dense Question-Localized Evidence Diagnostic Design

## Objective

Test whether increasing visual sampling from 4 FPS to 12 FPS inside a short,
question-relevant video window recovers answer-critical facts that are absent
from the frozen question-blind scene map. This experiment tests local visual
evidence recovery. It does not test answer prompting, multiple candidate
answers, counting specialization, a shared router, or a global scene-map rerun.

## Decision and scope

Run a paired selected-mechanism diagnostic over at most eight brief
visual-action targets and four supported controls from the completed Issue #35
sufficiency audit. Every selected row receives the same short video clip in two
conditions:

- question-conditioned local video sampled at 4 FPS;
- the byte-identical local video sampled at 12 FPS.

The existing scene map is used only to identify the local temporal region and
provide coarse context. It is not treated as complete evidence. All 475 source
rows remain eligible, and prior inspection is retained as an exposure flag
rather than an exclusion.

The experiment is capped at 24 Gemini calls, with one attempt per arm and no
retry. It stops after evidence recovery. A later answer-generation experiment
requires a separate freeze and authorization even if this diagnostic passes.

## Evidence validity boundary

The source sufficiency audit is already label-aware. Therefore this diagnostic
is a selected, oracle-screened mechanism test, not a fresh-panel prevalence
estimate and not a deployable localization evaluation. Its valid claim is only
whether dense local sampling can expose missing brief visual facts under the
tested clips and provider condition.

Freeze the probe treatment before using references or label-aware audit fields
to form the panel. The inference runner must consume only the provider-safe
projection. References, direct predictions, sufficiency verdicts, previous
candidate answers, and scoring labels never enter provider requests.

## Panel formation

Start from the 17 rows adjudicated `insufficient_or_ambiguous` in
`manual-audit-corrected-v3.json`. Classify every row into exactly one recovery
modality:

- `brief_visual_action`;
- `speech_or_audio`;
- `onscreen_text`;
- `fine_spatial_detail`;
- `genuinely_absent_or_ambiguous`.

The classification is label-aware and must record its evidence and rationale.
Select up to the first eight `brief_visual_action` rows in frozen audit order.
If fewer than two credible visual targets exist, preserve a zero-call negative
feasibility result and do not contact the provider.

Select up to four `supported_control` rows from the same audit in frozen order.
Controls must contain a visually observable requested fact and admit a local
window under the same procedure. The minimum runnable panel is two targets and
two controls; the maximum is eight targets and four controls.

Record existing exposure flags for every row. Do not exclude previously
inspected rows.

## Local-window and media contract

For each selected row, freeze one short window from the existing question and
frozen map timestamps. Prefer a 4-second interval and permit 2–6 seconds when
required by clip boundaries or the mapped anchor event. The window may use the
label-aware source audit to screen mechanism suitability, but it may not use
the reference to choose different 4-FPS and 12-FPS content. Both arms use the
same source bytes, start time, end time, audio stream, resolution, and prompt.
Only `VideoMetadata.fps` changes.

Create one lossless or visually lossless local clip per row and record:

- source media identity and SHA-256;
- clip SHA-256;
- exact start and end times;
- decoded duration and stream metadata;
- source-map SHA-256 and cited map line numbers;
- target/control role and exposure flags.

The clip must be under the Gemini inline-data size limit. The runner uses the
direct Google API with one inline video per request, the text prompt after the
video part, and high media resolution. The provider supports custom FPS through
`VideoMetadata.fps`; the stored request projection must prove whether `4.0` or
`12.0` was requested.

## Frozen probe treatment

Use `gemini-3.1-pro-preview`, temperature `0.0`, thinking level `HIGH`, high
media resolution, and a maximum of 2,048 output tokens. The exact model name
must be revalidated against the already completed direct-Gemini runner before
authorization; a provider-side model change blocks the run rather than silently
substituting another model.

The prompt provides only the question, selected scene-map lines, the local clip,
and the clip's offset within the source video. It asks the model to return one
strict JSON object containing:

- `question_interpretation`: requested entity, visual fact, and temporal
  qualifier;
- `fact_observed`: boolean;
- `observed_fact`: concise description or an empty string;
- `evidence_timestamps`: zero or more clip-relative timestamps;
- `conflicts_or_uncertainty`: explicit ambiguity or missing evidence;
- `needs_more_evidence`: boolean.

The probe must not produce a competition answer or multiple candidate facts.
When the requested fact is not visibly supported, it must set
`fact_observed=false` and `needs_more_evidence=true` rather than guess.

Freeze and commit the prompt, schema, validator, runner hash, provider
configuration, call cap, decision rule, and source bindings before panel
selection.

## Execution and freezing

Order rows deterministically by audit order and arms as 4 FPS then 12 FPS.
Issue at most one request for each planned row-arm pair. Do not retry provider,
parse, schema, safety, or empty-response failures.

For every attempt, preserve request projection, raw response, metadata, parsed
response when valid, terminal state, usage, and hashes. Freeze all terminal
rows and predictions before loading references or performing the label-aware
recovery audit. Any incomplete pair remains in the denominator and blocks a
positive gate.

## Post-freeze audit and metrics

Inspect every target and control in both arms. A label-aware audit records:

- whether the reference-level requested visual fact is actually present in the
  clip;
- whether the model's observed fact is supported by the cited timestamps;
- whether the 4-FPS response recovers that fact;
- whether the 12-FPS response recovers that fact;
- whether 12 FPS adds a true recovery over 4 FPS;
- whether either arm invents or changes a control fact;
- whether the row or reference remains ambiguous.

Primary descriptive measurements are:

- target recovery count and rate by arm;
- paired incremental recovery count (`4 FPS false`, `12 FPS true`);
- dense-only supported-fact count;
- unsupported-observation count by arm;
- control preservation and regression counts;
- terminal completion denominator;
- results split by prior-exposure status.

Do not calculate METEOR, BERTScore, STS, or semantic composite because this
stage does not emit final answers.

## Decision rule

The diagnostic passes only if all conditions hold:

- every planned pair reaches a terminal state and every row is manually
  inspected;
- 12 FPS produces at least two supported incremental target recoveries over
  4 FPS;
- 12 FPS recovers at least half of the auditable brief-visual target rows;
- neither arm emits an unsupported observed fact;
- no supported control regresses under 12 FPS.

If the 4-FPS question-conditioned arm recovers nearly all facts that 12 FPS
recovers, interpret the bottleneck as question-blind map salience rather than
sampling density. If neither arm recovers the facts, stop higher-FPS work. If
12 FPS passes the gate, the next experiment may test answering over frozen
recovered evidence with matched controls, under a new authorization.

## Artifacts and reporting

Store complete private write-once artifacts under a new benchmark root in the
canonical shared dataset. The tracked sanitized bundle contains only counts,
hashes, schema descriptions, gate outcomes, costs, completion, and aggregate
results. It must not contain row IDs, questions, references, evidence text,
video frames, clips, raw responses, or predictions.

Extend the existing dated frontier report and Routing theme. Use the standard
evidence-validity and decision-disposition axes. Preserve a zero-call
feasibility failure, incomplete run, negative result, or failed promotion gate
rather than replacing it.

## Verification

Before calls, verify dataset linkage, treatment-before-selection ordering,
write-once behavior, source and media hashes, identical clip identity across
arms, exact FPS-only arm difference, label-free requests, deterministic row
order, exact call cap, no-retry behavior, and authorization binding.

After calls, verify complete raw/metadata/parsed/terminal artifact chains,
prediction freeze before audit, audit completeness, paired metric
recalculation, privacy scans, checksums, generated QMD/Markdown/HTML
consistency, focused and neighboring tests, Ruff, `git diff --check`, and an
independent review before any push or issue update.
