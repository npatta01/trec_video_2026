# VQA Annotator, Boundary Selector, and All-Modality Answering Design

**Date:** 2026-08-12
**Status:** Approved in conversation; implementation not started

## Objective

Build a top-1 video question-answering system that first understands and
indexes what happens throughout a video, then uses the question to select the
right temporal boundaries, gathers all available modalities from those
boundaries, and asks a separate multimodal answerer for one supported answer.

The annotation is a temporal map and evidence control plane. It does not
replace the native video, and it does not produce the final answer.

## Decisions already made

- Optimize for the single best-supported answer. Ranked answers two through ten
  remain a downstream submission concern.
- Annotate the full native video and audio before answering.
- Preserve a natural-language scene description plus structured entities,
  actions, state changes, timestamps, audio, uncertainty, and evidence
  references.
- Use the question to decompose compound requests and select one or more
  relevant spans.
- Send all modalities from selected spans to the answerer by default, including
  native video and audio.
- Permit the complete native video when the question is global or the selector
  cannot bound the evidence safely.
- Use coarse-to-fine refinement. Default-resolution evidence localizes a detail;
  source-resolution crops, neighboring frames, dense sampling, tracking, OCR,
  or optional super-resolution refine it before the final answer is frozen.
- Super-resolution is auxiliary evidence and may not be the sole support for a
  textual or visual claim.
- Multiple inference paths are allowed when they reveal different evidence,
  not merely to manufacture answer diversity.

## Non-goals

- Do not make the scene annotation itself answer a known question.
- Do not force every question through ten prompts, seeds, or candidate passes.
- Do not use a global entity ontology. Entity identity is local to one video.
- Do not discard native video after converting it to text.
- Do not route each question to only one modality.
- Do not optimize this design around runtime amortization or the current
  one-question-per-video development layout.
- Do not implement final top-ten answer construction in this architecture.

## Architecture

```text
full native video/audio
  -> video annotator
  -> reusable scene and entity map

question + scene map
  -> question decomposer
  -> evidence obligations
  -> temporal boundary selector
  -> selected spans and neighboring context
  -> all-modality evidence acquisition and targeted refinement
  -> frozen evidence packet
  -> separate multimodal answerer
  -> one cited top-1 answer
```

The design has six modules with explicit seams.

### 1. Video annotator

The annotator consumes the complete native video and audio. It is blind to the
specific VQA question and to labels, but it is aware of the task's evidence
needs: actions, entities, state transitions, causality, counting and tracking,
visible text, speech, non-speech sound, and global context.

It emits a coarse video summary and a chronological set of semantic scene or
event records. Scene boundaries may overlap when an action or transition
crosses a cut; the representation must not require every event to fit inside
one disjoint shot.

The annotator records uncertainty and omitted or unreadable details instead of
guessing. It does not emit `top1_answer`, question relevance, or any equivalent
answer field.

### 2. Question decomposer

The decomposer converts a question into one or more atomic evidence
obligations. An obligation states what must be established, not what the answer
probably is.

For example, the question "What does the sign say, and what does the man do
after reading it?" creates at least these obligations:

1. identify and read the sign;
2. establish which man observes it;
3. identify the subsequent action;
4. preserve the temporal relation between the observation and action.

Obligations form a small dependency graph so that temporal words such as
`before`, `after`, `while`, `until`, and `because` remain explicit. Independent
obligations may be processed concurrently.

### 3. Temporal boundary selector

The selector consumes the question obligations and the reusable scene map. It
chooses the smallest spans that cover each obligation without cutting off the
required before, after, or cross-scene context.

The selector emits:

- selected primary spans;
- included predecessor or successor scenes;
- the obligations each span supports;
- the reason for each selection;
- unresolved evidence needs;
- a global-video flag when localization would be unsafe.

Overlapping selections are merged for delivery but retain their obligation
links. A global or broadly summarizing question receives the complete native
video and the scene map rather than an artificial narrow clip.

### 4. All-modality evidence acquisition and refinement

Every selected interval carries all available modalities:

- native video and native audio;
- scene descriptions and structured events;
- aligned transcript;
- sound-event observations;
- keyframes and source-resolution frames;
- visible-text observations and OCR outputs;
- entity tracks, actions, relations, and state changes;
- neighboring temporal context;
- provenance and uncertainty.

Question type controls additional processing, not modality removal:

- OCR/detail: source-resolution frames, targeted crops, adjacent frames,
  multi-frame agreement, OCR, then optional super-resolution.
- Counting/tracking: denser temporal samples and cross-frame entity tracks.
- Speech: aligned transcript plus native audio and visual speaker context.
- Sound: native audio plus source, manner, intensity, quality, and onset/offset
  descriptions when discernible.
- Temporal/causal: predecessor, focal, and successor events with explicit
  state transitions.
- Global: full native video, coarse timeline, and all relevant scene records.

Default-resolution processing may produce a provisional perception hypothesis,
but not a frozen final answer. Refinement occurs before final answer generation.
Super-resolved imagery must be supplied alongside its original source crop and
may support a claim only when the reading or observation is consistent with
original or neighboring source frames.

### 5. Frozen evidence packet

The packet is the only input boundary between evidence acquisition and final
answering. It contains the original question, obligation graph, selected scene
records, native media references or payloads, derived evidence, provenance,
and unresolved gaps.

The packet is immutable for a scored condition. Any changed boundary, added
crop, new OCR result, or extra native-video inspection creates a new evidence
condition rather than silently repairing the old one.

### 6. Multimodal answerer

The production answerer receives the frozen packet, including selected native
video/audio. It returns one direct answer only after addressing every required
obligation.

Its output contains:

- one top-1 answer;
- atomic answer claims;
- obligation IDs satisfied by each claim;
- supporting scene IDs, spans, and modalities;
- unresolved obligations;
- credible reasons the answer could be wrong.

Unsupported details must remain in uncertainty rather than enter the answer.
When an obligation remains unresolved, the answerer must say that evidence is
insufficient instead of inventing a completion.

## Data contracts

### Video memory

```text
video_id
duration_seconds
global_description
scenes[]
entity_tracks[]
provenance
```

### Scene record

```text
scene_id
span: {start_seconds, end_seconds}
description
entities[]: {entity_id, description, aliases, confidence}
actions[]: {subject_ids, predicate, object_ids, span, confidence}
relations[]
state_before
state_changes[]
state_after
speech[]: {span, speaker_id, transcript, confidence}
sound_events[]: {span, source, description, manner, intensity, confidence}
visible_text[]: {span, text, location, confidence}
keyframes[]
uncertainties[]
evidence_refs[]
```

`evidence_refs` bind semantic records back to native intervals, source frames,
audio spans, crops, and their hashes. The exact JSON schema belongs in the
implementation plan, but these semantic fields are required.

### Question plan

```text
question_id
original_question
obligations[]:
  obligation_id
  subquestion
  evidence_requirements[]
  required_precision
  temporal_dependencies[]
```

### Evidence request

```text
obligation_ids[]
selected_spans[]
include_neighbor_scenes
modalities: all
refinements[]
global_video_required
selection_reason
```

### Evidence packet

```text
question_plan
selected_annotations[]
native_media[]
derived_evidence[]
provenance
gaps[]
packet_sha256
```

### Grounded answer

```text
top1_answer
claims[]: {text, obligation_ids, scene_ids, spans, modalities}
unresolved_obligations[]
could_be_wrong_because[]
```

## Error handling and fallback behavior

- Annotation uncertainty remains visible to downstream stages.
- If a question obligation cannot be matched to a scene, the selector requests
  full-video or targeted native inspection; it does not choose the nearest
  scene merely to satisfy a schema.
- If boundaries are uncertain, include adjacent context or the complete video.
- If one modality is corrupt or contradictory, preserve both sources and mark
  the conflict. A derived transcript or OCR result cannot silently override
  native evidence.
- If enhancement changes an apparent reading, the answerer sees the original
  and enhanced evidence plus an explicit conflict marker.
- If one obligation fails, preserve the supported portions while marking the
  unsupported portion unresolved.
- Provider, schema, upload, and parsing failures retain the billable raw
  response and attempt provenance under the repository's existing runner
  safety rules.

## Experiment design

The primary research theme is Routing, Fusion, and Answer Selection, with
cross-links to Visual and Temporal Evidence, Native Audio, Speech and
Transcripts, and Models.

### Question

Does a full-video annotator plus question-driven boundary selection and
all-modality native clips produce better-supported top-1 answers than direct
full-video answering, and where do remaining failures occur?

### Hypothesis

Separating annotation, boundary selection, evidence refinement, and answering
will reduce unsupported answer claims while retaining label-critical detail.
Native clips will prevent the scene descriptions from becoming an irreversible
information bottleneck.

### Falsifiers

The architecture is not supported under the tested condition if one or more of
these dominate:

- complete annotations fail to identify answer-relevant scenes;
- selected boundaries omit evidence available in the scene map;
- selected native clips plus annotations perform materially worse than the
  same answerer with full native video plus annotations;
- the separate answerer continues to embellish or discard supported detail;
- full-video fallback is required for most non-global questions;
- the observed improvement comes only from answer style rather than evidence
  support or semantic correctness.

### Phase 0: existing two-row decoupling check

Reuse the two frozen Gemini canary maps and media; do not rerun annotation.
Remove answer-related fields, create evidence packets from the cited spans and
all local modalities, and run the frozen answerer. This tests whether separating
answering repairs the known unsupported temporal clause and underspecified
sound answer.

### Phase 1: seven-category canary

Use one previously unused frozen video from each of causal, counting,
OCR/detail, other/global, sound, speech, and temporal. References remain hidden
until every prediction is frozen.

The experiment compares meaningfully different evidence conditions:

1. Existing Qwen3.5-4B uniform-12 baseline, reused without inference.
2. Direct frontier answer from the question and complete native AV.
3. Complete native AV plus neutral annotations to the separate answerer.
4. Selector-chosen native clips plus annotations and all derived modalities to
   the same answerer. This is the proposed production path.
5. Annotations and derived text evidence without native clips to the same
   answerer. This is a diagnostic information-bottleneck control, not the
   proposed production path.

Conditions three through five hold the answer model, prompt, decoding, and
output contract fixed. A single neutral annotation pass supplies conditions
three through five. These arms expose different information and are not
candidate-diversity passes.

### Phase 2: directional 21-row gate

If all seven canaries complete with valid, inspectable artifacts, add two more
frozen unused rows per category for 21 total. Keep the remaining frozen rows as
a confirmation boundary. Do not adapt the schema or prompts after seeing the
21-row labels and then report the same rows as confirmation.

### Metrics and failure attribution

Report the existing official-style top-1 metrics: METEOR, BERTScore-F1, STS,
semantic composite, token F1, and normalized exact match. Single-answer NDCG is
degenerate and is not quality evidence.

Also report:

- obligation completion;
- answer-critical evidence coverage in the complete scene map;
- selector recall relative to the complete map;
- clause-level citation support;
- unsupported-detail rate;
- refinement and full-video fallback frequency;
- category-level completion, latency, and observed cost;
- a post-freeze strong-judge failure classification.

Every wrong or unsupported answer is classified as one or more of:

1. question decomposition failure;
2. annotation or perception failure;
3. boundary-selection failure;
4. refinement failure;
5. correct evidence but reasoning failure;
6. answer normalization failure;
7. genuinely ambiguous or insufficient evidence.

Automated strong-model review should perform the first-pass classification.
The timestamped media and derived evidence remain preserved for human
inspection when needed, but manual review is not required to complete the
initial scorecard.

### Decision logic

- If full AV plus annotations succeeds but selected clips fail, improve the
  selector or context policy rather than the answer model.
- If selected clips plus annotations succeed but annotations-only fail, native
  evidence is correctly carrying details that text compression loses.
- If annotations-only matches the native-clip arm, the memory may be sufficient
  for that category, but native evidence remains the audit and fallback layer.
- If all separate-answerer conditions fail with supported evidence present,
  strengthen or constrain the answerer.
- If direct full AV remains clearly better, retain it as the primary path while
  diagnosing whether annotation, selection, or decoupling caused the loss.
- Advance beyond 21 rows only after the production arm completes, its cited
  evidence is auditable, and gains are not confined to one category.

## Artifact requirements

Preserve, privately where necessary:

- frozen video panel and label-isolation declaration;
- annotator, selector, refiner, and answerer model/configuration identities;
- schemas, prompts, code revision, and hashes;
- scene maps, question obligations, selected boundaries, and selection reasons;
- source and delivered native media hashes;
- clips, frames, crops, audio, transcripts, OCR, enhancements, and provenance;
- every evidence packet and its hash;
- provider responses, attempt ledger, token usage, latency, and cost;
- frozen predictions before labels;
- post-freeze metrics and failure taxonomy;
- immutable manifest and `SHA256SUMS`.

Competition questions, references, predictions, licensed media, credentials,
and raw provider responses remain outside tracked public artifacts.

## Implementation boundary

Implementation should introduce deep modules around the contracts above rather
than extend the current one-pass frontier runner into a monolith. The existing
Gemini/OpenRouter delivery, raw-response preservation, artifact freezing, and
media-provenance code should be reused behind the new module interfaces.

The implementation plan must sequence local contract validation before any
paid inference and must not launch the seven-row canary until one synthetic or
existing frozen end-to-end packet passes schema, provenance, and accessibility
checks.
