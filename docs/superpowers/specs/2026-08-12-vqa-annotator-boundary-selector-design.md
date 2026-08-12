# VQA Annotator, Boundary Selector, and All-Modality Answering Design

**Date:** 2026-08-12
**Status:** Revised and approved in conversation; implementation not started

## Objective

Build a top-1 video question-answering system that first understands and
indexes what happens throughout a video, then uses the question to select the
right temporal boundaries, gathers all available modalities from those
boundaries, and asks a separate multimodal answerer for one supported answer.

The annotation is a temporal map and evidence control plane. It does not
replace the native video, and it does not produce the final answer.

The immediate work is a frontier-model capability experiment, not construction
of this complete architecture. Gemini must first demonstrate that it can create
a faithful question-blind map and use that map to answer on a small frozen
panel. The decomposer, selector, specialist refinements, and local-model system
remain a future design until that evidence exists.

## Decisions already made

- Optimize for the single best-supported answer. Ranked answers two through ten
  remain a downstream submission concern.
- Annotate the full native video and audio without the VQA question, reference,
  or any answer hypothesis.
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
- The first frontier screen uses exactly one annotation pass per video. The
  annotator interface may support later refinement passes, but only a measured
  omission, boundary error, or perception gap can justify testing one.
- Question-type routing and specialist evidence tools are deferred. The
  frontier screen supplies complete native video and audio to both answer
  conditions and tests the map before adding OCR, Whisper, tracking, dense
  frames, crops, or super-resolution.

## Non-goals

- Do not make the scene annotation itself answer a known question.
- Do not force every question through ten prompts, seeds, or candidate passes.
- Do not use a global entity ontology. Entity identity is local to one video.
- Do not discard native video after converting it to text.
- Do not route each question to only one modality.
- Do not optimize this design around runtime amortization or the current
  one-question-per-video development layout.
- Do not implement final top-ten answer construction in this architecture.
- Do not build the decomposer, selector, refinement pipeline, or local-model
  production path before the frontier capability gate is interpreted.

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
specific VQA question, reference, labels, and answer hypotheses, but it is
aware of the task's general evidence needs: actions, entities, state
transitions, causality, counting and tracking, visible text, speech,
non-speech sound, and global context. Question-blindness is a required property
of the annotation layer, not an experimental variable in the frontier screen.

It emits a coarse video summary and a chronological set of semantic scene or
event records. Scene boundaries may overlap when an action or transition
crosses a cut; the representation must not require every event to fit inside
one disjoint shot.

The annotator records uncertainty and omitted or unreadable details instead of
guessing. It does not emit `top1_answer`, question relevance, or any equivalent
answer field.

The interface permits a future annotation-refinement pass that receives the
frozen first map and native media but still no VQA question or reference. Phase
0 does not exercise that interface. A second pass becomes a separate condition
only after audited first-pass failures establish what it must recover.

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

In a future production system, question type may request additional processing
but may not remove a modality:

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

All processing in this subsection is deferred during the frontier screen. Both
answer conditions receive the complete native video and audio. Phase 0 tests
whether the map itself helps before introducing question-type routing or any
specialist evidence operation.

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

The obligation-linked grounded answer is the future production contract. Phase
0 uses the smaller matched frontier answer record below because the decomposer
is not implemented:

```text
top1_answer
claims[]:
  text
  supporting_spans[]
  supporting_modalities[]
  map_event_ids[]
could_be_wrong_because[]
```

The direct and map-guided conditions share this exact schema. For direct
answers, `map_event_ids` must be an empty array. For map-guided answers, every
material claim must cite at least one supporting stable map event ID.

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

Can Gemini 3.1 Pro Preview create a faithful question-blind event/state map of
a complete native-audio/video clip, and does providing that frozen map improve
the same model's evidence localization and top-1 answer over direct native-AV
answering?

### Hypothesis

One neutral annotation pass will preserve the important scenes, entities,
actions, state transitions, visible text, speech, sound, and timestamps needed
for varied VQA questions. Given the same complete native video/audio, the map
will help Gemini bind answer claims to evidence without removing label-critical
detail.

### Falsifiers

The frontier foundation is not supported under the tested condition if one or
more of these dominate the seven fresh rows:

- question-blind maps omit or materially mis-time answer-critical events;
- maps contain answer-changing false claims or collapse entity identity;
- map-guided answers cite map events that do not support their claims;
- map-guided answers continue to embellish or discard supported detail;
- direct native-AV answers are consistently stronger than map-guided answers;
- any apparent gain comes from different answer instructions rather than map
  availability;
- both Gemini answer conditions fail even when audited native evidence is
  available.

### Phase 0A: one spent-row neutral-annotation pilot

Use one of the two already-spent Gemini preflight videos, outside the 35 unused
evaluation rows. Run exactly one new question-blind annotation call. Do not
send the question or reference and do not rerun an answer condition.

The pilot validates only the neutral map contract and transport. Its map must:

- cover the provider-perceived duration and the important chronological
  scenes;
- assign stable event and within-video entity IDs;
- record event spans and key timestamps within source duration;
- separate visual action, visible text, speech, and non-speech sound;
- record uncertainty rather than guess;
- contain no question, question relevance, answer, or answer hypothesis;
- pass strict local schema validation without semantic repair.

The primary agent audits it against the complete native video/audio and grades
each material map item as confirmed, contradicted, or unverifiable. This pilot
does not estimate quality and does not count toward the seven-row result. If it
is too coarse, invalid, or uninspectable, revise and re-review the contract
before using fresh rows; do not consume evaluation rows to debug the prompt.

### Phase 0B: seven-category frontier mechanism screen

Deterministically select exactly one previously unused frozen video from each
of causal, counting, OCR/detail, other/global, sound, speech, and temporal.
Record IDs, categories, durations, selection rule, source hashes, and matched
Qwen baseline availability before any provider call. References remain hidden
until all predictions and pre-reference audits are frozen.

For each video, run these three calls with `google/gemini-3.1-pro-preview`:

1. **Question-blind annotation:** complete native video/audio without the VQA
   question or reference produces one neutral map. There is no refinement pass.
2. **Direct answer control:** question plus complete native video/audio produces
   one top-1 answer with atomic claims and timestamped supporting spans.
3. **Map-guided answer:** the same question, complete native video/audio, and
   frozen neutral map produce one top-1 answer under the identical answer
   contract. Claims must additionally cite stable map event IDs.

Conditions two and three use the same model revision, native media, answer
instruction, output schema, reasoning setting, temperature, and token ceiling.
The direct arm leaves the shared `map_event_ids` field empty; the guided arm
must populate it. Map availability and that required map-event binding are the
only intended differences. The complete native video/audio remains available
in both arms; the experiment does not replace it with selected clips.

Reuse the existing matched Qwen3.5-4B answer as a no-inference reference. Do not
run question-type routing, OCR, Whisper, independent audio checking, tracking,
dense-frame inspection, crops, super-resolution, specialist models, candidate
diversity, or a second annotation pass.

### Audit and freeze order

For every fresh row:

1. Freeze provider responses, parsed records, usage, cost, and hashes.
2. Audit the neutral map against the native video/audio without the question or
   reference. Grade event coverage, ordering, identity, action, state change,
   text, sound/speech observations, and boundaries as confirmed, contradicted,
   or unverifiable.
3. Reveal the question but not the reference. Audit whether the map contains
   the evidence required by the question and whether each direct and map-guided
   answer claim is supported by its cited native span. For the guided arm,
   verify the cited map event as well.
4. Freeze those audits and predictions before joining reference answers.
5. Score both Gemini arms and the exact matched existing Qwen baseline.

The answer contract permits no semantic retry. One identical retry is allowed
only after an explicit transport, rate-limit, or empty/truncated response
failure. Every attempt and possible duplicate charge remains in the ledger.

### Cost boundary

The prior two Gemini content calls averaged USD 0.031865 each. At that observed
rate, the one-call pilot plus 21 fresh calls would cost about USD 0.70. Budget
USD 1–3 for output and media variation, project remaining cost after the pilot
and first complete fresh row, and stop before cumulative new paid inference
reaches the user-authorized hard ceiling of USD 25.

### Later confirmation boundary

Seven rows are a mechanism screen, not a population benchmark and not proof
that Gemini is a permanent model winner. A favorable result promotes Gemini to
foundation candidate for a separately reviewed larger gate. It does not
authorize the remaining 28 frozen rows, all 475 labeled rows, the in-house
pipeline, or competition-test inference.

### Metrics and failure attribution

Report the existing official-style top-1 metrics: METEOR, BERTScore-F1, STS,
semantic composite, token F1, and normalized exact match. Single-answer NDCG is
degenerate and is not quality evidence.

Also report:

- neutral-map event fidelity and answer-critical evidence coverage;
- answer-claim native-span support;
- guided-answer map-event citation validity;
- clause-level citation support;
- unsupported-detail rate;
- direct-versus-guided row-level wins, losses, and ties;
- category-level completion, latency, and observed cost;
- a post-freeze strong-judge failure classification.

Every wrong or unsupported answer is classified as one or more of:

1. neutral annotation coverage, timing, or perception failure;
2. question or evidence-requirement misunderstanding;
3. answer evidence-localization or map-use failure;
4. native visual/audio perception failure;
5. correct evidence but reasoning failure;
6. answer normalization failure;
7. genuinely ambiguous or insufficient evidence.

Automated strong-model review may assist the post-freeze classification, but it
does not replace the primary agent's required native-media map and citation
audit. The user is not required to evaluate manually.

### Decision logic

- Faithful maps plus stronger map-guided answers promote Gemini and the map
  architecture to a larger confirmation gate.
- Faithful maps plus equally strong direct and guided answers also promote both
  conditions to the next gate; the map is retained provisionally for grounding
  and auditability, not claimed to improve answer quality.
- Unfaithful maps plus tied answers require repairing neutral annotation before
  any larger gate.
- Faithful maps plus weak guided answers isolate an evidence-to-answer binding
  or interface failure; do not add annotation passes to solve it.
- Direct answers consistently stronger than guided answers retain direct
  Gemini as the near-term candidate and trigger map-interface diagnosis.
- Missing or mis-timed map events justify considering one targeted,
  question-blind refinement-pass experiment on the observed failure mechanism.
- Both Gemini answer conditions failing despite available evidence block Gemini
  as the sole foundation until a stronger answer strategy or frontier
  comparator is reviewed.

## Artifact requirements

Preserve, privately where necessary:

- frozen video panel and label-isolation declaration;
- annotator and answerer model/configuration identities for Phase 0;
- schemas, prompts, code revision, and hashes;
- question-blind scene maps, stable event IDs, cited boundaries, and audit
  judgments;
- source and delivered native media hashes;
- native media, audit frames/audio when extracted, and provenance;
- every provider request/response record and its hash;
- provider responses, attempt ledger, token usage, latency, and cost;
- frozen predictions before labels;
- post-freeze metrics and failure taxonomy;
- immutable manifest and `SHA256SUMS`.

Competition questions, references, predictions, licensed media, credentials,
and raw provider responses remain outside tracked public artifacts.

## Implementation boundary

The next implementation plan covers only the frontier Phase 0 contract,
artifacts, one spent-row neutral-annotation pilot, and seven-row mechanism
screen. It must not implement the future decomposer, selector, specialist
refinements, local model pipeline, or top-ten submission layer.

Reuse the existing Gemini/OpenRouter delivery, raw-response preservation,
artifact freezing, and media-provenance code behind small annotation and answer
contracts. Validate schemas and artifact accessibility locally before the one
paid pilot. Do not select or call the seven fresh rows until the pilot map and
its required audit are complete and the projected total remains below USD 25.
