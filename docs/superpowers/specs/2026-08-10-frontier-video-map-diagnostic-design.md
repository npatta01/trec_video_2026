# Frontier Video Event/State Map Diagnostic Design

**Date:** 2026-08-10<br>
**Status:** Approved design; awaiting written-spec review<br>
**Primary theme:** Visual and temporal evidence<br>
**Related themes:** Routing, fusion, and answer selection; models and adaptation;
speech and transcript evidence; experiment integrity and evaluation

## Decision to make

Test whether a frontier native-audio/video model can, in one question-aware
pass, build a reliable timestamped account of a short video and use that
account to produce the best-supported top-1 answer. This is the cheapest useful
test of the proposed foundation:

> understand the video and its state transitions first, then use the question
> to identify the supporting event or interval.

The experiment is an architecture-feasibility diagnostic, not a population
benchmark. It must tell us whether general whole-video understanding is already
good enough to serve as the first stage of a proper VQA system, or whether the
next system needs explicit sliding windows, denser local inspection, OCR,
tracking, transcription, or other adaptive tools.

This design supersedes the ten-pass candidate-diversity architecture as the
next substantial experiment. Ranked top-10 answer construction remains a
downstream serialization problem and does not shape this test.

## Hypothesis and falsifier

**Hypothesis.** Given the original short video with native audio and the
question, Gemini 3.1 Pro Preview will usually produce a faithful chronological
event/state map whose timestamps point to the evidence required for the
answer. A top-1 answer grounded in that map will expose whether remaining
failures come from localization, perception, reasoning, or answer formatting.

**Falsifier.** Do not promote one-pass whole-video mapping as the system
foundation if fewer than 28 of the 35 evaluation videos have usable Gemini
maps, if any question category has fewer than three usable maps out of five,
or if answer-changing hallucinations and missing transitions are common enough
to fail those gates. A correct final answer does not rescue a bad map.

## Panel

Use 35 unique videos from the 475 locally available annotated VQA-2025 rows:
five from each of these seven question/evidence categories:

1. temporal order or transition;
2. counting or repeated action;
3. causal or outcome;
4. speech;
5. non-speech sound;
6. OCR or fine visual detail; and
7. other/global visual understanding.

The panel is selected before provider inference. Selection may use video ID,
duration, the question, and existing question-derived category labels, but not
reference answers, historical answer correctness, semantic metric values, or
provider outputs. A reviewer will manually verify category purity from the
question alone because the current automatic labels contain known noisy cases.
Within each category, choose short, typical, and long clips and cover at least
three distinct evidence topologies, such as a moment, a before/after change, a
repeated action, speech, sound, or visible text. Freeze exact IDs, categories,
durations, media hashes, questions, and the selection rationale in a panel
manifest.

Use two additional videos outside the 35 as operational preflight cases. They
exercise both providers but never contribute to the 35-video gate or quality
summary. All 37 source clips must be locally present and within the providers'
short-video limits. The six annotated rows without local media and all VTT or
search-only collections are out of scope.

## Conditions and model roles

Each provider independently receives the same original local video, its native
audio, the same question, and a provider-equivalent version of one frozen
instruction/schema contract.

- **Primary:** OpenRouter model `google/gemini-3.1-pro-preview`. Its map and
  top-1 answer determine the architecture gate.
- **Comparator:** Reka Chat `Reka Core`. It tests whether conclusions are
  provider/model-specific. Its output is scored and audited separately; it is
  never used to repair, select, vote on, or replace the Gemini answer.
- **Existing baseline:** reuse the already frozen Qwen3.5-4B 12-frame plus
  accepted-Whisper answers for matched panel IDs. Do not rerun that baseline.
  It is an answer-quality reference, not a map-quality control.

There is exactly one content inference per provider per video. The experiment
does not add extracted frames, Whisper, independent audio analysis, OCR tools,
or a second model pass. The absence of those tools is deliberate: failures
identify what a later adaptive system must add.

Use the original local video bytes, not a fresh YouTube retrieval. No lossy
transcode, clipping, frame-rate reduction, or audio extraction is allowed. If a
provider cannot ingest a source container, the row is an operational failure;
do not silently change its evidence.

## One-pass structured contract

The response must validate against one fixed JSON Schema and contain:

- provider/model/request identity and source duration;
- a concise interpretation of the question and the evidence it requires;
- a chronological `events` array covering the video, where each event has:
  - stable event ID;
  - start and end seconds;
  - one or more keyframe timestamps;
  - scene, actors/entities, actions, and object relations;
  - state before, change/action, and state after when a transition occurs;
  - observed visual evidence;
  - observed speech, non-speech sound, and visible text, each separated and
    marked absent when not observed;
  - relevance to the question; and
  - confidence and explicit gaps;
- up to three answer-support spans that cite event IDs and source times;
- one concise top-1 answer;
- a statement of why the cited evidence supports that answer; and
- credible reasons the answer could be wrong, without manufacturing ten
  candidate answers.

Events must be ordered, nonnegative, within source duration, and sufficiently
complete to reconstruct the important scenes and state transitions. Support
spans must reference declared events. The schema prohibits chain-of-thought;
it preserves conclusions, observations, uncertainties, and evidence links.

## Inference discipline

- Freeze prompts, JSON Schema, provider model IDs, reasoning setting,
  temperature, token ceiling, timeout, and source hashes before preflight.
- Use high reasoning and low/deterministic sampling where supported.
- Disable provider fallback to a different model.
- Preserve provider-reported model and route identity. For Gemini, the
  successful OpenRouter preflight route is pinned for the 35-row run rather
  than mixing Google routes within the panel.
- An invalid or truncated response is an operational failure. There is no
  prompt repair, JSON repair by another model, or content retry.
- A retry is allowed only for an identical request after an explicit transport
  or rate-limit failure. Preserve every attempt and possible duplicate charge.
- Complete and freeze all provider outputs before revealing references or
  calculating answer metrics.

## Map audit

The primary agent audits every Gemini and Reka map. The user is not required to
manually evaluate the panel.

Audit each map in two ordered views:

1. **Question-blind fidelity:** inspect the original video and the frozen map
   without the question or reference answer. Mask the map's question
   interpretation, relevance fields, support spans, answer, and uncertainty
   alternatives for this view. Check event coverage, ordering, entities,
   actions, state changes, OCR, and timestamp boundaries.
2. **Question-aware sufficiency:** reveal the question, but not the reference.
   Check whether the map contains and points to the evidence needed to answer,
   and whether the top-1 answer cites that evidence.

Only after both audits are frozen may the reference answer be revealed. Then
score the answer and assign one or more failure causes:

1. question misunderstood;
2. wrong evidence modality selected;
3. wrong temporal/local evidence retrieved;
4. visual or audio perception failure;
5. correct evidence but reasoning failure;
6. answer-generation or normalization failure; or
7. genuinely ambiguous or insufficient evidence.

The visual audit checks all material events, state transitions, visible text,
identities, and timestamps rather than sampling only incorrect answers. There
is no independent audio check in this experiment. Speech and non-speech audio
claims are explicitly reported as provider-observed and cannot receive
independently verified audio status.

## Usable-map rule and gate

A map is usable only when all of the following hold:

- it contains, or accurately points to, the evidence needed for the question;
- it preserves the material before/action/after relationship;
- it contains no answer-changing false claim;
- its relevant event boundaries are within approximately one to two seconds of
  the visible transition or evidence interval; and
- its answer cites actual support in the map.

The Gemini architecture gate passes only when at least 28 of 35 maps are usable
and every category has at least three usable maps out of five. Report stricter
four-of-five category results as a diagnostic, not as an additional gate.

Answer quality is secondary but mandatory. On the frozen 35 rows, report
normalized exact match, token F1, METEOR, BERTScore-F1, STS, and the mean of
METEOR/BERTScore-F1/STS for Gemini, Reka, and the matched Qwen baseline. Report
paired wins, losses, and ties. NDCG is explicitly degenerate for a single
answer and is recorded as such rather than used for this top-1 decision.

Interpret outcomes as follows:

- **Map gate passes and answers improve:** use the map as the foundation for
  question-driven adaptive evidence acquisition.
- **Map gate passes but answers fail:** keep the map; focus next on evidence-to-
  answer reasoning and normalization.
- **Map gate fails because evidence is omitted or mistimed:** add explicit
  temporal decomposition, sliding windows, or denser local inspection.
- **Map points correctly but perception fails:** add the measured specialist
  tool, such as OCR, higher resolution, tracking, or transcription.
- **Reka succeeds where Gemini fails:** treat model/provider choice as a live
  variable; do not infer that the architecture itself failed.
- **Both fail similarly:** general one-pass video understanding is not a
  sufficient foundation under the tested condition.

## Execution order and stopping rules

1. Freeze the 35-row panel, two preflight rows, prompts, schema, configuration,
   and label-isolation checks.
2. Run both providers on the two preflight rows.
3. Verify media ingestion, native audio/video handling, JSON validity,
   timestamps, hashes, latency, usage, and artifact completeness.
4. Project total cost from provider-reported preflight usage. Stop for review if
   projected cost exceeds $15; never exceed $30 without new authorization.
5. Run the unchanged one-pass requests on all 35 rows, resumably and with an
   immutable completion ledger.
6. Freeze predictions and raw outputs.
7. Complete and freeze the two-stage map audit.
8. Reveal references, calculate answer metrics, assign failure categories, and
   write the condition-bound result.

The run stops immediately on credential leakage, reference leakage, source
media mismatch, provider model substitution, systematic missing audio/video,
or artifact corruption. Partial outputs remain evidence but do not satisfy the
completed 35-row gate.

## Cost boundary

The 475 local clips total 7,988.945 seconds and average 16.819 seconds. A
duration-balanced 35-row panel plus two typical preflight videos is expected to
contain about 10.4 minutes of media per provider.

At current list prices:

- Gemini 3.1 Pro Preview through OpenRouter costs $2 per million input tokens
  and $12 per million output tokens including reasoning. At roughly 300 video
  tokens per second, the video contributes about 187,000 input tokens. Prompt
  and schema overhead should keep total input near 250,000–320,000 tokens
  across the 37 calls, or about $0.50–$0.65. Four to eight thousand billed
  output/reasoning tokens per call would cost about $1.78–$3.55. Expected
  Gemini cost: **$2.50–$4.50**.
- Reka Core lists video at $0.08 per minute, text input at $2 per million
  tokens, and output at $6 per million tokens. Media should cost about $0.83;
  expected structured outputs and text overhead bring the Reka estimate to
  roughly **$1.30–$2.00**.

Expected combined spend is therefore **about $4–$7**, with **$10** as a
practical contingency. The preflight review threshold is $15 and the absolute
authorization ceiling is $30. Record provider-reported tokens, media duration,
cost, and failed-attempt charges per request. Pricing sources are the
[Gemini/OpenRouter model page](https://openrouter.ai/google/gemini-3.1-pro-preview/providers),
[Gemini video-token guide](https://ai.google.dev/gemini-api/docs/video-understanding),
and [Reka API pricing](https://docs.reka.ai/pricing).

## Artifact and privacy contract

Keep three boundaries:

1. **Private pre-reference bundle:** frozen panel manifest, source hashes,
   prompts/schema/configuration, sanitized request metadata, raw provider
   responses, validated maps, top-1 predictions, usage/cost, attempts,
   timestamps, route identity, completion ledger, and checksums. Do not persist
   credentials or duplicate base64 media payloads.
2. **Private post-freeze audit bundle:** fidelity and sufficiency judgments,
   visual inspection references, revealed labels, row-level metrics, failure
   categories, paired comparisons, and checksums.
3. **Tracked sanitized projection:** protocol, provider/model identity,
   panel IDs/categories/durations, aggregate and per-category gate results,
   metric summaries, failure counts, cost totals, validity statement, and
   immutable evidence hashes. It excludes media, credentials, references, raw
   answers, transcripts, private paths, and private URLs.

Credentials are runtime environment variables only and must never appear in
commands, logs, process listings, source files, shell history, request
artifacts, or Git. Only the authorized 2025 labeled YouTube VQA development
media is sent to external providers. Licensed 2026 movie clips, VTT/search
collections, and unrelated data are excluded.

The completed result extends the canonical visual/temporal evidence record in
the project ledger and links to related themes rather than duplicating claims.
QMD remains the editable report source; generated Markdown and HTML are never
hand-edited.

## Implementation boundary after spec approval

Implementation will add only the smallest reusable pieces needed for this
diagnostic: a frozen panel builder, provider-neutral request/result contracts,
Gemini/OpenRouter and Reka adapters, schema validation, an immutable resumable
runner, freeze enforcement, audit records, scoring, privacy projection, and a
thin CLI. Offline tests use fake transports and fixture media; paid calls occur
only through the explicit preflight and run commands.
