# Submission-First Hierarchical Multi-Pass VQA Design

**Date:** 2026-08-10  
**Status:** Approved direction; optimize end-to-end quality before reverse ablation  
**Primary theme:** Routing, fusion, and answer selection  
**Related themes:** Models and adaptation; visual and temporal evidence;
speech and transcript evidence; experiment integrity and evaluation

## Objective

Build the strongest practical VideoQA submission pipeline first. The system
must understand the complete short video, timestamp scenes and transitions,
retrieve both global and dense local evidence, generate multiple independently
grounded answers, and elect one final answer. Reverse ablations follow only
after the complete stack demonstrates a material quality gain.

Only the 475 annotated VQA development videos with local media determine
quality. VTT captions and search collections are out of scope. The six VQA
annotations without local media remain explicit exclusions.

## Submission architecture

The pipeline is hierarchical but retains a direct path so a failed intermediate
stage cannot erase useful source evidence.

### Stage 1: whole-video understanding

`gemini-3.1-pro-preview` receives the complete native video with audio and the
question. It emits a schema-constrained, label-blind record containing:

- a chronological event timeline with source-relative start and end seconds;
- scene transitions and before/after state changes;
- up to three question-relevant support spans;
- evidence modality flags: visual, speech, non-speech audio, visible text;
- a concise global description; and
- uncertainty and evidence-completeness flags.

The stage does not emit the final submission answer. Its output is an
inspectable retrieval plan, not trusted evidence. Timestamps are clamped and
validated against the source duration. Invalid output gets one fixed schema
repair attempt; a second failure activates deterministic global evidence only.

### Stage 2: global and local evidence materialization

Materialize one immutable evidence packet per row:

- eight chronological global frames spanning the full duration;
- up to 24 local frames across the validated support spans, allocated by
  span duration with at least four frames per non-empty span;
- local sampling capped at four source frames per second and decoded with the
  existing nearest-unused actual-PTS allocator;
- accepted full Whisper transcript with timestamped segments, or explicit null;
- verbatim transcript segments overlapping support spans;
- source audio and bounded support-span audio clips;
- the Stage-1 timeline and support-span record; and
- contact sheets, requested timestamps, decoded PTS, hashes, fallback reasons,
  and source-media identity.

Temporal/counting questions receive the full 32-frame packet. OCR/detail rows
add deterministic high-resolution source frames or crops at the same selected
timestamps; this may not replace global coverage. Other rows may use the same
packet initially. Efficiency routing is a reverse-ablation concern.

### Stage 3: ten answer passes

Generate ten candidates with the same frontier model and a short-answer
contract. The ten passes are required by the AG ranked-answer submission and
are deliberately different evidence views rather than repeated sampling. The
model may reason internally; persisted outputs contain only the answer, support
citations, confidence, status, and request provenance.

1. **Direct AV concise:** complete native video/audio plus question.
2. **Direct AV alternate:** complete native video/audio, requiring a distinct
   supported interpretation when ambiguity exists.
3. **Global transcript:** chronological global frames plus accepted Whisper.
4. **Global/local timeline:** eight global frames, up to 24 local frames,
   accepted Whisper, and the timestamped event timeline.
5. **Question specialist:** the same union evidence with count, transition,
   OCR, speech, causal, or general short-answer instructions.
6. **Local-only:** dense support-span frames and overlapping transcript text.
7. **Audio-first:** global visual context plus full and support-span audio text,
   with instructions to reject irrelevant speech.
8. **Scene-boundary:** frames immediately before and after detected transitions
   plus global context.
9. **Detail/temporal route:** high-resolution frames for OCR/detail questions;
   dense chronological frames for counting/temporal questions; global/local
   fallback for all other rows.
10. **Conservative synthesis:** union evidence and the first nine answers,
    requesting one additional answer only when independently supportable and
    otherwise the strongest concise paraphrase not byte-identical to an
    existing candidate.

Every answer includes zero to three source-time support spans. Unsupported
specificity is discouraged. Candidate order and identity remain preserved.
After normalized deduplication, a fixed alternate-generation request may
backfill missing unique candidates twice. A row with fewer than ten valid
ranked answers after backfill cannot enter the top-10 run and uses the validated
baseline AG answer-list fallback rather than duplicate padding.

### Stage 4: reference-blind verification and election

Use a separate `gemini-3.1-pro-preview` invocation with randomized candidate
order. It receives the question, the union evidence packet, the frozen timeline,
and the ten candidate answers. It never receives the reference answer,
per-candidate metric scores, producer pass names, or historical outcomes.

The verifier returns:

- a complete ranking of candidate IDs;
- support and contradiction status for each candidate;
- a selected candidate ID; and
- a concise answer-normalization instruction restricted to whitespace and
  punctuation. It cannot synthesize a new semantic answer in the first run.

If verification fails, deterministic fallback ranks completed candidates by
support status, confidence, and the fixed pass precedence 4, 1, 5, 9, 3, 6, 8,
7, 2, 10. The existing validated Qwen3.5-4B ranked-answer output remains the
operational final fallback when the frontier pipeline cannot produce the
required answer count.

## Development and promotion sequence

### Operational canary

Run six frozen, label-free rows: two temporal/counting, one audio-critical, one
OCR/detail, one causal, and one general visual. The canary validates only:

- paid-model authentication and input support;
- complete native-video and image requests;
- schema validity and timestamp ranges;
- actual-PTS evidence extraction;
- all ten answer candidates and verifier completion;
- request/output hashes, latency, usage, and estimated cost; and
- absence of labels or references in every inference request.

All six rows must produce a final answer; at least five must complete without
the Qwen fallback; no request may contain a reference or annotation path. The
canary stops immediately on data-boundary, credential, or integrity failure.

### Frozen 60-row quality gate

After the operational canary, run the complete submission stack on 60 unique
development rows:

- **40 representative effectiveness rows:** five audio/text, six causal, seven
  counting, thirteen other, and nine temporal rows. Selection uses only frozen
  IDs, question-derived strata, source metadata, and deterministic SHA-256
  ordering.
- **20 failure-enriched diagnostic rows:** two prompt/schema, three
  reference/metric ambiguity, five transition/localization, four audio, three
  FPS/resolution, and three evidence-present reasoning candidates. These rows
  never enter population-effect estimates.

The primary metric is the existing semantic composite
`C = mean(METEOR, BERTScore-F1, STS)` on the 40 representative rows. Report
normalized exact match, token F1, METEOR, BERTScore-F1, STS, `C`, paired
win/loss/tie counts, strata, and seeded paired bootstrap intervals. NDCG is
reported as degenerate and excluded for one-answer effectiveness.

Promote the complete pipeline to the 475-row development panel when:

- final elected answers improve `C` by at least `+0.020` over the frozen
  Qwen3.5-4B 12-frame plus accepted-Whisper baseline;
- METEOR and BERTScore-F1 point deltas are positive;
- STS regresses by no more than `0.005`;
- at least 59 of 60 rows produce a frontier final answer and all 60 produce an
  answer including declared fallback; and
- no integrity, leakage, or artifact-preservation failure occurs.

Because 40 representative rows have wide uncertainty, the bootstrap interval
is reported but is not a hard promotion requirement. A point gain below
`+0.010` stops scaling. A gain between `+0.010` and `+0.020` expands the
representative gate to 120 rows before deciding.

### Full development and submission scale

If promoted, run the unchanged stack over all 475 development rows and compare
against the frozen baseline using failure-zero scoring. Freeze the architecture,
prompts, provider model ID, evidence budget, and verifier before generating any
unlabeled 2026 submission answers. Preserve the validator-passing Qwen fallback
so provider or rate-limit failure cannot prevent submission.

Generate a submission portfolio from the same frozen ranking: top-1, top-5,
and top-10 JSON files following the official 2026 `Q_ID`, `Video_ID`, and
`Answers[{Rank, Answer, Time}]` schema. Run the official validator on each and
retain only validator-passing files. This protects answer-quality metrics from
weak tail candidates while retaining a ranked-answer/NDCG run.

Reverse ablation occurs only after the full stack is frozen. Remove or replace
one mechanism at a time: verifier, specialist pass, timeline, local frames,
dense temporal frames, high-resolution detail, transcript, then direct AV.

## Deep failure analysis

Deep-inspect all 20 diagnostic rows plus every representative-row hard failure,
rescue, regression, large semantic movement, metric disagreement, transcript
problem, PTS/FPS anomaly, or suspected localization miss.

Use multi-label causes:

- source/reference unsupported or ambiguous;
- retrieval/localization miss;
- transition or temporal-density miss;
- resolution/OCR/small-object miss;
- transcript/native-audio absence, misalignment, or conflict;
- evidence present but visual interpretation/reasoning failed;
- verifier/election failure;
- prompt/schema/answer-format failure; and
- infrastructure or decoding failure.

Pinned Qwen3-Omni-30B-A3B-Thinking on the private dstack path adjudicates
evidence sufficiency and semantic equivalence after prediction freeze. Invalid
or uncertain adjudications remain explicitly uncertain. Human review is limited
to those disagreements or uncertainties, not a manual full-panel evaluation.
Every diagnosis points to its frames, audio, transcript, timestamps, timeline,
requests, responses, hashes, competing explanation, and falsifying next test.

## Model and provider boundary

The primary paid provider is Gemini Developer API paid tier using
`gemini-3.1-pro-preview`. Paid-tier requests are required because Google states
paid inputs are not used to improve its products. The implementation accepts
credentials only through the provider SDK's standard private authentication;
it never reads, logs, copies, or persists credential values.

No media is uploaded until the six-row canary begins. The user explicitly
authorized bounded paid external processing. Uploaded file objects must be
deleted after successful result preservation or terminal failure, and deletion
status must be recorded. If paid API authentication is unavailable, stop before
upload and use the existing private dstack Nemotron/Qwen path only as an
explicitly weaker operational fallback; do not relabel it as the frontier run.

## Artifact contract

Keep three boundaries:

1. **Private label-free run bundle:** immutable manifest, source hashes,
   prompts, requests, timelines, raw responses, predictions, selected and dense
   frames, audio, transcripts, contact sheets, usage, retries, deletions,
   terminal status, and `SHA256SUMS`.
2. **Private post-freeze audit bundle:** references, joined scores, adjudication,
   row-level failure labels, inspection HTML, and independent `SHA256SUMS`.
3. **Tracked sanitized projection:** protocol and model identity, IDs/strata,
   statuses, aggregate and per-stratum metrics, paired effects, failure counts,
   evidence hashes/markers, correction record, and checksums. It excludes raw
   media, transcripts, references, raw answers, private paths, and private URLs.

One versioned run manifest binds the panel digest, conditions, prompts, model
ID, SDK/runtime revision, source hashes, request hashes, evidence plan, output
hashes, prediction freeze, and scorer revision. Labels remain inaccessible until
all predictions and the freeze marker are complete.

The editable report is a dated QMD under `docs/experiments/`; routing owns the
primary claim. Related theme pages link to the evidence record without copying
it. Generated Markdown and HTML are rendered outputs and are not hand-edited.

## Implementation boundaries

Add focused modules rather than another monolithic experiment script:

- immutable run/cohort contracts;
- provider-neutral multimodal request and result types;
- Gemini adapter with injected transport for offline tests;
- timeline validation and deterministic fallback;
- evidence materialization over existing PTS/hash utilities;
- answer-pass construction;
- verifier/election and deterministic fallback;
- prediction freeze, privacy projection, scoring, and audit assembly; and
- a thin resumable CLI orchestrator.

All pure contracts and transformations are test-driven. Tests exercise real
filesystem artifacts and deterministic fake transports; they never call the
paid API. A live canary is an explicit integration command, not part of the
test suite.

## Advisor disposition

Three Luna investigations supplied the cohort, model/runtime, and artifact
design. A Sol/Fable/Gemini panel agreed that a strong frontier answerer and
whole-video path are warranted. Sol and Fable warned that native video at an
approximately one-frame-per-second default is not an evidence ceiling and that
joint timeline-plus-answer output confounds understanding with schema burden.
This design responds by using a separate timeline stage, retaining a direct AV
answer, adding explicit dense local evidence, and postponing reverse ablation
until the complete architecture is measured.
