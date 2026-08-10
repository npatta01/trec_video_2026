# Qwen3.5-4B Model-Assisted Oracle Evidence Ceiling Experiment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Production behavior follows `superpowers:test-driven-development`; every task
> uses checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the current Qwen3.5-4B VQA failure is caused by
missing question-relevant visual/audio evidence or by failure to answer when
that evidence is supplied.

**Architecture:** Run a frozen, fully automatic strong-model evidence selector
over all 475 locally available VQA 2025 rows, then use a small human-verified
audit subset to estimate how far that selector is from genuinely ideal
evidence. The selector produces only timestamps, modality enums, and
confidence/abstain values; it never produces captions, rationales, or answers.
Use the selected intervals to construct fixed-budget visual and verbatim
transcript conditions, while reusing the verified global uniform-12 baseline
where byte identity permits. Keep selector recall, strong-model packet
answerability, human-audit sufficiency, oracle-evidence answer quality, and
end-to-end quality as separate artifacts. Only after evidence and predictions
freeze may references be loaded for blind semantic grading and automatic
scoring. A full-panel automatic selector result is a **strong-model-assisted
selector benchmark**, not a human-oracle or ceiling result. Only the
independently human-verified audit subset can support a conditional empirical
oracle-ceiling claim.

**Tech Stack:** Python 3.12, pytest, stdlib JSON/hash/dataclasses/statistics,
PyAV, ffmpeg/ffprobe, Whisper large-v3 timestamped transcripts, Qwen3.5-4B,
rootless dstack with a pinned NVIDIA CUDA L40S runtime, BERTScore using the
existing pinned `FacebookAI/roberta-large` revision as a frozen post-hoc
secondary metric, and Quarto project-status reporting.

## Global Constraints

- Primary theme: **Visual and Temporal Evidence**; related themes: Routing and
  Speech and Transcript Evidence.
- Panel: 481 VQA 2025 training annotations; run only the 475 rows with local
  media. The six rows without local media are excluded and remain in the
  denominator record.
- Never copy, move, delete, stage, or commit dataset contents, private media,
  questions, answers, predictions, transcripts, frames, or raw logs.
- Store private artifacts only under the ignored run directory
  `data/vqa/benchmarks/qwen35-oracle-evidence-ceiling-20260803/`.
- Freeze model revision, runtime image, prompt, decoding, preprocessing,
  panel order, media hashes, and frame hashes before inference.
- Only the matched c2 source condition IDs
  `qwen35-4b-12f-frames-only` and
  `qwen35-4b-12f-whisper-large-v3` may be reused; `uniform12-none` and
  `uniform12-accepted-whisper-policy` are analysis aliases only. Reuse requires a canonical
  model-input digest binding frame bytes, requested timestamp text, actual-PTS
  provenance, transcript bytes or null, chat-template serialization, processor
  revision, runtime image, model revision, decoding, and condition identity.
  If any digest differs, rerun that control; the standalone c1 baseline is not
  interchangeable with c2.
- Primary oracle visual budget is exactly 12 decoded, uncropped, chronological
  frames. Run oracle-24 only for rows independently judged unable to represent
  the required evidence with 12 frames; do not make global-24 the baseline.
- The current Qwen path receives timestamped transcript text, not native audio.
  Transcript-oracle claims exclude non-speech sound, music, prosody, and impact
  questions unless a separate native-audio condition is authorized.
- Evidence annotators see only the frozen question-plus-media packet. They may
  not see references, predictions, grades, current frame timestamps, contact
  sheets, prior failure audits, or condition identities.
- The primary selector is question-conditioned but reference-, prediction-,
  and answer-denied. For visual selection it receives muted video; for speech
  selection it receives only the frozen timestamped ASR bytes. Its output
  schema forbids captions, summaries, rationales, answer text, and generated
  transcript text. A generated description or answer is never passed to Qwen.
- The primary full-panel selector condition is automatic and uncorrected.
  Human corrections are restricted to the calibration/audit subset and are
  reported as a separate human-assisted ceiling condition.
- Oracle intervals may control offline frame/audio construction but must never
  enter model prompts as interval names, rationales, captions, answer hints, or
  confidence labels.
- Freeze all predictions before loading references or grading. Human-audit arms
  are conditional empirical ceiling diagnostics; automatic selector arms are
  selector benchmarks, never deployable selectors or competition baselines.
- Pin the BERTScore encoder to the existing cached
  `FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`; if the
  frozen runtime cannot provide it without an unapproved download, record
  BERTScore as unavailable rather than changing the inference environment.
- No dstack/GPU launch, external publication, or report-status update is part
  of this plan approval; those actions require a separate execution decision.
- No paid model call is authorized by this plan alone. Before sending any
  video, frames, audio, question, or annotations to an external provider,
  record the provider/model, maximum spend, exact payload, retention/training
  policy, and dataset-rights check, then obtain an explicit execution decision.
  Local or already-authorized dstack inference may be evaluated first.
- The checkout is currently detached/dirty and this plan is untracked. Before
  any production annotation or inference, freeze the controller sources and
  plan from a clean revision or bind every source hash in an immutable freeze
  record.

---

## Scientific contract

### Questions

1. Does uniform-12 fail because it misses the answer-bearing visual moment?
2. Does the full Whisper transcript fail because relevant speech is not
   selected or because ASR is wrong/noisy?
3. Can the same Qwen3.5-4B answer correctly when independently audited,
   question-relevant evidence is supplied?
4. Does 12-frame representation capacity suffice, or do some rows require 24
   oracle-selected frames or a non-still representation?

### Conditions and estimands

Do not execute a flat 2x4 factorial. Define modality-specific paired
contrasts, retain all 475 candidate rows in the denominator, and report the
analysis population for every contrast. Rows that are ambiguous, not
answerable, non-still-representable, or native-audio-required remain visible
as exclusions or not-applicable cells rather than silently disappearing.

| Estimand | Population | Primary contrast |
|---|---|---|
| Human-audit empirical ceiling | Untouched 48-row confirmatory audit sample; calibration, flagged, and exception rows are separate | Derived `oracle12-ideal-evidence` policy versus c2 `uniform12-accepted-whisper-policy` |
| Model-assisted visual selection | All 475 candidate rows under an explicit fallback policy; also report usable-selector coverage separately | c2 `uniform12-accepted-whisper-policy` versus `vlm12f-accepted-whisper-policy` |
| Model-assisted transcript selection | All 475 rows under an explicit fallback policy; report the fixed full-clip/selected-span common-nonnull subset separately | `uniform12-asrsel-fallback` versus c2 `uniform12-accepted-whisper-policy`, with a coverage-conditional `uniform12-asrsel-required-span` contrast |
| Model-assisted composite evidence | All 475 candidate rows under explicit visual and transcript fallback policies | c2 `uniform12-accepted-whisper-policy` versus `vlm12f-asrself-fallback` |
| Visual selection | Visual-only rows and mixed rows with common text held fixed | c2 `uniform12-accepted-whisper-policy` versus `oracle12-accepted-whisper-policy` |
| Transcript selection | Speech/mixed rows; include null ASR rows in the full population and report the predeclared common-nonnull set separately | `oracle12-whisper-fullclip` versus `oracle12-whisper-required-span`, with a direct `oracle12-whisper-required-span` minus `oracle12-sham-required-span` contrast |
| ASR/translation pipeline | Speech/mixed rows where both human and Whisper evidence are defined; report empty/null rows separately | `oracle12-whisper-required-span` versus `oracle12-human-required-span` |
| Frame capacity | Rows whose O12 packet is independently frozen as insufficient before inference; population labels remain visible | O12 versus nested O24, with O12 frames included unchanged in O24 |

The exact new condition IDs and c2 aliases are:

- c2 `qwen35-4b-12f-frames-only` (analysis alias `uniform12-none`): existing
  frames-only control;
- c2 `qwen35-4b-12f-whisper-large-v3` (analysis alias
  `uniform12-accepted-whisper-policy`): existing 424-text/51-null
  acceptance-policy treatment, retained as an operational comparator;
- `oracle12-none`: Oracle-12 frames with no transcript;
- `oracle12-accepted-whisper-policy`: Oracle-12 frames with the exact c2
  per-row Whisper bytes/nulls;
- `oracle12-whisper-fullclip`: Oracle-12 frames with unfiltered frozen Whisper
  output, including nonempty output that the c2 quality gate rejected and true
  empty output as null;
- `oracle12-whisper-required-span`: Oracle-12 frames with the frozen Whisper
  translation restricted to the required speech interval;
- `oracle12-human-required-span`: Oracle-12 frames with an independently
  reviewed literal English transcript of the required speech interval;
- `oracle12-human-fullclip`: Oracle-12 frames with an independently reviewed
  full-clip literal English transcript for span-capacity exceptions;
- `oracle12-sham-required-span`: Oracle-12 frames with a matched,
  time-shifted non-required Whisper span, speech-only and secondary.
- `vlm12-accepted-whisper-policy`: automatically VLM-selected visual frames
  with the exact c2 accepted-Whisper bytes/nulls, coverage-conditional on a
  valid selector output; the selector sees muted video and the question only.
- `vlm12f-accepted-whisper-policy`: the all-475 visual-selection policy. Use
  the valid `vlm12` packet when available; on abstain, malformed output,
  non-still/native-video indication, or failed construction, use the exact c2
  Uniform-12 packet and record the fallback reason. The `f` suffix is part of
  the condition identity, not an analysis alias.
- `uniform12-asrsel-fullclip`: exact c2 Uniform-12 frames with the full frozen
  Whisper output and no speech-span selection.
- `uniform12-asrsel-required-span`: exact c2 Uniform-12 frames with a
  question-conditioned selector's verbatim Whisper span, using the frozen raw
  ASR bytes and timestamps only, coverage-conditional on a valid span.
- `uniform12-asrsel-fallback`: the all-475 transcript-selection policy. Use the
  selected verbatim span when valid; on selector abstain, malformed output,
  non-speech/no-span, or empty ASR, use the exact c2 accepted-Whisper
  bytes/null and record the fallback reason.
- `vlm12f-asrself-fallback`: the all-475 composite policy: apply the visual
  `vlm12f` fallback and transcript `asrsel-fallback` policy independently.

The `vlm12*` and `asrsel*` conditions are automatic selector conditions, not
human-oracle conditions. Abstentions remain visible in the full 475-row report
and are handled only by the predeclared fallback policy above, never by human
choices. The automatic transcript population is defined from frozen ASR
availability and selector output validity, not from human-adjudicated modality:
`fullclip_nonempty && valid_selected_span` is the coverage-conditional set;
all other rows use the recorded fallback. `asrsel` is a transcript-span
selector, not an audio-understanding condition: the current Qwen request still
contains text, so these conditions cannot establish a ceiling for non-speech
sound, music, prosody, or impact audio.

`oracle12-ideal-evidence` is a derived per-row policy used only on the untouched
48-row confirmatory audit sample, not an all-475 estimand or inference ID: it
selects `oracle12-human-required-span` when the human audit independently finds
speech evidence, `oracle12-human-fullclip` for a declared span-capacity
exception, and `oracle12-none` for visual-only rows. Calibration, flagged, and
native-audio-required rows are reported separately. ASR-empty rows remain null
in the full-population transcript analyses; only the fixed fullclip-nonempty /
valid-span subset is used for the clean automatic Whisper span contrast.

The transcript-selection common-nonnull set is frozen as rows whose adjudicated
modality is speech or mixed, whose full-clip Whisper output is nonempty, and
whose required-span Whisper output from the same frozen raw transcript is also
nonempty. The ASR/translation common set is the subset of those rows with a
reviewed human required-span transcript. The full 475-row reports retain rows
outside these sets as null/not-applicable; no contrast silently changes its
population.

Conditional O24 IDs are `oracle24-none`,
`oracle24-accepted-whisper-policy`, `oracle24-whisper-fullclip`,
`oracle24-whisper-required-span`, `oracle24-human-required-span`, and
`oracle24-human-fullclip`; each is permitted only for the corresponding
pre-inference O12 capacity trigger. The oracle Whisper span uses the same
frozen Whisper translation output as the full-clip arm, restricted to
adjudicated speech intervals. Human transcripts are literal English
transcriptions/translations, never summaries or manually composed answers.

### Model-assisted evidence localization and human audit

The primary locator is automatic, not a 475-row human annotation exercise.
For visual selection, primary selector `S_v` receives the question plus muted
full video and returns only numeric intervals, still-frame representability,
modality enums, and confidence/abstain values. For speech selection, primary
selector `S_t` receives the question plus frozen timestamped ASR bytes and
returns only verbatim segment IDs/timestamps, confidence/abstain, and no
rewritten text. An independently implemented checker for each modality,
`S_v^check` and `S_t^check`, receives the same reference-denied inputs but no
primary output. Selector disagreement means either checker abstains while the
primary returns a packet, the primary abstains while the checker returns one,
malformed output, modality/still-frame conflict, or no matched interval set
with temporal IoU at least 0.50 and required-interval recall at least 0.80.
Neither selector sees references, predictions, grades, existing timestamps,
prior failure audits, or condition identities. Generated captions, summaries,
rationales, and answers are forbidden by schema and are never passed to Qwen.

First run a 20-row human-reviewed calibration pilot, stratified by a frozen,
question-only classifier with the labels `speech`, `sound`, `temporal`,
`counting`, `spatial`, `attribute`, `main_topic`, and `other`. The classifier
uses only normalized question text and a versioned rule table; it never reads
answers, media, predictions, or prior failure labels. Sample the pilot with
frozen seed `20260803-pilot`, then exclude those row IDs permanently from the
confirmatory audit. Freeze selector prompts, model revisions, output schema,
confidence thresholds, and the rubric before the full-panel selector run.
Pilot rows are calibration evidence, not an independent prevalence or gain
estimate.

After the pilot is frozen, sample a disjoint proportional-stratified 48-row
confirmatory audit from the remaining 455 rows with seed `20260803-audit` and
deterministic largest-remainder allocation across the eight question-only
strata. Preserve stratum weights `N_h/n_h` for audit estimates. Run the frozen
selectors and their checkers over all 475 candidate rows. Independently audit
the 48-row sample plus every row in the unique union of selector disagreement,
abstention, low-confidence, non-still, suspected audio-only, and malformed
output flags. Flagged-row results are a separate diagnostic census, never
pooled with the random-sample prevalence or gain estimate. The primary
`vlm12*` arm is never human-corrected. Human corrections create a separately
labeled `human-audit-ceiling` subset only.

Two independent human reviewers annotate the calibration and random-audit
rows from question plus full media only, without seeing selector outputs. A
third reviewer adjudicates disagreements and flagged rows after the two files
freeze. Each reviewed row records:

- eligibility: `answerable`, `not_answerable`, `ambiguous`, or
  `insufficient_media`;
- required modality: visual-static, visual-temporal, speech, non-speech, or
  mixed;
- one or more alternative sufficient visual interval sets, with conjunctive
  intervals for anchors, transitions, ordering, counts, or co-occurrence;
- up to two disjoint speech intervals, total span budget 15 seconds and 80
  words unless a capacity exception is recorded;
- whether the answer is representable by still frames and whether it requires
  native audio;
- structured reason codes for each exclusion or uncertainty;
- reviewer identity, rubric revision, packet hash, start/end timestamps, and
  annotation hash, plus a prior-label-exposure attestation.

The audit-quality gate is frozen before calibration: eligibility agreement
must be at least 90% on the reviewed sample; modality agreement must be at
least 80% among jointly answerable reviewed rows; structurally matched
interval sets must have median temporal IoU at least 0.50 and at least 80% of
matched rows must have IoU at least 0.50; all reviewed disagreements must be
adjudicated. If selector flags exceed 20% of the 475 rows or the audit gate
fails, stop and recalibrate the automatic selector rather than silently
turning the experiment into 475 human reviews. The raw selector result may be
reported as exploratory, but it cannot support a ceiling claim until the audit
passes. There is no hidden minimum-n stop: every population remains reported,
while any mechanism label for a population with fewer than 30 analyzable rows
is explicitly descriptive-only.

The selector-validity gate is separate from reviewer reliability: on the
weighted 48-row confirmatory audit, at least 80% of eligible rows must have a
primary selected interval set matching an adjudicated sufficient set at
temporal IoU at least 0.50 and required-interval recall at least 0.80. Report
the one-sided 95% Wilson lower bound for that match rate and require it to be
at least 0.60 to pass. If the validity gate fails, the selector result remains
exploratory; do not call it a ceiling or choose a deployable selector. A
revised selector requires a new version and a new untouched confirmatory audit
sample; do not recalibrate against the same 48 rows.

The evidence locator, selected-audio transcriber, packet-only sufficiency
auditor, transcript reviewer, and answer grader remain separate roles. Human
oracle claims are restricted to the untouched confirmatory audit and separately
reported flagged/exception subsets; the 475-row automatic condition is a
strong-model-assisted selector benchmark.

### Visual frame construction

For each adjudicated oracle interval set:

- decode actual source frames with PTS, frame index, time base, media hash, and
  frame hash;
- allocate all 12 slots across the shortest sufficient interval set, preserving
  every required interval and chronological order; require 12 distinct decoded
  frame identities/actual PTS, while allowing identical pixel hashes in static
  scenes;
- fill unused slots deterministically from uniform-12 only when the sufficient
  evidence needs fewer than 12 unique frames;
- bind requested timestamps to nearest unused actual decoded PTS, preferring
  the earlier frame on ties;
- record requested time, actual PTS, frame index, slot type, frame hash, and
  extraction identity;
- freeze the O12 packet and its sufficiency audit before deciding whether a row
  receives O24;
- make O24 a strict chronological superset of O12: the exact 12 O12 decoded
  frames, pixels, preprocessing, prompt, and transcript remain unchanged, and
  12 additional decoded frames are inserted deterministically;
- run oracle-24 only when the frozen O12 audit says O12 cannot preserve the
  required relation; classify true greater-than-12-frame need, failed O12
  construction, and non-still/native-video need separately.

### Packet sufficiency audit

Construct a reference- and prediction-denied Uniform-12 packet from the exact
c2 control frame bytes, a matched automatically selected packet, and a matched
human-audit packet where available. An independent packet auditor `A`, from a
different model family than the selector and target, answers the exact packet
or records `insufficient` before references or model predictions are exposed.
The auditor receives no selector rationale or condition identity. It also sees
question-only and sham/mismatched-packet controls, so a correct answer without
the packet cannot establish packet sufficiency. On the human-audit subset,
two packet-only human reviewers independently answer the exact packet and a
third adjudicates disagreements. Freeze `uniform12_packet_answer`,
`vlm12_packet_answer`, and `human_audit_packet_answer` with their provenance.
Record interval recall separately from packet answerability: interval contact
alone cannot prove that a frame packet is sufficient, and a strong-model
answer from the packet cannot prove that a human or automatic selector would
find it.

### Transcript construction

Whisper spans are selected from frozen timestamped output using either the
automatic transcript selector or the independently adjudicated human-audit
intervals. Human transcribers are used only for the calibration/audit subset
and declared span-capacity exceptions. They receive only selected audio and
timestamps; they do not receive the question, references, predictions, grades,
or locator rationale. A second question-blind transcript reviewer
independently checks every human transcript/translation against the same
selected audio and timestamps before it is admitted to an inference request.
Their target is literal English translation with the same timestamp rendering
and line formatting as Whisper's current `task="translate"` output.

The 15-second/80-word/two-span limit is a compact-capacity arm, not an
unannounced definition of ideal audio. If the adjudicated required speech
exceeds that limit, record a `span-capacity-exception` and run a full-clip
verified-transcript capacity condition for that row. For rows requiring
non-speech evidence, mark transcript conditions not applicable rather than
treating an absent transcript as negative evidence.

### Scoring and grading

After prediction freeze:

- create an opaque, condition-blind grading packet;
- have two condition-blind strong text judges assign `C` / `P` / `W` and
  confidence/abstain to every unique answer; use different model families or
  genuinely independent providers rather than two calls to the same model;
- send all judge disagreements, abstentions, and a frozen random spot sample
  to human adjudication; keep a separate human audit of the 20-row calibration
  set and 48-row video audit, including any answer regressions;
- declare dual-model semantic grading primary, not full-panel human grading;
- if human disagreement on binary `C` versus not-`C` exceeds 5% of the
  human-reviewed judge items, or the absolute human-disagreement-rate
  difference between any two conditions exceeds 5 percentage points, stop and
  recalibrate the judge prompts using a new calibration sample before
  interpreting full-panel scores;
- report paired semantic correctness and C/P/W ordinal scores first;
- report normalized exact match, token F1, METEOR, and post-hoc BERTScore F1
  as secondary diagnostics;
- report selector recall and evidence sufficiency independently of answer
  quality.
- define a unique grading item as `(row, candidate answer)`, deduplicating only
  within a row; randomize candidates so condition variants are not grouped.

### Decision rules and uncertainty

The primary outcome is semantic correctness, `C` versus `P/W`, on paired
candidate rows. C/P/W ordinal score (`C=2`, `P=1`, `W=0`), exact match, token
F1, METEOR, and BERTScore are secondary. For each declared contrast, compute a
paired row-bootstrap 95% percentile interval with 10,000 resamples and frozen
seed `20260803`; use the one-sided 95% lower bound only for a predeclared
positive-gain materiality statement and report two-sided 95% intervals for
descriptive comparisons. All intervals are panel-bound to these 2025 training
rows, not generalization intervals for 2026.

The selector-validity gate and 5-percentage-point gain value are operational
materiality thresholds, not proof of causality. Do not infer equivalence from a
non-significant result. Report regression counts and intervals rather than
claiming a raw two-point non-inferiority gate on an underpowered stratum.

Classify the bottleneck before choosing a deployable selector:

- **Model-assisted selection-supported:** the selector-validity audit passes;
  the all-475 `vlm12f` policy's fallback rate and usable-selector coverage are
  reported; the independent packet auditor marks the selected packet
  answerable on the applicable rows; and the corresponding automatic condition
  raises semantic-C rate by at least 5 percentage points with a one-sided
  paired 95% lower bound above zero. This supports that specific automatic
  selector, not ideal evidence in general. Report rescue/regression counts on
  packet-insufficient, interval-miss, usable-selector, fallback, and
  baseline-correct strata.
- **Human-oracle selection-supported:** on the untouched 48-row confirmatory
  audit, the independently human-verified packet is packet-sufficient and the
  human-audit Oracle-12 condition raises semantic-C rate by at least 5
  percentage points with a one-sided paired 95% lower bound above zero. Report
  the stratification-weighted audit estimate and its interval; report pilot,
  flagged, O24, and native-audio-required rows separately. This is a
  conditional empirical ceiling for the audited, still/text-representable
  population, not a 475-row upper bound.
- **Frame-capacity-supported:** on the human-audit subset, the frozen O12
  packet is human-answerable but cannot represent the required relation, and
  nested O24 improves the same row under an identical O12 subset,
  preprocessing, prompt, and transcript.
- **ASR/translation-pipeline-supported:** on the human-audit subset, human
  English beats the frozen Oracle-Whisper span on the same intervals, visual
  frames, formatting, and declared speech population.
- **Transcript-selection-supported:** on the fixed automatic common-nonnull
  set, the selected verbatim Whisper span beats full-clip Whisper on the same
  O12 frames, while the direct selected-span minus sham contrast is positive;
  the sham result is reported, not treated as a proof of equivalence. The full
  475-row fallback/null-inclusive transcript report remains secondary context.
- **Answering/grounding-supported:** on the human-audit subset, packet-only
  human answers are reference-correct on the exact oracle packet, but Qwen
  remains wrong on that packet; the paired oracle gain and uncertainty interval
  are reported. If the primary contrast is underpowered or thresholds
  conflict, classify the result as **mixed/indeterminate**, not as a null
  finding.

The result is diagnostic even when no class is supported. It must not be used
to promote an oracle selector or claim 2026 competition performance.

## Planned file map

### Tracked implementation

- Create: `src/trec_video_2026/vqa_experiments/oracle_evidence_ceiling.py`
  — annotation schema, blind-packet validation, interval adjudication,
  selector recall, actual-PTS oracle frame planning, transcript-span planning,
  condition matrix, and freeze bindings.
- Create: `tests/test_oracle_evidence_ceiling.py` — validation, leakage
  rejection, interval agreement, frame allocation, PTS binding, audio-span
  limits, sham mapping, condition identity, and freeze/tamper tests.
- Create: `scripts/build_qwen35_oracle_evidence_packet.py` — produce the
  reference-denied 475-row reviewer packet from the frozen label-free manifest.
- Create: `scripts/build_qwen35_oracle_evidence_plan.py` — construct and hash
  actual-PTS frame plans, Whisper spans, human-transcript bindings, and the
  sham-span map after adjudication.
- Create: `scripts/score_qwen35_oracle_evidence_ceiling.py` — verify frozen
  predictions, compute exact/F1/METEOR/BERTScore diagnostics, assemble blind
  grading packets, and calculate paired selector/oracle decisions.
- Create: `dstack/qwen35-4b-oracle-evidence-ceiling.dstack.yml` — pinned
  label-free inference configuration for the declared modality-specific
  contrasts.
- Create: `scripts/run-qwen35-oracle-evidence-ceiling-dstack.sh` — explicit
  preflight, upload, run, retrieval, checksum verification, and no-label mount
  boundary.
- Modify: `src/trec_video_2026/vqa_models.py` only if the existing controlled
  message builder cannot bind a selected transcript span without leaking
  annotation metadata; preserve the current prompt wording otherwise.

### Private/ignored execution artifacts

- Create: `data/vqa/benchmarks/qwen35-oracle-evidence-ceiling-20260803/protocol.json`
  and its SHA-256 freeze record.
- Create: `.../reviewer-packet.json`, independent reviewer files, adjudication,
  agreement, selector-recall, and sufficiency-audit artifacts.
- Create: `.../frame-inventory/`, `.../frame-plans/`,
  `.../frame-contact-sheets/`, `.../whisper-spans/`, and
  `.../human-transcript-spans/` with source/hash provenance.
- Create: `.../requests/`, `.../predictions/`, `.../freeze/`,
  `.../blind-grading/`, and `.../scores/`.

### Reporting after completion only

- Modify the canonical visual claim in
  `docs/project-status/research-visual-evidence.qmd` with the completed
  Oracle-12/O24 result.
- Modify the canonical transcript claim in
  `docs/project-status/research-speech-transcripts.qmd` with the completed
  transcript-span result.
- Link both claims from `docs/project-status/research-routing.qmd`, which owns
  the selector/fusion synthesis but does not duplicate either full result.
- Regenerate sibling Markdown/HTML from QMD and update the hub only if the
  operational baseline or next branch changes.

## Execution tasks

### Task 1: Freeze the protocol and label-free panel

**Files:**
- Create: `protocol.json` and freeze record under the private run directory.
- Create: `scripts/build_qwen35_oracle_evidence_packet.py`.
- Test: `tests/test_oracle_evidence_ceiling.py`.

- [ ] Write the failing tests for the exact 475-row panel, six excluded
  no-media records, forbidden answer/reference/prediction/condition fields,
  stable panel ordering, media hashes, and protocol hash binding.
- [ ] Run `uv run pytest tests/test_oracle_evidence_ceiling.py -q` and verify
  the new tests fail for missing contracts.
- [ ] Implement the packet builder using the existing label-free manifest and
  shared-data symlink; refuse a missing or conflicting `data` path.
- [ ] Write the frozen protocol with the modality-specific estimands,
  annotation rubric, frame/audio budgets, freeze order, and decision rules
  above.
- [ ] Run the focused tests, `git diff --check`, and a private-data scan before
  saving the packet hash.

### Task 2: Calibrate and audit blinded required-evidence selectors

**Files:**
- Create: private selector outputs, calibration packet, two de novo human audit
  files, third de novo disagreement/QA file, adjudication, agreement, and
  selector-recall artifacts.
- Modify: `src/trec_video_2026/vqa_experiments/oracle_evidence_ceiling.py`.
- Test: `tests/test_oracle_evidence_ceiling.py`.

- [ ] Add validation tests for interval topology, duration bounds, alternative
  sufficient sets, modality values, transcript span limits, and reference
  leakage at every JSON nesting level.
- [ ] Implement the reference-denied visual selector and frozen-ASR transcript
  selector with output schemas that permit timestamps/enums/confidence only;
  reject captions, summaries, rationales, answers, and rewritten transcript
  text.
- [ ] Run and freeze the 20-row human-reviewed rubric-calibration pilot before
  the full-panel selector; revise prompts, thresholds, and rubric only before
  selector outputs are frozen.
- [ ] Run both automatic selectors over all 475 rows and freeze their raw
  outputs before any references, predictions, or condition labels are loaded.
- [ ] Have two human reviewers independently annotate the calibration rows and
  a frozen stratified 48-row audit sample without seeing selector outputs;
  review every selector disagreement, abstention, low-confidence, non-still,
  audio-only, and malformed-output row.
- [ ] Have a third reviewer adjudicate reviewed disagreements and flagged rows
  after the first two files freeze. Preserve human corrections as a separate
  `human-audit-ceiling` subset; never patch the primary automatic selector.
- [ ] Compute selector recall, disagreement/flag rates, adjudicated intervals,
  still-frame representability, native-audio-required rows, analysis
  populations, and the frozen stop/run decision.
- [ ] Stop and recalibrate if selector flags exceed 20% of the panel or the
  audit-quality gate fails; do not silently expand to 475 human reviews. Do
  not drop small populations; mark every population with fewer than 30
  analyzable rows descriptive-only.

### Task 3: Build and audit oracle visual/audio evidence

**Files:**
- Create: `scripts/build_qwen35_oracle_evidence_plan.py`.
- Modify: `src/trec_video_2026/vqa_experiments/oracle_evidence_ceiling.py`.
- Test: `tests/test_oracle_evidence_ceiling.py`.

- [ ] Add tests for exact 12-frame allocation, distinct decoded frame
  identities/PTS even when pixel hashes repeat, chronological order, interval
  coverage, conditional nested O24 planning, transcript span word/time limits,
  matching timestamp rendering, and Qwen-token-count/timestamp-structure sham
  spans.
- [ ] Decode actual frame inventories with PTS/frame index/time base and bind
  every planned image to source media and extraction hashes.
- [ ] Construct exact c2 Uniform-12, coverage-conditional `vlm12`, all-panel
  `vlm12f`, human-audit Oracle-12, and conditional Oracle-24 plans without
  putting interval metadata into model requests.
- [ ] Select timestamped Whisper segments using the frozen transcript selector
  and construct both coverage-conditional `uniform12-asrsel-required-span`
  and all-panel `uniform12-asrsel-fallback` requests; bind human literal
  English transcript files only for the calibration/audit subset and declared
  capacity exceptions. Keep transcribers and transcript reviewers blind to the
  question, reference, prediction, and locator rationale.
- [ ] Have the independent packet auditor answer each exact Uniform-12 and
  VLM-selected-12 packet, or record `insufficient`, before references or
  predictions are exposed; include question-only and sham/mismatched-packet
  controls. On the human-audit subset, use two packet-only human auditors and
  adjudicate disagreements.
- [ ] Freeze `uniform12_packet_answer`, `vlm12_packet_answer`, and
  `human_audit_packet_answer` with their provenance; record fallback status and
  keep interval recall as a separate measurement.
- [ ] Keep failed O12 construction, true O12 representation insufficiency, and
  non-still/native-video requirements as separate reason codes.
- [ ] Freeze all evidence plans, contact sheets, transcript bytes, sham mapping,
  and request hashes before any GPU invocation.

### Task 4: Run label-free matched inference

**Files:**
- Create: `dstack/qwen35-4b-oracle-evidence-ceiling.dstack.yml`.
- Create: `scripts/run-qwen35-oracle-evidence-ceiling-dstack.sh`.
- Create: private request, runtime, prediction, and freeze artifacts.

- [ ] Verify the dstack image, model revision, GPU, media/evidence hashes,
  label-free mounts, selector revision and outputs, independent packet-auditor
  revision, c2 baseline condition IDs, canonical model-input digests, and
  complete declared condition set before provisioning.
- [ ] Run the declared compatibility subset and stop on any contract,
  provenance, or nonempty-answer failure.
- [ ] Run only the new declared contrasts with identical Qwen revision, prompt,
  decoding, frame preprocessing, and panel order: coverage-conditional
  `vlm12-accepted-whisper-policy`, all-panel
  `vlm12f-accepted-whisper-policy`, all-panel
  `uniform12-asrsel-fallback`, coverage-conditional
  `uniform12-asrsel-required-span`, and all-panel
  `vlm12f-asrself-fallback`. Run the human-audit ceiling only on the frozen
  confirmatory audit/exception subset. Reuse existing arms only after verifying
  frame bytes, requested timestamp text, actual-PTS provenance, transcript
  bytes/null, chat-template serialization, processor revision, runtime, model,
  decoding, and c2 condition identity.
- [ ] Retrieve output and verify every remote file against the frozen checksum
  manifest without copying raw videos into the repository.
- [ ] Freeze predictions before loading references, answer labels, or grading
  packets.

### Task 5: Grade and score the frozen predictions

**Files:**
- Create: `scripts/score_qwen35_oracle_evidence_ceiling.py`.
- Test: `tests/test_oracle_evidence_ceiling.py`.
- Create: private blind grading and score artifacts.

- [ ] Add tests for condition-blind packet construction, one-time unblinding,
  paired C/P/W transitions, missing-answer rejection, and score arithmetic.
- [ ] Generate the opaque grading packet after prediction freeze and keep the
  condition map separate.
- [ ] Obtain two condition-blind strong-text-judge grades with confidence or
  abstain, keep judge identities separate from selector identity, and send
  disagreements/abstentions plus a frozen random spot sample to human
  adjudication before unblinding.
- [ ] Compute semantic C/P/W first, then exact match, token F1, METEOR, and
  BERTScore F1 from the same frozen answer/reference pairs.
- [ ] Report selector recall, model-audited evidence sufficiency, human-audit
  sufficiency, automatic-selector gains, human-audit ceiling gains, audio-span
  gains, O24 gains, direct selected-Whisper-minus-sham contrast,
  ASR/translation gains, and all regressions by modality and declared
  population.
- [ ] Compute paired row-bootstrap percentile intervals with 10,000 resamples,
  frozen seed, declared sidedness, and panel-bound interpretation; do not use
  non-significance as an equivalence claim.
- [ ] Apply the predeclared decision rules and classify the result as selection,
  frame-capacity, ASR/translation, transcript-selection, grounding, or
  mixed/indeterminate bottleneck.

### Task 6: Record the claim and decide the next branch

**Files:**
- Modify: `docs/project-status/research-routing.qmd`.
- Modify: `docs/project-status/research-visual-evidence.qmd` when visual
  selection interpretation changes.
- Modify: `docs/project-status/research-speech-transcripts.qmd` when transcript
  interpretation changes.
- Modify: `docs/project-status/experiment-ledger.qmd` only if the operational
  baseline or branch queue changes.

- [ ] Record observed results separately from interpretation and preserve the
  exact panel/condition boundary.
- [ ] State what the oracle result rules out and what it does not rule out.
- [ ] Keep the visual claim canonical in `research-visual-evidence.qmd`, the
  transcript claim canonical in `research-speech-transcripts.qmd`, and link
  both from `research-routing.qmd`.
- [ ] If selection is causal, plan a label-blind question-only selector against
  the oracle ceiling on held-out data.
- [ ] If oracle evidence does not help, stop selector work and plan a stronger
  answering/grounding model or prompt under the same evidence contract.
- [ ] Regenerate QMD-derived Markdown/HTML, run link/privacy/accessibility
  checks, and verify no private dataset artifacts entered reports.

## Review checklist before execution

- [x] Sol review completed on the revised exact plan and all decisive changes
  recorded.
- [ ] No answer/reference/prediction leakage path remains in reviewer packets,
  oracle plans, prompts, or model requests.
- [ ] Existing baseline/treatment identity checks are specified before reuse.
- [ ] Primary 12-frame and conditional 24-frame decisions are separated.
- [ ] Transcript ceiling is not misreported as native-audio quality.
- [ ] Dual-model condition-blind semantic grading is primary; human grading is
  limited to calibration, disagreements, abstentions, and spot audits;
  automatic lexical metrics are secondary.
- [ ] Before execution, commit the plan/controller changes or freeze their
  source hashes from a clean revision; do not execute from an untracked or
  unrelated dirty worktree.
- [ ] GPU launch remains a separate authorization decision.
