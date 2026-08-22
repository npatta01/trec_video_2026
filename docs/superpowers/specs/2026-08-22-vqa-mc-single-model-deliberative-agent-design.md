# VQA MC Single-Model Deliberative Agent Design

Date: 2026-08-22

## Objective

Test whether a bounded but genuinely deliberative Gemini agent improves the
frozen 20-video multiple-choice diagnostic over both the direct native-AV
control and the failed two-call adaptive prototype.

The experiment holds the model fixed at `gemini-3.7-flash` with high thinking.
Gemini owns semantic reasoning, hypothesis revision, evidence interpretation,
inspection choice, and the final answer. Python provides only a reliable tool
runtime, append-only memory, structural validation, artifact freezing, and
hard operational limits.

## Design principle: fixed shell, dynamic reasoning

The protocol has a fixed outer shell so the experiment is reproducible:

1. two independent complete-video analyses;
2. a model-directed evidence-search loop;
3. one fresh complete-video final judgment.

The reasoning inside that shell is not a fixed task workflow. Gemini decides
whether additional evidence is useful, what uncertainty matters, whether to
rewatch the complete clip or inspect a local interval, which FPS and resolution
to request, whether to broaden or backtrack, and when to stop. The controller
does not classify questions or decide what observations mean.

This design deliberately excludes controller branches for OCR, counting,
gestures, speech, temporal order, ambiguity, or any other semantic category.

## Scientific design

### Question

Does a single-model deliberative loop with independent initial analyses,
model-selected native-AV investigation, append-only evidence, and fresh final
judgment improve answer quality without the regressions created by the earlier
last-answer-wins prototype?

### Hypothesis

The existing Gemini model has enough latent capability to solve more of the
panel when it is required to construct competing hypotheses, seek
disconfirming evidence, preserve earlier observations, and reconcile the full
case before answering. Adaptive extra pixels alone are not the treatment;
deliberative evidence use is.

### Falsifier

Do not advance this protocol if any of the following occurs on the frozen
development panel:

- final top-1 does not exceed the 15/20 direct native-AV control;
- regressions from correct direct answers are at least as numerous as repairs;
- either of the two known prior-prototype regressions is repeated;
- the agent overwhelmingly follows one uniform inspection path rather than
  adapting or stopping;
- fewer than 20 final rankings are valid;
- structural notebook or provenance integrity is incomplete; or
- the predeclared cost cap is exceeded.

A positive result on this inspected panel establishes only that the mechanism
is worth a fresh label-isolated test. It does not establish hidden-set
accuracy.

### Treatment and controls

- **Frozen operational control:** direct Gemini native AV, complete video,
  4 FPS, low media resolution, 15/20 top-1.
- **Historical negative comparison:** the two-call adaptive native-AV agent,
  15/20 final, with two repairs and two regressions relative to the operational
  control.
- **Treatment:** the single-model deliberative protocol defined here.

Hold fixed the 20 canonical rows, option order, exact option strings, video
assets, Gemini model revision, high thinking level, baseline media settings,
seed policy, label isolation, and scoring implementation. Only the reasoning
and evidence-acquisition protocol changes.

## Agent protocol

### Phase 1: complete-video anchor

Gemini receives the complete native audio-video, question, and all four exact
options. It returns a concise decision record rather than exposed private
chain-of-thought:

- a complete A-D ranking;
- a short hypothesis for every option;
- literal timestamped observations already noticed;
- evidence supporting and challenging every live option;
- the strongest reason its leader could be wrong;
- unresolved questions; and
- up to two possible observations that could materially change the ranking.

Literal observation and interpretation are separate fields. For example, a
transcribed visible string belongs in an observation field; a conclusion about
what that string means belongs in interpretation.

### Phase 2: blind complete-video solver

A fresh Gemini context receives the same complete video, question, and options
but no anchor response, ranking, or notebook. It emits the same decision-record
shape.

The controller assigns neutral solver IDs and does not label either analysis as
primary when later presenting them. This creates a second opportunity to
notice evidence without exposing a current leader and inducing confirmation.
It remains one-model testing: both calls use the same Gemini model and settings.

### Phase 3: investigation planner

A text-only fresh Gemini context receives:

- the question and exact options;
- both anonymized solver records;
- the append-only notebook;
- the available native-AV actions; and
- the exact remaining call, media-seconds, token, and cost budgets.

The planner evaluates all four options and identifies the most consequential
unresolved distinction. It must choose one of:

- `finish_search`, with a provisional complete ranking and an explanation of
  why another observation is unlikely to reorder it; or
- one media action, with a neutral investigative focus and pre-registered
  expected observations under the relevant competing interpretations.

The controller treats the focus and expectations as opaque model-authored
text. It checks their presence and size but never interprets them.

### Phase 4: blinded native-AV observer

The observer is another fresh Gemini context. It sees only:

- the requested native-AV view;
- the neutral investigative focus;
- any literal transcription or event-description precision requested by the
  planner; and
- the original-video time origin.

It does not see solver rankings, option IDs, current hypotheses, or which
answer would benefit from any observation. It returns:

- literal timestamped observations;
- a separate concise interpretation;
- limitations of the view; and
- whether the requested feature was absent, present, or not resolvable.

The controller adds immutable view provenance: actual interval, actual video
duration covered, requested FPS, resolution, audio availability, provider
artifact reference, usage, and response hash.

### Phase 5: adaptive continuation

After the first observation, a fresh planner call receives the updated
notebook and chooses `finish_search` or one final media action. It may rewatch
the complete clip, inspect a new interval, expand the previous interval,
increase fidelity, return to a discarded option, or pursue a different
uncertainty.

The loop permits at most two observer actions. If a second observation is
requested, it is appended and the protocol proceeds directly to final
judgment. This preserves the final-call budget and bounds the experiment.

### Phase 6: fresh final judge

A fresh Gemini context receives:

- the original question and exact options;
- the complete native audio-video at the baseline settings;
- literal observations and their media provenance first;
- the two anonymized initial analyses;
- later interpretations and unresolved conflicts; and
- the complete audit-safe notebook, excluding provider-private reasoning.

The judge solves the item afresh. It is not told to preserve the anchor or the
latest provisional answer. It returns:

- a complete final A-D ranking;
- the decisive observation IDs;
- the strongest counterargument;
- why that counterargument did or did not change the answer;
- unresolved ambiguity, if present; and
- a concise evidence-based rationale.

The competition still requires one answer, so ambiguity does not suppress the
ranking. It is recorded as diagnostic metadata for later adjudication and
metric interpretation.

## Native-AV action interface

The first prototype exposes only capabilities already used successfully:

- `rewatch_full(fps, resolution, investigative_focus)`;
- `inspect_interval(start_seconds, end_seconds, fps, resolution,
  investigative_focus)`.

Allowed FPS values remain `1`, `4`, and `8`. Resolution remains `low` or
`high`. Native audio stays attached when the provider view supports it. The
controller does not advertise independent audio control until provider support
is verified.

An invalid action is returned to the planner as a mechanical tool error. The
controller never silently clamps timestamps or substitutes settings. A planner
may correct the request only if budget remains.

## Evidence notebook

The notebook is the sole cross-call state and is append-only. It contains:

- immutable run, row, model, media, and prompt-contract identifiers;
- the two neutral solver records;
- planner decisions and pre-registered expectations;
- controller-issued view IDs and realized media parameters;
- observer entries with stable observation IDs;
- literal observations separate from interpretations;
- provisional rankings and open uncertainties;
- all schema or tool errors;
- per-call usage and cost; and
- the final judge response and cited observation IDs.

The controller verifies only structure and referential integrity. It can prove
that an observation ID exists and predates the judgment; it cannot decide
whether the observation semantically supports an option.

Notebook entries have bounded field lengths. The final prompt renders literal
observations before hypotheses and interpretations so early conclusions do not
bury the actual evidence.

## Thin-controller boundary

The controller may:

- call Gemini and execute requested media views;
- validate schemas, rankings, numeric ranges, and supported media parameters;
- enforce action, call, token, media-seconds, and cost budgets;
- reserve capacity for the final judge before authorizing an inspection;
- attach provenance and hashes;
- enforce append-only ordering and cited-ID existence;
- freeze every stage before the next label-isolated stage; and
- apply explicit transport/schema fallbacks.

The controller must not:

- predict a question category;
- determine required semantic scope;
- decide whether an observation is correct or sufficient;
- score confidence or expected information gain itself;
- override a valid model ranking based on semantic rules;
- fuse different model families; or
- access answer labels during inference.

## Budgets and stopping

The maximum hard-item path is seven Gemini calls:

1. anchor media call;
2. blind solver media call;
3. first planner text call;
4. first observer media call;
5. second planner text call;
6. optional second observer media call; and
7. final judge media call.

If the first planner chooses `finish_search`, the item uses four calls. If the
second planner stops, it uses six. The controller always reserves the final
judge call before approving any observer action.

Before credentials or inference are loaded, estimate the full 20-item
worst-case path using actual media durations and configured provider pricing.
Reject the run if the conservative estimate exceeds `$0.75`. Record observed
cost and usage after every call; terminate further investigation, but still
run the reserved final judge, if an item's remaining budget cannot cover both
the proposed view and final call.

## Failure and fallback behavior

- Invalid anchor or blind-solver output is recorded and omitted from planning;
  the other valid record may continue.
- If both initial solvers are invalid, skip investigation and run the final
  judge on the complete video.
- Invalid planner actions receive one mechanical correction opportunity only
  when doing so cannot consume the final-call reserve.
- An invalid observer response is recorded; it never replaces earlier notebook
  evidence.
- An invalid final response falls back to the most recent complete valid
  ranking, preferring the last planner ranking, then a valid solver ranking.
- Every fallback changes `final_source`; nothing fails silently.

Provider retries that would create a new semantic sample are not automatic.
Terminal provider failures are frozen once and handled through the declared
fallback path.

## Artifact and integrity lifecycle

The runner writes immutable, exclusive-creation artifacts in this order:

1. manifest, input identity check, and conservative cost preflight;
2. two initial raw-response freezes and parsed solver records;
3. notebook initialization freeze;
4. per-round planner decision, requested action, realized view, raw observer
   response, parsed observation, and notebook snapshot;
5. final-judge raw response and final prediction freeze;
6. completion and usage summary; and
7. scorer output after predictions are frozen.

Inference artifacts reject label-bearing request bundles. Labels remain in the
separate frozen scoring artifact and are loaded only by the scorer.

Questions, options, labels, row identifiers, predictions, media, transcripts,
notebooks, and provider responses remain under the ignored private `data/`
tree. Tracked reports contain sanitized aggregate results and hashes only.

## Metrics and decision rule

Report:

- top-1 accuracy and MRR for direct control, each initial solver, planner
  provisional rankings, and final judge;
- repairs, regressions, unchanged hits, and unchanged misses versus the frozen
  direct control;
- preservation of the two previously regressed direct-control hits;
- calls, observer actions, full rewatches, local inspections, requested FPS,
  requested resolution, inspected-duration fraction, usage, and cost;
- planner stop points and the distribution of zero-, one-, and two-action
  paths;
- schema, tool, observer, judge, and fallback counts;
- notebook completeness, provenance completeness, and citation integrity;
- final departures from both initial solvers and whether they cite observations
  produced after a pre-registered investigation; and
- ambiguity-flag rate, reported separately from official-label accuracy.

The protocol is promising on this development panel only if:

- all 20 final rankings are valid;
- final top-1 exceeds 15/20;
- repairs strictly exceed regressions;
- the two known prior-prototype regressions remain correct;
- no structural integrity check fails;
- at least one item stops without inspection and at least two distinct media
  action configurations occur, demonstrating that the agent is not merely a
  fixed uniform second pass; and
- observed cost stays at or below `$0.75`.

Even if this gate passes, promotion requires a fresh label-isolated panel
selected before seeing results. The already inspected 20 rows remain a
mechanism and regression test only.

## Implementation boundaries

- Add a new provider-independent module for deliberative schemas, notebook
  transitions, action validation, budgets, prompts, and fallback selection.
  Do not mutate the frozen two-call module or its artifacts.
- Add a new Gemini runner for uploads, model calls, media views, immutable
  freezes, usage accounting, and the CLI.
- Add a separate post-freeze scorer that compares the treatment with the
  frozen direct and two-call results.
- Add focused unit tests for schemas, visibility boundaries, blind solver
  inputs, action validation, final-call reservation, append-only notebooks,
  provenance, citation IDs, failure fallbacks, label rejection, and scoring.

The implementation plan will name exact paths after this design is approved.

## Verification before any provider run

- Provider-independent unit tests pass using mock responses for the zero-,
  one-, and two-inspection paths.
- Tests prove that the blind solver cannot receive anchor state and that the
  observer cannot receive option IDs or provisional rankings.
- Tests prove that no semantic category or semantic answer-change gate exists
  in controller code.
- Tests prove that final-call capacity is reserved on every path.
- Existing direct and two-call MC tests remain passing.
- `ruff check` and `git diff --check` pass on changed files.
- Shared `data` resolves to the canonical dataset directory.
- A no-credential preflight verifies 20 input identities, label isolation,
  media availability, action limits, immutable output paths, and the `$0.75`
  worst-case cap.

Only after these checks pass should the paid 20-video development run begin.
