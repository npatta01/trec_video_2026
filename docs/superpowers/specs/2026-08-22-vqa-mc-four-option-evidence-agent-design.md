# VQA MC Four-Option Evidence Agent Design

Date: 2026-08-22

## Objective

Test whether one Gemini model can improve the frozen 20-video multiple-choice
diagnostic when it behaves as an evidence-seeking agent instead of repeating
the same complete-video answer in several roles.

This is an inference-only experiment. It does not train, fine-tune, calibrate,
or update a model. All semantic roles use the same Gemini model in fresh
contexts with deliberately different information visibility. Python supplies
the state machine, native-video view execution, schema validation, immutable
artifacts, label isolation, and budget enforcement; it does not decide what an
answer means.

The experiment first runs a five-video operational canary consisting of the
five misses from the frozen direct Gemini control. If the canary passes its
structural, behavioral, accuracy, and cost gates, the same frozen protocol runs
on all 20 videos. This familiar panel is a regression and mechanism suite only,
not unbiased promotion evidence.

## Approved scope

- Model family: Gemini only.
- Model training: none.
- Candidate set: all four displayed choices, A through D.
- Media: native video with audio; the model may request a complete rewatch or
  an interval at a supported FPS and media resolution.
- Evidence collection: at least one blinded observation for every item, with
  at most three observations per item.
- Reasoning allowance: high thinking and the provider's advertised maximum
  output allowance for every semantic role; there is no smaller role-specific
  answer-token ceiling.
- Recovery: a response ended by the output limit, malformed structured data,
  or an incomplete response is an operational failure, never an answer. It is
  retried or continued under the declared recovery policy.
- Cost: combined spend for the canary, any declared operational recovery, and
  the 20-row run may not exceed `$1.50`.
- Execution: run the five-video canary first, then automatically run all 20
  only if the canary and the full-run preflight pass.

## Why the previous architecture failed

The earlier deliberative agent used two nominally independent complete-video
solvers with the same model, prompt, seed, and media settings. They agreed on
top choice in 19/19 rows where both were valid. The planner interpreted that
agreement as sufficient, opened zero observer views, and the final judge
copied the blind solver's top choice in 20/20 rows. Its 14/20 result was worse
than the 15/20 frozen direct control.

The defect was architectural, not a lack of available resolution: the
controller allowed the model to stop before gathering new evidence. The new
protocol removes solver voting and confidence-based early stopping. It
requires an observation chosen to distinguish live answer hypotheses, hides
the answer choices from the observer, and subjects any stop proposal to a
separate adversarial review.

## Considered approaches

### Selected: one model, asymmetric roles, host-enforced investigation

A protected direct answer, four-option evidence ledger, evidence planner,
blinded observer, adversarial stop critic, and fresh final judge all use the
same Gemini model in separate contexts. At least one observation is mandatory.
This isolates the value of agentic evidence collection without introducing a
second model family or training.

The main risk is correlated reasoning because every semantic role uses the
same model. Information barriers and role-specific contracts reduce, but do
not eliminate, that risk. The canary explicitly tests whether the observer
produces new literal evidence and whether the final decision can repair direct
misses without treating repeated opinion as evidence.

### Rejected for this test: multiple same-model solvers and consensus

Several complete-video samples could be generated with different seeds and
voted or fused. The prior run already showed that same-model agreement can be
duplicated confidence. More samples would spend money without guaranteeing a
new observation, so consensus is not the treatment in this experiment.

### Deferred: heterogeneous Gemini/Qwen deliberation

A second model family could create useful disagreement, but it would confound
the question of whether one capable native-AV model can improve through better
evidence search. It also adds infrastructure and cost. Qwen remains a later
ablation only after this one-model mechanism is understood.

## Architecture

```text
frozen direct Gemini answer (protected baseline)
                         |
complete native-AV evidence initialization
                         |
                  A/B/C/D ledger
                         |
     +------------- evidence planner <------------------+
     |                    |                              |
     |              neutral request                     |
     |                    |                              |
     |        blinded native-AV observer                 |
     |                    |                              |
     +------------ append literal evidence              |
                          |                              |
                 adversarial stop critic ---- continue -+
                          |
                    stop accepted
                          |
     fresh complete-native-AV final judge
                          |
       final A/B/C/D ranking + cited evidence
```

Every semantic box is a fresh Gemini context. Only the ledger and immutable
evidence notebook cross role boundaries. The observer never receives the
question, answer options, option IDs, current ranking, protected direct answer,
or model-authored predictions about which option an observation would favor.

## State and contracts

### Protected direct answer

The experiment reuses the already frozen, label-isolated direct Gemini ranking
for each canonical row. It does not resample the control. The direct top choice
is a protected default, not a vote and not evidence.

The final judge may retain it or replace it. A replacement must cite at least
one new literal observer entry and explain both how that evidence contradicts
the direct choice and how it supports the replacement. The host validates that
the cited entries exist and predate the judgment; the adversarial critic and
final judge assess the semantic adequacy of the evidence.

### Four-option evidence ledger

The complete native-AV initializer sees the question and exact displayed
options. For each A-D option it records:

- an observable claim that would have to be true;
- literal supporting evidence IDs;
- literal contradicting evidence IDs;
- important untested claims;
- a status of `viable`, `disfavored`, or `inconsistent`; and
- a concise interpretation kept separate from literal observation.

It also emits a complete provisional ranking. No option may be dropped merely
because it is outside an initial top two. A status change must cite evidence or
remain explicitly provisional.

### Evidence planner

The planner sees the question, all four choices, the protected ranking, and the
current ledger. It chooses the single unresolved distinction whose observation
would most change the ordering of the still-viable options.

For each requested observation it returns:

- a neutral investigative request that can be shown to the observer without
  exposing answer hypotheses;
- the native-video view to inspect;
- the unresolved ledger claims the request tests;
- expected literal outcomes for every affected viable option; and
- why the request is more discriminating than another complete answer pass.

The planner cannot stop before the first valid blinded observation. It never
selects an answer on the observer's behalf.

### Native-video view

The action vocabulary is intentionally small while the reasoning that chooses
an action remains dynamic:

- `rewatch_full(fps, resolution, neutral_request)`; or
- `inspect_interval(start_seconds, end_seconds, fps, resolution,
  neutral_request)`.

The runner discovers and records the exact FPS and resolution values supported
by the installed Gemini API before inference. The intended starting set is 1,
4, and 8 FPS with low and high media resolution. Unsupported values fail
preflight; the runner never silently substitutes them. Native audio remains
attached whenever the provider supports it for that view.

A later round may widen an interval, increase FPS or resolution, rewatch the
complete clip, inspect a different interval, or test a different option. These
are model choices, not hard-coded question-category branches.

### Blinded observer

The observer receives only the selected native-AV view, neutral request, and
original timeline origin. It returns:

- literal timestamped observations;
- an exact visible-text or heard-speech transcription when requested;
- whether the requested feature is `present`, `absent`, or `unresolved`;
- view limitations such as occlusion, missing context, or insufficient detail;
  and
- a concise interpretation separate from the literal observations.

The observer cannot rank choices because it never sees them. Its raw response,
parsed record, realized media settings, timestamps, content hash, usage, and
cost are frozen before the ledger is updated.

### Ledger updater

A fresh text-only Gemini context maps the blinded observation back onto all
four option claims. It may add support, contradiction, or an unresolved note;
it may not rewrite or delete earlier evidence. Contradictory observations stay
visible. Each status or ranking change must cite one or more immutable evidence
IDs.

### Adversarial stop critic

After each updated ledger, a fresh critic tries to reject stopping. It checks:

- whether every viable option has at least one tested discriminating claim;
- whether the current leader is supported by literal evidence rather than
  solver agreement or narrative plausibility;
- whether any contrary observation remains unexplained;
- whether a feasible different view could change the ranking; and
- whether replacing the protected direct answer meets the positive-evidence
  requirement.

The critic returns `continue` with one unresolved distinction or
`accept_stop` with a reason. After the first observation, `accept_stop` ends
the search. A `continue` decision starts another planner/observer/update round
while fewer than three observations have been made.

If three observations are exhausted and the critic still rejects stopping,
the item records `forced_stop`. The final judge still answers because the task
requires one choice, but the result is not evidence-certified.

### Final judge

The final judge receives the complete native video with audio, the original
question and A-D choices, the protected direct ranking, and the immutable
notebook rendered with literal observations before interpretations. It solves
the item afresh and returns:

- a complete A-D ranking;
- the selected top choice;
- decisive evidence IDs;
- the strongest counterevidence and how it was resolved;
- whether the protected answer was retained or replaced;
- for a replacement, the cited contradiction and positive support required by
  the protection rule; and
- `evidence_certified` or `forced_stop` status.

The judge does not receive answer labels or earlier model-private reasoning.

## Output allowance and recovery

The runner queries or pins the model's advertised output limit and configures
that provider maximum for initializer, planner, observer, updater, critic, and
judge. It does not assign smaller token ceilings by role. High thinking remains
enabled.

The following are operational failures, not semantic answers:

- provider finish reason `MAX_TOKENS` or its equivalent;
- incomplete JSON or a structurally incomplete object;
- missing A-D option records or a non-permutation ranking; and
- a response that ends inside a field even if the provider reports a normal
  stop.

One same-contract recovery is allowed for each failed call. If the API
supports continuation from a valid prefix without changing the semantic
request, continue it; otherwise retry in a fresh context with the original
inputs plus a mechanical instruction to return the complete schema. Both raw
attempts and usage are retained. A second failure invokes the declared stage
fallback and marks the item operationally invalid; it is never scored as a
wrong model answer.

The trajectory, not reasoning length, is bounded: at most three observations,
one recovery per failed semantic call, and the `$1.50` experiment cap.

## Canary and automatic continuation

### Canary panel

The five-video canary is exactly the set of rows missed by the frozen 15/20
direct Gemini control. A private, frozen canary manifest identifies those rows
without containing their labels or reference answers. Labels remain
unavailable to every inference component and are loaded only by the scorer
after all five predictions and notebook hashes freeze.

### Canary gates

Continue automatically to the 20-row run only when all conditions hold:

- five final rankings are structurally valid;
- every row contains at least one valid blinded observation;
- every observer prompt passes a logged information-barrier audit;
- every final row contains complete A-D ledger entries and valid citations;
- no truncated or incomplete response was treated as an answer;
- the canary repairs at least one of the five frozen-control misses;
- observed spend plus the conservative remaining 20-row estimate is at or
  below `$1.50`; and
- the frozen protocol, prompts, schemas, model settings, and recovery policy do
  not need a semantic change.

If a structural or transport defect requires a correction, stop automatic
continuation, version the protocol, rerun the canary, and count all prior spend
against `$1.50`. Never patch a live run in place.

### Full 20-row run

If the canary passes without a protocol change, its five immutable row outputs
may be reused in the 20-row result because they used the identical frozen
protocol. The remaining 15 rows then run without label access. If reuse cannot
be proven by request, media, prompt, model, and configuration hashes, rerun all
20 subject to a fresh cost preflight.

## Cost enforcement

Before credentials or paid calls are loaded, the runner computes:

1. a five-row canary estimate;
2. a conservative 20-row estimate at the maximum three-observation path;
3. maximum permitted recovery calls; and
4. cumulative spend from any earlier attempt under this protocol version.

The estimate uses actual clip durations, the provider's active price schedule,
native-video token estimate, text context size, and advertised maximum output
allowance. It assumes the maximum permitted trajectory and recovery count and
prices every response at the provider output maximum. If that formal worst-case
total exceeds `$1.50`, no paid call is made and the runner reports the estimate
and its assumptions. An empirical estimate based on prior high-thinking usage
may also be reported for planning, but it cannot override this gate.

During execution, the host reserves the maximum cost of the next call before
authorizing it and always preserves a final-judge reserve. It stops new calls
before observed spend plus formal reserves can exceed `$1.50`.

## Failure behavior

- Invalid initializer: create A-D entries with `unknown` evidence state and
  continue to the mandatory inspection; mark initializer invalid.
- Invalid planner after its one recovery: request a complete high-resolution
  native-AV observation with the neutral instruction to produce a gapless,
  literal timeline; record `planner_fallback`.
- Invalid observer after its one recovery: retain the failed view provenance,
  append no semantic evidence, and let the critic request a different view if
  rounds remain.
- Invalid updater after its one recovery: preserve the old ledger and append
  the observer evidence as unmapped; the critic sees it directly.
- Invalid critic after its one recovery: continue if rounds remain; otherwise
  record `forced_stop`.
- Invalid final judge after its one recovery: retain the protected direct
  ranking as the explicit fallback and mark the item operationally invalid.
- Budget exhaustion: authorize no new investigative call, preserve the final
  judge reserve when possible, and mark `budget_stop`.

No fallback silently becomes a successful agent answer.

## Integrity and private artifacts

The runner uses exclusive file creation and freezes each stage before the next
one. Private artifacts include exact questions, options, media references,
model requests and responses, ledgers, notebooks, predictions, labels used by
the post-freeze scorer, and row-level comparisons. They stay in the ignored
private experiment directory.

The tracked aggregate includes only counts, costs, distributions, hashes, and
decision-gate outcomes. It must not contain video IDs, exact questions,
options, transcripts, labels, row-level predictions, or raw model text.

Integrity checks cover:

- canonical input and frozen-control hashes;
- absence of labels from every inference request and notebook;
- observer-prompt absence of the question, choices, option IDs, rankings,
  protected answer, and option-conditioned expected outcomes;
- append-only observation IDs and valid citations;
- exact requested and realized media parameters;
- model, prompt, schema, pricing, and configuration hashes;
- call usage and cumulative cost reconciliation; and
- canary-output identity before any reuse in the 20-row result.

## Testing and verification

Implementation follows test-first development for every new behavior. Unit
tests must prove:

- the four-option ledger rejects missing or duplicate option entries;
- no stop is accepted before a valid blinded observation;
- observer requests contain none of the prohibited answer information;
- all viable options are represented in planner and critic contracts;
- a proposed direct-answer replacement requires cited contradiction and
  positive support;
- three rejected stops produce `forced_stop` rather than an unbounded loop;
- truncation and incomplete structures trigger recovery and never scoring;
- provider-level output allowance is used without smaller role caps;
- canary continuation and reuse require identical protocol hashes;
- labels cannot enter inference artifacts;
- dollar reservation prevents calls that could breach `$1.50`; and
- sanitized aggregates exclude private row data.

Integration tests use a fake Gemini client to exercise retain, evidence-backed
replacement, forced-stop, malformed-output recovery, planner fallback, judge
fallback, canary-stop, automatic-continuation, and cost-stop paths. The paid
canary begins only after the focused suite, existing deliberative-agent
regressions, artifact-integrity verifier, dataset-symlink check, and dry-run
cost preflight pass.

## Decision rules

The five-row canary is a mechanism gate, not a performance estimate. It must
repair at least one known direct miss and meet every structural gate before the
full run is justified.

On the frozen 20-row regression panel, advance the protocol only as a candidate
for a fresh label-isolated evaluation when:

- final top-1 exceeds the 15/20 direct control;
- repairs outnumber regressions relative to that control;
- 20/20 final rankings are valid;
- every row contains at least one blinded observation;
- all notebook, citation, information-barrier, and cost checks pass; and
- no semantic prompt or schema changed after the canary began.

Twenty out of 20 is the aspirational result, not a promised outcome or a gate
chosen after seeing results. Even 20/20 on this inspected development panel
would require confirmation on a fresh label-isolated panel before submission
or promotion.
