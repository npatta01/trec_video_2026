# Premise-Tolerant Grounded Answering Design

Date: 2026-08-14

## Objective

Test whether a narrower answer/verification policy repairs false abstentions in
the completed issue #35 experiment. The treatment must judge support for the
requested answer separately from literal agreement with every locator or
presupposition in the question.

This is a selected, spent-row mechanism test. It can show whether the observed
false-abstention mechanism is repairable under one frozen policy; it cannot
estimate prevalence or generalization across the 475-row panel.

## Scope

The experiment uses only questions and exact evidence lines already frozen in
`grounded-map-answering-issue35-v2/protocol.json`. It uses Gemini 3.1 Pro
Preview with temperature `0`, seed `42`, high thinking, one candidate, one
attempt per row, and no retry.

It does not use native video or audio, acquire evidence, retrieve additional
map lines, generate multiple answers, rank candidates, implement counting
specialist #34, or create router framework #36.

## Diagnosis being tested

The first treatment succeeded when the answer was explicit and lexically
aligned with the question. It falsely abstained when the requested answer was
supported but an incidental question premise was absent, approximate, or
harmlessly inconsistent with the map. More inference compute is not the
treatment: high thinking was already enabled and produced deliberate,
rule-consistent abstentions.

The policy under test separates:

1. **Locator/premise resolution:** Does the question identify one relevant
   event or entity under ordinary paraphrase and harmless attribute variance?
2. **Requested-answer support:** Is the requested action, state, speech,
   temporal successor, or causal statement explicitly supported by the frozen
   evidence?

A harmless locator mismatch is recorded but does not block an answer. A
mismatch blocks answering only when it changes the requested answer, leaves
multiple plausible referents or successors, or makes the answer contradictory.

## Allowed and forbidden entailments

The answerer may use only these narrow entailments:

- timestamped scene order supports `before` and `after`;
- quoted speech supports what that speaker explicitly says, including a stated
  feeling or reason;
- ordinary paraphrases and object/location aliases are acceptable when they
  still identify one relevant event;
- a directly stated event can answer a broad `what happens` question.

The answerer must not:

- infer a purpose from apparatus mechanics;
- infer a causal reason from mere co-occurrence;
- infer a unique-entity count from repeated appearances;
- overlook two competing candidate answers;
- repair missing map content with world knowledge.

## Response contract

Each response contains:

- `question_interpretation`;
- `premise_status`, one of `supported`, `harmless_locator_variance`, or
  `answer_blocking`;
- `answer_claims`, each with supplied evidence IDs;
- `conflicts_or_missing_qualifiers`;
- `needs_verification`;
- one concise top-1 `answer` of at most 24 words.

For a non-verification response, the final answer must exactly equal one cited
claim. A verification response has no claims and uses exactly
`NEEDS VERIFICATION`. Malformed paid responses are preserved before terminal
classification.

## Frozen 11-row panel

The label-aware panel membership and gate are frozen before inference, but the
provider receives only each question and its frozen evidence lines.

### Five false-abstention repair rows

| Video ID | Boundary under test |
|---|---|
| `D_HTq7IiSPo` | causal/noisy wording: answer observed state and cited speech without demanding a literal table/counter match |
| `KBb0DfsH0eg` | temporal: use the explicit later head-pat scene even though the brag has no precise timestamp |
| `LNb5uNQXkwg` | temporal/state: answer typing and illumination although the turn-on action is only in the question |
| `PWP17NlIgs8` | temporal/noisy premise: use the explicit subsequent yelling despite water/Coca-Cola and kitchen/split-screen wording differences |
| `BDKkVYmWqdM` | harmless locator variance: answer riding across the tennis net despite court/platform wording |

The lion-count row `PTwap56YrbE` is excluded because repeated appearances do
not establish unique identity and the repair belongs to #34.

### Two detectable insufficient-evidence negatives

| Video ID | Required behavior |
|---|---|
| `G2w7PrdwYhM` | verify because the question asks purpose while the map supplies mechanics only |
| `LkdfiZcQ7lU` | verify because the map supplies neither the pink/red composition nor the pencil-removal reason |

### One known map-only limit

`HcvOhX3n784` is inspected but not gated as a required verification route. Its
map supplies a plausible end state—holding or posing with the dog—while
omitting the reference's dancing action. A map-only answerer cannot detect that
the plausible supported answer is incomplete without acquiring evidence or
seeing the label.

### Three answered stability controls

- `Bvv6lMUTlEw` — causal;
- `GWCGF1WpMPc` — temporal;
- `JTXpUxVg144` — other/speech.

These must remain supported, concise answers rather than become abstentions or
unsupported elaborations.

## Freeze and data flow

1. Read and hash the completed v2 protocol and exact 11 selected rows.
2. Write a label-free provider protocol containing only IDs, questions, frozen
   evidence, model settings, and prompt/schema hashes.
3. Write a separate pre-inference gate freeze containing panel roles,
   thresholds, and source hashes. Never include it in a provider request.
4. Execute one immutable call per row and persist request, raw response,
   metadata, parsed response, and terminal state.
5. Freeze terminal predictions and all row artifacts before opening
   annotations, prior predictions, or manual scoring rubrics.
6. Score automatic metrics and manually inspect all changed rows. Preserve
   operational and contract failures in the denominator.

The private root is additive and must not modify either issue #35 v1 or v2:

`data/vqa/benchmarks/grounded-map-answering-issue35-premise-v3/`

## Decision gate

The policy passes this selected mechanism test only if all conditions hold:

- at least 4/5 false-abstention rows produce correct, supported answers;
- both detectable insufficient-evidence rows route to verification;
- all 3 answered stability controls remain correct and supported;
- zero final answers contain unsupported claims;
- every changed row is manually inspected.

The known map-only limit is reported separately and cannot pass or fail the
gate. Automatic METEOR and BERTScore-F1 remain descriptive. Legacy UMBC STS and
the three-metric composite remain `null` unless the exact historical service
becomes available; no substitute metric is presented as comparable STS.

## Interpretation and next action

A pass means the false-abstention mechanism is repairable on this deliberately
tuned development panel. It does not justify deployment; the next step would
be a small fresh frozen holdout sampled from existing maps.

A failure on ordinary noisy-premise rows rejects further generic prompt
tuning. A failure concentrated on missing-anchor temporal rows supports one
explicit question-driven temporal or speech verification operation over frozen
global context. Any unsupported answer rejects the relaxed policy until its
material-mismatch rule is narrowed.

## Verification and reporting

Tests cover panel identity, label isolation, prompt/schema hashing, exact
answer-to-claim binding, harmless versus blocking premise outcomes, raw JSON
preservation, immutable row artifacts, freeze drift, gate arithmetic, known-
limit exclusion, and changed-row audit completeness.

The result—positive or negative—updates the existing dated QMD, routing claim,
sanitized artifact index, issue #35, and draft PR #37. Complete questions,
evidence, labels, responses, and audit rows remain private under the canonical
dataset root.
