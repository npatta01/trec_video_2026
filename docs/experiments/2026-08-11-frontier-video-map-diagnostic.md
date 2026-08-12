# Gemini Native-AV Event-Map Canary


## Decision

Gemini native-video/audio inference works through the corrected
OpenRouter delivery path, and both paid responses contain
strict-schema-valid event/state maps with useful timestamps. This is a
**Diagnostic signal**, not a model or architecture promotion: one top-1
answer added an unsupported detail and the other lost label-critical
sound specificity. Do not run the frozen 35-row panel unchanged. Tighten
answer-to-evidence binding within the same one-pass contract, then test
unused videos across the remaining categories.

The canary cost **\$0.063730 total**. No Reka inference ran because the
supplied credential returned HTTP 403 from the official authenticated
endpoint.

## Question and planned claim

**Question.** Can one question-conditioned Gemini native-AV pass produce
a usable event/state map, key scene timestamps, supporting spans,
uncertainty, and one best-supported direct answer without a separate
transcript, audio pass, or candidate-diversity loop?

**Hypothesis.** A frontier native-video model will map the relevant
transitions and expose better evidence structure than fixed global
frames, making the map a credible input contract for top-1 VQA.

**Falsifier.** The path would not advance if either paid response lacked
a valid, inspectable map; if selected timestamps did not correspond to
real events; or if the final answer remained materially disconnected
from the map.

**Decision rule.** Inspect and freeze the two maps before any larger
paid run. Do not launch the 35-row panel automatically. This operational
rule was predeclared; no numerical two-row quality threshold was
predeclared.

## Tested condition and boundary

- Two frozen VQA-2025 training examples: one sound question and one
  temporal before/after question.
- One content inference per video with `google/gemini-3.1-pro-preview`,
  native video plus native audio, and no independent Whisper or audio
  check.
- The response contract contains question interpretation, temporally
  ordered events and state changes, event keyframes, support spans, a
  top-1 answer, and credible uncertainty.
- Original MKV sources were losslessly remuxed to WebM with stream copy.
  Video and audio packet hashes, codecs, stream counts, duration, file
  identity, and uploaded-byte digest were verified. No decoding,
  re-encoding, clipping, resize, frame-rate change, or audio extraction
  occurred.
- References remained unavailable to inference. Labels were joined only
  after both responses and their terminal records froze.
- Raw media, questions, answers, responses, selected frames, and private
  paths remain under the ignored canonical data root. The tracked bundle
  contains only sanitized aggregates and immutable evidence hashes.

## Operational result

OpenRouter authentication and the Google model route were live. The
initial MKV requests failed before content generation because MKV is not
an accepted video-input container on that route. Lossless WebM delivery
fixed that boundary.

Native provider-side JSON schema translation remained unreliable: a
schema was rejected while claiming a declared property was unspecified.
The working path therefore uses prompt-JSON plus strict local validation
and permits no semantic content repair. The first paid response was
preserved but initially marked `invalid_json` because Gemini enclosed
the JSON in an exact Markdown fence. A deterministic parser now removes
only that exact outer envelope. The second paid response then completed
end to end. Both response bodies validate against the same strict local
event-map schema.

| Observation                                           |     Result |
|-------------------------------------------------------|-----------:|
| Paid Gemini responses                                 |          2 |
| Strict-schema-valid maps after exact-envelope parsing |          2 |
| Events represented                                    |          7 |
| Model-selected keyframe timestamps                    |         15 |
| Input tokens                                          |      5,471 |
| Output tokens                                         |      4,399 |
| Total tokens                                          |      9,870 |
| Observed cost                                         | \$0.063730 |

The private audit packet validates and contains two six-frame
whole-video overview sheets plus all 15 event-keyframe JPEGs, each bound
to its requested timestamp and source-media hash.

## Quality result

The temporal map localized the trigger action, the subsequent kiss, the
person appearing, and the final embrace. Its answer retained the
reference’s core outcome, but added a claim about placing a rose in the
other person’s mouth. Neither the cited 12s and 14s keyframes nor four
additional checks from 12.25s through 13.75s show that action. This is
at least an answer-to-citation failure; the available still-frame audit
does not mathematically exclude a briefer intervening gesture.

The sound map identified a cat vocalization and selected frames showing
the cat opening its mouth. Per the canary constraint, no independent
audio audit was performed. Post-freeze comparison found that its
one-word answer was too generic relative to the reference’s
screaming-like distinction. The temporal row thus said too much, while
the sound row said too little: brevity alone is not the fix.

| Post-freeze metric     |                         Two-row mean |
|------------------------|-------------------------------------:|
| METEOR                 |                               0.2532 |
| BERTScore-F1           |                               0.8949 |
| STS                    |                               0.4392 |
| Semantic composite     |                               0.5291 |
| Token F1               |                               0.1739 |
| Normalized exact match |                               0.0000 |
| NDCG                   | 1.0000 (degenerate; not informative) |

The temporal semantic composite was 0.7225 and the sound composite was
0.3357. These two selected rows do not estimate distributional
performance. METEOR and token overlap also penalize answer/reference
length mismatch, so the sound score’s magnitude should not be treated as
a pure perception measure.

## Interpretation and advisor review

The result supports the narrow claim that a single frontier native-AV
pass can produce useful question-conditioned event/state maps and
timestamps. It does not establish that the map improves answer accuracy,
that Gemini used native audio rather than visual cues, or that the
current top-1 answer field is safely grounded.

Sol, Fable, and Gemini Pro independently reviewed the same sanitized
evidence packet. All three rejected both running 35 rows unchanged and
abandoning event maps. They identified **answer–evidence decoupling** as
the primary mechanism: the map found useful evidence, but the answer was
free to embellish or collapse important specificity. Fable additionally
cautioned that a merely “minimal” answer rule would worsen the sound
failure.

## Current condition-bound status

- **Evidence validity:** Diagnostic.
- **Decision disposition:** Diagnostic signal; not promoted as the
  panel-scale answer path under the tested contract.
- **Gate outcome:** Operational map viability passed after deterministic
  envelope handling; answer-grounding quality remains insufficient for
  scaling unchanged.
- **Completion denominator:** 2/2 paid response bodies produced valid
  maps; 1/2 was accepted by the runner at call time and 1/2 was
  recovered from the frozen exact-fenced response after the parser
  defect was identified.
- **Confirmed defect:** the preflight freeze formerly omitted
  raw-response-file hashes for failed-but-billable responses. The code
  now binds every response with a response hash, regardless of terminal
  status.
- **Uncertainties:** selected panel, no matched direct-answer-only
  control, provider audio use unattested, and continuous micro-actions
  not exhaustively reviewed.

## What this rules out now

- Treating Gemini, OpenRouter authentication, or native video delivery
  as categorically nonfunctional.
- Running the existing 35-row contract unchanged merely because the
  per-video cost is low.
- Concluding from the earlier transport/schema failures that event maps
  are the wrong abstraction.
- Treating answer brevity by itself as the grounding solution.

## What it does not rule out

- A tightened one-pass map-plus-answer contract.
- A later multi-stage production architecture whose stages expose
  genuinely different evidence rather than prompt/seed diversity.
- Direct Gemini API or another frontier native-video provider if routed
  audio behavior remains uncertain.
- A matched direct-answer-only control after the grounded contract is
  stable.

## Next bounded experiment

Create contract v2 without rerunning these two videos:

1.  Express the top-1 answer as atomic claims, each linked to event IDs
    and a supporting time span from the same response.
2.  Require one direct, maximally specific, fully supported answer;
    unsupported detail must move to uncertainty rather than the answer.
3.  Add sound-event attributes such as vocalization quality, intensity,
    and onset/offset so native-audio specificity is behaviorally
    testable.
4.  Automatically extract every cited keyframe/span for review.
5.  Run one previously unused frozen video from each of the five
    categories not represented here. Combine those with this canary only
    for architecture coverage, not as a matched prompt comparison.
    Expand to three per category only if schema validity holds and cited
    claims are supported.

This keeps one Gemini content pass per video and tests new category
coverage before spending calls on a larger panel.

## Evidence of record

The compact tracked bundle is
[`artifacts/frontier-video-map-20260811/`](artifacts/frontier-video-map-20260811/).
It contains sanitized completion, metric, cost, and immutable-hash
records. The private run and audit locations are documented in the
repository `AGENTS.md` for access from every shared checkout.
