# Blind Structured Frame Counting Design

**Date:** 2026-08-16

## Decision

Prepare the smallest model-facing counting path that removes human-authored
evidence from inference. The first supported unit is one target's repeated
**visible appearance episodes**. Code owns frame extraction, coverage,
validation, episode reduction, and abstention. A future authorized model call
may only assign a state to each explicitly supplied frame.

This design does not authorize Gemini, GPU, provider, or paid inference. It
prepares and tests the zero-cost interfaces required for a later fresh,
label-blind canary. It does not rerun any frozen issue #34 condition.

## Why this is a new seam

The hardened Gemini workflow requested 12 FPS but still merged two appearances,
omitted the final appearance, and falsely certified full-video coverage. The
post-label feasibility audit showed that an endpoint-inclusive fixed frame grid
can represent the intended four episodes. Therefore, another native-video
prompt is not the next test. The project must explicitly deliver and bind the
frames whose states the model classifies.

The existing `vqa_count_frame_presence` reducer remains the arithmetic and
coverage layer. This design adds only the missing model-facing preparation,
response, and challenge contracts.

## Scope

### In scope

- A strict, label-blind visible-appearance `CountSpec`.
- A deterministic request manifest binding every supplied frame ID, decoded
  index, PTS, artifact hash, and derived presentation artifact.
- A strict response parser requiring exactly one
  `present`/`absent`/`uncertain`/`unobservable` decision per required frame.
- A deterministic challenger selector.
- A coverage certificate that composes the request, response, challenge, and
  existing frame-presence reducer result.
- Synthetic fixtures and a zero-network dry-run CLI.
- A frozen future-canary protocol and go/no-go rule, without selecting private
  rows or accessing references during inference preparation.

### Out of scope

- Any model, provider, GPU, or paid call.
- Human review as an inference input.
- Simultaneous distinct-object cardinality, action-cycle counting, visible-part
  ownership, or identity through occlusion.
- Reusing the unblinded four-appearance row as scored blind evidence.
- Rerunning or modifying frozen counting artifacts.
- Issue #35 or #36 work.

## Architecture

```text
question-only CountSpec
          |
          v
decoded-frame manifest ---> presentation manifest
          |                         |
          +------------+------------+
                       v
              model decision request
                       |
                       v
       exact one-decision-per-frame validator
                       |
                       v
              preliminary presence runs
                       |
                       v
 deterministic challenge selection
 (endpoints, transitions, run/gap interiors)
                       |
                       v
          verified decisions or abstention
                       |
                       v
       existing deterministic episode reducer
                       |
                       v
              coverage certificate
```

The model never emits temporal ranges or a final integer. Native video may be
used in a future question-understanding stage, but it is not evidence of frame
coverage. The counting request uses explicit numbered images only.

## Supported CountSpec

The strict schema contains:

- `unit = visible_appearance_episode`;
- an observable target description derived from the question only;
- full-video temporal scope;
- `episode_rule = maximal contiguous run of present samples`;
- a frozen maximum PTS gap or `every_decoded_frame = true`;
- an explicit cut policy: `continuous_across_cuts` or `split_at_cuts`;
- the four allowed decision states;
- an explicit statement that occluded identity is unsupported.

Unknown fields and label-like keys are rejected recursively. Any other count
unit produces a typed `unsupported` result rather than being forced into the
presence reducer.

The implementation accepts the original question, derives an opaque spec ID
from its SHA-256, and derives the target only through the supported question
grammar. Callers cannot supply target text or identifiers that could encode a
reference answer, failure/control role, or other post-label information.

## Request and presentation contract

The request builder consumes a `CountSpec` and immutable decoded-frame
inventory. It emits an ordered manifest containing the first and last decoded
frames, every required intermediate frame, actual PTS, artifact hashes, and the
maximum observed gap.

Presentation artifacts in this smallest implementation may be numbered contact
sheets for global context or full-resolution individual frames for
verification. Every tile or individual image must map back to source frame
hashes. Triplets are deferred until their overlapping membership has a separate
strict schema. Burned-in numbers and timestamps
must appear in a margin rather than cover source pixels.

Every response names the exact presentation IDs it received. Those IDs bind
the rendered byte hashes, sizes, mode, and source-frame membership. Primary
and challenge modes must use different bytes, not merely different mode names.
The zero-network CLI recomputes every referenced artifact hash and size from a
content-addressed artifact root before building the request.

The model receives no references, previous predictions, correctness, expected
count, or panel role. It must return one enum state for every manifest frame ID.
Missing IDs, duplicate IDs, extra IDs, invented timestamps, or unknown states
invalidate the response and force abstention. No permissive partial parser is
allowed.

## Challenger contract

For a canary with at most 100 required frames, challenge every frame using a
different presentation: the first pass may use contact sheets, while the
challenge uses full-resolution individual frames.

For a later larger request, a predeclared selector must include:

- absolute first and last frames;
- both sides of every predicted state transition;
- one neighboring frame on each side when available;
- at least three deterministic interior samples from every predicted present
  run and absent gap.

The same model on a different rendering is only partially independent.
Agreement is necessary, not proof of correctness. Any first-pass/challenge
disagreement becomes `uncertain` and therefore makes the existing reducer
abstain. There is no majority vote and no third pass that converts disagreement
into exactness.

## Coverage certificate

The certificate binds canonical hashes for:

- `CountSpec`;
- source and decoder identity;
- ordered frame inventory and true PTS;
- presentation-to-frame membership;
- exact request and response;
- challenger selection and response;
- final reconciled ledger;
- interface and reducer source hashes, source revision, and result.

The source binding covers the complete local trust base used by the dry run,
including the interface, reducer, label-isolation helpers, serializer, CLI,
project metadata, and lockfile. Certification reruns the structural validators
from the normalized objects; self-consistent hashes alone are insufficient.

It verifies endpoint inclusion, the maximum PTS gap, one-to-one decision
coverage, all mandatory challenge records, and abstention on unresolved states.
It proves delivery and structural completeness only. It does not claim that a
model perceived the target correctly.

The dry-run freeze preserves the normalized CountSpec, request, primary
response, challenge response, reconciled ledger, reducer result, and
certificate as separate write-once JSON files so every certificate binding can
be reconstructed locally. Input JSON rejects duplicate keys. An exclusive
filesystem lock covers conflict checking and publication, all files are staged
first, and the certificate is published last as the terminal marker.

## Public interfaces and agreed test seams

Tests exercise public behavior at these seams:

1. `parse_visible_episode_count_spec(payload) -> VisibleEpisodeCountSpec | UnsupportedCountSpec`
2. `build_frame_decision_request(spec, inventory, presentations) -> FrameDecisionRequest`
3. `parse_frame_decisions(request, payload, *, required_frame_ids=None) -> FrameDecisionSet`; the default requires every request frame, while a large-request challenge passes the exact deterministic selector output.
4. `select_challenge_frames(request, decisions) -> tuple[str, ...]`
5. `reconcile_frame_decisions(request, primary, challenge) -> FramePresenceLedger`
6. `certify_frame_presence_result(...) -> FramePresenceCoverageCertificate`
7. A zero-network CLI that reads synthetic JSON files and freezes the canonical
   CountSpec, request, both response passes, reconciled ledger, reducer result,
   and certificate.

Tests assert observable return values and reason codes rather than private
helpers. Each production behavior is added through a red-green test cycle.

## Failure handling

- Unsupported or ambiguous count unit: abstain before frame classification.
- Missing endpoint, frame, PTS, hash, or presentation binding: validation
  error; no request freeze.
- Missing, duplicate, extra, or invalid model decision: abstain.
- `uncertain` or `unobservable`: abstain.
- First-pass/challenge disagreement: abstain.
- Excessive PTS gap: abstain.
- Unsupported cut or occlusion semantics: abstain.
- Any label-key leakage: reject before writing an inference request.

No fallback creates an exact answer from prose, repairs malformed model JSON,
or trusts a model-declared completeness statement.

## Zero-cost implementation verification

Synthetic fixtures cover:

- two appearances separated by one absent frame;
- an appearance at the first or last frame;
- a falsely continuous run whose interior challenge disagrees;
- missing/duplicate/extra decisions;
- uncertainty and unobservability;
- unsupported units;
- excessive gaps and missing endpoints;
- cut-splitting policy;
- hash and presentation-membership mutation;
- recursive label-key leakage;
- canonical byte-identical dry-run replay.

No private row is needed to prove these mechanics.

## Future blind canary, separately authorized

Before requesting inference authorization, run a question-only eligibility
audit. If fewer than five untouched failures and three matched controls use the
supported visible-appearance unit, do not run this canary.

Otherwise freeze ten rows before reference access:

- five untouched eligible failures;
- three matched eligible correct controls;
- two unsupported-unit routing sentinels expected to abstain.

The already-unblinded feasibility row is excluded from scoring. References and
prior predictions remain unavailable through request construction, inference,
response validation, reconciliation, and prediction freeze.

The predeclared go rule is:

- at least four of five failures rescued exactly;
- all three controls preserved exactly;
- both unsupported sentinels abstain;
- zero incorrect non-abstained answers;
- every exact answer has a valid coverage certificate.

A pass would justify a larger visible-appearance replication only. It would not
promote the method for all count types.

## Success boundary

This implementation succeeds when a future model can be connected only through
a strict, fully covered frame-decision interface and the code deterministically
returns an exact visible-episode count or a reasoned abstention. The zero-cost
implementation itself is infrastructure and protocol evidence, not a counting
quality result.
