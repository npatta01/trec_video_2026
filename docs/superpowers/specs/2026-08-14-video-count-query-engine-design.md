# Video Count Query Engine Design

**Date:** 2026-08-14

**Status:** Approved for implementation

**Parent result:** issue #34 atomic counting canary, draft PR #38

## Outcome

Build a coherent counting core that treats video counting as a typed query over
temporal evidence. The core accepts one `CountQuery` and one `EvidenceGraph`,
then returns a `ProofState` containing an exact count or bounded abstention,
selected evidence, coverage findings, and reasons. It does not call a model,
inspect labels, or contain row-specific reducers.

This is the first subproject of the broader video-counting system. Automated
question compilation, detection, tracking, part estimation, action proposal,
and cross-shot re-identification remain replaceable evidence producers outside
this implementation. They will be evaluated only after the core abstraction
passes a zero-provider-cost vertical slice.

## Why the previous mechanism is insufficient

The completed canary asked one model to choose the count unit, inspect the
video, enumerate records, and implicitly choose aggregation semantics. A local
cardinality reducer then counted the records. Six of six schemas were valid,
but only one of five failed answers was rescued.

The failure inspection separates distinct layers:

- question semantics: distinct people versus maximum simultaneously visible;
- ontology and visibility: physical parts versus visible pixels;
- temporal segmentation: four edited appearances collapsed into two spans;
- scope: an action already underway at clip start counted as a new onset;
- perception recall: a background person was omitted or recovered.

Adding one special reducer for each row would preserve this conflation. The new
design instead creates one stable seam between question meaning and temporal
evidence, then executes a small compositional algebra over that seam.

## Approaches considered

### Selected: compiled query over a shared temporal graph

A typed `CountQuery` expresses counting semantics. An `EvidenceGraph` expresses
observations, identity hypotheses, temporal structure, and search coverage. A
deterministic executor returns a `ProofState` with count bounds.

This approach is selected because the same graph and operators support entity,
appearance, simultaneity, part, and event-transition counts. Learned models may
populate the graph later without gaining authority over arithmetic or proof
completion.

### Rejected for this phase: end-to-end learned counter

An end-to-end model would be simpler at inference, but the current 66-row
population is too small to train or validate it. It would also make semantic,
perceptual, identity, and aggregation errors difficult to distinguish.

### Rejected: router over specialist counting scripts

Separate hand, shot, person, and action pipelines without a shared
representation would be fast to prototype but would create the patchwork this
design is intended to avoid. Specialist evidence producers are allowed only as
adapters that emit the common graph schema.

## Deep module and seam

The external module is `count_video(query, graph) -> ProofState`.

Its interface includes:

- exact typed schemas and schema versions;
- deterministic validation and error modes;
- no filesystem, model, network, or label access;
- stable ordering and canonical serialization;
- an exact result only when every interpretation and coverage obligation
  supports the same singleton count.

The implementation hides filtering, temporal relations, identity quotienting,
interval overlap, transition counting, evidence selection, bound propagation,
and abstention logic. Tests and callers use the same seam.

## `CountQuery`

`CountQuery` is a versioned declarative query, not a collection of prompt flags.
It contains the following fields.

### Target

- `target_kind`: `entity`, `part`, or `event`.
- `target_predicate`: a stable symbolic predicate such as `person`, `finger`,
  or `pass`. The executor matches it exactly against graph node predicates.
- `owner_predicate`: optional for part queries, such as parts owned by a hand.

### Count unit

- `physical_entity`: count identity-equivalent entity nodes once.
- `appearance`: count qualifying appearance episodes; shot boundaries may
  separate appearances without asserting different physical identity.
- `visible_instance`: count a qualifying visible node at an instant.
- `part`: count qualifying part nodes under an explicit ownership policy.
- `event_onset`: count qualifying inactive-to-active transitions.
- `active_event`: count qualifying event intervals.

### Temporal scope

- `scope_start_s` and `scope_end_s`: inclusive clip-relative interval.
- `boundary_policy`:
  - `starts_within`: the unit begins inside the scope;
  - `overlaps`: any overlap with the scope qualifies;
  - `completes_within`: the unit ends inside the scope.

An event active at `scope_start_s` is left-censored. It qualifies for
`overlaps`, but not for `starts_within` unless an onset is explicitly observed
at the boundary.

### Identity and edit policy

- `identity_policy`:
  - `physical`: quotient entity observations by a confirmed physical identity;
  - `within_shot`: quotient only within a shot;
  - `appearance_episode`: do not merge qualifying episodes across cuts.
- `cross_shot_unknown`: `abstain` or `separate`. The default is `abstain` for
  physical-identity counts and `separate` for appearance counts.

A cut is an observation discontinuity. It never forces physical identity to
merge or split.

### Visibility and aggregation

- `visibility_policy`: `visible`, `partially_visible`, or `amodal`.
- `aggregation`:
  - `cardinality`: count qualifying units;
  - `distinct`: count identity equivalence classes;
  - `max_concurrent`: maximum overlap of qualifying visible intervals;
  - `transition_count`: count qualifying event onsets.

Invalid combinations are rejected, such as `max_concurrent` over
`event_onset` or `transition_count` over entities.

### Interpretation alternatives and obligations

One request contains one or more `QueryInterpretation` values. Multiple values
represent genuine ambiguity rather than a silent compiler choice. Each
interpretation declares `EvidenceObligation` values such as:

- scope interval searched at the required temporal resolution;
- shot boundaries covered;
- target predicate searched;
- part visibility searched;
- event boundary state resolved;
- cross-shot identity resolved when physical identity is required.

## `EvidenceGraph`

The graph is a versioned, canonical JSON-compatible value with no embedded
reference answer or correctness fields.

### Temporal structure

- video duration;
- non-overlapping ordered shots;
- optional edit-boundary confidence;
- time intervals use seconds and satisfy `0 <= start <= end <= duration`.

### Evidence nodes

- `EntityObservation`: predicate, interval, shot, visibility, provenance, and
  optional identity hypothesis.
- `PartObservation`: predicate, owner node, interval, visibility, and
  provenance.
- `EventObservation`: predicate, interval, onset status, completion status,
  participants, and provenance.

Every node has a unique stable ID. Provenance contains timestamps and source
identities but no label-derived assertions.

### Identity relations

Identity is an explicit relation with `relation_state` `confirmed_same`,
`confirmed_different`, or `unresolved`. The executor computes equivalence
classes only from `confirmed_same`. An unresolved relation can widen an upper
bound or force abstention; it cannot be silently merged.

### Coverage

`CoverageClaim` states what was searched, where, and at what granularity. It
contains target kind/predicate, interval, spatial region, temporal resolution,
and `coverage_state` `complete` or `partial`. Detector confidence is not
coverage.

Absence of a node is evidence of absence only inside a matching complete
coverage claim.

## Deterministic algebra

The executor uses a fixed set of general operations:

1. validate schemas and compatible query fields;
2. filter nodes by target predicate, visibility, temporal relation, and scope;
3. project observations to the declared count unit;
4. quotient by the declared identity relation;
5. apply `cardinality`, `distinct`, `max_concurrent`, or `transition_count`;
6. calculate lower and upper bounds from unresolved identity and incomplete
   coverage;
7. compare all query interpretations;
8. produce an exact count only when every interpretation yields the same
   singleton bound and all obligations pass.

The operator set is versioned. Adding a new operator requires evidence that a
question class cannot be expressed compositionally; a single failed row is not
sufficient justification.

## `ProofState`

The result contains:

- `outcome`: `exact` or `abstain`;
- `lower_bound` and `upper_bound`, where an unknown upper bound is `null`;
- `exact_count`, present only for `exact`;
- one trace per interpretation;
- selected node and relation IDs;
- satisfied and failed evidence obligations;
- stable abstention reasons such as `ambiguous_interpretation`,
  `incomplete_coverage`, `unresolved_identity`, or `unresolved_boundary_state`;
- canonical result hash.

If the competition requires a point answer, a later answer adapter may choose a
best estimate while retaining the internal abstention. That policy is outside
the counting core.

## Data flow

```text
question text
    -> replaceable compiler
    -> CountQuery

video
    -> replaceable evidence producers
    -> EvidenceGraph

CountQuery + EvidenceGraph
    -> deterministic count_video module
    -> ProofState
    -> exact answer or explicit unresolved state
```

The first implementation begins at the `CountQuery + EvidenceGraph` seam.
Neither the question compiler nor automated video perception is required to
validate the seam.

## Zero-cost vertical slice

The implementation is tested in three layers.

### Synthetic operator tests

Small graphs prove all operators and failure modes:

- the same four entity observations produce four distinct physical entities
  but a maximum concurrency of two;
- four appearance episodes across cuts produce four appearances without
  asserting four physical identities;
- an event active at clip start is excluded by `starts_within` and included by
  `overlaps`;
- confirmed-same observations deduplicate while unresolved identity abstains;
- incomplete target coverage prevents an exact result;
- alternative interpretations that disagree force abstention;
- visible versus amodal part policy changes only through explicit graph facts.

### Existing-artifact adapter

An internal adapter converts the already-purchased atomic records into graph
nodes without changing them. It demonstrates that evidence producers can vary
behind the common schema. The adapter cannot claim complete coverage merely
because a model returned a complete-looking list.

### Reference-blind oracle slice

After the core and schemas are frozen, select 12 held-out rows from the existing
panel: eight confirmed failures and four correct controls, balanced across
question semantics and hash-ranked within each stratum. Selection may use the
frozen panel role, but the builder receives only a recursively label-free packet
containing IDs, questions, media identities, durations, and navigation maps.
Construct query and evidence values from question/video only, freeze them, and
reveal references only in a separate scorer. The five already inspected
mechanisms are the design set and cannot enter the held-out evaluation. The
evaluation rows cannot change after unblinding.

The architecture gate is:

- every independently answerable evaluation question is expressible without a
  row-specific field or operator;
- no post-unblinding schema or query changes;
- no incorrect non-abstained result;
- at least 9/12 evaluation rows yield a singleton exact count;
- all correct controls remain exact and correct.

Failure rejects or revises the system seam before automated perception work.

## Validation and error handling

- Unknown fields, unknown enum values, duplicate IDs, invalid intervals,
  overlapping shots, dangling relations, cyclic part ownership, and invalid
  query/operator combinations are hard validation errors.
- Missing evidence produces bounds or abstention, not a fabricated zero.
- Serialization is canonical and self-hashed.
- Labels, references, correctness, expected counts, and metrics are rejected
  recursively from query and graph payloads.
- The executor is pure and deterministic.

## Testing strategy

- Test-driven development: every production behavior begins with an observed
  failing test.
- Property-style invariants with deterministic fixtures:
  - lower bound never exceeds upper bound;
  - adding a confirmed observation cannot lower cardinality;
  - adding a confirmed-same relation cannot increase a distinct count;
  - max concurrency never exceeds qualifying interval count;
  - exact implies a singleton bound and no failed obligations.
- Regression tests cover the existing label-isolation validator and atomic
  evidence adapter.
- The relevant existing issue #34 and submission suites remain green.

## Provenance and privacy

Private row-level queries, graphs, proof states, media paths, and references
remain under the canonical ignored benchmark root. Tracked artifacts contain
only schema documentation, aggregate outcomes, hashes, and sanitized failure
classes. The existing 475-row Gemini panel is never rerun or modified.

## Scope boundary

This implementation includes:

- typed immutable domain values;
- strict validators and canonical serialization;
- pure deterministic executor and proof traces;
- adapter for existing atomic evidence;
- synthetic and reference-blind offline validation tooling;
- sanitized documentation and results.

It excludes:

- new paid inference;
- learned question compilation;
- automated object detection, tracking, pose/part estimation, action
  recognition, or cross-shot re-identification;
- issue #35 speech/temporal verification;
- issue #36 appearance-aware answer synthesis;
- a production-scale reusable graph for all 475 videos.

## Architecture falsifier

Reject this version of the architecture if a clearly answerable held-out
question cannot be expressed with the frozen fields and operator algebra, or if
the executor returns the wrong count from a reference-blind graph that
independent inspection finds complete and correct. Do not repair that failure
with a row-specific field or operator.
