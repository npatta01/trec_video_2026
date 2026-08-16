# Counting Coverage Certificate Design

**Date:** 2026-08-14

## Decision

Add a mechanically checkable evidence-adequacy layer in front of exact count
proofs. The layer does not infer objects, events, identities, or answers. It
verifies whether the evidence-review process was strong enough to support the
coverage obligations declared by a `CountQuery`. When it is not, it emits a
deterministic gap manifest that a later local retrieval step can satisfy or
leave unresolved.

This is the smallest coherent continuation of issue #34. The typed reducer is
preserved as the arithmetic layer, and the immutable v1 proof/score plus v2
audit are not replayed or modified.

## Problem

The count-query engine currently accepts a `CoverageClaim` whose
`coverage_state` is `complete`. That claim records a temporal scope, a spatial
scope string, and a declared resolution, but it is not derived from a review
ledger proving which timestamps and image regions were actually inspected.
Consequently, an incomplete graph can authorize an incorrect exact answer.

The four incorrect exact answers in the completed oracle slice demonstrate
three distinct adequacy failures:

1. spatial visibility: relevant shelf locations were occluded;
2. predicate adequacy: a red visual mark was treated as a physical patch;
3. temporal segmentation: brief appearances or transitions were collapsed.

A single sampling-density flag cannot address all three.

## Scope

### In scope

- A strict, label-blind `CoverageCertificate` schema.
- A pure checker that verifies a certificate against a parsed `CountQuery`,
  `EvidenceGraph`, and an immutable inventory of locally available evidence.
- Four independent obligation families:
  - temporal review density;
  - spatial region/tile review;
  - observability and occlusion;
  - predicate-disambiguation evidence.
- A deterministic `CoverageDecision` with satisfied obligations, failed
  obligations, and exact artifact-grounded reasons.
- A deterministic `GapManifest` describing additional timestamps, intervals,
  regions, crops, or predicate comparisons required for another review.
- A conservative composition function that prevents an otherwise exact
  `ProofState` from remaining exact when certificate obligations fail.
- Synthetic unit and integration tests plus a private additive diagnostic that
  audits coverage without rewriting or rerunning the frozen v1/v2 proofs.

### Out of scope

- Provider inference or paid calls.
- Rerunning the Gemini 475-row panel, generic atomic enumerator, or frozen
  12-row count-query proof execution.
- Adding count reducers or row-specific rules.
- Automatically interpreting newly retrieved frames.
- Claiming that finite sampling proves unrestricted real-world absence.
- Issue #35 temporal verification or issue #36 speech verification.
- Promotion based on the already-unblinded development rows.

## Architecture

The evidence flow becomes:

```text
CountQuery ────────────────┐
                          ├─> coverage checker ─> CoverageDecision
EvidenceGraph ─────────────┤                          │
                          │                    sufficient?
EvidenceReviewLedger ──────┘                     /          \
                                             yes            no
                                              │              │
                                     retain proof       GapManifest
                                                             │
                                                   local retrieval/review
                                                             │
                                                    new additive ledger
```

The checker is a pure module. It does not read files, decode videos, call
models, or inspect references. A separate workflow script resolves and hashes
the available local artifacts, builds the ledger, freezes the certificate
decision and gap manifest, and only then permits a scorer to load reference
answers for diagnostic evaluation.

The first implementation stops at the deterministic gap manifest. It may
identify locally satisfiable requests, but it does not pretend that extracting
a frame also interprets its contents. This keeps acquisition, semantic
judgment, and arithmetic as separate trust boundaries.

## Domain model

### EvidenceReviewLedger

An immutable ledger contains:

- video and query bindings;
- source-video duration and hash when available;
- reviewed timestamp samples using decoded presentation timestamps;
- reviewed intervals for continuous or dense review;
- spatial regions using normalized `[x_min, y_min, x_max, y_max]` coordinates;
- artifact IDs and hashes for frames or contact sheets;
- visibility state per reviewed region/interval: `visible`, `occluded`,
  `offscreen`, `blurred`, or `unknown`;
- predicate checks identifying the target distinction, compared alternatives,
  the graph node IDs whose classifications they evaluate, supporting artifact
  IDs, and outcome `supported`, `rejected`, or `unresolved`.

Every referenced artifact must exist in the supplied immutable inventory and
match its recorded hash. Review timestamps must be within video duration.
Normalized regions must be nonempty and remain within `[0, 1]`.

### CoverageRequirement

Requirements are query-derived and frozen before evidence review. They contain:

- obligation family;
- target kind and predicate;
- temporal scope;
- maximum allowed review gap or an explicit every-decoded-frame requirement;
- required normalized regions or whole-frame tiling;
- allowed visibility states;
- predicate alternatives that must be resolved when semantic distinctions can
  change the count.

The checker never invents a minimum event duration from the observed answer.
If a defensible maximum review gap was not frozen, temporal absence cannot
authorize exactness.

### CoverageDecision

The result contains:

- `outcome`: `sufficient` or `insufficient`;
- satisfied and failed requirement IDs;
- selected ledger entry and artifact IDs;
- stable reason codes;
- a canonical hash binding the query, graph, requirements, ledger, inventory,
  and decision.

Failure of one mandatory requirement makes the decision insufficient.

### GapManifest

Each failed requirement produces one or more exact acquisition or review
requests:

- inspect a timestamp interval at a maximum gap;
- inspect a normalized spatial region or crop;
- resolve an occluded/unknown region if another visible interval exists;
- compare the target predicate against named alternatives using cited
  artifacts.

Requests state whether the immutable local inventory can satisfy them. An
unsatisfiable gap remains explicit and prevents exactness.

## Certificate semantics

### Temporal adequacy

A temporal requirement passes only when the union of reviewed intervals and
samples covers its scope with no gap larger than the frozen threshold.
Endpoint handling is deterministic. Contact-sheet display order is not treated
as timing provenance unless every cell is bound to a decoded PTS.

For onset or transition counting, the requirement must also cover neighboring
states around each proposed boundary. If the minimum possible target duration
is unknown and not every decoded frame was reviewed, absence is uncertifiable.

### Spatial adequacy

Required regions must be fully covered by reviewed regions at the required
temporal samples. A free-form claim such as `whole frame` is insufficient
unless represented by normalized geometry. Region union is computed
deterministically.

### Observability

Any required region/interval marked `occluded`, `offscreen`, `blurred`, or
`unknown` fails exact coverage. The certificate cannot infer what exists behind
an obstruction. It emits a gap when another local view could resolve it;
otherwise the gap is unsatisfiable and exactness is prohibited.

### Predicate adequacy

Positive observations counted in a proof must have a supported predicate check
when the requirement declares confusable alternatives. An unresolved visual
mark versus physical patch distinction therefore cannot support an exact
physical-patch count. This obligation validates the existence of a grounded
review judgment; it does not claim that a machine can prove semantic truth.

## Proof composition

The existing `count_video()` result remains unchanged. A new composition
function accepts the parsed query and graph, their `ProofState`, and the
matching `CoverageDecision`, then returns a separately typed
`CertifiedProofState`:

- an abstaining proof remains unchanged;
- an exact proof with a sufficient certificate remains exact;
- an exact proof with an insufficient certificate becomes an abstention;
- its lower bound includes only artifact-grounded, predicate-supported positive
  observations;
- its upper bound is `null` unless a general, mechanically supported capacity
  bound exists;
- the composed trace records the failed certificate requirements and gap IDs.

The function verifies all query, graph, proof, and certificate identity hashes
before composition. Predicate-supported lower bounds are derived from graph
nodes explicitly bound by ledger predicate checks; they are never inferred
from the original exact integer alone.

This layer never changes one exact integer into a different exact integer.
Recovery requires additive evidence review followed by a new, separately named
graph/proof chain in a future authorized experiment.

## Integrity and privacy

- Query, graph, requirements, ledger, inventory, decisions, and gap manifests
  are recursively label-blind.
- Reference answers, correctness, predictions, metrics, panel roles, and
  expected counts are rejected from all pre-score payloads.
- Authoring and audit packets use independent hash ordering, not failure/control
  ordering.
- Frozen issue #34 artifacts remain write-once and untouched.
- New private row-level records live only under the canonical benchmark root.
- Tracked artifacts contain sanitized aggregate decisions, schemas, hashes, and
  no private media or competition rows.

## First diagnostic

The first diagnostic is explicitly developmental because its failure modes are
already known. It does not support promotion.

1. Freeze generic certificate requirements without reading references during
   the checker run.
2. Inventory the existing local inspection artifacts and their hashes.
3. Build a review ledger from actual timestamp/region provenance only; missing
   metadata remains missing rather than being inferred from filenames or
   memory.
4. Audit four previously incorrect exact cases plus two correct exact controls
   selected across spatial and temporal strata.
5. Freeze decisions and gap manifests before scoring.
6. Score only whether wrong exactness was blocked and controls remained exact;
   separately report whether gaps appear locally satisfiable.

The safety target is to block all four known incorrect exact answers while
preserving both controls. Failure to preserve either control or failure to
block any wrong exact answer rejects the certificate contract as currently
specified. Passing this target validates only the safety mechanism on known
development cases. Evidence recovery requires a later held-out test.

## Testing

Unit tests cover:

- strict parsing, unknown fields, duplicate IDs, and label-key rejection;
- artifact hash and binding failures;
- temporal gap boundaries and decoded-PTS grounding;
- spatial union, uncovered tiles, and invalid geometry;
- each non-visible state blocking sufficiency;
- resolved and unresolved predicate alternatives;
- deterministic gap requests and canonical hashes;
- conservative proof composition and bound invariants.

Integration tests use synthetic queries, graphs, ledgers, and inventories. They
prove that a false `complete` graph claim cannot override a deficient
certificate and that adding valid review evidence can satisfy a gap without
changing reducer semantics.

## Failure handling

Malformed or integrity-invalid payloads raise precise validation errors and
produce no decision. Valid but insufficient evidence produces an auditable
`insufficient` decision, not an exception. Missing local artifacts create
unsatisfiable gap requests. No fallback silently converts a filename, prompt
claim, or contact sheet into timestamp or region provenance.

## Success boundary

This work succeeds when the repository has a tested, deterministic mechanism
that prevents ungrounded completeness assertions from authorizing exact counts
and exposes the exact evidence gaps a recovery system must address. It does not
claim that video counting is solved until a future held-out, label-isolated
retrieval test improves rescue without incorrect exact answers.
