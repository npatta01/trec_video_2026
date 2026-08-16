# Gemini Native-Video Counting Recovery Canary Design

**Date:** 2026-08-15

## Decision

Test whether a question-driven, multi-pass Gemini native-video workflow can
recover brief counting evidence that the completed generic atomic enumerator
and typed reducer missed. The workflow uses the complete video only to locate
candidate intervals, then sends timestamp-clipped high-frame-rate native-video
views to verify atomic instances, and finally challenges the proposed count for
omissions, duplicates, ambiguity, and uncovered time.

This is a three-row developmental canary with a hard **$1.00 total provider-cost
ceiling**. It does not rerun the 475-row panel, the frozen generic enumerator, or
the frozen count-query proofs. A positive result can justify a later isolated
held-out test; it cannot promote the method by itself.

## Question and falsifier

**Question:** Does question-driven localization followed by timestamp-clipped,
high-FPS Gemini verification recover atomic counting instances while preserving
a matched correct temporal control?

The hypothesis is falsified if either known temporal failure is not corrected,
the control changes from correct to incorrect, the final count lacks a complete
timestamped instance ledger, any mandatory interval remains unreviewed, or the
run cannot complete below the authorized ceiling.

## Frozen panel

The private preparation selects exactly three existing developmental rows:

- two known temporal failures representing brief appearance episodes and
  repeated action transitions;
- one correct temporal control representing a discrete transfer event.

Selection metadata may contain panel roles, but inference requests may contain
only the video binding, question, neutral counting contract, and artifacts
created by earlier label-blind stages of this canary. Reference answers,
correctness, previous predictions, expected counts, and panel roles remain
unavailable until all three final results are frozen.

Because these rows and their failure modes are already known to the experiment
designer, the result is explicitly developmental and cannot estimate
generalization.

## Three-pass workflow

### Pass 1: locate over the complete native video

Gemini 3.1 Pro Preview receives the original complete native video and the
question. The video is supplied at 4 FPS, matching the established direct
Gemini path and avoiding the provider's default 1-FPS sampling. Gemini returns:

- the counting unit in plain observable terms;
- mutually exclusive candidate intervals;
- a short reason for every interval;
- uncertain or potentially missed intervals;
- a partition of the complete source duration into reviewed and unresolved
  spans.

This pass does not return the final answer and cannot declare coverage
complete. Its candidate intervals are proposals, not evidence certificates.

### Pass 2: verify timestamp-clipped native-video bursts

The runner canonicalizes and unions the proposed intervals, then creates one
verification envelope from the earliest proposed start through the latest
proposed end. It adds 1.5 seconds of source-time padding on both sides and clips
the envelope to the source duration. Gemini receives that relevant subset
through actual `VideoMetadata(start_offset, end_offset, fps)` bounds at 12 FPS
and high media resolution. Prompt text also states the absolute source-time
bounds so returned timestamps remain in source coordinates. An empty or
malformed candidate set falls back to the complete duration rather than
silently skipping evidence.

For every burst, Gemini must emit an atomic ledger. Each proposed instance has:

- stable instance ID;
- start, peak, and end source timestamps;
- observable before/during/after state;
- identity or episode-separation evidence;
- target-predicate support;
- confidence and an explicit ambiguity flag.

No free-form integer is accepted as the evidence result. Static spatial crops
and object tracking are outside this canary because the panel is deliberately
limited to temporal counts.

### Pass 3: challenge coverage and identity

candidate intervals, and the verified atomic ledger. It is instructed to seek
counterexamples rather than agree with the ledger:
Gemini receives the complete native video at 12 FPS, the question, the frozen
candidate intervals, and the verified atomic ledger. The full-duration
high-FPS view is the protection against a locator miss. Gemini is instructed to
seek counterexamples rather than agree with the ledger:
candidate intervals, and the verified atomic ledger. It is instructed to seek
counterexamples rather than agree with the ledger:

- missed instances outside candidate intervals;
- two ledger entries representing one continuous episode;
- one entry containing multiple distinct transitions;
- target-predicate or actor mismatch;
- boundary events hidden by clip edges;
- unresolved time requiring another burst.

The challenger may request additional timestamp bounds. At most one additive
verification round is allowed per row; it uses the same 1.5-second padding and
12-FPS contract. If unresolved intervals remain afterward, the row abstains.

## Deterministic result construction

Code, not Gemini prose, constructs the final result. It validates timestamps,
source bounds, stable IDs, interval ordering, before/during/after states, and
challenge dispositions. Accepted atomic instances are counted only when:

1. the verifier marked the predicate supported and the instance unambiguous;
2. the challenger did not identify an unresolved duplicate, split, or mismatch;
3. all challenger-requested bursts were completed; and
4. the complete-duration 12-FPS challenge has no unresolved interval or
   boundary ambiguity.

Otherwise the result is an auditable bounded abstention. The runner never
changes an uncertain result into an exact integer merely to produce an answer.

## Treatment, control, and held-fixed variables

The treatment is the three-pass workflow. The matched baseline is the already
frozen direct Gemini native-AV answer for the same row; it is loaded only after
the new final-result freeze. Model family, original media, questions, and
semantic scoring remain fixed. The new factors are question-driven
localization, clipped 12-FPS verification, atomic ledger structure, and an
adversarial coverage pass.

The primary outcome is exact numeric correctness or justified abstention.
Secondary outcomes are failure rescue, control preservation, atomic-instance
precision on inspection, timestamp grounding, completeness disposition, cost,
and METEOR, BERTScore-F1, STS, and their established composite for any concise
answer rendered from the frozen count.

## Decision rule

The mechanism passes this developmental canary only if:

- both temporal failures are rescued with correct exact counts;
- the control remains correct;
- all three results have valid timestamped atomic ledgers and complete
  challenge dispositions;
- manual inspection finds no unsupported or merged/split accepted instance;
- no row relies on an unresolved coverage claim; and
- total provider cost is at most $1.00.

Any miss, control regression, invalid evidence chain, or budget stop yields
**Not promoted under the tested condition**. Passing yields only permission to
design a fresh label-isolated held-out canary; it does not authorize scale-up.

## Cost and execution safety

The direct Google Gemini API is the only provider route for this test. There is
no silent OpenRouter fallback. Before the first paid request, the runner:

- verifies the exact model identity and credential availability without
  printing credentials;
- freezes the three-row request manifest and source hashes;
- estimates the worst-case remaining calls using the greater of current
  pricing and the highest observed per-call cost plus a safety margin;
- obtains an exclusive run and per-call claim; and
- refuses a call when projected cumulative cost could exceed $1.00.

Passes execute sequentially so later calls occur only when earlier artifacts
are valid. Provider errors are frozen. No paid call is automatically retried.
The canary stops immediately on a budget, integrity, authentication, schema,
or model-identity failure.

## Provenance, privacy, and freezing

Complete private evidence lives under a new write-once directory in the
canonical benchmark root. It includes request manifests, source and delivered
media hashes, exact start/end/FPS metadata, raw provider responses, parsed
records, usage and cost, call claims, validation results, final predictions,
and post-freeze scores. Every stage binds its input and source revision by
hash.

Tracked artifacts contain only sanitized aggregate results, schemas, reason
codes, hashes, costs, and no competition questions, answers, private media, or
raw responses. The existing issue #34 v1/v2 roots remain untouched.

## Verification

Tests must cover:

- recursive label-key rejection for every pre-score payload;
- exact construction of complete-video 4-FPS and clipped 12-FPS video parts;
- padding, clipping, interval merge, absolute timestamp conversion, and source
  duration boundaries;
- malformed, duplicate, overlapping, and out-of-range atomic records;
- challenger additions, deduplication, splits, mismatches, and abstention;
- uncovered-duration gaps preventing exactness;
- exclusive claims, no automatic retry, immutable stage freezes, and complete
  hash-chain authentication;
- conservative worst-case budget refusal before a call;
- synthetic end-to-end success, abstention, provider failure, and budget-stop
  paths without network access.

The cheapest live check is the appearance-episode failure executed
sequentially. It must finish with a valid provenance chain, a complete
timestamp ledger, and cumulative cost no greater than $0.34 before either
remaining row may run. After two rows, cumulative cost must be no greater than
$0.67. These proportional gates reserve enough budget to finish or stop safely;
the global $1.00 gate remains authoritative.

## Reporting boundary

The completed result extends the existing counting claim in the frontier
boundary-verifier report. Observations, evidence validity, and decision
disposition remain separate. A negative or incomplete result is preserved.
Issue #34 receives a concise status with completion denominator, rescue and
control rates, evidence-integrity result, metric summary, cost, and the next
decision. No issue #35 speech or issue #36 general temporal-answering work is
included.
