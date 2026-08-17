# Grouped-Rewrite Fallback Repair Design

## Objective

Reduce the `g37_grouped_unique` fallback count from 610 while preserving its
core safety properties: the existing fusion rank-one answer remains byte-exact,
no reference answers or labels enter the workflow, every source artifact stays
immutable, and every unresolved question falls back to the already validated
`g37_q36_fusion` row.

The repair has two independently auditable treatments:

1. deterministically repair the 320 current-schema question failures without
   generating new text; and
2. issue fresh Gemini calls only for the 290 questions whose earlier calls used
   an incompatible schema or received no provider response.

## Reconciled starting state

The existing 610 fallbacks have these exact causes:

| Cause | Questions |
|---|---:|
| Earlier paid responses use incompatible group/answer schemas | 280 |
| Earlier request received provider HTTP 400 with no usable response | 10 |
| Current-schema answer repetition after case/whitespace normalization | 314 |
| Current-schema group omitted an exact source answer | 5 |
| Current-schema rank/group representation was invalid | 1 |

The existing salvage already retains 1,273 accepted questions and independently
recovers 560 valid questions from failed five-question batches. Those 1,833
rows are frozen inputs to this repair and must not change.

## Treatment A: deterministic repair of 320 questions

The repair reads the frozen current-schema raw response and the frozen source
row. It never constructs a provider client and never creates new answer text.

For the 314 repeated-answer questions:

- preserve the first occurrence of each answer in provider order;
- normalize only for duplicate detection using whitespace collapse plus
  case-folding, matching the frozen generator validator;
- preserve the existing fusion rank-one answer and its source group exactly;
- require complete, non-overlapping source-group coverage and preserved
  predeclared conflict separation;
- require at least one exact frozen source answer for every semantic group;
- fill any vacated lower ranks with unused answers from that question's
  validated fusion row; and
- record each filler as a fusion-derived answer rather than attributing it to a
  Gemini semantic group.

For the five missing-exact-source cases and the one invalid group-representation
case:

- retain only structurally valid generated answers;
- insert the missing group's exact frozen source answer when necessary;
- remove the lowest-ranked non-rank-one variant from an overrepresented group
  when a slot is required;
- fill any remaining gap from unused validated fusion answers; and
- reassign consecutive ranks 1 through 10 without changing answer order beyond
  the required removal/insertion.

The deterministic repair rejects the question and uses the fusion fallback if
any invariant cannot be proven. It does not weaken the original source hash,
question/video binding, rank-one binding, answer-count, nonempty-answer, or
official uniqueness checks.

## Treatment B: fresh repair of 290 questions

The fresh repair uses a new immutable run root. It sends only the 290 frozen
questions and their frozen rank-one candidate answers. It does not send video,
audio, transcripts, reference answers, annotations, labels, prior Gemini
rationales, or the speculative Qwen3.5 ranked tail.

The request contract is the paid-proven five-question grouped-rewrite contract:

- Gemini 3.7 Flash;
- batches of five questions, producing 58 calls;
- the same frozen grouping and no-new-facts instructions used by the successful
  A3 batches;
- the same nested response schema and output-token setting;
- one attempt per batch and no automatic retry;
- exact source-once coverage, preserved predeclared conflicts, ten ranked
  nonempty answers, byte-exact fusion rank one, every semantic group
  represented, and an exact frozen source answer retained for every group.

Any provider error, interruption, malformed response, validation failure, or
cost-cap stop produces per-question fusion fallback. A failed batch cannot
invalidate successful questions in another batch.

## Cost and execution boundary

Before the first paid call, the runner computes a conservative reservation for
all 58 requests from the exact wire payloads. The new-run cap is $5 and the
combined historical-plus-repair cost is reported separately. Dispatch stops
before spending if committed cost plus active reservations would exceed the
new-run cap.

The execution uses a bounded maximum of four concurrent calls, the same
deadline-tested level as the completed A3 run. Attempts, requests, raw
responses, usage, parsed responses, and terminal markers are written
immutably. A resume skips terminal batches, refuses ambiguous attempts, and
never repeats a provider call whose outcome is uncertain.

## Merge and precedence

The final per-question precedence is:

1. a newly validated fresh repair for one of the 290 questions;
2. a validated deterministic repair for one of the 320 questions;
3. the existing accepted or raw-salvaged grouped row for the other 1,833
   questions; or
4. the byte-exact validated fusion fallback.

The merger must prove that the first three sources have disjoint query-ID sets,
that all 2,443 released question/video bindings are covered exactly once, and
that every non-rank-one change is bound to immutable source evidence. It writes
a new immutable repair root; it never overwrites the original A3, salvage, or
tracked upload artifacts.

## Evidence and reporting

The new repair root contains:

- a manifest binding all six frozen source artifacts, the original salvage
  freeze, both repair inputs, the exact request bundle, schema, prompt,
  settings, model, cap, and code revision;
- per-batch provider evidence for the fresh treatment;
- per-question deterministic-repair decisions and reason codes;
- a final organizer file and per-question provenance;
- a summary with accepted, deterministic-repaired, fresh-repaired, and fallback
  counts plus separate historical, new, and cumulative cost; and
- SHA-256 bindings for every final artifact.

The tracked `g37_grouped_unique/submission.json` and its ledger are updated only
after the new root passes all gates. The prior tracked file remains recoverable
from Git history and its old hashes stay documented as superseded evidence.

## Verification gates

Implementation follows test-first development. Tests must first fail for the
missing repair behavior and then pass after implementation. Required coverage:

- repeated-answer removal and fusion filling;
- exact-source insertion and group-representation repair;
- refusal when rank one, source coverage, conflicts, or source hashes drift;
- exact selection of the 290 fresh query IDs;
- no provider calls in deterministic or dry-run paths;
- one-attempt, resume, ambiguity, error, and cost-cap behavior;
- disjoint merge precedence and byte-preservation of the existing 1,833 rows;
- final 2,443-by-10 shape, consecutive ranks, nonempty answers, finite times,
  and official normalized uniqueness;
- official NIST validator with zero errors and zero warnings;
- full provenance/hash verification and privacy/credential scans; and
- comparison of fallback count, rank-one equality, stricter duplicate rate,
  non-answer rate, answer lengths, and a label-free manual sample before release.

The repair is promoted only if rank one remains identical for all 2,443
questions, the official validator passes, no labels were used, the new evidence
chain is complete, and the fallback count strictly decreases. A fresh result
that is valid but not better remains preserved without replacing the tracked
submission.
