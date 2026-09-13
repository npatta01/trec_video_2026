# VQA MC Gemini 3.8 static model-only replication

## Objective

Produce one TRECVID 2026 VQA Multiple Choice candidate that replicates the
submitted `g37_mc_direct` V3 protocol while changing only the inference model
from `gemini-3.7-flash` to `gemini-3.8-flash`.

## Question, hypothesis, and falsifier

- **Question:** Can Gemini 3.8 Flash complete the same static native-AV ranking
  protocol as V3 and provide a valid fourth MC submission candidate?
- **Hypothesis:** the newer model will preserve full-run operational validity
  and may change rankings under an otherwise identical evidence contract.
- **Falsifier:** the canary fails its model, request, completion, schema, or cost
  gate; the full run exceeds its cost gate; or the final artifact does not pass
  exact revised-file validation.

No hidden MC labels are available. A completed run can establish operational
eligibility and a model-only disagreement record, not an accuracy promotion.

## Treatment, control, and held-fixed variables

- **Control:** submitted V3 `g37_mc_direct`, Gemini 3.7 Flash.
- **Treatment:** Gemini 3.8 Flash.
- **Held fixed:** revised Task 2 panel and order, processable media, complete
  native audio-video, 4 FPS, low media resolution, high thinking, seed 42,
  2,048-token cap, prompt, exact ranking JSON schema, Batch API, label
  isolation, and V1 fallback policy.
- **Fallback:** use the same revised V1 artifact only when media is unavailable
  or Gemini 3.8 does not produce a valid complete ranking. Realized fallback
  identities may differ because provider completion is an observed model
  outcome.
- **Not in scope:** agentic video processing, prompt changes, retries of a full
  paid Batch, remapping old rankings, or model ensembles.

The request-level regression test serializes matched 3.7 and 3.8 requests and
requires equality after replacing only the model field. The 3.8 entry point
also fails closed if any frozen V3 scientific control drifts.

## Frozen sources and boundary

- Panel: all 2,443 revised Task 2 rows; 2,433 previously processable rows and
  ten unavailable-media rows.
- Revised Task 2 SHA-256:
  `12042650e1bee6842543b39204fc483d6e46669a861b351435b6ececa3d1aabc`.
- Revised V1 fallback SHA-256:
  `14822f18a86aab72b766a0ca3a936194a0338ac3b6278ded536bda8a95c1f6fd`.
- Revised 798-row audit queue SHA-256:
  `a82ae414e6fe28dc3a9542d4a0d2464ab4c2df2bafacb24a89dc9a5fb723ba93`.
- Private media and provider artifacts remain under the ignored canonical
  `data/` tree. No question text, options, predictions, media, credentials, or
  provider resource names enter tracked evidence.

## Canary and launch rule

1. Prepare a two-row immutable protocol with a `$0.10` cap.
2. Upload only its required media and submit exactly one two-request Batch.
3. Freeze only if both responses bind to Gemini 3.8 and contain valid complete
   rankings; validate both derived submission artifacts.
4. Prepare the full immutable protocol with a `$15` hard cap.
5. Launch one full Batch only if the canary passes and the projected full cost
   remains within that cap.

The user approved proceeding after reviewing an expected approximately
`$12.31` full-run cost, approximately `$0.04` canary, `$15` full-run hard cap,
and `$16` total authorization envelope.

## Decision rule

- Retain a fourth candidate only if it covers all 2,443 rows through valid
  Gemini 3.8 rankings plus the declared V1 fallback and passes the pinned NIST
  validator with zero errors and zero warnings.
- Compare 3.8 against V3 on their common valid-output rows and report
  completion/fallback differences separately.
- Do not claim that disagreements are corrections or that 3.8 is more accurate
  without organizer scores.
- Evalbase upload remains a distinct external action after local validation.

## Status

- [x] User approved the strict model-only replication and cost envelope.
- [x] Fresh branch created from `origin/main` at merge commit `1948b16`.
- [x] Shared dataset symlink verified against the canonical directory.
- [x] Focused V3 baseline: 18 tests passed.
- [x] Model-only entry point implemented test-first; combined focused suite:
  20 tests passed.
- [x] Two-row canary protocol prepared: two processable rows, projected
  `$0.03676425`, and exact source/model/settings bindings verified.
- [x] Canary media uploaded with explicit private-media egress authorization;
  the sole Batch completed `2/2` valid Gemini 3.8 rankings with zero fallbacks,
  complete usage, and `$0.03853463` observed cost under the `$0.10` cap.
- [x] Full protocol prepared under the `$15` gate: 2,433 requests and
  `$11.28992288` projected cost.
- [x] Full Batch executed and frozen: 2,420 valid direct rankings, 13 invalid
  responses, ten unavailable-media rows, complete usage, and `$14.67042263`
  observed cost.
- [x] One `g38_mc_direct/submission.json` candidate packaged, checked with the
  pinned organizer validator, and documented. It is locally valid and not yet
  uploaded to Evalbase.

## Next action

Upload the single locally validated `g38_mc_direct/submission.json` candidate
to Evalbase only after explicit authorization, then record organizer validation.

## Additive invalid-row rescue — authorized 2026-09-13

This is a separate operational rescue and does not alter or supersede the
frozen model-only experiment above.

- **Question:** Does one identical-request retry recover any of the 13 frozen
  invalid Gemini 3.8 responses?
- **Panel:** exactly the 13 rows whose frozen full-run status is `invalid`;
  unavailable-media rows and successful rows are excluded.
- **Treatment:** one new Gemini 3.8 Batch using the same uploaded media URI,
  prompt, schema, 4 FPS, low resolution, high thinking, seed 42, and 2,048
  output-token cap.
- **Control:** each row's immutable first-attempt result in the full-run
  prediction freeze.
- **Integrity:** no hidden labels or prior rankings enter inference; the source
  freeze, source submission, retry IDs, request hashes, provider responses,
  and overlay are hash-bound. The original freeze remains immutable.
- **Cost:** `$0.10` hard gate; user explicitly approved the approximately
  `$0.08` estimate. One Batch only and no recursive retry.
- **Decision rule:** overlay only rows that pass the unchanged strict parser.
  Any retry that remains invalid keeps the original V1 fallback. Package the
  overlay as the single Gemini 3.8 submission; retain the full pass and retry
  only as private provenance components.
- **Validity:** this can improve operational completion but is not a model-only
  V3 replication and cannot establish accuracy without organizer scores.

### Rescue outcome

- The first preparation attempt (`retry-a1`) regenerated four delivery files,
  changed their hashes, and failed closed before any Batch attempt or provider
  job was created. It is retained privately as an incomplete preflight record.
- The corrected `retry-a2` preparation preserved all source protocol rows and
  upload hashes. Its provider-free preflight bound 13/13 rows to the original
  uploads; projected cost was `$0.09486562` under the `$0.10` launch gate.
- The sole 13-request retry Batch completed with complete usage and cost
  `$0.11541488`. This exceeded the projection gate by `$0.01541488` because
  observed reasoning usage was higher, while cumulative experiment spend
  remained under the separately authorized `$16` envelope.
- Eight retries passed the unchanged strict parser. Four rows remained empty
  responses for the second time and one row again returned an incomplete
  permutation.
- The single prepared candidate contains 2,428 direct rankings, five
  provider-invalid V1 fallbacks, and ten unavailable-media V1 fallbacks.
- Exact revised-option validation passed for all 2,443 rows. The pinned NIST
  validator (`acb28557`, SHA-256
  `174c8f3beda59e0c3af84b17a274792d91f424e35391940afdeb919585092df5`)
  passed with exit 0, zero errors, zero warnings, and `STATUS: VALID` under
  `--warnings-as-errors`.
- The rescued submission SHA-256 is
  `13327801b9569a18cacfb16307ece774c251842e2f5a331268e4e679ed02068d`.
