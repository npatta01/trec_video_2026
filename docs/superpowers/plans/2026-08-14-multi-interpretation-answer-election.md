# Multi-Interpretation Answer Election Implementation Plan

> **Execution note:** Follow this plan task by task with test-first development.
> The approved design is
> `docs/superpowers/specs/2026-08-14-multi-interpretation-answer-election-design.md`.

**Goal:** Determine whether three evidence-citing interpretations plus a fixed,
label-blind election rule improves emitted top-1 answers over one
unique-event interpretation on a fresh frozen 12-row panel.

**Architecture:** A deterministic selector creates a source-bound panel while
excluding all earlier issue #35 and seven-row canary rows. A provider runner
executes two label-free Gemini arms with one no-retry response per row. The
multi arm validates three candidates and elects locally without labels. A
separate scorer loads references only after both prediction/evidence freezes,
then applies a predeclared role-aware gate and emits sanitized artifacts.

**Constraints:** Text-only frozen maps; no native-video calls; no evidence
acquisition; no counting specialist #34; no shared router #36; Gemini 3.1 Pro
Preview, temperature 0, seed 42, high thinking; 24 total calls; preserve every
failure and negative result.

**Current status (completed 2026-08-14):** The fresh 4/4/4 panel, both arm
protocols, all 24 no-retry calls, prediction/evidence freezes, 12-row manual
audit, automatic metrics, and gate are complete. All calls passed their
contracts. The gate is negative: multi answered 4/4 premise rows but improved
only 1 over single; both arms verified 3/4 safeguards; multi preserved 3/4
controls; unsupported emitted answers were 0. Oracle candidate coverage was
12/12 but elected multi correctness was 10/12, so candidate generation showed
headroom while this election rule was not safe enough to promote. Combined
integrity SHA-256 is `c3ddeedb0bea446bc9d65c7de6b5e6fe6c8a2f78600a373cf7f0d2de042d6280`;
the additive metric-provenance score SHA-256 is
`4b413a28cbb077118a37aaa084595ab5e231f4ad2ed7e2ca8ec268dd721816c7`.
Next action is final render/checksum/privacy verification, independent review,
then push/update PR #37 and issue #35. The reconciled research route is
explicit temporal or speech verification over fresh frozen context, not a
repeat of this ensemble.

---

## Task 1: Implement deterministic fresh-panel selection

**Files:**

- Create: `configs/vqa/multi_interpretation_answer_election_issue35.json`
- Create: `scripts/select_multi_interpretation_answer_election.py`
- Create: `tests/test_select_multi_interpretation_answer_election.py`

1. Write failing tests for exact exclusion of all issue #35 v2/v3 and earlier
   seven-row canary IDs, canonical source-hash verification, deterministic
   ordering, unique role counts of 4/4/4, exact cited map-line extraction, and
   rejection of duplicate or previously spent rows.
2. Implement the smallest selector and tracked configuration needed to produce
   a reviewable candidate shortlist from the frozen 475-row maps, direct
   predictions, references, and automatic metrics.
3. Generate the shortlist, manually adjudicate only the predeclared role
   definitions, record concise selection notes, and freeze the final 12 IDs,
   roles, questions, evidence spans, exclusions, and source SHA-256 values under
   `data/vqa/benchmarks/multi-interpretation-answer-election-issue35-v1/`.
4. Rerun selection tests and demonstrate that changing a source, exclusion, ID,
   role, or evidence span invalidates the freeze.
5. Commit the selection code/config/tests before any provider call.

## Task 2: Implement arm contracts and deterministic election

**Files:**

- Create: `scripts/run_multi_interpretation_answer_election.py`
- Create: `tests/test_run_multi_interpretation_answer_election.py`

1. Write failing contract tests for the single-arm answer/verification schema
   and the multi-arm exactly-three-candidate schema.
2. Add negative cases for unknown citations, cited-claim/final-answer mismatch,
   an answer under `answer_blocking`, empty locator-variance qualifiers,
   malformed provider JSON, missing candidates, and duplicate interpretation
   types.
3. Write failing election tests for schema rejection, normalization,
   two-candidate agreement, one-answer-plus-verifications, answer conflict,
   no-valid-answer verification, shortest-answer tie breaking, and the
   prohibition on any label/metric/model input.
4. Implement prompts and validators shared only where their invariants are
   identical. Keep generation and election separate, and persist candidate
   validity, rejection reasons, normalized groups, and emitted top-1 result.
5. Add a fake-provider end-to-end test proving one call per row/arm, no retries,
   raw response/metadata persistence before parsing, terminal preservation, and
   resumability without duplicate calls.
6. Run focused tests, Ruff, and `py_compile`.

## Task 3: Implement immutable two-arm execution and freezes

**Files:**

- Modify: `scripts/run_multi_interpretation_answer_election.py`
- Modify: `tests/test_run_multi_interpretation_answer_election.py`

1. Write failing tests for separate label-free protocols, terminal manifests,
   raw-response hashes, parsed-result hashes, request hashes, metadata hashes,
   and a combined evidence-freeze manifest.
2. Bind the selection freeze, model/settings, prompt text, schema, election
   implementation, and pre-inference gate freeze into both arm protocols.
3. Make protocol, prediction, and evidence freezes additive and immutable.
   Refuse resume on any drift; never overwrite a terminal row or freeze.
4. Verify deliberate tampering of every bound artifact is detected.

## Task 4: Implement label-safe scoring, audit, and gate

**Files:**

- Create: `scripts/score_multi_interpretation_answer_election.py`
- Create: `tests/test_score_multi_interpretation_answer_election.py`

1. Write failing tests proving the scorer refuses to load references or audit
   labels until selection, gate, both protocols, both prediction freezes, and
   both evidence freezes verify.
2. Implement direct-baseline, single-arm, and elected-multi METEOR and
   BERTScore-F1. Reuse historical STS only when the exact implementation and
   dependency are available; otherwise record it as unavailable and do not
   fabricate a composite.
3. Emit all-row and per-role aggregates without treating verification sentinels
   as ordinary semantic answers. Keep oracle any-candidate coverage in a
   clearly diagnostic-only section.
4. Add a 12-row manual-audit template covering every candidate set and every
   emitted-result change. Validate verdict consistency with terminal state,
   verification routing, citation support, and role.
5. Implement the exact 3/4 premise, 2 improvements, 4/4 safeguards, 4/4
   controls, zero unsupported, complete-audit gate. Operational and contract
   failures remain in denominators.
6. Add tamper and gate-boundary tests, then run focused and neighboring tests.

## Task 5: Freeze and execute the paid 24-call experiment

**Private root:**
`data/vqa/benchmarks/multi-interpretation-answer-election-issue35-v1/`

1. Verify the `data` symlink, exact Gemini credential file permissions, source
   hashes, exclusion set, clean label-free protocols, and frozen 12-row gate.
2. Run the full fake-provider preflight and privacy scan. Inspect generated
   requests to confirm they contain only question and frozen evidence.
3. Execute one single-arm and one multi-arm canary call on the same frozen row.
   Inspect transport and schema only; do not change prompts, schemas, election,
   panel, or gate after the first paid call.
4. Run the remaining calls with bounded concurrency. Make exactly 24 total
   provider attempts, with no retries. Preserve raw failures and metadata.
5. Create and independently verify both prediction freezes and evidence
   freezes before exposing references or audit labels to scoring.

## Task 6: Score and inspect every row

1. Generate both arms' deterministic emitted outputs, automatic metrics,
   per-role aggregates, oracle diagnostic coverage, and the blank audit packet.
2. Manually inspect all 12 candidate sets and every row where direct, single,
   or elected-multi output differs. Base judgments only on frozen question,
   evidence, candidate trace, terminal state, and post-freeze reference.
3. Freeze the completed audit additively, run the gate, and preserve the result
   even when negative.
4. Run a consistency review for unsupported answers, false verification,
   candidate/elected mismatch, role drift, and accidental oracle substitution.

## Task 7: Produce durable evidence and report

**Files:**

- Create: `docs/experiments/artifacts/vqa-multi-interpretation-answer-election-20260814/`
- Modify: `docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`
- Modify: `docs/project-status/research-routing.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Modify derived Markdown/HTML only through the repository render commands.

1. Export a sanitized summary, source/protocol/freeze hashes, role-aware metrics,
   gate result, manual-audit summary, candidate/election diagnostics, README,
   and `SHA256SUMS`. Keep raw prompts/responses, references, and datasets private.
2. Report the result under the existing answer-use theme with explicit
   `Evidence validity` and `Decision disposition`; reconcile routing and ledger
   only if the result changes the next action.
3. Regenerate QMD-derived outputs and the private tailnet-only report portal.
   Verify live HTTPS content and tailnet-only exposure.
4. Run focused tests, neighboring tests, full pytest with known-baseline
   classification, Ruff, formatting, `py_compile`, JSON validation, checksum
   verification, freeze verification, diff checks, and privacy scans.
5. Request independent review of the complete diff and artifact chain. Address
   all Critical/Important findings and rerun affected verification.
6. Commit intentionally, push the existing branch, update focused PR #37, and
   post a concise progress/result comment to issue #35. Do not merge.

## Completion criteria

- Exactly 12 fresh rows and 24 no-retry provider calls are durably frozen.
- The elected top-1 result is reproducible without labels or another model.
- Every candidate set and changed row has a frozen manual verdict.
- Metrics, oracle diagnostics, and the predeclared gate are clearly separated.
- A positive or negative result is preserved with verified private evidence,
  sanitized tracked artifacts, reconciled status docs, and a reviewed PR.
