# VQA Submission Ledger Design

## Goal

Preserve the prepared TRECVID 2026 Answer Generation submission as durable,
accessible repository memory without merging its one-off ten-answer completion
policy into reusable model code or committing private competition predictions.

## Design

The repository gains a small `docs/submissions/` ledger. Its index lists each
organizer run tag and lifecycle state. A dated run record explains the model,
evidence policy, ten-answer construction, unavailable-video handling, compute,
validator result, immutable hashes, and the exact metadata intended for
Evalbase.

The answer JSON, raw predictions, questions, frames, and logs remain in the
ignored canonical `data/` tree. The ledger identifies private artifacts by
repository-relative local path and SHA-256, so a maintainer can verify the
intended upload without exposing its contents.

The executed implementation remains at commit
`c3477a794f60fca1e174a91cd6f1185b507ce1fe` behind an archival Git ref. No
submission runner or fallback policy is imported into the current baseline.

## Lifecycle

The initial ledger state is **prepared, validated, not uploaded**. A later,
separate change records the Evalbase run identifier, upload time, and accepted
validator response after the external submission succeeds. The ledger never
claims organizer acceptance based only on the local validator.

## Verification

- Recompute the private artifact hashes and compare them with the ledger.
- Confirm all 2,443 official test queries and all 15 organizer-designated
  unavailable-video queries remain in the top-10 file.
- Confirm the pinned NIST validator report contains zero errors and warnings.
- Scan the Git diff for questions, answer rows, credentials, raw logs, and
  media paths beyond the approved repository-relative private artifact root.
- Run Markdown formatting, link, and `git diff --check` checks.
