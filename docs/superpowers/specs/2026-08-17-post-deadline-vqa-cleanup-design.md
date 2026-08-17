# Post-Deadline VQA Cleanup Design

## Goal

Close the TRECVID 2026 Answer Generation deadline work with a small,
documentation-first follow-up. Preserve the four organizer-validated
submissions and the code and evidence needed to understand how they were
produced, while making the repository's final state and maintenance boundary
clear.

## Scope

- Record all four submitted and organizer-validated run tags in the submission
  ledger.
- Correct stale integration wording and broken or misleading links.
- Add a concise maintenance note that distinguishes reproducibility artifacts
  from supported reusable tooling.
- Record known repository-wide test-environment failures as follow-up context,
  not as failures introduced by the submission work.
- Preserve all deadline runners, dstack configurations, frozen evidence,
  tracked submission files, and private local artifacts.

## Non-goals

This change does not delete or archive operational files, rewrite historical
source hashes, rerun models, alter a submitted answer, repair unrelated test
failures, or publish private evidence. Retirement of deadline-only code needs
its own dependency audit and change set.

## Documentation structure

The submission ledger remains the authoritative organizer-run index. A new
post-deadline maintenance page records the final submission state, identifies
the retained deadline implementation as historical reproducibility tooling,
and gives future maintainers a bounded cleanup checklist. The repository
README links to that page through the submission ledger rather than treating
historical deadline scripts as supported CLIs.

## Verification

- Confirm the four tracked upload files still match the immutable hashes in
  their run records.
- Run the focused submission and grouped-rewrite tests that cover the retained
  artifacts and validation contracts.
- Check all changed Markdown links and formatting.
- Scan the change for credentials, absolute private paths, labels,
  annotations, raw logs, and accidentally tracked dataset content.
- Run `git diff --check` and keep the PR limited to documentation unless a
  clearly stale reference requires a small correction.

## Integration

Use a squash-friendly follow-up PR from `codex/post-deadline-vqa-cleanup`.
The PR should remain independently reviewable and must not depend on modifying
the already merged submission artifacts.
