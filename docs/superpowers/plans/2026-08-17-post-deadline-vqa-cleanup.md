# Post-Deadline VQA Cleanup Implementation Plan

**Goal:** Document the closed TRECVID 2026 Answer Generation submission state
and establish a conservative maintenance boundary without changing submitted
answers or retiring reproducibility tooling.

**Design:**
[`2026-08-17-post-deadline-vqa-cleanup-design.md`](../specs/2026-08-17-post-deadline-vqa-cleanup-design.md)

## Task 1: Verify the merged closure state

- Confirm PR #43's actual merge topology and correct any stale merge-method
  wording.
- Recompute the four tracked submission hashes and compare them with their run
  records.
- Inventory deadline runners/configurations and their test/import references;
  make no deletions.

## Task 2: Add the maintenance record

- Create `docs/submissions/2026-vqa-post-deadline-maintenance.md` with:
  - the four organizer-validated run tags;
  - the immutable submission boundary;
  - the distinction between supported CLIs and historical reproducibility
    tooling;
  - a dependency-aware retirement checklist; and
  - a concise baseline test-health note.
- Link the maintenance record from `docs/submissions/README.md`.
- Update the root `README.md` only where a newcomer needs the closure link.

## Task 3: Verify documentation and retained artifacts

- Run focused submission/grouped-rewrite tests.
- Validate changed Markdown links and formatting.
- Recompute tracked upload hashes and run the organizer validator where the
  existing local command is available.
- Scan the diff for credentials, private absolute paths, raw logs, labels,
  annotations, and dataset content.
- Run `git diff --check` and inspect the final diff scope.

## Task 4: Publish for review

- Commit the documentation update.
- Push `codex/post-deadline-vqa-cleanup`.
- Open a draft PR with a squash-friendly title, scoped summary, and exact
  verification evidence.
