# Recover All TREC Video Work

## Objective

Create one reviewable pull request that preserves all repository-worthy TREC Video benchmark code, tests, configurations, and findings that remain outside `origin/main`, so the source worktrees can be archived after the PR merges.

## Scope

- Preserve the detached benchmark history ending at `7209acb154d9a7d4232622df2d7ba6d40e9b87a1`.
- Preserve the detached dstack recovery history ending at `57359bd5c244e9d1c40728ba94cacf95bedafdf2`.
- Include reviewed, repository-worthy uncommitted reports and configurations from the source worktrees.
- Exclude datasets, model artifacts, runtime logs, secrets, duplicated files, and ambiguous personal scratch notes.
- Preserve all work already merged into current `origin/main`.

## Integration approach

Merge both detached histories with non-fast-forward merge commits. Resolve conflicts by retaining the current-main experiment conclusions and infrastructure while adding the older benchmark capabilities. Keep provenance intact rather than flattening the histories into one copy-only commit.

## Tasks

1. Audit committed histories and all modified/untracked files for overlap, secrets, generated output, and repository value.
2. Merge the benchmark history and resolve conflicts against current `origin/main`.
3. Merge the dstack recovery history and resolve any conflicts.
4. Add only the approved loose reports/configurations, recording deliberate exclusions.
5. Verify the NIST scoring submodule is safely pinned and reproducible.
6. Run the full test, lint, formatting, shell/configuration, secret, and dataset-tracking checks available in the combined tree.
7. Obtain an independent review of the integrated diff and address actionable findings.
8. Commit, push, and open one pull request against `main`.

## Completion gate

- All intended commit histories are ancestors of the recovery branch.
- No dataset contents, secrets, raw logs, or model artifacts are tracked.
- The combined test and lint suites pass.
- The pull request clearly lists included work, deliberate exclusions, verification evidence, and the archive safety condition.
- Source worktrees are not removed until the pull request has merged into `origin/main`.

## Current status

The recovery branch is based on `origin/main` at `fb06b41`. Its clean baseline passes 137 tests and Ruff checks. Detached-history and loose-file audits are in progress.
