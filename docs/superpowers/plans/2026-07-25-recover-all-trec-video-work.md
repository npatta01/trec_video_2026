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

The recovery branch is based on `origin/main` at `fb06b41`. Its clean baseline passed 137 tests and Ruff checks. Both detached histories have been merged without textual conflicts. The combined benchmark history passes 706 tests, and the NIST submodule initializes at its pinned commit.

Reviewed loose artifacts selected for inclusion are seven durable research reports and one provider-neutral, bounded GPU smoke test. Deliberate exclusions are duplicate/provider-specific scratch configurations, `notes.txt`, the stale unchecked evidence-routing umbrella plan, and colliding internal `.superpowers/sdd` reports containing raw-adjacent outputs and local paths.

Remaining work is final combined verification, independent diff review, push, and pull-request creation.

The final advisor review identified and resolved one label-isolation blocker:
the omni compatibility launcher now mounts only `src/` instead of the full
repository and mounts only the frozen contract plus its video, audio, and frame
evidence roots instead of the full dataset. NIST and dataset reference answers
are therefore unreachable to inference code. A real container mount canary
verified that required evidence is readable while both label locations are
absent. Two non-blocking recovery caveats are documented explicitly:
GPU cost ceilings belong on consuming tasks, and the model-specific historical
Containerfiles require preservation of their exact host-local base image for
bit-identical replay.
