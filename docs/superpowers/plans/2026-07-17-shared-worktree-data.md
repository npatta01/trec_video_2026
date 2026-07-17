# Shared Worktree Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one ignored canonical `data/` directory available from every registered Git worktree.

**Architecture:** A Bash helper resolves the canonical checkout through `git rev-parse --git-common-dir`, then creates absolute symlinks in linked worktrees. It never moves data and rejects conflicting paths.

**Tech Stack:** Bash, Git worktrees, pytest

## Global Constraints

- Keep `/home/npatta01/data/competitions/trec_video_2026/data` as the only physical dataset copy.
- Never delete, move, overwrite, or commit dataset contents.
- Existing unrelated working-tree changes remain untouched.

---

### Task 1: Shared-data linker

**Files:**
- Create: `scripts/link-shared-data.sh`
- Create: `tests/test_link_shared_data.py`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Consumes: a normal Git repository with its physical `data/` directory beside the common `.git` directory
- Produces: `scripts/link-shared-data.sh [--all]`, which configures the current or all registered worktrees

- [x] **Step 1: Write failing behavioral tests**

Create temporary repositories and verify current-worktree linking, `--all`,
idempotence, and non-destructive conflict failure.

- [x] **Step 2: Run tests and verify the missing helper is the failure**

Run: `uv run pytest tests/test_link_shared_data.py -q`

Expected: FAIL because `scripts/link-shared-data.sh` does not exist.

- [x] **Step 3: Implement the minimal helper and ignore rule**

Resolve absolute repository paths with Git, validate the canonical directory,
and use `ln -s` only when the destination path is absent. Add `/data` to
`.gitignore`.

- [x] **Step 4: Run focused and full tests**

Run: `uv run pytest tests/test_link_shared_data.py -q && uv run pytest -q`

Expected: all tests pass.

- [x] **Step 5: Configure and verify live worktrees**

Run: `scripts/link-shared-data.sh --all`, then confirm every `data` path resolves
to the canonical directory and stays ignored.

- [x] **Step 6: Document and commit**

Document canonical ownership and future-worktree setup in `README.md`, run
`git diff --check`, then commit only the shared-data implementation files.
