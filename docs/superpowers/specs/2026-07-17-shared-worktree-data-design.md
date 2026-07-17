# Shared Worktree Data Design

## Goal

Make the repository's ignored `data/` tree available at the same relative path
in every Git worktree without copying or moving the 17 GiB dataset.

## Design

The canonical checkout at the parent of Git's common directory remains the
only physical data location. Linked worktrees use an absolute `data` symlink to
that canonical directory. An absolute link is intentional because linked
worktrees may live at unrelated directory depths.

The repository provides `scripts/link-shared-data.sh`. With no arguments it
configures the current worktree; with `--all` it configures every registered
worktree. The helper is idempotent and refuses to replace a real directory,
file, or incorrect symlink. It also refuses to proceed if the canonical data
directory is missing. The canonical checkout's real `data/` directory is
accepted unchanged.

The root `data` path is ignored in both the tracked `.gitignore` and Git's
shared local exclude file. The latter keeps links hidden immediately in older
detached worktrees whose checked-out commits lack the new ignore rule. README
documentation records the canonical location and the command needed after
creating future worktrees.

## Verification

Automated tests create temporary Git repositories and linked worktrees to
exercise link creation, all-worktree mode, idempotence, and conflict handling.
The live repository is then checked to ensure every registered worktree's
`data` resolves to the canonical directory and no dataset files become tracked.
