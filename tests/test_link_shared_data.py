from __future__ import annotations

import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "link-shared-data.sh"


def run(*args: str | Path, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def make_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    worktree = tmp_path / "linked-worktree"
    repository.mkdir()
    run("git", "init", "-q", cwd=repository)
    run("git", "config", "user.name", "Shared Data Test", cwd=repository)
    run("git", "config", "user.email", "shared-data@example.invalid", cwd=repository)
    (repository / "README.md").write_text("fixture\n")
    run("git", "add", "README.md", cwd=repository)
    run("git", "commit", "-qm", "fixture", cwd=repository)
    (repository / "data").mkdir()
    (repository / "data" / "marker.txt").write_text("canonical\n")
    run("git", "worktree", "add", "-q", "--detach", worktree, cwd=repository)
    return repository, worktree


def test_links_current_worktree_to_canonical_data_idempotently(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "shared-data helper has not been implemented"
    repository, worktree = make_repository(tmp_path)

    first = run(SCRIPT, cwd=worktree)
    second = run(SCRIPT, cwd=worktree)

    link = worktree / "data"
    assert link.is_symlink()
    assert link.readlink() == repository / "data"
    assert link.resolve() == (repository / "data").resolve()
    assert "linked" in first.stdout
    assert "already linked" in second.stdout
    assert run("git", "status", "--short", cwd=worktree).stdout == ""


def test_all_links_every_registered_worktree(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "shared-data helper has not been implemented"
    repository, first_worktree = make_repository(tmp_path)
    second_worktree = tmp_path / "second-worktree"
    run("git", "worktree", "add", "-q", "--detach", second_worktree, cwd=repository)

    result = run(SCRIPT, "--all", cwd=repository)

    assert (repository / "data").is_dir()
    assert not (repository / "data").is_symlink()
    assert (first_worktree / "data").resolve() == (repository / "data").resolve()
    assert (second_worktree / "data").resolve() == (repository / "data").resolve()
    assert "canonical data directory" in result.stdout


def test_refuses_to_replace_conflicting_data_path(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "shared-data helper has not been implemented"
    _, worktree = make_repository(tmp_path)
    conflict = worktree / "data"
    conflict.mkdir()
    sentinel = conflict / "keep-me.txt"
    sentinel.write_text("do not delete\n")

    result = run(SCRIPT, cwd=worktree, check=False)

    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert sentinel.read_text() == "do not delete\n"
