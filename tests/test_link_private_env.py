from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "link-private-env.sh"
PLACEHOLDER = "placeholder-value-not-a-real-secret"


def run(
    *args: str | Path,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
        env=env,
    )


def make_repository(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    repository = tmp_path / "repository"
    worktree = tmp_path / "linked-worktree"
    repository.mkdir()
    run("git", "init", "-q", cwd=repository)
    run("git", "config", "user.name", "Private Env Test", cwd=repository)
    run("git", "config", "user.email", "private-env@example.invalid", cwd=repository)
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    private_env = repository / "private-env"
    private_env.mkdir()
    (private_env / ".gitkeep").write_text("", encoding="utf-8")
    (private_env / ".gitignore").write_text(
        "*\n!.gitignore\n!.gitkeep\n", encoding="utf-8"
    )
    run("git", "add", "README.md", "private-env", cwd=repository)
    run("git", "commit", "-qm", "fixture", cwd=repository)

    source = tmp_path / "config" / "trec_video_2026" / "gemini.env"
    source.parent.mkdir(parents=True)
    source.write_text(f"GEMINI_API_KEY={PLACEHOLDER}\n", encoding="utf-8")
    source.chmod(0o600)

    run("git", "worktree", "add", "-q", "--detach", worktree, cwd=repository)
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path / "home")
    environment["XDG_CONFIG_HOME"] = str(source.parents[1])
    return repository, worktree, source, environment


def test_links_current_worktree_to_canonical_env_idempotently(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "private-env helper has not been implemented"
    repository, worktree, source, environment = make_repository(tmp_path)

    first = run(SCRIPT, cwd=worktree, env=environment)
    second = run(SCRIPT, cwd=worktree, env=environment)

    link = worktree / "private-env" / "gemini.env"
    assert link.is_symlink()
    assert link.readlink() == source
    assert link.resolve() == source.resolve()
    assert "linked" in first.stdout
    assert "already linked" in second.stdout
    assert PLACEHOLDER not in first.stdout + first.stderr + second.stdout + second.stderr
    assert run("git", "check-ignore", "-q", link, cwd=worktree).returncode == 0
    assert run("git", "status", "--short", cwd=worktree).stdout == ""
    assert repository.exists()


def test_all_links_every_registered_worktree(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "private-env helper has not been implemented"
    repository, first_worktree, source, environment = make_repository(tmp_path)
    second_worktree = tmp_path / "second-worktree"
    run("git", "worktree", "add", "-q", "--detach", second_worktree, cwd=repository)

    result = run(SCRIPT, "--all", cwd=repository, env=environment)

    for worktree in (repository, first_worktree, second_worktree):
        link = worktree / "private-env" / "gemini.env"
        assert link.is_symlink()
        assert link.resolve() == source.resolve()
    assert "linked" in result.stdout
    assert PLACEHOLDER not in result.stdout + result.stderr


def test_refuses_to_replace_conflicting_private_env_path(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "private-env helper has not been implemented"
    _, worktree, _, environment = make_repository(tmp_path)
    conflict = worktree / "private-env" / "gemini.env"
    conflict.write_text("keep this file\n", encoding="utf-8")

    result = run(SCRIPT, cwd=worktree, check=False, env=environment)

    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert conflict.read_text(encoding="utf-8") == "keep this file\n"


def test_rejects_canonical_env_without_0600_permissions(tmp_path: Path) -> None:
    assert SCRIPT.is_file(), "private-env helper has not been implemented"
    _, worktree, source, environment = make_repository(tmp_path)
    source.chmod(0o640)

    result = run(SCRIPT, cwd=worktree, check=False, env=environment)

    assert result.returncode != 0
    assert "0600" in result.stderr
    assert not (worktree / "private-env" / "gemini.env").exists()
