import os
import stat
import subprocess
from pathlib import Path

import pytest
import trec_video_2026.vqa_download as vqa_download

from trec_video_2026.vqa_download import (
    AttemptResult,
    RATE_LIMIT_MARKER,
    build_ytdlp_command,
    count_final_media,
    load_proxy_credentials,
    main,
    pid_is_active,
    prepare_manifests,
    read_archive_ids,
    read_unavailable_ids,
    read_video_ids,
    run_url,
    run_dataset,
    run_downloads,
    start_detached,
    write_proxy_config,
    write_manifest,
)


def _secure_env(path: Path) -> Path:
    path.write_text("placeholder only\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_proxy_credentials_loads_nul_delimited_placeholders(tmp_path: Path):
    env_path = _secure_env(tmp_path / "proxy.env")
    captured = {}

    def runner(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command, 0, stdout=b"proxy.example:7000\0user/name\0p@ss word\0"
        )

    assert load_proxy_credentials(env_path, runner=runner) == (
        "proxy.example:7000",
        "user/name",
        "p@ss word",
    )
    assert captured["command"][-1] == str(env_path)
    assert captured["capture_output"] is True
    assert captured["check"] is True


def test_proxy_config_url_encodes_credentials_and_is_mode_0600(tmp_path: Path):
    env_path = _secure_env(tmp_path / "proxy.env")
    config_path = tmp_path / "runtime/proxy.conf"

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=b"proxy.example:7000\0user/name\0p@ss word\0"
        )

    assert write_proxy_config(env_path, config_path, runner=runner) == config_path
    assert config_path.read_text(encoding="utf-8") == (
        "--proxy\nhttp://user%2Fname:p%40ss%20word@proxy.example:7000\n"
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_proxy_credentials_rejects_wrong_permissions(tmp_path: Path):
    env_path = _secure_env(tmp_path / "proxy.env")
    env_path.chmod(0o640)

    with pytest.raises(RuntimeError, match="permissions"):
        load_proxy_credentials(env_path, runner=lambda *args, **kwargs: None)


def test_proxy_credentials_rejects_symlink(tmp_path: Path):
    target = _secure_env(tmp_path / "target.env")
    env_path = tmp_path / "proxy.env"
    env_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="symbolic link"):
        load_proxy_credentials(env_path, runner=lambda *args, **kwargs: None)


@pytest.mark.parametrize(
    "values",
    [
        b"proxy.example:7000\0user\0",
        b"\0user\0password\0",
        b"https://proxy.example:7000\0user\0password\0",
        b"user:pass@proxy.example:7000\0user\0password\0",
        b"proxy.example 7000\0user\0password\0",
    ],
)
def test_proxy_credentials_rejects_missing_or_invalid_values(
    tmp_path: Path, values: bytes
):
    env_path = _secure_env(tmp_path / "proxy.env")

    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=values)

    with pytest.raises(RuntimeError, match="credentials"):
        load_proxy_credentials(env_path, runner=runner)


def test_proxy_credentials_sanitizes_runner_failure(tmp_path: Path):
    env_path = _secure_env(tmp_path / "proxy.env")
    placeholder_secret = "PLACEHOLDER-SECRET-DO-NOT-LEAK"

    def runner(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr=placeholder_secret.encode()
        )

    with pytest.raises(RuntimeError) as caught:
        load_proxy_credentials(env_path, runner=runner)
    assert placeholder_secret not in str(caught.value)


def test_run_dataset_paces_every_ordinary_url(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive.txt"
    calls, sleeps = [], []

    run_dataset(
        manifest,
        tmp_path / "videos",
        archive,
        url_runner=lambda command: calls.append(command)
        or AttemptResult(False, False),
        sleeper=sleeps.append,
    )

    assert len(calls) == 2
    assert sleeps == [10.0, 10.0]


def test_run_dataset_retries_ordinary_failures_after_archive_progress(
    tmp_path: Path,
):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive.txt"
    calls: list[str] = []

    def runner(command: list[str]) -> AttemptResult:
        video_id = command[-1].rsplit("v=", 1)[-1]
        calls.append(video_id)
        if video_id == "abcDEF123_0":
            archive.write_text("youtube abcDEF123_0\n", encoding="utf-8")
            return AttemptResult(True, False)
        return AttemptResult(False, False)

    sleeps: list[float] = []
    run_dataset(
        manifest,
        tmp_path / "videos",
        archive,
        url_runner=runner,
        sleeper=sleeps.append,
    )

    assert calls == ["abcDEF123_0", "xyzXYZ987-1", "xyzXYZ987-1"]
    assert sleeps == [10.0, 10.0, 10.0]


def test_run_dataset_cools_down_and_resumes_from_archive(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n", encoding="utf-8"
    )
    archive = tmp_path / "archive.txt"
    outcomes = iter(
        [
            AttemptResult(False, True),
            AttemptResult(False, True),
            AttemptResult(True, False),
        ]
    )
    calls, sleeps = [], []

    def runner(command: list[str]) -> AttemptResult:
        calls.append(command)
        outcome = next(outcomes)
        if outcome.completed:
            archive.write_text("youtube abcDEF123_0\n", encoding="utf-8")
        return outcome

    run_dataset(
        manifest,
        tmp_path / "videos",
        archive,
        url_runner=runner,
        sleeper=sleeps.append,
    )

    assert len(calls) == 3
    assert sleeps == [3600.0, 7200.0, 10.0]


def test_run_dataset_resets_cooldown_after_archived_ordinary_attempt(
    tmp_path: Path,
):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive.txt"
    outcomes = iter(
        [
            AttemptResult(False, True),
            AttemptResult(True, False),
            AttemptResult(False, True),
            AttemptResult(True, False),
        ]
    )
    calls, sleeps = [], []

    def runner(command: list[str]) -> AttemptResult:
        calls.append(command)
        outcome = next(outcomes)
        if outcome.completed:
            video_id = command[-1].rsplit("v=", 1)[-1]
            with archive.open("a", encoding="utf-8") as handle:
                handle.write(f"youtube {video_id}\n")
        return outcome

    run_dataset(
        manifest,
        tmp_path / "videos",
        archive,
        url_runner=runner,
        sleeper=sleeps.append,
    )

    assert len(calls) == 4
    assert sleeps == [3600.0, 10.0, 3600.0, 10.0]


def test_manifest_excludes_unavailable_and_preserves_order(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        '1 , abcDEF123_0, "q1"\n2 , xyzXYZ987-1, "q2"\n', encoding="utf-8"
    )
    readme = tmp_path / "README.txt"
    readme.write_text("abcDEF123_0.json\n", encoding="utf-8")
    manifest = tmp_path / "urls.txt"

    assert read_video_ids(csv_path) == ["abcDEF123_0", "xyzXYZ987-1"]
    assert read_unavailable_ids(readme) == {"abcDEF123_0"}
    assert write_manifest(csv_path, readme, manifest) == ["xyzXYZ987-1"]
    assert manifest.read_text(encoding="utf-8") == (
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n"
    )


def test_read_archive_ids_accepts_ytdlp_archive_rows(tmp_path: Path):
    archive = tmp_path / "archive.txt"
    archive.write_text(
        "youtube abcDEF123_0\nyoutube xyzXYZ987-1\n", encoding="utf-8"
    )

    assert read_archive_ids(archive) == {"abcDEF123_0", "xyzXYZ987-1"}


def test_read_archive_ids_returns_empty_for_missing_file(tmp_path: Path):
    assert read_archive_ids(tmp_path / "missing.archive.txt") == set()


def test_read_archive_ids_filters_invalid_rows(tmp_path: Path):
    archive = tmp_path / "archive.txt"
    archive.write_text(
        "youtube abcDEF123_0\n"
        "youtube too-short\n"
        "malformed\n"
        "youtube abcDEF123_0 extra\n",
        encoding="utf-8",
    )

    assert read_archive_ids(archive) == {"abcDEF123_0"}


def test_ytdlp_command_paces_requests_without_per_stream_sleep(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("VQA_NODE_PATH", "/opt/node/bin/node")
    command = build_ytdlp_command(
        "https://www.youtube.com/watch?v=abcDEF123_0",
        tmp_path / "videos",
        tmp_path / "archive.txt",
    )
    joined = " ".join(command)

    assert "bestvideo*+bestaudio/best" in command
    assert "--merge-output-format" in command
    assert "mkv" in command
    assert "--write-info-json" in command
    assert "--download-archive" in command
    assert command[command.index("--js-runtimes") + 1] == (
        "node:/opt/node/bin/node"
    )
    assert command[command.index("--sleep-requests") + 1] == "1.5"
    assert "--sleep-interval" not in command
    assert "--max-sleep-interval" not in command
    assert "--batch-file" not in command
    assert command[-1] == "https://www.youtube.com/watch?v=abcDEF123_0"
    assert not any(token in joined for token in ("recode", "convert", "x264", "720"))


def test_ytdlp_command_uses_private_config_path_without_proxy_credentials(
    tmp_path: Path,
):
    command = build_ytdlp_command(
        "https://www.youtube.com/watch?v=abcDEF123_0",
        tmp_path / "videos",
        tmp_path / "archive.txt",
        proxy_config=Path("/private/proxy.conf"),
    )

    assert command[command.index("--config-locations") + 1] == (
        "/private/proxy.conf"
    )
    assert "--proxy" not in command
    assert not any("@" in argument for argument in command)
    assert command[-1] == "https://www.youtube.com/watch?v=abcDEF123_0"


def test_run_url_streams_output_and_detects_rate_limit(capsys):
    class Child:
        stdout = iter(["progress\n", f"ERROR: {RATE_LIMIT_MARKER}\n"])

        def wait(self):
            return 1

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=False, rate_limited=True)
    assert "progress" in capsys.readouterr().out


def test_run_url_zero_exit_without_marker_is_completed():
    class Child:
        stdout = iter(["finished\n"])

        def wait(self):
            return 0

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=True, rate_limited=False)


@pytest.mark.parametrize(
    "marker",
    [RATE_LIMIT_MARKER, RATE_LIMIT_MARKER.replace("’", "'")],
)
def test_run_url_zero_exit_with_marker_is_rate_limited_and_incomplete(marker: str):
    class Child:
        stdout = iter([f"ERROR: {marker}\n"])

        def wait(self):
            return 0

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=False, rate_limited=True)


def test_run_url_detects_youtube_session_rate_limit_wording():
    class Child:
        stdout = iter(
            [
                "ERROR: [youtube] tfTu-B4qIDQ: Video unavailable. This content "
                "isn't available, try again later. The current session has been "
                "rate-limited by YouTube for up to an hour.\n"
            ]
        )

        def wait(self):
            return 1

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=False, rate_limited=True)


@pytest.mark.parametrize(
    "output",
    [
        "ERROR: Sign in to confirm you're not a bot\n",
        "ERROR: This video is private\n",
        "ERROR: Video unavailable\n",
        "ERROR: This content isn't available, try again later.\n",
    ],
)
def test_run_url_does_not_classify_other_errors_as_rate_limits(output: str):
    class Child:
        stdout = iter([output])

        def wait(self):
            return 1

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=False, rate_limited=False)


def test_run_downloads_processes_each_training_url_before_testing(tmp_path: Path):
    training = tmp_path / "data/vqa/2025/training"
    testing = tmp_path / "data/vqa/2025/testing"
    (training / "annotations").mkdir(parents=True)
    testing.mkdir(parents=True)
    (training / "annotations/train.2025.csv").write_text(
        '1 , abcDEF123_0, "q"\n2 , qrsQRS456_2, "q"\n', encoding="utf-8"
    )
    (training / "README.txt").write_text("", encoding="utf-8")
    (testing / "testing.dataset.vqa.2025.csv").write_text(
        '1 , xyzXYZ987-1, "q"\n', encoding="utf-8"
    )
    (testing / "README.txt").write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(command: list[str]) -> AttemptResult:
        calls.append(command)
        return AttemptResult(False, False)

    assert run_downloads(tmp_path, runner=runner, sleeper=lambda _: None) == 0
    assert [command[-1] for command in calls] == [
        "https://www.youtube.com/watch?v=abcDEF123_0",
        "https://www.youtube.com/watch?v=qrsQRS456_2",
        "https://www.youtube.com/watch?v=xyzXYZ987-1",
    ]
    assert not any(".urls.txt" in argument for command in calls for argument in command)


def test_pid_is_active_rejects_missing_and_stale_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "download.pid"
    assert pid_is_active(pid_path) is False
    pid_path.write_text("12345\n", encoding="utf-8")

    def missing_process(pid: int, signal: int):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", missing_process)
    assert pid_is_active(pid_path) is False


def test_start_detached_records_child_pid_and_node_path(
    tmp_path: Path, monkeypatch
):
    runtime = tmp_path / "data/vqa/download_runtime"
    captured: dict[str, object] = {}

    class Child:
        pid = 4242

    def popen(command: list[str], **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Child()

    def fake_write(source: Path, target: Path, runner=subprocess.run):
        target.write_text("placeholder config\n", encoding="utf-8")
        return target

    monkeypatch.setattr(vqa_download, "write_proxy_config", fake_write)

    assert start_detached(
        tmp_path,
        popen=popen,
        node_path="/opt/node/bin/node",
        proxy_env_path=tmp_path / "placeholder.env",
    ) == 4242
    assert (runtime / "download.pid").read_text(encoding="utf-8") == "4242\n"
    assert captured["start_new_session"] is True
    assert captured["cwd"] == tmp_path
    assert captured["stderr"] == -2
    assert captured["env"]["VQA_NODE_PATH"] == "/opt/node/bin/node"


def test_start_detached_prepares_private_proxy_config_without_leaking_secrets(
    tmp_path: Path, monkeypatch
):
    runtime = tmp_path / "data/vqa/download_runtime"
    env_path = tmp_path / "placeholder.env"
    placeholder_secret = "PLACEHOLDER-SECRET-DO-NOT-LEAK"
    captured: dict[str, object] = {}

    def fake_write(source: Path, target: Path, runner=subprocess.run):
        captured["source"] = source
        captured["target"] = target
        target.write_text("placeholder config\n", encoding="utf-8")
        return target

    class Child:
        pid = 4242

    def popen(command: list[str], **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Child()

    monkeypatch.setattr(vqa_download, "write_proxy_config", fake_write)
    monkeypatch.setenv("DECODO_PASSWORD", placeholder_secret)
    assert start_detached(
        tmp_path,
        popen=popen,
        node_path="/opt/node/bin/node",
        proxy_env_path=env_path,
    ) == 4242

    config_path = runtime / "yt-dlp-proxy.conf"
    assert captured["source"] == env_path
    assert captured["target"] == config_path
    assert captured["env"]["VQA_PROXY_CONFIG_PATH"] == str(config_path)
    assert "DECODO_PASSWORD" not in captured["env"]
    assert placeholder_secret not in " ".join(captured["command"])
    assert placeholder_secret not in (runtime / "download.log").read_text(
        encoding="utf-8"
    )


def test_start_detached_rejects_an_active_job(tmp_path: Path):
    runtime = tmp_path / "data/vqa/download_runtime"
    runtime.mkdir(parents=True)
    (runtime / "download.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already running"):
        start_detached(tmp_path)


def test_prepare_manifests_reports_usable_counts(tmp_path: Path):
    training = tmp_path / "data/vqa/2025/training"
    testing = tmp_path / "data/vqa/2025/testing"
    (training / "annotations").mkdir(parents=True)
    testing.mkdir(parents=True)
    (training / "annotations/train.2025.csv").write_text(
        '1 , abcDEF123_0, "q"\n2 , xyzXYZ987-1, "q"\n', encoding="utf-8"
    )
    (training / "README.txt").write_text("abcDEF123_0.json\n", encoding="utf-8")
    (testing / "testing.dataset.vqa.2025.csv").write_text(
        '1 , qrsQRS456_2, "q"\n', encoding="utf-8"
    )
    (testing / "README.txt").write_text("", encoding="utf-8")

    assert prepare_manifests(tmp_path) == {"training": 1, "testing": 1}


def test_status_command_reports_stopped_job(tmp_path: Path, capsys):
    assert main(["status"], repo_root=tmp_path) == 0
    output = capsys.readouterr().out
    assert "state: stopped" in output
    assert "training completed: 0" in output
    assert "testing completed: 0" in output


def test_count_final_media_excludes_unmerged_format_fragments(tmp_path: Path):
    (tmp_path / "abcDEF123_0.mkv").touch()
    (tmp_path / "xyzXYZ987-1.f251.webm").touch()
    (tmp_path / "qrsQRS456_2.info.json").touch()

    assert count_final_media(tmp_path) == 1
