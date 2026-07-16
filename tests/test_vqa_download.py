import os
from pathlib import Path

import pytest

from trec_video_2026.vqa_download import (
    AttemptResult,
    RATE_LIMIT_MARKER,
    build_ytdlp_command,
    count_final_media,
    main,
    pid_is_active,
    prepare_manifests,
    read_archive_ids,
    read_unavailable_ids,
    read_video_ids,
    run_url,
    run_downloads,
    start_detached,
    write_manifest,
)


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


def test_run_url_streams_output_and_detects_rate_limit(capsys):
    class Child:
        stdout = iter(["progress\n", f"ERROR: {RATE_LIMIT_MARKER}\n"])

        def wait(self):
            return 1

    result = run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())

    assert result == AttemptResult(completed=False, rate_limited=True)
    assert "progress" in capsys.readouterr().out


def test_run_downloads_processes_training_before_testing(tmp_path: Path):
    training = tmp_path / "data/vqa/2025/training"
    testing = tmp_path / "data/vqa/2025/testing"
    (training / "annotations").mkdir(parents=True)
    testing.mkdir(parents=True)
    (training / "annotations/train.2025.csv").write_text(
        '1 , abcDEF123_0, "q"\n', encoding="utf-8"
    )
    (training / "README.txt").write_text("", encoding="utf-8")
    (testing / "testing.dataset.vqa.2025.csv").write_text(
        '1 , xyzXYZ987-1, "q"\n', encoding="utf-8"
    )
    (testing / "README.txt").write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(command: list[str], check: bool):
        calls.append(command)

    assert run_downloads(tmp_path, runner=runner) == 0
    assert len(calls) == 2
    assert "training.urls.txt" in " ".join(calls[0])
    assert "testing.urls.txt" in " ".join(calls[1])


def test_pid_is_active_rejects_missing_and_stale_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "download.pid"
    assert pid_is_active(pid_path) is False
    pid_path.write_text("12345\n", encoding="utf-8")

    def missing_process(pid: int, signal: int):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", missing_process)
    assert pid_is_active(pid_path) is False


def test_start_detached_records_child_pid_and_node_path(tmp_path: Path):
    runtime = tmp_path / "data/vqa/download_runtime"
    captured: dict[str, object] = {}

    class Child:
        pid = 4242

    def popen(command: list[str], **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return Child()

    assert start_detached(
        tmp_path, popen=popen, node_path="/opt/node/bin/node"
    ) == 4242
    assert (runtime / "download.pid").read_text(encoding="utf-8") == "4242\n"
    assert captured["start_new_session"] is True
    assert captured["cwd"] == tmp_path
    assert captured["stderr"] == -2
    assert captured["env"]["VQA_NODE_PATH"] == "/opt/node/bin/node"


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
