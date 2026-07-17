"""Tests for the single generic YouTube dataset download engine."""

from pathlib import Path
import json
import subprocess

import pytest

from trec_video_2026 import youtube_dataset
from trec_video_2026 import vqa_download


def test_generic_engine_owns_reusable_download_behavior():
    reusable_names = (
        "load_youtube_api_key",
        "fetch_available_video_ids",
        "load_proxy_credentials",
        "write_proxy_config",
        "build_ytdlp_command",
        "run_url",
        "run_dataset",
    )

    for name in reusable_names:
        generic = getattr(youtube_dataset, name)
        compatibility_export = getattr(vqa_download, name)
        assert generic.__module__ == "trec_video_2026.youtube_dataset"
        assert compatibility_export is generic


def test_vqa_specific_manifest_parsing_remains_in_vqa_adapter():
    adapter_names = (
        "read_video_ids",
        "read_unavailable_ids",
        "write_manifest",
        "prepare_manifests",
        "prepare_available_manifests",
        "run_downloads",
        "start_detached",
        "main",
    )

    for name in adapter_names:
        assert getattr(vqa_download, name).__module__ == "trec_video_2026.vqa_download"

    assert not hasattr(youtube_dataset, "DEFAULT_PROXY_ENV_PATH")
    assert not hasattr(youtube_dataset, "DEFAULT_YOUTUBE_API_ENV_PATH")


def test_repeated_proxy_502_failures_stop_the_pass_promptly(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n"
        "https://www.youtube.com/watch?v=qrsQRS456_2\n",
        encoding="utf-8",
    )
    calls = []

    def runner(command):
        calls.append(command[-1])
        return youtube_dataset.AttemptResult(False, False, proxy_failure=True)

    youtube_dataset.run_dataset(
        manifest,
        tmp_path / "videos",
        tmp_path / "archive.txt",
        url_runner=runner,
        sleeper=lambda _: None,
    )

    assert len(calls) == 2


def test_ordinary_video_failures_do_not_trigger_proxy_stop(tmp_path: Path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
        "https://www.youtube.com/watch?v=xyzXYZ987-1\n"
        "https://www.youtube.com/watch?v=qrsQRS456_2\n",
        encoding="utf-8",
    )
    calls = []

    def runner(command):
        calls.append(command[-1])
        return youtube_dataset.AttemptResult(False, False)

    youtube_dataset.run_dataset(
        manifest,
        tmp_path / "videos",
        tmp_path / "archive.txt",
        url_runner=runner,
        sleeper=lambda _: None,
    )

    assert len(calls) == 3


def test_run_url_classifies_bad_gateway_as_proxy_failure():
    class Child:
        stdout = iter(["ERROR: Proxy returned HTTP Error 502: Bad Gateway\n"])

        def wait(self):
            return 1

    result = youtube_dataset.run_url(
        ["yt-dlp", "url"], popen=lambda *args, **kwargs: Child()
    )

    assert result.proxy_failure is True


@pytest.mark.parametrize("line", [
    "ERROR: HTTP Error 429: Too Many Requests\n",
    "ERROR: Unable to download webpage: HTTP Error 429\n",
])
def test_run_url_classifies_explicit_http_429_as_rate_limit(line):
    class Child:
        stdout = iter([line])
        def wait(self): return 1
    result = youtube_dataset.run_url(["yt-dlp", "url"], popen=lambda *args, **kwargs: Child())
    assert result.rate_limited is True


def test_run_url_does_not_treat_unrelated_429_digits_as_rate_limit():
    class Child:
        stdout = iter(["[download] 429 bytes received\n"])
        def wait(self): return 1
    assert youtube_dataset.run_url(["yt-dlp", "url"], popen=lambda *a, **k: Child()).rate_limited is False


def test_prepare_publishes_one_authoritative_atomic_snapshot(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(tmp_path / "source.txt", tmp_path / "media", tmp_path / "state")
    spec.source.write_text("abcDEF123_0\nxyzXYZ987-1\n")
    youtube_dataset.prepare_dataset(spec, "key", fetcher=lambda ids, key: {ids[0]})
    snapshot = json.loads((spec.state_dir / "availability-snapshot.json").read_text())
    assert snapshot["visible"] == ["abcDEF123_0"]
    assert snapshot["omitted"] == ["xyzXYZ987-1"]
    assert snapshot["counts"] == {"total": 2, "pending_available": 1, "api_omitted": 1}


def test_snapshot_remains_authoritative_if_legacy_projection_fails(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(tmp_path / "source.txt", tmp_path / "media", tmp_path / "state")
    spec.source.write_text("abcDEF123_0\n")
    original = youtube_dataset._atomic_write_text
    def injected(path, contents):
        if path.name == "pending-available.urls.txt":
            raise OSError("injected")
        return original(path, contents)
    monkeypatch.setattr(youtube_dataset, "_atomic_write_text", injected)
    with pytest.raises(OSError, match="injected"):
        youtube_dataset.prepare_dataset(spec, "key", fetcher=lambda ids, key: set(ids))
    assert json.loads((spec.state_dir / "availability-snapshot.json").read_text())["visible"] == ["abcDEF123_0"]


def test_snapshot_publication_failure_preserves_previous_authority(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(tmp_path / "source.txt", tmp_path / "media", tmp_path / "state")
    spec.source.write_text("abcDEF123_0\n")
    spec.state_dir.mkdir()
    snapshot_path = spec.state_dir / "availability-snapshot.json"
    old = {"version": 1, "visible": [], "omitted": ["abcDEF123_0"], "counts": {"total": 1, "pending_available": 0, "api_omitted": 1}}
    youtube_dataset._atomic_write_json(snapshot_path, old)
    original_replace = youtube_dataset.os.replace
    def fail_snapshot_replace(source, destination):
        if Path(destination) == snapshot_path:
            raise OSError("injected publication failure")
        return original_replace(source, destination)
    monkeypatch.setattr(youtube_dataset.os, "replace", fail_snapshot_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        youtube_dataset.prepare_dataset(spec, "key", fetcher=lambda ids, key: set(ids))
    assert json.loads(snapshot_path.read_text()) == old


def test_run_spec_derives_manifest_from_authoritative_snapshot(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(tmp_path / "source", tmp_path / "media", tmp_path / "state")
    spec.state_dir.mkdir()
    youtube_dataset._atomic_write_json(spec.state_dir / "availability-snapshot.json", {
        "version": 1, "visible": ["abcDEF123_0"], "omitted": [],
        "counts": {"total": 1, "pending_available": 1, "api_omitted": 0},
    })
    (spec.state_dir / "pending-available.urls.txt").write_text("https://www.youtube.com/watch?v=xyzXYZ987-1\n")
    calls = []
    youtube_dataset.run_spec(spec, url_runner=lambda command: calls.append(command[-1]) or youtube_dataset.AttemptResult(False, False), sleeper=lambda _: None)
    assert calls == ["https://www.youtube.com/watch?v=abcDEF123_0"]


def test_runtime_state_persists_progress_cooldown_and_failure_categories(tmp_path: Path):
    manifest = tmp_path / "pending.txt"
    archive = tmp_path / "download.archive.txt"
    manifest.write_text("https://www.youtube.com/watch?v=abcDEF123_0\n")
    sleeps = []
    attempts = iter([
        youtube_dataset.AttemptResult(False, True, failure_category="rate_limit", retryable=True),
        youtube_dataset.AttemptResult(False, False, failure_category="retryable", retryable=True),
    ])
    youtube_dataset.run_dataset(
        manifest, tmp_path / "media", archive,
        url_runner=lambda command: next(attempts),
        sleeper=lambda seconds: sleeps.append(json.loads((tmp_path / "runtime-state.json").read_text())["cooldown_deadline"]),
        initial_cooldown=0,
    )
    state = json.loads((tmp_path / "runtime-state.json").read_text())
    assert sleeps[0] is not None
    assert state["cooldown_deadline"] is None
    assert state["failures"]["rate_limit"] == ["abcDEF123_0"]
    assert state["retryable_failures"] == ["abcDEF123_0"]


def test_non_vqa_dataset_spec_drives_api_preparation_and_run_paths(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(
        source=tmp_path / "arbitrary/input.txt",
        output_dir=tmp_path / "elsewhere/media",
        state_dir=tmp_path / "custom-state",
    )
    spec.source.parent.mkdir(parents=True)
    spec.source.write_text(
        "abcDEF123_0\nhttps://www.youtube.com/watch?v=xyzXYZ987-1\n",
        encoding="utf-8",
    )

    config = youtube_dataset.DownloadConfig(
        api_key_file=tmp_path / "credentials/api.env",
        proxy_file=tmp_path / "credentials/proxy.env",
    )
    seen = []
    monkeypatch.setattr(
        youtube_dataset,
        "load_youtube_api_key",
        lambda path: seen.append(path) or "placeholder-key",
    )
    summary = youtube_dataset.prepare_dataset(spec, config, fetcher=lambda ids, key: {ids[0]})

    assert summary == {"total": 2, "pending_available": 1, "api_omitted": 1}
    assert seen == [config.api_key_file]
    assert (spec.state_dir / "pending-available.urls.txt").read_text() == (
        "https://www.youtube.com/watch?v=abcDEF123_0\n"
    )
    calls = []
    youtube_dataset.run_spec(
        spec,
        url_runner=lambda command: calls.append(command) or youtube_dataset.AttemptResult(False, False),
        sleeper=lambda _: None,
    )
    assert calls[0][-1] == "https://www.youtube.com/watch?v=abcDEF123_0"
    assert any(str(spec.output_dir) in argument for argument in calls[0])
    assert str(spec.state_dir / "download.archive.txt") in calls[0]
    status = youtube_dataset.dataset_status(spec)
    assert {key: status[key] for key in (
        "running", "completed", "media_files", "pending_available", "api_omitted"
    )} == {
        "running": False,
        "completed": 0,
        "media_files": 0,
        "pending_available": 1,
        "api_omitted": 1,
    }


def test_detached_spec_binds_explicit_worker_command_and_config(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(
        source=tmp_path / "input.txt", output_dir=tmp_path / "media", state_dir=tmp_path / "state"
    )
    proxy_file = tmp_path / "proxy.env"
    config = youtube_dataset.DownloadConfig(
        api_key_file=tmp_path / "api.env",
        proxy_file=proxy_file,
        worker_command=("python", "worker.py", "--run"),
        node_path="/opt/node/bin/node",
    )
    captured = {}

    def fake_write(source, target, runner=subprocess.run):
        target.write_text("placeholder\n")
        return target

    class Child:
        pid = 4242

    monkeypatch.setattr(youtube_dataset, "write_proxy_config", fake_write)
    pid = youtube_dataset.start_detached(
        spec,
        config,
        popen=lambda command, **kwargs: captured.update(command=command, kwargs=kwargs) or Child(),
    )

    assert pid == 4242
    assert captured["command"] == list(config.worker_command)
    assert captured["kwargs"]["cwd"] == spec.state_dir


def _generic_args(tmp_path: Path, command: str) -> list[str]:
    return [
        command,
        "--source", str(tmp_path / "source.txt"),
        "--output-dir", str(tmp_path / "media"),
        "--state-dir", str(tmp_path / "state"),
        "--api-env", str(tmp_path / "api.env"),
        "--proxy-env", str(tmp_path / "proxy.env"),
    ]


def test_generic_cli_requires_every_explicit_path():
    with pytest.raises(SystemExit) as caught:
        youtube_dataset.main(["status", "--source", "input.txt"])
    assert caught.value.code == 2


def test_generic_prepare_routes_through_shared_engine(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "source.txt").write_text("abcDEF123_0\n")
    seen = {}
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda spec, allow: seen.update(safety=(spec, allow)))
    monkeypatch.setattr(youtube_dataset, "prepare_dataset", lambda spec, config: seen.update(call=(spec, config)) or {"total": 1, "pending_available": 1, "api_omitted": 0})

    assert youtube_dataset.main(_generic_args(tmp_path, "prepare")) == 0
    spec, config = seen["call"]
    assert spec.source == tmp_path / "source.txt"
    assert spec.output_dir == tmp_path / "media"
    assert spec.state_dir == tmp_path / "state"
    assert config.api_key_file == tmp_path / "api.env"
    assert config.proxy_file == tmp_path / "proxy.env"
    assert "pending available: 1" in capsys.readouterr().out


def test_generic_run_fails_closed_before_download_on_proxy_preflight(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    monkeypatch.setattr(youtube_dataset, "prepare_dataset", lambda spec, config: calls.append("api") or {})
    monkeypatch.setattr(youtube_dataset, "preflight_proxy", lambda path: (_ for _ in ()).throw(RuntimeError("proxy unavailable")))
    monkeypatch.setattr(youtube_dataset, "run_spec", lambda *args, **kwargs: calls.append("download"))

    assert youtube_dataset.main(_generic_args(tmp_path, "run")) == 1
    assert calls == ["api"]


def test_generic_run_fails_closed_before_proxy_or_download_on_api_preflight(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    monkeypatch.setattr(youtube_dataset, "prepare_dataset", lambda *args: (_ for _ in ()).throw(RuntimeError("API unavailable")))
    monkeypatch.setattr(youtube_dataset, "preflight_proxy", lambda path: calls.append("proxy"))
    monkeypatch.setattr(youtube_dataset, "run_spec", lambda *args, **kwargs: calls.append("download"))
    assert youtube_dataset.main(_generic_args(tmp_path, "run")) == 1
    assert calls == []


def test_generic_status_uses_shared_status_without_preflight(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    monkeypatch.setattr(youtube_dataset, "dataset_status", lambda spec: {"running": False, "completed": 2, "media_files": 2, "pending_available": 3, "api_omitted": 1})
    assert youtube_dataset.main(_generic_args(tmp_path, "status")) == 0
    output = capsys.readouterr().out
    assert "state: stopped" in output
    assert "completed: 2" in output


def test_generic_status_prints_persisted_progress_cooldown_and_failures(tmp_path: Path, monkeypatch, capsys):
    state_dir = tmp_path / "state"
    youtube_dataset._atomic_write_json(state_dir / "runtime-state.json", {
        "worker_pid": None, "current_id": None,
        "last_progress_at": "2026-07-17T12:00:00+00:00",
        "cooldown_deadline": "2026-07-17T13:00:00+00:00",
        "failures": {"private": ["abcDEF123_0"], "members_only": ["xyzXYZ987-1"]},
        "retryable_failures": ["qrsQRS456_2"],
    })
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    assert youtube_dataset.main(_generic_args(tmp_path, "status")) == 0
    output = capsys.readouterr().out
    assert "pid: none" in output
    assert "last progress: 2026-07-17T12:00:00+00:00" in output
    assert "cooldown deadline: 2026-07-17T13:00:00+00:00" in output
    assert "failures private: 1" in output
    assert "failures members-only: 1" in output
    assert "retryable failures: 1" in output


def test_git_tracked_output_or_state_is_refused_unless_overridden(tmp_path: Path, monkeypatch):
    spec = youtube_dataset.DatasetSpec(tmp_path / "source", tmp_path / "output", tmp_path / "state")
    monkeypatch.setattr(youtube_dataset, "_git_tracked", lambda path: path == spec.output_dir)
    with pytest.raises(RuntimeError, match="tracked by Git"):
        youtube_dataset.refuse_git_tracked_runtime_paths(spec, False)
    youtube_dataset.refuse_git_tracked_runtime_paths(spec, True)


def test_generic_start_prepares_then_uses_shared_detached_launcher(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    monkeypatch.setattr(youtube_dataset, "prepare_dataset", lambda *args: calls.append("prepare") or {"pending_available": 1})
    monkeypatch.setattr(youtube_dataset, "preflight_proxy", lambda *args: calls.append("proxy"))
    monkeypatch.setattr(youtube_dataset, "start_detached", lambda spec, config: calls.append((spec, config)) or 4321)
    assert youtube_dataset.main(_generic_args(tmp_path, "start")) == 0
    assert calls[:2] == ["prepare", "proxy"]
    assert calls[2][1].worker_command[3] == "run"
    assert calls[2][1].worker_command[-1] == "--detached-worker"


def test_generic_start_resolves_relative_paths_before_detached_worker_changes_cwd(
    tmp_path: Path, monkeypatch
):
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    captured = {}
    monkeypatch.setattr(youtube_dataset, "refuse_git_tracked_runtime_paths", lambda *args: None)
    monkeypatch.setattr(youtube_dataset, "prepare_dataset", lambda *args: {"pending_available": 0})
    monkeypatch.setattr(youtube_dataset, "preflight_proxy", lambda *args: None)
    monkeypatch.setattr(
        youtube_dataset,
        "start_detached",
        lambda spec, config: captured.update(spec=spec, config=config) or 1234,
    )

    assert youtube_dataset.main([
        "start", "--source", "inputs/source.txt", "--output-dir", "media",
        "--state-dir", "runtime", "--api-env", "secrets/api.env",
        "--proxy-env", "secrets/proxy.env",
    ]) == 0

    spec = captured["spec"]
    config = captured["config"]
    assert spec.source == invocation / "inputs/source.txt"
    assert spec.output_dir == invocation / "media"
    assert spec.state_dir == invocation / "runtime"
    assert config.api_key_file == invocation / "secrets/api.env"
    assert config.proxy_file == invocation / "secrets/proxy.env"
    for relative in ("inputs/source.txt", "media", "runtime", "secrets/api.env", "secrets/proxy.env"):
        assert relative not in config.worker_command
