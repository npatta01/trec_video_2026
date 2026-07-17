"""Tests for the single generic YouTube dataset download engine."""

from pathlib import Path
import subprocess

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
    assert youtube_dataset.dataset_status(spec) == {
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
