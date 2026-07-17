"""Tests for the single generic YouTube dataset download engine."""

from pathlib import Path

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
        "start_detached",
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
