"""Tests for the single generic YouTube dataset download engine."""

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
