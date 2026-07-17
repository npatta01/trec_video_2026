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
    assert vqa_download.read_video_ids.__module__ == "trec_video_2026.vqa_download"
    assert vqa_download.prepare_manifests.__module__ == (
        "trec_video_2026.vqa_download"
    )
