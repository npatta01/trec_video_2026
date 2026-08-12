from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import trec_video_2026.vqa_frontier_map_media as media_module
from trec_video_2026.vqa_frontier_map_media import (
    DELIVERY_POLICY_VERSION,
    DeliveryMediaError,
    prepare_delivery_media,
    verify_delivery_media,
)


def _write_mkv(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "0.4",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "100k",
            "-c:a",
            "libopus",
            str(path),
        ],
        check=True,
    )


def test_gemini_mkv_is_losslessly_remuxed_with_packet_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)

    delivery = prepare_delivery_media(
        run_path=tmp_path / "run",
        source_path=source,
        video_id="video-1",
        provider="gemini",
    )

    assert delivery.path == tmp_path / "run" / "delivery-media" / "video-1.webm"
    assert delivery.mime_type == "video/webm"
    assert delivery.policy_version == DELIVERY_POLICY_VERSION
    assert delivery.derived is True
    assert delivery.sha256 != delivery.source_sha256
    provenance = json.loads(delivery.provenance_path.read_text(encoding="utf-8"))
    assert provenance["source_streams"] == provenance["delivery_streams"]
    assert provenance["source_packet_sha256"] == provenance["delivery_packet_sha256"]
    assert set(provenance["source_packet_sha256"]) == {"audio:0", "video:0"}
    assert verify_delivery_media(delivery) == delivery

    repeated = prepare_delivery_media(
        run_path=tmp_path / "run",
        source_path=source,
        video_id="video-1",
        provider="gemini",
    )
    assert repeated == delivery


def test_existing_delivery_tamper_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)
    delivery = prepare_delivery_media(
        run_path=tmp_path / "run",
        source_path=source,
        video_id="video-1",
        provider="gemini",
    )
    delivery.path.write_bytes(delivery.path.read_bytes() + b"tamper")

    with pytest.raises(DeliveryMediaError, match="identity"):
        prepare_delivery_media(
            run_path=tmp_path / "run",
            source_path=source,
            video_id="video-1",
            provider="gemini",
        )


def test_compatible_gemini_media_passes_through_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"bounded fixture media")

    delivery = prepare_delivery_media(
        run_path=tmp_path / "run",
        source_path=source,
        video_id="video-1",
        provider="gemini",
    )

    assert delivery.path == source.resolve()
    assert delivery.mime_type == "video/mp4"
    assert delivery.derived is False
    assert delivery.sha256 == delivery.source_sha256
    assert verify_delivery_media(delivery) == delivery


def test_delivery_refuses_symlinked_run_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(DeliveryMediaError, match="symlink"):
        prepare_delivery_media(
            run_path=linked,
            source_path=source,
            video_id="video-1",
            provider="gemini",
        )


def test_delivery_refuses_concurrent_preparation_claim(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)
    delivery_root = tmp_path / "run" / "delivery-media"
    delivery_root.mkdir(parents=True)
    (delivery_root / ".gemini-video-1.lock").write_text("claimed\n", encoding="utf-8")

    with pytest.raises(DeliveryMediaError, match="claimed"):
        prepare_delivery_media(
            run_path=tmp_path / "run",
            source_path=source,
            video_id="video-1",
            provider="gemini",
        )

    assert not (delivery_root / "video-1.webm").exists()


def test_delivery_target_install_never_overwrites_racing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)
    target = tmp_path / "run/delivery-media/video-1.webm"
    original_link = media_module.os.link

    def racing_link(source_path: object, destination_path: object) -> None:
        destination = Path(destination_path)
        if destination == target:
            destination.write_bytes(b"racing writer")
        original_link(source_path, destination_path)

    monkeypatch.setattr(media_module.os, "link", racing_link)

    with pytest.raises(DeliveryMediaError, match="appeared|exists"):
        prepare_delivery_media(
            run_path=tmp_path / "run",
            source_path=source,
            video_id="video-1",
            provider="gemini",
        )

    assert target.read_bytes() == b"racing writer"


def test_provenance_failure_preserves_installed_delivery_for_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mkv"
    _write_mkv(source)
    target = tmp_path / "run/delivery-media/video-1.webm"

    def fail_provenance(path: Path, payload: object) -> None:
        assert path.name == "gemini-video-1.json"
        assert payload is not None
        raise DeliveryMediaError("injected provenance failure")

    monkeypatch.setattr(media_module, "_write_new_json", fail_provenance)

    with pytest.raises(DeliveryMediaError, match="injected"):
        prepare_delivery_media(
            run_path=tmp_path / "run",
            source_path=source,
            video_id="video-1",
            provider="gemini",
        )

    assert target.is_file()
    assert target.stat().st_size > 0
