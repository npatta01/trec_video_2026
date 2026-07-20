from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from trec_video_2026.vqa_benchmark import (
    EvidenceParameters,
    SourceFingerprint,
    StaleEvidenceError,
    bin_midpoint_timestamps,
    build_frame_evidence,
    cache_key,
    load_cached_evidence,
    load_manifest,
    probe_media,
    serialize_manifest,
    source_fingerprint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data"
GATE_CONFIG = REPO_ROOT / "configs/vqa/three_video_gate.json"


def test_fixed_manifest_resolves_annotations_and_media_beneath_dataset():
    manifest = load_manifest(GATE_CONFIG, DATASET_ROOT)

    assert [example.video_id for example in manifest.examples] == [
        "BDKkVYmWqdM",
        "O60MtMuSpf8",
        "Fk_j6re_Uiw",
    ]
    assert all(example.annotation_path.is_file() for example in manifest.examples)
    assert all(example.media_path.is_file() for example in manifest.examples)
    assert all(
        example.media_path.is_relative_to(DATASET_ROOT.resolve())
        for example in manifest.examples
    )


def test_manifest_rejects_paths_outside_dataset(tmp_path: Path):
    config = tmp_path / "manifest.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "bad",
                "examples": [
                    {
                        "video_id": "BDKkVYmWqdM",
                        "category": "temporal",
                        "annotation": "../outside.json",
                        "media": "vqa/2025/training/videos_original/BDKkVYmWqdM.mkv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="beneath the dataset root"):
        load_manifest(config, DATASET_ROOT)


def test_inference_records_do_not_contain_reference_answers():
    manifest = load_manifest(GATE_CONFIG, DATASET_ROOT)

    inference = [asdict(record) for record in manifest.inference_examples]
    references = [asdict(record) for record in manifest.references]

    assert all("reference" not in key for row in inference for key in row)
    assert all("answer" not in key for row in inference for key in row)
    assert references[0]["reference_answer"] == (
        "The bicyclist drives his bike across a tennis net."
    )


def test_manifest_serialization_is_deterministic_and_keeps_labels_separate():
    manifest = load_manifest(GATE_CONFIG, DATASET_ROOT)

    first = serialize_manifest(manifest)
    second = serialize_manifest(manifest)

    assert first == second
    payload = json.loads(first)
    assert list(payload) == ["schema_version", "gate_id", "examples", "references"]
    assert "reference_answer" not in payload["examples"][0]
    assert payload["references"][0]["video_id"] == payload["examples"][0]["video_id"]


@pytest.mark.parametrize("video_id", ["BDKkVYmWqdM", "O60MtMuSpf8", "Fk_j6re_Uiw"])
def test_fixed_media_has_video_and_audio_streams(video_id: str):
    manifest = load_manifest(GATE_CONFIG, DATASET_ROOT)
    example = next(item for item in manifest.examples if item.video_id == video_id)

    metadata = probe_media(example.media_path)

    assert metadata.duration_seconds > 0
    assert metadata.has_video
    assert metadata.has_audio


def test_bin_midpoint_timestamps_cover_twelve_equal_bins():
    assert bin_midpoint_timestamps(24.0) == tuple(float(value) for value in range(1, 24, 2))
    assert bin_midpoint_timestamps(1.0, count=4) == (0.125, 0.375, 0.625, 0.875)


def test_source_fingerprint_changes_with_media_content(tmp_path: Path):
    media = tmp_path / "clip.bin"
    media.write_bytes(b"first")
    first = source_fingerprint(media)
    media.write_bytes(b"second")

    assert source_fingerprint(media) != first
    assert first.sha256 == "a7937b64b8caa58f03721bb6bacf5c78cb235febe0e70b1b84cd99541461a08e"
    assert first.size_bytes == 5


def test_cache_key_covers_source_and_extraction_parameters(tmp_path: Path):
    media = tmp_path / "clip.bin"
    media.write_bytes(b"content")
    fingerprint = source_fingerprint(media)

    key = cache_key(fingerprint, EvidenceParameters(frame_count=12))

    assert key == cache_key(fingerprint, EvidenceParameters(frame_count=12))
    assert key != cache_key(fingerprint, EvidenceParameters(frame_count=6))


def test_cached_evidence_verifies_checksums_and_rejects_stale_files(tmp_path: Path):
    cache_dir = tmp_path / "cached"
    cache_dir.mkdir()
    frame = cache_dir / "frame-01.png"
    frame.write_bytes(b"valid")
    source = SourceFingerprint(sha256="source", size_bytes=6)
    parameters = EvidenceParameters(frame_count=1)
    expected_key = cache_key(source, parameters)
    record = {
        "schema_version": 1,
        "cache_key": expected_key,
        "source": asdict(source),
        "parameters": asdict(parameters),
        "duration_seconds": 1.0,
        "timestamps_seconds": [0.5],
        "frames": [
            {
                "path": "frame-01.png",
                "sha256": source_fingerprint(frame).sha256,
            }
        ],
    }
    (cache_dir / "evidence.json").write_text(json.dumps(record), encoding="utf-8")

    assert load_cached_evidence(cache_dir, expected_key).cache_key == expected_key
    frame.write_bytes(b"corrupt")

    with pytest.raises(StaleEvidenceError, match="checksum"):
        load_cached_evidence(cache_dir, expected_key)


def test_cached_evidence_rejects_metadata_that_does_not_match_cache_key(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cached"
    cache_dir.mkdir()
    frame = cache_dir / "frame-01.png"
    frame.write_bytes(b"valid")
    source = SourceFingerprint(sha256="source", size_bytes=6)
    parameters = EvidenceParameters(frame_count=1)
    expected_key = cache_key(source, parameters)
    record = {
        "schema_version": 1,
        "cache_key": expected_key,
        "source": {"sha256": "tampered", "size_bytes": 6},
        "parameters": asdict(parameters),
        "duration_seconds": 1.0,
        "timestamps_seconds": [0.5],
        "frames": [
            {
                "path": "frame-01.png",
                "sha256": source_fingerprint(frame).sha256,
            }
        ],
    }
    (cache_dir / "evidence.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(StaleEvidenceError, match="identity"):
        load_cached_evidence(cache_dir, expected_key)


def test_cached_evidence_rejects_non_midpoint_timestamps(tmp_path: Path):
    cache_dir = tmp_path / "cached"
    cache_dir.mkdir()
    frame = cache_dir / "frame-01.png"
    frame.write_bytes(b"valid")
    source = SourceFingerprint(sha256="source", size_bytes=6)
    parameters = EvidenceParameters(frame_count=1)
    expected_key = cache_key(source, parameters)
    record = {
        "schema_version": 1,
        "cache_key": expected_key,
        "source": asdict(source),
        "parameters": asdict(parameters),
        "duration_seconds": 1.0,
        "timestamps_seconds": [0.25],
        "frames": [
            {
                "path": "frame-01.png",
                "sha256": source_fingerprint(frame).sha256,
            }
        ],
    }
    (cache_dir / "evidence.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(StaleEvidenceError, match="timestamp"):
        load_cached_evidence(cache_dir, expected_key)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg tools are required",
)
def test_frame_evidence_integration_extracts_twelve_timestamped_images(tmp_path: Path):
    media = tmp_path / "tiny.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(media),
        ],
        check=True,
    )

    evidence = build_frame_evidence(media, tmp_path / "evidence")

    assert len(evidence.frames) == 12
    assert len(evidence.timestamps_seconds) == 12
    assert all(frame.path.is_file() for frame in evidence.frames)
    assert evidence.timestamps_seconds == bin_midpoint_timestamps(2.0)
    assert load_cached_evidence(evidence.cache_dir, evidence.cache_key) == evidence
