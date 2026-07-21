from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

import trec_video_2026.vqa_benchmark as vqa_benchmark
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
NATIVE_AUDIO_CONFIG = REPO_ROOT / "configs/vqa/native_audio_benchmark_30.json"

FROZEN_IDS = [
    "LKiaCf1QZFw",
    "Ie-KCtcZ7uc",
    "J1CS47pBX_M",
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "NUmb2P3Bzu8",
    "IRTTtnSRTyY",
    "IWSPAfCKtV8",
    "Jc_vZjKwtvg",
    "MdMKHhFFqJQ",
    "NkXPPJoZ-h8",
    "OzNdzgQi1CA",
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
    "PcDuz7j9LL8",
    "C4j3Pp79KSg",
    "Hw0bRvn1ZIo",
    "J-sG9vePDmI",
    "Jc2BdtyAF48",
    "HyOKrz1C4zU",
    "Msbpqv5aOxk",
    "JOClNGH6yL8",
    "EMI_anpHII4",
    "IGL7j9-JRG4",
    "JSOh_pKFq84",
    "QFljXc4WpFA",
    "PiIrinu42DQ",
    "LusRurRrmf0",
]
GATE_IDS = [
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "Jc_vZjKwtvg",
    "NkXPPJoZ-h8",
    "EtNOkUJd5w8",
    "PcDuz7j9LL8",
    "Jc2BdtyAF48",
    "HyOKrz1C4zU",
    "EMI_anpHII4",
    "IGL7j9-JRG4",
]
STRATA = {
    "Dialogue and speech",
    "Non-speech sound and music",
    "Temporal and action",
    "Spatial, OCR, and color",
    "Visual controls",
}


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


def test_native_audio_manifest_freezes_version_ids_strata_and_gate():
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)

    assert manifest.manifest_version == "trecvid-native-audio-2026-07-21-v1"
    assert [example.video_id for example in manifest.examples] == FROZEN_IDS
    assert {example.category for example in manifest.examples} == STRATA
    assert [example.video_id for example in manifest.gate_examples] == GATE_IDS
    assert all(
        sum(item.category == category for item in manifest.gate_examples) == 2
        for category in STRATA
    )


def test_native_audio_manifest_uses_shared_root_and_resolves_media_extensions():
    payload = json.loads(NATIVE_AUDIO_CONFIG.read_text(encoding="utf-8"))
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)

    assert DATASET_ROOT.is_symlink()
    assert DATASET_ROOT.resolve() == Path(
        "/home/npatta01/data/competitions/trec_video_2026/data"
    )
    assert all(Path(raw["media"]).suffix == "" for raw in payload["examples"])
    assert all(
        example.media_path.is_relative_to(DATASET_ROOT.resolve())
        for example in manifest.examples
    )
    assert {item.media_path.suffix for item in manifest.examples} == {".mkv", ".mp4"}


def test_native_audio_manifest_freezes_questions_but_not_references():
    payload = json.loads(NATIVE_AUDIO_CONFIG.read_text(encoding="utf-8"))
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)

    assert [item["question"] for item in payload["examples"]] == [
        item.question for item in manifest.examples
    ]
    assert all(
        "answer" not in key.lower() for item in payload["examples"] for key in item
    )
    assert all(
        "reference_answer" not in asdict(item) for item in manifest.inference_examples
    )
    assert all("reference_answer" not in asdict(item) for item in manifest.examples)
    assert all(
        item.reference_answer for item in vqa_benchmark.load_references(manifest)
    )


def test_manifest_loading_does_not_require_reference_values(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "annotations").mkdir(parents=True)
    (dataset_root / "media").mkdir()
    (dataset_root / "annotations/example.json").write_text(
        json.dumps(
            {
                "video_url": "https://www.youtube.com/watch?v=example",
                "question": "What happens?",
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "media/example.mkv").write_bytes(b"not probed")
    config = tmp_path / "manifest.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gate_id": "label-free",
                "examples": [
                    {
                        "video_id": "example",
                        "category": "control",
                        "question": "What happens?",
                        "annotation": "annotations/example.json",
                        "media": "media/example",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(config, dataset_root)

    assert manifest.examples[0].question == "What happens?"
    with pytest.raises(ValueError, match="correct_answer"):
        vqa_benchmark.load_references(manifest)


def test_native_audio_gate_selection_hash_is_deterministic():
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)

    assert manifest.gate_selection_sha256 == (
        "099dedfa4d74aa7ed1c135a013c8b342a62e422a6692934507c94839a2803f69"
    )
    assert manifest.gate_selection_sha256 == vqa_benchmark.gate_selection_hash(
        manifest.manifest_version,
        manifest.gate_marker,
        manifest.gate_examples,
    )


def test_versioned_native_audio_manifest_requires_gate_selection_hash(
    tmp_path: Path,
):
    payload = json.loads(NATIVE_AUDIO_CONFIG.read_text(encoding="utf-8"))
    del payload["gate_selection_sha256"]
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gate_selection_sha256 is required"):
        load_manifest(config, DATASET_ROOT)


@pytest.mark.skipif(
    shutil.which("ffprobe") is None,
    reason="ffprobe is required",
)
def test_native_audio_resolved_manifest_records_media_provenance():
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)
    destination = (
        DATASET_ROOT
        / "vqa/benchmarks/test-artifacts/native_audio_benchmark_30.resolved.json"
    )

    vqa_benchmark.write_resolved_manifest(manifest, destination)

    resolved = json.loads(destination.read_text(encoding="utf-8"))
    assert "references" not in resolved
    assert [item["video_id"] for item in resolved["examples"]] == FROZEN_IDS
    assert all(len(item["source"]["sha256"]) == 64 for item in resolved["examples"])
    assert all(item["source"]["size_bytes"] > 0 for item in resolved["examples"])
    assert all(item["media"]["duration_seconds"] > 0 for item in resolved["examples"])
    assert all(item["media"]["has_video"] for item in resolved["examples"])
    assert all(item["media"]["has_audio"] for item in resolved["examples"])
    assert all("video" in item["media"]["stream_types"] for item in resolved["examples"])
    assert all("audio" in item["media"]["stream_types"] for item in resolved["examples"])
    assert all(item["media"]["streams"] for item in resolved["examples"])
    six_channel = next(
        item for item in resolved["examples"] if item["video_id"] == "EtNOkUJd5w8"
    )
    assert {
        stream["codec_name"] for stream in six_channel["media"]["streams"]
    } == {"vp9", "aac"}
    assert next(
        stream["channels"]
        for stream in six_channel["media"]["streams"]
        if stream["codec_type"] == "audio"
    ) == 6


def test_resolved_manifest_rejects_destinations_outside_ignored_benchmark_data(
    tmp_path: Path,
):
    manifest = load_manifest(NATIVE_AUDIO_CONFIG, DATASET_ROOT)

    with pytest.raises(ValueError, match="data/vqa/benchmarks"):
        vqa_benchmark.write_resolved_manifest(manifest, tmp_path / "tracked.json")


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
    references = [
        asdict(record) for record in vqa_benchmark.load_references(manifest)
    ]

    assert all("reference" not in key for row in inference for key in row)
    assert all("answer" not in key for row in inference for key in row)
    assert references[0]["reference_answer"] == (
        "The bicyclist drives his bike across a tennis net."
    )


def test_manifest_serialization_is_deterministic_and_keeps_labels_separate():
    manifest = load_manifest(GATE_CONFIG, DATASET_ROOT)
    references = vqa_benchmark.load_references(manifest)

    first = serialize_manifest(manifest, references)
    second = serialize_manifest(manifest, references)

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


def test_twelve_and_twenty_four_frame_policies_have_stable_distinct_identities():
    source = SourceFingerprint(sha256="fixed-source", size_bytes=123)
    twelve = EvidenceParameters(frame_count=12)
    twenty_four = EvidenceParameters(frame_count=24)

    assert bin_midpoint_timestamps(24.0, twelve.frame_count) == tuple(
        float(value) for value in range(1, 24, 2)
    )
    assert bin_midpoint_timestamps(24.0, twenty_four.frame_count) == tuple(
        value + 0.5 for value in range(24)
    )
    assert cache_key(source, twelve) != cache_key(source, twenty_four)


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
