"""Contract tests for the two-row Gemini 3.7 Flash smoke protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import trec_video_2026.vqa_gemini37_flash_smoke as smoke
from trec_video_2026.vqa_gemini37_flash_smoke import (
    ARMS,
    build_inlined_request,
    prepare_protocol,
    verify_inputs,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def historical_fixture(tmp_path: Path) -> dict[str, Path]:
    source_root = tmp_path / "historical"
    source_root.mkdir()
    deliveries = source_root / "deliveries"
    deliveries.mkdir()

    source_rows: list[dict[str, Any]] = []
    upload_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    sources = source_root / "sources"
    sources.mkdir()
    for index in range(475):
        row_id = f"row-{index:03d}"
        video_id = f"video-{index:03d}"
        duration = 18.308 if index == 0 else 21.268 if index == 1 else 1.0
        source_row: dict[str, Any] = {
            "row_id": row_id,
            "video_id": video_id,
            "source_order_index": index,
            "question": f"What happens in {video_id}?",
            "duration_seconds": duration,
        }
        if index < 2:
            source_path = sources / f"{video_id}.mkv"
            source_path.write_bytes(f"synthetic source {video_id}".encode())
            source_row.update(
                {
                    "source_path": str(source_path),
                    "source_sha256": _sha256(source_path),
                    "source_size_bytes": source_path.stat().st_size,
                }
            )
        source_rows.append(source_row)
        if index < 2:
            delivery_path = deliveries / f"{video_id}.webm"
            delivery_path.write_bytes(f"synthetic video {video_id}".encode())
            upload_rows.append(
                {
                    "row_id": row_id,
                    "video_id": video_id,
                    "source_sha256": source_row["source_sha256"],
                    "source_size_bytes": source_row["source_size_bytes"],
                    "file_uri": f"gs://frozen-smoke/{video_id}.webm",
                    "delivery_sha256": _sha256(delivery_path),
                    "delivery_size_bytes": delivery_path.stat().st_size,
                    "delivery_mime_type": "video/webm",
                }
            )
            provenance_rows.append(
                {
                    "row_id": row_id,
                    "video_id": video_id,
                    "delivery_path": str(delivery_path.relative_to(source_root)),
                    "sha256": _sha256(delivery_path),
                    "delivery_size_bytes": delivery_path.stat().st_size,
                    "delivery_mime_type": "video/webm",
                    "source_sha256": source_row["source_sha256"],
                    "source_size_bytes": source_row["source_size_bytes"],
                    "duration_seconds": duration,
                }
            )

    source_protocol = source_root / "source-protocol.json"
    historical_upload = source_root / "historical-upload.json"
    delivery_provenance = source_root / "delivery-provenance.json"
    _write_json(
        source_protocol,
        {
            "schema_version": 1,
            "experiment_id": "direct-gemini-batch-475-v1",
            "rows": source_rows,
        },
    )
    _write_json(
        historical_upload,
        {"schema_version": 1, "rows": upload_rows},
    )
    _write_json(
        delivery_provenance,
        {"schema_version": 1, "rows": provenance_rows},
    )
    return {
        "source_root": source_root,
        "source_protocol": source_protocol,
        "historical_upload": historical_upload,
        "delivery_provenance": delivery_provenance,
        "deliveries": deliveries,
    }


@pytest.fixture
def canonical_fixture(tmp_path: Path) -> dict[str, Path]:
    source_root = tmp_path / "canonical"
    source_root.mkdir()
    sources = source_root / "source-media"
    uploads = source_root / "uploads"
    deliveries = source_root / "deliveries"
    sources.mkdir()
    uploads.mkdir()
    deliveries.mkdir()

    rows: list[dict[str, Any]] = []
    for index in range(475):
        video_id = f"video-{index:03d}"
        duration = 18.308 if index == 0 else 21.268 if index == 1 else 1.0
        source_path = sources / f"{video_id}.mkv"
        if index < 2:
            source_path.write_bytes(f"canonical source {video_id}".encode())
            source_sha256 = _sha256(source_path)
            source_size_bytes = source_path.stat().st_size
        else:
            source_sha256 = "0" * 64
            source_size_bytes = 0
        row = {
            "video_id": video_id,
            "question": f"What happens in {video_id}?",
            "question_stratum": "temporal",
            "duration_seconds": duration,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_size_bytes": source_size_bytes,
        }
        rows.append(row)
        if index < 2:
            delivery_root = deliveries / video_id / "delivery-media"
            delivery_root.mkdir(parents=True)
            delivery_path = delivery_root / f"{video_id}.webm"
            delivery_path.write_bytes(f"canonical delivery {video_id}".encode())
            delivery_sha256 = _sha256(delivery_path)
            delivery_size = delivery_path.stat().st_size
            provenance_path = delivery_root / f"gemini-{video_id}.json"
            _write_json(
                provenance_path,
                {
                    "schema_version": 1,
                    "policy_version": "frontier-map-delivery-v2-webm-copy",
                    "provider": "gemini",
                    "video_id": video_id,
                    "derived": False,
                    "source_sha256": source_sha256,
                    "source_size_bytes": source_size_bytes,
                    "delivery_sha256": delivery_sha256,
                    "delivery_size_bytes": delivery_size,
                    "delivery_mime_type": "video/webm",
                },
            )
            _write_json(
                uploads / f"{video_id}.json",
                {
                    "video_id": video_id,
                    "source_sha256": source_sha256,
                    "source_size_bytes": source_path.stat().st_size,
                    "delivery_path": str(delivery_path),
                    "delivery_sha256": delivery_sha256,
                    "delivery_size_bytes": delivery_size,
                    "delivery_mime_type": "video/webm",
                    "file_name": f"files/{video_id}",
                    "file_uri": f"https://generativelanguage.googleapis.com/v1beta/files/{video_id}",
                    "file_state": "ACTIVE",
                },
            )
    protocol_path = source_root / "protocol.json"
    _write_json(
        protocol_path,
        {
            "schema_version": 1,
            "experiment_id": "direct-gemini-batch-475-v1",
            "rows": rows,
        },
    )
    return {
        "source_root": source_root,
        "source_protocol": protocol_path,
        "uploads": uploads,
        "deliveries": deliveries,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _assert_private(value: Any) -> None:
    forbidden = {
        "answer",
        "answers",
        "label",
        "labels",
        "metric",
        "metrics",
        "reference",
        "references",
        "prior_output",
        "prior_outputs",
        "score",
        "scores",
    }
    if isinstance(value, dict):
        assert not forbidden.intersection(key.lower() for key in value)
        for child in value.values():
            _assert_private(child)
    elif isinstance(value, list):
        for child in value:
            _assert_private(child)


def test_prepare_protocol_binds_two_regular_deliveries_without_private_scores(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    root = tmp_path / "smoke"
    protocol = prepare_protocol(
        source_root=historical_fixture["source_root"],
        root=root,
        created_at="2026-08-16T12:00:00Z",
    )

    assert len(protocol["rows"]) == 2
    assert protocol["rows"][0]["source_order_index"] == 0
    assert protocol["rows"][1]["source_order_index"] == 1
    assert protocol["total_duration_seconds"] == 39.576
    assert protocol["estimated_cost_ceiling_usd"] <= 0.15
    assert protocol["arms"] == ARMS
    _assert_private(protocol)

    assert protocol["source_protocol"]["sha256"] == _sha256(
        historical_fixture["source_protocol"]
    )
    assert protocol["historical_upload"]["sha256"] == _sha256(
        historical_fixture["historical_upload"]
    )
    assert protocol["delivery_provenance"]["sha256"] == _sha256(
        historical_fixture["delivery_provenance"]
    )
    for row in protocol["rows"]:
        assert row["delivery_sha256"] == _sha256(
            historical_fixture["source_root"] / row["delivery_path"]
        )
        assert row["historical_upload_sha256"] == protocol["historical_upload"][
            "sha256"
        ]
        assert row["delivery_provenance_sha256"] == protocol["delivery_provenance"][
            "sha256"
        ]
    assert (root / "protocol.json").is_file()
    assert (root / "inputs-freeze.json").is_file()


def test_prepare_protocol_freezes_schema_and_excludes_historical_private_fields(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    protocol = prepare_protocol(
        source_root=historical_fixture["source_root"],
        root=tmp_path / "smoke",
        created_at="2026-08-16T12:00:00Z",
    )

    assert protocol["schema_version"] == 1
    assert protocol["experiment_id"] == "gemini37-flash-two-video-smoke-v1"
    assert protocol["status"] == "inputs_frozen"
    assert set(protocol["rows"][0]) == {
        "row_id",
        "video_id",
        "source_order_index",
        "question",
        "duration_seconds",
        "source_sha256",
        "source_size_bytes",
        "delivery_path",
        "delivery_sha256",
        "delivery_size_bytes",
        "delivery_mime_type",
        "historical_upload_sha256",
        "delivery_provenance_sha256",
    }


def test_prepare_protocol_adapts_canonical_per_video_layout(
    canonical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    protocol = prepare_protocol(
        source_root=canonical_fixture["source_root"],
        root=tmp_path / "smoke",
        created_at="2026-08-16T12:00:00Z",
    )

    assert [row["video_id"] for row in protocol["rows"]] == [
        "video-000",
        "video-001",
    ]
    assert [row["source_order_index"] for row in protocol["rows"]] == [0, 1]
    assert [row["duration_seconds"] for row in protocol["rows"]] == [18.308, 21.268]
    assert all(row["delivery_size_bytes"] > 0 for row in protocol["rows"])
    assert all(row["delivery_mime_type"] == "video/webm" for row in protocol["rows"])
    assert all(len(row["source_sha256"]) == 64 for row in protocol["rows"])


def test_canonical_scan_reads_only_selected_per_row_records_and_media(
    canonical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    protocol = prepare_protocol(
        source_root=canonical_fixture["source_root"],
        root=tmp_path / "smoke",
        created_at="2026-08-16T12:00:00Z",
    )

    assert [row["source_order_index"] for row in protocol["rows"]] == [0, 1]
    assert not (canonical_fixture["uploads"] / "video-002.json").exists()
    assert not (
        canonical_fixture["source_root"] / "source-media" / "video-002.mkv"
    ).exists()


@pytest.mark.parametrize("bad_id", ["../escape", "video/escape", "video.json"])
def test_prepare_protocol_rejects_unsafe_canonical_video_ids(
    canonical_fixture: dict[str, Path], tmp_path: Path, bad_id: str
) -> None:
    path = canonical_fixture["source_protocol"]
    source = _read_json(path)
    source["rows"][0]["video_id"] = bad_id
    _write_json(path, source)

    with pytest.raises(ValueError, match="video ID"):
        prepare_protocol(
            source_root=canonical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


@pytest.mark.parametrize("directory", ["uploads", "deliveries"])
def test_prepare_protocol_rejects_symlinked_canonical_directories(
    canonical_fixture: dict[str, Path], tmp_path: Path, directory: str
) -> None:
    original = canonical_fixture[directory]
    target = canonical_fixture["source_root"] / f"{directory}-real"
    original.rename(target)
    original.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        prepare_protocol(
            source_root=canonical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_and_verify_accept_relative_canonical_source_root(
    canonical_fixture: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_source_root = Path("canonical")
    root = tmp_path / "smoke"
    prepare_protocol(
        source_root=relative_source_root,
        root=root,
        created_at="2026-08-16T12:00:00Z",
    )

    assert verify_inputs(root)["status"] == "verified"


def test_prepare_protocol_rejects_wrong_canonical_source_identity(
    canonical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    path = canonical_fixture["source_protocol"]
    source = _read_json(path)
    source["experiment_id"] = "another-experiment"
    _write_json(path, source)

    with pytest.raises(ValueError, match="experiment"):
        prepare_protocol(
            source_root=canonical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


@pytest.mark.parametrize("mutation", ["count", "duplicate", "duration"])
def test_prepare_protocol_rejects_invalid_source_protocol(
    historical_fixture: dict[str, Path], tmp_path: Path, mutation: str
) -> None:
    path = historical_fixture["source_protocol"]
    source = _read_json(path)
    if mutation == "count":
        source["rows"].pop()
    elif mutation == "duplicate":
        source["rows"][1]["row_id"] = source["rows"][0]["row_id"]
    else:
        source["rows"][1]["duration_seconds"] = 19.577
    _write_json(path, source)

    with pytest.raises(ValueError):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_same_total_duration_drift(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    path = historical_fixture["source_protocol"]
    source = _read_json(path)
    source["rows"][0]["duration_seconds"] = 18.307
    source["rows"][1]["duration_seconds"] = 21.269
    _write_json(path, source)

    with pytest.raises(ValueError, match="duration"):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


@pytest.mark.parametrize("field", ["delivery_size_bytes", "delivery_mime_type", "source_sha256"])
def test_prepare_protocol_rejects_canonical_upload_identity_mutation(
    canonical_fixture: dict[str, Path], tmp_path: Path, field: str
) -> None:
    path = canonical_fixture["uploads"] / "video-000.json"
    upload = _read_json(path)
    upload[field] = (
        upload[field] + 1
        if field == "delivery_size_bytes"
        else "video/mp4"
        if field == "delivery_mime_type"
        else "f" * 64
    )
    _write_json(path, upload)

    with pytest.raises(ValueError, match="identity|mismatch|size|MIME|source"):
        prepare_protocol(
            source_root=canonical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_canonical_provenance_source_cross_binding(
    canonical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    path = canonical_fixture["deliveries"] / "video-000" / "delivery-media" / "gemini-video-000.json"
    provenance = _read_json(path)
    provenance["source_sha256"] = "f" * 64
    _write_json(path, provenance)

    with pytest.raises(ValueError, match="source|identity|mismatch"):
        prepare_protocol(
            source_root=canonical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_cost_estimate_above_boundary(
    historical_fixture: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "_COST_PER_SECOND_USD", 1.0)

    with pytest.raises(ValueError, match="cost"):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_raw_cost_just_above_boundary(
    historical_fixture: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        smoke,
        "_COST_PER_SECOND_USD",
        (0.15 + 0.0000001) / 39.576,
    )

    with pytest.raises(ValueError, match="cost"):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


@pytest.mark.parametrize("kind", ["missing", "symlink", "hash"])
def test_prepare_protocol_rejects_missing_symlinked_or_mismatched_delivery(
    historical_fixture: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    delivery = historical_fixture["deliveries"] / "video-000.webm"
    if kind == "missing":
        delivery.unlink()
    elif kind == "symlink":
        replacement = historical_fixture["source_root"] / "replacement.webm"
        replacement.write_bytes(b"replacement")
        delivery.unlink()
        delivery.symlink_to(replacement)
    else:
        delivery.write_bytes(b"changed bytes")

    with pytest.raises(ValueError):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_forbidden_source_keys(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    path = historical_fixture["source_protocol"]
    source = _read_json(path)
    source["rows"][0]["answer"] = "private historical answer"
    _write_json(path, source)

    with pytest.raises(ValueError, match="forbidden"):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=tmp_path / "smoke",
            created_at="2026-08-16T12:00:00Z",
        )


def test_prepare_protocol_rejects_existing_root(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    root = tmp_path / "smoke"
    root.mkdir()
    with pytest.raises(FileExistsError):
        prepare_protocol(
            source_root=historical_fixture["source_root"],
            root=root,
            created_at="2026-08-16T12:00:00Z",
        )


def test_verify_inputs_rehashes_all_historical_and_new_artifacts(
    historical_fixture: dict[str, Path], tmp_path: Path
) -> None:
    root = tmp_path / "smoke"
    prepare_protocol(
        source_root=historical_fixture["source_root"],
        root=root,
        created_at="2026-08-16T12:00:00Z",
    )
    result = verify_inputs(root)
    assert result["status"] == "verified"
    assert result["rows_verified"] == 2

    for path in (
        historical_fixture["source_protocol"],
        historical_fixture["historical_upload"],
        historical_fixture["delivery_provenance"],
        historical_fixture["deliveries"] / "video-000.webm",
        root / "protocol.json",
        root / "inputs-freeze.json",
    ):
        original = path.read_bytes()
        path.write_bytes(original + b"\nmutation")
        with pytest.raises(ValueError):
            verify_inputs(root)
        path.write_bytes(original)


def _upload() -> dict[str, str]:
    return {
        "file_uri": "gs://frozen-smoke/video-000.webm",
        "mime_type": "video/webm",
    }


def _request_dump(arm: str) -> dict[str, Any]:
    request = build_inlined_request(
        row={"video_id": "video-000", "question": "What happens?"},
        upload=_upload(),
        arm=arm,
    )
    return request.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.parametrize("arm", ["control_31pro", "candidate_37flash"])
def test_build_inlined_request_serializes_native_video_and_direct_answer_prompt(
    arm: str,
) -> None:
    payload = _request_dump(arm)
    assert payload["model"] == ARMS[arm]["model"]
    part_video, part_prompt = payload["contents"][0]["parts"]
    assert part_video["fileData"] == {
        "fileUri": "gs://frozen-smoke/video-000.webm",
        "mimeType": "video/webm",
    }
    assert part_video["videoMetadata"]["fps"] == 4.0
    assert part_video["mediaResolution"]["level"] == "MEDIA_RESOLUTION_HIGH"
    assert part_prompt["text"] == (
        "Answer the question using the complete native video and audio. "
        "Identify the requested relationship and inspect all relevant moments. "
        "For temporal questions, include multiple supported subsequent events "
        "when the wording is ambiguous rather than stopping at an incidental "
        "micro-action. Give one concise best-supported answer only.\n\n"
        "Question: What happens?"
    )
    assert payload["config"]["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingLevel": "HIGH",
    }
    assert payload["config"]["seed"] == 42
    assert payload["config"]["responseMimeType"] == "text/plain"
    assert payload["config"]["maxOutputTokens"] == 2048


def test_build_inlined_request_only_control_arm_sets_temperature() -> None:
    control = _request_dump("control_31pro")["config"]
    candidate = _request_dump("candidate_37flash")["config"]
    assert control["temperature"] == 0.0
    for key in ("temperature", "topP", "topK", "candidateCount"):
        assert key not in candidate


def test_build_inlined_request_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="arm"):
        build_inlined_request(row={"question": "What?"}, upload=_upload(), arm="other")


def test_build_inlined_request_rejects_non_video_upload() -> None:
    upload = _upload()
    upload["mime_type"] = "audio/wav"
    with pytest.raises(ValueError, match="video"):
        build_inlined_request(
            row={"question": "What?"}, upload=upload, arm="candidate_37flash"
        )
