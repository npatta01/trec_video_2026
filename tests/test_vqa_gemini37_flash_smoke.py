"""Contract tests for the two-row Gemini 3.7 Flash smoke protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_gemini37_flash_smoke import (
    ARMS,
    DIRECT_ANSWER_PROMPT,
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
    for index in range(475):
        row_id = f"row-{index:03d}"
        video_id = f"video-{index:03d}"
        duration = 20.0 if index == 0 else 19.576 if index == 1 else 1.0
        source_rows.append(
            {
                "row_id": row_id,
                "video_id": video_id,
                "source_order_index": index,
                "question": f"What happens in {video_id}?",
                "duration_seconds": duration,
            }
        )
        if index < 2:
            delivery_path = deliveries / f"{video_id}.webm"
            delivery_path.write_bytes(f"synthetic video {video_id}".encode())
            upload_rows.append(
                {
                    "row_id": row_id,
                    "video_id": video_id,
                    "file_uri": f"gs://frozen-smoke/{video_id}.webm",
                    "mime_type": "video/webm",
                }
            )
            provenance_rows.append(
                {
                    "row_id": row_id,
                    "video_id": video_id,
                    "delivery_path": str(delivery_path.relative_to(source_root)),
                    "sha256": _sha256(delivery_path),
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
            "experiment_id": "historical-full-panel-v1",
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
        "delivery_path",
        "delivery_sha256",
        "historical_upload_sha256",
        "delivery_provenance_sha256",
    }


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
    assert part_prompt["text"] == f"{DIRECT_ANSWER_PROMPT}\nQuestion: What happens?"
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
