"""Contract tests for the two-row Gemini 3.7 Flash smoke protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import trec_video_2026.vqa_gemini37_flash_smoke as smoke
from trec_video_2026.vqa_gemini37_flash_smoke import (
    ARMS,
    build_inlined_request,
    freeze_predictions,
    prepare_protocol,
    preflight_models,
    record_status,
    submit_arm,
    upload_rows,
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


class _FakeModels:
    def __init__(self, *, missing: str | None = None) -> None:
        self.calls: list[str] = []
        self.missing = missing

    def get(self, *, model: str) -> Any:
        self.calls.append(model)
        if model == self.missing:
            return None
        return SimpleNamespace(
            name=f"models/{model}",
            version=f"{model}-version-20260816",
            display_name=f"Synthetic {model}",
            supported_actions=["generateContent"],
            input_token_limit=1_048_576,
            output_token_limit=65_536,
            api_key="must never be serialized",
        )


class _FakeFiles:
    def __init__(self, *, state: str = "ACTIVE") -> None:
        self.upload_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self.state = state
        self._files: dict[str, Any] = {}

    def upload(self, *, file: Path, config: Any) -> Any:
        self.upload_calls.append({"file": file, "config": config})
        row_id = file.stem
        name = f"files/synthetic-{row_id}"
        uri = f"https://generativelanguage.googleapis.com/v1beta/{name}"
        uploaded = SimpleNamespace(
            name=name,
            uri=uri,
            mime_type="video/webm",
            state="PROCESSING" if self.state == "ACTIVE_AFTER_POLL" else self.state,
        )
        self._files[name] = uploaded
        return uploaded

    def get(self, *, name: str) -> Any:
        self.get_calls.append(name)
        uploaded = self._files[name]
        if self.state == "ACTIVE_AFTER_POLL":
            uploaded.state = "ACTIVE"
        return uploaded


def _response(*, response_id: str, model_version: str, prompt: int = 20_000) -> dict[str, Any]:
    return {
        "response_id": response_id,
        "model_version": model_version,
        "finish_reason": "STOP",
        "text": f"synthetic answer for {response_id}",
        "usage_metadata": {
            "prompt_token_count": prompt,
            "candidates_token_count": 100,
            "total_token_count": prompt + 100,
        },
    }


class _FakeBatches:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.jobs: dict[str, Any] = {}
        self.next_state = "JOB_STATE_SUCCEEDED"
        self.responses: list[dict[str, Any]] | None = None

    def create(self, *, model: str, src: Any, config: Any) -> Any:
        call = {"model": model, "src": src, "config": config}
        self.create_calls.append(call)
        name = f"batches/synthetic-{len(self.create_calls)}"
        version = f"{model}-version-20260816"
        responses = self.responses or [
            _response(response_id=f"resp-{index}", model_version=version)
            for index in range(2)
        ]
        responses = [
            item
            if isinstance(item, dict) and "metadata" in item
            else {
                "metadata": src.inlined_requests[index].metadata,
                "response": item,
            }
            for index, item in enumerate(responses)
        ]
        job = SimpleNamespace(
            name=name,
            display_name=getattr(config, "display_name", None),
            model=model,
            state=self.next_state,
            responses=responses,
        )
        self.jobs[name] = job
        return job

    def get(self, *, name: str) -> Any:
        return self.jobs[name]


class _FakeClient:
    def __init__(self, *, missing_model: str | None = None, file_state: str = "ACTIVE") -> None:
        self.models = _FakeModels(missing=missing_model)
        self.files = _FakeFiles(state=file_state)
        self.batches = _FakeBatches()


def _prepared_provider_root(tmp_path: Path) -> Path:
    source = tmp_path / "historical"
    source.mkdir(parents=True)
    # Keep the provider fixtures independent from the existing protocol fixtures.
    deliveries = source / "deliveries"
    deliveries.mkdir()
    sources = source / "sources"
    sources.mkdir()
    source_rows: list[dict[str, Any]] = []
    upload_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for index in range(475):
        video_id = f"video-{index:03d}"
        duration = 18.308 if index == 0 else 21.268 if index == 1 else 1.0
        source_path = sources / f"{video_id}.mkv"
        source_path.write_bytes(f"provider source {video_id}".encode())
        source_hash = _sha256(source_path)
        row: dict[str, Any] = {
            "row_id": f"row-{index:03d}",
            "video_id": video_id,
            "source_order_index": index,
            "question": f"What happens in {video_id}?",
            "duration_seconds": duration,
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "source_size_bytes": source_path.stat().st_size,
        }
        source_rows.append(row)
        if index < 2:
            delivery = deliveries / f"{video_id}.webm"
            delivery.write_bytes(f"provider delivery {video_id}".encode())
            delivery_hash = _sha256(delivery)
            upload_rows.append(
                {
                    "row_id": row["row_id"],
                    "video_id": video_id,
                    "source_sha256": source_hash,
                    "source_size_bytes": source_path.stat().st_size,
                    "file_uri": f"https://old.example/{video_id}",
                    "delivery_path": str(delivery.relative_to(source)),
                    "delivery_sha256": delivery_hash,
                    "delivery_size_bytes": delivery.stat().st_size,
                    "delivery_mime_type": "video/webm",
                }
            )
            provenance_rows.append(
                {
                    "row_id": row["row_id"],
                    "video_id": video_id,
                    "delivery_path": str(delivery.relative_to(source)),
                    "sha256": delivery_hash,
                    "delivery_size_bytes": delivery.stat().st_size,
                    "delivery_mime_type": "video/webm",
                    "source_sha256": source_hash,
                    "source_size_bytes": source_path.stat().st_size,
                    "duration_seconds": duration,
                }
            )
    _write_json(source / "source-protocol.json", {"experiment_id": "direct-gemini-batch-475-v1", "rows": source_rows})
    _write_json(source / "historical-upload.json", {"rows": upload_rows})
    _write_json(source / "delivery-provenance.json", {"rows": provenance_rows})
    root = tmp_path / "smoke"
    prepare_protocol(source_root=source, root=root, created_at="2026-08-16T12:00:00Z")
    return root


def _ready_for_batches(tmp_path: Path) -> tuple[Path, _FakeClient]:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient(file_state="ACTIVE_AFTER_POLL")
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    upload_rows(root, client, sleep=lambda _: None)
    return root, client


def test_preflight_models_fetches_exact_models_without_credentials(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient()
    result = preflight_models(root, client, "2026-08-16T12:01:00Z")
    assert client.models.calls == [ARMS["control_31pro"]["model"], ARMS["candidate_37flash"]["model"]]
    assert [result["models"][arm]["model_id"] for arm in ARMS] == [item["model"] for item in ARMS.values()]
    serialized = (root / "models.json").read_text()
    assert "api_key" not in serialized
    assert "must never be serialized" not in serialized


def test_preflight_models_refuses_missing_model(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient(missing_model=ARMS["candidate_37flash"]["model"])
    with pytest.raises(ValueError, match="model"):
        preflight_models(root, client, "2026-08-16T12:01:00Z")
    assert not (root / "models.json").exists()


def test_upload_rows_uploads_each_delivery_once_and_polls_active(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient(file_state="ACTIVE_AFTER_POLL")
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    result = upload_rows(root, client, sleep=lambda _: None)
    assert result["active_count"] == 2
    assert len(client.files.upload_calls) == 2
    assert len(client.files.get_calls) == 2
    assert sorted((root / "uploads").glob("*.json"))
    assert all(record["file_state"] == "ACTIVE" for record in result["uploads"])


def test_upload_rows_rejects_failed_or_duplicate_upload_records(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient(file_state="FAILED")
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    with pytest.raises(ValueError, match="FAILED"):
        upload_rows(root, client, sleep=lambda _: None)
    root, client = _ready_for_batches(tmp_path / "second")
    with pytest.raises(ValueError, match="already|duplicate"):
        upload_rows(root, client, sleep=lambda _: None)


def test_submit_arm_uses_one_exact_two_request_job_and_shared_bindings(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    result = submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    assert len(client.batches.create_calls) == 1
    call = client.batches.create_calls[0]
    assert call["model"] == ARMS["control_31pro"]["model"]
    assert len(call["src"].inlined_requests) == 2
    assert result["display_name"] == "gemini37-smoke-control-31pro"
    assert (root / "jobs" / "control_31pro.json").is_file()
    for request in result["requests"]:
        assert request["protocol_sha256"] == result["protocol_sha256"]
        assert request["mime_type"] == "video/webm"


def test_submit_arm_rejects_unknown_or_duplicate_arm(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    with pytest.raises(ValueError, match="arm"):
        submit_arm(root, "unknown", client, submitted_at="2026-08-16T12:02:00Z")
    submit_arm(root, "candidate_37flash", client, submitted_at="2026-08-16T12:02:00Z")
    with pytest.raises(ValueError, match="already|duplicate"):
        submit_arm(root, "candidate_37flash", client, submitted_at="2026-08-16T12:02:01Z")


def test_record_status_is_additive_and_freeze_retains_usage_and_cost(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:04:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["status"] == "predictions_frozen"
    assert freeze["decision"] == "quality_ready"
    assert freeze["arms"]["control_31pro"]["response_count"] == 2
    assert freeze["arms"]["candidate_37flash"]["prompt_tokens"] == 40_000
    assert freeze["observed_cost_usd"] == 0.056575
    assert len(list((root / "status" / "control_31pro").glob("*.json"))) == 2
    assert "prior" not in (root / "predictions-freeze.json").read_text().lower()


@pytest.mark.parametrize("prompt", [34_192, 61_549])
def test_freeze_predictions_marks_custom_fps_out_of_band(tmp_path: Path, prompt: int) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        client.batches.responses = [
            _response(response_id="resp-0", model_version=f"{ARMS[arm]['model']}-version-20260816", prompt=prompt // 2),
            _response(response_id="resp-1", model_version=f"{ARMS[arm]['model']}-version-20260816", prompt=prompt - prompt // 2),
        ]
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["decision"] == "custom_fps_not_verified"
    assert freeze["status"] == "predictions_frozen"


def test_freeze_predictions_rejects_missing_usage_and_duplicate_response_ids(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        version = f"{ARMS[arm]['model']}-version-20260816"
        client.batches.responses = [_response(response_id="same", model_version=version), _response(response_id="same", model_version=version)]
        client.batches.responses[0].pop("usage_metadata")
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["decision"] == "not_ready"
    assert freeze["failure_reasons"]


def test_submit_requests_carry_exact_binding_metadata(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    result = submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    requests = client.batches.create_calls[0]["src"].inlined_requests
    assert len(requests) == 2
    for index, request in enumerate(requests):
        assert request.metadata == {
            "arm": "control_31pro",
            "row_id": result["requests"][index]["row_id"],
            "video_id": result["requests"][index]["video_id"],
            "question": result["requests"][index]["question"],
            "file_uri": result["requests"][index]["file_uri"],
            "delivery_sha256": result["requests"][index]["delivery_sha256"],
            "mime_type": result["requests"][index]["mime_type"],
            "protocol_sha256": result["requests"][index]["protocol_sha256"],
        }


def test_freeze_merges_top_level_response_metadata_and_error(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        result = submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        requests = client.batches.create_calls[-1]["src"].inlined_requests
        version = f"{ARMS[arm]['model']}-version-20260816"
        client.batches.jobs[result["name"]].responses = [
            {
                "metadata": request.metadata,
                "response": _response(response_id=f"resp-{index}", model_version=version),
            }
            for index, request in enumerate(requests)
        ]
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["decision"] == "quality_ready"
    assert freeze["arms"]["control_31pro"]["responses"][0]["metadata"]["arm"] == "control_31pro"


def test_freeze_rejects_top_level_response_error(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        result = submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        requests = client.batches.create_calls[-1]["src"].inlined_requests
        version = f"{ARMS[arm]['model']}-version-20260816"
        client.batches.jobs[result["name"]].responses = [
            {
                "metadata": request.metadata,
                "response": _response(response_id=f"resp-{index}", model_version=version),
                "error": {"code": 400, "message": "provider row error", "details": []},
            }
            if index == 0
            else {
                "metadata": request.metadata,
                "response": _response(response_id=f"resp-{index}", model_version=version),
            }
            for index, request in enumerate(requests)
        ]
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    assert freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")["decision"] == "not_ready"


def test_submit_reserves_paid_attempt_before_malformed_provider_result(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    client.batches.create = lambda **_: SimpleNamespace(state="JOB_STATE_SUCCEEDED")
    result = submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    assert result["status"] == "failed"
    assert (root / "jobs" / "control_31pro.attempt.json").is_file()
    with pytest.raises(ValueError, match="attempt|submission"):
        submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:01Z")


def test_submit_reserves_paid_attempt_when_provider_raises(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)

    def fail_create(**_: Any) -> Any:
        raise RuntimeError("provider failure")

    client.batches.create = fail_create
    result = submit_arm(root, "candidate_37flash", client, submitted_at="2026-08-16T12:02:00Z")
    assert result["status"] == "failed"
    assert (root / "jobs" / "candidate_37flash.attempt.json").is_file()


def test_upload_writes_terminal_attempt_and_failure_evidence(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient(file_state="FAILED")
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    with pytest.raises(ValueError, match="FAILED"):
        upload_rows(root, client, sleep=lambda _: None)
    assert (root / "uploads" / "row-000.attempt.json").is_file()
    failure = json.loads((root / "uploads" / "row-000.json").read_text())
    assert failure["status"] == "failed"
    with pytest.raises(ValueError, match="duplicate|already"):
        upload_rows(root, client, sleep=lambda _: None)


def test_historical_single_mapping_upload_uri_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "upload.json"
    _write_json(path, {"video_id": "video-000", "file_uri": "https://old.example/video-000"})
    assert smoke._historical_uris({"historical_upload": {"path": str(path)}}) == {
        "https://old.example/video-000"
    }


def test_partial_job_state_is_not_success() -> None:
    assert smoke._state("JOB_STATE_PARTIALLY_SUCCEEDED") == "failed"


def test_pending_with_missing_usage_is_not_ready_not_custom_fps(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    client.batches.next_state = "JOB_STATE_PENDING"
    client.batches.responses = [
        {"response_id": "resp-0", "model_version": "v", "text": ""},
        {"response_id": "resp-1", "model_version": "v", "text": ""},
    ]
    for arm in ARMS:
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    assert freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")["decision"] == "not_ready"


def test_usage_metadata_is_preserved_and_control_rate_is_frozen() -> None:
    assert smoke._BATCH_RATES_USD_PER_MILLION["control_31pro"]["output"] == 6.0


def test_record_status_rejects_after_prediction_freeze(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    with pytest.raises(ValueError, match="freeze|terminal"):
        record_status(root, "control_31pro", client, checked_at="2026-08-16T12:06:00Z")


def test_preflight_requires_generate_content_capability(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient()
    client.models._models = None
    original_get = client.models.get

    def get_without_generate(*, model: str) -> Any:
        result = original_get(model=model)
        result.supported_actions = ["embedContent"]
        return result

    client.models.get = get_without_generate
    with pytest.raises(ValueError, match="generateContent"):
        preflight_models(root, client, "2026-08-16T12:01:00Z")


def test_failed_batch_error_is_preserved_safely(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    client.batches.next_state = "JOB_STATE_FAILED"
    for arm in ARMS:
        result = submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        client.batches.jobs[result["name"]].error = {
            "code": 429,
            "message": "quota exhausted",
            "details": [{"reason": "rateLimit"}],
        }
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    status = json.loads(next((root / "status" / "control_31pro").glob("*.json")).read_text())
    assert status["provider_snapshot"]["error"]["code"] == 429
    assert status["provider_snapshot"]["error"]["details"][0]["reason"] == "rateLimit"
    assert freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")["decision"] == "not_ready"


def test_mixed_hard_failure_precedes_custom_fps_candidate(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        version = f"{ARMS[arm]['model']}-version-20260816"
        if arm == "control_31pro":
            client.batches.responses = [
                _response(response_id="empty-0", model_version=version),
                _response(response_id="empty-1", model_version=version),
            ]
            client.batches.responses[0]["text"] = ""
        else:
            client.batches.responses = [
                _response(response_id="fps-0", model_version=version, prompt=17_096),
                _response(response_id="fps-1", model_version=version, prompt=17_097),
            ]
        submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    assert freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")["decision"] == "not_ready"


def test_submit_rejects_symlinked_jobs_parent(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    target = tmp_path / "jobs-target"
    target.mkdir()
    (root / "jobs").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|regular"):
        submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")


def test_status_rejects_symlinked_status_parent(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    target = tmp_path / "status-target"
    target.mkdir()
    (root / "status").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|regular"):
        record_status(root, "control_31pro", client, checked_at="2026-08-16T12:03:00Z")


def test_response_privacy_keys_and_raw_http_are_rejected(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    result = submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    client.batches.jobs[result["name"]].responses = [
        {
            "metadata": request.metadata,
            "response": {
                "priorOutput": "private",
                "sdkHttpResponse": {"raw": "must not be persisted"},
            },
        }
        for request in client.batches.create_calls[-1]["src"].inlined_requests
    ]
    with pytest.raises(ValueError, match="forbidden"):
        record_status(root, "control_31pro", client, checked_at="2026-08-16T12:03:00Z")


def test_provider_raw_http_field_is_not_serialized(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    result = submit_arm(root, "control_31pro", client, submitted_at="2026-08-16T12:02:00Z")
    requests = client.batches.create_calls[-1]["src"].inlined_requests
    version = f"{ARMS['control_31pro']['model']}-version-20260816"
    client.batches.jobs[result["name"]].responses = [
        {
            "metadata": request.metadata,
            "response": _response(response_id=f"raw-{index}", model_version=version),
            "sdkHttpResponse": {"headers": {"authorization": "private"}},
        }
        for index, request in enumerate(requests)
    ]
    record_status(root, "control_31pro", client, checked_at="2026-08-16T12:03:00Z")
    serialized = next((root / "status" / "control_31pro").glob("*.json")).read_text()
    assert "sdkHttpResponse" not in serialized
    assert "authorization" not in serialized


def test_provider_upload_bytes_are_retained_for_drift_checks(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient()
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    original_upload = client.files.upload

    def upload_with_provider_bytes(*, file: Path, config: Any) -> Any:
        value = original_upload(file=file, config=config)
        value.size_bytes = file.stat().st_size
        value.sha256_hash = _sha256(file)
        return value

    client.files.upload = upload_with_provider_bytes
    upload_rows(root, client, sleep=lambda _: None)
    records = json.loads((root / "uploads" / "row-000.json").read_text())
    assert records["provider_snapshot"]["size_bytes"] > 0
    assert records["provider_snapshot"]["sha256_hash"] == records["delivery_sha256"]


@pytest.mark.parametrize("field", ["uri", "file_state", "size_bytes", "sha256_hash"])
def test_upload_record_rejects_provider_identity_or_bytes_drift(tmp_path: Path, field: str) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient()
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    upload_rows(root, client, sleep=lambda _: None)
    path = root / "uploads" / "row-000.json"
    record = json.loads(path.read_text())
    if field == "uri":
        record["provider_snapshot"][field] = "https://different.example/file"
    elif field == "file_state":
        record["provider_snapshot"][field] = "FAILED"
    elif field == "size_bytes":
        record["provider_snapshot"][field] = record["delivery_size_bytes"] + 1
    else:
        record["provider_snapshot"][field] = "f" * 64
    body = dict(record)
    body.pop("_artifact_sha256", None)
    record["_artifact_sha256"] = smoke._sha256_bytes(smoke._canonical_json(body))
    path.write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(ValueError, match="provider|drift|identity|hash|size"):
        smoke._upload_records(root, smoke._verify_artifact(root / "protocol.json"), smoke._sha256_file(root / "protocol.json"))


def test_actual_sdk_response_preserves_complete_usage_metadata(tmp_path: Path) -> None:
    root, client = _ready_for_batches(tmp_path)
    for arm in ARMS:
        result = submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        requests = client.batches.create_calls[-1]["src"].inlined_requests
        version = f"{ARMS[arm]['model']}-version-20260816"
        client.batches.jobs[result["name"]].responses = [
            smoke.types.InlinedResponse(
                metadata=request.metadata,
                response=smoke.types.GenerateContentResponse(
                    response_id=f"sdk-{index}",
                    model_version=version,
                    candidates=[
                        smoke.types.Candidate(
                            content=smoke.types.Content(
                                role="model",
                                parts=[smoke.types.Part(text="sdk-shaped answer")],
                            ),
                            finish_reason="STOP",
                        )
                    ],
                    usage_metadata=smoke.types.GenerateContentResponseUsageMetadata(
                        prompt_token_count=20_000,
                        candidates_token_count=100,
                        total_token_count=20_100,
                        thoughts_token_count=7,
                        cached_content_token_count=3,
                        traffic_type="ON_DEMAND",
                    ),
                ),
            )
            for index, request in enumerate(requests)
        ]
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["decision"] == "quality_ready"
    usage = freeze["arms"]["control_31pro"]["responses"][0]["usage_metadata"]
    assert usage["thoughtsTokenCount"] == 7
    assert usage["cachedContentTokenCount"] == 3
    assert usage["trafficType"] == "ON_DEMAND"
    assert freeze["arms"]["control_31pro"]["responses"][0]["model_version"] == "gemini-3.1-pro-preview-version-20260816"


def test_model_short_version_binds_to_full_response_version(tmp_path: Path) -> None:
    root = _prepared_provider_root(tmp_path)
    client = _FakeClient()
    original_get = client.models.get

    def short_version_model(*, model: str) -> Any:
        value = original_get(model=model)
        value.version = "001"
        return value

    client.models.get = short_version_model
    preflight_models(root, client, "2026-08-16T12:01:00Z")
    upload_rows(root, client, sleep=lambda _: None)
    for arm in ARMS:
        result = submit_arm(root, arm, client, submitted_at="2026-08-16T12:02:00Z")
        requests = client.batches.create_calls[-1]["src"].inlined_requests
        client.batches.jobs[result["name"]].responses = [
            {
                "metadata": request.metadata,
                "response": _response(
                    response_id=f"full-{index}",
                    model_version=f"{ARMS[arm]['model']}-001",
                ),
            }
            for index, request in enumerate(requests)
        ]
        record_status(root, arm, client, checked_at="2026-08-16T12:03:00Z")
    freeze = freeze_predictions(root, frozen_at="2026-08-16T12:05:00Z")
    assert freeze["decision"] == "quality_ready"
    assert freeze["arms"]["candidate_37flash"]["model_version"] == "001"
