"""Freeze and serialize the two-row Gemini 3.7 Flash smoke protocol.

This module deliberately stops at input preparation and request serialization.  It
does not call a provider, score answers, or carry any historical answer fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from google.genai import types

EXPERIMENT_ID = "gemini37-flash-two-video-smoke-v1"
DIRECT_ANSWER_PROMPT = (
    "You answer questions about video evidence. Respond with one short phrase "
    "or sentence and no explanation."
)

ARMS: dict[str, dict[str, Any]] = {
    "control_31pro": {
        "model": "gemini-3.1-pro-preview",
        "temperature": 0.0,
    },
    "candidate_37flash": {
        "model": "gemini-3.7-flash",
        "temperature": None,
    },
}

_SELECTED_SOURCE_ORDER = (0, 1)
_EXPECTED_TOTAL_DURATION = 39.576
_COST_PER_SECOND_USD = 0.0025
_ARTIFACT_HASH_KEY = "_artifact_sha256"
_FORBIDDEN_KEYS = frozenset(
    {
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
)

_SOURCE_PROTOCOL_NAMES = (
    "source-protocol.json",
    "source_protocol.json",
    "protocol.json",
)
_UPLOAD_NAMES = (
    "historical-upload.json",
    "historical_upload.json",
    "upload-record.json",
    "upload_record.json",
    "uploads.json",
    "upload.json",
)
_PROVENANCE_NAMES = (
    "delivery-provenance.json",
    "delivery_provenance.json",
    "provenance.json",
    "deliveries.json",
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot hash file: {path}") from error


def _is_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return not path.is_symlink() and stat.S_ISREG(mode)


def _read_json(path: Path) -> Any:
    if not _is_regular_file(path):
        raise ValueError(f"JSON artifact is missing or not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _reject_forbidden(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden private key {key!r} in {context}")
            _reject_forbidden(child, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_forbidden(child, context=context)


def _rows(payload: Mapping[str, Any], *, context: str) -> list[dict[str, Any]]:
    candidate: Any = payload
    if isinstance(payload, Mapping):
        for key in ("rows", "records", "uploads", "deliveries", "items"):
            if isinstance(payload.get(key), list):
                candidate = payload[key]
                break
    if not isinstance(candidate, list):
        raise ValueError(f"{context} must contain a rows list")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(candidate):
        if not isinstance(row, dict):
            raise ValueError(f"{context} row {index} must be an object")
        result.append(row)
    return result


def _find_named_file(root: Path, names: tuple[str, ...], *, context: str) -> Path:
    for name in names:
        path = root / name
        if path.exists() or path.is_symlink():
            return path
    raise ValueError(f"{context} file is missing under {root}")


def _row_id(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("row_id", row.get("id", row.get("video_id", row.get("videoId"))))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} row_id must be a non-empty string")
    return value


def _video_id(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("video_id", row.get("videoId", row.get("row_id", row.get("id"))))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} video_id must be a non-empty string")
    return value


def _duration(row: Mapping[str, Any], *, context: str) -> float:
    value = row.get(
        "duration_seconds",
        row.get("durationSeconds", row.get("duration")),
    )
    if isinstance(value, bool):
        raise ValueError(f"{context} duration_seconds must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} duration_seconds must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} duration_seconds must be positive and finite")
    return result


def _index(row: Mapping[str, Any], *, context: str) -> int:
    value = row.get(
        "source_order_index",
        row.get("sourceOrderIndex", row.get("order_index", row.get("index"))),
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} source_order_index must be an integer")
    return value


def _file_uri(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("file_uri", row.get("fileUri", row.get("uri")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} file_uri must be a non-empty string")
    return value


def _mime_type(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("mime_type", row.get("mimeType"))
    if not isinstance(value, str) or not value.startswith("video/"):
        raise ValueError(f"{context} mime_type must be a video MIME type")
    return value


def _provenance_path(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("delivery_path", row.get("path", row.get("delivery")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} delivery_path must be a non-empty string")
    return value


def _provenance_hash(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("sha256", row.get("delivery_sha256"))
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context} sha256 must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} sha256 must be a SHA-256 hex digest") from error
    return value


def _artifact_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop(_ARTIFACT_HASH_KEY, None)
    return body


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body[_ARTIFACT_HASH_KEY] = _sha256_bytes(_canonical_json(body))
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except OSError as error:
        raise ValueError(f"could not exclusively create {path}") from error
    return _sha256_bytes(serialized.encode("utf-8"))


def _verify_artifact(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    expected = value.get(_ARTIFACT_HASH_KEY)
    if not isinstance(expected, str) or expected != _sha256_bytes(
        _canonical_json(_artifact_body(value))
    ):
        raise ValueError(f"artifact hash mismatch: {path}")
    return value


def _resolve_delivery(source_root: Path, delivery_path: str, *, context: str) -> Path:
    candidate = Path(delivery_path)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    try:
        candidate.resolve().relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError(f"{context} delivery escapes source root") from error
    if not _is_regular_file(candidate):
        raise ValueError(f"{context} delivery is missing or not regular: {candidate}")
    return candidate


def _load_and_validate_inputs(
    source_root: Path,
) -> tuple[
    Path,
    Path,
    Path,
    str,
    str,
    str,
    list[dict[str, Any]],
]:
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError(f"source_root must be a regular directory: {source_root}")
    source_path = _find_named_file(source_root, _SOURCE_PROTOCOL_NAMES, context="source protocol")
    upload_path = _find_named_file(source_root, _UPLOAD_NAMES, context="historical upload")
    provenance_path = _find_named_file(
        source_root, _PROVENANCE_NAMES, context="delivery provenance"
    )
    source = _read_object(source_path)
    upload = _read_object(upload_path)
    provenance = _read_object(provenance_path)
    _reject_forbidden(source, context="source protocol")
    _reject_forbidden(upload, context="historical upload")
    _reject_forbidden(provenance, context="delivery provenance")

    source_rows = _rows(source, context="source protocol")
    if len(source_rows) != 475:
        raise ValueError("source protocol must contain exactly 475 rows")
    source_by_id: dict[str, dict[str, Any]] = {}
    source_by_video: dict[str, dict[str, Any]] = {}
    source_indices: list[int] = []
    for index, row in enumerate(source_rows):
        context = f"source protocol row {index}"
        row_id = _row_id(row, context=context)
        video_id = _video_id(row, context=context)
        source_index = _index(row, context=context)
        if row_id in source_by_id or video_id in source_by_video:
            raise ValueError("source protocol contains duplicate IDs")
        source_by_id[row_id] = row
        source_by_video[video_id] = row
        source_indices.append(source_index)
    if sorted(source_indices) != list(range(475)):
        raise ValueError("source protocol source_order_index must be 0..474")

    upload_rows = _rows(upload, context="historical upload")
    upload_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(upload_rows):
        row_id = _row_id(row, context=f"historical upload row {index}")
        if row_id in upload_by_id:
            raise ValueError("historical upload contains duplicate IDs")
        _file_uri(row, context=f"historical upload row {index}")
        _mime_type(row, context=f"historical upload row {index}")
        upload_by_id[row_id] = row

    provenance_rows = _rows(provenance, context="delivery provenance")
    provenance_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(provenance_rows):
        row_id = _row_id(row, context=f"delivery provenance row {index}")
        if row_id in provenance_by_id:
            raise ValueError("delivery provenance contains duplicate IDs")
        provenance_by_id[row_id] = row

    selected: list[dict[str, Any]] = []
    for source_index in _SELECTED_SOURCE_ORDER:
        source_row = next(row for row in source_rows if _index(row, context="source row") == source_index)
        context = f"selected source row {source_index}"
        row_id = _row_id(source_row, context=context)
        video_id = _video_id(source_row, context=context)
        duration = _duration(source_row, context=context)
        upload_row = upload_by_id.get(row_id)
        provenance_row = provenance_by_id.get(row_id)
        if upload_row is None or provenance_row is None:
            raise ValueError(f"missing historical upload or provenance for {row_id}")
        if _video_id(upload_row, context=f"upload {row_id}") != video_id:
            raise ValueError(f"upload video ID mismatch for {row_id}")
        if _video_id(provenance_row, context=f"provenance {row_id}") != video_id:
            raise ValueError(f"provenance video ID mismatch for {row_id}")
        provenance_duration = _duration(provenance_row, context=f"provenance {row_id}")
        if not math.isclose(duration, provenance_duration, abs_tol=1e-6):
            raise ValueError(f"duration drift for {row_id}")
        delivery = _resolve_delivery(
            source_root,
            _provenance_path(provenance_row, context=f"provenance {row_id}"),
            context=context,
        )
        actual_hash = _sha256_file(delivery)
        if actual_hash != _provenance_hash(provenance_row, context=f"provenance {row_id}"):
            raise ValueError(f"delivery hash mismatch for {row_id}")
        selected.append(
            {
                "row_id": row_id,
                "video_id": video_id,
                "source_order_index": source_index,
                "question": source_row.get("question"),
                "duration_seconds": duration,
                "delivery_path": str(delivery.relative_to(source_root)),
                "delivery_sha256": actual_hash,
                "historical_upload_sha256": _sha256_file(upload_path),
                "delivery_provenance_sha256": _sha256_file(provenance_path),
            }
        )
    total_duration = round(sum(row["duration_seconds"] for row in selected), 3)
    if total_duration != _EXPECTED_TOTAL_DURATION:
        raise ValueError(
            f"selected duration must be {_EXPECTED_TOTAL_DURATION}, got {total_duration}"
        )
    if any(not isinstance(row["question"], str) or not row["question"].strip() for row in selected):
        raise ValueError("selected source rows require a non-empty question")
    return (
        source_path,
        upload_path,
        provenance_path,
        _sha256_file(source_path),
        _sha256_file(upload_path),
        _sha256_file(provenance_path),
        selected,
    )


def prepare_protocol(*, source_root: Path, root: Path, created_at: str) -> dict[str, Any]:
    """Validate historical inputs and exclusively freeze the two-row protocol."""

    source_root = Path(source_root)
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"protocol root already exists: {root}")
    (
        source_path,
        upload_path,
        provenance_path,
        source_hash,
        upload_hash,
        provenance_hash,
        rows,
    ) = _load_and_validate_inputs(source_root)
    root.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "created_at": created_at,
        "source_protocol": {"path": str(source_path), "sha256": source_hash},
        "historical_upload": {"path": str(upload_path), "sha256": upload_hash},
        "delivery_provenance": {
            "path": str(provenance_path),
            "sha256": provenance_hash,
        },
        "arms": json.loads(json.dumps(ARMS)),
        "rows": rows,
        "total_duration_seconds": _EXPECTED_TOTAL_DURATION,
        "estimated_cost_ceiling_usd": round(
            _EXPECTED_TOTAL_DURATION * _COST_PER_SECOND_USD, 4
        ),
    }
    protocol_path = root / "protocol.json"
    protocol_file_hash = _write_json_exclusive(protocol_path, protocol)
    freeze = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "created_at": created_at,
        "source_protocol": {"path": str(source_path), "sha256": source_hash},
        "historical_upload": {"path": str(upload_path), "sha256": upload_hash},
        "delivery_provenance": {
            "path": str(provenance_path),
            "sha256": provenance_hash,
        },
        "protocol_sha256": protocol_file_hash,
        "rows": [
            {
                "row_id": row["row_id"],
                "delivery_path": row["delivery_path"],
                "delivery_sha256": row["delivery_sha256"],
            }
            for row in rows
        ],
    }
    _write_json_exclusive(root / "inputs-freeze.json", freeze)
    return _read_object(protocol_path)


def verify_inputs(root: Path) -> dict[str, Any]:
    """Re-hash every historical and newly frozen input and fail closed on drift."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"protocol root is missing or not a directory: {root}")
    protocol_path = root / "protocol.json"
    freeze_path = root / "inputs-freeze.json"
    protocol = _verify_artifact(protocol_path)
    freeze = _verify_artifact(freeze_path)
    if protocol.get("experiment_id") != EXPERIMENT_ID or freeze.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment ID")
    if not isinstance(protocol.get("source_protocol"), Mapping):
        raise ValueError("protocol source_protocol is missing")
    source_meta = protocol["source_protocol"]
    upload_meta = protocol.get("historical_upload")
    provenance_meta = protocol.get("delivery_provenance")
    if not isinstance(upload_meta, Mapping) or not isinstance(provenance_meta, Mapping):
        raise ValueError("protocol historical input metadata is missing")
    source_path = Path(str(source_meta.get("path", "")))
    upload_path = Path(str(upload_meta.get("path", "")))
    provenance_path = Path(str(provenance_meta.get("path", "")))
    (
        expected_source_path,
        expected_upload_path,
        expected_provenance_path,
        source_hash,
        upload_hash,
        provenance_hash,
        rows,
    ) = _load_and_validate_inputs(source_path.parent)
    if (source_path, upload_path, provenance_path) != (
        expected_source_path,
        expected_upload_path,
        expected_provenance_path,
    ):
        raise ValueError("historical input paths changed")
    if source_hash != source_meta.get("sha256") or upload_hash != upload_meta.get("sha256"):
        raise ValueError("historical input hash mismatch")
    if provenance_hash != provenance_meta.get("sha256"):
        raise ValueError("delivery provenance hash mismatch")
    if protocol.get("rows") != rows:
        raise ValueError("protocol rows changed")
    expected_freeze_rows = [
        {
            "row_id": row["row_id"],
            "delivery_path": row["delivery_path"],
            "delivery_sha256": row["delivery_sha256"],
        }
        for row in rows
    ]
    if freeze.get("rows") != expected_freeze_rows:
        raise ValueError("inputs-freeze rows changed")
    for key, expected in (
        ("source_protocol", {"path": str(source_path), "sha256": source_hash}),
        ("historical_upload", {"path": str(upload_path), "sha256": upload_hash}),
        (
            "delivery_provenance",
            {"path": str(provenance_path), "sha256": provenance_hash},
        ),
    ):
        if freeze.get(key) != expected:
            raise ValueError(f"inputs-freeze {key} changed")
    if freeze.get("protocol_sha256") != _sha256_file(protocol_path):
        raise ValueError("inputs-freeze protocol hash mismatch")
    return {
        "status": "verified",
        "rows_verified": len(rows),
        "protocol_sha256": _sha256_file(protocol_path),
        "inputs_freeze_sha256": _sha256_file(freeze_path),
    }


def build_inlined_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any], arm: str
) -> types.InlinedRequest:
    """Build one native-AV request with the arm's exact generation config."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    question = row.get("question", row.get("question_text", row.get("questionText")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question must be a non-empty string")
    file_uri = _file_uri(upload, context="upload")
    mime_type = _mime_type(upload, context="upload")
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=4.0),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )
    prompt = types.Part(text=f"{DIRECT_ANSWER_PROMPT}\nQuestion: {question.strip()}")
    config_values: dict[str, Any] = {
        "max_output_tokens": 2048,
        "response_mime_type": "text/plain",
        "seed": 42,
        "thinking_config": types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    }
    if arm == "control_31pro":
        config_values["temperature"] = 0.0
    return types.InlinedRequest(
        model=ARMS[arm]["model"],
        contents=[types.Content(role="user", parts=[video, prompt])],
        config=types.GenerateContentConfig(**config_values),
    )
