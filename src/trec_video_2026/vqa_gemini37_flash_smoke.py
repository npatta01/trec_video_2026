"""Freeze and serialize the two-row Gemini 3.7 Flash smoke protocol.

Provider transitions remain explicit and one-attempt.  Post-freeze scoring is a
small dependency-free diagnostic and is only allowed after every immutable
prediction binding has been re-verified.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from google.genai import types

EXPERIMENT_ID = "gemini37-flash-two-video-smoke-v2"
DIRECT_ANSWER_PROMPT = (
    "Answer the question using the complete native video and audio. Identify the "
    "requested relationship and inspect all relevant moments. For temporal "
    "questions, include multiple supported subsequent events when the wording "
    "is ambiguous rather than stopping at an incidental micro-action. Give one "
    "concise best-supported answer only."
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
_EXPECTED_DURATIONS = (18.308, 21.268)
_EXPECTED_TOTAL_DURATION = 39.576
_MAX_ESTIMATED_COST_USD = 0.15
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
_SAFE_CANONICAL_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_CANONICAL_ROW_ID = re.compile(r"^[A-Za-z0-9_-]+$")


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


def _hash_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if not _is_regular_file(path):
            raise ValueError(f"artifact is missing or not a regular file: {path}")
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            forbidden = {
                re.sub(r"[^a-z0-9]", "", forbidden_key.casefold())
                for forbidden_key in _FORBIDDEN_KEYS
            }
            if normalized_key in forbidden:
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
    value = value.strip()
    if not _SAFE_CANONICAL_ROW_ID.fullmatch(value):
        raise ValueError(f"{context} row_id is unsafe for canonical paths")
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
    value = row.get(
        "mime_type",
        row.get("mimeType", row.get("delivery_mime_type", row.get("deliveryMimeType"))),
    )
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


def _source_hash(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("source_sha256", row.get("sourceSha256"))
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{context} source_sha256 must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{context} source_sha256 must be a SHA-256 hex digest") from error
    return value


def _size_bytes(row: Mapping[str, Any], *, context: str, delivery: bool) -> int:
    names = (
        ("delivery_size_bytes", "deliverySizeBytes", "size_bytes", "sizeBytes")
        if delivery
        else ("source_size_bytes", "sourceSizeBytes")
    )
    value = next((row.get(name) for name in names if name in row), None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        label = "delivery" if delivery else "source"
        raise ValueError(f"{context} {label}_size_bytes must be a non-negative integer")
    return value


def _source_path(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("source_path", row.get("sourcePath"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} source_path must be a non-empty string")
    return value


def _validate_canonical_video_id(video_id: str, *, context: str) -> None:
    if not _SAFE_CANONICAL_VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"{context} video ID is unsafe for canonical paths")


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


def _verified_artifact_file_hash(path: Path, *, label: str) -> str:
    _verify_artifact(path)
    try:
        return _sha256_file(path)
    except ValueError as error:
        raise ValueError(f"{label} is missing or not a regular file") from error


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


def _resolve_source(source_root: Path, source_path: str, *, context: str) -> Path:
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    if not _is_regular_file(candidate):
        raise ValueError(f"{context} source is missing or not regular: {candidate}")
    return candidate


def _optional_size(row: Mapping[str, Any], *, context: str, delivery: bool) -> int | None:
    names = (
        ("delivery_size_bytes", "deliverySizeBytes", "size_bytes", "sizeBytes")
        if delivery
        else ("source_size_bytes", "sourceSizeBytes")
    )
    if not any(name in row for name in names):
        return None
    return _size_bytes(row, context=context, delivery=delivery)


def _validate_selected_duration(selected: Sequence[Mapping[str, Any]]) -> None:
    for index, (row, expected) in enumerate(zip(selected, _EXPECTED_DURATIONS, strict=True)):
        if not math.isclose(float(row["duration_seconds"]), expected, abs_tol=1e-6):
            raise ValueError(f"duration drift at selected row {index}")
    total_duration = round(sum(float(row["duration_seconds"]) for row in selected), 3)
    if total_duration != _EXPECTED_TOTAL_DURATION:
        raise ValueError(
            f"selected duration must be {_EXPECTED_TOTAL_DURATION}, got {total_duration}"
        )


def _validate_source_identity(
    source_root: Path, source_row: Mapping[str, Any], *, context: str
) -> tuple[str, int]:
    source_path = _resolve_source(source_root, _source_path(source_row, context=context), context=context)
    source_hash = _source_hash(source_row, context=context)
    source_size = _optional_size(source_row, context=context, delivery=False)
    actual_hash = _sha256_file(source_path)
    actual_size = source_path.stat().st_size
    if actual_hash != source_hash:
        raise ValueError(f"{context} source hash mismatch")
    if source_size is not None and actual_size != source_size:
        raise ValueError(f"{context} source size mismatch")
    return source_hash, actual_size


def _selected_row(
    *,
    source_root: Path,
    source_row: Mapping[str, Any],
    source_index: int,
    upload_row: Mapping[str, Any],
    provenance_row: Mapping[str, Any],
    upload_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    context = f"selected source row {source_index}"
    row_id = _row_id(source_row, context=context)
    video_id = _video_id(source_row, context=context)
    if _video_id(upload_row, context=f"upload {row_id}") != video_id:
        raise ValueError(f"upload video ID mismatch for {row_id}")
    if _video_id(provenance_row, context=f"provenance {row_id}") != video_id:
        raise ValueError(f"provenance video ID mismatch for {row_id}")
    source_hash, source_size = _validate_source_identity(
        source_root, source_row, context=context
    )
    upload_source_hash = _source_hash(upload_row, context=f"upload {row_id}")
    if upload_source_hash != source_hash:
        raise ValueError(f"upload source hash mismatch for {row_id}")
    upload_source_size = _optional_size(
        upload_row, context=f"upload {row_id}", delivery=False
    )
    if upload_source_size is not None and upload_source_size != source_size:
        raise ValueError(f"upload source size mismatch for {row_id}")
    provenance_source_hash = _source_hash(
        provenance_row, context=f"provenance {row_id}"
    )
    if provenance_source_hash != source_hash:
        raise ValueError(f"provenance source hash mismatch for {row_id}")
    provenance_source_size = _size_bytes(
        provenance_row, context=f"provenance {row_id}", delivery=False
    )
    if provenance_source_size != source_size:
        raise ValueError(f"provenance source size mismatch for {row_id}")
    delivery_value = upload_row.get(
        "delivery_path", upload_row.get("path", upload_row.get("delivery"))
    )
    if not isinstance(delivery_value, str) or not delivery_value.strip():
        delivery_value = _provenance_path(
            provenance_row, context=f"provenance {row_id}"
        )
    delivery = _resolve_delivery(source_root, delivery_value, context=context)
    actual_hash = _sha256_file(delivery)
    actual_size = delivery.stat().st_size
    upload_hash = _provenance_hash(upload_row, context=f"upload {row_id}")
    provenance_hash = _provenance_hash(provenance_row, context=f"provenance {row_id}")
    if actual_hash != upload_hash or actual_hash != provenance_hash:
        raise ValueError(f"delivery hash mismatch for {row_id}")
    upload_size = _optional_size(upload_row, context=f"upload {row_id}", delivery=True)
    provenance_size = _size_bytes(
        provenance_row, context=f"provenance {row_id}", delivery=True
    )
    if upload_size is not None and upload_size != actual_size:
        raise ValueError(f"upload delivery size mismatch for {row_id}")
    if provenance_size != actual_size:
        raise ValueError(f"provenance delivery size mismatch for {row_id}")
    mime_type = _mime_type(upload_row, context=f"upload {row_id}")
    provenance_mime = _mime_type(provenance_row, context=f"provenance {row_id}")
    if mime_type != provenance_mime:
        raise ValueError(f"delivery MIME mismatch for {row_id}")
    duration = _duration(source_row, context=context)
    provenance_duration = provenance_row.get(
        "delivery_duration_seconds", provenance_row.get("duration_seconds")
    )
    if provenance_duration is not None and not math.isclose(
        duration, float(provenance_duration), abs_tol=1e-6
    ):
        raise ValueError(f"duration drift for {row_id}")
    question = source_row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"question missing for {row_id}")
    return {
        "row_id": row_id,
        "video_id": video_id,
        "source_order_index": source_index,
        "question": question,
        "duration_seconds": duration,
        "source_sha256": source_hash,
        "source_size_bytes": source_size,
        "delivery_path": str(delivery.relative_to(source_root)),
        "delivery_sha256": actual_hash,
        "delivery_size_bytes": actual_size,
        "delivery_mime_type": mime_type,
        "historical_upload_sha256": _sha256_file(upload_path),
        "delivery_provenance_sha256": _sha256_file(provenance_path),
    }


def _load_canonical_inputs(
    source_root: Path, source_path: Path, source: Mapping[str, Any]
) -> tuple[Path, Path, Path, str, str, str, list[dict[str, Any]]]:
    if source_path.name != "protocol.json":
        raise ValueError("canonical source protocol must be protocol.json")
    if source.get("experiment_id") != "direct-gemini-batch-475-v1":
        raise ValueError("canonical source protocol experiment identity mismatch")
    source_rows = _rows(source, context="source protocol")
    if len(source_rows) != 475:
        raise ValueError("source protocol must contain exactly 475 rows")
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    upload_paths: list[Path] = []
    provenance_paths: list[Path] = []
    for source_index, source_row in enumerate(source_rows):
        video_id = _video_id(source_row, context=f"source protocol row {source_index}")
        _validate_canonical_video_id(
            video_id, context=f"source protocol row {source_index}"
        )
        if video_id in seen:
            raise ValueError("source protocol contains duplicate IDs")
        seen.add(video_id)
        if source_index >= 2:
            continue
        upload_path = source_root / "uploads" / f"{video_id}.json"
        upload = _read_object(upload_path)
        _reject_forbidden(upload, context=f"historical upload {video_id}")
        delivery = _resolve_delivery(
            source_root,
            _provenance_path(upload, context=f"upload {video_id}"),
            context=f"upload {video_id}",
        )
        provenance_path = delivery.parent / f"gemini-{video_id}.json"
        provenance = _read_object(provenance_path)
        _reject_forbidden(provenance, context=f"delivery provenance {video_id}")
        selected.append(
            _selected_row(
                source_root=source_root,
                source_row=source_row,
                source_index=source_index,
                upload_row=upload,
                provenance_row=provenance,
                upload_path=upload_path,
                provenance_path=provenance_path,
            )
        )
        upload_paths.append(upload_path)
        provenance_paths.append(provenance_path)
    _validate_selected_duration(selected)
    return (
        source_path,
        source_root / "uploads",
        source_root / "deliveries",
        _sha256_file(source_path),
        _hash_paths(upload_paths),
        _hash_paths(provenance_paths),
        selected,
    )


def _load_and_validate_inputs(
    source_root: Path,
) -> tuple[Path, Path, Path, str, str, str, list[dict[str, Any]]]:
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"source_root must be a regular directory: {source_root}")
    try:
        source_root = source_root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"source_root cannot be resolved: {source_root}") from error
    source_path = _find_named_file(source_root, _SOURCE_PROTOCOL_NAMES, context="source protocol")
    source = _read_object(source_path)
    _reject_forbidden(source, context="source protocol")
    uploads_dir = source_root / "uploads"
    deliveries_dir = source_root / "deliveries"
    if uploads_dir.is_symlink() or deliveries_dir.is_symlink():
        raise ValueError("canonical uploads/deliveries directories must not be symlinks")
    if uploads_dir.is_dir() and deliveries_dir.is_dir():
        return _load_canonical_inputs(source_root, source_path, source)
    if source.get("experiment_id") != "direct-gemini-batch-475-v1":
        raise ValueError("source protocol experiment identity mismatch")
    upload_path = _find_named_file(source_root, _UPLOAD_NAMES, context="historical upload")
    provenance_path = _find_named_file(
        source_root, _PROVENANCE_NAMES, context="delivery provenance"
    )
    upload = _read_object(upload_path)
    provenance = _read_object(provenance_path)
    _reject_forbidden(upload, context="historical upload")
    _reject_forbidden(provenance, context="delivery provenance")
    source_rows = _rows(source, context="source protocol")
    if len(source_rows) != 475:
        raise ValueError("source protocol must contain exactly 475 rows")
    source_by_id: dict[str, dict[str, Any]] = {}
    source_indices: list[int] = []
    for index, row in enumerate(source_rows):
        context = f"source protocol row {index}"
        row_id = _row_id(row, context=context)
        video_id = _video_id(row, context=context)
        source_index = _index(row, context=context)
        if row_id in source_by_id or video_id in { _video_id(item, context="source row") for item in source_by_id.values() }:
            raise ValueError("source protocol contains duplicate IDs")
        source_by_id[row_id] = row
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
        source_row = next(
            row for row in source_rows if _index(row, context="source row") == source_index
        )
        row_id = _row_id(source_row, context=f"source row {source_index}")
        upload_row = upload_by_id.get(row_id)
        provenance_row = provenance_by_id.get(row_id)
        if upload_row is None or provenance_row is None:
            raise ValueError(f"missing historical upload or provenance for {row_id}")
        selected.append(
            _selected_row(
                source_root=source_root,
                source_row=source_row,
                source_index=source_index,
                upload_row=upload_row,
                provenance_row=provenance_row,
                upload_path=upload_path,
                provenance_path=provenance_path,
            )
        )
    _validate_selected_duration(selected)
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
    estimated_cost_raw = _EXPECTED_TOTAL_DURATION * _COST_PER_SECOND_USD
    if estimated_cost_raw > _MAX_ESTIMATED_COST_USD:
        raise ValueError(
            f"estimated cost ${estimated_cost_raw:.8f} exceeds ${_MAX_ESTIMATED_COST_USD:.2f}"
        )
    estimated_cost = round(estimated_cost_raw, 4)
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
        "estimated_cost_ceiling_usd": estimated_cost,
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
                "delivery_size_bytes": row["delivery_size_bytes"],
                "delivery_mime_type": row["delivery_mime_type"],
                "source_sha256": row["source_sha256"],
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
            "delivery_size_bytes": row["delivery_size_bytes"],
            "delivery_mime_type": row["delivery_mime_type"],
            "source_sha256": row["source_sha256"],
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
    prompt = types.Part(text=f"{DIRECT_ANSWER_PROMPT}\n\nQuestion: {question.strip()}")
    metadata = {
        "arm": arm,
        "row_id": str(row.get("row_id", row.get("video_id", ""))),
        "video_id": str(row.get("video_id", "")),
        "question": question.strip(),
        "file_uri": file_uri,
        "delivery_sha256": str(row.get("delivery_sha256", "")),
        "mime_type": mime_type,
        "protocol_sha256": str(row.get("protocol_sha256", "")),
    }
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
        metadata=metadata,
        config=types.GenerateContentConfig(**config_values),
    )


# Task 2 provider lifecycle constants.  These are deliberately kept here rather
# than in the CLI so the provider-free contract can be exercised with SDK-shaped
# fakes and the transition functions remain the only writers of paid evidence.
_MODELS_FILENAME = "models.json"
_UPLOADS_DIRNAME = "uploads"
_JOBS_DIRNAME = "jobs"
_STATUS_DIRNAME = "status"
_PREDICTIONS_FILENAME = "predictions-freeze.json"
_FPS_PROMPT_TOKEN_MIN = 34_193
_FPS_PROMPT_TOKEN_MAX = 61_548
_MAX_UPLOAD_POLLS = 30
_UPLOAD_POLL_SECONDS = 1.0
_BATCH_RATES_USD_PER_MILLION: dict[str, dict[str, float]] = {
    # Frozen Batch rates for the authorized comparison window.  The 3.7 rate is
    # the documented introductory Batch/Flex price; the 3.1 rate is pinned for
    # this control and is never read from provider/client state.
    "control_31pro": {"prompt": 1.0, "output": 6.0},
    "candidate_37flash": {"prompt": 0.375, "output": 1.875},
}


def _arm_observed_cost_usd(arm: str, prompt_tokens: int, output_tokens: int) -> float:
    rates = _BATCH_RATES_USD_PER_MILLION[arm]
    return prompt_tokens / 1_000_000 * rates["prompt"] + output_tokens / 1_000_000 * rates["output"]


def _attr(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        try:
            result = getattr(value, name)
        except AttributeError:
            continue
        if result is not None:
            return result
    return default


def _scalar(value: Any) -> Any:
    enum_value = _attr(value, "value", default=None)
    if enum_value is not None and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _plain(value: Any, *, depth: int = 0) -> Any:
    """Convert SDK-shaped values without serializing client/config state."""

    if depth > 8:
        raise ValueError("provider object is too deeply nested")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        blocked = {
            "apikey",
            "api_key",
            "token",
            "accesstoken",
            "access_token",
            "credential",
            "credentials",
            "secret",
            "authorization",
            "client",
            "transport",
            "environment",
            "client_config",
            "_client",
        }
        return {
            str(key): _plain(child, depth=depth + 1)
            for key, child in value.items()
            if str(key).casefold().replace("-", "_") not in blocked
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(child, depth=depth + 1) for child in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            dumped = model_dump()
        return _plain(dumped, depth=depth + 1)
    # Small SDK-shaped fakes often expose fields directly instead of providing
    # Pydantic's model_dump.  Read only response-shaped fields; never walk vars()
    # on an arbitrary provider object.
    known_fields = (
        "response_id",
        "responseId",
        "id",
        "model_version",
        "modelVersion",
        "finish_reason",
        "finishReason",
        "text",
        "candidates",
        "content",
        "parts",
        "role",
        "prompt_token_count",
        "promptTokenCount",
        "total_token_count",
        "totalTokenCount",
        "candidates_token_count",
        "candidatesTokenCount",
        "thoughts_token_count",
        "thoughtsTokenCount",
        "usage_metadata",
        "usageMetadata",
        "error",
        "key",
        "response",
    )
    attributes: dict[str, Any] = {}
    for name in known_fields:
        try:
            child = getattr(value, name)
        except AttributeError:
            continue
        attributes[name] = _plain(child, depth=depth + 1)
    if attributes:
        return attributes
    # Do not fall back to vars(value): arbitrary SDK objects may carry a client,
    # transport, credential, or environment object.  Callers use allowlists for
    # all provider snapshots instead.
    return _scalar(value)


def _nonempty_string(value: Any, *, label: str) -> str:
    value = _scalar(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _model_snapshot(model: Any, *, arm: str) -> dict[str, Any]:
    requested = ARMS[arm]["model"]
    if model is None:
        raise ValueError(f"preflight model is missing for {arm}")
    name = _nonempty_string(_attr(model, "name", "model", "model_id"), label=f"{arm} model name")
    model_id = name.removeprefix("models/")
    if model_id != requested:
        raise ValueError(f"preflight model identity mismatch for {arm}")
    version = _nonempty_string(
        _attr(model, "version", "model_version", "modelVersion"),
        label=f"{arm} model version",
    )
    supported = _attr(model, "supported_actions", "supportedActions", "capabilities", default=[])
    if isinstance(supported, Mapping):
        capabilities: Any = _plain(supported)
    elif isinstance(supported, Sequence) and not isinstance(supported, (str, bytes, bytearray)):
        capabilities = [_nonempty_string(item, label=f"{arm} capability") for item in supported]
    else:
        capabilities = [_nonempty_string(supported, label=f"{arm} capability")]
    capability_names = (
        {
            str(item).casefold().replace("_", "").replace("-", "")
            for item in capabilities
            if isinstance(item, str)
        }
        if isinstance(capabilities, Sequence) and not isinstance(capabilities, (str, bytes, bytearray))
        else {
            str(item).casefold().replace("_", "").replace("-", "")
            for item in capabilities
            if isinstance(item, str)
        }
        if isinstance(capabilities, Mapping)
        else set()
    )
    if "generatecontent" not in capability_names:
        raise ValueError(f"{arm} model lacks generateContent capability")
    snapshot = {
        "arm": arm,
        "requested_model": requested,
        "model_id": model_id,
        "name": name,
        "version": version,
        "capabilities": capabilities,
    }
    for source_name, output_name in (
        ("display_name", "display_name"),
        ("input_token_limit", "input_token_limit"),
        ("output_token_limit", "output_token_limit"),
    ):
        value = _attr(model, source_name, output_name, default=None)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            snapshot[output_name] = value
    return snapshot


def _protocol_context(root: Path) -> tuple[Path, dict[str, Any], str]:
    root = Path(root)
    verification = verify_inputs(root)
    protocol = _verify_artifact(root / "protocol.json")
    if protocol.get("status") != "inputs_frozen":
        raise ValueError("protocol inputs are not frozen")
    return root, protocol, str(verification["protocol_sha256"])


def _models_artifact(root: Path, protocol_hash: str) -> dict[str, Any]:
    models = _verify_artifact(root / _MODELS_FILENAME)
    if models.get("protocol_sha256") != protocol_hash:
        raise ValueError("model preflight is bound to a different protocol")
    records = models.get("models")
    if not isinstance(records, Mapping) or set(records) != set(ARMS):
        raise ValueError("model preflight is incomplete")
    for arm, expected in ARMS.items():
        record = records.get(arm)
        if not isinstance(record, Mapping):
            raise ValueError(f"model preflight record is missing for {arm}")
        if record.get("model_id") != expected["model"]:
            raise ValueError(f"model preflight identity mismatch for {arm}")
        capabilities = record.get("capabilities")
        if isinstance(capabilities, Mapping):
            capability_names = {
                str(item).casefold().replace("_", "").replace("-", "")
                for item in capabilities
                if isinstance(item, str)
            }
        elif isinstance(capabilities, Sequence) and not isinstance(capabilities, (str, bytes, bytearray)):
            capability_names = {
                str(item).casefold().replace("_", "").replace("-", "")
                for item in capabilities
                if isinstance(item, str)
            }
        else:
            capability_names = set()
        if "generatecontent" not in capability_names:
            raise ValueError(f"{arm} model lacks generateContent capability")
        snapshot_hash = record.get("provider_snapshot_sha256")
        snapshot_body = dict(record)
        snapshot_body.pop("provider_snapshot_sha256", None)
        if snapshot_hash != _sha256_bytes(_canonical_json(snapshot_body)):
            raise ValueError(f"model preflight snapshot hash mismatch for {arm}")
    return models


def _historical_uris(protocol: Mapping[str, Any]) -> set[str]:
    metadata = protocol.get("historical_upload")
    if not isinstance(metadata, Mapping):
        return set()
    path = Path(str(metadata.get("path", "")))
    candidates: list[Path] = []
    if _is_regular_file(path):
        candidates = [path]
    elif path.is_dir() and not path.is_symlink():
        candidates = sorted(path.glob("*.json"))
    uris: set[str] = set()
    for candidate in candidates:
        try:
            payload = _read_json(candidate)
        except ValueError:
            continue
        if isinstance(payload, Mapping):
            if any(isinstance(payload.get(key), list) for key in ("rows", "records", "uploads", "deliveries", "items")):
                rows = _rows(payload, context=f"historical upload {candidate}")
            elif any(key in payload for key in ("file_uri", "fileUri", "uri")):
                rows = [dict(payload)]
            else:
                rows = []
        else:
            rows = []
        for row in rows:
            uri = _attr(row, "file_uri", "fileUri", "uri", default=None)
            if isinstance(uri, str) and uri.strip():
                uris.add(uri.strip())
    return uris


def _protocol_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = protocol.get("rows")
    if not isinstance(rows, list) or len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("protocol must contain exactly two rows")
    return rows


def _upload_file_path(protocol: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    source_meta = protocol.get("source_protocol")
    if not isinstance(source_meta, Mapping):
        raise ValueError("protocol source path is missing")
    source_root = Path(str(source_meta.get("path", ""))).parent
    delivery = _resolve_delivery(source_root, _provenance_path(row, context="protocol row"), context="protocol row")
    if _mime_type(row, context="protocol row") != "video/webm":
        raise ValueError("only video/webm deliveries are accepted")
    expected_hash = _provenance_hash(row, context="protocol row")
    expected_size = _size_bytes(row, context="protocol row", delivery=True)
    if _sha256_file(delivery) != expected_hash or delivery.stat().st_size != expected_size:
        raise ValueError(f"delivery mutation detected for {row.get('row_id')}")
    return delivery


def _upload_records(root: Path, protocol: Mapping[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = _protocol_rows(protocol)
    upload_dir = root / _UPLOADS_DIRNAME
    if not upload_dir.is_dir() or upload_dir.is_symlink():
        raise ValueError("upload records are missing")
    for row in rows:
        row_id = _row_id(row, context="protocol row")
        path = upload_dir / f"{row_id}.json"
        record = _verify_artifact(path)
        attempt_path = upload_dir / f"{row_id}.attempt.json"
        attempt = _verify_artifact(attempt_path)
        attempt_sha256 = _verified_artifact_file_hash(attempt_path, label=f"upload attempt {row_id}")
        if record.get("attempt_sha256") != attempt_sha256:
            raise ValueError(f"upload record attempt binding mismatch for {row_id}")
        if (
            attempt.get("protocol_sha256") != protocol_hash
            or attempt.get("row_id") != row_id
            or attempt.get("video_id") != row["video_id"]
            or attempt.get("mime_type") != row["delivery_mime_type"]
        ):
            raise ValueError(f"upload attempt binding mismatch for {row_id}")
        if record.get("protocol_sha256") != protocol_hash:
            raise ValueError("upload record protocol binding mismatch")
        for key in ("row_id", "video_id", "delivery_sha256", "delivery_size_bytes", "mime_type"):
            expected = {
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "delivery_sha256": row["delivery_sha256"],
                "delivery_size_bytes": row["delivery_size_bytes"],
                "mime_type": row["delivery_mime_type"],
            }[key]
            if record.get(key) != expected:
                raise ValueError(f"upload record binding mismatch for {row_id}")
        delivery = _upload_file_path(protocol, row)
        if delivery.stat().st_size != record.get("delivery_size_bytes"):
            raise ValueError(f"delivery mutation detected for {row_id}")
        provider = record.get("provider_snapshot")
        if not isinstance(provider, Mapping):
            raise ValueError(f"upload provider snapshot is missing for {row_id}")
        if (
            provider.get("file_state") != "ACTIVE"
            or provider.get("name") != record.get("file_name")
            or provider.get("uri") != record.get("file_uri")
            or provider.get("mime_type") != record.get("mime_type")
        ):
            raise ValueError(f"upload provider identity/state drift for {row_id}")
        if provider.get("size_bytes") is not None and provider.get("size_bytes") != delivery.stat().st_size:
            raise ValueError(f"upload provider size drift for {row_id}")
        if provider.get("sha256_hash") is not None and provider.get("sha256_hash") != record.get("delivery_sha256"):
            raise ValueError(f"upload provider hash drift for {row_id}")
        records.append(record)
    return records


def _state(value: Any) -> str:
    value = str(_scalar(value) or "").upper().strip()
    if value == "ACTIVE":
        return "ACTIVE"
    if value in {"SUCCEEDED", "COMPLETED", "JOB_STATE_SUCCEEDED", "JOB_STATE_COMPLETED"}:
        return "succeeded"
    if value in {
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "TIMEOUT",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_CANCELED",
        "JOB_STATE_EXPIRED",
    } or "PARTIALLY_SUCCEEDED" in value:
        return "failed"
    return "pending"


def _ensure_directory(path: Path, *, label: str, create: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{label} must be a regular directory")
        return
    if create:
        path.mkdir()
        return
    raise ValueError(f"{label} is missing")


def _safe_error(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None

    def sanitize_text(text: str) -> str:
        result = text[:512]
        for pattern in (
            r"AIza[0-9A-Za-z_-]{20,}",
            r"sk-[A-Za-z0-9_-]{20,}",
            r"(?i)Bearer\s+\S+",
            r"(?i)credential(?:s)?[=:]\s*\S+",
        ):
            result = re.sub(pattern, "<redacted-secret>", result)
        return result

    def safe_value(child: Any) -> Any:
        if isinstance(child, (bool, int, float)) or child is None:
            return child
        if isinstance(child, Mapping):
            safe: dict[str, Any] = {}
            for key, nested in child.items():
                normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if normalized in {
                    "apikey",
                    "token",
                    "accesstoken",
                    "credential",
                    "credentials",
                    "secret",
                    "authorization",
                    "password",
                    "privatekey",
                    "clientsecret",
                    "cookie",
                    "setcookie",
                    "sdkhttpresponse",
                    "rawhttp",
                    "client",
                    "transport",
                    "environment",
                    "clientconfig",
                } or normalized.endswith("token"):
                    continue
                safe[str(key)] = safe_value(nested)
            return safe
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes, bytearray)):
            return [safe_value(item) for item in child]
        if isinstance(child, str):
            return sanitize_text(child)
        if isinstance(child, BaseException):
            return sanitize_text(str(child))
        model_dump = getattr(child, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
            except TypeError:
                dumped = model_dump()
            return safe_value(dumped)
        return sanitize_text(str(_scalar(child)))

    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": sanitize_text(str(value))}
    if isinstance(value, Mapping):
        allowed = {"code", "status", "message", "details", "reason", "type"}
        return {
            str(key): safe_value(child)
            for key, child in value.items()
            if str(key) in allowed
        }
    result: dict[str, Any] = {}
    for name in ("code", "status", "message", "details", "reason", "type"):
        child = _attr(value, name, default=None)
        if child is not None:
            result[name] = safe_value(child)
    if result:
        return result
    return {"type": type(value).__name__}


def _provider_digest_hex(value: Any, *, label: str) -> str:
    """Normalize provider SHA-256 hex/base64 into canonical lowercase hex."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a SHA-256 hex or base64 digest")
    digest = value.strip()
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return digest.casefold()
    if not re.fullmatch(r"[A-Za-z0-9_+/-]+={0,2}", digest):
        raise ValueError(f"{label} is not valid base64")
    standard = digest.replace("-", "+").replace("_", "/")
    if len(standard) % 4 == 1:
        raise ValueError(f"{label} is not valid base64")
    standard += "=" * (-len(standard) % 4)
    try:
        decoded = base64.b64decode(standard, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"{label} is not valid base64") from error
    if len(decoded) == hashlib.sha256().digest_size:
        return decoded.hex()
    if len(decoded) == hashlib.sha256().digest_size * 2:
        try:
            decoded_hex = decoded.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} is not a SHA-256 digest") from error
        if re.fullmatch(r"[0-9a-fA-F]{64}", decoded_hex):
            return decoded_hex.casefold()
    raise ValueError(f"{label} is not a SHA-256 digest")


def _provider_file_metadata(file: Any, *, expected_mime: str) -> dict[str, Any]:
    name = _nonempty_string(_attr(file, "name", "file_name", "fileName"), label="uploaded file name")
    uri = _nonempty_string(_attr(file, "uri", "file_uri", "fileUri"), label="uploaded file URI")
    mime = _nonempty_string(_attr(file, "mime_type", "mimeType"), label="uploaded MIME type")
    if mime != expected_mime:
        raise ValueError("provider returned the wrong MIME type")
    file_state = _state(_attr(file, "state", "file_state", "fileState", default=""))
    if file_state not in {"ACTIVE", "pending", "failed"}:
        file_state = str(_scalar(_attr(file, "state", default="")))
    metadata = {"name": name, "uri": uri, "mime_type": mime, "file_state": file_state}
    size = _attr(file, "size_bytes", "sizeBytes", default=None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        metadata["size_bytes"] = size
    digest = _attr(file, "sha256_hash", "sha256Hash", default=None)
    if isinstance(digest, str) and digest:
        metadata["sha256_hash"] = _provider_digest_hex(digest, label="provider file sha256_hash")
    error = _safe_error(_attr(file, "error", default=None))
    if error is not None:
        metadata["error"] = error
    return metadata


def preflight_models(root: Path, client: Any, checked_at: str) -> dict[str, Any]:
    """Fetch and freeze both exact model records without an inference call."""

    root, protocol, protocol_hash = _protocol_context(root)
    path = root / _MODELS_FILENAME
    if path.exists() or path.is_symlink():
        raise ValueError("model preflight already exists")
    records: dict[str, Any] = {}
    for arm in ARMS:
        model = client.models.get(model=ARMS[arm]["model"])
        record = _model_snapshot(model, arm=arm)
        record["provider_snapshot_sha256"] = _sha256_bytes(_canonical_json(record))
        records[arm] = record
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "models_frozen",
        "checked_at": checked_at,
        "protocol_sha256": protocol_hash,
        "models": records,
    }
    _write_json_exclusive(path, body)
    return _read_object(path)


def upload_rows(root: Path, client: Any, *, sleep: Callable[[float], None]) -> dict[str, Any]:
    """Upload exactly the two frozen deliveries once and wait for ACTIVE."""

    root, protocol, protocol_hash = _protocol_context(root)
    _models_artifact(root, protocol_hash)
    upload_dir = root / _UPLOADS_DIRNAME
    if upload_dir.exists() or upload_dir.is_symlink():
        raise ValueError("upload records already exist; refusing a duplicate upload")
    _ensure_directory(upload_dir, label="upload records", create=True)
    old_uris = _historical_uris(protocol)
    records: list[dict[str, Any]] = []
    for row in _protocol_rows(protocol):
        row_id = _row_id(row, context="protocol row")
        attempt_path = upload_dir / f"{row_id}.attempt.json"
        record_path = upload_dir / f"{row_id}.json"
        delivery = _upload_file_path(protocol, row)
        before_hash = _sha256_file(delivery)
        before_size = delivery.stat().st_size
        attempt = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "upload_attempt_reserved",
            "protocol_sha256": protocol_hash,
            "row_id": row_id,
            "video_id": row["video_id"],
            "delivery_sha256": before_hash,
            "delivery_size_bytes": before_size,
            "mime_type": row["delivery_mime_type"],
        }
        _write_json_exclusive(attempt_path, attempt)
        attempt_sha256 = _verified_artifact_file_hash(
            attempt_path, label=f"upload attempt {row_id}"
        )
        provider_identity: dict[str, Any] | None = None
        try:
            uploaded = client.files.upload(
                file=delivery,
                config=types.UploadFileConfig(mime_type=row["delivery_mime_type"]),
            )
            if _sha256_file(delivery) != before_hash or delivery.stat().st_size != before_size:
                raise ValueError("delivery mutated during upload")
            initial = _provider_file_metadata(uploaded, expected_mime=row["delivery_mime_type"])
            provider_identity = initial
            if initial["uri"] in old_uris:
                raise ValueError("provider reused an expired historical upload URI")
            if initial.get("size_bytes") is not None and initial["size_bytes"] != before_size:
                raise ValueError("provider upload size does not match delivery")
            if initial.get("sha256_hash") is not None and initial["sha256_hash"] != before_hash:
                raise ValueError("provider upload hash does not match delivery")
            current = initial
            for poll_index in range(_MAX_UPLOAD_POLLS):
                current_file = client.files.get(name=initial["name"])
                current = _provider_file_metadata(
                    current_file, expected_mime=row["delivery_mime_type"]
                )
                provider_identity = current
                if current["uri"] != initial["uri"] or current["name"] != initial["name"]:
                    raise ValueError("provider upload identity changed while polling")
                if current.get("size_bytes") is not None and current["size_bytes"] != before_size:
                    raise ValueError("provider upload size changed while polling")
                if current.get("sha256_hash") is not None and current["sha256_hash"] != before_hash:
                    raise ValueError("provider upload hash changed while polling")
                if current["file_state"] == "ACTIVE":
                    break
                if current["file_state"] == "failed":
                    raise ValueError("provider upload entered FAILED state")
                if poll_index == _MAX_UPLOAD_POLLS - 1:
                    raise ValueError("provider upload timed out before ACTIVE")
                sleep(_UPLOAD_POLL_SECONDS)
            if _sha256_file(delivery) != before_hash or delivery.stat().st_size != before_size:
                raise ValueError("delivery mutated after upload")
            body = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "status": "uploaded",
                "protocol_sha256": protocol_hash,
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "delivery_sha256": before_hash,
                "delivery_size_bytes": before_size,
                "mime_type": row["delivery_mime_type"],
                "attempt_sha256": attempt_sha256,
                "file_name": current["name"],
                "file_uri": current["uri"],
                "file_state": "ACTIVE",
                "provider_snapshot": current,
            }
            _write_json_exclusive(record_path, body)
            records.append(_read_object(record_path))
        except Exception as error:  # noqa: BLE001 - preserve one terminal attempt
            failure = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "status": "failed",
                "protocol_sha256": protocol_hash,
                "row_id": row_id,
                "video_id": row["video_id"],
                "delivery_sha256": before_hash,
                "delivery_size_bytes": before_size,
                "mime_type": row["delivery_mime_type"],
                "attempt_sha256": attempt_sha256,
                "provider_snapshot": provider_identity,
                "provider_error": _safe_error(error),
            }
            if not record_path.exists() and not record_path.is_symlink():
                _write_json_exclusive(record_path, failure)
            raise
    return {
        "status": "uploaded",
        "protocol_sha256": protocol_hash,
        "active_count": len(records),
        "uploads": records,
    }


def _request_record(
    *, row: Mapping[str, Any], upload: Mapping[str, Any], arm: str, protocol_hash: str, request: Any
) -> dict[str, Any]:
    request_dump = _plain(request.model_dump(mode="json", by_alias=True, exclude_none=True))
    return {
        "row_id": row["row_id"],
        "video_id": row["video_id"],
        "source_order_index": row["source_order_index"],
        "question": row["question"],
        "file_uri": upload["file_uri"],
        "delivery_sha256": row["delivery_sha256"],
        "mime_type": row["delivery_mime_type"],
        "protocol_sha256": protocol_hash,
        "request_metadata": _plain(request.metadata),
        "request_sha256": _sha256_bytes(_canonical_json(request_dump)),
    }


def _bind_request_row(row: Mapping[str, Any], protocol_hash: str) -> dict[str, Any]:
    bound = dict(row)
    bound["protocol_sha256"] = protocol_hash
    return bound


def _expected_request_records(
    protocol: Mapping[str, Any], uploads: Sequence[Mapping[str, Any]], arm: str, protocol_hash: str
) -> list[dict[str, Any]]:
    rows = _protocol_rows(protocol)
    if len(uploads) != len(rows):
        raise ValueError("upload and protocol row counts differ")
    requests = [
        build_inlined_request(row=_bind_request_row(row, protocol_hash), upload=upload, arm=arm)
        for row, upload in zip(rows, uploads, strict=True)
    ]
    return [
        _request_record(row=row, upload=upload, arm=arm, protocol_hash=protocol_hash, request=request)
        for row, upload, request in zip(rows, uploads, requests, strict=True)
    ]


def _provider_model_matches(observed: Any, expected: str) -> bool:
    """Accept the documented model ID and the live API resource-name form."""

    return observed is None or observed in {expected, f"models/{expected}"}


def _validated_job_record(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    uploads: Sequence[Mapping[str, Any]],
    arm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_directory(root / _JOBS_DIRNAME, label="jobs")
    path = root / _JOBS_DIRNAME / f"{arm}.json"
    job = _verify_artifact(path)
    attempt_path = root / _JOBS_DIRNAME / f"{arm}.attempt.json"
    attempt = _verify_artifact(attempt_path)
    attempt_sha256 = _verified_artifact_file_hash(attempt_path, label=f"{arm} submission attempt")
    if job.get("attempt_sha256") != attempt_sha256:
        raise ValueError(f"job attempt binding mismatch for {arm}")
    if (
        attempt.get("protocol_sha256") != protocol_hash
        or attempt.get("arm") != arm
        or attempt.get("model") != ARMS[arm]["model"]
    ):
        raise ValueError(f"job attempt binding mismatch for {arm}")
    expected_requests = _expected_request_records(protocol, uploads, arm, protocol_hash)
    if (
        job.get("protocol_sha256") != protocol_hash
        or job.get("arm") != arm
        or job.get("model") != ARMS[arm]["model"]
        or job.get("request_count") != 2
        or job.get("requests") != expected_requests
    ):
        raise ValueError(f"job request binding mismatch for {arm}")
    expected_hash = _sha256_bytes(_canonical_json({"arm": arm, "requests": expected_requests}))
    if job.get("request_hash") != expected_hash:
        raise ValueError(f"job request hash mismatch for {arm}")
    snapshot = job.get("provider_snapshot")
    if isinstance(snapshot, Mapping) and not _provider_model_matches(
        snapshot.get("model"), ARMS[arm]["model"]
    ):
        raise ValueError(f"job provider model mismatch for {arm}")
    if job.get("provider_snapshot_sha256") != _sha256_bytes(_canonical_json(snapshot or {})):
        raise ValueError(f"job provider snapshot hash mismatch for {arm}")
    return job, expected_requests


def _job_snapshot(job: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for source_name, output_name in (
        ("name", "name"),
        ("display_name", "display_name"),
        ("model", "model"),
        ("state", "state"),
        ("create_time", "create_time"),
        ("update_time", "update_time"),
    ):
        value = _attr(job, source_name, output_name, default=None)
        if value is not None:
            snapshot[output_name] = _scalar(value)
    for name in ("error", "failure", "status_error"):
        error = _safe_error(_attr(job, name, default=None))
        if error is not None:
            snapshot["error"] = error
            break
    return snapshot


def submit_arm(root: Path, arm: str, client: Any, *, submitted_at: str) -> dict[str, Any]:
    """Submit exactly one two-request inline Batch job for one arm."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    root, protocol, protocol_hash = _protocol_context(root)
    _models_artifact(root, protocol_hash)
    uploads = _upload_records(root, protocol, protocol_hash)
    jobs_dir = root / _JOBS_DIRNAME
    _ensure_directory(jobs_dir, label="jobs", create=True)
    job_path = jobs_dir / f"{arm}.json"
    if job_path.exists() or job_path.is_symlink():
        raise ValueError(f"submission already exists for {arm}")
    attempt_path = jobs_dir / f"{arm}.attempt.json"
    if attempt_path.exists() or attempt_path.is_symlink():
        raise ValueError(f"submission attempt already exists for {arm}")
    response_path = root / "responses" / f"{arm}.json"
    if response_path.exists() or response_path.is_symlink():
        raise ValueError(f"response evidence already exists for {arm}")
    status_dir = root / _STATUS_DIRNAME / arm
    if status_dir.exists() and any(status_dir.iterdir()):
        raise ValueError(f"status evidence already exists for {arm}")
    rows = _protocol_rows(protocol)
    requests = [
        build_inlined_request(row=_bind_request_row(row, protocol_hash), upload=upload, arm=arm)
        for row, upload in zip(rows, uploads, strict=True)
    ]
    if len(requests) != 2:
        raise ValueError("each Batch job must contain exactly two requests")
    request_records = [
        _request_record(row=row, upload=upload, arm=arm, protocol_hash=protocol_hash, request=request)
        for row, upload, request in zip(rows, uploads, requests, strict=True)
    ]
    request_hash = _sha256_bytes(_canonical_json({"arm": arm, "requests": request_records}))
    config = types.CreateBatchJobConfig(
        display_name=(
            "gemini37-smoke-control-31pro"
            if arm == "control_31pro"
            else "gemini37-smoke-candidate-37flash"
        )
    )
    reservation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "attempt_reserved",
        "arm": arm,
        "submitted_at": submitted_at,
        "protocol_sha256": protocol_hash,
        "model": ARMS[arm]["model"],
        "display_name": config.display_name,
        "request_count": 2,
        "requests": request_records,
        "request_hash": request_hash,
    }
    _write_json_exclusive(attempt_path, reservation)
    attempt_sha256 = _verified_artifact_file_hash(
        attempt_path, label=f"{arm} submission attempt"
    )
    try:
        job = client.batches.create(
            model=ARMS[arm]["model"],
            src=types.BatchJobSource(inlined_requests=requests),
            config=config,
        )
    except Exception as error:  # noqa: BLE001 - preserve the sole terminal call
        body = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "failed",
            "arm": arm,
            "submitted_at": submitted_at,
            "protocol_sha256": protocol_hash,
            "model": ARMS[arm]["model"],
            "display_name": config.display_name,
            "request_hash": request_hash,
            "request_count": 2,
            "requests": request_records,
            "provider_error": _safe_error(error),
            "attempt_sha256": attempt_sha256,
            "provider_snapshot": None,
            "provider_snapshot_sha256": _sha256_bytes(_canonical_json({})),
        }
        _write_json_exclusive(job_path, body)
        return _read_object(job_path)
    try:
        snapshot = _job_snapshot(job)
        name = _nonempty_string(snapshot.get("name"), label="Batch job name")
        body = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "submitted",
            "arm": arm,
            "submitted_at": submitted_at,
            "protocol_sha256": protocol_hash,
            "model": ARMS[arm]["model"],
            "display_name": config.display_name,
            "name": name,
            "request_count": 2,
            "requests": request_records,
            "request_hash": request_hash,
            "attempt_sha256": attempt_sha256,
            "provider_snapshot": snapshot,
            "provider_snapshot_sha256": _sha256_bytes(_canonical_json(snapshot)),
        }
        _write_json_exclusive(job_path, body)
        return _read_object(job_path)
    except Exception as error:  # noqa: BLE001 - terminalize malformed post-call result
        body = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "failed",
            "arm": arm,
            "submitted_at": submitted_at,
            "protocol_sha256": protocol_hash,
            "model": ARMS[arm]["model"],
            "display_name": config.display_name,
            "request_count": 2,
            "requests": request_records,
            "request_hash": request_hash,
            "provider_error": _safe_error(error),
            "attempt_sha256": attempt_sha256,
            "provider_snapshot": None,
            "provider_snapshot_sha256": _sha256_bytes(_canonical_json({})),
        }
        _write_json_exclusive(job_path, body)
        return _read_object(job_path)


def _extract_responses(job: Any) -> list[Any]:
    for name in ("responses", "inlined_responses", "results"):
        value = _attr(job, name, default=None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    destination = _attr(job, "dest", "destination", default=None)
    for name in ("responses", "inlined_responses", "results"):
        value = _attr(destination, name, default=None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return []


_RESPONSE_KEYS = frozenset(
    {
        "response",
        "key",
        "metadata",
        "error",
        "response_id",
        "responseId",
        "id",
        "model_version",
        "modelVersion",
        "candidates",
        "text",
        "finish_reason",
        "finishReason",
        "usage_metadata",
        "usageMetadata",
        "arm",
        "row_id",
        "video_id",
    }
)


def _safe_response_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        source = raw
    else:
        model_dump = getattr(raw, "model_dump", None)
        if callable(model_dump):
            try:
                source = model_dump(mode="json", by_alias=True, exclude_none=True)
            except TypeError:
                source = model_dump()
        else:
            source = {
                name: getattr(raw, name)
                for name in _RESPONSE_KEYS
                if hasattr(raw, name)
            }
    if not isinstance(source, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in source.items():
        if str(key) not in _RESPONSE_KEYS:
            continue
        if str(key) == "response" and value is not raw:
            safe[str(key)] = _safe_response_object(value)
        elif str(key) == "error":
            safe[str(key)] = _safe_error(value)
        else:
            safe[str(key)] = _plain(value)
    return safe


def record_status(root: Path, arm: str, client: Any, *, checked_at: str) -> dict[str, Any]:
    """Append one sanitized provider status snapshot for an already submitted arm."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    root, protocol, protocol_hash = _protocol_context(root)
    if (root / _PREDICTIONS_FILENAME).exists() or (root / _PREDICTIONS_FILENAME).is_symlink():
        raise ValueError("prediction freeze is terminal; status cannot be appended")
    _models_artifact(root, protocol_hash)
    uploads = _upload_records(root, protocol, protocol_hash)
    job_record = _verify_artifact(root / _JOBS_DIRNAME / f"{arm}.json")
    if job_record.get("protocol_sha256") != protocol_hash:
        raise ValueError("job protocol binding mismatch")
    validated_job, _ = _validated_job_record(root, protocol, protocol_hash, uploads, arm)
    name = job_record.get("name")
    if not isinstance(name, str) or not name.strip():
        return {
            "status": "failed",
            "arm": arm,
            "provider_error_type": job_record.get("provider_error_type", "submission_failed"),
            "provider_error": job_record.get("provider_error"),
        }
    job = client.batches.get(name=name)
    snapshot = _job_snapshot(job)
    raw_responses = _extract_responses(job)
    for raw_response in raw_responses:
        _reject_forbidden(_plain(raw_response), context=f"{arm} provider responses")
    snapshot["responses"] = [_safe_response_object(raw) for raw in raw_responses]
    state = _state(_attr(job, "state", "status", default=snapshot.get("state", "")))
    status_root = root / _STATUS_DIRNAME
    _ensure_directory(status_root, label="status", create=True)
    status_dir = status_root / arm
    _ensure_directory(status_dir, label=f"status/{arm}", create=True)
    sequence = len(list(status_dir.glob("*.json"))) + 1
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "arm": arm,
        "checked_at": checked_at,
        "protocol_sha256": protocol_hash,
        "job_sha256": _sha256_file(root / _JOBS_DIRNAME / f"{arm}.json"),
        "job_attempt_sha256": validated_job["attempt_sha256"],
        "job_name": name,
        "upload_attempt_sha256": {
            record["row_id"]: record["attempt_sha256"] for record in uploads
        },
        "provider_snapshot": snapshot,
        "provider_snapshot_sha256": _sha256_bytes(_canonical_json(snapshot)),
        "response_count": len(snapshot["responses"]),
    }
    path = status_dir / f"{sequence:04d}.json"
    _write_json_exclusive(path, body)
    return _read_object(path)


def _latest_status(root: Path, arm: str, protocol_hash: str) -> tuple[dict[str, Any], Path]:
    status_dir = root / _STATUS_DIRNAME / arm
    _ensure_directory(root / _STATUS_DIRNAME, label="status")
    _ensure_directory(status_dir, label=f"status/{arm}")
    paths = sorted(status_dir.glob("*.json")) if status_dir.is_dir() else []
    if not paths:
        raise ValueError(f"no status snapshot exists for {arm}")
    path = paths[-1]
    snapshot = _verify_artifact(path)
    if snapshot.get("protocol_sha256") != protocol_hash or snapshot.get("arm") != arm:
        raise ValueError(f"status snapshot binding mismatch for {arm}")
    job_path = root / _JOBS_DIRNAME / f"{arm}.json"
    if snapshot.get("job_sha256") != _sha256_file(job_path):
        raise ValueError(f"status job binding mismatch for {arm}")
    job = _verify_artifact(job_path)
    job_attempt_sha256 = _verified_artifact_file_hash(
        root / _JOBS_DIRNAME / f"{arm}.attempt.json", label=f"{arm} submission attempt"
    )
    if (
        snapshot.get("job_attempt_sha256") != job.get("attempt_sha256")
        or job.get("attempt_sha256") != job_attempt_sha256
    ):
        raise ValueError(f"status job attempt binding mismatch for {arm}")
    upload_attempts = snapshot.get("upload_attempt_sha256")
    if not isinstance(upload_attempts, Mapping):
        raise ValueError(f"status upload attempt binding is missing for {arm}")
    current_upload_attempts: dict[str, str] = {}
    upload_dir = root / _UPLOADS_DIRNAME
    _ensure_directory(upload_dir, label="upload records")
    for upload_path in sorted(upload_dir.glob("*.json")):
        if upload_path.name.endswith(".attempt.json"):
            continue
        upload_record = _verify_artifact(upload_path)
        row_id = upload_record.get("row_id")
        if not isinstance(row_id, str) or not _SAFE_CANONICAL_ROW_ID.fullmatch(row_id):
            raise ValueError(f"upload record row ID is unsafe for {arm}")
        attempt_sha256 = upload_record.get("attempt_sha256")
        if not isinstance(attempt_sha256, str):
            raise ValueError(f"upload record attempt binding is missing for {row_id}")
        current_upload_attempts[row_id] = attempt_sha256
    if set(upload_attempts) != set(current_upload_attempts):
        raise ValueError(f"status upload attempt row set mismatch for {arm}")
    if dict(upload_attempts) != current_upload_attempts:
        raise ValueError(f"status upload attempt binding mismatch for {arm}")
    for row_id, expected_hash in upload_attempts.items():
        if not isinstance(row_id, str) or not _SAFE_CANONICAL_ROW_ID.fullmatch(row_id):
            raise ValueError(f"status upload attempt row ID is unsafe for {arm}")
        attempt_path = root / _UPLOADS_DIRNAME / f"{row_id}.attempt.json"
        if expected_hash != _verified_artifact_file_hash(
            attempt_path, label=f"upload attempt {row_id}"
        ):
            raise ValueError(f"status upload attempt binding mismatch for {row_id}")
    provider_snapshot = snapshot.get("provider_snapshot")
    if snapshot.get("provider_snapshot_sha256") != _sha256_bytes(
        _canonical_json(provider_snapshot if isinstance(provider_snapshot, Mapping) else {})
    ):
        raise ValueError(f"status provider snapshot hash mismatch for {arm}")
    return snapshot, path


def _response_payload(raw: Any) -> dict[str, Any]:
    value = _safe_response_object(raw)
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("response")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        for key, child in value.items():
            if key != "response":
                merged[key] = child
        return merged
    return dict(value)


def _response_text(response: Mapping[str, Any]) -> str | None:
    text = response.get("text")
    if isinstance(text, str):
        return text
    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        content = candidates[0].get("content")
        if isinstance(content, Mapping):
            parts = content.get("parts")
            if isinstance(parts, list):
                chunks = [part.get("text") for part in parts if isinstance(part, Mapping) and isinstance(part.get("text"), str)]
                if chunks:
                    return "".join(chunks)
    return None


def _usage(response: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage_metadata", response.get("usageMetadata"))
    if not isinstance(usage, Mapping):
        return None
    aliases = {
        "prompt_tokens": ("prompt_token_count", "promptTokenCount"),
        "total_tokens": ("total_token_count", "totalTokenCount"),
        "output_tokens": ("candidates_token_count", "candidatesTokenCount"),
    }
    result: dict[str, Any] = {}
    for output, names in aliases.items():
        found = next((usage.get(name) for name in names if name in usage), None)
        if isinstance(found, bool) or not isinstance(found, int) or found < 0:
            return None
        result[output] = found
    if result["total_tokens"] < result["prompt_tokens"]:
        return None
    metadata = {
        str(key): _plain(value)
        for key, value in usage.items()
        if str(key).casefold().replace("-", "_")
        not in {"sdk_http_response", "sdkhttpresponse", "client", "transport", "authorization"}
    }
    result["metadata"] = metadata
    return result


def _model_versions_match(model_id: str, expected: str, observed: Any) -> bool:
    if not isinstance(observed, str) or not observed.strip():
        return False
    observed = observed.strip()
    if observed == expected:
        return True
    return observed == f"{model_id}-{expected}"


def _validated_responses(
    *,
    responses: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    expected_version: str,
    expected_model: str,
    expected_arm: str,
    expected_requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    normalized: list[dict[str, Any]] = []
    failures: list[str] = []
    usage_complete = True
    seen_ids: set[str] = set()
    for index, raw in enumerate(responses):
        response = _response_payload(raw)
        response_id = response.get("response_id", response.get("responseId", response.get("id")))
        if not isinstance(response_id, str) or not response_id.strip():
            failures.append(f"response_{index}_missing_id")
            response_id = ""
        elif response_id in seen_ids:
            failures.append("duplicate_response_id")
        seen_ids.add(response_id)
        model_version = response.get("model_version", response.get("modelVersion"))
        if not _model_versions_match(expected_model, expected_version, model_version):
            failures.append(f"response_{index}_model_version")
        if response.get("arm") not in (None, expected_arm):
            failures.append(f"response_{index}_arm")
        expected_row = rows[index] if index < len(rows) else {}
        expected_metadata = (
            expected_requests[index].get("request_metadata")
            if index < len(expected_requests)
            else None
        )
        response_metadata = response.get("metadata")
        if not isinstance(response_metadata, Mapping) or dict(response_metadata) != expected_metadata:
            failures.append(f"response_{index}_metadata")
        if response.get("row_id") not in (None, expected_row.get("row_id")):
            failures.append(f"response_{index}_row_id")
        if response.get("video_id") not in (None, expected_row.get("video_id")):
            failures.append(f"response_{index}_video_id")
        error = response.get("error")
        if error:
            failures.append(f"response_{index}_error")
        text = _response_text(response)
        if not isinstance(text, str) or not text.strip():
            failures.append(f"response_{index}_empty_text")
            text = ""
        finish = response.get("finish_reason", response.get("finishReason"))
        if not isinstance(finish, str) or not finish.strip():
            candidates = response.get("candidates")
            if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
                finish = candidates[0].get("finish_reason", candidates[0].get("finishReason"))
        if not isinstance(finish, str) or not finish.strip():
            failures.append(f"response_{index}_missing_finish_reason")
            finish = ""
        usage = _usage(response)
        if usage is None:
            usage_complete = False
        normalized.append(
            {
                "row_id": rows[index]["row_id"] if index < len(rows) else None,
                "video_id": rows[index]["video_id"] if index < len(rows) else None,
                "response_id": response_id,
                "finish_reason": finish,
                "model_version": model_version,
                "metadata": dict(response_metadata) if isinstance(response_metadata, Mapping) else None,
                "error": _safe_error(response.get("error")),
                "text": text,
                "usage_metadata": usage.get("metadata") if usage is not None else None,
                "usage": usage,
            }
        )
    if len(responses) != 2:
        failures.append("response_count")
    if len(normalized) == 2 and len({item["response_id"] for item in normalized}) != 2:
        if "duplicate_response_id" not in failures:
            failures.append("duplicate_response_id")
    return normalized, failures, usage_complete


def freeze_predictions(root: Path, *, frozen_at: str) -> dict[str, Any]:
    """Create one immutable prediction freeze from terminal status snapshots."""

    root, protocol, protocol_hash = _protocol_context(root)
    path = root / _PREDICTIONS_FILENAME
    if path.exists() or path.is_symlink():
        raise ValueError("prediction freeze already exists")
    models = _models_artifact(root, protocol_hash)
    upload_records = _upload_records(root, protocol, protocol_hash)
    upload_attempt_sha256 = {
        record["row_id"]: record["attempt_sha256"] for record in upload_records
    }
    rows = _protocol_rows(protocol)
    arm_results: dict[str, Any] = {}
    failure_reasons: list[str] = []
    hard_failure = False
    custom_fps_candidate = False
    observed_cost = 0.0
    for arm in ARMS:
        try:
            job_record, expected_requests = _validated_job_record(
                root, protocol, protocol_hash, upload_records, arm
            )
        except ValueError as error:
            failure_reasons.append(f"{arm}:job_binding:{type(error).__name__}")
            arm_results[arm] = {"status": "not_ready", "response_count": 0}
            hard_failure = True
            continue
        try:
            status, status_path = _latest_status(root, arm, protocol_hash)
        except ValueError as error:
            failure_reasons.append(f"{arm}:{type(error).__name__}")
            arm_results[arm] = {"status": "not_ready", "response_count": 0}
            hard_failure = True
            continue
        state = status.get("status")
        responses_raw = _attr(status.get("provider_snapshot"), "responses", default=[])
        if not isinstance(responses_raw, list):
            responses_raw = []
        provider_snapshot = status.get("provider_snapshot")
        model_record = models["models"][arm]
        expected_version = model_record["version"]
        normalized, failures, usage_complete = _validated_responses(
            responses=responses_raw,
            rows=rows,
            expected_version=expected_version,
            expected_model=ARMS[arm]["model"],
            expected_arm=arm,
            expected_requests=expected_requests,
        )
        if state != "succeeded":
            failures.append(f"job_{state}")
        if isinstance(provider_snapshot, Mapping) and not _provider_model_matches(
            provider_snapshot.get("model"), ARMS[arm]["model"]
        ):
            failures.append("provider_model")
        if job_record.get("requests") != expected_requests:
            failures.append("request_binding")
        if failures:
            failure_reasons.extend(f"{arm}:{reason}" for reason in failures)
        prompt_tokens = sum(
            item["usage"]["prompt_tokens"] for item in normalized if item["usage"] is not None
        )
        output_tokens = sum(
            item["usage"]["total_tokens"] - item["usage"]["prompt_tokens"]
            for item in normalized
            if item["usage"] is not None
        )
        fps_ok = usage_complete and _FPS_PROMPT_TOKEN_MIN <= prompt_tokens <= _FPS_PROMPT_TOKEN_MAX
        response_valid = not failures and state == "succeeded" and len(normalized) == 2
        if response_valid and not usage_complete:
            custom_fps_candidate = True
            failure_reasons.append(f"{arm}:usage_missing")
        if response_valid and usage_complete and not fps_ok:
            custom_fps_candidate = True
            failure_reasons.append(f"{arm}:prompt_tokens_out_of_band")
        if failures:
            hard_failure = True
        arm_cost = _arm_observed_cost_usd(arm, prompt_tokens, output_tokens)
        observed_cost += arm_cost
        arm_results[arm] = {
            "status": "succeeded" if response_valid and fps_ok else "not_ready",
            "status_snapshot_sha256": _sha256_file(status_path),
            "provider_snapshot_sha256": status.get("provider_snapshot_sha256"),
            "request_hash": _verify_artifact(root / _JOBS_DIRNAME / f"{arm}.json").get("request_hash"),
            "job_attempt_sha256": job_record["attempt_sha256"],
            "model_snapshot_sha256": model_record["provider_snapshot_sha256"],
            "model_version": expected_version,
            "response_count": len(normalized),
            "responses": normalized,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "observed_cost_usd": round(arm_cost, 8),
            "custom_fps_verified": fps_ok,
        }
    if hard_failure:
        decision = "not_ready"
    elif custom_fps_candidate:
        decision = "custom_fps_not_verified"
    else:
        decision = "quality_ready"
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "predictions_frozen",
        "decision": decision,
        "frozen_at": frozen_at,
        "protocol_sha256": protocol_hash,
        "inputs_freeze_sha256": _sha256_file(root / "inputs-freeze.json"),
        "models_sha256": _sha256_file(root / _MODELS_FILENAME),
        "upload_record_sha256": _hash_paths(sorted((root / _UPLOADS_DIRNAME).glob("*.json"))),
        "upload_attempt_sha256": upload_attempt_sha256,
        "arms": arm_results,
        "observed_cost_usd": round(observed_cost, 8),
        "failure_reasons": failure_reasons,
    }
    _write_json_exclusive(path, body)
    return _read_object(path)


# Post-freeze scoring is intentionally separate from provider lifecycle code.
# These helpers never run until the prediction freeze and every upstream hash
# binding have been checked, so an incomplete or mutated freeze cannot cause an
# annotation file to be opened.
_SCORE_FILENAME = "postfreeze-score.json"
_SCORE_FREEZE_FILENAME = "postfreeze-score-freeze.json"
_ANNOTATION_FREEZE_FILENAME = "annotations-freeze.json"
_ANNOTATION_REFERENCE_KEYS = (
    "correct_answer",
    "correctAnswer",
    "reference",
    "reference_answer",
    "referenceAnswer",
    "answer",
    "answers",
)


def _normalized_answer(value: str) -> str:
    return " ".join(_answer_tokens(value))


def _answer_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _multiset_token_f1(reference: str, answer: str) -> float:
    reference_tokens = _answer_tokens(reference)
    answer_tokens = _answer_tokens(answer)
    if not reference_tokens and not answer_tokens:
        return 1.0
    if not reference_tokens or not answer_tokens:
        return 0.0
    from collections import Counter

    overlap = sum((Counter(reference_tokens) & Counter(answer_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _annotation_reference(payload: Any, *, label: str) -> str:
    if isinstance(payload, str):
        value = payload
    elif isinstance(payload, Mapping):
        value = None
        for key in _ANNOTATION_REFERENCE_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, str):
                value = candidate
                break
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                strings = [item for item in candidate if isinstance(item, str)]
                if strings:
                    value = strings[0]
                    break
        if value is None:
            nested = payload.get("annotation")
            if nested is not payload:
                return _annotation_reference(nested, label=label)
    else:
        value = None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"canonical annotation reference is missing for {label}")
    return value.strip()


def _canonical_annotation_path(annotations_dir: Path, row: Mapping[str, Any]) -> Path:
    if annotations_dir.is_symlink() or not annotations_dir.is_dir():
        raise ValueError("canonical annotations directory is missing or not regular")
    video_id = _video_id(row, context="score row")
    _row_id(row, context="score row")
    _validate_canonical_video_id(video_id, context="score video ID")
    candidates = [annotations_dir / f"{video_id}.json"]
    existing = [candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()]
    if len(existing) != 1:
        raise ValueError(f"canonical annotation binding is missing or ambiguous for {video_id}")
    path = existing[0]
    if not _is_regular_file(path):
        raise ValueError(f"canonical annotation is missing or not regular for {video_id}")
    return path


def _raw_annotation_bytes(path: Path) -> bytes:
    if not _is_regular_file(path):
        raise ValueError("canonical annotation is missing or not regular")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError("canonical annotation bytes cannot be read") from error


def freeze_annotation_bindings(
    root: Path, annotations_dir: Path, *, bound_at: str
) -> dict[str, Any]:
    """Freeze exact canonical annotation paths and raw-byte hashes post-prediction."""

    root = Path(root)
    manifest_path = root / _ANNOTATION_FREEZE_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("annotation bindings are already frozen")
    freeze, protocol, protocol_hash = _verify_prediction_freeze_for_score(root)
    annotations_dir = Path(annotations_dir)
    if annotations_dir.is_symlink() or not annotations_dir.is_dir():
        raise ValueError("canonical annotations directory is missing or not regular")
    annotations_dir = annotations_dir.resolve(strict=True)
    rows = _protocol_rows(protocol)
    files: list[dict[str, Any]] = []
    for row in rows:
        path = _canonical_annotation_path(annotations_dir, row).resolve(strict=True)
        raw = _raw_annotation_bytes(path)
        files.append(
            {
                "video_id": row["video_id"],
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "size_bytes": len(raw),
            }
        )
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "annotations_frozen",
        "bound_at": bound_at,
        "protocol_sha256": protocol_hash,
        "prediction_freeze_sha256": _sha256_file(root / _PREDICTIONS_FILENAME),
        "annotations_dir": str(annotations_dir),
        "files": files,
    }
    _write_json_exclusive(manifest_path, body)
    return _read_object(manifest_path)


def _verified_annotation_bindings(
    root: Path,
    annotations_dir: Path,
    freeze: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path, bytes]]]:
    """Verify the private raw-byte annotation manifest before any JSON parse."""

    manifest_path = Path(root) / _ANNOTATION_FREEZE_FILENAME
    if not _is_regular_file(manifest_path):
        raise ValueError("annotation freeze manifest is missing")
    manifest = _verify_artifact(manifest_path)
    if manifest.get("status") != "annotations_frozen":
        raise ValueError("annotation freeze manifest is incomplete")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("annotation freeze manifest experiment mismatch")
    if manifest.get("protocol_sha256") != freeze.get("protocol_sha256"):
        raise ValueError("annotation freeze protocol hash mismatch")
    if manifest.get("prediction_freeze_sha256") != _sha256_file(
        Path(root) / _PREDICTIONS_FILENAME
    ):
        raise ValueError("annotation freeze prediction hash mismatch")
    annotations_dir = Path(annotations_dir)
    if annotations_dir.is_symlink() or not annotations_dir.is_dir():
        raise ValueError("canonical annotations directory is missing or not regular")
    annotations_dir = annotations_dir.resolve(strict=True)
    if manifest.get("annotations_dir") != str(annotations_dir):
        raise ValueError("annotation freeze directory binding mismatch")
    rows = _protocol_rows(protocol)
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(rows):
        raise ValueError("annotation freeze manifest is incomplete")
    expected_ids = [str(row["video_id"]) for row in rows]
    observed_ids = [item.get("video_id") if isinstance(item, Mapping) else None for item in files]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(expected_ids):
        raise ValueError("annotation freeze manifest IDs/order are invalid")
    checked: list[tuple[dict[str, Any], Path, bytes]] = []
    for row, item in zip(rows, files, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "video_id", "path", "sha256", "size_bytes"
        }:
            raise ValueError("annotation freeze manifest file entry is invalid")
        path = _canonical_annotation_path(annotations_dir, row).resolve(strict=True)
        if item.get("path") != str(path):
            raise ValueError("annotation freeze manifest path binding mismatch")
        raw = _raw_annotation_bytes(path)
        if item.get("size_bytes") != len(raw) or item.get("sha256") != _sha256_bytes(raw):
            raise ValueError("annotation freeze annotation hash mismatch")
        checked.append((dict(item), path, raw))
    return manifest, checked


def _derived_arm_freeze_summary(
    root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    uploads: Sequence[Mapping[str, Any]],
    models: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    job, expected_requests = _validated_job_record(
        root, protocol, protocol_hash, uploads, arm
    )
    status, status_path = _latest_status(root, arm, protocol_hash)
    if status.get("status") != "succeeded":
        raise ValueError(f"prediction freeze status is incomplete for {arm}")
    model_record = models["models"][arm]
    provider_snapshot = status.get("provider_snapshot")
    if not isinstance(provider_snapshot, Mapping):
        raise ValueError(f"prediction freeze provider snapshot is missing for {arm}")
    if not _provider_model_matches(provider_snapshot.get("model"), ARMS[arm]["model"]):
        raise ValueError(f"prediction freeze provider model mismatch for {arm}")
    responses_raw = provider_snapshot.get("responses")
    if not isinstance(responses_raw, list):
        raise ValueError(f"prediction freeze responses are missing for {arm}")
    normalized, failures, usage_complete = _validated_responses(
        responses=responses_raw,
        rows=rows,
        expected_version=model_record.get("version"),
        expected_model=ARMS[arm]["model"],
        expected_arm=arm,
        expected_requests=expected_requests,
    )
    prompt_tokens = sum(
        item["usage"]["prompt_tokens"] for item in normalized if item["usage"] is not None
    )
    output_tokens = sum(
        item["usage"]["total_tokens"] - item["usage"]["prompt_tokens"]
        for item in normalized
        if item["usage"] is not None
    )
    fps_verified = (
        usage_complete
        and _FPS_PROMPT_TOKEN_MIN <= prompt_tokens <= _FPS_PROMPT_TOKEN_MAX
    )
    if failures or not usage_complete or len(normalized) != 2 or not fps_verified:
        raise ValueError(f"prediction freeze responses are incomplete for {arm}")
    arm_cost = _arm_observed_cost_usd(arm, prompt_tokens, output_tokens)
    return {
        "status": "succeeded",
        "status_snapshot_sha256": _sha256_file(status_path),
        "provider_snapshot_sha256": status.get("provider_snapshot_sha256"),
        "request_hash": job.get("request_hash"),
        "job_attempt_sha256": job.get("attempt_sha256"),
        "model_snapshot_sha256": model_record.get("provider_snapshot_sha256"),
        "model_version": model_record.get("version"),
        "response_count": len(normalized),
        "responses": normalized,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "observed_cost_usd": round(arm_cost, 8),
        "custom_fps_verified": fps_verified,
    }


def _verify_prediction_freeze_for_score(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    freeze_path = Path(root) / _PREDICTIONS_FILENAME
    if not _is_regular_file(freeze_path):
        raise ValueError("prediction freeze is missing")
    freeze = _verify_artifact(freeze_path)
    if freeze.get("status") != "predictions_frozen":
        raise ValueError("prediction freeze is incomplete")
    if freeze.get("decision") != "quality_ready":
        raise ValueError("prediction freeze is not quality-ready")
    failure_reasons = freeze.get("failure_reasons")
    if not isinstance(failure_reasons, list) or failure_reasons:
        raise ValueError("prediction freeze contains failure reasons")
    root, protocol, protocol_hash = _protocol_context(Path(root))
    if freeze.get("protocol_sha256") != protocol_hash:
        raise ValueError("prediction freeze protocol hash mismatch")
    for filename, key in (
        ("inputs-freeze.json", "inputs_freeze_sha256"),
        (_MODELS_FILENAME, "models_sha256"),
    ):
        if freeze.get(key) != _sha256_file(root / filename):
            raise ValueError(f"prediction freeze {key} mismatch")
    models = _models_artifact(root, protocol_hash)
    uploads = _upload_records(root, protocol, protocol_hash)
    upload_hash = _hash_paths(sorted((root / _UPLOADS_DIRNAME).glob("*.json")))
    if freeze.get("upload_record_sha256") != upload_hash:
        raise ValueError("prediction freeze upload record hash mismatch")
    rows = _protocol_rows(protocol)
    arms = freeze.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(ARMS):
        raise ValueError("prediction freeze arms are incomplete")
    expected_cost = 0.0
    expected_upload_attempts = {
        record["row_id"]: record["attempt_sha256"] for record in uploads
    }
    if freeze.get("upload_attempt_sha256") != expected_upload_attempts:
        raise ValueError("prediction freeze upload attempt binding mismatch")
    for arm in ARMS:
        result = arms.get(arm)
        if not isinstance(result, Mapping):
            raise ValueError(f"prediction freeze arm is missing for {arm}")
        expected = _derived_arm_freeze_summary(
            root, protocol, protocol_hash, uploads, models, rows, arm
        )
        if dict(result) != expected:
            raise ValueError(f"prediction freeze arm summary mismatch for {arm}")
        expected_cost += _arm_observed_cost_usd(
            arm, expected["prompt_tokens"], expected["output_tokens"]
        )
    if freeze.get("observed_cost_usd") != round(expected_cost, 8):
        raise ValueError("prediction freeze observed cost mismatch")
    if freeze.get("decision") != "quality_ready":
        raise ValueError("prediction freeze decision mismatch")
    return freeze, protocol, protocol_hash


def _verify_prediction_freeze_terminal(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Re-derive any terminal prediction decision, including non-quality outcomes."""

    freeze_path = Path(root) / _PREDICTIONS_FILENAME
    if not _is_regular_file(freeze_path):
        raise ValueError("prediction freeze is missing")
    freeze = _verify_artifact(freeze_path)
    if freeze.get("status") != "predictions_frozen":
        raise ValueError("prediction freeze is incomplete")
    decision = freeze.get("decision")
    if decision not in {"quality_ready", "custom_fps_not_verified", "not_ready"}:
        raise ValueError("prediction freeze decision is invalid")
    root, protocol, protocol_hash = _protocol_context(Path(root))
    if freeze.get("protocol_sha256") != protocol_hash:
        raise ValueError("prediction freeze protocol hash mismatch")
    if freeze.get("inputs_freeze_sha256") != _sha256_file(root / "inputs-freeze.json"):
        raise ValueError("prediction freeze inputs hash mismatch")
    if freeze.get("models_sha256") != _sha256_file(root / _MODELS_FILENAME):
        raise ValueError("prediction freeze models hash mismatch")
    models = _models_artifact(root, protocol_hash)
    uploads = _upload_records(root, protocol, protocol_hash)
    if freeze.get("upload_record_sha256") != _hash_paths(
        sorted((root / _UPLOADS_DIRNAME).glob("*.json"))
    ):
        raise ValueError("prediction freeze upload record hash mismatch")
    expected_upload_attempts = {
        record["row_id"]: record["attempt_sha256"] for record in uploads
    }
    if freeze.get("upload_attempt_sha256") != expected_upload_attempts:
        raise ValueError("prediction freeze upload attempt binding mismatch")
    rows = _protocol_rows(protocol)
    arm_results: dict[str, Any] = {}
    failure_reasons: list[str] = []
    hard_failure = False
    custom_fps_candidate = False
    observed_cost = 0.0
    for arm in ARMS:
        try:
            job_record, expected_requests = _validated_job_record(
                root, protocol, protocol_hash, uploads, arm
            )
        except ValueError as error:
            failure_reasons.append(f"{arm}:job_binding:{type(error).__name__}")
            arm_results[arm] = {"status": "not_ready", "response_count": 0}
            hard_failure = True
            continue
        try:
            status, status_path = _latest_status(root, arm, protocol_hash)
        except ValueError as error:
            failure_reasons.append(f"{arm}:{type(error).__name__}")
            arm_results[arm] = {"status": "not_ready", "response_count": 0}
            hard_failure = True
            continue
        state = status.get("status")
        provider_snapshot = status.get("provider_snapshot")
        responses_raw = _attr(provider_snapshot, "responses", default=[])
        if not isinstance(responses_raw, list):
            responses_raw = []
        model_record = models["models"][arm]
        normalized, failures, usage_complete = _validated_responses(
            responses=responses_raw,
            rows=rows,
            expected_version=model_record["version"],
            expected_model=ARMS[arm]["model"],
            expected_arm=arm,
            expected_requests=expected_requests,
        )
        if state != "succeeded":
            failures.append(f"job_{state}")
        if isinstance(provider_snapshot, Mapping) and not _provider_model_matches(
            provider_snapshot.get("model"), ARMS[arm]["model"]
        ):
            failures.append("provider_model")
        if job_record.get("requests") != expected_requests:
            failures.append("request_binding")
        if failures:
            failure_reasons.extend(f"{arm}:{reason}" for reason in failures)
        prompt_tokens = sum(
            item["usage"]["prompt_tokens"] for item in normalized if item["usage"] is not None
        )
        output_tokens = sum(
            item["usage"]["total_tokens"] - item["usage"]["prompt_tokens"]
            for item in normalized
            if item["usage"] is not None
        )
        fps_ok = usage_complete and _FPS_PROMPT_TOKEN_MIN <= prompt_tokens <= _FPS_PROMPT_TOKEN_MAX
        response_valid = not failures and state == "succeeded" and len(normalized) == 2
        if response_valid and not usage_complete:
            custom_fps_candidate = True
            failure_reasons.append(f"{arm}:usage_missing")
        if response_valid and usage_complete and not fps_ok:
            custom_fps_candidate = True
            failure_reasons.append(f"{arm}:prompt_tokens_out_of_band")
        if failures:
            hard_failure = True
        arm_cost = _arm_observed_cost_usd(arm, prompt_tokens, output_tokens)
        observed_cost += arm_cost
        arm_results[arm] = {
            "status": "succeeded" if response_valid and fps_ok else "not_ready",
            "status_snapshot_sha256": _sha256_file(status_path),
            "provider_snapshot_sha256": status.get("provider_snapshot_sha256"),
            "request_hash": job_record.get("request_hash"),
            "job_attempt_sha256": job_record["attempt_sha256"],
            "model_snapshot_sha256": model_record["provider_snapshot_sha256"],
            "model_version": model_record["version"],
            "response_count": len(normalized),
            "responses": normalized,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "observed_cost_usd": round(arm_cost, 8),
            "custom_fps_verified": fps_ok,
        }
    expected_decision = (
        "not_ready"
        if hard_failure
        else "custom_fps_not_verified"
        if custom_fps_candidate
        else "quality_ready"
    )
    if decision != expected_decision:
        raise ValueError("prediction freeze decision precedence mismatch")
    if freeze.get("arms") != arm_results:
        raise ValueError("prediction freeze arm summary mismatch")
    if freeze.get("failure_reasons") != failure_reasons:
        raise ValueError("prediction freeze failure summary mismatch")
    if freeze.get("observed_cost_usd") != round(observed_cost, 8):
        raise ValueError("prediction freeze observed cost mismatch")
    return freeze, protocol, protocol_hash


def score_postfreeze(
    root: Path, annotations_dir: Path, *, scored_at: str
) -> dict[str, Any]:
    """Score the immutable two-arm prediction freeze against canonical answers.

    The returned object and private score artifact contain row-level evidence;
    callers such as the CLI must reduce it to aggregate fields before printing.
    """

    root = Path(root)
    score_path = root / _SCORE_FILENAME
    score_freeze_path = root / _SCORE_FREEZE_FILENAME
    if (
        score_path.exists()
        or score_path.is_symlink()
        or score_freeze_path.exists()
        or score_freeze_path.is_symlink()
    ):
        raise ValueError("post-freeze score already exists")
    freeze, protocol, protocol_hash = _verify_prediction_freeze_for_score(root)
    rows = _protocol_rows(protocol)
    annotation_manifest, bound_annotations = _verified_annotation_bindings(
        root, annotations_dir, freeze, protocol
    )
    annotations: list[dict[str, Any]] = []
    # Raw bytes were checked against the approved manifest before parsing.
    for row, (_, path, raw) in zip(rows, bound_annotations, strict=True):
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("canonical annotation JSON is invalid") from error
        if isinstance(payload, Mapping):
            for key, expected in (
                ("video_id", row["video_id"]),
                ("videoId", row["video_id"]),
                ("row_id", row["row_id"]),
                ("rowId", row["row_id"]),
            ):
                if key in payload and payload[key] != expected:
                    raise ValueError(f"canonical annotation identity mismatch for {row['video_id']}")
        reference = _annotation_reference(payload, label=str(row["video_id"]))
        annotations.append(
            {
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "reference": reference,
            }
        )
    if len(annotations) != 2 or len({item["row_id"] for item in annotations}) != 2:
        raise ValueError("canonical annotations must contain exactly two unique rows")
    response_by_arm = {
        arm: freeze["arms"][arm]["responses"] for arm in ARMS
    }
    rows_scored: list[dict[str, Any]] = []
    wins = ties = losses = 0
    exact_counts = {arm: 0 for arm in ARMS}
    f1_totals = {arm: 0.0 for arm in ARMS}
    for index, annotation in enumerate(annotations):
        reference = annotation["reference"]
        answers: dict[str, str] = {}
        metrics: dict[str, dict[str, Any]] = {}
        raw_f1: dict[str, float] = {}
        for arm in ARMS:
            responses = response_by_arm[arm]
            if not isinstance(responses, list) or index >= len(responses):
                raise ValueError(f"prediction response is missing for {arm}")
            answer = responses[index].get("text")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"prediction answer is empty for {arm}")
            answer = answer.strip()
            answers[arm] = answer
            exact = _normalized_answer(reference) == _normalized_answer(answer)
            f1 = _multiset_token_f1(reference, answer)
            raw_f1[arm] = f1
            metrics[arm] = {"exact_match": exact, "token_f1": round(f1, 8)}
            exact_counts[arm] += int(exact)
            f1_totals[arm] += f1
        control_f1 = raw_f1["control_31pro"]
        candidate_f1 = raw_f1["candidate_37flash"]
        if candidate_f1 > control_f1:
            wins += 1
        elif candidate_f1 < control_f1:
            losses += 1
        else:
            ties += 1
        rows_scored.append(
            {
                "row_id": annotation["row_id"],
                "video_id": annotation["video_id"],
                "question": rows[index]["question"],
                "reference": reference,
                "answers": answers,
                "metrics": metrics,
            }
        )
    aggregate = {
        "row_count": len(rows_scored),
        "exact_match": {
            arm: {"count": exact_counts[arm], "rate": exact_counts[arm] / len(rows_scored)}
            for arm in ARMS
        },
        "token_f1": {
            arm: f1_totals[arm] / len(rows_scored) for arm in ARMS
        },
        "paired": {"candidate_wins": wins, "ties": ties, "candidate_losses": losses},
    }
    decision = (
        "ready_for_larger_canary"
        if losses == 0 and wins > 0
        else "compatible_no_directional_claim"
    )
    body = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "scored_postfreeze",
        "decision": decision,
        "scored_at": scored_at,
        "protocol_sha256": protocol_hash,
        "prediction_freeze_sha256": _sha256_file(root / _PREDICTIONS_FILENAME),
        "annotation_bindings_sha256": _sha256_file(root / _ANNOTATION_FREEZE_FILENAME),
        "annotation_root": annotation_manifest["annotations_dir"],
        "annotation_hashes": {
            item["video_id"]: item["sha256"] for item in annotations
        },
        "aggregate": aggregate,
        "observed_cost_usd": freeze.get("observed_cost_usd"),
        "rows": rows_scored,
    }
    _write_json_exclusive(score_path, body)
    _write_json_exclusive(
        score_freeze_path,
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "scored_postfreeze",
            "score_sha256": _sha256_file(score_path),
            "annotation_bindings_sha256": _sha256_file(
                root / _ANNOTATION_FREEZE_FILENAME
            ),
        },
    )
    return _read_object(score_path)


def show_status(root: Path) -> dict[str, Any]:
    """Return a private status snapshot for the CLI to reduce to aggregates."""

    root = Path(root)
    prediction_path = root / _PREDICTIONS_FILENAME
    if prediction_path.exists() or prediction_path.is_symlink():
        prediction, _, _ = _verify_prediction_freeze_terminal(root)
        arms = prediction.get("arms")
        return {
            "status": prediction.get("status"),
            "decision": prediction.get("decision"),
            "arms": arms if isinstance(arms, Mapping) else {},
            "observed_cost_usd": prediction.get("observed_cost_usd"),
        }
    protocol_path = root / "protocol.json"
    if protocol_path.exists() or protocol_path.is_symlink():
        verification = verify_inputs(root)
        protocol = _verify_artifact(protocol_path)
        if protocol.get("status") != "inputs_frozen":
            raise ValueError("protocol status is invalid")
        rows = protocol.get("rows")
        return {
            "status": protocol.get("status"),
            "decision": "not_ready",
            "rows": rows if isinstance(rows, list) else [],
            "rows_verified": verification["rows_verified"],
            "estimated_cost_ceiling_usd": protocol.get("estimated_cost_ceiling_usd"),
        }
    return {"status": "not_ready", "decision": "not_ready", "rows": []}
