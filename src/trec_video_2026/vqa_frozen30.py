"""Fail-closed preparation and resumption controls for the frozen 30 panel."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from trec_video_2026.vqa_gate import (
    RunnerRequest,
    canonical_request_sha256,
    saved_request_matches,
)


EXPECTED_FROZEN_COUNT = 30
_FORBIDDEN_LABEL_PARTS = (
    "correct_answer",
    "incorrect_answer",
    "reference",
    "alternative",
    "evaluation",
    "metric",
    "score",
    "label",
)


@dataclass(frozen=True)
class FrozenExecution:
    """Immutable execution identity required for a reusable passed row."""

    condition_id: str
    model_id: str
    model_revision: str
    runtime_image: str
    device: str = "cuda:0 (ROCm)"
    precision: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in (
            "condition_id",
            "model_id",
            "model_revision",
            "runtime_image",
            "device",
            "precision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Frozen execution {name} cannot be empty")
        if len(self.model_revision) < 7:
            raise ValueError("Frozen execution requires an immutable model revision")
        if "@sha256:" not in self.runtime_image:
            raise ValueError("Frozen execution requires a digest-pinned runtime image")


class SerialBenchmarkLock:
    """Process lock that prevents two frozen benchmark conditions overlapping."""

    def __init__(self, path: Path, *, condition_id: str) -> None:
        self.path = path
        self.condition_id = condition_id
        self._handle = None

    def __enter__(self) -> SerialBenchmarkLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise RuntimeError(
                "Another benchmark condition is already running"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"condition_id": self.condition_id, "pid": os.getpid()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def audit_frozen_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    expected_video_ids: Sequence[str],
    condition_id: str,
) -> dict[str, Any]:
    """Validate an exact, ordered, label-free request set before launch."""

    expected = list(expected_video_ids)
    if len(expected) != EXPECTED_FROZEN_COUNT or len(set(expected)) != len(expected):
        raise ValueError("Frozen panel must contain exactly 30 unique video IDs")
    if len(requests) != EXPECTED_FROZEN_COUNT:
        raise ValueError("Frozen request set must contain exactly 30 rows")
    actual = [str(request.get("video_id", "")) for request in requests]
    if actual != expected:
        raise ValueError("Frozen requests must preserve the exact ordered panel")
    if any(request.get("condition_id") != condition_id for request in requests):
        raise ValueError("Frozen request condition identity does not match")

    canonical_fields = {field.name for field in fields(RunnerRequest)}
    for request in requests:
        unknown = sorted(set(request) - canonical_fields)
        if unknown:
            raise ValueError(
                "Runner request contains unrecognized or possible label fields: "
                + ", ".join(unknown)
            )
        _reject_label_fields(request)
        _validate_canonical_request(request)

    return {
        "row_count": len(requests),
        "condition_id": condition_id,
        "video_ids": actual,
        "request_set_sha256": canonical_request_sha256({"requests": list(requests)}),
        "references_loaded": False,
    }


def _validate_canonical_request(request: Mapping[str, Any]) -> None:
    required = {field.name for field in fields(RunnerRequest)} - {"audio_mode"}
    missing = sorted(required - set(request))
    if missing:
        raise ValueError("Canonical runner request is missing: " + ", ".join(missing))
    try:
        RunnerRequest(
            run_id=_text(request, "run_id"),
            condition_id=_text(request, "condition_id"),
            video_id=_text(request, "video_id"),
            question=_text(request, "question"),
            frame_paths=tuple(Path(value) for value in request["frame_paths"]),
            frame_timestamps=tuple(request["frame_timestamps"]),
            frame_descriptions=(
                tuple(request["frame_descriptions"])
                if request["frame_descriptions"] is not None
                else None
            ),
            transcript=request["transcript"],
            media_path=(
                Path(request["media_path"])
                if request["media_path"] is not None
                else None
            ),
            audio_path=(
                Path(request["audio_path"])
                if request["audio_path"] is not None
                else None
            ),
            audio_mode=request.get("audio_mode"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid canonical runner request") from error


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be text")
    return value


def _reject_label_fields(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if any(part in normalized for part in _FORBIDDEN_LABEL_PARTS):
                location = f"{path}.{key}" if path else str(key)
                raise ValueError(f"Runner request contains a label field: {location}")
            _reject_label_fields(nested, path=f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_label_fields(nested, path=f"{path}[{index}]")


def evidence_identity_sha256(request: Mapping[str, Any]) -> str:
    """Bind the request's media-derived files and exact transcript bytes."""

    file_fields: dict[str, Any] = {}
    for key in ("media_path", "audio_path"):
        value = request.get(key)
        file_fields[key] = _file_identity(Path(value)) if value else None
    frames = request.get("frame_paths", [])
    if not isinstance(frames, list):
        raise ValueError("frame_paths must be a list")
    file_fields["frame_paths"] = [_file_identity(Path(value)) for value in frames]
    identity = {
        "request_sha256": canonical_request_sha256(request),
        "files": file_fields,
        "transcript": request.get("transcript"),
        "transcript_source": request.get("transcript_source"),
    }
    return canonical_request_sha256(identity)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def append_launch_decision(
    path: Path,
    decision: Mapping[str, Any],
    *,
    attempts_path: Path,
) -> dict[str, Any]:
    """Append one pre-launch promotion record and fsync it."""

    required = {
        "schema_version",
        "manifest_version",
        "gate_selection_sha256",
        "condition_id",
        "decision",
        "rationale",
        "recorded_at_utc",
        "request_set_sha256",
    }
    missing = sorted(required - set(decision))
    if missing:
        raise ValueError("Launch decision is missing: " + ", ".join(missing))
    if decision.get("schema_version") != 1:
        raise ValueError("Unsupported launch-decision schema")
    if decision.get("decision") != "promote":
        raise ValueError("Only promoted conditions can receive launch authorization")
    for key in required - {"schema_version"}:
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError(f"Launch decision {key} cannot be empty")

    existing = _read_jsonl(path)
    condition_id = decision["condition_id"]
    if any(
        row.get("condition_id") == condition_id for row in _read_jsonl(attempts_path)
    ):
        raise ValueError("Promotion decision must be recorded before its first attempt")
    if any(row.get("condition_id") == condition_id for row in existing):
        raise ValueError(f"Launch decision already recorded for {condition_id}")
    digest = canonical_request_sha256(dict(decision))
    recorded = {**dict(decision), "launch_decision_sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(recorded, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return recorded


def validate_launch_decision(record: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that an append-only launch record still matches its checksum."""

    recorded_digest = record.get("launch_decision_sha256")
    if not isinstance(recorded_digest, str):
        raise ValueError("Launch decision checksum is missing")
    unsigned = {
        key: value for key, value in record.items() if key != "launch_decision_sha256"
    }
    if canonical_request_sha256(unsigned) != recorded_digest:
        raise ValueError("Launch decision checksum does not match its contents")
    return dict(record)


def append_attempt_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append complete attempt rows and fsync once; never rewrite the ledger."""

    if not rows:
        raise ValueError("Attempt append requires at least one row")
    required = {
        "run_id",
        "condition_id",
        "video_id",
        "status",
        "answer",
        "request",
    }
    encoded = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError("Attempt is missing: " + ", ".join(missing))
        request = row["request"]
        if not isinstance(request, Mapping):
            raise ValueError("Attempt request must be an object")
        for field in ("run_id", "condition_id", "video_id"):
            if row.get(field) != request.get(field):
                raise ValueError(f"Attempt/request identity mismatch for {field}")
        encoded.append(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in encoded))
        handle.flush()
        os.fsync(handle.fileno())


def validate_frozen_attempt(
    request: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    execution: FrozenExecution,
    launch_decision_sha256: str,
) -> bool:
    """Return true only for a scoreable exact-request pass with full provenance."""

    try:
        current_evidence = evidence_identity_sha256(request)
    except (OSError, TypeError, ValueError):
        return False
    return (
        request.get("condition_id") == execution.condition_id
        and attempt.get("run_id") == request.get("run_id")
        and attempt.get("condition_id") == request.get("condition_id")
        and attempt.get("video_id") == request.get("video_id")
        and attempt.get("status") == "passed"
        and isinstance(attempt.get("answer"), str)
        and bool(attempt["answer"].strip())
        and saved_request_matches(attempt, request)
        and attempt.get("launch_decision_sha256") == launch_decision_sha256
        and attempt.get("evidence_identity_sha256") == current_evidence
        and attempt.get("model_id") == execution.model_id
        and attempt.get("model_revision") == execution.model_revision
        and attempt.get("runtime_image") == execution.runtime_image
        and attempt.get("device") == execution.device
        and attempt.get("precision") == execution.precision
        and isinstance(attempt.get("elapsed_seconds"), (int, float))
        and attempt["elapsed_seconds"] >= 0
        and (
            attempt.get("peak_memory_bytes") is None
            or (
                isinstance(attempt.get("peak_memory_bytes"), int)
                and attempt["peak_memory_bytes"] >= 0
            )
        )
        and attempt.get("error_type") is None
        and attempt.get("error_message") is None
    )


def pending_frozen_requests(
    requests: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    execution: FrozenExecution,
    launch_decision_sha256: str,
) -> list[dict[str, Any]]:
    """Return requests without any prior exact, provenance-valid pass."""

    return [
        dict(request)
        for request in requests
        if not any(
            validate_frozen_attempt(
                request,
                attempt,
                execution=execution,
                launch_decision_sha256=launch_decision_sha256,
            )
            for attempt in attempts
        )
    ]


def select_complete_frozen_rows(
    requests: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    *,
    execution: FrozenExecution,
    launch_decision_sha256: str,
) -> list[dict[str, Any]]:
    """Select one valid pass per ordered request or fail before scoring."""

    selected = []
    for request in requests:
        matches = [
            dict(attempt)
            for attempt in attempts
            if validate_frozen_attempt(
                request,
                attempt,
                execution=execution,
                launch_decision_sha256=launch_decision_sha256,
            )
        ]
        if matches:
            selected.append(matches[-1])
    if len(selected) != EXPECTED_FROZEN_COUNT:
        raise ValueError(
            f"Frozen condition expected 30 valid rows, found {len(selected)}"
        )
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    return rows
