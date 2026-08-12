#!/usr/bin/env python3
"""Build private, blinded VQA frontier-map audit packets and freezes.

This module deliberately stops at the visual-audit boundary.  It never reads
annotations or reference answers, never calls providers, and never scores
predictions.  Its only job is to copy already-derived visual evidence into a
private packet, write blinded/question-aware reviewer manifests, and freeze the
two-view audit verdicts once every provider row has exactly one verdict in each
view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from trec_video_2026.vqa_benchmark import (
        bin_midpoint_timestamps,
        probe_media,
        source_fingerprint,
    )
    from trec_video_2026.vqa_frontier_map import (
        mask_question_fields,
        validate_event_map,
    )
    from trec_video_2026.vqa_frontier_map_panel import CANONICAL_DATA_ROOT
    from trec_video_2026.vqa_submission import canonical_sha256
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from trec_video_2026.vqa_benchmark import (
        bin_midpoint_timestamps,
        probe_media,
        source_fingerprint,
    )
    from trec_video_2026.vqa_frontier_map import (
        mask_question_fields,
        validate_event_map,
    )
    from trec_video_2026.vqa_frontier_map_panel import CANONICAL_DATA_ROOT
    from trec_video_2026.vqa_submission import canonical_sha256


ALLOWED_PROVIDER_AUDIO_STATES = frozenset(
    {"provider_reported", "not_claimed", "unverified"}
)
BLIND_REQUIRED_FIELDS = frozenset(
    {
        "event_coverage",
        "order",
        "entities",
        "actions",
        "state_changes",
        "visible_text",
        "timestamp_boundary_error",
        "answer_changing_false_claims",
        "notes",
    }
)
AWARE_REQUIRED_FIELDS = frozenset(
    {
        "question_understanding",
        "modality_choice",
        "evidence_presence",
        "evidence_localization",
        "support_citation",
        "usable_map_decision",
        "confidence",
        "notes",
    }
)
QUESTION_FIELD = "question"
QUESTION_AWARE_FIELDS = (
    "top1_answer",
    "support_span",
    "support_spans",
    "support_explanation",
    "explanation",
)
BLIND_MASKED_FIELDS = frozenset(
    {
        "question",
        "top1_answer",
        "answer",
        "support_span",
        "support_spans",
        "relevance",
        "could_be_wrong",
        "explanation",
        "reference",
        "reference_answer",
        "correct_answer",
    }
)
REFERENCE_FIELDS = frozenset(
    {
        "reference",
        "references",
        "reference_answer",
        "correct_answer",
        "incorrect_answers",
        "annotation",
        "annotations",
    }
)
OVERVIEW_FRAME_COUNT = 6


class AuditPacket:
    """Paths produced by :func:`build_audit_packet`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.blind_manifest = root / "blind-manifest.json"
        self.aware_manifest = root / "aware-manifest.json"
        self.packet_manifest = root / "packet-manifest.json"
        self.verdicts_blind = root / "verdicts.blind.jsonl"
        self.verdicts_aware = root / "verdicts.aware.jsonl"
        self.freeze_json = root / "freeze.json"


def build_audit_packet(
    frozen_run: Mapping[str, Any], *, output: str | Path
) -> AuditPacket:
    """Create private visual-audit manifests from frozen provider rows.

    ``frozen_run`` is intentionally a plain mapping so tests and upstream
    freezing code can pass already-frozen JSON without importing a provider
    runtime.  Required media paths must be normal files, must not be symlinks,
    and must match the supplied SHA-256 identities before they are copied.
    """

    root = _prepare_output_root(Path(output))
    rows = _provider_rows(frozen_run)
    packet = AuditPacket(root)
    manifest_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    aware_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        provider = _required_text(row, "provider")
        video_id = _required_text(row, "video_id")
        row_dir = root / "rows" / _safe_segment(provider) / _safe_segment(video_id)
        keyframe_dir = row_dir / "event-keyframes"
        keyframe_dir.mkdir(parents=True, exist_ok=True)

        map_payload = row.get("map")
        failure_code = row.get("failure_code")
        map_payload_present = isinstance(map_payload, Mapping)
        if failure_code is not None and not isinstance(failure_code, str):
            raise ValueError("failure_code must be text when present")
        if not map_payload_present and (
            not isinstance(failure_code, str) or not failure_code.strip()
        ):
            raise ValueError("absent map payload requires a frozen failure_code")
        if failure_code and map_payload_present:
            raise ValueError("failed provider row cannot include map content")

        overview_dest, keyframe_entries, overview_timestamps = _materialize_audit_media(
            row,
            map_payload=map_payload if map_payload_present else None,
            row_dir=row_dir,
            keyframe_dir=keyframe_dir,
            root=root,
        )

        common = _common_manifest_fields(
            row,
            index=index,
            overview=overview_dest,
            keyframes=keyframe_entries,
            root=root,
            map_payload_present=map_payload_present,
            failure_code=failure_code,
            overview_timestamps=overview_timestamps,
        )
        blind = _blind_manifest(common, map_payload)
        aware = _aware_manifest(common, row, map_payload)
        _reject_reference_fields(blind)
        _reject_reference_fields(aware)
        _reject_blind_leakage(blind)

        _write_json(row_dir / "blind.json", blind)
        _write_json(row_dir / "aware.json", aware)

        manifest_rows.append(
            {
                "provider": provider,
                "video_id": video_id,
                "row_dir": str(row_dir.relative_to(root)),
                "blind_manifest": str((row_dir / "blind.json").relative_to(root)),
                "aware_manifest": str((row_dir / "aware.json").relative_to(root)),
                "overview": str(overview_dest.relative_to(root)),
                "event_keyframes": [entry["path"] for entry in keyframe_entries],
                "source_hashes": _ordered_source_hashes(row),
                "map_payload_present": map_payload_present,
                "failure_code": failure_code,
            }
        )
        blind_rows.append(blind)
        aware_rows.append(aware)

    packet_manifest = {
        "schema_version": 1,
        "boundary": "private-vqa-frontier-map-visual-audit",
        "reference_policy": "not-read-not-serialized",
        "provider_call_policy": "not-called",
        "row_count": len(manifest_rows),
        "panel_manifest": frozen_run.get("panel_manifest", {}),
        "rows": manifest_rows,
    }
    _write_json(packet.packet_manifest, packet_manifest)
    _write_json(packet.blind_manifest, {"schema_version": 1, "rows": blind_rows})
    _write_json(packet.aware_manifest, {"schema_version": 1, "rows": aware_rows})
    packet.verdicts_blind.touch(exist_ok=True)
    packet.verdicts_aware.touch(exist_ok=True)
    return packet


def freeze_audit(audit: str | Path) -> dict[str, Any]:
    """Validate verdict completeness and persist an immutable audit freeze."""

    root = Path(audit)
    _validate_audit_root(root)
    freeze_path = root / "freeze.json"
    if freeze_path.exists() or freeze_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing freeze: {freeze_path}")
    packet_manifest_path = root / "packet-manifest.json"
    packet_manifest = _read_json(packet_manifest_path)
    packet_rows = _sequence(packet_manifest.get("rows"), "packet-manifest rows")
    blind_verdicts = _read_verdicts(root / "verdicts.blind.jsonl", "blind")
    aware_verdicts = _read_verdicts(root / "verdicts.aware.jsonl", "aware")

    expected_keys: list[tuple[str, str]] = []
    freeze_rows: list[dict[str, Any]] = []
    for row in packet_rows:
        if not isinstance(row, Mapping):
            raise TypeError("packet-manifest rows must be objects")
        key = (_required_text(row, "provider"), _required_text(row, "video_id"))
        expected_keys.append(key)
        if key not in blind_verdicts or key not in aware_verdicts:
            raise ValueError(f"missing verdict for {key[0]}/{key[1]}")
        aware = aware_verdicts[key]
        if row.get("failure_code") and aware.get("usable_map_decision") != "unusable":
            raise ValueError("failed provider row requires an unusable aware verdict")
        artifact_hashes = _row_artifact_hashes(root, row)
        freeze_rows.append(
            {
                "provider": key[0],
                "video_id": key[1],
                "map_payload_present": bool(row.get("map_payload_present")),
                "failure_code": row.get("failure_code"),
                "packet_row_sha256": _sha256_json(row),
                **artifact_hashes,
                "blind_verdict_sha256": _sha256_json(blind_verdicts[key]),
                "aware_verdict_sha256": _sha256_json(aware),
            }
        )

    extra_blind = sorted(set(blind_verdicts) - set(expected_keys))
    extra_aware = sorted(set(aware_verdicts) - set(expected_keys))
    if extra_blind or extra_aware:
        raise ValueError("verdict contains row not present in packet manifest")

    freeze = {
        "schema_version": 1,
        "frozen": True,
        "boundary": "private-vqa-frontier-map-visual-audit",
        "reference_policy": "not-read-not-serialized",
        "provider_call_policy": "not-called",
        "bindings": {
            "provider_rows": len(expected_keys),
            "blind_verdicts": len(blind_verdicts),
            "aware_verdicts": len(aware_verdicts),
            "packet_manifest_sha256": _sha256_path(packet_manifest_path),
            "blind_manifest_sha256": _sha256_path(root / "blind-manifest.json"),
            "aware_manifest_sha256": _sha256_path(root / "aware-manifest.json"),
            "blind_verdicts_sha256": _sha256_path(root / "verdicts.blind.jsonl"),
            "aware_verdicts_sha256": _sha256_path(root / "verdicts.aware.jsonl"),
        },
        "rows": freeze_rows,
    }
    _write_json(freeze_path, freeze)
    return freeze


def verify_audit_freeze(audit: str | Path) -> dict[str, Any]:
    """Verify that every audit artifact still matches the persisted freeze."""

    root = Path(audit)
    _validate_audit_root(root)
    freeze = _read_json(root / "freeze.json")
    if freeze.get("frozen") is not True:
        raise ValueError("audit freeze is not frozen")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("audit freeze bindings must be an object")

    packet_manifest_path = root / "packet-manifest.json"
    packet_manifest = _read_bound_json(
        packet_manifest_path,
        bindings.get("packet_manifest_sha256"),
        "packet manifest",
        validate=lambda value: _validate_packet_manifest_paths(root, value),
    )
    _require_hash(
        _sha256_path(root / "blind-manifest.json"),
        bindings.get("blind_manifest_sha256"),
        "blind manifest",
    )
    _require_hash(
        _sha256_path(root / "aware-manifest.json"),
        bindings.get("aware_manifest_sha256"),
        "aware manifest",
    )
    _require_hash(
        _sha256_path(root / "verdicts.blind.jsonl"),
        bindings.get("blind_verdicts_sha256"),
        "blind verdicts",
    )
    _require_hash(
        _sha256_path(root / "verdicts.aware.jsonl"),
        bindings.get("aware_verdicts_sha256"),
        "aware verdicts",
    )

    packet_rows = _sequence(packet_manifest.get("rows"), "packet-manifest rows")
    freeze_rows = _sequence(freeze.get("rows"), "freeze rows")
    if len(packet_rows) != len(freeze_rows):
        raise ValueError("audit freeze row count hash mismatch")

    blind_verdicts = _read_verdicts(root / "verdicts.blind.jsonl", "blind")
    aware_verdicts = _read_verdicts(root / "verdicts.aware.jsonl", "aware")
    for packet_row, freeze_row in zip(packet_rows, freeze_rows, strict=True):
        if not isinstance(packet_row, Mapping) or not isinstance(freeze_row, Mapping):
            raise TypeError("audit freeze rows must be objects")
        key = (
            _required_text(packet_row, "provider"),
            _required_text(packet_row, "video_id"),
        )
        if (
            freeze_row.get("provider") != key[0]
            or freeze_row.get("video_id") != key[1]
        ):
            raise ValueError("audit freeze row identity hash mismatch")
        _require_hash(
            _sha256_json(packet_row),
            freeze_row.get("packet_row_sha256"),
            f"{key[0]}/{key[1]} packet row",
        )
        for field, digest in _row_artifact_hashes(root, packet_row).items():
            _require_hash(
                digest,
                freeze_row.get(field),
                f"{key[0]}/{key[1]} {field}",
            )
        _require_hash(
            _sha256_json(blind_verdicts[key]),
            freeze_row.get("blind_verdict_sha256"),
            f"{key[0]}/{key[1]} blind verdict",
        )
        _require_hash(
            _sha256_json(aware_verdicts[key]),
            freeze_row.get("aware_verdict_sha256"),
            f"{key[0]}/{key[1]} aware verdict",
        )
    return freeze


def validate_audit_packet(audit: str | Path) -> dict[str, Any]:
    """Validate a built audit packet, and verify the freeze when present."""

    root = Path(audit)
    _validate_audit_root(root)
    if (root / "freeze.json").exists():
        freeze = verify_audit_freeze(root)
        return {
            "status": "valid",
            "row_count": len(_sequence(freeze.get("rows"), "freeze rows")),
            "frozen": True,
        }

    packet_manifest = _read_json(root / "packet-manifest.json")
    _validate_packet_manifest_paths(root, packet_manifest)
    rows = _sequence(packet_manifest.get("rows"), "packet-manifest rows")
    blind = _read_json(root / "blind-manifest.json")
    aware = _read_json(root / "aware-manifest.json")
    blind_rows = _sequence(blind.get("rows"), "blind-manifest rows")
    aware_rows = _sequence(aware.get("rows"), "aware-manifest rows")
    if len(rows) != len(blind_rows) or len(rows) != len(aware_rows):
        raise ValueError("audit manifest row count mismatch")
    for row in blind_rows:
        _reject_blind_leakage(row)
        _reject_reference_fields(row)
    for row in aware_rows:
        _reject_reference_fields(row)
    return {"status": "valid", "row_count": len(rows), "frozen": False}


def _provider_rows(frozen_run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = frozen_run.get("provider_results")
    if not isinstance(rows, list):
        raise TypeError("frozen_run provider_results must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"provider row {index} must be an object")
    return rows


def _common_manifest_fields(
    row: Mapping[str, Any],
    *,
    index: int,
    overview: Path,
    keyframes: list[Mapping[str, Any]],
    root: Path,
    map_payload_present: bool,
    failure_code: Any,
    overview_timestamps: list[int | float],
) -> dict[str, Any]:
    provider_audio_state = row.get("provider_audio_state", "unverified")
    if provider_audio_state not in ALLOWED_PROVIDER_AUDIO_STATES:
        raise ValueError(
            "provider_audio_state must be one of: "
            + ", ".join(sorted(ALLOWED_PROVIDER_AUDIO_STATES))
        )
    events = _map_events(row.get("map")) if map_payload_present else []
    return {
        "schema_version": 1,
        "row_index": index,
        "provider": _required_text(row, "provider"),
        "video_id": _required_text(row, "video_id"),
        "panel_row_id": row.get("panel_row_id"),
        "category": row.get("category"),
        "provider_audio_state": provider_audio_state,
        "map_payload_present": map_payload_present,
        "failure_code": failure_code,
        "overview_path": str(overview.relative_to(root)),
        "overview_sha256": _sha256_path(overview),
        "overview_timestamps": overview_timestamps,
        "event_keyframes": list(keyframes),
        "event_timestamps": [
            frame["requested_timestamp_seconds"]
            for frame in keyframes
            if "requested_timestamp_seconds" in frame
        ]
        or _event_timestamps(events),
    }


def _blind_manifest(
    common: Mapping[str, Any], map_payload: Any
) -> dict[str, Any]:
    blind = dict(common)
    if isinstance(map_payload, Mapping):
        blind["map"] = mask_question_fields(map_payload)
    return _drop_empty(blind)


def _aware_manifest(
    common: Mapping[str, Any], row: Mapping[str, Any], map_payload: Any
) -> dict[str, Any]:
    aware = dict(common)
    if QUESTION_FIELD in row and row[QUESTION_FIELD] is not None:
        aware[QUESTION_FIELD] = row[QUESTION_FIELD]
    if isinstance(map_payload, Mapping):
        for field in QUESTION_AWARE_FIELDS:
            if field in row and row[field] is not None:
                aware[field] = row[field]
        aware["map"] = _strip_reference_fields(map_payload)
        if "explanation" in map_payload and "explanation" not in aware:
            aware["explanation"] = map_payload["explanation"]
    return _drop_empty(aware)


def _strip_reference_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_reference_fields(nested)
            for key, nested in value.items()
            if str(key) not in REFERENCE_FIELDS
        }
    if isinstance(value, list):
        return [_strip_reference_fields(item) for item in value]
    return value


def _reject_reference_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in REFERENCE_FIELDS:
                raise ValueError(f"reference field crossed audit boundary: {key}")
            _reject_reference_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_reference_fields(nested)


def _reject_blind_leakage(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in BLIND_MASKED_FIELDS:
                raise ValueError(f"blind manifest contains masked field: {key}")
            _reject_blind_leakage(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_blind_leakage(nested)


def _map_events(map_payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(map_payload, Mapping):
        return []
    events = map_payload.get("events", [])
    if events is None:
        return []
    if not isinstance(events, list):
        raise TypeError("map events must be a list")
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("map events must be objects")
    return events


def _event_timestamps(events: Sequence[Mapping[str, Any]]) -> list[int | float]:
    timestamps: list[int | float] = []
    for event in events:
        keyframe_seconds = event.get("keyframe_seconds")
        if isinstance(keyframe_seconds, list):
            timestamps.extend(
                timestamp
                for timestamp in keyframe_seconds
                if isinstance(timestamp, int | float) and not isinstance(timestamp, bool)
            )
            continue
        timestamp = event.get("timestamp")
        if isinstance(timestamp, int | float) and not isinstance(timestamp, bool):
            timestamps.append(timestamp)
    return timestamps


def _source_hashes(row: Mapping[str, Any]) -> Mapping[str, str]:
    source_hashes = row.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise TypeError("source_hashes must be an object")
    for key, value in source_hashes.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("source_hashes must map text paths to SHA-256 text")
    return source_hashes


def _ordered_source_hashes(row: Mapping[str, Any]) -> dict[str, str]:
    hashes = _source_hashes(row)
    return {key: hashes[key] for key in sorted(hashes)}


def _validated_source_path(value: Any, source_hashes: Mapping[str, str]) -> Path:
    if not isinstance(value, str):
        raise TypeError("source media path must be text")
    path = Path(value)
    if _contains_symlink(path):
        raise ValueError(f"source media path cannot be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"source media path is not a file: {path}")
    expected = source_hashes.get(str(path))
    if expected is None:
        raise ValueError(f"source hash missing for {path}")
    actual = _sha256_path(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {path}")
    return path


def _prepare_output_root(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError(f"audit output root cannot be a symlink: {root}")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"audit output root is not a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(f"refusing to overwrite nonempty audit output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_audit_root(root: Path) -> None:
    if root.is_symlink():
        raise ValueError(f"audit root cannot be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"audit root is not a directory: {root}")


def _contains_symlink(path: Path) -> bool:
    candidates = [path, *path.parents]
    existing = [candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()]
    return any(candidate.is_symlink() for candidate in existing)


def _read_verdicts(path: Path, view: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    verdicts: dict[tuple[str, str], Mapping[str, Any]] = {}
    required = BLIND_REQUIRED_FIELDS if view == "blind" else AWARE_REQUIRED_FIELDS
    if not path.exists():
        raise ValueError(f"{view} verdict file is missing")
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        verdict = json.loads(line)
        if not isinstance(verdict, Mapping):
            raise TypeError(f"{view} verdict line {line_number} must be an object")
        if verdict.get("view") != view:
            raise ValueError(f"{view} verdict line {line_number} has wrong view")
        missing = sorted(required - set(verdict))
        if missing:
            raise ValueError(
                f"{view} verdict line {line_number} missing fields: "
                + ", ".join(missing)
            )
        provider = _required_text(verdict, "provider")
        video_id = _required_text(verdict, "video_id")
        key = (provider, video_id)
        if key in verdicts:
            raise ValueError(f"duplicate {view} verdict for {provider}/{video_id}")
        verdicts[key] = verdict
    return verdicts


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def _safe_segment(value: str) -> str:
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"path segment is unsafe: {value}")
    return value


def _sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _drop_empty(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative_path(root: Path, relative_path: Any, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{field} must be a nonempty relative path")
    parsed = Path(relative_path)
    if parsed.is_absolute():
        raise ValueError(f"{field} path escape: {relative_path}")
    candidate = root / parsed
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    if (
        candidate_resolved != root_resolved
        and root_resolved not in candidate_resolved.parents
    ):
        raise ValueError(f"{field} path escape: {relative_path}")
    if _contains_symlink(candidate):
        raise ValueError(f"{field} path cannot be a symlink: {relative_path}")
    if not candidate.is_file():
        raise ValueError(f"{field} path is missing: {relative_path}")
    return candidate


def _row_artifact_hashes(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    blind_path = _safe_relative_path(root, row.get("blind_manifest"), "blind_manifest")
    aware_path = _safe_relative_path(root, row.get("aware_manifest"), "aware_manifest")
    overview_path = _safe_relative_path(root, row.get("overview"), "overview")
    keyframes = []
    for index, relative_path in enumerate(
        _sequence(row.get("event_keyframes"), "event_keyframes")
    ):
        path = _safe_relative_path(root, relative_path, f"event_keyframes[{index}]")
        keyframes.append({"path": relative_path, "sha256": _sha256_path(path)})
    return {
        "blind_manifest_sha256": _sha256_path(blind_path),
        "aware_manifest_sha256": _sha256_path(aware_path),
        "overview_sha256": _sha256_path(overview_path),
        "event_keyframes_sha256": keyframes,
    }


def _validate_packet_manifest_paths(root: Path, packet_manifest: Mapping[str, Any]) -> None:
    rows = _sequence(packet_manifest.get("rows"), "packet-manifest rows")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"packet-manifest row {index} must be an object")
        _safe_relative_path(root, row.get("blind_manifest"), "blind_manifest")
        _safe_relative_path(root, row.get("aware_manifest"), "aware_manifest")
        _safe_relative_path(root, row.get("overview"), "overview")
        for frame_index, relative_path in enumerate(
            _sequence(row.get("event_keyframes"), "event_keyframes")
        ):
            _safe_relative_path(
                root, relative_path, f"event_keyframes[{frame_index}]"
            )


def _require_hash(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} hash mismatch")


def _read_bound_json(
    path: Path,
    expected_sha256: Any,
    label: str,
    *,
    validate: Any | None = None,
) -> dict[str, Any]:
    actual = _sha256_path(path)
    try:
        value = _read_json(path)
        if validate is not None:
            validate(value)
    except Exception as error:
        if actual != expected_sha256:
            if isinstance(error, ValueError) and "path escape" in str(error):
                raise
            raise ValueError(f"{label} hash mismatch") from error
        raise
    _require_hash(actual, expected_sha256, label)
    return value


def _load_prepared_frozen_run(prepared: str | Path) -> dict[str, Any]:
    root = Path(prepared)
    if _contains_symlink(root):
        raise ValueError(f"prepared run path cannot be a symlink: {root}")
    if not root.is_dir():
        raise ValueError(f"prepared run path is not a directory: {root}")
    artifact_path = root / "provider-results.json"
    freeze_path = root / "freeze.json"
    if _contains_symlink(artifact_path) or _contains_symlink(freeze_path):
        raise ValueError("prepared provider freeze paths cannot be symlinks")
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("provider freeze status must be frozen")
    expected_artifact_sha256 = freeze.get("artifact_sha256")
    _require_hash(
        _sha256_path(artifact_path),
        expected_artifact_sha256,
        "provider freeze artifact",
    )
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("provider freeze bindings must be an object")
    _require_hash(
        _sha256_json(bindings),
        freeze.get("bindings_sha256"),
        "provider freeze bindings",
    )
    artifact = _read_json(artifact_path)
    for key, expected in bindings.items():
        if artifact.get(key) != expected:
            raise ValueError(f"provider freeze binding mismatch: {key}")
    rows = artifact.get("provider_results")
    if not isinstance(rows, list):
        return _reconstruct_prepared_provider_rows(root, artifact)
    return {
        "panel_manifest": artifact.get("panel_manifest", {}),
        "provider_results": rows,
    }


def _reconstruct_prepared_provider_rows(
    root: Path, artifact: Mapping[str, Any]
) -> dict[str, Any]:
    panel_path = _safe_relative_path(root, "panel.json", "prepared panel")
    panel = _read_json(panel_path)
    _require_hash(
        canonical_sha256(panel), artifact.get("panel_sha256"), "prepared panel"
    )
    examples = _sequence(panel.get("examples"), "prepared panel examples")
    evaluation_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(examples):
        if not isinstance(row, Mapping):
            raise TypeError(f"prepared panel example {index} must be an object")
        if row.get("phase") != "evaluation":
            continue
        video_id = _required_text(row, "video_id")
        if video_id in evaluation_rows:
            raise ValueError(f"duplicate prepared panel video_id: {video_id}")
        evaluation_rows[video_id] = row

    provider_summaries = artifact.get("providers")
    if not isinstance(provider_summaries, Mapping) or not provider_summaries:
        raise ValueError("provider freeze artifact providers must be an object")
    frozen_results: dict[str, list[Mapping[str, Any]]] = {}
    for provider in sorted(provider_summaries):
        _safe_segment(str(provider))
        summary = provider_summaries[provider]
        if not isinstance(summary, Mapping):
            raise TypeError(f"provider summary must be an object: {provider}")
        result_path = _safe_relative_path(
            root, f"results/{provider}.jsonl", f"{provider} results"
        )
        panel_results = [
            row
            for row in _read_jsonl(result_path, f"{provider} results")
            if row.get("phase") == "panel"
        ]
        if len(panel_results) != summary.get("panel_results"):
            raise ValueError(f"{provider} result count mismatch")
        _require_hash(
            canonical_sha256(panel_results),
            summary.get("results_sha256"),
            f"{provider} results",
        )
        frozen_results[str(provider)] = panel_results
    _require_hash(
        canonical_sha256(frozen_results),
        artifact.get("results_sha256"),
        "provider results",
    )
    if sum(map(len, frozen_results.values())) != artifact.get(
        "provider_result_count"
    ):
        raise ValueError("provider result count mismatch")

    reconstructed = [
        _reconstruct_provider_row(root, artifact, provider, result, evaluation_rows)
        for provider, results in frozen_results.items()
        for result in results
    ]
    expected = {
        (provider, video_id)
        for provider in frozen_results
        for video_id in evaluation_rows
    }
    actual = {(row["provider"], row["video_id"]) for row in reconstructed}
    if actual != expected or len(reconstructed) != len(expected):
        raise ValueError("provider result coverage mismatch")
    return {
        "panel_manifest": {
            "run_id": panel.get("run_id"),
            "panel_sha256": artifact.get("panel_sha256"),
            "evaluation_count": len(evaluation_rows),
        },
        "provider_results": reconstructed,
    }


def _reconstruct_provider_row(
    root: Path,
    artifact: Mapping[str, Any],
    provider: str,
    result: Mapping[str, Any],
    evaluation_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    video_id = _required_text(result, "video_id")
    panel_row = evaluation_rows.get(video_id)
    if panel_row is None:
        raise ValueError(f"provider result is outside evaluation panel: {video_id}")
    if result.get("provider") != provider:
        raise ValueError(f"provider result identity mismatch: {provider}/{video_id}")
    for field in ("run_id", "panel_sha256", "schema_sha256"):
        if result.get(field) != artifact.get(field):
            raise ValueError(f"provider result {field} mismatch: {provider}/{video_id}")
    for result_field, panel_field in (
        ("source_sha256", "media_sha256"),
        ("source_size_bytes", "size_bytes"),
        ("category", "category"),
        ("duration_seconds", "duration_seconds"),
    ):
        if result.get(result_field) != panel_row.get(panel_field):
            raise ValueError(
                f"provider result {result_field} mismatch: {provider}/{video_id}"
            )

    status = result.get("status")
    if status not in {"succeeded", "failed"}:
        raise ValueError(f"provider result status is not terminal: {provider}/{video_id}")
    failure_code = result.get("failure_code") if status == "failed" else None
    if status == "failed" and (
        not isinstance(failure_code, str) or not failure_code.strip()
    ):
        raise ValueError(f"failed provider result has no failure_code: {provider}/{video_id}")
    map_payload = _verified_raw_map(
        root,
        provider=provider,
        video_id=video_id,
        result=result,
        duration_seconds=panel_row.get("duration_seconds"),
        parse=status == "succeeded",
    )
    reconstructed = {
        "provider": provider,
        "video_id": video_id,
        "panel_row_id": panel_row.get("ordinal"),
        "category": panel_row.get("category"),
        "question": panel_row.get("question"),
        "provider_audio_state": "unverified",
        "map": map_payload,
        "failure_code": failure_code,
    }
    if all(
        field in panel_row
        for field in ("overview_path", "keyframe_paths", "source_hashes")
    ):
        overview, keyframes, source_hashes = _prepared_audit_media(root, panel_row)
        reconstructed.update(
            {
                "overview_path": str(overview),
                "keyframe_paths": [str(path) for path in keyframes],
                "source_hashes": source_hashes,
            }
        )
        return reconstructed

    canonical_source = _resolve_canonical_source(panel_row)
    reconstructed.update(
        {
            "_canonical_source": canonical_source,
            "source_hashes": {
                str(canonical_source["relative_path"]): str(
                    canonical_source["sha256"]
                )
            },
        }
    )
    return reconstructed


def _verified_raw_map(
    root: Path,
    *,
    provider: str,
    video_id: str,
    result: Mapping[str, Any],
    duration_seconds: Any,
    parse: bool,
) -> Mapping[str, Any] | None:
    response_sha256 = result.get("response_sha256")
    if response_sha256 is None:
        if parse:
            raise ValueError(f"successful provider result has no response hash: {provider}/{video_id}")
        return None
    raw_path = _safe_relative_path(
        root, f"raw/{provider}/{video_id}.json", f"{provider}/{video_id} raw response"
    )
    raw = _read_json(raw_path)
    for field, expected in (
        ("run_id", result.get("run_id")),
        ("provider", provider),
        ("video_id", video_id),
    ):
        if raw.get(field) != expected:
            raise ValueError(f"raw response {field} mismatch: {provider}/{video_id}")
    raw_response = raw.get("raw_response")
    if not isinstance(raw_response, str):
        raise TypeError(f"raw response must be text: {provider}/{video_id}")
    _require_hash(
        hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
        response_sha256,
        f"{provider}/{video_id} raw response",
    )
    _require_hash(
        canonical_sha256(raw_response),
        raw.get("raw_response_sha256"),
        f"{provider}/{video_id} raw response record",
    )
    if not parse:
        return None
    payload = json.loads(raw_response)
    if not isinstance(payload, Mapping):
        raise TypeError(f"provider map must be an object: {provider}/{video_id}")
    validate_event_map(payload, duration_seconds)
    return payload


def _prepared_audit_media(
    root: Path, panel_row: Mapping[str, Any]
) -> tuple[Path, list[Path], dict[str, str]]:
    hashes = _source_hashes(panel_row)
    values = [
        panel_row.get("overview_path"),
        *_sequence(panel_row.get("keyframe_paths"), "keyframe_paths"),
    ]
    resolved: list[Path] = []
    normalized_hashes: dict[str, str] = {}
    for index, value in enumerate(values):
        field = "overview_path" if index == 0 else f"keyframe_paths[{index - 1}]"
        path = _safe_relative_path(root, value, field)
        expected = hashes.get(value)
        _require_hash(_sha256_path(path), expected, field)
        resolved.append(path)
        normalized_hashes[str(path)] = str(expected)
    return resolved[0], resolved[1:], normalized_hashes


def _canonical_data_root() -> Path:
    return CANONICAL_DATA_ROOT.resolve(strict=True)


def _resolve_canonical_source(panel_row: Mapping[str, Any]) -> dict[str, Any]:
    relative_path = _required_text(panel_row, "media_relative_path")
    parsed = Path(relative_path)
    if parsed.is_absolute():
        raise ValueError(f"source media path escape: {relative_path}")
    root = _canonical_data_root()
    if _contains_symlink(root):
        raise ValueError(f"canonical dataset root cannot be a symlink: {root}")
    candidate = root / parsed
    if _contains_symlink(candidate):
        raise ValueError(f"source media path cannot be a symlink: {relative_path}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"source media path is missing: {relative_path}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"source media path escape: {relative_path}")
    source = {
        "path": str(resolved),
        "relative_path": relative_path,
        "sha256": panel_row.get("media_sha256"),
        "size_bytes": panel_row.get("size_bytes"),
        "duration_seconds": panel_row.get("duration_seconds"),
        "has_audio": panel_row.get("has_audio"),
        "has_video": panel_row.get("has_video"),
    }
    _verify_canonical_source(source)
    return source


def _verify_canonical_source(source: Mapping[str, Any]) -> Path:
    path_value = source.get("path")
    if not isinstance(path_value, str):
        raise TypeError("canonical source path must be text")
    path = Path(path_value)
    root = _canonical_data_root()
    if _contains_symlink(path):
        raise ValueError(f"source media path cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("source media path must stay beneath canonical dataset root")
    identity = source_fingerprint(resolved)
    if (
        identity.sha256 != source.get("sha256")
        or identity.size_bytes != source.get("size_bytes")
    ):
        raise ValueError("source media bytes do not match frozen panel identity")
    metadata = probe_media(resolved)
    duration = source.get("duration_seconds")
    if not isinstance(duration, int | float) or isinstance(duration, bool):
        raise TypeError("frozen source duration_seconds must be numeric")
    if (
        not math.isclose(
            metadata.duration_seconds,
            float(duration),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or metadata.has_audio != source.get("has_audio")
        or metadata.has_video != source.get("has_video")
    ):
        raise ValueError("source media metadata does not match frozen panel identity")
    return resolved


def _materialize_audit_media(
    row: Mapping[str, Any],
    *,
    map_payload: Mapping[str, Any] | None,
    row_dir: Path,
    keyframe_dir: Path,
    root: Path,
) -> tuple[Path, list[dict[str, Any]], list[int | float]]:
    canonical_source = row.get("_canonical_source")
    if isinstance(canonical_source, Mapping):
        return _generate_audit_media(
            canonical_source,
            map_payload=map_payload,
            row_dir=row_dir,
            keyframe_dir=keyframe_dir,
            root=root,
        )

    source_hashes = _source_hashes(row)
    overview_source = _validated_source_path(row.get("overview_path"), source_hashes)
    overview = row_dir / "overview.jpg"
    shutil.copyfile(overview_source, overview)
    timestamps = _event_timestamps(_map_events(map_payload)) if map_payload else []
    keyframe_sources = (
        [
            _validated_source_path(value, source_hashes)
            for value in _sequence(row.get("keyframe_paths"), "keyframe_paths")
        ]
        if map_payload is not None
        else []
    )
    keyframes: list[dict[str, Any]] = []
    for frame_index, keyframe_source in enumerate(keyframe_sources, start=1):
        destination = keyframe_dir / keyframe_source.name
        if destination.exists():
            destination = keyframe_dir / f"{frame_index:04d}{keyframe_source.suffix}"
        shutil.copyfile(keyframe_source, destination)
        entry: dict[str, Any] = {
            "path": str(destination.relative_to(root)),
            "sha256": _sha256_path(destination),
            "source_sha256": _sha256_path(keyframe_source),
        }
        if frame_index <= len(timestamps):
            entry["requested_timestamp_seconds"] = timestamps[frame_index - 1]
        keyframes.append(entry)
    return overview, keyframes, []


def _generate_audit_media(
    source: Mapping[str, Any],
    *,
    map_payload: Mapping[str, Any] | None,
    row_dir: Path,
    keyframe_dir: Path,
    root: Path,
) -> tuple[Path, list[dict[str, Any]], list[int | float]]:
    media = _verify_canonical_source(source)
    duration = float(source["duration_seconds"])
    overview_timestamps = list(
        bin_midpoint_timestamps(duration, count=OVERVIEW_FRAME_COUNT)
    )
    overview = row_dir / "overview.jpg"
    with tempfile.TemporaryDirectory(prefix=".overview-", dir=row_dir) as temporary:
        temporary_root = Path(temporary)
        for index, timestamp in enumerate(overview_timestamps, start=1):
            _extract_jpeg(
                media,
                timestamp,
                temporary_root / f"frame-{index:04d}.jpg",
            )
        _build_contact_sheet(temporary_root, overview)

    keyframes: list[dict[str, Any]] = []
    timestamps = _event_timestamps(_map_events(map_payload)) if map_payload else []
    for index, timestamp in enumerate(timestamps, start=1):
        destination = keyframe_dir / f"{index:04d}.jpg"
        _extract_jpeg(media, float(timestamp), destination)
        keyframes.append(
            {
                "path": str(destination.relative_to(root)),
                "requested_timestamp_seconds": timestamp,
                "sha256": _sha256_path(destination),
                "source_media_sha256": source["sha256"],
            }
        )
    _verify_canonical_source(source)
    return overview, keyframes, overview_timestamps


def _extract_jpeg(media: Path, timestamp: float, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build audit media")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(media),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    _validate_generated_jpeg(output)


def _build_contact_sheet(frame_dir: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build audit media")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-framerate",
            "1",
            "-start_number",
            "1",
            "-i",
            str(frame_dir / "frame-%04d.jpg"),
            "-vf",
            "scale=320:-2,tile=3x2:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    _validate_generated_jpeg(output)


def _validate_generated_jpeg(path: Path) -> None:
    if _contains_symlink(path) or not path.is_file():
        raise ValueError(f"ffmpeg did not create a regular audit JPEG: {path}")
    payload = path.read_bytes()
    if len(payload) < 4 or not payload.startswith(b"\xff\xd8") or not payload.endswith(
        b"\xff\xd9"
    ):
        raise ValueError(f"ffmpeg output is not a complete JPEG: {path}")


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def _write_stdout_json(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build")
    build.add_argument("--prepared", required=True)
    build.add_argument("--output", required=True)

    validate = subcommands.add_parser("validate")
    validate.add_argument("--audit", required=True)

    freeze = subcommands.add_parser("freeze")
    freeze.add_argument("--audit", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            frozen_run = _load_prepared_frozen_run(args.prepared)
            packet = build_audit_packet(frozen_run, output=args.output)
            packet_manifest = _read_json(packet.packet_manifest)
            _write_stdout_json(
                {
                    "status": "built",
                    "row_count": packet_manifest["row_count"],
                    "audit": str(packet.root),
                }
            )
            return 0
        if args.command == "validate":
            _write_stdout_json(validate_audit_packet(args.audit))
            return 0
        if args.command == "freeze":
            _write_stdout_json(freeze_audit(args.audit))
            return 0
    except Exception as error:
        _write_stdout_json({"status": "error", "error": str(error)})
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
