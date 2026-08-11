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

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence


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
QUESTION_AWARE_FIELDS = (
    "question",
    "top1_answer",
    "support_span",
    "support_spans",
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

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
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

        source_hashes = _source_hashes(row)
        overview_source = _validated_source_path(row.get("overview_path"), source_hashes)
        overview_dest = row_dir / "overview.jpg"
        shutil.copyfile(overview_source, overview_dest)

        keyframe_sources = [
            _validated_source_path(value, source_hashes)
            for value in _sequence(row.get("keyframe_paths"), "keyframe_paths")
        ]
        keyframe_entries = []
        for frame_index, keyframe_source in enumerate(keyframe_sources, start=1):
            destination = keyframe_dir / keyframe_source.name
            if destination.exists():
                destination = keyframe_dir / f"{frame_index:04d}{keyframe_source.suffix}"
            shutil.copyfile(keyframe_source, destination)
            keyframe_entries.append(
                {
                    "path": str(destination.relative_to(root)),
                    "sha256": _sha256_path(destination),
                    "source_sha256": _sha256_path(keyframe_source),
                }
            )

        map_payload = row.get("map")
        failure_code = row.get("failure_code")
        map_payload_present = isinstance(map_payload, Mapping)
        if failure_code is not None and not isinstance(failure_code, str):
            raise ValueError("failure_code must be text when present")
        if failure_code and map_payload_present:
            raise ValueError("failed provider row cannot include map content")

        common = _common_manifest_fields(
            row,
            index=index,
            overview=overview_dest,
            keyframes=keyframe_entries,
            root=root,
            map_payload_present=map_payload_present,
            failure_code=failure_code,
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
        freeze_rows.append(
            {
                "provider": key[0],
                "video_id": key[1],
                "map_payload_present": bool(row.get("map_payload_present")),
                "failure_code": row.get("failure_code"),
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
            "blind_verdicts_sha256": _sha256_path(root / "verdicts.blind.jsonl"),
            "aware_verdicts_sha256": _sha256_path(root / "verdicts.aware.jsonl"),
        },
        "rows": freeze_rows,
    }
    _write_json(root / "freeze.json", freeze)
    return freeze


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
        "event_keyframes": list(keyframes),
        "event_timestamps": _event_timestamps(events),
    }


def _blind_manifest(
    common: Mapping[str, Any], map_payload: Any
) -> dict[str, Any]:
    blind = dict(common)
    if isinstance(map_payload, Mapping):
        blind["map"] = _mask_question_fields(map_payload)
    return _drop_empty(blind)


def _aware_manifest(
    common: Mapping[str, Any], row: Mapping[str, Any], map_payload: Any
) -> dict[str, Any]:
    aware = dict(common)
    for field in QUESTION_AWARE_FIELDS:
        if field in row and row[field] is not None:
            aware[field] = row[field]
    if isinstance(map_payload, Mapping):
        aware["map"] = _strip_reference_fields(map_payload)
        if "explanation" in map_payload and "explanation" not in aware:
            aware["explanation"] = map_payload["explanation"]
    return _drop_empty(aware)


def _mask_question_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _mask_question_fields(nested)
            for key, nested in value.items()
            if str(key) not in BLIND_MASKED_FIELDS
        }
    if isinstance(value, list):
        return [_mask_question_fields(item) for item in value]
    return value


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
        timestamp = event.get("timestamp")
        if isinstance(timestamp, int | float):
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
