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
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from trec_video_2026.vqa_frontier_map import mask_question_fields
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from trec_video_2026.vqa_frontier_map import mask_question_fields


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
        if not map_payload_present and (
            not isinstance(failure_code, str) or not failure_code.strip()
        ):
            raise ValueError("absent map payload requires a frozen failure_code")
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
    if isinstance(bindings, Mapping) and "bindings_sha256" in freeze:
        _require_hash(
            _sha256_json(bindings),
            freeze.get("bindings_sha256"),
            "provider freeze bindings",
        )
    artifact = _read_json(artifact_path)
    rows = artifact.get("provider_results")
    if not isinstance(rows, list):
        raise ValueError("provider freeze artifact does not contain provider rows")
    return {
        "panel_manifest": artifact.get("panel_manifest", {}),
        "provider_results": rows,
    }


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
