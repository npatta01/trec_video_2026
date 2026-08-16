#!/usr/bin/env python3
"""Freeze the fresh issue #35 multi-interpretation diagnostic panel."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SOURCE_FILES = (
    "protocol.json",
    "maps.question-blind.json",
    "predictions.direct-answer.json",
    "postfreeze-score.json",
)
ROLES = (
    "premise_variance",
    "insufficiency_safeguard",
    "stability_control",
)
EXPECTED_PER_ROLE = 4
EXPECTED_ROWS = EXPECTED_PER_ROLE * len(ROLES)
PROMOTION_GATE = {
    "minimum_premise_correct_multi": 3,
    "minimum_premise_improvements_over_single": 2,
    "required_safeguard_verifications_each_arm": 4,
    "required_multi_controls_preserved": 4,
    "maximum_unsupported_emitted_answers": 0,
    "require_all_candidate_sets_audited": True,
    "oracle_is_diagnostic_only": True,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from error


def _index(rows: Any, *, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{source} rows must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} row must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in result:
            raise ValueError(f"{source} video IDs must be unique")
        result[video_id] = dict(row)
    return result


def _spent_ids(path: Path) -> set[str]:
    payload = _load(path)
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return set(_index(rows, source=str(path)))


def _evidence(map_text: str, line_numbers: Any, *, video_id: str) -> list[dict[str, Any]]:
    if (
        not isinstance(line_numbers, list)
        or not line_numbers
        or any(isinstance(value, bool) or not isinstance(value, int) for value in line_numbers)
        or len(line_numbers) != len(set(line_numbers))
    ):
        raise ValueError(f"{video_id} evidence lines must be unique integers")
    lines = map_text.splitlines()
    result = []
    for index, number in enumerate(line_numbers, 1):
        if number < 1 or number > len(lines) or not lines[number - 1].strip():
            raise ValueError(f"invalid evidence line for {video_id}: {number}")
        result.append(
            {
                "evidence_id": f"E{index:02d}",
                "map_line_number": number,
                "text": lines[number - 1].strip(),
            }
        )
    return result


def prepare(
    source_root: Path,
    annotations: Path,
    v2_protocol: Path,
    canary_manifest: Path,
    config_path: Path,
    output: Path,
) -> None:
    """Validate and immutably freeze provider-safe rows and label-bearing roles."""

    config = _load(config_path)
    expected_sources = config.get("source_sha256")
    if not isinstance(expected_sources, Mapping) or set(expected_sources) != set(SOURCE_FILES):
        raise ValueError("source hash manifest is incomplete")
    for name in SOURCE_FILES:
        if sha256_path(source_root / name) != expected_sources[name]:
            raise ValueError(f"source hash mismatch: {name}")
    exclusions = {
        "issue35_v2_protocol": v2_protocol,
        "seven_row_canary_manifest": canary_manifest,
    }
    expected_exclusions = config.get("exclusion_sha256")
    if not isinstance(expected_exclusions, Mapping) or set(expected_exclusions) != set(exclusions):
        raise ValueError("exclusion hash manifest is incomplete")
    for name, path in exclusions.items():
        if sha256_path(path) != expected_exclusions[name]:
            raise ValueError(f"exclusion hash mismatch: {name}")

    source_protocol = _index(_load(source_root / "protocol.json").get("rows"), source="protocol")
    maps = _index(_load(source_root / "maps.question-blind.json"), source="maps")
    predictions = _index(_load(source_root / "predictions.direct-answer.json"), source="predictions")
    score_payload = _load(source_root / "postfreeze-score.json")
    scores = _index(
        score_payload.get("rows", {}).get("gemini-3.1-pro-native-av-4fps"),
        source="scores",
    )
    spent = _spent_ids(v2_protocol) | _spent_ids(canary_manifest)
    selected = config.get("selected_rows")
    if not isinstance(selected, list) or len(selected) != EXPECTED_ROWS:
        raise ValueError(f"selection requires exactly {EXPECTED_ROWS} rows")
    ids = [row.get("video_id") for row in selected if isinstance(row, Mapping)]
    if len(ids) != EXPECTED_ROWS or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("selection IDs are invalid")
    if len(ids) != len(set(ids)):
        raise ValueError("selection IDs must be unique")
    overlap = set(ids) & spent
    if overlap:
        raise ValueError(f"selection contains previously spent rows: {sorted(overlap)}")
    role_counts = {role: 0 for role in ROLES}
    for row in selected:
        role = row.get("role")
        if role not in role_counts:
            raise ValueError("selection role is invalid")
        role_counts[role] += 1
    if role_counts != {role: EXPECTED_PER_ROLE for role in ROLES}:
        raise ValueError("selection role counts are invalid")
    if config.get("gate") != PROMOTION_GATE:
        raise ValueError("promotion gate drift")

    protocol_rows = []
    frozen_rows = []
    for row in selected:
        video_id = row["video_id"]
        if video_id not in source_protocol or video_id not in maps or video_id not in predictions or video_id not in scores:
            raise ValueError(f"selected row missing from a source: {video_id}")
        if maps[video_id].get("status") != "passed" or predictions[video_id].get("status") != "passed":
            raise ValueError(f"selected source row did not pass: {video_id}")
        note = row.get("selection_note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"selection note is missing: {video_id}")
        evidence = _evidence(
            maps[video_id].get("map", ""),
            row.get("evidence_line_numbers"),
            video_id=video_id,
        )
        annotation_path = annotations / f"{video_id}.json"
        annotation = _load(annotation_path)
        if not isinstance(annotation.get("correct_answer"), str):
            raise ValueError(f"annotation is invalid: {video_id}")
        protocol_rows.append(
            {
                "video_id": video_id,
                "question": source_protocol[video_id]["question"],
                "evidence": evidence,
            }
        )
        frozen_rows.append(
            {
                "video_id": video_id,
                "role": row["role"],
                "selection_note": note.strip(),
                "direct_prediction": predictions[video_id]["answer"],
                "direct_semantic_composite": scores[video_id]["semantic_composite"],
                "question_stratum": scores[video_id]["question_stratum"],
                "annotation_sha256": sha256_path(annotation_path),
            }
        )

    created_at = datetime.now(UTC).isoformat()
    protocol = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_at": created_at,
        "label_isolation": "provider rows contain only video_id, question, and frozen evidence",
        "rows": protocol_rows,
    }
    _write_once(output / "protocol.source.json", protocol)
    selection_freeze = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": config["experiment_id"],
        "created_at": created_at,
        "row_count": EXPECTED_ROWS,
        "role_counts": role_counts,
        "rows": frozen_rows,
        "config_sha256": sha256_path(config_path),
        "source_sha256": dict(expected_sources),
        "exclusion_sha256": dict(expected_exclusions),
        "protocol_source_sha256": sha256_path(output / "protocol.source.json"),
    }
    _write_once(output / "selection-freeze.json", selection_freeze)
    gate_freeze = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": config["experiment_id"],
        "selection_freeze_sha256": sha256_path(output / "selection-freeze.json"),
        "protocol_source_sha256": sha256_path(output / "protocol.source.json"),
        "panel": {role: [row["video_id"] for row in selected if row["role"] == role] for role in ROLES},
        "gate": PROMOTION_GATE,
    }
    _write_once(output / "preinference-gate-freeze.json", gate_freeze)
    _write_once(
        output / "selection-integrity.json",
        {
            "protocol_source_sha256": sha256_path(output / "protocol.source.json"),
            "selection_freeze_sha256": sha256_path(output / "selection-freeze.json"),
            "preinference_gate_freeze_sha256": sha256_path(output / "preinference-gate-freeze.json"),
        },
    )


def verify_selection(root: Path) -> None:
    integrity = _load(root / "selection-integrity.json")
    expected = {
        "protocol_source_sha256": sha256_path(root / "protocol.source.json"),
        "selection_freeze_sha256": sha256_path(root / "selection-freeze.json"),
        "preinference_gate_freeze_sha256": sha256_path(root / "preinference-gate-freeze.json"),
    }
    if integrity != expected:
        raise ValueError("selection integrity mismatch")
    selection = _load(root / "selection-freeze.json")
    gate = _load(root / "preinference-gate-freeze.json")
    if selection.get("protocol_source_sha256") != expected["protocol_source_sha256"]:
        raise ValueError("selection integrity protocol mismatch")
    if gate.get("selection_freeze_sha256") != expected["selection_freeze_sha256"]:
        raise ValueError("selection integrity gate mismatch")
    if gate.get("gate") != PROMOTION_GATE:
        raise ValueError("selection integrity gate drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.annotations, args.v2_protocol, args.canary_manifest, args.config, args.output)
    verify_selection(args.output)


if __name__ == "__main__":
    main()
