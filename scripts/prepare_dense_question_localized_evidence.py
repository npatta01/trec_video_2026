#!/usr/bin/env python3
"""Freeze the dense question-localized evidence panel and local media.

This module is intentionally provider-free.  It consumes the completed
question-blind protocol, its maps, and the corrected private Issue #35 audit;
it never calls a model.  The resulting write-once JSON files and clips are
the private inputs consumed by ``run_dense_question_localized_evidence.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
DEFAULT_ROOT = ROOT / "data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1"
DEFAULT_PROTOCOL = ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1/protocol.json"
DEFAULT_MAPS = ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1/maps.question-blind.json"
DEFAULT_AUDIT = ROOT / "data/vqa/benchmarks/task-support-sufficiency-audit-issue35-v1/manual-audit-corrected-v3.json"
TASK1_RUNNER = ROOT / "scripts/run_dense_question_localized_evidence.py"
TASK1_CONFIG = ROOT / "configs/vqa/dense_question_localized_evidence_issue35.json"

MODALITIES = (
    "brief_visual_action",
    "speech_or_audio",
    "onscreen_text",
    "fine_spatial_detail",
    "genuinely_absent_or_ambiguous",
)
TARGET_ROLE = "target"
CONTROL_ROLE = "control"
MAX_CLIP_BYTES = 100 * 1024 * 1024
MIN_WINDOW_SECONDS = 2.0
MAX_WINDOW_SECONDS = 6.0
STAGING_DIR_NAME = ".panel-staging"
INCOMPLETE_NAME = "panel-preparation-incomplete.json"
EXPOSURE_KEYS = (
    "grounded_v2_24",
    "grounded_v3_11",
    "hard_audit_30",
    "multi_interpretation_12",
    "scope_control",
    "seven_row_canary",
    "synthesis_followup",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable artifact: {path}") from error


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _root(value: Path | str | None) -> Path:
    return Path(value) if value is not None else DEFAULT_ROOT


def _resolve(path: Path | str, *, base: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, list):
        raise ValueError(f"{field} must contain a rows list")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{field} row {index} must be an object")
        result.append(dict(row))
    return result


def _by_video(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        video_id = _string(row.get("video_id"), f"{field} row {index}.video_id")
        if video_id in result:
            raise ValueError(f"{field} contains duplicate video_id: {video_id}")
        result[video_id] = dict(row)
    return result


def _audit_rows(corrected_audit: Path) -> list[dict[str, Any]]:
    rows = _rows(_load_json(corrected_audit), str(corrected_audit))
    seen: set[int] = set()
    for index, row in enumerate(rows):
        audit_index = _positive_int(row.get("audit_index"), f"audit row {index}.audit_index")
        if audit_index in seen:
            raise ValueError(f"corrected audit has duplicate audit_index: {audit_index}")
        seen.add(audit_index)
        role = _string(row.get("role"), f"audit row {index}.role")
        if role not in {"insufficient_or_ambiguous", "supported_control"}:
            continue
    return rows


def _map_lines(map_text: str) -> list[dict[str, Any]]:
    text = _string(map_text, "map")
    return [{"line_number": index, "text": line} for index, line in enumerate(text.splitlines(), 1) if line.strip()]


def _map_index(source_maps: Path) -> dict[str, dict[str, Any]]:
    rows = _rows(_load_json(source_maps), str(source_maps))
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        video_id = _string(row.get("video_id"), f"maps row {index}.video_id")
        map_text = _string(row.get("map"), f"maps row {index}.map")
        if video_id in result:
            raise ValueError(f"maps contain duplicate video_id: {video_id}")
        result[video_id] = {
            "video_id": video_id,
            "map": map_text,
            "map_sha256": _sha256_bytes(map_text.encode()),
            "map_lines": _map_lines(map_text),
            "condition_id": row.get("condition_id"),
            "status": row.get("status"),
        }
    return result


def _protocol_index(source_protocol: Path) -> dict[str, dict[str, Any]]:
    rows = _rows(_load_json(source_protocol), str(source_protocol))
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        video_id = _string(row.get("video_id"), f"protocol row {index}.video_id")
        if video_id in result:
            raise ValueError(f"protocol contains duplicate video_id: {video_id}")
        result[video_id] = dict(row)
    return result


def _task1_module(runner_path: Path):
    spec = importlib.util.spec_from_file_location("dense_question_localized_evidence_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Task 1 runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task1_hashes(runner_path: Path, config_path: Path) -> dict[str, Any]:
    runner = _task1_module(runner_path)
    if not runner_path.is_file() or not config_path.is_file():
        raise ValueError("Task 1 runner/config is missing")
    config = _load_json(config_path)
    expected = {
        "model": runner.MODEL,
        "settings": runner.EXPECTED_SETTINGS,
        "expected_fps": list(runner.EXPECTED_FPS),
        "max_calls": runner.MAX_CALLS,
        "compatibility_canary_pairs": runner.COMPATIBILITY_CANARY_PAIRS,
        "decision_gate": runner.EXPECTED_DECISION_GATE,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Task 1 config {key} does not match committed runner")
    return {
        "runner_sha256": _sha256(runner_path),
        "config_sha256": _sha256(config_path),
        "prompt_template_sha256": runner._prompt_template_sha256(),
        "response_schema_sha256": runner._schema_sha256(),
        "provider_response_schema_sha256": runner._provider_schema_sha256(),
        "validator_sha256": runner._validator_sha256(),
        "model": runner.MODEL,
        "settings": runner.EXPECTED_SETTINGS,
        "expected_fps": list(runner.EXPECTED_FPS),
        "max_calls": runner.MAX_CALLS,
        "compatibility_canary_pairs": runner.COMPATIBILITY_CANARY_PAIRS,
        "decision_gate": runner.EXPECTED_DECISION_GATE,
    }


def _validate_task1_treatment(
    treatment: Mapping[str, Any],
    *,
    runner_path: Path = TASK1_RUNNER,
    config_path: Path = TASK1_CONFIG,
) -> dict[str, Any]:
    """Rebind a treatment freeze to the currently committed Task 1 bytes."""

    current = _task1_hashes(_resolve(runner_path), _resolve(config_path))
    for field in (
        "runner_sha256",
        "config_sha256",
        "prompt_template_sha256",
        "response_schema_sha256",
        "provider_response_schema_sha256",
        "validator_sha256",
        "model",
        "settings",
        "expected_fps",
        "max_calls",
        "compatibility_canary_pairs",
        "decision_gate",
    ):
        if treatment.get(field) != current[field]:
            raise ValueError(f"treatment {field} does not match current Task 1 bytes")
    if treatment.get("task1_treatment_sha256") != _sha256_bytes(_canonical(current)):
        raise ValueError("exact Task 1 treatment hash drifted")
    return current


def _source_bundle(
    source_protocol: Path,
    source_maps: Path,
    corrected_audit: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    protocol_rows = _protocol_index(source_protocol)
    map_rows = _map_index(source_maps)
    audit_rows = _audit_rows(corrected_audit)
    eligible = [
        row for row in audit_rows
        if row.get("role") in {"insufficient_or_ambiguous", "supported_control"}
    ]
    for row in eligible:
        video_id = _string(row.get("video_id"), f"audit row {row['audit_index']}.video_id")
        if video_id not in protocol_rows:
            raise ValueError(f"audit video is absent from full protocol: {video_id}")
        if video_id not in map_rows:
            raise ValueError(f"audit video is absent from full-panel maps: {video_id}")
    return eligible, protocol_rows, map_rows, audit_rows


def freeze_treatment(
    *,
    root: Path | str = DEFAULT_ROOT,
    source_protocol: Path | str = DEFAULT_PROTOCOL,
    source_maps: Path | str = DEFAULT_MAPS,
    corrected_audit: Path | str = DEFAULT_AUDIT,
    runner_path: Path | str = TASK1_RUNNER,
    config_path: Path | str = TASK1_CONFIG,
    protocol: Path | str | None = None,
    maps: Path | str | None = None,
    audit: Path | str | None = None,
    task1_runner: Path | str | None = None,
    task1_config: Path | str | None = None,
) -> Path:
    """Freeze the exact Task 1 treatment and all source-byte identities."""

    root_path = _root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    protocol_path = _resolve(protocol or source_protocol)
    maps_path = _resolve(maps or source_maps)
    audit_path = _resolve(audit or corrected_audit)
    runner_file = _resolve(task1_runner or runner_path)
    config_file = _resolve(task1_config or config_path)
    eligible, _protocol, _maps, audit_rows = _source_bundle(protocol_path, maps_path, audit_path)
    task1 = _task1_hashes(runner_file, config_file)
    role_counts = Counter(
        TARGET_ROLE if row.get("role") == "insufficient_or_ambiguous" else CONTROL_ROLE
        for row in eligible
    )
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "model": task1["model"],
        "settings": task1["settings"],
        "expected_fps": task1["expected_fps"],
        "max_calls": task1["max_calls"],
        "compatibility_canary_pairs": task1["compatibility_canary_pairs"],
        "decision_gate": task1["decision_gate"],
        "runner_sha256": task1["runner_sha256"],
        "config_sha256": task1["config_sha256"],
        "prompt_template_sha256": task1["prompt_template_sha256"],
        "response_schema_sha256": task1["response_schema_sha256"],
        "provider_response_schema_sha256": task1["provider_response_schema_sha256"],
        "validator_sha256": task1["validator_sha256"],
        "task1_treatment_sha256": _sha256_bytes(_canonical(task1)),
        "source_protocol": str(protocol_path),
        "source_protocol_sha256": _sha256(protocol_path),
        "source_maps": str(maps_path),
        "source_maps_sha256": _sha256(maps_path),
        "corrected_audit": str(audit_path),
        "corrected_audit_sha256": _sha256(audit_path),
        "audit_row_count": len(audit_rows),
        "eligible_row_count": len(eligible),
        "insufficiency_count": role_counts[TARGET_ROLE],
        "control_count": role_counts[CONTROL_ROLE],
        "exposure_policy": "reporting flags only; no eligibility exclusions",
        "label_isolation": "provider receives only question, frozen map lines, clip offset, and local clip",
    }
    output = root_path / "treatment-freeze.json"
    _write_once(output, payload)
    return output


def _require_treatment(root: Path) -> dict[str, Any]:
    path = root / "treatment-freeze.json"
    if not path.is_file():
        raise ValueError("committed treatment freeze is required before modality audit")
    treatment = _load_json(path)
    if not isinstance(treatment, Mapping) or treatment.get("status") != "frozen":
        raise ValueError("treatment freeze must be committed with status=frozen")
    return dict(treatment)


def _template_rows(
    eligible: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Mapping[str, Any]],
    maps: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in eligible:
        audit_index = _positive_int(row.get("audit_index"), "audit row.audit_index")
        video_id = _string(row.get("video_id"), f"audit row {audit_index}.video_id")
        source = protocol[video_id]
        map_data = maps[video_id]
        role = TARGET_ROLE if row.get("role") == "insufficient_or_ambiguous" else CONTROL_ROLE
        evidence = row.get("evidence_line_numbers", [])
        if not isinstance(evidence, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in evidence
        ):
            raise ValueError(f"audit row {audit_index}.evidence_line_numbers is invalid")
        result.append(
            {
                "audit_index": audit_index,
                "video_id": video_id,
                "role": role,
                "source_role": row.get("role"),
                "question": _string(row.get("question", source.get("question")), f"audit row {audit_index}.question"),
                "question_stratum": source.get("question_stratum", row.get("question_stratum")),
                # Source media is required when clips are frozen, but keeping
                # these fields nullable in the blank audit lets reviewers
                # classify a synthetic/template row before media preparation.
                "source_path": str(source.get("source_path") or row.get("source_path") or ""),
                "source_sha256": str(source.get("source_sha256") or row.get("source_sha256") or ""),
                "source_duration_seconds": float(source.get("duration_seconds") or row.get("source_duration_seconds") or 30.0),
                "map_sha256": map_data["map_sha256"],
                "map_lines": map_data["map_lines"],
                "evidence_line_numbers": evidence,
                "exposure_flags": {
                    key: bool(row.get("exposure_flags", {}).get(key, False))
                    for key in EXPOSURE_KEYS
                },
                "modality": None,
                "rationale": None,
                "map_line_numbers": [],
                "window_start_seconds": None,
                "window_end_seconds": None,
                "include": None,
                "visually_eligible": None,
            }
        )
    return result


def prepare_modality_audit(
    *,
    root: Path | str = DEFAULT_ROOT,
    source_protocol: Path | str = DEFAULT_PROTOCOL,
    source_maps: Path | str = DEFAULT_MAPS,
    corrected_audit: Path | str = DEFAULT_AUDIT,
    runner_path: Path | str = TASK1_RUNNER,
    config_path: Path | str = TASK1_CONFIG,
    protocol: Path | str | None = None,
    maps: Path | str | None = None,
    audit: Path | str | None = None,
    task1_runner: Path | str | None = None,
    task1_config: Path | str | None = None,
    manual_audit: Path | str | None = None,
    modality_template: Path | str | None = None,
    **_unused: Any,
) -> Path:
    """Create the blank label-aware audit after treatment is committed."""

    root_path = _root(root)
    treatment = _require_treatment(root_path)
    _validate_task1_treatment(
        treatment,
        runner_path=_resolve(task1_runner or runner_path),
        config_path=_resolve(task1_config or config_path),
    )
    protocol_path = _resolve(protocol or source_protocol)
    maps_path = _resolve(maps or source_maps)
    audit_path = _resolve(audit or corrected_audit)
    eligible, protocol, maps, _audit = _source_bundle(protocol_path, maps_path, audit_path)
    rows = _template_rows(eligible, protocol, maps)
    source_hashes = {
        "protocol_sha256": _sha256(protocol_path),
        "maps_sha256": _sha256(maps_path),
        "corrected_audit_sha256": _sha256(audit_path),
        "treatment_freeze_sha256": _sha256(root_path / "treatment-freeze.json"),
        "task1_treatment_sha256": treatment["task1_treatment_sha256"],
    }
    payload = {
        "schema_version": 1,
        "status": "template",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "private": True,
        "provider_input": False,
        "created_at": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes,
        "treatment_freeze_sha256": source_hashes["treatment_freeze_sha256"],
        "target_count": sum(row["role"] == TARGET_ROLE for row in rows),
        "control_count": sum(row["role"] == CONTROL_ROLE for row in rows),
        "rows": rows,
        "blank_fields": [
            "modality",
            "rationale",
            "map_line_numbers",
            "window_start_seconds",
            "window_end_seconds",
            "include",
        ],
    }
    template_path = _resolve(modality_template, base=root_path) if modality_template else root_path / "modality-audit-template.json"
    _write_once(template_path, payload)
    manual = {**payload, "status": "manual_pending"}
    manual_path = _resolve(manual_audit, base=root_path) if manual_audit else root_path / "manual-modality-audit.json"
    _write_once(manual_path, manual)
    return manual_path


def _validate_audit_rows(
    root: Path,
    manual: Mapping[str, Any],
    template: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _rows(manual, "manual-modality-audit.json")
    template_rows = _rows(template, "modality-audit-template.json")
    expected_ids = [row["audit_index"] for row in expected_rows]
    actual_ids = [row.get("audit_index") for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("manual modality audit ordering or audit indexes drifted")
    if len(rows) != len(expected_rows):
        raise ValueError(f"manual modality audit must classify all {len(expected_rows)} rows")
    if [row.get("audit_index") for row in template_rows] != expected_ids:
        raise ValueError("modality template ordering drifted")
    for index, (row, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        if row.get("video_id") != expected.get("video_id") or row.get("role") != expected.get("role"):
            raise ValueError(f"manual modality row {index} identity drifted")
        if row.get("question") != expected.get("question"):
            raise ValueError(f"question drift for audit_index {row['audit_index']}")
        if row.get("map_lines") != expected.get("map_lines"):
            raise ValueError(f"map-line drift for audit_index {row['audit_index']}")
        for field in ("source_path", "source_sha256", "source_duration_seconds"):
            if row.get(field) != expected.get(field):
                raise ValueError(f"{field} drift for audit_index {row['audit_index']}")
        exposure_flags = row.get("exposure_flags")
        if (
            not isinstance(exposure_flags, Mapping)
            or set(exposure_flags) != set(EXPOSURE_KEYS)
            or any(not isinstance(exposure_flags[key], bool) for key in EXPOSURE_KEYS)
        ):
            raise ValueError(f"exposure flags are invalid for audit_index {row['audit_index']}")
        if exposure_flags != expected.get("exposure_flags"):
            raise ValueError(f"exposure flag drift for audit_index {row['audit_index']}")
        if row.get("map_sha256") != maps[row["video_id"]]["map_sha256"]:
            raise ValueError(f"map-line drift for audit_index {row['audit_index']}")
        if row.get("role") == TARGET_ROLE:
            modality = row.get("modality")
            if modality not in MODALITIES:
                raise ValueError(f"audit_index {row['audit_index']} needs one valid modality")
            include = row.get("include")
            if not isinstance(include, bool):
                raise ValueError(f"audit_index {row['audit_index']} include must be boolean")
            if include:
                _validate_included_row(row, maps[row["video_id"]])
        else:
            eligible = row.get("visually_eligible")
            if not isinstance(eligible, bool):
                raise ValueError(f"control audit_index {row['audit_index']} needs visually_eligible boolean")
            include = row.get("include")
            if include is not None and not isinstance(include, bool):
                raise ValueError(f"control audit_index {row['audit_index']} include must be boolean or null")
            if eligible:
                if include is not True:
                    raise ValueError(f"eligible control audit_index {row['audit_index']} must include=true")
                _validate_included_row(row, maps[row["video_id"]])
            elif include is True:
                raise ValueError(f"ineligible control audit_index {row['audit_index']} cannot be included")
    return rows


def _validate_included_row(row: Mapping[str, Any], map_data: Mapping[str, Any]) -> None:
    rationale = row.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"audit_index {row['audit_index']} included row needs rationale")
    line_numbers = row.get("map_line_numbers")
    if not isinstance(line_numbers, list) or not line_numbers:
        raise ValueError(f"audit_index {row['audit_index']} included row needs map lines")
    available = {line["line_number"] for line in map_data["map_lines"]}
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in available for value in line_numbers):
        raise ValueError(f"audit_index {row['audit_index']} has invalid map line numbers")
    if len(line_numbers) != len(set(line_numbers)):
        raise ValueError(f"audit_index {row['audit_index']} has duplicate map line numbers")
    start = _number(row.get("window_start_seconds"), f"audit_index {row['audit_index']}.window_start_seconds")
    end = _number(row.get("window_end_seconds"), f"audit_index {row['audit_index']}.window_end_seconds")
    if end <= start or end - start < MIN_WINDOW_SECONDS or end - start > MAX_WINDOW_SECONDS:
        raise ValueError(
            f"audit_index {row['audit_index']} window must be between 2 and 6 seconds"
        )
    duration = _number(row.get("source_duration_seconds"), f"audit_index {row['audit_index']}.source_duration_seconds")
    if start < 0 or end > duration + 0.05:
        raise ValueError(f"audit_index {row['audit_index']} window is outside source duration")


def _selected_rows(manual_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visual_targets = [
        dict(row) for row in manual_rows
        if row.get("role") == TARGET_ROLE
        and row.get("include") is True
        and row.get("modality") == "brief_visual_action"
    ]
    eligible_controls = [
        dict(row) for row in manual_rows
        if row.get("role") == CONTROL_ROLE
        and row.get("visually_eligible") is True
        and row.get("include") is True
    ]
    return visual_targets[:8], eligible_controls[:4]


def _run_ffmpeg_clip(source: Path, output: Path, start: float, end: float) -> dict[str, Any]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ValueError("ffmpeg and ffprobe are required to create local clips")
    duration = end - start
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-map",
        "0:a:0?",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr[-1000:] if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise ValueError(f"ffmpeg failed for {source}: {detail}") from error
    if not output.is_file():
        raise ValueError(f"ffmpeg did not create clip: {output}")
    if output.stat().st_size > MAX_CLIP_BYTES:
        raise ValueError(f"output clip exceeds 100 MB: {output}")
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,pix_fmt,avg_frame_rate",
        "-of",
        "json",
        str(output),
    ]
    try:
        probe = subprocess.run(probe_command, check=True, capture_output=True, text=True)
        metadata = json.loads(probe.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ValueError(f"ffprobe failed for {output}") from error
    streams = metadata.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping) or video.get("codec_name") != "h264":
        raise ValueError(f"clip does not contain H.264 video: {output}")
    duration_value = metadata.get("format", {}).get("duration")
    if isinstance(duration_value, str):
        try:
            duration_value = float(duration_value)
        except ValueError as error:
            raise ValueError(f"ffprobe duration is invalid: {output}") from error
    actual_duration = _number(duration_value, f"ffprobe duration {output}")
    if actual_duration <= 0:
        raise ValueError(f"clip duration is not positive: {output}")
    return {
        "duration_seconds": round(actual_duration, 6),
        "video_codec": video.get("codec_name"),
        "video_pixel_format": video.get("pix_fmt"),
        "video_frame_rate": video.get("avg_frame_rate"),
        "audio_codec": audio.get("codec_name") if isinstance(audio, Mapping) else None,
        "ffmpeg_command": command,
        "ffprobe_command": probe_command,
        "probe": metadata,
    }


def _make_clip(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    source = _resolve(row["source_path"])
    if not source.is_file():
        raise ValueError(f"source media is missing: {source}")
    source_sha = _sha256(source)
    if source_sha != row["source_sha256"]:
        raise ValueError(f"source SHA drift for video_id {row['video_id']}")
    start = _number(row["window_start_seconds"], f"audit_index {row['audit_index']}.window_start_seconds")
    end = _number(row["window_end_seconds"], f"audit_index {row['audit_index']}.window_end_seconds")
    output = root / "clips" / f"{row['audit_index']:02d}-{row['video_id']}.mp4"
    if output.exists():
        raise ValueError(f"refusing to overwrite existing clip: {output}")
    temporary = output.with_name(output.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        probe = _run_ffmpeg_clip(source, temporary, start, end)
        clip_sha = _sha256(temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "audit_index": row["audit_index"],
        "video_id": row["video_id"],
        "role": row["role"],
        "clip_path": str(output.relative_to(root)),
        "clip_sha256": clip_sha,
        "source_path": row["source_path"],
        "source_sha256": source_sha,
        "clip_start_seconds": start,
        "clip_end_seconds": end,
        "clip_duration_seconds": probe["duration_seconds"],
        "mime_type": "video/mp4",
        "video_codec": probe["video_codec"],
        "audio_codec": probe["audio_codec"],
        "command_policy": {
            "input_seeking": True,
            "video_codec": "libx264",
            "video_crf": 0,
            "pixel_format": "yuv420p",
            "audio_codec": "aac_when_present",
            "audio_bitrate": "192k",
            "faststart": True,
        },
        "ffmpeg_command": probe["ffmpeg_command"],
        "ffprobe_command": probe["ffprobe_command"],
        "probed_codecs": {
            "video": probe["video_codec"],
            "audio": probe["audio_codec"],
        },
        "arms": {
            "4fps": {"fps": 4.0, "clip_sha256": clip_sha},
            "12fps": {"fps": 12.0, "clip_sha256": clip_sha},
        },
    }


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Write mutable staging state; final evidence remains write-once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _serialized_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _stage_panel_clips(
    root: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    treatment: Mapping[str, Any],
    manual_path: Path,
) -> list[dict[str, Any]]:
    """Create or resume a complete private clip set before final promotion."""

    stage = root / STAGING_DIR_NAME
    stage.mkdir(parents=True, exist_ok=True)
    progress_path = stage / "progress.json"
    signature = _sha256_bytes(
        _canonical(
            {
                "selected_rows": [row["audit_index"] for row in selected_rows],
                "treatment_freeze_sha256": _sha256(root / "treatment-freeze.json"),
                "manual_modality_audit_sha256": _sha256(manual_path),
            }
        )
    )
    if progress_path.is_file():
        progress = _load_json(progress_path)
        if progress.get("signature") != signature:
            raise ValueError("panel staging signature drifted; refusing to resume")
    else:
        progress = {
            "schema_version": 1,
            "status": "incomplete",
            "signature": signature,
            "selected_rows": [row["audit_index"] for row in selected_rows],
            "completed_rows": [],
            "media_rows": [],
        }
        _write_progress(progress_path, progress)
    completed = {row["audit_index"]: row for row in progress.get("media_rows", []) if isinstance(row, Mapping)}
    selected_ids = [row["audit_index"] for row in selected_rows]
    if set(completed) - set(selected_ids):
        raise ValueError("panel staging contains an unexpected row")
    staged_rows: list[dict[str, Any]] = []
    try:
        for row in selected_rows:
            audit_index = row["audit_index"]
            previous = completed.get(audit_index)
            if previous is not None:
                clip_path = stage / previous["clip_path"]
                if not clip_path.is_file() or _sha256(clip_path) != previous.get("clip_sha256"):
                    raise ValueError(f"staged clip is missing or changed for audit_index {audit_index}")
                staged_rows.append(dict(previous))
                continue
            staged = _make_clip(stage, row)
            staged_rows.append(staged)
            completed[audit_index] = staged
            progress["completed_rows"] = [item for item in selected_ids if item in completed]
            progress["media_rows"] = [completed[item] for item in progress["completed_rows"]]
            _write_progress(progress_path, progress)
    except BaseException as error:
        incomplete_path = root / INCOMPLETE_NAME
        if not incomplete_path.exists():
            _write_once(
                incomplete_path,
                {
                    "schema_version": 1,
                    "status": "incomplete",
                    "outcome": "panel_preparation_incomplete",
                    "signature": signature,
                    "selected_rows": selected_ids,
                    "completed_rows": [item for item in selected_ids if item in completed],
                    "provider_calls_made": 0,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    if len(staged_rows) != len(selected_rows):
        raise ValueError("panel staging did not complete every selected row")
    progress["status"] = "complete"
    progress["completed_rows"] = selected_ids
    progress["media_rows"] = staged_rows
    _write_progress(progress_path, progress)
    return staged_rows


def _bundle_signature(root: Path, selected_rows: Sequence[Mapping[str, Any]], manual_path: Path) -> str:
    return _sha256_bytes(
        _canonical(
            {
                "selected_rows": [row["audit_index"] for row in selected_rows],
                "treatment_freeze_sha256": _sha256(root / "treatment-freeze.json"),
                "manual_modality_audit_sha256": _sha256(manual_path),
            }
        )
    )


def _promote_bundle_artifact(root: Path, artifact: Mapping[str, Any]) -> None:
    """Promote one staged byte set without ever overwriting final evidence."""

    final = root / artifact["final"]
    staged = root / artifact["staged"]
    expected_sha = artifact["sha256"]
    if final.exists():
        if not final.is_file() or _sha256(final) != expected_sha:
            raise ValueError(f"promoted artifact drifted: {final}")
        return
    if not staged.is_file():
        raise ValueError(f"staged artifact is missing: {staged}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(final)
    if _sha256(final) != expected_sha:
        raise ValueError(f"promoted artifact hash mismatch: {final}")


def _resume_bundle_promotion(root: Path, expected_signature: str | None = None) -> None:
    ledger_path = root / "bundle-promotion.json"
    if not ledger_path.is_file():
        raise ValueError("bundle promotion ledger is missing")
    ledger = _load_json(ledger_path)
    if expected_signature is not None and ledger.get("signature") != expected_signature:
        raise ValueError("bundle promotion signature drifted; refusing recovery")
    artifacts = ledger.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("bundle promotion ledger has no artifacts")
    progress_path = root / "bundle-promotion-progress.json"
    progress = _load_json(progress_path) if progress_path.is_file() else {"completed": []}
    completed = set(progress.get("completed", []))
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"bundle promotion artifact {index} is invalid")
        artifact_id = str(artifact.get("final"))
        _promote_bundle_artifact(root, artifact)
        completed.add(artifact_id)
        _write_progress(
            progress_path,
            {
                "schema_version": 1,
                "status": "complete" if len(completed) == len(artifacts) else "promoting",
                "ledger_sha256": _sha256(ledger_path),
                "completed": [item["final"] for item in artifacts if item["final"] in completed],
            },
        )
    if len(completed) != len(artifacts):
        raise ValueError("bundle promotion did not complete every artifact")


def _promote_bundle(
    root: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    clips: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Any],
    signature: str,
) -> None:
    """Stage JSON and promote clips/JSON with an immutable recovery ledger."""

    stage = root / STAGING_DIR_NAME
    final_stage = stage / "final"
    final_stage.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for clip in clips:
        staged = stage / clip["clip_path"]
        final = Path(clip["clip_path"])
        artifacts.append(
            {
                "kind": "clip",
                "staged": str(staged.relative_to(root)),
                "final": str(final),
                "sha256": clip["clip_sha256"],
            }
        )
    for name, payload in payloads.items():
        staged = final_stage / name
        staged.write_bytes(_serialized_json(payload))
        artifacts.append(
            {
                "kind": "json",
                "staged": str(staged.relative_to(root)),
                "final": name,
                "sha256": _sha256(staged),
            }
        )
    ledger_path = root / "bundle-promotion.json"
    if not ledger_path.exists():
        _write_once(
            ledger_path,
            {
                "schema_version": 1,
                "status": "promoting",
                "signature": signature,
                "selected_rows": [row["audit_index"] for row in selected_rows],
                "artifacts": artifacts,
                "provider_calls_made": 0,
            },
        )
    else:
        ledger = _load_json(ledger_path)
        if ledger.get("signature") != signature or ledger.get("artifacts") != artifacts:
            raise ValueError("bundle promotion ledger drifted; refusing overwrite")
    _resume_bundle_promotion(root, expected_signature=signature)


def _provider_row(row: Mapping[str, Any]) -> dict[str, Any]:
    lines = row.get("map_lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"audit_index {row['audit_index']} has no map lines")
    line_by_number = {line["line_number"]: line for line in lines}
    selected_numbers = row.get("map_line_numbers")
    if not isinstance(selected_numbers, list) or not selected_numbers:
        raise ValueError(f"audit_index {row['audit_index']} has no selected map lines")
    try:
        selected_lines = [line_by_number[number] for number in selected_numbers]
    except (KeyError, TypeError) as error:
        raise ValueError(f"audit_index {row['audit_index']} selected map lines drifted") from error
    return {
        "audit_index": row["audit_index"],
        "video_id": row["video_id"],
        "role": row["role"],
        "question": row["question"],
        "map_lines": [{"line_number": line["line_number"], "text": line["text"]} for line in selected_lines],
        "clip_start_seconds": row["window_start_seconds"],
        "clip_end_seconds": row["window_end_seconds"],
        "clip_duration_seconds": row["window_end_seconds"] - row["window_start_seconds"],
    }


def _exposure_report(rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_ids = {row["audit_index"] for row in selected}
    result: dict[str, Any] = {}
    for key in EXPOSURE_KEYS:
        audited = sum(bool(row.get("exposure_flags", {}).get(key, False)) for row in rows)
        chosen = sum(
            row["audit_index"] in selected_ids and bool(row.get("exposure_flags", {}).get(key, False))
            for row in rows
        )
        result[key] = {"audited": audited, "selected": chosen}
    return result


def _write_feasibility_freeze(
    root: Path,
    treatment: Mapping[str, Any],
    manual_path: Path,
    target_count: int,
    control_count: int,
    reason: str,
) -> Path:
    """Persist a write-once zero-call result when the panel is infeasible."""

    payload = {
        "schema_version": 1,
        "status": "frozen",
        "outcome": "infeasible",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "target_count": target_count,
        "control_count": control_count,
        "minimum_targets": treatment["decision_gate"]["minimum_targets"],
        "minimum_controls": treatment["decision_gate"]["minimum_controls"],
        "planned_call_count": 0,
        "provider_calls_made": 0,
        "treatment_freeze_sha256": _sha256(root / "treatment-freeze.json"),
        "modality_audit_sha256": _sha256(manual_path),
        "decision_gate": treatment["decision_gate"],
    }
    output = root / "feasibility-freeze.json"
    _write_once(output, payload)
    return output


def freeze_panel(
    *,
    root: Path | str = DEFAULT_ROOT,
    source_protocol: Path | str = DEFAULT_PROTOCOL,
    source_maps: Path | str = DEFAULT_MAPS,
    corrected_audit: Path | str = DEFAULT_AUDIT,
    runner_path: Path | str = TASK1_RUNNER,
    config_path: Path | str = TASK1_CONFIG,
    protocol: Path | str | None = None,
    maps: Path | str | None = None,
    audit: Path | str | None = None,
    task1_runner: Path | str | None = None,
    task1_config: Path | str | None = None,
    manual_audit: Path | str | None = None,
    modality_template: Path | str | None = None,
    **_unused: Any,
) -> Path:
    """Validate the completed audit, select the panel, and freeze local clips."""

    root_path = _root(root)
    treatment = _require_treatment(root_path)
    _validate_task1_treatment(
        treatment,
        runner_path=_resolve(task1_runner or runner_path),
        config_path=_resolve(task1_config or config_path),
    )
    template_path = _resolve(modality_template, base=root_path) if modality_template else root_path / "modality-audit-template.json"
    manual_path = _resolve(manual_audit, base=root_path) if manual_audit else root_path / "manual-modality-audit.json"
    if not template_path.is_file() or not manual_path.is_file():
        raise ValueError("modality template and completed manual audit are required")
    protocol_path = _resolve(protocol or source_protocol)
    maps_path = _resolve(maps or source_maps)
    audit_path = _resolve(audit or corrected_audit)
    eligible, protocol, maps, _audit = _source_bundle(protocol_path, maps_path, audit_path)
    expected_rows = _template_rows(eligible, protocol, maps)
    template = _load_json(template_path)
    manual = _load_json(manual_path)
    expected_hashes = {
        "protocol_sha256": _sha256(protocol_path),
        "maps_sha256": _sha256(maps_path),
        "corrected_audit_sha256": _sha256(audit_path),
        "treatment_freeze_sha256": _sha256(root_path / "treatment-freeze.json"),
        "task1_treatment_sha256": treatment["task1_treatment_sha256"],
    }
    if template.get("source_hashes") != expected_hashes:
        raise ValueError("modality template source or treatment hash drifted")
    if manual.get("source_hashes") != expected_hashes:
        raise ValueError("manual modality audit source or treatment hash drifted")
    manual_rows = _validate_audit_rows(root_path, manual, template, expected_rows, maps)
    target_rows, control_rows = _selected_rows(manual_rows)
    if len(target_rows) < 2:
        _write_feasibility_freeze(
            root_path,
            treatment,
            manual_path,
            len(target_rows),
            len(control_rows),
            f"fewer than 2 credible visual targets remain: {len(target_rows)}",
        )
        raise ValueError(f"fewer than 2 credible visual targets remain: {len(target_rows)}")
    if len(control_rows) < 2:
        _write_feasibility_freeze(
            root_path,
            treatment,
            manual_path,
            len(target_rows),
            len(control_rows),
            f"fewer than 2 visually eligible controls remain: {len(control_rows)}",
        )
        raise ValueError(f"fewer than 2 visually eligible controls remain: {len(control_rows)}")
    selected_rows = target_rows + control_rows
    signature = _bundle_signature(root_path, selected_rows, manual_path)
    if (root_path / "bundle-promotion.json").is_file():
        _resume_bundle_promotion(root_path, expected_signature=signature)
        if (root_path / "selection-freeze.json").is_file():
            return root_path / "selection-freeze.json"
    final_outputs = (
        root_path / "clips",
        root_path / "media-manifest.json",
        root_path / "provider-rows.json",
        root_path / "selection-freeze.json",
    )
    if any(path.exists() for path in final_outputs):
        raise ValueError("panel already has final artifacts; refusing to overwrite")
    clips = _stage_panel_clips(root_path, selected_rows, treatment, manual_path)
    clip_duration_by_index = {
        clip["audit_index"]: clip["clip_duration_seconds"] for clip in clips
    }
    provider_rows = [_provider_row(row) for row in selected_rows]
    # Bind the provider-safe duration to the ffprobe-measured clip duration;
    # seeking/codec muxing can differ from the requested window by a frame.
    for provider_row in provider_rows:
        provider_row["clip_duration_seconds"] = clip_duration_by_index[provider_row["audit_index"]]
    runner = _task1_module(_resolve(task1_runner or runner_path))
    provider_hash = runner.provider_rows_sha256(provider_rows)
    media_payload = {
        "schema_version": 1,
        "status": "frozen",
        "private": True,
        "rows": clips,
    }
    media_hash = _sha256_bytes(_serialized_json(media_payload))
    provider_payload = {"schema_version": 1, "status": "frozen", "rows": provider_rows}
    selected_ids = [row["audit_index"] for row in selected_rows]
    selection_payload = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "treatment_freeze_sha256": _sha256(root_path / "treatment-freeze.json"),
        "task1_treatment_sha256": treatment["task1_treatment_sha256"],
        "modality_audit_sha256": _sha256(manual_path),
        "modality_template_sha256": _sha256(template_path),
        "provider_rows_sha256": provider_hash,
        "media_manifest_sha256": media_hash,
        "selected_rows": selected_ids,
        "selected_row_metadata": [
            {
                "audit_index": row["audit_index"],
                "video_id": row["video_id"],
                "role": row["role"],
                "modality": row["modality"],
                "map_line_numbers": row["map_line_numbers"],
                "window_start_seconds": row["window_start_seconds"],
                "window_end_seconds": row["window_end_seconds"],
            }
            for row in selected_rows
        ],
        "target_count": len(target_rows),
        "control_count": len(control_rows),
        "planned_call_count": len(selected_rows) * 2,
        "expected_fps": [4.0, 12.0],
        "max_calls": 24,
        "compatibility_canary_pairs": treatment["compatibility_canary_pairs"],
        "decision_gate": treatment["decision_gate"],
        "exposure_reporting": _exposure_report(manual_rows, selected_rows),
    }
    if selection_payload["planned_call_count"] > 24:
        raise ValueError("planned call count exceeds 24-call maximum")
    _promote_bundle(
        root_path,
        selected_rows,
        clips,
        {
            "media-manifest.json": media_payload,
            "provider-rows.json": provider_payload,
            "selection-freeze.json": selection_payload,
        },
        signature,
    )
    return root_path / "selection-freeze.json"


def authorize(
    *,
    root: Path | str = DEFAULT_ROOT,
    runner_path: Path | str = TASK1_RUNNER,
    config_path: Path | str = TASK1_CONFIG,
    task1_runner: Path | str | None = None,
    task1_config: Path | str | None = None,
    manual_audit: Path | str | None = None,
    **_unused: Any,
) -> Path:
    """Write the provider authorization only for a valid frozen panel."""

    root_path = _root(root)
    treatment_path = root_path / "treatment-freeze.json"
    selection_path = root_path / "selection-freeze.json"
    provider_path = root_path / "provider-rows.json"
    media_path = root_path / "media-manifest.json"
    manual_path = _resolve(manual_audit, base=root_path) if manual_audit else root_path / "manual-modality-audit.json"
    required = (treatment_path, selection_path, provider_path, media_path, manual_path)
    if any(not path.is_file() for path in required):
        raise ValueError("all treatment, audit, selection, provider, and media freezes are required")
    treatment = _load_json(treatment_path)
    selection = _load_json(selection_path)
    provider_payload = _load_json(provider_path)
    media_payload = _load_json(media_path)
    if selection.get("status") != "frozen":
        raise ValueError("selection freeze must be frozen before authorization")
    if selection.get("compatibility_canary_pairs") != treatment.get("compatibility_canary_pairs"):
        raise ValueError("selection compatibility canary policy drifted before authorization")
    _validate_task1_treatment(
        treatment,
        runner_path=_resolve(task1_runner or runner_path),
        config_path=_resolve(task1_config or config_path),
    )
    if selection.get("task1_treatment_sha256") != treatment.get("task1_treatment_sha256"):
        raise ValueError("selection Task 1 treatment hash drifted before authorization")
    if selection.get("modality_audit_sha256") != _sha256(manual_path):
        raise ValueError("manual modality audit hash drifted before authorization")
    runner = _task1_module(_resolve(task1_runner or runner_path))
    provider_rows = _rows(provider_payload, "provider-rows.json")
    provider_hash = runner.provider_rows_sha256(provider_rows)
    if selection.get("provider_rows_sha256") != provider_hash:
        raise ValueError("provider rows hash drifted before authorization")
    media_hash = _sha256(media_path)
    treatment_hash = _sha256(treatment_path)
    selection_hash = _sha256(selection_path)
    if selection.get("treatment_freeze_sha256") != treatment_hash:
        raise ValueError("selection treatment hash drifted before authorization")
    if selection.get("media_manifest_sha256") != media_hash:
        raise ValueError("selection media manifest hash drifted before authorization")
    rows = _rows(media_payload, "media-manifest.json")
    if len(rows) != len(provider_rows):
        raise ValueError("provider/media panel row count drifted")
    target_count = selection.get("target_count")
    control_count = selection.get("control_count")
    planned = selection.get("planned_call_count")
    if not isinstance(target_count, int) or target_count < 2 or not isinstance(control_count, int) or control_count < 2:
        raise ValueError("provider authorization requires at least two targets and controls")
    if planned != len(provider_rows) * 2 or planned > 24:
        raise ValueError("provider authorization planned call count drifted")
    payload = {
        "schema_version": 1,
        "status": "authorized",
        "promotion_gate": "PASS",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "authorized_at": datetime.now(UTC).isoformat(),
        "decision_gate": treatment["decision_gate"],
        "treatment_freeze_sha256": treatment_hash,
        "selection_freeze_sha256": selection_hash,
        "modality_audit_sha256": _sha256(manual_path),
        "provider_rows_sha256": provider_hash,
        "media_manifest_sha256": media_hash,
        "planned_call_count": planned,
        "max_calls": 24,
        "compatibility_canary_pairs": treatment["compatibility_canary_pairs"],
        "target_count": target_count,
        "control_count": control_count,
        "provider_calls_made": 0,
        "exposure_reporting": selection.get("exposure_reporting", {}),
        "authorization_scope": "one attempt per selected row at 4fps and 12fps; no retry",
    }
    output = root_path / "provider-authorization.json"
    _write_once(output, payload)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze-treatment", "prepare-modality-audit", "freeze-panel", "authorize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--corrected-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--runner-path", type=Path, default=TASK1_RUNNER)
    parser.add_argument("--config-path", type=Path, default=TASK1_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "root": args.root,
        "source_protocol": args.source_protocol,
        "source_maps": args.source_maps,
        "corrected_audit": args.corrected_audit,
    }
    if args.phase == "freeze-treatment":
        output = freeze_treatment(**common, runner_path=args.runner_path, config_path=args.config_path)
    elif args.phase == "prepare-modality-audit":
        output = prepare_modality_audit(**common)
    elif args.phase == "freeze-panel":
        output = freeze_panel(**common, runner_path=args.runner_path, config_path=args.config_path)
    else:
        output = authorize(root=args.root, runner_path=args.runner_path, config_path=args.config_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
