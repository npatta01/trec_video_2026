#!/usr/bin/env python3
"""Run the immutable paired question-localized visual evidence probe.

Task 2 writes the treatment, selection, provider-row, media, and authorization
freezes consumed here.  This runner has one provider attempt per row/arm,
persists the raw response before parsing, and freezes every resulting byte
before any reference-aware analysis is allowed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping, Sequence


MODEL = "gemini-3.1-pro-preview"
EXPECTED_FPS = (4.0, 12.0)
MAX_CALLS = 24
COMPATIBILITY_CANARY_PAIRS = 1
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = SCRIPT_PATH.parents[1] / "configs" / "vqa" / "dense_question_localized_evidence_issue35.json"
EXPECTED_SETTINGS: dict[str, Any] = {
    "temperature": 0.0,
    "seed": 42,
    "thinking_level": "HIGH",
    "media_resolution": "HIGH",
    "candidate_count": 1,
    "max_output_tokens": 2048,
    "retry_count": 0,
}
EXPECTED_DECISION_GATE: dict[str, Any] = {
    "minimum_targets": 2,
    "maximum_targets": 8,
    "minimum_controls": 2,
    "maximum_controls": 4,
    "maximum_calls": 24,
    "requires_complete_terminal_pairs": True,
    "minimum_incremental_target_recoveries": 2,
    "minimum_12fps_target_recovery_fraction": 0.5,
    "maximum_unsupported_observations": 0,
    "maximum_12fps_control_regressions": 0,
    "requires_manual_inspection": True,
}
ROW_ARTIFACT_NAMES = (
    "request.json",
    "raw-response.txt",
    "metadata.json",
    "parsed.json",
    "terminal.json",
)
TERMINAL_STATUSES = {
    "passed",
    "invalid_json",
    "contract_error",
    "provider_error",
    "interrupted",
    "incomplete",
}
CallFunction = Callable[
    [bytes, str, str, float], tuple[str, Mapping[str, Any]]
]


PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question_interpretation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "requested_entity": {"type": "string", "minLength": 1},
                "requested_visual_fact": {"type": "string", "minLength": 1},
                "temporal_qualifier": {"type": "string", "minLength": 1},
            },
            "required": [
                "requested_entity",
                "requested_visual_fact",
                "temporal_qualifier",
            ],
        },
        "fact_observed": {"type": "boolean"},
        "observed_fact": {"type": "string"},
        "evidence_timestamps": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
        },
        "conflicts_or_uncertainty": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "needs_more_evidence": {"type": "boolean"},
    },
    "required": [
        "question_interpretation",
        "fact_observed",
        "observed_fact",
        "evidence_timestamps",
        "conflicts_or_uncertainty",
        "needs_more_evidence",
    ],
}


def _without_additional_properties(value: Any) -> Any:
    """Project the strict local schema to Gemini's provider-compatible schema."""

    if isinstance(value, Mapping):
        return {
            key: _without_additional_properties(child)
            for key, child in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [_without_additional_properties(child) for child in value]
    return value


PROVIDER_RESPONSE_SCHEMA: dict[str, Any] = _without_additional_properties(PROBE_SCHEMA)


_FORBIDDEN_PROVIDER_KEYS = frozenset(
    {
        "answer",
        "annotation",
        "audit",
        "audit_notes",
        "correct",
        "direct_answer",
        "expected_answer",
        "gold",
        "label",
        "manual_audit",
        "prediction",
        "reference",
        "reference_answer",
        "reviewer",
        "rationale",
        "score",
        "verdict",
    }
)
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _schema_sha256() -> str:
    return hashlib.sha256(_canonical(PROBE_SCHEMA)).hexdigest()


def _provider_schema_sha256() -> str:
    return hashlib.sha256(_canonical(PROVIDER_RESPONSE_SCHEMA)).hexdigest()


def _validate_config() -> None:
    config = _load_json(CONFIG_PATH)
    if not isinstance(config, Mapping):
        raise ValueError("runner config must be an object")
    if config.get("model") != MODEL:
        raise ValueError("runner config model does not match runner")
    if config.get("settings") != EXPECTED_SETTINGS:
        raise ValueError("runner config settings do not match runner")
    if config.get("expected_fps") != list(EXPECTED_FPS):
        raise ValueError("runner config expected FPS does not match runner")
    if config.get("max_calls") != MAX_CALLS:
        raise ValueError("runner config call cap does not match runner")
    if config.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("runner config compatibility canary policy does not match runner")
    if config.get("decision_gate") != EXPECTED_DECISION_GATE:
        raise ValueError("runner config decision gate does not match runner")


def _validator_sha256() -> str:
    """Hash validator source so Task 2 freezes validation semantics too."""

    source = inspect.getsource(validate_probe)
    return hashlib.sha256(source.encode()).hexdigest()


def _line_text(item: Mapping[str, Any], field: str) -> str:
    text = item.get("text", item.get("claim", item.get("description")))
    return _string(text, f"{field}.text")


def _normalise_map_lines(value: Any, field: str = "map_lines") -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    lines: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field}[{index}] must be an object")
        _reject_provider_fields(raw, f"{field}[{index}]")
        text = _line_text(raw, f"{field}[{index}]")
        line_number = raw.get("line_number", raw.get("index", index + 1))
        if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number < 1:
            raise ValueError(f"{field}[{index}].line_number must be a positive integer")
        line: dict[str, Any] = {"line_number": line_number, "text": text}
        for key in ("start_seconds", "end_seconds"):
            if key in raw:
                line[key] = _number(raw[key], f"{field}[{index}].{key}")
        lines.append(line)
    return lines


def _reject_provider_fields(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} has a non-text field name")
            normalized = key.casefold().replace("-", "_")
            if normalized in _FORBIDDEN_PROVIDER_KEYS:
                raise ValueError(f"provider-safe rows contain forbidden field: {field}.{key}")
            _reject_provider_fields(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_provider_fields(child, f"{field}[{index}]")


def _provider_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"provider row {index} must be an object")
        _reject_provider_fields(raw, f"provider row {index}")
        audit_index = raw.get("audit_index", raw.get("index"))
        if isinstance(audit_index, bool) or not isinstance(audit_index, int) or audit_index < 1:
            raise ValueError(f"provider row {index}.audit_index must be a positive integer")
        question = _string(raw.get("question"), f"provider row {index}.question")
        map_lines = _normalise_map_lines(raw.get("map_lines", raw.get("evidence")), f"provider row {index}.map_lines")
        item: dict[str, Any] = {
            "audit_index": audit_index,
            "question": question,
            "map_lines": map_lines,
        }
        # A stable source-row ID is safe navigation metadata.  It is retained
        # in the private digest but is never inserted into the prompt.
        if "video_id" in raw:
            item["video_id"] = _string(raw["video_id"], f"provider row {index}.video_id")
        for key in ("clip_start_seconds", "clip_end_seconds", "clip_duration_seconds"):
            if key in raw:
                item[key] = _number(raw[key], f"provider row {index}.{key}")
        projection.append(item)
    if len({item["audit_index"] for item in projection}) != len(projection):
        raise ValueError("provider rows contain duplicate audit_index")
    return projection


def provider_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered provider-safe projection, excluding all labels."""

    return hashlib.sha256(_canonical(_provider_projection(rows))).hexdigest()


def build_prompt(
    question: str,
    map_lines: Sequence[Mapping[str, Any]],
    clip_start_seconds: float,
    clip_end_seconds: float,
) -> str:
    """Build the frozen question-localized, evidence-only probe prompt."""

    question_text = _string(question, "question")
    start = _number(clip_start_seconds, "clip_start_seconds")
    end = _number(clip_end_seconds, "clip_end_seconds")
    if start < 0 or end <= start:
        raise ValueError("clip offsets must satisfy 0 <= start < end")
    lines = _normalise_map_lines(list(map_lines), "map_lines")
    evidence = "\n".join(
        f"[{line['line_number']}] {line['text']}" for line in lines
    )
    return f"""Inspect the local video clip first. The clip's source-video offset is {start:.3f} to {end:.3f} seconds. Timestamp every observation with a clip-relative timestamp from 0.000 through the clip duration; never use source-video timestamps in evidence_timestamps.

Answer only whether the requested visual fact in the question is directly observable in this clip. Use the frozen scene-map lines only as navigation. Do not answer a competition question and do not produce answer candidates or an answer candidate.

Do not use references, previous answers, labels, scores, audit verdicts, or world knowledge. Do not guess or infer purpose, cause, identity, exact speech, OCR text, counts, or any fact not visibly supported in this clip. Do not rely on audio to assert an exact speech fact. Do not describe what should be present.

Return exactly the JSON fields in the supplied schema. Interpret the requested entity, requested visual fact, and temporal qualifier. If and only if the fact is visibly supported, set fact_observed to true, give one concise observed_fact, and provide one or more clip-relative evidence_timestamps. If the fact is not supported or is ambiguous, set fact_observed to false, observed_fact to an empty string, evidence_timestamps to an empty list, conflicts_or_uncertainty to a non-empty explanation, and needs_more_evidence to true. Do not return references, previous predictions, multiple candidates, purpose, exact speech, OCR, or counts.

Question:
{question_text}

Frozen scene-map lines:
{evidence}"""


def _prompt_template_sha256() -> str:
    return hashlib.sha256(
        build_prompt("{{QUESTION}}", [{"line_number": 1, "text": "{{MAP_LINE}}"}], 1.25, 3.5).encode()
    ).hexdigest()


def validate_probe(
    payload: Any,
    clip_duration_seconds: float,
    *,
    nominal_duration_seconds: float | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Validate the strict evidence-recovery response and normalize it."""

    if not isinstance(payload, Mapping):
        raise ValueError("probe response must be an object")
    required = set(PROBE_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("probe response fields are invalid")
    interpretation_payload = payload["question_interpretation"]
    interpretation_fields = {
        "requested_entity",
        "requested_visual_fact",
        "temporal_qualifier",
    }
    if not isinstance(interpretation_payload, Mapping) or set(interpretation_payload) != interpretation_fields:
        raise ValueError("question_interpretation fields are invalid")
    interpretation = {
        key: _string(interpretation_payload[key], f"question_interpretation.{key}")
        for key in sorted(interpretation_fields)
    }
    fact_observed = payload["fact_observed"]
    if not isinstance(fact_observed, bool):
        raise ValueError("fact_observed must be boolean")
    observed_fact = payload["observed_fact"]
    if not isinstance(observed_fact, str):
        raise ValueError("observed_fact must be text")
    if fact_observed and not observed_fact.strip():
        raise ValueError("observed_fact is required when fact_observed is true")
    if not fact_observed and observed_fact != "":
        raise ValueError("observed_fact must be empty when fact_observed is false")
    timestamp_payload = payload["evidence_timestamps"]
    if not isinstance(timestamp_payload, list):
        raise ValueError("evidence_timestamps must be a list")
    duration = _number(clip_duration_seconds, "clip_duration_seconds")
    if duration <= 0:
        raise ValueError("clip_duration_seconds must be positive")
    nominal = duration if nominal_duration_seconds is None else _number(
        nominal_duration_seconds, "nominal_duration_seconds"
    )
    if nominal <= 0:
        raise ValueError("nominal_duration_seconds must be positive")
    if nominal_duration_seconds is not None:
        if fps is None or _number(fps, "fps") <= 0:
            raise ValueError("fps is required for nominal endpoint validation")
        frame = Decimal(1) / Decimal(str(_number(fps, "fps")))
        nominal_decimal = Decimal(str(nominal))
        duration_decimal = Decimal(str(duration))
        if duration > nominal:
            raise ValueError("encoded duration exceeds nominal duration")
        shortfall = nominal_decimal - duration_decimal
        if shortfall > frame:
            raise ValueError("encoding shortfall exceeds one frame")
    else:
        frame = 0.0
        shortfall = 0.0
    timestamps: list[float] = []
    for index, value in enumerate(timestamp_payload):
        timestamp = _number(value, f"evidence_timestamps[{index}]")
        timestamp_decimal = Decimal(str(timestamp))
        if timestamp < 0 or timestamp_decimal > Decimal(str(nominal)):
            raise ValueError(f"evidence timestamp is outside nominal duration: {timestamp}")
        if timestamp > duration and (
            nominal_duration_seconds is None
            or timestamp_decimal - duration_decimal > shortfall
            or timestamp_decimal - duration_decimal > frame
        ):
            raise ValueError(f"evidence timestamp exceeds encoded duration shortfall: {timestamp}")
        timestamps.append(timestamp)
    if timestamps != sorted(set(timestamps)):
        raise ValueError("evidence_timestamps must be sorted and unique")
    uncertainty_payload = payload["conflicts_or_uncertainty"]
    if not isinstance(uncertainty_payload, list):
        raise ValueError("conflicts_or_uncertainty must be a list")
    uncertainty = [
        _string(value, f"conflicts_or_uncertainty[{index}]")
        for index, value in enumerate(uncertainty_payload)
    ]
    needs_more_evidence = payload["needs_more_evidence"]
    if not isinstance(needs_more_evidence, bool):
        raise ValueError("needs_more_evidence must be boolean")
    if fact_observed:
        if not timestamps:
            raise ValueError("observed fact requires an evidence timestamp")
        if needs_more_evidence:
            raise ValueError("supported observed fact cannot need more evidence")
    else:
        if timestamps:
            raise ValueError("unobserved fact must not include timestamps")
        if not uncertainty:
            raise ValueError("unobserved fact requires conflicts_or_uncertainty")
        if not needs_more_evidence:
            raise ValueError("unobserved fact requires needs_more_evidence")
    return {
        "question_interpretation": interpretation,
        "fact_observed": fact_observed,
        "observed_fact": observed_fact,
        "evidence_timestamps": timestamps,
        "conflicts_or_uncertainty": uncertainty,
        "needs_more_evidence": needs_more_evidence,
    }


def _nominal_duration(media: Mapping[str, Any]) -> float:
    """Derive the requested window without binary float subtraction drift."""
    return float(
        Decimal(str(media["clip_end_seconds"]))
        - Decimal(str(media["clip_start_seconds"]))
    )


def _rows_payload(payload: Any, filename: str) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{filename} must contain a non-empty rows list")
    return [dict(row) if isinstance(row, Mapping) else row for row in rows]


def _index_by_audit(rows: Sequence[Mapping[str, Any]], filename: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{filename} row {index} must be an object")
        value = row.get("audit_index", row.get("index"))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{filename} row {index}.audit_index is invalid")
        if value in result:
            raise ValueError(f"{filename} contains duplicate audit_index: {value}")
        result[value] = dict(row)
    return result


def _media_row(row: Mapping[str, Any], root: Path) -> dict[str, Any]:
    audit_index = row.get("audit_index", row.get("index"))
    if isinstance(audit_index, bool) or not isinstance(audit_index, int) or audit_index < 1:
        raise ValueError("media row audit_index is invalid")
    path_value = row.get("clip_path", row.get("path", row.get("media_path")))
    path = Path(_string(path_value, f"media row {audit_index}.clip_path"))
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"clip is missing: {path}")
    declared_sha = row.get("clip_sha256", row.get("sha256", row.get("media_sha256")))
    declared_sha = _string(declared_sha, f"media row {audit_index}.clip_sha256")
    actual_sha = _sha256(path)
    if declared_sha != actual_sha:
        raise ValueError(f"clip hash mismatch for audit_index {audit_index}")
    mime_type = _string(row.get("mime_type", "video/mp4"), f"media row {audit_index}.mime_type")
    start = _number(row.get("clip_start_seconds", row.get("start_seconds")), f"media row {audit_index}.clip_start_seconds")
    end = _number(row.get("clip_end_seconds", row.get("end_seconds")), f"media row {audit_index}.clip_end_seconds")
    duration = _number(row.get("clip_duration_seconds", row.get("duration_seconds", end - start)), f"media row {audit_index}.clip_duration_seconds")
    if start < 0 or end <= start or duration <= 0 or end - start <= 0:
        raise ValueError(f"media offsets are invalid for audit_index {audit_index}")
    arms = row.get("arms", {})
    if not isinstance(arms, Mapping):
        raise ValueError(f"media row {audit_index}.arms must be an object")
    for fps in EXPECTED_FPS:
        arm_name = _arm_name(fps)
        arm = arms.get(arm_name, arms.get(str(fps)))
        if not isinstance(arm, Mapping):
            raise ValueError(f"media row {audit_index} arm {arm_name} is missing")
        arm_fps = _number(arm.get("fps", fps), f"media row {audit_index} arm {arm_name}.fps")
        if arm_fps != fps:
            raise ValueError(f"FPS drift for audit_index {audit_index}: expected {fps}")
        arm_sha = arm.get("clip_sha256", arm.get("sha256", declared_sha))
        if arm_sha != declared_sha:
            raise ValueError(f"paired arms are not byte-identical for audit_index {audit_index}")
    role = _string(row.get("role", "unknown"), f"media row {audit_index}.role")
    return {
        "audit_index": audit_index,
        "role": role,
        "clip_path": path,
        "clip_sha256": declared_sha,
        "mime_type": mime_type,
        "clip_start_seconds": start,
        "clip_end_seconds": end,
        "clip_duration_seconds": duration,
    }


def _arm_name(fps: float) -> str:
    return f"{int(fps)}fps"


def _role_counts(
    selection: Mapping[str, Any],
    media_by_index: Mapping[int, Mapping[str, Any]],
    raw_role_by_index: Mapping[int, str],
) -> dict[str, int]:
    metadata_roles: dict[int, str] = {}
    metadata_rows = selection.get("selected_row_metadata", [])
    if isinstance(metadata_rows, list):
        for item in metadata_rows:
            if (
                isinstance(item, Mapping)
                and isinstance(item.get("audit_index"), int)
                and isinstance(item.get("role"), str)
            ):
                metadata_roles[item["audit_index"]] = item["role"]
    counts = {"target": 0, "control": 0}
    for audit_index, media in media_by_index.items():
        role = str(media.get("role", "unknown"))
        if role == "unknown":
            role = metadata_roles.get(audit_index, raw_role_by_index.get(audit_index, "unknown"))
        if role not in counts:
            raise ValueError(f"selected row {audit_index} has unknown role: {role}")
        counts[role] += 1
    return counts


def _validate_role_counts(role_counts: Mapping[str, int]) -> None:
    if not (EXPECTED_DECISION_GATE["minimum_targets"] <= role_counts["target"] <= EXPECTED_DECISION_GATE["maximum_targets"]):
        raise ValueError(
            "authorized panel must contain 2-8 target rows; "
            f"found {role_counts['target']}"
        )
    if not (EXPECTED_DECISION_GATE["minimum_controls"] <= role_counts["control"] <= EXPECTED_DECISION_GATE["maximum_controls"]):
        raise ValueError(
            "authorized panel must contain 2-4 control rows; "
            f"found {role_counts['control']}"
        )


def _validate_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    required = (
        "treatment-freeze.json",
        "selection-freeze.json",
        "provider-rows.json",
        "media-manifest.json",
        "provider-authorization.json",
    )
    paths = {name: root / name for name in required}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing required input: {path}")
    treatment = _load_json(paths["treatment-freeze.json"])
    selection = _load_json(paths["selection-freeze.json"])
    provider_payload = _load_json(paths["provider-rows.json"])
    media_payload = _load_json(paths["media-manifest.json"])
    authorization = _load_json(paths["provider-authorization.json"])
    if not isinstance(treatment, Mapping) or not isinstance(selection, Mapping) or not isinstance(authorization, Mapping):
        raise ValueError("treatment, selection, and authorization freezes must be objects")
    _validate_config()
    if treatment.get("status") != "frozen":
        raise ValueError("treatment freeze must have status=frozen")
    if treatment.get("model") not in {None, MODEL}:
        raise ValueError("treatment model does not match runner")
    if treatment.get("settings") != EXPECTED_SETTINGS:
        raise ValueError("treatment settings do not match runner")
    if treatment.get("expected_fps") != list(EXPECTED_FPS):
        raise ValueError("treatment expected FPS does not match runner")
    if treatment.get("max_calls") != MAX_CALLS:
        raise ValueError("treatment call cap does not match runner")
    if treatment.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("treatment compatibility canary policy does not match runner")
    if treatment.get("decision_gate") != EXPECTED_DECISION_GATE:
        raise ValueError("treatment decision gate does not match frozen decision rule")
    expected_hashes = {
        "runner_sha256": _sha256(SCRIPT_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "prompt_template_sha256": _prompt_template_sha256(),
        "response_schema_sha256": _schema_sha256(),
        "provider_response_schema_sha256": _provider_schema_sha256(),
        "validator_sha256": _validator_sha256(),
    }
    for field, expected in expected_hashes.items():
        if treatment.get(field) != expected:
            raise ValueError(f"treatment {field} does not match current bytes")
    provider_rows = _rows_payload(provider_payload, "provider-rows.json")
    raw_role_by_index: dict[int, str] = {}
    for row in provider_rows:
        if not isinstance(row, Mapping):
            continue
        row_index = row.get("audit_index", row.get("index"))
        if isinstance(row_index, int) and not isinstance(row_index, bool) and isinstance(row.get("role"), str):
            raw_role_by_index[row_index] = row["role"]
    projection = _provider_projection(provider_rows)
    provider_by_index = {row["audit_index"]: row for row in projection}
    media_raw = _rows_payload(media_payload, "media-manifest.json")
    media_by_index = {row["audit_index"]: _media_row(row, root) for row in media_raw}
    if set(provider_by_index) != set(media_by_index):
        raise ValueError("provider rows and media manifest audit indexes do not match")
    provider_sha = provider_rows_sha256(provider_rows)
    media_sha = _sha256(paths["media-manifest.json"])
    treatment_sha = _sha256(paths["treatment-freeze.json"])
    selection_sha = _sha256(paths["selection-freeze.json"])
    if selection.get("status") != "frozen":
        raise ValueError("selection freeze must have status=frozen")
    if selection.get("decision_gate") != EXPECTED_DECISION_GATE:
        raise ValueError("selection decision gate does not match frozen rule")
    if selection.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("selection compatibility canary policy does not match runner")
    if selection.get("treatment_freeze_sha256") != treatment_sha:
        raise ValueError("selection treatment hash does not match current bytes")
    if selection.get("provider_rows_sha256") != provider_sha:
        raise ValueError("selection provider rows hash does not match current bytes")
    if selection.get("media_manifest_sha256") != media_sha:
        raise ValueError("selection media manifest hash does not match current bytes")
    selection_role_by_index: dict[int, str] = {}
    metadata_rows = selection.get("selected_row_metadata", [])
    if isinstance(metadata_rows, list):
        for item in metadata_rows:
            if isinstance(item, Mapping) and isinstance(item.get("audit_index"), int) and isinstance(item.get("role"), str):
                selection_role_by_index[item["audit_index"]] = item["role"]
    selected = selection.get("selected_rows")
    if not isinstance(selected, list):
        raise ValueError("selection selected_rows must be a list")
    selected_ids = [item if isinstance(item, (int, str)) else item.get("audit_index") if isinstance(item, Mapping) else None for item in selected]
    expected_selected = [row.get("video_id", row["audit_index"]) for row in projection]
    if selected_ids != expected_selected and selected_ids != list(provider_by_index):
        raise ValueError("selection selected_rows do not match provider row order")
    planned_calls = len(provider_by_index) * len(EXPECTED_FPS)
    if planned_calls > MAX_CALLS:
        raise ValueError("planned call count exceeds frozen call cap")
    if selection.get("planned_call_count") != planned_calls:
        raise ValueError("selection planned call count drift")
    role_counts = _role_counts(selection, media_by_index, raw_role_by_index)
    if selection.get("target_count") != role_counts["target"]:
        raise ValueError("selection target count does not match selected rows")
    if selection.get("control_count") != role_counts["control"]:
        raise ValueError("selection control count does not match selected rows")
    _validate_role_counts(role_counts)
    if authorization.get("status") != "authorized":
        raise ValueError("provider authorization status must be exactly authorized")
    if authorization.get("promotion_gate") != "PASS":
        raise ValueError("provider authorization promotion gate must be PASS")
    if authorization.get("decision_gate") != EXPECTED_DECISION_GATE:
        raise ValueError("provider authorization decision gate does not match frozen rule")
    if authorization.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("provider authorization compatibility canary policy does not match runner")
    auth_expected = {
        "treatment_freeze_sha256": treatment_sha,
        "selection_freeze_sha256": selection_sha,
        "provider_rows_sha256": provider_sha,
        "media_manifest_sha256": media_sha,
    }
    for field, expected in auth_expected.items():
        if authorization.get(field) != expected:
            raise ValueError(f"provider authorization {field} does not match current bytes")
    if authorization.get("target_count") != role_counts["target"]:
        raise ValueError("provider authorization target count does not match selected rows")
    if authorization.get("control_count") != role_counts["control"]:
        raise ValueError("provider authorization control count does not match selected rows")
    if authorization.get("planned_call_count") != planned_calls:
        raise ValueError("provider authorization planned call count drift")
    if authorization.get("max_calls") != MAX_CALLS:
        raise ValueError("provider authorization call cap drift")
    provider_calls_made = authorization.get("provider_calls_made")
    if (
        isinstance(provider_calls_made, bool)
        or not isinstance(provider_calls_made, int)
        or provider_calls_made != 0
    ):
        raise ValueError("provider authorization provider_calls_made must be exactly 0")
    rows: list[dict[str, Any]] = []
    for projection_row in projection:
        audit_index = projection_row["audit_index"]
        media = media_by_index[audit_index]
        for key in ("clip_start_seconds", "clip_end_seconds", "clip_duration_seconds"):
            if key in projection_row and projection_row[key] != media[key]:
                raise ValueError(f"provider/media {key} drift for audit_index {audit_index}")
        if media["role"] == "unknown":
            media = {
                **media,
                "role": selection_role_by_index.get(
                    audit_index, raw_role_by_index.get(audit_index, "unknown")
                ),
            }
        rows.append({"provider": projection_row, "media": media})
    return rows, provider_by_index, media_by_index


def _provider_result(result: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("provider call must return (raw_text, metadata)")
    raw, metadata = result
    if not isinstance(raw, str):
        raise ValueError("provider raw response must be text")
    if not isinstance(metadata, Mapping):
        raise ValueError("provider metadata must be an object")
    return raw, dict(metadata)


def _call_provider(clip_bytes: bytes, mime_type: str, prompt: str, fps: float) -> tuple[str, dict[str, Any]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    video = types.Part(
        inline_data=types.Blob(data=clip_bytes, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=fps),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )
    started = time.monotonic()
    response = client.models.generate_content(
        model=MODEL,
        contents=[video, prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            seed=42,
            candidate_count=1,
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=PROVIDER_RESPONSE_SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    metadata: dict[str, Any] = {
        "elapsed_seconds": time.monotonic() - started,
        "response_text_sha256": _sha256_bytes(raw.encode()),
        "model_version": getattr(response, "model_version", None),
        "response_id": getattr(response, "response_id", None),
    }
    if usage is not None:
        for name in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(usage, name, None)
            if value is not None:
                metadata[name] = value
    return raw, metadata


def _arm_directory(root: Path, audit_index: int, role: str, fps: float) -> Path:
    if not _SAFE_ID_RE.fullmatch(role):
        raise ValueError(f"row role is not path-safe: {role}")
    return root / "rows" / f"{audit_index:02d}-{role}-{_arm_name(fps)}"


def _request_payload(
    provider: Mapping[str, Any],
    media: Mapping[str, Any],
    fps: float,
    prompt: str,
    canary_pair_index: int | None,
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "settings": EXPECTED_SETTINGS,
        "fps": fps,
        "compatibility_canary_pairs": COMPATIBILITY_CANARY_PAIRS,
        "canary_pair_index": canary_pair_index,
        "mime_type": media["mime_type"],
        "clip_sha256": media["clip_sha256"],
        "clip_start_seconds": media["clip_start_seconds"],
        "clip_end_seconds": media["clip_end_seconds"],
        "clip_duration_seconds": media["clip_duration_seconds"],
        "contents": [
            {
                "type": "video",
                "mime_type": media["mime_type"],
                "clip_sha256": media["clip_sha256"],
            },
            {"type": "text", "text": prompt},
        ],
        "question": provider["question"],
        "map_lines": provider["map_lines"],
        "prompt": prompt,
        "response_schema": PROVIDER_RESPONSE_SCHEMA,
    }


def _write_request(
    row_root: Path,
    provider: Mapping[str, Any],
    media: Mapping[str, Any],
    fps: float,
    canary_pair_index: int | None,
) -> str:
    prompt = build_prompt(
        provider["question"],
        provider["map_lines"],
        media["clip_start_seconds"],
        media["clip_end_seconds"],
    )
    _write_once(
        row_root / "request.json",
        _request_payload(provider, media, fps, prompt, canary_pair_index),
    )
    return prompt


def _canary_pair_index(task_index: int) -> int | None:
    canary_tasks = 2 * COMPATIBILITY_CANARY_PAIRS
    if task_index >= canary_tasks:
        return None
    return task_index // len(EXPECTED_FPS) + 1


def _ordered_terminal_records(
    root: Path, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Read ordered request/terminal artifacts and derive their statuses."""

    records: list[dict[str, Any]] = []
    for task_index, (item, fps) in enumerate(
        ( (item, fps) for item in rows for fps in EXPECTED_FPS )
    ):
        provider = item["provider"]
        media = item["media"]
        audit_index = provider["audit_index"]
        role = media["role"]
        row_root = _arm_directory(root, audit_index, role, fps)
        request_path = row_root / "request.json"
        terminal_path = row_root / "terminal.json"
        if not request_path.is_file() or not terminal_path.is_file():
            raise ValueError(f"row request/terminal artifacts are missing: {row_root.name}")
        request = _load_json(request_path)
        terminal = _load_json(terminal_path)
        if not isinstance(request, Mapping):
            raise ValueError(f"row request is not an object: {row_root.name}")
        if request.get("fps") != fps:
            raise ValueError(f"row request FPS ordering mismatch: {row_root.name}")
        if request.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
            raise ValueError(f"row request canary policy mismatch: {row_root.name}")
        if request.get("canary_pair_index") != _canary_pair_index(task_index):
            raise ValueError(f"row request canary ordering mismatch: {row_root.name}")
        if request.get("response_schema") != PROVIDER_RESPONSE_SCHEMA:
            raise ValueError(f"row request provider schema mismatch: {row_root.name}")
        if not isinstance(terminal, Mapping):
            raise ValueError(f"row terminal is not an object: {row_root.name}")
        status = terminal.get("status")
        if terminal.get("audit_index") != audit_index or terminal.get("fps") != fps:
            raise ValueError(f"terminal row identity mismatch: {row_root.name}")
        if terminal.get("role") != role:
            raise ValueError(f"terminal row role mismatch: {row_root.name}")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unknown terminal status: {row_root.name}")
        records.append(
            {
                "audit_index": audit_index,
                "role": role,
                "fps": fps,
                "status": status,
                "row_root": row_root,
            }
        )
    return records


def _derived_canary_status(records: Sequence[Mapping[str, Any]]) -> str:
    canary = list(records[: 2 * COMPATIBILITY_CANARY_PAIRS])
    if len(canary) != 2 * COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("ordered artifacts do not contain the complete canary pair")
    statuses = [item["status"] for item in canary]
    if any(status == "incomplete" for status in statuses):
        return "not_reached"
    return "passed" if all(status == "passed" for status in statuses) else "failed"


def _validate_execution_branch(
    records: Sequence[Mapping[str, Any]],
    *,
    planned: int,
    freeze_status: str,
    stop_reason: Any,
    call_count: Any,
    canary_pair_status: Any,
) -> None:
    """Validate branch declarations against ordered terminal artifacts."""

    completed = sum(item["status"] != "incomplete" for item in records)
    derived_canary = _derived_canary_status(records)
    if canary_pair_status != derived_canary:
        raise ValueError("predictions freeze canary status is not derived from terminal artifacts")
    if freeze_status == "frozen":
        if completed != planned or call_count != planned or derived_canary != "passed":
            raise ValueError("complete predictions freeze does not match ordered terminal artifacts")
        return
    if stop_reason == "canary_failure":
        if (
            completed != 2
            or call_count != 2
            or derived_canary != "failed"
            or any(item["status"] == "incomplete" for item in records[:2])
            or any(item["status"] != "incomplete" for item in records[2:])
        ):
            raise ValueError("canary failure freeze does not match exactly two canary terminals")
        return
    if stop_reason == "continuation_canary_failure":
        if (
            completed != 4
            or call_count != 4
            or any(item["status"] == "incomplete" for item in records[:4])
            or any(item["status"] != "incomplete" for item in records[4:])
            or all(item["status"] == "passed" for item in records[2:4])
        ):
            raise ValueError("continuation canary freeze does not match four combined terminals")
        return
    if freeze_status == "frozen" and len(records) == 24:
        if any(item["status"] != "passed" for item in records[2:4]):
            raise ValueError("complete continuation freeze requires both fresh canary arms")
    if call_count != completed:
        raise ValueError("incomplete predictions freeze call count does not match terminals")


def _write_incomplete_freeze(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    call_count: int,
    error: BaseException,
    *,
    stop_reason: str = "interrupted",
    interrupted: bool = True,
) -> None:
    """Persist every pending arm after a terminal stop, without retrying."""

    planned = len(rows) * len(EXPECTED_FPS)
    ordered_records = _ordered_terminal_records(root, rows)
    derived_call_count = sum(item["status"] != "incomplete" for item in ordered_records)
    if call_count != derived_call_count:
        raise ValueError("provider call count does not match ordered terminal artifacts")
    derived_canary_status = _derived_canary_status(ordered_records)
    _validate_execution_branch(
        ordered_records,
        planned=planned,
        freeze_status="incomplete",
        stop_reason=stop_reason,
        call_count=derived_call_count,
        canary_pair_status=(
            derived_canary_status if stop_reason != "canary_failure" else "failed"
        ),
    )
    frozen_rows: list[dict[str, Any]] = []
    completed = 0
    passed = 0
    for item in rows:
        audit_index = item["provider"]["audit_index"]
        role = item["media"]["role"]
        for fps in EXPECTED_FPS:
            row_root = _arm_directory(root, audit_index, role, fps)
            terminal_path = row_root / "terminal.json"
            if not terminal_path.is_file():
                raise ValueError(f"stopped row has no terminal state: {row_root.name}")
            terminal = _load_json(terminal_path)
            status = terminal.get("status")
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"stopped row has unknown status: {row_root.name}")
            if status != "incomplete":
                completed += 1
            if status == "passed":
                passed += 1
            entries = {path.name for path in row_root.iterdir()}
            names = {path.name for path in row_root.iterdir() if path.is_file()}
            required = {"request.json", "metadata.json", "terminal.json"}
            if status in {"passed", "invalid_json", "contract_error"}:
                required.add("raw-response.txt")
            if status == "passed":
                required.add("parsed.json")
            if entries != names or not required.issubset(names):
                raise ValueError(f"stopped row artifacts are incomplete: {row_root.name}")
            frozen_rows.append(
                {
                    "audit_index": audit_index,
                    "role": role,
                    "fps": fps,
                    "status": status,
                    "artifacts": {name: _sha256(row_root / name) for name in sorted(names)},
                }
            )
    input_hashes = _freeze_inputs(root)
    freeze = {
        "schema_version": 1,
        "status": "incomplete",
        "interrupted": interrupted,
        "stop_reason": stop_reason,
        "interruption_type": type(error).__name__,
        "interruption_message": str(error),
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "model": MODEL,
        "expected_fps": list(EXPECTED_FPS),
        "compatibility_canary_pairs": COMPATIBILITY_CANARY_PAIRS,
        "canary_pair_status": derived_canary_status,
        "decision_gate": EXPECTED_DECISION_GATE,
        "frozen_at": datetime.now(UTC).isoformat(),
        "row_count": planned,
        "planned_call_count": planned,
        "completed_count": completed,
        "call_count": call_count,
        "passed_count": passed,
        "failed_count": planned - passed,
        **input_hashes,
        "runner_sha256": _sha256(SCRIPT_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "prompt_template_sha256": _prompt_template_sha256(),
        "response_schema_sha256": _schema_sha256(),
        "provider_response_schema_sha256": _provider_schema_sha256(),
        "validator_sha256": _validator_sha256(),
        "rows": frozen_rows,
    }
    _write_once(root / "predictions-freeze.json", freeze)
    _verify_freeze_contents(root, freeze)


def _write_pending_incomplete(
    root: Path,
    tasks: Sequence[tuple[Mapping[str, Any], float]],
    start_index: int,
    reason: str,
) -> None:
    for pending_index, (pending_item, pending_fps) in enumerate(
        tasks[start_index:], start=start_index
    ):
        pending_provider = pending_item["provider"]
        pending_media = pending_item["media"]
        audit_index = pending_provider["audit_index"]
        role = pending_media["role"]
        pending_root = _arm_directory(root, audit_index, role, pending_fps)
        _write_request(
            pending_root,
            pending_provider,
            pending_media,
            pending_fps,
            _canary_pair_index(pending_index),
        )
        _write_once(
            pending_root / "metadata.json",
            {"status": "incomplete", "reason": reason},
        )
        _write_once(
            pending_root / "terminal.json",
            {
                "audit_index": audit_index,
                "role": role,
                "fps": pending_fps,
                "status": "incomplete",
                "reason": reason,
            },
        )


def run(root: Path, call: CallFunction | None = None) -> None:
    """Run one immutable provider attempt for each selected row and FPS arm."""

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"run root does not exist: {root}")
    continuation = (root / "continuation-authorization.json").exists()
    if continuation:
        validate_continuation_authorization(root)
    if (root / "predictions-freeze.json").exists() or ((root / "rows").exists() and not continuation):
        raise ValueError("output already has artifacts; refusing to overwrite")
    rows, _provider_by_index, _media_by_index = _validate_inputs(root)
    provider = call or _call_provider
    planned = len(rows) * len(EXPECTED_FPS)
    if planned > MAX_CALLS:
        raise ValueError("planned call count exceeds MAX_CALLS")
    calls = 2 if continuation else 0
    tasks = [(item, fps) for item in rows for fps in EXPECTED_FPS]
    canary_statuses: list[str] = []
    start_index = 2 if continuation else 0
    response_ids: set[str] = set()
    if continuation:
        auth = _load_json(root / "continuation-authorization.json")
        for item in auth.get("imported_prefix", []):
            response_id = item.get("source_response_id")
            if not isinstance(response_id, str) or not response_id.strip():
                raise ValueError("continuation imported response ID is missing")
            if response_id in response_ids:
                raise ValueError("duplicate imported provider response ID")
            response_ids.add(response_id)

    def stop_after_failed_canary(task_index: int) -> bool:
        canary_end = (start_index + 2 - 1) if continuation else (2 * COMPATIBILITY_CANARY_PAIRS - 1)
        if task_index != canary_end:
            return False
        if continuation:
            fresh = []
            for pending_item, pending_fps in tasks[start_index : start_index + 2]:
                pending_root = _arm_directory(root, pending_item["provider"]["audit_index"], pending_item["media"]["role"], pending_fps)
                fresh.append(_load_json(pending_root / "terminal.json").get("status"))
            if all(status == "passed" for status in fresh):
                return False
        elif all(status == "passed" for status in canary_statuses):
            return False
        stop_error = RuntimeError("compatibility canary pair did not pass")
        _write_pending_incomplete(
            root,
            tasks,
            task_index + 1,
            "compatibility canary pair failed before provider attempt",
        )
        _write_incomplete_freeze(
            root,
            rows,
            calls,
            stop_error,
            stop_reason="continuation_canary_failure" if continuation else "canary_failure",
            interrupted=False,
        )
        return True

    for task_index, (item, fps) in enumerate(tasks[start_index:], start=start_index):
        provider_row = item["provider"]
        media = item["media"]
        audit_index = provider_row["audit_index"]
        role = _string(media["role"], f"media row {audit_index}.role")
        clip_path = media["clip_path"]
        clip_bytes = clip_path.read_bytes()
        if _sha256_bytes(clip_bytes) != media["clip_sha256"]:
            raise ValueError(f"clip hash changed before call for audit_index {audit_index}")
        row_root = _arm_directory(root, audit_index, role, fps)
        prompt = _write_request(
            row_root,
            provider_row,
            media,
            fps,
            _canary_pair_index(task_index),
        )
        calls += 1
        try:
            raw, metadata = _provider_result(provider(clip_bytes, media["mime_type"], prompt, fps))
            if continuation:
                response_id = metadata.get("response_id")
                if not isinstance(response_id, str) or not response_id.strip():
                    raise ValueError("continuation provider response ID is required")
                if response_id in response_ids:
                    raise ValueError("duplicate continuation provider response ID")
                response_ids.add(response_id)
        except BaseException as error:
            if isinstance(error, Exception):
                _write_once(row_root / "metadata.json", {"error_type": type(error).__name__, "error": str(error)})
                _write_once(
                    row_root / "terminal.json",
                    {"audit_index": audit_index, "role": role, "fps": fps, "status": "provider_error", "error_type": type(error).__name__},
                )
                if _canary_pair_index(task_index) is not None:
                    canary_statuses.append("provider_error")
                if stop_after_failed_canary(task_index):
                    return
                continue
            _write_once(row_root / "metadata.json", {"error_type": type(error).__name__, "error": str(error), "interrupted": True})
            _write_once(
                row_root / "terminal.json",
                {"audit_index": audit_index, "role": role, "fps": fps, "status": "interrupted", "error_type": type(error).__name__, "error": str(error)},
            )
            _write_pending_incomplete(
                root,
                tasks,
                task_index + 1,
                "run interrupted before provider attempt",
            )
            _write_incomplete_freeze(root, rows, calls, error)
            raise
        _write_text_once(row_root / "raw-response.txt", raw)
        metadata = {**metadata, "raw_response_sha256": _sha256_bytes(raw.encode())}
        _write_once(row_root / "metadata.json", metadata)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            _write_once(
                row_root / "terminal.json",
                {"audit_index": audit_index, "role": role, "fps": fps, "status": "invalid_json", "error_type": type(error).__name__, "error": str(error), "raw_response_sha256": _sha256_bytes(raw.encode())},
            )
            if _canary_pair_index(task_index) is not None:
                canary_statuses.append("invalid_json")
            if stop_after_failed_canary(task_index):
                return
            continue
        try:
            parsed = validate_probe(
                payload,
                media["clip_duration_seconds"],
                nominal_duration_seconds=_nominal_duration(media),
                fps=fps,
            )
        except ValueError as error:
            _write_once(
                row_root / "terminal.json",
                {"audit_index": audit_index, "role": role, "fps": fps, "status": "contract_error", "error_type": type(error).__name__, "error": str(error), "raw_response_sha256": _sha256_bytes(raw.encode())},
            )
            if _canary_pair_index(task_index) is not None:
                canary_statuses.append("contract_error")
            if stop_after_failed_canary(task_index):
                return
            continue
        _write_once(row_root / "parsed.json", parsed)
        _write_once(
            row_root / "terminal.json",
            {"audit_index": audit_index, "role": role, "fps": fps, "status": "passed", "raw_response_sha256": _sha256_bytes(raw.encode()), **({"response_id": metadata.get("response_id")} if continuation else {})},
        )
        if _canary_pair_index(task_index) is not None:
            canary_statuses.append("passed")
        if stop_after_failed_canary(task_index):
            return
    if calls != planned:
        raise ValueError("provider call count drift")


def _freeze_inputs(root: Path) -> dict[str, Any]:
    _validate_inputs(root)
    result = {
        "treatment_freeze_sha256": _sha256(root / "treatment-freeze.json"),
        "selection_freeze_sha256": _sha256(root / "selection-freeze.json"),
        "provider_rows_sha256": _sha256(root / "provider-rows.json"),
        "media_manifest_sha256": _sha256(root / "media-manifest.json"),
        "provider_authorization_sha256": _sha256(root / "provider-authorization.json"),
    }
    continuation = root / "continuation-authorization.json"
    replay = root / "canary-replay-freeze.json"
    if continuation.is_file() or replay.is_file():
        if not continuation.is_file() or not replay.is_file():
            raise ValueError("continuation freeze inputs are incomplete")
        auth = _load_json(continuation)
        replay_payload = _load_json(replay)
        source_root = Path(str(replay_payload.get("source_root", "")))
        source_freeze = source_root / "predictions-freeze.json"
        if not source_freeze.is_file():
            raise ValueError("continuation source freeze is missing")
        result.update({
            "continuation_authorization_sha256": _sha256(continuation),
            "canary_replay_freeze_sha256": _sha256(replay),
            "source_predictions_freeze_sha256": _sha256(source_freeze),
            "prior_provider_calls": auth.get("prior_provider_calls"),
            "combined_call_maximum": auth.get("combined_call_maximum"),
            "remaining_new_call_maximum": auth.get("remaining_new_call_maximum"),
            "fresh_canary_positions": [3, 4],
        })
    return result


def _verify_freeze_contents(root: Path, freeze: Mapping[str, Any]) -> None:
    rows, _provider_by_index, _media_by_index = _validate_inputs(root)
    ordered_records = _ordered_terminal_records(root, rows)
    expected_inputs = _freeze_inputs(root)
    for key, value in expected_inputs.items():
        if freeze.get(key) != value:
            raise ValueError(f"predictions freeze does not bind current {key}")
    expected_metadata = {
        "runner_sha256": _sha256(SCRIPT_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "prompt_template_sha256": _prompt_template_sha256(),
        "response_schema_sha256": _schema_sha256(),
        "provider_response_schema_sha256": _provider_schema_sha256(),
        "validator_sha256": _validator_sha256(),
    }
    for key, value in expected_metadata.items():
        if freeze.get(key) != value:
            raise ValueError(f"predictions freeze does not bind current {key}")
    planned = len(rows) * len(EXPECTED_FPS)
    status_value = freeze.get("status")
    if status_value not in {"frozen", "incomplete"}:
        raise ValueError("predictions freeze status is invalid")
    if freeze.get("decision_gate") != EXPECTED_DECISION_GATE:
        raise ValueError("predictions freeze decision gate does not match frozen rule")
    if freeze.get("compatibility_canary_pairs") != COMPATIBILITY_CANARY_PAIRS:
        raise ValueError("predictions freeze compatibility canary policy drifted")
    if freeze.get("row_count") != planned:
        raise ValueError("predictions freeze row count is inconsistent")
    frozen_rows = freeze.get("rows")
    if not isinstance(frozen_rows, list) or len(frozen_rows) != planned:
        raise ValueError("predictions freeze rows are inconsistent")
    expected_order = [
        (item["audit_index"], item["fps"], item["status"])
        for item in ordered_records
    ]
    actual_order = [
        (item.get("audit_index"), item.get("fps"), item.get("status"))
        for item in frozen_rows
        if isinstance(item, Mapping)
    ]
    if actual_order != expected_order:
        raise ValueError("predictions freeze row order does not match ordered artifacts")
    expected_keys: set[tuple[int, float]] = set()
    passed = 0
    completed = 0
    interrupted = False
    for item in rows:
        audit_index = item["provider"]["audit_index"]
        role = item["media"]["role"]
        for fps in EXPECTED_FPS:
            expected_keys.add((audit_index, fps))
            row_root = _arm_directory(root, audit_index, role, fps)
            terminal_path = row_root / "terminal.json"
            if not terminal_path.is_file():
                raise ValueError(f"row has no terminal state: {row_root.name}")
            terminal = _load_json(terminal_path)
            status = terminal.get("status")
            if terminal.get("audit_index") != audit_index or terminal.get("fps") != fps:
                raise ValueError(f"terminal row identity mismatch: {row_root.name}")
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"unknown terminal status: {row_root.name}")
            if status != "incomplete":
                completed += 1
            if status == "interrupted":
                interrupted = True
            if status == "passed":
                passed += 1
            required = {"request.json", "metadata.json", "terminal.json"}
            if status in {"passed", "invalid_json", "contract_error"}:
                required.add("raw-response.txt")
            if status == "passed":
                required.add("parsed.json")
                validate_probe(
                    _load_json(row_root / "parsed.json"),
                    item["media"]["clip_duration_seconds"],
                    nominal_duration_seconds=_nominal_duration(item["media"]),
                    fps=fps,
                )
            entries = {path.name for path in row_root.iterdir()}
            names = {path.name for path in row_root.iterdir() if path.is_file()}
            if entries != names or not required.issubset(names) or not names.issubset(set(ROW_ARTIFACT_NAMES)):
                raise ValueError(f"row artifact set is invalid: {row_root.name}")
            if status == "passed":
                validate_probe(
                    _load_json(row_root / "parsed.json"),
                    item["media"]["clip_duration_seconds"],
                    nominal_duration_seconds=_nominal_duration(item["media"]),
                    fps=fps,
                )
            frozen = next((value for value in frozen_rows if isinstance(value, Mapping) and value.get("audit_index") == audit_index and value.get("fps") == fps), None)
            if frozen is None:
                raise ValueError(f"predictions freeze row is missing: {row_root.name}")
            if frozen.get("status") != status or frozen.get("role") != role:
                raise ValueError(f"predictions freeze row identity/status mismatch: {row_root.name}")
            artifacts = frozen.get("artifacts")
            if not isinstance(artifacts, Mapping) or set(artifacts) != names:
                raise ValueError(f"frozen row artifact set changed: {row_root.name}")
            for name, digest in artifacts.items():
                if _sha256(row_root / name) != digest:
                    raise ValueError(f"frozen row artifact changed: {row_root.name}/{name}")
    if freeze.get("passed_count") != passed or freeze.get("failed_count") != planned - passed:
        raise ValueError("predictions freeze status counts are inconsistent")
    if freeze.get("completed_count") != completed:
        raise ValueError("predictions freeze completed count is inconsistent")
    _validate_execution_branch(
        ordered_records,
        planned=planned,
        freeze_status=status_value,
        stop_reason=freeze.get("stop_reason"),
        call_count=freeze.get("call_count"),
        canary_pair_status=freeze.get("canary_pair_status"),
    )
    if status_value == "frozen":
        if completed != planned or freeze.get("call_count") != planned:
            raise ValueError("complete predictions freeze has incomplete call count")
        if interrupted or any(
            isinstance(item, Mapping) and item.get("status") == "incomplete"
            for item in frozen_rows
        ):
            raise ValueError("complete predictions freeze contains incomplete terminals")
        if freeze.get("canary_pair_status") != "passed":
            raise ValueError("complete predictions freeze does not bind a passed canary pair")
    else:
        if freeze.get("stop_reason") == "canary_failure":
            if freeze.get("interrupted") or interrupted or freeze.get("canary_pair_status") != "failed":
                raise ValueError("canary failure freeze has invalid interruption binding")
        elif freeze.get("stop_reason") == "continuation_canary_failure":
            if freeze.get("interrupted") or interrupted or freeze.get("canary_pair_status") not in {"passed", "failed"}:
                raise ValueError("continuation canary freeze has invalid interruption binding")
        elif not freeze.get("interrupted") or not interrupted:
            raise ValueError("incomplete predictions freeze must bind an interruption")
        if not isinstance(freeze.get("call_count"), int) or not 0 < freeze["call_count"] < planned:
            raise ValueError("incomplete predictions freeze call count is invalid")


def verify_predictions_freeze(root: Path) -> None:
    """Create the prediction freeze once, then reject any byte drift."""

    root = Path(root)
    rows, _provider_by_index, _media_by_index = _validate_inputs(root)
    ordered_records = _ordered_terminal_records(root, rows)
    planned = len(rows) * len(EXPECTED_FPS)
    frozen_rows: list[dict[str, Any]] = []
    passed = 0
    incomplete_terminal = False
    for item in rows:
        audit_index = item["provider"]["audit_index"]
        role = item["media"]["role"]
        for fps in EXPECTED_FPS:
            row_root = _arm_directory(root, audit_index, role, fps)
            terminal_path = row_root / "terminal.json"
            if not terminal_path.is_file():
                raise ValueError(f"row has no terminal state: {row_root.name}")
            terminal = _load_json(terminal_path)
            if terminal.get("status") == "passed":
                passed += 1
            status = terminal.get("status")
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"unknown terminal status: {row_root.name}")
            if status in {"interrupted", "incomplete"}:
                incomplete_terminal = True
            required = {"request.json", "metadata.json", "terminal.json"}
            if status in {"passed", "invalid_json", "contract_error"}:
                required.add("raw-response.txt")
            if status == "passed":
                required.add("parsed.json")
            entries = {path.name for path in row_root.iterdir()}
            names = {path.name for path in row_root.iterdir() if path.is_file()}
            missing = required - names
            if missing:
                raise ValueError(f"row evidence artifacts missing: {row_root.name}: {sorted(missing)}")
            if entries != names or not names.issubset(set(ROW_ARTIFACT_NAMES)):
                raise ValueError(f"row has unexpected artifacts: {row_root.name}")
            if status == "passed":
                validate_probe(
                    _load_json(row_root / "parsed.json"),
                    item["media"]["clip_duration_seconds"],
                    nominal_duration_seconds=_nominal_duration(item["media"]),
                    fps=fps,
                )
            frozen_rows.append(
                {
                    "audit_index": audit_index,
                    "role": role,
                    "fps": fps,
                    "status": status,
                    "artifacts": {name: _sha256(row_root / name) for name in sorted(names)},
                }
            )
    input_hashes = _freeze_inputs(root)
    prediction_path = root / "predictions-freeze.json"
    if prediction_path.exists():
        _verify_freeze_contents(root, _load_json(prediction_path))
        return
    if incomplete_terminal:
        raise ValueError("incomplete terminal states require the interruption freeze")
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "model": MODEL,
        "expected_fps": list(EXPECTED_FPS),
        "decision_gate": EXPECTED_DECISION_GATE,
        "frozen_at": datetime.now(UTC).isoformat(),
        "row_count": planned,
        "completed_count": planned,
        "call_count": planned,
        "passed_count": passed,
        "failed_count": planned - passed,
        **input_hashes,
        "runner_sha256": _sha256(SCRIPT_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
        "prompt_template_sha256": _prompt_template_sha256(),
        "response_schema_sha256": _schema_sha256(),
        "provider_response_schema_sha256": _provider_schema_sha256(),
        "validator_sha256": _validator_sha256(),
        "compatibility_canary_pairs": COMPATIBILITY_CANARY_PAIRS,
        "canary_pair_status": _derived_canary_status(ordered_records),
        "rows": frozen_rows,
    }
    _write_once(prediction_path, freeze)
    _verify_freeze_contents(root, freeze)


def _source_rows_for_replay(source_root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    provider_payload = _load_json(source_root / "provider-rows.json")
    media_payload = _load_json(source_root / "media-manifest.json")
    provider_rows = _provider_projection(_rows_payload(provider_payload, "provider-rows.json"))
    media_by_index = {
        row["audit_index"]: _media_row(row, source_root)
        for row in _rows_payload(media_payload, "media-manifest.json")
    }
    provider_by_index = {row["audit_index"]: row for row in provider_rows}
    rows = [{"provider": row, "media": media_by_index[row["audit_index"]]} for row in provider_rows]
    return rows, provider_by_index, media_by_index


def prepare_continuation(source_root: Path, output_root: Path) -> None:
    """Prepare a zero-call amended-validator replay and continuation candidate.

    The source run is treated as immutable.  Only its first, failed-canary
    pair is imported; the next pair remains the fresh compatibility canary.
    """
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError("continuation output already exists")
    prediction_path = source_root / "predictions-freeze.json"
    if not prediction_path.is_file():
        raise ValueError("source prediction freeze is required")
    source_freeze = _load_json(prediction_path)
    if source_freeze.get("status") != "incomplete" or source_freeze.get("call_count") != 2:
        raise ValueError("source must be the immutable two-call canary freeze")
    rows, _provider_by_index, _media_by_index = _source_rows_for_replay(source_root)
    tasks = [(item, fps) for item in rows for fps in EXPECTED_FPS]
    imported = tasks[:2]
    output_root.mkdir(parents=True)
    source_clips = source_root / "clips"
    if not source_clips.is_dir():
        raise ValueError("source clip bundle is unavailable")
    os.symlink(source_clips.resolve(), output_root / "clips", target_is_directory=True)
    for name in ("provider-rows.json", "media-manifest.json"):
        shutil.copy2(source_root / name, output_root / name)
    treatment = _load_json(source_root / "treatment-freeze.json")
    treatment["runner_sha256"] = _sha256(SCRIPT_PATH)
    treatment["validator_sha256"] = _validator_sha256()
    _write_once(output_root / "treatment-freeze.json", treatment)
    selection = _load_json(source_root / "selection-freeze.json")
    selection["treatment_freeze_sha256"] = _sha256(output_root / "treatment-freeze.json")
    _write_once(output_root / "selection-freeze.json", selection)
    authorization = _load_json(source_root / "provider-authorization.json")
    authorization["treatment_freeze_sha256"] = _sha256(output_root / "treatment-freeze.json")
    authorization["selection_freeze_sha256"] = _sha256(output_root / "selection-freeze.json")
    _write_once(output_root / "provider-authorization.json", authorization)
    prefix: list[dict[str, Any]] = []
    for item, fps in imported:
        audit_index = item["provider"]["audit_index"]
        role = item["media"]["role"]
        source_arm = _arm_directory(source_root, audit_index, role, fps)
        target_arm = _arm_directory(output_root, audit_index, role, fps)
        target_arm.mkdir(parents=True, exist_ok=False)
        for name in ("request.json", "raw-response.txt", "metadata.json"):
            shutil.copy2(source_arm / name, target_arm / name)
        raw = (source_arm / "raw-response.txt").read_text()
        parsed = validate_probe(
            json.loads(raw),
            item["media"]["clip_duration_seconds"],
            nominal_duration_seconds=_nominal_duration(item["media"]),
            fps=fps,
        )
        _write_once(target_arm / "parsed.json", parsed)
        _write_once(target_arm / "terminal.json", {
            "audit_index": audit_index, "role": role, "fps": fps,
            "status": "passed", "raw_response_sha256": _sha256_bytes(raw.encode()),
        })
        source_artifacts = {name: _sha256(source_arm / name) for name in sorted(path.name for path in source_arm.iterdir() if path.is_file())}
        prefix.append({
            "audit_index": audit_index, "role": role, "fps": fps,
            "source_artifacts": source_artifacts,
            "source_response_id": _load_json(source_arm / "metadata.json").get("response_id"),
            "replayed_artifacts": {name: _sha256(target_arm / name) for name in sorted(path.name for path in target_arm.iterdir() if path.is_file())},
        })
    replay = {
        "schema_version": 1, "status": "frozen", "provider_calls_made": 0,
        "source_root": str(source_root),
        "source_predictions_freeze_sha256": _sha256(prediction_path),
        "imported_prefix": prefix,
        "canary_pair_status": "passed",
        "runner_sha256": _sha256(SCRIPT_PATH),
        "validator_sha256": _validator_sha256(),
    }
    _write_once(output_root / "canary-replay-freeze.json", replay)
    auth = {
        "schema_version": 1, "status": "authorized", "authorization_type": "continuation",
        "provider_calls_made": 0, "prior_provider_calls": 2,
        "remaining_new_call_maximum": 22, "combined_call_maximum": 24,
        "next_arm_position": 3, "fresh_canary_pair": {"start_position": 3, "arms": 2},
        "imported_prefix": prefix,
        "canary_replay_freeze_sha256": _sha256(output_root / "canary-replay-freeze.json"),
        "source_predictions_freeze_sha256": _sha256(prediction_path),
        "treatment_freeze_sha256": _sha256(output_root / "treatment-freeze.json"),
        "selection_freeze_sha256": _sha256(output_root / "selection-freeze.json"),
        "provider_rows_sha256": _sha256(output_root / "provider-rows.json"),
        "media_manifest_sha256": _sha256(output_root / "media-manifest.json"),
        "model": MODEL, "settings": EXPECTED_SETTINGS,
    }
    _write_once(output_root / "continuation-authorization.json", auth)


def validate_continuation_authorization(root: Path) -> None:
    """Validate the zero-call continuation boundary and imported prefix."""
    root = Path(root)
    auth_path = root / "continuation-authorization.json"
    replay_path = root / "canary-replay-freeze.json"
    if not auth_path.is_file() or not replay_path.is_file():
        raise ValueError("continuation authorization and replay freeze are required")
    auth = _load_json(auth_path)
    replay = _load_json(replay_path)
    if auth.get("authorization_type") != "continuation" or auth.get("status") != "authorized":
        raise ValueError("invalid continuation authorization")
    if auth.get("provider_calls_made") != 0 or auth.get("prior_provider_calls") != 2:
        raise ValueError("continuation call counts are invalid")
    if auth.get("remaining_new_call_maximum") != 22 or auth.get("combined_call_maximum") != 24:
        raise ValueError("continuation call cap is invalid")
    if auth.get("next_arm_position") != 3:
        raise ValueError("continuation must begin at arm 3")
    if auth.get("canary_replay_freeze_sha256") != _sha256(replay_path):
        raise ValueError("continuation replay hash drift")
    rows, _provider_by_index, _media_by_index = _validate_inputs(root)
    if len(rows) != 12 or len(rows) * len(EXPECTED_FPS) != 24:
        raise ValueError("continuation panel must contain exactly 12 rows and 24 arms")
    for field, filename in (
        ("treatment_freeze_sha256", "treatment-freeze.json"),
        ("selection_freeze_sha256", "selection-freeze.json"),
        ("provider_rows_sha256", "provider-rows.json"),
        ("media_manifest_sha256", "media-manifest.json"),
    ):
        if auth.get(field) != _sha256(root / filename):
            raise ValueError(f"continuation {field} drift")
    if replay.get("status") != "frozen" or replay.get("canary_pair_status") != "passed":
        raise ValueError("continuation replay has not passed")
    prefix = auth.get("imported_prefix")
    if not isinstance(prefix, list) or len(prefix) != 2 or prefix != replay.get("imported_prefix"):
        raise ValueError("continuation imported prefix drift")
    if [item.get("fps") for item in prefix] != list(EXPECTED_FPS):
        raise ValueError("continuation imported prefix order is invalid")
    if [item.get("role") for item in prefix] != ["target", "target"]:
        raise ValueError("continuation imported prefix roles are invalid")
    source_root = Path(str(replay.get("source_root", "")))
    if not source_root.is_dir():
        raise ValueError("continuation source root is unavailable")
    source_rows, _provider_by_index, _media_by_index = _source_rows_for_replay(source_root)
    source_freeze_path = source_root / "predictions-freeze.json"
    if replay.get("source_predictions_freeze_sha256") != _sha256(source_freeze_path):
        raise ValueError("continuation source prediction freeze drift")
    source_freeze = _load_json(source_freeze_path)
    source_freeze_rows = source_freeze.get("rows", [])
    for item, source_item in zip(prefix, [(source_rows[0], EXPECTED_FPS[0]), (source_rows[0], EXPECTED_FPS[1])], strict=True):
        audit_index = source_item[0]["provider"]["audit_index"]
        role = source_item[0]["media"]["role"]
        arm = _arm_directory(source_root, audit_index, role, source_item[1])
        for name, expected in item.get("source_artifacts", {}).items():
            path = arm / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError("continuation imported source artifact drift")
        expected_row = next((row for row in source_freeze_rows if row.get("audit_index") == audit_index and row.get("fps") == source_item[1]), None)
        if not isinstance(expected_row, Mapping) or not item.get("source_artifacts") or item["source_artifacts"] != expected_row.get("artifacts"):
            raise ValueError("continuation imported source artifacts do not bind v4 freeze")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run(args.root)
    else:
        verify_predictions_freeze(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
