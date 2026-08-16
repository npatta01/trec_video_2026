#!/usr/bin/env python3
"""Freeze an additive, zero-call correction of the dense v22 parser defect.

This module intentionally does not call the provider or import the historical
runner's validator.  It reads the already-frozen v22 raw responses, applies a
new symmetric encoded-duration tolerance, and writes parsed JSON under a new
private correction root.  The v22 source tree is never modified.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


EXPECTED_FPS = (4.0, 12.0)
EXPECTED_ARM_COUNT = 24
EXPECTED_SOURCE_ARTIFACTS = (
    "request.json",
    "raw-response.txt",
    "metadata.json",
    "terminal.json",
)
OUTPUT_TOP_LEVEL = ("correction-freeze.json", "parsed")
TRUSTED_V22_PREDICTION_FREEZE_SHA256 = (
    "fb002af0bb8e663e495afcd2b1888dd37d11e0c24ef57d0f14f5ecf65434f758"
)
RULE = {
    "name": "symmetric_encoded_duration_one_frame_v1",
    "nominal_duration": "Decimal(clip_end_seconds) - Decimal(clip_start_seconds)",
    "encoded_duration_tolerance": "abs(encoded_duration - nominal_duration) <= 1 / fps",
    "evidence_timestamp_bound": "0 <= timestamp <= nominal_duration",
    "timestamp_rewrite": "forbidden",
    "other_contract_rules": "unchanged strict probe schema and cross-field rules",
}
FORBIDDEN_KEYS = frozenset(
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
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROBE_FIELDS = frozenset(
    {
        "question_interpretation",
        "fact_observed",
        "observed_fact",
        "evidence_timestamps",
        "conflicts_or_uncertainty",
        "needs_more_evidence",
    }
)
INTERPRETATION_FIELDS = frozenset(
    {"requested_entity", "requested_visual_fact", "temporal_qualifier"}
)
SCRIPT_PATH = Path(__file__).resolve()
RULE_SHA256 = hashlib.sha256(
    json.dumps(RULE, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


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
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _regular_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a regular file: {path}")
    return path


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        raise ValueError(f"{field} must be a number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _reject_forbidden_keys(value: Any, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} has a non-text field name")
            normalized = key.casefold().replace("-", "_")
            if normalized in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden label/reference/scoring key: {field}.{key}")
            _reject_forbidden_keys(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{field}[{index}]")


def validate_corrected_probe(
    payload: Any,
    *,
    encoded_duration_seconds: Any,
    nominal_duration_seconds: Any,
    fps: Any,
) -> dict[str, Any]:
    """Apply the unchanged strict probe contract plus the symmetric duration rule."""

    if not isinstance(payload, Mapping):
        raise ValueError("probe response must be an object")
    if set(payload) != PROBE_FIELDS:
        raise ValueError("probe response fields are invalid")
    _reject_forbidden_keys(payload, "probe response")

    interpretation_payload = payload["question_interpretation"]
    if not isinstance(interpretation_payload, Mapping) or set(interpretation_payload) != INTERPRETATION_FIELDS:
        raise ValueError("question_interpretation fields are invalid")
    interpretation = {
        key: _string(interpretation_payload[key], f"question_interpretation.{key}")
        for key in sorted(INTERPRETATION_FIELDS)
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

    encoded = _decimal(encoded_duration_seconds, "encoded_duration_seconds")
    nominal = _decimal(nominal_duration_seconds, "nominal_duration_seconds")
    frame_rate = _decimal(fps, "fps")
    if encoded <= 0:
        raise ValueError("encoded_duration_seconds must be positive")
    if nominal <= 0:
        raise ValueError("nominal_duration_seconds must be positive")
    if frame_rate <= 0:
        raise ValueError("fps must be positive")
    if abs(encoded - nominal) > Decimal(1) / frame_rate:
        raise ValueError("encoded duration mismatch exceeds one frame")

    timestamps: list[float] = []
    for index, value in enumerate(timestamp_payload):
        timestamp = _number(value, f"evidence_timestamps[{index}]")
        timestamp_decimal = _decimal(value, f"evidence_timestamps[{index}]")
        if timestamp < 0 or timestamp_decimal > nominal:
            raise ValueError(f"evidence timestamp is outside nominal duration: {timestamp}")
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


def nominal_duration_seconds(clip_start_seconds: Any, clip_end_seconds: Any) -> Decimal:
    """Derive the requested window using Decimal endpoints, never float subtraction."""

    start = _decimal(clip_start_seconds, "clip_start_seconds")
    end = _decimal(clip_end_seconds, "clip_end_seconds")
    if start < 0 or end <= start:
        raise ValueError("clip offsets must satisfy 0 <= start < end")
    return end - start


def _arm_path(root: Path, audit_index: int, role: str, fps: float) -> Path:
    if not isinstance(audit_index, int) or isinstance(audit_index, bool) or audit_index < 1:
        raise ValueError("audit_index must be a positive integer")
    if not isinstance(role, str) or SAFE_ID_RE.fullmatch(role) is None:
        raise ValueError("role is not path-safe")
    if fps not in EXPECTED_FPS:
        raise ValueError("unexpected FPS")
    return root / "rows" / f"{audit_index:02d}-{role}-{int(fps)}fps"


def _output_arm_path(root: Path, position: int, audit_index: int, role: str, fps: float) -> Path:
    if not isinstance(position, int) or position < 1:
        raise ValueError("position must be positive")
    if not isinstance(role, str) or SAFE_ID_RE.fullmatch(role) is None:
        raise ValueError("role is not path-safe")
    return root / "parsed" / f"{position:02d}-{audit_index:02d}-{role}-{int(fps)}fps"


def _write_once(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite immutable correction artifact: {path}") from error


def _source_records(source_root: Path) -> list[dict[str, Any]]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    freeze_path = _regular_file(source_root / "predictions-freeze.json", "prediction freeze")
    freeze_sha = _sha256(freeze_path)
    if freeze_sha != TRUSTED_V22_PREDICTION_FREEZE_SHA256:
        raise ValueError("source prediction freeze hash is not the trusted v22 freeze")
    freeze = _load_json(freeze_path)
    if not isinstance(freeze, Mapping) or freeze.get("status") != "frozen":
        raise ValueError("source prediction freeze is not complete")
    rows = freeze.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ARM_COUNT:
        raise ValueError("source prediction freeze must contain exactly 24 arms")
    seen: set[tuple[int, str, float]] = set()
    responses: set[str] = set()
    records: list[dict[str, Any]] = []
    rows_root = source_root / "rows"
    if not rows_root.is_dir() or rows_root.is_symlink():
        raise ValueError("source rows directory is invalid")
    expected_names: set[str] = set()
    for position, frozen in enumerate(rows, start=1):
        if not isinstance(frozen, Mapping):
            raise ValueError("source freeze arm is invalid")
        audit_index = frozen.get("audit_index")
        role = frozen.get("role")
        fps = frozen.get("fps")
        original_status = frozen.get("status")
        identity = (audit_index, role, fps)
        if not isinstance(audit_index, int) or isinstance(audit_index, bool) or not isinstance(role, str) or fps not in EXPECTED_FPS:
            raise ValueError("source freeze arm identity is invalid")
        if identity in seen:
            raise ValueError("source freeze contains duplicate arm identity")
        seen.add(identity)
        if original_status not in {"passed", "contract_error", "invalid_json"}:
            raise ValueError("source freeze arm has no raw response status")
        source_arm = _arm_path(source_root, audit_index, role, fps)
        expected_names.add(source_arm.name)
        if not source_arm.is_dir() or source_arm.is_symlink():
            raise ValueError(f"source arm is missing: {source_arm.name}")
        artifacts = frozen.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"source arm artifact map is invalid: {source_arm.name}")
        expected_artifacts = set(EXPECTED_SOURCE_ARTIFACTS)
        if original_status == "passed":
            expected_artifacts.add("parsed.json")
        if set(artifacts) != expected_artifacts:
            raise ValueError(f"source arm artifact set drift: {source_arm.name}")
        actual_names = {path.name for path in source_arm.iterdir()}
        if actual_names != expected_artifacts:
            raise ValueError(f"source arm has extra or missing artifacts: {source_arm.name}")
        for name in expected_artifacts:
            path = _regular_file(source_arm / name, f"source arm {source_arm.name}/{name}")
            if _sha256(path) != artifacts[name]:
                raise ValueError(f"source arm artifact hash drift: {source_arm.name}/{name}")
        request = _load_json(source_arm / "request.json")
        metadata = _load_json(source_arm / "metadata.json")
        terminal = _load_json(source_arm / "terminal.json")
        if not isinstance(request, Mapping) or not isinstance(metadata, Mapping) or not isinstance(terminal, Mapping):
            raise ValueError(f"source arm metadata envelope is invalid: {source_arm.name}")
        _reject_forbidden_keys(request, f"{source_arm.name}/request.json")
        _reject_forbidden_keys(metadata, f"{source_arm.name}/metadata.json")
        _reject_forbidden_keys(terminal, f"{source_arm.name}/terminal.json")
        if request.get("fps") != fps or request.get("clip_start_seconds") is None or request.get("clip_end_seconds") is None:
            raise ValueError(f"source request identity drift: {source_arm.name}")
        if terminal.get("audit_index") != audit_index or terminal.get("role") != role or terminal.get("fps") != fps or terminal.get("status") != original_status:
            raise ValueError(f"source terminal identity/status drift: {source_arm.name}")
        raw_path = source_arm / "raw-response.txt"
        raw_bytes = raw_path.read_bytes()
        raw_sha = _sha256_bytes(raw_bytes)
        if metadata.get("raw_response_sha256") != raw_sha or terminal.get("raw_response_sha256") != raw_sha:
            raise ValueError(f"source raw response hash binding drift: {source_arm.name}")
        response_id = metadata.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError(f"source arm response ID is missing: {source_arm.name}")
        response_id = response_id.strip()
        if response_id in responses:
            raise ValueError(f"duplicate source response ID: {response_id}")
        responses.add(response_id)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            corrected_status = "invalid_json"
            parsed = None
            correction_error = str(error)
        else:
            try:
                parsed = validate_corrected_probe(
                    payload,
                    encoded_duration_seconds=request["clip_duration_seconds"],
                    nominal_duration_seconds=nominal_duration_seconds(
                        request["clip_start_seconds"], request["clip_end_seconds"]
                    ),
                    fps=fps,
                )
            except ValueError as error:
                corrected_status = "contract_error"
                parsed = None
                correction_error = str(error)
            else:
                corrected_status = "passed"
                correction_error = None
        original_passed = original_status == "passed"
        corrected_passed = corrected_status == "passed"
        records.append(
            {
                "position": position,
                "audit_index": audit_index,
                "role": role,
                "fps": fps,
                "original_status": original_status,
                "corrected_status": corrected_status,
                "corrected_acceptance_changed": original_passed != corrected_passed,
                "response_id": response_id,
                "source_artifacts": {
                    name: artifacts[name] for name in EXPECTED_SOURCE_ARTIFACTS
                },
                "source_original_parsed_sha256": artifacts.get("parsed.json"),
                "parsed": parsed,
                "correction_error": correction_error,
            }
        )
    actual_arm_names = {
        path.name for path in rows_root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if actual_arm_names != expected_names:
        raise ValueError("source rows contain extra or missing arm directories")
    if len(responses) != EXPECTED_ARM_COUNT:
        raise ValueError("source response IDs are not unique across all 24 arms")
    return records


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "arms": len(records),
        "original_passed": sum(item["original_status"] == "passed" for item in records),
        "original_contract_error": sum(item["original_status"] == "contract_error" for item in records),
        "corrected_passed": sum(item["corrected_status"] == "passed" for item in records),
        "corrected_contract_error": sum(item["corrected_status"] == "contract_error" for item in records),
        "corrected_acceptance_changes": sum(
            bool(item["corrected_acceptance_changed"]) for item in records
        ),
    }


def _public_arm_record(record: Mapping[str, Any], parsed_sha256: str | None) -> dict[str, Any]:
    result = {
        key: record[key]
        for key in (
            "position",
            "audit_index",
            "role",
            "fps",
            "original_status",
            "corrected_status",
            "corrected_acceptance_changed",
            "response_id",
            "source_artifacts",
            "source_original_parsed_sha256",
        )
    }
    result["parsed_output_sha256"] = parsed_sha256
    return result


def _freeze_payload(source_freeze_sha256: str, records: Sequence[Mapping[str, Any]], aggregate: Mapping[str, int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen",
        "correction_version": "run-v22-correction-v1",
        "source_prediction_freeze_sha256": source_freeze_sha256,
        "correction_script_sha256": _sha256(SCRIPT_PATH),
        "correction_rule_sha256": RULE_SHA256,
        "provider_calls": 0,
        "aggregate": dict(aggregate),
        "arms": [
            _public_arm_record(item, item.get("parsed_output_sha256")) for item in records
        ],
    }


def correct_run(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Replay frozen raw responses and freeze corrected parsed outputs additively."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"refusing to overwrite existing correction root: {output_root}")
    records = _source_records(source_root)
    aggregate = _aggregate(records)
    if aggregate["arms"] != EXPECTED_ARM_COUNT:
        raise ValueError("correction replay did not process all 24 arms")
    output_root.mkdir(parents=True)
    try:
        for record in records:
            parsed = record["parsed"]
            parsed_sha256: str | None = None
            if parsed is not None:
                parsed_bytes = _json_bytes(parsed)
                arm_root = _output_arm_path(
                    output_root,
                    record["position"],
                    record["audit_index"],
                    record["role"],
                    record["fps"],
                )
                _write_once(arm_root / "parsed.json", parsed_bytes)
                parsed_sha256 = _sha256_bytes(parsed_bytes)
            record["parsed_output_sha256"] = parsed_sha256
        freeze = _freeze_payload(TRUSTED_V22_PREDICTION_FREEZE_SHA256, records, aggregate)
        _write_once(output_root / "correction-freeze.json", _json_bytes(freeze))
    except BaseException:
        # The root remains additive and immutable; callers can remove an
        # incomplete temporary root explicitly, but no source bytes are touched.
        raise
    verify_correction(source_root, output_root)
    return {"provider_calls": 0, "aggregate": aggregate}


def verify_correction(source_root: Path, output_root: Path) -> None:
    """Independently verify source bindings, corrected outputs, and freeze bytes."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("correction root is invalid")
    actual_top = {path.name for path in output_root.iterdir()}
    if actual_top != set(OUTPUT_TOP_LEVEL):
        raise ValueError("correction root has extra or missing artifacts")
    parsed_root = output_root / "parsed"
    if not parsed_root.is_dir() or parsed_root.is_symlink():
        raise ValueError("correction parsed root is invalid")
    records = _source_records(source_root)
    aggregate = _aggregate(records)
    freeze = _load_json(_regular_file(output_root / "correction-freeze.json", "correction freeze"))
    if not isinstance(freeze, Mapping) or freeze.get("status") != "frozen":
        raise ValueError("correction freeze is invalid")
    if freeze.get("source_prediction_freeze_sha256") != TRUSTED_V22_PREDICTION_FREEZE_SHA256:
        raise ValueError("correction freeze source prediction hash drift")
    if freeze.get("correction_script_sha256") != _sha256(SCRIPT_PATH):
        raise ValueError("correction script hash drift")
    if freeze.get("correction_rule_sha256") != RULE_SHA256:
        raise ValueError("correction rule hash drift")
    if freeze.get("provider_calls") != 0:
        raise ValueError("correction provider call count is not zero")
    if freeze.get("aggregate") != aggregate:
        raise ValueError("correction aggregate drift")
    frozen_arms = freeze.get("arms")
    if not isinstance(frozen_arms, list) or len(frozen_arms) != EXPECTED_ARM_COUNT:
        raise ValueError("correction freeze arm list is invalid")
    expected_output_names: set[str] = set()
    for record, frozen in zip(records, frozen_arms, strict=True):
        if not isinstance(frozen, Mapping):
            raise ValueError("correction freeze arm is invalid")
        arm_root = _output_arm_path(
            output_root,
            record["position"],
            record["audit_index"],
            record["role"],
            record["fps"],
        )
        expected_output_names.add(arm_root.name)
        expected_parsed = record["parsed"]
        parsed_sha256: str | None = None
        if expected_parsed is not None:
            parsed_path = _regular_file(arm_root / "parsed.json", "corrected parsed output")
            parsed_bytes = parsed_path.read_bytes()
            parsed_sha256 = _sha256_bytes(parsed_bytes)
            if parsed_sha256 != _sha256_bytes(_json_bytes(expected_parsed)):
                raise ValueError(f"corrected parsed output drift: {arm_root.name}")
            parsed_payload = _load_json(parsed_path)
            _reject_forbidden_keys(parsed_payload, f"{arm_root.name}/parsed.json")
            if parsed_payload != expected_parsed:
                raise ValueError(f"corrected parsed semantic drift: {arm_root.name}")
        elif arm_root.exists():
            if list(arm_root.iterdir()):
                raise ValueError(f"unexpected corrected output for rejected arm: {arm_root.name}")
        expected_frozen = _public_arm_record(record, parsed_sha256)
        if dict(frozen) != expected_frozen:
            raise ValueError(f"correction freeze arm binding drift: {arm_root.name}")
    actual_output_names = {
        path.name for path in parsed_root.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if actual_output_names != expected_output_names:
        raise ValueError("correction parsed root has extra or missing arms")
    _reject_forbidden_keys(freeze, "correction-freeze.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("correct", "verify"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--source-root", type=Path, required=True)
        command_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "correct":
        correct_run(args.source_root, args.output_root)
    else:
        verify_correction(args.source_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
