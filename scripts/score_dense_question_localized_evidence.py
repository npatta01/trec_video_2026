#!/usr/bin/env python3
"""Score the frozen dense local-evidence probe after a blinded audit.

The provider prediction freeze and the parser correction are verified before
this module reads the corrected Issue #35 references.  The only quality
measurements produced here are paired evidence-recovery counts and the frozen
mechanism gate; no answer-quality metric is appropriate for this probe.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


EXPECTED_FPS = (4.0, 12.0)
EXPECTED_ROLES = ("target", "control")
CORRECTION_VERSION = "run-v22-correction-v2"
AUDIT_SCHEMA_VERSION = 2
AUDIT_FREEZE_VERSION = 3
SCORE_VERSION = 3
AUDIT_V2_NAME = "manual-postfreeze-audit-v2.json"
AUDIT_FREEZE_V3_NAME = "postfreeze-audit-freeze-v3.json"
SCORE_V3_NAME = "postfreeze-score-v3.json"
SCORE_INTEGRITY_V3_NAME = "postfreeze-score-integrity-v3.json"
CANONICAL_REFERENCE_DIR = "task-support-sufficiency-audit-issue35-v1"
CANONICAL_REFERENCE_NAME = "manual-audit-corrected-v3.json"
CANONICAL_REFERENCE_CORRECTION_NAME = "audit-correction-v3.json"
CANONICAL_REFERENCE_FREEZE_NAME = "corrected-audit-freeze-v3.json"
EXPECTED_AUDIT_TOP_FIELDS = frozenset(
    {
        "completed_at",
        "correction_freeze_sha256",
        "correction_version",
        "experiment_id",
        "inspection_scope",
        "reference_correction_sha256",
        "reference_freeze_sha256",
        "reference_sha256",
        "rows",
        "schema_version",
        "source_prediction_freeze_sha256",
        "status",
    }
)
EXPECTED_INSPECTION_SCOPE_FIELDS = frozenset(
    {
        "arms_inspected",
        "auditable_target_policy",
        "changed_acceptance_arms_inspected",
        "clips_inspected",
        "disagreements_inspected",
        "source",
    }
)
EXPECTED_AUDIT_FIELDS = frozenset(
    {
        "audit_index",
        "role",
        "fps",
        "video_id",
        "question",
        "reference",
        "map_line_numbers",
        "clip_sha256",
        "raw_response_sha256",
        "corrected_parsed_sha256",
        "correction_acceptance_changed",
        "evidence_timestamps",
        "fact_present_in_clip",
        "model_fact_supported",
        "recovered",
        "timestamp_supported",
        "unsupported_observation",
        "ambiguous_reference",
        "reference_issue",
        "exposure_flags",
        "reviewer_note",
        "inspected",
    }
)
EXPECTED_EXPOSURE_FIELDS = frozenset(
    {
        "grounded_v2_24",
        "grounded_v3_11",
        "hard_audit_30",
        "multi_interpretation_12",
        "scope_control",
        "seven_row_canary",
        "synthesis_followup",
    }
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ARM_RE = re.compile(r"^(?P<position>[0-9]+)-(?P<index>[0-9]+)-(?P<role>[A-Za-z0-9_.-]+)-(?P<fps>[0-9]+)fps$")


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


def _regular_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} must be a regular file: {path}")
    return path


def _load_json(path: Path, description: str = "JSON file") -> Any:
    try:
        return json.loads(_regular_file(path, description).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        if path.read_bytes() != data:
            raise ValueError(f"write-once artifact already exists with different bytes: {path}") from error


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load verifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_roots(root: Path) -> tuple[Path, Path]:
    root = Path(root).resolve()
    if (root / "run-v22").is_dir():
        return root, root / "run-v22"
    if root.name == "run-v22":
        return root.parent, root
    return root.parent, root


def _canonical_reference_paths(benchmark_root: Path) -> tuple[Path, Path, Path]:
    root = benchmark_root.parent / CANONICAL_REFERENCE_DIR
    return (
        root / CANONICAL_REFERENCE_NAME,
        root / CANONICAL_REFERENCE_CORRECTION_NAME,
        root / CANONICAL_REFERENCE_FREEZE_NAME,
    )


def _exact_keys(payload: Mapping[str, Any], expected: frozenset[str], description: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{description} schema drift: missing={missing}, extra={extra}")


def _verify_reference_bundle(
    reference_path: Path,
    correction_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    reference = _load_json(reference_path, "canonical corrected-v3 references")
    correction = _load_json(correction_path, "canonical corrected-v3 correction")
    freeze = _load_json(freeze_path, "canonical corrected-v3 audit freeze")
    if not isinstance(reference, Mapping) or not isinstance(correction, Mapping) or not isinstance(freeze, Mapping):
        raise ValueError("canonical corrected-v3 bundle members must be objects")
    _exact_keys(
        reference,
        frozenset(
            {
                "private",
                "provider_input",
                "rows",
                "schema_version",
                "status",
                "template_sha256",
                "correction_id",
                "base_manual_audit_sha256",
                "base_template_sha256",
                "correction_basis",
                "base_manual_audit_v2_sha256",
            }
        ),
        "canonical corrected-v3 reference",
    )
    _exact_keys(
        correction,
        frozenset(
            {
                "schema_version",
                "status",
                "correction_id",
                "correction_basis",
                "provider_calls_made",
                "provider_authorization",
                "supersedes",
                "manual_audit_corrected_v3_sha256",
                "reviewed_row_count",
                "changed_rows",
                "corrected_role_counts",
                "reviewer_finding_status",
                "source_bindings",
                "notes",
            }
        ),
        "canonical corrected-v3 correction",
    )
    _exact_keys(
        freeze,
        frozenset(
            {
                "schema_version",
                "status",
                "experiment_id",
                "correction_id",
                "frozen_at",
                "manual_audit_corrected_v3_sha256",
                "audit_correction_v3_sha256",
                "superseded_v2",
                "original_v1_selection",
                "audit_denominator",
                "all_rows_adjudicated",
                "corrected_role_counts",
                "feasibility_gate",
                "provider_authorization",
                "provider_calls_made",
                "replacement_selection_created",
                "replacement_protocol_created",
                "exposure_policy",
                "reviewer_finding_status",
            }
        ),
        "canonical corrected-v3 audit freeze",
    )
    if (
        reference.get("private") is not True
        or reference.get("provider_input") is not False
        or reference.get("schema_version") != 3
        or reference.get("status") != "corrected_v3"
        or correction.get("schema_version") != 3
        or correction.get("status") != "corrected_v3"
        or freeze.get("schema_version") != 3
        or freeze.get("status") != "frozen"
    ):
        raise ValueError("canonical corrected-v3 reference status/schema is invalid")
    if reference.get("correction_id") != correction.get("correction_id") or reference.get("correction_id") != freeze.get("correction_id"):
        raise ValueError("canonical corrected-v3 correction lineage drift")
    reference_sha = _sha256(reference_path)
    correction_sha = _sha256(correction_path)
    freeze_sha = _sha256(freeze_path)
    if reference.get("template_sha256") != reference.get("base_template_sha256"):
        raise ValueError("canonical corrected-v3 template lineage drift")
    if correction.get("manual_audit_corrected_v3_sha256") != reference_sha:
        raise ValueError("canonical corrected-v3 correction/reference hash drift")
    if freeze.get("manual_audit_corrected_v3_sha256") != reference_sha:
        raise ValueError("canonical corrected-v3 freeze/reference hash drift")
    if freeze.get("audit_correction_v3_sha256") != correction_sha:
        raise ValueError("canonical corrected-v3 freeze/correction hash drift")
    source_bindings = correction.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise ValueError("canonical corrected-v3 source bindings are invalid")
    source_files = {
        "treatment_freeze_sha256": "treatment-freeze.json",
        "candidate_sheet_sha256": "candidate-sheet.json",
        "candidate_sheet_freeze_sha256": "candidate-sheet-freeze.json",
        "audit_template_sha256": "audit-template.json",
        "original_selection_freeze_sha256": "selection-freeze.json",
        "original_protocol_sha256": "protocol.json",
    }
    source_root = correction_path.parent
    for field, filename in source_files.items():
        path = source_root / filename
        if source_bindings.get(field) != _sha256(_regular_file(path, f"canonical source {filename}")):
            raise ValueError(f"canonical corrected-v3 source hash drift: {field}")
    if reference.get("template_sha256") != source_bindings.get("audit_template_sha256"):
        raise ValueError("canonical corrected-v3 reference/template hash drift")
    superseded_v2 = correction.get("supersedes")
    freeze_superseded_v2 = freeze.get("superseded_v2")
    if superseded_v2 != freeze_superseded_v2:
        raise ValueError("canonical corrected-v3 superseded lineage drift")
    superseded_files = {
        "manual_audit_corrected_v2_sha256": "manual-audit-corrected-v2.json",
        "audit_correction_v2_sha256": "audit-correction-v2.json",
        "corrected_audit_freeze_v2_sha256": "corrected-audit-freeze-v2.json",
        "feasibility_result_v2_sha256": "feasibility-result-v2.json",
    }
    for field, filename in superseded_files.items():
        if superseded_v2.get(field) != _sha256(_regular_file(source_root / filename, f"canonical superseded source {filename}")):
            raise ValueError(f"canonical corrected-v3 superseded hash drift: {field}")
    rows = reference.get("rows")
    if not isinstance(rows, list) or len(rows) != 32 or freeze.get("audit_denominator") != 32 or correction.get("reviewed_row_count") != 32:
        raise ValueError("canonical corrected-v3 denominator drift")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("canonical corrected-v3 row is invalid")
        _exact_keys(
            row,
            frozenset(
                {
                    "adjudication",
                    "annotation_file_sha256",
                    "audit_index",
                    "direct_answer",
                    "evidence_line_numbers",
                    "exposure_flags",
                    "frozen_map",
                    "question",
                    "question_stratum",
                    "queue",
                    "reference",
                    "role",
                    "verdict",
                    "video_id",
                }
            ),
            "canonical corrected-v3 reference row",
        )
        adjudication = row.get("adjudication")
        frozen_map = row.get("frozen_map")
        exposure = row.get("exposure_flags")
        if not isinstance(adjudication, Mapping) or not isinstance(frozen_map, Mapping) or not isinstance(exposure, Mapping):
            raise ValueError("canonical corrected-v3 nested row payload is invalid")
        _exact_keys(
            adjudication,
            frozenset(
                {
                    "requested_entity",
                    "requested_fact",
                    "qualifier",
                    "answer_unit",
                    "evidence_line_numbers",
                    "conflicts",
                    "role",
                    "verdict",
                    "note",
                }
            ),
            "canonical corrected-v3 adjudication",
        )
        _exact_keys(frozen_map, frozenset({"lines", "map", "map_sha256"}), "canonical corrected-v3 frozen map")
        _exact_keys(exposure, EXPECTED_EXPOSURE_FIELDS, "canonical corrected-v3 exposure flags")
        if not isinstance(frozen_map["lines"], list):
            raise ValueError("canonical corrected-v3 map lines are invalid")
        for line in frozen_map["lines"]:
            if not isinstance(line, Mapping):
                raise ValueError("canonical corrected-v3 map line is invalid")
            _exact_keys(line, frozenset({"claim", "line_number"}), "canonical corrected-v3 map line")
        if row.get("role") != adjudication.get("role") or row.get("verdict") != adjudication.get("verdict"):
            raise ValueError("canonical corrected-v3 row/adjudication lineage drift")
    role_counts: dict[str, int] = {}
    for row in rows:
        role_counts[str(row["role"])] = role_counts.get(str(row["role"]), 0) + 1
    if role_counts != correction.get("corrected_role_counts") or role_counts != freeze.get("corrected_role_counts"):
        raise ValueError("canonical corrected-v3 role-count lineage drift")
    if freeze.get("all_rows_adjudicated") is not True or freeze.get("provider_calls_made") != 0 or correction.get("provider_calls_made") != 0:
        raise ValueError("canonical corrected-v3 freeze is not terminal")
    return {
        "reference_path": reference_path,
        "correction_path": correction_path,
        "freeze_path": freeze_path,
        "reference": reference,
        "correction": correction,
        "freeze": freeze,
        "reference_sha256": reference_sha,
        "correction_sha256": correction_sha,
        "freeze_sha256": freeze_sha,
    }


def verify_canonical_references(benchmark_root: Path, explicit: Path | None = None) -> dict[str, Any]:
    canonical_reference, correction_path, freeze_path = _canonical_reference_paths(Path(benchmark_root).resolve())
    for path, description in (
        (canonical_reference, "canonical corrected-v3 references"),
        (correction_path, "canonical corrected-v3 correction"),
        (freeze_path, "canonical corrected-v3 audit freeze"),
    ):
        _regular_file(path, description)
    bundle = _verify_reference_bundle(canonical_reference, correction_path, freeze_path)
    if explicit is not None:
        explicit_path = Path(explicit)
        _regular_file(explicit_path, "explicit corrected-v3 references")
        if explicit_path.read_bytes() != canonical_reference.read_bytes():
            raise ValueError("explicit references are not an exact canonical corrected-v3 byte copy")
    return bundle


def _verify_prediction_inputs(benchmark_root: Path, run_root: Path, reference_path: Path | None = None) -> dict[str, Any]:
    """Verify all immutable provider/correction bytes before reading references."""

    correction_root = run_root.parent / "run-v22-correction-v2"
    if not correction_root.is_dir():
        raise ValueError("correction-v2 root is unavailable")
    correction = _module(Path(__file__).with_name("correct_dense_question_localized_evidence.py"), "dense_correction_verifier")
    correction.verify_correction(run_root, correction_root)
    runner = _module(Path(__file__).with_name("run_dense_question_localized_evidence.py"), "dense_prediction_verifier")
    runner.verify_predictions_freeze(run_root)
    correction_freeze = _load_json(correction_root / "correction-freeze.json", "correction freeze")
    source_freeze = _load_json(run_root / "predictions-freeze.json", "prediction freeze")
    if correction_freeze.get("correction_version") != CORRECTION_VERSION:
        raise ValueError("wrong correction version")
    if correction_freeze.get("source_prediction_freeze_sha256") != _sha256(run_root / "predictions-freeze.json"):
        raise ValueError("correction source prediction freeze binding drift")
    if source_freeze.get("status") != "frozen" or source_freeze.get("completed_count") != source_freeze.get("row_count"):
        raise ValueError("prediction freeze is incomplete")
    if source_freeze.get("row_count") != 24 or source_freeze.get("call_count") != 24:
        raise ValueError("prediction freeze denominator is not 24")
    selection = _load_json(run_root / "selection-freeze.json", "selection freeze")
    media = _load_json(run_root / "media-manifest.json", "media manifest")
    provider = _load_json(run_root / "provider-rows.json", "provider rows")
    treatment = _load_json(run_root / "treatment-freeze.json", "treatment freeze")
    reference_bundle = verify_canonical_references(benchmark_root, reference_path)
    return {
        "correction_root": correction_root,
        "correction_freeze": correction_freeze,
        "source_freeze": source_freeze,
        "selection": selection,
        "media": media,
        "provider": provider,
        "treatment": treatment,
        "reference_path": reference_bundle["reference_path"],
        "reference_correction_path": reference_bundle["correction_path"],
        "reference_freeze_path": reference_bundle["freeze_path"],
        "reference_sha256": reference_bundle["reference_sha256"],
        "reference_correction_sha256": reference_bundle["correction_sha256"],
        "reference_freeze_sha256": reference_bundle["freeze_sha256"],
    }


def _selected_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = selection.get("selected_row_metadata")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection has no selected rows")
    result = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("selection row is not an object")
        audit_index = row.get("audit_index")
        role = row.get("role")
        key = (audit_index, role)
        if not isinstance(audit_index, int) or role not in EXPECTED_ROLES or key in seen:
            raise ValueError("selection row identity is invalid or duplicated")
        seen.add(key)
        result.append(dict(row))
    return result


def _reference_rows(reference_payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = reference_payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("corrected references have no rows")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("audit_index"), int):
            raise ValueError("corrected reference row is invalid")
        index = int(row["audit_index"])
        if index in result:
            raise ValueError(f"duplicate corrected reference row: {index}")
        result[index] = dict(row)
    return result


def _validate_audit_schema(audit: Mapping[str, Any]) -> None:
    _exact_keys(audit, EXPECTED_AUDIT_TOP_FIELDS, "manual post-freeze audit")
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION or audit.get("status") != "completed":
        raise ValueError("manual post-freeze audit status/schema is invalid")
    scope = audit.get("inspection_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("manual audit inspection scope is invalid")
    _exact_keys(scope, EXPECTED_INSPECTION_SCOPE_FIELDS, "manual audit inspection scope")
    if not isinstance(audit.get("rows"), list):
        raise ValueError("manual audit rows are invalid")
    for row in audit["rows"]:
        if not isinstance(row, Mapping):
            raise ValueError("manual audit row is not an object")
        _exact_keys(row, EXPECTED_AUDIT_FIELDS, "manual audit row")
        exposure = row.get("exposure_flags")
        if not isinstance(exposure, Mapping):
            raise ValueError("manual audit exposure flags are invalid")
        _exact_keys(exposure, EXPECTED_EXPOSURE_FIELDS, "manual audit exposure flags")


def validate_audit_schema(audit: Mapping[str, Any]) -> None:
    """Validate the exact top-level, row, and nested audit schema."""

    _validate_audit_schema(audit)


def _bind_reference_labels(audit: Mapping[str, Any], references: Mapping[int, Mapping[str, Any]]) -> None:
    """Check that the private audit copied only the verified corrected labels."""

    _validate_audit_schema(audit)
    rows = audit.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manual audit has no rows")
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("audit_index"), int):
            raise ValueError("manual audit reference row is invalid")
        reference = references.get(row["audit_index"])
        if reference is None:
            raise ValueError("manual audit reference row is missing")
        for field in ("reference", "exposure_flags"):
            if row.get(field) != reference.get(field):
                raise ValueError(f"manual audit corrected reference binding drift: {field}")


def _verify_audit_freeze_bindings(
    audit_freeze: Mapping[str, Any],
    audit_path: Path,
    run_root: Path,
    correction_root: Path,
    reference_sha256: str,
) -> None:
    expected_schema = frozenset(
        {
            "correction_freeze_sha256",
            "correction_version",
            "experiment_id",
            "frozen_at",
            "manual_audit_sha256",
            "media_manifest_sha256",
            "provider_rows_sha256",
            "reference_correction_sha256",
            "reference_freeze_sha256",
            "reference_sha256",
            "row_count",
            "schema_version",
            "selection_sha256",
            "source_prediction_freeze_sha256",
            "status",
        }
    )
    _exact_keys(audit_freeze, expected_schema, "post-freeze audit freeze")
    if audit_freeze.get("schema_version") != AUDIT_FREEZE_VERSION:
        raise ValueError("post-freeze audit freeze schema is invalid")
    if audit_freeze.get("status") != "frozen":
        raise ValueError("post-freeze audit is not frozen")
    if audit_freeze.get("row_count") != 24:
        raise ValueError("post-freeze audit denominator is not 24")
    expected = {
        "manual_audit_sha256": _sha256(audit_path),
        "reference_sha256": reference_sha256,
        "reference_correction_sha256": _sha256(
            run_root.parent.parent / CANONICAL_REFERENCE_DIR / CANONICAL_REFERENCE_CORRECTION_NAME
        ),
        "reference_freeze_sha256": _sha256(
            run_root.parent.parent / CANONICAL_REFERENCE_DIR / CANONICAL_REFERENCE_FREEZE_NAME
        ),
        "selection_sha256": _sha256(run_root / "selection-freeze.json"),
        "media_manifest_sha256": _sha256(run_root / "media-manifest.json"),
        "provider_rows_sha256": _sha256(run_root / "provider-rows.json"),
        "correction_freeze_sha256": _sha256(correction_root / "correction-freeze.json"),
        "source_prediction_freeze_sha256": _sha256(run_root / "predictions-freeze.json"),
    }
    for field, value in expected.items():
        if audit_freeze.get(field) != value:
            raise ValueError(f"post-freeze audit binding drift: {field}")


def _prediction_rows(
    run_root: Path,
    correction_root: Path,
    correction_freeze: Mapping[str, Any],
) -> dict[tuple[int, str, float], dict[str, Any]]:
    arms = correction_freeze.get("arms")
    if not isinstance(arms, list) or len(arms) != 24:
        raise ValueError("correction freeze arm count is invalid")
    result: dict[tuple[int, str, float], dict[str, Any]] = {}
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError("correction arm is invalid")
        key = (arm.get("audit_index"), arm.get("role"), arm.get("fps"))
        if key in result or not isinstance(key[0], int) or key[1] not in EXPECTED_ROLES or key[2] not in EXPECTED_FPS:
            raise ValueError("correction arm identity is invalid or duplicated")
        position = arm.get("position")
        if not isinstance(position, int):
            raise ValueError("correction arm position is invalid")
        source_arm = run_root / "rows" / f"{key[0]:02d}-{key[1]}-{int(key[2])}fps"
        corrected_candidates = list((correction_root / "parsed").glob(f"*-{key[0]:02d}-{key[1]}-{int(key[2])}fps/parsed.json"))
        if len(corrected_candidates) != 1:
            raise ValueError(f"corrected parsed arm is missing or duplicated: {key}")
        parsed_path = corrected_candidates[0]
        parsed_sha = _sha256(parsed_path)
        if parsed_sha != arm.get("parsed_output_sha256"):
            raise ValueError(f"corrected parsed hash drift: {key}")
        parsed = _load_json(parsed_path, "corrected parsed output")
        source_artifacts = arm.get("source_artifacts")
        if not isinstance(source_artifacts, Mapping):
            raise ValueError("correction source artifact binding is invalid")
        if source_arm.name != f"{key[0]:02d}-{key[1]}-{int(key[2])}fps":
            raise ValueError("prediction arm path binding drift")
        raw_path = source_arm / "raw-response.txt"
        request_path = source_arm / "request.json"
        metadata_path = source_arm / "metadata.json"
        terminal_path = source_arm / "terminal.json"
        for path in (raw_path, request_path, metadata_path, terminal_path):
            _regular_file(path, "prediction arm artifact")
        if any(_sha256(source_arm / name) != digest for name, digest in source_artifacts.items()):
            raise ValueError(f"source arm hash drift: {key}")
        result[key] = {
            "audit_index": key[0],
            "role": key[1],
            "fps": key[2],
            "status": arm.get("corrected_status"),
            "correction_acceptance_changed": arm.get("corrected_acceptance_changed"),
            "parsed": parsed,
            "evidence_timestamps": parsed.get("evidence_timestamps"),
            "parsed_output_sha256": parsed_sha,
            "raw_response_sha256": _sha256(raw_path),
            "request_sha256": _sha256(request_path),
            "metadata_sha256": _sha256(metadata_path),
            "terminal_sha256": _sha256(terminal_path),
            "source_original_parsed_sha256": arm.get("source_original_parsed_sha256"),
            "metadata": _load_json(metadata_path, "prediction metadata"),
        }
    return result


def _selection_index(selection: Mapping[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(row["audit_index"], row["role"]): row for row in _selected_rows(selection)}


def _bool_field(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"manual audit field {field} must be boolean")
    return value


def _validate_audit_rows(
    selection: Mapping[str, Any],
    predictions: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = _selection_index(selection)
    pred_payload = predictions.get("rows")
    audit_rows = audit.get("rows")
    if not isinstance(pred_payload, list) or not isinstance(audit_rows, list):
        raise ValueError("selection, predictions, and audit rows are required")
    pred_index: dict[tuple[int, str, float], dict[str, Any]] = {}
    for row in pred_payload:
        key = (row.get("audit_index"), row.get("role"), row.get("fps"))
        if key in pred_index:
            raise ValueError("duplicate prediction arm")
        pred_index[key] = dict(row)
    expected = {(index, role, fps) for index, role in selected for fps in EXPECTED_FPS}
    if set(pred_index) != expected:
        raise ValueError("prediction arms are missing or extra")
    seen: set[tuple[int, str, float]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in audit_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("manual audit row is not an object")
        missing = EXPECTED_AUDIT_FIELDS - set(raw)
        if missing:
            raise ValueError(f"manual audit row is missing fields: {sorted(missing)}")
        key = (raw.get("audit_index"), raw.get("role"), raw.get("fps"))
        if key in seen:
            raise ValueError("duplicate manual audit arm")
        if key not in expected:
            raise ValueError("manual audit arm is missing or extra")
        seen.add(key)
        selected_row = selected[(key[0], key[1])]
        prediction = pred_index[key]
        if prediction.get("status") != "passed":
            raise ValueError("manual audit includes a non-passed terminal arm")
        for field in ("video_id", "question"):
            expected_value = prediction.get(field, selected_row.get(field))
            if raw[field] != expected_value:
                raise ValueError(f"manual audit {field} binding drift")
        if raw["clip_sha256"] != prediction.get("clip_sha256"):
            raise ValueError("manual audit clip hash binding drift")
        if raw["corrected_parsed_sha256"] != prediction.get("parsed_output_sha256"):
            raise ValueError("manual audit corrected parsed binding drift")
        if raw["raw_response_sha256"] != prediction.get("raw_response_sha256"):
            raise ValueError("manual audit raw response binding drift")
        if raw["correction_acceptance_changed"] != prediction.get("correction_acceptance_changed"):
            raise ValueError("manual audit correction acceptance binding drift")
        if raw["map_line_numbers"] != selected_row.get("map_line_numbers"):
            raise ValueError("manual audit map-line binding drift")
        if not _bool_field(raw, "inspected"):
            raise ValueError("manual audit inspection is incomplete")
        for field in (
            "fact_present_in_clip",
            "model_fact_supported",
            "recovered",
            "timestamp_supported",
            "unsupported_observation",
            "ambiguous_reference",
            "correction_acceptance_changed",
        ):
            _bool_field(raw, field)
        if not isinstance(raw["reference_issue"], str) or not isinstance(raw["reviewer_note"], str) or not raw["reviewer_note"].strip():
            raise ValueError("manual audit reference issue/note is invalid")
        if not isinstance(raw["evidence_timestamps"], list):
            raise ValueError("manual audit evidence timestamps must be a list")
        parsed = prediction.get("parsed")
        if raw["evidence_timestamps"] != parsed.get("evidence_timestamps"):
            raise ValueError("manual audit timestamp binding drift")
        fact_observed = parsed.get("fact_observed") is True
        if raw["recovered"] and (
            not raw["fact_present_in_clip"]
            or not raw["model_fact_supported"]
            or not raw["timestamp_supported"]
            or not fact_observed
            or raw["ambiguous_reference"]
        ):
            raise ValueError("recovery requires a supported observed fact and timestamp")
        expected_unsupported = fact_observed and (
            not raw["model_fact_supported"] or not raw["timestamp_supported"]
        )
        if raw["unsupported_observation"] != expected_unsupported:
            raise ValueError("unsupported observation flag is inconsistent")
        normalized.append(dict(raw))
    if seen != expected:
        raise ValueError("manual audit is incomplete")
    return normalized


def validate_manual_audit(
    selection: Mapping[str, Any], predictions: Mapping[str, Any], audit: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate one frozen audit arm for every selected row and FPS."""

    _validate_audit_schema(audit)
    return _validate_audit_rows(selection, predictions, audit)


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], dict[float, Mapping[str, Any]]]:
    pairs: dict[tuple[int, str], dict[float, Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("audit_index"), row.get("role"))
        fps = row.get("fps")
        if key[0] is None or key[1] not in EXPECTED_ROLES or fps not in EXPECTED_FPS:
            raise ValueError("audit arm identity is invalid")
        bucket = pairs.setdefault(key, {})
        if fps in bucket:
            raise ValueError("duplicate audit arm")
        bucket[fps] = row
    if any(set(bucket) != set(EXPECTED_FPS) for bucket in pairs.values()):
        raise ValueError("incomplete terminal pair")
    return pairs


def _validate_pair_invariants(pairs: Mapping[tuple[int, str], Mapping[float, Mapping[str, Any]]]) -> None:
    shared_fields = (
        "video_id",
        "question",
        "reference",
        "reference_issue",
        "fact_present_in_clip",
        "ambiguous_reference",
        "clip_sha256",
        "map_line_numbers",
        "exposure_flags",
        "correction_acceptance_changed",
    )
    for key, pair in pairs.items():
        four = pair[4.0]
        twelve = pair[12.0]
        for field in shared_fields:
            if four.get(field) != twelve.get(field):
                raise ValueError(f"paired arm invariant drift for {key}: {field}")
        if four.get("ambiguous_reference") is not True and four.get("reference_issue") is None:
            raise ValueError(f"paired reference issue is missing for {key}")


def _count_exposure(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    flags: set[str] = set()
    for row in rows:
        payload = row.get("exposure_flags")
        if not isinstance(payload, Mapping):
            raise ValueError("exposure flags are invalid")
        flags.update(str(key) for key in payload)
    result: dict[str, dict[str, int]] = {}
    for flag in sorted(flags):
        subset = [row for row in rows if row["exposure_flags"].get(flag) is True]
        result[flag] = {
            "arms": len(subset),
            "target_4fps_recovered": sum(1 for row in subset if row["role"] == "target" and row["fps"] == 4.0 and row["recovered"]),
            "target_12fps_recovered": sum(1 for row in subset if row["role"] == "target" and row["fps"] == 12.0 and row["recovered"]),
        }
    return result


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate paired arm labels without computing answer metrics."""

    if not rows:
        raise ValueError("cannot aggregate an empty audit")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("audit row is not an object")
        if row.get("inspected", True) is not True:
            raise ValueError("inspection is incomplete")
        if row.get("recovered") is True and (
            row.get("fact_present_in_clip") is not True
            or row.get("model_fact_supported") is not True
            or row.get("timestamp_supported") is not True
            or row.get("ambiguous_reference") is True
        ):
            raise ValueError("recovery requires a supported observed fact and timestamp")
    pairs = _pair_rows(rows)
    _validate_pair_invariants(pairs)
    target_pairs = [pair for key, pair in pairs.items() if key[1] == "target"]
    control_pairs = [pair for key, pair in pairs.items() if key[1] == "control"]
    auditable_targets = [pair for pair in target_pairs if not pair[4.0]["ambiguous_reference"]]
    def recovered(pair: Mapping[float, Mapping[str, Any]], fps: float) -> bool:
        return bool(pair[fps]["recovered"])

    target_4 = sum(recovered(pair, 4.0) for pair in auditable_targets)
    target_12 = sum(recovered(pair, 12.0) for pair in auditable_targets)
    incremental = sum(not recovered(pair, 4.0) and recovered(pair, 12.0) for pair in auditable_targets)
    control_4 = sum(recovered(pair, 4.0) for pair in control_pairs)
    control_12 = sum(recovered(pair, 12.0) for pair in control_pairs)
    regressions = sum(recovered(pair, 4.0) and not recovered(pair, 12.0) for pair in control_pairs)
    unsupported_4 = sum(row["unsupported_observation"] for row in rows if row["fps"] == 4.0)
    unsupported_12 = sum(row["unsupported_observation"] for row in rows if row["fps"] == 12.0)
    result: dict[str, Any] = {
        "expected_arm_count": len(pairs) * 2,
        "completed_arm_count": len(rows),
        "complete_terminal_pairs": len(rows) == len(pairs) * 2,
        "manual_inspection_complete": all(row.get("inspected", True) is True for row in rows),
        "target_count": len(target_pairs),
        "target_ambiguous_count": len(target_pairs) - len(auditable_targets),
        "target_auditable_count": len(auditable_targets),
        "target_4fps_recovered": target_4,
        "target_12fps_recovered": target_12,
        "incremental_recoveries": incremental,
        "control_count": len(control_pairs),
        "control_4fps_recovered": control_4,
        "control_12fps_recovered": control_12,
        "control_regressions_12fps": regressions,
        "unsupported_observations_4fps": unsupported_4,
        "unsupported_observations_12fps": unsupported_12,
        "exposure_splits": _count_exposure(rows),
    }
    result["auditable_target_recovery_fraction_12fps"] = (
        target_12 / len(auditable_targets) if auditable_targets else None
    )
    return result


def classify_gate(result: Mapping[str, Any]) -> str:
    fraction = result.get("auditable_target_recovery_fraction_12fps")
    checks = (
        result.get("complete_terminal_pairs") is True,
        result.get("manual_inspection_complete") is True,
        int(result.get("incremental_recoveries", -1)) >= 2,
        fraction is not None and float(fraction) >= 0.5,
        int(result.get("unsupported_observations_4fps", -1)) == 0,
        int(result.get("unsupported_observations_12fps", -1)) == 0,
        int(result.get("control_regressions_12fps", -1)) == 0,
    )
    return "PASS" if all(checks) else "FAIL"


def _build_predictions_payload(
    prediction_map: Mapping[tuple[int, str, float], Mapping[str, Any]],
    selection: Mapping[str, Any],
    media: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> dict[str, Any]:
    media_by_index = {row["audit_index"]: row for row in media.get("rows", [])}
    provider_by_key = {(row["audit_index"], row["role"]): row for row in provider.get("rows", [])}
    result = []
    for key, prediction in sorted(prediction_map.items()):
        media_row = media_by_index.get(key[0])
        if not isinstance(media_row, Mapping):
            raise ValueError("prediction media row is missing")
        provider_row = provider_by_key.get((key[0], key[1]))
        if not isinstance(provider_row, Mapping):
            raise ValueError("prediction provider row is missing")
        result.append(
            {
                **prediction,
                "clip_sha256": media_row.get("clip_sha256"),
                "video_id": provider_row.get("video_id"),
                "question": provider_row.get("question"),
                "correction_acceptance_changed": prediction.get("correction_acceptance_changed"),
            }
        )
    return {"rows": result}


def _build_template(
    verified: Mapping[str, Any],
    prediction_map: Mapping[tuple[int, str, float], Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    selection = verified["selection"]
    media_by_index = {row["audit_index"]: row for row in verified["media"].get("rows", [])}
    provider_by_index = {row["audit_index"]: row for row in verified["provider"].get("rows", [])}
    rows = []
    for selected in _selected_rows(selection):
        index = selected["audit_index"]
        reference = references.get(index)
        if reference is None:
            raise ValueError(f"reference row is missing: {index}")
        media = media_by_index[index]
        provider = provider_by_index[index]
        exposure = reference.get("exposure_flags")
        for fps in EXPECTED_FPS:
            prediction = prediction_map[(index, selected["role"], fps)]
            rows.append(
                {
                    "audit_index": index,
                    "role": selected["role"],
                    "fps": fps,
                    "video_id": provider["video_id"],
                    "question": provider["question"],
                    "reference": reference["reference"],
                    "map_line_numbers": selected["map_line_numbers"],
                    "clip_sha256": media["clip_sha256"],
                    "raw_response_sha256": prediction["raw_response_sha256"],
                    "corrected_parsed_sha256": prediction["parsed_output_sha256"],
                    "correction_acceptance_changed": prediction["correction_acceptance_changed"],
                    "evidence_timestamps": prediction["evidence_timestamps"],
                    "fact_present_in_clip": None,
                    "model_fact_supported": None,
                    "recovered": None,
                    "timestamp_supported": None,
                    "unsupported_observation": None,
                    "ambiguous_reference": None,
                    "reference_issue": "",
                    "exposure_flags": exposure,
                    "reviewer_note": "",
                    "inspected": False,
                }
            )
    return {
        "schema_version": 1,
        "status": "template",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "correction_version": CORRECTION_VERSION,
        "rows": rows,
    }


def write_audit_template(root: Path, references_path: Path | None = None) -> Path:
    benchmark_root, run_root = _resolve_roots(root)
    verified = _verify_prediction_inputs(benchmark_root, run_root, references_path)
    references = _reference_rows(_load_json(verified["reference_path"], "corrected references"))
    prediction_map = _prediction_rows(run_root, verified["correction_root"], verified["correction_freeze"])
    payload = _build_template(verified, prediction_map, references)
    path = run_root / "postfreeze-audit-template.json"
    _write_once(path, payload)
    return path


def write_audit_v2(root: Path) -> Path:
    """Create an additive lineage-bound copy without changing manual labels."""

    benchmark_root, run_root = _resolve_roots(root)
    source_path = run_root / "manual-postfreeze-audit.json"
    source = _load_json(source_path, "manual post-freeze audit")
    if not isinstance(source, Mapping) or source.get("status") != "completed":
        raise ValueError("manual post-freeze audit must be completed before migration")
    verified = _verify_prediction_inputs(benchmark_root, run_root)
    references = _reference_rows(_load_json(verified["reference_path"], "canonical corrected-v3 references"))
    payload = dict(source)
    payload.update(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "reference_sha256": verified["reference_sha256"],
            "reference_correction_sha256": verified["reference_correction_sha256"],
            "reference_freeze_sha256": verified["reference_freeze_sha256"],
            "correction_freeze_sha256": _sha256(verified["correction_root"] / "correction-freeze.json"),
            "source_prediction_freeze_sha256": _sha256(run_root / "predictions-freeze.json"),
        }
    )
    _validate_audit_schema(payload)
    _bind_reference_labels(payload, references)
    path = run_root / AUDIT_V2_NAME
    _write_once(path, payload)
    return path


def freeze_audit(root: Path) -> dict[str, Any]:
    benchmark_root, run_root = _resolve_roots(root)
    audit_path = run_root / AUDIT_V2_NAME
    if not audit_path.exists():
        write_audit_v2(root)
    audit = _load_json(audit_path, "manual post-freeze audit")
    if audit.get("status") != "completed":
        raise ValueError("manual post-freeze audit must be completed before freezing")
    verified = _verify_prediction_inputs(benchmark_root, run_root)
    references = _reference_rows(_load_json(verified["reference_path"], "corrected references"))
    _bind_reference_labels(audit, references)
    prediction_map = _prediction_rows(run_root, verified["correction_root"], verified["correction_freeze"])
    prediction_payload = _build_predictions_payload(
        prediction_map, verified["selection"], verified["media"], verified["provider"]
    )
    rows = validate_manual_audit(verified["selection"], prediction_payload, audit)
    frozen = {
        "schema_version": AUDIT_FREEZE_VERSION,
        "status": "frozen",
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "correction_version": CORRECTION_VERSION,
        "manual_audit_sha256": _sha256(audit_path),
        "reference_sha256": verified["reference_sha256"],
        "reference_correction_sha256": verified["reference_correction_sha256"],
        "reference_freeze_sha256": verified["reference_freeze_sha256"],
        "selection_sha256": _sha256(run_root / "selection-freeze.json"),
        "media_manifest_sha256": _sha256(run_root / "media-manifest.json"),
        "provider_rows_sha256": _sha256(run_root / "provider-rows.json"),
        "correction_freeze_sha256": _sha256(verified["correction_root"] / "correction-freeze.json"),
        "source_prediction_freeze_sha256": _sha256(run_root / "predictions-freeze.json"),
        "row_count": len(rows),
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    path = run_root / AUDIT_FREEZE_V3_NAME
    _write_once(path, frozen)
    return frozen


def score(root: Path, *, references_path: Path | None = None) -> dict[str, Any]:
    """Validate a frozen audit and write only the predeclared gate metrics."""

    benchmark_root, run_root = _resolve_roots(root)
    verified = _verify_prediction_inputs(benchmark_root, run_root, references_path)
    audit_path = run_root / AUDIT_V2_NAME
    audit_freeze_path = run_root / AUDIT_FREEZE_V3_NAME
    audit = _load_json(audit_path, "manual post-freeze audit")
    if audit.get("status") != "completed":
        raise ValueError("manual audit is not complete")
    audit_freeze = _load_json(audit_freeze_path, "post-freeze audit freeze")
    _verify_audit_freeze_bindings(
        audit_freeze,
        audit_path,
        run_root,
        verified["correction_root"],
        verified["reference_sha256"],
    )
    references = _reference_rows(_load_json(verified["reference_path"], "corrected references"))
    _bind_reference_labels(audit, references)
    prediction_map = _prediction_rows(run_root, verified["correction_root"], verified["correction_freeze"])
    prediction_payload = _build_predictions_payload(
        prediction_map, verified["selection"], verified["media"], verified["provider"]
    )
    rows = validate_manual_audit(verified["selection"], prediction_payload, audit)
    metrics = aggregate(rows)
    metrics["gate"] = classify_gate(metrics)
    metrics["provider_usage"] = {
        "call_count": verified["source_freeze"].get("call_count"),
        "prompt_tokens": sum(int(row["metadata"].get("prompt_token_count", 0)) for row in prediction_map.values()),
        "candidate_tokens": sum(int(row["metadata"].get("candidates_token_count", 0)) for row in prediction_map.values()),
        "total_tokens": sum(int(row["metadata"].get("total_token_count", 0)) for row in prediction_map.values()),
        "model_versions": sorted({str(row["metadata"].get("model_version")) for row in prediction_map.values()}),
        "cost": verified["source_freeze"].get("cost"),
    }
    metrics["input_bindings"] = {
        "correction_freeze_sha256": _sha256(verified["correction_root"] / "correction-freeze.json"),
        "source_prediction_freeze_sha256": _sha256(run_root / "predictions-freeze.json"),
        "selection_sha256": _sha256(run_root / "selection-freeze.json"),
        "media_manifest_sha256": _sha256(run_root / "media-manifest.json"),
        "provider_rows_sha256": _sha256(run_root / "provider-rows.json"),
        "treatment_sha256": _sha256(run_root / "treatment-freeze.json"),
        "reference_sha256": verified["reference_sha256"],
        "reference_correction_sha256": verified["reference_correction_sha256"],
        "reference_freeze_sha256": verified["reference_freeze_sha256"],
        "manual_audit_sha256": _sha256(audit_path),
        "audit_freeze_sha256": _sha256(audit_freeze_path),
        "scorer_sha256": _sha256(Path(__file__).resolve()),
    }
    metrics["schema_version"] = SCORE_VERSION
    metrics["status"] = "frozen"
    metrics["experiment_id"] = "dense-question-localized-evidence-issue35-v1"
    metrics["correction_version"] = CORRECTION_VERSION
    score_path = run_root / SCORE_V3_NAME
    _write_once(score_path, metrics)
    integrity = {
        "schema_version": SCORE_VERSION,
        "status": "verified",
        "score_sha256": _sha256(score_path),
        "score_bytes_sha256": _sha256_bytes(score_path.read_bytes()),
        "input_bindings": metrics["input_bindings"],
    }
    _write_once(run_root / SCORE_INTEGRITY_V3_NAME, integrity)
    return metrics


def verify_score(root: Path, *, references_path: Path | None = None) -> None:
    benchmark_root, run_root = _resolve_roots(root)
    verified = _verify_prediction_inputs(benchmark_root, run_root, references_path)
    score_path = _regular_file(run_root / SCORE_V3_NAME, "post-freeze score")
    integrity_path = _regular_file(run_root / SCORE_INTEGRITY_V3_NAME, "post-freeze score integrity")
    score_payload = _load_json(score_path, "post-freeze score")
    integrity = _load_json(integrity_path, "post-freeze score integrity")
    if integrity.get("score_sha256") != _sha256(score_path) or integrity.get("score_bytes_sha256") != _sha256_bytes(score_path.read_bytes()):
        raise ValueError("score integrity hash drift")
    audit_path = _regular_file(run_root / AUDIT_V2_NAME, "manual audit")
    freeze_path = _regular_file(run_root / AUDIT_FREEZE_V3_NAME, "audit freeze")
    _verify_audit_freeze_bindings(
        _load_json(freeze_path, "audit freeze"),
        audit_path,
        run_root,
        verified["correction_root"],
        verified["reference_sha256"],
    )
    expected = score(run_root, references_path=references_path)
    if score_path.read_bytes() != _json_bytes(expected):
        raise ValueError("score bytes are not deterministic")
    if score_payload != expected:
        raise ValueError("score semantic drift")
    del verified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("template", "freeze-audit", "score", "verify"):
        child = sub.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--references", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "template":
        print(write_audit_template(args.root, args.references))
    elif args.command == "freeze-audit":
        freeze_audit(args.root)
    elif args.command == "score":
        print(json.dumps(score(args.root, references_path=args.references), indent=2, sort_keys=True))
    else:
        verify_score(args.root, references_path=args.references)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
