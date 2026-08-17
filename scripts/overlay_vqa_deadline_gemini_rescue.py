#!/usr/bin/env python3
"""Overlay successful targeted Gemini rescues onto the frozen full-panel result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_PREDICTIONS_FREEZE_SHA256 = (
    "e3f0e4cf6d9dd65986f328f3834e96b753b180b3b6123a646418ec1424078cb0"
)
MODEL = "gemini-3.7-flash"
MAX_OUTPUT_TOKENS = 8_192
EXPECTED_REASON_BY_QUERY = {
    116: "max_tokens",
    139: "max_tokens",
    175: "empty_text",
    310: "empty_text",
    647: "max_tokens",
    886: "empty_text",
    932: "max_tokens",
    963: "max_tokens",
    995: "provider_upload_failed",
    1013: "empty_text",
    1022: "provider_upload_failed",
    1055: "provider_upload_failed",
    1063: "provider_upload_failed",
    1084: "empty_text",
    1090: "provider_upload_failed",
    1127: "max_tokens",
    1140: "max_tokens",
    1168: "provider_upload_failed",
    1173: "provider_upload_failed",
    1190: "max_tokens",
    1462: "max_tokens",
    1563: "empty_text",
    1771: "empty_text",
    2279: "empty_text",
    2385: "max_tokens",
    2404: "empty_text",
    2421: "max_tokens",
}
RESCUE_QUERY_IDS = tuple(EXPECTED_REASON_BY_QUERY)
_FORBIDDEN_FIELDS = {
    "annotation",
    "annotations",
    "ground_truth",
    "label",
    "labels",
    "reference",
    "references",
}
_TERMINAL_STATUSES = {
    "cost_cap_blocked",
    "failed",
    "succeeded",
    "unavailable",
    "unsupported",
    "upload_failed",
}
_EXPECTED_MODEL_SETTINGS = {
    "video_fps": 4.0,
    "media_resolution": "MEDIA_RESOLUTION_LOW",
    "thinking_level": "HIGH",
    "max_output_tokens": MAX_OUTPUT_TOKENS,
}


class OverlayError(ValueError):
    """Raised when frozen rescue evidence cannot be overlaid safely."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_artifact(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OverlayError(f"invalid frozen artifact: {path}") from error
    if not isinstance(value, dict):
        raise OverlayError(f"frozen artifact is not an object: {path}")
    body = dict(value)
    artifact_sha = body.pop("_artifact_sha256", None)
    if artifact_sha != hashlib.sha256(_canonical(body)).hexdigest():
        raise OverlayError(f"artifact hash mismatch: {path}")
    return value, _sha256_file(path)


def _rows(value: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OverlayError(f"{name} rows are invalid")
    if value.get("row_count") != len(rows):
        raise OverlayError(f"{name} row_count is invalid")
    ids = [row.get("query_id") for row in rows]
    if any(isinstance(query_id, bool) or not isinstance(query_id, int) for query_id in ids):
        raise OverlayError(f"{name} query IDs are invalid")
    if len(ids) != len(set(ids)):
        raise OverlayError(f"{name} query IDs are duplicated")
    return [dict(row) for row in rows]


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_FIELDS:
                raise OverlayError("rescue evidence contains a forbidden label field")
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def _validate_answer(row: Mapping[str, Any]) -> None:
    status = row.get("status")
    if status not in _TERMINAL_STATUSES:
        raise OverlayError("rescue row status is not terminal")
    if status == "succeeded":
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise OverlayError("successful rescue answer is empty")
        elapsed = row.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise OverlayError("successful rescue time is invalid")


def _manifest_rows(value: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise OverlayError(f"{name} manifest rows are invalid")
    return [dict(row) for row in rows]


def _validate_manifest_bindings(
    *,
    base_freeze_path: Path,
    base: Mapping[str, Any],
    base_sha: str,
    rescue_freeze_path: Path,
    rescue: Mapping[str, Any],
) -> None:
    base_manifest, base_manifest_sha = _read_artifact(
        base_freeze_path.parent / "manifest.json"
    )
    rescue_manifest, rescue_manifest_sha = _read_artifact(
        rescue_freeze_path.parent / "manifest.json"
    )
    _reject_forbidden(rescue)
    _reject_forbidden(rescue_manifest)
    if base.get("manifest_sha256") != base_manifest_sha:
        raise OverlayError("base manifest binding changed")
    if rescue.get("manifest_sha256") != rescue_manifest_sha:
        raise OverlayError("rescue manifest binding changed")
    if base_manifest.get("status") != "inputs_frozen":
        raise OverlayError("base manifest is not terminal")
    if rescue_manifest.get("status") != "inputs_frozen":
        raise OverlayError("rescue manifest is not terminal")
    if base_manifest.get("model") != MODEL or rescue_manifest.get("model") != MODEL:
        raise OverlayError("manifest model binding changed")
    if rescue_manifest.get("base_predictions_freeze_sha256") != base_sha:
        raise OverlayError("rescue manifest base-freeze binding changed")
    if rescue_manifest.get("base_manifest_sha256") != base_manifest_sha:
        raise OverlayError("rescue base-manifest binding changed")
    if rescue_manifest.get("selected_query_ids") != list(RESCUE_QUERY_IDS):
        raise OverlayError("rescue manifest selection changed")
    if rescue_manifest.get("model_settings") != _EXPECTED_MODEL_SETTINGS:
        raise OverlayError("rescue manifest model settings changed")
    prompt = base_manifest.get("prompt")
    if not isinstance(prompt, str) or rescue_manifest.get("prompt") != prompt:
        raise OverlayError("rescue prompt binding changed")
    source_input = base_manifest.get("input")
    if not isinstance(source_input, Mapping) or rescue_manifest.get("input") != source_input:
        raise OverlayError("rescue input binding changed")
    input_sha256 = source_input.get("sha256")
    if not isinstance(input_sha256, str) or len(input_sha256) != 64:
        raise OverlayError("base input hash is invalid")

    base_rows = _manifest_rows(base_manifest, name="base")
    base_by_id = {row.get("query_id"): row for row in base_rows}
    if len(base_by_id) != len(base_rows):
        raise OverlayError("base manifest query IDs are invalid")
    rescue_rows = _manifest_rows(rescue_manifest, name="rescue")
    if [row.get("query_id") for row in rescue_rows] != list(RESCUE_QUERY_IDS):
        raise OverlayError("rescue manifest rows do not match selection")
    for rescue_row in rescue_rows:
        query_id = rescue_row["query_id"]
        base_row = base_by_id.get(query_id)
        if base_row is None:
            raise OverlayError("rescue manifest query is absent from base manifest")
        for key, expected in base_row.items():
            if rescue_row.get(key) != expected:
                raise OverlayError(f"rescue manifest source binding changed: {key}")
        if rescue_row.get("input_sha256") != input_sha256:
            raise OverlayError("rescue row input binding changed")


def _write_artifact(path: Path, body: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise OverlayError(f"refusing to overwrite output: {path}")
    value = dict(body)
    value["_artifact_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def overlay_rescue(
    *,
    base_freeze: Path,
    rescue_freeze: Path,
    output_path: Path,
    expected_base_sha256: str = BASE_PREDICTIONS_FREEZE_SHA256,
) -> dict[str, Any]:
    """Return and freeze a full-panel envelope with successful rescues applied."""

    base_freeze = Path(base_freeze)
    rescue_freeze = Path(rescue_freeze)
    base, base_sha = _read_artifact(base_freeze)
    if base_sha != expected_base_sha256:
        raise OverlayError("base predictions freeze identity changed")
    if base.get("status") != "predictions_frozen" or base.get("model") != MODEL:
        raise OverlayError("base predictions freeze is not terminal or model-bound")
    base_rows = _rows(base, name="base")
    if len(base_rows) != 2_443:
        raise OverlayError("base predictions freeze must contain 2443 rows")
    base_by_id = {int(row["query_id"]): row for row in base_rows}
    if set(base_by_id) != set(range(1, 2_444)):
        raise OverlayError("base predictions query IDs are incomplete")
    actual_reasons = {
        query_id: base_by_id[query_id].get("failure_reason")
        for query_id in RESCUE_QUERY_IDS
    }
    if actual_reasons != EXPECTED_REASON_BY_QUERY:
        raise OverlayError("base rescue reasons changed")

    rescue, rescue_sha = _read_artifact(rescue_freeze)
    if rescue.get("status") != "predictions_frozen" or rescue.get("model") != MODEL:
        raise OverlayError("rescue predictions freeze is not terminal or model-bound")
    if rescue.get("base_predictions_freeze_sha256") != base_sha:
        raise OverlayError("rescue base-freeze binding changed")
    if rescue.get("selected_query_ids") != list(RESCUE_QUERY_IDS):
        raise OverlayError("rescue selected-query binding changed")
    if rescue.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise OverlayError("rescue output-token contract changed")
    _validate_manifest_bindings(
        base_freeze_path=base_freeze,
        base=base,
        base_sha=base_sha,
        rescue_freeze_path=rescue_freeze,
        rescue=rescue,
    )
    rescue_rows = _rows(rescue, name="rescue")
    if [row["query_id"] for row in rescue_rows] != list(RESCUE_QUERY_IDS):
        raise OverlayError("rescue rows do not match the frozen selection")

    success_count = 0
    merged: list[dict[str, Any]] = []
    rescue_by_id = {int(row["query_id"]): row for row in rescue_rows}
    for base_row in base_rows:
        query_id = int(base_row["query_id"])
        if query_id not in rescue_by_id:
            merged.append(base_row)
            continue
        rescue_row = rescue_by_id[query_id]
        _validate_answer(rescue_row)
        if rescue_row.get("video_id") != base_row.get("video_id"):
            raise OverlayError("rescue video binding changed")
        if rescue_row.get("status") == "succeeded":
            success_count += 1
            merged.append(rescue_row)
        else:
            merged.append(base_row)

    statuses: dict[str, int] = {}
    for row in merged:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    body = {
        "schema_version": 1,
        "experiment_id": "deadline-submission-rescue-20260817-gemini8192-a1-overlay",
        "status": "predictions_frozen",
        "model": MODEL,
        "row_count": len(merged),
        "completed_count": statuses.get("succeeded", 0),
        "unavailable_count": statuses.get("unavailable", 0),
        "upload_failed_count": statuses.get("upload_failed", 0),
        "base_predictions_freeze_sha256": base_sha,
        "rescue_predictions_freeze_sha256": rescue_sha,
        "selected_query_ids": list(RESCUE_QUERY_IDS),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "rescue_success_count": success_count,
        "rescue_preserved_fallback_count": len(RESCUE_QUERY_IDS) - success_count,
        "rows": merged,
    }
    _write_artifact(Path(output_path), body)
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-freeze", type=Path, required=True)
    parser.add_argument("--rescue-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = overlay_rescue(
        base_freeze=args.base_freeze,
        rescue_freeze=args.rescue_freeze,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "row_count": result["row_count"],
                "rescue_success_count": result["rescue_success_count"],
                "rescue_preserved_fallback_count": result[
                    "rescue_preserved_fallback_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
