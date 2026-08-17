#!/usr/bin/env python3
"""Overlay successful MP4 compatibility rescues onto the first Gemini overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import overlay_vqa_deadline_gemini_rescue as common


ROOT = Path(__file__).resolve().parents[1]
FIRST_OVERLAY_PATH = ROOT / (
    "data/vqa/benchmarks/deadline-submission-rescue-20260817/"
    "gemini-27/overlay/predictions-freeze.json"
)
MP4_RESCUE_PATH = ROOT / (
    "data/vqa/benchmarks/deadline-submission-rescue-20260817/"
    "gemini-27-mp4/predictions-freeze.json"
)
SOURCE_MANIFEST_PATH = ROOT / (
    "data/vqa/benchmarks/deadline-submission-full-20260816/gemini/manifest.json"
)
OUTPUT_PATH = MP4_RESCUE_PATH.parent / "overlay/predictions-freeze.json"

MODEL = "gemini-3.7-flash"
MAX_OUTPUT_TOKENS = 8_192
EXPECTED_PANEL_COUNT = 2_443
EXPECTED_QUERY_IDS = (175, 310, 886, 1563, 2279, 2404)
FIRST_OVERLAY_SHA256 = (
    "92bd5ea9eb6997662794f8741db3c4b179fa2d6433b474efa3028f8b82652eee"
)
MP4_RESCUE_SHA256 = (
    "49fc5b8a627a4a7b3a4872e9c68a0f4205307af2a5408c918df3314641c76288"
)
ORIGINAL_FREEZE_SHA256 = (
    "e3f0e4cf6d9dd65986f328f3834e96b753b180b3b6123a646418ec1424078cb0"
)
FIRST_RESCUE_FREEZE_SHA256 = (
    "445b58ed61124583b128dd0db7724e43f8b8de3bf982def47edcd6d3ced6e940"
)
FIRST_RESCUE_MANIFEST_SHA256 = (
    "5dbce4878c88f5e629de016b47362ef63e6a110c24c11f2a7db294a1ca731391"
)
SOURCE_MANIFEST_SHA256 = (
    "18fe59579ed9ef5bfb3da5d717ecba4a23f081744dcb87567d29a242d4c1c83d"
)
EXPECTED_MODEL_SETTINGS = {
    "video_fps": 4.0,
    "media_resolution": "MEDIA_RESOLUTION_LOW",
    "thinking_level": "HIGH",
    "max_output_tokens": MAX_OUTPUT_TOKENS,
}

OverlayError = common.OverlayError


def _indexed_rows(value: Mapping[str, Any], *, name: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = common._rows(value, name=name)
    indexed = {int(row["query_id"]): row for row in rows}
    return rows, indexed


def overlay_mp4_rescue(
    *,
    base_freeze: Path = FIRST_OVERLAY_PATH,
    rescue_freeze: Path = MP4_RESCUE_PATH,
    source_manifest: Path = SOURCE_MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    base, base_sha = common._read_artifact(Path(base_freeze))
    if base_sha != FIRST_OVERLAY_SHA256:
        raise OverlayError("first-overlay identity changed")
    if base.get("status") != "predictions_frozen" or base.get("model") != MODEL:
        raise OverlayError("first overlay is not terminal or model-bound")
    base_rows, base_by_id = _indexed_rows(base, name="first overlay")
    if len(base_rows) != EXPECTED_PANEL_COUNT:
        raise OverlayError("first overlay panel count changed")
    if set(base_by_id) != set(range(1, EXPECTED_PANEL_COUNT + 1)):
        raise OverlayError("first overlay query IDs are incomplete")
    if any(
        base_by_id[query_id].get("failure_reason") != "empty_text"
        for query_id in EXPECTED_QUERY_IDS
    ):
        raise OverlayError("first-overlay MP4 rescue reasons changed")

    source, source_sha = common._read_artifact(Path(source_manifest))
    if source_sha != SOURCE_MANIFEST_SHA256 or source.get("status") != "inputs_frozen":
        raise OverlayError("source manifest identity changed")
    source_rows = source.get("rows")
    if not isinstance(source_rows, list) or any(not isinstance(row, dict) for row in source_rows):
        raise OverlayError("source manifest rows are invalid")
    source_by_id = {row.get("query_id"): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise OverlayError("source manifest query IDs are invalid")

    rescue, rescue_sha = common._read_artifact(Path(rescue_freeze))
    if rescue_sha != MP4_RESCUE_SHA256:
        raise OverlayError("MP4 rescue identity changed")
    common._reject_forbidden(rescue)
    if rescue.get("status") != "predictions_frozen" or rescue.get("model") != MODEL:
        raise OverlayError("MP4 rescue is not terminal or model-bound")
    if rescue.get("selected_query_ids") != list(EXPECTED_QUERY_IDS):
        raise OverlayError("MP4 rescue selection changed")
    if rescue.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise OverlayError("MP4 rescue output-token contract changed")
    if rescue.get("original_predictions_freeze_sha256") != ORIGINAL_FREEZE_SHA256:
        raise OverlayError("MP4 original-freeze binding changed")
    if rescue.get("first_rescue_predictions_freeze_sha256") != FIRST_RESCUE_FREEZE_SHA256:
        raise OverlayError("MP4 first-rescue freeze binding changed")
    if rescue.get("first_rescue_manifest_sha256") != FIRST_RESCUE_MANIFEST_SHA256:
        raise OverlayError("MP4 first-rescue manifest binding changed")

    rescue_manifest, rescue_manifest_sha = common._read_artifact(Path(rescue_freeze).with_name("manifest.json"))
    common._reject_forbidden(rescue_manifest)
    if rescue.get("manifest_sha256") != rescue_manifest_sha:
        raise OverlayError("MP4 manifest binding changed")
    if rescue_manifest.get("status") != "inputs_frozen" or rescue_manifest.get("model") != MODEL:
        raise OverlayError("MP4 manifest is not terminal or model-bound")
    if rescue_manifest.get("selected_query_ids") != list(EXPECTED_QUERY_IDS):
        raise OverlayError("MP4 manifest selection changed")
    if rescue_manifest.get("model_settings") != EXPECTED_MODEL_SETTINGS:
        raise OverlayError("MP4 manifest model settings changed")
    for field, expected in (
        ("original_predictions_freeze_sha256", ORIGINAL_FREEZE_SHA256),
        ("first_rescue_predictions_freeze_sha256", FIRST_RESCUE_FREEZE_SHA256),
        ("first_rescue_manifest_sha256", FIRST_RESCUE_MANIFEST_SHA256),
    ):
        if rescue_manifest.get(field) != expected:
            raise OverlayError(f"MP4 manifest {field.replace('_', '-')} changed")

    rescue_rows, rescue_by_id = _indexed_rows(rescue, name="MP4 rescue")
    if [row["query_id"] for row in rescue_rows] != list(EXPECTED_QUERY_IDS):
        raise OverlayError("MP4 rescue rows changed")
    manifest_rows = rescue_manifest.get("rows")
    if not isinstance(manifest_rows, list) or [row.get("query_id") for row in manifest_rows] != list(EXPECTED_QUERY_IDS):
        raise OverlayError("MP4 manifest rows changed")
    manifest_by_id = {int(row["query_id"]): row for row in manifest_rows}
    for query_id in EXPECTED_QUERY_IDS:
        base_row = base_by_id[query_id]
        rescue_row = rescue_by_id[query_id]
        manifest_row = manifest_by_id[query_id]
        source_row = source_by_id.get(query_id)
        if source_row is None:
            raise OverlayError("MP4 query is absent from source manifest")
        for key in ("query_id", "video_id"):
            expected = source_row.get(key)
            if base_row.get(key) != expected or rescue_row.get(key) != expected or manifest_row.get(key) != expected:
                raise OverlayError(f"MP4 source binding changed: {key}")
        if manifest_row.get("question") != source_row.get("question"):
            raise OverlayError("MP4 source binding changed: question")
        common._validate_answer(rescue_row)

    success_count = 0
    merged: list[dict[str, Any]] = []
    for base_row in base_rows:
        query_id = int(base_row["query_id"])
        rescue_row = rescue_by_id.get(query_id)
        if rescue_row is not None and rescue_row.get("status") == "succeeded":
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
        "experiment_id": "deadline-submission-rescue-20260817-gemini-mp4-overlay",
        "status": "predictions_frozen",
        "model": MODEL,
        "row_count": len(merged),
        "completed_count": statuses.get("succeeded", 0),
        "unavailable_count": statuses.get("unavailable", 0),
        "upload_failed_count": statuses.get("upload_failed", 0),
        "base_predictions_freeze_sha256": base_sha,
        "rescue_predictions_freeze_sha256": rescue_sha,
        "source_manifest_sha256": source_sha,
        "original_predictions_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "first_rescue_predictions_freeze_sha256": FIRST_RESCUE_FREEZE_SHA256,
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "rescue_success_count": success_count,
        "rescue_preserved_fallback_count": len(EXPECTED_QUERY_IDS) - success_count,
        "rows": merged,
    }
    common._write_artifact(Path(output_path), body)
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-freeze", type=Path, default=FIRST_OVERLAY_PATH)
    parser.add_argument("--rescue-freeze", type=Path, default=MP4_RESCUE_PATH)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = overlay_mp4_rescue(
        base_freeze=args.base_freeze,
        rescue_freeze=args.rescue_freeze,
        source_manifest=args.source_manifest,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "row_count": result["row_count"],
                "rescue_success_count": result["rescue_success_count"],
                "rescue_preserved_fallback_count": result["rescue_preserved_fallback_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
