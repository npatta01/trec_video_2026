#!/usr/bin/env python3
"""Freeze the private, label-free frontier Phase 0 panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from trec_video_2026.vqa_frontier_phase0_panel import (
    build_phase0_panel,
    validate_phase0_panel,
    write_json_atomically,
)


CANONICAL_DATA_ROOT = (Path(__file__).resolve().parents[1] / "data").resolve()
PILOT_FREEZE_BINDING_KEYS = frozenset(
    {
        "delivery_policy_version",
        "panel_sha256",
        "prompt_sha256",
        "provider",
        "provider_schema_sha256",
        "raw_files_sha256",
        "response_format_policy",
        "result_count",
        "results_sha256",
        "schema_sha256",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--successful-pilot-results", type=Path, required=True)
    parser.add_argument("--successful-pilot-freeze", type=Path, required=True)
    parser.add_argument("--qwen-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    import hashlib

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"pilot result line {line_number} must be an object")
        rows.append(value)
    return rows


def _private_output_path(path: Path) -> Path:
    root = CANONICAL_DATA_ROOT.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("canonical private data root is not a directory")
    if path.is_symlink():
        raise ValueError(f"refusing symlinked output: {path}")
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("output must resolve beneath the canonical private data root")
    return resolved


def _verify_successful_pilot(
    source_panel: Mapping[str, object],
    results_path: Path,
    freeze_path: Path,
) -> str:
    aggregate = _read_json(results_path)
    freeze = _read_json(freeze_path)
    if not isinstance(aggregate, Mapping) or not isinstance(freeze, Mapping):
        raise ValueError("successful pilot artifacts must be JSON objects")
    sibling = results_path.parent / "results" / "gemini.jsonl"
    if not sibling.is_file() or sibling.is_symlink():
        raise ValueError(
            "successful pilot sibling results/gemini.jsonl is missing or unsafe"
        )
    rows = _read_jsonl(sibling)
    if len(rows) != 1:
        raise ValueError(
            "successful pilot sibling results must contain exactly one row"
        )
    if aggregate.get("provider") != "gemini":
        raise ValueError("successful pilot aggregate provider must be gemini")
    if aggregate.get("run_id") != source_panel.get("run_id"):
        raise ValueError("successful pilot run_id does not match source panel")
    if (
        aggregate.get("result_count") != 1
        or aggregate.get("succeeded_count") != 1
        or aggregate.get("failed_count") != 0
    ):
        raise ValueError(
            "successful pilot aggregate is not a one-row successful result"
        )
    if aggregate.get("panel_sha256") != _canonical_sha256(source_panel):
        raise ValueError(
            "successful pilot aggregate is bound to a different source panel"
        )
    if aggregate.get("results_sha256") != _canonical_sha256(rows):
        raise ValueError("successful pilot aggregate does not match sibling results")

    if freeze.get("status") != "frozen":
        raise ValueError("successful pilot freeze status is not frozen")
    if (
        type(freeze.get("schema_version")) is not int
        or freeze.get("schema_version") != 1
    ):
        raise ValueError("successful pilot freeze schema_version does not match")
    if freeze.get("artifact_sha256") != _file_sha256(results_path):
        raise ValueError(
            "successful pilot freeze artifact_sha256 does not match aggregate artifact"
        )
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("successful pilot freeze is missing bindings")
    if set(bindings) != PILOT_FREEZE_BINDING_KEYS:
        raise ValueError(
            "successful pilot freeze bindings must contain exactly the preserved keys"
        )
    if freeze.get("bindings_sha256") != _canonical_sha256(bindings):
        raise ValueError(
            "successful pilot freeze bindings_sha256 does not match bindings"
        )
    for field in PILOT_FREEZE_BINDING_KEYS:
        if field not in aggregate or bindings.get(field) != aggregate.get(field):
            raise ValueError(
                f"successful pilot freeze binding does not match aggregate: {field}"
            )

    result = rows[0]
    if (
        result.get("provider") != "gemini"
        or result.get("run_id") != source_panel.get("run_id")
        or result.get("phase") != "preflight"
        or result.get("status") != "succeeded"
    ):
        raise ValueError("sibling result is not a successful preflight result")
    if result.get("panel_sha256") != aggregate.get("panel_sha256"):
        raise ValueError("sibling result panel binding does not match aggregate")
    video_id = result.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("successful pilot result has no video_id")
    source_rows = source_panel.get("examples")
    if not isinstance(source_rows, list):
        raise ValueError("source panel examples are missing")
    source_row = next(
        (
            row
            for row in source_rows
            if isinstance(row, Mapping) and row.get("video_id") == video_id
        ),
        None,
    )
    if not isinstance(source_row, Mapping) or source_row.get("phase") != "preflight":
        raise ValueError(
            "successful pilot result video_id is not a source preflight row"
        )
    if result.get("ordinal") is not None and result.get("ordinal") != source_row.get(
        "ordinal"
    ):
        raise ValueError("successful pilot result ordinal does not match source panel")
    if result.get("source_sha256") != source_row.get("media_sha256"):
        raise ValueError(
            "successful pilot result source hash does not match source panel"
        )
    return video_id


def _duration_summary(
    rows: list[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for phase in ("pilot", "screen"):
        durations = [
            float(row["duration_seconds"]) for row in rows if row.get("phase") == phase
        ]
        summary[phase] = {
            "count": len(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "mean_seconds": round(sum(durations) / len(durations), 6),
        }
    return summary


def _safe_summary(payload: Mapping[str, object], output: Path) -> dict[str, object]:
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("manifest examples are missing")
    rows = [row for row in examples if isinstance(row, Mapping)]
    return {
        "run_id": payload["run_id"],
        "pilot_count": payload["pilot_count"],
        "screen_count": payload["screen_count"],
        "categories": sorted(
            {str(row["category"]) for row in rows if row.get("phase") == "screen"}
        ),
        "duration_summary": _duration_summary(rows),
        "manifest_sha256": _file_sha256(output),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = _private_output_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    source_panel = _read_json(args.source_panel)
    qwen_rows = _read_json(args.qwen_predictions)
    if not isinstance(source_panel, Mapping):
        raise ValueError("source panel must be a JSON object")
    if not isinstance(qwen_rows, list) or any(
        not isinstance(row, Mapping) for row in qwen_rows
    ):
        raise ValueError("Qwen predictions must be a JSON list of objects")
    pilot_video_id = _verify_successful_pilot(
        source_panel,
        args.successful_pilot_results,
        args.successful_pilot_freeze,
    )
    payload = build_phase0_panel(source_panel, pilot_video_id, qwen_rows)
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("built Phase 0 panel is missing bindings")
    bindings["source_panel_file_sha256"] = _file_sha256(args.source_panel)
    bindings["qwen_prediction_file_sha256"] = _file_sha256(args.qwen_predictions)
    validated = validate_phase0_panel(
        payload, source_panel=source_panel, qwen_rows=qwen_rows
    )
    write_json_atomically(output_path, validated.to_payload())
    print(
        json.dumps(_safe_summary(validated.to_payload(), output_path), sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
