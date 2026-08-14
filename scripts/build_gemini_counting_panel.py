#!/usr/bin/env python3
"""Freeze the corrected count census and label-blind issue #34 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_counting_evidence import build_counting_panel


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting freeze: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def prepare(
    *, frozen_root: Path, annotations_root: Path, adjudications_path: Path
) -> dict[str, object]:
    protocol_path = frozen_root / "protocol.json"
    predictions_path = frozen_root / "predictions.direct-answer.json"
    maps_path = frozen_root / "maps.question-blind.json"
    protocol = load_json(protocol_path)
    predictions = load_json(predictions_path)
    maps = load_json(maps_path)
    if not isinstance(protocol, dict) or not isinstance(protocol.get("rows"), list):
        raise TypeError("protocol.json must contain rows")
    if not isinstance(predictions, list) or not isinstance(maps, list):
        raise TypeError("prediction and map freezes must be lists")

    annotations: dict[str, dict[str, object]] = {}
    uploads: dict[str, dict[str, object]] = {}
    for row in protocol["rows"]:
        video_id = row["video_id"]
        annotation_path = annotations_root / f"{video_id}.json"
        upload_path = frozen_root / "uploads" / f"{video_id}.json"
        annotation = load_json(annotation_path)
        upload = load_json(upload_path)
        if not isinstance(annotation, dict) or not isinstance(upload, dict):
            raise TypeError(f"invalid source record for {video_id}")
        annotations[video_id] = annotation
        uploads[video_id] = upload

    adjudications = load_json(adjudications_path)
    if not isinstance(adjudications, dict):
        raise TypeError("adjudications must be an object")
    artifacts = build_counting_panel(
        protocol_rows=protocol["rows"],
        predictions=predictions,
        maps=maps,
        annotations=annotations,
        uploads=uploads,
        adjudications=adjudications,
    )
    artifacts["provenance"] = {
        "schema_version": 1,
        "frozen_root": str(frozen_root.resolve()),
        "inputs": {
            "protocol.json": file_sha256(protocol_path),
            "predictions.direct-answer.json": file_sha256(predictions_path),
            "maps.question-blind.json": file_sha256(maps_path),
            "predictions-freeze.json": file_sha256(
                frozen_root / "predictions-freeze.json"
            ),
            "adjudications.json": file_sha256(adjudications_path),
        },
    }
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--annotations-root", required=True, type=Path)
    parser.add_argument("--adjudications", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = prepare(
        frozen_root=args.frozen_root,
        annotations_root=args.annotations_root,
        adjudications_path=args.adjudications,
    )
    names = {
        "census": "census.json",
        "panel": "panel.freeze.json",
        "requests": "requests.label-blind.json",
        "provenance": "preparation-provenance.json",
    }
    for key, filename in names.items():
        write_once_json(args.output_root / filename, artifacts[key])
    print(
        json.dumps(
            {
                "census_counts": artifacts["census"]["counts"],
                "panel_rows": len(artifacts["panel"]["rows"]),
                "canary_rows": len(artifacts["requests"]["canary_video_ids"]),
                "request_manifest_sha256": artifacts["requests"]["complete_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
