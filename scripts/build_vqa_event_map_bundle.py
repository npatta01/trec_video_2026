#!/usr/bin/env python3
"""Build the private label-free controller and immutable event-map requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from trec_video_2026.vqa_experiments import nemotron3_omni_event_map as event_map

RUN_ID = "vqa-event-map-inputs-20260809-a3"
EXPECTED_COUNT = event_map.EXPECTED_ROW_COUNT
EXPECTED_IDS_SHA256 = event_map.CANONICAL_PANEL_IDS_SHA256
SHARD_COUNTS = event_map.SHARD_COUNTS
RESERVED_FILES = {
    "event-map-controller.json",
    "event-map-media-manifest.json",
    "event-map-protocol.json",
    "event-map-summary.json",
    "event-map-canary.jsonl",
    *(f"v-map-chunk-{index:02d}.jsonl" for index in range(4)),
    *(f"av-map-chunk-{index:02d}.jsonl" for index in range(4)),
    "SHA256SUMS",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answers",
        "annotation",
        "annotations",
        "correctanswer",
        "expectedanswer",
        "groundtruth",
        "label",
        "labels",
        "reference",
        "referenceanswer",
        "metric",
        "metrics",
        "score",
        "scores",
        "prediction",
        "predictions",
        "transcript",
        "transcripts",
        "whisper",
        "whispersegments",
        "selectedtranscript",
    }
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}") from error


def _reject_labels(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in _FORBIDDEN_KEYS or any(
                token in normalized for token in ("reference", "groundtruth", "correctanswer")
            ):
                raise ValueError(f"label-bearing field at {location}.{key}")
            _reject_labels(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_labels(nested, location=f"{location}[{index}]")


def _sha256(path: Path) -> str:
    return event_map.sha256_file(path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _ordered_id_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return event_map.ordered_ids_sha256(rows)


def partition_rows(rows: Sequence[Mapping[str, Any]], *, chunk_count: int) -> list[list[dict[str, Any]]]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    base, remainder = divmod(len(rows), chunk_count)
    result: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(chunk_count):
        size = base + (1 if index < remainder else 0)
        result.append([dict(row) for row in rows[start : start + size]])
        start += size
    return result


def _safe_media_name(video_id: str, media_path: str) -> str:
    pure = PurePosixPath(media_path)
    name = pure.name
    if pure.is_absolute() or ".." in pure.parts or pure.suffix.casefold() not in {".mkv", ".mp4", ".webm", ".mov"}:
        raise ValueError(f"{video_id} media path is unsafe")
    if PurePosixPath(name).stem != video_id:
        raise ValueError(f"{video_id} media path does not match video ID")
    return name


def _indexed(rows: Any, *, source: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        rows = rows.get("examples", rows.get("rows", rows.get("files")))
    if not isinstance(rows, list):
        raise ValueError(f"{source} must contain a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("video_id"), str):
            raise ValueError(f"{source} rows must have video_id")
        if row["video_id"] in indexed:
            raise ValueError(f"{source} has duplicate video_id")
        indexed[row["video_id"]] = row
    return indexed


def _source_rows(manifest: Mapping[str, Any], expected_count: int) -> list[Mapping[str, Any]]:
    examples = manifest.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_count:
        raise ValueError(f"manifest must contain exactly {expected_count} rows")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(examples):
        if not isinstance(row, Mapping) or set(row) != {"video_id", "question"}:
            raise ValueError(f"manifest row {index} must contain only video_id and question")
        if not isinstance(row["video_id"], str) or not row["video_id"]:
            raise ValueError("manifest video IDs must be nonempty")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("manifest questions must be nonempty")
        rows.append(row)
    ids = [row["video_id"] for row in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest IDs must be unique and canonical sorted order")
    return rows


def _row_media_fields(row: Mapping[str, Any]) -> tuple[str, str, float]:
    path = row.get("media_path", row.get("source_media_path", row.get("path")))
    digest = row.get("media_sha256", row.get("source_media_sha256", row.get("sha256")))
    duration = row.get("duration_seconds", row.get("source_duration_seconds", row.get("duration")))
    if not isinstance(path, str) or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("media inventory path/hash is invalid")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError("media duration must be positive")
    return path, digest, float(duration)


def _request_row(
    row: Mapping[str, Any], *, run_id: str, condition_id: str, slice_id: str
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "condition_id": condition_id,
        "slice_id": slice_id,
        "video_id": row["video_id"],
        "question": row["question"],
        "question_stratum": row["question_stratum"],
        "media_path": row["media_path"],
        "media_sha256": row["media_sha256"],
        "duration_seconds": row["duration_seconds"],
    }
    result["request_sha256"] = event_map.canonical_sha256(result)
    return result


def _request_identity(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = path.read_bytes()
    request_ids = "\n".join(
        f"{row.get('condition_id', '')}/{row.get('video_id', '')}" for row in rows
    ) + "\n"
    return {
        "row_count": len(rows),
        "ids_sha256": hashlib.sha256(request_ids.encode("utf-8")).hexdigest(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_bundle(
    *,
    manifest: Path,
    evidence: Path,
    media_root: Path,
    output_dir: Path,
    run_id: str = RUN_ID,
    media_inventory: Path | None = None,
    expected_count: int = EXPECTED_COUNT,
    expected_ids_sha256: str | None = EXPECTED_IDS_SHA256,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = _load_json(manifest)
    evidence_payload = _load_json(evidence)
    _reject_labels(manifest_payload)
    _reject_labels(evidence_payload)
    if not isinstance(manifest_payload, Mapping) or not isinstance(evidence_payload, Mapping):
        raise ValueError("manifest and evidence must be objects")
    source_rows = _source_rows(manifest_payload, expected_count)
    ids_sha256 = _ordered_id_digest(source_rows)
    if expected_ids_sha256 is not None and ids_sha256 != expected_ids_sha256:
        raise ValueError("canonical panel ID digest mismatch")
    evidence_rows = _indexed(evidence_payload, source="evidence")
    inventory_rows = _indexed(_load_json(media_inventory), source="media inventory") if media_inventory else {}
    if set(evidence_rows) != {row["video_id"] for row in source_rows}:
        raise ValueError("evidence IDs do not match manifest")
    if inventory_rows and set(inventory_rows) != set(evidence_rows):
        raise ValueError("media inventory IDs do not match manifest")

    controller_rows: list[dict[str, Any]] = []
    for source in source_rows:
        video_id = source["video_id"]
        evidence_row = evidence_rows[video_id]
        inventory_row = inventory_rows.get(video_id, {})
        if evidence_row.get("question") != source["question"]:
            raise ValueError(f"{video_id} question mismatch")
        path_value, digest_value, duration_value = _row_media_fields(evidence_row)
        if inventory_row:
            inventory_path, inventory_digest, inventory_duration = _row_media_fields(inventory_row)
            if (inventory_digest, inventory_duration) != (digest_value, duration_value):
                raise ValueError(f"{video_id} media inventory identity mismatch")
            path_value = inventory_path
        media_name = _safe_media_name(video_id, path_value)
        media_path = media_root / media_name
        if media_path.is_symlink() or not media_path.is_file():
            raise ValueError(f"{video_id} media is missing")
        if _sha256(media_path) != digest_value:
            raise ValueError(f"{video_id} media hash mismatch")
        controller_rows.append(
            {
                "video_id": video_id,
                "question": source["question"].strip(),
                "question_stratum": event_map.question_stratum(source["question"]),
                "media_path": f"media/{media_name}",
                "media_sha256": digest_value,
                "duration_seconds": duration_value,
            }
        )

    chunks = partition_rows(controller_rows, chunk_count=4)
    canary_rows = event_map.select_canary_rows(controller_rows)
    controller = {
        "schema_version": 1,
        "run_id": run_id,
        "label_policy": "label-free-controller",
        "panel": {
            "row_count": len(controller_rows),
            "ids_sha256": ids_sha256,
            "shard_counts": [len(chunk) for chunk in chunks],
            "canary_count": len(canary_rows),
            "canary_ids": [row["video_id"] for row in canary_rows],
            "canary_ids_sha256": _ordered_id_digest(canary_rows),
        },
        "rows": controller_rows,
    }
    media_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "file_count": len(controller_rows),
        "files": [
            {
                "video_id": row["video_id"],
                "relative_path": row["media_path"],
                "sha256": row["media_sha256"],
                "duration_seconds": row["duration_seconds"],
            }
            for row in controller_rows
        ],
    }
    controller_path = output_dir / "event-map-controller.json"
    media_manifest_path = output_dir / "event-map-media-manifest.json"
    _write_json(controller_path, controller)
    _write_json(media_manifest_path, media_manifest)
    controller_sha256 = _sha256(controller_path)
    media_manifest_sha256 = _sha256(media_manifest_path)
    source_bundle_sha256 = event_map.canonical_sha256(
        {"controller_sha256": controller_sha256, "media_manifest_sha256": media_manifest_sha256}
    )

    request_groups: dict[str, list[dict[str, Any]]] = {}
    canary_requests = [
        _request_row(row, run_id=run_id, condition_id=condition, slice_id="canary")
        for row in canary_rows
        for condition in event_map.CONDITION_ORDER
    ]
    request_groups["event-map-canary.jsonl"] = canary_requests
    for condition in event_map.CONDITION_ORDER:
        for index, chunk in enumerate(chunks):
            request_groups[f"{condition.replace('-', '-')}-chunk-{index:02d}.jsonl"] = [
                _request_row(row, run_id=run_id, condition_id=condition, slice_id=f"{index:02d}")
                for row in chunk
            ]
    # Keep the four v-map and four av-map files named exactly as the dstack jobs.
    request_groups = {
        (name.replace("v-map-chunk", "v-map-chunk").replace("av-map-chunk", "av-map-chunk")): rows
        for name, rows in request_groups.items()
    }
    protocol_without_hash = {
        "schema_version": 1,
        "run_id": run_id,
        "source_bundle_sha256": source_bundle_sha256,
        "controller_sha256": controller_sha256,
        "media_manifest_sha256": media_manifest_sha256,
        "panel": {
            "row_count": len(controller_rows),
            "ids_sha256": ids_sha256,
            "shard_counts": list(SHARD_COUNTS) if len(controller_rows) == EXPECTED_COUNT else [len(chunk) for chunk in chunks],
        },
        "canary": {
            "paired": True,
            "row_count": len(canary_rows),
            "ids_sha256": _ordered_id_digest(canary_rows),
            "request_row_count": len(canary_requests),
            "strata": list(event_map.CANARY_STRATA),
        },
        "conditions": list(event_map.CONDITION_ORDER),
        "model": {
            "model_id": event_map.MODEL_ID,
            "revision": event_map.MODEL_REVISION,
            "runtime_image": event_map.RUNTIME_IMAGE,
            "runtime_image_digest": event_map.RUNTIME_IMAGE_DIGEST,
        },
        "decoder": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "enable_thinking": False,
            "max_output_tokens": event_map.MAX_OUTPUT_TOKENS,
            "max_retries": 1,
            "response_schema_sha256": event_map.canonical_sha256(event_map.EVENT_MAP_RESPONSE_SCHEMA),
            "prompt_sha256": event_map.canonical_sha256(event_map.EVENT_MAP_PROMPT),
            "retry_prompt_sha256": event_map.canonical_sha256(
                event_map.EVENT_MAP_RETRY_PROMPT
            ),
            "measured_duration_in_prompt": True,
        },
        "video_sampling": {
            "fps": event_map.VIDEO_FPS,
            "max_frames": event_map.VIDEO_MAX_FRAMES,
            "video_pruning_rate": event_map.VIDEO_PRUNING_RATE,
            "kv_cache_dtype": event_map.KV_CACHE_DTYPE,
            "reasoning_parser": event_map.REASONING_PARSER,
        },
        "inference_module_sha256": event_map.sha256_file(
            Path(event_map.__file__).resolve(strict=True)
        ),
        "request_files": {},
    }
    protocol_path = output_dir / "event-map-protocol.json"
    for name, rows in request_groups.items():
        path = output_dir / name
        path.write_bytes(_jsonl_bytes(rows))
        protocol_without_hash["request_files"][name] = _request_identity(path, rows)
    protocol = {
        **protocol_without_hash,
        "protocol_sha256": event_map.canonical_sha256(protocol_without_hash),
    }
    _write_json(protocol_path, protocol)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "row_count": len(controller_rows),
        "ids_sha256": ids_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "controller_sha256": controller_sha256,
        "protocol_sha256": protocol["protocol_sha256"],
        "request_set_sha256": event_map.canonical_sha256(request_groups),
    }
    summary_path = output_dir / "event-map-summary.json"
    _write_json(summary_path, summary)
    payload_paths = [path for path in output_dir.iterdir() if path.name != "SHA256SUMS"]
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  ./{path.name}\n" for path in sorted(payload_paths, key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    summary["bundle_sha256"] = _sha256(checksum_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--media-inventory", type=Path)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--expected-ids-sha256", default=EXPECTED_IDS_SHA256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_bundle(
        manifest=args.manifest,
        evidence=args.evidence,
        media_inventory=args.media_inventory,
        media_root=args.media_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_count=args.expected_count,
        expected_ids_sha256=args.expected_ids_sha256,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
