#!/usr/bin/env python3
"""Run the 20-video MC POC through matched Qwen3.6 native/transcript views."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    CONDITIONS,
    OPTION_IDS,
    canonical_json,
    parse_ranking,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@sha256:"
    "77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
EXPECTED_ROWS = 20
EXPECTED_VARIANTS = 27
EXPECTED_REQUESTS = 54
MAX_OUTPUT_TOKENS = 64
DEFAULT_REQUESTS = ROOT / "data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json"
DEFAULT_MEDIA_ROOT = ROOT / "data/vqa/2025/training/videos_original"
DEFAULT_OUTPUT = ROOT / "data/vqa/benchmarks/vqa-mc-poc20-20260821/qwen-a1"
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "correct_answer",
        "incorrect_answers",
        "correct_option",
        "correct_option_id",
        "expected_answer",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_answer",
    }
)
RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {"type": "string", "enum": list(OPTION_IDS)},
            "minItems": 4,
            "maxItems": 4,
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}
_SAFE_ERROR = re.compile(r"[^a-z0-9_-]+")


def _reject_labels(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label field {key!r} at {location}")
            _reject_labels(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_labels(child, f"{location}[{index}]")


def _resolve_media(media_root: Path, row: Mapping[str, Any]) -> Path:
    video_id = str(row["video_id"])
    source_path = Path(str(row["source_path"]))
    candidates = (
        media_root / f"{video_id}.mkv",
        media_root / f"{video_id}.mp4",
        media_root / source_path.name,
        media_root / source_path,
    )
    media_root_resolved = media_root.resolve(strict=True)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(media_root_resolved)
        except (OSError, ValueError):
            continue
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if sha256_file(candidate) != row.get("source_sha256"):
            raise ValueError(f"video SHA-256 mismatch for {video_id}")
        return candidate
    raise ValueError(f"media is unavailable for {video_id}")


def load_requests(
    path: Path,
    *,
    media_root: Path = DEFAULT_MEDIA_ROOT,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _reject_labels(payload)
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be an object")
    rows = payload.get("rows")
    variants = payload.get("variants")
    conditions = payload.get("conditions")
    if not isinstance(rows, list) or not isinstance(variants, list):
        raise ValueError("request bundle must contain rows and variants")
    if conditions != list(CONDITIONS):
        raise ValueError("Qwen conditions do not match the frozen dual-view contract")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"request bundle must contain {expected_rows} rows")
    if expected_variants is not None and len(variants) != expected_variants:
        raise ValueError(f"request bundle must contain {expected_variants} variants")
    indexed: dict[str, dict[str, Any]] = {}
    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("request row is invalid")
        row = dict(raw)
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in indexed:
            raise ValueError("video IDs must be unique nonempty strings")
        options = row.get("options")
        if not isinstance(options, list) or [option.get("option_id") for option in options if isinstance(option, Mapping)] != list(OPTION_IDS):
            raise ValueError(f"options are invalid for {video_id}")
        transcript = row.get("transcript")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"transcript is unavailable for {video_id}")
        if sha256_bytes(transcript.encode("utf-8")) != row.get("transcript_sha256"):
            raise ValueError(f"transcript SHA-256 mismatch for {video_id}")
        row["video_path"] = str(_resolve_media(media_root, row))
        indexed[video_id] = row
        normalized_rows.append(row)
    seen: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("variant is invalid")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or variant_id in seen:
            raise ValueError("variant IDs must be unique strings")
        if variant.get("video_id") not in indexed:
            raise ValueError(f"unknown variant video: {variant.get('video_id')}")
        display_order = variant.get("display_order")
        if not isinstance(display_order, list) or sorted(display_order) != list(OPTION_IDS):
            raise ValueError(f"display order is invalid for {variant_id}")
        seen.add(variant_id)
    payload["rows"] = normalized_rows
    return payload


def _prompt(
    row: Mapping[str, Any], variant: Mapping[str, Any], condition_id: str
) -> str:
    by_id = {option["option_id"]: option["text"] for option in row["options"]}
    option_lines = "\n".join(
        f"{option_id}. {by_id[option_id]}" for option_id in variant["display_order"]
    )
    text = (
        "Use the complete native video and audio to rank all four answer options from "
        "most likely to least likely. Return only JSON matching this exact shape: "
        '{"ranking":["A","B","C","D"]}. Every option ID must appear exactly once.\n\n'
        f"Question: {row['question']}\nOptions (displayed order):\n{option_lines}"
    )
    if condition_id == "video-plus-transcript":
        text += f"\n\nTimestamped Whisper transcript:\n{row['transcript']}"
    return text


def build_request_specs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    indexed = {row["video_id"]: row for row in payload["rows"]}
    variants_by_video: dict[str, list[Mapping[str, Any]]] = {
        video_id: [] for video_id in indexed
    }
    for variant in payload["variants"]:
        variants_by_video[variant["video_id"]].append(variant)
    specs: list[dict[str, Any]] = []
    for video_id, row in indexed.items():
        for variant in variants_by_video[video_id]:
            for condition_id in CONDITIONS:
                specs.append(
                    {
                        "request_id": f"{variant['variant_id']}:{condition_id}",
                        "video_id": video_id,
                        "variant_id": variant["variant_id"],
                        "variant_kind": variant["variant_kind"],
                        "condition_id": condition_id,
                        "video_path": row["video_path"],
                        "video_sha256": row["source_sha256"],
                        "transcript_sha256": row["transcript_sha256"] if condition_id == "video-plus-transcript" else None,
                        "prompt": _prompt(row, variant, condition_id),
                    }
                )
    return specs


def video_data_uri(video_path: Path) -> str:
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/x-matroska"
    return f"data:{mime_type};base64,{base64.b64encode(video_path.read_bytes()).decode('ascii')}"


def build_payload(spec: Mapping[str, Any], *, data_uri: str) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_uri}},
                    {"type": "text", "text": spec["prompt"]},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "complete_option_ranking",
                "strict": True,
                "schema": RANKING_SCHEMA,
            },
        },
    }


def post_json(url: str, payload: Mapping[str, Any]) -> bytes:
    request = urllib_request.Request(
        url,
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=600.0) as response:
        return response.read()


def _safe_error(error: BaseException) -> str:
    value = _SAFE_ERROR.sub("_", type(error).__name__.casefold()).strip("_")
    return value or "request_failed"


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value))


def _extract_content(raw: bytes) -> str:
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("provider response is not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("provider response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("provider response has no content")
    return message["content"].strip()


def preflight(
    *,
    requests_path: Path,
    media_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    payload = load_requests(requests_path, media_root=media_root)
    specs = build_request_specs(payload)
    if len(specs) != EXPECTED_REQUESTS:
        raise ValueError(f"Qwen contract must produce {EXPECTED_REQUESTS} requests")
    result = {
        "status": "preflight_passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "video_count": len(payload["rows"]),
        "variant_count": len(payload["variants"]),
        "request_count": len(specs),
        "condition_ids": list(CONDITIONS),
    }
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_exclusive(output_root / "preflight.json", result)
    return result


def run(
    *,
    requests_path: Path,
    media_root: Path,
    output_root: Path,
    server_url: str,
    post: Callable[[str, Mapping[str, Any]], bytes | Mapping[str, Any]] = post_json,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    payload = load_requests(
        requests_path,
        media_root=media_root,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    specs = build_request_specs(payload)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw-responses"
    raw_root.mkdir()
    _write_exclusive(
        output_root / "requests-freeze.json",
        {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "requests_sha256": sha256_file(requests_path),
            "requests": [
                {key: value for key, value in spec.items() if key != "video_path"}
                for spec in specs
            ],
        },
    )
    sender = post
    response_records: list[dict[str, Any]] = []
    current_video_id: str | None = None
    current_data_uri: str | None = None
    for index, spec in enumerate(specs):
        if spec["video_id"] != current_video_id:
            current_video_id = str(spec["video_id"])
            current_data_uri = video_data_uri(Path(str(spec["video_path"])))
        assert current_data_uri is not None
        started = time.monotonic()
        try:
            response = sender(server_url, build_payload(spec, data_uri=current_data_uri))
            raw = canonical_json(response) if isinstance(response, Mapping) else bytes(response)
            if not raw:
                raise ValueError("provider response is empty")
            status = "received"
            error_type = None
        except Exception as error:  # noqa: BLE001 - one terminal request attempt
            error_type = _safe_error(error)
            raw = canonical_json({"error_type": error_type})
            status = "failed"
        raw_path = raw_root / f"{index:02d}.json"
        with raw_path.open("xb") as handle:
            handle.write(raw)
        response_records.append(
            {
                "request_id": spec["request_id"],
                "status": status,
                "error_type": error_type,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                "raw_sha256": sha256_bytes(raw),
            }
        )
    _write_exclusive(
        output_root / "raw-freeze.json",
        {"schema_version": 1, "status": "raw_frozen", "rows": response_records},
    )
    predictions: list[dict[str, Any]] = []
    for index, (spec, record) in enumerate(zip(specs, response_records, strict=True)):
        status = record["status"]
        error_type = record["error_type"]
        ranking = None
        if status == "received":
            try:
                ranking = parse_ranking(_extract_content((raw_root / f"{index:02d}.json").read_bytes()))
                status = "passed"
            except Exception as error:  # noqa: BLE001 - freeze parser failure category
                status = "failed"
                error_type = _safe_error(error)
        predictions.append(
            {
                "request_id": spec["request_id"],
                "video_id": spec["video_id"],
                "variant_id": spec["variant_id"],
                "variant_kind": spec["variant_kind"],
                "condition_id": spec["condition_id"],
                "status": status,
                "ranking": ranking,
                "error_type": error_type,
                "raw_sha256": record["raw_sha256"],
            }
        )
    prediction_freeze = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "raw_freeze_sha256": sha256_file(output_root / "raw-freeze.json"),
        "rows": predictions,
    }
    _write_exclusive(output_root / "predictions-freeze.json", prediction_freeze)
    summary = {
        "status": "completed",
        "request_count": len(predictions),
        "valid_count": sum(row["status"] == "passed" for row in predictions),
        "invalid_count": sum(row["status"] != "passed" for row in predictions),
    }
    _write_exclusive(output_root / "summary.json", summary)
    write_checksums(output_root)
    return summary


def write_checksums(root: Path) -> Path:
    target = root / "SHA256SUMS"
    excluded = {target, root / "vllm-server.log"}
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(path for path in root.rglob("*") if path.is_file() and path not in excluded)
    ]
    with target.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_vqa_mc_qwen36_poc20")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument(
        "--media-root", type=Path, default=Path(os.environ.get("MEDIA_ROOT", DEFAULT_MEDIA_ROOT))
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1/chat/completions")
    args = parser.parse_args(argv)
    if args.preflight:
        result = preflight(
            requests_path=args.requests,
            media_root=args.media_root,
            output_root=args.output,
        )
    else:
        result = run(
            requests_path=args.requests,
            media_root=args.media_root,
            output_root=args.output,
            server_url=args.server_url,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
