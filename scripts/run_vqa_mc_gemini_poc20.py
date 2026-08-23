#!/usr/bin/env python3
"""Run the label-isolated 20-video MC POC with Gemini native video/audio."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026 import vqa_gemini37_flash_smoke as smoke
from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    OPTION_IDS,
    canonical_json,
    parse_ranking,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "native-av-lowres-4fps-high-thinking"
VIDEO_FPS = 4.0
MAX_OUTPUT_TOKENS = 2048
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
ESTIMATED_INPUT_TOKENS_PER_SECOND = 1_600
ESTIMATED_TEXT_TOKENS_PER_REQUEST = 250
MAX_COST_USD = 0.40
EXPECTED_ROWS = 20
EXPECTED_VARIANTS = 27
MIME_TYPE = "video/webm"
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")
DEFAULT_REQUESTS = ROOT / "data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json"
DEFAULT_HISTORICAL_ROOT = ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1"
DEFAULT_OUTPUT = ROOT / "data/vqa/benchmarks/vqa-mc-poc20-20260821/gemini-a1"
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


def _reject_labels(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label field {key!r} at {location}")
            _reject_labels(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_labels(child, f"{location}[{index}]")


def _validate_options(row: Mapping[str, Any]) -> None:
    options = row.get("options")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("each row must contain four options")
    ids = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("option is not an object")
        option_id = option.get("option_id")
        text = option.get("text")
        if not isinstance(option_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("option ID or text is invalid")
        ids.append(option_id)
    if ids != list(OPTION_IDS):
        raise ValueError("options must use canonical A-D IDs")


def load_requests(
    path: Path,
    *,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _reject_labels(payload)
    if not isinstance(payload, dict):
        raise ValueError("request bundle must be an object")
    rows = payload.get("rows")
    variants = payload.get("variants")
    if not isinstance(rows, list) or not isinstance(variants, list):
        raise ValueError("request bundle must contain rows and variants")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"request bundle must contain {expected_rows} rows")
    if expected_variants is not None and len(variants) != expected_variants:
        raise ValueError(f"request bundle must contain {expected_variants} variants")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("request row is invalid")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in indexed:
            raise ValueError("video IDs must be unique nonempty strings")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"question is invalid for {video_id}")
        if not isinstance(row.get("duration_seconds"), (int, float)):
            raise ValueError(f"duration is invalid for {video_id}")
        _validate_options(row)
        indexed[video_id] = row
    seen_variants: set[str] = set()
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("variant is invalid")
        variant_id = variant.get("variant_id")
        video_id = variant.get("video_id")
        display_order = variant.get("display_order")
        if not isinstance(variant_id, str) or variant_id in seen_variants:
            raise ValueError("variant IDs must be unique strings")
        if video_id not in indexed:
            raise ValueError(f"variant has unknown video ID: {video_id}")
        if not isinstance(display_order, list) or sorted(display_order) != list(OPTION_IDS):
            raise ValueError(f"variant display order is invalid: {variant_id}")
        seen_variants.add(variant_id)
    return payload


def _prompt(row: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    by_id = {option["option_id"]: option["text"] for option in row["options"]}
    option_lines = "\n".join(
        f"{option_id}. {by_id[option_id]}" for option_id in variant["display_order"]
    )
    return (
        "Watch and listen to the complete video. Rank all four answer options from most "
        "likely to least likely. Return only JSON matching this exact shape: "
        '{"ranking":["A","B","C","D"]}. Every option ID must appear exactly once.\n\n'
        f"Question: {row['question']}\nOptions (displayed order):\n{option_lines}"
    )


def build_request(
    *, row: Mapping[str, Any], variant: Mapping[str, Any], upload: Mapping[str, Any]
) -> types.InlinedRequest:
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type", MIME_TYPE)
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload file URI is missing")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload MIME type is invalid")
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
        ),
    )
    prompt = types.Part(text=_prompt(row, variant))
    return types.InlinedRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[video, prompt])],
        metadata={
            "arm": MODEL_ARM,
            "video_id": str(row["video_id"]),
            "variant_id": str(variant["variant_id"]),
        },
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=RANKING_SCHEMA,
            seed=42,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=types.ThinkingLevel.HIGH,
            ),
        ),
    )


def select_variants(
    payload: Mapping[str, Any], *, variant_kind: str | None = None
) -> list[Mapping[str, Any]]:
    variants = payload["variants"]
    if variant_kind is None:
        return list(variants)
    selected = [row for row in variants if row.get("variant_kind") == variant_kind]
    if not selected:
        raise ValueError(f"no variants match kind: {variant_kind}")
    return selected


def estimate_cost(
    payload: Mapping[str, Any], *, variant_kind: str | None = None
) -> dict[str, Any]:
    indexed = {row["video_id"]: row for row in payload["rows"]}
    variants = select_variants(payload, variant_kind=variant_kind)
    seconds = sum(
        float(indexed[variant["video_id"]]["duration_seconds"])
        for variant in variants
    )
    request_count = len(variants)
    prompt_tokens = math.ceil(seconds * ESTIMATED_INPUT_TOKENS_PER_SECOND) + request_count * ESTIMATED_TEXT_TOKENS_PER_REQUEST
    output_tokens = request_count * MAX_OUTPUT_TOKENS
    cost = prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    return {
        "request_count": request_count,
        "upload_count": len(payload["rows"]),
        "video_seconds": round(seconds, 3),
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def _delivery_path(historical_root: Path, video_id: str) -> Path:
    root = historical_root.resolve(strict=True)
    candidate = root / "deliveries" / video_id / "delivery-media" / f"{video_id}.webm"
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"delivery escapes historical root for {video_id}") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"preserved WebM delivery is missing for {video_id}")
    return candidate


def preflight(
    *,
    requests_path: Path = DEFAULT_REQUESTS,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    max_cost_usd: float = MAX_COST_USD,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
    variant_kind: str | None = None,
) -> dict[str, Any]:
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    estimate = estimate_cost(payload, variant_kind=variant_kind)
    if estimate["estimated_cost_usd"] > max_cost_usd:
        raise ValueError("Gemini estimate exceeds authorized cost cap")
    deliveries = []
    for row in payload["rows"]:
        path = _delivery_path(historical_root, row["video_id"])
        deliveries.append(
            {
                "video_id": row["video_id"],
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"status": "preflight_passed", **estimate, "deliveries": deliveries}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value))


def _upload(client: Any, delivery: Path, *, sleep: Callable[[float], None]) -> dict[str, Any]:
    uploaded = client.files.upload(
        file=delivery,
        config=types.UploadFileConfig(mime_type=MIME_TYPE),
    )
    current = smoke._provider_file_metadata(uploaded, expected_mime=MIME_TYPE)
    for index in range(30):
        if current["file_state"] == "ACTIVE":
            return {"file_uri": current["uri"], "mime_type": MIME_TYPE, "provider": current}
        if current["file_state"] == "failed":
            raise ValueError("provider upload failed")
        if index == 29:
            raise ValueError("provider upload did not become active")
        sleep(1.0)
        current = smoke._provider_file_metadata(
            client.files.get(name=current["name"]), expected_mime=MIME_TYPE
        )
    raise AssertionError("unreachable")


def _response_text(raw: Any) -> tuple[dict[str, Any], str, Mapping[str, Any] | None]:
    safe = smoke._safe_response_object(raw)
    payload = smoke._response_payload(safe)
    text = smoke._response_text(payload)
    return safe, text.strip() if isinstance(text, str) else "", smoke._usage(payload)


def run(
    *,
    requests_path: Path = DEFAULT_REQUESTS,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT,
    client: Any,
    sleep: Callable[[float], None] = time.sleep,
    variant_kind: str | None = None,
) -> dict[str, Any]:
    gate = preflight(
        requests_path=requests_path,
        historical_root=historical_root,
        variant_kind=variant_kind,
    )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    raw_root = output_root / "raw"
    raw_root.mkdir()
    payload = load_requests(requests_path)
    indexed = {row["video_id"]: row for row in payload["rows"]}
    variants = select_variants(payload, variant_kind=variant_kind)
    _write_json(
        output_root / "run-manifest.json",
        {
            "schema_version": 1,
            "experiment_id": payload["experiment_id"],
            "model": MODEL,
            "model_arm": MODEL_ARM,
            "variant_kind": variant_kind,
            "requests_sha256": sha256_file(requests_path),
            "cost": {key: value for key, value in gate.items() if key != "deliveries"},
            "deliveries": gate["deliveries"],
        },
    )
    uploads = {
        row["video_id"]: _upload(
            client, _delivery_path(historical_root, row["video_id"]), sleep=sleep
        )
        for row in payload["rows"]
    }
    response_records: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        row = indexed[variant["video_id"]]
        request = build_request(row=row, variant=variant, upload=uploads[row["video_id"]])
        try:
            raw = client.models.generate_content(
                model=MODEL, contents=request.contents, config=request.config
            )
            safe_raw = smoke._safe_response_object(raw)
            raw_record = {"status": "received", "response": safe_raw}
        except Exception as error:  # noqa: BLE001 - preserve one terminal request failure
            raw_record = {"status": "failed", "error_type": smoke._safe_error(error)}
        raw_path = raw_root / f"{index:02d}.json"
        _write_json(raw_path, raw_record)
        response_records.append(
            {
                "variant_id": variant["variant_id"],
                "raw_sha256": sha256_file(raw_path),
                "status": raw_record["status"],
            }
        )
    _write_json(
        output_root / "raw-freeze.json",
        {"schema_version": 1, "status": "raw_frozen", "rows": response_records},
    )
    predictions: list[dict[str, Any]] = []
    prompt_tokens = 0
    output_tokens = 0
    complete_usage = True
    for index, (variant, response_record) in enumerate(
        zip(variants, response_records, strict=True)
    ):
        raw_record = json.loads((raw_root / f"{index:02d}.json").read_text())
        ranking = None
        usage = None
        error_type = raw_record.get("error_type")
        status = "failed"
        if response_record["status"] == "received":
            try:
                _, answer, usage = _response_text(raw_record["response"])
                ranking = parse_ranking(answer)
                status = "passed"
                error_type = None
            except Exception as error:  # noqa: BLE001 - freeze parser failure
                error_type = smoke._safe_error(error)
            if (
                isinstance(usage, Mapping)
                and isinstance(usage.get("prompt_tokens"), int)
                and isinstance(usage.get("output_tokens"), int)
            ):
                prompt_tokens += usage["prompt_tokens"]
                output_tokens += usage["output_tokens"]
            else:
                complete_usage = False
        else:
            complete_usage = False
        predictions.append(
            {
                "variant_id": variant["variant_id"],
                "video_id": variant["video_id"],
                "variant_kind": variant["variant_kind"],
                "status": status,
                "ranking": ranking,
                "usage": usage,
                "error_type": error_type,
            }
        )
    observed_cost = None
    if complete_usage:
        observed_cost = prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    prediction_freeze = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model": MODEL,
        "raw_freeze_sha256": sha256_file(output_root / "raw-freeze.json"),
        "rows": predictions,
    }
    _write_json(output_root / "predictions-freeze.json", prediction_freeze)
    summary = {
        "status": "completed",
        "request_count": len(predictions),
        "valid_count": sum(row["status"] == "passed" for row in predictions),
        "invalid_count": sum(row["status"] != "passed" for row in predictions),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "observed_cost_usd": round(observed_cost, 8) if observed_cost is not None else None,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def _api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    if PRIVATE_ENV_FILE.stat().st_mode & 0o777 != 0o600:
        raise ValueError("Gemini credential file permissions are not 0600")
    for line in PRIVATE_ENV_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        if key.strip() == "GEMINI_API_KEY":
            parsed = shlex.split(raw.strip(), comments=True)
            if parsed and parsed[0].strip():
                return parsed[0].strip()
    raise ValueError("GEMINI_API_KEY is unavailable")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_vqa_mc_gemini_poc20")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-only", action="store_true")
    args = parser.parse_args(argv)
    variant_kind = "canonical" if args.canonical_only else None
    try:
        if args.preflight:
            result = preflight(
                requests_path=args.requests,
                historical_root=args.historical_root,
                variant_kind=variant_kind,
            )
            result = {key: value for key, value in result.items() if key != "deliveries"}
        else:
            result = run(
                requests_path=args.requests,
                historical_root=args.historical_root,
                output_root=args.output,
                client=genai.Client(api_key=_api_key()),
                variant_kind=variant_kind,
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:  # noqa: BLE001 - do not leak paths or provider responses
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
