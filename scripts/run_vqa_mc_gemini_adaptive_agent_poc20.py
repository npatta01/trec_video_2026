#!/usr/bin/env python3
"""Run the bounded two-stage Gemini inspection agent on the MC POC panel."""

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
from trec_video_2026.vqa_experiments.vqa_mc_adaptive_agent import (
    FINAL_SCHEMA,
    PLANNER_SCHEMA,
    InspectionPlan,
    choose_final_ranking,
    fallback_plan,
    parse_stage1,
    parse_stage2,
    plan_dict,
    stage1_prompt,
    stage2_prompt,
)
from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    OPTION_IDS,
    canonical_json,
    parse_ranking,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "adaptive-native-av-two-stage-v1"
STAGE1_FPS = 4.0
MAX_OUTPUT_TOKENS = 2048
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
STAGE1_TOKENS_PER_SECOND = 312
WORST_STAGE2_TOKENS_PER_SECOND = 2_272
ESTIMATED_TEXT_TOKENS_PER_REQUEST = 500
MAX_COST_USD = 0.75
EXPECTED_ROWS = 20
EXPECTED_VARIANTS = 27
MIME_TYPE = "video/webm"
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")
DEFAULT_REQUESTS = (
    ROOT / "data/vqa/benchmarks/vqa-mc-poc20-20260821/inputs/requests.json"
)
DEFAULT_HISTORICAL_ROOT = ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1"
DEFAULT_OUTPUT = (
    ROOT / "data/vqa/benchmarks/vqa-mc-adaptive-agent-poc20-20260821-a1"
)
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
RESOLUTION_LEVELS = {
    "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
}


def _reject_labels(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label field {key!r} at {location}")
            _reject_labels(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_labels(child, f"{location}[{index}]")


def _validate_options(row: Mapping[str, Any]) -> None:
    options = row.get("options")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("each row must contain four options")
    option_ids: list[str] = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("option is not an object")
        option_id = option.get("option_id")
        text = option.get("text")
        if not isinstance(option_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("option ID or text is invalid")
        option_ids.append(option_id)
    if option_ids != list(OPTION_IDS):
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
        question = row.get("question")
        duration = row.get("duration_seconds")
        if not isinstance(video_id, str) or not video_id or video_id in indexed:
            raise ValueError("video IDs must be unique nonempty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question is invalid for {video_id}")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
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
        if not isinstance(display_order, list) or sorted(display_order) != list(
            OPTION_IDS
        ):
            raise ValueError(f"variant display order is invalid: {variant_id}")
        seen_variants.add(variant_id)
    return payload


def canonical_variants(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = [
        row for row in payload["variants"] if row.get("variant_kind") == "canonical"
    ]
    if len(selected) != len(payload["rows"]):
        raise ValueError("canonical variants must cover every row exactly once")
    if any(variant.get("display_order") != list(OPTION_IDS) for variant in selected):
        raise ValueError("canonical variants must display options in A-D order")
    if len({variant["video_id"] for variant in selected}) != len(selected):
        raise ValueError("canonical variants contain duplicate video IDs")
    return selected


def estimate_worst_case_cost(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonical_variants(payload)
    indexed = {row["video_id"]: row for row in payload["rows"]}
    seconds = sum(float(indexed[row["video_id"]]["duration_seconds"]) for row in canonical)
    row_count = len(canonical)
    request_count = row_count * 2
    input_tokens = math.ceil(
        seconds * (STAGE1_TOKENS_PER_SECOND + WORST_STAGE2_TOKENS_PER_SECOND)
    ) + request_count * ESTIMATED_TEXT_TOKENS_PER_REQUEST
    output_tokens = request_count * MAX_OUTPUT_TOKENS
    cost = (
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    return {
        "row_count": row_count,
        "request_count": request_count,
        "upload_count": row_count,
        "video_seconds_per_stage": round(seconds, 3),
        "worst_case_input_tokens": input_tokens,
        "worst_case_output_tokens": output_tokens,
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
) -> dict[str, Any]:
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    estimate = estimate_worst_case_cost(payload)
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


def _video_part(
    *,
    upload: Mapping[str, Any],
    fps: float,
    resolution: str,
    start_offset: str | None = None,
    end_offset: str | None = None,
) -> types.Part:
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type", MIME_TYPE)
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload file URI is missing")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload MIME type is invalid")
    if resolution not in RESOLUTION_LEVELS:
        raise ValueError("media resolution is unsupported")
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(
            fps=fps,
            start_offset=start_offset,
            end_offset=end_offset,
        ),
        media_resolution=types.PartMediaResolution(
            level=RESOLUTION_LEVELS[resolution]
        ),
    )


def _config(schema: Mapping[str, Any]) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_json_schema=dict(schema),
        seed=42,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    )


def build_stage1_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any]
) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    _video_part(
                        upload=upload,
                        fps=STAGE1_FPS,
                        resolution="low",
                    ),
                    types.Part(text=stage1_prompt(row)),
                ],
            )
        ],
        metadata={"arm": MODEL_ARM, "stage": "stage1", "video_id": row["video_id"]},
        config=_config(PLANNER_SCHEMA),
    )


def build_stage2_request(
    *,
    row: Mapping[str, Any],
    upload: Mapping[str, Any],
    plan: InspectionPlan,
    stage1_ranking: Sequence[str] | None,
) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    _video_part(
                        upload=upload,
                        fps=plan.fps,
                        resolution=plan.resolution,
                        start_offset=f"{plan.start_seconds:.3f}s",
                        end_offset=f"{plan.end_seconds:.3f}s",
                    ),
                    types.Part(text=stage2_prompt(row, plan, stage1_ranking)),
                ],
            )
        ],
        metadata={"arm": MODEL_ARM, "stage": "stage2", "video_id": row["video_id"]},
        config=_config(FINAL_SCHEMA),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(value))


def _upload(
    client: Any, delivery: Path, *, sleep: Callable[[float], None]
) -> dict[str, Any]:
    uploaded = client.files.upload(
        file=delivery,
        config=types.UploadFileConfig(mime_type=MIME_TYPE),
    )
    current = smoke._provider_file_metadata(uploaded, expected_mime=MIME_TYPE)
    for index in range(30):
        if current["file_state"] == "ACTIVE":
            return {
                "file_uri": current["uri"],
                "mime_type": MIME_TYPE,
                "provider": current,
            }
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


def _provider_call(client: Any, request: types.InlinedRequest) -> dict[str, Any]:
    try:
        raw = client.models.generate_content(
            model=MODEL,
            contents=request.contents,
            config=request.config,
        )
        return {"status": "received", "response": smoke._safe_response_object(raw)}
    except Exception as error:  # noqa: BLE001 - preserve one terminal failure
        return {"status": "failed", "error_type": smoke._safe_error(error)}


def _usage_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0
    output_tokens = 0
    complete = True
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            complete = False
            continue
        prompt = usage.get("prompt_tokens")
        total = usage.get("total_tokens")
        candidates = usage.get("output_tokens")
        if not all(isinstance(value, int) for value in (prompt, total, candidates)):
            complete = False
            continue
        prompt_tokens += int(prompt)
        output_tokens += max(int(candidates), int(total) - int(prompt))
    observed_cost = None
    if complete:
        observed_cost = (
            prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
            + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
        )
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "usage_complete": complete,
        "observed_cost_usd": (
            round(observed_cost, 8) if observed_cost is not None else None
        ),
    }


def _parse_stage1_with_fallback(
    answer: str, *, duration_seconds: float
) -> tuple[list[str] | None, InspectionPlan, str, str | None]:
    try:
        parsed = parse_stage1(answer, duration_seconds=duration_seconds)
        return parsed["ranking"], parsed["inspection"], "passed", None
    except Exception as full_error:  # noqa: BLE001 - preserve strict parse failure
        ranking = None
        try:
            payload = json.loads(answer)
            if isinstance(payload, Mapping):
                ranking = parse_ranking({"ranking": payload.get("ranking")})
        except Exception:  # noqa: BLE001 - ranking remains invalid
            pass
        return (
            ranking,
            fallback_plan(duration_seconds),
            "passed" if ranking is not None else "failed",
            smoke._safe_error(full_error),
        )


def run(
    *,
    requests_path: Path = DEFAULT_REQUESTS,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT,
    client: Any,
    sleep: Callable[[float], None] = time.sleep,
    upload_file: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    gate = preflight(
        requests_path=requests_path,
        historical_root=historical_root,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    indexed = {row["video_id"]: row for row in payload["rows"]}
    variants = canonical_variants(payload)
    _write_json(
        output_root / "run-manifest.json",
        {
            "schema_version": 1,
            "experiment_id": payload["experiment_id"],
            "model": MODEL,
            "model_arm": MODEL_ARM,
            "stage_count": 2,
            "requests_sha256": sha256_file(requests_path),
            "cost": {key: value for key, value in gate.items() if key != "deliveries"},
            "deliveries": gate["deliveries"],
        },
    )
    uploader = upload_file or (
        lambda provider_client, path: _upload(
            provider_client, path, sleep=sleep
        )
    )
    uploads = {
        row["video_id"]: dict(
            uploader(client, _delivery_path(historical_root, row["video_id"]))
        )
        for row in payload["rows"]
    }

    stage1_raw_root = output_root / "stage1/raw"
    stage1_records: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        row = indexed[variant["video_id"]]
        request = build_stage1_request(row=row, upload=uploads[row["video_id"]])
        raw_record = _provider_call(client, request)
        raw_path = stage1_raw_root / f"{index:02d}.json"
        _write_json(raw_path, raw_record)
        stage1_records.append(
            {
                "video_id": row["video_id"],
                "raw_sha256": sha256_file(raw_path),
                "status": raw_record["status"],
            }
        )
    _write_json(
        output_root / "stage1/raw-freeze.json",
        {"schema_version": 1, "status": "raw_frozen", "rows": stage1_records},
    )

    stage1_predictions: list[dict[str, Any]] = []
    inspection_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        row = indexed[variant["video_id"]]
        duration = float(row["duration_seconds"])
        raw_record = json.loads((stage1_raw_root / f"{index:02d}.json").read_text())
        ranking = None
        plan = fallback_plan(duration)
        status = "failed"
        usage = None
        error_type = raw_record.get("error_type")
        if raw_record.get("status") == "received":
            try:
                _, answer, usage = _response_text(raw_record["response"])
                ranking, plan, status, error_type = _parse_stage1_with_fallback(
                    answer, duration_seconds=duration
                )
            except Exception as error:  # noqa: BLE001 - preserve parser failure
                error_type = smoke._safe_error(error)
        stage1_predictions.append(
            {
                "video_id": row["video_id"],
                "variant_kind": "canonical",
                "status": status,
                "ranking": ranking,
                "inspection_source": plan.source,
                "usage": usage,
                "error_type": error_type,
            }
        )
        inspection_rows.append(
            {
                "video_id": row["video_id"],
                "stage1_top_two_unordered": sorted(ranking[:2]) if ranking else None,
                "inspection": plan_dict(plan),
            }
        )
    stage1_freeze = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model": MODEL,
        "raw_freeze_sha256": sha256_file(output_root / "stage1/raw-freeze.json"),
        "rows": stage1_predictions,
    }
    _write_json(output_root / "stage1/predictions-freeze.json", stage1_freeze)
    _write_json(
        output_root / "inspection-requests-freeze.json",
        {
            "schema_version": 1,
            "status": "inspection_requests_frozen",
            "stage1_predictions_sha256": sha256_file(
                output_root / "stage1/predictions-freeze.json"
            ),
            "rows": inspection_rows,
        },
    )

    stage2_raw_root = output_root / "stage2/raw"
    stage2_records: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        row = indexed[variant["video_id"]]
        plan = InspectionPlan(**inspection_rows[index]["inspection"])
        request = build_stage2_request(
            row=row,
            upload=uploads[row["video_id"]],
            plan=plan,
            stage1_ranking=stage1_predictions[index]["ranking"],
        )
        raw_record = _provider_call(client, request)
        raw_path = stage2_raw_root / f"{index:02d}.json"
        _write_json(raw_path, raw_record)
        stage2_records.append(
            {
                "video_id": row["video_id"],
                "raw_sha256": sha256_file(raw_path),
                "status": raw_record["status"],
            }
        )
    _write_json(
        output_root / "stage2/raw-freeze.json",
        {"schema_version": 1, "status": "raw_frozen", "rows": stage2_records},
    )

    final_predictions: list[dict[str, Any]] = []
    stage2_usage_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        row = indexed[variant["video_id"]]
        raw_record = json.loads((stage2_raw_root / f"{index:02d}.json").read_text())
        stage2_ranking = None
        evidence = None
        stage2_status = "failed"
        usage = None
        error_type = raw_record.get("error_type")
        if raw_record.get("status") == "received":
            try:
                _, answer, usage = _response_text(raw_record["response"])
                parsed = parse_stage2(
                    answer, duration_seconds=float(row["duration_seconds"])
                )
                stage2_ranking = parsed["ranking"]
                evidence = parsed["evidence"]
                stage2_status = "passed"
                error_type = None
            except Exception as error:  # noqa: BLE001 - preserve parser failure
                error_type = smoke._safe_error(error)
        final_ranking, final_source = choose_final_ranking(
            stage1_predictions[index]["ranking"], stage2_ranking
        )
        stage2_usage_rows.append({"usage": usage})
        final_predictions.append(
            {
                "video_id": row["video_id"],
                "variant_kind": "canonical",
                "status": "passed" if final_ranking is not None else "failed",
                "ranking": final_ranking,
                "stage1_ranking": stage1_predictions[index]["ranking"],
                "stage2_ranking": stage2_ranking,
                "stage1_status": stage1_predictions[index]["status"],
                "stage2_status": stage2_status,
                "final_source": final_source,
                "inspection_source": inspection_rows[index]["inspection"]["source"],
                "evidence": evidence,
                "usage": usage,
                "error_type": error_type,
            }
        )
    _write_json(
        output_root / "final-predictions-freeze.json",
        {
            "schema_version": 1,
            "status": "predictions_frozen",
            "model": MODEL,
            "stage1_predictions_sha256": sha256_file(
                output_root / "stage1/predictions-freeze.json"
            ),
            "inspection_requests_sha256": sha256_file(
                output_root / "inspection-requests-freeze.json"
            ),
            "stage2_raw_freeze_sha256": sha256_file(
                output_root / "stage2/raw-freeze.json"
            ),
            "rows": final_predictions,
        },
    )
    usage = _usage_totals([*stage1_predictions, *stage2_usage_rows])
    summary = {
        "status": "completed",
        "row_count": len(variants),
        "request_count": len(variants) * 2,
        "stage1_valid_count": sum(
            row["status"] == "passed" for row in stage1_predictions
        ),
        "stage2_valid_count": sum(
            row["stage2_status"] == "passed" for row in final_predictions
        ),
        "final_valid_count": sum(
            row["status"] == "passed" for row in final_predictions
        ),
        "inspection_fallback_count": sum(
            row["inspection"]["source"] == "fallback" for row in inspection_rows
        ),
        **usage,
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
    parser = argparse.ArgumentParser(prog="run_vqa_mc_gemini_adaptive_agent_poc20")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        if not args.canonical_only:
            raise ValueError("adaptive experiment requires --canonical-only")
        if args.preflight:
            result = preflight(
                requests_path=args.requests,
                historical_root=args.historical_root,
            )
            result = {key: value for key, value in result.items() if key != "deliveries"}
        else:
            result = run(
                requests_path=args.requests,
                historical_root=args.historical_root,
                output_root=args.output,
                client=genai.Client(api_key=_api_key()),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:  # noqa: BLE001 - do not leak paths or provider responses
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
