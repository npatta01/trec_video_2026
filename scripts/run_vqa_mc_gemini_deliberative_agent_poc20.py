#!/usr/bin/env python3
"""Run the bounded single-model deliberative Gemini MC agent."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from google.genai import types

from trec_video_2026.vqa_experiments.vqa_mc_deliberative_agent import (
    JUDGE_SCHEMA,
    OBSERVER_SCHEMA,
    PLANNER_SCHEMA,
    SOLVER_SCHEMA,
    MediaAction,
    judge_prompt,
    observer_prompt,
    planner_prompt,
    solver_prompt,
)
from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    OPTION_IDS,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "single-model-deliberative-native-av-v1"
BASELINE_FPS = 4.0
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
ROLE_OUTPUT_LIMITS = {
    "solver": 1024,
    "planner": 768,
    "observer": 768,
    "judge": 1024,
}
MEDIA_TOKENS_PER_SECOND = {
    (1.0, "low"): 78,
    (4.0, "low"): 312,
    (8.0, "low"): 624,
    (1.0, "high"): 284,
    (4.0, "high"): 1136,
    (8.0, "high"): 2272,
}
MAX_OBSERVER_INPUT_TOKENS_PER_ROW = 40_000
MAX_TEXT_INPUT_TOKENS_PER_ROW = 12_000
MAX_CALLS_PER_ROW = 7
MAX_OBSERVER_ACTIONS = 2
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
    ROOT
    / "data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a1"
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
    if not isinstance(options, list) or len(options) != len(OPTION_IDS):
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
    variants = canonical_variants(payload)
    indexed = {row["video_id"]: row for row in payload["rows"]}
    seconds = sum(
        float(indexed[variant["video_id"]]["duration_seconds"])
        for variant in variants
    )
    row_count = len(variants)
    base_media = math.ceil(
        seconds * 3 * MEDIA_TOKENS_PER_SECOND[(BASELINE_FPS, "low")]
    )
    observer_media = row_count * MAX_OBSERVER_INPUT_TOKENS_PER_ROW
    text_input = row_count * MAX_TEXT_INPUT_TOKENS_PER_ROW
    output_tokens = row_count * (
        2 * ROLE_OUTPUT_LIMITS["solver"]
        + 2 * ROLE_OUTPUT_LIMITS["planner"]
        + 2 * ROLE_OUTPUT_LIMITS["observer"]
        + ROLE_OUTPUT_LIMITS["judge"]
    )
    input_tokens = base_media + observer_media + text_input
    cost = (
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    return {
        "row_count": row_count,
        "request_count_ceiling": row_count * MAX_CALLS_PER_ROW,
        "upload_count": row_count,
        "video_seconds": round(seconds, 3),
        "max_calls_per_row": MAX_CALLS_PER_ROW,
        "max_observer_actions": MAX_OBSERVER_ACTIONS,
        "final_call_reserved": True,
        "base_media_input_tokens": base_media,
        "observer_input_token_allowance": observer_media,
        "text_input_token_allowance": text_input,
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


def estimate_action_input_tokens(action: MediaAction) -> int:
    seconds = action.end_seconds - action.start_seconds
    return math.ceil(seconds * MEDIA_TOKENS_PER_SECOND[(action.fps, action.resolution)])


def authorize_action(
    action: MediaAction,
    *,
    calls_used: int,
    observer_input_tokens_remaining: int,
) -> dict[str, Any]:
    estimate = estimate_action_input_tokens(action)
    if calls_used + 1 >= MAX_CALLS_PER_ROW:
        return {
            "authorized": False,
            "reason": "final_call_reserve",
            "estimated_input_tokens": estimate,
            "remaining_input_tokens": observer_input_tokens_remaining,
        }
    if estimate > observer_input_tokens_remaining:
        return {
            "authorized": False,
            "reason": "observer_input_budget",
            "estimated_input_tokens": estimate,
            "remaining_input_tokens": observer_input_tokens_remaining,
        }
    return {
        "authorized": True,
        "estimated_input_tokens": estimate,
        "remaining_input_tokens": observer_input_tokens_remaining - estimate,
    }


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


def _config(schema: Mapping[str, Any], *, max_output_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=dict(schema),
        seed=42,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    )


def build_solver_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any], stage: str
) -> types.InlinedRequest:
    if stage not in {"anchor", "blind"}:
        raise ValueError("solver stage is invalid")
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    _video_part(
                        upload=upload, fps=BASELINE_FPS, resolution="low"
                    ),
                    types.Part(text=solver_prompt(row)),
                ],
            )
        ],
        metadata={"arm": MODEL_ARM, "stage": stage, "video_id": row["video_id"]},
        config=_config(SOLVER_SCHEMA, max_output_tokens=ROLE_OUTPUT_LIMITS["solver"]),
    )


def build_planner_request(
    *,
    row: Mapping[str, Any],
    notebook: Mapping[str, Any],
    budget: Mapping[str, Any],
    round_index: int,
) -> types.InlinedRequest:
    if round_index not in {1, 2}:
        raise ValueError("planner round is invalid")
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=planner_prompt(row, notebook, budget))],
            )
        ],
        metadata={
            "arm": MODEL_ARM,
            "stage": f"planner_{round_index}",
            "video_id": row["video_id"],
        },
        config=_config(
            PLANNER_SCHEMA, max_output_tokens=ROLE_OUTPUT_LIMITS["planner"]
        ),
    )


def build_observer_request(
    *, upload: Mapping[str, Any], action: MediaAction
) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    _video_part(
                        upload=upload,
                        fps=action.fps,
                        resolution=action.resolution,
                        start_offset=f"{action.start_seconds:.3f}s",
                        end_offset=f"{action.end_seconds:.3f}s",
                    ),
                    types.Part(text=observer_prompt(action)),
                ],
            )
        ],
        metadata={"arm": MODEL_ARM, "stage": "observer"},
        config=_config(
            OBSERVER_SCHEMA, max_output_tokens=ROLE_OUTPUT_LIMITS["observer"]
        ),
    )


def build_judge_request(
    *,
    row: Mapping[str, Any],
    upload: Mapping[str, Any],
    notebook: Mapping[str, Any],
) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    _video_part(
                        upload=upload, fps=BASELINE_FPS, resolution="low"
                    ),
                    types.Part(text=judge_prompt(row, notebook)),
                ],
            )
        ],
        metadata={"arm": MODEL_ARM, "stage": "judge", "video_id": row["video_id"]},
        config=_config(JUDGE_SCHEMA, max_output_tokens=ROLE_OUTPUT_LIMITS["judge"]),
    )


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
    parser = argparse.ArgumentParser(
        prog="run_vqa_mc_gemini_deliberative_agent_poc20"
    )
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
            raise ValueError("deliberative experiment requires --canonical-only")
        if args.preflight:
            result = preflight(
                requests_path=args.requests,
                historical_root=args.historical_root,
            )
            result = {key: value for key, value in result.items() if key != "deliveries"}
        else:
            _api_key()
            raise NotImplementedError("run orchestration is not implemented")
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
