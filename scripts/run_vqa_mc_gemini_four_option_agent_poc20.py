#!/usr/bin/env python3
"""Run the Gemini four-option native-AV evidence agent."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.genai import types

from trec_video_2026.vqa_experiments.vqa_mc_common import (
    OPTION_IDS,
    canonical_json,
    parse_ranking,
    sha256_file,
)
from trec_video_2026.vqa_experiments.vqa_mc_four_option_agent import (
    CRITIC_SCHEMA,
    INITIALIZER_SCHEMA,
    JUDGE_SCHEMA,
    OBSERVER_SCHEMA,
    PLANNER_SCHEMA,
    UPDATER_SCHEMA,
    MediaAction,
    critic_prompt,
    initializer_prompt,
    judge_prompt,
    observer_prompt,
    planner_prompt,
    updater_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "four-option-evidence-native-av-v1"
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
MAX_OBSERVATIONS = 3
MAX_SEMANTIC_CALLS_PER_ROW = 14
MAX_PROVIDER_CALLS_PER_ROW = 28
MAX_COST_USD = 1.50
BASELINE_FPS = 4.0
MIME_TYPE = "video/webm"
MAX_TEXT_INPUT_TOKENS_PER_ROW = 60_000
MEDIA_TOKENS_PER_SECOND = {
    (1.0, "low"): 78,
    (4.0, "low"): 312,
    (8.0, "low"): 624,
    (1.0, "high"): 284,
    (4.0, "high"): 1136,
    (8.0, "high"): 2272,
}
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


@dataclass(frozen=True)
class ModelCapabilities:
    output_token_limit: int
    input_token_limit: int | None = None
    model_version: str | None = None
    supported_actions: tuple[str, ...] = ()


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
    identifiers: list[str] = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("option is not an object")
        option_id = option.get("option_id")
        text = option.get("text")
        if option_id not in OPTION_IDS or not isinstance(text, str) or not text.strip():
            raise ValueError("option ID or text is invalid")
        identifiers.append(str(option_id))
    if identifiers != list(OPTION_IDS):
        raise ValueError("options must use canonical A-D IDs")


def load_requests(
    path: Path,
    *,
    expected_rows: int | None = 20,
    expected_variants: int | None = 20,
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
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("request row is invalid")
        video_id = row.get("video_id")
        question = row.get("question")
        duration = row.get("duration_seconds")
        if not isinstance(video_id, str) or not video_id or video_id in identifiers:
            raise ValueError("video IDs must be unique nonempty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is invalid")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise ValueError("duration is invalid")
        _validate_options(row)
        identifiers.append(video_id)
    variant_ids: list[str] = []
    variant_video_ids: list[str] = []
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("variant is invalid")
        variant_id = variant.get("variant_id")
        video_id = variant.get("video_id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in variant_ids:
            raise ValueError("variant IDs must be unique nonempty strings")
        if variant.get("variant_kind") != "canonical":
            raise ValueError("only canonical variants are permitted")
        if video_id not in identifiers:
            raise ValueError("variant has unknown video ID")
        if variant.get("display_order") != list(OPTION_IDS):
            raise ValueError("variant display order must be A-D")
        variant_ids.append(variant_id)
        variant_video_ids.append(str(video_id))
    if variant_video_ids != identifiers:
        raise ValueError("canonical variant identity drift")
    return payload


def load_protected_control(
    path: Path, *, request_ids: Sequence[str]
) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _reject_labels(payload)
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("protected control rows are invalid")
    indexed: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("variant_kind") != "canonical":
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id in indexed:
            raise ValueError("protected control identity drift")
        if row.get("status") != "passed":
            raise ValueError("protected control contains an invalid row")
        indexed[video_id] = parse_ranking({"ranking": row.get("ranking")})
    if list(indexed) != list(request_ids):
        raise ValueError("protected control identity drift")
    return indexed


def get_model_capabilities(client: Any) -> ModelCapabilities:
    model = client.models.get(model=MODEL)
    output_limit = getattr(model, "output_token_limit", None)
    if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
        raise ValueError("model output token limit is unavailable")
    input_limit = getattr(model, "input_token_limit", None)
    if isinstance(input_limit, bool) or not isinstance(input_limit, int) or input_limit <= 0:
        input_limit = None
    version = getattr(model, "version", None)
    if not isinstance(version, str) or not version.strip():
        version = None
    actions = getattr(model, "supported_actions", None)
    supported = (
        tuple(str(value) for value in actions)
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes))
        else ()
    )
    return ModelCapabilities(
        output_token_limit=output_limit,
        input_token_limit=input_limit,
        model_version=version,
        supported_actions=supported,
    )


def _config(
    schema: Mapping[str, Any], *, capabilities: ModelCapabilities
) -> types.GenerateContentConfig:
    if capabilities.output_token_limit <= 0:
        raise ValueError("output token limit must be positive")
    return types.GenerateContentConfig(
        max_output_tokens=capabilities.output_token_limit,
        response_mime_type="application/json",
        response_json_schema=dict(schema),
        seed=42,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=types.ThinkingLevel.HIGH,
        ),
    )


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


def _request(
    *,
    stage: str,
    video_id: str,
    parts: list[types.Part],
    schema: Mapping[str, Any],
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return types.InlinedRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        metadata={"arm": MODEL_ARM, "stage": stage, "video_id": video_id},
        config=_config(schema, capabilities=capabilities),
    )


def build_initializer_request(
    *,
    row: Mapping[str, Any],
    upload: Mapping[str, Any],
    protected_ranking: Sequence[str],
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage="initializer",
        video_id=str(row["video_id"]),
        parts=[
            _video_part(upload=upload, fps=BASELINE_FPS, resolution="low"),
            types.Part(text=initializer_prompt(row, protected_ranking)),
        ],
        schema=INITIALIZER_SCHEMA,
        capabilities=capabilities,
    )


def build_planner_request(
    *,
    row: Mapping[str, Any],
    notebook: Mapping[str, Any],
    round_index: int,
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage=f"planner_{round_index}",
        video_id=str(row["video_id"]),
        parts=[types.Part(text=planner_prompt(row, notebook, round_index))],
        schema=PLANNER_SCHEMA,
        capabilities=capabilities,
    )


def build_observer_request(
    *,
    upload: Mapping[str, Any],
    action: MediaAction,
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage="observer",
        video_id="blinded",
        parts=[
            _video_part(
                upload=upload,
                fps=action.fps,
                resolution=action.resolution,
                start_offset=f"{action.start_seconds:.3f}s",
                end_offset=f"{action.end_seconds:.3f}s",
            ),
            types.Part(
                text=observer_prompt(
                    action, original_time_origin_seconds=action.start_seconds
                )
            ),
        ],
        schema=OBSERVER_SCHEMA,
        capabilities=capabilities,
    )


def build_updater_request(
    *,
    row: Mapping[str, Any],
    notebook: Mapping[str, Any],
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage="updater",
        video_id=str(row["video_id"]),
        parts=[types.Part(text=updater_prompt(row, notebook))],
        schema=UPDATER_SCHEMA,
        capabilities=capabilities,
    )


def build_critic_request(
    *,
    row: Mapping[str, Any],
    notebook: Mapping[str, Any],
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage="critic",
        video_id=str(row["video_id"]),
        parts=[types.Part(text=critic_prompt(row, notebook))],
        schema=CRITIC_SCHEMA,
        capabilities=capabilities,
    )


def build_judge_request(
    *,
    row: Mapping[str, Any],
    upload: Mapping[str, Any],
    notebook: Mapping[str, Any],
    capabilities: ModelCapabilities,
) -> types.InlinedRequest:
    return _request(
        stage="judge",
        video_id=str(row["video_id"]),
        parts=[
            _video_part(upload=upload, fps=BASELINE_FPS, resolution="low"),
            types.Part(text=judge_prompt(row, notebook)),
        ],
        schema=JUDGE_SCHEMA,
        capabilities=capabilities,
    )


def _request_text(request: types.InlinedRequest) -> str:
    values: list[str] = []
    for content in request.contents or []:
        for part in content.parts or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                values.append(text)
    return "\n".join(values)


def audit_observer_request(
    request: types.InlinedRequest, *, prohibited_values: Sequence[object]
) -> dict[str, Any]:
    rendered = _request_text(request).casefold()
    schema = json.dumps(OBSERVER_SCHEMA, sort_keys=True).casefold()
    matches: list[str] = []
    for raw in prohibited_values:
        if not isinstance(raw, str) or not raw.strip():
            continue
        value = raw.strip().casefold()
        if len(value) == 1:
            short_patterns = (f'"option_id":"{value}"', f"{value}. ")
            found = any(pattern in rendered or pattern in schema for pattern in short_patterns)
        else:
            found = value in rendered or value in schema
        if found:
            matches.append(hashlib.sha256(value.encode("utf-8")).hexdigest())
    return {
        "passed": not matches,
        "matched_value_hashes": sorted(set(matches)),
    }


def estimate_formal_cost(
    payload: Mapping[str, Any], *, capabilities: ModelCapabilities
) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("request rows are invalid")
    seconds = sum(float(row["duration_seconds"]) for row in rows)
    row_count = len(rows)
    provider_calls = row_count * MAX_PROVIDER_CALLS_PER_ROW
    output_tokens = provider_calls * capabilities.output_token_limit
    complete_media = math.ceil(
        2 * seconds * MEDIA_TOKENS_PER_SECOND[(BASELINE_FPS, "low")]
    )
    observer_media = math.ceil(
        MAX_OBSERVATIONS * seconds * MEDIA_TOKENS_PER_SECOND[(8.0, "high")]
    )
    text_input = row_count * MAX_TEXT_INPUT_TOKENS_PER_ROW
    input_tokens = complete_media + observer_media + text_input
    input_cost = input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
    output_cost = output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    return {
        "row_count": row_count,
        "semantic_call_ceiling": row_count * MAX_SEMANTIC_CALLS_PER_ROW,
        "provider_call_ceiling": provider_calls,
        "output_token_limit": capabilities.output_token_limit,
        "output_token_ceiling": output_tokens,
        "input_token_ceiling": input_tokens,
        "video_seconds": round(seconds, 3),
        "formal_input_cost_usd": round(input_cost, 8),
        "formal_output_cost_usd": round(output_cost, 8),
        "formal_cost_usd": round(input_cost + output_cost, 8),
    }


def _delivery_path(historical_root: Path, video_id: str) -> Path:
    root = historical_root.resolve(strict=True)
    candidate = root / "deliveries" / video_id / "delivery-media" / f"{video_id}.webm"
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("delivery escapes historical root") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("preserved WebM delivery is missing")
    return candidate


def preflight(
    *,
    requests_path: Path,
    protected_control_path: Path,
    historical_root: Path,
    capabilities: ModelCapabilities,
    max_cost_usd: float = MAX_COST_USD,
    expected_rows: int | None = 20,
    expected_variants: int | None = 20,
    delivery_probe: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    request_ids = [str(row["video_id"]) for row in payload["rows"]]
    load_protected_control(protected_control_path, request_ids=request_ids)
    estimate = estimate_formal_cost(payload, capabilities=capabilities)
    if estimate["formal_cost_usd"] > max_cost_usd:
        raise ValueError("formal worst-case cost exceeds authorized cap")
    deliveries: list[dict[str, Any]] = []
    for video_id in request_ids:
        delivery = _delivery_path(historical_root, video_id)
        if delivery_probe is not None:
            delivery_probe(delivery)
        deliveries.append(
            {
                "video_id": video_id,
                "path": str(delivery),
                "sha256": sha256_file(delivery),
                "size_bytes": delivery.stat().st_size,
            }
        )
    identity = {
        "model": MODEL,
        "model_arm": MODEL_ARM,
        "model_version": capabilities.model_version,
        "input_token_limit": capabilities.input_token_limit,
        "output_token_limit": capabilities.output_token_limit,
        "supported_actions": list(capabilities.supported_actions),
        "requests_sha256": sha256_file(requests_path),
        "protected_control_sha256": sha256_file(protected_control_path),
        "pricing_sha256": hashlib.sha256(
            canonical_json(
                {
                    "input_usd_per_million": INPUT_USD_PER_MILLION,
                    "output_usd_per_million": OUTPUT_USD_PER_MILLION,
                }
            )
        ).hexdigest(),
    }
    return {
        "status": "preflight_passed",
        "paid_run_allowed": True,
        "max_cost_usd": max_cost_usd,
        "identity": identity,
        "cost": estimate,
        "deliveries": deliveries,
    }
