#!/usr/bin/env python3
"""Run the Gemini four-option native-AV evidence agent."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shlex
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026 import vqa_gemini37_flash_smoke as smoke
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
    append_round_stage,
    choose_final_ranking,
    critic_prompt,
    initialize_notebook,
    initializer_prompt,
    judge_prompt,
    live_option_ids,
    next_search_state,
    notebook_observation_ids,
    observer_prompt,
    parse_critic,
    parse_initializer,
    parse_judge,
    parse_observer,
    parse_planner,
    parse_update,
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
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")
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


def build_experiment_preflight(
    *,
    canary_requests_path: Path,
    full_requests_path: Path,
    protected_control_path: Path,
    historical_root: Path,
    capabilities: ModelCapabilities,
    max_cost_usd: float = MAX_COST_USD,
    delivery_probe: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    canary = load_requests(
        canary_requests_path, expected_rows=5, expected_variants=5
    )
    full = load_requests(full_requests_path, expected_rows=20, expected_variants=20)
    full_rows = {str(row["video_id"]): row for row in full["rows"]}
    canary_ids = [str(row["video_id"]) for row in canary["rows"]]
    if len(set(canary_ids)) != 5 or any(video_id not in full_rows for video_id in canary_ids):
        raise ValueError("canary row identity is not a subset of the full panel")
    for row in canary["rows"]:
        if canonical_json(row) != canonical_json(full_rows[str(row["video_id"])]):
            raise ValueError("canary request row differs from the full panel")
    full_ids = [str(row["video_id"]) for row in full["rows"]]
    load_protected_control(protected_control_path, request_ids=full_ids)
    estimate = estimate_formal_cost(full, capabilities=capabilities)
    allowed = estimate["formal_cost_usd"] <= max_cost_usd
    deliveries: list[dict[str, Any]] = []
    if allowed:
        for video_id in full_ids:
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
    return {
        "schema_version": 1,
        "status": "preflight_passed" if allowed else "preflight_blocked",
        "paid_run_allowed": allowed,
        "block_reason": None if allowed else "formal_worst_case_exceeds_cost_cap",
        "max_cost_usd": max_cost_usd,
        "cost": estimate,
        "identity": {
            "model": MODEL,
            "model_arm": MODEL_ARM,
            "model_version": capabilities.model_version,
            "input_token_limit": capabilities.input_token_limit,
            "output_token_limit": capabilities.output_token_limit,
            "supported_actions": list(capabilities.supported_actions),
            "prompt_contract_sha256": _prompt_contract_sha256(),
            "canary_requests_sha256": sha256_file(canary_requests_path),
            "full_requests_sha256": sha256_file(full_requests_path),
            "protected_control_sha256": sha256_file(protected_control_path),
            "pricing": {
                "input_usd_per_million": INPUT_USD_PER_MILLION,
                "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            },
        },
        "canary_video_ids": canary_ids,
        "deliveries": deliveries,
    }


def _write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _prompt_contract_sha256() -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schemas": [
                    INITIALIZER_SCHEMA,
                    PLANNER_SCHEMA,
                    OBSERVER_SCHEMA,
                    UPDATER_SCHEMA,
                    CRITIC_SCHEMA,
                    JUDGE_SCHEMA,
                ],
                "model_arm": MODEL_ARM,
            }
        )
    ).hexdigest()


def _finish_reason(response: Mapping[str, Any]) -> str:
    value = response.get("finish_reason", response.get("finishReason"))
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        value = candidates[0].get(
            "finish_reason", candidates[0].get("finishReason")
        )
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return "FINISH_REASON_UNSPECIFIED"


def _response_usage(response: Mapping[str, Any]) -> dict[str, int] | None:
    value = smoke._usage(response)
    if not isinstance(value, Mapping):
        return None
    keys = ("prompt_tokens", "output_tokens", "total_tokens")
    if not all(isinstance(value.get(key), int) for key in keys):
        return None
    return {key: int(value[key]) for key in keys}


def _usage_cost(usage: Mapping[str, Any] | None) -> float | None:
    if usage is None:
        return None
    prompt = usage.get("prompt_tokens")
    output = usage.get("output_tokens")
    if (
        isinstance(prompt, bool)
        or not isinstance(prompt, int)
        or isinstance(output, bool)
        or not isinstance(output, int)
    ):
        return None
    return (
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + output / 1_000_000 * OUTPUT_USD_PER_MILLION
    )


def _provider_call(client: Any, request: types.InlinedRequest) -> dict[str, Any]:
    try:
        raw = client.models.generate_content(
            model=request.model,
            contents=request.contents,
            config=request.config,
        )
    except Exception as error:  # noqa: BLE001 - provider errors are frozen safely
        return {
            "status": "error",
            "finish_reason": "PROVIDER_ERROR",
            "response": {},
            "usage": None,
            "error_type": smoke._safe_error(error),
        }
    safe = smoke._safe_response_object(raw)
    return {
        "status": "received",
        "finish_reason": _finish_reason(safe),
        "response": safe,
        "usage": _response_usage(safe),
        "error_type": None,
    }


def _observed_cost(usages: Sequence[Mapping[str, Any] | None]) -> float | None:
    costs = [_usage_cost(usage) for usage in usages]
    if any(value is None for value in costs):
        return None
    return round(sum(float(value) for value in costs), 8)


def call_structured_with_recovery(
    *,
    client: Any,
    request: types.InlinedRequest,
    parser: Callable[[object], dict[str, Any]],
    stage_dir: Path,
    role: str,
    capabilities: ModelCapabilities,
    budget: dict[str, Any],
) -> dict[str, Any]:
    attempt_paths: list[str] = []
    last_error: Any = None
    usages: list[dict[str, int] | None] = []
    for attempt in (1, 2):
        maximum_output_cost = (
            capabilities.output_token_limit
            / 1_000_000
            * OUTPUT_USD_PER_MILLION
        )
        if float(budget["observed_cost_usd"]) + maximum_output_cost > float(
            budget["max_cost_usd"]
        ):
            return {
                "status": "budget_blocked",
                "record": None,
                "attempt_count": attempt - 1,
                "recovered": False,
                "usage": None,
                "attempt_paths": attempt_paths,
                "error_type": "maximum_call_reserve_exceeds_remaining_budget",
            }
        raw = _provider_call(client, request)
        budget["provider_request_count"] += 1
        budget["provider_call_counts_by_role"][role] += 1
        usage = raw.get("usage")
        usages.append(usage if isinstance(usage, Mapping) else None)
        cost = _usage_cost(usage if isinstance(usage, Mapping) else None)
        if cost is not None:
            budget["observed_cost_usd"] += cost
        path = stage_dir / f"attempt-{attempt}.json"
        _write_json(path, raw)
        attempt_paths.append(path.as_posix())
        if raw.get("status") != "received":
            last_error = raw.get("error_type") or "provider_error"
            continue
        finish = raw.get("finish_reason")
        response = raw.get("response")
        text = smoke._response_text(response) if isinstance(response, Mapping) else None
        if finish == "MAX_TOKENS":
            last_error = "max_tokens"
            continue
        if finish != "STOP":
            last_error = f"finish_reason_{str(finish).casefold()}"
            continue
        if not isinstance(text, str) or not text.strip():
            last_error = "empty_response"
            continue
        try:
            record = parser(text)
        except Exception as error:  # noqa: BLE001 - schema/parser failure is recoverable
            last_error = smoke._safe_error(error)
            continue
        return {
            "status": "passed",
            "record": record,
            "attempt_count": attempt,
            "recovered": attempt == 2,
            "usage": {
                "attempt_usages": usages,
                "observed_cost_usd": _observed_cost(usages),
            },
            "attempt_paths": attempt_paths,
            "error_type": None,
        }
    return {
        "status": "failed",
        "record": None,
        "attempt_count": 2,
        "recovered": False,
        "usage": {
            "attempt_usages": usages,
            "observed_cost_usd": _observed_cost(usages),
        },
        "attempt_paths": attempt_paths,
        "error_type": last_error,
    }


def _fallback_initializer(
    row: Mapping[str, Any], protected_ranking: Sequence[str]
) -> dict[str, Any]:
    return {
        "protected_ranking": list(protected_ranking),
        "provisional_ranking": list(protected_ranking),
        "ledger": [
            {
                "option_id": str(option["option_id"]),
                "observable_claim": "The observable claim was not established because initialization failed.",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "untested_claims": ["All answer-relevant visual and audio evidence remains untested."],
                "status": "viable",
                "interpretation": "No initializer interpretation is available.",
            }
            for option in row["options"]
        ],
        "literal_observations": [],
        "unresolved_distinctions": ["The complete item remains unresolved."],
    }


def _fallback_planner(
    *, row: Mapping[str, Any], protected_ranking: Sequence[str], round_index: int
) -> dict[str, Any]:
    duration = float(row["duration_seconds"])
    return {
        "round_index": round_index,
        "provisional_ranking": list(protected_ranking),
        "unresolved_distinction": "The complete literal sequence remains unresolved.",
        "action": MediaAction(
            kind="rewatch_full",
            start_seconds=0.0,
            end_seconds=duration,
            fps=8.0,
            resolution="high",
            neutral_request=(
                "Produce a gapless literal timeline of visible actions, exact visible text, and audible speech."
            ),
            tested_option_ids=tuple(OPTION_IDS),
            expected_outcomes=tuple(
                {
                    "option_id": option_id,
                    "literal_outcome": "Record literal evidence relevant to this unresolved possibility.",
                }
                for option_id in OPTION_IDS
            ),
        ),
        "why_discriminating": "A complete high-detail timeline is the neutral mechanical fallback.",
    }


def _current_ledger(notebook: Mapping[str, Any]) -> list[dict[str, Any]]:
    rounds = notebook.get("rounds")
    if isinstance(rounds, list):
        for entry in reversed(rounds):
            if isinstance(entry, Mapping) and isinstance(entry.get("updater"), Mapping):
                ledger = entry["updater"].get("ledger")
                if isinstance(ledger, list):
                    return copy.deepcopy(ledger)
    initializer = notebook.get("initializer")
    if isinstance(initializer, Mapping) and isinstance(initializer.get("ledger"), list):
        return copy.deepcopy(initializer["ledger"])
    raise ValueError("notebook ledger is unavailable")


def _fallback_update(
    notebook: Mapping[str, Any], protected_ranking: Sequence[str]
) -> dict[str, Any]:
    return {
        "provisional_ranking": list(protected_ranking),
        "ledger": _current_ledger(notebook),
        "mapped_observation_ids": [],
        "unresolved_distinctions": ["The newest observation could not be mapped."],
    }


def _fallback_critic(notebook: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": "continue",
        "tested_viable_option_ids": sorted(live_option_ids(notebook)),
        "leader_evidence_ids": [],
        "unexplained_counterevidence_ids": [],
        "next_unresolved_distinction": "The stop review was invalid, so investigation must continue.",
        "direct_replacement_supported": False,
        "reason": "Invalid stop review cannot certify evidence sufficiency.",
    }


def _prompt_prohibited_values(
    row: Mapping[str, Any], protected_ranking: Sequence[str]
) -> list[str]:
    options = row.get("options")
    option_texts = [
        str(option["text"])
        for option in options
        if isinstance(option, Mapping) and isinstance(option.get("text"), str)
    ] if isinstance(options, list) else []
    return [
        str(row.get("question", "")),
        *option_texts,
        *OPTION_IDS,
        "".join(protected_ranking),
        " > ".join(protected_ranking),
    ]


def _relative_paths(root: Path, paths: Sequence[str]) -> list[str]:
    return [Path(path).relative_to(root).as_posix() for path in paths]


def run_row(
    *,
    index: int,
    row: Mapping[str, Any],
    protected_ranking: Sequence[str],
    upload: Mapping[str, Any],
    output_root: Path,
    capabilities: ModelCapabilities,
    client: Any,
    budget: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_root = output_root / "rows" / f"{index:02d}"
    operational_invalid = False
    recovery_count = 0

    initializer_result = call_structured_with_recovery(
        client=client,
        request=build_initializer_request(
            row=row,
            upload=upload,
            protected_ranking=protected_ranking,
            capabilities=capabilities,
        ),
        parser=lambda raw: parse_initializer(
            raw,
            duration_seconds=float(row["duration_seconds"]),
            protected_ranking=protected_ranking,
        ),
        stage_dir=row_root / "initializer",
        role="initializer",
        capabilities=capabilities,
        budget=budget,
    )
    recovery_count += int(initializer_result["recovered"])
    initializer = initializer_result.get("record")
    initializer_status = "passed"
    if not isinstance(initializer, Mapping):
        initializer = _fallback_initializer(row, protected_ranking)
        initializer_status = "fallback_unknown_ledger"
        operational_invalid = True
    notebook = initialize_notebook(
        str(row["video_id"]), protected_ranking, initializer
    )
    observer_count = 0
    stop_status = "forced_stop"
    observer_audit_paths: list[str] = []
    stage_results: list[dict[str, Any]] = [initializer_result]

    for round_index in range(1, MAX_OBSERVATIONS + 1):
        round_root = row_root / f"round-{round_index}"
        planner_result = call_structured_with_recovery(
            client=client,
            request=build_planner_request(
                row=row,
                notebook=notebook,
                round_index=round_index,
                capabilities=capabilities,
            ),
            parser=lambda raw, current=round_index: parse_planner(
                raw,
                duration_seconds=float(row["duration_seconds"]),
                round_index=current,
            ),
            stage_dir=round_root / "planner",
            role="planner",
            capabilities=capabilities,
            budget=budget,
        )
        recovery_count += int(planner_result["recovered"])
        planner = planner_result.get("record")
        if not isinstance(planner, Mapping):
            planner = _fallback_planner(
                row=row,
                protected_ranking=protected_ranking,
                round_index=round_index,
            )
            operational_invalid = True
        notebook = append_round_stage(
            notebook,
            round_index=round_index,
            stage="planner",
            record=planner,
        )
        stage_results.append(planner_result)

        action = planner.get("action")
        if not isinstance(action, MediaAction):
            action = MediaAction(**dict(action)) if isinstance(action, Mapping) else None
        if not isinstance(action, MediaAction):
            raise ValueError("planner action is unavailable")
        observer_request = build_observer_request(
            upload=upload, action=action, capabilities=capabilities
        )
        audit = audit_observer_request(
            observer_request,
            prohibited_values=_prompt_prohibited_values(row, protected_ranking),
        )
        audit_path = round_root / "observer/request-audit.json"
        _write_json(audit_path, audit)
        observer_audit_paths.append(audit_path.relative_to(output_root).as_posix())
        if not audit["passed"]:
            raise ValueError("observer request failed information-barrier audit")
        observer_result = call_structured_with_recovery(
            client=client,
            request=observer_request,
            parser=lambda raw, current_action=action, current_round=round_index: parse_observer(
                raw,
                action=current_action,
                duration_seconds=float(row["duration_seconds"]),
                view_id=f"view-{current_round}",
            ),
            stage_dir=round_root / "observer",
            role="observer",
            capabilities=capabilities,
            budget=budget,
        )
        recovery_count += int(observer_result["recovered"])
        observer = observer_result.get("record")
        if not isinstance(observer, Mapping):
            operational_invalid = True
            stage_results.append(observer_result)
            continue
        observer_count += 1
        observer_record = dict(observer)
        observer_record["provenance"] = {
            "view_id": observer_record["view_id"],
            "start_seconds": action.start_seconds,
            "end_seconds": action.end_seconds,
            "fps": action.fps,
            "resolution": action.resolution,
            "request_audit_path": audit_path.relative_to(output_root).as_posix(),
            "request_audit_sha256": sha256_file(audit_path),
        }
        notebook = append_round_stage(
            notebook,
            round_index=round_index,
            stage="observer",
            record=observer_record,
        )
        stage_results.append(observer_result)

        updater_result = call_structured_with_recovery(
            client=client,
            request=build_updater_request(
                row=row, notebook=notebook, capabilities=capabilities
            ),
            parser=lambda raw: parse_update(raw, notebook=notebook),
            stage_dir=round_root / "updater",
            role="updater",
            capabilities=capabilities,
            budget=budget,
        )
        recovery_count += int(updater_result["recovered"])
        updater = updater_result.get("record")
        if not isinstance(updater, Mapping):
            updater = _fallback_update(notebook, protected_ranking)
            operational_invalid = True
        notebook = append_round_stage(
            notebook,
            round_index=round_index,
            stage="updater",
            record=updater,
        )
        stage_results.append(updater_result)

        critic_result = call_structured_with_recovery(
            client=client,
            request=build_critic_request(
                row=row, notebook=notebook, capabilities=capabilities
            ),
            parser=lambda raw: parse_critic(
                raw, notebook=notebook, observation_count=observer_count
            ),
            stage_dir=round_root / "critic",
            role="critic",
            capabilities=capabilities,
            budget=budget,
        )
        recovery_count += int(critic_result["recovered"])
        critic = critic_result.get("record")
        if not isinstance(critic, Mapping):
            critic = _fallback_critic(notebook)
            operational_invalid = True
        notebook = append_round_stage(
            notebook,
            round_index=round_index,
            stage="critic",
            record=critic,
        )
        stage_results.append(critic_result)
        stop_status = next_search_state(
            decision=str(critic["decision"]), observation_count=observer_count
        )
        if stop_status != "continue":
            break

    judge_result = call_structured_with_recovery(
        client=client,
        request=build_judge_request(
            row=row,
            upload=upload,
            notebook=notebook,
            capabilities=capabilities,
        ),
        parser=lambda raw: parse_judge(
            raw, notebook=notebook, protected_ranking=protected_ranking
        ),
        stage_dir=row_root / "judge",
        role="judge",
        capabilities=capabilities,
        budget=budget,
    )
    recovery_count += int(judge_result["recovered"])
    judge = judge_result.get("record")
    if isinstance(judge, Mapping) and judge.get("stop_status") != stop_status:
        judge = None
        judge_result["status"] = "failed"
        judge_result["error_type"] = "judge_stop_status_mismatch"
    if not isinstance(judge, Mapping):
        operational_invalid = True
    stage_results.append(judge_result)
    ranking, final_source = choose_final_ranking(judge, protected_ranking)
    final_notebook = copy.deepcopy(notebook)
    final_notebook["judge"] = {
        "status": "passed" if judge is not None else "failed",
        "record": copy.deepcopy(judge),
        "error_type": judge_result.get("error_type"),
    }
    notebook_path = row_root / "final-notebook.json"
    _write_json(notebook_path, final_notebook)
    return (
        {
            "video_id": str(row["video_id"]),
            "variant_kind": "canonical",
            "status": "operationally_invalid" if operational_invalid else "passed",
            "ranking": ranking,
            "final_source": final_source,
            "initializer_status": initializer_status,
            "judge_status": "passed" if judge is not None else "failed",
            "judge": judge,
            "stop_status": stop_status,
            "observer_action_count": observer_count,
            "recovery_call_count": recovery_count,
            "notebook_path": notebook_path.relative_to(output_root).as_posix(),
            "notebook_sha256": sha256_file(notebook_path),
            "observer_audit_paths": observer_audit_paths,
            "known_observation_count": len(notebook_observation_ids(final_notebook)),
        },
        {
            "operational_invalid": operational_invalid,
            "recovery_call_count": recovery_count,
            "stage_results": stage_results,
        },
    )


def run_phase(
    *,
    requests_path: Path,
    protected_control_path: Path,
    historical_root: Path,
    output_root: Path,
    capabilities: ModelCapabilities,
    client: Any,
    upload_file: Callable[[Any, Path], Mapping[str, Any]],
    max_cost_usd: float = MAX_COST_USD,
    expected_rows: int | None = 20,
    expected_variants: int | None = 20,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("output root already exists")
    gate = preflight(
        requests_path=requests_path,
        protected_control_path=protected_control_path,
        historical_root=historical_root,
        capabilities=capabilities,
        max_cost_usd=max_cost_usd,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    output_root.mkdir(parents=True)
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    request_ids = [str(row["video_id"]) for row in payload["rows"]]
    controls = load_protected_control(
        protected_control_path, request_ids=request_ids
    )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "model": MODEL,
        "model_arm": MODEL_ARM,
        "model_capabilities": asdict(capabilities),
        "requests_sha256": sha256_file(requests_path),
        "request_row_sha256s": {
            str(row["video_id"]): hashlib.sha256(canonical_json(row)).hexdigest()
            for row in payload["rows"]
        },
        "protected_control_sha256": sha256_file(protected_control_path),
        "prompt_contract_sha256": _prompt_contract_sha256(),
        "cost": gate["cost"],
    }
    _write_json(output_root / "run-manifest.json", manifest)
    _write_json(output_root / "preflight.json", gate)
    budget: dict[str, Any] = {
        "max_cost_usd": max_cost_usd,
        "observed_cost_usd": 0.0,
        "provider_request_count": 0,
        "provider_call_counts_by_role": Counter(),
    }
    final_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(payload["rows"]):
        video_id = str(row["video_id"])
        upload = upload_file(client, _delivery_path(historical_root, video_id))
        final, diagnostic = run_row(
            index=index,
            row=row,
            protected_ranking=controls[video_id],
            upload=upload,
            output_root=output_root,
            capabilities=capabilities,
            client=client,
            budget=budget,
        )
        final_rows.append(final)
        diagnostics.append(diagnostic)
    notebook_index_path = output_root / "notebooks-freeze.json"
    _write_json(
        notebook_index_path,
        {
            "schema_version": 1,
            "status": "notebooks_frozen",
            "rows": [
                {
                    "video_id": row["video_id"],
                    "path": row["notebook_path"],
                    "sha256": row["notebook_sha256"],
                }
                for row in final_rows
            ],
        },
    )
    manifest_path = output_root / "run-manifest.json"
    final_path = output_root / "final-predictions-freeze.json"
    _write_json(
        final_path,
        {
            "schema_version": 1,
            "status": "predictions_frozen",
            "model": MODEL,
            "source_paths": {
                "manifest": "run-manifest.json",
                "notebooks": "notebooks-freeze.json",
            },
            "source_sha256s": {
                "manifest": sha256_file(manifest_path),
                "notebooks": sha256_file(notebook_index_path),
            },
            "rows": final_rows,
        },
    )
    recovery_count = sum(int(row["recovery_call_count"]) for row in final_rows)
    operational_invalid_count = sum(
        bool(item["operational_invalid"]) for item in diagnostics
    )
    summary = {
        "schema_version": 1,
        "status": "completed",
        "row_count": len(final_rows),
        "final_valid_count": sum(row["status"] == "passed" for row in final_rows),
        "operational_invalid_count": operational_invalid_count,
        "observer_call_count": sum(
            int(row["observer_action_count"]) for row in final_rows
        ),
        "recovery_call_count": recovery_count,
        "provider_request_count": budget["provider_request_count"],
        "provider_call_counts_by_role": dict(
            sorted(budget["provider_call_counts_by_role"].items())
        ),
        "observed_cost_usd": round(float(budget["observed_cost_usd"]), 8),
        "formal_cost_usd": gate["cost"]["formal_cost_usd"],
        "max_cost_usd": max_cost_usd,
        "final_predictions_sha256": sha256_file(final_path),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def verify_output(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve(strict=True)
    final_path = root / "final-predictions-freeze.json"
    summary_path = root / "summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = final.get("rows") if isinstance(final, Mapping) else None
    if not isinstance(rows, list) or len(rows) != summary.get("row_count"):
        raise ValueError("final row count does not match summary")
    paths = final.get("source_paths")
    hashes = final.get("source_sha256s")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("final source declarations are missing")
    for key, relative in paths.items():
        candidate = (root / str(relative)).resolve(strict=True)
        candidate.relative_to(root)
        if sha256_file(candidate) != hashes.get(key):
            raise ValueError("final source hash mismatch")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("final row is invalid")
        notebook_path = (root / str(row.get("notebook_path"))).resolve(strict=True)
        notebook_path.relative_to(root)
        if sha256_file(notebook_path) != row.get("notebook_sha256"):
            raise ValueError("final notebook hash mismatch")
        for relative in row.get("observer_audit_paths", []):
            audit_path = (root / str(relative)).resolve(strict=True)
            audit_path.relative_to(root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit != {"passed": True, "matched_value_hashes": []}:
                raise ValueError("observer audit is invalid")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        if isinstance(row.get("judge"), Mapping):
            cited = set(row["judge"].get("decisive_evidence_ids", []))
            if not cited <= set(notebook_observation_ids(notebook)):
                raise ValueError("judge citation is invalid")
    if summary.get("final_predictions_sha256") != sha256_file(final_path):
        raise ValueError("summary final hash mismatch")
    return {
        "status": "verified",
        "row_count": len(rows),
        "final_predictions_sha256": sha256_file(final_path),
        "summary_sha256": sha256_file(summary_path),
    }


def verify_canary_reuse(
    *,
    canary_root: Path,
    full_requests_path: Path,
    canary_gate_path: Path,
) -> dict[str, Any]:
    if canary_root.is_symlink():
        raise ValueError("canary root must not be a symlink")
    root = canary_root.resolve(strict=True)
    manifest_path = root / "run-manifest.json"
    final_path = root / "final-predictions-freeze.json"
    gate = json.loads(canary_gate_path.read_text(encoding="utf-8"))
    if not isinstance(gate, Mapping) or gate.get("continue_to_full") is not True:
        raise ValueError("canary gate does not authorize full reuse")
    if (
        gate.get("canary_manifest_sha256") != sha256_file(manifest_path)
        or gate.get("canary_final_sha256") != sha256_file(final_path)
    ):
        raise ValueError("canary protocol identity hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("model") != MODEL
        or manifest.get("model_arm") != MODEL_ARM
        or manifest.get("prompt_contract_sha256") != _prompt_contract_sha256()
    ):
        raise ValueError("canary protocol identity does not match current runner")
    declared_rows = manifest.get("request_row_sha256s")
    if not isinstance(declared_rows, Mapping) or not declared_rows:
        raise ValueError("canary protocol identity lacks row hashes")
    full = load_requests(
        full_requests_path, expected_rows=None, expected_variants=None
    )
    full_rows = {str(row["video_id"]): row for row in full["rows"]}
    for video_id, digest in declared_rows.items():
        row = full_rows.get(str(video_id))
        if (
            row is None
            or not isinstance(digest, str)
            or hashlib.sha256(canonical_json(row)).hexdigest() != digest
        ):
            raise ValueError("canary request row identity does not match full panel")
    verify_output(root)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final_ids = [str(row.get("video_id")) for row in final.get("rows", [])]
    if final_ids != list(declared_rows):
        raise ValueError("canary final row identity does not match manifest")
    return {
        "status": "canary_reuse_verified",
        "reusable_row_count": len(final_ids),
        "reusable_video_ids": final_ids,
        "manifest_sha256": sha256_file(manifest_path),
        "final_predictions_sha256": sha256_file(final_path),
    }


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


def _upload(client: Any, delivery: Path) -> dict[str, Any]:
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
        time.sleep(1.0)
        current = smoke._provider_file_metadata(
            client.files.get(name=current["name"]), expected_mime=MIME_TYPE
        )
    raise AssertionError("unreachable")


def _require_paid_preflight(
    path: Path,
    *,
    requests_path: Path,
    protected_control_path: Path,
    capabilities: ModelCapabilities,
) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    identity = artifact.get("identity") if isinstance(artifact, Mapping) else None
    if (
        not isinstance(identity, Mapping)
        or artifact.get("paid_run_allowed") is not True
        or identity.get("model") != MODEL
        or identity.get("model_arm") != MODEL_ARM
        or identity.get("output_token_limit") != capabilities.output_token_limit
        or identity.get("prompt_contract_sha256") != _prompt_contract_sha256()
        or identity.get("protected_control_sha256")
        != sha256_file(protected_control_path)
        or sha256_file(requests_path)
        not in {
            identity.get("canary_requests_sha256"),
            identity.get("full_requests_sha256"),
        }
    ):
        raise ValueError("paid preflight does not authorize this run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_vqa_mc_gemini_four_option_agent_poc20"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-canary", action="store_true")
    mode.add_argument("--verify-output", type=Path)
    parser.add_argument("--canary-requests", type=Path)
    parser.add_argument("--full-requests", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--protected-control", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--preflight-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_output is not None:
            result = verify_output(args.verify_output)
        else:
            required = {
                "protected_control": args.protected_control,
                "historical_root": args.historical_root,
                "output": args.output,
            }
            if any(value is None for value in required.values()):
                raise ValueError("required path is missing")
            client = genai.Client(api_key=_api_key())
            capabilities = get_model_capabilities(client)
            if args.preflight:
                if args.canary_requests is None or args.full_requests is None:
                    raise ValueError("preflight requires canary and full requests")
                result = build_experiment_preflight(
                    canary_requests_path=args.canary_requests,
                    full_requests_path=args.full_requests,
                    protected_control_path=args.protected_control,
                    historical_root=args.historical_root,
                    capabilities=capabilities,
                )
                _write_json(args.output, result)
            else:
                if args.requests is None or args.preflight_artifact is None:
                    raise ValueError("canary run requires requests and preflight artifact")
                _require_paid_preflight(
                    args.preflight_artifact,
                    requests_path=args.requests,
                    protected_control_path=args.protected_control,
                    capabilities=capabilities,
                )
                result = run_phase(
                    requests_path=args.requests,
                    protected_control_path=args.protected_control,
                    historical_root=args.historical_root,
                    output_root=args.output,
                    capabilities=capabilities,
                    client=client,
                    upload_file=_upload,
                    expected_rows=5,
                    expected_variants=5,
                )
        safe = {
            key: value
            for key, value in result.items()
            if key not in {"deliveries", "canary_video_ids"}
        }
        print(json.dumps(safe, sort_keys=True))
        return 0
    except Exception:  # noqa: BLE001 - CLI must not leak credentials or private paths
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
