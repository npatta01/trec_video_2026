#!/usr/bin/env python3
"""Run the bounded single-model deliberative Gemini MC agent."""

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
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026 import vqa_gemini37_flash_smoke as smoke
from trec_video_2026.vqa_experiments.vqa_mc_deliberative_agent import (
    JUDGE_SCHEMA,
    OBSERVER_SCHEMA,
    PLANNER_SCHEMA,
    SOLVER_SCHEMA,
    MediaAction,
    append_observer,
    append_planner,
    choose_final_ranking,
    initialize_notebook,
    judge_prompt,
    notebook_observation_ids,
    observer_prompt,
    parse_judge,
    parse_observer,
    parse_planner,
    parse_solver,
    planner_prompt,
    solver_prompt,
)
from trec_video_2026.vqa_experiments.vqa_mc_common import (
    OPTION_IDS,
    canonical_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "single-model-deliberative-native-av-v1"
BASELINE_FPS = 4.0
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
ROLE_OUTPUT_LIMITS = {
    "solver": 2048,
    "planner": 1536,
    "observer": 1536,
    "judge": 2048,
}
MEDIA_TOKENS_PER_SECOND = {
    (1.0, "low"): 78,
    (4.0, "low"): 312,
    (8.0, "low"): 624,
    (1.0, "high"): 284,
    (4.0, "high"): 1136,
    (8.0, "high"): 2272,
}
MAX_OBSERVER_INPUT_TOKENS_PER_ROW = 10_000
MAX_TEXT_INPUT_TOKENS_PER_ROW = 4_000
MAX_CALLS_PER_ROW = 7
MAX_OBSERVER_ACTIONS = 2
MAX_COST_USD = 0.75
INVALID_A1_COST_USD = 0.0952995
INTERRUPTED_A2_COST_USD = 0.06064163
PRIOR_SPEND_USD = INVALID_A1_COST_USD + INTERRUPTED_A2_COST_USD
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
    / "data/vqa/benchmarks/vqa-mc-deliberative-agent-poc20-20260822-a3"
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


def estimate_worst_case_cost(
    payload: Mapping[str, Any],
    *,
    cached_solver_keys: Collection[tuple[str, str]] = (),
) -> dict[str, Any]:
    variants = canonical_variants(payload)
    indexed = {row["video_id"]: row for row in payload["rows"]}
    seconds = sum(
        float(indexed[variant["video_id"]]["duration_seconds"])
        for variant in variants
    )
    row_count = len(variants)
    expected_solver_keys = {
        (stage, str(variant["video_id"]))
        for stage in ("anchor", "blind")
        for variant in variants
    }
    cached = set(cached_solver_keys)
    if not cached <= expected_solver_keys:
        raise ValueError("resume cache contains unexpected solver rows")
    missing_solver_seconds = sum(
        float(indexed[video_id]["duration_seconds"])
        for stage, video_id in expected_solver_keys - cached
    )
    base_media = math.ceil(
        (seconds + missing_solver_seconds)
        * MEDIA_TOKENS_PER_SECOND[(BASELINE_FPS, "low")]
    )
    observer_media = row_count * MAX_OBSERVER_INPUT_TOKENS_PER_ROW
    text_input = row_count * MAX_TEXT_INPUT_TOKENS_PER_ROW
    output_tokens = (
        (2 * row_count - len(cached)) * ROLE_OUTPUT_LIMITS["solver"]
        + row_count * (
            2 * ROLE_OUTPUT_LIMITS["planner"]
        + 2 * ROLE_OUTPUT_LIMITS["observer"]
        + ROLE_OUTPUT_LIMITS["judge"]
        )
    )
    input_tokens = base_media + observer_media + text_input
    cost = (
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    return {
        "row_count": row_count,
        "request_count_ceiling": row_count * MAX_CALLS_PER_ROW - len(cached),
        "reused_solver_call_count": len(cached),
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
        "prior_spend_usd": round(PRIOR_SPEND_USD, 8),
        "combined_estimated_cost_usd": round(
            PRIOR_SPEND_USD + cost, 8
        ),
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


def _resume_solver_cache(
    *,
    resume_from: Path | None,
    requests_path: Path,
    variants: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any] | None]:
    if resume_from is None:
        return {}, None
    if resume_from.is_symlink():
        raise ValueError("resume root must not be a symlink")
    root = resume_from.resolve(strict=True)
    manifest_path = root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("model") != MODEL
        or manifest.get("model_arm") != MODEL_ARM
        or manifest.get("requests_sha256") != sha256_file(requests_path)
    ):
        raise ValueError("resume manifest does not match this experiment")
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    raw_hashes: list[str] = []
    for stage in ("anchor", "blind"):
        for index, variant in enumerate(variants):
            path = root / stage / "raw" / f"{index:02d}.json"
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("resume raw solver artifact is invalid")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("status") != "received":
                raise ValueError("resume raw solver call was not received")
            _answer_usage(record)
            video_id = str(variant["video_id"])
            cache[(stage, video_id)] = record
            raw_hashes.append(sha256_file(path))
    if not cache:
        raise ValueError("resume root contains no reusable solver calls")
    return cache, {
        "manifest_sha256": sha256_file(manifest_path),
        "reused_solver_call_count": len(cache),
        "raw_sha256s_sha256": sha256_file_list(raw_hashes),
    }


def sha256_file_list(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(list(values))).hexdigest()


def preflight(
    *,
    requests_path: Path = DEFAULT_REQUESTS,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    max_cost_usd: float = MAX_COST_USD,
    resume_from: Path | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    payload = load_requests(
        requests_path,
        expected_rows=expected_rows,
        expected_variants=expected_variants,
    )
    variants = canonical_variants(payload)
    resume_cache, resume_metadata = _resume_solver_cache(
        resume_from=resume_from,
        requests_path=requests_path,
        variants=variants,
    )
    estimate = estimate_worst_case_cost(
        payload, cached_solver_keys=resume_cache
    )
    if estimate["combined_estimated_cost_usd"] > max_cost_usd:
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
    return {
        "status": "preflight_passed",
        **estimate,
        "resume": resume_metadata,
        "deliveries": deliveries,
    }


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


def build_planner_correction_request(
    *, row: Mapping[str, Any], invalid_response: str, round_index: int
) -> types.InlinedRequest:
    if round_index not in {1, 2}:
        raise ValueError("planner round is invalid")
    prompt = "\n".join(
        [
            "Repair the candidate planner JSON so it satisfies the attached schema.",
            "This is mechanical format repair, not a new analysis pass.",
            "Preserve its ranking, evidence references, decision, and intended action.",
            "Do not invent evidence or observation IDs.",
            "If its intended action cannot be represented validly, use finish_search",
            "with action null. Return only the repaired JSON object.",
            f"Video ID: {row['video_id']}",
            "Candidate planner JSON:",
            invalid_response,
        ]
    )
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part(text=prompt)])
        ],
        metadata={
            "arm": MODEL_ARM,
            "stage": f"planner_{round_index}_correction",
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


def _answer_usage(raw_record: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    if raw_record.get("status") != "received":
        raise ValueError("provider response was not received")
    safe = raw_record.get("response")
    payload = smoke._response_payload(safe)
    text = smoke._response_text(payload)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("provider response text is empty")
    return text.strip(), smoke._usage(payload)


def _usage_totals(usages: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    prompt_tokens = 0
    output_tokens = 0
    complete = True
    for usage in usages:
        if not isinstance(usage, Mapping):
            complete = False
            continue
        prompt = usage.get("prompt_tokens")
        output = usage.get("output_tokens")
        total = usage.get("total_tokens")
        if not isinstance(prompt, int) or not isinstance(total, int):
            complete = False
            continue
        if not isinstance(output, int):
            output = max(total - prompt, 0)
        prompt_tokens += prompt
        output_tokens += output
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


def _freeze_raw_stage(
    *, output_root: Path, stage_path: str, rows: Sequence[Mapping[str, Any]]
) -> Path:
    path = output_root / stage_path / "raw-freeze.json"
    _write_json(
        path,
        {"schema_version": 1, "status": "raw_frozen", "rows": list(rows)},
    )
    return path


def _planner_json(planner: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if planner is None:
        return None
    result = copy.deepcopy(dict(planner))
    if isinstance(result.get("action"), MediaAction):
        result["action"] = asdict(result["action"])
    return result


def _planner_references_are_valid(
    planner: Mapping[str, Any], notebook: Mapping[str, Any]
) -> bool:
    known = notebook_observation_ids(notebook)
    assessments = planner.get("option_assessments")
    if not isinstance(assessments, list):
        return False
    for assessment in assessments:
        if not isinstance(assessment, Mapping):
            return False
        for key in ("supporting_observation_ids", "opposing_observation_ids"):
            identifiers = assessment.get(key)
            if not isinstance(identifiers, list) or not set(identifiers) <= known:
                return False
    return True


def _record_tool_result(
    notebook: Mapping[str, Any], *, round_index: int, result: Mapping[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(notebook))
    rounds = updated.get("rounds")
    if not isinstance(rounds, list) or not 1 <= round_index <= len(rounds):
        raise ValueError("tool-result round does not exist")
    entry = rounds[round_index - 1]
    if not isinstance(entry, dict) or "tool_result" in entry:
        raise ValueError("tool result already exists")
    entry["tool_result"] = copy.deepcopy(dict(result))
    return updated


def _freeze_notebooks(
    *, output_root: Path, stage_name: str, variants: Sequence[Mapping[str, Any]],
    notebooks: Mapping[str, Mapping[str, Any]]
) -> Path:
    index_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        video_id = str(variant["video_id"])
        relative = Path("notebooks") / stage_name / f"{index:02d}.json"
        path = output_root / relative
        _write_json(path, notebooks[video_id])
        index_rows.append(
            {
                "video_id": video_id,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
            }
        )
    freeze = output_root / "notebooks" / f"{stage_name}-freeze.json"
    _write_json(
        freeze,
        {"schema_version": 1, "status": "notebooks_frozen", "rows": index_rows},
    )
    return freeze


def run(
    *,
    requests_path: Path = DEFAULT_REQUESTS,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT,
    resume_from: Path | None = None,
    client: Any,
    sleep: Callable[[float], None] = time.sleep,
    upload_file: Callable[[Any, Path], Mapping[str, Any]] | None = None,
    expected_rows: int | None = EXPECTED_ROWS,
    expected_variants: int | None = EXPECTED_VARIANTS,
) -> dict[str, Any]:
    gate = preflight(
        requests_path=requests_path,
        historical_root=historical_root,
        resume_from=resume_from,
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
    resume_cache, resume_metadata = _resume_solver_cache(
        resume_from=resume_from,
        requests_path=requests_path,
        variants=variants,
    )
    _write_json(
        output_root / "run-manifest.json",
        {
            "schema_version": 1,
            "experiment_id": payload["experiment_id"],
            "model": MODEL,
            "model_arm": MODEL_ARM,
            "requests_sha256": sha256_file(requests_path),
            "cost": {key: value for key, value in gate.items() if key != "deliveries"},
            "resume": resume_metadata,
            "deliveries": gate["deliveries"],
        },
    )
    uploader = upload_file or (
        lambda provider_client, path: _upload(provider_client, path, sleep=sleep)
    )
    uploads = {
        row["video_id"]: dict(
            uploader(client, _delivery_path(historical_root, row["video_id"]))
        )
        for row in payload["rows"]
    }
    usage_rows: list[Mapping[str, Any] | None] = []
    call_counts: Counter[str] = Counter()
    provider_call_counts: Counter[str] = Counter()
    reused_solver_call_count = 0
    budgets = {
        str(variant["video_id"]): {
            "calls_used": 0,
            "observer_input_tokens_remaining": MAX_OBSERVER_INPUT_TOKENS_PER_ROW,
            "observer_actions": 0,
            "search_finished": False,
        }
        for variant in variants
    }

    solver_outputs: dict[str, dict[str, Mapping[str, Any] | None]] = {
        str(variant["video_id"]): {} for variant in variants
    }
    solver_freezes: dict[str, Path] = {}
    for stage, solver_id in (("anchor", "solver_1"), ("blind", "solver_2")):
        raw_index: list[dict[str, Any]] = []
        parsed_rows: list[dict[str, Any]] = []
        for index, variant in enumerate(variants):
            video_id = str(variant["video_id"])
            row = indexed[video_id]
            request = build_solver_request(row=row, upload=uploads[video_id], stage=stage)
            cached_record = resume_cache.get((stage, video_id))
            if cached_record is not None:
                raw_record = copy.deepcopy(cached_record)
                reused_solver_call_count += 1
            else:
                raw_record = _provider_call(client, request)
                provider_call_counts[stage] += 1
            budgets[video_id]["calls_used"] += 1
            call_counts[stage] += 1
            raw_path = output_root / stage / "raw" / f"{index:02d}.json"
            _write_json(raw_path, raw_record)
            raw_index.append(
                {
                    "video_id": video_id,
                    "status": raw_record["status"],
                    "raw_sha256": sha256_file(raw_path),
                }
            )
            record = None
            usage = None
            error_type = raw_record.get("error_type")
            status = "failed"
            if raw_record.get("status") == "received":
                try:
                    answer, usage = _answer_usage(raw_record)
                    record = parse_solver(
                        answer,
                        duration_seconds=float(row["duration_seconds"]),
                        solver_id=solver_id,
                    )
                    status = "passed"
                    error_type = None
                except Exception as error:  # noqa: BLE001 - freeze parser failure
                    error_type = smoke._safe_error(error)
            usage_rows.append(usage)
            solver_outputs[video_id][solver_id] = record
            parsed_rows.append(
                {
                    "video_id": video_id,
                    "variant_kind": "canonical",
                    "status": status,
                    "record": record,
                    "ranking": record["ranking"] if record else None,
                    "usage": usage,
                    "error_type": error_type,
                }
            )
        raw_freeze = _freeze_raw_stage(
            output_root=output_root, stage_path=stage, rows=raw_index
        )
        parsed_freeze = output_root / stage / "predictions-freeze.json"
        _write_json(
            parsed_freeze,
            {
                "schema_version": 1,
                "status": "predictions_frozen",
                "raw_freeze_sha256": sha256_file(raw_freeze),
                "rows": parsed_rows,
            },
        )
        solver_freezes[stage] = parsed_freeze

    notebooks: dict[str, dict[str, Any]] = {}
    for variant in variants:
        video_id = str(variant["video_id"])
        valid = [
            record
            for record in (
                solver_outputs[video_id].get("solver_1"),
                solver_outputs[video_id].get("solver_2"),
            )
            if isinstance(record, Mapping)
        ]
        notebooks[video_id] = initialize_notebook(video_id, valid)
    initial_notebooks = _freeze_notebooks(
        output_root=output_root,
        stage_name="initial",
        variants=variants,
        notebooks=notebooks,
    )

    round_notebook_freezes: list[Path] = []
    for round_index in (1, 2):
        stage_root = f"round-{round_index}"
        planner_raw_index: list[dict[str, Any]] = []
        planner_parsed_rows: list[dict[str, Any]] = []
        observer_raw_index: list[dict[str, Any]] = []
        observer_parsed_rows: list[dict[str, Any]] = []
        for index, variant in enumerate(variants):
            video_id = str(variant["video_id"])
            row = indexed[video_id]
            budget = budgets[video_id]
            eligible = bool(notebooks[video_id]["solvers"]) and not budget[
                "search_finished"
            ]
            if round_index == 2:
                prior_rounds = notebooks[video_id]["rounds"]
                eligible = bool(
                    eligible
                    and prior_rounds
                    and isinstance(prior_rounds[-1], Mapping)
                    and isinstance(prior_rounds[-1].get("observer"), Mapping)
                )
            planner_raw: dict[str, Any]
            planner = None
            planner_usage = None
            correction_usage = None
            planner_error = None
            correction_attempted = False
            if eligible and int(budget["calls_used"]) + 1 < MAX_CALLS_PER_ROW:
                budget_view = {
                    "calls_used": budget["calls_used"],
                    "calls_remaining": MAX_CALLS_PER_ROW - int(budget["calls_used"]),
                    "observer_input_tokens_remaining": budget[
                        "observer_input_tokens_remaining"
                    ],
                    "final_call_reserved": True,
                    "cost_cap_usd": MAX_COST_USD,
                }
                request = build_planner_request(
                    row=row,
                    notebook=notebooks[video_id],
                    budget=budget_view,
                    round_index=round_index,
                )
                planner_raw = _provider_call(client, request)
                provider_call_counts["planner"] += 1
                budget["calls_used"] += 1
                call_counts["planner"] += 1
                if planner_raw.get("status") == "received":
                    invalid_answer = None
                    try:
                        answer, planner_usage = _answer_usage(planner_raw)
                        invalid_answer = answer
                        planner = parse_planner(
                            answer, duration_seconds=float(row["duration_seconds"])
                        )
                        if not _planner_references_are_valid(
                            planner, notebooks[video_id]
                        ):
                            raise ValueError("planner cites an unknown observation ID")
                    except Exception as error:  # noqa: BLE001 - freeze parser failure
                        planner_error = smoke._safe_error(error)
                        planner = None
                    if (
                        planner is None
                        and invalid_answer is not None
                        and int(budget["calls_used"]) + 1 < MAX_CALLS_PER_ROW
                    ):
                        correction_attempted = True
                        correction_request = build_planner_correction_request(
                            row=row,
                            invalid_response=invalid_answer,
                            round_index=round_index,
                        )
                        correction_raw = _provider_call(client, correction_request)
                        provider_call_counts["planner_correction"] += 1
                        planner_raw["correction"] = correction_raw
                        budget["calls_used"] += 1
                        call_counts["planner_correction"] += 1
                        if correction_raw.get("status") == "received":
                            try:
                                corrected_answer, correction_usage = _answer_usage(
                                    correction_raw
                                )
                                planner = parse_planner(
                                    corrected_answer,
                                    duration_seconds=float(row["duration_seconds"]),
                                )
                                if not _planner_references_are_valid(
                                    planner, notebooks[video_id]
                                ):
                                    raise ValueError(
                                        "corrected planner cites an unknown observation ID"
                                    )
                                planner_error = None
                            except Exception as error:  # noqa: BLE001 - parser failure
                                planner_error = smoke._safe_error(error)
                                planner = None
            else:
                planner_raw = {"status": "skipped", "reason": "not_eligible"}
            if planner_raw["status"] != "skipped":
                usage_rows.append(planner_usage)
            if correction_attempted:
                usage_rows.append(correction_usage)
            planner_raw_path = (
                output_root / stage_root / "planner/raw" / f"{index:02d}.json"
            )
            _write_json(planner_raw_path, planner_raw)
            planner_raw_index.append(
                {
                    "video_id": video_id,
                    "status": planner_raw["status"],
                    "raw_sha256": sha256_file(planner_raw_path),
                }
            )
            if planner is not None:
                notebooks[video_id] = append_planner(
                    notebooks[video_id], round_index=round_index, planner=planner
                )
                if planner["decision"] == "finish_search":
                    budget["search_finished"] = True
            elif eligible:
                budget["search_finished"] = True
            planner_parsed_rows.append(
                {
                    "video_id": video_id,
                    "status": "passed" if planner is not None else "skipped" if not eligible else "failed",
                    "planner": _planner_json(planner),
                    "usage": planner_usage,
                    "correction_attempted": correction_attempted,
                    "correction_usage": correction_usage,
                    "error_type": planner_error,
                }
            )

            observer_raw: dict[str, Any] = {"status": "skipped", "reason": "not_requested"}
            observer = None
            observer_usage = None
            observer_error = None
            tool_result: dict[str, Any] | None = None
            if planner is not None and planner["decision"] == "inspect":
                action = planner["action"]
                if not isinstance(action, MediaAction):
                    raise AssertionError("parsed inspect action is missing")
                authorization = authorize_action(
                    action,
                    calls_used=int(budget["calls_used"]),
                    observer_input_tokens_remaining=int(
                        budget["observer_input_tokens_remaining"]
                    ),
                )
                tool_result = authorization
                if authorization["authorized"]:
                    request = build_observer_request(
                        upload=uploads[video_id], action=action
                    )
                    observer_raw = _provider_call(client, request)
                    provider_call_counts["observer"] += 1
                    budget["calls_used"] += 1
                    budget["observer_actions"] += 1
                    budget["observer_input_tokens_remaining"] = authorization[
                        "remaining_input_tokens"
                    ]
                    call_counts["observer"] += 1
                    if observer_raw.get("status") == "received":
                        try:
                            answer, observer_usage = _answer_usage(observer_raw)
                            observer = parse_observer(
                                answer,
                                action=action,
                                duration_seconds=float(row["duration_seconds"]),
                                view_id=f"{video_id}-round-{round_index}",
                            )
                        except Exception as error:  # noqa: BLE001 - parser failure
                            observer_error = smoke._safe_error(error)
                    if observer is not None:
                        notebooks[video_id] = append_observer(
                            notebooks[video_id],
                            round_index=round_index,
                            action=action,
                            observer=observer,
                            provenance={
                                "view_id": f"{video_id}-round-{round_index}",
                                "start_seconds": action.start_seconds,
                                "end_seconds": action.end_seconds,
                                "fps": action.fps,
                                "resolution": action.resolution,
                                "estimated_input_tokens": authorization[
                                    "estimated_input_tokens"
                                ],
                            },
                        )
                    else:
                        budget["search_finished"] = True
                else:
                    notebooks[video_id] = _record_tool_result(
                        notebooks[video_id],
                        round_index=round_index,
                        result=authorization,
                    )
                    budget["search_finished"] = True
            if observer_raw["status"] != "skipped":
                usage_rows.append(observer_usage)
            observer_raw_path = (
                output_root / stage_root / "observer/raw" / f"{index:02d}.json"
            )
            _write_json(observer_raw_path, observer_raw)
            observer_raw_index.append(
                {
                    "video_id": video_id,
                    "status": observer_raw["status"],
                    "raw_sha256": sha256_file(observer_raw_path),
                }
            )
            observer_parsed_rows.append(
                {
                    "video_id": video_id,
                    "status": "passed" if observer is not None else "skipped" if observer_raw["status"] == "skipped" else "failed",
                    "observer": observer,
                    "tool_result": tool_result,
                    "usage": observer_usage,
                    "error_type": observer_error,
                }
            )
        planner_raw_freeze = _freeze_raw_stage(
            output_root=output_root,
            stage_path=f"{stage_root}/planner",
            rows=planner_raw_index,
        )
        _write_json(
            output_root / stage_root / "planner/predictions-freeze.json",
            {
                "schema_version": 1,
                "status": "predictions_frozen",
                "raw_freeze_sha256": sha256_file(planner_raw_freeze),
                "rows": planner_parsed_rows,
            },
        )
        observer_raw_freeze = _freeze_raw_stage(
            output_root=output_root,
            stage_path=f"{stage_root}/observer",
            rows=observer_raw_index,
        )
        _write_json(
            output_root / stage_root / "observer/observations-freeze.json",
            {
                "schema_version": 1,
                "status": "observations_frozen",
                "raw_freeze_sha256": sha256_file(observer_raw_freeze),
                "rows": observer_parsed_rows,
            },
        )
        round_notebook_freezes.append(
            _freeze_notebooks(
                output_root=output_root,
                stage_name=f"round-{round_index}",
                variants=variants,
                notebooks=notebooks,
            )
        )

    judge_raw_index: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    final_notebook_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        video_id = str(variant["video_id"])
        row = indexed[video_id]
        budget = budgets[video_id]
        if int(budget["calls_used"]) >= MAX_CALLS_PER_ROW:
            raise ValueError("final judge reserve was consumed")
        request = build_judge_request(
            row=row, upload=uploads[video_id], notebook=notebooks[video_id]
        )
        raw_record = _provider_call(client, request)
        provider_call_counts["judge"] += 1
        budget["calls_used"] += 1
        call_counts["judge"] += 1
        raw_path = output_root / "judge/raw" / f"{index:02d}.json"
        _write_json(raw_path, raw_record)
        judge_raw_index.append(
            {
                "video_id": video_id,
                "status": raw_record["status"],
                "raw_sha256": sha256_file(raw_path),
            }
        )
        judge = None
        usage = None
        error_type = raw_record.get("error_type")
        if raw_record.get("status") == "received":
            try:
                answer, usage = _answer_usage(raw_record)
                judge = parse_judge(
                    answer,
                    observation_ids=set(notebook_observation_ids(notebooks[video_id])),
                )
                error_type = None
            except Exception as error:  # noqa: BLE001 - freeze parser failure
                error_type = smoke._safe_error(error)
        usage_rows.append(usage)
        final_notebook = copy.deepcopy(notebooks[video_id])
        final_notebook["judge"] = {
            "status": "passed" if judge is not None else "failed",
            "record": judge,
            "error_type": error_type,
        }
        relative = Path("notebooks/final") / f"{index:02d}.json"
        notebook_path = output_root / relative
        _write_json(notebook_path, final_notebook)
        final_notebook_rows.append(
            {
                "video_id": video_id,
                "path": relative.as_posix(),
                "sha256": sha256_file(notebook_path),
            }
        )
        planner_records = [
            entry["planner"]
            for entry in notebooks[video_id]["rounds"]
            if isinstance(entry, Mapping) and isinstance(entry.get("planner"), Mapping)
        ]
        solver_records = [
            record
            for record in notebooks[video_id]["solvers"]
            if isinstance(record, Mapping)
        ]
        ranking, source = choose_final_ranking(
            judge, planner_records, solver_records
        )
        final_rows.append(
            {
                "video_id": video_id,
                "variant_kind": "canonical",
                "status": "passed" if ranking is not None else "failed",
                "ranking": ranking,
                "final_source": source,
                "judge_status": "passed" if judge is not None else "failed",
                "judge": judge,
                "notebook_path": relative.as_posix(),
                "notebook_sha256": sha256_file(notebook_path),
                "call_count": budget["calls_used"],
                "observer_action_count": budget["observer_actions"],
                "observer_input_tokens_remaining": budget[
                    "observer_input_tokens_remaining"
                ],
                "usage": usage,
                "error_type": error_type,
            }
        )
    judge_raw_freeze = _freeze_raw_stage(
        output_root=output_root, stage_path="judge", rows=judge_raw_index
    )
    final_notebook_freeze = output_root / "notebooks/final-freeze.json"
    _write_json(
        final_notebook_freeze,
        {
            "schema_version": 1,
            "status": "notebooks_frozen",
            "rows": final_notebook_rows,
        },
    )
    source_paths = {
        "anchor": solver_freezes["anchor"],
        "blind": solver_freezes["blind"],
        "initial_notebooks": initial_notebooks,
        "round_1_notebooks": round_notebook_freezes[0],
        "round_2_notebooks": round_notebook_freezes[1],
        "judge_raw": judge_raw_freeze,
        "final_notebooks": final_notebook_freeze,
    }
    final_freeze = output_root / "final-predictions-freeze.json"
    _write_json(
        final_freeze,
        {
            "schema_version": 1,
            "status": "predictions_frozen",
            "model": MODEL,
            "source_sha256s": {
                key: sha256_file(path) for key, path in source_paths.items()
            },
            "source_paths": {
                key: path.relative_to(output_root).as_posix()
                for key, path in source_paths.items()
            },
            "rows": final_rows,
        },
    )
    usage = _usage_totals(usage_rows)
    path_counts = Counter(
        f"{row['observer_action_count']}_actions" for row in final_rows
    )
    summary = {
        "status": "completed",
        "row_count": len(variants),
        "request_count": sum(int(row["call_count"]) for row in final_rows),
        "provider_request_count": sum(provider_call_counts.values()),
        "provider_call_counts_by_role": dict(sorted(provider_call_counts.items())),
        "reused_solver_call_count": reused_solver_call_count,
        "call_counts_by_role": dict(sorted(call_counts.items())),
        "observer_call_count": call_counts["observer"],
        "final_valid_count": sum(row["status"] == "passed" for row in final_rows),
        "judge_valid_count": sum(row["judge_status"] == "passed" for row in final_rows),
        "path_distribution": dict(sorted(path_counts.items())),
        "estimated_cost_ceiling_usd": gate["estimated_cost_usd"],
        "prior_nonreused_cost_usd": INVALID_A1_COST_USD,
        "interrupted_reused_cost_usd": INTERRUPTED_A2_COST_USD,
        "combined_estimated_cost_ceiling_usd": gate[
            "combined_estimated_cost_usd"
        ],
        **usage,
    }
    summary["combined_observed_cost_usd"] = (
        round(INVALID_A1_COST_USD + float(summary["observed_cost_usd"]), 8)
        if summary["observed_cost_usd"] is not None
        else None
    )
    _write_json(output_root / "summary.json", summary)
    return summary


def verify_output(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve(strict=True)
    final_path = root / "final-predictions-freeze.json"
    summary_path = root / "summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = final.get("rows")
    if not isinstance(rows, list) or len(rows) != summary.get("row_count"):
        raise ValueError("final row count does not match summary")
    for key, relative in final.get("source_paths", {}).items():
        candidate = (root / str(relative)).resolve(strict=True)
        candidate.relative_to(root)
        if sha256_file(candidate) != final["source_sha256s"].get(key):
            raise ValueError("source freeze hash mismatch")
    for row in rows:
        relative = row.get("notebook_path")
        if not isinstance(relative, str):
            raise ValueError("final notebook path is invalid")
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
        if sha256_file(candidate) != row.get("notebook_sha256"):
            raise ValueError("final notebook hash mismatch")
        notebook = json.loads(candidate.read_text(encoding="utf-8"))
        judge = row.get("judge")
        if isinstance(judge, Mapping):
            if not set(judge.get("decisive_observation_ids", [])) <= set(
                notebook_observation_ids(notebook)
            ):
                raise ValueError("final judge citation mismatch")
    return {
        "status": "verified",
        "row_count": len(rows),
        "final_predictions_sha256": sha256_file(final_path),
        "summary_sha256": sha256_file(summary_path),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_vqa_mc_gemini_deliberative_agent_poc20"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify-output", type=Path)
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.preflight or args.run) and not args.canonical_only:
            raise ValueError("deliberative experiment requires --canonical-only")
        if args.preflight:
            result = preflight(
                requests_path=args.requests,
                historical_root=args.historical_root,
                resume_from=args.resume_from,
            )
            result = {key: value for key, value in result.items() if key != "deliveries"}
        elif args.run:
            result = run(
                requests_path=args.requests,
                historical_root=args.historical_root,
                output_root=args.output,
                resume_from=args.resume_from,
                client=genai.Client(api_key=_api_key()),
            )
        else:
            result = verify_output(args.verify_output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:  # noqa: BLE001 - do not leak paths or provider responses
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
