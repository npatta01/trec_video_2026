#!/usr/bin/env python3
"""Run and freeze the label-blind Gemini atomic counting treatment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026.vqa_counting_evidence import (
    ATOMIC_COUNT_UNITS,
    build_atomic_evidence_prompt,
    derive_count,
    validate_atomic_evidence,
    validate_label_blind,
)
from trec_video_2026.vqa_submission import canonical_sha256
from trec_video_2026.vqa_submission_gemini import ProviderResponseEvidenceError
from trec_video_2026.vqa_submission_openrouter import (
    OpenRouterRequestPolicy,
    OpenRouterTransport,
)

MODEL = "gemini-3.1-pro-preview"
FPS = 4.0
MEDIA_RESOLUTION = "HIGH"
THINKING_LEVEL = "HIGH"
TEMPERATURE = 0.0
SEED = 42
MAX_OUTPUT_TOKENS = 8192
BUDGET_CEILING_USD = 3.0
# This issue's frozen panel is shorter than 30 seconds. At the pinned standard
# rates, $0.50 reserves the maximum sub-200k input charge plus all 8,192 output
# tokens. Requests outside that frozen duration boundary are rejected.
STANDARD_REQUEST_RESERVATION_USD = 0.50
BATCH_REQUEST_RESERVATION_USD = 0.25
MAX_FROZEN_DURATION_SECONDS = 30.0

RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "count_unit",
        "target",
        "scope",
        "records",
        "exclusions",
        "uncertainty",
    ],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1, "maximum": 1},
        "count_unit": {"type": "string", "enum": sorted(ATOMIC_COUNT_UNITS)},
        "target": {"type": "string"},
        "scope": {"type": "string"},
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "record_id",
                    "start_seconds",
                    "end_seconds",
                    "scene",
                    "rationale",
                ],
                "properties": {
                    "record_id": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "scene": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "exclusions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["start_seconds", "end_seconds", "reason"],
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "uncertainty": {"type": "string"},
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, value: object, *, raw: bool = False) -> None:
    if raw:
        if not isinstance(value, str):
            raise TypeError("raw writes require a string")
        rendered = value
    else:
        rendered = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _replace_json(path: Path, value: Mapping[str, object]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def reserve_attempt(*, row_root: Path, attempt: Mapping[str, object]) -> Path:
    """Atomically and permanently claim one paid attempt before provider access."""

    row_root.mkdir(parents=True, exist_ok=True)
    claim_path = row_root / "paid-attempt-claim.json"
    claim = {
        "schema_version": 1,
        "claimed_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        **attempt,
    }
    rendered = json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        raise FileExistsError(f"paid attempt already claimed: {claim_path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def _self_hashed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def _load_budget_ledger(path: Path) -> dict[str, object]:
    if not path.exists():
        return _self_hashed(
            {
                "schema_version": 1,
                "budget_ceiling_usd": BUDGET_CEILING_USD,
                "reservations": {},
            }
        )
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError("issue budget ledger must be an object")
    expected = payload.get("complete_sha256")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    )
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("issue budget ledger hash changed")
    if payload.get("budget_ceiling_usd") != BUDGET_CEILING_USD:
        raise ValueError("issue budget ledger ceiling changed")
    if not isinstance(payload.get("reservations"), dict):
        raise TypeError("issue budget reservations must be an object")
    return payload


def _observed_issue_spend(budget_root: Path) -> float:
    costs: list[float] = []
    seen: set[Path] = set()
    for path in sorted(budget_root.glob("**/rows/*/complete.json")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        complete = _load_json(path)
        if not isinstance(complete, Mapping):
            raise TypeError(f"terminal complete must be an object: {path}")
        value = complete.get("estimated_cost_usd")
        if isinstance(value, int | float) and not isinstance(value, bool):
            costs.append(float(value))
    return sum(costs)


def reserve_issue_budget(
    *, budget_root: Path, reservation_id: str, reserved_usd: float
) -> dict[str, object]:
    """Reserve issue-level spend under an exclusive cross-process lock."""

    if not reservation_id:
        raise ValueError("reservation_id must be non-empty")
    budget_root.mkdir(parents=True, exist_ok=True)
    lock_path = budget_root / "issue-34-budget.lock"
    ledger_path = budget_root / "issue-34-budget-ledger.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = _load_budget_ledger(ledger_path)
        reservations = ledger["reservations"]
        assert isinstance(reservations, dict)
        if reservation_id in reservations:
            raise FileExistsError(
                f"budget reservation already exists: {reservation_id}"
            )
        pending = sum(
            float(value["reserved_usd"])
            for value in reservations.values()
            if isinstance(value, Mapping) and value.get("status") == "reserved"
        )
        spent = _observed_issue_spend(budget_root)
        assert_budget_allows(
            spent_usd=spent + pending,
            projected_request_usd=reserved_usd,
        )
        reservation = {
            "status": "reserved",
            "reserved_usd": reserved_usd,
            "created_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        }
        reservations[reservation_id] = reservation
        ledger["observed_spend_usd_at_reservation"] = spent
        ledger["updated_at"] = datetime.now(UTC).isoformat()
        ledger.pop("complete_sha256", None)
        _replace_json(ledger_path, _self_hashed(ledger))
        return reservation


def settle_issue_budget(
    *,
    budget_root: Path,
    reservation_id: str,
    actual_usd: float,
    allow_missing: bool = False,
) -> None:
    lock_path = budget_root / "issue-34-budget.lock"
    ledger_path = budget_root / "issue-34-budget-ledger.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if allow_missing and not ledger_path.exists():
            return
        ledger = _load_budget_ledger(ledger_path)
        reservations = ledger["reservations"]
        assert isinstance(reservations, dict)
        reservation = reservations.get(reservation_id)
        if allow_missing and reservation is None:
            return
        if not isinstance(reservation, dict) or reservation.get("status") != "reserved":
            raise ValueError(f"budget reservation is not pending: {reservation_id}")
        reserved = float(reservation["reserved_usd"])
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError("actual cost must be finite and non-negative")
        reservation.update(
            {
                "status": "settled" if actual_usd <= reserved else "exceeded",
                "actual_usd": actual_usd,
                "settled_at": datetime.now(UTC).isoformat(),
            }
        )
        ledger["updated_at"] = datetime.now(UTC).isoformat()
        ledger.pop("complete_sha256", None)
        _replace_json(ledger_path, _self_hashed(ledger))
        if actual_usd > reserved:
            raise RuntimeError(
                f"actual request cost {actual_usd:.6f} exceeded reserved {reserved:.6f}"
            )


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def estimate_cost_usd(
    *,
    prompt_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    pricing: Mapping[str, object],
    mode: str = "standard",
) -> float:
    for name, value in {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    threshold = int(pricing.get("high_context_threshold_tokens", 200_000))
    if mode not in {"standard", "batch"}:
        raise ValueError(f"unknown pricing mode: {mode}")
    prefix = "batch_" if mode == "batch" else ""
    if prompt_tokens >= threshold:
        input_rate = float(
            pricing[f"{prefix}high_context_input_usd_per_million_tokens"]
        )
        output_rate = float(
            pricing[f"{prefix}high_context_output_and_thinking_usd_per_million_tokens"]
        )
    else:
        input_rate = float(pricing[f"{prefix}input_usd_per_million_tokens"])
        output_rate = float(
            pricing[f"{prefix}output_and_thinking_usd_per_million_tokens"]
        )
    return (
        prompt_tokens * input_rate + (output_tokens + thinking_tokens) * output_rate
    ) / 1_000_000


def assert_budget_allows(
    *,
    spent_usd: float,
    projected_request_usd: float,
    ceiling_usd: float = BUDGET_CEILING_USD,
) -> None:
    if not all(
        math.isfinite(value) and value >= 0
        for value in (spent_usd, projected_request_usd)
    ):
        raise ValueError("cost values must be finite and non-negative")
    if spent_usd + projected_request_usd > ceiling_usd + 1e-12:
        raise ValueError(
            f"projected request would exceed the USD {ceiling_usd:.2f} ceiling: "
            f"spent={spent_usd:.6f}, projected_request={projected_request_usd:.6f}"
        )


def assert_terminal_allows_continuation(result: Mapping[str, object]) -> None:
    if result.get("status") != "passed":
        raise RuntimeError(
            f"stopping phase after {result.get('video_id')} terminal status "
            f"{result.get('status')}"
        )


def assert_request_reservation_bound(row: Mapping[str, object]) -> None:
    duration = row.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or not math.isfinite(float(duration))
        or float(duration) <= 0
        or float(duration) > MAX_FROZEN_DURATION_SECONDS
    ):
        raise ValueError("request exceeds the 30-second frozen-panel bound")


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=TEMPERATURE,
        seed=SEED,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH,
            include_thoughts=False,
        ),
    )


def _video_part(row: Mapping[str, object]) -> types.Part:
    return types.Part(
        file_data=types.FileData(
            file_uri=str(row["file_uri"]), mime_type=str(row["delivery_mime_type"])
        ),
        video_metadata=types.VideoMetadata(fps=FPS),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )


def build_batch_request(
    *,
    row: Mapping[str, object],
    request_manifest_sha256: str,
    pricing_sha256: str,
) -> types.InlinedRequest:
    validate_label_blind(row)
    prompt = build_atomic_evidence_prompt(
        question=str(row["question"]),
        duration_seconds=float(row["duration_seconds"]),
        frozen_map=str(row["frozen_map"]),
    )
    return types.InlinedRequest(
        contents=types.Content(
            role="user",
            parts=[_video_part(row), types.Part(text=prompt)],
        ),
        metadata={
            "condition": "atomic-counting-evidence-v1",
            "pricing_sha256": pricing_sha256,
            "request_manifest_sha256": request_manifest_sha256,
            "video_id": str(row["video_id"]),
        },
        config=_config(),
    )


def _usage_metadata(response: object, elapsed_seconds: float) -> dict[str, object]:
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or []
    finish_reason = None
    if candidates:
        value = getattr(candidates[0], "finish_reason", None)
        finish_reason = (
            getattr(value, "name", str(value)) if value is not None else None
        )
    return {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "prompt_token_count": int(getattr(usage, "prompt_token_count", 0) or 0),
        "candidates_token_count": int(getattr(usage, "candidates_token_count", 0) or 0),
        "thoughts_token_count": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "cached_content_token_count": int(
            getattr(usage, "cached_content_token_count", 0) or 0
        ),
        "total_token_count": int(getattr(usage, "total_token_count", 0) or 0),
        "model_version": getattr(response, "model_version", None),
        "response_id": getattr(response, "response_id", None),
        "finish_reason": finish_reason,
    }


def _verify_media(row: Mapping[str, object]) -> None:
    source = Path(str(row["source_path"]))
    if not source.is_file():
        raise FileNotFoundError(f"source media missing: {source}")
    if _file_sha256(source) != row["source_sha256"]:
        raise ValueError("source media SHA-256 changed")


def _terminal_complete(
    *, row_root: Path, video_id: str, result: Mapping[str, object]
) -> dict[str, object]:
    complete = {
        "schema_version": 1,
        "video_id": video_id,
        "status": result["status"],
        "request_sha256": _file_sha256(row_root / "request.json"),
        "raw_response_sha256": (
            _file_sha256(row_root / "raw-response.txt")
            if (row_root / "raw-response.txt").is_file()
            else None
        ),
        "metadata_sha256": (
            _file_sha256(row_root / "provider-metadata.json")
            if (row_root / "provider-metadata.json").is_file()
            else None
        ),
        "result_sha256": _file_sha256(row_root / "result.json"),
        "attempt_claim_sha256": (
            _file_sha256(row_root / "paid-attempt-claim.json")
            if (row_root / "paid-attempt-claim.json").is_file()
            else None
        ),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
    }
    _write_once(row_root / "complete.json", complete)
    return complete


def _load_existing_terminal(row_root: Path) -> dict[str, object]:
    complete = _load_json(row_root / "complete.json")
    result = _load_json(row_root / "result.json")
    if not isinstance(complete, Mapping) or not isinstance(result, dict):
        raise TypeError("existing terminal artifacts must be objects")
    if _file_sha256(row_root / "result.json") != complete.get("result_sha256"):
        raise ValueError("existing terminal result hash changed")
    if result.get("status") != complete.get("status"):
        raise ValueError("existing terminal status changed")
    return result


def _settle_if_cost_known(
    *, budget_root: Path, reservation_id: str, result: Mapping[str, object]
) -> None:
    actual = result.get("estimated_cost_usd")
    if isinstance(actual, int | float) and not isinstance(actual, bool):
        settle_issue_budget(
            budget_root=budget_root,
            reservation_id=reservation_id,
            actual_usd=float(actual),
        )


def run_one(
    *,
    root: Path,
    row: Mapping[str, object],
    request_manifest_sha256: str,
    pricing: Mapping[str, object],
    client: object,
    spent_usd: float,
    projected_request_usd: float,
    verify_media: bool = False,
    budget_root: Path | None = None,
    reserved_request_usd: float = STANDARD_REQUEST_RESERVATION_USD,
) -> dict[str, object]:
    validate_label_blind(row)
    video_id = str(row["video_id"])
    row_root = root / "rows" / video_id
    complete_path = row_root / "complete.json"
    if complete_path.is_file():
        return _load_existing_terminal(row_root)
    assert_budget_allows(
        spent_usd=spent_usd, projected_request_usd=projected_request_usd
    )
    assert_request_reservation_bound(row)
    reserve_attempt(
        row_root=row_root,
        attempt={
            "video_id": video_id,
            "provider": "google-direct",
            "request_manifest_sha256": request_manifest_sha256,
        },
    )
    issue_root = budget_root or root
    reservation_id = f"google-direct:{request_manifest_sha256}:{video_id}"
    reserve_issue_budget(
        budget_root=issue_root,
        reservation_id=reservation_id,
        reserved_usd=reserved_request_usd,
    )
    if verify_media:
        _verify_media(row)

    uploaded = client.files.get(name=str(row["file_name"]))
    state_value = getattr(
        getattr(uploaded, "state", None), "name", getattr(uploaded, "state", None)
    )
    if state_value != "ACTIVE":
        raise ValueError(f"frozen Gemini upload is not ACTIVE: {state_value}")
    if getattr(uploaded, "uri", None) != row["file_uri"]:
        raise ValueError("live Gemini file URI differs from the frozen upload")

    prompt = build_atomic_evidence_prompt(
        question=str(row["question"]),
        duration_seconds=float(row["duration_seconds"]),
        frozen_map=str(row["frozen_map"]),
    )
    request = {
        "schema_version": 1,
        "request_manifest_sha256": request_manifest_sha256,
        "video_id": video_id,
        "source_sha256": row["source_sha256"],
        "delivery_sha256": row["delivery_sha256"],
        "file_name": row["file_name"],
        "file_uri": row["file_uri"],
        "model": MODEL,
        "fps": FPS,
        "media_resolution": MEDIA_RESOLUTION,
        "thinking_level": THINKING_LEVEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_schema_sha256": canonical_sha256(RESPONSE_SCHEMA),
        "prompt": prompt,
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "pricing_sha256": canonical_sha256(pricing),
    }
    validate_label_blind(request)
    _write_once(row_root / "request.json", request)

    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[_video_part(row), prompt],
            config=_config(),
        )
    except Exception as error:  # noqa: BLE001 - persist provider failure evidence
        terminal_error = {
            "schema_version": 1,
            "video_id": video_id,
            "status": "provider_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "estimated_cost_usd": None,
        }
        _write_once(row_root / "result.json", terminal_error)
        _terminal_complete(row_root=row_root, video_id=video_id, result=terminal_error)
        _settle_if_cost_known(
            budget_root=issue_root,
            reservation_id=reservation_id,
            result=terminal_error,
        )
        return terminal_error

    raw_text = getattr(response, "text", None) or ""
    metadata = _usage_metadata(response, time.monotonic() - started)
    estimated_cost = estimate_cost_usd(
        prompt_tokens=int(metadata["prompt_token_count"]),
        output_tokens=int(metadata["candidates_token_count"]),
        thinking_tokens=int(metadata["thoughts_token_count"]),
        pricing=pricing,
    )
    metadata["estimated_cost_usd"] = estimated_cost
    metadata["pricing_sha256"] = canonical_sha256(pricing)
    metadata["raw_response_sha256"] = _sha256_bytes(raw_text.encode())
    _write_once(row_root / "raw-response.txt", raw_text, raw=True)
    _write_once(row_root / "provider-metadata.json", metadata)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        result: dict[str, object] = {
            "schema_version": 1,
            "video_id": video_id,
            "status": "invalid_json",
            "parse_error": str(error),
            "derived_count": None,
            "estimated_cost_usd": estimated_cost,
        }
    else:
        try:
            evidence = validate_atomic_evidence(
                parsed, duration_seconds=float(row["duration_seconds"])
            )
        except ValueError as error:
            result = {
                "schema_version": 1,
                "video_id": video_id,
                "status": "invalid_schema",
                "validation_error": str(error),
                "derived_count": None,
                "estimated_cost_usd": estimated_cost,
            }
        else:
            result = {
                "schema_version": 1,
                "video_id": video_id,
                "status": "passed",
                "count_unit": evidence.count_unit,
                "derived_count": derive_count(evidence),
                "record_count": len(evidence.records),
                "evidence": parsed,
                "estimated_cost_usd": estimated_cost,
            }
    _write_once(row_root / "result.json", result)
    _terminal_complete(row_root=row_root, video_id=video_id, result=result)
    _settle_if_cost_known(
        budget_root=issue_root,
        reservation_id=reservation_id,
        result=result,
    )
    return result


def _source_video_mime_type(path: Path) -> str:
    values = {
        ".mkv": "video/x-matroska",
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }
    try:
        return values[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported source media suffix: {path.suffix}") from error


def run_openrouter_one(
    *,
    root: Path,
    row: Mapping[str, object],
    request_manifest_sha256: str,
    pricing: Mapping[str, object],
    transport: object,
    spent_usd: float,
    projected_request_usd: float,
    budget_root: Path | None = None,
    reserved_request_usd: float = STANDARD_REQUEST_RESERVATION_USD,
) -> dict[str, object]:
    validate_label_blind(row)
    video_id = str(row["video_id"])
    row_root = root / "rows" / video_id
    if (row_root / "complete.json").is_file():
        return _load_existing_terminal(row_root)
    assert_budget_allows(
        spent_usd=spent_usd, projected_request_usd=projected_request_usd
    )
    assert_request_reservation_bound(row)
    reserve_attempt(
        row_root=row_root,
        attempt={
            "video_id": video_id,
            "provider": "openrouter",
            "request_manifest_sha256": request_manifest_sha256,
        },
    )
    issue_root = budget_root or root
    reservation_id = f"openrouter:{request_manifest_sha256}:{video_id}"
    reserve_issue_budget(
        budget_root=issue_root,
        reservation_id=reservation_id,
        reserved_usd=reserved_request_usd,
    )
    _verify_media(row)
    source = Path(str(row["source_path"]))
    mime_type = _source_video_mime_type(source)
    uploaded = transport.upload(source, mime_type=mime_type)
    prompt = build_atomic_evidence_prompt(
        question=str(row["question"]),
        duration_seconds=float(row["duration_seconds"]),
        frozen_map=str(row["frozen_map"]),
    )
    request = {
        "schema_version": 1,
        "request_manifest_sha256": request_manifest_sha256,
        "video_id": video_id,
        "source_path": str(source),
        "source_sha256": row["source_sha256"],
        "source_mime_type": mime_type,
        "uploaded_file_id": uploaded.file_id,
        "uploaded_sha256": uploaded.sha256,
        "provider": "openrouter",
        "model": MODEL,
        "reasoning_effort": "high",
        "temperature": TEMPERATURE,
        "seed_policy": "content-addressed-openrouter-transport",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_schema_sha256": canonical_sha256(RESPONSE_SCHEMA),
        "prompt": prompt,
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "pricing_sha256": canonical_sha256(pricing),
    }
    validate_label_blind(request)
    _write_once(row_root / "request.json", request)
    started = time.monotonic()
    try:
        generation = transport.generate(
            model=MODEL,
            parts=[{"file_id": uploaded.file_id, "mime_type": mime_type}, prompt],
            schema=RESPONSE_SCHEMA,
        )
    except Exception as error:  # noqa: BLE001 - persist provider failure evidence
        evidence = (
            error.generation_result
            if isinstance(error, ProviderResponseEvidenceError)
            else None
        )
        estimated_cost: float | None = None
        if evidence is not None:
            raw_text = evidence.raw_response
            provider_metadata = dict(evidence.provider_metadata)
            reported = provider_metadata.get("reported_cost_usd")
            estimated_cost = (
                float(reported)
                if isinstance(reported, int | float) and not isinstance(reported, bool)
                else estimate_cost_usd(
                    prompt_tokens=evidence.usage.input_tokens,
                    output_tokens=evidence.usage.output_tokens,
                    thinking_tokens=0,
                    pricing=pricing,
                )
            )
            _write_once(row_root / "raw-response.txt", raw_text, raw=True)
            _write_once(
                row_root / "provider-metadata.json",
                {
                    **provider_metadata,
                    "input_tokens": evidence.usage.input_tokens,
                    "output_tokens": evidence.usage.output_tokens,
                    "estimated_cost_usd": estimated_cost,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                },
            )
        result = {
            "schema_version": 1,
            "video_id": video_id,
            "status": "provider_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "derived_count": None,
            "estimated_cost_usd": estimated_cost,
        }
        _write_once(row_root / "result.json", result)
        _terminal_complete(row_root=row_root, video_id=video_id, result=result)
        _settle_if_cost_known(
            budget_root=issue_root,
            reservation_id=reservation_id,
            result=result,
        )
        return result
    finally:
        transport.delete(uploaded.file_id)

    provider_metadata = dict(generation.provider_metadata)
    reported = provider_metadata.get("reported_cost_usd")
    estimated_cost = (
        float(reported)
        if isinstance(reported, int | float) and not isinstance(reported, bool)
        else estimate_cost_usd(
            prompt_tokens=generation.usage.input_tokens,
            output_tokens=generation.usage.output_tokens,
            thinking_tokens=0,
            pricing=pricing,
        )
    )
    raw_text = generation.raw_response
    metadata = {
        **provider_metadata,
        "input_tokens": generation.usage.input_tokens,
        "output_tokens": generation.usage.output_tokens,
        "total_tokens": generation.usage.total_tokens,
        "cached_input_tokens": generation.usage.cached_input_tokens,
        "estimated_cost_usd": estimated_cost,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "raw_response_sha256": _sha256_bytes(raw_text.encode()),
        "pricing_sha256": canonical_sha256(pricing),
    }
    _write_once(row_root / "raw-response.txt", raw_text, raw=True)
    _write_once(row_root / "provider-metadata.json", metadata)
    try:
        evidence = validate_atomic_evidence(
            generation.payload, duration_seconds=float(row["duration_seconds"])
        )
    except ValueError as error:
        result = {
            "schema_version": 1,
            "video_id": video_id,
            "status": "invalid_schema",
            "validation_error": str(error),
            "derived_count": None,
            "estimated_cost_usd": estimated_cost,
        }
    else:
        result = {
            "schema_version": 1,
            "video_id": video_id,
            "status": "passed",
            "count_unit": evidence.count_unit,
            "derived_count": derive_count(evidence),
            "record_count": len(evidence.records),
            "evidence": generation.payload,
            "estimated_cost_usd": estimated_cost,
        }
    _write_once(row_root / "result.json", result)
    _terminal_complete(row_root=row_root, video_id=video_id, result=result)
    _settle_if_cost_known(
        budget_root=issue_root,
        reservation_id=reservation_id,
        result=result,
    )
    return result


def freeze_predictions(
    *,
    root: Path,
    video_ids: Sequence[str],
    request_manifest_sha256: str,
    filename: str,
) -> dict[str, object]:
    rows = []
    statuses: Counter[str] = Counter()
    cost = 0.0
    for video_id in video_ids:
        complete_path = root / "rows" / video_id / "complete.json"
        if not complete_path.is_file():
            raise ValueError(f"incomplete row: {video_id}")
        complete = _load_json(complete_path)
        status = str(complete["status"])
        statuses[status] += 1
        row_cost = complete.get("estimated_cost_usd")
        if isinstance(row_cost, int | float) and not isinstance(row_cost, bool):
            cost += float(row_cost)
        rows.append(
            {
                "video_id": video_id,
                "complete_sha256": _file_sha256(complete_path),
                "result_sha256": complete["result_sha256"],
                "status": status,
            }
        )
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "request_manifest_sha256": request_manifest_sha256,
        "row_count": len(rows),
        "terminal_status_counts": dict(sorted(statuses.items())),
        "estimated_cost_usd": cost,
        "rows": rows,
    }
    freeze["complete_sha256"] = canonical_sha256(freeze)
    _write_once(root / filename, freeze)
    return freeze


def _batch_usage(response: Mapping[str, object]) -> dict[str, object]:
    usage = response.get("usageMetadata", {})
    if not isinstance(usage, Mapping):
        usage = {}
    candidates = response.get("candidates", [])
    finish_reason = None
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, Mapping):
            finish_reason = first.get("finishReason")
    return {
        "prompt_token_count": int(usage.get("promptTokenCount", 0) or 0),
        "candidates_token_count": int(usage.get("candidatesTokenCount", 0) or 0),
        "thoughts_token_count": int(usage.get("thoughtsTokenCount", 0) or 0),
        "cached_content_token_count": int(usage.get("cachedContentTokenCount", 0) or 0),
        "total_token_count": int(usage.get("totalTokenCount", 0) or 0),
        "model_version": response.get("modelVersion"),
        "response_id": response.get("responseId"),
        "finish_reason": finish_reason,
    }


def _batch_response_text(response: Mapping[str, object]) -> str:
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, Mapping):
        return ""
    content = first.get("content", {})
    if not isinstance(content, Mapping):
        return ""
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, Mapping) and part.get("text") is not None
    )


def _batch_request_artifact(
    *,
    row: Mapping[str, object],
    request_manifest_sha256: str,
    pricing: Mapping[str, object],
) -> dict[str, object]:
    prompt = build_atomic_evidence_prompt(
        question=str(row["question"]),
        duration_seconds=float(row["duration_seconds"]),
        frozen_map=str(row["frozen_map"]),
    )
    request = {
        "schema_version": 1,
        "request_manifest_sha256": request_manifest_sha256,
        "video_id": row["video_id"],
        "source_sha256": row["source_sha256"],
        "delivery_sha256": row["delivery_sha256"],
        "file_name": row["file_name"],
        "file_uri": row["file_uri"],
        "model": MODEL,
        "mode": "batch",
        "fps": FPS,
        "media_resolution": MEDIA_RESOLUTION,
        "thinking_level": THINKING_LEVEL,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_schema_sha256": canonical_sha256(RESPONSE_SCHEMA),
        "prompt": prompt,
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "pricing_sha256": canonical_sha256(pricing),
    }
    validate_label_blind(request)
    return request


def materialize_batch_payload(
    *,
    root: Path,
    payload: Mapping[str, object],
    rows_by_id: Mapping[str, Mapping[str, object]],
    expected_video_ids: Sequence[str],
    request_manifest_sha256: str,
    pricing: Mapping[str, object],
) -> list[dict[str, object]]:
    dest = payload.get("dest", {})
    if not isinstance(dest, Mapping):
        raise TypeError("batch payload has no destination")
    items = dest.get("inlinedResponses", [])
    if not isinstance(items, list):
        raise TypeError("batch payload has no inlined responses")
    by_id: dict[str, Mapping[str, object]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("batch response item must be an object")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("batch response metadata must be an object")
        video_id = metadata.get("video_id")
        if not isinstance(video_id, str) or video_id not in rows_by_id:
            raise ValueError("batch response video_id is outside the frozen panel")
        if metadata.get("condition") != "atomic-counting-evidence-v1":
            raise ValueError("batch response condition changed")
        if metadata.get("request_manifest_sha256") != request_manifest_sha256:
            raise ValueError("batch response request manifest binding changed")
        if video_id in by_id:
            raise ValueError(f"duplicate batch response: {video_id}")
        by_id[video_id] = item
    if set(by_id) != set(expected_video_ids):
        raise ValueError("batch response IDs do not match the submitted phase")

    results: list[dict[str, object]] = []
    for video_id in expected_video_ids:
        row = rows_by_id[video_id]
        item = by_id[video_id]
        row_root = root / "rows" / video_id
        request = _batch_request_artifact(
            row=row,
            request_manifest_sha256=request_manifest_sha256,
            pricing=pricing,
        )
        _write_once(row_root / "request.json", request)
        if item.get("error") is not None:
            result = {
                "schema_version": 1,
                "video_id": video_id,
                "status": "provider_error",
                "provider_error": item["error"],
                "derived_count": None,
                "estimated_cost_usd": None,
            }
            _write_once(row_root / "result.json", result)
            _terminal_complete(row_root=row_root, video_id=video_id, result=result)
            results.append(result)
            continue
        response = item.get("response", {})
        if not isinstance(response, Mapping):
            raise TypeError(f"batch response is not an object: {video_id}")
        raw_text = _batch_response_text(response)
        metadata = _batch_usage(response)
        estimated_cost = estimate_cost_usd(
            prompt_tokens=int(metadata["prompt_token_count"]),
            output_tokens=int(metadata["candidates_token_count"]),
            thinking_tokens=int(metadata["thoughts_token_count"]),
            pricing=pricing,
            mode="batch",
        )
        metadata.update(
            {
                "batch_job_name": payload.get("name"),
                "estimated_cost_usd": estimated_cost,
                "pricing_sha256": canonical_sha256(pricing),
                "raw_response_sha256": _sha256_bytes(raw_text.encode()),
            }
        )
        _write_once(row_root / "raw-response.txt", raw_text, raw=True)
        _write_once(row_root / "provider-metadata.json", metadata)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as error:
            result = {
                "schema_version": 1,
                "video_id": video_id,
                "status": "invalid_json",
                "parse_error": str(error),
                "derived_count": None,
                "estimated_cost_usd": estimated_cost,
            }
        else:
            try:
                evidence = validate_atomic_evidence(
                    parsed, duration_seconds=float(row["duration_seconds"])
                )
            except ValueError as error:
                result = {
                    "schema_version": 1,
                    "video_id": video_id,
                    "status": "invalid_schema",
                    "validation_error": str(error),
                    "derived_count": None,
                    "estimated_cost_usd": estimated_cost,
                }
            else:
                result = {
                    "schema_version": 1,
                    "video_id": video_id,
                    "status": "passed",
                    "count_unit": evidence.count_unit,
                    "derived_count": derive_count(evidence),
                    "record_count": len(evidence.records),
                    "evidence": parsed,
                    "estimated_cost_usd": estimated_cost,
                }
        _write_once(row_root / "result.json", result)
        _terminal_complete(row_root=row_root, video_id=video_id, result=result)
        results.append(result)
    return results


def _phase_video_ids(
    manifest: Mapping[str, object], phase: str, root: Path
) -> list[str]:
    all_ids = [str(row["video_id"]) for row in manifest["rows"]]
    canary_ids = [str(value) for value in manifest["canary_video_ids"]]
    if phase == "canary":
        return canary_ids
    if phase != "remaining":
        raise ValueError(f"unknown phase: {phase}")
    decision = _load_json(root / "canary-decision.json")
    if decision.get("authorized") is not True:
        raise ValueError("canary did not authorize remaining inference")
    if decision.get("request_manifest_sha256") != manifest["complete_sha256"]:
        raise ValueError("canary decision is bound to a different request manifest")
    return [video_id for video_id in all_ids if video_id not in canary_ids]


def submit_batch(*, root: Path, phase: str, pricing_path: Path) -> None:
    manifest = _manifest(root)
    pricing = _load_json(pricing_path)
    video_ids = _phase_video_ids(manifest, phase, root)
    budget_root = _issue_budget_root(root)
    reserved_batch_usd = len(video_ids) * BATCH_REQUEST_RESERVATION_USD
    assert_budget_allows(
        spent_usd=_spent_and_projection(budget_root)[0],
        projected_request_usd=reserved_batch_usd,
    )
    rows_by_id = {str(row["video_id"]): row for row in manifest["rows"]}
    pricing_sha256 = canonical_sha256(pricing)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    requests = []
    request_records = []
    for video_id in video_ids:
        row = rows_by_id[video_id]
        assert_request_reservation_bound(row)
        _verify_media(row)
        uploaded = client.files.get(name=str(row["file_name"]))
        state_value = getattr(
            getattr(uploaded, "state", None), "name", getattr(uploaded, "state", None)
        )
        if state_value != "ACTIVE" or getattr(uploaded, "uri", None) != row["file_uri"]:
            raise ValueError(f"frozen upload preflight failed: {video_id}")
        request_records.append(
            _batch_request_artifact(
                row=row,
                request_manifest_sha256=str(manifest["complete_sha256"]),
                pricing=pricing,
            )
        )
        requests.append(
            build_batch_request(
                row=row,
                request_manifest_sha256=str(manifest["complete_sha256"]),
                pricing_sha256=pricing_sha256,
            )
        )
    request_freeze = {
        "schema_version": 1,
        "phase": phase,
        "request_manifest_sha256": manifest["complete_sha256"],
        "pricing_sha256": pricing_sha256,
        "reserved_cost_usd": reserved_batch_usd,
        "rows": request_records,
    }
    request_freeze["complete_sha256"] = canonical_sha256(request_freeze)
    _write_once(root / "batch-requests" / f"{phase}.json", request_freeze)
    record = root / "jobs" / f"{phase}.json"
    if record.exists():
        raise FileExistsError(f"refusing duplicate paid batch submission: {record}")
    reserve_attempt(
        row_root=root / "jobs" / "claims" / phase,
        attempt={
            "phase": phase,
            "provider": "google-batch",
            "request_manifest_sha256": manifest["complete_sha256"],
        },
    )
    reservation_id = f"google-batch:{manifest['complete_sha256']}:{phase}"
    reserve_issue_budget(
        budget_root=budget_root,
        reservation_id=reservation_id,
        reserved_usd=reserved_batch_usd,
    )
    try:
        job = client.batches.create(
            model=MODEL,
            src=requests,
            config={"display_name": f"issue-34-counting-{phase}"},
        )
    except Exception as error:
        _write_once(
            root / "jobs" / f"{phase}-submission-error.json",
            {
                "schema_version": 1,
                "phase": phase,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "request_freeze_sha256": _file_sha256(
                    root / "batch-requests" / f"{phase}.json"
                ),
            },
        )
        raise
    payload = {
        "schema_version": 1,
        "phase": phase,
        "submitted_at": datetime.now(UTC).isoformat(),
        "video_ids": video_ids,
        "request_freeze_sha256": _file_sha256(
            root / "batch-requests" / f"{phase}.json"
        ),
        "job": job.model_dump(mode="json", by_alias=True, exclude_none=True),
    }
    _write_once(record, payload)
    print(json.dumps({"phase": phase, "job_name": job.name, "state": str(job.state)}))


def poll_batch(*, root: Path, phase: str, pricing_path: Path) -> None:
    manifest = _manifest(root)
    pricing = _load_json(pricing_path)
    record = _load_json(root / "jobs" / f"{phase}.json")
    expected_ids = [str(value) for value in record["video_ids"]]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    job = client.batches.get(name=record["job"]["name"])
    state = getattr(job.state, "name", str(job.state))
    payload = job.model_dump(mode="json", by_alias=True, exclude_none=True)
    checked_at = datetime.now(UTC)
    _write_once(
        root / "status" / phase / f"{checked_at.strftime('%Y%m%dT%H%M%S%fZ')}.json",
        {
            "schema_version": 1,
            "checked_at": checked_at.isoformat(),
            "state": state,
            "job": payload,
        },
    )
    print(json.dumps({"phase": phase, "name": job.name, "state": state}))
    if state not in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }:
        return
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"batch terminated without success: {state}")
    rows_by_id = {str(row["video_id"]): row for row in manifest["rows"]}
    results = materialize_batch_payload(
        root=root,
        payload=payload,
        rows_by_id=rows_by_id,
        expected_video_ids=expected_ids,
        request_manifest_sha256=str(manifest["complete_sha256"]),
        pricing=pricing,
    )
    freeze_predictions(
        root=root,
        video_ids=expected_ids
        if phase == "canary"
        else [str(row["video_id"]) for row in manifest["rows"]],
        request_manifest_sha256=str(manifest["complete_sha256"]),
        filename="canary-predictions-freeze.json"
        if phase == "canary"
        else "predictions-freeze.json",
    )
    row_costs = [result.get("estimated_cost_usd") for result in results]
    if all(
        isinstance(cost, int | float) and not isinstance(cost, bool)
        for cost in row_costs
    ):
        settle_issue_budget(
            budget_root=_issue_budget_root(root),
            reservation_id=f"google-batch:{manifest['complete_sha256']}:{phase}",
            actual_usd=sum(float(cost) for cost in row_costs),
            allow_missing=True,
        )
    nonpassing = [result for result in results if result["status"] != "passed"]
    if nonpassing:
        raise RuntimeError(f"batch has {len(nonpassing)} non-passing terminal rows")


def _spent_and_projection(root: Path) -> tuple[float, float]:
    return _observed_issue_spend(root), STANDARD_REQUEST_RESERVATION_USD


def _issue_budget_root(root: Path) -> Path:
    manifest_path = root / "requests.label-blind.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"request manifest is missing: {manifest_path}")
    return manifest_path.resolve().parent


def _manifest(root: Path) -> dict[str, object]:
    payload = _load_json(root / "requests.label-blind.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise TypeError("invalid request manifest")
    validate_label_blind(payload)
    if canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    ) != payload.get("complete_sha256"):
        raise ValueError("request manifest complete_sha256 is invalid")
    return payload


def run_phase(*, root: Path, phase: str, pricing_path: Path) -> None:
    manifest = _manifest(root)
    pricing = _load_json(pricing_path)
    if pricing.get("model") != MODEL:
        raise ValueError("pricing model does not match the runner")
    rows_by_id = {str(row["video_id"]): row for row in manifest["rows"]}
    canary_ids = [str(value) for value in manifest["canary_video_ids"]]
    if phase == "canary":
        video_ids = canary_ids
    elif phase == "remaining":
        decision = _load_json(root / "canary-decision.json")
        if decision.get("authorized") is not True:
            raise ValueError("canary did not authorize remaining inference")
        if decision.get("request_manifest_sha256") != manifest["complete_sha256"]:
            raise ValueError("canary decision is bound to a different request manifest")
        video_ids = [video_id for video_id in rows_by_id if video_id not in canary_ids]
    else:
        raise ValueError(f"unknown phase: {phase}")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    budget_root = _issue_budget_root(root)
    for video_id in video_ids:
        spent, projection = _spent_and_projection(budget_root)
        result = run_one(
            root=root,
            row=rows_by_id[video_id],
            request_manifest_sha256=str(manifest["complete_sha256"]),
            pricing=pricing,
            client=client,
            spent_usd=spent,
            projected_request_usd=projection,
            verify_media=True,
            budget_root=budget_root,
        )
        print(
            json.dumps(
                {
                    "video_id": video_id,
                    "status": result["status"],
                    "derived_count": result.get("derived_count"),
                    "estimated_cost_usd": result.get("estimated_cost_usd"),
                },
                sort_keys=True,
            )
        )
        assert_terminal_allows_continuation(result)


def run_openrouter_phase(*, root: Path, phase: str, pricing_path: Path) -> None:
    manifest = _manifest(root)
    pricing = _load_json(pricing_path)
    rows_by_id = {str(row["video_id"]): row for row in manifest["rows"]}
    video_ids = _phase_video_ids(manifest, phase, root)
    policy = OpenRouterRequestPolicy(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        allow_fallbacks=False,
        use_structured_outputs=True,
    )
    transport = OpenRouterTransport(os.environ["OPENROUTER_API_KEY"], policy=policy)
    budget_root = _issue_budget_root(root)
    preflight = transport.preflight(model_id=MODEL)
    _write_once(
        root / "openrouter-preflight.json",
        {
            "schema_version": 1,
            **preflight,
            "pricing_sha256": canonical_sha256(pricing),
            "request_manifest_sha256": manifest["complete_sha256"],
            "policy": {
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "reasoning_effort": "high",
                "allow_fallbacks": False,
                "structured_outputs": True,
            },
        },
    )
    for video_id in video_ids:
        spent, projection = _spent_and_projection(budget_root)
        result = run_openrouter_one(
            root=root,
            row=rows_by_id[video_id],
            request_manifest_sha256=str(manifest["complete_sha256"]),
            pricing=pricing,
            transport=transport,
            spent_usd=spent,
            projected_request_usd=projection,
            budget_root=budget_root,
        )
        print(
            json.dumps(
                {
                    "video_id": video_id,
                    "status": result["status"],
                    "derived_count": result.get("derived_count"),
                    "estimated_cost_usd": result.get("estimated_cost_usd"),
                },
                sort_keys=True,
            )
        )
        assert_terminal_allows_continuation(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", required=True, type=Path)
    run_parser.add_argument("--phase", required=True, choices=("canary", "remaining"))
    run_parser.add_argument("--pricing", required=True, type=Path)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--root", required=True, type=Path)
    freeze_parser.add_argument("--phase", required=True, choices=("canary", "full"))
    submit_batch_parser = subparsers.add_parser("submit-batch")
    submit_batch_parser.add_argument("--root", required=True, type=Path)
    submit_batch_parser.add_argument(
        "--phase", required=True, choices=("canary", "remaining")
    )
    submit_batch_parser.add_argument("--pricing", required=True, type=Path)
    poll_batch_parser = subparsers.add_parser("poll-batch")
    poll_batch_parser.add_argument("--root", required=True, type=Path)
    poll_batch_parser.add_argument(
        "--phase", required=True, choices=("canary", "remaining")
    )
    poll_batch_parser.add_argument("--pricing", required=True, type=Path)
    openrouter_parser = subparsers.add_parser("run-openrouter")
    openrouter_parser.add_argument("--root", required=True, type=Path)
    openrouter_parser.add_argument(
        "--phase", required=True, choices=("canary", "remaining")
    )
    openrouter_parser.add_argument("--pricing", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "run":
        run_phase(root=args.root, phase=args.phase, pricing_path=args.pricing)
        return
    if args.command == "submit-batch":
        submit_batch(root=args.root, phase=args.phase, pricing_path=args.pricing)
        return
    if args.command == "poll-batch":
        poll_batch(root=args.root, phase=args.phase, pricing_path=args.pricing)
        return
    if args.command == "run-openrouter":
        run_openrouter_phase(
            root=args.root, phase=args.phase, pricing_path=args.pricing
        )
        return
    manifest = _manifest(args.root)
    all_ids = [str(row["video_id"]) for row in manifest["rows"]]
    video_ids = (
        [str(value) for value in manifest["canary_video_ids"]]
        if args.phase == "canary"
        else all_ids
    )
    filename = (
        "canary-predictions-freeze.json"
        if args.phase == "canary"
        else "predictions-freeze.json"
    )
    freeze = freeze_predictions(
        root=args.root,
        video_ids=video_ids,
        request_manifest_sha256=str(manifest["complete_sha256"]),
        filename=filename,
    )
    print(
        json.dumps(
            {
                "row_count": freeze["row_count"],
                "terminal_status_counts": freeze["terminal_status_counts"],
                "estimated_cost_usd": freeze["estimated_cost_usd"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
