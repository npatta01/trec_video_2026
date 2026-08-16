#!/usr/bin/env python3
"""Run the direct-Google multi-pass Gemini counting recovery canary."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_gemini_counting_recovery import (
    COUNT_UNITS,
    AtomicLedger,
    TimeRange,
    build_challenger_prompt,
    build_locator_prompt,
    build_verifier_prompt,
    parse_atomic_ledger,
    parse_challenge,
    parse_locator,
    reduce_recovery_result,
    verification_envelope,
)
from trec_video_2026.vqa_submission import canonical_sha256

MODEL = "gemini-3.1-pro-preview"
LOCATOR_FPS = 4.0
VERIFY_FPS = 12.0
BUDGET_CEILING_USD = 1.0
TEMPERATURE = 0.0
SEED = 42
MAX_OUTPUT_TOKENS = 2048
ORDERED_VIDEO_IDS = ("RZrkA52TGBA", "KoQD9x6VrSk", "Loyn1vxCI6E")
CUMULATIVE_LIMITS_USD = (0.34, 0.67, 1.0)

_SPAN_SCHEMA = {
    "type": "object",
    "required": ["start_seconds", "end_seconds"],
    "properties": {
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
    },
}
_INSTANCE_SCHEMA = {
    "type": "object",
    "required": [
        "instance_id",
        "start_seconds",
        "peak_seconds",
        "end_seconds",
        "before_state",
        "during_state",
        "after_state",
        "separation_evidence",
        "predicate_supported",
        "ambiguous",
    ],
    "properties": {
        "instance_id": {"type": "string"},
        "start_seconds": {"type": "number"},
        "peak_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "before_state": {"type": "string"},
        "during_state": {"type": "string"},
        "after_state": {"type": "string"},
        "separation_evidence": {"type": "string"},
        "predicate_supported": {"type": "boolean"},
        "ambiguous": {"type": "boolean"},
    },
}
LOCATOR_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "count_unit",
        "target",
        "candidates",
        "reviewed_spans",
        "unresolved_spans",
    ],
    "properties": {
        "schema_version": {"type": "integer"},
        "count_unit": {"type": "string", "enum": sorted(COUNT_UNITS)},
        "target": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "candidate_id",
                    "start_seconds",
                    "end_seconds",
                    "reason",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        },
        "reviewed_spans": {"type": "array", "items": _SPAN_SCHEMA},
        "unresolved_spans": {"type": "array", "items": _SPAN_SCHEMA},
    },
}
LEDGER_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "instances", "unresolved_observations"],
    "properties": {
        "schema_version": {"type": "integer"},
        "instances": {"type": "array", "items": _INSTANCE_SCHEMA},
        "unresolved_observations": {"type": "array", "items": {"type": "string"}},
    },
}
CHALLENGE_RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "instance_dispositions",
        "missed_instances",
        "requested_ranges",
        "unresolved_reasons",
    ],
    "properties": {
        "schema_version": {"type": "integer"},
        "instance_dispositions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["instance_id", "disposition", "reason"],
                "properties": {
                    "instance_id": {"type": "string"},
                    "disposition": {
                        "type": "string",
                        "enum": [
                            "accept",
                            "ambiguous",
                            "duplicate",
                            "predicate_mismatch",
                            "split",
                        ],
                    },
                    "reason": {"type": "string"},
                },
            },
        },
        "missed_instances": {"type": "array", "items": _INSTANCE_SCHEMA},
        "requested_ranges": {
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
        "unresolved_reasons": {"type": "array", "items": {"type": "string"}},
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def _write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _write_once_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def build_video_part(
    *,
    file_uri: str,
    mime_type: str,
    fps: float,
    time_range: TimeRange | None,
) -> types.Part:
    metadata: dict[str, object] = {"fps": fps}
    if time_range is not None:
        metadata.update(
            {
                "start_offset": f"{time_range.start_seconds:.3f}s",
                "end_offset": f"{time_range.end_seconds:.3f}s",
            }
        )
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(**metadata),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )


def reserve_paid_stage(
    stage_root: Path, request_identity: Mapping[str, object]
) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    claim_path = stage_root / "paid-call-claim.json"
    request_hash = _verify_self_hash(
        request_identity, name=f"{request_identity.get('stage', 'paid stage')} request"
    )
    claim = _self_hashed(
        {
            "schema_version": 1,
            "claimed_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "stage": request_identity.get("stage"),
            "request_complete_sha256": request_hash,
        }
    )
    rendered = json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        raise FileExistsError(f"paid stage already claimed: {claim_path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def reserve_upload(*, row_root: Path, row: Mapping[str, object]) -> Path:
    row_root.mkdir(parents=True, exist_ok=True)
    claim_path = row_root / "upload-claim.json"
    claim = _self_hashed(
        {
            "schema_version": 1,
            "claimed_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "video_id": row["video_id"],
            "delivery_sha256": row["delivery_sha256"],
            "delivery_mime_type": row["delivery_mime_type"],
        }
    )
    rendered = json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(
            claim_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        raise FileExistsError(f"upload already claimed: {claim_path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def _initial_ledger() -> dict[str, object]:
    return _self_hashed(
        {
            "schema_version": 1,
            "budget_ceiling_usd": BUDGET_CEILING_USD,
            "reservations": {},
        }
    )


def _load_budget_ledger(path: Path) -> dict[str, object]:
    if not path.exists():
        return _initial_ledger()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("budget ledger must be an object")
    expected = payload.pop("complete_sha256", None)
    actual = canonical_sha256(payload)
    payload["complete_sha256"] = expected
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("budget ledger hash changed")
    if payload.get("budget_ceiling_usd") != BUDGET_CEILING_USD:
        raise ValueError("budget ledger ceiling changed")
    if not isinstance(payload.get("reservations"), dict):
        raise TypeError("budget reservations must be an object")
    return payload


def _exposure(reservations: Mapping[str, object]) -> float:
    total = 0.0
    for value in reservations.values():
        if not isinstance(value, Mapping):
            raise TypeError("budget reservation must be an object")
        status = value.get("status")
        if status == "reserved":
            total += float(value["reserved_usd"])
        elif status == "settled":
            total += float(value["observed_usd"])
        else:
            raise ValueError("unknown budget reservation status")
    return total


def reserve_budget(
    *, budget_root: Path, reservation_id: str, reserved_usd: float
) -> dict[str, object]:
    if not reservation_id or reserved_usd <= 0.0:
        raise ValueError("budget reservation requires a positive amount and ID")
    budget_root.mkdir(parents=True, exist_ok=True)
    lock_path = budget_root / "budget.lock"
    ledger_path = budget_root / "budget-ledger.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = _load_budget_ledger(ledger_path)
        reservations = ledger["reservations"]
        assert isinstance(reservations, dict)
        if reservation_id in reservations:
            raise FileExistsError(
                f"budget reservation already exists: {reservation_id}"
            )
        projected = _exposure(reservations) + reserved_usd
        if projected > BUDGET_CEILING_USD + 1e-12:
            raise RuntimeError(
                f"budget ceiling would be exceeded: {projected:.6f} > "
                f"{BUDGET_CEILING_USD:.6f}"
            )
        reservations[reservation_id] = {
            "status": "reserved",
            "reserved_usd": reserved_usd,
            "reserved_at": datetime.now(UTC).isoformat(),
        }
        updated = _self_hashed(
            {
                "schema_version": 1,
                "budget_ceiling_usd": BUDGET_CEILING_USD,
                "reservations": reservations,
            }
        )
        _replace_json(ledger_path, updated)
        return updated


def settle_budget(
    *, budget_root: Path, reservation_id: str, observed_usd: float
) -> dict[str, object]:
    if observed_usd < 0.0:
        raise ValueError("observed cost must be non-negative")
    lock_path = budget_root / "budget.lock"
    ledger_path = budget_root / "budget-ledger.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = _load_budget_ledger(ledger_path)
        reservations = ledger["reservations"]
        assert isinstance(reservations, dict)
        value = reservations.get(reservation_id)
        if not isinstance(value, Mapping) or value.get("status") != "reserved":
            raise ValueError(f"reservation is not pending: {reservation_id}")
        reserved = float(value["reserved_usd"])
        if observed_usd > reserved + 1e-12:
            raise RuntimeError(
                f"observed cost {observed_usd:.6f} exceeds reservation {reserved:.6f}"
            )
        reservations[reservation_id] = {
            **value,
            "status": "settled",
            "observed_usd": observed_usd,
            "settled_at": datetime.now(UTC).isoformat(),
        }
        updated = _self_hashed(
            {
                "schema_version": 1,
                "budget_ceiling_usd": BUDGET_CEILING_USD,
                "reservations": reservations,
            }
        )
        _replace_json(ledger_path, updated)
        return updated


def prepare_request_manifest(
    *,
    source_rows: Sequence[Mapping[str, object]],
    ordered_video_ids: tuple[str, ...],
    delivery_root: Path,
    condition: str = "gemini-native-counting-recovery-v1",
) -> dict[str, object]:
    by_id = {str(row.get("video_id")): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("source rows contain duplicate video IDs")
    rows: list[dict[str, object]] = []
    for video_id in ordered_video_ids:
        if video_id not in by_id:
            raise ValueError(f"selected video is missing: {video_id}")
        source = by_id[video_id]
        source_path = Path(str(source["source_path"]))
        delivery_path = delivery_root / video_id / "delivery-media" / f"{video_id}.webm"
        if _file_sha256(source_path) != source["source_sha256"]:
            raise ValueError(f"source hash changed: {video_id}")
        if _file_sha256(delivery_path) != source["delivery_sha256"]:
            raise ValueError(f"delivery hash changed: {video_id}")
        rows.append(
            {
                "video_id": video_id,
                "question": source["question"],
                "duration_seconds": source["duration_seconds"],
                "source_path": str(source_path),
                "source_sha256": source["source_sha256"],
                "delivery_path": str(delivery_path),
                "delivery_sha256": source["delivery_sha256"],
                "delivery_mime_type": source["delivery_mime_type"],
            }
        )
    payload = _self_hashed(
        {
            "schema_version": 1,
            "condition": condition,
            "model": MODEL,
            "ordered_video_ids": list(ordered_video_ids),
            "rows": rows,
        }
    )
    validate_label_blind(payload)
    return payload


def ensure_upload(
    *, client: Any, row: Mapping[str, object], row_root: Path
) -> dict[str, object]:
    media = Path(str(row["delivery_path"]))
    expected_sha256 = str(row["delivery_sha256"])
    if _file_sha256(media) != expected_sha256:
        raise ValueError("delivery media hash changed before upload")
    record_path = row_root / "upload.json"
    if record_path.exists():
        record = _load_mapping(record_path)
        expected = record.pop("complete_sha256", None)
        actual = canonical_sha256(record)
        record["complete_sha256"] = expected
        if not isinstance(expected, str) or expected != actual:
            raise ValueError("upload record hash changed")
        uploaded = client.files.get(name=str(record["file_name"]))
        state = getattr(getattr(uploaded, "state", None), "name", None)
        if state != "ACTIVE" or str(uploaded.uri) != record["file_uri"]:
            raise RuntimeError("recorded Gemini upload is no longer active")
        return record
    upload_claim = reserve_upload(row_root=row_root, row=row)
    upload_claim_payload = _load_mapping(upload_claim)
    upload_claim_hash = _verify_self_hash(upload_claim_payload, name="upload claim")
    uploaded = client.files.upload(
        file=str(media),
        config={
            "mime_type": str(row["delivery_mime_type"]),
            "display_name": f"counting-recovery-{row['video_id']}",
        },
    )
    for _ in range(55):
        uploaded = client.files.get(name=uploaded.name)
        state = getattr(getattr(uploaded, "state", None), "name", None)
        if state == "ACTIVE":
            record = _self_hashed(
                {
                    "schema_version": 1,
                    "video_id": row["video_id"],
                    "delivery_sha256": expected_sha256,
                    "delivery_mime_type": row["delivery_mime_type"],
                    "upload_claim_complete_sha256": upload_claim_hash,
                    "file_name": uploaded.name,
                    "file_uri": uploaded.uri,
                    "file_state": state,
                    "expiration_time": str(
                        getattr(uploaded, "expiration_time", "not_reported")
                    ),
                }
            )
            _write_once_json(record_path, record)
            return record
        if state == "FAILED":
            raise RuntimeError("Gemini file processing failed")
        time.sleep(1)
    raise TimeoutError("Gemini file processing did not become active within 55 seconds")


def fixture_pricing() -> dict[str, object]:
    return {
        "input_usd_per_million_tokens": 2.0,
        "output_and_thinking_usd_per_million_tokens": 12.0,
        "high_context_threshold_tokens": 200_000,
        "high_context_input_usd_per_million_tokens": 4.0,
        "high_context_output_and_thinking_usd_per_million_tokens": 18.0,
    }


def estimate_cost_usd(
    *,
    prompt_tokens: int,
    output_tokens: int,
    thinking_tokens: int,
    pricing: Mapping[str, object],
) -> float:
    high = prompt_tokens >= int(pricing["high_context_threshold_tokens"])
    input_rate = float(
        pricing[
            "high_context_input_usd_per_million_tokens"
            if high
            else "input_usd_per_million_tokens"
        ]
    )
    output_rate = float(
        pricing[
            "high_context_output_and_thinking_usd_per_million_tokens"
            if high
            else "output_and_thinking_usd_per_million_tokens"
        ]
    )
    return (
        prompt_tokens * input_rate + (output_tokens + thinking_tokens) * output_rate
    ) / 1_000_000


def budget_exposure(budget_root: Path) -> float:
    ledger_path = budget_root / "budget-ledger.json"
    if not ledger_path.exists():
        return 0.0
    lock_path = budget_root / "budget.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        ledger = _load_budget_ledger(ledger_path)
        reservations = ledger["reservations"]
        assert isinstance(reservations, Mapping)
        return _exposure(reservations)


def estimate_worst_case_request_cost(
    *,
    prompt: str,
    fps: float,
    duration_seconds: float,
    pricing: Mapping[str, object],
) -> float:
    frame_tokens = math.ceil(duration_seconds * fps) * 258
    text_and_envelope_tokens = math.ceil(len(prompt) / 4) + 2_000
    return estimate_cost_usd(
        prompt_tokens=frame_tokens + text_and_envelope_tokens,
        output_tokens=MAX_OUTPUT_TOKENS,
        thinking_tokens=0,
        pricing=pricing,
    )


def _stage_reservation(
    *,
    budget_root: Path,
    cumulative_limit_usd: float,
    counted_input_tokens: int,
    pricing: Mapping[str, object],
) -> float:
    exposure = budget_exposure(budget_root)
    available = cumulative_limit_usd - exposure
    worst_case = estimate_cost_usd(
        prompt_tokens=counted_input_tokens + 4_000,
        output_tokens=MAX_OUTPUT_TOKENS,
        thinking_tokens=0,
        pricing=pricing,
    )
    if available <= 0.0 or worst_case > available + 1e-12:
        raise RuntimeError(
            "proportional budget gate blocks provider call: "
            f"worst_case={worst_case:.6f}, available={available:.6f}"
        )
    return available


def _validated_locator(
    payload: Mapping[str, object], *, duration_seconds: float
) -> Mapping[str, object]:
    parse_locator(payload, duration_seconds=duration_seconds)
    return payload


def _validated_ledger(
    payload: Mapping[str, object], *, duration_seconds: float, envelope: TimeRange
) -> Mapping[str, object]:
    parse_atomic_ledger(
        payload,
        duration_seconds=duration_seconds,
        envelope=envelope,
    )
    return payload


def _validated_challenge(
    payload: Mapping[str, object], *, ledger: AtomicLedger, duration_seconds: float
) -> Mapping[str, object]:
    parse_challenge(payload, ledger=ledger, duration_seconds=duration_seconds)
    return payload


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"artifact must be an object: {path}")
    return payload


def _verify_self_hash(payload: Mapping[str, object], *, name: str) -> str:
    expected = payload.get("complete_sha256")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    )
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{name} self-hash changed")
    return expected


def _execute_row_stage(
    *,
    client: Any,
    row_root: Path,
    budget_root: Path,
    video_id: str,
    stage: str,
    file_uri: str,
    mime_type: str,
    fps: float,
    time_range: TimeRange | None,
    prompt: str,
    response_schema: Mapping[str, object],
    validate_response: Callable[[Mapping[str, object]], Mapping[str, object]],
    prior_complete_sha256: str | None,
    request_manifest_sha256: str,
    upload_complete_sha256: str,
    delivery_sha256: str,
    pricing: Mapping[str, object],
    cumulative_limit_usd: float,
    source_duration_seconds: float,
) -> dict[str, object]:
    del source_duration_seconds
    video_part = build_video_part(
        file_uri=file_uri,
        mime_type=mime_type,
        fps=fps,
        time_range=time_range,
    )
    contents = [video_part, types.Part(text=prompt)]
    counted = client.models.count_tokens(model=MODEL, contents=contents)
    counted_input_tokens = int(getattr(counted, "total_tokens", 0) or 0)
    if counted_input_tokens <= 0:
        raise RuntimeError("Gemini count_tokens returned no usable input count")
    reserved = _stage_reservation(
        budget_root=budget_root,
        cumulative_limit_usd=cumulative_limit_usd,
        counted_input_tokens=counted_input_tokens,
        pricing=pricing,
    )
    return execute_paid_stage(
        client=client,
        stage_root=row_root / stage,
        stage=stage,
        video_part=video_part,
        prompt=prompt,
        response_schema=response_schema,
        prior_complete_sha256=prior_complete_sha256,
        request_manifest_sha256=request_manifest_sha256,
        upload_complete_sha256=upload_complete_sha256,
        delivery_sha256=delivery_sha256,
        reservation={
            "reservation_id": f"{video_id}-{stage}",
            "reserved_usd": reserved,
        },
        budget_root=budget_root,
        pricing=pricing,
        validate_response=validate_response,
    )


def _additive_envelope(
    *,
    challenge: Any,
    duration_seconds: float,
    padding_seconds: float = 1.5,
) -> TimeRange:
    starts = [item.start_seconds for item in challenge.requested_ranges]
    ends = [item.end_seconds for item in challenge.requested_ranges]
    starts.extend(item.start_seconds for item in challenge.missed_instances)
    ends.extend(item.end_seconds for item in challenge.missed_instances)
    if not starts:
        raise ValueError("additive envelope requires a challenger request")
    return TimeRange(
        max(0.0, min(starts) - padding_seconds),
        min(duration_seconds, max(ends) + padding_seconds),
    )


def run_row(
    *,
    client: Any,
    row: Mapping[str, object],
    file_uri: str,
    upload_complete_sha256: str,
    row_root: Path,
    budget_root: Path,
    request_manifest_sha256: str,
    pricing: Mapping[str, object],
    cumulative_limit_usd: float,
) -> dict[str, object]:
    validate_label_blind(row)
    video_id = str(row["video_id"])
    duration = float(row["duration_seconds"])
    mime_type = str(row["delivery_mime_type"])
    upload = _load_mapping(row_root / "upload.json")
    upload_hash = _verify_self_hash(upload, name=f"row {video_id} upload")
    if upload_hash != upload_complete_sha256:
        raise ValueError("upload record binding changed")
    upload_claim = _load_mapping(row_root / "upload-claim.json")
    upload_claim_hash = _verify_self_hash(
        upload_claim, name=f"row {video_id} upload claim"
    )
    if upload.get("upload_claim_complete_sha256") != upload_claim_hash:
        raise ValueError("upload claim binding changed")
    if upload.get("file_uri") != file_uri:
        raise ValueError("uploaded media binding changed")
    if upload.get("video_id") != video_id:
        raise ValueError("upload row identity changed")
    if upload.get("delivery_sha256") != row.get("delivery_sha256"):
        raise ValueError("upload delivery binding changed")
    if upload.get("delivery_mime_type") != mime_type:
        raise ValueError("upload MIME binding changed")

    correction_path = row_root / "locate-correction-v2" / "complete.json"
    if correction_path.exists():
        invalid_locate = _load_mapping(row_root / "locate" / "complete.json")
        invalid_hash = _verify_self_hash(invalid_locate, name="invalid locator")
        locate_complete = _load_mapping(correction_path)
        correction_hash = _verify_self_hash(
            locate_complete, name="locator alias correction"
        )
        if locate_complete.get("prior_complete_sha256") != invalid_hash:
            raise ValueError("locator correction prior binding changed")
        locator_payload = _load_mapping(
            row_root / "locate-correction-v2" / "validated.json"
        )
        initial_stage_completes = {
            "locate": invalid_hash,
            "locate-correction-v2": correction_hash,
        }
    else:
        locator_prompt = build_locator_prompt(row)
        locate_complete = _execute_row_stage(
            client=client,
            row_root=row_root,
            budget_root=budget_root,
            video_id=video_id,
            stage="locate",
            file_uri=file_uri,
            mime_type=mime_type,
            fps=LOCATOR_FPS,
            time_range=None,
            prompt=locator_prompt,
            response_schema=LOCATOR_RESPONSE_SCHEMA,
            validate_response=lambda payload: _validated_locator(
                payload, duration_seconds=duration
            ),
            prior_complete_sha256=None,
            request_manifest_sha256=request_manifest_sha256,
            upload_complete_sha256=upload_complete_sha256,
            delivery_sha256=str(row["delivery_sha256"]),
            pricing=pricing,
            cumulative_limit_usd=cumulative_limit_usd,
            source_duration_seconds=duration,
        )
        locator_payload = _load_mapping(row_root / "locate" / "validated.json")
        initial_stage_completes = {"locate": str(locate_complete["complete_sha256"])}
    locator = parse_locator(locator_payload, duration_seconds=duration)
    envelope = verification_envelope(locator, duration)

    verifier_prompt = build_verifier_prompt(row, locator, envelope)
    verify_complete = _execute_row_stage(
        client=client,
        row_root=row_root,
        budget_root=budget_root,
        video_id=video_id,
        stage="verify",
        file_uri=file_uri,
        mime_type=mime_type,
        fps=VERIFY_FPS,
        time_range=envelope,
        prompt=verifier_prompt,
        response_schema=LEDGER_RESPONSE_SCHEMA,
        validate_response=lambda payload: _validated_ledger(
            payload,
            duration_seconds=duration,
            envelope=envelope,
        ),
        prior_complete_sha256=str(locate_complete["complete_sha256"]),
        request_manifest_sha256=request_manifest_sha256,
        upload_complete_sha256=upload_complete_sha256,
        delivery_sha256=str(row["delivery_sha256"]),
        pricing=pricing,
        cumulative_limit_usd=cumulative_limit_usd,
        source_duration_seconds=duration,
    )
    ledger_payload = _load_mapping(row_root / "verify" / "validated.json")
    ledger = parse_atomic_ledger(
        ledger_payload,
        duration_seconds=duration,
        envelope=envelope,
    )

    challenger_prompt = build_challenger_prompt(row, locator, ledger)
    challenge_complete = _execute_row_stage(
        client=client,
        row_root=row_root,
        budget_root=budget_root,
        video_id=video_id,
        stage="challenge",
        file_uri=file_uri,
        mime_type=mime_type,
        fps=VERIFY_FPS,
        time_range=None,
        prompt=challenger_prompt,
        response_schema=CHALLENGE_RESPONSE_SCHEMA,
        validate_response=lambda payload: _validated_challenge(
            payload,
            ledger=ledger,
            duration_seconds=duration,
        ),
        prior_complete_sha256=str(verify_complete["complete_sha256"]),
        request_manifest_sha256=request_manifest_sha256,
        upload_complete_sha256=upload_complete_sha256,
        delivery_sha256=str(row["delivery_sha256"]),
        pricing=pricing,
        cumulative_limit_usd=cumulative_limit_usd,
        source_duration_seconds=duration,
    )
    challenge_payload = _load_mapping(row_root / "challenge" / "validated.json")
    challenge = parse_challenge(
        challenge_payload,
        ledger=ledger,
        duration_seconds=duration,
    )

    additive: AtomicLedger | None = None
    stage_completes = {
        **initial_stage_completes,
        "verify": str(verify_complete["complete_sha256"]),
        "challenge": str(challenge_complete["complete_sha256"]),
    }
    if challenge.requested_ranges or challenge.missed_instances:
        additive_range = _additive_envelope(
            challenge=challenge,
            duration_seconds=duration,
        )
        additive_prompt = (
            build_verifier_prompt(row, locator, additive_range)
            + "\nVerify only challenger-proposed new instances. Reuse no existing instance IDs."
        )
        additive_complete = _execute_row_stage(
            client=client,
            row_root=row_root,
            budget_root=budget_root,
            video_id=video_id,
            stage="additional-verify",
            file_uri=file_uri,
            mime_type=mime_type,
            fps=VERIFY_FPS,
            time_range=additive_range,
            prompt=additive_prompt,
            response_schema=LEDGER_RESPONSE_SCHEMA,
            validate_response=lambda payload: _validated_ledger(
                payload,
                duration_seconds=duration,
                envelope=additive_range,
            ),
            prior_complete_sha256=str(challenge_complete["complete_sha256"]),
            request_manifest_sha256=request_manifest_sha256,
            upload_complete_sha256=upload_complete_sha256,
            delivery_sha256=str(row["delivery_sha256"]),
            pricing=pricing,
            cumulative_limit_usd=cumulative_limit_usd,
            source_duration_seconds=duration,
        )
        additive_payload = _load_mapping(
            row_root / "additional-verify" / "validated.json"
        )
        additive = parse_atomic_ledger(
            additive_payload,
            duration_seconds=duration,
            envelope=additive_range,
        )
        stage_completes["additional-verify"] = str(additive_complete["complete_sha256"])

    result = _self_hashed(asdict(reduce_recovery_result(ledger, challenge, additive)))
    _write_once_json(row_root / "result.json", result)
    total_cost = sum(
        float(_load_mapping(row_root / stage / "complete.json")["estimated_cost_usd"])
        for stage in stage_completes
    )
    complete = _self_hashed(
        {
            "schema_version": 1,
            "outcome": "success",
            "video_id": video_id,
            "request_manifest_sha256": request_manifest_sha256,
            "upload_complete_sha256": upload_complete_sha256,
            "delivery_sha256": row["delivery_sha256"],
            "stage_complete_sha256": dict(sorted(stage_completes.items())),
            "result_sha256": _file_sha256(row_root / "result.json"),
            "estimated_cost_usd": total_cost,
        }
    )
    _write_once_json(row_root / "complete.json", complete)
    return complete


def resume_row_after_locator_correction(**kwargs: Any) -> dict[str, object]:
    row_root = kwargs.get("row_root")
    if (
        not isinstance(row_root, Path)
        or not (row_root / "locate-correction-v2" / "complete.json").is_file()
    ):
        raise ValueError("locator alias correction is unavailable")
    return run_row(**kwargs)


def freeze_predictions(
    *, root: Path, manifest: Mapping[str, object], row_limit: int | None = None
) -> dict[str, object]:
    rows_payload = manifest.get("rows")
    if not isinstance(rows_payload, list):
        raise TypeError("request manifest rows must be an array")
    if row_limit is None:
        selected_rows = rows_payload
    else:
        if not 1 <= row_limit <= len(rows_payload):
            raise ValueError("row_limit must select at least one manifest row")
        selected_rows = rows_payload[:row_limit]
    frozen_rows: list[dict[str, object]] = []
    for row in selected_rows:
        if not isinstance(row, Mapping):
            raise TypeError("request row must be an object")
        video_id = str(row["video_id"])
        row_root = root / "rows" / video_id
        complete = _load_mapping(row_root / "complete.json")
        complete_hash = _verify_self_hash(complete, name=f"row {video_id} complete")
        if complete.get("outcome") != "success":
            raise ValueError(f"row {video_id} did not succeed")
        if complete.get("video_id") != video_id:
            raise ValueError(f"row {video_id} identity changed")
        if complete.get("request_manifest_sha256") != manifest.get("complete_sha256"):
            raise ValueError(f"row {video_id} manifest binding changed")
        upload = _load_mapping(row_root / "upload.json")
        upload_hash = _verify_self_hash(upload, name=f"row {video_id} upload")
        upload_claim = _load_mapping(row_root / "upload-claim.json")
        upload_claim_hash = _verify_self_hash(
            upload_claim, name=f"row {video_id} upload claim"
        )
        if upload.get("upload_claim_complete_sha256") != upload_claim_hash:
            raise ValueError(f"row {video_id} upload claim binding changed")
        if upload_hash != complete.get("upload_complete_sha256"):
            raise ValueError(f"row {video_id} upload binding changed")
        if upload.get("video_id") != video_id:
            raise ValueError(f"row {video_id} upload identity changed")
        if upload.get("delivery_sha256") != row.get("delivery_sha256"):
            raise ValueError(f"row {video_id} upload delivery binding changed")
        if complete.get("delivery_sha256") != row.get("delivery_sha256"):
            raise ValueError(f"row {video_id} delivery binding changed")
        stage_hashes = complete.get("stage_complete_sha256")
        if not isinstance(stage_hashes, Mapping):
            raise TypeError("row stage hashes must be an object")
        mandatory = {"locate", "verify", "challenge"}
        allowed = mandatory | {"locate-correction-v2", "additional-verify"}
        if not mandatory.issubset(stage_hashes):
            raise ValueError(f"row {video_id} mandatory stages are missing")
        if not set(stage_hashes).issubset(allowed):
            raise ValueError(f"row {video_id} has unknown stages")
        prior: str | None = None
        for stage in (
            "locate",
            "locate-correction-v2",
            "verify",
            "challenge",
            "additional-verify",
        ):
            if stage not in stage_hashes:
                continue
            stage_complete = _load_mapping(row_root / stage / "complete.json")
            stage_hash = _verify_self_hash(
                stage_complete, name=f"row {video_id} {stage} complete"
            )
            if stage_hash != stage_hashes[stage]:
                raise ValueError(f"row {video_id} {stage} binding changed")
            if stage_complete.get("stage") != stage:
                raise ValueError(f"row {video_id} {stage} identity changed")
            if stage_complete.get("prior_complete_sha256") != prior:
                raise ValueError(f"row {video_id} {stage} prior binding changed")
            corrected_locator = (
                stage == "locate" and "locate-correction-v2" in stage_hashes
            )
            expected_outcome = "invalid_response" if corrected_locator else "success"
            if stage_complete.get("outcome") != expected_outcome:
                raise ValueError(f"row {video_id} {stage} outcome changed")
            if stage == "locate-correction-v2":
                referenced = {
                    "invalid_parsed_sha256": row_root / "locate" / "parsed.json",
                    "invalid_raw_response_sha256": row_root
                    / "locate"
                    / "raw-response.txt",
                    "validated_sha256": row_root / stage / "validated.json",
                }
            else:
                request = _load_mapping(row_root / stage / "request.json")
                request_hash = _verify_self_hash(
                    request, name=f"row {video_id} {stage} request"
                )
                if request_hash != stage_complete.get("request_sha256"):
                    raise ValueError(f"row {video_id} {stage} request binding changed")
                if request.get("stage") != stage:
                    raise ValueError(f"row {video_id} {stage} request identity changed")
                if request.get("prior_complete_sha256") != prior:
                    raise ValueError(
                        f"row {video_id} {stage} request prior binding changed"
                    )
                if request.get("request_manifest_sha256") != manifest.get(
                    "complete_sha256"
                ):
                    raise ValueError(f"row {video_id} {stage} manifest binding changed")
                if request.get("upload_complete_sha256") != upload_hash:
                    raise ValueError(f"row {video_id} {stage} upload binding changed")
                if request.get("delivery_sha256") != row.get("delivery_sha256"):
                    raise ValueError(f"row {video_id} {stage} delivery binding changed")
                if request.get("file_uri") != upload.get("file_uri"):
                    raise ValueError(
                        f"row {video_id} {stage} uploaded media binding changed"
                    )
                if request.get("mime_type") != row.get("delivery_mime_type"):
                    raise ValueError(f"row {video_id} {stage} MIME binding changed")
                expected_schema = {
                    "locate": LOCATOR_RESPONSE_SCHEMA,
                    "verify": LEDGER_RESPONSE_SCHEMA,
                    "challenge": CHALLENGE_RESPONSE_SCHEMA,
                    "additional-verify": LEDGER_RESPONSE_SCHEMA,
                }[stage]
                expected_request_fields = {
                    "model": MODEL,
                    "fps": LOCATOR_FPS if stage == "locate" else VERIFY_FPS,
                    "response_schema_sha256": canonical_sha256(expected_schema),
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                    "thinking_level": "LOW",
                    "media_resolution": "HIGH",
                }
                if any(
                    request.get(field) != value
                    for field, value in expected_request_fields.items()
                ):
                    raise ValueError(
                        f"row {video_id} {stage} request configuration changed"
                    )
                clipped = stage in {"verify", "additional-verify"}
                offsets = (
                    request.get("start_offset"),
                    request.get("end_offset"),
                )
                if (
                    clipped
                    and not all(isinstance(value, str) and value for value in offsets)
                ) or (not clipped and offsets != (None, None)):
                    raise ValueError(f"row {video_id} {stage} request offsets changed")
                referenced = {
                    "raw_response_sha256": row_root / stage / "raw-response.txt",
                    "parsed_sha256": row_root / stage / "parsed.json",
                    "validated_sha256": row_root / stage / "validated.json",
                    "provider_metadata_sha256": row_root
                    / stage
                    / "provider-metadata.json",
                }
                claim = _load_mapping(row_root / stage / "paid-call-claim.json")
                claim_hash = _verify_self_hash(
                    claim, name=f"row {video_id} {stage} paid-call claim"
                )
                if claim.get("request_complete_sha256") != request_hash:
                    raise ValueError(
                        f"row {video_id} {stage} paid-call request binding changed"
                    )
                if claim_hash != stage_complete.get("paid_call_claim_sha256"):
                    raise ValueError(
                        f"row {video_id} {stage} paid-call claim binding changed"
                    )
            for field, path in referenced.items():
                expected = stage_complete.get(field)
                if not isinstance(expected, str):
                    raise ValueError(
                        f"row {video_id} {stage} {field} binding is missing"
                    )
                label = field.removesuffix("_sha256").replace("_", " ")
                if not path.is_file() or _file_sha256(path) != expected:
                    raise ValueError(f"row {video_id} {stage} {label} hash changed")
            prior = stage_hash
        result_path = row_root / "result.json"
        if _file_sha256(result_path) != complete.get("result_sha256"):
            raise ValueError(f"row {video_id} result hash changed")
        result = _load_mapping(result_path)
        _verify_self_hash(result, name=f"row {video_id} result")
        frozen_rows.append(
            {
                "video_id": video_id,
                "row_complete_sha256": complete_hash,
                "estimated_cost_usd": complete["estimated_cost_usd"],
                "result": result,
            }
        )
    freeze = _self_hashed(
        {
            "schema_version": 1,
            "condition": manifest["condition"],
            "model": MODEL,
            "request_manifest_sha256": manifest["complete_sha256"],
            "completion": {
                "completed": len(frozen_rows),
                "total": len(selected_rows),
            },
            "manifest_row_count": len(rows_payload),
            "rows": frozen_rows,
            "total_estimated_cost_usd": sum(
                float(row["estimated_cost_usd"]) for row in frozen_rows
            ),
        }
    )
    validate_label_blind(freeze)
    _write_once_json(root / "predictions-freeze.json", freeze)
    return freeze


def _load_self_hashed(path: Path, *, name: str) -> dict[str, object]:
    payload = _load_mapping(path)
    _verify_self_hash(payload, name=name)
    return payload


def apply_locator_alias_correction(
    *, row_root: Path, duration_seconds: float
) -> dict[str, object]:
    locate_root = row_root / "locate"
    terminal = _load_mapping(locate_root / "complete.json")
    terminal_hash = _verify_self_hash(terminal, name="invalid locator terminal")
    if terminal.get("outcome") != "invalid_response" or not str(
        terminal.get("error_message", "")
    ).startswith("unsupported count_unit:"):
        raise ValueError("locator correction applies only to count-unit alias failures")
    raw_path = locate_root / "raw-response.txt"
    if _file_sha256(raw_path) != terminal.get("raw_response_sha256"):
        raise ValueError("invalid locator raw-response binding changed")
    parsed_path = locate_root / "parsed.json"
    parsed = _load_mapping(parsed_path)
    locator = parse_locator(parsed, duration_seconds=duration_seconds)
    corrected = {**parsed, "count_unit": locator.count_unit}
    validate_label_blind(corrected)
    correction_root = row_root / "locate-correction-v2"
    _write_once_json(correction_root / "validated.json", corrected)
    complete = _self_hashed(
        {
            "schema_version": 1,
            "stage": "locate-correction-v2",
            "outcome": "success",
            "correction_reason": "label_independent_count_unit_alias",
            "prior_complete_sha256": terminal_hash,
            "invalid_raw_response_sha256": _file_sha256(raw_path),
            "invalid_parsed_sha256": _file_sha256(parsed_path),
            "validated_sha256": _file_sha256(correction_root / "validated.json"),
            "runner_sha256": _file_sha256(Path(__file__)),
            "estimated_cost_usd": 0.0,
        }
    )
    _write_once_json(correction_root / "complete.json", complete)
    return complete


def prepare_experiment(
    *,
    root: Path,
    source_manifest_path: Path,
    delivery_root: Path,
    condition: str = "gemini-native-counting-recovery-v1",
) -> dict[str, object]:
    source_manifest = _load_self_hashed(
        source_manifest_path, name="source counting request manifest"
    )
    source_rows = source_manifest.get("rows")
    if not isinstance(source_rows, list) or not all(
        isinstance(row, Mapping) for row in source_rows
    ):
        raise TypeError("source counting request rows must be objects")
    manifest = prepare_request_manifest(
        source_rows=source_rows,
        ordered_video_ids=ORDERED_VIDEO_IDS,
        delivery_root=delivery_root,
        condition=condition,
    )
    provenance = _self_hashed(
        {
            "schema_version": 1,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": source_manifest["complete_sha256"],
            "request_manifest_sha256": manifest["complete_sha256"],
            "delivery_root": str(delivery_root),
            "runner_sha256": _file_sha256(Path(__file__)),
        }
    )
    _write_once_json(root / "requests.label-blind.json", manifest)
    _write_once_json(root / "preparation-provenance.json", provenance)
    return manifest


def run_canary(
    *, root: Path, pricing_path: Path, max_rows: int = len(ORDERED_VIDEO_IDS)
) -> dict[str, object]:
    manifest = _load_self_hashed(
        root / "requests.label-blind.json", name="recovery request manifest"
    )
    validate_label_blind(manifest)
    pricing = _load_self_hashed(pricing_path, name="Gemini pricing")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ORDERED_VIDEO_IDS):
        raise ValueError("recovery request manifest must contain exactly three rows")
    if not 1 <= max_rows <= len(ORDERED_VIDEO_IDS):
        raise ValueError("max_rows must be between one and three")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is unavailable")
    client = genai.Client(api_key=api_key)
    ordered = list(zip(rows, ORDERED_VIDEO_IDS, CUMULATIVE_LIMITS_USD, strict=True))
    for row, video_id, limit in ordered[:max_rows]:
        if not isinstance(row, Mapping) or row.get("video_id") != video_id:
            raise ValueError("recovery row ordering changed")
        row_root = root / "rows" / video_id
        upload = ensure_upload(client=client, row=row, row_root=row_root)
        run_row(
            client=client,
            row=row,
            file_uri=str(upload["file_uri"]),
            upload_complete_sha256=str(upload["complete_sha256"]),
            row_root=row_root,
            budget_root=root / "budget",
            request_manifest_sha256=str(manifest["complete_sha256"]),
            pricing=pricing,
            cumulative_limit_usd=limit,
        )
        exposure = budget_exposure(root / "budget")
        if exposure > limit + 1e-12:
            raise RuntimeError(
                f"post-row proportional cost gate failed: {exposure:.6f} > {limit:.6f}"
            )
    return freeze_predictions(root=root, manifest=manifest, row_limit=max_rows)


def status_payload(root: Path) -> dict[str, object]:
    manifest_path = root / "requests.label-blind.json"
    manifest = _load_self_hashed(manifest_path, name="recovery request manifest")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise TypeError("recovery request rows must be an array")
    row_status = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("recovery request row must be an object")
        video_id = str(row["video_id"])
        row_root = root / "rows" / video_id
        stages = {
            stage: (row_root / stage / "complete.json").exists()
            for stage in ("locate", "verify", "challenge", "additional-verify")
        }
        row_status.append(
            {
                "video_id": video_id,
                "stages": stages,
                "row_complete": (row_root / "complete.json").exists(),
            }
        )
    return {
        "root": str(root),
        "budget_exposure_usd": budget_exposure(root / "budget"),
        "predictions_frozen": (root / "predictions-freeze.json").exists(),
        "rows": row_status,
    }


def _request_config(
    response_schema: Mapping[str, object],
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=TEMPERATURE,
        seed=SEED,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_json_schema=dict(response_schema),
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
            include_thoughts=False,
        ),
    )


def _usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    return (
        int(getattr(usage, "prompt_token_count", 0) or 0),
        int(getattr(usage, "candidates_token_count", 0) or 0),
        int(getattr(usage, "thoughts_token_count", 0) or 0),
    )


def execute_paid_stage(
    *,
    client: Any,
    stage_root: Path,
    stage: str,
    video_part: types.Part,
    prompt: str,
    response_schema: Mapping[str, object],
    prior_complete_sha256: str | None,
    request_manifest_sha256: str,
    upload_complete_sha256: str,
    delivery_sha256: str,
    reservation: Mapping[str, object],
    budget_root: Path,
    pricing: Mapping[str, object],
    validate_response: Callable[[Mapping[str, object]], Mapping[str, object]]
    | None = None,
) -> dict[str, object]:
    request = _self_hashed(
        {
            "schema_version": 1,
            "stage": stage,
            "model": MODEL,
            "request_manifest_sha256": request_manifest_sha256,
            "upload_complete_sha256": upload_complete_sha256,
            "delivery_sha256": delivery_sha256,
            "file_uri": video_part.file_data.file_uri,
            "mime_type": video_part.file_data.mime_type,
            "prior_complete_sha256": prior_complete_sha256,
            "fps": video_part.video_metadata.fps,
            "start_offset": video_part.video_metadata.start_offset,
            "end_offset": video_part.video_metadata.end_offset,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_schema_sha256": canonical_sha256(response_schema),
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "thinking_level": "LOW",
            "media_resolution": "HIGH",
        }
    )
    validate_label_blind(request)
    claim_path = reserve_paid_stage(stage_root, request)
    paid_call_claim_sha256 = _verify_self_hash(
        _load_mapping(claim_path), name=f"{stage} paid-call claim"
    )
    reservation_id = str(reservation["reservation_id"])
    reserved_usd = float(reservation["reserved_usd"])
    reserve_budget(
        budget_root=budget_root,
        reservation_id=reservation_id,
        reserved_usd=reserved_usd,
    )
    _write_once_json(stage_root / "request.json", request)
    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[video_part, types.Part(text=prompt)],
            config=_request_config(response_schema),
        )
    except Exception as error:
        terminal = _self_hashed(
            {
                "schema_version": 1,
                "stage": stage,
                "outcome": "provider_error",
                "attempted_provider_calls": 1,
                "request_sha256": request["complete_sha256"],
                "paid_call_claim_sha256": paid_call_claim_sha256,
                "prior_complete_sha256": prior_complete_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "estimated_cost_usd": reserved_usd,
            }
        )
        _write_once_json(stage_root / "complete.json", terminal)
        settle_budget(
            budget_root=budget_root,
            reservation_id=reservation_id,
            observed_usd=reserved_usd,
        )
        raise
    raw = response.text or ""
    _write_once_text(stage_root / "raw-response.txt", raw)
    prompt_tokens, output_tokens, thinking_tokens = _usage(response)
    observed_cost = estimate_cost_usd(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        pricing=pricing,
    )
    if observed_cost > reserved_usd + 1e-12:
        raise RuntimeError(
            f"observed provider cost {observed_cost:.6f} exceeds reservation "
            f"{reserved_usd:.6f}"
        )
    provider = _self_hashed(
        {
            "schema_version": 1,
            "model_version": getattr(response, "model_version", None),
            "response_id": getattr(response, "response_id", None),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "estimated_cost_usd": observed_cost,
        }
    )
    _write_once_json(stage_root / "provider-metadata.json", provider)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise TypeError("provider response must be a JSON object")
        _write_once_json(stage_root / "parsed.json", parsed)
        validated = validate_response(parsed) if validate_response else parsed
        if not isinstance(validated, Mapping):
            raise TypeError("validated response must be an object")
        _write_once_json(stage_root / "validated.json", validated)
    except Exception as error:
        terminal = _self_hashed(
            {
                "schema_version": 1,
                "stage": stage,
                "outcome": "invalid_response",
                "attempted_provider_calls": 1,
                "request_sha256": request["complete_sha256"],
                "paid_call_claim_sha256": paid_call_claim_sha256,
                "raw_response_sha256": _file_sha256(stage_root / "raw-response.txt"),
                "provider_metadata_sha256": _file_sha256(
                    stage_root / "provider-metadata.json"
                ),
                "prior_complete_sha256": prior_complete_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "estimated_cost_usd": observed_cost,
            }
        )
        _write_once_json(stage_root / "complete.json", terminal)
        settle_budget(
            budget_root=budget_root,
            reservation_id=reservation_id,
            observed_usd=observed_cost,
        )
        raise
    terminal = _self_hashed(
        {
            "schema_version": 1,
            "stage": stage,
            "outcome": "success",
            "attempted_provider_calls": 1,
            "request_sha256": request["complete_sha256"],
            "paid_call_claim_sha256": paid_call_claim_sha256,
            "raw_response_sha256": _file_sha256(stage_root / "raw-response.txt"),
            "parsed_sha256": _file_sha256(stage_root / "parsed.json"),
            "validated_sha256": _file_sha256(stage_root / "validated.json"),
            "provider_metadata_sha256": _file_sha256(
                stage_root / "provider-metadata.json"
            ),
            "prior_complete_sha256": prior_complete_sha256,
            "elapsed_seconds": time.monotonic() - started,
            "estimated_cost_usd": observed_cost,
        }
    )
    _write_once_json(stage_root / "complete.json", terminal)
    settle_budget(
        budget_root=budget_root,
        reservation_id=reservation_id,
        observed_usd=observed_cost,
    )
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", required=True, type=Path)
    prepare_parser.add_argument("--source-manifest", required=True, type=Path)
    prepare_parser.add_argument("--delivery-root", required=True, type=Path)
    prepare_parser.add_argument(
        "--condition", default="gemini-native-counting-recovery-v1"
    )
    run_parser = subparsers.add_parser("run-canary")
    run_parser.add_argument("--root", required=True, type=Path)
    run_parser.add_argument("--pricing", required=True, type=Path)
    run_parser.add_argument(
        "--max-rows", type=int, choices=range(1, 4), default=len(ORDERED_VIDEO_IDS)
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_experiment(
            root=args.root,
            source_manifest_path=args.source_manifest,
            delivery_root=args.delivery_root,
            condition=args.condition,
        )
    elif args.command == "run-canary":
        result = run_canary(
            root=args.root, pricing_path=args.pricing, max_rows=args.max_rows
        )
    else:
        result = status_payload(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
