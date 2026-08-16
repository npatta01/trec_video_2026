#!/usr/bin/env python3
"""Run the bounded direct-Google multi-pass counting canary."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026.vqa_bounded_multipass_counting import (
    CandidateView,
    CountSpec,
    UnsupportedCountSpec,
    build_count_spec_prompt,
    build_coverage_prompt,
    build_evidence_prompt,
    count_result_to_dict,
    count_spec_to_dict,
    coverage_review_to_dict,
    evidence_ledger_to_dict,
    merge_refinement,
    parse_count_spec,
    parse_coverage_review,
    parse_evidence,
    reduce_count,
    schedule_evidence_views,
)
from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256


MODEL = "gemini-3.1-pro-preview"
TEMPERATURE = 0.0
SEED = 7
MAX_OUTPUT_TOKENS = 8192
THINKING_LEVEL = "LOW"
GLOBAL_CEILING_USD = 30.0
SMOKE_CEILING_USD = 3.0
PRE_REFINEMENT_CEILING_USD = 15.0
IMPLEMENTATION_SOURCE_PATHS = (
    "pyproject.toml",
    "scripts/run_bounded_multipass_counting.py",
    "src/trec_video_2026/vqa_bounded_multipass_counting.py",
    "src/trec_video_2026/vqa_counting_evidence.py",
    "src/trec_video_2026/vqa_gate.py",
    "src/trec_video_2026/vqa_submission.py",
    "uv.lock",
)


def _render_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _self_hashed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("complete_sha256", None)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def write_self_hashed_once(
    path: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    normalized = _self_hashed(payload)
    rendered = _render_json(normalized)
    if path.exists():
        if path.read_text() != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return normalized


def reserve_claim(
    stage_root: Path, identity: Mapping[str, object]
) -> dict[str, object]:
    stage_root.mkdir(parents=True, exist_ok=True)
    claim_path = stage_root / "paid-call-claim.json"
    claim = _self_hashed(
        {
            "schema_version": "bounded-multipass-paid-claim-v1",
            "identity": dict(identity),
            "owner_pid": os.getpid(),
        }
    )
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_render_json(claim))
        handle.flush()
        os.fsync(handle.fileno())
    return claim


@contextmanager
def _locked_budget(budget_root: Path):
    budget_root.mkdir(parents=True, exist_ok=True)
    lock_path = budget_root / "budget.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _budget_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "bounded-multipass-budget-v1",
            "reservations": {},
        }
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("reservations"), dict):
        raise ValueError("budget ledger is invalid")
    return value


def _budget_exposure(ledger: Mapping[str, object]) -> float:
    reservations = ledger.get("reservations")
    if not isinstance(reservations, Mapping):
        raise ValueError("budget reservations are invalid")
    total = 0.0
    for value in reservations.values():
        if not isinstance(value, Mapping):
            raise ValueError("budget reservation is invalid")
        observed = value.get("observed_usd")
        reserved = value.get("reserved_usd")
        total += float(reserved if observed is None else observed)
    return total


def _replace_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_render_json(payload))
    os.replace(temporary, path)


def reserve_budget(
    *,
    budget_root: Path,
    reservation_id: str,
    reserved_usd: float,
    hard_ceiling_usd: float,
) -> None:
    if reserved_usd <= 0 or hard_ceiling_usd <= 0:
        raise ValueError("budget values must be positive")
    path = budget_root / "budget-ledger.json"
    with _locked_budget(budget_root):
        ledger = _budget_ledger(path)
        reservations = ledger["reservations"]
        if reservation_id in reservations:
            raise RuntimeError(f"duplicate budget reservation: {reservation_id}")
        if _budget_exposure(ledger) + reserved_usd > hard_ceiling_usd + 1e-12:
            raise RuntimeError("budget ceiling would be exceeded")
        reservations[reservation_id] = {
            "reserved_usd": float(reserved_usd),
            "observed_usd": None,
        }
        _replace_json(path, ledger)


def settle_budget(
    *, budget_root: Path, reservation_id: str, observed_usd: float
) -> None:
    if observed_usd < 0:
        raise ValueError("observed cost must be nonnegative")
    path = budget_root / "budget-ledger.json"
    with _locked_budget(budget_root):
        ledger = _budget_ledger(path)
        reservation = ledger["reservations"].get(reservation_id)
        if not isinstance(reservation, dict):
            raise RuntimeError(f"unknown budget reservation: {reservation_id}")
        if reservation.get("observed_usd") is not None:
            raise RuntimeError(f"budget reservation already settled: {reservation_id}")
        reservation["observed_usd"] = float(observed_usd)
        _replace_json(path, ledger)


def build_temporal_parts(
    *,
    file_uri: str,
    mime_type: str,
    views: Sequence[CandidateView],
    fps: float,
) -> tuple[types.Part, ...]:
    parts = []
    for view in views:
        if (
            view.kind != "window"
            or view.start_seconds is None
            or view.end_seconds is None
        ):
            raise ValueError("temporal evidence views must be bounded windows")
        parts.append(
            types.Part(
                file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
                video_metadata=types.VideoMetadata(
                    fps=fps,
                    start_offset=f"{view.start_seconds:.3f}s",
                    end_offset=f"{view.end_seconds:.3f}s",
                ),
                media_resolution=types.PartMediaResolution(
                    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
                ),
            )
        )
    return tuple(parts)


def build_full_video_part(*, file_uri: str, mime_type: str, fps: float) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=fps),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )


def build_stage_request(
    *,
    stage: str,
    arm: str,
    video_id: str,
    model: str,
    prompt: str,
    response_schema: Mapping[str, object],
    media_binding: Mapping[str, object],
    predecessor_sha256: str | None,
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_sources = {
        relative: _file_sha256(repository / relative)
        for relative in IMPLEMENTATION_SOURCE_PATHS
    }
    request = _self_hashed(
        {
            "schema_version": "bounded-multipass-stage-request-v1",
            "stage": stage,
            "arm": arm,
            "video_id": video_id,
            "model": model,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response_schema_sha256": canonical_sha256(response_schema),
            "media_binding": dict(media_binding),
            "predecessor_sha256": predecessor_sha256,
            "source_revision": source_revision,
            "implementation_sources": implementation_sources,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "thinking_level": THINKING_LEVEL,
            "media_resolution": "HIGH",
        }
    )
    validate_label_blind(request)
    return request


def count_spec_schema() -> dict[str, object]:
    window = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["window"]},
            "start_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
        },
        "required": ["view_id", "kind", "start_seconds", "end_seconds"],
    }
    frame = {
        "type": "object",
        "properties": {
            "view_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["shot_frame"]},
            "timestamp_seconds": {"type": "number"},
        },
        "required": ["view_id", "kind", "timestamp_seconds"],
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["bounded-multipass-count-spec-v1"],
            },
            "question_sha256": {"type": "string"},
            "target": {"type": "string"},
            "unit": {"type": "string"},
            "route": {"type": "string", "enum": ["temporal", "static"]},
            "scope": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                },
                "required": ["start_seconds", "end_seconds"],
            },
            "inclusions": {"type": "array", "items": {"type": "string"}},
            "exclusions": {"type": "array", "items": {"type": "string"}},
            "identity_policy": {"type": "string"},
            "candidate_views": {
                "type": "array",
                "items": {"anyOf": [window, frame]},
                "minItems": 1,
                "maxItems": 6,
            },
            "uncertainty": {"type": "string"},
        },
        "required": [
            "schema_version",
            "question_sha256",
            "target",
            "unit",
            "route",
            "scope",
            "inclusions",
            "exclusions",
            "identity_policy",
            "candidate_views",
            "uncertainty",
        ],
    }


def evidence_schema(*, route: str) -> dict[str, object]:
    common: dict[str, object] = {
        "record_id": {"type": "string"},
        "view_ids": {"type": "array", "items": {"type": "string"}},
        "target_supported": {"type": "boolean"},
        "ambiguous": {"type": "boolean"},
        "descriptor": {"type": "string"},
    }
    if route == "temporal":
        properties = {
            **common,
            "start_seconds": {"type": "number"},
            "peak_seconds": {"type": "number"},
            "end_seconds": {"type": "number"},
        }
        required = [*common, "start_seconds", "peak_seconds", "end_seconds"]
    elif route == "static":
        properties = {
            **common,
            "shot_id": {"type": "string"},
            "timestamp_seconds": {"type": "number"},
            "bbox_norm": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        }
        required = [*common, "shot_id", "timestamp_seconds", "bbox_norm"]
    else:
        raise ValueError("route must be temporal or static")
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["bounded-multipass-evidence-v1"],
            },
            "reviewed_view_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
        "required": ["schema_version", "reviewed_view_ids", "records"],
    }


def coverage_schema() -> dict[str, object]:
    refinement = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["window", "shot_frame", "crop"]},
            "start_seconds": {"type": ["number", "null"]},
            "end_seconds": {"type": ["number", "null"]},
            "view_id": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": [
            "kind",
            "start_seconds",
            "end_seconds",
            "view_id",
            "reason",
        ],
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": ["bounded-multipass-coverage-v1"],
            },
            "complete": {"type": "boolean"},
            "reviewed_view_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "accepted_record_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rejected_record_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_view_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "disputes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["record_id", "reason"],
                },
            },
            "refinement_request": {"anyOf": [refinement, {"type": "null"}]},
            "notes": {"type": "string"},
        },
        "required": [
            "schema_version",
            "complete",
            "reviewed_view_ids",
            "accepted_record_ids",
            "rejected_record_ids",
            "missing_view_ids",
            "disputes",
            "refinement_request",
            "notes",
        ],
    }


def request_config(
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


def observed_cost_usd(
    *, prompt_tokens: int, output_tokens: int, thought_tokens: int
) -> float:
    return (
        prompt_tokens * 2.0 / 1_000_000
        + (output_tokens + thought_tokens) * 12.0 / 1_000_000
    )


def worst_case_call_cost_usd(*, duration_seconds: float, fps: float) -> float:
    video_tokens = max(1, int(math.ceil(duration_seconds * fps))) * 258
    projected = observed_cost_usd(
        prompt_tokens=video_tokens + 4096,
        output_tokens=MAX_OUTPUT_TOKENS,
        thought_tokens=0,
    )
    return max(0.30, projected * 1.25)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _verify_self_hash(payload: Mapping[str, object], *, name: str) -> str:
    claimed = payload.get("complete_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{name} lacks complete_sha256")
    value = dict(payload)
    value.pop("complete_sha256")
    if canonical_sha256(value) != claimed:
        raise ValueError(f"{name} self-hash mismatch")
    return claimed


def ensure_upload(
    *, client: Any, row: Mapping[str, object], row_root: Path
) -> dict[str, object]:
    upload_path = row_root / "upload.json"
    if upload_path.exists():
        upload = _load_json(upload_path)
        _verify_self_hash(upload, name="upload")
        if upload.get("delivery_sha256") != row.get("delivery_sha256"):
            raise ValueError("upload does not bind the current delivery")
        return upload
    claim_path = row_root / "upload-claim.json"
    claim = _self_hashed(
        {
            "schema_version": "bounded-multipass-upload-claim-v1",
            "video_id": row["video_id"],
            "delivery_path": row["delivery_path"],
            "delivery_sha256": row["delivery_sha256"],
            "owner_pid": os.getpid(),
        }
    )
    row_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("upload already claimed by another process") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_render_json(claim))
        handle.flush()
        os.fsync(handle.fileno())
    uploaded = client.files.upload(file=Path(str(row["delivery_path"])))
    for _attempt in range(180):
        state = getattr(getattr(uploaded, "state", None), "name", None)
        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError("Google file processing failed")
        time.sleep(1)
        uploaded = client.files.get(name=uploaded.name)
    else:
        raise TimeoutError("Google file processing timed out")
    record = write_self_hashed_once(
        upload_path,
        {
            "schema_version": "bounded-multipass-upload-v1",
            "video_id": row["video_id"],
            "delivery_path": row["delivery_path"],
            "delivery_sha256": row["delivery_sha256"],
            "delivery_mime_type": row["delivery_mime_type"],
            "upload_claim_complete_sha256": claim["complete_sha256"],
            "file_name": str(uploaded.name),
            "file_uri": str(uploaded.uri),
            "file_state": "ACTIVE",
            "expiration_time": str(getattr(uploaded, "expiration_time", "")),
        },
    )
    return record


def normalize_coverage_payload(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    refinement = result.get("refinement_request")
    if isinstance(refinement, Mapping):
        normalized = dict(refinement)
        kind = normalized.get("kind")
        if kind == "window":
            normalized.pop("view_id", None)
        elif kind in {"shot_frame", "crop"}:
            normalized.pop("start_seconds", None)
            normalized.pop("end_seconds", None)
        result["refinement_request"] = {
            key: value for key, value in normalized.items() if value is not None
        }
    return result


def execute_stage(
    *,
    client: Any,
    stage_root: Path,
    request: Mapping[str, object],
    parts: Sequence[types.Part],
    prompt: str,
    response_schema: Mapping[str, object],
    budget_root: Path,
    reserved_usd: float,
    hard_ceiling_usd: float,
    validator: Any,
) -> dict[str, object]:
    if stage_root.joinpath("complete.json").exists():
        terminal = _load_json(stage_root / "complete.json")
        _verify_self_hash(terminal, name="stage terminal")
        return terminal
    request_hash = _verify_self_hash(request, name="stage request")
    claim = reserve_claim(
        stage_root,
        {"request_sha256": request_hash, "stage": request["stage"]},
    )
    reservation_id = f"{request['arm']}:{request['video_id']}:{request['stage']}"
    reserve_budget(
        budget_root=budget_root,
        reservation_id=reservation_id,
        reserved_usd=reserved_usd,
        hard_ceiling_usd=hard_ceiling_usd,
    )
    write_self_hashed_once(stage_root / "request.json", request)
    write_self_hashed_once(
        stage_root / "response-schema.json",
        {
            "schema_version": "bounded-multipass-response-schema-v1",
            "payload": dict(response_schema),
        },
    )
    started = time.monotonic()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[*parts, types.Part(text=prompt)],
            config=request_config(response_schema),
        )
    except Exception as error:
        settle_budget(
            budget_root=budget_root,
            reservation_id=reservation_id,
            observed_usd=reserved_usd,
        )
        return write_self_hashed_once(
            stage_root / "complete.json",
            {
                "schema_version": "bounded-multipass-stage-complete-v1",
                "stage": request["stage"],
                "outcome": "provider_error",
                "attempted_provider_calls": 1,
                "request_sha256": request_hash,
                "paid_call_claim_sha256": claim["complete_sha256"],
                "error_type": type(error).__name__,
                "error_message": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "observed_cost_usd": reserved_usd,
            },
        )
    raw = str(response.text or "")
    raw_path = stage_root / "raw-response.txt"
    _write_text_once(raw_path, raw)
    prompt_tokens, output_tokens, thought_tokens = _usage(response)
    cost = observed_cost_usd(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        thought_tokens=thought_tokens,
    )
    settle_budget(
        budget_root=budget_root,
        reservation_id=reservation_id,
        observed_usd=cost,
    )
    metadata = write_self_hashed_once(
        stage_root / "provider-metadata.json",
        {
            "schema_version": "bounded-multipass-provider-metadata-v1",
            "model": MODEL,
            "model_version": str(getattr(response, "model_version", "")),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thought_tokens": thought_tokens,
            "observed_cost_usd": cost,
            "elapsed_seconds": time.monotonic() - started,
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "finish_reason": str(
                getattr(
                    (getattr(response, "candidates", None) or [None])[0],
                    "finish_reason",
                    "",
                )
            ),
        },
    )
    if metadata["model_version"] != MODEL:
        return write_self_hashed_once(
            stage_root / "complete.json",
            {
                "schema_version": "bounded-multipass-stage-complete-v1",
                "stage": request["stage"],
                "outcome": "invalid_provider_model",
                "attempted_provider_calls": 1,
                "request_sha256": request_hash,
                "paid_call_claim_sha256": claim["complete_sha256"],
                "provider_metadata_sha256": metadata["complete_sha256"],
                "error_type": "ProviderModelMismatch",
                "error_message": (
                    f"provider response model_version {metadata['model_version']!r} "
                    f"does not equal expected {MODEL!r}"
                ),
                "observed_cost_usd": cost,
            },
        )
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("provider response must be a JSON object")
        normalized = validator(parsed)
    except Exception as error:
        return write_self_hashed_once(
            stage_root / "complete.json",
            {
                "schema_version": "bounded-multipass-stage-complete-v1",
                "stage": request["stage"],
                "outcome": "invalid_response",
                "attempted_provider_calls": 1,
                "request_sha256": request_hash,
                "paid_call_claim_sha256": claim["complete_sha256"],
                "provider_metadata_sha256": metadata["complete_sha256"],
                "error_type": type(error).__name__,
                "error_message": str(error),
                "observed_cost_usd": cost,
            },
        )
    parsed_record = write_self_hashed_once(
        stage_root / "parsed.json",
        {
            "schema_version": "bounded-multipass-parsed-v1",
            "request_sha256": request_hash,
            "payload": parsed,
        },
    )
    validated_record = write_self_hashed_once(
        stage_root / "validated.json",
        {
            "schema_version": "bounded-multipass-validated-v1",
            "request_sha256": request_hash,
            "payload": normalized,
        },
    )
    return write_self_hashed_once(
        stage_root / "complete.json",
        {
            "schema_version": "bounded-multipass-stage-complete-v1",
            "stage": request["stage"],
            "outcome": "complete",
            "attempted_provider_calls": 1,
            "request_sha256": request_hash,
            "paid_call_claim_sha256": claim["complete_sha256"],
            "provider_metadata_sha256": metadata["complete_sha256"],
            "parsed_sha256": parsed_record["complete_sha256"],
            "validated_sha256": validated_record["complete_sha256"],
            "observed_cost_usd": cost,
        },
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _validated_payload(stage_root: Path) -> dict[str, Any]:
    terminal = _load_json(stage_root / "complete.json")
    _verify_self_hash(terminal, name=f"{stage_root.name} terminal")
    if terminal.get("outcome") != "complete":
        raise RuntimeError(f"stage did not complete: {stage_root}")
    validated = _load_json(stage_root / "validated.json")
    validated_hash = _verify_self_hash(validated, name=f"{stage_root.name} validated")
    if validated_hash != terminal.get("validated_sha256"):
        raise ValueError("stage terminal does not bind validated payload")
    payload = validated.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("validated payload must be an object")
    return payload


def extract_static_parts(
    *,
    delivery_path: Path,
    views: Sequence[CandidateView],
    output_root: Path,
) -> tuple[tuple[types.Part, ...], list[dict[str, object]]]:
    parts = []
    bindings = []
    for view in views:
        if view.kind != "shot_frame" or view.timestamp_seconds is None:
            raise ValueError("static evidence views must be timestamped shot frames")
        output_path = output_root / f"{view.view_id}-{view.timestamp_seconds:.3f}.png"
        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.tmp.png"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{view.timestamp_seconds:.6f}",
                    "-i",
                    str(delivery_path),
                    "-frames:v",
                    "1",
                    "-c:v",
                    "png",
                    "-y",
                    str(temporary),
                ],
                check=True,
            )
            os.replace(temporary, output_path)
        data = output_path.read_bytes()
        if not data:
            raise ValueError(f"empty extracted frame: {output_path}")
        parts.append(types.Part.from_bytes(data=data, mime_type="image/png"))
        bindings.append(
            {
                "view_id": view.view_id,
                "timestamp_seconds": view.timestamp_seconds,
                "path": str(output_path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "mime_type": "image/png",
            }
        )
    return tuple(parts), bindings


def _stage_media_binding(
    *,
    upload: Mapping[str, object],
    fps: float | None,
    views: Sequence[CandidateView],
    static_bindings: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "file_uri": upload["file_uri"],
        "mime_type": upload["delivery_mime_type"],
        "delivery_sha256": upload["delivery_sha256"],
        "upload_complete_sha256": upload["complete_sha256"],
        "fps": fps,
        "views": [
            {
                "view_id": view.view_id,
                "kind": view.kind,
                "start_seconds": view.start_seconds,
                "end_seconds": view.end_seconds,
                "timestamp_seconds": view.timestamp_seconds,
            }
            for view in views
        ],
        "static_artifacts": [dict(value) for value in static_bindings],
    }


def _incomplete_row(
    *,
    row_root: Path,
    video_id: str,
    arm: str,
    reason: str,
    stage_hashes: Mapping[str, object],
) -> dict[str, object]:
    return write_self_hashed_once(
        row_root / "complete.json",
        {
            "schema_version": "bounded-multipass-row-complete-v1",
            "video_id": video_id,
            "arm": arm,
            "outcome": "incomplete",
            "reason": reason,
            "stage_complete_sha256": dict(stage_hashes),
            "result_sha256": None,
        },
    )


def run_fixed_row(
    *,
    client: Any,
    row: Mapping[str, object],
    root: Path,
    hard_ceiling_usd: float,
) -> dict[str, object]:
    video_id = str(row["video_id"])
    fixed_root = root / "rows" / video_id / "fixed"
    if fixed_root.joinpath("complete.json").exists():
        terminal = _load_json(fixed_root / "complete.json")
        _verify_self_hash(terminal, name="fixed row terminal")
        return terminal
    upload = ensure_upload(client=client, row=row, row_root=root / "rows" / video_id)
    stage_hashes: dict[str, object] = {}
    duration = float(row["duration_seconds"])
    question = str(row["question"])

    spec_prompt = build_count_spec_prompt(question=question, duration_seconds=duration)
    spec_part = build_full_video_part(
        file_uri=str(upload["file_uri"]),
        mime_type=str(upload["delivery_mime_type"]),
        fps=4.0,
    )
    spec_request = build_stage_request(
        stage="count-spec",
        arm="fixed",
        video_id=video_id,
        model=MODEL,
        prompt=spec_prompt,
        response_schema=count_spec_schema(),
        media_binding=_stage_media_binding(
            upload=upload,
            fps=4.0,
            views=(CandidateView("full", "window", 0.0, duration),),
        ),
        predecessor_sha256=None,
    )
    spec_stage_root = fixed_root / "count-spec"
    _write_text_once(spec_stage_root / "prompt.txt", spec_prompt)

    def validate_spec(payload: Mapping[str, object]) -> dict[str, object]:
        parsed = parse_count_spec(
            payload,
            question=question,
            duration_seconds=duration,
            require_complete_views=False,
        )
        if isinstance(parsed, UnsupportedCountSpec):
            return {"unit": parsed.unit, "unsupported_reason": parsed.reason}
        return count_spec_to_dict(schedule_evidence_views(parsed))

    spec_terminal = execute_stage(
        client=client,
        stage_root=spec_stage_root,
        request=spec_request,
        parts=(spec_part,),
        prompt=spec_prompt,
        response_schema=count_spec_schema(),
        budget_root=root / "budget",
        reserved_usd=worst_case_call_cost_usd(duration_seconds=duration, fps=4.0),
        hard_ceiling_usd=hard_ceiling_usd,
        validator=validate_spec,
    )
    stage_hashes["count_spec"] = spec_terminal["complete_sha256"]
    if spec_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=fixed_root,
            video_id=video_id,
            arm="fixed",
            reason="count_spec_stage_failed",
            stage_hashes=stage_hashes,
        )
    spec_payload = _validated_payload(spec_stage_root)
    if "unsupported_reason" in spec_payload:
        result = write_self_hashed_once(
            fixed_root / "result.json",
            {
                "schema_version": "bounded-multipass-result-v1",
                "video_id": video_id,
                "arm": "fixed",
                "outcome": "abstain",
                "value": None,
                "reasons": [spec_payload["unsupported_reason"]],
                "selected_record_ids": [],
                "count_spec_sha256": spec_terminal["validated_sha256"],
            },
        )
        return write_self_hashed_once(
            fixed_root / "complete.json",
            {
                "schema_version": "bounded-multipass-row-complete-v1",
                "video_id": video_id,
                "arm": "fixed",
                "outcome": "complete",
                "reason": "unsupported_count_spec",
                "stage_complete_sha256": stage_hashes,
                "result_sha256": result["complete_sha256"],
            },
        )
    parsed_spec = parse_count_spec(
        spec_payload, question=question, duration_seconds=duration
    )
    if not isinstance(parsed_spec, CountSpec):
        raise AssertionError("validated supported CountSpec became unsupported")

    evidence_prompt = build_evidence_prompt(parsed_spec)
    static_bindings: list[dict[str, object]] = []
    if parsed_spec.route == "temporal":
        evidence_parts = build_temporal_parts(
            file_uri=str(upload["file_uri"]),
            mime_type=str(upload["delivery_mime_type"]),
            views=parsed_spec.candidate_views,
            fps=7.5,
        )
        evidence_fps: float | None = 7.5
    else:
        evidence_parts, static_bindings = extract_static_parts(
            delivery_path=Path(str(row["delivery_path"])),
            views=parsed_spec.candidate_views,
            output_root=fixed_root / "evidence" / "frames",
        )
        evidence_fps = None
    evidence_request = build_stage_request(
        stage="evidence",
        arm="fixed",
        video_id=video_id,
        model=MODEL,
        prompt=evidence_prompt,
        response_schema=evidence_schema(route=parsed_spec.route),
        media_binding=_stage_media_binding(
            upload=upload,
            fps=evidence_fps,
            views=parsed_spec.candidate_views,
            static_bindings=static_bindings,
        ),
        predecessor_sha256=str(spec_terminal["complete_sha256"]),
    )
    evidence_stage_root = fixed_root / "evidence"
    _write_text_once(evidence_stage_root / "prompt.txt", evidence_prompt)

    def validate_evidence(payload: Mapping[str, object]) -> dict[str, object]:
        return evidence_ledger_to_dict(
            parse_evidence(
                payload,
                spec=parsed_spec,
                requested_views=parsed_spec.candidate_views,
            )
        )

    evidence_terminal = execute_stage(
        client=client,
        stage_root=evidence_stage_root,
        request=evidence_request,
        parts=evidence_parts,
        prompt=evidence_prompt,
        response_schema=evidence_schema(route=parsed_spec.route),
        budget_root=root / "budget",
        reserved_usd=worst_case_call_cost_usd(duration_seconds=duration, fps=7.5),
        hard_ceiling_usd=hard_ceiling_usd,
        validator=validate_evidence,
    )
    stage_hashes["evidence"] = evidence_terminal["complete_sha256"]
    if evidence_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=fixed_root,
            video_id=video_id,
            arm="fixed",
            reason="evidence_stage_failed",
            stage_hashes=stage_hashes,
        )
    ledger = parse_evidence(
        _validated_payload(evidence_stage_root),
        spec=parsed_spec,
        requested_views=parsed_spec.candidate_views,
    )

    coverage_prompt = build_coverage_prompt(parsed_spec, ledger)
    coverage_part = build_full_video_part(
        file_uri=str(upload["file_uri"]),
        mime_type=str(upload["delivery_mime_type"]),
        fps=7.5,
    )
    coverage_request = build_stage_request(
        stage="coverage",
        arm="fixed",
        video_id=video_id,
        model=MODEL,
        prompt=coverage_prompt,
        response_schema=coverage_schema(),
        media_binding=_stage_media_binding(
            upload=upload,
            fps=7.5,
            views=(CandidateView("full", "window", 0.0, duration),),
        ),
        predecessor_sha256=str(evidence_terminal["complete_sha256"]),
    )
    coverage_stage_root = fixed_root / "coverage"
    _write_text_once(coverage_stage_root / "prompt.txt", coverage_prompt)

    def validate_coverage(payload: Mapping[str, object]) -> dict[str, object]:
        normalized = normalize_coverage_payload(payload)
        return coverage_review_to_dict(
            parse_coverage_review(normalized, spec=parsed_spec, ledger=ledger)
        )

    coverage_terminal = execute_stage(
        client=client,
        stage_root=coverage_stage_root,
        request=coverage_request,
        parts=(coverage_part,),
        prompt=coverage_prompt,
        response_schema=coverage_schema(),
        budget_root=root / "budget",
        reserved_usd=worst_case_call_cost_usd(duration_seconds=duration, fps=7.5),
        hard_ceiling_usd=hard_ceiling_usd,
        validator=validate_coverage,
    )
    stage_hashes["coverage"] = coverage_terminal["complete_sha256"]
    if coverage_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=fixed_root,
            video_id=video_id,
            arm="fixed",
            reason="coverage_stage_failed",
            stage_hashes=stage_hashes,
        )
    review = parse_coverage_review(
        _validated_payload(coverage_stage_root), spec=parsed_spec, ledger=ledger
    )
    count = reduce_count(parsed_spec, ledger, review)
    result = write_self_hashed_once(
        fixed_root / "result.json",
        {
            "schema_version": "bounded-multipass-result-v1",
            "video_id": video_id,
            "arm": "fixed",
            **count_result_to_dict(count),
            "count_spec_sha256": spec_terminal["validated_sha256"],
            "evidence_sha256": evidence_terminal["validated_sha256"],
            "coverage_sha256": coverage_terminal["validated_sha256"],
        },
    )
    return write_self_hashed_once(
        fixed_root / "complete.json",
        {
            "schema_version": "bounded-multipass-row-complete-v1",
            "video_id": video_id,
            "arm": "fixed",
            "outcome": "complete",
            "reason": None,
            "stage_complete_sha256": stage_hashes,
            "result_sha256": result["complete_sha256"],
        },
    )


def _copy_fixed_result_to_refinement(
    *, fixed_result: Mapping[str, object], refinement_root: Path, fixed_hash: str
) -> dict[str, object]:
    result_payload = dict(fixed_result)
    result_payload.pop("complete_sha256", None)
    result_payload["arm"] = "refinement"
    result_payload["fixed_result_sha256"] = fixed_hash
    result = write_self_hashed_once(refinement_root / "result.json", result_payload)
    return write_self_hashed_once(
        refinement_root / "complete.json",
        {
            "schema_version": "bounded-multipass-row-complete-v1",
            "video_id": fixed_result["video_id"],
            "arm": "refinement",
            "outcome": "complete",
            "reason": "no_refinement_requested",
            "fixed_result_sha256": fixed_hash,
            "stage_complete_sha256": {},
            "result_sha256": result["complete_sha256"],
        },
    )


def run_refinement_row(
    *,
    client: Any,
    row: Mapping[str, object],
    root: Path,
    hard_ceiling_usd: float,
) -> dict[str, object]:
    video_id = str(row["video_id"])
    fixed_root = root / "rows" / video_id / "fixed"
    refinement_root = root / "rows" / video_id / "refinement"
    if refinement_root.joinpath("complete.json").exists():
        terminal = _load_json(refinement_root / "complete.json")
        _verify_self_hash(terminal, name="refinement row terminal")
        return terminal
    fixed_terminal = _load_json(fixed_root / "complete.json")
    fixed_terminal_hash = _verify_self_hash(fixed_terminal, name="fixed row terminal")
    if fixed_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=refinement_root,
            video_id=video_id,
            arm="refinement",
            reason="fixed_arm_incomplete",
            stage_hashes={},
        )
    fixed_result = _load_json(fixed_root / "result.json")
    fixed_result_hash = _verify_self_hash(fixed_result, name="fixed result")
    if fixed_result_hash != fixed_terminal.get("result_sha256"):
        raise ValueError("fixed terminal does not bind fixed result")
    question = str(row["question"])
    duration = float(row["duration_seconds"])
    spec_payload = _validated_payload(fixed_root / "count-spec")
    parsed_spec = parse_count_spec(
        spec_payload, question=question, duration_seconds=duration
    )
    if not isinstance(parsed_spec, CountSpec):
        return _copy_fixed_result_to_refinement(
            fixed_result=fixed_result,
            refinement_root=refinement_root,
            fixed_hash=fixed_result_hash,
        )
    initial = parse_evidence(
        _validated_payload(fixed_root / "evidence"),
        spec=parsed_spec,
        requested_views=parsed_spec.candidate_views,
    )
    initial_review = parse_coverage_review(
        _validated_payload(fixed_root / "coverage"),
        spec=parsed_spec,
        ledger=initial,
    )
    request_value = initial_review.refinement_request
    if request_value is None:
        return _copy_fixed_result_to_refinement(
            fixed_result=fixed_result,
            refinement_root=refinement_root,
            fixed_hash=fixed_result_hash,
        )
    upload = ensure_upload(client=client, row=row, row_root=root / "rows" / video_id)
    if request_value.kind == "window":
        refinement_view = CandidateView(
            "R0",
            "window",
            start_seconds=request_value.start_seconds,
            end_seconds=request_value.end_seconds,
        )
        refinement_parts = build_temporal_parts(
            file_uri=str(upload["file_uri"]),
            mime_type=str(upload["delivery_mime_type"]),
            views=(refinement_view,),
            fps=12.0,
        )
        refinement_fps: float | None = 12.0
        static_bindings: list[dict[str, object]] = []
    elif request_value.kind == "crop":
        return _incomplete_row(
            row_root=refinement_root,
            video_id=video_id,
            arm="refinement",
            reason="static_crop_refinement_not_implemented",
            stage_hashes={},
        )
    else:
        original = next(
            (
                view
                for view in parsed_spec.candidate_views
                if view.view_id == request_value.view_id
            ),
            None,
        )
        if original is None or original.timestamp_seconds is None:
            return _incomplete_row(
                row_root=refinement_root,
                video_id=video_id,
                arm="refinement",
                reason="invalid_static_refinement_view",
                stage_hashes={},
            )
        refinement_view = CandidateView(
            "R0", "shot_frame", timestamp_seconds=original.timestamp_seconds
        )
        refinement_parts, static_bindings = extract_static_parts(
            delivery_path=Path(str(row["delivery_path"])),
            views=(refinement_view,),
            output_root=refinement_root / "refinement-evidence" / "frames",
        )
        refinement_fps = None
    refinement_prompt = (
        build_evidence_prompt(parsed_spec)
        + "\nThis is the sole targeted refinement. Review only view R0. Use new record "
        "IDs that do not duplicate the initial ledger."
    )
    refinement_request = build_stage_request(
        stage="refinement-evidence",
        arm="refinement",
        video_id=video_id,
        model=MODEL,
        prompt=refinement_prompt,
        response_schema=evidence_schema(route=parsed_spec.route),
        media_binding=_stage_media_binding(
            upload=upload,
            fps=refinement_fps,
            views=(refinement_view,),
            static_bindings=static_bindings,
        ),
        predecessor_sha256=fixed_terminal_hash,
    )
    stage_root = refinement_root / "refinement-evidence"
    _write_text_once(stage_root / "prompt.txt", refinement_prompt)

    def validate_refinement(payload: Mapping[str, object]) -> dict[str, object]:
        return evidence_ledger_to_dict(
            parse_evidence(
                payload, spec=parsed_spec, requested_views=(refinement_view,)
            )
        )

    refinement_terminal = execute_stage(
        client=client,
        stage_root=stage_root,
        request=refinement_request,
        parts=refinement_parts,
        prompt=refinement_prompt,
        response_schema=evidence_schema(route=parsed_spec.route),
        budget_root=root / "budget",
        reserved_usd=worst_case_call_cost_usd(
            duration_seconds=(
                duration
                if refinement_view.start_seconds is None
                else float(refinement_view.end_seconds)
                - float(refinement_view.start_seconds)
            ),
            fps=12.0,
        ),
        hard_ceiling_usd=hard_ceiling_usd,
        validator=validate_refinement,
    )
    stage_hashes: dict[str, object] = {
        "refinement_evidence": refinement_terminal["complete_sha256"]
    }
    if refinement_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=refinement_root,
            video_id=video_id,
            arm="refinement",
            reason="refinement_evidence_stage_failed",
            stage_hashes=stage_hashes,
        )
    refinement = parse_evidence(
        _validated_payload(stage_root),
        spec=parsed_spec,
        requested_views=(refinement_view,),
    )
    merged = merge_refinement(initial, refinement)

    final_prompt = build_coverage_prompt(parsed_spec, merged)
    final_part = build_full_video_part(
        file_uri=str(upload["file_uri"]),
        mime_type=str(upload["delivery_mime_type"]),
        fps=7.5,
    )
    final_request = build_stage_request(
        stage="final-coverage",
        arm="refinement",
        video_id=video_id,
        model=MODEL,
        prompt=final_prompt,
        response_schema=coverage_schema(),
        media_binding=_stage_media_binding(
            upload=upload,
            fps=7.5,
            views=(CandidateView("full", "window", 0.0, duration),),
        ),
        predecessor_sha256=str(refinement_terminal["complete_sha256"]),
    )
    final_root = refinement_root / "final-coverage"
    _write_text_once(final_root / "prompt.txt", final_prompt)

    def validate_final(payload: Mapping[str, object]) -> dict[str, object]:
        normalized = normalize_coverage_payload(payload)
        return coverage_review_to_dict(
            parse_coverage_review(
                normalized,
                spec=parsed_spec,
                ledger=merged,
                requested_views=(*parsed_spec.candidate_views, refinement_view),
            )
        )

    final_terminal = execute_stage(
        client=client,
        stage_root=final_root,
        request=final_request,
        parts=(final_part,),
        prompt=final_prompt,
        response_schema=coverage_schema(),
        budget_root=root / "budget",
        reserved_usd=worst_case_call_cost_usd(duration_seconds=duration, fps=7.5),
        hard_ceiling_usd=hard_ceiling_usd,
        validator=validate_final,
    )
    stage_hashes["final_coverage"] = final_terminal["complete_sha256"]
    if final_terminal.get("outcome") != "complete":
        return _incomplete_row(
            row_root=refinement_root,
            video_id=video_id,
            arm="refinement",
            reason="final_coverage_stage_failed",
            stage_hashes=stage_hashes,
        )
    final_review = parse_coverage_review(
        _validated_payload(final_root),
        spec=parsed_spec,
        ledger=merged,
        requested_views=(*parsed_spec.candidate_views, refinement_view),
    )
    count = reduce_count(parsed_spec, merged, final_review)
    result = write_self_hashed_once(
        refinement_root / "result.json",
        {
            "schema_version": "bounded-multipass-result-v1",
            "video_id": video_id,
            "arm": "refinement",
            **count_result_to_dict(count),
            "fixed_result_sha256": fixed_result_hash,
            "refinement_evidence_sha256": refinement_terminal["validated_sha256"],
            "final_coverage_sha256": final_terminal["validated_sha256"],
        },
    )
    return write_self_hashed_once(
        refinement_root / "complete.json",
        {
            "schema_version": "bounded-multipass-row-complete-v1",
            "video_id": video_id,
            "arm": "refinement",
            "outcome": "complete",
            "reason": None,
            "fixed_result_sha256": fixed_result_hash,
            "stage_complete_sha256": stage_hashes,
            "result_sha256": result["complete_sha256"],
        },
    )


def _load_preparation(root: Path) -> dict[str, Any]:
    preparation = _load_json(root / "preparation-complete.json")
    _verify_self_hash(preparation, name="preparation terminal")
    requests = _load_json(root / "requests.label-blind.json")
    requests_hash = _verify_self_hash(requests, name="requests freeze")
    if requests_hash != preparation.get("requests_complete_sha256"):
        raise ValueError("preparation terminal does not bind requests")
    request_rows = requests.get("rows")
    if not isinstance(request_rows, list):
        raise ValueError("label-blind request rows are missing")
    if len(request_rows) != 12:
        raise ValueError("preparation must contain exactly twelve label-blind rows")
    if not all(isinstance(row, Mapping) for row in request_rows):
        raise ValueError("label-blind request row is invalid")
    request_ids = [row.get("video_id") for row in request_rows]
    if len(set(request_ids)) != 12 or not all(
        isinstance(video_id, str) and video_id for video_id in request_ids
    ):
        raise ValueError("label-blind request identities are invalid")
    validate_label_blind(requests)
    smoke_id = requests.get("smoke_video_id") or preparation.get("smoke_video_id")
    if not isinstance(smoke_id, str) or smoke_id not in set(request_ids):
        raise ValueError("label-blind preparation must declare a valid smoke_video_id")
    return requests


def _declared_smoke_id(*, root: Path, requests: Mapping[str, object]) -> str:
    preparation = _load_json(root / "preparation-complete.json")
    smoke_id = requests.get("smoke_video_id") or preparation.get("smoke_video_id")
    if not isinstance(smoke_id, str) or not smoke_id:
        raise ValueError("label-blind preparation must declare a valid smoke_video_id")
    return smoke_id


def run_preflight(*, root: Path, client: Any) -> dict[str, object]:
    _load_preparation(root)
    model = client.models.get(model=MODEL)
    returned_name = str(getattr(model, "name", ""))
    if MODEL not in returned_name:
        raise RuntimeError(
            f"provider returned unexpected model identity: {returned_name}"
        )
    pricing = write_self_hashed_once(
        root / "pricing.json",
        {
            "schema_version": "bounded-multipass-pricing-v1",
            "model": MODEL,
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "input_usd_per_million_tokens": 2.0,
            "output_usd_per_million_tokens": 12.0,
            "thinking_usd_per_million_tokens": 12.0,
            "global_ceiling_usd": GLOBAL_CEILING_USD,
            "smoke_ceiling_usd": SMOKE_CEILING_USD,
            "pre_refinement_ceiling_usd": PRE_REFINEMENT_CEILING_USD,
        },
    )
    return write_self_hashed_once(
        root / "preflight.json",
        {
            "schema_version": "bounded-multipass-preflight-v1",
            "model": MODEL,
            "returned_model_name": returned_name,
            "pricing_complete_sha256": pricing["complete_sha256"],
            "response_mime_type": "application/json",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_level": THINKING_LEVEL,
            "provider": "direct_google",
            "credential_present": True,
        },
    )


def _row_by_id(requests: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = requests.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("requests rows are invalid")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("video_id"), str):
            raise ValueError("request row is invalid")
        video_id = str(row["video_id"])
        if video_id in result:
            raise ValueError(f"duplicate request video ID: {video_id}")
        result[video_id] = row
    return result


def run_smoke(*, root: Path, client: Any) -> dict[str, object]:
    requests = _load_preparation(root)
    smoke_id = _declared_smoke_id(root=root, requests=requests)
    row = _row_by_id(requests)[smoke_id]
    terminal = run_fixed_row(
        client=client,
        row=row,
        root=root,
        hard_ceiling_usd=SMOKE_CEILING_USD,
    )
    return write_self_hashed_once(
        root / "smoke-complete.json",
        {
            "schema_version": "bounded-multipass-smoke-complete-v1",
            "video_id": smoke_id,
            "row_complete_sha256": terminal["complete_sha256"],
            "outcome": terminal["outcome"],
            "structurally_valid": terminal["outcome"] == "complete",
        },
    )


def run_fixed_arm(*, root: Path, client: Any) -> dict[str, object]:
    requests = _load_preparation(root)
    smoke = _load_json(root / "smoke-complete.json")
    _verify_self_hash(smoke, name="smoke terminal")
    if smoke.get("structurally_valid") is not True:
        raise RuntimeError("smoke is not structurally valid")
    terminals = {}
    for video_id, row in _row_by_id(requests).items():
        terminal = run_fixed_row(
            client=client,
            row=row,
            root=root,
            hard_ceiling_usd=PRE_REFINEMENT_CEILING_USD,
        )
        terminals[video_id] = terminal["complete_sha256"]
        if terminal.get("outcome") != "complete":
            break
    return write_self_hashed_once(
        root / "fixed-arm-complete.json",
        {
            "schema_version": "bounded-multipass-arm-complete-v1",
            "arm": "fixed",
            "planned_rows": 12,
            "completed_rows": sum(
                1
                for video_id in terminals
                if _load_json(root / "rows" / video_id / "fixed" / "complete.json").get(
                    "outcome"
                )
                == "complete"
            ),
            "row_complete_sha256": terminals,
        },
    )


def _authenticate_predictions_freeze(
    *, root: Path, arm: str, requests: Mapping[str, object]
) -> dict[str, Any]:
    freeze_path = root / f"{arm}-predictions-freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError(f"{arm} predictions freeze is required before refinement")
    freeze = _load_json(freeze_path)
    freeze_hash = _verify_self_hash(freeze, name=f"{arm} predictions freeze")
    if freeze.get("arm") != arm:
        raise ValueError(f"{arm} predictions freeze has the wrong arm")
    arm_terminal = _load_json(root / f"{arm}-arm-complete.json")
    arm_terminal_hash = _verify_self_hash(arm_terminal, name=f"{arm} arm terminal")
    if freeze.get("arm_complete_sha256") != arm_terminal_hash:
        raise ValueError(f"{arm} predictions freeze does not bind arm terminal")
    if arm_terminal.get("completed_rows") != 12:
        raise RuntimeError(f"{arm} arm is not complete")
    freeze_rows = freeze.get("rows")
    request_rows = requests.get("rows")
    if not isinstance(freeze_rows, list) or not isinstance(request_rows, list):
        raise ValueError(f"{arm} freeze/request rows are missing")
    request_ids = {str(row["video_id"]) for row in request_rows}
    freeze_by_id = {str(row["video_id"]): row for row in freeze_rows}
    if len(freeze_rows) != 12 or set(freeze_by_id) != request_ids:
        raise ValueError(f"{arm} predictions freeze identity set differs from requests")
    terminal_rows = arm_terminal.get("row_complete_sha256")
    if not isinstance(terminal_rows, Mapping) or set(terminal_rows) != request_ids:
        raise ValueError(f"{arm} arm terminal identity set differs from requests")
    for video_id in request_ids:
        row_root = root / "rows" / video_id / arm
        terminal = _load_json(row_root / "complete.json")
        terminal_hash = _verify_self_hash(terminal, name=f"{video_id} {arm} terminal")
        if terminal_hash != terminal_rows.get(video_id):
            raise ValueError(f"{video_id} {arm} arm binding changed")
        if terminal.get("outcome") != "complete":
            raise ValueError(f"{video_id} {arm} row is not complete")
        freeze_row = freeze_by_id[video_id]
        if freeze_row.get("row_complete_sha256") != terminal_hash:
            raise ValueError(f"{video_id} {arm} freeze row binding changed")
        result = _load_json(row_root / "result.json")
        result_hash = _verify_self_hash(result, name=f"{video_id} {arm} result")
        if result_hash != terminal.get(
            "result_sha256"
        ) or result_hash != freeze_row.get("result_sha256"):
            raise ValueError(f"{video_id} {arm} freeze result binding changed")
        for field in ("outcome", "value", "reasons", "selected_record_ids"):
            if result.get(field) != freeze_row.get(field):
                raise ValueError(f"{video_id} {arm} freeze {field} changed")
    freeze["complete_sha256"] = freeze_hash
    return freeze


def run_refinement_arm(*, root: Path, client: Any) -> dict[str, object]:
    requests = _load_preparation(root)
    _authenticate_predictions_freeze(root=root, arm="fixed", requests=requests)
    terminals = {}
    for video_id, row in _row_by_id(requests).items():
        terminal = run_refinement_row(
            client=client,
            row=row,
            root=root,
            hard_ceiling_usd=GLOBAL_CEILING_USD,
        )
        terminals[video_id] = terminal["complete_sha256"]
        if terminal.get("outcome") != "complete":
            break
    return write_self_hashed_once(
        root / "refinement-arm-complete.json",
        {
            "schema_version": "bounded-multipass-arm-complete-v1",
            "arm": "refinement",
            "planned_rows": 12,
            "completed_rows": sum(
                1
                for video_id in terminals
                if _load_json(
                    root / "rows" / video_id / "refinement" / "complete.json"
                ).get("outcome")
                == "complete"
            ),
            "row_complete_sha256": terminals,
        },
    )


def freeze_predictions(*, root: Path, arm: str) -> dict[str, object]:
    if arm not in {"fixed", "refinement"}:
        raise ValueError("arm must be fixed or refinement")
    requests = _load_preparation(root)
    arm_terminal = _load_json(root / f"{arm}-arm-complete.json")
    arm_hash = _verify_self_hash(arm_terminal, name=f"{arm} arm terminal")
    if arm_terminal.get("completed_rows") != 12:
        raise RuntimeError(f"{arm} arm is incomplete")
    rows = []
    for video_id in _row_by_id(requests):
        terminal = _load_json(root / "rows" / video_id / arm / "complete.json")
        terminal_hash = _verify_self_hash(terminal, name=f"{arm} row terminal")
        if terminal.get("outcome") != "complete":
            raise RuntimeError(f"{arm} row is incomplete: {video_id}")
        result = _load_json(root / "rows" / video_id / arm / "result.json")
        result_hash = _verify_self_hash(result, name=f"{arm} row result")
        if result_hash != terminal.get("result_sha256"):
            raise ValueError(f"{arm} terminal does not bind result for {video_id}")
        rows.append(
            {
                "video_id": video_id,
                "outcome": result["outcome"],
                "value": result["value"],
                "reasons": result["reasons"],
                "selected_record_ids": result["selected_record_ids"],
                "row_complete_sha256": terminal_hash,
                "result_sha256": result_hash,
            }
        )
    return write_self_hashed_once(
        root / f"{arm}-predictions-freeze.json",
        {
            "schema_version": "bounded-multipass-predictions-freeze-v1",
            "arm": arm,
            "arm_complete_sha256": arm_hash,
            "rows": rows,
        },
    )


def status_payload(root: Path) -> dict[str, object]:
    budget_path = root / "budget" / "budget-ledger.json"
    exposure = (
        _budget_exposure(_budget_ledger(budget_path)) if budget_path.exists() else 0.0
    )
    return {
        "root": str(root),
        "model": MODEL,
        "budget_exposure_usd": exposure,
        "preflight": (root / "preflight.json").exists(),
        "smoke": (root / "smoke-complete.json").exists(),
        "fixed_arm": (root / "fixed-arm-complete.json").exists(),
        "refinement_arm": (root / "refinement-arm-complete.json").exists(),
        "fixed_frozen": (root / "fixed-predictions-freeze.json").exists(),
        "refinement_frozen": (root / "refinement-predictions-freeze.json").exists(),
    }


def client_from_env() -> Any:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is unavailable")
    return genai.Client(api_key=api_key)


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "run-smoke",
            "run-fixed",
            "run-refinement",
            "freeze-fixed",
            "freeze-refinement",
            "status",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "status":
        result = status_payload(args.root)
    else:
        client = client_from_env()
        if args.command == "preflight":
            result = run_preflight(root=args.root, client=client)
        elif args.command == "run-smoke":
            result = run_smoke(root=args.root, client=client)
        elif args.command == "run-fixed":
            result = run_fixed_arm(root=args.root, client=client)
        elif args.command == "run-refinement":
            result = run_refinement_arm(root=args.root, client=client)
        elif args.command == "freeze-fixed":
            result = freeze_predictions(root=args.root, arm="fixed")
        else:
            result = freeze_predictions(root=args.root, arm="refinement")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
