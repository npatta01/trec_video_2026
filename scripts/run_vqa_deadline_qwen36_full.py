#!/usr/bin/env python3
"""Resumable four-shard Qwen3.6 native-video deadline runner.

Each query has two matched conditions: native video and video plus the frozen
accepted Whisper transcript.  The default mode is provider-free preflight;
``--run`` is required for calls.  Raw provider bytes are frozen before parsing
and every call has a durable one-attempt marker for safe resume.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import request as urllib_request


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@sha256:"
    "77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
TRANSCRIPT_CONDITION_ID = "video-plus-transcript"
NATIVE_CONDITION_ID = "native-video"
CONDITION_IDS = (TRANSCRIPT_CONDITION_ID, NATIVE_CONDITION_ID)
DIRECT_ANSWER_PROMPT = "Use the complete native video and audio to answer the question. Return one short phrase or sentence, answer only."
EXPECTED_QUERY_COUNT = 2_443
EXPECTED_SHARD_COUNT = 4
_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "answer",
        "answers",
        "annotation",
        "annotations",
        "expected_answer",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_answer",
        "reference_answers",
    }
)
_SAFE_ERROR = re.compile(r"[^a-z0-9_-]+")
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_labels(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label/reference field {key!r} at {location}")
            _reject_labels(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_labels(child, f"{location}[{index}]")


def _load_json(path: Path) -> object:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON input: {path}") from error
    _reject_labels(value)
    return value


def _load_accepted_transcripts(path: Path) -> dict[str, str]:
    payload = _load_json(path)
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("accepted Whisper input must contain rows")
    transcripts: dict[str, str] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("accepted Whisper row is invalid")
        quality = raw.get("quality")
        if isinstance(quality, Mapping) and quality.get("status") != "accepted":
            continue
        status = raw.get("status")
        if status is not None and status != "accepted":
            continue
        video_id = raw.get("video_id")
        transcript = raw.get("transcript") or raw.get("timestamped_text") or raw.get("text")
        if not isinstance(video_id, str):
            raise ValueError("accepted Whisper row is missing video_id")
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        transcript = transcript.strip()
        if video_id in transcripts and transcripts[video_id] != transcript:
            raise ValueError(f"accepted Whisper rows disagree for {video_id}")
        transcripts[video_id] = transcript
    return transcripts


def _verify_input_contract(requests_path: Path) -> dict[str, Any] | None:
    root = requests_path.parent
    freeze_path = root / "inputs-freeze.json"
    contract_path = root / "media-contract.json"
    sums_path = root / "SHA256SUMS"
    if not (freeze_path.is_file() and contract_path.is_file() and sums_path.is_file()):
        return None
    freeze = _load_json(freeze_path)
    contract = _load_json(contract_path)
    if not isinstance(freeze, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("frozen input contract artifacts must be objects")
    if freeze.get("status") != "inputs_frozen" or contract.get("status") != "inputs_frozen":
        raise ValueError("frozen input contract status is invalid")
    if freeze.get("requests_sha256") != sha256_file(requests_path):
        raise ValueError("frozen requests hash mismatch")
    if freeze.get("media_contract_sha256") != sha256_file(contract_path):
        raise ValueError("frozen media contract hash mismatch")
    expected_sums = {
        "inputs-freeze.json": sha256_file(freeze_path),
        "media-contract.json": sha256_file(contract_path),
        "requests.json": sha256_file(requests_path),
    }
    seen: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            seen[parts[1].strip()] = parts[0]
    for name, digest in expected_sums.items():
        if seen.get(name) != digest:
            raise ValueError(f"frozen input checksum mismatch: {name}")
    return {"freeze": dict(freeze), "contract": dict(contract)}


def _resolve_media(media_root: Path, media_file: str) -> Path | None:
    candidate = (media_root / media_file).resolve()
    try:
        candidate.relative_to(media_root.resolve())
    except ValueError as error:
        raise ValueError(f"media path escapes input root: {media_file}") from error
    if candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def load_input_rows(
    requests_path: Path,
    *,
    whisper_path: Path,
    media_root: Path,
    expected_count: int | None = EXPECTED_QUERY_COUNT,
    verify_contract: bool = True,
) -> list[dict[str, Any]]:
    """Load the frozen label-free query/media contract and accepted transcripts."""

    requests_path = Path(requests_path).resolve(strict=True)
    if verify_contract:
        contract = _verify_input_contract(requests_path)
        if contract is None:
            raise ValueError("frozen input contract artifacts are required")
        frozen = contract["freeze"]
        if expected_count is not None and frozen.get("query_count") != expected_count:
            raise ValueError("frozen input query count mismatch")
    payload = _load_json(requests_path)
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("requests input must contain rows")
    if expected_count is not None and len(raw_rows) != expected_count:
        raise ValueError(f"requests input must contain {expected_count} rows")
    transcripts = _load_accepted_transcripts(Path(whisper_path).resolve(strict=True))
    media_root = Path(media_root).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request row {index} is invalid")
        query_id = raw.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id <= 0:
            raise ValueError(f"request row {index} query_id is invalid")
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        video_id = raw.get("video_id")
        question = raw.get("question")
        availability = raw.get("availability")
        media_file = raw.get("media_file")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"request row {index} video_id is missing")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"request row {index} question is missing")
        if availability not in {"available", "unavailable"}:
            raise ValueError(f"request row {index} availability is invalid")
        source_sha256 = raw.get("source_sha256")
        media_path = _resolve_media(media_root, media_file) if isinstance(media_file, str) else None
        status = "ready"
        reason: str | None = None
        media_sha256: str | None = None
        if availability != "available" or media_path is None:
            status = "unavailable"
            reason = "input_unavailable" if availability != "available" else "media_missing"
        else:
            if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256):
                raise ValueError(f"source SHA-256 is missing for {video_id}")
            media_sha256 = sha256_file(media_path)
            if media_sha256.casefold() != source_sha256.casefold():
                raise ValueError(f"media SHA-256 mismatch for {video_id}")
        transcript = transcripts.get(video_id)
        rows.append(
            {
                "query_id": query_id,
                "video_id": video_id,
                "question": question.strip(),
                "availability": availability,
                "status": status,
                "reason": reason,
                "media_file": media_file,
                "media_path": str(media_path) if media_path else None,
                "video_sha256": media_sha256,
                "transcript": transcript,
                "transcript_sha256": sha256_bytes(transcript.encode()) if transcript else None,
            }
        )
    return rows


def _video_data_uri(video_path: Path) -> str:
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/webm"
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build exactly two matched condition records per query row."""

    requests: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["query_id"])):
        for condition_id in CONDITION_IDS:
            transcript = row.get("transcript") if condition_id == TRANSCRIPT_CONDITION_ID else None
            status = str(row.get("status", "ready"))
            reason = row.get("reason")
            if condition_id == TRANSCRIPT_CONDITION_ID and not transcript:
                status = "unavailable"
                reason = "accepted_transcript_missing"
            requests.append(
                {
                    "request_id": f"{int(row['query_id'])}:{condition_id}",
                    "query_id": int(row["query_id"]),
                    "video_id": str(row["video_id"]),
                    "condition_id": condition_id,
                    "question": str(row["question"]),
                    "prompt": DIRECT_ANSWER_PROMPT,
                    "transcript": transcript,
                    "transcript_sha256": row.get("transcript_sha256") if transcript else None,
                    "video_path": row.get("media_path"),
                    "video_sha256": row.get("video_sha256"),
                    "status": status,
                    "reason": reason,
                }
            )
    return requests


def partition_rows(rows: Sequence[Mapping[str, Any]], *, shard_count: int = EXPECTED_SHARD_COUNT) -> list[list[dict[str, Any]]]:
    if shard_count != EXPECTED_SHARD_COUNT:
        raise ValueError("the full protocol requires exactly four shards")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["query_id"]))
    return [ordered[index::shard_count] for index in range(shard_count)]


def _safe_error(error: Exception) -> str:
    value = _SAFE_ERROR.sub("_", type(error).__name__.casefold()).strip("_")
    return value or "request_failed"


def _safe_name(request_id: str) -> str:
    return _SAFE_COMPONENT.sub("_", request_id)


def _extract_answer(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("provider response was not JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("provider response was not an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("provider choices are missing")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("provider answer content is missing")
    answer = message["content"].strip()
    if not answer:
        raise ValueError("provider answer is empty")
    return answer


def parse_response(*, raw: bytes, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        answer = _extract_answer(raw)
        status = "succeeded"
        error_type = None
    except ValueError as error:
        answer = ""
        status = "failed"
        error_type = _safe_error(error)
    return {
        "query_id": int(request["query_id"]),
        "video_id": str(request["video_id"]),
        "condition_id": str(request["condition_id"]),
        "status": status,
        "answer": answer,
        "error_type": error_type,
        "raw_response_sha256": sha256_bytes(raw),
    }


def post_json(url: str, payload: Mapping[str, Any]) -> bytes:
    body = canonical_json(payload)
    http_request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(http_request, timeout=600.0) as response:
        return response.read()


def _payload(request: Mapping[str, Any]) -> dict[str, Any]:
    video_path = request.get("video_path")
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("request media is unavailable")
    text = f"{DIRECT_ANSWER_PROMPT}\nQuestion: {request['question']}"
    transcript = request.get("transcript")
    if transcript:
        text += f"\nTimestamped transcript:\n{transcript}"
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": _video_data_uri(Path(video_path))}},
                    {"type": "text", "text": text},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _read_artifact(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid artifact: {path}") from error


def _request_manifest(requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "request_id",
        "query_id",
        "video_id",
        "condition_id",
        "question",
        "prompt",
        "video_sha256",
        "transcript_sha256",
        "status",
        "reason",
    )
    return [{field: request[field] for field in fields} for request in requests]


def write_checksums(root: Path) -> Path:
    target = root / "SHA256SUMS"
    excluded = {target, root / "vllm-server.log"}
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(path for path in root.rglob("*") if path.is_file() and path not in excluded)
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_shard(
    *,
    requests_path: Path,
    whisper_path: Path,
    media_root: Path,
    output_root: Path,
    shard_index: int,
    shard_count: int = EXPECTED_SHARD_COUNT,
    expected_count: int | None = EXPECTED_QUERY_COUNT,
    limit: int | None = None,
    server_url: str = "http://127.0.0.1:8000/v1/chat/completions",
    post: Callable[[str, Mapping[str, Any]], bytes | Mapping[str, Any]] | None = None,
    parse_response: Callable[..., dict[str, Any]] = parse_response,
    created_at: str | None = None,
    verify_contract: bool = True,
) -> dict[str, Any]:
    if shard_count != EXPECTED_SHARD_COUNT or shard_index not in range(shard_count):
        raise ValueError("the full protocol requires shard index 0..3 and shard count 4")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    rows = load_input_rows(
        requests_path,
        whisper_path=whisper_path,
        media_root=media_root,
        expected_count=expected_count,
        verify_contract=verify_contract,
    )
    selected_rows = partition_rows(rows, shard_count=shard_count)[shard_index]
    if limit is not None:
        selected_rows = selected_rows[:limit]
    requests = build_requests(selected_rows)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "raw-responses").mkdir(exist_ok=True)
    (output_root / "requests").mkdir(exist_ok=True)
    input_hash = sha256_file(Path(requests_path))
    transcript_hash = sha256_file(Path(whisper_path))
    created_at = created_at or _now()
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "limit": limit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "prompt": DIRECT_ANSWER_PROMPT,
        "requests_path_sha256": input_hash,
        "whisper_path_sha256": transcript_hash,
        "query_ids": [int(row["query_id"]) for row in selected_rows],
        "request_count": len(requests),
        "created_at": created_at,
        "requests": _request_manifest(requests),
    }
    manifest_path = output_root / "run-manifest.json"
    if manifest_path.exists():
        existing_manifest = _read_artifact(manifest_path)
        prior_created_at = existing_manifest.get("created_at")
        if not isinstance(prior_created_at, str) or not prior_created_at:
            raise ValueError("existing shard manifest has no creation timestamp")
        manifest["created_at"] = prior_created_at
        created_at = prior_created_at
        if existing_manifest != manifest:
            raise ValueError("existing shard manifest does not match input or shard selection")
    else:
        _write_exclusive(manifest_path, canonical_json(manifest))
    sender = post or post_json
    response_records: list[dict[str, Any]] = []
    for request in requests:
        request_key = _safe_name(str(request["request_id"]))
        raw_path = output_root / "raw-responses" / f"{request_key}.json"
        if raw_path.exists():
            raw = raw_path.read_bytes()
            if request["status"] == "ready":
                record = {
                    "request_id": request["request_id"],
                    "query_id": request["query_id"],
                    "video_id": request["video_id"],
                    "condition_id": request["condition_id"],
                    "status": "received",
                    "error_type": None,
                    "raw_response_sha256": sha256_bytes(raw),
                }
            else:
                record = {
                    "request_id": request["request_id"],
                    "query_id": request["query_id"],
                    "video_id": request["video_id"],
                    "condition_id": request["condition_id"],
                    "status": "unavailable",
                    "error_type": request["reason"],
                    "raw_response_sha256": None,
                }
            response_records.append(record)
            continue
        if request["status"] != "ready":
            raw = canonical_json({"status": "unavailable", "reason": request["reason"]})
            record = {
                "request_id": request["request_id"],
                "query_id": request["query_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "status": "unavailable",
                "error_type": request["reason"],
                "raw_response_sha256": None,
            }
            _write_exclusive(raw_path, raw)
            _write_exclusive(
                output_root / "requests" / f"{request_key}.json",
                canonical_json({"request": dict(request), "payload_sha256": None}),
            )
            response_records.append(record)
            continue
        request_path = output_root / "requests" / f"{request_key}.json"
        attempt_path = output_root / "requests" / f"{request_key}.attempt"
        payload = _payload(request)
        payload_sha256 = sha256_bytes(canonical_json(payload))
        request_record = {"request": dict(request), "payload_sha256": payload_sha256}
        if request_path.exists() and not attempt_path.exists():
            existing_request = _read_artifact(request_path)
            if existing_request != request_record:
                raise ValueError(f"request artifact changed: {request['request_id']}")
        elif not request_path.exists():
            _write_exclusive(request_path, canonical_json(request_record))
        if attempt_path.exists():
            raise ValueError(f"ambiguous request attempt: {request['request_id']}")
        _write_exclusive(attempt_path, b"reserved\n")
        try:
            response = sender(server_url, payload)
            raw = canonical_json(response) if isinstance(response, Mapping) else bytes(response)
            if not raw:
                raise ValueError("provider response was empty")
            record = {
                "request_id": request["request_id"],
                "query_id": request["query_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "status": "received",
                "error_type": None,
                "raw_response_sha256": sha256_bytes(raw),
            }
        except Exception as error:  # noqa: BLE001 - terminalize one attempt
            raw = canonical_json({"error_type": _safe_error(error)})
            record = {
                "request_id": request["request_id"],
                "query_id": request["query_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "status": "failed",
                "error_type": _safe_error(error),
                "raw_response_sha256": sha256_bytes(raw),
            }
        _write_exclusive(raw_path, raw)
        response_records.append(record)

    response_order = {request["request_id"]: index for index, request in enumerate(requests)}
    response_records.sort(key=lambda record: response_order[record["request_id"]])
    response_manifest_path = output_root / "raw-response-manifest.json"
    if response_manifest_path.exists():
        frozen_manifest = _read_artifact(response_manifest_path)
        frozen_records = frozen_manifest.get("responses") if isinstance(frozen_manifest, Mapping) else None
        if not isinstance(frozen_records, list):
            raise ValueError("raw response manifest is invalid")
        frozen_by_id = {record.get("request_id"): record for record in frozen_records if isinstance(record, Mapping)}
        if set(frozen_by_id) != set(response_order) or len(frozen_by_id) != len(frozen_records):
            raise ValueError("raw response manifest coverage changed during resume")
        for record in frozen_records:
            request_id = record["request_id"]
            raw_path = output_root / "raw-responses" / f"{_safe_name(request_id)}.json"
            if record.get("status") in {"received", "failed"}:
                if not raw_path.is_file() or record.get("raw_response_sha256") != sha256_file(raw_path):
                    raise ValueError(f"raw response changed during resume: {request_id}")
        response_records = [dict(frozen_by_id[request_id]) for request_id in response_order]
    else:
        _write_exclusive(
            response_manifest_path,
            canonical_json({"schema_version": 1, "responses": response_records}),
        )
    raw_freeze_path = output_root / "raw-freeze.json"
    raw_freeze = {
        "schema_version": 1,
        "status": "raw_responses_frozen",
        "model_id": MODEL_ID,
        "request_count": len(requests),
        "responses": response_records,
        "raw_response_manifest_sha256": sha256_file(output_root / "raw-response-manifest.json"),
    }
    if raw_freeze_path.exists():
        if _read_artifact(raw_freeze_path) != raw_freeze:
            raise ValueError("raw freeze changed during resume")
    else:
        _write_exclusive(raw_freeze_path, canonical_json(raw_freeze))

    predictions: list[dict[str, Any]] = []
    by_id = {request["request_id"]: request for request in requests}
    for record in response_records:
        request = by_id[record["request_id"]]
        if record["status"] == "received":
            parsed = parse_response(raw=(output_root / "raw-responses" / f"{_safe_name(record['request_id'])}.json").read_bytes(), request=request)
            parsed["raw_response_sha256"] = record["raw_response_sha256"]
        elif record["status"] == "unavailable":
            parsed = {
                "query_id": request["query_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "status": "unavailable",
                "answer": "",
                "error_type": record["error_type"],
                "raw_response_sha256": None,
            }
        else:
            parsed = {
                "query_id": request["query_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "status": "failed",
                "answer": "",
                "error_type": record["error_type"],
                "raw_response_sha256": record["raw_response_sha256"],
            }
        predictions.append(parsed)
    predictions.sort(key=lambda row: (int(row["query_id"]), CONDITION_IDS.index(row["condition_id"])))
    predictions_freeze = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model_id": MODEL_ID,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "raw_freeze_sha256": sha256_file(raw_freeze_path),
        "row_count": len(predictions),
        "rows": predictions,
    }
    predictions_path = output_root / "predictions-freeze.json"
    if predictions_path.exists():
        if _read_artifact(predictions_path) != predictions_freeze:
            raise ValueError("predictions freeze changed during resume")
    else:
        _write_exclusive(predictions_path, canonical_json(predictions_freeze))
    write_checksums(output_root)
    return {
        "status": "predictions_frozen",
        "shard_index": shard_index,
        "row_count": len(predictions),
        "request_count": len(requests),
        "succeeded_count": sum(row["status"] == "succeeded" for row in predictions),
        "failed_count": sum(row["status"] == "failed" for row in predictions),
        "unavailable_count": sum(row["status"] == "unavailable" for row in predictions),
    }


def aggregate_shards(
    shard_roots: Sequence[Path], *, output_root: Path,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    shard_count: int = EXPECTED_SHARD_COUNT,
) -> dict[str, Any]:
    if shard_count != EXPECTED_SHARD_COUNT or len(shard_roots) != shard_count:
        raise ValueError("aggregation requires exactly four shard roots")
    shard_freezes: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    rows: list[dict[str, Any]] = []
    expected_manifest: dict[str, Any] | None = None
    for root in shard_roots:
        manifest = _read_artifact(Path(root) / "run-manifest.json")
        freeze = _read_artifact(Path(root) / "predictions-freeze.json")
        shard_index = manifest.get("shard_index")
        if shard_index in seen_shards or shard_index not in range(shard_count):
            raise ValueError("shard indices must be unique 0..3")
        seen_shards.add(shard_index)
        if manifest.get("shard_count") != shard_count or freeze.get("shard_index") != shard_index:
            raise ValueError("shard contract mismatch")
        identity = {
            key: manifest.get(key)
            for key in ("model_id", "model_revision", "runtime_image", "prompt", "requests_path_sha256", "whisper_path_sha256")
        }
        if expected_manifest is None:
            expected_manifest = identity
        elif identity != expected_manifest:
            raise ValueError("shards do not share one frozen input/model contract")
        shard_rows = freeze.get("rows")
        if not isinstance(shard_rows, list):
            raise ValueError("shard predictions rows are missing")
        rows.extend(dict(row) for row in shard_rows)
        shard_freezes.append(
            {
                "shard_index": shard_index,
                "predictions_freeze_sha256": sha256_file(Path(root) / "predictions-freeze.json"),
                "row_count": len(shard_rows),
            }
        )
    if seen_shards != set(range(shard_count)):
        raise ValueError("all four shard indices are required")
    expected_rows = expected_query_count * len(CONDITION_IDS)
    if len(rows) != expected_rows:
        raise ValueError(f"aggregate requires exactly {expected_rows} rows")
    keys = {(int(row["query_id"]), row["condition_id"]) for row in rows}
    expected_keys = {
        (query_id, condition)
        for query_id in range(1, expected_query_count + 1)
        for condition in CONDITION_IDS
    }
    if keys != expected_keys:
        raise ValueError("aggregate query/condition coverage is incomplete or duplicated")
    rows.sort(key=lambda row: (int(row["query_id"]), CONDITION_IDS.index(row["condition_id"])))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model_id": expected_manifest["model_id"] if expected_manifest else MODEL_ID,
        "model_revision": expected_manifest["model_revision"] if expected_manifest else MODEL_REVISION,
        "runtime_image": expected_manifest["runtime_image"] if expected_manifest else RUNTIME_IMAGE,
        "prompt": expected_manifest["prompt"] if expected_manifest else DIRECT_ANSWER_PROMPT,
        "query_count": expected_query_count,
        "row_count": len(rows),
        "shard_count": shard_count,
        "shards": sorted(shard_freezes, key=lambda item: item["shard_index"]),
        "rows": rows,
    }
    freeze_path = output_root / "predictions-freeze.json"
    if freeze_path.exists():
        if _read_artifact(freeze_path) != aggregate:
            raise ValueError("aggregate freeze changed")
    else:
        _write_exclusive(freeze_path, canonical_json(aggregate))
    write_checksums(output_root)
    return {"status": "predictions_frozen", "query_count": expected_query_count, "row_count": len(rows), "shard_count": shard_count}


def preflight(
    requests_path: Path, whisper_path: Path, media_root: Path,
    *, expected_count: int | None = EXPECTED_QUERY_COUNT, verify_contract: bool = True
) -> dict[str, Any]:
    rows = load_input_rows(
        requests_path,
        whisper_path=whisper_path,
        media_root=media_root,
        expected_count=expected_count,
        verify_contract=verify_contract,
    )
    requests = build_requests(rows)
    return {
        "status": "preflight_passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "query_count": len(rows),
        "request_count": len(requests),
        "native_request_count": sum(request["status"] == "ready" and request["condition_id"] == NATIVE_CONDITION_ID for request in requests),
        "transcript_request_count": sum(request["status"] == "ready" and request["condition_id"] == TRANSCRIPT_CONDITION_ID for request in requests),
        "shard_count": EXPECTED_SHARD_COUNT,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_qwen36_full")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--whisper", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=EXPECTED_SHARD_COUNT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-root", action="append", type=Path)
    parser.add_argument("--allow-unverified-contract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verify_contract = not args.allow_unverified_contract
    if args.aggregate:
        result = aggregate_shards(args.shard_root or [], output_root=args.output, shard_count=args.shard_count)
    else:
        if args.requests is None or args.whisper is None or args.media_root is None:
            raise SystemExit("--requests, --whisper, and --media-root are required")
        if args.preflight:
            result = preflight(args.requests, args.whisper, args.media_root, verify_contract=verify_contract)
        else:
            if args.shard_index is None:
                raise SystemExit("--shard-index is required for --run")
            result = run_shard(
                requests_path=args.requests,
                whisper_path=args.whisper,
                media_root=args.media_root,
                output_root=args.output,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                limit=args.limit,
                server_url=args.server_url,
                verify_contract=verify_contract,
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
