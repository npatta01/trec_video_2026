#!/usr/bin/env python3
"""Build frozen direct-video MC rankings and derive V2/V3 submissions.

The inference request is intentionally blind to the Task 1 answer, V1 option
ranking, weak-row audit, and hidden labels.  Provider-mutating transitions are
added only behind explicit CLI commands; all helpers in this first slice are
provider-free and testable.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from google.genai import types
from google import genai

import run_vqa_deadline_gemini_full as deadline
from trec_video_2026 import vqa_gemini37_flash_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.7-flash"
MODEL_ARM = "native-av-lowres-4fps-high-thinking-direct-options"
VIDEO_FPS = 4.0
MAX_OUTPUT_TOKENS = 2048
BATCH_INPUT_USD_PER_MILLION = 0.375
BATCH_OUTPUT_USD_PER_MILLION = 1.875
HISTORICAL_READY_ROWS = 2_433
HISTORICAL_VIDEO_SECONDS = 85_407.454
HISTORICAL_PROMPT_TOKENS = 24_712_790
HISTORICAL_OUTPUT_TOKENS = 1_078_734
MAX_INLINE_BATCH_BYTES = 20_000_000
TASK2_COMMIT = "34f1d1e"
TASK2_GIT_PATH = "datasets/trecvid-2026/vqa/task2/vqa.2026.testingDataset.Task2.json"
REQUESTS_PATH = ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
MEDIA_ROOT = ROOT / "data/vqa/2026/testing"
V1_PATH = ROOT / "data/vqa/benchmarks/vqa-mc-text-rank1-20260822-a1/submission.json"
WEAK_QUEUE_PATH = (
    ROOT
    / "data/vqa/benchmarks/vqa-mc-text-audit-20260822-a1/video-review-queue.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2-v3-20260822-a1"


def _json_source(source: Path | bytes) -> Any:
    try:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        return json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON source") from error


def load_panel(
    *,
    task2: Path | bytes,
    requests_path: Path,
    media_root: Path,
    expected_count: int | None = 2_443,
) -> list[dict[str, Any]]:
    """Align released options with the canonical label-free media requests."""

    task_rows = _json_source(task2)
    if not isinstance(task_rows, list) or not all(
        isinstance(row, Mapping) for row in task_rows
    ):
        raise ValueError("Task 2 must be an object list")
    media_rows = deadline.load_input_rows(
        Path(requests_path),
        media_root=Path(media_root),
        expected_count=expected_count,
    )
    if expected_count is not None and len(task_rows) != expected_count:
        raise ValueError(f"Task 2 must contain {expected_count} rows")
    if len(task_rows) != len(media_rows):
        raise ValueError("Task 2 and media request counts differ")

    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for task_row, media_row in zip(task_rows, media_rows, strict=True):
        query_id = task_row.get("Q_ID")
        if (
            isinstance(query_id, bool)
            or not isinstance(query_id, int)
            or query_id <= 0
            or query_id in seen
        ):
            raise ValueError("Task 2 Q_ID coverage is invalid")
        seen.add(query_id)
        if query_id != media_row["query_id"]:
            raise ValueError(f"query order mismatch for Q_ID {query_id}")
        video_id = task_row.get("Video_ID")
        question = task_row.get("Question")
        if video_id != media_row["video_id"]:
            raise ValueError(f"video identity mismatch for Q_ID {query_id}")
        if not isinstance(question, str) or question.strip() != media_row["question"]:
            raise ValueError(f"question mismatch for Q_ID {query_id}")
        raw_options = task_row.get("Options")
        if not isinstance(raw_options, list) or not 2 <= len(raw_options) <= 11:
            raise ValueError(f"invalid options for Q_ID {query_id}")
        options: list[str] = []
        for option in raw_options:
            text = option.get("option") if isinstance(option, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"invalid option for Q_ID {query_id}")
            options.append(text)
        if len(options) != len(set(options)):
            raise ValueError(f"duplicate option for Q_ID {query_id}")
        normalized = dict(media_row)
        normalized.update(
            {
                "query_id": query_id,
                "video_id": video_id,
                "question": question.strip(),
                "options": options,
            }
        )
        output.append(normalized)
    return output


def _prompt(row: Mapping[str, Any]) -> str:
    option_lines = "\n".join(
        f"{index}. {option}"
        for index, option in enumerate(row["options"], start=1)
    )
    return (
        "Watch and listen to the complete video. Answer the question using only "
        "evidence from the video. Rank every supplied answer option from most likely "
        "to least likely. If no wording is perfect, still choose the best available "
        "option first. Return only JSON with one field named ranking. The ranking must "
        "contain every numeric option ID exactly once and no other value.\n\n"
        f"Question: {row['question']}\nOptions:\n{option_lines}"
    )


def _response_schema(option_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ranking": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": option_count,
                },
                "minItems": option_count,
                "maxItems": option_count,
            }
        },
        "required": ["ranking"],
    }


def build_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any]
) -> types.InlinedRequest:
    """Build one ranking-only native-AV request with no prior-answer fields."""

    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 11:
        raise ValueError("row options are invalid")
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type")
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload file_uri is invalid")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload mime_type is invalid")
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
        ),
    )
    return types.InlinedRequest(
        model=MODEL,
        contents=[
            types.Content(
                role="user", parts=[video, types.Part(text=_prompt(row))]
            )
        ],
        metadata={
            "arm": MODEL_ARM,
            "query_id": str(row["query_id"]),
            "video_id": str(row["video_id"]),
            "protocol_sha256": str(row.get("input_sha256", "")),
        },
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=_response_schema(len(options)),
            seed=42,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=types.ThinkingLevel.HIGH,
            ),
        ),
    )


def parse_ranking(payload: object, *, option_count: int) -> list[int]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("ranking is invalid JSON") from error
    if not isinstance(payload, Mapping) or set(payload) != {"ranking"}:
        raise ValueError("ranking response shape is invalid")
    ranking = payload["ranking"]
    if (
        not isinstance(ranking, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ranking)
        or sorted(ranking) != list(range(1, option_count + 1))
    ):
        raise ValueError("ranking must be a complete option permutation")
    return list(ranking)


def _v1_index(
    submission: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    if len(submission) != len(rows):
        raise ValueError("V1 and panel counts differ")
    indexed: dict[int, dict[str, Any]] = {}
    for source, row in zip(submission, rows, strict=True):
        query_id = source.get("Q_ID")
        if query_id != row["query_id"] or source.get("Video_ID") != row["video_id"]:
            raise ValueError("V1 identity differs from panel")
        answers = source.get("Answers")
        expected = row["options"]
        if not isinstance(answers, list) or sorted(
            answer.get("Answer") for answer in answers if isinstance(answer, Mapping)
        ) != sorted(expected):
            raise ValueError("V1 options differ from Task 2")
        indexed[int(query_id)] = dict(source)
    return indexed


def _direct_submission_row(
    row: Mapping[str, Any], ranking: Sequence[int]
) -> dict[str, Any]:
    options = row["options"]
    return {
        "Q_ID": row["query_id"],
        "Video_ID": row["video_id"],
        "Answers": [
            {"Rank": rank, "Answer": options[option_id - 1]}
            for rank, option_id in enumerate(ranking, start=1)
        ],
    }


def assemble_submissions(
    *,
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Mapping[str, Any]],
    v1_submission: Sequence[Mapping[str, Any]],
    weak_query_ids: set[int],
) -> dict[str, Any]:
    """Derive V2 and V3 from one direct-video prediction freeze."""

    v1 = _v1_index(v1_submission, rows)
    if not weak_query_ids <= set(v1):
        raise ValueError("weak Q_ID set is outside the panel")
    v2: list[dict[str, Any]] = []
    v3: list[dict[str, Any]] = []
    v2_direct = 0
    v3_direct = 0
    v3_fallback = 0
    for row in rows:
        query_id = int(row["query_id"])
        prediction = predictions.get(query_id, {})
        direct = prediction.get("status") == "passed"
        direct_row = None
        if direct:
            ranking = parse_ranking(
                {"ranking": prediction.get("ranking")},
                option_count=len(row["options"]),
            )
            direct_row = _direct_submission_row(row, ranking)
        if query_id in weak_query_ids and direct_row is not None:
            v2.append(direct_row)
            v2_direct += 1
        else:
            v2.append(v1[query_id])
        if direct_row is not None:
            v3.append(direct_row)
            v3_direct += 1
        else:
            v3.append(v1[query_id])
            v3_fallback += 1
    return {
        "v2": v2,
        "v3": v3,
        "v2_direct_count": v2_direct,
        "v2_fallback_count": len(rows) - v2_direct,
        "v3_direct_count": v3_direct,
        "v3_fallback_count": v3_fallback,
    }


def estimate_cost(
    *,
    ready_rows: int,
    ready_video_seconds: float,
    historical_prompt_tokens: int = HISTORICAL_PROMPT_TOKENS,
    historical_output_tokens: int = HISTORICAL_OUTPUT_TOKENS,
) -> dict[str, Any]:
    if ready_rows < 0 or not math.isfinite(ready_video_seconds) or ready_video_seconds < 0:
        raise ValueError("ready panel size is invalid")
    prompt_scale = ready_video_seconds / HISTORICAL_VIDEO_SECONDS
    output_scale = ready_rows / HISTORICAL_READY_ROWS
    prompt_tokens = math.ceil(historical_prompt_tokens * prompt_scale)
    output_tokens = math.ceil(historical_output_tokens * output_scale)
    projected_cost = (
        prompt_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * BATCH_OUTPUT_USD_PER_MILLION
    )
    return {
        "basis": "matched_prior_full_native_av_observed_usage",
        "ready_rows": ready_rows,
        "ready_video_seconds": round(ready_video_seconds, 3),
        "projected_prompt_tokens": prompt_tokens,
        "projected_output_tokens": output_tokens,
        "projected_cost_usd": round(projected_cost, 8),
        "batch_input_usd_per_million": BATCH_INPUT_USD_PER_MILLION,
        "batch_output_usd_per_million": BATCH_OUTPUT_USD_PER_MILLION,
    }


def assert_cost_gate(estimate: Mapping[str, Any], *, max_cost_usd: float) -> None:
    projected = estimate.get("projected_cost_usd")
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
        or isinstance(projected, bool)
        or not isinstance(projected, (int, float))
        or not math.isfinite(projected)
        or projected > max_cost_usd
    ):
        raise ValueError("projected Gemini Batch cost exceeds cost cap")


def limit_inputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    v1_submission: Sequence[Mapping[str, Any]],
    weak_query_ids: set[int],
    query_limit: int | None,
) -> dict[str, Any]:
    if query_limit is None:
        selected_rows = [dict(row) for row in rows]
        selected_v1 = [dict(row) for row in v1_submission]
    else:
        if isinstance(query_limit, bool) or not isinstance(query_limit, int) or query_limit <= 0:
            raise ValueError("query limit must be a positive integer")
        selected_rows = [dict(row) for row in rows[:query_limit]]
        selected_v1 = [dict(row) for row in v1_submission[:query_limit]]
    _v1_index(selected_v1, selected_rows)
    selected_ids = {int(row["query_id"]) for row in selected_rows}
    return {
        "rows": selected_rows,
        "v1_submission": selected_v1,
        "weak_query_ids": set(weak_query_ids) & selected_ids,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_value_once(path: Path, value: Any) -> str:
    serialized = _canonical(value) + b"\n"
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != serialized:
            raise ValueError(f"immutable artifact changed: {path.name}")
        return _sha_bytes(serialized)
    path.write_bytes(serialized)
    return _sha_bytes(serialized)


def _read_value(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is missing: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError(f"artifact is invalid: {path.name}") from error


def _row_protocol_hash(row: Mapping[str, Any]) -> str:
    return _sha_bytes(
        _canonical(
            {
                "query_id": row["query_id"],
                "video_id": row["video_id"],
                "question": row["question"],
                "options": row["options"],
                "status": row["status"],
                "media_sha256": row.get("media_sha256"),
            }
        )
    )


def prepare_run(
    *,
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    v1_submission: Sequence[Mapping[str, Any]],
    weak_query_ids: set[int],
    task2_sha256: str,
    v1_sha256: str,
    weak_queue_sha256: str,
    max_cost_usd: float,
    created_at: str,
    experiment_id: str = "vqa-mc-gemini-direct-v2-v3-20260822",
) -> dict[str, Any]:
    """Freeze a private provider-free protocol before upload or inference."""

    root = Path(root)
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment ID is invalid")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"output already exists: {root}")
    for digest in (task2_sha256, v1_sha256, weak_queue_sha256):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("source SHA-256 is invalid")
    mutable_rows = [dict(row) for row in rows]
    _v1_index(v1_submission, mutable_rows)
    if not weak_query_ids <= {int(row["query_id"]) for row in mutable_rows}:
        raise ValueError("weak Q_ID set is outside the panel")

    root.mkdir(parents=True)
    for name in ("uploads", "jobs", "status", "deliveries"):
        (root / name).mkdir()
    deadline._prepare_deliveries(mutable_rows, root / "deliveries")
    for row in mutable_rows:
        row["input_sha256"] = _row_protocol_hash(row)
    ready = [row for row in mutable_rows if row["status"] == "ready"]
    estimate = estimate_cost(
        ready_rows=len(ready),
        ready_video_seconds=sum(float(row["duration_seconds"]) for row in ready),
    )
    assert_cost_gate(estimate, max_cost_usd=max_cost_usd)
    protocol = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "prepared",
        "created_at": created_at,
        "model": MODEL,
        "model_arm": MODEL_ARM,
        "settings": {
            "video_fps": VIDEO_FPS,
            "media_resolution": "MEDIA_RESOLUTION_LOW",
            "thinking_level": "HIGH",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "source": {
            "task2_sha256": task2_sha256,
            "v1_sha256": v1_sha256,
            "weak_queue_sha256": weak_queue_sha256,
        },
        "weak_query_ids": sorted(weak_query_ids),
        "estimate": estimate,
        "max_cost_usd": max_cost_usd,
        "rows": mutable_rows,
    }
    deadline._write_json(root / "protocol.json", protocol)
    v1_artifact_sha256 = _write_value_once(root / "v1-submission.json", list(v1_submission))
    manifest = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "status": "prepared",
        "created_at": created_at,
        "protocol_sha256": deadline._sha256_file(root / "protocol.json"),
        "v1_artifact_sha256": v1_artifact_sha256,
        "row_count": len(mutable_rows),
        "ready_count": len(ready),
        "unavailable_count": sum(row["status"] == "unavailable" for row in mutable_rows),
        "unsupported_count": sum(row["status"] == "unsupported" for row in mutable_rows),
        "unique_ready_video_count": len({str(row["video_id"]) for row in ready}),
        "estimate": estimate,
        "max_cost_usd": max_cost_usd,
    }
    deadline._write_json(root / "manifest.json", manifest)
    return {
        "status": "prepared",
        "row_count": len(mutable_rows),
        "ready_count": len(ready),
        "unavailable_count": manifest["unavailable_count"],
        "projected_cost_usd": estimate["projected_cost_usd"],
    }


def _protocol(root: Path) -> dict[str, Any]:
    root = Path(root)
    protocol = deadline._read_json(root / "protocol.json")
    manifest = deadline._read_json(root / "manifest.json")
    if manifest.get("protocol_sha256") != deadline._sha256_file(root / "protocol.json"):
        raise ValueError("manifest protocol binding mismatch")
    if protocol.get("model") != MODEL or protocol.get("model_arm") != MODEL_ARM:
        raise ValueError("protocol model binding mismatch")
    rows = protocol.get("rows")
    if not isinstance(rows, list):
        raise ValueError("protocol rows are missing")
    return protocol


def upload_run(
    *, root: Path, client: Any, sleep: Any
) -> dict[str, Any]:
    """Resume the bounded upload pool without repeating terminal uploads."""

    root = Path(root)
    protocol = _protocol(root)
    ready_by_video: dict[str, Mapping[str, Any]] = {}
    for row in protocol["rows"]:
        if row.get("status") == "ready":
            ready_by_video.setdefault(str(row["video_id"]), row)
    pending: list[tuple[str, Mapping[str, Any]]] = []
    uploaded_count = 0
    failed_count = 0
    for video_id, row in ready_by_video.items():
        terminal = root / "uploads" / f"{video_id}.json"
        if terminal.exists():
            record = deadline._read_json(terminal)
            if record.get("media_sha256") != row.get("media_sha256"):
                raise ValueError(f"upload media hash changed: {video_id}")
            if record.get("status") == "uploaded":
                uploaded_count += 1
            elif record.get("status") == "upload_failed":
                failed_count += 1
            else:
                raise ValueError(f"upload record is not terminal: {video_id}")
            continue
        if (root / "uploads" / f"{video_id}.attempt.json").exists():
            raise ValueError(f"ambiguous upload attempt: {video_id}")
        pending.append((video_id, row))
    records = deadline._upload_pending(
        pending=pending,
        client=client,
        output_root=root,
        sleep=sleep,
    )
    uploaded_count += sum(record["status"] == "uploaded" for record in records.values())
    failed_count += sum(record["status"] == "upload_failed" for record in records.values())
    return {
        "status": "uploaded",
        "uploaded_count": uploaded_count,
        "failed_count": failed_count,
    }


def _terminal_uploads(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_video = {
        str(row["video_id"]): row for row in rows if row.get("status") == "ready"
    }
    result: dict[str, dict[str, Any]] = {}
    for video_id, row in by_video.items():
        path = root / "uploads" / f"{video_id}.json"
        record = deadline._read_json(path)
        if record.get("media_sha256") != row.get("media_sha256"):
            raise ValueError(f"upload media hash changed: {video_id}")
        if record.get("status") not in {"uploaded", "upload_failed"}:
            raise ValueError(f"upload is not terminal: {video_id}")
        result[video_id] = record
    return result


def submit_batch(
    *, root: Path, client: Any, submitted_at: str
) -> dict[str, Any]:
    """Create exactly one inline Batch job after reserving its immutable attempt."""

    root = Path(root)
    protocol = _protocol(root)
    job_path = root / "jobs" / "direct-options.json"
    attempt_path = root / "jobs" / "direct-options.attempt.json"
    if job_path.exists() or attempt_path.exists():
        raise ValueError("batch submission already exists")
    rows = protocol["rows"]
    uploads = _terminal_uploads(root, rows)
    request_rows = [
        row
        for row in rows
        if row.get("status") == "ready"
        and uploads[str(row["video_id"])].get("status") == "uploaded"
    ]
    requests = [
        build_request(
            row=row,
            upload={
                "file_uri": uploads[str(row["video_id"])]["file_uri"],
                "mime_type": uploads[str(row["video_id"])]["mime_type"],
            },
        )
        for row in request_rows
    ]
    request_dumps = [
        request.model_dump(mode="json", by_alias=True, exclude_none=True)
        for request in requests
    ]
    request_bytes = _canonical(request_dumps)
    if len(request_bytes) >= MAX_INLINE_BATCH_BYTES:
        raise ValueError("inline Batch request exceeds 20MB limit")
    request_hash = _sha_bytes(request_bytes)
    request_records = [
        {
            "query_id": row["query_id"],
            "video_id": row["video_id"],
            "input_sha256": row["input_sha256"],
            "request_sha256": _sha_bytes(_canonical(dump)),
        }
        for row, dump in zip(request_rows, request_dumps, strict=True)
    ]
    _write_value_once(root / "requests.json", request_records)
    config = types.CreateBatchJobConfig(
        display_name="trecvid-2026-vqa-mc-direct-options-v2-v3"
    )
    attempt = {
        "schema_version": 1,
        "status": "attempt_reserved",
        "submitted_at": submitted_at,
        "model": MODEL,
        "display_name": config.display_name,
        "request_count": len(requests),
        "request_bytes": len(request_bytes),
        "request_hash": request_hash,
        "requests_artifact_sha256": deadline._sha256_file(root / "requests.json"),
        "protocol_sha256": deadline._sha256_file(root / "protocol.json"),
    }
    deadline._write_json(attempt_path, attempt)
    attempt_sha256 = deadline._sha256_file(attempt_path)
    try:
        job = client.batches.create(
            model=MODEL,
            src=types.BatchJobSource(inlined_requests=requests),
            config=config,
        )
        snapshot = smoke._job_snapshot(job)
        name = snapshot.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("provider Batch job name is missing")
        record = {
            "schema_version": 1,
            "status": "submitted",
            "submitted_at": submitted_at,
            "model": MODEL,
            "name": name,
            "request_count": len(requests),
            "request_hash": request_hash,
            "attempt_sha256": attempt_sha256,
            "provider_snapshot": snapshot,
        }
    except Exception as error:  # noqa: BLE001 - freeze one terminal create attempt
        record = {
            "schema_version": 1,
            "status": "failed",
            "submitted_at": submitted_at,
            "model": MODEL,
            "request_count": len(requests),
            "request_hash": request_hash,
            "attempt_sha256": attempt_sha256,
            "error": smoke._safe_error(error),
        }
    deadline._write_json(job_path, record)
    return {
        "status": record["status"],
        "request_count": len(requests),
        "request_bytes": len(request_bytes),
    }


def record_status(
    *, root: Path, client: Any, checked_at: str
) -> dict[str, Any]:
    """Append one sanitized status snapshot for the sole frozen Batch job."""

    root = Path(root)
    _protocol(root)
    if (root / "predictions-freeze.json").exists():
        raise ValueError("prediction freeze is terminal")
    job_record = deadline._read_json(root / "jobs" / "direct-options.json")
    if job_record.get("status") != "submitted":
        raise ValueError("Batch job was not submitted")
    name = job_record.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Batch job name is missing")
    job = client.batches.get(name=name)
    snapshot = smoke._job_snapshot(job)
    responses = smoke._extract_responses(job)
    snapshot["responses"] = [smoke._safe_response_object(row) for row in responses]
    state = smoke._state(getattr(job, "state", snapshot.get("state", "")))
    status_root = root / "status"
    sequence = len(list(status_root.glob("*.json"))) + 1
    body = {
        "schema_version": 1,
        "status": state,
        "checked_at": checked_at,
        "job_sha256": deadline._sha256_file(root / "jobs" / "direct-options.json"),
        "provider_snapshot": snapshot,
        "response_count": len(responses),
    }
    deadline._write_json(status_root / f"{sequence:04d}.json", body)
    return {"status": state, "response_count": len(responses)}


def _latest_status(root: Path) -> dict[str, Any]:
    paths = sorted((Path(root) / "status").glob("*.json"))
    if not paths:
        raise ValueError("no Batch status exists")
    status = deadline._read_json(paths[-1])
    if status.get("job_sha256") != deadline._sha256_file(
        Path(root) / "jobs" / "direct-options.json"
    ):
        raise ValueError("status job binding mismatch")
    return status


def _usage_cost(usage: Mapping[str, Any] | None) -> tuple[int, int, float] | None:
    if usage is None:
        return None
    prompt = usage.get("prompt_tokens")
    total = usage.get("total_tokens")
    if (
        isinstance(prompt, bool)
        or not isinstance(prompt, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total < prompt
    ):
        return None
    billed_output = total - prompt
    cost = (
        prompt / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
        + billed_output / 1_000_000 * BATCH_OUTPUT_USD_PER_MILLION
    )
    return prompt, billed_output, cost


def freeze_run(*, root: Path, frozen_at: str) -> dict[str, Any]:
    """Parse terminal responses and immutably derive both complete submissions."""

    root = Path(root)
    existing_freeze_path = root / "predictions-freeze.json"
    if existing_freeze_path.exists():
        existing_freeze = deadline._read_json(existing_freeze_path)
        existing_frozen_at = existing_freeze.get("frozen_at")
        if not isinstance(existing_frozen_at, str) or not existing_frozen_at:
            raise ValueError("existing prediction freeze timestamp is invalid")
        frozen_at = existing_frozen_at
    protocol = _protocol(root)
    status = _latest_status(root)
    if status.get("status") != "succeeded":
        raise ValueError("Batch job is not succeeded")
    provider = status.get("provider_snapshot")
    responses = provider.get("responses") if isinstance(provider, Mapping) else None
    if not isinstance(responses, list):
        raise ValueError("Batch responses are missing")
    request_records = _read_value(root / "requests.json")
    if not isinstance(request_records, list) or len(request_records) != len(responses):
        raise ValueError("Batch response count differs from requests")

    rows = protocol["rows"]
    by_id = {int(row["query_id"]): row for row in rows}
    responses_by_query_id: dict[int, Mapping[str, Any]] = {}
    for raw in responses:
        payload = smoke._response_payload(raw)
        metadata = payload.get("metadata")
        raw_query_id = metadata.get("query_id") if isinstance(metadata, Mapping) else None
        try:
            response_query_id = int(raw_query_id)
        except (TypeError, ValueError) as error:
            raise ValueError("response metadata Q_ID is missing") from error
        if (
            isinstance(raw_query_id, bool)
            or response_query_id not in by_id
            or response_query_id in responses_by_query_id
        ):
            raise ValueError("response metadata Q_ID coverage is invalid")
        responses_by_query_id[response_query_id] = raw
    expected_response_ids = {
        record.get("query_id") for record in request_records if isinstance(record, Mapping)
    }
    if set(responses_by_query_id) != expected_response_ids:
        raise ValueError("Batch response Q_ID coverage differs from requests")
    predictions: dict[int, dict[str, Any]] = {}
    prompt_tokens = 0
    billed_output_tokens = 0
    observed_cost = 0.0
    usage_complete = True
    response_ids: set[str] = set()
    for record in request_records:
        query_id = record.get("query_id") if isinstance(record, Mapping) else None
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id not in by_id:
            raise ValueError("request record Q_ID is invalid")
        raw = responses_by_query_id[query_id]
        payload = smoke._response_payload(raw)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("query_id") != str(query_id):
            raise ValueError(f"response metadata mismatch for Q_ID {query_id}")
        response_id = payload.get("response_id", payload.get("responseId", payload.get("id")))
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise ValueError("response identity is missing or duplicated")
        response_ids.add(response_id)
        text = smoke._response_text(payload)
        ranking = None
        row_status = "invalid"
        error = None
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("response text is empty")
            ranking = parse_ranking(text, option_count=len(by_id[query_id]["options"]))
            row_status = "passed"
        except ValueError as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        usage = smoke._usage(payload)
        cost = _usage_cost(usage)
        if cost is None:
            usage_complete = False
        else:
            prompt_tokens += cost[0]
            billed_output_tokens += cost[1]
            observed_cost += cost[2]
        predictions[query_id] = {
            "query_id": query_id,
            "video_id": by_id[query_id]["video_id"],
            "status": row_status,
            "ranking": ranking,
            "response_id": response_id,
            "model_version": payload.get("model_version", payload.get("modelVersion")),
            "usage": usage,
            "error": error,
        }

    uploads = _terminal_uploads(root, rows)
    for row in rows:
        query_id = int(row["query_id"])
        if query_id in predictions:
            continue
        status_name = str(row["status"])
        if status_name == "ready":
            status_name = str(uploads[str(row["video_id"])]["status"])
        predictions[query_id] = {
            "query_id": query_id,
            "video_id": row["video_id"],
            "status": status_name,
            "ranking": None,
            "response_id": None,
            "model_version": None,
            "usage": None,
            "error": None,
        }
    ordered_predictions = [predictions[int(row["query_id"])] for row in rows]
    prediction_freeze = {
        "schema_version": 1,
        "status": "predictions_frozen",
        "frozen_at": frozen_at,
        "model": MODEL,
        "protocol_sha256": deadline._sha256_file(root / "protocol.json"),
        "status_sha256": deadline._sha256_file(
            sorted((root / "status").glob("*.json"))[-1]
        ),
        "row_count": len(rows),
        "passed_count": sum(row["status"] == "passed" for row in ordered_predictions),
        "fallback_count": sum(row["status"] != "passed" for row in ordered_predictions),
        "usage_complete": usage_complete,
        "prompt_tokens": prompt_tokens,
        "billed_output_tokens_including_reasoning": billed_output_tokens,
        "observed_cost_usd": round(observed_cost, 8) if usage_complete else None,
        "rows": ordered_predictions,
    }
    deadline._write_json_once(root / "predictions-freeze.json", prediction_freeze)

    v1 = _read_value(root / "v1-submission.json")
    if not isinstance(v1, list):
        raise ValueError("V1 artifact is invalid")
    weak = protocol.get("weak_query_ids")
    if not isinstance(weak, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in weak
    ):
        raise ValueError("weak Q_ID artifact is invalid")
    assembled = assemble_submissions(
        rows=rows,
        predictions=predictions,
        v1_submission=v1,
        weak_query_ids=set(weak),
    )
    v2_sha256 = _write_value_once(root / "submission-v2.json", assembled["v2"])
    v3_sha256 = _write_value_once(root / "submission-v3.json", assembled["v3"])
    summary = {
        "schema_version": 1,
        "status": "complete",
        "frozen_at": frozen_at,
        "row_count": len(rows),
        "passed_count": prediction_freeze["passed_count"],
        "fallback_count": prediction_freeze["fallback_count"],
        "v2_direct_count": assembled["v2_direct_count"],
        "v2_fallback_count": assembled["v2_fallback_count"],
        "v3_direct_count": assembled["v3_direct_count"],
        "v3_fallback_count": assembled["v3_fallback_count"],
        "prompt_tokens": prompt_tokens,
        "billed_output_tokens_including_reasoning": billed_output_tokens,
        "observed_cost_usd": prediction_freeze["observed_cost_usd"],
        "predictions_freeze_sha256": deadline._sha256_file(
            root / "predictions-freeze.json"
        ),
        "submission_v2_sha256": v2_sha256,
        "submission_v3_sha256": v3_sha256,
    }
    deadline._write_json_once(root / "summary.json", summary)
    return summary


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _task2_bytes() -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{TASK2_COMMIT}:{TASK2_GIT_PATH}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("frozen Task 2 git object is unavailable") from error


def _weak_query_ids(path: Path = WEAK_QUEUE_PATH) -> set[int]:
    result: set[int] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("weak queue is unavailable") from error
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("weak queue is invalid") from error
        query_id = row.get("Q_ID") if isinstance(row, Mapping) else None
        if (
            isinstance(query_id, bool)
            or not isinstance(query_id, int)
            or query_id <= 0
            or query_id in result
        ):
            raise ValueError("weak queue Q_ID coverage is invalid")
        result.add(query_id)
    if len(result) != 818:
        raise ValueError("weak queue must contain 818 rows")
    return result


def _client() -> Any:
    return genai.Client(api_key=deadline._private_api_key())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_mc_gemini_direct_full")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--query-limit", type=int)
    prepare.add_argument("--max-cost-usd", type=float, required=True)
    for name in ("upload", "submit", "status", "freeze", "show"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _prepare_cli(*, output: Path, query_limit: int | None, max_cost_usd: float) -> dict[str, Any]:
    task2 = _task2_bytes()
    rows = load_panel(
        task2=task2,
        requests_path=REQUESTS_PATH,
        media_root=MEDIA_ROOT,
    )
    v1 = _json_source(V1_PATH)
    if not isinstance(v1, list):
        raise ValueError("V1 submission is invalid")
    selected = limit_inputs(
        rows=rows,
        v1_submission=v1,
        weak_query_ids=_weak_query_ids(),
        query_limit=query_limit,
    )
    return prepare_run(
        rows=selected["rows"],
        root=output,
        v1_submission=selected["v1_submission"],
        weak_query_ids=selected["weak_query_ids"],
        task2_sha256=_sha_bytes(task2),
        v1_sha256=deadline._sha256_file(V1_PATH),
        weak_queue_sha256=deadline._sha256_file(WEAK_QUEUE_PATH),
        max_cost_usd=max_cost_usd,
        created_at=_now(),
    )


def _emit(result: Mapping[str, Any]) -> None:
    allowed = (
        "status",
        "row_count",
        "ready_count",
        "unavailable_count",
        "uploaded_count",
        "failed_count",
        "request_count",
        "request_bytes",
        "response_count",
        "passed_count",
        "fallback_count",
        "v2_direct_count",
        "v3_direct_count",
        "projected_cost_usd",
        "observed_cost_usd",
    )
    print(json.dumps({key: result[key] for key in allowed if key in result}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "prepare":
            result = _prepare_cli(
                output=args.output,
                query_limit=args.query_limit,
                max_cost_usd=args.max_cost_usd,
            )
        elif args.command == "upload":
            result = upload_run(root=args.output, client=_client(), sleep=time.sleep)
        elif args.command == "submit":
            result = submit_batch(root=args.output, client=_client(), submitted_at=_now())
        elif args.command == "status":
            result = record_status(root=args.output, client=_client(), checked_at=_now())
        elif args.command == "freeze":
            result = freeze_run(root=args.output, frozen_at=_now())
        elif args.command == "show":
            summary_path = args.output / "summary.json"
            if summary_path.exists():
                result = deadline._read_json(summary_path)
            else:
                result = _protocol(args.output)
        else:  # pragma: no cover
            raise ValueError("invalid command")
        _emit(result)
        return 0
    except Exception:  # noqa: BLE001 - CLI boundary must not leak private evidence
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
