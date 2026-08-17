#!/usr/bin/env python3
"""Restartable, one-answer-per-query Gemini native-AV full-panel runner.

The command is deliberately dry-run by default.  A live run requires the
explicit ``--run`` flag and writes only private, immutable request/response
artifacts under the selected output root.  No reference or annotation path is
accepted by this runner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from trec_video_2026 import vqa_gemini37_flash_smoke as smoke


_REPO_ROOT = Path(__file__).resolve().parents[1]
REQUESTS_PATH = (
    _REPO_ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
)
MEDIA_ROOT = _REPO_ROOT / "data/vqa/2026/testing"
PRIVATE_RUN_ROOT = (
    _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-full-20260816/gemini"
)
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")

MODEL = "gemini-3.7-flash"
PROMPT = smoke.DIRECT_ANSWER_PROMPT
MAX_OUTPUT_TOKENS = 2048
EXPECTED_QUERY_COUNT = 2_443
MAX_ESTIMATED_COST_USD = 50.0
MAX_OBSERVED_COST_USD = 50.0
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75
# The promoted five-row profile observed 289 prompt tokens/video-second.
# Keep a 56% input margin while avoiding the disproven HIGH-resolution ceiling.
ESTIMATED_INPUT_TOKENS_PER_SECOND = 450
ESTIMATED_QUESTION_TOKENS = 100
VIDEO_FPS = 4.0
MEDIA_RESOLUTION = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
THINKING_LEVEL = types.ThinkingLevel.HIGH
MAX_UPLOAD_POLLS = 30
UPLOAD_POLL_SECONDS = 1.0
# After a ten-minute cooldown and 55 clean serial uploads, use two workers for
# the resumable remainder.  The exact-operation 503 retry below bounds an
# isolated provider overload while preserving deadline margin.
UPLOAD_WORKERS = 2
PROVIDER_RETRY_POLICY = {
    "max_retries": 1,
    "code": 503,
    "backoff_seconds": 10,
    "workers": UPLOAD_WORKERS,
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "label",
    "labels",
    "metric",
    "metrics",
    "reference",
    "references",
    "prior_output",
    "prior_outputs",
    "score",
    "scores",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return not stat.S_ISLNK(mode) and stat.S_ISREG(mode)


def _artifact_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("_artifact_sha256", None)
    return body


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["_artifact_sha256"] = _sha256_bytes(_canonical_json(body))
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return _sha256_bytes(serialized.encode("utf-8"))


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        existing = _read_json(path)
        if _artifact_body(existing) != dict(value):
            raise ValueError(f"immutable artifact changed: {path.name}")
        return _sha256_file(path)
    return _write_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    if not _is_regular_file(path):
        raise ValueError(f"artifact is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object: {path.name}")
    expected = value.get("_artifact_sha256")
    body = _artifact_body(value)
    if expected != _sha256_bytes(_canonical_json(body)):
        raise ValueError(f"artifact hash mismatch: {path.name}")
    return value


def _reject_forbidden(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            forbidden = {
                re.sub(r"[^a-z0-9]", "", item.casefold()) for item in _FORBIDDEN_KEYS
            }
            if normalized in forbidden:
                raise ValueError(f"forbidden private key {key!r} in {context}")
            _reject_forbidden(child, context=context)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_forbidden(child, context=context)


def resolve_media(video_id: str, *, media_root: Path = MEDIA_ROOT) -> Path | None:
    """Resolve one canonical testing video without consulting any labels."""

    if not _SAFE_NAME.fullmatch(video_id):
        raise ValueError(f"unsafe video_id: {video_id!r}")
    root = Path(media_root).resolve(strict=True)
    candidates = [
        root / "videos" / video_id,
        root / "videos" / f"{video_id}.webm",
        root / "videos" / f"{video_id}.mp4",
        root / "videos_original" / f"{video_id}.mkv",
        root / "videos_original" / f"{video_id}.mp4",
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if _is_regular_file(resolved):
            return resolved
    return None


def probe_delivery_plan(path: Path) -> dict[str, Any] | None:
    """Return a supported Gemini delivery plan, or ``None`` for a blocker."""

    suffix = path.suffix.casefold()
    if suffix == ".webm":
        return {
            "delivery_mode": "source",
            "delivery_suffix": ".webm",
            "delivery_mime_type": "video/webm",
        }
    if suffix == ".mp4":
        return {
            "delivery_mode": "source",
            "delivery_suffix": ".mp4",
            "delivery_mime_type": "video/mp4",
        }
    if suffix != ".mkv":
        return None
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams = payload.get("streams", [])
    codecs = {
        stream.get("codec_type"): stream.get("codec_name")
        for stream in streams
        if isinstance(stream, Mapping)
    }
    video_codec = codecs.get("video")
    audio_codec = codecs.get("audio")
    if video_codec in {"vp8", "vp9", "av1"} and audio_codec == "opus":
        return {
            "delivery_mode": "copy_webm",
            "delivery_suffix": ".webm",
            "delivery_mime_type": "video/webm",
            "video_codec": video_codec,
            "audio_codec": audio_codec,
        }
    if video_codec == "h264" and audio_codec in {"aac", "opus"}:
        return {
            "delivery_mode": "copy_mp4" if audio_codec == "aac" else "aac_mp4",
            "delivery_suffix": ".mp4",
            "delivery_mime_type": "video/mp4",
            "video_codec": video_codec,
            "audio_codec": audio_codec,
        }
    return None


def _duration_seconds(row: Mapping[str, Any]) -> float:
    timestamps = row.get("frame_timestamps")
    if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):
        return 0.0
    values: list[float] = []
    for value in timestamps:
        if isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed >= 0:
            values.append(parsed)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    step = max(values[1] - values[0], 0.0)
    # Twelve evenly spaced selected frames cover eleven intervals; this is the
    # same conservative duration estimator used by the frozen Qwen requests.
    return max(values) + step / 2.0


def load_input_rows(
    requests_path: Path = REQUESTS_PATH,
    *,
    media_root: Path = MEDIA_ROOT,
    expected_count: int | None = EXPECTED_QUERY_COUNT,
) -> list[dict[str, Any]]:
    """Load and integrity-check label-free requests plus canonical local media."""

    path = Path(requests_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _reject_forbidden(payload, context=path.name)
    if not isinstance(payload, list):
        raise ValueError("requests input must be a JSON array")
    if expected_count is not None and len(payload) != expected_count:
        raise ValueError(f"requests input must contain {expected_count} rows")

    rows: list[dict[str, Any]] = []
    seen_query_ids: set[int] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"request row {index} must be an object")
        query_id = raw.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id <= 0:
            raise ValueError(f"request row {index} query_id must be a positive integer")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        video_id = raw.get("video_id")
        question = raw.get("question")
        availability = raw.get("availability")
        if (
            not isinstance(video_id, str)
            or not video_id.strip()
            or not _SAFE_NAME.fullmatch(video_id)
        ):
            raise ValueError(f"request row {index} video_id is invalid")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"request row {index} question is missing")
        if availability not in {"available", "unavailable"}:
            raise ValueError(f"request row {index} availability is invalid")

        normalized: dict[str, Any] = {
            "query_id": query_id,
            "video_id": video_id,
            "question": question.strip(),
            "availability": availability,
            "duration_seconds": _duration_seconds(raw),
        }
        if availability == "unavailable":
            normalized.update(
                {
                    "status": "unavailable",
                    "reason": "input_unavailable",
                    "media_path": None,
                    "media_sha256": None,
                    "media_size_bytes": None,
                    "media_mime_type": None,
                }
            )
            rows.append(normalized)
            continue

        source_sha256 = raw.get("source_sha256")
        if not isinstance(source_sha256, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", source_sha256
        ):
            raise ValueError(f"request row {index} source_sha256 is invalid")
        media = resolve_media(video_id, media_root=Path(media_root))
        if media is None:
            normalized.update(
                {
                    "status": "unsupported",
                    "reason": "media_unavailable",
                    "media_path": None,
                    "media_sha256": None,
                    "media_size_bytes": None,
                    "media_mime_type": None,
                }
            )
            rows.append(normalized)
            continue
        actual_sha256 = _sha256_file(media)
        if actual_sha256.casefold() != source_sha256.casefold():
            raise ValueError(f"local media hash mismatch for {video_id}")
        delivery_plan = probe_delivery_plan(media)
        if delivery_plan is None:
            normalized.update(
                {
                    "status": "unsupported",
                    "reason": "unsupported_media_delivery",
                    "media_path": str(media),
                    "media_sha256": actual_sha256,
                    "media_size_bytes": media.stat().st_size,
                    "media_mime_type": None,
                    "source_media_path": str(media),
                    "source_media_sha256": actual_sha256,
                    "source_media_size_bytes": media.stat().st_size,
                    "delivery_plan": None,
                }
            )
            rows.append(normalized)
            continue
        normalized.update(
            {
                "status": "ready",
                "reason": None,
                "media_path": str(media),
                "media_sha256": actual_sha256,
                "media_size_bytes": media.stat().st_size,
                "media_mime_type": delivery_plan["delivery_mime_type"],
                "source_media_path": str(media),
                "source_media_sha256": actual_sha256,
                "source_media_size_bytes": media.stat().st_size,
                "delivery_plan": delivery_plan,
            }
        )
        rows.append(normalized)
    return rows


def estimate_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row.get("status") == "ready"]
    seconds = sum(float(row.get("duration_seconds", 0.0)) for row in ready)
    prompt_tokens = math.ceil(seconds * ESTIMATED_INPUT_TOKENS_PER_SECOND) + (
        len(ready) * ESTIMATED_QUESTION_TOKENS
    )
    output_tokens = len(ready) * MAX_OUTPUT_TOKENS
    cost = (
        prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    return {
        "duration_seconds": round(seconds, 3),
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def build_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any]
) -> types.InlinedRequest:
    """Build the promoted 4-FPS, LOW-resolution native-AV request."""

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question must be a non-empty string")
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type")
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload file_uri must be a non-empty string")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload mime_type must be a video MIME type")
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        media_resolution=types.PartMediaResolution(level=MEDIA_RESOLUTION),
    )
    prompt = types.Part(text=f"{PROMPT}\n\nQuestion: {question.strip()}")
    return types.InlinedRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[video, prompt])],
        metadata={
            "arm": "candidate_37flash_lowres_4fps_highthink",
            "row_id": str(row["query_id"]),
            "video_id": str(row["video_id"]),
            "question": question.strip(),
            "file_uri": file_uri,
            "delivery_sha256": str(row.get("media_sha256", "")),
            "mime_type": mime_type,
            "protocol_sha256": str(row.get("input_sha256", "")),
            "video_fps": str(VIDEO_FPS),
            "media_resolution": MEDIA_RESOLUTION.value,
            "thinking_level": THINKING_LEVEL.value,
        },
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
            seed=42,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=THINKING_LEVEL,
            ),
        ),
    )


def _is_retryable_provider_error(error: BaseException) -> bool:
    return (
        isinstance(error, genai_errors.ServerError)
        and type(getattr(error, "code", None)) is int
        and error.code == PROVIDER_RETRY_POLICY["code"]
    )


def _call_with_provider_retry(
    *,
    operation: Callable[[], Any],
    phase: str,
    retry_state: dict[str, Any],
    sleep: Callable[[float], None],
) -> Any:
    """Call one provider operation with one shared, exact-operation retry."""

    try:
        return operation()
    except Exception as error:  # noqa: BLE001 - classify only the approved 503
        if (
            retry_state["retry_count"] >= PROVIDER_RETRY_POLICY["max_retries"]
            or not _is_retryable_provider_error(error)
        ):
            raise
        retry_state["retry_count"] += 1
        first_error = smoke._safe_error(error) or {"type": type(error).__name__}
        first_error["code"] = PROVIDER_RETRY_POLICY["code"]
        retry_state["retry_first_error"] = first_error
        retry_state["retry_phase"] = phase
        retry_state["possible_orphan_duplicate"] = phase == "upload"
        sleep(PROVIDER_RETRY_POLICY["backoff_seconds"])
        return operation()


def _upload_once(
    *,
    client: Any,
    row: Mapping[str, Any],
    sleep: Callable[[float], None],
    retry_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if retry_state is None:
        retry_state = {
            "retry_count": 0,
            "retry_first_error": None,
            "retry_phase": None,
            "possible_orphan_duplicate": False,
        }
    media = Path(str(row["media_path"]))
    mime_type = str(row["media_mime_type"])
    before_hash = str(row["media_sha256"])
    before_size = int(row["media_size_bytes"])
    upload_config = types.UploadFileConfig(mime_type=mime_type)
    uploaded = _call_with_provider_retry(
        operation=lambda: client.files.upload(file=media, config=upload_config),
        phase="upload",
        retry_state=retry_state,
        sleep=sleep,
    )
    current = smoke._provider_file_metadata(uploaded, expected_mime=mime_type)
    if current.get("size_bytes") is not None and current["size_bytes"] != before_size:
        raise ValueError("provider upload size does not match canonical media")
    if current.get("sha256_hash") is not None and current["sha256_hash"] != before_hash:
        raise ValueError("provider upload hash does not match canonical media")
    for poll_index in range(MAX_UPLOAD_POLLS):
        if current["file_state"] == "ACTIVE":
            break
        if current["file_state"] == "failed":
            raise ValueError("provider upload entered FAILED state")
        if poll_index == MAX_UPLOAD_POLLS - 1:
            raise ValueError("provider upload timed out before ACTIVE")
        current = smoke._provider_file_metadata(
            _call_with_provider_retry(
                operation=lambda: client.files.get(name=current["name"]),
                phase="get",
                retry_state=retry_state,
                sleep=sleep,
            ),
            expected_mime=mime_type,
        )
        if current["file_state"] != "ACTIVE":
            sleep(UPLOAD_POLL_SECONDS)
    if _sha256_file(media) != before_hash or media.stat().st_size != before_size:
        raise ValueError("canonical media changed during upload")
    return {
        **current,
        "media_sha256": before_hash,
        "media_size_bytes": before_size,
        "mime_type": mime_type,
        "file_uri": current["uri"],
        "file_name": current["name"],
    }


def _prepare_deliveries(rows: list[dict[str, Any]], delivery_root: Path) -> None:
    """Materialize supported MP4/WebM deliveries before any provider call."""

    delivery_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("status") != "ready":
            continue
        source = Path(str(row["source_media_path"]))
        plan = row.get("delivery_plan")
        if not isinstance(plan, Mapping):
            raise ValueError(
                f"ready row has no supported delivery plan: {row['video_id']}"
            )
        if plan["delivery_mode"] == "source":
            delivery = source
        else:
            video_id = str(row["video_id"])
            delivery = delivery_root / f"{video_id}{plan['delivery_suffix']}"
            if not _is_regular_file(delivery):
                temporary = delivery.with_name(
                    f".{video_id}.partial{plan['delivery_suffix']}"
                )
                if temporary.exists() or temporary.is_symlink():
                    raise ValueError(f"partial media delivery exists: {temporary.name}")
                command = [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-xerror",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-map_metadata",
                    "-1",
                    "-map_chapters",
                    "-1",
                ]
                if plan["delivery_mode"] == "copy_webm":
                    command.extend(["-c:v", "copy", "-c:a", "copy"])
                elif plan["delivery_mode"] == "copy_mp4":
                    command.extend(
                        ["-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart"]
                    )
                elif plan["delivery_mode"] == "aac_mp4":
                    command.extend(
                        [
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "192k",
                            "-ar",
                            "48000",
                            "-ac",
                            "2",
                            "-movflags",
                            "+faststart",
                        ]
                    )
                else:
                    raise ValueError(f"unknown delivery plan: {plan['delivery_mode']}")
                command.extend(["-shortest", str(temporary)])
                try:
                    subprocess.run(command, check=True)
                    if not _is_regular_file(temporary):
                        raise ValueError(
                            f"ffmpeg produced no delivery: {delivery.name}"
                        )
                    os.replace(temporary, delivery)
                except Exception:
                    if temporary.exists() and not temporary.is_symlink():
                        temporary.unlink()
                    raise
        if not _is_regular_file(delivery):
            raise ValueError(f"delivery is missing: {delivery}")
        delivery_sha256 = _sha256_file(delivery)
        row["media_path"] = str(delivery)
        row["media_sha256"] = delivery_sha256
        row["media_size_bytes"] = delivery.stat().st_size
        row["media_mime_type"] = plan["delivery_mime_type"]
        row["delivery_path"] = str(delivery)
        row["delivery_sha256"] = delivery_sha256
        row["delivery_size_bytes"] = delivery.stat().st_size


def parse_response(*, raw: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = smoke._response_payload(raw)
    text = smoke._response_text(payload)
    answer = text.strip() if isinstance(text, str) else ""
    usage = smoke._usage(payload)
    response_id = payload.get(
        "response_id", payload.get("responseId", payload.get("id"))
    )
    finish_reason = payload.get("finish_reason", payload.get("finishReason"))
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
            finish_reason = candidates[0].get(
                "finish_reason", candidates[0].get("finishReason")
            )
    finish_reason = finish_reason if isinstance(finish_reason, str) else None
    failure: str | None = None
    if finish_reason in {"PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
        failure = "prohibited_content" if finish_reason == "PROHIBITED_CONTENT" else "blocked_content"
    elif finish_reason in {"MAX_TOKENS", "MAX_TOKENS_REACHED"}:
        failure = "max_tokens"
    elif not answer:
        failure = "empty_text"
    if payload.get("error"):
        failure = "provider_error"
    return {
        "query_id": int(row["query_id"]),
        "video_id": str(row["video_id"]),
        "status": "succeeded" if failure is None else "failed",
        "answer": answer,
        "raw_response_sha256": None,
        "response_id": response_id if isinstance(response_id, str) else None,
        "model_version": payload.get("model_version", payload.get("modelVersion")),
        "finish_reason": finish_reason,
        "usage": usage,
        "error": smoke._safe_error(payload.get("error")) if failure else None,
        "failure_reason": failure,
    }


def _usage_cost(usage: Mapping[str, Any] | None) -> tuple[int, int, float] | None:
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    candidate_output = usage.get("output_tokens")
    total = usage.get("total_tokens")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (prompt, candidate_output, total)
    ):
        return None
    if total < prompt:
        return None
    # Gemini bills hidden thinking tokens at the output rate.  The SDK's
    # candidates count excludes them, while total - prompt retains them.
    billed_output = max(candidate_output, total - prompt)
    return (
        prompt,
        billed_output,
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + billed_output / 1_000_000 * OUTPUT_USD_PER_MILLION,
    )


def _estimated_row_cost(row: Mapping[str, Any]) -> float:
    prompt = (
        math.ceil(
            float(row.get("duration_seconds", 0.0)) * ESTIMATED_INPUT_TOKENS_PER_SECOND
        )
        + ESTIMATED_QUESTION_TOKENS
    )
    return (
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + MAX_OUTPUT_TOKENS / 1_000_000 * OUTPUT_USD_PER_MILLION
    )


def _observed_cost(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, float | None]:
    prompt = 0
    output = 0
    complete = True
    for row in rows:
        if row.get("status") in {
            "cost_cap_blocked",
            "unavailable",
            "unsupported",
            "upload_failed",
        }:
            continue
        cost = _usage_cost(row.get("usage") if isinstance(row, Mapping) else None)
        if cost is None:
            complete = False
            continue
        prompt_value, output_value, _ = cost
        prompt += prompt_value
        output += output_value
    if not complete:
        return prompt, output, None
    return (
        prompt,
        output,
        round(
            prompt / 1_000_000 * INPUT_USD_PER_MILLION
            + output / 1_000_000 * OUTPUT_USD_PER_MILLION,
            8,
        ),
    )


def _manifest(
    *,
    requests_path: Path,
    rows: Sequence[Mapping[str, Any]],
    estimate: Mapping[str, Any],
    max_estimated_cost_usd: float,
    max_observed_cost_usd: float,
    created_at: str,
) -> dict[str, Any]:
    ready = [row for row in rows if row["status"] == "ready"]
    unavailable = [row for row in rows if row["status"] == "unavailable"]
    unsupported = [row for row in rows if row["status"] == "unsupported"]
    media_ids = {str(row["video_id"]) for row in ready}
    return {
        "schema_version": 1,
        "experiment_id": "deadline-submission-full-20260816-gemini",
        "status": "inputs_frozen",
        "created_at": created_at,
        "input": {"path": str(requests_path), "sha256": _sha256_file(requests_path)},
        "model": MODEL,
        "prompt": PROMPT,
        "model_settings": {
            "video_fps": VIDEO_FPS,
            "media_resolution": MEDIA_RESOLUTION.value,
            "thinking_level": THINKING_LEVEL.value,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
        },
        "request_count": len(ready),
        "panel_row_count": len(rows),
        "native_request_count": len(ready),
        "upload_count": len(media_ids),
        "unavailable_count": len(unavailable),
        "unsupported_count": len(unsupported),
        "cost": dict(estimate),
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "max_observed_cost_usd": max_observed_cost_usd,
        "rows": [dict(row) for row in rows],
    }


def _query_filename(query_id: int) -> str:
    return f"{query_id:06d}.json"


def _upload_one(
    *,
    client: Any,
    video_id: str,
    row: Mapping[str, Any],
    output_root: Path,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Upload one video and freeze its own terminal artifact.

    The worker, rather than the coordinator, reserves the attempt immediately
    before touching the provider.  This is what lets a rolling pool be
    interrupted without leaving queued videos looking in-flight.
    """

    upload_dir = output_root / "uploads"
    path = upload_dir / f"{video_id}.json"
    attempt = upload_dir / f"{video_id}.attempt.json"
    if path.exists():
        raise ValueError(f"upload terminal already exists: {video_id}")
    if attempt.exists():
        raise ValueError(f"upload attempt already exists: {video_id}")
    _write_json(
        attempt,
        {
            "schema_version": 1,
            "status": "upload_attempt_reserved",
            "video_id": video_id,
            "media_sha256": row["media_sha256"],
            "media_size_bytes": row["media_size_bytes"],
            "mime_type": row["media_mime_type"],
            "provider_retry_policy": dict(PROVIDER_RETRY_POLICY),
        },
    )
    retry_state = {
        "retry_count": 0,
        "retry_first_error": None,
        "retry_phase": None,
        "possible_orphan_duplicate": False,
    }
    try:
        provider = _upload_once(
            client=client, row=row, sleep=sleep, retry_state=retry_state
        )
        record = {
            "schema_version": 1,
            "status": "uploaded",
            "video_id": video_id,
            "media_sha256": row["media_sha256"],
            "media_size_bytes": row["media_size_bytes"],
            "mime_type": row["media_mime_type"],
            "file_uri": provider["file_uri"],
            "file_name": provider["file_name"],
            "provider_snapshot": provider,
        }
    except Exception as error:  # noqa: BLE001 - freeze per-video upload failure
        record = {
            "schema_version": 1,
            "status": "upload_failed",
            "video_id": video_id,
            "media_sha256": row["media_sha256"],
            "media_size_bytes": row["media_size_bytes"],
            "mime_type": row["media_mime_type"],
            "error": smoke._safe_error(error),
        }
    record["provider_retry_policy"] = dict(PROVIDER_RETRY_POLICY)
    record["retry_count"] = retry_state["retry_count"]
    if retry_state["retry_count"]:
        record.update(
            {
                "retry_first_error": retry_state["retry_first_error"],
                "retry_phase": retry_state["retry_phase"],
                "possible_orphan_duplicate": retry_state[
                    "possible_orphan_duplicate"
                ],
            }
        )
    _write_json(path, record)
    return record


def _upload_pending(
    *,
    pending: Sequence[tuple[str, Mapping[str, Any]]],
    client: Any,
    output_root: Path,
    sleep: Callable[[float], None],
) -> dict[str, dict[str, Any]]:
    """Run a bounded rolling upload pool and wait for all active workers.

    Only up to ``UPLOAD_WORKERS`` futures are submitted at a time.  On an
    interruption, queued items therefore have no attempt artifact; active
    workers are allowed to drain and freeze their terminals before the
    interruption is re-raised for a safe restart.
    """

    if not pending:
        return {}
    executor = ThreadPoolExecutor(
        max_workers=min(UPLOAD_WORKERS, len(pending)),
        thread_name_prefix="gemini-upload",
    )
    futures: dict[Future[dict[str, Any]], str] = {}
    pending_iter = iter(pending)

    def submit_next() -> None:
        try:
            video_id, row = next(pending_iter)
        except StopIteration:
            return
        future = executor.submit(
            _upload_one,
            client=client,
            video_id=video_id,
            row=row,
            output_root=output_root,
            sleep=sleep,
        )
        futures[future] = video_id

    for _ in range(min(UPLOAD_WORKERS, len(pending))):
        submit_next()

    records: dict[str, dict[str, Any]] = {}
    try:
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                video_id = futures.pop(future)
                # Provider failures are converted to terminal upload_failed by
                # the worker.  BaseException is intentionally allowed through
                # so the coordinator can drain and preserve a safe resume.
                records[video_id] = future.result()
                submit_next()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return records


def _run_live(
    *,
    rows: Sequence[Mapping[str, Any]],
    requests_path: Path,
    output_root: Path,
    client: Any,
    estimate: Mapping[str, Any],
    max_estimated_cost_usd: float,
    max_observed_cost_usd: float,
    created_at: str,
    sleep: Callable[[float], None],
    parser: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("uploads", "requests", "raw"):
        (output_root / name).mkdir(exist_ok=True)
    mutable_rows = [dict(row) for row in rows]
    _prepare_deliveries(mutable_rows, output_root / "deliveries")
    rows = mutable_rows
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        existing_body = _artifact_body(existing)
        prior_created_at = existing_body.get("created_at")
        if not isinstance(prior_created_at, str) or not prior_created_at:
            raise ValueError("existing run manifest has no creation timestamp")
        manifest = _manifest(
            requests_path=requests_path,
            rows=rows,
            estimate=estimate,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_observed_cost_usd=max_observed_cost_usd,
            created_at=prior_created_at,
        )
        if _artifact_body(existing) != manifest:
            raise ValueError(
                "existing run manifest does not match requested input/protocol"
            )
        created_at = prior_created_at
    else:
        manifest = _manifest(
            requests_path=requests_path,
            rows=rows,
            estimate=estimate,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_observed_cost_usd=max_observed_cost_usd,
            created_at=created_at,
        )
        _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    uploads: dict[str, dict[str, Any]] = {}
    upload_failures: dict[str, dict[str, Any]] = {}
    ready_by_video: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row["status"] == "ready":
            ready_by_video.setdefault(str(row["video_id"]), row)
    pending_uploads: list[tuple[str, Mapping[str, Any]]] = []
    for video_id, row in ready_by_video.items():
        path = output_root / "uploads" / f"{video_id}.json"
        if path.exists():
            record = _read_json(path)
            if record.get("media_sha256") != row["media_sha256"]:
                raise ValueError(f"upload media hash changed: {video_id}")
            if record.get("status") == "uploaded":
                uploads[video_id] = record
            elif record.get("status") == "upload_failed":
                upload_failures[video_id] = record
            else:
                raise ValueError(f"upload record is not terminal: {video_id}")
            continue
        attempt = output_root / "uploads" / f"{video_id}.attempt.json"
        if attempt.exists():
            raise ValueError(
                f"upload attempt is ambiguous; inspect before resuming: {video_id}"
            )
        pending_uploads.append((video_id, row))

    for video_id, record in _upload_pending(
        pending=pending_uploads,
        client=client,
        output_root=output_root,
        sleep=sleep,
    ).items():
        if record["status"] == "uploaded":
            uploads[video_id] = record
        else:
            upload_failures[video_id] = record

    raw_rows: list[dict[str, Any]] = []
    running_cost_usd = 0.0
    for row in rows:
        query_id = int(row["query_id"])
        raw_path = output_root / "raw" / _query_filename(query_id)
        if raw_path.exists():
            raw_record = _read_json(raw_path)
        elif row["status"] != "ready":
            raw_record = {
                "schema_version": 1,
                "status": row["status"],
                "query_id": query_id,
                "video_id": row["video_id"],
                "reason": row["reason"],
                "response": None,
            }
            _write_json(raw_path, raw_record)
        elif str(row["video_id"]) in upload_failures:
            raw_record = {
                "schema_version": 1,
                "status": "upload_failed",
                "query_id": query_id,
                "video_id": row["video_id"],
                "reason": "provider_upload_failed",
                "response": None,
            }
            _write_json(raw_path, raw_record)
        elif running_cost_usd + _estimated_row_cost(row) > max_observed_cost_usd:
            raw_record = {
                "schema_version": 1,
                "status": "cost_cap_blocked",
                "query_id": query_id,
                "video_id": row["video_id"],
                "reason": "next_request_cost_ceiling_exceeds_cap",
                "response": None,
            }
            _write_json(raw_path, raw_record)
        else:
            video_id = str(row["video_id"])
            upload = uploads[video_id]
            request = build_request(row=row, upload=upload)
            request_dump = smoke._plain(
                request.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
            request_path = output_root / "requests" / _query_filename(query_id)
            request_record = {
                "query_id": query_id,
                "video_id": video_id,
                "model": MODEL,
                "prompt": PROMPT,
                "media_sha256": row["media_sha256"],
                "request_sha256": _sha256_bytes(_canonical_json(request_dump)),
            }
            _write_json_once(request_path, request_record)
            attempt_path = output_root / "requests" / f"{query_id:06d}.attempt.json"
            if attempt_path.exists():
                raise ValueError(
                    f"request attempt is ambiguous; inspect before resuming: {query_id}"
                )
            _write_json(
                attempt_path,
                {
                    "schema_version": 1,
                    "status": "request_attempt_reserved",
                    "query_id": query_id,
                    "video_id": video_id,
                    "request_sha256": request_record["request_sha256"],
                },
            )
            request_started = time.monotonic()
            try:
                raw = client.models.generate_content(
                    model=MODEL,
                    contents=request.contents,
                    config=request.config,
                )
                raw_record = {
                    "schema_version": 1,
                    "status": "response_received",
                    "query_id": query_id,
                    "video_id": video_id,
                    "request_sha256": _sha256_file(request_path),
                    "elapsed_seconds": max(0.0, time.monotonic() - request_started),
                    "response": smoke._safe_response_object(raw),
                }
            except Exception as error:  # noqa: BLE001 - preserve terminal response failure
                raw_record = {
                    "schema_version": 1,
                    "status": "request_failed",
                    "query_id": query_id,
                    "video_id": video_id,
                    "request_sha256": _sha256_file(request_path),
                    "elapsed_seconds": max(0.0, time.monotonic() - request_started),
                    "error": smoke._safe_error(error),
                    "response": None,
                }
            _write_json(raw_path, raw_record)
        if raw_record["status"] == "response_received":
            payload = smoke._response_payload(raw_record["response"])
            usage_cost = _usage_cost(smoke._usage(payload))
            running_cost_usd += (
                usage_cost[2] if usage_cost is not None else _estimated_row_cost(row)
            )
        elif raw_record["status"] == "request_failed":
            # A transport failure may happen after provider work was billed.
            # Reserve the per-row ceiling before deciding whether another call
            # can fit inside the configured spend cap.
            running_cost_usd += _estimated_row_cost(row)
        raw_rows.append(
            {
                "query_id": query_id,
                "video_id": row["video_id"],
                "status": raw_record["status"],
                "raw_response_sha256": (
                    _sha256_file(raw_path)
                    if raw_record["status"] == "response_received"
                    else None
                ),
            }
        )

    raw_freeze = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "raw_frozen",
        "frozen_at": created_at,
        "model": MODEL,
        "manifest_sha256": manifest_sha256,
        "rows": raw_rows,
    }
    raw_freeze_path = output_root / "raw-freeze.json"
    raw_freeze_sha256 = _write_json_once(raw_freeze_path, raw_freeze)

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        query_id = int(row["query_id"])
        raw_path = output_root / "raw" / _query_filename(query_id)
        raw_record = _read_json(raw_path)
        response_hash = (
            _sha256_file(raw_path)
            if raw_record["status"] == "response_received"
            else None
        )
        if raw_record["status"] == "response_received":
            parsed = parser(raw=raw_record["response"], row=row)
            parsed["raw_response_sha256"] = response_hash
        elif raw_record["status"] == "request_failed":
            parsed = {
                "query_id": query_id,
                "video_id": row["video_id"],
                "status": "failed",
                "answer": "",
                "raw_response_sha256": None,
                "response_id": None,
                "model_version": None,
                "finish_reason": None,
                "usage": None,
                "error": raw_record.get("error"),
                "failure_reason": "request_failed",
            }
        else:
            parsed = {
                "query_id": query_id,
                "video_id": row["video_id"],
                "status": raw_record["status"],
                "answer": "",
                "raw_response_sha256": None,
                "response_id": None,
                "model_version": None,
                "finish_reason": None,
                "usage": None,
                "error": None,
                "failure_reason": raw_record.get("reason"),
            }
        elapsed = raw_record.get("elapsed_seconds", 0.0)
        parsed["elapsed_seconds"] = (
            float(elapsed)
            if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
            else 0.0
        )
        parsed_rows.append(parsed)

    prompt_tokens, output_tokens, observed_cost = _observed_cost(parsed_rows)
    status = "predictions_frozen"
    if observed_cost is not None and observed_cost > max_observed_cost_usd:
        status = "observed_cost_cap_exceeded"
    predictions = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": status,
        "frozen_at": created_at,
        "model": MODEL,
        "prompt": PROMPT,
        "manifest_sha256": manifest_sha256,
        "raw_freeze_sha256": raw_freeze_sha256,
        "row_count": len(parsed_rows),
        "completed_count": sum(row["status"] == "succeeded" for row in parsed_rows),
        "unavailable_count": sum(row["status"] == "unavailable" for row in parsed_rows),
        "unsupported_count": sum(row["status"] == "unsupported" for row in parsed_rows),
        "upload_failed_count": sum(
            row["status"] == "upload_failed" for row in parsed_rows
        ),
        "rows": parsed_rows,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "observed_cost_usd": observed_cost,
        "observed_cost_cap_usd": max_observed_cost_usd,
    }
    predictions_path = output_root / "predictions-freeze.json"
    predictions_sha256 = _write_json_once(predictions_path, predictions)
    summary = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": status,
        "row_count": len(parsed_rows),
        "completed_count": predictions["completed_count"],
        "native_request_count": sum(row["status"] == "ready" for row in rows),
        "upload_count": len(uploads),
        "request_count": sum(row["status"] == "ready" for row in rows),
        "panel_row_count": len(parsed_rows),
        "unavailable_count": predictions["unavailable_count"],
        "unsupported_count": predictions["unsupported_count"],
        "upload_failed_count": predictions["upload_failed_count"],
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "estimated_cost_cap_usd": max_estimated_cost_usd,
        "observed_cost_usd": observed_cost,
        "observed_cost_cap_usd": max_observed_cost_usd,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "manifest_sha256": manifest_sha256,
        "raw_freeze_sha256": raw_freeze_sha256,
        "predictions_freeze_sha256": predictions_sha256,
    }
    _write_json_once(output_root / "summary.json", summary)
    return summary


def run_full(
    *,
    requests_path: Path = REQUESTS_PATH,
    media_root: Path = MEDIA_ROOT,
    output_root: Path = PRIVATE_RUN_ROOT,
    client: Any | None = None,
    dry_run: bool = False,
    expected_count: int | None = EXPECTED_QUERY_COUNT,
    max_estimated_cost_usd: float = MAX_ESTIMATED_COST_USD,
    max_observed_cost_usd: float = MAX_OBSERVED_COST_USD,
    created_at: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    parse_response: Callable[..., dict[str, Any]] = parse_response,
) -> dict[str, Any]:
    requests_path = Path(requests_path).resolve(strict=True)
    rows = load_input_rows(
        requests_path, media_root=Path(media_root), expected_count=expected_count
    )
    estimate = estimate_cost(rows)
    if estimate["estimated_cost_usd"] > max_estimated_cost_usd:
        raise ValueError(
            f"estimated cost {estimate['estimated_cost_usd']:.8f} exceeds cost cap "
            f"{max_estimated_cost_usd:.8f}"
        )
    ready = [row for row in rows if row["status"] == "ready"]
    unavailable = [row for row in rows if row["status"] == "unavailable"]
    unsupported = [row for row in rows if row["status"] == "unsupported"]
    result = {
        "status": "dry_run",
        "row_count": len(rows),
        "native_request_count": len(ready),
        "upload_count": len({str(row["video_id"]) for row in ready}),
        "request_count": len(ready),
        "panel_row_count": len(rows),
        "unavailable_count": len(unavailable),
        "unsupported_count": len(unsupported),
        "estimated_cost_cap_usd": max_estimated_cost_usd,
        "observed_cost_cap_usd": max_observed_cost_usd,
        **estimate,
    }
    if dry_run:
        return result
    if client is None:
        raise ValueError("provider client is required for a live run")
    return _run_live(
        rows=rows,
        requests_path=requests_path,
        output_root=Path(output_root),
        client=client,
        estimate=estimate,
        max_estimated_cost_usd=max_estimated_cost_usd,
        max_observed_cost_usd=max_observed_cost_usd,
        created_at=created_at or _now(),
        sleep=sleep,
        parser=parse_response,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def client_factory(*, api_key: str) -> Any:
    return genai.Client(api_key=api_key)


def _private_api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    try:
        mode = PRIVATE_ENV_FILE.stat().st_mode & 0o777
    except OSError as error:
        raise ValueError("approved Gemini credential file is unavailable") from error
    if mode != 0o600:
        raise ValueError("approved Gemini credential file permissions are not 0600")
    try:
        lines = PRIVATE_ENV_FILE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("approved Gemini credential file cannot be read") from error
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != "GEMINI_API_KEY":
            continue
        parsed = shlex.split(raw_value.strip(), comments=True)
        if parsed and parsed[0].strip():
            return parsed[0].strip()
    raise ValueError("GEMINI_API_KEY is not configured")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_gemini_full")
    parser.add_argument(
        "--run", action="store_true", help="perform provider calls (default is dry-run)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and estimate only"
    )
    parser.add_argument("--requests", type=Path, default=REQUESTS_PATH)
    parser.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=PRIVATE_RUN_ROOT)
    parser.add_argument(
        "--max-estimated-cost-usd", type=float, default=MAX_ESTIMATED_COST_USD
    )
    parser.add_argument(
        "--max-observed-cost-usd", type=float, default=MAX_OBSERVED_COST_USD
    )
    return parser


def _emit(result: Mapping[str, Any]) -> None:
    keys = (
        "status",
        "row_count",
        "native_request_count",
        "upload_count",
        "request_count",
        "unavailable_count",
        "unsupported_count",
        "completed_count",
        "estimated_cost_usd",
        "estimated_cost_cap_usd",
        "observed_cost_usd",
        "observed_cost_cap_usd",
    )
    print(
        json.dumps({key: result[key] for key in keys if key in result}, sort_keys=True)
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory_override: Callable[..., Any] | None = None,
) -> int:
    try:
        args = _parser().parse_args(argv)
        dry_run = not args.run
        if args.dry_run and args.run:
            raise ValueError("--run and --dry-run are mutually exclusive")
        factory = client_factory_override or client_factory
        client = None if dry_run else factory(api_key=_private_api_key())
        result = run_full(
            requests_path=args.requests,
            media_root=args.media_root,
            output_root=args.output,
            client=client,
            dry_run=dry_run,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_observed_cost_usd=args.max_observed_cost_usd,
        )
        _emit(result)
        return 0
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - do not leak private paths/provider text
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
