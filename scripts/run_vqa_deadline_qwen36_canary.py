#!/usr/bin/env python3
"""Run the five-video Qwen3.6 native-video deadline canary.

The input bundle is label-free.  Each video is sent twice (video plus the
accepted frozen Whisper transcript, then native video alone).  Every provider
body is written verbatim before response parsing so the raw-response freeze is
independent of any later scoring or reference access.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
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
DIRECT_ANSWER_PROMPT = (
    "Use the complete native video and audio to answer the question. "
    "Return one short phrase or sentence, answer only."
)
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


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


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
    except (OSError, json.JSONDecodeError) as error:
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
        video_id = raw.get("video_id")
        transcript = raw.get("transcript") or raw.get("text") or raw.get("timestamped_text")
        if not isinstance(video_id, str):
            raise ValueError("accepted Whisper row is missing video_id")
        # The frozen request file retains rows with no usable speech; those
        # rows are simply unavailable, while the five selected rows must each
        # resolve to a non-empty transcript in load_input_rows below.
        if not isinstance(transcript, str) or not transcript.strip():
            continue
        if video_id in transcripts and transcripts[video_id] != transcript:
            raise ValueError(f"accepted Whisper rows disagree for {video_id}")
        transcripts[video_id] = transcript
    return transcripts


def load_input_rows(path: Path, *, whisper_path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate exactly five label-free video/question rows."""

    payload = _load_json(path)
    if isinstance(payload, Mapping):
        raw_rows = payload.get("rows")
    else:
        raw_rows = payload
    if not isinstance(raw_rows, list) or len(raw_rows) != 5:
        raise ValueError("deadline canary input must contain exactly five rows")

    accepted_transcripts = _load_accepted_transcripts(whisper_path) if whisper_path else {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"input row {index} is not an object")
        video_id = raw.get("video_id")
        question = raw.get("question")
        video_path_raw = raw.get("video_path") or raw.get("media_path") or raw.get("source_media_path") or raw.get("source_path")
        transcript = raw.get("transcript") or raw.get("accepted_whisper_transcript") or raw.get("whisper_transcript") or accepted_transcripts.get(video_id)
        if not isinstance(video_id, str) or not video_id.strip() or video_id in seen:
            raise ValueError("input video IDs must be five unique nonempty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"input question is missing for {video_id}")
        if not isinstance(video_path_raw, str) or not video_path_raw:
            raise ValueError(f"input video path is missing for {video_id}")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"accepted transcript is missing for {video_id}")
        video_path = Path(video_path_raw)
        candidates = [
            path.parent / video_path if not video_path.is_absolute() else video_path,
        ]
        media_root_raw = os.environ.get("MEDIA_ROOT")
        if media_root_raw:
            media_root = Path(media_root_raw)
            candidates.extend(
                (
                    media_root / video_id / "delivery-media" / f"{video_id}.webm",
                    media_root / video_id / f"{video_id}.webm",
                    media_root / f"{video_id}.mkv",
                    media_root / f"{video_id}.webm",
                    media_root / Path(video_path_raw).name,
                )
            )
        video_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        if video_path.is_symlink() or not video_path.is_file():
            raise ValueError(f"video file is missing or symlinked for {video_id}")
        video_sha256 = sha256_file(video_path)
        declared_video_sha256 = raw.get("video_sha256") or raw.get("source_sha256")
        if declared_video_sha256 is not None and declared_video_sha256 != video_sha256:
            raise ValueError(f"video SHA-256 mismatch for {video_id}")
        transcript_sha256 = sha256_bytes(transcript.encode("utf-8"))
        declared_transcript_sha256 = raw.get("transcript_sha256")
        if declared_transcript_sha256 is not None and declared_transcript_sha256 != transcript_sha256:
            raise ValueError(f"transcript SHA-256 mismatch for {video_id}")
        seen.add(video_id)
        rows.append(
            {
                "video_id": video_id,
                "question": question,
                "video_path": str(video_path),
                "video_sha256": video_sha256,
                "transcript": transcript,
                "transcript_sha256": transcript_sha256,
            }
        )
    return rows


def _video_data_uri(video_path: Path) -> str:
    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/webm"
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_requests(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build the matched transcript/native payloads in deterministic order."""

    if len(rows) != 5:
        raise ValueError("canary must build requests for exactly five rows")
    requests: list[dict[str, Any]] = []
    for row in rows:
        video_id = row.get("video_id")
        question = row.get("question")
        video_path_raw = row.get("video_path")
        transcript = row.get("transcript")
        if not all(isinstance(value, str) and value for value in (video_id, question, video_path_raw, transcript)):
            raise ValueError("input row is missing required request fields")
        video_path = Path(video_path_raw)
        video_uri = _video_data_uri(video_path)
        video_sha256 = row.get("video_sha256") or sha256_file(video_path)
        transcript_sha256 = row.get("transcript_sha256") or sha256_bytes(transcript.encode("utf-8"))
        for condition_id, condition_transcript in (
            (TRANSCRIPT_CONDITION_ID, transcript),
            (NATIVE_CONDITION_ID, None),
        ):
            text = f"{DIRECT_ANSWER_PROMPT}\nQuestion: {question}"
            if condition_transcript is not None:
                text += f"\nTimestamped transcript:\n{condition_transcript}"
            payload: dict[str, Any] = {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": video_uri}},
                            {"type": "text", "text": text},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            requests.append(
                {
                    "request_id": f"{video_id}:{condition_id}",
                    "video_id": video_id,
                    "condition_id": condition_id,
                    "question": question,
                    "prompt": DIRECT_ANSWER_PROMPT,
                    "video_path": str(video_path),
                    "video_sha256": video_sha256,
                    "transcript_sha256": transcript_sha256 if condition_transcript else None,
                    "payload": payload,
                    "request_sha256": sha256_bytes(canonical_json(payload)),
                }
            )
    return requests


def _request_manifest(requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "request_id",
        "video_id",
        "condition_id",
        "question",
        "prompt",
        "video_sha256",
        "transcript_sha256",
        "request_sha256",
    )
    return [{field: request[field] for field in fields} for request in requests]


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


def _safe_error(error: Exception) -> str:
    value = _SAFE_ERROR.sub("_", type(error).__name__.casefold()).strip("_")
    return value or "request_failed"


def _extract_answer(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
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


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def run_canary(
    *,
    requests_path: Path,
    whisper_path: Path | None = None,
    output_root: Path,
    server_url: str,
    post: Callable[[str, Mapping[str, Any]], bytes | Mapping[str, Any]] | None = None,
) -> dict[str, int | str]:
    """Run ten calls, freeze raw responses, then parse answers."""

    rows = load_input_rows(requests_path, whisper_path=whisper_path)
    requests = build_requests(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw-responses"
    raw_root.mkdir()
    _exclusive_write(
        output_root / "requests-freeze.json",
        canonical_json(
            {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "prompt": DIRECT_ANSWER_PROMPT,
                "requests": _request_manifest(requests),
            }
        ),
    )

    sender = post or post_json
    response_records: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        started = time.monotonic()
        try:
            response = sender(server_url, request["payload"])
            raw = canonical_json(response) if isinstance(response, Mapping) else bytes(response)
            if not raw:
                raise ValueError("provider response was empty")
            record = {"status": "received", "error_type": None}
        except Exception as error:  # one attempt per matched request; no repair retry
            raw = canonical_json({"error_type": _safe_error(error)})
            record = {"status": "failed", "error_type": _safe_error(error)}
        raw_path = raw_root / f"{index:02d}.json"
        _exclusive_write(raw_path, raw)
        response_records.append(
            {
                "request_id": request["request_id"],
                "video_id": request["video_id"],
                "condition_id": request["condition_id"],
                "raw_response_sha256": sha256_bytes(raw),
                "elapsed_seconds": max(0.0, time.monotonic() - started),
                **record,
            }
        )

    _exclusive_write(
        output_root / "raw-response-manifest.json",
        canonical_json({"schema_version": 1, "responses": response_records}),
    )
    _exclusive_write(
        output_root / "raw-freeze.json",
        canonical_json(
            {
                "schema_version": 1,
                "status": "raw_responses_frozen",
                "raw_response_count": len(response_records),
                "raw_response_manifest_sha256": sha256_file(output_root / "raw-response-manifest.json"),
                "raw_response_sha256": [record["raw_response_sha256"] for record in response_records],
            }
        ),
    )

    predictions: list[dict[str, Any]] = []
    for index, (request, record) in enumerate(zip(requests, response_records, strict=True)):
        row: dict[str, Any] = {
            "request_id": request["request_id"],
            "video_id": request["video_id"],
            "condition_id": request["condition_id"],
            "question": request["question"],
            "raw_response_sha256": record["raw_response_sha256"],
            "status": record["status"],
            "error_type": record["error_type"],
        }
        if record["status"] == "received":
            try:
                row["answer"] = _extract_answer((raw_root / f"{index:02d}.json").read_bytes())
                row["status"] = "passed"
            except ValueError as error:
                row["status"] = "failed"
                row["error_type"] = _safe_error(error)
        predictions.append(row)
    _exclusive_write(output_root / "predictions.json", canonical_json(predictions))
    _exclusive_write(
        output_root / "predictions-freeze.json",
        canonical_json(
            {
                "schema_version": 1,
                "status": "predictions_frozen",
                "raw_freeze_sha256": sha256_file(output_root / "raw-freeze.json"),
                "raw_response_count": len(response_records),
                "rows": predictions,
            }
        ),
    )
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "request_count": len(requests),
        "raw_response_count": len(response_records),
        "passed_count": sum(row["status"] == "passed" for row in predictions),
        "failed_count": sum(row["status"] == "failed" for row in predictions),
    }
    _exclusive_write(output_root / "run-manifest.json", canonical_json(manifest))
    write_checksums(output_root)
    return {"status": "completed", **{key: value for key, value in manifest.items() if key.endswith("count")}}


def write_checksums(root: Path) -> Path:
    """Write checksums for immutable artifacts, excluding the live server log."""

    target = root / "SHA256SUMS"
    lines = []
    excluded = {target, root / "vllm-server.log"}
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path not in excluded):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    _exclusive_write(target, ("\n".join(lines) + "\n").encode("utf-8"))
    return target


def preflight(
    requests_path: Path,
    output_root: Path,
    *,
    whisper_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_input_rows(requests_path, whisper_path=whisper_path)
    requests = build_requests(rows)
    result = {
        "status": "preflight_passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "request_count": len(requests),
        "video_count": len(rows),
        "condition_ids": [TRANSCRIPT_CONDITION_ID, NATIVE_CONDITION_ID],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _exclusive_write(output_root / "preflight.json", canonical_json(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_qwen36_canary")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--whisper", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1/chat/completions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        result = preflight(args.requests, args.output, whisper_path=args.whisper)
    else:
        result = run_canary(
            requests_path=args.requests,
            whisper_path=args.whisper,
            output_root=args.output,
            server_url=args.server_url,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
