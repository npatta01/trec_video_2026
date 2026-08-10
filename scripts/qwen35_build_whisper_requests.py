#!/usr/bin/env python3
"""Bind accepted, label-free Whisper evidence to the shared frame requests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "correct_answer",
    "incorrect_answers",
    "expected_answer",
    "ground_truth",
    "label",
    "labels",
    "reference",
    "reference_answer",
    "reference_answers",
}
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def reject_labels(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise SystemExit(f"forbidden label-bearing field {key!r} at {location}")
            reject_labels(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_labels(child, f"{location}[{index}]")


def validate_frame_paths(request: dict[str, Any], frame_root: Path | None) -> None:
    video_id = request.get("video_id")
    if not isinstance(video_id, str) or YOUTUBE_ID_PATTERN.fullmatch(video_id) is None:
        raise SystemExit("request video_id is not a valid YouTube ID")
    paths = request.get("frame_paths")
    if not isinstance(paths, list) or not paths:
        raise SystemExit(f"frame paths are missing for {video_id}")
    if frame_root is None:
        return
    root = frame_root.resolve(strict=True)
    for raw_path in paths:
        path = Path(str(raw_path)).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"frame path escapes the declared evidence root: {raw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--frame-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame_requests = json.loads(args.frames.read_text(encoding="utf-8"))
    whisper_payload = json.loads(args.whisper.read_text(encoding="utf-8"))
    whisper_rows = whisper_payload.get("results")
    if not isinstance(frame_requests, list) or len(frame_requests) != args.count:
        raise SystemExit("frame request count mismatch")
    if not isinstance(whisper_rows, list) or len(whisper_rows) != args.count:
        raise SystemExit("Whisper result count mismatch")
    if whisper_payload.get("label_policy") != "label-free-transcript-evidence":
        raise SystemExit("Whisper artifact is not label-free")
    reject_labels(frame_requests, "frames")
    reject_labels(whisper_payload, "whisper")

    requests: list[dict[str, Any]] = []
    for index, (frame_request, whisper_row) in enumerate(
        zip(frame_requests, whisper_rows, strict=True)
    ):
        if frame_request.get("video_id") != whisper_row.get("video_id"):
            raise SystemExit(f"video ID mismatch at row {index}")
        if frame_request.get("condition_id") != "qwen35-4b-12f-frames-only":
            raise SystemExit(f"unexpected frame condition at row {index}")
        validate_frame_paths(frame_request, args.frame_root)
        status = whisper_row.get("status")
        if status not in {"passed", "no-audio-stream"}:
            raise SystemExit(f"Whisper failed at row {index}: {status!r}")
        transcript = whisper_row.get("selected_transcript")
        if whisper_row.get("transcript_included") is True:
            if not isinstance(transcript, str) or not transcript.strip():
                raise SystemExit(f"accepted Whisper row has no transcript at {index}")
        else:
            transcript = None
        request = dict(frame_request)
        request["condition_id"] = args.condition_id
        request["transcript"] = transcript
        requests.append(request)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"request_count": len(requests), "accepted_transcripts": sum(row.get("transcript_included") is True for row in whisper_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
