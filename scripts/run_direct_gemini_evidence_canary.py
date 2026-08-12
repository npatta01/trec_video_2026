#!/usr/bin/env python3
"""Run a small direct-Gemini timeline, retrieval, and answer canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from trec_video_2026.vqa_frontier_map_media import prepare_delivery_media


MODEL = "gemini-3.1-pro-preview"
FPS = 4.0
_TIMELINE = re.compile(
    r"^\s*(?:\d+:)?(?P<start>\d+(?:\.\d+)?)\s*-\s*"
    r"(?:\d+:)?(?P<end>\d+(?:\.\d+)?)\s*\|\s*(?P<description>.+?)\s*$"
)
_SPAN = re.compile(
    r"^\s*(?:[-*]\s*)?SPAN(?:\s+\d+)?\s*:\s*(?P<start>[\d:.]+)\s*-\s*"
    r"(?P<end>[\d:.]+)\s*\|\s*(?P<reason>.+?)\s*$",
    re.IGNORECASE,
)
_ANSWER = re.compile(
    r"^\s*(?:[-*]\s*)?ANSWER\s*:\s*(?P<answer>.+?)\s*$", re.IGNORECASE
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError("timestamp has too many components")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def parse_timeline(text: str, duration: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _TIMELINE.match(line)
        if not match:
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        if start < 0 or end <= start or end > duration + 0.05:
            raise ValueError("timeline span is outside the video")
        events.append(
            {"start_seconds": start, "end_seconds": min(end, duration), "description": match.group("description")}
        )
    if not events:
        raise ValueError("timeline contains no parseable events")
    return events


def parse_selection(text: str, duration: float) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    answer: str | None = None
    for line in text.splitlines():
        span = _SPAN.match(line)
        if span:
            start = _seconds(span.group("start"))
            end = _seconds(span.group("end"))
            if start < 0 or end <= start or end > duration + 0.05:
                raise ValueError("selected span is outside the video")
            spans.append(
                {"start_seconds": start, "end_seconds": min(end, duration), "reason": span.group("reason")}
            )
            continue
        selected_answer = _ANSWER.match(line)
        if selected_answer:
            answer = selected_answer.group("answer")
    if not spans or not answer:
        raise ValueError("selection must contain at least one SPAN and one ANSWER")
    return {"spans": spans, "answer": answer}


def frame_timestamps(spans: list[dict[str, Any]], duration: float, maximum: int = 8) -> list[float]:
    values: list[float] = []
    for span in spans:
        start = float(span["start_seconds"])
        end = float(span["end_seconds"])
        epsilon = min(0.25, (end - start) / 4)
        values.extend((start + epsilon, (start + end) / 2, end - epsilon))
    unique = sorted({round(max(0.0, min(value, duration - 0.001)), 3) for value in values})
    if len(unique) <= maximum:
        return unique
    indexes = [round(index * (len(unique) - 1) / (maximum - 1)) for index in range(maximum)]
    return [unique[index] for index in indexes]


def _metadata(response: Any, elapsed: float) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    text = response.text or ""
    candidates = getattr(response, "candidates", None) or []
    finish_reason = None
    if candidates:
        value = getattr(candidates[0], "finish_reason", None)
        finish_reason = getattr(value, "name", str(value)) if value is not None else None
    return {
        "elapsed_seconds": round(elapsed, 6),
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_chars": len(text),
        "prompt_token_count": getattr(usage, "prompt_token_count", None),
        "candidates_token_count": getattr(usage, "candidates_token_count", None),
        "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
        "total_token_count": getattr(usage, "total_token_count", None),
        "cached_content_token_count": getattr(usage, "cached_content_token_count", None),
        "model_version": getattr(response, "model_version", None),
        "response_id": getattr(response, "response_id", None),
        "finish_reason": finish_reason,
    }


def _config(max_output_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.0,
        seed=42,
        max_output_tokens=max_output_tokens,
        response_mime_type="text/plain",
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH, include_thoughts=False
        ),
    )


def _call(client: Any, contents: Any, max_output_tokens: int) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    response = client.models.generate_content(
        model=MODEL, contents=contents, config=_config(max_output_tokens)
    )
    return response.text or "", _metadata(response, time.monotonic() - started)


def _upload(client: Any, media: Path) -> Any:
    uploaded = client.files.upload(
        file=str(media), config={"mime_type": "video/webm", "display_name": "vqa-evidence-canary"}
    )
    for _ in range(90):
        uploaded = client.files.get(name=uploaded.name)
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state == "ACTIVE":
            return uploaded
        if state == "FAILED":
            raise RuntimeError("Google file processing failed")
        time.sleep(1)
    raise TimeoutError("Google file processing timed out")


def _video_part(uploaded: Any) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=uploaded.uri, mime_type="video/webm"),
        video_metadata=types.VideoMetadata(fps=FPS),
        media_resolution=types.PartMediaResolution(
            level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
        ),
    )


def _timeline_prompt(duration: float) -> str:
    return f"""Watch and listen to the complete {duration:.3f}-second video from beginning to end. Create a question-blind chronological timeline of everything observably happening. Include continuous states for their full duration, brief transitions, interactions, speech, non-speech sound, visible text, counts, and events near the end when present. Do not infer motives or facts that are not visible or audible.

Return plain text only, one event per line:
start_seconds-end_seconds | observable description

Cover the full video. Use precise timestamps, not evenly spaced schematic intervals."""


def _direct_prompt(question: str) -> str:
    return f"""Answer the question using the complete native video and audio. Identify the requested relationship and inspect all relevant moments. For temporal questions, include multiple supported subsequent events when the wording is ambiguous rather than stopping at an incidental micro-action. Give one concise best-supported answer only.

Question: {question}"""


def _selection_prompt(question: str, timeline: str) -> str:
    return f"""Use the question-blind timeline to answer the question and select every video interval needed to verify that answer. Interpret the requested relationship (temporal, count, cause, speech, sound, visible text, entity/action, or global context). For temporal questions, inspect the complete relevant sequence and do not stop at an incidental micro-action.

Return plain text only in this form, with one or more SPAN lines and exactly one ANSWER line:
SPAN: start_seconds-end_seconds | why this interval is relevant
ANSWER: one concise best-supported answer

Question: {question}

Timeline:
{timeline}"""


def _frame_answer_prompt(question: str, timeline: str, frames: list[dict[str, Any]]) -> str:
    labels = "\n".join(f"- frame {index + 1}: {frame['timestamp_seconds']:.3f}s" for index, frame in enumerate(frames))
    return f"""Answer the question using the frozen question-blind timeline and the timestamped frames selected from relevant intervals. Reconcile all evidence. For temporal questions, include multiple supported subsequent events when necessary to represent the complete outcome. Give one concise best-supported answer only.

Question: {question}

Timeline:
{timeline}

Attached frame timestamps:
{labels}"""


def _extract_frames(media: Path, timestamps: list[float], output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps, 1):
        path = output / f"frame-{index:02d}-{timestamp:07.3f}.jpg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-ss", f"{timestamp:.3f}", "-i", str(media), "-frames:v", "1", "-q:v", "2", str(path)],
            check=True,
        )
        frames.append(
            {"timestamp_seconds": timestamp, "path": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    return frames


def prepare(panel: Path, output: Path) -> None:
    payload = json.loads(panel.read_text(encoding="utf-8"))
    rows = [row for row in payload["examples"] if row["phase"] == "screen"]
    manifest = {
        "schema_version": 1,
        "experiment_id": "direct-gemini-evidence-canary-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_policy": "frozen-phase0-screen-one-per-category",
        "model": MODEL,
        "video_fps": FPS,
        "media_resolution": "HIGH",
        "thinking_level": "HIGH",
        "temperature": 0.0,
        "seed": 42,
        "label_isolation": "references unavailable until predictions-freeze.json exists",
        "source_panel": str(panel.resolve()),
        "source_panel_sha256": _sha256(panel),
        "rows": rows,
    }
    _write_once(output / "manifest.json", manifest)


def run_row(root: Path, video_id: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    row = next((item for item in manifest["rows"] if item["video_id"] == video_id), None)
    if row is None:
        raise ValueError("video is not in the frozen panel")
    row_root = root / "rows" / video_id
    if (row_root / "complete.json").exists():
        print(json.dumps({"video_id": video_id, "status": "already-complete"}))
        return
    source = Path("data") / row["media_relative_path"]
    if _sha256(source) != row["media_sha256"]:
        raise ValueError("source media identity changed")
    delivery = prepare_delivery_media(
        run_path=row_root / "delivery", source_path=source, video_id=video_id, provider="gemini"
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = None
    cleanup = "not-uploaded"
    try:
        timeline_path = row_root / "timeline" / "response.txt"
        if timeline_path.is_file():
            timeline = timeline_path.read_text(encoding="utf-8")
            parse_timeline(timeline, float(row["duration_seconds"]))
        else:
            uploaded = _upload(client, delivery.path)
            video = _video_part(uploaded)
            timeline_prompt = _timeline_prompt(float(row["duration_seconds"]))
            timeline, timeline_metadata = _call(client, [video, timeline_prompt], 8192)
            _write_once(row_root / "timeline" / "request.json", {"prompt": timeline_prompt})
            _write_once(row_root / "timeline" / "response.txt", timeline)
            _write_once(row_root / "timeline" / "metadata.json", timeline_metadata)
            events = parse_timeline(timeline, float(row["duration_seconds"]))
            _write_once(row_root / "timeline" / "events.json", events)

        direct_path = row_root / "direct-answer" / "response.txt"
        if direct_path.is_file():
            direct_answer = direct_path.read_text(encoding="utf-8")
        else:
            if uploaded is None:
                uploaded = _upload(client, delivery.path)
            video = _video_part(uploaded)
            direct_prompt = _direct_prompt(row["question"])
            direct_answer, direct_metadata = _call(client, [video, direct_prompt], 2048)
            _write_once(row_root / "direct-answer" / "request.json", {"question": row["question"], "prompt": direct_prompt})
            _write_once(row_root / "direct-answer" / "response.txt", direct_answer)
            _write_once(row_root / "direct-answer" / "metadata.json", direct_metadata)

        selection_prompt = _selection_prompt(row["question"], timeline)
        selection_path = row_root / "timeline-answer" / "response.txt"
        if selection_path.is_file():
            selection_text = selection_path.read_text(encoding="utf-8")
        else:
            if (
                timeline_path.is_file()
                and direct_path.is_file()
                and (row_root / "cleanup.json").is_file()
                and not (row_root / "timeline-answer" / "retry.json").is_file()
            ):
                _write_once(
                    row_root / "timeline-answer" / "pre-preservation-audit-gap.json",
                    {"status": "one_text_selector_response_not_preserved", "video_calls_reused": True},
                )
            selection_text, selection_metadata = _call(client, selection_prompt, 4096)
            _write_once(row_root / "timeline-answer" / "request.json", {"question": row["question"], "timeline_sha256": hashlib.sha256(timeline.encode()).hexdigest(), "prompt": selection_prompt})
            _write_once(row_root / "timeline-answer" / "response.txt", selection_text)
            _write_once(row_root / "timeline-answer" / "metadata.json", selection_metadata)
        selection = parse_selection(selection_text, float(row["duration_seconds"]))
        selection_json = row_root / "timeline-answer" / "selection.json"
        if not selection_json.exists():
            _write_once(selection_json, selection)

        frame_manifest = row_root / "selected-frames" / "manifest.json"
        if frame_manifest.is_file():
            frames = json.loads(frame_manifest.read_text(encoding="utf-8"))
        else:
            timestamps = frame_timestamps(selection["spans"], float(row["duration_seconds"]))
            frames = _extract_frames(delivery.path, timestamps, row_root / "selected-frames")
            _write_once(frame_manifest, frames)
        frame_prompt = _frame_answer_prompt(row["question"], timeline, frames)
        frame_answer_path = row_root / "frame-answer" / "response.txt"
        if frame_answer_path.is_file():
            frame_answer = frame_answer_path.read_text(encoding="utf-8")
        else:
            frame_parts: list[Any] = [frame_prompt]
            for frame in frames:
                frame_parts.append(types.Part.from_bytes(data=(row_root / "selected-frames" / frame["path"]).read_bytes(), mime_type="image/jpeg"))
            frame_answer, frame_metadata = _call(client, frame_parts, 2048)
            _write_once(row_root / "frame-answer" / "request.json", {"question": row["question"], "timeline_sha256": hashlib.sha256(timeline.encode()).hexdigest(), "frames": frames, "prompt": frame_prompt})
            _write_once(row_root / "frame-answer" / "response.txt", frame_answer)
            _write_once(row_root / "frame-answer" / "metadata.json", frame_metadata)

        _write_once(
            row_root / "complete.json",
            {
                "video_id": video_id,
                "category": row["category"],
                "question": row["question"],
                "timeline_sha256": hashlib.sha256(timeline.encode()).hexdigest(),
                "direct_answer": direct_answer.strip(),
                "timeline_answer": selection["answer"].strip(),
                "frame_answer": frame_answer.strip(),
                "selected_spans": selection["spans"],
                "selected_frame_count": len(frames),
            },
        )
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
                cleanup = "succeeded"
            except Exception:
                cleanup = "failed"
        cleanup_path = row_root / "cleanup.json"
        if uploaded is not None and not cleanup_path.exists():
            _write_once(cleanup_path, {"uploaded_file_deletion": cleanup})
    print(json.dumps({"video_id": video_id, "status": "complete", "cleanup": cleanup}))


def freeze(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for row in manifest["rows"]:
        path = root / "rows" / row["video_id"] / "complete.json"
        if not path.is_file():
            raise ValueError(f"incomplete row: {row['video_id']}")
        rows.append({"video_id": row["video_id"], "complete_sha256": _sha256(path)})
    _write_once(
        root / "predictions-freeze.json",
        {"schema_version": 1, "status": "frozen", "row_count": len(rows), "rows": rows},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--panel", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    row_parser = sub.add_parser("run-row")
    row_parser.add_argument("--root", type=Path, required=True)
    row_parser.add_argument("--video-id", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.panel, args.output)
    elif args.command == "run-row":
        run_row(args.root, args.video_id)
    else:
        freeze(args.root)


if __name__ == "__main__":
    main()
