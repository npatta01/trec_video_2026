#!/usr/bin/env python3
"""Materialize selected-window frames and matched audio for the AV panel.

This consumes only the label-free plaintext event-map result and the existing
accepted Whisper transcript artifact. It records requested timestamps, actual
decoded PTS, frame hashes, contact sheets, and two matched request sets:
frames-only and frames-plus-Whisper.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from qwen35_prepare_matched_evidence import (  # noqa: E402
    extract_audio,
    extract_one_frame,
    locate_media,
    make_contact_sheet,
    sha256,
)
from imageio_ffmpeg import get_ffmpeg_exe  # noqa: E402
from trec_video_2026.vqa_experiments import nemotron3_omni_event_map as event_map  # noqa: E402


FRAME_COUNT = 12


def _extract_frame_pyav(media: Path, timestamp: float, output: Path) -> float:
    """Fallback decoder for clips where FFmpeg omits showinfo PTS output."""

    import av

    chosen = None
    with av.open(str(media)) as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        stream_time_base = float(stream.time_base) if stream.time_base is not None else 1e-6
        container.seek(max(0, int(timestamp / stream_time_base)), backward=True, stream=stream)
        last = None
        for frame in container.decode(stream):
            last = frame
            frame_time = frame.time
            if frame_time is None or frame_time >= timestamp:
                chosen = frame
                break
        if chosen is None:
            chosen = last
    if chosen is None:
        # A few VP9/MKV files expose metadata that makes a stream seek land
        # past the final packet. Decode from the beginning as a bounded fallback.
        with av.open(str(media)) as container:
            stream = next(stream for stream in container.streams if stream.type == "video")
            last = None
            for frame in container.decode(stream):
                last = frame
                frame_time = frame.time
                if frame_time is None or frame_time >= timestamp:
                    chosen = frame
                    break
            if chosen is None:
                chosen = last
    if chosen is None:
        raise RuntimeError(f"PyAV could not decode a frame for {media}")
    output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_image().convert("RGB").save(output, format="JPEG", quality=90)
    return float(chosen.time) if chosen.time is not None else timestamp


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object row")
            rows.append(value)
    return rows


def _selected_times(
    windows: list[dict[str, Any]], *, duration: float, count: int = FRAME_COUNT
) -> tuple[list[float], str]:
    intervals: list[tuple[float, float]] = []
    for window in windows:
        start = float(window.get("start_seconds", window.get("start")))
        end = float(window.get("end_seconds", window.get("end")))
        if 0 <= start < end <= duration:
            intervals.append((start, end))
    intervals.sort()
    if not intervals:
        # Keep failed/empty map rows runnable, but expose that they used the
        # deterministic uniform fallback rather than pretending selection worked.
        return [round((index + 0.5) * duration / count, 6) for index in range(count)], "uniform-fallback"
    total = sum(end - start for start, end in intervals)
    times: list[float] = []
    for index in range(count):
        target = (index + 0.5) * total / count
        cursor = 0.0
        for start, end in intervals:
            width = end - start
            if target <= cursor + width or (start, end) == intervals[-1]:
                times.append(round(start + max(0.0, target - cursor), 6))
                break
            cursor += width
    return times, "selected-windows"


def _whisper_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Whisper artifact results are missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("video_id"), str):
            raise ValueError("Whisper row is malformed")
        result[row["video_id"]] = row
    return result


def _prepare_one(
    item: tuple[int, dict[str, Any], dict[str, Any], Path, Path, Path, Path, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ordinal, panel, result, input_root, frame_root, audio_root, sheet_root, whisper = item
    video_id = str(panel["video_id"])
    media = locate_media(input_root, video_id)
    if sha256(media) != panel.get("media_sha256"):
        raise ValueError(f"source media hash mismatch: {video_id}")
    duration = float(panel["duration_seconds"])
    raw_windows = result.get("selected_windows", []) if isinstance(result, dict) else []
    windows = raw_windows if isinstance(raw_windows, list) else []
    if not windows and isinstance(result, dict) and isinstance(result.get("events"), list):
        events = result["events"]
        if events:
            windows = event_map.expand_selected_anchors(
                events,
                [str(events[0]["event_id"])],
                duration_seconds=duration,
            )
            selection_mode = "selector-fallback-first-event"
        else:
            selection_mode = "uniform-fallback"
    else:
        selection_mode = "selected-windows"
    times, derived_selection_mode = _selected_times(windows, duration=duration)
    if selection_mode != "selector-fallback-first-event":
        selection_mode = derived_selection_mode
    frame_dir = frame_root / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    frames: list[dict[str, Any]] = []
    pts_fallback_count = 0
    for index, timestamp in enumerate(times, start=1):
        output = frame_dir / f"frame-{index:02d}.jpg"
        try:
            actual_pts = extract_one_frame(ffmpeg, media, timestamp, output)
            pts_source = "ffmpeg-showinfo"
        except RuntimeError:
            try:
                actual_pts = _extract_frame_pyav(media, timestamp, output)
                pts_source = "pyav-fallback"
                pts_fallback_count += 1
            except RuntimeError:
                if output.is_file() and output.stat().st_size > 0:
                    actual_pts = timestamp
                    pts_source = "requested-timestamp-fallback"
                    pts_fallback_count += 1
                else:
                    raise
        frames.append(
            {
                "path": output.name,
                "requested_timestamp_seconds": timestamp,
                "selected_frame_timestamp_seconds": round(actual_pts, 6),
                "pts_source": pts_source,
                "sha256": sha256(output),
            }
        )
    sheet = sheet_root / f"{video_id}.jpg"
    sheet_sha256 = make_contact_sheet(frame_dir, frames, sheet)
    audio = extract_audio(media, audio_root)
    whisper_row = whisper.get(video_id)
    transcript = None
    whisper_status = "missing"
    if isinstance(whisper_row, dict) and whisper_row.get("status") == "passed":
        transcript = whisper_row.get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            transcript = None
        else:
            whisper_status = "accepted"
    frame_paths = [str((frame_dir / frame["path"]).resolve()) for frame in frames]
    common = {
        "run_id": "vqa-plaintext-selected-evidence-20260810",
        "video_id": video_id,
        "question": panel["question"],
        "frame_paths": frame_paths,
        "frame_timestamps": [frame["requested_timestamp_seconds"] for frame in frames],
    }
    requests = {
        "frames": {**common, "condition_id": "selected-frames-only", "transcript": None},
        "whisper": {**common, "condition_id": "selected-frames-whisper", "transcript": transcript},
    }
    evidence = {
        "ordinal": ordinal,
        "video_id": video_id,
        "question": panel["question"],
        "media_path": media.name,
        "media_sha256": panel["media_sha256"],
        "duration_seconds": duration,
        "selection_mode": selection_mode,
        "selected_windows": windows,
        "frame_count": len(frames),
        "frames": frames,
        "contact_sheet": sheet.name,
        "contact_sheet_sha256": sheet_sha256,
        "audio": audio,
        "whisper_status": whisper_status,
        "pts_fallback_count": pts_fallback_count,
        "whisper_source_row_sha256": (
            sha256(Path(str(whisper_row["_source_path"])))
            if isinstance(whisper_row, dict) and isinstance(whisper_row.get("_source_path"), str)
            else None
        ),
    }
    return requests, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    panels = controller.get("rows") if isinstance(controller, dict) else None
    if not isinstance(panels, list) or not panels:
        raise ValueError("controller rows are missing")
    results = {row["video_id"]: row for row in _read_jsonl(args.results)}
    whisper_rows = _whisper_map(args.whisper)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_root = args.output_dir / "frames"
    audio_root = args.output_dir / "audio"
    sheet_root = args.output_dir / "contact-sheets"
    for directory in (frame_root, audio_root, sheet_root):
        directory.mkdir(parents=True, exist_ok=True)
    items = [
        (index, panel, results.get(panel["video_id"], {}), args.input_root, frame_root, audio_root, sheet_root, whisper_rows)
        for index, panel in enumerate(panels, start=1)
    ]
    with ThreadPoolExecutor(max_workers=min(args.workers, len(items))) as executor:
        prepared = list(executor.map(_prepare_one, items))
    frame_requests = [request["frames"] for request, _evidence in prepared]
    whisper_requests = [request["whisper"] for request, _evidence in prepared]
    evidence = [row for _request, row in prepared]
    (args.output_dir / "requests.frames.json").write_text(json.dumps(frame_requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "requests.whisper.json").write_text(json.dumps(whisper_requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "evidence.json").write_text(json.dumps({"schema_version": 1, "label_policy": "label-free-selected-evidence", "example_count": len(evidence), "examples": evidence}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "label_policy": "label-free-selected-evidence", "example_count": len(panels), "examples": [{"video_id": row["video_id"], "question": row["question"]} for row in panels]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(evidence), "accepted_whisper": sum(row["whisper_status"] == "accepted" for row in evidence), "uniform_fallback": sum(row["selection_mode"] == "uniform-fallback" for row in evidence)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
