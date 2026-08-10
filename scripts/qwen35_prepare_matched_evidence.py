#!/usr/bin/env python3
"""Prepare one shared, label-free frame/audio evidence set for two Qwen arms."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import subprocess
from pathlib import Path

import av
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw


SUPPORTED_SUFFIXES = (".mkv", ".mp4", ".webm", ".mov")
FRAME_COLUMNS = 4
FRAME_ROWS = 3
CELL_WIDTH = 320
CELL_HEIGHT = 250
IMAGE_HEIGHT = 220
FRAME_MAX_WIDTH = 1024
YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video_id(video_id: object) -> str:
    if not isinstance(video_id, str) or YOUTUBE_ID_PATTERN.fullmatch(video_id) is None:
        raise ValueError("video_id must match the YouTube ID grammar ^[A-Za-z0-9_-]{11}$")
    return video_id


def safe_child(root: Path, *parts: str, strict: bool = False) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = root.joinpath(*parts).resolve(strict=strict)
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"derived path escapes root {resolved_root}: {candidate}")
    return candidate


def duration_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = next(
            stream for stream in container.streams if stream.type == "video"
        )
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            raise RuntimeError(f"cannot determine duration: {path}")
    if duration <= 0:
        raise RuntimeError(f"non-positive duration: {path}")
    return duration


def locate_media(input_root: Path, video_id: str) -> Path:
    video_id = validate_video_id(video_id)
    root = input_root.resolve(strict=True)
    directory = safe_child(root, "vqa", "2025", "training", "videos_original", strict=True)
    matches = [
        safe_child(directory, f"{video_id}{suffix}", strict=True)
        for suffix in SUPPORTED_SUFFIXES
        if (directory / f"{video_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one local media file for {video_id}, found {matches}"
        )
    return matches[0]


def frame_time(frame: av.VideoFrame) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return None


def _save_frame(frame: av.VideoFrame, output: Path) -> None:
    image = frame.to_image().convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def extract_one_frame(
    ffmpeg: str, media: Path, timestamp: float, output: Path
) -> float:
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-ss",
            f"{timestamp:.6f}",
            "-copyts",
            "-i",
            str(media),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"showinfo,scale=w='min({FRAME_MAX_WIDTH},iw)':h=-2",
            "-frames:v",
            "1",
            "-c:v",
            "mjpeg",
            "-q:v",
            "4",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected_timestamps = re.findall(
        r"pts_time:([-+]?(?:\d+(?:\.\d*)?|\.\d+))", completed.stderr
    )
    if not selected_timestamps:
        raise RuntimeError(f"ffmpeg did not report a selected timestamp for {media}")
    return float(selected_timestamps[-1])


def extract_frames(
    media: Path, frame_directory: Path, frame_count: int
) -> tuple[float, list[dict[str, object]]]:
    duration = duration_seconds(media)
    requested = [
        round((index + 0.5) * duration / frame_count, 6)
        for index in range(frame_count)
    ]
    frame_directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    records: list[dict[str, object]] = []
    for index, target in enumerate(requested, start=1):
        output = frame_directory / f"frame-{index:02d}.jpg"
        selected_time = extract_one_frame(ffmpeg, media, target, output)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no frame for {media} at {target}")
        records.append(
            {
                "path": output.name,
                "requested_timestamp_seconds": target,
                "selected_frame_timestamp_seconds": round(selected_time, 6),
                "sha256": sha256(output),
            }
        )
    if len(records) != frame_count:
        raise RuntimeError(f"expected {frame_count} frames for {media}")
    return duration, records


def make_contact_sheet(
    frame_directory: Path,
    records: list[dict[str, object]],
    output: Path,
) -> str:
    sheet = Image.new(
        "RGB",
        (FRAME_COLUMNS * CELL_WIDTH, FRAME_ROWS * CELL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        frame_path = safe_child(frame_directory, str(record["path"]), strict=True)
        frame = Image.open(frame_path).convert("RGB")
        frame.thumbnail((CELL_WIDTH - 8, IMAGE_HEIGHT - 4), Image.Resampling.LANCZOS)
        x = (index % FRAME_COLUMNS) * CELL_WIDTH
        y = (index // FRAME_COLUMNS) * CELL_HEIGHT
        image_x = x + (CELL_WIDTH - frame.width) // 2
        image_y = y + 24 + (IMAGE_HEIGHT - frame.height) // 2
        sheet.paste(frame, (image_x, image_y))
        timestamp = float(record["requested_timestamp_seconds"])
        selected = float(record["selected_frame_timestamp_seconds"])
        draw.text((x + 5, y + 5), f"{index + 1:02d} req {timestamp:.3f}s", fill="black")
        draw.text((x + 5, y + IMAGE_HEIGHT + 28), f"pts {selected:.3f}s", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    return sha256(output)


def extract_audio(media: Path, audio_root: Path) -> dict[str, object]:
    audio_root.mkdir(parents=True, exist_ok=True)
    audio_output = safe_child(audio_root, f"{media.stem}.wav")
    with av.open(str(media)) as container:
        has_audio = any(stream.type == "audio" for stream in container.streams)
    if not has_audio:
        return {
            "status": "no-audio-stream",
            "path": None,
            "sha256": None,
            "duration_seconds": None,
        }
    ffmpeg = get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-vn",
            "-y",
            str(audio_output),
        ],
        check=True,
        capture_output=True,
    )
    if not audio_output.is_file() or audio_output.stat().st_size == 0:
        raise RuntimeError(f"audio extractor produced no file: {media}")
    with av.open(str(audio_output)) as container:
        duration = float(container.duration / av.time_base) if container.duration else None
    return {
        "status": "extracted",
        "path": str(audio_output.resolve()),
        "sha256": sha256(audio_output),
        "duration_seconds": duration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--contact-sheet-root", type=Path, required=True)
    parser.add_argument("--requests-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def prepare_example(
    item: tuple[int, dict[str, object], argparse.Namespace],
) -> tuple[dict[str, object], dict[str, object]]:
    index, example, args = item
    if set(example) != {"video_id", "question"}:
        raise ValueError(f"manifest example {index - 1} has unexpected fields")
    video_id = example["video_id"]
    question = example["question"]
    if not isinstance(question, str):
        raise ValueError("video_id and question must be strings")
    video_id = validate_video_id(video_id)
    media = locate_media(args.input_root, video_id)
    frame_dir = safe_child(args.frame_root, video_id)
    duration, frame_records = extract_frames(media, frame_dir, args.frame_count)
    audio = extract_audio(media, args.audio_root)
    contact_sheet = safe_child(args.contact_sheet_root, f"{video_id}.jpg")
    contact_sheet_sha256 = make_contact_sheet(frame_dir, frame_records, contact_sheet)
    request = {
        "run_id": args.run_id,
        "condition_id": "qwen35-4b-12f-frames-only",
        "video_id": video_id,
        "question": question,
        "frame_paths": [
            str(safe_child(frame_dir, str(record["path"]), strict=True))
            for record in frame_records
        ],
        "frame_timestamps": [
            float(record["requested_timestamp_seconds"]) for record in frame_records
        ],
        "transcript": None,
    }
    evidence = {
        "ordinal": index,
        "video_id": video_id,
        "question": question,
        "media_path": media.name,
        "media_sha256": sha256(media),
        "duration_seconds": duration,
        "frame_count": args.frame_count,
        "frames": frame_records,
        "contact_sheet": contact_sheet.name,
        "contact_sheet_sha256": contact_sheet_sha256,
        "audio": audio,
    }
    return request, evidence


def main() -> int:
    args = parse_args()
    args.input_root = args.input_root.resolve(strict=True)
    for name in ("frame_root", "audio_root", "contact_sheet_root"):
        root = getattr(args, name).resolve()
        root.mkdir(parents=True, exist_ok=True)
        setattr(args, name, root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise SystemExit("manifest examples must be a non-empty list")
    count = args.count or len(examples)
    if count <= 0 or count > len(examples):
        raise SystemExit("count must be positive and no larger than manifest size")
    if args.workers <= 0 or args.workers > 8:
        raise SystemExit("workers must be between 1 and 8")
    with ThreadPoolExecutor(max_workers=min(args.workers, count)) as executor:
        prepared = list(
            executor.map(
                prepare_example,
                ((index, example, args) for index, example in enumerate(examples[:count], 1)),
            )
        )
    requests = [request for request, _ in prepared]
    evidence = [row for _, row in prepared]
    args.requests_output.parent.mkdir(parents=True, exist_ok=True)
    args.requests_output.write_text(
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "label_policy": "label-free-evidence",
                "frame_policy": "12 uniform temporal midpoint requests, FFmpeg accurate seek, actual PTS recorded, 1024px JPEG frames",
                "example_count": len(evidence),
                "examples": evidence,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"request_count": len(requests), "evidence_count": len(evidence)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
