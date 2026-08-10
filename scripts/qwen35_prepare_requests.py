#!/usr/bin/env python3
"""Prepare label-free Qwen3.5 requests and uniform-12 frame evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import av
from imageio_ffmpeg import get_ffmpeg_exe


SUPPORTED_SUFFIXES = (".mkv", ".mp4", ".webm", ".mov")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    directory = input_root / "vqa" / "2025" / "training" / "videos_original"
    matches = [
        directory / f"{video_id}{suffix}"
        for suffix in SUPPORTED_SUFFIXES
        if (directory / f"{video_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one local media file for {video_id}, found {matches}"
        )
    return matches[0]


def extract_frames(
    media: Path, frame_directory: Path, frame_count: int
) -> tuple[list[Path], list[float]]:
    duration = duration_seconds(media)
    timestamps = [
        round((index + 0.5) * duration / frame_count, 6)
        for index in range(frame_count)
    ]
    frame_directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    frames: list[Path] = []
    records: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output = frame_directory / f"frame-{index:02d}.png"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(media),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-y",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        if not output.is_file():
            raise RuntimeError(f"frame extractor produced no file: {output}")
        frames.append(output)
        records.append(
            {
                "path": output.name,
                "timestamp_seconds": timestamp,
                "sha256": sha256(output),
            }
        )
    (frame_directory / "evidence.json").write_text(
        json.dumps(
            {
                "media": str(media),
                "media_sha256": sha256(media),
                "duration_seconds": duration,
                "frame_count": frame_count,
                "frames": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return frames, timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--frame-count", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise SystemExit("manifest examples must be a non-empty list")
    if args.count <= 0 or args.count > len(examples):
        raise SystemExit("count must be positive and no larger than manifest size")
    requests = []
    for example in examples[: args.count]:
        if set(example) != {"video_id", "question"}:
            raise SystemExit("manifest must contain only video_id and question fields")
        video_id = example["video_id"]
        question = example["question"]
        if not isinstance(video_id, str) or not isinstance(question, str):
            raise SystemExit("video_id and question must be strings")
        media = locate_media(args.input_root, video_id)
        frames, timestamps = extract_frames(
            media,
            args.frame_root / video_id,
            args.frame_count,
        )
        requests.append(
            {
                "run_id": args.run_id,
                "condition_id": args.condition_id,
                "video_id": video_id,
                "question": question,
                "frame_paths": [str(path.resolve()) for path in frames],
                "frame_timestamps": timestamps,
                "transcript": None,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(requests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"request_count": len(requests), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
