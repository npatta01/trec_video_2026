#!/usr/bin/env python3
"""Build a label-free VQA manifest from local annotations and media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_SUFFIXES = (".mkv", ".mp4", ".webm", ".mov")


def locate_media(media_root: Path, video_id: str) -> Path | None:
    matches = [
        media_root / f"{video_id}{suffix}"
        for suffix in SUPPORTED_SUFFIXES
        if (media_root / f"{video_id}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple local media files for {video_id}: {matches}")
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-annotations", type=int, default=481)
    parser.add_argument("--expected-media", type=int, default=475)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve(strict=True)
    annotation_root = data_root / "vqa/2025/training/annotations"
    media_root = data_root / "vqa/2025/training/videos_original"
    annotation_paths = sorted(annotation_root.glob("*.json"))
    if len(annotation_paths) != args.expected_annotations:
        raise SystemExit(
            f"expected {args.expected_annotations} annotation files, "
            f"found {len(annotation_paths)}"
        )

    examples: list[dict[str, str]] = []
    missing_media: list[str] = []
    seen_ids: set[str] = set()
    for annotation_path in annotation_paths:
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        video_id = annotation_path.stem
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise SystemExit(f"missing question in {annotation_path}")
        if video_id in seen_ids:
            raise SystemExit(f"duplicate annotation ID: {video_id}")
        seen_ids.add(video_id)
        if locate_media(media_root, video_id) is None:
            missing_media.append(video_id)
            continue
        examples.append({"video_id": video_id, "question": question})

    if len(examples) != args.expected_media:
        raise SystemExit(
            f"expected {args.expected_media} annotation/media intersections, "
            f"found {len(examples)}; missing media={missing_media}"
        )
    examples.sort(key=lambda row: row["video_id"])
    manifest = {
        "schema_version": 1,
        "manifest_version": "trecvid-qwen35-4b-12f-full-label-free-2026-08-03-v1",
        "label_policy": "label-free-inference-input",
        "example_count": len(examples),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "annotation_count": len(annotation_paths),
                "local_media_intersection": len(examples),
                "missing_media_count": len(missing_media),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
