"""Download original-quality videos for the TRECVID VQA datasets."""

from __future__ import annotations

import csv
import re
from pathlib import Path

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def read_video_ids(csv_path: Path) -> list[str]:
    """Read unique YouTube IDs from a NIST VQA CSV in source order."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        ids = [row[1].strip() for row in csv.reader(handle) if len(row) >= 2]
    if any(not VIDEO_ID.fullmatch(video_id) for video_id in ids):
        raise ValueError(f"Invalid YouTube ID in {csv_path}")
    return list(dict.fromkeys(ids))


def read_unavailable_ids(readme_path: Path) -> set[str]:
    """Extract the unavailable video IDs listed in a NIST README."""
    text = readme_path.read_text(encoding="utf-8")
    names = re.findall(r"(?m)^([A-Za-z0-9_-]{11})\.json\s*$", text)
    quoted = re.findall(r"['\"]([A-Za-z0-9_-]{11})['\"]", text)
    return set(names) | set(quoted)


def write_manifest(
    csv_path: Path, readme_path: Path, output_path: Path
) -> list[str]:
    """Write usable YouTube URLs and return their video IDs."""
    excluded = read_unavailable_ids(readme_path)
    ids = [
        video_id for video_id in read_video_ids(csv_path) if video_id not in excluded
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"https://www.youtube.com/watch?v={video_id}\n" for video_id in ids),
        encoding="utf-8",
    )
    return ids
