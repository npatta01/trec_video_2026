"""Download original-quality videos for the TRECVID VQA datasets."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections.abc import Callable
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


def build_ytdlp_command(
    manifest: Path, output_dir: Path, archive: Path
) -> list[str]:
    """Build a resumable yt-dlp command that never transcodes streams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--batch-file",
        str(manifest),
        "--format",
        "bestvideo*+bestaudio/best",
        "--merge-output-format",
        "mkv",
        "--output",
        str(output_dir / "%(id)s.%(ext)s"),
        "--write-info-json",
        "--continue",
        "--no-overwrites",
        "--download-archive",
        str(archive),
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--extractor-retries",
        "3",
        "--ignore-errors",
        "--no-playlist",
        "--newline",
    ]


def run_downloads(
    repo_root: Path,
    runner: Callable[..., object] = subprocess.run,
) -> int:
    """Download training and testing videos sequentially."""
    vqa_root = repo_root / "data/vqa"
    runtime = vqa_root / "download_runtime"
    datasets = (
        (
            "training",
            vqa_root / "2025/training/annotations/train.2025.csv",
            vqa_root / "2025/training/README.txt",
            vqa_root / "2025/training/videos_original",
        ),
        (
            "testing",
            vqa_root / "2025/testing/testing.dataset.vqa.2025.csv",
            vqa_root / "2025/testing/README.txt",
            vqa_root / "2025/testing/videos_original",
        ),
    )
    try:
        for name, csv_path, readme_path, output_dir in datasets:
            manifest = runtime / f"{name}.urls.txt"
            archive = runtime / f"{name}.archive.txt"
            write_manifest(csv_path, readme_path, manifest)
            runner(
                build_ytdlp_command(manifest, output_dir, archive),
                check=False,
            )
    except OSError as error:
        print(f"Unable to start yt-dlp: {error}", file=sys.stderr)
        return 1
    return 0
