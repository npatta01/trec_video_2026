"""Download original-quality videos for the TRECVID VQA datasets."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def pid_is_active(pid_path: Path) -> bool:
    """Return whether a PID file identifies a running process."""
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def start_detached(
    repo_root: Path,
    popen: Callable[..., object] = subprocess.Popen,
) -> int:
    """Start the download worker detached and return its PID."""
    runtime = repo_root / "data/vqa/download_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    pid_path = runtime / "download.pid"
    if pid_is_active(pid_path):
        raise RuntimeError(
            f"VQA download is already running with PID "
            f"{pid_path.read_text(encoding='utf-8').strip()}"
        )
    log_path = runtime / "download.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        child = popen(
            [sys.executable, "-m", "trec_video_2026.vqa_download", "run"],
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid = int(child.pid)
    temporary_pid_path = runtime / "download.pid.tmp"
    temporary_pid_path.write_text(f"{pid}\n", encoding="utf-8")
    temporary_pid_path.replace(pid_path)
    return pid


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


def prepare_manifests(repo_root: Path) -> dict[str, int]:
    """Generate training/testing URL manifests and return usable counts."""
    vqa_root = repo_root / "data/vqa"
    runtime = vqa_root / "download_runtime"
    training = write_manifest(
        vqa_root / "2025/training/annotations/train.2025.csv",
        vqa_root / "2025/training/README.txt",
        runtime / "training.urls.txt",
    )
    testing = write_manifest(
        vqa_root / "2025/testing/testing.dataset.vqa.2025.csv",
        vqa_root / "2025/testing/README.txt",
        runtime / "testing.urls.txt",
    )
    return {"training": len(training), "testing": len(testing)}


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
    prepare_manifests(repo_root)
    datasets = (
        (
            "training",
            vqa_root / "2025/training/videos_original",
        ),
        (
            "testing",
            vqa_root / "2025/testing/videos_original",
        ),
    )
    try:
        for name, output_dir in datasets:
            manifest = runtime / f"{name}.urls.txt"
            archive = runtime / f"{name}.archive.txt"
            runner(
                build_ytdlp_command(manifest, output_dir, archive),
                check=False,
            )
    except OSError as error:
        print(f"Unable to start yt-dlp: {error}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    """Run manifest preparation, worker, detached launch, or status reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "start", "status"))
    args = parser.parse_args(argv)
    root = repo_root or Path(__file__).resolve().parents[2]
    runtime = root / "data/vqa/download_runtime"

    if args.command == "prepare":
        counts = prepare_manifests(root)
        print(f"training URLs: {counts['training']}")
        print(f"testing URLs: {counts['testing']}")
        return 0
    if args.command == "run":
        return run_downloads(root)
    if args.command == "start":
        try:
            pid = start_detached(root)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"started VQA download with PID {pid}")
        print(f"log: {runtime / 'download.log'}")
        return 0

    pid_path = runtime / "download.pid"
    active = pid_is_active(pid_path)
    pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "none"
    print(f"state: {'running' if active else 'stopped'}")
    print(f"pid: {pid}")
    for name in ("training", "testing"):
        archive = runtime / f"{name}.archive.txt"
        completed = (
            len(archive.read_text(encoding="utf-8").splitlines())
            if archive.exists()
            else 0
        )
        media_dir = root / f"data/vqa/2025/{name}/videos_original"
        media = sum(
            1
            for path in media_dir.glob("*")
            if path.is_file() and path.suffix.lower() in {".mkv", ".mp4", ".webm", ".mov"}
        )
        print(f"{name} completed: {completed}")
        print(f"{name} media files: {media}")
    print(f"log: {runtime / 'download.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
