"""Download original-quality videos for the TRECVID VQA datasets."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
MEDIA_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov"}
RATE_LIMIT_MARKER = "Sign in to confirm you’re not a bot"
RATE_LIMIT_MARKERS = (
    "Sign in to confirm you're not a bot",
    "The current session has been rate-limited by YouTube for up to an hour.",
)


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one yt-dlp URL attempt."""

    completed: bool
    rate_limited: bool


def count_final_media(media_dir: Path) -> int:
    """Count finalized ID-named media, excluding yt-dlp format fragments."""
    return sum(
        1
        for path in media_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and VIDEO_ID.fullmatch(path.stem)
    )


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
    node_path: str | None = None,
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
    resolved_node = node_path or shutil.which("node")
    if not resolved_node:
        raise RuntimeError("Node.js is required for complete YouTube format extraction")
    environment = dict(os.environ)
    environment["VQA_NODE_PATH"] = resolved_node
    log_path = runtime / "download.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        child = popen(
            [sys.executable, "-m", "trec_video_2026.vqa_download", "run"],
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
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


def read_archive_ids(archive: Path) -> set[str]:
    """Read valid video IDs from yt-dlp archive rows."""
    if not archive.exists():
        return set()
    return {
        fields[-1]
        for line in archive.read_text(encoding="utf-8").splitlines()
        if (fields := line.split()) and VIDEO_ID.fullmatch(fields[-1])
    }


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
    url: str, output_dir: Path, archive: Path
) -> list[str]:
    """Build a resumable yt-dlp command that never transcodes streams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    node_path = os.environ.get("VQA_NODE_PATH") or shutil.which("node")
    node_runtime = f"node:{node_path}" if node_path else "node"
    return [
        sys.executable,
        "-m",
        "yt_dlp",
        "--format",
        "bestvideo*+bestaudio/best",
        "--js-runtimes",
        node_runtime,
        "--sleep-requests",
        "1.5",
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
        str(url),
    ]


def run_url(
    command: list[str], popen: Callable[..., object] = subprocess.Popen
) -> AttemptResult:
    """Run one URL while streaming output and classifying rate limiting."""
    child = popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    rate_limited = False
    for line in child.stdout:
        print(line, end="", flush=True)
        normalized_line = line.replace("’", "'")
        rate_limited = rate_limited or any(
            marker in normalized_line for marker in RATE_LIMIT_MARKERS
        )
    return_code = child.wait()
    return AttemptResult(
        completed=return_code == 0 and not rate_limited,
        rate_limited=rate_limited,
    )


def run_dataset(
    manifest: Path,
    output_dir: Path,
    archive: Path,
    *,
    url_runner: Callable[[list[str]], AttemptResult] = run_url,
    sleeper: Callable[[float], None] = time.sleep,
    ordinary_delay: float = 10.0,
    initial_cooldown: float = 3600.0,
    maximum_cooldown: float = 7200.0,
) -> None:
    """Download one manifest sequentially, cooling down after rate limits."""
    urls = [
        url
        for url in manifest.read_text(encoding="utf-8").splitlines()
        if url
    ]
    cooldown = initial_cooldown
    while True:
        rate_limited = False
        for url in urls:
            video_id = url.rsplit("v=", 1)[-1].split("&", 1)[0]
            if video_id in read_archive_ids(archive):
                continue
            result = url_runner(build_ytdlp_command(url, output_dir, archive))
            if result.rate_limited:
                retry_utc = datetime.now(timezone.utc) + timedelta(seconds=cooldown)
                retry_local = retry_utc.astimezone()
                print(
                    f"Rate limited; cooling down for {cooldown:g} seconds. "
                    f"Retry at {retry_utc.isoformat()} UTC / "
                    f"{retry_local.isoformat()} local.",
                    flush=True,
                )
                sleeper(cooldown)
                cooldown = min(cooldown * 2, maximum_cooldown)
                rate_limited = True
                break
            sleeper(ordinary_delay)
            if video_id in read_archive_ids(archive):
                cooldown = initial_cooldown
        if not rate_limited:
            return


def run_downloads(
    repo_root: Path,
    runner: Callable[[list[str]], AttemptResult] = run_url,
    sleeper: Callable[[float], None] = time.sleep,
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
            run_dataset(
                manifest,
                output_dir,
                archive,
                url_runner=runner,
                sleeper=sleeper,
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
        media = count_final_media(media_dir)
        print(f"{name} completed: {completed}")
        print(f"{name} media files: {media}")
    print(f"log: {runtime / 'download.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
