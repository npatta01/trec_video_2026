"""TRECVID-specific adapter for the shared YouTube dataset engine."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from . import youtube_dataset as _engine

DEFAULT_PROXY_ENV_PATH = Path("/home/npatta01/.config/trec-video/decodo.env")
DEFAULT_YOUTUBE_API_ENV_PATH = Path("/home/npatta01/.config/trec-video/youtube.env")

# Compatibility exports deliberately preserve object identity.
AttemptResult = _engine.AttemptResult
DatasetSpec = _engine.DatasetSpec
DownloadConfig = _engine.DownloadConfig
VIDEO_ID = _engine.VIDEO_ID
MEDIA_SUFFIXES = _engine.MEDIA_SUFFIXES
RATE_LIMIT_MARKER = _engine.RATE_LIMIT_MARKER
RATE_LIMIT_MARKERS = _engine.RATE_LIMIT_MARKERS
PROXY_ENV_VARIABLES = _engine.PROXY_ENV_VARIABLES
WORKER_LOCK_FD_ENV = _engine.WORKER_LOCK_FD_ENV
load_proxy_credentials = _engine.load_proxy_credentials
load_youtube_api_key = _engine.load_youtube_api_key
fetch_available_video_ids = _engine.fetch_available_video_ids
write_proxy_config = _engine.write_proxy_config
validate_proxy_config = _engine.validate_proxy_config
count_final_media = _engine.count_final_media
pid_is_active = _engine.pid_is_active
read_archive_ids = _engine.read_archive_ids
build_ytdlp_command = _engine.build_ytdlp_command
run_url = _engine.run_url
run_dataset = _engine.run_dataset
start_detached = _engine.start_detached
_startup_lock = _engine._startup_lock
_validate_runtime_directory = _engine._validate_runtime_directory
_validate_inherited_worker_lock = _engine._validate_inherited_worker_lock


def read_video_ids(csv_path: Path) -> list[str]:
    """Read unique YouTube IDs from a NIST VQA CSV in source order."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        ids = [row[1].strip() for row in csv.reader(handle) if len(row) >= 2]
    if any(not VIDEO_ID.fullmatch(video_id) for video_id in ids):
        raise ValueError(f"Invalid YouTube ID in {csv_path}")
    return list(dict.fromkeys(ids))


def read_unavailable_ids(readme_path: Path) -> set[str]:
    """Extract unavailable video IDs listed in a NIST README."""
    text = readme_path.read_text(encoding="utf-8")
    names = re.findall(r"(?m)^([A-Za-z0-9_-]{11})\.json\s*$", text)
    quoted = re.findall(r"['\"]([A-Za-z0-9_-]{11})['\"]", text)
    return set(names) | set(quoted)


def write_manifest(csv_path: Path, readme_path: Path, output_path: Path) -> list[str]:
    excluded = read_unavailable_ids(readme_path)
    ids = [video_id for video_id in read_video_ids(csv_path) if video_id not in excluded]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"https://www.youtube.com/watch?v={video_id}\n" for video_id in ids),
        encoding="utf-8",
    )
    return ids


def prepare_manifests(repo_root: Path) -> dict[str, int]:
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


def prepare_available_manifests(repo_root: Path, api_key: str, fetcher=fetch_available_video_ids):
    return _engine.prepare_available_manifests(repo_root, api_key, fetcher=fetcher)


def run_downloads(repo_root: Path, runner=run_url, sleeper=None) -> int:
    kwargs = {"runner": runner}
    if sleeper is not None:
        kwargs["sleeper"] = sleeper
    return _engine.run_downloads(repo_root, **kwargs)


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "start", "status"))
    parser.add_argument("--detached-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = repo_root or Path(__file__).resolve().parents[2]
    runtime = root / "data/vqa/download_runtime"
    if args.command == "prepare":
        counts = prepare_manifests(root)
        print(f"training URLs: {counts['training']}")
        print(f"testing URLs: {counts['testing']}")
        return 0
    if args.command == "run":
        runtime.mkdir(parents=True, exist_ok=True)
        try:
            _validate_runtime_directory(runtime)
            if args.detached_worker:
                value = os.environ.get(WORKER_LOCK_FD_ENV)
                if not value:
                    raise RuntimeError("Inherited VQA worker lock is required")
                try:
                    descriptor = int(value)
                except ValueError:
                    raise RuntimeError("Inherited VQA worker lock is invalid") from None
                _validate_inherited_worker_lock(runtime, descriptor)
                config_value = os.environ.get("VQA_PROXY_CONFIG_PATH")
                if not config_value:
                    raise RuntimeError("Prepared proxy configuration is required")
                validate_proxy_config(Path(config_value))
                try:
                    return run_downloads(root)
                finally:
                    os.close(descriptor)
            with _startup_lock(runtime):
                config = write_proxy_config(DEFAULT_PROXY_ENV_PATH, runtime / "yt-dlp-proxy.conf")
                os.environ["VQA_PROXY_CONFIG_PATH"] = str(config)
                for variable in PROXY_ENV_VARIABLES:
                    os.environ.pop(variable, None)
                return run_downloads(root)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
    if args.command == "start":
        try:
            prepare_manifests(root)
            key = load_youtube_api_key(DEFAULT_YOUTUBE_API_ENV_PATH)
            availability = prepare_available_manifests(root, key)
            pid = start_detached(root, proxy_env_path=DEFAULT_PROXY_ENV_PATH)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        for name in ("training", "testing"):
            counts = availability[name]
            print(f"{name} API precheck: {counts['pending_available']} available, {counts['unavailable_or_private']} unavailable/private")
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
        completed = len(archive.read_text(encoding="utf-8").splitlines()) if archive.exists() else 0
        media = count_final_media(root / f"data/vqa/2025/{name}/videos_original")
        print(f"{name} completed: {completed}")
        print(f"{name} media files: {media}")
    print(f"log: {runtime / 'download.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
