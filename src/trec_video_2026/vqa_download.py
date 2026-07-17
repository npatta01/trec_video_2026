"""TRECVID-specific adapter for the shared YouTube dataset engine."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
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
ALL_PROXY_ENV_VARIABLES = _engine.ALL_PROXY_ENV_VARIABLES
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
run_spec = _engine.run_spec
preflight_proxy = _engine.preflight_proxy
_startup_lock = _engine._startup_lock
_validate_runtime_directory = _engine._validate_runtime_directory
_validate_inherited_worker_lock = _engine._validate_inherited_worker_lock


def start_detached(
    repo_root: Path,
    popen=subprocess.Popen,
    node_path: str | None = None,
    proxy_env_path: Path | None = None,
    credential_runner=subprocess.run,
) -> int:
    """Start the shared worker using established TRECVID paths and defaults."""
    runtime = repo_root / "data/vqa/download_runtime"
    spec = DatasetSpec(
        source=runtime / "testing.pending-available.urls.txt",
        output_dir=repo_root / "data/vqa/2025/testing/videos_original",
        state_dir=runtime,
    )
    config = DownloadConfig(
        api_key_file=DEFAULT_YOUTUBE_API_ENV_PATH,
        proxy_file=proxy_env_path or DEFAULT_PROXY_ENV_PATH,
        worker_command=(
            sys.executable,
            "-m",
            "trec_video_2026.vqa_download",
            "run",
            "--detached-worker",
        ),
        node_path=node_path,
    )
    return _engine.start_detached(
        spec,
        config,
        popen=popen,
        credential_runner=credential_runner,
    )


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
    runtime = repo_root / "data/vqa/download_runtime"
    snapshots = {}
    summary = {}
    for name in ("training", "testing"):
        spec = DatasetSpec(
            source=runtime / f"{name}.urls.txt",
            output_dir=repo_root / f"data/vqa/2025/{name}/videos_original",
            state_dir=runtime / name,
        )
        ids = _engine.read_source_ids(spec.source)
        archived = read_archive_ids(runtime / f"{name}.archive.txt")
        pending = [video_id for video_id in ids if video_id not in archived]
        available = fetcher(pending, api_key)
        visible = [video_id for video_id in pending if video_id in available]
        omitted = [video_id for video_id in pending if video_id not in available]
        snapshots[name] = (visible, omitted)
        summary[name] = {
            "unarchived": len(pending), "pending_available": len(visible),
            "unavailable_or_private": len(omitted),
        }
    for name, (visible, omitted) in snapshots.items():
        _engine._atomic_write_json(
            runtime / f"{name}.availability-snapshot.json",
            {
                "version": 1,
                "visible": visible,
                "omitted": omitted,
                "counts": {
                    "total": len(visible) + len(omitted),
                    "pending_available": len(visible),
                    "api_omitted": len(omitted),
                },
            },
        )
        _engine._atomic_write_text(runtime / f"{name}.pending-available.urls.txt", "".join(f"https://www.youtube.com/watch?v={video_id}\n" for video_id in visible))
        _engine._atomic_write_text(runtime / f"{name}.unavailable-or-private.txt", "".join(f"{video_id}\n" for video_id in omitted))
    return summary


def run_downloads(repo_root: Path, runner=run_url, sleeper=None) -> int:
    runtime = repo_root / "data/vqa/download_runtime"
    try:
        for name in ("training", "testing"):
            spec = DatasetSpec(
                source=runtime / f"{name}.pending-available.urls.txt",
                output_dir=repo_root / f"data/vqa/2025/{name}/videos_original",
                state_dir=runtime,
                archive_path=runtime / f"{name}.archive.txt",
                prepared_manifest_path=runtime
                / f"{name}.pending-available.urls.txt",
                availability_snapshot_path=runtime
                / f"{name}.availability-snapshot.json",
            )
            kwargs = {"url_runner": runner}
            if sleeper is not None:
                kwargs["sleeper"] = sleeper
            run_spec(spec, **kwargs)
    except OSError as error:
        print(f"Unable to start yt-dlp: {error}", file=sys.stderr)
        return 1
    return 0


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
                config_value = os.environ.get("YOUTUBE_DATASET_PROXY_CONFIG_PATH")
                if not config_value:
                    raise RuntimeError("Prepared proxy configuration is required")
                validate_proxy_config(Path(config_value))
                try:
                    return run_downloads(root)
                finally:
                    os.close(descriptor)
            with _startup_lock(runtime):
                config = write_proxy_config(DEFAULT_PROXY_ENV_PATH, runtime / "yt-dlp-proxy.conf")
                preflight_proxy(DEFAULT_PROXY_ENV_PATH)
                os.environ["VQA_PROXY_CONFIG_PATH"] = str(config)
                os.environ["YOUTUBE_DATASET_PROXY_CONFIG_PATH"] = str(config)
                for variable in ALL_PROXY_ENV_VARIABLES:
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
            preflight_proxy(DEFAULT_PROXY_ENV_PATH)
            pid = start_detached(root)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        for name in ("training", "testing"):
            counts = availability[name]
            print(f"{name} API precheck: {counts['pending_available']} available, {counts['unavailable_or_private']} unavailable/private")
        print(f"started VQA download with PID {pid}")
        print(f"log: {runtime / 'download.log'}")
        return 0
    worker = _engine.worker_status(runtime)
    print(f"state: {'running' if worker['running'] else 'stopped'}")
    print(f"pid: {worker['pid'] or 'none'}")
    worker_state = worker["runtime"]
    print(f"current ID: {worker_state['current_id'] or 'none'}")
    print(f"last progress: {worker_state['last_progress_at'] or 'none'}")
    print(f"cooldown deadline: {worker_state['cooldown_deadline'] or 'none'}")
    failure_counts = {
        category: len(ids)
        for category, ids in dict(worker_state["failures"]).items()
    }
    for category in (
        "private", "unavailable", "members_only", "authentication_required",
        "proxy_transport", "rate_limit", "retryable",
    ):
        print(
            f"failures {category.replace('_', '-')}: "
            f"{failure_counts.get(category, 0)}"
        )
    print(f"retryable failures: {len(worker_state['retryable_failures'])}")
    for name in ("training", "testing"):
        archive = runtime / f"{name}.archive.txt"
        completed = len(archive.read_text(encoding="utf-8").splitlines()) if archive.exists() else 0
        media = count_final_media(root / f"data/vqa/2025/{name}/videos_original")
        print(f"{name} completed: {completed}")
        print(f"{name} media files: {media}")
        pending = runtime / f"{name}.pending-available.urls.txt"
        unavailable = runtime / f"{name}.unavailable-or-private.txt"
        snapshot_path = runtime / f"{name}.availability-snapshot.json"
        if snapshot_path.exists():
            snapshot_counts = _engine._read_availability_snapshot(snapshot_path)[
                "counts"
            ]
            print(
                f"{name} API-available pending: "
                f"{snapshot_counts['pending_available']}"
            )
            print(
                f"{name} API unavailable/private: "
                f"{snapshot_counts['api_omitted']}"
            )
        else:
            print(f"{name} authoritative snapshot: missing")
            if pending.exists():
                print(
                    f"{name} legacy/non-authoritative API-available projection: "
                    f"{_engine.count_lines(pending)}"
                )
            if unavailable.exists():
                print(
                    f"{name} legacy/non-authoritative API-omitted projection: "
                    f"{_engine.count_lines(unavailable)}"
                )
    print(f"log: {runtime / 'download.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
