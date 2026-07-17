"""Download original-quality videos for the TRECVID VQA datasets."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
MEDIA_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov"}
RATE_LIMIT_MARKER = (
    "The current session has been rate-limited by YouTube for up to an hour."
)
DEFAULT_PROXY_ENV_PATH = Path("/home/npatta01/.config/trec-video/decodo.env")


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one yt-dlp URL attempt."""

    completed: bool
    rate_limited: bool


def load_proxy_credentials(
    env_path: Path, runner: Callable[..., object] = subprocess.run
) -> tuple[str, str, str]:
    """Load validated Decodo credentials without exposing their values."""
    descriptor = -1
    try:
        descriptor = os.open(env_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        message = (
            "Proxy credentials file must not be a symbolic link"
            if error.errno == getattr(os, "ELOOP", 40)
            else "Unable to open proxy credentials file"
        )
        raise RuntimeError(message) from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Proxy credentials file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError("Proxy credentials file must have permissions 0600")
        if metadata.st_uid != os.getuid():
            raise RuntimeError(
                "Proxy credentials file must be owned by the current user"
            )
        script = (
            'set -a; source "$1" >/dev/null 2>&1 || exit 1; '
            'printf "%s\\0%s\\0%s\\0" "${DECODO_ENDPOINT:-}" '
            '"${DECODO_USER:-}" "${DECODO_PASSWORD:-}"'
        )
        try:
            result = runner(
                ["bash", "-c", script, "_", f"/proc/self/fd/{descriptor}"],
                capture_output=True,
                check=True,
                pass_fds=(descriptor,),
            )
            output = result.stdout
            if isinstance(output, str):
                output = output.encode()
            fields = output.split(b"\0")
            if len(fields) != 4 or fields[-1] != b"":
                raise ValueError
            endpoint, username, password = (field.decode() for field in fields[:3])
            endpoint_match = re.fullmatch(r"([^:/@\s]+):(\d{1,5})", endpoint)
            if (
                not endpoint_match
                or not username
                or not password
                or not 1 <= int(endpoint_match.group(2)) <= 65535
            ):
                raise ValueError
        except Exception:
            raise RuntimeError("Unable to load valid proxy credentials") from None
    finally:
        os.close(descriptor)
    return endpoint, username, password


def write_proxy_config(
    env_path: Path,
    config_path: Path,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    """Atomically create a private yt-dlp proxy config."""
    endpoint, username, password = load_proxy_credentials(env_path, runner=runner)
    proxy_url = (
        f"http://{quote(username, safe='')}:{quote(password, safe='')}@{endpoint}"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            dir=config_path.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(f"--proxy\n{proxy_url}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
        temporary = None
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if temporary is not None:
                temporary.unlink()
        except OSError:
            pass
        raise RuntimeError("Unable to create private proxy configuration") from None
    return config_path


def validate_proxy_config(config_path: Path) -> Path:
    """Require a regular, current-user, mode-0600 prepared proxy config."""
    descriptor = -1
    try:
        descriptor = os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise OSError
    except OSError:
        raise RuntimeError("Prepared proxy configuration is not secure") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return config_path


def _record_pid(pid_path: Path, pid: int) -> None:
    """Atomically record a child PID without following predictable paths."""
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{pid_path.name}.", suffix=".tmp", dir=pid_path.parent
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(f"{pid}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pid_path)
        temporary = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


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
    proxy_env_path: Path | None = None,
    credential_runner: Callable[..., object] = subprocess.run,
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
    credentials = proxy_env_path or DEFAULT_PROXY_ENV_PATH
    proxy_config = write_proxy_config(
        credentials,
        runtime / "yt-dlp-proxy.conf",
        runner=credential_runner,
    )
    environment = dict(os.environ)
    for variable in ("DECODO_ENDPOINT", "DECODO_USER", "DECODO_PASSWORD"):
        environment.pop(variable, None)
    environment["VQA_NODE_PATH"] = resolved_node
    environment["VQA_PROXY_CONFIG_PATH"] = str(proxy_config)
    log_path = runtime / "download.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        try:
            child = popen(
                [
                    sys.executable,
                    "-m",
                    "trec_video_2026.vqa_download",
                    "run",
                    "--detached-worker",
                ],
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
            )
        except Exception:
            raise RuntimeError("Unable to start detached VQA worker") from None
    try:
        pid = int(child.pid)
        _record_pid(pid_path, pid)
    except Exception:
        try:
            child.terminate()
        except Exception:
            pass
        try:
            child.wait()
        except Exception:
            pass
        raise RuntimeError("Unable to record detached VQA worker PID") from None
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
    url: str,
    output_dir: Path,
    archive: Path,
    proxy_config: Path | None = None,
) -> list[str]:
    """Build a resumable yt-dlp command that never transcodes streams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    node_path = os.environ.get("VQA_NODE_PATH") or shutil.which("node")
    node_runtime = f"node:{node_path}" if node_path else "node"
    command = [
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
    ]
    selected_proxy_config = proxy_config or (
        Path(environment_path)
        if (environment_path := os.environ.get("VQA_PROXY_CONFIG_PATH"))
        else None
    )
    if selected_proxy_config is not None:
        command.extend(["--config-locations", str(selected_proxy_config)])
    command.append(str(url))
    return command


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
        rate_limited = rate_limited or RATE_LIMIT_MARKER in line
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
        archived_ids = read_archive_ids(archive)
        archived_at_pass_start = archived_ids.copy()
        rate_limited = False
        for url in urls:
            video_id = url.rsplit("v=", 1)[-1].split("&", 1)[0]
            if video_id in archived_ids:
                continue
            result = url_runner(build_ytdlp_command(url, output_dir, archive))
            archived_ids = read_archive_ids(archive)
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
            if video_id in archived_ids:
                cooldown = initial_cooldown
        made_progress = bool(archived_ids - archived_at_pass_start)
        if not rate_limited and not made_progress:
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
        try:
            if args.detached_worker:
                config_value = os.environ.get("VQA_PROXY_CONFIG_PATH")
                if not config_value:
                    raise RuntimeError("Prepared proxy configuration is required")
                validate_proxy_config(Path(config_value))
            else:
                config_path = write_proxy_config(
                    DEFAULT_PROXY_ENV_PATH, runtime / "yt-dlp-proxy.conf"
                )
                os.environ["VQA_PROXY_CONFIG_PATH"] = str(config_path)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
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
