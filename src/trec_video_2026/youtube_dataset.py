"""Reusable YouTube dataset download engine."""

from __future__ import annotations

import fcntl
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
MEDIA_SUFFIXES = {".mkv", ".mp4", ".webm", ".mov"}
RATE_LIMIT_MARKER = (
    "The current session has been rate-limited by YouTube for up to an hour."
)
RATE_LIMIT_MARKERS = (
    RATE_LIMIT_MARKER,
    "Sign in to confirm you're not a bot",
)
PROXY_ENV_VARIABLES = ("DECODO_ENDPOINT", "DECODO_USER", "DECODO_PASSWORD")
WORKER_LOCK_FD_ENV = "YOUTUBE_DATASET_WORKER_LOCK_FD"
PROCESS_STOP_TIMEOUT = 5.0


@dataclass(frozen=True)
class DatasetSpec:
    """Explicit paths describing one generic YouTube dataset."""

    source: Path
    output_dir: Path
    state_dir: Path


@dataclass(frozen=True)
class DownloadConfig:
    """Explicit credential and pacing configuration for a dataset run."""

    api_key_file: Path
    proxy_file: Path
    ordinary_delay: float = 10.0
    initial_cooldown: float = 3600.0
    maximum_cooldown: float = 7200.0
    worker_command: tuple[str, ...] = ()
    node_path: str | None = None


@dataclass(frozen=True)
class AttemptResult:
    """Outcome of one yt-dlp URL attempt."""

    completed: bool
    rate_limited: bool
    proxy_failure: bool = False


@dataclass
class _StartupLockHandle:
    descriptor: int
    handed_off: bool = False


def _valid_proxy_host(host: str) -> bool:
    """Return whether host is an IPv4 literal or RFC-style DNS hostname."""
    try:
        return isinstance(ipaddress.ip_address(host), ipaddress.IPv4Address)
    except ValueError:
        pass
    if not host or len(host) > 253 or all(character in ".0123456789" for character in host):
        return False
    label_pattern = re.compile(
        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    )
    return all(label_pattern.fullmatch(label) for label in host.split("."))


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
            "unset DECODO_ENDPOINT DECODO_USER DECODO_PASSWORD; "
            'set -a; source "$1" >/dev/null 2>&1 || exit 1; '
            'printf "%s\\0%s\\0%s\\0" "${DECODO_ENDPOINT:-}" '
            '"${DECODO_USER:-}" "${DECODO_PASSWORD:-}"'
        )
        try:
            environment = dict(os.environ)
            for variable in PROXY_ENV_VARIABLES:
                environment.pop(variable, None)
            result = runner(
                ["bash", "-c", script, "_", f"/proc/self/fd/{descriptor}"],
                capture_output=True,
                check=True,
                pass_fds=(descriptor,),
                env=environment,
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
                or not _valid_proxy_host(endpoint_match.group(1))
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


def load_youtube_api_key(
    env_path: Path, runner: Callable[..., object] = subprocess.run
) -> str:
    """Load a YouTube API key from a validated private environment file."""
    descriptor = -1
    try:
        descriptor = os.open(env_path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        message = (
            "YouTube API key file must not be a symbolic link"
            if error.errno == getattr(os, "ELOOP", 40)
            else "Unable to open YouTube API key file"
        )
        raise RuntimeError(message) from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("YouTube API key file must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RuntimeError("YouTube API key file must have permissions 0600")
        if metadata.st_uid != os.getuid():
            raise RuntimeError(
                "YouTube API key file must be owned by the current user"
            )
        script = (
            "unset YOUTUBE_API_KEY; "
            'set -a; source "$1" >/dev/null 2>&1 || exit 1; '
            'printf "%s\\0" "${YOUTUBE_API_KEY:-}"'
        )
        try:
            environment = dict(os.environ)
            environment.pop("YOUTUBE_API_KEY", None)
            result = runner(
                ["bash", "-c", script, "_", f"/proc/self/fd/{descriptor}"],
                capture_output=True,
                check=True,
                pass_fds=(descriptor,),
                env=environment,
            )
            output = result.stdout
            if isinstance(output, str):
                output = output.encode()
            fields = output.split(b"\0")
            if len(fields) != 2 or fields[-1] != b"" or not fields[0]:
                raise ValueError
            api_key = fields[0].decode()
        except Exception:
            raise RuntimeError("Unable to load valid YouTube API key") from None
    finally:
        os.close(descriptor)
    return api_key


def fetch_available_video_ids(
    ids: list[str],
    api_key: str,
    opener: Callable[..., object] = urlopen,
) -> set[str]:
    """Return IDs visible through the YouTube Data API in batches of 50."""
    if any(not VIDEO_ID.fullmatch(video_id) for video_id in ids):
        raise ValueError("Invalid YouTube video ID")
    available: set[str] = set()
    failed = False
    try:
        for offset in range(0, len(ids), 50):
            batch = ids[offset : offset + 50]
            query = urlencode({"part": "status", "id": ",".join(batch)})
            request = Request(
                f"https://www.googleapis.com/youtube/v3/videos?{query}",
                headers={"X-goog-api-key": api_key},
            )
            with opener(request, timeout=30) as response:
                payload = json.loads(response.read())
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ValueError
            if any(
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not VIDEO_ID.fullmatch(item["id"])
                for item in items
            ):
                raise ValueError
            returned = {item["id"] for item in items}
            if not returned <= set(batch):
                raise ValueError
            available.update(returned)
    except (OSError, URLError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        failed = True
    if failed:
        raise RuntimeError("YouTube availability check failed") from None
    return available


def read_source_ids(source: Path) -> list[str]:
    """Normalize a generic ID/URL source while preserving first occurrence order."""
    ids = []
    for line in source.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        video_id = value.rsplit("v=", 1)[-1].split("&", 1)[0]
        if not VIDEO_ID.fullmatch(video_id):
            raise ValueError(f"Invalid YouTube ID or URL in {source}")
        ids.append(video_id)
    return list(dict.fromkeys(ids))


def prepare_dataset(
    spec: DatasetSpec,
    api_key: str | DownloadConfig,
    fetcher: Callable[[list[str], str], set[str]] = fetch_available_video_ids,
) -> dict[str, int]:
    """Atomically prepare API-visible and omitted snapshots for one dataset."""
    if isinstance(api_key, DownloadConfig):
        api_key = load_youtube_api_key(api_key.api_key_file)
    ids = read_source_ids(spec.source)
    archived = read_archive_ids(spec.state_dir / "download.archive.txt")
    pending = [video_id for video_id in ids if video_id not in archived]
    available = fetcher(pending, api_key)
    visible = [video_id for video_id in pending if video_id in available]
    omitted = [video_id for video_id in pending if video_id not in available]
    _atomic_write_text(
        spec.state_dir / "pending-available.urls.txt",
        "".join(f"https://www.youtube.com/watch?v={video_id}\n" for video_id in visible),
    )
    _atomic_write_text(
        spec.state_dir / "api-omitted.txt",
        "".join(f"{video_id}\n" for video_id in omitted),
    )
    return {"total": len(pending), "pending_available": len(visible), "api_omitted": len(omitted)}


def _atomic_write_text(path: Path, contents: str) -> None:
    """Atomically replace a runtime text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def _open_proxy_config(config_path: Path) -> int:
    """Open and validate a proxy config for descriptor-bound child use."""
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
        if descriptor >= 0:
            os.close(descriptor)
        raise RuntimeError("Prepared proxy configuration is not secure") from None
    return descriptor


@contextmanager
def _startup_lock(runtime: Path):
    """Hold the secure per-runtime startup lock without blocking."""
    descriptor = -1
    handle: _StartupLockHandle | None = None
    try:
        descriptor = os.open(
            runtime / "startup.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise OSError
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("VQA download startup already in progress") from None
        handle = _StartupLockHandle(descriptor)
        yield handle
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("Unable to secure VQA download startup lock") from None
    finally:
        if descriptor >= 0:
            if handle is None or not handle.handed_off:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(descriptor)


def _validate_runtime_directory(runtime: Path) -> None:
    """Require a trusted runtime directory before creating or using locks."""
    descriptor = -1
    try:
        descriptor = os.open(
            runtime, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise OSError
    except OSError:
        raise RuntimeError("VQA runtime directory is not secure") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_inherited_worker_lock(runtime: Path, descriptor: int) -> None:
    """Validate that an inherited descriptor is the secure runtime lock."""
    try:
        metadata = os.fstat(descriptor)
        lock_metadata = (runtime / "startup.lock").stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or (metadata.st_dev, metadata.st_ino)
            != (lock_metadata.st_dev, lock_metadata.st_ino)
        ):
            raise OSError
    except OSError:
        raise RuntimeError("Inherited VQA worker lock is not secure") from None


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


def count_lines(path: Path) -> int:
    """Count non-empty state inventory rows without interpreting dataset semantics."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def dataset_status(spec: DatasetSpec) -> dict[str, int | bool]:
    """Return generic state counts using only paths declared by the spec."""
    archive = spec.state_dir / "download.archive.txt"
    pending = spec.state_dir / "pending-available.urls.txt"
    omitted = spec.state_dir / "api-omitted.txt"
    return {
        "running": pid_is_active(spec.state_dir / "download.pid"),
        "completed": count_lines(archive) if archive.exists() else 0,
        "media_files": count_final_media(spec.output_dir),
        "pending_available": count_lines(pending) if pending.exists() else 0,
        "api_omitted": count_lines(omitted) if omitted.exists() else 0,
    }


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
    spec: DatasetSpec,
    config: DownloadConfig,
    popen: Callable[..., object] = subprocess.Popen,
    credential_runner: Callable[..., object] = subprocess.run,
) -> int:
    """Start the download worker detached and return its PID."""
    runtime = spec.state_dir
    runtime.mkdir(parents=True, exist_ok=True)
    _validate_runtime_directory(runtime)
    with _startup_lock(runtime) as lock_handle:
        pid = _start_detached_locked(
            spec,
            config,
            runtime,
            popen=popen,
            credential_runner=credential_runner,
            lock_descriptor=lock_handle.descriptor,
        )
        lock_handle.handed_off = True
        return pid


def _start_detached_locked(
    spec: DatasetSpec,
    config: DownloadConfig,
    runtime: Path,
    *,
    popen: Callable[..., object],
    credential_runner: Callable[..., object],
    lock_descriptor: int,
) -> int:
    """Start a detached worker while the caller holds the startup lock."""
    pid_path = runtime / "download.pid"
    if pid_is_active(pid_path):
        raise RuntimeError(
            f"YouTube dataset download is already running with PID "
            f"{pid_path.read_text(encoding='utf-8').strip()}"
        )
    resolved_node = config.node_path or shutil.which("node")
    if not resolved_node:
        raise RuntimeError("Node.js is required for complete YouTube format extraction")
    proxy_config = write_proxy_config(
        config.proxy_file,
        runtime / "yt-dlp-proxy.conf",
        runner=credential_runner,
    )
    environment = dict(os.environ)
    for variable in PROXY_ENV_VARIABLES:
        environment.pop(variable, None)
    environment["YOUTUBE_DATASET_NODE_PATH"] = resolved_node
    environment["YOUTUBE_DATASET_PROXY_CONFIG_PATH"] = str(proxy_config)
    environment[WORKER_LOCK_FD_ENV] = str(lock_descriptor)
    log_path = runtime / "download.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        try:
            child = popen(
                list(config.worker_command),
                cwd=runtime,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
                pass_fds=(lock_descriptor,),
            )
        except Exception:
            raise RuntimeError("Unable to start detached YouTube dataset worker") from None
    try:
        pid = int(child.pid)
        _record_pid(pid_path, pid)
    except Exception:
        _rollback_child(child)
        raise RuntimeError("Unable to record detached YouTube dataset worker PID") from None
    return pid


def _rollback_child(child: object) -> None:
    """Bound termination of a spawned worker, escalating to kill on timeout."""
    try:
        child.terminate()
    except Exception:
        pass
    try:
        child.wait(timeout=PROCESS_STOP_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    try:
        child.kill()
    except Exception:
        pass
    try:
        child.wait(timeout=PROCESS_STOP_TIMEOUT)
    except Exception:
        pass


def read_archive_ids(archive: Path) -> set[str]:
    """Read valid video IDs from yt-dlp archive rows."""
    if not archive.exists():
        return set()
    return {
        fields[-1]
        for line in archive.read_text(encoding="utf-8").splitlines()
        if (fields := line.split()) and VIDEO_ID.fullmatch(fields[-1])
    }


def build_ytdlp_command(
    url: str,
    output_dir: Path,
    archive: Path,
    proxy_config: Path | None = None,
    node_path: str | None = None,
) -> list[str]:
    """Build a resumable yt-dlp command that never transcodes streams."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    node_path = node_path or os.environ.get("YOUTUBE_DATASET_NODE_PATH") or shutil.which("node")
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
        Path(value) if (value := os.environ.get("YOUTUBE_DATASET_PROXY_CONFIG_PATH")) else None
    )
    if selected_proxy_config is not None:
        command.extend(["--config-locations", str(selected_proxy_config)])
    command.append(str(url))
    return command


def run_url(
    command: list[str], popen: Callable[..., object] = subprocess.Popen
) -> AttemptResult:
    """Run one URL while streaming output and classifying rate limiting."""
    child_command = command.copy()
    descriptor = -1
    try:
        if "--config-locations" in child_command:
            config_index = child_command.index("--config-locations") + 1
            descriptor = _open_proxy_config(Path(child_command[config_index]))
            child_command[config_index] = f"/proc/self/fd/{descriptor}"
        environment = dict(os.environ)
        for variable in PROXY_ENV_VARIABLES:
            environment.pop(variable, None)
        child = popen(
            child_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            pass_fds=(descriptor,) if descriptor >= 0 else (),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    rate_limited = False
    proxy_failure = False
    for line in child.stdout:
        print(line, end="", flush=True)
        normalized_line = line.replace("’", "'")
        rate_limited = rate_limited or any(
            marker in normalized_line for marker in RATE_LIMIT_MARKERS
        )
        proxy_failure = proxy_failure or (
            "502" in normalized_line and "proxy" in normalized_line.lower()
        )
    return_code = child.wait()
    return AttemptResult(
        completed=return_code == 0 and not rate_limited,
        rate_limited=rate_limited,
        proxy_failure=proxy_failure,
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
    maximum_consecutive_proxy_failures: int = 2,
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
        consecutive_proxy_failures = 0
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
            if result.proxy_failure:
                consecutive_proxy_failures += 1
                if consecutive_proxy_failures >= maximum_consecutive_proxy_failures:
                    print("Repeated proxy transport failures; stopping this pass.", flush=True)
                    return
            else:
                consecutive_proxy_failures = 0
            sleeper(ordinary_delay)
            if video_id in archived_ids:
                cooldown = initial_cooldown
        made_progress = bool(archived_ids - archived_at_pass_start)
        if not rate_limited and not made_progress:
            return


def run_spec(
    spec: DatasetSpec,
    *,
    url_runner: Callable[[list[str]], AttemptResult] = run_url,
    sleeper: Callable[[float], None] = time.sleep,
    config: DownloadConfig | None = None,
) -> None:
    """Run the prepared manifest and state paths declared by a dataset spec."""
    run_dataset(
        spec.state_dir / "pending-available.urls.txt",
        spec.output_dir,
        spec.state_dir / "download.archive.txt",
        url_runner=url_runner,
        sleeper=sleeper,
        ordinary_delay=config.ordinary_delay if config else 10.0,
        initial_cooldown=config.initial_cooldown if config else 3600.0,
        maximum_cooldown=config.maximum_cooldown if config else 7200.0,
    )
