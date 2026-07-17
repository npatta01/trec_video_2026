"""Reusable YouTube dataset download engine."""

from __future__ import annotations

import argparse
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
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

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
GENERIC_PROXY_ENV_VARIABLES = (
    "PROXY_URL",
    "PROXY_SCHEME",
    "PROXY_ENDPOINT",
    "PROXY_USER",
    "PROXY_PASSWORD",
)
ALL_PROXY_ENV_VARIABLES = GENERIC_PROXY_ENV_VARIABLES + PROXY_ENV_VARIABLES
WORKER_LOCK_FD_ENV = "YOUTUBE_DATASET_WORKER_LOCK_FD"
PROCESS_STOP_TIMEOUT = 5.0


@dataclass(frozen=True)
class DatasetSpec:
    """Explicit paths describing one generic YouTube dataset."""

    source: Path
    output_dir: Path
    state_dir: Path
    archive_path: Path | None = None
    prepared_manifest_path: Path | None = None
    availability_snapshot_path: Path | None = None
    api_omitted_path: Path | None = None


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
    failure_category: str | None = None
    retryable: bool = False


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
    if (
        not host
        or len(host) > 253
        or all(character in ".0123456789" for character in host)
    ):
        return False
    label_pattern = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
    return all(label_pattern.fullmatch(label) for label in host.split("."))


def load_proxy_settings(
    env_path: Path, runner: Callable[..., object] = subprocess.run
) -> tuple[str, str, str, str]:
    """Load one validated generic or legacy proxy form without secret exposure."""
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
            "unset PROXY_URL PROXY_SCHEME PROXY_ENDPOINT PROXY_USER PROXY_PASSWORD; "
            'set -a; source "$1" >/dev/null 2>&1 || exit 1; '
            'printf "%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0%s\\0" '
            '"${PROXY_SCHEME:-}" "${PROXY_ENDPOINT:-}" "${PROXY_USER:-}" '
            '"${PROXY_PASSWORD:-}" "${PROXY_URL:-}" "${DECODO_ENDPOINT:-}" '
            '"${DECODO_USER:-}" "${DECODO_PASSWORD:-}"'
        )
        try:
            environment = dict(os.environ)
            for variable in ALL_PROXY_ENV_VARIABLES:
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
            if fields[-1] != b"":
                raise ValueError
            if len(fields) == 4:  # compatibility for injected legacy runners
                endpoint, username, password = (field.decode() for field in fields[:3])
                scheme = "http"
            elif len(fields) in (8, 9):
                if len(fields) == 8:  # compatibility for injected test runners
                    fields.insert(-1, b"")
                values = [field.decode() for field in fields[:8]]
                (
                    scheme,
                    endpoint,
                    username,
                    password,
                    proxy_url,
                    legacy_endpoint,
                    legacy_user,
                    legacy_password,
                ) = values
                forms = sum(
                    (
                        bool(proxy_url),
                        any((scheme, endpoint, username, password)),
                        any((legacy_endpoint, legacy_user, legacy_password)),
                    )
                )
                if forms != 1:
                    raise ValueError
                if proxy_url:
                    parsed = urlsplit(proxy_url)
                    if (
                        parsed.scheme not in ("http", "https")
                        or not parsed.hostname
                        or parsed.port is None
                        or parsed.username is None
                        or parsed.password is None
                        or parsed.path not in ("", "/")
                        or parsed.query
                        or parsed.fragment
                    ):
                        raise ValueError
                    scheme = parsed.scheme
                    endpoint = f"{parsed.hostname}:{parsed.port}"
                    username, password = (
                        unquote(parsed.username),
                        unquote(parsed.password),
                    )
                elif legacy_endpoint or legacy_user or legacy_password:
                    scheme, endpoint, username, password = (
                        "http",
                        legacy_endpoint,
                        legacy_user,
                        legacy_password,
                    )
                elif not all((scheme, endpoint, username, password)):
                    raise ValueError
            else:
                raise ValueError
            endpoint_match = re.fullmatch(r"([^:/@\s]+):(\d{1,5})", endpoint)
            if (
                scheme not in ("http", "https")
                or not endpoint_match
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
    return scheme, endpoint, username, password


def load_proxy_credentials(
    env_path: Path, runner: Callable[..., object] = subprocess.run
) -> tuple[str, str, str]:
    """Compatibility view of securely loaded proxy credentials."""
    _, endpoint, username, password = load_proxy_settings(env_path, runner=runner)
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
            raise RuntimeError("YouTube API key file must be owned by the current user")
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
    except (
        OSError,
        URLError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
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
        if VIDEO_ID.fullmatch(value):
            video_id = value
        else:
            parsed = urlsplit(value)
            host = (parsed.hostname or "").lower()
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid YouTube ID or URL in {source}")
            if host in ("youtube.com", "www.youtube.com", "m.youtube.com"):
                values = parse_qs(parsed.query)
                if parsed.path != "/watch" or len(values.get("v", [])) != 1:
                    raise ValueError(f"Invalid YouTube ID or URL in {source}")
                video_id = values["v"][0]
            elif host in ("youtu.be", "www.youtu.be"):
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) != 1:
                    raise ValueError(f"Invalid YouTube ID or URL in {source}")
                video_id = parts[0]
            else:
                raise ValueError(f"Invalid YouTube ID or URL in {source}")
        if not VIDEO_ID.fullmatch(video_id):
            raise ValueError(f"Invalid YouTube ID or URL in {source}")
        ids.append(video_id)
    return list(dict.fromkeys(ids))


def prepare_dataset(
    spec: DatasetSpec,
    api_key: str | DownloadConfig,
    fetcher: Callable[[list[str], str], set[str]] = fetch_available_video_ids,
    pending_writes: list[tuple[Path, str]] | None = None,
) -> dict[str, int]:
    """Atomically prepare API-visible and omitted snapshots for one dataset."""
    if isinstance(api_key, DownloadConfig):
        api_key = load_youtube_api_key(api_key.api_key_file)
    ids = read_source_ids(spec.source)
    archive_path = spec.archive_path or spec.state_dir / "download.archive.txt"
    archived = read_archive_ids(archive_path)
    pending = [video_id for video_id in ids if video_id not in archived]
    available = fetcher(pending, api_key)
    visible = [video_id for video_id in pending if video_id in available]
    omitted = [video_id for video_id in pending if video_id not in available]
    summary = {
        "total": len(pending),
        "pending_available": len(visible),
        "api_omitted": len(omitted),
    }
    snapshot_path = (
        spec.availability_snapshot_path or spec.state_dir / "availability-snapshot.json"
    )
    manifest_path = (
        spec.prepared_manifest_path or spec.state_dir / "pending-available.urls.txt"
    )
    omitted_path = spec.api_omitted_path or spec.state_dir / "api-omitted.txt"
    writes = [
        (
            snapshot_path,
            json.dumps(
                {
                    "version": 1,
                    "visible": visible,
                    "omitted": omitted,
                    "counts": summary,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
        ),
        (
            manifest_path,
            "".join(
                f"https://www.youtube.com/watch?v={video_id}\n" for video_id in visible
            ),
        ),
        (omitted_path, "".join(f"{video_id}\n" for video_id in omitted)),
    ]
    if pending_writes is None:
        commit_prepared_datasets(writes)
    else:
        pending_writes.extend(writes)
    return summary


def commit_prepared_datasets(writes: list[tuple[Path, str]]) -> None:
    """Commit engine-generated projections after all API checks succeed."""
    for path, contents in writes:
        _atomic_write_text(path, contents)


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


def _atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace a machine-readable state document."""
    _atomic_write_text(path, json.dumps(value, sort_keys=True, indent=2) + "\n")


def _read_availability_snapshot(path: Path) -> dict[str, object]:
    """Read and strictly validate the authoritative API availability snapshot."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        visible = value["visible"]
        omitted = value["omitted"]
        counts = value["counts"]
        if (
            value.get("version") != 1
            or not isinstance(visible, list)
            or not isinstance(omitted, list)
            or any(
                not isinstance(item, str) or not VIDEO_ID.fullmatch(item)
                for item in visible + omitted
            )
            or len(set(visible + omitted)) != len(visible + omitted)
            or counts
            != {
                "total": len(visible) + len(omitted),
                "pending_available": len(visible),
                "api_omitted": len(omitted),
            }
        ):
            raise ValueError
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RuntimeError("Authoritative availability snapshot is invalid") from None
    return value


def _runtime_state(path: Path) -> dict[str, object]:
    default = {
        "worker_pid": None,
        "worker_identity": None,
        "current_id": None,
        "last_progress_at": None,
        "cooldown_deadline": None,
        "failures": {},
        "retryable_failures": [],
    }
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError
    except (OSError, ValueError, json.JSONDecodeError):
        raise RuntimeError("Runtime state is invalid") from None
    return default | loaded


def _update_runtime_state(path: Path, **changes: object) -> dict[str, object]:
    state = _runtime_state(path)
    state.update(changes)
    _atomic_write_json(path, state)
    return state


def _record_failure(path: Path, video_id: str, category: str, retryable: bool) -> None:
    state = _runtime_state(path)
    failures = {key: list(value) for key, value in dict(state["failures"]).items()}
    for ids in failures.values():
        while video_id in ids:
            ids.remove(video_id)
    failures.setdefault(category, [])
    if video_id not in failures[category]:
        failures[category].append(video_id)
    retries = [item for item in state["retryable_failures"] if item != video_id]
    if retryable and video_id not in retries:
        retries.append(video_id)
    state.update(failures=failures, retryable_failures=retries)
    _atomic_write_json(path, state)


def _clear_failure(path: Path, video_id: str) -> None:
    """Remove an ID from every failure inventory after archived success."""
    state = _runtime_state(path)
    failures = {
        key: [item for item in value if item != video_id]
        for key, value in dict(state["failures"]).items()
    }
    retries = [item for item in state["retryable_failures"] if item != video_id]
    state.update(failures=failures, retryable_failures=retries)
    _atomic_write_json(path, state)


def write_proxy_config(
    env_path: Path,
    config_path: Path,
    runner: Callable[..., object] = subprocess.run,
) -> Path:
    """Atomically create a private yt-dlp proxy config."""
    scheme, endpoint, username, password = load_proxy_settings(env_path, runner=runner)
    proxy_url = (
        f"{scheme}://{quote(username, safe='')}:{quote(password, safe='')}@{endpoint}"
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
            raise RuntimeError("YouTube dataset startup already in progress") from None
        handle = _StartupLockHandle(descriptor)
        yield handle
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("Unable to secure YouTube dataset startup lock") from None
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
        descriptor = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise OSError
    except OSError:
        raise RuntimeError("YouTube dataset runtime directory is not secure") from None
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
        raise RuntimeError(
            "Inherited YouTube dataset worker lock is not secure"
        ) from None


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


def dataset_status(spec: DatasetSpec) -> dict[str, object]:
    """Return generic state counts using only paths declared by the spec."""
    archive = spec.state_dir / "download.archive.txt"
    snapshot_path = (
        spec.availability_snapshot_path or spec.state_dir / "availability-snapshot.json"
    )
    counts = _read_availability_snapshot(snapshot_path)["counts"]
    pending_count = counts["pending_available"]
    omitted_count = counts["api_omitted"]
    worker = worker_status(spec.state_dir)
    runtime = worker["runtime"]
    running = worker["running"]
    pid = worker["pid"]
    return {
        "running": running,
        "pid": pid,
        "current_id": runtime["current_id"],
        "completed": count_lines(archive) if archive.exists() else 0,
        "media_files": count_final_media(spec.output_dir),
        "pending_available": pending_count,
        "api_omitted": omitted_count,
        "last_progress_at": runtime["last_progress_at"],
        "cooldown_deadline": runtime["cooldown_deadline"],
        "failure_counts": {
            key: len(value) for key, value in dict(runtime["failures"]).items()
        },
        "retryable_failures": len(runtime["retryable_failures"]),
    }


def worker_status(state_dir: Path) -> dict[str, object]:
    """Return persisted worker details reconciled with validated live PIDs."""
    runtime = _runtime_state(state_dir / "runtime-state.json")
    pid_path = state_dir / "download.pid"
    identity = runtime.get("worker_identity")
    running = False
    pid = None
    if isinstance(identity, dict) and isinstance(identity.get("pid"), int):
        current = _process_identity(identity["pid"])
        running = current == identity
        if running:
            pid = str(identity["pid"])
    elif identity is None:
        if pid_is_active(pid_path):
            try:
                pid = str(int(pid_path.read_text(encoding="utf-8").strip()))
                running = True
            except (FileNotFoundError, ValueError):
                pass
        runtime_pid = runtime.get("worker_pid")
        if (
            not running
            and isinstance(runtime_pid, int)
            and _process_active(runtime_pid)
        ):
            running = True
            pid = str(runtime_pid)
    if not running and (
        identity is not None
        or runtime.get("worker_pid") is not None
        or pid_path.exists()
    ):
        pid_path.unlink(missing_ok=True)
        runtime = _update_runtime_state(
            state_dir / "runtime-state.json",
            worker_pid=None,
            worker_identity=None,
            current_id=None,
        )
    return {"running": running, "pid": pid, "runtime": runtime}


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


def _process_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_identity(pid: int) -> dict[str, object] | None:
    """Return Linux PID/start-time identity, immune to PID reuse."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, separator, remainder = text.rpartition(")")
        fields = remainder.strip().split()
        if not separator or len(fields) < 20:
            raise ValueError
        return {"pid": pid, "start_time": fields[19]}
    except (OSError, ValueError):
        return None


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
    active = worker_status(runtime)
    if active["running"]:
        raise RuntimeError(
            f"YouTube dataset download is already running with PID {active['pid']}"
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
    for variable in ALL_PROXY_ENV_VARIABLES:
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
            raise RuntimeError(
                "Unable to start detached YouTube dataset worker"
            ) from None
    try:
        pid = int(child.pid)
        _record_pid(pid_path, pid)
        _update_runtime_state(
            runtime / "runtime-state.json",
            worker_pid=pid,
            worker_identity=_process_identity(pid),
        )
    except Exception:
        _rollback_child(child)
        raise RuntimeError(
            "Unable to record detached YouTube dataset worker PID"
        ) from None
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
    node_path = (
        node_path or os.environ.get("YOUTUBE_DATASET_NODE_PATH") or shutil.which("node")
    )
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
        Path(value)
        if (value := os.environ.get("YOUTUBE_DATASET_PROXY_CONFIG_PATH"))
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
    child_command = command.copy()
    descriptor = -1
    try:
        if "--config-locations" in child_command:
            config_index = child_command.index("--config-locations") + 1
            descriptor = _open_proxy_config(Path(child_command[config_index]))
            child_command[config_index] = f"/proc/self/fd/{descriptor}"
        environment = dict(os.environ)
        for variable in ALL_PROXY_ENV_VARIABLES:
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
    failure_category = None
    for line in child.stdout:
        print(line, end="", flush=True)
        normalized_line = line.replace("’", "'")
        rate_limited = (
            rate_limited
            or any(marker in normalized_line for marker in RATE_LIMIT_MARKERS)
            or bool(
                re.search(
                    r"(?:HTTP Error 429\b|\bToo Many Requests\b)",
                    normalized_line,
                    re.I,
                )
            )
        )
        proxy_failure = proxy_failure or (
            "502" in normalized_line and "proxy" in normalized_line.lower()
        )
        lowered = normalized_line.lower()
        if "members-only" in lowered or "members only" in lowered:
            failure_category = "members_only"
        elif "private video" in lowered or "video is private" in lowered:
            failure_category = "private"
        elif "age-restricted" in lowered or "age restricted" in lowered:
            failure_category = "authentication_required"
        elif "sign in" in lowered and (
            "confirm your age" in lowered or "authentication" in lowered
        ):
            failure_category = "authentication_required"
        elif (
            "video unavailable" in lowered
            or "video is unavailable" in lowered
            or "not available" in lowered
            or "isn't available" in lowered
        ):
            failure_category = "unavailable"
    return_code = child.wait()
    return AttemptResult(
        completed=return_code == 0 and not rate_limited,
        rate_limited=rate_limited,
        proxy_failure=proxy_failure,
        failure_category=(
            "rate_limit"
            if rate_limited
            else "proxy_transport"
            if proxy_failure
            else failure_category or ("retryable" if return_code else None)
        ),
        retryable=(
            rate_limited
            or proxy_failure
            or (return_code != 0 and failure_category is None)
        ),
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
    urls = [url for url in manifest.read_text(encoding="utf-8").splitlines() if url]
    cooldown = initial_cooldown
    state_path = archive.parent / "runtime-state.json"
    while True:
        archived_ids = read_archive_ids(archive)
        archived_at_pass_start = archived_ids.copy()
        rate_limited = False
        consecutive_proxy_failures = 0
        for url in urls:
            video_id = url.rsplit("v=", 1)[-1].split("&", 1)[0]
            if video_id in archived_ids:
                continue
            _update_runtime_state(state_path, current_id=video_id)
            result = url_runner(build_ytdlp_command(url, output_dir, archive))
            archived_ids = read_archive_ids(archive)
            if video_id in archived_ids:
                _clear_failure(state_path, video_id)
                _update_runtime_state(
                    state_path,
                    last_progress_at=datetime.now(timezone.utc).isoformat(),
                )
            elif result.failure_category:
                _record_failure(
                    state_path, video_id, result.failure_category, result.retryable
                )
            if result.rate_limited:
                retry_utc = datetime.now(timezone.utc) + timedelta(seconds=cooldown)
                retry_local = retry_utc.astimezone()
                print(
                    f"Rate limited; cooling down for {cooldown:g} seconds. "
                    f"Retry at {retry_utc.isoformat()} UTC / "
                    f"{retry_local.isoformat()} local.",
                    flush=True,
                )
                _update_runtime_state(
                    state_path, cooldown_deadline=retry_utc.isoformat()
                )
                sleeper(cooldown)
                _update_runtime_state(state_path, cooldown_deadline=None)
                cooldown = min(cooldown * 2, maximum_cooldown)
                rate_limited = True
                break
            if result.proxy_failure:
                consecutive_proxy_failures += 1
                if consecutive_proxy_failures >= maximum_consecutive_proxy_failures:
                    print(
                        "Repeated proxy transport failures; stopping this pass.",
                        flush=True,
                    )
                    _update_runtime_state(state_path, current_id=None)
                    return
            else:
                consecutive_proxy_failures = 0
            sleeper(ordinary_delay)
            if video_id in archived_ids:
                cooldown = initial_cooldown
        made_progress = bool(archived_ids - archived_at_pass_start)
        if not rate_limited and not made_progress:
            _update_runtime_state(state_path, current_id=None, cooldown_deadline=None)
            return


def run_spec(
    spec: DatasetSpec,
    *,
    url_runner: Callable[[list[str]], AttemptResult] = run_url,
    sleeper: Callable[[float], None] = time.sleep,
    config: DownloadConfig | None = None,
) -> None:
    """Run the prepared manifest and state paths declared by a dataset spec."""
    manifest = spec.prepared_manifest_path
    snapshot_path = spec.availability_snapshot_path or (
        spec.state_dir / "availability-snapshot.json"
        if spec.prepared_manifest_path is None
        else None
    )
    if spec.availability_snapshot_path is not None:
        snapshot = _read_availability_snapshot(spec.availability_snapshot_path)
        manifest = manifest or spec.state_dir / "pending-available.urls.txt"
        _atomic_write_text(
            manifest,
            "".join(
                f"https://www.youtube.com/watch?v={video_id}\n"
                for video_id in snapshot["visible"]
            ),
        )
    elif snapshot_path is not None and snapshot_path.exists():
        snapshot = _read_availability_snapshot(snapshot_path)
        manifest = manifest or spec.state_dir / "pending-available.urls.txt"
        _atomic_write_text(
            manifest,
            "".join(
                f"https://www.youtube.com/watch?v={video_id}\n"
                for video_id in snapshot["visible"]
            ),
        )
    elif manifest is None:
        raise RuntimeError("Authoritative availability snapshot is required")
    pid = os.getpid()
    _update_runtime_state(
        spec.state_dir / "runtime-state.json",
        worker_pid=pid,
        worker_identity=_process_identity(pid),
    )
    try:
        run_dataset(
            manifest,
            spec.output_dir,
            spec.archive_path or spec.state_dir / "download.archive.txt",
            url_runner=url_runner,
            sleeper=sleeper,
            ordinary_delay=config.ordinary_delay if config else 10.0,
            initial_cooldown=config.initial_cooldown if config else 3600.0,
            maximum_cooldown=config.maximum_cooldown if config else 7200.0,
        )
    finally:
        _update_runtime_state(
            spec.state_dir / "runtime-state.json",
            worker_pid=None,
            worker_identity=None,
            current_id=None,
        )


def _git_tracked(path: Path) -> bool:
    """Return whether Git tracks the path itself or anything below it."""
    resolved = path.resolve(strict=False)
    anchor = resolved.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    probe = subprocess.run(
        ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        return False
    root = Path(probe.stdout.strip())
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return False
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", str(relative)],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(tracked.stdout.strip())


def refuse_git_tracked_runtime_paths(spec: DatasetSpec, allow: bool = False) -> None:
    """Refuse writes into Git-tracked output/state paths without an override."""
    if allow:
        return
    tracked = [path for path in (spec.output_dir, spec.state_dir) if _git_tracked(path)]
    if tracked:
        raise RuntimeError(
            "Output or state path is tracked by Git; use --allow-git-tracked to override"
        )


def preflight_proxy(
    env_path: Path,
    opener_factory: Callable[..., object] = build_opener,
    credential_runner: Callable[..., object] = subprocess.run,
) -> None:
    """Fail closed unless the configured proxy reaches neutral HTTPS and YouTube."""
    scheme, endpoint, username, password = load_proxy_settings(
        env_path, runner=credential_runner
    )
    proxy = (
        f"{scheme}://{quote(username, safe='')}:{quote(password, safe='')}@{endpoint}"
    )
    try:
        opener = opener_factory(ProxyHandler({"https": proxy}))
        for url in ("https://www.google.com/generate_204", "https://www.youtube.com/"):
            with opener.open(Request(url), timeout=30) as response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 400:
                    raise OSError
    except Exception:
        raise RuntimeError("Proxy connectivity preflight failed") from None


def _generic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "start", "status"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--api-env", required=True, type=Path)
    parser.add_argument("--proxy-env", required=True, type=Path)
    parser.add_argument("--allow-git-tracked", action="store_true")
    parser.add_argument(
        "--detached-worker", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the explicit-path generic YouTube dataset CLI."""
    args = _generic_parser().parse_args(argv)
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    state_dir = args.state_dir.resolve()
    api_env = args.api_env.resolve()
    proxy_env = args.proxy_env.resolve()
    spec = DatasetSpec(source, output_dir, state_dir)
    worker_command = (
        sys.executable,
        "-m",
        "trec_video_2026.youtube_dataset",
        "run",
        "--source",
        str(spec.source),
        "--output-dir",
        str(spec.output_dir),
        "--state-dir",
        str(spec.state_dir),
        "--api-env",
        str(api_env),
        "--proxy-env",
        str(proxy_env),
        "--detached-worker",
    )
    if args.allow_git_tracked:
        worker_command += ("--allow-git-tracked",)
    config = DownloadConfig(api_env, proxy_env, worker_command=worker_command)
    try:
        refuse_git_tracked_runtime_paths(spec, args.allow_git_tracked)
        if args.command == "prepare":
            summary = prepare_dataset(spec, config)
            print(f"total: {summary['total']}")
            print(f"pending available: {summary['pending_available']}")
            print(f"API omitted: {summary['api_omitted']}")
            return 0
        if args.command == "status":
            status = dataset_status(spec)
            print(f"state: {'running' if status['running'] else 'stopped'}")
            print(f"pid: {status.get('pid') or 'none'}")
            print(f"current ID: {status.get('current_id') or 'none'}")
            for name in (
                "completed",
                "media_files",
                "pending_available",
                "api_omitted",
            ):
                print(f"{name.replace('_', ' ')}: {status[name]}")
            print(f"last progress: {status.get('last_progress_at') or 'none'}")
            print(f"cooldown deadline: {status.get('cooldown_deadline') or 'none'}")
            failure_counts = status.get("failure_counts", {})
            for category in (
                "private",
                "unavailable",
                "members_only",
                "authentication_required",
                "proxy_transport",
                "rate_limit",
                "retryable",
            ):
                print(
                    f"failures {category.replace('_', '-')}: {failure_counts.get(category, 0)}"
                )
            print(f"retryable failures: {status.get('retryable_failures', 0)}")
            print(f"log: {spec.state_dir / 'download.log'}")
            return 0
        spec.state_dir.mkdir(parents=True, exist_ok=True)
        _validate_runtime_directory(spec.state_dir)
        if args.command == "start":
            summary = prepare_dataset(spec, config)
            preflight_proxy(config.proxy_file)
            pid = start_detached(spec, config)
            print(f"pending available: {summary['pending_available']}")
            print(f"started YouTube dataset download with PID {pid}")
            print(f"log: {spec.state_dir / 'download.log'}")
            return 0
        if args.detached_worker:
            value = os.environ.get(WORKER_LOCK_FD_ENV)
            if not value:
                raise RuntimeError("Inherited YouTube dataset worker lock is required")
            try:
                descriptor = int(value)
            except ValueError:
                raise RuntimeError(
                    "Inherited YouTube dataset worker lock is invalid"
                ) from None
            _validate_inherited_worker_lock(spec.state_dir, descriptor)
            config_path = os.environ.get("YOUTUBE_DATASET_PROXY_CONFIG_PATH")
            if not config_path:
                raise RuntimeError("Prepared proxy configuration is required")
            validate_proxy_config(Path(config_path))
            try:
                run_spec(spec, config=config)
            finally:
                os.close(descriptor)
            return 0
        with _startup_lock(spec.state_dir):
            prepare_dataset(spec, config)
            preflight_proxy(config.proxy_file)
            proxy_config = write_proxy_config(
                config.proxy_file, spec.state_dir / "yt-dlp-proxy.conf"
            )
            os.environ["YOUTUBE_DATASET_PROXY_CONFIG_PATH"] = str(proxy_config)
            run_spec(spec, config=config)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
