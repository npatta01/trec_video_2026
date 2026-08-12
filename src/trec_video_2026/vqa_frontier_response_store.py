"""Exact-response parsing and conflict-safe billable response storage."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from trec_video_2026.vqa_submission import canonical_sha256


SCHEMA_VERSION = 1
_SAFE_PATH_SEGMENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


class RawResponseConflictError(RuntimeError):
    """Raised after preserving a different paid response without overwriting."""


def parse_prompt_json_response(raw_response: bytes | str) -> object:
    """Parse a response, removing only one exact lowercase JSON fence."""

    if isinstance(raw_response, bytes):
        raw_response = raw_response.decode("utf-8")
    stripped = raw_response.strip()
    prefix = "```json\n"
    suffix = "\n```"
    if stripped.startswith(prefix) and stripped.endswith(suffix):
        candidate = stripped[len(prefix) : -len(suffix)]
        if "```" not in candidate:
            raw_response = candidate
    return json.loads(raw_response)


def preserve_raw_response(
    root: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    raw_response: bytes | str,
) -> Mapping[str, object]:
    """Install an exact response, preserving conflicting bodies separately."""

    return _preserve_raw_response(
        root,
        run_id=run_id,
        condition_id=condition_id,
        video_id=video_id,
        raw_response=raw_response,
        record_identity_key="condition_id",
        conflict_layout="nested",
        fallback_on_conflict_symlink=False,
        legacy_path=None,
    )


def verify_raw_response(
    path: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    response_sha256: str,
) -> None:
    """Verify the stored response identity and its transport-byte hash."""

    _verify_raw_response(
        path,
        run_id=run_id,
        condition_id=condition_id,
        video_id=video_id,
        response_sha256=response_sha256,
        record_identity_key="condition_id",
    )


def _preserve_legacy_raw_response(
    root: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    raw_response: bytes | str,
) -> Mapping[str, object]:
    """Preserve the old runner's provider-shaped records and conflict paths."""

    return _preserve_raw_response(
        root,
        run_id=run_id,
        condition_id=condition_id,
        video_id=video_id,
        raw_response=raw_response,
        record_identity_key="provider",
        conflict_layout="legacy",
        fallback_on_conflict_symlink=True,
        legacy_path=root,
    )


def _verify_legacy_raw_response(
    path: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    response_sha256: str,
) -> None:
    _verify_raw_response(
        path,
        run_id=run_id,
        condition_id=condition_id,
        video_id=video_id,
        response_sha256=response_sha256,
        record_identity_key="provider",
    )


def _preserve_raw_response(
    root: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    raw_response: bytes | str,
    record_identity_key: Literal["condition_id", "provider"],
    conflict_layout: Literal["nested", "legacy"],
    fallback_on_conflict_symlink: bool,
    legacy_path: Path | None,
) -> Mapping[str, object]:
    _validate_path_segment(condition_id, "condition_id")
    _validate_path_segment(video_id, "video_id")
    raw_text = _raw_text(raw_response)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        record_identity_key: condition_id,
        "video_id": video_id,
        "raw_response_sha256": canonical_sha256(raw_text),
        "raw_response": raw_text,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    root_fd = _open_root(root)
    raw_fd: int | None = None
    condition_fd: int | None = None
    try:
        try:
            raw_fd = _open_or_create_nofollow_directory(root_fd, "raw")
            condition_fd = _open_or_create_nofollow_directory(raw_fd, condition_id)
        except OSError as error:
            raise _unsafe_path_error("raw response path", error) from error

        target_name = f"{video_id}.json"
        existing = _read_json_at(condition_fd, target_name)
        if existing == payload:
            return payload
        if existing is not None or _path_exists_at(condition_fd, target_name):
            preserved = _preserve_conflict(
                root_fd,
                condition_id=condition_id,
                video_id=video_id,
                payload=payload,
                rendered=rendered,
                layout=conflict_layout,
                fallback_on_symlink=fallback_on_conflict_symlink,
                root_path=legacy_path,
            )
            raise RawResponseConflictError(
                f"raw response path conflicts with the paid response; current response preserved at {preserved}"
            )

        try:
            _install_json_exclusive(
                condition_fd,
                target_name,
                rendered,
                directory_path=(
                    legacy_path / "raw" / condition_id
                    if legacy_path is not None
                    else None
                ),
            )
        except FileExistsError:
            existing = _read_json_at(condition_fd, target_name)
            if existing == payload:
                return payload
            preserved = _preserve_conflict(
                root_fd,
                condition_id=condition_id,
                video_id=video_id,
                payload=payload,
                rendered=rendered,
                layout=conflict_layout,
                fallback_on_symlink=fallback_on_conflict_symlink,
                root_path=legacy_path,
            )
            raise RawResponseConflictError(
                f"raw response path conflicts with the paid response; current response preserved at {preserved}"
            ) from None

        installed = _read_json_at(condition_fd, target_name)
        if installed != payload:
            preserved = _preserve_conflict(
                root_fd,
                condition_id=condition_id,
                video_id=video_id,
                payload=payload,
                rendered=rendered,
                layout=conflict_layout,
                fallback_on_symlink=fallback_on_conflict_symlink,
                root_path=legacy_path,
            )
            raise RawResponseConflictError(
                f"raw response path conflicts with the paid response; current response preserved at {preserved}"
            )
        return payload
    finally:
        if condition_fd is not None:
            os.close(condition_fd)
        if raw_fd is not None:
            os.close(raw_fd)
        os.close(root_fd)


def _verify_raw_response(
    path: Path,
    *,
    run_id: str,
    condition_id: str,
    video_id: str,
    response_sha256: str,
    record_identity_key: Literal["condition_id", "provider"],
) -> None:
    if path.is_symlink():
        raise ValueError("billable raw response path must not be a symlink")
    try:
        record = _read_json_path(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("billable preflight raw response is invalid") from error
    raw_response = record.get("raw_response")
    if not isinstance(raw_response, str):
        raise ValueError("billable preflight raw response is invalid")
    expected = {
        "run_id": run_id,
        record_identity_key: condition_id,
        "video_id": video_id,
        "raw_response_sha256": canonical_sha256(raw_response),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("billable preflight raw response identity changed")
    if hashlib.sha256(raw_response.encode("utf-8")).hexdigest() != response_sha256:
        raise ValueError("billable preflight response hash changed")


def _raw_text(raw_response: bytes | str) -> str:
    if isinstance(raw_response, bytes):
        return raw_response.decode("utf-8", errors="replace")
    return raw_response


def _validate_path_segment(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(character not in _SAFE_PATH_SEGMENT for character in value)
    ):
        raise ValueError(f"{name} must be a safe path segment")


def _open_root(root: Path) -> int:
    try:
        return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise _unsafe_path_error("response root", error) from error


def _open_or_create_nofollow_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    return os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )


def _install_json_exclusive(
    directory_fd: int,
    name: str,
    rendered: str,
    *,
    directory_path: Path | None = None,
) -> None:
    temporary_name = f".raw-response-{secrets.token_hex(8)}.tmp"
    fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if directory_path is None:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        else:
            os.link(directory_path / temporary_name, directory_path / name)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _preserve_conflict(
    root_fd: int,
    *,
    condition_id: str,
    video_id: str,
    payload: Mapping[str, object],
    rendered: str,
    layout: Literal["nested", "legacy"],
    fallback_on_symlink: bool,
    root_path: Path | None,
) -> Path:
    response = payload.get("raw_response")
    if not isinstance(response, str):
        raise TypeError("raw response payload is invalid")
    response_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    conflict_root_fd: int | None = None
    conflict_condition_fd: int | None = None
    conflict_video_fd: int | None = None
    directory_fd = root_fd
    directory_path = Path(".")
    filesystem_directory_path: Path | None = None
    try:
        try:
            conflict_root_fd = _open_or_create_nofollow_directory(
                root_fd, "raw-conflicts"
            )
            conflict_condition_fd = _open_or_create_nofollow_directory(
                conflict_root_fd, condition_id
            )
            if layout == "nested":
                conflict_video_fd = _open_or_create_nofollow_directory(
                    conflict_condition_fd, video_id
                )
                directory_fd = conflict_video_fd
                directory_path = Path("raw-conflicts") / condition_id / video_id
            else:
                directory_fd = conflict_condition_fd
                directory_path = Path("raw-conflicts") / condition_id
            if root_path is not None:
                filesystem_directory_path = root_path / directory_path
        except OSError as error:
            if not fallback_on_symlink:
                raise _unsafe_path_error("raw conflict path", error) from error
            directory_fd = root_fd
            directory_path = Path(".")
            filesystem_directory_path = root_path

        if layout == "nested":
            prefix = f"{response_sha256}-"
        else:
            prefix = f"{video_id}-{response_sha256}-"
        if directory_path == Path("."):
            prefix = f".raw-conflict-{condition_id}-{video_id}-{response_sha256}-"
        for _ in range(32):
            name = f"{prefix}{secrets.token_hex(8)}.json"
            try:
                _install_json_exclusive(
                    directory_fd,
                    name,
                    rendered,
                    directory_path=filesystem_directory_path,
                )
            except FileExistsError:
                continue
            if _read_json_at(directory_fd, name) != payload:
                raise RawResponseConflictError(
                    "conflicting paid response could not be preserved"
                )
            return directory_path / name
        raise RawResponseConflictError("conflicting paid response could not be preserved")
    finally:
        if conflict_video_fd is not None:
            os.close(conflict_video_fd)
        if conflict_condition_fd is not None:
            os.close(conflict_condition_fd)
        if conflict_root_fd is not None:
            os.close(conflict_root_fd)


def _path_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _unsafe_path_error("raw response file", error) from error
        raise
    return True


def _read_json_at(directory_fd: int, name: str) -> Mapping[str, object] | None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _unsafe_path_error("raw response file", error) from error
        raise
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _read_json_path(path: Path) -> Mapping[str, object]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("JSON artifact must be an object")
    return payload


def _unsafe_path_error(label: str, error: OSError) -> ValueError:
    return ValueError(f"{label} contains an unsafe or symlinked path")


__all__ = [
    "RawResponseConflictError",
    "parse_prompt_json_response",
    "preserve_raw_response",
    "verify_raw_response",
]
