"""Auditable, lossless delivery media for frontier video-map providers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DELIVERY_POLICY_VERSION = "frontier-map-delivery-v2-webm-copy"
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".webm": "video/webm",
}
_GEMINI_PASSTHROUGH = frozenset({".mov", ".mp4", ".mpeg", ".mpg", ".webm"})


class DeliveryMediaError(ValueError):
    """Raised when provider delivery media cannot be prepared or verified."""


@dataclass(frozen=True, slots=True)
class DeliveryMedia:
    path: Path
    provenance_path: Path
    mime_type: str
    sha256: str
    size_bytes: int
    source_sha256: str
    source_size_bytes: int
    policy_version: str
    derived: bool

    def request_identity(self) -> dict[str, object]:
        """Return only the immutable fields that bind a provider request."""

        return {
            "delivery_policy_version": self.policy_version,
            "delivery_sha256": self.sha256,
            "delivery_size_bytes": self.size_bytes,
            "delivery_mime_type": self.mime_type,
            "delivery_derived": self.derived,
        }


def prepare_delivery_media(
    *,
    run_path: Path,
    source_path: Path,
    video_id: str,
    provider: str,
) -> DeliveryMedia:
    """Prepare or verify one provider-compatible, evidence-preserving input."""

    run_path = Path(run_path)
    if run_path.is_symlink():
        raise DeliveryMediaError("delivery run directory must not be a symlink")
    run_path.mkdir(parents=True, exist_ok=True)
    run_path = run_path.resolve(strict=True)
    if _VIDEO_ID.fullmatch(video_id) is None:
        raise DeliveryMediaError("invalid delivery video ID")
    delivery_root = run_path / "delivery-media"
    if delivery_root.is_symlink():
        raise DeliveryMediaError("delivery media directory must not be a symlink")
    delivery_root.mkdir(mode=0o700, exist_ok=True)
    with _delivery_claim(delivery_root, provider=provider, video_id=video_id):
        return _prepare_delivery_media_claimed(
            run_path=run_path,
            source_path=source_path,
            video_id=video_id,
            provider=provider,
        )


def _prepare_delivery_media_claimed(
    *,
    run_path: Path,
    source_path: Path,
    video_id: str,
    provider: str,
) -> DeliveryMedia:
    """Prepare delivery media while holding its exclusive filesystem claim."""

    if provider not in {"gemini", "reka"}:
        raise DeliveryMediaError("unsupported delivery provider")
    if _VIDEO_ID.fullmatch(video_id) is None:
        raise DeliveryMediaError("invalid delivery video ID")
    run_path = Path(run_path)
    if run_path.is_symlink():
        raise DeliveryMediaError("delivery run directory must not be a symlink")
    run_path.mkdir(parents=True, exist_ok=True)
    run_path = run_path.resolve(strict=True)
    source_path = Path(source_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise DeliveryMediaError("delivery source must be a regular file")
    source_path = source_path.resolve(strict=True)
    suffix = source_path.suffix.lower()
    source_mime = _MIME_TYPES.get(suffix)
    if source_mime is None:
        raise DeliveryMediaError("unsupported source container")
    source_sha256 = _sha256_file(source_path)
    source_size_bytes = source_path.stat().st_size

    delivery_root = run_path / "delivery-media"
    if delivery_root.is_symlink():
        raise DeliveryMediaError("delivery media directory must not be a symlink")
    delivery_root.mkdir(mode=0o700, exist_ok=True)
    provenance_path = delivery_root / f"{provider}-{video_id}.json"

    should_remux = provider == "gemini" and suffix not in _GEMINI_PASSTHROUGH
    if should_remux and suffix != ".mkv":
        raise DeliveryMediaError("Gemini source container cannot be losslessly delivered")
    if not should_remux:
        payload = {
            "schema_version": 1,
            "policy_version": DELIVERY_POLICY_VERSION,
            "provider": provider,
            "video_id": video_id,
            "derived": False,
            "source_name": source_path.name,
            "source_sha256": source_sha256,
            "source_size_bytes": source_size_bytes,
            "delivery_name": source_path.name,
            "delivery_sha256": source_sha256,
            "delivery_size_bytes": source_size_bytes,
            "delivery_mime_type": source_mime,
        }
        _write_or_verify_provenance(provenance_path, payload)
        return verify_delivery_media(
            DeliveryMedia(
                path=source_path,
                provenance_path=provenance_path,
                mime_type=source_mime,
                sha256=source_sha256,
                size_bytes=source_size_bytes,
                source_sha256=source_sha256,
                source_size_bytes=source_size_bytes,
                policy_version=DELIVERY_POLICY_VERSION,
                derived=False,
            )
        )

    target = delivery_root / f"{video_id}.webm"
    if target.is_symlink() or provenance_path.is_symlink():
        raise DeliveryMediaError("delivery artifacts must not be symlinks")
    if target.exists() or provenance_path.exists():
        if not target.is_file() or not provenance_path.is_file():
            raise DeliveryMediaError("incomplete existing delivery artifacts")
        payload = _read_json(provenance_path)
        delivery = _delivery_from_payload(
            payload,
            path=target,
            provenance_path=provenance_path,
            source_path=source_path,
        )
        return verify_delivery_media(delivery)

    source_probe = _probe(source_path)
    source_streams, source_indexes = _stream_records(source_probe)
    source_packet_sha256 = _packet_sha256(source_path, source_indexes)
    source_duration = _duration(source_probe)
    with tempfile.TemporaryDirectory(prefix=f".{video_id}-", dir=delivery_root) as temp:
        temporary = Path(temp) / f"{video_id}.webm"
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-n",
                "-i",
                str(source_path),
                "-map",
                "0",
                "-c",
                "copy",
                str(temporary),
            ],
            "lossless WebM remux failed",
        )
        delivery_probe = _probe(temporary)
        delivery_streams, delivery_indexes = _stream_records(delivery_probe)
        delivery_packet_sha256 = _packet_sha256(temporary, delivery_indexes)
        delivery_duration = _duration(delivery_probe)
        if source_streams != delivery_streams:
            raise DeliveryMediaError("delivery stream identity changed")
        if source_packet_sha256 != delivery_packet_sha256:
            raise DeliveryMediaError("delivery encoded packet identity changed")
        if not math.isclose(
            source_duration, delivery_duration, rel_tol=0.0, abs_tol=0.05
        ):
            raise DeliveryMediaError("delivery duration changed")
        delivery_sha256 = _sha256_file(temporary)
        delivery_size_bytes = temporary.stat().st_size
        payload = {
            "schema_version": 1,
            "policy_version": DELIVERY_POLICY_VERSION,
            "provider": provider,
            "video_id": video_id,
            "derived": True,
            "source_name": source_path.name,
            "source_sha256": source_sha256,
            "source_size_bytes": source_size_bytes,
            "source_duration_seconds": source_duration,
            "source_streams": source_streams,
            "source_packet_sha256": source_packet_sha256,
            "delivery_name": target.name,
            "delivery_sha256": delivery_sha256,
            "delivery_size_bytes": delivery_size_bytes,
            "delivery_mime_type": "video/webm",
            "delivery_duration_seconds": delivery_duration,
            "delivery_streams": delivery_streams,
            "delivery_packet_sha256": delivery_packet_sha256,
            "ffmpeg_version": _ffmpeg_version(),
            "remux_policy": "ffmpeg -map 0 -c copy",
        }
        if target.exists() or provenance_path.exists():
            raise DeliveryMediaError("delivery artifact appeared during preparation")
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise DeliveryMediaError(
                "delivery artifact appeared during preparation"
            ) from error
    try:
        _write_new_json(provenance_path, payload)
    except Exception:
        # Preserve the installed, verified delivery artifact for audit. Deleting
        # by pathname here cannot be made race-free against an external writer.
        raise
    return verify_delivery_media(
        DeliveryMedia(
            path=target,
            provenance_path=provenance_path,
            mime_type="video/webm",
            sha256=delivery_sha256,
            size_bytes=delivery_size_bytes,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            policy_version=DELIVERY_POLICY_VERSION,
            derived=True,
        )
    )


def verify_delivery_media(delivery: DeliveryMedia) -> DeliveryMedia:
    """Revalidate delivery and provenance identities before provider upload."""

    if not isinstance(delivery, DeliveryMedia):
        raise TypeError("delivery must be DeliveryMedia")
    if delivery.path.is_symlink() or not delivery.path.is_file():
        raise DeliveryMediaError("delivery media is not a regular file")
    if delivery.provenance_path.is_symlink() or not delivery.provenance_path.is_file():
        raise DeliveryMediaError("delivery provenance is not a regular file")
    if (
        delivery.path.stat().st_size != delivery.size_bytes
        or _sha256_file(delivery.path) != delivery.sha256
    ):
        raise DeliveryMediaError("delivery media identity mismatch")
    payload = _read_json(delivery.provenance_path)
    expected = {
        "policy_version": delivery.policy_version,
        "derived": delivery.derived,
        "source_sha256": delivery.source_sha256,
        "source_size_bytes": delivery.source_size_bytes,
        "delivery_sha256": delivery.sha256,
        "delivery_size_bytes": delivery.size_bytes,
        "delivery_mime_type": delivery.mime_type,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise DeliveryMediaError("delivery provenance identity mismatch")
    return delivery


def expected_delivery_identity(*, source_sha256: str, source_size_bytes: int, source_suffix: str, provider: str) -> dict[str, object]:
    """Return a non-dispatchable identity used only to record source failures."""

    suffix = source_suffix.lower()
    derived = provider == "gemini" and suffix == ".mkv"
    return {
        "delivery_policy_version": DELIVERY_POLICY_VERSION,
        "delivery_sha256": source_sha256,
        "delivery_size_bytes": source_size_bytes,
        "delivery_mime_type": "video/webm" if derived else _MIME_TYPES.get(suffix, "application/octet-stream"),
        "delivery_derived": derived,
    }


def _delivery_from_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    provenance_path: Path,
    source_path: Path,
) -> DeliveryMedia:
    if (
        payload.get("policy_version") != DELIVERY_POLICY_VERSION
        or payload.get("derived") is not True
        or payload.get("source_sha256") != _sha256_file(source_path)
        or payload.get("source_size_bytes") != source_path.stat().st_size
    ):
        raise DeliveryMediaError("existing delivery source identity mismatch")
    digest = payload.get("delivery_sha256")
    size = payload.get("delivery_size_bytes")
    mime = payload.get("delivery_mime_type")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or mime != "video/webm"
    ):
        raise DeliveryMediaError("existing delivery provenance is invalid")
    return DeliveryMedia(
        path=path,
        provenance_path=provenance_path,
        mime_type=mime,
        sha256=digest,
        size_bytes=size,
        source_sha256=str(payload["source_sha256"]),
        source_size_bytes=int(payload["source_size_bytes"]),
        policy_version=DELIVERY_POLICY_VERSION,
        derived=True,
    )


def _probe(path: Path) -> dict[str, Any]:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        "media probe failed",
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise DeliveryMediaError("media probe returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise DeliveryMediaError("media probe returned an invalid object")
    return payload


def _stream_records(probe: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise DeliveryMediaError("media streams are missing")
    counters = {"audio": 0, "video": 0}
    records: list[dict[str, str]] = []
    indexes: dict[str, int] = {}
    for stream in streams:
        if not isinstance(stream, dict):
            raise DeliveryMediaError("media stream record is invalid")
        codec_type = stream.get("codec_type")
        if codec_type not in counters:
            raise DeliveryMediaError("non-audio/video streams cannot be losslessly remuxed")
        codec_name = stream.get("codec_name")
        absolute_index = stream.get("index")
        if not isinstance(codec_name, str) or not isinstance(absolute_index, int):
            raise DeliveryMediaError("media stream identity is invalid")
        key = f"{codec_type}:{counters[codec_type]}"
        counters[codec_type] += 1
        records.append({"key": key, "codec_type": codec_type, "codec_name": codec_name})
        indexes[key] = absolute_index
    if counters["audio"] < 1 or counters["video"] < 1:
        raise DeliveryMediaError("delivery media must contain audio and video")
    return sorted(records, key=lambda item: item["key"]), indexes


def _packet_sha256(path: Path, indexes: dict[str, int]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, index in sorted(indexes.items()):
        completed = _run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                f"0:{index}",
                "-c",
                "copy",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ],
            "encoded packet hash failed",
        )
        line = completed.stdout.strip()
        if not line.startswith("SHA256=") or len(line) != 71:
            raise DeliveryMediaError("encoded packet hash was invalid")
        hashes[key] = line.removeprefix("SHA256=").lower()
    return hashes


def _duration(probe: dict[str, Any]) -> float:
    form = probe.get("format")
    value = form.get("duration") if isinstance(form, dict) else None
    try:
        duration = float(value)
    except (TypeError, ValueError) as error:
        raise DeliveryMediaError("media duration is invalid") from error
    if not math.isfinite(duration) or duration <= 0:
        raise DeliveryMediaError("media duration is invalid")
    return duration


def _ffmpeg_version() -> str:
    completed = _run(["ffmpeg", "-version"], "ffmpeg version check failed")
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    if not first.startswith("ffmpeg version "):
        raise DeliveryMediaError("ffmpeg version was invalid")
    return first


def _run(command: list[str], message: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeliveryMediaError(message) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeliveryMediaError("delivery provenance is unreadable") from error
    if not isinstance(payload, dict):
        raise DeliveryMediaError("delivery provenance is invalid")
    return payload


def _write_or_verify_provenance(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise DeliveryMediaError("delivery provenance must not be a symlink")
    if path.exists():
        if _read_json(path) != payload:
            raise DeliveryMediaError("existing delivery provenance identity mismatch")
        return
    _write_new_json(path, payload)


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    fd: int | None = None
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise DeliveryMediaError("delivery provenance already exists")
        os.link(temporary, path)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def _delivery_claim(
    root: Path, *, provider: str, video_id: str
) -> Any:
    lock = root / f".{provider}-{video_id}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise DeliveryMediaError("delivery preparation is already claimed") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{DELIVERY_POLICY_VERSION}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


__all__ = [
    "DELIVERY_POLICY_VERSION",
    "DeliveryMedia",
    "DeliveryMediaError",
    "expected_delivery_identity",
    "prepare_delivery_media",
    "verify_delivery_media",
]
