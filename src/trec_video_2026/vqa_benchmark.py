"""Deterministic inputs and evidence caching for local VQA benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


EXTRACTOR_VERSION = "ffmpeg-bin-midpoint-v1"


@dataclass(frozen=True)
class InferenceExample:
    video_id: str
    category: str
    question: str
    media_path: Path


@dataclass(frozen=True)
class ReferenceAnswer:
    video_id: str
    reference_answer: str


@dataclass(frozen=True)
class ManifestExample:
    video_id: str
    category: str
    question: str
    media_path: Path
    annotation_path: Path
    reference_answer: str

    @property
    def inference(self) -> InferenceExample:
        return InferenceExample(
            video_id=self.video_id,
            category=self.category,
            question=self.question,
            media_path=self.media_path,
        )

    @property
    def reference(self) -> ReferenceAnswer:
        return ReferenceAnswer(self.video_id, self.reference_answer)


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: int
    gate_id: str
    examples: tuple[ManifestExample, ...]

    @property
    def inference_examples(self) -> tuple[InferenceExample, ...]:
        return tuple(example.inference for example in self.examples)

    @property
    def references(self) -> tuple[ReferenceAnswer, ...]:
        return tuple(example.reference for example in self.examples)


@dataclass(frozen=True)
class MediaMetadata:
    duration_seconds: float
    has_video: bool
    has_audio: bool
    stream_types: tuple[str, ...]


@dataclass(frozen=True)
class SourceFingerprint:
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class EvidenceParameters:
    frame_count: int = 12
    extractor_version: str = EXTRACTOR_VERSION


@dataclass(frozen=True)
class FrameArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True)
class EvidenceRecord:
    cache_key: str
    cache_dir: Path
    source: SourceFingerprint
    parameters: EvidenceParameters
    duration_seconds: float
    timestamps_seconds: tuple[float, ...]
    frames: tuple[FrameArtifact, ...]


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    video_id: str
    condition_id: str
    started_at: str


@dataclass(frozen=True)
class ResultRecord:
    attempt_id: str
    status: str
    answer: str | None
    elapsed_seconds: float


class StaleEvidenceError(RuntimeError):
    """Raised when cached evidence does not match its declared identity."""


def _resolved_child(root: Path, relative: str, kind: str) -> Path:
    root = root.resolve(strict=True)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"{kind} path must resolve beneath the dataset root")
    if not candidate.is_file():
        raise ValueError(f"{kind} file does not exist: {candidate}")
    return candidate


def load_manifest(config_path: Path, dataset_root: Path) -> BenchmarkManifest:
    """Resolve a benchmark manifest while keeping inference and labels separate."""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema version")
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("Manifest examples must be a non-empty list")

    examples = []
    seen = set()
    for raw in raw_examples:
        video_id = _required_text(raw, "video_id")
        if video_id in seen:
            raise ValueError(f"Duplicate video ID: {video_id}")
        seen.add(video_id)
        annotation_path = _resolved_child(
            dataset_root, _required_text(raw, "annotation"), "Annotation"
        )
        media_path = _resolved_child(
            dataset_root, _required_text(raw, "media"), "Media"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("video_url", "").rsplit("=", 1)[-1] != video_id:
            raise ValueError(f"Annotation video ID does not match {video_id}")
        examples.append(
            ManifestExample(
                video_id=video_id,
                category=_required_text(raw, "category"),
                question=_required_text(annotation, "question"),
                media_path=media_path,
                annotation_path=annotation_path,
                reference_answer=_required_text(annotation, "correct_answer"),
            )
        )
    return BenchmarkManifest(
        schema_version=1,
        gate_id=_required_text(payload, "gate_id"),
        examples=tuple(examples),
    )


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def serialize_manifest(manifest: BenchmarkManifest) -> str:
    """Return canonical JSON with references isolated from inference records."""
    examples = [
        {
            "video_id": item.video_id,
            "category": item.category,
            "question": item.question,
            "media_path": str(item.media_path),
        }
        for item in manifest.inference_examples
    ]
    references = [asdict(item) for item in manifest.references]
    payload = {
        "schema_version": manifest.schema_version,
        "gate_id": manifest.gate_id,
        "examples": examples,
        "references": references,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def probe_media(path: Path, runner=subprocess.run) -> MediaMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    completed = runner(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream_types = tuple(
        stream["codec_type"]
        for stream in payload.get("streams", [])
        if stream.get("codec_type")
    )
    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unable to determine media duration: {path}") from error
    return MediaMetadata(
        duration_seconds=duration,
        has_video="video" in stream_types,
        has_audio="audio" in stream_types,
        stream_types=stream_types,
    )


def bin_midpoint_timestamps(duration_seconds: float, count: int = 12) -> tuple[float, ...]:
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive")
    if count <= 0:
        raise ValueError("Frame count must be positive")
    width = duration_seconds / count
    return tuple(round((index + 0.5) * width, 6) for index in range(count))


def source_fingerprint(path: Path) -> SourceFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return SourceFingerprint(sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def cache_key(
    source: SourceFingerprint, parameters: EvidenceParameters = EvidenceParameters()
) -> str:
    identity = {
        "source": asdict(source),
        "parameters": asdict(parameters),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cached_evidence(cache_dir: Path, expected_key: str) -> EvidenceRecord:
    try:
        payload = json.loads((cache_dir / "evidence.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StaleEvidenceError("Cached evidence metadata is missing or invalid") from error
    if payload.get("cache_key") != expected_key:
        raise StaleEvidenceError("Cached evidence key does not match")
    try:
        declared_source = SourceFingerprint(**payload["source"])
        declared_parameters = EvidenceParameters(**payload["parameters"])
    except (KeyError, TypeError) as error:
        raise StaleEvidenceError("Cached evidence identity is invalid") from error
    if cache_key(declared_source, declared_parameters) != expected_key:
        raise StaleEvidenceError("Cached evidence identity does not match its key")

    frames = []
    for raw in payload.get("frames", []):
        path = (cache_dir / raw["path"]).resolve()
        if not path.is_relative_to(cache_dir.resolve()) or not path.is_file():
            raise StaleEvidenceError("Cached frame path is missing or invalid")
        actual = source_fingerprint(path).sha256
        if actual != raw.get("sha256"):
            raise StaleEvidenceError("Cached frame checksum does not match")
        frames.append(FrameArtifact(path=path, sha256=actual))

    parameters = declared_parameters
    if len(frames) != parameters.frame_count:
        raise StaleEvidenceError("Cached frame count does not match")
    try:
        duration_seconds = float(payload["duration_seconds"])
        timestamps = tuple(float(value) for value in payload["timestamps_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise StaleEvidenceError("Cached timestamp metadata is invalid") from error
    if len(timestamps) != parameters.frame_count:
        raise StaleEvidenceError("Cached timestamp count does not match")
    if timestamps != bin_midpoint_timestamps(duration_seconds, parameters.frame_count):
        raise StaleEvidenceError("Cached timestamps are not bin midpoints")
    return EvidenceRecord(
        cache_key=expected_key,
        cache_dir=cache_dir.resolve(),
        source=declared_source,
        parameters=parameters,
        duration_seconds=duration_seconds,
        timestamps_seconds=timestamps,
        frames=tuple(frames),
    )


def build_frame_evidence(
    media_path: Path,
    evidence_root: Path,
    parameters: EvidenceParameters = EvidenceParameters(),
    runner=subprocess.run,
) -> EvidenceRecord:
    """Extract checksum-verified frames and publish the cache atomically."""
    source = source_fingerprint(media_path)
    key = cache_key(source, parameters)
    evidence_root.mkdir(parents=True, exist_ok=True)
    destination = evidence_root.resolve() / key
    if destination.exists():
        return load_cached_evidence(destination, key)

    metadata = probe_media(media_path, runner=runner)
    if not metadata.has_video:
        raise ValueError(f"Media has no video stream: {media_path}")
    timestamps = bin_midpoint_timestamps(
        metadata.duration_seconds, count=parameters.frame_count
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=evidence_root))
    try:
        frames = []
        for index, timestamp in enumerate(timestamps, start=1):
            output = staging / f"frame-{index:02d}.png"
            runner(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.6f}",
                    "-i",
                    str(media_path),
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-y",
                    str(output),
                ],
                check=True,
                capture_output=True,
            )
            frames.append(
                {"path": output.name, "sha256": source_fingerprint(output).sha256}
            )
        record = {
            "schema_version": 1,
            "cache_key": key,
            "source": asdict(source),
            "parameters": asdict(parameters),
            "duration_seconds": metadata.duration_seconds,
            "timestamps_seconds": list(timestamps),
            "frames": frames,
        }
        metadata_path = staging / "evidence.json"
        metadata_path.write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        with metadata_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_cached_evidence(destination, key)
