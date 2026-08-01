"""Label-isolated three-condition TimeLens2 comparison contracts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .timelens2_selector import (
    MODEL_ID as SELECTOR_MODEL_ID,
    MODEL_REVISION as SELECTOR_MODEL_REVISION,
    RUNTIME_IMAGE,
    _video_duration,
    build_messages,
    build_normalized_interval_result,
    build_hybrid_timestamps,
    canonical_sha256,
    normalize_intervals,
    parse_raw_interval_result,
)
from trec_video_2026.vqa_transformers_runner import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    generate_answer,
    validate_bfloat16_accelerator,
)

CONDITIONS = ("uniform-12", "hybrid-12", "combined-18")
CONDITION_FRAME_COUNTS = {
    "uniform-12": 12,
    "hybrid-12": 12,
    "combined-18": 18,
}
REQUEST_SCHEMA_VERSION = "timelens2-three-condition-requests-v1"
FREEZE_SCHEMA_VERSION = "timelens2-three-condition-inference-freeze-v1"
EXPECTED_ROWS = 30
EXPECTED_ANSWERS = EXPECTED_ROWS * len(CONDITIONS)
ANSWER_MODEL_ID = "Qwen/Qwen3.5-4B"
ANSWER_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "annotation",
    "answer",
    "answers",
    "correct_answer",
    "incorrect_answer",
    "incorrect_answers",
    "reference",
    "references",
    "grade",
    "judgment",
    "prediction",
    "predictions",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class ThreeConditionEvidenceStore:
    """Incrementally retain phase, attempt, and completion evidence."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.phase = "initializing"
        self.selector_attempts: list[dict[str, object]] = []
        self.answer_attempts: list[dict[str, object]] = []
        self._persist(status="running", failure=None)

    def checkpoint(self, phase: str) -> None:
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be non-empty text")
        self.phase = phase
        self._persist(status="running", failure=None)

    def record_selector_attempt(self, attempt: Mapping[str, object]) -> None:
        self.selector_attempts.append(dict(attempt))
        self._persist(status="running", failure=None)

    def record_answer_attempt(self, attempt: Mapping[str, object]) -> None:
        self.answer_attempts.append(dict(attempt))
        self._persist(status="running", failure=None)

    def _completion(self) -> dict[str, dict[str, int]]:
        return {
            "videos": {
                "accepted": sum(
                    row.get("status") == "accepted" for row in self.selector_attempts
                ),
                "attempted": len(self.selector_attempts),
                "required": EXPECTED_ROWS,
            },
            "answers": {
                "accepted": sum(
                    row.get("status") == "passed" for row in self.answer_attempts
                ),
                "attempted": len(self.answer_attempts),
                "required": EXPECTED_ANSWERS,
            },
        }

    def _persist(
        self, *, status: str, failure: Mapping[str, object] | None
    ) -> None:
        _write_jsonl(self.output_dir / "selector-attempts.jsonl", self.selector_attempts)
        _write_jsonl(self.output_dir / "answer-attempts.jsonl", self.answer_attempts)
        _write_json(
            self.output_dir / "phase-status.json",
            {
                "status": status,
                "phase": self.phase,
                "completion": self._completion(),
                "failure": dict(failure) if failure is not None else None,
            },
        )

    def finalize(self, *, failure: Mapping[str, object] | None) -> None:
        status = "failed" if failure is not None else "completed"
        self._persist(status=status, failure=failure)
        _write_json(
            self.output_dir / "three-condition-summary.json",
            {
                "status": status,
                "phase": self.phase,
                "completion": self._completion(),
                "failure": dict(failure) if failure is not None else None,
            },
        )


def make_ffmpeg_frame_decoder(
    ffmpeg_executable: str,
) -> Callable[[Path, float, Path], None]:
    """Return the exact one-frame decoder used by the CUDA workload."""

    executable = Path(ffmpeg_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("ffmpeg executable must be an absolute file")

    def decode(media_path: Path, timestamp: float, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(executable),
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
                str(output_path),
            ],
            check=True,
        )

    return decode


def _video_frame_timestamps(media_path: Path) -> list[float]:
    import av

    timestamps: list[float] = []
    with av.open(str(media_path)) as container:
        if not container.streams.video:
            raise ValueError("media has no video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is not None and stream.time_base is not None:
                timestamps.append(float(frame.pts * stream.time_base))
    if not timestamps:
        raise ValueError("media has no timestamped video frames")
    return timestamps


def build_synthetic_canary_assets(
    *, output_dir: Path, ffmpeg_executable: str
) -> dict[str, object]:
    """Generate a tiny non-private video and 18 frames for runtime validation."""

    executable = Path(ffmpeg_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("ffmpeg executable must be an absolute file")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "synthetic-canary.mp4"
    subprocess.run(
        [
            str(executable),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=6",
            "-t",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
    )
    decoder = make_ffmpeg_frame_decoder(str(executable))
    frame_paths = []
    timestamps = [3.0 * (index + 0.5) / 18 for index in range(18)]
    for index, timestamp in enumerate(timestamps, start=1):
        path = output_dir / "synthetic-frames" / f"frame-{index:02d}.png"
        decoder(video_path, timestamp, path)
        frame_paths.append(str(path.resolve()))
    return {
        "video_path": str(video_path.resolve()),
        "frame_paths": frame_paths,
        "frame_timestamps": timestamps,
    }


def ffprobe_duration(
    media_path: Path, *, ffprobe_executable: str = "/usr/bin/ffprobe"
) -> float:
    """Read a finite positive container duration with the host CLI."""

    executable = Path(ffprobe_executable)
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("ffprobe executable must be an absolute file")
    completed = subprocess.run(
        [
            str(executable),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe returned an invalid duration")
    return duration


def _reject_label_keys(value: object, *, location: str = "source") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label leakage in {location}: {normalized}")
            _reject_label_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_keys(child, location=f"{location}[{index}]")


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("media must be a non-empty relative path")
    unresolved = Path(relative)
    candidate = (root / unresolved).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError("media must resolve beneath dataset_root")
    if unresolved.suffix:
        if not candidate.is_file():
            raise ValueError("media is not a file")
        return candidate.resolve(strict=True)
    matches = [
        Path(f"{candidate}{suffix}").resolve(strict=True)
        for suffix in (".mkv", ".mp4", ".webm", ".mov")
        if Path(f"{candidate}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise ValueError("media stem must resolve to exactly one supported file")
    return matches[0]


def prepare_frozen30_source(
    *,
    source_path: Path,
    dataset_root: Path,
    output_path: Path,
    container_media_root: Path,
    expected_source_sha256: str,
    duration_provider: Callable[[Path], float] = _video_duration,
) -> dict[str, object]:
    """Bind the exact label-free frozen-30 source to media bytes and duration."""

    source_path = source_path.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    if not _SHA256.fullmatch(expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256")
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source does not match the frozen label-free source")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _reject_label_keys(source)
    if not isinstance(source, dict):
        raise ValueError("frozen source must be an object")
    examples = source.get("examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_ROWS:
        raise ValueError("frozen source must contain exactly 30 examples")

    requests: list[dict[str, object]] = []
    seen: set[str] = set()
    required_example_keys = {"category", "media", "question", "video_id"}
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict) or set(example) != required_example_keys:
            raise ValueError(f"example {index} must contain only label-free fields")
        video_id = example["video_id"]
        question = example["question"]
        category = example["category"]
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("frozen source requires 30 distinct video identities")
        seen.add(video_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"example {index} has invalid question")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"example {index} has invalid category")
        media = _inside(dataset_root, example["media"])
        duration = float(duration_provider(media))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"example {index} has invalid duration")
        identity: dict[str, object] = {
            "request_id": f"frozen30-{index:02d}",
            "video_id": video_id,
            "category": category,
            "query": question.strip(),
            "duration_seconds": duration,
            "media_sha256": _file_sha256(media),
            "media_path": str(container_media_root / media.name),
            "source_sha256": source_sha256,
        }
        requests.append({**identity, "request_sha256": canonical_sha256(identity)})
    prepared: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "requests": requests,
    }
    _write_json(output_path, prepared)
    return prepared


def validate_runtime_requests(
    path: Path, *, expected_source_sha256: str
) -> list[dict[str, object]]:
    """Validate the exact prepared source and runtime media before model load."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("runtime request manifest has the wrong schema")
    if payload.get("source_sha256") != expected_source_sha256:
        raise ValueError("runtime requests do not bind the expected source")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != EXPECTED_ROWS:
        raise ValueError("runtime request manifest must contain exactly 30 rows")
    required = {
        "request_id",
        "video_id",
        "category",
        "query",
        "duration_seconds",
        "media_sha256",
        "media_path",
        "source_sha256",
        "request_sha256",
    }
    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    for index, request in enumerate(requests, start=1):
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(f"runtime request {index} has the wrong schema")
        _reject_label_keys(request, location=f"runtime request {index}")
        video_id = request.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("runtime requests require 30 distinct identities")
        seen.add(video_id)
        if request.get("source_sha256") != expected_source_sha256:
            raise ValueError(f"runtime request {index} source hash mismatch")
        identity = {key: value for key, value in request.items() if key != "request_sha256"}
        if request.get("request_sha256") != canonical_sha256(identity):
            raise ValueError(f"runtime request {index} hash mismatch")
        media_path = Path(str(request.get("media_path")))
        if not media_path.is_absolute() or not media_path.is_file():
            raise ValueError(f"runtime request {index} media is missing")
        if request.get("media_sha256") != _file_sha256(media_path):
            raise ValueError(f"runtime request {index} media hash mismatch")
        duration = float(request.get("duration_seconds", 0.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"runtime request {index} duration is invalid")
        validated.append(request)
    return validated


def _uniform_midpoints(duration: float, count: int) -> list[float]:
    return [duration * (index + 0.5) / count for index in range(count)]


def _global_slot(prefix: str, index: int, timestamp: float) -> dict[str, object]:
    return {
        "slot_id": f"{prefix}-{index:02d}",
        "role": "global",
        "requested_timestamp_seconds": timestamp,
    }


def build_three_condition_recipes(
    *,
    duration_seconds: float,
    normalized_intervals: Sequence[Sequence[float]],
) -> dict[str, dict[str, object]]:
    """Build uniform-12, hybrid-12, and uniform-plus-local combined-18."""

    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    intervals = normalize_intervals(normalized_intervals)
    hybrid = build_hybrid_timestamps(
        duration_seconds=duration,
        normalized_intervals=intervals,
    )
    uniform_slots = [
        _global_slot("uniform", index, timestamp)
        for index, timestamp in enumerate(_uniform_midpoints(duration, 12), start=1)
    ]
    hybrid_global_slots = [
        _global_slot("hybrid-global", index, float(timestamp))
        for index, timestamp in enumerate(hybrid["global"], start=1)
    ]
    local_slots: list[dict[str, object]] = []
    local_index = 0
    offset = 0
    for interval_index, ((start, end), count) in enumerate(
        zip(intervals, hybrid["local_counts"], strict=True)
    ):
        for timestamp in hybrid["local"][offset : offset + count]:
            local_index += 1
            local_slots.append(
                {
                    "slot_id": f"timelens-local-{local_index:02d}",
                    "role": "local",
                    "interval_index": interval_index,
                    "interval_seconds": [start, end],
                    "requested_timestamp_seconds": float(timestamp),
                }
            )
        offset += count
    if len(local_slots) != 6:
        raise ValueError("TimeLens local allocation must contain six slots")

    def ordered(slots: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        return [
            dict(slot)
            for slot in sorted(
                slots,
                key=lambda row: (
                    float(row["requested_timestamp_seconds"]),
                    str(row["slot_id"]),
                ),
            )
        ]

    return {
        "uniform-12": {"frame_count": 12, "slots": ordered(uniform_slots)},
        "hybrid-12": {
            "frame_count": 12,
            "slots": ordered(hybrid_global_slots + local_slots),
            "interval_padding_seconds": 0.0,
        },
        "combined-18": {
            "frame_count": 18,
            "slots": ordered(uniform_slots + local_slots),
            "interval_padding_seconds": 0.0,
        },
    }


def _default_frame_decoder(media: Path, timestamp: float, output: Path) -> None:
    raise RuntimeError("frame_decoder must be supplied by the runtime")


def extract_three_condition_frames(
    *,
    video_id: str,
    media_path: Path,
    recipes: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    frame_decoder: Callable[[Path, float, Path], None] = _default_frame_decoder,
    frame_timestamps_provider: Callable[[Path], Sequence[float]],
) -> dict[str, object]:
    """Decode shared slots once and enforce distinct images within conditions."""

    if set(recipes) != set(CONDITIONS):
        raise ValueError("three-condition extraction requires every condition")
    slot_definitions: dict[str, dict[str, object]] = {}
    slot_conditions: dict[str, set[str]] = {}
    condition_slot_ids: dict[str, list[str]] = {}
    for condition in CONDITIONS:
        recipe = recipes[condition]
        slots = recipe.get("slots")
        expected = CONDITION_FRAME_COUNTS[condition]
        if not isinstance(slots, list) or len(slots) != expected:
            raise ValueError(f"{condition} requires {expected} slots")
        ids: list[str] = []
        for raw_slot in slots:
            if not isinstance(raw_slot, Mapping):
                raise ValueError(f"{condition} slot must be an object")
            slot = dict(raw_slot)
            slot_id = slot.get("slot_id")
            if not isinstance(slot_id, str) or not slot_id:
                raise ValueError(f"{condition} slot_id is invalid")
            existing = slot_definitions.setdefault(slot_id, slot)
            if existing != slot:
                raise ValueError(f"shared slot {slot_id} has conflicting definitions")
            ids.append(slot_id)
            slot_conditions.setdefault(slot_id, set()).add(condition)
        if len(set(ids)) != expected:
            raise ValueError(f"{condition} slot identities must be distinct")
        condition_slot_ids[condition] = ids

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_hashes = {condition: set() for condition in CONDITIONS}
    resolved_rows: dict[str, dict[str, object]] = {}
    source_timestamps: list[float] | None = None

    def decode(slot_id: str, timestamp: float) -> dict[str, object]:
        output = output_dir / f"{slot_id}.png"
        frame_decoder(media_path, timestamp, output)
        if not output.is_file():
            raise RuntimeError(f"decoder did not create {slot_id}")
        slot = slot_definitions[slot_id]
        requested = float(slot["requested_timestamp_seconds"])
        return {
            "slot_id": slot_id,
            "role": slot["role"],
            "requested_timestamp_seconds": requested,
            "timestamp_seconds": timestamp,
            "resolution": (
                "requested"
                if timestamp == requested
                else "nearest-unused-source-pts"
            ),
            "path": str(output.resolve()),
            "sha256": _file_sha256(output),
        }

    processing_order = sorted(
        slot_definitions,
        key=lambda slot_id: (
            slot_definitions[slot_id].get("role") == "local",
            float(slot_definitions[slot_id]["requested_timestamp_seconds"]),
            slot_id,
        ),
    )
    for slot_id in processing_order:
        slot = slot_definitions[slot_id]
        requested = float(slot["requested_timestamp_seconds"])
        row = decode(slot_id, requested)
        conditions = slot_conditions[slot_id]

        def conflicts(candidate: Mapping[str, object]) -> bool:
            return any(
                candidate["sha256"] in selected_hashes[condition]
                for condition in conditions
            )

        if conflicts(row):
            if slot.get("role") != "local":
                raise ValueError("global frame collision within a condition")
            interval = slot.get("interval_seconds")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("local collision lacks interval provenance")
            start, end = float(interval[0]), float(interval[1])
            if source_timestamps is None:
                source_timestamps = sorted(
                    {
                        float(value)
                        for value in frame_timestamps_provider(media_path)
                        if math.isfinite(float(value)) and float(value) >= 0
                    }
                )
            candidates = sorted(
                (
                    value
                    for value in source_timestamps
                    if start < value < end and value != requested
                ),
                key=lambda value: (abs(value - requested), value),
            )
            for candidate in candidates:
                candidate_row = decode(slot_id, candidate)
                if not conflicts(candidate_row):
                    row = candidate_row
                    break
            else:
                raise ValueError(
                    "local frame collision has no unused source frame inside interval"
                )
        resolved_rows[slot_id] = row
        for condition in conditions:
            selected_hashes[condition].add(str(row["sha256"]))

    condition_rows = {
        condition: [dict(resolved_rows[slot_id]) for slot_id in condition_slot_ids[condition]]
        for condition in CONDITIONS
    }
    for condition, rows in condition_rows.items():
        if len({row["sha256"] for row in rows}) != CONDITION_FRAME_COUNTS[condition]:
            raise ValueError(f"{condition} decoded frames must be distinct")
    return {
        "schema_version": "timelens2-three-condition-frame-manifest-v1",
        "video_id": video_id,
        "media_sha256": _file_sha256(media_path),
        "frames": [dict(resolved_rows[slot_id]) for slot_id in processing_order],
        "conditions": condition_rows,
        "resolved_local_collisions": sum(
            row["resolution"] == "nearest-unused-source-pts"
            for row in resolved_rows.values()
        ),
    }


def freeze_three_condition_inference(
    *,
    requests: Sequence[Mapping[str, object]],
    raw_results: Sequence[Mapping[str, object]],
    normalized_results: Sequence[Mapping[str, object]],
    recipes: Sequence[Mapping[str, object]],
    frame_manifests: Sequence[Mapping[str, object]],
    answers: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, object]:
    """Freeze a complete 30-row, 90-answer label-free inference result."""

    if len(requests) != EXPECTED_ROWS:
        raise ValueError("freeze requires exactly 30 requests")
    for name, rows in (
        ("raw selector", raw_results),
        ("normalized selector", normalized_results),
        ("recipe", recipes),
        ("frame manifest", frame_manifests),
    ):
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"freeze requires exactly 30 {name} rows")
    if len(answers) != EXPECTED_ANSWERS:
        raise ValueError("freeze requires 90 complete answers")

    request_index = {str(row.get("video_id")): row for row in requests}
    if len(request_index) != EXPECTED_ROWS or "" in request_index:
        raise ValueError("freeze requires 30 distinct request identities")
    expected_ids = set(request_index)
    for name, rows in (
        ("raw", raw_results),
        ("normalized", normalized_results),
        ("recipe", recipes),
        ("frame", frame_manifests),
    ):
        index = {str(row.get("video_id")): row for row in rows}
        if set(index) != expected_ids:
            raise ValueError(f"{name} identities do not match requests")
    recipe_index = {str(row["video_id"]): row for row in recipes}
    frame_index = {str(row["video_id"]): row for row in frame_manifests}
    for video_id in expected_ids:
        for owner, index in (("recipe", recipe_index), ("frame", frame_index)):
            conditions = index[video_id].get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
                raise ValueError(f"{video_id} {owner} conditions are incomplete")
            for condition, count in CONDITION_FRAME_COUNTS.items():
                value = conditions[condition]
                rows = value.get("slots") if owner == "recipe" and isinstance(value, Mapping) else value
                if not isinstance(rows, list) or len(rows) != count:
                    raise ValueError(f"{video_id} {condition} has wrong frame count")
                if owner == "frame":
                    hashes = [row.get("sha256") for row in rows if isinstance(row, Mapping)]
                    if len(hashes) != count or len(set(hashes)) != count:
                        raise ValueError(f"{video_id} {condition} frames are not distinct")

    answer_index: dict[tuple[str, str], Mapping[str, object]] = {}
    for answer in answers:
        key = (str(answer.get("video_id")), str(answer.get("condition_id")))
        if key in answer_index or key[0] not in expected_ids or key[1] not in CONDITIONS:
            raise ValueError(f"unexpected or duplicate answer pair: {key}")
        request = request_index[key[0]]
        if answer.get("request_sha256") != request.get("request_sha256"):
            raise ValueError(f"answer request hash mismatch for {key}")
        if answer.get("status") != "passed" or not str(answer.get("answer", "")).strip():
            raise ValueError(f"answer did not pass for {key}")
        answer_index[key] = answer
    expected_keys = {(video_id, condition) for video_id in expected_ids for condition in CONDITIONS}
    if set(answer_index) != expected_keys:
        raise ValueError("freeze requires 90 complete answers")
    freeze: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "label_access": "permitted_after_verified_freeze",
        "conditions": list(CONDITIONS),
        "completion": {
            "videos": {"accepted": EXPECTED_ROWS, "required": EXPECTED_ROWS},
            "answers": {"accepted": EXPECTED_ANSWERS, "required": EXPECTED_ANSWERS},
        },
        "requests_sha256": canonical_sha256(list(requests)),
        "raw_results_sha256": canonical_sha256(list(raw_results)),
        "normalized_results_sha256": canonical_sha256(list(normalized_results)),
        "recipes_sha256": canonical_sha256(list(recipes)),
        "frame_manifests_sha256": canonical_sha256(list(frame_manifests)),
        "answers_sha256": canonical_sha256(list(answers)),
    }
    _write_json(output_path, freeze)
    return freeze


def build_blind_grading_packet(
    *,
    freeze: Mapping[str, object],
    answers: Sequence[Mapping[str, object]],
    references: Mapping[str, str],
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create answer candidates without condition identities and a separate key."""

    completion = freeze.get("completion")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("status") != "frozen"
        or not isinstance(completion, Mapping)
        or completion.get("answers") != {"accepted": 90, "required": 90}
    ):
        raise ValueError("blind grading requires a verified 30/90 freeze")
    answer_index = {
        (str(row.get("video_id")), str(row.get("condition_id"))): row
        for row in answers
    }
    video_ids = sorted(references)
    if len(video_ids) != EXPECTED_ROWS:
        raise ValueError("grading requires exactly 30 references")
    if set(answer_index) != {(video_id, condition) for video_id in video_ids for condition in CONDITIONS}:
        raise ValueError("grading answers do not match references and conditions")
    rng = random.Random(seed)
    packet_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, str]] = []
    for row_index, video_id in enumerate(video_ids, start=1):
        candidates = [
            {
                "condition_id": condition,
                "answer": str(answer_index[(video_id, condition)]["answer"]),
            }
            for condition in CONDITIONS
        ]
        rng.shuffle(candidates)
        public_candidates = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"row-{row_index:02d}-candidate-{candidate_index}"
            public_candidates.append(
                {"candidate_id": candidate_id, "answer": candidate["answer"]}
            )
            key_rows.append(
                {
                    "candidate_id": candidate_id,
                    "video_id": video_id,
                    "condition_id": str(candidate["condition_id"]),
                }
            )
        packet_rows.append(
            {
                "row_id": f"row-{row_index:02d}",
                "video_id": video_id,
                "reference": references[video_id],
                "candidates": public_candidates,
            }
        )
    return (
        {"schema_version": "timelens2-blind-grading-packet-v1", "rows": packet_rows},
        {"schema_version": "timelens2-blind-grading-key-v1", "rows": key_rows},
    )


def score_blind_judgments(
    *,
    packet: Mapping[str, object],
    key: Mapping[str, object],
    judgments: Mapping[str, str],
) -> dict[str, object]:
    """Reveal complete C/P/W judgments and summarize paired transitions."""

    packet_rows = packet.get("rows")
    key_rows = key.get("rows")
    if not isinstance(packet_rows, list) or len(packet_rows) != EXPECTED_ROWS:
        raise ValueError("judgment packet must contain 30 rows")
    if not isinstance(key_rows, list) or len(key_rows) != EXPECTED_ANSWERS:
        raise ValueError("judgment key must contain 90 rows")
    mapping: dict[str, tuple[str, str]] = {}
    for row in key_rows:
        if not isinstance(row, Mapping):
            raise ValueError("judgment key row must be an object")
        candidate_id = str(row.get("candidate_id"))
        mapping[candidate_id] = (str(row.get("video_id")), str(row.get("condition_id")))
    if set(judgments) != set(mapping) or any(value not in {"C", "P", "W"} for value in judgments.values()):
        raise ValueError("judgments must provide exactly 90 C/P/W grades")
    grade_by_pair = {mapping[candidate]: grade for candidate, grade in judgments.items()}
    counts = {
        condition: {
            grade: sum(
                value == grade
                for (video_id, candidate_condition), value in grade_by_pair.items()
                if candidate_condition == condition
            )
            for grade in ("C", "P", "W")
        }
        for condition in CONDITIONS
    }
    ordinal = {"W": 0, "P": 1, "C": 2}
    paired: dict[str, dict[str, int]] = {}
    video_ids = sorted({video_id for video_id, _ in grade_by_pair})
    for condition in CONDITIONS[1:]:
        improved = regressed = tied = 0
        for video_id in video_ids:
            baseline = ordinal[grade_by_pair[(video_id, "uniform-12")]]
            treatment = ordinal[grade_by_pair[(video_id, condition)]]
            if treatment > baseline:
                improved += 1
            elif treatment < baseline:
                regressed += 1
            else:
                tied += 1
        paired[condition] = {
            "improved": improved,
            "regressed": regressed,
            "tied": tied,
        }
    return {
        "schema_version": "timelens2-blind-grading-result-v1",
        "completion": {"accepted": len(judgments), "required": EXPECTED_ANSWERS},
        "conditions": counts,
        "paired": paired,
    }


def _selector_completion(
    *,
    processor: Any,
    model: Any,
    media_path: Path,
    query: str,
    process_vision_info: Callable[..., Any],
    torch: Any,
    max_new_tokens: int = 1024,
) -> str:
    messages = build_messages(media_path, query)
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=True)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=rendered,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    torch.manual_seed(20260801)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.01,
            top_p=0.001,
            top_k=1,
            repetition_penalty=1.0,
        )
    answer_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated, strict=True)
    ]
    return processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _validate_cuda(torch: Any) -> None:
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("NVIDIA CUDA is unavailable")
    if torch.version.hip is not None:
        raise RuntimeError("non-CUDA accelerator runtime is not permitted")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU lacks bfloat16 support")


def run_canary(
    *,
    selector_model_dir: Path,
    answer_model_dir: Path,
    output_dir: Path,
    ffmpeg_executable: str,
) -> int:
    """Exercise exact selector and 12/18-frame answer paths on synthetic media."""

    output_dir.mkdir(parents=True, exist_ok=True)
    failure: dict[str, object] | None = None
    result: dict[str, object] = {
        "schema_version": "timelens2-three-condition-canary-v1",
        "status": "running",
        "selector_model_id": SELECTOR_MODEL_ID,
        "selector_model_revision": SELECTOR_MODEL_REVISION,
        "answer_model_id": ANSWER_MODEL_ID,
        "answer_model_revision": ANSWER_MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
    }
    try:
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _validate_cuda(torch)
        assets = build_synthetic_canary_assets(
            output_dir=output_dir / "synthetic",
            ffmpeg_executable=ffmpeg_executable,
        )
        selector_processor = AutoProcessor.from_pretrained(
            selector_model_dir,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        selector_model = AutoModelForImageTextToText.from_pretrained(
            selector_model_dir,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        selector_output = _selector_completion(
            processor=selector_processor,
            model=selector_model,
            media_path=Path(str(assets["video_path"])),
            query="Where do visible colors or patterns change?",
            process_vision_info=process_vision_info,
            torch=torch,
            max_new_tokens=128,
        )
        if not selector_output.strip():
            raise RuntimeError("selector canary returned empty output")
        result["selector"] = {
            "raw_output": selector_output,
            "latency_seconds": time.monotonic() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0),
            "precision": str(selector_model.dtype).removeprefix("torch."),
        }
        del selector_model, selector_processor
        gc.collect()
        torch.cuda.empty_cache()

        answer_processor = AutoProcessor.from_pretrained(
            answer_model_dir,
            local_files_only=True,
            min_pixels=MIN_PIXELS_PER_FRAME,
            max_pixels=MAX_PIXELS_PER_FRAME,
        )
        answer_model = AutoModelForImageTextToText.from_pretrained(
            answer_model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        device, precision, attestation = validate_bfloat16_accelerator(
            answer_model, torch
        )
        answer_results = []
        all_paths = [Path(value) for value in assets["frame_paths"]]
        all_timestamps = [float(value) for value in assets["frame_timestamps"]]
        for count in (12, 18):
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            answer = generate_answer(
                processor=answer_processor,
                model=answer_model,
                question="What visual pattern is shown?",
                frame_paths=tuple(all_paths[:count]),
                frame_timestamps=tuple(all_timestamps[:count]),
                transcript=None,
                max_new_tokens=32,
            )
            if not answer.strip():
                raise RuntimeError(f"{count}-frame canary returned empty answer")
            answer_results.append(
                {
                    "frame_count": count,
                    "answer": answer,
                    "latency_seconds": time.monotonic() - started,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "device": device,
                    "precision": precision,
                }
            )
        result.update(
            {
                "status": "passed",
                "transformers_version": transformers.__version__,
                "runtime_attestation": attestation,
                "answers": answer_results,
            }
        )
    except Exception as error:  # noqa: BLE001 - durable canary failure
        failure = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        result.update({"status": "failed", "failure": failure})
    finally:
        _write_json(output_dir / "canary.json", result)
    return 0 if failure is None else 1


def run_three_condition_inference(
    *,
    requests_path: Path,
    expected_source_sha256: str,
    selector_model_dir: Path,
    answer_model_dir: Path,
    output_dir: Path,
    ffmpeg_executable: str,
) -> int:
    """Run 30 selectors and 90 answers without accepting any label input."""

    store = ThreeConditionEvidenceStore(output_dir)
    failure: dict[str, object] | None = None
    succeeded = False
    try:
        store.checkpoint("runtime-imports")
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        store.checkpoint("cuda-validation")
        _validate_cuda(torch)
        store.checkpoint("request-validation")
        requests = validate_runtime_requests(
            requests_path,
            expected_source_sha256=expected_source_sha256,
        )
        decoder = make_ffmpeg_frame_decoder(ffmpeg_executable)

        store.checkpoint("selector:model-load")
        selector_model = AutoModelForImageTextToText.from_pretrained(
            selector_model_dir,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        selector_processor = AutoProcessor.from_pretrained(
            selector_model_dir,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        raw_results: list[dict[str, object]] = []
        normalized_results: list[dict[str, object]] = []
        for request in requests:
            request_id = str(request["request_id"])
            store.checkpoint(f"selector:{request_id}:preprocess")
            try:
                media_path = Path(str(request["media_path"]))
                measured_duration = _video_duration(media_path)
                frozen_duration = float(request["duration_seconds"])
                if abs(measured_duration - frozen_duration) > 0.05:
                    raise ValueError("runtime duration differs from frozen duration")
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                raw_output = _selector_completion(
                    processor=selector_processor,
                    model=selector_model,
                    media_path=media_path,
                    query=str(request["query"]),
                    process_vision_info=process_vision_info,
                    torch=torch,
                )
                latency = time.monotonic() - started
                metadata = {
                    "request_sha256": request["request_sha256"],
                    "model_id": SELECTOR_MODEL_ID,
                    "model_revision": SELECTOR_MODEL_REVISION,
                    "runtime_image": RUNTIME_IMAGE,
                    "transformers_version": transformers.__version__,
                    "latency_seconds": latency,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "device": torch.cuda.get_device_name(0),
                    "precision": str(selector_model.dtype).removeprefix("torch."),
                }
                raw = parse_raw_interval_result(
                    request_id=request_id,
                    video_id=str(request["video_id"]),
                    duration_seconds=measured_duration,
                    raw_output=raw_output,
                    metadata=metadata,
                )
                normalized = build_normalized_interval_result(raw)
                normalized["request_sha256"] = request["request_sha256"]
                raw_results.append(raw)
                normalized_results.append(normalized)
                store.record_selector_attempt({"status": "accepted", **raw})
                _write_jsonl(output_dir / "raw-results.jsonl", raw_results)
                _write_jsonl(
                    output_dir / "normalized-results.jsonl", normalized_results
                )
            except Exception as error:  # noqa: BLE001 - retain per-row failure
                store.record_selector_attempt(
                    {
                        "status": "rejected",
                        "request_id": request_id,
                        "video_id": request["video_id"],
                        "request_sha256": request["request_sha256"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        del selector_model, selector_processor
        gc.collect()
        torch.cuda.empty_cache()
        if len(raw_results) != EXPECTED_ROWS:
            raise RuntimeError("selector phase did not complete all 30 rows")

        store.checkpoint("frame-recipes")
        recipes: list[dict[str, object]] = []
        for request, normalized in zip(requests, normalized_results, strict=True):
            conditions = build_three_condition_recipes(
                duration_seconds=float(normalized["duration_seconds"]),
                normalized_intervals=normalized["intervals"],  # type: ignore[arg-type]
            )
            recipes.append({"video_id": request["video_id"], "conditions": conditions})
            _write_jsonl(output_dir / "frame-recipes.jsonl", recipes)

        store.checkpoint("frame-extraction")
        frame_manifests: list[dict[str, object]] = []
        for request, recipe in zip(requests, recipes, strict=True):
            manifest = extract_three_condition_frames(
                video_id=str(request["video_id"]),
                media_path=Path(str(request["media_path"])),
                recipes=recipe["conditions"],  # type: ignore[arg-type]
                output_dir=output_dir / "frames" / str(request["request_id"]),
                frame_decoder=decoder,
                frame_timestamps_provider=_video_frame_timestamps,
            )
            frame_manifests.append(manifest)
            _write_jsonl(output_dir / "frame-manifests.jsonl", frame_manifests)

        store.checkpoint("answer:model-load")
        answer_processor = AutoProcessor.from_pretrained(
            answer_model_dir,
            local_files_only=True,
            min_pixels=MIN_PIXELS_PER_FRAME,
            max_pixels=MAX_PIXELS_PER_FRAME,
        )
        answer_model = AutoModelForImageTextToText.from_pretrained(
            answer_model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        answer_device, answer_precision, answer_attestation = (
            validate_bfloat16_accelerator(answer_model, torch)
        )
        frame_by_video = {str(row["video_id"]): row for row in frame_manifests}
        answers: list[dict[str, object]] = []
        for request in requests:
            video_id = str(request["video_id"])
            for condition in CONDITIONS:
                store.checkpoint(f"answer:{request['request_id']}:{condition}")
                frames = frame_by_video[video_id]["conditions"][condition]  # type: ignore[index]
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                try:
                    answer_text = generate_answer(
                        processor=answer_processor,
                        model=answer_model,
                        question=str(request["query"]),
                        frame_paths=tuple(Path(str(row["path"])) for row in frames),
                        frame_timestamps=tuple(
                            float(row["timestamp_seconds"]) for row in frames
                        ),
                        transcript=None,
                        max_new_tokens=32,
                    )
                    result: dict[str, object] = {
                        "video_id": video_id,
                        "condition_id": condition,
                        "request_sha256": request["request_sha256"],
                        "status": "passed",
                        "answer": answer_text,
                        "model_id": ANSWER_MODEL_ID,
                        "model_revision": ANSWER_MODEL_REVISION,
                        "runtime_image": RUNTIME_IMAGE,
                        "runtime": (
                            f"transformers {transformers.__version__}; "
                            f"torch {torch.__version__}"
                        ),
                        "latency_seconds": time.monotonic() - started,
                        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                        "device": answer_device,
                        "precision": answer_precision,
                        "runtime_attestation": answer_attestation,
                    }
                    answers.append(result)
                except Exception as error:  # noqa: BLE001 - retain answer failure
                    result = {
                        "video_id": video_id,
                        "condition_id": condition,
                        "request_sha256": request["request_sha256"],
                        "status": "failed",
                        "answer": None,
                        "model_id": ANSWER_MODEL_ID,
                        "model_revision": ANSWER_MODEL_REVISION,
                        "latency_seconds": time.monotonic() - started,
                        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                        "device": answer_device,
                        "precision": answer_precision,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                store.record_answer_attempt(result)
                _write_jsonl(output_dir / "answers.jsonl", answers)
                gc.collect()
                torch.cuda.empty_cache()
        del answer_model, answer_processor
        gc.collect()
        torch.cuda.empty_cache()

        store.checkpoint("inference-freeze")
        freeze_three_condition_inference(
            requests=requests,
            raw_results=raw_results,
            normalized_results=normalized_results,
            recipes=recipes,
            frame_manifests=frame_manifests,
            answers=answers,
            output_path=output_dir / "inference-freeze.json",
        )
        succeeded = True
    except Exception as error:  # noqa: BLE001 - structured terminal failure
        failure = {
            "phase": store.phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    finally:
        store.finalize(failure=failure)
    return 0 if succeeded else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--container-media-root", type=Path, required=True)
    prepare.add_argument("--expected-source-sha256", required=True)
    prepare.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    for name in ("canary", "run"):
        command = commands.add_parser(name)
        command.add_argument("--selector-model-dir", type=Path, required=True)
        command.add_argument("--answer-model-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--ffmpeg", required=True)
        if name == "run":
            command.add_argument("--requests", type=Path, required=True)
            command.add_argument("--expected-source-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_frozen30_source(
            source_path=args.source,
            dataset_root=args.dataset_root,
            output_path=args.output,
            container_media_root=args.container_media_root,
            expected_source_sha256=args.expected_source_sha256,
            duration_provider=lambda path: ffprobe_duration(
                path,
                ffprobe_executable=args.ffprobe,
            ),
        )
        return 0
    if args.command == "canary":
        return run_canary(
            selector_model_dir=args.selector_model_dir,
            answer_model_dir=args.answer_model_dir,
            output_dir=args.output_dir,
            ffmpeg_executable=args.ffmpeg,
        )
    return run_three_condition_inference(
        requests_path=args.requests,
        expected_source_sha256=args.expected_source_sha256,
        selector_model_dir=args.selector_model_dir,
        answer_model_dir=args.answer_model_dir,
        output_dir=args.output_dir,
        ffmpeg_executable=args.ffmpeg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
