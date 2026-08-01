"""Label-blind contracts for the TimeLens2 selector quality gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .timelens2_selector import build_hybrid_timestamps, canonical_sha256

QUALITY_REQUEST_SCHEMA_VERSION = "timelens2-quality-requests-v1"
EXPECTED_ROW_COUNT = 10
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "answer",
    "answers",
    "correct_answer",
    "incorrect_answer",
    "incorrect_answers",
    "reference",
    "reference_answer",
    "references",
    "prediction",
    "predictions",
    "sufficient_sets",
    "preferred_set_index",
}
_FORBIDDEN_REFERENCE_KEYS = _FORBIDDEN_LABEL_KEYS - {"answer", "answers"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_label_keys(value: object, *, location: str = "source") -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_LABEL_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"label leakage in {location}: {min(leaked)}")
        for key, child in value.items():
            _reject_label_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_keys(child, location=f"{location}[{index}]")


def _reject_reference_keys(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_REFERENCE_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"label leakage in {location}: {min(leaked)}")
        for key, child in value.items():
            _reject_reference_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_reference_keys(child, location=f"{location}[{index}]")


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return converted


def prepare_quality_requests(
    *,
    source_path: Path,
    dataset_root: Path,
    output_path: Path,
    container_media_root: Path,
    expected_source_sha256: str,
) -> dict[str, object]:
    """Bind an exact frozen label-free panel to ten checksum-verified media."""

    source_path = source_path.resolve()
    dataset_root = dataset_root.resolve()
    if not _SHA256.fullmatch(expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256")
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source does not match the frozen ten-row source")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _reject_label_keys(source)
    if not isinstance(source, dict):
        raise TypeError("frozen ten-row source must be an object")
    examples = source.get("examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_ROW_COUNT:
        raise ValueError("frozen ten-row source must contain exactly ten examples")

    requests: list[dict[str, object]] = []
    seen_video_ids: set[str] = set()
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            raise TypeError(f"frozen ten-row source example {index} must be an object")
        video_id = example.get("video_id")
        question = example.get("question")
        media_sha256 = example.get("media_sha256")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"example {index} has invalid video_id")
        if video_id in seen_video_ids:
            raise ValueError("frozen source must contain ten distinct video identities")
        seen_video_ids.add(video_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"example {index} has invalid question")
        if not isinstance(media_sha256, str) or not _SHA256.fullmatch(media_sha256):
            raise ValueError(f"example {index} has invalid media_sha256")
        duration = _finite_positive(
            example.get("duration_seconds"),
            name=f"example {index} duration_seconds",
        )
        local_media_path = (
            dataset_root
            / "vqa/2025/training/videos_original"
            / f"{video_id}.mkv"
        )
        if not local_media_path.is_file():
            raise ValueError(f"example {index} media is missing")
        if _file_sha256(local_media_path) != media_sha256:
            raise ValueError(f"example {index} media hash mismatch")
        identity: dict[str, object] = {
            "request_id": f"quality-{index:02d}",
            "video_id": video_id,
            "query": question.strip(),
            "duration_seconds": duration,
            "media_sha256": media_sha256,
            "media_path": str(container_media_root / f"{video_id}.mkv"),
            "source_sha256": source_sha256,
        }
        requests.append({**identity, "request_sha256": canonical_sha256(identity)})

    prepared: dict[str, object] = {
        "schema_version": QUALITY_REQUEST_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "requests": requests,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prepared, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prepared


def _uniform_midpoints(duration_seconds: float, count: int) -> list[float]:
    return [duration_seconds * (index + 0.5) / count for index in range(count)]


def build_matched_recipes(
    *,
    duration_seconds: float,
    normalized_intervals: Sequence[Sequence[float]],
) -> dict[str, dict[str, Any]]:
    """Build uniform-12 and no-padding hybrid-12 timestamp recipes."""

    duration = _finite_positive(duration_seconds, name="duration_seconds")
    uniform = _uniform_midpoints(duration, 12)
    hybrid = build_hybrid_timestamps(
        duration_seconds=duration,
        normalized_intervals=normalized_intervals,
    )
    global_timestamps = list(hybrid["global"])
    local_timestamps = list(hybrid["local"])
    combined = sorted(global_timestamps + local_timestamps)
    if len(combined) != 12 or len(set(combined)) != 12:
        raise ValueError("hybrid-12 requires twelve distinct timestamps")
    return {
        "uniform-12": {
            "timestamps_seconds": uniform,
            "frame_count": 12,
        },
        "hybrid-12": {
            "timestamps_seconds": combined,
            "global_timestamps_seconds": global_timestamps,
            "local_timestamps_seconds": local_timestamps,
            "local_counts": list(hybrid["local_counts"]),
            "interval_padding_seconds": 0.0,
            "frame_count": 12,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_quality_inference(
    *,
    requests: Sequence[Mapping[str, object]],
    raw_results: Sequence[Mapping[str, object]],
    normalized_results: Sequence[Mapping[str, object]],
    recipes: Sequence[Mapping[str, object]],
    frame_manifests: Sequence[Mapping[str, object]],
    answers: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, object]:
    """Freeze the complete selector, frame, and answer set before labels."""

    for name, value in (
        ("requests", requests),
        ("raw_results", raw_results),
        ("normalized_results", normalized_results),
        ("recipes", recipes),
        ("frame_manifests", frame_manifests),
        ("answers", answers),
    ):
        _reject_reference_keys(value, location=name)
    if len(requests) != EXPECTED_ROW_COUNT:
        raise ValueError("quality freeze requires exactly ten requests")
    if len(raw_results) != EXPECTED_ROW_COUNT or len(normalized_results) != EXPECTED_ROW_COUNT:
        raise ValueError("quality freeze requires ten complete selector results")
    if len(recipes) != EXPECTED_ROW_COUNT or len(frame_manifests) != EXPECTED_ROW_COUNT:
        raise ValueError("quality freeze requires ten complete frame recipe results")
    if len(answers) != EXPECTED_ROW_COUNT * 2:
        raise ValueError("quality freeze requires 20 complete answers")

    request_index: dict[str, Mapping[str, object]] = {}
    for request in requests:
        video_id = request.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("quality request has invalid video_id")
        if video_id in request_index:
            raise ValueError("quality requests contain duplicate video identities")
        request_sha256 = request.get("request_sha256")
        if not isinstance(request_sha256, str) or not _SHA256.fullmatch(request_sha256):
            raise ValueError(f"{video_id} has invalid request_sha256")
        request_index[video_id] = request

    raw_index = {str(row.get("video_id")): row for row in raw_results}
    normalized_index = {str(row.get("video_id")): row for row in normalized_results}
    recipe_index = {str(row.get("video_id")): row for row in recipes}
    frame_index = {str(row.get("video_id")): row for row in frame_manifests}
    expected_video_ids = set(request_index)
    if any(
        set(index) != expected_video_ids
        for index in (raw_index, normalized_index, recipe_index, frame_index)
    ):
        raise ValueError("quality inference rows do not match the request identities")
    for video_id, request in request_index.items():
        raw = raw_index[video_id]
        normalized = normalized_index[video_id]
        if raw.get("request_sha256") != request["request_sha256"]:
            raise ValueError(f"{video_id} selector request hash mismatch")
        if normalized.get("request_id") != raw.get("request_id"):
            raise ValueError(f"{video_id} normalized selector identity mismatch")
        conditions = recipe_index[video_id].get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != {
            "uniform-12",
            "hybrid-12",
        }:
            raise ValueError(f"{video_id} recipe conditions are incomplete")
        frame_conditions = frame_index[video_id].get("conditions")
        if not isinstance(frame_conditions, Mapping) or set(frame_conditions) != {
            "uniform-12",
            "hybrid-12",
        }:
            raise ValueError(f"{video_id} frame conditions are incomplete")
        for condition in ("uniform-12", "hybrid-12"):
            frame_hashes = frame_conditions[condition]
            if (
                not isinstance(frame_hashes, list)
                or len(frame_hashes) != 12
                or len(set(frame_hashes)) != 12
            ):
                raise ValueError(f"{video_id} {condition} must contain 12 distinct frames")

    answer_index: dict[tuple[str, str], Mapping[str, object]] = {}
    for answer in answers:
        key = (str(answer.get("video_id")), str(answer.get("condition_id")))
        if key in answer_index:
            raise ValueError(f"duplicate answer pair: {key}")
        request = request_index.get(key[0])
        if request is None or key[1] not in {"uniform-12", "hybrid-12"}:
            raise ValueError(f"unexpected answer pair: {key}")
        if answer.get("request_sha256") != request["request_sha256"]:
            raise ValueError(f"answer request hash mismatch for {key}")
        if answer.get("status") != "passed":
            raise ValueError(f"answer did not pass for {key}")
        text = answer.get("answer")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"answer is empty for {key}")
        answer_index[key] = answer
    expected_answer_keys = {
        (video_id, condition)
        for video_id in expected_video_ids
        for condition in ("uniform-12", "hybrid-12")
    }
    if set(answer_index) != expected_answer_keys:
        raise ValueError("quality freeze requires 20 complete answers")

    freeze: dict[str, object] = {
        "schema_version": "timelens2-quality-inference-freeze-v1",
        "status": "frozen",
        "label_access": "permitted_after_verified_freeze",
        "completion": {
            "videos": {"accepted": 10, "required": 10},
            "answers": {"accepted": 20, "required": 20},
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


class QualityEvidenceStore:
    """Incrementally retain quality attempts and final completion evidence."""

    def __init__(
        self,
        output_dir: Path,
        *,
        required_videos: int = EXPECTED_ROW_COUNT,
        required_answers: int = EXPECTED_ROW_COUNT * 2,
    ) -> None:
        self.output_dir = output_dir
        self.required_videos = required_videos
        self.required_answers = required_answers
        self.phase = "initializing"
        self.selector_attempts: list[dict[str, object]] = []
        self.answer_attempts: list[dict[str, object]] = []
        self._persist(status="running", failure=None)

    def checkpoint(self, phase: str) -> None:
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        self.phase = phase
        self._persist(status="running", failure=None)

    def record_selector_attempt(self, attempt: Mapping[str, object]) -> None:
        self.selector_attempts.append(dict(attempt))
        self._persist(status="running", failure=None)

    def record_answer_attempt(self, attempt: Mapping[str, object]) -> None:
        self.answer_attempts.append(dict(attempt))
        self._persist(status="running", failure=None)

    def finalize(self, *, failure: Mapping[str, object] | None) -> None:
        status = "failed" if failure is not None else "completed"
        self._persist(status=status, failure=failure)
        _write_json(
            self.output_dir / "quality-summary.json",
            {
                "status": status,
                "phase": self.phase,
                "completion": self._completion(),
                "failure": dict(failure) if failure is not None else None,
            },
        )

    def _completion(self) -> dict[str, dict[str, int]]:
        return {
            "videos": {
                "accepted": sum(
                    row.get("status") == "accepted" for row in self.selector_attempts
                ),
                "attempted": len(self.selector_attempts),
                "required": self.required_videos,
            },
            "answers": {
                "accepted": sum(
                    row.get("status") == "passed" for row in self.answer_attempts
                ),
                "attempted": len(self.answer_attempts),
                "required": self.required_answers,
            },
        }

    def _persist(
        self,
        *,
        status: str,
        failure: Mapping[str, object] | None,
    ) -> None:
        _write_jsonl(
            self.output_dir / "selector-attempts.jsonl",
            self.selector_attempts,
        )
        _write_jsonl(
            self.output_dir / "answer-attempts.jsonl",
            self.answer_attempts,
        )
        _write_json(
            self.output_dir / "phase-status.json",
            {
                "status": status,
                "phase": self.phase,
                "completion": self._completion(),
                "failure": dict(failure) if failure is not None else None,
            },
        )
