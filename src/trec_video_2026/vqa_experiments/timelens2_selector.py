"""Strict TimeLens2 interval contracts and fixed-budget frame allocation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

RAW_SCHEMA_VERSION = "timelens2-raw-interval-result-v1"
NORMALIZED_SCHEMA_VERSION = "timelens2-normalized-interval-result-v1"
REQUEST_SCHEMA_VERSION = "timelens2-compatibility-requests-v1"
MEDIA_PREFLIGHT_SCHEMA_VERSION = "timelens2-media-preflight-v1"
MODEL_ID = "MCG-NJU/TimeLens2-2B"
MODEL_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
RUNTIME_IMAGE_DIGEST = RUNTIME_IMAGE.rsplit("@", 1)[1]
FROZEN_GATE_SHA256 = "1f00bc94868c29cd58f6b27a73add337ba67c35db0806d99ef2301646858e05f"
FROZEN_COMPATIBILITY_ROWS = (
    (
        "compatibility-01",
        "BDKkVYmWqdM",
        "BDKkVYmWqdM.mkv",
        "7ae4937b33b0caedbe30f6b2790c06affea52f5fb5f9666bb052424cda49a323",
        "5dfe17948025a8e9407aa9535bc2d308ff9e36bbb298a6fdf7182159946912fa",
    ),
    (
        "compatibility-02",
        "O60MtMuSpf8",
        "O60MtMuSpf8.mkv",
        "46606eb3b9ec901014a4ab2d0f3a8a562356a7af52527b1f16f3df7b88caa2bf",
        "b70521f9413ecc991631425483fd381e8e244a3db77c2d101325b7537a5383c2",
    ),
    (
        "compatibility-03",
        "Fk_j6re_Uiw",
        "Fk_j6re_Uiw.mkv",
        "aa2e987f61a2962bab5db68940b62e25376078b53c4748cffdfd2cd2c67a5c4b",
        "806eb114abcca191c0fd0fa6e10991071be7c65d5379bbfad9c522cba4bcb578",
    ),
)

_METADATA_FIELDS = {
    "request_sha256",
    "model_id",
    "model_revision",
    "runtime_image",
    "transformers_version",
    "latency_seconds",
    "peak_memory_bytes",
    "device",
    "precision",
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_raw_interval_result(
    *,
    request_id: str,
    video_id: str,
    duration_seconds: float,
    raw_output: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Parse exact JSON while preserving interval order and overlap."""

    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a non-empty string")
    duration = _finite_number(duration_seconds, name="duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("raw_output must be a non-empty string")
    _validate_metadata(metadata)
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise ValueError("raw_output must be parseable JSON") from error
    if not isinstance(decoded, list):
        raise ValueError("raw_output must be a JSON array of interval arrays")

    intervals: list[list[float]] = []
    for index, candidate in enumerate(decoded):
        if not isinstance(candidate, list) or len(candidate) != 2:
            raise ValueError(f"interval {index} must be a two-number array")
        start = _finite_number(candidate[0], name=f"interval {index} start")
        end = _finite_number(candidate[1], name=f"interval {index} end")
        if start > end:
            raise ValueError(f"interval {index} is reversed")
        if start < 0 or end > duration:
            raise ValueError(f"interval {index} is outside video duration")
        intervals.append([start, end])

    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "request_id": request_id,
        "video_id": video_id,
        "duration_seconds": duration,
        "raw_output": raw_output,
        "intervals": intervals,
        **dict(metadata),
    }


def normalize_intervals(intervals: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return a sorted overlap-merged copy without changing the raw intervals."""

    validated: list[list[float]] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
            raise ValueError(f"interval {index} must be a two-number sequence")
        if len(interval) != 2:
            raise ValueError(f"interval {index} must contain two numbers")
        start = _finite_number(interval[0], name=f"interval {index} start")
        end = _finite_number(interval[1], name=f"interval {index} end")
        if start > end:
            raise ValueError(f"interval {index} is reversed")
        validated.append([start, end])
    validated.sort(key=lambda item: (item[0], item[1]))
    merged: list[list[float]] = []
    for start, end in validated:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def build_normalized_interval_result(
    raw_result: Mapping[str, object],
) -> dict[str, object]:
    """Build the separate normalized representation linked to raw bytes."""

    if raw_result.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("normalized result requires the strict raw result schema")
    intervals = raw_result.get("intervals")
    if not isinstance(intervals, list):
        raise ValueError("raw result intervals must be a list")
    raw_output = raw_result.get("raw_output")
    if not isinstance(raw_output, str):
        raise ValueError("raw result must preserve raw_output")
    return {
        "schema_version": NORMALIZED_SCHEMA_VERSION,
        "request_id": raw_result["request_id"],
        "video_id": raw_result["video_id"],
        "duration_seconds": raw_result["duration_seconds"],
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "intervals": normalize_intervals(intervals),
    }


def build_messages(video_path: Path, query: str) -> list[dict[str, object]]:
    """Build the official TimeLens2 video-grounding prompt."""

    if not video_path.is_absolute():
        raise ValueError("video_path must be absolute")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    prompt = (
        f'Given the query: "{query}", return ALL time spans (in seconds) '
        "where the query is relevant.\n"
        "Output format MUST be a JSON array of [start, end] pairs.\n"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path.as_uri(),
                    "fps": 2.0,
                    "min_pixels": 32 * 32,
                    "max_pixels": 480 * 480,
                    "total_pixels": 128000 * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def freeze_compatibility_results(
    *,
    raw_results: Sequence[Mapping[str, object]],
    normalized_results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, object]:
    """Freeze the exact complete three-row selector result set."""

    if len(raw_results) != 3 or len(normalized_results) != 3:
        raise ValueError("compatibility freeze requires exactly three complete results")
    for index, (raw, normalized) in enumerate(
        zip(raw_results, normalized_results, strict=True)
    ):
        if raw.get("schema_version") != RAW_SCHEMA_VERSION:
            raise ValueError(f"raw result {index} has the wrong schema")
        if normalized.get("schema_version") != NORMALIZED_SCHEMA_VERSION:
            raise ValueError(f"normalized result {index} has the wrong schema")
        if (raw.get("request_id"), raw.get("video_id")) != (
            normalized.get("request_id"),
            normalized.get("video_id"),
        ):
            raise ValueError(f"raw and normalized result {index} identities differ")
        raw_output = raw.get("raw_output")
        if not isinstance(raw_output, str):
            raise ValueError(f"raw result {index} does not preserve output")
        if (
            normalized.get("raw_output_sha256")
            != hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"normalized result {index} is not linked to raw output")
    freeze = {
        "status": "frozen",
        "completion": {"accepted": 3, "required": 3},
        "raw_results_sha256": canonical_sha256(list(raw_results)),
        "normalized_results_sha256": canonical_sha256(list(normalized_results)),
        "request_sha256s": [str(row["request_sha256"]) for row in raw_results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(freeze, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return freeze


def allocate_hamilton(weights: Sequence[float], *, seats: int) -> list[int]:
    """Allocate integer seats proportionally using stable Hamilton rounding."""

    if isinstance(seats, bool) or not isinstance(seats, int) or seats < 0:
        raise ValueError("seats must be a non-negative integer")
    if not weights:
        if seats:
            raise ValueError("positive seats require at least one weight")
        return []
    clean = [
        _finite_number(value, name=f"weight {index}")
        for index, value in enumerate(weights)
    ]
    if any(value < 0 for value in clean):
        raise ValueError("weights cannot be negative")
    total = sum(clean)
    if total == 0:
        if seats == 0:
            return [0] * len(clean)
        raise ValueError("positive seats require positive total weight")
    quotas = [value * seats / total for value in clean]
    allocation = [math.floor(quota) for quota in quotas]
    remainder_seats = seats - sum(allocation)
    order = sorted(
        range(len(quotas)),
        key=lambda index: (-(quotas[index] - allocation[index]), index),
    )
    for index in order[:remainder_seats]:
        allocation[index] += 1
    return allocation


def build_hybrid_timestamps(
    *,
    duration_seconds: float,
    normalized_intervals: Sequence[Sequence[float]],
) -> dict[str, list[float] | list[int]]:
    """Build hybrid-12: six global and six Hamilton-allocated local frames."""

    duration = _finite_number(duration_seconds, name="duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    intervals = normalize_intervals(normalized_intervals)
    if not intervals:
        raise ValueError("hybrid allocation requires at least one interval")
    if intervals[0][0] < 0 or intervals[-1][1] > duration:
        raise ValueError("normalized intervals must be inside video duration")

    global_timestamps = _midpoints(0.0, duration, 6)
    counts = allocate_hamilton(
        [end - start for start, end in intervals],
        seats=6,
    )
    local_timestamps = [
        timestamp
        for (start, end), count in zip(intervals, counts, strict=True)
        for timestamp in _midpoints(start, end, count)
    ]
    return {
        "global": global_timestamps,
        "local": local_timestamps,
        "local_counts": counts,
        "all": global_timestamps + local_timestamps,
    }


def prepare_compatibility_requests(
    *,
    gate_path: Path,
    dataset_root: Path,
    output_path: Path,
    container_media_root: Path,
    expected_gate_sha256: str = FROZEN_GATE_SHA256,
    expected_rows: Sequence[tuple[str, str, str, str, str]] = (
        FROZEN_COMPATIBILITY_ROWS
    ),
) -> dict[str, object]:
    """Write a three-row private manifest containing questions but no labels."""

    gate_path = gate_path.resolve()
    dataset_root = dataset_root.resolve()
    gate_sha256 = _file_sha256(gate_path)
    if gate_sha256 != expected_gate_sha256:
        raise ValueError("source does not match the frozen compatibility gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    examples = gate.get("examples")
    if not isinstance(examples, list) or len(examples) != len(expected_rows):
        raise ValueError("source does not match the frozen compatibility gate")

    requests: list[dict[str, object]] = []
    for index, (example, expected) in enumerate(
        zip(examples, expected_rows, strict=True)
    ):
        if not isinstance(example, dict):
            raise ValueError(f"gate example {index} must be an object")
        (
            expected_request_id,
            expected_video_id,
            expected_filename,
            expected_media_hash,
            expected_query_hash,
        ) = expected
        video_id = example.get("video_id")
        if video_id != expected_video_id:
            raise ValueError("source does not match the frozen compatibility gate")
        media_path = _inside(dataset_root, example.get("media"), name="media")
        if media_path.name != expected_filename:
            raise ValueError("source does not match the frozen compatibility gate")
        annotation_path = _inside(
            dataset_root, example.get("annotation"), name="annotation"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        question = annotation.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gate example {index} has no question")
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != expected_query_hash:
            raise ValueError("source does not match the frozen compatibility gate")
        media_sha256 = _file_sha256(media_path)
        if media_sha256 != expected_media_hash:
            raise ValueError("source does not match the frozen compatibility gate")
        request_identity = {
            "request_id": expected_request_id,
            "video_id": video_id,
            "query": question,
            "media_path": str(container_media_root / media_path.name),
            "media_sha256": media_sha256,
        }
        requests.append(
            {
                **request_identity,
                "request_sha256": canonical_sha256(request_identity),
            }
        )
    prepared: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_gate_sha256": gate_sha256,
        "requests": requests,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prepared


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    fields = set(metadata)
    if fields != _METADATA_FIELDS:
        missing = sorted(_METADATA_FIELDS - fields)
        extra = sorted(fields - _METADATA_FIELDS)
        raise ValueError(f"metadata fields mismatch: missing={missing}, extra={extra}")
    request_sha256 = metadata["request_sha256"]
    if not isinstance(request_sha256, str) or len(request_sha256) != 64:
        raise ValueError("request_sha256 must contain 64 hexadecimal characters")
    try:
        int(request_sha256, 16)
    except ValueError as error:
        raise ValueError(
            "request_sha256 must contain 64 hexadecimal characters"
        ) from error
    for field in (
        "model_id",
        "model_revision",
        "runtime_image",
        "transformers_version",
        "device",
        "precision",
    ):
        if not isinstance(metadata[field], str) or not str(metadata[field]).strip():
            raise ValueError(f"{field} must be a non-empty string")
    latency = _finite_number(metadata["latency_seconds"], name="latency_seconds")
    if latency < 0:
        raise ValueError("latency_seconds cannot be negative")
    peak_memory = metadata["peak_memory_bytes"]
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory < 0
    ):
        raise ValueError("peak_memory_bytes must be a non-negative integer")


def _midpoints(start: float, end: float, count: int) -> list[float]:
    if count == 0:
        return []
    width = (end - start) / count
    if width == 0:
        return [start] * count
    return [start + (index + 0.5) * width for index in range(count)]


def _inside(root: Path, relative: object, *, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} path must be a non-empty string")
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise ValueError(f"{name} path escapes dataset root")
    if not candidate.is_file():
        raise ValueError(f"{name} file is missing")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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


class CompatibilityEvidenceStore:
    """Incrementally persist a three-row compatibility attempt."""

    required = 3

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = "setup"
        self.attempts: list[dict[str, object]] = []
        self.raw_results: list[dict[str, object]] = []
        self.normalized_results: list[dict[str, object]] = []
        self._inflight: dict[str, object] | None = None
        self._persist_rows()
        self._write_phase(status="running", failure=None)

    @property
    def accepted_count(self) -> int:
        return len(self.raw_results)

    def checkpoint(self, phase: str) -> None:
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("phase must be a non-empty string")
        self.phase = phase
        self._write_phase(status="running", failure=None)

    def begin_request(
        self,
        request: Mapping[str, object],
        *,
        phase: str,
    ) -> None:
        if self._inflight is not None:
            raise RuntimeError("cannot begin a request while another is in flight")
        identity = {
            key: request.get(key)
            for key in ("request_id", "video_id", "request_sha256")
        }
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError("request evidence identity is incomplete")
        self._inflight = identity
        self.checkpoint(phase)

    def observe_inflight(self, observation: Mapping[str, object]) -> None:
        """Persist private observations without copying request text or media paths."""

        if self._inflight is None:
            raise RuntimeError("no request is in flight")
        forbidden = {"query", "media_path", "media_sha256", "status"}
        if forbidden.intersection(observation):
            raise ValueError("in-flight observations contain forbidden request fields")
        self._inflight.update(observation)
        self._write_phase(status="running", failure=None)

    def record_attempt(
        self,
        attempt: Mapping[str, object],
        *,
        raw_result: Mapping[str, object] | None = None,
        normalized_result: Mapping[str, object] | None = None,
    ) -> None:
        if self._inflight is None:
            raise RuntimeError("no request is in flight")
        if (
            attempt.get("request_id"),
            attempt.get("video_id"),
            attempt.get("request_sha256"),
        ) != (
            self._inflight["request_id"],
            self._inflight["video_id"],
            self._inflight["request_sha256"],
        ):
            raise ValueError("attempt identity differs from the in-flight request")
        status = attempt.get("status")
        if status == "accepted":
            if raw_result is None or normalized_result is None:
                raise ValueError("accepted attempts require raw and normalized results")
            self.raw_results.append(dict(raw_result))
            self.normalized_results.append(dict(normalized_result))
        elif status != "rejected":
            raise ValueError("attempt status must be accepted or rejected")
        elif raw_result is not None or normalized_result is not None:
            raise ValueError("rejected attempts cannot add accepted results")
        self.attempts.append(dict(attempt))
        self._inflight = None
        self._persist_rows()
        self._write_phase(status="running", failure=None)

    def record_inflight_failure(
        self,
        failure: Mapping[str, object],
    ) -> None:
        if self._inflight is None:
            return
        self.attempts.append(
            {
                "status": "failed",
                **self._inflight,
                **dict(failure),
            }
        )
        self._inflight = None
        self._persist_rows()

    def finalize(
        self,
        *,
        gate_passed: bool,
        failure: Mapping[str, object] | None,
    ) -> bool:
        complete = (
            gate_passed and failure is None and self.accepted_count == self.required
        )
        completion = self._completion()
        attempts_freeze = {
            "status": "frozen",
            "evidence_status": "complete" if complete else "incomplete",
            "attempts_sha256": canonical_sha256(self.attempts),
            "completion": completion,
        }
        _write_json(self.output_dir / "attempts-freeze.json", attempts_freeze)
        if complete:
            freeze_compatibility_results(
                raw_results=self.raw_results,
                normalized_results=self.normalized_results,
                output_path=self.output_dir / "selector-freeze.json",
            )
        final_phase = str(failure["phase"]) if failure is not None else "complete"
        self.phase = final_phase
        _write_json(
            self.output_dir / "compatibility-summary.json",
            {
                "compatibility_gate": "passed" if complete else "failed",
                "completion": completion,
                "execution_status": "failed" if failure is not None else "completed",
                "failure": dict(failure) if failure is not None else None,
                "focused_selector_pilot": "incomplete",
                "focused_selector_pilot_reason": (
                    "temporal-relevant-scene-oracle-10 was absent before submission"
                ),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "phase": final_phase,
                "runtime_image": RUNTIME_IMAGE,
            },
        )
        self._persist_rows()
        self._write_phase(
            status="failed" if failure is not None else "completed",
            failure=failure,
        )
        return complete

    def _completion(self) -> dict[str, int]:
        return {
            "accepted": self.accepted_count,
            "attempted": len(self.attempts),
            "required": self.required,
        }

    def _persist_rows(self) -> None:
        _write_jsonl(self.output_dir / "raw-attempts.jsonl", self.attempts)
        _write_jsonl(self.output_dir / "raw-results.jsonl", self.raw_results)
        _write_jsonl(
            self.output_dir / "normalized-results.jsonl",
            self.normalized_results,
        )

    def _write_phase(
        self,
        *,
        status: str,
        failure: Mapping[str, object] | None,
    ) -> None:
        _write_json(
            self.output_dir / "phase-status.json",
            {
                "status": status,
                "phase": self.phase,
                "completion": self._completion(),
                "failure": dict(failure) if failure is not None else None,
                "inflight": dict(self._inflight)
                if self._inflight is not None
                else None,
            },
        )


def run_with_compatibility_evidence(
    output_dir: Path,
    workload: Callable[[CompatibilityEvidenceStore], bool],
) -> int:
    """Always freeze partial compatibility evidence and its denominator."""

    store = CompatibilityEvidenceStore(output_dir)
    failure: dict[str, object] | None = None
    gate_passed = False
    try:
        gate_passed = workload(store)
    except Exception as error:
        failure = {
            "phase": store.phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        store.record_inflight_failure(failure)
    finally:
        gate_passed = store.finalize(
            gate_passed=gate_passed,
            failure=failure,
        )
    return 0 if gate_passed else 1


def _load_processor(auto_processor: Any, model_dir: Path) -> Any:
    return auto_processor.from_pretrained(
        model_dir,
        local_files_only=True,
        fix_mistral_regex=True,
    )


def _run_canary(*, model_dir: Path, output_dir: Path) -> int:
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("NVIDIA CUDA is unavailable")
    if torch.version.hip is not None:
        raise RuntimeError("ROCm is not permitted")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU lacks bfloat16 support")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    processor = _load_processor(AutoProcessor, model_dir)
    smoke = parse_raw_interval_result(
        request_id="canary",
        video_id="synthetic-no-media",
        duration_seconds=2.0,
        raw_output="[[0, 1], [1, 2]]",
        metadata={
            "request_sha256": canonical_sha256({"canary": True}),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
            "transformers_version": transformers.__version__,
            "latency_seconds": 0.0,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0),
            "precision": "bfloat16",
        },
    )
    _write_json(
        output_dir / "canary.json",
        {
            "status": "passed",
            "model_type": config.model_type,
            "processor_class": processor.__class__.__name__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "precision": "bfloat16",
            "adapter_smoke": smoke,
        },
    )
    return 0


def _video_duration(path: Path) -> float:
    from decord import VideoReader

    reader = VideoReader(str(path))
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0 or len(reader) <= 0:
        raise ValueError("video has invalid duration metadata")
    return len(reader) / fps


def run_media_preflight_requests(
    *,
    requests: Sequence[Mapping[str, object]],
    output_path: Path,
    process_vision_info: Callable[..., tuple[object, object, object]],
    video_duration: Callable[[Path], float] = _video_duration,
) -> int:
    """Decode the frozen panel sequentially and reject backend fallback."""

    required = 3
    if len(requests) != required:
        raise ValueError("media preflight requires exactly three requests")
    summary: dict[str, object] = {
        "schema_version": MEDIA_PREFLIGHT_SCHEMA_VERSION,
        "status": "running",
        "forced_backend": "torchvision",
        "runtime_image": RUNTIME_IMAGE,
        "completion": {"accepted": 0, "attempted": 0, "required": required},
        "rows": [],
        "failure": None,
    }
    _write_json(output_path, summary)

    completion = summary["completion"]
    rows = summary["rows"]
    assert isinstance(completion, dict)
    assert isinstance(rows, list)
    for request in requests:
        identity = {
            key: request.get(key)
            for key in ("request_id", "video_id", "request_sha256")
        }
        if any(not isinstance(value, str) or not value for value in identity.values()):
            raise ValueError("media preflight request identity is incomplete")
        completion["attempted"] = int(completion["attempted"]) + 1
        started = time.monotonic()
        try:
            media_path = Path(str(request["media_path"]))
            duration = _finite_number(
                video_duration(media_path),
                name="duration_seconds",
            )
            if duration <= 0:
                raise ValueError("duration_seconds must be positive")
            messages = build_messages(media_path, str(request["query"]))
            _, videos, _ = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if not isinstance(videos, Sequence) or len(videos) != 1:
                raise RuntimeError("media preflight requires exactly one video")
            video_record = videos[0]
            if not isinstance(video_record, Sequence) or len(video_record) != 2:
                raise RuntimeError("media preflight requires video metadata")
            video, metadata = video_record
            if not isinstance(metadata, Mapping):
                raise RuntimeError("media preflight requires video metadata")
            backend = metadata.get("video_backend")
            if backend != "torchvision":
                raise RuntimeError("media preflight requires torchvision backend")
            shape = [int(value) for value in getattr(video, "shape", ())]
            if len(shape) != 4 or any(value <= 0 for value in shape):
                raise RuntimeError("media preflight returned an invalid video tensor")
            completion["accepted"] = int(completion["accepted"]) + 1
            rows.append(
                {
                    "status": "accepted",
                    **identity,
                    "backend": backend,
                    "duration_seconds": duration,
                    "frames": shape[0],
                    "shape": shape,
                    "latency_seconds": time.monotonic() - started,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
            del video, videos
            gc.collect()
        except Exception as error:
            failure = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            rows.append({"status": "failed", **identity, **failure})
            summary["status"] = "failed"
            summary["failure"] = failure
            _write_json(output_path, summary)
            return 1
        _write_json(output_path, summary)

    summary["status"] = "passed"
    _write_json(output_path, summary)
    return 0


def validate_compatibility_requests(
    path: Path,
    *,
    expected_gate_sha256: str = FROZEN_GATE_SHA256,
    expected_rows: Sequence[tuple[str, str, str, str, str]] = (
        FROZEN_COMPATIBILITY_ROWS
    ),
) -> list[dict[str, object]]:
    """Validate the exact frozen gate, identities, and mounted media bytes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "source_gate_sha256", "requests"}:
        raise ValueError(
            "request manifest does not match the frozen compatibility gate"
        )
    if (
        payload.get("schema_version") != REQUEST_SCHEMA_VERSION
        or payload.get("source_gate_sha256") != expected_gate_sha256
    ):
        raise ValueError(
            "request manifest does not match the frozen compatibility gate"
        )
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != len(expected_rows):
        raise ValueError(
            "request manifest does not match the frozen compatibility gate"
        )
    identities = [
        (request.get("request_id"), request.get("video_id"))
        for request in requests
        if isinstance(request, dict)
    ]
    if len(identities) != len(expected_rows) or len(set(identities)) != len(
        expected_rows
    ):
        raise ValueError(
            "request identities do not match the frozen compatibility gate"
        )
    validated: list[dict[str, object]] = []
    required = {
        "request_id",
        "video_id",
        "query",
        "media_path",
        "media_sha256",
        "request_sha256",
    }
    for index, (request, expected) in enumerate(
        zip(requests, expected_rows, strict=True)
    ):
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(f"request {index} has unexpected fields")
        (
            expected_request_id,
            expected_video_id,
            expected_filename,
            expected_media_hash,
            expected_query_hash,
        ) = expected
        if (
            request["request_id"] != expected_request_id
            or request["video_id"] != expected_video_id
            or Path(str(request["media_path"])).name != expected_filename
            or request["media_sha256"] != expected_media_hash
        ):
            raise ValueError(
                f"request {index} does not match the frozen compatibility gate"
            )
        if not isinstance(request["query"], str) or not request["query"].strip():
            raise ValueError(f"request {index} has an empty query")
        if (
            hashlib.sha256(request["query"].encode("utf-8")).hexdigest()
            != expected_query_hash
        ):
            raise ValueError(
                f"request {index} does not match the frozen compatibility gate"
            )
        identity = {key: request[key] for key in required - {"request_sha256"}}
        if canonical_sha256(identity) != request["request_sha256"]:
            raise ValueError(f"request {index} hash mismatch")
        media_path = Path(str(request["media_path"]))
        if not media_path.is_absolute() or not media_path.is_file():
            raise ValueError(f"request {index} media is missing")
        if _file_sha256(media_path) != request["media_sha256"]:
            raise ValueError(
                f"request {index} media does not match the frozen compatibility gate"
            )
        validated.append(request)
    return validated


def _run_compatibility(
    *,
    model_dir: Path,
    requests_path: Path,
    output_dir: Path,
) -> int:
    def workload(store: CompatibilityEvidenceStore) -> bool:
        store.checkpoint("runtime-imports")
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        store.checkpoint("cuda-validation")
        if not torch.cuda.is_available() or not torch.version.cuda:
            raise RuntimeError("NVIDIA CUDA is unavailable")
        if torch.version.hip is not None:
            raise RuntimeError("ROCm is not permitted")

        store.checkpoint("request-validation")
        requests = validate_compatibility_requests(requests_path)

        store.checkpoint("model-load")
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        )
        processor = _load_processor(AutoProcessor, model_dir)

        for request in requests:
            request_id = str(request["request_id"])
            store.begin_request(
                request,
                phase=f"panel:{request_id}:media",
            )
            media_path = Path(str(request["media_path"]))
            duration = _video_duration(media_path)
            store.observe_inflight({"duration_seconds": duration})

            store.checkpoint(f"panel:{request_id}:preprocess")
            messages = build_messages(media_path, str(request["query"]))
            text = processor.apply_chat_template(
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
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            ).to(model.device)

            store.checkpoint(f"panel:{request_id}:inference")
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(20260730)
            started = time.monotonic()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.01,
                    top_p=0.001,
                    top_k=1,
                    repetition_penalty=1.0,
                )
            latency = time.monotonic() - started
            output_ids = [
                output[len(input_ids) :]
                for input_ids, output in zip(
                    inputs.input_ids,
                    output_ids,
                    strict=True,
                )
            ]
            raw_output = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            metadata = {
                "request_sha256": request["request_sha256"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "runtime_image": RUNTIME_IMAGE,
                "transformers_version": transformers.__version__,
                "latency_seconds": latency,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "device": torch.cuda.get_device_name(0),
                "precision": str(model.dtype).removeprefix("torch."),
            }
            store.observe_inflight(
                {
                    "raw_output": raw_output,
                    **metadata,
                }
            )
            store.checkpoint(f"panel:{request_id}:parse")
            try:
                raw = parse_raw_interval_result(
                    request_id=request_id,
                    video_id=str(request["video_id"]),
                    duration_seconds=duration,
                    raw_output=raw_output,
                    metadata=metadata,
                )
            except ValueError as error:
                store.record_attempt(
                    {
                        "status": "rejected",
                        "request_id": request_id,
                        "video_id": request["video_id"],
                        "duration_seconds": duration,
                        "raw_output": raw_output,
                        **metadata,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                continue
            normalized = build_normalized_interval_result(raw)
            store.record_attempt(
                {"status": "accepted", **raw},
                raw_result=raw,
                normalized_result=normalized,
            )

        return store.accepted_count == store.required

    return run_with_compatibility_evidence(output_dir, workload)


def _run_media_preflight(*, requests_path: Path, output_dir: Path) -> int:
    if os.environ.get("FORCE_QWENVL_VIDEO_READER") != "torchvision":
        raise RuntimeError("media preflight must force torchvision before import")
    from qwen_vl_utils import process_vision_info

    requests = validate_compatibility_requests(requests_path)
    return run_media_preflight_requests(
        requests=requests,
        output_path=output_dir / "media-preflight.json",
        process_vision_info=process_vision_info,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-compatibility")
    prepare.add_argument("--gate", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--container-media-root",
        type=Path,
        default=Path("/dstack/input/media"),
    )
    run = subparsers.add_parser("run")
    run.add_argument(
        "--mode",
        choices=("canary", "media-preflight", "compatibility"),
        required=True,
    )
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--requests", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-compatibility":
        prepared = prepare_compatibility_requests(
            gate_path=args.gate,
            dataset_root=args.dataset_root,
            output_path=args.output,
            container_media_root=args.container_media_root,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "requests": len(prepared["requests"]),
                    "sha256": _file_sha256(args.output),
                },
                sort_keys=True,
            )
        )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "canary":
        if args.requests is not None:
            raise ValueError("canary must not receive private requests")
        return _run_canary(model_dir=args.model_dir, output_dir=args.output_dir)
    if args.requests is None:
        raise ValueError(f"{args.mode} mode requires --requests")
    if args.mode == "media-preflight":
        return _run_media_preflight(
            requests_path=args.requests,
            output_dir=args.output_dir,
        )
    return _run_compatibility(
        model_dir=args.model_dir,
        requests_path=args.requests,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
