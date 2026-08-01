"""Label-blind contracts for the TimeLens2 selector quality gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .timelens2_selector import (
    MODEL_ID as SELECTOR_MODEL_ID,
    MODEL_REVISION as SELECTOR_MODEL_REVISION,
    RUNTIME_IMAGE,
    _video_duration,
    build_hybrid_timestamps,
    build_messages,
    build_normalized_interval_result,
    canonical_sha256,
    parse_raw_interval_result,
)
from trec_video_2026.vqa_transformers_runner import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    generate_answer,
    validate_bfloat16_accelerator,
)

QUALITY_REQUEST_SCHEMA_VERSION = "timelens2-quality-requests-v1"
EXPECTED_ROW_COUNT = 10
EXPECTED_SOURCE_SHA256 = "d2cfec431652bcf7c6c83be59a4932fc0ed75016d6a3dc973e49cebdd9ed0241"
ANSWER_MODEL_ID = "Qwen/Qwen3.5-4B"
ANSWER_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
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


def extract_recipe_frames(
    *,
    video_id: str,
    media_path: Path,
    recipes: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    ffmpeg_executable: str,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    """Extract and hash the union of two exact timestamp recipes."""

    if set(recipes) != {"uniform-12", "hybrid-12"}:
        raise ValueError("frame extraction requires both matched conditions")
    condition_timestamps: dict[str, list[float]] = {}
    for condition, recipe in recipes.items():
        values = recipe.get("timestamps_seconds")
        if not isinstance(values, list) or len(values) != 12:
            raise ValueError(f"{condition} requires twelve timestamps")
        timestamps = [float(value) for value in values]
        if any(not math.isfinite(value) or value < 0 for value in timestamps):
            raise ValueError(f"{condition} contains an invalid timestamp")
        if timestamps != sorted(timestamps) or len(set(timestamps)) != 12:
            raise ValueError(f"{condition} timestamps must be distinct and ordered")
        condition_timestamps[condition] = timestamps

    unique_timestamps = sorted(
        set(condition_timestamps["uniform-12"])
        | set(condition_timestamps["hybrid-12"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    by_timestamp: dict[float, dict[str, object]] = {}
    all_frames: list[dict[str, object]] = []
    for index, timestamp in enumerate(unique_timestamps, start=1):
        frame_path = (output_dir / f"frame-{index:02d}.png").resolve()
        runner(
            [
                ffmpeg_executable,
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
                str(frame_path),
            ],
            check=True,
            capture_output=True,
        )
        if not frame_path.is_file():
            raise RuntimeError(f"ffmpeg did not create frame {index}")
        row: dict[str, object] = {
            "timestamp_seconds": timestamp,
            "path": str(frame_path),
            "sha256": _file_sha256(frame_path),
        }
        by_timestamp[timestamp] = row
        all_frames.append(row)
    hashes = [str(row["sha256"]) for row in all_frames]
    if len(set(hashes)) != len(hashes):
        raise ValueError("matched recipes must decode to distinct decoded frames")
    return {
        "schema_version": "timelens2-quality-frame-manifest-v1",
        "video_id": video_id,
        "media_sha256": _file_sha256(media_path),
        "requested_unique_timestamps": len(unique_timestamps),
        "frames": all_frames,
        "conditions": {
            condition: [dict(by_timestamp[value]) for value in timestamps]
            for condition, timestamps in condition_timestamps.items()
        },
    }


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
            frame_rows = frame_conditions[condition]
            if isinstance(frame_rows, list):
                frame_hashes = [
                    row.get("sha256") if isinstance(row, Mapping) else row
                    for row in frame_rows
                ]
            else:
                frame_hashes = frame_rows
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


def validate_quality_requests(path: Path) -> list[dict[str, object]]:
    """Validate the runtime copy produced from the exact frozen source."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != QUALITY_REQUEST_SCHEMA_VERSION:
        raise ValueError("quality request manifest has the wrong schema")
    if payload.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("quality requests do not bind the frozen ten-row source")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != EXPECTED_ROW_COUNT:
        raise ValueError("quality request manifest must contain exactly ten rows")
    seen: set[str] = set()
    validated: list[dict[str, object]] = []
    required = {
        "request_id",
        "video_id",
        "query",
        "duration_seconds",
        "media_sha256",
        "media_path",
        "source_sha256",
        "request_sha256",
    }
    for index, request in enumerate(requests, start=1):
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(f"quality request {index} has the wrong schema")
        _reject_label_keys(request, location=f"request[{index}]")
        video_id = request.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("quality requests require ten distinct video identities")
        seen.add(video_id)
        if request.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"quality request {index} source hash mismatch")
        identity = {key: value for key, value in request.items() if key != "request_sha256"}
        if request.get("request_sha256") != canonical_sha256(identity):
            raise ValueError(f"quality request {index} hash mismatch")
        media_path = Path(str(request.get("media_path")))
        if not media_path.is_absolute() or not media_path.is_file():
            raise ValueError(f"quality request {index} media is missing")
        if request.get("media_sha256") != _file_sha256(media_path):
            raise ValueError(f"quality request {index} media hash mismatch")
        _finite_positive(
            request.get("duration_seconds"),
            name=f"quality request {index} duration_seconds",
        )
        validated.append(request)
    return validated


def run_quality_inference(
    *,
    requests_path: Path,
    selector_model_dir: Path,
    answer_model_dir: Path,
    output_dir: Path,
    ffmpeg_executable: str,
) -> int:
    """Run selector and matched answers without accepting any label input."""

    store = QualityEvidenceStore(output_dir)
    failure: dict[str, object] | None = None
    succeeded = False
    try:
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
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU lacks bfloat16 support")

        store.checkpoint("request-validation")
        requests = validate_quality_requests(requests_path)

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
                duration = _video_duration(media_path)
                messages = build_messages(media_path, str(request["query"]))
                rendered = selector_processor.apply_chat_template(
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
                inputs = selector_processor(
                    text=rendered,
                    images=images,
                    videos=videos,
                    video_metadata=video_metadatas,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(selector_model.device)
                store.checkpoint(f"selector:{request_id}:inference")
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(20260730)
                started = time.monotonic()
                with torch.inference_mode():
                    generated = selector_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.01,
                        top_p=0.001,
                        top_k=1,
                        repetition_penalty=1.0,
                    )
                latency = time.monotonic() - started
                answer_ids = [
                    output[len(input_ids) :]
                    for input_ids, output in zip(inputs.input_ids, generated, strict=True)
                ]
                raw_output = selector_processor.batch_decode(
                    answer_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
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
                    duration_seconds=duration,
                    raw_output=raw_output,
                    metadata=metadata,
                )
                normalized = build_normalized_interval_result(raw)
                raw_results.append(raw)
                normalized_results.append(normalized)
                store.record_selector_attempt({"status": "accepted", **raw})
                _write_jsonl(output_dir / "raw-results.jsonl", raw_results)
                _write_jsonl(
                    output_dir / "normalized-results.jsonl",
                    normalized_results,
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
        del selector_model, selector_processor
        gc.collect()
        torch.cuda.empty_cache()
        if len(raw_results) != EXPECTED_ROW_COUNT:
            raise RuntimeError("selector gate did not complete all ten rows")

        store.checkpoint("frame-recipes")
        recipes: list[dict[str, object]] = []
        for request, normalized in zip(requests, normalized_results, strict=True):
            recipe = build_matched_recipes(
                duration_seconds=float(normalized["duration_seconds"]),
                normalized_intervals=normalized["intervals"],  # type: ignore[arg-type]
            )
            recipes.append({"video_id": request["video_id"], "conditions": recipe})
            _write_jsonl(output_dir / "frame-recipes.jsonl", recipes)

        store.checkpoint("frame-extraction")
        frame_manifests: list[dict[str, object]] = []
        for request, recipe in zip(requests, recipes, strict=True):
            manifest = extract_recipe_frames(
                video_id=str(request["video_id"]),
                media_path=Path(str(request["media_path"])),
                recipes=recipe["conditions"],  # type: ignore[arg-type]
                output_dir=output_dir / "frames" / str(request["request_id"]),
                ffmpeg_executable=ffmpeg_executable,
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
        answers: list[dict[str, object]] = []
        frame_by_video = {str(row["video_id"]): row for row in frame_manifests}
        for request in requests:
            video_id = str(request["video_id"])
            for condition in ("uniform-12", "hybrid-12"):
                store.checkpoint(f"answer:{request['request_id']}:{condition}")
                frames = frame_by_video[video_id]["conditions"][condition]  # type: ignore[index]
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
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
                except Exception as error:  # noqa: BLE001 - retain per-row failure
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
        del answer_model, answer_processor
        gc.collect()
        torch.cuda.empty_cache()

        store.checkpoint("inference-freeze")
        freeze_quality_inference(
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--container-media-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    run = subparsers.add_parser("run")
    run.add_argument("--requests", type=Path, required=True)
    run.add_argument("--selector-model-dir", type=Path, required=True)
    run.add_argument("--answer-model-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--ffmpeg", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_quality_requests(
            source_path=args.source,
            dataset_root=args.dataset_root,
            output_path=args.output,
            container_media_root=args.container_media_root,
            expected_source_sha256=args.expected_source_sha256,
        )
        return 0
    return run_quality_inference(
        requests_path=args.requests,
        selector_model_dir=args.selector_model_dir,
        answer_model_dir=args.answer_model_dir,
        output_dir=args.output_dir,
        ffmpeg_executable=args.ffmpeg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
