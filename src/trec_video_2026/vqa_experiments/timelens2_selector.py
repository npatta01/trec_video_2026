"""Strict TimeLens2 interval contracts and fixed-budget frame allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RAW_SCHEMA_VERSION = "timelens2-raw-interval-result-v1"
NORMALIZED_SCHEMA_VERSION = "timelens2-normalized-interval-result-v1"
REQUEST_SCHEMA_VERSION = "timelens2-compatibility-requests-v1"
MODEL_ID = "MCG-NJU/TimeLens2-2B"
MODEL_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
RUNTIME_IMAGE_DIGEST = RUNTIME_IMAGE.rsplit("@", 1)[1]

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
    if sum(clean) == 0:
        clean = [1.0] * len(clean)
    total = sum(clean)
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
) -> dict[str, object]:
    """Write a three-row private manifest containing questions but no labels."""

    gate_path = gate_path.resolve()
    dataset_root = dataset_root.resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    examples = gate.get("examples")
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("compatibility gate must contain exactly three examples")

    requests: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"gate example {index} must be an object")
        video_id = example.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"gate example {index} has no video_id")
        media_path = _inside(dataset_root, example.get("media"), name="media")
        annotation_path = _inside(
            dataset_root, example.get("annotation"), name="annotation"
        )
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        question = annotation.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gate example {index} has no question")
        media_sha256 = _file_sha256(media_path)
        request_identity = {
            "request_id": f"compatibility-{index + 1:02d}",
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
        "source_gate_sha256": _file_sha256(gate_path),
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
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
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
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


def _validate_requests(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("request manifest has the wrong schema")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != 3:
        raise ValueError("request manifest must contain exactly three rows")
    validated: list[dict[str, object]] = []
    required = {
        "request_id",
        "video_id",
        "query",
        "media_path",
        "media_sha256",
        "request_sha256",
    }
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(f"request {index} has unexpected fields")
        identity = {key: request[key] for key in required - {"request_sha256"}}
        if canonical_sha256(identity) != request["request_sha256"]:
            raise ValueError(f"request {index} hash mismatch")
        media_path = Path(str(request["media_path"]))
        if not media_path.is_absolute() or not media_path.is_file():
            raise ValueError(f"request {index} media is missing")
        if _file_sha256(media_path) != request["media_sha256"]:
            raise ValueError(f"request {index} media hash mismatch")
        validated.append(request)
    return validated


def _run_compatibility(
    *,
    model_dir: Path,
    requests_path: Path,
    output_dir: Path,
) -> int:
    import torch
    import transformers
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("NVIDIA CUDA is unavailable")
    if torch.version.hip is not None:
        raise RuntimeError("ROCm is not permitted")
    requests = _validate_requests(requests_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    raw_results: list[dict[str, object]] = []
    normalized_results: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    for request in requests:
        media_path = Path(str(request["media_path"]))
        duration = _video_duration(media_path)
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
            for input_ids, output in zip(inputs.input_ids, output_ids, strict=True)
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
        try:
            raw = parse_raw_interval_result(
                request_id=str(request["request_id"]),
                video_id=str(request["video_id"]),
                duration_seconds=duration,
                raw_output=raw_output,
                metadata=metadata,
            )
        except ValueError as error:
            attempts.append(
                {
                    "status": "rejected",
                    "request_id": request["request_id"],
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
        raw_results.append(raw)
        normalized_results.append(normalized)
        attempts.append({"status": "accepted", **raw})

    _write_jsonl(output_dir / "raw-attempts.jsonl", attempts)
    _write_jsonl(output_dir / "raw-results.jsonl", raw_results)
    _write_jsonl(output_dir / "normalized-results.jsonl", normalized_results)
    attempts_freeze = {
        "status": "frozen",
        "attempts_sha256": canonical_sha256(attempts),
        "completion": {"accepted": len(raw_results), "required": 3},
    }
    _write_json(output_dir / "attempts-freeze.json", attempts_freeze)
    gate_passed = len(raw_results) == 3
    if gate_passed:
        freeze_compatibility_results(
            raw_results=raw_results,
            normalized_results=normalized_results,
            output_path=output_dir / "selector-freeze.json",
        )
    _write_json(
        output_dir / "compatibility-summary.json",
        {
            "compatibility_gate": "passed" if gate_passed else "failed",
            "completion": {"accepted": len(raw_results), "required": 3},
            "focused_selector_pilot": "incomplete",
            "focused_selector_pilot_reason": (
                "temporal-relevant-scene-oracle-10 was absent before submission"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
        },
    )
    return 0 if gate_passed else 1


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
    run.add_argument("--mode", choices=("canary", "compatibility"), required=True)
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
        raise ValueError("compatibility mode requires --requests")
    return _run_compatibility(
        model_dir=args.model_dir,
        requests_path=args.requests,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
