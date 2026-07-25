"""Label-free, paired-condition input contract for the Qwen3-Omni panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
PANEL_ID = "qwen3-omni-quality-10"
VIDEO_IDS = (
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "NUmb2P3Bzu8",
    "NkXPPJoZ-h8",
    "IWSPAfCKtV8",
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "EMI_anpHII4",
    "LusRurRrmf0",
)
CONDITION_SPECS = (
    ("frames-only", False),
    ("frames-native-audio", True),
)
FROZEN30_MANIFEST_PATH = "vqa/benchmarks/manifests/native_audio_benchmark_30.resolved.json"
FROZEN12_REQUEST_PATH = (
    "vqa/benchmarks/runs/native-audio-benchmark-v2/requests/"
    "qwen35-4b-12f-frozen30.json"
)
FROZEN12_REQUEST_SHA256 = "310790e29283ab72ed764ae91a9f1dc6fa25056b7c37cff0b8c231b2eb3d01b1"
NATIVE_AUDIO_DIRECTORY = "vqa/benchmarks/runs/native-audio-benchmark-v2/audio"
HORSE_VIDEO_ID = "NN7G28sBOlY"
HORSE_QUESTION = "What does the bystander answer when asked how old the horse is?"
HEX_DIGITS = frozenset("0123456789abcdef")
FORBIDDEN_FIELD_NAMES = frozenset(
    {"correct_answer", "reference", "distractor", "transcript"}
)
FROZEN_REQUEST_ROW_FIELDS = frozenset(
    {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_descriptions",
        "frame_paths",
        "frame_timestamps",
        "transcript",
        "media_path",
        "audio_path",
        "audio_mode",
    }
)
NULL_SOURCE_PLACEHOLDER_FIELDS = frozenset(
    {"transcript", "frame_descriptions", "audio_path", "audio_mode"}
)
FORBIDDEN_SOURCE_FIELD_NAMES = frozenset(
    {"correct_answer", "answer", "prior_answer", "prediction", "reference", "distractor"}
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _conditions() -> list[dict[str, Any]]:
    return [
        {"condition_id": condition_id, "audio_enabled": audio_enabled}
        for condition_id, audio_enabled in CONDITION_SPECS
    ]


def _is_forbidden_source_field_name(key: object) -> bool:
    return (
        isinstance(key, str)
        and (
            key.lower() in FORBIDDEN_SOURCE_FIELD_NAMES
            or "score" in key.lower()
        )
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value).issubset(HEX_DIGITS)
    )


def _resolve_relative(data_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a safe data-root-relative path")
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[0] != "vqa"
    ):
        raise ValueError(f"{field} must be a safe data-root-relative path")
    data_root = data_root.resolve()
    candidate = data_root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"{field} escapes data_root") from error
    return resolved


def _as_relative(data_root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is required")
    try:
        relative = Path(value).resolve().relative_to(data_root)
    except ValueError as error:
        raise ValueError(f"{field} must be inside data_root") from error
    result = relative.as_posix()
    _resolve_relative(data_root, result, field)
    return result


def _source_inputs(data_root: Path) -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": _sha256_file(_resolve_relative(data_root, path, "path"))}
        for path in (FROZEN30_MANIFEST_PATH, FROZEN12_REQUEST_PATH)
    ]


def _read_requests(data_root: Path) -> dict[str, Mapping[str, Any]]:
    request_path = _resolve_relative(data_root, FROZEN12_REQUEST_PATH, "request path")
    request_bytes = request_path.read_bytes()
    if _sha256_bytes(request_bytes) != FROZEN12_REQUEST_SHA256:
        raise ValueError("frozen request hash does not match the approved source")
    parsed = json.loads(request_bytes)
    if not isinstance(parsed, list):
        raise ValueError("frozen request file must contain a list")

    selected: dict[str, Mapping[str, Any]] = {}
    for request in parsed:
        if not isinstance(request, Mapping):
            raise ValueError("frozen request rows must be objects")
        video_id = request.get("video_id")
        if video_id in VIDEO_IDS:
            if any(_is_forbidden_source_field_name(key) for key in request):
                raise ValueError("forbidden source row key")
            if set(request) != FROZEN_REQUEST_ROW_FIELDS:
                raise ValueError("selected source row has an invalid schema")
            if any(
                request[field] is not None for field in NULL_SOURCE_PLACEHOLDER_FIELDS
            ):
                raise ValueError("selected source row has non-null placeholder metadata")
            if video_id in selected:
                raise ValueError(f"duplicate frozen request for {video_id}")
            selected[video_id] = request
    missing = [video_id for video_id in VIDEO_IDS if video_id not in selected]
    if missing:
        raise ValueError(f"missing frozen requests for {', '.join(missing)}")
    return selected


def _frozen_example_recipe(
    request: Mapping[str, Any], data_root: Path, video_id: str
) -> dict[str, Any]:
    question = request.get("question")
    frame_paths = request.get("frame_paths")
    timestamps = request.get("frame_timestamps")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{video_id}: question is required")
    if not isinstance(frame_paths, list) or len(frame_paths) != 12:
        raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
    if not isinstance(timestamps, list) or len(timestamps) != 12:
        raise ValueError(f"{video_id}: frame_timestamps must contain 12 values")
    return {
        "question": HORSE_QUESTION if video_id == HORSE_VIDEO_ID else question.strip(),
        "frame_paths": [
            _as_relative(data_root, path, f"{video_id}: frame_paths")
            for path in frame_paths
        ],
        "frame_timestamps": timestamps,
    }


def _frozen_recipes(data_root: Path) -> dict[str, dict[str, Any]]:
    return {
        video_id: _frozen_example_recipe(request, data_root, video_id)
        for video_id, request in _read_requests(data_root).items()
    }


def _example_from_request(
    request: Mapping[str, Any], data_root: Path, video_id: str
) -> dict[str, Any]:
    recipe = _frozen_example_recipe(request, data_root, video_id)
    audio_path = f"{NATIVE_AUDIO_DIRECTORY}/{video_id}.wav"
    return {
        "video_id": video_id,
        "question": recipe["question"],
        "prompt_repair": video_id == HORSE_VIDEO_ID,
        "frame_paths": recipe["frame_paths"],
        "frame_sha256s": [
            _sha256_file(_resolve_relative(data_root, path, "frame_paths"))
            for path in recipe["frame_paths"]
        ],
        "frame_timestamps": recipe["frame_timestamps"],
        "audio_path": audio_path,
        "audio_sha256": _sha256_file(
            _resolve_relative(data_root, audio_path, "audio_path")
        ),
    }


def build_contract(data_root: Path) -> dict[str, Any]:
    """Build the fixed ten-row, label-free Qwen3-Omni input contract."""
    data_root = data_root.resolve()
    requests = _read_requests(data_root)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "panel_id": PANEL_ID,
        "conditions": _conditions(),
        "source_inputs": _source_inputs(data_root),
        "examples": [
            _example_from_request(requests[video_id], data_root, video_id)
            for video_id in VIDEO_IDS
        ],
    }
    validate_quality_contract(contract, data_root, verify_files=True)
    return contract


def _reject_label_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_FIELD_NAMES:
                raise ValueError(f"label-bearing field is forbidden: {key}")
            _reject_label_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_label_fields(nested)


def _validate_path_hash(
    data_root: Path,
    video_id: str,
    field: str,
    relative: Any,
    expected_hash: Any,
    *,
    verify_files: bool,
) -> None:
    if not _is_sha256(expected_hash):
        raise ValueError(f"{video_id}: {field} hash must be a SHA-256")
    path = _resolve_relative(data_root, relative, field)
    if verify_files:
        if not path.is_file():
            raise ValueError(f"{video_id}: missing {field}: {relative}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"{video_id}: {field} hash does not match {relative}")


def validate_quality_contract(
    contract: Mapping[str, Any], data_root: Path, *, verify_files: bool
) -> None:
    """Reject malformed, unsafe, label-bearing, or changed quality inputs."""
    _reject_label_fields(contract)
    if set(contract) != {
        "schema_version",
        "panel_id",
        "conditions",
        "source_inputs",
        "examples",
    }:
        raise ValueError("contract has an invalid schema")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if contract.get("panel_id") != PANEL_ID:
        raise ValueError(f"panel_id must be {PANEL_ID}")
    if contract.get("conditions") != _conditions():
        raise ValueError("conditions must be the paired frames/audio conditions")

    source_inputs = contract.get("source_inputs")
    if not isinstance(source_inputs, list) or len(source_inputs) != 2:
        raise ValueError("source_inputs must contain the frozen manifest and request")
    source_paths = [item.get("path") if isinstance(item, Mapping) else None for item in source_inputs]
    if source_paths != [FROZEN30_MANIFEST_PATH, FROZEN12_REQUEST_PATH]:
        raise ValueError("source_inputs must use the frozen manifest and request paths")

    data_root = data_root.resolve()
    frozen_recipes = _frozen_recipes(data_root)
    for source in source_inputs:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ValueError("source input has an invalid schema")
        _validate_path_hash(
            data_root,
            "source input",
            "path",
            source["path"],
            source["sha256"],
            verify_files=verify_files,
        )

    examples = contract.get("examples")
    if not isinstance(examples, list) or len(examples) != len(VIDEO_IDS):
        raise ValueError("examples must contain exactly ten rows")
    if [row.get("video_id") if isinstance(row, Mapping) else None for row in examples] != list(VIDEO_IDS):
        raise ValueError("examples must use the fixed video order")

    expected_fields = {
        "video_id",
        "question",
        "prompt_repair",
        "frame_paths",
        "frame_sha256s",
        "frame_timestamps",
        "audio_path",
        "audio_sha256",
    }
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("example has an invalid schema")
        video_id = row["video_id"]
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{video_id}: question is required")
        if row["prompt_repair"] is not (video_id == HORSE_VIDEO_ID):
            raise ValueError(f"{video_id}: prompt_repair does not match the fixed panel")
        if video_id == HORSE_VIDEO_ID and row["question"] != HORSE_QUESTION:
            raise ValueError(f"{video_id}: question must use the approved prompt repair")

        frozen_recipe = frozen_recipes[video_id]
        if row["question"] != frozen_recipe["question"]:
            raise ValueError(f"{video_id}: question does not match frozen request question")

        frame_paths = row["frame_paths"]
        frame_hashes = row["frame_sha256s"]
        timestamps = row["frame_timestamps"]
        if not isinstance(frame_paths, list) or len(frame_paths) != 12:
            raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
        if not isinstance(frame_hashes, list) or len(frame_hashes) != 12:
            raise ValueError(f"{video_id}: frame_sha256s must contain 12 hashes")
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != 12
            or not all(isinstance(timestamp, (int, float)) for timestamp in timestamps)
        ):
            raise ValueError(f"{video_id}: frame_timestamps must contain 12 numbers")
        if (
            frame_paths != frozen_recipe["frame_paths"]
            or timestamps != frozen_recipe["frame_timestamps"]
        ):
            raise ValueError(f"{video_id}: frozen request evidence does not match")
        for path, digest in zip(frame_paths, frame_hashes, strict=True):
            _validate_path_hash(
                data_root,
                video_id,
                "frame_paths",
                path,
                digest,
                verify_files=verify_files,
            )
        _validate_path_hash(
            data_root,
            video_id,
            "audio_path",
            row["audio_path"],
            row["audio_sha256"],
            verify_files=verify_files,
        )


def validate_uploaded_quality_contract(
    contract: Mapping[str, Any], data_root: Path, *, verify_files: bool
) -> None:
    """Validate the frozen contract using only evidence uploaded with the task.

    The local builder deliberately checks its frozen request and manifest recipes.
    Those build-time inputs are not part of the remote task, so remote validation
    instead verifies the pinned contract's schema and every mapped evidence file.
    """
    _reject_label_fields(contract)
    if set(contract) != {
        "schema_version",
        "panel_id",
        "conditions",
        "source_inputs",
        "examples",
    }:
        raise ValueError("contract has an invalid schema")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if contract.get("panel_id") != PANEL_ID:
        raise ValueError(f"panel_id must be {PANEL_ID}")
    if contract.get("conditions") != _conditions():
        raise ValueError("conditions must be the paired frames/audio conditions")

    source_inputs = contract.get("source_inputs")
    if not isinstance(source_inputs, list) or len(source_inputs) != 2:
        raise ValueError("source_inputs must contain the frozen manifest and request")
    source_paths = [item.get("path") if isinstance(item, Mapping) else None for item in source_inputs]
    if source_paths != [FROZEN30_MANIFEST_PATH, FROZEN12_REQUEST_PATH]:
        raise ValueError("source_inputs must use the frozen manifest and request paths")
    for source in source_inputs:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ValueError("source input has an invalid schema")
        if not _is_sha256(source["sha256"]):
            raise ValueError("source input hash must be a SHA-256")

    data_root = data_root.resolve()
    examples = contract.get("examples")
    if not isinstance(examples, list) or len(examples) != len(VIDEO_IDS):
        raise ValueError("examples must contain exactly ten rows")
    if [row.get("video_id") if isinstance(row, Mapping) else None for row in examples] != list(VIDEO_IDS):
        raise ValueError("examples must use the fixed video order")

    expected_fields = {
        "video_id",
        "question",
        "prompt_repair",
        "frame_paths",
        "frame_sha256s",
        "frame_timestamps",
        "audio_path",
        "audio_sha256",
    }
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("example has an invalid schema")
        video_id = row["video_id"]
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{video_id}: question is required")
        if row["prompt_repair"] is not (video_id == HORSE_VIDEO_ID):
            raise ValueError(f"{video_id}: prompt_repair does not match the fixed panel")
        if video_id == HORSE_VIDEO_ID and row["question"] != HORSE_QUESTION:
            raise ValueError(f"{video_id}: question must use the approved prompt repair")

        frame_paths = row["frame_paths"]
        frame_hashes = row["frame_sha256s"]
        timestamps = row["frame_timestamps"]
        if not isinstance(frame_paths, list) or len(frame_paths) != 12:
            raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
        if not isinstance(frame_hashes, list) or len(frame_hashes) != 12:
            raise ValueError(f"{video_id}: frame_sha256s must contain 12 hashes")
        if (
            not isinstance(timestamps, list)
            or len(timestamps) != 12
            or not all(isinstance(timestamp, (int, float)) for timestamp in timestamps)
        ):
            raise ValueError(f"{video_id}: frame_timestamps must contain 12 numbers")
        for path, digest in zip(frame_paths, frame_hashes, strict=True):
            _validate_path_hash(
                data_root,
                video_id,
                "frame_paths",
                path,
                digest,
                verify_files=verify_files,
            )
        expected_audio_path = f"{NATIVE_AUDIO_DIRECTORY}/{video_id}.wav"
        if row["audio_path"] != expected_audio_path:
            raise ValueError(f"{video_id}: audio_path does not match the fixed panel")
        _validate_path_hash(
            data_root,
            video_id,
            "audio_path",
            row["audio_path"],
            row["audio_sha256"],
            verify_files=verify_files,
        )


def quality_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return the stable digest used to bind every paired run to this contract."""
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = build_contract(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(quality_contract_sha256(contract))


if __name__ == "__main__":
    main()
