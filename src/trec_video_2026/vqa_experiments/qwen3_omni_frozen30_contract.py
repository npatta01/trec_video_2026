"""Build and validate the label-free Qwen3-Omni frozen-30 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from trec_video_2026.vqa_experiments.qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    PROFILES,
    Frozen30ExperimentProfile,
    profile_for_key,
)

SCHEMA_VERSION = 1
PANEL_ID = "qwen3-omni-frozen30"
RUN_ID = "qwen3-omni-frozen30-audio-20260731-a4"
CANARY_VIDEO_ID = "NkXPPJoZ-h8"
FROZEN30_MANIFEST_PATH = (
    "vqa/benchmarks/manifests/native_audio_benchmark_30.resolved.json"
)
FROZEN30_MANIFEST_SHA256 = (
    "0beca69d5013aaba67b726f1de8e80aca920ed32be34d3da43f3c1e0f42db6d1"
)
FROZEN12_REQUEST_PATH = (
    "vqa/benchmarks/runs/native-audio-benchmark-v2/requests/qwen35-4b-12f-frozen30.json"
)
FROZEN12_REQUEST_SHA256 = (
    "310790e29283ab72ed764ae91a9f1dc6fa25056b7c37cff0b8c231b2eb3d01b1"
)
NATIVE_AUDIO_DIRECTORY = "vqa/benchmarks/runs/native-audio-benchmark-v2/audio"

VIDEO_IDS = (
    "LKiaCf1QZFw",
    "Ie-KCtcZ7uc",
    "J1CS47pBX_M",
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "NUmb2P3Bzu8",
    "IRTTtnSRTyY",
    "IWSPAfCKtV8",
    "Jc_vZjKwtvg",
    "MdMKHhFFqJQ",
    "NkXPPJoZ-h8",
    "OzNdzgQi1CA",
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
    "PcDuz7j9LL8",
    "C4j3Pp79KSg",
    "Hw0bRvn1ZIo",
    "J-sG9vePDmI",
    "Jc2BdtyAF48",
    "HyOKrz1C4zU",
    "Msbpqv5aOxk",
    "JOClNGH6yL8",
    "EMI_anpHII4",
    "IGL7j9-JRG4",
    "JSOh_pKFq84",
    "QFljXc4WpFA",
    "PiIrinu42DQ",
    "LusRurRrmf0",
)
CONDITION_SPECS = (
    ("qwen3-omni-frames-only-frozen30", False),
    ("qwen3-omni-frames-native-audio-frozen30", True),
)

HEX_DIGITS = frozenset("0123456789abcdef")
REQUEST_ROW_FIELDS = frozenset(
    {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "frame_descriptions",
        "transcript",
        "media_path",
        "audio_path",
        "audio_mode",
    }
)
NULL_SOURCE_PLACEHOLDER_FIELDS = frozenset(
    {"transcript", "frame_descriptions", "audio_path", "audio_mode"}
)
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "answer",
        "correct_answer",
        "reference",
        "distractor",
        "transcript",
        "prediction",
        "prior_answer",
        "category",
        "annotation",
        "annotation_path",
        "media",
        "media_path",
        "grade",
    }
)
FORBIDDEN_SOURCE_FIELD_NAMES = frozenset(
    {
        "answer",
        "correct_answer",
        "reference",
        "distractor",
        "prediction",
        "prior_answer",
        "grade",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conditions(profile: Frozen30ExperimentProfile) -> list[dict[str, Any]]:
    return [
        {"condition_id": condition_id, "audio_enabled": audio_enabled}
        for condition_id, audio_enabled in profile.condition_specs
    ]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value).issubset(HEX_DIGITS)
    )


def _is_valid_frame_timestamps(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 12
        and all(
            type(timestamp) in (int, float) and math.isfinite(timestamp)
            for timestamp in value
        )
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
    root = data_root.resolve()
    resolved = root.joinpath(*relative.parts).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes data_root") from error
    return resolved


def _as_relative(data_root: Path, value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is required")
    try:
        relative = Path(value).resolve().relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} must be inside data_root") from error
    result = relative.as_posix()
    _resolve_relative(data_root, result, field)
    return result


def _load_pinned_json(
    data_root: Path, relative: str, expected_sha256: str, label: str
) -> Any:
    source = _resolve_relative(data_root, relative, f"{label} path")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{label} hash does not match the approved source")
    return json.loads(raw)


def _read_manifest(data_root: Path) -> dict[str, Mapping[str, Any]]:
    parsed = _load_pinned_json(
        data_root, FROZEN30_MANIFEST_PATH, FROZEN30_MANIFEST_SHA256, "manifest"
    )
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("examples"), list):
        raise ValueError("frozen manifest must contain an examples list")
    rows = parsed["examples"]
    if len(rows) != len(VIDEO_IDS):
        raise ValueError("frozen manifest must contain exactly thirty rows")
    result: dict[str, Mapping[str, Any]] = {}
    for expected_id, row in zip(VIDEO_IDS, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("video_id") != expected_id:
            raise ValueError("frozen manifest IDs must use the approved order")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{expected_id}: manifest question is required")
        if expected_id in result:
            raise ValueError(f"duplicate frozen manifest for {expected_id}")
        result[expected_id] = row
    return result


def _read_requests(data_root: Path) -> dict[str, Mapping[str, Any]]:
    parsed = _load_pinned_json(
        data_root, FROZEN12_REQUEST_PATH, FROZEN12_REQUEST_SHA256, "frozen request"
    )
    if not isinstance(parsed, list) or len(parsed) != len(VIDEO_IDS):
        raise ValueError("frozen request file must contain exactly thirty rows")
    result: dict[str, Mapping[str, Any]] = {}
    for expected_id, row in zip(VIDEO_IDS, parsed, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("frozen request rows must be objects")
        if row.get("video_id") != expected_id:
            if row.get("video_id") in result:
                raise ValueError(f"duplicate frozen request for {row['video_id']}")
            raise ValueError("frozen request IDs must use the approved order")
        if any(
            isinstance(key, str)
            and (key.lower() in FORBIDDEN_SOURCE_FIELD_NAMES or "score" in key.lower())
            for key in row
        ):
            raise ValueError("forbidden source row key")
        if set(row) != REQUEST_ROW_FIELDS:
            raise ValueError("selected source row has an invalid schema")
        if any(row[field] is not None for field in NULL_SOURCE_PLACEHOLDER_FIELDS):
            raise ValueError("selected source row has non-null placeholder metadata")
        if not isinstance(row["media_path"], str) or not row["media_path"]:
            raise ValueError("selected source row requires non-null media_path")
        if expected_id in result:
            raise ValueError(f"duplicate frozen request for {expected_id}")
        result[expected_id] = row
    return result


def _frozen_recipes(data_root: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_manifest(data_root)
    requests = _read_requests(data_root)
    recipes: dict[str, dict[str, Any]] = {}
    for video_id in VIDEO_IDS:
        request = requests[video_id]
        question = request.get("question")
        frame_paths = request.get("frame_paths")
        timestamps = request.get("frame_timestamps")
        if question != manifest[video_id].get("question"):
            raise ValueError(f"{video_id}: frozen source questions do not match")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{video_id}: question is required")
        if not isinstance(frame_paths, list) or len(frame_paths) != 12:
            raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
        if not _is_valid_frame_timestamps(timestamps):
            raise ValueError(
                f"{video_id}: frame_timestamps must contain 12 finite numbers"
            )
        recipes[video_id] = {
            "question": question,
            "frame_paths": [
                _as_relative(data_root, path, f"{video_id}: frame_paths")
                for path in frame_paths
            ],
            "frame_timestamps": timestamps,
        }
    return recipes


def _source_inputs() -> list[dict[str, str]]:
    return [
        {"path": FROZEN30_MANIFEST_PATH, "sha256": FROZEN30_MANIFEST_SHA256},
        {"path": FROZEN12_REQUEST_PATH, "sha256": FROZEN12_REQUEST_SHA256},
    ]


def build_contract(
    data_root: Path, profile: Frozen30ExperimentProfile = A4_PROFILE
) -> dict[str, Any]:
    """Build the immutable, label-free paired inference input contract."""
    data_root = data_root.resolve()
    recipes = _frozen_recipes(data_root)
    examples = []
    for video_id in VIDEO_IDS:
        recipe = recipes[video_id]
        audio_path = f"{NATIVE_AUDIO_DIRECTORY}/{video_id}.wav"
        examples.append(
            {
                "video_id": video_id,
                "question": recipe["question"],
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
        )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "panel_id": PANEL_ID,
        "run_id": profile.run_id,
        "conditions": _conditions(profile),
        "canary": {
            "video_id": profile.canary_video_id,
            "condition_id": profile.canary_condition_id,
        },
        "source_inputs": _source_inputs(),
        "examples": examples,
    }
    validate_contract(contract, data_root, verify_files=True, profile=profile)
    return contract


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and (
                key.lower() in FORBIDDEN_FIELD_NAMES or "score" in key.lower()
            ):
                raise ValueError(f"label-bearing field is forbidden: {key}")
            _reject_forbidden_keys(nested)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _validate_path_hash(
    data_root: Path,
    video_id: str,
    field: str,
    relative: Any,
    digest: Any,
    *,
    verify_files: bool,
) -> None:
    if not _is_sha256(digest):
        raise ValueError(f"{video_id}: {field} hash must be a SHA-256")
    path = _resolve_relative(data_root, relative, field)
    if verify_files:
        if not path.is_file():
            raise ValueError(f"{video_id}: missing {field}: {relative}")
        if _sha256_file(path) != digest:
            raise ValueError(f"{video_id}: {field} hash does not match {relative}")


def _validate_shape(
    contract: Mapping[str, Any], profile: Frozen30ExperimentProfile
) -> list[Mapping[str, Any]]:
    _reject_forbidden_keys(contract)
    if set(contract) != {
        "schema_version",
        "panel_id",
        "run_id",
        "conditions",
        "canary",
        "source_inputs",
        "examples",
    }:
        raise ValueError("contract has an invalid schema")
    if (
        type(contract.get("schema_version")) is not int
        or contract["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be the integer {SCHEMA_VERSION}")
    if type(contract.get("panel_id")) is not str or contract["panel_id"] != PANEL_ID:
        raise ValueError(f"panel_id must be the string {PANEL_ID}")
    if (
        type(contract.get("run_id")) is not str
        or contract["run_id"] != profile.run_id
    ):
        raise ValueError(f"run_id must be the string {profile.run_id}")

    conditions = contract.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != len(profile.condition_specs):
        raise ValueError("conditions must be the paired frames/audio conditions")
    for condition, (expected_id, expected_audio) in zip(
        conditions, profile.condition_specs, strict=True
    ):
        if not isinstance(condition, Mapping) or set(condition) != {
            "condition_id",
            "audio_enabled",
        }:
            raise ValueError("condition has an invalid schema")
        if (
            type(condition["condition_id"]) is not str
            or condition["condition_id"] != expected_id
        ):
            raise ValueError("condition_id does not match the frozen panel")
        if (
            type(condition["audio_enabled"]) is not bool
            or condition["audio_enabled"] is not expected_audio
        ):
            raise ValueError("audio_enabled must be the exact frozen boolean")

    canary = contract.get("canary")
    if (
        not isinstance(canary, Mapping)
        or set(canary) != {"video_id", "condition_id"}
        or type(canary["video_id"]) is not str
        or canary["video_id"] != profile.canary_video_id
        or type(canary["condition_id"]) is not str
        or canary["condition_id"] != profile.canary_condition_id
    ):
        raise ValueError("canary must use the approved native-audio row")

    source_inputs = contract.get("source_inputs")
    if not isinstance(source_inputs, list) or len(source_inputs) != 2:
        raise ValueError("source_inputs must use the approved frozen hashes")
    for source, expected in zip(source_inputs, _source_inputs(), strict=True):
        if (
            not isinstance(source, Mapping)
            or set(source) != {"path", "sha256"}
            or type(source["path"]) is not str
            or source["path"] != expected["path"]
            or type(source["sha256"]) is not str
            or source["sha256"] != expected["sha256"]
        ):
            raise ValueError("source_inputs must use the approved frozen hashes")
    examples = contract.get("examples")
    if not isinstance(examples, list) or len(examples) != len(VIDEO_IDS):
        raise ValueError("examples must contain exactly thirty rows")
    if [
        row.get("video_id") if isinstance(row, Mapping) else None for row in examples
    ] != list(VIDEO_IDS):
        raise ValueError("examples must use the fixed video order")
    rows: list[Mapping[str, Any]] = []
    expected_fields = {
        "video_id",
        "question",
        "frame_paths",
        "frame_sha256s",
        "frame_timestamps",
        "audio_path",
        "audio_sha256",
    }
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("example has an invalid schema")
        rows.append(row)
    return rows


def _validate_uploaded_examples(
    rows: list[Mapping[str, Any]], data_root: Path, *, verify_files: bool
) -> None:
    evidence_paths: list[str] = []
    for row in rows:
        video_id = row["video_id"]
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{video_id}: question is required")
        paths, hashes, timestamps = (
            row["frame_paths"],
            row["frame_sha256s"],
            row["frame_timestamps"],
        )
        if not isinstance(paths, list) or len(paths) != 12:
            raise ValueError(f"{video_id}: frame_paths must contain 12 paths")
        if not isinstance(hashes, list) or len(hashes) != 12:
            raise ValueError(f"{video_id}: frame_sha256s must contain 12 hashes")
        if not _is_valid_frame_timestamps(timestamps):
            raise ValueError(
                f"{video_id}: frame_timestamps must contain 12 finite numbers"
            )
        for path, digest in zip(paths, hashes, strict=True):
            _validate_path_hash(
                data_root,
                video_id,
                "frame_paths",
                path,
                digest,
                verify_files=verify_files,
            )
            evidence_paths.append(path)
        expected_audio = f"{NATIVE_AUDIO_DIRECTORY}/{video_id}.wav"
        _resolve_relative(data_root, row["audio_path"], "audio_path")
        if row["audio_path"] != expected_audio:
            raise ValueError(f"{video_id}: audio_path does not match the fixed panel")
        _validate_path_hash(
            data_root,
            video_id,
            "audio_path",
            row["audio_path"],
            row["audio_sha256"],
            verify_files=verify_files,
        )
        evidence_paths.append(row["audio_path"])
    if len(evidence_paths) != 390 or len(set(evidence_paths)) != 390:
        raise ValueError("contract must name exactly 390 unique evidence paths")


def validate_contract(
    contract: Mapping[str, Any],
    data_root: Path,
    *,
    verify_files: bool = True,
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> None:
    """Validate the local contract against the pinned manifest and request sources."""
    data_root = data_root.resolve()
    rows = _validate_shape(contract, profile)
    recipes = _frozen_recipes(data_root)
    for row in rows:
        recipe = recipes[row["video_id"]]
        if row["question"] != recipe["question"]:
            raise ValueError(
                f"{row['video_id']}: frozen request question does not match"
            )
        if (
            row["frame_paths"] != recipe["frame_paths"]
            or row["frame_timestamps"] != recipe["frame_timestamps"]
        ):
            raise ValueError(
                f"{row['video_id']}: frozen request evidence does not match"
            )
    _validate_uploaded_examples(rows, data_root, verify_files=verify_files)


def validate_uploaded_contract(
    contract: Mapping[str, Any],
    data_root: Path,
    *,
    verify_files: bool = True,
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> None:
    """Validate an upload using only the contract and uploaded frames and WAVs."""
    rows = _validate_shape(contract, profile)
    _validate_uploaded_examples(rows, data_root.resolve(), verify_files=verify_files)


def contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return the stable digest that binds the paired run to this input contract."""
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=PROFILES, default="a4")
    args = parser.parse_args()
    contract = build_contract(args.data_root, profile_for_key(args.experiment))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(contract_sha256(contract))


if __name__ == "__main__":
    main()
