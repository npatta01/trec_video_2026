#!/usr/bin/env python3
"""Build the label-free VQA factorial manifests and validate its profiles.

The validator is intentionally kept in this script so the dstack shell entrypoint
can fail closed before installing packages or loading any model resources.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_INPUT_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
    "experiments/qwen35-4b-12f-full-20260803-b1-inputs/inputs"
)
EXPECTED_EVIDENCE_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
    "experiments/vqa-factorial-full-evidence-20260809-f1"
)
EXPECTED_RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
WHISPER_MODEL_ID = "openai/whisper-large-v3"
WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"

MODEL_REVISIONS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
}
ANSWER_POLICIES = {"current", "question-aware"}
REQUEST_SOURCES = {
    "shared-accepted-whisper",
    "shared-accepted-whisper-ocr-only",
}
BASE_BOUNDS = (65_536, 262_144)
HIGH_RES_BOUNDS = (65_536, 589_824)
CANARY_IDS = {
    "CtinYSH3nVw",
    "LKiaCf1QZFw",
    "IVqpg_1aWXU",
}
CANARY_CONDITION_IDS = {
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
    "8b-question-aware-ocr-highres",
    "4b-current-repeat",
}
FULL_CONDITION_IDS = {
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
}
CONDITION_FIELDS = {
    "condition_id",
    "model_id",
    "revision",
    "answer_policy",
    "min_pixels",
    "max_pixels",
    "request_source",
}
COMMON_PROFILE_FIELDS = {
    "run_name",
    "input_uri",
    "remote_uri",
    "manifest",
    "expected_count",
    "frame_count",
}
CANARY_PROFILE_FIELDS = COMMON_PROFILE_FIELDS | {"conditions", "repeat_condition"}
FULL_CONDITION_PROFILE_FIELDS = COMMON_PROFILE_FIELDS | {"conditions"}
SOURCE_FIELDS = {"schema_version", "manifest_version", "label_policy", "example_count", "examples"}
LABEL_KEY_PATTERN = re.compile(
    r"(?:answer|reference|ground[-_ ]?truth|annotation|label|expected|correct)",
    re.IGNORECASE,
)
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")
SAFE_URI = re.compile(r"^hf://buckets/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")


def _fail(message: str) -> None:
    raise ValueError(message)


def reject_label_like_keys(value: Any, location: str = "$") -> None:
    """Reject answer/reference/label-bearing keys recursively."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if LABEL_KEY_PATTERN.search(key_text):
                _fail(f"label-bearing key {key_text!r} at {location}")
            reject_label_like_keys(child, f"{location}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_label_like_keys(child, f"{location}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"could not read JSON {path}: {error}")
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


def _validate_source(payload: Mapping[str, Any], expected_count: int) -> list[dict[str, str]]:
    if set(payload) != SOURCE_FIELDS:
        _fail(f"source fields must be exactly {sorted(SOURCE_FIELDS)}")
    if payload.get("schema_version") != 1:
        _fail("source schema_version must be 1")
    if payload.get("label_policy") != "label-free-inference-input":
        _fail("source label_policy must be label-free-inference-input")
    examples = payload.get("examples")
    if payload.get("example_count") != expected_count or not isinstance(examples, list):
        _fail(f"source must contain expected {expected_count} rows")
    if len(examples) != expected_count:
        _fail(f"source must contain expected {expected_count} rows")
    ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, row in enumerate(examples):
        if isinstance(row, dict):
            reject_label_like_keys(row, f"$.examples[{index}]")
        if not isinstance(row, dict) or set(row) != {"video_id", "question"}:
            _fail(f"source row {index} must contain exactly video_id and question")
        video_id = row["video_id"]
        question = row["question"]
        if not isinstance(video_id, str) or not video_id:
            _fail(f"source row {index} video_id must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            _fail(f"source row {index} question must be a non-empty string")
        if video_id in ids:
            _fail(f"duplicate video_id in source: {video_id}")
        ids.add(video_id)
        normalized.append({"video_id": video_id, "question": question})
    return normalized


def build_canary_manifest(
    source_path: Path, output_path: Path, *, expected_source_count: int
) -> dict[str, Any]:
    """Select the frozen canary IDs in a stable, question-only order."""

    source = _load_json(source_path)
    rows = _validate_source(source, expected_source_count)
    by_id = {row["video_id"]: row for row in rows}
    if set(by_id) < CANARY_IDS:
        _fail(f"source is missing frozen canary IDs: {sorted(CANARY_IDS - set(by_id))}")
    selected = [by_id[video_id] for video_id in ("CtinYSH3nVw", "LKiaCf1QZFw", "IVqpg_1aWXU")]
    payload = {
        "schema_version": 1,
        "manifest_version": "trecvid-vqa-factorial-canary-2026-08-09-v1",
        "label_policy": "label-free-inference-input",
        "example_count": len(selected),
        "examples": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def validate_manifest(path: Path, *, expected_count: int) -> list[dict[str, str]]:
    """Validate a frozen label-free manifest and return its identity rows."""

    return _validate_source(_load_json(path), expected_count)


def _require_fields(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        _fail(f"{label} fields must be exactly {sorted(expected)}")


def _validate_run_name(value: Any, field: str = "run_name") -> str:
    if not isinstance(value, str) or not SAFE_RUN_NAME.fullmatch(value):
        _fail(f"{field} is not a safe run identity")
    return value


def _validate_uri(value: Any, field: str) -> str:
    if not isinstance(value, str) or ".." in value or not SAFE_URI.fullmatch(value):
        _fail(f"{field} is not a safe HF bucket URI")
    return value


def _validate_profile_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("/dstack/run/"):
        _fail(f"{field} must be an absolute path below /dstack/run")
    path = PurePosixPath(value)
    if path.is_absolute() is False or ".." in path.parts or any(part == "" for part in path.parts):
        _fail(f"{field} contains an unsafe path")
    if path.suffix.lower() != ".json":
        _fail(f"{field} must point to a JSON file")
    return value


def _validate_common_profile(payload: Mapping[str, Any], *, expected_count: int, input_uri: str) -> None:
    if not COMMON_PROFILE_FIELDS <= set(payload):
        _fail(f"profile is missing common fields: {sorted(COMMON_PROFILE_FIELDS - set(payload))}")
    _validate_run_name(payload["run_name"])
    if payload["input_uri"] != input_uri:
        _fail(f"input_uri must be the pinned URI {input_uri}")
    _validate_uri(payload["input_uri"], "input_uri")
    _validate_uri(payload["remote_uri"], "remote_uri")
    _validate_profile_path(payload["manifest"], "manifest")
    if payload["expected_count"] != expected_count:
        _fail(f"expected_count must be {expected_count}")
    if payload["frame_count"] != 12:
        _fail("frame_count must be 12")


def _validate_condition(condition: Mapping[str, Any], *, bounds: tuple[int, int] | None = None) -> None:
    _require_fields(condition, CONDITION_FIELDS, "condition")
    condition_id = condition["condition_id"]
    if not isinstance(condition_id, str) or not SAFE_RUN_NAME.fullmatch(condition_id):
        _fail("condition_id is not a safe identity")
    model_id = condition["model_id"]
    if model_id not in MODEL_REVISIONS:
        _fail(f"model_id is not in the exact allowlist: {model_id!r}")
    if condition["revision"] != MODEL_REVISIONS[model_id]:
        _fail(f"revision is not pinned for {model_id}")
    if condition["answer_policy"] not in ANSWER_POLICIES:
        _fail("answer_policy is outside the closed vocabulary")
    if condition["request_source"] not in REQUEST_SOURCES:
        _fail("request_source is outside the closed vocabulary")
    lower = condition["min_pixels"]
    upper = condition["max_pixels"]
    if isinstance(lower, bool) or not isinstance(lower, int) or lower <= 0:
        _fail("min_pixels must be a positive integer")
    if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
        _fail("max_pixels must be a positive integer")
    if lower > upper:
        _fail("pixel bounds must not be inverted")
    if bounds is not None and (lower, upper) != bounds:
        _fail(f"condition must use pixel bounds {bounds[0]}..{bounds[1]}")


def validate_canary_profile(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, CANARY_PROFILE_FIELDS, "canary profile")
    _validate_common_profile(payload, expected_count=3, input_uri=EXPECTED_INPUT_URI)
    conditions = payload["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 6:
        _fail("canary must declare exactly six condition invocations")
    ids: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            _fail("canary conditions must be objects")
        _validate_condition(condition)
        ids.append(condition["condition_id"])
    if len(set(ids)) != len(ids):
        _fail("canary condition IDs must be unique")
    if set(ids) != CANARY_CONDITION_IDS:
        _fail(f"canary condition IDs must be {sorted(CANARY_CONDITION_IDS)}")
    by_id = {condition["condition_id"]: condition for condition in conditions}
    for condition_id in CANARY_CONDITION_IDS - {"8b-question-aware-ocr-highres"}:
        _validate_condition(by_id[condition_id], bounds=BASE_BOUNDS)
    highres = by_id["8b-question-aware-ocr-highres"]
    _validate_condition(highres, bounds=HIGH_RES_BOUNDS)
    if highres["model_id"] != "Qwen/Qwen3-VL-8B-Instruct" or highres["answer_policy"] != "question-aware":
        _fail("high-resolution canary must be the 8B question-aware cell")
    if highres["request_source"] != "shared-accepted-whisper-ocr-only":
        _fail("high-resolution canary must be OCR-only")
    repeat_id = payload["repeat_condition"]
    if not isinstance(repeat_id, str) or repeat_id not in by_id:
        _fail("repeat_condition must name a declared canary condition")
    repeat = by_id[repeat_id]
    base = by_id["4b-current"]
    if repeat_id != "4b-current-repeat" or {
        key: repeat[key] for key in CONDITION_FIELDS if key != "condition_id"
    } != {
        key: base[key] for key in CONDITION_FIELDS if key != "condition_id"
    }:
        _fail("repeat condition must be a distinct identity of 4b-current")
    if repeat_id == "4b-current":
        _fail("repeat condition cannot overwrite its original condition identity")


def validate_full_evidence_profile(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, COMMON_PROFILE_FIELDS, "full evidence profile")
    _validate_common_profile(payload, expected_count=475, input_uri=EXPECTED_INPUT_URI)
    if payload["run_name"] != "vqa-factorial-full-evidence-20260809-f1":
        _fail("evidence run_name is not the pinned identity")


def validate_full_condition_profiles(payload: Mapping[str, Any]) -> None:
    _require_fields(payload, FULL_CONDITION_PROFILE_FIELDS, "full condition profile")
    _validate_common_profile(payload, expected_count=475, input_uri=EXPECTED_EVIDENCE_URI)
    conditions = payload["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 4:
        _fail("full condition profile must declare exactly four conditions")
    ids: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            _fail("full conditions must be objects")
        _validate_condition(condition, bounds=BASE_BOUNDS)
        ids.append(condition["condition_id"])
    if set(ids) != FULL_CONDITION_IDS or len(ids) != len(set(ids)):
        _fail("full condition IDs must be the four unique model/policy cells")
    derived_outputs = {payload["remote_uri"].rstrip("/") + "/" + identifier for identifier in ids}
    if len(derived_outputs) != 4:
        _fail("full condition output identities must be unique")


def validate_profile(
    path: Path,
    kind: str,
    *,
    condition_id: str | None = None,
    expected_run_name: str | None = None,
    expected_input_uri: str | None = None,
    expected_remote_uri: str | None = None,
    expected_manifest: str | None = None,
) -> dict[str, Any]:
    payload = _load_json(path)
    if kind == "canary":
        validate_canary_profile(payload)
        selected = payload
        if condition_id is not None:
            by_id = {item["condition_id"]: item for item in payload["conditions"]}
            if condition_id not in by_id:
                _fail(f"unknown condition_id: {condition_id}")
            selected = {**payload, "condition": by_id[condition_id]}
    elif kind == "evidence":
        validate_full_evidence_profile(payload)
        selected = payload
    elif kind == "condition":
        validate_full_condition_profiles(payload)
        if not condition_id:
            _fail("condition profile validation requires condition_id")
        by_id = {item["condition_id"]: item for item in payload["conditions"]}
        if condition_id not in by_id:
            _fail(f"unknown condition_id: {condition_id}")
        selected = {
            **payload,
            "run_name": f"{payload['run_name']}-{condition_id}",
            "remote_uri": payload["remote_uri"].rstrip("/") + "/" + condition_id,
            "condition": by_id[condition_id],
        }
    else:
        _fail(f"unknown profile kind: {kind}")
    expected = {
        "run_name": expected_run_name,
        "input_uri": expected_input_uri,
        "remote_uri": expected_remote_uri,
        "manifest": expected_manifest,
    }
    for field, value in expected.items():
        if value is not None and selected[field] != value:
            _fail(f"{field} does not match the validated profile")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--canary-output", type=Path)
    parser.add_argument("--expected-source-count", type=int)
    parser.add_argument("--validate-profile", type=Path)
    parser.add_argument("--validate-manifest", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--profile-kind", choices=("canary", "evidence", "condition"))
    parser.add_argument("--condition-id")
    parser.add_argument("--expected-run-name")
    parser.add_argument("--expected-input-uri")
    parser.add_argument("--expected-remote-uri")
    parser.add_argument("--expected-manifest")
    args = parser.parse_args()
    if args.source is not None:
        if args.canary_output is None or args.expected_source_count is None:
            parser.error("--source requires --canary-output and --expected-source-count")
        if args.validate_profile is not None or args.validate_manifest is not None:
            parser.error("manifest generation and validation modes are mutually exclusive")
    elif args.validate_manifest is not None:
        if args.expected_count is None:
            parser.error("--validate-manifest requires --expected-count")
        if args.validate_profile is not None:
            parser.error("manifest and profile validation are mutually exclusive")
    elif args.validate_profile is not None:
        if args.profile_kind is None:
            parser.error("--validate-profile requires --profile-kind")
    else:
        parser.error("provide --source or --validate-profile")
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.source is not None:
            build_canary_manifest(
                args.source, args.canary_output, expected_source_count=args.expected_source_count
            )
        elif args.validate_manifest is not None:
            validate_manifest(args.validate_manifest, expected_count=args.expected_count)
            print(json.dumps({"manifest_valid": True, "example_count": args.expected_count}))
        else:
            selected = validate_profile(
                args.validate_profile,
                args.profile_kind,
                condition_id=args.condition_id,
                expected_run_name=args.expected_run_name,
                expected_input_uri=args.expected_input_uri,
                expected_remote_uri=args.expected_remote_uri,
                expected_manifest=args.expected_manifest,
            )
            summary = {
                "profile_valid": True,
                **{
                    key: selected[key]
                    for key in (
                        "run_name",
                        "input_uri",
                        "remote_uri",
                        "manifest",
                        "expected_count",
                        "frame_count",
                    )
                },
            }
            if "condition" in selected:
                summary["condition"] = selected["condition"]
            print(json.dumps(summary, sort_keys=True))
    except ValueError as error:
        print(f"error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
