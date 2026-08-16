"""Immutable four-arm host harness for the Qwen empty-output diagnostic.

This module deliberately contains no model code.  It selects a label-free
failure/control panel from the frozen Stage 0.5d evidence, constructs one
offline Podman command per arm, preserves one terminal attempt per arm, and
freezes enough hashes to make later mutation observable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "qwen-empty-output-diagnostic.v1"
EXPERIMENT_ID = "qwen-aligner-stage05e-20260816-empty-output-v1"
CANONICAL_DATA_ROOT = Path(
    "/home/npatta01/data/competitions/trec_video_2026/data"
).resolve()
PRIVATE_ROOT = CANONICAL_DATA_ROOT / (
    "vqa/benchmarks/model-verifier-canaries-20260814/"
    "stage05e-qwen-empty-output-diagnostic-v1"
)
EXPECTED_SOURCE_ROOT = CANONICAL_DATA_ROOT / (
    "vqa/benchmarks/model-verifier-canaries-20260814/"
    "stage05d-qwen-aligner-quality-v1"
)
EXPECTED_SOURCE_EXPERIMENT_ID = "qwen-aligner-stage05d-20260816-quality-v1"

ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
RUNTIME_IMAGE = (
    "localhost/trec-vqa-audio-rocm@sha256:"
    "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
)
CONDITION_ID = "qwen3-asr-1.7b-forced-aligner-0.6b"
DIAGNOSTIC_RUNNER_PATH = Path(__file__).with_name(
    "vqa_qwen_empty_output_runner.py"
).resolve()
HELPER_SOURCE_PATH = Path(__file__).with_name("vqa_audio_runner.py").resolve()

ARMS: dict[str, dict[str, Any]] = {
    "aligned_auto": {
        "aligner_loaded": True,
        "timestamps": "on",
        "language": "auto",
        "purpose": "baseline_reproduction_control",
    },
    "base_auto": {
        "aligner_loaded": False,
        "timestamps": "off",
        "language": "auto",
        "purpose": "aligner_load_localization",
    },
    "aligner_plain_auto": {
        "aligner_loaded": True,
        "timestamps": "off",
        "language": "auto",
        "purpose": "timestamp_path_localization",
    },
    "aligned_english": {
        "aligner_loaded": True,
        "timestamps": "on",
        "language": "English",
        "purpose": "language_selection_treatment",
    },
}
ARM_ORDER = tuple(ARMS)
EMPTY_ERROR_TYPE = "ValueError"
EMPTY_ERROR_MESSAGE = "Qwen3-ASR returned empty text"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_PATTERNS = (
    r"hf_[A-Za-z0-9]{16,}",
    r"Bearer\s+[A-Za-z0-9._~+/-]{16,}",
    r"AIza[0-9A-Za-z_-]{20,}",
    r"sk-[A-Za-z0-9_-]{20,}",
)
_FORBIDDEN_KEY_TOKENS = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "gold",
    "label",
    "labels",
    "map",
    "metric",
    "metrics",
    "question",
    "reference",
    "references",
    "transcript",
    "transcripts",
    "whisper",
    "grade",
    "grades",
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the diagnostic's canonical encoding."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot read source file: {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return payload


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing artifact: {path}") from exc


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(value)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing artifact: {path}") from exc


def _freeze_file(path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "artifact_sha256": _sha256_file(path),
    }


def _verify_file_freeze(path: Path, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"invalid freeze schema for {path.name}")
    if record.get("status") != "frozen":
        raise ValueError(f"invalid freeze status for {path.name}")
    expected = record.get("artifact_sha256")
    if not _is_sha256(expected) or expected != _sha256_file(path):
        raise ValueError(f"artifact hash does not match freeze: {path.name}")


def _freeze_json(path: Path, *, bindings: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_object(path)
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"artifact {key} does not match binding")
    result = _freeze_file(path)
    result["bindings"] = dict(bindings)
    result["bindings_sha256"] = canonical_sha256(dict(bindings))
    return result


def _verify_json_freeze(path: Path, record: Mapping[str, Any]) -> None:
    _verify_file_freeze(path, record)
    bindings = record.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError(f"missing bindings for {path.name}")
    if record.get("bindings_sha256") != canonical_sha256(dict(bindings)):
        raise ValueError(f"binding hash does not match freeze: {path.name}")
    payload = _read_object(path)
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"live binding does not match freeze: {path.name}")


def _ensure_inside(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ValueError(f"{label} must be inside the canonical data root")
    return resolved


def _walk_forbidden_keys(value: Any, *, allow_null_values: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact keys must be strings")
            normalized = re.sub(r"[^a-z]", "", key.casefold())
            tokens = {
                token.casefold()
                for token in re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", key)
                if token
            }
            if (
                (normalized in _FORBIDDEN_KEY_TOKENS or tokens & _FORBIDDEN_KEY_TOKENS)
                and not (allow_null_values and child is None)
            ):
                raise ValueError("label-bearing fields are forbidden")
            _walk_forbidden_keys(child, allow_null_values=allow_null_values)
    elif isinstance(value, list):
        for child in value:
            _walk_forbidden_keys(child, allow_null_values=allow_null_values)


def _source_artifact(path: Path, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") not in {1, SCHEMA_VERSION}:
        raise ValueError(f"invalid source freeze schema for {path.name}")
    expected_hash = record.get("artifact_sha256")
    if not _is_sha256(expected_hash) or expected_hash != _sha256_file(path):
        raise ValueError(f"source artifact hash does not match freeze: {path.name}")
    if "bindings" in record:
        bindings = record.get("bindings")
        if not isinstance(bindings, Mapping) or record.get("bindings_sha256") != canonical_sha256(dict(bindings)):
            raise ValueError(f"source binding freeze is invalid: {path.name}")
        payload = _read_object(path)
        for key, expected in bindings.items():
            if payload.get(key) != expected:
                raise ValueError(f"source binding does not match: {path.name}")


def _verify_source_freeze(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prediction_path = source_root / "predictions-freeze.json"
    input_path = source_root / "inputs-freeze.json"
    prediction = _read_object(prediction_path)
    if prediction.get("experiment_id") != EXPECTED_SOURCE_EXPERIMENT_ID:
        raise ValueError("Stage 0.5d source experiment identity is invalid")
    if prediction.get("status") != "predictions_frozen":
        raise ValueError("Stage 0.5d source is not predictions_frozen")
    if prediction.get("decision") != "not_ready":
        raise ValueError("Stage 0.5d source must be not_ready")
    if prediction.get("terminal_status") != "completed":
        raise ValueError("Stage 0.5d source terminal is not completed")
    input_hash = prediction.get("input_freeze_sha256")
    if not _is_sha256(input_hash) or input_hash != _sha256_file(input_path):
        raise ValueError("Stage 0.5d input freeze hash is invalid")
    inputs = _read_object(input_path)
    if inputs.get("status") != "inputs_frozen":
        raise ValueError("Stage 0.5d inputs are not frozen")
    input_artifacts = inputs.get("artifacts")
    if not isinstance(input_artifacts, Mapping):
        raise ValueError("Stage 0.5d input artifacts are missing")
    if inputs.get("artifacts_sha256") != canonical_sha256(dict(input_artifacts)):
        raise ValueError("Stage 0.5d input artifact hash is invalid")
    for name, filename in (("requests", "requests.json"), ("protocol", "protocol.json")):
        record = input_artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage 0.5d input artifact {name} is missing")
        _source_artifact(source_root / filename, record)

    prediction_artifacts = prediction.get("artifacts")
    if not isinstance(prediction_artifacts, Mapping):
        raise ValueError("Stage 0.5d prediction artifacts are missing")
    if prediction.get("artifacts_sha256") != canonical_sha256(dict(prediction_artifacts)):
        raise ValueError("Stage 0.5d prediction artifact hash is invalid")
    for name, filename in (
        ("raw-results", "raw-results.json"),
        ("terminal", "terminal.json"),
        ("validated-results", "validated-results.json"),
    ):
        record = prediction_artifacts.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Stage 0.5d prediction artifact {name} is missing")
        _source_artifact(source_root / filename, record)

    requests = _read_object(source_root / "requests.json")
    protocol = _read_object(source_root / "protocol.json")
    validation = _read_object(source_root / "validated-results.json")
    terminal = _read_object(source_root / "terminal.json")
    _walk_forbidden_keys(requests)
    if protocol.get("experiment_id") != EXPECTED_SOURCE_EXPERIMENT_ID:
        raise ValueError("Stage 0.5d protocol experiment identity is invalid")
    if requests.get("experiment_id") != EXPECTED_SOURCE_EXPERIMENT_ID:
        raise ValueError("Stage 0.5d requests experiment identity is invalid")
    if protocol.get("condition_id") != CONDITION_ID:
        raise ValueError("Stage 0.5d condition identity is invalid")
    if protocol.get("runner_sha256") != _sha256_file(HELPER_SOURCE_PATH):
        raise ValueError("Stage 0.5d runner source hash does not match helper source")
    if terminal.get("status") != "completed" or terminal.get("exit_code") != 0:
        raise ValueError("Stage 0.5d terminal is not a successful completed run")
    if validation.get("status") != "not_ready":
        raise ValueError("Stage 0.5d validation must be not_ready")

    request_rows = requests.get("requests")
    raw_results = _read_json(source_root / "raw-results.json")
    _walk_forbidden_keys(raw_results, allow_null_values=True)
    if not isinstance(request_rows, list) or not request_rows:
        raise ValueError("Stage 0.5d requests are missing")
    if not isinstance(raw_results, list) or len(raw_results) != len(request_rows):
        raise ValueError("Stage 0.5d result count does not match requests")
    if protocol.get("row_order") != [row.get("video_id") for row in request_rows]:
        raise ValueError("Stage 0.5d request order does not match protocol")
    if protocol.get("row_count") != len(request_rows):
        raise ValueError("Stage 0.5d row count is invalid")
    if validation.get("required_count") != len(request_rows):
        raise ValueError("Stage 0.5d validation denominator is invalid")

    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    audio_hashes = protocol.get("audio_sha256")
    durations = protocol.get("duration_seconds")
    if not isinstance(audio_hashes, Mapping) or not isinstance(durations, Mapping):
        raise ValueError("Stage 0.5d audio bindings are missing")
    for index, (request, result) in enumerate(zip(request_rows, raw_results, strict=True)):
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise ValueError("Stage 0.5d request/result rows must be objects")
        video_id = request.get("video_id")
        if not isinstance(video_id, str) or result.get("video_id") != video_id:
            raise ValueError("Stage 0.5d request/result identity mismatch")
        if (
            request.get("run_id") != EXPECTED_SOURCE_EXPERIMENT_ID
            or request.get("condition_id") != CONDITION_ID
            or request.get("mode") != "transcription"
            or result.get("run_id") != EXPECTED_SOURCE_EXPERIMENT_ID
            or result.get("condition_id") != CONDITION_ID
            or result.get("mode") != "transcription"
        ):
            raise ValueError("Stage 0.5d row identity is invalid")
        audio_path_value = request.get("audio_path")
        if not isinstance(audio_path_value, str):
            raise ValueError("Stage 0.5d audio path is missing")
        audio_path = _ensure_inside(Path(audio_path_value), CANONICAL_DATA_ROOT, label="audio path")
        if not audio_path.is_file():
            raise ValueError("selected Stage 0.5d audio is missing")
        expected_hash = audio_hashes.get(video_id)
        if not _is_sha256(expected_hash) or expected_hash != _sha256_file(audio_path):
            raise ValueError("Stage 0.5d audio hash does not match source")
        duration = durations.get(video_id)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("Stage 0.5d duration is invalid")
        if result.get("source_audio_sha256") != expected_hash:
            raise ValueError("Stage 0.5d result audio hash does not match source")
        metadata = {
            "source_request_index": index,
            "video_id": video_id,
            "audio_path": str(audio_path),
            "audio_sha256": expected_hash,
            "duration_seconds": float(duration),
        }
        if result.get("status") == "failed":
            if result.get("error_type") == EMPTY_ERROR_TYPE and result.get("error_message") == EMPTY_ERROR_MESSAGE:
                failures.append(metadata)
        elif result.get("status") == "passed":
            # Deliberately inspect only whether text is nonempty; never copy it.
            chunks = result.get("chunks")
            previous_end = 0.0
            valid_chunks = isinstance(chunks, list) and bool(chunks)
            if valid_chunks:
                for chunk in chunks:
                    if not isinstance(chunk, Mapping):
                        valid_chunks = False
                        break
                    try:
                        start = _number(chunk.get("start"), "source chunk start")
                        end = _number(chunk.get("end"), "source chunk end")
                    except ValueError:
                        valid_chunks = False
                        break
                    if (
                        start < 0
                        or end < start
                        or end > float(duration)
                        or start < previous_end
                        or not isinstance(chunk.get("text"), str)
                        or not chunk["text"].strip()
                    ):
                        valid_chunks = False
                        break
                    previous_end = end
            if (
                isinstance(result.get("text"), str)
                and bool(result["text"].strip())
                and isinstance(result.get("language"), str)
                and bool(result["language"].strip())
                and valid_chunks
            ):
                successes.append(metadata)
    if len(failures) != 2:
        raise ValueError("Stage 0.5d source must contain exactly two empty-output failures")
    if len(successes) < 2:
        raise ValueError("Stage 0.5d source must contain at least two passing controls")
    selected = sorted(
        [*failures, *successes[:2]], key=lambda row: row["source_request_index"]
    )
    for row in selected:
        row["role"] = "failure" if row in failures else "control"
    return prediction, protocol, selected, requests


def _verify_live_bindings(root: Path) -> dict[str, Any]:
    protocol = _read_object(root / "protocol.json")
    runner_path_value = protocol.get("diagnostic_runner_path")
    if not isinstance(runner_path_value, str):
        raise ValueError("diagnostic runner path is missing")
    runner_path = Path(runner_path_value)
    if runner_path.resolve() != DIAGNOSTIC_RUNNER_PATH.resolve():
        raise ValueError("diagnostic runner path is not the executed module")
    if not runner_path.is_file():
        raise ValueError("diagnostic runner source is missing")
    if protocol.get("diagnostic_runner_sha256") != _sha256_file(runner_path):
        raise ValueError("diagnostic runner source hash does not match protocol")
    helper_path_value = protocol.get("helper_source_path")
    if not isinstance(helper_path_value, str):
        raise ValueError("helper source path is missing")
    helper_path = Path(helper_path_value)
    if helper_path.resolve() != HELPER_SOURCE_PATH.resolve():
        raise ValueError("helper source path is not the imported helper")
    if protocol.get("helper_source_sha256") != _sha256_file(helper_path):
        raise ValueError("helper source hash does not match protocol")
    return protocol


def _verify_inputs(root: Path) -> dict[str, Any]:
    if Path(root).resolve() != PRIVATE_ROOT.resolve():
        raise ValueError("diagnostic root is not the declared private root")
    protocol = _verify_live_bindings(root)
    source_root_value = protocol.get("source_root")
    if not isinstance(source_root_value, str):
        raise ValueError("source root is missing")
    source_root = Path(source_root_value)
    if source_root.resolve() != EXPECTED_SOURCE_ROOT.resolve():
        raise ValueError("source root is not the declared Stage 0.5d root")
    _ensure_inside(source_root, CANONICAL_DATA_ROOT, label="source root")
    prediction, source_protocol, selected, _requests = _verify_source_freeze(source_root)
    if protocol.get("source_prediction_freeze_sha256") != _sha256_file(source_root / "predictions-freeze.json"):
        raise ValueError("source prediction freeze hash does not match protocol")
    if protocol.get("source_freeze_sha256") != protocol.get("source_prediction_freeze_sha256"):
        raise ValueError("source freeze alias does not match protocol")
    if protocol.get("source_runner_sha256") != source_protocol.get("runner_sha256"):
        raise ValueError("source runner evidence does not match protocol")
    if prediction.get("decision") != "not_ready":
        raise ValueError("source prediction decision changed")
    selection = _read_object(root / "selection.json")
    if selection.get("experiment_id") != EXPERIMENT_ID or selection.get("source_root") != str(source_root.resolve()):
        raise ValueError("diagnostic selection identity is invalid")
    if selection.get("source_prediction_freeze_sha256") != protocol.get("source_prediction_freeze_sha256"):
        raise ValueError("diagnostic selection source binding is invalid")
    if selection.get("rows") != selected:
        raise ValueError("diagnostic selection is not the deterministic source selection")
    if protocol.get("row_order") != [row["video_id"] for row in selected]:
        raise ValueError("diagnostic protocol row order is invalid")
    if (
        protocol.get("condition_id") != CONDITION_ID
        or protocol.get("row_count") != 4
        or protocol.get("asr_model_id") != ASR_MODEL_ID
        or protocol.get("asr_revision") != ASR_REVISION
        or protocol.get("forced_aligner_model_id") != ALIGNER_MODEL_ID
        or protocol.get("forced_aligner_revision") != ALIGNER_REVISION
        or protocol.get("runtime_image") != RUNTIME_IMAGE
        or protocol.get("one_attempt_only") is not True
        or protocol.get("paid_compute_usd") != 0.0
        or protocol.get("arms") != ARMS
    ):
        raise ValueError("diagnostic protocol contract is invalid")
    input_freeze = _read_object(root / "inputs-freeze.json")
    if input_freeze.get("status") != "inputs_frozen":
        raise ValueError("diagnostic inputs are not frozen")
    artifacts = input_freeze.get("artifacts")
    if not isinstance(artifacts, Mapping) or input_freeze.get("artifacts_sha256") != canonical_sha256(dict(artifacts)):
        raise ValueError("diagnostic input artifact hash is invalid")
    _verify_json_freeze(root / "selection.json", artifacts.get("selection", {}))
    _verify_json_freeze(root / "protocol.json", artifacts.get("protocol", {}))
    request_artifacts = artifacts.get("requests")
    if not isinstance(request_artifacts, Mapping) or set(request_artifacts) != set(ARM_ORDER):
        raise ValueError("diagnostic request freeze is invalid")
    for arm_id in ARM_ORDER:
        _verify_json_freeze(root / "requests" / f"{arm_id}.json", request_artifacts[arm_id])
        requests = _read_object(root / "requests" / f"{arm_id}.json")
        _validate_prepared_requests(protocol, requests, arm_id)
    return protocol


def prepare_diagnostic(root: Path, source_root: Path, *, runner_path: Path) -> dict[str, Any]:
    """Select and exclusively freeze the four-row, four-arm input contract."""

    root = Path(root)
    source_root = Path(source_root)
    if root.resolve() != PRIVATE_ROOT.resolve():
        raise ValueError("diagnostic root must equal the declared private root")
    if source_root.resolve() != EXPECTED_SOURCE_ROOT.resolve():
        raise ValueError("source root must equal the declared Stage 0.5d root")
    _ensure_inside(root, CANONICAL_DATA_ROOT, label="diagnostic root")
    _ensure_inside(source_root, CANONICAL_DATA_ROOT, label="source root")
    if root.exists() and any(root.iterdir()):
        raise ValueError("diagnostic root must be new and empty")
    if not runner_path.is_file():
        raise ValueError("diagnostic runner source is missing")
    runner_path = runner_path.resolve()
    if runner_path.resolve() != DIAGNOSTIC_RUNNER_PATH.resolve():
        raise ValueError("runner path must equal the executed dedicated Qwen runner")
    if root.exists() and not root.is_dir():
        raise ValueError("diagnostic root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    prediction, source_protocol, selected, _source_requests = _verify_source_freeze(source_root)
    runner_sha = _sha256_file(runner_path)
    helper_sha = _sha256_file(HELPER_SOURCE_PATH)
    if source_protocol.get("runner_sha256") != helper_sha:
        raise ValueError("source runner evidence does not match imported helper")

    selection = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_root": str(source_root.resolve()),
        "source_prediction_freeze_sha256": _sha256_file(source_root / "predictions-freeze.json"),
        "rows": selected,
        "selection_policy": "both exact empty failures plus first two passing rows in frozen request order",
    }
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "condition_id": CONDITION_ID,
        "row_count": 4,
        "row_order": [row["video_id"] for row in selected],
        "audio_path": {row["video_id"]: row["audio_path"] for row in selected},
        "audio_sha256": {row["video_id"]: row["audio_sha256"] for row in selected},
        "duration_seconds": {row["video_id"]: row["duration_seconds"] for row in selected},
        "source_root": str(source_root.resolve()),
        "source_prediction_freeze_sha256": _sha256_file(source_root / "predictions-freeze.json"),
        "source_freeze_sha256": _sha256_file(source_root / "predictions-freeze.json"),
        "source_runner_path": str(HELPER_SOURCE_PATH),
        "source_runner_sha256": source_protocol["runner_sha256"],
        "diagnostic_runner_path": str(runner_path),
        "diagnostic_runner_sha256": runner_sha,
        "helper_source_path": str(HELPER_SOURCE_PATH),
        "helper_source_sha256": helper_sha,
        "asr_model_id": ASR_MODEL_ID,
        "asr_revision": ASR_REVISION,
        "forced_aligner_model_id": ALIGNER_MODEL_ID,
        "forced_aligner_revision": ALIGNER_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "one_attempt_only": True,
        "paid_compute_usd": 0.0,
        "arms": {arm_id: dict(config) for arm_id, config in ARMS.items()},
    }
    _write_json_exclusive(root / "selection.json", selection)
    _write_json_exclusive(root / "protocol.json", protocol)
    request_freezes: dict[str, Any] = {}
    for arm_id in ARM_ORDER:
        requests = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "requests": [
                {
                    "run_id": EXPERIMENT_ID,
                    "condition_id": CONDITION_ID,
                    "video_id": row["video_id"],
                    "audio_path": row["audio_path"],
                    "mode": "transcription",
                }
                for row in selected
            ],
        }
        request_path = root / "requests" / f"{arm_id}.json"
        _write_json_exclusive(request_path, requests)
        request_freezes[arm_id] = _freeze_json(
            request_path, bindings={"experiment_id": EXPERIMENT_ID, "arm_id": arm_id}
        )
    input_artifacts = {
        "selection": _freeze_json(root / "selection.json", bindings={"experiment_id": EXPERIMENT_ID}),
        "protocol": _freeze_json(root / "protocol.json", bindings={"experiment_id": EXPERIMENT_ID}),
        "requests": request_freezes,
    }
    input_freeze = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "artifacts": input_artifacts,
        "artifacts_sha256": canonical_sha256(input_artifacts),
    }
    _write_json_exclusive(root / "inputs-freeze.json", input_freeze)
    _verify_inputs(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "inputs_frozen",
        "row_count": 4,
        "arm_count": 4,
        "source_prediction_freeze_sha256": protocol["source_prediction_freeze_sha256"],
        "diagnostic_runner_sha256": runner_sha,
        "helper_source_sha256": helper_sha,
    }


def _validate_arm_id(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise ValueError(f"unknown diagnostic arm: {arm_id}")
    return ARMS[arm_id]


def _validate_prepared_requests(
    protocol: Mapping[str, Any], requests: Mapping[str, Any], arm_id: str
) -> list[dict[str, Any]]:
    """Validate the semantic request identity in addition to its byte freeze."""

    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("diagnostic protocol experiment identity is invalid")
    if requests.get("experiment_id") != EXPERIMENT_ID or requests.get("arm_id") != arm_id:
        raise ValueError("diagnostic request identity is invalid")
    rows = requests.get("requests")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("diagnostic request count is invalid")
    if protocol.get("row_order") != [row.get("video_id") for row in rows]:
        raise ValueError("diagnostic request order is invalid")
    audio_paths = protocol.get("audio_path")
    audio_hashes = protocol.get("audio_sha256")
    if not isinstance(audio_paths, Mapping) or not isinstance(audio_hashes, Mapping):
        raise ValueError("diagnostic audio bindings are invalid")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("diagnostic request row must be an object")
        if (
            row.get("run_id") != EXPERIMENT_ID
            or row.get("condition_id") != CONDITION_ID
            or row.get("mode") != "transcription"
            or not isinstance(row.get("video_id"), str)
            or not isinstance(row.get("audio_path"), str)
            or row.get("audio_path") != audio_paths.get(row.get("video_id"))
        ):
            raise ValueError("diagnostic request row identity is invalid")
        audio_path = _ensure_inside(Path(row["audio_path"]), CANONICAL_DATA_ROOT, label="audio path")
        if not audio_path.is_file() or _sha256_file(audio_path) != audio_hashes.get(row["video_id"]):
            raise ValueError("diagnostic audio binding is invalid")
        normalized.append(dict(row))
    return normalized


def build_arm_command(
    root: Path,
    arm_id: str,
    *,
    hub_cache: Path,
    data_root: Path,
) -> list[str]:
    """Build an exact, network-disabled Podman command for one arm."""

    config = _validate_arm_id(arm_id)
    protocol = _verify_inputs(Path(root))
    if not Path(hub_cache).is_dir():
        raise ValueError("Hugging Face cache is missing")
    resolved_data_root = Path(data_root).resolve()
    if resolved_data_root != CANONICAL_DATA_ROOT or not resolved_data_root.is_dir():
        raise ValueError("canonical data root is missing or does not match")
    resolved_root = Path(root).resolve()
    if not resolved_root.is_relative_to(resolved_data_root):
        raise ValueError("diagnostic root must be inside canonical data root")
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("diagnostic experiment identity is invalid")
    repo_root = Path(__file__).resolve().parents[2]
    container_repo = Path("/workspace/trec_video_2026")
    command = [
        "podman",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "keep-groups",
        "--security-opt",
        "label=disable",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--volume",
        f"{Path(hub_cache).resolve()}:/root/.cache/huggingface/hub:ro",
        "--volume",
        f"{resolved_data_root}:{resolved_data_root}:ro",
        "--volume",
        f"{repo_root}:{container_repo}:ro",
        "--workdir",
        str(container_repo),
        "--env",
        f"PYTHONPATH={container_repo / 'src'}",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "HF_HOME=/root/.cache/huggingface",
        "--env",
        "HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub",
        "--env",
        "HOME=/tmp/home",
        "--env",
        f"VQA_RUNTIME_IMAGE={RUNTIME_IMAGE}",
        RUNTIME_IMAGE,
        "/opt/venv/bin/python",
        "-m",
        "trec_video_2026.vqa_qwen_empty_output_runner",
        "--model-id",
        ASR_MODEL_ID,
        "--revision",
        ASR_REVISION,
    ]
    if config["aligner_loaded"]:
        command.extend(
            [
                "--forced-aligner-id",
                ALIGNER_MODEL_ID,
                "--forced-aligner-revision",
                ALIGNER_REVISION,
            ]
        )
    command.extend(["--language", config["language"], "--timestamps", config["timestamps"]])
    return command


def _validate_command(command: Sequence[str], arm_id: str, protocol: Mapping[str, Any]) -> list[str]:
    normalized = list(command)
    if not normalized or any(not isinstance(item, str) for item in normalized):
        raise ValueError("arm command must be a nonempty string array")
    expected_module = "trec_video_2026.vqa_qwen_empty_output_runner"
    if normalized[:2] != ["podman", "run"] or "-i" not in normalized:
        raise ValueError("arm command must be an interactive podman run")
    if "--network" not in normalized or normalized[normalized.index("--network") + 1] != "none":
        raise ValueError("arm command must disable the network")
    if "--read-only" not in normalized:
        raise ValueError("arm command must be read-only")
    if "-i" not in normalized or RUNTIME_IMAGE not in normalized:
        raise ValueError("interactive stdin and pinned image are required")
    if normalized.index("-i") > normalized.index(RUNTIME_IMAGE):
        raise ValueError("interactive stdin must precede the pinned image")
    if RUNTIME_IMAGE not in normalized or expected_module not in normalized:
        raise ValueError("arm command is not pinned to the dedicated runner")
    allowed_options = {
        "--rm",
        "--network",
        "--read-only",
        "--device",
        "--group-add",
        "--security-opt",
        "--tmpfs",
        "--volume",
        "--workdir",
        "--env",
        "--model-id",
        "--revision",
        "--forced-aligner-id",
        "--forced-aligner-revision",
        "--language",
        "--timestamps",
    }
    if any(token.startswith("--") and token not in allowed_options for token in normalized):
        raise ValueError("arm command contains an unexpected option")
    if ASR_REVISION not in normalized:
        raise ValueError("arm command is not pinned to the ASR revision")
    rendered = " ".join(normalized)
    if any(token in rendered for token in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN", "OPENAI_API_KEY")):
        raise ValueError("arm command must not contain credentials")
    if "--env" not in normalized or "HF_HUB_OFFLINE=1" not in normalized or "TRANSFORMERS_OFFLINE=1" not in normalized:
        raise ValueError("arm command must enforce offline model loading")
    volumes = [normalized[index + 1] for index, item in enumerate(normalized[:-1]) if item == "--volume"]
    if not any(value.endswith(":/root/.cache/huggingface/hub:ro") for value in volumes):
        raise ValueError("arm command must mount the read-only Hugging Face cache")
    if not any(value.endswith(f":{CANONICAL_DATA_ROOT}:ro") for value in volumes):
        raise ValueError("arm command must mount the canonical data root read-only")
    config = _validate_arm_id(arm_id)
    if "--language" not in normalized or normalized[normalized.index("--language") + 1] != config["language"]:
        raise ValueError("arm command language does not match frozen arm")
    if "--timestamps" not in normalized or normalized[normalized.index("--timestamps") + 1] != config["timestamps"]:
        raise ValueError("arm command timestamps do not match frozen arm")
    has_aligner = "--forced-aligner-id" in normalized and "--forced-aligner-revision" in normalized
    if has_aligner != bool(config["aligner_loaded"]):
        raise ValueError("arm command aligner mode does not match frozen arm")
    if has_aligner and ALIGNER_REVISION not in normalized:
        raise ValueError("arm command is not pinned to the aligner revision")
    root_value = protocol.get("source_root")
    if not isinstance(root_value, str):
        raise ValueError("protocol source root is missing")
    return normalized


def _sanitize_stderr(stderr: str, requests: Mapping[str, Any]) -> str:
    sanitized = stderr[:1_000_000]
    rows = requests.get("requests")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                audio_path = row.get("audio_path")
                if isinstance(audio_path, str):
                    sanitized = sanitized.replace(audio_path, "<private-audio-path>")
    for pattern in _SECRET_PATTERNS:
        sanitized = re.sub(pattern, "<redacted-secret>", sanitized)
    return sanitized


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_results(
    protocol: Mapping[str, Any],
    requests: Mapping[str, Any],
    arm_id: str,
    results: Any,
) -> dict[str, Any]:
    request_rows = _validate_prepared_requests(protocol, requests, arm_id)
    if not isinstance(results, list) or len(results) != len(request_rows):
        raise ValueError("runner output must contain exactly four rows")
    expected_order = protocol.get("row_order")
    if not isinstance(expected_order, list) or [row.get("video_id") for row in request_rows] != expected_order:
        raise ValueError("request order does not match frozen protocol")
    expected_hashes = protocol.get("audio_sha256")
    durations = protocol.get("duration_seconds")
    if not isinstance(expected_hashes, Mapping) or not isinstance(durations, Mapping):
        raise ValueError("frozen audio bindings are invalid")
    success_count = 0
    failure_types: list[str] = []
    aligned_success_count = 0
    for index, (request, result) in enumerate(zip(request_rows, results, strict=True)):
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise ValueError("arm request/result rows must be objects")
        video_id = expected_order[index]
        for field in ("run_id", "condition_id", "video_id", "mode"):
            if result.get(field) != request.get(field):
                raise ValueError(f"result {field} does not match frozen request")
        if result.get("source_audio_sha256") != expected_hashes.get(video_id):
            raise ValueError("result audio hash does not match frozen audio")
        if result.get("status") != "passed":
            failure_types.append(str(result.get("error_type") or result.get("status") or "unknown"))
            continue
        text = result.get("text")
        language = result.get("language")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("passed result text must be nonempty")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("passed result language must be nonempty")
        success_count += 1
        if ARMS[arm_id]["aligner_loaded"] and ARMS[arm_id]["timestamps"] == "on":
            chunks = result.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("aligned success must contain timestamp chunks")
            previous_end = 0.0
            duration = _number(durations.get(video_id), "audio duration")
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    raise ValueError("aligned chunks must be objects")
                start = _number(chunk.get("start"), "chunk start")
                end = _number(chunk.get("end"), "chunk end")
                if start < 0 or end < start or end > duration or start < previous_end:
                    raise ValueError("aligned chunk bounds are invalid")
                if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                    raise ValueError("aligned chunk text must be nonempty")
                previous_end = end
            aligned_success_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "arm_id": arm_id,
        "status": "ready"
        if success_count == 4
        and (
            ARMS[arm_id]["timestamps"] != "on" or aligned_success_count == 4
        )
        else "not_ready",
        "contract_valid": True,
        "completion_count": success_count,
        "required_count": 4,
        "success_count": success_count,
        "failure_count": 4 - success_count,
        "aligned_success_count": aligned_success_count,
        "failure_types": failure_types,
        "results_sha256": canonical_sha256(results),
        "paid_compute_usd": 0.0,
    }


def record_arm(
    root: Path,
    arm_id: str,
    command: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute exactly one arm attempt and preserve all terminal artifacts."""

    root = Path(root)
    _validate_arm_id(arm_id)
    protocol = _verify_inputs(root)
    arm_root = root / "arms" / arm_id
    if arm_root.exists() and any(arm_root.iterdir()):
        raise ValueError("refusing a second arm attempt")
    arm_position = ARM_ORDER.index(arm_id)
    for prior_arm in ARM_ORDER[:arm_position]:
        if not (root / "arms" / prior_arm / "terminal.json").is_file():
            raise ValueError("arms must be recorded in immutable A-through-D order")
    arm_root.mkdir(parents=True, exist_ok=True)
    normalized = _validate_command(command, arm_id, protocol)
    requests = _read_object(root / "requests" / f"{arm_id}.json")
    command_hash = canonical_sha256(normalized)
    _write_json_exclusive(
        arm_root / "command.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "status": "command_frozen",
            "argv": normalized,
            "command_sha256": command_hash,
        },
    )
    payload = json.dumps(requests["requests"], ensure_ascii=False, separators=(",", ":"))
    started = time.monotonic()
    launch_error: str | None = None
    try:
        completed = runner(normalized, input=payload, text=True, capture_output=True, check=False)
    except Exception as exc:  # noqa: BLE001 - preserve launch failures as terminals
        launch_error = type(exc).__name__
        completed = subprocess.CompletedProcess(
            normalized,
            127,
            stdout="",
            stderr=f"{launch_error}: subprocess launch failed\n",
        )
    elapsed = max(0.0, time.monotonic() - started)
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    sanitized_stderr = _sanitize_stderr(stderr, requests)
    _write_text_exclusive(arm_root / "raw-results.json", stdout)
    _write_text_exclusive(arm_root / "stderr.txt", sanitized_stderr)
    validation: dict[str, Any]
    try:
        parsed = json.loads(stdout)
        validation = _validate_results(protocol, requests, arm_id, parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        validation = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "arm_id": arm_id,
            "status": "not_ready",
            "contract_valid": False,
            "completion_count": 0,
            "required_count": 4,
            "success_count": 0,
            "failure_count": 4,
            "aligned_success_count": 0,
            "failure_types": [type(exc).__name__],
            "validation_error_type": type(exc).__name__,
            "results_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "paid_compute_usd": 0.0,
        }
    if int(completed.returncode) != 0:
        validation["status"] = "not_ready"
        failure_types = validation.get("failure_types")
        if not isinstance(failure_types, list):
            failure_types = []
        terminal_type = launch_error or "NonzeroExitCode"
        if terminal_type not in failure_types:
            failure_types.append(terminal_type)
        validation["failure_types"] = failure_types
    terminal = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "arm_id": arm_id,
        "status": "completed" if int(completed.returncode) == 0 else "failed",
        "attempt_index": 1,
        "exit_code": int(completed.returncode),
        "wall_seconds": elapsed,
        "command_sha256": command_hash,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(sanitized_stderr.encode("utf-8")).hexdigest(),
        "paid_compute_usd": 0.0,
    }
    _write_json_exclusive(arm_root / "terminal.json", terminal)
    _write_json_exclusive(arm_root / "validated-results.json", validation)
    return terminal


def _recompute_arm_validation(
    root: Path, arm_id: str, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Recompute the contract result from immutable raw stdout."""

    arm_root = root / "arms" / arm_id
    try:
        parsed = json.loads((arm_root / "raw-results.json").read_text(encoding="utf-8"))
        return _validate_results(
            protocol,
            _read_object(root / "requests" / f"{arm_id}.json"),
            arm_id,
            parsed,
        ), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__


def _verify_arm(root: Path, arm_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _verify_live_bindings(root)
    arm_root = root / "arms" / arm_id
    command = _read_object(arm_root / "command.json")
    terminal = _read_object(arm_root / "terminal.json")
    validation = _read_object(arm_root / "validated-results.json")
    if command.get("arm_id") != arm_id or terminal.get("arm_id") != arm_id or validation.get("arm_id") != arm_id:
        raise ValueError(f"{arm_id} artifact identity is invalid")
    argv = command.get("argv")
    if not isinstance(argv, list):
        raise ValueError(f"{arm_id} command argv is invalid")
    _validate_command(argv, arm_id, protocol)
    if command.get("command_sha256") != canonical_sha256(argv):
        raise ValueError(f"{arm_id} command hash does not match command")
    if terminal.get("command_sha256") != command.get("command_sha256"):
        raise ValueError(f"{arm_id} command hash does not match terminal")
    if terminal.get("stdout_sha256") != hashlib.sha256((arm_root / "raw-results.json").read_bytes()).hexdigest():
        raise ValueError(f"{arm_id} raw output hash does not match terminal")
    if terminal.get("stderr_sha256") != hashlib.sha256((arm_root / "stderr.txt").read_bytes()).hexdigest():
        raise ValueError(f"{arm_id} stderr hash does not match terminal")
    if terminal.get("status") == "completed" and terminal.get("exit_code") != 0:
        raise ValueError(f"{arm_id} completed terminal has nonzero exit")
    if terminal.get("status") not in {"completed", "failed"}:
        raise ValueError(f"{arm_id} terminal status is invalid")
    recomputed, error_type = _recompute_arm_validation(root, arm_id, protocol)
    saved_contract_valid = validation.get("contract_valid") is True
    if recomputed is None:
        if saved_contract_valid or validation.get("validation_error_type") != error_type:
            raise ValueError(f"{arm_id} validation artifact does not describe invalid raw output")
    else:
        if not saved_contract_valid:
            raise ValueError(f"{arm_id} validation artifact hides a valid raw output")
        expected_validation = dict(recomputed)
        if terminal.get("exit_code") != 0:
            expected_validation["status"] = "not_ready"
            failure_types = list(expected_validation.get("failure_types", []))
            if "NonzeroExitCode" not in failure_types:
                failure_types.append("NonzeroExitCode")
            expected_validation["failure_types"] = failure_types
        compare_fields = (
            "status",
            "completion_count",
            "required_count",
            "success_count",
            "failure_count",
            "aligned_success_count",
            "failure_types",
            "results_sha256",
            "paid_compute_usd",
        )
        for field in compare_fields:
            if validation.get(field) != expected_validation.get(field):
                raise ValueError(f"{arm_id} validation does not match raw output")
    return command, terminal, validation


def freeze_diagnostic(root: Path) -> dict[str, Any]:
    """Freeze all four attempts and compute the predeclared decision."""

    root = Path(root)
    _verify_inputs(root)
    freeze_path = root / "diagnostic-freeze.json"
    if freeze_path.exists():
        raise ValueError("refusing to overwrite diagnostic freeze")
    arm_freezes: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    arm_records: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for arm_id in ARM_ORDER:
        arm_root = root / "arms" / arm_id
        required = ["command.json", "raw-results.json", "stderr.txt", "terminal.json", "validated-results.json"]
        if not all((arm_root / name).is_file() for name in required):
            raise ValueError("freeze requires one terminal attempt for every arm")
        command, terminal, validation = _verify_arm(root, arm_id)
        arm_records[arm_id] = (command, terminal, validation)
        arm_freezes[arm_id] = {
            "command": _freeze_json(arm_root / "command.json", bindings={"experiment_id": EXPERIMENT_ID, "arm_id": arm_id}),
            "raw-results": _freeze_file(arm_root / "raw-results.json"),
            "stderr": _freeze_file(arm_root / "stderr.txt"),
            "terminal": _freeze_json(arm_root / "terminal.json", bindings={"experiment_id": EXPERIMENT_ID, "arm_id": arm_id}),
            "validated-results": _freeze_json(arm_root / "validated-results.json", bindings={"experiment_id": EXPERIMENT_ID, "arm_id": arm_id}),
        }
        aggregate[arm_id] = {
            "terminal_status": terminal.get("status"),
            "exit_code": terminal.get("exit_code"),
            "contract_valid": validation.get("contract_valid") is True,
            "success_count": validation.get("success_count", 0),
            "failure_count": validation.get("failure_count", 4),
            "aligned_success_count": validation.get("aligned_success_count", 0),
            "validation_status": validation.get("status"),
        }

    selection = _read_object(root / "selection.json")
    selected_rows = selection.get("rows")
    if not isinstance(selected_rows, list) or len(selected_rows) != 4:
        raise ValueError("selection rows are invalid")
    role_by_id = {row.get("video_id"): row.get("role") for row in selected_rows if isinstance(row, Mapping)}
    baseline_reproducible = True
    baseline_validation = arm_records["aligned_auto"][2]
    if (
        arm_records["aligned_auto"][1].get("status") != "completed"
        or baseline_validation.get("contract_valid") is not True
        or baseline_validation.get("status") not in {"ready", "not_ready"}
    ):
        baseline_reproducible = False
    try:
        baseline_results = json.loads((root / "arms" / "aligned_auto" / "raw-results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        baseline_results = None
    if not isinstance(baseline_results, list) or len(baseline_results) != 4:
        baseline_reproducible = False
    else:
        for result in baseline_results:
            if not isinstance(result, Mapping):
                baseline_reproducible = False
                continue
            role = role_by_id.get(result.get("video_id"))
            if role == "failure":
                if result.get("status") != "failed" or result.get("error_type") != EMPTY_ERROR_TYPE or result.get("error_message") != EMPTY_ERROR_MESSAGE:
                    baseline_reproducible = False
            elif role == "control":
                if (
                    result.get("status") != "passed"
                    or not isinstance(result.get("text"), str)
                    or not result["text"].strip()
                    or not isinstance(result.get("language"), str)
                    or not result["language"].strip()
                ):
                    baseline_reproducible = False
                chunks = result.get("chunks")
                if (
                    not isinstance(chunks, list)
                    or not chunks
                    or any(
                        not isinstance(chunk, Mapping)
                        or not isinstance(chunk.get("text"), str)
                        or not chunk["text"].strip()
                        for chunk in chunks
                    )
                ):
                    baseline_reproducible = False
            else:
                baseline_reproducible = False

    any_terminal_invalid = any(
        record[1].get("status") != "completed" or record[1].get("exit_code") != 0
        for record in arm_records.values()
    )
    d_passes = (
        arm_records["aligned_english"][1].get("status") == "completed"
        and arm_records["aligned_english"][1].get("exit_code") == 0
        and arm_records["aligned_english"][2].get("status") == "ready"
        and arm_records["aligned_english"][2].get("contract_valid") is True
        and arm_records["aligned_english"][2].get("success_count") == 4
        and arm_records["aligned_english"][2].get("aligned_success_count") == 4
    )
    all_contract_valid = all(
        record[2].get("contract_valid") is True for record in arm_records.values()
    )
    if any_terminal_invalid or not all_contract_valid:
        decision = "not_ready"
    elif not baseline_reproducible:
        decision = "unstable_baseline"
    elif d_passes:
        decision = "ready_for_fresh_aligned_quality_panel"
    else:
        decision = "not_ready"

    explanatory = {
        "base_auto_all_rows_passed": aggregate["base_auto"]["validation_status"] == "ready",
        "aligner_plain_auto_all_rows_passed": aggregate["aligner_plain_auto"]["validation_status"] == "ready",
        "aligner_load_localization": aggregate["base_auto"]["validation_status"] == "ready",
        "timestamp_path_localization": aggregate["aligner_plain_auto"]["validation_status"] == "ready",
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic_frozen",
        "aggregate": aggregate,
        "arm_artifacts": arm_freezes,
        "baseline_reproducible": baseline_reproducible,
        "explanatory_localization": explanatory,
        "advancing_arm": "aligned_english" if decision == "ready_for_fresh_aligned_quality_panel" else None,
        "decision": decision,
        "paid_compute_usd": 0.0,
        "one_attempt_only": True,
        "input_freeze_sha256": _sha256_file(root / "inputs-freeze.json"),
    }
    payload["freeze_payload_sha256"] = canonical_sha256(payload)
    _write_json_exclusive(freeze_path, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic_frozen",
        "decision": decision,
        "baseline_reproducible": baseline_reproducible,
        "advancing_arm": payload["advancing_arm"],
        "aggregate": aggregate,
        "explanatory_localization": explanatory,
        "paid_compute_usd": 0.0,
    }


def _verify_diagnostic_freeze(root: Path) -> dict[str, Any]:
    payload = _read_object(root / "diagnostic-freeze.json")
    expected_hash = payload.get("freeze_payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("freeze_payload_sha256", None)
    if not _is_sha256(expected_hash) or expected_hash != canonical_sha256(without_hash):
        raise ValueError("diagnostic freeze payload hash is invalid")
    if payload.get("status") != "diagnostic_frozen" or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("diagnostic freeze identity is invalid")
    if payload.get("input_freeze_sha256") != _sha256_file(root / "inputs-freeze.json"):
        raise ValueError("diagnostic input freeze changed")
    artifacts = payload.get("arm_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARM_ORDER):
        raise ValueError("diagnostic arm freezes are invalid")
    for arm_id in ARM_ORDER:
        arm_root = root / "arms" / arm_id
        entries = artifacts[arm_id]
        if not isinstance(entries, Mapping):
            raise ValueError("diagnostic arm freeze entry is invalid")
        for name, filename in (("command", "command.json"), ("raw-results", "raw-results.json"), ("stderr", "stderr.txt"), ("terminal", "terminal.json"), ("validated-results", "validated-results.json")):
            record = entries.get(name)
            if not isinstance(record, Mapping):
                raise ValueError(f"diagnostic arm freeze missing {name}")
            if name in {"command", "terminal", "validated-results"}:
                _verify_json_freeze(arm_root / filename, record)
            else:
                _verify_file_freeze(arm_root / filename, record)
        _verify_arm(root, arm_id)
    return payload


def status_diagnostic(root: Path) -> dict[str, Any]:
    """Verify the live immutable state and return aggregate-only status."""

    root = Path(root)
    protocol = _verify_inputs(root)
    freeze_path = root / "diagnostic-freeze.json"
    if freeze_path.is_file():
        payload = _verify_diagnostic_freeze(root)
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "diagnostic_frozen",
            "decision": payload["decision"],
            "baseline_reproducible": payload["baseline_reproducible"],
            "advancing_arm": payload["advancing_arm"],
            "aggregate": payload["aggregate"],
            "explanatory_localization": payload["explanatory_localization"],
            "paid_compute_usd": 0.0,
        }
    attempted: list[str] = []
    for arm_id in ARM_ORDER:
        arm_root = root / "arms" / arm_id
        if (arm_root / "terminal.json").is_file():
            _verify_arm(root, arm_id)
            attempted.append(arm_id)
    if not attempted:
        state = "inputs_frozen"
    elif len(attempted) < len(ARM_ORDER):
        state = "arms_in_progress"
    else:
        state = "arms_completed"
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "attempted_arm_count": len(attempted),
        "required_arm_count": len(ARM_ORDER),
        "paid_compute_usd": 0.0,
        "helper_source_sha256": protocol["helper_source_sha256"],
    }


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "ASR_MODEL_ID",
    "ASR_REVISION",
    "ALIGNER_MODEL_ID",
    "ALIGNER_REVISION",
    "CANONICAL_DATA_ROOT",
    "CONDITION_ID",
    "DIAGNOSTIC_RUNNER_PATH",
    "EXPERIMENT_ID",
    "EXPECTED_SOURCE_EXPERIMENT_ID",
    "EXPECTED_SOURCE_ROOT",
    "HELPER_SOURCE_PATH",
    "PRIVATE_ROOT",
    "RUNTIME_IMAGE",
    "build_arm_command",
    "canonical_sha256",
    "freeze_diagnostic",
    "prepare_diagnostic",
    "record_arm",
    "status_diagnostic",
]
