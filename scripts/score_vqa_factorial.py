#!/usr/bin/env python3
"""Score the explicit four-cell VQA model-by-prompt factorial.

The scorer deliberately has a separate entry point from
``score_vqa_semantic_panel.py``.  That older entry point is the frozen two-cell
Task 1 scorer; this module owns the four-cell Phase 1 analysis and never
widens the older contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for _path in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score_vqa_semantic_panel as task1  # noqa: E402
from trec_video_2026.vqa_evaluation import (  # noqa: E402
    normalized_exact_match,
    semantic_composite,
    token_f1,
)


EXPECTED_SCHEMA_VERSION = 2
EXPECTED_BOOTSTRAP_SEED = 20260809
EXPECTED_BOOTSTRAP_ITERATIONS = 2000
EXPECTED_BERT_MODEL_SNAPSHOT = "722cf37b1afa9454edce342e7895e588b6ff1d59"
EXPECTED_BERT_SCORE_MODULE_VERSION = "0.3.12"
EXPECTED_STS_TIMEOUT = 10.0
EXPECTED_STS_RETRIES = 3
STS_ENDPOINT = task1.STS_ENDPOINT
EXPECTED_BASELINE_ARTIFACT_SHA256 = (
    "f6259ee8457463c3bcf3fc8233e063f63f9a1c103062e69012b1d6861380dfd1"
)
EXPECTED_BASELINE_ROW_KEY = "qwen35-4b-12f-whisper-large-v3"
EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256 = (
    "ad9c9aabace33d3091c076e2f06c10ef8e55d355be9e5e392d790a80dbd78377"
)
EXPECTED_BASELINE_SCORER_SHA256 = (
    "6124abd3ae7e0ecc5cb6fccc4b0ddec218c8ea5ec3730f54c58d246eb7a347da"
)
EXPECTED_FULL_CANONICAL_ID_DIGEST = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXPECTED_FULL_MANIFEST_SHA256 = (
    "092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8"
)
EXPECTED_CONDITION_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
DEFAULT_PROFILE_PATH = (
    REPOSITORY_ROOT / "configs/vqa/vqa_factorial_full_condition_profiles.json"
)
EXPECTED_PROFILE_REMOTE_URIS = {
    "vqa-factorial-full-20260809-f1": (
        "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
        "experiments/vqa-factorial-full-20260809-f1/conditions"
    ),
    "vqa-factorial-full-20260809-f2": (
        "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
        "experiments/vqa-factorial-full-20260809-f2/conditions"
    ),
}
TASK1_SCORER_PATH = REPOSITORY_ROOT / "scripts/score_vqa_semantic_panel.py"
PANEL_ID_DIGEST_ALGORITHM = task1.PANEL_ID_DIGEST_ALGORITHM
EXPECTED_RUNTIME_CONTROLS = {
    "runtime_image": (
        "docker.io/pytorch/pytorch@sha256:"
        "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
    ),
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "frame_count": 12,
    "max_new_tokens": 64,
    "do_sample": False,
    "thinking": False,
    "min_pixels": 65536,
    "max_pixels": 262144,
    "transcript_policy": "shared-accepted-whisper",
}
EXPECTED_RUNTIME_DEPENDENCIES = {
    "Pillow": "11.3.0",
    "soundfile": "0.13.1",
    "accelerate": "1.12.0",
    "av": "16.0.1",
    "imageio-ffmpeg": "0.6.0",
    "qwen-vl-utils": "0.0.14",
}
EXPECTED_QUESTION_TYPES = frozenset(
    {"count", "visible-text", "speech", "temporal", "causal-procedural", "general"}
)
EXPECTED_CONDITION_BINDINGS = {
    "4b-current": (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "current",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "4b-question-aware": (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "question-aware",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "8b-current": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "current",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "8b-question-aware": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "question-aware",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
}
METRIC_FIELDS = ("meteor", "bertscore_f1", "sts", "semantic_composite")
PRODUCER_PREDICTION_FIELDS = frozenset(
    {
        "status",
        "answer",
        "elapsed_seconds",
        "model_load_seconds",
        "model_id",
        "model_revision",
        "runtime",
        "runtime_image",
        "precision",
        "device",
        "runtime_attestation",
        "peak_memory_bytes",
        "error_type",
        "error_message",
        "answer_policy",
        "question_type",
        "requested_min_pixels_per_frame",
        "requested_max_pixels_per_frame",
        "min_pixels_per_frame",
        "max_pixels_per_frame",
        "run_id",
        "condition_id",
        "video_id",
    }
)
PRODUCER_ATTESTATION_FIELDS = frozenset(
    {
        "parameter_tensor_count",
        "parameter_devices",
        "parameter_dtypes",
        "buffer_tensor_count",
        "buffer_devices",
        "buffer_dtypes",
        "accelerator_backend",
        "torch_cuda",
        "torch_hip",
    }
)
_LABEL_KEYS = task1._LABEL_KEYS
MetricFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]


PanelProfile = task1.PanelProfile
CANONICAL_PANEL_PROFILE = task1.CANONICAL_PANEL_PROFILE
sha256_path = task1.sha256_path
sorted_video_id_set_sha256 = task1.sorted_video_id_set_sha256
normalize_nist_text = task1.normalize_nist_text
_finite = task1._finite
_load_annotations = task1._load_annotations
_validate_panel_identity = task1._validate_panel_identity
umbc_sts_score = task1.umbc_sts_score
urllib = task1.urllib


class STSCache:
    """Private in-memory STS cache; keys are opaque identity hashes."""

    def __init__(
        self,
        values: Mapping[str, float] | None = None,
        identity: str | None = None,
    ) -> None:
        self.values = dict(values or {})
        self.identity = identity


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def profile_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(profile)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON input: {path}") from error


def validate_factorial_profile(
    payload: Mapping[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Validate the exact four-cell profile and return a detached copy."""

    required = {
        "run_name",
        "input_uri",
        "remote_uri",
        "manifest",
        "expected_count",
        "frame_count",
        "conditions",
    }
    if set(payload) != required:
        raise ValueError("factorial profile fields are not exact")
    if strict:
        expected_remote_uri = EXPECTED_PROFILE_REMOTE_URIS.get(payload.get("run_name"))
        if expected_remote_uri is None:
            raise ValueError("factorial profile run_name is not pinned")
        expected_header = {
            "input_uri": (
                "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
                "experiments/vqa-factorial-full-evidence-20260809-f1"
            ),
            "remote_uri": expected_remote_uri,
            "manifest": "/dstack/run/configs/vqa/qwen35_vqa2025_475_label_free.json",
            "expected_count": 475,
            "frame_count": 12,
        }
        for field_name, expected in expected_header.items():
            if payload.get(field_name) != expected:
                raise ValueError(f"factorial profile {field_name} is not pinned")
    if not isinstance(payload["conditions"], list):
        raise ValueError("factorial profile conditions must be a list")
    if tuple(
        condition.get("condition_id")
        for condition in payload["conditions"]
        if isinstance(condition, Mapping)
    ) != EXPECTED_CONDITION_IDS:
        raise ValueError("factorial profile must declare the four condition IDs in order")
    if len(payload["conditions"]) != len(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial profile must declare exactly four conditions")
    for condition_id, condition in zip(EXPECTED_CONDITION_IDS, payload["conditions"], strict=True):
        if not isinstance(condition, Mapping):
            raise ValueError("factorial profile conditions must be objects")
        if set(condition) != {
            "condition_id",
            "model_id",
            "revision",
            "answer_policy",
            "min_pixels",
            "max_pixels",
            "request_source",
        }:
            raise ValueError(f"condition {condition_id} fields are not exact")
        actual = (
            condition["model_id"],
            condition["revision"],
            condition["answer_policy"],
            condition["min_pixels"],
            condition["max_pixels"],
            condition["request_source"],
        )
        if actual != EXPECTED_CONDITION_BINDINGS[condition_id]:
            raise ValueError(f"condition {condition_id} model/policy binding mismatch")
    if isinstance(payload["expected_count"], bool) or int(payload["expected_count"]) <= 0:
        raise ValueError("factorial profile expected_count must be positive")
    if payload["frame_count"] != EXPECTED_RUNTIME_CONTROLS["frame_count"]:
        raise ValueError("factorial profile frame_count is not pinned")
    return copy.deepcopy(dict(payload))


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("factorial profile must be an object")
    return validate_factorial_profile(payload, strict=True)


def _profile_conditions(profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(condition["condition_id"]): condition for condition in profile["conditions"]
    }


def _condition_tuple(condition: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        condition.get("model_id"),
        condition.get("revision"),
        condition.get("answer_policy"),
        condition.get("min_pixels"),
        condition.get("max_pixels"),
        condition.get("request_source"),
    )


def _bundle_relative_path(raw: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("bundle checksum path is not a string")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe bundle checksum path: {raw}")
    return candidate


def _verify_checksum_file(
    root: Path, *, allow_unbound_evidence_manifest: bool = False
) -> tuple[str, dict[str, str]]:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError(f"bundle checksum manifest is missing: {checksum_path}")
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, raw_path = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("bundle checksum manifest is malformed")
        relative = _bundle_relative_path(raw_path)
        key = relative.as_posix()
        if key in entries or key in {"SHA256SUMS", "terminal-status.json"}:
            raise ValueError("bundle checksum manifest has a duplicate or forbidden path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"bundle checksum verification failed: {key}")
        entries[key] = digest
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and item not in {root / "SHA256SUMS", root / "terminal-status.json"}
    }
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("bundle payload cannot contain symlinks")
    expected_entries = actual
    if allow_unbound_evidence_manifest and "evidence/SHA256SUMS" in actual:
        expected_entries = actual - {"evidence/SHA256SUMS"}
    if set(entries) != expected_entries:
        raise ValueError("bundle checksum manifest does not cover the exact payload")
    return sha256_path(checksum_path), entries


def _producer_layout(root: Path, condition_id: str) -> tuple[Path, Path, Path]:
    """Resolve the downloaded producer root, payload, and condition directory."""

    root = root.resolve()
    payload_root = root / "payload"
    terminal_path = root / "terminal-status.json"
    if not root.is_dir() or not payload_root.is_dir() or not terminal_path.is_file():
        raise ValueError(
            "production bundle must be the downloaded root containing terminal-status.json beside payload/"
        )
    condition_root = payload_root / "conditions" / condition_id
    if not condition_root.is_dir():
        raise ValueError(f"condition {condition_id} files are missing under payload/conditions")
    return payload_root, condition_root, terminal_path


def _parse_checksum_entries(checksum_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, raw_path = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("evidence checksum manifest is malformed")
        relative = _bundle_relative_path(raw_path)
        key = relative.as_posix()
        if key in entries or key in {"SHA256SUMS", "terminal-status.json"}:
            raise ValueError("evidence checksum manifest has a duplicate or forbidden path")
        entries[key] = digest
    return entries


def _evidence_bytes_digest(evidence_root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(relative_paths):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update((evidence_root / key).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_evidence_checksum(
    payload_root: Path,
    marker_path: Path,
    root_entries: Mapping[str, str],
) -> tuple[str, str, str]:
    """Verify the nested evidence manifest and every evidence byte exactly."""

    evidence_root = payload_root / "evidence"
    checksum_path = evidence_root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError("evidence checksum manifest is missing")
    if root_entries.get("evidence/SHA256SUMS") != sha256_path(checksum_path):
        raise ValueError("nested evidence checksum is not root-bound")
    entries = _parse_checksum_entries(checksum_path)
    normalized: dict[str, str] = {}
    for key, digest in entries.items():
        relative = Path(key)
        if relative.parts and relative.parts[0] == "evidence":
            relative = Path(*relative.parts[1:])
        if not relative.parts:
            raise ValueError("evidence checksum contains an empty path")
        normalized_key = relative.as_posix()
        if normalized_key in normalized:
            raise ValueError("evidence checksum contains duplicate normalized paths")
        path = evidence_root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(evidence_root) or not path.is_file():
            raise ValueError(f"evidence checksum verification failed: {key}")
        if sha256_path(path) != digest:
            raise ValueError(f"evidence checksum verification failed: {key}")
        normalized[normalized_key] = digest
    actual = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if set(normalized) != actual:
        raise ValueError("evidence checksum manifest does not cover the exact evidence bytes")
    if normalized.get("evidence-marker.json") != sha256_path(marker_path):
        raise ValueError("evidence marker checksum is invalid")
    return (
        sha256_path(checksum_path),
        _evidence_bytes_digest(evidence_root, sorted(normalized)),
        sha256_path(marker_path),
    )


def _finite_positive(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _read_operational_evidence(
    root: Path,
    *,
    condition_id: str,
    expected_rows: int,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    timing = _load_json(root / "timing.json")
    if not isinstance(timing, Mapping) or timing.get("condition_id") != condition_id:
        raise ValueError(f"condition {condition_id} timing identity is invalid")
    required_timing_fields = {
        "condition_id",
        "row_count",
        "wall_seconds",
        "peak_memory_bytes",
        "model_load_seconds",
        "runtime_identity",
    }
    optional_timing_fields = {"prepare_wall_seconds"}
    if not required_timing_fields <= set(timing) or not set(timing) <= required_timing_fields | optional_timing_fields:
        raise ValueError(f"condition {condition_id} timing schema is not exact")
    if timing.get("row_count") != expected_rows:
        raise ValueError(f"condition {condition_id} timing row count is invalid")
    wall_seconds = _finite_positive(timing.get("wall_seconds"), field="wall_seconds")
    preparation = timing.get("prepare_wall_seconds")
    if preparation is not None:
        preparation = _finite(preparation, field="prepare_wall_seconds")
        if preparation < 0:
            raise ValueError("prepare_wall_seconds must not be negative")
    timing_peak = timing.get("peak_memory_bytes")
    if isinstance(timing_peak, bool) or not isinstance(timing_peak, int) or timing_peak <= 0:
        raise ValueError("timing peak_memory_bytes must be a positive integer")
    model_load_seconds = _finite(timing.get("model_load_seconds"), field="model_load_seconds")
    if model_load_seconds < 0:
        raise ValueError("model_load_seconds must not be negative")
    runtime_identity = timing.get("runtime_identity")
    if not isinstance(runtime_identity, list) or len(runtime_identity) != 1:
        raise ValueError("timing runtime identity is invalid")
    telemetry_path = root / "telemetry" / "nvidia-smi.csv"
    if not telemetry_path.is_file():
        raise ValueError("NVIDIA telemetry CSV is missing")
    reader = csv.DictReader(telemetry_path.read_text(encoding="utf-8").splitlines())
    required = {
        "timestamp",
        "index",
        "utilization.gpu [%]",
        "memory.used [MiB]",
        "memory.total [MiB]",
    }
    if set(reader.fieldnames or ()) != required:
        raise ValueError("NVIDIA telemetry CSV schema is invalid")
    utilization: list[float] = []
    telemetry_peak = 0.0
    samples = 0
    for record in reader:
        try:
            index = int(record["index"])
            gpu_util = _finite(record["utilization.gpu [%]"], field="gpu utilization")
            memory_used = _finite(record["memory.used [MiB]"], field="memory used")
            memory_total = _finite(record["memory.total [MiB]"], field="memory total")
        except KeyError as error:
            raise ValueError("NVIDIA telemetry row is incomplete") from error
        if index < 0 or not record["timestamp"].strip() or not 0 <= gpu_util <= 100 or memory_used < 0 or memory_total <= 0 or memory_used > memory_total:
            raise ValueError("NVIDIA telemetry values are invalid")
        utilization.append(gpu_util)
        telemetry_peak = max(telemetry_peak, memory_used * 1024 * 1024)
        samples += 1
    if not utilization:
        raise ValueError("NVIDIA telemetry has no samples")
    prediction_peaks = [
        row["peak_memory_bytes"]
        for row in prediction_rows
        if isinstance(row.get("peak_memory_bytes"), int)
    ]
    if len(prediction_peaks) != expected_rows:
        raise ValueError("prediction rows do not contain complete PyTorch memory measurements")
    observed_runtime_identity = [
        [
            prediction_rows[0].get("runtime"),
            prediction_rows[0].get("runtime_image"),
            prediction_rows[0].get("device"),
            prediction_rows[0].get("precision"),
        ]
    ]
    if runtime_identity != observed_runtime_identity:
        raise ValueError("timing runtime identity does not match prediction rows")
    row_model_loads = {row.get("model_load_seconds") for row in prediction_rows}
    if row_model_loads != {model_load_seconds}:
        raise ValueError("timing model-load identity does not match prediction rows")
    if timing_peak != max(prediction_peaks):
        raise ValueError("timing peak_memory_bytes does not match the maximum row peak")
    if timing_peak < 1 or any(peak <= 0 for peak in prediction_peaks):
        raise ValueError("PyTorch max-allocated memory measurements are invalid")
    return {
        "wall_seconds": wall_seconds,
        "gpu_seconds": wall_seconds,
        "gpu_hours": wall_seconds / 3600.0,
        "rows_per_second": expected_rows / wall_seconds,
        "preparation_wall_seconds": preparation,
        "model_load_seconds": model_load_seconds,
        "torch_peak_memory_bytes": int(timing_peak),
        "torch_observed_peak_memory_bytes": int(max(prediction_peaks)),
        "nvidia_smi_peak_memory_bytes": int(telemetry_peak),
        "utilization_mean_percent": statistics.fmean(utilization),
        "utilization_max_percent": max(utilization),
        "utilization_sample_count": samples,
        "timing_sha256": sha256_path(root / "timing.json"),
        "telemetry_sha256": sha256_path(telemetry_path),
    }


def _load_condition_bundle(
    root: Path,
    *,
    condition_id: str,
    profile_condition: Mapping[str, Any],
    expected_rows: int,
    expected_id_digest: str,
    profile_run_name: str | None = None,
    profile_remote_uri: str | None = None,
    shared_evidence_identity: tuple[str, str, str, str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    payload_root, condition_root, terminal_path = _producer_layout(root, condition_id)
    checksum_sha256, checksum_entries = _verify_checksum_file(
        payload_root, allow_unbound_evidence_manifest=shared_evidence_identity is not None
    )
    terminal = _load_json(terminal_path)
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("payload_sha256") != checksum_sha256
        or not isinstance(terminal.get("run_name"), str)
        or not terminal.get("run_name", "").strip()
        or not isinstance(terminal.get("payload_uri"), str)
        or not terminal.get("payload_uri", "").strip()
    ):
        raise ValueError(f"condition {condition_id} terminal status is not passed and verified")
    if profile_remote_uri is not None:
        remote = profile_remote_uri.rstrip("/")
        expected_payload_uris = {f"{remote}/payload", f"{remote}/{condition_id}/payload"}
        if terminal["payload_uri"] not in expected_payload_uris:
            raise ValueError(f"condition {condition_id} terminal payload URI is not profile-bound")
    status = _load_json(payload_root / "status.json")
    if (
        not isinstance(status, Mapping)
        or status.get("schema_version") != 1
        or status.get("status") != "passed"
        or status.get("mode") != "condition"
        or status.get("run_name") != terminal.get("run_name")
        or status.get("expected_count") != expected_rows
        or status.get("frame_count") != EXPECTED_RUNTIME_CONTROLS["frame_count"]
        or status.get("workload_returncode") != 0
        or status.get("remote_payload_sync") != "frozen"
        or status.get("labels_joined") is not False
    ):
        raise ValueError(f"condition {condition_id} status marker is invalid")
    identity_payload = _load_json(condition_root / "condition-identity.json")
    if not isinstance(identity_payload, Mapping):
        raise ValueError(f"condition {condition_id} identity is invalid")
    nested_condition = identity_payload.get("condition")
    if nested_condition is not None and not isinstance(nested_condition, Mapping):
        raise ValueError(f"condition {condition_id} nested identity is invalid")
    nested_condition = nested_condition if isinstance(nested_condition, Mapping) else identity_payload
    decoding = identity_payload.get("deterministic_decoding", {})
    if not isinstance(decoding, Mapping):
        raise ValueError(f"condition {condition_id} decoding identity is invalid")
    identity = {
        **dict(identity_payload),
        **dict(nested_condition),
        "run_id": identity_payload.get(
            "run_id", identity_payload.get("run_name", nested_condition.get("run_id"))
        ),
        "model_revision": nested_condition.get(
            "model_revision", nested_condition.get("revision")
        ),
        "requested_min_pixels_per_frame": identity_payload.get(
            "requested_min_pixels_per_frame", nested_condition.get("min_pixels")
        ),
        "requested_max_pixels_per_frame": identity_payload.get(
            "requested_max_pixels_per_frame", nested_condition.get("max_pixels")
        ),
        "applied_min_pixels_per_frame": identity_payload.get(
            "applied_min_pixels_per_frame", nested_condition.get("min_pixels")
        ),
        "applied_max_pixels_per_frame": identity_payload.get(
            "applied_max_pixels_per_frame", nested_condition.get("max_pixels")
        ),
        "max_new_tokens": identity_payload.get(
            "max_new_tokens", decoding.get("max_new_tokens")
        ),
        "do_sample": identity_payload.get("do_sample", decoding.get("do_sample")),
        "thinking": identity_payload.get(
            "thinking", decoding.get("enable_thinking")
        ),
    }
    expected_tuple = _condition_tuple(profile_condition)
    observed_tuple = (
        identity.get("model_id"),
        identity.get("model_revision"),
        identity.get("answer_policy"),
        identity.get("applied_min_pixels_per_frame"),
        identity.get("applied_max_pixels_per_frame"),
        identity.get("request_source"),
    )
    if identity.get("condition_id") != condition_id or observed_tuple != expected_tuple:
        raise ValueError(f"condition {condition_id} identity tuple does not match the profile")
    condition_dirs = [
        path for path in (payload_root / "conditions").iterdir()
        if path.is_dir() and not path.is_symlink()
    ]
    if profile_run_name:
        expected_terminal_run = (
            profile_run_name
            if len(condition_dirs) > 1
            else f"{profile_run_name}-{condition_id}"
        )
        if terminal["run_name"] != expected_terminal_run:
            raise ValueError(f"condition {condition_id} terminal run identity is not profile-bound")
        expected_run_ids = {
            f"{profile_run_name}-{condition_id}"
            if len(condition_dirs) > 1
            else expected_terminal_run
        }
    else:
        expected_run_ids = {str(terminal["run_name"])}
        if len(condition_dirs) > 1:
            expected_run_ids.add(f"{terminal['run_name']}-{condition_id}")
    if identity.get("run_id") not in expected_run_ids:
        raise ValueError(f"condition {condition_id} run identity is not terminal-bound")
    for key, expected in {
        "requested_min_pixels_per_frame": expected_tuple[3],
        "requested_max_pixels_per_frame": expected_tuple[4],
        "runtime_image": EXPECTED_RUNTIME_CONTROLS["runtime_image"],
        "transformers_revision": EXPECTED_RUNTIME_CONTROLS["transformers_revision"],
        "frame_count": EXPECTED_RUNTIME_CONTROLS["frame_count"],
        "max_new_tokens": EXPECTED_RUNTIME_CONTROLS["max_new_tokens"],
        "do_sample": EXPECTED_RUNTIME_CONTROLS["do_sample"],
        "thinking": EXPECTED_RUNTIME_CONTROLS["thinking"],
    }.items():
        if identity.get(key) != expected:
            raise ValueError(f"condition {condition_id} observed runtime control mismatch: {key}")
    prediction_path = condition_root / "predictions.json"
    if not isinstance(_load_json(prediction_path), list):
        raise ValueError(f"condition {condition_id} prediction payload wrapper is not the producer schema")
    raw_rows, _metadata = _load_prediction_payload(prediction_path)
    if len(raw_rows) != expected_rows:
        raise ValueError(f"Condition {condition_id} has an incomplete prediction payload")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        _reject_label_fields(raw, f"$[{index}]")
        if set(raw) != PRODUCER_PREDICTION_FIELDS:
            raise ValueError(f"condition {condition_id} prediction schema is not exact")
        video_id = raw.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError(f"condition {condition_id} has a duplicate or invalid video ID")
        seen.add(video_id)
        expected_row = {
            "run_id": identity["run_id"],
            "condition_id": condition_id,
            "model_id": expected_tuple[0],
            "model_revision": expected_tuple[1],
            "answer_policy": expected_tuple[2],
            "requested_min_pixels_per_frame": expected_tuple[3],
            "requested_max_pixels_per_frame": expected_tuple[4],
            "min_pixels_per_frame": expected_tuple[3],
            "max_pixels_per_frame": expected_tuple[4],
            "runtime_image": EXPECTED_RUNTIME_CONTROLS["runtime_image"],
            "precision": "bfloat16",
            "device": "cuda:0 (NVIDIA CUDA)",
        }
        if any(raw.get(key) != value for key, value in expected_row.items()):
            raise ValueError(f"condition {condition_id} prediction row binding mismatch")
        attestation = raw.get("runtime_attestation")
        if not isinstance(attestation, Mapping) or set(attestation) != PRODUCER_ATTESTATION_FIELDS:
            raise ValueError(f"condition {condition_id} prediction lacks NVIDIA BF16 attestation")
        for label, count_key, devices_key, dtypes_key in (
            ("parameter", "parameter_tensor_count", "parameter_devices", "parameter_dtypes"),
            ("buffer", "buffer_tensor_count", "buffer_devices", "buffer_dtypes"),
        ):
            count = attestation[count_key]
            devices = attestation[devices_key]
            dtypes = attestation[dtypes_key]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or (count <= 0 if label == "parameter" else count < 0)
                or not isinstance(devices, Mapping)
                or not isinstance(dtypes, Mapping)
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in devices.values())
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in dtypes.values())
                or sum(devices.values()) != count
                or sum(dtypes.values()) != count
                or any(key != "cuda:0" for key in devices)
                or (label == "parameter" and any(key != "torch.bfloat16" for key in dtypes))
                or ((not devices or not dtypes) if count else (devices or dtypes))
            ):
                raise ValueError(f"condition {condition_id} {label} attestation counts are invalid")
        if (
            attestation["accelerator_backend"] != "nvidia-cuda"
            or not isinstance(attestation["torch_cuda"], str)
            or not attestation["torch_cuda"].strip()
            or attestation["torch_hip"] is not None
        ):
            raise ValueError(f"condition {condition_id} prediction lacks NVIDIA CUDA attestation")
        peak = raw.get("peak_memory_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
            raise ValueError(f"condition {condition_id} prediction peak memory is invalid")
        for field in ("elapsed_seconds", "model_load_seconds"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"condition {condition_id} prediction {field} is invalid")
        if raw.get("question_type") not in EXPECTED_QUESTION_TYPES:
            raise ValueError(f"condition {condition_id} question_type is invalid")
        if raw.get("error_type") is not None or raw.get("error_message") is not None:
            raise ValueError(f"condition {condition_id} passed prediction carries an error")
        if raw.get("status") != "passed" or not isinstance(raw.get("answer"), str) or not raw["answer"].strip():
            raise ValueError(f"condition {condition_id} has an incomplete prediction row")
        selected.append({"condition_id": condition_id, "video_id": video_id, "answer": raw["answer"]})
    order = [row["video_id"] for row in selected]
    if order != sorted(order) or sorted_video_id_set_sha256(order) != expected_id_digest:
        raise ValueError(f"condition {condition_id} canonical ID order/digest mismatch")
    evidence_marker_path = payload_root / "evidence" / "evidence-marker.json"
    evidence_marker = _load_json(evidence_marker_path)
    if (
        not isinstance(evidence_marker, Mapping)
        or evidence_marker.get("schema_version") != 1
        or evidence_marker.get("label_policy") != "label-free-evidence"
        or evidence_marker.get("example_count") != expected_rows
        or evidence_marker.get("canonical_id_digest") != expected_id_digest
        or evidence_marker.get("checksum_file") != "SHA256SUMS"
        or not isinstance(evidence_marker.get("evidence_bundle_id"), str)
        or not evidence_marker.get("evidence_bundle_id", "").strip()
    ):
        raise ValueError(f"condition {condition_id} evidence marker is invalid")
    if expected_rows == 475 and evidence_marker.get("manifest_sha256") != EXPECTED_FULL_MANIFEST_SHA256:
        raise ValueError(f"condition {condition_id} evidence manifest identity is not pinned")
    runtime = _load_json(payload_root / "runtime.json")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {
            "runtime",
            "runtime_image",
            "transformers_version",
            "transformers_revision",
            "torch_version",
            "torch_cuda",
            "torch_hip",
            "cuda_device_name",
            "cuda_bf16_supported",
            "dependencies",
        }
        or runtime.get("runtime") != raw_rows[0].get("runtime")
        or runtime.get("runtime_image") != EXPECTED_RUNTIME_CONTROLS["runtime_image"]
        or runtime.get("transformers_version") != "5.15.0.dev0"
        or runtime.get("transformers_revision") != EXPECTED_RUNTIME_CONTROLS["transformers_revision"]
        or not isinstance(runtime.get("torch_version"), str)
        or not runtime.get("torch_version", "").strip()
        or runtime.get("cuda_bf16_supported") is not True
        or not isinstance(runtime.get("torch_cuda"), str)
        or not runtime.get("torch_cuda", "").strip()
        or runtime.get("torch_hip") is not None
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime.get("cuda_device_name", "").strip()
        or not isinstance(runtime.get("dependencies"), Mapping)
        or dict(runtime["dependencies"]) != EXPECTED_RUNTIME_DEPENDENCIES
    ):
        raise ValueError(f"condition {condition_id} runtime attestation or dependency versions are invalid")
    for raw in raw_rows:
        if raw["runtime"] != runtime["runtime"] or raw["runtime_image"] != runtime["runtime_image"]:
            raise ValueError(f"condition {condition_id} prediction/runtime identity mismatch")
        if raw["runtime_attestation"]["torch_cuda"] != runtime["torch_cuda"]:
            raise ValueError(f"condition {condition_id} prediction CUDA identity mismatch")
    operational = _read_operational_evidence(
        condition_root,
        condition_id=condition_id,
        expected_rows=expected_rows,
        prediction_rows=raw_rows,
    )
    nested_evidence_identity: tuple[str, str, str, str]
    try:
        nested_checksum, evidence_bytes, marker_sha = _verify_evidence_checksum(
            payload_root, evidence_marker_path, checksum_entries
        )
        nested_evidence_identity = (
            marker_sha,
            nested_checksum,
            evidence_bytes,
            str(evidence_marker.get("evidence_bundle_id")),
        )
    except ValueError:
        if shared_evidence_identity is None:
            raise
        if shared_evidence_identity[0] != sha256_path(evidence_marker_path):
            raise ValueError(f"condition {condition_id} evidence marker does not match external evidence")
        evidence_root = payload_root / "evidence"
        local_evidence_files = sorted(
            path.relative_to(evidence_root).as_posix()
            for path in evidence_root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
        )
        if _evidence_bytes_digest(evidence_root, local_evidence_files) != shared_evidence_identity[2]:
            raise ValueError(f"condition {condition_id} evidence bytes do not match external evidence")
        nested_evidence_identity = shared_evidence_identity
    return {
        "rows": selected,
        "prediction_sha256": sha256_path(prediction_path),
        "prediction_path": prediction_path,
        "evidence_identity": (
            *nested_evidence_identity,
        ),
        "evidence_marker_sha256": sha256_path(evidence_marker_path),
        "evidence_checksum_sha256": nested_evidence_identity[1],
        "evidence_bytes_sha256": nested_evidence_identity[2],
        "operational": operational,
        "observed_identity": {
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "answer_policy": identity["answer_policy"],
            "min_pixels": identity["applied_min_pixels_per_frame"],
            "max_pixels": identity["applied_max_pixels_per_frame"],
            "request_source": identity["request_source"],
            "runtime_image_sha256": hashlib.sha256(identity["runtime_image"].encode()).hexdigest(),
        },
        "checksum_sha256": checksum_sha256,
        "checksum_entries": checksum_entries,
    }


def _load_evidence_bundle_identity(
    root: Path, *, expected_rows: int, expected_id_digest: str
) -> tuple[str, str, str, str]:
    root = root.resolve()
    if not root.is_dir() or not (root / "payload").is_dir() or not (root / "terminal-status.json").is_file():
        raise ValueError("evidence bundle must be the downloaded root containing terminal-status.json beside payload/")
    payload_root = root / "payload"
    checksum_sha256, checksum_entries = _verify_checksum_file(payload_root)
    terminal = _load_json(root / "terminal-status.json")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("payload_sha256") != checksum_sha256
        or not isinstance(terminal.get("run_name"), str)
        or not isinstance(terminal.get("payload_uri"), str)
    ):
        raise ValueError("evidence bundle terminal marker is not verified")
    status = _load_json(payload_root / "status.json")
    if (
        not isinstance(status, Mapping)
        or status.get("schema_version") != 1
        or status.get("status") != "passed"
        or status.get("mode") != "evidence"
        or status.get("run_name") != terminal.get("run_name")
        or status.get("expected_count") != expected_rows
        or status.get("frame_count") != EXPECTED_RUNTIME_CONTROLS["frame_count"]
        or status.get("workload_returncode") != 0
        or status.get("remote_payload_sync") != "frozen"
        or status.get("labels_joined") is not False
    ):
        raise ValueError("evidence bundle status marker is invalid")
    marker_path = payload_root / "evidence" / "evidence-marker.json"
    if not marker_path.is_file():
        raise ValueError("evidence bundle marker is missing")
    marker = _load_json(marker_path)
    if (
        not isinstance(marker, Mapping)
        or marker.get("schema_version") != 1
        or marker.get("label_policy") != "label-free-evidence"
        or marker.get("example_count") != expected_rows
        or marker.get("canonical_id_digest") != expected_id_digest
        or marker.get("checksum_file") != "SHA256SUMS"
        or not isinstance(marker.get("evidence_bundle_id"), str)
        or not marker.get("evidence_bundle_id", "").strip()
        or (expected_rows == 475 and marker.get("manifest_sha256") != EXPECTED_FULL_MANIFEST_SHA256)
    ):
        raise ValueError("evidence bundle marker does not match the canonical panel")
    evidence_files = {
        key.removeprefix("evidence/")
        for key in checksum_entries
        if key.startswith("evidence/") and key != "evidence/SHA256SUMS"
    }
    actual_evidence_files = {
        path.relative_to(payload_root / "evidence").as_posix()
        for path in (payload_root / "evidence").rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    }
    if evidence_files != actual_evidence_files:
        raise ValueError("evidence bundle root manifest does not cover exact evidence bytes")
    if checksum_entries.get("evidence/evidence-marker.json") != sha256_path(marker_path):
        raise ValueError("evidence bundle marker is not root-bound")
    return (
        sha256_path(marker_path),
        checksum_sha256,
        _evidence_bytes_digest(payload_root / "evidence", sorted(actual_evidence_files)),
        str(marker.get("evidence_bundle_id")),
    )


def _validate_runtime_metadata(
    condition_id: str,
    metadata: Mapping[str, Any],
    profile_condition: Mapping[str, Any],
) -> None:
    expected = {
        "condition_id": condition_id,
        "model_id": profile_condition["model_id"],
        "revision": profile_condition["revision"],
        "model_revision": profile_condition["revision"],
        "answer_policy": profile_condition["answer_policy"],
        "min_pixels": profile_condition["min_pixels"],
        "max_pixels": profile_condition["max_pixels"],
        "request_source": profile_condition["request_source"],
        **EXPECTED_RUNTIME_CONTROLS,
    }
    aliases = {"model_revision": "revision"}
    for raw_key, value in metadata.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key in expected and value != expected[key]:
            raise ValueError(f"condition {condition_id} runtime/model binding mismatch: {raw_key}")


def _reject_label_fields(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized_key in {
                re.sub(r"[^a-z0-9]", "", label)
                for label in _LABEL_KEYS
            } or normalized_key in {
                "question",
                "questions",
                "reference",
                "references",
                "prediction",
                "predictions",
                "answertext",
                "transcript",
                "transcripts",
                "cachequery",
                "querytext",
            }:
                raise ValueError(f"Prediction contains label-bearing field at {location}.{key}")
            _reject_label_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_label_fields(child, f"{location}[{index}]")


def _load_prediction_payload(path: Path) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read prediction input: {path}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid prediction JSON or JSONL: {path}") from error
    _reject_label_fields(payload)
    metadata: Mapping[str, Any] = {}
    if isinstance(payload, Mapping):
        rows = payload.get("predictions", payload.get("rows"))
        metadata_parts = []
        for key in ("metadata", "condition", "runtime"):
            part = payload.get(key)
            if part is not None:
                if not isinstance(part, Mapping):
                    raise ValueError("prediction metadata must be an object")
                metadata_parts.append(part)
        merged_metadata: dict[str, Any] = {}
        for part in metadata_parts:
            merged_metadata.update(part)
        metadata = merged_metadata
        payload = rows
    if not isinstance(payload, list) or not payload or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"Predictions must be a non-empty list of objects: {path}")
    return [dict(row) for row in payload], metadata


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_id: str,
    profile_condition: Mapping[str, Any],
    expected_rows: int,
) -> list[dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(
            f"Condition {condition_id} has {len(rows)} prediction rows; expected {expected_rows}"
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        _reject_label_fields(raw, f"$[{index}]")
        video_id = raw.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Prediction row {index} has an invalid video_id")
        video_id = video_id.strip()
        if video_id in seen:
            raise ValueError(f"Duplicate prediction ID in {condition_id}: {video_id}")
        seen.add(video_id)
        if raw.get("condition_id") not in (None, condition_id):
            raise ValueError(f"Prediction condition mismatch for {video_id}")
        _validate_runtime_metadata(condition_id, raw, profile_condition)
        answer = raw.get("answer")
        if raw.get("status") != "passed" or not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Condition {condition_id} is incomplete at {video_id}; completed {len(selected)}/{expected_rows}"
            )
        selected.append({"condition_id": condition_id, "video_id": video_id, "answer": answer})
    return selected


def _normalization_map(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"video_id": str(row["video_id"]), "answer": normalize_nist_text(str(row["answer"]))}
        for row in sorted(rows, key=lambda row: str(row["video_id"]))
    ]


def normalized_answer_map_sha256(value: Path | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(value, Path):
        rows, _metadata = _load_prediction_payload(value)
    else:
        rows = list(value)
    return hashlib.sha256(_canonical_json(_normalization_map(rows))).hexdigest()


def _cache_identity(profile_digest: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "scorer": "vqa-factorial-semantic-v2",
                "profile_sha256": profile_digest,
                "normalization": "strip, lowercase, collapse whitespace",
                "endpoint": STS_ENDPOINT,
                "timeout_seconds": EXPECTED_STS_TIMEOUT,
                "max_attempts": EXPECTED_STS_RETRIES,
            }
        )
    ).hexdigest()


def _sts_pair_key(identity: str, prediction: str, reference: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "identity": identity,
                "prediction": prediction,
                "reference": reference,
            }
        )
    ).hexdigest()


def _mean(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    if not rows:
        raise ValueError("Cannot aggregate an empty row set")
    values = [_finite(row[field_name], field=field_name) for row in rows]
    return statistics.fmean(values)


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, include_strata: bool = True
) -> dict[str, Any]:
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["stratum"])].append(row)
    result: dict[str, Any] = {
        "row_count": len(rows),
        "completed_count": len(rows),
        "completion_rate": 1.0 if rows else 0.0,
        "mean_answer_token_count": statistics.fmean(row["answer_token_count"] for row in rows),
        "ndcg": _mean(rows, "ndcg"),
        "ndcg_degenerate": all(bool(row["ndcg_degenerate"]) for row in rows),
    }
    for field_name in ("normalized_exact_match", "token_f1", *METRIC_FIELDS):
        result[field_name] = _mean(rows, field_name)
    if include_strata:
        result["strata"] = {
            name: _aggregate_rows(group, include_strata=False)
            for name, group in sorted(strata.items())
        }
    return result


def _bootstrap_intervals(
    vectors: Mapping[str, Sequence[float]],
    *,
    iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    seed: int = EXPECTED_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    if seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"Bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if iterations <= 0 or not 0 < confidence < 1:
        raise ValueError("Bootstrap configuration is invalid")
    lengths = {len(values) for values in vectors.values()}
    if not lengths or lengths == {0} or len(lengths) != 1:
        raise ValueError("Bootstrap vectors must have the same non-zero length")
    size = next(iter(lengths))
    source = random.Random(seed)
    sampled: dict[str, list[float]] = {field_name: [] for field_name in vectors}
    for _ in range(iterations):
        indexes = source.choices(range(size), k=size)
        for field_name, values in vectors.items():
            sampled[field_name].append(statistics.fmean(values[index] for index in indexes))
    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, Any]] = {}
    for field_name, values in sampled.items():
        ordered = sorted(values)
        lower_index = min(iterations - 1, int(tail * iterations))
        upper_index = min(iterations - 1, max(0, int((1.0 - tail) * iterations)))
        result[field_name] = {
            "lower": ordered[lower_index],
            "upper": ordered[upper_index],
            "confidence": confidence,
            "iterations": iterations,
            "seed": seed,
            "exploratory": True,
        }
    return result


def _effect_rows(condition_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_condition = {
        condition_id: {str(row["video_id"]): row for row in rows}
        for condition_id, rows in condition_rows.items()
    }
    if set(by_condition) != set(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial effects require the four exact conditions")
    ids = set(by_condition[EXPECTED_CONDITION_IDS[0]])
    if any(set(by_condition[condition_id]) != ids for condition_id in EXPECTED_CONDITION_IDS):
        raise ValueError("factorial effect rows must have one aligned ID set")
    effects = {"model": [], "prompt": [], "interaction": []}
    for video_id in sorted(ids):
        rows = {condition_id: by_condition[condition_id][video_id] for condition_id in EXPECTED_CONDITION_IDS}
        stratum = rows["4b-current"].get("stratum", rows["4b-current"].get("question_stratum"))
        for condition_id, row in rows.items():
            if row.get("stratum", row.get("question_stratum")) != stratum:
                raise ValueError(f"factorial strata do not align for {video_id}/{condition_id}")
        for effect_name in effects:
            effect_row: dict[str, Any] = {
                "video_id": video_id,
                "stratum": stratum,
            }
            for field_name in METRIC_FIELDS:
                current = rows["4b-current"][field_name]
                question_4b = rows["4b-question-aware"][field_name]
                current_8b = rows["8b-current"][field_name]
                question_8b = rows["8b-question-aware"][field_name]
                if effect_name == "model":
                    value = ((current_8b + question_8b) / 2) - ((current + question_4b) / 2)
                elif effect_name == "prompt":
                    value = ((question_4b + question_8b) / 2) - ((current + current_8b) / 2)
                else:
                    value = question_8b - current_8b - question_4b + current
                effect_row[field_name] = _finite(value, field=field_name)
            effects[effect_name].append(effect_row)
    return effects


def _effect_summary(rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    summary = {field_name: statistics.fmean(row[field_name] for row in rows) for field_name in METRIC_FIELDS}
    summary.update(
        {
            "paired_count": len(rows),
            "bootstrap_intervals": _bootstrap_intervals(
                {field_name: [row[field_name] for row in rows] for field_name in METRIC_FIELDS},
                iterations=iterations,
                seed=seed,
            ),
        }
    )
    return summary


def compute_factorial_effects(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    effects = _effect_rows(condition_rows)
    result: dict[str, Any] = {}
    for effect_name, rows in effects.items():
        result[effect_name] = {
            "overall": _effect_summary(rows, iterations=bootstrap_iterations, seed=bootstrap_seed),
            "strata": {
                stratum: _effect_summary(
                    [row for row in rows if row["stratum"] == stratum],
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                for stratum in sorted({str(row["stratum"]) for row in rows})
            },
        }
    return result


def _paired_rows(
    baseline_rows: Sequence[Mapping[str, Any]], treatment_rows: Sequence[Mapping[str, Any]], condition_id: str
) -> list[dict[str, Any]]:
    baseline = {str(row["video_id"]): row for row in baseline_rows}
    treatment = {str(row["video_id"]): row for row in treatment_rows}
    if set(baseline) != set(treatment):
        raise ValueError("paired conditions do not have identical video ID sets")
    result = []
    for video_id in sorted(baseline):
        if baseline[video_id]["stratum"] != treatment[video_id]["stratum"]:
            raise ValueError(f"paired strata differ for {video_id}")
        result.append(
            {
                "condition_id": condition_id,
                "video_id": video_id,
                "stratum": baseline[video_id]["stratum"],
                **{
                    field_name: treatment[video_id][field_name] - baseline[video_id][field_name]
                    for field_name in METRIC_FIELDS
                },
            }
        )
    return result


def _paired_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    treatment_id: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = _paired_rows(baseline_rows, treatment_rows, treatment_id)
    return {
        "paired_count": len(rows),
        "metric_deltas": {field_name: statistics.fmean(row[field_name] for row in rows) for field_name in METRIC_FIELDS},
        "bootstrap_intervals": _bootstrap_intervals(
            {field_name: [row[field_name] for row in rows] for field_name in METRIC_FIELDS},
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "rows": rows,
        "ndcg_degenerate": True,
        "excluded_metrics": {"ndcg": "single-response value is degenerate"},
    }


def promotion_decision(values: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "delta_C>=0.015": float(values["delta_C"]) >= 0.015,
        "lower_CI_C>0": float(values["lower_CI_C"]) > 0,
        "delta_METEOR>0": float(values["delta_METEOR"]) > 0,
        "delta_BERTScore>0": float(values["delta_BERTScore"]) > 0,
        "delta_STS>=-0.005": float(values["delta_STS"]) >= -0.005,
        "completed=475": values["completed"] == 475,
        "integrity": values["integrity"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "promote" if not failed else "stop",
        "promote": not failed,
        "failed_gates": failed,
        "gates": checks,
    }


def _sanitized_row(row: Mapping[str, Any], *, condition_id: str | None = None) -> dict[str, Any]:
    field_names = (
        "condition_id",
        "video_id",
        "stratum",
        "normalized_exact_match",
        "token_f1",
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
        "ndcg_degenerate",
    )
    result = {field_name: row[field_name] for field_name in field_names}
    if condition_id is not None:
        result["condition_id"] = condition_id
    for field_name in field_names:
        if field_name not in {"condition_id", "video_id", "stratum", "ndcg_degenerate"}:
            result[field_name] = _finite(result[field_name], field=field_name)
    if not isinstance(result["video_id"], str) or not result["video_id"]:
        raise ValueError("sanitized row has an invalid video ID")
    return result


def _baseline_sanitized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read the older Task 1 row schema without importing any text."""

    stratum = row.get("stratum", row.get("question_stratum"))
    if not isinstance(stratum, str) or not stratum:
        raise ValueError("baseline row has no stratum")
    result = {
        "condition_id": "4b-current",
        "video_id": row["video_id"],
        "stratum": stratum,
        "normalized_exact_match": row.get("normalized_exact_match"),
        "token_f1": row.get("token_f1"),
        "answer_token_count": row["answer_token_count"],
        "meteor": row["meteor"],
        "bertscore_f1": row["bertscore_f1"],
        "sts": row["sts"],
        "semantic_composite": row["semantic_composite"],
        "ndcg": row["ndcg"],
        "ndcg_degenerate": row["ndcg_degenerate"],
    }
    for field_name in (
        "normalized_exact_match",
        "token_f1",
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
    ):
        result[field_name] = _finite(result[field_name], field=field_name)
    if not isinstance(result["video_id"], str) or not result["video_id"]:
        raise ValueError("baseline row has an invalid video ID")
    return result


def _sanitize_rows(condition_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        condition_id: [_sanitized_row(row, condition_id=condition_id) for row in rows]
        for condition_id, rows in condition_rows.items()
    }


def _baseline_scorer_binding(scorer_payload: Mapping[str, Any]) -> bool:
    required = {
        "normalization": "strip, lowercase, collapse whitespace",
        "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
        "bert_score_module_version": EXPECTED_BERT_SCORE_MODULE_VERSION,
        "bert_num_layers": 17,
        "bert_idf": False,
        "bert_rescale_with_baseline": False,
        "sts_endpoint": STS_ENDPOINT,
        "sts_timeout_seconds": EXPECTED_STS_TIMEOUT,
        "sts_max_attempts": EXPECTED_STS_RETRIES,
    }
    return all(scorer_payload.get(field_name) == expected for field_name, expected in required.items())


def baseline_content_sha256(payload: Mapping[str, Any]) -> str:
    candidate = copy.deepcopy(dict(payload))
    inputs = candidate.get("inputs")
    if isinstance(inputs, Mapping):
        inputs = dict(inputs)
        inputs.pop("artifact_content_sha256", None)
        candidate["inputs"] = inputs
    return hashlib.sha256(_canonical_json(candidate)).hexdigest()


def _validate_baseline_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    source_prediction_path: Path | None,
    panel_identity: Mapping[str, Any],
    annotation_hash: str,
    bert_model: Path,
    profile_digest: str,
    current_answer_map_hash: str,
    expected_rows: int,
    expected_ids: Sequence[str],
) -> tuple[bool, str, list[dict[str, Any]] | None]:
    if sha256_path(artifact_path) != EXPECTED_BASELINE_ARTIFACT_SHA256:
        return False, "binding mismatch: pinned baseline artifact hash", None
    if source_prediction_path is None:
        return False, "binding mismatch: explicit baseline source predictions are required", None
    if sha256_path(source_prediction_path) != EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256:
        return False, "binding mismatch: pinned baseline source prediction hash", None
    try:
        source_rows, _source_metadata = _load_prediction_payload(source_prediction_path)
    except (OSError, ValueError, TypeError) as error:
        return False, f"binding mismatch: baseline source schema ({type(error).__name__})", None
    source_map_hash = normalized_answer_map_sha256(source_rows)
    if source_map_hash != current_answer_map_hash:
        return False, "binding mismatch: normalized baseline source answer map", None
    checks: list[tuple[str, bool]] = []
    checks.append(("schema", payload.get("schema_version") == 1))
    checks.append(("panel", payload.get("panel", {}).get("identity") == panel_identity))
    inputs = payload.get("inputs", {})
    checks.extend(
        [
            ("annotation", inputs.get("annotation_root_sha256") == annotation_hash),
            ("bert snapshot", inputs.get("bert_model_snapshot") == EXPECTED_BERT_MODEL_SNAPSHOT),
            (
                "bert hash",
                inputs.get("bert_model_snapshot_sha256") == sha256_path(bert_model),
            ),
            ("scorer", inputs.get("scorer_sha256") == EXPECTED_BASELINE_SCORER_SHA256),
            (
                "config",
                inputs.get("profile_sha256", inputs.get("config_sha256")) == profile_digest,
            ),
            ("answer-map", inputs.get("normalized_answer_map_sha256") == current_answer_map_hash),
            ("scorer configuration", _baseline_scorer_binding(payload.get("scorer", {}))),
            ("source prediction", inputs.get("source_prediction_sha256") == EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256),
        ]
    )
    if not all(passed for _name, passed in checks):
        failed = next(name for name, passed in checks if not passed)
        return False, f"binding mismatch: {failed}", None
    rows_payload = payload.get("rows")
    if not isinstance(rows_payload, Mapping):
        return False, "binding mismatch: baseline rows", None
    if set(rows_payload) != {EXPECTED_BASELINE_ROW_KEY}:
        return False, "binding mismatch: baseline 4b-current rows", None
    candidate_rows = rows_payload.get(EXPECTED_BASELINE_ROW_KEY)
    if not isinstance(candidate_rows, list) or len(candidate_rows) != expected_rows:
        return False, "binding mismatch: baseline 4b-current rows", None
    try:
        sanitized = [_baseline_sanitized_row(row) for row in candidate_rows]
    except (KeyError, TypeError, ValueError):
        return False, "binding mismatch: baseline row schema", None
    by_video_id = {row["video_id"]: row for row in sanitized}
    if len(by_video_id) != expected_rows or set(by_video_id) != set(expected_ids):
        return False, "binding mismatch: baseline row ID set", None
    return True, "all bindings matched", [by_video_id[video_id] for video_id in expected_ids]


def _score_unique_pairs(
    condition_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    annotations: Mapping[str, Mapping[str, Any]],
    *,
    bert_model: Path,
    meteor_fn: MetricFunction,
    bertscore_fn: BertScoreFunction,
    sts_fn: MetricFunction,
    sts_cache: STSCache,
    sts_identity: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    pairs: set[tuple[str, str]] = set()
    normalized_by_condition: dict[str, list[tuple[str, str, str]]] = {}
    for condition_id, predictions in condition_predictions.items():
        entries = []
        for prediction in predictions:
            video_id = str(prediction["video_id"])
            if video_id not in annotations:
                raise ValueError(f"Missing reference: prediction ID not present in declared annotation set: {video_id}")
            normalized_prediction = normalize_nist_text(str(prediction["answer"]))
            normalized_reference = normalize_nist_text(str(annotations[video_id]["reference"]))
            pair = (normalized_prediction, normalized_reference)
            pairs.add(pair)
            entries.append((video_id, normalized_prediction, normalized_reference))
        normalized_by_condition[condition_id] = entries
    ordered_pairs = sorted(pairs)
    meteor_values: dict[tuple[str, str], float] = {}
    for prediction, reference in ordered_pairs:
        meteor_values[(prediction, reference)] = _finite(
            meteor_fn(prediction, reference), field="meteor"
        )
    bert_predictions = [pair[0] for pair in ordered_pairs]
    bert_references = [pair[1] for pair in ordered_pairs]
    bert_values = list(bertscore_fn(bert_predictions, bert_references, bert_model))
    if len(bert_values) != len(ordered_pairs):
        raise ValueError(f"BERTScore returned {len(bert_values)} rows for {len(ordered_pairs)} unique pairs")
    bert_map = {
        pair: _finite(value, field="bertscore_f1")
        for pair, value in zip(ordered_pairs, bert_values, strict=True)
    }
    sts_values: dict[tuple[str, str], float] = {}
    hit_count = 0
    miss_count = 0
    live_request_count = 0
    for prediction, reference in ordered_pairs:
        key = _sts_pair_key(sts_identity, prediction, reference)
        if key in sts_cache.values:
            value = _finite(sts_cache.values[key], field="sts")
            hit_count += 1
        else:
            miss_count += 1
            live_request_count += 1
            value = _finite(sts_fn(prediction, reference), field="sts")
            sts_cache.values[key] = value
        sts_values[(prediction, reference)] = value
    rows: dict[str, list[dict[str, Any]]] = {}
    for condition_id, entries in normalized_by_condition.items():
        condition_rows = []
        for video_id, prediction, reference in entries:
            annotation = annotations[video_id]
            metrics = {
                "meteor": meteor_values[(prediction, reference)],
                "bertscore_f1": bert_map[(prediction, reference)],
                "sts": sts_values[(prediction, reference)],
            }
            composite = semantic_composite(metrics)
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "video_id": video_id,
                    "stratum": annotation["question_stratum"],
                    "normalized_exact_match": normalized_exact_match(prediction, reference),
                    "token_f1": token_f1(prediction, reference),
                    "answer_token_count": len(prediction.split()),
                    **metrics,
                    "semantic_composite": composite,
                    "ndcg": 1.0,
                    "ndcg_degenerate": True,
                }
            )
        rows[condition_id] = condition_rows
    return rows, {
        "unique_pair_count": len(ordered_pairs),
        "meteor_pair_count": len(ordered_pairs),
        "bertscore_pair_count": len(ordered_pairs),
        "sts_pair_count": len(ordered_pairs),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "live_request_count": live_request_count,
    }


def _decision_for_factorial(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    paired: Mapping[str, Mapping[str, Any]],
    effects: Mapping[str, Any],
    *,
    integrity: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    treatments: dict[str, Any] = {}
    for condition_id in EXPECTED_CONDITION_IDS[1:]:
        summary = paired[condition_id]
        deltas = summary["metric_deltas"]
        decision = promotion_decision(
            {
                "delta_C": deltas["semantic_composite"],
                "lower_CI_C": summary["bootstrap_intervals"]["semantic_composite"]["lower"],
                "delta_METEOR": deltas["meteor"],
                "delta_BERTScore": deltas["bertscore_f1"],
                "delta_STS": deltas["sts"],
                "completed": len(condition_rows[condition_id]),
                "integrity": integrity,
            }
        )
        treatments[condition_id] = {
            **decision,
            "delta_C": deltas["semantic_composite"],
            "lower_CI_C": summary["bootstrap_intervals"]["semantic_composite"]["lower"],
            "delta_METEOR": deltas["meteor"],
            "delta_BERTScore": deltas["bertscore_f1"],
            "delta_STS": deltas["sts"],
            "completed": len(condition_rows[condition_id]),
        }
    model_rows = _effect_rows(condition_rows)["model"]
    prompt_rows = _effect_rows(condition_rows)["prompt"]
    comparison = _bootstrap_intervals(
        {
            "model_minus_prompt": [
                model_row["semantic_composite"] - prompt_row["semantic_composite"]
                for model_row, prompt_row in zip(model_rows, prompt_rows, strict=True)
            ],
            "prompt_minus_model": [
                prompt_row["semantic_composite"] - model_row["semantic_composite"]
                for model_row, prompt_row in zip(model_rows, prompt_rows, strict=True)
            ],
        },
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    model_effect = effects["model"]["overall"]["semantic_composite"]
    prompt_effect = effects["prompt"]["overall"]["semantic_composite"]
    if model_effect >= 0.020 and comparison["model_minus_prompt"]["lower"] > 0:
        dominance = "model"
    elif prompt_effect >= 0.020 and comparison["prompt_minus_model"]["lower"] > 0:
        dominance = "prompt"
    else:
        dominance = "mixed_bottleneck"
    return {
        "treatments": treatments,
        "dominance": dominance,
        "dominance_threshold": 0.020,
        "dominance_comparison": comparison,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "injected-or-runtime-provided"


def _artifact_safe(value: object, location: str = "$") -> None:
    if location == "$" and isinstance(value, Mapping):
        expected_top_level = {
            "schema_version",
            "experiment",
            "panel",
            "scorer",
            "baseline_binding",
            "conditions",
            "rows",
            "paired_vs_baseline",
            "factorial_effects",
            "decision",
            "integrity",
            "inputs",
            "dependencies",
            "cache_summary",
        }
        if set(value) != expected_top_level:
            raise ValueError("artifact does not match the positive top-level schema")
    forbidden_tokens = {
        "answer",
        "reference",
        "question",
        "transcript",
        "prediction",
        "cachepath",
        "cachequery",
        "querytext",
        "query",
        "uri",
        "path",
        "filepath",
        "raw",
    }
    allowed_key_exceptions = {
        "cachesummary",
        "conditioninputsha256",
        "runtimeimagesha256",
        "evidencemarkersha256",
        "evidencechecksumsha256",
        "stsendpointsha256",
        "answertokencount",
        "meananswertokencount",
        "answerpolicy",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = re.sub(r"[^a-z0-9]", "", key_text.lower())
            condition_key_location = any(
                marker in location
                for marker in (
                    "$.conditions",
                    "$.rows",
                    "$.panel.completed_per_condition",
                    "$.paired_vs_baseline",
                    "$.decision.treatments",
                    "$.inputs.condition_input_sha256",
                    "$.inputs.condition_bundle_checksum_sha256",
                )
            )
            if (
                normalized not in allowed_key_exceptions
                and not (key_text in EXPECTED_CONDITION_IDS and condition_key_location)
                and not (location == "$.integrity" and "prediction" in normalized)
                and normalized in {"file", "files"}
            ):
                raise ValueError(f"sanitized artifact contains forbidden field at {location}.{key_text}")
            if (
                normalized not in allowed_key_exceptions
                and not (key_text in EXPECTED_CONDITION_IDS and condition_key_location)
                and not (location == "$.integrity" and "prediction" in normalized)
                and any(
                token in normalized for token in forbidden_tokens
                )
            ):
                raise ValueError(f"sanitized artifact contains forbidden field at {location}.{key_text}")
            if normalized in {"videoid", "conditionid"} and not any(
                marker in location
                for marker in (
                    ".rows[",
                    "$.rows.",
                    ".paired_vs_baseline.",
                    ".factorial_effects.",
                )
            ):
                raise ValueError(f"dynamic ID is outside its schema location: {location}.{key_text}")
            _artifact_safe(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _artifact_safe(child, f"{location}[{index}]")
    elif isinstance(value, str):
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            raise ValueError(f"sanitized artifact contains a URI at {location}")
        parent_key = location.rsplit(".", 1)[-1].lower()
        if (
            "/" in value
            or "\\" in value
            or value.startswith(("/", "./", "../", "~/"))
        ) and parent_key not in {"model_id", "revision", "completion_denominator"}:
            raise ValueError(f"sanitized artifact contains a path-like value at {location}")
        if re.search(
            r"(?:^|[/\\])[^/\\]+\.(?:json|jsonl|csv|txt|pt|bin)$", value, re.IGNORECASE
        ):
            if parent_key not in {"model_id", "revision"}:
                raise ValueError(f"sanitized artifact contains a path-like value at {location}")


def score_factorial(
    *,
    annotation_root: Path,
    condition_paths: Mapping[str, Path] | None = None,
    condition_bundles: Mapping[str, Path] | None = None,
    evidence_bundle_path: Path | None = None,
    bert_model: Path,
    profile_path: Path | None = DEFAULT_PROFILE_PATH,
    profile: Mapping[str, Any] | None = None,
    panel_profile: PanelProfile | None = None,
    expected_rows: int | None = None,
    expected_annotation_files: int | None = None,
    baseline_artifact_path: Path | None = None,
    baseline_source_prediction_path: Path | None = None,
    rescore_baseline: bool = False,
    output_path: Path | None = None,
    meteor_fn: MetricFunction | None = None,
    bertscore_fn: BertScoreFunction | None = None,
    sts_fn: MetricFunction | None = None,
    sts_cache: STSCache | None = None,
    sts_cache_path: Path | None = None,
    bootstrap_iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, score, summarize, and optionally atomically publish the factorial."""

    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if bootstrap_seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"Bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if bootstrap_iterations != EXPECTED_BOOTSTRAP_ITERATIONS:
        raise ValueError(f"Bootstrap iterations must be {EXPECTED_BOOTSTRAP_ITERATIONS}")
    if not bert_model.is_dir() or bert_model.name != EXPECTED_BERT_MODEL_SNAPSHOT:
        raise ValueError(f"BERT model must be the pinned snapshot directory {EXPECTED_BERT_MODEL_SNAPSHOT}")
    if profile is None:
        if profile_path is None:
            profile_path = DEFAULT_PROFILE_PATH
        resolved_profile = load_profile(profile_path)
    else:
        resolved_profile = validate_factorial_profile(profile, strict=False)
    profile_digest = profile_sha256(resolved_profile)
    profile_conditions = _profile_conditions(resolved_profile)
    rows_expected = int(expected_rows if expected_rows is not None else resolved_profile["expected_count"])
    files_expected = int(
        expected_annotation_files
        if expected_annotation_files is not None
        else (panel_profile.expected_annotation_files if panel_profile else 481)
    )
    if rows_expected != int(resolved_profile["expected_count"]):
        raise ValueError("expected_rows does not match factorial profile")
    if condition_bundles is not None and condition_paths is not None:
        raise ValueError("provide condition bundles, not legacy condition paths")
    supplied_conditions = condition_bundles if condition_bundles is not None else condition_paths
    if supplied_conditions is None:
        raise ValueError("factorial scoring requires four condition bundles")
    if condition_bundles is None and rows_expected == 475:
        raise ValueError("production scoring requires --condition-bundle directories")
    if evidence_bundle_path is not None and condition_bundles is None:
        raise ValueError("--evidence-bundle requires production condition bundles")
    if set(supplied_conditions) != set(EXPECTED_CONDITION_IDS) or len(supplied_conditions) != len(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial scoring requires exactly the four condition IDs")
    supplied_conditions = {condition_id: Path(path) for condition_id, path in supplied_conditions.items()}
    resolved_panel = panel_profile or CANONICAL_PANEL_PROFILE
    if rows_expected != resolved_panel.expected_rows or files_expected != resolved_panel.expected_annotation_files:
        raise ValueError("panel profile dimensions do not match the declared factorial panel")
    expected_id_digest = resolved_panel.canonical_id_set_sha256
    if expected_rows == 475 and expected_id_digest != EXPECTED_FULL_CANONICAL_ID_DIGEST:
        raise ValueError("the 475-row panel digest is not the pinned canonical digest")
    # Freeze and validate all production bundle/prediction bytes before loading labels.
    frozen_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    prediction_paths: dict[str, Path] = {}
    condition_evidence: dict[str, dict[str, Any]] = {}
    frozen_order: list[str] | None = None
    external_identity = None
    if condition_bundles is not None and evidence_bundle_path is not None:
        external_identity = _load_evidence_bundle_identity(
            evidence_bundle_path,
            expected_rows=rows_expected,
            expected_id_digest=expected_id_digest,
        )
    for condition_id in EXPECTED_CONDITION_IDS:
        path = supplied_conditions[condition_id]
        if condition_bundles is not None:
            bundle = _load_condition_bundle(
                path,
                condition_id=condition_id,
                profile_condition=profile_conditions[condition_id],
                expected_rows=rows_expected,
                expected_id_digest=expected_id_digest,
                profile_run_name=str(resolved_profile["run_name"]),
                profile_remote_uri=str(resolved_profile["remote_uri"]),
                shared_evidence_identity=external_identity,
            )
            selected = bundle["rows"]
            prediction_hashes[condition_id] = bundle["prediction_sha256"]
            prediction_paths[condition_id] = bundle["prediction_path"].resolve()
            condition_evidence[condition_id] = bundle
        else:
            raw_rows, metadata = _load_prediction_payload(path)
            _validate_runtime_metadata(condition_id, metadata, profile_conditions[condition_id])
            selected = _validate_prediction_rows(
                raw_rows,
                condition_id=condition_id,
                profile_condition=profile_conditions[condition_id],
                expected_rows=rows_expected,
            )
            prediction_hashes[condition_id] = sha256_path(path)
        current_order = [row["video_id"] for row in selected]
        if frozen_order is None:
            frozen_order = current_order
        elif current_order != frozen_order:
            raise ValueError("Conditions do not have the same frozen prediction ID set/order")
        frozen_predictions[condition_id] = selected
    if frozen_order is None or len(frozen_order) != rows_expected:
        raise ValueError("No complete frozen factorial prediction panel")
    if sorted_video_id_set_sha256(frozen_order) != expected_id_digest:
        raise ValueError("frozen predictions do not match the canonical profile ID digest")
    if condition_bundles is not None:
        if len(set(prediction_hashes.values())) != len(prediction_hashes):
            raise ValueError("condition prediction payload hashes must be distinct")
        if len(set(prediction_paths.values())) != len(prediction_paths):
            raise ValueError("condition prediction payload paths must be distinct")
        evidence_identities = {bundle["evidence_identity"] for bundle in condition_evidence.values()}
        if len(evidence_identities) != 1:
            raise ValueError("all condition bundles must share one evidence marker/checksum identity")
        if external_identity is not None and external_identity != next(iter(evidence_identities)):
            raise ValueError("external evidence bundle identity does not match condition bundles")
    annotations = _load_annotations(annotation_root)
    if len(annotations) != files_expected:
        raise ValueError(f"Annotation root contains {len(annotations)} annotation files; expected {files_expected}")
    panel_identity = _validate_panel_identity(
        resolved_panel,
        annotation_ids=annotations,
        condition_ids={condition_id: (row["video_id"] for row in rows) for condition_id, rows in frozen_predictions.items()},
    )
    annotation_hash = sha256_path(annotation_root)
    bert_score_versions = task1._bert_score_versions(required=bertscore_fn is None)
    if bertscore_fn is not None:
        bert_score_versions = {
            "module": EXPECTED_BERT_SCORE_MODULE_VERSION,
            "distribution": "injected",
        }
    if bert_score_versions["module"] != EXPECTED_BERT_SCORE_MODULE_VERSION and bertscore_fn is None:
        raise ValueError("BERTScore module version is not pinned")
    meteor_fn = meteor_fn or task1._default_meteor
    bertscore_fn = bertscore_fn or task1._default_bertscore
    sts_fn = sts_fn or (lambda prediction, reference: umbc_sts_score(
        prediction,
        reference,
        timeout=EXPECTED_STS_TIMEOUT,
        retries=EXPECTED_STS_RETRIES,
    ))
    cache = sts_cache or STSCache()
    identity = _cache_identity(profile_digest)
    if cache.identity is None:
        cache.identity = identity
    elif cache.identity != identity:
        raise ValueError("STS cache identity does not match scorer/config/endpoint identity")
    current_answer_map_hash = normalized_answer_map_sha256(frozen_predictions["4b-current"])

    baseline_binding: dict[str, Any] = {
        "requested": baseline_artifact_path is not None,
        "reused": False,
        "reason": "not requested" if baseline_artifact_path is None else "rescore requested",
    }
    baseline_rows: list[dict[str, Any]] | None = None
    baseline_payload: Mapping[str, Any] | None = None
    baseline_hash: str | None = None
    if baseline_artifact_path is not None and not rescore_baseline:
        try:
            baseline_payload = _load_json(baseline_artifact_path)
            if not isinstance(baseline_payload, Mapping):
                raise ValueError("baseline payload is not an object")
            baseline_hash = sha256_path(baseline_artifact_path)
            valid, reason, baseline_rows = _validate_baseline_binding(
                baseline_payload,
                artifact_path=baseline_artifact_path,
                source_prediction_path=baseline_source_prediction_path,
                panel_identity=panel_identity,
                annotation_hash=annotation_hash,
                bert_model=bert_model,
                profile_digest=profile_digest,
                current_answer_map_hash=current_answer_map_hash,
                expected_rows=rows_expected,
                expected_ids=frozen_order,
            )
            baseline_binding.update({"reused": valid, "reason": reason, "artifact_sha256": baseline_hash})
        except (OSError, ValueError, KeyError, TypeError) as error:
            baseline_binding.update({"reused": False, "reason": f"binding mismatch: {type(error).__name__}: explicit rescore"})
    elif baseline_artifact_path is not None:
        baseline_binding.update({"reason": "forced rescore (--rescore-baseline)"})
    condition_predictions = {
        condition_id: frozen_predictions[condition_id]
        for condition_id in EXPECTED_CONDITION_IDS
        if condition_id != "4b-current" or baseline_rows is None
    }
    scored_rows, cache_counts = _score_unique_pairs(
        condition_predictions,
        annotations,
        bert_model=bert_model,
        meteor_fn=meteor_fn,
        bertscore_fn=bertscore_fn,
        sts_fn=sts_fn,
        sts_cache=cache,
        sts_identity=identity,
    )
    if baseline_rows is not None:
        scored_rows["4b-current"] = [dict(row) for row in baseline_rows]
    condition_rows = {condition_id: scored_rows[condition_id] for condition_id in EXPECTED_CONDITION_IDS}
    if any(len(rows) != rows_expected for rows in condition_rows.values()):
        raise ValueError("factorial completion gate failed after scoring")
    safe_rows = _sanitize_rows(condition_rows)
    aggregate = {}
    for condition_id in EXPECTED_CONDITION_IDS:
        condition_report = _aggregate_rows(condition_rows[condition_id])
        observed = condition_evidence.get(condition_id, {}).get("observed_identity")
        if observed is None:
            observed = {
                "model_id": profile_conditions[condition_id]["model_id"],
                "model_revision": profile_conditions[condition_id]["revision"],
                "answer_policy": profile_conditions[condition_id]["answer_policy"],
                "min_pixels": profile_conditions[condition_id]["min_pixels"],
                "max_pixels": profile_conditions[condition_id]["max_pixels"],
                "request_source": profile_conditions[condition_id]["request_source"],
                "runtime_image_sha256": hashlib.sha256(
                    EXPECTED_RUNTIME_CONTROLS["runtime_image"].encode()
                ).hexdigest(),
            }
        condition_report.update(observed)
        condition_report["runtime_controls_sha256"] = hashlib.sha256(
            _canonical_json(EXPECTED_RUNTIME_CONTROLS)
        ).hexdigest()
        if condition_id in condition_evidence:
            condition_report["bundle_checksum_sha256"] = condition_evidence[condition_id]["checksum_sha256"]
            condition_report["operational_evidence"] = condition_evidence[condition_id]["operational"]
            condition_report["evidence_marker_sha256"] = condition_evidence[condition_id]["evidence_marker_sha256"]
            condition_report["evidence_checksum_sha256"] = condition_evidence[condition_id]["evidence_checksum_sha256"]
        aggregate[condition_id] = condition_report
    paired = {"baseline_condition_id": "4b-current"}
    for condition_id in EXPECTED_CONDITION_IDS[1:]:
        paired[condition_id] = _paired_summary(
            condition_rows["4b-current"],
            condition_rows[condition_id],
            treatment_id=condition_id,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    effects = compute_factorial_effects(
        condition_rows,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    integrity = {
        "labels_joined_after_prediction_freeze": True,
        "all_conditions_complete": all(len(rows) == rows_expected for rows in condition_rows.values()),
        "completion_denominator": f"{rows_expected}/{rows_expected} per condition",
        "duplicate_prediction_ids_checked": True,
        "same_canonical_id_set_and_order_checked": True,
        "canonical_profile_digest_checked": True,
        "condition_bindings_checked": True,
        "finite_metrics_checked": True,
        "label_leakage_checked": True,
        "runtime_bundle_evidence_checked": condition_bundles is not None,
        "distinct_condition_payloads_checked": condition_bundles is not None,
        "operational_evidence_checked": condition_bundles is not None,
        "evidence_identity_checked": condition_bundles is not None,
    }
    integrity_ok = all(value is True for key, value in integrity.items() if key != "completion_denominator")
    decision = _decision_for_factorial(
        condition_rows,
        paired,
        effects,
        integrity=integrity_ok,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    experiment_name = str(resolved_profile["run_name"]).replace(
        "vqa-factorial-full-", "vqa-factorial-475-", 1
    )
    artifact: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "experiment": {
            "name": experiment_name,
            "design": "2x2 model-by-answer-policy factorial",
            "semantic_composite": "mean(meteor, bertscore_f1, sts)",
            "bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed, "confidence": 0.95, "exploratory": True},
        },
        "panel": {
            "collection": "declared annotation panel",
            "expected_annotation_files": files_expected,
            "expected_rows_per_condition": rows_expected,
            "declared_conditions": list(EXPECTED_CONDITION_IDS),
            "completed_per_condition": {condition_id: len(condition_rows[condition_id]) for condition_id in EXPECTED_CONDITION_IDS},
            "identity": panel_identity,
        },
        "scorer": {
            "name": "NIST 2025 semantic VQA factorial scorer",
            "normalization": "strip, lowercase, collapse whitespace",
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_score_module_version": bert_score_versions["module"],
            "bert_score_distribution_version": bert_score_versions["distribution"],
            "bert_num_layers": 17,
            "bert_idf": False,
            "bert_rescale_with_baseline": False,
            "sts_endpoint_sha256": hashlib.sha256(STS_ENDPOINT.encode()).hexdigest(),
            "sts_timeout_seconds": EXPECTED_STS_TIMEOUT,
            "sts_max_attempts": EXPECTED_STS_RETRIES,
            "single_response_ndcg": {"value": 1.0, "degenerate": True, "informative": False},
            "runtime_controls_sha256": hashlib.sha256(_canonical_json(EXPECTED_RUNTIME_CONTROLS)).hexdigest(),
        },
        "baseline_binding": baseline_binding,
        "conditions": aggregate,
        "rows": safe_rows,
        "paired_vs_baseline": paired,
        "factorial_effects": effects,
        "decision": decision,
        "integrity": integrity,
        "inputs": {
            "annotation_root_sha256": annotation_hash,
            "condition_input_sha256": {condition_id: prediction_hashes[condition_id] for condition_id in EXPECTED_CONDITION_IDS},
            "condition_bundle_checksum_sha256": {
                condition_id: condition_evidence[condition_id]["checksum_sha256"]
                for condition_id in EXPECTED_CONDITION_IDS
            } if condition_evidence else {},
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_model_snapshot_sha256": sha256_path(bert_model),
            "profile_sha256": profile_digest,
            "runtime_controls_sha256": hashlib.sha256(_canonical_json(EXPECTED_RUNTIME_CONTROLS)).hexdigest(),
            "scorer_sha256": sha256_path(Path(__file__)),
            "task1_scorer_sha256": sha256_path(TASK1_SCORER_PATH),
            "sts_endpoint_sha256": hashlib.sha256(STS_ENDPOINT.encode()).hexdigest(),
        },
        "dependencies": {
            "nltk_distribution": _distribution_version("nltk"),
            "bert_score_module": bert_score_versions["module"],
            "bert_score_distribution": bert_score_versions["distribution"],
        },
        "cache_summary": {"identity": identity, **cache_counts, "private": True},
    }
    if condition_evidence:
        common_evidence = next(iter(condition_evidence.values()))
        artifact["inputs"]["evidence_marker_sha256"] = common_evidence["evidence_marker_sha256"]
        artifact["inputs"]["evidence_checksum_sha256"] = common_evidence["evidence_checksum_sha256"]
        artifact["inputs"]["evidence_bytes_sha256"] = common_evidence["evidence_bytes_sha256"]
        wall_seconds = [
            condition_evidence[condition_id]["operational"]["wall_seconds"]
            for condition_id in EXPECTED_CONDITION_IDS
        ]
        total_rows = rows_expected * len(EXPECTED_CONDITION_IDS)
        artifact["experiment"]["operational_projection"] = {
            "parallel_wall_seconds": max(wall_seconds),
            "declared_wall_seconds": max(wall_seconds),
            "gpu_seconds": statistics.fsum(wall_seconds),
            "gpu_hours": statistics.fsum(wall_seconds) / 3600.0,
            "parallel_rows_per_second": total_rows / max(wall_seconds),
            "declared_rows_per_second": total_rows / max(wall_seconds),
            "serial_rows_per_second": total_rows / statistics.fsum(wall_seconds),
        }
    _artifact_safe(artifact)
    if sts_cache_path is not None:
        _write_private_cache(sts_cache_path, cache)
    if output_path is not None:
        atomic_write_json(output_path, artifact)
    return artifact


def _parse_conditions(values: Sequence[str]) -> dict[str, Path]:
    conditions: dict[str, Path] = {}
    for value in values:
        condition_id, separator, path = value.partition("=")
        if not separator or not condition_id.strip() or not path.strip():
            raise ValueError(f"Condition must be ID=PATH: {value}")
        if condition_id.strip() in conditions:
            raise ValueError(f"Duplicate condition ID: {condition_id.strip()}")
        conditions[condition_id.strip()] = Path(path)
    return conditions


def _load_private_cache(path: Path, identity: str) -> STSCache:
    payload = _load_json(path) if path.exists() else {"identity": identity, "values": {}}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values", {}), Mapping):
        raise ValueError("STS cache must be an object with private values")
    cache = STSCache(
        values={
            str(key): _finite(value, field="cached sts")
            for key, value in payload["values"].items()
        },
        identity=payload.get("identity"),
    )
    if cache.identity != identity:
        raise ValueError("STS cache identity does not match scorer/config/endpoint identity")
    return cache


def _write_private_cache(path: Path, cache: STSCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary cache already exists: {temporary}")
    payload = {"identity": cache.identity, "values": dict(sorted(cache.values.items()))}
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--condition-bundle", action="append", required=True, metavar="ID=DIR"
    )
    parser.add_argument("--evidence-bundle", type=Path)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--baseline-source-predictions", type=Path)
    parser.add_argument("--rescore-baseline", action="store_true")
    parser.add_argument("--sts-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    profile = load_profile(args.profile)
    identity = _cache_identity(profile_sha256(profile))
    cache = _load_private_cache(args.sts_cache, identity) if args.sts_cache else STSCache(identity=identity)
    artifact = score_factorial(
        annotation_root=args.annotations,
        condition_bundles=_parse_conditions(args.condition_bundle),
        evidence_bundle_path=args.evidence_bundle,
        bert_model=args.bert_model,
        profile=profile,
        baseline_artifact_path=args.baseline_artifact,
        baseline_source_prediction_path=args.baseline_source_predictions,
        rescore_baseline=args.rescore_baseline,
        sts_cache=cache,
        sts_cache_path=args.sts_cache,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output), "completed_per_condition": artifact["panel"]["completed_per_condition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
