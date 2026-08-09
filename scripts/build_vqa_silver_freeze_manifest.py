#!/usr/bin/env python3
"""Build and verify the label-free VQA silver scoring trust anchor.

The input locator contains private local paths, but it contributes no trust.
Every producer, judge, evidence, selector, protocol, and coverage byte is
validated and reduced to a deterministic path-free manifest.  That manifest
is intended to be built before annotations are available and then pinned by
SHA-256 in ``score_vqa_silver_oracle.py`` before canonical scoring.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 1
MANIFEST_TYPE = "vqa-silver-oracle-label-free-freeze"
EXPECTED_RUN_ID = "vqa-silver-oracle-election-20260809-a1"
EXPECTED_ARM_IDS = (
    "G/N",
    "G/F",
    "G/S",
    "H/N",
    "H/F",
    "H/S",
    "L/N",
    "L/F",
    "L/S",
)
EXPECTED_FACTORIAL_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
EXPECTED_SOURCE_IDS = EXPECTED_FACTORIAL_IDS + EXPECTED_ARM_IDS
EXPECTED_POOL_IDS = ("A", "B", "C")
CELL_BY_ARM = {
    "G/N": "gn",
    "G/F": "gf",
    "G/S": "gs",
    "H/N": "hn",
    "H/F": "hf",
    "H/S": "hs",
    "L/N": "ln",
    "L/F": "lf",
    "L/S": "ls",
}
EXPECTED_EVIDENCE_CELLS = tuple(CELL_BY_ARM.values())
EXPECTED_CANONICAL_PANEL_SHA256 = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXPECTED_VERIFIED_4B_CURRENT_PREDICTION_SHA256 = (
    "973989e0e10cffb2a9f211ad754008a373f69171319a67bcf17d6bd9e9b8f623"
)
EXPECTED_VERIFIED_4B_CURRENT_SOURCE_PREDICTION_SHA256 = (
    "ad9c9aabace33d3091c076e2f06c10ef8e55d355be9e5e392d790a80dbd78377"
)
EXPECTED_ANSWER_RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
EXPECTED_TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
EXPECTED_JUDGE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
EXPECTED_JUDGE_MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
EXPECTED_JUDGE_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@sha256:"
    "71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{2,64}")
_FACTORIAL_REQUEST_SOURCE = "shared-accepted-whisper"
_FACTORIAL_QUESTION_TYPES = {
    "count",
    "visible-text",
    "speech",
    "temporal",
    "causal-procedural",
    "general",
}
_FACTORIAL_RUNTIME_DEPENDENCIES = {
    "Pillow": "11.3.0",
    "soundfile": "0.13.1",
    "accelerate": "1.12.0",
    "av": "16.0.1",
    "imageio-ffmpeg": "0.6.0",
    "qwen-vl-utils": "0.0.14",
}
_FACTORIAL_PREDICTION_FIELDS = {
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
_FACTORIAL_ATTESTATION_FIELDS = {
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
_FORBIDDEN_LABEL_KEYS = {
    "annotation",
    "annotations",
    "correctanswer",
    "expectedanswer",
    "groundtruth",
    "label",
    "labels",
    "reference",
    "referenceanswer",
    "referenceanswers",
    "meteor",
    "bertscore",
    "bertscoref1",
    "sts",
    "semanticcomposite",
    "ndcg",
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_value_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(raw)


def canonical_manifest_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, *, kind: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{kind} must be a regular non-symlink file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {kind} JSON") from error


def _reject_labels(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label-bearing field at {location}.{key}")
            _reject_labels(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_labels(child, location=f"{location}[{index}]")


def _reject_answer_inputs(value: Any, *, location: str) -> None:
    _reject_labels(value, location=location)
    if isinstance(value, Mapping):
        for key, child in value.items():
            if re.sub(r"[^a-z0-9]", "", str(key).lower()) in {
                "answer",
                "answers",
                "prediction",
                "predictions",
            }:
                raise ValueError(f"answer-bearing request field at {location}.{key}")
            _reject_answer_inputs(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_answer_inputs(child, location=f"{location}[{index}]")


def _normalize_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "prediction_bundles",
        "election_bundles",
        "evidence_bundle",
        "selector",
        "selector_protocol",
        "localized_coverage",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != required:
        raise ValueError("freeze input locator schema is not exact")
    prediction_bundles = inputs["prediction_bundles"]
    election_bundles = inputs["election_bundles"]
    if not isinstance(prediction_bundles, Mapping) or set(prediction_bundles) != set(
        EXPECTED_SOURCE_IDS
    ):
        raise ValueError("freeze inputs require all exact 13 prediction sources")
    if not isinstance(election_bundles, Mapping) or set(election_bundles) != set(
        EXPECTED_POOL_IDS
    ):
        raise ValueError("freeze inputs require exact judge Pools A/B/C")
    return {
        "prediction_bundles": {
            source_id: Path(prediction_bundles[source_id])
            for source_id in EXPECTED_SOURCE_IDS
        },
        "election_bundles": {
            pool_id: Path(election_bundles[pool_id]) for pool_id in EXPECTED_POOL_IDS
        },
        "evidence_bundle": Path(inputs["evidence_bundle"]),
        "selector": Path(inputs["selector"]),
        "selector_protocol": Path(inputs["selector_protocol"]),
        "localized_coverage": Path(inputs["localized_coverage"]),
    }


def _safe_relative_path(raw: object) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("checksum path must be a non-empty string")
    path = Path(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.name
        in {
            "SHA256SUMS",
            "terminal-status.json",
        }
    ):
        raise ValueError("checksum manifest contains an unsafe path")
    return path


def _verify_checksums(root: Path) -> tuple[str, dict[str, str]]:
    root = root.resolve(strict=True)
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("payload checksum manifest is missing")
    entries: dict[str, str] = {}
    listed: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise ValueError("payload checksum manifest is malformed")
        relative = _safe_relative_path(match.group(2))
        key = relative.as_posix()
        if key in entries:
            raise ValueError("payload checksum manifest contains a duplicate path")
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
            or sha256_path(path) != match.group(1)
        ):
            raise ValueError(f"payload checksum verification failed: {key}")
        entries[key] = match.group(1)
        listed.append(key)
    if listed != sorted(listed):
        raise ValueError("payload checksum manifest is not canonically ordered")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("payload contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "terminal-status.json"}
    }
    if set(entries) != actual:
        raise ValueError("payload checksum manifest does not cover exact payload bytes")
    return sha256_path(checksum_path), entries


def _verify_factorial_checksums(root: Path) -> tuple[str, dict[str, str]]:
    """Verify the factorial producer's payload-wide checksum envelope.

    The factorial payload can carry a nested ``evidence/SHA256SUMS`` file.  It
    is a payload byte and therefore must be covered by the top-level manifest;
    unlike the Task5 envelope, a nested checksum file is not itself ambiguous.
    """

    root = root.resolve(strict=True)
    checksum_path = root / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("factorial payload checksum manifest is missing")
    entries: dict[str, str] = {}
    listed: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise ValueError("factorial payload checksum manifest is malformed")
        relative = Path(match.group(2))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.as_posix() == "SHA256SUMS"
            or relative.name == "terminal-status.json"
        ):
            raise ValueError("factorial payload checksum manifest contains an unsafe path")
        key = relative.as_posix()
        if key in entries:
            raise ValueError("factorial payload checksum manifest contains a duplicate path")
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
            or sha256_path(path) != match.group(1)
        ):
            raise ValueError(f"factorial payload checksum verification failed: {key}")
        entries[key] = match.group(1)
        listed.append(key)
    if listed != sorted(listed):
        raise ValueError("factorial payload checksum manifest is not canonically ordered")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("factorial payload contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    }
    allowed_entries = [actual]
    if "evidence/SHA256SUMS" in actual:
        allowed_entries.append(actual - {"evidence/SHA256SUMS"})
    if not any(set(entries) == candidate for candidate in allowed_entries):
        raise ValueError("factorial payload checksum manifest does not cover exact payload bytes")
    return sha256_path(checksum_path), entries


def _bundle_layout(root: Path) -> tuple[Path, Path]:
    root = Path(root)
    if root.is_symlink():
        raise ValueError("remote bundle root must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("remote bundle root must be a non-symlink directory")
    terminal = root / "terminal-status.json"
    payload_child = root / "payload"
    if payload_child.is_dir():
        if payload_child.is_symlink():
            raise ValueError("remote payload root must not be a symlink")
        if (root / "SHA256SUMS").exists():
            raise ValueError("remote bundle has ambiguous payload roots")
        payload = payload_child
    else:
        payload = root
    if terminal.is_symlink() or not terminal.is_file():
        raise ValueError("remote bundle terminal status is missing")
    return payload.resolve(strict=True), terminal


def _extract_rows(payload: Any, *, kind: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        for key in ("predictions", "rows", "requests", "results"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(
        isinstance(row, Mapping) for row in payload
    ):
        raise ValueError(f"{kind} must contain a list of objects")
    return list(payload)


def _row_ids(payload: Any, *, kind: str, expected_rows: int) -> tuple[str, ...]:
    rows = _extract_rows(payload, kind=kind)
    if len(rows) != expected_rows:
        raise ValueError(f"{kind} completion count is not {expected_rows}")
    ids: list[str] = []
    for index, row in enumerate(rows):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or _VIDEO_ID.fullmatch(video_id) is None:
            raise ValueError(f"{kind} row {index} has an invalid video ID")
        ids.append(video_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{kind} has duplicate video IDs")
    return tuple(ids)


def _panel_digest(ids: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(ids)) + "\n").encode("utf-8"))


def _expected_condition_identity(source_id: str) -> dict[str, Any]:
    if source_id in EXPECTED_ARM_IDS:
        condition_id = CELL_BY_ARM[source_id]
        model_id = "Qwen/Qwen3.5-4B"
        model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        answer_policy = "current"
    elif source_id.startswith("4b-"):
        condition_id = source_id
        model_id = "Qwen/Qwen3.5-4B"
        model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        answer_policy = source_id.removeprefix("4b-")
    else:
        condition_id = source_id
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        model_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        answer_policy = source_id.removeprefix("8b-")
    return {
        "source_id": source_id,
        "condition_id": condition_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "answer_policy": answer_policy,
        "min_pixels_per_frame": 65_536,
        "max_pixels_per_frame": 262_144,
        "runtime_image": EXPECTED_ANSWER_RUNTIME_IMAGE,
        "transformers_revision": EXPECTED_TRANSFORMERS_REVISION,
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 64,
    }


def _marker_matches_source(
    marker: Mapping[str, Any], source_id: str, condition_id: str
) -> bool:
    checks = []
    if "source_id" in marker:
        checks.append(marker["source_id"] == source_id)
    if "condition_id" in marker:
        checks.append(marker["condition_id"] == condition_id)
    if "answer_cell" in marker:
        checks.append(marker["answer_cell"] == condition_id)
    return bool(checks) and all(checks)


def _verify_evidence(
    paths: Mapping[str, Any], *, expected_rows: int
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, bool]]:
    root = Path(paths["evidence_bundle"])
    if root.is_symlink():
        raise ValueError("evidence bundle root must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence bundle must be a non-symlink directory")
    checksum_sha, entries = _verify_checksums(root)
    required_files = {
        "evidence-marker.json",
        "summary.json",
        "evidence.json",
        "leakage-audit.json",
        *(f"requests/{cell}.json" for cell in EXPECTED_EVIDENCE_CELLS),
    }
    if not required_files <= set(entries):
        raise ValueError(
            "evidence checksum manifest omits required marker/request bytes"
        )
    marker_path = root / "evidence-marker.json"
    summary_path = root / "summary.json"
    marker = _load_json(marker_path, kind="evidence marker")
    summary = _load_json(summary_path, kind="evidence summary")
    evidence_payload = _load_json(root / "evidence.json", kind="evidence payload")
    leakage = _load_json(root / "leakage-audit.json", kind="leakage audit")
    for value, location in (
        (marker, "$.evidence_marker"),
        (summary, "$.evidence_summary"),
        (evidence_payload, "$.evidence"),
        (leakage, "$.leakage_audit"),
    ):
        _reject_labels(value, location=location)
    if not isinstance(marker, Mapping) or marker != {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "example_count": expected_rows,
        "canonical_id_digest": marker.get("canonical_id_digest"),
        "checksum_file": "SHA256SUMS",
    }:
        raise ValueError("evidence marker identity is invalid")
    panel_digest = _require_sha256(
        marker["canonical_id_digest"], field="evidence panel digest"
    )
    expected_summary_fields = {
        "schema_version",
        "run_id",
        "row_count",
        "cell_count",
        "cells",
        "request_counts",
        "request_set_sha256",
        "evidence_sha256",
        "leakage_audit_sha256",
        "canonical_id_digest",
        "protocol_sha256",
        "selector_protocol_sha256",
    }
    if (
        not isinstance(summary, Mapping)
        or set(summary) != expected_summary_fields
        or summary.get("schema_version") != 1
        or summary.get("row_count") != expected_rows
        or summary.get("cell_count") != len(EXPECTED_EVIDENCE_CELLS)
        or summary.get("cells") != list(EXPECTED_EVIDENCE_CELLS)
        or summary.get("request_counts")
        != {cell: expected_rows for cell in EXPECTED_EVIDENCE_CELLS}
        or summary.get("canonical_id_digest") != panel_digest
    ):
        raise ValueError("evidence summary identity is invalid")
    if _canonical_value_sha256(evidence_payload) != summary["evidence_sha256"]:
        raise ValueError("evidence payload digest does not match its summary")
    if (
        not isinstance(leakage, Mapping)
        or leakage.get("references_loaded") is not False
        or _canonical_value_sha256(leakage) != summary["leakage_audit_sha256"]
    ):
        raise ValueError("evidence leakage audit is not label-free and bound")

    protocol_path = Path(paths["selector_protocol"])
    protocol = _load_json(protocol_path, kind="selector protocol")
    _reject_labels(protocol, location="$.selector_protocol")
    protocol_sha = sha256_path(protocol_path)
    if (
        summary.get("protocol_sha256") != protocol_sha
        or summary.get("selector_protocol_sha256") != protocol_sha
    ):
        raise ValueError("evidence selector protocol identity mismatch")

    request_sets = summary.get("request_set_sha256")
    if not isinstance(request_sets, Mapping) or set(request_sets) != set(
        EXPECTED_EVIDENCE_CELLS
    ):
        raise ValueError("evidence request-set identities are incomplete")
    request_file_hashes: dict[str, str] = {}
    request_set_hashes: dict[str, str] = {}
    request_ids: tuple[str, ...] | None = None
    for arm_id in EXPECTED_ARM_IDS:
        cell = CELL_BY_ARM[arm_id]
        request_path = root / "requests" / f"{cell}.json"
        request_payload = _load_json(request_path, kind=f"evidence request {cell}")
        _reject_answer_inputs(request_payload, location=f"$.requests.{cell}")
        ids = _row_ids(
            request_payload,
            kind=f"evidence request {cell}",
            expected_rows=expected_rows,
        )
        if request_ids is None:
            request_ids = ids
        elif ids != request_ids:
            raise ValueError("evidence requests do not share one exact ID order")
        canonical_sha = _canonical_value_sha256(request_payload)
        if request_sets[cell] != canonical_sha:
            raise ValueError(f"evidence request digest mismatch for {cell}")
        request_file_hashes[arm_id] = sha256_path(request_path)
        request_set_hashes[arm_id] = canonical_sha
    if request_ids is None or _panel_digest(request_ids) != panel_digest:
        raise ValueError("evidence request IDs do not match the frozen panel")

    selector_path = Path(paths["selector"])
    selector_payload = _load_json(selector_path, kind="selector freeze")
    _reject_answer_inputs(selector_payload, location="$.selector")
    selector_rows = _extract_rows(selector_payload, kind="selector freeze")
    if len(selector_rows) != expected_rows:
        raise ValueError("selector freeze completion count is invalid")
    selector_coverage: dict[str, bool] = {}
    selector_ids: list[str] = []
    for index, row in enumerate(selector_rows):
        video_id = row.get("video_id")
        selector = row.get("selector")
        if (
            not isinstance(video_id, str)
            or not isinstance(selector, Mapping)
            or row.get("protocol_sha256") != protocol_sha
        ):
            raise ValueError(f"selector protocol identity mismatch at row {index}")
        status = selector.get("status")
        if status not in {"localized", "non_localizable"}:
            raise ValueError(f"selector status is invalid at row {index}")
        if video_id in selector_coverage:
            raise ValueError("selector freeze contains duplicate video IDs")
        selector_ids.append(video_id)
        selector_coverage[video_id] = status == "localized"
    if tuple(selector_ids) != request_ids:
        raise ValueError("selector IDs/order do not match evidence requests")

    coverage_path = Path(paths["localized_coverage"])
    coverage = _load_json(coverage_path, kind="localized coverage")
    if (
        not isinstance(coverage, Mapping)
        or set(coverage) != set(selector_coverage)
        or any(not isinstance(value, bool) for value in coverage.values())
        or dict(coverage) != selector_coverage
    ):
        raise ValueError("localized coverage is not derived from the frozen selector")

    return (
        {
            "checksum_sha256": checksum_sha,
            "marker_sha256": sha256_path(marker_path),
            "summary_sha256": sha256_path(summary_path),
            "protocol_sha256": protocol_sha,
            "selector_protocol_sha256": protocol_sha,
            "selector_sha256": sha256_path(selector_path),
            "coverage_sha256": sha256_path(coverage_path),
            "request_file_sha256": request_file_hashes,
            "request_set_sha256": request_set_hashes,
            "localized_count": sum(selector_coverage.values()),
            "non_localized_count": expected_rows - sum(selector_coverage.values()),
        },
        request_ids,
        selector_coverage,
    )


def _verify_prediction_bundle(
    root: Path,
    *,
    source_id: str,
    expected_rows: int,
    expected_ids: Sequence[str],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload_root, terminal_path = _bundle_layout(root)
    checksum_sha, entries = _verify_checksums(payload_root)
    required = {"completion.json", "predictions.json", "requests.json"}
    if not required <= set(entries):
        raise ValueError(f"{source_id} payload checksum omits required files")
    completion_path = payload_root / "completion.json"
    prediction_path = payload_root / "predictions.json"
    request_path = payload_root / "requests.json"
    terminal = _load_json(terminal_path, kind=f"{source_id} terminal")
    completion = _load_json(completion_path, kind=f"{source_id} completion")
    prediction_payload = _load_json(prediction_path, kind=f"{source_id} predictions")
    request_payload = _load_json(request_path, kind=f"{source_id} requests")
    _reject_labels(terminal, location=f"$.prediction_sources.{source_id}.terminal")
    _reject_labels(completion, location=f"$.prediction_sources.{source_id}.completion")
    _reject_labels(
        prediction_payload,
        location=f"$.prediction_sources.{source_id}.predictions",
    )
    _reject_answer_inputs(
        request_payload, location=f"$.prediction_sources.{source_id}.requests"
    )
    prediction_sha = sha256_path(prediction_path)
    request_sha = sha256_path(request_path)
    completion_sha = sha256_path(completion_path)
    identity = _expected_condition_identity(source_id)
    condition_id = str(identity["condition_id"])
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("payload_sha256") != checksum_sha
        or terminal.get("completion_sha256") != completion_sha
        or terminal.get("request_sha256") != request_sha
        or terminal.get("prediction_sha256") != prediction_sha
        or not _marker_matches_source(terminal, source_id, condition_id)
    ):
        raise ValueError(f"{source_id} terminal digest chain is invalid")
    if (
        not isinstance(completion, Mapping)
        or completion.get("schema_version") != 1
        or completion.get("status") != "passed"
        or completion.get("expected_count") != expected_rows
        or completion.get("request_sha256") != request_sha
        or completion.get("prediction_sha256") != prediction_sha
        or completion.get("labels_joined") is not False
        or completion.get("evidence_marker_sha256") != evidence["marker_sha256"]
        or not _marker_matches_source(completion, source_id, condition_id)
    ):
        raise ValueError(f"{source_id} completion digest chain is invalid")
    source_request_sha = _require_sha256(
        completion.get("task4_request_set_sha256"),
        field=f"{source_id} source request digest",
    )
    if (
        source_id in EXPECTED_ARM_IDS
        and source_request_sha != evidence["request_set_sha256"][source_id]
    ):
        raise ValueError(f"{source_id} Task 4 request identity is invalid")

    rows = _extract_rows(prediction_payload, kind=f"{source_id} predictions")
    if len(rows) != expected_rows:
        raise ValueError(f"{source_id} prediction completion count is invalid")
    ids: list[str] = []
    runtime: str | None = None
    for index, row in enumerate(rows):
        _reject_labels(row, location=f"$.prediction_sources.{source_id}[{index}]")
        video_id = row.get("video_id")
        answer = row.get("answer")
        expected_row_identity = {
            "condition_id": identity["condition_id"],
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "answer_policy": identity["answer_policy"],
            "requested_min_pixels_per_frame": identity["min_pixels_per_frame"],
            "requested_max_pixels_per_frame": identity["max_pixels_per_frame"],
            "min_pixels_per_frame": identity["min_pixels_per_frame"],
            "max_pixels_per_frame": identity["max_pixels_per_frame"],
            "runtime_image": identity["runtime_image"],
        }
        if (
            row.get("status") != "passed"
            or not isinstance(video_id, str)
            or not isinstance(answer, str)
            or not answer.strip()
            or any(
                row.get(key) != value for key, value in expected_row_identity.items()
            )
            or not isinstance(row.get("runtime"), str)
            or not row["runtime"].strip()
        ):
            raise ValueError(
                f"{source_id} prediction identity is invalid at row {index}"
            )
        if runtime is None:
            runtime = row["runtime"]
        elif row["runtime"] != runtime:
            raise ValueError(f"{source_id} runtime identity varies across rows")
        ids.append(video_id)
    if tuple(ids) != tuple(expected_ids):
        raise ValueError(
            f"{source_id} prediction IDs/order do not match the evidence freeze"
        )
    request_ids = _row_ids(
        request_payload, kind=f"{source_id} requests", expected_rows=expected_rows
    )
    if request_ids != tuple(expected_ids):
        raise ValueError(
            f"{source_id} request IDs/order do not match the evidence freeze"
        )
    return {
        **identity,
        "runtime": runtime,
        "terminal_sha256": sha256_path(terminal_path),
        "payload_sha256": checksum_sha,
        "checksum_sha256": checksum_sha,
        "completion_sha256": completion_sha,
        "prediction_sha256": prediction_sha,
        "request_sha256": request_sha,
        "source_request_sha256": source_request_sha,
    }


def _factorial_layout(root: Path, condition_id: str) -> tuple[Path, Path, Path]:
    """Resolve one downloaded factorial producer bundle without reshaping it."""

    root = Path(root)
    if root.is_symlink():
        raise ValueError("factorial bundle root must not be a symlink")
    root = root.resolve(strict=True)
    payload_root = root / "payload"
    terminal_path = root / "terminal-status.json"
    if (
        not root.is_dir()
        or payload_root.is_symlink()
        or not payload_root.is_dir()
        or terminal_path.is_symlink()
        or not terminal_path.is_file()
    ):
        raise ValueError(
            "factorial bundle must be the downloaded root containing terminal-status.json beside payload/"
        )
    condition_root = payload_root / "conditions" / condition_id
    if condition_root.is_symlink() or not condition_root.is_dir():
        raise ValueError(
            f"factorial condition {condition_id} is missing under payload/conditions"
        )
    return payload_root, condition_root, terminal_path


def _finite_factorial_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"factorial {field} must be finite")
    return float(value)


def _verify_factorial_prediction_bundle(
    root: Path,
    *,
    source_id: str,
    expected_rows: int,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate one of the four legacy sources in the factorial layout.

    These sources were produced by ``score_vqa_factorial.py``'s producer
    contract, not by the later Task5 answer runner.  In particular, there is
    no completion marker or retroactive Task4 request-set attestation to
    invent or require here.
    """

    payload_root, condition_root, terminal_path = _factorial_layout(root, source_id)
    checksum_sha, _entries = _verify_factorial_checksums(payload_root)
    terminal = _load_json(terminal_path, kind=f"{source_id} factorial terminal")
    _reject_labels(terminal, location=f"$.prediction_sources.{source_id}.terminal")
    terminal_fields = {
        "payload_sha256",
        "payload_uri",
        "run_name",
        "schema_version",
        "status",
    }
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != terminal_fields
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("payload_sha256") != checksum_sha
        or not isinstance(terminal.get("run_name"), str)
        or not terminal["run_name"].strip()
        or not isinstance(terminal.get("payload_uri"), str)
        or not terminal["payload_uri"].strip()
    ):
        raise ValueError(f"{source_id} factorial terminal status is not passed and verified")

    status = _load_json(payload_root / "status.json", kind=f"{source_id} factorial status")
    _reject_labels(status, location=f"$.prediction_sources.{source_id}.status")
    status_fields = {
        "expected_count",
        "frame_count",
        "labels_joined",
        "mode",
        "remote_payload_sync",
        "run_name",
        "schema_version",
        "status",
        "workload_returncode",
    }
    if (
        not isinstance(status, Mapping)
        or set(status) != status_fields
        or status.get("schema_version") != 1
        or status.get("status") != "passed"
        or status.get("mode") != "condition"
        or status.get("run_name") != terminal["run_name"]
        or status.get("expected_count") != expected_rows
        or status.get("frame_count") != 12
        or status.get("workload_returncode") != 0
        or status.get("remote_payload_sync") != "frozen"
        or status.get("labels_joined") is not False
    ):
        raise ValueError(f"{source_id} factorial status marker is invalid")

    runtime = _load_json(payload_root / "runtime.json", kind=f"{source_id} factorial runtime")
    required_runtime = {
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
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != required_runtime
        or not isinstance(runtime.get("runtime"), str)
        or not runtime["runtime"].strip()
        or runtime.get("runtime_image") != EXPECTED_ANSWER_RUNTIME_IMAGE
        or not isinstance(runtime.get("transformers_version"), str)
        or not runtime["transformers_version"].strip()
        or runtime.get("transformers_revision") != EXPECTED_TRANSFORMERS_REVISION
        or not isinstance(runtime.get("torch_version"), str)
        or not runtime["torch_version"].strip()
        or not isinstance(runtime.get("torch_cuda"), str)
        or not runtime["torch_cuda"].strip()
        or runtime.get("torch_hip") is not None
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime["cuda_device_name"].strip()
        or runtime.get("cuda_bf16_supported") is not True
        or runtime.get("dependencies") != _FACTORIAL_RUNTIME_DEPENDENCIES
    ):
        raise ValueError(f"{source_id} factorial runtime attestation is invalid")

    identity_path = condition_root / "condition-identity.json"
    identity_payload = _load_json(
        identity_path, kind=f"{source_id} factorial condition identity"
    )
    if not isinstance(identity_payload, Mapping):
        raise ValueError(f"{source_id} factorial condition identity is invalid")
    nested = identity_payload.get("condition")
    decoding = identity_payload.get("deterministic_decoding")
    expected_top_level_fields = {
        "condition",
        "deterministic_decoding",
        "expected_count",
        "frame_count",
        "input_uri",
        "manifest",
        "model_class",
        "profile_valid",
        "remote_uri",
        "run_name",
        "runtime_image",
    }
    expected_condition_fields = {
        "answer_policy",
        "condition_id",
        "max_pixels",
        "min_pixels",
        "model_id",
        "request_source",
        "revision",
    }
    expected_decoding_fields = {"do_sample", "enable_thinking", "max_new_tokens"}
    if (
        set(identity_payload) != expected_top_level_fields
        or not isinstance(nested, Mapping)
        or set(nested) != expected_condition_fields
        or not isinstance(decoding, Mapping)
        or set(decoding) != expected_decoding_fields
    ):
        raise ValueError(f"{source_id} factorial condition identity schema is invalid")
    expected_identity = _expected_condition_identity(source_id)
    if (
        identity_payload.get("run_name") != terminal["run_name"]
        or identity_payload.get("expected_count") != expected_rows
        or identity_payload.get("frame_count") != 12
        or identity_payload.get("model_class") != "multimodal"
        or identity_payload.get("profile_valid") is not True
        or identity_payload.get("runtime_image") != EXPECTED_ANSWER_RUNTIME_IMAGE
        or identity_payload.get("manifest")
        != "/dstack/run/configs/vqa/qwen35_vqa2025_475_label_free.json"
        or not isinstance(identity_payload.get("input_uri"), str)
        or not identity_payload["input_uri"].startswith("hf://buckets/")
        or not isinstance(identity_payload.get("remote_uri"), str)
        or terminal["payload_uri"] != f"{identity_payload['remote_uri']}/payload"
        or nested.get("condition_id") != expected_identity["condition_id"]
        or nested.get("model_id") != expected_identity["model_id"]
        or nested.get("revision") != expected_identity["model_revision"]
        or nested.get("answer_policy") != expected_identity["answer_policy"]
        or nested.get("min_pixels") != expected_identity["min_pixels_per_frame"]
        or nested.get("max_pixels") != expected_identity["max_pixels_per_frame"]
        or nested.get("request_source") != _FACTORIAL_REQUEST_SOURCE
        or decoding
        != {"do_sample": False, "enable_thinking": False, "max_new_tokens": 64}
    ):
        raise ValueError(f"{source_id} factorial condition identity is not terminal-bound")
    identity = {
        "run_id": identity_payload["run_name"],
        "condition_id": nested["condition_id"],
        "model_id": nested["model_id"],
        "model_revision": nested["revision"],
        "answer_policy": nested["answer_policy"],
    }

    prediction_path = condition_root / "predictions.json"
    prediction_payload = _load_json(
        prediction_path, kind=f"{source_id} factorial predictions"
    )
    if not isinstance(prediction_payload, list) or len(prediction_payload) != expected_rows:
        raise ValueError(f"{source_id} factorial prediction payload is incomplete")
    prediction_rows: list[Mapping[str, Any]] = []
    ids: list[str] = []
    runtime_name: str | None = None
    for index, row in enumerate(prediction_payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source_id} factorial prediction row {index} is invalid")
        _reject_labels(row, location=f"$.prediction_sources.{source_id}[{index}]")
        if set(row) != _FACTORIAL_PREDICTION_FIELDS:
            raise ValueError(f"{source_id} factorial prediction schema is not exact")
        video_id = row.get("video_id")
        answer = row.get("answer")
        expected_row = {
            "run_id": identity["run_id"],
            "condition_id": source_id,
            "model_id": expected_identity["model_id"],
            "model_revision": expected_identity["model_revision"],
            "answer_policy": expected_identity["answer_policy"],
            "requested_min_pixels_per_frame": expected_identity["min_pixels_per_frame"],
            "requested_max_pixels_per_frame": expected_identity["max_pixels_per_frame"],
            "min_pixels_per_frame": expected_identity["min_pixels_per_frame"],
            "max_pixels_per_frame": expected_identity["max_pixels_per_frame"],
            "runtime_image": EXPECTED_ANSWER_RUNTIME_IMAGE,
            "precision": "bfloat16",
            "device": "cuda:0 (NVIDIA CUDA)",
        }
        if (
            any(row.get(key) != value for key, value in expected_row.items())
            or not isinstance(video_id, str)
            or _VIDEO_ID.fullmatch(video_id) is None
            or not isinstance(answer, str)
            or not answer.strip()
            or row.get("status") != "passed"
            or row.get("error_type") is not None
            or row.get("error_message") is not None
            or row.get("question_type") not in _FACTORIAL_QUESTION_TYPES
        ):
            raise ValueError(f"{source_id} factorial prediction identity is invalid at row {index}")
        _finite_factorial_number(row.get("elapsed_seconds"), field="elapsed_seconds")
        _finite_factorial_number(row.get("model_load_seconds"), field="model_load_seconds")
        if row["elapsed_seconds"] < 0 or row["model_load_seconds"] < 0:
            raise ValueError(f"{source_id} factorial prediction timing is invalid at row {index}")
        peak = row.get("peak_memory_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
            raise ValueError(f"{source_id} factorial prediction peak memory is invalid")
        attestation = row.get("runtime_attestation")
        if not isinstance(attestation, Mapping) or set(attestation) != _FACTORIAL_ATTESTATION_FIELDS:
            raise ValueError(f"{source_id} factorial runtime attestation schema is invalid")
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
                raise ValueError(f"{source_id} factorial {label} attestation is invalid")
        if (
            attestation["accelerator_backend"] != "nvidia-cuda"
            or not isinstance(attestation["torch_cuda"], str)
            or not attestation["torch_cuda"].strip()
            or attestation["torch_cuda"] != runtime["torch_cuda"]
            or attestation["torch_hip"] is not None
        ):
            raise ValueError(f"{source_id} factorial accelerator attestation is invalid")
        row_runtime = row.get("runtime")
        if not isinstance(row_runtime, str) or not row_runtime.strip() or row_runtime != runtime["runtime"]:
            raise ValueError(f"{source_id} factorial prediction/runtime identity mismatch")
        if runtime_name is None:
            runtime_name = row_runtime
        elif row_runtime != runtime_name:
            raise ValueError(f"{source_id} factorial runtime identity varies across rows")
        prediction_rows.append(row)
        ids.append(video_id)
    if len(ids) != len(set(ids)) or tuple(ids) != tuple(expected_ids):
        raise ValueError(f"{source_id} factorial prediction IDs/order do not match the frozen panel")

    timing_path = condition_root / "timing.json"
    timing = _load_json(timing_path, kind=f"{source_id} factorial timing")
    if not isinstance(timing, Mapping):
        raise ValueError(f"{source_id} factorial timing is invalid")
    timing_required = {
        "condition_id",
        "row_count",
        "wall_seconds",
        "peak_memory_bytes",
        "model_load_seconds",
        "runtime_identity",
    }
    if (
        not timing_required <= set(timing)
        or not set(timing) <= timing_required | {"prepare_wall_seconds"}
        or timing.get("condition_id") != source_id
        or timing.get("row_count") != expected_rows
    ):
        raise ValueError(f"{source_id} factorial timing schema is invalid")
    wall_seconds = _finite_factorial_number(timing["wall_seconds"], field="wall_seconds")
    model_load_seconds = _finite_factorial_number(
        timing["model_load_seconds"], field="model_load_seconds"
    )
    if wall_seconds <= 0 or model_load_seconds < 0:
        raise ValueError(f"{source_id} factorial timing values are invalid")
    if "prepare_wall_seconds" in timing:
        if _finite_factorial_number(timing["prepare_wall_seconds"], field="prepare_wall_seconds") < 0:
            raise ValueError(f"{source_id} factorial preparation timing is invalid")
    timing_peak = timing["peak_memory_bytes"]
    if isinstance(timing_peak, bool) or not isinstance(timing_peak, int) or timing_peak <= 0:
        raise ValueError(f"{source_id} factorial timing peak memory is invalid")
    if timing.get("runtime_identity") != [
        [runtime_name, EXPECTED_ANSWER_RUNTIME_IMAGE, "cuda:0 (NVIDIA CUDA)", "bfloat16"]
    ]:
        raise ValueError(f"{source_id} factorial timing runtime identity is invalid")
    if {row["model_load_seconds"] for row in prediction_rows} != {model_load_seconds}:
        raise ValueError(f"{source_id} factorial model-load identity is invalid")
    if timing_peak != max(row["peak_memory_bytes"] for row in prediction_rows):
        raise ValueError(f"{source_id} factorial timing peak memory does not match predictions")

    telemetry_path = condition_root / "telemetry" / "nvidia-smi.csv"
    if telemetry_path.is_symlink() or not telemetry_path.is_file():
        raise ValueError(f"{source_id} factorial NVIDIA telemetry is missing")
    telemetry = csv.DictReader(telemetry_path.read_text(encoding="utf-8").splitlines())
    telemetry_fields = {
        "timestamp",
        "index",
        "utilization.gpu [%]",
        "memory.used [MiB]",
        "memory.total [MiB]",
    }
    if set(telemetry.fieldnames or ()) != telemetry_fields:
        raise ValueError(f"{source_id} factorial NVIDIA telemetry schema is invalid")
    samples = 0
    for record in telemetry:
        try:
            index = int(record["index"])
            utilization = _finite_factorial_number(
                float(record["utilization.gpu [%]"]), field="GPU utilization"
            )
            memory_used = _finite_factorial_number(
                float(record["memory.used [MiB]"]), field="GPU memory used"
            )
            memory_total = _finite_factorial_number(
                float(record["memory.total [MiB]"]), field="GPU memory total"
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{source_id} factorial NVIDIA telemetry row is invalid") from error
        if (
            index < 0
            or not record["timestamp"].strip()
            or not 0 <= utilization <= 100
            or memory_used < 0
            or memory_total <= 0
            or memory_used > memory_total
        ):
            raise ValueError(f"{source_id} factorial NVIDIA telemetry values are invalid")
        samples += 1
    if samples == 0:
        raise ValueError(f"{source_id} factorial NVIDIA telemetry has no samples")

    request_path = condition_root / "requests.json"
    if request_path.exists():
        request_payload = _load_json(request_path, kind=f"{source_id} factorial requests")
        _reject_answer_inputs(request_payload, location=f"$.prediction_sources.{source_id}.requests")
        if _row_ids(request_payload, kind=f"{source_id} factorial requests", expected_rows=expected_rows) != tuple(expected_ids):
            raise ValueError(f"{source_id} factorial request IDs/order do not match the frozen panel")
        request_sha = sha256_path(request_path)
    else:
        # Some archived factorial bundles omit the duplicate request copy.  Bind
        # the immutable condition identity rather than inventing Task4 provenance.
        request_sha = sha256_path(identity_path)

    evidence_checksum_path = payload_root / "evidence" / "SHA256SUMS"
    if evidence_checksum_path.is_symlink() or not evidence_checksum_path.is_file():
        raise ValueError(f"{source_id} factorial nested evidence checksum is missing")

    return {
        **expected_identity,
        "runtime": runtime_name,
        "terminal_sha256": sha256_path(terminal_path),
        "payload_sha256": checksum_sha,
        "checksum_sha256": checksum_sha,
        "evidence_checksum_sha256": sha256_path(evidence_checksum_path),
        "prediction_sha256": sha256_path(prediction_path),
        "request_sha256": request_sha,
    }


def _verify_election_bundle(
    root: Path,
    *,
    pool_id: str,
    expected_rows: int,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    payload_root, terminal_path = _bundle_layout(root)
    checksum_sha, entries = _verify_checksums(payload_root)
    required = {"rankings.json", "requests.json"}
    if not required <= set(entries):
        raise ValueError(f"Pool {pool_id} payload checksum omits request/ranking files")
    request_path = payload_root / "requests.json"
    ranking_path = payload_root / "rankings.json"
    terminal = _load_json(terminal_path, kind=f"Pool {pool_id} terminal")
    request_payload = _load_json(request_path, kind=f"Pool {pool_id} requests")
    ranking_payload = _load_json(ranking_path, kind=f"Pool {pool_id} rankings")
    for value, location in (
        (terminal, f"$.elections.{pool_id}.terminal"),
        (request_payload, f"$.elections.{pool_id}.requests"),
        (ranking_payload, f"$.elections.{pool_id}.rankings"),
    ):
        _reject_labels(value, location=location)
    request_sha = sha256_path(request_path)
    ranking_sha = sha256_path(ranking_path)
    expected_identity = {
        "pool_id": pool_id,
        "judge_model_id": EXPECTED_JUDGE_MODEL_ID,
        "judge_model_revision": EXPECTED_JUDGE_MODEL_REVISION,
        "runtime_image": EXPECTED_JUDGE_RUNTIME_IMAGE,
    }
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("payload_sha256") != checksum_sha
        or terminal.get("request_sha256") != request_sha
        or terminal.get("ranking_sha256") != ranking_sha
        or terminal.get("labels_joined") is not False
        or any(terminal.get(key) != value for key, value in expected_identity.items())
    ):
        raise ValueError(
            f"Pool {pool_id} terminal request/ranking digest chain is invalid"
        )
    if _row_ids(
        request_payload, kind=f"Pool {pool_id} requests", expected_rows=expected_rows
    ) != tuple(expected_ids):
        raise ValueError(f"Pool {pool_id} request IDs/order are not frozen")
    if _row_ids(
        ranking_payload, kind=f"Pool {pool_id} rankings", expected_rows=expected_rows
    ) != tuple(expected_ids):
        raise ValueError(f"Pool {pool_id} ranking IDs/order are not frozen")
    return {
        **expected_identity,
        "terminal_sha256": sha256_path(terminal_path),
        "payload_sha256": checksum_sha,
        "checksum_sha256": checksum_sha,
        "request_sha256": request_sha,
        "ranking_sha256": ranking_sha,
    }


def build_freeze_manifest(
    inputs: Mapping[str, Any], *, expected_rows: int = 475
) -> dict[str, Any]:
    """Validate label-free frozen artifacts and return their path-free manifest."""

    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise ValueError("expected_rows must be a positive integer")
    paths = _normalize_inputs(inputs)
    evidence, ids, _coverage = _verify_evidence(paths, expected_rows=expected_rows)
    panel_digest = _panel_digest(ids)
    prediction_sources: dict[str, dict[str, Any]] = {}
    for source_id in EXPECTED_SOURCE_IDS:
        if source_id in EXPECTED_FACTORIAL_IDS:
            prediction_sources[source_id] = _verify_factorial_prediction_bundle(
                paths["prediction_bundles"][source_id],
                source_id=source_id,
                expected_rows=expected_rows,
                expected_ids=ids,
            )
        else:
            prediction_sources[source_id] = _verify_prediction_bundle(
                paths["prediction_bundles"][source_id],
                source_id=source_id,
                expected_rows=expected_rows,
                expected_ids=ids,
                evidence=evidence,
            )
    elections = {
        pool_id: _verify_election_bundle(
            paths["election_bundles"][pool_id],
            pool_id=pool_id,
            expected_rows=expected_rows,
            expected_ids=ids,
        )
        for pool_id in EXPECTED_POOL_IDS
    }
    if expected_rows == 475:
        if panel_digest != EXPECTED_CANONICAL_PANEL_SHA256:
            raise ValueError(
                "freeze manifest does not bind the canonical 475-row panel"
            )
        if (
            prediction_sources["4b-current"]["prediction_sha256"]
            != EXPECTED_VERIFIED_4B_CURRENT_PREDICTION_SHA256
        ):
            raise ValueError(
                "freeze manifest does not bind the verified 4b-current bytes"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "expected_rows": expected_rows,
        "panel_id_set_sha256": panel_digest,
        "verified_4b_current_source_prediction_sha256": (
            EXPECTED_VERIFIED_4B_CURRENT_SOURCE_PREDICTION_SHA256
        ),
        "prediction_sources": prediction_sources,
        "elections": elections,
        "evidence": evidence,
    }


def write_freeze_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write one deterministic freeze manifest without overwriting a freeze."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_manifest_bytes(manifest))
    except FileExistsError:
        raise FileExistsError(f"freeze manifest already exists: {path}") from None


def verify_freeze_manifest(
    manifest_path: Path,
    inputs: Mapping[str, Any],
    *,
    expected_rows: int = 475,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Revalidate every byte and require exact equality with a frozen manifest."""

    path = Path(manifest_path)
    raw = path.read_bytes()
    actual_sha = _sha256_bytes(raw)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256, field="expected freeze-manifest digest"
        )
        if expected == "0" * 64 or actual_sha != expected:
            raise ValueError(
                "freeze-manifest bytes do not match the pinned trust anchor"
            )
    try:
        frozen = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("freeze manifest is not valid JSON") from error
    rebuilt = build_freeze_manifest(inputs, expected_rows=expected_rows)
    if frozen != rebuilt or raw != canonical_manifest_bytes(rebuilt):
        raise ValueError(
            "current label-free bytes do not match the frozen manifest identity"
        )
    return {
        "manifest": rebuilt,
        "manifest_sha256": actual_sha,
        "inputs": _normalize_inputs(inputs),
    }


def _load_locator(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path, kind="freeze input locator")
    if not isinstance(payload, Mapping):
        raise ValueError("freeze input locator must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pre-annotation VQA silver freeze manifest"
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=475)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_freeze_manifest(
        _load_locator(args.inputs), expected_rows=args.expected_rows
    )
    write_freeze_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
