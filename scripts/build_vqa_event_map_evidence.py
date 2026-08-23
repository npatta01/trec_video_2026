#!/usr/bin/env python3
"""Materialize label-blind evidence from frozen AV event-map selections.

The event-map output is used only as a numeric selector.  Its generated answer,
confidence, and event summaries are deliberately discarded before any
model-facing request is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from itertools import permutations
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trec_video_2026.vqa_experiments.silver_oracle import (
    build_frame_plans,
)
from trec_video_2026.vqa_experiments.nemotron3_omni_event_map import (
    validate_private_response,
)


RUN_ID = "vqa-event-map-evidence-20260809-a1"
EXPECTED_COUNT = 475
CANONICAL_PANEL_ID_DIGEST = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
PLAN_IDS = ("F0", "H/F", "G/S", "H/S", "Shifted H/S")
PLAN_DIRS = {
    "F0": "f0",
    "H/F": "hf",
    "G/S": "gs",
    "H/S": "hs",
    "Shifted H/S": "shifted-hs",
}
PLAN_TRANSCRIPTS = {
    "F0": "full",
    "H/F": "full",
    "G/S": "selected",
    "H/S": "selected",
    "Shifted H/S": "selected",
}
SHA256 = re.compile(r"[0-9a-f]{64}")
YOUTUBE_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_TIME_EPSILON = 1e-6

_BASELINE_CONTRACT_FIELDS = frozenset(
    {
        "run_name",
        "payload_uri",
        "terminal_sha256",
        "checksums_sha256",
        "evidence_marker_sha256",
        "manifest_sha256",
        "requests_sha256",
        "frames_requests_sha256",
        "evidence_sha256",
        "whisper_sha256",
        "whisper_summary_sha256",
        "whisper_rows_sha256",
        "model_id",
        "model_revision",
        "example_count",
        "available_count",
        "quality_rejected_count",
        "canonical_id_digest",
    }
)
PRODUCTION_BASELINE_CONTRACT = {
    "run_name": "vqa-factorial-full-evidence-20260809-f1",
    "payload_uri": "hf://buckets/Npatta01/trec_video_2026/experiments/vqa-factorial-full-evidence-20260809-f1/payload",
    "terminal_sha256": "1d7c834b6ea74222f40992582706b32cedd9ac5f2898dcc3f953419dacfe507b",
    "checksums_sha256": "b7610cbdf5716269b06b7d66fdf81376873cb8c98f24a9aba25a502f9176926c",
    "evidence_marker_sha256": "cb520be47beef9286f262717ea0ee709be39d8fbae713eda0ffd93768e164ceb",
    "manifest_sha256": "092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8",
    "requests_sha256": "2047271801a03a6d48780362f90757319240d7b4538e757f481d6ee5860fc166",
    "frames_requests_sha256": "2ed42955c189a4df0be53a284549d50960dcba41e9828efafe34a778ba5df0d5",
    "evidence_sha256": "1785d8a05a786db410293b5accf0cfb518914c9be559a7592b471daf9dd8f33f",
    "whisper_sha256": "4407a6db63e8cc06a742c7741d8c993b4a361de929259d43761ffd633ceb39f8",
    "whisper_summary_sha256": "71c0f7bbf4d40455cb72c11cbead638801535628368d3737a5c516617cd8a2da",
    "whisper_rows_sha256": "3d1501363738ed27da95e64555d7bcba02be292e72208f813916574eca0239d0",
    "model_id": "openai/whisper-large-v3",
    "model_revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "example_count": 475,
    "available_count": 427,
    "quality_rejected_count": 48,
    "canonical_id_digest": CANONICAL_PANEL_ID_DIGEST,
}

PanelInventoryProvider = Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
FrameRenderer = Callable[
    [Mapping[str, Any], Mapping[str, Any], str, Path], Mapping[str, Any]
]
ContactSheetBuilder = Callable[..., Mapping[str, Any]]
BatchFrameDecoder = Callable[
    [Mapping[str, Any], Mapping[tuple[int, int], Mapping[str, Any]], Path],
    Mapping[tuple[int, int], Mapping[str, Any]],
]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _validate_video_id(value: Any, *, field: str = "video_id") -> str:
    if not isinstance(value, str) or YOUTUBE_VIDEO_ID.fullmatch(value) is None:
        raise ValueError(
            f"{field} must match the canonical 11-character YouTube ID grammar"
        )
    return value


def _canonical_id_digest(video_ids: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(video_ids) + "\n").encode("utf-8"))


def _exact_time_base(value: Any) -> tuple[str, Fraction]:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            parsed = Fraction(value.numerator, value.denominator)
        else:
            parsed = Fraction(str(value))
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            "decoded frame time_base must be a positive rational"
        ) from error
    if parsed <= 0:
        raise ValueError("decoded frame time_base must be a positive rational")
    return f"{parsed.numerator}/{parsed.denominator}", parsed


def _safe_json_rows(
    payload: Any, *, source: str, names: Sequence[str]
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw_rows = payload
    elif isinstance(payload, Mapping):
        raw_rows = next(
            (
                payload.get(name)
                for name in names
                if isinstance(payload.get(name), list)
            ),
            None,
        )
        if raw_rows is None:
            raise ValueError(f"{source} must contain one of {', '.join(names)}")
    else:
        raise ValueError(f"{source} must be a JSON list or object")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise ValueError(f"{source} rows must be objects")
    return [dict(row) for row in raw_rows]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error


def _rows_source_path(path: Path, *, source: str) -> Path:
    if not path.is_dir():
        if not path.is_file():
            raise ValueError(f"{source} rows source is missing: {path}")
        return path
    candidates = tuple(
        path / name for name in ("predictions.jsonl", "results.jsonl", "rows.jsonl")
    )
    candidate = next((item for item in candidates if item.is_file()), None)
    if candidate is None:
        raise ValueError(f"{source} directory has no predictions/results/rows JSONL")
    return candidate


def _load_rows(
    path: Path, *, source: str, names: Sequence[str]
) -> list[dict[str, Any]]:
    if path.is_dir():
        terminal = path / "terminal-status.json"
        if terminal.is_file():
            marker = _load_json(terminal)
            if (
                not isinstance(marker, Mapping)
                or marker.get("status") not in {"complete", "frozen"}
                and marker.get("freeze_status") != "complete"
            ):
                raise ValueError(f"{source} terminal marker is not complete")
        candidate = _rows_source_path(path, source=source)
        lines = [
            line
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        try:
            payload = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            raise ValueError(f"{source} JSONL is malformed: {error}") from error
        return _safe_json_rows(payload, source=source, names=names)
    if path.suffix.casefold() == ".jsonl":
        try:
            payload = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read {source} JSONL {path}: {error}"
            ) from error
        return _safe_json_rows(payload, source=source, names=names)
    return _safe_json_rows(_load_json(path), source=source, names=names)


PRODUCER_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "video_id",
        "condition_id",
        "request_sha256",
        "primary_request_sha256",
        "retry_request_sha256",
        "primary_valid",
        "retry_valid",
        "retry_used",
        "status",
        "error_type",
        "response_sha256",
        "model_id",
        "model_revision",
        "runtime_image",
        "memory_telemetry",
        "runtime_identity",
        "answer",
        "abstain",
        "confidence",
        "events",
        "selected_intervals",
        "elapsed_seconds",
        "run_id",
        "slice_id",
        "question",
        "question_stratum",
        "controller_request_sha256",
        "media_path",
        "source_media_sha256",
        "normalized_media_sha256",
        "source_duration_seconds",
        "normalized_duration_seconds",
    }
)
PRODUCER_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "video_id",
        "condition_id",
        "request_sha256",
        "primary_request_sha256",
        "retry_request_sha256",
        "primary_valid",
        "retry_valid",
        "retry_used",
        "status",
        "error_type",
        "response_sha256",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_identity",
        "answer",
        "abstain",
        "confidence",
        "events",
        "selected_intervals",
        "elapsed_seconds",
        "run_id",
        "slice_id",
        "question",
        "media_path",
        "source_media_sha256",
        "normalized_media_sha256",
        "source_duration_seconds",
        "normalized_duration_seconds",
    }
)
FREEZE_IDENTITY_FIELDS = frozenset(
    {
        "source_bundle_sha256",
        "controller_sha256",
        "protocol_sha256",
        "inference_module_sha256",
        "runner_sha256",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_identity",
        "row_run_id",
        "producer_run_prefix",
        "output_uri_prefix",
    }
)
PRODUCER_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "attempt_uuid",
        "output_identity",
        "payload_checksum",
        "row_count",
        "source_bundle_sha256",
        "controller_sha256",
        "protocol_sha256",
        "request_file_sha256",
        "inference_module_sha256",
        "runner_sha256",
        "producer_timings",
    }
)


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], name: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{name} schema is not exact")


def _canonical_uuid(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{field} must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID")
    return value


def _validate_producer_timings(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("producer terminal timings must be an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "clock",
            "bootstrap_download_seconds",
            "normalization",
            "model_startup_seconds",
            "inference_seconds",
            "finalization_seconds",
            "phase_total_seconds",
        },
        "producer terminal timings",
    )
    normalization = value.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("producer normalization timing must be an object")
    _exact_keys(
        normalization, {"media_count", "seconds"}, "producer normalization timing"
    )
    if value.get("schema_version") != 1 or value.get("clock") != "monotonic":
        raise ValueError("producer terminal timing identity mismatch")
    media_count = normalization.get("media_count")
    if (
        isinstance(media_count, bool)
        or not isinstance(media_count, int)
        or media_count <= 0
    ):
        raise ValueError("producer normalization media count is invalid")
    phases = [
        _finite(value.get("bootstrap_download_seconds"), "bootstrap timing"),
        _finite(normalization.get("seconds"), "normalization timing"),
        _finite(value.get("model_startup_seconds"), "model startup timing"),
        _finite(value.get("inference_seconds"), "inference timing"),
        _finite(value.get("finalization_seconds"), "finalization timing"),
    ]
    if any(item <= 0 for item in phases) or not math.isclose(
        _finite(value.get("phase_total_seconds"), "phase total timing"),
        sum(phases),
        rel_tol=1e-9,
        abs_tol=0.001,
    ):
        raise ValueError("producer terminal timings do not reconcile")


def _validate_frozen_producer_row(
    row: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    shard_index: int,
    expected_fields: frozenset[str],
) -> dict[str, Any]:
    _exact_keys(row, expected_fields, "aggregate freeze producer row")
    video_id = _validate_video_id(row.get("video_id"), field="producer video_id")
    if (
        row.get("schema_version") != 1
        or row.get("condition_id") != "av-map"
        or row.get("run_id") != identity["row_run_id"]
        or row.get("slice_id") != f"{shard_index:02d}"
        or row.get("status") != "passed"
        or row.get("error_type") is not None
    ):
        raise ValueError("aggregate freeze producer row identity/status mismatch")
    if (
        row.get("model_id") != identity["model_id"]
        or row.get("model_revision") != identity["model_revision"]
        or row.get("runtime_image") != identity["runtime_image"]
        or row.get("runtime_identity") != identity["runtime_identity"]
    ):
        raise ValueError("aggregate freeze producer model/runtime identity mismatch")
    for field in (
        "request_sha256",
        "primary_request_sha256",
        "response_sha256",
        "controller_request_sha256",
        "source_media_sha256",
        "normalized_media_sha256",
    ):
        if not _is_sha256(row.get(field)):
            raise ValueError(f"aggregate freeze producer {field} is invalid")
    retry_used = row.get("retry_used")
    if type(retry_used) is not bool or type(row.get("primary_valid")) is not bool:
        raise ValueError("aggregate freeze producer retry state is malformed")
    if retry_used:
        if (
            row.get("primary_valid") is not False
            or row.get("retry_valid") is not True
            or not _is_sha256(row.get("retry_request_sha256"))
            or row.get("request_sha256") != row.get("retry_request_sha256")
        ):
            raise ValueError("aggregate freeze producer retry state is invalid")
    elif (
        row.get("primary_valid") is not True
        or row.get("retry_valid") is not None
        or row.get("retry_request_sha256") is not None
        or row.get("request_sha256") != row.get("primary_request_sha256")
    ):
        raise ValueError("aggregate freeze producer primary state is invalid")
    media_path = row.get("media_path")
    relative_media = PurePosixPath(media_path) if isinstance(media_path, str) else None
    if (
        relative_media is None
        or relative_media.is_absolute()
        or len(relative_media.parts) != 2
        or relative_media.parts[0] != "media"
        or relative_media.stem != video_id
    ):
        raise ValueError("aggregate freeze producer media path is unsafe")
    source_duration = _finite(
        row.get("source_duration_seconds"), "producer source duration"
    )
    normalized_duration = _finite(
        row.get("normalized_duration_seconds"), "producer normalized duration"
    )
    if source_duration <= 0 or abs(source_duration - normalized_duration) > 0.25:
        raise ValueError("aggregate freeze producer duration mismatch")
    _finite(row.get("elapsed_seconds"), "producer elapsed seconds")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("aggregate freeze producer question is missing")
    _extract_private_map_response(row, duration=normalized_duration)
    return dict(row)


def validate_aggregate_freeze_manifest(
    path: Path,
    *,
    expected_count: int = EXPECTED_COUNT,
    expected_panel_id_digest: str = CANONICAL_PANEL_ID_DIGEST,
    expected_shard_counts: Sequence[int] = (119, 119, 119, 118),
    expected_identities: Mapping[str, Any],
    expected_producer_row_fields: Sequence[str] = tuple(sorted(PRODUCER_ROW_FIELDS)),
    expected_av_shards: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate one aggregate freeze and return its ordered AV projection."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("aggregate freeze manifest must be a regular file")
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("aggregate freeze manifest must be an object")
    _exact_keys(
        payload,
        {
            "schema_version",
            "artifact_type",
            "status",
            "identities",
            "terminals",
            "av_projection",
        },
        "aggregate freeze manifest",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "vqa-event-map-aggregate-freeze"
        or payload.get("status") != "frozen"
    ):
        raise ValueError("aggregate freeze manifest identity is invalid")
    identity = payload.get("identities")
    if not isinstance(identity, Mapping) or not isinstance(
        expected_identities, Mapping
    ):
        raise ValueError("aggregate freeze identities must be objects")
    _exact_keys(identity, FREEZE_IDENTITY_FIELDS, "aggregate freeze identities")
    _exact_keys(
        expected_identities,
        FREEZE_IDENTITY_FIELDS,
        "expected aggregate freeze identities",
    )
    if dict(identity) != dict(expected_identities):
        raise ValueError("aggregate freeze identity mismatch")
    if (
        not isinstance(expected_producer_row_fields, Sequence)
        or isinstance(expected_producer_row_fields, (str, bytes))
        or any(
            not isinstance(field, str) or not field
            for field in expected_producer_row_fields
        )
    ):
        raise ValueError("expected producer row fields are malformed")
    producer_row_fields = frozenset(expected_producer_row_fields)
    if len(producer_row_fields) != len(
        expected_producer_row_fields
    ) or not PRODUCER_REQUIRED_FIELDS.issubset(producer_row_fields):
        raise ValueError("expected producer row schema lacks required generic fields")
    if any(_key_tokens(field) & _FORBIDDEN_KEY_PARTS for field in producer_row_fields):
        raise ValueError(
            "expected producer row schema contains a label/reference field"
        )
    for field in (
        "source_bundle_sha256",
        "controller_sha256",
        "protocol_sha256",
        "inference_module_sha256",
        "runner_sha256",
    ):
        if not _is_sha256(identity[field]):
            raise ValueError(f"aggregate freeze {field} is invalid")
    for field in (
        "model_id",
        "model_revision",
        "runtime_image",
        "row_run_id",
        "producer_run_prefix",
        "output_uri_prefix",
    ):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"aggregate freeze {field} is missing")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", identity["model_revision"]) is None:
        raise ValueError("aggregate freeze model_revision is not immutable")
    if re.search(r"@sha256:[0-9a-f]{64}$", identity["runtime_image"]) is None:
        raise ValueError("aggregate freeze runtime_image is not immutable")
    if not identity["output_uri_prefix"].startswith("hf://"):
        raise ValueError("aggregate freeze output URI prefix is invalid")
    if (
        not isinstance(identity["runtime_identity"], Mapping)
        or not identity["runtime_identity"]
    ):
        raise ValueError("aggregate freeze runtime_identity is missing")
    if (
        identity["runtime_identity"].get("runtime_image") is not None
        and identity["runtime_identity"]["runtime_image"] != identity["runtime_image"]
    ):
        raise ValueError("aggregate freeze runtime identities disagree")
    if (
        not isinstance(expected_shard_counts, Sequence)
        or isinstance(expected_shard_counts, (str, bytes))
        or any(type(value) is not int for value in expected_shard_counts)
    ):
        raise ValueError("aggregate freeze shard counts are invalid")
    shard_counts = tuple(expected_shard_counts)
    if (
        len(shard_counts) != 4
        or any(value < 0 for value in shard_counts)
        or sum(shard_counts) != expected_count
    ):
        raise ValueError("aggregate freeze shard counts do not cover the panel")
    expected_av_by_index: dict[int, dict[str, Any]] = {}
    if expected_av_shards is None:
        if expected_count == EXPECTED_COUNT:
            raise ValueError(
                "production aggregate freeze requires exact expected AV shards"
            )
    else:
        if (
            not isinstance(expected_av_shards, Sequence)
            or isinstance(expected_av_shards, (str, bytes))
            or len(expected_av_shards) != 4
        ):
            raise ValueError("expected AV shard identities must contain four rows")
        expected_av_fields = {
            "shard_index",
            "run_id",
            "attempt_uuid",
            "output_identity",
            "payload_checksum",
            "row_count",
            "predictions_sha256",
            "terminal_sha256",
        }
        for raw_expected in expected_av_shards:
            if not isinstance(raw_expected, Mapping):
                raise ValueError("expected AV shard identity must be an object")
            _exact_keys(raw_expected, expected_av_fields, "expected AV shard identity")
            shard_index = raw_expected.get("shard_index")
            if type(shard_index) is not int or not 0 <= shard_index < 4:
                raise ValueError("expected AV shard index is invalid")
            if shard_index in expected_av_by_index:
                raise ValueError("expected AV shard identity is duplicated")
            _canonical_uuid(
                raw_expected.get("attempt_uuid"), field="expected AV attempt_uuid"
            )
            for field in (
                "payload_checksum",
                "predictions_sha256",
                "terminal_sha256",
            ):
                if not _is_sha256(raw_expected.get(field)):
                    raise ValueError(f"expected AV shard {field} is invalid")
            if (
                raw_expected.get("row_count") != shard_counts[shard_index]
                or not isinstance(raw_expected.get("run_id"), str)
                or not isinstance(raw_expected.get("output_identity"), str)
            ):
                raise ValueError("expected AV shard identity/count is invalid")
            expected_av_by_index[shard_index] = dict(raw_expected)

    terminals = payload.get("terminals")
    if not isinstance(terminals, list) or len(terminals) != 8:
        raise ValueError("aggregate freeze must contain all 8 producer terminals")
    terminal_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for wrapper in terminals:
        if not isinstance(wrapper, Mapping):
            raise ValueError("aggregate freeze terminal binding must be an object")
        _exact_keys(
            wrapper,
            {"condition_id", "shard_index", "terminal_sha256", "terminal"},
            "aggregate freeze terminal binding",
        )
        condition = wrapper.get("condition_id")
        shard_index = wrapper.get("shard_index")
        if (
            condition not in {"v-map", "av-map"}
            or type(shard_index) is not int
            or not 0 <= shard_index < 4
        ):
            raise ValueError("aggregate freeze terminal shard identity is invalid")
        key = (str(condition), shard_index)
        if key in terminal_by_key:
            raise ValueError("aggregate freeze contains a duplicate producer terminal")
        terminal = wrapper.get("terminal")
        if not isinstance(terminal, Mapping):
            raise ValueError("aggregate freeze producer terminal must be an object")
        _exact_keys(
            terminal, PRODUCER_TERMINAL_FIELDS, "aggregate freeze producer terminal"
        )
        expected_run = (
            f"{identity['producer_run_prefix']}-{condition}-chunk-{shard_index:02d}"
        )
        if (
            terminal.get("schema_version") != 1
            or terminal.get("status") != "complete"
            or terminal.get("run_id") != expected_run
            or terminal.get("output_identity")
            != f"{identity['output_uri_prefix']}-{expected_run}"
            or terminal.get("row_count") != shard_counts[shard_index]
        ):
            raise ValueError("aggregate freeze producer terminal identity mismatch")
        _canonical_uuid(terminal.get("attempt_uuid"), field="producer attempt_uuid")
        for field in (
            "payload_checksum",
            "request_file_sha256",
            "source_bundle_sha256",
            "controller_sha256",
            "protocol_sha256",
            "inference_module_sha256",
            "runner_sha256",
        ):
            if not _is_sha256(terminal.get(field)):
                raise ValueError(
                    f"aggregate freeze producer terminal {field} is invalid"
                )
        for field in (
            "source_bundle_sha256",
            "controller_sha256",
            "protocol_sha256",
            "inference_module_sha256",
            "runner_sha256",
        ):
            if terminal[field] != identity[field]:
                raise ValueError(
                    "aggregate freeze producer terminal source identity mismatch"
                )
        _validate_producer_timings(terminal.get("producer_timings"))
        if wrapper.get("terminal_sha256") != _canonical_sha256(terminal):
            raise ValueError("aggregate freeze producer terminal hash mismatch")
        if condition == "av-map" and expected_av_by_index:
            expected_av = expected_av_by_index[shard_index]
            for expected_field, terminal_field in (
                ("run_id", "run_id"),
                ("attempt_uuid", "attempt_uuid"),
                ("output_identity", "output_identity"),
                ("payload_checksum", "payload_checksum"),
                ("row_count", "row_count"),
            ):
                if terminal.get(terminal_field) != expected_av[expected_field]:
                    raise ValueError("aggregate freeze AV shard identity mismatch")
            if wrapper.get("terminal_sha256") != expected_av["terminal_sha256"]:
                raise ValueError("aggregate freeze AV terminal binding mismatch")
        terminal_by_key[key] = dict(wrapper)
    if set(terminal_by_key) != {
        (condition, shard) for condition in ("v-map", "av-map") for shard in range(4)
    }:
        raise ValueError("aggregate freeze does not cover all 8 producer terminals")

    projection = payload.get("av_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("aggregate freeze AV projection must be an object")
    _exact_keys(
        projection,
        {
            "condition_id",
            "row_count",
            "canonical_id_digest",
            "rows_sha256",
            "shards",
            "rows",
        },
        "aggregate freeze AV projection",
    )
    rows = projection.get("rows")
    shards = projection.get("shards")
    if (
        projection.get("condition_id") != "av-map"
        or projection.get("row_count") != expected_count
        or projection.get("canonical_id_digest") != expected_panel_id_digest
        or not isinstance(rows, list)
        or len(rows) != expected_count
        or not isinstance(shards, list)
        or len(shards) != 4
    ):
        raise ValueError("aggregate freeze AV projection identity mismatch")
    if projection.get("rows_sha256") != _canonical_sha256(rows):
        raise ValueError("aggregate freeze AV projection rows hash mismatch")
    normalized_rows: list[dict[str, Any]] = []
    cursor = 0
    for shard_index, binding in enumerate(shards):
        if not isinstance(binding, Mapping):
            raise ValueError("aggregate freeze AV shard binding must be an object")
        _exact_keys(
            binding,
            {
                "shard_index",
                "row_start",
                "row_stop",
                "row_count",
                "rows_sha256",
                "predictions_sha256",
                "terminal_sha256",
            },
            "aggregate freeze AV shard binding",
        )
        stop = cursor + shard_counts[shard_index]
        shard_rows = rows[cursor:stop]
        if (
            binding.get("shard_index") != shard_index
            or binding.get("row_start") != cursor
            or binding.get("row_stop") != stop
            or binding.get("row_count") != shard_counts[shard_index]
            or binding.get("rows_sha256") != _canonical_sha256(shard_rows)
            or not _is_sha256(binding.get("predictions_sha256"))
            or binding.get("terminal_sha256")
            != terminal_by_key[("av-map", shard_index)]["terminal_sha256"]
        ):
            raise ValueError("aggregate freeze AV shard projection mismatch")
        if expected_av_by_index and (
            binding.get("predictions_sha256")
            != expected_av_by_index[shard_index]["predictions_sha256"]
            or binding.get("terminal_sha256")
            != expected_av_by_index[shard_index]["terminal_sha256"]
        ):
            raise ValueError("aggregate freeze AV shard checksum identity mismatch")
        normalized_rows.extend(
            _validate_frozen_producer_row(
                row,
                identity=identity,
                shard_index=shard_index,
                expected_fields=producer_row_fields,
            )
            for row in shard_rows
        )
        cursor = stop
    video_ids = [row["video_id"] for row in normalized_rows]
    if (
        video_ids != sorted(video_ids)
        or _canonical_id_digest(video_ids) != expected_panel_id_digest
    ):
        raise ValueError(
            "aggregate freeze AV projection canonical order/digest mismatch"
        )
    return {
        "rows": normalized_rows,
        "manifest_sha256": _sha256_file(path),
        "rows_sha256": projection["rows_sha256"],
        "canonical_id_digest": expected_panel_id_digest,
        "row_count": expected_count,
        "terminal_count": len(terminals),
        "av_terminal_count": 4,
        "identities": dict(identity),
        "producer_row_fields": sorted(producer_row_fields),
        "producer_row_schema_sha256": _canonical_sha256(sorted(producer_row_fields)),
        "av_shards": [
            {
                "shard_index": shard_index,
                "run_id": terminal_by_key[("av-map", shard_index)]["terminal"][
                    "run_id"
                ],
                "attempt_uuid": terminal_by_key[("av-map", shard_index)]["terminal"][
                    "attempt_uuid"
                ],
                "output_identity": terminal_by_key[("av-map", shard_index)]["terminal"][
                    "output_identity"
                ],
                "payload_checksum": terminal_by_key[("av-map", shard_index)][
                    "terminal"
                ]["payload_checksum"],
                "row_count": shard_counts[shard_index],
                "predictions_sha256": shards[shard_index]["predictions_sha256"],
                "terminal_sha256": terminal_by_key[("av-map", shard_index)][
                    "terminal_sha256"
                ],
            }
            for shard_index in range(4)
        ],
    }


def _checksum_entries(checksum_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split("  ", maxsplit=1)
        if len(fields) != 2 or not SHA256.fullmatch(fields[0]):
            raise ValueError(
                f"terminal checksum manifest line {line_number} is malformed"
            )
        relative = fields[1].removeprefix("./")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("terminal checksum manifest contains an unsafe path")
        if relative in entries:
            raise ValueError("terminal checksum manifest contains duplicate paths")
        entries[relative] = fields[0]
    if not entries:
        raise ValueError("terminal checksum manifest is empty")
    return entries


def _validate_whisper_counts(
    *,
    example_count: Any,
    available_count: Any,
    quality_rejected_count: Any,
    expected_count: int,
    expected_available: int,
    expected_quality_rejected: int,
) -> None:
    counts = (example_count, available_count, quality_rejected_count)
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Whisper availability counts are malformed")
    if available_count + quality_rejected_count != example_count:
        raise ValueError("Whisper availability counts do not reconcile")
    if counts != (expected_count, expected_available, expected_quality_rejected):
        raise ValueError(
            "Whisper availability contract mismatch; reject superseded artifacts "
            f"and require {expected_available}/{expected_quality_rejected}"
        )


def _required_bundle_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"frozen baseline source is missing regular file {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("frozen baseline source path escapes bundle")
    return resolved


def _validated_baseline_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("frozen baseline contract must be an object")
    _exact_keys(value, _BASELINE_CONTRACT_FIELDS, "frozen baseline contract")
    for field in _BASELINE_CONTRACT_FIELDS:
        if field.endswith("sha256") and not _is_sha256(value.get(field)):
            raise ValueError(f"frozen baseline contract {field} is invalid")
    for field in ("run_name", "payload_uri", "model_id", "model_revision"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"frozen baseline contract {field} is missing")
    _validate_whisper_counts(
        example_count=value.get("example_count"),
        available_count=value.get("available_count"),
        quality_rejected_count=value.get("quality_rejected_count"),
        expected_count=value["example_count"],
        expected_available=value["available_count"],
        expected_quality_rejected=value["quality_rejected_count"],
    )
    return dict(value)


def validate_frozen_baseline_bundle(
    root: Path,
    *,
    contract: Mapping[str, Any] = PRODUCTION_BASELINE_CONTRACT,
) -> dict[str, Any]:
    """Authenticate the immutable Whisper/G-F bundle and expose safe sources."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen baseline bundle must be a regular directory")
    root = root.resolve(strict=True)
    expected = _validated_baseline_contract(contract)
    relative_files = {
        "terminal_sha256": "terminal-status.json",
        "checksums_sha256": "payload/SHA256SUMS",
        "evidence_marker_sha256": "payload/evidence/evidence-marker.json",
        "manifest_sha256": "payload/evidence/manifest.json",
        "requests_sha256": "payload/evidence/requests.accepted-whisper.json",
        "frames_requests_sha256": "payload/evidence/requests.frames.json",
        "evidence_sha256": "payload/evidence/evidence.json",
        "whisper_sha256": "payload/evidence/whisper.json",
        "whisper_summary_sha256": "payload/evidence/whisper-summary.json",
    }
    paths: dict[str, Path] = {}
    for contract_field, relative in relative_files.items():
        path = _required_bundle_file(root, relative)
        if _sha256_file(path) != expected[contract_field]:
            raise ValueError(f"frozen baseline {relative} hash mismatch")
        paths[contract_field] = path

    terminal = _load_json(paths["terminal_sha256"])
    if not isinstance(terminal, Mapping):
        raise ValueError("frozen baseline terminal must be an object")
    _exact_keys(
        terminal,
        {"payload_sha256", "payload_uri", "run_name", "schema_version", "status"},
        "frozen baseline terminal",
    )
    if (
        terminal.get("schema_version") != 1
        or terminal.get("status") != "passed"
        or terminal.get("run_name") != expected["run_name"]
        or terminal.get("payload_sha256") != expected["checksums_sha256"]
    ):
        raise ValueError("frozen baseline terminal identity mismatch")
    checksum_entries = _checksum_entries(paths["checksums_sha256"])
    for contract_field, relative in relative_files.items():
        if contract_field in {"terminal_sha256", "checksums_sha256"}:
            continue
        payload_relative = relative.removeprefix("payload/")
        if checksum_entries.get(payload_relative) != expected[contract_field]:
            raise ValueError(
                f"frozen baseline checksum does not bind {payload_relative}"
            )

    marker = _load_json(paths["evidence_marker_sha256"])
    if not isinstance(marker, Mapping):
        raise ValueError("frozen baseline evidence marker must be an object")
    _exact_keys(
        marker,
        {
            "canonical_id_digest",
            "checksum_file",
            "evidence_bundle_id",
            "example_count",
            "frame_policy",
            "label_policy",
            "manifest_sha256",
            "requests_sha256",
            "schema_version",
            "whisper_policy",
            "whisper_sha256",
        },
        "frozen baseline evidence marker",
    )
    if (
        marker.get("schema_version") != 1
        or marker.get("label_policy") != "label-free-evidence"
        or marker.get("checksum_file") != "SHA256SUMS"
        or marker.get("evidence_bundle_id") != expected["run_name"]
        or marker.get("example_count") != expected["example_count"]
        or marker.get("canonical_id_digest") != expected["canonical_id_digest"]
        or marker.get("manifest_sha256") != expected["manifest_sha256"]
        or marker.get("requests_sha256") != expected["requests_sha256"]
        or marker.get("whisper_sha256") != expected["whisper_sha256"]
    ):
        raise ValueError("frozen baseline evidence marker identity mismatch")

    manifest = _load_json(paths["manifest_sha256"])
    whisper_payload = _load_json(paths["whisper_sha256"])
    whisper_summary = _load_json(paths["whisper_summary_sha256"])
    evidence_payload = _load_json(paths["evidence_sha256"])
    requests_payload = _load_json(paths["requests_sha256"])
    frames_requests_payload = _load_json(paths["frames_requests_sha256"])
    if not isinstance(manifest, Mapping) or not isinstance(whisper_payload, Mapping):
        raise ValueError("frozen baseline manifest/Whisper payload is malformed")
    manifest_rows = _safe_json_rows(
        manifest, source="frozen baseline manifest", names=("examples",)
    )
    whisper_rows = _safe_json_rows(
        whisper_payload, source="frozen baseline Whisper", names=("results",)
    )
    evidence_rows = _safe_json_rows(
        evidence_payload, source="frozen baseline evidence", names=("examples",)
    )
    request_rows = _safe_json_rows(
        requests_payload, source="frozen baseline requests", names=("rows",)
    )
    frames_request_rows = _safe_json_rows(
        frames_requests_payload,
        source="frozen baseline frames requests",
        names=("rows",),
    )
    count = expected["example_count"]
    if not all(
        len(rows) == count
        for rows in (
            manifest_rows,
            whisper_rows,
            evidence_rows,
            request_rows,
            frames_request_rows,
        )
    ):
        raise ValueError("frozen baseline row coverage mismatch")
    ordered_ids = [
        _validate_video_id(row.get("video_id"), field="frozen baseline video_id")
        for row in manifest_rows
    ]
    if (
        ordered_ids != sorted(ordered_ids)
        or _canonical_id_digest(ordered_ids) != expected["canonical_id_digest"]
    ):
        raise ValueError("frozen baseline canonical panel digest mismatch")
    for source_name, rows in (
        ("Whisper", whisper_rows),
        ("evidence", evidence_rows),
        ("requests", request_rows),
        ("frames requests", frames_request_rows),
    ):
        if [row.get("video_id") for row in rows] != ordered_ids:
            raise ValueError(f"frozen baseline {source_name} order/coverage mismatch")
    if _canonical_sha256(whisper_rows) != expected["whisper_rows_sha256"]:
        raise ValueError("frozen baseline whole Whisper rows hash mismatch")
    model = whisper_payload.get("model")
    summary_model = (
        whisper_summary.get("model") if isinstance(whisper_summary, Mapping) else None
    )
    if (
        whisper_payload.get("schema_version") != 1
        or whisper_payload.get("label_policy") != "label-free-transcript-evidence"
        or whisper_payload.get("transcription_mode") != "translate-only"
        or not isinstance(model, Mapping)
        or model.get("model_id") != expected["model_id"]
        or model.get("model_revision") != expected["model_revision"]
        or not isinstance(summary_model, Mapping)
        or summary_model.get("model_id") != expected["model_id"]
        or summary_model.get("model_revision") != expected["model_revision"]
    ):
        raise ValueError("frozen baseline Whisper model/revision mismatch")
    available_count = sum(
        row.get("transcript_included") is True for row in whisper_rows
    )
    rejected_count = sum(
        row.get("transcript_included") is False for row in whisper_rows
    )
    _validate_whisper_counts(
        example_count=whisper_payload.get("example_count"),
        available_count=available_count,
        quality_rejected_count=rejected_count,
        expected_count=count,
        expected_available=expected["available_count"],
        expected_quality_rejected=expected["quality_rejected_count"],
    )
    if (
        whisper_payload.get("accepted_count") != available_count
        or not isinstance(whisper_summary, Mapping)
        or whisper_summary.get("example_count") != count
        or whisper_summary.get("accepted_count") != available_count
    ):
        raise ValueError("frozen baseline Whisper summary/count mismatch")

    artifact_sha256 = _canonical_sha256(
        {field: expected[field] for field in relative_files}
    )
    f0_rows: list[dict[str, Any]] = []
    for index, video_id in enumerate(ordered_ids):
        manifest_row = manifest_rows[index]
        whisper_row = whisper_rows[index]
        evidence_row = evidence_rows[index]
        request = request_rows[index]
        frames_request = frames_request_rows[index]
        question = manifest_row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{video_id} frozen baseline question is missing")
        if any(
            item.get("question") != question
            for item in (evidence_row, request, frames_request)
        ):
            raise ValueError(f"{video_id} frozen baseline question mismatch")
        _exact_keys(
            request,
            {
                "run_id",
                "condition_id",
                "video_id",
                "question",
                "frame_paths",
                "frame_timestamps",
                "transcript",
            },
            "frozen F0 request",
        )
        _exact_keys(frames_request, set(request), "frozen frames-only request")
        expected_transcript = (
            whisper_row.get("selected_transcript")
            if whisper_row.get("transcript_included") is True
            else None
        )
        if request.get("transcript") != expected_transcript:
            raise ValueError(f"{video_id} frozen F0 transcript mismatch")
        if frames_request.get("transcript") is not None:
            raise ValueError(f"{video_id} frozen frames-only transcript must be null")
        raw_frames = evidence_row.get("frames")
        request_paths = request.get("frame_paths")
        request_times = request.get("frame_timestamps")
        frames_request_paths = frames_request.get("frame_paths")
        frames_request_times = frames_request.get("frame_timestamps")
        if not all(
            isinstance(value, list) and len(value) == 12
            for value in (
                raw_frames,
                request_paths,
                request_times,
                frames_request_paths,
                frames_request_times,
            )
        ):
            raise ValueError(f"{video_id} frozen F0 evidence budget mismatch")
        frames: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(raw_frames, start=1):
            if not isinstance(frame, Mapping) or set(frame) != {
                "path",
                "requested_timestamp_seconds",
                "selected_frame_timestamp_seconds",
                "sha256",
            }:
                raise ValueError(f"{video_id} frozen F0 frame schema mismatch")
            name = frame.get("path")
            name_path = PurePosixPath(name) if isinstance(name, str) else None
            if name_path is None or len(name_path.parts) != 1:
                raise ValueError(f"{video_id} frozen F0 frame path is unsafe")
            relative = f"evidence/frames/{video_id}/{name}"
            source = _required_bundle_file(root / "payload", relative)
            digest = frame.get("sha256")
            if (
                not _is_sha256(digest)
                or checksum_entries.get(relative) != digest
                or _sha256_file(source) != digest
            ):
                raise ValueError(f"{video_id} frozen F0 frame checksum/hash mismatch")
            request_path = PurePosixPath(str(request_paths[frame_index - 1]))
            frames_request_path = PurePosixPath(
                str(frames_request_paths[frame_index - 1])
            )
            expected_tail = ("evidence", "frames", video_id, name)
            if (
                tuple(request_path.parts[-4:]) != expected_tail
                or tuple(frames_request_path.parts[-4:]) != expected_tail
                or request_times[frame_index - 1]
                != frame.get("requested_timestamp_seconds")
                or frames_request_times[frame_index - 1]
                != frame.get("requested_timestamp_seconds")
            ):
                raise ValueError(f"{video_id} frozen F0 request/evidence drift")
            frames.append(
                {
                    "source_frame_ordinal": frame_index,
                    "source_path": str(source),
                    "source_relative_path": relative,
                    "requested_timestamp_seconds": frame["requested_timestamp_seconds"],
                    "actual_pts_seconds": frame["selected_frame_timestamp_seconds"],
                    "frame_byte_sha256": digest,
                }
            )
        f0_rows.append(
            {
                "video_id": video_id,
                "question": question,
                "duration_seconds": evidence_row.get("duration_seconds"),
                "source_media_sha256": evidence_row.get("media_sha256"),
                "source_artifact_sha256": artifact_sha256,
                "source_request_sha256": _canonical_sha256(request),
                "source_evidence_sha256": _canonical_sha256(evidence_row),
                "source_frames_request_sha256": _canonical_sha256(frames_request),
                "source_whisper_sha256": _canonical_sha256(whisper_row),
                "transcript": expected_transcript,
                "transcript_sha256": (
                    _sha256_bytes(expected_transcript.encode("utf-8"))
                    if isinstance(expected_transcript, str)
                    else None
                ),
                "frames": frames,
            }
        )
    return {
        "whisper_rows": whisper_rows,
        "f0_rows": f0_rows,
        "whisper_binding": {
            "artifact_sha256": artifact_sha256,
            "terminal_sha256": expected["terminal_sha256"],
            "checksums_sha256": expected["checksums_sha256"],
            "evidence_marker_sha256": expected["evidence_marker_sha256"],
            "whisper_sha256": expected["whisper_sha256"],
            "whisper_rows_sha256": expected["whisper_rows_sha256"],
            "model_id": expected["model_id"],
            "model_revision": expected["model_revision"],
            "canonical_id_digest": expected["canonical_id_digest"],
            "row_count": count,
            "available_count": available_count,
            "quality_rejected_count": rejected_count,
        },
        "f0_binding": {
            "artifact_sha256": artifact_sha256,
            "evidence_sha256": expected["evidence_sha256"],
            "requests_sha256": expected["requests_sha256"],
            "frames_requests_sha256": expected["frames_requests_sha256"],
            "manifest_sha256": expected["manifest_sha256"],
            "canonical_id_digest": expected["canonical_id_digest"],
            "row_count": count,
        },
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        video_id = raw.get("video_id")
        try:
            video_id = _validate_video_id(
                video_id, field=f"{source} row {index} video_id"
            )
        except ValueError as error:
            raise ValueError(str(error)) from error
        if video_id in indexed:
            raise ValueError(f"{source} contains duplicate video_id {video_id}")
        indexed[video_id] = dict(raw)
    return indexed


_FORBIDDEN_KEY_PARTS = frozenset(
    {
        "annotation",
        "annotations",
        "reference",
        "references",
        "ground_truth",
        "groundtruth",
        "label",
        "labels",
        "metric",
        "metrics",
        "score",
        "scores",
        "grade",
        "grades",
        "prediction",
        "predictions",
        "rationale",
        "reasoning",
    }
)


def _key_tokens(key: Any) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", str(key).casefold()) if part}


def _audit_label_free(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _key_tokens(key) & _FORBIDDEN_KEY_PARTS:
                raise ValueError(f"label/reference leakage in {location}.{key}")
            _audit_label_free(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _audit_label_free(nested, location=f"{location}[{index}]")


_PANEL_FIELDS = frozenset(
    {
        "video_id",
        "question",
        "duration_seconds",
        "media_path",
        "media_sha256",
        "whisper_segment_count",
        "question_stratum",
        "source_media_sha256",
        "source_duration_seconds",
    }
)


def _validate_panel_row(
    raw: Mapping[str, Any], *, media_root: Path | None
) -> dict[str, Any]:
    unknown = set(raw) - _PANEL_FIELDS
    if unknown:
        raise ValueError(f"panel schema drift: unexpected fields {sorted(unknown)}")
    _audit_label_free(raw, location="panel")
    video_id = raw.get("video_id")
    question = raw.get("question")
    video_id = _validate_video_id(video_id, field="panel video_id")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{video_id} question must be nonempty text")
    duration = _finite(raw.get("duration_seconds"), f"{video_id} duration_seconds")
    if duration <= 0:
        raise ValueError(f"{video_id} duration_seconds must be positive")
    media_sha256 = raw.get("media_sha256", raw.get("source_media_sha256"))
    if not _is_sha256(media_sha256):
        raise ValueError(f"{video_id} media_sha256 must be lowercase SHA-256")
    media_value = raw.get("media_path")
    if not isinstance(media_value, str) or not media_value:
        raise ValueError(f"{video_id} media_path must be nonempty text")
    media = Path(media_value)
    if not media.is_absolute() and media_root is not None:
        media = media_root / media
    if media.is_symlink():
        raise ValueError(f"{video_id} media must not be a symlink")
    try:
        media = media.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{video_id} media is missing") from error
    if media.is_symlink() or not media.is_file() or media.stat().st_size <= 0:
        raise ValueError(f"{video_id} media must be a nonempty regular file")
    if _sha256_file(media) != media_sha256:
        raise ValueError(f"{video_id} source media hash mismatch")
    count = raw.get("whisper_segment_count")
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
    ):
        raise ValueError(f"{video_id} whisper_segment_count must be non-negative")
    declared_source_duration = raw.get("source_duration_seconds")
    if (
        declared_source_duration is not None
        and abs(
            _finite(declared_source_duration, f"{video_id} source duration") - duration
        )
        > _TIME_EPSILON
    ):
        raise ValueError(f"{video_id} panel source duration mismatch")
    return {
        "video_id": video_id,
        "question": question.strip(),
        "duration_seconds": duration,
        "media_path": str(media),
        "media_sha256": media_sha256,
        "whisper_segment_count": count,
    }


_EVENT_MAP_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "condition_id",
        "slice_id",
        "video_id",
        "question",
        "question_stratum",
        "media_path",
        "media_sha256",
        "source_media_sha256",
        "duration_seconds",
        "source_duration_seconds",
        "normalized_duration_seconds",
        "normalized_media_path",
        "normalized_media_sha256",
        "status",
        "error_type",
        "primary_valid",
        "retry_valid",
        "retry_used",
        "request_sha256",
        "controller_request_sha256",
        "primary_request_sha256",
        "retry_request_sha256",
        "response_sha256",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_identity",
        "memory_telemetry",
        "elapsed_seconds",
        "answer",
        "abstain",
        "confidence",
        "events",
        "selected_intervals",
        "response",
        "result",
    }
)


def _extract_private_map_response(
    row: Mapping[str, Any], *, duration: float
) -> dict[str, Any]:
    candidate = row.get("response", row.get("result"))
    if candidate is None:
        required = {"answer", "abstain", "confidence", "events", "selected_intervals"}
        if not required.issubset(row):
            raise ValueError("event-map row lacks its complete terminal response")
        candidate = {key: row[key] for key in required}
    if not isinstance(candidate, Mapping):
        raise ValueError("event-map terminal response must be an object")
    required = {"answer", "abstain", "confidence", "events", "selected_intervals"}
    if set(candidate) != required:
        raise ValueError("event-map terminal response schema drift")
    try:
        return validate_private_response(
            {key: candidate[key] for key in required},
            duration_seconds=duration,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"event-map terminal response is invalid: {error}") from error


def _validate_event_map_row(
    raw: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str] = _EVENT_MAP_FIELDS,
) -> dict[str, Any]:
    unknown = set(raw) - allowed_fields
    if unknown:
        raise ValueError(f"event-map schema drift: unexpected fields {sorted(unknown)}")
    video_id = raw.get("video_id")
    if video_id != panel["video_id"]:
        raise ValueError(f"{panel['video_id']} event-map video_id mismatch")
    if raw.get("condition_id") != "av-map":
        raise ValueError(f"{video_id} event-map condition must be av-map")
    if raw.get("status") != "passed":
        raise ValueError(f"{video_id} event-map row did not pass")
    if raw.get("error_type") not in (None, ""):
        raise ValueError(f"{video_id} event-map row contains an error")
    if raw.get("question") != panel["question"]:
        raise ValueError(f"{video_id} event-map question mismatch")
    source_media = raw.get("source_media_sha256", raw.get("media_sha256"))
    if source_media != panel["media_sha256"]:
        raise ValueError(f"{video_id} event-map media identity mismatch")
    source_duration = raw.get("source_duration_seconds", raw.get("duration_seconds"))
    if (
        abs(
            _finite(source_duration, f"{video_id} source duration")
            - panel["duration_seconds"]
        )
        > _TIME_EPSILON
    ):
        raise ValueError(f"{video_id} event-map duration mismatch")
    normalized_duration = raw.get("normalized_duration_seconds")
    if (
        normalized_duration is not None
        and abs(
            _finite(normalized_duration, f"{video_id} normalized duration")
            - panel["duration_seconds"]
        )
        > 0.25
    ):
        raise ValueError(f"{video_id} normalized duration drift")
    if not _is_sha256(raw.get("response_sha256")):
        raise ValueError(f"{video_id} event-map response hash is missing or invalid")
    response = _extract_private_map_response(raw, duration=panel["duration_seconds"])
    intervals = [
        {
            "event_id": interval["event_id"],
            "start_seconds": float(interval["start"]),
            "end_seconds": float(interval["end"]),
        }
        for interval in response["selected_intervals"]
    ]
    if not intervals:
        raise ValueError(f"{video_id} event-map selected_intervals are empty")
    return {
        "video_id": video_id,
        "selected_intervals": intervals,
        "response_sha256": raw["response_sha256"],
        "map_run_id": raw.get("run_id"),
    }


def _format_whisper_timestamp(seconds: float) -> str:
    """Reproduce the frozen producer's millisecond-truncating formatter."""

    total_milliseconds = max(0, int(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _whisper_line_records(
    text: str, *, duration: float, chunks: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind source-line bytes one-to-one to the producer's true chunk bounds."""

    if not isinstance(chunks, list):
        raise ValueError("Whisper english_translation.chunks must be a list")
    raw_lines = text.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    source_line_index = 0
    previous_nonempty_start: float | None = None
    for chunk_index, raw_chunk in enumerate(chunks):
        if not isinstance(raw_chunk, Mapping):
            raise ValueError("Whisper chunks must contain objects")
        chunk_text = raw_chunk.get("text")
        if not isinstance(chunk_text, str):
            raise ValueError("Whisper chunk text must be a string")
        start = _finite(raw_chunk.get("start"), "Whisper chunk start")
        if start < 0:
            raise ValueError("Whisper chunk start must be nonnegative")
        if not chunk_text:
            # The frozen artifact contains one empty decoder chunk with inverted
            # bounds. It produced no model-facing source line and is not a
            # selectable transcript segment.
            continue
        if (
            previous_nonempty_start is not None
            and start <= previous_nonempty_start + _TIME_EPSILON
        ):
            raise ValueError("Whisper nonempty chunk starts must be strictly ordered")
        previous_nonempty_start = start
        if source_line_index >= len(raw_lines):
            raise ValueError("Whisper chunk has no matching timestamped source line")
        source_line_bytes = raw_lines[source_line_index].encode("utf-8")
        source_line = raw_lines[source_line_index].rstrip("\r\n")
        expected_line = f"[{_format_whisper_timestamp(start)}] {chunk_text}"
        if source_line != expected_line:
            raise ValueError(
                "Whisper chunk does not exactly match timestamped source-line bytes"
            )
        source_hash = _sha256_bytes(source_line_bytes)
        end_value = raw_chunk.get("end")
        if end_value is None:
            if start < duration:
                raise ValueError("Whisper in-duration chunk has no true end boundary")
            excluded.append(
                {
                    "segment_id": chunk_index,
                    "source_line": source_line,
                    "source_line_bytes_sha256": source_hash,
                    "start_seconds": start,
                    "end_seconds": None,
                    "reason": "out_of_duration_unbounded",
                }
            )
            source_line_index += 1
            continue
        end = _finite(end_value, "Whisper chunk end")
        if end <= start:
            raise ValueError("Whisper nonempty chunk bounds are invalid")
        if start >= duration:
            excluded.append(
                {
                    "segment_id": chunk_index,
                    "source_line": source_line,
                    "source_line_bytes_sha256": source_hash,
                    "start_seconds": start,
                    "end_seconds": end,
                    "reason": "out_of_duration_bounded",
                }
            )
            source_line_index += 1
            continue
        records.append(
            {
                "segment_id": chunk_index,
                "source_line_index": source_line_index,
                "start_seconds": start,
                "end_seconds": end,
                "text": chunk_text,
                "source_line": source_line,
                "source_line_bytes_sha256": source_hash,
                "segment_sha256": source_hash,
                "original_segment_sha256": source_hash,
            }
        )
        source_line_index += 1
    if source_line_index != len(raw_lines):
        raise ValueError(
            "Whisper timestamped source line has no matching nonempty chunk"
        )
    return records, excluded


def _validate_whisper_row(
    raw: Mapping[str, Any], panel: Mapping[str, Any]
) -> dict[str, Any]:
    allowed = {
        "video_id",
        "status",
        "text",
        "transcript",
        "selected_transcript",
        "timestamped_text",
        "source_sha256",
        "whisper_sha256",
        "source_audio_sha256",
        "source_audio_duration_seconds",
        "transcript_included",
        "original_language_transcript",
        "english_translation",
        "quality",
        "source_language",
        "elapsed_seconds",
        "segment_count",
        "runtime_identity",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Whisper schema drift: unexpected fields {sorted(unknown)}")
    _audit_label_free(raw, location="Whisper")
    video_id = raw.get("video_id")
    if video_id != panel["video_id"]:
        raise ValueError(f"{panel['video_id']} Whisper video_id mismatch")
    if raw.get("status") not in (None, "passed"):
        raise ValueError(f"{video_id} Whisper row did not pass")
    transcript_included = raw.get("transcript_included")
    if transcript_included is False:
        quality = raw.get("quality")
        quality_status = quality.get("status") if isinstance(quality, Mapping) else None
        if not isinstance(quality_status, str) or not quality_status.startswith(
            "rejected"
        ):
            raise ValueError(
                f"{video_id} unavailable Whisper quality status is invalid"
            )
        source_audio_sha256 = raw.get("source_audio_sha256")
        if source_audio_sha256 is not None and not _is_sha256(source_audio_sha256):
            raise ValueError(f"{video_id} Whisper source audio hash is invalid")
        source_audio_duration = raw.get("source_audio_duration_seconds")
        if source_audio_duration is not None:
            source_audio_duration = _finite(
                source_audio_duration, f"{video_id} Whisper source audio duration"
            )
            if source_audio_duration <= 0:
                raise ValueError(f"{video_id} Whisper source audio duration is invalid")
        return {
            "video_id": video_id,
            "text": None,
            "source_sha256": None,
            "segments": [],
            "excluded_segments": [],
            "transcript_available": False,
            "availability_status": "quality_rejected",
            "availability_provenance": {
                "quality_status": quality_status,
                "source_audio_sha256": source_audio_sha256,
                "source_audio_duration_seconds": source_audio_duration,
                "source_row_sha256": _canonical_sha256(raw),
            },
        }
    translation = raw.get("english_translation")
    if not isinstance(translation, Mapping):
        raise ValueError(f"{video_id} Whisper english_translation is missing")
    text = translation.get("timestamped_text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{video_id} Whisper transcript is missing")
    for field in ("timestamped_text", "selected_transcript"):
        if raw.get(field) != text:
            raise ValueError(
                f"{video_id} Whisper {field} drifts from English translation"
            )
    source_sha256 = _sha256_bytes(text.encode("utf-8"))
    for field in ("source_sha256", "whisper_sha256"):
        if raw.get(field) is not None and raw[field] != source_sha256:
            raise ValueError(f"{video_id} Whisper {field} mismatch")
    segments, excluded_segments = _whisper_line_records(
        text,
        duration=panel["duration_seconds"],
        chunks=translation.get("chunks"),
    )
    declared_count = raw.get("segment_count")
    if declared_count is not None and declared_count != len(segments):
        raise ValueError(f"{video_id} Whisper segment count mismatch")
    expected_count = panel.get("whisper_segment_count")
    if expected_count is not None and expected_count != len(segments):
        raise ValueError(f"{video_id} Whisper segment count mismatches panel")
    return {
        "video_id": video_id,
        "text": text,
        "source_sha256": source_sha256,
        "segments": segments,
        "excluded_segments": excluded_segments,
        "transcript_available": True,
        "availability_status": "accepted",
        "availability_provenance": {
            "quality_status": (
                raw["quality"].get("status")
                if isinstance(raw.get("quality"), Mapping)
                else None
            ),
            "source_audio_sha256": raw.get("source_audio_sha256"),
            "source_audio_duration_seconds": raw.get("source_audio_duration_seconds"),
            "source_row_sha256": _canonical_sha256(raw),
        },
    }


def _select_whisper_segments(
    whisper: Mapping[str, Any], intervals: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if whisper.get("transcript_available") is False:
        return {
            "full_text": None,
            "full_source_sha256": None,
            "selected_text": None,
            "selected_source_sha256": None,
            "selected_sha256": None,
            "selected_segment_ids": [],
            "selected_segments": [],
            "excluded_segments": [],
            "transcript_available": False,
            "availability_status": whisper.get("availability_status"),
            "availability_provenance": dict(whisper.get("availability_provenance", {})),
        }
    selected = [
        segment
        for segment in whisper["segments"]
        if any(
            segment["end_seconds"] > interval["start_seconds"] + _TIME_EPSILON
            and segment["start_seconds"] < interval["end_seconds"] - _TIME_EPSILON
            for interval in intervals
        )
    ]
    selected_ids = [segment["segment_id"] for segment in selected]
    source_lines = whisper["text"].splitlines(keepends=True)
    selected_bytes = b"".join(
        source_lines[segment["source_line_index"]].encode("utf-8")
        for segment in selected
    )
    selected_text = selected_bytes.decode("utf-8")
    return {
        "full_text": whisper["text"],
        "full_source_sha256": whisper["source_sha256"],
        "selected_text": selected_text or None,
        "selected_source_sha256": whisper["source_sha256"],
        "selected_sha256": _sha256_bytes(selected_bytes),
        "selected_segment_ids": selected_ids,
        "selected_segments": selected,
        "excluded_segments": list(whisper.get("excluded_segments", [])),
        "transcript_available": True,
        "availability_status": whisper.get("availability_status", "accepted"),
        "availability_provenance": dict(whisper.get("availability_provenance", {})),
    }


def _transcript_provenance(transcript: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transcript_available": bool(transcript["transcript_available"]),
        "availability_status": transcript.get("availability_status"),
        "availability_provenance": dict(transcript.get("availability_provenance", {})),
        "full_source_sha256": transcript.get("full_source_sha256"),
        "selected_source_sha256": transcript.get("selected_source_sha256"),
        "selected_sha256": transcript.get("selected_sha256"),
        "selected_segment_ids": list(transcript["selected_segment_ids"]),
        "selected_segments": list(transcript["selected_segments"]),
        "excluded_segments": list(transcript.get("excluded_segments", [])),
    }


def _transcript_budget(
    selected: Mapping[str, Any], shifted: Mapping[str, Any]
) -> dict[str, Any]:
    selected_bytes = len((selected.get("selected_text") or "").encode("utf-8"))
    shifted_bytes = len((shifted.get("selected_text") or "").encode("utf-8"))
    segment_count_equal = len(selected["selected_segment_ids"]) == len(
        shifted["selected_segment_ids"]
    )
    byte_count_equal = selected_bytes == shifted_bytes
    mismatch_reason = None
    if not segment_count_equal:
        mismatch_reason = "whisper_overlap_segment_count_differs"
    elif not byte_count_equal:
        mismatch_reason = "whisper_overlap_bytes_differ"
    return {
        "selected_segment_count": len(selected["selected_segment_ids"]),
        "shifted_segment_count": len(shifted["selected_segment_ids"]),
        "selected_byte_count": selected_bytes,
        "shifted_byte_count": shifted_bytes,
        "segment_count_equal": segment_count_equal,
        "byte_count_equal": byte_count_equal,
        "mismatch_reason": mismatch_reason,
    }


_FROZEN_F0_ROW_FIELDS = frozenset(
    {
        "video_id",
        "question",
        "duration_seconds",
        "source_media_sha256",
        "source_artifact_sha256",
        "source_request_sha256",
        "source_evidence_sha256",
        "source_frames_request_sha256",
        "source_whisper_sha256",
        "transcript",
        "transcript_sha256",
        "frames",
    }
)
_FROZEN_F0_FRAME_FIELDS = frozenset(
    {
        "source_frame_ordinal",
        "source_path",
        "source_relative_path",
        "requested_timestamp_seconds",
        "actual_pts_seconds",
        "frame_byte_sha256",
    }
)


def _validate_frozen_f0_row(
    raw: Mapping[str, Any],
    *,
    panel: Mapping[str, Any],
    whisper: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(raw, _FROZEN_F0_ROW_FIELDS, "frozen F0 row")
    _audit_label_free(raw, location="frozen F0")
    video_id = panel["video_id"]
    if (
        raw.get("video_id") != video_id
        or raw.get("question") != panel["question"]
        or raw.get("source_media_sha256") != panel["media_sha256"]
        or abs(
            _finite(raw.get("duration_seconds"), f"{video_id} frozen F0 duration")
            - panel["duration_seconds"]
        )
        > _TIME_EPSILON
    ):
        raise ValueError(f"{video_id} frozen F0 row identity mismatch")
    for field in (
        "source_artifact_sha256",
        "source_request_sha256",
        "source_evidence_sha256",
        "source_frames_request_sha256",
        "source_whisper_sha256",
    ):
        if not _is_sha256(raw.get(field)):
            raise ValueError(f"{video_id} frozen F0 {field} is invalid")
    expected_transcript = whisper["text"] if whisper["transcript_available"] else None
    transcript = raw.get("transcript")
    if transcript != expected_transcript:
        raise ValueError(
            f"{video_id} frozen F0 transcript drifts from accepted Whisper"
        )
    transcript_sha256 = (
        _sha256_bytes(transcript.encode("utf-8"))
        if isinstance(transcript, str)
        else None
    )
    if raw.get("transcript_sha256") != transcript_sha256:
        raise ValueError(f"{video_id} frozen F0 transcript hash mismatch")
    raw_frames = raw.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != 12:
        raise ValueError(f"{video_id} frozen F0 must contain exactly twelve frames")
    frames: list[dict[str, Any]] = []
    for index, frame in enumerate(raw_frames, start=1):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{video_id} frozen F0 frame must be an object")
        _exact_keys(frame, _FROZEN_F0_FRAME_FIELDS, "frozen F0 frame")
        if frame.get("source_frame_ordinal") != index:
            raise ValueError(f"{video_id} frozen F0 frame ordinal mismatch")
        relative_value = frame.get("source_relative_path")
        relative = (
            PurePosixPath(relative_value) if isinstance(relative_value, str) else None
        )
        if relative is None or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{video_id} frozen F0 frame relative path is unsafe")
        source_value = frame.get("source_path")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"{video_id} frozen F0 frame source path is missing")
        source_candidate = Path(source_value)
        if source_candidate.is_symlink():
            raise ValueError(f"{video_id} frozen F0 frame must not be a symlink")
        try:
            source = source_candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{video_id} frozen F0 frame is missing") from error
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"{video_id} frozen F0 frame is empty")
        digest = frame.get("frame_byte_sha256")
        if not _is_sha256(digest) or _sha256_file(source) != digest:
            raise ValueError(f"{video_id} frozen F0 frame byte hash mismatch")
        requested = _finite(
            frame.get("requested_timestamp_seconds"),
            f"{video_id} frozen F0 requested timestamp",
        )
        actual = _finite(
            frame.get("actual_pts_seconds"), f"{video_id} frozen F0 actual PTS"
        )
        if (
            requested < 0
            or requested > panel["duration_seconds"]
            or actual < 0
            or actual > panel["duration_seconds"] + 0.25
        ):
            raise ValueError(f"{video_id} frozen F0 timestamp is outside duration")
        frames.append(
            {
                **dict(frame),
                "source_path": str(source),
                "requested_timestamp_seconds": round(requested, 6),
                "actual_pts_seconds": round(actual, 6),
            }
        )
    for field in ("requested_timestamp_seconds", "actual_pts_seconds"):
        values = [frame[field] for frame in frames]
        if values != sorted(values) or len(set(values)) != 12:
            raise ValueError(f"{video_id} frozen F0 {field} must be unique and ordered")
    return {**dict(raw), "frames": frames, "transcript_sha256": transcript_sha256}


def _bind_frozen_f0_frames(
    f0: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Bind legacy frozen frame PTS to current decoded identities without rerendering."""

    available = [dict(frame) for frame in inventory]
    used: set[tuple[int, int]] = set()
    bound: list[dict[str, Any]] = []
    for source in f0["frames"]:
        candidates = [
            frame
            for frame in available
            if (frame["stream_index"], frame["frame_index"]) not in used
        ]
        selected = min(
            candidates,
            key=lambda frame: (
                abs(frame["actual_pts_seconds"] - source["actual_pts_seconds"]),
                frame["actual_pts_seconds"],
                frame["stream_index"],
                frame["frame_index"],
            ),
        )
        delta = abs(selected["actual_pts_seconds"] - source["actual_pts_seconds"])
        if delta > 0.002 + _TIME_EPSILON:
            raise ValueError(
                f"{f0['video_id']} frozen F0 PTS cannot bind to a decoded frame identity"
            )
        identity = (selected["stream_index"], selected["frame_index"])
        used.add(identity)
        bound.append(
            {
                **dict(source),
                "stream_index": selected["stream_index"],
                "frame_index": selected["frame_index"],
                "decoded_frame_ordinal": selected.get(
                    "decoded_frame_ordinal", selected["frame_index"]
                ),
                "time_base": selected["time_base"],
                "raw_pts": selected.get("raw_pts"),
                "identity_actual_pts_seconds": selected["actual_pts_seconds"],
            }
        )
    return bound


def _materialize_frozen_f0_frames(
    *,
    row: Mapping[str, Any],
    f0: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    stage: Path,
) -> list[dict[str, Any]]:
    bound = _bind_frozen_f0_frames(f0, inventory)
    frame_plan_sha256 = _canonical_sha256(
        [
            {
                "requested_timestamp_seconds": frame["requested_timestamp_seconds"],
                "actual_pts_seconds": frame["actual_pts_seconds"],
                "source_frame_ordinal": frame["source_frame_ordinal"],
                "frame_byte_sha256": frame["frame_byte_sha256"],
            }
            for frame in bound
        ]
    )
    materialized: list[dict[str, Any]] = []
    for index, source in enumerate(bound):
        target = (
            stage
            / "frames"
            / row["video_id"]
            / PLAN_DIRS["F0"]
            / f"frame-{index:03d}.jpg"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["source_path"], target)
        digest = _sha256_file(target)
        if digest != source["frame_byte_sha256"]:
            raise ValueError(f"{row['video_id']} frozen F0 byte reuse mismatch")
        materialized.append(
            {
                "requested_timestamp_seconds": source["requested_timestamp_seconds"],
                "actual_pts_seconds": source["actual_pts_seconds"],
                "stream_index": source["stream_index"],
                "frame_index": source["frame_index"],
                "decoded_frame_ordinal": source["decoded_frame_ordinal"],
                "time_base": source["time_base"],
                "raw_pts": source.get("raw_pts"),
                "identity_actual_pts_seconds": source["identity_actual_pts_seconds"],
                "decoder_runtime_identity": {
                    "decoder": "frozen-gf-ffmpeg",
                    "runtime_version": None,
                    "runtime_recorded": False,
                },
                "decoder_runtime_version": None,
                "source_media_sha256": row["media_sha256"],
                "frame_byte_sha256": digest,
                "frame_sha256": digest,
                "slot_type": "global",
                "frame_plan_sha256": frame_plan_sha256,
                "path": target.relative_to(stage).as_posix(),
                "construction_fallback": None,
                "frozen_source_frame_ordinal": source["source_frame_ordinal"],
                "frozen_source_relative_path": source["source_relative_path"],
            }
        )
    _validate_plan_frames("F0", materialized)
    return materialized


def _validate_inventory(
    records: Sequence[Mapping[str, Any]], *, row: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise ValueError(f"{row['video_id']} decoded frame inventory is missing")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    points: list[float] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{row['video_id']} inventory row {index} is malformed")
        stream_index = raw.get("stream_index")
        frame_index = raw.get("frame_index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(
                f"{row['video_id']} inventory row {index} has bad identity"
            )
        identity = (stream_index, frame_index)
        if identity in identities:
            raise ValueError(
                f"{row['video_id']} inventory has duplicate frame identity"
            )
        identities.add(identity)
        ordinal = raw.get("decoded_frame_ordinal", frame_index)
        if ordinal != frame_index:
            raise ValueError(f"{row['video_id']} inventory ordinal/frame mismatch")
        time_base, _ = _exact_time_base(raw.get("time_base"))
        actual = _finite(
            raw.get("actual_pts_seconds"), f"{row['video_id']} inventory PTS"
        )
        if actual < 0 or actual > row["duration_seconds"] + 0.25:
            raise ValueError(f"{row['video_id']} inventory PTS is outside duration")
        if raw.get("source_media_sha256") != row["media_sha256"]:
            raise ValueError(f"{row['video_id']} inventory media identity mismatch")
        raw_pts = raw.get("raw_pts")
        if raw_pts is not None and (
            isinstance(raw_pts, bool) or not isinstance(raw_pts, int)
        ):
            raise ValueError(f"{row['video_id']} inventory raw_pts is malformed")
        normalized.append(
            {
                **dict(raw),
                "stream_index": stream_index,
                "frame_index": frame_index,
                "decoded_frame_ordinal": frame_index,
                "time_base": time_base,
                "actual_pts_seconds": round(actual, 6),
                "source_media_sha256": row["media_sha256"],
            }
        )
        points.append(round(actual, 6))
    if points != sorted(points) or len(set(points)) != len(points):
        raise ValueError(
            f"{row['video_id']} inventory actual PTS must be unique and ordered"
        )
    if len(normalized) < 12:
        raise ValueError(
            f"{row['video_id']} inventory must contain at least twelve frames"
        )
    return normalized


def _probe_frame_inventory(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read exact frame identities without a pixel decode before one PyAV pass."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to inventory production video frames")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=index,time_base:frame=pts,best_effort_timestamp",
            "-of",
            "json",
            row["media_path"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"{row['video_id']} ffprobe inventory is malformed") from error
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    if (
        not isinstance(streams, list)
        or len(streams) != 1
        or not isinstance(frames, list)
    ):
        raise ValueError(
            f"{row['video_id']} ffprobe found no unique primary video stream"
        )
    stream = streams[0]
    if not isinstance(stream, Mapping):
        raise ValueError(f"{row['video_id']} ffprobe stream metadata is malformed")
    stream_index = stream.get("index")
    if type(stream_index) is not int or stream_index < 0:
        raise ValueError(f"{row['video_id']} ffprobe stream index is malformed")
    time_base, fraction = _exact_time_base(stream.get("time_base"))
    inventory: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"{row['video_id']} ffprobe frame metadata is malformed")
        raw_pts = frame.get("pts")
        if raw_pts is None:
            raw_pts = frame.get("best_effort_timestamp")
        if type(raw_pts) is not int:
            raise ValueError(f"{row['video_id']} ffprobe frame PTS is malformed")
        actual = float(raw_pts * fraction)
        if not math.isfinite(actual) or actual < 0:
            raise ValueError(f"{row['video_id']} ffprobe frame PTS is invalid")
        inventory.append(
            {
                "stream_index": stream_index,
                "frame_index": ordinal,
                "decoded_frame_ordinal": ordinal,
                "actual_pts_seconds": round(actual, 6),
                "time_base": time_base,
                "raw_pts": raw_pts,
                "source_media_sha256": row["media_sha256"],
            }
        )
    return _validate_inventory(inventory, row=row)


def _unique_planned_frames(
    planned_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[int, int], dict[str, Any]]:
    needed: dict[tuple[int, int], dict[str, Any]] = {}
    for frames in planned_by_id.values():
        for frame in frames:
            identity = (int(frame["stream_index"]), int(frame["frame_index"]))
            existing = needed.get(identity)
            if existing is not None:
                for field in ("actual_pts_seconds", "time_base", "source_media_sha256"):
                    if existing.get(field) != frame.get(field):
                        raise ValueError(
                            f"planned frame identity {identity} has conflicting {field}"
                        )
                continue
            needed[identity] = dict(frame)
    return needed


def _decode_unique_frames(
    row: Mapping[str, Any],
    needed: Mapping[tuple[int, int], Mapping[str, Any]],
    cache_dir: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Decode all uncached identities in one video-open/decode traversal."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    missing: dict[tuple[int, int], Mapping[str, Any]] = {}
    for identity, frame in needed.items():
        source = frame.get("frame_path", frame.get("path", frame.get("image_path")))
        if isinstance(source, str) and source:
            source_candidate = Path(source)
            if source_candidate.is_symlink():
                raise ValueError(
                    f"{row['video_id']} inventory frame must not be a symlink"
                )
            source_path = source_candidate.resolve(strict=True)
            if not source_path.is_file():
                raise ValueError(f"{row['video_id']} inventory frame is missing")
            target = cache_dir / f"{identity[0]}-{identity[1]}.png"
            shutil.copyfile(source_path, target)
            result[identity] = {
                **dict(frame),
                "frame_path": str(target),
                "decoded_frame_ordinal": frame.get(
                    "decoded_frame_ordinal", identity[1]
                ),
                "frame_byte_sha256": _sha256_file(target),
            }
        else:
            missing[identity] = frame
    if not missing:
        return result
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "PyAV is required to decode production evidence frames"
        ) from error
    stream_indexes = {identity[0] for identity in missing}
    if len(stream_indexes) != 1:
        raise ValueError("planned frames span multiple video streams")
    stream_index = next(iter(stream_indexes))
    with av.open(str(row["media_path"])) as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        stream = next(
            (candidate for candidate in streams if candidate.index == stream_index),
            None,
        )
        if stream is None:
            raise ValueError(f"{row['video_id']} planned video stream is missing")
        for ordinal, decoded in enumerate(container.decode(stream)):
            identity = (stream.index, ordinal)
            if identity not in missing:
                continue
            raw_pts = getattr(decoded, "pts", None)
            time_base_value = getattr(decoded, "time_base", None) or getattr(
                stream, "time_base", None
            )
            time_base, fraction = _exact_time_base(time_base_value)
            actual = (
                float(raw_pts * fraction)
                if raw_pts is not None
                else _finite(getattr(decoded, "time", None), "decoded PTS")
            )
            target = cache_dir / f"{identity[0]}-{identity[1]}.png"
            decoded.to_image().convert("RGB").save(target, format="PNG", optimize=True)
            result[identity] = {
                **dict(missing[identity]),
                "frame_path": str(target),
                "stream_index": stream.index,
                "frame_index": ordinal,
                "decoded_frame_ordinal": ordinal,
                "actual_pts_seconds": round(actual, 6),
                "time_base": time_base,
                "raw_pts": raw_pts,
                "frame_byte_sha256": _sha256_file(target),
            }
            if len(result) == len(needed):
                break
    if set(result) != set(needed):
        missing_identity = sorted(set(needed) - set(result))[0]
        raise ValueError(
            f"{row['video_id']} decoded frame {missing_identity} is missing"
        )
    return result


def _contact_sheet_cell_label(
    *, video_id: str, plan_id: str, index: int, frame: Mapping[str, Any]
) -> str:
    return (
        f"{video_id} {plan_id} {index + 1:02d}\n"
        f"req={float(frame['requested_timestamp_seconds']):.6f}s "
        f"pts={float(frame['actual_pts_seconds']):.6f}s"
    )


def _default_contact_sheet(
    *,
    video_id: str,
    plan_id: str,
    frame_paths: list[str],
    frame_metadata: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> Mapping[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required to render contact sheets") from error
    if len(frame_paths) != 12 or len(frame_metadata) != 12:
        raise ValueError(f"{video_id} {plan_id} contact sheet requires twelve frames")
    width, height = 320, 250
    sheet = Image.new("RGB", (width * 4, height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(frame_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((width - 8, 220), Image.Resampling.LANCZOS)
        x = (index % 4) * width
        y = (index // 4) * height
        sheet.paste(
            image, (x + (width - image.width) // 2, y + 28 + (220 - image.height) // 2)
        )
        draw.multiline_text(
            (x + 5, y + 5),
            _contact_sheet_cell_label(
                video_id=video_id,
                plan_id=plan_id,
                index=index,
                frame=frame_metadata[index],
            ),
            fill="black",
            spacing=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "frame_count": 12,
    }


def _shift_intervals(
    intervals: Sequence[Mapping[str, Any]], *, duration: float
) -> tuple[list[dict[str, Any]], str | None]:
    """Place same-length intervals in the complement of the selected union."""

    normalized = [
        {
            "event_id": interval["event_id"],
            "start_seconds": float(interval["start_seconds"]),
            "end_seconds": float(interval["end_seconds"]),
        }
        for interval in intervals
    ]
    free: list[tuple[float, float]] = []
    cursor = 0.0
    for interval in normalized:
        if interval["start_seconds"] > cursor + _TIME_EPSILON:
            free.append((cursor, interval["start_seconds"]))
        cursor = max(cursor, interval["end_seconds"])
    if cursor < duration - _TIME_EPSILON:
        free.append((cursor, duration))
    lengths = [item["end_seconds"] - item["start_seconds"] for item in normalized]

    def search(
        order: tuple[int, ...], position: int, previous_end: float
    ) -> list[tuple[int, float, float]] | None:
        if position == len(order):
            return []
        source_index = order[position]
        length = lengths[source_index]
        for free_start, free_end in free:
            start = max(free_start, previous_end)
            if start + length > free_end + _TIME_EPSILON:
                continue
            start = round(start, 6)
            end = round(start + length, 6)
            tail = search(order, position + 1, end)
            if tail is not None:
                return [(source_index, start, end), *tail]
        return None

    placement = next(
        (
            candidate
            for order in permutations(range(len(lengths)))
            if (candidate := search(order, 0, 0.0)) is not None
        ),
        None,
    )
    if placement is None:
        return [], "no_disjoint_shift_possible"
    shifted = sorted(
        [
            {
                "event_id": normalized[source_index]["event_id"],
                "start_seconds": start,
                "end_seconds": end,
            }
            for source_index, start, end in placement
        ],
        key=lambda interval: (interval["start_seconds"], interval["end_seconds"]),
    )
    if shifted == normalized:
        raise AssertionError("disjoint shifted intervals must not equal treatment")
    shifted_lengths = [item["end_seconds"] - item["start_seconds"] for item in shifted]
    if any(
        not math.isclose(actual, expected, abs_tol=_TIME_EPSILON)
        for actual, expected in zip(
            sorted(shifted_lengths), sorted(lengths), strict=True
        )
    ):
        raise AssertionError("shifted interval lengths changed")
    if any(
        shifted_item["start_seconds"] < selected_item["end_seconds"] - _TIME_EPSILON
        and shifted_item["end_seconds"] > selected_item["start_seconds"] + _TIME_EPSILON
        for shifted_item in shifted
        for selected_item in normalized
    ):
        raise AssertionError("shifted intervals overlap treatment")
    return shifted, None


def _validate_plan_frames(plan_id: str, frames: Sequence[Mapping[str, Any]]) -> None:
    if len(frames) != 12:
        raise ValueError(f"{plan_id} must contain exactly twelve frames")
    points = [float(frame["actual_pts_seconds"]) for frame in frames]
    identities = [(frame["stream_index"], frame["frame_index"]) for frame in frames]
    if points != sorted(points) or len(set(points)) != 12:
        raise ValueError(f"{plan_id} actual PTS must be unique and ordered")
    if len(set(identities)) != 12:
        raise ValueError(f"{plan_id} decoded frame identities must be unique")
    if plan_id in {"H/F", "H/S", "Shifted H/S"}:
        if sum(frame["slot_type"] == "global" for frame in frames) != 6:
            raise ValueError(f"{plan_id} must contain six global slots")
        if sum(frame["slot_type"] == "local" for frame in frames) != 6:
            raise ValueError(f"{plan_id} must contain six local slots")
    else:
        if any(frame["slot_type"] != "global" for frame in frames):
            raise ValueError(f"{plan_id} must contain only global slots")


def _materialize_frames(
    *,
    row: Mapping[str, Any],
    plan_id: str,
    planned: Sequence[Mapping[str, Any]],
    stage: Path,
    renderer: FrameRenderer | None,
    frame_cache: Mapping[tuple[int, int], Mapping[str, Any]] | None,
    decoder_runtime_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frame_plan_sha256 = _canonical_sha256(
        [
            {
                key: frame.get(key)
                for key in (
                    "slot_kind",
                    "desired_time_seconds",
                    "requested_time_seconds",
                    "stream_index",
                    "frame_index",
                    "actual_pts_seconds",
                    "time_base",
                    "source_media_sha256",
                )
            }
            for frame in planned
        ]
    )
    decoder_runtime_version = decoder_runtime_identity.get(
        "decoder_runtime_version", decoder_runtime_identity.get("version", "unknown")
    )
    materialized: list[dict[str, Any]] = []
    plan_dir = stage / "frames" / row["video_id"] / PLAN_DIRS[plan_id]
    for index, source in enumerate(planned):
        target = plan_dir / f"frame-{index:03d}.png"
        identity = (source["stream_index"], source["frame_index"])
        if frame_cache is not None:
            rendered = frame_cache.get(identity)
            if rendered is None:
                raise ValueError(
                    f"{row['video_id']} {plan_id} batch decoder omitted frame {identity}"
                )
        else:
            if renderer is None:
                raise RuntimeError("a frame renderer or batch decoder is required")
            rendered = renderer(row, source, plan_id, target)
        if not isinstance(rendered, Mapping):
            raise TypeError("frame renderer must return an object")
        required_identity = {
            "stream_index",
            "frame_index",
            "decoded_frame_ordinal",
            "time_base",
            "actual_pts_seconds",
        }
        if not required_identity.issubset(rendered):
            missing = sorted(required_identity - set(rendered))
            raise ValueError(
                f"{row['video_id']} {plan_id} renderer lacks actual identity {missing[0]}"
            )
        path_value = rendered.get("frame_path", rendered.get("path", target))
        path_candidate = Path(str(path_value))
        if path_candidate.is_symlink():
            raise ValueError(f"{row['video_id']} {plan_id} frame must not be a symlink")
        path = path_candidate.resolve(strict=True)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"{row['video_id']} {plan_id} frame is missing")
        if path != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            path = target.resolve(strict=True)
        expected_identity = {
            "stream_index": source["stream_index"],
            "frame_index": source["frame_index"],
            "decoded_frame_ordinal": source.get(
                "decoded_frame_ordinal", source["frame_index"]
            ),
            "time_base": source["time_base"],
            "actual_pts_seconds": round(float(source["actual_pts_seconds"]), 6),
        }
        for key, expected in expected_identity.items():
            if key in rendered:
                actual = rendered[key]
                if key == "actual_pts_seconds":
                    if abs(float(actual) - float(expected)) > _TIME_EPSILON:
                        raise ValueError(
                            f"{row['video_id']} {plan_id} frame actual PTS mismatch"
                        )
                elif actual != expected:
                    raise ValueError(
                        f"{row['video_id']} {plan_id} frame {key} mismatch"
                    )
        frame_bytes_sha256 = _sha256_file(path)
        declared_hash = rendered.get("frame_byte_sha256", rendered.get("sha256"))
        if declared_hash is not None and declared_hash != frame_bytes_sha256:
            raise ValueError(f"{row['video_id']} {plan_id} frame byte hash mismatch")
        relative = path.relative_to(stage.resolve()).as_posix()
        materialized.append(
            {
                "requested_timestamp_seconds": round(
                    float(source["requested_time_seconds"]), 6
                ),
                "actual_pts_seconds": round(float(source["actual_pts_seconds"]), 6),
                "stream_index": source["stream_index"],
                "frame_index": source["frame_index"],
                "decoded_frame_ordinal": source.get(
                    "decoded_frame_ordinal", source["frame_index"]
                ),
                "time_base": source["time_base"],
                "decoder_runtime_identity": dict(decoder_runtime_identity),
                "decoder_runtime_version": decoder_runtime_version,
                "source_media_sha256": row["media_sha256"],
                "frame_byte_sha256": frame_bytes_sha256,
                "frame_sha256": frame_bytes_sha256,
                "slot_type": source["slot_kind"],
                "frame_plan_sha256": frame_plan_sha256,
                "path": relative,
                "construction_fallback": source.get("construction_fallback"),
            }
        )
    _validate_plan_frames(plan_id, materialized)
    return materialized


def _validate_sheet(
    sheet: Mapping[str, Any], *, stage: Path, plan_id: str, video_id: str
) -> dict[str, Any]:
    path_value = sheet.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{video_id} {plan_id} contact sheet path is missing")
    path_candidate = Path(path_value)
    if path_candidate.is_symlink():
        raise ValueError(f"{video_id} {plan_id} contact sheet must not be a symlink")
    path = path_candidate.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(stage.resolve()):
        raise ValueError(f"{video_id} {plan_id} contact sheet escapes output")
    digest = sheet.get("sha256")
    actual = _sha256_file(path)
    if digest != actual:
        raise ValueError(f"{video_id} {plan_id} contact sheet hash mismatch")
    return {
        "path": path.relative_to(stage.resolve()).as_posix(),
        "sha256": actual,
        "frame_count": 12,
    }


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _write_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )


def _write_checksums(stage: Path) -> Path:
    entries = []
    for path in sorted(
        item
        for item in stage.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        entries.append(
            f"{_sha256_file(path)}  ./{path.relative_to(stage).as_posix()}\n"
        )
    target = stage / "SHA256SUMS"
    _write_atomic(target, "".join(entries).encode("utf-8"))
    return target


def _runtime_identity(
    value: Mapping[str, Any] | None, *, fixture_mode: bool
) -> dict[str, Any]:
    if value is not None:
        if not fixture_mode:
            raise ValueError("production decoder identity cannot be caller supplied")
        if not isinstance(value, Mapping) or not value:
            raise ValueError("decoder_runtime_identity must be a nonempty object")
        return dict(value)
    try:
        import av
        from PIL import Image, features
    except ImportError as error:
        raise RuntimeError(
            "production materialization requires PyAV and Pillow"
        ) from error
    library_versions = {
        name: ".".join(str(part) for part in version)
        for name, version in sorted(getattr(av, "library_versions", {}).items())
    }
    if "libavcodec" not in library_versions:
        raise RuntimeError("PyAV did not expose its libavcodec runtime identity")
    jpeg_version = features.version("jpg")
    if not isinstance(jpeg_version, str) or not jpeg_version:
        raise RuntimeError("Pillow did not expose its libjpeg runtime identity")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to pin the FFmpeg runtime identity")
    ffprobe_version = subprocess.run(
        [ffprobe, "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "decoder": "PyAV",
        "decoder_runtime_version": str(av.__version__),
        "ffmpeg_library_versions": library_versions,
        "ffprobe_version": ffprobe_version,
        "pillow_version": str(Image.__version__),
        "libjpeg_version": jpeg_version,
        "python_version": platform.python_version(),
    }


def _source_binding_records(
    *,
    event_map_rows: Sequence[Mapping[str, Any]],
    whisper_rows: Sequence[Mapping[str, Any]],
    frozen_f0_rows: Sequence[Mapping[str, Any]],
    aggregate_freeze_binding: Mapping[str, Any] | None,
    whisper_binding: Mapping[str, Any] | None,
    frozen_f0_binding: Mapping[str, Any] | None,
    fixture_mode: bool,
) -> dict[str, Any]:
    if fixture_mode:
        fixture_row_fields = sorted(set().union(*(set(row) for row in event_map_rows)))
        return {
            "fixture_mode": True,
            "aggregate_freeze": {
                "manifest_sha256": _canonical_sha256(event_map_rows),
                "rows_sha256": _canonical_sha256(event_map_rows),
                "producer_row_fields": fixture_row_fields,
                "producer_row_schema_sha256": _canonical_sha256(fixture_row_fields),
            },
            "whisper": {
                "artifact_sha256": _canonical_sha256(whisper_rows),
                "whisper_rows_sha256": _canonical_sha256(whisper_rows),
            },
            "frozen_f0": {
                "artifact_sha256": _canonical_sha256(
                    [row.get("source_artifact_sha256") for row in frozen_f0_rows]
                )
            },
        }
    assert isinstance(aggregate_freeze_binding, Mapping)
    assert isinstance(whisper_binding, Mapping)
    assert isinstance(frozen_f0_binding, Mapping)
    required = {
        "aggregate freeze": (
            aggregate_freeze_binding,
            (
                "manifest_sha256",
                "rows_sha256",
                "producer_row_schema_sha256",
                "canonical_id_digest",
            ),
        ),
        "Whisper": (
            whisper_binding,
            (
                "artifact_sha256",
                "terminal_sha256",
                "checksums_sha256",
                "evidence_marker_sha256",
                "whisper_sha256",
                "whisper_rows_sha256",
                "canonical_id_digest",
            ),
        ),
        "frozen F0": (
            frozen_f0_binding,
            (
                "artifact_sha256",
                "evidence_sha256",
                "requests_sha256",
                "frames_requests_sha256",
                "manifest_sha256",
                "canonical_id_digest",
            ),
        ),
    }
    for name, (binding, fields) in required.items():
        for field in fields:
            if not _is_sha256(binding.get(field)):
                raise ValueError(f"authenticated {name} binding lacks valid {field}")
    producer_fields = aggregate_freeze_binding.get("producer_row_fields")
    av_shards = aggregate_freeze_binding.get("av_shards")
    if (
        not isinstance(producer_fields, list)
        or _canonical_sha256(producer_fields)
        != aggregate_freeze_binding["producer_row_schema_sha256"]
        or not isinstance(av_shards, list)
        or len(av_shards) != 4
        or aggregate_freeze_binding.get("row_count") != EXPECTED_COUNT
        or whisper_binding.get("row_count") != EXPECTED_COUNT
        or frozen_f0_binding.get("row_count") != EXPECTED_COUNT
    ):
        raise ValueError("authenticated aggregate freeze row schema binding is invalid")
    return {
        "fixture_mode": False,
        "aggregate_freeze": {
            "manifest_sha256": aggregate_freeze_binding["manifest_sha256"],
            "rows_sha256": aggregate_freeze_binding["rows_sha256"],
            "canonical_id_digest": aggregate_freeze_binding["canonical_id_digest"],
            "row_count": aggregate_freeze_binding["row_count"],
            "producer_row_fields": list(producer_fields),
            "producer_row_schema_sha256": aggregate_freeze_binding[
                "producer_row_schema_sha256"
            ],
            "terminal_count": aggregate_freeze_binding.get("terminal_count"),
            "av_terminal_count": aggregate_freeze_binding.get("av_terminal_count"),
            "identities": dict(aggregate_freeze_binding.get("identities", {})),
            "av_shards": [dict(shard) for shard in av_shards],
            "av_shards_sha256": _canonical_sha256(av_shards),
        },
        "whisper": {
            key: whisper_binding.get(key)
            for key in (
                "artifact_sha256",
                "terminal_sha256",
                "checksums_sha256",
                "evidence_marker_sha256",
                "whisper_sha256",
                "whisper_rows_sha256",
                "model_id",
                "model_revision",
                "available_count",
                "quality_rejected_count",
                "canonical_id_digest",
                "row_count",
            )
        },
        "frozen_f0": {
            key: frozen_f0_binding.get(key)
            for key in (
                "artifact_sha256",
                "evidence_sha256",
                "requests_sha256",
                "frames_requests_sha256",
                "manifest_sha256",
                "canonical_id_digest",
                "row_count",
            )
        },
    }


def build_evidence(
    *,
    panel_rows: Sequence[Mapping[str, Any]],
    event_map_rows: Sequence[Mapping[str, Any]],
    whisper_rows: Sequence[Mapping[str, Any]],
    frozen_f0_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    expected_count: int = EXPECTED_COUNT,
    expected_panel_id_digest: str = CANONICAL_PANEL_ID_DIGEST,
    media_root: Path | None = None,
    frame_inventory_provider: PanelInventoryProvider | None = None,
    frame_renderer: FrameRenderer | None = None,
    frame_batch_decoder: BatchFrameDecoder | None = None,
    contact_sheet_builder: ContactSheetBuilder | None = None,
    decoder_runtime_identity: Mapping[str, Any] | None = None,
    aggregate_freeze_binding: Mapping[str, Any] | None = None,
    whisper_binding: Mapping[str, Any] | None = None,
    frozen_f0_binding: Mapping[str, Any] | None = None,
    fixture_mode: bool = False,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Build five exact downstream evidence plans per frozen AV-map row."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be nonempty text")
    if type(fixture_mode) is not bool:
        raise TypeError("fixture_mode must be a bool")
    if not panel_rows:
        raise ValueError("panel_rows must be nonempty")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    if len(panel_rows) != expected_count:
        raise ValueError(f"panel must contain exactly {expected_count} rows")
    if not _is_sha256(expected_panel_id_digest):
        raise ValueError("expected panel ID digest must be lowercase SHA-256")
    if (
        expected_count == EXPECTED_COUNT
        and expected_panel_id_digest != CANONICAL_PANEL_ID_DIGEST
    ):
        raise ValueError("the 475-row panel must use the canonical ordered ID digest")
    injected_rendering = any(
        value is not None
        for value in (
            frame_inventory_provider,
            frame_renderer,
            frame_batch_decoder,
            contact_sheet_builder,
        )
    )
    if injected_rendering and not fixture_mode:
        raise ValueError(
            "frame providers/renderers are available only in explicit fixture mode"
        )
    if not fixture_mode and (
        expected_count != EXPECTED_COUNT
        or expected_panel_id_digest != CANONICAL_PANEL_ID_DIGEST
    ):
        raise ValueError(
            "production materialization requires the canonical 475-row panel"
        )
    if not fixture_mode and any(
        not isinstance(value, Mapping)
        for value in (aggregate_freeze_binding, whisper_binding, frozen_f0_binding)
    ):
        raise ValueError(
            "production materialization requires all authenticated source bindings"
        )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing evidence output {output_dir}"
        )
    normalized_panel = [
        _validate_panel_row(raw, media_root=media_root) for raw in panel_rows
    ]
    panel_by_id = _index_rows(normalized_panel, source="panel")
    panel_ids = list(panel_by_id)
    if panel_ids != sorted(panel_ids):
        raise ValueError("panel IDs must be in canonical sorted order")
    if _canonical_id_digest(panel_ids) != expected_panel_id_digest:
        raise ValueError("canonical ordered panel ID digest mismatch")
    event_by_id = _index_rows(event_map_rows, source="event-map")
    whisper_by_id = _index_rows(whisper_rows, source="Whisper")
    frozen_f0_by_id = _index_rows(frozen_f0_rows, source="frozen F0")
    if set(event_by_id) != set(panel_by_id):
        raise ValueError("event-map IDs do not exactly match panel IDs")
    if set(whisper_by_id) != set(panel_by_id):
        raise ValueError("Whisper IDs do not exactly match panel IDs")
    if set(frozen_f0_by_id) != set(panel_by_id):
        raise ValueError("frozen F0 IDs do not exactly match panel IDs")
    normalized_maps = {
        video_id: _validate_event_map_row(
            event_by_id[video_id],
            panel_by_id[video_id],
            allowed_fields=(
                frozenset(aggregate_freeze_binding["producer_row_fields"])
                if not fixture_mode
                and isinstance(aggregate_freeze_binding, Mapping)
                and isinstance(
                    aggregate_freeze_binding.get("producer_row_fields"), list
                )
                else _EVENT_MAP_FIELDS
            ),
        )
        for video_id in sorted(panel_by_id)
    }
    normalized_whisper = {
        video_id: _validate_whisper_row(whisper_by_id[video_id], panel_by_id[video_id])
        for video_id in sorted(panel_by_id)
    }
    normalized_f0 = {
        video_id: _validate_frozen_f0_row(
            frozen_f0_by_id[video_id],
            panel=panel_by_id[video_id],
            whisper=normalized_whisper[video_id],
        )
        for video_id in sorted(panel_by_id)
    }
    source_bindings = _source_binding_records(
        event_map_rows=event_map_rows,
        whisper_rows=whisper_rows,
        frozen_f0_rows=frozen_f0_rows,
        aggregate_freeze_binding=aggregate_freeze_binding,
        whisper_binding=whisper_binding,
        frozen_f0_binding=frozen_f0_binding,
        fixture_mode=fixture_mode,
    )
    if not fixture_mode:
        if (
            _canonical_sha256(event_map_rows)
            != source_bindings["aggregate_freeze"]["rows_sha256"]
        ):
            raise ValueError("authenticated aggregate freeze rows changed before build")
        if (
            _canonical_sha256(whisper_rows)
            != source_bindings["whisper"]["whisper_rows_sha256"]
        ):
            raise ValueError("authenticated Whisper rows changed before build")
        if any(
            row["source_artifact_sha256"]
            != source_bindings["frozen_f0"]["artifact_sha256"]
            for row in normalized_f0.values()
        ):
            raise ValueError("authenticated frozen F0 rows changed before build")
        if any(
            source_bindings[name]["canonical_id_digest"] != expected_panel_id_digest
            for name in ("aggregate_freeze", "whisper", "frozen_f0")
        ):
            raise ValueError("authenticated source canonical panel digests disagree")
        available_count = sum(
            row["transcript_available"] for row in normalized_whisper.values()
        )
        if (
            available_count != source_bindings["whisper"]["available_count"]
            or EXPECTED_COUNT - available_count
            != source_bindings["whisper"]["quality_rejected_count"]
        ):
            raise ValueError("authenticated Whisper availability counts changed")
    runtime = _runtime_identity(decoder_runtime_identity, fixture_mode=fixture_mode)
    provider = frame_inventory_provider or _probe_frame_inventory
    renderer = frame_renderer
    parent = output_dir.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    frozen_frame_bytes = sum(
        Path(frame["source_path"]).stat().st_size
        for f0 in normalized_f0.values()
        for frame in f0["frames"]
    )
    estimated_output_bytes = max(
        frozen_frame_bytes * 8,
        len(normalized_panel) * 8 * 1024 * 1024,
    )
    available_bytes = shutil.disk_usage(parent).free
    if available_bytes < estimated_output_bytes:
        raise OSError(
            "evidence preflight estimates insufficient free disk: "
            f"need {estimated_output_bytes}, have {available_bytes}"
        )
    preflight = {
        "bounded_output_bytes_estimate": estimated_output_bytes,
        "planned_video_metadata_probes": len(normalized_panel),
        "planned_video_pixel_decode_passes": len(normalized_panel),
        "maximum_plan_count_per_video": len(PLAN_IDS),
    }
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(parent))
    )
    try:
        manifest_rows: list[dict[str, Any]] = []
        request_rows: dict[str, dict[str, Any]] = {}
        for video_id in sorted(panel_by_id):
            row = panel_by_id[video_id]
            selected = normalized_maps[video_id]["selected_intervals"]
            shifted, shift_reason = _shift_intervals(
                selected, duration=row["duration_seconds"]
            )
            transcript = _select_whisper_segments(
                normalized_whisper[video_id], selected
            )
            shifted_transcript: dict[str, Any] | None = None
            if shift_reason is None:
                shifted_transcript = _select_whisper_segments(
                    normalized_whisper[video_id], shifted
                )
                transcript_budget = _transcript_budget(transcript, shifted_transcript)
                if not transcript_budget["segment_count_equal"]:
                    shift_reason = "transcript_segment_count_mismatch"
            else:
                transcript_budget = {
                    "selected_segment_count": len(transcript["selected_segment_ids"]),
                    "shifted_segment_count": None,
                    "selected_byte_count": len(
                        (transcript.get("selected_text") or "").encode("utf-8")
                    ),
                    "shifted_byte_count": None,
                    "segment_count_equal": None,
                    "byte_count_equal": None,
                    "mismatch_reason": shift_reason,
                }
            shift_eligible = shift_reason is None
            inventory = _validate_inventory(provider(row), row=row)
            selector_row = {
                "duration_seconds": row["duration_seconds"],
                "whisper_segment_count": len(normalized_whisper[video_id]["segments"]),
            }
            selected_selector = {
                "status": "localized",
                "intervals": [
                    {
                        "start_seconds": item["start_seconds"],
                        "end_seconds": item["end_seconds"],
                    }
                    for item in selected
                ],
                "whisper_segment_ids": [],
            }
            selected_plans = build_frame_plans(
                selector_row, selected_selector, inventory
            )
            planned_by_id = {
                "H/F": selected_plans["H"],
                "G/S": selected_plans["G"],
                "H/S": selected_plans["H"],
            }
            if shift_eligible:
                shifted_selector = {
                    "status": "localized",
                    "intervals": [
                        {
                            "start_seconds": item["start_seconds"],
                            "end_seconds": item["end_seconds"],
                        }
                        for item in shifted
                    ],
                    "whisper_segment_ids": [],
                }
                shifted_plans = build_frame_plans(
                    selector_row, shifted_selector, inventory
                )
                planned_by_id["Shifted H/S"] = shifted_plans["H"]
            frame_cache: Mapping[tuple[int, int], Mapping[str, Any]] | None = None
            frame_cache_dir = stage / ".frame-cache" / video_id
            if renderer is None:
                needed = _unique_planned_frames(planned_by_id)
                batch_decoder = frame_batch_decoder or _decode_unique_frames
                decoded = batch_decoder(row, needed, frame_cache_dir)
                if not isinstance(decoded, Mapping) or set(decoded) != set(needed):
                    raise ValueError(
                        f"{video_id} batch decoder did not return exactly the planned frame identities"
                    )
                frame_cache = decoded
            plan_records: dict[str, dict[str, Any]] = {}
            request_records: dict[str, dict[str, Any]] = {}
            sheets_dir = stage / "contact-sheets"
            for plan_id in ("F0", *planned_by_id):
                plan_transcript = (
                    shifted_transcript if plan_id == "Shifted H/S" else transcript
                )
                if plan_transcript is None:
                    raise AssertionError(
                        "eligible shifted plan lacks shifted transcript"
                    )
                if plan_id == "F0":
                    frames = _materialize_frozen_f0_frames(
                        row=row,
                        f0=normalized_f0[video_id],
                        inventory=inventory,
                        stage=stage,
                    )
                else:
                    frames = _materialize_frames(
                        row=row,
                        plan_id=plan_id,
                        planned=planned_by_id[plan_id],
                        stage=stage,
                        renderer=renderer,
                        frame_cache=frame_cache,
                        decoder_runtime_identity=runtime,
                    )
                frame_paths = [str(stage / frame["path"]) for frame in frames]
                sheet_path = sheets_dir / f"{video_id}-{PLAN_DIRS[plan_id]}.jpg"
                if contact_sheet_builder is None:
                    sheet = _default_contact_sheet(
                        video_id=video_id,
                        plan_id=plan_id,
                        frame_paths=frame_paths,
                        frame_metadata=frames,
                        output_path=sheet_path,
                    )
                else:
                    sheet = contact_sheet_builder(
                        video_id=video_id,
                        plan_id=plan_id,
                        frame_paths=frame_paths,
                        output_path=sheet_path,
                    )
                if not isinstance(sheet, Mapping):
                    raise TypeError("contact_sheet_builder must return an object")
                normalized_sheet = _validate_sheet(
                    sheet, stage=stage, plan_id=plan_id, video_id=video_id
                )
                transcript_mode = PLAN_TRANSCRIPTS[plan_id]
                transcript_text = (
                    normalized_f0[video_id]["transcript"]
                    if plan_id == "F0"
                    else plan_transcript["full_text"]
                    if transcript_mode == "full"
                    else plan_transcript["selected_text"]
                )
                transcript_sha256 = (
                    normalized_f0[video_id]["transcript_sha256"]
                    if plan_id == "F0"
                    else plan_transcript["full_source_sha256"]
                    if transcript_mode == "full"
                    else plan_transcript["selected_sha256"]
                )
                plan_records[plan_id] = {
                    "frame_count": 12,
                    "frames": frames,
                    "contact_sheet": normalized_sheet,
                    "transcript_mode": transcript_mode,
                    "transcript_sha256": transcript_sha256,
                    "selected_segment_ids": plan_transcript["selected_segment_ids"],
                }
                if plan_id == "F0":
                    plan_records[plan_id]["frozen_source_sha256"] = normalized_f0[
                        video_id
                    ]["source_artifact_sha256"]
                    plan_records[plan_id]["frozen_input_hashes"] = {
                        field: normalized_f0[video_id][field]
                        for field in (
                            "source_request_sha256",
                            "source_evidence_sha256",
                            "source_frames_request_sha256",
                            "source_whisper_sha256",
                        )
                    }
                request_records[plan_id] = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "condition_id": plan_id,
                    "video_id": video_id,
                    "question": row["question"],
                    "frame_paths": [frame["path"] for frame in frames],
                    "frame_timestamps": [
                        frame["requested_timestamp_seconds"] for frame in frames
                    ],
                    "transcript": transcript_text,
                }
                _audit_label_free(
                    request_records[plan_id], location=f"request[{video_id}.{plan_id}]"
                )
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "question": row["question"],
                    "duration_seconds": row["duration_seconds"],
                    "source_media_sha256": row["media_sha256"],
                    "selected_intervals": selected,
                    "shifted_intervals": shifted,
                    "shift_fallback_reason": shift_reason,
                    "shift_disjoint": shift_eligible,
                    "shifted_control": {
                        "eligible": shift_eligible,
                        "reason": shift_reason,
                        "transcript_quantity_measure": "selected_segment_count",
                    },
                    "transcript": _transcript_provenance(transcript),
                    "shifted_transcript": (
                        _transcript_provenance(shifted_transcript)
                        if shifted_transcript is not None
                        else None
                    ),
                    "transcript_budget": transcript_budget,
                    "plans": plan_records,
                }
            )
            request_rows[video_id] = request_records
            _write_atomic(
                stage / "transcripts" / video_id / "full.txt",
                (transcript["full_text"] or "").encode("utf-8"),
            )
            _write_atomic(
                stage / "transcripts" / video_id / "selected.txt",
                (transcript["selected_text"] or "").encode("utf-8"),
            )
            if shift_eligible and shifted_transcript is not None:
                _write_atomic(
                    stage / "transcripts" / video_id / "shifted.txt",
                    (shifted_transcript["selected_text"] or "").encode("utf-8"),
                )
            if frame_cache_dir.exists():
                shutil.rmtree(frame_cache_dir)
            if _sha256_file(Path(row["media_path"])) != row["media_sha256"]:
                raise ValueError(
                    f"{video_id} source media mutated during materialization"
                )
        manifest = {
            "schema_version": 2,
            "label_policy": "label-blind-event-map-evidence",
            "run_id": run_id,
            "plan_ids": list(PLAN_IDS),
            "required_plan_ids": list(PLAN_IDS[:4]),
            "decoder_runtime_identity": runtime,
            "source_bindings": source_bindings,
            "adapter_contract": {
                "request_layout": "grouped-by-display-video-id",
                "display_ids_are_internal": True,
                "later_adapter_must_flatten": True,
                "later_adapter_must_assign_safe_machine_ids": True,
                "later_adapter_must_resolve_absolute_mount_paths": True,
            },
            "rows": manifest_rows,
        }
        _audit_label_free({"rows": request_rows}, location="model-facing evidence")
        _write_json(stage / "manifest.json", manifest)
        for video_id, records in request_rows.items():
            _write_json(stage / "requests" / f"{video_id}.json", records)
        summary = {
            "schema_version": 2,
            "run_id": run_id,
            "row_count": len(manifest_rows),
            "plan_ids": list(PLAN_IDS),
            "required_plan_ids": list(PLAN_IDS[:4]),
            "request_count": sum(len(records) for records in request_rows.values()),
            "shifted_control_counts": {
                "eligible": sum(
                    row["shifted_control"]["eligible"] for row in manifest_rows
                ),
                "ineligible": sum(
                    not row["shifted_control"]["eligible"] for row in manifest_rows
                ),
            },
            "shifted_control_ineligibility_reasons": {
                reason: sum(
                    row["shifted_control"]["reason"] == reason for row in manifest_rows
                )
                for reason in sorted(
                    {
                        row["shifted_control"]["reason"]
                        for row in manifest_rows
                        if row["shifted_control"]["reason"] is not None
                    }
                )
            },
            "transcript_counts": {
                "available": sum(
                    normalized_whisper[video_id]["transcript_available"]
                    for video_id in normalized_whisper
                ),
                "unavailable": sum(
                    not normalized_whisper[video_id]["transcript_available"]
                    for video_id in normalized_whisper
                ),
            },
            "manifest_sha256": _sha256_file(stage / "manifest.json"),
            "aggregate_freeze_manifest_sha256": source_bindings["aggregate_freeze"][
                "manifest_sha256"
            ],
            "whisper_artifact_sha256": source_bindings["whisper"]["artifact_sha256"],
            "frozen_f0_artifact_sha256": source_bindings["frozen_f0"][
                "artifact_sha256"
            ],
            "canonical_panel_id_digest": expected_panel_id_digest,
            "preflight": preflight,
            "references_loaded": False,
        }
        _write_json(stage / "summary.json", summary)
        marker = {
            "schema_version": 1,
            "artifact_type": "vqa-event-map-evidence-completion",
            "status": "complete",
            "run_id": run_id,
            "row_count": len(manifest_rows),
            "request_count": summary["request_count"],
            "canonical_panel_id_digest": expected_panel_id_digest,
            "aggregate_freeze_manifest_sha256": source_bindings["aggregate_freeze"][
                "manifest_sha256"
            ],
            "aggregate_freeze_rows_sha256": source_bindings["aggregate_freeze"][
                "rows_sha256"
            ],
            "aggregate_freeze_row_schema_sha256": source_bindings[
                "aggregate_freeze"
            ].get("producer_row_schema_sha256"),
            "aggregate_freeze_av_shards_sha256": source_bindings[
                "aggregate_freeze"
            ].get("av_shards_sha256"),
            "whisper_artifact_sha256": source_bindings["whisper"]["artifact_sha256"],
            "frozen_f0_artifact_sha256": source_bindings["frozen_f0"][
                "artifact_sha256"
            ],
            "source_media_projection_sha256": _canonical_sha256(
                [
                    {
                        "video_id": row["video_id"],
                        "source_media_sha256": row["source_media_sha256"],
                    }
                    for row in manifest_rows
                ]
            ),
            "manifest_sha256": summary["manifest_sha256"],
            "summary_sha256": _sha256_file(stage / "summary.json"),
            "checksum_file": "SHA256SUMS",
            "adapter_contract": manifest["adapter_contract"],
        }
        _write_json(stage / "evidence-marker.json", marker)
        _write_checksums(stage)
        os.replace(stage, output_dir.resolve())
        return summary
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


build_evidence_bundle = build_evidence
materialize_evidence = build_evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--aggregate-freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-freeze-contract", type=Path, required=True)
    parser.add_argument("--frozen-baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--run-id", default=RUN_ID)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    panel_rows = _load_rows(args.panel, source="panel", names=("rows", "examples"))
    expected_freeze = _load_json(args.expected_freeze_contract)
    if not isinstance(expected_freeze, Mapping):
        raise ValueError("expected freeze contract must be a JSON object")
    _exact_keys(
        expected_freeze,
        {"identities", "producer_row_fields", "av_shards"},
        "expected freeze contract",
    )
    expected_identities = expected_freeze.get("identities")
    producer_row_fields = expected_freeze.get("producer_row_fields")
    expected_av_shards = expected_freeze.get("av_shards")
    if (
        not isinstance(expected_identities, Mapping)
        or not isinstance(producer_row_fields, list)
        or not isinstance(expected_av_shards, list)
    ):
        raise ValueError("expected freeze contract identities/schema are malformed")
    aggregate = validate_aggregate_freeze_manifest(
        args.aggregate_freeze_manifest,
        expected_identities=expected_identities,
        expected_producer_row_fields=producer_row_fields,
        expected_av_shards=expected_av_shards,
    )
    baseline = validate_frozen_baseline_bundle(args.frozen_baseline_root)
    summary = build_evidence(
        panel_rows=panel_rows,
        event_map_rows=aggregate["rows"],
        whisper_rows=baseline["whisper_rows"],
        frozen_f0_rows=baseline["f0_rows"],
        output_dir=args.output_dir,
        media_root=args.media_root,
        aggregate_freeze_binding=aggregate,
        whisper_binding=baseline["whisper_binding"],
        frozen_f0_binding=baseline["f0_binding"],
        run_id=args.run_id,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
