#!/usr/bin/env python3
"""Adjudicate the private, label-free VQA event-map canary.

The input artifact directory is expected to be downloaded privately from the
reserved dstack output prefix.  ``terminal-status.json`` is trusted only when
it binds the exact payload checksum and immutable identities; it is excluded
from ``SHA256SUMS`` because it is published after the payload.  The scheduler
record is intentionally a separate, already-sanitized JSON object with no
run-spec, user, secret, or raw scheduler fields.

Both the Python API and CLI emit a bounded decision.  Row answers, event
summaries, media paths, scheduler regions, and raw error messages never enter
that decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


RUN_ID = "vqa-event-map-canary-a2"
INPUT_RUN_ID = "vqa-event-map-inputs-20260809-a3"
ATTEMPT_UUID = "d6d3a33e-84c5-4067-bcea-fe82cdd10e2f"
OUTPUT_IDENTITY = (
    "hf://buckets/Npatta01/trec-video-2026-event-map-20260809-vqa-event-map-canary-a2"
)
# The original a1 attempt is retained only as rejected/superseded provenance.
SUPERSEDED_RUN_ID = "vqa-event-map-canary"
SUPERSEDED_ATTEMPT_UUID = "4baaf48c-b690-45f0-b6bf-065b9dc1e527"
SOURCE_BUNDLE_SHA256 = (
    "3dbfa1e31b4550a9ed1c77a2abdbad55c15d1594c7e2a153ad26dcc2101b138b"
)
CONTROLLER_SHA256 = "91815765952ede4e7986b09eec64ce2f831536259a1b3e1f12d0127b244be5a8"
PROTOCOL_SHA256 = "bc140b5ace4f1cfce9feb98434ca5276466602ed7e2487261788fc163c9fc4f2"
REQUEST_FILE_SHA256 = "eeb15558494948a77c881e557c14b5728865fd838a38ef4823faf2db5d85709c"
REQUEST_IDS_SHA256 = "c564f7979108d2d8fd0190bea16b3e95457ff89273026cf68128366cb699103a"
INFERENCE_MODULE_SHA256 = (
    "0deaf7eccafc9225ba533982c19ea894fdc8571c9081db4a52a3ddf11d5ca366"
)
MEDIA_MANIFEST_SHA256 = (
    "4967a4e718d2a443d2d57429d5c73bcac4ee00629bb6e7b561fc44a2062b74ed"
)
PANEL_IDS_SHA256 = "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
CANARY_IDS_SHA256 = "987a48ce72b6e2d292bc96debaafa26c403fc57515427976c226724d9c24002f"
MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
MODEL_REVISION = "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
RUNTIME_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@"
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
RUNTIME_IMAGE_DIGEST = RUNTIME_IMAGE.rsplit("@", 1)[1]
CONDITIONS = ("v-map", "av-map")
EXPECTED_REQUEST_FILENAME = "event-map-canary.jsonl"
EXPECTED_INPUT_FILES = frozenset(
    {
        "SHA256SUMS",
        "av-map-chunk-00.jsonl",
        "av-map-chunk-01.jsonl",
        "av-map-chunk-02.jsonl",
        "av-map-chunk-03.jsonl",
        EXPECTED_REQUEST_FILENAME,
        "event-map-controller.json",
        "event-map-media-manifest.json",
        "event-map-protocol.json",
        "event-map-summary.json",
        "v-map-chunk-00.jsonl",
        "v-map-chunk-01.jsonl",
        "v-map-chunk-02.jsonl",
        "v-map-chunk-03.jsonl",
    }
)
EXPECTED_PAYLOAD_FILES = frozenset(
    {
        "gpu-identity.json",
        "preflight.json",
        "predictions.jsonl",
        "runtime.json",
        "telemetry/nvidia-smi.csv",
    }
)
MAX_CANARY_SECONDS = 45 * 60
MAX_SCHEMA_RETRIES_PER_ROW = 1
FULL_CALL_COUNT = 950
FULL_ROW_COUNT = 475
FULL_SHARD_COUNTS = (119, 119, 119, 118)
MAX_AGGREGATE_GPU_HOURS = 28.0
MAX_LONGEST_SHARD_HOURS = 3.5
MAX_PROJECTED_COST_USD = 30.0
INFERENCE_PROJECTION_MARGIN = 1.10
FULL_TASK_COUNT = len(FULL_SHARD_COUNTS) * len(CONDITIONS)
PHASE_RECONCILIATION_ABSOLUTE_SECONDS = 120.0
PHASE_RECONCILIATION_RELATIVE = 0.10
INFERENCE_RECONCILIATION_SECONDS = 5.0
BILLING_TOLERANCE_USD = 0.05
BILLING_TOLERANCE_RELATIVE = 0.05
MAX_EVENTS = 6
MAX_SELECTED_INTERVALS = 2
MAX_SELECTED_SECONDS = 6.0
COVERAGE_TOLERANCE_SECONDS = 0.25
MIN_GPU_MEMORY_MIB = 45_000
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EVENT_ID = re.compile(r"^event-[A-Za-z0-9_-]{1,63}$")
FORBIDDEN_KEY_PARTS = (
    "reference",
    "groundtruth",
    "metric",
    "score",
    "annotation",
    "correctanswer",
    "expectedanswer",
    "label",
    "prediction",
    "transcript",
    "whisper",
    "reasoning",
    "rationale",
    "analysis",
    "thinking",
)
SAFE_IDENTITY_KEYS = frozenset({"labelpolicy", "enablethinking", "reasoningparser"})
EXPECTED_COMMAND = (
    "set -euo pipefail\nexec bash /dstack/run/scripts/run-vqa-event-map-dstack.sh\n"
)
EVENT_MAP_PROMPT = (
    "Watch the complete local video and answer the question. Return only the "
    "literal JSON object required by the schema. Do not think aloud, explain, "
    "write markdown, include a transcript, or include references, labels, "
    "metrics, or reasoning. Your answer must be at most 20 words. Build one "
    "to six chronological non-overlapping events that cover the measured clip. "
    "Select one or two short intervals inside event boundaries; their total "
    "duration must not exceed 6 seconds. Event summaries are private evidence."
)
EVENT_MAP_RETRY_PROMPT = (
    "Return exactly one JSON object and no other bytes. Use exactly these keys: "
    "answer, abstain, confidence, events, selected_intervals. Do not use a "
    "markdown fence, thinking tags, analysis, reasoning, transcript, label, "
    "reference, score, or metric. The answer is at most 20 words. Events must "
    "cover the clip, be chronological and non-overlapping, and selections must "
    "be inside their event with total duration at most 6 seconds."
)
EVENT_MAP_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "abstain", "confidence", "events", "selected_intervals"],
    "properties": {
        "answer": {"type": "string", "minLength": 1, "maxLength": 240},
        "abstain": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EVENTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "event_id",
                    "start",
                    "end",
                    "visual_summary",
                    "audio_summary",
                ],
                "properties": {
                    "event_id": {
                        "type": "string",
                        "pattern": r"^event-[A-Za-z0-9_-]+$",
                    },
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "visual_summary": {"type": "string", "minLength": 1},
                    "audio_summary": {"type": ["string", "null"]},
                },
            },
        },
        "selected_intervals": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SELECTED_INTERVALS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event_id", "start", "end"],
                "properties": {
                    "event_id": {
                        "type": "string",
                        "pattern": r"^event-[A-Za-z0-9_-]+$",
                    },
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
            },
        },
    },
}


class EvidenceError(ValueError):
    """A private evidence contract was absent or inconsistent."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("invalid JSON evidence") from error


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{field} is not an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceError(f"{field} is not an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{field} schema mismatch")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{field} is not finite")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} is not nonempty text")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if HEX_SHA256.fullmatch(text) is None:
        raise EvidenceError(f"{field} is not SHA-256")
    return text


def _canonical_uuid(value: Any, field: str) -> str:
    text = _text(value, field)
    try:
        parsed = uuid.UUID(text)
    except ValueError as error:
        raise EvidenceError(f"{field} is not a UUID") from error
    if parsed.int == 0 or str(parsed) != text:
        raise EvidenceError(f"{field} is not a canonical nonzero UUID")
    return text


def _timestamp(value: Any, field: str) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{field} is not a timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{field} lacks a timezone")
    return parsed


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _reject_leakage(value: Any, *, allow_answer: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in SAFE_IDENTITY_KEYS:
                _reject_leakage(child, allow_answer=allow_answer)
                continue
            if allow_answer and normalized == "answer":
                continue
            if any(part in normalized for part in FORBIDDEN_KEY_PARTS):
                raise EvidenceError("label/reference/metric leakage")
            _reject_leakage(child, allow_answer=allow_answer)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_leakage(child, allow_answer=allow_answer)


def _safe_relative(path_text: str) -> PurePosixPath:
    if not path_text.startswith("./"):
        raise EvidenceError("unsafe checksum path")
    relative = PurePosixPath(path_text[2:])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise EvidenceError("unsafe checksum path")
    return relative


def _read_checksum_entries(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise EvidenceError("checksum manifest missing")
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (\./[^\r\n]+)", line)
        if match is None:
            raise EvidenceError("checksum manifest schema mismatch")
        relative = _safe_relative(match.group(2)).as_posix()
        if relative in entries or relative in {"SHA256SUMS", "terminal-status.json"}:
            raise EvidenceError("checksum manifest has duplicate or trusted file")
        entries[relative] = match.group(1)
    return entries


def _verify_file_set(
    root: Path,
    expected: frozenset[str],
    *,
    kind: str,
    allow_terminal: bool = False,
) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise EvidenceError(f"{kind} root is missing")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise EvidenceError(f"{kind} contains a symlink")
    excluded = {"SHA256SUMS"}
    if allow_terminal:
        excluded.add("terminal-status.json")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if actual != set(expected):
        raise EvidenceError(f"{kind} file set mismatch")
    entries = _read_checksum_entries(root)
    if set(entries) != actual:
        raise EvidenceError(f"{kind} checksum coverage mismatch")
    for relative, expected_hash in entries.items():
        if _sha256_file(root / relative) != expected_hash:
            raise EvidenceError(f"{kind} byte checksum mismatch")
    return entries


def _expected_config() -> dict[str, Any]:
    return {
        "type": "task",
        "name": RUN_ID,
        "image": RUNTIME_IMAGE,
        "entrypoint": "/bin/bash -c",
        "shell": "bash",
        "working_dir": "/dstack/run",
        "env": [
            "HF_TOKEN=${{ secrets.hf_token }}",
            "HF_HUB_DISABLE_TELEMETRY=1",
            "PYTHON_BIN=/usr/bin/python3",
            "PYTHONPATH=/dstack/run/src",
            "EXPERIMENT_MODE=canary",
            f"EXPERIMENT_RUN_ID={RUN_ID}",
            "EVENT_MAP_CONDITIONS=v-map,av-map",
            "EVENT_MAP_SLICE=",
            "HF_INPUT_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/vqa-event-map-inputs-20260809-a3/inputs",
            f"HF_REMOTE_URI={OUTPUT_IDENTITY}",
            f"RUNTIME_IMAGE={RUNTIME_IMAGE}",
            f"RUNTIME_IMAGE_DIGEST={RUNTIME_IMAGE_DIGEST}",
            f"MODEL_ID={MODEL_ID}",
            f"MODEL_REVISION={MODEL_REVISION}",
            "VLLM_RESPONSE_FORMAT_SUPPORTED=1",
            f"ATTEMPT_UUID={ATTEMPT_UUID}",
            f"EXPECTED_SOURCE_BUNDLE_SHA256={SOURCE_BUNDLE_SHA256}",
            f"EXPECTED_CONTROLLER_SHA256={CONTROLLER_SHA256}",
            f"EXPECTED_PROTOCOL_SHA256={PROTOCOL_SHA256}",
            f"EXPECTED_INFERENCE_MODULE_SHA256={INFERENCE_MODULE_SHA256}",
            "EXPECTED_METADATA_BUNDLE_SHA256=a7a2037f210a3fc1038fedd48678b327a44b970f19b5005e270966ab8bd83f2a",
            f"EXPECTED_REQUEST_FILE_SHA256={REQUEST_FILE_SHA256}",
        ],
        "files": [
            "../scripts/run-vqa-event-map-dstack.sh:/dstack/run/scripts/run-vqa-event-map-dstack.sh",
            "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
        ],
        "commands": [EXPECTED_COMMAND],
        "fleets": ["gpu-marketplace"],
        "resources": {
            "gpu": "L40S:1:48GB..",
            "cpu": "8..",
            "memory": "64GB..",
            "disk": "100GB",
        },
        "backends": ["runpod", "vastai"],
        "spot_policy": "on-demand",
        "max_price": 1.0,
        "max_duration": "45m",
        "retry": {"on_events": ["no-capacity"], "duration": "20m"},
    }


def _validate_config(path: Path) -> str:
    try:
        config_bytes = path.read_bytes()
        config = yaml.safe_load(config_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise EvidenceError("canary config is invalid") from error
    config = _mapping(config, "canary config")
    expected = _expected_config()
    _exact_keys(config, set(expected), "canary config")
    if config != expected:
        raise EvidenceError("canary config pin mismatch")
    return hashlib.sha256(config_bytes).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError("JSONL payload is unreadable") from error
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            raise EvidenceError("JSONL payload contains a blank row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError("JSONL payload contains invalid JSON") from error
        rows.append(dict(_mapping(value, "JSONL row")))
    return rows


def _validate_inputs(input_dir: Path) -> list[dict[str, Any]]:
    input_entries = _verify_file_set(
        input_dir, EXPECTED_INPUT_FILES - {"SHA256SUMS"}, kind="input"
    )
    if (
        _sha256_file(input_dir / "SHA256SUMS")
        != "a7a2037f210a3fc1038fedd48678b327a44b970f19b5005e270966ab8bd83f2a"
    ):
        raise EvidenceError("input checksum manifest pin mismatch")
    if input_entries.get("event-map-controller.json") != CONTROLLER_SHA256:
        raise EvidenceError("controller checksum pin mismatch")
    if (
        input_entries.get("event-map-protocol.json")
        != "2b935dd66fe0bf48648276773c0bff0aa6bc6d297cec2dd4ebc7f2f14938869a"
    ):
        raise EvidenceError("protocol checksum pin mismatch")
    if input_entries.get("event-map-media-manifest.json") != MEDIA_MANIFEST_SHA256:
        raise EvidenceError("media manifest checksum pin mismatch")
    if input_entries.get(EXPECTED_REQUEST_FILENAME) != REQUEST_FILE_SHA256:
        raise EvidenceError("request checksum pin mismatch")

    controller = _mapping(_json(input_dir / "event-map-controller.json"), "controller")
    protocol = _mapping(_json(input_dir / "event-map-protocol.json"), "protocol")
    _reject_leakage(controller)
    _reject_leakage(protocol)
    _exact_keys(
        controller,
        {"schema_version", "run_id", "label_policy", "panel", "rows"},
        "controller",
    )
    if (
        controller.get("schema_version") != 1
        or controller.get("run_id") != INPUT_RUN_ID
    ):
        raise EvidenceError("controller identity mismatch")
    if controller.get("label_policy") != "label-free-controller":
        raise EvidenceError("controller is not label-free")
    controller_rows = _sequence(controller.get("rows"), "controller rows")
    panel = _mapping(controller.get("panel"), "controller panel")
    if len(controller_rows) != FULL_ROW_COUNT:
        raise EvidenceError("controller row count mismatch")
    if (
        panel.get("row_count") != FULL_ROW_COUNT
        or panel.get("ids_sha256") != PANEL_IDS_SHA256
    ):
        raise EvidenceError("controller panel identity mismatch")
    if panel.get("canary_ids") != [
        "G6zu9D0j8hk",
        "GuSGaDJsgsE",
        "DC4FE3hI67U",
        "B24dV_uori4",
    ] or panel.get("shard_counts") != list(FULL_SHARD_COUNTS):
        raise EvidenceError("controller canary or shard identity mismatch")
    for row in controller_rows:
        row_mapping = _mapping(row, "controller row")
        _exact_keys(
            row_mapping,
            {
                "video_id",
                "question",
                "question_stratum",
                "media_path",
                "media_sha256",
                "duration_seconds",
            },
            "controller row",
        )
        _text(row_mapping.get("video_id"), "controller video_id")
        _text(row_mapping.get("question"), "controller question")
        _sha256(row_mapping.get("media_sha256"), "controller media_sha256")
        if (
            _finite_number(row_mapping.get("duration_seconds"), "controller duration")
            <= 0
        ):
            raise EvidenceError("controller duration mismatch")
    ids = [str(_mapping(row, "controller row")["video_id"]) for row in controller_rows]
    if hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest() != PANEL_IDS_SHA256:
        raise EvidenceError("controller ID digest mismatch")

    protocol_without_hash = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    if protocol.get("protocol_sha256") != _canonical_sha256(protocol_without_hash):
        raise EvidenceError("protocol self-checksum mismatch")
    if protocol.get("protocol_sha256") != PROTOCOL_SHA256:
        raise EvidenceError("protocol identity mismatch")
    if protocol.get("schema_version") != 1 or protocol.get("run_id") != INPUT_RUN_ID:
        raise EvidenceError("protocol run identity mismatch")
    if (
        protocol.get("controller_sha256") != CONTROLLER_SHA256
        or protocol.get("media_manifest_sha256") != MEDIA_MANIFEST_SHA256
    ):
        raise EvidenceError("protocol source identity mismatch")
    if protocol.get("source_bundle_sha256") != SOURCE_BUNDLE_SHA256:
        raise EvidenceError("source bundle identity mismatch")
    if protocol.get("inference_module_sha256") != INFERENCE_MODULE_SHA256:
        raise EvidenceError("inference module identity mismatch")
    if protocol.get("conditions") != list(CONDITIONS):
        raise EvidenceError("condition order mismatch")
    decoder = _mapping(protocol.get("decoder"), "decoder protocol")
    if (
        decoder.get("max_retries") != MAX_SCHEMA_RETRIES_PER_ROW
        or decoder.get("enable_thinking") is not False
        or decoder.get("max_output_tokens") != 768
    ):
        raise EvidenceError("decoder retry or output pin mismatch")
    model = _mapping(protocol.get("model"), "model protocol")
    if (
        model.get("model_id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("runtime_image") != RUNTIME_IMAGE
        or model.get("runtime_image_digest") != RUNTIME_IMAGE_DIGEST
    ):
        raise EvidenceError("model/runtime protocol identity mismatch")
    panel_protocol = _mapping(protocol.get("panel"), "panel protocol")
    if (
        panel_protocol.get("row_count") != FULL_ROW_COUNT
        or panel_protocol.get("ids_sha256") != PANEL_IDS_SHA256
        or panel_protocol.get("shard_counts") != list(FULL_SHARD_COUNTS)
    ):
        raise EvidenceError("panel protocol identity mismatch")
    canary_protocol = _mapping(protocol.get("canary"), "canary protocol")
    if (
        canary_protocol.get("row_count") != 4
        or canary_protocol.get("request_row_count") != 8
        or canary_protocol.get("paired") is not True
        or canary_protocol.get("ids_sha256") != CANARY_IDS_SHA256
    ):
        raise EvidenceError("canary protocol identity mismatch")
    request_protocol = _mapping(protocol.get("request_files"), "request file protocol")
    request_identity = _mapping(
        request_protocol.get(EXPECTED_REQUEST_FILENAME), "canary request identity"
    )
    if (
        request_identity.get("file_sha256") != REQUEST_FILE_SHA256
        or request_identity.get("row_count") != 8
        or request_identity.get("ids_sha256") != REQUEST_IDS_SHA256
    ):
        raise EvidenceError("canary request protocol identity mismatch")

    requests = _read_jsonl(input_dir / EXPECTED_REQUEST_FILENAME)
    if len(requests) != 8:
        raise EvidenceError("canary request count mismatch")
    expected_canary_ids = panel.get("canary_ids")
    expected_pairs = [
        (video_id, condition)
        for video_id in expected_canary_ids
        for condition in CONDITIONS
    ]
    if [
        (row.get("video_id"), row.get("condition_id")) for row in requests
    ] != expected_pairs:
        raise EvidenceError("canary request order mismatch")
    controller_by_id = {
        _mapping(row, "controller row")["video_id"]: _mapping(row, "controller row")
        for row in controller_rows
    }
    for row, (video_id, condition) in zip(requests, expected_pairs, strict=True):
        _exact_keys(
            row,
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
                "duration_seconds",
                "request_sha256",
            },
            "canary request row",
        )
        if (
            row.get("schema_version") != 1
            or row.get("run_id") != INPUT_RUN_ID
            or row.get("condition_id") != condition
            or row.get("slice_id") != "canary"
        ):
            raise EvidenceError("canary request identity mismatch")
        source = controller_by_id.get(video_id)
        if source is None or any(
            row.get(key) != source.get(key)
            for key in (
                "video_id",
                "question",
                "question_stratum",
                "media_path",
                "media_sha256",
                "duration_seconds",
            )
        ):
            raise EvidenceError("canary request/controller mismatch")
        row_without_hash = {
            key: value for key, value in row.items() if key != "request_sha256"
        }
        if row.get("request_sha256") != _canonical_sha256(row_without_hash):
            raise EvidenceError("canary request row hash mismatch")
    return requests


def _validate_scheduler(
    value: Mapping[str, Any],
    *,
    expected_dstack_run_id: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "run_id",
            "run_name",
            "config_sha256",
            "status",
            "cost",
            "started_at",
            "finished_at",
            "backend",
            "region",
            "gpu",
            "price",
        },
        "scheduler record",
    )
    dstack_run_id = _canonical_uuid(value.get("run_id"), "scheduler run_id")
    if dstack_run_id != _canonical_uuid(
        expected_dstack_run_id, "expected dstack run_id"
    ):
        raise EvidenceError("scheduler run identity mismatch")
    config_sha256 = _sha256(value.get("config_sha256"), "scheduler config hash")
    if (
        value.get("run_name") != RUN_ID
        or config_sha256 != _sha256(expected_config_sha256, "expected config hash")
        or value.get("status") != "done"
    ):
        raise EvidenceError("scheduler run identity mismatch")
    started = _timestamp(value.get("started_at"), "scheduler started_at")
    finished = _timestamp(value.get("finished_at"), "scheduler finished_at")
    elapsed = (finished - started).total_seconds()
    if elapsed <= 0:
        raise EvidenceError("scheduler timestamps are reversed")
    cost = _finite_number(value.get("cost"), "scheduler cost")
    price = _finite_number(value.get("price"), "scheduler price")
    if cost < 0 or price <= 0:
        raise EvidenceError("scheduler cost or price is invalid")
    backend = _text(value.get("backend"), "scheduler backend")
    if backend not in {"runpod", "vastai"}:
        raise EvidenceError("scheduler backend is not allowed")
    _text(value.get("region"), "scheduler region")
    expected_cost = elapsed / 3600 * price
    billing_tolerance = max(
        BILLING_TOLERANCE_USD,
        expected_cost * BILLING_TOLERANCE_RELATIVE,
    )
    if abs(cost - expected_cost) > billing_tolerance:
        raise EvidenceError("scheduler cost does not reconcile")
    gpu = _mapping(value.get("gpu"), "scheduler GPU")
    _exact_keys(gpu, {"name", "memory_mib"}, "scheduler GPU")
    if (
        gpu.get("name") not in {"L40S", "NVIDIA L40S"}
        or not isinstance(gpu.get("memory_mib"), int)
        or isinstance(gpu.get("memory_mib"), bool)
        or gpu["memory_mib"] < MIN_GPU_MEMORY_MIB
    ):
        raise EvidenceError("scheduler GPU identity mismatch")
    return {
        "dstack_run_id": dstack_run_id,
        "run_name": RUN_ID,
        "config_sha256": config_sha256,
        "elapsed_seconds": elapsed,
        "cost": cost,
        "price": price,
        "backend": backend,
        "gpu_memory_mib": gpu["memory_mib"],
    }


def sanitize_dstack_scheduler_json(
    value: Mapping[str, Any] | Path,
    *,
    expected_dstack_run_id: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Validate and return only safe scheduler facts.

    The accepted input has exactly ``run_id``, non-sensitive ``run_name`` and
    ``config_sha256``, ``status``, ``cost``, ``started_at``, ``finished_at``,
    ``backend``, ``region``, ``gpu`` (``name`` and ``memory_mib``), and
    ``price``. It is not a raw ``dstack ps`` response. Cost may differ from
    elapsed-hours times price by at most five percent or $0.05, whichever is
    larger.
    """

    if isinstance(value, Path):
        value = _mapping(_json(value), "scheduler record")
    safe = _validate_scheduler(
        value,
        expected_dstack_run_id=expected_dstack_run_id,
        expected_config_sha256=expected_config_sha256,
    )
    return {
        "schema_version": 1,
        "dstack_run_id": safe["dstack_run_id"],
        "run_name": safe["run_name"],
        "config_sha256": safe["config_sha256"],
        "status": "done",
        "elapsed_seconds": safe["elapsed_seconds"],
        "cost": safe["cost"],
        "price": safe["price"],
        "backend": safe["backend"],
        "gpu_memory_mib": safe["gpu_memory_mib"],
    }


def _expected_runner_sha256() -> str:
    runner = Path(__file__).resolve().parent / "run-vqa-event-map-dstack.sh"
    return _sha256_file(runner)


def _parse_producer_timings(value: Any) -> dict[str, Any]:
    timings = _mapping(value, "producer timings")
    _exact_keys(
        timings,
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
        "producer timings",
    )
    normalization = _mapping(timings.get("normalization"), "normalization timing")
    _exact_keys(normalization, {"media_count", "seconds"}, "normalization timing")
    media_count = normalization.get("media_count")
    if (
        timings.get("schema_version") != 1
        or timings.get("clock") != "monotonic"
        or not isinstance(media_count, int)
        or isinstance(media_count, bool)
        or media_count != 4
    ):
        raise EvidenceError("producer timing identity mismatch")
    parsed = {
        "schema_version": 1,
        "clock": "monotonic",
        "bootstrap_download_seconds": _finite_number(
            timings.get("bootstrap_download_seconds"), "bootstrap/download timing"
        ),
        "normalization": {
            "media_count": media_count,
            "seconds": _finite_number(
                normalization.get("seconds"), "normalization timing"
            ),
        },
        "model_startup_seconds": _finite_number(
            timings.get("model_startup_seconds"), "model startup timing"
        ),
        "inference_seconds": _finite_number(
            timings.get("inference_seconds"), "inference timing"
        ),
        "finalization_seconds": _finite_number(
            timings.get("finalization_seconds"), "finalization timing"
        ),
        "phase_total_seconds": _finite_number(
            timings.get("phase_total_seconds"), "phase total timing"
        ),
    }
    phase_values = (
        parsed["bootstrap_download_seconds"],
        parsed["normalization"]["seconds"],
        parsed["model_startup_seconds"],
        parsed["inference_seconds"],
        parsed["finalization_seconds"],
    )
    if any(value <= 0 for value in phase_values):
        raise EvidenceError("producer timing must be positive")
    if not math.isclose(
        parsed["phase_total_seconds"],
        sum(phase_values),
        rel_tol=1e-9,
        abs_tol=0.001,
    ):
        raise EvidenceError("producer phase total mismatch")
    return parsed


def _validate_terminal(
    artifact_dir: Path, payload_entries: Mapping[str, str]
) -> dict[str, Any]:
    terminal_path = artifact_dir / "terminal-status.json"
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise EvidenceError("trusted terminal marker is missing")
    terminal = _mapping(_json(terminal_path), "terminal marker")
    _exact_keys(
        terminal,
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
        },
        "terminal marker",
    )
    if terminal.get("schema_version") != 1 or terminal.get("status") != "complete":
        raise EvidenceError("terminal marker is not complete")
    expected = {
        "run_id": RUN_ID,
        "attempt_uuid": ATTEMPT_UUID,
        "output_identity": OUTPUT_IDENTITY,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "controller_sha256": CONTROLLER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "request_file_sha256": REQUEST_FILE_SHA256,
        "inference_module_sha256": INFERENCE_MODULE_SHA256,
        "runner_sha256": _expected_runner_sha256(),
    }
    for key, expected_value in expected.items():
        if terminal.get(key) != expected_value:
            raise EvidenceError("terminal identity mismatch")
    if (
        terminal.get("row_count") != 8
        or _canonical_uuid(terminal.get("attempt_uuid"), "terminal attempt_uuid")
        != ATTEMPT_UUID
    ):
        raise EvidenceError("terminal row or attempt mismatch")
    if terminal.get("payload_checksum") != _sha256_file(artifact_dir / "SHA256SUMS"):
        raise EvidenceError("terminal payload checksum mismatch")
    if set(payload_entries) != set(EXPECTED_PAYLOAD_FILES):
        raise EvidenceError("terminal payload coverage mismatch")
    return _parse_producer_timings(terminal.get("producer_timings"))


def _validate_runtime(artifact_dir: Path) -> dict[str, Any]:
    runtime = _mapping(_json(artifact_dir / "runtime.json"), "runtime")
    _exact_keys(
        runtime,
        {
            "schema_version",
            "runtime_image",
            "vllm_version",
            "torch_version",
            "torch_cuda",
            "torch_hip",
            "gpu_name",
            "gpu_memory_mib",
            "enable_thinking",
            "max_output_tokens",
            "gpu_telemetry_path",
        },
        "runtime",
    )
    if (
        runtime.get("schema_version") != 1
        or runtime.get("runtime_image") != RUNTIME_IMAGE
        or runtime.get("vllm_version") != "0.20.0"
        or runtime.get("enable_thinking") is not False
        or runtime.get("max_output_tokens") != 768
        or runtime.get("gpu_telemetry_path") != "telemetry/nvidia-smi.csv"
    ):
        raise EvidenceError("runtime pin mismatch")
    _text(runtime.get("torch_version"), "torch version")
    _text(runtime.get("torch_cuda"), "CUDA version")
    if runtime.get("torch_hip") is not None:
        raise EvidenceError("runtime must be CUDA, not HIP")
    if (
        runtime.get("gpu_name") not in {"L40S", "NVIDIA L40S"}
        or not isinstance(runtime.get("gpu_memory_mib"), int)
        or isinstance(runtime.get("gpu_memory_mib"), bool)
        or runtime["gpu_memory_mib"] < MIN_GPU_MEMORY_MIB
    ):
        raise EvidenceError("runtime GPU identity mismatch")
    gpu_identity = _mapping(_json(artifact_dir / "gpu-identity.json"), "GPU identity")
    _exact_keys(gpu_identity, {"gpu_name", "gpu_memory_mib"}, "GPU identity")
    if gpu_identity.get("gpu_name") != "L40S" or gpu_identity.get(
        "gpu_memory_mib"
    ) != runtime.get("gpu_memory_mib"):
        raise EvidenceError("GPU identity mismatch")
    telemetry = artifact_dir / "telemetry/nvidia-smi.csv"
    try:
        with telemetry.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            samples = list(reader)
    except (OSError, UnicodeDecodeError, StopIteration, csv.Error) as error:
        raise EvidenceError("GPU telemetry is unreadable") from error
    if (
        header
        != [
            "timestamp",
            "gpu_name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu",
        ]
        or not samples
    ):
        raise EvidenceError("GPU telemetry schema mismatch")
    previous: datetime | None = None
    for sample in samples:
        if len(sample) != 5:
            raise EvidenceError("GPU telemetry row mismatch")
        timestamp = _timestamp(sample[0], "GPU telemetry timestamp")
        if previous is not None and timestamp < previous:
            raise EvidenceError("GPU telemetry order mismatch")
        previous = timestamp
        if sample[1] not in {"L40S", "NVIDIA L40S"}:
            raise EvidenceError("GPU telemetry identity mismatch")
        try:
            used = int(sample[2])
            total = int(sample[3])
            utilization = float(sample[4])
        except ValueError as error:
            raise EvidenceError("GPU telemetry number mismatch") from error
        if (
            total != runtime["gpu_memory_mib"]
            or total < MIN_GPU_MEMORY_MIB
            or not 0 <= used <= total
            or not 0 <= utilization <= 100
        ):
            raise EvidenceError("GPU telemetry bounds mismatch")
    return dict(runtime)


def _validate_event_map(response: Mapping[str, Any], duration: float) -> None:
    _exact_keys(
        response,
        {"answer", "abstain", "confidence", "events", "selected_intervals"},
        "event-map response",
    )
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip() or len(answer.split()) > 20:
        raise EvidenceError("answer is empty or unbounded")
    if type(response.get("abstain")) is not bool:
        raise EvidenceError("abstain schema mismatch")
    confidence = _finite_number(response.get("confidence"), "confidence")
    if not 0 <= confidence <= 1:
        raise EvidenceError("confidence bounds mismatch")
    events = _sequence(response.get("events"), "events")
    if not 1 <= len(events) <= MAX_EVENTS:
        raise EvidenceError("event count mismatch")
    seen: set[str] = set()
    event_by_id: dict[str, Mapping[str, Any]] = {}
    previous_end = -math.inf
    for index, raw_event in enumerate(events):
        event = _mapping(raw_event, f"event {index}")
        _exact_keys(
            event,
            {"event_id", "start", "end", "visual_summary", "audio_summary"},
            "event",
        )
        event_id = _text(event.get("event_id"), "event_id")
        if EVENT_ID.fullmatch(event_id) is None or event_id in seen:
            raise EvidenceError("event ID mismatch")
        seen.add(event_id)
        start = _finite_number(event.get("start"), "event start")
        end = _finite_number(event.get("end"), "event end")
        if (
            start >= end
            or start < -COVERAGE_TOLERANCE_SECONDS
            or end > duration + COVERAGE_TOLERANCE_SECONDS
            or start < previous_end - 1e-9
            or (index and start > previous_end + COVERAGE_TOLERANCE_SECONDS)
        ):
            raise EvidenceError("event bounds or coverage mismatch")
        _text(event.get("visual_summary"), "visual summary")
        audio = event.get("audio_summary")
        if audio is not None:
            _text(audio, "audio summary")
        event_by_id[event_id] = event
        previous_end = end
    first = _mapping(events[0], "first event")
    last = _mapping(events[-1], "last event")
    if (
        _finite_number(first["start"], "first event start") > COVERAGE_TOLERANCE_SECONDS
        or _finite_number(last["end"], "last event end")
        < duration - COVERAGE_TOLERANCE_SECONDS
    ):
        raise EvidenceError("event coverage mismatch")
    intervals = _sequence(response.get("selected_intervals"), "selected intervals")
    if not 1 <= len(intervals) <= MAX_SELECTED_INTERVALS:
        raise EvidenceError("selected interval count mismatch")
    previous_end = -math.inf
    total = 0.0
    for index, raw_interval in enumerate(intervals):
        interval = _mapping(raw_interval, f"selected interval {index}")
        _exact_keys(interval, {"event_id", "start", "end"}, "selected interval")
        event_id = _text(interval.get("event_id"), "selected interval event_id")
        event = event_by_id.get(event_id)
        if event is None:
            raise EvidenceError("selected interval event mismatch")
        start = _finite_number(interval.get("start"), "selected interval start")
        end = _finite_number(interval.get("end"), "selected interval end")
        event_start = _finite_number(event["start"], "event start")
        event_end = _finite_number(event["end"], "event end")
        if (
            not 0 <= start < end <= duration
            or start < event_start
            or end > event_end
            or start < previous_end
        ):
            raise EvidenceError("selected interval bounds mismatch")
        previous_end = end
        total += end - start
    if total > MAX_SELECTED_SECONDS + 1e-9:
        raise EvidenceError("selected interval duration mismatch")


def _expected_chat_request(
    request: Mapping[str, Any],
    *,
    normalized_duration_seconds: float,
    retry: bool,
) -> dict[str, Any]:
    video_id = _text(request.get("video_id"), "request video_id")
    if re.fullmatch(r"[A-Za-z0-9_-]+", video_id) is None:
        raise EvidenceError("request video_id is unsafe")
    question = _text(request.get("question"), "request question")
    condition = request.get("condition_id")
    if condition not in CONDITIONS:
        raise EvidenceError("request condition mismatch")
    duration = _finite_number(normalized_duration_seconds, "normalized duration")
    if duration <= 0:
        raise EvidenceError("normalized duration is invalid")
    prompt = EVENT_MAP_RETRY_PROMPT if retry else EVENT_MAP_PROMPT
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": (f"file:///dstack/input/normalized/{video_id}.mp4")
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            prompt
                            + f"\nMeasured clip duration: {duration:.3f} seconds. "
                            + "Cover continuously from 0.000 through "
                            + f"{duration:.3f} seconds."
                            + "\nQuestion: "
                            + question
                        ),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": 768,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": condition == "av-map"},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "nemotron_event_map",
                "strict": True,
                "schema": EVENT_MAP_RESPONSE_SCHEMA,
            },
        },
    }


def _expected_request_sha256(
    request: Mapping[str, Any],
    *,
    normalized_duration_seconds: float,
    retry: bool,
) -> str:
    return _canonical_sha256(
        _expected_chat_request(
            request,
            normalized_duration_seconds=normalized_duration_seconds,
            retry=retry,
        )
    )


def _validate_predictions(
    artifact_dir: Path,
    requests: Sequence[dict[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _read_jsonl(artifact_dir / "predictions.jsonl")
    if len(rows) != len(requests) or len(rows) != 8:
        raise EvidenceError("prediction row count mismatch")
    expected_keys = {
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
    retry_count = 0
    elapsed_by_condition = {condition: [] for condition in CONDITIONS}
    for index, (row, request) in enumerate(zip(rows, requests, strict=True)):
        if (row.get("video_id"), row.get("condition_id")) != (
            request.get("video_id"),
            request.get("condition_id"),
        ):
            raise EvidenceError("prediction row identity mismatch")
        if (
            row.get("status") != "passed"
            and row.get("retry_used") is True
            and row.get("retry_valid") is False
            and row.get("error_type") == "request_failed"
        ):
            raise EvidenceError("retry transport failure")
        _exact_keys(row, expected_keys, f"prediction row {index}")
        _reject_leakage(row, allow_answer=True)
        if (
            row.get("schema_version") != 1
            or row.get("run_id") != INPUT_RUN_ID
            or row.get("slice_id") != "canary"
        ):
            raise EvidenceError("prediction identity mismatch")
        for key, expected in (
            ("question", request["question"]),
            ("question_stratum", request["question_stratum"]),
            ("media_path", request["media_path"]),
            ("controller_request_sha256", request["request_sha256"]),
            ("source_media_sha256", request["media_sha256"]),
            ("condition_id", request["condition_id"]),
            ("video_id", request["video_id"]),
        ):
            if row.get(key) != expected:
                raise EvidenceError("prediction request binding mismatch")
        if (
            row.get("model_id") != MODEL_ID
            or row.get("model_revision") != MODEL_REVISION
            or row.get("runtime_image") != RUNTIME_IMAGE
            or row.get("runtime_identity") != runtime
        ):
            raise EvidenceError("prediction model/runtime identity mismatch")
        memory = _mapping(row.get("memory_telemetry"), "memory telemetry binding")
        if memory != {
            "path": "telemetry/nvidia-smi.csv",
            "sampling_interval_seconds": 5,
        }:
            raise EvidenceError("prediction telemetry binding mismatch")
        source_duration = _finite_number(
            row.get("source_duration_seconds"), "source duration"
        )
        normalized_duration = _finite_number(
            row.get("normalized_duration_seconds"), "normalized duration"
        )
        request_duration = _finite_number(
            request["duration_seconds"], "request duration"
        )
        if (
            source_duration <= 0
            or normalized_duration <= 0
            or abs(source_duration - normalized_duration) > COVERAGE_TOLERANCE_SECONDS
            or abs(source_duration - request_duration) > COVERAGE_TOLERANCE_SECONDS
        ):
            raise EvidenceError("duration binding mismatch")
        primary_hash = _sha256(
            row.get("primary_request_sha256"), "primary request hash"
        )
        request_hash = _sha256(row.get("request_sha256"), "request hash")
        expected_primary_hash = _expected_request_sha256(
            request,
            normalized_duration_seconds=normalized_duration,
            retry=False,
        )
        if primary_hash != expected_primary_hash:
            raise EvidenceError("request hash binding mismatch")
        retry_hash = row.get("retry_request_sha256")
        if retry_hash is not None:
            retry_hash = _sha256(retry_hash, "retry request hash")
        response_hash = row.get("response_sha256")
        if row.get("status") != "passed" or row.get("error_type") is not None:
            raise EvidenceError("prediction status is incomplete")
        _sha256(response_hash, "response hash")
        if (
            type(row.get("retry_used")) is not bool
            or type(row.get("primary_valid")) is not bool
        ):
            raise EvidenceError("retry flags schema mismatch")
        if row["retry_used"]:
            retry_count += 1
            expected_retry_hash = _expected_request_sha256(
                request,
                normalized_duration_seconds=normalized_duration,
                retry=True,
            )
            if (
                row.get("primary_valid") is not False
                or row.get("retry_valid") is not True
                or retry_hash is None
                or retry_hash != expected_retry_hash
                or request_hash != retry_hash
            ):
                raise EvidenceError("retry hash semantics mismatch")
        elif (
            row.get("primary_valid") is not True
            or row.get("retry_valid") is not None
            or retry_hash is not None
            or request_hash != primary_hash
        ):
            raise EvidenceError("primary hash semantics mismatch")
        elapsed = _finite_number(row.get("elapsed_seconds"), "row elapsed_seconds")
        if elapsed < 0:
            raise EvidenceError("row elapsed_seconds is negative")
        elapsed_by_condition[str(row["condition_id"])].append(elapsed)
        _sha256(row.get("normalized_media_sha256"), "normalized media hash")
        _validate_event_map(
            {
                key: row[key]
                for key in (
                    "answer",
                    "abstain",
                    "confidence",
                    "events",
                    "selected_intervals",
                )
            },
            normalized_duration,
        )
    return {
        "rows": len(rows),
        "retry_count": retry_count,
        "elapsed_by_condition": elapsed_by_condition,
        "elapsed_seconds": sum(sum(values) for values in elapsed_by_condition.values()),
    }


def _validate_phase_reconciliation(
    scheduler: Mapping[str, Any],
    row_summary: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> None:
    scheduler_seconds = float(scheduler["elapsed_seconds"])
    phase_total_seconds = float(timings["phase_total_seconds"])
    tolerance = max(
        PHASE_RECONCILIATION_ABSOLUTE_SECONDS,
        scheduler_seconds * PHASE_RECONCILIATION_RELATIVE,
    )
    if abs(scheduler_seconds - phase_total_seconds) > tolerance:
        raise EvidenceError("producer phases do not reconcile with scheduler")
    if float(timings["inference_seconds"]) + INFERENCE_RECONCILIATION_SECONDS < float(
        row_summary["elapsed_seconds"]
    ):
        raise EvidenceError("producer inference timing is shorter than row timings")


def _projection(
    scheduler: Mapping[str, Any],
    row_summary: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> dict[str, float | int]:
    elapsed_by_condition = row_summary["elapsed_by_condition"]
    condition_means = {
        condition: sum(float(value) for value in elapsed_by_condition[condition])
        / len(elapsed_by_condition[condition])
        for condition in CONDITIONS
    }
    normalization = _mapping(timings["normalization"], "normalization timing")
    normalization_seconds_per_media = float(normalization["seconds"]) / int(
        normalization["media_count"]
    )
    scheduler_residual_seconds = max(
        0.0,
        float(scheduler["elapsed_seconds"]) - float(timings["phase_total_seconds"]),
    )
    inference_residual_seconds = max(
        0.0,
        float(timings["inference_seconds"])
        - float(row_summary["elapsed_seconds"]),
    )
    fixed_seconds_per_task = (
        float(timings["bootstrap_download_seconds"])
        + float(timings["model_startup_seconds"])
        + float(timings["finalization_seconds"])
        + scheduler_residual_seconds
        + inference_residual_seconds
    )
    aggregate_inference_seconds = (
        sum(condition_means.values()) * FULL_ROW_COUNT * INFERENCE_PROJECTION_MARGIN
    )
    aggregate_seconds = (
        fixed_seconds_per_task * FULL_TASK_COUNT
        + normalization_seconds_per_media * FULL_CALL_COUNT
        + aggregate_inference_seconds
    )
    longest_shard_rows = max(FULL_SHARD_COUNTS)
    longest_shard_seconds = (
        fixed_seconds_per_task
        + normalization_seconds_per_media * longest_shard_rows
        + max(condition_means.values())
        * INFERENCE_PROJECTION_MARGIN
        * longest_shard_rows
    )
    aggregate_hours = aggregate_seconds / 3600
    longest_hours = longest_shard_seconds / 3600
    return {
        "calls": FULL_CALL_COUNT,
        "parallel_tasks": FULL_TASK_COUNT,
        "inference_margin": INFERENCE_PROJECTION_MARGIN,
        "aggregate_gpu_hours": aggregate_hours,
        "longest_shard_hours": longest_hours,
        "wall_time_hours": longest_hours,
        "observed_price_usd_per_hour": float(scheduler["price"]),
        "projected_cost_usd": aggregate_hours * float(scheduler["price"]),
    }


def _new_gates() -> dict[str, bool]:
    return {
        "config": False,
        "scheduler": False,
        "input_integrity": False,
        "terminal": False,
        "artifact_integrity": False,
        "identity": False,
        "rows": False,
        "request_hashes": False,
        "retry_count": False,
        "leakage": False,
        "telemetry": False,
        "phase_timing": False,
        "canary_timeout": False,
        "aggregate_gpu_hours": False,
        "longest_shard_hours": False,
        "projected_cost": False,
    }


def _empty_projection() -> dict[str, float | int | None]:
    return {
        "calls": FULL_CALL_COUNT,
        "parallel_tasks": FULL_TASK_COUNT,
        "inference_margin": INFERENCE_PROJECTION_MARGIN,
        "aggregate_gpu_hours": None,
        "longest_shard_hours": None,
        "wall_time_hours": None,
        "observed_price_usd_per_hour": None,
        "projected_cost_usd": None,
    }


def _bounded_no_go(reason: str = "collector_input") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "NO-GO",
        "run_id": RUN_ID,
        "attempt_uuid": ATTEMPT_UUID,
        "dstack_run_id": None,
        "gates": _new_gates(),
        "observed": {
            "valid_rows": 0,
            "schema_retries": 0,
            "elapsed_seconds": 0.0,
            "normalization_media_count": 0,
            "producer_phase_seconds": 0.0,
        },
        "projection": _empty_projection(),
        "reasons": [reason],
    }


def _adjudicate_canary(
    artifact_dir: Path,
    scheduler: Mapping[str, Any] | Path,
    *,
    input_dir: Path | None = None,
    config_path: Path | None = None,
    dstack_run_id: str | None = None,
) -> dict[str, Any]:
    """Return a bounded ``GO`` or ``NO-GO`` decision for the canary."""

    artifact_dir = Path(artifact_dir)
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else Path(__file__).resolve().parents[1]
        / "data/vqa/benchmarks/vqa-event-map-inputs-20260809-a3/inputs"
    )
    config_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[1]
        / "dstack/vqa-event-map-canary.dstack.yml"
    )
    reasons: list[str] = []
    gates = _new_gates()
    safe_scheduler: dict[str, Any] | None = None
    config_sha256: str | None = None
    requests: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    producer_timings: dict[str, Any] | None = None
    row_summary: dict[str, Any] = {
        "retry_count": 0,
        "elapsed_seconds": 0.0,
        "elapsed_by_condition": {condition: [] for condition in CONDITIONS},
    }
    projection = _empty_projection()

    try:
        config_sha256 = _validate_config(config_path)
        gates["config"] = True
    except EvidenceError:
        reasons.append("config_identity")

    if config_sha256 is not None:
        try:
            scheduler_value = (
                _mapping(_json(scheduler), "scheduler record")
                if isinstance(scheduler, Path)
                else _mapping(scheduler, "scheduler record")
            )
            safe_scheduler = sanitize_dstack_scheduler_json(
                scheduler_value,
                expected_dstack_run_id=dstack_run_id,
                expected_config_sha256=config_sha256,
            )
            gates["scheduler"] = True
        except EvidenceError:
            reasons.append("scheduler_identity")

    try:
        requests = _validate_inputs(input_dir)
        gates["input_integrity"] = True
    except EvidenceError:
        reasons.append("input_integrity")

    try:
        payload_entries = _verify_file_set(
            artifact_dir,
            EXPECTED_PAYLOAD_FILES,
            kind="artifact",
            allow_terminal=True,
        )
        gates["artifact_integrity"] = True
    except EvidenceError:
        payload_entries = {}
        reasons.append("artifact_integrity")

    if payload_entries:
        try:
            producer_timings = _validate_terminal(artifact_dir, payload_entries)
            gates["terminal"] = True
        except EvidenceError:
            reasons.append("terminal")

    if gates["input_integrity"] and gates["config"] and gates["terminal"]:
        try:
            preflight = _mapping(_json(artifact_dir / "preflight.json"), "preflight")
            _exact_keys(
                preflight,
                {
                    "schema_version",
                    "run_id",
                    "attempt_uuid",
                    "source_bundle_sha256",
                    "controller_sha256",
                    "protocol_sha256",
                    "request_file_sha256",
                    "inference_module_sha256",
                    "runner_sha256",
                    "row_count",
                    "condition_id",
                    "slice_id",
                },
                "preflight",
            )
            if preflight != {
                "schema_version": 1,
                "run_id": INPUT_RUN_ID,
                "attempt_uuid": ATTEMPT_UUID,
                "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
                "controller_sha256": CONTROLLER_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "request_file_sha256": REQUEST_FILE_SHA256,
                "inference_module_sha256": INFERENCE_MODULE_SHA256,
                "runner_sha256": _expected_runner_sha256(),
                "row_count": 8,
                "condition_id": "v-map,av-map",
                "slice_id": "",
            }:
                raise EvidenceError("preflight identity mismatch")
            _canonical_uuid(preflight["attempt_uuid"], "preflight attempt_uuid")
            gates["identity"] = True
        except EvidenceError:
            reasons.append("identity")

    if gates["artifact_integrity"] and gates["terminal"]:
        try:
            runtime = _validate_runtime(artifact_dir)
            gates["telemetry"] = True
        except EvidenceError:
            reasons.append("telemetry")

    if gates["identity"] and gates["telemetry"] and requests:
        try:
            row_summary = _validate_predictions(artifact_dir, requests, runtime)
            gates["rows"] = row_summary["rows"] == 8
            gates["request_hashes"] = True
            gates["retry_count"] = True
            gates["leakage"] = True
        except EvidenceError as error:
            message = str(error)
            if message == "retry transport failure":
                reasons.append("retry_transport_failure")
            elif "hash" in message:
                reasons.append("request_hashes")
            elif "row identity" in message or "request binding" in message:
                reasons.append("row_identity")
            elif "leakage" in message:
                reasons.append("leakage")
            elif "retry" in message:
                reasons.append("retry_count")
            elif (
                "row count" in message
                or "status" in message
                or "event" in message
                or "answer" in message
                or "duration" in message
                or "selected interval" in message
            ):
                reasons.append("rows")
            else:
                reasons.append("identity")
    elif not requests:
        reasons.append("rows")

    if safe_scheduler is not None:
        if float(safe_scheduler["elapsed_seconds"]) >= MAX_CANARY_SECONDS:
            reasons.append("canary_timeout")
        else:
            gates["canary_timeout"] = True

    if (
        safe_scheduler is not None
        and producer_timings is not None
        and gates["rows"]
        and gates["retry_count"]
    ):
        try:
            _validate_phase_reconciliation(
                safe_scheduler, row_summary, producer_timings
            )
            gates["phase_timing"] = True
        except EvidenceError:
            reasons.append("phase_timing")

    if (
        safe_scheduler is not None
        and producer_timings is not None
        and gates["phase_timing"]
    ):
        projection = _projection(safe_scheduler, row_summary, producer_timings)
        gates["aggregate_gpu_hours"] = (
            float(projection["aggregate_gpu_hours"]) <= MAX_AGGREGATE_GPU_HOURS
        )
        gates["longest_shard_hours"] = (
            float(projection["longest_shard_hours"]) <= MAX_LONGEST_SHARD_HOURS
        )
        gates["projected_cost"] = (
            float(projection["projected_cost_usd"]) <= MAX_PROJECTED_COST_USD
        )
        if not gates["aggregate_gpu_hours"]:
            reasons.append("aggregate_gpu_hours")
        if not gates["longest_shard_hours"]:
            reasons.append("longest_shard_hours")
        if not gates["projected_cost"]:
            reasons.append("projected_cost")

    ordered_reasons = list(dict.fromkeys(reasons))
    all_gates = all(gates.values())
    normalization = (
        _mapping(producer_timings["normalization"], "normalization timing")
        if producer_timings is not None
        else {}
    )
    return {
        "schema_version": 1,
        "status": "GO" if all_gates else "NO-GO",
        "run_id": RUN_ID,
        "attempt_uuid": ATTEMPT_UUID,
        "dstack_run_id": safe_scheduler.get("dstack_run_id")
        if safe_scheduler
        else None,
        "gates": gates,
        "observed": {
            "valid_rows": row_summary.get("rows", 0),
            "schema_retries": row_summary.get("retry_count", 0),
            "elapsed_seconds": row_summary.get("elapsed_seconds", 0.0),
            "normalization_media_count": normalization.get("media_count", 0),
            "producer_phase_seconds": producer_timings.get("phase_total_seconds", 0.0)
            if producer_timings
            else 0.0,
        },
        "projection": projection,
        "reasons": ordered_reasons,
    }


def adjudicate_canary(
    artifact_dir: Path,
    scheduler: Mapping[str, Any] | Path,
    *,
    input_dir: Path | None = None,
    config_path: Path | None = None,
    dstack_run_id: str | None = None,
) -> dict[str, Any]:
    """Return a bounded, fail-closed ``GO`` or ``NO-GO`` decision.

    The dstack run UUID is mandatory. Malformed paths, I/O, UTF-8, YAML,
    JSON, JSONL, and value types never escape this API as exceptions.
    """

    try:
        return _adjudicate_canary(
            artifact_dir,
            scheduler,
            input_dir=input_dir,
            config_path=config_path,
            dstack_run_id=dstack_run_id,
        )
    except Exception:
        return _bounded_no_go()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--scheduler-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dstack-run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    decision = adjudicate_canary(
        args.artifacts,
        args.scheduler_json,
        input_dir=args.input_dir,
        config_path=args.config,
        dstack_run_id=args.dstack_run_id,
    )
    encoded = json.dumps(
        decision, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if decision["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
