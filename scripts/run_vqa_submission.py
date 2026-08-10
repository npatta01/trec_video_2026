"""Small, deterministic phase-A VQA submission orchestrator.

This module intentionally implements only the label-free fixture path.  It is
safe to exercise in tests without media, credentials, or a provider SDK.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_benchmark import probe_media, source_fingerprint
from trec_video_2026.vqa_submission_evidence import (
    EvidencePlan,
    FrameSelection,
    ValidatedTimeline,
    build_global_local_frame_plan,
    select_overlapping_transcript_segments,
    validate_timeline,
)
from trec_video_2026.vqa_submission_gemini import (
    AUTHENTICATION_ERROR_MESSAGE,
    GeminiSubmissionClient,
    ProviderAuthenticationError,
    create_google_transport,
    preflight_google_transport,
)
from trec_video_2026.vqa_submission_openrouter import (
    create_openrouter_transport,
    preflight_openrouter_transport,
)
from trec_video_2026.vqa_submission import (
    MODEL_ID,
    FALLBACK_PASS_PRECEDENCE,
    PASS_IDS,
    CandidateRecord,
    RankedPrediction,
    build_answer_requests,
    build_backfill_request,
    build_verifier_request,
    audit_label_free_payload,
    deduplicate_candidates,
    freeze_predictions,
    verify_prediction_freeze,
    _normalized_answer as canonical_candidate_normalize,
)


SCHEMA_VERSION = 1
CANONICAL_DATA_ROOT = Path(
    "/home/npatta01/data/competitions/trec_video_2026/data"
)
PROTOCOL_VERSION = "task5-resumable-v2"
EVIDENCE_POLICY_VERSION = "global8-local24-pts-hash-v1"
_JSON_SEPARATORS = (",", ":")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=_JSON_SEPARATORS,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _evidence_without_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "resolved_manifest_sha256",
            "timeline_request_sha256",
            "timeline_response_sha256",
        }
    }


def _write_bytes_once(path: Path, payload: bytes) -> str:
    """Materialize deterministic fixture bytes without overwriting them."""

    if path.is_symlink():
        raise ValueError(f"refusing to write through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _sha256_bytes(payload)
    if path.exists():
        actual = _sha256_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"fixture artifact hash mismatch: {path}")
        return actual
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        actual = _sha256_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"fixture artifact hash mismatch: {path}")
    return expected


def _usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
    }


def _failure_code(error: BaseException) -> str:
    allowed = {
        "provider_request",
        "provider_upload",
        "provider_generation",
        "provider_failure",
        "protocol_failure",
        "protocol_schema",
        "protocol_no_response",
        "cleanup_delete",
    }
    code = getattr(error, "failure_code", None)
    return code if isinstance(code, str) and code in allowed else "provider_failure"


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    """Write a JSON object atomically, optionally refusing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_canonical_bytes(payload).decode("utf-8"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Replace a JSONL checkpoint atomically with canonical row ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_bytes(row).decode("utf-8"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one fsynced checkpoint record without rewriting history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(row) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("resolved_manifest_sha256", None)
    return _sha256_json(payload)


def _panel_hash(examples: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(
        [
            {
                "query_id": item["query_id"],
                "video_id": item["video_id"],
                "category": item.get("category", "general"),
                "question": item["question"],
            }
            for item in examples
        ]
    )


def _hash_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_evidence_placeholder(value: Mapping[str, Any], run_id: str) -> bool:
    return (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("run_id") == run_id
        and isinstance(value.get("rows"), list)
        and all(
            isinstance(row, Mapping)
            and set(row) == {"query_id", "video_id"}
            for row in value["rows"]
        )
    )


def _evidence_aggregate_can_extend(
    current: Mapping[str, Any], desired: Mapping[str, Any]
) -> bool:
    current_rows = current.get("rows")
    desired_rows = desired.get("rows")
    return (
        current.get("schema_version") == desired.get("schema_version")
        and current.get("run_id") == desired.get("run_id")
        and isinstance(current_rows, list)
        and isinstance(desired_rows, list)
        and len(current_rows) <= len(desired_rows)
        and current_rows == desired_rows[: len(current_rows)]
    )


def _is_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _reject_run_symlinks(run_path: Path) -> None:
    if run_path.is_symlink():
        raise ValueError(f"run path must not be a symlink: {run_path}")
    for child in run_path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"run artifact must not be a symlink: {child}")


def _safe_media_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "duration_seconds"):
        return {
            "duration_seconds": float(value.duration_seconds),
            "has_video": bool(value.has_video),
            "has_audio": bool(value.has_audio),
            "stream_types": list(value.stream_types),
            "streams": [
                {
                    "codec_type": item.codec_type,
                    "codec_name": item.codec_name,
                    "channels": item.channels,
                }
                for item in value.streams
            ],
        }
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("probe_media must return media metadata")


def _safe_source_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "sha256"):
        return {"sha256": str(value.sha256), "size_bytes": int(value.size_bytes)}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("source_fingerprint must return source metadata")


def _example_key(example: Mapping[str, Any]) -> tuple[int, str]:
    query_id = example.get("query_id")
    video_id = example.get("video_id")
    if isinstance(query_id, bool) or not isinstance(query_id, int):
        raise ValueError("each example query_id must be an integer")
    if query_id < 0 or not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("each example needs a non-negative query_id and video_id")
    return query_id, video_id


def _validate_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {SCHEMA_VERSION}")
    if not isinstance(config.get("run_id"), str) or not config["run_id"].strip():
        raise ValueError("config run_id must be a non-empty string")
    if config.get("model_id") != MODEL_ID:
        raise ValueError(f"config model_id must be {MODEL_ID}")
    if not isinstance(config.get("fixture_mode"), bool):
        raise ValueError("config fixture_mode must be explicitly true or false")
    if "require_audio" in config and not isinstance(config["require_audio"], bool):
        raise ValueError("config require_audio must be boolean")
    examples = config.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("config examples must be a non-empty list")
    baseline = config.get("baseline_ranked_list", ("fixture baseline response",))
    if (
        not isinstance(baseline, (list, tuple))
        or not baseline
        or not all(isinstance(item, str) and item.strip() for item in baseline)
    ):
        raise ValueError("baseline_ranked_list must be a non-empty list of strings")
    audit_label_free_payload(config)
    normalized: list[dict[str, Any]] = []
    seen_queries: set[int] = set()
    seen_videos: set[str] = set()
    for raw_example in examples:
        if not isinstance(raw_example, dict):
            raise TypeError("each config example must be an object")
        example = dict(raw_example)
        query_id, video_id = _example_key(example)
        if query_id in seen_queries or video_id in seen_videos:
            raise ValueError("config examples must have unique query_id and video_id")
        if not isinstance(example.get("question"), str) or not example["question"].strip():
            raise ValueError("each example question must be a non-empty string")
        seen_queries.add(query_id)
        seen_videos.add(video_id)
        normalized.append(example)
    return tuple(normalized)


def _baseline_ranked_list(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("baseline_ranked_list", ("fixture baseline response",))
    return tuple(str(item) for item in raw)


@dataclass(frozen=True)
class PreparedRun:
    """The immutable inputs and location of a prepared run."""

    path: Path
    run_id: str
    model_id: str
    examples: tuple[Mapping[str, Any], ...]
    baseline_ranked_list: tuple[str, ...] = ()
    fixture_mode: bool = True
    resolved_manifest_sha256: str = ""
    panel_sha256: str = ""

    @property
    def run_dir(self) -> Path:
        return self.path

    @property
    def output_dir(self) -> Path:
        return self.path

    @classmethod
    def load(cls, path: Path | str) -> "PreparedRun":
        run_path = Path(path)
        _reject_run_symlinks(run_path)
        manifest = _read_json(run_path / "run-manifest.json")
        declared_hash = manifest.get("resolved_manifest_sha256")
        if not isinstance(declared_hash, str) or declared_hash != _manifest_hash(manifest):
            raise ValueError("resolved manifest hash does not match run-manifest.json")
        raw_examples = manifest.get("examples")
        if not isinstance(raw_examples, list):
            raise ValueError("run manifest examples must be a list")
        if manifest.get("model_id") != MODEL_ID:
            raise ValueError("prepared manifest model_id does not match the protocol")
        if tuple(manifest.get("pass_ids", ())) != PASS_IDS:
            raise ValueError("prepared manifest pass_ids do not match the protocol")
        if manifest.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("prepared manifest protocol version does not match")
        if manifest.get("evidence_policy_version") != EVIDENCE_POLICY_VERSION:
            raise ValueError("prepared manifest evidence policy does not match")
        if manifest.get("panel_sha256") != _panel_hash(raw_examples):
            raise ValueError("prepared manifest panel hash does not match examples")
        raw_baseline = manifest.get("baseline_ranked_list", ())
        if not isinstance(raw_baseline, list):
            raise ValueError("run manifest baseline_ranked_list must be a list")
        return cls(
            path=run_path,
            run_id=str(manifest["run_id"]),
            model_id=str(manifest["model_id"]),
            examples=tuple(dict(example) for example in raw_examples),
            baseline_ranked_list=tuple(str(item) for item in raw_baseline),
            fixture_mode=bool(manifest.get("fixture_mode")),
            resolved_manifest_sha256=declared_hash,
            panel_sha256=str(manifest["panel_sha256"]),
        )


@dataclass(frozen=True)
class RunSummary:
    """Counts for one resumable invocation."""

    prepared_rows: int
    attempted_rows: int
    completed_rows: int
    reused_rows: int
    generated_rows: int = 0
    failed_rows: int = 0
    inferred_rows: int = 0
    candidate_shortfall: int = 0
    frozen: bool = False
    labels_loaded: bool = False


class Transport(Protocol):
    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class FixtureTransport:
    """Credential-free deterministic transport used by phase-A tests."""

    def __init__(self) -> None:
        self.generate_count = 0

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.generate_count += 1
        stage = str(request.get("stage", "answer"))
        if stage == "timeline":
            duration = float(request.get("duration_seconds", 10.0))
            return {
                "schema_version": 1,
                "duration_seconds": duration,
                "events": [
                    {
                        "span_id": "fixture-event",
                        "start_seconds": 0.0,
                        "end_seconds": min(2.0, duration),
                        "modalities": ["visual"],
                        "description": "A deterministic fixture event.",
                    }
                ],
                "support_spans": [
                    {
                        "span_id": "fixture-support",
                        "start_seconds": min(2.0, duration / 2),
                        "end_seconds": min(4.0, duration),
                        "modalities": ["visual", "speech"],
                        "description": "A deterministic fixture support span.",
                    }
                ],
            }
        if stage == "verifier":
            return {
                "ranked_pass_ids": list(PASS_IDS),
                "selected_pass_id": PASS_IDS[0],
                "support_by_pass": {},
                "status_by_pass": {pass_id: "supported" for pass_id in PASS_IDS},
            }
        pass_id = str(request.get("pass_id", "fixture"))
        if stage == "backfill":
            return {
                "answer": f"Fixture backfill response {request.get('attempt', 1)}.",
                "confidence": 0.5,
                "status": "passed",
            }
        return {
            "answer": f"Fixture response for {pass_id}.",
            "confidence": 0.5,
            "status": "passed",
        }


def _resolve_live_examples(
    config: Mapping[str, Any], dataset_root: Path
) -> tuple[dict[str, Any], ...]:
    canonical_root = CANONICAL_DATA_ROOT.resolve(strict=True)
    resolved_root = dataset_root.resolve(strict=True)
    if resolved_root != canonical_root:
        raise ValueError("dataset-root must resolve to the canonical shared data root")
    examples: list[dict[str, Any]] = []
    for raw in config["examples"]:
        relative = raw.get("media", raw.get("media_path"))
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("non-fixture examples require a media path")
        media_path = (resolved_root / relative).resolve(strict=True)
        if not media_path.is_relative_to(resolved_root) or not _is_regular_file(media_path):
            raise ValueError("media path must be a regular file beneath dataset-root")
        metadata = _safe_media_metadata(probe_media(media_path))
        duration = float(metadata.get("duration_seconds", 0.0))
        if duration < 1.0 or not metadata.get("has_video", False):
            raise ValueError("media must contain video and have duration at least one second")
        if config.get("require_audio", True) and not metadata.get("has_audio", False):
            raise ValueError("live canary requires an audio stream")
        source = _safe_source_metadata(source_fingerprint(media_path))
        example = dict(raw)
        example.pop("media", None)
        example["media_path"] = str(media_path)
        example["source"] = source
        example["media"] = metadata
        audio_relative = raw.get("audio", raw.get("audio_path"))
        if audio_relative is not None:
            if not isinstance(audio_relative, str) or not audio_relative.strip():
                raise ValueError("audio path must be a non-empty string")
            audio_path = (resolved_root / audio_relative).resolve(strict=True)
            if not audio_path.is_relative_to(resolved_root) or not _is_regular_file(audio_path):
                raise ValueError("audio path must be a regular file beneath dataset-root")
            example["audio_path"] = str(audio_path)
            example["audio_source"] = _safe_source_metadata(source_fingerprint(audio_path))
        examples.append(example)
    return tuple(examples)


def prepare(
    config_path: Path | str,
    output: Path | str,
    *,
    dataset_root: Path | str | None = None,
) -> PreparedRun:
    """Prepare a label-free run directory atomically and exactly once."""

    config_file = Path(config_path)
    output_path = Path(output)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("config must contain a JSON object")
    examples = _validate_config(config)
    if config["fixture_mode"]:
        resolved_examples = examples
    else:
        if dataset_root is None:
            raise ValueError("non-fixture prepare requires --dataset-root")
        resolved_examples = _resolve_live_examples(config, Path(dataset_root))

    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=parent))
    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": config["run_id"],
            "model_id": config["model_id"],
            "fixture_mode": config["fixture_mode"],
            "config_sha256": _sha256_bytes(config_bytes),
            "examples": [dict(example) for example in resolved_examples],
            "pass_ids": list(PASS_IDS),
            "baseline_ranked_list": list(_baseline_ranked_list(config)),
            "protocol_version": PROTOCOL_VERSION,
            "evidence_policy_version": EVIDENCE_POLICY_VERSION,
            "protocol_sha256": _sha256_bytes(PROTOCOL_VERSION.encode()),
            "evidence_policy_sha256": _sha256_bytes(EVIDENCE_POLICY_VERSION.encode()),
            "panel_sha256": _panel_hash(resolved_examples),
        }
        manifest["resolved_manifest_sha256"] = _manifest_hash(manifest)
        _write_json(temporary / "run-manifest.json", manifest, overwrite=False)
        _write_json(
            temporary / "evidence" / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": config["run_id"],
                "rows": [
                    {"query_id": example["query_id"], "video_id": example["video_id"]}
                    for example in resolved_examples
                ],
            },
            overwrite=False,
        )
        for name in ("timeline.jsonl", "candidates.jsonl", "failures.jsonl"):
            _write_jsonl(temporary / name, ())
        _write_json(
            temporary / "terminal-status.json",
            {
                "status": "prepared",
                "prepared_rows": len(resolved_examples),
                "inferred_rows": 0,
                "frozen": False,
                "labels_loaded": False,
            },
            overwrite=False,
        )
        if output_path.exists():
            raise FileExistsError(output_path)
        os.rename(temporary, output_path)
    except Exception:
        if temporary.exists():
            for child in sorted(temporary.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            temporary.rmdir()
        raise
    return PreparedRun.load(output_path)


def _request(prepared: PreparedRun, example: Mapping[str, Any], pass_id: str) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "model_id": prepared.model_id,
        "query_id": example["query_id"],
        "video_id": example["video_id"],
        "category": example.get("category", "general"),
        "question": example["question"],
        "pass_id": pass_id,
    }
    return _bind_request(prepared, request)


def _bind_request(
    prepared: PreparedRun, request: Mapping[str, Any]
) -> dict[str, Any]:
    bound = dict(request)
    bound.update(
        {
            "resolved_manifest_sha256": prepared.resolved_manifest_sha256,
            "model_id": prepared.model_id,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": _sha256_bytes(PROTOCOL_VERSION.encode()),
            "evidence_policy_version": EVIDENCE_POLICY_VERSION,
            "evidence_policy_sha256": _sha256_bytes(EVIDENCE_POLICY_VERSION.encode()),
            "panel_sha256": prepared.panel_sha256,
        }
    )
    audit_label_free_payload(bound)
    return bound


def _validate_prepared_sources(prepared: PreparedRun) -> None:
    _reject_run_symlinks(prepared.path)
    for example in prepared.examples:
        if prepared.fixture_mode:
            continue
        media_path = Path(str(example["media_path"]))
        if not _is_regular_file(media_path):
            raise ValueError("prepared media path is missing or not a regular file")
        source = _safe_source_metadata(source_fingerprint(media_path))
        declared = example.get("source")
        if not isinstance(declared, Mapping) or source.get("sha256") != declared.get("sha256"):
            raise ValueError("prepared source hash does not match current media")
        audio_path_value = example.get("audio_path")
        if audio_path_value is not None:
            audio_path = Path(str(audio_path_value))
            audio_source = example.get("audio_source")
            if not _is_regular_file(audio_path) or not isinstance(audio_source, Mapping):
                raise ValueError("prepared audio source is missing or invalid")
            actual_audio = _safe_source_metadata(source_fingerprint(audio_path))
            if actual_audio.get("sha256") != audio_source.get("sha256"):
                raise ValueError("prepared source hash does not match current audio")


def _matches(record: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    saved_request = record.get("request")
    if not isinstance(saved_request, Mapping):
        return False
    try:
        saved_hash = canonical_request_sha256(saved_request)
        current_hash = canonical_request_sha256(request)
    except (TypeError, ValueError):
        return False
    return (
        saved_hash == current_hash
        and record.get("request_sha256") == saved_hash
        and record.get("status") == "completed"
    )


def _candidate_record(
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    elapsed_seconds: float,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    candidate_pass_id: str | None = None,
    provider: str = "fixture",
    support_span_ids: Sequence[str] = ("fixture",),
) -> dict[str, Any]:
    response_hash = _sha256_json(response)
    answer = response.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("transport response answer must be a non-empty string")
    return {
        "query_id": request["query_id"],
        "video_id": request["video_id"],
        "pass_id": candidate_pass_id or request["pass_id"],
        "request": dict(request),
        "request_sha256": canonical_request_sha256(request),
        "response": dict(response),
        "response_sha256": response_hash,
        "answer": answer,
        "confidence": float(response.get("confidence", 0.5)),
        "elapsed_seconds": elapsed_seconds,
        "status": "completed",
        "stage": request.get("stage", "answer"),
        "usage": dict((checkpoint or {}).get("usage", _usage())),
        "uploaded_file_deletion_status": dict(
            (checkpoint or {}).get("uploaded_file_deletion_status", {})
        ),
        "failure_code": (checkpoint or {}).get("failure_code"),
        "provider": provider,
        "runtime": provider,
        "support_span_ids": list(support_span_ids),
    }


def _runtime_provider(transport: object, *, fixture_mode: bool) -> str:
    if fixture_mode:
        return "fixture"
    provider = getattr(transport, "provider_name", None)
    if not isinstance(provider, str) or not provider:
        provider = getattr(getattr(transport, "_transport", None), "provider_name", None)
    return provider if isinstance(provider, str) and provider else "gemini"


def _timeline_request(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    *,
    provider_video: Mapping[str, str] | None = None,
    provider_name: str = "gemini",
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "stage": "timeline",
        "run_id": prepared.run_id,
        "model_id": prepared.model_id,
        "query_id": example["query_id"],
        "video_id": example["video_id"],
        "question": example["question"],
        "duration_seconds": float(
            example.get("duration_seconds", example.get("media", {}).get("duration_seconds", 10.0))
        ),
    }
    if not prepared.fixture_mode:
        media_path = str(
            provider_video["path"] if provider_video is not None else example["media_path"]
        )
        audio_path, audio_sha = _live_audio_path(prepared, example)
        request["media_path"] = media_path
        request["audio_path"] = str(audio_path)
        request["media_sha256"] = (
            provider_video["sha256"]
            if provider_video is not None
            else example["source"]["sha256"]
        )
        if provider_name == "openrouter" or provider_video is not None:
            request["source_media_sha256"] = example["source"]["sha256"]
        request["audio_sha256"] = audio_sha
        if provider_video is not None:
            request["provider_video_sha256"] = provider_video["sha256"]
    return _bind_request(prepared, request)


def _attempt_key(request: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(request["query_id"]),
        str(request.get("stage", "answer")),
        str(request.get("pass_id", "")),
    )


def _attempt_matches(
    record: Mapping[str, Any], request: Mapping[str, Any]
) -> bool:
    if record.get("status") != "completed":
        return False
    saved_request = record.get("request")
    if not isinstance(saved_request, Mapping):
        return False
    try:
        request_hash = canonical_request_sha256(request)
        saved_hash = canonical_request_sha256(saved_request)
        response = record.get("response")
        response_hash = None if response is None else _sha256_json(response)
    except (TypeError, ValueError):
        return False
    return (
        request_hash == saved_hash
        and record.get("request_sha256") == request_hash
        and record.get("request_bytes") == _canonical_bytes(request).decode("utf-8")
        and record.get("response_payload_sha256", record.get("response_sha256"))
        == response_hash
    )


def _request_hash_matches(record: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    saved_request = record.get("request")
    if not isinstance(saved_request, Mapping):
        return False
    try:
        expected = canonical_request_sha256(request)
        actual = canonical_request_sha256(saved_request)
    except (TypeError, ValueError):
        return False
    return actual == expected and record.get("request_sha256") == expected


def _result_payload(result: Any, stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    usage_value = getattr(result, "usage", None)
    usage = {
        field: int(getattr(usage_value, field, 0))
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
        )
    }
    metadata = {
        "usage": usage,
        "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
        "uploaded_file_deletion_status": dict(
            getattr(result, "uploaded_file_deletion_status", {})
        ),
        "response_sha256": getattr(result, "response_sha256", None),
        "failure_code": None,
    }
    if stage == "timeline":
        value = getattr(result, "timeline", result)
        payload = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    elif stage in {"answer", "backfill"}:
        payload = {"answer": getattr(result, "answer", "")}
    else:
        decision = getattr(result, "decision", result)
        if hasattr(decision, "to_dict"):
            payload = decision.to_dict()
        else:
            payload = dict(decision)
    return payload, metadata


def _call_transport(
    transport: Any, request: Mapping[str, Any], provider_request: Any = None
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    stage = str(request.get("stage", "answer"))
    if callable(getattr(transport, "generate", None)):
        response = transport.generate(request)
        if not isinstance(response, Mapping):
            raise TypeError("transport.generate must return a mapping")
        return dict(response), {"usage": _usage(), "uploaded_file_deletion_status": {}}
    if stage == "timeline":
        result = transport.timeline(provider_request or request)
    elif stage in {"answer", "backfill"}:
        result = transport.answer(provider_request)
    else:
        verifier = getattr(transport, "verify", None) or getattr(transport, "verifier")
        result = verifier(provider_request)
    return _result_payload(result, stage)


def _run_attempt(
    request: Mapping[str, Any],
    transport: Transport,
    attempts: dict[tuple[int, str, str], dict[str, Any]],
    failures: dict[tuple[int, str, str], dict[str, Any]],
    *,
    attempts_path: Path | None = None,
    failures_path: Path | None = None,
    provider_request: Any = None,
    validator: Any = None,
) -> tuple[Mapping[str, Any] | None, dict[str, Any], bool]:
    key = _attempt_key(request)
    saved = attempts.get(key)
    if saved is not None and saved.get("status") == "started":
        if _request_hash_matches(saved, request):
            raise ValueError("in_flight_unknown")
        raise ValueError("in_flight_unknown")
    if saved is not None and _attempt_matches(saved, request):
        response = saved.get("response")
        try:
            if validator is not None:
                validator(response)
        except Exception:
            pass
        else:
            return (response if isinstance(response, Mapping) else None), saved, True
    request_bytes = _canonical_bytes(request)
    started_record: dict[str, Any] = {
        "query_id": request["query_id"],
        "video_id": request["video_id"],
        "stage": request.get("stage", "answer"),
        "pass_id": request.get("pass_id"),
        "request": dict(request),
        "request_bytes": request_bytes.decode("utf-8"),
        "request_sha256": _sha256_bytes(request_bytes),
        "response": None,
        "response_bytes": None,
        "response_sha256": None,
        "response_payload_sha256": None,
        "status": "started",
        "elapsed_seconds": 0.0,
        "usage": _usage(),
        "uploaded_file_deletion_status": {},
        "failure_code": None,
    }
    if attempts_path is not None:
        _append_jsonl(attempts_path, started_record)
    started = time.monotonic()
    record = dict(started_record)
    record["status"] = "completed"
    try:
        response_dict, metadata = _call_transport(transport, request, provider_request)
        if validator is not None:
            validator(response_dict)
        response_bytes = _canonical_bytes(response_dict)
        record["response"] = response_dict
        record["response_bytes"] = response_bytes.decode("utf-8")
        record["response_payload_sha256"] = _sha256_bytes(response_bytes)
        record["response_sha256"] = metadata.get("response_sha256") or _sha256_bytes(response_bytes)
        record["usage"] = dict(metadata.get("usage", _usage()))
        record["uploaded_file_deletion_status"] = dict(
            metadata.get("uploaded_file_deletion_status", {})
        )
        record["elapsed_seconds"] = max(
            float(metadata.get("elapsed_seconds", 0.0)),
            time.monotonic() - started,
        )
    except Exception as error:  # noqa: BLE001 - bounded transport checkpoint
        response = record["response"] if isinstance(record["response"], Mapping) else None
        record["status"] = "failed"
        failure = getattr(error, "failure_record", None)
        record["failure_code"] = (
            getattr(failure, "failure_code", None) or _failure_code(error)
        )
        if failure is not None:
            record["usage"] = failure.to_dict().get("usage", _usage())
            record["elapsed_seconds"] = float(failure.elapsed_seconds)
            record["response_sha256"] = failure.response_sha256
            record["uploaded_file_deletion_status"] = dict(
                failure.uploaded_file_deletion_status
            )
        else:
            record["elapsed_seconds"] = time.monotonic() - started
        failures[key] = record
    attempts[key] = record
    if attempts_path is not None:
        _append_jsonl(attempts_path, record)
    if record["status"] == "failed" and failures_path is not None:
        _append_jsonl(failures_path, record)
    if record["status"] == "completed":
        response = record["response"]
    return response, record, False


def _fixture_evidence(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    timeline: ValidatedTimeline,
) -> tuple[EvidencePlan, dict[str, Any]]:
    duration = float(example.get("duration_seconds", 10.0))
    evidence_root = prepared.path / "evidence" / "rows" / str(example["query_id"])
    media_path = evidence_root / "fixture-video.mp4"
    audio_path = evidence_root / "fixture-audio.wav"
    media_sha = _write_bytes_once(
        media_path,
        f"fixture-media|{prepared.run_id}|{example['video_id']}".encode(),
    )
    _write_bytes_once(
        audio_path,
        f"fixture-audio|{prepared.run_id}|{example['video_id']}".encode(),
    )
    inventory: list[dict[str, Any]] = []
    frame_hashes: list[str] = []
    for frame_index in range(40):
        frame_path = evidence_root / "frames" / f"frame-{frame_index:03d}.bin"
        frame_hash = _write_bytes_once(
            frame_path,
            f"fixture-frame|{prepared.run_id}|{example['video_id']}|{frame_index}".encode(),
        )
        frame_hashes.append(frame_hash)
        inventory.append(
            {
                "stream_index": 0,
                "frame_index": frame_index,
                "actual_pts_seconds": round((frame_index + 0.5) * duration / 40, 6),
                "time_base": "1/1000",
                "source_media_sha256": media_sha,
                "image_sha256": frame_hash,
            }
        )
    plan_without_paths = build_global_local_frame_plan(duration, inventory, timeline)
    frame_paths = {
        frame.frame_index: evidence_root / "frames" / f"frame-{frame.frame_index:03d}.bin"
        for frame in plan_without_paths.global_frames + plan_without_paths.local_frames
    }
    plan = EvidencePlan(
        duration_seconds=plan_without_paths.duration_seconds,
        global_frames=tuple(
            FrameSelection(
                requested_time_seconds=frame.requested_time_seconds,
                actual_pts_seconds=frame.actual_pts_seconds,
                stream_index=frame.stream_index,
                frame_index=frame.frame_index,
                time_base=frame.time_base,
                source_media_sha256=frame.source_media_sha256,
                image_sha256=frame_hashes[frame.frame_index],
                frame_path=frame_paths[frame.frame_index],
                slot_kind=frame.slot_kind,
                support_span_id=frame.support_span_id,
                fallback_reason=frame.fallback_reason,
            )
            for frame in plan_without_paths.global_frames
        ),
        local_frames=tuple(
            FrameSelection(
                requested_time_seconds=frame.requested_time_seconds,
                actual_pts_seconds=frame.actual_pts_seconds,
                stream_index=frame.stream_index,
                frame_index=frame.frame_index,
                time_base=frame.time_base,
                source_media_sha256=frame.source_media_sha256,
                image_sha256=frame_hashes[frame.frame_index],
                frame_path=frame_paths[frame.frame_index],
                slot_kind=frame.slot_kind,
                support_span_id=frame.support_span_id,
                fallback_reason=frame.fallback_reason,
            )
            for frame in plan_without_paths.local_frames
        ),
        support_spans=plan_without_paths.support_spans,
    )
    manifest = plan.to_dict()
    manifest.update(
        {
            "run_id": prepared.run_id,
            "query_id": example["query_id"],
            "video_id": example["video_id"],
            "source_media_sha256": media_sha,
            "audio_sha256": _sha256_bytes(audio_path.read_bytes()),
            "materialized_frame_hashes": frame_hashes,
            "timeline_fallback_reason": timeline.fallback_reason,
        }
    )
    manifest_path = evidence_root / "manifest.json"
    if manifest_path.exists():
        current = _read_json(manifest_path)
        if current != manifest:
            if _evidence_without_binding(current) == _evidence_without_binding(manifest):
                manifest = current
            elif current.get("timeline_request_sha256"):
                raise ValueError("refusing to overwrite mismatched evidence manifest")
            else:
                _write_json(manifest_path, manifest, overwrite=True)
    else:
        _write_json(manifest_path, manifest, overwrite=False)
    aggregate_path = prepared.path / "evidence" / "manifest.json"
    if len(prepared.examples) == 1:
        aggregate = manifest
    else:
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "run_id": prepared.run_id,
            "rows": [
                _read_json(
                    prepared.path
                    / "evidence"
                    / "rows"
                    / str(item["query_id"])
                    / "manifest.json"
                )
                for item in prepared.examples
                if (
                    prepared.path
                    / "evidence"
                    / "rows"
                    / str(item["query_id"])
                    / "manifest.json"
                ).exists()
            ],
        }
    if aggregate_path.exists():
        current_aggregate = _read_json(aggregate_path)
        if current_aggregate != aggregate and not (
            _is_evidence_placeholder(current_aggregate, prepared.run_id)
            or _evidence_aggregate_can_extend(current_aggregate, aggregate)
            or not any(
                isinstance(row, Mapping) and not row.get("timeline_request_sha256")
                for row in current_aggregate.get("rows", [])
            )
        ):
            raise ValueError("refusing to overwrite mismatched evidence manifest")
        if current_aggregate == aggregate:
            aggregate = current_aggregate
        else:
            _write_json(aggregate_path, aggregate, overwrite=True)
    else:
        _write_json(aggregate_path, aggregate, overwrite=False)
    shadow_root = Path(tempfile.gettempdir()) / f"vqa-fixture-{_sha256_json({'run': prepared.run_id, 'query': example['query_id']})[:20]}"
    shadow_media = shadow_root / "fixture-video.mp4"
    shadow_audio = shadow_root / "fixture-audio.wav"
    _write_bytes_once(shadow_media, media_path.read_bytes())
    _write_bytes_once(shadow_audio, audio_path.read_bytes())
    shadow_plan = EvidencePlan(
        duration_seconds=plan.duration_seconds,
        global_frames=tuple(
            FrameSelection(
                requested_time_seconds=frame.requested_time_seconds,
                actual_pts_seconds=frame.actual_pts_seconds,
                stream_index=frame.stream_index,
                frame_index=frame.frame_index,
                time_base=frame.time_base,
                source_media_sha256=frame.source_media_sha256,
                image_sha256=frame.image_sha256,
                frame_path=shadow_root / "frames" / f"frame-{frame.frame_index:03d}.bin",
                slot_kind=frame.slot_kind,
                support_span_id=frame.support_span_id,
                fallback_reason=frame.fallback_reason,
            )
            for frame in plan.global_frames
        ),
        local_frames=tuple(
            FrameSelection(
                requested_time_seconds=frame.requested_time_seconds,
                actual_pts_seconds=frame.actual_pts_seconds,
                stream_index=frame.stream_index,
                frame_index=frame.frame_index,
                time_base=frame.time_base,
                source_media_sha256=frame.source_media_sha256,
                image_sha256=frame.image_sha256,
                frame_path=shadow_root / "frames" / f"frame-{frame.frame_index:03d}.bin",
                slot_kind=frame.slot_kind,
                support_span_id=frame.support_span_id,
                fallback_reason=frame.fallback_reason,
            )
            for frame in plan.local_frames
        ),
        support_spans=plan.support_spans,
    )
    for frame_index, frame_hash in enumerate(frame_hashes):
        _write_bytes_once(
            shadow_root / "frames" / f"frame-{frame_index:03d}.bin",
            (evidence_root / "frames" / f"frame-{frame_index:03d}.bin").read_bytes(),
        )
    return plan, {
        "media_path": media_path,
        "audio_path": audio_path,
        "media_sha256": media_sha,
        "audio_sha256": _sha256_bytes(audio_path.read_bytes()),
        "model_media_path": shadow_media,
        "model_audio_path": shadow_audio,
        "model_plan": shadow_plan,
    }


def _decode_live_frame_inventory(media_path: Path, duration: float) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_duration_time",
            "-of",
            "json",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    source_sha = _safe_source_metadata(source_fingerprint(media_path))["sha256"]
    inventory: list[dict[str, Any]] = []
    for frame_index, raw in enumerate(payload.get("frames", ())):
        try:
            pts = float(raw["best_effort_timestamp_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= pts <= duration:
            inventory.append(
                {
                    "stream_index": 0,
                    "frame_index": frame_index,
                    "actual_pts_seconds": round(pts, 6),
                    "time_base": "1/1000",
                    "source_media_sha256": source_sha,
                }
            )
    return inventory


def _extract_live_frame(
    media_path: Path, frame: FrameSelection, destination: Path
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"refusing to reuse unsafe frame path: {destination}")
        return _hash_path(destination)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{frame.actual_pts_seconds:.6f}",
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    if not _is_regular_file(destination):
        raise ValueError(f"frame extraction did not create a regular file: {destination}")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return _hash_path(destination)


def _extract_live_audio(media_path: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"refusing to reuse unsafe audio path: {destination}")
        return _hash_path(destination)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    if not _is_regular_file(destination):
        raise ValueError(f"audio extraction did not create a regular file: {destination}")
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())
    return _hash_path(destination)


def _live_provider_video(
    prepared: PreparedRun, example: Mapping[str, Any]
) -> dict[str, str]:
    """Create or validate the durable video-only WebM sent through OpenRouter."""

    media_path = Path(str(example["media_path"]))
    source_sha = _safe_source_metadata(source_fingerprint(media_path))["sha256"]
    evidence_root = prepared.path / "evidence" / "rows" / str(example["query_id"])
    destination = evidence_root / "provider-video.webm"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("refusing to reuse unsafe provider video path")

    expected_sha: str | None = None
    manifest_path = evidence_root / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        declared_source = manifest.get("source_media_sha256")
        declared_path = manifest.get("provider_video_path")
        declared_provider = manifest.get("provider_video_sha256")
        if declared_source is not None and declared_source != source_sha:
            raise ValueError("provider video source hash does not match current media")
        if declared_path is not None and declared_path != str(destination):
            raise ValueError("provider video path does not match the row evidence")
        if declared_provider is not None:
            if not isinstance(declared_provider, str):
                raise ValueError("provider video hash is invalid")
            expected_sha = declared_provider

    if destination.exists():
        actual_sha = _hash_path(destination)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError("provider video hash does not match the evidence manifest")
        return {
            "path": str(destination),
            "sha256": actual_sha,
            "source_sha256": source_sha,
        }

    evidence_root.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".provider-video.", suffix=".webm.tmp", dir=evidence_root
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(media_path),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-f",
                "webm",
                "-y",
                str(temporary),
            ],
            check=True,
            capture_output=True,
        )
        if not _is_regular_file(temporary):
            raise ValueError("provider video stream-copy did not create a regular file")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise ValueError("provider video artifact appeared during creation")
        os.replace(temporary, destination)
        directory_fd = os.open(evidence_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except subprocess.CalledProcessError:
        raise ValueError("provider video stream-copy failed") from None
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": _hash_path(destination),
        "source_sha256": source_sha,
    }


def _live_audio_path(prepared: PreparedRun, example: Mapping[str, Any]) -> tuple[Path, str]:
    media_path = Path(str(example["media_path"]))
    audio_path = prepared.path / "evidence" / "rows" / str(example["query_id"]) / "audio.wav"
    return audio_path, _extract_live_audio(media_path, audio_path)


def _live_evidence(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    timeline: ValidatedTimeline,
    *,
    provider_video: Mapping[str, str] | None = None,
) -> tuple[EvidencePlan, dict[str, Any]]:
    media_path = Path(str(example["media_path"]))
    source_media_sha = _safe_source_metadata(source_fingerprint(media_path))["sha256"]
    model_media_path = (
        Path(provider_video["path"])
        if provider_video is not None
        else media_path
    )
    model_media_sha = (
        provider_video["sha256"]
        if provider_video is not None
        else source_media_sha
    )
    audio_path, audio_sha = _live_audio_path(prepared, example)
    duration = float(example["media"]["duration_seconds"])
    inventory = _decode_live_frame_inventory(media_path, duration)
    plan_without_paths = build_global_local_frame_plan(duration, inventory, timeline)
    evidence_root = prepared.path / "evidence" / "rows" / str(example["query_id"])
    frame_root = evidence_root / "frames"
    materialized: list[FrameSelection] = []
    for index, frame in enumerate(plan_without_paths.all_frames):
        frame_path = frame_root / f"frame-{index:03d}.png"
        image_sha = _extract_live_frame(media_path, frame, frame_path)
        materialized.append(
            FrameSelection(
                requested_time_seconds=frame.requested_time_seconds,
                actual_pts_seconds=frame.actual_pts_seconds,
                stream_index=frame.stream_index,
                frame_index=frame.frame_index,
                time_base=frame.time_base,
                source_media_sha256=frame.source_media_sha256,
                image_sha256=image_sha,
                frame_path=frame_path,
                slot_kind=frame.slot_kind,
                support_span_id=frame.support_span_id,
                fallback_reason=frame.fallback_reason,
            )
        )
    global_count = len(plan_without_paths.global_frames)
    transcript = None
    full_transcript = example.get(
        "full_transcript",
        example.get("timestamped_whisper", example.get("whisper_text")),
    )
    if full_transcript is not None:
        transcript = select_overlapping_transcript_segments(
            str(full_transcript), timeline, duration_seconds=duration
        )
    plan = EvidencePlan(
        duration_seconds=duration,
        global_frames=tuple(materialized[:global_count]),
        local_frames=tuple(materialized[global_count:]),
        support_spans=plan_without_paths.support_spans,
        transcript=transcript,
    )
    manifest = plan.to_dict()
    manifest.update(
        {
            "run_id": prepared.run_id,
            "query_id": example["query_id"],
            "video_id": example["video_id"],
            "source_media_sha256": source_media_sha,
            "audio_sha256": audio_sha,
            "materialized_frame_hashes": [frame.image_sha256 for frame in materialized],
            "timeline_fallback_reason": timeline.fallback_reason,
        }
    )
    if provider_video is not None:
        manifest.update(
            {
                "provider_video_path": provider_video["path"],
                "provider_video_sha256": provider_video["sha256"],
            }
        )
    manifest_path = evidence_root / "manifest.json"
    if manifest_path.exists():
        current_manifest = _read_json(manifest_path)
        if current_manifest != manifest:
            if _evidence_without_binding(current_manifest) == _evidence_without_binding(manifest):
                manifest = current_manifest
            elif current_manifest.get("timeline_request_sha256"):
                raise ValueError("refusing to overwrite mismatched evidence manifest")
            else:
                _write_json(manifest_path, manifest, overwrite=True)
    else:
        _write_json(manifest_path, manifest, overwrite=False)
    aggregate_path = prepared.path / "evidence" / "manifest.json"
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared.run_id,
        "rows": [
            _read_json(
                prepared.path / "evidence" / "rows" / str(item["query_id"]) / "manifest.json"
            )
            for item in prepared.examples
            if (
                prepared.path / "evidence" / "rows" / str(item["query_id"]) / "manifest.json"
            ).exists()
        ],
    }
    if aggregate_path.exists():
        current_aggregate = _read_json(aggregate_path)
        if current_aggregate != aggregate and not (
            _is_evidence_placeholder(current_aggregate, prepared.run_id)
            or _evidence_aggregate_can_extend(current_aggregate, aggregate)
            or not any(
                isinstance(row, Mapping) and not row.get("timeline_request_sha256")
                for row in current_aggregate.get("rows", [])
            )
        ):
            raise ValueError("refusing to overwrite mismatched evidence manifest")
        if current_aggregate != aggregate:
            _write_json(aggregate_path, aggregate, overwrite=True)
    else:
        _write_json(aggregate_path, aggregate, overwrite=False)
    return plan, {
        "media_path": model_media_path,
        "audio_path": audio_path,
        "media_sha256": model_media_sha,
        "source_media_sha256": manifest["source_media_sha256"],
        "audio_sha256": manifest["audio_sha256"],
        "provider_video_path": manifest.get("provider_video_path"),
        "provider_video_sha256": manifest.get("provider_video_sha256"),
        "model_media_path": model_media_path,
        "model_audio_path": audio_path,
        "model_plan": plan,
    }


def _bind_evidence_manifest(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    timeline_checkpoint: Mapping[str, Any],
) -> None:
    """Bind the immutable evidence selection to its successful timeline call."""

    if timeline_checkpoint.get("status") != "completed":
        return
    row_path = prepared.path / "evidence" / "rows" / str(example["query_id"]) / "manifest.json"
    manifest = _read_json(row_path)
    binding = {
        "resolved_manifest_sha256": prepared.resolved_manifest_sha256,
        "timeline_request_sha256": timeline_checkpoint.get("request_sha256"),
        "timeline_response_sha256": timeline_checkpoint.get("response_payload_sha256"),
    }
    if any(value is None for value in binding.values()):
        raise ValueError("timeline checkpoint is missing evidence binding hashes")
    for key, value in binding.items():
        if key in manifest and manifest[key] != value:
            raise ValueError("evidence manifest timeline binding does not match")
    manifest.update(binding)
    _write_json(row_path, manifest, overwrite=True)
    aggregate_path = prepared.path / "evidence" / "manifest.json"
    aggregate = _read_json(aggregate_path)
    if isinstance(aggregate.get("rows"), list):
        rows = [
            manifest if int(row.get("query_id", -1)) == int(example["query_id"]) else row
            for row in aggregate["rows"]
        ]
        aggregate["rows"] = rows
    else:
        aggregate = manifest
    _write_json(aggregate_path, aggregate, overwrite=True)


def _fixture_row(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    evidence_files: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(example)
    row.update(
        {
            "run_id": prepared.run_id,
            "media_path": evidence_files["model_media_path"],
            "audio_path": evidence_files["model_audio_path"],
            "media_sha256": evidence_files["media_sha256"],
            "audio_sha256": evidence_files["audio_sha256"],
        }
    )
    if evidence_files.get("provider_video_path") is not None:
        row["provider_video_sha256"] = evidence_files["provider_video_sha256"]
        row["source_media_sha256"] = evidence_files["source_media_sha256"]
    return row


def _bind_provider_video_request(
    request: Mapping[str, Any], provider_video: Mapping[str, str] | None
) -> dict[str, Any]:
    if provider_video is None:
        return dict(request)
    bound = dict(request)
    bound.update(
        {
            "media_path": provider_video["path"],
            "media_sha256": provider_video["sha256"],
            "source_media_sha256": provider_video["source_sha256"],
            "provider_video_sha256": provider_video["sha256"],
        }
    )
    audit_label_free_payload(bound)
    return bound


def _provider_request_without_local_paths(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value for key, value in request.items() if key != "provider_video_path"
    }


def _candidate_model(record: Mapping[str, Any]) -> CandidateRecord:
    provider = str(record.get("provider", "fixture"))
    support_span_ids = tuple(str(item) for item in record.get("support_span_ids", ("fixture",)))
    runtime_provenance = record.get("runtime_provenance")
    if not isinstance(runtime_provenance, Mapping):
        runtime_provenance = {
            "model_id": MODEL_ID,
            "provider": provider,
            "runtime": str(record.get("runtime", provider)),
            "status": "passed",
            "request_sha256": str(record["request_sha256"]),
            "response_sha256": str(record["response_sha256"]),
        }
    else:
        runtime_provenance = dict(runtime_provenance)
        if "provider" in record:
            runtime_provenance["provider"] = str(record["provider"])
        if "runtime" in record:
            runtime_provenance["runtime"] = str(record["runtime"])
    return CandidateRecord(
        video_id=str(record["video_id"]),
        pass_id=str(record["pass_id"]),
        answer=str(record["answer"]),
        elapsed_seconds=float(record["elapsed_seconds"]),
        confidence=float(record["confidence"]),
        support_span_ids=support_span_ids,
        request_sha256=str(record["request_sha256"]),
        runtime_provenance=runtime_provenance,
    )


def _candidate_record_from_prediction(candidate: Mapping[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        video_id=str(candidate["video_id"]),
        pass_id=str(candidate["pass_id"]),
        answer=str(candidate["answer"]),
        elapsed_seconds=float(candidate["elapsed_seconds"]),
        confidence=float(candidate["confidence"]),
        support_span_ids=tuple(str(item) for item in candidate["support_span_ids"]),
        request_sha256=str(candidate["request_sha256"]),
        runtime_provenance=candidate["runtime_provenance"],
    )


def _valid_ranked_passes(response: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    if not isinstance(response, Mapping):
        return None
    ranked = response.get("ranked_pass_ids")
    selected = response.get("selected_pass_id")
    if (
        not isinstance(ranked, list)
        or tuple(ranked) not in (tuple(PASS_IDS), tuple(FALLBACK_PASS_PRECEDENCE))
        or selected != ranked[0]
    ):
        return None
    return tuple(ranked)


def _valid_baseline_answers(values: Sequence[str]) -> bool:
    if len(values) != len(PASS_IDS):
        return False
    normalized = [canonical_candidate_normalize(value) for value in values]
    return all(normalized) and len(set(normalized)) == len(normalized)


_FREEZE_FORBIDDEN_TERMS = ("reference", "annotation", "label", "score", "gold", "oracle")


def _audit_pre_freeze_artifacts(run_path: Path) -> None:
    """Audit local run artifacts without opening reference-like paths."""

    for artifact in sorted(run_path.rglob("*")):
        relative_parts = artifact.relative_to(run_path).parts
        lowered_parts = tuple(part.casefold() for part in relative_parts)
        if any(
            any(term in part for term in _FREEZE_FORBIDDEN_TERMS)
            for part in lowered_parts
        ):
            raise ValueError("pre-freeze artifact audit rejected a reference-like path")
        if not artifact.is_file():
            continue
        if artifact.suffix not in {".json", ".jsonl"}:
            continue
        try:
            value: Any
            if artifact.suffix == ".json":
                value = _read_json(artifact)
            else:
                value = _read_jsonl(artifact)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"pre-freeze artifact is not valid JSON: {artifact}") from error
        def without_ordinary_text(current: Any) -> Any:
            if isinstance(current, Mapping):
                return {
                    key: without_ordinary_text(nested)
                    for key, nested in current.items()
                    if str(key).casefold()
                    not in {
                        "answer",
                        "question",
                        "description",
                        "text",
                        "labels_loaded",
                        "request_bytes",
                        "response_bytes",
                    }
                }
            if isinstance(current, list):
                return [without_ordinary_text(nested) for nested in current]
            return current

        audit_label_free_payload(without_ordinary_text(value))


def _latest_attempts(run_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows = _read_jsonl(run_path / "attempts.jsonl")
    return {
        _attempt_key(row["request"]): row
        for row in rows
        if isinstance(row.get("request"), Mapping)
    }


def _validate_timeline_artifacts(prepared: PreparedRun) -> None:
    rows = _read_jsonl(prepared.path / "timeline.jsonl")
    latest = _latest_attempts(prepared.path)
    for example in prepared.examples:
        query_id = int(example["query_id"])
        completed = [
            row
            for row in rows
            if int(row.get("query_id", -1)) == query_id
            and row.get("status") == "completed"
        ]
        if len(completed) != 1:
            raise ValueError("freeze requires one successful timeline artifact per row")
        checkpoint = latest.get((query_id, "timeline", ""))
        if checkpoint is None or checkpoint.get("status") != "completed":
            raise ValueError("freeze requires a completed timeline attempt")
        row = completed[0]
        if row.get("request_sha256") != checkpoint.get("request_sha256"):
            raise ValueError("timeline artifact does not match completed attempt")
        if row.get("response_payload_sha256") != checkpoint.get("response_payload_sha256"):
            raise ValueError("timeline artifact response hash is invalid")
        _validate_timeline_response_for_duration(
            checkpoint.get("response"),
            float(
                example.get(
                    "duration_seconds",
                    example.get("media", {}).get("duration_seconds", 10.0),
                )
            ),
        )


def _request_frame_pairs(request: Mapping[str, Any]) -> tuple[tuple[str, str], ...] | None:
    """Return declared frame path/hash pairs, or None for a request without frames."""

    payloads: list[Mapping[str, Any]] = [request]
    model_payload = request.get("model_payload")
    if isinstance(model_payload, Mapping):
        payloads.append(model_payload)
    for payload in payloads:
        if "frame_paths" not in payload:
            continue
        raw_paths = payload.get("frame_paths")
        raw_provenance = payload.get("frame_provenance")
        if not isinstance(raw_paths, list) or not isinstance(raw_provenance, list):
            raise ValueError("frame provenance is incomplete")
        if len(raw_paths) != len(raw_provenance):
            raise ValueError("frame provenance count does not match frame paths")
        pairs: list[tuple[str, str]] = []
        for path, provenance in zip(raw_paths, raw_provenance, strict=True):
            if not isinstance(path, str) or not isinstance(provenance, Mapping):
                raise ValueError("frame provenance is malformed")
            image_sha = provenance.get("image_sha256")
            if not isinstance(image_sha, str):
                raise ValueError("frame provenance image hash is missing")
            pairs.append((str(Path(path).resolve()), image_sha))
        return tuple(pairs)
    return None


def _validate_freeze_cross_checks(
    prepared: PreparedRun,
    predictions: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_candidate = {
        (int(row["query_id"]), str(row["pass_id"])): row for row in candidate_rows
    }
    latest = _latest_attempts(prepared.path)
    for example, prediction in zip(prepared.examples, predictions, strict=True):
        query_id = int(example["query_id"])
        row_manifest_path = (
            prepared.path / "evidence" / "rows" / str(query_id) / "manifest.json"
        )
        if not row_manifest_path.is_file():
            raise ValueError("freeze evidence manifest is missing")
        evidence = _read_json(row_manifest_path)
        if evidence.get("resolved_manifest_sha256") != prepared.resolved_manifest_sha256:
            raise ValueError("freeze evidence manifest binding is invalid")
        timeline_rows = [
            row
            for row in _read_jsonl(prepared.path / "timeline.jsonl")
            if int(row.get("query_id", -1)) == query_id
            and row.get("status") == "completed"
        ]
        if len(timeline_rows) != 1 or evidence.get("timeline_request_sha256") != timeline_rows[0].get(
            "request_sha256"
        ) or evidence.get("timeline_response_sha256") != timeline_rows[0].get(
            "response_payload_sha256"
        ):
            raise ValueError("freeze evidence is not bound to the successful timeline")
        if not prepared.fixture_mode and evidence.get("source_media_sha256") != example["source"]["sha256"]:
            raise ValueError("freeze evidence source hash does not match manifest")
        media_artifact = (
            Path(str(example["media_path"]))
            if not prepared.fixture_mode
            else prepared.path / "evidence" / "rows" / str(query_id) / "fixture-video.mp4"
        )
        if not _is_regular_file(media_artifact) or _hash_path(media_artifact) != evidence.get(
            "source_media_sha256"
        ):
            raise ValueError("freeze evidence media hash does not match artifact")
        audio_path = prepared.path / "evidence" / "rows" / str(query_id) / (
            "audio.wav" if not prepared.fixture_mode else "fixture-audio.wav"
        )
        if evidence.get("audio_sha256") is not None:
            if not _is_regular_file(audio_path) or _hash_path(audio_path) != evidence["audio_sha256"]:
                raise ValueError("freeze evidence audio hash does not match artifact")
        frames = evidence.get("frames")
        global_count = evidence.get("global_frame_count")
        local_count = evidence.get("local_frame_count")
        if (
            not isinstance(frames, list)
            or global_count != 8
            or not isinstance(local_count, int)
            or isinstance(local_count, bool)
            or not 0 <= local_count <= 24
            or len(frames) != global_count + local_count
        ):
            raise ValueError("freeze evidence frame counts are invalid")
        support_ids = {
            str(span.get("span_id"))
            for span in evidence.get("support_spans", [])
            if isinstance(span, Mapping) and isinstance(span.get("span_id"), str)
        }
        identities: set[tuple[int, int]] = set()
        expected_frame_pairs: dict[str, str] = {}
        for index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise ValueError("freeze evidence frame entry is invalid")
            identity = frame.get("decoded_identity")
            if not isinstance(identity, Mapping):
                raise ValueError("freeze evidence decoded identity is missing")
            stream_index = identity.get("stream_index")
            frame_index = identity.get("frame_index")
            if not isinstance(stream_index, int) or isinstance(stream_index, bool) or not isinstance(
                frame_index, int
            ) or isinstance(frame_index, bool):
                raise ValueError("freeze evidence decoded identity is invalid")
            decoded = (stream_index, frame_index)
            if decoded in identities:
                raise ValueError("freeze evidence reuses a decoded frame identity")
            identities.add(decoded)
            expected_slot = "global" if index < global_count else "local"
            if frame.get("slot_kind") != expected_slot:
                raise ValueError("freeze evidence slot semantics are invalid")
            support_id = frame.get("support_span_id")
            if expected_slot == "global" and support_id is not None:
                raise ValueError("global evidence frame has a support span")
            if expected_slot == "local" and support_id not in support_ids:
                raise ValueError("local evidence frame has an undeclared support span")
            frame_path = Path(str(frame["frame_path"]))
            if not _is_regular_file(frame_path) or _hash_path(frame_path) != frame.get("image_sha256"):
                raise ValueError("freeze evidence frame hash does not match artifact")
            expected_frame_pairs[str(frame_path.resolve())] = str(frame.get("image_sha256"))
        expected_frame_hashes = set(expected_frame_pairs.values())
        frame_hashes = evidence.get("materialized_frame_hashes")
        if not isinstance(frame_hashes, list) or any(
            pair[1] not in frame_hashes for pair in expected_frame_pairs.items()
        ):
            raise ValueError("freeze evidence frame manifest is inconsistent")
        for candidate in prediction["candidates"]:
            key = (query_id, str(candidate["pass_id"]))
            stored = by_candidate.get(key)
            if stored is None:
                raise ValueError("freeze candidate is missing from candidates.jsonl")
            try:
                persisted_candidate = _candidate_model(stored)
                ranked_candidate = _candidate_record_from_prediction(candidate)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("freeze candidate provenance is invalid") from error
            if persisted_candidate.to_dict() != ranked_candidate.to_dict():
                raise ValueError("freeze candidate provenance does not match candidates.jsonl")
            if stored.get("answer") != candidate.get("answer"):
                raise ValueError("freeze candidate answer does not match candidates.jsonl")
            request = stored.get("request")
            if not isinstance(request, Mapping):
                raise ValueError("freeze candidate request provenance is missing")
            if canonical_request_sha256(request) != stored.get("request_sha256"):
                raise ValueError("freeze candidate request hash is invalid")
            if stored.get("request_sha256") != candidate.get("request_sha256"):
                raise ValueError("freeze candidate request hash does not match prediction")
            runtime_provenance = candidate.get("runtime_provenance")
            if not isinstance(runtime_provenance, Mapping) or runtime_provenance.get(
                "response_sha256"
            ) != stored.get("response_sha256"):
                raise ValueError("freeze candidate response hash does not match prediction")
            response = stored.get("response")
            if not isinstance(response, Mapping) or _sha256_json(response) != stored.get("response_sha256"):
                raise ValueError("freeze candidate response hash is invalid")
            if str(request.get("stage")) == "baseline-fallback":
                continue
            attempt = latest.get(_attempt_key(request))
            if attempt is None or attempt.get("status") != "completed":
                raise ValueError("freeze candidate lacks a completed terminal attempt")
            if attempt.get("response_payload_sha256") != stored.get("response_sha256"):
                raise ValueError("freeze candidate response hash does not match attempt")
            frame_pairs = _request_frame_pairs(request)
            if frame_pairs is not None and any(
                image_sha not in expected_frame_hashes
                or not _is_regular_file(Path(path))
                or _hash_path(Path(path)) != image_sha
                for path, image_sha in frame_pairs
            ):
                raise ValueError("freeze candidate frame provenance does not match evidence")
        verifier_attempts = [
            row
            for row in _read_jsonl(prepared.path / "attempts.jsonl")
            if int(row.get("query_id", -1)) == query_id
            and row.get("stage") == "verifier"
            and row.get("status") == "completed"
        ]
        if len(verifier_attempts) != 1:
            raise ValueError("freeze verifier attempt is missing or ambiguous")
        verifier_pairs = _request_frame_pairs(verifier_attempts[0].get("request", {}))
        if verifier_pairs is None or {
            image_sha for _, image_sha in verifier_pairs
        } != expected_frame_hashes or any(
            not _is_regular_file(Path(path)) or _hash_path(Path(path)) != image_sha
            for path, image_sha in verifier_pairs
        ):
            raise ValueError("freeze verifier frame provenance does not match evidence")


def _validate_freeze_inputs(prepared: PreparedRun, payload: Mapping[str, Any]) -> None:
    _validate_prepared_sources(prepared)
    if payload.get("run_id") != prepared.run_id or payload.get("model_id") != MODEL_ID:
        raise ValueError("prediction artifact run/model binding is invalid")
    if payload.get("resolved_manifest_sha256") != prepared.resolved_manifest_sha256:
        raise ValueError("prediction artifact manifest binding is invalid")
    if payload.get("panel_sha256") != prepared.panel_sha256:
        raise ValueError("prediction artifact panel binding is invalid")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("prediction artifact protocol binding is invalid")
    predictions = payload.get("predictions")
    if payload.get("candidate_shortfall", 0) != 0:
        raise ValueError("cannot freeze a candidate shortfall")
    terminal_path = prepared.path / "terminal-status.json"
    terminal = _read_json(terminal_path)
    if type(terminal.get("labels_loaded")) is not bool or terminal.get("labels_loaded") is not False:
        raise ValueError("labels_loaded must be the boolean false")
    if terminal.get("status") in {"partial", "prepared", "in_flight_unknown", "candidate_shortfall"}:
        raise ValueError("cannot freeze an incomplete or unresolved run")
    if not isinstance(predictions, list) or len(predictions) != len(prepared.examples):
        raise ValueError("freeze requires one prediction for every ordered manifest row")
    candidate_rows = _read_jsonl(prepared.path / "candidates.jsonl")
    expected_candidate_keys = {
        (int(example["query_id"]), pass_id)
        for example in prepared.examples
        for pass_id in PASS_IDS
    }
    actual_candidate_keys = {
        (int(row["query_id"]), str(row["pass_id"])) for row in candidate_rows
    }
    if len(candidate_rows) != len(expected_candidate_keys) or actual_candidate_keys != expected_candidate_keys:
        raise ValueError("freeze candidates.jsonl does not contain the exact ordered ten passes")
    _validate_timeline_artifacts(prepared)
    _validate_freeze_cross_checks(prepared, predictions, candidate_rows)
    expected = [(int(row["query_id"]), str(row["video_id"])) for row in prepared.examples]
    actual = [(row.get("query_id"), row.get("video_id")) for row in predictions]
    if actual != expected:
        raise ValueError("freeze predictions must match the ordered manifest")
    for prediction in predictions:
        candidates = prediction.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(PASS_IDS):
            raise ValueError("freeze requires exactly ten candidates per row")
        pass_ids = [candidate.get("pass_id") for candidate in candidates]
        answers = [candidate.get("answer") for candidate in candidates]
        if pass_ids != list(PASS_IDS) or len(set(pass_ids)) != len(PASS_IDS):
            raise ValueError("freeze requires exact ordered private pass IDs")
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError("freeze requires ten non-empty candidate answers")
        if len({canonical_candidate_normalize(answer) for answer in answers}) != len(PASS_IDS):
            raise ValueError("freeze requires ten unique candidate answers")
        ranked = prediction.get("ranked_pass_ids")
        if not isinstance(ranked, list) or len(ranked) != len(PASS_IDS) or set(ranked) != set(PASS_IDS):
            raise ValueError("freeze requires a complete ranked pass permutation")
    latest = _latest_attempts(prepared.path)
    if any(
        record.get("stage") == "verifier" and record.get("status") != "completed"
        for record in latest.values()
    ):
        raise ValueError("freeze requires no unresolved verifier failure")
    _audit_pre_freeze_artifacts(prepared.path)


def _baseline_candidates(
    prepared: PreparedRun,
    example: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pass_id, answer in zip(PASS_IDS, prepared.baseline_ranked_list, strict=True):
        request = _bind_request(
            prepared,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "baseline-fallback",
                "run_id": prepared.run_id,
                "query_id": example["query_id"],
                "video_id": example["video_id"],
                "pass_id": pass_id,
                "question": example["question"],
            },
        )
        records.append(
            _candidate_record(
                request,
                {"answer": answer, "confidence": 0.0, "status": "passed"},
                0.0,
                candidate_pass_id=pass_id,
                provider="fixture" if prepared.fixture_mode else "baseline",
                support_span_ids=("baseline-fallback",),
            )
        )
    return records


def _validate_answer_response(response: Any) -> None:
    if not isinstance(response, Mapping) or not isinstance(response.get("answer"), str):
        raise ValueError("answer response schema is invalid")
    if not response["answer"].strip():
        raise ValueError("answer response is empty")


def _validate_timeline_response(response: Any) -> None:
    _validate_timeline_response_for_duration(response, None)


def _validate_timeline_response_for_duration(
    response: Any, expected_duration: float | None
) -> None:
    """Apply the strict Task 4 response contract before caching a timeline."""

    if not isinstance(response, Mapping):
        raise ValueError("timeline response schema is invalid")
    if response.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("timeline response schema is invalid")
    duration = response.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 1.0
    ):
        raise ValueError("timeline response duration is invalid")
    if expected_duration is not None and not math.isclose(
        float(duration), float(expected_duration), rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("timeline response duration does not match source")
    if not isinstance(response.get("events"), list) or not isinstance(
        response.get("support_spans"), list
    ):
        raise ValueError("timeline response schema is invalid")
    # validate_timeline delegates nested object validation to the shared Task 4
    # TimelineSpan contract: IDs, finite bounded times, modalities, and text.
    validate_timeline(response, float(expected_duration or duration), allow_fallback=False)


def _validate_verifier_response(response: Any) -> None:
    if _valid_ranked_passes(response) is None:
        raise ValueError("verifier response schema is invalid")


def _record_sort_key(record: Mapping[str, Any]) -> tuple[int, int]:
    pass_id = str(record.get("pass_id", ""))
    return int(record["query_id"]), PASS_IDS.index(pass_id) if pass_id in PASS_IDS else len(PASS_IDS)


def _reconstruct_timeline_artifact(
    path: Path, query_id: int, checkpoint: Mapping[str, Any], timeline: ValidatedTimeline
) -> None:
    if checkpoint.get("status") != "completed":
        return
    rows = _read_jsonl(path)
    existing = [row for row in rows if int(row.get("query_id", -1)) == query_id]
    completed = [row for row in existing if row.get("status") == "completed"]
    if completed:
        if any(
            row.get("request_sha256") != checkpoint.get("request_sha256")
            for row in completed
        ):
            raise ValueError("timeline artifact does not match completed timeline checkpoint")
        return
    # Failed attempts remain in attempts.jsonl/failures.jsonl, but are not
    # authoritative timeline artifacts and must not block later recovery.
    _append_jsonl(
        path,
        {
            **checkpoint,
            "timeline_fallback_reason": timeline.fallback_reason,
        },
    )


def _build_ranked_predictions(
    prepared: PreparedRun,
    records: Sequence[Mapping[str, Any]],
    ranking_by_query: Mapping[int, tuple[str, ...]],
    source_by_query: Mapping[int, str],
) -> list[dict[str, Any]]:
    by_key = {(int(row["query_id"]), str(row["pass_id"])): row for row in records}
    predictions: list[dict[str, Any]] = []
    for example in prepared.examples:
        query_id = int(example["query_id"])
        row_records = [by_key.get((query_id, pass_id)) for pass_id in PASS_IDS]
        if any(record is None for record in row_records):
            continue
        candidates = tuple(_candidate_model(record) for record in row_records if record is not None)
        if len(candidates) != len(PASS_IDS):
            continue
        prediction = RankedPrediction(
            query_id=query_id,
            video_id=str(example["video_id"]),
            candidates=candidates,
            ranked_pass_ids=ranking_by_query.get(query_id, PASS_IDS),
            ranking_source=source_by_query.get(query_id, "fixture-deterministic"),
            fallback_status=None,
        )
        predictions.append(prediction.to_dict())
    return predictions


def _run_prepared_locked(
    prepared: PreparedRun | Path | str,
    *,
    transport: Transport,
    limit: int | None = None,
) -> RunSummary:
    """Run the label-free fixture state machine with durable checkpoints."""

    prepared_run = (
        prepared if isinstance(prepared, PreparedRun) else PreparedRun.load(prepared)
    )
    _validate_prepared_sources(prepared_run)
    if (prepared_run.path / "freeze.json").exists():
        raise FileExistsError("cannot resume a frozen run")
    if limit is not None and (isinstance(limit, bool) or limit < 0):
        raise ValueError("limit must be non-negative")
    selected = prepared_run.examples if limit is None else prepared_run.examples[:limit]
    attempt_rows = _read_jsonl(prepared_run.path / "attempts.jsonl")
    attempts = {_attempt_key(row["request"]): row for row in attempt_rows}
    failure_rows = _read_jsonl(prepared_run.path / "failures.jsonl")
    failures = {
        _attempt_key(row["request"]): row
        for row in failure_rows
        if isinstance(row.get("request"), Mapping)
    }
    candidate_rows = {
        (int(row["query_id"]), str(row["pass_id"])): row
        for row in _read_jsonl(prepared_run.path / "candidates.jsonl")
    }
    ranking_by_query: dict[int, tuple[str, ...]] = {}
    source_by_query: dict[int, str] = {}
    shortfall_by_query: dict[int, int] = {}
    reused_rows = 0
    generated_rows = 0
    completed_rows = 0

    for example in selected:
        query_id = int(example["query_id"])
        row_reused = True
        provider_name = _runtime_provider(
            transport, fixture_mode=prepared_run.fixture_mode
        )
        provider_video = (
            _live_provider_video(prepared_run, example)
            if not prepared_run.fixture_mode and provider_name == "openrouter"
            else None
        )
        timeline_request = _timeline_request(
            prepared_run,
            example,
            provider_video=provider_video,
            provider_name=provider_name,
        )
        duration = float(
            example.get(
                "duration_seconds",
                example.get("media", {}).get("duration_seconds", 10.0),
            )
        )
        timeline_response, timeline_checkpoint, reused = _run_attempt(
            timeline_request,
            transport,
            attempts,
            failures,
            attempts_path=prepared_run.path / "attempts.jsonl",
            failures_path=prepared_run.path / "failures.jsonl",
            validator=lambda response: _validate_timeline_response_for_duration(
                response, duration
            ),
            provider_request=_provider_request_without_local_paths(timeline_request),
        )
        row_reused &= reused
        if timeline_checkpoint.get("status") == "completed":
            timeline_payload = timeline_response if isinstance(timeline_response, Mapping) else {}
            timeline = validate_timeline(timeline_payload, duration, allow_fallback=False)
        else:
            timeline = ValidatedTimeline(
                duration_seconds=duration,
                events=(),
                support_spans=(),
                fallback_reason="invalid_timeline",
            )
        if prepared_run.fixture_mode:
            evidence_plan, evidence_files = _fixture_evidence(
                prepared_run, example, timeline
            )
        else:
            evidence_plan, evidence_files = _live_evidence(
                prepared_run,
                example,
                timeline,
                provider_video=provider_video,
            )
        _reconstruct_timeline_artifact(
            prepared_run.path / "timeline.jsonl",
            query_id,
            timeline_checkpoint,
            timeline,
        )
        _bind_evidence_manifest(prepared_run, example, timeline_checkpoint)
        row = _fixture_row(prepared_run, example, evidence_files)
        request_plan = evidence_files.get("model_plan", evidence_plan)
        candidate_provider = provider_name
        candidate_support_spans = tuple(
            span.span_id for span in request_plan.support_spans
        )
        answer_requests = build_answer_requests(row, timeline, request_plan, candidates=())
        primary_candidates: list[dict[str, Any]] = []
        for pass_id in PASS_IDS[:9]:
            request = dict(answer_requests[pass_id].to_dict())
            request["stage"] = "answer"
            request["query_id"] = query_id
            request = _bind_request(prepared_run, request)
            request = _bind_provider_video_request(request, provider_video)
            response, checkpoint, pass_reused = _run_attempt(
                request,
                transport,
                attempts,
                failures,
                attempts_path=prepared_run.path / "attempts.jsonl",
                failures_path=prepared_run.path / "failures.jsonl",
                provider_request=answer_requests[pass_id],
                validator=_validate_answer_response,
            )
            row_reused &= pass_reused
            if isinstance(response, Mapping) and isinstance(response.get("answer"), str):
                primary_candidates.append(
                    _candidate_record(
                        request,
                        response,
                        float(checkpoint["elapsed_seconds"]),
                        checkpoint=checkpoint,
                        provider=candidate_provider,
                        support_span_ids=candidate_support_spans,
                    )
                )
        candidates = [_candidate_model(record) for record in primary_candidates]
        synthesis_requests = build_answer_requests(
            row, timeline, request_plan, candidates=candidates
        )
        pass_id = PASS_IDS[9]
        request = dict(synthesis_requests[pass_id].to_dict())
        request["stage"] = "answer"
        request["query_id"] = query_id
        request = _bind_request(prepared_run, request)
        request = _bind_provider_video_request(request, provider_video)
        response, checkpoint, pass_reused = _run_attempt(
            request,
            transport,
            attempts,
            failures,
            attempts_path=prepared_run.path / "attempts.jsonl",
            failures_path=prepared_run.path / "failures.jsonl",
            provider_request=synthesis_requests[pass_id],
            validator=_validate_answer_response,
        )
        row_reused &= pass_reused
        if isinstance(response, Mapping) and isinstance(response.get("answer"), str):
            primary_candidates.append(
                _candidate_record(
                    request,
                    response,
                    float(checkpoint["elapsed_seconds"]),
                    checkpoint=checkpoint,
                    provider=candidate_provider,
                    support_span_ids=candidate_support_spans,
                )
            )
        candidates = [_candidate_model(record) for record in primary_candidates]
        candidate_records_by_pass = {
            str(record["pass_id"]): record for record in primary_candidates
        }
        unique_candidates = list(deduplicate_candidates(candidates))
        for attempt in (1, 2):
            if len(unique_candidates) >= len(PASS_IDS):
                break
            backfill_request_object = build_backfill_request(
                row, request_plan, unique_candidates, attempt=attempt
            )
            request = dict(backfill_request_object.to_dict())
            request["stage"] = "backfill"
            request["attempt"] = attempt
            request["query_id"] = query_id
            request = _bind_request(prepared_run, request)
            request = _bind_provider_video_request(request, provider_video)
            response, checkpoint, backfill_reused = _run_attempt(
                request,
                transport,
                attempts,
                failures,
                attempts_path=prepared_run.path / "attempts.jsonl",
                failures_path=prepared_run.path / "failures.jsonl",
                provider_request=backfill_request_object,
                validator=_validate_answer_response,
            )
            row_reused &= backfill_reused
            if isinstance(response, Mapping) and isinstance(response.get("answer"), str):
                backfill_record = _candidate_record(
                    request,
                    response,
                    float(checkpoint["elapsed_seconds"]),
                    checkpoint=checkpoint,
                    provider=candidate_provider,
                    support_span_ids=candidate_support_spans,
                )
                backfill_probe = dict(backfill_record)
                backfill_probe["pass_id"] = PASS_IDS[0]
                backfill_candidate = _candidate_model(backfill_probe)
                candidate_count_before = len(unique_candidates)
                candidate_count_after = len(
                    deduplicate_candidates(candidates + [backfill_candidate])
                )
                if candidate_count_after > candidate_count_before:
                    kept_pass_ids = {
                        candidate.pass_id
                        for candidate in deduplicate_candidates(candidates)
                    }
                    missing_passes = [
                        pass_id for pass_id in PASS_IDS if pass_id not in candidate_records_by_pass
                    ]
                    duplicate_passes = [
                        candidate.pass_id
                        for candidate in candidates
                        if candidate.pass_id not in kept_pass_ids
                    ]
                    replacement_pass = (
                        missing_passes[0]
                        if missing_passes
                        else max(duplicate_passes, key=PASS_IDS.index)
                        if duplicate_passes
                        else None
                    )
                    if replacement_pass is not None:
                        backfill_record["pass_id"] = replacement_pass
                        replacement = _candidate_model(backfill_record)
                        candidate_records_by_pass[replacement_pass] = backfill_record
                        candidates = [
                            replacement if candidate.pass_id == replacement_pass else candidate
                            for candidate in candidates
                        ]
                        if replacement_pass not in {
                            candidate.pass_id for candidate in candidates
                        }:
                            candidates.append(replacement)
                    unique_candidates = list(deduplicate_candidates(candidates))
        primary_candidates = []
        for candidate in candidates:
            record = candidate.to_dict()
            record["query_id"] = query_id
            source_record = candidate_records_by_pass.get(str(candidate.pass_id), {})
            record["request"] = dict(source_record.get("request", {}))
            record["response"] = dict(source_record.get("response", {}))
            record["request_sha256"] = str(
                source_record.get("request_sha256", record["request_sha256"])
            )
            record["response_sha256"] = str(
                source_record.get(
                    "response_sha256", candidate.runtime_provenance["response_sha256"]
                )
            )
            record["provider"] = str(
                source_record.get("provider", candidate.runtime_provenance["provider"])
            )
            record["runtime"] = str(
                source_record.get("runtime", candidate.runtime_provenance["runtime"])
            )
            record["support_span_ids"] = list(candidate.support_span_ids)
            source_record = candidate_records_by_pass.get(str(candidate.pass_id), {})
            record["usage"] = dict(source_record.get("usage", _usage()))
            record["uploaded_file_deletion_status"] = dict(
                source_record.get("uploaded_file_deletion_status", {})
            )
            record["failure_code"] = source_record.get("failure_code")
            primary_candidates.append(record)
        for record in primary_candidates:
            candidate_rows[(query_id, str(record["pass_id"]))] = record

        shortfall = len(PASS_IDS) - len(unique_candidates)
        shortfall_by_query[query_id] = shortfall
        if shortfall > 0:
            if _valid_baseline_answers(prepared_run.baseline_ranked_list):
                primary_candidates = _baseline_candidates(prepared_run, example)
                candidates = [_candidate_model(record) for record in primary_candidates]
                unique_candidates = list(deduplicate_candidates(candidates))
                for record in primary_candidates:
                    candidate_rows[(query_id, str(record["pass_id"]))] = record
                shortfall = len(PASS_IDS) - len(unique_candidates)
                shortfall_by_query[query_id] = shortfall
                ranking_by_query[query_id] = PASS_IDS
                source_by_query[query_id] = "configured-baseline-ranked-list"
            else:
                ranking_by_query[query_id] = PASS_IDS
                source_by_query[query_id] = "configured-baseline-ranked-list"
        else:
            try:
                verifier_request = build_verifier_request(
                    row, timeline, request_plan, candidates, seed=query_id
                )
                verifier_payload = dict(verifier_request.to_model_payload())
                verifier_payload["stage"] = "verifier"
                verifier_payload["query_id"] = query_id
                verifier_payload["request_sha256"] = verifier_request.request_sha256
                verifier_payload = _bind_request(prepared_run, verifier_payload)
                verifier_payload = _bind_provider_video_request(
                    verifier_payload, provider_video
                )
                verifier_response, verifier_checkpoint, verifier_reused = _run_attempt(
                    verifier_payload,
                    transport,
                    attempts,
                    failures,
                    attempts_path=prepared_run.path / "attempts.jsonl",
                    failures_path=prepared_run.path / "failures.jsonl",
                    provider_request=verifier_request,
                    validator=_validate_verifier_response,
                )
                row_reused &= verifier_reused
                ranked = _valid_ranked_passes(verifier_response)
                if ranked is None:
                    ranking_by_query[query_id] = tuple(FALLBACK_PASS_PRECEDENCE)
                    source_by_query[query_id] = "deterministic-fallback"
                else:
                    ranking_by_query[query_id] = ranked
                    source_by_query[query_id] = "verifier"
            except (TypeError, ValueError):
                ranking_by_query[query_id] = tuple(FALLBACK_PASS_PRECEDENCE)
                source_by_query[query_id] = "deterministic-fallback"
        if shortfall == 0:
            completed_rows += 1
        if not row_reused:
            generated_rows += 1
        else:
            reused_rows += 1

    ordered_records = sorted(candidate_rows.values(), key=_record_sort_key)
    _write_jsonl(prepared_run.path / "candidates.jsonl", ordered_records)
    predictions = _build_ranked_predictions(
        prepared_run, ordered_records, ranking_by_query, source_by_query
    )
    candidate_shortfall = max(shortfall_by_query.values(), default=0)
    prediction_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared_run.run_id,
        "model_id": prepared_run.model_id,
        "resolved_manifest_sha256": prepared_run.resolved_manifest_sha256,
        "panel_sha256": prepared_run.panel_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "prediction_count": len(predictions),
        "predictions": predictions,
        "candidate_shortfall": candidate_shortfall,
    }
    if candidate_shortfall:
        prediction_payload.update(
            {
                "ranking_source": "configured-baseline-ranked-list",
                "baseline_ranked_list": list(prepared_run.baseline_ranked_list),
            }
        )
    _write_json(
        prepared_run.path / "ranked-predictions.json", prediction_payload, overwrite=True
    )
    _write_json(
        prepared_run.path / "terminal-status.json",
        {
            "status": (
                "candidate_shortfall"
                if candidate_shortfall and len(selected) == len(prepared_run.examples)
                else "inferred"
                if len(selected) == len(prepared_run.examples)
                and completed_rows == len(prepared_run.examples)
                else "partial"
            ),
            "prepared_rows": len(prepared_run.examples),
            "inferred_rows": completed_rows,
            "completed_rows": completed_rows,
            "reused_rows": reused_rows,
            "generated_rows": generated_rows,
            "candidate_shortfall": candidate_shortfall,
            "labels_loaded": False,
            "frozen": False,
        },
        overwrite=True,
    )
    return RunSummary(
        prepared_rows=len(prepared_run.examples),
        attempted_rows=len(selected),
        completed_rows=completed_rows,
        reused_rows=reused_rows,
        generated_rows=generated_rows,
        failed_rows=len(failures),
        inferred_rows=completed_rows,
        candidate_shortfall=candidate_shortfall,
    )


def _acquire_run_lock(prepared: PreparedRun) -> Path:
    lock_path = prepared.path / "run.lock"
    payload = _canonical_bytes({"run_id": prepared.run_id, "model_id": prepared.model_id}) + b"\n"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as error:
        raise ValueError("stale run lock; recover by starting a new run ID") from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path


def run_prepared(
    prepared: PreparedRun | Path | str,
    *,
    transport: Transport,
    limit: int | None = None,
) -> RunSummary:
    prepared_run = prepared if isinstance(prepared, PreparedRun) else PreparedRun.load(prepared)
    _validate_prepared_sources(prepared_run)
    lock_path = _acquire_run_lock(prepared_run)
    clean = False
    try:
        result = _run_prepared_locked(
            prepared_run,
            transport=transport,
            limit=limit,
        )
        clean = True
        return result
    finally:
        if clean:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def status(prepared: PreparedRun | Path | str) -> dict[str, Any]:
    """Read and summarize local artifacts without constructing a transport."""

    prepared_run = (
        prepared if isinstance(prepared, PreparedRun) else PreparedRun.load(prepared)
    )
    candidates = _read_jsonl(prepared_run.path / "candidates.jsonl")
    failures = _read_jsonl(prepared_run.path / "failures.jsonl")
    terminal_path = prepared_run.path / "terminal-status.json"
    terminal = _read_json(terminal_path) if terminal_path.exists() else {}
    return {
        "run_id": prepared_run.run_id,
        "model_id": prepared_run.model_id,
        "prepared_rows": len(prepared_run.examples),
        "candidate_rows": len(candidates),
        "failure_rows": len(failures),
        "completed_rows": terminal.get("completed_rows", 0),
        "inferred_rows": terminal.get("inferred_rows", terminal.get("completed_rows", 0)),
        "status": terminal.get("status", "prepared"),
        "frozen": (prepared_run.path / "freeze.json").is_file(),
        "labels_loaded": False,
    }


def freeze(prepared: PreparedRun | Path | str) -> dict[str, Any]:
    prepared_run = (
        prepared if isinstance(prepared, PreparedRun) else PreparedRun.load(prepared)
    )
    prediction_path = prepared_run.path / "ranked-predictions.json"
    payload = _read_json(prediction_path)
    _validate_freeze_inputs(prepared_run, payload)
    bindings = {
        "run_id": prepared_run.run_id,
        "model_id": prepared_run.model_id,
        "resolved_manifest_sha256": prepared_run.resolved_manifest_sha256,
        "panel_sha256": prepared_run.panel_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "prediction_count": payload.get("prediction_count"),
    }
    sidecar_path = prepared_run.path / "freeze.json"
    if sidecar_path.exists():
        try:
            existing = _read_json(sidecar_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("invalid or partial freeze sidecar; refusing recovery") from error
        verify_prediction_freeze(prediction_path, existing)
        if existing.get("bindings") != bindings:
            raise ValueError("existing freeze sidecar bindings do not match current run")
        freeze_record = existing
    else:
        freeze_record = freeze_predictions(
            prediction_path,
            bindings=bindings,
            output_path=sidecar_path,
        )
    terminal_path = prepared_run.path / "terminal-status.json"
    terminal = _read_json(terminal_path) if terminal_path.exists() else {}
    terminal.update(
        {
            "status": "frozen",
            "prediction_count": payload.get("prediction_count"),
            "frozen": True,
            "labels_loaded": False,
        }
    )
    _write_json(terminal_path, terminal, overwrite=True)
    return freeze_record


def _google_sdk_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "unavailable"


def preflight_live(
    model_id: str = MODEL_ID, *, provider: str = "gemini"
) -> dict[str, Any]:
    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID}")
    if provider == "openrouter":
        return preflight_openrouter_transport(model_id=model_id)
    if provider != "gemini":
        raise ValueError("provider must be gemini or openrouter")
    preflight_google_transport()
    return {
        "model_id": model_id,
        "sdk_version": _google_sdk_version(),
        "authentication_configured": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True, type=Path)
    prepare_parser.add_argument("--dataset-root", type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prepared", required=True, type=Path)
    run_parser.add_argument(
        "--transport", choices=("fixture", "gemini", "openrouter"), required=True
    )
    run_parser.add_argument("--limit", type=int)

    preflight_parser = subparsers.add_parser("preflight-live")
    preflight_parser.add_argument("--model-id", default=MODEL_ID)
    preflight_parser.add_argument(
        "--provider", choices=("gemini", "openrouter"), default="gemini"
    )

    for name in ("freeze", "status"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--prepared", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.config, args.output, dataset_root=args.dataset_root)
    elif args.command == "run":
        prepared_run = PreparedRun.load(args.prepared)
        if args.transport == "fixture" and not prepared_run.fixture_mode:
            raise ValueError("fixture transport is only valid for fixture_mode runs")
        if args.transport in {"gemini", "openrouter"} and prepared_run.fixture_mode:
            raise ValueError(f"{args.transport} transport requires a non-fixture prepared run")
        transport: Any
        if args.transport == "fixture":
            transport = FixtureTransport()
        elif args.transport == "gemini":
            transport = GeminiSubmissionClient(
                create_google_transport(), model_id=prepared_run.model_id
            )
        else:
            transport = GeminiSubmissionClient(
                create_openrouter_transport(), model_id=prepared_run.model_id
            )
        run_prepared(
            prepared_run,
            transport=transport,
            limit=args.limit,
        )
    elif args.command == "freeze":
        freeze(args.prepared)
    elif args.command == "status":
        print(_canonical_bytes(status(args.prepared)).decode("utf-8"))
    elif args.command == "preflight-live":
        try:
            print(
                _canonical_bytes(
                    preflight_live(args.model_id, provider=args.provider)
                ).decode("utf-8")
            )
        except ProviderAuthenticationError as error:
            print(str(error) or AUTHENTICATION_ERROR_MESSAGE, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
