"""Small, deterministic phase-A VQA submission orchestrator.

This module intentionally implements only the label-free fixture path.  It is
safe to exercise in tests without media, credentials, or a provider SDK.
"""

import argparse
import hashlib
import json
import os
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
    validate_timeline,
)
from trec_video_2026.vqa_submission_gemini import (
    AUTHENTICATION_ERROR_MESSAGE,
    GeminiSubmissionClient,
    ProviderAuthenticationError,
    create_google_transport,
    preflight_google_transport,
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


def _write_bytes_once(path: Path, payload: bytes) -> str:
    """Materialize deterministic fixture bytes without overwriting them."""

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
        source = _safe_source_metadata(source_fingerprint(media_path))
        example = dict(raw)
        example.pop("media", None)
        example["media_path"] = str(media_path)
        example["source"] = source
        example["media"] = metadata
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
        source = _safe_source_metadata(source_fingerprint(media_path))
        declared = example.get("source")
        if not isinstance(declared, Mapping) or source.get("sha256") != declared.get("sha256"):
            raise ValueError("prepared source hash does not match current media")


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
    }


def _timeline_request(
    prepared: PreparedRun, example: Mapping[str, Any]
) -> dict[str, Any]:
    request = {
        "schema_version": SCHEMA_VERSION,
        "stage": "timeline",
        "run_id": prepared.run_id,
        "model_id": prepared.model_id,
        "query_id": example["query_id"],
        "video_id": example["video_id"],
        "question": example["question"],
        "duration_seconds": float(example.get("duration_seconds", 10.0)),
    }
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
        verifier = getattr(transport, "verify", transport.verifier)
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
    record: dict[str, Any] = {
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
        "status": "completed",
        "elapsed_seconds": 0.0,
        "usage": _usage(),
        "uploaded_file_deletion_status": {},
        "failure_code": None,
    }
    started = time.monotonic()
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
            "materialized_frame_hashes": frame_hashes,
            "timeline_fallback_reason": timeline.fallback_reason,
        }
    )
    manifest_path = evidence_root / "manifest.json"
    if manifest_path.exists():
        current = _read_json(manifest_path)
        if "global_frame_count" in current and current != manifest:
            raise ValueError("refusing to overwrite mismatched evidence manifest")
        if current != manifest:
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
    _write_json(aggregate_path, aggregate, overwrite=True)
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
    return row


def _candidate_model(record: Mapping[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        video_id=str(record["video_id"]),
        pass_id=str(record["pass_id"]),
        answer=str(record["answer"]),
        elapsed_seconds=float(record["elapsed_seconds"]),
        confidence=float(record["confidence"]),
        support_span_ids=("fixture",),
        request_sha256=str(record["request_sha256"]),
        runtime_provenance={
            "model_id": MODEL_ID,
            "provider": "fixture",
            "runtime": "fixture",
            "status": "passed",
            "request_sha256": str(record["request_sha256"]),
            "response_sha256": str(record["response_sha256"]),
        },
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
    normalized = [value.strip().casefold() for value in values]
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
        if artifact.name == "terminal-status.json":
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
        stack: list[tuple[Any, str]] = [(value, str(artifact))]
        while stack:
            current, location = stack.pop()
            if isinstance(current, Mapping):
                for key, nested in current.items():
                    key_text = str(key).casefold()
                    if any(term in key_text for term in _FREEZE_FORBIDDEN_TERMS):
                        raise ValueError("pre-freeze artifact contains label metadata")
                    stack.append((nested, f"{location}.{key}"))
            elif isinstance(current, (list, tuple)):
                stack.extend((nested, location) for nested in current)
            elif isinstance(current, str) and any(
                any(term in component.casefold() for term in _FREEZE_FORBIDDEN_TERMS)
                for component in current.replace("\\", "/").split("/")
            ):
                raise ValueError("pre-freeze artifact contains a forbidden path component")


def _latest_attempts(run_path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows = _read_jsonl(run_path / "attempts.jsonl")
    return {
        _attempt_key(row["request"]): row
        for row in rows
        if isinstance(row.get("request"), Mapping)
    }


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
    if not isinstance(predictions, list) or len(predictions) != len(prepared.examples):
        raise ValueError("freeze requires one prediction for every ordered manifest row")
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
        if len({answer.strip().casefold() for answer in answers}) != len(PASS_IDS):
            raise ValueError("freeze requires ten unique candidate answers")
        ranked = prediction.get("ranked_pass_ids")
        if not isinstance(ranked, list) or len(ranked) != len(PASS_IDS) or set(ranked) != set(PASS_IDS):
            raise ValueError("freeze requires a complete ranked pass permutation")
    latest = _latest_attempts(prepared.path)
    if any(record.get("status") != "completed" for record in latest.values()):
        raise ValueError("freeze requires no unresolved terminal failures")
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
            )
        )
    return records


def _validate_answer_response(response: Any) -> None:
    if not isinstance(response, Mapping) or not isinstance(response.get("answer"), str):
        raise ValueError("answer response schema is invalid")
    if not response["answer"].strip():
        raise ValueError("answer response is empty")


def _validate_timeline_response(response: Any) -> None:
    if not isinstance(response, Mapping):
        raise ValueError("timeline response schema is invalid")


def _validate_verifier_response(response: Any) -> None:
    if _valid_ranked_passes(response) is None:
        raise ValueError("verifier response schema is invalid")


def _record_sort_key(record: Mapping[str, Any]) -> tuple[int, int]:
    pass_id = str(record.get("pass_id", ""))
    return int(record["query_id"]), PASS_IDS.index(pass_id) if pass_id in PASS_IDS else len(PASS_IDS)


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


def run_prepared(
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
        timeline_request = _timeline_request(prepared_run, example)
        timeline_response, timeline_checkpoint, reused = _run_attempt(
            timeline_request,
            transport,
            attempts,
            failures,
            attempts_path=prepared_run.path / "attempts.jsonl",
            failures_path=prepared_run.path / "failures.jsonl",
            validator=_validate_timeline_response,
        )
        row_reused &= reused
        duration = float(example.get("duration_seconds", 10.0))
        timeline_payload = timeline_response if isinstance(timeline_response, Mapping) else {}
        timeline = validate_timeline(
            timeline_payload,
            duration,
            allow_fallback=True,
        )
        evidence_plan, evidence_files = _fixture_evidence(
            prepared_run, example, timeline
        )
        _append_jsonl(
            prepared_run.path / "timeline.jsonl",
            {
                **timeline_checkpoint,
                "timeline_fallback_reason": timeline.fallback_reason,
            },
        )
        row = _fixture_row(prepared_run, example, evidence_files)
        request_plan = evidence_files.get("model_plan", evidence_plan)
        answer_requests = build_answer_requests(row, timeline, request_plan)
        primary_candidates: list[dict[str, Any]] = []
        for pass_id in PASS_IDS:
            request = dict(answer_requests[pass_id].to_dict())
            request["stage"] = "answer"
            request["query_id"] = query_id
            request = _bind_request(prepared_run, request)
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
                    )
                )
        candidates = [_candidate_model(record) for record in primary_candidates]
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
                )
                normalized = str(response["answer"]).strip().casefold()
                if all(candidate.answer.strip().casefold() != normalized for candidate in unique_candidates):
                    duplicate_index = next(
                        (
                            index
                            for index, candidate in enumerate(candidates)
                            if candidate.answer.strip().casefold()
                            == unique_candidates[-1].answer.strip().casefold()
                        ),
                        None,
                    )
                    if duplicate_index is not None:
                        backfill_record["pass_id"] = candidates[duplicate_index].pass_id
                        candidates[duplicate_index] = _candidate_model(backfill_record)
                    unique_candidates = list(deduplicate_candidates(candidates))
        primary_candidates = []
        for candidate in candidates:
            record = candidate.to_dict()
            record["query_id"] = query_id
            record["response_sha256"] = str(
                candidate.runtime_provenance["response_sha256"]
            )
            record["usage"] = _usage()
            record["uploaded_file_deletion_status"] = {}
            record["failure_code"] = None
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
            "status": "inferred" if completed_rows == len(selected) else "partial",
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


def status(prepared: PreparedRun | Path | str) -> dict[str, Any]:
    """Read and summarize local artifacts without constructing a transport."""

    prepared_run = (
        prepared if isinstance(prepared, PreparedRun) else PreparedRun.load(prepared)
    )
    _validate_prepared_sources(prepared_run)
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
        existing = _read_json(sidecar_path)
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


def preflight_live(model_id: str = MODEL_ID) -> dict[str, Any]:
    if model_id != MODEL_ID:
        raise ValueError(f"model_id must be {MODEL_ID}")
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
    run_parser.add_argument("--transport", choices=("fixture", "gemini"), required=True)
    run_parser.add_argument("--limit", type=int)

    preflight_parser = subparsers.add_parser("preflight-live")
    preflight_parser.add_argument("--model-id", default=MODEL_ID)

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
        if args.transport == "gemini" and prepared_run.fixture_mode:
            raise ValueError("gemini transport requires a non-fixture prepared run")
        transport: Any
        if args.transport == "fixture":
            transport = FixtureTransport()
        else:
            transport = GeminiSubmissionClient(
                create_google_transport(), model_id=prepared_run.model_id
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
            print(_canonical_bytes(preflight_live(args.model_id)).decode("utf-8"))
        except ProviderAuthenticationError as error:
            print(str(error) or AUTHENTICATION_ERROR_MESSAGE, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
