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
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from trec_video_2026.vqa_gate import canonical_request_sha256
from trec_video_2026.vqa_submission_evidence import (
    EvidencePlan,
    ValidatedTimeline,
    build_global_local_frame_plan,
    validate_timeline,
)
from trec_video_2026.vqa_submission_gemini import (
    AUTHENTICATION_ERROR_MESSAGE,
    ProviderAuthenticationError,
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
)


SCHEMA_VERSION = 1
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

    @property
    def run_dir(self) -> Path:
        return self.path

    @property
    def output_dir(self) -> Path:
        return self.path

    @classmethod
    def load(cls, path: Path | str) -> "PreparedRun":
        run_path = Path(path)
        manifest = _read_json(run_path / "run-manifest.json")
        raw_examples = manifest.get("examples")
        if not isinstance(raw_examples, list):
            raise ValueError("run manifest examples must be a list")
        raw_baseline = manifest.get("baseline_ranked_list", ())
        if not isinstance(raw_baseline, list):
            raise ValueError("run manifest baseline_ranked_list must be a list")
        return cls(
            path=run_path,
            run_id=str(manifest["run_id"]),
            model_id=str(manifest["model_id"]),
            examples=tuple(dict(example) for example in raw_examples),
            baseline_ranked_list=tuple(str(item) for item in raw_baseline),
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


def prepare(config_path: Path | str, output: Path | str) -> PreparedRun:
    """Prepare a label-free run directory atomically and exactly once."""

    config_file = Path(config_path)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(output_path)
    config_bytes = config_file.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("config must contain a JSON object")
    examples = _validate_config(config)

    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=parent))
    try:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": config["run_id"],
            "model_id": config["model_id"],
            "fixture": bool(config.get("fixture", False)),
            "config_sha256": _sha256_bytes(config_bytes),
            "examples": [dict(example) for example in examples],
            "pass_ids": list(PASS_IDS),
            "baseline_ranked_list": list(_baseline_ranked_list(config)),
        }
        _write_json(temporary / "run-manifest.json", manifest, overwrite=False)
        _write_json(
            temporary / "evidence" / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": config["run_id"],
                "rows": [
                    {"query_id": example["query_id"], "video_id": example["video_id"]}
                    for example in examples
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
                "prepared_rows": len(examples),
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
    audit_label_free_payload(request)
    return request


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
    audit_label_free_payload(request)
    return request


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
        and record.get("response_sha256") == response_hash
    )


def _run_attempt(
    request: Mapping[str, Any],
    transport: Transport,
    attempts: dict[tuple[int, str, str], dict[str, Any]],
    failures: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[Mapping[str, Any] | None, dict[str, Any], bool]:
    key = _attempt_key(request)
    saved = attempts.get(key)
    if saved is not None and _attempt_matches(saved, request):
        response = saved.get("response")
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
        "status": "completed",
        "elapsed_seconds": 0.0,
        "usage": _usage(),
        "uploaded_file_deletion_status": {},
        "failure_code": None,
    }
    try:
        response = transport.generate(request)
        if not isinstance(response, Mapping):
            raise TypeError("transport.generate must return a mapping")
        response_dict = dict(response)
        response_bytes = _canonical_bytes(response_dict)
        record["response"] = response_dict
        record["response_bytes"] = response_bytes.decode("utf-8")
        record["response_sha256"] = _sha256_bytes(response_bytes)
    except Exception as error:  # noqa: BLE001 - bounded transport checkpoint
        response = None
        record["status"] = "failed"
        record["failure_code"] = _failure_code(error)
        failures[key] = record
    attempts[key] = record
    return response, record, False


def _fixture_evidence(
    prepared: PreparedRun,
    example: Mapping[str, Any],
    timeline: ValidatedTimeline,
) -> tuple[EvidencePlan, dict[str, Any]]:
    duration = float(example.get("duration_seconds", 10.0))
    evidence_root = prepared.path / "evidence"
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
                "frame_path": frame_path,
            }
        )
    plan = build_global_local_frame_plan(duration, inventory, timeline)
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
    return plan, {
        "media_path": media_path,
        "audio_path": audio_path,
        "media_sha256": media_sha,
        "audio_sha256": _sha256_bytes(audio_path.read_bytes()),
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
            "media_path": evidence_files["media_path"],
            "audio_path": evidence_files["audio_path"],
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
    candidate_rows: dict[tuple[int, str], dict[str, Any]] = {}
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
            timeline_request, transport, attempts, failures
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
        _write_jsonl(
            prepared_run.path / "timeline.jsonl",
            [
                {
                    **timeline_checkpoint,
                    "timeline_fallback_reason": timeline.fallback_reason,
                }
            ],
        )
        row = _fixture_row(prepared_run, example, evidence_files)
        answer_requests = build_answer_requests(row, timeline, evidence_plan)
        primary_candidates: list[dict[str, Any]] = []
        for pass_id in PASS_IDS:
            request = dict(answer_requests[pass_id].to_dict())
            request["stage"] = "answer"
            request["query_id"] = query_id
            response, checkpoint, pass_reused = _run_attempt(
                request, transport, attempts, failures
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
                row, evidence_plan, unique_candidates, attempt=attempt
            )
            request = dict(backfill_request_object.to_dict())
            request["stage"] = "backfill"
            request["attempt"] = attempt
            request["query_id"] = query_id
            response, checkpoint, backfill_reused = _run_attempt(
                request, transport, attempts, failures
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
            ranking_by_query[query_id] = PASS_IDS
            source_by_query[query_id] = "configured-baseline-ranked-list"
        else:
            try:
                verifier_request = build_verifier_request(
                    row, timeline, evidence_plan, candidates, seed=query_id
                )
                verifier_payload = dict(verifier_request.to_model_payload())
                verifier_payload["stage"] = "verifier"
                verifier_payload["query_id"] = query_id
                verifier_payload["request_sha256"] = verifier_request.request_sha256
                verifier_response, verifier_checkpoint, verifier_reused = _run_attempt(
                    verifier_payload, transport, attempts, failures
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
    attempt_ordered = sorted(
        attempts.values(), key=lambda row: (int(row["query_id"]), str(row.get("stage")), str(row.get("pass_id")))
    )
    _write_jsonl(prepared_run.path / "attempts.jsonl", attempt_ordered)
    _write_jsonl(
        prepared_run.path / "failures.jsonl",
        sorted(failures.values(), key=lambda row: _attempt_key(row["request"])),
    )
    predictions = _build_ranked_predictions(
        prepared_run, ordered_records, ranking_by_query, source_by_query
    )
    candidate_shortfall = max(shortfall_by_query.values(), default=0)
    prediction_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prepared_run.run_id,
        "model_id": prepared_run.model_id,
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
    freeze_record = freeze_predictions(
        prediction_path,
        bindings={
            "run_id": prepared_run.run_id,
            "model_id": prepared_run.model_id,
            "prediction_count": payload.get("prediction_count"),
        },
        output_path=prepared_run.path / "freeze.json",
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
    prepare_parser.add_argument("--output", required=True, type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prepared", required=True, type=Path)
    run_parser.add_argument("--transport", choices=("fixture",), required=True)
    run_parser.add_argument("--limit", type=int)

    preflight_parser = subparsers.add_parser("preflight-live")
    preflight_parser.add_argument("--model-id", default=MODEL_ID)

    for name in ("freeze", "status"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--prepared", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.config, args.output)
    elif args.command == "run":
        run_prepared(
            args.prepared,
            transport=FixtureTransport(),
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
