"""Build and freeze the private two-view native-evidence Phase 0 audit.

The audit is deliberately an offline boundary. It reads already-frozen runner
artifacts and native media, extracts deterministic inspection frames, and
leaves all human verdict files empty for the operator to complete. It does
not read references, call providers, run ASR, or auto-judge evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from trec_video_2026.vqa_benchmark import (
    bin_midpoint_timestamps,
    probe_media,
    source_fingerprint,
)
from trec_video_2026.vqa_frontier_phase0_contract import (
    validate_grounded_answer,
    validate_neutral_map,
)
from trec_video_2026.vqa_frontier_phase0_panel import CONDITIONS
from trec_video_2026.vqa_frontier_phase0_runner import (
    _verify_freeze_file,
    load_prepared,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    freeze_json_artifact,
    verify_json_artifact,
)
from trec_video_2026.vqa_submission import canonical_sha256


SCHEMA_VERSION = 1
AUDIT_ID = "frontier-phase0-native-evidence-audit-v1"
CANONICAL_DATA_ROOT = Path(
    "/home/npatta01/data/competitions/trec_video_2026/data"
)
OVERVIEW_FRAME_COUNT = 6
ANSWER_CONDITIONS = ("direct_answer", "map_guided_answer")
OBSERVATION_STATES = frozenset({"confirmed", "contradicted", "unverifiable"})
COVERAGE_STATES = frozenset({"complete", "partial", "insufficient"})
MAP_DECISIONS = frozenset({"usable_for_screen", "revise_contract", "stop"})
PILOT_ANSWER_EVIDENCE = frozenset({"not_assessed"})
SCREEN_ANSWER_EVIDENCE = frozenset({"present", "partial", "absent"})
CLAIM_SUPPORT_STATES = frozenset(
    {"supported", "partial", "unsupported", "unverifiable"}
)
CLAIM_CITATION_STATES = frozenset(
    {"valid", "invalid", "unverifiable", "not_applicable"}
)
REFERENCE_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "correctanswer",
        "gold",
        "goldanswer",
        "groundtruth",
        "label",
        "labels",
        "oracle",
        "reference",
        "references",
        "referenceanswer",
        "referenceanswers",
    }
)
BLIND_KEYS = frozenset(
    {
        "answer",
        "answers",
        "question",
        "questions",
        "reference",
        "references",
        "top1answer",
    }
)
SAFE_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class AuditPacket:
    """Paths in a newly built private audit packet."""

    root: Path

    @property
    def blind_manifest(self) -> Path:
        return self.root / "blind-manifest.json"

    @property
    def aware_manifest(self) -> Path:
        return self.root / "aware-manifest.json"

    @property
    def private_manifest(self) -> Path:
        return self.root / "private-manifest.json"

    @property
    def packet_manifest(self) -> Path:
        return self.root / "packet-manifest.json"

    @property
    def map_verdicts(self) -> Path:
        return self.root / "map-verdicts.jsonl"

    @property
    def answer_verdicts(self) -> Path:
        return self.root / "answer-verdicts.jsonl"

    @property
    def audit_json(self) -> Path:
        return self.root / "audit.json"

    @property
    def freeze_json(self) -> Path:
        return self.root / "freeze.json"


def build_audit_packet(
    prepared: Path, *, phase: Literal["pilot", "screen"], output: Path
) -> AuditPacket:
    """Build a private audit packet from an exact frozen runner phase."""

    if phase not in {"pilot", "screen"}:
        raise ValueError("phase must be pilot or screen")
    root = _prepare_output_root(Path(output))
    prepared_obj = load_prepared(Path(prepared))
    rows, prediction = _frozen_phase(prepared_obj, phase)
    packet = AuditPacket(root)
    blind_rows: list[dict[str, Any]] = []
    aware_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []

    for row in rows:
        row_results = _results_for_row(prediction, row.video_id, phase)
        map_payload = _validated_map(row_results["neutral_map"], row.duration_seconds)
        answer_results = {
            condition: row_results.get(condition) for condition in ANSWER_CONDITIONS
        }
        map_event_ids = {
            event["event_id"]
            for event in map_payload.get("events", [])
            if isinstance(event, Mapping) and isinstance(event.get("event_id"), str)
        }
        answer_payloads = {
            condition: _validated_answer(
                result,
                row.duration_seconds,
                allowed_map_event_ids=(
                    map_event_ids if condition == "map_guided_answer" else None
                ),
            )
            for condition, result in answer_results.items()
        }
        source = _resolve_source(prepared_obj.path, row)
        row_dir = root / "rows" / row.video_id
        row_dir.mkdir(parents=True, exist_ok=True)
        media = _materialize_media(
            source,
            row_dir=row_dir,
            duration=float(source["duration_seconds"]),
            map_payload=map_payload,
            answer_payloads=answer_payloads,
        )
        result_hashes = {
            condition: _result_hash(result) for condition, result in row_results.items()
        }
        request_hashes = {
            condition: _request_hash(result)
            for condition, result in row_results.items()
        }
        raw_paths = {
            condition: result.get("raw_response_path")
            for condition, result in row_results.items()
            if result.get("raw_response_path")
        }
        common = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "run_id": prepared_obj.run_id,
            "panel_sha256": prepared_obj.panel_sha256,
            "video_id": row.video_id,
            "source_media_sha256": source["sha256"],
            "source_media_size_bytes": source["size_bytes"],
            "source_duration_seconds": source["duration_seconds"],
            "has_audio": source["has_audio"],
            "has_video": source["has_video"],
            "result_hashes": result_hashes,
            "request_hashes": request_hashes,
            "map": map_payload,
            "media": _public_media(media),
        }
        blind = dict(common)
        _reject_reference_fields(blind)
        _reject_blind_leakage(blind)
        blind_rows.append(blind)

        aware = dict(common)
        aware["question"] = row.question
        aware["frozen_answers"] = {
            condition: payload
            for condition, payload in answer_payloads.items()
            if payload is not None
        }
        aware["answer_citations"] = {
            condition: media["answer_citations"].get(condition, [])
            for condition in ANSWER_CONDITIONS
        }
        _reject_reference_fields(aware)
        aware_rows.append(aware)

        private_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "run_id": prepared_obj.run_id,
                "panel_sha256": prepared_obj.panel_sha256,
                "video_id": row.video_id,
                "panel_media_relative_path": row.media_relative_path,
                "native_media_path": source["path"],
                "source_media_sha256": source["sha256"],
                "source_media_size_bytes": source["size_bytes"],
                "source_duration_seconds": source["duration_seconds"],
                "ffmpeg_version": media["ffmpeg_version"],
                "ffprobe_version": media["ffprobe_version"],
                "result_hashes": result_hashes,
                "request_hashes": request_hashes,
                "raw_response_paths": raw_paths,
                "overview_frames": media["overview_frames"],
                "event_keyframes": media["event_keyframes"],
                "answer_citations": media["answer_citations"],
                "media": media,
            }
        )
        packet_rows.append(
            {
                "video_id": row.video_id,
                "blind_manifest": f"rows/{row.video_id}/blind.json",
                "aware_manifest": f"rows/{row.video_id}/aware.json",
                "source_media_sha256": source["sha256"],
                "result_hashes": result_hashes,
                "request_hashes": request_hashes,
                "media_sha256": canonical_sha256(media),
            }
        )
        _write_json_once(row_dir / "blind.json", blind)
        _write_json_once(row_dir / "aware.json", aware)

    _write_json_once(
        packet.blind_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_id": AUDIT_ID,
            "view": "blind-map",
            "phase": phase,
            "rows": blind_rows,
        },
    )
    _write_json_once(
        packet.aware_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_id": AUDIT_ID,
            "view": "question-aware-no-labels",
            "phase": phase,
            "rows": aware_rows,
        },
    )
    _write_json_once(
        packet.private_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_id": AUDIT_ID,
            "run_id": prepared_obj.run_id,
            "panel_sha256": prepared_obj.panel_sha256,
            "phase": phase,
            "prediction_freeze_sha256": _sha256_file(
                prepared_obj.path / f"{phase}-freeze.json"
            ),
            "prediction_artifact_sha256": _sha256_file(
                prepared_obj.path / f"{phase}-results.json"
            ),
            "rows": private_rows,
        },
    )
    _write_json_once(
        packet.packet_manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "audit_id": AUDIT_ID,
            "run_id": prepared_obj.run_id,
            "panel_sha256": prepared_obj.panel_sha256,
            "phase": phase,
            "row_count": len(packet_rows),
            "rows": packet_rows,
        },
    )
    _create_empty(packet.map_verdicts)
    _create_empty(packet.answer_verdicts)
    return packet


def validate_audit_packet(root: Path) -> Mapping[str, object]:
    """Validate packet structure, leakage boundaries, and any freeze."""

    packet = AuditPacket(_validate_audit_root(Path(root)))
    packet_manifest = _read_json(packet.packet_manifest)
    blind_manifest = _read_json(packet.blind_manifest)
    aware_manifest = _read_json(packet.aware_manifest)
    private_manifest = _read_json(packet.private_manifest)
    if packet_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("packet manifest schema_version is invalid")
    phase = _required_text(packet_manifest.get("phase"), "phase")
    if phase not in {"pilot", "screen"}:
        raise ValueError("packet phase is invalid")
    _validate_manifest_rows(packet, packet_manifest, blind_manifest, aware_manifest)
    _reject_reference_fields(private_manifest)
    map_verdicts = _read_jsonl(packet.map_verdicts, "map verdicts")
    answer_verdicts = _read_jsonl(packet.answer_verdicts, "answer verdicts")
    expected_ids = _expected_video_ids(packet_manifest)
    _validate_map_verdicts(
        map_verdicts,
        expected_ids=expected_ids,
        packet=packet,
        phase=phase,
        require_complete=False,
    )
    _validate_answer_verdicts(
        answer_verdicts,
        expected_ids=expected_ids,
        packet=packet,
        phase=phase,
        require_complete=False,
    )
    if packet.freeze_json.exists():
        freeze = _verify_audit_freeze(packet)
        return {
            "status": "valid",
            "phase": phase,
            "row_count": len(expected_ids),
            "map_verdict_count": len(map_verdicts),
            "answer_verdict_count": len(answer_verdicts),
            "frozen": True,
            "decision": freeze["bindings"]["decision"],
        }
    return {
        "status": "valid",
        "phase": phase,
        "row_count": len(expected_ids),
        "map_verdict_count": len(map_verdicts),
        "answer_verdict_count": len(answer_verdicts),
        "frozen": False,
    }


def freeze_pilot_audit(root: Path) -> Mapping[str, object]:
    """Freeze a complete pilot map audit without making a judgment."""

    packet = AuditPacket(_validate_audit_root(Path(root)))
    validate_audit_packet(packet.root)
    if _read_json(packet.packet_manifest).get("phase") != "pilot":
        raise ValueError("pilot freeze requires a pilot audit packet")
    if packet.audit_json.exists() or packet.freeze_json.exists():
        raise FileExistsError("audit freeze artifacts already exist")
    packet_manifest = _read_json(packet.packet_manifest)
    map_verdicts = _read_jsonl(packet.map_verdicts, "map verdicts")
    answer_verdicts = _read_jsonl(packet.answer_verdicts, "answer verdicts")
    ids = _expected_video_ids(packet_manifest)
    _validate_map_verdicts(
        map_verdicts,
        expected_ids=ids,
        packet=packet,
        phase="pilot",
        require_complete=True,
    )
    _validate_answer_verdicts(
        answer_verdicts,
        expected_ids=ids,
        packet=packet,
        phase="pilot",
        require_complete=True,
    )
    decision = map_verdicts[0]["overall_decision"]
    artifact = _audit_artifact(packet, decision=decision, screen_freeze=None)
    _write_json_once(packet.audit_json, artifact)
    freeze = freeze_json_artifact(
        packet.audit_json,
        bindings={
            "run_id": artifact["run_id"],
            "panel_sha256": artifact["panel_sha256"],
            "phase": "pilot",
            "decision": decision,
        },
    )
    _write_json_once(packet.freeze_json, freeze)
    return freeze


def freeze_screen_audit(root: Path) -> Mapping[str, object]:
    """Freeze the seven-row, fourteen-answer screen audit."""

    packet = AuditPacket(_validate_audit_root(Path(root)))
    validate_audit_packet(packet.root)
    if _read_json(packet.packet_manifest).get("phase") != "screen":
        raise ValueError("screen freeze requires a screen audit packet")
    if packet.audit_json.exists() or packet.freeze_json.exists():
        raise FileExistsError("audit freeze artifacts already exist")
    packet_manifest = _read_json(packet.packet_manifest)
    private_manifest = _read_json(packet.private_manifest)
    map_verdicts = _read_jsonl(packet.map_verdicts, "map verdicts")
    answer_verdicts = _read_jsonl(packet.answer_verdicts, "answer verdicts")
    ids = _expected_video_ids(packet_manifest)
    if len(ids) != 7:
        raise ValueError("screen audit requires seven panel rows")
    _validate_map_verdicts(
        map_verdicts,
        expected_ids=ids,
        packet=packet,
        phase="screen",
        require_complete=True,
    )
    _validate_answer_verdicts(
        answer_verdicts,
        expected_ids=ids,
        packet=packet,
        phase="screen",
        require_complete=True,
    )
    screen_freeze = private_manifest.get("prediction_freeze_sha256")
    if not isinstance(screen_freeze, str) or not SHA256.fullmatch(screen_freeze):
        raise ValueError("screen prediction freeze hash is missing")
    artifact = _audit_artifact(packet, decision="complete", screen_freeze=screen_freeze)
    _write_json_once(packet.audit_json, artifact)
    freeze = freeze_json_artifact(
        packet.audit_json,
        bindings={
            "run_id": artifact["run_id"],
            "panel_sha256": artifact["panel_sha256"],
            "phase": "screen",
            "decision": "complete",
            "screen_prediction_freeze_sha256": screen_freeze,
        },
    )
    _write_json_once(packet.freeze_json, freeze)
    return freeze


def _frozen_phase(prepared: Any, phase: str) -> tuple[list[Any], Mapping[str, Any]]:
    freeze_path = prepared.path / f"{phase}-freeze.json"
    artifact_path = prepared.path / f"{phase}-results.json"
    if not freeze_path.is_file() or not artifact_path.is_file():
        raise ValueError(f"{phase} prediction freeze is missing")
    _verify_freeze_file(prepared, freeze_path)
    artifact = _read_json(artifact_path)
    rows = [prepared.pilot_row] if phase == "pilot" else list(prepared.screen_rows)
    if artifact.get("phase") != phase or artifact.get("run_id") != prepared.run_id:
        raise ValueError("prediction freeze phase binding is invalid")
    if artifact.get("panel_sha256") != prepared.panel_sha256:
        raise ValueError("prediction freeze panel binding is invalid")
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("prediction results are missing")
    return rows, artifact


def _results_for_row(
    prediction: Mapping[str, Any], video_id: str, phase: str
) -> dict[str, Mapping[str, Any]]:
    results = prediction.get("results")
    if not isinstance(results, list):
        raise ValueError("prediction results are missing")
    found: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("prediction result must be an object")
        if result.get("video_id") != video_id or result.get("phase") != phase:
            continue
        condition = result.get("condition_id")
        if condition not in CONDITIONS:
            raise ValueError("prediction result condition is invalid")
        if condition in found:
            raise ValueError("duplicate prediction result for audit row")
        found[str(condition)] = result
    expected = {"neutral_map"} if phase == "pilot" else set(CONDITIONS)
    if set(found) != expected:
        raise ValueError("prediction freeze is missing an audit condition")
    return found


def _validated_map(result: Mapping[str, Any], duration: float) -> Mapping[str, Any]:
    if result.get("status") != "succeeded":
        raise ValueError("audit requires a successful neutral map result")
    payload = result.get("validated_output")
    if not isinstance(payload, Mapping):
        raise ValueError("neutral map validated output is missing")
    validate_neutral_map(payload, duration)
    _reject_reference_fields(payload)
    return dict(payload)


def _validated_answer(
    result: Mapping[str, Any] | None,
    duration: float,
    *,
    allowed_map_event_ids: set[str] | None,
) -> Mapping[str, Any] | None:
    if result is None:
        return None
    if result.get("status") != "succeeded":
        raise ValueError("audit requires successful answer results")
    payload = result.get("validated_output")
    if not isinstance(payload, Mapping):
        raise ValueError("answer validated output is missing")
    validate_grounded_answer(
        payload, duration, allowed_map_event_ids=allowed_map_event_ids
    )
    _reject_reference_fields(payload)
    return dict(payload)


def _resolve_source(prepared: Path, row: Any) -> dict[str, Any]:
    relative = Path(row.media_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source media path escapes its private root")
    suffix = relative.suffix.lower()
    fixture_root = prepared / "fixture-media"
    candidates = [
        fixture_root / f"{row.video_id}{candidate_suffix}"
        for candidate_suffix in (suffix, ".webm", ".mp4", ".mkv")
    ]
    data_root = _canonical_root()
    candidates.append(data_root / relative)
    source: Path | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.is_symlink():
            raise ValueError("source media path cannot be a symlink")
        resolved = candidate.resolve(strict=True)
        if candidate == candidates[-1] and not resolved.is_relative_to(data_root):
            raise ValueError("source media path escapes canonical private data root")
        if resolved.is_file():
            source = resolved
            break
    if source is None:
        raise ValueError(f"source media is missing for {row.video_id}")
    fingerprint = source_fingerprint(source)
    if fingerprint.sha256 != row.media_sha256 or fingerprint.size_bytes != row.size_bytes:
        raise ValueError("source media bytes do not match frozen panel identity")
    metadata = probe_media(source)
    if not math.isclose(metadata.duration_seconds, row.duration_seconds, abs_tol=0.1):
        raise ValueError("source duration does not match frozen panel identity")
    if metadata.has_audio != row.has_audio or metadata.has_video != row.has_video:
        raise ValueError("source stream metadata does not match frozen panel identity")
    return {
        "path": str(source),
        "sha256": fingerprint.sha256,
        "size_bytes": fingerprint.size_bytes,
        "duration_seconds": metadata.duration_seconds,
        "has_audio": metadata.has_audio,
        "has_video": metadata.has_video,
    }


def _materialize_media(
    source: Mapping[str, Any],
    *,
    row_dir: Path,
    duration: float,
    map_payload: Mapping[str, Any],
    answer_payloads: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    ffmpeg_version = _tool_version("ffmpeg")
    ffprobe_version = _tool_version("ffprobe")
    overview_frames: list[dict[str, Any]] = []
    event_keyframes: list[dict[str, Any]] = []
    answer_citations: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in ANSWER_CONDITIONS
    }
    overview_path: str | None = None
    overview_sha256: str | None = None
    if source["has_video"]:
        overview_path_obj = row_dir / "overview.jpg"
        overview_timestamps = bin_midpoint_timestamps(
            float(duration), count=OVERVIEW_FRAME_COUNT
        )
        with tempfile.TemporaryDirectory(prefix=".overview-", dir=row_dir) as temp:
            temp_root = Path(temp)
            for index, desired in enumerate(overview_timestamps, start=1):
                frame_path = temp_root / f"frame-{index:04d}.jpg"
                entry = _extract_frame(
                    Path(str(source["path"])),
                    desired,
                    frame_path,
                    duration=float(duration),
                    source_sha256=str(source["sha256"]),
                    ffmpeg_version=ffmpeg_version,
                )
                overview_frames.append(entry)
            _contact_sheet(temp_root, overview_path_obj)
        overview_path = overview_path_obj.relative_to(row_dir.parent.parent).as_posix()
        overview_sha256 = _sha256_file(overview_path_obj)

        events = map_payload.get("events")
        if not isinstance(events, list):
            raise ValueError("neutral map events are missing")
        event_dir = row_dir / "event-keyframes"
        event_dir.mkdir(parents=True, exist_ok=True)
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("neutral map event must be an object")
            event_id = _required_text(event.get("event_id"), "event_id")
            if SAFE_SEGMENT.fullmatch(event_id) is None:
                raise ValueError("event_id is not a safe path segment")
            timestamps = event.get("keyframe_seconds")
            if not isinstance(timestamps, list) or not timestamps:
                raise ValueError("neutral map event keyframes are missing")
            for index, desired in enumerate(timestamps, start=1):
                frame_path = event_dir / f"{event_id}-{index}.jpg"
                entry = _extract_frame(
                    Path(str(source["path"])),
                    float(desired),
                    frame_path,
                    duration=float(duration),
                    source_sha256=str(source["sha256"]),
                    ffmpeg_version=ffmpeg_version,
                )
                entry.update({"event_id": event_id, "index": index})
                event_keyframes.append(entry)

        for condition, answer in answer_payloads.items():
            if answer is None:
                continue
            claims = answer.get("claims")
            if not isinstance(claims, list):
                raise ValueError("answer claims are missing")
            citation_dir = row_dir / "answer-citations" / condition
            citation_dir.mkdir(parents=True, exist_ok=True)
            for claim in claims:
                if not isinstance(claim, Mapping):
                    raise ValueError("answer claim must be an object")
                claim_id = _required_text(claim.get("claim_id"), "claim_id")
                if SAFE_SEGMENT.fullmatch(claim_id) is None:
                    raise ValueError("claim_id is not a safe path segment")
                spans = claim.get("supporting_spans")
                if not isinstance(spans, list) or not spans:
                    raise ValueError("answer claim citations are missing")
                for span_index, span in enumerate(spans, start=1):
                    if not isinstance(span, Mapping):
                        raise ValueError("answer citation span must be an object")
                    start = _finite_nonnegative(span.get("start_seconds"), "start_seconds")
                    end = _finite_nonnegative(span.get("end_seconds"), "end_seconds")
                    if end < start:
                        raise ValueError("answer citation span ends before it starts")
                    for point_index, desired in enumerate(
                        (start, (start + end) / 2.0, end), start=1
                    ):
                        frame_path = citation_dir / f"{claim_id}-{span_index}-{point_index}.jpg"
                        entry = _extract_frame(
                            Path(str(source["path"])),
                            desired,
                            frame_path,
                            duration=float(duration),
                            source_sha256=str(source["sha256"]),
                            ffmpeg_version=ffmpeg_version,
                        )
                        entry.update(
                            {
                                "claim_id": claim_id,
                                "span_index": span_index,
                                "point": ("start", "mid", "end")[point_index - 1],
                            }
                        )
                        answer_citations[condition].append(entry)
    return {
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "overview_path": overview_path,
        "overview_sha256": overview_sha256,
        "overview_frames": overview_frames,
        "event_keyframes": event_keyframes,
        "answer_citations": answer_citations,
    }


def _extract_frame(
    source: Path,
    desired: float,
    output: Path,
    *,
    duration: float,
    source_sha256: str,
    ffmpeg_version: str,
) -> dict[str, Any]:
    clamped = min(max(float(desired), 0.0), max(0.0, duration - 1e-6))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"frame already exists: {output.name}")
    def run_at(seek: float) -> subprocess.CompletedProcess[str]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-ss",
            f"{seek:.6f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "showinfo",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ]
        return subprocess.run(command, check=True, capture_output=True, text=True)

    try:
        completed = run_at(clamped)
    except subprocess.CalledProcessError:
        output.unlink(missing_ok=True)
        fallback = max(0.0, duration - max(0.3, duration * 0.1))
        completed = run_at(min(clamped, fallback))
    match = re.search(r"pts_time:([-+0-9.eE]+)", completed.stderr)
    actual = float(match.group(1)) if match else None
    if actual is None or not math.isfinite(actual):
        raise ValueError("ffmpeg did not report a decoded PTS")
    if not output.is_file() or output.is_symlink():
        raise ValueError("ffmpeg did not produce a regular JPEG frame")
    return {
        "path": output.relative_to(output.parents[2]).as_posix(),
        "desired_pts": round(float(desired), 6),
        "actual_pts": round(actual, 6),
        "clamped_pts": round(clamped, 6),
        "frame_sha256": _sha256_file(output),
        "source_media_sha256": source_sha256,
        "ffmpeg_version": ffmpeg_version,
    }


def _contact_sheet(frame_dir: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"overview already exists: {output.name}")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-framerate",
            "1",
            "-i",
            str(frame_dir / "frame-%04d.jpg"),
            "-vf",
            "scale=320:-2,tile=3x2:padding=4:margin=4:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    if not output.is_file() or output.is_symlink():
        raise ValueError("ffmpeg did not produce the overview")


def _public_media(media: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in media.items() if key != "answer_citations"}


def _audit_artifact(
    packet: AuditPacket, *, decision: str, screen_freeze: str | None
) -> dict[str, Any]:
    packet_manifest = _read_json(packet.packet_manifest)
    private_manifest = _read_json(packet.private_manifest)
    phase = _required_text(packet_manifest.get("phase"), "phase")
    private_rows = private_manifest.get("rows")
    if not isinstance(private_rows, list):
        raise ValueError("private manifest rows are missing")
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "frozen",
        "run_id": _required_text(packet_manifest.get("run_id"), "run_id"),
        "panel_sha256": _required_sha256(
            packet_manifest.get("panel_sha256"), "panel_sha256"
        ),
        "phase": phase,
        "decision": decision,
        "prediction_freeze_sha256": private_manifest.get(
            "prediction_freeze_sha256"
        ),
        "prediction_artifact_sha256": private_manifest.get(
            "prediction_artifact_sha256"
        ),
        "result_hashes_sha256": canonical_sha256(
            [row.get("result_hashes") for row in private_rows]
        ),
        "request_hashes_sha256": canonical_sha256(
            [row.get("request_hashes") for row in private_rows]
        ),
        "source_media_sha256": canonical_sha256(
            [row.get("source_media_sha256") for row in private_rows]
        ),
        "manifest_sha256": {
            "packet": _sha256_file(packet.packet_manifest),
            "blind": _sha256_file(packet.blind_manifest),
            "aware": _sha256_file(packet.aware_manifest),
            "private": _sha256_file(packet.private_manifest),
        },
        "verdict_sha256": {
            "map": _sha256_file(packet.map_verdicts),
            "answer": _sha256_file(packet.answer_verdicts),
        },
        "counts": {
            "rows": len(private_rows),
            "map_verdicts": len(_read_jsonl(packet.map_verdicts, "map verdicts")),
            "answer_verdicts": len(
                _read_jsonl(packet.answer_verdicts, "answer verdicts")
            ),
        },
    }
    if screen_freeze is not None:
        audit["screen_prediction_freeze_sha256"] = screen_freeze
    return audit


def _verify_audit_freeze(packet: AuditPacket) -> Mapping[str, Any]:
    freeze = _read_json(packet.freeze_json)
    verify_json_artifact(packet.audit_json, dict(freeze))
    if freeze.get("schema_version") != SCHEMA_VERSION or freeze.get("status") != "frozen":
        raise ValueError("audit freeze schema or status is invalid")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("audit freeze bindings are missing")
    required = {"run_id", "panel_sha256", "phase", "decision"}
    if not required.issubset(bindings):
        raise ValueError("audit freeze core bindings are incomplete")
    artifact = _read_json(packet.audit_json)
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("status") != "frozen":
        raise ValueError("audit artifact schema or status is invalid")
    for key in required:
        if artifact.get(key) != bindings.get(key):
            raise ValueError("audit freeze core binding changed")
    if bindings["phase"] == "screen" and artifact.get(
        "screen_prediction_freeze_sha256"
    ) != bindings.get("screen_prediction_freeze_sha256"):
        raise ValueError("screen prediction freeze binding changed")
    manifest_hashes = artifact.get("manifest_sha256")
    verdict_hashes = artifact.get("verdict_sha256")
    if not isinstance(manifest_hashes, Mapping) or not isinstance(verdict_hashes, Mapping):
        raise ValueError("audit artifact manifest bindings are missing")
    expected_manifests = {
        "packet": _sha256_file(packet.packet_manifest),
        "blind": _sha256_file(packet.blind_manifest),
        "aware": _sha256_file(packet.aware_manifest),
        "private": _sha256_file(packet.private_manifest),
    }
    expected_verdicts = {
        "map": _sha256_file(packet.map_verdicts),
        "answer": _sha256_file(packet.answer_verdicts),
    }
    if dict(manifest_hashes) != expected_manifests or dict(verdict_hashes) != expected_verdicts:
        raise ValueError("audit manifest or verdict binding changed")
    return freeze


def _validate_manifest_rows(
    packet: AuditPacket,
    packet_manifest: Mapping[str, Any],
    blind_manifest: Mapping[str, Any],
    aware_manifest: Mapping[str, Any],
) -> None:
    rows = packet_manifest.get("rows")
    blind_rows = blind_manifest.get("rows")
    aware_rows = aware_manifest.get("rows")
    if not isinstance(rows, list) or not isinstance(blind_rows, list) or not isinstance(aware_rows, list):
        raise ValueError("audit manifests must contain rows")
    if not len(rows) == len(blind_rows) == len(aware_rows):
        raise ValueError("audit manifest row counts differ")
    if packet_manifest.get("row_count") != len(rows):
        raise ValueError("packet manifest row count is invalid")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("packet manifest row must be an object")
        video_id = _required_text(row.get("video_id"), "video_id")
        if video_id in seen:
            raise ValueError("duplicate audit video row")
        seen.add(video_id)
        for field in ("blind_manifest", "aware_manifest"):
            path = _safe_packet_path(packet.root, row.get(field), field)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"audit row manifest is missing: {field}")
        _reject_blind_leakage(blind_rows[index])
        _reject_reference_fields(blind_rows[index])
        _reject_reference_fields(aware_rows[index])


def _validate_map_verdicts(
    rows: list[Mapping[str, Any]],
    *,
    expected_ids: list[str],
    packet: AuditPacket,
    phase: str,
    require_complete: bool,
) -> None:
    by_id = _unique_by(rows, "video_id", "map verdicts")
    if require_complete and set(by_id) != set(expected_ids):
        raise ValueError("map verdicts must contain every audit video")
    for video_id, verdict in by_id.items():
        if video_id not in expected_ids:
            raise ValueError("map verdict contains an unknown video")
        _validate_verdict_keys(
            verdict,
            {
                "video_id",
                "event_checks",
                "coverage",
                "entity_identity",
                "ordering",
                "visible_text",
                "speech_and_sound",
                "answer_critical_evidence",
                "overall_decision",
                "notes",
            },
            "map verdict",
        )
        event_checks = verdict["event_checks"]
        if not isinstance(event_checks, list):
            raise ValueError("map event_checks must be a list")
        expected_events = _event_ids_for_video(packet, video_id)
        event_ids = _unique_by(event_checks, "event_id", "map event checks")
        if require_complete and set(event_ids) != set(expected_events):
            raise ValueError("map verdicts must contain every neutral-map event")
        if set(event_ids) - set(expected_events):
            raise ValueError("map verdict contains an unknown neutral-map event")
        for event in event_checks:
            _validate_verdict_keys(
                event,
                {"event_id", "status", "boundary_status", "notes"},
                "map event verdict",
            )
            if event["status"] not in OBSERVATION_STATES or event["boundary_status"] not in OBSERVATION_STATES:
                raise ValueError("map event verdict state is invalid")
        if verdict["coverage"] not in COVERAGE_STATES:
            raise ValueError("map coverage state is invalid")
        for field in ("entity_identity", "ordering", "visible_text", "speech_and_sound"):
            if verdict[field] not in OBSERVATION_STATES:
                raise ValueError("map observation state is invalid")
        allowed_evidence = PILOT_ANSWER_EVIDENCE if phase == "pilot" else SCREEN_ANSWER_EVIDENCE
        if verdict["answer_critical_evidence"] not in allowed_evidence:
            raise ValueError("answer-critical evidence state is invalid")
        if verdict["overall_decision"] not in MAP_DECISIONS:
            raise ValueError("map overall decision is invalid")
        _reject_reference_fields(verdict)


def _validate_answer_verdicts(
    rows: list[Mapping[str, Any]],
    *,
    expected_ids: list[str],
    packet: AuditPacket,
    phase: str,
    require_complete: bool,
) -> None:
    if phase == "pilot":
        if rows:
            raise ValueError("pilot audit must not contain answer verdicts")
        return
    by_key = _unique_by(
        rows,
        lambda row: f"{row.get('video_id')}:{row.get('condition_id')}",
        "answer verdicts",
    )
    expected_keys = {
        f"{video_id}:{condition}"
        for video_id in expected_ids
        for condition in ANSWER_CONDITIONS
    }
    if require_complete and set(by_key) != expected_keys:
        raise ValueError("screen audit requires fourteen complete answer verdicts")
    if set(by_key) - expected_keys:
        raise ValueError("answer verdict contains an unknown row or condition")
    for verdict in by_key.values():
        _validate_verdict_keys(
            verdict,
            {
                "video_id",
                "condition_id",
                "question",
                "frozen_answer",
                "claim_checks",
                "unsupported_detail",
                "failure_causes",
                "supported_top1",
                "notes",
            },
            "answer verdict",
        )
        if verdict["condition_id"] not in ANSWER_CONDITIONS:
            raise ValueError("answer condition is invalid")
        if not isinstance(verdict["question"], str) or not verdict["question"].strip():
            raise ValueError("answer verdict question is missing")
        if not isinstance(verdict["frozen_answer"], str) or not verdict["frozen_answer"].strip():
            raise ValueError("answer verdict frozen answer is missing")
        claims = verdict["claim_checks"]
        if not isinstance(claims, list):
            raise ValueError("answer claim_checks must be a list")
        answer = _aware_answer(packet, str(verdict["video_id"]), str(verdict["condition_id"]))
        expected_claims = _claim_ids(answer)
        claim_rows = _unique_by(claims, "claim_id", "answer claim checks")
        if require_complete and set(claim_rows) != set(expected_claims):
            raise ValueError("answer verdict must contain every frozen answer claim")
        if set(claim_rows) - set(expected_claims):
            raise ValueError("answer verdict contains an unknown claim")
        for claim in claims:
            _validate_verdict_keys(
                claim,
                {
                    "claim_id",
                    "native_span_support",
                    "guided_map_event_citation",
                    "notes",
                },
                "answer claim verdict",
            )
            if claim["native_span_support"] not in CLAIM_SUPPORT_STATES:
                raise ValueError("native span support state is invalid")
            if claim["guided_map_event_citation"] not in CLAIM_CITATION_STATES:
                raise ValueError("guided map citation state is invalid")
        if not isinstance(verdict["unsupported_detail"], bool):
            raise ValueError("unsupported_detail must be boolean")
        if not isinstance(verdict["failure_causes"], list) or not all(
            isinstance(value, str) and value.strip()
            for value in verdict["failure_causes"]
        ):
            raise ValueError("failure_causes must be a list of text")
        if not isinstance(verdict["supported_top1"], bool):
            raise ValueError("supported_top1 must be boolean")
        _reject_reference_fields(verdict)


def _event_ids_for_video(packet: AuditPacket, video_id: str) -> list[str]:
    manifest = _read_json(packet.aware_manifest)
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("aware manifest rows are missing")
    for row in rows:
        if isinstance(row, Mapping) and row.get("video_id") == video_id:
            map_payload = row.get("map")
            events = map_payload.get("events") if isinstance(map_payload, Mapping) else None
            if isinstance(events, list):
                return [
                    _required_text(event.get("event_id"), "event_id")
                    for event in events
                    if isinstance(event, Mapping)
                ]
    raise ValueError("audit video is missing from aware manifest")


def _aware_answer(
    packet: AuditPacket, video_id: str, condition: str
) -> Mapping[str, Any]:
    manifest = _read_json(packet.aware_manifest)
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("aware manifest rows are missing")
    for row in rows:
        if isinstance(row, Mapping) and row.get("video_id") == video_id:
            answers = row.get("frozen_answers")
            if isinstance(answers, Mapping) and isinstance(answers.get(condition), Mapping):
                return answers[condition]
    raise ValueError("answer condition is missing from aware manifest")


def _claim_ids(answer: Mapping[str, Any]) -> list[str]:
    claims = answer.get("claims")
    if not isinstance(claims, list):
        raise ValueError("frozen answer claims are missing")
    return [
        _required_text(claim.get("claim_id"), "claim_id")
        for claim in claims
        if isinstance(claim, Mapping)
    ]


def _prepare_output_root(output: Path) -> Path:
    root = _canonical_root()
    if output.is_symlink():
        raise ValueError("audit output cannot be a symlink")
    resolved = output.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("audit output must resolve beneath canonical private data root")
    _reject_symlink_components(root, resolved)
    if output.exists() and not output.is_dir():
        raise FileExistsError("audit output is not a directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("audit output already contains artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_audit_root(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("audit root must be a regular directory")
    resolved = root.resolve(strict=True)
    canonical = _canonical_root()
    if not resolved.is_relative_to(canonical):
        raise ValueError("audit root must be beneath canonical private data root")
    _reject_symlink_components(canonical, resolved)
    return resolved


def _canonical_root() -> Path:
    root = Path(CANONICAL_DATA_ROOT)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("canonical private data root is not a regular directory")
    return root.resolve(strict=True)


def _reject_symlink_components(root: Path, target: Path) -> None:
    relative = target.relative_to(root)
    current = root
    if current.is_symlink():
        raise ValueError("canonical private data root cannot be a symlink")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("audit path contains a symlink escape")


def _safe_packet_path(root: Path, value: object, field: str) -> Path:
    relative = _required_text(value, field)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} escapes audit root")
    resolved = (root / candidate).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{field} escapes audit root")
    _reject_symlink_components(root, resolved)
    return resolved


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing audit artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid audit JSON: {path.name}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"audit JSON must be an object: {path.name}")
    return payload


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing {label} file")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid {label} line {line_number}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(payload)
    return rows


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {path.name}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _create_empty(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(tool: str) -> str:
    try:
        completed = subprocess.run(
            [tool, "-version"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"{tool} is required for audit media") from error
    line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if not line:
        raise ValueError(f"{tool} did not report a version")
    return line


def _result_hash(result: Mapping[str, Any]) -> str:
    value = result.get("result_sha256")
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("prediction result hash is invalid")
    return value


def _request_hash(result: Mapping[str, Any]) -> str:
    value = result.get("request_sha256")
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("prediction request hash is invalid")
    return value


def _expected_video_ids(packet_manifest: Mapping[str, Any]) -> list[str]:
    rows = packet_manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("packet manifest rows are missing")
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("packet row must be an object")
        ids.append(_required_text(row.get("video_id"), "video_id"))
    return ids


def _unique_by(
    rows: Sequence[Mapping[str, Any]], key: str | Any, label: str
) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} rows must be objects")
        value = key(row) if callable(key) else row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} identity is missing")
        if value in found:
            raise ValueError(f"duplicate {label} identity")
        found[value] = row
    return found


def _validate_verdict_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the strict schema")


def _reject_reference_fields(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in REFERENCE_KEYS or "reference" in normalized:
                raise ValueError(
                    f"reference field crossed audit boundary at {location}.{key}"
                )
            _reject_reference_fields(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_reference_fields(nested, f"{location}[{index}]")


def _reject_blind_leakage(value: object, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in BLIND_KEYS:
                raise ValueError(
                    f"blind view contains private field at {location}.{key}"
                )
            _reject_blind_leakage(nested, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_blind_leakage(nested, f"{location}[{index}]")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _required_sha256(value: object, field: str) -> str:
    text = _required_text(value, field)
    if not SHA256.fullmatch(text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


__all__ = [
    "AUDIT_ID",
    "AuditPacket",
    "CANONICAL_DATA_ROOT",
    "build_audit_packet",
    "freeze_pilot_audit",
    "freeze_screen_audit",
    "validate_audit_packet",
]
