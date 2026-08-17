#!/usr/bin/env python3
"""Six-row Gemini compatibility rescue using one uniform H.264/AAC MP4 arm.

This is intentionally separate from both the original 2,443-row run and the
first 27-row rescue.  It is dry-run by default and never selects from labels,
references, metrics, or prior answer text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_vqa_deadline_gemini_full as full  # noqa: E402


ORIGINAL_FREEZE_PATH = (
    _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-full-20260816/gemini/predictions-freeze.json"
)
FIRST_RESCUE_ROOT = (
    _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-rescue-20260817/gemini-27"
)
FIRST_RESCUE_FREEZE_PATH = FIRST_RESCUE_ROOT / "predictions-freeze.json"
FIRST_RESCUE_MANIFEST_PATH = FIRST_RESCUE_ROOT / "manifest.json"
REQUESTS_PATH = _REPO_ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
MEDIA_ROOT = _REPO_ROOT / "data/vqa/2026/testing"
PRIVATE_RUN_ROOT = (
    _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-rescue-20260817/gemini-27-mp4"
)

ORIGINAL_FREEZE_SHA256 = (
    "e3f0e4cf6d9dd65986f328f3834e96b753b180b3b6123a646418ec1424078cb0"
)
FIRST_RESCUE_FREEZE_SHA256 = (
    "445b58ed61124583b128dd0db7724e43f8b8de3bf982def47edcd6d3ced6e940"
)
FIRST_RESCUE_MANIFEST_SHA256 = (
    "5dbce4878c88f5e629de016b47362ef63e6a110c24c11f2a7db294a1ca731391"
)
EXPECTED_QUERY_IDS = (175, 310, 886, 1563, 2279, 2404)
EXPECTED_PANEL_COUNT = 2_443
MAX_OUTPUT_TOKENS = 8_192
MAX_ESTIMATED_COST_USD = 0.50
MAX_OBSERVED_COST_USD = 0.50
UPLOAD_WORKERS = 1
PROVIDER_RETRY_POLICY = {
    "max_retries": 1,
    "code": 503,
    "backoff_seconds": 10,
    "workers": UPLOAD_WORKERS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("_artifact_sha256", None)
    return body


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        existing = full._read_json(path)
        if _artifact_body(existing) != dict(value):
            raise ValueError(f"immutable artifact changed: {path.name}")
        return _sha256_file(path)
    return full._write_json(path, value)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_cap(value: float, *, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or float(value) > MAX_ESTIMATED_COST_USD
    ):
        raise ValueError(f"{name} cap must be finite, nonnegative, and no greater than $0.50")


def _read_bound_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"approved artifact hash changed: {path.name}")
    return full._read_json(path)


def select_compatibility_rows(
    *,
    original_freeze_path: Path = ORIGINAL_FREEZE_PATH,
    first_rescue_freeze_path: Path = FIRST_RESCUE_FREEZE_PATH,
    first_rescue_manifest_path: Path = FIRST_RESCUE_MANIFEST_PATH,
    requests_path: Path = REQUESTS_PATH,
    media_root: Path = MEDIA_ROOT,
) -> list[dict[str, Any]]:
    """Select exactly the six first-rescue empty responses and bind inputs."""

    original = _read_bound_artifact(original_freeze_path, ORIGINAL_FREEZE_SHA256)
    if original.get("status") != "predictions_frozen":
        raise ValueError("original predictions freeze is not terminal")
    first_freeze = _read_bound_artifact(
        first_rescue_freeze_path, FIRST_RESCUE_FREEZE_SHA256
    )
    first_manifest = _read_bound_artifact(
        first_rescue_manifest_path, FIRST_RESCUE_MANIFEST_SHA256
    )
    if first_freeze.get("base_predictions_freeze_sha256") != ORIGINAL_FREEZE_SHA256:
        raise ValueError("first rescue is not bound to the original freeze")
    if first_freeze.get("manifest_sha256") != FIRST_RESCUE_MANIFEST_SHA256:
        raise ValueError("first rescue freeze is not bound to its manifest")
    if first_manifest.get("base_predictions_freeze_sha256") != ORIGINAL_FREEZE_SHA256:
        raise ValueError("first rescue manifest has the wrong original freeze binding")
    requests_path = Path(requests_path).resolve(strict=True)
    if first_manifest.get("input", {}).get("sha256") != _sha256_file(requests_path):
        raise ValueError("first rescue manifest is not bound to the requests input")

    frozen_rows = first_freeze.get("rows")
    manifest_rows = first_manifest.get("rows")
    if not isinstance(frozen_rows, list) or not isinstance(manifest_rows, list):
        raise ValueError("first rescue artifacts have no row bindings")
    selected_frozen = [
        row
        for row in frozen_rows
        if isinstance(row, Mapping) and row.get("failure_reason") == "empty_text"
    ]
    selected_ids = [row.get("query_id") for row in selected_frozen]
    if selected_ids != list(EXPECTED_QUERY_IDS):
        raise ValueError("compatibility selector did not reproduce the six approved IDs")
    if any(row.get("status") != "failed" for row in selected_frozen):
        raise ValueError("compatibility rows are not terminal failed rows")

    loaded = full.load_input_rows(
        requests_path, media_root=Path(media_root), expected_count=EXPECTED_PANEL_COUNT
    )
    loaded_by_id = {int(row["query_id"]): row for row in loaded}
    manifest_by_id = {
        int(row["query_id"]): row
        for row in manifest_rows
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), int)
    }
    selected: list[dict[str, Any]] = []
    for frozen in selected_frozen:
        query_id = int(frozen["query_id"])
        source = loaded_by_id.get(query_id)
        bound = manifest_by_id.get(query_id)
        if source is None or bound is None:
            raise ValueError(f"missing source binding for query {query_id}")
        for key in (
            "query_id",
            "video_id",
            "question",
            "availability",
            "source_media_sha256",
            "source_media_size_bytes",
            "duration_seconds",
        ):
            if source.get(key) != bound.get(key):
                raise ValueError(f"first rescue source binding changed for query {query_id}: {key}")
        if frozen.get("video_id") != source.get("video_id"):
            raise ValueError(f"first rescue video binding changed for query {query_id}")
        source_path = bound.get("source_media_path")
        if not isinstance(source_path, str) or not Path(source_path).is_file():
            raise ValueError(f"source media is unavailable for query {query_id}")
        if _sha256_file(Path(source_path)) != bound.get("source_media_sha256"):
            raise ValueError(f"source media hash changed for query {query_id}")
        row = dict(source)
        row["compatibility_source_failure_reason"] = "empty_text"
        row["compatibility_source_media_path"] = source_path
        row["compatibility_source_media_sha256"] = bound["source_media_sha256"]
        row["compatibility_source_media_size_bytes"] = bound["source_media_size_bytes"]
        selected.append(row)
    return selected


def _tool_version(tool: str) -> str:
    result = subprocess.run(
        [tool, "-version"], check=True, capture_output=True, text=True
    )
    output = (result.stdout or result.stderr).strip()
    if not output:
        raise ValueError(f"{tool} returned no version")
    return output.splitlines()[0]


def _probe_media(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError(f"ffprobe returned no streams: {path.name}")
    video = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    audio = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    try:
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ffprobe returned no duration: {path.name}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"ffprobe duration is invalid: {path.name}")
    return {
        "duration_seconds": duration,
        "video_codec": video.get("codec_name") if isinstance(video, Mapping) else None,
        "audio_codec": audio.get("codec_name") if isinstance(audio, Mapping) else None,
        "stream_count": len(streams),
    }


def _transcode_one(
    *, row: Mapping[str, Any], delivery_root: Path, ffmpeg_version: str, ffprobe_version: str
) -> dict[str, Any]:
    source = Path(str(row["compatibility_source_media_path"])).resolve(strict=True)
    expected_source_sha256 = str(row["compatibility_source_media_sha256"])
    if _sha256_file(source) != expected_source_sha256:
        raise ValueError(f"source media hash changed: {source.name}")
    delivery_root.mkdir(parents=True, exist_ok=True)
    video_id = str(row["video_id"])
    if not full._SAFE_NAME.fullmatch(video_id):
        raise ValueError(f"unsafe video_id: {video_id}")
    delivery = delivery_root / f"{video_id}.mp4"
    if delivery.exists() or delivery.is_symlink():
        raise ValueError(f"refusing to overwrite MP4 delivery: {delivery.name}")
    # Keep an .mp4 suffix so ffmpeg selects the MP4 muxer for the atomic temp.
    temporary = delivery.with_name(f".{delivery.stem}.partial.mp4")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"partial MP4 delivery already exists: {temporary.name}")
    source_probe = _probe_media(source)
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not full._is_regular_file(temporary):
            raise ValueError(f"ffmpeg produced no MP4 delivery: {delivery.name}")
        os.replace(temporary, delivery)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    output_probe = _probe_media(delivery)
    if output_probe["video_codec"] != "h264" or output_probe["audio_codec"] != "aac":
        raise ValueError(f"MP4 codecs are not H.264/AAC: {delivery.name}")
    if abs(output_probe["duration_seconds"] - source_probe["duration_seconds"]) > max(
        1.0, source_probe["duration_seconds"] * 0.05
    ):
        raise ValueError(f"MP4 duration changed unexpectedly: {delivery.name}")
    return {
        "video_id": video_id,
        "source_path": str(source),
        "source_sha256": expected_source_sha256,
        "source_size_bytes": source.stat().st_size,
        "source_probe": source_probe,
        "delivery_path": str(delivery),
        "delivery_sha256": _sha256_file(delivery),
        "delivery_size_bytes": delivery.stat().st_size,
        "delivery_mime_type": "video/mp4",
        "output_probe": output_probe,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "command": command,
    }


def transcode_rows(rows: Sequence[Mapping[str, Any]], delivery_root: Path) -> list[dict[str, Any]]:
    """Transcode each unique source exactly once and return immutable provenance."""

    ffmpeg_version = _tool_version("ffmpeg")
    ffprobe_version = _tool_version("ffprobe")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        source_path = str(row["compatibility_source_media_path"])
        if source_path in seen:
            continue
        seen.add(source_path)
        records.append(
            _transcode_one(
                row=row,
                delivery_root=Path(delivery_root),
                ffmpeg_version=ffmpeg_version,
                ffprobe_version=ffprobe_version,
            )
        )
    return records


def _freeze_transcode_records(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Create the evidence directory before writing immutable per-video records."""

    output_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        _write_json_once(output_root / f"{record['video_id']}.json", record)


def _mp4_rows(rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source = {str(record["source_path"]): record for record in records}
    converted: list[dict[str, Any]] = []
    for source_row in rows:
        record = by_source[str(source_row["compatibility_source_media_path"])]
        row = dict(source_row)
        row["compatibility_delivery"] = dict(record)
        row["source_media_path"] = record["delivery_path"]
        row["source_media_sha256"] = record["delivery_sha256"]
        row["source_media_size_bytes"] = record["delivery_size_bytes"]
        row["media_path"] = record["delivery_path"]
        row["media_sha256"] = record["delivery_sha256"]
        row["media_size_bytes"] = record["delivery_size_bytes"]
        row["media_mime_type"] = "video/mp4"
        row["delivery_plan"] = {
            "delivery_mode": "source",
            "delivery_suffix": ".mp4",
            "delivery_mime_type": "video/mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
        }
        converted.append(row)
    return converted


def build_compatibility_request(*, row: Mapping[str, Any], upload: Mapping[str, Any]) -> Any:
    question = row.get("question")
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload URI is required")
    if mime_type != "video/mp4":
        raise ValueError("compatibility arm requires video/mp4")
    video = full.types.Part(
        file_data=full.types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=full.types.VideoMetadata(fps=4.0),
        media_resolution=full.types.PartMediaResolution(level=full.MEDIA_RESOLUTION),
    )
    prompt = full.types.Part(text=f"{full.PROMPT}\n\nQuestion: {question.strip()}")
    return full.types.InlinedRequest(
        model=full.MODEL,
        contents=[full.types.Content(role="user", parts=[video, prompt])],
        metadata={
            "arm": "compatibility_mp4_37flash_lowres_4fps_highthink_8192",
            "row_id": str(row["query_id"]),
            "video_id": str(row["video_id"]),
            "mime_type": mime_type,
            "video_fps": "4.0",
            "media_resolution": full.MEDIA_RESOLUTION.value,
            "thinking_level": full.THINKING_LEVEL.value,
            "max_output_tokens": str(MAX_OUTPUT_TOKENS),
        },
        config=full.types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
            seed=42,
            thinking_config=full.types.ThinkingConfig(
                include_thoughts=False, thinking_level=full.THINKING_LEVEL
            ),
        ),
    )


@contextmanager
def _compatibility_protocol() -> Iterator[None]:
    saved = {
        "MAX_OUTPUT_TOKENS": full.MAX_OUTPUT_TOKENS,
        "UPLOAD_WORKERS": full.UPLOAD_WORKERS,
        "PROVIDER_RETRY_POLICY": full.PROVIDER_RETRY_POLICY,
        "build_request": full.build_request,
    }
    full.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
    full.UPLOAD_WORKERS = UPLOAD_WORKERS
    full.PROVIDER_RETRY_POLICY = dict(PROVIDER_RETRY_POLICY)
    full.build_request = build_compatibility_request
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(full, key, value)


def _write_manifest(
    *, output_root: Path, rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
    estimate: Mapping[str, Any], requests_path: Path, created_at: str
) -> str:
    body = {
        "schema_version": 1,
        "experiment_id": "deadline-submission-rescue-20260817-gemini-27-mp4",
        "status": "inputs_frozen",
        "created_at": created_at,
        "original_predictions_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "first_rescue_predictions_freeze_sha256": FIRST_RESCUE_FREEZE_SHA256,
        "first_rescue_manifest_sha256": FIRST_RESCUE_MANIFEST_SHA256,
        "input": {"path": str(Path(requests_path).resolve()), "sha256": _sha256_file(Path(requests_path))},
        "model": full.MODEL,
        "prompt": full.PROMPT,
        "model_settings": {
            "video_fps": 4.0,
            "media_resolution": full.MEDIA_RESOLUTION.value,
            "thinking_level": full.THINKING_LEVEL.value,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "provider_retry_policy": dict(PROVIDER_RETRY_POLICY),
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "row_count": len(rows),
        "unique_video_count": len(records),
        "cost": dict(estimate),
        "max_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        "max_observed_cost_usd": MAX_OBSERVED_COST_USD,
        "transcodes": [dict(record) for record in records],
        "rows": [
            {
                key: row[key]
                for key in (
                    "query_id",
                    "video_id",
                    "question",
                    "duration_seconds",
                    "compatibility_source_media_path",
                    "compatibility_source_media_sha256",
                    "compatibility_source_media_size_bytes",
                    "compatibility_source_failure_reason",
                )
                if key in row
            }
            for row in rows
        ],
        "provider_artifacts_root": str((output_root / "provider-artifacts").resolve()),
    }
    return _write_json_once(output_root / "manifest.json", body)


def _estimate_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = sum(float(row.get("duration_seconds", 0.0)) for row in rows)
    prompt_tokens = math.ceil(seconds * full.ESTIMATED_INPUT_TOKENS_PER_SECOND) + len(rows) * full.ESTIMATED_QUESTION_TOKENS
    output_tokens = len(rows) * MAX_OUTPUT_TOKENS
    cost = prompt_tokens / 1_000_000 * full.INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * full.OUTPUT_USD_PER_MILLION
    return {
        "duration_seconds": round(seconds, 3),
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def _finalize(
    *, output_root: Path, rows: Sequence[Mapping[str, Any]], estimate: Mapping[str, Any],
    rescue_manifest_sha256: str
) -> dict[str, Any]:
    core_root = output_root / "provider-artifacts"
    core = full._read_json(core_root / "predictions-freeze.json")
    if core.get("status") != "predictions_frozen":
        raise ValueError("compatibility provider artifacts did not reach predictions_frozen")
    allowed = {"succeeded", "failed", "upload_failed", "unavailable", "unsupported", "cost_cap_blocked"}
    if any(row.get("status") not in allowed for row in core.get("rows", [])):
        raise ValueError("compatibility freeze contains a non-terminal row")
    predictions = {
        **_artifact_body(core),
        "experiment_id": "deadline-submission-rescue-20260817-gemini-27-mp4",
        "manifest_sha256": rescue_manifest_sha256,
        "original_predictions_freeze_sha256": ORIGINAL_FREEZE_SHA256,
        "first_rescue_predictions_freeze_sha256": FIRST_RESCUE_FREEZE_SHA256,
        "first_rescue_manifest_sha256": FIRST_RESCUE_MANIFEST_SHA256,
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "delivery_codec": {"container": "mp4", "video": "h264", "audio": "aac"},
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "estimated_cost_cap_usd": MAX_ESTIMATED_COST_USD,
        "observed_cost_cap_usd": MAX_OBSERVED_COST_USD,
        "provider_retry_policy": dict(PROVIDER_RETRY_POLICY),
    }
    prediction_sha256 = _write_json_once(output_root / "predictions-freeze.json", predictions)
    summary = {
        "schema_version": 1,
        "experiment_id": predictions["experiment_id"],
        "status": predictions["status"],
        "row_count": len(rows),
        "completed_count": predictions.get("completed_count", 0),
        "failed_count": sum(row.get("status") == "failed" for row in predictions["rows"]),
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "estimated_cost_cap_usd": MAX_ESTIMATED_COST_USD,
        "observed_cost_usd": predictions.get("observed_cost_usd"),
        "observed_cost_cap_usd": MAX_OBSERVED_COST_USD,
        "predictions_freeze_sha256": prediction_sha256,
        "provider_predictions_freeze_sha256": _sha256_file(core_root / "predictions-freeze.json"),
    }
    _write_json_once(output_root / "summary.json", summary)
    return summary


def run_compatibility(
    *,
    original_freeze_path: Path = ORIGINAL_FREEZE_PATH,
    first_rescue_freeze_path: Path = FIRST_RESCUE_FREEZE_PATH,
    first_rescue_manifest_path: Path = FIRST_RESCUE_MANIFEST_PATH,
    requests_path: Path = REQUESTS_PATH,
    media_root: Path = MEDIA_ROOT,
    output_root: Path = PRIVATE_RUN_ROOT,
    client: Any | None = None,
    dry_run: bool | None = None,
    run: bool = False,
    max_estimated_cost_usd: float = MAX_ESTIMATED_COST_USD,
    max_observed_cost_usd: float = MAX_OBSERVED_COST_USD,
    sleep: Any = full.time.sleep,
) -> dict[str, Any]:
    if dry_run is None:
        dry_run = not run
    if dry_run and run:
        raise ValueError("dry_run and run are mutually exclusive")
    if not dry_run and not run:
        raise ValueError("live compatibility rescue requires run=True")
    _validate_cap(max_estimated_cost_usd, name="max_estimated_cost_usd")
    _validate_cap(max_observed_cost_usd, name="max_observed_cost_usd")
    rows = select_compatibility_rows(
        original_freeze_path=original_freeze_path,
        first_rescue_freeze_path=first_rescue_freeze_path,
        first_rescue_manifest_path=first_rescue_manifest_path,
        requests_path=requests_path,
        media_root=media_root,
    )
    estimate = _estimate_cost(rows)
    if estimate["estimated_cost_usd"] > max_estimated_cost_usd:
        raise ValueError(f"estimated cost {estimate['estimated_cost_usd']:.8f} exceeds cap")
    if dry_run:
        # Exercise the exact delivery conversion in an ephemeral directory.
        # This validates codecs, duration, atomic publication, and hashes while
        # leaving the private run root untouched and making no provider call.
        with tempfile.TemporaryDirectory(prefix="trec-vqa-gemini-mp4-dry-run-") as temp_root:
            records = transcode_rows(rows, Path(temp_root))
        return {
            "status": "dry_run",
            "row_count": len(rows),
            "unique_video_count": len({str(row["video_id"]) for row in rows}),
            "request_count": len(rows),
            "upload_count": len(rows),
            "transcode_count": len(records),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "delivery_codec": {"container": "mp4", "video": "h264", "audio": "aac"},
            "estimated_cost_cap_usd": max_estimated_cost_usd,
            "observed_cost_cap_usd": max_observed_cost_usd,
            **estimate,
        }
    if client is None:
        raise ValueError("provider client is required for live compatibility rescue")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    created_at = _now()
    records = transcode_rows(rows, output_root / "deliveries")
    _freeze_transcode_records(output_root / "transcodes", records)
    converted = _mp4_rows(rows, records)
    rescue_manifest_sha256 = _write_manifest(
        output_root=output_root,
        rows=converted,
        records=records,
        estimate=estimate,
        requests_path=Path(requests_path),
        created_at=created_at,
    )
    with _compatibility_protocol():
        full._run_live(
            rows=converted,
            requests_path=Path(requests_path).resolve(),
            output_root=output_root / "provider-artifacts",
            client=client,
            estimate=estimate,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_observed_cost_usd=max_observed_cost_usd,
            created_at=created_at,
            sleep=sleep,
            parser=full.parse_response,
        )
    return _finalize(
        output_root=output_root,
        rows=converted,
        estimate=estimate,
        rescue_manifest_sha256=rescue_manifest_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_gemini_rescue_mp4")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--original-freeze", type=Path, default=ORIGINAL_FREEZE_PATH)
    parser.add_argument("--first-rescue-freeze", type=Path, default=FIRST_RESCUE_FREEZE_PATH)
    parser.add_argument("--first-rescue-manifest", type=Path, default=FIRST_RESCUE_MANIFEST_PATH)
    parser.add_argument("--requests", type=Path, default=REQUESTS_PATH)
    parser.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=PRIVATE_RUN_ROOT)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=MAX_ESTIMATED_COST_USD)
    parser.add_argument("--max-observed-cost-usd", type=float, default=MAX_OBSERVED_COST_USD)
    return parser


def main(argv: Sequence[str] | None = None, *, client_factory_override: Any = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.run and args.dry_run:
            raise ValueError("--run and --dry-run are mutually exclusive")
        live = bool(args.run)
        client = None if not live else (client_factory_override or full.client_factory)(api_key=full._private_api_key())
        result = run_compatibility(
            original_freeze_path=args.original_freeze,
            first_rescue_freeze_path=args.first_rescue_freeze,
            first_rescue_manifest_path=args.first_rescue_manifest,
            requests_path=args.requests,
            media_root=args.media_root,
            output_root=args.output,
            client=client,
            dry_run=not live,
            run=live,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_observed_cost_usd=args.max_observed_cost_usd,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
