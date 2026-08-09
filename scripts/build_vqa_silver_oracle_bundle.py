#!/usr/bin/env python3
"""Build the private, checksum-bound full-panel silver-oracle selector input."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from trec_video_2026.vqa_experiments.silver_oracle import (
    INTERVAL_DURATION_BUDGET_FORMULA,
    parse_timestamped_whisper,
    protocol_sha256,
)

RUN_ID = "vqa-silver-oracle-election-20260809-b4"
EXPECTED_COUNT = 475
EXPECTED_IDS_SHA256 = "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
SELECTOR_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
SELECTOR_MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
SELECTOR_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@"
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
SELECTOR_PROMPT = (
    "You are selecting evidence for a video question. You may use the provided "
    "reference answer only to find where the supporting evidence occurs. Return "
    "one terminal JSON object and no prose. The object must have exactly: status "
    "(localized or non_localizable), intervals, and whisper_segment_ids. Return "
    "no more than two intervals; they must be sorted, non-overlapping, inside the "
    "listed video range, and their combined interval duration must not exceed the "
    "listed limit. Return no more than three sorted unique Whisper segment IDs "
    "inside the listed ID range. For non_localizable, return both arrays empty. "
    'Valid localized JSON shape (example times only; do not copy them): {"status":'
    '"localized","intervals":[{"start_seconds":0.0,"end_seconds":1.0}],'
    '"whisper_segment_ids":[]}. '
    "Do not return an answer, caption, rationale, confidence, hint, rewritten "
    "transcript, or any additional key."
)
REPAIR_PROMPT = (
    "Your previous response did not match the required JSON schema. Re-evaluate "
    "the same evidence and return only one valid terminal JSON object with exactly "
    "status, intervals, and whisper_segment_ids. Return no more than two intervals "
    "and obey the listed range, ordering, non-overlap, and combined interval "
    "duration limit. Return no more than three Whisper segment IDs and obey the "
    "listed ID range, uniqueness, and ordering. Valid localized JSON shape "
    '(example times only; do not copy them): {"status":"localized","intervals":'
    '[{"start_seconds":0.0,"end_seconds":1.0}],"whisper_segment_ids":[]}. '
    "Do not add prose or any other key."
)
RESERVED_OUTPUTS = (
    "selector-requests.jsonl",
    "selector-canary.jsonl",
    "selector-chunk-00.jsonl",
    "selector-chunk-01.jsonl",
    "selector-chunk-02.jsonl",
    "selector-chunk-03.jsonl",
    "media-manifest.json",
    "protocol.json",
    "summary.json",
    "SHA256SUMS",
)
STRATUM_TARGETS = (
    ("temporal", 2),
    ("counting", 2),
    ("audio_text", 2),
    ("causal", 1),
    ("ocr_detail", 1),
    ("other", 2),
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _id_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ids = [row.get("video_id") for row in rows]
    if any(not isinstance(video_id, str) or not video_id for video_id in ids):
        raise ValueError("every row must have a nonempty video_id")
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("panel video IDs must be unique and sorted")
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _ordered_id_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ids = [row.get("video_id") for row in rows]
    if any(not isinstance(video_id, str) or not video_id for video_id in ids):
        raise ValueError("every request row must have a nonempty video_id")
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _indexed_rows(
    payload: Mapping[str, Any], field: str, *, source: str
) -> dict[str, Mapping[str, Any]]:
    rows = payload.get(field)
    if not isinstance(rows, list):
        raise ValueError(f"{source} {field} must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} row {index} must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{source} row {index} has invalid video_id")
        if video_id in indexed:
            raise ValueError(f"{source} contains duplicate video_id {video_id}")
        indexed[video_id] = row
    return indexed


def question_stratum(question: str) -> str:
    """Classify one question using only frozen lexical rules."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    text = question.casefold()
    if re.search(
        r"\b(after|before|then|first|last|when|end|begin|next|while|change|happen)\w*\b",
        text,
    ):
        return "temporal"
    if re.search(r"\b(how many|count|number of)\b", text):
        return "counting"
    if re.search(
        r"\b(say|says|saying|hear|heard|sound|sing|song|called|name|word)\w*\b", text
    ):
        return "audio_text"
    if re.search(r"\b(why|cause|causes|caused|makes|result|reason)\b", text):
        return "causal"
    if re.search(r"\b(written|text|sign|color|brand|label|read|wearing)\b", text):
        return "ocr_detail"
    return "other"


def select_canary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select the fixed 2/2/2/1/1/2 question-only operational canary."""

    classified = [
        {
            **dict(row),
            "question_stratum": question_stratum(str(row.get("question", ""))),
        }
        for row in rows
    ]
    selected: list[dict[str, Any]] = []
    for stratum, required in STRATUM_TARGETS:
        candidates = [row for row in classified if row["question_stratum"] == stratum]
        if len(candidates) < required:
            raise ValueError(f"canary stratum {stratum} has fewer than {required} rows")
        selected.extend(candidates[:required])
    return selected


def select_resource_sentinel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the maximum native-video/input burden without consulting labels."""

    candidates: list[tuple[tuple[float, int, str], dict[str, Any]]] = []
    for row in rows:
        video_id = row.get("video_id")
        duration = row.get("duration_seconds")
        segments = row.get("whisper_segments")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("resource sentinel row has invalid video_id")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            raise ValueError("resource sentinel row has invalid duration")
        if not isinstance(segments, list):
            raise ValueError("resource sentinel row has invalid Whisper segments")
        candidates.append(((float(duration), len(segments), video_id), dict(row)))
    if not candidates:
        raise ValueError("resource sentinel selection requires at least one row")
    return max(candidates, key=lambda candidate: candidate[0])[1]


def partition_rows(
    rows: Sequence[Mapping[str, Any]], *, chunk_count: int
) -> list[list[dict[str, Any]]]:
    """Partition rows contiguously with at most one row of size imbalance."""

    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
    ):
        raise ValueError("chunk_count must be a positive integer")
    base, remainder = divmod(len(rows), chunk_count)
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(chunk_count):
        size = base + (1 if index < remainder else 0)
        chunks.append([dict(row) for row in rows[start : start + size]])
        start += size
    return chunks


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_bytes(_jsonl_bytes(rows))


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _request_file_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = _jsonl_bytes(rows)
    return {
        "row_count": len(rows),
        "ids_sha256": _ordered_id_digest(rows),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _request_row(
    manifest_row: Mapping[str, Any],
    evidence_row: Mapping[str, Any],
    whisper_row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    media_dir: Path,
    *,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    video_id = str(manifest_row["video_id"])
    question = manifest_row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{video_id} manifest question must be nonempty")
    if (
        evidence_row.get("question") != question
        or annotation.get("question") != question
    ):
        raise ValueError(
            f"{video_id} question mismatch across manifest/evidence/annotation"
        )
    reference = annotation.get("correct_answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{video_id} annotation reference must be nonempty")
    duration = evidence_row.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        raise ValueError(f"{video_id} evidence duration must be positive")
    media_sha = evidence_row.get("media_sha256")
    if (
        not isinstance(media_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", media_sha) is None
    ):
        raise ValueError(f"{video_id} evidence media SHA-256 is invalid")
    evidence_media_path = evidence_row.get("media_path")
    if not isinstance(evidence_media_path, str):
        raise ValueError(f"{video_id} evidence media_path must be text")
    relative_media = PurePosixPath(evidence_media_path)
    if (
        relative_media.is_absolute()
        or ".." in relative_media.parts
        or relative_media.stem != video_id
        or relative_media.suffix.casefold() not in {".mkv", ".mp4", ".webm"}
    ):
        raise ValueError(f"{video_id} evidence media_path is unsafe or mismatched")
    media_basename = relative_media.name
    media_path = media_dir / media_basename
    if not media_path.is_file() or media_path.is_symlink():
        raise ValueError(f"{video_id} media must be a regular non-symlink file")
    actual_media_sha = _sha256_file(media_path)
    if actual_media_sha != media_sha:
        raise ValueError(f"{video_id} media hash mismatch")
    if whisper_row.get("status") != "passed":
        raise ValueError(f"{video_id} Whisper row did not complete")
    selected_transcript = (
        whisper_row.get("selected_transcript")
        if whisper_row.get("transcript_included") is True
        else None
    )
    if selected_transcript is not None and not isinstance(selected_transcript, str):
        raise ValueError(f"{video_id} selected Whisper transcript must be text or null")
    segments = parse_timestamped_whisper(
        selected_transcript,
        duration_seconds=float(duration),
        drop_trailing_overrun=True,
    )
    source_line_count = (
        len(selected_transcript.splitlines()) if selected_transcript else 0
    )
    whisper_segments = [
        {
            "segment_id": segment.segment_id,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "source_line": segment.source_line,
        }
        for segment in segments
    ]
    request = {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": video_id,
        "video_path": f"media/{media_basename}",
        "media_sha256": media_sha,
        "duration_seconds": float(duration),
        "question": question.strip(),
        "question_stratum": question_stratum(question),
        "reference": reference.strip(),
        "whisper_segments": whisper_segments,
        "whisper_dropped_trailing_line_count": source_line_count - len(segments),
        "whisper_sha256": hashlib.sha256(
            (selected_transcript or "").encode("utf-8")
        ).hexdigest(),
    }
    request["request_sha256"] = _canonical_sha256(request)
    media_record = {
        "video_id": video_id,
        "relative_path": f"media/{media_basename}",
        "size_bytes": media_path.stat().st_size,
        "sha256": media_sha,
    }
    return request, media_record


def build_bundle(
    *,
    manifest: Path,
    evidence: Path,
    whisper: Path,
    annotations_dir: Path,
    media_dir: Path,
    output_dir: Path,
    run_id: str,
    expected_count: int,
    expected_ids_sha256: str | None,
) -> dict[str, Any]:
    """Build all private selector inputs after complete in-memory validation."""

    if not run_id.strip():
        raise ValueError("run_id must be nonempty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RESERVED_OUTPUTS:
        candidate = output_dir / name
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite {candidate}")
    manifest_payload = _load_json(manifest)
    evidence_payload = _load_json(evidence)
    whisper_payload = _load_json(whisper)
    manifest_rows = manifest_payload.get("examples")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != expected_count:
        raise ValueError(f"manifest must contain exactly {expected_count} examples")
    if any(not isinstance(row, Mapping) for row in manifest_rows):
        raise ValueError("manifest examples must be objects")
    ids_sha256 = _id_digest(manifest_rows)
    if expected_ids_sha256 is not None and ids_sha256 != expected_ids_sha256:
        raise ValueError(
            "panel ID digest mismatch: "
            f"expected {expected_ids_sha256}, got {ids_sha256}"
        )
    evidence_by_id = _indexed_rows(evidence_payload, "examples", source="evidence")
    whisper_by_id = _indexed_rows(whisper_payload, "results", source="Whisper")
    expected_ids = {str(row["video_id"]) for row in manifest_rows}
    if set(evidence_by_id) != expected_ids or set(whisper_by_id) != expected_ids:
        raise ValueError("evidence and Whisper ID sets must match the manifest")

    requests: list[dict[str, Any]] = []
    media_records: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        video_id = str(manifest_row["video_id"])
        annotation_path = annotations_dir / f"{video_id}.json"
        annotation = _load_json(annotation_path)
        request, media_record = _request_row(
            manifest_row,
            evidence_by_id[video_id],
            whisper_by_id[video_id],
            annotation,
            media_dir,
            run_id=run_id,
        )
        requests.append(request)
        media_records.append(media_record)

    has_stratified_canary = len(requests) >= sum(
        required for _, required in STRATUM_TARGETS
    )
    canary_rows = (
        select_canary_rows(requests)
        if has_stratified_canary
        else [
            {**row, "question_stratum": question_stratum(row["question"])}
            for row in requests
        ]
    )
    resource_sentinel_count = 0
    resource_sentinel_sha256 = None
    if has_stratified_canary:
        selected_ids = {row["video_id"] for row in canary_rows}
        sentinel_candidates = [
            row for row in requests if row["video_id"] not in selected_ids
        ]
        resource_sentinel = select_resource_sentinel(sentinel_candidates or requests)
        sentinel_id = str(resource_sentinel["video_id"])
        if sentinel_id not in selected_ids:
            canary_rows.append(resource_sentinel)
            resource_sentinel_count = 1
        else:
            raise ValueError(
                "resource sentinel must be distinct from the ten-row canary"
            )
        resource_sentinel_sha256 = hashlib.sha256(sentinel_id.encode()).hexdigest()
    chunks = partition_rows(requests, chunk_count=4)
    request_rows_by_name: dict[str, Sequence[Mapping[str, Any]]] = {
        "selector-canary.jsonl": canary_rows,
        "selector-chunk-00.jsonl": chunks[0],
        "selector-chunk-01.jsonl": chunks[1],
        "selector-chunk-02.jsonl": chunks[2],
        "selector-chunk-03.jsonl": chunks[3],
        "selector-requests.jsonl": requests,
    }
    request_files = {
        name: _request_file_identity(rows)
        for name, rows in request_rows_by_name.items()
    }
    protocol_without_hash = {
        "schema_version": 1,
        "run_id": run_id,
        "panel": {
            "count": len(requests),
            "ids_sha256": ids_sha256,
            "canary_count": len(canary_rows),
            "stratified_canary_count": len(canary_rows) - resource_sentinel_count,
            "resource_sentinel_count": resource_sentinel_count,
            "resource_sentinel_id_sha256": resource_sentinel_sha256,
            "chunk_counts": [len(chunk) for chunk in chunks],
        },
        "selector": {
            "model_id": SELECTOR_MODEL_ID,
            "model_revision": SELECTOR_MODEL_REVISION,
            "runtime_image": SELECTOR_RUNTIME_IMAGE,
            "primary_seed": 42,
            "repair_seed": 43,
            "max_retries": 1,
            "prompt": SELECTOR_PROMPT,
            "repair_prompt": REPAIR_PROMPT,
            "max_intervals": 2,
            "max_whisper_segments": 3,
            "interval_duration_budget_formula": INTERVAL_DURATION_BUDGET_FORMULA,
        },
        "request_files": request_files,
        "decision": {
            "full_schema_failure_rate_stop": 0.05,
            "evidence_delta_c": 0.015,
            "election_delta_c": 0.010,
            "bootstrap_resamples": 10000,
        },
    }
    protocol = {
        **protocol_without_hash,
        "protocol_sha256": protocol_sha256(protocol_without_hash),
    }
    media_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "file_count": len(media_records),
        "files": media_records,
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "row_count": len(requests),
        "ids_sha256": ids_sha256,
        "protocol_sha256": protocol["protocol_sha256"],
        "request_set_sha256": _canonical_sha256(requests),
        "media_manifest_sha256": _canonical_sha256(media_manifest),
    }

    for name, rows in request_rows_by_name.items():
        _write_jsonl(output_dir / name, rows)
    _write_json(output_dir / "media-manifest.json", media_manifest)
    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "summary.json", summary)
    payload_names = [name for name in RESERVED_OUTPUTS if name != "SHA256SUMS"]
    checksum_lines = [
        f"{_sha256_file(output_dir / name)}  {name}\n" for name in sorted(payload_names)
    ]
    (output_dir / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--expected-ids-sha256", default=EXPECTED_IDS_SHA256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_bundle(
        manifest=args.manifest,
        evidence=args.evidence,
        whisper=args.whisper,
        annotations_dir=args.annotations_dir,
        media_dir=args.media_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_count=args.expected_count,
        expected_ids_sha256=args.expected_ids_sha256,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
