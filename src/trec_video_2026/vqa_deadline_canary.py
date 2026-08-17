"""Small, strict contracts for the deadline VQA decision canary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_STRATA = {"temporal", "causal", "ocr_detail", "counting", "speech"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path, *, data_root: Path) -> list[dict[str, Any]]:
    """Load and verify the exact five labeled development rows."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("rows"), list):
        raise ValueError("invalid panel schema")
    rows = payload["rows"]
    if len(rows) != 5:
        raise ValueError("panel must contain exactly five rows")
    video_ids = [row.get("video_id") for row in rows]
    if len(set(video_ids)) != 5 or not all(
        isinstance(video_id, str) and _VIDEO_ID.fullmatch(video_id)
        for video_id in video_ids
    ):
        raise ValueError("panel video IDs must be five unique canonical IDs")
    if {row.get("stratum") for row in rows} != REQUIRED_STRATA:
        raise ValueError("panel must cover the five required strata exactly once")

    canonical_root = data_root.resolve(strict=True)
    verified: list[dict[str, Any]] = []
    for row in rows:
        relative = Path(str(row.get("source_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("panel source paths must be canonical data-relative paths")
        source = (canonical_root / relative).resolve(strict=True)
        if not source.is_relative_to(canonical_root):
            raise ValueError("panel source escaped canonical data root")
        if _sha256_file(source) != row.get("source_sha256"):
            raise ValueError(f"source hash mismatch for {row['video_id']}")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("panel question must be non-empty")
        duration = row.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("panel duration must be positive")
        verified_row = dict(row)
        verified_row["source_path"] = str(source)
        verified.append(verified_row)
    return verified


def validate_ranked_submission(
    rows: Iterable[Mapping[str, Any]], *, expected_query_ids: set[int]
) -> None:
    """Validate the organizer's ten-ranked-answer record shape."""

    materialized = list(rows)
    query_ids = [row.get("Q_ID") for row in materialized]
    if set(query_ids) != expected_query_ids or len(query_ids) != len(expected_query_ids):
        raise ValueError("submission query IDs do not match the expected query IDs")
    for row in materialized:
        if not isinstance(row.get("Q_ID"), int) or not isinstance(row.get("Video_ID"), str):
            raise ValueError("invalid query or video identifier")
        answers = row.get("Answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError("each query must have exactly ten answers")
        if [answer.get("Rank") for answer in answers] != list(range(1, 11)):
            raise ValueError("answer ranks must be consecutive from 1 through 10")
        normalized: list[str] = []
        for answer in answers:
            text = answer.get("Answer")
            elapsed = answer.get("Time")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("answers must be non-empty strings")
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("answer times must be finite and non-negative")
            normalized.append(" ".join(text.casefold().split()))
        if len(set(normalized)) != 10:
            raise ValueError("each query must have ten distinct answers")
