"""Contracts for the full-panel reference-assisted VQA evidence diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .temporal_bottleneck import (
    relevant_scene_candidate_plans,
    uniform_decoded_frame_plan,
)

SELECTOR_RESULT_FIELDS = frozenset({"status", "intervals", "whisper_segment_ids"})
SELECTOR_STATUSES = frozenset({"localized", "non_localizable"})
_TIMESTAMPED_LINE = re.compile(
    r"^\[(?P<hours>\d{2}):(?P<minutes>\d{2}):"
    r"(?P<seconds>\d{2}(?:\.\d{1,6})?)\][ \t]*(?P<text>.*)$"
)


@dataclass(frozen=True, slots=True)
class WhisperSegment:
    """One verbatim timestamped Whisper line with an inferred end time."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    text: str
    source_line: str


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _row_context(row: Mapping[str, object]) -> tuple[float, int]:
    if not isinstance(row, Mapping):
        raise TypeError("selector row must be an object")
    duration = _finite_number(row.get("duration_seconds"), "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    segment_count = row.get("whisper_segment_count")
    if (
        isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or segment_count < 0
    ):
        raise ValueError("whisper_segment_count must be a non-negative integer")
    return duration, segment_count


def _interval_budget(duration: float) -> float:
    return min(duration, max(12.0, min(30.0, 0.2 * duration)))


def validate_selector_result(
    payload: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, object]:
    """Validate and normalize the selector's numeric-only terminal object."""

    duration, segment_count = _row_context(row)
    if not isinstance(payload, Mapping):
        raise TypeError("selector result must be an object")
    if set(payload) != SELECTOR_RESULT_FIELDS:
        raise ValueError(
            "selector result must contain exactly status, intervals, and "
            "whisper_segment_ids"
        )
    status = payload.get("status")
    if status not in SELECTOR_STATUSES:
        raise ValueError(f"selector status must be one of {sorted(SELECTOR_STATUSES)}")

    raw_intervals = payload.get("intervals")
    if not isinstance(raw_intervals, list):
        raise ValueError("selector intervals must be a list")
    if len(raw_intervals) > 2:
        raise ValueError("selector may return at most two intervals")
    intervals: list[dict[str, float]] = []
    previous_end: float | None = None
    for index, raw_interval in enumerate(raw_intervals):
        if not isinstance(raw_interval, Mapping) or set(raw_interval) != {
            "start_seconds",
            "end_seconds",
        }:
            raise ValueError(
                f"selector interval {index} must contain exactly "
                "start_seconds and end_seconds"
            )
        start = _finite_number(
            raw_interval.get("start_seconds"), f"interval {index} start_seconds"
        )
        end = _finite_number(
            raw_interval.get("end_seconds"), f"interval {index} end_seconds"
        )
        if start < 0 or end <= start or end > duration:
            raise ValueError(f"selector interval {index} is outside video duration")
        if previous_end is not None and start < previous_end:
            raise ValueError("selector intervals overlap or are not sorted")
        previous_end = end
        intervals.append({"start_seconds": start, "end_seconds": end})
    if sum(item["end_seconds"] - item["start_seconds"] for item in intervals) > (
        _interval_budget(duration) + 1e-9
    ):
        raise ValueError("selector intervals exceed the combined duration budget")

    raw_segment_ids = payload.get("whisper_segment_ids")
    if not isinstance(raw_segment_ids, list):
        raise ValueError("whisper_segment_ids must be a list")
    if len(raw_segment_ids) > 3:
        raise ValueError("selector may return at most three Whisper segment IDs")
    segment_ids: list[int] = []
    for index, segment_id in enumerate(raw_segment_ids):
        if isinstance(segment_id, bool) or not isinstance(segment_id, int):
            raise ValueError(f"Whisper segment ID {index} must be an integer")
        if not 0 <= segment_id < segment_count:
            raise ValueError(
                f"Whisper segment ID {segment_id} is outside the transcript"
            )
        if segment_id in segment_ids:
            raise ValueError("Whisper segment IDs must be unique")
        segment_ids.append(segment_id)
    if segment_ids != sorted(segment_ids):
        raise ValueError("Whisper segment IDs must be sorted")

    if status == "non_localizable" and (intervals or segment_ids):
        raise ValueError(
            "non_localizable selector results must have empty evidence arrays"
        )
    if status == "localized" and not (intervals or segment_ids):
        raise ValueError(
            "localized selector results must select visual or transcript evidence"
        )
    return {
        "status": status,
        "intervals": intervals,
        "whisper_segment_ids": segment_ids,
    }


def _timestamp_seconds(match: re.Match[str]) -> float:
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("timestamped Whisper line has an invalid clock value")
    return round(hours * 3600 + minutes * 60 + seconds, 6)


def parse_timestamped_whisper(
    text: str | None, *, duration_seconds: float
) -> list[WhisperSegment]:
    """Parse frozen start-stamped Whisper lines without rewriting their text."""

    duration = _finite_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    if text is None or not text.strip():
        return []
    if not isinstance(text, str):
        raise TypeError("timestamped Whisper text must be a string or null")
    parsed: list[tuple[float, str, str]] = []
    for index, source_line in enumerate(text.splitlines()):
        match = _TIMESTAMPED_LINE.fullmatch(source_line)
        if match is None:
            raise ValueError(f"timestamped Whisper line {index} has invalid syntax")
        start = _timestamp_seconds(match)
        if start > duration:
            raise ValueError(f"timestamped Whisper line {index} exceeds video duration")
        if parsed and start <= parsed[-1][0]:
            raise ValueError("timestamped Whisper starts must be strictly increasing")
        parsed.append((start, match.group("text"), source_line))
    return [
        WhisperSegment(
            segment_id=index,
            start_seconds=start,
            end_seconds=parsed[index + 1][0] if index + 1 < len(parsed) else duration,
            text=segment_text,
            source_line=source_line,
        )
        for index, (start, segment_text, source_line) in enumerate(parsed)
    ]


def build_frame_plans(
    row: Mapping[str, object],
    selector: Mapping[str, object],
    inventory: list[dict[str, Any]],
) -> dict[str, object]:
    """Build deterministic global, hybrid, and local 12-frame plans."""

    duration, _ = _row_context(row)
    normalized_selector = validate_selector_result(selector, row)
    global_plan = uniform_decoded_frame_plan(duration, inventory, count=12)
    raw_intervals = normalized_selector["intervals"]
    assert isinstance(raw_intervals, list)
    if not raw_intervals:
        reason = (
            "non_localizable"
            if normalized_selector["status"] == "non_localizable"
            else "no_visual_intervals"
        )
        return {
            "G": global_plan,
            "H": copy.deepcopy(global_plan),
            "L": copy.deepcopy(global_plan),
            "localized": False,
            "fallback_reason": reason,
        }
    sufficient_sets = [
        [
            [interval["start_seconds"], interval["end_seconds"]]
            for interval in raw_intervals
        ]
    ]
    candidates = relevant_scene_candidate_plans(
        duration,
        sufficient_sets,
        0,
        inventory,
    )
    return {
        "G": global_plan,
        "H": candidates["oracle-hybrid-12"]["frames"],
        "L": candidates["oracle-local-12"]["frames"],
        "localized": True,
        "fallback_reason": None,
    }


def protocol_sha256(protocol: Any) -> str:
    """Hash canonical UTF-8 JSON for immutable protocol identity."""

    encoded = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
