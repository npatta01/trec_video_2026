"""Contracts for the full-panel reference-assisted VQA evidence diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .temporal_bottleneck import (
    relevant_scene_candidate_plans,
    uniform_decoded_frame_plan,
)

SELECTOR_RESULT_FIELDS = frozenset({"status", "intervals", "whisper_segment_ids"})
SELECTOR_STATUSES = frozenset({"localized", "non_localizable"})
INTERVAL_DURATION_BUDGET_FORMULA = (
    "min(video_duration, max(12, min(30, 0.2 * video_duration)))"
)
FRAME_VIEWS = ("G", "H", "L")
TRANSCRIPT_VIEWS = ("N", "F", "S")
EVIDENCE_CELLS = tuple(
    f"{frame_view.lower()}{transcript_view.lower()}"
    for frame_view in FRAME_VIEWS
    for transcript_view in TRANSCRIPT_VIEWS
)
ANSWER_REQUEST_FIELDS = frozenset(
    {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "transcript",
    }
)
_TIMESTAMPED_LINE = re.compile(
    r"^\[(?P<hours>\d{2}):(?P<minutes>\d{2}):"
    r"(?P<seconds>\d{2}(?:\.\d{1,6})?)\][ \t]*(?P<text>.*)$"
)
_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
_FORBIDDEN_REQUEST_KEY_PARTS = frozenset(
    {
        "answer",
        "answers",
        "reference",
        "references",
        "prediction",
        "predictions",
        "grade",
        "grades",
        "metric",
        "metrics",
        "score",
        "scores",
        "label",
        "labels",
        "gold",
        "ground_truth",
        "correct",
        "incorrect",
        "rationale",
        "confidence",
        "selector",
        "interval",
        "intervals",
        "whisper_segment_ids",
    }
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


def selector_interval_budget(duration: float) -> float:
    """Return the frozen combined selector-interval budget for one video."""

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
        selector_interval_budget(duration) + 1e-9
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
    text: str | None,
    *,
    duration_seconds: float,
    drop_trailing_overrun: bool = False,
) -> list[WhisperSegment]:
    """Parse frozen start-stamped Whisper lines without rewriting their text."""

    duration = _finite_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(drop_trailing_overrun, bool):
        raise TypeError("drop_trailing_overrun must be a bool")
    if text is None:
        return []
    if not isinstance(text, str):
        raise TypeError("timestamped Whisper text must be a string or null")
    if not text.strip():
        return []
    parsed: list[tuple[float, str, str]] = []
    previous_raw_start: float | None = None
    overrun_started = False
    for index, source_line in enumerate(text.splitlines()):
        match = _TIMESTAMPED_LINE.fullmatch(source_line)
        if match is None:
            raise ValueError(f"timestamped Whisper line {index} has invalid syntax")
        start = _timestamp_seconds(match)
        if previous_raw_start is not None and start <= previous_raw_start:
            raise ValueError("timestamped Whisper starts must be strictly increasing")
        previous_raw_start = start
        if start >= duration:
            if drop_trailing_overrun:
                overrun_started = True
                continue
            raise ValueError(f"timestamped Whisper line {index} exceeds video duration")
        if overrun_started:
            raise ValueError("out-of-range Whisper lines must be trailing")
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
    global_plan = _enrich_frame_plan(
        uniform_decoded_frame_plan(duration, inventory, count=12), inventory
    )
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
    try:
        candidates = relevant_scene_candidate_plans(
            duration,
            sufficient_sets,
            0,
            inventory,
        )
        hybrid_plan = _enrich_frame_plan(
            candidates["oracle-hybrid-12"]["frames"], inventory
        )
        local_plan = _enrich_frame_plan(
            candidates["oracle-local-12"]["frames"], inventory
        )
        construction_fallback = None
    except ValueError as error:
        if "unique decoded frames" not in str(error):
            raise
        hybrid_plan, local_plan = _collision_fallback_plans(
            duration,
            sufficient_sets[0],
            inventory,
        )
        construction_fallback = "collision_fallback"
    _validate_view_plans({"G": global_plan, "H": hybrid_plan, "L": local_plan})
    return {
        "G": global_plan,
        "H": hybrid_plan,
        "L": local_plan,
        "localized": True,
        "fallback_reason": construction_fallback,
    }


def _enrich_frame_plan(
    plan: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Carry verified image identity alongside the temporal allocator output."""

    by_frame = {
        (record.get("stream_index"), record.get("frame_index")): record
        for record in inventory
    }
    enriched: list[dict[str, Any]] = []
    for frame in plan:
        record = dict(frame)
        source = by_frame.get((record.get("stream_index"), record.get("frame_index")))
        if source is not None:
            for source_key, target_key in (
                ("frame_path", "frame_path"),
                ("path", "frame_path"),
                ("image_path", "frame_path"),
                ("image_sha256", "image_sha256"),
                ("sha256", "image_sha256"),
            ):
                if target_key not in record and source_key in source:
                    record[target_key] = source[source_key]
        enriched.append(record)
    return enriched


def _collision_fallback_plans(
    duration: float,
    intervals: Sequence[Sequence[float]],
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct H/L plans when a local interval cannot supply every slot.

    Local slots retain their semantic ``slot_kind``. A frame borrowed outside
    the interval is marked explicitly so the construction cannot be mistaken
    for fully localized evidence.
    """

    normalized_intervals = [
        (float(interval[0]), float(interval[1])) for interval in intervals
    ]
    normalized_inventory = [dict(record) for record in inventory]
    used: set[tuple[Any, Any]] = set()

    def choose(
        desired_time: float,
        interval: tuple[float, float] | None,
        *,
        slot_kind: str,
    ) -> dict[str, Any]:
        candidates = [
            record
            for record in normalized_inventory
            if (record.get("stream_index"), record.get("frame_index")) not in used
            and (
                interval is None
                or interval[0] <= float(record["actual_pts_seconds"]) <= interval[1]
            )
        ]
        fallback = None
        if not candidates and interval is not None:
            candidates = [
                record
                for record in normalized_inventory
                if (record.get("stream_index"), record.get("frame_index")) not in used
            ]
            fallback = "global_frame"
        if not candidates:
            raise ValueError("collision fallback cannot supply twelve unique frames")
        selected = min(
            candidates,
            key=lambda record: (
                abs(float(record["actual_pts_seconds"]) - desired_time),
                float(record["actual_pts_seconds"]),
                int(record["stream_index"]),
                int(record["frame_index"]),
            ),
        )
        used.add((selected["stream_index"], selected["frame_index"]))
        result = {
            "slot_kind": slot_kind,
            "desired_time_seconds": round(desired_time, 6),
            "requested_time_seconds": round(desired_time, 6),
            "stream_index": selected["stream_index"],
            "frame_index": selected["frame_index"],
            "actual_pts_seconds": round(float(selected["actual_pts_seconds"]), 6),
            "time_base": selected["time_base"],
            "source_media_sha256": selected["source_media_sha256"],
        }
        for key in ("frame_path", "path", "image_path"):
            if key in selected:
                result["frame_path"] = selected[key]
                break
        for key in ("image_sha256", "sha256"):
            if key in selected:
                result["image_sha256"] = selected[key]
                break
        if fallback is not None:
            result["construction_fallback"] = fallback
        return result

    def local_slots(slot_count: int) -> list[tuple[float, tuple[float, float]]]:
        allocations = _hamilton_allocations(normalized_intervals, slot_count=slot_count)
        slots: list[tuple[float, tuple[float, float]]] = []
        for interval, allocation in zip(normalized_intervals, allocations, strict=True):
            start, end = interval
            width = (end - start) / allocation
            slots.extend(
                (round(start + (index + 0.5) * width, 6), interval)
                for index in range(allocation)
            )
        return slots

    local_frames = [
        choose(desired, interval, slot_kind="local")
        for desired, interval in local_slots(6)
    ]
    hybrid_global_frames = [
        choose(round((index + 0.5) * duration / 6, 6), None, slot_kind="global")
        for index in range(6)
    ]
    hybrid = sorted(
        [*local_frames[:6], *hybrid_global_frames],
        key=lambda frame: (
            frame["actual_pts_seconds"],
            frame["desired_time_seconds"],
            frame["slot_kind"],
        ),
    )

    # Rebuild L independently so its twelve local slots are not coupled to H.
    used.clear()
    local_frames = [
        choose(desired, interval, slot_kind="local")
        for desired, interval in local_slots(12)
    ]
    local = sorted(
        local_frames,
        key=lambda frame: (
            frame["actual_pts_seconds"],
            frame["desired_time_seconds"],
            frame["slot_kind"],
        ),
    )
    return hybrid, local


def _hamilton_allocations(
    intervals: Sequence[tuple[float, float]], *, slot_count: int
) -> list[int]:
    if not intervals or len(intervals) > slot_count:
        raise ValueError("cannot allocate local slots to the selected intervals")
    allocations = [1] * len(intervals)
    remaining = slot_count - len(intervals)
    durations = [end - start for start, end in intervals]
    total = sum(durations)
    quotas = [remaining * duration / total for duration in durations]
    floors = [math.floor(quota) for quota in quotas]
    allocations = [
        base + floor for base, floor in zip(allocations, floors, strict=True)
    ]
    remainder_order = sorted(
        range(len(intervals)),
        key=lambda index: (
            -(quotas[index] - floors[index]),
            intervals[index][0],
            index,
        ),
    )
    for index in remainder_order[: remaining - sum(floors)]:
        allocations[index] += 1
    return allocations


def _validate_view_plans(
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_local_slots: bool = True,
) -> None:
    for view in FRAME_VIEWS:
        frames = plans.get(view)
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError(f"{view} frame plan must be a sequence")
        if len(frames) != 12:
            raise ValueError(f"{view} frame plan must contain twelve frames")
        pts = [float(frame["actual_pts_seconds"]) for frame in frames]
        frame_ids = [(frame["stream_index"], frame["frame_index"]) for frame in frames]
        if pts != sorted(pts) or len(set(pts)) != 12:
            raise ValueError(f"{view} actual PTS must be unique and ordered")
        if len(set(frame_ids)) != 12:
            raise ValueError(f"{view} decoded frame identities must be unique")
    if require_local_slots:
        if sum(frame["slot_kind"] == "local" for frame in plans["H"]) != 6:
            raise ValueError("H frame plan must contain six local slots")
        if sum(frame["slot_kind"] == "global" for frame in plans["H"]) != 6:
            raise ValueError("H frame plan must contain six global slots")
        if any(frame["slot_kind"] != "local" for frame in plans["L"]):
            raise ValueError("L frame plan must contain twelve local slots")


def assert_global_plan_reuse(
    actual: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> None:
    """Require G to retain every verified decoded/image identity in order."""

    if len(actual) != 12 or len(baseline) != 12:
        raise ValueError("global plan reuse requires twelve frames")
    fields = (
        "slot_kind",
        "desired_time_seconds",
        "requested_time_seconds",
        "stream_index",
        "frame_index",
        "actual_pts_seconds",
        "time_base",
        "source_media_sha256",
        "frame_path",
        "image_sha256",
    )
    for index, (candidate, expected) in enumerate(zip(actual, baseline, strict=True)):
        for field in fields:
            if field in expected and candidate.get(field) != expected[field]:
                raise ValueError(f"global plan frame {index} does not reuse {field}")


def build_transcript_views(
    text: str | None,
    selected_segment_ids: Sequence[int],
    *,
    duration_seconds: float,
) -> dict[str, dict[str, Any]]:
    """Build exact N/F/S transcript payloads from frozen timestamped bytes."""

    if text is not None and not isinstance(text, str):
        raise TypeError("timestamped Whisper text must be a string or null")
    if not isinstance(selected_segment_ids, Sequence) or isinstance(
        selected_segment_ids, (str, bytes)
    ):
        raise TypeError("selected_segment_ids must be a sequence")
    ids = list(selected_segment_ids)
    if len(ids) > 3 or any(
        isinstance(segment_id, bool) or not isinstance(segment_id, int)
        for segment_id in ids
    ):
        raise ValueError("selected transcript IDs must contain at most three integers")
    if ids != sorted(set(ids)):
        raise ValueError("selected transcript IDs must be sorted and unique")
    parsed = parse_timestamped_whisper(
        text,
        duration_seconds=duration_seconds,
        drop_trailing_overrun=True,
    )
    by_id = {segment.segment_id: segment for segment in parsed}
    if any(segment_id not in by_id for segment_id in ids):
        raise ValueError("selected transcript ID is outside the frozen transcript")
    full_text = text if text is not None and text.strip() else None
    source_sha256 = (
        hashlib.sha256(text.encode("utf-8")).hexdigest()
        if full_text is not None
        else None
    )
    selected_text = "\n".join(by_id[segment_id].source_line for segment_id in ids)
    return {
        "N": {
            "text": None,
            "selected_line_indices": [],
            "source_sha256": None,
            "selected_sha256": None,
        },
        "F": {
            "text": full_text,
            "selected_line_indices": list(range(len(parsed))) if full_text else [],
            "source_sha256": source_sha256,
            "selected_sha256": source_sha256,
        },
        "S": {
            "text": selected_text if ids else None,
            "selected_line_indices": ids,
            "source_sha256": source_sha256,
            "selected_sha256": (
                hashlib.sha256(selected_text.encode("utf-8")).hexdigest()
                if ids
                else None
            ),
        },
    }


def _request_frame_fields(
    frames: Sequence[Mapping[str, Any]], *, cell_id: str
) -> tuple[list[str], list[float]]:
    if len(frames) != 12:
        raise ValueError(f"{cell_id} must contain exactly twelve frames")
    paths: list[str] = []
    timestamps: list[float] = []
    actual_pts: list[float] = []
    decoded_identities: list[tuple[object, object]] = []
    for index, frame in enumerate(frames):
        path = frame.get("frame_path", frame.get("path", frame.get("image_path")))
        if not isinstance(path, str) or not path:
            raise ValueError(f"{cell_id} frame {index} lacks frame_path")
        path = str(Path(path).resolve())
        timestamp = _finite_number(
            frame.get("requested_time_seconds"),
            f"{cell_id} frame {index} requested_time_seconds",
        )
        if timestamp < 0:
            raise ValueError(f"{cell_id} frame {index} timestamp is negative")
        pts = _finite_number(
            frame.get("actual_pts_seconds"),
            f"{cell_id} frame {index} actual_pts_seconds",
        )
        if pts < 0:
            raise ValueError(f"{cell_id} frame {index} actual PTS is negative")
        stream_index = frame.get("stream_index")
        frame_index = frame.get("frame_index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
        ):
            raise ValueError(f"{cell_id} frame {index} lacks a decoded identity")
        paths.append(path)
        timestamps.append(timestamp)
        actual_pts.append(pts)
        decoded_identities.append((stream_index, frame_index))
    if len(set(paths)) != 12:
        raise ValueError(f"{cell_id} materialized frame paths must be unique")
    if len(set(actual_pts)) != 12:
        raise ValueError(f"{cell_id} actual decoded PTS must be unique")
    if len(set(decoded_identities)) != 12:
        raise ValueError(f"{cell_id} decoded frame identities must be unique")
    return paths, timestamps


def _request_key_part(key: object) -> str:
    return _KEY_BOUNDARY.sub("_", str(key)).casefold().strip("_")


def audit_label_free_request(value: Any, *, location: str = "request") -> None:
    """Reject answer/reference/selector material at every nested request level."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _request_key_part(key)
            tokens = set(normalized.split("_"))
            forbidden = sorted(
                part
                for part in _FORBIDDEN_REQUEST_KEY_PARTS
                if part in tokens or part == normalized
            )
            if forbidden:
                raise ValueError(
                    f"label leakage in {location}.{key}: forbidden key {forbidden[0]}"
                )
            audit_label_free_request(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            audit_label_free_request(nested, location=f"{location}[{index}]")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_sha256(request: Mapping[str, Any]) -> str:
    """Hash one exact label-free runner request."""

    audit_label_free_request(request)
    return _canonical_sha256(request)


def model_input_sha256(request: Mapping[str, Any]) -> str:
    """Hash only bytes/fields presented to the Qwen3.5 answerer."""

    raw_paths = request.get("frame_paths")
    frame_inputs: dict[str, Any] = {"frame_paths": raw_paths}
    if isinstance(raw_paths, list) and all(
        isinstance(raw_path, str) for raw_path in raw_paths
    ):
        frame_sha256s: list[str] = []
        for raw_path in raw_paths:
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file():
                break
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            frame_sha256s.append(digest.hexdigest())
        if len(frame_sha256s) == len(raw_paths):
            frame_inputs = {"frame_sha256s": frame_sha256s}
    return _canonical_sha256(
        {
            "question": request.get("question"),
            **frame_inputs,
            "frame_timestamps": request.get("frame_timestamps"),
            "transcript": request.get("transcript"),
        }
    )


def held_fixed_identity_sha256(request: Mapping[str, Any]) -> str:
    """Hash the row/model contract while excluding the tested evidence arms."""

    return _canonical_sha256(
        {
            "run_id": request.get("run_id"),
            "video_id": request.get("video_id"),
            "question": request.get("question"),
            "frame_count": len(request.get("frame_paths", [])),
        }
    )


def build_request_matrix(
    *,
    row: Mapping[str, Any],
    frame_plans: Mapping[str, Sequence[Mapping[str, Any]]],
    transcript_views: Mapping[str, Mapping[str, Any]],
    run_id: str,
    media_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the exact nine canonical, label-free Qwen3.5 requests."""

    if not isinstance(row, Mapping):
        raise TypeError("request row must be an object")
    audit_label_free_request(row, location="request row")
    video_id = row.get("video_id")
    question = row.get("question")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("request row video_id must be nonempty text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("request row question must be nonempty text")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be nonempty text")
    if media_path is not None and not isinstance(media_path, str):
        raise TypeError("media_path must be text or null")
    if not set(FRAME_VIEWS).issubset(frame_plans):
        raise ValueError("frame plans must contain G, H, and L")
    if set(transcript_views) != set(TRANSCRIPT_VIEWS):
        raise ValueError("transcript views must contain exactly N, F, and S")
    _validate_view_plans(
        frame_plans,
        require_local_slots=frame_plans.get("localized") is not False,
    )

    matrix: dict[str, dict[str, Any]] = {}
    for frame_view in FRAME_VIEWS:
        paths, timestamps = _request_frame_fields(
            frame_plans[frame_view], cell_id=frame_view
        )
        for transcript_view in TRANSCRIPT_VIEWS:
            cell_id = f"{frame_view.lower()}{transcript_view.lower()}"
            view = transcript_views[transcript_view]
            text = view.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{cell_id} transcript must be text or null")
            if transcript_view == "N" and text is not None:
                raise ValueError("N requests must contain zero transcript bytes")
            if transcript_view != "N" and text is not None and not text:
                raise ValueError(f"{cell_id} transcript cannot be empty")
            request = {
                "run_id": run_id,
                "condition_id": cell_id,
                "video_id": video_id,
                "question": question.strip(),
                "frame_paths": paths.copy(),
                "frame_timestamps": timestamps.copy(),
                "transcript": text,
            }
            audit_label_free_request(request)
            matrix[cell_id] = request
    if set(matrix) != set(EVIDENCE_CELLS):
        raise AssertionError("request matrix did not cover the exact nine cells")
    return matrix


def audit_request_matrix(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Audit exact nine-cell coverage, leakage, and held-fixed row identity."""

    if set(matrix) != set(EVIDENCE_CELLS):
        raise ValueError("request matrix must contain exactly the nine evidence cells")
    fixed_identities: set[str] = set()
    request_hashes: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for cell_id in EVIDENCE_CELLS:
        request = matrix[cell_id]
        audit_label_free_request(request, location=f"request[{cell_id}]")
        if set(request) != ANSWER_REQUEST_FIELDS:
            raise ValueError(f"{cell_id} request does not use the exact answer schema")
        if request.get("condition_id") != cell_id:
            raise ValueError(f"{cell_id} condition identity is not held fixed")
        request_hashes[cell_id] = request_sha256(request)
        input_hashes[cell_id] = model_input_sha256(request)
        fixed_identities.add(held_fixed_identity_sha256(request))
    if len(fixed_identities) != 1:
        raise ValueError("cross-cell held-fixed identity mismatch")
    return {
        "cell_count": len(EVIDENCE_CELLS),
        "cells": list(EVIDENCE_CELLS),
        "request_sha256": request_hashes,
        "model_input_sha256": input_hashes,
        "held_fixed_identity_sha256": next(iter(fixed_identities)),
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
