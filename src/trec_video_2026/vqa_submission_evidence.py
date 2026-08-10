"""Label-free timeline and decoded-frame evidence planning for submission VQA."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_benchmark import (
    bin_midpoint_timestamps,
    probe_media,
    source_fingerprint,
)
from trec_video_2026.vqa_experiments.silver_oracle import (
    WhisperSegment,
    parse_timestamped_whisper,
)
from trec_video_2026.vqa_submission import (
    TimelineSpan,
    audit_label_free_payload,
)


SCHEMA_VERSION = 1
GLOBAL_FRAME_COUNT = 8
MAX_LOCAL_FRAME_COUNT = 24
MAX_LOCAL_FRAMES_PER_SECOND = 4
MAX_SUPPORT_SPANS = 3
ALLOWED_MODALITIES = frozenset(
    {"visual", "speech", "non_speech_audio", "visible_text"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPAN_ID_KEYS = ("span_id", "event_id", "id")
_START_KEYS = ("start_seconds", "start", "start_time")
_END_KEYS = ("end_seconds", "end", "end_time")
_MODALITY_KEYS = ("modalities", "modality")
_SHORT_TIMESTAMP_LINE = re.compile(
    r"^\[(?P<minutes>\d{2}):(?P<seconds>\d{2}(?:\.\d{1,6})?)\][ \t]*(?P<text>.*)$"
)


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if nonnegative and number < 0:
        raise ValueError(f"{field} cannot be negative")
    return number


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-hex SHA-256")
    return value


def _sequence(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return tuple(value)


def _first_mapping_value(mapping: Mapping[str, Any], keys: Sequence[str], field: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise ValueError(f"{field} is required")


def _coerce_span(raw: Any, *, field: str) -> TimelineSpan:
    if isinstance(raw, TimelineSpan):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field} must contain TimelineSpan objects")

    span_id = _first_mapping_value(raw, _SPAN_ID_KEYS, f"{field}.span_id")
    start = _first_mapping_value(raw, _START_KEYS, f"{field}.start_seconds")
    end = _first_mapping_value(raw, _END_KEYS, f"{field}.end_seconds")
    modalities = _first_mapping_value(raw, _MODALITY_KEYS, f"{field}.modalities")
    description = raw.get("description", raw.get("text"))
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{field}.description must be a non-empty string")
    if isinstance(modalities, str):
        modalities = (modalities,)
    return TimelineSpan(
        span_id=span_id,
        start_seconds=start,
        end_seconds=end,
        modalities=tuple(modalities),
        description=description,
    )


@dataclass(frozen=True, slots=True)
class ValidatedTimeline:
    """Validated chronological events and bounded support spans."""

    duration_seconds: float
    events: tuple[TimelineSpan, ...] = ()
    support_spans: tuple[TimelineSpan, ...] = ()
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        duration = _finite_number(
            self.duration_seconds, "duration_seconds", nonnegative=True
        )
        if duration < 1.0:
            raise ValueError("media duration must be at least one second")
        events = tuple(self.events)
        support_spans = tuple(self.support_spans)
        if not all(isinstance(item, TimelineSpan) for item in events + support_spans):
            raise TypeError("timeline records must contain TimelineSpan values")
        if self.fallback_reason not in {None, "invalid_timeline"}:
            raise ValueError("fallback_reason must be invalid_timeline or None")
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "support_spans", support_spans)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "duration_seconds": self.duration_seconds,
            "events": [event.to_dict() for event in self.events],
            "support_spans": [span.to_dict() for span in self.support_spans],
        }
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        return payload


def _validate_span(
    span: TimelineSpan,
    *,
    duration_seconds: float,
    field: str,
) -> TimelineSpan:
    if not math.isfinite(span.start_seconds) or not math.isfinite(span.end_seconds):
        raise ValueError(f"{field} must contain finite times")
    if span.start_seconds < 0 or span.end_seconds > duration_seconds:
        raise ValueError(f"{field} is outside media duration")
    if span.start_seconds >= span.end_seconds:
        raise ValueError(f"{field} must have start before end")
    if not span.modalities or any(
        modality not in ALLOWED_MODALITIES for modality in span.modalities
    ):
        raise ValueError(f"{field} contains an undeclared modality")
    if len(span.description) > 512:
        raise ValueError(f"{field}.description is too long")
    return span


def _expand_support_span(span: TimelineSpan, *, duration_seconds: float) -> TimelineSpan:
    if span.end_seconds - span.start_seconds >= 1.0:
        return span
    midpoint = (span.start_seconds + span.end_seconds) / 2.0
    start = max(0.0, min(midpoint - 0.5, duration_seconds - 1.0))
    end = start + 1.0
    return TimelineSpan(
        span_id=span.span_id,
        start_seconds=round(start, 6),
        end_seconds=round(end, 6),
        modalities=span.modalities,
        description=span.description,
    )


def _timeline_parts(timeline: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if isinstance(timeline, Mapping):
        raw_events = timeline.get("events", ())
        raw_support_spans = timeline.get("support_spans", ())
    else:
        raw_events = timeline
        raw_support_spans = ()
    return _sequence(raw_events, "events"), _sequence(
        raw_support_spans, "support_spans"
    )


def _validate_timeline_or_raise(
    timeline: Any, *, duration_seconds: float
) -> ValidatedTimeline:
    raw_events, raw_support_spans = _timeline_parts(timeline)
    if len(raw_support_spans) > MAX_SUPPORT_SPANS:
        raise ValueError("support_spans may contain at most three spans")

    events = tuple(
        _validate_span(
            _coerce_span(raw, field=f"events[{index}]"),
            duration_seconds=duration_seconds,
            field=f"events[{index}]",
        )
        for index, raw in enumerate(raw_events)
    )
    support_spans = tuple(
        _validate_span(
            _coerce_span(raw, field=f"support_spans[{index}]"),
            duration_seconds=duration_seconds,
            field=f"support_spans[{index}]",
        )
        for index, raw in enumerate(raw_support_spans)
    )

    all_ids = [span.span_id for span in events + support_spans]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("timeline span IDs must be unique")
    if any(
        current.start_seconds < previous.start_seconds
        for previous, current in zip(events, events[1:])
    ):
        raise ValueError("events must be sorted chronologically")
    if any(
        current.start_seconds < previous.end_seconds
        for previous, current in zip(support_spans, support_spans[1:])
    ):
        raise ValueError("support_spans must not overlap")

    expanded_support_spans = tuple(
        _expand_support_span(span, duration_seconds=duration_seconds)
        for span in support_spans
    )
    if any(
        current.start_seconds < previous.end_seconds
        for previous, current in zip(expanded_support_spans, expanded_support_spans[1:])
    ):
        raise ValueError("expanded support_spans must not overlap")
    return ValidatedTimeline(
        duration_seconds=duration_seconds,
        events=events,
        support_spans=expanded_support_spans,
    )


def _normalize_support_spans(
    support_spans: Any, *, duration_seconds: float
) -> tuple[TimelineSpan, ...]:
    """Apply the timeline support-span contract to every raw-span entry point."""

    audit_label_free_payload(support_spans)
    return _validate_timeline_or_raise(
        {"events": (), "support_spans": support_spans},
        duration_seconds=duration_seconds,
    ).support_spans


def validate_timeline(
    timeline: Any,
    duration_seconds: float,
    *,
    allow_fallback: bool = False,
) -> ValidatedTimeline:
    """Validate a label-free timeline, optionally returning a fixed fallback."""

    duration = _finite_number(
        duration_seconds, "duration_seconds", nonnegative=True
    )
    if duration < 1.0:
        raise ValueError("media duration must be at least one second")
    if not isinstance(allow_fallback, bool):
        raise TypeError("allow_fallback must be a bool")
    audit_label_free_payload(timeline)
    try:
        return _validate_timeline_or_raise(timeline, duration_seconds=duration)
    except (TypeError, ValueError):
        if not allow_fallback:
            raise
        return ValidatedTimeline(
            duration_seconds=duration,
            events=(),
            support_spans=(),
            fallback_reason="invalid_timeline",
        )


@dataclass(frozen=True, slots=True)
class FrameSelection:
    """One requested timestamp bound to one unique decoded frame identity."""

    requested_time_seconds: float
    actual_pts_seconds: float
    stream_index: int
    frame_index: int
    time_base: str
    source_media_sha256: str
    image_sha256: str | None = None
    frame_path: Path | None = None
    slot_kind: str = "global"
    support_span_id: str | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        requested = _finite_number(
            self.requested_time_seconds,
            "requested_time_seconds",
            nonnegative=True,
        )
        actual = _finite_number(
            self.actual_pts_seconds, "actual_pts_seconds", nonnegative=True
        )
        for value, field in (
            (self.stream_index, "stream_index"),
            (self.frame_index, "frame_index"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if not isinstance(self.time_base, str) or not self.time_base.strip():
            raise ValueError("time_base must be a non-empty string")
        source_sha = _sha256(self.source_media_sha256, "source_media_sha256")
        image_sha = self.image_sha256
        if image_sha is not None:
            image_sha = _sha256(image_sha, "image_sha256")
        if self.frame_path is not None:
            frame_path = Path(self.frame_path)
        else:
            frame_path = None
        if self.slot_kind not in {"global", "local"}:
            raise ValueError("slot_kind must be global or local")
        if self.support_span_id is not None and not isinstance(
            self.support_span_id, str
        ):
            raise TypeError("support_span_id must be a string or None")
        if self.fallback_reason not in {None, "local_collision_fallback"}:
            raise ValueError(
                "fallback_reason must be local_collision_fallback or None"
            )
        object.__setattr__(self, "requested_time_seconds", round(requested, 6))
        object.__setattr__(self, "actual_pts_seconds", round(actual, 6))
        object.__setattr__(self, "source_media_sha256", source_sha)
        object.__setattr__(self, "image_sha256", image_sha)
        object.__setattr__(self, "frame_path", frame_path)

    @property
    def decoded_identity(self) -> tuple[int, int]:
        return (self.stream_index, self.frame_index)

    @property
    def source_sha256(self) -> str:
        return self.source_media_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_time_seconds": self.requested_time_seconds,
            "actual_pts_seconds": self.actual_pts_seconds,
            "decoded_identity": {
                "stream_index": self.stream_index,
                "frame_index": self.frame_index,
            },
            "stream_index": self.stream_index,
            "frame_index": self.frame_index,
            "time_base": self.time_base,
            "frame_path": str(self.frame_path) if self.frame_path is not None else None,
            "image_sha256": self.image_sha256,
            "source_media_sha256": self.source_media_sha256,
            "slot_kind": self.slot_kind,
            "support_span_id": self.support_span_id,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class TranscriptSelection:
    """Verbatim timestamped transcript lines overlapping support spans."""

    text: str
    source_sha256: str
    segments: tuple[WhisperSegment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def selected_segments(self) -> tuple[WhisperSegment, ...]:
        return self.segments

    @property
    def selected_segment_ids(self) -> tuple[int, ...]:
        return tuple(segment.segment_id for segment in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_sha256": self.source_sha256,
            "selected_segment_ids": list(self.selected_segment_ids),
        }


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    """Global and local frame selections plus optional transcript evidence."""

    duration_seconds: float = 0.0
    global_frames: tuple[FrameSelection, ...] = ()
    local_frames: tuple[FrameSelection, ...] = ()
    support_spans: tuple[TimelineSpan, ...] = ()
    transcript: TranscriptSelection | None = None

    def __post_init__(self) -> None:
        duration = _finite_number(
            self.duration_seconds, "duration_seconds", nonnegative=True
        )
        global_frames = tuple(self.global_frames)
        local_frames = tuple(self.local_frames)
        support_spans = tuple(self.support_spans)
        if not all(
            isinstance(frame, FrameSelection)
            for frame in global_frames + local_frames
        ):
            raise TypeError("frame plans must contain FrameSelection values")
        if not all(isinstance(span, TimelineSpan) for span in support_spans):
            raise TypeError("support_spans must contain TimelineSpan values")
        if not global_frames and not local_frames:
            raise ValueError("evidence plan cannot be empty")
        if len(global_frames) != GLOBAL_FRAME_COUNT:
            raise ValueError("global_frames must contain exactly eight frames")
        if len(local_frames) > MAX_LOCAL_FRAME_COUNT:
            raise ValueError("local_frames cannot contain more than 24 frames")
        if duration <= 0:
            raise ValueError("evidence plan duration must be positive")
        normalized_support_spans = _validate_timeline_or_raise(
            {"events": (), "support_spans": support_spans},
            duration_seconds=duration,
        ).support_spans
        if normalized_support_spans != support_spans:
            raise ValueError("support_spans must be validated and normalized")
        for frame in global_frames + local_frames:
            if not 0 <= frame.requested_time_seconds <= duration:
                raise ValueError("frame requested timestamp is outside duration")
            if not 0 <= frame.actual_pts_seconds <= duration:
                raise ValueError("frame actual PTS is outside duration")
        if any(
            frame.slot_kind != "global" or frame.support_span_id is not None
            for frame in global_frames
        ):
            raise ValueError("global frames must have global slots and no support span")
        support_span_ids = {span.span_id for span in support_spans}
        for frame in local_frames:
            if frame.slot_kind != "local":
                raise ValueError("local frames must have local slots")
            if not frame.support_span_id:
                raise ValueError("local frames require support_span_id")
            if frame.support_span_id not in support_span_ids:
                raise ValueError("local support_span_id is not declared")
        identities = [frame.decoded_identity for frame in global_frames + local_frames]
        if len(set(identities)) != len(identities):
            raise ValueError("evidence plan must not reuse decoded frame identities")
        if self.transcript is not None and not isinstance(
            self.transcript, TranscriptSelection
        ):
            raise TypeError("transcript must be a TranscriptSelection or None")
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "global_frames", global_frames)
        object.__setattr__(self, "local_frames", local_frames)
        object.__setattr__(self, "support_spans", support_spans)

    @property
    def all_frames(self) -> tuple[FrameSelection, ...]:
        return self.global_frames + self.local_frames

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "duration_seconds": self.duration_seconds,
            "support_spans": [span.to_dict() for span in self.support_spans],
            "frames": [frame.to_dict() for frame in self.all_frames],
            "global_frame_count": len(self.global_frames),
            "local_frame_count": len(self.local_frames),
        }
        if self.transcript is not None:
            payload["transcript"] = self.transcript.to_dict()
        return payload


def _normalize_inventory(
    duration_seconds: float, inventory: Any
) -> list[dict[str, Any]]:
    audit_label_free_payload(inventory)
    records = _sequence(inventory, "inventory")
    if not records:
        raise ValueError("inventory must contain decoded frames")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    pts_by_stream: dict[int, set[float]] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"inventory[{index}] must be an object")
        stream_index = raw.get("stream_index")
        frame_index = raw.get("frame_index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
        ):
            raise ValueError(f"inventory[{index}].stream_index is invalid")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(f"inventory[{index}].frame_index is invalid")
        identity = (stream_index, frame_index)
        if identity in identities:
            raise ValueError("inventory contains duplicate decoded identities")
        identities.add(identity)
        actual = round(
            _finite_number(
                raw.get("actual_pts_seconds"),
                f"inventory[{index}].actual_pts_seconds",
                nonnegative=True,
            ),
            6,
        )
        if actual > duration_seconds:
            raise ValueError(
                f"inventory[{index}].actual_pts_seconds is outside duration"
            )
        stream_pts = pts_by_stream.setdefault(stream_index, set())
        if actual in stream_pts:
            raise ValueError(
                "inventory contains duplicate decoded PTS within stream"
            )
        stream_pts.add(actual)
        time_base = raw.get("time_base")
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"inventory[{index}].time_base is required")
        source_sha = raw.get("source_media_sha256", raw.get("source_sha256"))
        source_sha = _sha256(source_sha, f"inventory[{index}].source_media_sha256")
        frame_path = raw.get("frame_path", raw.get("path", raw.get("image_path")))
        image_sha = raw.get("image_sha256", raw.get("sha256"))
        if image_sha is not None:
            image_sha = _sha256(image_sha, f"inventory[{index}].image_sha256")
        normalized.append(
            {
                "stream_index": stream_index,
                "frame_index": frame_index,
                "actual_pts_seconds": actual,
                "time_base": time_base,
                "source_media_sha256": source_sha,
                "image_sha256": image_sha,
                "frame_path": Path(frame_path) if frame_path is not None else None,
            }
        )
    normalized.sort(
        key=lambda record: (
            record["actual_pts_seconds"],
            record["stream_index"],
            record["frame_index"],
        )
    )
    source_hashes = {record["source_media_sha256"] for record in normalized}
    if len(source_hashes) != 1:
        raise ValueError("inventory source hashes must match")
    return normalized


def _choose_nearest(
    candidates: Sequence[dict[str, Any]], desired_time_seconds: float
) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda record: (
            abs(record["actual_pts_seconds"] - desired_time_seconds),
            record["actual_pts_seconds"],
            record["stream_index"],
            record["frame_index"],
        ),
    )


def _make_selection(
    record: Mapping[str, Any],
    *,
    desired_time_seconds: float,
    slot_kind: str,
    support_span_id: str | None = None,
    fallback_reason: str | None = None,
) -> FrameSelection:
    return FrameSelection(
        requested_time_seconds=desired_time_seconds,
        actual_pts_seconds=record["actual_pts_seconds"],
        stream_index=record["stream_index"],
        frame_index=record["frame_index"],
        time_base=record["time_base"],
        source_media_sha256=record["source_media_sha256"],
        image_sha256=record.get("image_sha256"),
        frame_path=record.get("frame_path"),
        slot_kind=slot_kind,
        support_span_id=support_span_id,
        fallback_reason=fallback_reason,
    )


def _hamilton_allocations(
    spans: Sequence[TimelineSpan], target: int, caps: Sequence[int]
) -> list[int]:
    if not spans or target <= 0:
        return [0 for _ in spans]
    allocations = [min(4, cap) for cap in caps]
    if target < sum(allocations):
        allocations = [0 for _ in spans]
        for index in sorted(
            range(len(spans)),
            key=lambda item: (-caps[item], spans[item].start_seconds, item),
        )[:target]:
            allocations[index] = 1
        return allocations

    remaining = target - sum(allocations)
    while remaining:
        active = [
            index for index, cap in enumerate(caps) if allocations[index] < cap
        ]
        if not active:
            break
        total_duration = sum(
            spans[index].end_seconds - spans[index].start_seconds
            for index in active
        )
        quotas = {
            index: remaining
            * (spans[index].end_seconds - spans[index].start_seconds)
            / total_duration
            for index in active
        }
        floors = {index: math.floor(quota) for index, quota in quotas.items()}
        added = 0
        for index in active:
            amount = min(floors[index], caps[index] - allocations[index])
            allocations[index] += amount
            added += amount
        remaining -= added
        if remaining == 0:
            break
        order = sorted(
            active,
            key=lambda index: (
                -(quotas[index] - floors[index]),
                spans[index].start_seconds,
                index,
            ),
        )
        for index in order:
            if remaining == 0:
                break
            if allocations[index] < caps[index]:
                allocations[index] += 1
                remaining -= 1
        if added == 0 and not order:
            break
    return allocations


def build_global_local_frame_plan(
    duration_seconds: float,
    inventory: Sequence[Mapping[str, Any]],
    support_spans: Sequence[TimelineSpan] | ValidatedTimeline,
) -> EvidencePlan:
    """Allocate eight global and up to twenty-four nearest decoded local frames."""

    duration = _finite_number(
        duration_seconds, "duration_seconds", nonnegative=True
    )
    if duration < 1.0:
        raise ValueError("media duration must be at least one second")
    if isinstance(support_spans, ValidatedTimeline):
        spans = support_spans.support_spans
    else:
        spans = _normalize_support_spans(
            support_spans,
            duration_seconds=duration,
        )
    normalized_inventory = _normalize_inventory(duration, inventory)
    if len(normalized_inventory) < GLOBAL_FRAME_COUNT:
        raise ValueError("inventory cannot supply eight unique global frames")

    used: set[tuple[int, int]] = set()
    global_frames: list[FrameSelection] = []
    for requested_time in bin_midpoint_timestamps(duration, count=GLOBAL_FRAME_COUNT):
        candidates = [
            record
            for record in normalized_inventory
            if (record["stream_index"], record["frame_index"]) not in used
        ]
        if not candidates:
            raise ValueError("inventory cannot supply unique global frames")
        selected = _choose_nearest(candidates, requested_time)
        used.add((selected["stream_index"], selected["frame_index"]))
        global_frames.append(
            _make_selection(
                selected,
                desired_time_seconds=requested_time,
                slot_kind="global",
            )
        )

    caps = [
        max(
            1,
            math.floor(
                (span.end_seconds - span.start_seconds)
                * MAX_LOCAL_FRAMES_PER_SECOND
                + 1e-9
            ),
        )
        for span in spans
    ]
    target = min(
        MAX_LOCAL_FRAME_COUNT,
        sum(caps),
        max(0, len(normalized_inventory) - len(global_frames)),
    )
    allocations = _hamilton_allocations(spans, target, caps)
    local_frames: list[FrameSelection] = []
    for span, allocation in zip(spans, allocations, strict=True):
        width = (span.end_seconds - span.start_seconds) / allocation if allocation else 0
        for slot_index in range(allocation):
            requested_time = round(
                span.start_seconds + (slot_index + 0.5) * width, 6
            )
            in_span = [
                record
                for record in normalized_inventory
                if (record["stream_index"], record["frame_index"]) not in used
                and span.start_seconds <= record["actual_pts_seconds"] <= span.end_seconds
            ]
            fallback_reason = None
            candidates = in_span
            if not candidates:
                candidates = [
                    record
                    for record in normalized_inventory
                    if (record["stream_index"], record["frame_index"]) not in used
                ]
                fallback_reason = "local_collision_fallback"
            if not candidates:
                raise ValueError("inventory cannot supply unique local frames")
            selected = _choose_nearest(candidates, requested_time)
            used.add((selected["stream_index"], selected["frame_index"]))
            local_frames.append(
                _make_selection(
                    selected,
                    desired_time_seconds=requested_time,
                    slot_kind="local",
                    support_span_id=span.span_id,
                    fallback_reason=fallback_reason,
                )
            )
    return EvidencePlan(
        duration_seconds=duration,
        global_frames=tuple(global_frames),
        local_frames=tuple(local_frames),
        support_spans=spans,
    )


def select_overlapping_transcript_segments(
    timestamped_text: str | None,
    support_spans: Sequence[TimelineSpan] | ValidatedTimeline,
    *,
    duration_seconds: float,
) -> TranscriptSelection:
    """Select only verbatim timestamped lines whose inferred intervals overlap."""

    duration = _finite_number(
        duration_seconds, "duration_seconds", nonnegative=True
    )
    if duration <= 0:
        raise ValueError("duration must be positive")
    if timestamped_text is not None and not isinstance(timestamped_text, str):
        raise TypeError("timestamped_text must be a string or None")
    if isinstance(support_spans, ValidatedTimeline):
        spans = support_spans.support_spans
    else:
        spans = _normalize_support_spans(
            support_spans,
            duration_seconds=duration,
        )
    validated_spans = tuple(spans)
    segments = _parse_timestamped_text(timestamped_text, duration_seconds=duration)
    selected = tuple(
        segment
        for segment in segments
        if any(
            segment.start_seconds < span.end_seconds
            and segment.end_seconds > span.start_seconds
            for span in validated_spans
        )
    )
    text = "\n".join(segment.source_line for segment in selected)
    source_sha256 = hashlib.sha256(
        (timestamped_text or "").encode("utf-8")
    ).hexdigest()
    return TranscriptSelection(
        text=text,
        source_sha256=source_sha256,
        segments=selected,
    )


def _parse_timestamped_text(
    timestamped_text: str | None, *, duration_seconds: float
) -> tuple[WhisperSegment, ...]:
    """Parse both the repository's full clock and the compact task notation."""

    if timestamped_text is None or not timestamped_text.strip():
        return ()
    try:
        parsed_text = parse_timestamped_whisper(
            timestamped_text,
            duration_seconds=duration_seconds,
        )
    except ValueError:
        parsed_text = None
    if parsed_text is not None:
        return tuple(parsed_text)

    source_lines = timestamped_text.splitlines()
    compact_matches = [_SHORT_TIMESTAMP_LINE.fullmatch(line) for line in source_lines]
    if not source_lines or any(match is None for match in compact_matches):
        # Re-run the canonical parser so malformed full-clock input retains its
        # precise validation error and out-of-range messages.
        return tuple(
            parse_timestamped_whisper(
                timestamped_text,
                duration_seconds=duration_seconds,
            )
        )
    normalized_lines = [
        f"[00:{match.group('minutes')}:{match.group('seconds')}] {match.group('text')}"
        for match in compact_matches
        if match is not None
    ]
    normalized = parse_timestamped_whisper(
        "\n".join(normalized_lines),
        duration_seconds=duration_seconds,
    )
    return tuple(
        WhisperSegment(
            segment_id=segment.segment_id,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.text,
            source_line=source_line,
        )
        for segment, source_line in zip(normalized, source_lines, strict=True)
    )


def _verify_materialized_frames(evidence_plan: EvidencePlan) -> None:
    for index, frame in enumerate(evidence_plan.all_frames):
        if frame.frame_path is None:
            raise ValueError(f"frame {index} frame_path is required")
        if frame.image_sha256 is None:
            raise ValueError(f"frame {index} image_sha256 is required")
        if not frame.frame_path.is_file():
            raise ValueError(f"frame {index} frame_path does not exist")
        try:
            actual_sha256 = source_fingerprint(frame.frame_path).sha256
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"frame {index} frame_path cannot be fingerprinted"
            ) from exc
        if actual_sha256 != frame.image_sha256:
            raise ValueError(f"frame {index} image_sha256 does not match frame_path")


def write_evidence_manifest(
    destination: str | Path,
    evidence_plan: EvidencePlan,
) -> dict[str, Any]:
    """Write a canonical evidence manifest exactly once using an atomic link."""

    if not isinstance(evidence_plan, EvidencePlan):
        raise TypeError("evidence_plan must be an EvidencePlan")
    path = Path(destination)
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite existing evidence manifest: {path}")
    payload = evidence_plan.to_dict()
    audit_label_free_payload(payload)
    _verify_materialized_frames(evidence_plan)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ValueError(
                f"refusing to overwrite existing evidence manifest: {path}"
            ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return payload


__all__ = [
    "ALLOWED_MODALITIES",
    "EvidencePlan",
    "FrameSelection",
    "GLOBAL_FRAME_COUNT",
    "MAX_LOCAL_FRAME_COUNT",
    "TranscriptSelection",
    "ValidatedTimeline",
    "build_global_local_frame_plan",
    "bin_midpoint_timestamps",
    "parse_timestamped_whisper",
    "probe_media",
    "select_overlapping_transcript_segments",
    "source_fingerprint",
    "validate_timeline",
    "write_evidence_manifest",
]
