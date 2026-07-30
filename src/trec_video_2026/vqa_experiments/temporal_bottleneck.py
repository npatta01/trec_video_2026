"""Pure-Python contracts for temporal required-moment evidence diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ROLE_PRIMARY_REVIEWER = "primary_reviewer"
ROLE_ADJUDICATOR = "adjudicator"
ANNOTATION_ROLES = frozenset({ROLE_PRIMARY_REVIEWER, ROLE_ADJUDICATOR})

ELIGIBILITY_LABELABLE_VISUAL = "labelable_visual"
ELIGIBILITY_NOT_VISUALLY_ANSWERABLE = "not_visually_answerable"
ELIGIBILITY_AMBIGUOUS_QUESTION = "ambiguous_question"
ELIGIBILITY_INSUFFICIENT_MEDIA = "insufficient_media"
ELIGIBILITY_VALUES = frozenset(
    {
        ELIGIBILITY_LABELABLE_VISUAL,
        ELIGIBILITY_NOT_VISUALLY_ANSWERABLE,
        ELIGIBILITY_AMBIGUOUS_QUESTION,
        ELIGIBILITY_INSUFFICIENT_MEDIA,
    }
)

FORBIDDEN_KEYS = frozenset(
    {"answer", "reference", "prediction", "grade", "condition", "frame_timestamps"}
)
FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "answer",
        "answers",
        "reference",
        "references",
        "prediction",
        "predictions",
        "grade",
        "grades",
        "condition",
        "conditions",
    }
)
_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")

MAX_INTERVALS_PER_SUFFICIENT_SET = 6
MIN_ELIGIBLE_ROWS = 8
MIN_STRUCTURAL_AGREEMENT_ROWS = 8
MIN_MEDIAN_TEMPORAL_IOU = 0.50
MAX_BOUNDARY_DRIFT_SECONDS = 1.0
STRICT_RECALL_TOLERANCE_SECONDS = 0.0
PRIMARY_RECALL_TOLERANCE_SECONDS = 0.5
RECALL_TOLERANCES = frozenset(
    {STRICT_RECALL_TOLERANCE_SECONDS, PRIMARY_RECALL_TOLERANCE_SECONDS}
)
# Scoring consumes post-request extraction records that echo the final request hash.
REQUIRED_FRAME_PROVENANCE_FIELDS = (
    "actual_pts_seconds",
    "stream_index",
    "frame_index",
    "time_base",
    "sha256",
    "requested_time_seconds",
    "extraction_command_identity",
    "prompt_source_identity",
    "request_sha256",
    "runtime_identity",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _lowercase_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def probe_frame_inventory(
    ffprobe_payload: dict[str, Any], *, media_sha256: str
) -> list[dict[str, Any]]:
    """Normalize exact decoded-frame PTS values from a bound ffprobe payload."""

    expected_media_sha256 = _lowercase_sha256(media_sha256, "media_sha256")
    if not isinstance(ffprobe_payload, Mapping):
        raise TypeError("ffprobe_payload must be an object")
    payload_media_sha256 = _lowercase_sha256(
        ffprobe_payload.get("source_media_sha256"), "source_media_sha256"
    )
    if payload_media_sha256 != expected_media_sha256:
        raise ValueError("source_media_sha256 does not match frozen media")

    streams = ffprobe_payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("ffprobe payload must contain streams")
    time_base_by_stream: dict[int, str] = {}
    for stream_position, stream in enumerate(streams):
        if not isinstance(stream, Mapping):
            raise TypeError(f"ffprobe stream {stream_position} must be an object")
        if stream.get("codec_type") != "video":
            continue
        stream_index = stream.get("index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
        ):
            raise ValueError(f"ffprobe stream {stream_position} has invalid index")
        if stream_index in time_base_by_stream:
            raise ValueError("ffprobe streams contain a duplicate index")
        time_base = stream.get("time_base")
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"ffprobe stream {stream_index} is missing time_base")
        time_base_by_stream[stream_index] = time_base
    if not time_base_by_stream:
        raise ValueError("ffprobe payload has no video stream with time_base")

    frames = ffprobe_payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("ffprobe payload must contain decoded frames")
    inventory: list[dict[str, Any]] = []
    seen_frame_indexes: set[tuple[int, int]] = set()
    previous_pts_by_stream: dict[int, float] = {}
    for frame_position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise TypeError(f"ffprobe frame {frame_position} must be an object")
        if frame.get("media_type", "video") != "video":
            continue
        stream_index = frame.get("stream_index")
        if stream_index not in time_base_by_stream:
            raise ValueError(
                f"ffprobe frame {frame_position} has no matching video time_base"
            )
        frame_index = frame.get("frame_index", frame_position)
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(f"ffprobe frame {frame_position} has invalid frame_index")
        frame_key = (stream_index, frame_index)
        if frame_key in seen_frame_indexes:
            raise ValueError("ffprobe frames contain a duplicate frame_index")
        seen_frame_indexes.add(frame_key)

        if "best_effort_timestamp_time" not in frame:
            raise ValueError(f"ffprobe frame {frame_position} is missing decoded PTS")
        try:
            pts = float(frame["best_effort_timestamp_time"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ffprobe frame {frame_position} decoded PTS must be numeric"
            ) from exc
        if not math.isfinite(pts):
            raise ValueError(
                f"ffprobe frame {frame_position} decoded PTS must be finite"
            )
        canonical_pts = round(pts, 6)
        previous_pts = previous_pts_by_stream.get(stream_index)
        if previous_pts is not None:
            if canonical_pts == previous_pts:
                raise ValueError("ffprobe frames contain a duplicate decoded PTS")
            if canonical_pts < previous_pts:
                raise ValueError("ffprobe decoded PTS must be strictly monotonic")
        previous_pts_by_stream[stream_index] = canonical_pts
        inventory.append(
            {
                "stream_index": stream_index,
                "frame_index": frame_index,
                "actual_pts_seconds": canonical_pts,
                "time_base": time_base_by_stream[stream_index],
                "source_media_sha256": expected_media_sha256,
            }
        )
    if not inventory:
        raise ValueError("ffprobe payload has no decoded video frames")
    return inventory


def _planning_inventory(
    duration: float, inventory: Any
) -> tuple[float, list[dict[str, Any]]]:
    normalized_duration = _numeric(duration, "duration")
    if normalized_duration <= 0:
        raise ValueError("duration must be positive")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("inventory must contain decoded frames")
    normalized: list[dict[str, Any]] = []
    frame_indexes: set[tuple[int, int]] = set()
    pts_values: set[float] = set()
    source_media_sha256: str | None = None
    previous_pts: float | None = None
    for position, record in enumerate(inventory):
        if not isinstance(record, Mapping):
            raise TypeError(f"inventory record {position} must be an object")
        stream_index = record.get("stream_index")
        frame_index = record.get("frame_index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
        ):
            raise ValueError(f"inventory record {position} has invalid stream_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(f"inventory record {position} has invalid frame_index")
        frame_key = (stream_index, frame_index)
        if frame_key in frame_indexes:
            raise ValueError("inventory contains a duplicate frame_index")
        frame_indexes.add(frame_key)
        actual_pts_seconds = round(
            _numeric(
                record.get("actual_pts_seconds"),
                f"inventory record {position} actual_pts_seconds",
            ),
            6,
        )
        if not 0 <= actual_pts_seconds <= normalized_duration:
            raise ValueError(
                f"inventory record {position} actual_pts_seconds is outside duration"
            )
        if actual_pts_seconds in pts_values:
            raise ValueError("inventory contains a duplicate decoded PTS")
        if previous_pts is not None and actual_pts_seconds < previous_pts:
            raise ValueError("inventory decoded PTS must be monotonic")
        previous_pts = actual_pts_seconds
        pts_values.add(actual_pts_seconds)
        time_base = record.get("time_base")
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"inventory record {position} is missing time_base")
        record_media_sha256 = _lowercase_sha256(
            record.get("source_media_sha256"),
            f"inventory record {position} source_media_sha256",
        )
        if source_media_sha256 is None:
            source_media_sha256 = record_media_sha256
        elif record_media_sha256 != source_media_sha256:
            raise ValueError("inventory source_media_sha256 values do not match")
        normalized.append(
            {
                "stream_index": stream_index,
                "frame_index": frame_index,
                "actual_pts_seconds": actual_pts_seconds,
                "time_base": time_base,
                "source_media_sha256": record_media_sha256,
            }
        )
    return normalized_duration, normalized


def _desired_times(start: float, end: float, count: int) -> list[float]:
    width = (end - start) / count
    return [round(start + (index + 0.5) * width, 6) for index in range(count)]


def _bind_slots(
    slots: Sequence[tuple[str, float, tuple[float, float] | None]],
    inventory: Sequence[dict[str, Any]],
    *,
    used_frame_indexes: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    used = used_frame_indexes if used_frame_indexes is not None else set()
    plan: list[dict[str, Any]] = []
    for slot_kind, desired_time, interval in slots:
        candidates = [
            frame
            for frame in inventory
            if (frame["stream_index"], frame["frame_index"]) not in used
            and (
                interval is None
                or interval[0] <= frame["actual_pts_seconds"] <= interval[1]
            )
        ]
        if not candidates:
            qualifier = "local interval" if interval is not None else "global slots"
            raise ValueError(f"cannot construct {qualifier} with unique decoded frames")
        selected = min(
            candidates,
            key=lambda frame: (
                abs(frame["actual_pts_seconds"] - desired_time),
                frame["actual_pts_seconds"],
                frame["stream_index"],
                frame["frame_index"],
            ),
        )
        used.add((selected["stream_index"], selected["frame_index"]))
        plan_record = {
            "slot_kind": slot_kind,
            "desired_time_seconds": desired_time,
            "requested_time_seconds": desired_time,
            "stream_index": selected["stream_index"],
            "frame_index": selected["frame_index"],
            "actual_pts_seconds": selected["actual_pts_seconds"],
            "time_base": selected["time_base"],
            "source_media_sha256": selected["source_media_sha256"],
        }
        plan_record["frame_plan_sha256"] = _sha256(plan_record)
        plan.append(plan_record)
    return plan


def uniform_decoded_frame_plan(
    duration: float, inventory: list[dict[str, Any]], count: int = 12
) -> list[dict[str, Any]]:
    """Bind equal-bin midpoint requests to unique decoded frames."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    normalized_duration, normalized_inventory = _planning_inventory(duration, inventory)
    if len(normalized_inventory) < count:
        raise ValueError(f"inventory cannot supply {count} unique decoded frames")
    slots = [
        ("global", desired_time, None)
        for desired_time in _desired_times(0.0, normalized_duration, count)
    ]
    return sorted(
        _bind_slots(slots, normalized_inventory),
        key=lambda frame: (
            frame["actual_pts_seconds"],
            frame["desired_time_seconds"],
        ),
    )


def _normalized_sufficient_sets(
    sufficient_sets: Any, *, duration: float
) -> list[list[tuple[float, float]]]:
    if isinstance(sufficient_sets, Mapping):
        forbidden = _forbidden_key(sufficient_sets)
        if forbidden is not None:
            raise ValueError(f"answer-bearing annotation key is forbidden: {forbidden}")
        raise TypeError("sufficient_sets must be the normalized interval list only")
    if not isinstance(sufficient_sets, list) or not sufficient_sets:
        raise ValueError("sufficient_sets must be a nonempty list")
    normalized_sets: list[list[tuple[float, float]]] = []
    for set_index, sufficient_set in enumerate(sufficient_sets):
        if not isinstance(sufficient_set, list) or not sufficient_set:
            raise ValueError(f"sufficient set {set_index} must not be empty")
        if len(sufficient_set) > 6:
            raise ValueError("sufficient sets may contain at most six intervals")
        normalized_set: list[tuple[float, float]] = []
        for interval_index, interval in enumerate(sufficient_set):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(
                    f"sufficient set {set_index} interval {interval_index} "
                    "must contain two bounds"
                )
            start = _numeric(interval[0], "interval start_seconds")
            end = _numeric(interval[1], "interval end_seconds")
            if start < 0 or end <= start or end > duration:
                raise ValueError("sufficient interval is outside duration")
            normalized_set.append((start, end))
        normalized_sets.append(normalized_set)
    return normalized_sets


def _hamilton_local_allocations(
    intervals: Sequence[tuple[float, float]], *, slot_count: int
) -> list[int]:
    if len(intervals) > slot_count:
        raise ValueError("cannot allocate a local slot to every interval")
    allocations = [1] * len(intervals)
    remaining = slot_count - len(intervals)
    if not remaining:
        return allocations
    durations = [end - start for start, end in intervals]
    total_duration = sum(durations)
    quotas = [
        remaining * interval_duration / total_duration
        for interval_duration in durations
    ]
    floors = [math.floor(quota) for quota in quotas]
    allocations = [
        base + floor for base, floor in zip(allocations, floors, strict=True)
    ]
    leftover = remaining - sum(floors)
    remainder_order = sorted(
        range(len(intervals)),
        key=lambda index: (
            -(quotas[index] - floors[index]),
            intervals[index][0],
            index,
        ),
    )
    for index in remainder_order[:leftover]:
        allocations[index] += 1
    return allocations


def oracle_hybrid_decoded_frame_plan(
    duration: float,
    sufficient_sets: list[list[list[float]]],
    inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind six local oracle slots and six label-free global context slots."""

    normalized_duration, normalized_inventory = _planning_inventory(duration, inventory)
    if len(normalized_inventory) < 12:
        raise ValueError("inventory cannot supply 12 unique decoded frames")
    normalized_sets = _normalized_sufficient_sets(
        sufficient_sets, duration=normalized_duration
    )
    selected_set = min(
        normalized_sets,
        key=lambda intervals: (
            sum(end - start for start, end in intervals),
            min(start for start, _ in intervals),
        ),
    )
    intervals = sorted(selected_set)
    allocations = _hamilton_local_allocations(intervals, slot_count=6)
    local_slots = [
        ("local", desired_time, interval)
        for interval, allocation in zip(intervals, allocations, strict=True)
        for desired_time in _desired_times(interval[0], interval[1], allocation)
    ]
    used: set[tuple[int, int]] = set()
    local_plan = _bind_slots(local_slots, normalized_inventory, used_frame_indexes=used)
    global_slots = [
        ("global", desired_time, None)
        for desired_time in _desired_times(0.0, normalized_duration, 6)
    ]
    global_plan = _bind_slots(
        global_slots, normalized_inventory, used_frame_indexes=used
    )
    return sorted(
        [*local_plan, *global_plan],
        key=lambda frame: (
            frame["actual_pts_seconds"],
            frame["desired_time_seconds"],
            frame["slot_kind"],
        ),
    )


TEMPORAL_CONDITIONS = ("uniform-12", "oracle-hybrid-12")
FRAME_PLAN_BINDING_FIELDS = (
    "slot_kind",
    "desired_time_seconds",
    "requested_time_seconds",
    "stream_index",
    "frame_index",
    "actual_pts_seconds",
    "time_base",
    "source_media_sha256",
)
REQUIRED_PRE_REQUEST_FRAME_PROVENANCE_FIELDS = (
    *FRAME_PLAN_BINDING_FIELDS,
    "frame_plan_sha256",
    "sha256",
    "extraction_command_identity",
    "prompt_source_identity",
    "runtime_identity",
)
INFERENCE_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "runtime_image",
    "prompt_source_identity",
    "runtime_identity",
    "decoding",
    "seed",
    "image_resolution",
)
_PINNED_RUNTIME_IMAGE = re.compile(r".+@sha256:[0-9a-f]{64}")


def _label_material_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                tokens = {
                    token.casefold() for token in _KEY_BOUNDARY.split(key) if token
                }
                if tokens & {
                    "answer",
                    "answers",
                    "reference",
                    "references",
                    "prediction",
                    "predictions",
                    "grade",
                    "grades",
                    "span",
                    "spans",
                } or key in {"sufficient_sets", "preferred_set_index"}:
                    return key
            found = _label_material_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _label_material_key(nested)
            if found is not None:
                return found
    return None


def _material_identity(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a material identity object")
    identity = value.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"{field}.identity must be a nonempty string")
    _lowercase_sha256(value.get("sha256"), f"{field}.sha256")
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
    return copy.deepcopy(dict(value))


def _validated_inference_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("inference_identity must be a nonempty object")
    for field in INFERENCE_IDENTITY_FIELDS:
        if field not in value:
            raise ValueError(f"inference_identity is missing {field}")
    for field in ("model_id", "model_revision"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"inference_identity {field} must be nonempty")
    runtime_image = value["runtime_image"]
    if (
        not isinstance(runtime_image, str)
        or _PINNED_RUNTIME_IMAGE.fullmatch(runtime_image) is None
    ):
        raise ValueError("inference_identity runtime_image must be digest-pinned")
    _material_identity(value["prompt_source_identity"], "prompt_source_identity")
    _material_identity(value["runtime_identity"], "runtime_identity")
    decoding = value["decoding"]
    if not isinstance(decoding, Mapping) or not decoding:
        raise ValueError("inference_identity decoding must be a nonempty object")
    seed = value["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("inference_identity seed must be an integer")
    image_resolution = value["image_resolution"]
    if (
        isinstance(image_resolution, bool)
        or not isinstance(image_resolution, int)
        or image_resolution <= 0
    ):
        raise ValueError("inference_identity image_resolution must be positive")
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("inference_identity must be JSON serializable") from exc
    return copy.deepcopy(dict(value))


def _request_frames(
    records: Any,
    *,
    video_id: str,
    duration: float,
    media_sha256: str,
    inference_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[float]]:
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError(f"{video_id} condition must contain exactly 12 frames")
    frames: list[dict[str, Any]] = []
    hashes: list[str] = []
    timestamps: list[float] = []
    frame_indexes: set[tuple[int, int]] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"{video_id} frame {position} must be an object")
        for field in REQUIRED_PRE_REQUEST_FRAME_PROVENANCE_FIELDS:
            if field not in record:
                raise ValueError(f"{video_id} frame {position} is missing {field}")
        leaked = _label_material_key(record)
        if leaked is not None:
            raise ValueError(f"forbidden label key in frame plan: {leaked}")
        stream_index = record.get("stream_index")
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
        ):
            raise ValueError(f"{video_id} frame {position} has invalid stream_index")
        frame_index = record.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(f"{video_id} frame {position} has invalid frame_index")
        frame_key = (stream_index, frame_index)
        if frame_key in frame_indexes:
            raise ValueError(f"{video_id} condition has a duplicate frame_index")
        frame_indexes.add(frame_key)
        timestamp = round(
            _numeric(
                record.get("actual_pts_seconds"),
                f"{video_id} frame {position} actual_pts_seconds",
            ),
            6,
        )
        if not 0 <= timestamp <= duration:
            raise ValueError(f"{video_id} frame {position} PTS is outside duration")
        desired_time = round(
            _numeric(
                record["desired_time_seconds"],
                f"{video_id} frame {position} desired_time_seconds",
            ),
            6,
        )
        requested_time = round(
            _numeric(
                record["requested_time_seconds"],
                f"{video_id} frame {position} requested_time_seconds",
            ),
            6,
        )
        if not 0 <= desired_time <= duration or not 0 <= requested_time <= duration:
            raise ValueError(
                f"{video_id} frame {position} requested time is outside duration"
            )
        slot_kind = record["slot_kind"]
        if slot_kind not in {"global", "local"}:
            raise ValueError(f"{video_id} frame {position} has invalid slot_kind")
        frame_sha256 = _lowercase_sha256(
            record.get("sha256"), f"{video_id} frame {position} sha256"
        )
        if record.get("source_media_sha256") != media_sha256:
            raise ValueError(
                f"{video_id} frame {position} source_media_sha256 mismatch"
            )
        time_base = record.get("time_base")
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"{video_id} frame {position} is missing time_base")
        frame_plan = {
            "slot_kind": slot_kind,
            "desired_time_seconds": desired_time,
            "requested_time_seconds": requested_time,
            "stream_index": stream_index,
            "frame_index": frame_index,
            "actual_pts_seconds": timestamp,
            "time_base": time_base,
            "source_media_sha256": media_sha256,
        }
        frame_plan_sha256 = _lowercase_sha256(
            record["frame_plan_sha256"],
            f"{video_id} frame {position} frame_plan_sha256",
        )
        if frame_plan_sha256 != _sha256(frame_plan):
            raise ValueError(f"{video_id} frame {position} frame_plan_sha256 mismatch")
        extraction_command_identity = _material_identity(
            record["extraction_command_identity"], "extraction_command_identity"
        )
        prompt_source_identity = _material_identity(
            record["prompt_source_identity"], "prompt_source_identity"
        )
        runtime_identity = _material_identity(
            record["runtime_identity"], "runtime_identity"
        )
        if prompt_source_identity != inference_identity["prompt_source_identity"]:
            raise ValueError(
                f"{video_id} frame {position} prompt_source_identity mismatch"
            )
        if runtime_identity != inference_identity["runtime_identity"]:
            raise ValueError(f"{video_id} frame {position} runtime_identity mismatch")
        frames.append(
            {
                "stream_index": stream_index,
                "frame_index": frame_index,
                "actual_pts_seconds": timestamp,
                "time_base": time_base,
                "source_media_sha256": media_sha256,
                "sha256": frame_sha256,
                "requested_time_seconds": requested_time,
                "extraction_command_identity": extraction_command_identity,
                "prompt_source_identity": prompt_source_identity,
                "frame_plan_sha256": frame_plan_sha256,
                "runtime_identity": runtime_identity,
            }
        )
        hashes.append(frame_sha256)
        timestamps.append(timestamp)
    if len(set(hashes)) != 12:
        raise ValueError(f"{video_id} condition must have 12 unique frame hashes")
    if len(set(timestamps)) != 12:
        raise ValueError(f"{video_id} condition must have 12 unique decoded PTS values")
    if timestamps != sorted(timestamps):
        raise ValueError(f"{video_id} condition frames must be chronological")
    return frames, hashes, timestamps


def build_temporal_requests(
    panel: dict[str, Any],
    condition_frames: dict[str, dict[str, list[dict[str, Any]]]],
    inference_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build matched, label-free uniform and oracle-hybrid inference requests."""

    if not isinstance(panel, Mapping):
        raise TypeError("panel must be an object")
    leaked = _label_material_key(panel)
    if leaked is not None:
        raise ValueError(f"forbidden label key in panel: {leaked}")
    if panel.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    panel_id = panel.get("panel_id")
    if not isinstance(panel_id, str) or not panel_id.strip():
        raise ValueError("panel_id must be a nonempty string")
    panel_rows = _panel_rows(panel)
    examples = panel.get("examples", panel.get("rows"))
    assert isinstance(examples, list)
    if not isinstance(condition_frames, Mapping):
        raise TypeError("condition_frames must be an object")
    if set(condition_frames) != set(TEMPORAL_CONDITIONS):
        raise ValueError(f"condition set must be exactly {set(TEMPORAL_CONDITIONS)}")
    normalized_inference_identity = _validated_inference_identity(inference_identity)
    leaked = _label_material_key(inference_identity)
    if leaked is not None:
        raise ValueError(f"forbidden label key in inference_identity: {leaked}")

    expected_video_ids = {video_id for video_id, _ in panel_rows}
    for condition in TEMPORAL_CONDITIONS:
        by_video = condition_frames[condition]
        if not isinstance(by_video, Mapping) or set(by_video) != expected_video_ids:
            raise ValueError(f"{condition} frame rows must match the frozen panel")

    panel_sha256 = _sha256(panel)
    requests: list[dict[str, Any]] = []
    for example, (video_id, duration) in zip(examples, panel_rows, strict=True):
        assert isinstance(example, Mapping)
        question = example.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{video_id} lacks a question")
        media_sha256 = _lowercase_sha256(
            example.get("media_sha256"), f"{video_id} media_sha256"
        )
        for condition in TEMPORAL_CONDITIONS:
            frames, frame_sha256s, frame_timestamps = _request_frames(
                condition_frames[condition][video_id],
                video_id=video_id,
                duration=duration,
                media_sha256=media_sha256,
                inference_identity=normalized_inference_identity,
            )
            request = {
                "schema_version": SCHEMA_VERSION,
                "panel_id": panel_id,
                "panel_sha256": panel_sha256,
                "video_id": video_id,
                "condition": condition,
                "condition_id": condition,
                "question": question.strip(),
                "source_media_sha256": media_sha256,
                "frames": frames,
                "frame_sha256s": frame_sha256s,
                "frame_timestamps": frame_timestamps,
                "frame_count": 12,
                "inference_identity": copy.deepcopy(normalized_inference_identity),
                "transcript": None,
                "media_path": None,
                "audio_path": None,
                "audio_mode": None,
            }
            request["request_sha256"] = _sha256(request)
            requests.append(request)
    return requests


def _read_json_artifact(path: Any) -> tuple[bytes, Mapping[str, Any]]:
    artifact_path = Path(path)
    try:
        raw = artifact_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read JSON artifact: {artifact_path.name}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("JSON artifact must contain a top-level object")
    return raw, payload


def freeze_json_artifact(path: Any, *, bindings: dict[str, Any]) -> dict[str, Any]:
    """Return a read-only content freeze after checking top-level bindings."""

    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("bindings must be a nonempty object")
    try:
        bindings_sha256 = _sha256(bindings)
    except (TypeError, ValueError) as exc:
        raise ValueError("bindings must be JSON serializable") from exc
    raw, payload = _read_json_artifact(path)
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"artifact {key} does not match freeze binding")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "bindings": copy.deepcopy(dict(bindings)),
        "bindings_sha256": bindings_sha256,
    }


def verify_json_artifact(path: Any, freeze_record: dict[str, Any]) -> None:
    """Verify frozen JSON bytes, the freeze record, and live bindings."""

    if not isinstance(freeze_record, Mapping):
        raise TypeError("freeze_record must be an object")
    if freeze_record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"freeze schema_version must be {SCHEMA_VERSION}")
    if freeze_record.get("status") != "frozen":
        raise ValueError("freeze status must be frozen")
    bindings = freeze_record.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("freeze bindings must be a nonempty object")
    if freeze_record.get("bindings_sha256") != _sha256(bindings):
        raise ValueError("freeze bindings_sha256 does not match bindings")
    raw, payload = _read_json_artifact(path)
    if freeze_record.get("artifact_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("freeze artifact_sha256 does not match artifact")
    for key, expected in bindings.items():
        if payload.get(key) != expected:
            raise ValueError(f"artifact {key} does not match freeze binding")


def _numeric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                tokens = {
                    token.casefold() for token in _KEY_BOUNDARY.split(key) if token
                }
                frame_timestamp = "frame" in tokens and bool(
                    tokens & {"timestamp", "timestamps"}
                )
                if (
                    key.casefold() in FORBIDDEN_KEYS
                    or tokens & FORBIDDEN_KEY_TOKENS
                    or frame_timestamp
                ):
                    return key
            found = _forbidden_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _forbidden_key(nested)
            if found is not None:
                return found
    return None


def _panel_rows(panel: Mapping[str, Any]) -> list[tuple[str, float]]:
    if not isinstance(panel, Mapping):
        raise TypeError("panel must be an object")
    rows = panel.get("examples", panel.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("panel must contain rows")

    result: list[tuple[str, float]] = []
    ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"panel row {index} must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"panel row {index} has an invalid video_id")
        if video_id in ids:
            raise ValueError("panel has duplicate video IDs")
        ids.add(video_id)
        duration_value = row.get("duration_seconds", row.get("duration"))
        duration = _numeric(duration_value, f"panel row {index} duration_seconds")
        if duration <= 0:
            raise ValueError(f"panel row {index} duration_seconds must be positive")
        result.append((video_id, duration))
    return result


def _annotation_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    hashed = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in {"annotation_sha256", "normalized_annotation_sha256"}
    }
    rows = hashed.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                # This is copied from the frozen panel during normalization; it
                # is not reviewer-authored annotation content.
                row.pop("duration_seconds", None)
    return hashed


def _normalized_annotation_hash_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key != "normalized_annotation_sha256"
    }


def _normalize_interval(
    interval: Any,
    *,
    duration: float,
    row_index: int,
    set_index: int,
    interval_index: int,
) -> list[float]:
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(
            f"row {row_index} sufficient set {set_index} interval {interval_index} "
            "must contain [start_seconds, end_seconds]"
        )
    start = _numeric(interval[0], "interval start_seconds")
    end = _numeric(interval[1], "interval end_seconds")
    if start < 0:
        raise ValueError("interval start_seconds must be nonnegative")
    if end <= start:
        raise ValueError("interval end_seconds must be after start_seconds")
    if end > duration:
        raise ValueError("interval end_seconds exceeds frozen duration")
    if not math.isclose(
        start * 10, round(start * 10), abs_tol=1e-9
    ) or not math.isclose(end * 10, round(end * 10), abs_tol=1e-9):
        raise ValueError("interval bounds must be recorded to 0.1 seconds")
    return [round(start, 1), round(end, 1)]


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an ISO-8601 UTC timestamp ending in Z"
        ) from exc


def validate_annotation(
    payload: dict[str, Any], panel: dict[str, Any], *, expected_role: str
) -> dict[str, Any]:
    """Validate and normalize one label-blind temporal annotation payload.

    The annotation is bound to the exact canonical panel payload.  Rows must
    already follow the frozen panel order; normalization converts valid numeric
    endpoints to one decimal place and returns a canonical annotation hash.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("annotation payload must be an object")
    forbidden = _forbidden_key(payload)
    if forbidden is not None:
        raise ValueError(f"forbidden key in annotation: {forbidden}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if expected_role not in ANNOTATION_ROLES:
        raise ValueError(f"expected_role must be one of {sorted(ANNOTATION_ROLES)}")
    role = payload.get("role")
    if role not in ANNOTATION_ROLES or role != expected_role:
        raise ValueError(f"annotation role must be {expected_role}")
    for field in ("reviewer", "rubric_revision", "packet_sha256"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} must be a nonempty string")

    started_at = _utc_timestamp(payload.get("started_at"), "started_at")
    ended_at = _utc_timestamp(payload.get("ended_at"), "ended_at")
    if ended_at <= started_at:
        raise ValueError("ended_at must be after started_at")

    input_hashes_field = "input_annotation_sha256s"
    if role == ROLE_PRIMARY_REVIEWER:
        if input_hashes_field in payload:
            raise ValueError(
                "primary reviewer annotation must not contain input_annotation_sha256s"
            )
    else:
        input_hashes = payload.get(input_hashes_field)
        if not isinstance(input_hashes, list) or len(input_hashes) != 3:
            raise ValueError(
                "input_annotation_sha256s must be a list of exactly three hashes"
            )
        normalized_input_hashes = [
            _lowercase_sha256(value, f"input_annotation_sha256s[{index}]")
            for index, value in enumerate(input_hashes)
        ]
        if len(set(normalized_input_hashes)) != 3:
            raise ValueError("input_annotation_sha256s must contain distinct hashes")

    expected_packet_sha256 = _sha256(panel)
    if payload["packet_sha256"] != expected_packet_sha256:
        raise ValueError("packet_sha256 does not match frozen panel")

    panel_rows = _panel_rows(panel)
    expected_ids = [video_id for video_id, _ in panel_rows]
    duration_by_id = dict(panel_rows)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("annotation rows must be a list")
    actual_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"annotation row {index} must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"annotation row {index} has an invalid video_id")
        actual_ids.append(video_id)
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("annotation has duplicate video IDs")
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("annotation rows must match frozen panel rows")
    if actual_ids != expected_ids:
        raise ValueError("annotation rows must use frozen panel order")

    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        normalized = copy.deepcopy(dict(row))
        duration = duration_by_id[row["video_id"]]
        if "duration_seconds" in row:
            supplied_duration = _numeric(
                row["duration_seconds"], f"row {row_index} duration_seconds"
            )
            if supplied_duration != duration:
                raise ValueError(
                    f"row {row_index} duration_seconds does not match frozen panel"
                )
        normalized["duration_seconds"] = duration
        eligibility = row.get("eligibility")
        if eligibility not in ELIGIBILITY_VALUES:
            raise ValueError(f"row {row_index} has invalid eligibility")
        if eligibility != ELIGIBILITY_LABELABLE_VISUAL:
            if "sufficient_sets" in row or "preferred_set_index" in row:
                raise ValueError(
                    "ineligible rows must not contain sufficient intervals"
                )
            normalized_rows.append(normalized)
            continue

        sufficient_sets = row.get("sufficient_sets")
        if not isinstance(sufficient_sets, list) or not sufficient_sets:
            raise ValueError("labelable rows require nonempty sufficient sets")
        normalized_sets: list[list[list[float]]] = []
        for set_index, sufficient_set in enumerate(sufficient_sets):
            if not isinstance(sufficient_set, list) or not sufficient_set:
                raise ValueError("sufficient sets must not be empty")
            if len(sufficient_set) > MAX_INTERVALS_PER_SUFFICIENT_SET:
                raise ValueError("sufficient sets may contain at most six intervals")
            normalized_sets.append(
                [
                    _normalize_interval(
                        interval,
                        duration=duration,
                        row_index=row_index,
                        set_index=set_index,
                        interval_index=interval_index,
                    )
                    for interval_index, interval in enumerate(sufficient_set)
                ]
            )
        preferred_set_index = row.get("preferred_set_index")
        if (
            isinstance(preferred_set_index, bool)
            or not isinstance(preferred_set_index, int)
            or not 0 <= preferred_set_index < len(normalized_sets)
        ):
            raise ValueError("labelable rows require a valid preferred_set_index")
        normalized["sufficient_sets"] = normalized_sets
        normalized_rows.append(normalized)

    normalized_payload = copy.deepcopy(dict(payload))
    normalized_payload["rows"] = normalized_rows
    supplied_hash = payload.get("annotation_sha256")
    computed_hash = _sha256(_annotation_hash_payload(payload))
    if supplied_hash is not None and supplied_hash != computed_hash:
        raise ValueError("annotation_sha256 does not match annotation payload")
    normalized_payload["annotation_sha256"] = computed_hash
    supplied_normalized_hash = payload.get("normalized_annotation_sha256")
    computed_normalized_hash = _sha256(
        _normalized_annotation_hash_payload(normalized_payload)
    )
    if (
        supplied_normalized_hash is not None
        and supplied_normalized_hash != computed_normalized_hash
    ):
        raise ValueError(
            "normalized_annotation_sha256 does not match normalized annotation"
        )
    normalized_payload["normalized_annotation_sha256"] = computed_normalized_hash
    return normalized_payload


def _merged_intervals(
    intervals: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> float:
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        total += max(0.0, end - start)
        if left[left_index][1] <= right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def _temporal_iou(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    left_merged = _merged_intervals(left)
    right_merged = _merged_intervals(right)
    intersection = _intersection_duration(left_merged, right_merged)
    union = _duration(left_merged) + _duration(right_merged) - intersection
    return 0.0 if union == 0 else intersection / union


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _interquartile_range(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return 0.0
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[-midpoint:]
    assert lower and upper
    return _median(upper) - _median(lower)  # type: ignore[operator]


def annotation_agreement(
    left: dict[str, Any], right: dict[str, Any], panel: dict[str, Any]
) -> dict[str, Any]:
    """Calculate primary-reviewer agreement for frozen temporal annotations."""

    normalized_left = validate_annotation(
        left, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )
    normalized_right = validate_annotation(
        right, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )
    rows: list[dict[str, Any]] = []
    temporal_ious: list[float] = []
    eligibility_match_count = 0
    structural_match_count = 0

    for left_row, right_row in zip(
        normalized_left["rows"], normalized_right["rows"], strict=True
    ):
        left_eligibility = left_row["eligibility"]
        right_eligibility = right_row["eligibility"]
        flags: list[str] = []
        eligibility_match = left_eligibility == right_eligibility
        if eligibility_match:
            eligibility_match_count += 1
        else:
            flags.append("eligibility_mismatch")

        left_count = 0
        right_count = 0
        temporal_iou: float | None = None
        maximum_boundary_drift: float | None = None
        structural_match = eligibility_match
        if left_eligibility == ELIGIBILITY_LABELABLE_VISUAL:
            left_set = left_row["sufficient_sets"][left_row["preferred_set_index"]]
            left_count = len(left_set)
        else:
            left_set = []
        if right_eligibility == ELIGIBILITY_LABELABLE_VISUAL:
            right_set = right_row["sufficient_sets"][right_row["preferred_set_index"]]
            right_count = len(right_set)
        else:
            right_set = []

        if (
            eligibility_match
            and left_eligibility == ELIGIBILITY_LABELABLE_VISUAL
            and left_count != right_count
        ):
            structural_match = False
            flags.append("topology_mismatch")
        if structural_match:
            structural_match_count += 1

        if structural_match and left_eligibility == ELIGIBILITY_LABELABLE_VISUAL:
            temporal_iou = _temporal_iou(left_set, right_set)
            temporal_ious.append(temporal_iou)
            if temporal_iou < MIN_MEDIAN_TEMPORAL_IOU:
                flags.append("temporal_iou_below_threshold")
            ordered_left = sorted((float(start), float(end)) for start, end in left_set)
            ordered_right = sorted(
                (float(start), float(end)) for start, end in right_set
            )
            maximum_boundary_drift = max(
                abs(left_boundary - right_boundary)
                for left_interval, right_interval in zip(
                    ordered_left, ordered_right, strict=True
                )
                for left_boundary, right_boundary in zip(
                    left_interval, right_interval, strict=True
                )
            )
            if maximum_boundary_drift > MAX_BOUNDARY_DRIFT_SECONDS:
                flags.append("boundary_drift_above_one_second")

        rows.append(
            {
                "video_id": left_row["video_id"],
                "left_eligibility": left_eligibility,
                "right_eligibility": right_eligibility,
                "left_preferred_interval_count": left_count,
                "right_preferred_interval_count": right_count,
                "eligibility_match": eligibility_match,
                "structural_match": structural_match,
                "temporal_iou": temporal_iou,
                "maximum_boundary_drift_seconds": maximum_boundary_drift,
                "flags": flags,
            }
        )

    median_temporal_iou = _median(temporal_ious)
    structural_threshold_passes = (
        structural_match_count >= MIN_STRUCTURAL_AGREEMENT_ROWS
    )
    median_temporal_iou_passes = (
        median_temporal_iou is not None
        and median_temporal_iou >= MIN_MEDIAN_TEMPORAL_IOU
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "left_annotation_sha256": normalized_left["annotation_sha256"],
        "right_annotation_sha256": normalized_right["annotation_sha256"],
        "rows": rows,
        "row_count": len(rows),
        "eligibility_match_count": eligibility_match_count,
        "structural_match_count": structural_match_count,
        "eligible_structural_match_count": len(temporal_ious),
        "flagged_row_count": sum(bool(row["flags"]) for row in rows),
        "structural_match_minimum": MIN_STRUCTURAL_AGREEMENT_ROWS,
        "structural_threshold_passes": structural_threshold_passes,
        "median_temporal_iou": median_temporal_iou,
        "temporal_iou_iqr": _interquartile_range(temporal_ious),
        "median_temporal_iou_minimum": MIN_MEDIAN_TEMPORAL_IOU,
        "median_temporal_iou_passes": median_temporal_iou_passes,
        "agreement_passes": structural_threshold_passes and median_temporal_iou_passes,
    }


def _actual_pts_records(
    records: Any, *, video_id: str, duration: float
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError(f"frames for {video_id} must be a list")
    normalized: list[dict[str, Any]] = []
    frame_indexes: set[tuple[int, int]] = set()
    frame_hashes: set[str] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"frame record {record_index} for {video_id} must be an object"
            )
        for field in REQUIRED_FRAME_PROVENANCE_FIELDS:
            if field not in record:
                raise ValueError(
                    f"frame record {record_index} for {video_id} is missing {field}"
                )
        actual_pts_seconds = _numeric(
            record["actual_pts_seconds"],
            f"frame record {record_index} actual_pts_seconds",
        )
        if not 0 <= actual_pts_seconds <= duration:
            raise ValueError(
                f"frame record {record_index} actual_pts_seconds is outside duration"
            )
        stream_index = record["stream_index"]
        if (
            isinstance(stream_index, bool)
            or not isinstance(stream_index, int)
            or stream_index < 0
        ):
            raise ValueError(
                f"frame record {record_index} stream_index must be a nonnegative integer"
            )
        frame_index = record["frame_index"]
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ValueError(
                f"frame record {record_index} frame_index must be a nonnegative integer"
            )
        frame_key = (stream_index, frame_index)
        if frame_key in frame_indexes:
            raise ValueError(f"frames for {video_id} contain a duplicate frame_index")
        frame_indexes.add(frame_key)
        time_base = record["time_base"]
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(
                f"frame record {record_index} time_base must be a nonempty string"
            )
        sha256 = record["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(
                f"frame record {record_index} sha256 must be a lowercase SHA-256"
            )
        if sha256 in frame_hashes:
            raise ValueError(f"frames for {video_id} contain a duplicate sha256")
        frame_hashes.add(sha256)
        requested_time_seconds = _numeric(
            record["requested_time_seconds"],
            f"frame record {record_index} requested_time_seconds",
        )
        if not 0 <= requested_time_seconds <= duration:
            raise ValueError(
                f"frame record {record_index} requested_time_seconds is outside duration"
            )
        for field in (
            "extraction_command_identity",
            "prompt_source_identity",
            "runtime_identity",
        ):
            identity = record[field]
            if isinstance(identity, str):
                valid_identity = bool(identity.strip())
            elif isinstance(identity, (Mapping, list)):
                valid_identity = bool(identity)
            else:
                valid_identity = False
            if not valid_identity:
                raise ValueError(
                    f"frame record {record_index} {field} must be a nonempty identity"
                )
            try:
                _canonical_json(identity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"frame record {record_index} {field} must be JSON serializable"
                ) from exc
        request_sha256 = record["request_sha256"]
        if (
            not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
        ):
            raise ValueError(
                f"frame record {record_index} request_sha256 must be a lowercase SHA-256"
            )
        normalized_record = copy.deepcopy(dict(record))
        normalized_record["actual_pts_seconds"] = actual_pts_seconds
        normalized_record["requested_time_seconds"] = requested_time_seconds
        normalized.append(normalized_record)
    return normalized


def score_selector_recall(
    annotation: dict[str, Any],
    frames_by_video: dict[str, list[dict[str, Any]]],
    *,
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Score decoded-frame recall against adjudicated sufficient intervals.

    Only recorded decoded-frame PTS values can hit a required interval.  Desired
    or requested timestamps never substitute for actual decoded evidence.
    """

    if tolerance_seconds not in RECALL_TOLERANCES:
        raise ValueError(
            f"tolerance_seconds must be one of {sorted(RECALL_TOLERANCES)}"
        )
    if not isinstance(frames_by_video, Mapping):
        raise TypeError("frames_by_video must be an object")
    if not isinstance(annotation, Mapping):
        raise TypeError("annotation must be an object")
    if annotation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if annotation.get("role") != ROLE_ADJUDICATOR:
        raise ValueError("selector recall requires an adjudicator annotation")
    annotation_sha256 = annotation.get("annotation_sha256")
    if not isinstance(annotation_sha256, str):
        raise ValueError("selector recall requires a normalized annotation_sha256")
    if annotation_sha256 != _sha256(_annotation_hash_payload(annotation)):
        raise ValueError("annotation_sha256 does not match annotation payload")
    normalized_annotation_sha256 = annotation.get("normalized_annotation_sha256")
    if not isinstance(normalized_annotation_sha256, str):
        raise ValueError(
            "selector recall requires a normalized_annotation_sha256 binding"
        )
    if normalized_annotation_sha256 != _sha256(
        _normalized_annotation_hash_payload(annotation)
    ):
        raise ValueError(
            "normalized_annotation_sha256 does not match normalized annotation"
        )
    forbidden = _forbidden_key(annotation)
    if forbidden is not None:
        raise ValueError(f"forbidden key in annotation: {forbidden}")
    rows = annotation.get("rows")
    if not isinstance(rows, list):
        raise TypeError("annotation rows must be a list")

    result_rows: list[dict[str, Any]] = []
    component_count = 0
    recalled_component_count = 0
    eligible_row_count = 0
    recalled_row_count = 0
    missed_video_ids: list[str] = []
    seen_video_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"annotation row {row_index} must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"annotation row {row_index} has an invalid video_id")
        if video_id in seen_video_ids:
            raise ValueError("annotation has duplicate video IDs")
        seen_video_ids.add(video_id)
        eligibility = row.get("eligibility")
        if eligibility not in ELIGIBILITY_VALUES:
            raise ValueError(f"row {row_index} has invalid eligibility")
        if eligibility != ELIGIBILITY_LABELABLE_VISUAL:
            result_rows.append(
                {
                    "video_id": video_id,
                    "eligibility": eligibility,
                    "included_in_recall": False,
                    "recalled_sufficient_set_indexes": [],
                }
            )
            continue

        duration = _numeric(
            row.get("duration_seconds"), f"row {row_index} duration_seconds"
        )
        if duration <= 0:
            raise ValueError(f"row {row_index} duration_seconds must be positive")
        if video_id not in frames_by_video:
            raise ValueError(f"missing frame records for eligible video {video_id}")
        frame_records = _actual_pts_records(
            frames_by_video[video_id], video_id=video_id, duration=duration
        )
        sufficient_sets = row.get("sufficient_sets")
        if not isinstance(sufficient_sets, list) or not sufficient_sets:
            raise ValueError("labelable rows require nonempty sufficient sets")
        sufficient_set_hits: list[list[bool]] = []
        recalled_sufficient_set_indexes: list[int] = []
        for set_index, sufficient_set in enumerate(sufficient_sets):
            if not isinstance(sufficient_set, list) or not sufficient_set:
                raise ValueError("sufficient sets must not be empty")
            component_hits: list[bool] = []
            for interval in sufficient_set:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError("sufficient intervals must contain two bounds")
                start = _numeric(interval[0], "interval start_seconds")
                end = _numeric(interval[1], "interval end_seconds")
                lower = max(0.0, start - tolerance_seconds)
                upper = min(duration, end + tolerance_seconds)
                hit = any(
                    lower <= frame["actual_pts_seconds"] <= upper
                    for frame in frame_records
                )
                component_hits.append(hit)
                component_count += 1
                recalled_component_count += int(hit)
            sufficient_set_hits.append(component_hits)
            if all(component_hits):
                recalled_sufficient_set_indexes.append(set_index)
        row_recalled = bool(recalled_sufficient_set_indexes)
        eligible_row_count += 1
        recalled_row_count += int(row_recalled)
        if not row_recalled:
            missed_video_ids.append(video_id)
        result_rows.append(
            {
                "video_id": video_id,
                "eligibility": eligibility,
                "included_in_recall": True,
                "sufficient_set_component_hits": sufficient_set_hits,
                "recalled_sufficient_set_indexes": recalled_sufficient_set_indexes,
                "row_recalled": row_recalled,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_sha256": annotation["annotation_sha256"],
        "tolerance_seconds": tolerance_seconds,
        "rows": result_rows,
        "eligible_row_count": eligible_row_count,
        "recalled_row_count": recalled_row_count,
        "component_count": component_count,
        "recalled_component_count": recalled_component_count,
        "component_recall": (
            recalled_component_count / component_count if component_count else None
        ),
        "row_recall": recalled_row_count / eligible_row_count
        if eligible_row_count
        else None,
        "missed_video_ids": missed_video_ids,
    }
