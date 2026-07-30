"""Pure-Python contracts for temporal required-moment evidence diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
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
REQUIRED_FRAME_PROVENANCE_FIELDS = (
    "actual_pts_seconds",
    "frame_index",
    "time_base",
    "sha256",
    "requested_time_seconds",
    "extraction_command_identity",
    "prompt_source_identity",
    "request_sha256",
    "runtime_identity",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
                    token.casefold()
                    for token in _KEY_BOUNDARY.split(key)
                    if token
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
        if key != "annotation_sha256"
    }
    rows = hashed.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                # This is copied from the frozen panel during normalization; it
                # is not reviewer-authored annotation content.
                row.pop("duration_seconds", None)
    return hashed


def _normalize_interval(
    interval: Any, *, duration: float, row_index: int, set_index: int, interval_index: int
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
    if not math.isclose(start * 10, round(start * 10), abs_tol=1e-9) or not math.isclose(
        end * 10, round(end * 10), abs_tol=1e-9
    ):
        raise ValueError("interval bounds must be recorded to 0.1 seconds")
    return [round(start, 1), round(end, 1)]


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
        normalized["duration_seconds"] = duration
        eligibility = row.get("eligibility")
        if eligibility not in ELIGIBILITY_VALUES:
            raise ValueError(f"row {row_index} has invalid eligibility")
        if eligibility != ELIGIBILITY_LABELABLE_VISUAL:
            if "sufficient_sets" in row or "preferred_set_index" in row:
                raise ValueError("ineligible rows must not contain sufficient intervals")
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
    return normalized_payload


def _merged_intervals(intervals: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
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
            ordered_right = sorted((float(start), float(end)) for start, end in right_set)
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
        "agreement_passes": structural_threshold_passes
        and median_temporal_iou_passes,
    }


def _actual_pts_records(
    records: Any, *, video_id: str, duration: float
) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError(f"frames for {video_id} must be a list")
    normalized: list[dict[str, Any]] = []
    frame_indexes: set[int] = set()
    frame_hashes: set[str] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"frame record {record_index} for {video_id} must be an object")
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
            raise ValueError(f"frame record {record_index} actual_pts_seconds is outside duration")
        frame_index = record["frame_index"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"frame record {record_index} frame_index must be a nonnegative integer")
        if frame_index in frame_indexes:
            raise ValueError(f"frames for {video_id} contain a duplicate frame_index")
        frame_indexes.add(frame_index)
        time_base = record["time_base"]
        if not isinstance(time_base, str) or not time_base.strip():
            raise ValueError(f"frame record {record_index} time_base must be a nonempty string")
        sha256 = record["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"frame record {record_index} sha256 must be a lowercase SHA-256")
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
            or any(
                character not in "0123456789abcdef"
                for character in request_sha256
            )
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
            "tolerance_seconds must be one of "
            f"{sorted(RECALL_TOLERANCES)}"
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

        duration = _numeric(row.get("duration_seconds"), f"row {row_index} duration_seconds")
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
        "row_recall": recalled_row_count / eligible_row_count if eligible_row_count else None,
        "missed_video_ids": missed_video_ids,
    }
