"""Strict model request and response seams for blind frame-state decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from trec_video_2026.vqa_count_frame_presence import TARGET_STATES


def _state_prompt(
    *, target: str, frame_ids: Sequence[str], subject: str = "supplied image"
) -> str:
    ids = ", ".join(frame_ids)
    return (
        f"For each {subject}, decide whether {target} is visibly present. "
        "Use only present, absent, uncertain, or unobservable. "
        "Return only a JSON array in the supplied frame order. Each object must "
        'contain exactly {"frame_id":"...","target_state":"..."}. '
        f"Required frame IDs: {ids}. Do not return ranges or explanations."
    )


def build_individual_state_request(
    *,
    target: str,
    frame_ids: Sequence[str],
    frame_paths: Sequence[Path],
    timestamps_seconds: Sequence[float],
) -> dict[str, object]:
    """Build one label-free request over exact individual frames."""

    ids = tuple(frame_ids)
    paths = tuple(frame_paths)
    timestamps = tuple(timestamps_seconds)
    if not ids or len(ids) != len(paths) or len(ids) != len(timestamps):
        raise ValueError("frame IDs, paths, and timestamps must be nonempty and equal")
    if len(set(ids)) != len(ids):
        raise ValueError("frame IDs must be unique")
    return {
        "question": _state_prompt(target=target, frame_ids=ids),
        "frame_paths": [str(path.resolve()) for path in paths],
        "frame_timestamps": list(timestamps),
        "frame_descriptions": [
            f"Exact frame {frame_id} at {timestamp:.3f} seconds:"
            for frame_id, timestamp in zip(ids, timestamps, strict=True)
        ],
        "transcript": None,
    }


def build_contact_sheet_state_request(
    *,
    target: str,
    sheets: Sequence[tuple[Path, Sequence[str], Sequence[float]]],
) -> dict[str, object]:
    """Build one label-free request over labeled contact-sheet frames."""

    if not sheets:
        raise ValueError("contact sheets must not be empty")
    frame_ids: list[str] = []
    descriptions: list[str] = []
    paths: list[str] = []
    timestamps: list[float] = []
    for number, (path, sheet_ids, sheet_timestamps) in enumerate(sheets, start=1):
        ids = tuple(sheet_ids)
        times = tuple(sheet_timestamps)
        if not ids or len(ids) != len(times):
            raise ValueError("each contact sheet must bind equal nonempty IDs and PTS")
        frame_ids.extend(ids)
        paths.append(str(path.resolve()))
        timestamps.append(times[0])
        bindings = "; ".join(
            f"{frame_id} at {timestamp:.3f} seconds"
            for frame_id, timestamp in zip(ids, times, strict=True)
        )
        descriptions.append(
            f"Contact sheet {number} contains labeled frames in reading order: "
            f"{bindings}."
        )
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("contact-sheet frame IDs must be unique")
    return {
        "question": _state_prompt(
            target=target,
            frame_ids=frame_ids,
            subject="labeled frame inside the supplied contact sheets",
        ),
        "frame_paths": paths,
        "frame_timestamps": timestamps,
        "frame_descriptions": descriptions,
        "transcript": None,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_state_batch_result(
    result: Mapping[str, object], *, expected_frame_ids: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Parse an exact, unfenced model JSON response for one frame batch."""

    if result.get("status") != "passed":
        raise ValueError("model batch did not pass")
    answer = result.get("answer")
    if not isinstance(answer, str):
        raise TypeError("model answer must be a string")
    payload = json.loads(answer, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(payload, list):
        raise TypeError("model answer must be a JSON array")
    parsed: list[tuple[str, str]] = []
    for position, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != {
            "frame_id",
            "target_state",
        }:
            raise ValueError(f"state row {position} has invalid fields")
        frame_id = item["frame_id"]
        target_state = item["target_state"]
        if not isinstance(frame_id, str) or not isinstance(target_state, str):
            raise TypeError("frame_id and target_state must be strings")
        if target_state not in TARGET_STATES:
            raise ValueError(f"unsupported target_state: {target_state}")
        parsed.append((frame_id, target_state))
    expected = tuple(expected_frame_ids)
    if tuple(frame_id for frame_id, _ in parsed) != expected:
        raise ValueError("model states must match exact request order")
    return tuple(parsed)


def aggregate_state_batches(
    *,
    expected_frame_ids: Sequence[str],
    batches: Sequence[Sequence[tuple[str, str]]],
) -> tuple[tuple[str, str], ...]:
    """Join strict disjoint batches in the original complete request order."""

    aggregated = tuple(item for batch in batches for item in batch)
    if tuple(frame_id for frame_id, _ in aggregated) != tuple(expected_frame_ids):
        raise ValueError("aggregated model states must match exact request order")
    return aggregated
