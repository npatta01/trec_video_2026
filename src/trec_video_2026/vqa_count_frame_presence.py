"""Mechanically covered frame-presence episode counting."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TARGET_STATES = frozenset({"present", "absent", "uncertain", "unobservable"})
EPISODE_RULE = "maximal contiguous run of present samples"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FrameObservation:
    frame_id: str
    decoded_frame_index: int
    timestamp_seconds: float
    artifact_sha256: str
    target_state: str
    is_first_decoded_frame: bool
    is_last_decoded_frame: bool


@dataclass(frozen=True)
class FramePresenceLedger:
    video_id: str
    source_sha256: str
    target: str
    episode_rule: str
    duration_seconds: float
    decoded_frame_count: int
    maximum_sample_gap_seconds: float
    frames: tuple[FrameObservation, ...]


@dataclass(frozen=True)
class PresenceEpisode:
    start_frame_id: str
    end_frame_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PresenceCountResult:
    outcome: str
    value: int | None
    episodes: tuple[PresenceEpisode, ...]
    reason_codes: tuple[str, ...]


def _strict_fields(
    value: Mapping[str, object], *, expected: set[str], name: str
) -> None:
    unknown = set(value).difference(expected)
    missing = expected.difference(value)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a nonempty string")
    return value


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} is out of range")
    return result


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} is out of range")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _sha256(value: object, *, field: str) -> str:
    result = _string(value, field=field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def parse_frame_presence_ledger(payload: Mapping[str, object]) -> FramePresenceLedger:
    """Parse a strict, label-free ledger of reviewed frame observations."""
    fields = {
        "schema_version",
        "video_id",
        "source_sha256",
        "target",
        "episode_rule",
        "duration_seconds",
        "decoded_frame_count",
        "maximum_sample_gap_seconds",
        "frames",
    }
    _strict_fields(payload, expected=fields, name="frame-presence ledger")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported frame-presence ledger schema_version")
    duration = _number(payload["duration_seconds"], field="duration_seconds")
    decoded_frame_count = _integer(
        payload["decoded_frame_count"], field="decoded_frame_count", minimum=1
    )
    maximum_gap = _number(
        payload["maximum_sample_gap_seconds"],
        field="maximum_sample_gap_seconds",
        minimum=1e-12,
    )
    raw_frames = payload["frames"]
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, str | bytes):
        raise TypeError("frames must be an array")
    if not raw_frames:
        raise ValueError("frames must not be empty")
    observations: list[FrameObservation] = []
    seen_ids: set[str] = set()
    previous_timestamp = -math.inf
    previous_decoded_frame_index = -1
    frame_fields = {
        "frame_id",
        "decoded_frame_index",
        "timestamp_seconds",
        "artifact_sha256",
        "target_state",
        "is_first_decoded_frame",
        "is_last_decoded_frame",
    }
    for index, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise TypeError(f"frames[{index}] must be an object")
        _strict_fields(raw, expected=frame_fields, name=f"frames[{index}]")
        frame_id = _string(raw["frame_id"], field=f"frames[{index}].frame_id")
        if frame_id in seen_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)
        decoded_frame_index = _integer(
            raw["decoded_frame_index"],
            field=f"frames[{index}].decoded_frame_index",
        )
        if decoded_frame_index >= decoded_frame_count:
            raise ValueError(f"frames[{index}].decoded_frame_index exceeds inventory")
        if decoded_frame_index <= previous_decoded_frame_index:
            raise ValueError("decoded frame indices must be strictly increasing")
        previous_decoded_frame_index = decoded_frame_index
        timestamp = _number(
            raw["timestamp_seconds"], field=f"frames[{index}].timestamp_seconds"
        )
        if timestamp > duration + 1e-9:
            raise ValueError(f"frames[{index}].timestamp_seconds exceeds duration")
        if timestamp <= previous_timestamp:
            raise ValueError("frame timestamps must be strictly increasing")
        previous_timestamp = timestamp
        state = _string(raw["target_state"], field=f"frames[{index}].target_state")
        if state not in TARGET_STATES:
            raise ValueError(f"unsupported target_state: {state}")
        observations.append(
            FrameObservation(
                frame_id=frame_id,
                decoded_frame_index=decoded_frame_index,
                timestamp_seconds=timestamp,
                artifact_sha256=_sha256(
                    raw["artifact_sha256"],
                    field=f"frames[{index}].artifact_sha256",
                ),
                target_state=state,
                is_first_decoded_frame=_boolean(
                    raw["is_first_decoded_frame"],
                    field=f"frames[{index}].is_first_decoded_frame",
                ),
                is_last_decoded_frame=_boolean(
                    raw["is_last_decoded_frame"],
                    field=f"frames[{index}].is_last_decoded_frame",
                ),
            )
        )
    episode_rule = _string(payload["episode_rule"], field="episode_rule")
    if episode_rule != EPISODE_RULE:
        raise ValueError("unsupported episode_rule")
    if any(frame.is_first_decoded_frame for frame in observations[1:]) or any(
        frame.is_last_decoded_frame for frame in observations[:-1]
    ):
        raise ValueError("decoded-frame endpoint marker is on the wrong frame")
    return FramePresenceLedger(
        video_id=_string(payload["video_id"], field="video_id"),
        source_sha256=_sha256(payload["source_sha256"], field="source_sha256"),
        target=_string(payload["target"], field="target"),
        episode_rule=episode_rule,
        duration_seconds=duration,
        decoded_frame_count=decoded_frame_count,
        maximum_sample_gap_seconds=maximum_gap,
        frames=tuple(observations),
    )


def reduce_presence_episodes(ledger: FramePresenceLedger) -> PresenceCountResult:
    """Count maximal contiguous runs only when frame coverage is sufficient."""
    reasons: list[str] = []
    if not ledger.frames[0].is_first_decoded_frame:
        reasons.append("missing_first_decoded_frame")
    if not ledger.frames[-1].is_last_decoded_frame:
        reasons.append("missing_last_decoded_frame")
    if ledger.frames[0].decoded_frame_index != 0:
        reasons.append("first_decoded_frame_index_not_covered")
    if ledger.frames[-1].decoded_frame_index != ledger.decoded_frame_count - 1:
        reasons.append("last_decoded_frame_index_not_covered")
    if not math.isclose(ledger.frames[0].timestamp_seconds, 0.0, abs_tol=1e-9):
        reasons.append("scope_start_not_covered")
    if not math.isclose(
        ledger.frames[-1].timestamp_seconds,
        ledger.duration_seconds,
        abs_tol=1e-9,
    ):
        reasons.append("scope_end_not_covered")
    if any(
        current.timestamp_seconds - previous.timestamp_seconds
        > ledger.maximum_sample_gap_seconds + 1e-9
        for previous, current in zip(ledger.frames, ledger.frames[1:], strict=False)
    ):
        reasons.append("sample_gap_exceeds_maximum")
    if any(
        observation.target_state not in {"present", "absent"}
        for observation in ledger.frames
    ):
        reasons.append("non_binary_frame_decision")
    if reasons:
        return PresenceCountResult("abstain", None, (), tuple(reasons))

    episodes: list[PresenceEpisode] = []
    start: FrameObservation | None = None
    end: FrameObservation | None = None
    for observation in ledger.frames:
        if observation.target_state == "present":
            start = observation if start is None else start
            end = observation
        elif start is not None and end is not None:
            episodes.append(
                PresenceEpisode(
                    start.frame_id,
                    end.frame_id,
                    start.timestamp_seconds,
                    end.timestamp_seconds,
                )
            )
            start = None
            end = None
    if start is not None and end is not None:
        episodes.append(
            PresenceEpisode(
                start.frame_id,
                end.frame_id,
                start.timestamp_seconds,
                end.timestamp_seconds,
            )
        )
    return PresenceCountResult("exact", len(episodes), tuple(episodes), ())
