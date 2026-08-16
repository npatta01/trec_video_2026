"""Label-blind structured frame decisions for visible-episode counting."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from trec_video_2026.vqa_count_frame_presence import (
    EPISODE_RULE,
    TARGET_STATES,
    FrameObservation,
    FramePresenceLedger,
)
from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256

CUT_POLICIES = frozenset({"continuous_across_cuts", "split_at_cuts"})
PRESENTATION_MODES = frozenset(
    {"numbered_contact_sheet", "individual_frame", "centered_triplet"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VisibleEpisodeCountSpec:
    spec_id: str
    target: str
    unit: str
    scope: str
    episode_rule: str
    maximum_sample_gap_seconds: float
    every_decoded_frame: bool
    cut_policy: str
    occluded_identity: str
    complete_sha256: str


@dataclass(frozen=True)
class UnsupportedCountSpec:
    spec_id: str
    unit: str
    outcome: str
    reason_codes: tuple[str, ...]
    complete_sha256: str


@dataclass(frozen=True)
class RequestFrame:
    frame_id: str
    decoded_frame_index: int
    timestamp_seconds: float
    artifact_sha256: str
    is_first_decoded_frame: bool
    is_last_decoded_frame: bool


@dataclass(frozen=True)
class PresentationBinding:
    presentation_id: str
    presentation_sha256: str
    frame_ids: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class FrameDecisionRequest:
    spec: VisibleEpisodeCountSpec
    video_id: str
    source_sha256: str
    duration_seconds: float
    decoded_frame_count: int
    frames: tuple[RequestFrame, ...]
    presentations: tuple[PresentationBinding, ...]
    maximum_observed_gap_seconds: float
    complete_sha256: str


@dataclass(frozen=True)
class FrameDecision:
    frame_id: str
    target_state: str


@dataclass(frozen=True)
class FrameDecisionSet:
    pass_id: str
    request_complete_sha256: str
    presentation_mode: str
    decisions: tuple[FrameDecision, ...]
    complete_sha256: str


def _strict_fields(
    payload: Mapping[str, object], *, expected: set[str], name: str
) -> None:
    unknown = set(payload).difference(expected)
    missing = expected.difference(payload)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a nonempty string")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} is out of range")
    return result


def _sha256(value: object, *, field: str) -> str:
    result = _nonempty_string(value, field=field)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def parse_visible_episode_count_spec(
    payload: Mapping[str, object],
) -> VisibleEpisodeCountSpec | UnsupportedCountSpec:
    """Parse the only count unit supported by the blind frame path."""

    validate_label_blind(payload)
    expected_fields = {
        "schema_version",
        "spec_id",
        "target",
        "unit",
        "scope",
        "episode_rule",
        "maximum_sample_gap_seconds",
        "every_decoded_frame",
        "cut_policy",
        "occluded_identity",
    }
    _strict_fields(payload, expected=expected_fields, name="count spec")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    for field in ("spec_id", "target", "unit"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise TypeError(f"{field} must be a nonempty string")
    maximum_gap = payload["maximum_sample_gap_seconds"]
    if (
        isinstance(maximum_gap, bool)
        or not isinstance(maximum_gap, int | float)
        or not math.isfinite(maximum_gap)
        or maximum_gap <= 0
    ):
        raise ValueError("maximum_sample_gap_seconds must be positive and finite")
    if not isinstance(payload["every_decoded_frame"], bool):
        raise TypeError("every_decoded_frame must be a boolean")
    if payload["cut_policy"] not in CUT_POLICIES:
        raise ValueError("unsupported cut_policy")
    if payload["occluded_identity"] != "unsupported":
        raise ValueError("occluded_identity must be unsupported")
    unsigned = dict(payload)
    complete_sha256 = canonical_sha256(unsigned)
    unit = str(payload["unit"])
    if unit != "visible_appearance_episode":
        return UnsupportedCountSpec(
            spec_id=str(payload["spec_id"]),
            unit=unit,
            outcome="unsupported",
            reason_codes=("unsupported_count_unit",),
            complete_sha256=complete_sha256,
        )
    if payload["scope"] != "full_video":
        raise ValueError("scope must be full_video")
    if payload["episode_rule"] != EPISODE_RULE:
        raise ValueError("unsupported episode_rule")
    return VisibleEpisodeCountSpec(
        spec_id=str(payload["spec_id"]),
        target=str(payload["target"]),
        unit=unit,
        scope=str(payload["scope"]),
        episode_rule=str(payload["episode_rule"]),
        maximum_sample_gap_seconds=float(maximum_gap),
        every_decoded_frame=bool(payload["every_decoded_frame"]),
        cut_policy=str(payload["cut_policy"]),
        occluded_identity=str(payload["occluded_identity"]),
        complete_sha256=complete_sha256,
    )


def build_frame_decision_request(
    spec: VisibleEpisodeCountSpec,
    inventory: Mapping[str, object],
    presentations: Sequence[Mapping[str, object]],
) -> FrameDecisionRequest:
    """Bind a visible-episode spec to decoded frames and presentations."""

    validate_label_blind(inventory)
    validate_label_blind(presentations)
    _strict_fields(
        inventory,
        expected={
            "schema_version",
            "video_id",
            "source_sha256",
            "duration_seconds",
            "decoded_frame_count",
            "frames",
        },
        name="frame inventory",
    )
    if inventory["schema_version"] != 1:
        raise ValueError("unsupported frame inventory schema_version")
    video_id = _nonempty_string(inventory["video_id"], field="video_id")
    source_sha256 = _sha256(inventory["source_sha256"], field="source_sha256")
    duration_seconds = _number(inventory["duration_seconds"], field="duration_seconds")
    decoded_frame_count = _integer(
        inventory["decoded_frame_count"], field="decoded_frame_count", minimum=1
    )
    raw_frames = inventory["frames"]
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, str | bytes):
        raise TypeError("frames must be an array")
    if not raw_frames:
        raise ValueError("frames must not be empty")
    frame_fields = {
        "frame_id",
        "decoded_frame_index",
        "timestamp_seconds",
        "artifact_sha256",
        "is_first_decoded_frame",
        "is_last_decoded_frame",
    }
    parsed_frames: list[RequestFrame] = []
    seen_frame_ids: set[str] = set()
    previous_index = -1
    previous_timestamp = -math.inf
    for position, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise TypeError(f"frames[{position}] must be an object")
        _strict_fields(raw_frame, expected=frame_fields, name=f"frames[{position}]")
        frame_id = _nonempty_string(
            raw_frame["frame_id"], field=f"frames[{position}].frame_id"
        )
        if frame_id in seen_frame_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        seen_frame_ids.add(frame_id)
        decoded_index = _integer(
            raw_frame["decoded_frame_index"],
            field=f"frames[{position}].decoded_frame_index",
        )
        if decoded_index <= previous_index:
            raise ValueError("decoded frame indices must be strictly increasing")
        previous_index = decoded_index
        timestamp = _number(
            raw_frame["timestamp_seconds"],
            field=f"frames[{position}].timestamp_seconds",
        )
        if timestamp <= previous_timestamp:
            raise ValueError("frame timestamps must be strictly increasing")
        previous_timestamp = timestamp
        first = raw_frame["is_first_decoded_frame"]
        last = raw_frame["is_last_decoded_frame"]
        if not isinstance(first, bool) or not isinstance(last, bool):
            raise TypeError("decoded-frame endpoint flags must be booleans")
        parsed_frames.append(
            RequestFrame(
                frame_id=frame_id,
                decoded_frame_index=decoded_index,
                timestamp_seconds=timestamp,
                artifact_sha256=_sha256(
                    raw_frame["artifact_sha256"],
                    field=f"frames[{position}].artifact_sha256",
                ),
                is_first_decoded_frame=first,
                is_last_decoded_frame=last,
            )
        )
    frames = tuple(parsed_frames)
    if (
        frames[0].decoded_frame_index != 0
        or not frames[0].is_first_decoded_frame
        or not math.isclose(frames[0].timestamp_seconds, 0.0, abs_tol=1e-9)
    ):
        raise ValueError("first decoded frame is not covered")
    if (
        frames[-1].decoded_frame_index != decoded_frame_count - 1
        or not frames[-1].is_last_decoded_frame
        or not math.isclose(
            frames[-1].timestamp_seconds, duration_seconds, abs_tol=1e-9
        )
    ):
        raise ValueError("last decoded frame is not covered")
    if any(frame.is_first_decoded_frame for frame in frames[1:]) or any(
        frame.is_last_decoded_frame for frame in frames[:-1]
    ):
        raise ValueError("decoded-frame endpoint marker is on the wrong frame")
    if spec.every_decoded_frame and [
        frame.decoded_frame_index for frame in frames
    ] != list(range(decoded_frame_count)):
        raise ValueError("every decoded frame is required")
    maximum_gap = max(
        (
            current.timestamp_seconds - previous.timestamp_seconds
            for previous, current in zip(frames, frames[1:], strict=False)
        ),
        default=0.0,
    )
    if maximum_gap > spec.maximum_sample_gap_seconds + 1e-9:
        raise ValueError("maximum observed frame gap exceeds CountSpec")

    presentation_fields = {
        "presentation_id",
        "presentation_sha256",
        "frame_ids",
        "mode",
    }
    parsed_bindings: list[PresentationBinding] = []
    seen_presentation_ids: set[str] = set()
    frame_ids_by_mode: dict[str, list[str]] = {}
    for position, presentation in enumerate(presentations):
        if not isinstance(presentation, Mapping):
            raise TypeError(f"presentations[{position}] must be an object")
        _strict_fields(
            presentation,
            expected=presentation_fields,
            name=f"presentations[{position}]",
        )
        presentation_id = _nonempty_string(
            presentation["presentation_id"],
            field=f"presentations[{position}].presentation_id",
        )
        if presentation_id in seen_presentation_ids:
            raise ValueError(f"duplicate presentation_id: {presentation_id}")
        seen_presentation_ids.add(presentation_id)
        mode = _nonempty_string(
            presentation["mode"], field=f"presentations[{position}].mode"
        )
        if mode not in PRESENTATION_MODES:
            raise ValueError(f"unsupported presentation mode: {mode}")
        raw_frame_ids = presentation["frame_ids"]
        if not isinstance(raw_frame_ids, Sequence) or isinstance(
            raw_frame_ids, str | bytes
        ):
            raise TypeError(f"presentations[{position}].frame_ids must be an array")
        frame_ids = tuple(
            _nonempty_string(
                frame_id,
                field=f"presentations[{position}].frame_ids",
            )
            for frame_id in raw_frame_ids
        )
        if not frame_ids:
            raise ValueError("presentation frame_ids must not be empty")
        frame_ids_by_mode.setdefault(mode, []).extend(frame_ids)
        parsed_bindings.append(
            PresentationBinding(
                presentation_id=presentation_id,
                presentation_sha256=_sha256(
                    presentation["presentation_sha256"],
                    field=f"presentations[{position}].presentation_sha256",
                ),
                frame_ids=frame_ids,
                mode=mode,
            )
        )
    if not parsed_bindings:
        raise ValueError("presentations must not be empty")
    required_frame_ids = [frame.frame_id for frame in frames]
    for mode, presented_frame_ids in frame_ids_by_mode.items():
        if presented_frame_ids != required_frame_ids:
            raise ValueError(
                f"presentation mode {mode} must cover each required frame exactly once"
            )
    bindings = tuple(parsed_bindings)
    unsigned = {
        "schema_version": 1,
        "spec": asdict(spec),
        "video_id": video_id,
        "source_sha256": source_sha256,
        "duration_seconds": duration_seconds,
        "decoded_frame_count": decoded_frame_count,
        "frames": [asdict(frame) for frame in frames],
        "presentations": [asdict(binding) for binding in bindings],
        "maximum_observed_gap_seconds": maximum_gap,
    }
    return FrameDecisionRequest(
        spec=spec,
        video_id=video_id,
        source_sha256=source_sha256,
        duration_seconds=duration_seconds,
        decoded_frame_count=decoded_frame_count,
        frames=frames,
        presentations=bindings,
        maximum_observed_gap_seconds=maximum_gap,
        complete_sha256=canonical_sha256(unsigned),
    )


def parse_frame_decisions(
    request: FrameDecisionRequest,
    payload: Mapping[str, object],
    *,
    required_frame_ids: Sequence[str] | None = None,
) -> FrameDecisionSet:
    """Parse one model pass over an authenticated frame request."""

    validate_label_blind(payload)
    _strict_fields(
        payload,
        expected={
            "schema_version",
            "pass_id",
            "request_complete_sha256",
            "presentation_mode",
            "decisions",
        },
        name="frame decisions",
    )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported frame decisions schema_version")
    pass_id = _nonempty_string(payload["pass_id"], field="pass_id")
    if pass_id not in {"primary", "challenge"}:
        raise ValueError("pass_id must be primary or challenge")
    request_sha256 = _sha256(
        payload["request_complete_sha256"], field="request_complete_sha256"
    )
    if request_sha256 != request.complete_sha256:
        raise ValueError("frame decisions request binding changed")
    presentation_mode = _nonempty_string(
        payload["presentation_mode"], field="presentation_mode"
    )
    if presentation_mode not in {binding.mode for binding in request.presentations}:
        raise ValueError("presentation_mode is not bound by the request")
    request_frame_ids = tuple(frame.frame_id for frame in request.frames)
    if required_frame_ids is None:
        expected_frame_ids = request_frame_ids
    else:
        expected_frame_ids = tuple(required_frame_ids)
        if not expected_frame_ids:
            raise ValueError("required_frame_ids must not be empty")
        request_positions = {
            frame_id: position for position, frame_id in enumerate(request_frame_ids)
        }
        if any(frame_id not in request_positions for frame_id in expected_frame_ids):
            raise ValueError("required_frame_ids contains an unknown request frame")
        if (
            len(set(expected_frame_ids)) != len(expected_frame_ids)
            or tuple(sorted(expected_frame_ids, key=request_positions.__getitem__))
            != expected_frame_ids
        ):
            raise ValueError("required_frame_ids must be unique and request ordered")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, Sequence) or isinstance(
        raw_decisions, str | bytes
    ):
        raise TypeError("decisions must be an array")
    parsed_decisions: list[FrameDecision] = []
    for position, raw_decision in enumerate(raw_decisions):
        if not isinstance(raw_decision, Mapping):
            raise TypeError(f"decisions[{position}] must be an object")
        _strict_fields(
            raw_decision,
            expected={"frame_id", "target_state"},
            name=f"decisions[{position}]",
        )
        frame_id = _nonempty_string(
            raw_decision["frame_id"], field=f"decisions[{position}].frame_id"
        )
        target_state = _nonempty_string(
            raw_decision["target_state"],
            field=f"decisions[{position}].target_state",
        )
        if target_state not in TARGET_STATES:
            raise ValueError(f"unsupported target_state: {target_state}")
        parsed_decisions.append(FrameDecision(frame_id, target_state))
    decisions = tuple(parsed_decisions)
    if tuple(decision.frame_id for decision in decisions) != expected_frame_ids:
        raise ValueError("decisions must cover every required frame exactly once")
    unsigned = {
        "schema_version": 1,
        "pass_id": pass_id,
        "request_complete_sha256": request_sha256,
        "presentation_mode": presentation_mode,
        "decisions": [asdict(decision) for decision in decisions],
    }
    return FrameDecisionSet(
        pass_id=pass_id,
        request_complete_sha256=request_sha256,
        presentation_mode=presentation_mode,
        decisions=decisions,
        complete_sha256=canonical_sha256(unsigned),
    )


def select_challenge_frames(
    request: FrameDecisionRequest, decisions: FrameDecisionSet
) -> tuple[str, ...]:
    """Select source frames that a different presentation must challenge."""

    if decisions.request_complete_sha256 != request.complete_sha256:
        raise ValueError("frame decisions request binding changed")
    if len(request.frames) <= 100:
        return tuple(frame.frame_id for frame in request.frames)

    states = [decision.target_state for decision in decisions.decisions]
    selected = {0, len(states) - 1}
    for index in range(1, len(states)):
        if states[index] != states[index - 1]:
            selected.update(
                candidate
                for candidate in (index - 2, index - 1, index, index + 1)
                if 0 <= candidate < len(states)
            )

    run_start = 0
    for index in range(1, len(states) + 1):
        if index < len(states) and states[index] == states[run_start]:
            continue
        run_end = index - 1
        interior = list(range(run_start + 1, run_end))
        if len(interior) <= 3:
            selected.update(interior)
        else:
            run_length = run_end - run_start + 1
            selected.update(
                run_start + (run_length * quartile) // 4 for quartile in (1, 2, 3)
            )
        run_start = index

    return tuple(request.frames[index].frame_id for index in sorted(selected))


def reconcile_frame_decisions(
    request: FrameDecisionRequest,
    primary: FrameDecisionSet,
    challenge: FrameDecisionSet,
) -> FramePresenceLedger:
    """Reconcile two differently presented model passes conservatively."""

    if primary.pass_id != "primary" or challenge.pass_id != "challenge":
        raise ValueError("primary and challenge pass IDs are required")
    if (
        primary.request_complete_sha256 != request.complete_sha256
        or challenge.request_complete_sha256 != request.complete_sha256
    ):
        raise ValueError("decision request binding changed")
    if primary.presentation_mode == challenge.presentation_mode:
        raise ValueError("primary and challenge presentations must differ")
    required_challenge_ids = select_challenge_frames(request, primary)
    if tuple(decision.frame_id for decision in challenge.decisions) != (
        required_challenge_ids
    ):
        raise ValueError("challenge decisions do not match the selected frames")

    challenge_by_id = {
        decision.frame_id: decision.target_state for decision in challenge.decisions
    }
    primary_by_id = {
        decision.frame_id: decision.target_state for decision in primary.decisions
    }
    observations = tuple(
        FrameObservation(
            frame_id=frame.frame_id,
            decoded_frame_index=frame.decoded_frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            artifact_sha256=frame.artifact_sha256,
            target_state=(
                "uncertain"
                if request.spec.cut_policy == "split_at_cuts"
                else (
                    primary_by_id[frame.frame_id]
                    if frame.frame_id not in challenge_by_id
                    or primary_by_id[frame.frame_id] == challenge_by_id[frame.frame_id]
                    else "uncertain"
                )
            ),
            is_first_decoded_frame=frame.is_first_decoded_frame,
            is_last_decoded_frame=frame.is_last_decoded_frame,
        )
        for frame in request.frames
    )
    return FramePresenceLedger(
        video_id=request.video_id,
        source_sha256=request.source_sha256,
        target=request.spec.target,
        episode_rule=request.spec.episode_rule,
        duration_seconds=request.duration_seconds,
        decoded_frame_count=request.decoded_frame_count,
        maximum_sample_gap_seconds=request.spec.maximum_sample_gap_seconds,
        frames=observations,
    )
