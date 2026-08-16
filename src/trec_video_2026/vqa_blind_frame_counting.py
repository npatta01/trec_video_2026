"""Label-blind structured frame decisions for visible-episode counting."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from trec_video_2026.vqa_count_frame_presence import (
    EPISODE_RULE,
    TARGET_STATES,
    FrameObservation,
    FramePresenceLedger,
    PresenceCountResult,
    reduce_presence_episodes,
)
from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256

CUT_POLICIES = frozenset({"continuous_across_cuts", "split_at_cuts"})
PRESENTATION_MODES = frozenset({"numbered_contact_sheet", "individual_frame"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_IMPLEMENTATION_SOURCE_PATHS = frozenset(
    {
        "src/trec_video_2026/vqa_blind_frame_counting.py",
        "src/trec_video_2026/vqa_count_frame_presence.py",
        "src/trec_video_2026/vqa_counting_evidence.py",
        "src/trec_video_2026/vqa_submission.py",
        "src/trec_video_2026/vqa_gate.py",
        "src/trec_video_2026/vqa_experiments/temporal_bottleneck.py",
        "scripts/prepare_blind_frame_counting.py",
        "pyproject.toml",
        "uv.lock",
    }
)


@dataclass(frozen=True)
class VisibleEpisodeCountSpec:
    spec_id: str
    question: str
    question_sha256: str
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
    question_sha256: str
    unit: str
    outcome: str
    reason_codes: tuple[str, ...]
    complete_sha256: str


@dataclass(frozen=True)
class DecoderBinding:
    name: str
    version: str
    configuration_sha256: str
    configuration_size_bytes: int


@dataclass(frozen=True)
class RequestFrame:
    frame_id: str
    decoded_frame_index: int
    timestamp_seconds: float
    artifact_sha256: str
    artifact_size_bytes: int
    is_first_decoded_frame: bool
    is_last_decoded_frame: bool


@dataclass(frozen=True)
class PresentationBinding:
    presentation_id: str
    presentation_sha256: str
    artifact_size_bytes: int
    frame_ids: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class FrameDecisionRequest:
    spec: VisibleEpisodeCountSpec
    video_id: str
    source_sha256: str
    source_size_bytes: int
    decoder: DecoderBinding
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
    presentation_ids: tuple[str, ...]
    presentation_bundle_sha256: str
    decisions: tuple[FrameDecision, ...]
    complete_sha256: str


@dataclass(frozen=True)
class SourceBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class FramePresenceCoverageCertificate:
    outcome: str
    value: int | None
    reason_codes: tuple[str, ...]
    spec_complete_sha256: str
    request_complete_sha256: str
    primary_complete_sha256: str
    challenge_complete_sha256: str
    ledger_sha256: str
    result_sha256: str
    interface_source_sha256: str
    reducer_source_sha256: str
    implementation_sources: tuple[SourceBinding, ...]
    source_revision: str
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
        "question",
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
    for field in ("question", "unit"):
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
    question = _nonempty_string(payload["question"], field="question")
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    spec_id = f"Q{question_sha256[:16]}"
    unit = str(payload["unit"])
    if unit != "visible_appearance_episode":
        return UnsupportedCountSpec(
            spec_id=spec_id,
            question_sha256=question_sha256,
            unit=unit,
            outcome="unsupported",
            reason_codes=("unsupported_count_unit",),
            complete_sha256=complete_sha256,
        )
    if payload["scope"] != "full_video":
        raise ValueError("scope must be full_video")
    if payload["episode_rule"] != EPISODE_RULE:
        raise ValueError("unsupported episode_rule")
    question_patterns = (
        re.compile(
            r"^How many times does (?:the )?(?P<target>.+?) appear"
            r"(?: in (?:the )?video)?\?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^How many times is (?:the )?(?P<target>.+?) visible"
            r"(?: in (?:the )?video)?\?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^How many times (?P<target>.+?) appear"
            r"(?: in (?:the )?video)?\?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^How many times (?:do )?we see (?P<target>.+?)"
            r"(?: in (?:the )?video)?\?$",
            re.IGNORECASE,
        ),
    )
    matched = next(
        (
            match
            for pattern in question_patterns
            if (match := pattern.fullmatch(question))
        ),
        None,
    )
    if matched is None:
        return UnsupportedCountSpec(
            spec_id=spec_id,
            question_sha256=question_sha256,
            unit=unit,
            outcome="unsupported",
            reason_codes=("unsupported_question_form",),
            complete_sha256=complete_sha256,
        )
    target = matched.group("target").strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9' -]{0,79}", target) is None or re.search(
        r"\b(reference|answer|expected|correct|failure|control|label|score)\b",
        target,
        re.IGNORECASE,
    ):
        return UnsupportedCountSpec(
            spec_id=spec_id,
            question_sha256=question_sha256,
            unit=unit,
            outcome="unsupported",
            reason_codes=("unsupported_target_derivation",),
            complete_sha256=complete_sha256,
        )
    return VisibleEpisodeCountSpec(
        spec_id=spec_id,
        question=question,
        question_sha256=question_sha256,
        target=target,
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
            "source_size_bytes",
            "decoder",
            "duration_seconds",
            "decoded_frame_count",
            "frames",
        },
        name="frame inventory",
    )
    if inventory["schema_version"] != 1:
        raise ValueError("unsupported frame inventory schema_version")
    source_sha256 = _sha256(inventory["source_sha256"], field="source_sha256")
    source_size_bytes = _integer(
        inventory["source_size_bytes"], field="source_size_bytes", minimum=1
    )
    raw_decoder = inventory["decoder"]
    if not isinstance(raw_decoder, Mapping):
        raise TypeError("decoder must be an object")
    _strict_fields(
        raw_decoder,
        expected={
            "name",
            "version",
            "configuration_sha256",
            "configuration_size_bytes",
        },
        name="decoder",
    )
    decoder = DecoderBinding(
        name=_nonempty_string(raw_decoder["name"], field="decoder.name"),
        version=_nonempty_string(raw_decoder["version"], field="decoder.version"),
        configuration_sha256=_sha256(
            raw_decoder["configuration_sha256"],
            field="decoder.configuration_sha256",
        ),
        configuration_size_bytes=_integer(
            raw_decoder["configuration_size_bytes"],
            field="decoder.configuration_size_bytes",
            minimum=1,
        ),
    )
    video_id = _nonempty_string(inventory["video_id"], field="video_id")
    if video_id != f"V{source_sha256[:16]}":
        raise ValueError("video_id must be the opaque source-derived ID")
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
        "artifact_size_bytes",
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
        if frame_id != f"F{decoded_index}":
            raise ValueError("frame_id must be derived from decoded_frame_index")
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
                artifact_size_bytes=_integer(
                    raw_frame["artifact_size_bytes"],
                    field=f"frames[{position}].artifact_size_bytes",
                    minimum=1,
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
        "artifact_size_bytes",
        "frame_ids",
        "mode",
    }
    parsed_bindings: list[PresentationBinding] = []
    seen_presentation_ids: set[str] = set()
    frame_ids_by_mode: dict[str, list[str]] = {}
    digests_by_mode: dict[str, set[str]] = {}
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
        presentation_sha256 = _sha256(
            presentation["presentation_sha256"],
            field=f"presentations[{position}].presentation_sha256",
        )
        if presentation_id != f"P{presentation_sha256}":
            raise ValueError("presentation_id must be the opaque artifact-derived ID")
        digests_by_mode.setdefault(mode, set()).add(presentation_sha256)
        parsed_bindings.append(
            PresentationBinding(
                presentation_id=presentation_id,
                presentation_sha256=presentation_sha256,
                artifact_size_bytes=_integer(
                    presentation["artifact_size_bytes"],
                    field=f"presentations[{position}].artifact_size_bytes",
                    minimum=1,
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
    modes = tuple(digests_by_mode)
    for index, mode in enumerate(modes):
        for other_mode in modes[index + 1 :]:
            if digests_by_mode[mode].intersection(digests_by_mode[other_mode]):
                raise ValueError(
                    "different presentation modes must use different bytes"
                )
    bindings = tuple(parsed_bindings)
    unsigned = {
        "schema_version": 1,
        "spec": asdict(spec),
        "video_id": video_id,
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
        "decoder": asdict(decoder),
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
        source_size_bytes=source_size_bytes,
        decoder=decoder,
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
            "presentation_ids",
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
    raw_presentation_ids = payload["presentation_ids"]
    if not isinstance(raw_presentation_ids, Sequence) or isinstance(
        raw_presentation_ids, str | bytes
    ):
        raise TypeError("presentation_ids must be an array")
    presentation_ids = tuple(
        _nonempty_string(item, field="presentation_ids")
        for item in raw_presentation_ids
    )
    if not presentation_ids or len(set(presentation_ids)) != len(presentation_ids):
        raise ValueError("presentation_ids must be nonempty and unique")
    bindings_by_id = {
        binding.presentation_id: binding for binding in request.presentations
    }
    try:
        selected_bindings = tuple(bindings_by_id[item] for item in presentation_ids)
    except KeyError as error:
        raise ValueError("presentation_ids contains an unbound presentation") from error
    if any(binding.mode != presentation_mode for binding in selected_bindings):
        raise ValueError("presentation_ids do not match presentation_mode")
    presented_frame_ids = tuple(
        frame_id for binding in selected_bindings for frame_id in binding.frame_ids
    )
    if presented_frame_ids != expected_frame_ids:
        raise ValueError("presentation_ids must cover the required frames exactly once")
    presentation_bundle_sha256 = canonical_sha256(
        [asdict(binding) for binding in selected_bindings]
    )
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
        "presentation_ids": list(presentation_ids),
        "presentation_bundle_sha256": presentation_bundle_sha256,
        "decisions": [asdict(decision) for decision in decisions],
    }
    return FrameDecisionSet(
        pass_id=pass_id,
        request_complete_sha256=request_sha256,
        presentation_mode=presentation_mode,
        presentation_ids=presentation_ids,
        presentation_bundle_sha256=presentation_bundle_sha256,
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
    if primary.presentation_mode != "numbered_contact_sheet":
        raise ValueError("primary must use a numbered contact sheet")
    if (
        primary.request_complete_sha256 != request.complete_sha256
        or challenge.request_complete_sha256 != request.complete_sha256
    ):
        raise ValueError("decision request binding changed")
    if primary.presentation_mode == challenge.presentation_mode:
        raise ValueError("primary and challenge presentations must differ")
    if challenge.presentation_mode != "individual_frame":
        raise ValueError("challenge must use individual frames")
    if primary.presentation_bundle_sha256 == challenge.presentation_bundle_sha256:
        raise ValueError("primary and challenge presentation bytes must differ")
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


def visible_episode_count_spec_to_dict(
    spec: VisibleEpisodeCountSpec,
) -> dict[str, object]:
    """Serialize a supported CountSpec in its canonical schema."""

    return {
        "schema_version": 1,
        "question": spec.question,
        "unit": spec.unit,
        "scope": spec.scope,
        "episode_rule": spec.episode_rule,
        "maximum_sample_gap_seconds": spec.maximum_sample_gap_seconds,
        "every_decoded_frame": spec.every_decoded_frame,
        "cut_policy": spec.cut_policy,
        "occluded_identity": spec.occluded_identity,
        "complete_sha256": spec.complete_sha256,
    }


def frame_decision_request_to_dict(
    request: FrameDecisionRequest,
) -> dict[str, object]:
    """Serialize an authenticated frame request."""

    return {
        "schema_version": 1,
        "spec": asdict(request.spec),
        "video_id": request.video_id,
        "source_sha256": request.source_sha256,
        "source_size_bytes": request.source_size_bytes,
        "decoder": asdict(request.decoder),
        "duration_seconds": request.duration_seconds,
        "decoded_frame_count": request.decoded_frame_count,
        "frames": [asdict(frame) for frame in request.frames],
        "presentations": [asdict(binding) for binding in request.presentations],
        "maximum_observed_gap_seconds": request.maximum_observed_gap_seconds,
        "complete_sha256": request.complete_sha256,
    }


def frame_decision_set_to_dict(decisions: FrameDecisionSet) -> dict[str, object]:
    """Serialize an authenticated model decision set."""

    return {
        "schema_version": 1,
        "pass_id": decisions.pass_id,
        "request_complete_sha256": decisions.request_complete_sha256,
        "presentation_mode": decisions.presentation_mode,
        "presentation_ids": list(decisions.presentation_ids),
        "presentation_bundle_sha256": decisions.presentation_bundle_sha256,
        "decisions": [asdict(decision) for decision in decisions.decisions],
        "complete_sha256": decisions.complete_sha256,
    }


def frame_presence_ledger_to_dict(
    ledger: FramePresenceLedger,
) -> dict[str, object]:
    """Serialize the reconciled ledger accepted by the existing reducer."""

    return {
        "schema_version": 1,
        "video_id": ledger.video_id,
        "source_sha256": ledger.source_sha256,
        "target": ledger.target,
        "episode_rule": ledger.episode_rule,
        "duration_seconds": ledger.duration_seconds,
        "decoded_frame_count": ledger.decoded_frame_count,
        "maximum_sample_gap_seconds": ledger.maximum_sample_gap_seconds,
        "frames": [asdict(frame) for frame in ledger.frames],
    }


def presence_count_result_to_dict(result: PresenceCountResult) -> dict[str, object]:
    """Serialize a deterministic reducer result."""

    return {
        "schema_version": 1,
        "outcome": result.outcome,
        "value": result.value,
        "episodes": [asdict(episode) for episode in result.episodes],
        "reason_codes": list(result.reason_codes),
    }


def frame_presence_certificate_to_dict(
    certificate: FramePresenceCoverageCertificate,
) -> dict[str, object]:
    """Serialize a coverage certificate."""

    payload = asdict(certificate)
    payload["schema_version"] = 1
    return {"schema_version": payload.pop("schema_version"), **payload}


def _verify_self_hash(payload: Mapping[str, object], *, name: str) -> str:
    embedded = payload.get("complete_sha256")
    if not isinstance(embedded, str):
        raise TypeError(f"{name} lacks complete_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "complete_sha256"
    }
    if canonical_sha256(unsigned) != embedded:
        raise ValueError(f"{name} complete_sha256 changed")
    return embedded


def _decision_set_input(decisions: FrameDecisionSet) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pass_id": decisions.pass_id,
        "request_complete_sha256": decisions.request_complete_sha256,
        "presentation_mode": decisions.presentation_mode,
        "presentation_ids": list(decisions.presentation_ids),
        "decisions": [asdict(decision) for decision in decisions.decisions],
    }


def _revalidate_chain_structure(
    spec: VisibleEpisodeCountSpec,
    request: FrameDecisionRequest,
    primary: FrameDecisionSet,
    challenge: FrameDecisionSet,
) -> None:
    raw_spec = visible_episode_count_spec_to_dict(spec)
    raw_spec.pop("complete_sha256")
    parsed_spec = parse_visible_episode_count_spec(raw_spec)
    if parsed_spec != spec:
        raise ValueError("CountSpec structure changed")
    inventory = {
        "schema_version": 1,
        "video_id": request.video_id,
        "source_sha256": request.source_sha256,
        "source_size_bytes": request.source_size_bytes,
        "decoder": asdict(request.decoder),
        "duration_seconds": request.duration_seconds,
        "decoded_frame_count": request.decoded_frame_count,
        "frames": [asdict(frame) for frame in request.frames],
    }
    presentations = [asdict(binding) for binding in request.presentations]
    rebuilt_request = build_frame_decision_request(spec, inventory, presentations)
    if rebuilt_request != request:
        raise ValueError("frame request structure changed")
    rebuilt_primary = parse_frame_decisions(request, _decision_set_input(primary))
    if rebuilt_primary != primary:
        raise ValueError("primary decision structure changed")
    challenge_ids = select_challenge_frames(request, primary)
    rebuilt_challenge = parse_frame_decisions(
        request,
        _decision_set_input(challenge),
        required_frame_ids=challenge_ids,
    )
    if rebuilt_challenge != challenge:
        raise ValueError("challenge decision structure changed")


def certify_frame_presence_result(
    *,
    spec: VisibleEpisodeCountSpec,
    request: FrameDecisionRequest,
    primary: FrameDecisionSet,
    challenge: FrameDecisionSet,
    ledger: FramePresenceLedger,
    result: PresenceCountResult,
    interface_source_sha256: str,
    reducer_source_sha256: str,
    implementation_sources: Mapping[str, str],
    source_revision: str,
) -> FramePresenceCoverageCertificate:
    """Authenticate a complete frame-decision chain and certify its result."""

    spec_hash = _verify_self_hash(
        visible_episode_count_spec_to_dict(spec), name="CountSpec"
    )
    request_hash = _verify_self_hash(
        frame_decision_request_to_dict(request), name="frame request"
    )
    primary_hash = _verify_self_hash(
        frame_decision_set_to_dict(primary), name="primary decisions"
    )
    challenge_hash = _verify_self_hash(
        frame_decision_set_to_dict(challenge), name="challenge decisions"
    )
    _revalidate_chain_structure(spec, request, primary, challenge)
    if request.spec != spec or request.spec.complete_sha256 != spec_hash:
        raise ValueError("request CountSpec binding changed")
    expected_ledger = reconcile_frame_decisions(request, primary, challenge)
    if frame_presence_ledger_to_dict(expected_ledger) != frame_presence_ledger_to_dict(
        ledger
    ):
        raise ValueError("reconciled frame ledger changed")
    expected_result = reduce_presence_episodes(ledger)
    if presence_count_result_to_dict(expected_result) != presence_count_result_to_dict(
        result
    ):
        raise ValueError("presence count result changed")
    interface_hash = _sha256(interface_source_sha256, field="interface_source_sha256")
    reducer_hash = _sha256(reducer_source_sha256, field="reducer_source_sha256")
    if set(implementation_sources) != REQUIRED_IMPLEMENTATION_SOURCE_PATHS:
        raise ValueError(
            "implementation_sources does not match the required trust base"
        )
    if (
        implementation_sources["src/trec_video_2026/vqa_blind_frame_counting.py"]
        != interface_hash
        or implementation_sources["src/trec_video_2026/vqa_count_frame_presence.py"]
        != reducer_hash
    ):
        raise ValueError("duplicate implementation source hashes disagree")
    source_bindings = tuple(
        SourceBinding(
            path=_nonempty_string(path, field="implementation_sources.path"),
            sha256=_sha256(digest, field=f"implementation_sources[{path}]"),
        )
        for path, digest in sorted(implementation_sources.items())
    )
    revision = _nonempty_string(source_revision, field="source_revision")
    ledger_hash = canonical_sha256(frame_presence_ledger_to_dict(ledger))
    result_hash = canonical_sha256(presence_count_result_to_dict(result))
    reason_codes = list(result.reason_codes)
    if spec.cut_policy == "split_at_cuts":
        reason_codes.insert(0, "unsupported_cut_policy")
    outcome = "exact" if result.outcome == "exact" and not reason_codes else "abstain"
    value = result.value if outcome == "exact" else None
    unsigned = {
        "schema_version": 1,
        "outcome": outcome,
        "value": value,
        "reason_codes": reason_codes,
        "spec_complete_sha256": spec_hash,
        "request_complete_sha256": request_hash,
        "primary_complete_sha256": primary_hash,
        "challenge_complete_sha256": challenge_hash,
        "ledger_sha256": ledger_hash,
        "result_sha256": result_hash,
        "interface_source_sha256": interface_hash,
        "reducer_source_sha256": reducer_hash,
        "implementation_sources": [asdict(binding) for binding in source_bindings],
        "source_revision": revision,
    }
    return FramePresenceCoverageCertificate(
        outcome=outcome,
        value=value,
        reason_codes=tuple(reason_codes),
        spec_complete_sha256=spec_hash,
        request_complete_sha256=request_hash,
        primary_complete_sha256=primary_hash,
        challenge_complete_sha256=challenge_hash,
        ledger_sha256=ledger_hash,
        result_sha256=result_hash,
        interface_source_sha256=interface_hash,
        reducer_source_sha256=reducer_hash,
        implementation_sources=source_bindings,
        source_revision=revision,
        complete_sha256=canonical_sha256(unsigned),
    )
