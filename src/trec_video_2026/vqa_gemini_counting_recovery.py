"""Pure contracts for multi-pass Gemini counting-evidence recovery."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trec_video_2026.vqa_counting_evidence import validate_label_blind

COUNT_UNITS = frozenset(
    {
        "appearance_episode",
        "event_transition",
        "physical_entity",
        "visible_instance",
        "part_cardinality",
    }
)
CHALLENGE_DISPOSITIONS = frozenset(
    {"accept", "duplicate", "split", "predicate_mismatch", "ambiguous"}
)


@dataclass(frozen=True)
class TimeRange:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    span: TimeRange
    reason: str


@dataclass(frozen=True)
class LocatorResult:
    count_unit: str
    target: str
    candidates: tuple[Candidate, ...]
    reviewed_spans: tuple[TimeRange, ...]
    unresolved_spans: tuple[TimeRange, ...]


@dataclass(frozen=True)
class AtomicInstance:
    instance_id: str
    start_seconds: float
    peak_seconds: float
    end_seconds: float
    before_state: str
    during_state: str
    after_state: str
    separation_evidence: str
    predicate_supported: bool
    ambiguous: bool


@dataclass(frozen=True)
class AtomicLedger:
    instances: tuple[AtomicInstance, ...]
    unresolved_observations: tuple[str, ...]


@dataclass(frozen=True)
class InstanceDisposition:
    instance_id: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class ChallengeResult:
    dispositions: tuple[InstanceDisposition, ...]
    missed_instances: tuple[AtomicInstance, ...]
    requested_ranges: tuple[TimeRange, ...]
    unresolved_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryResult:
    outcome: str
    value: int | None
    accepted_instance_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, object], *, expected: frozenset[str], field: str
) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise ValueError(
            f"{field} fields changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a boolean")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be an array")
    return value


def _time_range(value: object, *, field: str, duration_seconds: float) -> TimeRange:
    payload = _mapping(value, field=field)
    _exact_keys(
        payload,
        expected=frozenset({"start_seconds", "end_seconds"}),
        field=field,
    )
    start = _number(payload["start_seconds"], field=f"{field}.start_seconds")
    end = _number(payload["end_seconds"], field=f"{field}.end_seconds")
    if not (0.0 <= start <= end <= duration_seconds):
        raise ValueError(f"{field} must remain inside the source duration")
    return TimeRange(start, end)


def _assert_partition(
    reviewed: tuple[TimeRange, ...], unresolved: tuple[TimeRange, ...], duration: float
) -> None:
    spans = sorted(
        (*reviewed, *unresolved),
        key=lambda item: (item.start_seconds, item.end_seconds),
    )
    if (
        not spans
        or spans[0].start_seconds > 1e-6
        or spans[-1].end_seconds < duration - 1e-6
    ):
        raise ValueError("locator spans must cover the complete source duration")
    cursor = 0.0
    for span in spans:
        if span.start_seconds > cursor + 1e-6:
            raise ValueError("locator spans must cover the complete source duration")
        cursor = max(cursor, span.end_seconds)
    if cursor < duration - 1e-6:
        raise ValueError("locator spans must cover the complete source duration")


def parse_locator(payload: object, *, duration_seconds: float) -> LocatorResult:
    validate_label_blind(payload)
    root = _mapping(payload, field="locator")
    _exact_keys(
        root,
        expected=frozenset(
            {
                "schema_version",
                "count_unit",
                "target",
                "candidates",
                "reviewed_spans",
                "unresolved_spans",
            }
        ),
        field="locator",
    )
    if root["schema_version"] != 1:
        raise ValueError("locator schema_version must be 1")
    count_unit = _string(root["count_unit"], field="locator.count_unit")
    if count_unit not in COUNT_UNITS:
        raise ValueError(f"unsupported count_unit: {count_unit}")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, value in enumerate(_sequence(root["candidates"], field="candidates")):
        item = _mapping(value, field=f"candidates[{index}]")
        _exact_keys(
            item,
            expected=frozenset(
                {"candidate_id", "start_seconds", "end_seconds", "reason"}
            ),
            field=f"candidates[{index}]",
        )
        candidate_id = _string(item["candidate_id"], field="candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        span = _time_range(
            {
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
            },
            field=f"candidates[{index}]",
            duration_seconds=duration_seconds,
        )
        candidates.append(
            Candidate(candidate_id, span, _string(item["reason"], field="reason"))
        )
    reviewed = tuple(
        _time_range(
            value, field=f"reviewed_spans[{index}]", duration_seconds=duration_seconds
        )
        for index, value in enumerate(
            _sequence(root["reviewed_spans"], field="reviewed_spans")
        )
    )
    unresolved = tuple(
        _time_range(
            value, field=f"unresolved_spans[{index}]", duration_seconds=duration_seconds
        )
        for index, value in enumerate(
            _sequence(root["unresolved_spans"], field="unresolved_spans")
        )
    )
    _assert_partition(reviewed, unresolved, duration_seconds)
    return LocatorResult(
        count_unit=count_unit,
        target=_string(root["target"], field="locator.target"),
        candidates=tuple(candidates),
        reviewed_spans=reviewed,
        unresolved_spans=unresolved,
    )


def verification_envelope(
    locator: LocatorResult,
    duration_seconds: float,
    padding_seconds: float = 1.5,
) -> TimeRange:
    if not locator.candidates:
        return TimeRange(0.0, duration_seconds)
    return TimeRange(
        max(
            0.0,
            min(candidate.span.start_seconds for candidate in locator.candidates)
            - padding_seconds,
        ),
        min(
            duration_seconds,
            max(candidate.span.end_seconds for candidate in locator.candidates)
            + padding_seconds,
        ),
    )


def _parse_instance(
    value: object, *, field: str, duration_seconds: float, envelope: TimeRange
) -> AtomicInstance:
    item = _mapping(value, field=field)
    _exact_keys(
        item,
        expected=frozenset(
            {
                "instance_id",
                "start_seconds",
                "peak_seconds",
                "end_seconds",
                "before_state",
                "during_state",
                "after_state",
                "separation_evidence",
                "predicate_supported",
                "ambiguous",
            }
        ),
        field=field,
    )
    start = _number(item["start_seconds"], field=f"{field}.start_seconds")
    peak = _number(item["peak_seconds"], field=f"{field}.peak_seconds")
    end = _number(item["end_seconds"], field=f"{field}.end_seconds")
    if not (0.0 <= start <= peak <= end <= duration_seconds):
        raise ValueError(
            f"{field} must satisfy start <= peak <= end inside source duration"
        )
    if start < envelope.start_seconds - 1e-6 or end > envelope.end_seconds + 1e-6:
        raise ValueError(f"{field} lies outside the verification envelope")
    return AtomicInstance(
        instance_id=_string(item["instance_id"], field=f"{field}.instance_id"),
        start_seconds=start,
        peak_seconds=peak,
        end_seconds=end,
        before_state=_string(item["before_state"], field=f"{field}.before_state"),
        during_state=_string(item["during_state"], field=f"{field}.during_state"),
        after_state=_string(item["after_state"], field=f"{field}.after_state"),
        separation_evidence=_string(
            item["separation_evidence"], field=f"{field}.separation_evidence"
        ),
        predicate_supported=_boolean(
            item["predicate_supported"], field=f"{field}.predicate_supported"
        ),
        ambiguous=_boolean(item["ambiguous"], field=f"{field}.ambiguous"),
    )


def parse_atomic_ledger(
    payload: object, *, duration_seconds: float, envelope: TimeRange
) -> AtomicLedger:
    validate_label_blind(payload)
    root = _mapping(payload, field="atomic_ledger")
    _exact_keys(
        root,
        expected=frozenset({"schema_version", "instances", "unresolved_observations"}),
        field="atomic_ledger",
    )
    if root["schema_version"] != 1:
        raise ValueError("atomic_ledger schema_version must be 1")
    instances = tuple(
        _parse_instance(
            value,
            field=f"instances[{index}]",
            duration_seconds=duration_seconds,
            envelope=envelope,
        )
        for index, value in enumerate(_sequence(root["instances"], field="instances"))
    )
    ids = [instance.instance_id for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate instance_id")
    unresolved = tuple(
        _string(value, field=f"unresolved_observations[{index}]")
        for index, value in enumerate(
            _sequence(root["unresolved_observations"], field="unresolved_observations")
        )
    )
    return AtomicLedger(instances, unresolved)


def parse_challenge(
    payload: object, *, ledger: AtomicLedger, duration_seconds: float
) -> ChallengeResult:
    validate_label_blind(payload)
    root = _mapping(payload, field="challenge")
    _exact_keys(
        root,
        expected=frozenset(
            {
                "schema_version",
                "instance_dispositions",
                "missed_instances",
                "requested_ranges",
                "unresolved_reasons",
            }
        ),
        field="challenge",
    )
    if root["schema_version"] != 1:
        raise ValueError("challenge schema_version must be 1")
    dispositions: list[InstanceDisposition] = []
    ledger_ids = {instance.instance_id for instance in ledger.instances}
    seen: set[str] = set()
    for index, value in enumerate(
        _sequence(root["instance_dispositions"], field="instance_dispositions")
    ):
        item = _mapping(value, field=f"instance_dispositions[{index}]")
        _exact_keys(
            item,
            expected=frozenset({"instance_id", "disposition", "reason"}),
            field=f"instance_dispositions[{index}]",
        )
        instance_id = _string(item["instance_id"], field="instance_id")
        if instance_id not in ledger_ids:
            raise ValueError(f"unknown challenge instance_id: {instance_id}")
        if instance_id in seen:
            raise ValueError(f"duplicate challenge instance_id: {instance_id}")
        seen.add(instance_id)
        disposition = _string(item["disposition"], field="disposition")
        if disposition not in CHALLENGE_DISPOSITIONS:
            raise ValueError(f"unknown challenge disposition: {disposition}")
        dispositions.append(
            InstanceDisposition(
                instance_id,
                disposition,
                _string(item["reason"], field="reason"),
            )
        )
    if seen != ledger_ids:
        raise ValueError("challenge must disposition every ledger instance")
    full = TimeRange(0.0, duration_seconds)
    missed = tuple(
        _parse_instance(
            value,
            field=f"missed_instances[{index}]",
            duration_seconds=duration_seconds,
            envelope=full,
        )
        for index, value in enumerate(
            _sequence(root["missed_instances"], field="missed_instances")
        )
    )
    requested = tuple(
        _time_range(
            {
                "start_seconds": _mapping(value, field=f"requested_ranges[{index}]")[
                    "start_seconds"
                ],
                "end_seconds": _mapping(value, field=f"requested_ranges[{index}]")[
                    "end_seconds"
                ],
            },
            field=f"requested_ranges[{index}]",
            duration_seconds=duration_seconds,
        )
        for index, value in enumerate(
            _sequence(root["requested_ranges"], field="requested_ranges")
        )
    )
    for index, value in enumerate(
        _sequence(root["requested_ranges"], field="requested_ranges")
    ):
        item = _mapping(value, field=f"requested_ranges[{index}]")
        _exact_keys(
            item,
            expected=frozenset({"start_seconds", "end_seconds", "reason"}),
            field=f"requested_ranges[{index}]",
        )
        _string(item["reason"], field=f"requested_ranges[{index}].reason")
    unresolved = tuple(
        _string(value, field=f"unresolved_reasons[{index}]")
        for index, value in enumerate(
            _sequence(root["unresolved_reasons"], field="unresolved_reasons")
        )
    )
    return ChallengeResult(tuple(dispositions), missed, requested, unresolved)


def reduce_recovery_result(
    ledger: AtomicLedger,
    challenge: ChallengeResult,
    additive: AtomicLedger | None,
) -> RecoveryResult:
    accepted = [
        instance
        for instance in ledger.instances
        if instance.predicate_supported and not instance.ambiguous
    ]
    disposition_by_id = {
        disposition.instance_id: disposition.disposition
        for disposition in challenge.dispositions
    }
    accepted = [
        instance
        for instance in accepted
        if disposition_by_id.get(instance.instance_id) == "accept"
    ]
    if (challenge.requested_ranges or challenge.missed_instances) and additive is None:
        return RecoveryResult(
            "abstain",
            None,
            tuple(instance.instance_id for instance in accepted),
            ("additional_verification_required",),
        )
    if ledger.unresolved_observations or challenge.unresolved_reasons:
        return RecoveryResult(
            "abstain",
            None,
            tuple(instance.instance_id for instance in accepted),
            ("unresolved_challenge",),
        )
    if any(
        disposition.disposition != "accept" for disposition in challenge.dispositions
    ):
        return RecoveryResult(
            "abstain",
            None,
            tuple(instance.instance_id for instance in accepted),
            ("identity_or_predicate_conflict",),
        )
    all_instances = [*accepted, *challenge.missed_instances]
    if additive is not None:
        all_instances.extend(additive.instances)
        if additive.unresolved_observations:
            return RecoveryResult(
                "abstain",
                None,
                tuple(instance.instance_id for instance in all_instances),
                ("unresolved_additive_verification",),
            )
    ids = tuple(instance.instance_id for instance in all_instances)
    if len(ids) != len(set(ids)):
        return RecoveryResult("abstain", None, ids, ("duplicate_instance_id",))
    return RecoveryResult("exact", len(all_instances), ids, ())


def _row_question(row: Mapping[str, object]) -> tuple[str, float]:
    validate_label_blind(row)
    return (
        _string(row.get("question"), field="question"),
        _number(row.get("duration_seconds"), field="duration_seconds"),
    )


def build_locator_prompt(row: Mapping[str, object]) -> str:
    question, duration = _row_question(row)
    return (
        "Inspect the complete native video and locate every interval potentially relevant "
        "to the counting question. Do not answer the question and do not emit a final count. "
        "Partition the complete duration into reviewed and unresolved spans.\n"
        f"Source duration: {duration:.3f} seconds\nQuestion: {question}\n"
        "Return strict JSON using schema_version 1, count_unit, target, candidates, "
        "reviewed_spans, and unresolved_spans."
    )


def build_verifier_prompt(
    row: Mapping[str, object], locator: LocatorResult, envelope: TimeRange
) -> str:
    question, duration = _row_question(row)
    candidate_text = json.dumps(
        [
            {
                "candidate_id": item.candidate_id,
                "start_seconds": item.span.start_seconds,
                "end_seconds": item.span.end_seconds,
                "reason": item.reason,
            }
            for item in locator.candidates
        ],
        sort_keys=True,
    )
    return (
        "Inspect the timestamp-clipped high-frame-rate native video and emit one atomic "
        "record per observed instance. Use absolute source timestamps, not clip-relative "
        "timestamps. Do not emit a final numeric answer.\n"
        f"Source duration: {duration:.3f} seconds\n"
        f"Clip bounds: {envelope.start_seconds:.3f}-{envelope.end_seconds:.3f} seconds\n"
        f"Question: {question}\nCandidates: {candidate_text}"
    )


def build_challenger_prompt(
    row: Mapping[str, object], locator: LocatorResult, ledger: AtomicLedger
) -> str:
    question, duration = _row_question(row)
    return (
        "Inspect the complete high-frame-rate native video and seek counterexamples to "
        "the proposed atomic ledger: missed instances, duplicates, splits, predicate "
        "mismatches, clip-edge events, and unresolved intervals. Do not merely agree.\n"
        f"Source duration: {duration:.3f} seconds\nQuestion: {question}\n"
        f"Locator: {locator!r}\nAtomic ledger: {ledger!r}"
    )
