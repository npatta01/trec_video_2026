"""Pure contracts for bounded multi-pass video counting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trec_video_2026.vqa_counting_evidence import validate_label_blind


_SUPPORTED_UNITS = frozenset(
    {
        "action_repetition",
        "appearance_episode",
        "static_cardinality",
        "max_simultaneous",
        "per_scene_sum",
    }
)
_TEMPORAL_UNITS = frozenset(
    {"action_repetition", "appearance_episode", "max_simultaneous"}
)
_STATIC_UNITS = frozenset({"static_cardinality", "per_scene_sum"})


@dataclass(frozen=True)
class Scope:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class CandidateView:
    view_id: str
    kind: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    timestamp_seconds: float | None = None


@dataclass(frozen=True)
class CountSpec:
    question_sha256: str
    target: str
    unit: str
    route: str
    scope: Scope
    inclusions: tuple[str, ...]
    exclusions: tuple[str, ...]
    identity_policy: str
    candidate_views: tuple[CandidateView, ...]
    uncertainty: str


@dataclass(frozen=True)
class UnsupportedCountSpec:
    unit: str
    reason: str


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    view_ids: tuple[str, ...]
    target_supported: bool
    ambiguous: bool
    descriptor: str
    start_seconds: float | None = None
    peak_seconds: float | None = None
    end_seconds: float | None = None
    shot_id: str | None = None
    timestamp_seconds: float | None = None
    bbox_norm: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class EvidenceLedger:
    reviewed_view_ids: tuple[str, ...]
    records: tuple[EvidenceRecord, ...]


@dataclass(frozen=True)
class Dispute:
    record_id: str
    reason: str


@dataclass(frozen=True)
class RefinementRequest:
    kind: str
    reason: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    view_id: str | None = None


@dataclass(frozen=True)
class CoverageReview:
    complete: bool
    reviewed_view_ids: tuple[str, ...]
    accepted_record_ids: tuple[str, ...]
    rejected_record_ids: tuple[str, ...]
    missing_view_ids: tuple[str, ...]
    disputes: tuple[Dispute, ...]
    refinement_request: RefinementRequest | None
    notes: str


@dataclass(frozen=True)
class CountResult:
    outcome: str
    value: int | None
    reasons: tuple[str, ...]
    selected_record_ids: tuple[str, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{field} must be an array")
    return value


def _strict_fields(
    value: Mapping[str, object], *, field: str, expected: frozenset[str]
) -> None:
    actual = frozenset(str(key) for key in value)
    if actual != expected:
        raise ValueError(
            f"{field} fields must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _strings(
    value: object, *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    result = tuple(
        _string(item, field=f"{field}[{index}]")
        for index, item in enumerate(_sequence(value, field=field))
    )
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _parse_view(
    value: object, *, index: int, scope: Scope, route: str
) -> CandidateView:
    payload = _mapping(value, field=f"candidate_views[{index}]")
    kind = _string(payload.get("kind"), field=f"candidate_views[{index}].kind")
    view_id = _string(payload.get("view_id"), field=f"candidate_views[{index}].view_id")
    if route == "temporal":
        _strict_fields(
            payload,
            field=f"candidate_views[{index}]",
            expected=frozenset({"view_id", "kind", "start_seconds", "end_seconds"}),
        )
        if kind != "window":
            raise ValueError("temporal candidate view kind must be window")
        start = _number(
            payload["start_seconds"], field=f"candidate_views[{index}].start_seconds"
        )
        end = _number(
            payload["end_seconds"], field=f"candidate_views[{index}].end_seconds"
        )
        if not scope.start_seconds <= start < end <= scope.end_seconds:
            raise ValueError("candidate view window must lie within scope")
        return CandidateView(view_id, kind, start_seconds=start, end_seconds=end)
    _strict_fields(
        payload,
        field=f"candidate_views[{index}]",
        expected=frozenset({"view_id", "kind", "timestamp_seconds"}),
    )
    if kind != "shot_frame":
        raise ValueError("static candidate view kind must be shot_frame")
    timestamp = _number(
        payload["timestamp_seconds"],
        field=f"candidate_views[{index}].timestamp_seconds",
    )
    if not scope.start_seconds <= timestamp <= scope.end_seconds:
        raise ValueError("candidate view timestamp must lie within scope")
    return CandidateView(view_id, kind, timestamp_seconds=timestamp)


def parse_count_spec(
    payload: object, *, question: str, duration_seconds: float
) -> CountSpec | UnsupportedCountSpec:
    validate_label_blind(payload)
    value = _mapping(payload, field="count_spec")
    _strict_fields(
        value,
        field="count_spec",
        expected=frozenset(
            {
                "schema_version",
                "question_sha256",
                "target",
                "unit",
                "route",
                "scope",
                "inclusions",
                "exclusions",
                "identity_policy",
                "candidate_views",
                "uncertainty",
            }
        ),
    )
    if value["schema_version"] != "bounded-multipass-count-spec-v1":
        raise ValueError("unsupported count spec schema_version")
    question_sha256 = _string(value["question_sha256"], field="question_sha256")
    expected_question_sha256 = hashlib.sha256(question.encode()).hexdigest()
    if question_sha256 != expected_question_sha256:
        raise ValueError("question_sha256 does not bind the supplied question")
    unit = _string(value["unit"], field="unit")
    if unit not in _SUPPORTED_UNITS:
        return UnsupportedCountSpec(unit=unit, reason=f"unsupported_unit:{unit}")
    route = _string(value["route"], field="route")
    if route not in {"temporal", "static"}:
        raise ValueError("route must be temporal or static")
    if (unit in _TEMPORAL_UNITS) != (route == "temporal"):
        raise ValueError("unit and route are incompatible")
    if unit in _STATIC_UNITS and route != "static":
        raise ValueError("unit and route are incompatible")
    duration = _number(duration_seconds, field="duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    scope_value = _mapping(value["scope"], field="scope")
    _strict_fields(
        scope_value,
        field="scope",
        expected=frozenset({"start_seconds", "end_seconds"}),
    )
    scope = Scope(
        _number(scope_value["start_seconds"], field="scope.start_seconds"),
        _number(scope_value["end_seconds"], field="scope.end_seconds"),
    )
    if not 0 <= scope.start_seconds < scope.end_seconds <= duration:
        raise ValueError("scope must lie within source duration")
    views = tuple(
        _parse_view(item, index=index, scope=scope, route=route)
        for index, item in enumerate(
            _sequence(value["candidate_views"], field="candidate_views")
        )
    )
    if not views:
        raise ValueError("candidate_views must not be empty")
    if len({item.view_id for item in views}) != len(views):
        raise ValueError("candidate view_id values must be unique")
    return CountSpec(
        question_sha256=question_sha256,
        target=_string(value["target"], field="target"),
        unit=unit,
        route=route,
        scope=scope,
        inclusions=_strings(value["inclusions"], field="inclusions"),
        exclusions=_strings(value["exclusions"], field="exclusions", allow_empty=True),
        identity_policy=_string(value["identity_policy"], field="identity_policy"),
        candidate_views=views,
        uncertainty=_string(value["uncertainty"], field="uncertainty"),
    )


def _parse_record(
    value: object,
    *,
    index: int,
    spec: CountSpec,
    allowed_view_ids: frozenset[str],
) -> EvidenceRecord:
    payload = _mapping(value, field=f"records[{index}]")
    common = frozenset(
        {
            "record_id",
            "view_ids",
            "target_supported",
            "ambiguous",
            "descriptor",
        }
    )
    temporal = frozenset({"start_seconds", "peak_seconds", "end_seconds"})
    static = frozenset({"shot_id", "timestamp_seconds", "bbox_norm"})
    _strict_fields(
        payload,
        field=f"records[{index}]",
        expected=common | (temporal if spec.route == "temporal" else static),
    )
    record_id = _string(payload["record_id"], field=f"records[{index}].record_id")
    view_ids = _strings(payload["view_ids"], field=f"records[{index}].view_ids")
    if not set(view_ids).issubset(allowed_view_ids):
        raise ValueError(f"records[{index}].view_ids contains an unrequested view")
    common_values = {
        "record_id": record_id,
        "view_ids": view_ids,
        "target_supported": _boolean(
            payload["target_supported"], field=f"records[{index}].target_supported"
        ),
        "ambiguous": _boolean(
            payload["ambiguous"], field=f"records[{index}].ambiguous"
        ),
        "descriptor": _string(
            payload["descriptor"], field=f"records[{index}].descriptor"
        ),
    }
    if spec.route == "temporal":
        start = _number(
            payload["start_seconds"], field=f"records[{index}].start_seconds"
        )
        peak = _number(payload["peak_seconds"], field=f"records[{index}].peak_seconds")
        end = _number(payload["end_seconds"], field=f"records[{index}].end_seconds")
        if (
            not spec.scope.start_seconds
            <= start
            <= peak
            <= end
            <= spec.scope.end_seconds
        ):
            raise ValueError(
                f"records[{index}] timestamps are outside scope or unordered"
            )
        if start == end:
            raise ValueError(f"records[{index}] must have positive duration")
        return EvidenceRecord(
            **common_values,
            start_seconds=start,
            peak_seconds=peak,
            end_seconds=end,
        )
    bbox_values = tuple(
        _number(item, field=f"records[{index}].bbox_norm[{offset}]")
        for offset, item in enumerate(
            _sequence(payload["bbox_norm"], field=f"records[{index}].bbox_norm")
        )
    )
    if len(bbox_values) != 4 or not all(0 <= item <= 1 for item in bbox_values):
        raise ValueError(
            f"records[{index}].bbox_norm must contain four normalized values"
        )
    if not bbox_values[0] < bbox_values[2] or not bbox_values[1] < bbox_values[3]:
        raise ValueError(f"records[{index}].bbox_norm must have positive area")
    timestamp = _number(
        payload["timestamp_seconds"], field=f"records[{index}].timestamp_seconds"
    )
    if not spec.scope.start_seconds <= timestamp <= spec.scope.end_seconds:
        raise ValueError(f"records[{index}].timestamp_seconds is outside scope")
    return EvidenceRecord(
        **common_values,
        shot_id=_string(payload["shot_id"], field=f"records[{index}].shot_id"),
        timestamp_seconds=timestamp,
        bbox_norm=bbox_values,
    )


def parse_evidence(
    payload: object,
    *,
    spec: CountSpec,
    requested_views: Sequence[CandidateView],
) -> EvidenceLedger:
    validate_label_blind(payload)
    value = _mapping(payload, field="evidence")
    _strict_fields(
        value,
        field="evidence",
        expected=frozenset({"schema_version", "reviewed_view_ids", "records"}),
    )
    if value["schema_version"] != "bounded-multipass-evidence-v1":
        raise ValueError("unsupported evidence schema_version")
    allowed = tuple(view.view_id for view in requested_views)
    if len(set(allowed)) != len(allowed):
        raise ValueError("requested view IDs must be unique")
    reviewed = _strings(
        value["reviewed_view_ids"], field="reviewed_view_ids", allow_empty=True
    )
    if set(reviewed) != set(allowed) or len(reviewed) != len(allowed):
        raise ValueError("reviewed_view_ids must exactly cover requested views")
    records = tuple(
        _parse_record(
            item,
            index=index,
            spec=spec,
            allowed_view_ids=frozenset(allowed),
        )
        for index, item in enumerate(_sequence(value["records"], field="records"))
    )
    ids = tuple(record.record_id for record in records)
    if len(set(ids)) != len(ids):
        raise ValueError("record_id values must be unique")
    return EvidenceLedger(reviewed_view_ids=reviewed, records=records)


def _parse_refinement(value: object, *, spec: CountSpec) -> RefinementRequest:
    payload = _mapping(value, field="refinement_request")
    kind = _string(payload.get("kind"), field="refinement_request.kind")
    if kind == "window":
        _strict_fields(
            payload,
            field="refinement_request",
            expected=frozenset({"kind", "start_seconds", "end_seconds", "reason"}),
        )
        start = _number(
            payload["start_seconds"], field="refinement_request.start_seconds"
        )
        end = _number(payload["end_seconds"], field="refinement_request.end_seconds")
        if not spec.scope.start_seconds <= start < end <= spec.scope.end_seconds:
            raise ValueError("refinement window must lie within scope")
        return RefinementRequest(
            kind=kind,
            start_seconds=start,
            end_seconds=end,
            reason=_string(payload["reason"], field="refinement_request.reason"),
        )
    if kind in {"shot_frame", "crop"}:
        _strict_fields(
            payload,
            field="refinement_request",
            expected=frozenset({"kind", "view_id", "reason"}),
        )
        return RefinementRequest(
            kind=kind,
            view_id=_string(payload["view_id"], field="refinement_request.view_id"),
            reason=_string(payload["reason"], field="refinement_request.reason"),
        )
    raise ValueError("unsupported refinement request kind")


def parse_coverage_review(
    payload: object, *, spec: CountSpec, ledger: EvidenceLedger
) -> CoverageReview:
    validate_label_blind(payload)
    value = _mapping(payload, field="coverage_review")
    _strict_fields(
        value,
        field="coverage_review",
        expected=frozenset(
            {
                "schema_version",
                "complete",
                "reviewed_view_ids",
                "accepted_record_ids",
                "rejected_record_ids",
                "missing_view_ids",
                "disputes",
                "refinement_request",
                "notes",
            }
        ),
    )
    if value["schema_version"] != "bounded-multipass-coverage-v1":
        raise ValueError("unsupported coverage schema_version")
    view_ids = frozenset(view.view_id for view in spec.candidate_views)
    record_ids = frozenset(record.record_id for record in ledger.records)
    reviewed = _strings(
        value["reviewed_view_ids"], field="reviewed_view_ids", allow_empty=True
    )
    missing = _strings(
        value["missing_view_ids"], field="missing_view_ids", allow_empty=True
    )
    if not set(reviewed).issubset(view_ids) or not set(missing).issubset(view_ids):
        raise ValueError("coverage view IDs must come from the CountSpec")
    if set(reviewed) & set(missing) or set(reviewed) | set(missing) != view_ids:
        raise ValueError("reviewed and missing view IDs must partition requested views")
    accepted = _strings(
        value["accepted_record_ids"], field="accepted_record_ids", allow_empty=True
    )
    rejected = _strings(
        value["rejected_record_ids"], field="rejected_record_ids", allow_empty=True
    )
    if not set(accepted).issubset(record_ids) or not set(rejected).issubset(record_ids):
        raise ValueError("coverage record IDs must come from the evidence ledger")
    if set(accepted) & set(rejected):
        raise ValueError("a record cannot be both accepted and rejected")
    disputes = []
    for index, item in enumerate(_sequence(value["disputes"], field="disputes")):
        dispute = _mapping(item, field=f"disputes[{index}]")
        _strict_fields(
            dispute,
            field=f"disputes[{index}]",
            expected=frozenset({"record_id", "reason"}),
        )
        record_id = _string(dispute["record_id"], field=f"disputes[{index}].record_id")
        if record_id not in record_ids:
            raise ValueError("dispute record_id must come from the evidence ledger")
        disputes.append(
            Dispute(
                record_id=record_id,
                reason=_string(dispute["reason"], field=f"disputes[{index}].reason"),
            )
        )
    if len({item.record_id for item in disputes}) != len(disputes):
        raise ValueError("dispute record_id values must be unique")
    complete = _boolean(value["complete"], field="complete")
    refinement = (
        None
        if value["refinement_request"] is None
        else _parse_refinement(value["refinement_request"], spec=spec)
    )
    if complete and (missing or disputes or refinement is not None):
        raise ValueError("complete coverage cannot have gaps, disputes, or refinement")
    return CoverageReview(
        complete=complete,
        reviewed_view_ids=reviewed,
        accepted_record_ids=accepted,
        rejected_record_ids=rejected,
        missing_view_ids=missing,
        disputes=tuple(disputes),
        refinement_request=refinement,
        notes=_string(value["notes"], field="notes"),
    )


def build_refinement_request(
    review: CoverageReview, *, requested_views: Sequence[CandidateView]
) -> RefinementRequest | None:
    del requested_views
    if review.complete:
        return None
    return review.refinement_request


def reduce_count(
    spec: CountSpec, ledger: EvidenceLedger, review: CoverageReview
) -> CountResult:
    if not review.complete:
        return CountResult("abstain", None, ("coverage_incomplete",), ())
    records_by_id = {record.record_id: record for record in ledger.records}
    selected = tuple(
        records_by_id[record_id]
        for record_id in review.accepted_record_ids
        if record_id in records_by_id
    )
    supported = tuple(
        record
        for record in selected
        if record.target_supported and not record.ambiguous
    )
    if len(supported) != len(selected):
        return CountResult("abstain", None, ("unresolved_record",), ())
    if not supported:
        return CountResult("abstain", None, ("empty_evidence",), ())
    if spec.route == "temporal":
        ordered = sorted(
            supported, key=lambda item: (item.start_seconds, item.record_id)
        )
        overlaps = any(
            previous.end_seconds is not None
            and current.start_seconds is not None
            and current.start_seconds < previous.end_seconds
            for previous, current in zip(ordered, ordered[1:], strict=False)
        )
        if spec.unit == "appearance_episode" and overlaps:
            return CountResult("abstain", None, ("overlapping_records",), ())
        if spec.unit == "max_simultaneous":
            boundaries = sorted(
                [
                    (timestamp, delta)
                    for record in ordered
                    for timestamp, delta in (
                        (record.start_seconds, 1),
                        (record.end_seconds, -1),
                    )
                    if timestamp is not None
                ],
                key=lambda item: (item[0], item[1]),
            )
            active = 0
            value = 0
            for _timestamp, delta in boundaries:
                active += delta
                value = max(value, active)
        else:
            value = len(ordered)
        ids = tuple(record.record_id for record in ordered)
    elif spec.unit == "per_scene_sum":
        value = len(supported)
        ids = tuple(sorted(record.record_id for record in supported))
    else:
        # Static records are one localized instance each in the selected shot/frame.
        if len({record.shot_id for record in supported}) != 1:
            return CountResult("abstain", None, ("multiple_static_shots",), ())
        value = len(supported)
        ids = tuple(sorted(record.record_id for record in supported))
    return CountResult("exact", value, (), ids)


def _view_to_dict(view: CandidateView) -> dict[str, object]:
    result: dict[str, object] = {"view_id": view.view_id, "kind": view.kind}
    if view.kind == "window":
        result.update(
            {
                "start_seconds": view.start_seconds,
                "end_seconds": view.end_seconds,
            }
        )
    else:
        result["timestamp_seconds"] = view.timestamp_seconds
    return result


def count_spec_to_dict(spec: CountSpec) -> dict[str, object]:
    return {
        "schema_version": "bounded-multipass-count-spec-v1",
        "question_sha256": spec.question_sha256,
        "target": spec.target,
        "unit": spec.unit,
        "route": spec.route,
        "scope": {
            "start_seconds": spec.scope.start_seconds,
            "end_seconds": spec.scope.end_seconds,
        },
        "inclusions": list(spec.inclusions),
        "exclusions": list(spec.exclusions),
        "identity_policy": spec.identity_policy,
        "candidate_views": [_view_to_dict(view) for view in spec.candidate_views],
        "uncertainty": spec.uncertainty,
    }


def _record_to_dict(record: EvidenceRecord) -> dict[str, object]:
    result: dict[str, object] = {
        "record_id": record.record_id,
        "view_ids": list(record.view_ids),
        "target_supported": record.target_supported,
        "ambiguous": record.ambiguous,
        "descriptor": record.descriptor,
    }
    if record.start_seconds is not None:
        result.update(
            {
                "start_seconds": record.start_seconds,
                "peak_seconds": record.peak_seconds,
                "end_seconds": record.end_seconds,
            }
        )
    else:
        result.update(
            {
                "shot_id": record.shot_id,
                "timestamp_seconds": record.timestamp_seconds,
                "bbox_norm": list(record.bbox_norm or ()),
            }
        )
    return result


def evidence_ledger_to_dict(ledger: EvidenceLedger) -> dict[str, object]:
    return {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": list(ledger.reviewed_view_ids),
        "records": [_record_to_dict(record) for record in ledger.records],
    }


def _refinement_to_dict(request: RefinementRequest) -> dict[str, object]:
    result: dict[str, object] = {"kind": request.kind, "reason": request.reason}
    if request.kind == "window":
        result.update(
            {
                "start_seconds": request.start_seconds,
                "end_seconds": request.end_seconds,
            }
        )
    else:
        result["view_id"] = request.view_id
    return result


def coverage_review_to_dict(review: CoverageReview) -> dict[str, object]:
    return {
        "schema_version": "bounded-multipass-coverage-v1",
        "complete": review.complete,
        "reviewed_view_ids": list(review.reviewed_view_ids),
        "accepted_record_ids": list(review.accepted_record_ids),
        "rejected_record_ids": list(review.rejected_record_ids),
        "missing_view_ids": list(review.missing_view_ids),
        "disputes": [
            {"record_id": dispute.record_id, "reason": dispute.reason}
            for dispute in review.disputes
        ],
        "refinement_request": (
            None
            if review.refinement_request is None
            else _refinement_to_dict(review.refinement_request)
        ),
        "notes": review.notes,
    }


def count_result_to_dict(result: CountResult) -> dict[str, object]:
    return {
        "outcome": result.outcome,
        "value": result.value,
        "reasons": list(result.reasons),
        "selected_record_ids": list(result.selected_record_ids),
    }


def build_count_spec_prompt(*, question: str, duration_seconds: float) -> str:
    schema = {
        "schema_version": "bounded-multipass-count-spec-v1",
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "target": "observable target phrase",
        "unit": sorted(_SUPPORTED_UNITS),
        "route": ["temporal", "static"],
        "scope": {"start_seconds": 0.0, "end_seconds": duration_seconds},
        "inclusions": ["observable inclusion rule"],
        "exclusions": ["observable exclusion rule"],
        "identity_policy": "observable identity rule",
        "candidate_views": ["windows or shot frames covering the complete scope"],
        "uncertainty": "none or a concise observable uncertainty",
    }
    return (
        "Define the observable counting task for the supplied complete video. "
        "Do not return a final count. Cover the complete temporal scope and return "
        "strict JSON matching this contract.\n"
        f"Question: {question}\n"
        f"Source duration seconds: {duration_seconds}\n"
        f"Contract: {json.dumps(schema, sort_keys=True)}"
    )


def build_evidence_prompt(spec: CountSpec) -> str:
    return (
        "Inspect every supplied view and emit strict JSON with one record per "
        "counted observable unit. Bind each record to its view IDs, use source-time "
        "timestamps for temporal records or normalized boxes for static records, "
        "and mark ambiguity explicitly. Do not emit a scalar total.\n"
        f"CountSpec: {json.dumps(count_spec_to_dict(spec), sort_keys=True)}"
    )


def build_coverage_prompt(spec: CountSpec, ledger: EvidenceLedger) -> str:
    return (
        "Audit coverage without computing or seeing a scalar total. Check source "
        "endpoints, every requested view, omissions, duplicate or merged records, "
        "scope violations, and shot boundaries. Return strict JSON using the "
        "bounded-multipass-coverage-v1 contract and request at most one targeted "
        "refinement.\n"
        f"CountSpec: {json.dumps(count_spec_to_dict(spec), sort_keys=True)}\n"
        f"Evidence: {json.dumps(evidence_ledger_to_dict(ledger), sort_keys=True)}"
    )
