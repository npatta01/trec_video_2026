"""Typed, label-blind video count queries and temporal evidence graphs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256

SCHEMA_VERSION = 1
PROOF_SCHEMA_VERSION = 2

TARGET_KINDS = frozenset({"entity", "part", "event"})
COUNT_UNITS = frozenset(
    {
        "physical_entity",
        "appearance",
        "visible_instance",
        "part",
        "event_onset",
        "active_event",
    }
)
BOUNDARY_POLICIES = frozenset({"starts_within", "overlaps", "completes_within"})
IDENTITY_POLICIES = frozenset({"physical", "within_shot", "appearance_episode"})
CROSS_SHOT_POLICIES = frozenset({"abstain", "separate"})
VISIBILITY_POLICIES = frozenset({"visible", "partially_visible", "amodal"})
AGGREGATIONS = frozenset(
    {"cardinality", "distinct", "max_concurrent", "transition_count"}
)
EVIDENCE_OBLIGATIONS = frozenset(
    {
        "target_coverage",
        "shot_boundaries",
        "part_visibility",
        "event_boundary_state",
        "cross_shot_identity",
    }
)
VISIBILITY_VALUES = frozenset({"visible", "partially_visible", "amodal"})
RELATION_STATES = frozenset({"confirmed_same", "confirmed_different", "unresolved"})
COVERAGE_STATES = frozenset({"complete", "partial"})
ONSET_STATES = frozenset({"observed", "left_censored", "unknown"})
COMPLETION_STATES = frozenset({"observed", "right_censored", "unknown"})

_QUERY_ROOT_FIELDS = frozenset(
    {"schema_version", "query_id", "interpretations", "complete_sha256"}
)
_INTERPRETATION_FIELDS = frozenset(
    {
        "interpretation_id",
        "target_kind",
        "target_predicate",
        "owner_predicate",
        "count_unit",
        "scope",
        "identity_policy",
        "cross_shot_unknown",
        "visibility_policy",
        "aggregation",
        "evidence_obligations",
    }
)
_SCOPE_FIELDS = frozenset({"start_seconds", "end_seconds", "boundary_policy"})
_GRAPH_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "graph_id",
        "duration_seconds",
        "shots",
        "entities",
        "parts",
        "events",
        "identity_relations",
        "coverage",
        "complete_sha256",
    }
)
_SHOT_FIELDS = frozenset({"shot_id", "start_seconds", "end_seconds"})
_PROVENANCE_FIELDS = frozenset({"source_kind", "source_id", "timestamps_seconds"})
_ENTITY_FIELDS = frozenset(
    {
        "node_id",
        "predicate",
        "start_seconds",
        "end_seconds",
        "shot_id",
        "visibility",
        "provenance",
    }
)
_PART_FIELDS = _ENTITY_FIELDS | {"owner_node_id"}
_EVENT_FIELDS = frozenset(
    {
        "node_id",
        "predicate",
        "start_seconds",
        "end_seconds",
        "shot_id",
        "onset_state",
        "completion_state",
        "participant_node_ids",
        "provenance",
    }
)
_RELATION_FIELDS = frozenset({"left_node_id", "right_node_id", "relation_state"})
_COVERAGE_FIELDS = frozenset(
    {
        "claim_id",
        "obligation",
        "target_kind",
        "target_predicate",
        "start_seconds",
        "end_seconds",
        "spatial_scope",
        "temporal_resolution_seconds",
        "coverage_state",
    }
)


@dataclass(frozen=True)
class TimeScope:
    start_seconds: float
    end_seconds: float
    boundary_policy: str


@dataclass(frozen=True)
class QueryInterpretation:
    interpretation_id: str
    target_kind: str
    target_predicate: str
    owner_predicate: str | None
    count_unit: str
    scope: TimeScope
    identity_policy: str
    cross_shot_unknown: str
    visibility_policy: str
    aggregation: str
    evidence_obligations: tuple[str, ...]


@dataclass(frozen=True)
class CountQuery:
    query_id: str
    interpretations: tuple[QueryInterpretation, ...]


@dataclass(frozen=True)
class Provenance:
    source_kind: str
    source_id: str
    timestamps_seconds: tuple[float, ...]


@dataclass(frozen=True)
class Shot:
    shot_id: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class EntityObservation:
    node_id: str
    predicate: str
    start_seconds: float
    end_seconds: float
    shot_id: str
    visibility: str
    provenance: Provenance


@dataclass(frozen=True)
class PartObservation:
    node_id: str
    predicate: str
    owner_node_id: str
    start_seconds: float
    end_seconds: float
    shot_id: str
    visibility: str
    provenance: Provenance


@dataclass(frozen=True)
class EventObservation:
    node_id: str
    predicate: str
    start_seconds: float
    end_seconds: float
    shot_id: str
    onset_state: str
    completion_state: str
    participant_node_ids: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True)
class IdentityRelation:
    left_node_id: str
    right_node_id: str
    relation_state: str


@dataclass(frozen=True)
class CoverageClaim:
    claim_id: str
    obligation: str
    target_kind: str
    target_predicate: str
    start_seconds: float
    end_seconds: float
    spatial_scope: str
    temporal_resolution_seconds: float
    coverage_state: str


@dataclass(frozen=True)
class EvidenceGraph:
    graph_id: str
    duration_seconds: float
    shots: tuple[Shot, ...]
    entities: tuple[EntityObservation, ...]
    parts: tuple[PartObservation, ...]
    events: tuple[EventObservation, ...]
    identity_relations: tuple[IdentityRelation, ...]
    coverage: tuple[CoverageClaim, ...]


@dataclass(frozen=True)
class InterpretationTrace:
    interpretation_id: str
    outcome: str
    lower_bound: int
    upper_bound: int | None
    exact_count: int | None
    selected_node_ids: tuple[str, ...]
    selected_identity_relation_keys: tuple[str, ...]
    selected_coverage_claim_ids: tuple[str, ...]
    satisfied_obligations: tuple[str, ...]
    failed_obligations: tuple[str, ...]
    abstention_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProofState:
    query_id: str
    graph_id: str
    outcome: str
    lower_bound: int
    upper_bound: int | None
    exact_count: int | None
    traces: tuple[InterpretationTrace, ...]
    abstention_reasons: tuple[str, ...]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{field} must be an array")
    return value


def _exact_fields(
    payload: Mapping[str, object], *, allowed: frozenset[str], field: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {field} fields: {unknown}")
    missing = sorted(allowed - {"complete_sha256"} - set(payload))
    if missing:
        raise ValueError(f"missing {field} fields: {missing}")


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _enum(value: object, *, field: str, allowed: frozenset[str]) -> str:
    result = _string(value, field=field)
    if result not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")
    return result


def _bounded_interval(
    payload: Mapping[str, object], *, field: str, duration_seconds: float
) -> tuple[float, float]:
    start = _number(payload.get("start_seconds"), field=f"{field}.start_seconds")
    end = _number(payload.get("end_seconds"), field=f"{field}.end_seconds")
    if start > end:
        raise ValueError(f"{field}.start_seconds must not exceed end_seconds")
    if start < 0 or end > duration_seconds:
        raise ValueError(f"{field} interval must be within video duration")
    return start, end


def _verify_embedded_hash(payload: Mapping[str, object], *, field: str) -> None:
    embedded = payload.get("complete_sha256")
    if embedded is None:
        return
    if not isinstance(embedded, str):
        raise TypeError(f"{field}.complete_sha256 must be a string")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    )
    if actual != embedded:
        raise ValueError(f"{field} complete_sha256 changed")


def parse_count_query(payload: Mapping[str, object]) -> CountQuery:
    validate_label_blind(payload)
    _exact_fields(payload, allowed=_QUERY_ROOT_FIELDS, field="root")
    _verify_embedded_hash(payload, field="query")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"query schema_version must be {SCHEMA_VERSION}")
    query_id = _string(payload.get("query_id"), field="query_id")
    raw_interpretations = _sequence(
        payload.get("interpretations"), field="interpretations"
    )
    if not raw_interpretations:
        raise ValueError("interpretations must not be empty")
    interpretations: list[QueryInterpretation] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_interpretations):
        item = _mapping(raw, field=f"interpretations[{index}]")
        _exact_fields(
            item, allowed=_INTERPRETATION_FIELDS, field=f"interpretations[{index}]"
        )
        interpretation_id = _string(
            item.get("interpretation_id"),
            field=f"interpretations[{index}].interpretation_id",
        )
        if interpretation_id in seen_ids:
            raise ValueError("interpretation IDs must be unique")
        seen_ids.add(interpretation_id)
        scope_payload = _mapping(
            item.get("scope"), field=f"interpretations[{index}].scope"
        )
        _exact_fields(
            scope_payload,
            allowed=_SCOPE_FIELDS,
            field=f"interpretations[{index}].scope",
        )
        start = _number(
            scope_payload.get("start_seconds"),
            field=f"interpretations[{index}].scope.start_seconds",
        )
        end = _number(
            scope_payload.get("end_seconds"),
            field=f"interpretations[{index}].scope.end_seconds",
        )
        if start < 0 or start > end:
            raise ValueError("query scope start_seconds must not exceed end_seconds")
        obligations = tuple(
            _enum(
                value,
                field=f"interpretations[{index}].evidence_obligations",
                allowed=EVIDENCE_OBLIGATIONS,
            )
            for value in _sequence(
                item.get("evidence_obligations"),
                field=f"interpretations[{index}].evidence_obligations",
            )
        )
        if len(obligations) != len(set(obligations)):
            raise ValueError("evidence obligations must be unique")
        if "target_coverage" not in obligations:
            raise ValueError("target_coverage obligation is required")
        interpretation = QueryInterpretation(
            interpretation_id=interpretation_id,
            target_kind=_enum(
                item.get("target_kind"),
                field=f"interpretations[{index}].target_kind",
                allowed=TARGET_KINDS,
            ),
            target_predicate=_string(
                item.get("target_predicate"),
                field=f"interpretations[{index}].target_predicate",
            ),
            owner_predicate=_optional_string(
                item.get("owner_predicate"),
                field=f"interpretations[{index}].owner_predicate",
            ),
            count_unit=_enum(
                item.get("count_unit"),
                field=f"interpretations[{index}].count_unit",
                allowed=COUNT_UNITS,
            ),
            scope=TimeScope(
                start_seconds=start,
                end_seconds=end,
                boundary_policy=_enum(
                    scope_payload.get("boundary_policy"),
                    field=f"interpretations[{index}].scope.boundary_policy",
                    allowed=BOUNDARY_POLICIES,
                ),
            ),
            identity_policy=_enum(
                item.get("identity_policy"),
                field=f"interpretations[{index}].identity_policy",
                allowed=IDENTITY_POLICIES,
            ),
            cross_shot_unknown=_enum(
                item.get("cross_shot_unknown"),
                field=f"interpretations[{index}].cross_shot_unknown",
                allowed=CROSS_SHOT_POLICIES,
            ),
            visibility_policy=_enum(
                item.get("visibility_policy"),
                field=f"interpretations[{index}].visibility_policy",
                allowed=VISIBILITY_POLICIES,
            ),
            aggregation=_enum(
                item.get("aggregation"),
                field=f"interpretations[{index}].aggregation",
                allowed=AGGREGATIONS,
            ),
            evidence_obligations=obligations,
        )
        _validate_query_combination(interpretation)
        interpretations.append(interpretation)
    return CountQuery(query_id=query_id, interpretations=tuple(interpretations))


def _validate_query_combination(interpretation: QueryInterpretation) -> None:
    allowed: dict[tuple[str, str], frozenset[str]] = {
        ("entity", "physical_entity"): frozenset({"distinct"}),
        ("entity", "appearance"): frozenset({"cardinality"}),
        ("entity", "visible_instance"): frozenset({"cardinality", "max_concurrent"}),
        ("part", "part"): frozenset({"cardinality", "distinct", "max_concurrent"}),
        ("event", "event_onset"): frozenset({"transition_count"}),
        ("event", "active_event"): frozenset({"cardinality", "max_concurrent"}),
    }
    aggregations = allowed.get((interpretation.target_kind, interpretation.count_unit))
    if aggregations is None or interpretation.aggregation not in aggregations:
        raise ValueError(
            "incompatible count query: "
            f"{interpretation.target_kind}/{interpretation.count_unit}/"
            f"{interpretation.aggregation}"
        )
    if (
        interpretation.owner_predicate is not None
        and interpretation.target_kind != "part"
    ):
        raise ValueError(
            "incompatible count query: owner_predicate requires part target"
        )
    required_identity_policy = {
        "physical_entity": "physical",
        "appearance": "appearance_episode",
    }.get(interpretation.count_unit)
    if (
        required_identity_policy is not None
        and interpretation.identity_policy != required_identity_policy
    ):
        raise ValueError(
            "incompatible identity policy: "
            f"{interpretation.count_unit} requires {required_identity_policy}"
        )
    if (
        interpretation.count_unit == "event_onset"
        and interpretation.scope.boundary_policy != "starts_within"
    ):
        raise ValueError(
            "incompatible boundary policy: event_onset requires starts_within"
        )


def _parse_provenance(
    raw: object,
    *,
    field: str,
    duration_seconds: float,
    observation_start_seconds: float,
    observation_end_seconds: float,
) -> Provenance:
    payload = _mapping(raw, field=field)
    _exact_fields(payload, allowed=_PROVENANCE_FIELDS, field=field)
    timestamps = tuple(
        _number(value, field=f"{field}.timestamps_seconds")
        for value in _sequence(
            payload.get("timestamps_seconds"), field=f"{field}.timestamps_seconds"
        )
    )
    if any(value < 0 or value > duration_seconds for value in timestamps):
        raise ValueError(f"{field}.timestamps_seconds must be within video duration")
    if not timestamps or any(
        value < observation_start_seconds or value > observation_end_seconds
        for value in timestamps
    ):
        raise ValueError(
            f"{field}.timestamps_seconds must ground the observation interval"
        )
    return Provenance(
        source_kind=_string(payload.get("source_kind"), field=f"{field}.source_kind"),
        source_id=_string(payload.get("source_id"), field=f"{field}.source_id"),
        timestamps_seconds=timestamps,
    )


def _validate_node_shot(
    *,
    node_id: str,
    shot_id: str,
    start_seconds: float,
    end_seconds: float,
    shots: Mapping[str, Shot],
) -> None:
    shot = shots.get(shot_id)
    if shot is None:
        raise ValueError(f"node {node_id} references unknown shot {shot_id}")
    if start_seconds < shot.start_seconds or end_seconds > shot.end_seconds:
        raise ValueError(f"node {node_id} interval must lie within its shot")


def parse_evidence_graph(payload: Mapping[str, object]) -> EvidenceGraph:
    validate_label_blind(payload)
    _exact_fields(payload, allowed=_GRAPH_ROOT_FIELDS, field="root")
    _verify_embedded_hash(payload, field="graph")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"graph schema_version must be {SCHEMA_VERSION}")
    graph_id = _string(payload.get("graph_id"), field="graph_id")
    duration = _number(payload.get("duration_seconds"), field="duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")

    shots: list[Shot] = []
    seen_shot_ids: set[str] = set()
    for index, raw in enumerate(_sequence(payload.get("shots"), field="shots")):
        item = _mapping(raw, field=f"shots[{index}]")
        _exact_fields(item, allowed=_SHOT_FIELDS, field=f"shots[{index}]")
        shot_id = _string(item.get("shot_id"), field=f"shots[{index}].shot_id")
        if shot_id in seen_shot_ids:
            raise ValueError("shot IDs must be unique")
        seen_shot_ids.add(shot_id)
        start, end = _bounded_interval(
            item, field=f"shots[{index}]", duration_seconds=duration
        )
        shots.append(Shot(shot_id, start, end))
    if not shots:
        raise ValueError("shots must not be empty")
    for previous, current in pairwise(shots):
        if previous.start_seconds > current.start_seconds:
            raise ValueError("shots must be ordered")
        if previous.end_seconds > current.start_seconds:
            raise ValueError("shots must not overlap")
    shots_by_id = {shot.shot_id: shot for shot in shots}

    entities: list[EntityObservation] = []
    parts: list[PartObservation] = []
    events: list[EventObservation] = []
    node_ids: set[str] = set()

    for index, raw in enumerate(_sequence(payload.get("entities"), field="entities")):
        item = _mapping(raw, field=f"entities[{index}]")
        _exact_fields(item, allowed=_ENTITY_FIELDS, field=f"entities[{index}]")
        node_id = _string(item.get("node_id"), field=f"entities[{index}].node_id")
        if node_id in node_ids:
            raise ValueError("node IDs must be unique")
        node_ids.add(node_id)
        start, end = _bounded_interval(
            item, field=f"entities[{index}]", duration_seconds=duration
        )
        shot_id = _string(item.get("shot_id"), field=f"entities[{index}].shot_id")
        _validate_node_shot(
            node_id=node_id,
            shot_id=shot_id,
            start_seconds=start,
            end_seconds=end,
            shots=shots_by_id,
        )
        entities.append(
            EntityObservation(
                node_id=node_id,
                predicate=_string(
                    item.get("predicate"), field=f"entities[{index}].predicate"
                ),
                start_seconds=start,
                end_seconds=end,
                shot_id=shot_id,
                visibility=_enum(
                    item.get("visibility"),
                    field=f"entities[{index}].visibility",
                    allowed=VISIBILITY_VALUES,
                ),
                provenance=_parse_provenance(
                    item.get("provenance"),
                    field=f"entities[{index}].provenance",
                    duration_seconds=duration,
                    observation_start_seconds=start,
                    observation_end_seconds=end,
                ),
            )
        )

    raw_parts = _sequence(payload.get("parts"), field="parts")
    for index, raw in enumerate(raw_parts):
        item = _mapping(raw, field=f"parts[{index}]")
        _exact_fields(item, allowed=_PART_FIELDS, field=f"parts[{index}]")
        node_id = _string(item.get("node_id"), field=f"parts[{index}].node_id")
        if node_id in node_ids:
            raise ValueError("node IDs must be unique")
        node_ids.add(node_id)
        start, end = _bounded_interval(
            item, field=f"parts[{index}]", duration_seconds=duration
        )
        shot_id = _string(item.get("shot_id"), field=f"parts[{index}].shot_id")
        _validate_node_shot(
            node_id=node_id,
            shot_id=shot_id,
            start_seconds=start,
            end_seconds=end,
            shots=shots_by_id,
        )
        parts.append(
            PartObservation(
                node_id=node_id,
                predicate=_string(
                    item.get("predicate"), field=f"parts[{index}].predicate"
                ),
                owner_node_id=_string(
                    item.get("owner_node_id"),
                    field=f"parts[{index}].owner_node_id",
                ),
                start_seconds=start,
                end_seconds=end,
                shot_id=shot_id,
                visibility=_enum(
                    item.get("visibility"),
                    field=f"parts[{index}].visibility",
                    allowed=VISIBILITY_VALUES,
                ),
                provenance=_parse_provenance(
                    item.get("provenance"),
                    field=f"parts[{index}].provenance",
                    duration_seconds=duration,
                    observation_start_seconds=start,
                    observation_end_seconds=end,
                ),
            )
        )

    for index, raw in enumerate(_sequence(payload.get("events"), field="events")):
        item = _mapping(raw, field=f"events[{index}]")
        _exact_fields(item, allowed=_EVENT_FIELDS, field=f"events[{index}]")
        node_id = _string(item.get("node_id"), field=f"events[{index}].node_id")
        if node_id in node_ids:
            raise ValueError("node IDs must be unique")
        node_ids.add(node_id)
        start, end = _bounded_interval(
            item, field=f"events[{index}]", duration_seconds=duration
        )
        shot_id = _string(item.get("shot_id"), field=f"events[{index}].shot_id")
        _validate_node_shot(
            node_id=node_id,
            shot_id=shot_id,
            start_seconds=start,
            end_seconds=end,
            shots=shots_by_id,
        )
        participants = tuple(
            _string(value, field=f"events[{index}].participant_node_ids")
            for value in _sequence(
                item.get("participant_node_ids"),
                field=f"events[{index}].participant_node_ids",
            )
        )
        if len(participants) != len(set(participants)):
            raise ValueError("event participant node IDs must be unique")
        events.append(
            EventObservation(
                node_id=node_id,
                predicate=_string(
                    item.get("predicate"), field=f"events[{index}].predicate"
                ),
                start_seconds=start,
                end_seconds=end,
                shot_id=shot_id,
                onset_state=_enum(
                    item.get("onset_state"),
                    field=f"events[{index}].onset_state",
                    allowed=ONSET_STATES,
                ),
                completion_state=_enum(
                    item.get("completion_state"),
                    field=f"events[{index}].completion_state",
                    allowed=COMPLETION_STATES,
                ),
                participant_node_ids=participants,
                provenance=_parse_provenance(
                    item.get("provenance"),
                    field=f"events[{index}].provenance",
                    duration_seconds=duration,
                    observation_start_seconds=start,
                    observation_end_seconds=end,
                ),
            )
        )

    entity_or_part_ids = {node.node_id for node in (*entities, *parts)}
    identity_signature_by_id = {
        **{node.node_id: ("entity", node.predicate) for node in entities},
        **{node.node_id: ("part", node.predicate) for node in parts},
    }
    all_node_ids = entity_or_part_ids | {node.node_id for node in events}
    for part in parts:
        if part.owner_node_id not in entity_or_part_ids:
            raise ValueError(f"part {part.node_id} references unknown owner node")
    for event in events:
        missing = sorted(set(event.participant_node_ids) - entity_or_part_ids)
        if missing:
            raise ValueError(f"event {event.node_id} has unknown participant nodes")
    _reject_part_cycles(parts)

    relations: list[IdentityRelation] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(
        _sequence(payload.get("identity_relations"), field="identity_relations")
    ):
        item = _mapping(raw, field=f"identity_relations[{index}]")
        _exact_fields(
            item, allowed=_RELATION_FIELDS, field=f"identity_relations[{index}]"
        )
        left = _string(
            item.get("left_node_id"),
            field=f"identity_relations[{index}].left_node_id",
        )
        right = _string(
            item.get("right_node_id"),
            field=f"identity_relations[{index}].right_node_id",
        )
        if left == right:
            raise ValueError("identity relation must join distinct nodes")
        if left not in entity_or_part_ids or right not in entity_or_part_ids:
            raise ValueError("identity relation references unknown node")
        if identity_signature_by_id[left] != identity_signature_by_id[right]:
            raise ValueError(
                "identity relation endpoints must have the same node kind and predicate"
            )
        pair_key = tuple(sorted((left, right)))
        if pair_key in seen_pairs:
            raise ValueError("identity relation pairs must be unique")
        seen_pairs.add(pair_key)
        relations.append(
            IdentityRelation(
                left_node_id=left,
                right_node_id=right,
                relation_state=_enum(
                    item.get("relation_state"),
                    field=f"identity_relations[{index}].relation_state",
                    allowed=RELATION_STATES,
                ),
            )
        )
    identity_classes = _UnionFind(tuple(sorted(entity_or_part_ids)))
    for relation in relations:
        if relation.relation_state == "confirmed_same":
            identity_classes.union(relation.left_node_id, relation.right_node_id)
    if any(
        relation.relation_state == "confirmed_different"
        and identity_classes.find(relation.left_node_id)
        == identity_classes.find(relation.right_node_id)
        for relation in relations
    ):
        raise ValueError("identity relations are contradictory")

    coverage: list[CoverageClaim] = []
    seen_claim_ids: set[str] = set()
    for index, raw in enumerate(_sequence(payload.get("coverage"), field="coverage")):
        item = _mapping(raw, field=f"coverage[{index}]")
        _exact_fields(item, allowed=_COVERAGE_FIELDS, field=f"coverage[{index}]")
        claim_id = _string(item.get("claim_id"), field=f"coverage[{index}].claim_id")
        if claim_id in seen_claim_ids:
            raise ValueError("coverage claim IDs must be unique")
        seen_claim_ids.add(claim_id)
        start, end = _bounded_interval(
            item, field=f"coverage[{index}]", duration_seconds=duration
        )
        resolution = _number(
            item.get("temporal_resolution_seconds"),
            field=f"coverage[{index}].temporal_resolution_seconds",
        )
        if resolution <= 0:
            raise ValueError("coverage temporal resolution must be positive")
        coverage.append(
            CoverageClaim(
                claim_id=claim_id,
                obligation=_enum(
                    item.get("obligation"),
                    field=f"coverage[{index}].obligation",
                    allowed=EVIDENCE_OBLIGATIONS,
                ),
                target_kind=_enum(
                    item.get("target_kind"),
                    field=f"coverage[{index}].target_kind",
                    allowed=TARGET_KINDS,
                ),
                target_predicate=_string(
                    item.get("target_predicate"),
                    field=f"coverage[{index}].target_predicate",
                ),
                start_seconds=start,
                end_seconds=end,
                spatial_scope=_string(
                    item.get("spatial_scope"),
                    field=f"coverage[{index}].spatial_scope",
                ),
                temporal_resolution_seconds=resolution,
                coverage_state=_enum(
                    item.get("coverage_state"),
                    field=f"coverage[{index}].coverage_state",
                    allowed=COVERAGE_STATES,
                ),
            )
        )

    if len(all_node_ids) != len(entities) + len(parts) + len(events):
        raise ValueError("node IDs must be unique")
    return EvidenceGraph(
        graph_id=graph_id,
        duration_seconds=duration,
        shots=tuple(shots),
        entities=tuple(entities),
        parts=tuple(parts),
        events=tuple(events),
        identity_relations=tuple(relations),
        coverage=tuple(coverage),
    )


def _reject_part_cycles(parts: Sequence[PartObservation]) -> None:
    owner_by_part = {part.node_id: part.owner_node_id for part in parts}
    for part_id in owner_by_part:
        seen: set[str] = set()
        current = part_id
        while current in owner_by_part:
            if current in seen:
                raise ValueError("part ownership must not contain a cycle")
            seen.add(current)
            current = owner_by_part[current]


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_compatible(item) for item in value]
    return value


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def query_to_dict(query: CountQuery) -> dict[str, object]:
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": query.query_id,
            "interpretations": _json_compatible(asdict(query)["interpretations"]),
        }
    )


def graph_to_dict(graph: EvidenceGraph) -> dict[str, object]:
    payload = _json_compatible(asdict(graph))
    assert isinstance(payload, dict)
    return _with_hash({"schema_version": SCHEMA_VERSION, **payload})


def _matches_scope(
    *, start_seconds: float, end_seconds: float, scope: TimeScope
) -> bool:
    if scope.boundary_policy == "starts_within":
        return scope.start_seconds <= start_seconds <= scope.end_seconds
    if scope.boundary_policy == "completes_within":
        return scope.start_seconds <= end_seconds <= scope.end_seconds
    return start_seconds < scope.end_seconds and end_seconds > scope.start_seconds


def _matches_visibility(observed: str, required: str) -> bool:
    levels = {"visible": 0, "partially_visible": 1, "amodal": 2}
    return levels[observed] <= levels[required]


def _event_scope_certainty(node: EventObservation, scope: TimeScope) -> str:
    if not _matches_scope(
        start_seconds=node.start_seconds,
        end_seconds=node.end_seconds,
        scope=scope,
    ):
        return "excluded"
    if scope.boundary_policy == "starts_within":
        if node.onset_state == "left_censored":
            return "excluded"
        if node.onset_state == "unknown":
            return "possible"
    elif scope.boundary_policy == "completes_within":
        if node.completion_state == "right_censored":
            return "excluded"
        if node.completion_state == "unknown":
            return "possible"
    return "definite"


def _satisfied_obligations(
    interpretation: QueryInterpretation,
    graph: EvidenceGraph,
    obligations: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    failed: list[str] = []
    selected_claim_ids: set[str] = set()
    for obligation in obligations:
        matching = [
            claim
            for claim in graph.coverage
            if claim.obligation == obligation
            and claim.target_kind == interpretation.target_kind
            and claim.target_predicate == interpretation.target_predicate
            and claim.start_seconds <= interpretation.scope.start_seconds
            and claim.end_seconds >= interpretation.scope.end_seconds
            and claim.coverage_state == "complete"
        ]
        if matching:
            satisfied.append(obligation)
            selected_claim_ids.update(claim.claim_id for claim in matching)
        else:
            failed.append(obligation)
    return tuple(satisfied), tuple(failed), tuple(sorted(selected_claim_ids))


def _selected_identity_relation_keys(
    nodes: Sequence[EntityObservation | PartObservation],
    graph: EvidenceGraph,
    *,
    identity_policy: str,
    identity_universe: Sequence[EntityObservation | PartObservation],
) -> tuple[str, ...]:
    relevant = {node.node_id for node in identity_universe}
    shot_by_node_id = {node.node_id: node.shot_id for node in identity_universe}
    applicable: list[IdentityRelation] = []
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in relevant}
    for relation in graph.identity_relations:
        if (
            relation.left_node_id not in relevant
            or relation.right_node_id not in relevant
            or (
                identity_policy != "physical"
                and shot_by_node_id[relation.left_node_id]
                != shot_by_node_id[relation.right_node_id]
            )
        ):
            continue
        applicable.append(relation)
        if relation.relation_state != "confirmed_different":
            adjacency[relation.left_node_id].add(relation.right_node_id)
            adjacency[relation.right_node_id].add(relation.left_node_id)
    reachable = {node.node_id for node in nodes}
    frontier = list(reachable)
    while frontier:
        node_id = frontier.pop()
        for neighbor in adjacency[node_id] - reachable:
            reachable.add(neighbor)
            frontier.append(neighbor)
    keys: list[str] = []
    for relation in applicable:
        if (
            relation.left_node_id not in reachable
            or relation.right_node_id not in reachable
        ):
            continue
        left, right = sorted((relation.left_node_id, relation.right_node_id))
        keys.append(f"{left}|{right}|{relation.relation_state}")
    return tuple(sorted(keys))


def _max_concurrent(
    nodes: Sequence[EntityObservation | PartObservation | EventObservation],
    scope: TimeScope,
    *,
    class_ids: Mapping[str, str] | None = None,
) -> tuple[int, tuple[str, ...]]:
    events: list[tuple[float, int, str]] = []
    for node in nodes:
        class_id = class_ids[node.node_id] if class_ids is not None else node.node_id
        start = max(node.start_seconds, scope.start_seconds)
        end = min(node.end_seconds, scope.end_seconds)
        if start >= end:
            continue
        events.append((start, 1, class_id))
        events.append((end, -1, class_id))
    active_counts: dict[str, int] = {}
    peak: tuple[str, ...] = ()
    for _, delta, class_id in sorted(events, key=lambda item: (item[0], item[1])):
        if delta < 0:
            remaining = active_counts[class_id] - 1
            if remaining:
                active_counts[class_id] = remaining
            else:
                del active_counts[class_id]
        else:
            active_counts[class_id] = active_counts.get(class_id, 0) + 1
            candidate = tuple(sorted(active_counts))
            if len(candidate) > len(peak):
                peak = candidate
    return len(peak), peak


class _UnionFind:
    def __init__(self, node_ids: Sequence[str]) -> None:
        self._parent = {node_id: node_id for node_id in node_ids}

    def find(self, node_id: str) -> str:
        parent = self._parent[node_id]
        if parent != node_id:
            self._parent[node_id] = self.find(parent)
        return self._parent[node_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self._parent[merge] = keep

    def class_count(self) -> int:
        return len({self.find(node_id) for node_id in self._parent})

    def class_ids(self) -> dict[str, str]:
        return {node_id: self.find(node_id) for node_id in self._parent}


def _confirmed_identity_classes(
    nodes: Sequence[EntityObservation | PartObservation],
    graph: EvidenceGraph,
    *,
    identity_policy: str,
    identity_universe: Sequence[EntityObservation | PartObservation] | None = None,
) -> dict[str, str]:
    universe = tuple(identity_universe if identity_universe is not None else nodes)
    classes = _UnionFind(tuple(node.node_id for node in universe))
    relevant = {node.node_id for node in universe}
    shot_by_node_id = {node.node_id: node.shot_id for node in universe}
    for relation in graph.identity_relations:
        if (
            relation.relation_state != "confirmed_same"
            or relation.left_node_id not in relevant
            or relation.right_node_id not in relevant
            or (
                identity_policy != "physical"
                and shot_by_node_id[relation.left_node_id]
                != shot_by_node_id[relation.right_node_id]
            )
        ):
            continue
        classes.union(relation.left_node_id, relation.right_node_id)
    class_ids = classes.class_ids()
    return {node.node_id: class_ids[node.node_id] for node in nodes}


def _distinct_bounds(
    nodes: Sequence[EntityObservation | PartObservation],
    graph: EvidenceGraph,
    *,
    identity_policy: str,
    unresolved_policy: str,
    identity_universe: Sequence[EntityObservation | PartObservation] | None = None,
) -> tuple[int, int, bool]:
    universe = tuple(identity_universe if identity_universe is not None else nodes)
    universe_ids = tuple(node.node_id for node in universe)
    selected_ids = {node.node_id for node in nodes}
    relevant = set(universe_ids)
    shot_by_node_id = {node.node_id: node.shot_id for node in universe}
    upper_classes = _UnionFind(universe_ids)
    possible_classes = _UnionFind(universe_ids)
    saw_unresolved = False
    relevant_relations: list[IdentityRelation] = []
    for relation in graph.identity_relations:
        if (
            relation.left_node_id not in relevant
            or relation.right_node_id not in relevant
            or (
                identity_policy != "physical"
                and shot_by_node_id[relation.left_node_id]
                != shot_by_node_id[relation.right_node_id]
            )
        ):
            continue
        relevant_relations.append(relation)
        if relation.relation_state == "confirmed_same":
            upper_classes.union(relation.left_node_id, relation.right_node_id)
            possible_classes.union(relation.left_node_id, relation.right_node_id)
        elif relation.relation_state == "unresolved":
            saw_unresolved = True
            if unresolved_policy == "abstain":
                possible_classes.union(relation.left_node_id, relation.right_node_id)
    upper_roots = upper_classes.class_ids()
    selected_upper_roots = {upper_roots[node_id] for node_id in selected_ids}
    upper = len(selected_upper_roots)
    if unresolved_policy == "abstain":
        possible_roots = possible_classes.class_ids()
        selected_possible_roots = {possible_roots[node_id] for node_id in selected_ids}
        conflicted_roots = {
            possible_roots[relation.left_node_id]
            for relation in relevant_relations
            if relation.relation_state == "confirmed_different"
            and upper_roots[relation.left_node_id] in selected_upper_roots
            and upper_roots[relation.right_node_id] in selected_upper_roots
            and possible_roots[relation.left_node_id]
            == possible_roots[relation.right_node_id]
        }
        lower = len(selected_possible_roots) + len(conflicted_roots)
    else:
        lower = upper
    return lower, upper, saw_unresolved and lower != upper


def _max_concurrent_identity_bounds(
    nodes: Sequence[EntityObservation | PartObservation],
    graph: EvidenceGraph,
    *,
    scope: TimeScope,
    identity_policy: str,
    unresolved_policy: str,
    identity_universe: Sequence[EntityObservation | PartObservation],
) -> tuple[int, int, tuple[str, ...], bool]:
    endpoints = sorted(
        {
            scope.start_seconds,
            scope.end_seconds,
            *(
                max(node.start_seconds, scope.start_seconds)
                for node in nodes
                if node.end_seconds > scope.start_seconds
                and node.start_seconds < scope.end_seconds
            ),
            *(
                min(node.end_seconds, scope.end_seconds)
                for node in nodes
                if node.end_seconds > scope.start_seconds
                and node.start_seconds < scope.end_seconds
            ),
        }
    )
    lower_peak = 0
    upper_peak = 0
    selected: tuple[str, ...] = ()
    for start, end in pairwise(endpoints):
        if start >= end:
            continue
        active = tuple(
            node
            for node in nodes
            if node.start_seconds < end and node.end_seconds > start
        )
        lower, upper, _ = _distinct_bounds(
            active,
            graph,
            identity_policy=identity_policy,
            unresolved_policy=unresolved_policy,
            identity_universe=identity_universe,
        )
        lower_peak = max(lower_peak, lower)
        if upper > upper_peak:
            upper_peak = upper
            class_ids = _confirmed_identity_classes(
                active,
                graph,
                identity_policy=identity_policy,
                identity_universe=identity_universe,
            )
            selected = tuple(sorted(set(class_ids.values())))
    return lower_peak, upper_peak, selected, lower_peak != upper_peak


def _execute_interpretation(
    interpretation: QueryInterpretation, graph: EvidenceGraph
) -> InterpretationTrace:
    event_nodes: tuple[EventObservation, ...] = ()
    possible_event_nodes: tuple[EventObservation, ...] = ()
    if interpretation.target_kind == "entity":
        nodes: Sequence[EntityObservation | PartObservation] = graph.entities
    elif interpretation.target_kind == "part":
        nodes = graph.parts
    else:
        nodes = ()
        matching_events = tuple(
            (node, _event_scope_certainty(node, interpretation.scope))
            for node in graph.events
            if node.predicate == interpretation.target_predicate
        )
        event_nodes = tuple(
            node for node, certainty in matching_events if certainty == "definite"
        )
        possible_event_nodes = tuple(
            node for node, certainty in matching_events if certainty != "excluded"
        )
    qualifying = tuple(
        node
        for node in nodes
        if node.predicate == interpretation.target_predicate
        and _matches_visibility(node.visibility, interpretation.visibility_policy)
        and _matches_scope(
            start_seconds=node.start_seconds,
            end_seconds=node.end_seconds,
            scope=interpretation.scope,
        )
    )
    if interpretation.target_kind == "part" and interpretation.owner_predicate:
        predicates_by_id = {
            node.node_id: node.predicate for node in (*graph.entities, *graph.parts)
        }
        qualifying = tuple(
            node
            for node in qualifying
            if isinstance(node, PartObservation)
            and predicates_by_id[node.owner_node_id] == interpretation.owner_predicate
        )
    identity_universe = tuple(
        node
        for node in nodes
        if node.predicate == interpretation.target_predicate
        and (
            interpretation.target_kind != "part"
            or interpretation.owner_predicate is None
            or (
                isinstance(node, PartObservation)
                and predicates_by_id[node.owner_node_id]
                == interpretation.owner_predicate
            )
        )
    )
    counted_scope_nodes: Sequence[
        EntityObservation | PartObservation | EventObservation
    ] = possible_event_nodes if interpretation.target_kind == "event" else qualifying
    obligations = list(interpretation.evidence_obligations)
    derived_obligations: list[str] = []
    if (
        interpretation.identity_policy == "physical"
        and len({node.shot_id for node in counted_scope_nodes}) > 1
    ):
        derived_obligations.append("cross_shot_identity")
    if interpretation.target_kind == "part":
        derived_obligations.append("part_visibility")
    if interpretation.target_kind == "event" and (
        interpretation.count_unit == "event_onset"
        or interpretation.scope.boundary_policy != "overlaps"
    ):
        derived_obligations.append("event_boundary_state")
    for obligation in derived_obligations:
        if obligation not in obligations:
            obligations.append(obligation)
    satisfied, failed, selected_coverage_claim_ids = _satisfied_obligations(
        interpretation, graph, obligations
    )
    selected_identity_relation_keys = (
        _selected_identity_relation_keys(
            qualifying,
            graph,
            identity_policy=interpretation.identity_policy,
            identity_universe=identity_universe,
        )
        if interpretation.target_kind != "event"
        else ()
    )
    selected = tuple(sorted(node.node_id for node in qualifying))
    unresolved_identity = False
    unresolved_boundary = False
    if interpretation.aggregation == "max_concurrent":
        concurrent_nodes: Sequence[
            EntityObservation | PartObservation | EventObservation
        ] = event_nodes if interpretation.target_kind == "event" else qualifying
        if interpretation.target_kind == "event":
            lower, selected = _max_concurrent(concurrent_nodes, interpretation.scope)
            upper, _ = _max_concurrent(possible_event_nodes, interpretation.scope)
            unresolved_boundary = lower != upper
        else:
            lower, upper, selected, unresolved_identity = (
                _max_concurrent_identity_bounds(
                    qualifying,
                    graph,
                    scope=interpretation.scope,
                    identity_policy=interpretation.identity_policy,
                    unresolved_policy=interpretation.cross_shot_unknown,
                    identity_universe=identity_universe,
                )
            )
    elif interpretation.aggregation == "cardinality":
        counted_nodes: Sequence[
            EntityObservation | PartObservation | EventObservation
        ] = event_nodes if interpretation.target_kind == "event" else qualifying
        selected = tuple(sorted(node.node_id for node in counted_nodes))
        if interpretation.target_kind == "event":
            lower = len(counted_nodes)
            upper = len(possible_event_nodes)
            unresolved_boundary = lower != upper
        else:
            lower, upper, unresolved_identity = _distinct_bounds(
                qualifying,
                graph,
                identity_policy=interpretation.identity_policy,
                unresolved_policy=interpretation.cross_shot_unknown,
                identity_universe=identity_universe,
            )
    elif interpretation.aggregation == "distinct":
        lower, upper, unresolved_identity = _distinct_bounds(
            qualifying,
            graph,
            identity_policy=interpretation.identity_policy,
            unresolved_policy=interpretation.cross_shot_unknown,
            identity_universe=identity_universe,
        )
    elif interpretation.aggregation == "transition_count":
        if interpretation.target_kind != "event":
            raise ValueError("transition_count requires an event target")
        selected = tuple(sorted(node.node_id for node in event_nodes))
        lower = len(event_nodes)
        upper = len(possible_event_nodes)
        unresolved_boundary = lower != upper
    else:
        raise NotImplementedError(
            f"aggregation is not implemented: {interpretation.aggregation}"
        )
    reasons = tuple(
        reason
        for reason, applies in (
            ("incomplete_coverage", bool(failed)),
            ("unresolved_identity", unresolved_identity),
            ("unresolved_boundary_state", unresolved_boundary),
        )
        if applies
    )
    exact = not reasons and lower == upper
    return InterpretationTrace(
        interpretation_id=interpretation.interpretation_id,
        outcome="exact" if exact else "abstain",
        lower_bound=lower,
        upper_bound=None if failed else upper,
        exact_count=lower if exact else None,
        selected_node_ids=selected,
        selected_identity_relation_keys=selected_identity_relation_keys,
        selected_coverage_claim_ids=selected_coverage_claim_ids,
        satisfied_obligations=satisfied,
        failed_obligations=failed,
        abstention_reasons=reasons,
    )


def count_video(query: CountQuery, graph: EvidenceGraph) -> ProofState:
    if any(
        interpretation.scope.end_seconds > graph.duration_seconds
        for interpretation in query.interpretations
    ):
        raise ValueError("query scope must be within graph duration")
    traces = tuple(
        _execute_interpretation(interpretation, graph)
        for interpretation in query.interpretations
    )
    exact_counts = {trace.exact_count for trace in traces if trace.outcome == "exact"}
    all_exact = all(trace.outcome == "exact" for trace in traces)
    if all_exact and len(exact_counts) == 1:
        exact_count = next(iter(exact_counts))
        assert exact_count is not None
        return ProofState(
            query_id=query.query_id,
            graph_id=graph.graph_id,
            outcome="exact",
            lower_bound=exact_count,
            upper_bound=exact_count,
            exact_count=exact_count,
            traces=traces,
            abstention_reasons=(),
        )
    reasons = tuple(
        sorted(
            {reason for trace in traces for reason in trace.abstention_reasons}
            | ({"ambiguous_interpretation"} if len(exact_counts) > 1 else set())
        )
    )
    return ProofState(
        query_id=query.query_id,
        graph_id=graph.graph_id,
        outcome="abstain",
        lower_bound=min(trace.lower_bound for trace in traces),
        upper_bound=(
            None
            if any(trace.upper_bound is None for trace in traces)
            else max(
                trace.upper_bound for trace in traces if trace.upper_bound is not None
            )
        ),
        exact_count=None,
        traces=traces,
        abstention_reasons=reasons,
    )


def proof_to_dict(proof: ProofState) -> dict[str, object]:
    payload = _json_compatible(asdict(proof))
    assert isinstance(payload, dict)
    return _with_hash({"schema_version": PROOF_SCHEMA_VERSION, **payload})
