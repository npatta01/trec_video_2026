"""Conservative adapter from legacy atomic records to the shared count graph."""

from __future__ import annotations

from dataclasses import dataclass

from trec_video_2026.vqa_count_query import (
    EntityObservation,
    EvidenceGraph,
    Shot,
    parse_evidence_graph,
)
from trec_video_2026.vqa_counting_evidence import AtomicEvidence

_PLAN_BY_ATOMIC_UNIT = {
    "unique_entities_objects": frozenset({("entity", "appearance")}),
    "simultaneously_visible_instances": frozenset({("entity", "appearance")}),
    "appearances_across_scenes": frozenset({("entity", "appearance")}),
    "edited_shots": frozenset({("event", "active_event")}),
    "repeated_actions_transitions": frozenset({("event", "event_onset")}),
    "background_people": frozenset({("entity", "appearance")}),
    "parts_components": frozenset({("part", "part")}),
}


@dataclass(frozen=True)
class AtomicGraphPlan:
    graph_id: str
    duration_seconds: float
    target_kind: str
    target_predicate: str
    visibility: str
    node_interpretation: str
    shots: tuple[Shot, ...] = ()
    declare_complete_coverage: bool = False
    part_owner: EntityObservation | None = None


def _shots_payload(plan: AtomicGraphPlan) -> list[dict[str, object]]:
    shots = plan.shots or (Shot("full-video", 0.0, plan.duration_seconds),)
    return [
        {
            "shot_id": shot.shot_id,
            "start_seconds": shot.start_seconds,
            "end_seconds": shot.end_seconds,
        }
        for shot in shots
    ]


def _record_shot_id(
    *, start_seconds: float, end_seconds: float, plan: AtomicGraphPlan
) -> str:
    shots = plan.shots or (Shot("full-video", 0.0, plan.duration_seconds),)
    matches = [
        shot.shot_id
        for shot in shots
        if shot.start_seconds <= start_seconds and shot.end_seconds >= end_seconds
    ]
    if len(matches) != 1:
        raise ValueError("atomic record must lie within exactly one planned shot")
    return matches[0]


def graph_from_atomic_evidence(
    evidence: AtomicEvidence, plan: AtomicGraphPlan
) -> EvidenceGraph:
    """Map atomic records to graph nodes without asserting identity or coverage."""

    if plan.declare_complete_coverage:
        raise ValueError("atomic evidence cannot declare complete coverage")
    if plan.duration_seconds <= 0 or any(
        record.end_seconds > plan.duration_seconds for record in evidence.records
    ):
        raise ValueError("atomic evidence exceeds planned video duration")
    if any(
        float(exclusion["end_seconds"]) > plan.duration_seconds
        for exclusion in evidence.exclusions
    ):
        raise ValueError("atomic exclusion exceeds planned video duration")
    combination = (plan.target_kind, plan.node_interpretation)
    if combination not in _PLAN_BY_ATOMIC_UNIT[evidence.count_unit]:
        raise ValueError("incompatible atomic count unit and graph node interpretation")
    if plan.target_kind == "part" and plan.part_owner is None:
        raise ValueError("part atomic graph plan requires a part owner")
    if plan.target_kind != "part" and plan.part_owner is not None:
        raise ValueError("part owner is only valid for part atomic graph plans")

    entities: list[dict[str, object]] = []
    parts: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    if plan.part_owner is not None:
        owner = plan.part_owner
        entities.append(
            {
                "node_id": owner.node_id,
                "predicate": owner.predicate,
                "start_seconds": owner.start_seconds,
                "end_seconds": owner.end_seconds,
                "shot_id": owner.shot_id,
                "visibility": owner.visibility,
                "provenance": {
                    "source_kind": owner.provenance.source_kind,
                    "source_id": owner.provenance.source_id,
                    "timestamps_seconds": list(owner.provenance.timestamps_seconds),
                },
            }
        )
    for record in evidence.records:
        shot_id = _record_shot_id(
            start_seconds=record.start_seconds,
            end_seconds=record.end_seconds,
            plan=plan,
        )
        common = {
            "node_id": record.record_id,
            "predicate": plan.target_predicate,
            "start_seconds": record.start_seconds,
            "end_seconds": record.end_seconds,
            "shot_id": shot_id,
            "provenance": {
                "source_kind": "atomic_record",
                "source_id": record.record_id,
                "timestamps_seconds": [record.start_seconds, record.end_seconds],
            },
        }
        if plan.target_kind == "entity":
            entities.append({**common, "visibility": plan.visibility})
        elif plan.target_kind == "part":
            assert plan.part_owner is not None
            parts.append(
                {
                    **common,
                    "owner_node_id": plan.part_owner.node_id,
                    "visibility": plan.visibility,
                }
            )
        else:
            observed_boundaries = plan.node_interpretation == "event_onset"
            events.append(
                {
                    **common,
                    "onset_state": "observed" if observed_boundaries else "unknown",
                    "completion_state": (
                        "observed" if observed_boundaries else "unknown"
                    ),
                    "participant_node_ids": [],
                }
            )
    return parse_evidence_graph(
        {
            "schema_version": 1,
            "graph_id": plan.graph_id,
            "duration_seconds": plan.duration_seconds,
            "shots": _shots_payload(plan),
            "entities": entities,
            "parts": parts,
            "events": events,
            "identity_relations": [],
            "coverage": [],
        }
    )
