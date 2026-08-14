from __future__ import annotations

from copy import deepcopy

import pytest

from trec_video_2026.vqa_count_graph_adapter import (
    AtomicGraphPlan,
    graph_from_atomic_evidence,
)
from trec_video_2026.vqa_count_query import EntityObservation, Provenance
from trec_video_2026.vqa_counting_evidence import validate_atomic_evidence


def atomic_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "count_unit": "repeated_actions_transitions",
        "target": "passes",
        "scope": "complete video",
        "records": [
            {
                "record_id": "R1",
                "start_seconds": 1.25,
                "end_seconds": 1.75,
                "scene": "first pass",
                "rationale": "one completed pass",
            },
            {
                "record_id": "R2",
                "start_seconds": 4.0,
                "end_seconds": 4.5,
                "scene": "second pass",
                "rationale": "a distinct completed pass",
            },
        ],
        "exclusions": [],
        "uncertainty": "none",
    }


def test_atomic_event_records_map_without_changing_intervals_or_ids() -> None:
    evidence = validate_atomic_evidence(atomic_payload(), duration_seconds=8.0)
    plan = AtomicGraphPlan(
        graph_id="atomic-graph-1",
        duration_seconds=8.0,
        target_kind="event",
        target_predicate="pass",
        visibility="visible",
        node_interpretation="event_onset",
    )

    graph = graph_from_atomic_evidence(evidence, plan)

    assert tuple((node.start_seconds, node.end_seconds) for node in graph.events) == (
        (1.25, 1.75),
        (4.0, 4.5),
    )
    assert tuple(node.provenance.source_id for node in graph.events) == ("R1", "R2")
    assert (
        graph.entities
        == graph.parts
        == graph.identity_relations
        == graph.coverage
        == ()
    )


def test_atomic_entity_and_part_records_map_to_typed_nodes() -> None:
    entity_payload = atomic_payload()
    entity_payload["count_unit"] = "appearances_across_scenes"
    entity = graph_from_atomic_evidence(
        validate_atomic_evidence(entity_payload, duration_seconds=8.0),
        AtomicGraphPlan(
            graph_id="entity-graph",
            duration_seconds=8.0,
            target_kind="entity",
            target_predicate="person",
            visibility="partially_visible",
            node_interpretation="appearance",
        ),
    )
    assert tuple(node.node_id for node in entity.entities) == ("R1", "R2")
    assert all(node.visibility == "partially_visible" for node in entity.entities)

    part_payload = atomic_payload()
    part_payload["count_unit"] = "parts_components"
    owner = EntityObservation(
        node_id="owner-1",
        predicate="hand",
        start_seconds=0.0,
        end_seconds=8.0,
        shot_id="full-video",
        visibility="visible",
        provenance=Provenance("authoring_context", "owner-1", (0.0, 8.0)),
    )
    part = graph_from_atomic_evidence(
        validate_atomic_evidence(part_payload, duration_seconds=8.0),
        AtomicGraphPlan(
            graph_id="part-graph",
            duration_seconds=8.0,
            target_kind="part",
            target_predicate="finger",
            visibility="visible",
            node_interpretation="part",
            part_owner=owner,
        ),
    )
    assert tuple(node.owner_node_id for node in part.parts) == ("owner-1", "owner-1")
    assert tuple(node.node_id for node in part.entities) == ("owner-1",)


@pytest.mark.parametrize(
    ("plan_mutation", "message"),
    [
        (lambda values: values.update(duration_seconds=4.25), "duration"),
        (
            lambda values: values.update(
                target_kind="entity", node_interpretation="appearance"
            ),
            "incompatible",
        ),
        (
            lambda values: values.update(declare_complete_coverage=True),
            "cannot declare complete coverage",
        ),
    ],
)
def test_adapter_rejects_unsafe_plan(plan_mutation, message: str) -> None:
    evidence = validate_atomic_evidence(atomic_payload(), duration_seconds=8.0)
    values = {
        "graph_id": "atomic-graph-1",
        "duration_seconds": 8.0,
        "target_kind": "event",
        "target_predicate": "pass",
        "visibility": "visible",
        "node_interpretation": "event_onset",
    }
    plan_mutation(values)

    with pytest.raises(ValueError, match=message):
        graph_from_atomic_evidence(evidence, AtomicGraphPlan(**deepcopy(values)))
