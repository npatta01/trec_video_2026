from __future__ import annotations

from copy import deepcopy

import pytest

from trec_video_2026.vqa_count_query import (
    graph_to_dict,
    parse_count_query,
    parse_evidence_graph,
    query_to_dict,
)


def valid_query_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "query_id": "query-1",
        "interpretations": [
            {
                "interpretation_id": "interpretation-1",
                "target_kind": "entity",
                "target_predicate": "person",
                "owner_predicate": None,
                "count_unit": "physical_entity",
                "scope": {
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "boundary_policy": "overlaps",
                },
                "identity_policy": "physical",
                "cross_shot_unknown": "abstain",
                "visibility_policy": "visible",
                "aggregation": "distinct",
                "evidence_obligations": [
                    "target_coverage",
                    "cross_shot_identity",
                ],
            }
        ],
    }


def valid_graph_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "graph_id": "graph-1",
        "duration_seconds": 10.0,
        "shots": [
            {"shot_id": "shot-1", "start_seconds": 0.0, "end_seconds": 5.0},
            {
                "shot_id": "shot-2",
                "start_seconds": 5.0,
                "end_seconds": 10.0,
            },
        ],
        "entities": [
            {
                "node_id": "entity-1",
                "predicate": "person",
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "shot_id": "shot-1",
                "visibility": "visible",
                "provenance": {
                    "source_kind": "manual_observation",
                    "source_id": "observation-1",
                    "timestamps_seconds": [1.0, 4.0],
                },
            },
            {
                "node_id": "entity-2",
                "predicate": "person",
                "start_seconds": 6.0,
                "end_seconds": 9.0,
                "shot_id": "shot-2",
                "visibility": "visible",
                "provenance": {
                    "source_kind": "manual_observation",
                    "source_id": "observation-2",
                    "timestamps_seconds": [6.0, 9.0],
                },
            },
        ],
        "parts": [],
        "events": [],
        "identity_relations": [
            {
                "left_node_id": "entity-1",
                "right_node_id": "entity-2",
                "relation_state": "unresolved",
            }
        ],
        "coverage": [
            {
                "claim_id": "coverage-1",
                "obligation": "target_coverage",
                "target_kind": "entity",
                "target_predicate": "person",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "spatial_scope": "full_frame",
                "temporal_resolution_seconds": 1.0,
                "coverage_state": "complete",
            }
        ],
    }


def test_parsers_accept_strict_label_blind_query_and_graph() -> None:
    query = parse_count_query(valid_query_payload())
    graph = parse_evidence_graph(valid_graph_payload())

    assert query.query_id == "query-1"
    assert query.interpretations[0].aggregation == "distinct"
    assert graph.graph_id == "graph-1"
    assert tuple(node.node_id for node in graph.entities) == (
        "entity-1",
        "entity-2",
    )


@pytest.mark.parametrize("target", ["query", "graph"])
def test_parsers_reject_unknown_root_fields(target: str) -> None:
    payload = valid_query_payload() if target == "query" else valid_graph_payload()
    payload["extra_field"] = True
    parser = parse_count_query if target == "query" else parse_evidence_graph

    with pytest.raises(ValueError, match="unknown root fields.*extra_field"):
        parser(payload)


def test_parsers_reject_nested_label_bearing_keys() -> None:
    query = valid_query_payload()
    query["interpretations"][0]["metadata"] = {"reference_answer": "two"}

    with pytest.raises(ValueError, match="forbidden label-bearing key.*reference"):
        parse_count_query(query)


def test_graph_rejects_duplicate_node_ids_across_node_kinds() -> None:
    graph = valid_graph_payload()
    graph["parts"].append(
        {
            "node_id": "entity-1",
            "predicate": "hand",
            "owner_node_id": "entity-2",
            "start_seconds": 6.0,
            "end_seconds": 7.0,
            "shot_id": "shot-2",
            "visibility": "visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": "part-1",
                "timestamps_seconds": [6.0],
            },
        }
    )

    with pytest.raises(ValueError, match="node IDs must be unique"):
        parse_evidence_graph(graph)


def test_graph_rejects_dangling_part_owner() -> None:
    graph = valid_graph_payload()
    graph["parts"].append(
        {
            "node_id": "part-1",
            "predicate": "finger",
            "owner_node_id": "missing-owner",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "shot_id": "shot-1",
            "visibility": "visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": "part-observation-1",
                "timestamps_seconds": [1.0],
            },
        }
    )

    with pytest.raises(ValueError, match="unknown owner node"):
        parse_evidence_graph(graph)


def test_graph_rejects_invalid_node_interval() -> None:
    graph = valid_graph_payload()
    graph["entities"][0]["start_seconds"] = 4.5
    graph["entities"][0]["end_seconds"] = 4.0

    with pytest.raises(ValueError, match="start_seconds must not exceed end_seconds"):
        parse_evidence_graph(graph)


def test_graph_rejects_overlapping_shots() -> None:
    graph = valid_graph_payload()
    graph["shots"][1]["start_seconds"] = 4.5

    with pytest.raises(ValueError, match="shots must not overlap"):
        parse_evidence_graph(graph)


def test_canonical_serialization_is_stable_and_mutation_sensitive() -> None:
    query_payload = valid_query_payload()
    graph_payload = valid_graph_payload()
    query_first = query_to_dict(parse_count_query(query_payload))
    query_second = query_to_dict(parse_count_query(deepcopy(query_payload)))
    graph_first = graph_to_dict(parse_evidence_graph(graph_payload))
    graph_second = graph_to_dict(parse_evidence_graph(deepcopy(graph_payload)))

    assert query_first == query_second
    assert graph_first == graph_second
    assert len(query_first["complete_sha256"]) == 64
    assert len(graph_first["complete_sha256"]) == 64

    query_payload["interpretations"][0]["target_predicate"] = "vehicle"
    graph_payload["entities"][0]["predicate"] = "vehicle"

    mutated_query = query_to_dict(parse_count_query(query_payload))
    mutated_graph = graph_to_dict(parse_evidence_graph(graph_payload))
    assert mutated_query["complete_sha256"] != query_first["complete_sha256"]
    assert mutated_graph["complete_sha256"] != graph_first["complete_sha256"]
