from __future__ import annotations

from copy import deepcopy

import pytest

from trec_video_2026.vqa_count_query import (
    count_video,
    graph_to_dict,
    parse_count_query,
    parse_evidence_graph,
    proof_to_dict,
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


def test_max_concurrent_counts_peak_overlap_across_edit_boundaries() -> None:
    query_payload = valid_query_payload()
    interpretation = query_payload["interpretations"][0]
    interpretation.update(
        count_unit="visible_instance",
        aggregation="max_concurrent",
        identity_policy="within_shot",
        cross_shot_unknown="separate",
        evidence_obligations=["target_coverage"],
    )
    graph_payload = valid_graph_payload()
    graph_payload["identity_relations"] = []
    graph_payload["entities"] = [
        {
            "node_id": node_id,
            "predicate": "person",
            "start_seconds": start,
            "end_seconds": end,
            "shot_id": shot_id,
            "visibility": "visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": f"source-{node_id}",
                "timestamps_seconds": [start, end],
            },
        }
        for node_id, start, end, shot_id in (
            ("person-a", 0.0, 5.0, "shot-1"),
            ("person-b", 0.0, 5.0, "shot-1"),
            ("person-c", 5.0, 10.0, "shot-2"),
            ("person-d", 5.0, 10.0, "shot-2"),
        )
    ]

    proof = count_video(
        parse_count_query(query_payload), parse_evidence_graph(graph_payload)
    )

    assert proof.outcome == "exact"
    assert (proof.lower_bound, proof.upper_bound, proof.exact_count) == (2, 2, 2)
    assert proof.traces[0].selected_node_ids == ("person-a", "person-b")


def _add_complete_identity_coverage(graph: dict[str, object]) -> None:
    graph["coverage"].append(
        {
            "claim_id": "coverage-identity",
            "obligation": "cross_shot_identity",
            "target_kind": "entity",
            "target_predicate": "person",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "spatial_scope": "full_frame",
            "temporal_resolution_seconds": 1.0,
            "coverage_state": "complete",
        }
    )


@pytest.mark.parametrize(
    ("relation_state", "want"),
    [("confirmed_same", 1), ("confirmed_different", 2)],
)
def test_distinct_physical_count_respects_confirmed_identity(
    relation_state: str, want: int
) -> None:
    graph_payload = valid_graph_payload()
    graph_payload["identity_relations"][0]["relation_state"] = relation_state
    _add_complete_identity_coverage(graph_payload)

    proof = count_video(
        parse_count_query(valid_query_payload()),
        parse_evidence_graph(graph_payload),
    )

    assert proof.outcome == "exact"
    assert proof.exact_count == want


def test_unresolved_physical_identity_produces_count_bounds() -> None:
    graph_payload = valid_graph_payload()
    _add_complete_identity_coverage(graph_payload)

    proof = count_video(
        parse_count_query(valid_query_payload()),
        parse_evidence_graph(graph_payload),
    )

    assert proof.outcome == "abstain"
    assert (proof.lower_bound, proof.upper_bound) == (1, 2)
    assert proof.abstention_reasons == ("unresolved_identity",)


def test_appearance_policy_counts_cross_shot_episodes_separately() -> None:
    query_payload = valid_query_payload()
    query_payload["interpretations"][0].update(
        count_unit="appearance",
        aggregation="cardinality",
        identity_policy="appearance_episode",
        cross_shot_unknown="separate",
        evidence_obligations=["target_coverage"],
    )

    proof = count_video(
        parse_count_query(query_payload),
        parse_evidence_graph(valid_graph_payload()),
    )

    assert proof.outcome == "exact"
    assert proof.exact_count == 2


def test_graph_rejects_multiple_states_for_one_identity_pair() -> None:
    graph_payload = valid_graph_payload()
    graph_payload["identity_relations"].append(
        {
            "left_node_id": "entity-2",
            "right_node_id": "entity-1",
            "relation_state": "confirmed_same",
        }
    )

    with pytest.raises(ValueError, match="identity relation pairs must be unique"):
        parse_evidence_graph(graph_payload)


def _event_query_payload(*, active: bool = False) -> dict[str, object]:
    query = valid_query_payload()
    query["interpretations"][0].update(
        target_kind="event",
        target_predicate="pass",
        count_unit="active_event" if active else "event_onset",
        scope={
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "boundary_policy": "overlaps" if active else "starts_within",
        },
        identity_policy="appearance_episode",
        cross_shot_unknown="separate",
        aggregation="cardinality" if active else "transition_count",
        evidence_obligations=["target_coverage", "event_boundary_state"],
    )
    return query


def _event_graph_payload(*, incoming_onset: str = "left_censored") -> dict[str, object]:
    graph = valid_graph_payload()
    graph["events"] = [
        {
            "node_id": node_id,
            "predicate": "pass",
            "start_seconds": start,
            "end_seconds": end,
            "shot_id": shot_id,
            "onset_state": onset_state,
            "completion_state": "observed",
            "participant_node_ids": [],
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": f"source-{node_id}",
                "timestamps_seconds": [start, end],
            },
        }
        for node_id, start, end, shot_id, onset_state in (
            ("incoming-pass", 0.0, 2.0, "shot-1", incoming_onset),
            ("pass-1", 3.0, 4.0, "shot-1", "observed"),
            ("pass-2", 6.0, 7.0, "shot-2", "observed"),
        )
    ]
    graph["coverage"] = [
        {
            "claim_id": f"coverage-{obligation}",
            "obligation": obligation,
            "target_kind": "event",
            "target_predicate": "pass",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "spatial_scope": "full_frame",
            "temporal_resolution_seconds": 0.5,
            "coverage_state": "complete",
        }
        for obligation in ("target_coverage", "event_boundary_state")
    ]
    return graph


def test_transition_count_excludes_event_already_active_at_clip_start() -> None:
    proof = count_video(
        parse_count_query(_event_query_payload()),
        parse_evidence_graph(_event_graph_payload()),
    )

    assert proof.outcome == "exact"
    assert proof.exact_count == 2
    assert proof.traces[0].selected_node_ids == ("pass-1", "pass-2")


def test_active_event_count_includes_left_censored_overlap() -> None:
    proof = count_video(
        parse_count_query(_event_query_payload(active=True)),
        parse_evidence_graph(_event_graph_payload()),
    )

    assert proof.outcome == "exact"
    assert proof.exact_count == 3


def test_unknown_event_onset_produces_boundary_bounds() -> None:
    proof = count_video(
        parse_count_query(_event_query_payload()),
        parse_evidence_graph(_event_graph_payload(incoming_onset="unknown")),
    )

    assert proof.outcome == "abstain"
    assert (proof.lower_bound, proof.upper_bound) == (2, 3)
    assert proof.abstention_reasons == ("unresolved_boundary_state",)


def test_complete_target_coverage_permits_exact_zero() -> None:
    query_payload = valid_query_payload()
    query_payload["interpretations"][0]["evidence_obligations"] = ["target_coverage"]
    graph_payload = valid_graph_payload()
    graph_payload["entities"] = []
    graph_payload["identity_relations"] = []

    proof = count_video(
        parse_count_query(query_payload), parse_evidence_graph(graph_payload)
    )

    assert proof.outcome == "exact"
    assert proof.exact_count == 0


def test_missing_coverage_never_turns_missing_nodes_into_exact_absence() -> None:
    query_payload = valid_query_payload()
    query_payload["interpretations"][0]["evidence_obligations"] = ["target_coverage"]
    graph_payload = valid_graph_payload()
    graph_payload["identity_relations"][0]["relation_state"] = "confirmed_different"
    graph_payload["coverage"] = []

    proof = count_video(
        parse_count_query(query_payload), parse_evidence_graph(graph_payload)
    )

    assert proof.outcome == "abstain"
    assert (proof.lower_bound, proof.upper_bound) == (2, None)
    assert proof.abstention_reasons == ("incomplete_coverage",)


def _two_interpretation_query(*, agree: bool) -> dict[str, object]:
    query = valid_query_payload()
    first = query["interpretations"][0]
    first.update(
        interpretation_id="cardinality-reading",
        count_unit="appearance",
        aggregation="cardinality",
        identity_policy="appearance_episode",
        cross_shot_unknown="separate",
        evidence_obligations=["target_coverage"],
    )
    second = deepcopy(first)
    second["interpretation_id"] = "alternative-reading"
    if not agree:
        second.update(
            count_unit="visible_instance",
            aggregation="max_concurrent",
            identity_policy="within_shot",
        )
    query["interpretations"].append(second)
    return query


def test_disagreeing_exact_interpretations_force_abstention() -> None:
    proof = count_video(
        parse_count_query(_two_interpretation_query(agree=False)),
        parse_evidence_graph(valid_graph_payload()),
    )

    assert proof.outcome == "abstain"
    assert (proof.lower_bound, proof.upper_bound) == (1, 2)
    assert proof.abstention_reasons == ("ambiguous_interpretation",)


def test_agreeing_exact_interpretations_return_shared_count() -> None:
    proof = count_video(
        parse_count_query(_two_interpretation_query(agree=True)),
        parse_evidence_graph(valid_graph_payload()),
    )

    assert proof.outcome == "exact"
    assert (proof.lower_bound, proof.upper_bound, proof.exact_count) == (2, 2, 2)
    assert all(not trace.failed_obligations for trace in proof.traces)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(target_kind="entity", count_unit="part"),
        lambda item: item.update(
            target_kind="event",
            count_unit="event_onset",
            aggregation="max_concurrent",
        ),
        lambda item: item.update(
            target_kind="part",
            count_unit="physical_entity",
            aggregation="distinct",
        ),
    ],
)
def test_query_rejects_incompatible_unit_and_aggregation(mutation) -> None:
    query = valid_query_payload()
    mutation(query["interpretations"][0])

    with pytest.raises(ValueError, match="incompatible count query"):
        parse_count_query(query)


def test_part_query_filters_by_owner_predicate() -> None:
    query = valid_query_payload()
    query["interpretations"][0].update(
        target_kind="part",
        target_predicate="finger",
        owner_predicate="hand",
        count_unit="part",
        identity_policy="within_shot",
        cross_shot_unknown="separate",
        visibility_policy="partially_visible",
        aggregation="cardinality",
        evidence_obligations=["target_coverage"],
    )
    graph = valid_graph_payload()
    graph["identity_relations"] = []
    graph["entities"][0]["predicate"] = "hand"
    graph["parts"] = [
        {
            "node_id": "finger-1",
            "predicate": "finger",
            "owner_node_id": "entity-1",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "shot_id": "shot-1",
            "visibility": "partially_visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": "finger-source-1",
                "timestamps_seconds": [1.0],
            },
        },
        {
            "node_id": "finger-2",
            "predicate": "finger",
            "owner_node_id": "entity-2",
            "start_seconds": 6.0,
            "end_seconds": 7.0,
            "shot_id": "shot-2",
            "visibility": "visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": "finger-source-2",
                "timestamps_seconds": [6.0],
            },
        },
    ]
    graph["coverage"] = [
        {
            "claim_id": "coverage-parts",
            "obligation": "target_coverage",
            "target_kind": "part",
            "target_predicate": "finger",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "spatial_scope": "full_frame",
            "temporal_resolution_seconds": 0.5,
            "coverage_state": "complete",
        }
    ]

    proof = count_video(parse_count_query(query), parse_evidence_graph(graph))

    assert proof.outcome == "exact"
    assert proof.exact_count == 1
    assert proof.traces[0].selected_node_ids == ("finger-1",)


def test_graph_rejects_transitive_identity_contradiction() -> None:
    graph = valid_graph_payload()
    graph["entities"].append(
        {
            "node_id": "entity-3",
            "predicate": "person",
            "start_seconds": 7.0,
            "end_seconds": 8.0,
            "shot_id": "shot-2",
            "visibility": "visible",
            "provenance": {
                "source_kind": "manual_observation",
                "source_id": "observation-3",
                "timestamps_seconds": [7.0, 8.0],
            },
        }
    )
    graph["identity_relations"] = [
        {
            "left_node_id": "entity-1",
            "right_node_id": "entity-2",
            "relation_state": "confirmed_same",
        },
        {
            "left_node_id": "entity-2",
            "right_node_id": "entity-3",
            "relation_state": "confirmed_same",
        },
        {
            "left_node_id": "entity-1",
            "right_node_id": "entity-3",
            "relation_state": "confirmed_different",
        },
    ]

    with pytest.raises(ValueError, match="identity relations are contradictory"):
        parse_evidence_graph(graph)


def test_max_concurrent_active_events_counts_peak_overlap() -> None:
    query = _event_query_payload(active=True)
    query["interpretations"][0]["aggregation"] = "max_concurrent"
    graph = _event_graph_payload()
    graph["events"][1]["start_seconds"] = 1.0
    graph["events"][1]["end_seconds"] = 4.0
    graph["events"][1]["provenance"]["timestamps_seconds"] = [1.0, 4.0]

    proof = count_video(parse_count_query(query), parse_evidence_graph(graph))

    assert proof.outcome == "exact"
    assert proof.exact_count == 2
    assert proof.traces[0].selected_node_ids == ("incoming-pass", "pass-1")


def test_proof_serialization_is_canonical_and_self_hashed() -> None:
    proof = count_video(
        parse_count_query(_two_interpretation_query(agree=True)),
        parse_evidence_graph(valid_graph_payload()),
    )

    first = proof_to_dict(proof)
    second = proof_to_dict(proof)

    assert first == second
    assert len(first["complete_sha256"]) == 64
    assert first["exact_count"] == first["lower_bound"] == first["upper_bound"]


def test_count_algebra_invariants() -> None:
    cardinality_query = valid_query_payload()
    cardinality_query["interpretations"][0].update(
        count_unit="appearance",
        aggregation="cardinality",
        identity_policy="appearance_episode",
        cross_shot_unknown="separate",
        evidence_obligations=["target_coverage"],
    )
    base_graph = valid_graph_payload()
    base = count_video(
        parse_count_query(cardinality_query), parse_evidence_graph(base_graph)
    )
    with_observation = deepcopy(base_graph)
    added = deepcopy(with_observation["entities"][0])
    added.update(
        node_id="entity-3",
        start_seconds=2.0,
        end_seconds=3.0,
        provenance={
            "source_kind": "manual_observation",
            "source_id": "observation-3",
            "timestamps_seconds": [2.0, 3.0],
        },
    )
    with_observation["entities"].append(added)
    expanded = count_video(
        parse_count_query(cardinality_query),
        parse_evidence_graph(with_observation),
    )

    separate_graph = valid_graph_payload()
    separate_graph["identity_relations"][0]["relation_state"] = "confirmed_different"
    _add_complete_identity_coverage(separate_graph)
    merged_graph = deepcopy(separate_graph)
    merged_graph["identity_relations"][0]["relation_state"] = "confirmed_same"
    separate = count_video(
        parse_count_query(valid_query_payload()), parse_evidence_graph(separate_graph)
    )
    merged = count_video(
        parse_count_query(valid_query_payload()), parse_evidence_graph(merged_graph)
    )

    concurrency_query = deepcopy(cardinality_query)
    concurrency_query["interpretations"][0].update(
        count_unit="visible_instance",
        aggregation="max_concurrent",
        identity_policy="within_shot",
    )
    concurrent = count_video(
        parse_count_query(concurrency_query), parse_evidence_graph(with_observation)
    )

    proofs = (base, expanded, separate, merged, concurrent)
    assert all(
        proof.upper_bound is None or proof.lower_bound <= proof.upper_bound
        for proof in proofs
    )
    assert expanded.exact_count >= base.exact_count
    assert merged.exact_count <= separate.exact_count
    assert concurrent.exact_count <= len(with_observation["entities"])
