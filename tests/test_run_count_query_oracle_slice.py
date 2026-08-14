from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_count_query_oracle_slice.py"
SPEC = importlib.util.spec_from_file_location("run_count_query_oracle_slice", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def query_payload(video_id: str) -> dict[str, object]:
    return _with_hash(
        {
            "schema_version": 1,
            "query_id": video_id,
            "interpretations": [
                {
                    "interpretation_id": "literal",
                    "target_kind": "entity",
                    "target_predicate": "book",
                    "owner_predicate": None,
                    "count_unit": "appearance",
                    "scope": {
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "boundary_policy": "overlaps",
                    },
                    "identity_policy": "appearance_episode",
                    "cross_shot_unknown": "separate",
                    "visibility_policy": "visible",
                    "aggregation": "cardinality",
                    "evidence_obligations": ["target_coverage"],
                }
            ],
        }
    )


def graph_payload(video_id: str) -> dict[str, object]:
    return _with_hash(
        {
            "schema_version": 1,
            "graph_id": video_id,
            "duration_seconds": 5.0,
            "shots": [{"shot_id": "full", "start_seconds": 0.0, "end_seconds": 5.0}],
            "entities": [
                {
                    "node_id": "book-1",
                    "predicate": "book",
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                    "shot_id": "full",
                    "visibility": "visible",
                    "provenance": {
                        "source_kind": "manual_observation",
                        "source_id": "frame-1",
                        "timestamps_seconds": [1.0, 4.0],
                    },
                }
            ],
            "parts": [],
            "events": [],
            "identity_relations": [],
            "coverage": [
                {
                    "claim_id": "coverage-1",
                    "obligation": "target_coverage",
                    "target_kind": "entity",
                    "target_predicate": "book",
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "spatial_scope": "full_frame",
                    "temporal_resolution_seconds": 0.5,
                    "coverage_state": "complete",
                }
            ],
        }
    )


def chain_fixture() -> tuple[dict[str, object], dict[str, object]]:
    request = {
        "video_id": "video-1",
        "question": "How many books?",
        "duration_seconds": 5.0,
        "frozen_map": "one scene",
        "source_path": "/private/video-1.mkv",
        "source_sha256": "source-video-1",
    }
    request["request_sha256"] = canonical_sha256(request)
    requests = _with_hash({"schema_version": 1, "rows": [request]})
    authored_row = {
        "video_id": "video-1",
        "request_sha256": request["request_sha256"],
        "source_sha256": request["source_sha256"],
        "query": query_payload("video-1"),
        "graph": graph_payload("video-1"),
    }
    authored_row["complete_sha256"] = canonical_sha256(authored_row)
    authored = _with_hash(
        {
            "schema_version": 1,
            "authoring_requests_complete_sha256": requests["complete_sha256"],
            "rows": [authored_row],
        }
    )
    return requests, authored


def test_runner_executes_and_binds_complete_label_blind_chain() -> None:
    requests, authored = chain_fixture()

    freeze = MODULE.execute_oracle_slice(
        authoring_requests=requests, authored_inputs=authored
    )

    assert freeze["authoring_requests_complete_sha256"] == requests["complete_sha256"]
    assert freeze["authored_inputs_complete_sha256"] == authored["complete_sha256"]
    assert freeze["rows"][0]["proof"]["exact_count"] == 1
    assert len(freeze["rows"][0]["complete_sha256"]) == 64
    assert len(freeze["complete_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda requests, authored: requests["rows"][0].update(
                question="How many changed books?"
            ),
            "authoring requests complete_sha256 changed",
        ),
        (
            lambda requests, authored: authored["rows"][0].update(
                source_sha256="changed-media"
            ),
            "authored inputs complete_sha256 changed",
        ),
        (
            lambda requests, authored: authored["rows"][0]["query"].update(
                query_id="other-video"
            ),
            "authored inputs complete_sha256 changed",
        ),
        (
            lambda requests, authored: authored["rows"].append(
                deepcopy(authored["rows"][0])
            ),
            "authored inputs complete_sha256 changed",
        ),
    ],
)
def test_runner_rejects_chain_mutation(mutation, message: str) -> None:
    requests, authored = chain_fixture()
    mutation(requests, authored)

    with pytest.raises(ValueError, match=message):
        MODULE.execute_oracle_slice(
            authoring_requests=requests, authored_inputs=authored
        )


def test_runner_rejects_rehashed_id_mismatch_and_label_leakage() -> None:
    requests, authored = chain_fixture()
    authored["rows"][0]["query"]["query_id"] = "other-video"
    authored["rows"][0]["query"]["complete_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in authored["rows"][0]["query"].items()
            if key != "complete_sha256"
        }
    )
    authored["rows"][0]["complete_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in authored["rows"][0].items()
            if key != "complete_sha256"
        }
    )
    authored["complete_sha256"] = canonical_sha256(
        {key: value for key, value in authored.items() if key != "complete_sha256"}
    )
    with pytest.raises(ValueError, match="query and graph IDs must equal video ID"):
        MODULE.execute_oracle_slice(
            authoring_requests=requests, authored_inputs=authored
        )

    requests, authored = chain_fixture()
    authored["rows"][0]["graph"]["metadata"] = {"reference_answer": "one"}
    authored["rows"][0]["complete_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in authored["rows"][0].items()
            if key != "complete_sha256"
        }
    )
    authored["complete_sha256"] = canonical_sha256(
        {key: value for key, value in authored.items() if key != "complete_sha256"}
    )
    with pytest.raises(ValueError, match="forbidden label-bearing key"):
        MODULE.execute_oracle_slice(
            authoring_requests=requests, authored_inputs=authored
        )
