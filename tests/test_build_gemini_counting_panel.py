from __future__ import annotations

from collections import Counter

import pytest

from trec_video_2026.vqa_counting_evidence import (
    build_counting_panel,
    validate_label_blind,
)


def source_fixture():
    protocol = []
    predictions = []
    maps = []
    annotations = {}
    uploads = {}

    def add(video_id: str, question: str, reference: str, prediction: str) -> None:
        protocol.append(
            {
                "video_id": video_id,
                "question": question,
                "duration_seconds": 10.0,
                "source_path": f"/private/{video_id}.mkv",
                "source_sha256": f"source-{video_id}",
            }
        )
        predictions.append(
            {"video_id": video_id, "answer": prediction, "status": "passed"}
        )
        maps.append(
            {"video_id": video_id, "map": f"map for {video_id}", "status": "passed"}
        )
        annotations[video_id] = {"correct_answer": reference}
        uploads[video_id] = {
            "video_id": video_id,
            "file_name": f"files/{video_id}",
            "file_uri": f"https://example.invalid/{video_id}",
            "file_state": "ACTIVE",
            "expiration_time": "2026-08-15T00:00:00Z",
            "delivery_mime_type": "video/webm",
            "delivery_sha256": f"delivery-{video_id}",
            "source_sha256": f"source-{video_id}",
        }

    for index in range(38):
        add(f"correct-{index:02}", "How many objects?", "Two", "There are 2.")
    for index in range(26):
        add(f"wrong-{index:02}", "How many objects?", "Two", "There are 3.")
    add("ambiguous-00", "How many beds?", "One bunk-bed structure", "2")
    add(
        "not-count-00",
        "The number of arrows changes depending on what?",
        "The arithmetic operation",
        "The gates",
    )
    return protocol, predictions, maps, annotations, uploads


def test_builder_freezes_corrected_census_and_requested_panel_denominators() -> None:
    protocol, predictions, maps, annotations, uploads = source_fixture()
    adjudications = {
        "status_overrides": {"ambiguous-00": "ambiguous"},
        "required_control_ids": ["correct-00"],
        "canary_failure_ids": [f"wrong-{index:02}" for index in range(5)],
        "canary_control_ids": ["correct-00"],
    }
    artifact = build_counting_panel(
        protocol_rows=protocol,
        predictions=predictions,
        maps=maps,
        annotations=annotations,
        uploads=uploads,
        adjudications=adjudications,
    )

    assert artifact["census"]["counts"] == {
        "ambiguous": 1,
        "correct": 38,
        "not_count": 1,
        "wrong": 26,
    }
    assert Counter(row["panel_role"] for row in artifact["panel"]["rows"]) == {
        "confirmed_failure": 26,
        "diagnostic_ambiguity": 1,
        "correct_control": 12,
    }
    assert artifact["panel"]["score_denominators"] == {
        "confirmed_failures": 26,
        "correct_controls": 12,
    }
    assert len(artifact["requests"]["rows"]) == 39
    assert len(artifact["requests"]["canary_video_ids"]) == 6
    validate_label_blind(artifact["requests"])


def test_request_projection_never_contains_roles_references_or_prior_predictions() -> (
    None
):
    protocol, predictions, maps, annotations, uploads = source_fixture()
    artifact = build_counting_panel(
        protocol_rows=protocol,
        predictions=predictions,
        maps=maps,
        annotations=annotations,
        uploads=uploads,
        adjudications={
            "status_overrides": {"ambiguous-00": "ambiguous"},
            "required_control_ids": ["correct-00"],
            "canary_failure_ids": [f"wrong-{index:02}" for index in range(5)],
            "canary_control_ids": ["correct-00"],
        },
    )
    row = artifact["requests"]["rows"][0]
    assert set(row) == {
        "delivery_mime_type",
        "delivery_sha256",
        "duration_seconds",
        "expiration_time",
        "file_name",
        "file_state",
        "file_uri",
        "frozen_map",
        "question",
        "source_path",
        "source_sha256",
        "video_id",
    }
    assert "panel_role" not in row
    assert "correct_answer" not in row
    assert "prediction" not in row


def test_builder_rejects_any_count_change_instead_of_silently_drifting() -> None:
    protocol, predictions, maps, annotations, uploads = source_fixture()
    predictions[0]["answer"] = "3"
    with pytest.raises(ValueError, match="corrected census counts"):
        build_counting_panel(
            protocol_rows=protocol,
            predictions=predictions,
            maps=maps,
            annotations=annotations,
            uploads=uploads,
            adjudications={
                "status_overrides": {"ambiguous-00": "ambiguous"},
                "required_control_ids": ["correct-00"],
                "canary_failure_ids": [f"wrong-{index:02}" for index in range(5)],
                "canary_control_ids": ["correct-00"],
            },
        )


def test_builder_rejects_duplicate_canary_ids() -> None:
    protocol, predictions, maps, annotations, uploads = source_fixture()
    with pytest.raises(ValueError, match="distinct canary failure IDs"):
        build_counting_panel(
            protocol_rows=protocol,
            predictions=predictions,
            maps=maps,
            annotations=annotations,
            uploads=uploads,
            adjudications={
                "status_overrides": {"ambiguous-00": "ambiguous"},
                "required_control_ids": ["correct-00"],
                "canary_failure_ids": [
                    "wrong-00",
                    "wrong-00",
                    "wrong-01",
                    "wrong-02",
                    "wrong-03",
                ],
                "canary_control_ids": ["correct-00"],
            },
        )
