from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_blind_frame_model import (
    aggregate_state_batches,
    build_contact_sheet_state_request,
    build_individual_state_request,
    parse_state_batch_result,
)


def test_contact_sheet_request_binds_every_labeled_frame(tmp_path: Path) -> None:
    request = build_contact_sheet_state_request(
        target="balls",
        sheets=(
            (
                tmp_path / "sheet-1.jpg",
                ("F0", "F4"),
                (0.0, 0.133),
            ),
        ),
    )

    assert request["frame_paths"] == [str((tmp_path / "sheet-1.jpg").resolve())]
    assert request["frame_descriptions"] == [
        "Contact sheet 1 contains labeled frames in reading order: "
        "F0 at 0.000 seconds; F4 at 0.133 seconds."
    ]
    assert "Required frame IDs: F0, F4" in request["question"]


def test_individual_request_is_label_free_and_requests_only_frame_states(
    tmp_path: Path,
) -> None:
    request = build_individual_state_request(
        target="balls",
        frame_ids=("F0", "F4"),
        frame_paths=(tmp_path / "F0.jpg", tmp_path / "F4.jpg"),
        timestamps_seconds=(0.0, 0.133),
    )

    assert request["frame_paths"] == [
        str((tmp_path / "F0.jpg").resolve()),
        str((tmp_path / "F4.jpg").resolve()),
    ]
    assert request["frame_descriptions"] == [
        "Exact frame F0 at 0.000 seconds:",
        "Exact frame F4 at 0.133 seconds:",
    ]
    prompt = request["question"]
    assert isinstance(prompt, str)
    assert "present, absent, uncertain, or unobservable" in prompt
    assert "final count" not in prompt.lower()
    assert all(
        forbidden not in prompt.lower()
        for forbidden in ("reference", "correct answer", "panel_role", "expected")
    )


def test_state_batch_parser_requires_exact_order_and_strict_json() -> None:
    answer = json.dumps(
        [
            {"frame_id": "F0", "target_state": "absent"},
            {"frame_id": "F4", "target_state": "present"},
        ]
    )

    parsed = parse_state_batch_result(
        {"status": "passed", "answer": answer}, expected_frame_ids=("F0", "F4")
    )

    assert parsed == (("F0", "absent"), ("F4", "present"))


@pytest.mark.parametrize(
    "answer",
    [
        '[{"frame_id":"F0","target_state":"absent"}]',
        '[{"frame_id":"F4","target_state":"present"},'
        '{"frame_id":"F0","target_state":"absent"}]',
        '[{"frame_id":"F0","target_state":"absent",'
        '"target_state":"present"},{"frame_id":"F4","target_state":"present"}]',
        '```json\n[{"frame_id":"F0","target_state":"absent"},'
        '{"frame_id":"F4","target_state":"present"}]\n```',
    ],
)
def test_state_batch_parser_rejects_incomplete_reordered_duplicate_or_fenced_json(
    answer: str,
) -> None:
    with pytest.raises((TypeError, ValueError, json.JSONDecodeError)):
        parse_state_batch_result(
            {"status": "passed", "answer": answer},
            expected_frame_ids=("F0", "F4"),
        )


def test_batch_aggregation_requires_complete_disjoint_request_order() -> None:
    aggregated = aggregate_state_batches(
        expected_frame_ids=("F0", "F4", "F8"),
        batches=(
            (("F0", "absent"), ("F4", "present")),
            (("F8", "absent"),),
        ),
    )

    assert aggregated == (
        ("F0", "absent"),
        ("F4", "present"),
        ("F8", "absent"),
    )

    with pytest.raises(ValueError, match="exact request order"):
        aggregate_state_batches(
            expected_frame_ids=("F0", "F4", "F8"),
            batches=(
                (("F0", "absent"), ("F4", "present")),
                (("F4", "present"), ("F8", "absent")),
            ),
        )
