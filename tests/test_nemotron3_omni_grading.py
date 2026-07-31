from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from trec_video_2026.vqa_experiments.nemotron3_omni_grading import (
    BlindGradingBundle,
    GradeResolution,
    build_blind_grading_bundle,
    reconcile_grades,
    score_pilot,
)


def _synthetic_rows(*, completed: bool = True) -> list[dict[str, object]]:
    roles = ("target", "target", "target", "target", "control", "control")
    rows: list[dict[str, object]] = []
    for index, role in enumerate(roles):
        rows.append(
            {
                "row_key": f"private-row-{index}",
                "role": role,
                "question": f"What happens in synthetic scene {index}?",
                "reference": f"reference-{index}",
                "predictions": {
                    "audio-off": {
                        "completed": completed or index != 5,
                        "answer": f"candidate-{index}-off",
                    },
                    "audio-on": {
                        "completed": True,
                        "answer": f"candidate-{index}-on",
                    },
                },
            }
        )
    return rows


def _submission(
    packet: Mapping[str, object],
    *,
    grades_by_answer: Mapping[str, str],
) -> dict[str, object]:
    items = packet["items"]
    assert isinstance(items, list)
    return {
        "grader_id": packet["grader_id"],
        "grades": [
            {
                "item_id": item["item_id"],
                "grade": grades_by_answer[item["candidate_answer"]],
            }
            for item in items
        ],
    }


def _final_resolution(
    rows: list[dict[str, object]],
    grades_by_answer: Mapping[str, str],
) -> tuple[BlindGradingBundle, GradeResolution]:
    bundle = build_blind_grading_bundle(
        rows,
        freeze_verified=True,
        grader_ids=("independent-a", "independent-b"),
        shuffle_seeds=(104729, 130363),
    )
    submissions = [
        _submission(packet, grades_by_answer=grades_by_answer)
        for packet in bundle.packets.values()
    ]
    resolution = reconcile_grades(bundle, submissions)
    assert resolution.status == "final"
    return bundle, resolution


def _all_grades(default: str = "C") -> dict[str, str]:
    return {
        f"candidate-{index}-{condition}": default
        for index in range(6)
        for condition in ("off", "on")
    }


def test_requires_externally_verified_complete_freeze_before_packet_building() -> None:
    rows = _synthetic_rows()

    with pytest.raises(ValueError, match="freeze"):
        build_blind_grading_bundle(
            rows,
            freeze_verified=False,
            grader_ids=("independent-a", "independent-b"),
            shuffle_seeds=(11, 29),
        )

    with pytest.raises(ValueError, match="12/12"):
        build_blind_grading_bundle(
            _synthetic_rows(completed=False),
            freeze_verified=True,
            grader_ids=("independent-a", "independent-b"),
            shuffle_seeds=(11, 29),
        )


def test_builds_two_independently_shuffled_opaque_packets_without_linkage_leaks() -> (
    None
):
    rows = _synthetic_rows()

    bundle = build_blind_grading_bundle(
        rows,
        freeze_verified=True,
        grader_ids=("independent-a", "independent-b"),
        shuffle_seeds=(104729, 130363),
    )

    assert set(bundle.packets) == {"independent-a", "independent-b"}
    first, second = bundle.packets.values()
    assert first["packet_id"] != second["packet_id"]
    assert [item["candidate_answer"] for item in first["items"]] != [
        item["candidate_answer"] for item in second["items"]
    ]
    for packet in bundle.packets.values():
        assert set(packet) == {"packet_id", "grader_id", "rubric", "items"}
        assert packet["rubric"] == {
            "C": "Correct",
            "P": "Partially correct",
            "W": "Wrong",
        }
        assert len(packet["items"]) == 12
        assert len({item["item_id"] for item in packet["items"]}) == 12
        assert all(
            set(item) == {"item_id", "question", "candidate_answer", "reference_answer"}
            for item in packet["items"]
        )

    grader_payload = json.dumps(bundle.packets, sort_keys=True)
    for row in rows:
        assert row["row_key"] not in grader_payload
    for forbidden in (
        '"row_key"',
        '"video_id"',
        '"role"',
        '"category"',
        '"condition"',
        '"audio-off"',
        '"audio-on"',
        '"target"',
        '"control"',
    ):
        assert forbidden not in grader_payload


def test_rejects_duplicate_grader_identity_and_non_cpw_grades() -> None:
    grades = _all_grades()
    bundle = build_blind_grading_bundle(
        _synthetic_rows(),
        freeze_verified=True,
        grader_ids=("independent-a", "independent-b"),
        shuffle_seeds=(11, 29),
    )
    first, second = bundle.packets.values()
    first_submission = _submission(first, grades_by_answer=grades)

    with pytest.raises(ValueError, match="independent"):
        reconcile_grades(bundle, [first_submission, first_submission])

    invalid = _submission(second, grades_by_answer=grades)
    invalid["grades"][0]["grade"] = "A"
    with pytest.raises(ValueError, match="C, P, or W"):
        reconcile_grades(bundle, [first_submission, invalid])


def test_disagreements_get_exact_third_party_adjudication_packet() -> None:
    grades_a = _all_grades()
    grades_b = dict(grades_a)
    grades_b["candidate-1-on"] = "P"
    grades_b["candidate-4-off"] = "W"
    bundle = build_blind_grading_bundle(
        _synthetic_rows(),
        freeze_verified=True,
        grader_ids=("independent-a", "independent-b"),
        shuffle_seeds=(11, 29),
    )
    submissions = [
        _submission(bundle.packets["independent-a"], grades_by_answer=grades_a),
        _submission(bundle.packets["independent-b"], grades_by_answer=grades_b),
    ]

    pending = reconcile_grades(bundle, submissions)

    assert pending.status == "adjudication_required"
    assert pending.final_grades is None
    assert pending.adjudication_packet is not None
    assert set(pending.adjudication_packet) == {"packet_id", "rubric", "items"}
    assert len(pending.adjudication_packet["items"]) == 2
    assert all(
        set(item) == {"case_id", "question", "candidate_answer", "reference_answer"}
        for item in pending.adjudication_packet["items"]
    )
    adjudication_payload = json.dumps(pending.adjudication_packet, sort_keys=True)
    assert "private-row-" not in adjudication_payload
    assert '"role"' not in adjudication_payload
    assert '"condition"' not in adjudication_payload

    adjudication = {
        "grader_id": "adjudicator-c",
        "grades": [
            {"case_id": item["case_id"], "grade": "P"}
            for item in pending.adjudication_packet["items"]
        ],
    }
    final = reconcile_grades(bundle, submissions, adjudication=adjudication)

    assert final.status == "final"
    assert final.adjudication_packet is None
    assert final.final_grades is not None
    assert len(final.final_grades) == 12


def test_adjudication_cannot_cover_agreements_or_reuse_an_independent_grader() -> None:
    grades = _all_grades()
    bundle, resolution = _final_resolution(_synthetic_rows(), grades)
    submissions = [
        _submission(packet, grades_by_answer=grades)
        for packet in bundle.packets.values()
    ]

    assert resolution.status == "final"
    with pytest.raises(ValueError, match="no disagreements"):
        reconcile_grades(
            bundle,
            submissions,
            adjudication={"grader_id": "third", "grades": []},
        )

    disagreeing = dict(grades)
    disagreeing["candidate-1-on"] = "P"
    submissions[1] = _submission(
        bundle.packets["independent-b"],
        grades_by_answer=disagreeing,
    )
    pending = reconcile_grades(bundle, submissions)
    assert pending.adjudication_packet is not None
    adjudication = {
        "grader_id": "independent-a",
        "grades": [
            {"case_id": item["case_id"], "grade": "P"}
            for item in pending.adjudication_packet["items"]
        ],
    }
    with pytest.raises(ValueError, match="third"):
        reconcile_grades(bundle, submissions, adjudication=adjudication)


def test_exactly_two_target_improvements_and_no_control_regression_passes() -> None:
    grades = _all_grades()
    grades.update(
        {
            "candidate-0-off": "W",
            "candidate-0-on": "P",
            "candidate-1-off": "P",
            "candidate-1-on": "C",
            "candidate-2-off": "C",
            "candidate-2-on": "C",
            "candidate-3-off": "P",
            "candidate-3-on": "P",
            "candidate-4-off": "C",
            "candidate-4-on": "C",
            "candidate-5-off": "P",
            "candidate-5-on": "P",
        }
    )
    rows = _synthetic_rows()
    _, resolution = _final_resolution(rows, grades)

    score = score_pilot(rows, resolution)

    assert score.completed == 12
    assert score.denominator == 12
    assert score.target_improvements == 2
    assert score.control_regressions == 0
    assert score.passed is True
    assert score.disposition == "pass"


def test_one_target_improvement_is_a_valid_complete_negative() -> None:
    grades = _all_grades()
    grades["candidate-0-off"] = "W"
    grades["candidate-0-on"] = "P"
    rows = _synthetic_rows()
    _, resolution = _final_resolution(rows, grades)

    score = score_pilot(rows, resolution)

    assert score.completed == 12
    assert score.target_improvements == 1
    assert score.control_regressions == 0
    assert score.passed is False
    assert score.disposition == "valid_negative"


def test_control_regression_is_a_valid_complete_negative() -> None:
    grades = _all_grades()
    grades.update(
        {
            "candidate-0-off": "W",
            "candidate-0-on": "P",
            "candidate-1-off": "P",
            "candidate-1-on": "C",
            "candidate-4-off": "C",
            "candidate-4-on": "P",
        }
    )
    rows = _synthetic_rows()
    _, resolution = _final_resolution(rows, grades)

    score = score_pilot(rows, resolution)

    assert score.completed == 12
    assert score.target_improvements == 2
    assert score.control_regressions == 1
    assert score.passed is False
    assert score.disposition == "valid_negative"


def test_incomplete_run_is_a_technical_failure_and_is_not_quality_scored() -> None:
    rows = _synthetic_rows(completed=False)

    score = score_pilot(rows, resolution=None)

    assert score.completed == 11
    assert score.denominator == 12
    assert score.target_improvements is None
    assert score.control_regressions is None
    assert score.passed is False
    assert score.disposition == "incomplete_technical_failure"
