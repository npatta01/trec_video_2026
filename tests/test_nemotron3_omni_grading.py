from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.nemotron3_omni_grading import (
    BlindGradingBundle,
    GradeResolution,
    build_blind_grading_bundle,
    reconcile_grades,
    score_pilot,
)
from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    MODEL_ID,
    MODEL_REVISION,
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


def _write_frozen_artifacts(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    frozen: bool = True,
) -> tuple[Path, Path, Path]:
    predictions_path = tmp_path / "predictions.jsonl"
    records: list[dict[str, object]] = []
    for pair_index, row in enumerate(rows):
        predictions = row["predictions"]
        assert isinstance(predictions, Mapping)
        for condition in ("audio-off", "audio-on"):
            prediction = predictions[condition]
            assert isinstance(prediction, Mapping)
            completed = prediction["completed"]
            records.append(
                {
                    "schema_version": 1,
                    "mode": "pilot",
                    "pair_index": pair_index,
                    "video_id": row["row_key"],
                    "role": row["role"],
                    "condition_id": condition,
                    "status": "passed" if completed else "failed",
                    "answer": prediction["answer"] if completed else "",
                }
            )
    predictions_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    completed = sum(record["status"] == "passed" for record in records)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "pilot",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "frozen": frozen,
                "completion": {"completed": completed, "expected": 12},
                "predictions_sha256": hashlib.sha256(
                    predictions_path.read_bytes()
                ).hexdigest(),
                "reference_answers_available": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums_path = tmp_path / "SHA256SUMS"
    checksums_path.write_text(
        (
            f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"
            "  ./manifest.json\n"
            f"{hashlib.sha256(predictions_path.read_bytes()).hexdigest()}"
            "  ./predictions.jsonl\n"
        ),
        encoding="utf-8",
    )
    return predictions_path, manifest_path, checksums_path


def _reference_loader(
    rows: list[dict[str, object]],
    calls: list[tuple[str, ...]] | None = None,
):
    by_key = {row["row_key"]: row for row in rows}

    def load(row_keys: tuple[str, ...]) -> list[dict[str, object]]:
        if calls is not None:
            calls.append(row_keys)
        return [
            {
                "row_key": row_key,
                "question": by_key[row_key]["question"],
                "reference": by_key[row_key]["reference"],
            }
            for row_key in row_keys
        ]

    return load


def _build_bundle(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    grader_ids: tuple[str, str] = ("independent-a", "independent-b"),
    shuffle_seeds: tuple[int, int] = (104729, 130363),
) -> BlindGradingBundle:
    predictions_path, manifest_path, checksums_path = _write_frozen_artifacts(
        tmp_path,
        rows,
    )
    return build_blind_grading_bundle(
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        reference_loader=_reference_loader(rows),
        grader_ids=grader_ids,
        shuffle_seeds=shuffle_seeds,
    )


def _final_resolution(
    tmp_path: Path,
    rows: list[dict[str, object]],
    grades_by_answer: Mapping[str, str],
) -> tuple[BlindGradingBundle, GradeResolution]:
    bundle = _build_bundle(tmp_path, rows)
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


def test_verifies_frozen_hash_before_loading_references(tmp_path: Path) -> None:
    """Catches reference loading based on a caller assertion or altered predictions."""
    rows = _synthetic_rows()
    predictions_path, manifest_path, checksums_path = _write_frozen_artifacts(
        tmp_path,
        rows,
    )
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    reference_calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="SHA-256"):
        build_blind_grading_bundle(
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            reference_loader=_reference_loader(rows, reference_calls),
            grader_ids=("independent-a", "independent-b"),
            shuffle_seeds=(11, 29),
        )

    assert reference_calls == []


def test_verifies_manifest_checksum_before_loading_references(tmp_path: Path) -> None:
    """Catches changing the freeze declaration and recomputing only its inner hash."""
    rows = _synthetic_rows()
    predictions_path, manifest_path, checksums_path = _write_frozen_artifacts(
        tmp_path,
        rows,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "tampered-after-freeze"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    reference_calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="manifest SHA-256"):
        build_blind_grading_bundle(
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            reference_loader=_reference_loader(rows, reference_calls),
            grader_ids=("independent-a", "independent-b"),
            shuffle_seeds=(11, 29),
        )

    assert reference_calls == []


@pytest.mark.parametrize(
    ("rows", "frozen", "error"),
    [
        (_synthetic_rows(), False, "frozen"),
        (_synthetic_rows(completed=False), True, "12/12"),
    ],
)
def test_rejects_unfrozen_or_incomplete_artifacts_before_loading_references(
    tmp_path: Path,
    rows: list[dict[str, object]],
    frozen: bool,
    error: str,
) -> None:
    predictions_path, manifest_path, checksums_path = _write_frozen_artifacts(
        tmp_path,
        rows,
        frozen=frozen,
    )
    reference_calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match=error):
        build_blind_grading_bundle(
            predictions_path=predictions_path,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            reference_loader=_reference_loader(rows, reference_calls),
            grader_ids=("independent-a", "independent-b"),
            shuffle_seeds=(11, 29),
        )

    assert reference_calls == []


def test_builds_two_independently_shuffled_opaque_packets_without_linkage_leaks(
    tmp_path: Path,
) -> None:
    rows = _synthetic_rows()

    bundle = _build_bundle(tmp_path, rows)

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


def test_rejects_duplicate_grader_identity_and_non_cpw_grades(
    tmp_path: Path,
) -> None:
    grades = _all_grades()
    bundle = _build_bundle(
        tmp_path,
        _synthetic_rows(),
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


def test_disagreements_get_exact_third_party_adjudication_packet(
    tmp_path: Path,
) -> None:
    grades_a = _all_grades()
    grades_b = dict(grades_a)
    grades_b["candidate-1-on"] = "P"
    grades_b["candidate-4-off"] = "W"
    bundle = _build_bundle(
        tmp_path,
        _synthetic_rows(),
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


def test_adjudication_cannot_cover_agreements_or_reuse_an_independent_grader(
    tmp_path: Path,
) -> None:
    grades = _all_grades()
    bundle, resolution = _final_resolution(tmp_path, _synthetic_rows(), grades)
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


def test_exactly_two_target_improvements_and_no_control_regression_passes(
    tmp_path: Path,
) -> None:
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
    _, resolution = _final_resolution(tmp_path, rows, grades)

    score = score_pilot(rows, resolution)

    assert score.completed == 12
    assert score.denominator == 12
    assert score.target_improvements == 2
    assert score.control_regressions == 0
    assert score.passed is True
    assert score.disposition == "pass"


def test_one_target_improvement_is_a_valid_complete_negative(
    tmp_path: Path,
) -> None:
    grades = _all_grades()
    grades["candidate-0-off"] = "W"
    grades["candidate-0-on"] = "P"
    rows = _synthetic_rows()
    _, resolution = _final_resolution(tmp_path, rows, grades)

    score = score_pilot(rows, resolution)

    assert score.completed == 12
    assert score.target_improvements == 1
    assert score.control_regressions == 0
    assert score.passed is False
    assert score.disposition == "valid_negative"


def test_control_regression_is_a_valid_complete_negative(tmp_path: Path) -> None:
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
    _, resolution = _final_resolution(tmp_path, rows, grades)

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
