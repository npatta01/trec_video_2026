"""Blind grading and decision logic for the Nemotron 3 Omni paired pilot."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_CONDITIONS = ("audio-off", "audio-on")
_GRADES = frozenset({"C", "P", "W"})
_RUBRIC = {
    "C": "Correct",
    "P": "Partially correct",
    "W": "Wrong",
}
_GRADE_RANK = {"W": 0, "P": 1, "C": 2}


@dataclass(frozen=True, repr=False)
class BlindGradingBundle:
    """Public packets plus private resolver state that must not reach graders."""

    packets: dict[str, dict[str, Any]]
    _answer_keys: dict[str, dict[str, tuple[str, str]]]
    _canonical_keys: tuple[tuple[str, str], ...]
    _item_content: dict[tuple[str, str], dict[str, str]]


@dataclass(frozen=True)
class GradeResolution:
    status: str
    final_grades: dict[tuple[str, str], str] | None
    adjudication_packet: dict[str, Any] | None


@dataclass(frozen=True)
class PilotScore:
    completed: int
    denominator: int
    target_improvements: int | None
    control_regressions: int | None
    passed: bool
    disposition: str


def _validated_panel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 6:
        raise ValueError("Nemotron pilot grading requires exactly six rows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    role_counts = {"target": 0, "control": 0}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every pilot row must be an object")
        row_key = row.get("row_key")
        role = row.get("role")
        question = row.get("question")
        reference = row.get("reference")
        predictions = row.get("predictions")
        if not isinstance(row_key, str) or not row_key or row_key in seen:
            raise ValueError("pilot row keys must be unique non-empty strings")
        if role not in role_counts:
            raise ValueError("pilot row role must be target or control")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("pilot questions must be non-empty strings")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("pilot references must be non-empty strings")
        if not isinstance(predictions, Mapping) or set(predictions) != set(_CONDITIONS):
            raise ValueError(
                "each pilot row requires audio-off and audio-on predictions"
            )
        normalized_predictions: dict[str, dict[str, Any]] = {}
        for condition in _CONDITIONS:
            prediction = predictions[condition]
            if not isinstance(prediction, Mapping):
                raise TypeError("each pilot prediction must be an object")
            completed = prediction.get("completed")
            answer = prediction.get("answer")
            if not isinstance(completed, bool):
                raise TypeError("prediction completion flags must be booleans")
            if completed and (not isinstance(answer, str) or not answer.strip()):
                raise ValueError("completed predictions require non-empty answers")
            normalized_predictions[condition] = {
                "completed": completed,
                "answer": answer if isinstance(answer, str) else "",
            }
        seen.add(row_key)
        role_counts[role] += 1
        normalized.append(
            {
                "row_key": row_key,
                "role": role,
                "question": question,
                "reference": reference,
                "predictions": normalized_predictions,
            }
        )
    if role_counts != {"target": 4, "control": 2}:
        raise ValueError("pilot panel requires four targets and two controls")
    return normalized


def _completed_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        bool(row["predictions"][condition]["completed"])
        for row in rows
        for condition in _CONDITIONS
    )


def build_blind_grading_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    freeze_verified: bool,
    grader_ids: tuple[str, str],
    shuffle_seeds: tuple[int, int],
) -> BlindGradingBundle:
    """Build two grader packets after the caller verifies the external freeze."""

    if freeze_verified is not True:
        raise ValueError("external prediction freeze must be verified before grading")
    panel = _validated_panel(rows)
    if _completed_count(panel) != 12:
        raise ValueError("frozen pilot must be 12/12 complete before grading")
    if (
        len(grader_ids) != 2
        or len(set(grader_ids)) != 2
        or not all(isinstance(grader_id, str) and grader_id for grader_id in grader_ids)
    ):
        raise ValueError("exactly two independent grader identities are required")
    if len(shuffle_seeds) != 2 or shuffle_seeds[0] == shuffle_seeds[1]:
        raise ValueError("two distinct shuffle seeds are required")

    canonical_keys: list[tuple[str, str]] = []
    item_content: dict[tuple[str, str], dict[str, str]] = {}
    for row in panel:
        for condition in _CONDITIONS:
            private_key = (row["row_key"], condition)
            canonical_keys.append(private_key)
            item_content[private_key] = {
                "question": row["question"],
                "candidate_answer": row["predictions"][condition]["answer"],
                "reference_answer": row["reference"],
            }

    packets: dict[str, dict[str, Any]] = {}
    answer_keys: dict[str, dict[str, tuple[str, str]]] = {}
    prior_order: list[tuple[str, str]] | None = None
    for grader_index, (grader_id, seed) in enumerate(
        zip(grader_ids, shuffle_seeds, strict=True)
    ):
        order = list(canonical_keys)
        random.Random(seed).shuffle(order)
        if prior_order == order:
            order = order[1:] + order[:1]
        prior_order = order
        packet_digest = hashlib.sha256(
            f"{grader_index}:{grader_id}:{seed}".encode()
        ).hexdigest()[:12]
        packet_id = f"blind-{grader_index + 1}-{packet_digest}"
        answer_key: dict[str, tuple[str, str]] = {}
        items: list[dict[str, str]] = []
        for item_index, private_key in enumerate(order, start=1):
            item_id = f"item-{item_index:03d}"
            answer_key[item_id] = private_key
            items.append({"item_id": item_id, **item_content[private_key]})
        packets[grader_id] = {
            "packet_id": packet_id,
            "grader_id": grader_id,
            "rubric": dict(_RUBRIC),
            "items": items,
        }
        answer_keys[grader_id] = answer_key

    return BlindGradingBundle(
        packets=packets,
        _answer_keys=answer_keys,
        _canonical_keys=tuple(canonical_keys),
        _item_content=item_content,
    )


def _submission_grades(
    bundle: BlindGradingBundle,
    submission: Mapping[str, Any],
) -> tuple[str, dict[tuple[str, str], str]]:
    grader_id = submission.get("grader_id")
    if grader_id not in bundle.packets:
        raise ValueError("submissions must come from the two independent graders")
    raw_grades = submission.get("grades")
    if not isinstance(raw_grades, list):
        raise TypeError("grader submissions require a grades list")
    answer_key = bundle._answer_keys[grader_id]
    observed: dict[str, str] = {}
    for raw_grade in raw_grades:
        if not isinstance(raw_grade, Mapping) or set(raw_grade) != {
            "item_id",
            "grade",
        }:
            raise ValueError("each grader result requires item_id and grade")
        item_id = raw_grade["item_id"]
        grade = raw_grade["grade"]
        if item_id not in answer_key or item_id in observed:
            raise ValueError(
                "grader results must cover each assigned item exactly once"
            )
        if grade not in _GRADES:
            raise ValueError("each grade must be C, P, or W")
        observed[item_id] = grade
    if set(observed) != set(answer_key):
        raise ValueError("grader results must cover each assigned item exactly once")
    return grader_id, {
        answer_key[item_id]: grade for item_id, grade in observed.items()
    }


def _adjudication_packet(
    bundle: BlindGradingBundle,
    disagreements: Sequence[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, tuple[str, str]]]:
    case_key: dict[str, tuple[str, str]] = {}
    items: list[dict[str, str]] = []
    for case_index, private_key in enumerate(disagreements, start=1):
        case_id = f"case-{case_index:03d}"
        case_key[case_id] = private_key
        items.append({"case_id": case_id, **bundle._item_content[private_key]})
    disagreement_fingerprint = "|".join(case_key)
    packet_id = (
        "adjudication-"
        + hashlib.sha256(disagreement_fingerprint.encode()).hexdigest()[:12]
    )
    return (
        {"packet_id": packet_id, "rubric": dict(_RUBRIC), "items": items},
        case_key,
    )


def reconcile_grades(
    bundle: BlindGradingBundle,
    submissions: Sequence[Mapping[str, Any]],
    *,
    adjudication: Mapping[str, Any] | None = None,
) -> GradeResolution:
    """Resolve two independent grades, using a third only for disagreements."""

    if len(submissions) != 2:
        raise ValueError("exactly two independent grader submissions are required")
    parsed = [_submission_grades(bundle, submission) for submission in submissions]
    grader_ids = [grader_id for grader_id, _ in parsed]
    if len(set(grader_ids)) != 2 or set(grader_ids) != set(bundle.packets):
        raise ValueError("submissions must come from two independent graders")
    grades_a = parsed[0][1]
    grades_b = parsed[1][1]
    disagreements = [
        private_key
        for private_key in bundle._canonical_keys
        if grades_a[private_key] != grades_b[private_key]
    ]
    if not disagreements:
        if adjudication is not None:
            raise ValueError(
                "adjudication is forbidden when there are no disagreements"
            )
        return GradeResolution(
            status="final",
            final_grades=dict(grades_a),
            adjudication_packet=None,
        )

    packet, case_key = _adjudication_packet(bundle, disagreements)
    if adjudication is None:
        return GradeResolution(
            status="adjudication_required",
            final_grades=None,
            adjudication_packet=packet,
        )

    adjudicator_id = adjudication.get("grader_id")
    if (
        not isinstance(adjudicator_id, str)
        or not adjudicator_id
        or adjudicator_id in grader_ids
    ):
        raise ValueError("adjudication requires a distinct third grader")
    raw_grades = adjudication.get("grades")
    if not isinstance(raw_grades, list):
        raise TypeError("adjudication requires a grades list")
    adjudicated: dict[str, str] = {}
    for raw_grade in raw_grades:
        if not isinstance(raw_grade, Mapping) or set(raw_grade) != {
            "case_id",
            "grade",
        }:
            raise ValueError("each adjudication result requires case_id and grade")
        case_id = raw_grade["case_id"]
        grade = raw_grade["grade"]
        if case_id not in case_key or case_id in adjudicated:
            raise ValueError("adjudication must cover only disagreements exactly once")
        if grade not in _GRADES:
            raise ValueError("each grade must be C, P, or W")
        adjudicated[case_id] = grade
    if set(adjudicated) != set(case_key):
        raise ValueError("adjudication must cover only disagreements exactly once")

    final_grades = {
        private_key: grades_a[private_key]
        for private_key in bundle._canonical_keys
        if private_key not in disagreements
    }
    final_grades.update(
        {case_key[case_id]: grade for case_id, grade in adjudicated.items()}
    )
    return GradeResolution(
        status="final",
        final_grades=final_grades,
        adjudication_packet=None,
    )


def score_pilot(
    rows: Sequence[Mapping[str, Any]],
    resolution: GradeResolution | None,
) -> PilotScore:
    """Apply the predeclared completion, target-improvement, and control gates."""

    panel = _validated_panel(rows)
    completed = _completed_count(panel)
    if completed != 12:
        return PilotScore(
            completed=completed,
            denominator=12,
            target_improvements=None,
            control_regressions=None,
            passed=False,
            disposition="incomplete_technical_failure",
        )
    if (
        resolution is None
        or resolution.status != "final"
        or resolution.final_grades is None
    ):
        raise ValueError("a complete pilot requires finalized blind grades")
    expected_keys = {
        (row["row_key"], condition) for row in panel for condition in _CONDITIONS
    }
    if set(resolution.final_grades) != expected_keys or any(
        grade not in _GRADES for grade in resolution.final_grades.values()
    ):
        raise ValueError("final grades must cover all 12 frozen predictions")

    target_improvements = 0
    control_regressions = 0
    for row in panel:
        off_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-off")]]
        on_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-on")]]
        if row["role"] == "target" and on_rank > off_rank:
            target_improvements += 1
        if row["role"] == "control" and on_rank < off_rank:
            control_regressions += 1

    passed = target_improvements >= 2 and control_regressions == 0
    return PilotScore(
        completed=completed,
        denominator=12,
        target_improvements=target_improvements,
        control_regressions=control_regressions,
        passed=passed,
        disposition="pass" if passed else "valid_negative",
    )
