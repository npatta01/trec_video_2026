"""Blind grading and decision logic for the Nemotron 3 Omni paired pilot."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    EXPANDED_ROWS,
    MODEL_ID,
    MODEL_REVISION,
)

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


@dataclass(frozen=True)
class ExpandedPilotScore:
    completed: int
    denominator: int
    sound_target_improvements: int | None
    non_control_improvements: int | None
    non_control_regressions: int | None
    visual_control_regressions: int | None
    sound_replication_passed: bool
    broader_expansion_passed: bool
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


def _verified_frozen_predictions(
    *,
    predictions_path: Path,
    manifest_path: Path,
    checksums_path: Path,
) -> list[dict[str, Any]]:
    if (
        not predictions_path.is_file()
        or not manifest_path.is_file()
        or not checksums_path.is_file()
    ):
        raise ValueError("frozen predictions, manifest, and checksums must be files")
    predictions_bytes = predictions_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    recorded_checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path in recorded_checksums
        ):
            raise ValueError("frozen SHA256SUMS is malformed")
        recorded_checksums[relative_path] = digest
    if (
        recorded_checksums.get("./manifest.json")
        != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("manifest SHA-256 does not match frozen SHA256SUMS")
    predictions_sha256 = hashlib.sha256(predictions_bytes).hexdigest()
    if recorded_checksums.get("./predictions.jsonl") != predictions_sha256:
        raise ValueError("prediction SHA-256 does not match frozen SHA256SUMS")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("frozen manifest must be valid JSON") from error
    if not isinstance(manifest, Mapping):
        raise TypeError("frozen manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mode") != "pilot"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("reference_answers_available") is not False
    ):
        raise ValueError("frozen manifest identity mismatch")
    if manifest.get("frozen") is not True:
        raise ValueError("prediction manifest must declare a frozen artifact")
    if manifest.get("completion") != {"completed": 12, "expected": 12}:
        raise ValueError("frozen pilot must be 12/12 complete before grading")
    if manifest.get("predictions_sha256") != predictions_sha256:
        raise ValueError("prediction file SHA-256 does not match the frozen manifest")

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(predictions_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"frozen prediction line {line_number} must be valid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError("frozen prediction records must be objects")
        records.append(record)
    if len(records) != 12:
        raise ValueError("frozen pilot must contain exactly 12 prediction records")

    rows: list[dict[str, Any]] = []
    for pair_index in range(6):
        pair = records[pair_index * 2 : pair_index * 2 + 2]
        row_key = pair[0].get("video_id")
        role = pair[0].get("role")
        if not isinstance(row_key, str) or not row_key:
            raise ValueError("frozen prediction video identity is invalid")
        if role not in {"target", "control"}:
            raise ValueError("frozen prediction role is invalid")
        predictions: dict[str, dict[str, Any]] = {}
        for condition_index, (record, condition) in enumerate(
            zip(pair, _CONDITIONS, strict=True)
        ):
            if (
                record.get("schema_version") != 1
                or record.get("mode") != "pilot"
                or record.get("pair_index") != pair_index
                or record.get("video_id") != row_key
                or record.get("role") != role
                or record.get("condition_id") != condition
                or record.get("status") != "passed"
            ):
                raise ValueError(
                    "frozen prediction order, identity, or completion mismatch"
                )
            answer = record.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"frozen prediction {pair_index}:{condition_index} has no answer"
                )
            predictions[condition] = {"completed": True, "answer": answer}
        rows.append(
            {
                "row_key": row_key,
                "role": role,
                "predictions": predictions,
            }
        )
    if len({row["row_key"] for row in rows}) != 6:
        raise ValueError("frozen pilot row identities must be unique")
    if [row["role"] for row in rows].count("target") != 4 or [
        row["role"] for row in rows
    ].count("control") != 2:
        raise ValueError("frozen pilot requires four targets and two controls")
    return rows


def _verified_expanded_frozen_predictions(
    *,
    predictions_path: Path,
    manifest_path: Path,
    checksums_path: Path,
) -> list[dict[str, Any]]:
    """Verify the exact expanded-12 freeze before any reference access."""

    if (
        not predictions_path.is_file()
        or not manifest_path.is_file()
        or not checksums_path.is_file()
    ):
        raise ValueError("frozen predictions, manifest, and checksums must be files")
    predictions_bytes = predictions_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    recorded_checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path in recorded_checksums
        ):
            raise ValueError("frozen SHA256SUMS is malformed")
        recorded_checksums[relative_path] = digest
    if (
        recorded_checksums.get("./manifest.json")
        != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("manifest SHA-256 does not match frozen SHA256SUMS")
    predictions_sha256 = hashlib.sha256(predictions_bytes).hexdigest()
    if recorded_checksums.get("./predictions.jsonl") != predictions_sha256:
        raise ValueError("prediction SHA-256 does not match frozen SHA256SUMS")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("frozen manifest must be valid JSON") from error
    if not isinstance(manifest, Mapping):
        raise TypeError("frozen manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mode") != "expanded-pilot"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("reference_answers_available") is not False
    ):
        raise ValueError("frozen manifest identity mismatch")
    if manifest.get("frozen") is not True:
        raise ValueError("prediction manifest must declare a frozen artifact")
    if manifest.get("completion") != {"completed": 24, "expected": 24}:
        raise ValueError("frozen expanded pilot must be 24/24 complete before grading")
    if manifest.get("predictions_sha256") != predictions_sha256:
        raise ValueError("frozen manifest prediction SHA-256 binding mismatch")

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(predictions_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"frozen prediction line {line_number} must be valid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError("frozen prediction records must be objects")
        records.append(record)
    if len(records) != 24:
        raise ValueError("frozen expanded pilot must contain exactly 24 prediction records")

    rows: list[dict[str, Any]] = []
    for pair_index, (video_id, category, role) in enumerate(EXPANDED_ROWS):
        pair = records[pair_index * 2 : pair_index * 2 + 2]
        predictions: dict[str, dict[str, Any]] = {}
        for condition_index, (record, condition) in enumerate(
            zip(pair, _CONDITIONS, strict=True)
        ):
            if (
                record.get("schema_version") != 1
                or record.get("mode") != "expanded-pilot"
                or record.get("pair_index") != pair_index
                or record.get("video_id") != video_id
                or record.get("category") != category
                or record.get("role") != role
                or record.get("condition_id") != condition
                or record.get("status") != "passed"
            ):
                raise ValueError(
                    "frozen expanded prediction order, identity, or completion mismatch"
                )
            answer = record.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"frozen prediction {pair_index}:{condition_index} has no answer"
                )
            predictions[condition] = {"completed": True, "answer": answer}
        rows.append(
            {
                "row_key": video_id,
                "category": category,
                "role": role,
                "predictions": predictions,
            }
        )
    return rows


def _load_references_after_freeze(
    *,
    frozen_rows: list[dict[str, Any]],
    reference_loader: Callable[
        [tuple[str, ...]],
        Sequence[Mapping[str, Any]],
    ],
) -> list[dict[str, Any]]:
    row_keys = tuple(row["row_key"] for row in frozen_rows)
    reference_rows = reference_loader(row_keys)
    if len(reference_rows) != len(frozen_rows):
        raise ValueError(
            f"reference loader must return exactly {len(frozen_rows)} rows"
        )
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in reference_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_key",
            "question",
            "reference",
        }:
            raise ValueError(
                "reference rows require only row_key, question, and reference"
            )
        row_key = row["row_key"]
        if not isinstance(row_key, str) or row_key in by_key:
            raise ValueError("reference row identities must be unique strings")
        by_key[row_key] = row
    if set(by_key) != set(row_keys):
        raise ValueError("reference rows do not match the frozen prediction rows")
    return [
        {
            **frozen_row,
            "question": by_key[frozen_row["row_key"]]["question"],
            "reference": by_key[frozen_row["row_key"]]["reference"],
        }
        for frozen_row in frozen_rows
    ]


def _validated_expanded_panel(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != len(EXPANDED_ROWS):
        raise ValueError("expanded pilot grading requires exactly twelve rows")
    normalized: list[dict[str, Any]] = []
    for row, (video_id, category, role) in zip(rows, EXPANDED_ROWS, strict=True):
        if not isinstance(row, Mapping):
            raise TypeError("every expanded pilot row must be an object")
        if (
            row.get("row_key") != video_id
            or row.get("category") != category
            or row.get("role") != role
        ):
            raise ValueError("expanded pilot row identity or order mismatch")
        question = row.get("question")
        reference = row.get("reference")
        predictions = row.get("predictions")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("expanded pilot questions must be non-empty strings")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("expanded pilot references must be non-empty strings")
        if not isinstance(predictions, Mapping) or set(predictions) != set(_CONDITIONS):
            raise ValueError(
                "each expanded pilot row requires audio-off and audio-on predictions"
            )
        normalized_predictions: dict[str, dict[str, Any]] = {}
        for condition in _CONDITIONS:
            prediction = predictions[condition]
            if not isinstance(prediction, Mapping):
                raise TypeError("each expanded pilot prediction must be an object")
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
        normalized.append(
            {
                "row_key": video_id,
                "category": category,
                "role": role,
                "question": question,
                "reference": reference,
                "predictions": normalized_predictions,
            }
        )
    return normalized


def build_blind_grading_bundle(
    *,
    predictions_path: Path,
    manifest_path: Path,
    checksums_path: Path,
    reference_loader: Callable[
        [tuple[str, ...]],
        Sequence[Mapping[str, Any]],
    ],
    grader_ids: tuple[str, str],
    shuffle_seeds: tuple[int, int],
) -> BlindGradingBundle:
    """Verify frozen bytes before loading references and building blind packets."""

    frozen_rows = _verified_frozen_predictions(
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
    )
    panel = _validated_panel(
        _load_references_after_freeze(
            frozen_rows=frozen_rows,
            reference_loader=reference_loader,
        )
    )
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


def build_expanded_blind_grading_bundle(
    *,
    predictions_path: Path,
    manifest_path: Path,
    checksums_path: Path,
    reference_loader: Callable[
        [tuple[str, ...]],
        Sequence[Mapping[str, Any]],
    ],
    grader_ids: tuple[str, str],
    shuffle_seeds: tuple[int, int],
) -> BlindGradingBundle:
    """Build two label-free packets only after the exact expanded freeze verifies."""

    frozen_rows = _verified_expanded_frozen_predictions(
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
    )
    panel = _validated_expanded_panel(
        _load_references_after_freeze(
            frozen_rows=frozen_rows,
            reference_loader=reference_loader,
        )
    )
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
            f"expanded:{grader_index}:{grader_id}:{seed}".encode()
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


def score_expanded_pilot(
    rows: Sequence[Mapping[str, Any]],
    resolution: GradeResolution | None,
) -> ExpandedPilotScore:
    """Apply the predeclared sound-replication and broader-expansion gates."""

    panel = _validated_expanded_panel(rows)
    completed = _completed_count(panel)
    if completed != 24:
        return ExpandedPilotScore(
            completed=completed,
            denominator=24,
            sound_target_improvements=None,
            non_control_improvements=None,
            non_control_regressions=None,
            visual_control_regressions=None,
            sound_replication_passed=False,
            broader_expansion_passed=False,
            disposition="incomplete_technical_failure",
        )
    if (
        resolution is None
        or resolution.status != "final"
        or resolution.final_grades is None
    ):
        raise ValueError("a complete expanded pilot requires finalized blind grades")
    expected_keys = {
        (row["row_key"], condition) for row in panel for condition in _CONDITIONS
    }
    if set(resolution.final_grades) != expected_keys or any(
        grade not in _GRADES for grade in resolution.final_grades.values()
    ):
        raise ValueError("final grades must cover all 24 frozen predictions")

    sound_target_improvements = 0
    non_control_improvements = 0
    non_control_regressions = 0
    visual_control_regressions = 0
    for row in panel:
        off_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-off")]]
        on_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-on")]]
        if row["role"] == "target":
            if on_rank > off_rank:
                non_control_improvements += 1
                if row["category"] == "Non-speech sound and music":
                    sound_target_improvements += 1
            if on_rank < off_rank:
                non_control_regressions += 1
        elif on_rank < off_rank:
            visual_control_regressions += 1

    sound_replication_passed = (
        sound_target_improvements >= 2 and visual_control_regressions == 0
    )
    broader_expansion_passed = (
        sound_replication_passed
        and non_control_improvements >= 4
        and non_control_regressions <= 1
    )
    if broader_expansion_passed:
        disposition = "broader_expansion_pass"
    elif sound_replication_passed:
        disposition = "sound_replication_pass"
    else:
        disposition = "valid_negative"
    return ExpandedPilotScore(
        completed=completed,
        denominator=24,
        sound_target_improvements=sound_target_improvements,
        non_control_improvements=non_control_improvements,
        non_control_regressions=non_control_regressions,
        visual_control_regressions=visual_control_regressions,
        sound_replication_passed=sound_replication_passed,
        broader_expansion_passed=broader_expansion_passed,
        disposition=disposition,
    )
