from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from typing import Any

import pytest

from trec_video_2026.vqa_frontier_phase0_panel import (
    CATEGORIES,
    CONDITIONS,
    PHASE0_RUN_ID,
    SELECTION_POLICY,
    build_phase0_panel,
    validate_phase0_panel,
)


QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_RUN_ID = "q35-12f-wa-20260803-c2"
QWEN_CONDITION_ID = "qwen35-4b-12f-frames-only"
QWEN_ROW_COUNT = 475


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_panel_fixture(
    durations: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0),
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    ordinal = 1
    for category in CATEGORIES:
        for index, duration in enumerate(durations):
            video_id = f"{category}-{index + 1}"
            examples.append(
                {
                    "ordinal": ordinal,
                    "phase": "evaluation",
                    "video_id": video_id,
                    "question": f"Synthetic {category} question {index}.",
                    "category": category,
                    "evidence_topology": "synthetic-topology",
                    "selection_rationale": "synthetic fixture",
                    "media_relative_path": f"media/{video_id}.mp4",
                    "media_sha256": _digest(video_id),
                    "size_bytes": 100 + ordinal,
                    "duration_seconds": duration,
                    "has_audio": True,
                    "has_video": True,
                }
            )
            ordinal += 1

    for video_id in ("spent-pilot", "unused-pilot"):
        examples.append(
            {
                "ordinal": ordinal,
                "phase": "preflight",
                "video_id": video_id,
                "question": "Synthetic preflight question.",
                "category": "temporal",
                "evidence_topology": "synthetic-topology",
                "selection_rationale": "synthetic fixture",
                "media_relative_path": f"media/{video_id}.mp4",
                "media_sha256": _digest(video_id),
                "size_bytes": 100 + ordinal,
                "duration_seconds": 12.0,
                "has_audio": True,
                "has_video": True,
            }
        )
        ordinal += 1

    return {
        "schema_version": 1,
        "run_id": "frontier-video-map-20260811",
        "selection_policy": "question-only-category-purity-duration-topology-v1",
        "reference_policy": "not-read-not-serialized",
        "evaluation_count": 35,
        "preflight_count": 2,
        "examples": examples,
    }


def qwen_fixture(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "video_id": row["video_id"],
            "model_id": QWEN_MODEL_ID,
            "model_revision": QWEN_MODEL_REVISION,
            "run_id": QWEN_RUN_ID,
            "condition_id": QWEN_CONDITION_ID,
            "status": "passed",
            "answer": "private baseline answer",
        }
        for row in source["examples"]
        if row["phase"] == "evaluation"
    ]
    for index in range(QWEN_ROW_COUNT - len(rows)):
        rows.append(
            {
                "video_id": f"qwen-filler-{index:03d}",
                "model_id": QWEN_MODEL_ID,
                "model_revision": QWEN_MODEL_REVISION,
                "run_id": QWEN_RUN_ID,
                "condition_id": QWEN_CONDITION_ID,
                "status": "passed",
                "answer": "private filler answer",
            }
        )
    return rows


def test_build_panel_selects_one_median_duration_row_per_category() -> None:
    source = source_panel_fixture(durations=(5.0, 10.0, 15.0, 20.0, 25.0))
    panel = build_phase0_panel(source, "spent-pilot", qwen_fixture(source))
    screen = [row for row in panel["examples"] if row["phase"] == "screen"]

    assert len(screen) == 7
    assert {row["category"] for row in screen} == set(CATEGORIES)
    assert {row["duration_seconds"] for row in screen} == {15.0}


def test_build_panel_rejects_missing_qwen_baseline() -> None:
    source = source_panel_fixture()
    qwen = [row for row in qwen_fixture(source) if row["video_id"] != "causal-3"]

    with pytest.raises(ValueError, match="exactly 475"):
        build_phase0_panel(source, "spent-pilot", qwen)


def test_build_panel_rejects_fabricated_or_wrong_provenance_qwen_rows() -> None:
    source = source_panel_fixture()
    qwen = qwen_fixture(source)

    fabricated = deepcopy(qwen)
    fabricated[-1]["video_id"] = "causal-3"
    with pytest.raises(ValueError, match="duplicate video_id"):
        build_phase0_panel(source, "spent-pilot", fabricated)

    wrong_provenance = deepcopy(qwen)
    wrong_provenance[0]["model_revision"] = "wrong-revision"
    with pytest.raises(ValueError, match="model_revision"):
        build_phase0_panel(source, "spent-pilot", wrong_provenance)


def test_build_panel_rejects_truncated_qwen_predictions() -> None:
    source = source_panel_fixture()
    truncated = qwen_fixture(source)[:-1]

    with pytest.raises(ValueError, match="exactly 475"):
        build_phase0_panel(source, "spent-pilot", truncated)


def test_panel_freezes_pilot_and_screen_condition_bindings() -> None:
    source = source_panel_fixture()
    panel = build_phase0_panel(source, "spent-pilot", qwen_fixture(source))

    assert panel["run_id"] == PHASE0_RUN_ID
    assert panel["selection_policy"] == SELECTION_POLICY
    assert panel["conditions"] == list(CONDITIONS)
    pilot = [row for row in panel["examples"] if row["phase"] == "pilot"]
    screen = [row for row in panel["examples"] if row["phase"] == "screen"]
    assert len(pilot) == 1
    assert pilot[0]["condition_ids"] == ["neutral_map"]
    assert all(row["condition_ids"] == list(CONDITIONS) for row in screen)
    assert all("answer" not in row for row in panel["examples"])


def test_validate_panel_rejects_recursive_reference_fields() -> None:
    source = source_panel_fixture()
    panel = build_phase0_panel(source, "spent-pilot", qwen_fixture(source))
    panel["examples"][0]["provenance"] = {"nested": [{"reference_answer": "private"}]}

    with pytest.raises(ValueError, match="forbidden field"):
        validate_phase0_panel(
            panel, source_panel=source, qwen_rows=qwen_fixture(source)
        )


def test_validate_panel_rejects_changed_source_or_qwen_binding() -> None:
    source = source_panel_fixture()
    qwen = qwen_fixture(source)
    panel = build_phase0_panel(source, "spent-pilot", qwen)

    changed_source = deepcopy(source)
    changed_source["examples"][0]["media_sha256"] = _digest("changed")
    with pytest.raises(ValueError, match="source panel"):
        validate_phase0_panel(panel, source_panel=changed_source, qwen_rows=qwen)

    changed_qwen = deepcopy(qwen)
    changed_qwen[
        next(
            index
            for index, row in enumerate(changed_qwen)
            if row["video_id"] == "causal-3"
        )
    ]["status"] = "failed"
    with pytest.raises(ValueError, match="matched Qwen baseline"):
        validate_phase0_panel(panel, source_panel=source, qwen_rows=changed_qwen)


def test_selection_uses_literal_ids_after_source_shuffle_and_ordinal_tie() -> None:
    source = source_panel_fixture(durations=(5.0, 10.0, 15.0, 15.0, 25.0))
    by_category = {
        category: [row for row in source["examples"] if row["category"] == category]
        for category in CATEGORIES
    }
    shuffled: list[dict[str, Any]] = []
    for category in reversed(CATEGORIES):
        category_rows = by_category[category]
        shuffled.extend(
            [
                category_rows[3],
                category_rows[0],
                category_rows[4],
                category_rows[1],
                category_rows[2],
            ]
        )
    shuffled.extend(row for row in source["examples"] if row["phase"] == "preflight")
    source["examples"] = shuffled
    random.Random(42).shuffle(source["examples"])

    panel = build_phase0_panel(source, "spent-pilot", qwen_fixture(source))
    selected = [
        (row["category"], row["video_id"])
        for row in panel["examples"]
        if row["phase"] == "screen"
    ]

    assert selected == [
        ("temporal", "temporal-3"),
        ("counting", "counting-3"),
        ("causal", "causal-3"),
        ("speech", "speech-3"),
        ("sound", "sound-3"),
        ("ocr_detail", "ocr_detail-3"),
        ("other_global", "other_global-3"),
    ]
