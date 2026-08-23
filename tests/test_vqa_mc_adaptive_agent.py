from __future__ import annotations

import math

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_adaptive_agent import (
    FINAL_SCHEMA,
    PLANNER_SCHEMA,
    InspectionPlan,
    choose_final_ranking,
    fallback_plan,
    parse_stage1,
    parse_stage2,
    stage1_prompt,
    stage2_prompt,
    validate_plan,
)


def _row() -> dict[str, object]:
    return {
        "video_id": "video-a",
        "question": "How many times does the person jump?",
        "duration_seconds": 6.0,
        "options": [
            {"option_id": option_id, "text": text}
            for option_id, text in zip(
                "ABCD", ("once", "twice", "three times", "four times"), strict=True
            )
        ],
    }


def _plan() -> InspectionPlan:
    return InspectionPlan(
        start_seconds=1.25,
        end_seconds=4.75,
        fps=8.0,
        resolution="high",
        evidence_gap="rapid_or_repeated_action",
        evidence_question="Count each completed repetition.",
        source="model",
    )


def test_schemas_require_complete_bounded_objects() -> None:
    assert PLANNER_SCHEMA["required"] == ["ranking", "inspection"]
    assert PLANNER_SCHEMA["additionalProperties"] is False
    inspection = PLANNER_SCHEMA["properties"]["inspection"]
    assert inspection["properties"]["fps"]["enum"] == [1.0, 4.0, 8.0]
    assert inspection["properties"]["resolution"]["enum"] == ["low", "high"]
    assert FINAL_SCHEMA["required"] == ["ranking", "evidence"]
    assert FINAL_SCHEMA["properties"]["evidence"]["minItems"] == 1
    assert FINAL_SCHEMA["properties"]["evidence"]["maxItems"] == 3


def test_stage1_parses_ranking_and_generic_inspection() -> None:
    parsed = parse_stage1(
        {
            "ranking": ["B", "A", "C", "D"],
            "inspection": {
                "start_seconds": 1.25,
                "end_seconds": 4.75,
                "fps": 8.0,
                "resolution": "high",
                "evidence_gap": "rapid_or_repeated_action",
                "evidence_question": "Count each completed repetition.",
            },
        },
        duration_seconds=6.0,
    )

    assert parsed["ranking"] == ["B", "A", "C", "D"]
    assert parsed["inspection"] == _plan()


@pytest.mark.parametrize(
    "updates",
    [
        {"start_seconds": math.nan},
        {"start_seconds": -0.1},
        {"end_seconds": 6.1},
        {"start_seconds": 4.0, "end_seconds": 4.0},
        {"start_seconds": 4.0, "end_seconds": 3.0},
        {"fps": 2.0},
        {"resolution": "ultra_high"},
        {"evidence_gap": "unknown-gap"},
        {"evidence_question": "   "},
    ],
)
def test_validate_plan_rejects_invalid_or_unbounded_requests(
    updates: dict[str, object],
) -> None:
    value = {
        "start_seconds": 1.25,
        "end_seconds": 4.75,
        "fps": 8.0,
        "resolution": "high",
        "evidence_gap": "rapid_or_repeated_action",
        "evidence_question": "Count each completed repetition.",
    }
    value.update(updates)

    with pytest.raises(ValueError):
        validate_plan(value, duration_seconds=6.0)


def test_fallback_is_complete_video_high_resolution_and_explicit() -> None:
    plan = fallback_plan(6.0)

    assert plan == InspectionPlan(
        start_seconds=0.0,
        end_seconds=6.0,
        fps=4.0,
        resolution="high",
        evidence_gap="other",
        evidence_question=(
            "Re-examine the complete video and verify which answer option is best supported."
        ),
        source="fallback",
    )


def test_stage1_prompt_requests_one_inspection_without_confidence_gate() -> None:
    prompt = stage1_prompt(_row())

    assert "always request exactly one follow-up inspection" in prompt
    assert "Video duration: 6.000 seconds" in prompt
    assert "A. once" in prompt
    assert "D. four times" in prompt


def test_stage2_prompt_hides_stage1_top_two_order() -> None:
    prompt = stage2_prompt(_row(), _plan(), ["B", "A", "C", "D"])

    assert "Stage-one top-two set (unordered): A, B" in prompt
    assert "top-two set (unordered): B, A" not in prompt
    assert "Count each completed repetition." in prompt
    assert "original-video seconds" in prompt


def test_stage2_parses_timestamped_evidence_inside_original_duration() -> None:
    parsed = parse_stage2(
        {
            "ranking": ["B", "A", "C", "D"],
            "evidence": [
                {
                    "timestamp_seconds": 2.5,
                    "observation": "The person completes the second jump.",
                }
            ],
        },
        duration_seconds=6.0,
    )

    assert parsed["ranking"] == ["B", "A", "C", "D"]
    assert parsed["evidence"][0]["timestamp_seconds"] == 2.5


@pytest.mark.parametrize("timestamp", [-0.1, 6.1, math.inf])
def test_stage2_rejects_out_of_duration_or_nonfinite_evidence(
    timestamp: float,
) -> None:
    with pytest.raises(ValueError):
        parse_stage2(
            {
                "ranking": ["B", "A", "C", "D"],
                "evidence": [
                    {"timestamp_seconds": timestamp, "observation": "A jump."}
                ],
            },
            duration_seconds=6.0,
        )


def test_final_selection_prefers_valid_stage2_then_falls_back_to_stage1() -> None:
    stage1 = ["A", "B", "C", "D"]
    stage2 = ["B", "A", "C", "D"]

    assert choose_final_ranking(stage1, stage2) == (stage2, "stage2")
    assert choose_final_ranking(stage1, None) == (stage1, "stage1_fallback")
    assert choose_final_ranking(None, None) == (None, "invalid")
