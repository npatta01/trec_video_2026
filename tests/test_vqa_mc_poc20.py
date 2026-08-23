from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    CANARY_VIDEO_IDS,
    CONDITIONS,
    EXPERIMENT_ID,
    build_bundle,
    fuse_borda,
    parse_ranking,
    score_rankings,
)


DATA_ROOT = Path("data")


def test_bundle_is_balanced_label_isolated_and_deterministic() -> None:
    """Catches panel drift, answer-position skew, or label leakage."""
    first = build_bundle(DATA_ROOT)
    second = build_bundle(DATA_ROOT)

    assert first == second
    requests = first["requests"]
    labels = first["labels"]
    assert requests["experiment_id"] == EXPERIMENT_ID
    assert len(requests["rows"]) == 20
    assert len(labels["rows"]) == 20
    assert {row["video_id"] for row in requests["rows"]}.issuperset(CANARY_VIDEO_IDS)
    assert {row["stratum"] for row in requests["rows"]} == {
        "temporal",
        "causal",
        "ocr_detail",
        "counting",
        "speech",
    }
    assert all(
        sum(row["stratum"] == stratum for row in requests["rows"]) == 4
        for stratum in {row["stratum"] for row in requests["rows"]}
    )
    assert all(row["transcript"].strip() for row in requests["rows"])
    request_text = json.dumps(requests).lower()
    assert "correct_answer" not in request_text
    assert "incorrect_answers" not in request_text
    assert "correct_option" not in request_text
    assert sorted(row["correct_option_id"] for row in labels["rows"]) == list("AAAAABBBBBCCCCCDDDDD")


def test_bundle_has_small_order_diagnostic_and_matched_qwen_views() -> None:
    """Catches accidental permutation voting or unmatched Qwen conditions."""
    bundle = build_bundle(DATA_ROOT)
    variants = bundle["requests"]["variants"]

    assert len(variants) == 27
    assert sum(row["variant_kind"] == "canonical" for row in variants) == 20
    assert sum(row["variant_kind"] == "reversed" for row in variants) == 5
    assert sum(row["variant_kind"] == "repeat" for row in variants) == 2
    assert {row["video_id"] for row in variants if row["variant_kind"] == "reversed"} == CANARY_VIDEO_IDS
    assert len(CONDITIONS) == 2
    assert len(variants) * len(CONDITIONS) == 54
    for row in variants:
        expected = list("DCBA") if row["variant_kind"] == "reversed" else list("ABCD")
        assert row["display_order"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ('{"ranking":["B","A","D","C"]}', ["B", "A", "D", "C"]),
        ({"ranking": ["D", "C", "B", "A"]}, ["D", "C", "B", "A"]),
    ),
)
def test_parse_ranking_accepts_only_complete_json_permutations(raw: object, expected: list[str]) -> None:
    """Catches ranking truncation or prose silently entering scoring."""
    assert parse_ranking(raw, option_ids=list("ABCD")) == expected


@pytest.mark.parametrize(
    "raw",
    (
        "A, B, C, D",
        '```json\n{"ranking":["A","B","C","D"]}\n```',
        '{"ranking":["A","B","C"]}',
        '{"ranking":["A","A","C","D"]}',
        '{"ranking":["A","B","C","E"]}',
        '{"ranking":["A","B","C","D"],"answer":"A"}',
    ),
)
def test_parse_ranking_rejects_noncanonical_outputs(raw: str) -> None:
    """Catches permissive parsing that hides provider contract failures."""
    with pytest.raises(ValueError, match="ranking"):
        parse_ranking(raw, option_ids=list("ABCD"))


def test_equal_family_borda_does_not_double_weight_qwen() -> None:
    """Catches two Qwen views being counted as two provider-family votes."""
    qwen = fuse_borda([["A", "B", "C", "D"], ["B", "A", "D", "C"]])
    final = fuse_borda([["D", "C", "B", "A"], qwen])

    assert qwen == ["A", "B", "C", "D"]
    assert final == ["A", "B", "C", "D"]


def test_scoring_reports_top1_mrr_and_invalids() -> None:
    """Catches incorrect rank metrics or invalid rows counted as wrong answers only."""
    labels = {"q1": "A", "q2": "D", "q3": "B"}
    predictions = {
        "q1": ["A", "B", "C", "D"],
        "q2": ["A", "D", "B", "C"],
        "q3": None,
    }

    result = score_rankings(predictions, labels)

    assert result["row_count"] == 3
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 1
    assert result["top1_correct"] == 1
    assert result["top1_accuracy"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx((1 + 1 / 2) / 3)
