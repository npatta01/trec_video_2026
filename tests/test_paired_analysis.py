import pytest

from trec_video_2026.vqa_experiments.paired_analysis import (
    classify_transition,
    summarize_paired_metrics,
    word_error_rate,
)


@pytest.mark.parametrize(
    ("frames_correct", "automatic_correct", "expected"),
    [
        (False, True, "rescued_by_automatic_transcript"),
        (True, False, "degraded_with_automatic_transcript"),
        (True, True, "remained_correct"),
        (False, False, "remained_incorrect"),
    ],
)
def test_classify_transition(frames_correct, automatic_correct, expected):
    assert classify_transition(frames_correct, automatic_correct) == expected


def test_summarize_paired_metrics_reports_deltas_and_transition_counts():
    rows = [
        {
            "video_id": "a",
            "frames_only": {"exact_match": 0.0, "token_f1": 0.0, "correct": False},
            "frames_automatic_transcript": {
                "exact_match": 1.0,
                "token_f1": 1.0,
                "correct": True,
            },
        },
        {
            "video_id": "b",
            "frames_only": {"exact_match": 0.0, "token_f1": 0.5, "correct": False},
            "frames_automatic_transcript": {
                "exact_match": 0.0,
                "token_f1": 0.25,
                "correct": False,
            },
        },
    ]

    summary = summarize_paired_metrics(rows, ("exact_match", "token_f1"))

    assert summary == {
        "row_count": 2,
        "condition_means": {
            "frames_only": {"exact_match": 0.0, "token_f1": 0.25},
            "frames_automatic_transcript": {
                "exact_match": 0.5,
                "token_f1": 0.625,
            },
        },
        "paired_deltas": {"exact_match": 0.5, "token_f1": 0.375},
        "transition_counts": {
            "rescued_by_automatic_transcript": 1,
            "degraded_with_automatic_transcript": 0,
            "remained_correct": 0,
            "remained_incorrect": 1,
        },
    }


def test_summarize_paired_metrics_rejects_duplicate_video_ids():
    row = {
        "video_id": "a",
        "frames_only": {"exact_match": 1.0, "correct": True},
        "frames_automatic_transcript": {"exact_match": 1.0, "correct": True},
    }

    with pytest.raises(ValueError, match="duplicate video_id"):
        summarize_paired_metrics([row, row], ("exact_match",))


def test_summarize_paired_metrics_requires_boolean_adjudication():
    rows = [
        {
            "video_id": "a",
            "frames_only": {"exact_match": 1.0, "correct": 1},
            "frames_automatic_transcript": {
                "exact_match": 1.0,
                "correct": True,
            },
        }
    ]

    with pytest.raises(TypeError, match="correct must be boolean"):
        summarize_paired_metrics(rows, ("exact_match",))


def test_word_error_rate_uses_reference_token_count():
    assert word_error_rate(
        "set green by e5 soon", "set green by e 5 soon"
    ) == pytest.approx(2 / 6)
    assert word_error_rate("bin green at h 2 now", "bin green at h 2 now") == 0.0


def test_word_error_rate_rejects_empty_reference():
    with pytest.raises(ValueError, match="reference"):
        word_error_rate("anything", "")
