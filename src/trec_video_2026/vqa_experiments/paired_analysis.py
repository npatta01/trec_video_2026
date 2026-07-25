"""Deterministic summaries for matched transcript-ablation results."""

from __future__ import annotations

import re
from collections.abc import Sequence


_TRANSITIONS = (
    "rescued_by_automatic_transcript",
    "degraded_with_automatic_transcript",
    "remained_correct",
    "remained_incorrect",
)
_WORD = re.compile(r"[a-z0-9]+")


def word_error_rate(prediction: str, reference: str) -> float:
    """Return case- and punctuation-insensitive word-level edit distance."""
    predicted = _WORD.findall(prediction.casefold())
    expected = _WORD.findall(reference.casefold())
    if not expected:
        raise ValueError("reference must contain at least one word")
    previous = list(range(len(predicted) + 1))
    for expected_index, expected_word in enumerate(expected, start=1):
        current = [expected_index]
        for predicted_index, predicted_word in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[predicted_index] + 1,
                    previous[predicted_index - 1]
                    + int(expected_word != predicted_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def classify_transition(frames_correct: bool, automatic_correct: bool) -> str:
    """Classify one manually adjudicated matched pair."""
    if not isinstance(frames_correct, bool) or not isinstance(automatic_correct, bool):
        raise TypeError("correct must be boolean")
    if not frames_correct and automatic_correct:
        return "rescued_by_automatic_transcript"
    if frames_correct and not automatic_correct:
        return "degraded_with_automatic_transcript"
    if frames_correct:
        return "remained_correct"
    return "remained_incorrect"


def _condition(row: dict, field: str) -> dict:
    value = row.get(field)
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _score(condition: dict, metric: str, video_id: str) -> float:
    value = condition.get(metric)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{video_id} {metric} must be numeric in [0, 1]")
    return float(value)


def _correct(condition: dict) -> bool:
    value = condition.get("correct")
    if not isinstance(value, bool):
        raise TypeError("correct must be boolean")
    return value


def summarize_paired_metrics(rows: Sequence[dict], metrics: Sequence[str]) -> dict:
    """Aggregate matched condition scores and adjudicated transitions."""
    if not rows:
        raise ValueError("rows must be nonempty")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be unique and nonempty")

    seen_ids = set()
    totals = {
        "frames_only": {metric: 0.0 for metric in metrics},
        "frames_automatic_transcript": {metric: 0.0 for metric in metrics},
    }
    transition_counts = {transition: 0 for transition in _TRANSITIONS}

    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id must be a nonempty string")
        if video_id in seen_ids:
            raise ValueError(f"duplicate video_id: {video_id}")
        seen_ids.add(video_id)

        frames = _condition(row, "frames_only")
        automatic = _condition(row, "frames_automatic_transcript")
        for metric in metrics:
            totals["frames_only"][metric] += _score(frames, metric, video_id)
            totals["frames_automatic_transcript"][metric] += _score(
                automatic, metric, video_id
            )
        transition = classify_transition(_correct(frames), _correct(automatic))
        transition_counts[transition] += 1

    row_count = len(rows)
    means = {
        condition: {
            metric: total / row_count for metric, total in condition_totals.items()
        }
        for condition, condition_totals in totals.items()
    }
    deltas = {
        metric: (
            means["frames_automatic_transcript"][metric] - means["frames_only"][metric]
        )
        for metric in metrics
    }
    return {
        "row_count": row_count,
        "condition_means": means,
        "paired_deltas": deltas,
        "transition_counts": transition_counts,
    }
