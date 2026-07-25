"""Behavioral tests for deterministic question-directed frame selection."""

from __future__ import annotations

import pytest

from trec_video_2026.vqa_experiments.temporal_sampling import (
    select_temporal_frames,
)


def test_after_question_samples_after_anchor() -> None:
    times = select_temporal_frames("What happens after the shove?", 30, 18, 12)

    assert len(times) == 12
    assert sum(time > 18 for time in times) >= 8


def test_before_question_samples_before_anchor() -> None:
    times = select_temporal_frames("What happens before the player falls?", 30, 12, 12)

    assert len(times) == 12
    assert sum(time < 12 for time in times) >= 8


def test_beginning_question_densifies_first_quarter() -> None:
    times = select_temporal_frames("What happens at the beginning?", 40, None, 12)

    assert sum(time <= 10 for time in times) >= 8


def test_end_question_densifies_last_quarter() -> None:
    times = select_temporal_frames("What happens at the end?", 40, None, 12)

    assert sum(time >= 30 for time in times) >= 8


def test_sampler_keeps_global_context_and_is_deterministic() -> None:
    first = select_temporal_frames("What happens at the end?", 40, None, 12)
    second = select_temporal_frames("What happens at the end?", 40, None, 12)

    assert first == second
    assert any(10 < time < 20 for time in first)
    assert first == sorted(first)
    assert len(first) == len(set(first))


def test_anchor_is_clamped_and_near_identical_times_are_deduplicated() -> None:
    times = select_temporal_frames("What happens after it starts?", 0.001, 100, 12)

    assert times
    assert all(0 <= time <= 0.001 for time in times)
    assert len(times) == len(set(times))


def test_microsecond_canonicalization_never_exceeds_clip_duration() -> None:
    times = select_temporal_frames("What happens after it starts?", 0.0000006, None, 12)

    assert all(0 <= time <= 0.0000006 for time in times)


@pytest.mark.parametrize(
    ("duration", "count"),
    [(0, 12), (-1, 12), (30, 0), (30, -1)],
)
def test_sampler_rejects_invalid_duration_and_count(duration: float, count: int) -> None:
    with pytest.raises(ValueError):
        select_temporal_frames("What happens at the end?", duration, None, count)
