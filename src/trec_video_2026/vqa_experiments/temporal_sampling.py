"""Deterministic frame selection for temporal video questions.

The selector always retains a small global view of the clip.  Question words
then route the remaining frame budget toward the beginning, end, or the side
of a supplied event anchor named by the question.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable


_BEFORE = re.compile(r"\b(before|prior to|preceding|previously)\b", re.I)
_AFTER = re.compile(r"\b(after|following|then|next)\b", re.I)
_BEGINNING = re.compile(r"\b(beginning|start|initially|first)\b", re.I)
_END = re.compile(r"\b(end|ending|finally|last)\b", re.I)


def _profile(question: str) -> str | None:
    """Classify the explicit temporal direction in *question*, if any."""

    if _BEFORE.search(question):
        return "before"
    if _AFTER.search(question):
        return "after"
    if _BEGINNING.search(question):
        return "beginning"
    if _END.search(question):
        return "end"
    return None


def _midpoints(start: float, end: float, count: int) -> list[float]:
    if count <= 0 or end <= start:
        return []
    width = (end - start) / count
    return [start + width * (index + 0.5) for index in range(count)]


def _deduplicate(times: Iterable[float], duration: float) -> list[float]:
    """Clamp, canonicalize microsecond-neighbors, and return sorted times."""

    # ffmpeg accepts fractional seconds, but preserving more than microsecond
    # differences makes evidence manifests unstable without changing a frame.
    precision = 6
    # Rounding 0.6 microseconds to one microsecond can otherwise move a valid
    # timestamp beyond an extremely short clip, so clamp once more afterward.
    clamped = {
        min(duration, max(0.0, round(min(duration, max(0.0, time)), precision)))
        for time in times
    }
    return sorted(clamped)


def _fallback_region(profile: str, duration: float) -> tuple[float, float]:
    quarter = duration / 4
    if profile in {"before", "beginning"}:
        return 0.0, quarter
    return duration - quarter, duration


def _directed_times(
    region: tuple[float, float], local_count: int, global_times: list[float]
) -> list[float]:
    """Fill the local budget without colliding with global-context samples."""

    if local_count <= 0:
        return []
    global_keys = {round(time, 6) for time in global_times}
    candidates = [
        time
        for time in _midpoints(*region, max(local_count * 3, local_count))
        if round(time, 6) not in global_keys
    ]
    if len(candidates) < local_count:
        # This only occurs for very short clips, where microsecond rounding
        # necessarily leaves fewer distinct frame timestamps.
        return candidates
    return [candidates[index * len(candidates) // local_count] for index in range(local_count)]


def _local_region(
    profile: str | None, duration: float, anchor: float | None
) -> tuple[float, float] | None:
    if profile is None:
        return None
    if profile == "beginning":
        return 0.0, duration / 4
    if profile == "end":
        return duration * 3 / 4, duration

    if anchor is None:
        return _fallback_region(profile, duration)
    bounded_anchor = min(duration, max(0.0, anchor))
    region = (
        (0.0, bounded_anchor)
        if profile == "before"
        else (bounded_anchor, duration)
    )
    # A clamped anchor can leave no sampleable direction.  Keep the direction's
    # nearest temporal quarter rather than manufacturing duplicate endpoints.
    return region if region[1] > region[0] else _fallback_region(profile, duration)


def select_temporal_frames(
    question: str, duration: float, anchor: float | None, count: int
) -> list[float]:
    """Return reproducible timestamps matched to an explicit temporal cue.

    ``before``, ``after``, ``beginning``, and ``end`` retain two to four
    global context frames when the requested count permits and use the rest in
    the directed interval. Questions without one of these cues use uniform
    samples. Anchor timestamps are clamped to the clip; output timestamps are
    clamped and deduplicated at the microsecond resolution used by the
    frame-extraction manifest.
    """

    if not isinstance(question, str):
        raise ValueError("question must be a string")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive finite number")
    if anchor is not None and (
        not isinstance(anchor, (int, float)) or not math.isfinite(anchor)
    ):
        raise ValueError("anchor must be a finite number or None")

    duration = float(duration)
    profile = _profile(question)
    region = _local_region(profile, duration, None if anchor is None else float(anchor))
    if region is None:
        return _deduplicate(_midpoints(0.0, duration, count), duration)

    # A dozen-frame run keeps four broad anchors (roughly each clip quarter),
    # leaving two thirds of the budget for the question-directed evidence.
    global_count = min(count, max(2, min(4, count // 3)))
    local_count = count - global_count
    times = _midpoints(0.0, duration, global_count)
    times.extend(_directed_times(region, local_count, times))
    return _deduplicate(times, duration)
