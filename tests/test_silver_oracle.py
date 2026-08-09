from __future__ import annotations

import hashlib
import json

import pytest

from trec_video_2026.vqa_experiments.silver_oracle import (
    build_frame_plans,
    parse_timestamped_whisper,
    protocol_sha256,
    validate_selector_result,
)


def selector_row(
    *, duration: float = 30.0, segment_count: int = 4
) -> dict[str, object]:
    return {
        "video_id": "video-01",
        "duration_seconds": duration,
        "whisper_segment_count": segment_count,
    }


def localized_result() -> dict[str, object]:
    return {
        "status": "localized",
        "intervals": [{"start_seconds": 3.0, "end_seconds": 8.0}],
        "whisper_segment_ids": [1, 2],
    }


def frame_inventory(
    *, duration: float = 30.0, count: int = 121
) -> list[dict[str, object]]:
    media_sha = "a" * 64
    return [
        {
            "stream_index": 0,
            "frame_index": index,
            "actual_pts_seconds": round(duration * index / (count - 1), 6),
            "time_base": "1/1000",
            "source_media_sha256": media_sha,
        }
        for index in range(count)
    ]


def test_validate_selector_result_accepts_exact_localized_schema() -> None:
    result = validate_selector_result(localized_result(), selector_row())

    assert result == localized_result()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"rationale": "the answer appears here"}, "exactly"),
        ({"answer": "forbidden"}, "exactly"),
        ({"confidence": 0.99}, "exactly"),
    ],
)
def test_validate_selector_result_rejects_generated_text_and_extra_keys(
    mutation: dict[str, object], message: str
) -> None:
    payload = {**localized_result(), **mutation}

    with pytest.raises(ValueError, match=message):
        validate_selector_result(payload, selector_row())


def test_validate_selector_result_rejects_interval_over_budget() -> None:
    payload = localized_result()
    payload["intervals"] = [{"start_seconds": 0.0, "end_seconds": 12.001}]

    with pytest.raises(ValueError, match="duration budget"):
        validate_selector_result(payload, selector_row(duration=30.0))


def test_validate_selector_result_rejects_overlapping_intervals() -> None:
    payload = localized_result()
    payload["intervals"] = [
        {"start_seconds": 2.0, "end_seconds": 7.0},
        {"start_seconds": 6.5, "end_seconds": 9.0},
    ]

    with pytest.raises(ValueError, match="overlap"):
        validate_selector_result(payload, selector_row())


def test_validate_selector_result_rejects_four_or_invalid_segment_ids() -> None:
    payload = localized_result()
    payload["whisper_segment_ids"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="at most three"):
        validate_selector_result(payload, selector_row())

    payload["whisper_segment_ids"] = [0, 4]
    with pytest.raises(ValueError, match="outside"):
        validate_selector_result(payload, selector_row(segment_count=4))


def test_validate_selector_result_requires_empty_non_localizable_payload() -> None:
    valid = {
        "status": "non_localizable",
        "intervals": [],
        "whisper_segment_ids": [],
    }
    assert validate_selector_result(valid, selector_row()) == valid

    invalid = {**valid, "whisper_segment_ids": [0]}
    with pytest.raises(ValueError, match="empty"):
        validate_selector_result(invalid, selector_row())


def test_parse_timestamped_whisper_preserves_verbatim_lines_and_bounds() -> None:
    text = (
        "[00:00:00.000] First line.\n"
        "[00:00:04.080] Second line.\n"
        "[00:00:09.960] Final line."
    )

    segments = parse_timestamped_whisper(text, duration_seconds=12.0)

    assert [segment.segment_id for segment in segments] == [0, 1, 2]
    assert [segment.start_seconds for segment in segments] == [0.0, 4.08, 9.96]
    assert [segment.end_seconds for segment in segments] == [4.08, 9.96, 12.0]
    assert segments[1].source_line == "[00:00:04.080] Second line."
    assert segments[1].text == "Second line."


def test_parse_timestamped_whisper_rejects_non_monotonic_or_overrun_timestamps() -> (
    None
):
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_timestamped_whisper(
            "[00:00:04.000] Later\n[00:00:04.000] Duplicate",
            duration_seconds=10.0,
        )


def test_parse_timestamped_whisper_can_drop_only_trailing_overrun_lines() -> None:
    segments = parse_timestamped_whisper(
        "[00:00:00.000] Valid\n[00:00:09.000] Valid too\n[00:00:11.000] Overrun",
        duration_seconds=10.0,
        drop_trailing_overrun=True,
    )

    assert [segment.segment_id for segment in segments] == [0, 1]
    assert [segment.end_seconds for segment in segments] == [9.0, 10.0]

    with pytest.raises(ValueError, match="duration"):
        parse_timestamped_whisper(
            "[00:00:11.000] Outside",
            duration_seconds=10.0,
        )


def test_build_frame_plans_constructs_exact_global_hybrid_and_local_views() -> None:
    plans = build_frame_plans(
        selector_row(),
        localized_result(),
        frame_inventory(),
    )

    assert set(plans) == {"G", "H", "L", "localized", "fallback_reason"}
    assert plans["localized"] is True
    assert plans["fallback_reason"] is None
    assert len(plans["G"]) == len(plans["H"]) == len(plans["L"]) == 12
    assert sum(frame["slot_kind"] == "local" for frame in plans["H"]) == 6
    assert sum(frame["slot_kind"] == "global" for frame in plans["H"]) == 6
    assert all(frame["slot_kind"] == "local" for frame in plans["L"])
    for view in ("G", "H", "L"):
        timestamps = [frame["actual_pts_seconds"] for frame in plans[view]]
        assert timestamps == sorted(timestamps)
        assert len(timestamps) == len(set(timestamps)) == 12


def test_build_frame_plans_falls_back_to_global_for_non_localizable_rows() -> None:
    selector = {
        "status": "non_localizable",
        "intervals": [],
        "whisper_segment_ids": [],
    }

    plans = build_frame_plans(selector_row(), selector, frame_inventory())

    assert plans["localized"] is False
    assert plans["fallback_reason"] == "non_localizable"
    assert plans["G"] == plans["H"] == plans["L"]


def test_protocol_sha256_uses_canonical_utf8_json() -> None:
    protocol = {"z": ["é", 2], "a": {"enabled": True}}
    expected = hashlib.sha256(
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert protocol_sha256(protocol) == expected
