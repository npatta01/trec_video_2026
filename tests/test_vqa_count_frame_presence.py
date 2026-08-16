from __future__ import annotations

import pytest

from trec_video_2026.vqa_count_frame_presence import (
    parse_frame_presence_ledger,
    reduce_presence_episodes,
)


def observation(
    frame_id: str,
    timestamp_seconds: float,
    state: str,
    *,
    first: bool = False,
    last: bool = False,
) -> dict[str, object]:
    decoded_frame_index = int(frame_id.removeprefix("F"))
    return {
        "frame_id": frame_id,
        "decoded_frame_index": decoded_frame_index,
        "timestamp_seconds": timestamp_seconds,
        "artifact_sha256": "a" * 64,
        "target_state": state,
        "is_first_decoded_frame": first,
        "is_last_decoded_frame": last,
    }


def ledger(frames: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "video_id": "video",
        "source_sha256": "b" * 64,
        "target": "ball visible in the frame",
        "episode_rule": "maximal contiguous run of present samples",
        "duration_seconds": 1.0,
        "decoded_frame_count": int(str(frames[-1]["frame_id"]).removeprefix("F")) + 1,
        "maximum_sample_gap_seconds": 0.26,
        "frames": frames,
    }


def test_endpoint_complete_presence_runs_include_left_and_right_censored_episodes() -> (
    None
):
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.0, "present", first=True),
                observation("F1", 0.25, "absent"),
                observation("F2", 0.50, "present"),
                observation("F3", 0.75, "present"),
                observation("F4", 1.0, "present", last=True),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "exact"
    assert result.value == 2
    assert [
        (episode.start_frame_id, episode.end_frame_id) for episode in result.episodes
    ] == [
        ("F0", "F0"),
        ("F2", "F4"),
    ]
    assert result.reason_codes == ()


@pytest.mark.parametrize(
    ("first", "last", "reason"),
    [
        (False, True, "missing_first_decoded_frame"),
        (True, False, "missing_last_decoded_frame"),
    ],
)
def test_missing_temporal_endpoint_abstains(
    first: bool, last: bool, reason: str
) -> None:
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.0, "absent", first=first),
                observation("F1", 0.25, "absent", last=last),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "abstain"
    assert result.value is None
    assert reason in result.reason_codes


@pytest.mark.parametrize("state", ["uncertain", "unobservable"])
def test_non_binary_frame_decision_abstains(state: str) -> None:
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.0, "absent", first=True),
                observation("F1", 0.25, state),
                observation("F2", 0.50, "present", last=True),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "abstain"
    assert result.value is None
    assert "non_binary_frame_decision" in result.reason_codes


def test_sampling_gap_above_frozen_maximum_abstains() -> None:
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.0, "absent", first=True),
                observation("F1", 0.50, "present", last=True),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "abstain"
    assert result.value is None
    assert "sample_gap_exceeds_maximum" in result.reason_codes


def test_endpoint_markers_do_not_replace_scope_timestamp_coverage() -> None:
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.50, "absent", first=True),
                observation("F1", 0.75, "present", last=True),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "abstain"
    assert result.value is None
    assert "scope_start_not_covered" in result.reason_codes
    assert "scope_end_not_covered" in result.reason_codes


def test_endpoint_markers_do_not_replace_decoded_inventory_coverage() -> None:
    payload = ledger(
        [
            observation("F0", 0.0, "absent", first=True),
            observation("F1", 1.0, "present", last=True),
        ]
    )
    payload["decoded_frame_count"] = 3

    result = reduce_presence_episodes(parse_frame_presence_ledger(payload))

    assert result.outcome == "abstain"
    assert result.value is None
    assert "last_decoded_frame_index_not_covered" in result.reason_codes


def test_complete_absent_ledger_returns_exact_zero() -> None:
    parsed = parse_frame_presence_ledger(
        ledger(
            [
                observation("F0", 0.0, "absent", first=True),
                observation("F1", 0.25, "absent"),
                observation("F2", 0.50, "absent"),
                observation("F3", 0.75, "absent"),
                observation("F4", 1.00, "absent", last=True),
            ]
        )
    )

    result = reduce_presence_episodes(parsed)

    assert result.outcome == "exact"
    assert result.value == 0
    assert result.episodes == ()


def test_parser_rejects_duplicate_frame_ids_and_unknown_fields() -> None:
    duplicate = ledger(
        [
            observation("F0", 0.0, "absent", first=True),
            observation("F0", 0.25, "absent", last=True),
        ]
    )
    with pytest.raises(ValueError, match="duplicate frame_id"):
        parse_frame_presence_ledger(duplicate)

    extra = ledger(
        [
            observation("F0", 0.0, "absent", first=True),
            observation("F1", 0.25, "absent", last=True),
        ]
    )
    extra["reference_answer"] = "two"
    with pytest.raises(ValueError, match="unknown frame-presence ledger fields"):
        parse_frame_presence_ledger(extra)


def test_parser_rejects_unsupported_episode_rule() -> None:
    payload = ledger(
        [
            observation("F0", 0.0, "absent", first=True),
            observation("F1", 0.25, "absent", last=True),
        ]
    )
    payload["episode_rule"] = "let the reviewer decide"

    with pytest.raises(ValueError, match="unsupported episode_rule"):
        parse_frame_presence_ledger(payload)


@pytest.mark.parametrize(
    ("frame_index", "flag"),
    [(1, "is_first_decoded_frame"), (0, "is_last_decoded_frame")],
)
def test_parser_rejects_endpoint_markers_on_interior_side(
    frame_index: int, flag: str
) -> None:
    frames = [
        observation("F0", 0.0, "absent", first=True),
        observation("F1", 0.25, "absent"),
        observation("F2", 0.50, "absent", last=True),
    ]
    frames[frame_index][flag] = True

    with pytest.raises(ValueError, match="decoded-frame endpoint marker"):
        parse_frame_presence_ledger(ledger(frames))
