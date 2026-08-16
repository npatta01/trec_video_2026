from __future__ import annotations

import copy

import pytest

from trec_video_2026.vqa_blind_frame_counting import (
    UnsupportedCountSpec,
    VisibleEpisodeCountSpec,
    build_frame_decision_request,
    parse_visible_episode_count_spec,
)


def visible_episode_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "spec_id": "spec-1",
        "target": "ball visibly present in the frame",
        "unit": "visible_appearance_episode",
        "scope": "full_video",
        "episode_rule": "maximal contiguous run of present samples",
        "maximum_sample_gap_seconds": 0.5,
        "every_decoded_frame": True,
        "cut_policy": "continuous_across_cuts",
        "occluded_identity": "unsupported",
    }


def three_frame_inventory() -> dict[str, object]:
    return {
        "schema_version": 1,
        "video_id": "video-1",
        "source_sha256": "a" * 64,
        "duration_seconds": 1.0,
        "decoded_frame_count": 3,
        "frames": [
            {
                "frame_id": "F0",
                "decoded_frame_index": 0,
                "timestamp_seconds": 0.0,
                "artifact_sha256": "b" * 64,
                "is_first_decoded_frame": True,
                "is_last_decoded_frame": False,
            },
            {
                "frame_id": "F1",
                "decoded_frame_index": 1,
                "timestamp_seconds": 0.5,
                "artifact_sha256": "c" * 64,
                "is_first_decoded_frame": False,
                "is_last_decoded_frame": False,
            },
            {
                "frame_id": "F2",
                "decoded_frame_index": 2,
                "timestamp_seconds": 1.0,
                "artifact_sha256": "d" * 64,
                "is_first_decoded_frame": False,
                "is_last_decoded_frame": True,
            },
        ],
    }


def one_sheet_presentations() -> list[dict[str, object]]:
    return [
        {
            "presentation_id": "sheet-1",
            "presentation_sha256": "e" * 64,
            "frame_ids": ["F0", "F1", "F2"],
            "mode": "numbered_contact_sheet",
        }
    ]


def test_supported_count_spec_and_endpoint_complete_request_are_hash_bound() -> None:
    parsed = parse_visible_episode_count_spec(visible_episode_spec())
    assert isinstance(parsed, VisibleEpisodeCountSpec)

    request = build_frame_decision_request(
        parsed,
        inventory=three_frame_inventory(),
        presentations=one_sheet_presentations(),
    )

    assert [frame.frame_id for frame in request.frames] == ["F0", "F1", "F2"]
    assert request.frames[0].is_first_decoded_frame
    assert request.frames[-1].is_last_decoded_frame
    assert request.maximum_observed_gap_seconds == 0.5
    assert len(request.complete_sha256) == 64


def test_count_spec_rejects_unknown_label_bearing_fields() -> None:
    payload = visible_episode_spec()
    payload["reference_answer"] = "four"

    with pytest.raises(ValueError, match="label|unknown"):
        parse_visible_episode_count_spec(payload)


def test_other_count_units_return_typed_unsupported_result() -> None:
    payload = visible_episode_spec()
    payload["unit"] = "distinct_simultaneous_objects"

    parsed = parse_visible_episode_count_spec(payload)

    assert isinstance(parsed, UnsupportedCountSpec)
    assert parsed.outcome == "unsupported"
    assert parsed.reason_codes == ("unsupported_count_unit",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("scope", "selected_clip", "scope"),
        ("episode_rule", "model decides", "episode_rule"),
        ("maximum_sample_gap_seconds", 0, "maximum_sample_gap_seconds"),
        ("every_decoded_frame", "yes", "every_decoded_frame"),
        ("cut_policy", "guess_at_cuts", "cut_policy"),
        ("occluded_identity", "track_it", "occluded_identity"),
    ],
)
def test_supported_count_spec_rejects_unsafe_semantics(
    field: str, value: object, message: str
) -> None:
    payload = visible_episode_spec()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        parse_visible_episode_count_spec(payload)


@pytest.mark.parametrize(
    "case",
    [
        "missing_first_frame",
        "missing_last_frame",
        "duplicate_frame_id",
        "duplicate_decoded_index",
        "non_monotonic_pts",
        "gap_above_limit",
        "skipped_index_when_every_frame_required",
        "unbound_frame",
        "duplicate_presentation_membership_within_mode",
        "incomplete_mode_coverage",
        "unknown_presentation_mode",
        "invalid_hash",
        "label_bearing_inventory",
    ],
)
def test_frame_request_rejects_incomplete_or_unauthenticated_coverage(
    case: str,
) -> None:
    spec_payload = visible_episode_spec()
    inventory = copy.deepcopy(three_frame_inventory())
    presentations = copy.deepcopy(one_sheet_presentations())
    frames = inventory["frames"]
    assert isinstance(frames, list)

    if case == "missing_first_frame":
        frames.pop(0)
    elif case == "missing_last_frame":
        frames.pop()
    elif case == "duplicate_frame_id":
        frames[1]["frame_id"] = "F0"
    elif case == "duplicate_decoded_index":
        frames[1]["decoded_frame_index"] = 0
    elif case == "non_monotonic_pts":
        frames[1]["timestamp_seconds"] = 1.1
    elif case == "gap_above_limit":
        spec_payload["maximum_sample_gap_seconds"] = 0.4
    elif case == "skipped_index_when_every_frame_required":
        inventory["decoded_frame_count"] = 4
        frames[-1]["decoded_frame_index"] = 3
    elif case == "unbound_frame":
        presentations[0]["frame_ids"] = ["F0", "F2"]
    elif case == "duplicate_presentation_membership_within_mode":
        presentations.append(copy.deepcopy(presentations[0]))
        presentations[-1]["presentation_id"] = "sheet-2"
    elif case == "incomplete_mode_coverage":
        presentations.append(
            {
                "presentation_id": "individual-1",
                "presentation_sha256": "f" * 64,
                "frame_ids": ["F0"],
                "mode": "individual_frame",
            }
        )
    elif case == "unknown_presentation_mode":
        presentations[0]["mode"] = "native_video"
    elif case == "invalid_hash":
        frames[0]["artifact_sha256"] = "not-a-hash"
    elif case == "label_bearing_inventory":
        inventory["reference_answer"] = 4

    spec = parse_visible_episode_count_spec(spec_payload)
    assert isinstance(spec, VisibleEpisodeCountSpec)
    with pytest.raises((TypeError, ValueError)):
        build_frame_decision_request(spec, inventory, presentations)


def test_frame_request_allows_same_frames_once_in_each_presentation_mode() -> None:
    spec = parse_visible_episode_count_spec(visible_episode_spec())
    assert isinstance(spec, VisibleEpisodeCountSpec)
    presentations = one_sheet_presentations()
    presentations.extend(
        {
            "presentation_id": f"individual-{index}",
            "presentation_sha256": str(index + 1) * 64,
            "frame_ids": [frame_id],
            "mode": "individual_frame",
        }
        for index, frame_id in enumerate(("F0", "F1", "F2"))
    )

    request = build_frame_decision_request(spec, three_frame_inventory(), presentations)

    assert {binding.mode for binding in request.presentations} == {
        "numbered_contact_sheet",
        "individual_frame",
    }
