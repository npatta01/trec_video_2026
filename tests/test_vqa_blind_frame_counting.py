from __future__ import annotations

import copy

import pytest

from trec_video_2026.vqa_count_frame_presence import reduce_presence_episodes
from trec_video_2026.vqa_blind_frame_counting import (
    FrameDecisionRequest,
    FrameDecisionSet,
    UnsupportedCountSpec,
    VisibleEpisodeCountSpec,
    build_frame_decision_request,
    parse_frame_decisions,
    parse_visible_episode_count_spec,
    reconcile_frame_decisions,
    select_challenge_frames,
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


def valid_request(
    *, cut_policy: str = "continuous_across_cuts"
) -> FrameDecisionRequest:
    spec_payload = visible_episode_spec()
    spec_payload["cut_policy"] = cut_policy
    spec = parse_visible_episode_count_spec(spec_payload)
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
    return build_frame_decision_request(spec, three_frame_inventory(), presentations)


def request_for_frame_count(frame_count: int) -> FrameDecisionRequest:
    spec_payload = visible_episode_spec()
    spec_payload["maximum_sample_gap_seconds"] = 0.011
    spec = parse_visible_episode_count_spec(spec_payload)
    assert isinstance(spec, VisibleEpisodeCountSpec)
    frame_ids = [f"F{index}" for index in range(frame_count)]
    inventory = {
        "schema_version": 1,
        "video_id": "long-video",
        "source_sha256": "a" * 64,
        "duration_seconds": 1.0,
        "decoded_frame_count": frame_count,
        "frames": [
            {
                "frame_id": frame_id,
                "decoded_frame_index": index,
                "timestamp_seconds": index / (frame_count - 1),
                "artifact_sha256": "b" * 64,
                "is_first_decoded_frame": index == 0,
                "is_last_decoded_frame": index == frame_count - 1,
            }
            for index, frame_id in enumerate(frame_ids)
        ],
    }
    presentations = [
        {
            "presentation_id": "sheet",
            "presentation_sha256": "c" * 64,
            "frame_ids": frame_ids,
            "mode": "numbered_contact_sheet",
        },
        *[
            {
                "presentation_id": f"individual-{index}",
                "presentation_sha256": "d" * 64,
                "frame_ids": [frame_id],
                "mode": "individual_frame",
            }
            for index, frame_id in enumerate(frame_ids)
        ],
    ]
    return build_frame_decision_request(spec, inventory, presentations)


def decision_payload(
    states: list[tuple[str, str]],
    *,
    request_sha256: str,
    pass_id: str = "primary",
    presentation_mode: str = "numbered_contact_sheet",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pass_id": pass_id,
        "request_complete_sha256": request_sha256,
        "presentation_mode": presentation_mode,
        "decisions": [
            {"frame_id": frame_id, "target_state": state} for frame_id, state in states
        ],
    }


def test_decisions_must_cover_every_required_frame_exactly_once() -> None:
    request = valid_request()

    decisions = parse_frame_decisions(
        request,
        decision_payload(
            [("F0", "present"), ("F1", "absent"), ("F2", "present")],
            request_sha256=request.complete_sha256,
        ),
    )

    assert isinstance(decisions, FrameDecisionSet)
    assert decisions.request_complete_sha256 == request.complete_sha256
    assert [item.frame_id for item in decisions.decisions] == ["F0", "F1", "F2"]


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate",
        "extra",
        "reordered",
        "unknown_state",
        "wrong_request_hash",
        "nested_label",
        "final_integer",
        "invalid_pass_id",
        "unbound_presentation_mode",
    ],
)
def test_frame_decisions_reject_nonexact_or_label_bearing_output(case: str) -> None:
    request = valid_request()
    payload = decision_payload(
        [("F0", "present"), ("F1", "absent"), ("F2", "present")],
        request_sha256=request.complete_sha256,
    )
    rows = payload["decisions"]
    assert isinstance(rows, list)

    if case == "missing":
        rows.pop()
    elif case == "duplicate":
        rows[1]["frame_id"] = "F0"
    elif case == "extra":
        rows.append({"frame_id": "F3", "target_state": "absent"})
    elif case == "reordered":
        rows.reverse()
    elif case == "unknown_state":
        rows[1]["target_state"] = "probably_absent"
    elif case == "wrong_request_hash":
        payload["request_complete_sha256"] = "0" * 64
    elif case == "nested_label":
        rows[1]["reference_label"] = "absent"
    elif case == "final_integer":
        payload["final_count"] = 2
    elif case == "invalid_pass_id":
        payload["pass_id"] = "self_consistency"
    elif case == "unbound_presentation_mode":
        payload["presentation_mode"] = "centered_triplet"

    with pytest.raises((TypeError, ValueError)):
        parse_frame_decisions(request, payload)


def test_small_canary_challenges_every_frame() -> None:
    request = valid_request()
    decisions = parse_frame_decisions(
        request,
        decision_payload(
            [("F0", "present"), ("F1", "absent"), ("F2", "present")],
            request_sha256=request.complete_sha256,
        ),
    )

    assert select_challenge_frames(request, decisions) == ("F0", "F1", "F2")


def test_large_request_challenges_endpoints_transitions_neighbors_and_interiors() -> (
    None
):
    request = request_for_frame_count(101)
    states = ["absent"] * 40 + ["present"] * 21 + ["absent"] * 40
    decisions = parse_frame_decisions(
        request,
        decision_payload(
            [
                (frame.frame_id, state)
                for frame, state in zip(request.frames, states, strict=True)
            ],
            request_sha256=request.complete_sha256,
        ),
    )

    assert select_challenge_frames(request, decisions) == (
        "F0",
        "F10",
        "F20",
        "F30",
        "F38",
        "F39",
        "F40",
        "F41",
        "F45",
        "F50",
        "F55",
        "F59",
        "F60",
        "F61",
        "F62",
        "F71",
        "F81",
        "F91",
        "F100",
    )


def test_large_challenge_parser_requires_exact_selected_subset() -> None:
    request = request_for_frame_count(101)
    primary = parse_frame_decisions(
        request,
        decision_payload(
            [(frame.frame_id, "absent") for frame in request.frames],
            request_sha256=request.complete_sha256,
        ),
    )
    selected = select_challenge_frames(request, primary)
    payload = decision_payload(
        [(frame_id, "absent") for frame_id in selected],
        request_sha256=request.complete_sha256,
        pass_id="challenge",
        presentation_mode="individual_frame",
    )

    parsed = parse_frame_decisions(request, payload, required_frame_ids=selected)
    assert tuple(decision.frame_id for decision in parsed.decisions) == selected

    payload["decisions"].pop()
    with pytest.raises(ValueError, match="exactly once"):
        parse_frame_decisions(request, payload, required_frame_ids=selected)


def decisions_for(
    request: FrameDecisionRequest, states: list[str], *, pass_id: str, mode: str
) -> FrameDecisionSet:
    return parse_frame_decisions(
        request,
        decision_payload(
            [
                (frame.frame_id, state)
                for frame, state in zip(request.frames, states, strict=True)
            ],
            request_sha256=request.complete_sha256,
            pass_id=pass_id,
            presentation_mode=mode,
        ),
    )


def test_reconciliation_counts_only_agreed_complete_frame_states() -> None:
    request = valid_request()
    primary = decisions_for(
        request,
        ["present", "absent", "present"],
        pass_id="primary",
        mode="numbered_contact_sheet",
    )
    challenge = decisions_for(
        request,
        ["present", "absent", "present"],
        pass_id="challenge",
        mode="individual_frame",
    )

    ledger = reconcile_frame_decisions(request, primary, challenge)
    result = reduce_presence_episodes(ledger)

    assert [frame.target_state for frame in ledger.frames] == [
        "present",
        "absent",
        "present",
    ]
    assert result.outcome == "exact"
    assert result.value == 2


def test_reconciliation_turns_disagreement_into_uncertain_and_abstains() -> None:
    request = valid_request()
    primary = decisions_for(
        request,
        ["present", "absent", "present"],
        pass_id="primary",
        mode="numbered_contact_sheet",
    )
    challenge = decisions_for(
        request,
        ["present", "present", "present"],
        pass_id="challenge",
        mode="individual_frame",
    )

    ledger = reconcile_frame_decisions(request, primary, challenge)
    result = reduce_presence_episodes(ledger)

    assert ledger.frames[1].target_state == "uncertain"
    assert result.outcome == "abstain"
    assert "non_binary_frame_decision" in result.reason_codes


def test_reconciliation_requires_different_primary_and_challenge_renderings() -> None:
    request = valid_request()
    primary = decisions_for(
        request,
        ["absent", "present", "absent"],
        pass_id="primary",
        mode="numbered_contact_sheet",
    )
    challenge = decisions_for(
        request,
        ["absent", "present", "absent"],
        pass_id="challenge",
        mode="numbered_contact_sheet",
    )

    with pytest.raises(ValueError, match="presentations must differ"):
        reconcile_frame_decisions(request, primary, challenge)


def test_split_at_cuts_abstains_until_cut_boundaries_are_bound() -> None:
    request = valid_request(cut_policy="split_at_cuts")
    primary = decisions_for(
        request,
        ["absent", "present", "absent"],
        pass_id="primary",
        mode="numbered_contact_sheet",
    )
    challenge = decisions_for(
        request,
        ["absent", "present", "absent"],
        pass_id="challenge",
        mode="individual_frame",
    )

    ledger = reconcile_frame_decisions(request, primary, challenge)
    result = reduce_presence_episodes(ledger)

    assert {frame.target_state for frame in ledger.frames} == {"uncertain"}
    assert result.outcome == "abstain"
