from __future__ import annotations

import pytest

from trec_video_2026.vqa_gemini_counting_recovery import (
    TimeRange,
    build_challenger_prompt,
    build_locator_prompt,
    build_verifier_prompt,
    parse_atomic_ledger,
    parse_challenge,
    parse_locator,
    reduce_recovery_result,
    verification_envelope,
)


def locator_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "count_unit": "appearance_episode",
        "target": "ball appearance",
        "candidates": [
            {
                "candidate_id": "C1",
                "start_seconds": 0.2,
                "end_seconds": 0.8,
                "reason": "first appearance",
            },
            {
                "candidate_id": "C2",
                "start_seconds": 6.5,
                "end_seconds": 7.0,
                "reason": "last appearance",
            },
        ],
        "reviewed_spans": [{"start_seconds": 0.0, "end_seconds": 7.268}],
        "unresolved_spans": [],
    }


def ledger_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "instances": [
            {
                "instance_id": "I1",
                "start_seconds": 0.25,
                "peak_seconds": 0.5,
                "end_seconds": 0.75,
                "before_state": "no ball visible",
                "during_state": "ball visible",
                "after_state": "ball absent",
                "separation_evidence": "new absence-delimited episode",
                "predicate_supported": True,
                "ambiguous": False,
            },
            {
                "instance_id": "I2",
                "start_seconds": 6.55,
                "peak_seconds": 6.7,
                "end_seconds": 6.9,
                "before_state": "no ball visible",
                "during_state": "ball visible",
                "after_state": "ball absent",
                "separation_evidence": "new absence-delimited episode",
                "predicate_supported": True,
                "ambiguous": False,
            },
        ],
        "unresolved_observations": [],
    }


def challenge_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance_dispositions": [
            {"instance_id": "I1", "disposition": "accept", "reason": "distinct"},
            {"instance_id": "I2", "disposition": "accept", "reason": "distinct"},
        ],
        "missed_instances": [],
        "requested_ranges": [],
        "unresolved_reasons": [],
    }


def test_verification_envelope_unions_candidates_and_clips_padding() -> None:
    locator = parse_locator(locator_payload(), duration_seconds=7.268)

    assert verification_envelope(locator, 7.268) == TimeRange(0.0, 7.268)


def test_empty_locator_uses_complete_video_for_verification() -> None:
    payload = locator_payload()
    payload["candidates"] = []

    locator = parse_locator(payload, duration_seconds=7.268)

    assert verification_envelope(locator, 7.268) == TimeRange(0.0, 7.268)


def test_locator_rejects_duplicate_ids_and_label_keys() -> None:
    duplicate = locator_payload()
    duplicate["candidates"] = [duplicate["candidates"][0], duplicate["candidates"][0]]
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        parse_locator(duplicate, duration_seconds=7.268)

    leaked = locator_payload()
    leaked["reference_answer"] = "four"
    with pytest.raises(ValueError, match="label-bearing key"):
        parse_locator(leaked, duration_seconds=7.268)


def test_atomic_ledger_requires_ordered_grounded_instances() -> None:
    payload = ledger_payload()
    payload["instances"][0]["peak_seconds"] = 0.1

    with pytest.raises(ValueError, match="start <= peak <= end"):
        parse_atomic_ledger(
            payload, duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
        )


def test_reducer_counts_only_accepted_unambiguous_instances() -> None:
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    challenge = parse_challenge(
        challenge_payload(), ledger=ledger, duration_seconds=7.268
    )

    result = reduce_recovery_result(ledger, challenge, additive=None)

    assert result.outcome == "exact"
    assert result.value == 2
    assert result.accepted_instance_ids == ("I1", "I2")
    assert result.reason_codes == ()


def test_reducer_abstains_when_challenge_requests_more_video() -> None:
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    payload = challenge_payload()
    payload["requested_ranges"] = [
        {"start_seconds": 3.0, "end_seconds": 4.0, "reason": "possible missed event"}
    ]
    challenge = parse_challenge(payload, ledger=ledger, duration_seconds=7.268)

    result = reduce_recovery_result(ledger, challenge, additive=None)

    assert result.outcome == "abstain"
    assert result.value is None
    assert result.reason_codes == ("additional_verification_required",)


def test_reducer_does_not_count_challenger_discovery_without_additive_verification() -> (
    None
):
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    payload = challenge_payload()
    payload["missed_instances"] = [
        {
            "instance_id": "I3",
            "start_seconds": 3.0,
            "peak_seconds": 3.2,
            "end_seconds": 3.4,
            "before_state": "no ball visible",
            "during_state": "ball visible",
            "after_state": "ball absent",
            "separation_evidence": "challenger saw a possible episode",
            "predicate_supported": True,
            "ambiguous": False,
        }
    ]
    challenge = parse_challenge(payload, ledger=ledger, duration_seconds=7.268)

    result = reduce_recovery_result(ledger, challenge, additive=None)

    assert result.outcome == "abstain"
    assert result.value is None
    assert result.reason_codes == ("additional_verification_required",)


def test_additive_verification_replaces_challenger_proposal_without_double_counting() -> (
    None
):
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    payload = challenge_payload()
    proposal = {
        "instance_id": "I3",
        "start_seconds": 3.0,
        "peak_seconds": 3.2,
        "end_seconds": 3.4,
        "before_state": "no ball visible",
        "during_state": "ball visible",
        "after_state": "ball absent",
        "separation_evidence": "challenger saw a possible episode",
        "predicate_supported": True,
        "ambiguous": False,
    }
    payload["missed_instances"] = [proposal]
    challenge = parse_challenge(payload, ledger=ledger, duration_seconds=7.268)
    additive = parse_atomic_ledger(
        {
            "schema_version": 1,
            "instances": [{**proposal, "instance_id": "A1"}],
            "unresolved_observations": [],
        },
        duration_seconds=7.268,
        envelope=TimeRange(1.5, 4.9),
    )

    result = reduce_recovery_result(ledger, challenge, additive=additive)

    assert result.outcome == "exact"
    assert result.value == 3
    assert result.accepted_instance_ids == ("I1", "I2", "A1")


def test_ambiguous_additive_verification_cannot_authorize_exactness() -> None:
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    payload = challenge_payload()
    payload["requested_ranges"] = [
        {"start_seconds": 3.0, "end_seconds": 4.0, "reason": "possible episode"}
    ]
    challenge = parse_challenge(payload, ledger=ledger, duration_seconds=7.268)
    instance = ledger_payload()["instances"][0]
    additive = parse_atomic_ledger(
        {
            "schema_version": 1,
            "instances": [
                {
                    **instance,
                    "instance_id": "A1",
                    "start_seconds": 3.1,
                    "peak_seconds": 3.3,
                    "end_seconds": 3.5,
                    "ambiguous": True,
                }
            ],
            "unresolved_observations": [],
        },
        duration_seconds=7.268,
        envelope=TimeRange(1.5, 5.5),
    )

    result = reduce_recovery_result(ledger, challenge, additive=additive)

    assert result.outcome == "abstain"
    assert result.reason_codes == ("unresolved_additive_verification",)


def test_prompts_assign_distinct_jobs_without_answer_or_labels() -> None:
    locator = parse_locator(locator_payload(), duration_seconds=7.268)
    ledger = parse_atomic_ledger(
        ledger_payload(), duration_seconds=7.268, envelope=TimeRange(0.0, 7.268)
    )
    row = {
        "video_id": "fixture",
        "question": "How many times does the ball appear?",
        "duration_seconds": 7.268,
    }

    locator_prompt = build_locator_prompt(row)
    verifier_prompt = build_verifier_prompt(row, locator, TimeRange(0.0, 7.268))
    challenger_prompt = build_challenger_prompt(row, locator, ledger)

    assert "Do not answer" in locator_prompt
    assert "absolute source timestamps" in verifier_prompt
    assert "seek counterexamples" in challenger_prompt
    combined = locator_prompt + verifier_prompt + challenger_prompt
    assert "reference_answer" not in combined
    assert "panel_role" not in combined
