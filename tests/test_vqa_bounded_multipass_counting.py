from __future__ import annotations

import hashlib

import pytest

from trec_video_2026.vqa_bounded_multipass_counting import (
    CountResult,
    UnsupportedCountSpec,
    build_count_spec_prompt,
    build_coverage_prompt,
    build_evidence_prompt,
    build_refinement_request,
    count_result_to_dict,
    count_spec_to_dict,
    coverage_review_to_dict,
    evidence_ledger_to_dict,
    parse_count_spec,
    parse_coverage_review,
    parse_evidence,
    reduce_count,
)


QUESTION = "How many times does the person jump?"
QUESTION_SHA = hashlib.sha256(QUESTION.encode()).hexdigest()


def spec_payload(*, unit: str = "action_repetition", route: str = "temporal"):
    return {
        "schema_version": "bounded-multipass-count-spec-v1",
        "question_sha256": QUESTION_SHA,
        "target": "the person's completed jump",
        "unit": unit,
        "route": route,
        "scope": {"start_seconds": 0.0, "end_seconds": 4.0},
        "inclusions": ["a completed upward-and-downward jump cycle"],
        "exclusions": ["preparatory crouches"],
        "identity_policy": "observable_episode",
        "candidate_views": [
            {
                "view_id": "W0",
                "kind": "window",
                "start_seconds": 0.0,
                "end_seconds": 2.5,
            },
            {
                "view_id": "W1",
                "kind": "window",
                "start_seconds": 1.5,
                "end_seconds": 4.0,
            },
        ],
        "uncertainty": "none",
    }


def evidence_payload(*, records=None):
    return {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["W0", "W1"],
        "records": records
        if records is not None
        else [
            {
                "record_id": "E0",
                "view_ids": ["W0"],
                "start_seconds": 0.5,
                "peak_seconds": 0.8,
                "end_seconds": 1.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "one complete jump",
            },
            {
                "record_id": "E1",
                "view_ids": ["W1"],
                "start_seconds": 2.5,
                "peak_seconds": 2.8,
                "end_seconds": 3.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "one complete jump",
            },
        ],
    }


def review_payload(*, complete=True, refinement_request=None, disputes=None):
    return {
        "schema_version": "bounded-multipass-coverage-v1",
        "complete": complete,
        "reviewed_view_ids": ["W0", "W1"],
        "accepted_record_ids": ["E0", "E1"],
        "rejected_record_ids": [],
        "missing_view_ids": [],
        "disputes": disputes or [],
        "refinement_request": refinement_request,
        "notes": "all temporal scope reviewed",
    }


def static_spec_payload(*, unit: str = "static_cardinality"):
    payload = spec_payload(unit=unit, route="static")
    payload["candidate_views"] = [
        {"view_id": "S0", "kind": "shot_frame", "timestamp_seconds": 1.0}
    ]
    return payload


def object_record(record_id: str, *, shot_id: str, left: float):
    return {
        "record_id": record_id,
        "view_ids": ["S0"],
        "shot_id": shot_id,
        "timestamp_seconds": 1.0,
        "bbox_norm": [left, 0.1, left + 0.1, 0.4],
        "target_supported": True,
        "ambiguous": False,
        "descriptor": f"localized object {record_id}",
    }


@pytest.mark.parametrize(
    ("unit", "route"),
    [
        ("action_repetition", "temporal"),
        ("appearance_episode", "temporal"),
        ("static_cardinality", "static"),
        ("max_simultaneous", "temporal"),
        ("per_scene_sum", "static"),
    ],
)
def test_parse_count_spec_accepts_supported_units(unit: str, route: str) -> None:
    payload = spec_payload(unit=unit, route=route)
    if route == "static":
        payload["candidate_views"] = [
            {"view_id": "S0", "kind": "shot_frame", "timestamp_seconds": 1.0}
        ]
    parsed = parse_count_spec(payload, question=QUESTION, duration_seconds=4.0)
    assert parsed.unit == unit
    assert parsed.route == route
    assert parsed.question_sha256 == QUESTION_SHA


@pytest.mark.parametrize("unit", ["distinct_identity", "visible_part_ownership"])
def test_parse_count_spec_routes_unsupported_units_to_abstention(unit: str) -> None:
    payload = spec_payload(unit=unit)
    parsed = parse_count_spec(payload, question=QUESTION, duration_seconds=4.0)
    assert isinstance(parsed, UnsupportedCountSpec)
    assert parsed.reason == f"unsupported_unit:{unit}"


def test_parse_count_spec_rejects_question_hash_mismatch() -> None:
    payload = spec_payload()
    payload["question_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="question_sha256"):
        parse_count_spec(payload, question=QUESTION, duration_seconds=4.0)


def test_parse_count_spec_rejects_scalar_answer_and_label_leakage() -> None:
    payload = spec_payload()
    payload["answer"] = 2
    with pytest.raises(ValueError, match="answer|field"):
        parse_count_spec(payload, question=QUESTION, duration_seconds=4.0)


def test_parse_evidence_requires_every_requested_view() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    payload = evidence_payload()
    payload["reviewed_view_ids"] = ["W0"]
    with pytest.raises(ValueError, match="reviewed_view_ids"):
        parse_evidence(payload, spec=spec, requested_views=spec.candidate_views)


def test_parse_evidence_rejects_unbound_or_out_of_range_record() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    payload = evidence_payload()
    payload["records"][0]["view_ids"] = ["invented"]
    with pytest.raises(ValueError, match="view_ids"):
        parse_evidence(payload, spec=spec, requested_views=spec.candidate_views)


def test_parse_evidence_rejects_duplicate_record_ids() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    records = evidence_payload()["records"]
    records[1]["record_id"] = "E0"
    with pytest.raises(ValueError, match="record_id"):
        parse_evidence(
            evidence_payload(records=records),
            spec=spec,
            requested_views=spec.candidate_views,
        )


def test_parse_coverage_review_allows_one_targeted_refinement() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(), spec=spec, requested_views=spec.candidate_views
    )
    payload = review_payload(
        complete=False,
        disputes=[{"record_id": "E1", "reason": "possible merged jumps"}],
        refinement_request={
            "kind": "window",
            "start_seconds": 2.0,
            "end_seconds": 3.5,
            "reason": "inspect possible merge",
        },
    )
    review = parse_coverage_review(payload, spec=spec, ledger=ledger)
    refinement = build_refinement_request(review, requested_views=spec.candidate_views)
    assert refinement is not None
    assert refinement.kind == "window"
    assert refinement.start_seconds == 2.0
    assert refinement.end_seconds == 3.5


def test_complete_review_cannot_request_refinement() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(), spec=spec, requested_views=spec.candidate_views
    )
    payload = review_payload(
        refinement_request={
            "kind": "window",
            "start_seconds": 2.0,
            "end_seconds": 3.0,
            "reason": "contradictory request",
        }
    )
    with pytest.raises(ValueError, match="complete"):
        parse_coverage_review(payload, spec=spec, ledger=ledger)


def test_reduce_count_counts_two_disjoint_supported_events() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(), spec=spec, requested_views=spec.candidate_views
    )
    review = parse_coverage_review(review_payload(), spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result == CountResult(
        outcome="exact",
        value=2,
        reasons=(),
        selected_record_ids=("E0", "E1"),
    )


def test_reduce_count_abstains_when_coverage_is_incomplete() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(), spec=spec, requested_views=spec.candidate_views
    )
    payload = review_payload(complete=False)
    payload["missing_view_ids"] = ["W1"]
    payload["reviewed_view_ids"] = ["W0"]
    payload["accepted_record_ids"] = ["E0"]
    review = parse_coverage_review(payload, spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "abstain"
    assert "coverage_incomplete" in result.reasons


def test_reduce_appearance_episode_abstains_on_overlapping_records() -> None:
    spec = parse_count_spec(
        spec_payload(unit="appearance_episode"),
        question=QUESTION,
        duration_seconds=4.0,
    )
    records = evidence_payload()["records"]
    records[1]["start_seconds"] = 1.0
    ledger = parse_evidence(
        evidence_payload(records=records),
        spec=spec,
        requested_views=spec.candidate_views,
    )
    review = parse_coverage_review(review_payload(), spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "abstain"
    assert "overlapping_records" in result.reasons


def test_reduce_count_abstains_on_empty_evidence() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(records=[]), spec=spec, requested_views=spec.candidate_views
    )
    payload = review_payload()
    payload["accepted_record_ids"] = []
    review = parse_coverage_review(payload, spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "abstain"
    assert "empty_evidence" in result.reasons


def test_reduce_static_cardinality_counts_localized_objects_in_one_shot() -> None:
    spec = parse_count_spec(
        static_spec_payload(), question=QUESTION, duration_seconds=4.0
    )
    payload = {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["S0"],
        "records": [
            object_record("O0", shot_id="shot-0", left=0.1),
            object_record("O1", shot_id="shot-0", left=0.4),
            object_record("O2", shot_id="shot-0", left=0.7),
        ],
    }
    ledger = parse_evidence(payload, spec=spec, requested_views=spec.candidate_views)
    review = parse_coverage_review(
        {
            "schema_version": "bounded-multipass-coverage-v1",
            "complete": True,
            "reviewed_view_ids": ["S0"],
            "accepted_record_ids": ["O0", "O1", "O2"],
            "rejected_record_ids": [],
            "missing_view_ids": [],
            "disputes": [],
            "refinement_request": None,
            "notes": "one shot completely reviewed",
        },
        spec=spec,
        ledger=ledger,
    )
    assert reduce_count(spec, ledger, review).value == 3


def test_reduce_static_cardinality_abstains_across_multiple_shots() -> None:
    spec = parse_count_spec(
        static_spec_payload(), question=QUESTION, duration_seconds=4.0
    )
    payload = {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["S0"],
        "records": [
            object_record("O0", shot_id="shot-0", left=0.1),
            object_record("O1", shot_id="shot-1", left=0.4),
        ],
    }
    ledger = parse_evidence(payload, spec=spec, requested_views=spec.candidate_views)
    review = parse_coverage_review(
        {
            "schema_version": "bounded-multipass-coverage-v1",
            "complete": True,
            "reviewed_view_ids": ["S0"],
            "accepted_record_ids": ["O0", "O1"],
            "rejected_record_ids": [],
            "missing_view_ids": [],
            "disputes": [],
            "refinement_request": None,
            "notes": "multiple shots would require identity or sum semantics",
        },
        spec=spec,
        ledger=ledger,
    )
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "abstain"
    assert "multiple_static_shots" in result.reasons


def test_reduce_max_simultaneous_uses_interval_overlap() -> None:
    payload = spec_payload(unit="max_simultaneous")
    spec = parse_count_spec(payload, question=QUESTION, duration_seconds=4.0)
    records = [
        {
            "record_id": "P0",
            "view_ids": ["W0"],
            "start_seconds": 0.2,
            "peak_seconds": 1.0,
            "end_seconds": 2.0,
            "target_supported": True,
            "ambiguous": False,
            "descriptor": "first visible person",
        },
        {
            "record_id": "P1",
            "view_ids": ["W0", "W1"],
            "start_seconds": 1.0,
            "peak_seconds": 2.0,
            "end_seconds": 3.0,
            "target_supported": True,
            "ambiguous": False,
            "descriptor": "second visible person",
        },
        {
            "record_id": "P2",
            "view_ids": ["W1"],
            "start_seconds": 2.5,
            "peak_seconds": 3.0,
            "end_seconds": 3.5,
            "target_supported": True,
            "ambiguous": False,
            "descriptor": "third visible person",
        },
    ]
    ledger = parse_evidence(
        evidence_payload(records=records),
        spec=spec,
        requested_views=spec.candidate_views,
    )
    coverage = review_payload()
    coverage["accepted_record_ids"] = ["P0", "P1", "P2"]
    review = parse_coverage_review(coverage, spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "exact"
    assert result.value == 2


def test_reduce_action_repetitions_allows_distinct_overlapping_events() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    records = evidence_payload()["records"]
    records[1]["start_seconds"] = 1.0
    records[1]["peak_seconds"] = 1.2
    records[1]["end_seconds"] = 1.4
    ledger = parse_evidence(
        evidence_payload(records=records),
        spec=spec,
        requested_views=spec.candidate_views,
    )
    review = parse_coverage_review(review_payload(), spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)
    assert result.outcome == "exact"
    assert result.value == 2


def test_serializers_and_prompts_preserve_structured_non_scalar_contract() -> None:
    spec = parse_count_spec(spec_payload(), question=QUESTION, duration_seconds=4.0)
    ledger = parse_evidence(
        evidence_payload(), spec=spec, requested_views=spec.candidate_views
    )
    review = parse_coverage_review(review_payload(), spec=spec, ledger=ledger)
    result = reduce_count(spec, ledger, review)

    assert count_spec_to_dict(spec)["unit"] == "action_repetition"
    assert evidence_ledger_to_dict(ledger)["records"][0]["record_id"] == "E0"
    assert coverage_review_to_dict(review)["complete"] is True
    assert count_result_to_dict(result) == {
        "outcome": "exact",
        "value": 2,
        "reasons": [],
        "selected_record_ids": ["E0", "E1"],
    }

    count_spec_prompt = build_count_spec_prompt(question=QUESTION, duration_seconds=4.0)
    evidence_prompt = build_evidence_prompt(spec)
    coverage_prompt = build_coverage_prompt(spec, ledger)
    assert QUESTION in count_spec_prompt
    assert "Do not return a final count" in count_spec_prompt
    assert "one record per counted observable unit" in evidence_prompt
    assert "provisional count" not in coverage_prompt.lower()
    assert (
        "reference"
        not in " ".join([count_spec_prompt, evidence_prompt, coverage_prompt]).lower()
    )
