from __future__ import annotations

import pytest

from trec_video_2026.vqa_counting_evidence import (
    ATOMIC_COUNT_UNITS,
    build_atomic_evidence_prompt,
    classify_count_row,
    derive_count,
    parse_prediction_count,
    parse_reference_count,
    strip_timestamps,
    validate_atomic_evidence,
    validate_label_blind,
)


def atomic_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "count_unit": "repeated_actions_transitions",
        "target": "passes",
        "scope": "complete video",
        "records": [
            {
                "record_id": "R1",
                "start_seconds": 1.25,
                "end_seconds": 1.75,
                "scene": "first pass",
                "rationale": "one completed pass",
            },
            {
                "record_id": "R2",
                "start_seconds": 4.0,
                "end_seconds": 4.5,
                "scene": "second pass",
                "rationale": "a distinct completed pass",
            },
        ],
        "exclusions": [
            {
                "start_seconds": 2.0,
                "end_seconds": 2.25,
                "reason": "replay of R1 rather than a new pass",
            }
        ],
        "uncertainty": "none",
    }


def test_timestamp_stripping_does_not_turn_zero_minutes_into_an_answer() -> None:
    text = "SCENE 00:06.000-00:11.500. Therefore, there are 4 appearances."
    assert strip_timestamps(text) == "SCENE . Therefore, there are 4 appearances."
    assert parse_prediction_count(text).value == 4


def test_reference_parser_uses_leading_answer_not_later_explanatory_numbers() -> None:
    text = "Two, one under the lower back and one under the neck."
    assert parse_reference_count(text).value == 2


def test_reference_parser_supports_compound_number_words_beyond_twenty() -> None:
    assert parse_reference_count("forty").value == 40
    assert parse_reference_count("forty-five fingers").value == 45
    assert parse_reference_count("twice").value == 2


def test_prediction_parser_marks_or_alternative_as_noncommittal() -> None:
    parsed = parse_prediction_count("There are 2 or 3 people carrying the mat.")
    assert parsed.value is None
    assert parsed.candidates == (2, 3)
    assert parsed.committed is False


def test_count_census_distinguishes_wrong_ambiguous_and_not_count() -> None:
    assert classify_count_row("How many fingers?", "forty", "34").status == "wrong"
    assert (
        classify_count_row(
            "How many beds are there?",
            "One double bunk bed",
            "2",
            adjudication="ambiguous",
        ).status
        == "ambiguous"
    )
    assert (
        classify_count_row(
            "The number of arrows change depending on what?",
            "The arithmetic operation",
            "The gates",
        ).status
        == "not_count"
    )


def test_label_blind_validator_rejects_forbidden_keys_at_any_depth() -> None:
    validate_label_blind({"question": "How many?", "map": "scene map"})
    with pytest.raises(ValueError, match="forbidden label-bearing key.*reference"):
        validate_label_blind({"request": {"reference_answer": "4"}})
    with pytest.raises(ValueError, match="forbidden label-bearing key.*panel_role"):
        validate_label_blind({"rows": [{"panel_role": "control"}]})


def test_prompt_has_no_reference_or_prior_answer_channel() -> None:
    prompt = build_atomic_evidence_prompt(
        question="How many passes occur?",
        duration_seconds=8.5,
        frozen_map="SCENE | 0-8.5 | a person passes an object",
    )
    assert "How many passes occur?" in prompt
    assert "8.500" in prompt
    assert "reference_answer" not in prompt
    assert "prior_answer" not in prompt
    assert "final_count" not in prompt
    assert "one JSON record for every counted unit" in prompt


def test_atomic_validator_accepts_all_declared_count_units() -> None:
    for unit in ATOMIC_COUNT_UNITS:
        payload = atomic_payload()
        payload["count_unit"] = unit
        assert (
            validate_atomic_evidence(payload, duration_seconds=8.0).count_unit == unit
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["records"][1].update(record_id="R1"), "record IDs must be unique"),
        (lambda p: p["records"][0].update(start_seconds=-0.1), "timestamp"),
        (lambda p: p["records"][0].update(end_seconds=9.0), "duration"),
        (
            lambda p: p["records"][0].update(start_seconds=2.0, end_seconds=1.0),
            "start_seconds",
        ),
        (lambda p: p.update(final_count=2), "unknown root fields"),
        (lambda p: p["records"][0].update(count=1), "unknown record fields"),
    ],
)
def test_atomic_validator_rejects_non_atomic_or_unbound_outputs(
    mutation, message
) -> None:
    payload = atomic_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        validate_atomic_evidence(payload, duration_seconds=8.0)


def test_deterministic_reducer_counts_records_not_model_summary_text() -> None:
    evidence = validate_atomic_evidence(atomic_payload(), duration_seconds=8.0)
    assert derive_count(evidence) == 2
