from __future__ import annotations

import copy
import inspect

import pytest

from trec_video_2026.vqa_frontier_phase0_contract import (
    GROUNDED_ANSWER_SCHEMA,
    PROVIDER_GROUNDED_ANSWER_SCHEMA,
    GroundedAnswer,
    NeutralVideoMap,
    build_grounded_answer_prompt,
    build_neutral_map_prompt,
    validate_grounded_answer,
    validate_neutral_map,
)


def neutral_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "perceived_duration_seconds": 20.2,
        "global_description": "A person handles a box on a table.",
        "entities": [
            {
                "entity_id": "P1",
                "description": "A person near the table.",
                "aliases": ["person"],
                "confidence": 0.95,
            },
            {
                "entity_id": "O1",
                "description": "A small box on the table.",
                "aliases": ["box"],
                "confidence": 0.9,
            },
        ],
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "keyframe_seconds": [1.0, 3.5],
                "description": "The person approaches the closed box.",
                "entity_ids": ["P1", "O1"],
                "actions": ["approaches"],
                "relations": ["box on table"],
                "state_before": "The box is closed.",
                "state_change": "The person reaches toward the box.",
                "state_after": "The box remains closed.",
                "visual_observations": ["The box is visible on the table."],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.9,
                "uncertainties": [],
            },
            {
                "event_id": "E2",
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "keyframe_seconds": [6.0, 9.5],
                "description": "The person opens the box.",
                "entity_ids": ["P1", "O1"],
                "actions": ["opens"],
                "relations": ["person holds box"],
                "state_before": "The box is closed.",
                "state_change": "The lid moves upward.",
                "state_after": "The box is open.",
                "visual_observations": [],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.85,
                "uncertainties": ["The inside of the box is partly obscured."],
            },
        ],
        "global_uncertainties": [],
    }


def answer_payload(*, map_event_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "top1_answer": "The person opens the box.",
        "claims": [
            {
                "claim_id": "C1",
                "text": "The person opens the box.",
                "supporting_spans": [
                    {"start_seconds": 5.0, "end_seconds": 9.5}
                ],
                "supporting_modalities": ["video"],
                "map_event_ids": map_event_ids,
            }
        ],
        "could_be_wrong_because": ["The view of the opening is brief."],
    }


def test_neutral_prompt_has_no_question_channel() -> None:
    prompt = build_neutral_map_prompt(18.25)

    assert "18.25" in prompt
    assert list(inspect.signature(build_neutral_map_prompt).parameters) == [
        "duration_seconds"
    ]
    assert "sentinel private question" not in prompt.lower()
    assert "Question:" not in prompt
    assert "question-blind" in prompt.lower()
    assert '"additionalProperties": false' in prompt


def test_neutral_map_accepts_empty_observation_arrays_and_freezes_records() -> None:
    result = validate_neutral_map(neutral_payload(), 20.0)

    assert isinstance(result, NeutralVideoMap)
    assert result.events[1].visual_observations == ()
    assert result.events[0].event_id == "E1"
    assert result.entities[0].entity_id == "P1"


def test_neutral_map_rejects_question_and_answer_fields() -> None:
    payload = neutral_payload()
    payload["question"] = "sentinel private question"

    with pytest.raises(ValueError, match="forbidden field"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_rejects_unknown_fields_recursively() -> None:
    payload = neutral_payload()
    entity = payload["entities"][0]
    assert isinstance(entity, dict)
    entity["extra"] = {"nested": True}

    with pytest.raises(ValueError, match="unexpected"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_requires_unique_event_and_entity_ids() -> None:
    payload = neutral_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[1]["event_id"] = events[0]["event_id"]

    with pytest.raises(ValueError, match="event IDs must be unique"):
        validate_neutral_map(payload, 20.0)

    payload = neutral_payload()
    entities = payload["entities"]
    assert isinstance(entities, list)
    entities[1]["entity_id"] = entities[0]["entity_id"]

    with pytest.raises(ValueError, match="entity IDs must be unique"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_rejects_missing_entity_references() -> None:
    payload = neutral_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[0]["entity_ids"] = ["E404"]

    with pytest.raises(ValueError, match="unknown entity"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_rejects_unordered_nonfinite_and_out_of_duration_values() -> None:
    payload = neutral_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[1]["start_seconds"] = 0.0
    events[1]["end_seconds"] = 2.0
    events[1]["keyframe_seconds"] = [1.0, 1.5]
    with pytest.raises(ValueError, match="chronological"):
        validate_neutral_map(payload, 20.0)

    payload = neutral_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[0]["start_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_neutral_map(payload, 20.0)

    payload = neutral_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[1]["end_seconds"] = 20.1
    events[1]["keyframe_seconds"] = [6.0, 20.1]
    with pytest.raises(ValueError, match="duration"):
        validate_neutral_map(payload, 20.0)


def test_neutral_map_rejects_perceived_duration_outside_half_second() -> None:
    payload = neutral_payload()
    payload["perceived_duration_seconds"] = 20.51

    with pytest.raises(ValueError, match="perceived_duration_seconds"):
        validate_neutral_map(payload, 20.0)


def test_contract_rejects_non_integer_schema_versions() -> None:
    payload = neutral_payload()
    payload["schema_version"] = 1.0

    with pytest.raises(ValueError, match="schema_version"):
        validate_neutral_map(payload, 20.0)


def test_direct_and_guided_use_the_same_schema() -> None:
    assert set(GROUNDED_ANSWER_SCHEMA["properties"]) == {
        "schema_version",
        "top1_answer",
        "claims",
        "could_be_wrong_because",
    }
    assert set(PROVIDER_GROUNDED_ANSWER_SCHEMA["properties"]) == set(
        GROUNDED_ANSWER_SCHEMA["properties"]
    )


def test_direct_answer_requires_empty_map_event_ids() -> None:
    payload = answer_payload(map_event_ids=["E1"])

    with pytest.raises(ValueError, match="direct answer map_event_ids must be empty"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_guided_answer_rejects_unknown_map_event_id() -> None:
    payload = answer_payload(map_event_ids=["E9"])

    with pytest.raises(ValueError, match="unknown map event"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids={"E1"})


def test_grounded_answer_accepts_direct_and_guided_forms() -> None:
    direct = validate_grounded_answer(
        answer_payload(map_event_ids=[]), 20.0, allowed_map_event_ids=None
    )
    guided = validate_grounded_answer(
        answer_payload(map_event_ids=["E1"]), 20.0, allowed_map_event_ids={"E1"}
    )

    assert isinstance(direct, GroundedAnswer)
    assert direct.claims[0].map_event_ids == ()
    assert guided.claims[0].map_event_ids == ("E1",)


def test_guided_answer_requires_each_claim_to_cite_a_map_event() -> None:
    payload = answer_payload(map_event_ids=[])

    with pytest.raises(ValueError, match="must cite at least one map event"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids={"E1"})


def test_grounded_answer_rejects_unknown_nested_fields_and_bad_spans() -> None:
    payload = answer_payload(map_event_ids=[])
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0]["supporting_spans"][0]["extra"] = "not allowed"

    with pytest.raises(ValueError, match="unexpected"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)

    payload = answer_payload(map_event_ids=[])
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0]["supporting_spans"][0]["end_seconds"] = 20.1

    with pytest.raises(ValueError, match="duration"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_grounded_answer_rejects_empty_supporting_modalities() -> None:
    payload = answer_payload(map_event_ids=[])
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0]["supporting_modalities"] = []

    with pytest.raises(ValueError, match="supporting_modalities must contain at least 1"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_neutral_map_enforces_declared_string_max_length() -> None:
    payload = neutral_payload()
    entities = payload["entities"]
    assert isinstance(entities, list)
    entities[0]["description"] = "x" * 4097

    with pytest.raises(ValueError, match="maximum length"):
        validate_neutral_map(payload, 20.0)


def test_grounded_answer_enforces_declared_string_max_length() -> None:
    payload = answer_payload(map_event_ids=[])
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0]["text"] = "x" * 4097

    with pytest.raises(ValueError, match="maximum length"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_neutral_map_enforces_entity_array_cap() -> None:
    payload = neutral_payload()
    entities = payload["entities"]
    assert isinstance(entities, list)
    entities.extend(
        {
            "entity_id": f"X{index}",
            "description": "an extra entity",
            "aliases": [],
            "confidence": 0.5,
        }
        for index in range(127)
    )

    with pytest.raises(ValueError, match="entities cannot contain more than 128"):
        validate_neutral_map(payload, 20.0)


def test_grounded_answer_enforces_claim_array_cap() -> None:
    payload = answer_payload(map_event_ids=[])
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims.extend(
        {
            "claim_id": f"C{index}",
            "text": "The person opens the box.",
            "supporting_spans": [{"start_seconds": 5.0, "end_seconds": 9.5}],
            "supporting_modalities": ["video"],
            "map_event_ids": [],
        }
        for index in range(2, 34)
    )

    with pytest.raises(ValueError, match="claims cannot contain more than 32"):
        validate_grounded_answer(payload, 20.0, allowed_map_event_ids=None)


def test_grounded_answer_prompt_shares_instructions_and_binds_map_mode() -> None:
    direct = build_grounded_answer_prompt("What changes?", 20.0, None)
    guided = build_grounded_answer_prompt(
        "What changes?", 20.0, neutral_payload()
    )

    assert "frontier-grounded-answer-v1" in direct
    assert '"additionalProperties": false' in direct
    assert "Question: What changes?" in direct
    assert "Source duration seconds: 20.0" in direct
    assert "Neutral map: not supplied" in direct
    assert "map_event_ids" in direct
    assert "empty" in direct.lower()
    assert "Neutral map: not supplied" not in guided
    assert '"event_id":"E1"' in guided
    assert "declared event IDs" in guided
    assert "native timestamp spans" in guided
    assert "unsupported details" in guided.lower()


def test_grounded_answer_prompt_rejects_invalid_raw_neutral_map() -> None:
    payload = neutral_payload()
    payload["question"] = "sentinel private question"

    with pytest.raises(ValueError, match="forbidden field"):
        build_grounded_answer_prompt("What changes?", 20.0, payload)


def test_contract_fixtures_are_not_mutated_by_validation() -> None:
    payload = neutral_payload()
    original = copy.deepcopy(payload)

    validate_neutral_map(payload, 20.0)

    assert payload == original
