from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from trec_video_2026.vqa_frontier_map import (
    EVENT_MAP_PROMPT_VERSION,
    EVENT_MAP_SCHEMA,
    PROVIDER_EVENT_MAP_SCHEMA,
    EventRecord,
    FrontierMap,
    SupportSpan,
    audit_pre_reference_payload,
    build_event_map_prompt,
    mask_question_fields,
    summarize_map_gate,
    validate_event_map,
)


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_interpretation": {
            "meaning": "Identify what changes after the person opens the box.",
            "required_modalities": ["visual"],
            "required_evidence": "The box before and after opening.",
        },
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "keyframe_seconds": [1.0, 3.5],
                "scene": "A closed box is on a table.",
                "entities": ["person", "box", "table"],
                "actions": ["person approaches box"],
                "relations": ["box on table"],
                "state_before": "box closed",
                "state_change": "person opens box",
                "state_after": "box open",
                "visual_observations": ["lid moves from closed to open"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "question_relevance": "direct",
                "confidence": 0.95,
                "gaps": [],
            }
        ],
        "support_spans": [
            {
                "event_ids": ["E1"],
                "start_seconds": 2.5,
                "end_seconds": 4.0,
                "modalities": ["visual"],
                "claim": "The box is open after the action.",
            }
        ],
        "top1_answer": "The box is open.",
        "support_explanation": "E1 shows the lid opening and the final open state.",
        "could_be_wrong_because": [
            "The object inside the box is not clearly visible."
        ],
    }


def mutate_invalid(payload: dict[str, object], mutation: str) -> None:
    if mutation == "overlap":
        events = payload["events"]
        assert isinstance(events, list)
        second = copy.deepcopy(events[0])
        second["event_id"] = "E2"
        second["start_seconds"] = 3.0
        second["end_seconds"] = 4.0
        second["keyframe_seconds"] = [3.25]
        events.append(second)
    elif mutation == "out_of_bounds":
        events = payload["events"]
        assert isinstance(events, list)
        events[0]["end_seconds"] = 4.5
    elif mutation == "unknown_event":
        spans = payload["support_spans"]
        assert isinstance(spans, list)
        spans[0]["event_ids"] = ["E404"]
    elif mutation == "reference_key":
        payload["reference_answer"] = "The box is open."
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


def test_validate_event_map_accepts_complete_bounded_map() -> None:
    result = validate_event_map(valid_payload(), duration_seconds=4.0)

    assert result.top1_answer == "The box is open."
    assert result.support_spans[0].event_ids == ("E1",)
    assert result.events[0].keyframe_seconds == (1.0, 3.5)
    assert isinstance(result, FrontierMap)


@pytest.mark.parametrize(
    "mutation", ["overlap", "out_of_bounds", "unknown_event", "reference_key"]
)
def test_validate_event_map_rejects_contract_violations(mutation: str) -> None:
    payload = valid_payload()
    mutate_invalid(payload, mutation)

    with pytest.raises((TypeError, ValueError)):
        validate_event_map(payload, duration_seconds=4.0)


def test_validate_event_map_accepts_long_support_span_contained_within_cited_event() -> None:
    payload = valid_payload()
    support_spans = payload["support_spans"]
    assert isinstance(support_spans, list)
    support_spans[0]["start_seconds"] = 1.5

    result = validate_event_map(payload, duration_seconds=4.0)

    assert result.support_spans[0].start_seconds == 1.5
    assert result.support_spans[0].end_seconds == 4.0


def test_validate_event_map_rejects_support_outside_cited_event() -> None:
    payload = valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    events[0]["end_seconds"] = 3.5
    events[0]["keyframe_seconds"] = [1.0, 3.0]

    with pytest.raises(ValueError, match="contained"):
        validate_event_map(payload, duration_seconds=4.0)


def test_validate_event_map_rejects_support_crossing_gap_between_cited_events() -> None:
    payload = valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    second = copy.deepcopy(events[0])
    events[0]["end_seconds"] = 1.5
    events[0]["keyframe_seconds"] = [1.0, 1.25]
    second["event_id"] = "E2"
    second["start_seconds"] = 2.5
    second["end_seconds"] = 4.0
    second["keyframe_seconds"] = [2.75, 3.5]
    events.append(second)
    support_spans = payload["support_spans"]
    assert isinstance(support_spans, list)
    support_spans[0]["event_ids"] = ["E1", "E2"]
    support_spans[0]["start_seconds"] = 1.0
    support_spans[0]["end_seconds"] = 3.0

    with pytest.raises(ValueError, match="contained"):
        validate_event_map(payload, duration_seconds=4.0)


def test_contract_records_are_frozen_and_slot_based() -> None:
    result = validate_event_map(valid_payload(), duration_seconds=4.0)

    assert is_dataclass(result.events[0])
    assert is_dataclass(result.support_spans[0])
    assert EventRecord.__slots__
    assert SupportSpan.__slots__
    assert FrontierMap.__slots__
    with pytest.raises(FrozenInstanceError):
        result.top1_answer = "changed"


def test_schema_closes_objects_caps_arrays_and_bounds_numbers() -> None:
    def walk(node: object) -> None:
        assert isinstance(node, dict)
        node_type = node.get("type")
        if node_type == "object":
            assert node["additionalProperties"] is False
            for child in node["properties"].values():
                walk(child)
        elif node_type == "array":
            assert node["maxItems"] > 0
            walk(node["items"])
        elif node_type in {"integer", "number"}:
            assert node["minimum"] >= 0

    walk(EVENT_MAP_SCHEMA)


def test_provider_schema_uses_only_gemini_supported_constraints() -> None:
    unsupported = {"const", "minLength", "maxLength"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert not unsupported.intersection(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(PROVIDER_EVENT_MAP_SCHEMA)
    version = PROVIDER_EVENT_MAP_SCHEMA["properties"]["schema_version"]
    assert version["enum"] == [1]


def test_build_event_map_prompt_is_versioned_question_aware_and_provider_neutral() -> None:
    prompt = build_event_map_prompt("What changes after opening the box?", 4.0)

    assert EVENT_MAP_PROMPT_VERSION in prompt
    assert "What changes after opening the box?" in prompt
    assert "4.0" in prompt
    assert "EVENT_MAP_SCHEMA" not in prompt
    assert "OpenRouter" not in prompt
    assert "Reka" not in prompt
    assert "ten-pass" not in prompt
    assert '"additionalProperties": false' in prompt


def test_mask_question_fields_keeps_only_blind_event_evidence() -> None:
    payload = valid_payload()
    masked = mask_question_fields(payload)

    assert set(masked) == {"schema_version", "events"}
    assert set(masked["events"][0]) == {
        "event_id",
        "start_seconds",
        "end_seconds",
        "keyframe_seconds",
        "scene",
        "entities",
        "actions",
        "relations",
        "state_before",
        "state_change",
        "state_after",
        "visual_observations",
        "speech_observations",
        "sound_observations",
        "visible_text",
        "confidence",
        "gaps",
    }
    assert "question_relevance" not in masked["events"][0]
    assert "top1_answer" not in masked
    assert payload["question_interpretation"]


def test_pre_reference_audit_returns_canonical_hash_and_rejects_reference_fields() -> None:
    payload = valid_payload()
    audit = audit_pre_reference_payload(payload)

    assert audit["payload_sha256"]
    assert audit["references_loaded"] is False
    assert audit["label_free"] is True

    payload["nested"] = {"reference_answer": "private"}
    with pytest.raises(ValueError, match="reference"):
        audit_pre_reference_payload(payload)


def test_summarize_map_gate_reports_total_usable_category_and_strict_counts() -> None:
    rows = [
        {"category": category, "usable": index < 4}
        for category in (
            "temporal",
            "counting",
            "causal",
            "speech",
            "sound",
            "ocr_detail",
            "other_global",
        )
        for index in range(5)
    ]

    summary = summarize_map_gate(rows)

    assert summary["total_maps"] == 35
    assert summary["usable_maps"] == 28
    assert summary["gate_passed"] is True
    assert summary["strict_four_of_five"] is True
    assert summary["category_counts"]["temporal"] == {
        "total": 5,
        "usable": 4,
        "passes_minimum": True,
        "passes_strict": True,
    }

    rows[0]["usable"] = False
    assert summarize_map_gate(rows)["gate_passed"] is False
