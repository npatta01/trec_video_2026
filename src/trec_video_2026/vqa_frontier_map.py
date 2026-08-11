"""Provider-neutral one-pass event/state-map contract for native-AV VQA."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from trec_video_2026.vqa_submission import canonical_sha256


SCHEMA_VERSION = 1
EVENT_MAP_PROMPT_VERSION = "frontier-event-map-v1"
MAX_EVENTS = 12
MAX_SUPPORT_SPANS = 3
_MAX_TEXT_ITEMS = 16
_MAX_GAPS = 8
_MAX_UNCERTAINTIES = 5
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "question_interpretation",
        "events",
        "support_spans",
        "top1_answer",
        "support_explanation",
        "could_be_wrong_because",
    }
)
_QUESTION_FIELDS = frozenset(
    {"meaning", "required_modalities", "required_evidence"}
)
_EVENT_FIELDS = frozenset(
    {
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
        "question_relevance",
        "confidence",
        "gaps",
    }
)
_SUPPORT_FIELDS = frozenset(
    {"event_ids", "start_seconds", "end_seconds", "modalities", "claim"}
)
_MASKED_EVENT_FIELDS = (
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
)
_REFERENCE_FIELD_TOKENS = frozenset(
    {"annotation", "gold", "label", "metric", "oracle", "reference", "score"}
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _string_schema(*, max_length: int = 1024) -> dict[str, object]:
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def _number_schema(*, maximum: float | None = None) -> dict[str, object]:
    schema: dict[str, object] = {"type": "number", "minimum": 0}
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _text_array_schema(
    *, min_items: int = 0, max_items: int = _MAX_TEXT_ITEMS
) -> dict[str, object]:
    return {
        "type": "array",
        "items": _string_schema(),
        "minItems": min_items,
        "maxItems": max_items,
    }


def _question_interpretation_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "meaning": _string_schema(),
            "required_modalities": _text_array_schema(min_items=1, max_items=4),
            "required_evidence": _string_schema(),
        },
        "required": sorted(_QUESTION_FIELDS),
        "additionalProperties": False,
    }


def _event_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "event_id": _string_schema(max_length=64),
            "start_seconds": _number_schema(),
            "end_seconds": _number_schema(),
            "keyframe_seconds": {
                "type": "array",
                "items": _number_schema(),
                "minItems": 1,
                "maxItems": MAX_EVENTS,
            },
            "scene": _string_schema(),
            "entities": _text_array_schema(),
            "actions": _text_array_schema(),
            "relations": _text_array_schema(),
            "state_before": _string_schema(),
            "state_change": _string_schema(),
            "state_after": _string_schema(),
            "visual_observations": _text_array_schema(),
            "speech_observations": _text_array_schema(),
            "sound_observations": _text_array_schema(),
            "visible_text": _text_array_schema(),
            "question_relevance": _string_schema(max_length=128),
            "confidence": _number_schema(maximum=1),
            "gaps": _text_array_schema(max_items=_MAX_GAPS),
        },
        "required": sorted(_EVENT_FIELDS),
        "additionalProperties": False,
    }


def _support_span_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "event_ids": {
                "type": "array",
                "items": _string_schema(max_length=64),
                "minItems": 1,
                "maxItems": MAX_EVENTS,
            },
            "start_seconds": _number_schema(),
            "end_seconds": _number_schema(),
            "modalities": _text_array_schema(min_items=1, max_items=4),
            "claim": _string_schema(),
        },
        "required": sorted(_SUPPORT_FIELDS),
        "additionalProperties": False,
    }


EVENT_MAP_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "minimum": 0, "const": SCHEMA_VERSION},
        "question_interpretation": _question_interpretation_schema(),
        "events": {
            "type": "array",
            "items": _event_schema(),
            "minItems": 1,
            "maxItems": MAX_EVENTS,
        },
        "support_spans": {
            "type": "array",
            "items": _support_span_schema(),
            "minItems": 1,
            "maxItems": MAX_SUPPORT_SPANS,
        },
        "top1_answer": _string_schema(),
        "support_explanation": _string_schema(),
        "could_be_wrong_because": {
            "type": "array",
            "items": _string_schema(),
            "minItems": 1,
            "maxItems": _MAX_UNCERTAINTIES,
        },
    },
    "required": sorted(_ROOT_FIELDS),
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    start_seconds: float
    end_seconds: float
    keyframe_seconds: tuple[float, ...]
    scene: str
    entities: tuple[str, ...]
    actions: tuple[str, ...]
    relations: tuple[str, ...]
    state_before: str
    state_change: str
    state_after: str
    visual_observations: tuple[str, ...]
    speech_observations: tuple[str, ...]
    sound_observations: tuple[str, ...]
    visible_text: tuple[str, ...]
    question_relevance: str
    confidence: float
    gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupportSpan:
    event_ids: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    modalities: tuple[str, ...]
    claim: str


@dataclass(frozen=True, slots=True)
class FrontierMap:
    schema_version: int
    question_interpretation: Mapping[str, object]
    events: tuple[EventRecord, ...]
    support_spans: tuple[SupportSpan, ...]
    top1_answer: str
    support_explanation: str
    could_be_wrong_because: tuple[str, ...]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")
            frozen[key] = _freeze_json(nested)
        return MappingProxyType(frozen)
    if isinstance(value, (str, bytes)):
        return copy.deepcopy(value)
    if isinstance(value, Sequence):
        return tuple(_freeze_json(item) for item in value)
    return copy.deepcopy(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return copy.deepcopy(value)


def _require_mapping(value: Any, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    keys = frozenset(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        problems: list[str] = []
        if missing:
            problems.append(f"missing {missing}")
        if extra:
            problems.append(f"unexpected {extra}")
        joined = "; ".join(problems)
        raise ValueError(f"{field} violates the contract: {joined}")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sequence(value: Any, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be an array")
    return value


def _require_number(value: Any, field: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number < 0:
        raise ValueError(f"{field} cannot be negative")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} cannot exceed {maximum}")
    return number


def _require_text_tuple(
    value: Any,
    field: str,
    *,
    min_items: int = 0,
    max_items: int = _MAX_TEXT_ITEMS,
    unique: bool = False,
) -> tuple[str, ...]:
    values = tuple(_require_text(item, f"{field}[{index}]") for index, item in enumerate(_require_sequence(value, field)))
    if len(values) < min_items:
        raise ValueError(f"{field} must contain at least {min_items} item(s)")
    if len(values) > max_items:
        raise ValueError(f"{field} cannot contain more than {max_items} item(s)")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _require_number_tuple(
    value: Any,
    field: str,
    *,
    min_items: int = 0,
    max_items: int = MAX_EVENTS,
) -> tuple[float, ...]:
    values = tuple(
        _require_number(item, f"{field}[{index}]")
        for index, item in enumerate(_require_sequence(value, field))
    )
    if len(values) < min_items:
        raise ValueError(f"{field} must contain at least {min_items} item(s)")
    if len(values) > max_items:
        raise ValueError(f"{field} cannot contain more than {max_items} item(s)")
    return values


def _key_tokens(key: str) -> tuple[str, ...]:
    separated = _CAMEL_BOUNDARY.sub("_", key)
    return tuple(token for token in re.split(r"[^A-Za-z0-9]+", separated.lower()) if token)


def _walk_reference_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            tokens = _key_tokens(str(key))
            if _REFERENCE_FIELD_TOKENS.intersection(tokens):
                raise ValueError(f"reference-like field is not allowed: {path}.{key}")
            _walk_reference_fields(nested, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _walk_reference_fields(nested, f"{path}[{index}]")


def _parse_question_interpretation(value: Any) -> Mapping[str, object]:
    mapping = _require_mapping(value, "question_interpretation")
    _require_exact_fields(mapping, _QUESTION_FIELDS, "question_interpretation")
    frozen = {
        "meaning": _require_text(mapping["meaning"], "question_interpretation.meaning"),
        "required_modalities": _require_text_tuple(
            mapping["required_modalities"],
            "question_interpretation.required_modalities",
            min_items=1,
            max_items=4,
            unique=True,
        ),
        "required_evidence": _require_text(
            mapping["required_evidence"], "question_interpretation.required_evidence"
        ),
    }
    return MappingProxyType(frozen)


def _parse_event(value: Any, field: str) -> EventRecord:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, _EVENT_FIELDS, field)
    start_seconds = _require_number(mapping["start_seconds"], f"{field}.start_seconds")
    end_seconds = _require_number(mapping["end_seconds"], f"{field}.end_seconds")
    if end_seconds <= start_seconds:
        raise ValueError(f"{field}.end_seconds must be greater than start_seconds")
    keyframes = _require_number_tuple(
        mapping["keyframe_seconds"], f"{field}.keyframe_seconds", min_items=1
    )
    if any(keyframe < start_seconds or keyframe > end_seconds for keyframe in keyframes):
        raise ValueError(f"{field}.keyframe_seconds must fall within the event bounds")
    return EventRecord(
        event_id=_require_text(mapping["event_id"], f"{field}.event_id"),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        keyframe_seconds=keyframes,
        scene=_require_text(mapping["scene"], f"{field}.scene"),
        entities=_require_text_tuple(mapping["entities"], f"{field}.entities"),
        actions=_require_text_tuple(mapping["actions"], f"{field}.actions"),
        relations=_require_text_tuple(mapping["relations"], f"{field}.relations"),
        state_before=_require_text(mapping["state_before"], f"{field}.state_before"),
        state_change=_require_text(mapping["state_change"], f"{field}.state_change"),
        state_after=_require_text(mapping["state_after"], f"{field}.state_after"),
        visual_observations=_require_text_tuple(
            mapping["visual_observations"], f"{field}.visual_observations"
        ),
        speech_observations=_require_text_tuple(
            mapping["speech_observations"], f"{field}.speech_observations"
        ),
        sound_observations=_require_text_tuple(
            mapping["sound_observations"], f"{field}.sound_observations"
        ),
        visible_text=_require_text_tuple(
            mapping["visible_text"], f"{field}.visible_text"
        ),
        question_relevance=_require_text(
            mapping["question_relevance"], f"{field}.question_relevance"
        ),
        confidence=_require_number(
            mapping["confidence"], f"{field}.confidence", maximum=1.0
        ),
        gaps=_require_text_tuple(
            mapping["gaps"], f"{field}.gaps", max_items=_MAX_GAPS
        ),
    )


def _parse_support_span(value: Any, field: str) -> SupportSpan:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, _SUPPORT_FIELDS, field)
    start_seconds = _require_number(mapping["start_seconds"], f"{field}.start_seconds")
    end_seconds = _require_number(mapping["end_seconds"], f"{field}.end_seconds")
    if end_seconds <= start_seconds:
        raise ValueError(f"{field}.end_seconds must be greater than start_seconds")
    return SupportSpan(
        event_ids=_require_text_tuple(
            mapping["event_ids"], f"{field}.event_ids", min_items=1, max_items=MAX_EVENTS, unique=True
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        modalities=_require_text_tuple(
            mapping["modalities"], f"{field}.modalities", min_items=1, max_items=4, unique=True
        ),
        claim=_require_text(mapping["claim"], f"{field}.claim"),
    )


def _validate_event_order(events: tuple[EventRecord, ...], *, duration_seconds: float) -> None:
    seen_ids: set[str] = set()
    previous_end = -1.0
    for index, event in enumerate(events):
        if event.event_id in seen_ids:
            raise ValueError(f"events[{index}].event_id must be unique")
        seen_ids.add(event.event_id)
        if event.end_seconds > duration_seconds:
            raise ValueError(f"events[{index}] exceeds the probed duration")
        if event.start_seconds < previous_end:
            raise ValueError("events must be chronologically ordered and non-overlapping")
        previous_end = event.end_seconds


def _validate_support_spans(
    support_spans: tuple[SupportSpan, ...],
    *,
    events_by_id: Mapping[str, EventRecord],
    duration_seconds: float,
) -> None:
    for index, span in enumerate(support_spans):
        if span.end_seconds > duration_seconds:
            raise ValueError(f"support_spans[{index}] exceeds the probed duration")
        cited_events: list[EventRecord] = []
        for event_id in span.event_ids:
            event = events_by_id.get(event_id)
            if event is None:
                raise ValueError(f"support_spans[{index}] cites unknown event_id {event_id}")
            cited_events.append(event)
        earliest = min(event.start_seconds for event in cited_events)
        latest = max(event.end_seconds for event in cited_events)
        if span.start_seconds < earliest or span.end_seconds > latest:
            raise ValueError(
                f"support_spans[{index}] must be contained within its cited events"
            )
        cited_duration = latest - earliest
        span_duration = span.end_seconds - span.start_seconds
        max_focused_duration = max(2.0, cited_duration / 2.0)
        if span_duration > max_focused_duration + 1e-9:
            raise ValueError(
                f"support_spans[{index}] must be contained within a focused portion of its cited events"
            )


def build_event_map_prompt(question: str, duration_seconds: float) -> str:
    """Build the one-pass, question-aware prompt for the event/state map."""

    prompt_question = _require_text(question, "question")
    prompt_duration = _require_number(duration_seconds, "duration_seconds")
    schema_text = json.dumps(EVENT_MAP_SCHEMA, ensure_ascii=False, sort_keys=True)
    return (
        f"{EVENT_MAP_PROMPT_VERSION}\n"
        "Produce one question-aware native-audio/video pass as strict JSON.\n"
        "Do not include chain-of-thought, provider labels, reference labels, or multiple candidate answers.\n"
        f"Question: {prompt_question}\n"
        f"Source duration seconds: {prompt_duration}\n"
        "Return a chronological event/state map with support spans, one top-1 answer, and explicit uncertainty.\n"
        "The JSON must satisfy this schema exactly:\n"
        f"{schema_text}\n"
    )


def validate_event_map(payload: Any, duration_seconds: float) -> FrontierMap:
    """Validate and freeze a structured event/state map payload."""

    _walk_reference_fields(payload)
    duration = _require_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    mapping = _require_mapping(payload, "payload")
    _require_exact_fields(mapping, _ROOT_FIELDS, "payload")
    schema_version = mapping["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    question_interpretation = _parse_question_interpretation(
        mapping["question_interpretation"]
    )
    events = tuple(
        _parse_event(item, f"events[{index}]")
        for index, item in enumerate(_require_sequence(mapping["events"], "events"))
    )
    if not events:
        raise ValueError("events must not be empty")
    if len(events) > MAX_EVENTS:
        raise ValueError(f"events cannot contain more than {MAX_EVENTS} item(s)")
    _validate_event_order(events, duration_seconds=duration)
    events_by_id = {event.event_id: event for event in events}
    support_spans = tuple(
        _parse_support_span(item, f"support_spans[{index}]")
        for index, item in enumerate(
            _require_sequence(mapping["support_spans"], "support_spans")
        )
    )
    if not support_spans:
        raise ValueError("support_spans must not be empty")
    if len(support_spans) > MAX_SUPPORT_SPANS:
        raise ValueError(
            f"support_spans cannot contain more than {MAX_SUPPORT_SPANS} item(s)"
        )
    _validate_support_spans(
        support_spans, events_by_id=events_by_id, duration_seconds=duration
    )
    return FrontierMap(
        schema_version=SCHEMA_VERSION,
        question_interpretation=question_interpretation,
        events=events,
        support_spans=support_spans,
        top1_answer=_require_text(mapping["top1_answer"], "top1_answer"),
        support_explanation=_require_text(
            mapping["support_explanation"], "support_explanation"
        ),
        could_be_wrong_because=_require_text_tuple(
            mapping["could_be_wrong_because"],
            "could_be_wrong_because",
            min_items=1,
            max_items=_MAX_UNCERTAINTIES,
        ),
    )


def mask_question_fields(map_payload: Any) -> dict[str, object]:
    """Remove question-aware and answer-bearing fields for blind-fidelity audit."""

    if isinstance(map_payload, FrontierMap):
        events = [
            {
                field: _thaw_json(getattr(event, field))
                for field in _MASKED_EVENT_FIELDS
            }
            for event in map_payload.events
        ]
        return {"schema_version": map_payload.schema_version, "events": events}
    mapping = _require_mapping(map_payload, "map_payload")
    masked_events: list[dict[str, object]] = []
    for index, event_value in enumerate(_require_sequence(mapping["events"], "events")):
        event = _require_mapping(event_value, f"events[{index}]")
        masked_events.append(
            {field: _thaw_json(event[field]) for field in _MASKED_EVENT_FIELDS}
        )
    return {
        "schema_version": copy.deepcopy(mapping["schema_version"]),
        "events": masked_events,
    }


def audit_pre_reference_payload(payload: Any) -> dict[str, object]:
    """Record a label-free audit marker before references are revealed."""

    _walk_reference_fields(payload)
    return {
        "payload_sha256": canonical_sha256(payload),
        "references_loaded": False,
        "label_free": True,
    }


def summarize_map_gate(verdict_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize the usable-map gate across the 35-row diagnostic panel."""

    category_counts: dict[str, dict[str, object]] = {}
    usable_maps = 0
    for index, row in enumerate(verdict_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"verdict_rows[{index}] must be an object")
        category = _require_text(row.get("category"), f"verdict_rows[{index}].category")
        usable = row.get("usable")
        if not isinstance(usable, bool):
            raise TypeError(f"verdict_rows[{index}].usable must be a boolean")
        bucket = category_counts.setdefault(category, {"total": 0, "usable": 0})
        bucket["total"] = int(bucket["total"]) + 1
        if usable:
            usable_maps += 1
            bucket["usable"] = int(bucket["usable"]) + 1
    normalized_counts = {
        category: {
            "total": int(counts["total"]),
            "usable": int(counts["usable"]),
            "passes_minimum": int(counts["total"]) >= 5 and int(counts["usable"]) >= 3,
            "passes_strict": int(counts["total"]) >= 5 and int(counts["usable"]) >= 4,
        }
        for category, counts in sorted(category_counts.items())
    }
    total_maps = len(verdict_rows)
    gate_passed = (
        total_maps >= 35
        and usable_maps >= 28
        and bool(normalized_counts)
        and all(counts["passes_minimum"] for counts in normalized_counts.values())
    )
    strict_four_of_five = (
        total_maps >= 35
        and bool(normalized_counts)
        and all(counts["passes_strict"] for counts in normalized_counts.values())
    )
    return {
        "total_maps": total_maps,
        "usable_maps": usable_maps,
        "gate_passed": gate_passed,
        "strict_four_of_five": strict_four_of_five,
        "category_counts": normalized_counts,
    }


__all__ = [
    "EVENT_MAP_PROMPT_VERSION",
    "EVENT_MAP_SCHEMA",
    "EventRecord",
    "FrontierMap",
    "MAX_EVENTS",
    "MAX_SUPPORT_SPANS",
    "SupportSpan",
    "audit_pre_reference_payload",
    "build_event_map_prompt",
    "mask_question_fields",
    "summarize_map_gate",
    "validate_event_map",
]
