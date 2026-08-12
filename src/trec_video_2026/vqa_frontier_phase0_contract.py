"""Strict question-blind map and matched-answer contracts for frontier VQA."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trec_video_2026.vqa_submission import canonical_sha256


SCHEMA_VERSION = 1
NEUTRAL_MAP_PROMPT_VERSION = "frontier-neutral-map-v4"
ANSWER_PROMPT_VERSION = "frontier-grounded-answer-v3"

NEUTRAL_MAP_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "perceived_duration_seconds",
        "global_description",
        "entities",
        "events",
        "global_uncertainties",
    }
)
NEUTRAL_ENTITY_FIELDS = frozenset(
    {"entity_id", "description", "aliases", "confidence"}
)
NEUTRAL_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "start_seconds",
        "end_seconds",
        "keyframe_seconds",
        "description",
        "entity_ids",
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
        "uncertainties",
    }
)
FORBIDDEN_NEUTRAL_KEY_TOKENS = frozenset(
    {
        "answer",
        "annotation",
        "correct",
        "gold",
        "label",
        "oracle",
        "question",
        "reference",
        "relevance",
        "score",
    }
)

ANSWER_ROOT_FIELDS = frozenset(
    {"schema_version", "top1_answer", "claims", "could_be_wrong_because"}
)
ANSWER_CLAIM_FIELDS = frozenset(
    {
        "claim_id",
        "text",
        "supporting_spans",
        "supporting_modalities",
        "map_event_ids",
    }
)
ANSWER_SPAN_FIELDS = frozenset({"start_seconds", "end_seconds"})
ANSWER_MODALITIES = frozenset(
    {"video", "audio", "speech", "visible_text", "neutral_map"}
)

_MAX_ALIASES = 8
_MAX_ENTITIES = 128
_MAX_EVENTS = 128
_MAX_KEYFRAMES = 32
_MAX_STRING_LENGTH = 4096
_MAX_TEXT_ITEMS = 128
_MAX_UNCERTAINTIES = 32
_MAX_CLAIMS = 32
_MAX_SPANS = 32
_MAX_ID_LENGTH = 128
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _string_schema(*, max_length: int = _MAX_STRING_LENGTH) -> dict[str, object]:
    return {"type": "string", "minLength": 1, "maxLength": max_length}


def _number_schema(*, maximum: float | None = None) -> dict[str, object]:
    schema: dict[str, object] = {"type": "number", "minimum": 0}
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _string_array_schema(*, max_items: int) -> dict[str, object]:
    return {
        "type": "array",
        "items": _string_schema(),
        "minItems": 0,
        "maxItems": max_items,
    }


def _id_array_schema(*, max_items: int) -> dict[str, object]:
    return {
        "type": "array",
        "items": _string_schema(max_length=_MAX_ID_LENGTH),
        "minItems": 0,
        "maxItems": max_items,
    }


def _neutral_entity_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "entity_id": _string_schema(max_length=_MAX_ID_LENGTH),
            "description": _string_schema(),
            "aliases": _string_array_schema(max_items=_MAX_ALIASES),
            "confidence": _number_schema(maximum=1),
        },
        "required": sorted(NEUTRAL_ENTITY_FIELDS),
        "additionalProperties": False,
    }


def _neutral_event_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "event_id": _string_schema(max_length=_MAX_ID_LENGTH),
            "start_seconds": _number_schema(),
            "end_seconds": _number_schema(),
            "keyframe_seconds": {
                "type": "array",
                "items": _number_schema(),
                "minItems": 1,
                "maxItems": _MAX_KEYFRAMES,
            },
            "description": _string_schema(),
            "entity_ids": _id_array_schema(max_items=_MAX_ENTITIES),
            "actions": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "relations": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "state_before": _string_schema(),
            "state_change": _string_schema(),
            "state_after": _string_schema(),
            "visual_observations": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "speech_observations": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "sound_observations": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "visible_text": _string_array_schema(max_items=_MAX_TEXT_ITEMS),
            "confidence": _number_schema(maximum=1),
            "uncertainties": _string_array_schema(max_items=_MAX_UNCERTAINTIES),
        },
        "required": sorted(NEUTRAL_EVENT_FIELDS),
        "additionalProperties": False,
    }


NEUTRAL_MAP_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "integer",
            "minimum": 0,
            "const": SCHEMA_VERSION,
        },
        "perceived_duration_seconds": _number_schema(),
        "global_description": _string_schema(),
        "entities": {
            "type": "array",
            "items": _neutral_entity_schema(),
            "minItems": 0,
            "maxItems": _MAX_ENTITIES,
        },
        "events": {
            "type": "array",
            "items": _neutral_event_schema(),
            "minItems": 1,
            "maxItems": _MAX_EVENTS,
        },
        "global_uncertainties": _string_array_schema(max_items=_MAX_UNCERTAINTIES),
    },
    "required": sorted(NEUTRAL_MAP_ROOT_FIELDS),
    "additionalProperties": False,
}


def _answer_span_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "start_seconds": _number_schema(),
            "end_seconds": _number_schema(),
        },
        "required": sorted(ANSWER_SPAN_FIELDS),
        "additionalProperties": False,
    }


def _answer_claim_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "claim_id": _string_schema(max_length=_MAX_ID_LENGTH),
            "text": _string_schema(),
            "supporting_spans": {
                "type": "array",
                "items": _answer_span_schema(),
                "minItems": 1,
                "maxItems": _MAX_SPANS,
            },
            "supporting_modalities": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(ANSWER_MODALITIES)},
                "minItems": 1,
                "maxItems": len(ANSWER_MODALITIES),
            },
            "map_event_ids": _id_array_schema(max_items=_MAX_EVENTS),
        },
        "required": sorted(ANSWER_CLAIM_FIELDS),
        "additionalProperties": False,
    }


GROUNDED_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "integer",
            "minimum": 0,
            "const": SCHEMA_VERSION,
        },
        "top1_answer": _string_schema(),
        "claims": {
            "type": "array",
            "items": _answer_claim_schema(),
            "minItems": 1,
            "maxItems": _MAX_CLAIMS,
        },
        "could_be_wrong_because": _string_array_schema(max_items=_MAX_UNCERTAINTIES),
    },
    "required": sorted(ANSWER_ROOT_FIELDS),
    "additionalProperties": False,
}


def _provider_schema(value: object) -> object:
    """Project a local schema to the provider's conservative JSON subset."""

    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, nested in value.items():
            if key in {
                "additionalProperties",
                "const",
                "maximum",
                "maxItems",
                "maxLength",
                "minimum",
                "minItems",
                "minLength",
            }:
                continue
            projected[str(key)] = _provider_schema(nested)
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_provider_schema(item) for item in value]
    return copy.deepcopy(value)


PROVIDER_NEUTRAL_MAP_SCHEMA = _provider_schema(NEUTRAL_MAP_SCHEMA)
PROVIDER_GROUNDED_ANSWER_SCHEMA = _provider_schema(GROUNDED_ANSWER_SCHEMA)
if not isinstance(PROVIDER_NEUTRAL_MAP_SCHEMA, dict):  # pragma: no cover
    raise TypeError("provider neutral map schema must be an object")
if not isinstance(PROVIDER_GROUNDED_ANSWER_SCHEMA, dict):  # pragma: no cover
    raise TypeError("provider grounded answer schema must be an object")


@dataclass(frozen=True, slots=True)
class NeutralEntity:
    entity_id: str
    description: str
    aliases: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "description": self.description,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class NeutralEvent:
    event_id: str
    start_seconds: float
    end_seconds: float
    keyframe_seconds: tuple[float, ...]
    description: str
    entity_ids: tuple[str, ...]
    actions: tuple[str, ...]
    relations: tuple[str, ...]
    state_before: str
    state_change: str
    state_after: str
    visual_observations: tuple[str, ...]
    speech_observations: tuple[str, ...]
    sound_observations: tuple[str, ...]
    visible_text: tuple[str, ...]
    confidence: float
    uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "keyframe_seconds": list(self.keyframe_seconds),
            "description": self.description,
            "entity_ids": list(self.entity_ids),
            "actions": list(self.actions),
            "relations": list(self.relations),
            "state_before": self.state_before,
            "state_change": self.state_change,
            "state_after": self.state_after,
            "visual_observations": list(self.visual_observations),
            "speech_observations": list(self.speech_observations),
            "sound_observations": list(self.sound_observations),
            "visible_text": list(self.visible_text),
            "confidence": self.confidence,
            "uncertainties": list(self.uncertainties),
        }


@dataclass(frozen=True, slots=True)
class NeutralVideoMap:
    schema_version: int
    perceived_duration_seconds: float
    global_description: str
    entities: tuple[NeutralEntity, ...]
    events: tuple[NeutralEvent, ...]
    global_uncertainties: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "perceived_duration_seconds": self.perceived_duration_seconds,
            "global_description": self.global_description,
            "entities": [entity.to_dict() for entity in self.entities],
            "events": [event.to_dict() for event in self.events],
            "global_uncertainties": list(self.global_uncertainties),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    start_seconds: float
    end_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    claim_id: str
    text: str
    supporting_spans: tuple[AnswerSpan, ...]
    supporting_modalities: tuple[str, ...]
    map_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "supporting_spans": [span.to_dict() for span in self.supporting_spans],
            "supporting_modalities": list(self.supporting_modalities),
            "map_event_ids": list(self.map_event_ids),
        }


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    schema_version: int
    top1_answer: str
    claims: tuple[AnswerClaim, ...]
    could_be_wrong_because: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "top1_answer": self.top1_answer,
            "claims": [claim.to_dict() for claim in self.claims],
            "could_be_wrong_because": list(self.could_be_wrong_because),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _require_mapping(value: Any, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{field} keys must be strings")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be an array")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    keys = frozenset(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{field} violates the contract: {'; '.join(details)}")


def _require_text(
    value: Any, field: str, *, max_length: int = _MAX_STRING_LENGTH
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    text = value.strip()
    if len(text) > max_length:
        raise ValueError(f"{field} exceeds maximum length {max_length}")
    return text


def _require_number(
    value: Any, field: str, *, maximum: float | None = None
) -> float:
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
    max_items: int,
    max_length: int = _MAX_STRING_LENGTH,
    unique: bool = False,
) -> tuple[str, ...]:
    values = tuple(
        _require_text(item, f"{field}[{index}]", max_length=max_length)
        for index, item in enumerate(_require_sequence(value, field))
    )
    if len(values) < min_items:
        raise ValueError(f"{field} must contain at least {min_items} item(s)")
    if len(values) > max_items:
        raise ValueError(f"{field} cannot contain more than {max_items} item(s)")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _require_number_tuple(
    value: Any, field: str, *, min_items: int, max_items: int
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


def _reject_forbidden_neutral_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if FORBIDDEN_NEUTRAL_KEY_TOKENS.intersection(_key_tokens(key)):
                raise ValueError(f"forbidden field in neutral map: {path}.{key}")
            _reject_forbidden_neutral_fields(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_forbidden_neutral_fields(nested, f"{path}[{index}]")


def _parse_neutral_entity(value: Any, field: str) -> NeutralEntity:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, NEUTRAL_ENTITY_FIELDS, field)
    return NeutralEntity(
        entity_id=_require_text(
            mapping["entity_id"], f"{field}.entity_id", max_length=_MAX_ID_LENGTH
        ),
        description=_require_text(mapping["description"], f"{field}.description"),
        aliases=_require_text_tuple(
            mapping["aliases"], f"{field}.aliases", max_items=_MAX_ALIASES
        ),
        confidence=_require_number(
            mapping["confidence"], f"{field}.confidence", maximum=1
        ),
    )


def _parse_neutral_event(
    value: Any, field: str, *, duration_seconds: float
) -> NeutralEvent:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, NEUTRAL_EVENT_FIELDS, field)
    start_seconds = _require_number(mapping["start_seconds"], f"{field}.start_seconds")
    end_seconds = _require_number(mapping["end_seconds"], f"{field}.end_seconds")
    if end_seconds <= start_seconds:
        raise ValueError(f"{field}.end_seconds must be greater than start_seconds")
    if end_seconds > duration_seconds:
        raise ValueError(f"{field} timestamp exceeds source duration")
    keyframes = _require_number_tuple(
        mapping["keyframe_seconds"],
        f"{field}.keyframe_seconds",
        min_items=1,
        max_items=_MAX_KEYFRAMES,
    )
    if any(keyframe < start_seconds or keyframe > end_seconds for keyframe in keyframes):
        raise ValueError(f"{field}.keyframe_seconds must fall within event bounds")
    return NeutralEvent(
        event_id=_require_text(
            mapping["event_id"], f"{field}.event_id", max_length=_MAX_ID_LENGTH
        ),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        keyframe_seconds=keyframes,
        description=_require_text(mapping["description"], f"{field}.description"),
        entity_ids=_require_text_tuple(
            mapping["entity_ids"],
            f"{field}.entity_ids",
            max_items=_MAX_ENTITIES,
            max_length=_MAX_ID_LENGTH,
            unique=True,
        ),
        actions=_require_text_tuple(
            mapping["actions"], f"{field}.actions", max_items=_MAX_TEXT_ITEMS
        ),
        relations=_require_text_tuple(
            mapping["relations"], f"{field}.relations", max_items=_MAX_TEXT_ITEMS
        ),
        state_before=_require_text(mapping["state_before"], f"{field}.state_before"),
        state_change=_require_text(mapping["state_change"], f"{field}.state_change"),
        state_after=_require_text(mapping["state_after"], f"{field}.state_after"),
        visual_observations=_require_text_tuple(
            mapping["visual_observations"],
            f"{field}.visual_observations",
            max_items=_MAX_TEXT_ITEMS,
        ),
        speech_observations=_require_text_tuple(
            mapping["speech_observations"],
            f"{field}.speech_observations",
            max_items=_MAX_TEXT_ITEMS,
        ),
        sound_observations=_require_text_tuple(
            mapping["sound_observations"],
            f"{field}.sound_observations",
            max_items=_MAX_TEXT_ITEMS,
        ),
        visible_text=_require_text_tuple(
            mapping["visible_text"], f"{field}.visible_text", max_items=_MAX_TEXT_ITEMS
        ),
        confidence=_require_number(
            mapping["confidence"], f"{field}.confidence", maximum=1
        ),
        uncertainties=_require_text_tuple(
            mapping["uncertainties"],
            f"{field}.uncertainties",
            max_items=_MAX_UNCERTAINTIES,
        ),
    )


def _validate_event_order(events: tuple[NeutralEvent, ...]) -> None:
    seen_ids: set[str] = set()
    previous_start = -math.inf
    previous_end = -math.inf
    for event in events:
        if event.event_id in seen_ids:
            raise ValueError("event IDs must be unique")
        seen_ids.add(event.event_id)
        if (
            event.start_seconds < previous_start
            or (
                event.start_seconds == previous_start
                and event.end_seconds < previous_end
            )
        ):
            raise ValueError("events must be chronologically ordered")
        previous_start = event.start_seconds
        previous_end = event.end_seconds


def validate_neutral_map(
    payload: object, duration_seconds: float
) -> NeutralVideoMap:
    """Validate and freeze a question-blind neutral video map."""

    _reject_forbidden_neutral_fields(payload)
    duration = _require_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    mapping = _require_mapping(payload, "payload")
    _require_exact_fields(mapping, NEUTRAL_MAP_ROOT_FIELDS, "payload")
    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    perceived_duration = _require_number(
        mapping["perceived_duration_seconds"], "perceived_duration_seconds"
    )
    if abs(perceived_duration - duration) > 0.5:
        raise ValueError(
            "perceived_duration_seconds must be within 0.5 seconds of source duration"
        )
    entity_values = _require_sequence(mapping["entities"], "entities")
    if len(entity_values) > _MAX_ENTITIES:
        raise ValueError(
            f"entities cannot contain more than {_MAX_ENTITIES} item(s)"
        )
    entities = tuple(
        _parse_neutral_entity(item, f"entities[{index}]")
        for index, item in enumerate(entity_values)
    )
    entity_ids = [entity.entity_id for entity in entities]
    if len(set(entity_ids)) != len(entity_ids):
        raise ValueError("entity IDs must be unique")
    events = tuple(
        _parse_neutral_event(
            item, f"events[{index}]", duration_seconds=duration
        )
        for index, item in enumerate(_require_sequence(mapping["events"], "events"))
    )
    if not events:
        raise ValueError("events must not be empty")
    if len(events) > _MAX_EVENTS:
        raise ValueError(f"events cannot contain more than {_MAX_EVENTS} item(s)")
    _validate_event_order(events)
    known_entity_ids = set(entity_ids)
    for index, event in enumerate(events):
        missing = sorted(set(event.entity_ids) - known_entity_ids)
        if missing:
            raise ValueError(
                f"events[{index}] references unknown entity ID {missing[0]}"
            )
    return NeutralVideoMap(
        schema_version=SCHEMA_VERSION,
        perceived_duration_seconds=perceived_duration,
        global_description=_require_text(
            mapping["global_description"], "global_description"
        ),
        entities=entities,
        events=events,
        global_uncertainties=_require_text_tuple(
            mapping["global_uncertainties"],
            "global_uncertainties",
            max_items=_MAX_UNCERTAINTIES,
        ),
    )


def _parse_answer_span(
    value: Any, field: str, *, duration_seconds: float
) -> AnswerSpan:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, ANSWER_SPAN_FIELDS, field)
    start_seconds = _require_number(mapping["start_seconds"], f"{field}.start_seconds")
    end_seconds = _require_number(mapping["end_seconds"], f"{field}.end_seconds")
    if end_seconds <= start_seconds:
        raise ValueError(f"{field}.end_seconds must be greater than start_seconds")
    if end_seconds > duration_seconds:
        raise ValueError(f"{field} timestamp exceeds source duration")
    return AnswerSpan(start_seconds, end_seconds)


def _parse_answer_claim(
    value: Any, field: str, *, duration_seconds: float
) -> AnswerClaim:
    mapping = _require_mapping(value, field)
    _require_exact_fields(mapping, ANSWER_CLAIM_FIELDS, field)
    modalities = _require_text_tuple(
        mapping["supporting_modalities"],
        f"{field}.supporting_modalities",
        min_items=1,
        max_items=len(ANSWER_MODALITIES),
        unique=True,
    )
    unknown_modalities = set(modalities) - ANSWER_MODALITIES
    if unknown_modalities:
        raise ValueError(
            f"{field}.supporting_modalities contains unsupported modality "
            f"{sorted(unknown_modalities)[0]}"
        )
    spans = tuple(
        _parse_answer_span(
            item, f"{field}.supporting_spans[{index}]", duration_seconds=duration_seconds
        )
        for index, item in enumerate(
            _require_sequence(mapping["supporting_spans"], f"{field}.supporting_spans")
        )
    )
    if not spans:
        raise ValueError(f"{field}.supporting_spans must not be empty")
    if len(spans) > _MAX_SPANS:
        raise ValueError(
            f"{field}.supporting_spans cannot contain more than {_MAX_SPANS} item(s)"
        )
    map_event_ids = _require_text_tuple(
        mapping["map_event_ids"],
        f"{field}.map_event_ids",
        max_items=_MAX_EVENTS,
        max_length=_MAX_ID_LENGTH,
        unique=True,
    )
    return AnswerClaim(
        claim_id=_require_text(
            mapping["claim_id"], f"{field}.claim_id", max_length=_MAX_ID_LENGTH
        ),
        text=_require_text(mapping["text"], f"{field}.text"),
        supporting_spans=spans,
        supporting_modalities=modalities,
        map_event_ids=map_event_ids,
    )


def validate_grounded_answer(
    payload: object,
    duration_seconds: float,
    *,
    allowed_map_event_ids: Collection[str] | None,
) -> GroundedAnswer:
    """Validate a direct or map-guided grounded answer under one schema."""

    duration = _require_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    mapping = _require_mapping(payload, "payload")
    _require_exact_fields(mapping, ANSWER_ROOT_FIELDS, "payload")
    schema_version = mapping["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    claim_values = _require_sequence(mapping["claims"], "claims")
    if len(claim_values) > _MAX_CLAIMS:
        raise ValueError(f"claims cannot contain more than {_MAX_CLAIMS} item(s)")
    claims = tuple(
        _parse_answer_claim(item, f"claims[{index}]", duration_seconds=duration)
        for index, item in enumerate(claim_values)
    )
    if not claims:
        raise ValueError("claims must not be empty")
    claim_ids = [claim.claim_id for claim in claims]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("claim IDs must be unique")
    allowed_ids = None if allowed_map_event_ids is None else set(allowed_map_event_ids)
    for claim in claims:
        if allowed_ids is None:
            if claim.map_event_ids:
                raise ValueError("direct answer map_event_ids must be empty")
            continue
        if not claim.map_event_ids:
            raise ValueError(
                "each guided answer claim must cite at least one map event"
            )
        unknown = set(claim.map_event_ids) - allowed_ids
        if unknown:
            raise ValueError(f"unknown map event ID {sorted(unknown)[0]}")
    return GroundedAnswer(
        schema_version=SCHEMA_VERSION,
        top1_answer=_require_text(mapping["top1_answer"], "top1_answer"),
        claims=claims,
        could_be_wrong_because=_require_text_tuple(
            mapping["could_be_wrong_because"],
            "could_be_wrong_because",
            max_items=_MAX_UNCERTAINTIES,
        ),
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(nested) for nested in value]
    return copy.deepcopy(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_neutral_map_prompt(duration_seconds: float) -> str:
    """Build the sole runtime-input, question-blind map prompt."""

    duration = _require_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    return (
        f"{NEUTRAL_MAP_PROMPT_VERSION}\n"
        "Return a JSON object instance only; this is the data instance. Do not return "
        "Markdown, a schema, or metadata.\n"
        "This is a question-blind chronological observation pass: do not use, infer, "
        "or answer any question. Record entities, event boundaries, state changes, "
        "visual observations, speech, sound, visible text, and uncertainty only.\n"
        "Record interactions and relations only when directly observed. An object in "
        "the background or near a person is not held, owned, or used merely because "
        "of proximity; require visible contact or motion evidence and otherwise record "
        "the uncertainty.\n"
        "Place each event start at its first observable onset and its end at the last "
        "observable evidence before the next state. Do not round a boundary earlier "
        "or later than the visible or audible transition.\n"
        "Calibrate confidence to the evidence: use confidence below 1 and add explicit "
        "uncertainties whenever identity, interaction, ordering, or timing is ambiguous.\n"
        f"Source duration seconds: {duration}\n"
        "Allowed root data keys (exactly): schema_version, "
        "perceived_duration_seconds, global_description, entities, events, "
        "global_uncertainties\n"
        "Field semantics: schema_version must be the integer 1; "
        "perceived_duration_seconds records observed duration; global_description "
        "summarizes the complete AV; entities contain stable entity IDs, "
        "descriptions, aliases, and confidence; events contain chronological "
        "timing, keyframes, state changes, actions, relations, AV observations, "
        "visible text, confidence, and uncertainties; global_uncertainties records "
        "uncertainty spanning the video.\n"
        "Do not return schema/meta keys such as properties, required, "
        "additionalProperties, or root type. The provider schema is separately "
        "enforced; fill the allowed root data keys with the complete observation.\n"
        "Observe the complete AV and fill this data instance."
    )


def build_grounded_answer_prompt(
    question: str,
    duration_seconds: float,
    neutral_map: Mapping[str, object] | None,
) -> str:
    """Build one shared direct/map-guided grounded-answer instruction body."""

    prompt_question = _require_text(question, "question")
    duration = _require_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    common = (
        f"{ANSWER_PROMPT_VERSION}\n"
        "Return a JSON object instance only; this is the data instance. Do not return "
        "Markdown, a schema, or metadata. Both direct and map-guided answers use this "
        "same contract and instruction body.\n"
        f"Question: {prompt_question}\n"
        f"Source duration seconds: {duration}\n"
        "Allowed root data keys (exactly): schema_version, top1_answer, claims, "
        "could_be_wrong_because\n"
        "Field semantics: schema_version must be the integer 1; top1_answer is "
        "the concise answer; claims are atomic evidence-backed claims; "
        "could_be_wrong_because records credible uncertainty. Each claim includes "
        "a claim ID, text, native timestamp spans, supporting modalities, and map "
        "event IDs (empty for direct answers).\n"
        "Every claim must include one or more native timestamp spans and supporting "
        "modalities. Keep unsupported details out of top1_answer; express credible "
        "uncertainty in could_be_wrong_because.\n"
    )
    if neutral_map is None:
        mode = (
            "Neutral map: not supplied. This is the direct answer condition; every "
            "claim must use an empty map_event_ids array.\n"
        )
    else:
        if isinstance(neutral_map, NeutralVideoMap):
            perceived_duration = _require_number(
                neutral_map.perceived_duration_seconds,
                "neutral_map.perceived_duration_seconds",
            )
            if abs(perceived_duration - duration) > 0.5:
                raise ValueError(
                    "perceived_duration_seconds must be within 0.5 seconds "
                    "of source duration"
                )
            frozen_map = neutral_map
        else:
            frozen_map = validate_neutral_map(neutral_map, duration)
        frozen_payload = frozen_map.to_dict()
        mode = (
            "Neutral map: supplied below as canonical JSON. This is the map-guided "
            "condition; every material claim must cite one or more declared event "
            "IDs from the map in map_event_ids.\n"
            f"Neutral map SHA-256: {canonical_sha256(frozen_payload)}\n"
            f"Neutral map JSON: {_canonical_json(frozen_payload)}\n"
        )
    return (
        common
        + mode
        + "Do not return schema/meta keys such as properties, required, "
        + "additionalProperties, or root type. The provider schema is separately "
        + "enforced; fill the allowed root data keys with the grounded answer.\n"
        + "Answer the question from the complete AV and fill this data instance."
    )


__all__ = [
    "ANSWER_CLAIM_FIELDS",
    "ANSWER_MODALITIES",
    "ANSWER_PROMPT_VERSION",
    "ANSWER_ROOT_FIELDS",
    "ANSWER_SPAN_FIELDS",
    "AnswerClaim",
    "AnswerSpan",
    "GROUNDED_ANSWER_SCHEMA",
    "GroundedAnswer",
    "FORBIDDEN_NEUTRAL_KEY_TOKENS",
    "NEUTRAL_ENTITY_FIELDS",
    "NEUTRAL_EVENT_FIELDS",
    "NEUTRAL_MAP_PROMPT_VERSION",
    "NEUTRAL_MAP_ROOT_FIELDS",
    "NEUTRAL_MAP_SCHEMA",
    "NeutralEntity",
    "NeutralEvent",
    "NeutralVideoMap",
    "PROVIDER_GROUNDED_ANSWER_SCHEMA",
    "PROVIDER_NEUTRAL_MAP_SCHEMA",
    "build_grounded_answer_prompt",
    "build_neutral_map_prompt",
    "validate_grounded_answer",
    "validate_neutral_map",
]
