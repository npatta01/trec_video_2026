"""Contracts for label-blind, question-driven atomic counting evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from trec_video_2026.vqa_submission import canonical_sha256

ATOMIC_COUNT_UNITS = frozenset(
    {
        "unique_entities_objects",
        "simultaneously_visible_instances",
        "appearances_across_scenes",
        "edited_shots",
        "repeated_actions_transitions",
        "background_people",
        "parts_components",
    }
)

_FORBIDDEN_LABEL_KEY_PARTS = frozenset(
    {
        "answer",
        "baseline",
        "correct",
        "expected",
        "gold",
        "incorrect",
        "label",
        "panel_role",
        "prediction",
        "reference",
        "status",
    }
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "count_unit",
        "target",
        "scope",
        "records",
        "exclusions",
        "uncertainty",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "start_seconds",
        "end_seconds",
        "scene",
        "rationale",
    }
)
_EXCLUSION_FIELDS = frozenset({"start_seconds", "end_seconds", "reason"})

_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "once": 1,
    "twice": 2,
    "thrice": 3,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS = frozenset(_SMALL) | frozenset(_TENS) | {"hundred", "and"}
_TIMESTAMP_RE = re.compile(
    r"(?<!\d)\d{1,2}:\d{2}(?:\.\d+)?(?:\s*[-–—]\s*\d{1,2}:\d{2}(?:\.\d+)?)?"
)
_TOKEN_RE = re.compile(r"\d+|[a-z]+", re.IGNORECASE)
_LEXICAL_COUNT_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|count(?:ed|ing|s)?)\b", re.IGNORECASE
)
_EXPECTED_CENSUS_COUNTS = {
    "ambiguous": 1,
    "correct": 38,
    "not_count": 1,
    "wrong": 26,
}


@dataclass(frozen=True)
class ParsedCount:
    value: int | None
    candidates: tuple[int, ...]
    committed: bool


@dataclass(frozen=True)
class CountDisposition:
    status: str
    reference_count: int | None
    prediction_count: int | None
    prediction_candidates: tuple[int, ...]


@dataclass(frozen=True)
class AtomicRecord:
    record_id: str
    start_seconds: float
    end_seconds: float
    scene: str
    rationale: str


@dataclass(frozen=True)
class AtomicEvidence:
    count_unit: str
    target: str
    scope: str
    records: tuple[AtomicRecord, ...]
    exclusions: tuple[Mapping[str, object], ...]
    uncertainty: str


def strip_timestamps(text: str) -> str:
    """Remove clock timestamps before interpreting answer numbers."""

    return _TIMESTAMP_RE.sub("", text)


def _parse_number_tokens(tokens: Sequence[str]) -> int | None:
    if not tokens:
        return None
    if len(tokens) == 1 and tokens[0].isdigit():
        return int(tokens[0])
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        lowered = token.lower()
        if lowered == "and":
            continue
        if lowered in _SMALL:
            current += _SMALL[lowered]
            saw_number = True
        elif lowered in _TENS:
            current += _TENS[lowered]
            saw_number = True
        elif lowered == "hundred":
            current = max(current, 1) * 100
            saw_number = True
        else:
            break
    total += current
    return total if saw_number else None


def _number_mentions(text: str) -> list[tuple[int, int, int]]:
    normalized = text.lower().replace("-", " ")
    matches = list(_TOKEN_RE.finditer(normalized))
    mentions: list[tuple[int, int, int]] = []
    index = 0
    while index < len(matches):
        token = matches[index].group(0)
        if token.isdigit():
            mentions.append((int(token), matches[index].start(), matches[index].end()))
            index += 1
            continue
        if token not in _NUMBER_WORDS:
            index += 1
            continue
        end = index
        parts: list[str] = []
        while end < len(matches) and matches[end].group(0) in _NUMBER_WORDS:
            if end > index:
                separator = normalized[matches[end - 1].end() : matches[end].start()]
                if not re.fullmatch(r"\s+", separator):
                    break
            parts.append(matches[end].group(0))
            end += 1
        value = _parse_number_tokens(parts)
        if value is not None:
            mentions.append((value, matches[index].start(), matches[end - 1].end()))
        index = end
    return mentions


def parse_reference_count(text: str) -> ParsedCount:
    cleaned = strip_timestamps(text).strip()
    mentions = _number_mentions(cleaned)
    if not mentions:
        return ParsedCount(None, (), False)
    first = mentions[0][0]
    return ParsedCount(first, tuple(value for value, _, _ in mentions), True)


def parse_prediction_count(text: str) -> ParsedCount:
    cleaned = strip_timestamps(text)
    mentions = _number_mentions(cleaned)
    candidates = tuple(value for value, _, _ in mentions)
    if not mentions:
        return ParsedCount(None, (), False)

    for left, right in pairwise(mentions):
        between = cleaned[left[2] : right[1]].lower()
        if re.fullmatch(r"\s*(?:or|/|to)\s*", between):
            return ParsedCount(None, candidates, False)

    priority_patterns = (
        (
            r"(?:there\s+(?:are|is)|total(?:\s+of)?|altogether|answer\s+is|"
            r"count\s+is|exhibits?)\D{0,30}$"
        ),
    )
    lowered = cleaned.lower()
    for value, start, _ in reversed(mentions):
        prefix = lowered[:start]
        if any(re.search(pattern, prefix) for pattern in priority_patterns):
            return ParsedCount(value, candidates, True)
    return ParsedCount(mentions[-1][0], candidates, True)


def is_count_request(question: str) -> bool:
    lowered = question.lower().strip()
    if "depending on what" in lowered or "depends on what" in lowered:
        return False
    return bool(
        re.search(
            r"\bhow\s+many\b|\bhow\s+often\b|\bnumber\s+of\b|^\s*count\b", lowered
        )
    )


def is_lexical_count_candidate(question: str) -> bool:
    return bool(_LEXICAL_COUNT_RE.search(question))


def classify_count_row(
    question: str,
    reference: str,
    prediction: str,
    *,
    adjudication: str | None = None,
) -> CountDisposition:
    if not is_count_request(question):
        return CountDisposition("not_count", None, None, ())
    reference_parse = parse_reference_count(reference)
    prediction_parse = parse_prediction_count(prediction)
    if adjudication is not None:
        if adjudication not in {"correct", "wrong", "ambiguous", "not_count"}:
            raise ValueError(f"unknown adjudication: {adjudication}")
        status = adjudication
    elif reference_parse.value is None:
        status = "ambiguous"
    elif not prediction_parse.committed:
        status = "wrong"
    elif prediction_parse.value == reference_parse.value:
        status = "correct"
    else:
        status = "wrong"
    return CountDisposition(
        status,
        reference_parse.value,
        prediction_parse.value,
        prediction_parse.candidates,
    )


def validate_label_blind(payload: object, *, _path: str = "root") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered != "question" and any(
                part in lowered for part in _FORBIDDEN_LABEL_KEY_PARTS
            ):
                raise ValueError(f"forbidden label-bearing key at {_path}.{key}: {key}")
            validate_label_blind(value, _path=f"{_path}.{key}")
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            validate_label_blind(value, _path=f"{_path}[{index}]")


def build_atomic_evidence_prompt(
    *, question: str, duration_seconds: float, frozen_map: str
) -> str:
    schema = {
        "schema_version": 1,
        "count_unit": sorted(ATOMIC_COUNT_UNITS),
        "target": "short noun phrase",
        "scope": "scope required by the question",
        "records": [
            {
                "record_id": "R1",
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "scene": "what is visibly happening",
                "rationale": "why this is one distinct counted unit",
            }
        ],
        "exclusions": [
            {"start_seconds": 0.0, "end_seconds": 0.0, "reason": "why not counted"}
        ],
        "uncertainty": "remaining evidence uncertainty, or none",
    }
    return (
        "Inspect the complete native video and audio to answer the counting question by "
        "recovering atomic evidence. The supplied question-blind map is navigation context "
        "only: verify every record against the native media. First decide what one counted "
        "unit means. Then emit one JSON record for every counted unit, with a representative "
        "timestamp or tight span. Do not merge edited shots, appearances, repeated actions, "
        "background people, or parts when the question requires them separately. Do not emit "
        "a final numeric answer or any field outside the schema.\n\n"
        f"Video duration: {duration_seconds:.3f} seconds\n"
        f"Question: {question}\n"
        f"Frozen question-blind map:\n{frozen_map}\n\n"
        f"Return exactly one JSON object matching this template:\n{json.dumps(schema, indent=2)}"
    )


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_time(value: object, *, field: str, duration_seconds: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} timestamp must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} timestamp must be finite and non-negative")
    if result > duration_seconds + 0.05:
        raise ValueError(f"{field} timestamp exceeds video duration")
    return result


def validate_atomic_evidence(
    payload: object, *, duration_seconds: float
) -> AtomicEvidence:
    if not isinstance(payload, Mapping):
        raise TypeError("atomic evidence must be a JSON object")
    unknown = set(payload) - _ROOT_FIELDS
    if unknown:
        raise ValueError(f"unknown root fields: {sorted(unknown)}")
    missing = _ROOT_FIELDS - set(payload)
    if missing:
        raise ValueError(f"missing root fields: {sorted(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    unit = _require_string(payload["count_unit"], field="count_unit")
    if unit not in ATOMIC_COUNT_UNITS:
        raise ValueError(f"unknown count_unit: {unit}")
    target = _require_string(payload["target"], field="target")
    scope = _require_string(payload["scope"], field="scope")
    uncertainty = _require_string(payload["uncertainty"], field="uncertainty")

    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise TypeError("records must be a list")
    records: list[AtomicRecord] = []
    record_ids: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"record {index} must be an object")
        unknown_record = set(raw) - _RECORD_FIELDS
        if unknown_record:
            raise ValueError(f"unknown record fields: {sorted(unknown_record)}")
        missing_record = _RECORD_FIELDS - set(raw)
        if missing_record:
            raise ValueError(f"missing record fields: {sorted(missing_record)}")
        record_id = _require_string(raw["record_id"], field="record_id")
        if record_id in record_ids:
            raise ValueError("record IDs must be unique")
        record_ids.add(record_id)
        start = _require_time(
            raw["start_seconds"],
            field="start_seconds",
            duration_seconds=duration_seconds,
        )
        end = _require_time(
            raw["end_seconds"], field="end_seconds", duration_seconds=duration_seconds
        )
        if start > end:
            raise ValueError("start_seconds must not exceed end_seconds")
        records.append(
            AtomicRecord(
                record_id,
                start,
                end,
                _require_string(raw["scene"], field="scene"),
                _require_string(raw["rationale"], field="rationale"),
            )
        )

    raw_exclusions = payload["exclusions"]
    if not isinstance(raw_exclusions, list):
        raise TypeError("exclusions must be a list")
    exclusions: list[Mapping[str, object]] = []
    for index, raw in enumerate(raw_exclusions):
        if not isinstance(raw, Mapping):
            raise TypeError(f"exclusion {index} must be an object")
        if set(raw) != _EXCLUSION_FIELDS:
            raise ValueError("exclusion fields must exactly match the schema")
        start = _require_time(
            raw["start_seconds"],
            field="start_seconds",
            duration_seconds=duration_seconds,
        )
        end = _require_time(
            raw["end_seconds"], field="end_seconds", duration_seconds=duration_seconds
        )
        if start > end:
            raise ValueError("start_seconds must not exceed end_seconds")
        exclusions.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "reason": _require_string(raw["reason"], field="reason"),
            }
        )
    return AtomicEvidence(
        unit, target, scope, tuple(records), tuple(exclusions), uncertainty
    )


def derive_count(evidence: AtomicEvidence) -> int:
    return len(evidence.records)


def _index_unique(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{name} row has invalid video_id")
        if video_id in result:
            raise ValueError(f"duplicate {name} video_id: {video_id}")
        result[video_id] = row
    return result


def _control_rank(video_id: str) -> str:
    return hashlib.sha256(
        f"issue-34-correct-control-v1:{video_id}".encode()
    ).hexdigest()


def build_counting_panel(
    *,
    protocol_rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    maps: Sequence[Mapping[str, object]],
    annotations: Mapping[str, Mapping[str, object]],
    uploads: Mapping[str, Mapping[str, object]],
    adjudications: Mapping[str, object],
) -> dict[str, object]:
    """Build the private label-bearing panel and its label-blind request projection."""

    prediction_by_id = _index_unique(predictions, name="prediction")
    map_by_id = _index_unique(maps, name="map")
    raw_overrides = adjudications.get("status_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("status_overrides must be an object")
    overrides = {str(key): str(value) for key, value in raw_overrides.items()}

    census_rows: list[dict[str, object]] = []
    protocol_by_id: dict[str, Mapping[str, object]] = {}
    for ordinal, protocol_row in enumerate(protocol_rows):
        video_id = protocol_row.get("video_id")
        question = protocol_row.get("question")
        if not isinstance(video_id, str) or not isinstance(question, str):
            raise TypeError("protocol rows require string video_id and question")
        if video_id in protocol_by_id:
            raise ValueError(f"duplicate protocol video_id: {video_id}")
        protocol_by_id[video_id] = protocol_row
        if not is_lexical_count_candidate(question):
            continue
        prediction_row = prediction_by_id.get(video_id)
        annotation = annotations.get(video_id)
        if prediction_row is None or annotation is None:
            raise ValueError(
                f"count candidate missing prediction or annotation: {video_id}"
            )
        answer = prediction_row.get("answer")
        reference = annotation.get("correct_answer")
        if not isinstance(answer, str) or not isinstance(reference, str):
            raise TypeError(f"count candidate has non-string answer: {video_id}")
        disposition = classify_count_row(
            question,
            reference,
            answer,
            adjudication=overrides.get(video_id),
        )
        census_rows.append(
            {
                "ordinal": ordinal,
                "video_id": video_id,
                "question": question,
                "reference_answer": reference,
                "baseline_prediction": answer,
                "status": disposition.status,
                "reference_count": disposition.reference_count,
                "prediction_count": disposition.prediction_count,
                "prediction_candidates": list(disposition.prediction_candidates),
            }
        )

    counts = {
        status: sum(row["status"] == status for row in census_rows)
        for status in sorted(_EXPECTED_CENSUS_COUNTS)
    }
    if len(census_rows) != 66 or counts != _EXPECTED_CENSUS_COUNTS:
        raise ValueError(
            f"corrected census counts changed: rows={len(census_rows)}, counts={counts}"
        )

    by_status = {
        status: [row for row in census_rows if row["status"] == status]
        for status in _EXPECTED_CENSUS_COUNTS
    }
    required_controls = adjudications.get("required_control_ids", [])
    if not isinstance(required_controls, list) or not all(
        isinstance(item, str) for item in required_controls
    ):
        raise ValueError("required_control_ids must be a list of strings")
    correct_by_id = {str(row["video_id"]): row for row in by_status["correct"]}
    if any(video_id not in correct_by_id for video_id in required_controls):
        raise ValueError("required control is not census-correct")
    remaining_controls = sorted(
        (video_id for video_id in correct_by_id if video_id not in required_controls),
        key=_control_rank,
    )
    control_ids = list(dict.fromkeys(required_controls + remaining_controls))[:12]
    if len(control_ids) != 12:
        raise ValueError("could not select 12 correct controls")

    panel_rows: list[dict[str, object]] = []
    for row in by_status["wrong"]:
        panel_rows.append({**row, "panel_role": "confirmed_failure"})
    for row in by_status["ambiguous"]:
        panel_rows.append({**row, "panel_role": "diagnostic_ambiguity"})
    for video_id in control_ids:
        panel_rows.append({**correct_by_id[video_id], "panel_role": "correct_control"})

    raw_canary_failures = adjudications.get("canary_failure_ids", [])
    raw_canary_controls = adjudications.get("canary_control_ids", [])
    if not isinstance(raw_canary_failures, list) or not isinstance(
        raw_canary_controls, list
    ):
        raise TypeError("canary IDs must be lists")
    wrong_ids = {str(row["video_id"]) for row in by_status["wrong"]}
    if len(raw_canary_failures) != 5 or len(set(raw_canary_failures)) != 5:
        raise ValueError("five distinct canary failure IDs are required")
    if set(raw_canary_failures) - wrong_ids:
        raise ValueError("canary failure IDs must identify confirmed failures")
    if len(raw_canary_controls) != 1 or len(set(raw_canary_controls)) != 1:
        raise ValueError("one distinct canary control ID is required")
    if set(raw_canary_controls) - set(control_ids):
        raise ValueError("canary control ID must identify a selected correct control")
    if set(raw_canary_failures) & set(raw_canary_controls):
        raise ValueError("canary failure and control IDs must be disjoint")
    canary_ids = [str(item) for item in raw_canary_failures + raw_canary_controls]

    request_rows: list[dict[str, object]] = []
    for panel_row in panel_rows:
        video_id = str(panel_row["video_id"])
        protocol_row = protocol_by_id[video_id]
        map_row = map_by_id.get(video_id)
        upload = uploads.get(video_id)
        if map_row is None or upload is None:
            raise ValueError(f"panel row missing map or upload: {video_id}")
        frozen_map = map_row.get("map")
        if not isinstance(frozen_map, str) or not frozen_map:
            raise ValueError(f"panel row has invalid frozen map: {video_id}")
        request_row = {
            "video_id": video_id,
            "question": protocol_row["question"],
            "duration_seconds": protocol_row["duration_seconds"],
            "source_path": protocol_row["source_path"],
            "source_sha256": protocol_row["source_sha256"],
            "frozen_map": frozen_map,
            "file_name": upload["file_name"],
            "file_uri": upload["file_uri"],
            "file_state": upload["file_state"],
            "expiration_time": upload["expiration_time"],
            "delivery_mime_type": upload["delivery_mime_type"],
            "delivery_sha256": upload["delivery_sha256"],
        }
        validate_label_blind(request_row)
        request_rows.append(request_row)

    panel = {
        "schema_version": 1,
        "selection_policy": "all-confirmed-failures-plus-ambiguity-and-hash-ranked-controls-v1",
        "rows": panel_rows,
        "score_denominators": {"confirmed_failures": 26, "correct_controls": 12},
    }
    panel["complete_sha256"] = canonical_sha256(panel)
    requests = {
        "schema_version": 1,
        "prompt_version": "atomic-counting-evidence-v1",
        "rows": request_rows,
        "canary_video_ids": canary_ids,
        "ordered_video_ids_sha256": canonical_sha256(
            [str(row["video_id"]) for row in request_rows]
        ),
    }
    validate_label_blind(requests)
    requests["complete_sha256"] = canonical_sha256(requests)
    return {
        "census": {
            "schema_version": 1,
            "selection_regex": _LEXICAL_COUNT_RE.pattern,
            "counts": counts,
            "rows": census_rows,
            "complete_sha256": canonical_sha256(census_rows),
        },
        "panel": panel,
        "requests": requests,
    }
