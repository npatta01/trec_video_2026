"""Pure Stage 1 contracts for the relevant-scene oracle audit."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from .temporal_bottleneck import (
    ELIGIBILITY_LABELABLE_VISUAL,
    SCHEMA_VERSION,
    _annotation_hash_payload,
    _normalized_annotation_hash_payload,
    _sha256,
    freeze_json_artifact,
    verify_json_artifact,
)

COMPONENT_ROLES = frozenset(
    {"anchor", "target", "context", "precondition", "outcome", "cooccurrence"}
)
SCENE_RELATIONSHIP_TYPES = frozenset(
    {"motion", "ordering", "duration", "transition", "cooccurrence"}
)

CANDIDATES = ("oracle-hybrid-12", "oracle-local-12")
VISIBILITY_VALUES = frozenset({"visible", "partial", "not_visible", "uncertain"})
RELATION_VALUES = frozenset(
    {
        "static_state",
        "motion",
        "ordering",
        "duration",
        "transition",
        "cooccurrence",
        "uncertain",
    }
)
PRESERVATION_VALUES = frozenset(
    {"preserved", "not_preserved", "not_applicable", "uncertain"}
)
SUFFICIENCY_VALUES = frozenset(
    {"answer_sufficient", "answer_insufficient", "uncertain"}
)
STILL_REPRESENTATION_VALUES = frozenset(
    {"still_frame_representable", "still_frame_incompatible", "uncertain"}
)

_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEY_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[^A-Za-z0-9]+")
_FORBIDDEN_AUDIT_KEY_TOKENS = frozenset(
    {
        "answer",
        "answers",
        "reference",
        "references",
        "prediction",
        "predictions",
        "grade",
        "grades",
        "condition",
        "conditions",
        "path",
        "paths",
        "filepath",
        "filepaths",
        "video",
        "videos",
    }
)
_CONDITION_IDENTITIES = frozenset({*CANDIDATES, "uniform-12"})
_CASEFOLDED_CONDITION_IDENTITIES = frozenset(
    identity.casefold() for identity in _CONDITION_IDENTITIES
)
_RELATIVE_PATH = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?$"
)
_ARTIFACT_FILE_EXTENSIONS = frozenset(
    {
        "csv",
        "html",
        "jpeg",
        "jpg",
        "json",
        "md",
        "mp4",
        "png",
        "qmd",
        "txt",
        "webp",
    }
)
_ARTIFACT_DIRECTORIES = frozenset(
    {"artifacts", "contact-sheets", "frames", "outputs", "private"}
)
_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})
_RESPONSE_KINDS = frozenset({"answer", "not_answerable"})


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{field} must be an ISO-8601 UTC timestamp ending in Z"
        ) from exc


def _timing(payload: Mapping[str, Any], *, after: Any, boundary: str) -> None:
    started_at = _timestamp(payload.get("started_at"), "started_at")
    ended_at = _timestamp(payload.get("ended_at"), "ended_at")
    if ended_at <= started_at:
        raise ValueError("ended_at must be after started_at")
    if started_at <= _timestamp(after, boundary):
        raise ValueError(f"started_at must be after bound {boundary}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _sha256_value(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    return _sha256({key: value for key, value in payload.items() if key != field})


def _finish_hash(
    payload: Mapping[str, Any], normalized: dict[str, Any], field: str
) -> dict[str, Any]:
    computed = _hash_without(normalized, field)
    supplied = payload.get(field)
    if supplied is not None and supplied != computed:
        raise ValueError(f"{field} does not match payload")
    normalized[field] = computed
    return normalized


def _adjudication_rows(adjudication: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if adjudication.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"adjudication schema_version must be {SCHEMA_VERSION}")
    if adjudication.get("role") != "adjudicator":
        raise ValueError("base adjudication must have the adjudicator role")
    if not isinstance(adjudication.get("annotation_sha256"), str):
        raise ValueError("base adjudication requires annotation_sha256")
    rows = adjudication.get("rows")
    if not isinstance(rows, list):
        raise TypeError("base adjudication rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"base adjudication row {index} must be an object")
    return rows


def _is_relative_artifact_path(value: str) -> bool:
    if _RELATIVE_PATH.fullmatch(value) is None:
        return False
    segments = [segment.casefold() for segment in value.split("/")]
    filename = segments[-1]
    extension = filename.rpartition(".")[2] if "." in filename else ""
    return extension in _ARTIFACT_FILE_EXTENSIONS or bool(
        set(segments[:-1]) & _ARTIFACT_DIRECTORIES
    )


def _forbidden_audit_value(
    value: Any, *, allowed_keys: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                tokens = {
                    token.casefold() for token in _KEY_BOUNDARY.split(key) if token
                }
                if key not in allowed_keys and tokens & _FORBIDDEN_AUDIT_KEY_TOKENS:
                    return ("key", key)
            forbidden = _forbidden_audit_value(nested, allowed_keys=allowed_keys)
            if forbidden is not None:
                return forbidden
    elif isinstance(value, list):
        for nested in value:
            forbidden = _forbidden_audit_value(nested, allowed_keys=allowed_keys)
            if forbidden is not None:
                return forbidden
    elif isinstance(value, str):
        if value.casefold() in _CASEFOLDED_CONDITION_IDENTITIES:
            return ("condition", value)
        if (
            value.startswith(("/", "~/", "./", "../"))
            or "://" in value
            or "\\" in value
            or re.match(r"^[A-Za-z]:/", value) is not None
            or _is_relative_artifact_path(value)
        ):
            return ("path", value)
    return None


def _reject_audit_leakage(
    payload: Mapping[str, Any], *, allowed_keys: frozenset[str] = frozenset()
) -> None:
    forbidden = _forbidden_audit_value(payload, allowed_keys=allowed_keys)
    if forbidden is None:
        return
    kind, identity = forbidden
    if kind == "key":
        raise ValueError(f"forbidden key in reviewer output: {identity}")
    if kind == "condition":
        raise ValueError(f"condition identity in reviewer output: {identity}")
    raise ValueError(f"path identity in reviewer output: {identity}")


def _packet_rows(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"packet schema_version must be {SCHEMA_VERSION}")
    _sha256_value(packet.get("packet_sha256"), "packet_sha256")
    rows = packet.get("rows")
    if not isinstance(rows, list):
        raise TypeError("packet rows must be a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"packet row {index} must be an object")
    return rows


def _audit_envelope(
    payload: Mapping[str, Any],
    packet: Mapping[str, Any],
    *,
    expected_reviewer: str,
) -> tuple[list[Mapping[str, Any]], list[Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("role") != "scene_auditor":
        raise ValueError("audit role must be scene_auditor")
    _nonempty_string(expected_reviewer, "expected_reviewer")
    if payload.get("reviewer") != expected_reviewer:
        raise ValueError("reviewer does not match expected_reviewer")
    if payload.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("packet_sha256 does not match frozen packet")
    started_at = _timestamp(payload.get("started_at"), "started_at")
    ended_at = _timestamp(payload.get("ended_at"), "ended_at")
    if ended_at <= started_at:
        raise ValueError("ended_at must be after started_at")
    packet_rows = _packet_rows(packet)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("audit rows must be a list")
    return packet_rows, rows


def _opaque_candidate_ids(row: Mapping[str, Any], *, source: str) -> list[str]:
    order = row.get("candidate_order")
    candidates = row.get("candidates")
    if not isinstance(order, list) or not isinstance(candidates, list):
        raise TypeError(f"{source} candidate_order and candidates must be lists")
    ids = [
        candidate.get("opaque_candidate_id") if isinstance(candidate, Mapping) else None
        for candidate in candidates
    ]
    if any(
        not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate or invalid opaque candidates")
    if order != ids:
        raise ValueError(
            f"{source} candidate order (candidate_order) must match candidate records"
        )
    return ids  # type: ignore[return-value]


def validate_scene_semantics(
    payload: dict[str, Any], adjudication: dict[str, Any]
) -> dict[str, Any]:
    """Validate exact-preferred-set semantic components and relationships."""

    if not isinstance(payload, Mapping):
        raise TypeError("scene semantics payload must be an object")
    if not isinstance(adjudication, Mapping):
        raise TypeError("adjudication must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("role") != "scene_semanticist":
        raise ValueError("scene semantics role must be scene_semanticist")
    _nonempty_string(payload.get("reviewer"), "reviewer")

    adjudication_rows = _adjudication_rows(adjudication)
    adjudication_sha256 = adjudication["annotation_sha256"]
    if payload.get("adjudication_sha256") != adjudication_sha256:
        raise ValueError("adjudication_sha256 does not match base adjudication")
    _timing(
        payload,
        after=adjudication.get("ended_at"),
        boundary="adjudication ended_at",
    )

    labelable_rows = [
        row
        for row in adjudication_rows
        if row.get("eligibility") == ELIGIBILITY_LABELABLE_VISUAL
    ]
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("scene semantics rows must be a list")
    expected_video_ids = [row.get("video_id") for row in labelable_rows]
    actual_video_ids = [
        row.get("video_id") if isinstance(row, Mapping) else None for row in rows
    ]
    if len(set(actual_video_ids)) != len(actual_video_ids):
        raise ValueError("scene semantics contains duplicate video IDs")
    if actual_video_ids != expected_video_ids:
        raise ValueError(
            "scene semantics rows must match labelable adjudication rows in order"
        )

    component_ids: set[str] = set()
    relationship_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row_index, (row, base_row) in enumerate(zip(rows, labelable_rows, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError(f"scene semantics row {row_index} must be an object")
        preferred_set_index = base_row.get("preferred_set_index")
        if row.get("preferred_set_index") != preferred_set_index:
            raise ValueError("scene semantics preferred_set_index does not match")
        preferred_intervals = base_row["sufficient_sets"][preferred_set_index]
        components = row.get("components")
        if not isinstance(components, list):
            raise TypeError("scene semantics components must be a list")
        interval_indexes = [
            component.get("interval_index") if isinstance(component, Mapping) else None
            for component in components
        ]
        if interval_indexes != list(range(len(preferred_intervals))):
            raise ValueError(
                "components must contain exactly one component per preferred interval"
            )

        row_component_ids: set[str] = set()
        normalized_components: list[dict[str, Any]] = []
        for component_index, component in enumerate(components):
            if not isinstance(component, Mapping):
                raise TypeError(f"scene component {component_index} must be an object")
            component_id = _nonempty_string(
                component.get("component_id"), "component_id"
            )
            if component_id in component_ids:
                raise ValueError("scene semantics contains a duplicate component_id")
            component_ids.add(component_id)
            row_component_ids.add(component_id)
            supplied_set_index = component.get(
                "sufficient_set_index", preferred_set_index
            )
            if supplied_set_index != preferred_set_index:
                raise ValueError("alternative-set components are forbidden")
            roles = component.get("roles")
            if (
                not isinstance(roles, list)
                or not roles
                or len(set(roles)) != len(roles)
                or any(role not in COMPONENT_ROLES for role in roles)
            ):
                raise ValueError("component roles contain an invalid component role")
            normalized_components.append(copy.deepcopy(dict(component)))

        relationships = row.get("required_relationships")
        if not isinstance(relationships, list):
            raise TypeError("required_relationships must be a list")
        normalized_relationships: list[dict[str, Any]] = []
        for relation_index, relationship in enumerate(relationships):
            if not isinstance(relationship, Mapping):
                raise TypeError(
                    f"required relationship {relation_index} must be an object"
                )
            relationship_id = _nonempty_string(
                relationship.get("relationship_id"), "relationship_id"
            )
            if relationship_id in relationship_ids:
                raise ValueError("scene semantics contains a duplicate relationship_id")
            relationship_ids.add(relationship_id)
            relationship_type = relationship.get("type")
            if relationship_type not in SCENE_RELATIONSHIP_TYPES:
                raise ValueError("required relationship has invalid relationship type")
            bound_component_ids = relationship.get("component_ids")
            if (
                not isinstance(bound_component_ids, list)
                or not bound_component_ids
                or len(set(bound_component_ids)) != len(bound_component_ids)
            ):
                raise ValueError(
                    "relationship component_ids must be unique and nonempty"
                )
            if any(
                component_id not in row_component_ids
                for component_id in bound_component_ids
            ):
                raise ValueError("relationship refers to an unbound component")
            if relationship_type == "ordering" and len(bound_component_ids) < 2:
                raise ValueError(
                    "ordering relationship requires at least two components"
                )
            normalized_relationships.append(copy.deepcopy(dict(relationship)))

        normalized_row = copy.deepcopy(dict(row))
        normalized_row["components"] = normalized_components
        normalized_row["required_relationships"] = normalized_relationships
        normalized_rows.append(normalized_row)

    normalized = copy.deepcopy(dict(payload))
    normalized["rows"] = normalized_rows
    return _finish_hash(payload, normalized, "semantics_sha256")


def validate_frame_first_audit(
    payload: dict[str, Any],
    packet: dict[str, Any],
    *,
    expected_reviewer: str,
) -> dict[str, Any]:
    """Validate a condition-blind frame-first audit in frozen packet order."""

    if not isinstance(payload, Mapping):
        raise TypeError("frame-first audit payload must be an object")
    if not isinstance(packet, Mapping):
        raise TypeError("packet must be an object")
    _reject_audit_leakage(payload)

    packet_rows, rows = _audit_envelope(
        payload, packet, expected_reviewer=expected_reviewer
    )
    expected_row_ids = [row.get("opaque_row_id") for row in packet_rows]
    actual_row_ids = [
        row.get("opaque_row_id") if isinstance(row, Mapping) else None for row in rows
    ]
    if len(set(actual_row_ids)) != len(actual_row_ids):
        raise ValueError("audit contains a duplicate opaque_row_id")
    if actual_row_ids != expected_row_ids:
        raise ValueError("audit rows must match frozen packet rows in order")

    normalized_rows: list[dict[str, Any]] = []
    for row_index, (row, packet_row) in enumerate(zip(rows, packet_rows, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError(f"audit row {row_index} must be an object")
        expected_candidate_ids = _opaque_candidate_ids(
            packet_row, source=f"packet row {row_index}"
        )
        if row.get("candidate_order") != expected_candidate_ids:
            raise ValueError("candidate_order does not match frozen packet")
        candidate_ids = _opaque_candidate_ids(row, source=f"audit row {row_index}")
        unknown = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in expected_candidate_ids
        ]
        if unknown:
            raise ValueError("audit contains an unknown candidate")
        if candidate_ids != expected_candidate_ids:
            raise ValueError("audit candidate order does not match frozen packet")

        normalized_candidates: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(row["candidates"]):
            assert isinstance(candidate, Mapping)
            response_kind = candidate.get("response_kind")
            if response_kind not in _RESPONSE_KINDS:
                raise ValueError("response_kind must be answer or not_answerable")
            response = candidate.get("response")
            if response_kind == "not_answerable" and response is not None:
                raise ValueError("not_answerable response must be null")
            if response_kind == "answer" and (
                not isinstance(response, str) or not response.strip()
            ):
                raise ValueError("answer response must be nonempty text")
            if candidate.get("confidence") not in _CONFIDENCE_VALUES:
                raise ValueError("confidence must be low, medium, or high")
            available_relations = candidate.get("available_relations")
            if (
                not isinstance(available_relations, list)
                or not available_relations
                or len(set(available_relations)) != len(available_relations)
                or any(
                    relation not in RELATION_VALUES for relation in available_relations
                )
            ):
                raise ValueError(
                    "available_relations must contain unique canonical relation values"
                )
            normalized_candidates.append(copy.deepcopy(dict(candidate)))
        normalized_row = copy.deepcopy(dict(row))
        normalized_row["candidates"] = normalized_candidates
        normalized_rows.append(normalized_row)

    normalized = copy.deepcopy(dict(payload))
    normalized["rows"] = normalized_rows
    return _finish_hash(payload, normalized, "audit_sha256")


def _verify_semantics_identity(scene_semantics: Mapping[str, Any]) -> str:
    semantics_sha256 = _sha256_value(
        scene_semantics.get("semantics_sha256"), "scene_semantics_sha256"
    )
    if semantics_sha256 != _hash_without(scene_semantics, "semantics_sha256"):
        raise ValueError("scene_semantics_sha256 does not match scene semantics")
    return semantics_sha256


def _verify_adjudication_identity(adjudication: Mapping[str, Any]) -> str:
    annotation_sha256 = _sha256_value(
        adjudication.get("annotation_sha256"), "annotation_sha256"
    )
    if annotation_sha256 != _sha256(_annotation_hash_payload(adjudication)):
        raise ValueError("annotation_sha256 does not match adjudication")
    normalized_sha256 = adjudication.get("normalized_annotation_sha256")
    if normalized_sha256 is not None:
        _sha256_value(normalized_sha256, "normalized_annotation_sha256")
        if normalized_sha256 != _sha256(
            _normalized_annotation_hash_payload(adjudication)
        ):
            raise ValueError("normalized_annotation_sha256 does not match adjudication")
    return annotation_sha256


def _verify_frame_first_freeze(
    freeze: Mapping[str, Any],
) -> tuple[str, Any, str, str]:
    if freeze.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"frame-first freeze schema_version must be {SCHEMA_VERSION}")
    if freeze.get("status") != "frozen":
        raise ValueError("frame-first freeze status must be frozen")
    _sha256_value(freeze.get("artifact_sha256"), "artifact_sha256")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("frame-first freeze bindings must be nonempty")
    if freeze.get("bindings_sha256") != _sha256(bindings):
        raise ValueError("frame-first freeze bindings_sha256 does not match")
    audit_sha256 = _sha256_value(bindings.get("audit_sha256"), "audit_sha256")
    ended_at = bindings.get("ended_at")
    _timestamp(ended_at, "frame-first ended_at")
    reviewer = _nonempty_string(bindings.get("reviewer"), "frame-first reviewer")
    packet_sha256 = _sha256_value(
        bindings.get("packet_sha256"), "frame-first packet_sha256"
    )
    return audit_sha256, ended_at, reviewer, packet_sha256


def _labelable_row_maps(
    adjudication: Mapping[str, Any], scene_semantics: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    base_rows = {
        row["video_id"]: row
        for row in _adjudication_rows(adjudication)
        if row.get("eligibility") == ELIGIBILITY_LABELABLE_VISUAL
    }
    semantics_rows = scene_semantics.get("rows")
    if not isinstance(semantics_rows, list):
        raise TypeError("scene semantics rows must be a list")
    semantics_by_video: dict[str, Mapping[str, Any]] = {}
    for row in semantics_rows:
        if not isinstance(row, Mapping) or row.get("video_id") not in base_rows:
            raise ValueError(
                "scene semantics row does not match labelable adjudication"
            )
        semantics_by_video[row["video_id"]] = row
    if set(semantics_by_video) != set(base_rows):
        raise ValueError("scene semantics must cover every labelable adjudication row")
    return base_rows, semantics_by_video


def _preferred_packet_intervals(
    packet_candidate: Mapping[str, Any],
    *,
    base_row: Mapping[str, Any],
) -> list[tuple[str, int, float, float, str]]:
    intervals = packet_candidate.get("intervals")
    if not isinstance(intervals, list):
        raise TypeError("packet candidate intervals must be a list")
    preferred_set_index = base_row["preferred_set_index"]
    preferred_intervals = base_row["sufficient_sets"][preferred_set_index]
    normalized: list[tuple[str, int, float, float, str]] = []
    for position, interval in enumerate(intervals):
        if not isinstance(interval, Mapping):
            raise TypeError("packet interval must be an object")
        opaque_interval_id = _nonempty_string(
            interval.get("opaque_interval_id"), "opaque_interval_id"
        )
        if interval.get("sufficient_set_index") != preferred_set_index:
            raise ValueError("packet contains an alternative-set interval")
        if interval.get("interval_index") != position:
            raise ValueError("packet intervals must match preferred-set intervals")
        slot_kind = interval.get("slot_kind")
        if slot_kind != "local":
            raise ValueError("preferred-set audit intervals must bind local frames")
        start, end = preferred_intervals[position]
        normalized.append((opaque_interval_id, position, start, end, slot_kind))
    if len(normalized) != len(preferred_intervals):
        raise ValueError("packet intervals must match preferred-set intervals")
    ids = [record[0] for record in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("packet contains duplicate opaque_interval_id values")
    return normalized


def _validate_repair_proposals(
    proposals: Any,
    *,
    interval_records: Sequence[tuple[str, int, float, float, str]],
    visibility_by_id: Mapping[str, str],
    allow_resolution_lineage: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(proposals, list):
        raise TypeError("repair_proposals must be a list")
    records_by_id = {record[0]: record for record in interval_records}
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise TypeError("repair proposal must be an object")
        resolution_fields = {
            "input_proposed_time_seconds",
            "resolution_rule",
            "resolution_reason",
        }
        if not allow_resolution_lineage and resolution_fields.intersection(proposal):
            raise ValueError(
                "independent auditor repair proposals cannot contain resolution lineage"
            )
        interval_id = proposal.get("opaque_interval_id")
        if interval_id not in records_by_id:
            raise ValueError("repair proposal must target a preferred-set interval")
        if interval_id in seen_ids:
            raise ValueError("only one repair may target each failed interval")
        seen_ids.add(interval_id)
        if visibility_by_id[interval_id] not in {"partial", "not_visible"}:
            raise ValueError("repairs require a partial or not_visible local gap")
        _, _, start, end, slot_kind = records_by_id[interval_id]
        if slot_kind != "local" or proposal.get("target_slot_kind", "local") != "local":
            raise ValueError("repair proposals may target local frames only")
        proposed_time = proposal.get("proposed_time_seconds")
        if isinstance(proposed_time, bool) or not isinstance(
            proposed_time, (int, float)
        ):
            raise TypeError("proposed_time_seconds must be numeric")
        proposed_time = float(proposed_time)
        if not math.isfinite(proposed_time):
            raise ValueError("proposed_time_seconds must be finite")
        if round(proposed_time, 1) != proposed_time:
            raise ValueError("repair proposal times must be recorded to 0.1 seconds")
        if not start <= proposed_time <= end:
            raise ValueError("repair proposal time must be inside its bound interval")
        _nonempty_string(proposal.get("replacement_reason"), "replacement_reason")
        normalized_proposal = copy.deepcopy(dict(proposal))
        normalized_proposal["proposed_time_seconds"] = proposed_time
        normalized.append(normalized_proposal)
    return normalized


def _validate_full_rows(
    rows: Any,
    *,
    packet_rows: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
    scene_semantics: Mapping[str, Any],
    allow_repair_resolution: bool = False,
    allow_derived_pass_after_repair: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("full-video audit rows must be a list")
    expected_row_ids = [row.get("opaque_row_id") for row in packet_rows]
    actual_row_ids = [
        row.get("opaque_row_id") if isinstance(row, Mapping) else None for row in rows
    ]
    if len(set(actual_row_ids)) != len(actual_row_ids):
        raise ValueError("full-video audit contains a duplicate opaque_row_id")
    if actual_row_ids != expected_row_ids:
        raise ValueError("full-video audit rows must match packet rows in order")

    base_by_video, semantics_by_video = _labelable_row_maps(
        adjudication, scene_semantics
    )
    packet_video_ids = [row.get("video_id") for row in packet_rows]
    if packet_video_ids != list(base_by_video):
        raise ValueError("packet rows must match labelable adjudication rows in order")

    normalized_rows: list[dict[str, Any]] = []
    for row_index, (row, packet_row) in enumerate(zip(rows, packet_rows, strict=True)):
        if not isinstance(row, Mapping):
            raise TypeError(f"full-video audit row {row_index} must be an object")
        still_representation = row.get("still_representation")
        if still_representation not in STILL_REPRESENTATION_VALUES:
            raise ValueError("still_representation has an invalid value")
        representation_classes = row.get("representation_classes")
        if (
            not isinstance(representation_classes, list)
            or not representation_classes
            or len(set(representation_classes)) != len(representation_classes)
            or any(value not in RELATION_VALUES for value in representation_classes)
        ):
            raise ValueError("representation_classes contain an invalid value")

        video_id = packet_row["video_id"]
        base_row = base_by_video[video_id]
        semantics_row = semantics_by_video[video_id]
        expected_relations = [
            relation["type"] for relation in semantics_row["required_relationships"]
        ]
        packet_candidate_ids = _opaque_candidate_ids(
            packet_row, source=f"packet row {row_index}"
        )
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("full-video audit candidates must be a list")
        candidate_ids = [
            candidate.get("opaque_candidate_id")
            if isinstance(candidate, Mapping)
            else None
            for candidate in candidates
        ]
        if candidate_ids != packet_candidate_ids:
            raise ValueError("full-video candidate order must match frozen packet")

        normalized_candidates: list[dict[str, Any]] = []
        for candidate, packet_candidate in zip(
            candidates, packet_row["candidates"], strict=True
        ):
            if not isinstance(candidate, Mapping) or not isinstance(
                packet_candidate, Mapping
            ):
                raise TypeError("candidate audit records must be objects")
            if "pass_after_repair" in candidate and not allow_derived_pass_after_repair:
                raise ValueError(
                    "pass_after_repair is derived and cannot be supplied by an auditor"
                )
            interval_records = _preferred_packet_intervals(
                packet_candidate, base_row=base_row
            )
            expected_interval_ids = [record[0] for record in interval_records]
            interval_judgments = candidate.get("intervals")
            if not isinstance(interval_judgments, list):
                raise TypeError("candidate intervals must be a list")
            actual_interval_ids = [
                interval.get("opaque_interval_id")
                if isinstance(interval, Mapping)
                else None
                for interval in interval_judgments
            ]
            if actual_interval_ids != expected_interval_ids:
                raise ValueError(
                    "candidate intervals must match exact preferred-set intervals"
                )
            visibility_by_id: dict[str, str] = {}
            normalized_intervals: list[dict[str, Any]] = []
            for interval in interval_judgments:
                assert isinstance(interval, Mapping)
                visibility = interval.get("visibility")
                if visibility not in VISIBILITY_VALUES:
                    raise ValueError("interval visibility has an invalid value")
                visibility_by_id[interval["opaque_interval_id"]] = visibility
                normalized_intervals.append(copy.deepcopy(dict(interval)))

            relations = candidate.get("relations")
            if not isinstance(relations, list):
                raise TypeError("candidate relations must be a list")
            actual_relations = [
                relation.get("relation") if isinstance(relation, Mapping) else None
                for relation in relations
            ]
            if actual_relations != expected_relations:
                raise ValueError("candidate relations must match frozen relationships")
            normalized_relations: list[dict[str, Any]] = []
            for relation in relations:
                assert isinstance(relation, Mapping)
                preservation = relation.get("preservation")
                if preservation not in PRESERVATION_VALUES:
                    raise ValueError("relation preservation has an invalid value")
                if preservation == "not_applicable":
                    raise ValueError(
                        "a frozen required relationship cannot be not_applicable"
                    )
                normalized_relations.append(copy.deepcopy(dict(relation)))
            answer_sufficiency = candidate.get("answer_sufficiency")
            if answer_sufficiency not in SUFFICIENCY_VALUES:
                raise ValueError("answer_sufficiency has an invalid value")
            normalized_repairs = _validate_repair_proposals(
                candidate.get("repair_proposals"),
                interval_records=interval_records,
                visibility_by_id=visibility_by_id,
                allow_resolution_lineage=allow_repair_resolution,
            )
            if normalized_repairs:
                if still_representation != "still_frame_representable":
                    raise ValueError(
                        "repairs require still_frame_representable evidence"
                    )
                if any(
                    relation["preservation"] != "preserved"
                    for relation in normalized_relations
                ):
                    raise ValueError(
                        "repairs cannot rescue a relationship preservation failure"
                    )
                if answer_sufficiency != "answer_sufficient":
                    raise ValueError("repairs cannot rescue answer_sufficiency failure")
                failed_interval_ids = {
                    interval_id
                    for interval_id, visibility in visibility_by_id.items()
                    if visibility != "visible"
                }
                repairable_interval_ids = {
                    interval_id
                    for interval_id, visibility in visibility_by_id.items()
                    if visibility in {"partial", "not_visible"}
                }
                proposed_interval_ids = {
                    proposal["opaque_interval_id"] for proposal in normalized_repairs
                }
                if (
                    failed_interval_ids != repairable_interval_ids
                    or proposed_interval_ids != failed_interval_ids
                ):
                    raise ValueError(
                        "repair proposals must target every failed local interval"
                    )

            all_intervals_visible = all(
                visibility == "visible" for visibility in visibility_by_id.values()
            )
            all_relationships_preserved = all(
                relation["preservation"] == "preserved"
                for relation in normalized_relations
            )
            candidate_passes = (
                still_representation == "still_frame_representable"
                and all_intervals_visible
                and all_relationships_preserved
                and answer_sufficiency == "answer_sufficient"
            )
            for field, expected in (
                ("all_intervals_visible", all_intervals_visible),
                ("all_relationships_preserved", all_relationships_preserved),
                ("candidate_passes", candidate_passes),
            ):
                if field in candidate and candidate[field] is not expected:
                    raise ValueError(f"derived {field} does not match judgments")
            normalized_candidate = copy.deepcopy(dict(candidate))
            normalized_candidate["intervals"] = normalized_intervals
            normalized_candidate["relations"] = normalized_relations
            normalized_candidate["repair_proposals"] = normalized_repairs
            normalized_candidate["all_intervals_visible"] = all_intervals_visible
            normalized_candidate["all_relationships_preserved"] = (
                all_relationships_preserved
            )
            normalized_candidate["candidate_passes"] = candidate_passes
            normalized_candidates.append(normalized_candidate)

        normalized_row = copy.deepcopy(dict(row))
        normalized_row["candidates"] = normalized_candidates
        normalized_rows.append(normalized_row)
    return normalized_rows


_DERIVED_AUDIT_FIELDS = frozenset(
    {
        "all_intervals_visible",
        "all_relationships_preserved",
        "candidate_passes",
    }
)


def _without_derived_audit_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_derived_audit_fields(nested)
            for key, nested in value.items()
            if key not in _DERIVED_AUDIT_FIELDS
        }
    if isinstance(value, list):
        return [_without_derived_audit_fields(nested) for nested in value]
    return value


def _finish_audit_hash(
    payload: Mapping[str, Any], normalized: dict[str, Any], field: str
) -> dict[str, Any]:
    source = {
        key: value
        for key, value in normalized.items()
        if key != field and key not in _DERIVED_AUDIT_FIELDS
    }
    computed = _sha256(_without_derived_audit_fields(source))
    supplied = payload.get(field)
    if supplied is not None and supplied != computed:
        raise ValueError(f"{field} does not match audit payload")
    normalized[field] = computed
    return normalized


def _verify_audit_adjudication_identity(
    audit_adjudication: Mapping[str, Any],
) -> str:
    stored = _sha256_value(
        audit_adjudication.get("audit_adjudication_sha256"),
        "audit_adjudication_sha256",
    )
    computed = _sha256(
        _without_derived_audit_fields(
            {
                key: value
                for key, value in audit_adjudication.items()
                if key != "audit_adjudication_sha256"
            }
        )
    )
    if stored != computed:
        raise ValueError("audit_adjudication_sha256 does not match audit adjudication")
    return stored


def validate_full_video_audit(
    payload: dict[str, Any],
    packet: dict[str, Any],
    adjudication: dict[str, Any],
    scene_semantics: dict[str, Any],
    frame_first_freeze: dict[str, Any],
    *,
    expected_reviewer: str,
) -> dict[str, Any]:
    """Validate full-video judgments against exact frozen preferred evidence."""

    if not all(
        isinstance(value, Mapping)
        for value in (
            payload,
            packet,
            adjudication,
            scene_semantics,
            frame_first_freeze,
        )
    ):
        raise TypeError("full-video audit inputs must be objects")
    _reject_audit_leakage(payload, allowed_keys=frozenset({"answer_sufficiency"}))
    packet_rows, rows = _audit_envelope(
        payload, packet, expected_reviewer=expected_reviewer
    )
    adjudication_sha256 = adjudication.get("annotation_sha256")
    if packet.get("adjudication_sha256") != adjudication_sha256:
        raise ValueError("packet adjudication_sha256 does not match adjudication")
    if payload.get("adjudication_sha256") != adjudication_sha256:
        raise ValueError("adjudication_sha256 does not match adjudication")
    semantics_sha256 = _verify_semantics_identity(scene_semantics)
    if payload.get("scene_semantics_sha256") != semantics_sha256:
        raise ValueError("scene_semantics_sha256 does not match scene semantics")
    (
        frame_first_sha256,
        frame_first_ended_at,
        frame_first_reviewer,
        frame_first_packet_sha256,
    ) = _verify_frame_first_freeze(frame_first_freeze)
    if payload.get("frame_first_audit_sha256") != frame_first_sha256:
        raise ValueError("frame_first_audit_sha256 does not match frozen audit")
    if frame_first_reviewer != expected_reviewer:
        raise ValueError(
            "frame-first freeze reviewer does not match full-video reviewer"
        )
    if frame_first_packet_sha256 != packet.get("packet_sha256"):
        raise ValueError("frame-first freeze packet_sha256 does not match packet")
    if _timestamp(payload.get("started_at"), "started_at") <= _timestamp(
        frame_first_ended_at, "frame-first ended_at"
    ):
        raise ValueError("started_at must be after bound frame-first freeze")

    normalized = copy.deepcopy(dict(payload))
    normalized["rows"] = _validate_full_rows(
        rows,
        packet_rows=packet_rows,
        adjudication=adjudication,
        scene_semantics=scene_semantics,
    )
    return _finish_audit_hash(payload, normalized, "audit_sha256")


_AGREEMENT_DIMENSIONS = (
    "representability",
    "interval_visibility",
    "relationship_preservation",
    "answer_sufficiency",
)
_AGREEMENT_COUNT_FIELDS = frozenset(
    {"agreement_count", "disagreement_count", "comparison_count"}
)


def _raw_judgments(audit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {
        dimension: {} for dimension in _AGREEMENT_DIMENSIONS
    }
    for row in audit["rows"]:
        row_id = row["opaque_row_id"]
        judgments["representability"][row_id] = {
            "still_representation": row["still_representation"],
            "representation_classes": row["representation_classes"],
        }
        for candidate in row["candidates"]:
            candidate_id = candidate["opaque_candidate_id"]
            prefix = f"{row_id}/{candidate_id}"
            for interval in candidate["intervals"]:
                judgments["interval_visibility"][
                    f"{prefix}/interval/{interval['opaque_interval_id']}"
                ] = interval["visibility"]
            for proposal in candidate["repair_proposals"]:
                judgments["interval_visibility"][
                    f"{prefix}/repair_proposal/{proposal['opaque_interval_id']}"
                ] = proposal["proposed_time_seconds"]
            for index, relation in enumerate(candidate["relations"]):
                judgments["relationship_preservation"][f"{prefix}/relation/{index}"] = (
                    relation["preservation"]
                )
            judgments["answer_sufficiency"][prefix] = candidate["answer_sufficiency"]
    return judgments


def _repair_proposals_by_identity(
    audit: Mapping[str, Any],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    proposals: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in audit["rows"]:
        row_id = row["opaque_row_id"]
        for candidate in row["candidates"]:
            candidate_id = candidate["opaque_candidate_id"]
            for proposal in candidate["repair_proposals"]:
                key = (row_id, candidate_id, proposal["opaque_interval_id"])
                if key in proposals:
                    raise ValueError(
                        "repair resolution contains a duplicate interval identity"
                    )
                proposals[key] = proposal
    return proposals


def _validate_repair_resolution_lineage(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
) -> None:
    left_proposals = _repair_proposals_by_identity(left)
    right_proposals = _repair_proposals_by_identity(right)
    final_proposals = _repair_proposals_by_identity({"rows": final_rows})
    if (
        left_proposals.keys() != right_proposals.keys()
        or left_proposals.keys() != final_proposals.keys()
    ):
        raise ValueError(
            "repair resolution interval identities must match both auditor proposals"
        )

    allowed_fields = {
        "opaque_interval_id",
        "target_slot_kind",
        "proposed_time_seconds",
        "replacement_reason",
        "input_proposed_time_seconds",
        "resolution_rule",
        "resolution_reason",
    }
    allowed_rules = {
        "agreement",
        "auditor_left",
        "auditor_right",
        "de_novo_event_center",
    }
    for key, final in final_proposals.items():
        if not set(final).issubset(allowed_fields):
            raise ValueError("repair resolution proposal contains an unknown field")
        left_time = left_proposals[key]["proposed_time_seconds"]
        right_time = right_proposals[key]["proposed_time_seconds"]
        inputs = final.get("input_proposed_time_seconds")
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                "input_proposed_time_seconds must contain two auditor proposals"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in inputs
        ):
            raise ValueError(
                "input_proposed_time_seconds must contain numeric auditor proposals"
            )
        if inputs != [left_time, right_time]:
            raise ValueError(
                "input_proposed_time_seconds must preserve proposals in auditor order"
            )
        rule = final.get("resolution_rule")
        if rule not in allowed_rules:
            raise ValueError("repair resolution_rule has an invalid value")
        final_time = final["proposed_time_seconds"]
        reason_present = "resolution_reason" in final
        if rule == "agreement":
            if left_time != right_time or final_time != left_time:
                raise ValueError(
                    "agreement requires equal auditor inputs and the same final time"
                )
        elif rule == "auditor_left":
            if left_time == right_time or final_time != left_time:
                raise ValueError(
                    "auditor_left requires disagreement and the left auditor time"
                )
        elif rule == "auditor_right":
            if left_time == right_time or final_time != right_time:
                raise ValueError(
                    "auditor_right requires disagreement and the right auditor time"
                )
        else:
            if left_time == right_time or final_time in {left_time, right_time}:
                raise ValueError(
                    "de_novo_event_center requires another time after disagreement"
                )
            _nonempty_string(final.get("resolution_reason"), "resolution_reason")
        if rule != "de_novo_event_center" and reason_present:
            raise ValueError(
                "resolution_reason is allowed only for de_novo_event_center"
            )


def _agreement_counts(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    final: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for dimension in _AGREEMENT_DIMENSIONS:
        left_values = left[dimension]
        right_values = right[dimension]
        final_values = final[dimension]
        if (
            left_values.keys() != right_values.keys()
            or left_values.keys() != final_values.keys()
        ):
            raise ValueError(
                f"audit adjudication {dimension} judgments do not match inputs"
            )
        agreement_count = 0
        for key, left_value in left_values.items():
            if left_value == right_values[key]:
                agreement_count += 1
                if final_values[key] != left_value:
                    raise ValueError(
                        "audit adjudication may not override auditor agreement"
                    )
        comparison_count = len(left_values)
        counts[dimension] = {
            "agreement_count": agreement_count,
            "disagreement_count": comparison_count - agreement_count,
            "comparison_count": comparison_count,
        }
    return counts


def _validate_raw_agreement_counts(
    audit_adjudication: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    dimensions = audit_adjudication.get("raw_agreement_by_dimension")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(
        _AGREEMENT_DIMENSIONS
    ):
        raise ValueError("raw agreement must contain exactly four dimensions")
    normalized: dict[str, dict[str, int]] = {}
    for dimension in _AGREEMENT_DIMENSIONS:
        counts = dimensions[dimension]
        if not isinstance(counts, Mapping) or set(counts) != _AGREEMENT_COUNT_FIELDS:
            raise ValueError(f"raw {dimension} counts are incomplete")
        normalized_counts: dict[str, int] = {}
        for field in _AGREEMENT_COUNT_FIELDS:
            value = counts[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"raw {dimension} {field} must be nonnegative")
            normalized_counts[field] = value
        if (
            normalized_counts["agreement_count"]
            + normalized_counts["disagreement_count"]
            != normalized_counts["comparison_count"]
        ):
            raise ValueError(f"raw {dimension} counts do not sum to comparisons")
        normalized[dimension] = normalized_counts
    expected_totals = {
        "raw_agreement_count": sum(
            counts["agreement_count"] for counts in normalized.values()
        ),
        "raw_disagreement_count": sum(
            counts["disagreement_count"] for counts in normalized.values()
        ),
        "raw_comparison_count": sum(
            counts["comparison_count"] for counts in normalized.values()
        ),
    }
    for field, expected in expected_totals.items():
        if audit_adjudication.get(field) != expected:
            raise ValueError(f"{field} does not match raw agreement dimensions")
    return copy.deepcopy(normalized)


def validate_audit_adjudication(
    payload: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    packet: dict[str, Any],
    adjudication: dict[str, Any],
    scene_semantics: dict[str, Any],
    *,
    round_name: str,
) -> dict[str, Any]:
    """Validate a fresh third auditor's resolution of two frozen audits."""

    if not all(
        isinstance(value, Mapping)
        for value in (payload, left, right, packet, adjudication, scene_semantics)
    ):
        raise TypeError("audit adjudication inputs must be objects")
    _reject_audit_leakage(payload, allowed_keys=frozenset({"answer_sufficiency"}))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("role") != "scene_audit_adjudicator":
        raise ValueError("role must be scene_audit_adjudicator")
    if (
        round_name not in {"initial", "repair"}
        or payload.get("round_name") != round_name
    ):
        raise ValueError(
            "round_name must be initial or repair and match expected round"
        )
    left_reviewer = left.get("reviewer")
    right_reviewer = right.get("reviewer")
    reviewer = payload.get("reviewer")
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or left_reviewer == right_reviewer
        or reviewer in {left_reviewer, right_reviewer}
    ):
        raise ValueError("adjudication requires a fresh third reviewer")
    expected_hashes = [
        _sha256_value(left.get("audit_sha256"), "left audit_sha256"),
        _sha256_value(right.get("audit_sha256"), "right audit_sha256"),
    ]
    input_hashes = payload.get("input_audit_sha256s")
    if not isinstance(input_hashes, list) or len(input_hashes) != 2:
        raise ValueError("input_audit_sha256s must contain exactly two hashes")
    if len(set(input_hashes)) != 2:
        raise ValueError("input_audit_sha256s must contain distinct hashes")
    for index, value in enumerate(input_hashes):
        _sha256_value(value, f"input_audit_sha256s[{index}]")
    if input_hashes != expected_hashes:
        raise ValueError("input_audit_sha256s do not match the two frozen audits")
    if left.get("audit_sha256") != _sha256(
        _without_derived_audit_fields(
            {key: value for key, value in left.items() if key != "audit_sha256"}
        )
    ) or right.get("audit_sha256") != _sha256(
        _without_derived_audit_fields(
            {key: value for key, value in right.items() if key != "audit_sha256"}
        )
    ):
        raise ValueError("input frozen audit hash does not match audit payload")
    expected_upstreams = {
        "packet_sha256": packet.get("packet_sha256"),
        "adjudication_sha256": adjudication.get("annotation_sha256"),
        "scene_semantics_sha256": scene_semantics.get("semantics_sha256"),
    }
    for side, audit in (("left", left), ("right", right)):
        for field, expected in expected_upstreams.items():
            if audit.get(field) != expected:
                raise ValueError(
                    f"{side} audit {field} does not match supplied artifact"
                )
    if payload.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("packet_sha256 does not match frozen packet")
    if payload.get("adjudication_sha256") != adjudication.get("annotation_sha256"):
        raise ValueError("adjudication_sha256 does not match adjudication")
    semantics_sha256 = _verify_semantics_identity(scene_semantics)
    if payload.get("scene_semantics_sha256") != semantics_sha256:
        raise ValueError("scene_semantics_sha256 does not match scene semantics")
    started_at = _timestamp(payload.get("started_at"), "started_at")
    ended_at = _timestamp(payload.get("ended_at"), "ended_at")
    if ended_at <= started_at:
        raise ValueError("ended_at must be after started_at")
    if started_at <= max(
        _timestamp(left.get("ended_at"), "left ended_at"),
        _timestamp(right.get("ended_at"), "right ended_at"),
    ):
        raise ValueError("adjudication must start after both frozen audits")

    packet_rows = _packet_rows(packet)
    normalized_rows = _validate_full_rows(
        payload.get("rows"),
        packet_rows=packet_rows,
        adjudication=adjudication,
        scene_semantics=scene_semantics,
        allow_repair_resolution=True,
        allow_derived_pass_after_repair=("audit_adjudication_sha256" in payload),
    )
    _validate_repair_resolution_lineage(left, right, normalized_rows)
    pass_after_repair = round_name == "repair"
    for row in normalized_rows:
        for candidate in row["candidates"]:
            expected = pass_after_repair and candidate["candidate_passes"]
            if (
                "pass_after_repair" in candidate
                and candidate["pass_after_repair"] is not expected
            ):
                raise ValueError(
                    "pass_after_repair does not match the final audit round"
                )
            candidate["pass_after_repair"] = expected
    left_judgments = _raw_judgments(left)
    right_judgments = _raw_judgments(right)
    final_judgments = _raw_judgments({"rows": normalized_rows})
    dimension_counts = _agreement_counts(
        left_judgments, right_judgments, final_judgments
    )
    raw_agreement_count = sum(
        counts["agreement_count"] for counts in dimension_counts.values()
    )
    raw_disagreement_count = sum(
        counts["disagreement_count"] for counts in dimension_counts.values()
    )
    raw_comparison_count = sum(
        counts["comparison_count"] for counts in dimension_counts.values()
    )
    normalized = copy.deepcopy(dict(payload))
    normalized["rows"] = normalized_rows
    normalized["raw_agreement_by_dimension"] = dimension_counts
    normalized["raw_agreement_count"] = raw_agreement_count
    normalized["raw_disagreement_count"] = raw_disagreement_count
    normalized["raw_comparison_count"] = raw_comparison_count
    for field in (
        "raw_agreement_by_dimension",
        "raw_agreement_count",
        "raw_disagreement_count",
        "raw_comparison_count",
    ):
        if field in payload and payload[field] != normalized[field]:
            raise ValueError(f"{field} does not match independent auditor judgments")
    return _finish_audit_hash(payload, normalized, "audit_adjudication_sha256")


def _verified_bindings(
    bindings: Mapping[str, Any],
    *,
    adjudication: Mapping[str, Any],
    scene_semantics: Mapping[str, Any] | None = None,
    audit_adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    if not bindings:
        raise ValueError("bindings must be a nonempty object")
    required = {
        "adjudication_sha256": adjudication.get("annotation_sha256"),
        "audit_adjudication_sha256": audit_adjudication.get(
            "audit_adjudication_sha256"
        ),
    }
    if scene_semantics is not None:
        required["scene_semantics_sha256"] = scene_semantics.get("semantics_sha256")
    for key, expected in required.items():
        if bindings.get(key) != expected:
            raise ValueError(f"bindings {key} does not match upstream artifact")
    normalized = copy.deepcopy(dict(bindings))
    for key, value in normalized.items():
        if not key.endswith("_sha256"):
            raise ValueError("bindings may contain hash identities only")
        _sha256_value(value, f"bindings {key}")
    return normalized


def _candidate_mapping(
    candidate_condition_map: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    candidate_ids: set[str] | None = None
    for row in audit_rows:
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("audit adjudication candidates must be a list")
        row_ids = {
            candidate.get("opaque_candidate_id")
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
        if len(row_ids) != len(candidates):
            raise ValueError("audit adjudication contains invalid opaque candidates")
        if candidate_ids is None:
            candidate_ids = row_ids  # type: ignore[assignment]
        elif row_ids != candidate_ids:
            raise ValueError("audit rows must use one stable opaque candidate set")
    if candidate_ids is None or set(candidate_condition_map) != candidate_ids:
        raise ValueError(
            "candidate condition map must cover the exact opaque candidates"
        )
    if set(candidate_condition_map.values()) != set(CANDIDATES) or len(
        set(candidate_condition_map.values())
    ) != len(CANDIDATES):
        raise ValueError("candidate condition map must be a bijection to candidates")
    return {str(key): str(value) for key, value in candidate_condition_map.items()}


def _derived_candidate_metrics(candidate: Mapping[str, Any]) -> dict[str, bool]:
    intervals = candidate.get("intervals")
    relations = candidate.get("relations")
    if not isinstance(intervals, list) or not isinstance(relations, list):
        raise TypeError("audit candidate intervals and relations must be lists")
    visible = bool(intervals) and all(
        isinstance(interval, Mapping) and interval.get("visibility") == "visible"
        for interval in intervals
    )
    relations_preserved = all(
        isinstance(relation, Mapping) and relation.get("preservation") == "preserved"
        for relation in relations
    )
    answer_sufficient = candidate.get("answer_sufficiency") == "answer_sufficient"
    return {
        "visible": visible,
        "relations_preserved": relations_preserved,
        "answer_sufficient": answer_sufficient,
    }


def _stage1_aggregates(
    adjudication: Mapping[str, Any],
    audit_adjudication: Mapping[str, Any],
    candidate_condition_map: Mapping[str, str],
) -> tuple[
    dict[str, int], dict[str, int], dict[str, dict[str, int]], list[dict[str, Any]]
]:
    rows = _adjudication_rows(adjudication)
    if len(rows) != 10:
        raise ValueError("Stage 1 requires exactly ten frozen adjudication rows")
    round_name = audit_adjudication.get("round_name")
    if round_name not in {"initial", "repair"}:
        raise ValueError("audit adjudication round_name must be initial or repair")
    eligibility_counts: dict[str, int] = {}
    labelable_rows: list[Mapping[str, Any]] = []
    for row in rows:
        eligibility = row.get("eligibility")
        if not isinstance(eligibility, str):
            raise ValueError("every adjudication row requires eligibility")
        eligibility_counts[eligibility] = eligibility_counts.get(eligibility, 0) + 1
        if eligibility == ELIGIBILITY_LABELABLE_VISUAL:
            labelable_rows.append(row)

    audit_rows = audit_adjudication.get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != len(labelable_rows):
        raise ValueError("audit adjudication must cover every labelable row")
    if any(not isinstance(row, Mapping) for row in audit_rows):
        raise TypeError("audit adjudication rows must be objects")

    still_counts = {
        "still_frame_representable": 0,
        "still_frame_incompatible": 0,
        "uncertain": 0,
    }
    candidate_counts = {
        candidate: {
            "all_ten_denominator": 10,
            "visible": 0,
            "relations_preserved": 0,
            "answer_sufficient": 0,
            "pass_after_repair": 0,
            "final_pass": 0,
        }
        for candidate in CANDIDATES
    }
    private_rows: list[dict[str, Any]] = []
    for base_row, audit_row in zip(labelable_rows, audit_rows, strict=True):
        assert isinstance(audit_row, Mapping)
        still_representation = audit_row.get("still_representation")
        if still_representation not in still_counts:
            raise ValueError("audit adjudication has invalid still_representation")
        still_counts[still_representation] += 1
        candidates = audit_row.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("audit adjudication candidates must be a list")
        row_conditions: dict[str, dict[str, bool]] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("audit candidate must be an object")
            opaque_id = candidate.get("opaque_candidate_id")
            if opaque_id not in candidate_condition_map:
                raise ValueError(
                    "audit candidate is absent from candidate condition map"
                )
            condition = candidate_condition_map[opaque_id]
            metrics = _derived_candidate_metrics(candidate)
            final_pass = (
                still_representation == "still_frame_representable"
                and metrics["visible"]
                and metrics["relations_preserved"]
                and metrics["answer_sufficient"]
            )
            supplied_pass = candidate.get("candidate_passes")
            if supplied_pass is not None and supplied_pass is not final_pass:
                raise ValueError("candidate_passes does not match final judgments")
            pass_after_repair = round_name == "repair" and final_pass
            if candidate.get("pass_after_repair") is not pass_after_repair:
                raise ValueError(
                    "pass_after_repair does not match the final audit round"
                )
            for metric, passed in metrics.items():
                candidate_counts[condition][metric] += int(passed)
            candidate_counts[condition]["pass_after_repair"] += int(pass_after_repair)
            candidate_counts[condition]["final_pass"] += int(final_pass)
            row_conditions[condition] = {
                **metrics,
                "pass_after_repair": pass_after_repair,
                "final_pass": final_pass,
            }
        if set(row_conditions) != set(CANDIDATES):
            raise ValueError("every audit row must contain both candidates")
        private_rows.append(
            {
                "video_id": base_row["video_id"],
                "still_representation": still_representation,
                "candidate_results": row_conditions,
            }
        )

    for row in rows:
        if row.get("eligibility") != ELIGIBILITY_LABELABLE_VISUAL:
            private_rows.append(
                {
                    "video_id": row["video_id"],
                    "exclusion": row["eligibility"],
                }
            )
    return (
        dict(sorted(eligibility_counts.items())),
        still_counts,
        candidate_counts,
        private_rows,
    )


def _derive_selection_decision(
    *, cohort_count: int, candidate_counts: Mapping[str, Mapping[str, int]]
) -> tuple[str, str]:
    hybrid_passes = candidate_counts["oracle-hybrid-12"]["final_pass"]
    local_passes = candidate_counts["oracle-local-12"]["final_pass"]
    if cohort_count < 8:
        return (
            "stop_before_inference",
            "fewer_than_eight_still_frame_representable_rows",
        )
    if hybrid_passes == cohort_count:
        return (
            "select_oracle_hybrid_12",
            "hybrid_passes_every_prospective_paired_row",
        )
    if local_passes == cohort_count:
        return (
            "select_oracle_local_12",
            "local_passes_every_prospective_paired_row",
        )
    return (
        "stop_before_inference",
        "no_candidate_passes_every_prospective_paired_row",
    )


def build_stage2_selection_record(
    adjudication: dict[str, Any],
    scene_semantics: dict[str, Any],
    audit_adjudication: dict[str, Any],
    candidate_condition_map: dict[str, Any],
    *,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build the deterministic Stage 2 select-one-or-stop decision record."""

    if not all(
        isinstance(value, Mapping)
        for value in (
            adjudication,
            scene_semantics,
            audit_adjudication,
            candidate_condition_map,
            bindings,
        )
    ):
        raise TypeError("selection inputs must be objects")
    _verify_adjudication_identity(adjudication)
    _verify_semantics_identity(scene_semantics)
    _verify_audit_adjudication_identity(audit_adjudication)
    raw_auditor_agreement = _validate_raw_agreement_counts(audit_adjudication)
    normalized_bindings = _verified_bindings(
        bindings,
        adjudication=adjudication,
        scene_semantics=scene_semantics,
        audit_adjudication=audit_adjudication,
    )
    labelable_rows = [
        row
        for row in _adjudication_rows(adjudication)
        if row.get("eligibility") == ELIGIBILITY_LABELABLE_VISUAL
    ]
    semantics_rows = scene_semantics.get("rows")
    if not isinstance(semantics_rows, list) or [
        row.get("video_id") if isinstance(row, Mapping) else None
        for row in semantics_rows
    ] != [row.get("video_id") for row in labelable_rows]:
        raise ValueError("scene semantics must match every labelable row in order")
    audit_rows = audit_adjudication.get("rows")
    if not isinstance(audit_rows, list) or any(
        not isinstance(row, Mapping) for row in audit_rows
    ):
        raise TypeError("audit adjudication rows must be a list of objects")
    mapping = _candidate_mapping(candidate_condition_map, audit_rows)
    eligibility_counts, still_counts, candidate_counts, private_rows = (
        _stage1_aggregates(adjudication, audit_adjudication, mapping)
    )
    cohort_count = still_counts["still_frame_representable"]
    decision, decision_reason = _derive_selection_decision(
        cohort_count=cohort_count, candidate_counts=candidate_counts
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "total_row_count": 10,
        "labelable_row_count": eligibility_counts.get(ELIGIBILITY_LABELABLE_VISUAL, 0),
        "prospective_paired_cohort_count": cohort_count,
        "minimum_prospective_paired_cohort_count": 8,
        "candidate_counts": candidate_counts,
        "decision": decision,
        "decision_reason": decision_reason,
        "raw_auditor_agreement": raw_auditor_agreement,
        "rows": private_rows,
        "candidate_condition_map": mapping,
        "candidate_condition_map_sha256": _sha256(mapping),
        "bindings": normalized_bindings,
        "bindings_sha256": _sha256(normalized_bindings),
    }
    record["selection_sha256"] = _sha256(record)
    return record


_PRIVATE_BINDING_TOKENS = frozenset(
    {
        "video",
        "videos",
        "question",
        "questions",
        "interval",
        "intervals",
        "path",
        "paths",
        "frame",
        "frames",
        "response",
        "responses",
        "repair",
        "repairs",
        "timestamp",
        "timestamps",
    }
)


def _reject_private_bindings(bindings: Mapping[str, Any]) -> None:
    for key in bindings:
        tokens = {token.casefold() for token in _KEY_BOUNDARY.split(str(key)) if token}
        if tokens & _PRIVATE_BINDING_TOKENS:
            raise ValueError("private material is forbidden in scoreboard bindings")


def build_stage1_scoreboard(
    adjudication: dict[str, Any],
    audit_adjudication: dict[str, Any],
    selection: dict[str, Any],
    *,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build a public aggregate scoreboard with no row-level material."""

    if not all(
        isinstance(value, Mapping)
        for value in (adjudication, audit_adjudication, selection, bindings)
    ):
        raise TypeError("scoreboard inputs must be objects")
    _verify_adjudication_identity(adjudication)
    _verify_audit_adjudication_identity(audit_adjudication)
    raw_auditor_agreement = _validate_raw_agreement_counts(audit_adjudication)
    _reject_private_bindings(bindings)
    normalized_bindings = _verified_bindings(
        bindings,
        adjudication=adjudication,
        audit_adjudication=audit_adjudication,
    )
    selection_sha256 = _sha256_value(
        selection.get("selection_sha256"), "selection_sha256"
    )
    if selection_sha256 != _hash_without(selection, "selection_sha256"):
        raise ValueError("selection_sha256 does not match selection record")
    if selection.get("bindings_sha256") != _sha256(normalized_bindings):
        raise ValueError("scoreboard bindings do not match selection bindings")

    audit_rows = audit_adjudication.get("rows")
    if not isinstance(audit_rows, list) or any(
        not isinstance(row, Mapping) for row in audit_rows
    ):
        raise TypeError("audit adjudication rows must be a list of objects")
    stored_map = selection.get("candidate_condition_map")
    if not isinstance(stored_map, Mapping):
        raise ValueError("selection is missing its private candidate condition map")
    condition_by_opaque = _candidate_mapping(stored_map, audit_rows)
    if selection.get("candidate_condition_map_sha256") != _sha256(condition_by_opaque):
        raise ValueError("candidate condition map hash does not match selection")
    eligibility_counts, still_counts, candidate_counts, _ = _stage1_aggregates(
        adjudication, audit_adjudication, condition_by_opaque
    )
    if candidate_counts != selection.get("candidate_counts"):
        raise ValueError("selection candidate counts do not match audit adjudication")
    if selection.get("raw_auditor_agreement") != raw_auditor_agreement:
        raise ValueError(
            "selection raw_auditor_agreement does not match audit adjudication"
        )
    labelable_count = eligibility_counts.get(ELIGIBILITY_LABELABLE_VISUAL, 0)
    cohort_count = still_counts["still_frame_representable"]
    decision, decision_reason = _derive_selection_decision(
        cohort_count=cohort_count, candidate_counts=candidate_counts
    )
    expected_selection_fields = {
        "total_row_count": 10,
        "labelable_row_count": labelable_count,
        "prospective_paired_cohort_count": cohort_count,
        "minimum_prospective_paired_cohort_count": 8,
        "decision": decision,
        "decision_reason": decision_reason,
    }
    for field, expected in expected_selection_fields.items():
        if selection.get(field) != expected:
            raise ValueError(f"selection {field} does not match derived decision")
    scoreboard = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "hashes": {
            "adjudication_sha256": adjudication["annotation_sha256"],
            "audit_adjudication_sha256": audit_adjudication[
                "audit_adjudication_sha256"
            ],
            "selection_sha256": selection_sha256,
            "bindings_sha256": _sha256(normalized_bindings),
        },
        "decision": decision,
        "denominators": {
            "all_rows": 10,
            "labelable_rows": labelable_count,
            "still_frame_representable_rows": still_counts["still_frame_representable"],
        },
        "eligibility_counts": eligibility_counts,
        "still_representation_counts": still_counts,
        "candidate_counts": candidate_counts,
        "raw_auditor_agreement": raw_auditor_agreement,
    }
    scoreboard["scoreboard_sha256"] = _sha256(scoreboard)
    return scoreboard


__all__ = [
    "CANDIDATES",
    "PRESERVATION_VALUES",
    "RELATION_VALUES",
    "STILL_REPRESENTATION_VALUES",
    "SUFFICIENCY_VALUES",
    "VISIBILITY_VALUES",
    "build_stage1_scoreboard",
    "build_stage2_selection_record",
    "freeze_json_artifact",
    "validate_audit_adjudication",
    "validate_frame_first_audit",
    "validate_full_video_audit",
    "validate_scene_semantics",
    "verify_json_artifact",
]
