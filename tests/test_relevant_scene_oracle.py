from __future__ import annotations

import copy
import hashlib
import json

import pytest

from trec_video_2026.vqa_experiments.relevant_scene_oracle import (
    build_stage1_scoreboard,
    build_stage2_selection_record,
    validate_audit_adjudication,
    validate_frame_first_audit,
    validate_full_video_audit,
    validate_scene_semantics,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    ROLE_ADJUDICATOR,
    validate_annotation,
)


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def adjudication() -> dict:
    panel = {
        "schema_version": 1,
        "panel_id": "synthetic-panel",
        "examples": [
            {"video_id": "synthetic-1", "duration_seconds": 10.0},
            {"video_id": "synthetic-2", "duration_seconds": 12.0},
            {"video_id": "synthetic-ineligible", "duration_seconds": 8.0},
        ],
    }
    payload = {
        "schema_version": 1,
        "reviewer": "base-adjudicator",
        "role": ROLE_ADJUDICATOR,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _sha256(panel),
        "started_at": "2026-07-30T10:00:00Z",
        "ended_at": "2026-07-30T10:30:00Z",
        "input_annotation_sha256s": ["1" * 64, "2" * 64, "3" * 64],
        "rows": [
            {
                "video_id": "synthetic-1",
                "eligibility": "labelable_visual",
                "sufficient_sets": [
                    [[1.0, 2.0], [4.0, 5.0]],
                    [[7.0, 8.0]],
                ],
                "preferred_set_index": 0,
            },
            {
                "video_id": "synthetic-2",
                "eligibility": "labelable_visual",
                "sufficient_sets": [[[2.0, 3.0]]],
                "preferred_set_index": 0,
            },
            {
                "video_id": "synthetic-ineligible",
                "eligibility": "not_visually_answerable",
            },
        ],
    }
    return validate_annotation(payload, panel, expected_role=ROLE_ADJUDICATOR)


@pytest.fixture
def scene_semantics_payload(adjudication: dict) -> dict:
    return {
        "schema_version": 1,
        "role": "scene_semanticist",
        "reviewer": "semanticist-a",
        "adjudication_sha256": adjudication["annotation_sha256"],
        "started_at": "2026-07-30T10:31:00Z",
        "ended_at": "2026-07-30T10:45:00Z",
        "rows": [
            {
                "video_id": "synthetic-1",
                "preferred_set_index": 0,
                "components": [
                    {
                        "component_id": "component-1a",
                        "interval_index": 0,
                        "roles": ["anchor", "precondition"],
                    },
                    {
                        "component_id": "component-1b",
                        "interval_index": 1,
                        "roles": ["target", "outcome"],
                    },
                ],
                "required_relationships": [
                    {
                        "relationship_id": "relation-1",
                        "type": "ordering",
                        "component_ids": ["component-1a", "component-1b"],
                    },
                    {
                        "relationship_id": "relation-2",
                        "type": "transition",
                        "component_ids": ["component-1b"],
                    },
                ],
            },
            {
                "video_id": "synthetic-2",
                "preferred_set_index": 0,
                "components": [
                    {
                        "component_id": "component-2a",
                        "interval_index": 0,
                        "roles": ["target"],
                    }
                ],
                "required_relationships": [
                    {
                        "relationship_id": "relation-3",
                        "type": "motion",
                        "component_ids": ["component-2a"],
                    }
                ],
            },
        ],
    }


def test_scene_semantics_freezes_every_preferred_component_and_relationship(
    adjudication: dict, scene_semantics_payload: dict
) -> None:
    result = validate_scene_semantics(scene_semantics_payload, adjudication)

    assert [row["video_id"] for row in result["rows"]] == [
        "synthetic-1",
        "synthetic-2",
    ]
    assert [
        component["interval_index"] for component in result["rows"][0]["components"]
    ] == [0, 1]
    assert result["semantics_sha256"] == _sha256(scene_semantics_payload)
    assert validate_scene_semantics(result, adjudication) == result


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["components"].pop(), "preferred.*interval"),
        (
            lambda row: row["components"].append(
                {
                    "component_id": "alternative-component",
                    "interval_index": 2,
                    "roles": ["context"],
                }
            ),
            "preferred.*interval",
        ),
        (
            lambda row: row["components"].__setitem__(
                1,
                {
                    "component_id": "component-1a",
                    "interval_index": 1,
                    "roles": ["target"],
                },
            ),
            "duplicate component_id",
        ),
        (
            lambda row: row["required_relationships"][0].__setitem__(
                "component_ids", ["component-1a", "unbound-component"]
            ),
            "unbound component",
        ),
        (
            lambda row: row["components"][0].__setitem__("roles", ["subject"]),
            "component role",
        ),
        (
            lambda row: row["required_relationships"][0].__setitem__(
                "type", "causality"
            ),
            "relationship type",
        ),
        (
            lambda row: row["required_relationships"][0].__setitem__(
                "component_ids", ["component-1a"]
            ),
            "ordering.*two",
        ),
    ],
)
def test_scene_semantics_rejects_nonexact_or_noncanonical_structure(
    adjudication: dict,
    scene_semantics_payload: dict,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(scene_semantics_payload)
    mutation(payload["rows"][0])

    with pytest.raises(ValueError, match=message):
        validate_scene_semantics(payload, adjudication)


def test_scene_semantics_rejects_duplicate_relationship_ids(
    adjudication: dict, scene_semantics_payload: dict
) -> None:
    payload = copy.deepcopy(scene_semantics_payload)
    payload["rows"][0]["required_relationships"][1]["relationship_id"] = "relation-1"

    with pytest.raises(ValueError, match="duplicate relationship_id"):
        validate_scene_semantics(payload, adjudication)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("started_at", "2026-07-30T10:30:00Z", "after.*adjudication"),
        ("ended_at", "2026-07-30T10:31:00Z", "after started_at"),
        ("adjudication_sha256", "f" * 64, "adjudication_sha256"),
    ],
)
def test_scene_semantics_requires_post_adjudication_bound_timestamps(
    adjudication: dict,
    scene_semantics_payload: dict,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = copy.deepcopy(scene_semantics_payload)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_scene_semantics(payload, adjudication)


@pytest.fixture
def audit_packet() -> dict:
    return {
        "schema_version": 1,
        "packet_sha256": "a" * 64,
        "rows": [
            {
                "opaque_row_id": "row-alpha",
                "candidate_order": ["candidate-k", "candidate-m"],
                "candidates": [
                    {"opaque_candidate_id": "candidate-k"},
                    {"opaque_candidate_id": "candidate-m"},
                ],
            },
            {
                "opaque_row_id": "row-beta",
                "candidate_order": ["candidate-m", "candidate-k"],
                "candidates": [
                    {"opaque_candidate_id": "candidate-m"},
                    {"opaque_candidate_id": "candidate-k"},
                ],
            },
        ],
    }


@pytest.fixture
def frame_first_payload() -> dict:
    return {
        "schema_version": 1,
        "role": "scene_auditor",
        "reviewer": "auditor-left",
        "packet_sha256": "a" * 64,
        "started_at": "2026-07-30T12:00:00Z",
        "ended_at": "2026-07-30T12:05:00Z",
        "rows": [
            {
                "opaque_row_id": "row-alpha",
                "candidate_order": ["candidate-k", "candidate-m"],
                "candidates": [
                    {
                        "opaque_candidate_id": "candidate-k",
                        "response_kind": "not_answerable",
                        "response": None,
                        "confidence": "low",
                        "available_relations": ["ordering"],
                    },
                    {
                        "opaque_candidate_id": "candidate-m",
                        "response_kind": "answer",
                        "response": "the object changes state",
                        "confidence": "high",
                        "available_relations": ["transition", "motion"],
                    },
                ],
            },
            {
                "opaque_row_id": "row-beta",
                "candidate_order": ["candidate-m", "candidate-k"],
                "candidates": [
                    {
                        "opaque_candidate_id": "candidate-m",
                        "response_kind": "not_answerable",
                        "response": None,
                        "confidence": "medium",
                        "available_relations": ["uncertain"],
                    },
                    {
                        "opaque_candidate_id": "candidate-k",
                        "response_kind": "answer",
                        "response": "the second event follows the first",
                        "confidence": "medium",
                        "available_relations": ["ordering", "duration"],
                    },
                ],
            },
        ],
    }


def test_frame_first_audit_preserves_opaque_packet_order_and_is_idempotent(
    audit_packet: dict, frame_first_payload: dict
) -> None:
    result = validate_frame_first_audit(
        frame_first_payload, audit_packet, expected_reviewer="auditor-left"
    )

    assert [row["opaque_row_id"] for row in result["rows"]] == [
        "row-alpha",
        "row-beta",
    ]
    assert result["rows"][1]["candidate_order"] == [
        "candidate-m",
        "candidate-k",
    ]
    assert result["audit_sha256"] == _sha256(frame_first_payload)
    assert (
        validate_frame_first_audit(
            result, audit_packet, expected_reviewer="auditor-left"
        )
        == result
    )


@pytest.mark.parametrize("response", ["yes/no/maybe", "red/green/blue", "up/down/left"])
def test_frame_first_audit_allows_slash_separated_answer_text(
    audit_packet: dict, frame_first_payload: dict, response: str
) -> None:
    payload = copy.deepcopy(frame_first_payload)
    payload["rows"][0]["candidates"][1]["response"] = response

    result = validate_frame_first_audit(
        payload, audit_packet, expected_reviewer="auditor-left"
    )

    assert result["rows"][0]["candidates"][1]["response"] == response
    assert result["audit_sha256"] == _sha256(payload)


@pytest.mark.parametrize(
    "response",
    [
        "/private/frame.png",
        "~/private/frame.png",
        "./private/frame.png",
        "../private/frame.png",
        "https://example.invalid/frame.png",
        "private\\frame.png",
        "C:/private/frame.png",
        "images/frame.png",
        "private/contact-sheets/row-alpha",
    ],
)
def test_frame_first_audit_rejects_explicit_path_syntax(
    audit_packet: dict, frame_first_payload: dict, response: str
) -> None:
    payload = copy.deepcopy(frame_first_payload)
    payload["rows"][0]["candidates"][1]["response"] = response

    with pytest.raises(ValueError, match="path identity"):
        validate_frame_first_audit(
            payload, audit_packet, expected_reviewer="auditor-left"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["rows"].pop(), "packet rows"),
        (
            lambda payload: payload["rows"][1].__setitem__(
                "opaque_row_id", "row-alpha"
            ),
            "duplicate opaque_row_id",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].__setitem__(
                "opaque_candidate_id", "candidate-unknown"
            ),
            "candidate order|unknown candidate",
        ),
        (
            lambda payload: payload["rows"][0]["candidate_order"].reverse(),
            "candidate_order",
        ),
        (
            lambda payload: payload.__setitem__("ended_at", payload["started_at"]),
            "after started_at",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].__setitem__(
                "response", "copied answer text"
            ),
            "not_answerable.*null",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].pop("confidence"),
            "confidence",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].__setitem__(
                "available_relations", ["before_after"]
            ),
            "available_relations",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__("reference", "private"),
            "forbidden key.*reference",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__(
                "frame_path", "/private/frame.png"
            ),
            "forbidden key.*frame_path",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][1].__setitem__(
                "response", "oracle-hybrid-12"
            ),
            "condition identity",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__(
                "video_id", "private-video-id"
            ),
            "forbidden key.*video_id",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][1].__setitem__(
                "response", "uniform-12"
            ),
            "condition identity",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][1].__setitem__(
                "response", "ORACLE-HYBRID-12"
            ),
            "condition identity",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][1].__setitem__(
                "response", "private/contact-sheets/row-alpha.png"
            ),
            "path identity",
        ),
    ],
)
def test_frame_first_audit_rejects_missing_reordered_or_leaking_output(
    audit_packet: dict,
    frame_first_payload: dict,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(frame_first_payload)
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_frame_first_audit(
            payload, audit_packet, expected_reviewer="auditor-left"
        )


def test_frame_first_audit_rejects_wrong_reviewer_or_packet_binding(
    audit_packet: dict, frame_first_payload: dict
) -> None:
    with pytest.raises(ValueError, match="reviewer"):
        validate_frame_first_audit(
            frame_first_payload, audit_packet, expected_reviewer="auditor-right"
        )

    payload = copy.deepcopy(frame_first_payload)
    payload["packet_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="packet_sha256"):
        validate_frame_first_audit(
            payload, audit_packet, expected_reviewer="auditor-left"
        )


@pytest.fixture
def full_video_packet(audit_packet: dict) -> dict:
    packet = copy.deepcopy(audit_packet)
    packet["adjudication_sha256"] = "placeholder"
    packet["rows"][0]["video_id"] = "synthetic-1"
    packet["rows"][1]["video_id"] = "synthetic-2"
    for row, interval_count in zip(packet["rows"], [2, 1], strict=True):
        for candidate in row["candidates"]:
            candidate["intervals"] = [
                {
                    "opaque_interval_id": (
                        f"{row['opaque_row_id']}-interval-{index + 1}"
                    ),
                    "sufficient_set_index": 0,
                    "interval_index": index,
                    "slot_kind": "local",
                }
                for index in range(interval_count)
            ]
    return packet


def _audit_candidate(
    opaque_candidate_id: str,
    interval_ids: list[str],
    relation_types: list[str],
) -> dict:
    return {
        "opaque_candidate_id": opaque_candidate_id,
        "intervals": [
            {"opaque_interval_id": interval_id, "visibility": "visible"}
            for interval_id in interval_ids
        ],
        "relations": [
            {"relation": relation, "preservation": "preserved"}
            for relation in relation_types
        ],
        "answer_sufficiency": "answer_sufficient",
        "repair_proposals": [],
    }


@pytest.fixture
def full_video_payload(full_video_packet: dict, adjudication: dict) -> dict:
    full_video_packet["adjudication_sha256"] = adjudication["annotation_sha256"]
    rows = []
    relation_types = [["ordering", "transition"], ["motion"]]
    for packet_row, relations in zip(
        full_video_packet["rows"], relation_types, strict=True
    ):
        rows.append(
            {
                "opaque_row_id": packet_row["opaque_row_id"],
                "still_representation": "still_frame_representable",
                "representation_classes": relations.copy(),
                "candidates": [
                    _audit_candidate(
                        candidate["opaque_candidate_id"],
                        [
                            interval["opaque_interval_id"]
                            for interval in candidate["intervals"]
                        ],
                        relations,
                    )
                    for candidate in packet_row["candidates"]
                ],
            }
        )
    return {
        "schema_version": 1,
        "role": "scene_auditor",
        "reviewer": "auditor-left",
        "packet_sha256": full_video_packet["packet_sha256"],
        "adjudication_sha256": adjudication["annotation_sha256"],
        "scene_semantics_sha256": "placeholder",
        "frame_first_audit_sha256": "placeholder",
        "started_at": "2026-07-30T12:06:00Z",
        "ended_at": "2026-07-30T12:15:00Z",
        "rows": rows,
    }


@pytest.fixture
def validated_semantics(adjudication: dict, scene_semantics_payload: dict) -> dict:
    return validate_scene_semantics(scene_semantics_payload, adjudication)


@pytest.fixture
def frame_first_freeze(audit_packet: dict, frame_first_payload: dict) -> dict:
    audit = validate_frame_first_audit(
        frame_first_payload, audit_packet, expected_reviewer="auditor-left"
    )
    bindings = {
        "audit_sha256": audit["audit_sha256"],
        "ended_at": audit["ended_at"],
        "reviewer": audit["reviewer"],
        "packet_sha256": audit["packet_sha256"],
    }
    return {
        "schema_version": 1,
        "status": "frozen",
        "artifact_sha256": "d" * 64,
        "bindings": bindings,
        "bindings_sha256": _sha256(bindings),
    }


def _bind_full_video_payload(
    payload: dict, semantics: dict, frame_first_freeze: dict
) -> dict:
    bound = copy.deepcopy(payload)
    bound["scene_semantics_sha256"] = semantics["semantics_sha256"]
    bound["frame_first_audit_sha256"] = frame_first_freeze["bindings"]["audit_sha256"]
    return bound


def test_full_video_audit_derives_exact_preferred_set_candidate_passes(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )

    result = validate_full_video_audit(
        payload,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        expected_reviewer="auditor-left",
    )

    assert result["rows"][0]["candidates"][0]["all_intervals_visible"] is True
    assert result["rows"][0]["candidates"][0]["all_relationships_preserved"] is True
    assert result["rows"][0]["candidates"][0]["candidate_passes"] is True
    assert result["audit_sha256"] == _sha256(payload)
    assert (
        validate_full_video_audit(
            result,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )
        == result
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["rows"][0]["candidates"][0]["intervals"].pop(),
            "preferred-set intervals",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0]["intervals"][
                0
            ].__setitem__("opaque_interval_id", "alternative-set-interval"),
            "preferred-set intervals",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0]["intervals"][
                0
            ].__setitem__("visibility", "mostly_visible"),
            "visibility",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0]["relations"][
                0
            ].__setitem__("preservation", "mostly_preserved"),
            "preservation",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__(
                "still_representation", "video_required"
            ),
            "still_representation",
        ),
        (
            lambda payload: payload["rows"][0].__setitem__(
                "representation_classes", ["causality"]
            ),
            "representation_classes",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].__setitem__(
                "answer_sufficiency", "mostly_sufficient"
            ),
            "answer_sufficiency",
        ),
        (
            lambda payload: payload["rows"][0]["candidates"][0].__setitem__(
                "condition", "oracle-hybrid-12"
            ),
            "forbidden key.*condition",
        ),
    ],
)
def test_full_video_audit_rejects_nonexact_or_noncanonical_judgments(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    mutation,
    message: str,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


def test_full_video_audit_requires_bound_hashes_and_post_freeze_start(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    payload["started_at"] = frame_first_freeze["bindings"]["ended_at"]
    with pytest.raises(ValueError, match="after.*frame-first"):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


@pytest.mark.parametrize(
    ("binding", "value", "message"),
    [
        ("reviewer", "auditor-right", "frame-first.*reviewer"),
        ("packet_sha256", "b" * 64, "frame-first.*packet_sha256"),
    ],
)
def test_full_video_audit_requires_same_reviewer_and_packet_in_frame_first_freeze(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    binding: str,
    value: str,
    message: str,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    freeze = copy.deepcopy(frame_first_freeze)
    freeze["bindings"][binding] = value
    freeze["bindings_sha256"] = _sha256(freeze["bindings"])

    with pytest.raises(ValueError, match=message):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            freeze,
            expected_reviewer="auditor-left",
        )

    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    payload["adjudication_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="adjudication_sha256"):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


@pytest.mark.parametrize(
    ("proposed_time", "extra", "message"),
    [
        (2.5, {}, "inside.*interval"),
        (1.55, {}, "0.1 seconds"),
        (1.5, {"target_slot_kind": "global"}, "local"),
    ],
)
def test_full_video_audit_rejects_invalid_repair_proposals(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    proposed_time: float,
    extra: dict,
    message: str,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    candidate = payload["rows"][0]["candidates"][0]
    candidate["intervals"][0]["visibility"] = "partial"
    candidate["repair_proposals"] = [
        {
            "opaque_interval_id": "row-alpha-interval-1",
            "proposed_time_seconds": proposed_time,
            "replacement_reason": "event center missed",
            **extra,
        }
    ]

    with pytest.raises(ValueError, match=message):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row, candidate: row.__setitem__(
                "still_representation", "still_frame_incompatible"
            ),
            "still_frame_representable",
        ),
        (
            lambda row, candidate: row.__setitem__("still_representation", "uncertain"),
            "still_frame_representable",
        ),
        (
            lambda row, candidate: candidate["relations"][0].__setitem__(
                "preservation", "not_preserved"
            ),
            "relationship",
        ),
        (
            lambda row, candidate: candidate["relations"][0].__setitem__(
                "preservation", "uncertain"
            ),
            "relationship",
        ),
        (
            lambda row, candidate: candidate.__setitem__(
                "answer_sufficiency", "answer_insufficient"
            ),
            "answer_sufficiency",
        ),
        (
            lambda row, candidate: candidate.__setitem__(
                "answer_sufficiency", "uncertain"
            ),
            "answer_sufficiency",
        ),
    ],
)
def test_full_video_repair_rejects_non_interval_failure_classes(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    mutation,
    message: str,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    row = payload["rows"][0]
    candidate = row["candidates"][0]
    candidate["intervals"][0]["visibility"] = "partial"
    candidate["repair_proposals"] = [
        {
            "opaque_interval_id": "row-alpha-interval-1",
            "proposed_time_seconds": 1.5,
            "replacement_reason": "event center missed",
        }
    ]
    mutation(row, candidate)

    with pytest.raises(ValueError, match=message):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


def test_full_video_repair_must_target_every_failed_local_interval(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
) -> None:
    payload = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    candidate = payload["rows"][0]["candidates"][0]
    for interval in candidate["intervals"]:
        interval["visibility"] = "partial"
    candidate["repair_proposals"] = [
        {
            "opaque_interval_id": "row-alpha-interval-1",
            "proposed_time_seconds": 1.5,
            "replacement_reason": "first event center missed",
        }
    ]

    with pytest.raises(ValueError, match="every failed local interval"):
        validate_full_video_audit(
            payload,
            full_video_packet,
            adjudication,
            validated_semantics,
            frame_first_freeze,
            expected_reviewer="auditor-left",
        )


def _validated_full_audit(
    payload: dict,
    packet: dict,
    adjudication: dict,
    semantics: dict,
    freeze: dict,
    reviewer: str,
) -> dict:
    candidate = copy.deepcopy(payload)
    candidate["reviewer"] = reviewer
    reviewer_freeze = copy.deepcopy(freeze)
    reviewer_freeze["bindings"]["reviewer"] = reviewer
    reviewer_freeze["bindings_sha256"] = _sha256(reviewer_freeze["bindings"])
    return validate_full_video_audit(
        candidate,
        packet,
        adjudication,
        semantics,
        reviewer_freeze,
        expected_reviewer=reviewer,
    )


def _audit_hash_payload(value: object) -> object:
    derived_fields = {
        "all_intervals_visible",
        "all_relationships_preserved",
        "candidate_passes",
    }
    if isinstance(value, dict):
        return {
            key: _audit_hash_payload(nested)
            for key, nested in value.items()
            if key not in derived_fields
        }
    if isinstance(value, list):
        return [_audit_hash_payload(nested) for nested in value]
    return value


def _rehash_audit(audit: dict, hash_field: str = "audit_sha256") -> None:
    audit[hash_field] = _sha256(
        _audit_hash_payload(
            {key: value for key, value in audit.items() if key != hash_field}
        )
    )


def test_third_auditor_adjudication_binds_two_audits_and_counts_raw_agreement(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
) -> None:
    bound = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    left = _validated_full_audit(
        bound,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-left",
    )
    right_payload = copy.deepcopy(bound)
    right_payload["rows"][0]["candidates"][0]["intervals"][0]["visibility"] = "partial"
    right = _validated_full_audit(
        right_payload,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-right",
    )
    payload = {
        "schema_version": 1,
        "role": "scene_audit_adjudicator",
        "reviewer": "auditor-third",
        "round_name": "initial",
        "packet_sha256": full_video_packet["packet_sha256"],
        "adjudication_sha256": adjudication["annotation_sha256"],
        "scene_semantics_sha256": validated_semantics["semantics_sha256"],
        "input_audit_sha256s": [left["audit_sha256"], right["audit_sha256"]],
        "started_at": "2026-07-30T12:16:00Z",
        "ended_at": "2026-07-30T12:25:00Z",
        "rows": copy.deepcopy(left["rows"]),
    }

    result = validate_audit_adjudication(
        payload,
        left,
        right,
        full_video_packet,
        adjudication,
        validated_semantics,
        round_name="initial",
    )

    assert result["raw_agreement_count"] > 0
    assert result["raw_disagreement_count"] == 1
    assert result["raw_comparison_count"] == (
        result["raw_agreement_count"] + result["raw_disagreement_count"]
    )
    assert result["raw_agreement_by_dimension"] == {
        "representability": {
            "agreement_count": 2,
            "disagreement_count": 0,
            "comparison_count": 2,
        },
        "interval_visibility": {
            "agreement_count": 5,
            "disagreement_count": 1,
            "comparison_count": 6,
        },
        "relationship_preservation": {
            "agreement_count": 6,
            "disagreement_count": 0,
            "comparison_count": 6,
        },
        "answer_sufficiency": {
            "agreement_count": 4,
            "disagreement_count": 0,
            "comparison_count": 4,
        },
    }
    assert result["rows"][0]["candidates"][0]["candidate_passes"] is True
    assert (
        validate_audit_adjudication(
            result,
            left,
            right,
            full_video_packet,
            adjudication,
            validated_semantics,
            round_name="initial",
        )
        == result
    )


@pytest.mark.parametrize(
    "field",
    ["packet_sha256", "adjudication_sha256", "scene_semantics_sha256"],
)
def test_audit_adjudication_rejects_self_rehashed_input_with_wrong_upstream(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    field: str,
) -> None:
    bound = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    left = _validated_full_audit(
        bound,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-left",
    )
    right = _validated_full_audit(
        bound,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-right",
    )
    left[field] = "f" * 64
    _rehash_audit(left)
    payload = {
        "schema_version": 1,
        "role": "scene_audit_adjudicator",
        "reviewer": "auditor-third",
        "round_name": "initial",
        "packet_sha256": full_video_packet["packet_sha256"],
        "adjudication_sha256": adjudication["annotation_sha256"],
        "scene_semantics_sha256": validated_semantics["semantics_sha256"],
        "input_audit_sha256s": [left["audit_sha256"], right["audit_sha256"]],
        "started_at": "2026-07-30T12:16:00Z",
        "ended_at": "2026-07-30T12:25:00Z",
        "rows": copy.deepcopy(right["rows"]),
    }

    with pytest.raises(ValueError, match=field):
        validate_audit_adjudication(
            payload,
            left,
            right,
            full_video_packet,
            adjudication,
            validated_semantics,
            round_name="initial",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("round_name", "second"), "round_name"),
        (
            lambda payload: payload.__setitem__("input_audit_sha256s", ["1" * 64]),
            "exactly two",
        ),
        (
            lambda payload: payload.__setitem__(
                "input_audit_sha256s", ["1" * 64, "1" * 64]
            ),
            "distinct",
        ),
        (
            lambda payload: payload.__setitem__("reviewer", "auditor-left"),
            "third reviewer",
        ),
    ],
)
def test_audit_adjudication_requires_third_reviewer_round_and_two_frozen_hashes(
    full_video_payload: dict,
    full_video_packet: dict,
    adjudication: dict,
    validated_semantics: dict,
    frame_first_freeze: dict,
    mutation,
    message: str,
) -> None:
    bound = _bind_full_video_payload(
        full_video_payload, validated_semantics, frame_first_freeze
    )
    left = _validated_full_audit(
        bound,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-left",
    )
    right = _validated_full_audit(
        bound,
        full_video_packet,
        adjudication,
        validated_semantics,
        frame_first_freeze,
        "auditor-right",
    )
    payload = {
        "schema_version": 1,
        "role": "scene_audit_adjudicator",
        "reviewer": "auditor-third",
        "round_name": "initial",
        "packet_sha256": full_video_packet["packet_sha256"],
        "adjudication_sha256": adjudication["annotation_sha256"],
        "scene_semantics_sha256": validated_semantics["semantics_sha256"],
        "input_audit_sha256s": [left["audit_sha256"], right["audit_sha256"]],
        "started_at": "2026-07-30T12:16:00Z",
        "ended_at": "2026-07-30T12:25:00Z",
        "rows": copy.deepcopy(left["rows"]),
    }
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_audit_adjudication(
            payload,
            left,
            right,
            full_video_packet,
            adjudication,
            validated_semantics,
            round_name="initial",
        )


def _stage1_inputs(
    cohort: int, hybrid_passes: int, local_passes: int
) -> tuple[dict, dict, dict, dict, dict]:
    if not 0 <= cohort <= 8:
        raise AssertionError("synthetic cohort must be between zero and eight")
    adjudication = {
        "schema_version": 1,
        "role": "adjudicator",
        "rows": [
            {
                "video_id": f"synthetic-{index}",
                "eligibility": (
                    "labelable_visual" if index < 8 else "ambiguous_question"
                ),
            }
            for index in range(10)
        ],
    }
    adjudication["annotation_sha256"] = _sha256(adjudication)
    scene_semantics = {
        "schema_version": 1,
        "rows": [
            {"video_id": f"synthetic-{index}", "components": []} for index in range(8)
        ],
    }
    scene_semantics["semantics_sha256"] = _sha256(scene_semantics)

    audit_rows = []
    for index in range(8):
        representable = index < cohort
        candidates = []
        for candidate_id, pass_count in (
            ("candidate-k", hybrid_passes),
            ("candidate-m", local_passes),
        ):
            passes = representable and index < pass_count
            candidates.append(
                {
                    "opaque_candidate_id": candidate_id,
                    "intervals": [
                        {
                            "opaque_interval_id": f"private-interval-{index}",
                            "visibility": "visible" if passes else "not_visible",
                        }
                    ],
                    "relations": [
                        {
                            "relation": "ordering",
                            "preservation": "preserved" if passes else "not_preserved",
                        }
                    ],
                    "answer_sufficiency": (
                        "answer_sufficient" if passes else "answer_insufficient"
                    ),
                    "all_intervals_visible": passes,
                    "all_relationships_preserved": passes,
                    "candidate_passes": passes,
                    "pass_after_repair": passes and index % 2 == 0,
                    "repair_proposals": (
                        []
                        if passes
                        else [
                            {
                                "opaque_interval_id": f"private-interval-{index}",
                                "proposed_time_seconds": 1.5,
                            }
                        ]
                    ),
                }
            )
        audit_rows.append(
            {
                "opaque_row_id": f"private-row-{index}",
                "still_representation": (
                    "still_frame_representable"
                    if representable
                    else "still_frame_incompatible"
                ),
                "representation_classes": ["ordering"],
                "candidates": candidates,
            }
        )
    audit_adjudication = {
        "schema_version": 1,
        "round_name": "repair",
        "raw_agreement_by_dimension": {
            "representability": {
                "agreement_count": 8,
                "disagreement_count": 0,
                "comparison_count": 8,
            },
            "interval_visibility": {
                "agreement_count": 16,
                "disagreement_count": 0,
                "comparison_count": 16,
            },
            "relationship_preservation": {
                "agreement_count": 16,
                "disagreement_count": 0,
                "comparison_count": 16,
            },
            "answer_sufficiency": {
                "agreement_count": 16,
                "disagreement_count": 0,
                "comparison_count": 16,
            },
        },
        "raw_agreement_count": 56,
        "raw_disagreement_count": 0,
        "raw_comparison_count": 56,
        "rows": audit_rows,
    }
    _rehash_audit(audit_adjudication, "audit_adjudication_sha256")
    candidate_condition_map = {
        "candidate-k": "oracle-hybrid-12",
        "candidate-m": "oracle-local-12",
    }
    bindings = {
        "protocol_sha256": "1" * 64,
        "adjudication_sha256": adjudication["annotation_sha256"],
        "scene_semantics_sha256": scene_semantics["semantics_sha256"],
        "audit_adjudication_sha256": audit_adjudication["audit_adjudication_sha256"],
        "oracle_hybrid_12_plan_sha256": "2" * 64,
        "oracle_local_12_plan_sha256": "3" * 64,
    }
    return (
        adjudication,
        scene_semantics,
        audit_adjudication,
        candidate_condition_map,
        bindings,
    )


def _selection(cohort: int, hybrid: int, local: int) -> dict:
    inputs = _stage1_inputs(cohort, hybrid, local)
    return build_stage2_selection_record(
        inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
    )


@pytest.mark.parametrize(
    ("cohort", "hybrid", "local", "decision"),
    [
        (7, 7, 7, "stop_before_inference"),
        (8, 8, 8, "select_oracle_hybrid_12"),
        (8, 7, 8, "select_oracle_local_12"),
        (8, 7, 7, "stop_before_inference"),
    ],
)
def test_stage2_selection_is_predeclared(
    cohort: int, hybrid: int, local: int, decision: str
) -> None:
    result = _selection(cohort, hybrid, local)

    assert result["decision"] == decision
    assert result["total_row_count"] == 10
    assert result["prospective_paired_cohort_count"] == cohort
    assert result["candidate_counts"]["oracle-hybrid-12"]["final_pass"] == hybrid
    assert result["candidate_counts"]["oracle-local-12"]["final_pass"] == local


def test_stage2_selection_uses_every_prospective_row_and_exact_candidate_map() -> None:
    inputs = _stage1_inputs(8, 7, 8)
    audit = inputs[2]
    audit["rows"][7]["candidates"][1]["candidate_passes"] = False
    audit["rows"][7]["candidates"][1]["all_intervals_visible"] = False
    audit["rows"][7]["candidates"][1]["intervals"][0]["visibility"] = "not_visible"
    _rehash_audit(audit, "audit_adjudication_sha256")

    result = build_stage2_selection_record(
        inputs[0],
        inputs[1],
        audit,
        inputs[3],
        bindings={
            **inputs[4],
            "audit_adjudication_sha256": audit["audit_adjudication_sha256"],
        },
    )
    assert result["decision"] == "stop_before_inference"

    with pytest.raises(ValueError, match="candidate condition map"):
        build_stage2_selection_record(
            inputs[0],
            inputs[1],
            audit,
            {"candidate-k": "oracle-hybrid-12"},
            bindings={
                **inputs[4],
                "audit_adjudication_sha256": audit["audit_adjudication_sha256"],
            },
        )


def test_stage2_selection_requires_all_ten_frozen_rows() -> None:
    inputs = list(_stage1_inputs(8, 8, 8))
    inputs[0]["rows"].pop()
    inputs[0]["annotation_sha256"] = _sha256(
        {key: value for key, value in inputs[0].items() if key != "annotation_sha256"}
    )
    inputs[4]["adjudication_sha256"] = inputs[0]["annotation_sha256"]

    with pytest.raises(ValueError, match="exactly ten"):
        build_stage2_selection_record(
            inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
        )


@pytest.mark.parametrize(
    ("artifact_index", "mutation", "message"),
    [
        (
            0,
            lambda artifact: artifact["rows"][0].__setitem__(
                "eligibility", "ambiguous_question"
            ),
            "annotation_sha256",
        ),
        (
            1,
            lambda artifact: artifact["rows"][0].__setitem__(
                "components", [{"component_id": "stale"}]
            ),
            "semantics_sha256",
        ),
        (
            2,
            lambda artifact: artifact["rows"][7]["candidates"][1].update(
                {
                    "intervals": [
                        {
                            "opaque_interval_id": "private-interval-7",
                            "visibility": "not_visible",
                        }
                    ],
                    "all_intervals_visible": False,
                    "candidate_passes": False,
                }
            ),
            "audit_adjudication_sha256",
        ),
    ],
)
def test_stage2_selection_rejects_stale_frozen_input_identity(
    artifact_index: int, mutation, message: str
) -> None:
    inputs = list(_stage1_inputs(8, 8, 8))
    mutation(inputs[artifact_index])

    with pytest.raises(ValueError, match=message):
        build_stage2_selection_record(
            inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
        )


def test_stage2_selection_rejects_self_rehashed_tampered_agreement_dimension() -> None:
    inputs = list(_stage1_inputs(8, 8, 8))
    audit = inputs[2]
    audit["raw_agreement_by_dimension"]["interval_visibility"]["agreement_count"] = 15
    _rehash_audit(audit, "audit_adjudication_sha256")
    inputs[4]["audit_adjudication_sha256"] = audit["audit_adjudication_sha256"]

    with pytest.raises(ValueError, match="raw.*interval_visibility"):
        build_stage2_selection_record(
            inputs[0], inputs[1], audit, inputs[3], bindings=inputs[4]
        )


def test_stage1_scoreboard_rejects_stale_audit_row_before_reaggregation() -> None:
    inputs = list(_stage1_inputs(8, 8, 8))
    selection = build_stage2_selection_record(
        inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
    )
    inputs[2]["rows"][7]["candidates"][1].update(
        {
            "intervals": [
                {
                    "opaque_interval_id": "private-interval-7",
                    "visibility": "not_visible",
                }
            ],
            "all_intervals_visible": False,
            "candidate_passes": False,
        }
    )

    with pytest.raises(ValueError, match="audit_adjudication_sha256"):
        build_stage1_scoreboard(inputs[0], inputs[2], selection, bindings=inputs[4])


def test_stage1_scoreboard_rejects_self_rehashed_path_valued_decision() -> None:
    inputs = _stage1_inputs(8, 8, 8)
    selection = build_stage2_selection_record(
        inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
    )
    selection["decision"] = "private/contact-sheets/row-alpha.png"
    selection["selection_sha256"] = _sha256(
        {key: value for key, value in selection.items() if key != "selection_sha256"}
    )

    with pytest.raises(ValueError, match="decision"):
        build_stage1_scoreboard(inputs[0], inputs[2], selection, bindings=inputs[4])


def _contains_forbidden_public_material(value: object) -> bool:
    forbidden_fragments = (
        "video_id",
        "private question text",
        "private-interval",
        "path",
        "candidate-k",
        "candidate-m",
        "frame_sha256",
        "response",
        "proposed_time",
        "repair_proposals",
    )
    serialized = json.dumps(value, sort_keys=True).casefold()
    return any(fragment in serialized for fragment in forbidden_fragments)


def test_stage1_scoreboard_contains_only_sanitized_all_ten_aggregates() -> None:
    inputs = _stage1_inputs(8, 7, 8)
    selection = build_stage2_selection_record(
        inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
    )

    scoreboard = build_stage1_scoreboard(
        inputs[0], inputs[2], selection, bindings=inputs[4]
    )

    assert scoreboard["valid"] is True
    assert scoreboard["decision"] == "select_oracle_local_12"
    assert scoreboard["denominators"] == {
        "all_rows": 10,
        "labelable_rows": 8,
        "still_frame_representable_rows": 8,
    }
    assert scoreboard["eligibility_counts"] == {
        "ambiguous_question": 2,
        "labelable_visual": 8,
    }
    assert scoreboard["still_representation_counts"] == {
        "still_frame_incompatible": 0,
        "still_frame_representable": 8,
        "uncertain": 0,
    }
    assert (
        scoreboard["raw_auditor_agreement"] == inputs[2]["raw_agreement_by_dimension"]
    )
    assert selection["raw_auditor_agreement"] == inputs[2]["raw_agreement_by_dimension"]
    for condition in ("oracle-hybrid-12", "oracle-local-12"):
        counts = scoreboard["candidate_counts"][condition]
        assert counts["all_ten_denominator"] == 10
        assert set(counts) == {
            "all_ten_denominator",
            "visible",
            "relations_preserved",
            "answer_sufficient",
            "pass_after_repair",
            "final_pass",
        }
    assert _contains_forbidden_public_material(scoreboard) is False


def test_stage1_scoreboard_rejects_private_material_in_copied_bindings() -> None:
    inputs = _stage1_inputs(8, 8, 8)
    selection = build_stage2_selection_record(
        inputs[0], inputs[1], inputs[2], inputs[3], bindings=inputs[4]
    )

    with pytest.raises(ValueError, match="private.*bindings"):
        build_stage1_scoreboard(
            inputs[0],
            inputs[2],
            selection,
            bindings={**inputs[4], "video_ids": ["synthetic-1"]},
        )
