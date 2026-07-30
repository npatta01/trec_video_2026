"""Public CPU-only contract tests for temporal required-moment evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from trec_video_2026.vqa_experiments import temporal_bottleneck
from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    ELIGIBILITY_LABELABLE_VISUAL,
    ELIGIBILITY_NOT_VISUALLY_ANSWERABLE,
    ROLE_ADJUDICATOR,
    ROLE_PRIMARY_REVIEWER,
    SCHEMA_VERSION,
    annotation_agreement,
    score_selector_recall,
    validate_annotation,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MEDIA_SHA = "a" * 64


def _ffprobe_payload() -> dict:
    return {
        "source_media_sha256": MEDIA_SHA,
        "streams": [{"index": 2, "codec_type": "video", "time_base": "1/1000"}],
        "frames": [
            {
                "media_type": "video",
                "stream_index": 2,
                "frame_index": 10,
                "best_effort_timestamp_time": "0.1250004",
            },
            {
                "media_type": "video",
                "stream_index": 2,
                "frame_index": 11,
                "best_effort_timestamp_time": "0.3750004",
            },
            {
                "media_type": "video",
                "stream_index": 2,
                "frame_index": 12,
                "best_effort_timestamp_time": "0.6250004",
            },
        ],
    }


def _probe_frame_inventory(payload: dict) -> list[dict]:
    probe = getattr(temporal_bottleneck, "probe_frame_inventory", None)
    assert callable(probe), "probe_frame_inventory must be implemented"
    return probe(payload, media_sha256=MEDIA_SHA)


def test_frame_inventory_binds_stable_indices_and_six_decimal_decoded_pts() -> None:
    inventory = _probe_frame_inventory(_ffprobe_payload())

    assert inventory == [
        {
            "stream_index": 2,
            "frame_index": 10,
            "actual_pts_seconds": 0.125,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        },
        {
            "stream_index": 2,
            "frame_index": 11,
            "actual_pts_seconds": 0.375,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        },
        {
            "stream_index": 2,
            "frame_index": 12,
            "actual_pts_seconds": 0.625,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        },
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["frames"][1].pop("best_effort_timestamp_time"), "PTS"),
        (
            lambda payload: payload["frames"][1].update(
                best_effort_timestamp_time="NaN"
            ),
            "finite",
        ),
        (
            lambda payload: payload["frames"][1].update(
                best_effort_timestamp_time="0.100000"
            ),
            "monotonic",
        ),
        (
            lambda payload: payload["frames"][1].update(
                best_effort_timestamp_time="0.1250004"
            ),
            "duplicate",
        ),
        (lambda payload: payload["streams"][0].pop("time_base"), "time_base"),
        (
            lambda payload: payload.update(source_media_sha256="b" * 64),
            "source_media_sha256",
        ),
    ],
)
def test_frame_inventory_rejects_unusable_or_unbound_pts(
    mutation: object, message: str
) -> None:
    payload = _ffprobe_payload()
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        _probe_frame_inventory(payload)


def _decoded_inventory(points: list[float]) -> list[dict]:
    return [
        {
            "stream_index": 0,
            "frame_index": index,
            "actual_pts_seconds": point,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        }
        for index, point in enumerate(points)
    ]


def _uniform_plan(
    duration: float, inventory: list[dict], count: int = 12
) -> list[dict]:
    planner = getattr(temporal_bottleneck, "uniform_decoded_frame_plan", None)
    assert callable(planner), "uniform_decoded_frame_plan must be implemented"
    return planner(duration, inventory, count=count)


def _oracle_plan(
    duration: float, sufficient_sets: object, inventory: list[dict]
) -> list[dict]:
    planner = getattr(temporal_bottleneck, "oracle_hybrid_decoded_frame_plan", None)
    assert callable(planner), "oracle_hybrid_decoded_frame_plan must be implemented"
    return planner(duration, sufficient_sets, inventory)


def test_uniform_plan_is_the_literal_twelve_midpoint_decoded_frame_plan() -> None:
    inventory = _decoded_inventory([index + 0.5 for index in range(12)])

    plan = _uniform_plan(12.0, inventory)

    assert plan == [
        {
            "slot_kind": "global",
            "desired_time_seconds": 0.5,
            "requested_time_seconds": 0.5,
            "stream_index": 0,
            "frame_index": 0,
            "actual_pts_seconds": 0.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "443eed9c5d81ce75cb435670e058f774c05aca7d18912df52a44fb0641b90e4f",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 1.5,
            "requested_time_seconds": 1.5,
            "stream_index": 0,
            "frame_index": 1,
            "actual_pts_seconds": 1.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "cce2cd4f1f1bcc943cf5720f3b33a0ef9492ec084240a7e4934f5fac5751130f",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 2.5,
            "requested_time_seconds": 2.5,
            "stream_index": 0,
            "frame_index": 2,
            "actual_pts_seconds": 2.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "364ec343ad8a6d28bd5821379d5839a82324ea2612ee996eabba223686408354",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 3.5,
            "requested_time_seconds": 3.5,
            "stream_index": 0,
            "frame_index": 3,
            "actual_pts_seconds": 3.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "34a26cb90801475db43e8ddafe64410ed0ba1ff9ccba226a2d5d54ced344e934",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 4.5,
            "requested_time_seconds": 4.5,
            "stream_index": 0,
            "frame_index": 4,
            "actual_pts_seconds": 4.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "b9c59a44897e89b06cd7835ede8ed530125041109464deb27dff25def8a73ca8",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 5.5,
            "requested_time_seconds": 5.5,
            "stream_index": 0,
            "frame_index": 5,
            "actual_pts_seconds": 5.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "c0056c1ccfcc69eeade35dc12a4dc1f66a8c647f777d2e1274109369f5621c25",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 6.5,
            "requested_time_seconds": 6.5,
            "stream_index": 0,
            "frame_index": 6,
            "actual_pts_seconds": 6.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "083d3afbb1382ad789d61d73a4977a2246a6fd00cc52132fb534974ab6a0b922",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 7.5,
            "requested_time_seconds": 7.5,
            "stream_index": 0,
            "frame_index": 7,
            "actual_pts_seconds": 7.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "40101fffa83a2df5a4f667221c8c4d24366dce5984747b080fe9209de4b8b688",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 8.5,
            "requested_time_seconds": 8.5,
            "stream_index": 0,
            "frame_index": 8,
            "actual_pts_seconds": 8.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "f12ac8810729723b446cbcb15ce6902f706cc2f7fd257f29f51cf46235046d4e",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 9.5,
            "requested_time_seconds": 9.5,
            "stream_index": 0,
            "frame_index": 9,
            "actual_pts_seconds": 9.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "bed00c8456c1f812a7ad838f1379043a2a5b8f94d8914034dec2f4775ba151cc",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 10.5,
            "requested_time_seconds": 10.5,
            "stream_index": 0,
            "frame_index": 10,
            "actual_pts_seconds": 10.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "495d311b48769ca83e29bfbe9c675af780e904d8176895763f2474d05fcf0ce8",
        },
        {
            "slot_kind": "global",
            "desired_time_seconds": 11.5,
            "requested_time_seconds": 11.5,
            "stream_index": 0,
            "frame_index": 11,
            "actual_pts_seconds": 11.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
            "frame_plan_sha256": "914460a19b61d2e5af5a1ef6200e50dfb4ca6df2f212792fee400ab001aa7046",
        },
    ]


def test_uniform_plan_binds_to_nearest_decoded_pts_with_earlier_tie_break() -> None:
    plan = _uniform_plan(4.0, _decoded_inventory([0.75, 1.25, 2.5, 3.5]), count=2)

    assert [frame["desired_time_seconds"] for frame in plan] == [1.0, 3.0]
    assert [frame["actual_pts_seconds"] for frame in plan] == [0.75, 2.5]


def test_uniform_plan_recovers_collisions_in_chronological_output_order() -> None:
    plan = _uniform_plan(4.0, _decoded_inventory([0.0, 0.8, 4.0]), count=3)

    assert [frame["actual_pts_seconds"] for frame in plan] == [0.0, 0.8, 4.0]
    assert len({frame["frame_index"] for frame in plan}) == 3


def test_frame_plans_have_a_stable_non_self_referential_binding_hash() -> None:
    plan = _uniform_plan(12.0, _decoded_inventory([index + 0.5 for index in range(12)]))

    assert all(
        frame["frame_plan_sha256"]
        == _canonical_sha256(
            {key: value for key, value in frame.items() if key != "frame_plan_sha256"}
        )
        for frame in plan
    )
    assert all("request_sha256" not in frame for frame in plan)


def test_frame_plan_preserves_stream_identity_when_frame_indexes_overlap() -> None:
    inventory = [
        {
            "stream_index": 0,
            "frame_index": 7,
            "actual_pts_seconds": 0.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        },
        {
            "stream_index": 1,
            "frame_index": 7,
            "actual_pts_seconds": 1.5,
            "time_base": "1/1000",
            "source_media_sha256": MEDIA_SHA,
        },
    ]

    plan = _uniform_plan(2.0, inventory, count=2)

    assert [(frame["stream_index"], frame["frame_index"]) for frame in plan] == [
        (0, 7),
        (1, 7),
    ]
    assert all(
        frame["frame_plan_sha256"]
        == _canonical_sha256(
            {key: value for key, value in frame.items() if key != "frame_plan_sha256"}
        )
        for frame in plan
    )


def test_oracle_hybrid_uses_six_global_and_six_interval_local_slots() -> None:
    inventory = _decoded_inventory(
        [
            0.5,
            1.0,
            1.25,
            1.75,
            2.5,
            3.0,
            4.25,
            4.75,
            5.25,
            5.75,
            7.0,
            9.0,
            11.0,
        ]
    )

    plan = _oracle_plan(12.0, [[[1.0, 2.0], [4.0, 6.0]]], inventory)

    assert [frame["slot_kind"] for frame in plan].count("global") == 6
    assert [frame["slot_kind"] for frame in plan].count("local") == 6
    local = [frame for frame in plan if frame["slot_kind"] == "local"]
    assert [frame["desired_time_seconds"] for frame in local] == [
        1.25,
        1.75,
        4.25,
        4.75,
        5.25,
        5.75,
    ]
    assert all(
        1.0 <= frame["actual_pts_seconds"] <= 2.0
        or 4.0 <= frame["actual_pts_seconds"] <= 6.0
        for frame in local
    )
    assert len({frame["frame_index"] for frame in plan}) == 12
    assert plan == sorted(plan, key=lambda frame: frame["actual_pts_seconds"])


@pytest.mark.parametrize(
    ("sufficient_sets", "selected_range"),
    [
        ([[[2.0, 5.0]], [[8.0, 9.0]]], (8.0, 9.0)),
        ([[[8.0, 9.0]], [[1.0, 2.0]]], (1.0, 2.0)),
    ],
)
def test_oracle_hybrid_selects_shortest_set_then_earliest_onset(
    sufficient_sets: list[list[list[float]]], selected_range: tuple[float, float]
) -> None:
    inventory = _decoded_inventory([index / 8 for index in range(97)])

    plan = _oracle_plan(12.0, sufficient_sets, inventory)

    local_pts = [
        frame["actual_pts_seconds"] for frame in plan if frame["slot_kind"] == "local"
    ]
    assert all(selected_range[0] <= point <= selected_range[1] for point in local_pts)


def test_oracle_hybrid_recovers_from_global_local_frame_collisions() -> None:
    points = [
        0.0,
        0.5,
        0.90,
        0.94,
        0.98,
        1.00,
        1.02,
        1.06,
        1.10,
        1.5,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
    ]

    plan = _oracle_plan(12.0, [[[0.9, 1.1]]], _decoded_inventory(points))

    local = [frame for frame in plan if frame["slot_kind"] == "local"]
    assert len(local) == 6
    assert all(0.9 <= frame["actual_pts_seconds"] <= 1.1 for frame in local)
    assert len({frame["frame_index"] for frame in plan}) == 12


def test_planners_reject_rows_that_cannot_supply_twelve_unique_frames() -> None:
    with pytest.raises(ValueError, match="12 unique"):
        _uniform_plan(12.0, _decoded_inventory([float(index) for index in range(11)]))
    with pytest.raises(ValueError, match="construct.*local"):
        _oracle_plan(
            12.0,
            [[[0.0, 4.0]]],
            _decoded_inventory([float(index) for index in range(12)]),
        )


def _temporal_panel() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": "synthetic-temporal-request-1",
        "examples": [
            {
                "video_id": "alpha",
                "question": "What happened after the door opened?",
                "duration_seconds": 12.0,
                "media_sha256": MEDIA_SHA,
            }
        ],
    }


def _condition_frames() -> dict:
    conditions = {}
    for condition_index, condition in enumerate(("uniform-12", "oracle-hybrid-12")):
        frames = []
        for index in range(12):
            plan_record = {
                "slot_kind": "local" if condition_index and index < 6 else "global",
                "desired_time_seconds": index + 0.25 + condition_index * 0.1,
                "requested_time_seconds": index + 0.25 + condition_index * 0.1,
                "stream_index": 0,
                "frame_index": index + condition_index * 100,
                "actual_pts_seconds": index + 0.25 + condition_index * 0.1,
                "time_base": "1/1000",
                "source_media_sha256": MEDIA_SHA,
            }
            plan_record["frame_plan_sha256"] = _canonical_sha256(plan_record)
            frames.append(
                {
                    **plan_record,
                    "sha256": f"{index + condition_index * 100:064x}",
                    "extraction_command_identity": {
                        "identity": "ffmpeg-extract-v1",
                        "sha256": "b" * 64,
                    },
                    "prompt_source_identity": {
                        "identity": "temporal-question-only-v1",
                        "sha256": "d" * 64,
                    },
                    "runtime_identity": {
                        "identity": "cpu-contract-v1",
                        "sha256": "e" * 64,
                    },
                }
            )
        conditions[condition] = {"alpha": frames}
    return conditions


def _inference_identity() -> dict:
    return {
        "model_id": "synthetic/model",
        "model_revision": "1" * 40,
        "runtime_image": "synthetic@sha256:" + "c" * 64,
        "prompt_source_identity": {
            "identity": "temporal-question-only-v1",
            "sha256": "d" * 64,
        },
        "runtime_identity": {
            "identity": "cpu-contract-v1",
            "sha256": "e" * 64,
        },
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "seed": 0,
        "image_resolution": 512,
    }


def _build_temporal_requests(
    panel_payload: dict, frames: dict, identity: dict
) -> list[dict]:
    builder = getattr(temporal_bottleneck, "build_temporal_requests", None)
    assert callable(builder), "build_temporal_requests must be implemented"
    return builder(panel_payload, frames, identity)


def test_temporal_requests_are_matched_label_free_and_stably_hashed() -> None:
    panel_payload = _temporal_panel()
    frames = _condition_frames()
    identity = _inference_identity()

    requests = _build_temporal_requests(panel_payload, frames, identity)
    repeated = _build_temporal_requests(
        copy.deepcopy(panel_payload), copy.deepcopy(frames), copy.deepcopy(identity)
    )

    assert {request["condition"] for request in requests} == {
        "uniform-12",
        "oracle-hybrid-12",
    }
    assert len(requests) == 2
    assert requests == repeated
    assert all(request["panel_id"] == panel_payload["panel_id"] for request in requests)
    assert all(
        request["panel_sha256"] == _canonical_sha256(panel_payload)
        for request in requests
    )
    assert all(request["inference_identity"] == identity for request in requests)
    assert all(request["frame_count"] == 12 for request in requests)
    assert all(len(set(request["frame_sha256s"])) == 12 for request in requests)
    assert all(len(set(request["frame_timestamps"])) == 12 for request in requests)
    assert all(
        all(
            set(frame)
            == {
                "frame_index",
                "stream_index",
                "actual_pts_seconds",
                "time_base",
                "source_media_sha256",
                "sha256",
                "requested_time_seconds",
                "extraction_command_identity",
                "prompt_source_identity",
                "frame_plan_sha256",
                "runtime_identity",
            }
            for frame in request["frames"]
        )
        for request in requests
    )
    assert all(
        all(
            frame["prompt_source_identity"] == identity["prompt_source_identity"]
            and frame["runtime_identity"] == identity["runtime_identity"]
            for frame in request["frames"]
        )
        for request in requests
    )
    assert all(request["transcript"] is None for request in requests)
    assert all(request["media_path"] is None for request in requests)
    assert all(request["audio_path"] is None for request in requests)
    assert all(request["audio_mode"] is None for request in requests)
    assert len({request["request_sha256"] for request in requests}) == 2
    assert all(
        request["request_sha256"]
        == _canonical_sha256(
            {key: value for key, value in request.items() if key != "request_sha256"}
        )
        for request in requests
    )
    serialized = json.dumps(requests)
    for forbidden in (
        "answer",
        "reference",
        "sufficient_sets",
        "preferred_set_index",
        "slot_kind",
        "desired_time_seconds",
    ):
        assert forbidden not in serialized


def test_temporal_requests_allow_same_frame_index_on_distinct_streams() -> None:
    frames = _condition_frames()
    for condition_frames in frames.values():
        records = condition_frames["alpha"]
        records[1]["frame_index"] = records[0]["frame_index"]
        records[1]["stream_index"] = 1
        plan_record = {
            key: records[1][key]
            for key in (
                "slot_kind",
                "desired_time_seconds",
                "requested_time_seconds",
                "stream_index",
                "frame_index",
                "actual_pts_seconds",
                "time_base",
                "source_media_sha256",
            )
        }
        records[1]["frame_plan_sha256"] = _canonical_sha256(plan_record)

    requests = _build_temporal_requests(
        _temporal_panel(), frames, _inference_identity()
    )

    for request in requests:
        first, second = request["frames"][:2]
        assert [first["stream_index"], second["stream_index"]] == [0, 1]
        assert first["frame_index"] == second["frame_index"]


def test_temporal_requests_reject_stream_identity_tampering() -> None:
    frames = _condition_frames()
    frames["uniform-12"]["alpha"][0]["stream_index"] = 1

    with pytest.raises(ValueError, match="frame_plan_sha256"):
        _build_temporal_requests(_temporal_panel(), frames, _inference_identity())


@pytest.mark.parametrize(
    "missing_field",
    [
        "stream_index",
        "requested_time_seconds",
        "extraction_command_identity",
        "prompt_source_identity",
        "frame_plan_sha256",
        "runtime_identity",
    ],
)
def test_temporal_requests_require_complete_frame_provenance(
    missing_field: str,
) -> None:
    frames = _condition_frames()
    frames["uniform-12"]["alpha"][0].pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        _build_temporal_requests(_temporal_panel(), frames, _inference_identity())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("requested_time_seconds", 11.875),
        (
            "prompt_source_identity",
            {"identity": "other-prompt", "sha256": "9" * 64},
        ),
        (
            "runtime_identity",
            {"identity": "other-runtime", "sha256": "8" * 64},
        ),
    ],
)
def test_temporal_requests_reject_plan_tampering_or_inference_identity_drift(
    field: str, replacement: object
) -> None:
    frames = _condition_frames()
    frames["uniform-12"]["alpha"][0][field] = replacement

    with pytest.raises(ValueError, match="frame_plan_sha256|identity"):
        _build_temporal_requests(_temporal_panel(), frames, _inference_identity())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda identity: identity.clear(), "inference_identity"),
        (lambda identity: identity.pop("model_revision"), "model_revision"),
        (
            lambda identity: identity.update(runtime_image="synthetic:latest"),
            "runtime_image",
        ),
        (
            lambda identity: identity.update(prompt_source_identity={}),
            "prompt_source_identity",
        ),
        (
            lambda identity: identity.update(runtime_identity="cpu-contract-v1"),
            "runtime_identity",
        ),
    ],
)
def test_temporal_requests_require_pinned_material_inference_identity(
    mutation: object, message: str
) -> None:
    identity = _inference_identity()
    mutation(identity)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        _build_temporal_requests(_temporal_panel(), _condition_frames(), identity)


def test_temporal_requests_reject_condition_drift_and_answer_material() -> None:
    panel_payload = _temporal_panel()
    frames = _condition_frames()
    frames["directed-12"] = frames.pop("oracle-hybrid-12")

    with pytest.raises(ValueError, match="condition set"):
        _build_temporal_requests(panel_payload, frames, _inference_identity())

    frames = _condition_frames()
    panel_payload["examples"][0]["reference_answer"] = "leaked"
    with pytest.raises(ValueError, match="reference_answer"):
        _build_temporal_requests(panel_payload, frames, _inference_identity())


def _freeze_json_artifact(path: object, bindings: dict) -> dict:
    freezer = getattr(temporal_bottleneck, "freeze_json_artifact", None)
    assert callable(freezer), "freeze_json_artifact must be implemented"
    return freezer(path, bindings=bindings)


def _verify_json_artifact(path: object, freeze_record: dict) -> None:
    verifier = getattr(temporal_bottleneck, "verify_json_artifact", None)
    assert callable(verifier), "verify_json_artifact must be implemented"
    verifier(path, freeze_record)


def test_json_artifact_freeze_is_read_only_and_verifies_bindings(tmp_path) -> None:
    artifact = tmp_path / "requests.json"
    payload = {
        "panel_sha256": "d" * 64,
        "requests_sha256": "e" * 64,
        "requests": [{"request_sha256": "f" * 64}],
    }
    artifact.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = artifact.read_bytes()
    bindings = {
        "panel_sha256": "d" * 64,
        "requests_sha256": "e" * 64,
    }

    freeze_record = _freeze_json_artifact(artifact, bindings)

    assert artifact.read_bytes() == before
    assert freeze_record == {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "artifact_sha256": hashlib.sha256(before).hexdigest(),
        "bindings": bindings,
        "bindings_sha256": _canonical_sha256(bindings),
    }
    _verify_json_artifact(artifact, freeze_record)


def test_json_artifact_freeze_rejects_binding_mismatch_and_tampering(tmp_path) -> None:
    artifact = tmp_path / "requests.json"
    artifact.write_text(
        json.dumps({"panel_sha256": "d" * 64, "requests": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="panel_sha256.*binding"):
        _freeze_json_artifact(artifact, {"panel_sha256": "e" * 64})

    freeze_record = _freeze_json_artifact(artifact, {"panel_sha256": "d" * 64})
    artifact.write_text(
        json.dumps({"panel_sha256": "d" * 64, "requests": ["tampered"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact_sha256"):
        _verify_json_artifact(artifact, freeze_record)

    artifact.write_text(
        json.dumps({"panel_sha256": "d" * 64, "requests": []}),
        encoding="utf-8",
    )
    freeze_record["bindings"]["panel_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="bindings_sha256"):
        _verify_json_artifact(artifact, freeze_record)


@pytest.fixture
def panel() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": "synthetic-temporal-2",
        "examples": [
            {"video_id": "alpha", "duration_seconds": 10.0},
            {"video_id": "beta", "duration_seconds": 4.0},
        ],
    }


@pytest.fixture
def valid_annotation(panel: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer": "reviewer-a",
        "role": ROLE_PRIMARY_REVIEWER,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "started_at": "2026-07-30T09:00:00Z",
        "ended_at": "2026-07-30T09:30:00Z",
        "rows": [
            {
                "video_id": "alpha",
                "eligibility": ELIGIBILITY_LABELABLE_VISUAL,
                "sufficient_sets": [[[1.0, 2.0], [4.0, 5.0]], [[7.0, 8.0]]],
                "preferred_set_index": 0,
            },
            {
                "video_id": "beta",
                "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE,
            },
        ],
    }


@pytest.mark.parametrize("missing_field", ["started_at", "ended_at"])
def test_validator_requires_both_utc_annotation_timestamps(
    panel: dict, valid_annotation: dict, missing_field: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("started_at", "not-a-timestamp", "started_at"),
        ("started_at", "2026-07-30T09:00:00+00:00", "UTC.*Z"),
        ("started_at", "2026-07-30T09:00:00-04:00", "UTC.*Z"),
        ("ended_at", "2026-07-30T08:59:59Z", "after started_at"),
        ("ended_at", "2026-07-30T09:00:00Z", "after started_at"),
    ],
)
def test_validator_rejects_noncanonical_or_nonincreasing_annotation_timestamps(
    panel: dict,
    valid_annotation: dict,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_primary_annotation_rejects_adjudicator_input_hashes(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["input_annotation_sha256s"] = ["1" * 64, "2" * 64, "3" * 64]

    with pytest.raises(ValueError, match="input_annotation_sha256s"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("input_hashes", "message"),
    [
        (None, "input_annotation_sha256s"),
        (["1" * 64, "2" * 64], "exactly three"),
        (["1" * 64, "1" * 64, "3" * 64], "distinct"),
        (["1" * 64, "2" * 64, "A" * 64], "lowercase SHA-256"),
    ],
)
def test_adjudicator_requires_three_distinct_ordered_primary_annotation_hashes(
    panel: dict,
    valid_annotation: dict,
    input_hashes: list[str] | None,
    message: str,
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["role"] = ROLE_ADJUDICATOR
    if input_hashes is not None:
        payload["input_annotation_sha256s"] = input_hashes

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_ADJUDICATOR)


def test_adjudicator_hash_order_is_preserved_across_idempotent_revalidation(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["role"] = ROLE_ADJUDICATOR
    payload["input_annotation_sha256s"] = ["3" * 64, "1" * 64, "2" * 64]

    normalized = validate_annotation(payload, panel, expected_role=ROLE_ADJUDICATOR)
    revalidated = validate_annotation(normalized, panel, expected_role=ROLE_ADJUDICATOR)

    assert revalidated["input_annotation_sha256s"] == ["3" * 64, "1" * 64, "2" * 64]
    assert revalidated["annotation_sha256"] == normalized["annotation_sha256"]


def test_validator_normalizes_duration_bounded_closed_intervals_in_panel_order(
    panel: dict, valid_annotation: dict
) -> None:
    annotation = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    assert [row["video_id"] for row in annotation["rows"]] == ["alpha", "beta"]
    assert annotation["rows"][0]["sufficient_sets"] == [
        [[1.0, 2.0], [4.0, 5.0]],
        [[7.0, 8.0]],
    ]
    assert annotation["reviewer"] == "reviewer-a"
    assert annotation["rubric_revision"] == "temporal-required-moment-v1"
    assert annotation["started_at"] == "2026-07-30T09:00:00Z"
    assert annotation["ended_at"] == "2026-07-30T09:30:00Z"
    assert annotation["annotation_sha256"] == _canonical_sha256(valid_annotation)


def test_validator_revalidates_unchanged_normalized_annotation_idempotently(
    panel: dict, valid_annotation: dict
) -> None:
    normalized = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    revalidated = validate_annotation(
        normalized, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    assert revalidated["annotation_sha256"] == normalized["annotation_sha256"]
    assert revalidated["rows"] == normalized["rows"]


def test_validator_rejects_normalized_duration_mutated_after_panel_binding(
    panel: dict, valid_annotation: dict
) -> None:
    normalized = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )
    normalized["rows"][0]["duration_seconds"] = 100.0

    with pytest.raises(ValueError, match="duration_seconds.*frozen panel"):
        validate_annotation(normalized, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_that_omits_a_frozen_panel_row(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"] = payload["rows"][:1]

    with pytest.raises(ValueError, match="panel rows"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_that_reorders_frozen_panel_ids(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"].reverse()

    with pytest.raises(ValueError, match="frozen panel order"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_with_duplicate_video_ids(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["video_id"] = "alpha"

    with pytest.raises(ValueError, match="duplicate"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_labelable_row_with_an_empty_sufficient_set(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[]]

    with pytest.raises(ValueError, match="empty"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_sufficient_set_with_more_than_six_intervals(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [
        [
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
            [4.0, 4.1],
            [5.0, 5.1],
            [6.0, 6.1],
        ]
    ]
    payload["rows"][0]["preferred_set_index"] = 0

    with pytest.raises(ValueError, match="at most six"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("interval", "message"),
    [([-0.1, 1.0], "nonnegative"), ([3.0, 3.0], "end"), ([3.0, 2.0], "end")],
)
def test_validator_rejects_negative_or_reversed_interval_bounds(
    panel: dict, valid_annotation: dict, interval: list[float], message: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[interval]]

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_interval_ending_beyond_frozen_duration(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[[9.9, 10.1]]]

    with pytest.raises(ValueError, match="duration"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_unknown_eligibility_value(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["eligibility"] = "probably_labelable"

    with pytest.raises(ValueError, match="eligibility"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_intervals_on_an_ineligible_row(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["sufficient_sets"] = [[[0.0, 1.0]]]

    with pytest.raises(ValueError, match="ineligible"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("packet_sha256", "0" * 64, "packet_sha256"),
        ("role", "adjudicator", "role"),
    ],
)
def test_validator_rejects_mismatched_packet_hash_or_role(
    panel: dict, valid_annotation: dict, field: str, value: str, message: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    "forbidden_key",
    ["answer", "reference", "prediction", "grade", "condition", "frame_timestamps"],
)
def test_validator_rejects_forbidden_label_or_condition_key_at_any_depth(
    panel: dict, valid_annotation: dict, forbidden_key: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"][0][0].append({forbidden_key: "secret"})

    with pytest.raises(ValueError, match=forbidden_key):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    "forbidden_key",
    ["prior_prediction", "previous_grade", "newPrediction", "selectedFrameTimestamps"],
)
def test_validator_rejects_compound_blinding_keys_but_allows_provenance_names(
    panel: dict, valid_annotation: dict, forbidden_key: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["provenance"] = {
        "requested_time_seconds": 1.0,
        "extraction_command_identity": "ffmpeg-extract-v1",
        "prompt_source_identity": "temporal-panel-prompt-v1",
        "request_sha256": "b" * 64,
        "runtime_identity": "cpu-contract-v1",
        "answerable_media_count": 1,
        forbidden_key: "secret",
    }

    with pytest.raises(ValueError, match=forbidden_key):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def _primary_annotation(panel: dict, reviewer: str, rows: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer": reviewer,
        "role": ROLE_PRIMARY_REVIEWER,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "started_at": "2026-07-30T09:00:00Z",
        "ended_at": "2026-07-30T09:30:00Z",
        "rows": rows,
    }


def _labelable(video_id: str, sufficient_sets: list[list[list[float]]]) -> dict:
    return {
        "video_id": video_id,
        "eligibility": ELIGIBILITY_LABELABLE_VISUAL,
        "sufficient_sets": sufficient_sets,
        "preferred_set_index": 0,
    }


def test_agreement_reports_identical_preferred_span_as_perfect_temporal_iou(
    panel: dict,
) -> None:
    rows = [
        _labelable("alpha", [[[1.0, 3.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", rows),
        _primary_annotation(panel, "reviewer-b", copy.deepcopy(rows)),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == 1.0
    assert result["rows"][0]["flags"] == []


def test_agreement_reports_disjoint_preferred_spans_as_zero_temporal_iou(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[3.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == 0.0
    assert "temporal_iou_below_threshold" in result["rows"][0]["flags"]


def test_agreement_uses_union_without_double_counting_for_partial_overlap(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 5.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[3.0, 7.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == pytest.approx(2 / 6)


def test_agreement_flags_preferred_set_topology_mismatch(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0], [3.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[1.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["structural_match"] is False
    assert "topology_mismatch" in result["rows"][0]["flags"]


def test_agreement_flags_eligibility_mismatch(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        {"video_id": "alpha", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["structural_match"] is False
    assert "eligibility_mismatch" in result["rows"][0]["flags"]


def test_agreement_flags_matched_boundary_drift_above_one_second(panel: dict) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[2.1, 3.1]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["maximum_boundary_drift_seconds"] == pytest.approx(1.1)
    assert "boundary_drift_above_one_second" in result["rows"][0]["flags"]


def _ten_row_panel() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": "synthetic-temporal-10",
        "examples": [
            {"video_id": f"video-{index}", "duration_seconds": 10.0}
            for index in range(10)
        ],
    }


def test_agreement_accepts_exactly_eight_of_ten_structural_matches() -> None:
    agreement_panel = _ten_row_panel()
    left = [_labelable(f"video-{index}", [[[1.0, 2.0]]]) for index in range(10)]
    right = copy.deepcopy(left)
    right[8]["sufficient_sets"] = [[[1.0, 2.0], [3.0, 4.0]]]
    right[9]["sufficient_sets"] = [[[1.0, 2.0], [3.0, 4.0]]]

    result = annotation_agreement(
        _primary_annotation(agreement_panel, "reviewer-a", left),
        _primary_annotation(agreement_panel, "reviewer-b", right),
        agreement_panel,
    )

    assert result["structural_match_count"] == 8
    assert result["structural_threshold_passes"] is True
    assert result["agreement_passes"] is True


def test_agreement_fails_when_median_temporal_iou_is_below_half() -> None:
    agreement_panel = _ten_row_panel()
    left = [_labelable(f"video-{index}", [[[1.0, 2.0]]]) for index in range(10)]
    right = copy.deepcopy(left)
    for index in range(6):
        right[index]["sufficient_sets"] = [[[3.0, 4.0]]]

    result = annotation_agreement(
        _primary_annotation(agreement_panel, "reviewer-a", left),
        _primary_annotation(agreement_panel, "reviewer-b", right),
        agreement_panel,
    )

    assert result["median_temporal_iou"] == 0.0
    assert result["median_temporal_iou_passes"] is False
    assert result["agreement_passes"] is False


def _adjudicated_annotation(panel: dict, rows: list[dict]) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "reviewer": "adjudicator-c",
        "role": ROLE_ADJUDICATOR,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "started_at": "2026-07-30T10:00:00Z",
        "ended_at": "2026-07-30T10:30:00Z",
        "input_annotation_sha256s": ["1" * 64, "2" * 64, "3" * 64],
        "rows": rows,
    }
    return validate_annotation(payload, panel, expected_role=ROLE_ADJUDICATOR)


def _frame(actual_pts_seconds: float, frame_index: int) -> dict:
    return {
        "actual_pts_seconds": actual_pts_seconds,
        "stream_index": 0,
        "frame_index": frame_index,
        "time_base": "1/1000",
        "sha256": f"{frame_index:064x}",
        "requested_time_seconds": actual_pts_seconds,
        "extraction_command_identity": "ffmpeg-extract-v1",
        "prompt_source_identity": "temporal-panel-prompt-v1",
        "request_sha256": "b" * 64,
        "runtime_identity": "cpu-contract-v1",
    }


def test_selector_recall_distinguishes_strict_from_half_second_boundary_hits(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    frames = {"alpha": [_frame(0.5, 1)]}

    strict = score_selector_recall(annotation, frames, tolerance_seconds=0.0)
    primary = score_selector_recall(annotation, frames, tolerance_seconds=0.5)

    assert strict["row_recall"] == 0.0
    assert strict["missed_video_ids"] == ["alpha"]
    assert primary["row_recall"] == 1.0
    assert primary["missed_video_ids"] == []


def test_selector_recall_requires_every_conjunctive_interval_to_hit(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1], [4.0, 4.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["component_recall"] == 0.5
    assert result["row_recall"] == 0.0
    assert result["missed_video_ids"] == ["alpha"]


def test_selector_recall_distinguishes_overlapping_indexes_by_stream(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1], [4.0, 4.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    first = _frame(1.0, 7)
    second = _frame(4.0, 7)
    second["stream_index"] = 1
    second["sha256"] = "f" * 64

    result = score_selector_recall(
        annotation, {"alpha": [first, second]}, tolerance_seconds=0.0
    )

    assert result["row_recall"] == 1.0


def test_selector_recall_accepts_any_complete_alternative_sufficient_set(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]], [[5.0, 5.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(5.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["component_recall"] == 0.5
    assert result["row_recall"] == 1.0
    assert result["rows"][0]["recalled_sufficient_set_indexes"] == [1]


def test_selector_recall_excludes_ineligible_rows_from_recall_denominators(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["eligible_row_count"] == 1
    assert result["component_count"] == 1
    assert result["row_recall"] == 1.0
    assert result["rows"][1]["included_in_recall"] is False


def test_selector_recall_rejects_requested_timestamp_substituted_for_actual_pts(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    requested_only = {
        "alpha": [
            {
                "requested_time_seconds": 1.0,
                "frame_index": 1,
                "time_base": "1/1000",
                "sha256": "a" * 64,
            }
        ]
    }

    with pytest.raises(ValueError, match="actual_pts_seconds"):
        score_selector_recall(annotation, requested_only, tolerance_seconds=0.0)


def test_selector_recall_rejects_annotation_mutated_after_hash_validation(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    annotation["rows"][0]["sufficient_sets"] = [[[5.0, 5.1]]]

    with pytest.raises(ValueError, match="annotation_sha256"):
        score_selector_recall(
            annotation, {"alpha": [_frame(5.0, 1)]}, tolerance_seconds=0.0
        )


def test_selector_recall_rejects_normalized_duration_mutated_after_validation(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    annotation["rows"][0]["duration_seconds"] = 100.0

    with pytest.raises(ValueError, match="normalized_annotation_sha256"):
        score_selector_recall(
            annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "stream_index",
        "requested_time_seconds",
        "extraction_command_identity",
        "prompt_source_identity",
        "request_sha256",
        "runtime_identity",
    ],
)
def test_selector_recall_requires_complete_preinference_frame_provenance(
    panel: dict, missing_field: str
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    incomplete_frame = _frame(1.0, 1)
    incomplete_frame.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        score_selector_recall(
            annotation, {"alpha": [incomplete_frame]}, tolerance_seconds=0.0
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("requested_time_seconds", float("nan")),
        ("extraction_command_identity", ""),
        ("prompt_source_identity", ""),
        ("request_sha256", "not-a-sha256"),
        ("runtime_identity", {}),
    ],
)
def test_selector_recall_rejects_empty_or_invalid_frame_provenance(
    panel: dict, field: str, invalid_value: object
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    invalid_frame = _frame(1.0, 1)
    invalid_frame[field] = invalid_value

    with pytest.raises((TypeError, ValueError), match=field):
        score_selector_recall(
            annotation, {"alpha": [invalid_frame]}, tolerance_seconds=0.0
        )
