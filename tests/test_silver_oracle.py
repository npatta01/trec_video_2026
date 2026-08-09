from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_video_2026.vqa_experiments.silver_oracle import (
    EVIDENCE_CELLS,
    assert_global_plan_reuse,
    build_frame_plans,
    build_request_matrix,
    build_transcript_views,
    audit_label_free_request,
    held_fixed_identity_sha256,
    model_input_sha256,
    parse_timestamped_whisper,
    protocol_sha256,
    selector_interval_budget,
    validate_selector_result,
)


def selector_row(
    *, duration: float = 30.0, segment_count: int = 4
) -> dict[str, object]:
    return {
        "video_id": "video-01",
        "duration_seconds": duration,
        "whisper_segment_count": segment_count,
    }


def localized_result() -> dict[str, object]:
    return {
        "status": "localized",
        "intervals": [{"start_seconds": 3.0, "end_seconds": 8.0}],
        "whisper_segment_ids": [1, 2],
    }


def frame_inventory(
    *, duration: float = 30.0, count: int = 121, media_sha: str = "a" * 64
) -> list[dict[str, object]]:
    return [
        {
            "stream_index": 0,
            "frame_index": index,
            "actual_pts_seconds": round(duration * index / (count - 1), 6),
            "time_base": "1/1000",
            "source_media_sha256": media_sha,
        }
        for index in range(count)
    ]


def test_validate_selector_result_accepts_exact_localized_schema() -> None:
    result = validate_selector_result(localized_result(), selector_row())

    assert result == localized_result()


@pytest.mark.parametrize(
    ("duration", "expected_budget"),
    [(5.0, 5.0), (12.0, 12.0), (30.0, 12.0)],
)
def test_selector_interval_budget_formula_edges(
    duration: float, expected_budget: float
) -> None:
    assert selector_interval_budget(duration) == expected_budget


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"rationale": "the answer appears here"}, "exactly"),
        ({"answer": "forbidden"}, "exactly"),
        ({"confidence": 0.99}, "exactly"),
    ],
)
def test_validate_selector_result_rejects_generated_text_and_extra_keys(
    mutation: dict[str, object], message: str
) -> None:
    payload = {**localized_result(), **mutation}

    with pytest.raises(ValueError, match=message):
        validate_selector_result(payload, selector_row())


def test_validate_selector_result_rejects_interval_over_budget() -> None:
    payload = localized_result()
    payload["intervals"] = [{"start_seconds": 0.0, "end_seconds": 12.001}]

    with pytest.raises(ValueError, match="duration budget"):
        validate_selector_result(payload, selector_row(duration=30.0))


def test_validate_selector_result_rejects_overlapping_intervals() -> None:
    payload = localized_result()
    payload["intervals"] = [
        {"start_seconds": 2.0, "end_seconds": 7.0},
        {"start_seconds": 6.5, "end_seconds": 9.0},
    ]

    with pytest.raises(ValueError, match="overlap"):
        validate_selector_result(payload, selector_row())


def test_validate_selector_result_rejects_four_or_invalid_segment_ids() -> None:
    payload = localized_result()
    payload["whisper_segment_ids"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="at most three"):
        validate_selector_result(payload, selector_row())

    payload["whisper_segment_ids"] = [0, 4]
    with pytest.raises(ValueError, match="outside"):
        validate_selector_result(payload, selector_row(segment_count=4))


def test_validate_selector_result_requires_empty_non_localizable_payload() -> None:
    valid = {
        "status": "non_localizable",
        "intervals": [],
        "whisper_segment_ids": [],
    }
    assert validate_selector_result(valid, selector_row()) == valid

    invalid = {**valid, "whisper_segment_ids": [0]}
    with pytest.raises(ValueError, match="empty"):
        validate_selector_result(invalid, selector_row())


def test_parse_timestamped_whisper_preserves_verbatim_lines_and_bounds() -> None:
    text = (
        "[00:00:00.000] First line.\n"
        "[00:00:04.080] Second line.\n"
        "[00:00:09.960] Final line."
    )

    segments = parse_timestamped_whisper(text, duration_seconds=12.0)

    assert [segment.segment_id for segment in segments] == [0, 1, 2]
    assert [segment.start_seconds for segment in segments] == [0.0, 4.08, 9.96]
    assert [segment.end_seconds for segment in segments] == [4.08, 9.96, 12.0]
    assert segments[1].source_line == "[00:00:04.080] Second line."
    assert segments[1].text == "Second line."


def test_parse_timestamped_whisper_rejects_non_monotonic_or_overrun_timestamps() -> (
    None
):
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_timestamped_whisper(
            "[00:00:04.000] Later\n[00:00:04.000] Duplicate",
            duration_seconds=10.0,
        )


def test_parse_timestamped_whisper_can_drop_only_trailing_overrun_lines() -> None:
    segments = parse_timestamped_whisper(
        "[00:00:00.000] Valid\n[00:00:09.000] Valid too\n[00:00:11.000] Overrun",
        duration_seconds=10.0,
        drop_trailing_overrun=True,
    )

    assert [segment.segment_id for segment in segments] == [0, 1]
    assert [segment.end_seconds for segment in segments] == [9.0, 10.0]

    with pytest.raises(ValueError, match="duration"):
        parse_timestamped_whisper(
            "[00:00:11.000] Outside",
            duration_seconds=10.0,
        )


def test_build_frame_plans_constructs_exact_global_hybrid_and_local_views() -> None:
    plans = build_frame_plans(
        selector_row(),
        localized_result(),
        frame_inventory(),
    )

    assert set(plans) == {"G", "H", "L", "localized", "fallback_reason"}
    assert plans["localized"] is True
    assert plans["fallback_reason"] is None
    assert len(plans["G"]) == len(plans["H"]) == len(plans["L"]) == 12
    assert sum(frame["slot_kind"] == "local" for frame in plans["H"]) == 6
    assert sum(frame["slot_kind"] == "global" for frame in plans["H"]) == 6
    assert all(frame["slot_kind"] == "local" for frame in plans["L"])
    for view in ("G", "H", "L"):
        timestamps = [frame["actual_pts_seconds"] for frame in plans[view]]
        assert timestamps == sorted(timestamps)
        assert len(timestamps) == len(set(timestamps)) == 12


def test_build_frame_plans_falls_back_to_global_for_non_localizable_rows() -> None:
    selector = {
        "status": "non_localizable",
        "intervals": [],
        "whisper_segment_ids": [],
    }

    plans = build_frame_plans(selector_row(), selector, frame_inventory())

    assert plans["localized"] is False
    assert plans["fallback_reason"] == "non_localizable"
    assert plans["G"] == plans["H"] == plans["L"]


def test_protocol_sha256_uses_canonical_utf8_json() -> None:
    protocol = {"z": ["é", 2], "a": {"enabled": True}}
    expected = hashlib.sha256(
        json.dumps(
            protocol,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert protocol_sha256(protocol) == expected


def rich_frame_inventory(
    timestamps: list[float], *, media_sha: str = "a" * 64
) -> list[dict[str, object]]:
    return [
        {
            "stream_index": 0,
            "frame_index": index,
            "actual_pts_seconds": timestamp,
            "time_base": "1/1000",
            "source_media_sha256": media_sha,
            "frame_path": f"frame-{index:03d}.png",
            "image_sha256": hashlib.sha256(
                f"image-{index}".encode("utf-8")
            ).hexdigest(),
        }
        for index, timestamp in enumerate(timestamps)
    ]


def test_build_frame_plans_records_global_reuse_and_collision_fallback() -> None:
    inventory = rich_frame_inventory(
        [
            0.0,
            1.0,
            2.0,
            3.5,
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
            12.5,
            13.5,
            14.5,
            15.5,
            16.5,
            17.5,
            18.5,
            19.5,
        ]
    )
    plans = build_frame_plans(
        {"duration_seconds": 20.0, "whisper_segment_count": 0},
        {
            "status": "localized",
            "intervals": [{"start_seconds": 3.4, "end_seconds": 3.6}],
            "whisper_segment_ids": [],
        },
        inventory,
    )

    assert_global_plan_reuse(plans["G"], plans["G"])
    with pytest.raises(ValueError, match="global plan"):
        altered = [dict(frame) for frame in plans["G"]]
        altered[0]["image_sha256"] = "b" * 64
        assert_global_plan_reuse(altered, plans["G"])

    assert sum(frame["slot_kind"] == "local" for frame in plans["H"]) == 6
    assert sum(frame["slot_kind"] == "global" for frame in plans["H"]) == 6
    assert all(frame["slot_kind"] == "local" for frame in plans["L"])
    assert any(
        frame.get("construction_fallback") == "global_frame"
        for view in ("H", "L")
        for frame in plans[view]
    )
    for view in ("G", "H", "L"):
        pts = [frame["actual_pts_seconds"] for frame in plans[view]]
        assert pts == sorted(pts)
        assert len(pts) == len(set(pts)) == 12


def test_build_transcript_views_preserves_full_and_selected_verbatim_identity() -> None:
    transcript = "[00:00:00.000]  First line.\n[00:00:04.080] Second line.\n"

    views = build_transcript_views(
        transcript,
        [1],
        duration_seconds=12.0,
    )

    assert views["N"]["text"] is None
    assert views["N"]["selected_line_indices"] == []
    assert views["F"]["text"] == transcript
    assert views["F"]["selected_line_indices"] == [0, 1]
    assert views["S"]["text"] == "[00:00:04.080] Second line."
    assert views["S"]["selected_line_indices"] == [1]
    assert (
        views["F"]["source_sha256"]
        == hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    )
    assert views["S"]["source_sha256"] == views["F"]["source_sha256"]


def test_request_matrix_has_exact_nine_cells_and_nfs_semantics() -> None:
    inventory = rich_frame_inventory([float(index) for index in range(20)])
    plans = build_frame_plans(
        {"duration_seconds": 20.0, "whisper_segment_count": 4},
        localized_result(),
        inventory,
    )
    views = build_transcript_views(
        "[00:00:00.000] Hello.\n[00:00:04.000] Goodbye.",
        [1],
        duration_seconds=20.0,
    )

    matrix = build_request_matrix(
        row={
            "video_id": "video-01",
            "question": "What happens?",
        },
        frame_plans=plans,
        transcript_views=views,
        run_id="run-1",
    )

    assert set(matrix) == set(EVIDENCE_CELLS)
    assert matrix["gn"]["transcript"] is None
    assert matrix["gf"]["transcript"] == views["F"]["text"]
    assert matrix["gs"]["transcript"] == views["S"]["text"]
    assert matrix["gn"]["frame_paths"] == matrix["gf"]["frame_paths"]
    assert matrix["gn"]["frame_timestamps"] == matrix["gf"]["frame_timestamps"]
    assert model_input_sha256(matrix["gn"]) != model_input_sha256(matrix["gf"])
    assert (
        len({held_fixed_identity_sha256(request) for request in matrix.values()}) == 1
    )
    expected_fields = {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "transcript",
    }
    for cell, request in matrix.items():
        assert set(request) == expected_fields
        assert request["condition_id"] == cell
        audit_label_free_request(request)


def test_model_input_digest_binds_frame_bytes_not_materialization_root(
    tmp_path: Path,
) -> None:
    first_paths = []
    second_paths = []
    for index in range(12):
        payload = f"frame-{index}".encode("utf-8")
        first = tmp_path / "first" / f"{index}.png"
        second = tmp_path / "second" / f"{index}.png"
        first.parent.mkdir(exist_ok=True)
        second.parent.mkdir(exist_ok=True)
        first.write_bytes(payload)
        second.write_bytes(payload)
        first_paths.append(str(first))
        second_paths.append(str(second))
    request = {
        "run_id": "baseline",
        "condition_id": "gf",
        "video_id": "video-01",
        "question": "What happens?",
        "frame_paths": first_paths,
        "frame_timestamps": [float(index) for index in range(12)],
        "transcript": "verbatim",
    }
    relocated = {**request, "frame_paths": second_paths}

    assert model_input_sha256(request) == model_input_sha256(relocated)
    Path(second_paths[0]).write_bytes(b"different-frame-bytes")
    assert model_input_sha256(request) != model_input_sha256(relocated)


def test_full_interval_h_allows_requested_timestamp_collisions() -> None:
    inventory = rich_frame_inventory(
        [round(10.0 * index / 120, 6) for index in range(121)]
    )
    plans = build_frame_plans(
        {"duration_seconds": 10.0, "whisper_segment_count": 0},
        {
            "status": "localized",
            "intervals": [{"start_seconds": 0.0, "end_seconds": 10.0}],
            "whisper_segment_ids": [],
        },
        inventory,
    )

    matrix = build_request_matrix(
        row={"video_id": "video-01", "question": "What happens?"},
        frame_plans=plans,
        transcript_views=build_transcript_views(
            None,
            [],
            duration_seconds=10.0,
        ),
        run_id="run-1",
    )

    hybrid = plans["H"]
    assert len(set(matrix["hn"]["frame_timestamps"])) == 6
    assert len({frame["frame_path"] for frame in hybrid}) == 12
    assert (
        len({(frame["stream_index"], frame["frame_index"]) for frame in hybrid}) == 12
    )
    assert len({frame["actual_pts_seconds"] for frame in hybrid}) == 12


def test_recursive_reference_leakage_is_rejected_from_requests() -> None:
    with pytest.raises(ValueError, match="reference_answer"):
        audit_label_free_request(
            {
                "question": "What?",
                "nested": [{"metadata": {"reference_answer": "hidden"}}],
            }
        )


def load_evidence_builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_vqa_silver_evidence.py"
    spec = importlib.util.spec_from_file_location("vqa_silver_evidence", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load VQA silver evidence builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task5_runner_tests():
    path = (
        Path(__file__).resolve().parents[1]
        / "tests/test_vqa_silver_answer_dstack_config.py"
    )
    spec = importlib.util.spec_from_file_location("vqa_silver_task5_tests", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load VQA silver Task 5 runner tests")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _id_digest(video_ids: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def _materialized_plans(
    plans: dict[str, object], *, output: Path, video_id: str
) -> dict[str, object]:
    result = dict(plans)
    for view in ("G", "H", "L"):
        result[view] = [
            {
                **frame,
                "frame_path": str(
                    output
                    / "frames"
                    / video_id
                    / view
                    / f"frame-{frame['frame_index']:03d}.png"
                ),
            }
            for frame in plans[view]
        ]
    return result


def _fake_frame_extractor(
    row: dict[str, object],
    frame: dict[str, object],
    view: str,
    output_path: Path,
) -> dict[str, object]:
    del row, view
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_path.as_posix().encode("utf-8"))
    return {
        "frame_path": str(output_path),
        "stream_index": frame["stream_index"],
        "frame_index": frame["frame_index"],
        "decoded_frame_ordinal": frame["frame_index"],
        "time_base": frame["time_base"],
        "raw_pts": None,
        "actual_pts_seconds": frame["actual_pts_seconds"],
        "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def _fake_contact_sheet(
    *, video_id: str, view: str, frame_paths: list[str], output_path: Path
) -> dict[str, object]:
    del video_id, view, frame_paths
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"contact-sheet")
    return {
        "path": str(output_path),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def _planned_materialization_frame(media_sha256: str) -> dict[str, object]:
    return {
        "slot_kind": "local",
        "desired_time_seconds": 1.19,
        "requested_time_seconds": 1.19,
        "stream_index": 9,
        "frame_index": 99,
        "actual_pts_seconds": 1.19,
        "time_base": "1/90000",
        "source_media_sha256": media_sha256,
        "frame_plan_sha256": "f" * 64,
    }


def test_pyav_extractor_reports_nearest_decoded_identity_not_planned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = load_evidence_builder()
    media = tmp_path / "nearest-frame.mp4"
    media.write_bytes(b"fake-container")
    output = tmp_path / "nearest-frame.png"

    class FakeImage:
        def __init__(self, raw_pts: int) -> None:
            self.raw_pts = raw_pts

        def convert(self, mode: str):
            assert mode == "RGB"
            return self

        def save(self, path: Path, **kwargs: object) -> None:
            assert kwargs == {"format": "PNG", "optimize": True}
            Path(path).write_bytes(f"actual-nearest-frame-{self.raw_pts}".encode())

    class FakeDecodedFrame:
        def __init__(self, raw_pts: int) -> None:
            self.pts = raw_pts
            self.time_base = Fraction(1, 100)
            self.time = float(self.pts * self.time_base)

        def to_image(self) -> FakeImage:
            return FakeImage(self.pts)

    stream = SimpleNamespace(type="video", index=4, time_base=Fraction(1, 100))

    class FakeContainer:
        streams = [stream]

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def decode(self, selected_stream: object):
            assert selected_stream is stream
            return iter(
                [FakeDecodedFrame(100), FakeDecodedFrame(120), FakeDecodedFrame(140)]
            )

    fake_av = SimpleNamespace(open=lambda path: FakeContainer())
    monkeypatch.setitem(sys.modules, "av", fake_av)

    result = builder.extract_frame(
        {
            "media_path": str(media),
            "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        },
        _planned_materialization_frame("a" * 64),
        "H",
        output,
    )

    assert result == {
        "frame_path": str(output.resolve()),
        "stream_index": 4,
        "frame_index": 1,
        "decoded_frame_ordinal": 1,
        "time_base": "1/100",
        "raw_pts": 120,
        "actual_pts_seconds": 1.2,
        "image_sha256": hashlib.sha256(b"actual-nearest-frame-120").hexdigest(),
    }


def test_materialize_frame_uses_extractor_identity_when_nearest_frame_differs(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    media_sha256 = "a" * 64

    def mismatched_nearest_extractor(
        row: dict[str, object],
        frame: dict[str, object],
        view: str,
        output_path: Path,
    ) -> dict[str, object]:
        del row, frame, view
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"different-nearest-frame")
        return {
            "frame_path": str(output_path),
            "stream_index": 4,
            "frame_index": 7,
            "decoded_frame_ordinal": 7,
            "time_base": "1/48000",
            "raw_pts": 60000,
            "actual_pts_seconds": 1.25,
            "image_sha256": hashlib.sha256(b"different-nearest-frame").hexdigest(),
        }

    result = builder._materialize_frame(
        {"video_id": "video-01", "media_sha256": media_sha256},
        _planned_materialization_frame(media_sha256),
        "H",
        tmp_path / "frames",
        frame_extractor=mismatched_nearest_extractor,
        runtime_identity={"runtime": "test"},
    )

    assert {
        key: result[key]
        for key in (
            "stream_index",
            "frame_index",
            "decoded_frame_ordinal",
            "time_base",
            "raw_pts",
            "actual_pts_seconds",
        )
    } == {
        "stream_index": 4,
        "frame_index": 7,
        "decoded_frame_ordinal": 7,
        "time_base": "1/48000",
        "raw_pts": 60000,
        "actual_pts_seconds": 1.25,
    }


def test_materialize_frame_rejects_extractor_without_actual_decoded_identity(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    media_sha256 = "a" * 64

    def identity_free_extractor(
        row: dict[str, object],
        frame: dict[str, object],
        view: str,
        output_path: Path,
    ) -> dict[str, object]:
        del row, frame, view
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"frame-without-identity")
        return {
            "frame_path": str(output_path),
            "actual_pts_seconds": 1.2,
            "image_sha256": hashlib.sha256(b"frame-without-identity").hexdigest(),
        }

    with pytest.raises(ValueError, match="actual decoded identity"):
        builder._materialize_frame(
            {"video_id": "video-01", "media_sha256": media_sha256},
            _planned_materialization_frame(media_sha256),
            "H",
            tmp_path / "frames",
            frame_extractor=identity_free_extractor,
            runtime_identity={"runtime": "test"},
        )


def test_verified_copy_requires_independently_bound_decoded_identity(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    media_sha256 = "a" * 64
    source = tmp_path / "verified.png"
    source.write_bytes(b"verified-frame")
    planned = {
        **_planned_materialization_frame(media_sha256),
        "frame_path": str(source),
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

    with pytest.raises(ValueError, match="verified decoded identity"):
        builder._materialize_frame(
            {"video_id": "video-01", "media_sha256": media_sha256},
            planned,
            "G",
            tmp_path / "frames",
            frame_extractor=None,
            runtime_identity={"runtime": "test"},
        )


def test_verified_copy_uses_independently_bound_actual_identity(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    media_sha256 = "a" * 64
    source = tmp_path / "verified.png"
    source.write_bytes(b"verified-frame")
    planned = {
        **_planned_materialization_frame(media_sha256),
        "frame_path": str(source),
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "verified_decoded_identity": {
            "stream_index": 3,
            "frame_index": 8,
            "decoded_frame_ordinal": 8,
            "time_base": "1/24000",
            "raw_pts": None,
            "actual_pts_seconds": 1.2,
        },
    }

    result = builder._materialize_frame(
        {"video_id": "video-01", "media_sha256": media_sha256},
        planned,
        "G",
        tmp_path / "frames",
        frame_extractor=None,
        runtime_identity={"runtime": "test"},
    )

    assert result["materialization_mode"] == "verified-copy"
    assert result["stream_index"] == 3
    assert result["frame_index"] == result["decoded_frame_ordinal"] == 8
    assert result["time_base"] == "1/24000"
    assert result["raw_pts"] is None
    assert result["actual_pts_seconds"] == 1.2


@pytest.mark.parametrize(
    ("collision", "message"),
    [
        ("decoded_identity", "decoded frame identities"),
        ("actual_pts", "actual PTS"),
    ],
)
def test_request_uniqueness_uses_materialized_actual_identity_and_pts(
    tmp_path: Path, collision: str, message: str
) -> None:
    builder = load_evidence_builder()
    media_sha256 = "a" * 64
    materialized = []
    for index in range(12):
        planned = {
            **_planned_materialization_frame(media_sha256),
            "desired_time_seconds": float(index),
            "requested_time_seconds": float(index),
            "frame_index": 100 + index,
        }

        def actual_extractor(
            row: dict[str, object],
            frame: dict[str, object],
            view: str,
            output_path: Path,
            *,
            actual_index: int = index,
        ) -> dict[str, object]:
            del row, frame, view
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"actual-{actual_index}".encode("utf-8"))
            decoded_index = (
                0
                if collision == "decoded_identity" and actual_index == 11
                else actual_index
            )
            actual_pts = (
                10.0
                if collision == "actual_pts" and actual_index == 11
                else float(actual_index)
            )
            return {
                "frame_path": str(output_path),
                "stream_index": 2,
                "frame_index": decoded_index,
                "decoded_frame_ordinal": decoded_index,
                "time_base": "1/1000",
                "raw_pts": int(actual_pts * 1000),
                "actual_pts_seconds": actual_pts,
                "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }

        materialized.append(
            builder._materialize_frame(
                {"video_id": "video-01", "media_sha256": media_sha256},
                planned,
                "H",
                tmp_path / "frames",
                frame_extractor=actual_extractor,
                runtime_identity={"runtime": "test"},
            )
        )

    with pytest.raises(ValueError, match=message):
        build_request_matrix(
            row={"video_id": "video-01", "question": "What happens?"},
            frame_plans={
                "G": materialized,
                "H": materialized,
                "L": materialized,
                "localized": False,
            },
            transcript_views=build_transcript_views(
                None,
                [],
                duration_seconds=12.0,
            ),
            run_id="run-1",
        )


def test_evidence_builder_writes_nine_cells_and_one_contact_sheet_per_view(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    media = tmp_path / "video-01.mp4"
    media.write_bytes(b"video-bytes")
    media_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    inventory = frame_inventory(duration=20.0, count=121, media_sha=media_sha)
    baseline_plan = build_frame_plans(
        {"duration_seconds": 20.0, "whisper_segment_count": 4},
        localized_result(),
        inventory,
    )
    global_baseline = [dict(frame) for frame in baseline_plan["G"]]
    for frame in global_baseline:
        source = tmp_path / "verified-global" / f"frame-{frame['frame_index']:03d}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"verified-{frame['frame_index']}".encode("utf-8"))
        frame["frame_path"] = str(source)
        frame["image_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline_views = build_transcript_views(
        "[00:00:00.000] Hello.\n[00:00:04.000] Goodbye.",
        [1],
        duration_seconds=20.0,
    )
    output = tmp_path / "evidence"
    baseline_request_plans = _materialized_plans(
        baseline_plan,
        output=output,
        video_id="video-01",
    )
    baseline_request_plans["G"] = global_baseline
    baseline_request = build_request_matrix(
        row={"video_id": "video-01", "question": "What happens?"},
        frame_plans=baseline_request_plans,
        transcript_views=baseline_views,
        run_id="baseline",
    )["gf"]
    contact_sheets: list[tuple[str, str, list[str]]] = []
    extracted_views: list[str] = []

    def fake_contact_sheet(
        *, video_id: str, view: str, frame_paths: list[str], output_path: Path
    ) -> dict[str, object]:
        contact_sheets.append((video_id, view, frame_paths))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"contact-sheet")
        return {
            "path": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    def fake_frame_extractor(
        row: dict[str, object],
        frame: dict[str, object],
        view: str,
        output_path: Path,
    ) -> dict[str, object]:
        extracted_views.append(view)
        return _fake_frame_extractor(row, frame, view, output_path)

    selector_protocol = "b" * 64
    summary = builder.build_evidence_bundle(
        rows=[
            {
                "video_id": "video-01",
                "question": "What happens?",
                "duration_seconds": 20.0,
                "whisper_segment_count": 4,
                "media_path": str(media),
                "media_sha256": media_sha,
            }
        ],
        selectors=[
            {
                "video_id": "video-01",
                "protocol_sha256": selector_protocol,
                "selector": {
                    "status": "localized",
                    "intervals": [{"start_seconds": 3.0, "end_seconds": 8.0}],
                    "whisper_segment_ids": [1],
                },
            }
        ],
        global_evidence=[
            {
                "video_id": "video-01",
                "source_media_sha256": media_sha,
                "inventory": inventory,
                "frames": global_baseline,
            }
        ],
        whisper=[
            {
                "video_id": "video-01",
                "text": "[00:00:00.000] Hello.\n[00:00:04.000] Goodbye.",
            }
        ],
        baseline_requests={"video-01": baseline_request},
        output_dir=output,
        expected_count=1,
        expected_panel_id_digest=_id_digest(["video-01"]),
        expected_selector_protocol_sha256=selector_protocol,
        run_id="run-1",
        model_identity={"model_id": "Qwen/Qwen3.5-4B", "revision": "r1"},
        runtime_identity={"image": "runtime@sha256:" + "a" * 64},
        prompt_source_identity="prompt-sha",
        contact_sheet_builder=fake_contact_sheet,
        frame_extractor=fake_frame_extractor,
    )

    assert summary["cell_count"] == 9
    assert summary["row_count"] == 1
    assert summary["canonical_id_digest"] == _id_digest(["video-01"])
    assert summary["protocol_sha256"] == selector_protocol
    assert summary["selector_protocol_sha256"] == selector_protocol
    assert {path.name for path in (output / "requests").glob("*.json")} == {
        f"{cell}.json" for cell in EVIDENCE_CELLS
    }
    requests = json.loads((output / "requests/gf.json").read_text(encoding="utf-8"))
    assert isinstance(requests, list) and len(requests) == 1
    assert set(requests[0]) == {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "transcript",
    }
    assert requests[0]["condition_id"] == "gf"
    marker = json.loads((output / "evidence-marker.json").read_text(encoding="utf-8"))
    assert marker == {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "example_count": 1,
        "canonical_id_digest": _id_digest(["video-01"]),
        "checksum_file": "SHA256SUMS",
    }
    assert set(extracted_views) == {"H", "L"}
    assert [(video_id, view) for video_id, view, _ in contact_sheets] == [
        ("video-01", "G"),
        ("video-01", "H"),
        ("video-01", "L"),
    ]
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    row = evidence["rows"][0]
    assert (
        row["views"]["G"]["frames"][0]["actual_pts_seconds"]
        == global_baseline[0]["actual_pts_seconds"]
    )
    assert row["views"]["G"]["frames"][0]["materialization_mode"] == "verified-copy"
    assert (
        row["views"]["G"]["frames"][0]["decoded_frame_ordinal"]
        == global_baseline[0]["frame_index"]
    )
    assert row["views"]["G"]["frames"][0]["raw_pts"] is None
    assert (
        row["views"]["G"]["frames"][0]["image_sha256"]
        == global_baseline[0]["image_sha256"]
    )
    assert row["views"]["G"]["frames"][0]["extraction_command_identity"]
    assert row["views"]["G"]["contact_sheet"]["path"]
    assert row["cells"]["gf"]["model_input_sha256"] == model_input_sha256(
        baseline_request
    )
    assert row["cells"]["gn"]["transcript"] is None
    assert row["cells"]["gs"]["selected_line_indices"] == [1]
    audit = json.loads((output / "leakage-audit.json").read_text(encoding="utf-8"))
    assert audit["rows"][0]["cells"]["gf"]["leakage"] is False


def test_evidence_builder_rejects_wrong_panel_digest_before_materialization(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    video_ids = [f"notcanon{i:03d}" for i in range(475)]
    extracted: list[str] = []

    def fail_if_extracted(*args, **kwargs):
        del args, kwargs
        extracted.append("called")
        raise AssertionError("frame materialization must not start")

    with pytest.raises(ValueError, match="canonical.*digest"):
        builder.build_evidence_bundle(
            rows=[{"video_id": video_id} for video_id in video_ids],
            selectors=[
                {
                    "video_id": video_id,
                    "protocol_sha256": "b" * 64,
                    "selector": {},
                }
                for video_id in video_ids
            ],
            global_evidence=[{"video_id": video_id} for video_id in video_ids],
            whisper=[{"video_id": video_id} for video_id in video_ids],
            baseline_requests={video_id: {} for video_id in video_ids},
            output_dir=tmp_path / "evidence",
            expected_selector_protocol_sha256="b" * 64,
            run_id="run-1",
            model_identity={"model_id": "model"},
            runtime_identity={"image": "runtime"},
            prompt_source_identity="prompt",
            frame_extractor=fail_if_extracted,
        )

    assert extracted == []
    assert not (tmp_path / "evidence").exists()


def test_evidence_builder_rejects_selector_protocol_before_materialization(
    tmp_path: Path,
) -> None:
    builder = load_evidence_builder()
    extracted: list[str] = []

    def fail_if_extracted(*args, **kwargs):
        del args, kwargs
        extracted.append("called")
        raise AssertionError("frame materialization must not start")

    with pytest.raises(ValueError, match="selector protocol"):
        builder.build_evidence_bundle(
            rows=[{"video_id": "video-01"}],
            selectors=[
                {
                    "video_id": "video-01",
                    "protocol_sha256": "c" * 64,
                    "selector": {},
                }
            ],
            global_evidence=[{"video_id": "video-01"}],
            whisper=[{"video_id": "video-01"}],
            baseline_requests={"video-01": {}},
            output_dir=tmp_path / "evidence",
            expected_count=1,
            expected_panel_id_digest=_id_digest(["video-01"]),
            expected_selector_protocol_sha256="b" * 64,
            run_id="run-1",
            model_identity={"model_id": "model"},
            runtime_identity={"image": "runtime"},
            prompt_source_identity="prompt",
            frame_extractor=fail_if_extracted,
        )

    assert extracted == []
    assert not (tmp_path / "evidence").exists()


def test_task4_bundle_is_consumed_by_task5_runner_contract(tmp_path: Path) -> None:
    builder = load_evidence_builder()
    task5 = load_task5_runner_tests()
    manifest = json.loads(task5.FULL_MANIFEST.read_text(encoding="utf-8"))
    examples = manifest["examples"]
    video_ids = [example["video_id"] for example in examples]
    assert _id_digest(video_ids) == task5.CANONICAL_ID_DIGEST

    media = tmp_path / "shared.mp4"
    media.write_bytes(b"label-free-video-fixture")
    media_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    inventory = frame_inventory(duration=10.0, count=12, media_sha=media_sha)
    selector = {
        "status": "localized",
        "intervals": [{"start_seconds": 0.0, "end_seconds": 10.0}],
        "whisper_segment_ids": [0],
    }
    protocol = "d" * 64
    output = tmp_path / "task4-evidence"
    rows = [
        {
            "video_id": example["video_id"],
            "question": example["question"],
            "duration_seconds": 10.0,
            "whisper_segment_count": 1,
            "media_path": str(media),
            "media_sha256": media_sha,
        }
        for example in examples
    ]
    selectors = [
        {
            "video_id": video_id,
            "protocol_sha256": protocol,
            "selector": selector,
        }
        for video_id in video_ids
    ]
    global_evidence = []
    baseline_requests = {}
    transcript_views = build_transcript_views(
        "[00:00:00.000] Fixture transcript.",
        [0],
        duration_seconds=10.0,
    )
    for row in rows:
        plans = build_frame_plans(row, selector, inventory)
        global_evidence.append(
            {
                "video_id": row["video_id"],
                "source_media_sha256": media_sha,
                "inventory": inventory,
                "frames": plans["G"],
            }
        )
        baseline_requests[row["video_id"]] = build_request_matrix(
            row=row,
            frame_plans=_materialized_plans(
                plans,
                output=output,
                video_id=row["video_id"],
            ),
            transcript_views=transcript_views,
            run_id="baseline",
        )["gf"]

    builder.build_evidence_bundle(
        rows=rows,
        selectors=selectors,
        global_evidence=global_evidence,
        whisper=[
            {
                "video_id": video_id,
                "text": "[00:00:00.000] Fixture transcript.",
            }
            for video_id in video_ids
        ],
        baseline_requests=baseline_requests,
        output_dir=output,
        expected_selector_protocol_sha256=protocol,
        run_id=task5.RUN_ID,
        model_identity={"model_id": task5.MODEL_ID, "revision": task5.MODEL_REVISION},
        runtime_identity={"image": task5.RUNTIME_IMAGE},
        prompt_source_identity="current",
        contact_sheet_builder=_fake_contact_sheet,
        frame_extractor=_fake_frame_extractor,
    )

    result = task5._run_runner(
        tmp_path / "task5-runner",
        input_root=output,
        cell="gn",
    )
    assert result.returncode == 0, result.stdout + result.stderr
