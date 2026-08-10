from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import TimelineSpan
from trec_video_2026.vqa_submission_evidence import (
    EvidencePlan,
    FrameSelection,
    build_global_local_frame_plan,
    select_overlapping_transcript_segments,
    validate_timeline,
    write_evidence_manifest,
)


SOURCE_SHA = "a" * 64


def _span(
    start_seconds: float,
    end_seconds: float,
    *,
    span_id: str = "span-1",
    modalities: tuple[str, ...] = ("visual",),
) -> TimelineSpan:
    return TimelineSpan(
        span_id=span_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        modalities=modalities,
        description="A bounded synthetic event",
    )


def _timeline(
    *,
    end_seconds: float = 4.0,
    support_spans: tuple[TimelineSpan, ...] | None = None,
) -> dict[str, object]:
    spans = support_spans or (_span(1.0, end_seconds - 1.0),)
    return {
        "events": (_span(0.0, 1.0, span_id="event-1"),),
        "support_spans": spans,
    }


def _inventory(count: int, *, duration_seconds: float = 15.0) -> list[dict[str, object]]:
    return [
        {
            "stream_index": 0,
            "frame_index": index,
            "actual_pts_seconds": round(index * duration_seconds / count, 6),
            "time_base": "1/1000",
            "source_media_sha256": SOURCE_SHA,
        }
        for index in range(count)
    ]


def _multi_stream_inventory() -> list[dict[str, object]]:
    return [
        {
            "stream_index": stream_index,
            "frame_index": frame_index,
            "actual_pts_seconds": round(frame_index * 15.0 / 16.0, 6),
            "time_base": "1/1000",
            "source_media_sha256": SOURCE_SHA,
        }
        for frame_index in range(16)
        for stream_index in range(2)
    ]


def _three_spans() -> tuple[TimelineSpan, ...]:
    return (
        _span(1.0, 5.0, span_id="span-1"),
        _span(6.0, 10.0, span_id="span-2"),
        _span(11.0, 15.0, span_id="span-3"),
    )


def _frame(
    *,
    requested_time_seconds: float = 0.5,
    actual_pts_seconds: float = 0.48,
    frame_index: int = 0,
    slot_kind: str = "global",
    support_span_id: str | None = None,
    fallback_reason: str | None = None,
    frame_path: Path | None = None,
    image_sha256: str | None = "b" * 64,
) -> FrameSelection:
    return FrameSelection(
        requested_time_seconds=requested_time_seconds,
        actual_pts_seconds=actual_pts_seconds,
        stream_index=0,
        frame_index=frame_index,
        time_base="1/1000",
        source_media_sha256=SOURCE_SHA,
        image_sha256=image_sha256,
        frame_path=frame_path or Path(f"/tmp/frame-{frame_index:03d}.jpg"),
        slot_kind=slot_kind,
        support_span_id=support_span_id,
        fallback_reason=fallback_reason,
    )


def _evidence_plan() -> EvidencePlan:
    frames = tuple(
        _frame(
            requested_time_seconds=0.5 + index * 0.4,
            actual_pts_seconds=0.48 + index * 0.4,
            frame_index=index,
        )
        for index in range(8)
    )
    return EvidencePlan(
        duration_seconds=4.0,
        global_frames=frames,
        local_frames=(),
        support_spans=(_span(1.0, 2.0),),
    )


def _materialized_evidence_plan(tmp_path: Path) -> EvidencePlan:
    plan = _evidence_plan()
    frames = []
    for index, frame in enumerate(plan.global_frames):
        frame_path = tmp_path / f"frame-{index:03d}.jpg"
        frame_path.write_bytes(f"synthetic-frame-{index}".encode())
        frames.append(
            dataclasses.replace(
                frame,
                frame_path=frame_path,
                image_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            )
        )
    return dataclasses.replace(plan, global_frames=tuple(frames))


def test_timeline_rejects_overlapping_or_out_of_range_spans() -> None:
    malformed = _timeline(
        end_seconds=31.0,
        support_spans=(_span(1.0, 31.0),),
    )

    with pytest.raises(ValueError, match="duration"):
        validate_timeline(malformed, duration_seconds=30.0)


def test_invalid_timeline_uses_declared_global_fallback() -> None:
    malformed = {
        "events": (_span(2.0, 3.0, span_id="event-1"), _span(1.0, 2.0, span_id="event-2")),
        "support_spans": (_span(1.0, 2.0),),
    }

    result = validate_timeline(malformed, duration_seconds=15.0, allow_fallback=True)

    assert result.fallback_reason == "invalid_timeline"
    assert result.support_spans == ()


def test_timeline_rejects_nested_reference_fields_before_normalization() -> None:
    malformed = {
        "events": (),
        "support_spans": (
            {
                "span_id": "span-1",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "modalities": ("visual",),
                "description": "bounded event",
                "reference_path": "/tmp/gold.json",
            },
        ),
    }

    with pytest.raises(ValueError, match="reference"):
        validate_timeline(malformed, duration_seconds=4.0)


def test_short_support_span_expands_to_one_second_and_validates_modalities() -> None:
    result = validate_timeline(
        _timeline(support_spans=(_span(4.8, 5.0),)), duration_seconds=10.0
    )

    assert result.support_spans[0].start_seconds == 4.4
    assert result.support_spans[0].end_seconds == 5.4


def test_timeline_rejects_media_shorter_than_one_second() -> None:
    with pytest.raises(ValueError, match="one second"):
        validate_timeline({"events": (), "support_spans": ()}, duration_seconds=0.8)


def test_frame_plan_keeps_eight_global_and_allocates_twenty_four_local() -> None:
    plan = build_global_local_frame_plan(15.0, _inventory(450), _three_spans())

    assert len(plan.global_frames) == 8
    assert len(plan.local_frames) == 24
    assert all(frame.actual_pts_seconds >= 0 for frame in plan.all_frames)
    assert len({frame.decoded_identity for frame in plan.all_frames}) == 32
    assert all(frame.slot_kind == "global" for frame in plan.global_frames)
    assert all(frame.slot_kind == "local" for frame in plan.local_frames)


def test_multistream_inventory_allows_shared_pts_but_keeps_decoded_identity_unique() -> None:
    plan = build_global_local_frame_plan(
        15.0,
        _multi_stream_inventory(),
        [_span(2.0, 3.0)],
    )

    assert len(plan.all_frames) == 12
    assert len({frame.decoded_identity for frame in plan.all_frames}) == 12
    assert len({frame.actual_pts_seconds for frame in plan.all_frames}) < 12


def test_raw_support_span_is_fully_normalized_for_frame_planning() -> None:
    plan = build_global_local_frame_plan(
        10.0,
        _inventory(300, duration_seconds=10.0),
        [_span(4.8, 5.0)],
    )

    assert len(plan.local_frames) == 4
    assert plan.support_spans[0].start_seconds == 4.4
    assert plan.support_spans[0].end_seconds == 5.4


def test_raw_support_span_ids_are_validated_for_transcript_selection() -> None:
    duplicate_ids = (
        _span(1.0, 2.0, span_id="duplicate"),
        _span(2.0, 3.0, span_id="duplicate"),
    )

    with pytest.raises(ValueError, match="unique"):
        select_overlapping_transcript_segments(
            "[00:00:00] words",
            duplicate_ids,
            duration_seconds=4.0,
        )


def test_raw_short_support_span_is_expanded_for_transcript_overlap() -> None:
    selected = select_overlapping_transcript_segments(
        "[00:05.00] fifth second",
        [_span(4.8, 5.0)],
        duration_seconds=6.0,
    )

    assert selected.text == "[00:05.00] fifth second"


def test_local_plan_caps_each_span_at_four_frames_per_second() -> None:
    plan = build_global_local_frame_plan(
        15.0, _inventory(450), [_span(2.0, 3.0)]
    )

    assert len(plan.local_frames) <= 4
    assert {frame.support_span_id for frame in plan.local_frames} == {"span-1"}


def test_local_collision_records_fallback_and_preserves_unique_decoded_frames() -> None:
    inventory = _inventory(40)
    plan = build_global_local_frame_plan(
        15.0,
        inventory,
        [_span(2.0, 2.1)],
    )

    assert plan.local_frames
    assert any(frame.fallback_reason == "local_collision_fallback" for frame in plan.local_frames)
    assert len({frame.decoded_identity for frame in plan.all_frames}) == len(plan.all_frames)


def test_selected_transcript_is_verbatim_and_overlap_only() -> None:
    timestamped_text = "[00:00.00] first line\n[00:01.00] second line\n[00:02.00] third line"

    selected = select_overlapping_transcript_segments(
        timestamped_text, [_span(1.0, 2.0)], duration_seconds=4.0
    )

    assert selected.text == "[00:01.00] second line"
    assert selected.source_sha256 == hashlib.sha256(timestamped_text.encode()).hexdigest()


def test_transcript_selection_rejects_media_overrun() -> None:
    with pytest.raises(ValueError, match="duration"):
        select_overlapping_transcript_segments(
            "[00:04.00] beyond", [_span(1.0, 2.0)], duration_seconds=4.0
        )


def test_manifest_records_requested_pts_and_hashes(tmp_path: Path) -> None:
    destination = tmp_path / "evidence.json"

    record = write_evidence_manifest(destination, _materialized_evidence_plan(tmp_path))

    assert record["frames"][0]["requested_time_seconds"] == 0.5
    assert record["frames"][0]["actual_pts_seconds"] == 0.48
    assert len(record["frames"][0]["image_sha256"]) == 64
    assert record["frames"][0]["source_media_sha256"] == SOURCE_SHA
    assert destination.exists()

    with pytest.raises(ValueError, match="overwrite"):
        write_evidence_manifest(destination, _evidence_plan())


def test_manifest_rejects_label_bearing_frame_path(tmp_path: Path) -> None:
    evidence = _evidence_plan()
    first = dataclasses.replace(
        evidence.global_frames[0],
        frame_path=Path("/tmp/reference/frame.jpg"),
    )
    evidence = dataclasses.replace(
        evidence,
        global_frames=(first,) + evidence.global_frames[1:],
    )

    with pytest.raises(ValueError, match="reference"):
        write_evidence_manifest(tmp_path / "evidence.json", evidence)


def test_manifest_rejects_null_frame_path(tmp_path: Path) -> None:
    evidence = _materialized_evidence_plan(tmp_path)
    first = dataclasses.replace(evidence.global_frames[0], frame_path=None)
    evidence = dataclasses.replace(
        evidence,
        global_frames=(first,) + evidence.global_frames[1:],
    )

    with pytest.raises(ValueError, match="frame_path"):
        write_evidence_manifest(tmp_path / "evidence.json", evidence)


def test_manifest_rejects_null_image_hash(tmp_path: Path) -> None:
    evidence = _materialized_evidence_plan(tmp_path)
    first = dataclasses.replace(evidence.global_frames[0], image_sha256=None)
    evidence = dataclasses.replace(
        evidence,
        global_frames=(first,) + evidence.global_frames[1:],
    )

    with pytest.raises(ValueError, match="image_sha256"):
        write_evidence_manifest(tmp_path / "evidence.json", evidence)


def test_manifest_rejects_missing_frame_file(tmp_path: Path) -> None:
    evidence = _evidence_plan()

    with pytest.raises(ValueError, match="frame_path"):
        write_evidence_manifest(tmp_path / "evidence.json", evidence)


def test_manifest_rejects_image_hash_mismatch(tmp_path: Path) -> None:
    evidence = _materialized_evidence_plan(tmp_path)
    first = dataclasses.replace(evidence.global_frames[0], image_sha256="b" * 64)
    evidence = dataclasses.replace(
        evidence,
        global_frames=(first,) + evidence.global_frames[1:],
    )

    with pytest.raises(ValueError, match="image_sha256"):
        write_evidence_manifest(tmp_path / "evidence.json", evidence)


def test_evidence_plan_enforces_durable_frame_shape() -> None:
    plan = _evidence_plan()

    with pytest.raises(ValueError, match="exactly eight"):
        dataclasses.replace(plan, global_frames=plan.global_frames[:-1])

    bad_timestamp = dataclasses.replace(plan.global_frames[0], actual_pts_seconds=4.1)
    with pytest.raises(ValueError, match="duration"):
        dataclasses.replace(plan, global_frames=(bad_timestamp,) + plan.global_frames[1:])

    bad_global_slot = dataclasses.replace(
        plan.global_frames[0], slot_kind="local", support_span_id="span-1"
    )
    with pytest.raises(ValueError, match="global"):
        dataclasses.replace(plan, global_frames=(bad_global_slot,) + plan.global_frames[1:])

    bad_local_slot = dataclasses.replace(
        plan.global_frames[0],
        frame_index=8,
        slot_kind="local",
        support_span_id=None,
    )
    with pytest.raises(ValueError, match="support_span_id"):
        dataclasses.replace(plan, local_frames=(bad_local_slot,))


def test_evidence_plan_rejects_empty_plan() -> None:
    with pytest.raises(ValueError, match="empty"):
        EvidencePlan(duration_seconds=4.0)
