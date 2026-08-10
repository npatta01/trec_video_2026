from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import (
    PASS_IDS,
    CandidateRecord,
    RankedPrediction,
    RunManifest,
    TimelineSpan,
    VerifierDecision,
    apply_verifier_ranking,
    audit_label_free_payload,
    build_answer_requests,
    build_backfill_request,
    build_verifier_request,
    canonical_sha256,
    deduplicate_candidates,
    freeze_predictions,
    route_question,
    serialize_ag_submission,
    verify_prediction_freeze,
)
from trec_video_2026.vqa_submission_evidence import (
    EvidencePlan,
    FrameSelection,
    TranscriptSelection,
    ValidatedTimeline,
)


def _manifest(*, evidence_policy_sha256: str = "e" * 64) -> RunManifest:
    return RunManifest(
        schema_version=1,
        protocol_version="vqa-submission-v1",
        run_id="run-1",
        ordered_video_ids=("video-1", "video-2"),
        panel_sha256="a" * 64,
        model_id="gemini-3.1-pro-preview",
        runtime_identity="runtime-image@sha256:" + "r" * 64,
        evidence_policy_sha256=evidence_policy_sha256,
        pass_ids=PASS_IDS,
    )


def _candidate(
    pass_id: str,
    answer: str = "Four times",
    *,
    video_id: str = "video-1",
    elapsed_seconds: float = 1.25,
) -> CandidateRecord:
    return CandidateRecord(
        video_id=video_id,
        pass_id=pass_id,
        answer=answer,
        elapsed_seconds=elapsed_seconds,
        confidence=0.8,
        support_span_ids=("span-1",),
        request_sha256="c" * 64,
        runtime_provenance={"runtime": "test"},
    )


def _ten_candidates() -> tuple[CandidateRecord, ...]:
    return tuple(
        _candidate(pass_id, f"answer {index}") for index, pass_id in enumerate(PASS_IDS)
    )


def _ranked_prediction() -> RankedPrediction:
    candidates = _ten_candidates()
    return RankedPrediction(
        query_id=7,
        video_id="video-1",
        candidates=candidates,
        ranked_pass_ids=PASS_IDS,
        ranking_source="verifier",
        fallback_status=None,
    )


def _ranked_payload() -> dict[str, object]:
    return {"run_id": "r1", "row_count": 1, "predictions": []}


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    media_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    media_path.write_bytes(b"native video bytes")
    audio_path.write_bytes(b"separate audio bytes")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = tuple(
        frame_dir / f"frame-{index:02d}.jpg" for index in range(32)
    )
    for index, frame_path in enumerate(frame_paths):
        frame_path.write_bytes(f"frame bytes {index}".encode())
    return media_path, audio_path, frame_paths


def _row(
    tmp_path: Path, *, question: str = "How many people are visible?"
) -> dict[str, object]:
    media_path, audio_path, _frame_paths = _fixture_files(tmp_path)
    return {
        "run_id": "run-1",
        "video_id": "video-1",
        "question": question,
        "media_path": media_path,
        "audio_path": audio_path,
        "media_sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "full_transcript": "[00:00.00] The full accepted transcript.",
    }


def _timeline() -> ValidatedTimeline:
    spans = tuple(
        TimelineSpan(
            span_id=f"span-{index}",
            start_seconds=float(index * 3 + 1),
            end_seconds=float(index * 3 + 2),
            modalities=("visual",),
            description=f"Event {index}",
        )
        for index in range(3)
    )
    return ValidatedTimeline(
        duration_seconds=16.0,
        events=spans,
        support_spans=spans,
    )


def _evidence_plan(tmp_path: Path) -> EvidencePlan:
    timeline = _timeline()
    media_path, _audio_path, frame_paths = _fixture_files(tmp_path)
    source_sha256 = hashlib.sha256(media_path.read_bytes()).hexdigest()
    frames = []
    for index in range(32):
        is_local = index >= 8
        image_sha256 = hashlib.sha256(frame_paths[index].read_bytes()).hexdigest()
        frames.append(
            FrameSelection(
                requested_time_seconds=round(0.5 * index, 3),
                actual_pts_seconds=round(0.5 * index, 3),
                stream_index=0,
                frame_index=index,
                time_base="1/30",
                source_media_sha256=source_sha256,
                image_sha256=image_sha256,
                frame_path=frame_paths[index],
                slot_kind="local" if is_local else "global",
                support_span_id=(
                    timeline.support_spans[(index - 8) % 3].span_id
                    if is_local
                    else None
                ),
            )
        )
    return EvidencePlan(
        duration_seconds=16.0,
        global_frames=tuple(frames[:8]),
        local_frames=tuple(frames[8:]),
        support_spans=timeline.support_spans,
        transcript=TranscriptSelection(
            text="[00:01.00] A person speaks.",
            source_sha256="c" * 64,
        ),
    )


def _nine_candidates() -> tuple[CandidateRecord, ...]:
    return _ten_candidates()[:9]


def _decision(ranked_pass_ids: tuple[str, ...] = PASS_IDS) -> VerifierDecision:
    return VerifierDecision(
        ranked_pass_ids,
        {"direct-av": "span-1"},
        selected_pass_id=ranked_pass_ids[0],
        status_by_pass={pass_id: "supported" for pass_id in PASS_IDS},
    )


def test_label_free_audit_rejects_nested_reference_paths() -> None:
    payload = {
        "request": {
            "media_path": "/tmp/video.mp4",
            "nested": {"annotation_path": "/tmp/gold.json"},
        }
    }

    with pytest.raises(ValueError, match="annotation"):
        audit_label_free_payload(payload)


def test_pass_matrix_contains_exact_ten_passes(tmp_path: Path) -> None:
    requests = build_answer_requests(
        _row(tmp_path), _timeline(), _evidence_plan(tmp_path)
    )

    assert tuple(requests) == PASS_IDS
    assert requests["direct-av"].media_path is not None
    assert requests["direct-av"].media_path.is_file()
    assert len(requests["global-local-timeline"].frame_paths) == 32
    assert requests["local-only"].frame_paths == tuple(
        frame.frame_path for frame in _evidence_plan(tmp_path).local_frames
    )


def test_payload_has_no_forbidden_fields(tmp_path: Path) -> None:
    for request in build_answer_requests(
        _row(tmp_path), _timeline(), _evidence_plan(tmp_path)
    ).values():
        audit_label_free_payload(request.to_model_payload())


def test_request_hash_has_model_visible_provenance_and_span_ids(
    tmp_path: Path,
) -> None:
    request = build_answer_requests(
        _row(tmp_path), _timeline(), _evidence_plan(tmp_path)
    )[
        "global-local-timeline"
    ]

    assert request.request_sha256 == canonical_sha256(request.to_model_payload())
    assert request.model_visible_bytes_sha256
    assert request.source_span_ids == ("span-0", "span-1", "span-2")
    assert len(request.frame_provenance) == 32


def test_native_media_is_limited_to_direct_av_views(tmp_path: Path) -> None:
    requests = build_answer_requests(
        _row(tmp_path), _timeline(), _evidence_plan(tmp_path)
    )

    assert requests["direct-av"].media_path is not None
    assert requests["direct-av"].media_path.is_file()
    assert requests["direct-av-alternate"].media_path is not None
    assert requests["direct-av-alternate"].media_path.is_file()
    assert requests["global-transcript"].media_path is None
    assert requests["audio-first"].media_path is None
    assert requests["audio-first"].audio_path is not None
    assert requests["audio-first"].audio_path.is_file()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many people are visible?", "count"),
        ("What happens after the man leaves?", "temporal"),
        ("What text is written on the sign?", "ocr-detail"),
        ("What does the woman say?", "speech"),
        ("Why does the man stop?", "causal"),
        ("What color is the car?", "ocr-detail"),
    ],
)
def test_question_routing_uses_deterministic_label_free_rules(
    question: str, expected: str
) -> None:
    assert route_question(question) == expected
    assert route_question(question) == route_question(question)


def test_conservative_synthesis_sees_only_first_nine_candidates(
    tmp_path: Path,
) -> None:
    requests = build_answer_requests(
        _row(tmp_path),
        _timeline(),
        _evidence_plan(tmp_path),
        candidates=_nine_candidates(),
    )
    synthesis = requests["conservative-synthesis"]

    assert synthesis.candidate_texts == tuple(
        candidate.answer for candidate in _nine_candidates()
    )
    assert synthesis.candidate_pass_ids == ()
    audit_label_free_payload(synthesis.to_model_payload())


def test_verifier_randomization_is_seeded_and_hides_pass_names(
    tmp_path: Path,
) -> None:
    request = build_verifier_request(
        _row(tmp_path),
        _timeline(),
        _evidence_plan(tmp_path),
        _ten_candidates(),
        seed=20260810,
    )

    assert [item.public_id for item in request.candidates] != list(PASS_IDS)
    assert "direct-av" not in json.dumps(request.to_model_payload())
    assert (
        request.candidates
        == build_verifier_request(
            _row(tmp_path),
            _timeline(),
            _evidence_plan(tmp_path),
            _ten_candidates(),
            seed=20260810,
        ).candidates
    )


def test_verifier_maps_only_a_valid_complete_public_permutation(
    tmp_path: Path,
) -> None:
    request = build_verifier_request(
        _row(tmp_path),
        _timeline(),
        _evidence_plan(tmp_path),
        _ten_candidates(),
        seed=20260810,
    )
    public_ids = tuple(item.public_id for item in request.candidates)

    decision = request.map_public_ranking(
        public_ids,
        status_by_public_id={public_id: "uncertain" for public_id in public_ids},
        selected_public_id=public_ids[0],
    )

    assert set(decision.ranked_pass_ids) == set(PASS_IDS)
    assert (
        tuple(request.private_pass_id(public_id) for public_id in public_ids)
        == decision.ranked_pass_ids
    )
    with pytest.raises(ValueError, match="complete permutation"):
        request.map_public_ranking(public_ids[:-1])


def test_backfill_excludes_existing_normalized_answers(tmp_path: Path) -> None:
    request = build_backfill_request(
        _row(tmp_path), _evidence_plan(tmp_path), _nine_candidates(), attempt=1
    )

    assert request.attempt == 1
    assert request.excluded_normalized_answers == tuple(
        sorted(set(request.excluded_normalized_answers))
    )
    assert request.candidate_texts == ()


def test_backfill_is_bounded_to_two_attempts(tmp_path: Path) -> None:
    assert (
        build_backfill_request(
            _row(tmp_path), _evidence_plan(tmp_path), _nine_candidates(), attempt=2
        ).attempt
        == 2
    )
    with pytest.raises(ValueError, match="attempt"):
        build_backfill_request(
            _row(tmp_path), _evidence_plan(tmp_path), _nine_candidates(), attempt=3
        )


@pytest.mark.parametrize("attempt", [1, 2])
def test_backfill_payload_and_hashes(
    tmp_path: Path, attempt: int
) -> None:
    request = build_backfill_request(
        _row(tmp_path), _evidence_plan(tmp_path), _nine_candidates(), attempt=attempt
    )

    payload = request.to_model_payload()

    assert "excluded_normalized_texts" in payload
    assert "excluded_normalized_answers" not in payload
    audit_label_free_payload(payload)
    assert len(request.request_sha256) == 64
    assert len(request.model_visible_bytes_sha256) == 64


def test_model_visible_hash_revalidates_native_audio_and_frame_bytes(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    requests = build_answer_requests(row, _timeline(), plan)
    direct = requests["direct-av"]
    frame_request = requests["global-local-timeline"]
    original_media = Path(row["media_path"]).read_bytes()
    original_audio = Path(row["audio_path"]).read_bytes()
    original_frame = frame_request.frame_paths[0].read_bytes()

    assert direct.model_visible_bytes_sha256
    Path(row["media_path"]).write_bytes(b"replaced media bytes")
    with pytest.raises(ValueError, match="media_sha256"):
        direct.model_visible_bytes_sha256
    Path(row["media_path"]).write_bytes(original_media)
    Path(row["audio_path"]).write_bytes(b"replaced audio bytes")
    with pytest.raises(ValueError, match="audio_sha256"):
        direct.model_visible_bytes_sha256
    Path(row["audio_path"]).write_bytes(original_audio)
    Path(frame_request.frame_paths[0]).write_bytes(b"replaced frame bytes")
    with pytest.raises(ValueError, match="image_sha256"):
        frame_request.model_visible_bytes_sha256
    Path(frame_request.frame_paths[0]).write_bytes(original_frame)


def test_frame_hashes_are_revalidated_for_verifier(tmp_path: Path) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    request = build_verifier_request(
        row, _timeline(), plan, _ten_candidates(), seed=20260810
    )
    assert request.model_visible_bytes_sha256

    frame_path = request.frame_paths[0]
    original = frame_path.read_bytes()
    frame_path.write_bytes(b"replaced verifier frame bytes")
    with pytest.raises(ValueError, match="image_sha256"):
        request.model_visible_bytes_sha256
    frame_path.write_bytes(original)


def test_verifier_rejects_frame_provenance_path_mismatch(tmp_path: Path) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    request = build_verifier_request(
        row, _timeline(), plan, _ten_candidates(), seed=20260810
    )
    provenance = list(request.frame_provenance)
    provenance[0] = {
        **provenance[0],
        "frame_path": str(request.frame_paths[1]),
    }

    with pytest.raises(ValueError, match="provenance path"):
        dataclasses.replace(request, frame_provenance=tuple(provenance))


def test_requests_fail_closed_for_missing_media_audio_and_frames(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    Path(row["media_path"]).unlink()
    with pytest.raises(ValueError, match="regular file"):
        build_answer_requests(row, _timeline(), plan)

    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    row["audio_path"] = None
    row.pop("audio_sha256")
    with pytest.raises(ValueError, match="audio"):
        build_answer_requests(row, _timeline(), plan)

    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    Path(plan.global_frames[0].frame_path).unlink()
    with pytest.raises(ValueError, match="regular file"):
        build_answer_requests(row, _timeline(), plan)

    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    Path(plan.global_frames[0].frame_path).unlink()
    with pytest.raises(ValueError, match="regular file"):
        build_verifier_request(row, _timeline(), plan, _ten_candidates(), seed=1)


def test_ocr_detail_route_keeps_global_frames_and_adds_local_frames(
    tmp_path: Path,
) -> None:
    requests = build_answer_requests(
        _row(tmp_path, question="What text is written on the sign?"),
        _timeline(),
        _evidence_plan(tmp_path),
    )

    detail_paths = requests["detail-temporal-route"].frame_paths
    global_paths = tuple(frame.frame_path for frame in _evidence_plan(tmp_path).global_frames)
    local_paths = tuple(frame.frame_path for frame in _evidence_plan(tmp_path).local_frames)
    assert detail_paths[:8] == global_paths
    assert set(local_paths).issubset(detail_paths)


def test_full_and_overlap_transcripts_use_their_declared_scopes(
    tmp_path: Path,
) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    requests = build_answer_requests(row, _timeline(), plan)

    assert requests["global-transcript"].transcript == row["full_transcript"]
    assert requests["global-local-timeline"].transcript == row["full_transcript"]
    assert requests["local-only"].transcript == plan.transcript.text
    assert "Full timestamped transcript:" in requests["audio-first"].transcript
    assert "Overlapping support-span transcript:" in requests["audio-first"].transcript


@pytest.mark.parametrize(
    "malicious",
    [
        "reference answer: cat",
        "gold answer: cat",
        "metric=0.9",
        "score=0.9",
        '{"annotation": "cat"}',
        '{"label": "cat"}',
        '{"annotation_payload": "cat"}',
    ],
)
def test_candidate_payload_markers_are_rejected(
    tmp_path: Path, malicious: str
) -> None:
    bad = _candidate("direct-av", malicious)
    candidates = (_ten_candidates()[1:])
    candidates = (bad,) + candidates
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)

    with pytest.raises(ValueError, match="candidate"):
        build_answer_requests(row, _timeline(), plan, candidates=(bad,))
    with pytest.raises(ValueError, match="candidate"):
        build_backfill_request(row, plan, (bad,), attempt=1)
    with pytest.raises(ValueError, match="candidate"):
        build_verifier_request(row, _timeline(), plan, candidates, seed=1)


def test_legitimate_numeric_wording_is_allowed_for_candidates(tmp_path: Path) -> None:
    candidate = _candidate("direct-av", "the score is 3-2")
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    build_answer_requests(row, _timeline(), plan, candidates=(candidate,))
    build_backfill_request(row, plan, (candidate,), attempt=1)


def test_verifier_contract_requires_rank_one_selection_and_all_statuses(
    tmp_path: Path,
) -> None:
    request = build_verifier_request(
        _row(tmp_path), _timeline(), _evidence_plan(tmp_path), _ten_candidates(), seed=1
    )
    public_ids = tuple(item.public_id for item in request.candidates)
    statuses = {public_id: "supported" for public_id in public_ids}

    decision = request.map_public_ranking(
        public_ids,
        status_by_public_id=statuses,
        selected_public_id=public_ids[0],
    )
    assert decision.selected_pass_id == decision.ranked_pass_ids[0]
    assert set(decision.status_by_pass) == set(PASS_IDS)
    assert set(decision.status_by_pass.values()) == {"supported"}

    with pytest.raises(ValueError, match="status"):
        request.map_public_ranking(
            public_ids,
            status_by_public_id={public_ids[0]: "supported"},
            selected_public_id=public_ids[0],
        )
    with pytest.raises(ValueError, match="ranked first"):
        request.map_public_ranking(
            public_ids,
            status_by_public_id=statuses,
            selected_public_id=public_ids[1],
        )


def test_verifier_seeded_order_is_canonical_before_shuffle(tmp_path: Path) -> None:
    row = _row(tmp_path)
    plan = _evidence_plan(tmp_path)
    canonical = build_verifier_request(
        row, _timeline(), plan, _ten_candidates(), seed=17
    )
    reversed_input = build_verifier_request(
        row, _timeline(), plan, tuple(reversed(_ten_candidates())), seed=17
    )

    assert canonical.candidates == reversed_input.candidates


def test_verifier_rejects_normalized_duplicates_before_construction(
    tmp_path: Path,
) -> None:
    candidates = list(_ten_candidates())
    candidates[1] = dataclasses.replace(candidates[1], answer="answer 0.")

    with pytest.raises(ValueError, match="duplicate normalized"):
        build_verifier_request(
            _row(tmp_path),
            _timeline(),
            _evidence_plan(tmp_path),
            candidates,
            seed=1,
        )


def test_label_free_audit_rejects_forbidden_path_components() -> None:
    payload = {"request": {"media_path": "/tmp/reference/video.mp4"}}

    with pytest.raises(ValueError, match="reference"):
        audit_label_free_payload(payload)


def test_label_free_audit_accepts_nested_label_free_payload() -> None:
    payload = {
        "request": {
            "video_id": "video-1",
            "frame_paths": ["/tmp/frame-001.jpg"],
            "question": "What happens next?",
        }
    }

    assert audit_label_free_payload(payload) is None


def test_run_manifest_hash_changes_when_evidence_policy_changes() -> None:
    first = _manifest(evidence_policy_sha256="a" * 64)
    second = _manifest(evidence_policy_sha256="b" * 64)

    assert first.sha256() != second.sha256()


def test_contract_dataclasses_are_frozen_and_serialize_tuples_as_lists() -> None:
    span = TimelineSpan(
        span_id="span-1",
        start_seconds=1.0,
        end_seconds=2.0,
        modalities=("visual", "audio"),
        description="A bounded event",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        span.description = "mutated"  # type: ignore[misc]

    assert span.to_dict() == {
        "span_id": "span-1",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "modalities": ["visual", "audio"],
        "description": "A bounded event",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("model_id", "other-model"),
        ("panel_sha256", "not-a-hash"),
        ("pass_ids", PASS_IDS[:-1]),
    ],
)
def test_run_manifest_rejects_invalid_contract_identity(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_manifest(), **{field: value})


def test_canonical_sha256_uses_sorted_utf8_json_without_whitespace() -> None:
    value = {"z": "café", "a": [1, "雪"]}
    expected = hashlib.sha256('{"a":[1,"雪"],"z":"café"}'.encode("utf-8")).hexdigest()

    assert canonical_sha256(value) == expected


def test_deduplication_keeps_first_candidate_identity() -> None:
    candidates = [
        _candidate("direct-av", "Four times"),
        _candidate("global-transcript", "four times."),
    ]

    assert [row.pass_id for row in deduplicate_candidates(candidates)] == ["direct-av"]


def test_deduplication_does_not_collapse_internal_whitespace() -> None:
    candidates = [
        _candidate("direct-av", "four  times"),
        _candidate("global-transcript", "four times"),
    ]

    assert [row.pass_id for row in deduplicate_candidates(candidates)] == [
        "direct-av",
        "global-transcript",
    ]


@pytest.mark.parametrize("elapsed_seconds", [-1.0, math.nan, math.inf, -math.inf])
def test_candidate_rejects_invalid_elapsed_seconds(elapsed_seconds: float) -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        _candidate("direct-av", elapsed_seconds=elapsed_seconds)


@pytest.mark.parametrize(
    "runtime_provenance",
    [
        {"raw_output": "Four times"},
        {"transcript": "spoken words"},
        {"media_path": "/private/video.mp4"},
        {"private_url": "https://private.example/run"},
    ],
)
def test_candidate_rejects_unsafe_runtime_provenance(
    runtime_provenance: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="runtime_provenance"):
        dataclasses.replace(
            _candidate("direct-av"), runtime_provenance=runtime_provenance
        )


def test_verifier_must_rank_every_unique_candidate_once() -> None:
    with pytest.raises(ValueError, match="complete permutation"):
        VerifierDecision(("direct-av",), {})


def test_verifier_rejects_duplicate_ranked_pass_ids() -> None:
    duplicate_rank = PASS_IDS[:-1] + (PASS_IDS[-2],)

    with pytest.raises(ValueError, match="complete permutation"):
        VerifierDecision(duplicate_rank, {})


def test_verifier_serializes_complete_ranking_and_consistent_support() -> None:
    decision = _decision()

    assert decision.to_dict() == {
        "ranked_pass_ids": list(PASS_IDS),
        "support_by_pass": {"direct-av": "span-1"},
        "selected_pass_id": "direct-av",
        "status_by_pass": {pass_id: "supported" for pass_id in PASS_IDS},
    }


def test_verifier_ranking_returns_candidates_in_decision_order() -> None:
    candidates = _ten_candidates()
    decision = _decision(tuple(reversed(PASS_IDS)))

    ranked = apply_verifier_ranking(candidates, decision)

    assert tuple(row.pass_id for row in ranked) == tuple(reversed(PASS_IDS))


def test_missing_verifier_uses_declared_fallback_precedence() -> None:
    candidates = _ten_candidates()

    ranked = apply_verifier_ranking(candidates, None)

    assert tuple(row.pass_id for row in ranked) == (
        "global-local-timeline",
        "direct-av",
        "question-specialist",
        "detail-temporal-route",
        "global-transcript",
        "local-only",
        "scene-boundary",
        "audio-first",
        "direct-av-alternate",
        "conservative-synthesis",
    )


def test_ten_candidate_prediction_requires_complete_pass_permutation() -> None:
    with pytest.raises(ValueError, match="exactly ten"):
        RankedPrediction(
            query_id=7,
            video_id="video-1",
            candidates=_ten_candidates()[:-1],
            ranked_pass_ids=PASS_IDS[:-1],
            ranking_source="verifier",
            fallback_status=None,
        )


def test_tampering_after_freeze_is_detected(tmp_path: Path) -> None:
    predictions = tmp_path / "ranked-predictions.json"
    predictions.write_text(json.dumps(_ranked_payload()), encoding="utf-8")

    freeze = freeze_predictions(predictions, bindings={"run_id": "r1", "row_count": 1})
    predictions.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_prediction_freeze(predictions, freeze)


def test_freeze_delegates_to_json_artifact_and_binds_payload(tmp_path: Path) -> None:
    predictions = tmp_path / "ranked-predictions.json"
    predictions.write_text(json.dumps(_ranked_payload()), encoding="utf-8")

    freeze = freeze_predictions(predictions, bindings={"run_id": "r1", "row_count": 1})

    assert freeze["status"] == "frozen"
    verify_prediction_freeze(predictions, freeze)


@pytest.mark.parametrize("cutoff", [1, 5, 10])
def test_ag_serialization_uses_official_json_schema(cutoff: int) -> None:
    payload = serialize_ag_submission([_ranked_prediction()], cutoff=cutoff)

    assert payload[0]["Q_ID"] == 7
    assert payload[0]["Video_ID"] == "video-1"
    assert len(payload[0]["Answers"]) == cutoff
    assert list(payload[0]["Answers"][0]) == ["Rank", "Answer", "Time"]
    assert payload[0]["Answers"][0]["Rank"] == 1


@pytest.mark.parametrize("cutoff", [0, 2, 11])
def test_ag_serialization_rejects_non_official_cutoffs(cutoff: int) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        serialize_ag_submission([_ranked_prediction()], cutoff=cutoff)


def test_ag_serialization_rejects_duplicate_answers() -> None:
    duplicate = dataclasses.replace(
        _ranked_prediction(),
        candidates=(_candidate(PASS_IDS[0], "answer 1"),)
        + tuple(_ten_candidates()[1:]),
    )
    with pytest.raises(ValueError, match="duplicate"):
        serialize_ag_submission([duplicate], cutoff=10)
