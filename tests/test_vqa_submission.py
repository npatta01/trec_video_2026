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
    canonical_sha256,
    deduplicate_candidates,
    freeze_predictions,
    serialize_ag_submission,
    verify_prediction_freeze,
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
        _candidate(pass_id, f"answer {index}")
        for index, pass_id in enumerate(PASS_IDS)
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


def test_label_free_audit_rejects_nested_reference_paths() -> None:
    payload = {
        "request": {
            "media_path": "/tmp/video.mp4",
            "nested": {"annotation_path": "/tmp/gold.json"},
        }
    }

    with pytest.raises(ValueError, match="annotation"):
        audit_label_free_payload(payload)


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
def test_run_manifest_rejects_invalid_contract_identity(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(_manifest(), **{field: value})


def test_canonical_sha256_uses_sorted_utf8_json_without_whitespace() -> None:
    value = {"z": "café", "a": [1, "雪"]}
    expected = hashlib.sha256(
        '{"a":[1,"雪"],"z":"café"}'.encode("utf-8")
    ).hexdigest()

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
    decision = VerifierDecision(PASS_IDS, {"direct-av": "span-1"})

    assert decision.to_dict() == {
        "ranked_pass_ids": list(PASS_IDS),
        "support_by_pass": {"direct-av": "span-1"},
    }


def test_verifier_ranking_returns_candidates_in_decision_order() -> None:
    candidates = _ten_candidates()
    decision = VerifierDecision(tuple(reversed(PASS_IDS)), {})

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

    freeze = freeze_predictions(
        predictions, bindings={"run_id": "r1", "row_count": 1}
    )
    predictions.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_prediction_freeze(predictions, freeze)


def test_freeze_delegates_to_json_artifact_and_binds_payload(tmp_path: Path) -> None:
    predictions = tmp_path / "ranked-predictions.json"
    predictions.write_text(json.dumps(_ranked_payload()), encoding="utf-8")

    freeze = freeze_predictions(
        predictions, bindings={"run_id": "r1", "row_count": 1}
    )

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
