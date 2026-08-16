from __future__ import annotations

import json

import pytest

from trec_video_2026.vqa_local_gemma_canary_freeze import build_smoke_freeze


def test_fenced_smoke_outputs_freeze_as_invalid_without_repair() -> None:
    preparation = {
        "complete_sha256": "a" * 64,
        "model_id": "google/gemma-3n-E4B-it",
        "model_revision": "d" * 40,
        "rows": [
            {
                "video_id": "video-1",
                "protocol_request": {
                    "frames": [{"frame_id": "F0"}, {"frame_id": "F1"}]
                },
            }
        ],
    }
    raw_results = [
        {
            "video_id": "video-1",
            "condition_id": "primary",
            "run_id": "smoke-v1",
            "model_id": (
                "/cache/models--google--gemma-3n-E4B-it/snapshots/" + "d" * 40
            ),
            "model_revision": "d" * 40,
            "answer_policy": "current",
            "status": "passed",
            "answer": "```json\n"
            + json.dumps(
                [
                    {"frame_id": "F0", "target_state": "absent"},
                    {"frame_id": "F1", "target_state": "present"},
                ]
            )
            + "\n```",
        }
    ]

    frozen = build_smoke_freeze(
        preparation=preparation,
        raw_results=raw_results,
        request_batch_sha256="b" * 64,
        raw_results_sha256="c" * 64,
        source_revision="e" * 40,
        expected_run_id="smoke-v1",
        expected_answer_policy="current",
    )

    assert frozen["status"] == "stopped_smoke_gate"
    assert frozen["planned_rows"] == 1
    assert frozen["attempted_rows"] == 1
    assert frozen["completed_rows"] == 0
    assert frozen["rows"] == [
        {
            "video_id": "video-1",
            "condition_id": "primary",
            "status": "invalid_json",
            "failure_type": "JSONDecodeError",
        }
    ]
    assert len(frozen["complete_sha256"]) == 64


def _valid_result(video_id: str) -> dict[str, object]:
    return {
        "video_id": video_id,
        "condition_id": "primary",
        "run_id": "smoke-v1",
        "model_id": ("/cache/models--google--gemma-3n-E4B-it/snapshots/" + "d" * 40),
        "model_revision": "d" * 40,
        "answer_policy": "current",
        "status": "passed",
        "answer": json.dumps(
            [
                {"frame_id": "F0", "target_state": "absent"},
                {"frame_id": "F1", "target_state": "present"},
            ]
        ),
    }


def _two_row_preparation() -> dict[str, object]:
    return {
        "complete_sha256": "a" * 64,
        "model_id": "google/gemma-3n-E4B-it",
        "model_revision": "d" * 40,
        "rows": [
            {
                "video_id": video_id,
                "protocol_request": {
                    "frames": [{"frame_id": "F0"}, {"frame_id": "F1"}]
                },
            }
            for video_id in ("video-1", "video-2")
        ],
    }


def test_smoke_freeze_requires_one_result_for_each_planned_video() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        build_smoke_freeze(
            preparation=_two_row_preparation(),
            raw_results=[_valid_result("video-1"), _valid_result("video-1")],
            request_batch_sha256="b" * 64,
            raw_results_sha256="c" * 64,
            source_revision="e" * 40,
            expected_run_id="smoke-v1",
            expected_answer_policy="current",
        )


def test_smoke_freeze_rejects_duplicate_planned_video() -> None:
    preparation = _two_row_preparation()
    preparation["rows"][1]["video_id"] = "video-1"  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate planned video"):
        build_smoke_freeze(
            preparation=preparation,
            raw_results=[_valid_result("video-1"), _valid_result("video-2")],
            request_batch_sha256="b" * 64,
            raw_results_sha256="c" * 64,
            source_revision="e" * 40,
            expected_run_id="smoke-v1",
            expected_answer_policy="current",
        )


def test_smoke_freeze_passes_only_for_exact_valid_identity_set() -> None:
    frozen = build_smoke_freeze(
        preparation=_two_row_preparation(),
        raw_results=[_valid_result("video-2"), _valid_result("video-1")],
        request_batch_sha256="b" * 64,
        raw_results_sha256="c" * 64,
        source_revision="e" * 40,
        expected_run_id="smoke-v1",
        expected_answer_policy="current",
    )

    assert frozen["status"] == "smoke_passed"
    assert frozen["completed_rows"] == 2


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("run_id", "other-run"),
        ("model_id", "other/model"),
        ("model_revision", "f" * 40),
        ("answer_policy", "other-policy"),
    ],
)
def test_smoke_freeze_rejects_result_from_wrong_condition(
    field: str, wrong_value: str
) -> None:
    result = _valid_result("video-1")
    result[field] = wrong_value
    with pytest.raises(ValueError, match="condition provenance"):
        build_smoke_freeze(
            preparation={
                **_two_row_preparation(),
                "rows": _two_row_preparation()["rows"][:1],  # type: ignore[index]
            },
            raw_results=[result],
            request_batch_sha256="b" * 64,
            raw_results_sha256="c" * 64,
            source_revision="e" * 40,
            expected_run_id="smoke-v1",
            expected_answer_policy="current",
        )


def test_smoke_freeze_rejects_invalid_hash_or_revision_tokens() -> None:
    with pytest.raises(ValueError, match="request batch SHA-256"):
        build_smoke_freeze(
            preparation={
                **_two_row_preparation(),
                "rows": _two_row_preparation()["rows"][:1],  # type: ignore[index]
            },
            raw_results=[_valid_result("video-1")],
            request_batch_sha256="not-a-hash",
            raw_results_sha256="c" * 64,
            source_revision="e" * 40,
            expected_run_id="smoke-v1",
            expected_answer_policy="current",
        )
