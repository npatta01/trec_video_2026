import copy
import hashlib

import pytest

from trec_video_2026.vqa_experiments.build_conditions import (
    build_matched_conditions,
    freeze_predictions,
    group_by_video,
)

MEDIA_SHA = "a" * 64


@pytest.fixture
def inputs() -> tuple[dict, dict, dict, dict]:
    manifest = {
        "panel_id": "oracle-speech-10",
        "label_access": "forbidden",
        "examples": [
            {
                "video_id": "video-01",
                "question": "What did the speaker say?",
                "media_path": "videos/video-01.mkv",
                "media_sha256": MEDIA_SHA,
                "language_guess": "fr",
                "operational_tag": "speech_content",
            }
        ],
    }
    frames = {
        "video-01": [
            {
                "path": f"frames/video-01-{index:02d}.jpg",
                "sha256": f"{index:064x}",
            }
            for index in range(12)
        ]
    }
    automatic = {
        "video-01": {
            "model_id": "openai/whisper-large-v3",
            "text": "automatic transcript",
        }
    }
    oracle = {
        "video-01": {
            "video_id": "video-01",
            "source_language": "fr",
            "source_text": "Texte vérifié.",
            "english_translation": "Verified transcript.",
            "segments": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "source_text": "Texte vérifié.",
                    "english_translation": "Verified transcript.",
                }
            ],
            "clip_duration_seconds": 2.0,
            "source_audio_sha256": MEDIA_SHA,
            "transcriber": "human-1",
            "reviewer": "human-2",
            "transcription_scope": "verbatim_full_clip",
            "label_access": "none",
            "review_status": "accepted",
        }
    }
    return manifest, frames, automatic, oracle


def test_conditions_differ_only_by_transcript(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    requests = build_matched_conditions(*inputs)
    grouped = group_by_video(requests)

    for rows in grouped.values():
        assert {row["condition"] for row in rows} == {
            "frames_only",
            "frames_automatic_transcript",
            "frames_oracle_transcript",
        }
        assert len({tuple(row["frame_sha256s"]) for row in rows}) == 1
        assert len({row["prompt_sha256"] for row in rows}) == 1
        assert all(
            row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
            for row in rows
        )
        assert len({row["model_id"] for row in rows}) == 1
        assert len({row["seed"] for row in rows}) == 1
        assert len({tuple(sorted(row["decoding"].items())) for row in rows}) == 1


def test_conditions_bind_request_hashes(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    requests = build_matched_conditions(*inputs)

    assert len(requests) == 3
    assert len({row["request_sha256"] for row in requests}) == 3
    assert all(
        len(row["request_sha256"]) == 64
        and row["request_sha256"]
        == hashlib.sha256(row["canonical_request"].encode()).hexdigest()
        for row in requests
    )


def test_conditions_reject_unreviewed_oracle(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, oracle = copy.deepcopy(inputs)
    oracle["video-01"]["review_status"] = "unreviewed"

    with pytest.raises(ValueError, match="independent review"):
        build_matched_conditions(manifest, frames, automatic, oracle)


def test_conditions_reject_label_leakage(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, oracle = copy.deepcopy(inputs)
    manifest["examples"][0]["correct_answer"] = "leaked"

    with pytest.raises(ValueError, match="label"):
        build_matched_conditions(manifest, frames, automatic, oracle)


def test_conditions_require_twelve_frames(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, oracle = copy.deepcopy(inputs)
    frames["video-01"].pop()

    with pytest.raises(ValueError, match="12 frames"):
        build_matched_conditions(manifest, frames, automatic, oracle)


def test_prediction_freeze_requires_complete_matched_rows(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    requests = build_matched_conditions(*inputs)
    predictions = [
        {
            "video_id": request["video_id"],
            "condition": request["condition"],
            "request_sha256": request["request_sha256"],
            "status": "success",
            "output": {"answer": "an answer", "evidence_quote": "support"},
        }
        for request in requests
    ]

    freeze = freeze_predictions(requests, predictions)

    assert freeze["status"] == "frozen"
    assert freeze["row_count"] == 3
    assert len(freeze["requests_sha256"]) == 64
    assert len(freeze["predictions_sha256"]) == 64


def test_prediction_freeze_rejects_reference_labels(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    requests = build_matched_conditions(*inputs)
    predictions = [
        {
            "video_id": request["video_id"],
            "condition": request["condition"],
            "request_sha256": request["request_sha256"],
            "status": "success",
            "output": {"answer": "an answer", "evidence_quote": "support"},
        }
        for request in requests
    ]
    predictions[0]["correct_answer"] = "leaked label"

    with pytest.raises(ValueError, match="label"):
        freeze_predictions(requests, predictions)
