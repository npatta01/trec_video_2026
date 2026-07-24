import copy
import hashlib

import pytest

from trec_video_2026.vqa_experiments.build_conditions import (
    build_automatic_ablation,
    build_dataset_reference_conditions,
    build_human_caption_reference_conditions,
    build_matched_conditions,
    freeze_predictions,
    group_by_video,
    normalize_runner_predictions,
    verify_prediction_freeze,
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
                "timestamp_seconds": index + 0.5,
            }
            for index in range(12)
        ]
    }
    automatic = {
        "video-01": {
            "status": "passed",
            "video_id": "video-01",
            "source_media_sha256": MEDIA_SHA,
            "source_audio_sha256": "b" * 64,
            "source_language": "French",
            "language_detection": {
                "method": "whisper_model_detect_language",
                "detected_code": "fr",
                "metadata_code": "fr-FR",
                "status": "accepted",
            },
            "original_language_transcript": {
                "text": "transcription automatique",
                "timestamped_text": ("[00:00:00.200] transcription automatique"),
                "language": "French",
            },
            "english_translation": {
                "text": "automatic transcript",
                "timestamped_text": "[00:00:00.200] automatic transcript",
                "language": "English",
            },
            "quality": {"status": "accepted"},
            "original_quality": {"status": "accepted"},
            "model_id": "openai/whisper-large-v3",
            "model_revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
            "runtime_image": (
                "localhost/trec-vqa-audio-rocm"
                "@sha256:"
                "faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
            ),
            "device": "cuda:0 (ROCm)",
            "precision": "bfloat16",
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
        assert len({tuple(row["frame_timestamps"]) for row in rows}) == 1
        assert len({row["prompt_sha256"] for row in rows}) == 1
        assert all(
            row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
            for row in rows
        )
        assert len({row["model_id"] for row in rows}) == 1
        assert len({row["seed"] for row in rows}) == 1
        assert len({tuple(sorted(row["decoding"].items())) for row in rows}) == 1


def test_automatic_ablation_builds_two_matched_conditions(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs

    requests = build_automatic_ablation(manifest, frames, automatic)
    grouped = group_by_video(requests)

    assert len(requests) == 2
    assert {row["condition"] for row in grouped["video-01"]} == {
        "frames_only",
        "frames_automatic_transcript",
    }
    assert {row["question"] for row in requests} == {"What did the speaker say?"}
    assert len({tuple(row["frame_sha256s"]) for row in requests}) == 1
    assert len({tuple(row["frame_timestamps"]) for row in requests}) == 1
    assert requests[0]["transcript"] is None
    assert requests[1]["transcript"] == ("[00:00:00.200] automatic transcript")


def test_dataset_reference_builds_distinct_non_oracle_condition(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    reference = {
        "video-01": {
            "video_id": "video-01",
            "source_media_sha256": MEDIA_SHA,
            "reference_kind": "dataset_recording_prompt_reference",
            "corpus": "Audio-Visual Lombard GRID Speech corpus",
            "corpus_url": "https://spandh.dcs.shef.ac.uk/avlombard/",
            "license": "CC BY 4.0",
            "utterance_code": "sgbe5s",
            "text": "set green by e 5 soon",
            "recording_status": "non_wrong",
        }
    }

    requests = build_dataset_reference_conditions(
        manifest, frames, automatic, reference
    )

    assert {request["condition"] for request in requests} == {
        "frames_only",
        "frames_automatic_transcript",
        "frames_dataset_reference",
    }
    assert requests[-1]["transcript"] == "set green by e 5 soon"


def test_human_caption_builds_distinct_non_oracle_condition(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    caption = {
        "video-01": {
            "video_id": "video-01",
            "provenance_kind": "uploader_subtitle",
            "source_media_sha256": MEDIA_SHA,
            "caption_sha256": "c" * 64,
            "language": "en",
            "text": "verified full-clip uploader caption",
            "coverage": {
                "clip_duration_seconds": 10.0,
                "caption_start_seconds": 0.0,
                "caption_end_seconds": 9.8,
                "coverage_ratio": 0.98,
                "status": "verified",
            },
        }
    }

    requests = build_human_caption_reference_conditions(
        manifest, frames, automatic, caption
    )

    assert {request["condition"] for request in requests} == {
        "frames_only",
        "frames_automatic_transcript",
        "frames_human_caption_reference",
    }
    assert requests[-1]["transcript"] == "verified full-clip uploader caption"


def test_prediction_freeze_accepts_complete_two_condition_ablation(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    requests = build_automatic_ablation(manifest, frames, automatic)
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
    assert freeze["row_count"] == 2


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


def test_normalize_runner_predictions_binds_provenance_and_request_hash(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    requests = build_automatic_ablation(manifest, frames, automatic)
    raw_rows = [
        {
            "run_id": request["run_id"],
            "video_id": request["video_id"],
            "condition_id": request["condition_id"],
            "request_sha256": request["request_sha256"],
            "status": "passed",
            "answer": "model answer",
            "model_id": request["model_id"],
            "model_revision": request["model_revision"],
            "runtime_image": request["runtime_image"],
            "device": "cuda:0 (ROCm)",
            "precision": "bfloat16",
        }
        for request in requests
    ]

    predictions = normalize_runner_predictions(requests, raw_rows)

    assert len(predictions) == 2
    assert predictions[0]["request_sha256"] == requests[0]["request_sha256"]
    assert predictions[0]["condition"] == requests[0]["condition"]
    assert predictions[0]["status"] == "success"
    assert predictions[0]["output"] == {
        "answer": "model answer",
        "evidence_quote": "",
    }


def test_normalize_runner_predictions_rejects_wrong_runtime(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    requests = build_automatic_ablation(manifest, frames, automatic)
    request = requests[0]
    raw_row = {
        "run_id": request["run_id"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": request["request_sha256"],
        "status": "passed",
        "answer": "model answer",
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "runtime_image": "mutable:latest",
        "device": "cuda:0 (ROCm)",
        "precision": "bfloat16",
    }

    with pytest.raises(ValueError, match="runtime_image"):
        normalize_runner_predictions([request], [raw_row])


@pytest.mark.parametrize("tamper", ("prompt", "frame_paths", "transcript"))
def test_normalize_runner_predictions_rejects_tampered_request_payload(
    inputs: tuple[dict, dict, dict, dict],
    tamper: str,
) -> None:
    manifest, frames, automatic, _ = inputs
    request = copy.deepcopy(build_automatic_ablation(manifest, frames, automatic)[1])
    if tamper == "prompt":
        request["prompt"] = "changed after request hashing"
    elif tamper == "frame_paths":
        request["frame_paths"].reverse()
    else:
        request["transcript"] = "changed after request hashing"
    raw_row = {
        "run_id": request["run_id"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "request_sha256": request["request_sha256"],
        "status": "passed",
        "answer": "model answer",
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "runtime_image": request["runtime_image"],
        "device": "cuda:0 (ROCm)",
        "precision": "bfloat16",
    }

    with pytest.raises(ValueError, match="canonical_request"):
        normalize_runner_predictions([request], [raw_row])


def test_normalize_runner_predictions_requires_echoed_request_hash(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    request = build_automatic_ablation(manifest, frames, automatic)[0]
    raw_row = {
        "run_id": request["run_id"],
        "video_id": request["video_id"],
        "condition_id": request["condition_id"],
        "status": "passed",
        "answer": "model answer",
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "runtime_image": request["runtime_image"],
        "device": "cuda:0 (ROCm)",
        "precision": "bfloat16",
    }

    with pytest.raises(ValueError, match="request_sha256"):
        normalize_runner_predictions([request], [raw_row])


def test_verify_prediction_freeze_rejects_post_freeze_mutation(
    inputs: tuple[dict, dict, dict, dict],
) -> None:
    manifest, frames, automatic, _ = inputs
    requests = build_automatic_ablation(manifest, frames, automatic)
    predictions = [
        {
            "video_id": request["video_id"],
            "condition": request["condition"],
            "request_sha256": request["request_sha256"],
            "status": "success",
            "output": {"answer": "original", "evidence_quote": ""},
        }
        for request in requests
    ]
    freeze = freeze_predictions(requests, predictions)
    predictions[0]["output"]["answer"] = "mutated after freeze"

    with pytest.raises(ValueError, match="freeze"):
        verify_prediction_freeze(requests, predictions, freeze)
