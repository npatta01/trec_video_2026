import copy

import pytest

from trec_video_2026.vqa_experiments.transcript_contract import (
    validate_oracle_transcript,
)

MEDIA_SHA = "a" * 64


@pytest.fixture
def valid_record() -> dict:
    return {
        "video_id": "example-video",
        "source_language": "fr",
        "source_text": "Je ne sais pas.",
        "english_translation": "I don't know.",
        "segments": [
            {
                "start_seconds": 0.2,
                "end_seconds": 1.8,
                "source_text": "Je ne sais pas.",
                "english_translation": "I don't know.",
            }
        ],
        "clip_duration_seconds": 2.0,
        "source_audio_sha256": MEDIA_SHA,
        "transcriber": "human-transcriber-1",
        "reviewer": "human-reviewer-2",
        "transcription_scope": "verbatim_full_clip",
        "label_access": "none",
        "review_status": "accepted",
    }


def test_oracle_transcript_requires_source_and_translation(
    valid_record: dict,
) -> None:
    record = valid_record | {"source_text": "", "english_translation": ""}

    with pytest.raises(ValueError, match="transcript text"):
        validate_oracle_transcript(record, MEDIA_SHA)


def test_oracle_transcript_is_bound_to_audio(valid_record: dict) -> None:
    with pytest.raises(ValueError, match="source_audio_sha256"):
        validate_oracle_transcript(valid_record, "0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("transcription_scope", "answer_only", "answer-only"),
        ("label_access", "question_and_answer", "label-blind"),
        ("review_status", "unreviewed", "independent review"),
    ],
)
def test_oracle_transcript_requires_label_blind_full_clip_review(
    valid_record: dict,
    field: str,
    value: str,
    message: str,
) -> None:
    record = valid_record | {field: value}

    with pytest.raises(ValueError, match=message):
        validate_oracle_transcript(record, MEDIA_SHA)


def test_oracle_transcript_requires_independent_people(valid_record: dict) -> None:
    record = valid_record | {"reviewer": valid_record["transcriber"]}

    with pytest.raises(ValueError, match="distinct"):
        validate_oracle_transcript(record, MEDIA_SHA)


@pytest.mark.parametrize("field", ["video_id", "source_language"])
def test_oracle_transcript_requires_identity_metadata(
    valid_record: dict,
    field: str,
) -> None:
    record = valid_record | {field: ""}

    with pytest.raises(ValueError, match=field):
        validate_oracle_transcript(record, MEDIA_SHA)


def test_oracle_transcript_rejects_timestamps_outside_clip(
    valid_record: dict,
) -> None:
    record = copy.deepcopy(valid_record)
    record["segments"][0]["end_seconds"] = 2.1

    with pytest.raises(ValueError, match="clip duration"):
        validate_oracle_transcript(record, MEDIA_SHA)


def test_oracle_transcript_preserves_segment_uncertainty(
    valid_record: dict,
) -> None:
    record = copy.deepcopy(valid_record)
    record["source_text"] = "Je ne sais pas."
    record["english_translation"] = "The horse is five years old."
    record["segments"][0]["english_translation"] = "I don't know."

    with pytest.raises(ValueError, match="translation.*segments"):
        validate_oracle_transcript(record, MEDIA_SHA)
