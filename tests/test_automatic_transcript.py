import copy

import pytest

from trec_video_2026.vqa_experiments.automatic_transcript import (
    select_automatic_transcript_text,
    validate_automatic_transcript,
    validate_dataset_recording_prompt_reference,
    validate_human_caption_reference,
)


MEDIA_SHA = "a" * 64
AUDIO_SHA = "b" * 64


@pytest.fixture
def valid_record() -> dict:
    return {
        "status": "passed",
        "video_id": "video-01",
        "source_media_sha256": MEDIA_SHA,
        "source_audio_sha256": AUDIO_SHA,
        "source_language": "French",
        "language_detection": {
            "method": "whisper_model_detect_language",
            "detected_code": "fr",
            "metadata_code": "fr-FR",
            "status": "accepted",
        },
        "original_language_transcript": {
            "text": "Je ne sais pas.",
            "timestamped_text": "[00:00:00.200] Je ne sais pas.",
            "language": "French",
        },
        "english_translation": {
            "text": "I don't know.",
            "timestamped_text": "[00:00:00.200] I don't know.",
            "language": "English",
        },
        "quality": {"status": "accepted"},
        "original_quality": {"status": "accepted"},
        "model_id": "openai/whisper-large-v3",
        "model_revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "runtime_image": (
            "localhost/trec-vqa-audio-rocm"
            "@sha256:faa70fc2896401f863fc1be1a83581585bba3bd9784cf4cf3b672d7815b6c234"
        ),
        "device": "cuda:0 (ROCm)",
        "precision": "bfloat16",
    }


def test_automatic_transcript_requires_detected_language_and_quality(
    valid_record: dict,
) -> None:
    for field, value, message in (
        ("source_language", None, "source_language"),
        ("quality", {"status": "rejected-empty"}, "quality"),
    ):
        record = valid_record | {field: value}
        with pytest.raises(ValueError, match=message):
            validate_automatic_transcript(record, MEDIA_SHA)


def test_automatic_transcript_retains_nonselected_original_quality(
    valid_record: dict,
) -> None:
    record = valid_record | {
        "original_quality": {
            "status": "rejected-timestamp-overrun",
            "timestamp_overrun_seconds": 0.8,
        }
    }

    validate_automatic_transcript(record, MEDIA_SHA)


def test_automatic_transcript_requires_accepted_language_detection(
    valid_record: dict,
) -> None:
    missing = valid_record.copy()
    missing.pop("language_detection")
    with pytest.raises(ValueError, match="language_detection"):
        validate_automatic_transcript(missing, MEDIA_SHA)

    mismatched = valid_record | {
        "language_detection": {
            "method": "whisper_model_detect_language",
            "detected_code": "en",
            "metadata_code": "fr-FR",
            "status": "rejected-metadata-mismatch",
        }
    }
    with pytest.raises(ValueError, match="language_detection"):
        validate_automatic_transcript(mismatched, MEDIA_SHA)


def test_automatic_transcript_pins_model_and_language_provenance(
    valid_record: dict,
) -> None:
    wrong_model = valid_record | {"model_id": "unreviewed/asr"}
    with pytest.raises(ValueError, match="model"):
        validate_automatic_transcript(wrong_model, MEDIA_SHA)

    wrong_source_language = valid_record | {"source_language": "English"}
    with pytest.raises(ValueError, match="source_language"):
        validate_automatic_transcript(wrong_source_language, MEDIA_SHA)

    wrong_original = copy.deepcopy(valid_record)
    wrong_original["original_language_transcript"]["language"] = "English"
    with pytest.raises(ValueError, match="original_language_transcript.language"):
        validate_automatic_transcript(wrong_original, MEDIA_SHA)


def test_automatic_transcript_is_bound_to_media_and_audio(
    valid_record: dict,
) -> None:
    with pytest.raises(ValueError, match="source_media_sha256"):
        validate_automatic_transcript(valid_record, "0" * 64)

    record = valid_record | {"source_audio_sha256": "not-a-sha"}
    with pytest.raises(ValueError, match="source_audio_sha256"):
        validate_automatic_transcript(record, MEDIA_SHA)


def test_automatic_transcript_rejects_label_fields(valid_record: dict) -> None:
    record = valid_record | {"correct_answer": "leaked"}

    with pytest.raises(ValueError, match="label"):
        validate_automatic_transcript(record, MEDIA_SHA)


def test_automatic_transcript_selects_timestamped_english_translation(
    valid_record: dict,
) -> None:
    validate_automatic_transcript(valid_record, MEDIA_SHA)

    assert select_automatic_transcript_text(valid_record) == (
        "[00:00:00.200] I don't know."
    )


def test_uploader_caption_requires_verified_coverage(valid_record: dict) -> None:
    caption = {
        "video_id": "video-01",
        "provenance_kind": "uploader_subtitle",
        "source_media_sha256": MEDIA_SHA,
        "caption_sha256": "c" * 64,
        "language": "en",
        "text": "I don't know.",
        "coverage": {
            "clip_duration_seconds": 10.0,
            "caption_start_seconds": 0.0,
            "caption_end_seconds": 9.5,
            "coverage_ratio": 0.95,
            "status": "verified",
        },
    }

    validate_human_caption_reference(caption, MEDIA_SHA)

    insufficient = copy.deepcopy(caption)
    insufficient["coverage"]["coverage_ratio"] = 0.4
    with pytest.raises(ValueError, match="coverage"):
        validate_human_caption_reference(insufficient, MEDIA_SHA)


def test_uploader_caption_is_not_an_oracle_tier(valid_record: dict) -> None:
    caption = {
        "video_id": "video-01",
        "provenance_kind": "oracle_transcript",
        "source_media_sha256": MEDIA_SHA,
        "caption_sha256": "c" * 64,
        "language": "en",
        "text": "I don't know.",
        "coverage": {
            "clip_duration_seconds": 10.0,
            "caption_start_seconds": 0.0,
            "caption_end_seconds": 10.0,
            "coverage_ratio": 1.0,
            "status": "verified",
        },
    }

    with pytest.raises(ValueError, match="human-caption reference"):
        validate_human_caption_reference(caption, MEDIA_SHA)


def test_dataset_recording_prompt_reference_has_distinct_provenance() -> None:
    record = {
        "video_id": "lgrid_s22_plain",
        "source_media_sha256": MEDIA_SHA,
        "reference_kind": "dataset_recording_prompt_reference",
        "corpus": "Audio-Visual Lombard GRID Speech corpus",
        "corpus_url": "https://spandh.dcs.shef.ac.uk/avlombard/",
        "license": "CC BY 4.0",
        "utterance_code": "sgbe5s",
        "text": "set green by e 5 soon",
        "recording_status": "non_wrong",
    }

    validate_dataset_recording_prompt_reference(record, MEDIA_SHA)


def test_dataset_reference_rejects_oracle_claim() -> None:
    record = {
        "video_id": "lgrid_s22_plain",
        "source_media_sha256": MEDIA_SHA,
        "reference_kind": "human_verified_oracle",
        "corpus": "Audio-Visual Lombard GRID Speech corpus",
        "corpus_url": "https://spandh.dcs.shef.ac.uk/avlombard/",
        "license": "CC BY 4.0",
        "utterance_code": "sgbe5s",
        "text": "set green by e 5 soon",
        "recording_status": "non_wrong",
    }

    with pytest.raises(ValueError, match="dataset recording-prompt"):
        validate_dataset_recording_prompt_reference(record, MEDIA_SHA)
