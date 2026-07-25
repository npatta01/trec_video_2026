"""Contracts for automatic ASR and uploader-caption evidence tiers."""

from __future__ import annotations

import re
from collections.abc import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_LANGUAGE_CODE = re.compile(r"[a-z]{2,3}")
_LANGUAGE_ALIASES = {
    "english": "en",
    "french": "fr",
}
_WHISPER_MODEL_ID = "openai/whisper-large-v3"
_WHISPER_MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
_FORBIDDEN_LABEL_KEYS = {
    "correct_answer",
    "incorrect_answers",
    "reference_answer",
    "references",
}


def _reject_label_keys(value: object, location: str = "record") -> None:
    if isinstance(value, Mapping):
        leaked = _FORBIDDEN_LABEL_KEYS.intersection(value)
        if leaked:
            raise ValueError(
                f"label leakage in {location}: forbidden key {min(leaked)}"
            )
        for key, child in value.items():
            _reject_label_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_keys(child, f"{location}[{index}]")


def _required_text(mapping: Mapping, field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _validate_sha(mapping: Mapping, field: str) -> str:
    value = _required_text(mapping, field)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validated_transcript_part(record: Mapping, field: str) -> Mapping:
    part = record.get(field)
    if not isinstance(part, Mapping):
        raise ValueError(f"{field} must be an object")
    _required_text(part, "text")
    _required_text(part, "timestamped_text")
    return part


def _normalized_language(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty language")
    normalized = value.strip().casefold()
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    if not _LANGUAGE_CODE.fullmatch(normalized):
        raise ValueError(f"{field} must be a supported language name or code")
    return normalized


def validate_automatic_transcript(record: dict, media_sha256: str) -> None:
    """Raise unless a transcript is accepted, traceable automatic ASR output."""
    _reject_label_keys(record)
    if record.get("status") != "passed":
        raise ValueError("automatic transcript status must be passed")
    _required_text(record, "video_id")
    if _validate_sha(record, "source_media_sha256") != media_sha256:
        raise ValueError("source_media_sha256 does not match frozen media")
    _validate_sha(record, "source_audio_sha256")
    source_language = _normalized_language(
        record.get("source_language"), "source_language"
    )
    detection = record.get("language_detection")
    if (
        not isinstance(detection, Mapping)
        or detection.get("method") != "whisper_model_detect_language"
        or detection.get("status") != "accepted"
    ):
        raise ValueError("language_detection must be accepted Whisper detection")
    detected_code = _normalized_language(
        detection.get("detected_code"), "language_detection.detected_code"
    )
    if source_language != detected_code:
        raise ValueError("source_language must agree with detected_code")

    original = _validated_transcript_part(record, "original_language_transcript")
    original_language = _normalized_language(
        original.get("language"), "original_language_transcript.language"
    )
    if original_language != detected_code:
        raise ValueError(
            "original_language_transcript.language must agree with detected_code"
        )
    translation = _validated_transcript_part(record, "english_translation")
    if (
        _normalized_language(
            translation.get("language"), "english_translation.language"
        )
        != "en"
    ):
        raise ValueError("english_translation.language must be English")
    quality = record.get("quality")
    if not isinstance(quality, Mapping) or quality.get("status") != "accepted":
        raise ValueError("quality must be accepted")
    original_quality = record.get("original_quality")
    if not isinstance(original_quality, Mapping):
        raise ValueError("original_quality must be recorded")
    _required_text(original_quality, "status")

    if _required_text(record, "model_id") != _WHISPER_MODEL_ID:
        raise ValueError("model_id must be the pinned Whisper large-v3 model")
    revision = _required_text(record, "model_revision")
    if revision != _WHISPER_MODEL_REVISION:
        raise ValueError("model_revision must be the pinned Whisper revision")
    runtime_image = _required_text(record, "runtime_image")
    if not re.search(r"@sha256:[0-9a-f]{64}$", runtime_image):
        raise ValueError("runtime_image must include an immutable SHA-256 digest")
    _required_text(record, "device")
    _required_text(record, "precision")


def select_automatic_transcript_text(record: Mapping) -> str:
    """Select the timestamped English translation used by downstream VQA."""
    translation = record.get("english_translation")
    if not isinstance(translation, Mapping):
        raise ValueError("english_translation must be an object")
    return _required_text(translation, "timestamped_text")


def validate_human_caption_reference(record: dict, media_sha256: str) -> None:
    """Validate one uploader-subtitle row as a human-caption reference tier."""
    _reject_label_keys(record)
    if record.get("provenance_kind") != "uploader_subtitle":
        raise ValueError(
            "caption must be labeled human-caption reference, not oracle transcript"
        )
    _required_text(record, "video_id")
    if _validate_sha(record, "source_media_sha256") != media_sha256:
        raise ValueError("source_media_sha256 does not match frozen media")
    _validate_sha(record, "caption_sha256")
    _required_text(record, "language")
    _required_text(record, "text")

    coverage = record.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("status") != "verified":
        raise ValueError("caption coverage must be verified")
    try:
        duration = float(coverage["clip_duration_seconds"])
        start = float(coverage["caption_start_seconds"])
        end = float(coverage["caption_end_seconds"])
        declared_ratio = float(coverage["coverage_ratio"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("caption coverage fields must be numeric") from error
    if duration <= 0 or start < 0 or end <= start or end > duration + 0.5:
        raise ValueError("caption coverage bounds are invalid")
    actual_ratio = min(duration, end) - start
    actual_ratio /= duration
    if (
        declared_ratio < 0.9
        or actual_ratio < 0.9
        or abs(declared_ratio - actual_ratio) > 0.02
    ):
        raise ValueError("caption coverage is insufficient or inconsistent")


def validate_dataset_recording_prompt_reference(
    record: dict, media_sha256: str
) -> None:
    """Validate a controlled-corpus prompt reference without calling it oracle."""
    _reject_label_keys(record)
    if record.get("reference_kind") != "dataset_recording_prompt_reference":
        raise ValueError("reference must use the dataset recording-prompt tier")
    _required_text(record, "video_id")
    if _validate_sha(record, "source_media_sha256") != media_sha256:
        raise ValueError("source_media_sha256 does not match frozen media")
    _required_text(record, "corpus")
    corpus_url = _required_text(record, "corpus_url")
    if not corpus_url.startswith("https://"):
        raise ValueError("corpus_url must use HTTPS")
    if _required_text(record, "license") != "CC BY 4.0":
        raise ValueError("dataset reference must declare CC BY 4.0")
    utterance_code = _required_text(record, "utterance_code")
    if not re.fullmatch(r"[a-z0-9]{6}", utterance_code):
        raise ValueError("utterance_code must be a six-character GRID code")
    _required_text(record, "text")
    if record.get("recording_status") != "non_wrong":
        raise ValueError("dataset reference must exclude _WRONG recordings")
