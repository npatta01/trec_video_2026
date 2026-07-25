from pathlib import Path

import pytest

from trec_video_2026.vqa_whisper import (
    format_timestamp,
    load_wav_audio,
    transcribe_audio,
    transcribe_multilingual_audio,
    transcript_quality,
    validate_rocm_bfloat16,
)


class FakePipeline:
    def __call__(self, audio_path: str, **kwargs):
        self.audio_path = audio_path
        self.kwargs = kwargs
        return {
            "text": "The first line. The second line.",
            "chunks": [
                {
                    "timestamp": (0.0, 1.2),
                    "text": " The first line.",
                    "language": "english",
                    "no_speech_probability": 0.02,
                },
                {
                    "timestamp": (1.2, 3.4),
                    "text": " The second line.",
                    "language": "english",
                    "no_speech_probability": 0.03,
                },
            ],
        }


def test_format_timestamp_is_stable() -> None:
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(61.2345) == "00:01:01.234"


def test_transcribe_audio_returns_frozen_timestamped_text(tmp_path: Path) -> None:
    pipeline = FakePipeline()
    audio = tmp_path / "clip.wav"

    result = transcribe_audio(pipeline, audio)

    assert result["text"] == "The first line. The second line."
    assert result["timestamped_text"] == (
        "[00:00:00.000] The first line.\n[00:00:01.200] The second line."
    )
    assert result["language"] == "english"
    assert result["no_speech_probability"] == 0.025
    assert result["diagnostics"] == {
        "chunk_count": 2,
        "languages": ["english"],
        "max_no_speech_probability": 0.03,
        "no_speech_probability_source": "chunk-median",
        "no_speech_metadata_available": True,
        "no_speech_probabilities": [0.02, 0.03],
    }
    assert pipeline.kwargs == {
        "return_timestamps": True,
        "return_language": True,
        "generate_kwargs": {
            "task": "translate",
            "condition_on_prev_tokens": False,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        },
    }


def test_transcribe_audio_marks_unavailable_no_speech_metadata() -> None:
    class PipelineWithoutDecoderDiagnostics:
        def __call__(self, audio, **kwargs):
            return {
                "text": "Hello",
                "chunks": [
                    {
                        "timestamp": (0.0, 1.0),
                        "text": " Hello",
                        "language": "english",
                    }
                ],
            }

    result = transcribe_audio(PipelineWithoutDecoderDiagnostics(), [0.1])

    assert result["no_speech_probability"] is None
    assert result["diagnostics"]["no_speech_probability_source"] == "unavailable"
    assert result["diagnostics"]["no_speech_metadata_available"] is False


def test_transcribe_audio_prefers_output_level_no_speech_probability() -> None:
    class OutputDiagnosticPipeline:
        def __call__(self, audio, **kwargs):
            return {
                "text": "A real spoken sentence.",
                "no_speech_probability": 0.04,
                "chunks": [
                    {
                        "timestamp": (0.0, 2.0),
                        "text": " A real spoken sentence.",
                        "language": "english",
                        "no_speech_probability": 0.98,
                    }
                ],
            }

    result = transcribe_audio(OutputDiagnosticPipeline(), [0.1])

    assert result["no_speech_probability"] == 0.04
    assert result["diagnostics"]["no_speech_probability_source"] == "output"


def test_transcribe_audio_chunk_median_resists_one_silent_segment() -> None:
    class MixedSegmentPipeline:
        def __call__(self, audio, **kwargs):
            return {
                "text": "Speech continues across the clip.",
                "chunks": [
                    {
                        "timestamp": (0.0, 2.0),
                        "text": " Speech continues",
                        "no_speech_probability": 0.02,
                    },
                    {
                        "timestamp": (2.0, 4.0),
                        "text": " across the clip.",
                        "no_speech_probability": 0.04,
                    },
                    {
                        "timestamp": (4.0, 6.0),
                        "text": "",
                        "no_speech_probability": 0.98,
                    },
                ],
            }

    result = transcribe_audio(MixedSegmentPipeline(), [0.1])
    quality = transcript_quality(
        result["text"],
        duration_seconds=6.0,
        no_speech_probability=result["no_speech_probability"],
    )

    assert result["no_speech_probability"] == 0.04
    assert result["diagnostics"]["no_speech_probability_source"] == "chunk-median"
    assert quality["status"] == "accepted"


def test_transcribe_audio_omits_blank_chunks_from_timestamped_text() -> None:
    class BlankChunkPipeline:
        def __call__(self, audio, **kwargs):
            return {
                "text": "",
                "chunks": [
                    {
                        "timestamp": (0.0, 30.0),
                        "text": "   ",
                        "language": "english",
                    }
                ],
            }

    result = transcribe_audio(BlankChunkPipeline(), [0.0])

    assert result["timestamped_text"] == ""
    assert result["chunks"] == [
        {"start": 0.0, "end": 30.0, "text": "", "language": "english"}
    ]


def test_transcribe_multilingual_audio_retains_original_and_english_translation() -> (
    None
):
    class MultilingualPipeline:
        def __call__(self, audio, **kwargs):
            task = kwargs["generate_kwargs"]["task"]
            if task == "transcribe":
                return {
                    "text": "नमस्ते",
                    "chunks": [
                        {
                            "timestamp": (0.0, 1.0),
                            "text": " नमस्ते",
                            "language": "hindi",
                            "no_speech_probability": 0.01,
                        }
                    ],
                }
            return {
                "text": "Hello",
                "chunks": [
                    {
                        "timestamp": (0.0, 1.0),
                        "text": " Hello",
                        "language": "hindi",
                        "no_speech_probability": 0.01,
                    }
                ],
            }

    result = transcribe_multilingual_audio(MultilingualPipeline(), [0.1, -0.1])

    assert result["original"]["text"] == "नमस्ते"
    assert result["translation"]["text"] == "Hello"
    assert result["source_language"] == "hindi"


def test_load_wav_audio_requires_the_frozen_16khz_mono_format(tmp_path: Path) -> None:
    class FakeSoundFile:
        @staticmethod
        def read(path, dtype):
            assert path == str(tmp_path / "clip.wav")
            assert dtype == "float32"
            return [0.1, -0.1], 16_000

    assert load_wav_audio(tmp_path / "clip.wav", FakeSoundFile) == [0.1, -0.1]


def test_transcript_quality_rejects_impossible_repetition_rate() -> None:
    repeated = " ".join(["They are living in the living room."] * 30)

    quality = transcript_quality(repeated, duration_seconds=14.0)

    assert quality["status"] == "rejected-hallucination"
    assert quality["words_per_second"] > 5


def test_transcript_quality_accepts_short_dialogue() -> None:
    quality = transcript_quality(
        "Where is the bag and what is inside it?", duration_seconds=19.0
    )

    assert quality["status"] == "accepted"


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_transcript_quality_rejects_empty_translation(text: str) -> None:
    quality = transcript_quality(
        text,
        duration_seconds=30.011813,
        mode="translate",
    )

    assert quality["status"] == "rejected-empty"
    assert quality["word_count"] == 0


def test_transcript_quality_rejects_timestamp_overrun() -> None:
    quality = transcript_quality(
        "A short ordinary sentence.",
        duration_seconds=30.011813,
        chunks=[
            {
                "start": 30.0,
                "end": 59.98,
                "text": "A short ordinary sentence.",
            }
        ],
    )

    assert quality["status"] == "rejected-timestamp-overrun"
    assert quality["max_timestamp_seconds"] == 59.98


def test_transcript_quality_rejects_repeated_phrase_at_plausible_word_rate() -> None:
    phrase = "A person walks through the open doorway"
    repeated = " ".join([phrase] * 5)

    quality = transcript_quality(repeated, duration_seconds=12.0)

    assert quality["status"] == "rejected-repetition"
    assert quality["repeated_window_count"] >= 4


def test_transcript_quality_rejects_known_whisper_boilerplate_once() -> None:
    quality = transcript_quality(
        "Subtitles by the Amara.org community",
        duration_seconds=17.0,
    )

    assert quality["status"] == "rejected-boilerplate"


def test_transcript_quality_rejects_decoder_no_speech_signal() -> None:
    quality = transcript_quality(
        "A plausible but unsupported sentence.",
        duration_seconds=10.0,
        no_speech_probability=0.92,
    )

    assert quality["status"] == "rejected-no-speech"
    assert quality["no_speech_probability"] == 0.92


def test_transcript_quality_accepts_legitimate_short_repetition() -> None:
    quality = transcript_quality("No, no, no!", duration_seconds=3.0)

    assert quality["status"] == "accepted"
    assert quality["repeated_window_count"] == 0


def test_transcript_quality_rejects_compressible_long_output() -> None:
    repeated = " ".join(["one two three four five six"] * 12)

    quality = transcript_quality(repeated, duration_seconds=30.0)

    assert quality["status"] == "rejected-repetition"
    assert quality["compression_ratio"] > 1.35


def test_transcript_quality_reaches_external_compression_gate_without_repetition() -> (
    None
):
    text = " ".join(
        f"observation{index:02d}_" + ("abcdefghij" * 5) for index in range(30)
    )

    quality = transcript_quality(text, duration_seconds=30.0)

    assert quality["repeated_window_count"] < 4
    assert quality["compression_ratio"] > 3.0
    assert quality["compression_ratio_threshold"] == 3.0
    assert quality["decoder_segment_compression_ratio_threshold"] == 1.35
    assert quality["compression_ratio_definition"] == (
        "whole-transcript UTF-8 bytes divided by zlib-compressed bytes"
    )
    assert quality["status"] == "rejected-compression"


@pytest.mark.parametrize(
    ("text", "mode", "no_speech_probability", "expected_status"),
    [
        ("Could you put the blue bag on the table?", "translate", 0.03, "accepted"),
        ("यह नीला बैग मेज़ पर रख दो", "transcribe", 0.04, "accepted"),
        ("هذه جملة منطوقة قصيرة", "transcribe", None, "accepted"),
        ("", "transcribe", 0.95, "rejected-empty"),
    ],
)
def test_transcript_quality_calibration_cases(
    text: str,
    mode: str,
    no_speech_probability: float | None,
    expected_status: str,
) -> None:
    quality = transcript_quality(
        text,
        duration_seconds=12.0,
        mode=mode,
        no_speech_probability=no_speech_probability,
    )

    assert quality["status"] == expected_status


def test_transcript_quality_rejects_untranslated_non_latin_text() -> None:
    quality = transcript_quality(
        "पाकिस्तान की लग रही है और कट्टा कहा है",
        duration_seconds=19.0,
        mode="translate",
    )

    assert quality["status"] == "rejected-non-english-script"
    assert quality["latin_letter_ratio"] < 0.8


def test_transcript_quality_accepts_non_latin_original_transcription() -> None:
    quality = transcript_quality(
        "पाकिस्तान की लग रही है और कट्टा कहा है",
        duration_seconds=19.0,
        mode="transcribe",
    )

    assert quality["status"] == "accepted"
    assert quality["latin_letter_ratio"] < 0.8


def test_emi_replay_is_rejected_for_timestamp_overrun_and_repetition() -> None:
    repeated_phrase = (
        "Thank you very much for your time, and I look forward to working "
        "with you in the future."
    )
    chunks = [
        {
            "start": float(index * 2),
            "end": float((index + 1) * 2),
            "text": repeated_phrase,
        }
        for index in range(15)
    ]
    chunks.append(
        {
            "start": 30.0,
            "end": 59.98,
            "text": "Thank you so much for joining us today, and we'll see you next time.",
        }
    )
    text = " ".join(chunk["text"] for chunk in chunks)

    quality = transcript_quality(
        text,
        duration_seconds=30.011813,
        chunks=chunks,
    )

    assert quality["status"] == "rejected-timestamp-overrun"
    assert quality["timestamp_overrun_seconds"] == pytest.approx(29.968187)
    assert quality["repeated_window_count"] >= 4


def test_validate_rocm_bfloat16_checks_actual_model_parameter() -> None:
    class Device:
        type = "cuda"

        def __str__(self):
            return "cuda:0"

    class Parameter:
        device = Device()
        dtype = "bfloat16"

    class Model:
        def parameters(self):
            return iter([Parameter()])

    class Cuda:
        @staticmethod
        def is_available():
            return True

    class Version:
        hip = "7.2"

    class Torch:
        cuda = Cuda()
        version = Version()
        bfloat16 = "bfloat16"

    assert validate_rocm_bfloat16(Model(), Torch()) == (
        "cuda:0 (ROCm)",
        "bfloat16",
    )


def test_validate_rocm_bfloat16_rejects_actual_float16_parameter() -> None:
    class Parameter:
        device = type("Device", (), {"type": "cuda"})()
        dtype = "float16"

    class Model:
        def parameters(self):
            return iter([Parameter()])

    class Torch:
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()
        version = type("Version", (), {"hip": "7.2"})()
        bfloat16 = "bfloat16"

    with pytest.raises(RuntimeError, match="not bfloat16"):
        validate_rocm_bfloat16(Model(), Torch())
