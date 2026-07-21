from pathlib import Path

from trec_video_2026.vqa_whisper import (
    format_timestamp,
    load_wav_audio,
    transcript_quality,
    transcribe_audio,
)


class FakePipeline:
    def __call__(self, audio_path: str, **kwargs):
        self.audio_path = audio_path
        self.kwargs = kwargs
        return {
            "text": "The first line. The second line.",
            "chunks": [
                {"timestamp": (0.0, 1.2), "text": " The first line."},
                {"timestamp": (1.2, 3.4), "text": " The second line."},
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
        "[00:00:00.000] The first line.\n"
        "[00:00:01.200] The second line."
    )
    assert pipeline.kwargs == {
        "return_timestamps": True,
        "generate_kwargs": {"task": "translate"},
    }


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


def test_transcript_quality_rejects_untranslated_non_latin_text() -> None:
    quality = transcript_quality(
        "पाकिस्तान की लग रही है और कट्टा कहा है", duration_seconds=19.0
    )

    assert quality["status"] == "rejected-non-english-script"
    assert quality["latin_letter_ratio"] < 0.8
