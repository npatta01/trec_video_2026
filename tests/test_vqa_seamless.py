import pytest

from trec_video_2026.vqa_seamless import (
    decode_translation,
    prepare_audio_inputs,
    request_identity,
    translate_audio,
)


class FakeInputs(dict):
    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


class FakeProcessor:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeInputs(input_features="features")

    def decode(self, *args, **kwargs):
        self.decode_args = args
        self.decode_kwargs = kwargs
        return "Where is the bag?"


class FakeModel:
    generate_kwargs = None

    def generate(self, **kwargs):
        type(self).generate_kwargs = {
            key: value for key, value in kwargs.items() if key == "tgt_lang"
        }
        return [[11, 12, 3]]


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_prepare_audio_inputs_uses_frozen_sample_rate_and_device() -> None:
    processor = FakeProcessor()

    prepared = prepare_audio_inputs(
        processor, [0.1, -0.1], device="cuda:0", dtype="float16"
    )

    assert processor.kwargs == {
        "audio": [0.1, -0.1],
        "sampling_rate": 16_000,
        "return_tensors": "pt",
    }
    assert prepared.to_kwargs == {"device": "cuda:0", "dtype": "float16"}


def test_decode_translation_decodes_first_sequence() -> None:
    processor = FakeProcessor()

    assert decode_translation(processor, [[11, 12, 3]]) == "Where is the bag?"
    assert processor.decode_args == ([11, 12, 3],)
    assert processor.decode_kwargs == {"skip_special_tokens": True}


def test_decode_translation_rejects_empty_text() -> None:
    class EmptyProcessor(FakeProcessor):
        def decode(self, *args, **kwargs):
            return "  "

    with pytest.raises(
        ValueError, match="SeamlessM4T returned an empty translation"
    ):
        decode_translation(EmptyProcessor(), [[3]])


def test_translate_audio_targets_english_without_sampling() -> None:
    result = translate_audio(
        FakeModel(),
        FakeProcessor(),
        [0.1],
        device="cuda:0",
        dtype="float16",
        inference_mode=FakeInferenceMode,
    )

    assert result == "Where is the bag?"
    assert FakeModel.generate_kwargs == {"tgt_lang": "eng"}


def test_request_identity_copies_only_protocol_identity() -> None:
    request = {
        "run_id": "run-1",
        "condition_id": "seamless-small",
        "video_id": "video-1",
        "audio_path": "/tmp/video-1.wav",
        "unexpected": "ignored",
    }

    assert request_identity(request) == {
        "run_id": "run-1",
        "condition_id": "seamless-small",
        "video_id": "video-1",
    }
