from __future__ import annotations

from pathlib import Path

import pytest

import trec_video_2026.vqa_transformers_runner as transformers_runner
from trec_video_2026.vqa_transformers_runner import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    attach_request_identity,
    generate_answer,
    generation_compat_kwargs,
    load_legacy_optional_processor,
    load_processor,
    register_legacy_default_rope,
    select_auto_model_class,
)


class FakeBatch(dict):
    def to(self, device: str):
        self["moved_to"] = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        self.messages = messages
        return FakeBatch(input_ids=FakeIds(3))

    def decode(self, token_ids, **kwargs):
        assert token_ids == [40, 41]
        assert kwargs["skip_special_tokens"] is True
        return "Answer: Across a tennis net."


class FakeIds:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class FakeRow:
    def __getitem__(self, item):
        assert item == slice(3, None)
        return [40, 41]


class FakeModel:
    device = "cuda:0"

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [FakeRow()]


def test_attach_request_identity_copies_only_protocol_identity() -> None:
    result = {"status": "passed", "answer": "pink"}
    request = {
        "run_id": "followup",
        "condition_id": "crops",
        "video_id": "video-1",
        "question": "What color?",
    }

    attached = attach_request_identity(result, request)

    assert attached == {
        "status": "passed",
        "answer": "pink",
        "run_id": "followup",
        "condition_id": "crops",
        "video_id": "video-1",
    }
    assert result == {"status": "passed", "answer": "pink"}


def test_generate_answer_uses_deterministic_non_thinking_decode(tmp_path: Path) -> None:
    processor = FakeProcessor()
    model = FakeModel()

    answer = generate_answer(
        processor=processor,
        model=model,
        question="What happens?",
        frame_paths=(tmp_path / "frame.jpg",),
        frame_timestamps=(0.5,),
        frame_descriptions=("Full frame:",),
        transcript=None,
        max_new_tokens=48,
    )

    assert answer == "Across a tennis net."
    assert processor.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 48
    assert model.kwargs["moved_to"] == "cuda:0"
    assert processor.messages[0]["content"][0]["text"] == "Full frame:"


def test_load_processor_caps_each_frames_visual_token_budget() -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return model_id, kwargs

    result = load_processor(FakeAutoProcessor, "Qwen/example", "a" * 40)

    assert result[1]["min_pixels"] == MIN_PIXELS_PER_FRAME == 65_536
    assert result[1]["max_pixels"] == MAX_PIXELS_PER_FRAME == 262_144


def test_load_processor_does_not_forward_qwen_pixel_kwargs_to_gemma_tokenizer() -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return model_id, kwargs

    result = load_processor(FakeAutoProcessor, "google/gemma-4-12B-it", "a" * 40)

    assert "min_pixels" not in result[1]
    assert "max_pixels" not in result[1]


def test_legacy_optional_processor_shim_is_scoped_and_restored() -> None:
    class FakeMixin:
        def __init__(self, component, **kwargs):
            if kwargs:
                raise TypeError(f"Unexpected keyword argument {next(iter(kwargs))}.")
            self.component = component

    original_init = FakeMixin.__init__

    class RemoteProcessor(FakeMixin):
        optional_attributes = ["image_use_col_tokens"]

        def __init__(self, component, image_use_col_tokens=True):
            super().__init__(component, image_use_col_tokens=image_use_col_tokens)

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return RemoteProcessor("image")

    processor = load_legacy_optional_processor(
        FakeAutoProcessor,
        FakeMixin,
        "allenai/Molmo2-4B",
        "a" * 40,
        min_pixels=MIN_PIXELS_PER_FRAME,
        max_pixels=MAX_PIXELS_PER_FRAME,
    )

    assert processor.image_use_col_tokens is True
    assert FakeMixin.__init__ is original_init


def test_register_legacy_default_rope_does_not_replace_existing_entry() -> None:
    existing = object()
    registry = {"default": existing}

    register_legacy_default_rope(registry)

    assert registry["default"] is existing


def test_register_legacy_default_rope_adds_missing_entry() -> None:
    registry = {}

    register_legacy_default_rope(registry)

    assert callable(registry["default"])


def test_molmo_generation_receives_initial_cache_positions() -> None:
    class Config:
        model_type = "molmo2"

    class Model:
        config = Config()
        device = "cuda:0"

    calls = []
    marker = object()

    result = generation_compat_kwargs(
        Model(),
        17,
        arange=lambda length, device: calls.append((length, device)) or marker,
    )

    assert result == {"cache_position": marker}
    assert calls == [(17, "cuda:0")]


def test_select_auto_model_class_supports_gemma_multimodal() -> None:
    assert select_auto_model_class("image-text", "image", "multimodal") == "image"
    assert select_auto_model_class("multimodal", "image", "multimodal") == "multimodal"


def test_qwen_runtime_attestation_uses_loaded_parameter_and_rocm_runtime() -> None:
    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:1"

    class Parameter:
        device = Device()
        dtype = "bf16"

    class Model:
        @staticmethod
        def parameters():
            return iter([Parameter()])

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": "6.4"})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    assert transformers_runner.validate_rocm_bfloat16(Model(), Torch()) == (
        "cuda:1 (ROCm)",
        "bfloat16",
    )


def test_qwen_runtime_attestation_rejects_cuda_build_without_hip() -> None:
    class Parameter:
        device = type("Device", (), {"type": "cuda"})()
        dtype = "bf16"

    class Model:
        @staticmethod
        def parameters():
            return iter([Parameter()])

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": None})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    with pytest.raises(RuntimeError, match="ROCm"):
        transformers_runner.validate_rocm_bfloat16(Model(), Torch())
