from __future__ import annotations

from pathlib import Path

from trec_video_2026.vqa_transformers_runner import (
    MAX_PIXELS_PER_FRAME,
    MIN_PIXELS_PER_FRAME,
    generate_answer,
    load_processor,
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


def test_generate_answer_uses_deterministic_non_thinking_decode(tmp_path: Path) -> None:
    processor = FakeProcessor()
    model = FakeModel()

    answer = generate_answer(
        processor=processor,
        model=model,
        question="What happens?",
        frame_paths=(tmp_path / "frame.jpg",),
        frame_timestamps=(0.5,),
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


def test_load_processor_caps_each_frames_visual_token_budget() -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return model_id, kwargs

    result = load_processor(FakeAutoProcessor, "Qwen/example", "a" * 40)

    assert result[1]["min_pixels"] == MIN_PIXELS_PER_FRAME == 65_536
    assert result[1]["max_pixels"] == MAX_PIXELS_PER_FRAME == 262_144
