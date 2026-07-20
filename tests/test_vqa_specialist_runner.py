from __future__ import annotations

from pathlib import Path

import pytest

from trec_video_2026.vqa_specialist_runner import (
    InternVLEvidence,
    build_internvl_prompt,
    build_minicpm_content,
    chat_internvl,
    chat_minicpm,
    classify_error_status,
    dynamic_tiles,
    load_internvl,
    load_minicpm,
    order_evidence,
    split_audio_at_frame_midpoints,
)


def request(tmp_path: Path, *, count: int = 12) -> dict:
    return {
        "run_id": "run-1",
        "condition_id": "condition-1",
        "video_id": "video-1",
        "question": "What happens?",
        "frame_paths": [str(tmp_path / f"frame-{i}.jpg") for i in range(count)],
        "frame_timestamps": [float(i) + 0.5 for i in range(count)],
        "transcript": "[00:01] A bell rings.",
        "media_path": str(tmp_path / "video.mp4"),
    }


def test_order_evidence_requires_exactly_twelve_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 12"):
        order_evidence(request(tmp_path, count=11))


def test_order_evidence_sorts_paths_with_their_timestamps(tmp_path: Path) -> None:
    payload = request(tmp_path)
    payload["frame_paths"] = list(reversed(payload["frame_paths"]))
    payload["frame_timestamps"] = list(reversed(payload["frame_timestamps"]))

    evidence = order_evidence(payload)

    assert evidence.timestamps == tuple(sorted(payload["frame_timestamps"]))
    assert evidence.frame_paths[0].name == "frame-0.jpg"


def test_internvl_prompt_has_timestamped_frame_tokens_and_transcript() -> None:
    evidence = InternVLEvidence(
        frame_paths=tuple(Path(f"{index}.jpg") for index in range(12)),
        timestamps=tuple(index + 0.25 for index in range(12)),
    )

    prompt = build_internvl_prompt(
        evidence, "What happens?", "[00:01] A bell rings."
    )

    assert prompt.startswith("Frame1 at 0.250 seconds: <image>\n")
    assert "Frame12 at 11.250 seconds: <image>" in prompt
    assert "Timestamped audio transcript:\n[00:01] A bell rings." in prompt
    assert prompt.endswith(
        "Answer with one short phrase or sentence. Return only the answer; do not explain your reasoning."
    )


def test_internvl_dynamic_tiling_never_exceeds_two_448_tiles() -> None:
    class WideImage:
        width = 1200
        height = 400

        def resize(self, size):
            assert size == (896, 448)
            return self

        def crop(self, box):
            return box

    assert dynamic_tiles(WideImage(), image_size=448, max_tiles=2) == [
        (0, 0, 448, 448),
        (448, 0, 896, 448),
    ]

class FakeAutoModel:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        return FakeLoadedModel()


class FakeAutoTokenizer:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        return "tokenizer"


class FakeLoadedModel:
    def eval(self):
        self.was_evaluated = True
        return self


class FakeTorch:
    bfloat16 = "bf16"


def test_load_internvl_uses_official_custom_model_without_flash_attention() -> None:
    FakeAutoModel.calls.clear()
    FakeAutoTokenizer.calls.clear()

    model, tokenizer = load_internvl(
        FakeAutoModel, FakeAutoTokenizer, FakeTorch, "OpenGVLab/model", "a" * 40
    )

    assert tokenizer == "tokenizer"
    assert model.was_evaluated is True
    assert FakeAutoModel.calls == [
        (
            "OpenGVLab/model",
            {
                "revision": "a" * 40,
                "trust_remote_code": True,
                "torch_dtype": "bf16",
                "device_map": "cuda:0",
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
                "use_flash_attn": False,
            },
        )
    ]
    assert FakeAutoTokenizer.calls[0][1]["use_fast"] is False


def test_load_minicpm_keeps_audio_and_disables_tts() -> None:
    FakeAutoModel.calls.clear()

    model = load_minicpm(
        FakeAutoModel, FakeTorch, "openbmb/model", "b" * 40
    )

    assert model.was_evaluated is True
    assert FakeAutoModel.calls[0][1] == {
        "revision": "b" * 40,
        "trust_remote_code": True,
        "torch_dtype": "bf16",
        "device_map": "cuda:0",
        "attn_implementation": "sdpa",
        "init_vision": True,
        "init_audio": True,
        "init_tts": False,
    }


def test_audio_is_split_at_frame_midpoints_without_dropping_samples() -> None:
    audio = list(range(80))

    chunks = split_audio_at_frame_midpoints(
        audio, timestamps=(1.0, 3.0, 6.0), sample_rate=10
    )

    assert chunks == [list(range(20)), list(range(20, 45)), list(range(45, 80))]
    assert sum(chunks, []) == audio


def test_minicpm_content_interleaves_ordered_images_and_audio() -> None:
    evidence = InternVLEvidence(
        frame_paths=(Path("later.jpg"), Path("earlier.jpg")),
        timestamps=(3.0, 1.0),
    )
    loaded = []

    class Image:
        @staticmethod
        def open(path):
            loaded.append(path)
            return FakeImage(path)

    content = build_minicpm_content(
        evidence=evidence,
        audio=list(range(40)),
        sample_rate=10,
        question="What happens?",
        transcript=None,
        image_module=Image,
    )

    assert loaded == [Path("earlier.jpg"), Path("later.jpg")]
    assert [type(item).__name__ for item in content[:-1]] == [
        "FakeImage",
        "list",
        "FakeImage",
        "list",
    ]
    assert content[-1].startswith("Question: What happens?")


class FakeImage:
    def __init__(self, path: Path) -> None:
        self.path = path

    def convert(self, mode: str):
        assert mode == "RGB"
        return self


def test_specialist_chat_calls_are_deterministic_and_bounded() -> None:
    class InternModel:
        def chat(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "Answer: ringing"

    intern_model = InternModel()
    assert (
        chat_internvl(
            intern_model,
            "tokenizer",
            "pixels",
            [2] * 12,
            "prompt",
            max_new_tokens=32,
        )
        == "ringing"
    )
    assert intern_model.kwargs["generation_config"] == {
        "do_sample": False,
        "max_new_tokens": 32,
    }
    assert intern_model.kwargs["num_patches_list"] == [2] * 12
    assert intern_model.kwargs["history"] is None
    assert intern_model.kwargs["return_history"] is False

    class MiniModel:
        def chat(self, **kwargs):
            self.kwargs = kwargs
            return "Answer: ringing"

    mini_model = MiniModel()
    assert chat_minicpm(mini_model, ["content"], max_new_tokens=24) == "ringing"
    assert mini_model.kwargs == {
        "msgs": [{"role": "user", "content": ["content"]}],
        "do_sample": False,
        "max_new_tokens": 24,
        "use_image_id": False,
        "max_slice_nums": 1,
        "use_tts_template": False,
        "enable_thinking": False,
        "omni_mode": True,
        "generate_audio": False,
    }


def test_rocm_kernel_failures_are_reported_as_unsupported() -> None:
    assert classify_error_status(RuntimeError("HIP error: invalid device function")) == (
        "unsupported"
    )
    assert classify_error_status(ValueError("bad request")) == "failed"
