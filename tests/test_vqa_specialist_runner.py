from __future__ import annotations

import argparse
import shutil
import wave
from pathlib import Path

import pytest

from trec_video_2026.vqa_specialist_runner import (
    InternVLEvidence,
    _decode_full_media_audio,
    _error_result,
    _load_audio,
    build_internvl_prompt,
    build_minicpm_content,
    chat_internvl,
    chat_minicpm,
    classify_error_status,
    dynamic_tiles,
    load_internvl,
    load_minicpm,
    minicpm_audio_enabled,
    minicpm_metadata,
    order_evidence,
    validate_rocm_bfloat16,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    prompt = build_internvl_prompt(evidence, "What happens?", "[00:01] A bell rings.")

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
    class Config:
        _name_or_path = "remote"

    config = Config()

    def eval(self):
        self.was_evaluated = True
        return self

    def prepare_processor(self):
        self.processor = f"processor:{self.config._name_or_path}"


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
                "attn_implementation": "eager",
                "use_flash_attn": False,
            },
        )
    ]
    assert FakeAutoTokenizer.calls[0][1]["use_fast"] is False


def test_load_minicpm_keeps_audio_and_disables_tts() -> None:
    FakeAutoModel.calls.clear()

    model, processor = load_minicpm(
        FakeAutoModel,
        FakeTorch,
        "openbmb/model",
        "b" * 40,
        snapshot_resolver=lambda model_id,
        revision,
        local_files_only: "/cache/snapshot",
    )

    assert model.was_evaluated is True
    assert processor == "processor:/cache/snapshot"
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


def test_minicpm_modes_use_identical_ordered_visual_evidence_and_no_transcript() -> (
    None
):
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

    audio = list(range(40))
    content_off = build_minicpm_content(
        evidence=evidence,
        audio=None,
        audio_enabled=False,
        question="What happens?",
        image_module=Image,
    )
    content_on = build_minicpm_content(
        evidence=evidence,
        audio=audio,
        audio_enabled=True,
        question="What happens?",
        image_module=Image,
    )

    assert loaded == [
        Path("earlier.jpg"),
        Path("later.jpg"),
        Path("earlier.jpg"),
        Path("later.jpg"),
    ]
    assert [item.path for item in content_off[:-1]] == [
        Path("earlier.jpg"),
        Path("later.jpg"),
    ]
    assert [item.path for item in content_on[:2]] == [
        Path("earlier.jpg"),
        Path("later.jpg"),
    ]
    assert content_on[2] is audio
    assert "transcript" not in content_off[-1].lower()
    assert "transcript" not in content_on[-1].lower()
    assert content_off[-1] == content_on[-1]


def test_minicpm_audio_on_passes_full_waveform_once_without_truncation() -> None:
    evidence = InternVLEvidence(
        frame_paths=tuple(Path(f"{index}.jpg") for index in range(12)),
        timestamps=tuple(index + 0.5 for index in range(12)),
    )

    class Image:
        @staticmethod
        def open(path):
            return FakeImage(path)

    audio = list(range(304_089))
    content = build_minicpm_content(
        evidence=evidence,
        audio=audio,
        audio_enabled=True,
        question="What happens?",
        image_module=Image,
    )

    assert content[-2] is audio
    assert sum(item is audio for item in content) == 1
    assert len(content[-2]) == 304_089


@pytest.mark.parametrize(
    ("mode", "enabled"), (("audio-off", False), ("audio-on", True))
)
def test_minicpm_audio_mode_is_explicit(mode: str, enabled: bool) -> None:
    assert minicpm_audio_enabled({"audio_mode": mode}) is enabled


def test_minicpm_audio_mode_rejects_missing_or_unknown_values() -> None:
    with pytest.raises(ValueError, match="audio_mode must be audio-off or audio-on"):
        minicpm_audio_enabled({})
    with pytest.raises(ValueError, match="audio_mode must be audio-off or audio-on"):
        minicpm_audio_enabled({"audio_mode": "automatic"})


def test_minicpm_metadata_distinguishes_the_matched_modes() -> None:
    assert minicpm_metadata(False) == {
        "evidence_mode": "selected_frames_audio_off",
        "frame_count": 12,
        "native_audio": False,
        "audio_sample_rate_hz": None,
        "audio_input_count": 0,
        "max_slice_nums": 1,
        "transcript_included": False,
    }
    assert minicpm_metadata(True) == {
        "evidence_mode": "selected_frames_native_audio_on",
        "frame_count": 12,
        "native_audio": True,
        "audio_sample_rate_hz": 16_000,
        "audio_input_count": 1,
        "max_slice_nums": 1,
        "transcript_included": False,
    }


def test_minicpm_failures_remain_structured_and_bounded() -> None:
    args = argparse.Namespace(
        kind="minicpm", model_id="openbmb/model", revision="b" * 40
    )

    result = _error_result(
        RuntimeError("x" * 5_000),
        0.0,
        args=args,
        runtime="runtime",
        peak_memory=123,
        audio_enabled=False,
    )

    assert result["status"] == "failed"
    assert result["answer"] is None
    assert result["native_audio"] is False
    assert len(result["error_message"]) == 4_000
    assert result["error_message"].endswith("…")


def test_minicpm_media_decode_preserves_sample_beyond_midpoint_derived_duration(
    tmp_path: Path,
) -> None:
    payload = request(tmp_path)
    payload["audio_path"] = None
    payload["media_path"] = str(tmp_path / "fractional.wav")
    sample_count = 192_001
    with wave.open(payload["media_path"], "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x01\x00" * sample_count)

    class Librosa:
        @staticmethod
        def load(source, **kwargs):
            raise AssertionError("media containers must use the full decoder")

    audio, sample_rate = _load_audio(
        payload,
        Librosa,
        full_media_decoder=lambda source: _decode_full_media_audio(
            source,
            lambda value: memoryview(value).cast("f"),
            ffmpeg_resolver=lambda: shutil.which("ffmpeg"),
        ),
    )

    assert sample_rate == 16_000
    assert (
        int(
            (min(payload["frame_timestamps"]) + max(payload["frame_timestamps"]))
            * 16_000
        )
        == 192_000
    )
    assert len(audio) == sample_count


class FakeParameter:
    def __init__(self, device_type="cuda", device="cuda:0", dtype="bf16"):
        self.device = type(
            "Device", (), {"type": device_type, "__str__": lambda self: device}
        )()
        self.dtype = dtype


class ParameterModel:
    def __init__(self, parameter):
        self.parameter = parameter

    def parameters(self):
        yield self.parameter


class ProvenanceTorch:
    bfloat16 = "bf16"
    version = type("Version", (), {"hip": "7.2"})()

    class cuda:
        @staticmethod
        def is_available():
            return True


def test_runtime_provenance_comes_from_loaded_rocm_bfloat16_model() -> None:
    provenance = validate_rocm_bfloat16(
        ParameterModel(FakeParameter()), ProvenanceTorch
    )

    assert provenance == ("cuda:0 (ROCm)", "bfloat16")


@pytest.mark.parametrize(
    ("parameter", "torch_module", "message"),
    [
        (FakeParameter(device_type="cpu", device="cpu"), ProvenanceTorch, "CUDA"),
        (FakeParameter(dtype="float32"), ProvenanceTorch, "bfloat16"),
        (
            FakeParameter(),
            type(
                "NoHipTorch",
                (),
                {
                    "bfloat16": "bf16",
                    "version": type("Version", (), {"hip": None})(),
                    "cuda": ProvenanceTorch.cuda,
                },
            ),
            "ROCm",
        ),
    ],
)
def test_runtime_provenance_fails_closed(parameter, torch_module, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_rocm_bfloat16(ParameterModel(parameter), torch_module)


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
    assert (
        chat_minicpm(mini_model, ["content"], processor="processor", max_new_tokens=24)
        == "ringing"
    )
    assert mini_model.kwargs == {
        "msgs": [{"role": "user", "content": ["content"]}],
        "processor": "processor",
        "do_sample": False,
        "max_new_tokens": 24,
        "use_image_id": False,
        "max_slice_nums": 1,
        "use_tts_template": False,
        "enable_thinking": False,
        "omni_mode": True,
        "generate_audio": False,
        "merge_audio_from_same_content": False,
    }


def test_minicpm_conditions_are_a_frozen_audio_off_on_pair() -> None:
    import json

    payload = json.loads(
        (REPO_ROOT / "configs/vqa/model_runs.json").read_text(encoding="utf-8")
    )
    conditions = {
        item["condition_id"]: item
        for item in payload["conditions"]
        if item["model_id"] == "openbmb/MiniCPM-o-4_5"
    }

    assert set(conditions) == {
        "minicpmo45-native-video-audio-off",
        "minicpmo45-native-video-audio-on",
    }
    off = conditions["minicpmo45-native-video-audio-off"]
    on = conditions["minicpmo45-native-video-audio-on"]
    for key in ("model_id", "model_revision", "runner_kind", "runtime_image"):
        assert off[key] == on[key]
    assert off["evidence"] == "selected-12-frames-audio-off"
    assert on["evidence"] == "selected-12-frames-native-audio-on"


def test_rocm_kernel_failures_are_reported_as_unsupported() -> None:
    assert classify_error_status(
        RuntimeError("HIP error: invalid device function")
    ) == ("unsupported")
    assert classify_error_status(ValueError("bad request")) == "failed"
