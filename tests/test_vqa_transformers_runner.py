from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

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
    parse_args,
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


def _install_fake_main_runtime(
    monkeypatch: pytest.MonkeyPatch,
    processor_calls: list[dict[str, object]],
    generation_calls: list[dict[str, object]],
) -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            processor_calls.append({"model_id": model_id, "kwargs": kwargs})
            return object()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class FakeCuda:
        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> int:
            return 7

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "test-torch"
    fake_torch.bfloat16 = object()
    fake_torch.version = types.SimpleNamespace(hip="test-hip", cuda=None)
    fake_torch.cuda = FakeCuda

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "test-transformers"
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.AutoModelForImageTextToText = FakeAutoModel
    fake_transformers.AutoModelForMultimodalLM = FakeAutoModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        transformers_runner,
        "validate_bfloat16_accelerator",
        lambda model, torch_module: ("fake-device", "bfloat16", {"fake": True}),
    )

    def fake_generate_answer(**kwargs):
        generation_calls.append(kwargs)
        return f"answer-{len(generation_calls)}"

    monkeypatch.setattr(transformers_runner, "generate_answer", fake_generate_answer)


def _main_request(
    video_id: str, question: str | None = None
) -> dict[str, object]:
    request: dict[str, object] = {
        "run_id": "run-1",
        "condition_id": "condition-1",
        "video_id": video_id,
        "frame_paths": ["frame.jpg"],
        "frame_timestamps": [0.5],
    }
    if question is not None:
        request["question"] = question
    return request


def _invoke_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: object,
    *,
    model_id: str = "Qwen/example",
    answer_policy: str = "question-aware",
) -> tuple[object, list[dict[str, object]], list[dict[str, object]]]:
    processor_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []
    _install_fake_main_runtime(monkeypatch, processor_calls, generation_calls)
    monkeypatch.setattr(
        transformers_runner.sys,
        "stdin",
        io.StringIO(json.dumps(payload)),
    )
    transformers_runner.main(
        [
            "--model-id",
            model_id,
            "--revision",
            "a" * 40,
            "--answer-policy",
            answer_policy,
            "--min-pixels-per-frame",
            "123457",
            "--max-pixels-per-frame",
            "654321",
        ]
    )
    return json.loads(capsys.readouterr().out), processor_calls, generation_calls


def test_main_single_payload_threads_policy_bounds_and_audit_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, processor_calls, generation_calls = _invoke_main(
        monkeypatch,
        capsys,
        _main_request("video-1", "How many cups are on the table?"),
    )

    assert isinstance(result, dict)
    assert processor_calls[0]["kwargs"]["min_pixels"] == 123_457
    assert processor_calls[0]["kwargs"]["max_pixels"] == 654_321
    assert generation_calls[0]["answer_policy"] == "question-aware"
    assert result["status"] == "passed"
    assert result["answer_policy"] == "question-aware"
    assert result["question_type"] == "count"
    assert result["requested_min_pixels_per_frame"] == 123_457
    assert result["requested_max_pixels_per_frame"] == 654_321
    assert result["min_pixels_per_frame"] == 123_457
    assert result["max_pixels_per_frame"] == 654_321


def test_main_batch_payload_serializes_each_result_and_forwards_controls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = [
        _main_request("video-1", "How many cups are on the table?"),
        _main_request("video-2", "What color is the bicycle?"),
    ]

    result, processor_calls, generation_calls = _invoke_main(
        monkeypatch,
        capsys,
        payload,
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert len(processor_calls) == 1
    assert len(generation_calls) == 2
    assert [call["answer_policy"] for call in generation_calls] == [
        "question-aware",
        "question-aware",
    ]
    assert [row["video_id"] for row in result] == ["video-1", "video-2"]
    assert [row["question_type"] for row in result] == ["count", "general"]
    for row in result:
        assert row["answer_policy"] == "question-aware"
        assert row["requested_min_pixels_per_frame"] == 123_457
        assert row["requested_max_pixels_per_frame"] == 654_321
        assert row["min_pixels_per_frame"] == 123_457
        assert row["max_pixels_per_frame"] == 654_321


def test_main_gemma_result_marks_requested_bounds_as_not_applied(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, processor_calls, _ = _invoke_main(
        monkeypatch,
        capsys,
        _main_request("video-1", "What color is the bicycle?"),
        model_id="google/gemma-4-12B-it",
        answer_policy="current",
    )

    assert "min_pixels" not in processor_calls[0]["kwargs"]
    assert "max_pixels" not in processor_calls[0]["kwargs"]
    assert result["requested_min_pixels_per_frame"] == 123_457
    assert result["requested_max_pixels_per_frame"] == 654_321
    assert result["min_pixels_per_frame"] is None
    assert result["max_pixels_per_frame"] is None


def test_main_single_missing_question_emits_structured_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, _, generation_calls = _invoke_main(
        monkeypatch,
        capsys,
        _main_request("video-1"),
    )

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert result["error_type"] == "KeyError"
    assert result["question_type"] == "unknown"
    assert result["answer_policy"] == "question-aware"
    assert generation_calls == []


def test_main_batch_missing_question_does_not_abort_other_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result, _, generation_calls = _invoke_main(
        monkeypatch,
        capsys,
        [
            _main_request("video-missing"),
            _main_request("video-valid", "What color is the bicycle?"),
        ],
    )

    assert isinstance(result, list)
    assert [row["status"] for row in result] == ["failed", "passed"]
    assert [row["video_id"] for row in result] == ["video-missing", "video-valid"]
    assert result[0]["error_type"] == "KeyError"
    assert result[0]["question_type"] == "unknown"
    assert result[1]["question_type"] == "general"
    assert len(generation_calls) == 1


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


def test_generate_answer_forwards_question_aware_policy(tmp_path: Path) -> None:
    processor = FakeProcessor()
    model = FakeModel()

    generate_answer(
        processor=processor,
        model=model,
        question="How many cups are on the table?",
        frame_paths=(tmp_path / "frame.jpg",),
        frame_timestamps=(0.5,),
        transcript=None,
        max_new_tokens=64,
        answer_policy="question-aware",
    )

    assert "count or quantity" in processor.messages[0]["content"][-1]["text"]


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 262_144), (-1, 262_144), (65_536, 0), (262_145, 262_144)],
)
def test_parse_args_rejects_non_positive_or_inverted_pixel_bounds(
    minimum: int, maximum: int
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-id",
                "Qwen/example",
                "--revision",
                "a" * 40,
                "--min-pixels-per-frame",
                str(minimum),
                "--max-pixels-per-frame",
                str(maximum),
            ]
        )


def test_load_processor_caps_each_frames_visual_token_budget() -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return model_id, kwargs

    result = load_processor(FakeAutoProcessor, "Qwen/example", "a" * 40)

    assert result[1]["min_pixels"] == MIN_PIXELS_PER_FRAME == 65_536
    assert result[1]["max_pixels"] == MAX_PIXELS_PER_FRAME == 262_144


def test_load_processor_forwards_requested_pixel_bounds() -> None:
    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return model_id, kwargs

    result = load_processor(
        FakeAutoProcessor,
        "Qwen/example",
        "a" * 40,
        min_pixels=123_457,
        max_pixels=654_321,
    )

    assert result[1]["min_pixels"] == 123_457
    assert result[1]["max_pixels"] == 654_321


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
        optional_attributes: ClassVar[list[str]] = ["image_use_col_tokens"]

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


def test_qwen_runtime_attestation_rejects_nonzero_cuda_device() -> None:
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

    with pytest.raises(RuntimeError, match="cuda:0"):
        transformers_runner.validate_rocm_bfloat16(Model(), Torch())


def test_runner_result_records_runtime_attestation() -> None:
    attestation = {
        "parameter_tensor_count": 723,
        "parameter_devices": {"cuda:0": 723},
        "parameter_dtypes": {"torch.bfloat16": 723},
        "buffer_tensor_count": 3,
        "buffer_devices": {"cuda:0": 3},
        "buffer_dtypes": {"torch.float32": 3},
        "torch_hip": "7.2.53211",
    }

    result = transformers_runner._result(
        status="passed",
        answer="answer",
        started=0.0,
        model_id="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        runtime="torch",
        peak_memory=1,
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        runtime_attestation=attestation,
    )

    assert result["runtime_attestation"] == attestation


def test_runner_result_records_policy_question_type_and_actual_pixel_bounds() -> None:
    result = transformers_runner._result(
        status="passed",
        answer="two",
        started=0.0,
        model_id="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        runtime="torch",
        peak_memory=1,
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        runtime_attestation=None,
        answer_policy="question-aware",
        question_type="count",
        min_pixels_per_frame=123_457,
        max_pixels_per_frame=654_321,
    )

    assert result["answer_policy"] == "question-aware"
    assert result["question_type"] == "count"
    assert result["min_pixels_per_frame"] == 123_457
    assert result["max_pixels_per_frame"] == 654_321


def test_runner_result_separates_requested_and_unknown_gemma_bounds() -> None:
    result = transformers_runner._result(
        status="passed",
        answer="answer",
        started=0.0,
        model_id="google/gemma-4-12B-it",
        revision="a" * 40,
        runtime="torch",
        peak_memory=1,
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        runtime_attestation=None,
        answer_policy="current",
        question_type="general",
        requested_min_pixels_per_frame=123_457,
        requested_max_pixels_per_frame=654_321,
        min_pixels_per_frame=None,
        max_pixels_per_frame=None,
    )

    assert result["requested_min_pixels_per_frame"] == 123_457
    assert result["requested_max_pixels_per_frame"] == 654_321
    assert result["min_pixels_per_frame"] is None
    assert result["max_pixels_per_frame"] is None


def test_runner_result_records_requested_and_applied_qwen_bounds() -> None:
    result = transformers_runner._result(
        status="passed",
        answer="answer",
        started=0.0,
        model_id="Qwen/example",
        revision="a" * 40,
        runtime="torch",
        peak_memory=1,
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        runtime_attestation=None,
        requested_min_pixels_per_frame=123_457,
        requested_max_pixels_per_frame=654_321,
        min_pixels_per_frame=123_457,
        max_pixels_per_frame=654_321,
    )

    assert result["requested_min_pixels_per_frame"] == 123_457
    assert result["requested_max_pixels_per_frame"] == 654_321
    assert result["min_pixels_per_frame"] == 123_457
    assert result["max_pixels_per_frame"] == 654_321


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


def test_accelerator_attestation_supports_nvidia_bfloat16_without_changing_rocm() -> (
    None
):
    class Tensor:
        def __init__(self, dtype: str = "bf16") -> None:
            self.device = type(
                "Device",
                (),
                {"__str__": lambda _self: "cuda:0"},
            )()
            self.dtype = dtype

    class Model:
        parameters = staticmethod(lambda: iter([Tensor(), Tensor()]))
        buffers = staticmethod(lambda: iter([Tensor("float32")]))

    class NvidiaTorch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": None, "cuda": "12.9"})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    class RocmTorch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": "6.4", "cuda": None})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    assert transformers_runner.validate_bfloat16_accelerator(
        Model(), NvidiaTorch()
    ) == (
        "cuda:0 (NVIDIA CUDA)",
        "bfloat16",
        {
            "parameter_tensor_count": 2,
            "parameter_devices": {"cuda:0": 2},
            "parameter_dtypes": {"bf16": 2},
            "buffer_tensor_count": 1,
            "buffer_devices": {"cuda:0": 1},
            "buffer_dtypes": {"float32": 1},
            "accelerator_backend": "nvidia-cuda",
            "torch_cuda": "12.9",
            "torch_hip": None,
        },
    )
    assert (
        transformers_runner.validate_bfloat16_accelerator(Model(), RocmTorch())[0]
        == "cuda:0 (ROCm)"
    )


def test_accelerator_attestation_rejects_build_without_rocm_or_cuda() -> None:
    class Parameter:
        device = type("Device", (), {"__str__": lambda _self: "cuda:0"})()
        dtype = "bf16"

    class Model:
        parameters = staticmethod(lambda: iter([Parameter()]))

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": None, "cuda": None})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    with pytest.raises(RuntimeError, match="ROCm nor NVIDIA CUDA"):
        transformers_runner.validate_bfloat16_accelerator(Model(), Torch())


def test_qwen_runtime_attestation_allows_fp32_cuda_rotary_buffers() -> None:
    class Tensor:
        def __init__(self, device_type: str = "cuda", dtype: str = "bf16") -> None:
            self.device = type(
                "Device",
                (),
                {
                    "type": device_type,
                    "__str__": lambda _self: (
                        "cuda:0" if device_type == "cuda" else device_type
                    ),
                },
            )()
            self.dtype = dtype

    class Model:
        @staticmethod
        def parameters():
            return iter([Tensor(), Tensor()])

        @staticmethod
        def buffers():
            return iter(
                [
                    Tensor(dtype="float32"),
                    Tensor(dtype="float32"),
                    Tensor(dtype="float32"),
                ]
            )

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": "6.4"})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    assert transformers_runner.validate_rocm_bfloat16(Model(), Torch()) == (
        "cuda:0 (ROCm)",
        "bfloat16",
        {
            "parameter_tensor_count": 2,
            "parameter_devices": {"cuda:0": 2},
            "parameter_dtypes": {"bf16": 2},
            "buffer_tensor_count": 3,
            "buffer_devices": {"cuda:0": 3},
            "buffer_dtypes": {"float32": 3},
            "torch_hip": "6.4",
        },
    )


def test_qwen_runtime_attestation_rejects_mixed_parameter_dtype_or_cpu_buffer() -> None:
    class Tensor:
        def __init__(self, device_type: str = "cuda", dtype: str = "bf16") -> None:
            self.device = type(
                "Device",
                (),
                {
                    "type": device_type,
                    "__str__": lambda _self: (
                        "cuda:0" if device_type == "cuda" else device_type
                    ),
                },
            )()
            self.dtype = dtype

    class Model:
        parameters = staticmethod(lambda: iter([Tensor(dtype="float32")]))
        buffers = staticmethod(lambda: iter([Tensor()]))

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": "6.4"})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    with pytest.raises(RuntimeError, match="parameters.*bfloat16"):
        transformers_runner.validate_rocm_bfloat16(Model(), Torch())

    Model.parameters = staticmethod(lambda: iter([Tensor()]))
    Model.buffers = staticmethod(lambda: iter([Tensor(device_type="cpu")]))
    with pytest.raises(RuntimeError, match="buffers.*cuda:0"):
        transformers_runner.validate_rocm_bfloat16(Model(), Torch())


def test_qwen_runtime_attestation_rejects_split_cuda_devices() -> None:
    class Tensor:
        def __init__(self, device: str) -> None:
            self.device = type(
                "Device",
                (),
                {
                    "type": "cuda",
                    "__str__": lambda _self: device,
                },
            )()
            self.dtype = "bf16"

    class Model:
        parameters = staticmethod(lambda: iter([Tensor("cuda:0"), Tensor("cuda:1")]))
        buffers = staticmethod(lambda: iter([Tensor("cuda:0")]))

    class Torch:
        bfloat16 = "bf16"
        version = type("Version", (), {"hip": "6.4"})()
        cuda = type("Cuda", (), {"is_available": staticmethod(lambda: True)})()

    with pytest.raises(RuntimeError, match="cuda:0"):
        transformers_runner.validate_rocm_bfloat16(Model(), Torch())
