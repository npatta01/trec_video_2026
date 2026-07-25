import builtins
import io
import json
import sys
import time
from types import ModuleType

import pytest

import trec_video_2026.vqa_seamless as seamless
from trec_video_2026.vqa_seamless import (
    decode_translation,
    decode_text_translation,
    prepare_audio_inputs,
    prepare_text_inputs,
    request_identity,
    translate_audio,
    translate_text,
    select_translation_attempts,
    translation_execution_identity,
    translation_request_sha256,
    validate_translation_attempt,
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
        type(self).generate_kwargs = kwargs
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

    with pytest.raises(ValueError, match="SeamlessM4T returned an empty translation"):
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
    assert FakeModel.generate_kwargs == {
        "input_features": "features",
        "tgt_lang": "eng",
    }


def test_prepare_text_inputs_records_source_language_without_casting_ids() -> None:
    processor = FakeProcessor()

    prepared = prepare_text_inputs(
        processor, "Bonjour le monde", source_language="fra", device="cuda:0"
    )

    assert processor.kwargs == {
        "text": "Bonjour le monde",
        "src_lang": "fra",
        "return_tensors": "pt",
    }
    assert prepared.to_kwargs == {"device": "cuda:0"}


def test_translate_text_is_deterministic_text_to_english_only() -> None:
    class FakeTextModel(FakeModel):
        def generate(self, **kwargs):
            type(self).generate_kwargs = kwargs
            return [[[11, 12, 3]]]

    result = translate_text(
        FakeTextModel(),
        FakeProcessor(),
        "Bonjour le monde",
        source_language="fra",
        device="cuda:0",
        inference_mode=FakeInferenceMode,
    )

    assert result == "Where is the bag?"
    assert FakeTextModel.generate_kwargs == {
        "input_features": "features",
        "tgt_lang": "eng",
        "generate_speech": False,
        "do_sample": False,
    }


def test_decode_text_translation_uses_unified_model_text_tokens() -> None:
    processor = FakeProcessor()

    assert decode_text_translation(processor, [[[11, 12, 3]]]) == (
        "Where is the bag?"
    )
    assert processor.decode_args == ([11, 12, 3],)


def text_translation_request() -> dict:
    return {
        "run_id": "task5-audio-v1",
        "condition_id": "qwen3-asr-seamless-t2t",
        "video_id": "video-1",
        "original_text": "Bonjour",
        "source_language": "French",
        "source_language_code": "fra",
        "source_attempt_sha256": "a" * 64,
    }


def valid_text_translation_attempt() -> dict:
    request = text_translation_request()
    identity = translation_execution_identity(request)
    return {
        "status": "passed",
        "translated_text": "Hello",
        "request": request,
        "request_sha256": translation_request_sha256(request),
        "reuse_identity_sha256": translation_request_sha256(identity),
        "run_id": request["run_id"],
        "condition_id": request["condition_id"],
        "video_id": request["video_id"],
        "original_text": request["original_text"],
        "source_attempt_sha256": request["source_attempt_sha256"],
        "source_language": "French",
        "source_language_code": "fra",
        "target_language": "English",
        "target_language_code": "eng",
        "model_id": identity["execution"]["model_id"],
        "model_revision": identity["execution"]["model_revision"],
        "runtime_image": identity["execution"]["runtime_image"],
        "runtime": "transformers 5.15.0.dev0; torch 2.10.0+rocm",
        "device": "cuda:0 (ROCm)",
        "precision": "bfloat16",
        "translation_mode": "text_to_text",
        "processor_interface": identity["execution"]["processor_interface"],
        "generate_speech": False,
        "do_sample": False,
    }


def test_translation_execution_identity_binds_source_and_deterministic_prompt() -> None:
    request = text_translation_request()

    assert translation_execution_identity(request) == {
        "request": request,
        "execution": {
            "model_id": "facebook/seamless-m4t-v2-large",
            "model_revision": "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
            "runtime_image": (
                "localhost/trec-vqa-gate-rocm@sha256:"
                "510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
            ),
            "translation_mode": "text_to_text",
            "processor_interface": "AutoProcessor(text,src_lang)",
            "target_language": "English",
            "target_language_code": "eng",
            "generate_speech": False,
            "do_sample": False,
            "device": "cuda:0 (ROCm)",
            "precision": "bfloat16",
        },
    }


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("translated_text", ""),
        ("source_attempt_sha256", "b" * 64),
        ("original_text", "mutated"),
        ("source_language", "Spanish"),
        ("runtime_image", "mutated-image"),
        ("model_revision", "mutated-revision"),
        ("device", "cpu"),
        ("precision", "float32"),
        ("translation_mode", "speech_to_text"),
        ("processor_interface", "mutated-interface"),
        ("generate_speech", True),
        ("do_sample", True),
        ("reuse_identity_sha256", "0" * 64),
    ],
)
def test_translation_attempt_validation_fails_closed_on_mutation(
    field, mutated
) -> None:
    attempt = valid_text_translation_attempt()
    attempt[field] = mutated

    assert not validate_translation_attempt(attempt, text_translation_request())


def test_translation_attempt_selection_requires_exact_expected_rows() -> None:
    request = text_translation_request()
    attempt = valid_text_translation_attempt()

    assert select_translation_attempts([attempt], [request]) == [attempt]
    with pytest.raises(ValueError, match="exactly one valid translation"):
        select_translation_attempts([], [request])
    with pytest.raises(ValueError, match="exactly one valid translation"):
        select_translation_attempts([attempt, attempt], [request])


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


def test_request_identity_is_failure_safe_for_partial_requests() -> None:
    assert request_identity({"run_id": "run-1"}) == {
        "run_id": "run-1",
        "condition_id": None,
        "video_id": None,
    }


def test_cli_rejects_model_or_revision_overrides() -> None:
    with pytest.raises(ValueError, match="pinned model and revision"):
        seamless.validate_cli_args(["--model-id", "caller/model"])
    with pytest.raises(ValueError, match="pinned model and revision"):
        seamless.validate_cli_args(["--revision", "caller-revision"])


def test_main_preserves_request_identity_when_cli_args_are_rejected(
    monkeypatch,
) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "run_id": "run-1",
                    "condition_id": "condition-1",
                    "video_id": "video-1",
                    "audio_path": "/tmp/audio.wav",
                }
            )
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main(["--model-id", "caller/model"]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["run_id"] == "run-1"
    assert result["condition_id"] == "condition-1"
    assert result["video_id"] == "video-1"
    assert result["error_type"] == "ValueError"
    assert result["error_message"] == ("This runner uses a pinned model and revision")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("run_id", None),
        ("run_id", ""),
        ("run_id", 7),
        ("condition_id", None),
        ("condition_id", ""),
        ("condition_id", 7),
        ("video_id", None),
        ("video_id", ""),
        ("video_id", 7),
    ],
)
def test_main_rejects_invalid_identity_before_runtime_imports(
    monkeypatch, field, bad_value
) -> None:
    request = {
        "run_id": "run-1",
        "condition_id": "condition-1",
        "video_id": "video-1",
        "audio_path": "/tmp/audio.wav",
    }
    if bad_value is None:
        del request[field]
    else:
        request[field] = bad_value
    runtime_imports = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name in {"soundfile", "torch", "transformers"}:
            runtime_imports.append(name)
        return real_import(name, *args, **kwargs)

    stdout = io.StringIO()
    monkeypatch.setattr(builtins, "__import__", tracking_import)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert result[field] is None
    for identity_field in {"run_id", "condition_id", "video_id"} - {field}:
        assert result[identity_field] == request[identity_field]
    assert runtime_imports == []


@pytest.mark.parametrize("audio_path", [None, "", 7, "relative/audio.wav"])
def test_main_rejects_invalid_audio_path_before_runtime_imports(
    monkeypatch, audio_path
) -> None:
    request = {
        "run_id": "run-1",
        "condition_id": "condition-1",
        "video_id": "video-1",
    }
    if audio_path is not None:
        request["audio_path"] = audio_path
    runtime_imports = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        if name in {"soundfile", "torch", "transformers"}:
            runtime_imports.append(name)
        return real_import(name, *args, **kwargs)

    stdout = io.StringIO()
    monkeypatch.setattr(builtins, "__import__", tracking_import)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["run_id"] == "run-1"
    assert result["condition_id"] == "condition-1"
    assert result["video_id"] == "video-1"
    assert result["error_type"] == "ValueError"
    assert runtime_imports == []


def test_result_uses_immutable_model_provenance() -> None:
    result = seamless._result(
        request={},
        status="failed",
        source_audio_sha256=None,
        text=None,
        quality=None,
        started=time.monotonic(),
        runtime="unavailable",
        peak_memory_bytes=None,
    )

    assert result["model_id"] == seamless.MODEL_ID
    assert result["model_revision"] == seamless.MODEL_REVISION


def test_main_emits_structured_json_for_malformed_input(monkeypatch) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO("{"))
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["run_id"] is None
    assert result["condition_id"] is None
    assert result["video_id"] is None
    assert result["model_id"] == seamless.MODEL_ID
    assert result["model_revision"] == seamless.MODEL_REVISION
    assert result["peak_memory_bytes"] is None
    assert result["error_type"] == "JSONDecodeError"


def test_main_emits_structured_json_for_runtime_import_failure(
    monkeypatch,
) -> None:
    stdout = io.StringIO()
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "soundfile":
            raise RuntimeError("soundfile import failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "run_id": "run-1",
                    "condition_id": "condition-1",
                    "video_id": "video-1",
                    "audio_path": "/tmp/audio.wav",
                }
            )
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["run_id"] == "run-1"
    assert result["peak_memory_bytes"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "soundfile import failed"


def test_main_preserves_cuda_reset_error_when_peak_query_also_fails(
    monkeypatch,
) -> None:
    class FailingCuda:
        @staticmethod
        def reset_peak_memory_stats():
            raise RuntimeError("CUDA reset failed")

        @staticmethod
        def max_memory_allocated():
            raise RuntimeError("CUDA peak query failed")

    torch_module = ModuleType("torch")
    torch_module.__version__ = "test"
    torch_module.cuda = FailingCuda()
    transformers_module = ModuleType("transformers")
    transformers_module.__version__ = "test"
    transformers_module.AutoProcessor = object()
    transformers_module.SeamlessM4Tv2ForSpeechToText = object()
    monkeypatch.setitem(sys.modules, "soundfile", ModuleType("soundfile"))
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    stdout = io.StringIO()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "run_id": "run-1",
                    "condition_id": "condition-1",
                    "video_id": "video-1",
                    "audio_path": "/tmp/audio.wav",
                }
            )
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "failed"
    assert result["peak_memory_bytes"] is None
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "CUDA reset failed"


def test_main_loads_only_the_module_pinned_model_and_revision(
    monkeypatch, tmp_path
) -> None:
    load_calls = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            load_calls.append(("processor", args, kwargs))
            return FakeProcessor()

    class LoadedModel(FakeModel):
        class Parameter:
            device = type(
                "Device",
                (),
                {"type": "cuda", "__str__": lambda self: "cuda:0"},
            )()
            dtype = "bfloat16"

        def to(self, device):
            self.device = device
            return self

        def parameters(self):
            return iter([self.Parameter()])

    class FakeSeamlessModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            load_calls.append(("model", args, kwargs))
            return LoadedModel()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 123

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    soundfile_module = ModuleType("soundfile")
    soundfile_module.read = lambda path, dtype: ([0.1, -0.1], 16_000)
    torch_module = ModuleType("torch")
    torch_module.__version__ = "test"
    torch_module.cuda = FakeCuda()
    torch_module.bfloat16 = "bfloat16"
    torch_module.version = type("Version", (), {"hip": "7.2"})()
    torch_module.inference_mode = InferenceMode
    transformers_module = ModuleType("transformers")
    transformers_module.__version__ = "test"
    transformers_module.AutoProcessor = FakeAutoProcessor
    transformers_module.SeamlessM4Tv2ForSpeechToText = FakeSeamlessModel
    monkeypatch.setitem(sys.modules, "soundfile", soundfile_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"frozen audio")
    stdout = io.StringIO()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "run_id": "run-1",
                    "condition_id": "condition-1",
                    "video_id": "video-1",
                    "audio_path": str(audio_path),
                }
            )
        ),
    )
    monkeypatch.setattr(sys, "stdout", stdout)

    assert seamless.main([]) == 0

    result = json.loads(stdout.getvalue())
    assert result["status"] == "passed"
    assert result["model_id"] == seamless.MODEL_ID
    assert result["model_revision"] == seamless.MODEL_REVISION
    assert result["device"] == "cuda:0 (ROCm)"
    assert result["precision"] == "bfloat16"
    assert load_calls == [
        (
            "processor",
            (seamless.MODEL_ID,),
            {"revision": seamless.MODEL_REVISION},
        ),
        (
            "model",
            (seamless.MODEL_ID,),
            {
                "revision": seamless.MODEL_REVISION,
                "dtype": "bfloat16",
                "low_cpu_mem_usage": True,
            },
        ),
    ]
