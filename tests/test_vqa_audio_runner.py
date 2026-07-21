from __future__ import annotations

from pathlib import Path
import json

import pytest

from trec_video_2026.vqa_audio_runner import (
    build_voxtral_messages,
    classify_error_status,
    decode_voxtral,
    normalize_qwen_asr_result,
    prepare_voxtral_inputs,
    request_identity,
    transcribe_qwen,
    validate_request,
    validate_rocm_bfloat16,
)


def audio_request(tmp_path: Path, *, mode: str = "transcription") -> dict:
    request = {
        "run_id": "task5",
        "condition_id": "voxtral-transcription",
        "video_id": "Mi-NpnCBegw",
        "audio_path": str(tmp_path / "Mi-NpnCBegw.wav"),
        "mode": mode,
    }
    if mode == "question_answer":
        request["question"] = "What word is defined at the end?"
    return request


def test_request_identity_copies_only_label_free_protocol_fields(tmp_path: Path) -> None:
    request = audio_request(tmp_path)
    request.update({"reference_answer": "mumble", "correct_answer": "mumble"})

    identity = request_identity(request)

    assert identity == {
        "run_id": "task5",
        "condition_id": "voxtral-transcription",
        "video_id": "Mi-NpnCBegw",
    }
    assert "reference_answer" not in identity


@pytest.mark.parametrize("mode", ["transcription", "translation"])
def test_validate_request_accepts_pure_audio_modes_without_question(
    tmp_path: Path, mode: str
) -> None:
    validate_request(audio_request(tmp_path, mode=mode), model_kind="voxtral")


def test_validate_request_requires_question_only_for_question_aware_mode(
    tmp_path: Path,
) -> None:
    request = audio_request(tmp_path, mode="question_answer")
    del request["question"]

    with pytest.raises(ValueError, match="question must be a nonempty string"):
        validate_request(request, model_kind="voxtral")


def test_validate_request_rejects_qwen_question_aware_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Qwen3-ASR supports transcription only"):
        validate_request(
            audio_request(tmp_path, mode="question_answer"), model_kind="qwen3_asr"
        )


def test_validate_request_requires_absolute_audio_path(tmp_path: Path) -> None:
    request = audio_request(tmp_path)
    request["audio_path"] = "relative.wav"

    with pytest.raises(ValueError, match="absolute path"):
        validate_request(request, model_kind="voxtral")


def test_voxtral_pure_transcription_uses_dedicated_auto_language_api(
    tmp_path: Path,
) -> None:
    processor = FakeProcessor()

    inputs = prepare_voxtral_inputs(
        processor,
        audio_request(tmp_path),
        model_id="mistralai/Voxtral-Mini-3B-2507",
    )

    assert inputs == {"input_ids": [[7, 8]]}
    assert processor.transcription_call == {
        "language": None,
        "audio": str(tmp_path / "Mi-NpnCBegw.wav"),
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
    }


def test_voxtral_question_aware_mode_keeps_question_separate_from_audio(
    tmp_path: Path,
) -> None:
    request = audio_request(tmp_path, mode="question_answer")

    messages = build_voxtral_messages(request)

    assert messages[0]["content"][0] == {
        "type": "audio",
        "path": request["audio_path"],
    }
    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "What word is defined at the end?",
    }


def test_voxtral_translation_uses_one_frozen_english_instruction(
    tmp_path: Path,
) -> None:
    messages = build_voxtral_messages(audio_request(tmp_path, mode="translation"))

    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "Translate the complete audio into English. Return only the translation.",
    }


class FakeBatch(dict):
    def to(self, *, device, dtype):
        self.device = device
        self.dtype = dtype
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return FakeBatch({"input_ids": [[7, 8]]})

    def apply_transcription_request(self, **kwargs):
        self.transcription_call = kwargs
        return FakeBatch({"input_ids": [[7, 8]]})

    def batch_decode(self, tokens, **kwargs):
        self.decode_call = (tokens, kwargs)
        return ["  Mumble.  "]


class FakeVoxtralModel:
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[7, 8, 21, 22]]


def test_voxtral_decode_uses_deterministic_generation_and_only_new_tokens(
    tmp_path: Path,
) -> None:
    processor = FakeProcessor()
    model = FakeVoxtralModel()

    text = decode_voxtral(
        model,
        processor,
        build_voxtral_messages(audio_request(tmp_path, mode="question_answer")),
        device="cuda:0",
        dtype="bf16",
        max_new_tokens=64,
    )

    assert text == "Mumble."
    assert processor.calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert model.generate_kwargs == {
        "input_ids": [[7, 8]],
        "do_sample": False,
        "max_new_tokens": 64,
    }
    assert processor.decode_call == (
        [[21, 22]],
        {"skip_special_tokens": True},
    )


class QwenSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start_time = start
        self.end_time = end
        self.text = text


class QwenResult:
    text = " Bonjour le monde "
    language = "French"
    time_stamps = [QwenSegment(0.0, 1.25, "Bonjour"), QwenSegment(1.25, 2.0, "le monde")]


def test_qwen_result_retains_language_text_and_timestamp_segments() -> None:
    result = normalize_qwen_asr_result(QwenResult())

    assert result == {
        "text": "Bonjour le monde",
        "language": "French",
        "timestamped_text": (
            "[00:00:00.000] Bonjour\n[00:00:01.250] le monde"
        ),
        "chunks": [
            {"start": 0.0, "end": 1.25, "text": "Bonjour"},
            {"start": 1.25, "end": 2.0, "text": "le monde"},
        ],
    }


def test_qwen_result_accepts_missing_timestamps_but_not_missing_language() -> None:
    result = QwenResult()
    result.time_stamps = None
    assert normalize_qwen_asr_result(result)["chunks"] == []

    result.language = None
    with pytest.raises(ValueError, match="language metadata"):
        normalize_qwen_asr_result(result)


class FakeQwenModel:
    def transcribe(self, **kwargs):
        self.transcribe_kwargs = kwargs
        return [QwenResult()]


def test_qwen_transcription_uses_auto_language_and_no_sampling_surface(
    tmp_path: Path,
) -> None:
    model = FakeQwenModel()

    result = transcribe_qwen(model, str(tmp_path / "audio.wav"))

    assert result["language"] == "French"
    assert model.transcribe_kwargs == {
        "audio": str(tmp_path / "audio.wav"),
        "language": None,
    }


class FakeDevice:
    type = "cuda"

    def __str__(self) -> str:
        return "cuda:0"


class FakeParameter:
    device = FakeDevice()
    dtype = "bf16"


class FakeModel:
    def parameters(self):
        yield FakeParameter()


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    bfloat16 = "bf16"
    cuda = FakeCuda()
    version = type("Version", (), {"hip": "6.4"})()


def test_runtime_validation_reports_actual_rocm_bfloat16() -> None:
    assert validate_rocm_bfloat16(FakeModel(), FakeTorch) == (
        "cuda:0 (ROCm)",
        "bfloat16",
    )


def test_runtime_validation_rejects_actual_non_bfloat16_parameter() -> None:
    parameter = FakeParameter()
    parameter.dtype = "float16"
    model = FakeModel()
    model.parameters = lambda: iter([parameter])

    with pytest.raises(RuntimeError, match="not bfloat16"):
        validate_rocm_bfloat16(model, FakeTorch)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NotImplementedError("kernel"), "unsupported"),
        (RuntimeError("HIP error: invalid device function"), "unsupported"),
        (ValueError("bad request"), "failed"),
    ],
)
def test_error_classification_distinguishes_runtime_incompatibility(
    error: Exception, status: str
) -> None:
    assert classify_error_status(error) == status


def test_registry_pins_task5_audio_conditions() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "configs/vqa/model_runs.json").read_text())
    by_id = {item["condition_id"]: item for item in registry["conditions"]}
    expected = {
        "voxtral-mini-3b-transcription": (
            "mistralai/Voxtral-Mini-3B-2507",
            "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
            "transcription",
        ),
        "voxtral-mini-3b-translation": (
            "mistralai/Voxtral-Mini-3B-2507",
            "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
            "translation",
        ),
        "voxtral-mini-3b-question-answer": (
            "mistralai/Voxtral-Mini-3B-2507",
            "3060fe34b35ba5d44202ce9ff3c097642914f8f3",
            "question_answer",
        ),
        "qwen3-asr-1.7b-transcription": (
            "Qwen/Qwen3-ASR-1.7B",
            "7278e1e70fe206f11671096ffdd38061171dd6e5",
            "transcription",
        ),
    }

    for condition_id, (model_id, revision, mode) in expected.items():
        condition = by_id[condition_id]
        assert condition["model_id"] == model_id
        assert condition["model_revision"] == revision
        assert condition["task_mode"] == mode
        assert condition["runner_kind"] == "isolated-audio"
        assert condition["runtime_image"].startswith(
            "localhost/trec-vqa-audio-rocm@sha256:"
        )


def test_audio_container_pins_official_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "containers/vqa-gate/Containerfile.audio").read_text()

    assert '"qwen-asr==0.0.6"' in text
    assert '"mistral-common[audio]==1.11.6"' in text
    assert '"transformers==4.57.6"' in text
