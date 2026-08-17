from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

import trec_video_2026.vqa_qwen_empty_output_runner as runner
from trec_video_2026.vqa_qwen_empty_output_runner import (
    parse_args,
    transcribe_qwen_mode,
)


class FakeQwenOutput:
    text = " hello world "
    language = "English"
    time_stamps: ClassVar[list] = []


class FakeEmptyQwenOutput:
    text = "   "
    language = "English"
    time_stamps: ClassVar[list] = []


class FakeQwenModel:
    def __init__(self, outputs: list[object] | None = None) -> None:
        self.outputs = outputs or [FakeQwenOutput()]
        self.calls: list[dict[str, object]] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs


@pytest.mark.parametrize(
    ("language", "timestamps", "expected"),
    (
        (None, False, {"audio": "/a.wav", "language": None}),
        (
            None,
            True,
            {
                "audio": "/a.wav",
                "language": None,
                "return_time_stamps": True,
            },
        ),
        (
            "English",
            True,
            {
                "audio": "/a.wav",
                "language": "English",
                "return_time_stamps": True,
            },
        ),
        ("English", False, {"audio": "/a.wav", "language": "English"}),
    ),
)
def test_transcribe_qwen_mode_uses_exact_decode_call(
    language: str | None, timestamps: bool, expected: dict[str, object]
) -> None:
    model = FakeQwenModel()

    result = transcribe_qwen_mode(
        model, "/a.wav", language=language, timestamps=timestamps
    )

    assert result["text"] == "hello world"
    assert model.calls == [expected]


def test_parse_args_rejects_timestamps_without_forced_aligner() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-id",
                "Qwen/model",
                "--revision",
                "a" * 40,
                "--timestamps",
                "on",
            ]
        )


def test_parse_args_accepts_the_four_mode_controls() -> None:
    args = parse_args(
        [
            "--model-id",
            "Qwen/model",
            "--revision",
            "a" * 40,
            "--forced-aligner-id",
            "Qwen/aligner",
            "--forced-aligner-revision",
            "b" * 40,
            "--language",
            "English",
            "--timestamps",
            "on",
        ]
    )

    assert args.language == "English"
    assert args.timestamps == "on"
    assert args.forced_aligner_id == "Qwen/aligner"


class FakeCuda:
    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 123

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def is_available() -> bool:
        return True


class FakeTorch:
    __version__ = "test-torch"
    bfloat16 = "bfloat16"
    version = types.SimpleNamespace(hip="test-hip")
    cuda = FakeCuda


class FakeSoundfile:
    @staticmethod
    def info(path: str):
        return types.SimpleNamespace(duration=3.5)


class FakeParameter:
    device = types.SimpleNamespace(type="cuda")
    dtype = FakeTorch.bfloat16


class FakeMainModel:
    def __init__(self) -> None:
        self.outputs: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []

    def parameters(self):
        return iter([FakeParameter()])

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs[kwargs["audio"]]
        if isinstance(output, Exception):
            raise output
        return [output]


def _install_fake_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> FakeMainModel:
    model = FakeMainModel()
    outputs = {
        "/a.wav": FakeQwenOutput(),
        "/b.wav": FakeEmptyQwenOutput(),
        "/c.wav": FakeQwenOutput(),
        "/d.wav": FakeQwenOutput(),
    }
    model.outputs = outputs

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundfile)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(__version__="test-transformers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        types.SimpleNamespace(Qwen3ASRModel=object),
    )
    monkeypatch.setattr(runner, "load_qwen", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        runner,
        "validate_rocm_bfloat16",
        lambda *_args: ("cuda:0 (ROCm)", "bfloat16"),
    )
    monkeypatch.setattr(
        runner,
        "audio_evidence_quality",
        lambda text, *, duration_seconds: {"status": "accepted", "text": text},
    )
    monkeypatch.setattr(
        runner,
        "finalize_result_metrics",
        lambda result, *, started, torch_module: result.update(
            elapsed_seconds=0.25, peak_memory_bytes=123
        ),
    )

    benchmark = __import__(
        "trec_video_2026.vqa_benchmark", fromlist=["source_fingerprint"]
    )
    monkeypatch.setattr(
        benchmark,
        "source_fingerprint",
        lambda path: types.SimpleNamespace(sha256=f"hash:{Path(path).name}"),
    )
    return model


def test_main_preserves_order_and_empty_failure_without_label_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _install_fake_main_dependencies(monkeypatch)
    payload = [
        {
            "run_id": "run-1",
            "condition_id": "arm",
            "video_id": video_id,
            "audio_path": f"/{letter}.wav",
            "mode": "transcription",
            "reference_answer": "private label",
        }
        for video_id, letter in zip(("v1", "v2", "v3", "v4"), "abcd")
    ]
    monkeypatch.setattr(runner.sys, "stdin", io.StringIO(json.dumps(payload)))

    assert (
        runner.main(
            [
                "--model-id",
                "Qwen/model",
                "--revision",
                "a" * 40,
                "--language",
                "auto",
                "--timestamps",
                "off",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert [row["video_id"] for row in output] == ["v1", "v2", "v3", "v4"]
    assert [row["status"] for row in output] == [
        "passed",
        "failed",
        "passed",
        "passed",
    ]
    assert output[0]["text"] == "hello world"
    assert output[1]["error_type"] == "ValueError"
    assert output[1]["error_message"] == "Qwen3-ASR returned empty text"
    assert all("reference_answer" not in row for row in output)
    assert all("correct_answer" not in row for row in output)
    assert model.calls == [
        {"audio": "/a.wav", "language": None},
        {"audio": "/b.wav", "language": None},
        {"audio": "/c.wav", "language": None},
        {"audio": "/d.wav", "language": None},
    ]
