from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_vqa_deadline_whisper_full.py"
CONFIG = ROOT / "dstack" / "qwen36-whisper-full.dstack.yml"


def _load_runner():
    spec = importlib.util.spec_from_file_location("deadline_whisper_full", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_media(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path


def test_scan_media_is_unique_and_preserves_no_audio_status(tmp_path: Path) -> None:
    runner = _load_runner()
    _touch_media(tmp_path, "videos/webm-one.webm")
    _touch_media(tmp_path, "videos_original/mkv-two.mkv")
    _touch_media(tmp_path, "videos_original/no-audio.mkv")
    (tmp_path / "videos_original" / "ignored.info.json").write_text("{}")

    def probe(path: Path) -> dict[str, object]:
        return {"has_audio": path.stem != "no-audio", "duration_seconds": 12.5}

    rows = runner.scan_media(tmp_path, probe=probe)

    assert [row["video_id"] for row in rows] == [
        "mkv-two",
        "no-audio",
        "webm-one",
    ]
    assert rows[1]["has_audio"] is False
    assert rows[0]["duration_seconds"] == 12.5


def test_scan_media_uses_frozen_contract_video_ids(tmp_path: Path) -> None:
    runner = _load_runner()
    media = _touch_media(tmp_path, "media/8384.webm")
    (tmp_path / "media-contract.json").write_text(
        json.dumps(
            {
                "unique_media_count": 1,
                "media": [
                    {
                        "video_id": "8384.webm",
                        "path": "media/8384.webm",
                        "source_sha256": "a" * 64,
                        "size_bytes": media.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = runner.scan_media(
        tmp_path, probe=lambda path: {"has_audio": True, "duration_seconds": 1.0}
    )
    assert rows[0]["video_id"] == "8384.webm"
    assert rows[0]["source_sha256"] == "a" * 64


def test_ffmpeg_extraction_command_has_a_wav_output_suffix() -> None:
    runner = _load_runner()
    command = runner.build_ffmpeg_command(
        "/usr/bin/ffmpeg", Path("clip.mkv"), Path("work/clip.tmp.wav")
    )
    assert command[-1].endswith(".wav")
    assert command[command.index("-map") + 1] == "0:a:0"


def test_transcribe_passes_decoded_pcm_not_a_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    marker = object()
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *_args, **_kwargs: (marker, 16_000)),
    )
    from trec_video_2026 import vqa_whisper

    def fake_transcribe(_asr, audio, *, task):
        assert audio is marker
        assert task == "translate"
        return {
            "text": "hello",
            "timestamped_text": "[00:00:00.000] hello",
            "chunks": [],
            "language": "en",
            "no_speech_probability": 0.0,
            "diagnostics": {},
        }

    monkeypatch.setattr(vqa_whisper, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(
        vqa_whisper,
        "transcript_quality",
        lambda *_args, **_kwargs: {"status": "accepted"},
    )

    result = runner._transcribe(object(), tmp_path / "audio.wav", 1.0)

    assert result["quality"]["status"] == "accepted"


def test_load_resumable_results_skips_completed_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    result_path = tmp_path / "whisper-results.jsonl"
    result_path.write_text(
        "\n".join(
            [
                json.dumps({"video_id": "already-done", "status": "passed"}),
                json.dumps({"video_id": "no-audio", "status": "no-audio-stream"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = runner.load_resumable_results(result_path)
    assert sorted(loaded) == ["already-done", "no-audio"]

    result_path.write_text(
        result_path.read_text(encoding="utf-8")
        + json.dumps({"video_id": "already-done", "status": "passed"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate video_id"):
        runner.load_resumable_results(result_path)


def test_accepted_adapter_has_exact_qwen_rows_shape() -> None:
    runner = _load_runner()
    rows = [
        {
            "video_id": "b",
            "status": "passed",
            "transcript": "ignored",
            "selected_transcript": "[00:00:00.000] second",
            "quality": {"accepted": True},
        },
        {
            "video_id": "a",
            "status": "passed",
            "transcript": "",
            "selected_transcript": "",
            "quality": {"accepted": False},
        },
        {"video_id": "c", "status": "no-audio-stream"},
    ]

    assert runner.build_accepted_adapter(rows) == {
        "rows": [{"video_id": "b", "transcript": "[00:00:00.000] second"}]
    }


def test_preview_uses_injected_probe_and_does_not_load_model(tmp_path: Path) -> None:
    runner = _load_runner()
    _touch_media(tmp_path, "videos/one.webm")
    _touch_media(tmp_path, "videos_original/two.mkv")

    preview = runner.preview_input(
        tmp_path,
        probe=lambda path: {"has_audio": path.stem == "one", "duration_seconds": 3.0},
    )

    assert preview["media_count"] == 2
    assert preview["no_audio_count"] == 1
    assert preview["audio_count"] == 1
    assert preview["total_duration_seconds"] == 6.0
    assert preview["model_id"] == runner.MODEL_ID
    assert preview["model_revision"] == runner.MODEL_REVISION


def test_dstack_config_pins_frozen_input_and_bounded_l40s_run() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    text = CONFIG.read_text(encoding="utf-8")
    assert config["resources"]["gpu"].startswith("L40S:1")
    assert config["max_duration"] == "8h"
    assert config["max_price"] == 1.0
    assert runner_constants()["input_uri"] in text
    assert runner_constants()["model_revision"] in text
    assert "openai/whisper-large-v3" in text
    assert "huggingface_hub" in text
    assert 'hf buckets sync "$HF_OUTPUT_URI" "$OUTPUT_DIR"' in text
    assert "  - WHISPER_LIMIT\n" in text
    assert "WHISPER_LIMIT:-" in text
    assert (
        'exec "$PYTHON_BIN" /dstack/run/scripts/run_vqa_deadline_whisper_full.py'
        not in text
    )


def runner_constants() -> dict[str, str]:
    runner = _load_runner()
    return {
        "input_uri": runner.INPUT_URI,
        "model_revision": runner.MODEL_REVISION,
    }
