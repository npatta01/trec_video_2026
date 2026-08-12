from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_direct_gemini_synthesis_treatment.py"
)
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("synthesis_treatment", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


def test_synthesis_prompt_is_question_blind_and_pattern_focused() -> None:
    assert MODULE is not None, "synthesis runner is not implemented"
    prompt = MODULE.synthesis_prompt("0-2 | a person jumps")

    assert "Question:" not in prompt
    assert "whole-video" in prompt
    assert "repeated" in prompt
    assert "0-2 | a person jumps" in prompt
    assert "observation" in prompt
    assert "inference" in prompt


def test_synthesis_schema_requires_grounded_patterns() -> None:
    assert MODULE is not None, "synthesis runner is not implemented"
    schema = MODULE.SYNTHESIS_SCHEMA

    assert schema["required"] == [
        "overview",
        "phases",
        "patterns",
        "state_changes",
        "salient_av",
        "inferences",
        "uncertainties",
    ]
    pattern = schema["properties"]["patterns"]["items"]
    assert "instances" in pattern["required"]
    assert "count" in pattern["required"]


def test_answer_prompt_includes_complete_map_and_synthesis() -> None:
    assert MODULE is not None, "synthesis runner is not implemented"
    prompt = MODULE.answer_prompt(
        "What happens?", "0-2 | event", '{"overview":"activity"}'
    )

    assert "What happens?" in prompt
    assert "0-2 | event" in prompt
    assert '"overview":"activity"' in prompt
    assert "complete native video" in prompt
    assert "repetition_pattern" in prompt


def test_retry_preserves_each_upload_cleanup_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert MODULE is not None, "synthesis runner is not implemented"
    row_root = tmp_path / "rows" / "video-1"
    row_root.mkdir(parents=True)
    (row_root / "cleanup.json").write_text('{"uploaded_file_deletion":"succeeded"}\n')
    media = tmp_path / "video.webm"
    timeline = tmp_path / "timeline.txt"
    media.write_bytes(b"video")
    timeline.write_text("0-1 | event")
    protocol = {
        "rows": [
            {
                "video_id": "video-1",
                "category": "temporal",
                "question": "What happens?",
                "media_path": str(media),
                "media_sha256": MODULE._sha256(media),
                "timeline_path": str(timeline),
                "timeline_sha256": MODULE._sha256(timeline),
            }
        ]
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))

    client = SimpleNamespace(files=SimpleNamespace(delete=lambda **_: None))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(MODULE.genai, "Client", lambda **_: client)
    monkeypatch.setattr(MODULE, "_upload", lambda *_: SimpleNamespace(name="upload"))
    monkeypatch.setattr(MODULE, "_video_part", lambda *_: "video-part")
    monkeypatch.setattr(MODULE, "_call_json", lambda *_: ({"overview": {}}, "{}", {}))
    monkeypatch.setattr(
        MODULE,
        "_call",
        lambda *_: (
            "SCOPE: temporal=sequence; target=outcome\nEVIDENCE: 00:00-00:01 | event\nANSWER: result\nUNCERTAINTY: none",
            {},
        ),
    )

    MODULE.run_row(tmp_path, "video-1")

    assert (
        json.loads((row_root / "cleanup.json").read_text())["uploaded_file_deletion"]
        == "succeeded"
    )
    assert (
        json.loads((row_root / "cleanup-attempt-2.json").read_text())[
            "uploaded_file_deletion"
        ]
        == "succeeded"
    )
