from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_vqa_mc_gemini_direct_full as v3  # noqa: E402


def _candidate_module():
    path = SCRIPTS / "run_vqa_mc_gemini38_static_full.py"
    assert path.is_file(), "Gemini 3.8 static runner is not implemented"
    spec = importlib.util.spec_from_file_location("g38_static_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(runner):
    return runner.build_request(
        row={
            "query_id": 7,
            "video_id": "clip.webm",
            "question": "What happens?",
            "options": ["First", "Second", "Third"],
            "input_sha256": "a" * 64,
        },
        upload={
            "file_uri": "https://files.invalid/clip",
            "mime_type": "video/webm",
        },
    ).model_dump(mode="json", by_alias=True, exclude_none=True)


def test_gemini38_request_differs_from_v3_only_by_model() -> None:
    control = _request(importlib.reload(v3))
    candidate = _candidate_module().configured_runner()
    treatment = _request(candidate)

    assert control["model"] == "gemini-3.7-flash"
    assert treatment["model"] == "gemini-3.8-flash"
    control["model"] = treatment["model"]
    assert json.dumps(treatment, sort_keys=True) == json.dumps(control, sort_keys=True)


def test_gemini38_configuration_rejects_v3_protocol_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate_module()
    runner = importlib.reload(v3)
    monkeypatch.setattr(runner, "VIDEO_FPS", 1.0)

    with pytest.raises(RuntimeError, match="V3 protocol drift"):
        candidate.configured_runner(runner)
