from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_direct_gemini_synthesis_treatment.py"
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
