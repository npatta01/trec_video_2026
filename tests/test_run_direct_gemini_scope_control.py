from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_direct_gemini_scope_control.py"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("scope_control", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


def test_parse_answer_requires_scope_evidence_answer_and_uncertainty() -> None:
    assert MODULE is not None, "scope-control runner is not implemented"
    parsed = MODULE.parse_answer(
        "SCOPE: temporal=repetition_pattern,whole_video; target=action,outcome\n"
        "EVIDENCE: 1.0-3.0, 5.0-7.0 | The action repeats.\n"
        "ANSWER: He repeats the jump several times.\n"
        "UNCERTAINTY: none\n"
    )

    assert parsed["temporal_scope"] == ["repetition_pattern", "whole_video"]
    assert parsed["targets"] == ["action", "outcome"]
    assert parsed["answer"] == "He repeats the jump several times."
    assert parsed["evidence"][0]["timestamps"] == "1.0-3.0, 5.0-7.0"


def test_parse_answer_rejects_unknown_scope() -> None:
    assert MODULE is not None, "scope-control runner is not implemented"
    with pytest.raises(ValueError, match="unknown temporal scope"):
        MODULE.parse_answer(
            "SCOPE: temporal=nearby; target=action\n"
            "EVIDENCE: 1-2 | event\nANSWER: result\nUNCERTAINTY: none\n"
        )


def test_prompt_is_scope_aware_and_contains_no_reference() -> None:
    assert MODULE is not None, "scope-control runner is not implemented"
    prompt = MODULE.answer_prompt("What happens next?")

    assert "complete video" in prompt
    assert "repetition_pattern" in prompt
    assert "whole_video" in prompt
    assert "Do not assume" in prompt
    assert "reference answer" not in prompt.lower()
