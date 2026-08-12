from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_direct_gemini_evidence_canary.py"
SPEC = importlib.util.spec_from_file_location("direct_canary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_timeline() -> None:
    events = MODULE.parse_timeline("0-2 | starts\n2.0-4.5 | finishes\n", 4.5)
    assert events == [
        {"start_seconds": 0.0, "end_seconds": 2.0, "description": "starts"},
        {"start_seconds": 2.0, "end_seconds": 4.5, "description": "finishes"},
    ]


def test_parse_selection_requires_span_and_answer() -> None:
    parsed = MODULE.parse_selection("SPAN: 1-3 | relevant\nANSWER: result\n", 4.0)
    assert parsed["answer"] == "result"
    assert parsed["spans"][0]["reason"] == "relevant"
    with pytest.raises(ValueError):
        MODULE.parse_selection("ANSWER: result", 4.0)


def test_frame_timestamps_are_bounded_and_capped() -> None:
    spans = [
        {"start_seconds": 0.0, "end_seconds": 2.0},
        {"start_seconds": 2.0, "end_seconds": 5.0},
        {"start_seconds": 5.0, "end_seconds": 8.0},
    ]
    values = MODULE.frame_timestamps(spans, 8.0, maximum=5)
    assert len(values) == 5
    assert values == sorted(values)
    assert values[0] >= 0
    assert values[-1] < 8.0
