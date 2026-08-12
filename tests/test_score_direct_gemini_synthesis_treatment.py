from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "score_direct_gemini_synthesis_treatment.py"
)
spec = importlib.util.spec_from_file_location("score_synthesis", SCRIPT)
assert spec and spec.loader
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_grounding_audit_checks_spans_and_pattern_counts() -> None:
    synthesis = {
        "overview": {"support": [{"start_seconds": 0, "end_seconds": 4}]},
        "phases": [{"start_seconds": 0, "end_seconds": 4}],
        "patterns": [
            {"count": 2, "instances": [{"start_seconds": 1, "end_seconds": 2}]}
        ],
        "state_changes": [],
        "salient_av": [],
        "inferences": [],
    }
    assert MODULE.audit_grounding(synthesis, 4) == ["patterns[0].count"]


def test_grounding_audit_accepts_fully_grounded_observations() -> None:
    synthesis = {
        "overview": {"support": [{"start_seconds": 0, "end_seconds": 4}]},
        "phases": [{"start_seconds": 0, "end_seconds": 4}],
        "patterns": [
            {"count": 1, "instances": [{"start_seconds": 1, "end_seconds": 2}]}
        ],
        "state_changes": [{"support": [{"start_seconds": 2, "end_seconds": 3}]}],
        "salient_av": [],
        "inferences": [],
    }
    assert MODULE.audit_grounding(synthesis, 4) == []
