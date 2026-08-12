from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "score_direct_gemini_scope_control.py"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("score_scope_control", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


def test_load_predictions_rejects_changed_frozen_row(tmp_path: Path) -> None:
    assert MODULE is not None, "scope scorer is not implemented"
    root = tmp_path / "run"
    row = root / "rows" / "v1"
    row.mkdir(parents=True)
    (row / "complete.json").write_text(
        json.dumps({"video_id": "v1", "question": "Q", "answer": "A"})
    )
    (root / "predictions-freeze.json").write_text(
        json.dumps(
            {"row_count": 7, "rows": [{"video_id": "v1", "complete_sha256": "bad"}]}
        )
    )

    with pytest.raises(ValueError, match="Frozen prediction changed"):
        MODULE.load_predictions(root)


def test_manual_gate_requires_repetition_and_no_supported_regressions() -> None:
    assert MODULE is not None, "scope scorer is not implemented"
    assessment = MODULE.manual_gate(
        {
            "FKsBrmJ2xSg": "wrong",
            "JTolYjiErBw": "wrong",
            "LSz_xBDBPNc": "wrong",
            "IRTTtnSRTyY": "good",
            "IVqpg_1aWXU": "good",
            "FBD7bvFnshE": "ambiguous",
            "Mj3kccs07fg": "ambiguous",
        }
    )

    assert assessment["passed"] is False
    assert assessment["repetition_repaired"] is False
    assert assessment["supported_answer_regressions"] == [
        "JTolYjiErBw",
        "LSz_xBDBPNc",
    ]
