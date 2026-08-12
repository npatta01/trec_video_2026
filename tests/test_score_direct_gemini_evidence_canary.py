from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "score_direct_gemini_evidence_canary.py"
SPEC = importlib.util.spec_from_file_location("score_direct_canary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_token_f1_uses_bag_overlap() -> None:
    assert MODULE.token_f1("put it down", "put it down") == 1.0
    assert MODULE.token_f1("drop it", "put it down") == pytest.approx(0.4)


def test_load_rows_rejects_changed_frozen_prediction(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    annotation_root = tmp_path / "annotations"
    row_root = run_root / "rows" / "v1"
    row_root.mkdir(parents=True)
    annotation_root.mkdir()
    complete = {
        "video_id": "v1",
        "question": "What happens?",
        "category": "temporal",
        "direct_answer": "A",
        "timeline_answer": "A",
        "frame_answer": "A",
    }
    (row_root / "complete.json").write_text(json.dumps(complete))
    (annotation_root / "v1.json").write_text(
        json.dumps({"question": "What happens?", "correct_answer": "A"})
    )
    (run_root / "predictions-freeze.json").write_text(
        json.dumps(
            {
                "row_count": 7,
                "rows": [{"video_id": "v1", "complete_sha256": "wrong"}],
            }
        )
    )

    with pytest.raises(ValueError, match="Frozen prediction changed"):
        MODULE._load_rows(run_root, annotation_root)
