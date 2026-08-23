from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_scoring", ROOT / "scripts/score_vqa_mc_poc20.py"
)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_scoring_keeps_qwen_views_one_family_vote(tmp_path: Path) -> None:
    requests = _write(
        tmp_path / "requests.json",
        {
            "rows": [
                {"video_id": "q1", "stratum": "temporal"},
                {"video_id": "q2", "stratum": "speech"},
            ]
        },
    )
    labels = _write(
        tmp_path / "labels.json",
        {
            "rows": [
                {"query_id": "q1", "correct_option_id": "A"},
                {"query_id": "q2", "correct_option_id": "D"},
            ]
        },
    )
    gemini = _write(
        tmp_path / "gemini.json",
        {
            "rows": [
                {"video_id": "q1", "variant_kind": "canonical", "status": "passed", "ranking": list("DCBA")},
                {"video_id": "q2", "variant_kind": "canonical", "status": "passed", "ranking": list("DCBA")},
            ]
        },
    )
    qwen = _write(
        tmp_path / "qwen.json",
        {
            "rows": [
                {"video_id": "q1", "variant_kind": "canonical", "condition_id": "video-plus-transcript", "status": "passed", "ranking": list("ABCD")},
                {"video_id": "q1", "variant_kind": "canonical", "condition_id": "native-video", "status": "passed", "ranking": list("BACD")},
                {"video_id": "q2", "variant_kind": "canonical", "condition_id": "video-plus-transcript", "status": "passed", "ranking": list("ABCD")},
                {"video_id": "q2", "variant_kind": "canonical", "condition_id": "native-video", "status": "failed", "ranking": None},
            ]
        },
    )

    result = scorer.score(
        requests_path=requests,
        labels_path=labels,
        gemini_path=gemini,
        qwen_path=qwen,
    )

    assert result["metrics"]["gemini"]["valid_count"] == 2
    assert result["metrics"]["qwen_dual_view"]["valid_count"] == 1
    assert result["metrics"]["equal_family_borda"]["valid_count"] == 1
    assert result["rows"][0]["qwen_dual_view_ranking"] == list("ABCD")
    assert result["rows"][0]["equal_family_borda_ranking"] == list("ABCD")
    assert result["rows"][1]["qwen_dual_view_ranking"] is None
    assert result["qwen_view_top1_agreement_count"] == 0
    assert result["qwen_view_comparable_count"] == 1


def test_scoring_rejects_label_or_prediction_identity_drift(tmp_path: Path) -> None:
    requests = _write(
        tmp_path / "requests.json",
        {"rows": [{"video_id": "q1", "stratum": "temporal"}]},
    )
    labels = _write(
        tmp_path / "labels.json",
        {"rows": [{"query_id": "different", "correct_option_id": "A"}]},
    )
    gemini = _write(tmp_path / "gemini.json", {"rows": []})

    try:
        scorer.score(
            requests_path=requests,
            labels_path=labels,
            gemini_path=gemini,
            qwen_path=None,
        )
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("identity drift was accepted")
