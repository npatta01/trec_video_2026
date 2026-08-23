from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_adaptive_scoring",
    ROOT / "scripts/score_vqa_mc_adaptive_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    requests = _write(
        tmp_path / "requests.json",
        {
            "rows": [
                {
                    "video_id": "q1",
                    "stratum": "counting",
                    "duration_seconds": 10.0,
                },
                {
                    "video_id": "q2",
                    "stratum": "speech",
                    "duration_seconds": 8.0,
                },
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
    control = _write(
        tmp_path / "control.json",
        {
            "rows": [
                {
                    "video_id": "q1",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("ABCD"),
                },
                {
                    "video_id": "q2",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("DCBA"),
                },
            ]
        },
    )
    stage1 = _write(
        tmp_path / "stage1.json",
        {
            "rows": [
                {
                    "video_id": "q1",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("BACD"),
                },
                {
                    "video_id": "q2",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("DCBA"),
                },
            ]
        },
    )
    final = _write(
        tmp_path / "final.json",
        {
            "rows": [
                {
                    "video_id": "q1",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("ABCD"),
                    "final_source": "stage2",
                },
                {
                    "video_id": "q2",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("ABCD"),
                    "final_source": "stage2",
                },
            ]
        },
    )
    inspections = _write(
        tmp_path / "inspections.json",
        {
            "rows": [
                {
                    "video_id": "q1",
                    "inspection": {
                        "start_seconds": 2.0,
                        "end_seconds": 5.0,
                        "fps": 8.0,
                        "resolution": "high",
                        "evidence_gap": "rapid_or_repeated_action",
                        "evidence_question": "Count the jumps.",
                        "source": "model",
                    },
                },
                {
                    "video_id": "q2",
                    "inspection": {
                        "start_seconds": 0.0,
                        "end_seconds": 8.0,
                        "fps": 4.0,
                        "resolution": "high",
                        "evidence_gap": "other",
                        "evidence_question": "Review the video.",
                        "source": "fallback",
                    },
                },
            ]
        },
    )
    return {
        "requests_path": requests,
        "labels_path": labels,
        "control_path": control,
        "stage1_path": stage1,
        "final_path": final,
        "inspections_path": inspections,
    }


def test_scorer_counts_repairs_regressions_and_plan_distribution(
    tmp_path: Path,
) -> None:
    result = scorer.score(**_artifacts(tmp_path))

    assert result["metrics"]["stage1"]["top1_correct"] == 1
    assert result["metrics"]["final"]["top1_correct"] == 1
    assert result["transitions_stage1_to_final"] == {
        "repaired": 1,
        "regressed": 1,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    assert result["inspection_distribution"]["fps"] == {"4.0": 1, "8.0": 1}
    assert result["inspection_distribution"]["resolution"] == {"high": 2}
    assert result["inspection_distribution"]["source"] == {
        "fallback": 1,
        "model": 1,
    }
    assert result["inspection_distribution"]["interval_fraction"] == {
        "0-25%": 0,
        "25-50%": 1,
        "50-75%": 0,
        "75-100%": 1,
    }
    assert result["final_source_counts"] == {"stage2": 2}
    assert result["decision_gate"]["promising_prototype"] is False


def test_scorer_penalizes_invalid_rankings_without_dropping_denominator(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    final = json.loads(paths["final_path"].read_text())
    final["rows"][1]["status"] = "failed"
    final["rows"][1]["ranking"] = None
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")

    result = scorer.score(**paths)

    assert result["metrics"]["final"]["row_count"] == 2
    assert result["metrics"]["final"]["valid_count"] == 1
    assert result["metrics"]["final"]["invalid_count"] == 1


def test_scorer_rejects_request_label_or_prediction_identity_drift(
    tmp_path: Path,
) -> None:
    paths = _artifacts(tmp_path)
    labels = json.loads(paths["labels_path"].read_text())
    labels["rows"][1]["query_id"] = "different"
    paths["labels_path"].write_text(json.dumps(labels), encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        scorer.score(**paths)


def test_scorer_rejects_duplicate_prediction_rows(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    stage1 = json.loads(paths["stage1_path"].read_text())
    stage1["rows"].append(dict(stage1["rows"][0]))
    paths["stage1_path"].write_text(json.dumps(stage1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        scorer.score(**paths)
