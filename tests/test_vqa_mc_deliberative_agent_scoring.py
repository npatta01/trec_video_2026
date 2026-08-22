from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_common import sha256_file


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_deliberative_scoring",
    ROOT / "scripts/score_vqa_mc_deliberative_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _prediction_rows(rankings: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": video_id,
            "variant_kind": "canonical",
            "status": "passed",
            "ranking": list(ranking),
        }
        for video_id, ranking in rankings.items()
    ]


def _solver(solver_id: str, ranking: str) -> dict[str, Any]:
    return {
        "solver_id": solver_id,
        "ranking": list(ranking),
        "observations": [
            {
                "observation_id": f"{solver_id.replace('_', '-')}-o1",
                "timestamp_seconds": 1.0,
                "literal_observation": "A visible event occurs.",
                "interpretation": "The event supports the ranking.",
            }
        ],
    }


def _planner(ranking: str, *, inspect: bool) -> dict[str, Any]:
    action = None
    if inspect:
        action = {
            "kind": "inspect_interval",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "fps": 8.0,
            "resolution": "high",
            "investigative_focus": "Resolve the visible event.",
            "expectations": [
                {"option_id": "A", "expected_observation": "One event."}
            ],
        }
    return {
        "ranking": list(ranking),
        "option_assessments": [],
        "remaining_uncertainty": "Small residual uncertainty.",
        "decision": "inspect" if inspect else "finish_search",
        "action": action,
    }


def _notebook(video_id: str, *, action_count: int) -> dict[str, Any]:
    notebook: dict[str, Any] = {
        "schema_version": 1,
        "video_id": video_id,
        "solvers": [_solver("solver_1", "BACD"), _solver("solver_2", "ABCD")],
        "rounds": [],
    }
    if action_count == 0:
        notebook["rounds"].append(
            {"round_index": 1, "planner": _planner("ABCD", inspect=False)}
        )
        citation = "solver-2-o1"
    else:
        action = _planner("DABC", inspect=True)
        notebook["rounds"].append(
            {
                "round_index": 1,
                "planner": action,
                "action": action["action"],
                "provenance": {
                    "view_id": f"{video_id}-round-1",
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                    "fps": 8.0,
                    "resolution": "high",
                    "estimated_input_tokens": 4544,
                },
                "observer": {
                    "view_id": f"{video_id}-round-1",
                    "observations": [
                        {
                            "observation_id": "round-1-o1",
                            "timestamp_seconds": 2.0,
                            "literal_observation": "The decisive event is visible.",
                            "interpretation": "This supports D.",
                        }
                    ],
                    "view_limitations": "The requested interval was shown.",
                    "focus_resolution": "present",
                },
            }
        )
        citation = "round-1-o1"
    notebook["judge"] = {
        "status": "passed",
        "record": {
            "ranking": list("ABCD" if video_id == "q1" else "DABC"),
            "decisive_observation_ids": [citation],
            "ambiguity_present": video_id == "q2",
        },
        "error_type": None,
    }
    return notebook


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    requests = _write(
        tmp_path / "requests.json",
        {
            "rows": [
                {"video_id": "q1", "stratum": "counting", "duration_seconds": 5.0},
                {"video_id": "q2", "stratum": "speech", "duration_seconds": 4.0},
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
        {"rows": _prediction_rows({"q1": "BACD", "q2": "DABC"})},
    )
    adaptive = _write(
        tmp_path / "adaptive.json",
        {"rows": _prediction_rows({"q1": "ABCD", "q2": "ADCB"})},
    )
    run_root = tmp_path / "run"
    anchor = _write(
        run_root / "anchor/predictions-freeze.json",
        {"rows": _prediction_rows({"q1": "BACD", "q2": "DABC"})},
    )
    blind = _write(
        run_root / "blind/predictions-freeze.json",
        {"rows": _prediction_rows({"q1": "ABCD", "q2": "ADCB"})},
    )
    final_rows: list[dict[str, Any]] = []
    notebook_index: list[dict[str, Any]] = []
    for index, (video_id, ranking, action_count) in enumerate(
        (("q1", "ABCD", 0), ("q2", "DABC", 1))
    ):
        relative = Path("notebooks/final") / f"{index:02d}.json"
        notebook_path = _write(
            run_root / relative,
            _notebook(video_id, action_count=action_count),
        )
        digest = sha256_file(notebook_path)
        notebook_index.append(
            {"video_id": video_id, "path": relative.as_posix(), "sha256": digest}
        )
        judge = json.loads(notebook_path.read_text())["judge"]["record"]
        final_rows.append(
            {
                "video_id": video_id,
                "variant_kind": "canonical",
                "status": "passed",
                "ranking": list(ranking),
                "final_source": "judge",
                "judge_status": "passed",
                "judge": judge,
                "notebook_path": relative.as_posix(),
                "notebook_sha256": digest,
                "call_count": 4 if action_count == 0 else 6,
                "observer_action_count": action_count,
            }
        )
    final_notebooks = _write(
        run_root / "notebooks/final-freeze.json",
        {"status": "notebooks_frozen", "rows": notebook_index},
    )
    final = _write(
        run_root / "final-predictions-freeze.json",
        {
            "status": "predictions_frozen",
            "source_paths": {
                "anchor": "anchor/predictions-freeze.json",
                "blind": "blind/predictions-freeze.json",
                "final_notebooks": "notebooks/final-freeze.json",
            },
            "source_sha256s": {
                "anchor": sha256_file(anchor),
                "blind": sha256_file(blind),
                "final_notebooks": sha256_file(final_notebooks),
            },
            "rows": final_rows,
        },
    )
    _write(
        run_root / "summary.json",
        {
            "status": "completed",
            "row_count": 2,
            "final_valid_count": 2,
            "request_count": 10,
            "usage_complete": True,
            "observed_cost_usd": 0.1,
        },
    )
    return {
        "requests_path": requests,
        "labels_path": labels,
        "control_path": control,
        "adaptive_path": adaptive,
        "run_root": run_root,
        "final_path": final,
    }


def _score(paths: dict[str, Path]) -> dict[str, Any]:
    return scorer.score(
        requests_path=paths["requests_path"],
        labels_path=paths["labels_path"],
        control_path=paths["control_path"],
        adaptive_path=paths["adaptive_path"],
        run_root=paths["run_root"],
        expected_control_top1=1,
    )


def test_score_reports_repairs_regressions_paths_and_integrity(
    tmp_path: Path,
) -> None:
    result = _score(_artifacts(tmp_path))

    assert result["metrics"]["final"]["top1_correct"] == 2
    assert result["transitions_control_to_final"]["repaired"] == 1
    assert result["transitions_control_to_final"]["regressed"] == 0
    assert result["path_distribution"] == {
        "zero_actions": 1,
        "one_action": 1,
        "two_actions": 0,
    }
    assert result["integrity"]["citation_valid_count"] == 2
    assert result["ambiguity_flag_count"] == 1


def test_gate_requires_improvement_repairs_and_known_hit_preservation(
    tmp_path: Path,
) -> None:
    result = _score(_artifacts(tmp_path))

    assert result["decision_gate"]["final_exceeds_control"] is True
    assert result["decision_gate"]["repairs_exceed_regressions"] is True
    assert result["decision_gate"]["known_regression_hits_preserved"] is True
    assert result["decision_gate"]["adaptive_paths"] is False
    assert result["decision_gate"]["promising_development_result"] is False


def test_score_rejects_identity_drift_and_duplicate_rows(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    labels = json.loads(paths["labels_path"].read_text())
    labels["rows"][1]["query_id"] = "different"
    paths["labels_path"].write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        _score(paths)

    paths = _artifacts(tmp_path / "duplicate")
    control = json.loads(paths["control_path"].read_text())
    control["rows"].append(dict(control["rows"][0]))
    paths["control_path"].write_text(json.dumps(control), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _score(paths)


def test_invalid_ranking_stays_in_denominator(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    final = json.loads(paths["final_path"].read_text())
    final["rows"][1]["ranking"] = ["A", "A", "B", "C"]
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")

    result = _score(paths)

    assert result["metrics"]["final"]["row_count"] == 2
    assert result["metrics"]["final"]["valid_count"] == 1
    assert result["metrics"]["final"]["invalid_count"] == 1


def test_score_rejects_wrong_frozen_control_artifact(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    control = json.loads(paths["control_path"].read_text())
    control["rows"][0]["status"] = "failed"
    control["rows"][0]["ranking"] = None
    paths["control_path"].write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen control"):
        _score(paths)


def test_score_rejects_missing_or_wrong_notebook_hash(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    final = json.loads(paths["final_path"].read_text())
    final["rows"][0].pop("notebook_sha256")
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="notebook hash"):
        _score(paths)


def test_missing_judge_citation_fails_structural_gate(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    final = json.loads(paths["final_path"].read_text())
    final["rows"][0]["judge"]["decisive_observation_ids"] = []
    paths["final_path"].write_text(json.dumps(final), encoding="utf-8")

    result = _score(paths)

    assert result["integrity"]["citation_valid_count"] == 1
    assert result["decision_gate"]["structural_integrity"] is False


def test_over_cap_fails_cost_gate(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    summary_path = paths["run_root"] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["observed_cost_usd"] = 0.1
    summary["combined_observed_cost_usd"] = 0.75000001
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = _score(paths)

    assert result["decision_gate"]["within_cost_cap"] is False


def test_sanitized_aggregate_excludes_private_row_data(tmp_path: Path) -> None:
    result = _score(_artifacts(tmp_path))
    sanitized = scorer.sanitized_aggregate(result)
    serialized = json.dumps(sanitized, sort_keys=True)

    assert "rows" not in sanitized
    assert "q1" not in serialized
    assert "correct_option_id" not in serialized
    assert "notebook_path" not in serialized
    assert "A visible event occurs" not in serialized


def test_private_audit_categorizes_transitions_and_ambiguity(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)
    result = _score(paths)

    audit = scorer.build_private_audit(
        requests_path=paths["requests_path"],
        run_root=paths["run_root"],
        scored=result,
    )

    assert audit["summary"]["control_repair_count"] == 1
    assert audit["summary"]["adaptive_repair_count"] == 1
    assert audit["summary"]["ambiguity_flag_count"] == 1
    assert audit["summary"]["final_miss_count"] == 0
    assert {row["query_id"] for row in audit["rows"]} == {"q1", "q2"}
