from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_common import sha256_file


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_four_option_scorer",
    ROOT / "scripts/score_vqa_mc_four_option_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scorer
SPEC.loader.exec_module(scorer)


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _request_rows(count: int) -> list[dict[str, Any]]:
    return [
        {
            "query_id": f"video-{index:02d}",
            "video_id": f"video-{index:02d}",
            "stratum": ("counting" if index % 2 == 0 else "speech"),
            "question": f"Question {index}?",
            "duration_seconds": 6.0,
            "options": [
                {"option_id": option_id, "text": f"Choice {option_id}-{index}"}
                for option_id in "ABCD"
            ],
        }
        for index in range(count)
    ]


def _requests(tmp_path: Path, count: int = 20) -> Path:
    rows = _request_rows(count)
    return _write(
        tmp_path / f"requests-{count}.json",
        {
            "schema_version": 1,
            "experiment_id": "test-four-option-agent",
            "rows": rows,
            "variants": [
                {
                    "variant_id": f"{row['video_id']}:canonical",
                    "video_id": row["video_id"],
                    "variant_kind": "canonical",
                    "display_order": list("ABCD"),
                }
                for row in rows
            ],
        },
    )


def _labels(tmp_path: Path, count: int = 20) -> Path:
    return _write(
        tmp_path / f"labels-{count}.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "query_id": f"video-{index:02d}",
                    "correct_option_id": "A",
                }
                for index in range(count)
            ],
        },
    )


def _control(tmp_path: Path, *, correct: int = 15, count: int = 20) -> Path:
    return _write(
        tmp_path / f"control-{correct}-{count}.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "video_id": f"video-{index:02d}",
                    "variant_kind": "canonical",
                    "status": "passed",
                    "ranking": list("ABCD" if index < correct else "BACD"),
                }
                for index in range(count)
            ],
        },
    )


def _passing_integrity(count: int) -> dict[str, Any]:
    return {
        "observer_blind_count": count,
        "mandatory_observation_count": count,
        "complete_ledger_count": count,
        "citation_valid_count": count,
        "truncation_answer_count": 0,
        "protocol_identity_valid": True,
        "operational_invalid_count": 0,
    }


def _scored(
    *,
    row_count: int,
    valid_count: int,
    top1: int,
    repaired: int,
    regressed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "scored",
        "phase": "canary" if row_count == 5 else "full",
        "metrics": {
            "frozen_control": {
                "row_count": row_count,
                "valid_count": row_count,
                "top1_correct": 15 if row_count == 20 else 0,
            },
            "final": {
                "row_count": row_count,
                "valid_count": valid_count,
                "top1_correct": top1,
            },
        },
        "transitions_control_to_final": {
            "repaired": repaired,
            "regressed": regressed,
            "unchanged_correct": row_count - repaired - regressed,
            "unchanged_wrong": 0,
        },
        "integrity": _passing_integrity(row_count),
        "usage": {
            "combined_reserved_cost_usd": 1.4,
            "cost_cap_usd": 1.5,
        },
        "source_sha256s": {"requests": "1" * 64},
        "rows": [{"query_id": f"private-{index}"} for index in range(row_count)],
    }


def test_prepare_canary_selects_exactly_the_five_control_misses_without_labels(
    tmp_path: Path,
) -> None:
    output = tmp_path / "canary-requests.json"

    result = scorer.prepare_canary(
        requests_path=_requests(tmp_path),
        labels_path=_labels(tmp_path),
        control_path=_control(tmp_path, correct=15),
        output_path=output,
    )

    payload = json.loads(output.read_text())
    assert result["row_count"] == 5
    assert [row["video_id"] for row in payload["rows"]] == [
        f"video-{index:02d}" for index in range(15, 20)
    ]
    serialized = json.dumps(payload).casefold()
    assert "correct_option" not in serialized
    assert "reference_answer" not in serialized
    assert '"label"' not in serialized


def test_prepare_canary_rejects_a_control_other_than_15_of_20(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="frozen 15/20 control"):
        scorer.prepare_canary(
            requests_path=_requests(tmp_path),
            labels_path=_labels(tmp_path),
            control_path=_control(tmp_path, correct=14),
            output_path=tmp_path / "canary-requests.json",
        )


def test_canary_gate_requires_one_repair_and_all_behavioral_checks() -> None:
    scored = _scored(
        row_count=5, valid_count=5, top1=1, repaired=1, regressed=0
    )

    gate = scorer.build_canary_gate(scored)

    assert gate["continue_to_full"] is True
    scored["integrity"]["observer_blind_count"] = 4
    assert scorer.build_canary_gate(scored)["continue_to_full"] is False


def test_full_gate_requires_more_than_15_and_repairs_exceed_regressions() -> None:
    scored = _scored(
        row_count=20, valid_count=20, top1=16, repaired=2, regressed=1
    )

    gate = scorer.build_full_gate(scored)

    assert gate["candidate_for_fresh_evaluation"] is True
    scored["metrics"]["final"]["top1_correct"] = 15
    assert scorer.build_full_gate(scored)["candidate_for_fresh_evaluation"] is False


def _single_run_root(tmp_path: Path, *, valid: bool) -> Path:
    root = tmp_path / ("valid-run" if valid else "invalid-run")
    notebook = {
        "schema_version": 1,
        "video_id": "video-00",
        "protected_ranking": list("ABCD"),
        "initializer": {
            "ledger": [
                {
                    "option_id": option_id,
                    "status": "viable",
                    "supporting_evidence_ids": [],
                    "contradicting_evidence_ids": [],
                }
                for option_id in "ABCD"
            ],
            "literal_observations": [],
        },
        "rounds": [
            {
                "round_index": 1,
                "observer": {
                    "observations": [
                        {
                            "observation_id": "round-1-o1",
                            "timestamp_seconds": 2.0,
                            "literal_observation": "One jump is visible.",
                            "interpretation": "One completed cycle.",
                        }
                    ],
                    "provenance": {
                        "request_audit_path": "rows/00/round-1/observer/request-audit.json"
                    },
                },
                "updater": {
                    "ledger": [
                        {
                            "option_id": option_id,
                            "status": "viable",
                            "supporting_evidence_ids": (
                                ["round-1-o1"] if option_id == "A" else []
                            ),
                            "contradicting_evidence_ids": [],
                        }
                        for option_id in "ABCD"
                    ]
                },
            }
        ],
    }
    notebook_path = _write(root / "rows/00/final-notebook.json", notebook)
    _write(
        root / "rows/00/round-1/observer/request-audit.json",
        {"passed": True, "matched_value_hashes": []},
    )
    manifest_path = _write(root / "run-manifest.json", {"schema_version": 1})
    notebook_index = _write(
        root / "notebooks-freeze.json",
        {
            "rows": [
                {
                    "video_id": "video-00",
                    "path": "rows/00/final-notebook.json",
                    "sha256": sha256_file(notebook_path),
                }
            ]
        },
    )
    final_path = _write(
        root / "final-predictions-freeze.json",
        {
            "source_paths": {
                "manifest": "run-manifest.json",
                "notebooks": "notebooks-freeze.json",
            },
            "source_sha256s": {
                "manifest": sha256_file(manifest_path),
                "notebooks": sha256_file(notebook_index),
            },
            "rows": [
                {
                    "video_id": "video-00",
                    "variant_kind": "canonical",
                    "status": "passed" if valid else "operationally_invalid",
                    "ranking": list("ABCD"),
                    "final_source": "judge" if valid else "protected_direct_fallback",
                    "observer_action_count": 1,
                    "notebook_path": "rows/00/final-notebook.json",
                    "notebook_sha256": sha256_file(notebook_path),
                    "observer_audit_paths": [
                        "rows/00/round-1/observer/request-audit.json"
                    ],
                    "judge": (
                        {
                            "ranking": list("ABCD"),
                            "decisive_evidence_ids": ["round-1-o1"],
                            "protected_answer_action": "retained",
                            "direct_contradiction_evidence_ids": [],
                            "replacement_support_evidence_ids": [],
                        }
                        if valid
                        else None
                    ),
                }
            ],
        },
    )
    _write(
        root / "summary.json",
        {
            "status": "completed",
            "row_count": 1,
            "final_valid_count": 1 if valid else 0,
            "operational_invalid_count": 0 if valid else 1,
            "observed_cost_usd": 0.01,
            "formal_cost_usd": 0.1,
            "max_cost_usd": 1.5,
            "final_predictions_sha256": sha256_file(final_path),
        },
    )
    return root


def test_operational_fallback_is_invalid_not_wrong(tmp_path: Path) -> None:
    scored = scorer.score_phase(
        phase="canary",
        requests_path=_requests(tmp_path, count=1),
        labels_path=_labels(tmp_path, count=1),
        control_path=_control(tmp_path, correct=1, count=1),
        run_root=_single_run_root(tmp_path, valid=False),
    )

    assert scored["metrics"]["final"]["valid_count"] == 0
    assert scored["metrics"]["final"]["invalid_count"] == 1
    assert scored["integrity"]["operational_invalid_count"] == 1


def test_sanitized_aggregate_excludes_private_rows_and_raw_text() -> None:
    scored = _scored(
        row_count=20, valid_count=20, top1=16, repaired=2, regressed=1
    )

    public = scorer.sanitized_aggregate(scored)
    serialized = json.dumps(public)

    assert "rows" not in public
    assert "private-0" not in serialized
    assert "Question 0?" not in serialized
    assert "Choice A-0" not in serialized
    assert "raw_response" not in serialized
