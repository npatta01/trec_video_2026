import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "score_multi_interpretation_answer_election.py"
SPEC = importlib.util.spec_from_file_location("multi_score", MODULE_PATH)
assert SPEC and SPEC.loader
multi_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi_score)


ROLES = ["premise_variance"] * 4 + ["insufficiency_safeguard"] * 4 + [
    "stability_control"
] * 4


def _rows():
    rows = []
    for index, role in enumerate(ROLES):
        verification = role == "insufficiency_safeguard"
        rows.append(
            {
                "video_id": f"v{index:02d}",
                "role": role,
                "single": {"status": "passed", "needs_verification": verification, "answer": "NEEDS VERIFICATION" if verification else "single"},
                "multi": {"status": "passed", "needs_verification": verification, "answer": "NEEDS VERIFICATION" if verification else "multi"},
            }
        )
    return rows


def _audit():
    rows = []
    for index, role in enumerate(ROLES):
        assessment = "correct_answer" if role != "insufficiency_safeguard" else "correct_verification"
        rows.append(
            {
                "video_id": f"v{index:02d}",
                "role": role,
                "single_assessment": assessment,
                "multi_assessment": assessment,
                "candidate_correctness": [assessment == "correct_answer"] * 2 + [assessment == "correct_verification"],
                "citation_support_single": True,
                "citation_support_multi": True,
                "note": "inspected frozen evidence and all candidates",
            }
        )
    return {"rows": rows}


def test_gate_passes_exact_predeclared_boundary():
    audit = _audit()
    frozen = _rows()
    for index in (0, 1):
        audit["rows"][index]["single_assessment"] = "unnecessary_verification"
        frozen[index]["single"] = {
            "status": "passed",
            "needs_verification": True,
            "answer": "NEEDS VERIFICATION",
        }
    audit["rows"][3]["multi_assessment"] = "incorrect_answer"
    result = multi_score.classify_gate(frozen, audit, multi_score.PROMOTION_GATE)
    assert result["passed"] is True
    assert result["premise_correct_multi"] == 3
    assert result["premise_improvements_over_single"] == 2
    assert result["oracle_any_candidate_correct"] == 12


def test_gate_fails_control_regression_and_unsupported_answer():
    audit = _audit()
    audit["rows"][8]["multi_assessment"] = "incorrect_answer"
    audit["rows"][8]["citation_support_multi"] = False
    result = multi_score.classify_gate(_rows(), audit, multi_score.PROMOTION_GATE)
    assert result["passed"] is False
    assert result["multi_controls_preserved"] == 3
    assert result["unsupported_emitted_answers"] == ["multi:v08"]


def test_gate_rejects_assessment_inconsistent_with_frozen_route():
    audit = _audit()
    audit["rows"][4]["multi_assessment"] = "correct_answer"
    with pytest.raises(ValueError, match="contradicts frozen route"):
        multi_score.classify_gate(_rows(), audit, multi_score.PROMOTION_GATE)


def test_gate_requires_all_candidate_judgments_and_notes():
    audit = _audit()
    audit["rows"][0]["candidate_correctness"] = [True, False]
    with pytest.raises(ValueError, match="three candidate"):
        multi_score.classify_gate(_rows(), audit, multi_score.PROMOTION_GATE)


def test_role_aggregates_are_separate_from_all_rows():
    scored = [
        {"role": role, "meteor": float(index), "bertscore_f1": float(index) / 10}
        for index, role in enumerate(ROLES)
    ]
    aggregates = multi_score.aggregate_metrics(scored)
    assert set(aggregates) == {"all", *set(ROLES)}
    assert aggregates["premise_variance"]["row_count"] == 4
    assert aggregates["all"]["row_count"] == 12
    assert aggregates["all"]["sts"] is None
    assert aggregates["all"]["semantic_composite"] is None


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_refuses_labels_before_combined_freeze(tmp_path: Path):
    root = tmp_path / "run"
    annotations = tmp_path / "annotations"
    _write(annotations / "v00.json", {"correct_answer": "reference"})
    with pytest.raises(ValueError, match="integrity"):
        multi_score.load_frozen_rows(root, annotations)


def test_audit_freeze_rejects_postfreeze_tampering(tmp_path: Path):
    root = tmp_path / "run"
    audit_path = root / "manual-audit.json"
    _write(audit_path, _audit())
    multi_score.freeze_audit(root, audit_path)
    multi_score.verify_audit_freeze(root, audit_path)
    payload = json.loads(audit_path.read_text())
    payload["rows"][0]["note"] = "changed"
    _write(audit_path, payload)
    with pytest.raises(ValueError, match="audit freeze"):
        multi_score.verify_audit_freeze(root, audit_path)


def test_metric_runtime_binds_model_view_and_upstream_snapshot(tmp_path: Path):
    view = tmp_path / "view"
    upstream = tmp_path / "upstream"
    (view / "config.json").parent.mkdir(parents=True)
    (view / "config.json").write_text("view")
    (upstream / "config.json").parent.mkdir(parents=True)
    (upstream / "config.json").write_text("upstream")
    runtime = multi_score.metric_runtime(
        view,
        upstream_snapshot=upstream,
        versions={"bert_score_module": "fixture"},
    )
    assert runtime["bert_model_view_sha256"] == multi_score.sha256_path(view)
    assert runtime["bert_upstream_snapshot_sha256"] == multi_score.sha256_path(upstream)
    assert runtime["versions"] == {"bert_score_module": "fixture"}
