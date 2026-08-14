from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
RUNNER_SCRIPT = ROOT / "scripts" / "run_premise_tolerant_grounded_answering.py"
SCORER_SCRIPT = ROOT / "scripts" / "score_premise_tolerant_grounded_answering.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load(RUNNER_SCRIPT, "premise_runner")
SCORER = _load(SCORER_SCRIPT, "premise_scorer") if SCORER_SCRIPT.exists() else None

PANEL = {
    "fix_false_abstention": [f"fix-{index}" for index in range(5)],
    "detectable_insufficiency": [f"insuff-{index}" for index in range(2)],
    "known_limit": ["known-0"],
    "stability_control": [f"control-{index}" for index in range(3)],
}
IDS = [video_id for values in PANEL.values() for video_id in values]
GATE = {
    "minimum_false_abstention_rescues": 4,
    "required_detectable_insufficiency_verifications": 2,
    "required_stability_controls_preserved": 3,
    "maximum_unsupported_final_answers": 0,
    "require_full_manual_audit": True,
}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_answer(answer: str) -> dict[str, Any]:
    return {
        "question_interpretation": "Interpret the requested answer.",
        "premise_status": "supported",
        "answer_claims": [{"claim": answer, "evidence_ids": ["E01"]}],
        "conflicts_or_missing_qualifiers": [],
        "needs_verification": False,
        "answer": answer,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    source = tmp_path / "v2-protocol.json"
    _write(
        source,
        {
            "schema_version": 1,
            "experiment_id": "v2",
            "rows": [
                {
                    "video_id": video_id,
                    "question": f"Question for {video_id}?",
                    "evidence": [
                        {
                            "evidence_id": "E01",
                            "map_line_number": 1,
                            "text": f"SCENE | Evidence for {video_id}",
                        }
                    ],
                }
                for video_id in IDS
            ],
        },
    )
    selection = tmp_path / "selection.json"
    _write(
        selection,
        {
            "schema_version": 1,
            "experiment_id": "premise-v3-test",
            "source_protocol_sha256": _sha(source),
            "panel": PANEL,
            "gate": GATE,
        },
    )
    root = tmp_path / "run"
    RUNNER.prepare(source, selection, root)
    for video_id in IDS:
        answer = f"New answer for {video_id}."
        payload = _valid_answer(answer)
        RUNNER.run_row(
            root,
            video_id,
            call_fn=lambda _prompt, value=payload: (
                json.dumps(value),
                {"total_token_count": 5},
            ),
        )
    RUNNER.freeze(root)
    annotations = tmp_path / "annotations"
    direct_rows: list[dict[str, Any]] = []
    v2_rows: list[dict[str, Any]] = []
    for video_id in IDS:
        _write(
            annotations / f"{video_id}.json",
            {
                "video_url": f"https://example.test/{video_id}",
                "question": f"Question for {video_id}?",
                "correct_answer": f"Reference for {video_id}.",
                "incorrect_answers": [],
            },
        )
        direct_rows.append(
            {
                "video_id": video_id,
                "condition_id": "direct",
                "status": "passed",
                "answer": f"Direct answer for {video_id}.",
            }
        )
        v2_rows.append(
            {
                "video_id": video_id,
                "status": "passed",
                "question": f"Question for {video_id}?",
                "answer": f"Old answer for {video_id}.",
            }
        )
    direct = tmp_path / "direct.json"
    _write(direct, direct_rows)
    v2_root = tmp_path / "v2"
    for name in ("protocol.json", "predictions-freeze.json", "evidence-freeze.json"):
        _write(v2_root / name, {"name": name})
    return root, direct, annotations, v2_root, v2_rows


def _audit_row(video_id: str, role: str, assessment: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "role": role,
        "assessment": assessment,
        "citation_support": True,
        "note": "Literal manual audit note.",
    }


def _passing_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, video_id in enumerate(PANEL["fix_false_abstention"]):
        rows.append(
            _audit_row(
                video_id,
                "fix_false_abstention",
                "rescued" if index < 4 else "not_rescued",
            )
        )
    for video_id in PANEL["detectable_insufficiency"]:
        rows.append(
            _audit_row(video_id, "detectable_insufficiency", "correct_verification")
        )
    rows.append(_audit_row("known-0", "known_limit", "observed_answer"))
    for video_id in PANEL["stability_control"]:
        rows.append(_audit_row(video_id, "stability_control", "preserved"))
    return {"rows": rows}


def _passing_treatment_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _passing_audit()["rows"]:
        assessment = item["assessment"]
        rows.append(
            {
                "video_id": item["video_id"],
                "role": item["role"],
                "status": "passed",
                "needs_verification": assessment
                in {"not_rescued", "correct_verification", "observed_verification"},
            }
        )
    return rows


def test_load_frozen_predictions_rejects_auxiliary_artifact_drift(
    tmp_path: Path,
) -> None:
    assert SCORER is not None, "premise-tolerant scorer is not implemented"
    root, _direct, _annotations, _v2_root, _v2 = _fixture(tmp_path)
    metadata = root / "rows" / IDS[0] / "metadata.json"
    metadata.write_text(metadata.read_text() + " ")

    with pytest.raises(ValueError, match="frozen row artifact changed"):
        SCORER.load_frozen_predictions(root)


def test_load_frozen_predictions_assigns_roles_from_separate_gate(
    tmp_path: Path,
) -> None:
    assert SCORER is not None
    root, _direct, _annotations, _v2_root, _v2 = _fixture(tmp_path)

    rows = SCORER.load_frozen_predictions(root)

    assert len(rows) == 11
    assert {row["role"] for row in rows} == set(PANEL)
    assert all("role" not in row["protocol_row"] for row in rows)


@pytest.mark.parametrize("mutation", ["roles", "threshold", "integrity"])
def test_load_frozen_predictions_rejects_postfreeze_gate_drift(
    tmp_path: Path, mutation: str
) -> None:
    assert SCORER is not None
    root, _direct, _annotations, _v2_root, _v2 = _fixture(tmp_path)
    if mutation == "integrity":
        path = root / "gate-integrity.json"
        payload = json.loads(path.read_text())
        payload["preinference_gate_freeze_sha256"] = "0" * 64
    else:
        path = root / "preinference-gate-freeze.json"
        payload = json.loads(path.read_text())
        if mutation == "roles":
            (
                payload["panel"]["fix_false_abstention"][0],
                payload["panel"]["stability_control"][0],
            ) = (
                payload["panel"]["stability_control"][0],
                payload["panel"]["fix_false_abstention"][0],
            )
        else:
            payload["gate"]["minimum_false_abstention_rescues"] = 0
    _write(path, payload)

    with pytest.raises(ValueError, match="gate (freeze differs|integrity)"):
        SCORER.load_frozen_predictions(root)


def test_classify_gate_passes_exact_threshold_and_excludes_known_limit() -> None:
    assert SCORER is not None

    result = SCORER.classify_gate(
        _passing_treatment_rows(), _passing_audit(), {"panel": PANEL, "gate": GATE}
    )

    assert result["passed"] is True
    assert result["false_abstention_rescues"] == 4
    assert result["detectable_insufficiency_verifications"] == 2
    assert result["stability_controls_preserved"] == 3
    assert result["known_limit"] == {
        "video_id": "known-0",
        "assessment": "observed_answer",
    }


def test_classify_gate_records_detectable_insufficiency_contract_failure() -> None:
    assert SCORER is not None
    audit = _passing_audit()
    row = next(item for item in audit["rows"] if item["video_id"] == "insuff-0")
    row["assessment"] = "contract_failure"

    treatment = _passing_treatment_rows()
    next(row for row in treatment if row["video_id"] == "insuff-0")["status"] = (
        "contract_error"
    )
    result = SCORER.classify_gate(treatment, audit, {"panel": PANEL, "gate": GATE})

    assert result["passed"] is False
    assert result["detectable_insufficiency_verifications"] == 1


@pytest.mark.parametrize(
    ("video_id", "assessment", "field"),
    [
        ("fix-3", "not_rescued", "false_abstention_rescues"),
        ("insuff-1", "unsafe_answer", "detectable_insufficiency_verifications"),
        ("control-2", "regressed", "stability_controls_preserved"),
    ],
)
def test_classify_gate_fails_each_role_requirement(
    video_id: str, assessment: str, field: str
) -> None:
    assert SCORER is not None
    audit = _passing_audit()
    row = next(item for item in audit["rows"] if item["video_id"] == video_id)
    row["assessment"] = assessment

    treatment = _passing_treatment_rows()
    changed = next(row for row in treatment if row["video_id"] == video_id)
    if assessment in {"not_rescued", "regressed"}:
        changed["needs_verification"] = True
    elif assessment == "unsafe_answer":
        changed["needs_verification"] = False
    result = SCORER.classify_gate(treatment, audit, {"panel": PANEL, "gate": GATE})

    assert result["passed"] is False
    assert (
        result[field]
        < GATE[
            {
                "false_abstention_rescues": "minimum_false_abstention_rescues",
                "detectable_insufficiency_verifications": (
                    "required_detectable_insufficiency_verifications"
                ),
                "stability_controls_preserved": "required_stability_controls_preserved",
            }[field]
        ]
    )


def test_classify_gate_fails_unsupported_answer_and_requires_full_audit() -> None:
    assert SCORER is not None
    audit = _passing_audit()
    audit["rows"][0]["citation_support"] = False
    assert (
        SCORER.classify_gate(
            _passing_treatment_rows(), audit, {"panel": PANEL, "gate": GATE}
        )["passed"]
        is False
    )
    audit = _passing_audit()
    audit["rows"].pop()
    with pytest.raises(ValueError, match="exactly match panel IDs"):
        SCORER.classify_gate(
            _passing_treatment_rows(), audit, {"panel": PANEL, "gate": GATE}
        )


def test_classify_gate_allows_answered_wrong_and_ignores_nonanswer_citation() -> None:
    assert SCORER is not None
    audit = _passing_audit()
    negative = next(row for row in audit["rows"] if row["video_id"] == "fix-4")
    negative["citation_support"] = False
    treatment = _passing_treatment_rows()
    frozen = next(row for row in treatment if row["video_id"] == "fix-4")

    result = SCORER.classify_gate(treatment, audit, {"panel": PANEL, "gate": GATE})
    assert result["unsupported_final_answer_rows"] == []

    frozen["needs_verification"] = False
    negative["citation_support"] = True
    assert (
        SCORER.classify_gate(treatment, audit, {"panel": PANEL, "gate": GATE})["passed"]
        is True
    )


@pytest.mark.parametrize(
    ("video_id", "assessment", "status", "needs_verification"),
    [
        ("fix-0", "rescued", "passed", True),
        ("insuff-0", "correct_verification", "passed", False),
        ("control-0", "preserved", "contract_error", None),
    ],
)
def test_classify_gate_rejects_assessment_that_contradicts_frozen_treatment(
    video_id: str, assessment: str, status: str, needs_verification: bool | None
) -> None:
    assert SCORER is not None
    audit = _passing_audit()
    next(row for row in audit["rows"] if row["video_id"] == video_id)["assessment"] = (
        assessment
    )
    treatment = _passing_treatment_rows()
    row = next(item for item in treatment if item["video_id"] == video_id)
    row["status"] = status
    row["needs_verification"] = needs_verification

    with pytest.raises(ValueError, match="contradicts frozen treatment"):
        SCORER.classify_gate(treatment, audit, {"panel": PANEL, "gate": GATE})


def test_score_rejects_question_mismatch_after_freeze(tmp_path: Path) -> None:
    assert SCORER is not None
    root, direct, annotations, v2_root, v2_rows = _fixture(tmp_path)
    annotation = annotations / f"{IDS[0]}.json"
    payload = json.loads(annotation.read_text())
    payload["question"] = "A different question?"
    _write(annotation, payload)

    with pytest.raises(ValueError, match="question mismatch"):
        SCORER.score(
            root=root,
            direct_predictions_path=direct,
            annotation_root=annotations,
            v2_root=v2_root,
            v2_load_fn=lambda _root: v2_rows,
            bert_model=tmp_path / "model",
            meteor_fn=lambda _prediction, _reference: 0.5,
            bertscore_fn=lambda predictions, _references, _model: [0.6]
            * len(predictions),
        )


def test_score_requires_bound_v2_artifacts(tmp_path: Path) -> None:
    assert SCORER is not None
    root, direct, annotations, v2_root, v2_rows = _fixture(tmp_path)
    (v2_root / "evidence-freeze.json").unlink()

    with pytest.raises(ValueError, match="missing frozen v2 artifact"):
        SCORER.score(
            root=root,
            direct_predictions_path=direct,
            annotation_root=annotations,
            v2_root=v2_root,
            bert_model=tmp_path / "model",
            v2_load_fn=lambda _root: v2_rows,
            meteor_fn=lambda _prediction, _reference: 0.5,
            bertscore_fn=lambda predictions, _references, _model: [0.6]
            * len(predictions),
        )


def test_score_reports_three_conditions_and_null_sts(tmp_path: Path) -> None:
    assert SCORER is not None
    root, direct, annotations, v2_root, v2_rows = _fixture(tmp_path)

    result = SCORER.score(
        root=root,
        direct_predictions_path=direct,
        annotation_root=annotations,
        v2_root=v2_root,
        v2_load_fn=lambda _root: v2_rows,
        bert_model=tmp_path / "model",
        meteor_fn=lambda _prediction, _reference: 0.5,
        bertscore_fn=lambda predictions, _references, _model: [0.6] * len(predictions),
    )

    assert set(result["conditions"]) == {
        "direct_baseline",
        "grounded_v2",
        "premise_v3",
    }
    assert {
        key: result["conditions"]["premise_v3"][key]
        for key in ("meteor", "bertscore_f1", "sts", "semantic_composite")
    } == {
        "meteor": 0.5,
        "bertscore_f1": 0.6,
        "sts": None,
        "semantic_composite": None,
    }
    assert set(result["conditions"]["premise_v3"]["roles"]) == set(PANEL)
    assert result["gate"]["status"] == "pending_manual_audit"
    assert result["metric_notes"]["sts_status"] == "unavailable"
