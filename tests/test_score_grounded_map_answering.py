from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "score_grounded_map_answering.py"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("score_grounded_map_answering", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


RULE = {
    "minimum_target_rescues": 6,
    "maximum_control_regressions": 1,
    "require_all_answer_claims_citation_supported": True,
    "require_no_unnecessary_verification_routes": True,
    "require_changed_row_audit_complete": True,
}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "run"
    annotations = tmp_path / "annotations"
    baseline_path = tmp_path / "baseline.json"
    ids = [f"target-{index:02d}" for index in range(12)] + [
        f"control-{index:02d}" for index in range(12)
    ]
    baseline_rows = [
        {
            "video_id": video_id,
            "condition_id": "direct",
            "status": "passed",
            "answer": f"Answer for {video_id}",
        }
        for video_id in ids
    ]
    _write(baseline_path, baseline_rows)
    protocol = {
        "schema_version": 1,
        "experiment_id": "issue35-test",
        "source_artifacts": {
            "direct_predictions": {
                "path": str(baseline_path),
                "sha256": _sha(baseline_path),
            }
        },
        "rows": [
            {
                "video_id": video_id,
                "pair_id": f"pair-{index % 12:02d}",
                "role": "target" if index < 12 else "control",
                "stratum": "other",
                "question": f"Question for {video_id}?",
                "evidence": [
                    {
                        "evidence_id": "E01",
                        "map_line_number": 1,
                        "text": f"SCENE | Evidence for {video_id}",
                    }
                ],
            }
            for index, video_id in enumerate(ids)
        ],
    }
    _write(root / "protocol.json", protocol)
    for index, video_id in enumerate(ids):
        answer = (
            "Changed target answer."
            if video_id == "target-00"
            else f"Answer for {video_id}"
        )
        _write(
            root / "rows" / video_id / "terminal.json",
            {
                "video_id": video_id,
                "pair_id": f"pair-{index % 12:02d}",
                "role": "target" if index < 12 else "control",
                "stratum": "other",
                "question": f"Question for {video_id}?",
                "status": "passed",
                "question_interpretation": "Interpret the question.",
                "answer_claims": [
                    {"claim": answer, "evidence_ids": ["E01"]}
                ],
                "conflicts_or_missing_qualifiers": [],
                "needs_verification": False,
                "answer": answer,
                "raw_response_sha256": "0" * 64,
            },
        )
        _write(root / "rows" / video_id / "request.json", {"video_id": video_id})
        (root / "rows" / video_id / "raw-response.txt").write_text(
            json.dumps({"answer": answer}), encoding="utf-8"
        )
        _write(root / "rows" / video_id / "metadata.json", {"total_token_count": 1})
        _write(root / "rows" / video_id / "parsed.json", {"answer": answer})
        _write(
            annotations / f"{video_id}.json",
            {
                "video_url": f"https://example.test/{video_id}",
                "question": f"Question for {video_id}?",
                "correct_answer": f"Answer for {video_id}",
                "incorrect_answers": [],
            },
        )
    freeze_rows = []
    for video_id in ids:
        terminal = root / "rows" / video_id / "terminal.json"
        freeze_rows.append(
            {
                "video_id": video_id,
                "status": "passed",
                "terminal_sha256": _sha(terminal),
            }
        )
    _write(
        root / "predictions-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": "issue35-test",
            "status": "frozen",
            "row_count": 24,
            "passed_count": 24,
            "failed_count": 0,
            "protocol_sha256": _sha(root / "protocol.json"),
            "rows": freeze_rows,
        },
    )
    _write(
        root / "preinference-gate-freeze.json",
        {
            "schema_version": 1,
            "experiment_id": "issue35-test",
            "protocol_sha256": _sha(root / "protocol.json"),
            "rule": RULE,
        },
    )
    MODULE.freeze_evidence(root)
    return root, baseline_path, annotations


def _audit_row(
    video_id: str,
    role: str,
    assessment: str,
    *,
    citation_support: bool = True,
    verification_assessment: str = "not_applicable",
) -> dict[str, Any]:
    failed = assessment in {"not_rescued", "regressed"}
    return {
        "video_id": video_id,
        "role": role,
        "assessment": assessment,
        "citation_support": citation_support,
        "verification_assessment": verification_assessment,
        "remaining_failure_mode": "reasoning_failure" if failed else None,
        "note": "Literal audit note.",
    }


def test_load_frozen_predictions_rejects_mutated_terminal(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, _baseline, _annotations = _run_fixture(tmp_path)
    terminal = root / "rows" / "target-00" / "terminal.json"
    terminal.write_text(terminal.read_text() + "\n")

    with pytest.raises(ValueError, match="frozen (terminal|row artifact) changed"):
        MODULE.load_frozen_predictions(root)


def test_load_frozen_predictions_rejects_mutated_auxiliary_evidence(
    tmp_path: Path,
) -> None:
    assert MODULE is not None
    root, _baseline, _annotations = _run_fixture(tmp_path)
    metadata = root / "rows" / "target-00" / "metadata.json"
    metadata.write_text(metadata.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen row artifact changed"):
        MODULE.load_frozen_predictions(root)


def test_classify_gate_requires_exactly_every_changed_row() -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    changed = [
        {
            "video_id": "target-00",
            "role": "target",
            "baseline_prediction": "old",
            "treatment_prediction": "new",
        }
    ]

    with pytest.raises(ValueError, match="manual audit IDs must exactly match"):
        MODULE.classify_gate(changed, {"rows": []}, RULE)
    with pytest.raises(ValueError, match="manual audit IDs must exactly match"):
        MODULE.classify_gate(
            changed,
            {
                "rows": [
                    _audit_row("target-00", "target", "rescued"),
                    _audit_row("control-00", "control", "preserved"),
                ]
            },
            RULE,
        )


def test_classify_gate_counts_rescues_regressions_and_support() -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    changed = [
        {
            "video_id": f"target-{index:02d}",
            "role": "target",
            "baseline_prediction": "old",
            "treatment_prediction": "new",
        }
        for index in range(6)
    ] + [
        {
            "video_id": "control-00",
            "role": "control",
            "baseline_prediction": "old",
            "treatment_prediction": "new",
        }
    ]
    audit = {
        "rows": [
            _audit_row(f"target-{index:02d}", "target", "rescued")
            for index in range(6)
        ]
        + [_audit_row("control-00", "control", "regressed")]
    }

    gate = MODULE.classify_gate(changed, audit, RULE)

    assert gate["passed"] is True
    assert gate["target_rescues"] == 6
    assert gate["control_regressions"] == 1
    assert gate["unsupported_citation_rows"] == []

    audit["rows"][0]["citation_support"] = False
    assert MODULE.classify_gate(changed, audit, RULE)["passed"] is False


def test_score_rejects_question_mismatch_after_freeze(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, baseline, annotations = _run_fixture(tmp_path)
    annotation = annotations / "target-00.json"
    payload = json.loads(annotation.read_text())
    payload["question"] = "Different question?"
    _write(annotation, payload)

    with pytest.raises(ValueError, match="question mismatch"):
        MODULE.score(
            root=root,
            baseline_predictions_path=baseline,
            annotation_root=annotations,
            bert_model=tmp_path / "bert",
            meteor_fn=lambda _prediction, _reference: 0.5,
            bertscore_fn=lambda predictions, _references, _model: [0.9]
            * len(predictions),
            sts_fn=lambda _prediction, _reference: 0.4,
        )


def test_score_accepts_annotation_question_trailing_whitespace(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, baseline, annotations = _run_fixture(tmp_path)
    annotation = annotations / "target-00.json"
    payload = json.loads(annotation.read_text())
    payload["question"] += " "
    _write(annotation, payload)

    result = MODULE.score(
        root=root,
        baseline_predictions_path=baseline,
        annotation_root=annotations,
        bert_model=tmp_path / "bert",
        meteor_fn=lambda _prediction, _reference: 0.5,
        bertscore_fn=lambda predictions, _references, _model: [0.9]
        * len(predictions),
        sts_fn=lambda _prediction, _reference: 0.4,
    )

    assert result["panel"]["row_count"] == 24


def test_score_reports_changed_rows_metrics_and_pending_manual_gate(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, baseline, annotations = _run_fixture(tmp_path)

    result = MODULE.score(
        root=root,
        baseline_predictions_path=baseline,
        annotation_root=annotations,
        bert_model=tmp_path / "bert",
        meteor_fn=lambda _prediction, _reference: 0.5,
        bertscore_fn=lambda predictions, _references, _model: [0.9]
        * len(predictions),
        sts_fn=lambda _prediction, _reference: 0.4,
    )

    assert result["panel"] == {
        "row_count": 24,
        "target_count": 12,
        "control_count": 12,
        "passed_treatment_count": 24,
    }
    assert result["changed_row_count"] == 1
    assert result["changed_rows"][0]["video_id"] == "target-00"
    assert result["conditions"]["direct_baseline"]["semantic_composite"] == pytest.approx(
        0.6
    )
    assert result["conditions"]["grounded_treatment"]["semantic_composite"] == pytest.approx(
        0.6
    )
    assert result["gate"] == {
        "status": "pending_manual_audit",
        "passed": None,
        "changed_rows_requiring_inspection": 1,
    }


def test_score_records_unavailable_sts_without_substituting_zero(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, baseline, annotations = _run_fixture(tmp_path)

    result = MODULE.score(
        root=root,
        baseline_predictions_path=baseline,
        annotation_root=annotations,
        bert_model=tmp_path / "bert",
        meteor_fn=lambda _prediction, _reference: 0.5,
        bertscore_fn=lambda predictions, _references, _model: [0.9]
        * len(predictions),
        sts_fn=lambda _prediction, _reference: pytest.fail("unexpected STS call"),
        sts_unavailable_reason="frozen endpoint unavailable from two egress paths",
    )

    for condition in result["conditions"].values():
        assert condition["sts"] is None
        assert condition["semantic_composite"] is None
        assert condition["meteor"] == 0.5
        assert condition["bertscore_f1"] == 0.9
    assert result["paired_metric_deltas"]["all"]["sts"] is None
    assert result["paired_metric_deltas"]["all"]["semantic_composite"] is None
    assert result["metric_notes"]["sts_status"] == "unavailable"
    assert result["metric_notes"]["sts_unavailable_reason"].startswith(
        "frozen endpoint unavailable"
    )


def test_persistent_sts_cache_writes_each_miss_and_reuses_hits(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    calls: list[tuple[str, str]] = []

    def live(prediction: str, reference: str) -> float:
        calls.append((prediction, reference))
        return 0.625

    cache_path = tmp_path / "sts-cache.json"
    first = MODULE.PersistentStsCache(cache_path, live_fn=live)
    assert first("A prediction", "A reference") == 0.625
    assert calls == [("A prediction", "A reference")]
    assert json.loads(cache_path.read_text())["values"]

    second = MODULE.PersistentStsCache(cache_path, live_fn=live)
    assert second("A prediction", "A reference") == 0.625
    assert calls == [("A prediction", "A reference")]
    assert second.stats == {"hits": 1, "misses": 0, "seeded": 0}


def test_seed_baseline_sts_cache_requires_bound_score_and_populates_panel(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    root, baseline, annotations = _run_fixture(tmp_path)
    ids = [row["video_id"] for row in json.loads(baseline.read_text())]
    prior_score = tmp_path / "prior-score.json"
    _write(
        prior_score,
        {
            "inputs": {
                "annotation_root_sha256": MODULE.sha256_path(annotations),
                "prediction_files": {
                    "gemini-3.1-pro-native-av-4fps": {"sha256": _sha(baseline)}
                },
            },
            "rows": {
                "gemini-3.1-pro-native-av-4fps": [
                    {"video_id": video_id, "sts": 0.75} for video_id in ids
                ]
            },
        },
    )
    cache = MODULE.PersistentStsCache(
        tmp_path / "sts-cache.json",
        live_fn=lambda _prediction, _reference: pytest.fail("unexpected live STS"),
    )

    MODULE.seed_baseline_sts_cache(
        cache=cache,
        prior_score_path=prior_score,
        baseline_predictions_path=baseline,
        annotation_root=annotations,
        video_ids=ids,
    )

    first = json.loads(baseline.read_text())[0]
    reference = json.loads((annotations / f"{first['video_id']}.json").read_text())[
        "correct_answer"
    ]
    assert cache(first["answer"], reference) == 0.75
    assert cache.stats == {"hits": 1, "misses": 0, "seeded": 24}

    payload = json.loads(prior_score.read_text())
    payload["inputs"]["prediction_files"]["gemini-3.1-pro-native-av-4fps"][
        "sha256"
    ] = "0" * 64
    _write(prior_score, payload)
    with pytest.raises(ValueError, match="baseline score prediction binding"):
        MODULE.seed_baseline_sts_cache(
            cache=cache,
            prior_score_path=prior_score,
            baseline_predictions_path=baseline,
            annotation_root=annotations,
            video_ids=ids,
        )


def test_cli_accepts_bound_baseline_score_and_persistent_sts_cache(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map scorer is not implemented"
    args = MODULE.build_parser().parse_args(
        [
            "--root",
            str(tmp_path / "run"),
            "--baseline-predictions",
            str(tmp_path / "baseline.json"),
            "--baseline-score",
            str(tmp_path / "baseline-score.json"),
            "--annotations",
            str(tmp_path / "annotations"),
            "--bert-model",
            str(tmp_path / "bert"),
            "--sts-cache",
            str(tmp_path / "sts-cache.json"),
            "--sts-unavailable-reason",
            "endpoint outage",
            "--output",
            str(tmp_path / "score.json"),
        ]
    )

    assert args.baseline_score == tmp_path / "baseline-score.json"
    assert args.sts_cache == tmp_path / "sts-cache.json"
    assert args.sts_unavailable_reason == "endpoint outage"
