from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/score_vqa_frontier_map.py"
CATEGORIES = (
    "temporal",
    "counting",
    "causal",
    "speech",
    "sound",
    "ocr_detail",
    "other_global",
)
METRICS = (
    "normalized_exact_match",
    "token_f1",
    "meteor",
    "bertscore_f1",
    "sts",
    "semantic_composite",
    "ndcg",
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_vqa_frontier_map", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scorer = _load_module()
score_frontier_map = scorer.score_frontier_map


@dataclass(frozen=True)
class ScoringFixture:
    kwargs: dict
    panel_manifest: Path
    provider_results: Path
    provider_freeze: Path
    audit_freeze: Path
    annotations_root: Path
    qwen_baseline: Path
    qwen_sha256sums: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _annotation_root_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _panel_rows() -> list[dict]:
    rows = []
    ordinal = 1
    for category in CATEGORIES:
        for category_index in range(5):
            video_id = f"v{ordinal:03d}"
            rows.append(
                {
                    "ordinal": ordinal,
                    "video_id": video_id,
                    "category": category,
                    "media_sha256": hashlib.sha256(
                        f"synthetic-media-{video_id}".encode()
                    ).hexdigest(),
                    "question_stratum": category,
                    "within_category_index": category_index,
                }
            )
            ordinal += 1
    return rows


def _provider_row(
    *,
    provider: str,
    panel_row: dict,
    completed: bool,
    answer: str | None,
    failure_causes: list[str] | None = None,
) -> dict:
    row = {
        "provider": provider,
        "video_id": panel_row["video_id"],
        "category": panel_row["category"],
        "source_sha256": panel_row["media_sha256"],
        "cost_usd": 0.01,
        "raw_response": "PRIVATE RAW PROVIDER TEXT",
        "media_path": "/private/synthetic/media.mp4",
        "failure_causes": failure_causes or [],
    }
    if completed:
        row |= {
            "status": "completed",
            "answer": answer,
            "map_payload_present": True,
        }
    else:
        row |= {
            "status": "failed",
            "failure_code": "provider_timeout",
            "map_payload_present": False,
        }
    return row


def _verdict(
    provider_row: dict,
    *,
    view: str,
    usable_map_decision: str,
) -> dict:
    base = {
        "provider": provider_row["provider"],
        "video_id": provider_row["video_id"],
        "view": view,
        "notes": "PRIVATE REVIEWER NOTES",
    }
    if view == "blind":
        return base | {
            "event_coverage": 4,
            "order": 4,
            "entities": 4,
            "actions": 4,
            "state_changes": 4,
            "visible_text": "unverified",
            "timestamp_boundary_error": False,
            "answer_changing_false_claims": False,
        }
    return base | {
        "question_understanding": 4,
        "modality_choice": "visual",
        "evidence_presence": 4,
        "evidence_localization": 4,
        "support_citation": 4,
        "usable_map_decision": usable_map_decision,
        "confidence": 4,
    }


def make_scoring_fixture(
    tmp_path: Path,
    *,
    bad_failure_label: bool = False,
) -> ScoringFixture:
    panel_rows = _panel_rows()
    panel_manifest = tmp_path / "panel.json"
    panel_payload = {
        "schema_version": 1,
        "panel_id": "synthetic-frontier-map-panel",
        "evaluation_count": 35,
        "examples": panel_rows,
    }
    _write_json(panel_manifest, panel_payload)
    ordered_ids = [row["video_id"] for row in panel_rows]
    order_digest = _canonical_sha256(ordered_ids)
    source_digest = _canonical_sha256(
        [
            {"video_id": row["video_id"], "source_sha256": row["media_sha256"]}
            for row in panel_rows
        ]
    )

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    references = {}
    for index, row in enumerate(panel_rows):
        reference = f"reference answer {index:02d}"
        references[row["video_id"]] = reference
        _write_json(
            annotations / f"{row['video_id']}.json",
            {
                "question_stratum": row["category"],
                "correct_answer": reference,
                "reference": reference,
            },
        )
    annotation_digest = _annotation_root_sha256(annotations)

    provider_rows = []
    aware_verdicts = []
    blind_verdicts = []
    for provider in ("gemini", "reka"):
        for index, panel_row in enumerate(panel_rows):
            if provider == "gemini":
                completed = panel_row["within_category_index"] != 4
                answer = references[panel_row["video_id"]] if index % 3 else "wrong answer"
                failure_causes = ["wrong_modality"] if not completed else []
            else:
                completed = index % 7 != 0
                answer = references[panel_row["video_id"]] if index % 4 else "other answer"
                failure_causes = ["perception_failure"] if not completed else []
            if bad_failure_label and provider == "gemini" and not completed:
                failure_causes = ["not_an_allowed_failure_cause"]
            provider_row = _provider_row(
                provider=provider,
                panel_row=panel_row,
                completed=completed,
                answer=answer if completed else None,
                failure_causes=failure_causes,
            )
            provider_rows.append(provider_row)
            usable = "usable" if completed else "unusable"
            blind_verdicts.append(_verdict(provider_row, view="blind", usable_map_decision=usable))
            aware_verdicts.append(_verdict(provider_row, view="aware", usable_map_decision=usable))

    provider_results = tmp_path / "provider-results.json"
    _write_json(
        provider_results,
        {
            "schema_version": 1,
            "rows": provider_rows,
            "ordered_video_ids_sha256": order_digest,
            "source_hashes_sha256": source_digest,
        },
    )
    provider_freeze = tmp_path / "provider-freeze.json"
    _write_json(
        provider_freeze,
        {
            "schema_version": 1,
            "frozen": True,
            "provider_results_sha256": _sha256(provider_results),
            "panel_manifest_sha256": _sha256(panel_manifest),
            "ordered_video_ids_sha256": order_digest,
            "source_hashes_sha256": source_digest,
            "providers": ["gemini", "reka"],
            "row_count": 70,
        },
    )

    audit_root = tmp_path / "audit"
    _write_jsonl(audit_root / "verdicts.blind.jsonl", blind_verdicts)
    _write_jsonl(audit_root / "verdicts.aware.jsonl", aware_verdicts)
    audit_freeze = audit_root / "freeze.json"
    _write_json(
        audit_freeze,
        {
            "schema_version": 1,
            "frozen": True,
            "bindings": {
                "provider_rows": 70,
                "blind_verdicts": 70,
                "aware_verdicts": 70,
                "blind_verdicts_sha256": _sha256(audit_root / "verdicts.blind.jsonl"),
                "aware_verdicts_sha256": _sha256(audit_root / "verdicts.aware.jsonl"),
            },
            "rows": [
                {
                    "provider": row["provider"],
                    "video_id": row["video_id"],
                    "map_payload_present": row["map_payload_present"],
                    "failure_code": row.get("failure_code"),
                    "blind_verdict_sha256": "0" * 64,
                    "aware_verdict_sha256": "1" * 64,
                }
                for row in provider_rows
            ],
        },
    )

    qwen_dir = tmp_path / "qwen"
    qwen_baseline = qwen_dir / "qwen-baseline.json"
    qwen_rows = [
        {"condition_id": "qwen_baseline", "video_id": "extra-not-in-panel", "answer": "unused"}
    ]
    for index, panel_row in enumerate(panel_rows):
        qwen_rows.append(
            {
                "condition_id": "qwen_baseline",
                "video_id": panel_row["video_id"],
                "answer": references[panel_row["video_id"]] if index % 5 else "baseline wrong",
                "raw_response": "PRIVATE BASELINE RAW RESPONSE",
            }
        )
    _write_json(
        qwen_baseline,
        {
            "schema_version": 1,
            "condition_id": "qwen_baseline",
            "scorer_identity": scorer.EXPECTED_SCORER_IDENTITY,
            "bert_model_snapshot": scorer.EXPECTED_BERT_MODEL_SNAPSHOT,
            "annotation_root_sha256": annotation_digest,
            "rows": qwen_rows,
        },
    )
    qwen_sha256sums = qwen_dir / "SHA256SUMS"
    qwen_sha256sums.write_text(
        f"{_sha256(qwen_baseline)}  {qwen_baseline.name}\n",
        encoding="utf-8",
    )

    return ScoringFixture(
        kwargs={
            "provider_results": provider_results,
            "provider_freeze": provider_freeze,
            "audit_freeze": audit_freeze,
            "panel_manifest": panel_manifest,
            "annotations_root": annotations,
            "qwen_baseline": qwen_baseline,
            "qwen_sha256sums": qwen_sha256sums,
            "expected_baseline_source_sha256": _sha256(qwen_baseline),
            "bootstrap_iterations": 25,
        },
        panel_manifest=panel_manifest,
        provider_results=provider_results,
        provider_freeze=provider_freeze,
        audit_freeze=audit_freeze,
        annotations_root=annotations,
        qwen_baseline=qwen_baseline,
        qwen_sha256sums=qwen_sha256sums,
    )


def test_score_requires_prediction_and_audit_freezes_before_annotations(
    tmp_path: Path,
) -> None:
    fixture = make_scoring_fixture(tmp_path)
    (fixture.annotations_root / "v001.json").write_text("{not valid json")
    fixture.audit_freeze.unlink()

    with pytest.raises(ValueError, match="audit freeze"):
        score_frontier_map(**fixture.kwargs)


def test_sanitized_projection_has_all_metrics_and_no_private_text(tmp_path: Path) -> None:
    result = score_frontier_map(**make_scoring_fixture(tmp_path).kwargs)

    assert set(result["conditions"]) == {"gemini", "reka", "qwen_baseline"}
    for metric in METRICS:
        assert metric in result["conditions"]["gemini"]
    serialized = json.dumps(result).lower()
    for forbidden in (
        "correct_answer",
        "reference_answer",
        "reference answer",
        "raw_response",
        "private raw",
        "media_path",
        "private reviewer notes",
        "/private/",
    ):
        assert forbidden not in serialized


def test_score_verifies_qwen_sha256sums_before_annotation_load(tmp_path: Path) -> None:
    fixture = make_scoring_fixture(tmp_path)
    (fixture.annotations_root / "v001.json").write_text("{not valid json")
    fixture.qwen_sha256sums.unlink()

    with pytest.raises(ValueError, match="Qwen SHA256SUMS"):
        score_frontier_map(**fixture.kwargs)


def test_score_reports_failure_zero_completed_case_gate_and_paired_comparisons(
    tmp_path: Path,
) -> None:
    result = score_frontier_map(**make_scoring_fixture(tmp_path).kwargs)

    gemini = result["conditions"]["gemini"]
    assert gemini["row_count"] == 35
    assert gemini["completed_count"] == 28
    assert gemini["completed_cases"]["row_count"] == 28
    assert gemini["failure_zero"]["row_count"] == 35
    assert gemini["semantic_composite"] < gemini["completed_cases"]["semantic_composite"]
    assert gemini["ndcg_degenerate"] is True

    gate = result["map_gate"]["gemini"]
    assert gate["usable_count"] == 28
    assert gate["overall_28_of_35"] is True
    assert gate["category_min_3_of_5"] is True
    assert gate["passed"] is True
    assert gate["stricter_4_of_5_category_count"] == 7

    assert result["failure_counts"]["gemini"] == {"wrong_modality": 7}
    comparison = result["paired_comparisons"]["gemini_vs_qwen_baseline"]
    assert set(comparison["wins_losses_ties"]) == {"wins", "losses", "ties"}
    assert "semantic_composite" in comparison["metric_deltas"]
    assert comparison["bootstrap_intervals"]["semantic_composite"]["iterations"] == 25
    assert result["degenerate_ndcg"]["explicit"] is True


def test_score_rejects_unapproved_failure_cause_labels(tmp_path: Path) -> None:
    fixture = make_scoring_fixture(tmp_path, bad_failure_label=True)

    with pytest.raises(ValueError, match="failure cause"):
        score_frontier_map(**fixture.kwargs)


def test_score_rejects_baseline_identity_and_order_binding_errors(tmp_path: Path) -> None:
    fixture = make_scoring_fixture(tmp_path)
    baseline = json.loads(fixture.qwen_baseline.read_text())
    baseline["bert_model_snapshot"] = "wrong-snapshot"
    _write_json(fixture.qwen_baseline, baseline)
    fixture.qwen_sha256sums.write_text(
        f"{_sha256(fixture.qwen_baseline)}  {fixture.qwen_baseline.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BERT"):
        score_frontier_map(
            **(
                fixture.kwargs
                | {"expected_baseline_source_sha256": _sha256(fixture.qwen_baseline)}
            )
        )

    fixture = make_scoring_fixture(tmp_path / "order")
    provider_payload = json.loads(fixture.provider_results.read_text())
    provider_payload["rows"][0]["video_id"] = "out-of-panel-order"
    _write_json(fixture.provider_results, provider_payload)
    freeze_payload = json.loads(fixture.provider_freeze.read_text())
    freeze_payload["provider_results_sha256"] = _sha256(fixture.provider_results)
    _write_json(fixture.provider_freeze, freeze_payload)

    with pytest.raises(ValueError, match="ordered panel"):
        score_frontier_map(**fixture.kwargs)
