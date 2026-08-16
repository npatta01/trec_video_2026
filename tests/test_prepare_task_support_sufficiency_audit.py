from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "prepare_task_support_sufficiency_audit.py"
spec = importlib.util.spec_from_file_location(
    "prepare_task_support_sufficiency_audit", SCRIPT
)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load preparation module from {SCRIPT}")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_full_panel_fixture(
    tmp_path: Path,
    row_count: int = 5,
    exposed: set[str] | None = None,
) -> dict[str, Any]:
    exposed = exposed or set()
    source = tmp_path / "source"
    output = tmp_path / "output"
    runner = tmp_path / "runner.py"
    runner.write_text("frozen runner\n")
    rows = [
        {
            "video_id": f"v{index:02d}",
            "question": f"What happens in row {index}?",
            "question_stratum": "temporal",
        }
        for index in range(row_count)
    ]
    _write(source / "protocol.json", {"rows": rows})
    _write(
        source / "maps.json",
        [
            {
                "video_id": row["video_id"],
                "map": "OVERVIEW | The event occurs.\nSCENE | 00:00.000-00:01.000 | The event occurs.",
                "status": "passed",
            }
            for row in rows
        ],
    )
    _write(
        source / "predictions.json",
        [
            {"video_id": row["video_id"], "answer": f"Answer {index}"}
            for index, row in enumerate(rows)
        ],
    )
    _write(
        source / "scores.json",
        {
            "rows": {
                "gemini-3.1-pro-native-av-4fps": [
                    {
                        "video_id": row["video_id"],
                        "question_stratum": row["question_stratum"],
                        "semantic_composite": (
                            0.30 + index * 0.001
                            if index < 24
                            else 0.80 if index < 32 else 0.50
                        ),
                    }
                    for index, row in enumerate(rows)
                ],
                "qwen35-4b-12f-whisper-large-v3": [
                    {
                        "video_id": row["video_id"],
                        "question_stratum": row["question_stratum"],
                        "semantic_composite": 0.99 - index * 0.001,
                    }
                    for index, row in enumerate(rows)
                ],
            }
        },
    )
    annotations = source / "annotations"
    for index, row in enumerate(rows):
        _write(
            annotations / f"{row['video_id']}.json",
            {"video_id": row["video_id"], "correct_answer": f"Reference {index}"},
        )
    _write(
        source / "exposure.json",
        {"video_ids": sorted(exposed), "label": "prior hard-audit exposure"},
    )
    config = tmp_path / "config.json"
    config_payload = {
        "schema_version": 1,
        "experiment_id": "task-support-test",
        "runner_path": str(runner),
        "source_artifacts": {
            "protocol": str(source / "protocol.json"),
            "maps": str(source / "maps.json"),
            "predictions": str(source / "predictions.json"),
            "scores": str(source / "scores.json"),
            "annotations": str(annotations),
        },
        "exposure_artifacts": {"hard_audit": str(source / "exposure.json")},
        "gate": {
            "low_audit_count": min(24, row_count),
            "control_audit_count": min(8, row_count),
            "minimum_per_role": 4,
        },
    }
    _write(config, config_payload)
    return {
        "config_path": config,
        "output": output,
    }


def write_auditable_fixture(tmp_path: Path) -> dict[str, Any]:
    fixture = write_full_panel_fixture(tmp_path, row_count=40)
    source = tmp_path / "source"
    protocol = json.loads((source / "protocol.json").read_text())
    rows = protocol["rows"]
    for index, row in enumerate(rows):
        row["queue"] = "low" if index < 24 else "control"
    _write(source / "protocol.json", protocol)
    return fixture


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def fill_audit_with_four_per_role(path: Path) -> None:
    payload = load(path)
    roles = (
        "answer_sufficient_failure",
        "insufficient_or_ambiguous",
        "supported_control",
    )
    for index, row in enumerate(payload["rows"]):
        row["adjudication"] = {
            "role": roles[index // 4] if index < 12 else "supported_control",
            "evidence_line_numbers": [1],
            "note": "fixture adjudication",
        }
    _write(path, payload)


def test_prepare_candidates_requires_existing_treatment_freeze(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    with pytest.raises(ValueError, match="treatment freeze"):
        prep.prepare_candidates(**fixture)


def test_candidate_sheet_keeps_exposed_rows_and_hides_labels(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5, exposed={"v01"})
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    rows = load(fixture["output"] / "candidate-sheet.json")["rows"]
    assert len(rows) == 5
    assert next(row for row in rows if row["video_id"] == "v01")["exposure_flags"]
    encoded = json.dumps(rows)
    assert "correct_answer" not in encoded
    assert "direct_prediction" not in encoded


def test_queue_and_selection_are_deterministic(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    template = load(fixture["output"] / "audit-template.json")
    assert [row["video_id"] for row in template["rows"][:3]] == [
        "v00",
        "v01",
        "v02",
    ]
    fill_audit_with_four_per_role(fixture["output"] / "manual-audit.json")
    prep.freeze_selection(**fixture)
    selection = load(fixture["output"] / "selection-freeze.json")
    assert selection["role_counts"] == {
        "answer_sufficient_failure": 4,
        "insufficient_or_ambiguous": 4,
        "supported_control": 4,
    }
    assert len(load(fixture["output"] / "protocol.json")["rows"]) == 12


def test_queue_uses_only_gemini_condition_and_thresholds(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    template = load(fixture["output"] / "audit-template.json")
    assert template["low_score_count"] == 24
    assert template["control_count"] == 8
    assert all(row["queue"] == "low_score" for row in template["rows"][:24])
    assert all(row["queue"] == "supported_control" for row in template["rows"][24:])
    assert [row["video_id"] for row in template["rows"][24:27]] == [
        "v24",
        "v25",
        "v26",
    ]


def test_real_two_condition_score_artifact_uses_gemini_rows(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "vqa" / "task_support_sufficiency_audit_issue35.json"
    config = load(config_path)
    score_spec = config["source_artifacts"]["scores"]
    score_path = Path(score_spec["path"])
    if not score_path.is_absolute():
        score_path = ROOT / score_path
    if not score_path.exists():
        pytest.skip("canonical private full-panel score artifact is unavailable")
    selected = prep._load_scores(score_path, condition=config["score_condition"])
    raw = load(score_path)
    gemini_rows = {row["video_id"]: row for row in raw["rows"]["gemini-3.1-pro-native-av-4fps"]}
    qwen_rows = {row["video_id"]: row for row in raw["rows"]["qwen35-4b-12f-whisper-large-v3"]}
    assert selected["B24dV_uori4"]["semantic_composite"] == gemini_rows["B24dV_uori4"]["semantic_composite"]
    assert selected["B24dV_uori4"]["semantic_composite"] != qwen_rows["B24dV_uori4"]["semantic_composite"]
    prep.freeze_treatment(config_path=config_path, output=tmp_path / "real")
    prep.prepare_candidates(config_path=config_path, output=tmp_path / "real")
    template = load(tmp_path / "real" / "audit-template.json")
    assert template["low_score_count"] == 24
    assert template["control_count"] == 8
    assert all(
        gemini_rows[row["video_id"]]["question_stratum"] in {"temporal", "audio_or_text"}
        for row in template["rows"]
    )


def test_real_queue_order_matches_independent_gemini_score_strata(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "vqa" / "task_support_sufficiency_audit_issue35.json"
    config = load(config_path)
    score_spec = config["source_artifacts"]["scores"]
    score_path = Path(score_spec["path"])
    if not score_path.is_absolute():
        score_path = ROOT / score_path
    if not score_path.exists():
        pytest.skip("canonical private full-panel score artifact is unavailable")

    raw = load(score_path)
    gemini_rows = raw["rows"]["gemini-3.1-pro-native-av-4fps"]
    eligible = [
        row
        for row in gemini_rows
        if row.get("question_stratum") in {"temporal", "audio_or_text"}
    ]
    expected_low = [
        row["video_id"]
        for row in sorted(
            (row for row in eligible if row["semantic_composite"] <= 0.45),
            key=lambda row: (row["semantic_composite"], row["video_id"]),
        )[:24]
    ]
    expected_control = [
        row["video_id"]
        for row in sorted(
            (row for row in eligible if row["semantic_composite"] >= 0.70),
            key=lambda row: (-row["semantic_composite"], row["video_id"]),
        )[:8]
    ]

    prep.freeze_treatment(config_path=config_path, output=tmp_path / "real")
    prep.prepare_candidates(config_path=config_path, output=tmp_path / "real")
    template = load(tmp_path / "real" / "audit-template.json")
    actual_low = [row["video_id"] for row in template["rows"] if row["queue"] == "low_score"]
    actual_control = [
        row["video_id"] for row in template["rows"] if row["queue"] == "supported_control"
    ]
    assert actual_low == expected_low
    assert actual_control == expected_control


def test_real_exposure_manifests_have_frozen_counts(tmp_path: Path) -> None:
    del tmp_path
    config_path = ROOT / "configs" / "vqa" / "task_support_sufficiency_audit_issue35.json"
    if not Path("/home/npatta01/data/competitions/trec_video_2026/data").exists():
        pytest.skip("canonical private dataset is unavailable")
    config = load(config_path)
    counts = {name: len(ids) for name, ids in prep._exposure_ids(config, config_path).items()}
    assert counts == {
        "hard_audit_30": 30,
        "grounded_v2_24": 24,
        "grounded_v3_11": 11,
        "multi_interpretation_12": 12,
        "seven_row_canary": 7,
        "scope_control": 7,
        "synthesis_followup": 7,
    }


def test_manual_audit_is_verdict_only_and_cannot_change_provider_inputs(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    manual_path = fixture["output"] / "manual-audit.json"
    manual = load(manual_path)
    fill_audit_with_four_per_role(manual_path)
    manual = load(manual_path)
    manual["rows"][0]["question"] = "LABEL LEAK QUESTION"
    manual["rows"][0]["exposure_flags"] = {"hard_audit": True}
    manual["rows"][0]["frozen_map"] = {"lines": [{"line_number": 1, "claim": "LABEL LEAK CLAIM"}]}
    _write(manual_path, manual)
    prep.freeze_selection(**fixture)
    protocol = load(fixture["output"] / "protocol.json")
    assert protocol["rows"][0]["question"] == "What happens in row 0?"
    assert "LABEL LEAK" not in json.dumps(protocol)


def test_reference_or_question_issue_is_audited_but_never_selected(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    manual_path = fixture["output"] / "manual-audit.json"
    fill_audit_with_four_per_role(manual_path)
    manual = load(manual_path)
    manual["rows"][0]["adjudication"]["role"] = "reference_or_question_issue"
    manual["rows"][12]["adjudication"]["role"] = "answer_sufficient_failure"
    _write(manual_path, manual)
    prep.freeze_selection(**fixture)
    selection = load(fixture["output"] / "selection-freeze.json")
    assert selection["audited_role_counts"]["reference_or_question_issue"] == 1
    assert "v00" not in selection["selected_rows"]


def test_promotion_gate_is_bound_and_tamper_rejected(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    prep.freeze_treatment(**fixture)
    treatment_path = fixture["output"] / "treatment-freeze.json"
    treatment = load(treatment_path)
    treatment["promotion_gate"]["target_rescues_min"] = 99
    _write(treatment_path, treatment)
    with pytest.raises(ValueError, match="promotion gate"):
        prep.prepare_candidates(**fixture)


def test_source_hash_drift_refuses_candidate_preparation(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    prep.freeze_treatment(**fixture)
    source = tmp_path / "source" / "maps.json"
    source.write_text(source.read_text().replace("The event occurs.", "Changed."))
    with pytest.raises(ValueError, match="source hash"):
        prep.prepare_candidates(**fixture)


def test_treatment_tampering_refuses_candidate_preparation(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    prep.freeze_treatment(**fixture)
    treatment_path = fixture["output"] / "treatment-freeze.json"
    treatment = load(treatment_path)
    treatment["runner_sha256"] = "0" * 64
    _write(treatment_path, treatment)
    with pytest.raises(ValueError, match="runner hash"):
        prep.prepare_candidates(**fixture)


def test_duplicate_protocol_ids_are_rejected_before_treatment_freeze(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    protocol_path = tmp_path / "source" / "protocol.json"
    protocol = load(protocol_path)
    protocol["rows"][1]["video_id"] = protocol["rows"][0]["video_id"]
    _write(protocol_path, protocol)
    with pytest.raises(ValueError, match="duplicate"):
        prep.freeze_treatment(**fixture)


def test_selection_rejects_invalid_line_and_insufficient_roles(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    manual_path = fixture["output"] / "manual-audit.json"
    manual = load(manual_path)
    for index, row in enumerate(manual["rows"]):
        row["adjudication"] = {
            "role": "answer_sufficient_failure",
            "evidence_line_numbers": [999] if index == 0 else [1],
        }
    _write(manual_path, manual)
    with pytest.raises(ValueError, match="line"):
        prep.freeze_selection(**fixture)


def test_selection_rejects_fewer_than_four_rows_in_any_role(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    manual_path = fixture["output"] / "manual-audit.json"
    manual = load(manual_path)
    for row in manual["rows"]:
        row["adjudication"] = {"role": "answer_sufficient_failure", "evidence_line_numbers": [1]}
    _write(manual_path, manual)
    with pytest.raises(ValueError, match="fewer than four"):
        prep.freeze_selection(**fixture)


def test_exposure_counts_are_stratified_without_exclusion(tmp_path: Path) -> None:
    fixture = write_auditable_fixture(tmp_path)
    source = tmp_path / "source"
    _write(source / "exposure.json", {"video_ids": ["v00", "v31"]})
    prep.freeze_treatment(**fixture)
    prep.prepare_candidates(**fixture)
    candidate_freeze = load(fixture["output"] / "candidate-sheet-freeze.json")
    assert candidate_freeze["eligible_row_count"] == 40
    assert candidate_freeze["exposure_counts"]["hard_audit"] == 2


def test_treatment_and_selection_are_write_once(tmp_path: Path) -> None:
    fixture = write_full_panel_fixture(tmp_path, row_count=5)
    prep.freeze_treatment(**fixture)
    with pytest.raises(ValueError, match="immutable"):
        prep.freeze_treatment(**fixture)
