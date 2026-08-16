from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
BENCHMARKS = ROOT / "data" / "vqa" / "benchmarks"
CANONICAL_AUDIT = BENCHMARKS / "task-support-sufficiency-audit-issue35-v1"
PROBE_BENCHMARK = BENCHMARKS / "dense-question-localized-evidence-issue35-v1"
SCRIPT = ROOT / "scripts" / "score_dense_question_localized_evidence.py"
spec = importlib.util.spec_from_file_location("dense_localized_scorer", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load scorer from {SCRIPT}")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def _arm(
    audit_index: int,
    role: str,
    fps: float,
    *,
    fact_present: bool = True,
    model_supported: bool = True,
    recovered: bool = True,
    timestamp_supported: bool = True,
    unsupported_observation: bool = False,
    ambiguous: bool = False,
    exposure: dict[str, bool] | None = None,
) -> dict[str, Any]:
    return {
        "audit_index": audit_index,
        "role": role,
        "fps": fps,
        "video_id": f"video-{audit_index}",
        "question": "what happened?",
        "reference": "reference",
        "fact_present_in_clip": fact_present,
        "model_fact_supported": model_supported,
        "recovered": recovered,
        "timestamp_supported": timestamp_supported,
        "unsupported_observation": unsupported_observation,
        "ambiguous_reference": ambiguous,
        "correction_acceptance_changed": False,
        "clip_sha256": "a" * 64,
        "map_line_numbers": [1],
        "reference_issue": "" if not ambiguous else "reference is not a concrete visual fact",
        "corrected_parsed_sha256": "c" * 64,
        "raw_response_sha256": "b" * 64,
        "evidence_timestamps": [],
        "exposure_flags": exposure
        or {
            "grounded_v2_24": False,
            "grounded_v3_11": False,
            "hard_audit_30": False,
            "multi_interpretation_12": False,
            "scope_control": False,
            "seven_row_canary": False,
            "synthesis_followup": False,
        },
        "reviewer_note": "fixture",
        "inspected": True,
    }


def paired_fixture(
    targets: list[tuple[bool, bool]],
    controls: list[tuple[bool, bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 1
    for role, pairs in (("target", targets), ("control", controls)):
        for four, twelve in pairs:
            for fps, recovered in ((4.0, four), (12.0, twelve)):
                rows.append(
                    _arm(
                        index,
                        role,
                        fps,
                        recovered=recovered,
                        fact_present=True,
                    )
                )
            index += 1
    return rows


def passing_gate_fixture() -> dict[str, Any]:
    return {
        "complete_terminal_pairs": True,
        "manual_inspection_complete": True,
        "target_auditable_count": 4,
        "target_4fps_recovered": 1,
        "target_12fps_recovered": 2,
        "incremental_recoveries": 2,
        "unsupported_observations_4fps": 0,
        "unsupported_observations_12fps": 0,
        "control_regressions_12fps": 0,
    }


def full_audit_fixture() -> dict[str, Any]:
    rows = [_arm(1, "target", fps, recovered=False) for fps in (4.0, 12.0)]
    return {
        "completed_at": "2026-08-16T00:00:00+00:00",
        "correction_freeze_sha256": "d" * 64,
        "correction_version": scorer.CORRECTION_VERSION,
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "inspection_scope": {
            "arms_inspected": 2,
            "auditable_target_policy": "fixture",
            "changed_acceptance_arms_inspected": 0,
            "clips_inspected": 1,
            "disagreements_inspected": 0,
            "source": "fixture",
        },
        "reference_correction_sha256": "e" * 64,
        "reference_freeze_sha256": "f" * 64,
        "reference_ledger_sha256": scorer.REFERENCE_LEDGER_SHA256,
        "reference_sha256": "0" * 64,
        "rows": rows,
        "schema_version": scorer.AUDIT_SCHEMA_VERSION,
        "source_prediction_freeze_sha256": "1" * 64,
        "status": "completed",
    }


def test_incremental_recovery_requires_4fps_false_and_12fps_true() -> None:
    rows = paired_fixture(targets=[(False, True), (True, True)], controls=[(True, True)])
    result = scorer.aggregate(rows)
    assert result["target_4fps_recovered"] == 1
    assert result["target_12fps_recovered"] == 2
    assert result["incremental_recoveries"] == 1


def test_gate_rejects_unsupported_observation_even_with_two_recoveries() -> None:
    result = passing_gate_fixture()
    result["unsupported_observations_12fps"] = 1
    assert scorer.classify_gate(result) == "FAIL"


def test_ambiguous_target_is_excluded_from_recovery_denominator_but_retained_in_completion() -> None:
    rows = paired_fixture(targets=[(False, True), (True, True)], controls=[])
    rows[2]["ambiguous_reference"] = True
    rows[3]["ambiguous_reference"] = True
    rows[2]["recovered"] = False
    rows[3]["recovered"] = False
    result = scorer.aggregate(rows)
    assert result["target_count"] == 2
    assert result["target_auditable_count"] == 1
    assert result["completed_arm_count"] == 4


def test_incomplete_pair_and_uninspected_arm_fail_validation() -> None:
    rows = paired_fixture(targets=[(False, True)], controls=[])
    rows[1]["inspected"] = False
    with pytest.raises(ValueError, match="inspection"):
        scorer.aggregate(rows)


def test_recovery_requires_supported_observation_and_supported_timestamp() -> None:
    rows = paired_fixture(targets=[(False, True)], controls=[])
    rows[1]["model_fact_supported"] = False
    with pytest.raises(ValueError, match="recovery"):
        scorer.aggregate(rows)
    rows = paired_fixture(targets=[(False, True)], controls=[])
    rows[1]["timestamp_supported"] = False
    with pytest.raises(ValueError, match="timestamp"):
        scorer.aggregate(rows)


def test_exposure_split_counts_are_reported_without_exclusion() -> None:
    rows = paired_fixture(targets=[(False, True)], controls=[])
    for row in rows:
        row["exposure_flags"] = {"prior_audit": True}
    result = scorer.aggregate(rows)
    assert result["exposure_splits"]["prior_audit"]["arms"] == 2
    assert result["exposure_splits"]["prior_audit"]["target_12fps_recovered"] == 1


def test_score_input_hash_drift_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hash|freeze|correction"):
        scorer.verify_score(tmp_path)


def test_canonical_corrected_v3_reference_chain_is_valid() -> None:
    bundle = scorer.verify_canonical_references(PROBE_BENCHMARK)
    assert bundle["reference_sha256"] == "db68cec9be89f4f89305b70578fbd7cb67893e0196c92f93143b5d154db6d3d2"
    assert bundle["correction_sha256"] == "4fb41decdca6d2c1df09ac8be2507d28eb02ca5427e607fde2e15ed8e29dcc4a"


@pytest.mark.parametrize("drift_name", ["manual-audit-corrected-v3.json", "corrected-audit-freeze-v3.json"])
def test_canonical_corrected_v3_reference_or_freeze_drift_is_rejected(tmp_path: Path, drift_name: str) -> None:
    root = tmp_path / "benchmarks"
    shutil.copytree(CANONICAL_AUDIT, root / "task-support-sufficiency-audit-issue35-v1")
    probe = root / "dense-question-localized-evidence-issue35-v1"
    probe.mkdir()
    drift_path = root / "task-support-sufficiency-audit-issue35-v1" / drift_name
    payload = json.loads(drift_path.read_text())
    if drift_name.startswith("manual"):
        payload["rows"][0]["verdict"] = "drift"
    else:
        payload["provider_calls_made"] = 1
    drift_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="canonical corrected-v3|location|hash|status"):
        scorer.verify_canonical_references(probe)


def test_run_local_or_explicit_reference_shadow_is_rejected(tmp_path: Path) -> None:
    shadow = tmp_path / "manual-audit-corrected-v3.json"
    shadow.write_bytes((CANONICAL_AUDIT / "manual-audit-corrected-v3.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact canonical"):
        scorer.verify_canonical_references(PROBE_BENCHMARK, shadow)


def test_exact_audit_schema_rejects_answer_metrics_and_nested_extras() -> None:
    audit = full_audit_fixture()
    scorer.validate_audit_schema(audit)
    answer_metrics = dict(audit)
    answer_metrics["answer_metrics"] = {}
    with pytest.raises(ValueError, match="schema drift"):
        scorer.validate_audit_schema(answer_metrics)
    row_extra = dict(audit)
    row_extra["rows"] = [dict(audit["rows"][0]), dict(audit["rows"][1])]
    row_extra["rows"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="schema drift"):
        scorer.validate_audit_schema(row_extra)
    nested_extra = dict(audit)
    nested_extra["rows"] = [dict(audit["rows"][0]), dict(audit["rows"][1])]
    nested_extra["rows"][0]["exposure_flags"] = dict(audit["rows"][0]["exposure_flags"])
    nested_extra["rows"][0]["exposure_flags"]["unexpected"] = False
    with pytest.raises(ValueError, match="schema drift"):
        scorer.validate_audit_schema(nested_extra)


def test_correction_acceptance_changed_is_bound_to_each_arm() -> None:
    rows = paired_fixture(targets=[(False, True)], controls=[])
    rows[0]["correction_acceptance_changed"] = True
    with pytest.raises(ValueError, match="invariant"):
        scorer.aggregate(rows)


@pytest.mark.parametrize("field", ["reference", "reference_issue", "fact_present_in_clip", "ambiguous_reference", "exposure_flags"])
def test_pair_shared_label_flip_is_rejected(field: str) -> None:
    rows = paired_fixture(targets=[(False, True)], controls=[])
    if field == "exposure_flags":
        rows[1][field] = dict(rows[1][field])
        rows[1][field]["grounded_v2_24"] = True
    elif field == "fact_present_in_clip":
        rows[1][field] = not rows[1][field]
        rows[1]["recovered"] = False
    elif field == "ambiguous_reference":
        rows[1][field] = True
        rows[1]["recovered"] = False
    else:
        rows[1][field] = "drift"
    with pytest.raises(ValueError, match="invariant"):
        scorer.aggregate(rows)


def test_real_frozen_score_rejects_treatment_input_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _copy_probe_with_patched_verified_inputs(tmp_path, monkeypatch)
    treatment = probe / "run-v22" / "treatment-freeze.json"
    treatment.write_bytes(treatment.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="write-once|drift|hash"):
        scorer.verify_score(probe)


def _copy_probe_with_patched_verified_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Copy the immutable inputs while bypassing path-bound runner verification.

    The freeze mutation tests intentionally exercise audit-only rejection.  The
    copied probe cannot pass the runner's continuation-path identity checks, so
    use the already verified real input objects while rebinding all file roots
    used by the scorer to the isolated copy.
    """

    benchmarks = tmp_path / "benchmarks"
    probe = benchmarks / "dense-question-localized-evidence-issue35-v1"
    shutil.copytree(PROBE_BENCHMARK, probe, symlinks=True)
    shutil.copytree(CANONICAL_AUDIT, benchmarks / "task-support-sufficiency-audit-issue35-v1")

    verified = scorer._verify_prediction_inputs(PROBE_BENCHMARK, PROBE_BENCHMARK / "run-v22")
    temp_verified = dict(verified)
    temp_verified.update(
        {
            "run_root": probe / "run-v22",
            "correction_root": probe / "run-v22-correction-v2",
            "correction_freeze": scorer._load_json(
                probe / "run-v22-correction-v2" / "correction-freeze.json"
            ),
            "source_freeze": scorer._load_json(probe / "run-v22" / "predictions-freeze.json"),
            "reference_path": benchmarks
            / "task-support-sufficiency-audit-issue35-v1"
            / "manual-audit-corrected-v3.json",
            "reference_correction_path": benchmarks
            / "task-support-sufficiency-audit-issue35-v1"
            / "audit-correction-v3.json",
            "reference_freeze_path": benchmarks
            / "task-support-sufficiency-audit-issue35-v1"
            / "corrected-audit-freeze-v3.json",
        }
    )
    monkeypatch.setattr(scorer, "_verify_prediction_inputs", lambda *args, **kwargs: temp_verified)
    return probe


@pytest.mark.parametrize("entrypoint", ["score", "verify_score"])
def test_prediction_verification_precedes_manual_or_audit_load(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    events: list[str] = []
    original_verify = scorer._verify_prediction_inputs
    original_load = scorer._load_json

    def wrapped_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("prediction-verify")
        return original_verify(*args, **kwargs)

    def wrapped_load(path: Path, description: str = "JSON file") -> Any:
        if path.name in {
            scorer.AUDIT_V2_NAME,
            scorer.AUDIT_V3_NAME,
            scorer.AUDIT_FREEZE_V3_NAME,
            scorer.AUDIT_FREEZE_V4_NAME,
            scorer.SCORE_V3_NAME,
            scorer.SCORE_V4_NAME,
            scorer.SCORE_V5_NAME,
            scorer.SCORE_INTEGRITY_V3_NAME,
            scorer.SCORE_INTEGRITY_V4_NAME,
            scorer.SCORE_INTEGRITY_V5_NAME,
        }:
            events.append(f"load:{path.name}")
        return original_load(path, description)

    monkeypatch.setattr(scorer, "_verify_prediction_inputs", wrapped_verify)
    monkeypatch.setattr(scorer, "_load_json", wrapped_load)
    getattr(scorer, entrypoint)(PROBE_BENCHMARK)
    assert events.index("prediction-verify") < next(
        index for index, event in enumerate(events) if event.startswith("load:")
    )


def test_reference_ledger_rejects_self_consistent_recomputed_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "benchmarks"
    shutil.copytree(CANONICAL_AUDIT, root / "task-support-sufficiency-audit-issue35-v1")
    probe = root / "dense-question-localized-evidence-issue35-v1"
    probe.mkdir()
    canonical = scorer._canonical_reference_paths(PROBE_BENCHMARK)
    replacement = tuple(root / "task-support-sufficiency-audit-issue35-v1" / path.name for path in canonical)
    monkeypatch.setattr(scorer, "_canonical_reference_paths", lambda _root: replacement)
    with pytest.raises(ValueError, match="canonical reference location"):
        scorer.verify_canonical_references(probe)


@pytest.mark.parametrize("field", ["ambiguous_reference", "reference_issue", "fact_present_in_clip"])
def test_freeze_rejects_one_arm_pair_flip_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    probe = _copy_probe_with_patched_verified_inputs(tmp_path, monkeypatch)
    scorer.write_audit_v3(probe)
    freeze = probe / "run-v22" / scorer.AUDIT_FREEZE_V4_NAME
    freeze.unlink(missing_ok=True)
    audit_path = probe / "run-v22" / scorer.AUDIT_V3_NAME
    audit = json.loads(audit_path.read_text())
    row = audit["rows"][1]
    if field == "ambiguous_reference":
        row[field] = True
    elif field == "fact_present_in_clip":
        row[field] = not row[field]
    else:
        row[field] = "one-arm drift"
    if field in {"ambiguous_reference", "fact_present_in_clip"}:
        row["recovered"] = False
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="invariant"):
        scorer.freeze_audit(probe)
    assert not freeze.exists()


@pytest.mark.parametrize(
    "field",
    ["experiment_id", "correction_version", "correction_freeze_sha256", "reference_sha256", "source_prediction_freeze_sha256"],
)
def test_freeze_rejects_forged_top_level_lineage_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    probe = _copy_probe_with_patched_verified_inputs(tmp_path, monkeypatch)
    scorer.write_audit_v3(probe)
    freeze = probe / "run-v22" / scorer.AUDIT_FREEZE_V4_NAME
    freeze.unlink(missing_ok=True)
    audit_path = probe / "run-v22" / scorer.AUDIT_V3_NAME
    audit = json.loads(audit_path.read_text())
    audit[field] = "forged"
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="lineage|binding"):
        scorer.freeze_audit(probe)
    assert not freeze.exists()


def test_manual_audit_binds_question_to_selection() -> None:
    selection = {
        "selected_row_metadata": [
            {
                "audit_index": 1,
                "role": "target",
                "video_id": "video-1",
                "question": "what happened?",
                "map_line_numbers": [1],
            }
        ]
    }
    predictions = {"rows": []}
    audit_rows = []
    for fps in (4.0, 12.0):
        predictions["rows"].append(
            {
                "audit_index": 1,
                "role": "target",
                "fps": fps,
                "status": "passed",
                "clip_sha256": "a" * 64,
                "raw_response_sha256": "b" * 64,
                "parsed_output_sha256": "c" * 64,
                "correction_acceptance_changed": False,
                "parsed": {"evidence_timestamps": [], "fact_observed": False},
                "video_id": "video-1",
                "question": "what happened?",
            }
        )
        audit_rows.append(
            {
                "audit_index": 1,
                "role": "target",
                "fps": fps,
                "video_id": "video-1",
                "question": "wrong question",
                "reference": "reference",
                "map_line_numbers": [1],
                "clip_sha256": "a" * 64,
                "raw_response_sha256": "b" * 64,
                "corrected_parsed_sha256": "c" * 64,
                "evidence_timestamps": [],
                "fact_present_in_clip": False,
                "model_fact_supported": False,
                "recovered": False,
                "timestamp_supported": False,
                "unsupported_observation": False,
                "ambiguous_reference": False,
                "correction_acceptance_changed": False,
                "reference_issue": "",
                "exposure_flags": {
                    "grounded_v2_24": False,
                    "grounded_v3_11": False,
                    "hard_audit_30": False,
                    "multi_interpretation_12": False,
                    "scope_control": False,
                    "seven_row_canary": False,
                    "synthesis_followup": False,
                },
                "reviewer_note": "inspected",
                "inspected": True,
            }
        )
    audit = {
        "completed_at": "2026-08-16T00:00:00+00:00",
        "correction_freeze_sha256": "d" * 64,
        "correction_version": scorer.CORRECTION_VERSION,
        "experiment_id": "dense-question-localized-evidence-issue35-v1",
        "inspection_scope": {
            "arms_inspected": 2,
            "auditable_target_policy": "fixture",
            "changed_acceptance_arms_inspected": 0,
            "clips_inspected": 1,
            "disagreements_inspected": 0,
            "source": "fixture",
        },
        "reference_correction_sha256": "e" * 64,
        "reference_freeze_sha256": "f" * 64,
        "reference_ledger_sha256": scorer.REFERENCE_LEDGER_SHA256,
        "reference_sha256": "0" * 64,
        "rows": audit_rows,
        "schema_version": scorer.AUDIT_SCHEMA_VERSION,
        "source_prediction_freeze_sha256": "1" * 64,
        "status": "completed",
    }
    with pytest.raises(ValueError, match="question"):
        scorer.validate_manual_audit(selection, predictions, audit)
