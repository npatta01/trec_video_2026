from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "score_gemini_counting_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("score_gemini_counting_recovery", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def self_hashed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = [
        ("failure-a", "confirmed_failure", "four", 4),
        ("failure-b", "confirmed_failure", "four", 4),
        ("control", "correct_control", "one", 1),
    ]
    manifest_rows = [
        {
            "video_id": video_id,
            "question": "How many events occur?",
            "duration_seconds": 8.0,
        }
        for video_id, _, _, _ in rows
    ]
    manifest = self_hashed({"schema_version": 1, "rows": manifest_rows})
    manifest_sha256 = manifest["complete_sha256"]
    panel_rows = []
    freeze_rows = []
    for video_id, role, reference, value in rows:
        panel_rows.append(
            {
                "video_id": video_id,
                "panel_role": role,
                "reference_answer": reference,
            }
        )
        row_root = root / "rows" / video_id
        prior = None
        stage_hashes = {}
        stages = ["locate", "verify", "challenge"]
        if video_id == "failure-a":
            stages.insert(1, "locate-correction-v2")
        for stage in stages:
            terminal = self_hashed(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "outcome": "invalid_response"
                    if stage == "locate" and video_id == "failure-a"
                    else "success",
                    "prior_complete_sha256": prior,
                    "estimated_cost_usd": 0.01,
                }
            )
            write_json(row_root / stage / "complete.json", terminal)
            prior = terminal["complete_sha256"]
            stage_hashes[stage] = prior
        result = self_hashed(
            {
                "outcome": "exact",
                "value": value,
                "accepted_instance_ids": ["I1"],
                "reason_codes": [],
            }
        )
        write_json(row_root / "result.json", result)
        complete = self_hashed(
            {
                "schema_version": 1,
                "outcome": "success",
                "video_id": video_id,
                "request_manifest_sha256": manifest_sha256,
                "stage_complete_sha256": stage_hashes,
                "result_sha256": file_sha256(row_root / "result.json"),
                "estimated_cost_usd": 0.03,
            }
        )
        write_json(row_root / "complete.json", complete)
        freeze_rows.append(
            {
                "video_id": video_id,
                "row_complete_sha256": complete["complete_sha256"],
                "estimated_cost_usd": 0.03,
                "result": result,
            }
        )
    freeze = self_hashed(
        {
            "schema_version": 1,
            "request_manifest_sha256": manifest_sha256,
            "completion": {"completed": 3, "total": 3},
            "rows": freeze_rows,
            "total_estimated_cost_usd": 0.09,
        }
    )
    panel = self_hashed({"schema_version": 1, "rows": panel_rows})
    return manifest, freeze, panel


def metric_rows() -> dict[str, dict[str, object]]:
    return {
        video_id: {
            "meteor": 1.0,
            "bertscore_f1": 1.0,
            "sts": 1.0,
            "sts_provenance": "fixture",
        }
        for video_id in ("failure-a", "failure-b", "control")
    }


def baseline_rows() -> dict[str, dict[str, object]]:
    return {
        video_id: {
            "meteor": 0.5,
            "bertscore_f1": 0.5,
            "sts": 0.5,
            "semantic_composite": 0.5,
        }
        for video_id in ("failure-a", "failure-b", "control")
    }


def test_score_authenticates_chain_and_passes_two_of_two_rescue_gate(
    tmp_path: Path,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    score = MODULE.build_score(
        root=tmp_path,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows=metric_rows(),
        baseline_rows=baseline_rows(),
    )

    assert score["completion"] == {"completed": 3, "total": 3}
    assert score["rescue"] == {"rescued": 2, "total": 2, "rate": 1.0}
    assert score["control"] == {"preserved": 1, "total": 1, "rate": 1.0}
    assert score["semantic_metrics"] == {
        "bertscore_f1": 1.0,
        "meteor": 1.0,
        "semantic_composite": 1.0,
        "sts": 1.0,
    }
    assert score["baseline_semantic_metrics"] == {
        "bertscore_f1": 0.5,
        "meteor": 0.5,
        "semantic_composite": 0.5,
        "sts": 0.5,
    }
    assert score["deltas_treatment_minus_baseline"]["semantic_composite"] == 0.5
    assert score["gate_passed"] is True


def test_score_rejects_post_freeze_result_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    result_path = tmp_path / "rows" / "failure-a" / "result.json"
    result = json.loads(result_path.read_text())
    result["value"] = 99
    write_json(result_path, result)

    with pytest.raises(ValueError, match="result hash changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_panel_mutation_after_selection_freeze(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    panel["rows"][0]["reference_answer"] = "three"

    with pytest.raises(ValueError, match="panel self-hash changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_sanitized_projection_removes_private_rows_and_answers(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    score = MODULE.build_score(
        root=tmp_path,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows=metric_rows(),
        baseline_rows=baseline_rows(),
    )

    sanitized = MODULE.sanitize_score(score)

    rendered = json.dumps(sanitized)
    assert "rows" not in sanitized
    assert "failure-a" not in rendered
    assert "reference_answer" not in rendered
    assert sanitized["rescue"]["rate"] == 1.0
