from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "score_gemini_counting_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("score_gemini_counting_evidence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sealed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["complete_sha256"] = MODULE.canonical_sha256(result)
    return result


def canary_fixture(tmp_path: Path):
    manifest = sealed(
        {
            "canary_video_ids": ["f1", "f2", "f3", "f4", "f5", "c1"],
            "rows": [
                {"video_id": video_id}
                for video_id in ["f1", "f2", "f3", "f4", "f5", "c1"]
            ],
        }
    )
    panel = sealed(
        {
            "rows": [
                *[
                    {
                        "video_id": f"f{index}",
                        "panel_role": "confirmed_failure",
                        "reference_count": 2,
                        "reference_answer": "two",
                    }
                    for index in range(1, 6)
                ],
                {
                    "video_id": "c1",
                    "panel_role": "correct_control",
                    "reference_count": 2,
                    "reference_answer": "two",
                },
            ]
        }
    )
    freeze_rows = []
    for video_id, derived in {
        "f1": 2,
        "f2": 2,
        "f3": 2,
        "f4": 3,
        "f5": 4,
        "c1": 2,
    }.items():
        result = {
            "video_id": video_id,
            "status": "passed",
            "count_unit": "unique_entities_objects",
            "derived_count": derived,
            "estimated_cost_usd": 0.02,
        }
        result_path = tmp_path / "rows" / video_id / "result.json"
        write_json(result_path, result)
        complete = {
            "video_id": video_id,
            "status": "passed",
            "result_sha256": MODULE.file_sha256(result_path),
            "estimated_cost_usd": 0.02,
        }
        complete_path = tmp_path / "rows" / video_id / "complete.json"
        write_json(complete_path, complete)
        freeze_rows.append(
            {
                "video_id": video_id,
                "status": "passed",
                "complete_sha256": MODULE.file_sha256(complete_path),
                "result_sha256": complete["result_sha256"],
            }
        )
    freeze = sealed(
        {
            "request_manifest_sha256": manifest["complete_sha256"],
            "row_count": 6,
            "rows": freeze_rows,
        }
    )
    write_json(tmp_path / "canary-predictions-freeze.json", freeze)
    return manifest, panel


def test_canary_gate_uses_only_five_failures_and_one_control(tmp_path: Path) -> None:
    manifest, panel = canary_fixture(tmp_path)

    decision = MODULE.build_canary_decision(
        root=tmp_path, panel=panel, manifest=manifest
    )

    assert decision["schema_validity"] == {"passed": 6, "required": 6}
    assert decision["failure_rescue"] == {"rescued": 3, "required": 3, "total": 5}
    assert decision["control_preservation"] == {
        "preserved": 1,
        "required": 1,
        "total": 1,
    }
    assert decision["authorized"] is True
    assert decision["projected_total_cost_usd"] < 3.0


def test_canary_gate_rejects_result_mutation_after_freeze(tmp_path: Path) -> None:
    manifest, panel = canary_fixture(tmp_path)
    result_path = tmp_path / "rows" / "f1" / "result.json"
    result = json.loads(result_path.read_text())
    result["derived_count"] = 999
    write_json(result_path, result)

    with pytest.raises(ValueError, match="result hash changed"):
        MODULE.build_canary_decision(root=tmp_path, panel=panel, manifest=manifest)


def test_metric_aggregation_uses_authenticated_rows_and_injected_scorers(
    tmp_path: Path,
) -> None:
    manifest, panel = canary_fixture(tmp_path)
    decision = MODULE.build_canary_decision(
        root=tmp_path, panel=panel, manifest=manifest
    )
    write_json(tmp_path / "canary-decision.json", decision)
    baseline_rows = [
        {
            "video_id": video_id,
            "meteor": 0.1,
            "bertscore_f1": 0.8,
            "sts": 0.2,
            "semantic_composite": (0.1 + 0.8 + 0.2) / 3,
        }
        for video_id in manifest["canary_video_ids"]
    ]
    identity = "sts-fixture"
    sts_cache = {
        "identity": identity,
        "values": {
            MODULE._sts_pair_key(identity, prediction, "two"): 0.3
            for prediction in ("2", "3", "4")
        },
    }
    bert_model = tmp_path / "bert-snapshot"
    bert_model.mkdir()
    (bert_model / "config.json").write_text("{}")

    metrics = MODULE.build_canary_metrics(
        root=tmp_path,
        panel=panel,
        manifest=manifest,
        baseline_score={"rows": {"gemini-3.1-pro-native-av-4fps": baseline_rows}},
        sts_cache=sts_cache,
        bert_model=bert_model,
        meteor_fn=lambda _prediction, _reference: 0.5,
        bertscore_fn=lambda predictions, _references, _model: [
            0.9 for _ in predictions
        ],
        metric_environment={"fixture": "1"},
    )

    assert metrics["conditions"]["atomic_treatment"] == {
        "row_count": 6,
        "meteor": 0.5,
        "bertscore_f1": 0.9,
        "sts": 0.3,
        "semantic_composite": pytest.approx((0.5 + 0.9 + 0.3) / 3),
    }
