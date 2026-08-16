from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path("scripts/score_bounded_multipass_counting.py")
SPEC = importlib.util.spec_from_file_location("score_bounded", SCRIPT)
assert SPEC and SPEC.loader
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)

RUNNER_SCRIPT = Path("scripts/run_bounded_multipass_counting.py")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_bounded_for_score", RUNNER_SCRIPT
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def row(
    video_id: str,
    *,
    role: str,
    baseline_correct: bool,
    fixed_correct: bool,
    refinement_correct: bool,
):
    return {
        "video_id": video_id,
        "panel_role": role,
        "baseline_correct": baseline_correct,
        "fixed_correct": fixed_correct,
        "refinement_correct": refinement_correct,
    }


def passing_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(8):
        rows.append(
            row(
                f"f{index}",
                role="confirmed_failure",
                baseline_correct=False,
                fixed_correct=index < 5,
                refinement_correct=index < 7,
            )
        )
    for index in range(4):
        rows.append(
            row(
                f"c{index}",
                role="correct_control",
                baseline_correct=True,
                fixed_correct=True,
                refinement_correct=True,
            )
        )
    return rows


def test_build_decision_promotes_fixed_and_retains_refinement_at_declared_gates() -> (
    None
):
    decision = scorer.build_decision(
        rows=passing_rows(), fixed_cost_usd=5.0, refinement_total_cost_usd=8.0
    )
    assert decision["fixed"] == {
        "rescued": 5,
        "failure_total": 8,
        "controls_preserved": 4,
        "control_total": 4,
        "gain_over_baseline": 5,
        "gate_passed": True,
    }
    assert decision["refinement"]["added_over_fixed"] == 2
    assert decision["refinement"]["regressions_from_fixed"] == 0
    assert decision["refinement"]["gate_passed"] is True
    assert decision["selected_arm"] == "refinement"


def test_build_decision_rejects_fixed_when_one_control_regresses() -> None:
    rows = passing_rows()
    rows[-1]["fixed_correct"] = False
    decision = scorer.build_decision(
        rows=rows, fixed_cost_usd=5.0, refinement_total_cost_usd=8.0
    )
    assert decision["fixed"]["gate_passed"] is False
    assert decision["selected_arm"] == "none"


def test_build_decision_keeps_fixed_when_refinement_gain_or_cost_gate_fails() -> None:
    rows = passing_rows()
    rows[5]["refinement_correct"] = False
    decision = scorer.build_decision(
        rows=rows, fixed_cost_usd=5.0, refinement_total_cost_usd=11.0
    )
    assert decision["fixed"]["gate_passed"] is True
    assert decision["refinement"]["gate_passed"] is False
    assert decision["selected_arm"] == "fixed"


def test_build_decision_rejects_wrong_panel_denominator() -> None:
    try:
        scorer.build_decision(
            rows=passing_rows()[:-1],
            fixed_cost_usd=5.0,
            refinement_total_cost_usd=8.0,
        )
    except ValueError as error:
        assert "8 failures and 4 controls" in str(error)
    else:
        raise AssertionError("wrong denominator passed")


class FakeFiles:
    def upload(self, *, file: Path):
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://example.invalid/files/{file.stem}",
            state=SimpleNamespace(name="ACTIVE"),
            expiration_time="future",
        )


class FakeModels:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)

    def generate_content(self, **_kwargs):
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=0,
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )


class FakeClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(payloads)


def test_authenticate_arm_detects_mutated_referenced_raw_response(
    tmp_path: Path,
) -> None:
    panel_rows = []
    request_rows = []
    payloads = []
    for index in range(12):
        video_id = f"v{index:02d}"
        question = f"How many times does person {index} jump?"
        source = tmp_path / f"{video_id}.mkv"
        delivery = tmp_path / f"{video_id}.webm"
        source.write_bytes(f"source-{video_id}".encode())
        delivery.write_bytes(f"delivery-{video_id}".encode())
        panel_rows.append(
            {
                "ordinal": index,
                "video_id": video_id,
                "question": question,
                "stratum": "failure_action" if index < 8 else "control_temporal",
                "status": "wrong" if index < 8 else "correct",
                "reference_answer": "2",
                "reference_count": 2,
            }
        )
        request_rows.append(
            {
                "ordinal": index,
                "video_id": video_id,
                "question": question,
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "delivery_path": str(delivery),
                "delivery_sha256": hashlib.sha256(delivery.read_bytes()).hexdigest(),
                "delivery_mime_type": "video/webm",
                "duration_seconds": 4.0,
            }
        )
        payloads.extend(
            [
                {
                    "schema_version": "bounded-multipass-count-spec-v1",
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "target": "completed jump",
                    "unit": "action_repetition",
                    "route": "temporal",
                    "scope": {"start_seconds": 0.0, "end_seconds": 4.0},
                    "inclusions": ["completed jump"],
                    "exclusions": [],
                    "identity_policy": "observable_episode",
                    "candidate_views": [
                        {
                            "view_id": "W0",
                            "kind": "window",
                            "start_seconds": 0.0,
                            "end_seconds": 4.0,
                        }
                    ],
                    "uncertainty": "none",
                },
                {
                    "schema_version": "bounded-multipass-evidence-v1",
                    "reviewed_view_ids": ["W0"],
                    "records": [
                        {
                            "record_id": "E0",
                            "view_ids": ["W0"],
                            "start_seconds": 0.5,
                            "peak_seconds": 0.8,
                            "end_seconds": 1.1,
                            "target_supported": True,
                            "ambiguous": False,
                            "descriptor": "first jump",
                        },
                        {
                            "record_id": "E1",
                            "view_ids": ["W0"],
                            "start_seconds": 2.5,
                            "peak_seconds": 2.8,
                            "end_seconds": 3.1,
                            "target_supported": True,
                            "ambiguous": False,
                            "descriptor": "second jump",
                        },
                    ],
                },
                {
                    "schema_version": "bounded-multipass-coverage-v1",
                    "complete": True,
                    "reviewed_view_ids": ["W0"],
                    "accepted_record_ids": ["E0", "E1"],
                    "rejected_record_ids": [],
                    "missing_view_ids": [],
                    "disputes": [],
                    "refinement_request": None,
                    "notes": "complete",
                },
            ]
        )
    root = tmp_path / "run"
    panel = runner.write_self_hashed_once(
        root / "panel.freeze.json",
        {"schema_version": "panel-v1", "rows": panel_rows},
    )
    requests = runner.write_self_hashed_once(
        root / "requests.label-blind.json",
        {
            "schema_version": "requests-v1",
            "panel_complete_sha256": panel["complete_sha256"],
            "rows": request_rows,
        },
    )
    runner.write_self_hashed_once(
        root / "preparation-complete.json",
        {
            "schema_version": "prep-v1",
            "panel_complete_sha256": panel["complete_sha256"],
            "requests_complete_sha256": requests["complete_sha256"],
            "planned_rows": 12,
        },
    )
    client = FakeClient(payloads)
    runner.run_smoke(root=root, client=client)
    runner.run_fixed_arm(root=root, client=client)
    runner.run_refinement_arm(root=root, client=client)
    runner.freeze_predictions(root=root, arm="fixed")
    runner.freeze_predictions(root=root, arm="refinement")

    authenticated = scorer.authenticate_arm(root=root, arm="fixed")
    assert len(authenticated) == 12
    raw = root / "rows/v00/fixed/count-spec/raw-response.txt"
    raw.write_text(raw.read_text() + " ")
    with pytest.raises(ValueError, match="raw response changed"):
        scorer.authenticate_arm(root=root, arm="fixed")
