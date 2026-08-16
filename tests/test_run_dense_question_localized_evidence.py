from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_dense_question_localized_evidence.py"
spec = importlib.util.spec_from_file_location("run_dense_question_localized_evidence", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load runner module from {SCRIPT}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _valid_probe(*, fact_observed: bool = True) -> dict[str, Any]:
    if fact_observed:
        return {
            "question_interpretation": {
                "requested_entity": "the robot",
                "requested_visual_fact": "final action",
                "temporal_qualifier": "at the end",
            },
            "fact_observed": True,
            "observed_fact": "The robot clicks its feet.",
            "evidence_timestamps": [0.75],
            "conflicts_or_uncertainty": [],
            "needs_more_evidence": False,
        }
    return {
        "question_interpretation": {
            "requested_entity": "the robot",
            "requested_visual_fact": "final action",
            "temporal_qualifier": "at the end",
        },
        "fact_observed": False,
        "observed_fact": "",
        "evidence_timestamps": [],
        "conflicts_or_uncertainty": ["The final action is not visible in this clip."],
        "needs_more_evidence": True,
    }


def _valid_call(*_args: Any) -> tuple[str, dict[str, Any]]:
    return json.dumps(_valid_probe()), {"provider": "fake", "total_token_count": 10}


def _write_authorized_fixture(
    tmp_path: Path,
    *,
    rows: int = 4,
    target_count: int = 2,
    control_count: int | None = None,
) -> Path:
    if control_count is not None:
        rows = target_count + control_count
    root = tmp_path / "run"
    root.mkdir()
    clips = []
    provider_rows = []
    for index in range(rows):
        audit_index = index + 1
        clip = root / "clips" / f"clip-{audit_index:02d}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"clip-{audit_index}".encode())
        clip_sha = _sha(clip)
        clips.append(
            {
                "audit_index": audit_index,
                "role": "target" if index < target_count else "control",
                "clip_path": str(clip),
                "clip_sha256": clip_sha,
                "mime_type": "video/mp4",
                "clip_start_seconds": 1.0,
                "clip_end_seconds": 3.0,
                "clip_duration_seconds": 2.0,
                "arms": {
                    "4fps": {"fps": 4.0, "clip_sha256": clip_sha},
                    "12fps": {"fps": 12.0, "clip_sha256": clip_sha},
                },
            }
        )
        provider_rows.append(
            {
                "audit_index": audit_index,
                "role": "target" if index < target_count else "control",
                "question": "What does the robot do at the end?",
                "map_lines": [
                    {"line_number": 1, "text": "The robot moves."},
                ],
            }
        )
    _write_json(root / "provider-rows.json", {"schema_version": 1, "rows": provider_rows})
    _write_json(root / "media-manifest.json", {"schema_version": 1, "rows": clips})
    _write_json(
        root / "treatment-freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "model": runner.MODEL,
            "settings": runner.EXPECTED_SETTINGS,
            "expected_fps": list(runner.EXPECTED_FPS),
            "max_calls": runner.MAX_CALLS,
            "compatibility_canary_pairs": runner.COMPATIBILITY_CANARY_PAIRS,
            "decision_gate": runner.EXPECTED_DECISION_GATE,
            "runner_sha256": _sha(runner.SCRIPT_PATH),
            "config_sha256": _sha(ROOT / "configs/vqa/dense_question_localized_evidence_issue35.json"),
            "prompt_template_sha256": runner._prompt_template_sha256(),
            "response_schema_sha256": runner._schema_sha256(),
            "provider_response_schema_sha256": runner._provider_schema_sha256(),
            "validator_sha256": runner._validator_sha256(),
        },
    )
    _write_json(
        root / "selection-freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "treatment_freeze_sha256": _sha(root / "treatment-freeze.json"),
            "provider_rows_sha256": runner.provider_rows_sha256(provider_rows),
            "media_manifest_sha256": _sha(root / "media-manifest.json"),
            "selected_rows": [item["audit_index"] for item in provider_rows],
            "target_count": target_count,
            "control_count": rows - target_count,
            "planned_call_count": rows * 2,
            "decision_gate": runner.EXPECTED_DECISION_GATE,
            "compatibility_canary_pairs": runner.COMPATIBILITY_CANARY_PAIRS,
        },
    )
    _write_json(
        root / "provider-authorization.json",
        {
            "schema_version": 1,
            "status": "authorized",
            "promotion_gate": "PASS",
            "decision_gate": runner.EXPECTED_DECISION_GATE,
            "compatibility_canary_pairs": runner.COMPATIBILITY_CANARY_PAIRS,
            "treatment_freeze_sha256": _sha(root / "treatment-freeze.json"),
            "selection_freeze_sha256": _sha(root / "selection-freeze.json"),
            "provider_rows_sha256": runner.provider_rows_sha256(provider_rows),
            "media_manifest_sha256": _sha(root / "media-manifest.json"),
            "planned_call_count": rows * 2,
            "max_calls": runner.MAX_CALLS,
            "provider_calls_made": 0,
            "target_count": target_count,
            "control_count": rows - target_count,
        },
    )
    return root


def test_probe_requires_supported_observation_or_explicit_more_evidence() -> None:
    observed = runner.validate_probe(
        {
            "question_interpretation": {
                "requested_entity": "the robot",
                "requested_visual_fact": "final action",
                "temporal_qualifier": "at the end",
            },
            "fact_observed": True,
            "observed_fact": "The robot clicks its feet.",
            "evidence_timestamps": [2.75],
            "conflicts_or_uncertainty": [],
            "needs_more_evidence": False,
        },
        clip_duration_seconds=4.0,
    )
    assert observed["evidence_timestamps"] == [2.75]
    with pytest.raises(ValueError, match="timestamp"):
        runner.validate_probe(
            {**observed, "evidence_timestamps": [4.5]},
            clip_duration_seconds=4.0,
        )


def test_probe_false_requires_explicit_uncertainty_and_more_evidence() -> None:
    payload = _valid_probe(fact_observed=False)
    assert runner.validate_probe(payload, clip_duration_seconds=2.0)["fact_observed"] is False
    for field, value in (
        ("observed_fact", "something"),
        ("evidence_timestamps", [0.5]),
        ("conflicts_or_uncertainty", []),
        ("needs_more_evidence", False),
    ):
        with pytest.raises(ValueError):
            runner.validate_probe({**payload, field: value}, clip_duration_seconds=2.0)


def test_probe_rejects_extra_and_competition_answer_fields() -> None:
    payload = _valid_probe()
    with pytest.raises(ValueError, match="fields"):
        runner.validate_probe({**payload, "answer": "The robot clicks."}, 2.0)
    with pytest.raises(ValueError, match="fields"):
        runner.validate_probe({**payload, "extra": True}, 2.0)


def test_provider_response_schema_omits_recursive_additional_properties_but_local_validation_stays_strict() -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert "additionalProperties" not in value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(runner.PROVIDER_RESPONSE_SCHEMA)
    assert "additionalProperties" in runner.PROBE_SCHEMA
    assert "additionalProperties" in runner.PROBE_SCHEMA["properties"]["question_interpretation"]
    with pytest.raises(ValueError, match="fields"):
        runner.validate_probe({**_valid_probe(), "extra": True}, 2.0)


def test_probe_accepts_nominal_endpoint_with_one_frame_encoding_shortfall() -> None:
    payload = _valid_probe()
    payload["evidence_timestamps"] = [4.0]
    result = runner.validate_probe(
        payload, 3.994, nominal_duration_seconds=4.0, fps=12.0
    )
    assert result["evidence_timestamps"] == [4.0]


def test_probe_rejects_timestamp_past_nominal_endpoint() -> None:
    payload = _valid_probe()
    payload["evidence_timestamps"] = [4.001]
    with pytest.raises(ValueError, match="nominal"):
        runner.validate_probe(payload, 3.994, nominal_duration_seconds=4.0, fps=12.0)


def test_probe_rejects_endpoint_overshoot_beyond_measured_shortfall() -> None:
    payload = _valid_probe()
    payload["evidence_timestamps"] = [4.0]
    with pytest.raises(ValueError, match="shortfall"):
        runner.validate_probe(payload, 3.9, nominal_duration_seconds=4.0, fps=12.0)


def test_probe_rejects_encoding_shortfall_over_one_frame() -> None:
    payload = _valid_probe()
    payload["evidence_timestamps"] = [4.0]
    with pytest.raises(ValueError, match="one frame"):
        runner.validate_probe(payload, 3.8, nominal_duration_seconds=4.0, fps=8.0)


def test_probe_accepts_decimal_exact_nominal_endpoint() -> None:
    payload = _valid_probe()
    payload["evidence_timestamps"] = [4.1]
    result = runner.validate_probe(
        payload, 4.0, nominal_duration_seconds=4.1, fps=4.0
    )
    assert result["evidence_timestamps"] == [4.1]


def test_continuation_validator_rejects_empty_imported_source_artifacts(tmp_path: Path) -> None:
    # A continuation candidate must never authorize a self-consistent but
    # unauditable imported prefix.
    root = tmp_path / "continuation"
    root.mkdir()
    _write_json(root / "continuation-authorization.json", {
        "status": "authorized", "authorization_type": "continuation",
        "provider_calls_made": 0, "prior_provider_calls": 2,
        "remaining_new_call_maximum": 22, "combined_call_maximum": 24,
        "next_arm_position": 3, "canary_replay_freeze_sha256": "x",
        "imported_prefix": [{"fps": 4.0, "role": "target"}, {"fps": 12.0, "role": "target"}],
    })
    _write_json(root / "canary-replay-freeze.json", {
        "status": "frozen", "canary_pair_status": "passed", "imported_prefix": [],
    })
    with pytest.raises(ValueError):
        runner.validate_continuation_authorization(root)


def test_continuation_uses_fixed_v4_prediction_anchor() -> None:
    assert runner.TRUSTED_V4_PREDICTION_FREEZE_SHA256 == (
        "7a5be02b759aec4a6b1b504d73f3616ea7987900369c6b6ba9273a1fdaa5b7a0"
    )


def test_continuation_uses_fixed_v4_provider_rows_anchor() -> None:
    assert runner.TRUSTED_V4_PROVIDER_ROWS_SHA256 == (
        "007fd6a66fd403d88782438c655028c0381a714ef84f70a812ed511d44eed293"
    )


def test_prompt_is_question_localized_and_prohibits_label_leakage() -> None:
    prompt = runner.build_prompt(
        "What does the robot do at the end?",
        [{"line_number": 2, "text": "The robot moves."}],
        clip_start_seconds=12.5,
        clip_end_seconds=15.0,
    ).lower()
    for phrase in (
        "12.5",
        "15.0",
        "clip-relative",
        "references",
        "previous answers",
        "guess",
        "answer candidates",
        "purpose",
        "exact speech",
        "ocr",
        "counts",
    ):
        assert phrase in prompt
    assert prompt.index("video") < prompt.index("question")


def test_run_writes_one_terminal_chain_per_pair_without_retry(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert len(calls) == 8
    assert len(list((root / "rows").glob("*/terminal.json"))) == 8
    runner.verify_predictions_freeze(root)
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["completed_count"] == 8
    assert freeze["call_count"] == 8


def test_run_rejects_label_bearing_provider_rows_before_call(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    payload = json.loads((root / "provider-rows.json").read_text())
    payload["rows"][0]["reference"] = "hidden"
    _write_json(root / "provider-rows.json", payload)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="provider-safe"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


def test_run_preserves_malformed_json_and_canary_schema_failure_without_retry(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    responses = iter(
        [
            ("{broken", {"provider": "fake"}),
            (json.dumps({"wrong": True}), {"provider": "fake"}),
            *[_valid_call() for _ in range(6)],
        ]
    )
    calls: list[tuple[Any, ...]] = []
    runner.run(root, call=lambda *args: calls.append(args) or next(responses))
    assert len(calls) == 2
    statuses = sorted(json.loads(path.read_text())["status"] for path in (root / "rows").glob("*/terminal.json"))
    assert statuses == ["contract_error", "incomplete", "incomplete", "incomplete", "incomplete", "incomplete", "incomplete", "invalid_json"]
    malformed = next(path.parent for path in (root / "rows").glob("*/terminal.json") if json.loads(path.read_text())["status"] == "invalid_json")
    assert (malformed / "raw-response.txt").read_text() == "{broken"


def test_run_refuses_write_once_and_detects_artifact_drift(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    runner.run(root, call=_valid_call)
    with pytest.raises(ValueError, match="artifacts"):
        runner.run(root, call=_valid_call)
    runner.verify_predictions_freeze(root)
    row = next((root / "rows").iterdir())
    request = row / "request.json"
    request.write_text(request.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        runner.verify_predictions_freeze(root)


def test_run_rejects_clip_hash_drift_before_calls(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    clip = next((root / "clips").glob("*.mp4"))
    clip.write_bytes(b"changed")
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="clip"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


@pytest.mark.parametrize(
    ("target_count", "control_count", "message"),
    [
        (1, 1, "2-8 target"),
        (12, 0, "2-8 target"),
        (2, 10, "2-4 control"),
    ],
)
def test_run_rejects_role_count_boundary_before_calls(
    tmp_path: Path, target_count: int, control_count: int, message: str
) -> None:
    root = _write_authorized_fixture(
        tmp_path,
        target_count=target_count,
        control_count=control_count,
    )
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match=message):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


def test_interruption_freezes_partial_progress_without_retry(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def interrupted(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt("operator stopped run")
        return _valid_call()

    with pytest.raises(KeyboardInterrupt, match="operator stopped"):
        runner.run(root, call=interrupted)
    assert len(calls) == 2
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["status"] == "incomplete"
    assert freeze["call_count"] == 2
    assert freeze["completed_count"] == 2
    assert freeze["interrupted"] is True
    statuses = [
        json.loads(path.read_text())["status"]
        for path in (root / "rows").glob("*/terminal.json")
    ]
    assert statuses.count("interrupted") == 1
    assert statuses.count("incomplete") == 6
    runner.verify_predictions_freeze(root)
    with pytest.raises(ValueError, match="artifacts"):
        runner.run(root, call=interrupted)


def test_run_rejects_decision_gate_drift_before_calls(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path)
    treatment_path = root / "treatment-freeze.json"
    treatment = json.loads(treatment_path.read_text())
    treatment["decision_gate"]["minimum_incremental_target_recoveries"] = 1
    _write_json(treatment_path, treatment)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="decision gate"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


@pytest.mark.parametrize("provider_calls_made", [None, 1])
def test_run_rejects_missing_or_nonzero_authorization_call_count(
    tmp_path: Path, provider_calls_made: int | None
) -> None:
    root = _write_authorized_fixture(tmp_path)
    authorization_path = root / "provider-authorization.json"
    authorization = json.loads(authorization_path.read_text())
    if provider_calls_made is None:
        authorization.pop("provider_calls_made")
    else:
        authorization["provider_calls_made"] = provider_calls_made
    _write_json(authorization_path, authorization)
    calls: list[tuple[Any, ...]] = []

    with pytest.raises(ValueError, match="provider_calls_made"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


def test_failed_first_compatibility_canary_stops_before_remaining_arms(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path, rows=12, target_count=8)
    responses = iter([json.dumps({"wrong": True}), json.dumps(_valid_probe())])
    calls: list[tuple[Any, ...]] = []

    runner.run(root, call=lambda *args: calls.append(args) or (next(responses), {"provider": "fake"}))

    assert len(calls) == 2
    terminals = [json.loads(path.read_text()) for path in (root / "rows").glob("*/terminal.json")]
    assert sum(item["status"] == "contract_error" for item in terminals) == 1
    assert sum(item["status"] == "passed" for item in terminals) == 1
    assert sum(item["status"] == "incomplete" for item in terminals) == 22
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["status"] == "incomplete"
    assert freeze["call_count"] == 2
    assert freeze["completed_count"] == 2
    assert freeze["compatibility_canary_pairs"] == 1
    runner.verify_predictions_freeze(root)


@pytest.mark.parametrize("tamper", ["call_count", "row_order"])
def test_failed_canary_freeze_derives_branch_from_ordered_artifacts(
    tmp_path: Path, tamper: str
) -> None:
    root = _write_authorized_fixture(tmp_path, rows=12, target_count=8)
    responses = iter([json.dumps({"wrong": True}), json.dumps(_valid_probe())])
    runner.run(root, call=lambda *_args: (next(responses), {"provider": "fake"}))
    freeze_path = root / "predictions-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if tamper == "call_count":
        freeze["call_count"] = 3
    else:
        freeze["rows"] = list(reversed(freeze["rows"]))
    _write_json(freeze_path, freeze)

    with pytest.raises(ValueError, match="canary|order|call count"):
        runner.verify_predictions_freeze(root)


def test_passed_compatibility_canary_allows_complete_panel_freeze(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path, rows=12, target_count=8)
    calls: list[tuple[Any, ...]] = []

    runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    runner.verify_predictions_freeze(root)

    assert len(calls) == 24
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["status"] == "frozen"
    assert freeze["call_count"] == 24
    assert freeze["completed_count"] == 24
    assert freeze["compatibility_canary_pairs"] == 1
    assert freeze["canary_pair_status"] == "passed"


# Task 3D continuation audit-chain fixtures.  These use the frozen local v4
# source and fake callbacks only; they never load labels/references or call a
# provider endpoint.
SOURCE_V4 = (
    ROOT
    / "data/vqa/benchmarks/dense-question-localized-evidence-issue35-v1/run-v4"
)


def _prepare_continuation_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "continuation"
    runner.prepare_continuation(SOURCE_V4, root)
    return root


def _id_call(response_id: str, *, valid: bool = True) -> tuple[str, dict[str, Any]]:
    raw = json.dumps(_valid_probe() if valid else {"wrong": True})
    return raw, {"provider": "fake", "response_id": response_id}


def _run_complete_continuation(root: Path) -> list[str]:
    response_ids: list[str] = []

    def call(*_args: Any) -> tuple[str, dict[str, Any]]:
        response_id = f"new-{len(response_ids) + 1:02d}"
        response_ids.append(response_id)
        return _id_call(response_id)

    runner.run(root, call=call)
    runner.verify_predictions_freeze(root)
    return response_ids


def _refresh_current_input_bindings(root: Path) -> None:
    provider_rows = json.loads((root / "provider-rows.json").read_text())["rows"]
    selection_path = root / "selection-freeze.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_rows"] = [row["audit_index"] for row in provider_rows]
    selection["provider_rows_sha256"] = runner.provider_rows_sha256(provider_rows)
    selection["media_manifest_sha256"] = _sha(root / "media-manifest.json")
    _write_json(selection_path, selection)
    authorization_path = root / "provider-authorization.json"
    authorization = json.loads(authorization_path.read_text())
    authorization["selection_freeze_sha256"] = _sha(selection_path)
    authorization["provider_rows_sha256"] = runner.provider_rows_sha256(provider_rows)
    authorization["media_manifest_sha256"] = _sha(root / "media-manifest.json")
    _write_json(authorization_path, authorization)
    continuation_path = root / "continuation-authorization.json"
    continuation = json.loads(continuation_path.read_text())
    continuation["selection_freeze_sha256"] = _sha(selection_path)
    continuation["provider_rows_sha256"] = _sha(root / "provider-rows.json")
    continuation["media_manifest_sha256"] = _sha(root / "media-manifest.json")
    _write_json(continuation_path, continuation)


def _freeze_row_for(root: Path, audit_index: int, fps: float) -> dict[str, Any]:
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    return next(
        row
        for row in freeze["rows"]
        if row["audit_index"] == audit_index and row["fps"] == fps
    )


def _rewrite_freeze_rows(root: Path) -> dict[str, Any]:
    freeze_path = root / "predictions-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    rebuilt: list[dict[str, Any]] = []
    passed = 0
    completed = 0
    provider_rows = json.loads((root / "provider-rows.json").read_text())["rows"]
    media_rows = json.loads((root / "media-manifest.json").read_text())["rows"]
    roles = {row["audit_index"]: row["role"] for row in media_rows}
    for provider_row in provider_rows:
        audit_index = provider_row["audit_index"]
        role = roles[audit_index]
        for fps in runner.EXPECTED_FPS:
            arm = runner._arm_directory(root, audit_index, role, fps)
            terminal = json.loads((arm / "terminal.json").read_text())
            status = terminal["status"]
            completed += status != "incomplete"
            passed += status == "passed"
            rebuilt.append(
                {
                    "audit_index": audit_index,
                    "role": role,
                    "fps": fps,
                    "status": status,
                    "artifacts": {
                        name: _sha(arm / name)
                        for name in sorted(path.name for path in arm.iterdir())
                    },
                }
            )
    freeze["rows"] = rebuilt
    freeze["passed_count"] = passed
    freeze["failed_count"] = len(rebuilt) - passed
    freeze["completed_count"] = completed
    freeze["call_count"] = completed
    _write_json(freeze_path, freeze)
    return freeze


def test_continuation_authorization_bytes_drift_after_freeze_is_rejected(tmp_path: Path) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["remaining_new_call_maximum"] = 21
    _write_json(auth_path, auth)
    with pytest.raises(ValueError, match="continuation authorization bytes"):
        runner.verify_predictions_freeze(root)


def test_continuation_rejects_self_consistent_remaining_row_order_drift_before_callback(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    provider_path = root / "provider-rows.json"
    provider = json.loads(provider_path.read_text())
    provider["rows"] = list(reversed(provider["rows"]))
    _write_json(provider_path, provider)
    _refresh_current_input_bindings(root)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="ordered source panel identity"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_continuation_rejects_forged_imported_response_id_before_callback(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    for filename in ("continuation-authorization.json", "canary-replay-freeze.json"):
        path = root / filename
        payload = json.loads(path.read_text())
        payload["imported_prefix"][0]["source_response_id"] = "forged-response-id"
        _write_json(path, payload)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(root / "canary-replay-freeze.json")
    _write_json(auth_path, auth)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="response ID does not bind source metadata"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_continuation_rejects_forged_imported_audit_index_before_callback(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    for filename in ("continuation-authorization.json", "canary-replay-freeze.json"):
        path = root / filename
        payload = json.loads(path.read_text())
        payload["imported_prefix"][0]["audit_index"] = 999
        _write_json(path, payload)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(root / "canary-replay-freeze.json")
    _write_json(auth_path, auth)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="audit index does not bind source arm"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_continuation_rejects_self_consistent_rewritten_source_terminal_and_freeze(
    tmp_path: Path,
) -> None:
    source_copy = tmp_path / "rewritten-source"
    shutil.copytree(SOURCE_V4, source_copy, symlinks=True, ignore=shutil.ignore_patterns("clips"))
    (source_copy / "clips").symlink_to(SOURCE_V4 / "clips", target_is_directory=True)
    terminal_path = source_copy / "rows/01-target-4fps/terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["audit_index"] = 999
    _write_json(terminal_path, terminal)
    freeze_path = source_copy / "predictions-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    for row in freeze["rows"]:
        if row["audit_index"] == 1 and row["fps"] == 4.0:
            row["artifacts"]["terminal.json"] = _sha(terminal_path)
    _write_json(freeze_path, freeze)
    root = _prepare_continuation_fixture(tmp_path)
    for filename in ("continuation-authorization.json", "canary-replay-freeze.json"):
        path = root / filename
        payload = json.loads(path.read_text())
        payload["source_root"] = str(source_copy)
        payload["source_predictions_freeze_sha256"] = _sha(freeze_path)
        _write_json(path, payload)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(root / "canary-replay-freeze.json")
    _write_json(auth_path, auth)
    with pytest.raises(ValueError, match="fixed source anchor"):
        runner.validate_continuation_authorization(root)


def test_duplicate_new_response_id_preserves_raw_metadata_and_blocks_freeze(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    ids = iter(["new-01", "new-02", "new-02", *[f"new-{i:02d}" for i in range(4, 23)]])
    calls: list[tuple[Any, ...]] = []
    runner.run(root, call=lambda *args: calls.append(args) or _id_call(next(ids)))
    assert len(calls) == 22
    duplicate_arm = runner._arm_directory(root, 3, "target", 4.0)
    assert (duplicate_arm / "raw-response.txt").is_file()
    metadata = json.loads((duplicate_arm / "metadata.json").read_text())
    assert metadata["response_id"] == "new-02"
    terminal = json.loads((duplicate_arm / "terminal.json").read_text())
    assert terminal["status"] == "contract_error"
    assert terminal["error_type"] == "DuplicateResponseID"
    with pytest.raises(ValueError, match="duplicate provider response ID"):
        runner.verify_predictions_freeze(root)


def test_empty_imported_source_artifact_map_is_rejected_after_hashes_are_corrected(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    for filename in ("continuation-authorization.json", "canary-replay-freeze.json"):
        path = root / filename
        payload = json.loads(path.read_text())
        payload["imported_prefix"][0]["source_artifacts"] = {}
        _write_json(path, payload)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(root / "canary-replay-freeze.json")
    _write_json(auth_path, auth)
    with pytest.raises(ValueError, match="source artifact map must be non-empty"):
        runner.validate_continuation_authorization(root)


def test_complete_combined_freeze_rejects_nonpassed_fresh_canary_position(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    arm = runner._arm_directory(root, 2, "target", 4.0)
    terminal = json.loads((arm / "terminal.json").read_text())
    terminal["status"] = "contract_error"
    terminal["error_type"] = "ValueError"
    _write_json(arm / "terminal.json", terminal)
    _rewrite_freeze_rows(root)
    with pytest.raises(ValueError, match="fresh canary"):
        runner.verify_predictions_freeze(root)


def test_failed_continuation_freeze_rejects_all_passed_fresh_canary_positions(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    provider_rows = json.loads((root / "provider-rows.json").read_text())["rows"]
    media_rows = json.loads((root / "media-manifest.json").read_text())["rows"]
    roles = {row["audit_index"]: row["role"] for row in media_rows}
    for provider_row in provider_rows[2:]:
        for fps in runner.EXPECTED_FPS:
            arm = runner._arm_directory(root, provider_row["audit_index"], roles[provider_row["audit_index"]], fps)
            for name in ("parsed.json", "raw-response.txt"):
                (arm / name).unlink(missing_ok=True)
            _write_json(
                arm / "metadata.json",
                {"status": "incomplete", "reason": "fresh canary test"},
            )
            _write_json(
                arm / "terminal.json",
                {
                    "audit_index": provider_row["audit_index"],
                    "role": roles[provider_row["audit_index"]],
                    "fps": fps,
                    "status": "incomplete",
                },
            )
    freeze = _rewrite_freeze_rows(root)
    freeze["status"] = "incomplete"
    freeze["stop_reason"] = "continuation_canary_failure"
    freeze["interrupted"] = False
    freeze["call_count"] = 4
    freeze["completed_count"] = 4
    freeze["failed_count"] = 20
    freeze["passed_count"] = 4
    freeze["canary_pair_status"] = "passed"
    _write_json(root / "predictions-freeze.json", freeze)
    with pytest.raises(ValueError, match="fresh canary"):
        runner.verify_predictions_freeze(root)


def test_passing_continuation_calls_only_remaining_arms_and_freezes_combined_history(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    response_ids: list[str] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        response_id = f"ordered-{len(calls):02d}"
        response_ids.append(response_id)
        return _id_call(response_id)

    runner.run(root, call=call)
    runner.verify_predictions_freeze(root)
    assert len(calls) == 22
    media_rows = json.loads((root / "media-manifest.json").read_text())["rows"]
    second_clip = root / media_rows[1]["clip_path"]
    assert calls[0][0] == second_clip.read_bytes()
    assert calls[0][3] == 4.0
    assert len(response_ids) == len(set(response_ids)) == 22
    assert json.loads((root / "predictions-freeze.json").read_text())["completed_count"] == 24
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["fresh_canary_pair_status"] == "passed"
    assert freeze["prior_provider_calls"] == 2
    assert freeze["remaining_new_call_maximum"] == 22
    assert freeze["combined_call_maximum"] == 24


def test_failed_continuation_calls_only_fresh_canary_and_freezes_incomplete_history(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    responses = iter([_id_call("fresh-01", valid=False), _id_call("fresh-02")])
    runner.run(root, call=lambda *args: calls.append(args) or next(responses))
    assert len(calls) == 2
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["completed_count"] == 4
    assert freeze["call_count"] == 4
    assert freeze["fresh_canary_pair_status"] == "failed"
    assert sum(
        json.loads(path.read_text())["status"] == "incomplete"
        for path in (root / "rows").glob("*/terminal.json")
    ) == 20
    runner.verify_predictions_freeze(root)
