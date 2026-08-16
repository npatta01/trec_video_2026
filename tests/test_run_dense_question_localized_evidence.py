from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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


def _expected_continuation_arms(
    root: Path, *, start_position: int, stop_position: int | None = None
) -> list[dict[str, Any]]:
    provider_rows = json.loads((root / "provider-rows.json").read_text())["rows"]
    media_rows = json.loads((root / "media-manifest.json").read_text())["rows"]
    media_by_index = {row["audit_index"]: row for row in media_rows}
    expected: list[dict[str, Any]] = []
    for position, provider in enumerate(provider_rows, start=1):
        for fps in runner.EXPECTED_FPS:
            arm_position = (position - 1) * len(runner.EXPECTED_FPS) + int(fps == 12.0) + 1
            if arm_position < start_position or (
                stop_position is not None and arm_position > stop_position
            ):
                continue
            media = media_by_index[provider["audit_index"]]
            clip = root / media["clip_path"]
            prompt = runner.build_prompt(
                provider["question"],
                provider["map_lines"],
                media["clip_start_seconds"],
                media["clip_end_seconds"],
            )
            expected.append(
                {
                    "position": arm_position,
                    "audit_index": provider["audit_index"],
                    "role": media["role"],
                    "fps": fps,
                    "clip_sha256": media["clip_sha256"],
                    "clip_bytes": clip.read_bytes(),
                    "mime_type": media["mime_type"],
                    "prompt": prompt,
                    "request": runner._request_payload(
                        provider,
                        {
                            "mime_type": media["mime_type"],
                            "clip_sha256": media["clip_sha256"],
                            "clip_start_seconds": media["clip_start_seconds"],
                            "clip_end_seconds": media["clip_end_seconds"],
                            "clip_duration_seconds": media["clip_duration_seconds"],
                        },
                        fps,
                        prompt,
                        runner._canary_pair_index(arm_position - 1),
                    ),
                }
            )
    return expected


def _assert_exact_continuation_callbacks(
    root: Path, calls: list[tuple[Any, ...]], expected: list[dict[str, Any]]
) -> None:
    assert len(calls) == len(expected)
    for call_args, item in zip(calls, expected, strict=True):
        clip_bytes, mime_type, prompt, fps = call_args
        assert clip_bytes == item["clip_bytes"]
        assert mime_type == item["mime_type"]
        assert prompt == item["prompt"]
        assert fps == item["fps"]
        assert _sha_bytes(clip_bytes) == item["clip_sha256"]
        arm = runner._arm_directory(root, item["audit_index"], item["role"], item["fps"])
        assert json.loads((arm / "request.json").read_text()) == item["request"]


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rewrite_imported_artifacts_self_consistently(root: Path) -> None:
    arm = runner._arm_directory(root, 1, "target", 4.0)
    raw = (arm / "raw-response.txt").read_text() + "\n"
    (arm / "raw-response.txt").write_text(raw)
    metadata = json.loads((arm / "metadata.json").read_text())
    metadata["response_id"] = "forged-imported-response"
    metadata["raw_response_sha256"] = _sha_bytes(raw.encode())
    metadata["response_text_sha256"] = _sha_bytes(raw.encode())
    _write_json(arm / "metadata.json", metadata)
    media = json.loads((root / "media-manifest.json").read_text())["rows"][0]
    parsed = runner.validate_probe(
        json.loads(raw),
        media["clip_duration_seconds"],
        nominal_duration_seconds=media["clip_end_seconds"] - media["clip_start_seconds"],
        fps=4.0,
    )
    _write_json(arm / "parsed.json", parsed)
    _write_json(
        arm / "terminal.json",
        {
            "audit_index": 1,
            "fps": 4.0,
            "metadata_sha256": _sha(arm / "metadata.json"),
            "parsed_semantic_sha256": _sha_bytes(runner._canonical(parsed)),
            "parsed_sha256": _sha(arm / "parsed.json"),
            "raw_response_sha256": _sha_bytes(raw.encode()),
            "request_sha256": _sha(arm / "request.json"),
            "response_id": "forged-imported-response",
            "role": "target",
            "status": "passed",
        },
    )
    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    for prefix_item in replay["imported_prefix"]:
        if prefix_item["audit_index"] == 1 and prefix_item["fps"] == 4.0:
            prefix_item["replayed_artifacts"] = {
                name: _sha(arm / name)
                for name in sorted(path.name for path in arm.iterdir())
            }
    _write_json(replay_path, replay)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    for prefix_item in auth["imported_prefix"]:
        if prefix_item["audit_index"] == 1 and prefix_item["fps"] == 4.0:
            prefix_item["replayed_artifacts"] = {
                name: _sha(arm / name)
                for name in sorted(path.name for path in arm.iterdir())
            }
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)


def _rewrite_replay_provider_calls(root: Path, value: Any = 17) -> None:
    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    replay["provider_calls_made"] = value
    _write_json(replay_path, replay)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)


def _rewrite_replay_bytes_self_consistently(root: Path) -> None:
    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=4) + "\n")
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)


def _detach_continuation_clips(root: Path, tmp_path: Path) -> None:
    """Give a continuation fixture private clip bytes for mutation tests."""

    source = (root / "clips").resolve()
    detached = tmp_path / "detached-clips"
    shutil.copytree(source, detached)
    (root / "clips").unlink()
    shutil.move(str(detached), str(root / "clips"))


def _clip_path_for(root: Path, audit_index: int) -> Path:
    rows = json.loads((root / "media-manifest.json").read_text())["rows"]
    row = next(row for row in rows if row["audit_index"] == audit_index)
    return root / row["clip_path"]


def _rewrite_replay_bytes_and_auth(root: Path) -> None:
    """Make a semantically equivalent replay/auth byte rewrite."""

    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=4) + "\n")
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)


def _rewrite_replay_auth_same_length_restore_mtime(root: Path) -> None:
    """Rewrite equivalent JSON while restoring size and timestamps."""

    replay_path = root / "canary-replay-freeze.json"
    replay_stat = replay_path.stat()
    replay_bytes = replay_path.read_bytes()
    marker = b"  "
    offset = replay_bytes.find(marker)
    assert offset >= 0
    replay_path.write_bytes(replay_bytes[:offset] + b" \t" + replay_bytes[offset + len(marker) :])
    os.utime(replay_path, ns=(replay_stat.st_atime_ns, replay_stat.st_mtime_ns))

    auth_path = root / "continuation-authorization.json"
    auth_stat = auth_path.stat()
    auth_bytes = auth_path.read_bytes()
    old_hash = _sha(replay_path)
    prior_hash = json.loads(auth_bytes.decode())["canary_replay_freeze_sha256"]
    assert len(old_hash) == len(prior_hash)
    auth_bytes = auth_bytes.replace(prior_hash.encode(), old_hash.encode(), 1)
    auth_path.write_bytes(auth_bytes)
    os.utime(auth_path, ns=(auth_stat.st_atime_ns, auth_stat.st_mtime_ns))


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
    with pytest.raises(
        ValueError,
        match="continuation (call cap is invalid|authorization bytes|snapshot drift)",
    ):
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


def test_self_consistent_imported_artifact_forgery_is_rejected_before_callback(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _rewrite_imported_artifacts_self_consistently(root)
    calls: list[tuple[Any, ...]] = []
    response_ids = iter(f"unused-{index}" for index in range(22))
    with pytest.raises(ValueError, match="imported raw response bytes"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call(next(response_ids)))
    assert calls == []


def test_post_run_imported_artifact_forgery_rejects_complete_freeze_creation(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    response_ids = iter(f"post-run-{index}" for index in range(22))
    runner.run(root, call=lambda *args: calls.append(args) or _id_call(next(response_ids)))
    _rewrite_imported_artifacts_self_consistently(root)
    with pytest.raises(ValueError, match="imported raw response bytes"):
        runner.verify_predictions_freeze(root)
    assert not (root / "predictions-freeze.json").exists()


def test_post_freeze_imported_artifact_forgery_rejects_existing_freeze_verification(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    _rewrite_imported_artifacts_self_consistently(root)
    with pytest.raises(ValueError, match="imported raw response bytes"):
        runner.verify_predictions_freeze(root)


def test_incomplete_freeze_creation_validates_imported_artifacts_before_write(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    responses = iter([_id_call("incomplete-01", valid=False), _id_call("incomplete-02")])

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        response = next(responses)
        if len(calls) == 2:
            _rewrite_imported_artifacts_self_consistently(root)
        return response

    with pytest.raises(ValueError, match="imported raw response bytes"):
        runner.run(root, call=call)
    assert len(calls) == 2
    assert not (root / "predictions-freeze.json").exists()


def test_post_freeze_incomplete_imported_artifact_forgery_rejects_verification(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    responses = iter([_id_call("incomplete-01", valid=False), _id_call("incomplete-02")])
    runner.run(root, call=lambda *args: next(responses))
    _rewrite_imported_artifacts_self_consistently(root)
    with pytest.raises(ValueError, match="imported raw response bytes"):
        runner.verify_predictions_freeze(root)


def test_post_run_replay_call_count_rejects_complete_freeze_creation(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    response_ids = iter(f"replay-create-{index}" for index in range(22))
    runner.run(root, call=lambda *_args: _id_call(next(response_ids)))
    _rewrite_replay_provider_calls(root)
    with pytest.raises(ValueError, match="replay provider_calls_made"):
        runner.verify_predictions_freeze(root)
    assert not (root / "predictions-freeze.json").exists()


def test_post_freeze_replay_call_count_rejects_existing_freeze_verification(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    _rewrite_replay_provider_calls(root)
    with pytest.raises(ValueError, match="replay provider_calls_made"):
        runner.verify_predictions_freeze(root)


@pytest.mark.parametrize("entry_kind", ["directory", "fifo"])
def test_continuation_rejects_non_regular_extra_entry_before_callback(
    tmp_path: Path, entry_kind: str
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    extra = runner._arm_directory(root, 1, "target", 4.0) / "unexpected-entry"
    if entry_kind == "directory":
        extra.mkdir()
    else:
        os.mkfifo(extra)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="exact regular-file artifact set"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_continuation_snapshot_stops_after_first_callback_replay_byte_drift(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        if len(calls) == 1:
            _rewrite_replay_bytes_self_consistently(root)
        return _id_call(f"snapshot-{len(calls)}")

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 1
    triggered = runner._arm_directory(root, 2, "target", 4.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not (triggered / "terminal.json").exists()
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_failed_continuation_snapshot_drift_after_position_three_never_calls_position_four(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        _rewrite_replay_bytes_self_consistently(root)
        return _id_call(f"snapshot-failed-{len(calls)}", valid=False)

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 1
    triggered = runner._arm_directory(root, 2, "target", 4.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not (triggered / "terminal.json").exists()
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_last_callback_snapshot_drift_fails_before_run_returns_or_freeze(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        if len(calls) == 22:
            _rewrite_replay_bytes_self_consistently(root)
        return _id_call(f"snapshot-last-{len(calls)}")

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 22
    triggered = runner._arm_directory(root, 29, "control", 12.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not (triggered / "terminal.json").exists()
    assert not (root / "predictions-freeze.json").exists()


@pytest.mark.parametrize("audit_index", [1, 3])
def test_continuation_snapshot_catches_imported_or_future_clip_drift_after_first_callback(
    tmp_path: Path, audit_index: int
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _detach_continuation_clips(root, tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        clip = _clip_path_for(root, audit_index)
        clip.write_bytes(clip.read_bytes() + b"clip drift")
        return _id_call(f"clip-drift-{len(calls)}")

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 1
    triggered = runner._arm_directory(root, 2, "target", 4.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not (triggered / "terminal.json").exists()
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_continuation_snapshot_catches_same_length_restored_mtime_replay_auth_drift(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        _rewrite_replay_auth_same_length_restore_mtime(root)
        return _id_call(f"same-length-drift-{len(calls)}")

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 1
    triggered = runner._arm_directory(root, 2, "target", 4.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not (triggered / "terminal.json").exists()
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_continuation_snapshot_catches_authorization_only_whitespace_drift(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        auth_path = root / "continuation-authorization.json"
        stat = auth_path.stat()
        raw = auth_path.read_bytes()
        changed = raw.replace(b'  "authorization_type"', b'\t "authorization_type"', 1)
        assert changed != raw and len(changed) == len(raw)
        auth_path.write_bytes(changed)
        os.utime(auth_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return _id_call("authorization-whitespace-drift", valid=False)

    with pytest.raises(ValueError, match="continuation snapshot"):
        runner.run(root, call=call)
    assert len(calls) == 1
    triggered = runner._arm_directory(root, 2, "target", 4.0)
    assert (triggered / "raw-response.txt").is_file()
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_continuation_freeze_is_independent_of_root_path_spelling(tmp_path: Path) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    relative_root = Path(os.path.relpath(root, Path.cwd()))
    responses = iter([_id_call("relative-01", valid=False), _id_call("relative-02")])
    runner.run(relative_root, call=lambda *_args: next(responses))
    runner.verify_predictions_freeze(root.resolve())


def test_continuation_raw_capture_collision_preserves_returned_raw_and_metadata(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    calls: list[tuple[Any, ...]] = []
    returned_raw = json.dumps(_valid_probe())

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        arm = runner._arm_directory(root, 2, "target", 4.0)
        (arm / "raw-response.txt").write_text("sentinel raw")
        _write_json(arm / "metadata.json", {"sentinel": True})
        return returned_raw, {"provider": "fake", "response_id": "collision-01"}

    with pytest.raises(ValueError, match="capture collision"):
        runner.run(root, call=call)
    assert len(calls) == 1
    capture_root = root / "callback-captures"
    capture_files = list(capture_root.rglob("*"))
    assert any(path.name == "raw-response.txt" and path.read_text() == returned_raw for path in capture_files)
    assert any(path.name == "metadata.json" for path in capture_files)
    assert any(path.name == "terminal.json" for path in capture_files)
    assert (runner._arm_directory(root, 2, "target", 4.0) / "raw-response.txt").read_text() == "sentinel raw"
    assert not runner._arm_directory(root, 2, "target", 12.0).exists()


def test_continuation_startup_rejects_stale_callback_capture_artifacts(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    stale = root / "callback-captures" / "position-03-target-4fps"
    stale.mkdir(parents=True)
    (stale / "raw-response.txt").write_text("stale")
    calls: list[tuple[Any, ...]] = []
    response_ids = iter(f"capture-stale-{index}" for index in range(22))
    with pytest.raises(ValueError, match="capture"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call(next(response_ids)))
    assert calls == []


def test_continuation_requires_committed_sanitized_ledger_before_callback(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["ledger_path"] = "docs/superpowers/ledgers/forged-ledger.json"
    _write_json(auth_path, auth)
    calls: list[tuple[Any, ...]] = []
    response_ids = iter(f"ledger-missing-{index}" for index in range(22))
    with pytest.raises(ValueError, match="committed ledger"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call(next(response_ids)))
    assert calls == []


def test_continuation_freeze_requires_call_time_snapshot_after_semantic_replay_rewrite(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    response_ids = iter(f"snapshot-freeze-{index}" for index in range(22))
    runner.run(root, call=lambda *_args: _id_call(next(response_ids)))
    _rewrite_replay_bytes_and_auth(root)
    with pytest.raises(ValueError, match="execution snapshot|continuation snapshot"):
        runner.verify_predictions_freeze(root)
    assert not (root / "predictions-freeze.json").exists()


def test_existing_continuation_freeze_rejects_semantic_replay_rewrite(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    _rewrite_replay_bytes_and_auth(root)
    with pytest.raises(ValueError, match="execution snapshot|continuation snapshot"):
        runner.verify_predictions_freeze(root)


def test_continuation_freeze_binds_write_once_execution_snapshot_for_complete_run(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _run_complete_continuation(root)
    snapshot = root / "execution-snapshot.json"
    finalized = root / "execution-snapshot-final.json"
    assert snapshot.is_file()
    assert finalized.is_file()
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["execution_snapshot_sha256"] == _sha(snapshot)
    assert freeze["execution_snapshot_final_sha256"] == _sha(finalized)
    assert json.loads(finalized.read_text())["call_count"] == 22


def test_continuation_failed_canary_freeze_binds_final_execution_snapshot(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    responses = iter([_id_call("snapshot-failed-01", valid=False), _id_call("snapshot-failed-02")])
    runner.run(root, call=lambda *_args: next(responses))
    snapshot = root / "execution-snapshot.json"
    finalized = root / "execution-snapshot-final.json"
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert snapshot.is_file()
    assert finalized.is_file()
    assert json.loads(finalized.read_text())["call_count"] == 2
    assert freeze["execution_snapshot_sha256"] == _sha(snapshot)
    assert freeze["execution_snapshot_final_sha256"] == _sha(finalized)


def _make_stale_pending_arm(root: Path, artifact: str) -> None:
    stale = runner._arm_directory(root, 2, "target", 4.0)
    stale.mkdir(parents=True)
    if artifact == "extra-dir":
        (stale / "unexpected").mkdir()
    else:
        if artifact == "raw-response.txt":
            (stale / artifact).write_text("stale raw")
        elif artifact == "metadata.json":
            _write_json(stale / artifact, {"stale": True})
        else:
            _write_json(stale / artifact, {"stale": True})


@pytest.mark.parametrize("artifact", ["request.json", "raw-response.txt", "metadata.json", "extra-dir"])
def test_continuation_startup_rejects_stale_pending_arm_without_callback(
    tmp_path: Path, artifact: str
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    _make_stale_pending_arm(root, artifact)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="continuation startup"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_continuation_pre_callback_rejects_extra_current_arm_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    original = runner._write_request

    def polluted_write_request(*args: Any, **kwargs: Any) -> str:
        prompt = original(*args, **kwargs)
        row_root = args[0]
        (row_root / "metadata.json").write_text("stale metadata")
        return prompt

    monkeypatch.setattr(runner, "_write_request", polluted_write_request)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="fresh callback arm"):
        runner.run(root, call=lambda *args: calls.append(args) or _id_call("unused"))
    assert calls == []


def test_replay_provider_calls_made_must_remain_integer_zero_with_recomputed_auth(
    tmp_path: Path,
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    replay["provider_calls_made"] = 17
    _write_json(replay_path, replay)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)
    with pytest.raises(ValueError, match="replay provider_calls_made"):
        runner.validate_continuation_authorization(root)


@pytest.mark.parametrize(
    "field",
    ("runner_sha256", "validator_sha256", "response_schema_sha256", "provider_response_schema_sha256"),
)
def test_replay_requires_current_runner_validator_and_schema_hashes(
    tmp_path: Path, field: str
) -> None:
    root = _prepare_continuation_fixture(tmp_path)
    replay_path = root / "canary-replay-freeze.json"
    replay = json.loads(replay_path.read_text())
    replay[field] = "forged-replay-hash"
    _write_json(replay_path, replay)
    auth_path = root / "continuation-authorization.json"
    auth = json.loads(auth_path.read_text())
    auth["canary_replay_freeze_sha256"] = _sha(replay_path)
    _write_json(auth_path, auth)
    with pytest.raises(ValueError, match=f"replay {field}"):
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
    expected = _expected_continuation_arms(root, start_position=3, stop_position=24)
    calls: list[tuple[Any, ...]] = []
    response_ids: list[str] = []

    def call(*args: Any) -> tuple[str, dict[str, Any]]:
        calls.append(args)
        response_id = f"ordered-{len(calls):02d}"
        response_ids.append(response_id)
        return _id_call(response_id)

    runner.run(root, call=call)
    runner.verify_predictions_freeze(root)
    _assert_exact_continuation_callbacks(root, calls, expected)
    assert [item["position"] for item in expected] == list(range(3, 25))
    provider_rows = json.loads((root / "provider-rows.json").read_text())["rows"]
    assert [item["audit_index"] for item in expected] == [
        row["audit_index"]
        for row in provider_rows
        for _ in runner.EXPECTED_FPS
    ][2:]
    assert [item["role"] for item in expected] == [
        next(
            row["role"]
            for row in json.loads((root / "media-manifest.json").read_text())["rows"]
            if row["audit_index"] == item["audit_index"]
        )
        for item in expected
    ]
    assert [item["fps"] for item in expected] == list(runner.EXPECTED_FPS) * 11
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
    expected = _expected_continuation_arms(root, start_position=3, stop_position=4)
    calls: list[tuple[Any, ...]] = []
    responses = iter([_id_call("fresh-01", valid=False), _id_call("fresh-02")])
    runner.run(root, call=lambda *args: calls.append(args) or next(responses))
    _assert_exact_continuation_callbacks(root, calls, expected)
    assert [item["position"] for item in expected] == [3, 4]
    assert [item["audit_index"] for item in expected] == [2, 2]
    assert [item["role"] for item in expected] == ["target", "target"]
    assert [item["fps"] for item in expected] == [4.0, 12.0]
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["completed_count"] == 4
    assert freeze["call_count"] == 4
    assert freeze["fresh_canary_pair_status"] == "failed"
    assert sum(
        json.loads(path.read_text())["status"] == "incomplete"
        for path in (root / "rows").glob("*/terminal.json")
    ) == 20
    runner.verify_predictions_freeze(root)
