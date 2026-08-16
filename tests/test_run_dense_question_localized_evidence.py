from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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


def _write_authorized_fixture(tmp_path: Path, *, rows: int = 2) -> Path:
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
                "role": "target" if index == 0 else "control",
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
                "role": "target" if index == 0 else "control",
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
            "runner_sha256": _sha(runner.SCRIPT_PATH),
            "config_sha256": _sha(ROOT / "configs/vqa/dense_question_localized_evidence_issue35.json"),
            "prompt_template_sha256": runner._prompt_template_sha256(),
            "response_schema_sha256": runner._schema_sha256(),
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
            "target_count": 1 if rows > 0 else 0,
            "control_count": max(rows - 1, 0),
            "planned_call_count": rows * 2,
        },
    )
    _write_json(
        root / "provider-authorization.json",
        {
            "schema_version": 1,
            "status": "authorized",
            "promotion_gate": "PASS",
            "treatment_freeze_sha256": _sha(root / "treatment-freeze.json"),
            "selection_freeze_sha256": _sha(root / "selection-freeze.json"),
            "provider_rows_sha256": runner.provider_rows_sha256(provider_rows),
            "media_manifest_sha256": _sha(root / "media-manifest.json"),
            "planned_call_count": rows * 2,
            "max_calls": runner.MAX_CALLS,
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
    root = _write_authorized_fixture(tmp_path, rows=2)
    calls: list[tuple[Any, ...]] = []
    runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert len(calls) == 4
    assert len(list((root / "rows").glob("*/terminal.json"))) == 4
    runner.verify_predictions_freeze(root)
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    assert freeze["completed_count"] == 4
    assert freeze["call_count"] == 4


def test_run_rejects_label_bearing_provider_rows_before_call(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path, rows=2)
    payload = json.loads((root / "provider-rows.json").read_text())
    payload["rows"][0]["reference"] = "hidden"
    _write_json(root / "provider-rows.json", payload)
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="provider-safe"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []


def test_run_preserves_malformed_json_and_schema_failure_without_retry(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path, rows=2)
    responses = iter(
        [
            ("{broken", {"provider": "fake"}),
            (json.dumps({"wrong": True}), {"provider": "fake"}),
            _valid_call(),
            _valid_call(),
        ]
    )
    calls: list[tuple[Any, ...]] = []
    runner.run(root, call=lambda *args: calls.append(args) or next(responses))
    assert len(calls) == 4
    statuses = sorted(json.loads(path.read_text())["status"] for path in (root / "rows").glob("*/terminal.json"))
    assert statuses == ["contract_error", "invalid_json", "passed", "passed"]
    malformed = next(path.parent for path in (root / "rows").glob("*/terminal.json") if json.loads(path.read_text())["status"] == "invalid_json")
    assert (malformed / "raw-response.txt").read_text() == "{broken"


def test_run_refuses_write_once_and_detects_artifact_drift(tmp_path: Path) -> None:
    root = _write_authorized_fixture(tmp_path, rows=2)
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
    root = _write_authorized_fixture(tmp_path, rows=2)
    clip = next((root / "clips").glob("*.mp4"))
    clip.write_bytes(b"changed")
    calls: list[tuple[Any, ...]] = []
    with pytest.raises(ValueError, match="clip"):
        runner.run(root, call=lambda *args: calls.append(args) or _valid_call())
    assert calls == []
