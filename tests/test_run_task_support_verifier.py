from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_task_support_verifier.py"
spec = importlib.util.spec_from_file_location("run_task_support_verifier", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load verifier module from {SCRIPT}")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _valid_payload() -> dict[str, Any]:
    return {
        "question_interpretation": {
            "requested_entity": "the man",
            "requested_fact": "action after sitting",
            "qualifier": "after",
            "answer_unit": "action",
        },
        "cited_evidence": [{"evidence_id": "E01", "claim": "He waves."}],
        "conflicts_or_missing_qualifiers": [],
        "needs_verification": False,
        "answer": "He waves.",
    }


def _write_protocol_fixture(tmp_path: Path, row_count: int = 12) -> Path:
    protocol_root = tmp_path / "protocol"
    rows = [
        {
            "video_id": f"v{index:02d}",
            "question": f"What does the man do in row {index}?",
            "evidence": [{"evidence_id": "E01", "claim": "He waves."}],
        }
        for index in range(row_count)
    ]
    _write(
        protocol_root / "protocol.json",
        {
            "schema_version": 1,
            "experiment_id": "task-support-test",
            "model": "gemini-3.1-pro-preview",
            "rows": rows,
        },
    )
    _write(
        protocol_root / "treatment-freeze.json",
        {"status": "frozen", "schema_version": 1},
    )
    _write(
        protocol_root / "selection-freeze.json",
        {"status": "frozen", "schema_version": 1, "row_count": row_count},
    )
    return protocol_root


def _fake_valid_call(_prompt: str) -> tuple[str, dict[str, Any]]:
    return json.dumps(_valid_payload()), {"provider": "fake", "total_token_count": 10}


def _fake_malformed_call(_prompt: str) -> tuple[str, dict[str, Any]]:
    return "{broken", {"provider": "fake"}


def test_validator_requires_unique_supported_answer_or_verification() -> None:
    answered = verifier.validate_answer(_valid_payload(), {"E01"})
    assert answered["answer"] == "He waves."

    with pytest.raises(ValueError, match="cited_evidence"):
        verifier.validate_answer({**answered, "cited_evidence": []}, {"E01"})
    with pytest.raises(ValueError, match="verification"):
        verifier.validate_answer(
            {**answered, "needs_verification": True, "answer": "He waves."},
            {"E01"},
        )


def test_validator_rejects_unknown_or_duplicate_evidence_ids() -> None:
    unknown = _valid_payload()
    unknown["cited_evidence"][0]["evidence_id"] = "E99"
    with pytest.raises(ValueError, match="unknown evidence"):
        verifier.validate_answer(unknown, {"E01"})

    duplicate = _valid_payload()
    duplicate["cited_evidence"].append(
        {"evidence_id": "E01", "claim": "He waves again."}
    )
    with pytest.raises(ValueError, match="unique"):
        verifier.validate_answer(duplicate, {"E01"})


def test_validator_rejects_absent_or_ambiguous_answered_conflicts() -> None:
    for conflict in ("The requested fact is absent.", "The evidence is ambiguous."):
        payload = _valid_payload()
        payload["conflicts_or_missing_qualifiers"] = [conflict]
        with pytest.raises(ValueError, match="verification"):
            verifier.validate_answer(payload, {"E01"})


def test_prompt_forbids_labels_guessing_and_unsupported_inference() -> None:
    prompt = verifier.build_prompt(
        "What does the man do?",
        [{"evidence_id": "E01", "claim": "He waves."}],
    ).lower()
    for required_text in (
        "labels",
        "reference",
        "purpose",
        "cause",
        "count",
        "speech",
        "multiple candidate",
    ):
        assert required_text in prompt


def test_run_writes_complete_one_attempt_chain_and_freeze(tmp_path: Path) -> None:
    protocol_root = _write_protocol_fixture(tmp_path)
    output_root = tmp_path / "run"

    verifier.run(protocol_root, output_root, call=_fake_valid_call)
    verifier.verify_predictions_freeze(output_root)

    assert len(list((output_root / "rows").glob("*/terminal.json"))) == 12
    freeze = json.loads((output_root / "predictions-freeze.json").read_text())
    assert freeze["completed_count"] == 12
    assert freeze["passed_count"] == 12
    for row_root in (output_root / "rows").iterdir():
        assert {
            "request.json",
            "raw-response.txt",
            "metadata.json",
            "parsed.json",
            "terminal.json",
        } == {path.name for path in row_root.iterdir()}


def test_run_preserves_malformed_raw_response(tmp_path: Path) -> None:
    protocol_root = _write_protocol_fixture(tmp_path)
    output_root = tmp_path / "run"

    verifier.run(protocol_root, output_root, call=_fake_malformed_call)

    row_root = next((output_root / "rows").iterdir())
    assert (row_root / "raw-response.txt").read_text() == "{broken"
    assert json.loads((row_root / "terminal.json").read_text())["status"] == "invalid_json"


def test_run_is_write_once_and_freeze_detects_artifact_drift(tmp_path: Path) -> None:
    protocol_root = _write_protocol_fixture(tmp_path)
    output_root = tmp_path / "run"
    verifier.run(protocol_root, output_root, call=_fake_valid_call)
    with pytest.raises(ValueError, match="already has artifacts"):
        verifier.run(protocol_root, output_root, call=_fake_valid_call)

    verifier.verify_predictions_freeze(output_root)
    row_root = next((output_root / "rows").iterdir())
    request = row_root / "request.json"
    request.write_text(request.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        verifier.verify_predictions_freeze(output_root)


def test_verify_predictions_freeze_rejects_protocol_drift(tmp_path: Path) -> None:
    protocol_root = _write_protocol_fixture(tmp_path)
    output_root = tmp_path / "run"
    verifier.run(protocol_root, output_root, call=_fake_valid_call)
    verifier.verify_predictions_freeze(output_root)

    protocol = output_root / "protocol.json"
    digest_before = hashlib.sha256(protocol.read_bytes()).hexdigest()
    protocol.write_text(protocol.read_text().replace("row 0", "changed row 0"))
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() != digest_before
    with pytest.raises(ValueError, match="protocol"):
        verifier.verify_predictions_freeze(output_root)
