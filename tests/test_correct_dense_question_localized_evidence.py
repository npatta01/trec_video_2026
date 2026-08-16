from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "correct_dense_question_localized_evidence.py"
spec = importlib.util.spec_from_file_location("correct_dense_question_localized_evidence", SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load correction module from {SCRIPT}")
correction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correction)


SOURCE_ROOT = (
    ROOT
    / "data"
    / "vqa"
    / "benchmarks"
    / "dense-question-localized-evidence-issue35-v1"
    / "run-v22"
)


def _valid_probe(*, timestamp: float = 0.5) -> dict[str, Any]:
    return {
        "question_interpretation": {
            "requested_entity": "the robot",
            "requested_visual_fact": "final action",
            "temporal_qualifier": "at the end",
        },
        "fact_observed": True,
        "observed_fact": "The robot clicks its feet.",
        "evidence_timestamps": [timestamp],
        "conflicts_or_uncertainty": [],
        "needs_more_evidence": False,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_source(tmp_path: Path) -> Path:
    destination = tmp_path / "source"
    shutil.copytree(SOURCE_ROOT, destination, symlinks=True)
    return destination


def test_one_frame_encoded_overrun_is_accepted_symmetrically() -> None:
    result = correction.validate_corrected_probe(
        _valid_probe(timestamp=4.0),
        encoded_duration_seconds=4.001,
        nominal_duration_seconds=4.0,
        fps=1000.0,
    )
    assert result["evidence_timestamps"] == [4.0]


def test_encoded_overrun_beyond_one_requested_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="one frame"):
        correction.validate_corrected_probe(
            _valid_probe(),
            encoded_duration_seconds=4.251,
            nominal_duration_seconds=4.0,
            fps=4.0,
        )


def test_nominal_endpoint_timestamp_is_accepted_without_rewriting() -> None:
    result = correction.validate_corrected_probe(
        _valid_probe(timestamp=4.0),
        encoded_duration_seconds=4.001,
        nominal_duration_seconds=4.0,
        fps=12.0,
    )
    assert result["evidence_timestamps"] == [4.0]


def test_timestamp_past_nominal_endpoint_is_rejected_even_when_encoded_duration_is_longer() -> None:
    with pytest.raises(ValueError, match="nominal duration"):
        correction.validate_corrected_probe(
            _valid_probe(timestamp=4.001),
            encoded_duration_seconds=4.001,
            nominal_duration_seconds=4.0,
            fps=12.0,
        )


def test_one_frame_shortfall_remains_accepted_but_larger_shortfall_is_rejected() -> None:
    assert correction.validate_corrected_probe(
        _valid_probe(timestamp=3.9),
        encoded_duration_seconds=3.75,
        nominal_duration_seconds=4.0,
        fps=4.0,
    )["evidence_timestamps"] == [3.9]
    with pytest.raises(ValueError, match="one frame"):
        correction.validate_corrected_probe(
            _valid_probe(),
            encoded_duration_seconds=3.7,
            nominal_duration_seconds=4.0,
            fps=4.0,
        )


def test_strict_extra_fields_are_rejected() -> None:
    payload = _valid_probe()
    with pytest.raises(ValueError, match="fields"):
        correction.validate_corrected_probe(
            {**payload, "answer": "The robot clicks."},
            encoded_duration_seconds=4.0,
            nominal_duration_seconds=4.0,
            fps=4.0,
        )


def test_all_v22_raws_replay_with_zero_calls_and_expected_correction_count(tmp_path: Path) -> None:
    output_root = tmp_path / "correction"
    result = correction.correct_run(SOURCE_ROOT, output_root)
    assert result["provider_calls"] == 0
    assert result["aggregate"] == {
        "arms": 24,
        "original_passed": 16,
        "original_contract_error": 8,
        "corrected_passed": 24,
        "corrected_contract_error": 0,
        "corrected_acceptance_changes": 8,
    }
    correction.verify_correction(SOURCE_ROOT, output_root)


def test_source_raw_drift_is_rejected_before_correction(tmp_path: Path) -> None:
    source = _copy_source(tmp_path)
    raw = source / "rows" / "19-target-12fps" / "raw-response.txt"
    raw.write_text(raw.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash|source"):
        correction.correct_run(source, tmp_path / "correction")


def test_freeze_drift_is_rejected_by_independent_verifier(tmp_path: Path) -> None:
    output_root = tmp_path / "correction"
    correction.correct_run(SOURCE_ROOT, output_root)
    freeze = output_root / "correction-freeze.json"
    payload = json.loads(freeze.read_text())
    payload["provider_calls"] = 1
    freeze.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="freeze|provider"):
        correction.verify_correction(SOURCE_ROOT, output_root)


def test_corrected_output_drift_is_rejected_by_independent_verifier(tmp_path: Path) -> None:
    output_root = tmp_path / "correction"
    correction.correct_run(SOURCE_ROOT, output_root)
    parsed = next((output_root / "parsed").glob("*/parsed.json"))
    parsed.write_text(parsed.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parsed|correction"):
        correction.verify_correction(SOURCE_ROOT, output_root)


def test_correction_artifacts_are_label_free_and_bind_all_ordered_source_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "correction"
    correction.correct_run(SOURCE_ROOT, output_root)
    freeze = json.loads((output_root / "correction-freeze.json").read_text())
    assert freeze["source_prediction_freeze_sha256"] == correction.TRUSTED_V22_PREDICTION_FREEZE_SHA256
    assert len(freeze["arms"]) == 24
    assert [item["position"] for item in freeze["arms"]] == list(range(1, 25))
    assert len({item["response_id"] for item in freeze["arms"]}) == 24
    assert all(
        set(item["source_artifacts"]) == {"request.json", "raw-response.txt", "metadata.json", "terminal.json"}
        for item in freeze["arms"]
    )
    forbidden = {"answer", "label", "reference", "score", "prediction", "gold"}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert not forbidden.intersection(key.casefold() for key in value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(freeze)
