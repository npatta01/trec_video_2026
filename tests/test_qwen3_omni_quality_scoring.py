import hashlib
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments import qwen3_omni_quality_scoring as scoring


def _write_frozen_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a complete one-pair frozen run and its hidden annotation."""
    data_root = tmp_path / "data"
    run_dir = data_root / "run"
    run_dir.mkdir(parents=True)
    annotation = data_root / "vqa/2025/training/annotations/Mi-NpnCBegw.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps({"correct_answer": "private label words"}), encoding="utf-8"
    )
    contract = {
        "examples": [{"video_id": "Mi-NpnCBegw"}],
        "conditions": [
            {"condition_id": "frames-only", "audio_enabled": False},
            {"condition_id": "frames-native-audio", "audio_enabled": True},
        ],
    }
    contract_path = run_dir / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    rows = [
        {
            "run_id": "test-run",
            "video_id": "Mi-NpnCBegw",
            "condition_id": "frames-only",
            "audio_enabled": False,
            "pair_index": 0,
            "status": "passed",
            "output_parse_status": "passed",
            "parsed_output": {"answer": "1"},
        },
        {
            "run_id": "test-run",
            "video_id": "Mi-NpnCBegw",
            "condition_id": "frames-native-audio",
            "audio_enabled": True,
            "pair_index": 0,
            "status": "passed",
            "output_parse_status": "passed",
            "parsed_output": {"answer": "two"},
        },
    ]
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    prediction_hash = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    contract_hash = scoring.canonical_sha256(contract)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "rows": 2,
                "succeeded": True,
                "canary_passed": True,
                "predictions_sha256": prediction_hash,
                "quality_contract_sha256": contract_hash,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "inputs": {"quality_contract_sha256": contract_hash},
                "outputs": {"predictions.jsonl": prediction_hash},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "terminal-status.json").write_text(
        json.dumps({"workload_returncode": 0}), encoding="utf-8"
    )
    return data_root, run_dir, contract_path


def _score(tmp_path: Path) -> dict[str, object]:
    data_root, run_dir, contract_path = _write_frozen_inputs(tmp_path)
    return scoring.score_run(
        data_root=data_root,
        run_dir=run_dir,
        contract_path=contract_path,
        expected_run_id="test-run",
        expected_predictions_sha256=hashlib.sha256(
            (run_dir / "predictions.jsonl").read_bytes()
        ).hexdigest(),
        expected_contract_sha256=scoring.canonical_sha256(
            json.loads(contract_path.read_text(encoding="utf-8"))
        ),
        manual_judgments={
            ("Mi-NpnCBegw", "frames-only"): {
                "judgment": "C",
                "reason": "Numeric equivalence under the frozen scorer rule.",
            },
            ("Mi-NpnCBegw", "frames-native-audio"): {
                "judgment": "W",
                "reason": "The response gives a different count.",
            },
        },
        scorer_source_path=Path(__file__),
    )


def test_score_run_binds_frozen_hashes_and_joins_adjacent_conditions(
    tmp_path: Path,
):
    """Catches scoring a changed prediction file or an unpaired condition order."""
    result = _score(tmp_path)

    assert result["predictions_sha256"] == hashlib.sha256(
        (tmp_path / "data/run/predictions.jsonl").read_bytes()
    ).hexdigest()
    assert result["aggregate"]["frames-only"]["normalized_exact_match"] == 0.0
    assert result["aggregate"]["frames-native-audio"]["normalized_exact_match"] == 0.0
    assert result["paired_deltas"]["audio_minus_frames"]["normalized_exact_match"] == 0.0


def test_score_run_keeps_hidden_reference_text_out_of_sanitized_result(
    tmp_path: Path,
):
    """Catches accidental serialization of annotation answers into tracked output."""
    result = _score(tmp_path)
    serialized = json.dumps(result)

    assert "private label words" not in serialized
    assert result["examples"][0]["reference_sha256"] == hashlib.sha256(
        b"private label words"
    ).hexdigest()
    assert result["examples"][0]["annotation_path"] == (
        "data/vqa/2025/training/annotations/Mi-NpnCBegw.json"
    )


def test_numeric_words_are_equivalent_for_exact_match_and_token_f1():
    """Catches regressing the documented 1/one numeric-equivalence rule."""
    assert scoring.normalized_exact_match("1", "one") == 1.0
    assert scoring.token_f1("1", "one") == 1.0


def test_score_run_rejects_incomplete_or_unfrozen_artifacts(tmp_path: Path):
    """Catches adjudicating an incomplete run or a manifest with changed inputs."""
    data_root, run_dir, contract_path = _write_frozen_inputs(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["succeeded"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="complete frozen run"):
        scoring.score_run(
            data_root=data_root,
            run_dir=run_dir,
            contract_path=contract_path,
            expected_run_id="test-run",
            expected_predictions_sha256=hashlib.sha256(
                (run_dir / "predictions.jsonl").read_bytes()
            ).hexdigest(),
            expected_contract_sha256=scoring.canonical_sha256(
                json.loads(contract_path.read_text(encoding="utf-8"))
            ),
            manual_judgments={},
            scorer_source_path=Path(__file__),
        )


def test_score_run_requires_every_answer_only_manual_judgment(tmp_path: Path):
    """Catches silently publishing C/P/W placeholders without adjudication."""
    data_root, run_dir, contract_path = _write_frozen_inputs(tmp_path)

    with pytest.raises(ValueError, match="manual judgment"):
        scoring.score_run(
            data_root=data_root,
            run_dir=run_dir,
            contract_path=contract_path,
            expected_run_id="test-run",
            expected_predictions_sha256=hashlib.sha256(
                (run_dir / "predictions.jsonl").read_bytes()
            ).hexdigest(),
            expected_contract_sha256=scoring.canonical_sha256(
                json.loads(contract_path.read_text(encoding="utf-8"))
            ),
            manual_judgments={
                ("Mi-NpnCBegw", "frames-only"): {
                    "judgment": "C",
                    "reason": "Numeric equivalence.",
                }
            },
            scorer_source_path=Path(__file__),
        )
