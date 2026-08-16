from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_grounded_map_answering.py"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location("grounded_map_answering", SCRIPT)
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    selection_path = tmp_path / "selection.json"
    output = tmp_path / "run"
    target_ids = [f"target-{index:02d}" for index in range(12)]
    control_ids = [f"control-{index:02d}" for index in range(12)]
    all_ids = target_ids + control_ids
    _write_json(
        source / "protocol.json",
        {
            "schema_version": 1,
            "experiment_id": "frozen-source",
            "rows": [
                {
                    "video_id": video_id,
                    "question": f"Question for {video_id}?",
                    "question_stratum": "temporal" if index % 2 else "other",
                }
                for index, video_id in enumerate(all_ids)
            ],
        },
    )
    _write_json(
        source / "maps.question-blind.json",
        [
            {
                "video_id": video_id,
                "condition_id": "frozen-map",
                "status": "passed",
                "map": (
                    f"OVERVIEW | Summary for {video_id}\n"
                    f"SCENE | 00:01.000-00:02.000 | Evidence for {video_id}"
                ),
            }
            for video_id in all_ids
        ],
    )
    _write_json(
        source / "predictions.direct-answer.json",
        [
            {
                "video_id": video_id,
                "condition_id": "direct-answer",
                "status": "passed",
                "answer": f"Baseline {video_id}",
            }
            for video_id in all_ids
        ],
    )
    _write_json(
        source / "failure-audit-v1" / "manual-review.json",
        {
            "schema_version": 1,
            "rows": [
                {
                    "video_id": video_id,
                    "verdict": "incorrect" if index % 2 else "partial",
                    "primary_failure": "reasoning_over_available_evidence",
                    "map_sufficient": True,
                    "note": "Prior label-aware audit note.",
                }
                for index, video_id in enumerate(target_ids)
            ],
        },
    )
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "experiment_id": "grounded-map-answering-issue35-v1",
            "expected_target_count": 12,
            "expected_control_count": 12,
            "pairs": [
                {
                    "pair_id": f"pair-{index + 1:02d}",
                    "stratum": "temporal" if index % 2 else "other",
                    "target": {
                        "video_id": target_id,
                        "evidence_line_numbers": [2],
                    },
                    "control": {
                        "video_id": control_id,
                        "evidence_line_numbers": [1],
                    },
                }
                for index, (target_id, control_id) in enumerate(
                    zip(target_ids, control_ids, strict=True)
                )
            ],
        },
    )
    return source, selection_path, output


def _label_keys(value: Any) -> set[str]:
    forbidden = {
        "annotation",
        "annotations",
        "correct_answer",
        "expected_answer",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_answer",
        "reference_answers",
    }
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                found.add(key)
            found.update(_label_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_label_keys(child))
    return found


def test_prepare_freezes_audited_targets_controls_and_exact_map_lines(
    tmp_path: Path,
) -> None:
    assert MODULE is not None, "grounded-map runner is not implemented"
    source, selection_path, output = _fixture(tmp_path)

    MODULE.prepare(source, selection_path, output)

    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert len(protocol["rows"]) == 24
    assert {row["role"] for row in protocol["rows"]} == {"target", "control"}
    assert sum(row["role"] == "target" for row in protocol["rows"]) == 12
    assert sum(row["role"] == "control" for row in protocol["rows"]) == 12
    assert protocol["rows"][0]["evidence"] == [
        {
            "evidence_id": "E01",
            "map_line_number": 2,
            "text": "SCENE | 00:01.000-00:02.000 | Evidence for target-00",
        }
    ]
    assert protocol["rows"][1]["evidence"] == [
        {
            "evidence_id": "E01",
            "map_line_number": 1,
            "text": "OVERVIEW | Summary for control-00",
        }
    ]
    assert _label_keys(protocol) == set()
    assert set(protocol["source_artifacts"]) == {
        "protocol",
        "maps",
        "direct_predictions",
        "manual_audit",
        "selection",
    }


def test_prepare_rejects_target_not_marked_map_sufficient(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map runner is not implemented"
    source, selection_path, output = _fixture(tmp_path)
    audit_path = source / "failure-audit-v1" / "manual-review.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["rows"][0]["map_sufficient"] = False
    _write_json(audit_path, audit)

    with pytest.raises(ValueError, match="map-sufficient incorrect/partial"):
        MODULE.prepare(source, selection_path, output)


def test_prepare_rejects_reused_control(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map runner is not implemented"
    source, selection_path, output = _fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["pairs"][1]["control"]["video_id"] = "control-00"
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="control video IDs must be unique"):
        MODULE.prepare(source, selection_path, output)


def test_prepare_rejects_map_line_outside_frozen_map(tmp_path: Path) -> None:
    assert MODULE is not None, "grounded-map runner is not implemented"
    source, selection_path, output = _fixture(tmp_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["pairs"][0]["target"]["evidence_line_numbers"] = [3]
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="map line 3 does not exist"):
        MODULE.prepare(source, selection_path, output)


def test_prepare_records_additive_run_id_without_changing_selection(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)

    MODULE.prepare(
        source,
        selection_path,
        output,
        experiment_id="grounded-map-answering-issue35-v2",
    )

    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["experiment_id"] == "grounded-map-answering-issue35-v2"
    assert protocol["source_selection_experiment_id"] == (
        "grounded-map-answering-issue35-v1"
    )


def _valid_answer(*, evidence_id: str = "E01") -> dict[str, Any]:
    return {
        "question_interpretation": "Identify what happens in the selected scene.",
        "answer_claims": [
            {"claim": "The event happens.", "evidence_ids": [evidence_id]}
        ],
        "conflicts_or_missing_qualifiers": [],
        "needs_verification": False,
        "answer": "The event happens.",
    }


def test_validate_answer_accepts_cited_concise_top_one() -> None:
    assert MODULE is not None

    assert MODULE.validate_answer(_valid_answer(), {"E01"}) == _valid_answer()


def test_validate_answer_rejects_unknown_evidence_citation() -> None:
    assert MODULE is not None

    with pytest.raises(ValueError, match="unknown evidence ID"):
        MODULE.validate_answer(_valid_answer(evidence_id="E99"), {"E01"})


def test_validate_answer_rejects_uncited_nonverification_answer() -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload["answer_claims"] = []

    with pytest.raises(ValueError, match="requires at least one cited claim"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_rejects_final_answer_not_bound_to_cited_claim() -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload["answer"] = "Aliens protected the adjacent buildings."

    with pytest.raises(ValueError, match="exactly match one cited claim"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_requires_fixed_verification_route_without_guess() -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload.update(
        {
            "answer_claims": [],
            "conflicts_or_missing_qualifiers": ["The supplied span lacks the count."],
            "needs_verification": True,
            "answer": "Probably three.",
        }
    )

    with pytest.raises(ValueError, match="answer must be NEEDS VERIFICATION"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_rejects_multiline_or_overlong_answer() -> None:
    assert MODULE is not None
    multiline = _valid_answer()
    multiline["answer"] = "First line.\nSecond line."
    with pytest.raises(ValueError, match="one line"):
        MODULE.validate_answer(multiline, {"E01"})

    overlong = _valid_answer()
    overlong["answer"] = " ".join(["word"] * 25)
    with pytest.raises(ValueError, match="at most 24 words"):
        MODULE.validate_answer(overlong, {"E01"})


def test_answer_prompt_contains_only_question_and_frozen_evidence() -> None:
    assert MODULE is not None
    prompt = MODULE.answer_prompt(
        "What happens next?",
        [{"evidence_id": "E01", "text": "SCENE | 1-2 | A person waves."}],
    )

    assert "What happens next?" in prompt
    assert "[E01] SCENE | 1-2 | A person waves." in prompt
    assert "needs_verification" in prompt
    assert "one concise top-1 answer" in prompt
    assert "native video" not in prompt.lower()


def test_google_response_schema_omits_unsupported_additional_properties() -> None:
    assert MODULE is not None

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert "additionalProperties" not in value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(MODULE.ANSWER_SCHEMA)


def test_run_row_records_terminal_provider_error(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)

    def fail_provider(_prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        MODULE.run_row(output, "target-00", call_fn=fail_provider)

    terminal = json.loads(
        (output / "rows" / "target-00" / "terminal.json").read_text()
    )
    assert terminal == {
        "error_type": "RuntimeError",
        "status": "provider_error",
        "video_id": "target-00",
    }


def test_run_row_preserves_raw_and_metadata_before_invalid_json_contract_error(
    tmp_path: Path,
) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)

    def malformed_provider(_prompt: str) -> tuple[str, dict[str, Any]]:
        return "not valid json", {"total_token_count": 17}

    with pytest.raises(json.JSONDecodeError):
        MODULE.run_row(output, "target-00", call_fn=malformed_provider)

    row_root = output / "rows" / "target-00"
    assert (row_root / "raw-response.txt").read_text() == "not valid json"
    assert json.loads((row_root / "metadata.json").read_text()) == {
        "total_token_count": 17
    }
    terminal = json.loads((row_root / "terminal.json").read_text())
    assert terminal["status"] == "contract_error"
    assert terminal["error_type"] == "JSONDecodeError"


def test_run_row_rejects_changed_frozen_source(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)
    maps_path = source / "maps.question-blind.json"
    maps_path.write_text(maps_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen source artifact changed"):
        MODULE.run_row(output, "target-00", call_fn=lambda _prompt: ({}, "", {}))


def test_freeze_binds_all_passed_terminal_predictions(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)
    protocol = json.loads((output / "protocol.json").read_text())

    def pass_provider(_prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
        payload = _valid_answer()
        return payload, json.dumps(payload), {"total_token_count": 10}

    for row in protocol["rows"]:
        MODULE.run_row(output, row["video_id"], call_fn=pass_provider)
    MODULE.freeze(output)

    freeze = json.loads((output / "predictions-freeze.json").read_text())
    assert freeze["status"] == "frozen"
    assert freeze["row_count"] == 24
    assert freeze["passed_count"] == 24
    assert freeze["failed_count"] == 0
    assert len(freeze["rows"]) == 24
    assert all(len(row["terminal_sha256"]) == 64 for row in freeze["rows"])
    assert len(freeze["protocol_sha256"]) == 64
    evidence_freeze = json.loads((output / "evidence-freeze.json").read_text())
    assert evidence_freeze["prediction_freeze_sha256"] == MODULE._sha256(
        output / "predictions-freeze.json"
    )
    assert set(evidence_freeze["rows"][0]["artifacts"]) == {
        "metadata.json",
        "parsed.json",
        "raw-response.txt",
        "request.json",
        "terminal.json",
    }


def test_freeze_requires_a_terminal_state_for_every_row(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)

    with pytest.raises(ValueError, match="row has no terminal state"):
        MODULE.freeze(output)


def test_freeze_gate_binds_exact_promotion_thresholds(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    MODULE.prepare(source, selection_path, output)

    MODULE.freeze_gate(output)

    gate = json.loads((output / "preinference-gate-freeze.json").read_text())
    assert gate["rule"] == {
        "minimum_target_rescues": 6,
        "maximum_control_regressions": 1,
        "require_all_answer_claims_citation_supported": True,
        "require_no_unnecessary_verification_routes": True,
        "require_changed_row_audit_complete": True,
    }
    assert len(gate["protocol_sha256"]) == 64
