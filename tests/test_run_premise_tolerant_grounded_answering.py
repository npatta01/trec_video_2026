from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_premise_tolerant_grounded_answering.py"
CONFIG = ROOT / "configs" / "vqa" / "premise_tolerant_grounded_answering_issue35.json"
MODULE = None
if SCRIPT.exists():
    spec = importlib.util.spec_from_file_location(
        "run_premise_tolerant_grounded_answering", SCRIPT
    )
    assert spec and spec.loader
    MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MODULE)


PANEL = {
    "fix_false_abstention": [
        "D_HTq7IiSPo",
        "KBb0DfsH0eg",
        "LNb5uNQXkwg",
        "PWP17NlIgs8",
        "BDKkVYmWqdM",
    ],
    "detectable_insufficiency": ["G2w7PrdwYhM", "LkdfiZcQ7lU"],
    "known_limit": ["HcvOhX3n784"],
    "stability_control": ["Bvv6lMUTlEw", "GWCGF1WpMPc", "JTXpUxVg144"],
}
IDS = [video_id for role in PANEL.values() for video_id in role]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source-protocol.json"
    _write(
        source,
        {
            "schema_version": 1,
            "experiment_id": "grounded-map-answering-issue35-v2",
            "rows": [
                {
                    "video_id": video_id,
                    "pair_id": f"old-pair-{index:02d}",
                    "role": "old-role",
                    "stratum": "old-stratum",
                    "question": f"Question for {video_id}?",
                    "evidence": [
                        {
                            "evidence_id": "E01",
                            "map_line_number": 1,
                            "text": f"SCENE | Evidence for {video_id}",
                        }
                    ],
                }
                for index, video_id in enumerate(IDS)
            ],
        },
    )
    selection = tmp_path / "selection.json"
    _write(
        selection,
        {
            "schema_version": 1,
            "experiment_id": "grounded-map-answering-issue35-premise-v3",
            "source_protocol_sha256": _sha(source),
            "panel": PANEL,
            "gate": {
                "minimum_false_abstention_rescues": 4,
                "required_detectable_insufficiency_verifications": 2,
                "required_stability_controls_preserved": 3,
                "maximum_unsupported_final_answers": 0,
                "require_full_manual_audit": True,
            },
        },
    )
    return source, selection, tmp_path / "run"


def _valid_answer(*, evidence_id: str = "E01") -> dict[str, Any]:
    return {
        "question_interpretation": "Identify the supported requested event.",
        "premise_status": "harmless_locator_variance",
        "answer_claims": [
            {"claim": "The supported event happens.", "evidence_ids": [evidence_id]}
        ],
        "conflicts_or_missing_qualifiers": ["The locator wording differs."],
        "needs_verification": False,
        "answer": "The supported event happens.",
    }


def test_tracked_selection_has_exact_approved_panel() -> None:
    assert CONFIG.is_file()
    selection = json.loads(CONFIG.read_text())
    assert selection["panel"] == PANEL
    assert selection["gate"] == {
        "minimum_false_abstention_rescues": 4,
        "required_detectable_insufficiency_verifications": 2,
        "required_stability_controls_preserved": 3,
        "maximum_unsupported_final_answers": 0,
        "require_full_manual_audit": True,
    }


def test_prepare_writes_label_free_protocol_and_separate_gate(tmp_path: Path) -> None:
    assert MODULE is not None, "premise-tolerant runner is not implemented"
    source, selection, output = _fixture(tmp_path)

    MODULE.prepare(source, selection, output)

    protocol = json.loads((output / "protocol.json").read_text())
    assert [row["video_id"] for row in protocol["rows"]] == IDS
    forbidden = {
        "role",
        "expected_route",
        "threshold",
        "reference",
        "baseline",
        "assessment",
        "label",
        "stratum",
        "pair_id",
    }
    assert not any(key in forbidden for row in protocol["rows"] for key in row)
    assert protocol["source_artifacts"]["v2_protocol"]["sha256"] == _sha(source)
    gate = json.loads((output / "preinference-gate-freeze.json").read_text())
    assert gate["panel"] == PANEL
    assert gate["protocol_sha256"] == _sha(output / "protocol.json")
    assert not (output / "rows").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source_hash", "source protocol hash"),
        ("duplicate", "panel IDs must be unique"),
        ("missing", "panel IDs must match"),
        ("role_count", "panel role counts"),
    ],
)
def test_prepare_rejects_invalid_selection(
    tmp_path: Path, mutation: str, message: str
) -> None:
    assert MODULE is not None
    source, selection_path, output = _fixture(tmp_path)
    selection = json.loads(selection_path.read_text())
    if mutation == "source_hash":
        selection["source_protocol_sha256"] = "0" * 64
    elif mutation == "duplicate":
        selection["panel"]["stability_control"][0] = "D_HTq7IiSPo"
    elif mutation == "missing":
        selection["panel"]["known_limit"][0] = "not-in-source"
    else:
        selection["panel"]["known_limit"].append("extra-known-limit")
    _write(selection_path, selection)

    with pytest.raises(ValueError, match=message):
        MODULE.prepare(source, selection_path, output)


@pytest.mark.parametrize(
    "premise_status",
    ["supported", "harmless_locator_variance", "answer_blocking"],
)
def test_validate_answer_accepts_all_premise_statuses(premise_status: str) -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload["premise_status"] = premise_status

    assert MODULE.validate_answer(payload, {"E01"})["premise_status"] == premise_status


def test_validate_answer_requires_final_answer_to_equal_cited_claim() -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload["answer"] = "An unsupported different answer."

    with pytest.raises(ValueError, match="exactly match one cited claim"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_rejects_unknown_citation_and_extra_fields() -> None:
    assert MODULE is not None
    with pytest.raises(ValueError, match="unknown evidence ID"):
        MODULE.validate_answer(_valid_answer(evidence_id="E99"), {"E01"})
    payload = _valid_answer()
    payload["confidence"] = 1
    with pytest.raises(ValueError, match="exactly the required fields"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_requires_clean_verification_route() -> None:
    assert MODULE is not None
    payload = _valid_answer()
    payload.update(
        {
            "premise_status": "answer_blocking",
            "answer_claims": [],
            "conflicts_or_missing_qualifiers": ["The requested purpose is absent."],
            "needs_verification": True,
            "answer": "NEEDS VERIFICATION",
        }
    )
    assert MODULE.validate_answer(payload, {"E01"}) == payload

    payload["answer_claims"] = [
        {"claim": "A guess.", "evidence_ids": ["E01"]}
    ]
    with pytest.raises(ValueError, match="must not contain answer claims"):
        MODULE.validate_answer(payload, {"E01"})


def test_validate_answer_rejects_multiline_and_overlong_answers() -> None:
    assert MODULE is not None
    multiline = _valid_answer()
    multiline["answer"] = multiline["answer_claims"][0]["claim"] = "one\ntwo"
    with pytest.raises(ValueError, match="one line"):
        MODULE.validate_answer(multiline, {"E01"})

    overlong = _valid_answer()
    overlong["answer"] = overlong["answer_claims"][0]["claim"] = " ".join(
        ["word"] * 25
    )
    with pytest.raises(ValueError, match="at most 24 words"):
        MODULE.validate_answer(overlong, {"E01"})


def test_run_row_preserves_malformed_paid_response(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection, output = _fixture(tmp_path)
    MODULE.prepare(source, selection, output)

    def malformed(_prompt: str) -> tuple[str, dict[str, Any]]:
        return "not json", {"total_token_count": 9}

    with pytest.raises(json.JSONDecodeError):
        MODULE.run_row(output, IDS[0], call_fn=malformed)

    row_root = output / "rows" / IDS[0]
    assert (row_root / "raw-response.txt").read_text() == "not json"
    assert json.loads((row_root / "metadata.json").read_text()) == {
        "total_token_count": 9
    }
    terminal = json.loads((row_root / "terminal.json").read_text())
    assert terminal["status"] == "contract_error"
    assert terminal["error_type"] == "JSONDecodeError"


def test_run_row_is_one_attempt_and_rejects_source_drift(tmp_path: Path) -> None:
    assert MODULE is not None
    source, selection, output = _fixture(tmp_path)
    MODULE.prepare(source, selection, output)

    payload = _valid_answer()
    MODULE.run_row(
        output,
        IDS[0],
        call_fn=lambda _prompt: (json.dumps(payload), {"total_token_count": 10}),
    )
    MODULE.run_row(
        output,
        IDS[0],
        call_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("called twice")),
    )

    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen source artifact changed"):
        MODULE.run_row(
            output,
            IDS[1],
            call_fn=lambda _prompt: (json.dumps(payload), {}),
        )


def test_freeze_binds_every_row_artifact_and_rejects_postfreeze_calls(
    tmp_path: Path,
) -> None:
    assert MODULE is not None
    source, selection, output = _fixture(tmp_path)
    MODULE.prepare(source, selection, output)
    payload = _valid_answer()
    for video_id in IDS:
        MODULE.run_row(
            output,
            video_id,
            call_fn=lambda _prompt: (json.dumps(payload), {"total_token_count": 10}),
        )

    MODULE.freeze(output)

    prediction_freeze = json.loads((output / "predictions-freeze.json").read_text())
    evidence_freeze = json.loads((output / "evidence-freeze.json").read_text())
    assert prediction_freeze["row_count"] == 11
    assert prediction_freeze["passed_count"] == 11
    assert evidence_freeze["prediction_freeze_sha256"] == _sha(
        output / "predictions-freeze.json"
    )
    MODULE.verify_evidence_freeze(output)
    with pytest.raises(ValueError, match="predictions are already frozen"):
        MODULE.run_row(output, IDS[0], call_fn=lambda _prompt: ("", {}))


@pytest.mark.parametrize(
    "artifact",
    ["request.json", "raw-response.txt", "metadata.json", "parsed.json", "terminal.json"],
)
def test_evidence_freeze_rejects_every_row_artifact_drift(
    tmp_path: Path, artifact: str
) -> None:
    assert MODULE is not None
    source, selection, output = _fixture(tmp_path)
    MODULE.prepare(source, selection, output)
    payload = _valid_answer()
    for video_id in IDS:
        MODULE.run_row(
            output,
            video_id,
            call_fn=lambda _prompt: (json.dumps(payload), {"total_token_count": 10}),
        )
    MODULE.freeze(output)
    path = output / "rows" / IDS[0] / artifact
    path.write_text(path.read_text() + " ")

    with pytest.raises(ValueError, match="frozen row artifact changed"):
        MODULE.verify_evidence_freeze(output)
