import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_multi_interpretation_answer_election.py"
SPEC = importlib.util.spec_from_file_location("multi_run", MODULE_PATH)
assert SPEC and SPEC.loader
multi_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi_run)


def _claim(answer="He smokes.", premise="supported", kind="literal_premise"):
    return {
        "interpretation_type": kind,
        "question_interpretation": "What happens after the greeting.",
        "premise_status": premise,
        "answer_claims": [{"claim": answer, "evidence_ids": ["E01"]}],
        "conflicts_or_missing_qualifiers": [],
        "needs_verification": False,
        "answer": answer,
    }


def _verify(kind="conservative_verification", premise="answer_blocking"):
    return {
        "interpretation_type": kind,
        "question_interpretation": "The requested fact may be absent.",
        "premise_status": premise,
        "answer_claims": [],
        "conflicts_or_missing_qualifiers": ["The map does not state it."],
        "needs_verification": True,
        "answer": "NEEDS VERIFICATION",
    }


def _payload(candidates):
    return {"candidates": candidates}


def test_election_uses_two_candidate_agreement_and_shortest_answer():
    first = _claim("He smokes.")
    second = _claim("he smokes", kind="unique_event_paraphrase")
    result = multi_run.elect_candidates(_payload([first, second, _verify()]), {"E01"})
    assert result["needs_verification"] is False
    assert result["answer"] == "he smokes"
    assert result["election_reason"] == "supported_candidate_agreement"
    assert result["elected_candidate_indices"] == [0, 1]


def test_election_accepts_one_answer_when_other_valid_candidates_verify():
    result = multi_run.elect_candidates(
        _payload(
            [
                _claim(),
                _verify("unique_event_paraphrase"),
                _verify(),
            ]
        ),
        {"E01"},
    )
    assert result["answer"] == "He smokes."
    assert result["election_reason"] == "sole_supported_answer"


def test_election_routes_conflicting_answers_to_verification():
    result = multi_run.elect_candidates(
        _payload(
            [
                _claim("He smokes."),
                _claim("He walks away.", kind="unique_event_paraphrase"),
                _verify(),
            ]
        ),
        {"E01"},
    )
    assert result["answer"] == "NEEDS VERIFICATION"
    assert result["election_reason"] == "supported_answer_conflict"


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (
            lambda row: row["answer_claims"][0]["evidence_ids"].append("E99"),
            "unknown evidence ID",
        ),
        (lambda row: row.update(answer="Different answer."), "exactly match"),
        (
            lambda row: row.update(premise_status="answer_blocking"),
            "requires verification",
        ),
    ],
)
def test_election_rejects_invalid_candidate_but_preserves_trace(mutate, reason):
    invalid = _claim()
    mutate(invalid)
    result = multi_run.elect_candidates(
        _payload([invalid, _verify("unique_event_paraphrase"), _verify()]), {"E01"}
    )
    assert result["answer"] == "NEEDS VERIFICATION"
    assert result["candidate_trace"][0]["valid"] is False
    assert reason in result["candidate_trace"][0]["rejection_reason"]


def test_election_requires_exact_unique_interpretation_types():
    payload = _payload([_claim(), _claim(kind="literal_premise"), _verify()])
    with pytest.raises(ValueError, match="interpretation types"):
        multi_run.elect_candidates(payload, {"E01"})


def test_locator_variance_requires_conflict_and_verification_has_reason():
    candidate = _claim(premise="harmless_locator_variance")
    with pytest.raises(ValueError, match="recorded conflict"):
        multi_run.validate_candidate(candidate, {"E01"})
    verification = _verify()
    verification["conflicts_or_missing_qualifiers"] = []
    with pytest.raises(ValueError, match="reason"):
        multi_run.validate_candidate(verification, {"E01"})


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    rows = [
        {
            "video_id": "v00",
            "question": "What happens next?",
            "evidence": [
                {
                    "evidence_id": "E01",
                    "map_line_number": 1,
                    "text": "SCENE | He smokes.",
                }
            ],
        }
    ]
    _write(root / "protocol.source.json", {"experiment_id": "fixture", "rows": rows})
    _write(root / "selection-freeze.json", {"status": "frozen", "row_count": 1})
    _write(root / "preinference-gate-freeze.json", {"status": "frozen", "gate": {}})
    _write(
        root / "selection-integrity.json",
        {
            "protocol_source_sha256": multi_run._sha256(root / "protocol.source.json"),
            "selection_freeze_sha256": multi_run._sha256(
                root / "selection-freeze.json"
            ),
            "preinference_gate_freeze_sha256": multi_run._sha256(
                root / "preinference-gate-freeze.json"
            ),
        },
    )
    multi_run.prepare_arms(root)
    return root


def test_fake_provider_end_to_end_is_one_attempt_resumable_and_frozen(tmp_path: Path):
    root = _prepared_root(tmp_path)
    calls = []

    def single_provider(prompt):
        calls.append(prompt)
        payload = copy.deepcopy(_claim())
        payload.pop("interpretation_type")
        return json.dumps(payload), {"tokens": 1}

    def multi_provider(prompt):
        calls.append(prompt)
        return json.dumps(
            _payload(
                [
                    _claim(),
                    _claim("he smokes", kind="unique_event_paraphrase"),
                    _verify(),
                ]
            )
        ), {"tokens": 2}

    multi_run.run_row(root, "single", "v00", call_fn=single_provider)
    multi_run.run_row(root, "single", "v00", call_fn=single_provider)
    multi_run.run_row(root, "multi", "v00", call_fn=multi_provider)
    assert len(calls) == 2
    multi_run.freeze_arm(root, "single", expected_rows=1)
    multi_run.freeze_arm(root, "multi", expected_rows=1)
    multi_run.freeze_combined(root)
    multi_run.verify_combined(root)
    terminal = json.loads(
        (root / "multi" / "rows" / "v00" / "terminal.json").read_text()
    )
    assert terminal["answer"] == "he smokes"
    raw = root / "multi" / "rows" / "v00" / "raw-response.txt"
    raw.write_text(raw.read_text() + "tamper")
    with pytest.raises(ValueError, match="frozen row artifact changed"):
        multi_run.verify_combined(root)


def test_malformed_json_preserves_paid_raw_and_terminal(tmp_path: Path):
    root = _prepared_root(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        multi_run.run_row(
            root, "multi", "v00", call_fn=lambda _: ("{bad", {"tokens": 3})
        )
    row = root / "multi" / "rows" / "v00"
    assert (row / "raw-response.txt").read_text() == "{bad"
    assert json.loads((row / "metadata.json").read_text()) == {"tokens": 3}
    assert json.loads((row / "terminal.json").read_text())["status"] == "contract_error"
