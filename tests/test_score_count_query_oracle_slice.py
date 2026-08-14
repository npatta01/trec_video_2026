from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "score_count_query_oracle_slice.py"
SPEC = importlib.util.spec_from_file_location("score_count_query_oracle_slice", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def scoring_fixture(
    *, incorrect_failure: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    selection_rows = []
    proof_rows = []
    for index in range(12):
        video_id = f"video-{index:02}"
        is_control = index >= 8
        exact = is_control or index < 5 or (incorrect_failure and index == 5)
        reference_count = 2
        exact_count = 3 if incorrect_failure and index == 5 else reference_count
        request_sha = f"request-{index:02}"
        selection_rows.append(
            {
                "video_id": video_id,
                "panel_role": "correct_control" if is_control else "confirmed_failure",
                "status": "correct" if is_control else "wrong",
                "question": f"How many in {video_id}?",
                "reference_answer": "two",
                "reference_count": reference_count,
                "baseline_prediction": "2" if is_control else "3",
                "prediction_count": 2 if is_control else 3,
                "prediction_candidates": [2 if is_control else 3],
                "ordinal": index,
                "semantic_stratum": "physical_entity",
                "request_sha256": request_sha,
            }
        )
        proof = _with_hash(
            {
                "schema_version": 1,
                "query_id": video_id,
                "graph_id": video_id,
                "outcome": "exact" if exact else "abstain",
                "lower_bound": exact_count if exact else 1,
                "upper_bound": exact_count if exact else None,
                "exact_count": exact_count if exact else None,
                "traces": [],
                "abstention_reasons": [] if exact else ["incomplete_coverage"],
            }
        )
        proof_row = {
            "video_id": video_id,
            "request_sha256": request_sha,
            "source_sha256": f"source-{index:02}",
            "query_complete_sha256": f"query-{index:02}",
            "graph_complete_sha256": f"graph-{index:02}",
            "proof": proof,
        }
        proof_row["complete_sha256"] = canonical_sha256(proof_row)
        proof_rows.append(proof_row)
    selection = _with_hash(
        {
            "schema_version": 1,
            "selection_policy": "fixture",
            "panel_complete_sha256": "panel",
            "requests_complete_sha256": "requests",
            "design_video_ids": [],
            "rows": selection_rows,
        }
    )
    proofs = _with_hash(
        {
            "schema_version": 1,
            "authoring_requests_complete_sha256": "requests",
            "authored_inputs_complete_sha256": "authored",
            "rows": proof_rows,
        }
    )
    return selection, proofs


def test_scorer_applies_predeclared_architecture_gate() -> None:
    selection, proofs = scoring_fixture()

    score = MODULE.score_oracle_slice(
        selection=selection,
        proofs=proofs,
        expected_selection_sha256=selection["complete_sha256"],
        expected_proofs_sha256=proofs["complete_sha256"],
    )

    assert score["exact_answered"] == 9
    assert score["abstained"] == 3
    assert score["incorrect_non_abstained"] == 0
    assert score["failure_rescue"] == {
        "denominator": 8,
        "rescued": 5,
        "rate": 0.625,
    }
    assert score["control_preservation"] == {
        "denominator": 4,
        "preserved": 4,
        "rate": 1.0,
    }
    assert score["architecture_gate"]["passed"] is True
    assert "video_id" not in str(score)
    assert "reference_answer" not in str(score)


def test_scorer_rejects_one_wrong_non_abstained_answer() -> None:
    selection, proofs = scoring_fixture(incorrect_failure=True)

    score = MODULE.score_oracle_slice(
        selection=selection,
        proofs=proofs,
        expected_selection_sha256=selection["complete_sha256"],
        expected_proofs_sha256=proofs["complete_sha256"],
    )

    assert score["exact_answered"] == 10
    assert score["incorrect_non_abstained"] == 1
    assert score["architecture_gate"]["passed"] is False


@pytest.mark.parametrize("target", ["selection", "proofs", "proof_row", "proof"])
def test_scorer_rejects_chain_mutation(target: str) -> None:
    selection, proofs = scoring_fixture()
    expected_selection = selection["complete_sha256"]
    expected_proofs = proofs["complete_sha256"]
    if target == "selection":
        selection["rows"][0]["reference_count"] = 9
    elif target == "proofs":
        proofs["authored_inputs_complete_sha256"] = "changed"
    elif target == "proof_row":
        proofs["rows"][0]["request_sha256"] = "changed"
    else:
        proofs["rows"][0]["proof"]["exact_count"] = 9

    with pytest.raises(ValueError, match="changed"):
        MODULE.score_oracle_slice(
            selection=selection,
            proofs=proofs,
            expected_selection_sha256=expected_selection,
            expected_proofs_sha256=expected_proofs,
        )
