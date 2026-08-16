#!/usr/bin/env python3
"""Score a frozen count-query oracle slice after references are revealed."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_submission import canonical_sha256


def _verify_hash(
    payload: Mapping[str, object], *, name: str, expected: str | None = None
) -> str:
    embedded = payload.get("complete_sha256")
    if not isinstance(embedded, str):
        raise TypeError(f"{name} lacks complete_sha256")
    if expected is not None and embedded != expected:
        raise ValueError(f"{name} expected complete_sha256 changed")
    unsigned = {
        key: value for key, value in payload.items() if key != "complete_sha256"
    }
    if canonical_sha256(unsigned) != embedded:
        raise ValueError(f"{name} complete_sha256 changed")
    return embedded


def _rows_by_id(
    payload: Mapping[str, object], *, name: str, verify_rows: bool
) -> dict[str, dict[str, object]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError(f"{name}.rows must be an array")
    rows: dict[str, dict[str, object]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("video_id"), str):
            raise TypeError(f"{name} row must contain video_id")
        video_id = raw["video_id"]
        if video_id in rows:
            raise ValueError(f"{name} video IDs must be unique")
        if verify_rows:
            _verify_hash(raw, name=f"{name} row {video_id}")
        rows[video_id] = raw
    return rows


def _validate_proof(
    proof: Mapping[str, object], *, video_id: str, expected_schema_version: int
) -> None:
    _verify_hash(proof, name=f"proof {video_id}")
    if proof.get("schema_version") != expected_schema_version:
        raise ValueError(f"proof schema version changed for {video_id}")
    if proof.get("query_id") != video_id or proof.get("graph_id") != video_id:
        raise ValueError(f"proof identity binding changed for {video_id}")
    outcome = proof.get("outcome")
    lower = proof.get("lower_bound")
    upper = proof.get("upper_bound")
    exact = proof.get("exact_count")
    if not isinstance(lower, int) or (upper is not None and not isinstance(upper, int)):
        raise TypeError(f"proof bounds invalid for {video_id}")
    if upper is not None and lower > upper:
        raise ValueError(f"proof bounds changed for {video_id}")
    if outcome == "exact":
        if not isinstance(exact, int) or lower != upper or exact != lower:
            raise ValueError(f"exact proof singleton changed for {video_id}")
        if proof.get("abstention_reasons"):
            raise ValueError(f"exact proof reasons changed for {video_id}")
    elif outcome == "abstain":
        if exact is not None:
            raise ValueError(f"abstaining proof exact count changed for {video_id}")
    else:
        raise ValueError(f"proof outcome changed for {video_id}")


def score_oracle_slice(
    *,
    selection: Mapping[str, object],
    proofs: Mapping[str, object],
    expected_selection_sha256: str,
    expected_proofs_sha256: str,
) -> dict[str, object]:
    """Authenticate immutable inputs and calculate only aggregate gate metrics."""

    selection_hash = _verify_hash(
        selection, name="selection", expected=expected_selection_sha256
    )
    proofs_hash = _verify_hash(proofs, name="proofs", expected=expected_proofs_sha256)
    proof_schema_version = proofs.get("schema_version")
    if proof_schema_version not in {1, 2}:
        raise ValueError("proof freeze schema version must be 1 or 2")
    selected = _rows_by_id(selection, name="selection", verify_rows=False)
    frozen = _rows_by_id(proofs, name="proofs", verify_rows=True)
    if set(selected) != set(frozen):
        raise ValueError("selection and proof IDs changed")
    role_counts = Counter(row.get("panel_role") for row in selected.values())
    if role_counts != {"confirmed_failure": 8, "correct_control": 4}:
        raise ValueError("selection role denominators changed")

    exact_answered = 0
    incorrect_non_abstained = 0
    failure_rescued = 0
    controls_preserved = 0
    abstention_reasons: Counter[str] = Counter()
    for video_id, selection_row in selected.items():
        proof_row = frozen[video_id]
        if proof_row.get("request_sha256") != selection_row.get("request_sha256"):
            raise ValueError(f"request binding changed for {video_id}")
        proof = proof_row.get("proof")
        if not isinstance(proof, Mapping):
            raise TypeError(f"proof payload missing for {video_id}")
        _validate_proof(
            proof,
            video_id=video_id,
            expected_schema_version=proof_schema_version,
        )
        exact = proof.get("outcome") == "exact"
        is_correct = exact and proof.get("exact_count") == selection_row.get(
            "reference_count"
        )
        if exact:
            exact_answered += 1
            if not is_correct:
                incorrect_non_abstained += 1
        else:
            reasons = proof.get("abstention_reasons")
            if not isinstance(reasons, list):
                raise TypeError(f"proof abstention reasons invalid for {video_id}")
            abstention_reasons.update(str(reason) for reason in reasons)
        if selection_row.get("panel_role") == "confirmed_failure" and is_correct:
            failure_rescued += 1
        if selection_row.get("panel_role") == "correct_control" and is_correct:
            controls_preserved += 1

    gate_passed = (
        incorrect_non_abstained == 0 and exact_answered >= 9 and controls_preserved == 4
    )
    result: dict[str, object] = {
        "schema_version": proof_schema_version,
        "selection_complete_sha256": selection_hash,
        "proofs_complete_sha256": proofs_hash,
        "row_count": 12,
        "exact_answered": exact_answered,
        "abstained": 12 - exact_answered,
        "incorrect_non_abstained": incorrect_non_abstained,
        "failure_rescue": {
            "denominator": 8,
            "rescued": failure_rescued,
            "rate": failure_rescued / 8,
        },
        "control_preservation": {
            "denominator": 4,
            "preserved": controls_preserved,
            "rate": controls_preserved / 4,
        },
        "abstention_reasons": dict(sorted(abstention_reasons.items())),
        "architecture_gate": {
            "passed": gate_passed,
            "requires_no_incorrect_non_abstained": True,
            "requires_exact_answered_at_least": 9,
            "requires_controls_preserved": 4,
        },
    }
    result["complete_sha256"] = canonical_sha256(result)
    return result


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_once_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting score: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--proofs", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-proofs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    score = score_oracle_slice(
        selection=_load_json(args.selection),
        proofs=_load_json(args.proofs),
        expected_selection_sha256=args.expected_selection_sha256,
        expected_proofs_sha256=args.expected_proofs_sha256,
    )
    _write_once_json(args.output, score)
    print(json.dumps(score, sort_keys=True))


if __name__ == "__main__":
    main()
