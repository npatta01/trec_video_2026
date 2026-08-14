#!/usr/bin/env python3
"""Score and gate the frozen issue #35 multi-interpretation diagnostic."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_grounded_map_answering import _load_json, _sha256, _write_once  # noqa: E402
from run_multi_interpretation_answer_election import (  # noqa: E402
    ARMS,
    EXPECTED_ROWS,
    verify_combined,
)
from select_multi_interpretation_answer_election import (  # noqa: E402
    PROMOTION_GATE,
    ROLES,
)
from score_vqa_semantic_panel import (  # noqa: E402
    _default_bertscore,
    _default_meteor,
)


ASSESSMENTS = {
    "correct_answer",
    "correct_verification",
    "incorrect_answer",
    "unnecessary_verification",
    "contract_failure",
    "provider_failure",
}
MeteorFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]


def _index(rows: Any, *, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{source} must be a row list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source} row must be an object")
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in result:
            raise ValueError(f"{source} video IDs must be unique")
        result[video_id] = dict(row)
    return result


def load_frozen_rows(root: Path, annotations: Path) -> list[dict[str, Any]]:
    """Verify both arms before exposing selected labels to the scorer."""

    try:
        verify_combined(root)
    except (OSError, ValueError) as error:
        raise ValueError("experiment integrity must verify before labels are loaded") from error
    source = _load_json(root / "protocol.source.json")
    selection = _load_json(root / "selection-freeze.json")
    selected = _index(selection.get("rows"), source="selection freeze")
    protocol = _index(source.get("rows"), source="source protocol")
    if set(selected) != set(protocol) or len(selected) != EXPECTED_ROWS:
        raise ValueError("selection and source protocol must contain the exact 12 rows")
    arm_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        freeze = _load_json(root / arm / "predictions-freeze.json")
        frozen = _index(freeze.get("rows"), source=f"{arm} prediction freeze")
        if set(frozen) != set(protocol):
            raise ValueError(f"{arm} prediction IDs differ from source protocol")
        joined: dict[str, dict[str, Any]] = {}
        for video_id, frozen_row in frozen.items():
            terminal_path = root / arm / "rows" / video_id / "terminal.json"
            if _sha256(terminal_path) != frozen_row.get("terminal_sha256"):
                raise ValueError(f"{arm} terminal hash mismatch: {video_id}")
            terminal = _load_json(terminal_path)
            if terminal.get("video_id") != video_id or terminal.get("status") != frozen_row.get("status"):
                raise ValueError(f"{arm} terminal identity mismatch: {video_id}")
            item = dict(terminal)
            parsed_path = root / arm / "rows" / video_id / "parsed.json"
            if parsed_path.is_file():
                item["parsed"] = _load_json(parsed_path)
            joined[video_id] = item
        arm_rows[arm] = joined

    rows = []
    for protocol_row in source["rows"]:
        video_id = protocol_row["video_id"]
        annotation_path = annotations / f"{video_id}.json"
        if _sha256(annotation_path) != selected[video_id].get("annotation_sha256"):
            raise ValueError(f"selected annotation hash mismatch: {video_id}")
        annotation = _load_json(annotation_path)
        reference = annotation.get("correct_answer")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"reference is invalid: {video_id}")
        rows.append(
            {
                "video_id": video_id,
                "role": selected[video_id]["role"],
                "question": protocol_row["question"],
                "evidence": protocol_row["evidence"],
                "reference": reference,
                "direct_prediction": selected[video_id]["direct_prediction"],
                "direct_semantic_composite": selected[video_id]["direct_semantic_composite"],
                "single": arm_rows["single"][video_id],
                "multi": arm_rows["multi"][video_id],
            }
        )
    return rows


def _route(row: Mapping[str, Any]) -> str:
    status = row.get("status")
    if status == "contract_error":
        return "contract_failure"
    if status == "provider_error":
        return "provider_failure"
    if status != "passed":
        raise ValueError("unknown frozen route")
    return "verification" if row.get("needs_verification") is True else "answer"


def classify_gate(
    frozen_rows: Sequence[Mapping[str, Any]],
    manual_audit: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    if dict(gate) != PROMOTION_GATE:
        raise ValueError("promotion gate differs from the predeclared rule")
    frozen = _index(list(frozen_rows), source="frozen rows")
    audit = _index(manual_audit.get("rows"), source="manual audit")
    if set(frozen) != set(audit) or len(frozen) != EXPECTED_ROWS:
        raise ValueError("manual audit must exactly cover the frozen 12 rows")
    role_counts = {role: 0 for role in ROLES}
    unsupported: list[str] = []
    premise_correct = 0
    premise_improvements = 0
    safeguard_both = 0
    controls = 0
    oracle = 0
    for video_id, verdict in audit.items():
        row = frozen[video_id]
        role = row.get("role")
        if role not in role_counts or verdict.get("role") != role:
            raise ValueError(f"manual audit role mismatch: {video_id}")
        role_counts[role] += 1
        note = verdict.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"manual audit note is missing: {video_id}")
        candidate_correctness = verdict.get("candidate_correctness")
        if (
            not isinstance(candidate_correctness, list)
            or len(candidate_correctness) != 3
            or any(not isinstance(value, bool) for value in candidate_correctness)
        ):
            raise ValueError(f"manual audit requires three candidate judgments: {video_id}")
        oracle += any(candidate_correctness)
        assessments = {}
        for arm in ARMS:
            assessment = verdict.get(f"{arm}_assessment")
            if assessment not in ASSESSMENTS:
                raise ValueError(f"invalid {arm} assessment: {video_id}")
            route = _route(row[arm])
            allowed_by_route = {
                "answer": {"correct_answer", "incorrect_answer"},
                "verification": {"correct_verification", "unnecessary_verification"},
                "contract_failure": {"contract_failure"},
                "provider_failure": {"provider_failure"},
            }
            if assessment not in allowed_by_route[route]:
                raise ValueError(f"{arm} assessment contradicts frozen route: {video_id}")
            support = verdict.get(f"citation_support_{arm}")
            if not isinstance(support, bool):
                raise ValueError(f"{arm} citation support must be boolean: {video_id}")
            if route == "answer" and not support:
                unsupported.append(f"{arm}:{video_id}")
            assessments[arm] = assessment
        if role == "premise_variance":
            premise_correct += assessments["multi"] == "correct_answer"
            premise_improvements += assessments["multi"] == "correct_answer" and assessments["single"] != "correct_answer"
        elif role == "insufficiency_safeguard":
            safeguard_both += assessments["single"] == assessments["multi"] == "correct_verification"
        else:
            controls += assessments["multi"] == "correct_answer"
    if role_counts != {role: 4 for role in ROLES}:
        raise ValueError("frozen role counts are invalid")
    passed = (
        premise_correct >= gate["minimum_premise_correct_multi"]
        and premise_improvements >= gate["minimum_premise_improvements_over_single"]
        and safeguard_both >= gate["required_safeguard_verifications_each_arm"]
        and controls >= gate["required_multi_controls_preserved"]
        and len(unsupported) <= gate["maximum_unsupported_emitted_answers"]
    )
    return {
        "status": "complete",
        "passed": passed,
        "premise_correct_multi": premise_correct,
        "premise_improvements_over_single": premise_improvements,
        "safeguard_verifications_both_arms": safeguard_both,
        "multi_controls_preserved": controls,
        "unsupported_emitted_answers": sorted(unsupported),
        "candidate_sets_audited": len(audit),
        "oracle_any_candidate_correct": oracle,
        "oracle_disposition": "diagnostic_only_not_used_in_gate",
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    groups = {"all": list(rows)}
    groups.update({role: [row for row in rows if row["role"] == role] for role in ROLES})
    for name, group in groups.items():
        result[name] = {
            "row_count": len(group),
            "meteor": statistics.fmean(float(row["meteor"]) for row in group),
            "bertscore_f1": statistics.fmean(float(row["bertscore_f1"]) for row in group),
            "sts": None,
            "semantic_composite": None,
        }
    return result


def build_audit_template(root: Path, annotations: Path, output: Path) -> None:
    rows = load_frozen_rows(root, annotations)
    _write_once(
        output,
        {
            "schema_version": 1,
            "experiment_id": "multi-interpretation-answer-election-issue35-v1",
            "instructions": "Inspect every candidate and both emitted arms against frozen evidence and post-freeze reference.",
            "rows": [
                {
                    "video_id": row["video_id"],
                    "role": row["role"],
                    "single_assessment": None,
                    "multi_assessment": None,
                    "candidate_correctness": [None, None, None],
                    "citation_support_single": None,
                    "citation_support_multi": None,
                    "note": "",
                }
                for row in rows
            ],
        },
    )


def freeze_audit(root: Path, audit_path: Path) -> None:
    _write_once(
        root / "manual-audit-freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "frozen_at": datetime.now(UTC).isoformat(),
            "manual_audit_sha256": _sha256(audit_path),
            "experiment_integrity_sha256": _sha256(root / "experiment-integrity.json") if (root / "experiment-integrity.json").is_file() else None,
        },
    )


def verify_audit_freeze(root: Path, audit_path: Path) -> None:
    freeze = _load_json(root / "manual-audit-freeze.json")
    if freeze.get("manual_audit_sha256") != _sha256(audit_path):
        raise ValueError("manual audit freeze mismatch")
    integrity = root / "experiment-integrity.json"
    if integrity.is_file() and freeze.get("experiment_integrity_sha256") != _sha256(integrity):
        raise ValueError("manual audit freeze experiment mismatch")


def _prediction(row: Mapping[str, Any], condition: str) -> str:
    if condition == "direct":
        return str(row["direct_prediction"])
    arm = row[condition]
    return str(arm["answer"]) if arm.get("status") == "passed" else "NO ANSWER"


def score(
    root: Path,
    annotations: Path,
    audit_path: Path,
    output: Path,
    bert_cache: Path,
    *,
    meteor_fn: MeteorFunction = _default_meteor,
    bertscore_fn: BertScoreFunction = _default_bertscore,
) -> dict[str, Any]:
    rows = load_frozen_rows(root, annotations)
    verify_audit_freeze(root, audit_path)
    audit = _load_json(audit_path)
    gate_freeze = _load_json(root / "preinference-gate-freeze.json")
    gate_result = classify_gate(rows, audit, gate_freeze["gate"])
    conditions: dict[str, list[dict[str, Any]]] = {}
    for condition in ("direct", "single", "multi"):
        predictions = [_prediction(row, condition) for row in rows]
        references = [row["reference"] for row in rows]
        bert_values = list(bertscore_fn(predictions, references, bert_cache))
        if len(bert_values) != EXPECTED_ROWS:
            raise ValueError("BERTScore returned the wrong row count")
        condition_rows = []
        for row, prediction, reference, bert in zip(rows, predictions, references, bert_values, strict=True):
            condition_rows.append(
                {
                    "video_id": row["video_id"],
                    "role": row["role"],
                    "prediction": prediction,
                    "reference": reference,
                    "meteor": float(meteor_fn(prediction, reference)),
                    "bertscore_f1": float(bert),
                    "sts": None,
                    "semantic_composite": None,
                }
            )
        conditions[condition] = condition_rows
    payload = {
        "schema_version": 1,
        "experiment_id": "multi-interpretation-answer-election-issue35-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_validity": "Diagnostic",
        "decision_disposition": "promoted_for_larger_validation" if gate_result["passed"] else "not_promoted_under_tested_condition",
        "sts": {"status": "unavailable_for_new_predictions", "reason": "historical UMBC-compatible service is unavailable; no substitute is comparable"},
        "conditions": conditions,
        "aggregates": {condition: aggregate_metrics(values) for condition, values in conditions.items()},
        "gate": gate_result,
        "integrity": {
            "experiment_integrity_sha256": _sha256(root / "experiment-integrity.json"),
            "manual_audit_freeze_sha256": _sha256(root / "manual-audit-freeze.json"),
            "manual_audit_sha256": _sha256(audit_path),
            "scorer_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_once(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit-template")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--annotations", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    freeze_parser = subparsers.add_parser("freeze-audit")
    freeze_parser.add_argument("--root", type=Path, required=True)
    freeze_parser.add_argument("--audit", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--root", type=Path, required=True)
    score_parser.add_argument("--annotations", type=Path, required=True)
    score_parser.add_argument("--audit", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--bert-cache", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-template":
        build_audit_template(args.root, args.annotations, args.output)
    elif args.command == "freeze-audit":
        freeze_audit(args.root, args.audit)
    else:
        score(args.root, args.annotations, args.audit, args.output, args.bert_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
