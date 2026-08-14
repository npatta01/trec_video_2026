#!/usr/bin/env python3
"""Score the frozen premise-tolerant issue #35 follow-up panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_grounded_map_answering import _sha256, verify_evidence_freeze  # noqa: E402
from run_premise_tolerant_grounded_answering import (  # noqa: E402
    EXPECTED_ROLE_COUNTS,
    EXPECTED_ROWS,
    _validate_panel,
    validate_answer,
    verify_gate_freeze,
)
from score_grounded_map_answering import load_frozen_predictions as load_v2  # noqa: E402
from score_vqa_semantic_panel import (  # noqa: E402
    _default_bertscore,
    _default_meteor,
    normalize_nist_text,
    sha256_path,
)

MeteorFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]
V2LoadFunction = Callable[[Path], Sequence[Mapping[str, Any]]]
METRICS = ("meteor", "bertscore_f1", "sts", "semantic_composite")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


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


def load_frozen_predictions(root: Path) -> list[dict[str, Any]]:
    """Verify both freezes and join provider-safe rows to post-inference roles."""

    verify_evidence_freeze(root)
    protocol_path = root / "protocol.json"
    protocol = _load(protocol_path)
    freeze = _load(root / "predictions-freeze.json")
    gate = _load(root / "preinference-gate-freeze.json")
    verify_gate_freeze(root, protocol)
    gate_integrity = _load(root / "gate-integrity.json")
    if freeze.get("status") != "frozen" or freeze.get("row_count") != EXPECTED_ROWS:
        raise ValueError("expected a frozen 11-row panel")
    protocol_hash = _sha256(protocol_path)
    if (
        freeze.get("protocol_sha256") != protocol_hash
        or gate.get("protocol_sha256") != protocol_hash
    ):
        raise ValueError("freeze does not bind the current protocol")
    gate_hash = _sha256(root / "preinference-gate-freeze.json")
    prediction_gate_hash = freeze.get("preinference_gate_freeze_sha256")
    if prediction_gate_hash is not None and prediction_gate_hash != gate_hash:
        raise ValueError("prediction freeze gate integrity mismatch")
    expected_integrity = {
        "protocol_sha256": protocol_hash,
        "selection_sha256": gate["selection_sha256"],
        "preinference_gate_freeze_sha256": gate_hash,
        "predictions_freeze_sha256": _sha256(root / "predictions-freeze.json"),
        "evidence_freeze_sha256": _sha256(root / "evidence-freeze.json"),
    }
    if any(
        gate_integrity.get(key) != value for key, value in expected_integrity.items()
    ):
        raise ValueError("gate integrity sidecar mismatch")
    panel_ids, panel = _validate_panel(gate.get("panel"))
    protocol_by_id = _index(protocol.get("rows"), source="protocol")
    frozen_by_id = _index(freeze.get("rows"), source="prediction freeze")
    if set(panel_ids) != set(protocol_by_id) or set(protocol_by_id) != set(
        frozen_by_id
    ):
        raise ValueError("panel IDs do not match frozen protocol IDs")
    role_by_id = {video_id: role for role, ids in panel.items() for video_id in ids}
    rows: list[dict[str, Any]] = []
    passed = 0
    for protocol_row in protocol["rows"]:
        video_id = protocol_row["video_id"]
        terminal_path = root / "rows" / video_id / "terminal.json"
        frozen = frozen_by_id[video_id]
        if not terminal_path.is_file() or _sha256(terminal_path) != frozen.get(
            "terminal_sha256"
        ):
            raise ValueError(f"frozen terminal changed: {video_id}")
        terminal = _load(terminal_path)
        status = terminal.get("status")
        if terminal.get("video_id") != video_id or status != frozen.get("status"):
            raise ValueError(f"frozen terminal identity mismatch: {video_id}")
        joined = {
            **protocol_row,
            "protocol_row": dict(protocol_row),
            "role": role_by_id[video_id],
            "status": status,
        }
        if status == "passed":
            if terminal.get("question") != protocol_row["question"]:
                raise ValueError(f"terminal question mismatch: {video_id}")
            fields = {
                key: terminal[key]
                for key in (
                    "question_interpretation",
                    "premise_status",
                    "answer_claims",
                    "conflicts_or_missing_qualifiers",
                    "needs_verification",
                    "answer",
                )
            }
            joined.update(
                validate_answer(
                    fields, {item["evidence_id"] for item in protocol_row["evidence"]}
                )
            )
            passed += 1
        elif status not in {"provider_error", "contract_error"}:
            raise ValueError(f"unknown frozen terminal status: {video_id}")
        rows.append(joined)
    if (
        freeze.get("passed_count") != passed
        or freeze.get("failed_count") != EXPECTED_ROWS - passed
    ):
        raise ValueError("prediction freeze counts are inconsistent")
    return rows


def classify_gate(
    treatment_rows: Sequence[Mapping[str, Any]],
    manual_audit: Mapping[str, Any],
    gate_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    panel_ids, panel = _validate_panel(gate_freeze.get("panel"))
    rule = gate_freeze.get("gate")
    if not isinstance(rule, Mapping):
        raise ValueError("gate rule is missing")
    audit = _index(manual_audit.get("rows"), source="manual audit")
    treatment = _index(treatment_rows, source="frozen treatment")
    if set(audit) != set(panel_ids):
        raise ValueError("manual audit IDs must exactly match panel IDs")
    if set(treatment) != set(panel_ids):
        raise ValueError("frozen treatment IDs must exactly match panel IDs")
    allowed = {
        "fix_false_abstention": {"rescued", "not_rescued"},
        "detectable_insufficiency": {
            "correct_verification",
            "unsafe_answer",
            "contract_failure",
        },
        "known_limit": {"observed_answer", "observed_verification"},
        "stability_control": {"preserved", "regressed"},
    }
    role_by_id = {video_id: role for role, ids in panel.items() for video_id in ids}
    unsupported: list[str] = []
    for video_id, row in audit.items():
        role = role_by_id[video_id]
        if row.get("role") != role or row.get("assessment") not in allowed[role]:
            raise ValueError(f"invalid manual audit assessment: {video_id}")
        if not isinstance(row.get("citation_support"), bool):
            raise ValueError(f"citation_support must be boolean: {video_id}")
        if not isinstance(row.get("note"), str) or not row["note"].strip():
            raise ValueError(f"manual audit requires a note: {video_id}")
        frozen = treatment[video_id]
        if frozen.get("role") != role:
            raise ValueError(f"frozen treatment role mismatch: {video_id}")
        answered = (
            frozen.get("status") == "passed"
            and frozen.get("needs_verification") is False
        )
        verified = (
            frozen.get("status") == "passed"
            and frozen.get("needs_verification") is True
        )
        consistent = {
            "rescued": answered,
            "not_rescued": True,
            "correct_verification": verified,
            "unsafe_answer": answered,
            "contract_failure": frozen.get("status") != "passed",
            "observed_answer": answered,
            "observed_verification": verified,
            "preserved": answered,
            "regressed": True,
        }[row["assessment"]]
        if not consistent:
            raise ValueError(
                f"manual assessment contradicts frozen treatment: {video_id}"
            )
        if answered and (
            not row["citation_support"] or row["assessment"] == "unsafe_answer"
        ):
            unsupported.append(video_id)
    rescues = sum(
        audit[item]["assessment"] == "rescued" for item in panel["fix_false_abstention"]
    )
    verifications = sum(
        audit[item]["assessment"] == "correct_verification"
        for item in panel["detectable_insufficiency"]
    )
    controls = sum(
        audit[item]["assessment"] == "preserved" for item in panel["stability_control"]
    )
    known_id = panel["known_limit"][0]
    passed = (
        rescues >= rule["minimum_false_abstention_rescues"]
        and verifications >= rule["required_detectable_insufficiency_verifications"]
        and controls >= rule["required_stability_controls_preserved"]
        and len(unsupported) <= rule["maximum_unsupported_final_answers"]
    )
    return {
        "status": "complete",
        "passed": passed,
        "false_abstention_rescues": rescues,
        "detectable_insufficiency_verifications": verifications,
        "stability_controls_preserved": controls,
        "unsupported_final_answer_rows": sorted(unsupported),
        "audited_row_count": len(audit),
        "known_limit": {
            "video_id": known_id,
            "assessment": audit[known_id]["assessment"],
        },
        "rule": dict(rule),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        result[metric] = (
            None if all(value is None for value in values) else statistics.fmean(values)
        )
    return result


def _aggregate_with_roles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = _aggregate(rows)
    result["roles"] = {
        role: _aggregate([row for row in rows if row["role"] == role])
        for role in EXPECTED_ROLE_COUNTS
    }
    return result


def score(
    *,
    root: Path,
    direct_predictions_path: Path,
    annotation_root: Path,
    v2_root: Path,
    bert_model: Path,
    manual_audit_path: Path | None = None,
    meteor_fn: MeteorFunction = _default_meteor,
    bertscore_fn: BertScoreFunction = _default_bertscore,
    v2_load_fn: V2LoadFunction = load_v2,
) -> dict[str, Any]:
    v3_rows = load_frozen_predictions(root)
    v2_artifacts = {
        name: v2_root / name
        for name in ("protocol.json", "predictions-freeze.json", "evidence-freeze.json")
    }
    if any(not path.is_file() for path in v2_artifacts.values()):
        raise ValueError("missing frozen v2 artifact")
    v2_rows = v2_load_fn(v2_root)
    direct = _index(_load(direct_predictions_path), source="direct predictions")
    v2 = _index(v2_rows, source="v2 predictions")
    pairs: list[dict[str, Any]] = []
    for row in v3_rows:
        video_id = row["video_id"]
        if (
            video_id not in direct
            or direct[video_id].get("status") != "passed"
            or video_id not in v2
            or v2[video_id].get("status") != "passed"
        ):
            raise ValueError(f"missing passed comparison prediction: {video_id}")
        annotation = _load(annotation_root / f"{video_id}.json")
        if (
            not isinstance(annotation.get("question"), str)
            or annotation["question"].strip() != row["question"].strip()
        ):
            raise ValueError(f"question mismatch: {video_id}")
        pairs.append(
            {
                "video_id": video_id,
                "role": row["role"],
                "question": row["question"],
                "reference": annotation["correct_answer"],
                "direct_baseline": direct[video_id]["answer"],
                "grounded_v2": v2[video_id]["answer"],
                "premise_v3": row.get("answer", ""),
                "v3_status": row["status"],
                "premise_status": row.get("premise_status"),
                "needs_verification": row.get("needs_verification"),
                "answer_claims": row.get("answer_claims", []),
                "conflicts_or_missing_qualifiers": row.get(
                    "conflicts_or_missing_qualifiers", []
                ),
            }
        )
    predictions = [
        pair[condition]
        for condition in ("direct_baseline", "grounded_v2", "premise_v3")
        for pair in pairs
    ]
    references = [
        pair["reference"]
        for _condition in ("direct_baseline", "grounded_v2", "premise_v3")
        for pair in pairs
    ]
    bert = list(bertscore_fn(predictions, references, bert_model))
    if len(bert) != len(predictions):
        raise ValueError("BERTScore result count mismatch")
    scored: list[dict[str, Any]] = []
    offset = 0
    for condition in ("direct_baseline", "grounded_v2", "premise_v3"):
        for pair in pairs:
            failed = condition == "premise_v3" and pair["v3_status"] != "passed"
            scored.append(
                {
                    "video_id": pair["video_id"],
                    "role": pair["role"],
                    "condition": condition,
                    "status": "failed" if failed else "passed",
                    "prediction": pair[condition],
                    "reference": pair["reference"],
                    "meteor": 0.0
                    if failed
                    else float(meteor_fn(pair[condition], pair["reference"])),
                    "bertscore_f1": 0.0 if failed else float(bert[offset]),
                    "sts": None,
                    "semantic_composite": None,
                }
            )
            offset += 1
    conditions = {
        condition: _aggregate_with_roles(
            [row for row in scored if row["condition"] == condition]
        )
        for condition in ("direct_baseline", "grounded_v2", "premise_v3")
    }
    changed = [
        {
            **pair,
            "v2_prediction": pair["grounded_v2"],
            "v3_prediction": pair["premise_v3"],
        }
        for pair in pairs
        if normalize_nist_text(pair["grounded_v2"])
        != normalize_nist_text(pair["premise_v3"])
    ]
    gate_freeze = _load(root / "preinference-gate-freeze.json")
    if manual_audit_path is None:
        gate: dict[str, Any] = {"status": "pending_manual_audit", "passed": None}
    else:
        audit = _load(manual_audit_path)
        if audit.get("prediction_freeze_sha256") != _sha256(
            root / "predictions-freeze.json"
        ):
            raise ValueError("manual audit does not bind prediction freeze")
        gate = classify_gate(v3_rows, audit, gate_freeze)
    return {
        "schema_version": 1,
        "experiment_id": _load(root / "protocol.json")["experiment_id"],
        "panel": {"row_count": EXPECTED_ROWS, "role_counts": EXPECTED_ROLE_COUNTS},
        "conditions": conditions,
        "changed_row_count": len(changed),
        "changed_rows": changed,
        "gate": gate,
        "rows": scored,
        "integrity": {
            "predictions_frozen_before_labels": True,
            "protocol_sha256": _sha256(root / "protocol.json"),
            "prediction_freeze_sha256": _sha256(root / "predictions-freeze.json"),
            "evidence_freeze_sha256": _sha256(root / "evidence-freeze.json"),
            "preinference_gate_freeze_sha256": _sha256(
                root / "preinference-gate-freeze.json"
            ),
            "gate_integrity_sha256": _sha256(root / "gate-integrity.json"),
            "v2_protocol_sha256": _sha256(v2_artifacts["protocol.json"]),
            "v2_predictions_freeze_sha256": _sha256(
                v2_artifacts["predictions-freeze.json"]
            ),
            "v2_evidence_freeze_sha256": _sha256(v2_artifacts["evidence-freeze.json"]),
            "direct_predictions_sha256": _sha256(direct_predictions_path),
            "bert_model_sha256": sha256_path(bert_model)
            if bert_model.exists()
            else "injected-test-model",
        },
        "metric_notes": {
            "sts_status": "unavailable",
            "sts_unavailable_reason": "The historical UMBC endpoint is unavailable; no comparable local STS implementation is frozen.",
            "semantic_composite": "unavailable because STS is unavailable",
            "selection": "diagnostic mechanism panel; not a prevalence estimate",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--direct-predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = score(
        root=args.root,
        direct_predictions_path=args.direct_predictions,
        annotation_root=args.annotations,
        v2_root=args.v2_root,
        bert_model=args.bert_model,
        manual_audit_path=args.manual_audit,
    )
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gate": artifact["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
