#!/usr/bin/env python3
"""Score the frozen issue #35 grounded-map treatment after prediction freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_grounded_map_answering import (  # noqa: E402
    PROMOTION_RULE,
    _sha256,
    freeze_evidence as _freeze_evidence,
    validate_answer,
    verify_evidence_freeze,
)

from score_vqa_semantic_panel import (  # noqa: E402
    _default_bertscore,
    _default_meteor,
    normalize_nist_text,
    sha256_path,
    STS_ENDPOINT,
    umbc_sts_score,
)

# Re-exported for scorer fixtures and audit tooling that construct a frozen run.
freeze_evidence = _freeze_evidence


EXPECTED_ROWS = 24
EXPECTED_TARGETS = 12
EXPECTED_CONTROLS = 12
METRICS = ("meteor", "bertscore_f1", "sts", "semantic_composite")
FAILURE_MODES = {
    "question_misunderstanding",
    "wrong_span_use",
    "reasoning_failure",
    "answer_specificity_or_normalization",
    "insufficient_frozen_evidence",
    "reference_or_question_issue",
}
MeteorFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]
StsFunction = Callable[[str, str], float]
BASELINE_SCORE_CONDITION = "gemini-3.1-pro-native-av-4fps"
STS_CACHE_IDENTITY = hashlib.sha256(
    (
        json.dumps(
            {
                "endpoint": STS_ENDPOINT,
                "pair_order": "phrase1=reference, phrase2=prediction",
                "request_user_agent": "NIST - TRECVID",
                "text_policy": "exact scorer inputs without pre-normalization",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error


def _finite_metric(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _sts_cache_key(prediction: str, reference: str) -> str:
    encoded = (
        json.dumps(
            {
                "identity": STS_CACHE_IDENTITY,
                "prediction": prediction,
                "reference": reference,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class PersistentStsCache:
    """Incrementally persist endpoint-identical STS values for safe resumption."""

    def __init__(self, path: Path, *, live_fn: StsFunction = umbc_sts_score) -> None:
        self.path = path
        self.live_fn = live_fn
        self.values: dict[str, float] = {}
        self.seed_sources: list[dict[str, str]] = []
        self.stats = {"hits": 0, "misses": 0, "seeded": 0}
        if path.exists():
            payload = _load_json(path)
            if payload.get("identity") != STS_CACHE_IDENTITY:
                raise ValueError("STS cache identity mismatch")
            raw_values = payload.get("values")
            if not isinstance(raw_values, dict):
                raise ValueError("STS cache requires a values object")
            self.values = {
                str(key): _finite_metric(value, field="cached STS")
                for key, value in raw_values.items()
            }
            raw_sources = payload.get("seed_sources", [])
            if not isinstance(raw_sources, list) or any(
                not isinstance(source, dict) for source in raw_sources
            ):
                raise ValueError("STS cache seed_sources must be an object list")
            self.seed_sources = [dict(source) for source in raw_sources]

    def _write(self) -> None:
        payload = {
            "schema_version": 1,
            "identity": STS_CACHE_IDENTITY,
            "endpoint": STS_ENDPOINT,
            "seed_sources": self.seed_sources,
            "values": dict(sorted(self.values.items())),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def seed(
        self,
        prediction: str,
        reference: str,
        value: float,
    ) -> None:
        key = _sts_cache_key(prediction, reference)
        numeric = _finite_metric(value, field="seeded STS")
        existing = self.values.get(key)
        if existing is not None and existing != numeric:
            raise ValueError("STS cache seed conflicts with an existing value")
        if existing is None:
            self.values[key] = numeric
            self.stats["seeded"] += 1

    def flush(self) -> None:
        self._write()

    def __call__(self, prediction: str, reference: str) -> float:
        key = _sts_cache_key(prediction, reference)
        if key in self.values:
            self.stats["hits"] += 1
            return self.values[key]
        value = _finite_metric(
            self.live_fn(prediction, reference), field="live STS"
        )
        self.values[key] = value
        self.stats["misses"] += 1
        self._write()
        return value


def seed_baseline_sts_cache(
    *,
    cache: PersistentStsCache,
    prior_score_path: Path,
    baseline_predictions_path: Path,
    annotation_root: Path,
    video_ids: Sequence[str],
) -> None:
    """Reuse only baseline STS rows bound to the exact predictions and labels."""

    prior = _load_json(prior_score_path)
    prior_inputs = prior.get("inputs", {})
    prediction_binding = (
        prior_inputs.get("prediction_files", {})
        .get(BASELINE_SCORE_CONDITION, {})
        .get("sha256")
    )
    if prediction_binding != _sha256(baseline_predictions_path):
        raise ValueError("baseline score prediction binding mismatch")
    if prior_inputs.get("annotation_root_sha256") != sha256_path(annotation_root):
        raise ValueError("baseline score annotation binding mismatch")
    baseline_by_id = _index_unique(
        _load_json(baseline_predictions_path), source="direct baseline"
    )
    prior_rows = prior.get("rows", {}).get(BASELINE_SCORE_CONDITION)
    if not isinstance(prior_rows, list):
        raise ValueError("baseline score rows are missing")
    score_by_id = _index_unique(prior_rows, source="baseline score")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("baseline STS seed IDs must be unique")
    for video_id in video_ids:
        baseline = baseline_by_id.get(video_id)
        scored = score_by_id.get(video_id)
        if baseline is None or scored is None:
            raise ValueError(f"baseline STS seed row is missing: {video_id}")
        annotation = _load_json(annotation_root / f"{video_id}.json")
        cache.seed(
            str(baseline["answer"]),
            str(annotation["correct_answer"]),
            _finite_metric(scored.get("sts"), field=f"baseline STS {video_id}"),
        )
    source = {
        "condition": BASELINE_SCORE_CONDITION,
        "score_sha256": _sha256(prior_score_path),
    }
    if source not in cache.seed_sources:
        cache.seed_sources.append(source)
    cache.flush()


def _index_unique(rows: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{source} has a row without video_id")
        if video_id in indexed:
            raise ValueError(f"{source} has duplicate video_id: {video_id}")
        indexed[video_id] = dict(row)
    return indexed


def load_frozen_predictions(root: Path) -> list[dict[str, Any]]:
    """Load terminal treatment rows after verifying both pre-inference freezes."""

    protocol_path = root / "protocol.json"
    freeze_path = root / "predictions-freeze.json"
    gate_path = root / "preinference-gate-freeze.json"
    protocol = _load_json(protocol_path)
    freeze = _load_json(freeze_path)
    gate = _load_json(gate_path)
    verify_evidence_freeze(root)
    if freeze.get("status") != "frozen" or freeze.get("row_count") != EXPECTED_ROWS:
        raise ValueError("expected a frozen 24-row prediction panel")
    protocol_hash = _sha256(protocol_path)
    if freeze.get("protocol_sha256") != protocol_hash:
        raise ValueError("prediction freeze does not bind the current protocol")
    if gate.get("protocol_sha256") != protocol_hash:
        raise ValueError("promotion gate does not bind the current protocol")
    if gate.get("rule") != PROMOTION_RULE:
        raise ValueError("promotion rule differs from the predeclared rule")
    protocol_by_id = _index_unique(protocol.get("rows", []), source="protocol")
    freeze_by_id = _index_unique(freeze.get("rows", []), source="prediction freeze")
    if len(protocol_by_id) != EXPECTED_ROWS or set(protocol_by_id) != set(freeze_by_id):
        raise ValueError("prediction freeze row IDs do not match the protocol")
    rows: list[dict[str, Any]] = []
    passed_count = 0
    for protocol_row in protocol["rows"]:
        video_id = protocol_row["video_id"]
        terminal_path = root / "rows" / video_id / "terminal.json"
        frozen = freeze_by_id[video_id]
        if not terminal_path.is_file() or _sha256(terminal_path) != frozen["terminal_sha256"]:
            raise ValueError(f"frozen terminal changed: {video_id}")
        terminal = _load_json(terminal_path)
        if terminal.get("video_id") != video_id or terminal.get("status") != frozen["status"]:
            raise ValueError(f"frozen terminal identity mismatch: {video_id}")
        status = terminal["status"]
        joined = {**protocol_row, "status": status}
        if status == "passed":
            for field in ("pair_id", "role", "stratum", "question"):
                if terminal.get(field) != protocol_row[field]:
                    raise ValueError(f"terminal {field} mismatch: {video_id}")
            answer_fields = {
                key: terminal[key]
                for key in (
                    "question_interpretation",
                    "answer_claims",
                    "conflicts_or_missing_qualifiers",
                    "needs_verification",
                    "answer",
                )
            }
            joined.update(
                validate_answer(
                    answer_fields,
                    {item["evidence_id"] for item in protocol_row["evidence"]},
                    require_answer_claim_match=(
                        int(protocol.get("answer_contract_version", 1)) >= 2
                    ),
                )
            )
            passed_count += 1
        elif status not in {"provider_error", "contract_error"}:
            raise ValueError(f"unknown frozen terminal status: {video_id}")
        rows.append(joined)
    if freeze.get("passed_count") != passed_count:
        raise ValueError("prediction freeze passed_count is inconsistent")
    if freeze.get("failed_count") != EXPECTED_ROWS - passed_count:
        raise ValueError("prediction freeze failed_count is inconsistent")
    return rows


def classify_gate(
    changed_rows: Sequence[Mapping[str, Any]],
    manual_audit: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate changed-row inspection and apply the exact predeclared gate."""

    changed_by_id = _index_unique(changed_rows, source="changed rows")
    audit_rows = manual_audit.get("rows")
    if not isinstance(audit_rows, list):
        raise ValueError("manual audit requires a rows list")
    audit_by_id = _index_unique(audit_rows, source="manual audit")
    if set(audit_by_id) != set(changed_by_id):
        raise ValueError("manual audit IDs must exactly match changed row IDs")
    target_rescues = 0
    control_regressions = 0
    unsupported_citations: list[str] = []
    unnecessary_verification: list[str] = []
    remaining_failures: dict[str, int] = {}
    for video_id, audit in audit_by_id.items():
        changed = changed_by_id[video_id]
        role = changed["role"]
        if audit.get("role") != role:
            raise ValueError(f"manual audit role mismatch: {video_id}")
        allowed = {"rescued", "not_rescued"} if role == "target" else {"preserved", "regressed"}
        assessment = audit.get("assessment")
        if assessment not in allowed:
            raise ValueError(f"invalid {role} assessment: {video_id}")
        if assessment == "rescued":
            target_rescues += 1
        elif assessment == "regressed":
            control_regressions += 1
        citation_support = audit.get("citation_support")
        if not isinstance(citation_support, bool):
            raise ValueError(f"citation_support must be boolean: {video_id}")
        if not citation_support:
            unsupported_citations.append(video_id)
        verification = audit.get("verification_assessment")
        if verification not in {"not_applicable", "appropriate", "unnecessary"}:
            raise ValueError(f"invalid verification assessment: {video_id}")
        if verification == "unnecessary":
            unnecessary_verification.append(video_id)
        failure_mode = audit.get("remaining_failure_mode")
        failed = assessment in {"not_rescued", "regressed"}
        if failed and failure_mode not in FAILURE_MODES:
            raise ValueError(f"remaining failure requires a failure mode: {video_id}")
        if not failed and failure_mode is not None:
            raise ValueError(f"successful row cannot have a remaining failure mode: {video_id}")
        if failed:
            remaining_failures[failure_mode] = remaining_failures.get(failure_mode, 0) + 1
        note = audit.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"manual audit requires a note: {video_id}")
    passed = (
        target_rescues >= rule["minimum_target_rescues"]
        and control_regressions <= rule["maximum_control_regressions"]
        and not unsupported_citations
        and not unnecessary_verification
    )
    return {
        "status": "complete",
        "passed": passed,
        "rule": dict(rule),
        "changed_rows_inspected": len(changed_by_id),
        "target_rescues": target_rescues,
        "control_regressions": control_regressions,
        "unsupported_citation_rows": sorted(unsupported_citations),
        "unnecessary_verification_rows": sorted(unnecessary_verification),
        "remaining_failure_counts": dict(sorted(remaining_failures.items())),
    }


def _score_values(
    prediction: str,
    reference: str,
    *,
    meteor_fn: MeteorFunction,
    sts_fn: StsFunction,
    bertscore: float,
) -> dict[str, float]:
    meteor = float(meteor_fn(prediction, reference))
    sts = float(sts_fn(prediction, reference))
    return {
        "meteor": meteor,
        "bertscore_f1": float(bertscore),
        "sts": sts,
        "semantic_composite": statistics.fmean((meteor, float(bertscore), sts)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def mean_metric(selected: Sequence[Mapping[str, Any]], metric: str) -> float | None:
        values = [row[metric] for row in selected]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"metric availability is inconsistent: {metric}")
        return statistics.fmean(values)

    return {
        **{
            metric: mean_metric(rows, metric)
            for metric in METRICS
        },
        "row_count": len(rows),
        "completed_count": sum(row["status"] == "passed" for row in rows),
        "roles": {
            role: {
                **{
                    metric: mean_metric(
                        [row for row in rows if row["role"] == role], metric
                    )
                    for metric in METRICS
                },
                "row_count": sum(row["role"] == role for row in rows),
            }
            for role in ("target", "control")
        },
    }


def score(
    *,
    root: Path,
    baseline_predictions_path: Path,
    annotation_root: Path,
    bert_model: Path,
    manual_audit_path: Path | None = None,
    meteor_fn: MeteorFunction = _default_meteor,
    bertscore_fn: BertScoreFunction = _default_bertscore,
    sts_fn: StsFunction = umbc_sts_score,
    sts_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Score direct baseline and grounded treatment on the frozen 24 rows."""

    treatment_rows = load_frozen_predictions(root)
    protocol = _load_json(root / "protocol.json")
    expected_baseline_hash = protocol["source_artifacts"]["direct_predictions"][
        "sha256"
    ]
    if _sha256(baseline_predictions_path) != expected_baseline_hash:
        raise ValueError("frozen direct baseline predictions changed")
    baseline_by_id = _index_unique(
        _load_json(baseline_predictions_path), source="direct baseline"
    )
    pairs: list[dict[str, Any]] = []
    for treatment in treatment_rows:
        video_id = treatment["video_id"]
        baseline = baseline_by_id.get(video_id)
        if baseline is None or baseline.get("status") != "passed":
            raise ValueError(f"missing passed direct baseline: {video_id}")
        annotation = _load_json(annotation_root / f"{video_id}.json")
        annotation_question = annotation.get("question")
        if (
            not isinstance(annotation_question, str)
            or annotation_question.strip() != treatment["question"].strip()
        ):
            raise ValueError(f"question mismatch: {video_id}")
        pairs.append(
            {
                "video_id": video_id,
                "pair_id": treatment["pair_id"],
                "role": treatment["role"],
                "stratum": treatment["stratum"],
                "question": treatment["question"],
                "reference": annotation["correct_answer"],
                "direct_baseline": baseline["answer"],
                "grounded_treatment": treatment.get("answer", ""),
                "treatment_status": treatment["status"],
                "needs_verification": treatment.get("needs_verification"),
                "answer_claims": treatment.get("answer_claims", []),
                "conflicts_or_missing_qualifiers": treatment.get(
                    "conflicts_or_missing_qualifiers", []
                ),
            }
        )
    bert_predictions: list[str] = []
    bert_references: list[str] = []
    bert_keys: list[tuple[str, str]] = []
    for condition in ("direct_baseline", "grounded_treatment"):
        for pair in pairs:
            if condition == "grounded_treatment" and pair["treatment_status"] != "passed":
                continue
            bert_predictions.append(pair[condition])
            bert_references.append(pair["reference"])
            bert_keys.append((condition, pair["video_id"]))
    bert_values = list(bertscore_fn(bert_predictions, bert_references, bert_model))
    if len(bert_values) != len(bert_keys):
        raise ValueError("BERTScore result count mismatch")
    bert_by_key = dict(zip(bert_keys, bert_values, strict=True))
    scored_rows: list[dict[str, Any]] = []
    for condition in ("direct_baseline", "grounded_treatment"):
        for pair in pairs:
            status = "passed"
            if condition == "grounded_treatment":
                status = pair["treatment_status"]
            if status == "passed":
                if sts_unavailable_reason is None:
                    metrics = _score_values(
                        pair[condition],
                        pair["reference"],
                        meteor_fn=meteor_fn,
                        sts_fn=sts_fn,
                        bertscore=float(bert_by_key[(condition, pair["video_id"])]),
                    )
                else:
                    metrics = {
                        "meteor": float(
                            meteor_fn(pair[condition], pair["reference"])
                        ),
                        "bertscore_f1": float(
                            bert_by_key[(condition, pair["video_id"])]
                        ),
                        "sts": None,
                        "semantic_composite": None,
                    }
            else:
                metrics = {
                    metric: (
                        None
                        if sts_unavailable_reason is not None
                        and metric in {"sts", "semantic_composite"}
                        else 0.0
                    )
                    for metric in METRICS
                }
            scored_rows.append(
                {
                    "video_id": pair["video_id"],
                    "pair_id": pair["pair_id"],
                    "role": pair["role"],
                    "stratum": pair["stratum"],
                    "condition": condition,
                    "status": status,
                    "prediction": pair[condition],
                    "reference": pair["reference"],
                    **metrics,
                }
            )
    by_condition = {
        condition: _aggregate(
            [row for row in scored_rows if row["condition"] == condition]
        )
        for condition in ("direct_baseline", "grounded_treatment")
    }
    def paired_delta(scope: str, metric: str) -> float | None:
        treatment_value = by_condition["grounded_treatment"]["roles"][scope][metric]
        baseline_value = by_condition["direct_baseline"]["roles"][scope][metric]
        if treatment_value is None and baseline_value is None:
            return None
        if treatment_value is None or baseline_value is None:
            raise ValueError(f"paired metric availability is inconsistent: {metric}")
        return treatment_value - baseline_value

    paired_deltas = {
        scope: {
            metric: paired_delta(scope, metric)
            for metric in METRICS
        }
        for scope in ("target", "control")
    }
    paired_deltas["all"] = {
        metric: (
            None
            if by_condition["grounded_treatment"][metric] is None
            and by_condition["direct_baseline"][metric] is None
            else by_condition["grounded_treatment"][metric]
            - by_condition["direct_baseline"][metric]
        )
        for metric in METRICS
    }
    changed_rows = [
        {
            **pair,
            "baseline_prediction": pair["direct_baseline"],
            "treatment_prediction": pair["grounded_treatment"],
        }
        for pair in pairs
        if normalize_nist_text(pair["direct_baseline"])
        != normalize_nist_text(pair["grounded_treatment"])
    ]
    gate_freeze = _load_json(root / "preinference-gate-freeze.json")
    if manual_audit_path is None:
        gate: dict[str, Any] = {
            "status": "pending_manual_audit",
            "passed": None,
            "changed_rows_requiring_inspection": len(changed_rows),
        }
    else:
        manual_audit = _load_json(manual_audit_path)
        if manual_audit.get("prediction_freeze_sha256") != _sha256(
            root / "predictions-freeze.json"
        ):
            raise ValueError("manual audit does not bind prediction freeze")
        gate = classify_gate(changed_rows, manual_audit, gate_freeze["rule"])
    return {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "panel": {
            "row_count": EXPECTED_ROWS,
            "target_count": EXPECTED_TARGETS,
            "control_count": EXPECTED_CONTROLS,
            "passed_treatment_count": sum(
                row["status"] == "passed" for row in treatment_rows
            ),
        },
        "conditions": by_condition,
        "paired_metric_deltas": paired_deltas,
        "changed_row_count": len(changed_rows),
        "changed_rows": changed_rows,
        "gate": gate,
        "rows": scored_rows,
        "integrity": {
            "predictions_frozen_before_labels": True,
            "protocol_sha256": _sha256(root / "protocol.json"),
            "preinference_gate_freeze_sha256": _sha256(
                root / "preinference-gate-freeze.json"
            ),
            "prediction_freeze_sha256": _sha256(
                root / "predictions-freeze.json"
            ),
            "evidence_freeze_sha256": _sha256(root / "evidence-freeze.json"),
            "direct_baseline_sha256": _sha256(baseline_predictions_path),
            "bert_model_sha256": sha256_path(bert_model)
            if bert_model.exists()
            else "injected-test-model",
        },
        "metric_notes": {
            "semantic_composite": "mean(METEOR, BERTScore-F1, STS)",
            "sts_status": (
                "unavailable" if sts_unavailable_reason is not None else "complete"
            ),
            "sts_unavailable_reason": sts_unavailable_reason,
            "failed_treatment_rows": "operational or contract failures receive zero metrics",
            "selection": "label-aware spent-row mechanism panel; descriptive, not a prevalence estimate",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--sts-cache", type=Path)
    parser.add_argument("--sts-unavailable-reason")
    parser.add_argument("--manual-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.baseline_score is not None and args.sts_cache is None:
        raise ValueError("--baseline-score requires --sts-cache")
    sts_cache = (
        PersistentStsCache(args.sts_cache)
        if args.sts_cache is not None
        else None
    )
    if args.baseline_score is not None:
        protocol = _load_json(args.root / "protocol.json")
        seed_baseline_sts_cache(
            cache=sts_cache,
            prior_score_path=args.baseline_score,
            baseline_predictions_path=args.baseline_predictions,
            annotation_root=args.annotations,
            video_ids=[row["video_id"] for row in protocol["rows"]],
        )
    artifact = score(
        root=args.root,
        baseline_predictions_path=args.baseline_predictions,
        annotation_root=args.annotations,
        bert_model=args.bert_model,
        manual_audit_path=args.manual_audit,
        sts_fn=sts_cache if sts_cache is not None else umbc_sts_score,
        sts_unavailable_reason=args.sts_unavailable_reason,
    )
    if sts_cache is not None:
        artifact["metric_notes"]["sts_cache_stats"] = dict(sts_cache.stats)
        artifact["integrity"]["sts_cache_sha256"] = _sha256(args.sts_cache)
        artifact["integrity"]["sts_cache_identity"] = STS_CACHE_IDENTITY
        if args.baseline_score is not None:
            artifact["integrity"]["baseline_score_sha256"] = _sha256(
                args.baseline_score
            )
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gate": artifact["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
