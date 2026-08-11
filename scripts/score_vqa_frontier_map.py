#!/usr/bin/env python3
"""Post-freeze scorer for the VQA frontier-map diagnostic.

The scorer is intentionally offline and private-boundary aware.  It verifies
provider, audit, panel, source, and baseline bindings before parsing annotation
JSON, then produces only a sanitized projection: aggregate metrics, safe row
identities, failure labels/counts, hashes, and gate decisions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for _path in (SCRIPTS_ROOT, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score_vqa_factorial as factorial  # noqa: E402


EXPECTED_PANEL_COUNT = 35
EXPECTED_BASELINE_CONDITION = "qwen_baseline"
EXPECTED_SCORER_IDENTITY = "score_vqa_frontier_map.py:offline-v1"
EXPECTED_BERT_MODEL_SNAPSHOT = factorial.EXPECTED_BERT_MODEL_SNAPSHOT
EXPECTED_BOOTSTRAP_SEED = factorial.EXPECTED_BOOTSTRAP_SEED
FAILURE_CAUSES = (
    "question_misunderstood",
    "wrong_modality",
    "wrong_temporal_or_local_evidence",
    "perception_failure",
    "reasoning_failure_with_correct_evidence",
    "answer_generation_or_normalization",
    "ambiguous_or_insufficient_evidence",
)
METRIC_FIELDS = (
    "normalized_exact_match",
    "token_f1",
    "meteor",
    "bertscore_f1",
    "sts",
    "semantic_composite",
    "ndcg",
)
_PRIVATE_ROW_KEYS = frozenset(
    {
        "answer",
        "raw_response",
        "media_path",
        "reference",
        "reference_answer",
        "correct_answer",
        "notes",
    }
)


def score_frontier_map(
    *,
    provider_results: str | Path,
    provider_freeze: str | Path,
    audit_freeze: str | Path,
    panel_manifest: str | Path,
    annotations_root: str | Path,
    qwen_baseline: str | Path,
    qwen_sha256sums: str | Path,
    expected_baseline_source_sha256: str,
    bootstrap_iterations: int = factorial.EXPECTED_BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    """Score frozen frontier-map predictions and return a safe projection."""

    provider_results_path = Path(provider_results)
    provider_freeze_path = Path(provider_freeze)
    audit_freeze_path = Path(audit_freeze)
    panel_manifest_path = Path(panel_manifest)
    annotations_path = Path(annotations_root)
    qwen_baseline_path = Path(qwen_baseline)
    qwen_sha256sums_path = Path(qwen_sha256sums)

    panel = _load_panel(panel_manifest_path)
    ordered_ids = [row["video_id"] for row in panel]
    categories = {row["video_id"]: row["category"] for row in panel}
    source_hashes = {row["video_id"]: row["media_sha256"] for row in panel}
    order_digest = _canonical_sha256(ordered_ids)
    source_digest = _canonical_sha256(
        [
            {"video_id": video_id, "source_sha256": source_hashes[video_id]}
            for video_id in ordered_ids
        ]
    )
    panel_sha256 = _sha256_path(panel_manifest_path)

    provider_payload = _verify_provider_freeze(
        provider_results_path,
        provider_freeze_path,
        panel_sha256=panel_sha256,
        order_digest=order_digest,
        source_digest=source_digest,
    )
    provider_rows = _provider_rows(
        provider_payload,
        ordered_ids=ordered_ids,
        categories=categories,
        source_hashes=source_hashes,
    )
    providers = tuple(sorted(provider_rows))
    provider_keys = [
        (provider, video_id) for provider in providers for video_id in ordered_ids
    ]
    aware_verdicts = _verify_audit_freeze(audit_freeze_path, provider_keys)
    _validate_failure_causes(provider_rows)

    qwen_sha256 = _verify_qwen_sha256sums(qwen_baseline_path, qwen_sha256sums_path)
    if qwen_sha256 != expected_baseline_source_sha256:
        raise ValueError("baseline source hash does not match expected binding")
    annotation_root_sha256 = _annotation_root_sha256(annotations_path)
    qwen_rows = _load_qwen_baseline(
        qwen_baseline_path,
        ordered_ids=ordered_ids,
        baseline_source_sha256=qwen_sha256,
        annotation_root_sha256=annotation_root_sha256,
    )

    annotations = _load_annotations(annotations_path, ordered_ids, categories)
    condition_predictions: dict[str, list[Mapping[str, Any]]] = {
        EXPECTED_BASELINE_CONDITION: qwen_rows,
    }
    for provider in providers:
        condition_predictions[provider] = [
            {
                "condition_id": provider,
                "video_id": row["video_id"],
                "answer": row["answer"],
            }
            for row in provider_rows[provider]
            if row["status"] == "completed"
        ]

    scored_completed, pair_counts = factorial._score_unique_pairs(
        condition_predictions,
        annotations,
        bert_model=Path(EXPECTED_BERT_MODEL_SNAPSHOT),
        meteor_fn=_offline_metric,
        bertscore_fn=_offline_bertscore,
        sts_fn=_offline_metric,
        sts_cache=factorial.STSCache(identity="frontier-map-offline"),
        sts_identity="frontier-map-offline",
    )

    qwen_scored = scored_completed[EXPECTED_BASELINE_CONDITION]
    qwen_by_video = {row["video_id"]: row for row in qwen_scored}
    conditions: dict[str, Any] = {
        EXPECTED_BASELINE_CONDITION: _condition_summary(
            qwen_scored,
            completed_count=EXPECTED_PANEL_COUNT,
            total_count=EXPECTED_PANEL_COUNT,
            completed_rows=qwen_scored,
        )
    }
    complete_panel_rows: dict[str, list[dict[str, Any]]] = {
        EXPECTED_BASELINE_CONDITION: qwen_scored
    }
    sanitized_rows: list[dict[str, Any]] = []
    failure_counts: dict[str, dict[str, int]] = {}

    for provider in providers:
        completed_by_video = {
            row["video_id"]: row for row in scored_completed.get(provider, [])
        }
        all_rows = []
        for provider_row in provider_rows[provider]:
            video_id = provider_row["video_id"]
            if video_id in completed_by_video:
                scored = dict(completed_by_video[video_id])
                scored["operational_failure"] = False
            else:
                scored = _zero_metric_row(provider, video_id, annotations[video_id])
            all_rows.append(scored)
            sanitized_rows.append(
                _safe_projection_row(
                    provider_row,
                    aware_verdicts[(provider, video_id)],
                    scored,
                )
            )
        complete_panel_rows[provider] = all_rows
        completed_rows = list(completed_by_video.values())
        conditions[provider] = _condition_summary(
            all_rows,
            completed_count=len(completed_rows),
            total_count=EXPECTED_PANEL_COUNT,
            completed_rows=completed_rows,
        )
        failure_counts[provider] = dict(
            sorted(
                Counter(
                    cause
                    for row in provider_rows[provider]
                    for cause in row["failure_causes"]
                ).items()
            )
        )

    paired = {
        f"{provider}_vs_{EXPECTED_BASELINE_CONDITION}": _paired_comparison(
            complete_panel_rows[provider],
            [qwen_by_video[video_id] for video_id in ordered_ids],
            bootstrap_iterations=bootstrap_iterations,
        )
        for provider in providers
    }
    map_gate = {
        provider: _map_gate(provider_rows[provider], aware_verdicts)
        for provider in providers
    }

    result = {
        "schema_version": 1,
        "scorer_identity": EXPECTED_SCORER_IDENTITY,
        "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
        "conditions": conditions,
        "paired_comparisons": paired,
        "failure_counts": failure_counts,
        "map_gate": map_gate,
        "reka_diagnostics": conditions.get("reka", {}),
        "degenerate_ndcg": {
            "explicit": True,
            "conditions": {
                condition: summary["ndcg_degenerate"]
                for condition, summary in conditions.items()
            },
        },
        "pair_scoring": pair_counts | {"helper": "score_vqa_factorial._score_unique_pairs"},
        "hashes": {
            "panel_manifest_sha256": panel_sha256,
            "provider_results_sha256": _sha256_path(provider_results_path),
            "provider_freeze_sha256": _sha256_path(provider_freeze_path),
            "audit_freeze_sha256": _sha256_path(audit_freeze_path),
            "qwen_baseline_sha256": qwen_sha256,
            "annotation_root_sha256": annotation_root_sha256,
            "ordered_video_ids_sha256": order_digest,
            "source_hashes_sha256": source_digest,
        },
        "rows": sanitized_rows,
    }
    _reject_private_projection(result)
    return result


def _load_panel(path: Path) -> list[dict[str, str]]:
    payload = _read_json(path)
    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_PANEL_COUNT:
        raise ValueError("panel manifest must contain the exact 35-row panel")
    rows = []
    seen: set[str] = set()
    for index, row in enumerate(examples):
        if not isinstance(row, Mapping):
            raise TypeError("panel rows must be objects")
        video_id = _text(row, "video_id")
        if video_id in seen:
            raise ValueError("panel manifest contains duplicate video IDs")
        seen.add(video_id)
        category = _text(row, "category")
        media_sha256 = _sha256_text(row.get("media_sha256"), "media_sha256")
        rows.append(
            {
                "ordinal": str(row.get("ordinal", index + 1)),
                "video_id": video_id,
                "category": category,
                "media_sha256": media_sha256,
            }
        )
    return rows


def _verify_provider_freeze(
    provider_results: Path,
    provider_freeze: Path,
    *,
    panel_sha256: str,
    order_digest: str,
    source_digest: str,
) -> Mapping[str, Any]:
    if not provider_freeze.is_file():
        raise ValueError("provider freeze is missing")
    freeze = _read_json(provider_freeze)
    if freeze.get("frozen") is not True:
        raise ValueError("provider freeze is not frozen")
    if freeze.get("provider_results_sha256") != _sha256_path(provider_results):
        raise ValueError("provider freeze does not bind provider results")
    checks = {
        "panel manifest": freeze.get("panel_manifest_sha256") == panel_sha256,
        "ordered panel": freeze.get("ordered_video_ids_sha256") == order_digest,
        "source hashes": freeze.get("source_hashes_sha256") == source_digest,
        "row count": freeze.get("row_count") == EXPECTED_PANEL_COUNT * 2,
    }
    for name, passed in checks.items():
        if not passed:
            raise ValueError(f"provider freeze binding mismatch: {name}")
    payload = _read_json(provider_results)
    if payload.get("ordered_video_ids_sha256") != order_digest:
        raise ValueError("provider results do not bind the ordered panel")
    if payload.get("source_hashes_sha256") != source_digest:
        raise ValueError("provider results source hashes do not match panel")
    return payload


def _provider_rows(
    provider_payload: Mapping[str, Any],
    *,
    ordered_ids: Sequence[str],
    categories: Mapping[str, str],
    source_hashes: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    rows = provider_payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("provider results rows must be a list")
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("provider rows must be objects")
        provider = _text(raw, "provider")
        video_id = _text(raw, "video_id")
        status = _text(raw, "status")
        if video_id not in categories:
            raise ValueError("provider results are not aligned to the ordered panel")
        if raw.get("category") != categories[video_id]:
            raise ValueError("provider row category does not match panel")
        if raw.get("source_sha256") != source_hashes[video_id]:
            raise ValueError("provider row source hash does not match panel")
        if status not in {"completed", "failed"}:
            raise ValueError("provider row status must be completed or failed")
        failure_causes = raw.get("failure_causes", [])
        if not isinstance(failure_causes, list) or not all(
            isinstance(cause, str) for cause in failure_causes
        ):
            raise TypeError("failure causes must be a list of strings")
        row = {
            "provider": provider,
            "video_id": video_id,
            "category": categories[video_id],
            "source_sha256": source_hashes[video_id],
            "status": status,
            "failure_causes": list(failure_causes),
            "failure_code": raw.get("failure_code"),
            "map_payload_present": bool(raw.get("map_payload_present")),
            "cost_usd": _number(raw.get("cost_usd", 0.0), "cost_usd"),
        }
        if status == "completed":
            row["answer"] = _text(raw, "answer")
        by_provider[provider].append(row)
    if set(by_provider) != {"gemini", "reka"}:
        raise ValueError("provider results must contain gemini and reka")
    for provider, provider_rows in by_provider.items():
        actual_order = [row["video_id"] for row in provider_rows]
        if actual_order != list(ordered_ids):
            raise ValueError(f"{provider} results do not match the ordered panel")
    return {provider: by_provider[provider] for provider in sorted(by_provider)}


def _verify_audit_freeze(
    audit_freeze: Path, provider_keys: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if not audit_freeze.is_file():
        raise ValueError("audit freeze is missing")
    freeze = _read_json(audit_freeze)
    if freeze.get("frozen") is not True:
        raise ValueError("audit freeze is not frozen")
    rows = freeze.get("rows")
    if not isinstance(rows, list):
        raise TypeError("audit freeze rows must be a list")
    expected = list(provider_keys)
    actual = [(_text(row, "provider"), _text(row, "video_id")) for row in rows]
    if sorted(actual) != sorted(expected):
        raise ValueError("audit freeze rows do not match provider rows")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("audit freeze bindings must be an object")
    if bindings.get("provider_rows") != len(expected):
        raise ValueError("audit freeze provider row count is invalid")
    root = audit_freeze.parent
    for name in ("blind", "aware"):
        digest_key = f"{name}_verdicts_sha256"
        verdict_path = root / f"verdicts.{name}.jsonl"
        if digest_key in bindings and bindings[digest_key] != _sha256_path(verdict_path):
            raise ValueError(f"audit freeze {name} verdict hash mismatch")
    aware = _read_aware_verdicts(root / "verdicts.aware.jsonl")
    if sorted(aware) != sorted(expected):
        raise ValueError("aware audit verdicts do not match provider rows")
    return aware


def _read_aware_verdicts(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    verdicts: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not path.is_file():
        raise ValueError("aware audit verdicts are missing")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        verdict = json.loads(line)
        if not isinstance(verdict, Mapping):
            raise TypeError(f"aware verdict line {line_number} must be an object")
        if verdict.get("view") != "aware":
            raise ValueError("aware verdict file contains a non-aware row")
        decision = verdict.get("usable_map_decision")
        if decision not in {"usable", "unusable"}:
            raise ValueError("aware verdict usable-map decision is invalid")
        key = (_text(verdict, "provider"), _text(verdict, "video_id"))
        if key in verdicts:
            raise ValueError("duplicate aware audit verdict")
        verdicts[key] = {
            "usable_map_decision": decision,
            "confidence": verdict.get("confidence"),
        }
    return verdicts


def _verify_qwen_sha256sums(qwen_baseline: Path, qwen_sha256sums: Path) -> str:
    if not qwen_sha256sums.is_file():
        raise ValueError("Qwen SHA256SUMS manifest is missing")
    if qwen_sha256sums.is_symlink() or qwen_baseline.is_symlink():
        raise ValueError("Qwen SHA256SUMS inputs cannot be symlinks")
    entries: dict[str, str] = {}
    for line in qwen_sha256sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator:
            raise ValueError("Qwen SHA256SUMS manifest is malformed")
        _sha256_text(digest, "Qwen SHA256SUMS digest")
        if Path(relative).name != relative:
            raise ValueError("Qwen SHA256SUMS path must be a file name")
        entries[relative] = digest
    digest = entries.get(qwen_baseline.name)
    if digest is None:
        raise ValueError("Qwen SHA256SUMS does not bind the baseline file")
    if digest != _sha256_path(qwen_baseline):
        raise ValueError("Qwen SHA256SUMS baseline hash mismatch")
    return digest


def _load_qwen_baseline(
    path: Path,
    *,
    ordered_ids: Sequence[str],
    baseline_source_sha256: str,
    annotation_root_sha256: str,
) -> list[dict[str, str]]:
    payload = _read_json(path)
    if payload.get("condition_id") != EXPECTED_BASELINE_CONDITION:
        raise ValueError("Qwen baseline condition identity is invalid")
    if payload.get("scorer_identity") != EXPECTED_SCORER_IDENTITY:
        raise ValueError("Qwen baseline scorer identity mismatch")
    if payload.get("bert_model_snapshot") != EXPECTED_BERT_MODEL_SNAPSHOT:
        raise ValueError("Qwen baseline BERT snapshot mismatch")
    if payload.get("annotation_root_sha256") != annotation_root_sha256:
        raise ValueError("Qwen baseline annotation-root hash mismatch")
    if _sha256_path(path) != baseline_source_sha256:
        raise ValueError("Qwen baseline source hash mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("Qwen baseline rows must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Qwen baseline rows must be objects")
        video_id = _text(row, "video_id")
        if video_id in by_id:
            raise ValueError("Qwen baseline contains duplicate video IDs")
        by_id[video_id] = row
    missing = [video_id for video_id in ordered_ids if video_id not in by_id]
    if missing:
        raise ValueError("Qwen baseline missing exact ordered 35-ID subset")
    return [
        {
            "condition_id": EXPECTED_BASELINE_CONDITION,
            "video_id": video_id,
            "answer": _text(by_id[video_id], "answer"),
        }
        for video_id in ordered_ids
    ]


def _load_annotations(
    root: Path,
    ordered_ids: Sequence[str],
    categories: Mapping[str, str],
) -> dict[str, Mapping[str, str]]:
    annotations = {}
    for video_id in ordered_ids:
        payload = _read_json(root / f"{video_id}.json")
        reference = payload.get("reference", payload.get("correct_answer"))
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("annotation reference is missing")
        stratum = payload.get("question_stratum", categories[video_id])
        if stratum != categories[video_id]:
            raise ValueError("annotation stratum does not match panel category")
        annotations[video_id] = {
            "reference": reference,
            "question_stratum": categories[video_id],
        }
    return annotations


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    completed_count: int,
    total_count: int,
    completed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failure_zero = factorial._aggregate_rows(rows)
    failure_zero["completed_count"] = completed_count
    failure_zero["completion_rate"] = completed_count / total_count
    failure_zero["failure_zero"] = True
    if completed_rows:
        completed_cases = factorial._aggregate_rows(completed_rows)
    else:
        completed_cases = _empty_completed_summary()
    summary = dict(failure_zero)
    summary.update(
        {
            "failure_zero": failure_zero,
            "completed_cases": completed_cases,
            "completion_denominator": {
                "completed": completed_count,
                "total": total_count,
            },
        }
    )
    return summary


def _empty_completed_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": 0,
        "completed_count": 0,
        "completion_rate": 0.0,
        "mean_answer_token_count": 0.0,
        "ndcg_degenerate": True,
    }
    for metric in METRIC_FIELDS:
        summary[metric] = 0.0
    return summary


def _zero_metric_row(
    condition_id: str, video_id: str, annotation: Mapping[str, Any]
) -> dict[str, Any]:
    row = {
        "condition_id": condition_id,
        "video_id": video_id,
        "stratum": annotation["question_stratum"],
        "answer_token_count": 0,
        "ndcg_degenerate": True,
        "operational_failure": True,
    }
    for metric in METRIC_FIELDS:
        row[metric] = 0.0
    return row


def _paired_comparison(
    provider_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    if len(provider_rows) != len(baseline_rows) or len(provider_rows) != EXPECTED_PANEL_COUNT:
        raise ValueError("paired comparisons require aligned 35-row vectors")
    deltas = {
        metric: [
            _metric_value(provider_row, metric) - _metric_value(baseline_row, metric)
            for provider_row, baseline_row in zip(provider_rows, baseline_rows, strict=True)
        ]
        for metric in METRIC_FIELDS
    }
    semantic_deltas = deltas["semantic_composite"]
    wins = sum(delta > 0 for delta in semantic_deltas)
    losses = sum(delta < 0 for delta in semantic_deltas)
    ties = len(semantic_deltas) - wins - losses
    return {
        "metric_deltas": {
            metric: statistics.fmean(values) for metric, values in deltas.items()
        },
        "wins_losses_ties": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "bootstrap_intervals": factorial._bootstrap_intervals(
            deltas,
            iterations=bootstrap_iterations,
            seed=EXPECTED_BOOTSTRAP_SEED,
        ),
    }


def _map_gate(
    provider_rows: Sequence[Mapping[str, Any]],
    aware_verdicts: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    usable_by_category: dict[str, int] = defaultdict(int)
    total_by_category: dict[str, int] = defaultdict(int)
    usable_count = 0
    for row in provider_rows:
        category = str(row["category"])
        total_by_category[category] += 1
        verdict = aware_verdicts[(str(row["provider"]), str(row["video_id"]))]
        if verdict["usable_map_decision"] == "usable":
            usable_count += 1
            usable_by_category[category] += 1
    category_counts = {
        category: {
            "usable": usable_by_category[category],
            "total": total_by_category[category],
        }
        for category in sorted(total_by_category)
    }
    overall = usable_count >= 28
    category_min = all(value["usable"] >= 3 for value in category_counts.values())
    return {
        "usable_count": usable_count,
        "total": len(provider_rows),
        "overall_28_of_35": overall,
        "category_min_3_of_5": category_min,
        "stricter_4_of_5_category_count": sum(
            value["usable"] >= 4 for value in category_counts.values()
        ),
        "passed": overall and category_min,
        "categories": category_counts,
    }


def _safe_projection_row(
    provider_row: Mapping[str, Any],
    aware_verdict: Mapping[str, Any],
    scored: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "condition_id": provider_row["provider"],
        "video_id": provider_row["video_id"],
        "category": provider_row["category"],
        "completed": provider_row["status"] == "completed",
        "operational_failure": bool(scored.get("operational_failure")),
        "map_usable": aware_verdict["usable_map_decision"] == "usable",
        "failure_code": provider_row.get("failure_code"),
        "failure_causes": list(provider_row["failure_causes"]),
        "cost_usd": provider_row["cost_usd"],
        "metrics": {metric: _metric_value(scored, metric) for metric in METRIC_FIELDS},
    }


def _validate_failure_causes(
    provider_rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    allowed = set(FAILURE_CAUSES)
    for provider, rows in provider_rows.items():
        for row in rows:
            causes = row["failure_causes"]
            if row["status"] == "failed" and not causes:
                raise ValueError(f"{provider} failed row requires a failure cause")
            unknown = sorted(set(causes) - allowed)
            if unknown:
                raise ValueError("unapproved failure cause label: " + ", ".join(unknown))


def _offline_metric(prediction: str, reference: str) -> float:
    return float(factorial.token_f1(prediction, reference))


def _offline_bertscore(
    predictions: list[str], references: list[str], _bert_model: Path
) -> list[float]:
    return [
        _offline_metric(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def _annotation_root_sha256(root: Path) -> str:
    if not root.is_dir():
        raise ValueError("annotation root is missing")
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        if path.is_symlink():
            raise ValueError("annotation root cannot contain symlinks")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reject_private_projection(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in _PRIVATE_ROW_KEYS:
                raise ValueError(f"private field leaked into projection: {key}")
            _reject_private_projection(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_private_projection(nested)
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            "private raw" in lowered
            or "private reviewer notes" in lowered
            or "reference answer" in lowered
            or "/private/" in lowered
        ):
            raise ValueError("private text leaked into projection")


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    value = row[metric]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{metric} must be numeric")
    return float(value)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty text")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from error
    if value.lower() != value:
        raise ValueError(f"{field} must be lowercase")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
