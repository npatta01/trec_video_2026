#!/usr/bin/env python3
"""Post-freeze gates and metrics for issue #34 counting evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from trec_video_2026.vqa_counting_evidence import (
    parse_prediction_count,
    parse_reference_count,
)
from trec_video_2026.vqa_submission import canonical_sha256

FULL_PANEL_ROWS = 39
CANARY_FAILURES = 5
CANARY_CONTROLS = 1
CANARY_REQUIRED_RESCUES = 3
CANARY_REQUIRED_CONTROLS = 1
INITIAL_PROJECTED_REQUEST_USD = 0.07
BUDGET_CEILING_USD = 3.0


def file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_once_json(path: Path, payload: object) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting score: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def verify_self_hash(payload: Mapping[str, object], *, name: str) -> str:
    expected = payload.get("complete_sha256")
    if not isinstance(expected, str):
        raise TypeError(f"{name} has no complete_sha256")
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    )
    if actual != expected:
        raise ValueError(f"{name} complete_sha256 changed")
    return expected


def authenticate_canary_chain(
    *, root: Path, panel: Mapping[str, object], manifest: Mapping[str, object]
) -> tuple[Mapping[str, object], dict[str, Mapping[str, object]]]:
    manifest_sha = verify_self_hash(manifest, name="request manifest")
    verify_self_hash(panel, name="panel")
    freeze = load_json(root / "canary-predictions-freeze.json")
    if not isinstance(freeze, Mapping):
        raise TypeError("canary freeze must be an object")
    verify_self_hash(freeze, name="canary freeze")
    if freeze.get("request_manifest_sha256") != manifest_sha:
        raise ValueError("canary freeze request-manifest binding changed")

    canary_ids = manifest.get("canary_video_ids")
    if (
        not isinstance(canary_ids, list)
        or len(canary_ids) != 6
        or len(set(canary_ids)) != 6
    ):
        raise ValueError("canary manifest must contain six unique IDs")
    freeze_rows = freeze.get("rows")
    if not isinstance(freeze_rows, list) or len(freeze_rows) != 6:
        raise ValueError("canary freeze must bind six terminal rows")
    freeze_by_id: dict[str, Mapping[str, object]] = {}
    for row in freeze_rows:
        if not isinstance(row, Mapping):
            raise TypeError("canary freeze row must be an object")
        video_id = str(row.get("video_id"))
        if video_id in freeze_by_id:
            raise ValueError("canary freeze IDs are not unique")
        freeze_by_id[video_id] = row
    if set(freeze_by_id) != set(canary_ids):
        raise ValueError("canary freeze IDs changed")

    results: dict[str, Mapping[str, object]] = {}
    for video_id_value in canary_ids:
        video_id = str(video_id_value)
        freeze_row = freeze_by_id[video_id]
        complete_path = root / "rows" / video_id / "complete.json"
        if file_sha256(complete_path) != freeze_row.get("complete_sha256"):
            raise ValueError(f"canary complete hash changed: {video_id}")
        complete = load_json(complete_path)
        if not isinstance(complete, Mapping):
            raise TypeError(f"complete must be an object: {video_id}")
        result_path = root / "rows" / video_id / "result.json"
        result_sha = file_sha256(result_path)
        if result_sha != complete.get("result_sha256"):
            raise ValueError(f"canary result hash changed: {video_id}")
        if result_sha != freeze_row.get("result_sha256"):
            raise ValueError(f"canary freeze result hash changed: {video_id}")
        result = load_json(result_path)
        if not isinstance(result, Mapping):
            raise TypeError(f"result must be an object: {video_id}")
        if result.get("status") != complete.get("status") or result.get(
            "status"
        ) != freeze_row.get("status"):
            raise ValueError(f"canary terminal status changed: {video_id}")
        results[video_id] = result
    return freeze, results


def build_canary_decision(
    *, root: Path, panel: Mapping[str, object], manifest: Mapping[str, object]
) -> dict[str, object]:
    _freeze, results = authenticate_canary_chain(
        root=root, panel=panel, manifest=manifest
    )
    manifest_sha = manifest["complete_sha256"]
    canary_ids = manifest.get("canary_video_ids")
    assert isinstance(canary_ids, list)

    panel_rows = panel.get("rows")
    if not isinstance(panel_rows, list):
        raise TypeError("panel must contain rows")
    panel_by_id = {str(row["video_id"]): row for row in panel_rows}
    failure_ids = [
        video_id
        for video_id in canary_ids
        if panel_by_id[video_id].get("panel_role") == "confirmed_failure"
    ]
    control_ids = [
        video_id
        for video_id in canary_ids
        if panel_by_id[video_id].get("panel_role") == "correct_control"
    ]
    if len(failure_ids) != CANARY_FAILURES or len(control_ids) != CANARY_CONTROLS:
        raise ValueError("canary roles changed")

    costs: list[float] = []
    for video_id in canary_ids:
        result = results[video_id]
        cost = result.get("estimated_cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            costs.append(float(cost))

    passed = sum(result.get("status") == "passed" for result in results.values())
    rescued_ids = [
        video_id
        for video_id in failure_ids
        if results[video_id].get("status") == "passed"
        and results[video_id].get("derived_count")
        == panel_by_id[video_id].get("reference_count")
    ]
    preserved_ids = [
        video_id
        for video_id in control_ids
        if results[video_id].get("status") == "passed"
        and results[video_id].get("derived_count")
        == panel_by_id[video_id].get("reference_count")
    ]
    observed_cost = sum(costs)
    projected_per_remaining = max(
        INITIAL_PROJECTED_REQUEST_USD,
        max(costs, default=0.0) * 1.25,
    )
    projected_total = (
        observed_cost + (FULL_PANEL_ROWS - len(canary_ids)) * projected_per_remaining
    )
    authorized = (
        passed == 6
        and len(rescued_ids) >= CANARY_REQUIRED_RESCUES
        and len(preserved_ids) >= CANARY_REQUIRED_CONTROLS
        and projected_total <= BUDGET_CEILING_USD
    )
    decision = {
        "schema_version": 1,
        "request_manifest_sha256": manifest_sha,
        "panel_sha256": panel["complete_sha256"],
        "canary_freeze_sha256": file_sha256(root / "canary-predictions-freeze.json"),
        "schema_validity": {"passed": passed, "required": 6},
        "failure_rescue": {
            "rescued": len(rescued_ids),
            "required": CANARY_REQUIRED_RESCUES,
            "total": CANARY_FAILURES,
        },
        "control_preservation": {
            "preserved": len(preserved_ids),
            "required": CANARY_REQUIRED_CONTROLS,
            "total": CANARY_CONTROLS,
        },
        "observed_canary_cost_usd": observed_cost,
        "projected_per_remaining_request_usd": projected_per_remaining,
        "projected_total_cost_usd": projected_total,
        "budget_ceiling_usd": BUDGET_CEILING_USD,
        "authorized": authorized,
        "rescued_video_ids": rescued_ids,
        "preserved_control_video_ids": preserved_ids,
    }
    decision["complete_sha256"] = canonical_sha256(decision)
    return decision


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _sts_pair_key(identity: str, prediction: str, reference: str) -> str:
    payload = (
        json.dumps(
            {"identity": identity, "prediction": prediction, "reference": reference},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _mean_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    fields = ("meteor", "bertscore_f1", "sts", "semantic_composite")
    return {
        "row_count": len(rows),
        **{
            field: statistics.fmean(float(row[field]) for row in rows)
            for field in fields
        },
    }


def build_canary_metrics(
    *,
    root: Path,
    panel: Mapping[str, object],
    manifest: Mapping[str, object],
    baseline_score: Mapping[str, object],
    sts_cache: Mapping[str, object],
    bert_model: Path,
    meteor_fn: Callable[[str, str], float] | None = None,
    bertscore_fn: Callable[[list[str], list[str], Path], Sequence[float]] | None = None,
    metric_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    freeze, results = authenticate_canary_chain(
        root=root, panel=panel, manifest=manifest
    )
    decision = load_json(root / "canary-decision.json")
    if not isinstance(decision, Mapping):
        raise TypeError("canary decision must be an object")
    verify_self_hash(decision, name="canary decision")
    if decision.get("request_manifest_sha256") != manifest.get("complete_sha256"):
        raise ValueError("canary decision binding changed")
    if decision.get("panel_sha256") != panel.get("complete_sha256"):
        raise ValueError("canary decision panel binding changed")
    if decision.get("canary_freeze_sha256") != file_sha256(
        root / "canary-predictions-freeze.json"
    ):
        raise ValueError("canary decision freeze binding changed")
    if freeze.get("complete_sha256") is None:
        raise ValueError("canary freeze is incomplete")
    canary_ids = [str(value) for value in manifest["canary_video_ids"]]
    panel_by_id = {str(row["video_id"]): row for row in panel["rows"]}
    predictions: list[str] = []
    references: list[str] = []
    for video_id in canary_ids:
        result = results[video_id]
        if result.get("status") != "passed":
            raise ValueError(f"canary result is not passed: {video_id}")
        predictions.append(_normalize(str(result["derived_count"])))
        references.append(_normalize(str(panel_by_id[video_id]["reference_answer"])))

    if meteor_fn is None:
        from nltk.tokenize import word_tokenize
        from nltk.translate.meteor_score import meteor_score

        def meteor_fn(prediction: str, reference: str) -> float:
            return float(
                meteor_score([word_tokenize(reference)], word_tokenize(prediction))
            )

    if bertscore_fn is None:
        import bert_score
        from bert_score import score

        def bertscore_fn(
            candidate_predictions: list[str],
            candidate_references: list[str],
            model: Path,
        ) -> Sequence[float]:
            _, _, f1 = score(
                candidate_predictions,
                candidate_references,
                model_type=str(model),
                num_layers=17,
                lang="en",
                idf=False,
                rescale_with_baseline=False,
                verbose=False,
                device="cpu",
            )
            return [float(value) for value in f1]

    if metric_environment is None:
        import bert_score

        metric_environment = {
            "python_runtime_version": platform.python_version(),
            "nltk_distribution_version": importlib.metadata.version("nltk"),
            "bert_score_distribution_version": importlib.metadata.version("bert-score"),
            "bert_score_module_version": str(bert_score.__version__),
            "torch_distribution_version": importlib.metadata.version("torch"),
            "transformers_distribution_version": importlib.metadata.version(
                "transformers"
            ),
            "meteor_tokenizer": "nltk.word_tokenize:english:punkt_tab",
            "bertscore_num_layers": "17",
            "bertscore_idf": "false",
            "bertscore_rescale_with_baseline": "false",
            "bertscore_device": "cpu",
        }

    meteor_values = [
        float(meteor_fn(prediction, reference))
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    bert_values = list(bertscore_fn(predictions, references, bert_model))
    if len(bert_values) != len(canary_ids):
        raise ValueError("BERTScore row count changed")
    identity = sts_cache.get("identity")
    values = sts_cache.get("values")
    if not isinstance(identity, str) or not isinstance(values, Mapping):
        raise TypeError("STS cache is invalid")
    sts_values: list[float] = []
    sts_provenance: list[str] = []
    for prediction, reference in zip(predictions, references, strict=True):
        key = _sts_pair_key(identity, prediction, reference)
        if key in values:
            sts_values.append(float(values[key]))
            sts_provenance.append("frozen_umbc_cache_hit")
            continue
        prediction_count = parse_prediction_count(prediction).value
        reference_count = parse_reference_count(reference).value
        if (
            prediction_count is not None
            and reference_count is not None
            and prediction_count != reference_count
        ):
            # UMBC EBIQUITY-CORE's published aligner compares numeric terms by
            # value and returns zero when those values differ. This path is
            # used only when the historical service is unavailable and the
            # exact pair is absent from the frozen cache.
            sts_values.append(0.0)
            sts_provenance.append("published_umbc_numeric_mismatch_rule")
            continue
        raise ValueError(
            f"STS pair is unavailable and not a numeric mismatch: {prediction!r}"
        )

    treatment_rows = []
    for index, video_id in enumerate(canary_ids):
        metrics = {
            "meteor": meteor_values[index],
            "bertscore_f1": float(bert_values[index]),
            "sts": sts_values[index],
        }
        treatment_rows.append(
            {
                "video_id": video_id,
                "prediction": predictions[index],
                "reference": references[index],
                **metrics,
                "semantic_composite": statistics.fmean(metrics.values()),
                "sts_provenance": sts_provenance[index],
                "count_unit": results[video_id]["count_unit"],
                "panel_role": panel_by_id[video_id]["panel_role"],
            }
        )

    baseline_rows_payload = baseline_score.get("rows")
    if not isinstance(baseline_rows_payload, Mapping):
        raise TypeError("baseline score rows are missing")
    baseline_all = baseline_rows_payload.get("gemini-3.1-pro-native-av-4fps")
    if not isinstance(baseline_all, list):
        raise TypeError("baseline Gemini score rows are missing")
    baseline_by_id = {str(row["video_id"]): row for row in baseline_all}
    baseline_rows = [baseline_by_id[video_id] for video_id in canary_ids]
    aggregates = {
        "baseline": _mean_metrics(baseline_rows),
        "atomic_treatment": _mean_metrics(treatment_rows),
    }
    deltas = {
        field: float(aggregates["atomic_treatment"][field])
        - float(aggregates["baseline"][field])
        for field in ("meteor", "bertscore_f1", "sts", "semantic_composite")
    }
    artifact = {
        "schema_version": 1,
        "panel_scope": "six-row canary only; scale-up blocked",
        "request_manifest_sha256": manifest["complete_sha256"],
        "canary_decision_sha256": file_sha256(root / "canary-decision.json"),
        "baseline_score_sha256": canonical_sha256(baseline_score),
        "bert_model_snapshot": bert_model.name,
        "bert_model_sha256": file_sha256(bert_model / "config.json"),
        "metric_environment": dict(metric_environment),
        "sts_cache_identity": identity,
        "sts_cache_hits": sts_provenance.count("frozen_umbc_cache_hit"),
        "sts_numeric_rule_uses": sts_provenance.count(
            "published_umbc_numeric_mismatch_rule"
        ),
        "conditions": aggregates,
        "deltas_treatment_minus_baseline": deltas,
        "rows": {
            "baseline": baseline_rows,
            "atomic_treatment": treatment_rows,
        },
    }
    artifact["complete_sha256"] = canonical_sha256(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-score", type=Path)
    parser.add_argument("--bert-model", type=Path)
    parser.add_argument("--sts-cache", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    panel = load_json(args.panel)
    manifest = load_json(args.manifest)
    if not isinstance(panel, Mapping) or not isinstance(manifest, Mapping):
        raise TypeError("panel and manifest must be objects")
    decision = build_canary_decision(root=args.root, panel=panel, manifest=manifest)
    write_once_json(args.root / "canary-decision.json", decision)
    if args.metrics_output is not None:
        if (
            args.baseline_score is None
            or args.bert_model is None
            or args.sts_cache is None
        ):
            raise ValueError(
                "metrics output requires baseline score, BERT model, and STS cache"
            )
        baseline_score = load_json(args.baseline_score)
        sts_cache = load_json(args.sts_cache)
        if not isinstance(baseline_score, Mapping) or not isinstance(
            sts_cache, Mapping
        ):
            raise ValueError("baseline score and STS cache must be objects")
        metrics = build_canary_metrics(
            root=args.root,
            panel=panel,
            manifest=manifest,
            baseline_score=baseline_score,
            sts_cache=sts_cache,
            bert_model=args.bert_model,
        )
        write_once_json(args.metrics_output, metrics)
    print(
        json.dumps(
            {
                "authorized": decision["authorized"],
                "schema_validity": decision["schema_validity"],
                "failure_rescue": decision["failure_rescue"],
                "control_preservation": decision["control_preservation"],
                "observed_canary_cost_usd": decision["observed_canary_cost_usd"],
                "projected_total_cost_usd": decision["projected_total_cost_usd"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
