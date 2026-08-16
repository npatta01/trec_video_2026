#!/usr/bin/env python3
"""Authenticate and score the Gemini native-video counting recovery canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from trec_video_2026.vqa_counting_evidence import (
    parse_prediction_count,
    parse_reference_count,
)
from trec_video_2026.vqa_submission import canonical_sha256

BUDGET_CEILING_USD = 1.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"artifact must be an object: {path}")
    return payload


def write_once_json(path: Path, payload: Mapping[str, object]) -> None:
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
    actual = canonical_sha256(
        {key: value for key, value in payload.items() if key != "complete_sha256"}
    )
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{name} self-hash changed")
    return expected


def _authenticate(
    *, root: Path, manifest: Mapping[str, object], freeze: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    manifest_hash = verify_self_hash(manifest, name="request manifest")
    verify_self_hash(freeze, name="prediction freeze")
    if freeze.get("request_manifest_sha256") != manifest_hash:
        raise ValueError("prediction freeze request-manifest binding changed")
    manifest_rows = manifest.get("rows")
    freeze_rows = freeze.get("rows")
    if not isinstance(manifest_rows, list) or not isinstance(freeze_rows, list):
        raise TypeError("manifest and freeze rows must be arrays")
    manifest_ids = [str(row["video_id"]) for row in manifest_rows]
    freeze_ids = [str(row["video_id"]) for row in freeze_rows]
    if len(manifest_ids) != 3 or len(set(manifest_ids)) != 3:
        raise ValueError("canary must contain three unique rows")
    if freeze_ids != manifest_ids:
        raise ValueError("prediction freeze row identity changed")
    completion = freeze.get("completion")
    if completion != {"completed": 3, "total": 3}:
        raise ValueError("prediction freeze is incomplete")

    results: dict[str, Mapping[str, object]] = {}
    for freeze_row in freeze_rows:
        if not isinstance(freeze_row, Mapping):
            raise TypeError("prediction freeze row must be an object")
        video_id = str(freeze_row["video_id"])
        row_root = root / "rows" / video_id
        complete = load_json(row_root / "complete.json")
        complete_hash = verify_self_hash(complete, name=f"row {video_id} complete")
        if complete_hash != freeze_row.get("row_complete_sha256"):
            raise ValueError(f"row {video_id} complete binding changed")
        if complete.get("request_manifest_sha256") != manifest_hash:
            raise ValueError(f"row {video_id} manifest binding changed")
        stage_hashes = complete.get("stage_complete_sha256")
        if not isinstance(stage_hashes, Mapping):
            raise TypeError(f"row {video_id} stage bindings must be an object")
        prior: str | None = None
        for stage in (
            "locate",
            "locate-correction-v2",
            "verify",
            "challenge",
            "additional-verify",
        ):
            if stage not in stage_hashes:
                continue
            terminal = load_json(row_root / stage / "complete.json")
            terminal_hash = verify_self_hash(
                terminal, name=f"row {video_id} {stage} complete"
            )
            if terminal_hash != stage_hashes[stage]:
                raise ValueError(f"row {video_id} {stage} binding changed")
            if terminal.get("prior_complete_sha256") != prior:
                raise ValueError(f"row {video_id} {stage} prior binding changed")
            corrected_locator = (
                stage == "locate" and "locate-correction-v2" in stage_hashes
            )
            valid_outcome = (
                terminal.get("outcome") == "invalid_response"
                if corrected_locator
                else terminal.get("outcome") == "success"
            )
            if not valid_outcome:
                raise ValueError(f"row {video_id} {stage} did not succeed")
            prior = terminal_hash
        result_path = row_root / "result.json"
        if file_sha256(result_path) != complete.get("result_sha256"):
            raise ValueError(f"row {video_id} result hash changed")
        result = load_json(result_path)
        verify_self_hash(result, name=f"row {video_id} result")
        if canonical_sha256(result) != canonical_sha256(freeze_row.get("result")):
            raise ValueError(f"row {video_id} frozen result changed")
        results[video_id] = result
    return results


def _metric_mean(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fields = ("meteor", "bertscore_f1", "sts")
    means = {
        field: statistics.fmean(float(row[field]) for row in rows) for field in fields
    }
    means["semantic_composite"] = statistics.fmean(means.values())
    return means


def build_score(
    *,
    root: Path,
    manifest: Mapping[str, object],
    freeze: Mapping[str, object],
    panel: Mapping[str, object],
    metric_rows: Mapping[str, Mapping[str, object]],
    baseline_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    results = _authenticate(root=root, manifest=manifest, freeze=freeze)
    verify_self_hash(panel, name="panel")
    panel_rows = panel.get("rows")
    if not isinstance(panel_rows, list):
        raise TypeError("panel rows must be an array")
    panel_by_id = {str(row["video_id"]): row for row in panel_rows}
    if (
        set(panel_by_id) != set(results)
        or set(metric_rows) != set(results)
        or set(baseline_rows) != set(results)
    ):
        raise ValueError("panel, metric, and prediction row identities differ")
    rows: list[dict[str, object]] = []
    rescued = 0
    failure_total = 0
    preserved = 0
    control_total = 0
    for row in manifest["rows"]:
        video_id = str(row["video_id"])
        result = results[video_id]
        panel_row = panel_by_id[video_id]
        reference = str(panel_row["reference_answer"])
        reference_count = parse_reference_count(reference).value
        exact = result.get("outcome") == "exact"
        value = result.get("value") if exact else None
        correct = exact and reference_count is not None and value == reference_count
        role = panel_row.get("panel_role")
        if role == "confirmed_failure":
            failure_total += 1
            rescued += int(correct)
        elif role == "correct_control":
            control_total += 1
            preserved += int(correct)
        else:
            raise ValueError(f"unsupported panel role: {role}")
        metrics = metric_rows[video_id]
        semantic_composite = statistics.fmean(
            float(metrics[field]) for field in ("meteor", "bertscore_f1", "sts")
        )
        rows.append(
            {
                "video_id": video_id,
                "panel_role": role,
                "reference_answer": reference,
                "reference_count": reference_count,
                "prediction": str(value) if exact else "",
                "outcome": result["outcome"],
                "value": value,
                "correct": correct,
                **metrics,
                "semantic_composite": semantic_composite,
            }
        )
    if failure_total != 2 or control_total != 1:
        raise ValueError("canary must contain two failures and one control")
    total_cost = float(freeze["total_estimated_cost_usd"])
    semantic_metrics = _metric_mean(rows)
    baseline_semantic_metrics = _metric_mean(list(baseline_rows.values()))
    score = {
        "schema_version": 1,
        "condition": "gemini-native-counting-recovery-v1",
        "completion": {"completed": len(rows), "total": 3},
        "rescue": {
            "rescued": rescued,
            "total": failure_total,
            "rate": rescued / failure_total,
        },
        "control": {
            "preserved": preserved,
            "total": control_total,
            "rate": preserved / control_total,
        },
        "semantic_metrics": semantic_metrics,
        "baseline_semantic_metrics": baseline_semantic_metrics,
        "deltas_treatment_minus_baseline": {
            field: semantic_metrics[field] - baseline_semantic_metrics[field]
            for field in (
                "meteor",
                "bertscore_f1",
                "sts",
                "semantic_composite",
            )
        },
        "exact_results": sum(row["outcome"] == "exact" for row in rows),
        "abstentions": sum(row["outcome"] == "abstain" for row in rows),
        "total_estimated_cost_usd": total_cost,
        "gate_passed": rescued == 2
        and preserved == 1
        and total_cost <= BUDGET_CEILING_USD,
        "evidence_validity": "Diagnostic",
        "decision_disposition": "Promising developmental mechanism"
        if rescued == 2 and preserved == 1 and total_cost <= BUDGET_CEILING_USD
        else "Not promoted under the tested condition",
        "request_manifest_sha256": manifest["complete_sha256"],
        "prediction_freeze_sha256": freeze["complete_sha256"],
        "rows": rows,
    }
    score["complete_sha256"] = canonical_sha256(score)
    return score


def sanitize_score(score: Mapping[str, object]) -> dict[str, object]:
    sanitized = {key: value for key, value in score.items() if key != "rows"}
    sanitized["private_rows_omitted"] = True
    sanitized.pop("complete_sha256", None)
    sanitized["complete_sha256"] = canonical_sha256(sanitized)
    return sanitized


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


def compute_metric_rows(
    *,
    predictions: Mapping[str, str],
    references: Mapping[str, str],
    bert_model: Path,
    sts_cache: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    from bert_score import score as bert_score
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    ids = list(predictions)
    if set(ids) != set(references):
        raise ValueError("metric prediction and reference identities differ")
    prediction_values = [_normalize(predictions[video_id]) for video_id in ids]
    reference_values = [_normalize(references[video_id]) for video_id in ids]
    meteor_values = [
        float(meteor_score([word_tokenize(reference)], word_tokenize(prediction)))
        for prediction, reference in zip(
            prediction_values, reference_values, strict=True
        )
    ]
    _, _, bert_f1 = bert_score(
        prediction_values,
        reference_values,
        model_type=str(bert_model),
        num_layers=17,
        lang="en",
        idf=False,
        rescale_with_baseline=False,
        verbose=False,
        device="cpu",
    )
    identity = sts_cache.get("identity")
    values = sts_cache.get("values")
    if not isinstance(identity, str) or not isinstance(values, Mapping):
        raise TypeError("STS cache is invalid")
    result: dict[str, dict[str, object]] = {}
    for index, video_id in enumerate(ids):
        prediction = prediction_values[index]
        reference = reference_values[index]
        key = _sts_pair_key(identity, prediction, reference)
        if key in values:
            sts = float(values[key])
            provenance = "frozen_umbc_cache_hit"
        else:
            prediction_count = parse_prediction_count(prediction).value
            reference_count = parse_reference_count(reference).value
            if not prediction:
                sts = 0.0
                provenance = "abstention_zero_rule"
            elif prediction_count is not None and reference_count is not None:
                sts = 1.0 if prediction_count == reference_count else 0.0
                provenance = "published_umbc_numeric_value_rule"
            else:
                raise ValueError(f"STS pair is unavailable: {prediction!r}")
        result[video_id] = {
            "meteor": meteor_values[index],
            "bertscore_f1": float(bert_f1[index]),
            "sts": sts,
            "sts_provenance": provenance,
        }
    environment = {
        "python_runtime_version": platform.python_version(),
        "nltk_distribution_version": importlib.metadata.version("nltk"),
        "bert_score_distribution_version": importlib.metadata.version("bert-score"),
        "torch_distribution_version": importlib.metadata.version("torch"),
        "transformers_distribution_version": importlib.metadata.version("transformers"),
        "bert_model_config_sha256": file_sha256(bert_model / "config.json"),
        "bertscore_device": "cpu",
        "bertscore_num_layers": "17",
    }
    return result, environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--baseline-score", required=True, type=Path)
    parser.add_argument("--bert-model", required=True, type=Path)
    parser.add_argument("--sts-cache", required=True, type=Path)
    parser.add_argument("--sanitized-output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    freeze = load_json(args.freeze)
    panel = load_json(args.panel)
    baseline_score = load_json(args.baseline_score)
    results = _authenticate(root=args.root, manifest=manifest, freeze=freeze)
    panel_by_id = {str(row["video_id"]): row for row in panel["rows"]}
    predictions = {
        video_id: str(result["value"]) if result.get("outcome") == "exact" else ""
        for video_id, result in results.items()
    }
    references = {
        video_id: str(panel_by_id[video_id]["reference_answer"]) for video_id in results
    }
    sts_cache = load_json(args.sts_cache)
    metric_rows, environment = compute_metric_rows(
        predictions=predictions,
        references=references,
        bert_model=args.bert_model,
        sts_cache=sts_cache,
    )
    baseline_payload = baseline_score.get("rows")
    if not isinstance(baseline_payload, Mapping):
        raise TypeError("baseline score rows are missing")
    baseline_condition = baseline_payload.get("gemini-3.1-pro-native-av-4fps")
    if not isinstance(baseline_condition, list):
        raise TypeError("baseline Gemini rows are missing")
    baseline_by_id = {
        str(row["video_id"]): row
        for row in baseline_condition
        if isinstance(row, Mapping) and str(row.get("video_id")) in results
    }
    score = build_score(
        root=args.root,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows=metric_rows,
        baseline_rows=baseline_by_id,
    )
    score["metric_environment"] = environment
    score.pop("complete_sha256")
    score["complete_sha256"] = canonical_sha256(score)
    write_once_json(args.root / "score.json", score)
    write_once_json(args.sanitized_output, sanitize_score(score))
    print(json.dumps(sanitize_score(score), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
