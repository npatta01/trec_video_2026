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
_MODEL = "gemini-3.1-pro-preview"
_SCHEMA_SHA256 = {
    "locate": "6350576f6623e3776b913430b3ee28b8816dbafcad187a92069a1eca45ab8218",
    "verify": "9015bd3b85d05f8fb60f93beab96734598c8216b7567d0bfeafb57c70cf05de9",
    "challenge": "011940d2a3b054a34f5febda00a5274d7cc94398e4a8498d870ce44c975fbb0e",
    "additional-verify": "9015bd3b85d05f8fb60f93beab96734598c8216b7567d0bfeafb57c70cf05de9",
}
_STAGE_ORDER = (
    "locate",
    "locate-correction-v2",
    "verify",
    "challenge",
    "additional-verify",
)
_MANDATORY_STAGES = ("locate", "verify", "challenge")
_STANDARD_TERMINAL_FILE_NAMES = {
    "raw_response_sha256": "raw-response.txt",
    "parsed_sha256": "parsed.json",
    "validated_sha256": "validated.json",
    "provider_metadata_sha256": "provider-metadata.json",
}
_CROSS_STAGE_FILE_NAMES = {
    "invalid_raw_response_sha256": ("locate", "raw-response.txt"),
    "invalid_parsed_sha256": ("locate", "parsed.json"),
}


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


def _verify_terminal_referenced_files(
    *,
    row_root: Path,
    stage: str,
    terminal: Mapping[str, object],
    required: set[str],
) -> None:
    references = {
        field: (row_root / stage / filename)
        for field, filename in _STANDARD_TERMINAL_FILE_NAMES.items()
        if field in terminal
    }
    references.update(
        {
            field: (row_root / referenced_stage / filename)
            for field, (referenced_stage, filename) in _CROSS_STAGE_FILE_NAMES.items()
            if field in terminal
        }
    )
    missing = required.difference(references)
    if missing:
        raise ValueError(
            f"row {row_root.name} {stage} terminal references missing: "
            f"{sorted(missing)}"
        )
    for field, path in references.items():
        expected = terminal[field]
        if not isinstance(expected, str):
            raise ValueError(f"row {row_root.name} {stage} {field} is not a hash")
        if not path.is_file():
            raise ValueError(
                f"row {row_root.name} {stage} {field.removesuffix('_sha256')} "
                "file missing"
            )
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"row {row_root.name} {stage} {field.removesuffix('_sha256')} "
                "file hash changed"
            )


def _verify_upload_chain(
    *,
    row_root: Path,
    row: Mapping[str, object],
    complete: Mapping[str, object],
    video_id: str,
) -> tuple[str, Mapping[str, object]]:
    upload = load_json(row_root / "upload.json")
    upload_hash = verify_self_hash(upload, name=f"row {video_id} upload")
    upload_claim = load_json(row_root / "upload-claim.json")
    upload_claim_hash = verify_self_hash(
        upload_claim, name=f"row {video_id} upload claim"
    )
    if upload.get("upload_claim_complete_sha256") != upload_claim_hash:
        raise ValueError(f"row {video_id} upload claim binding changed")
    if complete.get("upload_complete_sha256") != upload_hash:
        raise ValueError(f"row {video_id} upload binding changed")
    if complete.get("delivery_sha256") != row.get("delivery_sha256"):
        raise ValueError(f"row {video_id} delivery binding changed")
    for name, payload in (("upload", upload), ("upload claim", upload_claim)):
        if payload.get("video_id") != video_id:
            raise ValueError(f"row {video_id} {name} identity changed")
        if payload.get("delivery_sha256") != row.get("delivery_sha256"):
            raise ValueError(f"row {video_id} {name} delivery binding changed")
        if payload.get("delivery_mime_type") != row.get("delivery_mime_type"):
            raise ValueError(f"row {video_id} {name} MIME binding changed")
    if upload.get("file_state") != "ACTIVE":
        raise ValueError(f"row {video_id} upload is not active")
    if not isinstance(upload.get("file_uri"), str) or not upload["file_uri"]:
        raise ValueError(f"row {video_id} upload URI is missing")
    return upload_hash, upload


def _verify_paid_stage_request(
    *,
    row_root: Path,
    stage: str,
    terminal: Mapping[str, object],
    row: Mapping[str, object],
    manifest_hash: str,
    upload: Mapping[str, object],
    upload_hash: str,
    prior: str | None,
    video_id: str,
) -> None:
    request = load_json(row_root / stage / "request.json")
    request_hash = verify_self_hash(request, name=f"row {video_id} {stage} request")
    if request.get("stage") != stage:
        raise ValueError(f"row {video_id} {stage} request stage identity changed")
    if terminal.get("request_sha256") != request_hash:
        raise ValueError(f"row {video_id} {stage} request binding changed")
    if request.get("prior_complete_sha256") != prior:
        raise ValueError(f"row {video_id} {stage} request prior binding changed")
    expected_bindings = {
        "request_manifest_sha256": manifest_hash,
        "upload_complete_sha256": upload_hash,
        "delivery_sha256": row.get("delivery_sha256"),
        "file_uri": upload.get("file_uri"),
        "mime_type": row.get("delivery_mime_type"),
    }
    for field, expected in expected_bindings.items():
        if request.get(field) != expected:
            label = field.removesuffix("_sha256").replace("_", " ")
            raise ValueError(f"row {video_id} {stage} {label} binding changed")
    expected_config = {
        "model": _MODEL,
        "fps": 4.0 if stage == "locate" else 12.0,
        "response_schema_sha256": _SCHEMA_SHA256[stage],
        "temperature": 0.0,
        "seed": 42,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
        "thinking_level": "LOW",
        "media_resolution": "HIGH",
    }
    if any(request.get(field) != value for field, value in expected_config.items()):
        raise ValueError(f"row {video_id} {stage} request configuration changed")
    clipped = stage in {"verify", "additional-verify"}
    offsets = (request.get("start_offset"), request.get("end_offset"))
    if (clipped and not all(isinstance(value, str) and value for value in offsets)) or (
        not clipped and offsets != (None, None)
    ):
        raise ValueError(f"row {video_id} {stage} request offsets changed")
    claim = load_json(row_root / stage / "paid-call-claim.json")
    claim_hash = verify_self_hash(claim, name=f"row {video_id} {stage} paid-call claim")
    if claim.get("stage") != stage:
        raise ValueError(f"row {video_id} {stage} paid-call claim stage changed")
    if claim.get("request_complete_sha256") != request_hash:
        raise ValueError(f"row {video_id} {stage} paid-call request binding changed")
    if terminal.get("paid_call_claim_sha256") != claim_hash:
        raise ValueError(f"row {video_id} {stage} paid-call claim binding changed")


def _stage_names(stage_hashes: Mapping[str, object], *, video_id: str) -> list[str]:
    names = set(stage_hashes)
    unknown = names.difference(_STAGE_ORDER)
    if unknown:
        raise ValueError(f"row {video_id} has unexpected stage(s): {sorted(unknown)}")
    missing = set(_MANDATORY_STAGES).difference(names)
    if missing:
        raise ValueError(
            f"row {video_id} is missing mandatory stage(s): {sorted(missing)}"
        )
    correction = "locate-correction-v2" in names
    additional = "additional-verify" in names
    ordered = ["locate"]
    if correction:
        ordered.append("locate-correction-v2")
    ordered.extend(("verify", "challenge"))
    if additional:
        ordered.append("additional-verify")
    if names != set(ordered):
        raise ValueError(f"row {video_id} stage chain is not exact")
    return ordered


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
    manifest_by_id = {str(row["video_id"]): row for row in manifest_rows}
    freeze_ids = [str(row["video_id"]) for row in freeze_rows]
    if len(manifest_ids) != 3 or len(set(manifest_ids)) != 3:
        raise ValueError("canary must contain three unique rows")
    completion = freeze.get("completion")
    if not isinstance(completion, Mapping):
        raise TypeError("prediction freeze completion must be an object")
    completed = completion.get("completed")
    total = completion.get("total")
    if (
        not isinstance(completed, int)
        or not isinstance(total, int)
        or completed != total
        or total != len(freeze_ids)
        or not 1 <= total <= len(manifest_ids)
    ):
        raise ValueError("prediction freeze is incomplete")
    if total < len(manifest_ids) and freeze.get("manifest_row_count") != len(
        manifest_ids
    ):
        raise ValueError("partial prediction freeze manifest row count changed")
    if freeze_ids != manifest_ids[:total]:
        raise ValueError("prediction freeze row identity changed")

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
        if complete.get("video_id") != video_id:
            raise ValueError(f"row {video_id} video_id changed")
        if complete.get("outcome") != "success":
            raise ValueError(f"row {video_id} complete outcome is not success")
        manifest_row = manifest_by_id[video_id]
        if not isinstance(manifest_row, Mapping):
            raise TypeError(f"row {video_id} manifest row must be an object")
        upload_hash, upload = _verify_upload_chain(
            row_root=row_root,
            row=manifest_row,
            complete=complete,
            video_id=video_id,
        )
        stage_hashes = complete.get("stage_complete_sha256")
        if not isinstance(stage_hashes, Mapping):
            raise TypeError(f"row {video_id} stage bindings must be an object")
        ordered_stages = _stage_names(stage_hashes, video_id=video_id)
        correction_present = "locate-correction-v2" in stage_hashes
        prior: str | None = None
        for stage in ordered_stages:
            terminal = load_json(row_root / stage / "complete.json")
            terminal_hash = verify_self_hash(
                terminal, name=f"row {video_id} {stage} complete"
            )
            if terminal_hash != stage_hashes.get(stage):
                raise ValueError(f"row {video_id} {stage} binding changed")
            if terminal.get("stage") != stage:
                raise ValueError(f"row {video_id} {stage} stage identity changed")
            if terminal.get("prior_complete_sha256") != prior:
                raise ValueError(f"row {video_id} {stage} prior binding changed")
            if stage == "locate":
                expected_outcome = (
                    "invalid_response" if correction_present else "success"
                )
                if terminal.get("outcome") != expected_outcome:
                    if correction_present:
                        raise ValueError(
                            f"row {video_id} locate must be invalid_response "
                            "when correction is present"
                        )
                    raise ValueError(
                        f"row {video_id} locator must succeed without correction"
                    )
            elif terminal.get("outcome") != "success":
                raise ValueError(f"row {video_id} {stage} did not succeed")
            if stage == "locate-correction-v2":
                required_files = {
                    "invalid_raw_response_sha256",
                    "invalid_parsed_sha256",
                    "validated_sha256",
                }
            else:
                _verify_paid_stage_request(
                    row_root=row_root,
                    stage=stage,
                    terminal=terminal,
                    row=manifest_row,
                    manifest_hash=manifest_hash,
                    upload=upload,
                    upload_hash=upload_hash,
                    prior=prior,
                    video_id=video_id,
                )
                required_files = {
                    "raw_response_sha256",
                    "parsed_sha256",
                    "validated_sha256",
                    "provider_metadata_sha256",
                }
                if terminal.get("outcome") == "invalid_response":
                    required_files.intersection_update(
                        {"raw_response_sha256", "provider_metadata_sha256"}
                    )
            _verify_terminal_referenced_files(
                row_root=row_root,
                stage=stage,
                terminal=terminal,
                required=required_files,
            )
            prior = terminal_hash
        if "additional-verify" in stage_hashes:
            challenge_payload = load_json(row_root / "challenge" / "validated.json")
            requested_ranges = challenge_payload.get("requested_ranges")
            missed_instances = challenge_payload.get("missed_instances")
            if not isinstance(requested_ranges, list) or not isinstance(
                missed_instances, list
            ):
                raise ValueError(
                    f"row {video_id} additional verification lacks challenge requests"
                )
            if not requested_ranges and not missed_instances:
                raise ValueError(
                    f"row {video_id} additional verification was not requested"
                )
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
        not set(results).issubset(panel_by_id)
        or set(metric_rows) != set(results)
        or set(baseline_rows) != set(results)
    ):
        raise ValueError("panel, metric, and prediction row identities differ")
    rows: list[dict[str, object]] = []
    rescued = 0
    failure_total = 0
    preserved = 0
    control_total = 0
    for freeze_row in freeze["rows"]:
        video_id = str(freeze_row["video_id"])
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
    gate_evaluable = failure_total == 2 and control_total == 1 and len(rows) == 3
    total_cost = float(freeze["total_estimated_cost_usd"])
    semantic_metrics = _metric_mean(rows)
    baseline_semantic_metrics = _metric_mean(list(baseline_rows.values()))
    score = {
        "schema_version": 1,
        "condition": manifest["condition"],
        "completion": {"completed": len(rows), "total": len(rows)},
        "rescue": {
            "rescued": rescued,
            "total": failure_total,
            "rate": rescued / failure_total if failure_total else None,
        },
        "control": {
            "preserved": preserved,
            "total": control_total,
            "rate": preserved / control_total if control_total else None,
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
        "gate_evaluable": gate_evaluable,
        "gate_passed": gate_evaluable
        and rescued == 2
        and preserved == 1
        and total_cost <= BUDGET_CEILING_USD,
        "evidence_validity": "Diagnostic",
        "decision_disposition": "Promising developmental mechanism"
        if gate_evaluable
        and rescued == 2
        and preserved == 1
        and total_cost <= BUDGET_CEILING_USD
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
