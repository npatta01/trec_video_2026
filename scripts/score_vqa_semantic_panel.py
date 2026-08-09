#!/usr/bin/env python3
"""Score frozen VQA predictions with the NIST 2025 semantic metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlencode
import urllib.error
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from trec_video_2026.vqa_evaluation import (  # noqa: E402
    ndcg_from_gains,
    semantic_composite,
)


EXPECTED_BOOTSTRAP_SEED = 20260809
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_STS_TIMEOUT = 10.0
DEFAULT_STS_RETRIES = 3
STS_ENDPOINT = "http://swoogle.umbc.edu/StsService/GetStsSim"
_LABEL_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "correct_answer",
        "expected_answer",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_answer",
        "reference_answers",
    }
)
MetricFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]


def sha256_path(path: Path) -> str:
    """Hash one file or a deterministic relative-path/file-hash directory."""

    path = path.resolve()
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(
            (item.relative_to(path) for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        ):
            digest.update(child.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_path(path / child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise ValueError(f"Input path does not exist: {path}")


def normalize_nist_text(text: str) -> str:
    """Apply the NIST 2025 strip/lower/collapse-whitespace normalization."""

    if not isinstance(text, str):
        raise ValueError("Scored text must be a string")
    return " ".join(text.strip().lower().split())


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "injected-or-runtime-provided"


def _default_meteor(prediction: str, reference: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    return float(
        meteor_score(
            [word_tokenize(reference)],
            word_tokenize(prediction),
        )
    )


def _default_bertscore(
    predictions: list[str], references: list[str], model: Path
) -> list[float]:
    from bert_score import score

    _, _, f1 = score(
        predictions,
        references,
        model_type=str(model),
        num_layers=17,
        lang="en",
        idf=False,
        rescale_with_baseline=False,
        verbose=False,
    )
    values = f1.tolist() if hasattr(f1, "tolist") else list(f1)
    return [_finite(value, field="bertscore_f1") for value in values]


def umbc_sts_score(
    prediction: str,
    reference: str,
    *,
    timeout: float = DEFAULT_STS_TIMEOUT,
    retries: int = DEFAULT_STS_RETRIES,
) -> float:
    """Call UMBC STS with encoded parameters and a bounded retry budget."""

    if timeout <= 0:
        raise ValueError("STS timeout must be positive")
    if isinstance(retries, bool) or retries <= 0:
        raise ValueError("STS retries must be positive")
    query = urlencode(
        {
            "operation": "api",
            "phrase1": reference,
            "phrase2": prediction,
        }
    )
    url = f"{STS_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for _attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "NIST - TRECVID"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = response.read().decode("utf-8").strip()
            return _finite(value, field="sts")
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
            last_error = error
    raise ValueError(f"STS request failed after {retries} attempts") from last_error


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON input: {path}") from error


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read prediction input: {path}") from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid prediction JSON or JSONL: {path}") from error
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(row, dict) for row in payload)
    ):
        raise ValueError(f"Predictions must be a non-empty list of objects: {path}")
    return payload


def _reject_label_fields(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _LABEL_KEYS:
                raise ValueError(
                    f"Prediction contains label-bearing field at {location}.{key}"
                )
            _reject_label_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_label_fields(child, f"{location}[{index}]")


def _validate_frozen_predictions(
    rows: Sequence[Mapping[str, Any]], *, condition_id: str, expected_rows: int
) -> list[dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(
            f"Condition {condition_id} has {len(rows)} prediction rows; "
            f"expected {expected_rows}"
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        _reject_label_fields(raw, f"$[{index}]")
        video_id = raw.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Prediction row {index} has an invalid video_id")
        video_id = video_id.strip()
        if video_id in seen:
            raise ValueError(f"Duplicate prediction ID in {condition_id}: {video_id}")
        seen.add(video_id)
        if raw.get("condition_id") not in (None, condition_id):
            raise ValueError(f"Prediction condition mismatch for {video_id}")
        answer = raw.get("answer")
        if (
            raw.get("status") != "passed"
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(
                f"Condition {condition_id} is incomplete at {video_id}; "
                f"completed {len(selected)}/{expected_rows}"
            )
        selected.append(
            {
                "condition_id": condition_id,
                "video_id": video_id,
                "answer": answer,
                "status": raw.get("status"),
            }
        )
    return selected


def _question_stratum(question: str) -> str:
    lowered = question.lower()
    if re.search(r"\bhow many\b|\bhow much\b|\bnumber of\b", lowered):
        return "counting"
    if re.search(r"\bwhy\b|\bwhat causes\b|\breason", lowered):
        return "causal"
    if re.search(r"\bwhen\b|\bafter\b|\bbefore\b|\bend\b|\bnext\b", lowered):
        return "temporal"
    if re.search(r"\bhear\b|\bsay\b|\bsound\b|\bmusic\b|\bspeak", lowered):
        return "audio_or_text"
    return "other"


def _load_annotations(annotation_root: Path) -> dict[str, dict[str, Any]]:
    if not annotation_root.is_dir():
        raise ValueError(f"Annotation root does not exist: {annotation_root}")
    annotations: dict[str, dict[str, Any]] = {}
    paths = sorted(annotation_root.glob("*.json"))
    if not paths:
        raise ValueError(f"Annotation root has no JSON files: {annotation_root}")
    for path in paths:
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Annotation must be an object: {path}")
        video_id = path.stem
        question = payload.get("question")
        reference = payload.get("correct_answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Annotation question is missing: {path}")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Annotation reference is missing: {path}")
        if video_id in annotations:
            raise ValueError(f"Duplicate annotation ID: {video_id}")
        annotations[video_id] = {
            "video_id": video_id,
            "question": question,
            "reference": reference,
            "question_stratum": _question_stratum(question),
        }
    return annotations


def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [_finite(row[field], field=field) for row in rows]
    return statistics.fmean(values)


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, include_strata: bool = True
) -> dict[str, Any]:
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["question_stratum"])].append(row)
    aggregate: dict[str, Any] = {
        "row_count": len(rows),
        "completed_count": len(rows),
        "completion_rate": 1.0 if rows else 0.0,
    }
    for field in ("meteor", "bertscore_f1", "sts", "semantic_composite", "ndcg"):
        aggregate[field] = _mean_metric(rows, field)
    aggregate["mean_answer_token_count"] = statistics.fmean(
        row["answer_token_count"] for row in rows
    )
    if include_strata:
        aggregate["question_strata"] = {
            name: _aggregate_rows(group, include_strata=False)
            for name, group in sorted(strata.items())
        }
    return aggregate


def _bootstrap_interval(
    values: Sequence[float], *, iterations: int, seed: int, confidence: float
) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot bootstrap an empty paired metric")
    source = random.Random(seed)
    sample_size = len(values)
    means = sorted(
        statistics.fmean(source.choices(list(values), k=sample_size))
        for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = min(iterations - 1, int(tail * iterations))
    upper_index = min(iterations - 1, max(0, int((1.0 - tail) * iterations)))
    return {
        "lower": means[lower_index],
        "upper": means[upper_index],
        "confidence": confidence,
        "iterations": iterations,
        "seed": seed,
        "exploratory": True,
    }


def _paired_summary(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any] | None:
    condition_ids = list(condition_rows)
    if len(condition_ids) < 2:
        return None
    baseline_id, treatment_id = condition_ids[:2]
    baseline = {str(row["video_id"]): row for row in condition_rows[baseline_id]}
    treatment = {str(row["video_id"]): row for row in condition_rows[treatment_id]}
    common_ids = sorted(set(baseline) & set(treatment))
    if len(common_ids) != len(baseline) or len(common_ids) != len(treatment):
        raise ValueError("Conditions do not have identical frozen prediction IDs")
    fields = ("meteor", "bertscore_f1", "sts", "semantic_composite", "ndcg")
    deltas = {
        field: [
            float(treatment[video_id][field]) - float(baseline[video_id][field])
            for video_id in common_ids
        ]
        for field in fields
    }
    return {
        "baseline_condition_id": baseline_id,
        "treatment_condition_id": treatment_id,
        "paired_count": len(common_ids),
        "metric_deltas": {
            field: statistics.fmean(values) for field, values in deltas.items()
        },
        "bootstrap_intervals": {
            field: _bootstrap_interval(
                values,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            for field, values in deltas.items()
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _score_condition(
    condition_id: str,
    predictions: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    *,
    bert_model: Path,
    meteor_fn: MetricFunction,
    bertscore_fn: BertScoreFunction,
    sts_fn: MetricFunction,
) -> list[dict[str, Any]]:
    for prediction in predictions:
        video_id = str(prediction["video_id"])
        if video_id not in annotations:
            raise ValueError(f"Missing reference for prediction: {video_id}")

    normalized_predictions = [
        normalize_nist_text(str(prediction["answer"])) for prediction in predictions
    ]
    normalized_references = [
        normalize_nist_text(str(annotations[str(prediction["video_id"])]["reference"]))
        for prediction in predictions
    ]
    bert_values = list(
        bertscore_fn(normalized_predictions, normalized_references, bert_model)
    )
    if len(bert_values) != len(predictions):
        raise ValueError(
            f"BERTScore returned {len(bert_values)} rows for {len(predictions)} predictions"
        )

    scored: list[dict[str, Any]] = []
    for prediction, normalized_prediction, normalized_reference, bert_value in zip(
        predictions,
        normalized_predictions,
        normalized_references,
        bert_values,
        strict=True,
    ):
        annotation = annotations[str(prediction["video_id"])]
        meteor = _finite(
            meteor_fn(normalized_prediction, normalized_reference), field="meteor"
        )
        bertscore_f1 = _finite(bert_value, field="bertscore_f1")
        sts = _finite(sts_fn(normalized_prediction, normalized_reference), field="sts")
        metrics = {
            "meteor": meteor,
            "bertscore_f1": bertscore_f1,
            "sts": sts,
        }
        scored.append(
            {
                "condition_id": condition_id,
                "video_id": prediction["video_id"],
                "question": annotation["question"],
                "question_stratum": annotation["question_stratum"],
                "answer": prediction["answer"],
                "reference": annotation["reference"],
                "normalized_answer": normalized_prediction,
                "normalized_reference": normalized_reference,
                "answer_token_count": len(normalized_prediction.split()),
                **metrics,
                "semantic_composite": semantic_composite(metrics),
                "ndcg": ndcg_from_gains([semantic_composite(metrics)]),
            }
        )
    return scored


def score_panel(
    *,
    annotation_root: Path,
    condition_paths: Mapping[str, Path],
    bert_model: Path,
    expected_rows: int,
    output_path: Path | None = None,
    meteor_fn: MetricFunction | None = None,
    bertscore_fn: BertScoreFunction | None = None,
    sts_fn: MetricFunction | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Score every frozen condition and optionally atomically write evidence."""

    if not condition_paths:
        raise ValueError("At least one condition is required")
    if expected_rows <= 0:
        raise ValueError("Expected rows must be positive")
    if not bert_model.exists():
        raise ValueError(f"BERT model path does not exist: {bert_model}")
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    normalized_condition_paths = {
        str(condition_id): Path(path) for condition_id, path in condition_paths.items()
    }
    if len(normalized_condition_paths) != len(condition_paths):
        raise ValueError("Condition IDs must be unique and non-empty")
    if any(not condition_id.strip() for condition_id in normalized_condition_paths):
        raise ValueError("Condition IDs must be non-empty")

    # Freeze and hash every prediction input before the annotation root is read.
    frozen_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    for condition_id, path in normalized_condition_paths.items():
        rows = _load_prediction_rows(path)
        frozen_predictions[condition_id] = _validate_frozen_predictions(
            rows,
            condition_id=condition_id,
            expected_rows=expected_rows,
        )
        prediction_hashes[condition_id] = sha256_path(path)

    annotations = _load_annotations(annotation_root)
    meteor_fn = meteor_fn or _default_meteor
    bertscore_fn = bertscore_fn or _default_bertscore
    sts_fn = sts_fn or umbc_sts_score
    condition_rows = {
        condition_id: _score_condition(
            condition_id,
            predictions,
            annotations,
            bert_model=bert_model,
            meteor_fn=meteor_fn,
            bertscore_fn=bertscore_fn,
            sts_fn=sts_fn,
        )
        for condition_id, predictions in frozen_predictions.items()
    }
    if any(len(rows) != expected_rows for rows in condition_rows.values()):
        raise ValueError("Completion gate failed after scoring")

    reports = {
        condition_id: _aggregate_rows(rows)
        for condition_id, rows in condition_rows.items()
    }
    all_rows = [row for rows in condition_rows.values() for row in rows]
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "scorer": {
            "name": "NIST 2025 semantic VQA scorer",
            "normalization": "strip, lowercase, collapse whitespace",
            "bert_model": str(bert_model),
            "bert_num_layers": 17,
            "bert_idf": False,
            "bert_rescale_with_baseline": False,
            "sts_endpoint": STS_ENDPOINT,
            "sts_timeout_seconds": DEFAULT_STS_TIMEOUT,
            "sts_max_attempts": DEFAULT_STS_RETRIES,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_confidence": confidence,
            "semantic_composite": "mean(meteor, bertscore_f1, sts)",
            "single_response_ndcg": "1.0, degenerate and non-informative",
        },
        "panel": {
            "collection": "VQA 2025 training",
            "annotation_files": len(annotations),
            "expected_rows_per_condition": expected_rows,
            "declared_conditions": list(normalized_condition_paths),
            "completed_per_condition": {
                condition_id: len(rows) for condition_id, rows in condition_rows.items()
            },
        },
        "conditions": reports,
        "aggregate": _aggregate_rows(all_rows),
        "paired": _paired_summary(
            condition_rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        ),
        "rows": dict(condition_rows),
        "integrity": {
            "labels_joined_after_prediction_freeze": True,
            "all_conditions_complete": True,
            "completion_denominator": f"{expected_rows}/{expected_rows} per condition",
            "duplicate_prediction_ids_checked": True,
            "finite_metrics_checked": True,
        },
        "inputs": {
            "annotation_root": str(annotation_root),
            "annotation_root_sha256": sha256_path(annotation_root),
            "prediction_files": {
                condition_id: {
                    "path": str(path),
                    "sha256": prediction_hashes[condition_id],
                }
                for condition_id, path in normalized_condition_paths.items()
            },
            "bert_model_sha256": sha256_path(bert_model),
            "scorer_sha256": sha256_path(Path(__file__)),
        },
        "dependencies": {
            name: _distribution_version(name) for name in ("nltk", "bert-score")
        },
    }
    if output_path is not None:
        _atomic_write_json(output_path, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="Frozen prediction JSON/JSONL condition; repeat for each condition",
    )
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sts-timeout", type=float, default=DEFAULT_STS_TIMEOUT)
    parser.add_argument("--sts-retries", type=int, default=DEFAULT_STS_RETRIES)
    return parser


def _parse_conditions(values: Sequence[str]) -> dict[str, Path]:
    conditions: dict[str, Path] = {}
    for value in values:
        condition_id, separator, path = value.partition("=")
        if not separator or not condition_id.strip() or not path.strip():
            raise ValueError(f"Condition must be ID=PATH: {value}")
        condition_id = condition_id.strip()
        if condition_id in conditions:
            raise ValueError(f"Duplicate condition ID: {condition_id}")
        conditions[condition_id] = Path(path)
    return conditions


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conditions = _parse_conditions(args.condition)
    sts = lambda prediction, reference: umbc_sts_score(  # noqa: E731
        prediction,
        reference,
        timeout=args.sts_timeout,
        retries=args.sts_retries,
    )
    artifact = score_panel(
        annotation_root=args.annotations,
        condition_paths=conditions,
        bert_model=args.bert_model,
        expected_rows=args.expected_rows,
        output_path=args.output,
        sts_fn=sts,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "completed_per_condition": artifact["panel"]["completed_per_condition"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
