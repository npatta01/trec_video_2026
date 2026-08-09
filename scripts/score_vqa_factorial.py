#!/usr/bin/env python3
"""Score the explicit four-cell VQA model-by-prompt factorial.

The scorer deliberately has a separate entry point from
``score_vqa_semantic_panel.py``.  That older entry point is the frozen two-cell
Task 1 scorer; this module owns the four-cell Phase 1 analysis and never
widens the older contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for _path in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score_vqa_semantic_panel as task1  # noqa: E402
from trec_video_2026.vqa_evaluation import (  # noqa: E402
    normalized_exact_match,
    semantic_composite,
    token_f1,
)


EXPECTED_SCHEMA_VERSION = 2
EXPECTED_BOOTSTRAP_SEED = 20260809
EXPECTED_BOOTSTRAP_ITERATIONS = 2000
EXPECTED_BERT_MODEL_SNAPSHOT = "722cf37b1afa9454edce342e7895e588b6ff1d59"
EXPECTED_BERT_SCORE_MODULE_VERSION = "0.3.12"
EXPECTED_STS_TIMEOUT = 10.0
EXPECTED_STS_RETRIES = 3
STS_ENDPOINT = task1.STS_ENDPOINT
EXPECTED_CONDITION_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
DEFAULT_PROFILE_PATH = (
    REPOSITORY_ROOT / "configs/vqa/vqa_factorial_full_condition_profiles.json"
)
TASK1_SCORER_PATH = REPOSITORY_ROOT / "scripts/score_vqa_semantic_panel.py"
PANEL_ID_DIGEST_ALGORITHM = task1.PANEL_ID_DIGEST_ALGORITHM
EXPECTED_RUNTIME_CONTROLS = {
    "runtime_image": (
        "docker.io/pytorch/pytorch@sha256:"
        "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
    ),
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "frame_count": 12,
    "max_new_tokens": 64,
    "do_sample": False,
    "thinking": False,
    "min_pixels": 65536,
    "max_pixels": 262144,
    "transcript_policy": "shared-accepted-whisper",
}
EXPECTED_CONDITION_BINDINGS = {
    "4b-current": (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "current",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "4b-question-aware": (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "question-aware",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "8b-current": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "current",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
    "8b-question-aware": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "question-aware",
        65536,
        262144,
        "shared-accepted-whisper",
    ),
}
METRIC_FIELDS = ("meteor", "bertscore_f1", "sts", "semantic_composite")
_LABEL_KEYS = task1._LABEL_KEYS
MetricFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]


PanelProfile = task1.PanelProfile
CANONICAL_PANEL_PROFILE = task1.CANONICAL_PANEL_PROFILE
sha256_path = task1.sha256_path
sorted_video_id_set_sha256 = task1.sorted_video_id_set_sha256
normalize_nist_text = task1.normalize_nist_text
_finite = task1._finite
_load_annotations = task1._load_annotations
_validate_panel_identity = task1._validate_panel_identity
umbc_sts_score = task1.umbc_sts_score
urllib = task1.urllib


class STSCache:
    """Private in-memory STS cache; keys are opaque identity hashes."""

    def __init__(
        self,
        values: Mapping[str, float] | None = None,
        identity: str | None = None,
    ) -> None:
        self.values = dict(values or {})
        self.identity = identity


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def profile_sha256(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(profile)).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON input: {path}") from error


def validate_factorial_profile(
    payload: Mapping[str, Any], *, strict: bool = True
) -> dict[str, Any]:
    """Validate the exact four-cell profile and return a detached copy."""

    required = {
        "run_name",
        "input_uri",
        "remote_uri",
        "manifest",
        "expected_count",
        "frame_count",
        "conditions",
    }
    if set(payload) != required:
        raise ValueError("factorial profile fields are not exact")
    if strict:
        expected_header = {
            "run_name": "vqa-factorial-full-20260809-f1",
            "input_uri": (
                "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
                "experiments/vqa-factorial-full-evidence-20260809-f1"
            ),
            "remote_uri": (
                "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
                "experiments/vqa-factorial-full-20260809-f1/conditions"
            ),
            "manifest": "/dstack/run/configs/vqa/qwen35_vqa2025_475_label_free.json",
            "expected_count": 475,
            "frame_count": 12,
        }
        for field_name, expected in expected_header.items():
            if payload.get(field_name) != expected:
                raise ValueError(f"factorial profile {field_name} is not pinned")
    if not isinstance(payload["conditions"], list):
        raise ValueError("factorial profile conditions must be a list")
    if tuple(
        condition.get("condition_id")
        for condition in payload["conditions"]
        if isinstance(condition, Mapping)
    ) != EXPECTED_CONDITION_IDS:
        raise ValueError("factorial profile must declare the four condition IDs in order")
    if len(payload["conditions"]) != len(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial profile must declare exactly four conditions")
    for condition_id, condition in zip(EXPECTED_CONDITION_IDS, payload["conditions"], strict=True):
        if not isinstance(condition, Mapping):
            raise ValueError("factorial profile conditions must be objects")
        if set(condition) != {
            "condition_id",
            "model_id",
            "revision",
            "answer_policy",
            "min_pixels",
            "max_pixels",
            "request_source",
        }:
            raise ValueError(f"condition {condition_id} fields are not exact")
        actual = (
            condition["model_id"],
            condition["revision"],
            condition["answer_policy"],
            condition["min_pixels"],
            condition["max_pixels"],
            condition["request_source"],
        )
        if actual != EXPECTED_CONDITION_BINDINGS[condition_id]:
            raise ValueError(f"condition {condition_id} model/policy binding mismatch")
    if isinstance(payload["expected_count"], bool) or int(payload["expected_count"]) <= 0:
        raise ValueError("factorial profile expected_count must be positive")
    if payload["frame_count"] != EXPECTED_RUNTIME_CONTROLS["frame_count"]:
        raise ValueError("factorial profile frame_count is not pinned")
    return copy.deepcopy(dict(payload))


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("factorial profile must be an object")
    return validate_factorial_profile(payload, strict=path.resolve() == DEFAULT_PROFILE_PATH.resolve())


def _profile_conditions(profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(condition["condition_id"]): condition for condition in profile["conditions"]
    }


def _condition_tuple(condition: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        condition.get("model_id"),
        condition.get("revision"),
        condition.get("answer_policy"),
        condition.get("min_pixels"),
        condition.get("max_pixels"),
        condition.get("request_source"),
    )


def _validate_runtime_metadata(
    condition_id: str,
    metadata: Mapping[str, Any],
    profile_condition: Mapping[str, Any],
) -> None:
    expected = {
        "condition_id": condition_id,
        "model_id": profile_condition["model_id"],
        "revision": profile_condition["revision"],
        "model_revision": profile_condition["revision"],
        "answer_policy": profile_condition["answer_policy"],
        "min_pixels": profile_condition["min_pixels"],
        "max_pixels": profile_condition["max_pixels"],
        "request_source": profile_condition["request_source"],
        **EXPECTED_RUNTIME_CONTROLS,
    }
    aliases = {"model_revision": "revision"}
    for raw_key, value in metadata.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key in expected and value != expected[key]:
            raise ValueError(f"condition {condition_id} runtime/model binding mismatch: {raw_key}")


def _reject_label_fields(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _LABEL_KEYS:
                raise ValueError(f"Prediction contains label-bearing field at {location}.{key}")
            _reject_label_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_label_fields(child, f"{location}[{index}]")


def _load_prediction_payload(path: Path) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
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
    _reject_label_fields(payload)
    metadata: Mapping[str, Any] = {}
    if isinstance(payload, Mapping):
        rows = payload.get("predictions", payload.get("rows"))
        metadata_parts = []
        for key in ("metadata", "condition", "runtime"):
            part = payload.get(key)
            if part is not None:
                if not isinstance(part, Mapping):
                    raise ValueError("prediction metadata must be an object")
                metadata_parts.append(part)
        merged_metadata: dict[str, Any] = {}
        for part in metadata_parts:
            merged_metadata.update(part)
        metadata = merged_metadata
        payload = rows
    if not isinstance(payload, list) or not payload or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"Predictions must be a non-empty list of objects: {path}")
    return [dict(row) for row in payload], metadata


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_id: str,
    profile_condition: Mapping[str, Any],
    expected_rows: int,
) -> list[dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(
            f"Condition {condition_id} has {len(rows)} prediction rows; expected {expected_rows}"
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
        _validate_runtime_metadata(condition_id, raw, profile_condition)
        answer = raw.get("answer")
        if raw.get("status") != "passed" or not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Condition {condition_id} is incomplete at {video_id}; completed {len(selected)}/{expected_rows}"
            )
        selected.append({"condition_id": condition_id, "video_id": video_id, "answer": answer})
    return selected


def _normalization_map(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"video_id": str(row["video_id"]), "answer": normalize_nist_text(str(row["answer"]))}
        for row in sorted(rows, key=lambda row: str(row["video_id"]))
    ]


def normalized_answer_map_sha256(value: Path | Sequence[Mapping[str, Any]]) -> str:
    if isinstance(value, Path):
        rows, _metadata = _load_prediction_payload(value)
    else:
        rows = list(value)
    return hashlib.sha256(_canonical_json(_normalization_map(rows))).hexdigest()


def _cache_identity(profile_digest: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "scorer": "vqa-factorial-semantic-v2",
                "profile_sha256": profile_digest,
                "normalization": "strip, lowercase, collapse whitespace",
                "endpoint": STS_ENDPOINT,
                "timeout_seconds": EXPECTED_STS_TIMEOUT,
                "max_attempts": EXPECTED_STS_RETRIES,
            }
        )
    ).hexdigest()


def _sts_pair_key(identity: str, prediction: str, reference: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "identity": identity,
                "prediction": prediction,
                "reference": reference,
            }
        )
    ).hexdigest()


def _mean(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    if not rows:
        raise ValueError("Cannot aggregate an empty row set")
    values = [_finite(row[field_name], field=field_name) for row in rows]
    return statistics.fmean(values)


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]], *, include_strata: bool = True
) -> dict[str, Any]:
    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["stratum"])].append(row)
    result: dict[str, Any] = {
        "row_count": len(rows),
        "completed_count": len(rows),
        "completion_rate": 1.0 if rows else 0.0,
        "mean_answer_token_count": statistics.fmean(row["answer_token_count"] for row in rows),
        "ndcg": _mean(rows, "ndcg"),
        "ndcg_degenerate": all(bool(row["ndcg_degenerate"]) for row in rows),
    }
    for field_name in ("normalized_exact_match", "token_f1", *METRIC_FIELDS):
        result[field_name] = _mean(rows, field_name)
    if include_strata:
        result["strata"] = {
            name: _aggregate_rows(group, include_strata=False)
            for name, group in sorted(strata.items())
        }
    return result


def _bootstrap_intervals(
    vectors: Mapping[str, Sequence[float]],
    *,
    iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    seed: int = EXPECTED_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    if seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"Bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if iterations <= 0 or not 0 < confidence < 1:
        raise ValueError("Bootstrap configuration is invalid")
    lengths = {len(values) for values in vectors.values()}
    if not lengths or lengths == {0} or len(lengths) != 1:
        raise ValueError("Bootstrap vectors must have the same non-zero length")
    size = next(iter(lengths))
    source = random.Random(seed)
    sampled: dict[str, list[float]] = {field_name: [] for field_name in vectors}
    for _ in range(iterations):
        indexes = source.choices(range(size), k=size)
        for field_name, values in vectors.items():
            sampled[field_name].append(statistics.fmean(values[index] for index in indexes))
    tail = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, Any]] = {}
    for field_name, values in sampled.items():
        ordered = sorted(values)
        lower_index = min(iterations - 1, int(tail * iterations))
        upper_index = min(iterations - 1, max(0, int((1.0 - tail) * iterations)))
        result[field_name] = {
            "lower": ordered[lower_index],
            "upper": ordered[upper_index],
            "confidence": confidence,
            "iterations": iterations,
            "seed": seed,
            "exploratory": True,
        }
    return result


def _effect_rows(condition_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_condition = {
        condition_id: {str(row["video_id"]): row for row in rows}
        for condition_id, rows in condition_rows.items()
    }
    if set(by_condition) != set(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial effects require the four exact conditions")
    ids = set(by_condition[EXPECTED_CONDITION_IDS[0]])
    if any(set(by_condition[condition_id]) != ids for condition_id in EXPECTED_CONDITION_IDS):
        raise ValueError("factorial effect rows must have one aligned ID set")
    effects = {"model": [], "prompt": [], "interaction": []}
    for video_id in sorted(ids):
        rows = {condition_id: by_condition[condition_id][video_id] for condition_id in EXPECTED_CONDITION_IDS}
        stratum = rows["4b-current"].get("stratum", rows["4b-current"].get("question_stratum"))
        for condition_id, row in rows.items():
            if row.get("stratum", row.get("question_stratum")) != stratum:
                raise ValueError(f"factorial strata do not align for {video_id}/{condition_id}")
        for effect_name in effects:
            effect_row: dict[str, Any] = {
                "video_id": video_id,
                "stratum": stratum,
            }
            for field_name in METRIC_FIELDS:
                current = rows["4b-current"][field_name]
                question_4b = rows["4b-question-aware"][field_name]
                current_8b = rows["8b-current"][field_name]
                question_8b = rows["8b-question-aware"][field_name]
                if effect_name == "model":
                    value = ((current_8b + question_8b) / 2) - ((current + question_4b) / 2)
                elif effect_name == "prompt":
                    value = ((question_4b + question_8b) / 2) - ((current + current_8b) / 2)
                else:
                    value = question_8b - current_8b - question_4b + current
                effect_row[field_name] = _finite(value, field=field_name)
            effects[effect_name].append(effect_row)
    return effects


def _effect_summary(rows: Sequence[Mapping[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    summary = {field_name: statistics.fmean(row[field_name] for row in rows) for field_name in METRIC_FIELDS}
    summary.update(
        {
            "paired_count": len(rows),
            "bootstrap_intervals": _bootstrap_intervals(
                {field_name: [row[field_name] for row in rows] for field_name in METRIC_FIELDS},
                iterations=iterations,
                seed=seed,
            ),
        }
    )
    return summary


def compute_factorial_effects(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    bootstrap_iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    effects = _effect_rows(condition_rows)
    result: dict[str, Any] = {}
    for effect_name, rows in effects.items():
        result[effect_name] = {
            "overall": _effect_summary(rows, iterations=bootstrap_iterations, seed=bootstrap_seed),
            "strata": {
                stratum: _effect_summary(
                    [row for row in rows if row["stratum"] == stratum],
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                for stratum in sorted({str(row["stratum"]) for row in rows})
            },
        }
    return result


def _paired_rows(
    baseline_rows: Sequence[Mapping[str, Any]], treatment_rows: Sequence[Mapping[str, Any]], condition_id: str
) -> list[dict[str, Any]]:
    baseline = {str(row["video_id"]): row for row in baseline_rows}
    treatment = {str(row["video_id"]): row for row in treatment_rows}
    if set(baseline) != set(treatment):
        raise ValueError("paired conditions do not have identical video ID sets")
    result = []
    for video_id in sorted(baseline):
        if baseline[video_id]["stratum"] != treatment[video_id]["stratum"]:
            raise ValueError(f"paired strata differ for {video_id}")
        result.append(
            {
                "condition_id": condition_id,
                "video_id": video_id,
                "stratum": baseline[video_id]["stratum"],
                **{
                    field_name: treatment[video_id][field_name] - baseline[video_id][field_name]
                    for field_name in METRIC_FIELDS
                },
            }
        )
    return result


def _paired_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    treatment_id: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = _paired_rows(baseline_rows, treatment_rows, treatment_id)
    return {
        "paired_count": len(rows),
        "metric_deltas": {field_name: statistics.fmean(row[field_name] for row in rows) for field_name in METRIC_FIELDS},
        "bootstrap_intervals": _bootstrap_intervals(
            {field_name: [row[field_name] for row in rows] for field_name in METRIC_FIELDS},
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "rows": rows,
        "ndcg_degenerate": True,
        "excluded_metrics": {"ndcg": "single-response value is degenerate"},
    }


def promotion_decision(values: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "delta_C>=0.015": float(values["delta_C"]) >= 0.015,
        "lower_CI_C>0": float(values["lower_CI_C"]) > 0,
        "delta_METEOR>0": float(values["delta_METEOR"]) > 0,
        "delta_BERTScore>0": float(values["delta_BERTScore"]) > 0,
        "delta_STS>=-0.005": float(values["delta_STS"]) >= -0.005,
        "completed=475": values["completed"] == 475,
        "integrity": values["integrity"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "promote" if not failed else "stop",
        "promote": not failed,
        "failed_gates": failed,
        "gates": checks,
    }


def _sanitized_row(row: Mapping[str, Any], *, condition_id: str | None = None) -> dict[str, Any]:
    field_names = (
        "condition_id",
        "video_id",
        "stratum",
        "normalized_exact_match",
        "token_f1",
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
        "ndcg_degenerate",
    )
    result = {field_name: row[field_name] for field_name in field_names}
    if condition_id is not None:
        result["condition_id"] = condition_id
    for field_name in field_names:
        if field_name not in {"condition_id", "video_id", "stratum", "ndcg_degenerate"}:
            result[field_name] = _finite(result[field_name], field=field_name)
    if not isinstance(result["video_id"], str) or not result["video_id"]:
        raise ValueError("sanitized row has an invalid video ID")
    return result


def _baseline_sanitized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read the older Task 1 row schema without importing any text."""

    stratum = row.get("stratum", row.get("question_stratum"))
    if not isinstance(stratum, str) or not stratum:
        raise ValueError("baseline row has no stratum")
    result = {
        "condition_id": "4b-current",
        "video_id": row["video_id"],
        "stratum": stratum,
        "normalized_exact_match": row.get("normalized_exact_match"),
        "token_f1": row.get("token_f1"),
        "answer_token_count": row["answer_token_count"],
        "meteor": row["meteor"],
        "bertscore_f1": row["bertscore_f1"],
        "sts": row["sts"],
        "semantic_composite": row["semantic_composite"],
        "ndcg": row["ndcg"],
        "ndcg_degenerate": row["ndcg_degenerate"],
    }
    for field_name in (
        "answer_token_count",
        "meteor",
        "bertscore_f1",
        "sts",
        "semantic_composite",
        "ndcg",
    ):
        result[field_name] = _finite(result[field_name], field=field_name)
    if not isinstance(result["video_id"], str) or not result["video_id"]:
        raise ValueError("baseline row has an invalid video ID")
    return result


def _sanitize_rows(condition_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        condition_id: [_sanitized_row(row, condition_id=condition_id) for row in rows]
        for condition_id, rows in condition_rows.items()
    }


def _baseline_scorer_binding(scorer_payload: Mapping[str, Any]) -> bool:
    required = {
        "normalization": "strip, lowercase, collapse whitespace",
        "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
        "bert_score_module_version": EXPECTED_BERT_SCORE_MODULE_VERSION,
        "bert_num_layers": 17,
        "bert_idf": False,
        "bert_rescale_with_baseline": False,
        "sts_endpoint": STS_ENDPOINT,
        "sts_timeout_seconds": EXPECTED_STS_TIMEOUT,
        "sts_max_attempts": EXPECTED_STS_RETRIES,
    }
    return all(scorer_payload.get(field_name) == expected for field_name, expected in required.items())


def baseline_content_sha256(payload: Mapping[str, Any]) -> str:
    candidate = copy.deepcopy(dict(payload))
    inputs = candidate.get("inputs")
    if isinstance(inputs, Mapping):
        inputs = dict(inputs)
        inputs.pop("artifact_content_sha256", None)
        candidate["inputs"] = inputs
    return hashlib.sha256(_canonical_json(candidate)).hexdigest()


def _validate_baseline_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path | None,
    panel_identity: Mapping[str, Any],
    annotation_hash: str,
    bert_model: Path,
    profile_digest: str,
    current_answer_map_hash: str,
    expected_rows: int,
    expected_ids: Sequence[str],
) -> tuple[bool, str, list[dict[str, Any]] | None]:
    checks: list[tuple[str, bool]] = []
    checks.append(("schema", payload.get("schema_version") == 1))
    checks.append(("panel", payload.get("panel", {}).get("identity") == panel_identity))
    inputs = payload.get("inputs", {})
    checks.extend(
        [
            ("annotation", inputs.get("annotation_root_sha256") == annotation_hash),
            ("bert snapshot", inputs.get("bert_model_snapshot") == EXPECTED_BERT_MODEL_SNAPSHOT),
            (
                "bert hash",
                inputs.get("bert_model_snapshot_sha256") == sha256_path(bert_model),
            ),
            ("scorer", inputs.get("scorer_sha256") == sha256_path(TASK1_SCORER_PATH)),
            (
                "config",
                inputs.get("profile_sha256", inputs.get("config_sha256")) == profile_digest,
            ),
            (
                "answer-map",
                inputs.get("normalized_answer_map_sha256") == current_answer_map_hash,
            ),
            ("scorer configuration", _baseline_scorer_binding(payload.get("scorer", {}))),
            (
                "artifact hash",
                inputs.get("artifact_content_sha256") == baseline_content_sha256(payload),
            ),
        ]
    )
    if "artifact_sha256" in inputs and artifact_path is not None:
        checks.append(("artifact file hash", inputs["artifact_sha256"] == sha256_path(artifact_path)))
    if not all(passed for _name, passed in checks):
        failed = next(name for name, passed in checks if not passed)
        return False, f"binding mismatch: {failed}", None
    rows_payload = payload.get("rows")
    if not isinstance(rows_payload, Mapping):
        return False, "binding mismatch: baseline rows", None
    candidate_rows: Sequence[Mapping[str, Any]] | None = None
    for key, rows in rows_payload.items():
        if key == "4b-current" or key not in EXPECTED_CONDITION_IDS:
            if isinstance(rows, list) and len(rows) == expected_rows:
                candidate_rows = rows
                break
    if candidate_rows is None:
        return False, "binding mismatch: baseline 4b-current rows", None
    try:
        sanitized = [_baseline_sanitized_row(row) for row in candidate_rows]
    except (KeyError, TypeError, ValueError):
        return False, "binding mismatch: baseline row schema", None
    by_video_id = {row["video_id"]: row for row in sanitized}
    if len(by_video_id) != expected_rows or set(by_video_id) != set(expected_ids):
        return False, "binding mismatch: baseline row ID set", None
    return True, "all bindings matched", [by_video_id[video_id] for video_id in expected_ids]


def _score_unique_pairs(
    condition_predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    annotations: Mapping[str, Mapping[str, Any]],
    *,
    bert_model: Path,
    meteor_fn: MetricFunction,
    bertscore_fn: BertScoreFunction,
    sts_fn: MetricFunction,
    sts_cache: STSCache,
    sts_identity: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    pairs: set[tuple[str, str]] = set()
    normalized_by_condition: dict[str, list[tuple[str, str, str]]] = {}
    for condition_id, predictions in condition_predictions.items():
        entries = []
        for prediction in predictions:
            video_id = str(prediction["video_id"])
            if video_id not in annotations:
                raise ValueError(f"Missing reference: prediction ID not present in declared annotation set: {video_id}")
            normalized_prediction = normalize_nist_text(str(prediction["answer"]))
            normalized_reference = normalize_nist_text(str(annotations[video_id]["reference"]))
            pair = (normalized_prediction, normalized_reference)
            pairs.add(pair)
            entries.append((video_id, normalized_prediction, normalized_reference))
        normalized_by_condition[condition_id] = entries
    ordered_pairs = sorted(pairs)
    meteor_values: dict[tuple[str, str], float] = {}
    for prediction, reference in ordered_pairs:
        meteor_values[(prediction, reference)] = _finite(
            meteor_fn(prediction, reference), field="meteor"
        )
    bert_predictions = [pair[0] for pair in ordered_pairs]
    bert_references = [pair[1] for pair in ordered_pairs]
    bert_values = list(bertscore_fn(bert_predictions, bert_references, bert_model))
    if len(bert_values) != len(ordered_pairs):
        raise ValueError(f"BERTScore returned {len(bert_values)} rows for {len(ordered_pairs)} unique pairs")
    bert_map = {
        pair: _finite(value, field="bertscore_f1")
        for pair, value in zip(ordered_pairs, bert_values, strict=True)
    }
    sts_values: dict[tuple[str, str], float] = {}
    hit_count = 0
    miss_count = 0
    live_request_count = 0
    for prediction, reference in ordered_pairs:
        key = _sts_pair_key(sts_identity, prediction, reference)
        if key in sts_cache.values:
            value = _finite(sts_cache.values[key], field="sts")
            hit_count += 1
        else:
            miss_count += 1
            live_request_count += 1
            value = _finite(sts_fn(prediction, reference), field="sts")
            sts_cache.values[key] = value
        sts_values[(prediction, reference)] = value
    rows: dict[str, list[dict[str, Any]]] = {}
    for condition_id, entries in normalized_by_condition.items():
        condition_rows = []
        for video_id, prediction, reference in entries:
            annotation = annotations[video_id]
            metrics = {
                "meteor": meteor_values[(prediction, reference)],
                "bertscore_f1": bert_map[(prediction, reference)],
                "sts": sts_values[(prediction, reference)],
            }
            composite = semantic_composite(metrics)
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "video_id": video_id,
                    "stratum": annotation["question_stratum"],
                    "normalized_exact_match": normalized_exact_match(prediction, reference),
                    "token_f1": token_f1(prediction, reference),
                    "answer_token_count": len(prediction.split()),
                    **metrics,
                    "semantic_composite": composite,
                    "ndcg": 1.0,
                    "ndcg_degenerate": True,
                }
            )
        rows[condition_id] = condition_rows
    return rows, {
        "unique_pair_count": len(ordered_pairs),
        "meteor_pair_count": len(ordered_pairs),
        "bertscore_pair_count": len(ordered_pairs),
        "sts_pair_count": len(ordered_pairs),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "live_request_count": live_request_count,
    }


def _decision_for_factorial(
    condition_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    paired: Mapping[str, Mapping[str, Any]],
    effects: Mapping[str, Any],
    *,
    integrity: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    treatments: dict[str, Any] = {}
    for condition_id in EXPECTED_CONDITION_IDS[1:]:
        summary = paired[condition_id]
        deltas = summary["metric_deltas"]
        decision = promotion_decision(
            {
                "delta_C": deltas["semantic_composite"],
                "lower_CI_C": summary["bootstrap_intervals"]["semantic_composite"]["lower"],
                "delta_METEOR": deltas["meteor"],
                "delta_BERTScore": deltas["bertscore_f1"],
                "delta_STS": deltas["sts"],
                "completed": len(condition_rows[condition_id]),
                "integrity": integrity,
            }
        )
        treatments[condition_id] = {
            **decision,
            "delta_C": deltas["semantic_composite"],
            "lower_CI_C": summary["bootstrap_intervals"]["semantic_composite"]["lower"],
            "delta_METEOR": deltas["meteor"],
            "delta_BERTScore": deltas["bertscore_f1"],
            "delta_STS": deltas["sts"],
            "completed": len(condition_rows[condition_id]),
        }
    model_rows = _effect_rows(condition_rows)["model"]
    prompt_rows = _effect_rows(condition_rows)["prompt"]
    comparison = _bootstrap_intervals(
        {
            "model_minus_prompt": [
                model_row["semantic_composite"] - prompt_row["semantic_composite"]
                for model_row, prompt_row in zip(model_rows, prompt_rows, strict=True)
            ],
            "prompt_minus_model": [
                prompt_row["semantic_composite"] - model_row["semantic_composite"]
                for model_row, prompt_row in zip(model_rows, prompt_rows, strict=True)
            ],
        },
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    model_effect = effects["model"]["overall"]["semantic_composite"]
    prompt_effect = effects["prompt"]["overall"]["semantic_composite"]
    if model_effect >= 0.020 and comparison["model_minus_prompt"]["lower"] > 0:
        dominance = "model"
    elif prompt_effect >= 0.020 and comparison["prompt_minus_model"]["lower"] > 0:
        dominance = "prompt"
    else:
        dominance = "mixed_bottleneck"
    return {
        "treatments": treatments,
        "dominance": dominance,
        "dominance_threshold": 0.020,
        "dominance_comparison": comparison,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "injected-or-runtime-provided"


def _artifact_safe(value: object, location: str = "$") -> None:
    forbidden_keys = {
        "answer",
        "reference",
        "question",
        "transcript",
        "normalized_answer",
        "normalized_reference",
        "cache_path",
        "raw_prediction_path",
        "raw_absolute_path",
    }
    if isinstance(value, Mapping):
        if forbidden_keys.intersection(str(key) for key in value):
            raise ValueError(f"sanitized artifact contains forbidden text field at {location}")
        for key, child in value.items():
            _artifact_safe(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _artifact_safe(child, f"{location}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"sanitized artifact contains an absolute path at {location}")


def score_factorial(
    *,
    annotation_root: Path,
    condition_paths: Mapping[str, Path],
    bert_model: Path,
    profile_path: Path | None = DEFAULT_PROFILE_PATH,
    profile: Mapping[str, Any] | None = None,
    panel_profile: PanelProfile | None = None,
    expected_rows: int | None = None,
    expected_annotation_files: int | None = None,
    baseline_artifact_path: Path | None = None,
    rescore_baseline: bool = False,
    output_path: Path | None = None,
    meteor_fn: MetricFunction | None = None,
    bertscore_fn: BertScoreFunction | None = None,
    sts_fn: MetricFunction | None = None,
    sts_cache: STSCache | None = None,
    bootstrap_iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, score, summarize, and optionally atomically publish the factorial."""

    if output_path is not None and output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if bootstrap_seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"Bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if bootstrap_iterations != EXPECTED_BOOTSTRAP_ITERATIONS:
        raise ValueError(f"Bootstrap iterations must be {EXPECTED_BOOTSTRAP_ITERATIONS}")
    if not bert_model.is_dir() or bert_model.name != EXPECTED_BERT_MODEL_SNAPSHOT:
        raise ValueError(f"BERT model must be the pinned snapshot directory {EXPECTED_BERT_MODEL_SNAPSHOT}")
    if profile is None:
        if profile_path is None:
            profile_path = DEFAULT_PROFILE_PATH
        resolved_profile = load_profile(profile_path)
    else:
        resolved_profile = validate_factorial_profile(profile, strict=False)
    profile_digest = profile_sha256(resolved_profile)
    profile_conditions = _profile_conditions(resolved_profile)
    rows_expected = int(expected_rows if expected_rows is not None else resolved_profile["expected_count"])
    files_expected = int(
        expected_annotation_files
        if expected_annotation_files is not None
        else (panel_profile.expected_annotation_files if panel_profile else 481)
    )
    if rows_expected != int(resolved_profile["expected_count"]):
        raise ValueError("expected_rows does not match factorial profile")
    if set(condition_paths) != set(EXPECTED_CONDITION_IDS) or len(condition_paths) != len(EXPECTED_CONDITION_IDS):
        raise ValueError("factorial scoring requires exactly the four condition IDs")
    if any(not isinstance(path, Path) for path in condition_paths.values()):
        condition_paths = {condition_id: Path(path) for condition_id, path in condition_paths.items()}
    # Freeze and validate all prediction bytes before loading any annotation labels.
    frozen_predictions: dict[str, list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    frozen_order: list[str] | None = None
    for condition_id in EXPECTED_CONDITION_IDS:
        path = Path(condition_paths[condition_id])
        raw_rows, metadata = _load_prediction_payload(path)
        _validate_runtime_metadata(condition_id, metadata, profile_conditions[condition_id])
        selected = _validate_prediction_rows(
            raw_rows,
            condition_id=condition_id,
            profile_condition=profile_conditions[condition_id],
            expected_rows=rows_expected,
        )
        current_order = [row["video_id"] for row in selected]
        if frozen_order is None:
            frozen_order = current_order
        elif current_order != frozen_order:
            raise ValueError("Conditions do not have the same frozen prediction ID set/order")
        frozen_predictions[condition_id] = selected
        prediction_hashes[condition_id] = sha256_path(path)
    if frozen_order is None or len(frozen_order) != rows_expected:
        raise ValueError("No complete frozen factorial prediction panel")

    resolved_panel = panel_profile or CANONICAL_PANEL_PROFILE
    if rows_expected != resolved_panel.expected_rows or files_expected != resolved_panel.expected_annotation_files:
        raise ValueError("panel profile dimensions do not match the declared factorial panel")
    annotations = _load_annotations(annotation_root)
    if len(annotations) != files_expected:
        raise ValueError(f"Annotation root contains {len(annotations)} annotation files; expected {files_expected}")
    panel_identity = _validate_panel_identity(
        resolved_panel,
        annotation_ids=annotations,
        condition_ids={condition_id: (row["video_id"] for row in rows) for condition_id, rows in frozen_predictions.items()},
    )
    annotation_hash = sha256_path(annotation_root)
    bert_score_versions = task1._bert_score_versions(required=bertscore_fn is None)
    if bert_score_versions["module"] != EXPECTED_BERT_SCORE_MODULE_VERSION and bertscore_fn is None:
        raise ValueError("BERTScore module version is not pinned")
    meteor_fn = meteor_fn or task1._default_meteor
    bertscore_fn = bertscore_fn or task1._default_bertscore
    sts_fn = sts_fn or (lambda prediction, reference: umbc_sts_score(
        prediction,
        reference,
        timeout=EXPECTED_STS_TIMEOUT,
        retries=EXPECTED_STS_RETRIES,
    ))
    cache = sts_cache or STSCache()
    identity = _cache_identity(profile_digest)
    if cache.identity is None:
        cache.identity = identity
    elif cache.identity != identity:
        raise ValueError("STS cache identity does not match scorer/config/endpoint identity")
    current_answer_map_hash = normalized_answer_map_sha256(frozen_predictions["4b-current"])

    baseline_binding: dict[str, Any] = {
        "requested": baseline_artifact_path is not None,
        "reused": False,
        "reason": "not requested" if baseline_artifact_path is None else "rescore requested",
    }
    baseline_rows: list[dict[str, Any]] | None = None
    baseline_payload: Mapping[str, Any] | None = None
    baseline_hash: str | None = None
    if baseline_artifact_path is not None and not rescore_baseline:
        try:
            baseline_payload = _load_json(baseline_artifact_path)
            if not isinstance(baseline_payload, Mapping):
                raise ValueError("baseline payload is not an object")
            baseline_hash = sha256_path(baseline_artifact_path)
            valid, reason, baseline_rows = _validate_baseline_binding(
                baseline_payload,
                artifact_path=baseline_artifact_path,
                panel_identity=panel_identity,
                annotation_hash=annotation_hash,
                bert_model=bert_model,
                profile_digest=profile_digest,
                current_answer_map_hash=current_answer_map_hash,
                expected_rows=rows_expected,
                expected_ids=frozen_order,
            )
            baseline_binding.update({"reused": valid, "reason": reason, "artifact_sha256": baseline_hash})
        except (OSError, ValueError, KeyError, TypeError) as error:
            baseline_binding.update({"reused": False, "reason": f"binding mismatch: {type(error).__name__}"})
    elif baseline_artifact_path is not None:
        baseline_binding.update({"reason": "forced rescore (--rescore-baseline)"})
    condition_predictions = {
        condition_id: frozen_predictions[condition_id]
        for condition_id in EXPECTED_CONDITION_IDS
        if condition_id != "4b-current" or baseline_rows is None
    }
    scored_rows, cache_counts = _score_unique_pairs(
        condition_predictions,
        annotations,
        bert_model=bert_model,
        meteor_fn=meteor_fn,
        bertscore_fn=bertscore_fn,
        sts_fn=sts_fn,
        sts_cache=cache,
        sts_identity=identity,
    )
    if baseline_rows is not None:
        answers_by_id = {
            row["video_id"]: normalize_nist_text(row["answer"])
            for row in frozen_predictions["4b-current"]
        }
        scored_rows["4b-current"] = [
            {
                **row,
                "stratum": annotations[row["video_id"]]["question_stratum"],
                "answer_token_count": len(answers_by_id[row["video_id"]].split()),
                "normalized_exact_match": normalized_exact_match(
                    answers_by_id[row["video_id"]],
                    normalize_nist_text(annotations[row["video_id"]]["reference"]),
                ),
                "token_f1": token_f1(
                    answers_by_id[row["video_id"]],
                    normalize_nist_text(annotations[row["video_id"]]["reference"]),
                ),
            }
            for row in baseline_rows
        ]
    condition_rows = {condition_id: scored_rows[condition_id] for condition_id in EXPECTED_CONDITION_IDS}
    if any(len(rows) != rows_expected for rows in condition_rows.values()):
        raise ValueError("factorial completion gate failed after scoring")
    safe_rows = _sanitize_rows(condition_rows)
    aggregate = {}
    for condition_id in EXPECTED_CONDITION_IDS:
        condition_report = _aggregate_rows(condition_rows[condition_id])
        condition_report["model_id"] = profile_conditions[condition_id]["model_id"]
        condition_report["revision"] = profile_conditions[condition_id]["revision"]
        condition_report["answer_policy"] = profile_conditions[condition_id]["answer_policy"]
        condition_report["min_pixels"] = profile_conditions[condition_id]["min_pixels"]
        condition_report["max_pixels"] = profile_conditions[condition_id]["max_pixels"]
        condition_report["request_source"] = profile_conditions[condition_id]["request_source"]
        condition_report["runtime_controls"] = EXPECTED_RUNTIME_CONTROLS
        aggregate[condition_id] = condition_report
    paired = {"baseline_condition_id": "4b-current"}
    for condition_id in EXPECTED_CONDITION_IDS[1:]:
        paired[condition_id] = _paired_summary(
            condition_rows["4b-current"],
            condition_rows[condition_id],
            treatment_id=condition_id,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
    effects = compute_factorial_effects(
        condition_rows,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    integrity = {
        "labels_joined_after_prediction_freeze": True,
        "all_conditions_complete": all(len(rows) == rows_expected for rows in condition_rows.values()),
        "completion_denominator": f"{rows_expected}/{rows_expected} per condition",
        "duplicate_prediction_ids_checked": True,
        "same_canonical_id_set_and_order_checked": True,
        "canonical_profile_digest_checked": True,
        "condition_bindings_checked": True,
        "finite_metrics_checked": True,
        "label_leakage_checked": True,
    }
    integrity_ok = all(value is True for key, value in integrity.items() if key != "completion_denominator")
    decision = _decision_for_factorial(
        condition_rows,
        paired,
        effects,
        integrity=integrity_ok,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    artifact: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "experiment": {
            "name": "vqa-factorial-475-20260809-f1",
            "design": "2x2 model-by-answer-policy factorial",
            "semantic_composite": "mean(meteor, bertscore_f1, sts)",
            "bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed, "confidence": 0.95, "exploratory": True},
        },
        "panel": {
            "collection": "declared annotation panel",
            "expected_annotation_files": files_expected,
            "expected_rows_per_condition": rows_expected,
            "declared_conditions": list(EXPECTED_CONDITION_IDS),
            "completed_per_condition": {condition_id: len(condition_rows[condition_id]) for condition_id in EXPECTED_CONDITION_IDS},
            "identity": panel_identity,
        },
        "scorer": {
            "name": "NIST 2025 semantic VQA factorial scorer",
            "normalization": "strip, lowercase, collapse whitespace",
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_score_module_version": bert_score_versions["module"],
            "bert_score_distribution_version": bert_score_versions["distribution"],
            "bert_num_layers": 17,
            "bert_idf": False,
            "bert_rescale_with_baseline": False,
            "sts_endpoint": STS_ENDPOINT,
            "sts_timeout_seconds": EXPECTED_STS_TIMEOUT,
            "sts_max_attempts": EXPECTED_STS_RETRIES,
            "single_response_ndcg": {"value": 1.0, "degenerate": True, "informative": False},
            "runtime_controls": EXPECTED_RUNTIME_CONTROLS,
        },
        "baseline_binding": baseline_binding,
        "conditions": aggregate,
        "rows": safe_rows,
        "paired_vs_baseline": paired,
        "factorial_effects": effects,
        "decision": decision,
        "integrity": integrity,
        "inputs": {
            "annotation_root_sha256": annotation_hash,
            "prediction_files": {condition_id: {"sha256": prediction_hashes[condition_id]} for condition_id in EXPECTED_CONDITION_IDS},
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_model_snapshot_sha256": sha256_path(bert_model),
            "profile_sha256": profile_digest,
            "runtime_controls_sha256": hashlib.sha256(_canonical_json(EXPECTED_RUNTIME_CONTROLS)).hexdigest(),
            "scorer_sha256": sha256_path(Path(__file__)),
            "task1_scorer_sha256": sha256_path(TASK1_SCORER_PATH),
        },
        "dependencies": {
            "nltk_distribution": _distribution_version("nltk"),
            "bert_score_module": bert_score_versions["module"],
            "bert_score_distribution": bert_score_versions["distribution"],
        },
        "cache_summary": {"identity": identity, **cache_counts, "private": True},
    }
    _artifact_safe(artifact)
    if output_path is not None:
        atomic_write_json(output_path, artifact)
    return artifact


def _parse_conditions(values: Sequence[str]) -> dict[str, Path]:
    conditions: dict[str, Path] = {}
    for value in values:
        condition_id, separator, path = value.partition("=")
        if not separator or not condition_id.strip() or not path.strip():
            raise ValueError(f"Condition must be ID=PATH: {value}")
        if condition_id.strip() in conditions:
            raise ValueError(f"Duplicate condition ID: {condition_id.strip()}")
        conditions[condition_id.strip()] = Path(path)
    return conditions


def _load_private_cache(path: Path, identity: str) -> STSCache:
    payload = _load_json(path) if path.exists() else {"identity": identity, "values": {}}
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values", {}), Mapping):
        raise ValueError("STS cache must be an object with private values")
    cache = STSCache(
        values={str(key): float(value) for key, value in payload["values"].items()},
        identity=payload.get("identity"),
    )
    if cache.identity != identity:
        raise ValueError("STS cache identity does not match scorer/config/endpoint identity")
    return cache


def _write_private_cache(path: Path, cache: STSCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"identity": cache.identity, "values": cache.values}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--condition", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--baseline-artifact", type=Path)
    parser.add_argument("--rescore-baseline", action="store_true")
    parser.add_argument("--sts-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = load_profile(args.profile)
    identity = _cache_identity(profile_sha256(profile))
    cache = _load_private_cache(args.sts_cache, identity) if args.sts_cache else STSCache(identity=identity)
    artifact = score_factorial(
        annotation_root=args.annotations,
        condition_paths=_parse_conditions(args.condition),
        bert_model=args.bert_model,
        profile=profile,
        baseline_artifact_path=args.baseline_artifact,
        rescore_baseline=args.rescore_baseline,
        sts_cache=cache,
    )
    if args.sts_cache:
        _write_private_cache(args.sts_cache, cache)
    atomic_write_json(args.output, artifact)
    print(json.dumps({"output": str(args.output), "completed_per_condition": artifact["panel"]["completed_per_condition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
