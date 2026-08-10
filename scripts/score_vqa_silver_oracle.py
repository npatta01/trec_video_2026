#!/usr/bin/env python3
"""Score the nine-arm VQA silver-oracle factorial and blind elections.

This entry point owns only the aggregate join after answer and judge freezes.
It reuses the pinned metric implementations and STS cache contract from the
Task 1/factorial scorer, while keeping answer, reference, candidate, and
ranking text in private in-memory tables.  The returned object is an
allowlisted aggregate artifact; callers may atomically publish it explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import random
import statistics
import string
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
EVALUATION_MODULE_PATH = SOURCE_ROOT / "trec_video_2026/vqa_evaluation.py"
for _path in (SCRIPTS_ROOT, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score_vqa_factorial as factorial  # noqa: E402
import build_vqa_silver_freeze_manifest as freeze_manifest  # noqa: E402
from trec_video_2026.vqa_evaluation import ndcg_from_gains  # noqa: E402


EXPECTED_SCHEMA_VERSION = 1
EXPECTED_BOOTSTRAP_SEED = 20260809
EXPECTED_BOOTSTRAP_ITERATIONS = 10_000
EXPECTED_CONFIDENCE = 0.95
EXPECTED_ARM_IDS = (
    "G/N",
    "G/F",
    "G/S",
    "H/N",
    "H/F",
    "H/S",
    "L/N",
    "L/F",
    "L/S",
)
EXPECTED_FACTORIAL_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
EXPECTED_POOL_IDS = ("A", "B", "C")
SEMANTIC_FIELDS = ("meteor", "bertscore_f1", "sts", "semantic_composite")
DIAGNOSTIC_FIELDS = ("normalized_exact_match", "token_f1")
PRIMARY_BASELINE = "G/F"
ELECTION_BASELINE = "4b-current"
STS_ENDPOINT = factorial.STS_ENDPOINT
EXPECTED_STS_TIMEOUT = factorial.EXPECTED_STS_TIMEOUT
EXPECTED_STS_RETRIES = factorial.EXPECTED_STS_RETRIES
EXPECTED_BERT_MODEL_SNAPSHOT = factorial.EXPECTED_BERT_MODEL_SNAPSHOT
EXPECTED_BERT_SCORE_MODULE_VERSION = factorial.EXPECTED_BERT_SCORE_MODULE_VERSION
EXPECTED_CANONICAL_PANEL_SHA256 = factorial.EXPECTED_FULL_CANONICAL_ID_DIGEST
EXPECTED_VERIFIED_4B_CURRENT_PREDICTION_SHA256 = (
    "973989e0e10cffb2a9f211ad754008a373f69171319a67bcf17d6bd9e9b8f623"
)
EXPECTED_VERIFIED_4B_CURRENT_SOURCE_PREDICTION_SHA256 = (
    factorial.EXPECTED_BASELINE_SOURCE_PREDICTION_SHA256
)
# PATCH_AFTER_REMOTE_FREEZE: primary replaces only the expected value with the
# SHA-256 of the canonical manifest emitted by build_vqa_silver_freeze_manifest.
# The nonzero sentinel keeps accidental all-zero pins out of production while
# making canonical scoring fail closed until that review/commit boundary.
CANONICAL_FREEZE_MANIFEST_SHA256_PLACEHOLDER = (
    "b09678855e9df7ba650c0984106d0f3a2e977a20d5398f3ada4defc70ceb82c9"
)
EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256 = CANONICAL_FREEZE_MANIFEST_SHA256_PLACEHOLDER
EXPECTED_ANSWER_MODEL_ID = "Qwen/Qwen3.5-4B"
EXPECTED_ANSWER_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_ANSWER_RUNTIME_IMAGE = factorial.EXPECTED_RUNTIME_CONTROLS["runtime_image"]
EXPECTED_TRANSFORMERS_REVISION = factorial.EXPECTED_RUNTIME_CONTROLS[
    "transformers_revision"
]
EXPECTED_JUDGE_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
EXPECTED_JUDGE_MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
EXPECTED_JUDGE_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@sha256:"
    "71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
EXPECTED_ARM_PRODUCER_IDS = {
    arm_id: arm_id.lower().replace("/", "") for arm_id in EXPECTED_ARM_IDS
}
EXPECTED_SOURCE_IDS = EXPECTED_FACTORIAL_IDS + EXPECTED_ARM_IDS
MetricFunction = Callable[[str, str], float]
BertScoreFunction = Callable[[list[str], list[str], Path], Sequence[float]]
STSCache = factorial.STSCache
umbc_sts_score = factorial.umbc_sts_score
normalize_nist_text = factorial.normalize_nist_text
sha256_path = factorial.sha256_path
_finite = factorial._finite
_load_annotations = factorial._load_annotations

ARM_ALIASES = {
    "gn": "G/N",
    "gf": "G/F",
    "gs": "G/S",
    "hn": "H/N",
    "hf": "H/F",
    "hs": "H/S",
    "ln": "L/N",
    "lf": "L/F",
    "ls": "L/S",
}
_FROZEN_INPUT_DIGESTS: dict[str, str] = {}


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_arm_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("arm ID must be a non-empty string")
    raw = value.strip()
    normalized = raw.lower().replace("_", "").replace("-", "").replace("/", "")
    canonical = ARM_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(f"unknown silver-oracle arm ID: {value}")
    return canonical


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _verify_manifest_trust_anchor(path: Path, *, expected_rows: int) -> str:
    """Verify the compile-time manifest anchor before any annotation access."""

    try:
        digest = _sha256_bytes(Path(path).read_bytes())
    except OSError as error:
        raise ValueError("cannot read the label-free freeze manifest") from error
    if expected_rows != 475:
        return digest
    expected = _require_sha256(
        EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256,
        field="canonical freeze-manifest pin",
    )
    if expected == "0" * 64:
        raise ValueError("canonical freeze-manifest pin must be nonzero")
    if expected == CANONICAL_FREEZE_MANIFEST_SHA256_PLACEHOLDER:
        raise ValueError(
            "canonical freeze-manifest pin is still the pre-freeze placeholder"
        )
    if digest != expected:
        raise ValueError(
            "freeze-manifest digest does not match the compile-time anchor"
        )
    return digest


def _bindings_from_freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project verified producer facts into the scorer's strict bindings."""

    prediction_sources = manifest["prediction_sources"]
    elections = manifest["elections"]
    evidence = manifest["evidence"]
    condition_fields = {
        "source_id",
        "condition_id",
        "prediction_sha256",
        "request_sha256",
        "model_id",
        "model_revision",
        "answer_policy",
        "min_pixels_per_frame",
        "max_pixels_per_frame",
        "runtime",
        "runtime_image",
        "transformers_revision",
        "do_sample",
        "enable_thinking",
        "max_new_tokens",
    }
    condition_bindings = {
        source_id: {
            field: prediction_sources[source_id][field] for field in condition_fields
        }
        for source_id in EXPECTED_SOURCE_IDS
    }
    election_fields = {
        "pool_id",
        "ranking_sha256",
        "request_sha256",
        "judge_model_id",
        "judge_model_revision",
        "runtime_image",
    }
    election_bindings = {
        pool_id: {field: elections[pool_id][field] for field in election_fields}
        for pool_id in EXPECTED_POOL_IDS
    }
    answer_freeze = {
        source_id: {
            "prediction_sha256": condition_bindings[source_id]["prediction_sha256"],
            "request_sha256": condition_bindings[source_id]["request_sha256"],
        }
        for source_id in EXPECTED_SOURCE_IDS
    }
    judge_freeze = {
        pool_id: {
            "ranking_sha256": election_bindings[pool_id]["ranking_sha256"],
            "request_sha256": election_bindings[pool_id]["request_sha256"],
        }
        for pool_id in EXPECTED_POOL_IDS
    }
    return {
        "schema_version": 1,
        "protocol_sha256": evidence["protocol_sha256"],
        "evidence_sha256": evidence["checksum_sha256"],
        "selector_sha256": evidence["selector_sha256"],
        "answer_freeze_sha256": _sha256_bytes(_canonical_json(answer_freeze)),
        "judge_freeze_sha256": _sha256_bytes(_canonical_json(judge_freeze)),
        "panel_id_set_sha256": manifest["panel_id_set_sha256"],
        "verified_4b_current_prediction_sha256": condition_bindings[ELECTION_BASELINE][
            "prediction_sha256"
        ],
        "verified_4b_current_source_prediction_sha256": manifest[
            "verified_4b_current_source_prediction_sha256"
        ],
        "conditions": condition_bindings,
        "elections": election_bindings,
    }


def _verify_label_free_freeze(
    *,
    freeze_manifest_path: Path,
    freeze_inputs: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    """Recheck every anchored label-free byte and derive trusted scorer inputs."""

    manifest_sha256 = _verify_manifest_trust_anchor(
        freeze_manifest_path, expected_rows=expected_rows
    )
    verified = freeze_manifest.verify_freeze_manifest(
        freeze_manifest_path,
        freeze_inputs,
        expected_rows=expected_rows,
        expected_manifest_sha256=(
            EXPECTED_CANONICAL_FREEZE_MANIFEST_SHA256 if expected_rows == 475 else None
        ),
    )
    coverage_path = verified["inputs"]["localized_coverage"]
    coverage_payload, coverage_sha256 = _read_frozen_file(coverage_path)
    if not isinstance(coverage_payload, Mapping):
        raise ValueError("anchored localized coverage must be a JSON object")
    if coverage_sha256 != verified["manifest"]["evidence"]["coverage_sha256"]:
        raise ValueError("localized coverage bytes changed after freeze verification")
    return {
        "manifest": verified["manifest"],
        "manifest_sha256": manifest_sha256,
        "bindings": _bindings_from_freeze_manifest(verified["manifest"]),
        "coverage": dict(coverage_payload),
    }


def _expected_condition_identity(source_id: str) -> dict[str, Any]:
    if source_id in EXPECTED_ARM_IDS:
        condition_id = EXPECTED_ARM_PRODUCER_IDS[source_id]
        model_id = EXPECTED_ANSWER_MODEL_ID
        model_revision = EXPECTED_ANSWER_MODEL_REVISION
        answer_policy = "current"
    else:
        try:
            (
                model_id,
                model_revision,
                answer_policy,
                _minimum,
                _maximum,
                _request_source,
            ) = factorial.EXPECTED_CONDITION_BINDINGS[source_id]
        except KeyError as error:
            raise ValueError(f"unknown frozen source binding: {source_id}") from error
        condition_id = source_id
    return {
        "source_id": source_id,
        "condition_id": condition_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "answer_policy": answer_policy,
        "min_pixels_per_frame": factorial.EXPECTED_RUNTIME_CONTROLS["min_pixels"],
        "max_pixels_per_frame": factorial.EXPECTED_RUNTIME_CONTROLS["max_pixels"],
        "runtime_image": EXPECTED_ANSWER_RUNTIME_IMAGE,
        "transformers_revision": EXPECTED_TRANSFORMERS_REVISION,
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": factorial.EXPECTED_RUNTIME_CONTROLS["max_new_tokens"],
    }


def _validate_frozen_bindings(
    bindings: Mapping[str, Any] | None, *, expected_rows: int
) -> dict[str, Any]:
    """Validate the independent source, request, runtime, and baseline freeze."""

    if not isinstance(bindings, Mapping):
        raise ValueError("explicit frozen provenance bindings are required")
    required_top = {
        "schema_version",
        "protocol_sha256",
        "evidence_sha256",
        "selector_sha256",
        "answer_freeze_sha256",
        "judge_freeze_sha256",
        "panel_id_set_sha256",
        "verified_4b_current_prediction_sha256",
        "verified_4b_current_source_prediction_sha256",
        "conditions",
        "elections",
    }
    if set(bindings) != required_top or bindings.get("schema_version") != 1:
        raise ValueError("frozen provenance binding schema is not exact")
    result = dict(bindings)
    for field in required_top - {"schema_version", "conditions", "elections"}:
        _require_sha256(result[field], field=f"binding {field}")

    conditions = result["conditions"]
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        EXPECTED_SOURCE_IDS
    ):
        raise ValueError("bindings require all exact condition IDs")
    condition_fields = {
        "source_id",
        "condition_id",
        "prediction_sha256",
        "request_sha256",
        "model_id",
        "model_revision",
        "answer_policy",
        "min_pixels_per_frame",
        "max_pixels_per_frame",
        "runtime",
        "runtime_image",
        "transformers_revision",
        "do_sample",
        "enable_thinking",
        "max_new_tokens",
    }
    normalized_conditions: dict[str, dict[str, Any]] = {}
    for source_id in EXPECTED_SOURCE_IDS:
        binding = conditions[source_id]
        if not isinstance(binding, Mapping) or set(binding) != condition_fields:
            raise ValueError(f"condition {source_id} binding schema is not exact")
        expected = _expected_condition_identity(source_id)
        if any(binding.get(field) != value for field, value in expected.items()):
            raise ValueError(f"condition {source_id} binding identity is not frozen")
        if (
            not isinstance(binding.get("runtime"), str)
            or not binding["runtime"].strip()
        ):
            raise ValueError(f"condition {source_id} runtime identity is invalid")
        _require_sha256(
            binding.get("prediction_sha256"),
            field=f"condition {source_id} prediction binding",
        )
        _require_sha256(
            binding.get("request_sha256"),
            field=f"condition {source_id} request binding",
        )
        normalized_conditions[source_id] = dict(binding)

    elections = result["elections"]
    if not isinstance(elections, Mapping) or set(elections) != set(EXPECTED_POOL_IDS):
        raise ValueError("bindings require exact Pool A/B/C identities")
    election_fields = {
        "pool_id",
        "ranking_sha256",
        "request_sha256",
        "judge_model_id",
        "judge_model_revision",
        "runtime_image",
    }
    normalized_elections: dict[str, dict[str, Any]] = {}
    for pool_id in EXPECTED_POOL_IDS:
        binding = elections[pool_id]
        if not isinstance(binding, Mapping) or set(binding) != election_fields:
            raise ValueError(f"Pool {pool_id} binding schema is not exact")
        expected = {
            "pool_id": pool_id,
            "judge_model_id": EXPECTED_JUDGE_MODEL_ID,
            "judge_model_revision": EXPECTED_JUDGE_MODEL_REVISION,
            "runtime_image": EXPECTED_JUDGE_RUNTIME_IMAGE,
        }
        if any(binding.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Pool {pool_id} binding identity is not frozen")
        _require_sha256(
            binding.get("ranking_sha256"),
            field=f"Pool {pool_id} ranking binding",
        )
        _require_sha256(
            binding.get("request_sha256"),
            field=f"Pool {pool_id} request binding",
        )
        normalized_elections[pool_id] = dict(binding)

    answer_freeze = {
        source_id: {
            "prediction_sha256": normalized_conditions[source_id]["prediction_sha256"],
            "request_sha256": normalized_conditions[source_id]["request_sha256"],
        }
        for source_id in EXPECTED_SOURCE_IDS
    }
    if _sha256_bytes(_canonical_json(answer_freeze)) != result["answer_freeze_sha256"]:
        raise ValueError("answer freeze digest does not match condition bindings")
    judge_freeze = {
        pool_id: {
            "ranking_sha256": normalized_elections[pool_id]["ranking_sha256"],
            "request_sha256": normalized_elections[pool_id]["request_sha256"],
        }
        for pool_id in EXPECTED_POOL_IDS
    }
    if _sha256_bytes(_canonical_json(judge_freeze)) != result["judge_freeze_sha256"]:
        raise ValueError("judge freeze digest does not match election bindings")
    current_digest = normalized_conditions[ELECTION_BASELINE]["prediction_sha256"]
    if result["verified_4b_current_prediction_sha256"] != current_digest:
        raise ValueError("verified 4b-current digest does not match its frozen input")
    if expected_rows == 475:
        if result["panel_id_set_sha256"] != EXPECTED_CANONICAL_PANEL_SHA256:
            raise ValueError("binding panel digest is not the canonical 475-row panel")
        if current_digest != EXPECTED_VERIFIED_4B_CURRENT_PREDICTION_SHA256:
            raise ValueError("4b-current prediction digest is not the verified input")
        if (
            result["verified_4b_current_source_prediction_sha256"]
            != EXPECTED_VERIFIED_4B_CURRENT_SOURCE_PREDICTION_SHA256
        ):
            raise ValueError("4b-current source digest is not the verified baseline")
    result["conditions"] = normalized_conditions
    result["elections"] = normalized_elections
    return result


def _reject_label_fields(value: Any, location: str = "$") -> None:
    """Reject label-bearing fields before any answer/reference join."""

    if isinstance(value, Mapping):
        forbidden = {
            "annotation",
            "annotations",
            "correctanswer",
            "expectedanswer",
            "groundtruth",
            "label",
            "labels",
            "reference",
            "referenceanswer",
            "referenceanswers",
            "question",
            "questions",
            "transcript",
            "transcripts",
            "prediction",
            "predictions",
            "answertext",
            "querytext",
            "cachequery",
        }
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in forbidden:
                # ``predictions`` is the frozen answer container and is safe at
                # the top level; individual prediction fields are still checked.
                if normalized == "predictions" and location == "$":
                    _reject_label_fields(child, f"{location}.{key}")
                    continue
                raise ValueError(
                    f"frozen input contains label-bearing field at {location}.{key}"
                )
            _reject_label_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_label_fields(child, f"{location}[{index}]")


def _load_json_bytes(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        try:
            return [
                json.loads(line)
                for line in raw.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("frozen input is not valid JSON or JSONL") from error


def _read_frozen_file(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError("cannot read frozen scorer input") from error
    return _load_json_bytes(raw), _sha256_bytes(raw)


def _payload_rows(
    payload: Any, *, kind: str
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    metadata: Mapping[str, Any] = {}
    if isinstance(payload, Mapping):
        metadata_parts = []
        for key in ("metadata", "provenance", "runtime"):
            part = payload.get(key)
            if part is not None:
                if not isinstance(part, Mapping):
                    raise ValueError(f"{kind} metadata must be an object")
                metadata_parts.append(part)
        merged: dict[str, Any] = {}
        for part in metadata_parts:
            merged.update(part)
        metadata = merged
        rows = payload.get("predictions", payload.get("rows"))
    else:
        rows = payload
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError(f"{kind} must contain a non-empty list of objects")
    return [dict(row) for row in rows], metadata


def _validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    expected_rows: int,
    binding: Mapping[str, Any],
) -> list[dict[str, str]]:
    if len(rows) != expected_rows:
        raise ValueError(
            f"{source_id} is incomplete: completed {len(rows)}/{expected_rows} rows"
        )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        factorial._reject_label_fields(raw, f"$[{index}]")
        computed_metric_keys = {
            "meteor",
            "bertscoref1",
            "sts",
            "semanticcomposite",
            "normalizedexactmatch",
            "tokenf1",
            "ndcg",
        }
        if any(
            re.sub(r"[^a-z0-9]", "", str(key).lower()) in computed_metric_keys
            for key in raw
        ):
            raise ValueError(f"{source_id} contains precomputed scorer fields")
        video_id = raw.get("video_id")
        answer = raw.get("answer")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"{source_id} has an invalid video ID")
        video_id = video_id.strip()
        if video_id in seen:
            raise ValueError(f"{source_id} has duplicate prediction ID: {video_id}")
        if (
            raw.get("status") != "passed"
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(f"{source_id} is incomplete at {video_id}")
        expected_identity = {
            "condition_id": binding["condition_id"],
            "model_id": binding["model_id"],
            "model_revision": binding["model_revision"],
            "answer_policy": binding["answer_policy"],
            "requested_min_pixels_per_frame": binding["min_pixels_per_frame"],
            "requested_max_pixels_per_frame": binding["max_pixels_per_frame"],
            "min_pixels_per_frame": binding["min_pixels_per_frame"],
            "max_pixels_per_frame": binding["max_pixels_per_frame"],
            "runtime": binding["runtime"],
            "runtime_image": binding["runtime_image"],
        }
        if any(raw.get(field) != value for field, value in expected_identity.items()):
            raise ValueError(f"{source_id} prediction condition identity is not bound")
        seen.add(video_id)
        result.append({"video_id": video_id, "answer": answer})
    return result


def _freeze_prediction_inputs(
    paths: Mapping[str, Path],
    *,
    expected_rows: int,
    source_kind: str,
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, str]]], dict[str, str], dict[str, Mapping[str, Any]]
]:
    frozen: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    frozen_order: list[str] | None = None
    for source_id, raw_path in paths.items():
        payload, digest = _read_frozen_file(Path(raw_path))
        if digest != bindings[source_id]["prediction_sha256"]:
            raise ValueError(
                f"{source_id} prediction bytes do not match the frozen binding"
            )
        _reject_label_fields(payload)
        rows, row_metadata = _payload_rows(payload, kind=source_kind)
        if row_metadata.get("labels_loaded_after_prediction_freeze") is False:
            raise ValueError(f"{source_id} metadata violates prediction freeze order")
        selected = _validate_prediction_rows(
            rows,
            source_id=source_id,
            expected_rows=expected_rows,
            binding=bindings[source_id],
        )
        order = [row["video_id"] for row in selected]
        if frozen_order is None:
            frozen_order = order
        elif order != frozen_order:
            raise ValueError(
                "all frozen prediction inputs must share one exact ID order"
            )
        frozen[source_id] = selected
        hashes[source_id] = digest
        metadata[source_id] = row_metadata
    if frozen_order is None:
        raise ValueError("no frozen prediction inputs were supplied")
    _FROZEN_INPUT_DIGESTS.clear()
    _FROZEN_INPUT_DIGESTS.update(hashes)
    return frozen, hashes, metadata


def _normalize_candidate_text(text: str) -> str:
    without_punctuation = text.translate(str.maketrans("", "", string.punctuation))
    return normalize_nist_text(without_punctuation)


def _candidate_entries(raw: Any, *, location: str) -> list[dict[str, Any]]:
    if isinstance(raw, Mapping):
        entries = []
        for candidate_id, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"{location} candidate values must be objects")
            if "candidate_id" in value:
                raise ValueError(f"{location} candidate IDs must not be duplicated")
            entries.append({"candidate_id": candidate_id, **dict(value)})
        return entries
    if isinstance(raw, list) and all(isinstance(item, Mapping) for item in raw):
        return [dict(item) for item in raw]
    raise ValueError(f"{location} candidates must be a list or object")


def _reject_election_metric_fields(value: Any, location: str) -> None:
    forbidden = {
        "reference",
        "question",
        "metric",
        "metrics",
        "score",
        "scores",
        "meteor",
        "bertscoref1",
        "sts",
        "semanticcomposite",
        "conditionid",
        "modelid",
        "modelrevision",
        "selector",
        "provenance",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in forbidden:
                raise ValueError(
                    f"election input contains forbidden field at {location}.{key}"
                )
            _reject_election_metric_fields(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_election_metric_fields(child, f"{location}[{index}]")


def _validate_election_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool_id: str,
    expected_rows: int,
    expected_order: Sequence[str],
    source_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[dict[str, Any]]:
    if len(rows) != expected_rows:
        raise ValueError(
            f"Pool {pool_id} is incomplete: completed {len(rows)}/{expected_rows}"
        )
    source_answers = {
        source_id: {row["video_id"]: row["answer"] for row in source}
        for source_id, source in source_rows.items()
    }
    expected_sources = {
        "A": set(EXPECTED_FACTORIAL_IDS),
        "B": set(EXPECTED_ARM_IDS),
        "C": set(EXPECTED_FACTORIAL_IDS) | set(EXPECTED_ARM_IDS),
    }[pool_id]
    if not expected_sources <= set(source_answers):
        raise ValueError(f"Pool {pool_id} references an unfrozen source arm")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        _reject_election_metric_fields(raw, f"Pool {pool_id}[{index}]")
        if set(raw) == {"video_id", "candidates", "ranking"}:
            candidates = raw["candidates"]
            ranking = raw["ranking"]
        elif set(raw) == {
            "video_id",
            "candidate_map",
            "ranked_candidate_ids",
        }:
            candidates = raw["candidate_map"]
            ranking = raw["ranked_candidate_ids"]
        else:
            raise ValueError(f"Pool {pool_id} row schema is not exact")
        video_id = raw.get("video_id")
        if video_id != expected_order[index]:
            raise ValueError(f"Pool {pool_id} does not use the frozen ID order")
        if (
            not isinstance(ranking, list)
            or not ranking
            or not all(
                isinstance(candidate_id, str) and candidate_id
                for candidate_id in ranking
            )
        ):
            raise ValueError(f"Pool {pool_id} row {video_id} has no valid ranking")
        entries = _candidate_entries(candidates, location=f"Pool {pool_id}/{video_id}")
        if not entries:
            raise ValueError(f"Pool {pool_id} has no candidates")
        expected_normalized_answers = {
            _normalize_candidate_text(source_answers[source_id][video_id])
            for source_id in expected_sources
        }
        if "" in expected_normalized_answers:
            raise ValueError(f"Pool {pool_id} has an empty normalized frozen answer")
        unique: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_sources: set[str] = set()
        seen_answers: set[str] = set()
        for candidate in entries:
            if "arm_id" not in candidate:
                raise ValueError(
                    f"Pool {pool_id} candidate arm_id/source identity is required"
                )
            if set(candidate) != {"candidate_id", "arm_id", "answer"}:
                raise ValueError(f"Pool {pool_id} candidate schema is not exact")
            candidate_id = candidate.get("candidate_id")
            answer = candidate.get("answer")
            source_id = candidate.get("arm_id")
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in seen_ids
                or not isinstance(answer, str)
                or not answer.strip()
                or not isinstance(source_id, str)
                or source_id not in expected_sources
                or source_id in seen_sources
            ):
                raise ValueError(
                    f"Pool {pool_id} candidate arm_id/source identity is invalid"
                )
            frozen_answer = source_answers[source_id].get(video_id)
            try:
                answer_bytes = answer.encode("utf-8")
                frozen_answer_bytes = (
                    frozen_answer.encode("utf-8")
                    if isinstance(frozen_answer, str)
                    else None
                )
            except UnicodeEncodeError as error:
                raise ValueError(
                    "election candidate answer is not valid UTF-8"
                ) from error
            if frozen_answer_bytes is None or answer_bytes != frozen_answer_bytes:
                raise ValueError("election candidate rewrites frozen answer bytes")
            normalized_answer = _normalize_candidate_text(answer)
            if not normalized_answer:
                raise ValueError(f"Pool {pool_id} candidate normalizes to empty text")
            seen_ids.add(candidate_id)
            seen_sources.add(source_id)
            if normalized_answer in seen_answers:
                # The first candidate in the frozen randomized order owns a
                # normalized duplicate.  Later IDs are intentionally hidden.
                continue
            seen_answers.add(normalized_answer)
            unique.append(
                {
                    "candidate_id": candidate_id,
                    "answer": answer,
                    "arm_id": source_id,
                }
            )
        if (
            seen_sources != expected_sources
            or seen_answers != expected_normalized_answers
        ):
            raise ValueError(
                f"Pool {pool_id} candidate membership is not the declared "
                "normalized-deduped set"
            )
        unique_ids = {candidate["candidate_id"] for candidate in unique}
        if set(ranking) != unique_ids or len(ranking) != len(unique_ids):
            raise ValueError(
                f"Pool {pool_id} ranking is not a complete candidate permutation"
            )
        result.append(
            {
                "video_id": video_id,
                "candidate_count": len(entries),
                "candidates": unique,
                "ranking": list(ranking),
            }
        )
    return result


def _metric_cache_identity(provenance: Mapping[str, Any]) -> str:
    config_digest = _sha256_bytes(
        _canonical_json(
            {
                "task": "vqa-silver-oracle-semantic-v1",
                "normalization": "strip, lowercase, collapse whitespace",
                "endpoint": STS_ENDPOINT,
                "timeout_seconds": EXPECTED_STS_TIMEOUT,
                "max_attempts": EXPECTED_STS_RETRIES,
                "protocol_sha256": provenance.get("protocol_sha256"),
            }
        )
    )
    return factorial._cache_identity(config_digest)


def _score_unique_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    bert_model: Path,
    meteor_fn: MetricFunction,
    bertscore_fn: BertScoreFunction,
    sts_fn: MetricFunction,
    sts_cache: STSCache,
    sts_identity: str,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, int]]:
    ordered_pairs = sorted(set(pairs))
    if not ordered_pairs:
        raise ValueError("cannot score an empty candidate set")
    meteor_values = {
        pair: _finite(meteor_fn(pair[0], pair[1]), field="meteor")
        for pair in ordered_pairs
    }
    bert_values = list(
        bertscore_fn(
            [pair[0] for pair in ordered_pairs],
            [pair[1] for pair in ordered_pairs],
            bert_model,
        )
    )
    if len(bert_values) != len(ordered_pairs):
        raise ValueError(
            f"BERTScore returned {len(bert_values)} rows for {len(ordered_pairs)} unique pairs"
        )
    hit_count = 0
    miss_count = 0
    live_request_count = 0
    sts_values: dict[tuple[str, str], float] = {}
    for pair in ordered_pairs:
        key = factorial._sts_pair_key(sts_identity, pair[0], pair[1])
        if key in sts_cache.values:
            value = _finite(sts_cache.values[key], field="sts")
            hit_count += 1
        else:
            value = _finite(sts_fn(pair[0], pair[1]), field="sts")
            sts_cache.values[key] = value
            miss_count += 1
            live_request_count += 1
        sts_values[pair] = value
    result: dict[tuple[str, str], dict[str, float]] = {}
    for index, pair in enumerate(ordered_pairs):
        metrics = {
            "meteor": meteor_values[pair],
            "bertscore_f1": _finite(bert_values[index], field="bertscore_f1"),
            "sts": sts_values[pair],
        }
        result[pair] = {
            **metrics,
            "semantic_composite": factorial.semantic_composite(metrics),
        }
    return result, {
        "unique_pair_count": len(ordered_pairs),
        "meteor_pair_count": len(ordered_pairs),
        "bertscore_pair_count": len(ordered_pairs),
        "sts_pair_count": len(ordered_pairs),
        "hit_count": hit_count,
        "miss_count": miss_count,
        "live_request_count": live_request_count,
    }


def paired_bootstrap(
    vectors: Mapping[str, Sequence[float]],
    *,
    iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    seed: int = EXPECTED_BOOTSTRAP_SEED,
    confidence: float = EXPECTED_CONFIDENCE,
) -> dict[str, dict[str, Any]]:
    """Return deterministic percentile intervals from shared row resamples."""

    if seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if iterations != EXPECTED_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            f"bootstrap iterations must be {EXPECTED_BOOTSTRAP_ITERATIONS}"
        )
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between zero and one")
    lengths = {len(values) for values in vectors.values()}
    if not lengths or lengths == {0} or len(lengths) != 1:
        raise ValueError("paired bootstrap vectors must have one non-empty length")
    size = next(iter(lengths))
    source = random.Random(seed)
    samples = {field: [] for field in vectors}
    for _ in range(iterations):
        indices = source.choices(range(size), k=size)
        for field, values in vectors.items():
            samples[field].append(statistics.fmean(values[index] for index in indices))
    tail = (1.0 - confidence) / 2.0
    lower_index = min(iterations - 1, int(tail * iterations))
    upper_index = min(iterations - 1, max(0, int((1.0 - tail) * iterations)))
    return {
        field: {
            "lower": sorted(values)[lower_index],
            "upper": sorted(values)[upper_index],
            "confidence": confidence,
            "iterations": iterations,
            "seed": seed,
            "exploratory": True,
        }
        for field, values in samples.items()
    }


def _metric_row(
    *, video_id: str, stratum: str, metrics: Mapping[str, float], answer: str
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "stratum": stratum,
        "normalized_exact_match": factorial.normalized_exact_match(answer, ""),
        "token_f1": factorial.token_f1(answer, ""),
        "answer_token_count": len(normalize_nist_text(answer).split()),
        **{field: _finite(metrics[field], field=field) for field in SEMANTIC_FIELDS},
    }


def _metric_row_with_reference(
    *,
    video_id: str,
    stratum: str,
    answer: str,
    reference: str,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    row = _metric_row(
        video_id=video_id, stratum=stratum, metrics=metrics, answer=answer
    )
    row["normalized_exact_match"] = factorial.normalized_exact_match(answer, reference)
    row["token_f1"] = factorial.token_f1(answer, reference)
    return row


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty row set")
    result = {
        "row_count": len(rows),
        "completed_count": len(rows),
        "completion_rate": 1.0,
        "mean_answer_token_count": statistics.fmean(
            row["answer_token_count"] for row in rows
        ),
    }
    for field in (*DIAGNOSTIC_FIELDS, *SEMANTIC_FIELDS):
        result[field] = statistics.fmean(
            _finite(row[field], field=field) for row in rows
        )
    return result


def _paired_rows(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    treatment_id: str,
) -> list[dict[str, Any]]:
    baseline_by_id = {str(row["video_id"]): row for row in baseline}
    treatment_by_id = {str(row["video_id"]): row for row in treatment}
    if set(baseline_by_id) != set(treatment_by_id):
        raise ValueError(f"paired contrast {treatment_id} has a mismatched ID set")
    result = []
    for video_id in sorted(baseline_by_id):
        if baseline_by_id[video_id]["stratum"] != treatment_by_id[video_id]["stratum"]:
            raise ValueError(f"paired contrast {treatment_id} has mismatched strata")
        result.append(
            {
                "video_id": video_id,
                "stratum": baseline_by_id[video_id]["stratum"],
                **{
                    field: treatment_by_id[video_id][field]
                    - baseline_by_id[video_id][field]
                    for field in SEMANTIC_FIELDS
                },
            }
        )
    return result


def _paired_summary(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    treatment_id: str,
) -> dict[str, Any]:
    rows = _paired_rows(baseline, treatment, treatment_id=treatment_id)
    vectors = {field: [row[field] for row in rows] for field in SEMANTIC_FIELDS}
    return {
        "baseline_id": treatment_id.split("-", 1)[-1] if False else None,
        "treatment_id": treatment_id,
        "paired_count": len(rows),
        "metric_deltas": {
            field: statistics.fmean(values) for field, values in vectors.items()
        },
        "bootstrap": paired_bootstrap(vectors),
    }


def _contrast_summary(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    contrast_id: str,
    baseline_id: str,
    treatment_id: str,
) -> dict[str, Any]:
    rows = _paired_rows(baseline, treatment, treatment_id=treatment_id)
    vectors = {field: [row[field] for row in rows] for field in SEMANTIC_FIELDS}
    return {
        "baseline_id": baseline_id,
        "treatment_id": treatment_id,
        "paired_count": len(rows),
        "metric_deltas": {
            field: statistics.fmean(values) for field, values in vectors.items()
        },
        "bootstrap": paired_bootstrap(vectors),
    }


def _contrast_value(contrast: Mapping[str, Any], field: str) -> float:
    deltas = contrast.get("metric_deltas", contrast)
    return float(deltas[field])


def evidence_gate(
    contrast: Mapping[str, Any],
    *,
    localized_coverage: float,
    complete: bool = True,
    integrity: bool = True,
) -> dict[str, Any]:
    lower = float(contrast["bootstrap"]["semantic_composite"]["lower"])
    checks = {
        "delta_C>=0.015": _contrast_value(contrast, "semantic_composite") >= 0.015,
        "lower_CI_C>0": lower > 0,
        "localized_coverage>=0.95": localized_coverage >= 0.95,
        "delta_METEOR>=0": _contrast_value(contrast, "meteor") >= 0,
        "delta_BERTScore>=0": _contrast_value(contrast, "bertscore_f1") >= 0,
        "delta_STS>=0": _contrast_value(contrast, "sts") >= 0,
        "complete": complete,
        "integrity": integrity,
    }
    return {
        "passed": all(checks.values()),
        "gates": checks,
        "localized_coverage": localized_coverage,
        "thresholds": {"delta_C": 0.015, "lower_CI_C": 0.0, "coverage": 0.95},
    }


def election_gate(
    contrast: Mapping[str, Any],
    *,
    capture_fraction: float,
    complete: bool = True,
    integrity: bool = True,
) -> dict[str, Any]:
    lower = float(contrast["bootstrap"]["semantic_composite"]["lower"])
    checks = {
        "delta_C>=0.010": _contrast_value(contrast, "semantic_composite") >= 0.010,
        "lower_CI_C>0": lower > 0,
        "oracle_capture>=0.30": capture_fraction >= 0.30,
        "delta_METEOR>=-0.002": _contrast_value(contrast, "meteor") >= -0.002,
        "delta_BERTScore>=-0.002": _contrast_value(contrast, "bertscore_f1") >= -0.002,
        "delta_STS>=-0.002": _contrast_value(contrast, "sts") >= -0.002,
        "complete": complete,
        "integrity": integrity,
    }
    return {
        "passed": all(checks.values()),
        "gates": checks,
        "capture_fraction": capture_fraction,
        "thresholds": {
            "delta_C": 0.010,
            "lower_CI_C": 0.0,
            "capture": 0.30,
            "component_regression": -0.002,
        },
    }


def _safe_provenance(
    bindings: Mapping[str, Any],
    *,
    localized_coverage: float,
    localized_coverage_sha256: str,
    localized_construction_count: int,
    freeze_manifest_sha256: str | None,
) -> dict[str, Any]:
    answer_runtime = {
        field: bindings["conditions"]["G/F"][field]
        for field in (
            "model_id",
            "model_revision",
            "answer_policy",
            "min_pixels_per_frame",
            "max_pixels_per_frame",
            "runtime",
            "runtime_image",
            "transformers_revision",
            "do_sample",
            "enable_thinking",
            "max_new_tokens",
        )
    }
    judge_runtime = {
        field: bindings["elections"]["A"][field]
        for field in (
            "judge_model_id",
            "judge_model_revision",
            "runtime_image",
        )
    }
    result = {
        "protocol_sha256": bindings["protocol_sha256"],
        "evidence_sha256": bindings["evidence_sha256"],
        "selector_sha256": bindings["selector_sha256"],
        "answer_freeze_sha256": bindings["answer_freeze_sha256"],
        "judge_freeze_sha256": bindings["judge_freeze_sha256"],
        "panel_id_set_sha256": bindings["panel_id_set_sha256"],
        "verified_4b_current_prediction_sha256": bindings[
            "verified_4b_current_prediction_sha256"
        ],
        "verified_4b_current_source_prediction_sha256": bindings[
            "verified_4b_current_source_prediction_sha256"
        ],
        "condition_bindings_sha256": _sha256_bytes(
            _canonical_json(bindings["conditions"])
        ),
        "election_bindings_sha256": _sha256_bytes(
            _canonical_json(bindings["elections"])
        ),
        "answer_model_id": EXPECTED_ANSWER_MODEL_ID,
        "answer_model_revision": EXPECTED_ANSWER_MODEL_REVISION,
        "answer_policy": "current",
        "min_pixels_per_frame": factorial.EXPECTED_RUNTIME_CONTROLS["min_pixels"],
        "max_pixels_per_frame": factorial.EXPECTED_RUNTIME_CONTROLS["max_pixels"],
        "runtime_image_sha256": _sha256_bytes(
            EXPECTED_ANSWER_RUNTIME_IMAGE.encode("utf-8")
        ),
        "transformers_revision": EXPECTED_TRANSFORMERS_REVISION,
        "answer_runtime_identity_sha256": _sha256_bytes(
            _canonical_json(answer_runtime)
        ),
        "judge_model_id": EXPECTED_JUDGE_MODEL_ID,
        "judge_model_revision": EXPECTED_JUDGE_MODEL_REVISION,
        "judge_runtime_image_sha256": _sha256_bytes(
            EXPECTED_JUDGE_RUNTIME_IMAGE.encode("utf-8")
        ),
        "judge_runtime_identity_sha256": _sha256_bytes(_canonical_json(judge_runtime)),
        "localized_construction_coverage": localized_coverage,
        "localized_construction_count": localized_construction_count,
        "localized_coverage_sha256": localized_coverage_sha256,
    }
    if freeze_manifest_sha256 is not None:
        result["freeze_manifest_sha256"] = freeze_manifest_sha256
    return result


def _coverage(
    evidence_coverage: Mapping[str, bool] | None, ids: Sequence[str]
) -> float:
    if not isinstance(evidence_coverage, Mapping):
        raise ValueError("explicit per-row localized coverage input is required")
    if set(evidence_coverage) != set(ids):
        raise ValueError("evidence coverage IDs do not match the frozen panel")
    if any(type(evidence_coverage[video_id]) is not bool for video_id in ids):
        raise ValueError("evidence coverage values must be booleans")
    return sum(evidence_coverage[video_id] for video_id in ids) / len(ids)


def _safe_aggregate(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "row_count",
        "completed_count",
        "completion_rate",
        "mean_answer_token_count",
        *DIAGNOSTIC_FIELDS,
        *SEMANTIC_FIELDS,
    ]
    return {field: value[field] for field in fields if field in value}


SAFE_TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment",
    "panel",
    "scorer",
    "provenance",
    "inputs",
    "completion",
    "arms",
    "factorial",
    "elections",
    "ceilings",
    "contrasts",
    "election_contrasts",
    "gates",
    "integrity",
    "cache_summary",
}
SAFE_KEY_EXCEPTIONS = {
    "answer_model_id",
    "answer_model_revision",
    "answer_policy",
    "answer_freeze_sha256",
    "answer_runtime_identity_sha256",
    "mean_answer_token_count",
    "labels_joined_after_freeze",
    "candidate_count",
    "unique_candidate_count",
    "top1_source_counts",
    "verified_4b_current_prediction_sha256",
    "verified_4b_current_source_prediction_sha256",
    "single_answer_ndcg",
}
SAFE_NESTED_FIELDS = {
    *SAFE_TOP_LEVEL_FIELDS,
    "name",
    "design",
    "semantic_composite",
    "exact_match_role",
    "bootstrap",
    "iterations",
    "seed",
    "confidence",
    "expected_rows",
    "expected_annotation_files",
    "id_set_sha256",
    "nine",
    "pool_a",
    "pool_b",
    "pool_c",
    "declared_arms",
    "declared_factorial_conditions",
    "declared_pools",
    "bert_model_snapshot",
    "bert_score_module_version",
    "bert_num_layers",
    "bert_idf",
    "bert_rescale_with_baseline",
    "sts_timeout_seconds",
    "sts_max_attempts",
    "semantic_fields",
    "exact_match_diagnostic_only",
    "single_answer_ndcg",
    "election_ndcg",
    "applicable",
    "reason",
    "contract",
    "cutoff",
    "gain",
    "role",
    "normalization",
    "scorer_sha256",
    "task1_scorer_sha256",
    "evaluation_module_sha256",
    "protocol_sha256",
    "evidence_sha256",
    "answer_freeze_sha256",
    "judge_freeze_sha256",
    "selector_sha256",
    "panel_id_set_sha256",
    "runtime_image_sha256",
    "judge_runtime_image_sha256",
    "answer_model_id",
    "answer_model_revision",
    "judge_model_id",
    "judge_model_revision",
    "verified_4b_current_prediction_sha256",
    "verified_4b_current_source_prediction_sha256",
    "condition_bindings_sha256",
    "election_bindings_sha256",
    "answer_policy",
    "min_pixels_per_frame",
    "max_pixels_per_frame",
    "transformers_revision",
    "answer_runtime_identity_sha256",
    "judge_runtime_identity_sha256",
    "localized_coverage_sha256",
    "freeze_manifest_sha256",
    "localized_construction_count",
    "bert_model_snapshot_sha256",
    "sts_endpoint_sha256",
    "arm_input_sha256",
    "arm_request_sha256",
    "factorial_input_sha256",
    "factorial_request_sha256",
    "election_input_sha256",
    "election_request_sha256",
    "bindings_sha256",
    "completed",
    "expected",
    "complete",
    "top1",
    "ndcg",
    "ndcg_k",
    "ndcg_k_min",
    "ndcg_k_max",
    "candidate_count",
    "unique_candidate_count",
    "candidate_count_min",
    "candidate_count_max",
    "unique_candidate_count_min",
    "unique_candidate_count_max",
    "complete_ranking_count",
    "baseline_id",
    "treatment_id",
    "paired_count",
    "metric_deltas",
    "lower",
    "upper",
    "exploratory",
    "passed",
    "gates",
    "localized_coverage",
    "thresholds",
    "evidence",
    "election",
    "delta_c",
    "lower_ci_c",
    "coverage",
    "capture",
    "component_regression",
    "oracle_capture",
    "oracle_gain_vs_4b_current",
    "capture_fraction",
    "localized_construction_coverage",
    "delta_meteor",
    "delta_bertscore",
    "delta_sts",
    "delta_c",
    "delta_C>=0.015",
    "delta_C>=0.010",
    "lower_CI_C>0",
    "localized_coverage>=0.95",
    "delta_METEOR>=0",
    "delta_BERTScore>=0",
    "delta_STS>=0",
    "oracle_capture>=0.30",
    "delta_METEOR>=-0.002",
    "delta_BERTScore>=-0.002",
    "delta_STS>=-0.002",
    "all_nine_arms_complete",
    "all_three_rankings_complete",
    "exact_input_id_set_and_order",
    "exact_election_id_set_and_order",
    "labels_joined_after_freeze",
    "unique_pairs_scored_once",
    "finite_metrics_checked",
    "provenance_checked",
    "panel_identity_checked",
    "candidate_membership_checked",
    "ranking_permutations_checked",
    "identity",
    "unique_pair_count",
    "meteor_pair_count",
    "bertscore_pair_count",
    "sts_pair_count",
    "hit_count",
    "miss_count",
    "live_request_count",
    "private",
    "row_count",
    "completed_count",
    "completion_rate",
    "mean_answer_token_count",
    "normalized_exact_match",
    "token_f1",
    "meteor",
    "bertscore_f1",
    "sts",
}
FORBIDDEN_ARTIFACT_TOKENS = {
    "answer",
    "reference",
    "question",
    "transcript",
    "prediction",
    "query",
    "raw",
    "path",
    "uri",
    "filepath",
}
PRIVATE_ROW_FIELDS = {
    "videoid",
    "videoids",
    "stratum",
    "strata",
    "row",
    "rows",
    "captures",
    "evaluationrow",
    "evaluationrows",
}
SAFE_LIST_VALUES = {
    "$.panel.declared_arms": EXPECTED_ARM_IDS,
    "$.panel.declared_factorial_conditions": EXPECTED_FACTORIAL_IDS,
    "$.panel.declared_pools": EXPECTED_POOL_IDS,
    "$.scorer.semantic_fields": SEMANTIC_FIELDS,
}


def _is_dynamic_safe_key(location: str, key: str) -> bool:
    expected_by_location = {
        "$.arms": set(EXPECTED_ARM_IDS),
        "$.factorial": set(EXPECTED_FACTORIAL_IDS),
        "$.elections": set(EXPECTED_POOL_IDS),
        "$.completion.arms": set(EXPECTED_ARM_IDS),
        "$.completion.factorial": set(EXPECTED_FACTORIAL_IDS),
        "$.completion.elections": set(EXPECTED_POOL_IDS),
        "$.inputs.arm_input_sha256": set(EXPECTED_ARM_IDS),
        "$.inputs.arm_request_sha256": set(EXPECTED_ARM_IDS),
        "$.inputs.factorial_input_sha256": set(EXPECTED_FACTORIAL_IDS),
        "$.inputs.factorial_request_sha256": set(EXPECTED_FACTORIAL_IDS),
        "$.inputs.election_input_sha256": set(EXPECTED_POOL_IDS),
        "$.inputs.election_request_sha256": set(EXPECTED_POOL_IDS),
        "$.contrasts": {
            "H/S-G/F",
            "H/F-G/F",
            "L/F-G/F",
            "G/S-G/F",
            "H/S-H/F",
            "L/S-L/F",
        },
        "$.election_contrasts": set(EXPECTED_POOL_IDS),
    }
    if key in expected_by_location.get(location, set()):
        return True
    if location.endswith(".top1_source_counts"):
        return key in set(EXPECTED_SOURCE_IDS)
    return False


def validate_safe_artifact(value: object, location: str = "$") -> None:
    """Validate the positive top-level and recursive safe artifact allowlist."""

    if location == "$" and isinstance(value, Mapping):
        unknown = set(value) - SAFE_TOP_LEVEL_FIELDS
        missing = SAFE_TOP_LEVEL_FIELDS - set(value)
        if unknown or missing:
            raise ValueError(
                "artifact top-level schema is not exact: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = re.sub(r"[^a-z0-9]", "", key_text.lower())
            if normalized in PRIVATE_ROW_FIELDS:
                raise ValueError(
                    f"artifact contains private row field at {location}.{key_text}"
                )
            dynamic_id = _is_dynamic_safe_key(location, key_text)
            normalized_allowlist = {
                re.sub(r"[^a-z0-9]", "", item.lower()) for item in SAFE_NESTED_FIELDS
            }
            if normalized in {
                re.sub(r"[^a-z0-9]", "", item.lower()) for item in SAFE_KEY_EXCEPTIONS
            }:
                pass
            elif not dynamic_id and normalized not in normalized_allowlist:
                raise ValueError(
                    f"artifact contains unallowlisted field at {location}.{key_text}"
                )
            elif not dynamic_id and any(
                token in normalized for token in FORBIDDEN_ARTIFACT_TOKENS
            ):
                raise ValueError(
                    f"artifact contains forbidden field at {location}.{key_text}"
                )
            if isinstance(child, (Path, bytes, bytearray)):
                raise ValueError(
                    f"artifact contains a private value at {location}.{key_text}"
                )
            validate_safe_artifact(child, f"{location}.{key_text}")
    elif isinstance(value, list):
        expected = SAFE_LIST_VALUES.get(location)
        if expected is None or tuple(value) != tuple(expected):
            raise ValueError(f"artifact contains an unallowlisted array at {location}")
        if any(isinstance(child, (Mapping, list)) for child in value):
            raise ValueError(f"artifact contains a row/object array at {location}")
        for index, child in enumerate(value):
            validate_safe_artifact(child, f"{location}[{index}]")
    elif isinstance(value, str):
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            raise ValueError(f"artifact contains a URI at {location}")
        parent = location.rsplit(".", 1)[-1].lower()
        path_like = value.startswith(("/", "./", "../", "~/")) or bool(
            re.search(
                r"(?:^|[/\\])[^/\\]+\.(?:json|jsonl|csv|txt|pt|bin)$",
                value,
                re.IGNORECASE,
            )
        )
        if path_like and parent not in {
            "model_id",
            "model_revision",
            "answer_model_id",
            "answer_model_revision",
            "judge_model_id",
            "judge_model_revision",
        }:
            raise ValueError(f"artifact contains a path-like value at {location}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"atomic temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_private_cache(path: Path, identity: str) -> STSCache:
    if not path.exists():
        return STSCache(identity=identity)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("STS cache is invalid") from error
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("values"), Mapping
    ):
        raise ValueError("STS cache must contain identity and values")
    if payload.get("identity") != identity:
        raise ValueError(
            "STS cache identity does not match scorer/config/endpoint identity"
        )
    return STSCache(
        values={
            str(key): _finite(value, field="cached sts")
            for key, value in payload["values"].items()
        },
        identity=identity,
    )


def _write_private_cache(path: Path, cache: STSCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"atomic temporary cache already exists: {temporary}")
    payload = {"identity": cache.identity, "values": dict(sorted(cache.values.items()))}
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_elections(
    paths: Mapping[str, Path],
    *,
    expected_rows: int,
    expected_order: Sequence[str],
    source_rows: Mapping[str, Sequence[Mapping[str, str]]],
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Mapping[str, Any]]
]:
    if set(paths) != set(EXPECTED_POOL_IDS):
        raise ValueError("election scoring requires exact Pool A/B/C inputs")
    elections: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for pool_id in EXPECTED_POOL_IDS:
        payload, digest = _read_frozen_file(Path(paths[pool_id]))
        if digest != bindings[pool_id]["ranking_sha256"]:
            raise ValueError(
                f"Pool {pool_id} ranking bytes do not match the frozen binding"
            )
        _reject_label_fields(payload)
        rows, row_metadata = _payload_rows(payload, kind=f"Pool {pool_id}")
        if row_metadata.get("labels_loaded_after_prediction_freeze") is False:
            raise ValueError(f"Pool {pool_id} metadata violates ranking freeze order")
        validated = _validate_election_rows(
            rows,
            pool_id=pool_id,
            expected_rows=expected_rows,
            expected_order=expected_order,
            source_rows=source_rows,
        )
        elections[pool_id] = validated
        hashes[pool_id] = digest
        metadata[pool_id] = row_metadata
    return elections, hashes, metadata


def _candidate_metrics(
    candidate: Mapping[str, Any],
    *,
    reference: str,
    stratum: str,
    metric_map: Mapping[tuple[str, str], Mapping[str, float]],
    video_id: str,
) -> dict[str, Any]:
    answer = str(candidate["answer"])
    pair = (normalize_nist_text(answer), normalize_nist_text(reference))
    return _metric_row_with_reference(
        video_id=video_id,
        stratum=stratum,
        answer=answer,
        reference=reference,
        metrics=metric_map[pair],
    )


def _select_oracle(
    rows_by_source: Mapping[str, Mapping[str, Any]], ids: Sequence[str]
) -> list[dict[str, Any]]:
    selected = []
    for video_id in ids:
        candidates = [
            (source_id, rows[video_id]) for source_id, rows in rows_by_source.items()
        ]
        source_id, row = max(
            candidates,
            key=lambda item: (
                item[1]["semantic_composite"],
                -list(ids).index(video_id),
                item[0],
            ),
        )
        selected.append({**row, "arm_id": source_id})
    return selected


def _election_rows(
    pool_rows: Sequence[Mapping[str, Any]],
    *,
    annotations: Mapping[str, Mapping[str, Any]],
    metric_map: Mapping[tuple[str, str], Mapping[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    top_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    candidate_counts: list[int] = []
    unique_counts: list[int] = []
    for pool_row in pool_rows:
        video_id = str(pool_row["video_id"])
        annotation = annotations[video_id]
        by_id = {
            candidate["candidate_id"]: candidate for candidate in pool_row["candidates"]
        }
        ordered = [by_id[candidate_id] for candidate_id in pool_row["ranking"]]
        candidate_metric_rows = [
            _candidate_metrics(
                candidate,
                reference=annotation["reference"],
                stratum=annotation["question_stratum"],
                metric_map=metric_map,
                video_id=video_id,
            )
            for candidate in ordered
        ]
        gains = [row["semantic_composite"] for row in candidate_metric_rows]
        k = min(10, len(gains))
        ndcg = ndcg_from_gains(gains, k)
        top = {**candidate_metric_rows[0], "arm_id": ordered[0].get("arm_id")}
        top_rows.append(top)
        evaluation_rows.append(
            {
                "video_id": video_id,
                "stratum": annotation["question_stratum"],
                "candidate_count": pool_row["candidate_count"],
                "unique_candidate_count": len(pool_row["candidates"]),
                "ndcg": ndcg,
                "ndcg_k": k,
            }
        )
        candidate_counts.append(int(pool_row["candidate_count"]))
        unique_counts.append(len(pool_row["candidates"]))
    return top_rows, evaluation_rows, candidate_counts, unique_counts


def _mean_oracle_gain(
    oracle: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> float:
    baseline_by_id = {row["video_id"]: row for row in baseline}
    return statistics.fmean(
        oracle_row["semantic_composite"]
        - baseline_by_id[oracle_row["video_id"]]["semantic_composite"]
        for oracle_row in oracle
    )


def score_silver_oracle(
    *,
    annotation_root: Path,
    arm_paths: Mapping[str, Path],
    factorial_paths: Mapping[str, Path] | None = None,
    election_paths: Mapping[str, Path],
    bert_model: Path,
    expected_rows: int = 475,
    expected_annotation_files: int = 481,
    panel_profile: factorial.PanelProfile | None = None,
    bindings: Mapping[str, Any] | None = None,
    evidence_coverage: Mapping[str, bool] | None = None,
    freeze_manifest_path: Path | None = None,
    freeze_inputs: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
    meteor_fn: MetricFunction | None = None,
    bertscore_fn: BertScoreFunction | None = None,
    sts_fn: MetricFunction | None = None,
    sts_cache: STSCache | None = None,
    sts_cache_path: Path | None = None,
    bootstrap_iterations: int = EXPECTED_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = EXPECTED_BOOTSTRAP_SEED,
    confidence: float = EXPECTED_CONFIDENCE,
) -> dict[str, Any]:
    """Validate, score, summarize, and optionally publish the safe aggregate."""

    if output_path is not None and output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if isinstance(expected_rows, bool) or expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if bootstrap_iterations != EXPECTED_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            f"bootstrap iterations must be {EXPECTED_BOOTSTRAP_ITERATIONS}"
        )
    if bootstrap_seed != EXPECTED_BOOTSTRAP_SEED:
        raise ValueError(f"bootstrap seed must be {EXPECTED_BOOTSTRAP_SEED}")
    if confidence != EXPECTED_CONFIDENCE:
        raise ValueError(f"bootstrap confidence must be {EXPECTED_CONFIDENCE}")
    if expected_rows == 475 and expected_annotation_files != 481:
        raise ValueError("the canonical 475-row panel requires 481 annotations")
    verified_freeze: dict[str, Any] | None = None
    if expected_rows == 475:
        if freeze_manifest_path is None or not isinstance(freeze_inputs, Mapping):
            raise ValueError(
                "canonical scoring requires the pinned freeze manifest and input locator"
            )
        if bindings is not None or evidence_coverage is not None:
            raise ValueError(
                "canonical scoring rejects caller-authored bindings and coverage"
            )
        verified_freeze = _verify_label_free_freeze(
            freeze_manifest_path=Path(freeze_manifest_path),
            freeze_inputs=freeze_inputs,
            expected_rows=expected_rows,
        )
        bindings = verified_freeze["bindings"]
        evidence_coverage = verified_freeze["coverage"]
    elif freeze_manifest_path is not None or freeze_inputs is not None:
        if freeze_manifest_path is None or not isinstance(freeze_inputs, Mapping):
            raise ValueError(
                "freeze manifest and input locator must be supplied together"
            )
        if bindings is not None or evidence_coverage is not None:
            raise ValueError(
                "anchored scoring rejects separate caller-authored bindings and coverage"
            )
        verified_freeze = _verify_label_free_freeze(
            freeze_manifest_path=Path(freeze_manifest_path),
            freeze_inputs=freeze_inputs,
            expected_rows=expected_rows,
        )
        bindings = verified_freeze["bindings"]
        evidence_coverage = verified_freeze["coverage"]
    resolved_bindings = _validate_frozen_bindings(bindings, expected_rows=expected_rows)
    if not isinstance(arm_paths, Mapping):
        raise ValueError("arm paths must be an ID-to-path mapping")
    if set(arm_paths) != set(EXPECTED_ARM_IDS):
        normalized = {_canonical_arm_id(key): value for key, value in arm_paths.items()}
        if set(normalized) != set(EXPECTED_ARM_IDS):
            raise ValueError("scoring requires exactly the nine silver-oracle arm IDs")
        arm_paths = normalized
    else:
        arm_paths = dict(arm_paths)
    if not isinstance(factorial_paths, Mapping) or set(factorial_paths) != set(
        EXPECTED_FACTORIAL_IDS
    ):
        raise ValueError(
            "factorial paths must contain all four exact legacy conditions"
        )
    factorial_paths = dict(factorial_paths)
    if any(
        not isinstance(path, Path)
        for path in [*arm_paths.values(), *factorial_paths.values()]
    ):
        arm_paths = {key: Path(path) for key, path in arm_paths.items()}
        factorial_paths = {key: Path(path) for key, path in factorial_paths.items()}
    all_paths = {**factorial_paths, **arm_paths}
    frozen_sources, source_hashes, _source_metadata = _freeze_prediction_inputs(
        all_paths,
        expected_rows=expected_rows,
        source_kind="prediction input",
        bindings=resolved_bindings["conditions"],
    )
    if len(set(source_hashes.values())) != len(source_hashes):
        raise ValueError("frozen source input files must be distinct")
    frozen_order = [
        row["video_id"] for row in frozen_sources[next(iter(frozen_sources))]
    ]
    frozen_elections, election_hashes, _election_metadata = _parse_elections(
        {str(pool_id): Path(path) for pool_id, path in election_paths.items()},
        expected_rows=expected_rows,
        expected_order=frozen_order,
        source_rows=frozen_sources,
        bindings=resolved_bindings["elections"],
    )
    if len(set(election_hashes.values())) != len(election_hashes):
        raise ValueError("frozen election input files must be distinct")
    frozen_panel_sha256 = factorial.sorted_video_id_set_sha256(frozen_order)
    if resolved_bindings["panel_id_set_sha256"] != frozen_panel_sha256:
        raise ValueError("binding panel digest does not match the frozen ID set")
    coverage = _coverage(evidence_coverage, frozen_order)
    if verified_freeze is not None:
        localized_coverage_sha256 = verified_freeze["manifest"]["evidence"][
            "coverage_sha256"
        ]
    else:
        localized_coverage_sha256 = _sha256_bytes(
            _canonical_json(
                {
                    video_id: evidence_coverage[video_id]
                    for video_id in sorted(frozen_order)
                }
            )
        )
    # This is intentionally after all answer/ranking bytes are frozen.
    annotations = _load_annotations(Path(annotation_root))
    if verified_freeze is not None:
        post_label_freeze = _verify_label_free_freeze(
            freeze_manifest_path=Path(freeze_manifest_path),
            freeze_inputs=freeze_inputs,
            expected_rows=expected_rows,
        )
        if (
            post_label_freeze["manifest_sha256"] != verified_freeze["manifest_sha256"]
            or post_label_freeze["bindings"] != verified_freeze["bindings"]
            or post_label_freeze["coverage"] != verified_freeze["coverage"]
        ):
            raise ValueError(
                "label-free freeze identity changed after annotations opened"
            )
    if len(annotations) != expected_annotation_files:
        raise ValueError(
            f"annotation root contains {len(annotations)} files; expected {expected_annotation_files}"
        )
    if set(annotations) < set(frozen_order):
        raise ValueError("frozen prediction IDs are not all present in annotations")
    if expected_rows == 475:
        profile = panel_profile or factorial.CANONICAL_PANEL_PROFILE
        panel_identity = factorial._validate_panel_identity(
            profile,
            annotation_ids=annotations,
            condition_ids={source_id: frozen_order for source_id in frozen_sources},
        )
    else:
        panel_identity = {
            "profile": panel_profile.name
            if panel_profile
            else "fixture-or-declared-panel",
            "canonical_id_set_sha256": factorial.sorted_video_id_set_sha256(
                frozen_order
            ),
            "id_set_digest_algorithm": factorial.PANEL_ID_DIGEST_ALGORITHM,
            "usable_video_id_count": expected_rows,
        }
    if not bert_model.is_dir() or bert_model.name != EXPECTED_BERT_MODEL_SNAPSHOT:
        raise ValueError(
            f"BERT model must be the pinned snapshot directory {EXPECTED_BERT_MODEL_SNAPSHOT}"
        )
    if panel_identity["canonical_id_set_sha256"] != frozen_panel_sha256:
        raise ValueError("validated panel identity does not match frozen input IDs")
    resolved_provenance = _safe_provenance(
        resolved_bindings,
        localized_coverage=coverage,
        localized_coverage_sha256=localized_coverage_sha256,
        localized_construction_count=sum(
            bool(evidence_coverage[video_id]) for video_id in frozen_order
        ),
        freeze_manifest_sha256=(
            verified_freeze["manifest_sha256"] if verified_freeze is not None else None
        ),
    )
    metric_identity = _metric_cache_identity(resolved_provenance)
    if sts_cache is None and sts_cache_path is not None:
        cache = _load_private_cache(Path(sts_cache_path), metric_identity)
    else:
        cache = sts_cache or STSCache(identity=metric_identity)
    if cache.identity is None:
        cache.identity = metric_identity
    elif cache.identity != metric_identity:
        raise ValueError(
            "STS cache identity does not match scorer/config/endpoint identity"
        )
    meteor_fn = meteor_fn or factorial.task1._default_meteor
    if bertscore_fn is None:
        versions = factorial.task1._bert_score_versions(required=True)
    else:
        versions = {
            "module": EXPECTED_BERT_SCORE_MODULE_VERSION,
            "distribution": "injected",
        }
    bertscore_fn = bertscore_fn or factorial.task1._default_bertscore
    sts_fn = sts_fn or (
        lambda prediction, reference: umbc_sts_score(
            prediction,
            reference,
            timeout=EXPECTED_STS_TIMEOUT,
            retries=EXPECTED_STS_RETRIES,
        )
    )
    pairs: list[tuple[str, str]] = []
    for rows in frozen_sources.values():
        for row in rows:
            pairs.append(
                (
                    normalize_nist_text(row["answer"]),
                    normalize_nist_text(annotations[row["video_id"]]["reference"]),
                )
            )
    for pool_rows in frozen_elections.values():
        for pool_row in pool_rows:
            reference = normalize_nist_text(
                annotations[pool_row["video_id"]]["reference"]
            )
            pairs.extend(
                (normalize_nist_text(candidate["answer"]), reference)
                for candidate in pool_row["candidates"]
            )
    metric_map, cache_counts = _score_unique_pairs(
        pairs,
        bert_model=bert_model,
        meteor_fn=meteor_fn,
        bertscore_fn=bertscore_fn,
        sts_fn=sts_fn,
        sts_cache=cache,
        sts_identity=metric_identity,
    )
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for source_id, predictions in frozen_sources.items():
        scored = []
        for prediction in predictions:
            annotation = annotations[prediction["video_id"]]
            pair = (
                normalize_nist_text(prediction["answer"]),
                normalize_nist_text(annotation["reference"]),
            )
            scored.append(
                _metric_row_with_reference(
                    video_id=prediction["video_id"],
                    stratum=annotation["question_stratum"],
                    answer=prediction["answer"],
                    reference=annotation["reference"],
                    metrics=metric_map[pair],
                )
            )
        source_rows[source_id] = scored
    source_by_id = {
        source_id: {row["video_id"]: row for row in rows}
        for source_id, rows in source_rows.items()
    }
    safe_arm_reports = {
        arm_id: _safe_aggregate(_aggregate_metric_rows(source_rows[arm_id]))
        for arm_id in EXPECTED_ARM_IDS
    }
    safe_factorial_reports = {
        condition_id: _safe_aggregate(_aggregate_metric_rows(source_rows[condition_id]))
        for condition_id in EXPECTED_FACTORIAL_IDS
        if condition_id in source_rows
    }
    safe_elections: dict[str, dict[str, Any]] = {}
    top_rows_by_pool: dict[str, list[dict[str, Any]]] = {}
    for pool_id in EXPECTED_POOL_IDS:
        top_rows, evaluation_rows, counts, unique_counts = _election_rows(
            frozen_elections[pool_id], annotations=annotations, metric_map=metric_map
        )
        top_rows_by_pool[pool_id] = top_rows
        aggregate_top = _safe_aggregate(_aggregate_metric_rows(top_rows))
        top_sources = Counter(
            row.get("arm_id") for row in top_rows if row.get("arm_id") is not None
        )
        ndcgs = [row["ndcg"] for row in evaluation_rows]
        ndcg_cutoffs = [row["ndcg_k"] for row in evaluation_rows]
        safe_elections[pool_id] = {
            "top1": aggregate_top,
            "ndcg": statistics.fmean(ndcgs),
            "ndcg_k": min(10, max(unique_counts)),
            "ndcg_k_min": min(ndcg_cutoffs),
            "ndcg_k_max": max(ndcg_cutoffs),
            "candidate_count": min(counts),
            "unique_candidate_count": min(unique_counts),
            "candidate_count_min": min(counts),
            "candidate_count_max": max(counts),
            "unique_candidate_count_min": min(unique_counts),
            "unique_candidate_count_max": max(unique_counts),
            "complete_ranking_count": len(top_rows),
            "top1_source_counts": dict(sorted(top_sources.items())),
        }
    nine_oracle_rows = _select_oracle(
        {arm_id: source_by_id[arm_id] for arm_id in EXPECTED_ARM_IDS},
        frozen_order,
    )
    pool_oracle_rows: dict[str, list[dict[str, Any]]] = {}
    for pool_id in EXPECTED_POOL_IDS:
        for pool_row in frozen_elections[pool_id]:
            # Select the reference-aware maximum only from this row's candidates.
            pool_oracle_rows.setdefault(pool_id, []).append(
                max(
                    [
                        _candidate_metrics(
                            candidate,
                            reference=annotations[pool_row["video_id"]]["reference"],
                            stratum=annotations[pool_row["video_id"]][
                                "question_stratum"
                            ],
                            metric_map=metric_map,
                            video_id=pool_row["video_id"],
                        )
                        for candidate in pool_row["candidates"]
                    ],
                    key=lambda row: row["semantic_composite"],
                )
            )
    baseline_rows = source_rows.get(ELECTION_BASELINE)
    if baseline_rows is None:
        raise ValueError("Pool-A election gate requires the frozen 4b-current baseline")
    baseline_aggregate = _aggregate_metric_rows(baseline_rows)
    ceilings: dict[str, Any] = {
        "nine": _safe_aggregate(_aggregate_metric_rows(nine_oracle_rows)),
        "pool_a": _safe_aggregate(_aggregate_metric_rows(pool_oracle_rows["A"])),
        "pool_b": _safe_aggregate(_aggregate_metric_rows(pool_oracle_rows["B"])),
        "pool_c": _safe_aggregate(_aggregate_metric_rows(pool_oracle_rows["C"])),
    }
    for pool_id in EXPECTED_POOL_IDS:
        oracle_gain = _mean_oracle_gain(pool_oracle_rows[pool_id], baseline_rows)
        election_delta = (
            safe_elections[pool_id]["top1"]["semantic_composite"]
            - baseline_aggregate["semantic_composite"]
        )
        capture = election_delta / oracle_gain if oracle_gain > 0 else 0.0
        ceilings[f"pool_{pool_id.lower()}"]["oracle_gain_vs_4b_current"] = oracle_gain
        ceilings[f"pool_{pool_id.lower()}"]["capture_fraction"] = capture
    declared_pairs = (
        ("H/S-G/F", PRIMARY_BASELINE, "H/S"),
        ("H/F-G/F", PRIMARY_BASELINE, "H/F"),
        ("L/F-G/F", PRIMARY_BASELINE, "L/F"),
        ("G/S-G/F", PRIMARY_BASELINE, "G/S"),
        ("H/S-H/F", "H/F", "H/S"),
        ("L/S-L/F", "L/F", "L/S"),
    )
    contrasts = {
        contrast_id: _contrast_summary(
            source_rows[baseline_id],
            source_rows[treatment_id],
            contrast_id=contrast_id,
            baseline_id=baseline_id,
            treatment_id=treatment_id,
        )
        for contrast_id, baseline_id, treatment_id in declared_pairs
    }
    election_contrasts = {
        pool_id: _contrast_summary(
            baseline_rows,
            top_rows_by_pool[pool_id],
            contrast_id=f"Pool-{pool_id}-top1-4b-current",
            baseline_id=ELECTION_BASELINE,
            treatment_id=f"Pool-{pool_id}-top1",
        )
        for pool_id in EXPECTED_POOL_IDS
    }
    integrity = {
        "all_nine_arms_complete": all(
            len(source_rows[arm_id]) == expected_rows for arm_id in EXPECTED_ARM_IDS
        ),
        "all_three_rankings_complete": all(
            safe_elections[pool_id]["complete_ranking_count"] == expected_rows
            for pool_id in EXPECTED_POOL_IDS
        ),
        "exact_input_id_set_and_order": True,
        "exact_election_id_set_and_order": True,
        "labels_joined_after_freeze": True,
        "unique_pairs_scored_once": True,
        "finite_metrics_checked": True,
        "provenance_checked": verified_freeze is not None,
        "panel_identity_checked": True,
        "candidate_membership_checked": True,
        "ranking_permutations_checked": True,
    }
    integrity_ok = all(integrity.values())
    gates = {
        "evidence": evidence_gate(
            contrasts["H/S-G/F"],
            localized_coverage=coverage,
            complete=integrity["all_nine_arms_complete"],
            integrity=integrity_ok,
        ),
        "election": election_gate(
            election_contrasts["A"],
            capture_fraction=ceilings["pool_a"]["capture_fraction"],
            complete=integrity["all_three_rankings_complete"],
            integrity=integrity_ok,
        ),
    }
    input_hashes = {
        "arm_input_sha256": dict(
            sorted({key: source_hashes[key] for key in EXPECTED_ARM_IDS}.items())
        ),
        "arm_request_sha256": {
            key: resolved_bindings["conditions"][key]["request_sha256"]
            for key in sorted(EXPECTED_ARM_IDS)
        },
        "factorial_input_sha256": dict(
            sorted({key: source_hashes[key] for key in factorial_paths}.items())
        ),
        "factorial_request_sha256": {
            key: resolved_bindings["conditions"][key]["request_sha256"]
            for key in sorted(EXPECTED_FACTORIAL_IDS)
        },
        "election_input_sha256": dict(sorted(election_hashes.items())),
        "election_request_sha256": {
            key: resolved_bindings["elections"][key]["request_sha256"]
            for key in sorted(EXPECTED_POOL_IDS)
        },
        "bindings_sha256": _sha256_bytes(_canonical_json(resolved_bindings)),
        "localized_coverage_sha256": localized_coverage_sha256,
        "scorer_sha256": sha256_path(Path(__file__)),
        "task1_scorer_sha256": sha256_path(factorial.TASK1_SCORER_PATH),
        "evaluation_module_sha256": sha256_path(EVALUATION_MODULE_PATH),
        "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
        "bert_model_snapshot_sha256": sha256_path(bert_model),
        "sts_endpoint_sha256": _sha256_bytes(STS_ENDPOINT.encode("utf-8")),
    }
    if verified_freeze is not None:
        input_hashes["freeze_manifest_sha256"] = verified_freeze["manifest_sha256"]
    artifact: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "experiment": {
            "name": "vqa-silver-oracle-election-475-20260809",
            "design": "3x3 evidence factorial plus reference-blind Pools A/B/C",
            "semantic_composite": "mean(meteor, bertscore_f1, sts)",
            "exact_match_role": "diagnostic_only",
            "bootstrap": {
                "iterations": EXPECTED_BOOTSTRAP_ITERATIONS,
                "seed": EXPECTED_BOOTSTRAP_SEED,
                "confidence": confidence,
            },
        },
        "panel": {
            "expected_rows": expected_rows,
            "expected_annotation_files": expected_annotation_files,
            "id_set_sha256": panel_identity["canonical_id_set_sha256"],
            "declared_arms": list(EXPECTED_ARM_IDS),
            "declared_factorial_conditions": list(EXPECTED_FACTORIAL_IDS),
            "declared_pools": list(EXPECTED_POOL_IDS),
        },
        "scorer": {
            "name": "NIST 2025 semantic VQA silver-oracle scorer",
            "normalization": "strip, lowercase, collapse whitespace",
            "bert_model_snapshot": EXPECTED_BERT_MODEL_SNAPSHOT,
            "bert_score_module_version": versions["module"],
            "bert_num_layers": 17,
            "bert_idf": False,
            "bert_rescale_with_baseline": False,
            "sts_timeout_seconds": EXPECTED_STS_TIMEOUT,
            "sts_max_attempts": EXPECTED_STS_RETRIES,
            "semantic_fields": list(SEMANTIC_FIELDS),
            "exact_match_diagnostic_only": True,
            "single_answer_ndcg": {
                "applicable": False,
                "reason": (
                    "competition evaluator marks one free-form answer NDCG "
                    "not applicable; a single-item ranking is degenerate at 1.0"
                ),
            },
            "election_ndcg": {
                "applicable": True,
                "contract": "NIST 2025 order-sensitive ranked-answer NDCG",
                "cutoff": "min(10, unique_candidate_count)",
                "gain": "semantic_composite",
                "role": "local_election_diagnostic",
            },
        },
        "provenance": resolved_provenance,
        "inputs": input_hashes,
        "completion": {
            "arms": {
                arm_id: {
                    "completed": len(source_rows[arm_id]),
                    "expected": expected_rows,
                }
                for arm_id in EXPECTED_ARM_IDS
            },
            "factorial": {
                condition_id: {
                    "completed": len(source_rows[condition_id]),
                    "expected": expected_rows,
                }
                for condition_id in EXPECTED_FACTORIAL_IDS
            },
            "elections": {
                pool_id: {
                    "complete": safe_elections[pool_id]["complete_ranking_count"],
                    "expected": expected_rows,
                }
                for pool_id in EXPECTED_POOL_IDS
            },
        },
        "arms": safe_arm_reports,
        "factorial": safe_factorial_reports,
        "elections": safe_elections,
        "ceilings": ceilings,
        "contrasts": contrasts,
        "election_contrasts": election_contrasts,
        "gates": gates,
        "integrity": integrity,
        "cache_summary": {"identity": metric_identity, **cache_counts, "private": True},
    }
    validate_safe_artifact(artifact)
    if sts_cache_path is not None:
        _write_private_cache(Path(sts_cache_path), cache)
    if output_path is not None:
        atomic_write_json(Path(output_path), artifact)
    return artifact


def _parse_pairs(values: Sequence[str], *, kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key.strip() or not raw_path.strip():
            raise ValueError(f"{kind} must be ID=PATH")
        if key.strip() in result:
            raise ValueError(f"duplicate {kind} ID: {key.strip()}")
        result[key.strip()] = Path(raw_path)
    return result


def _load_mapping_file(path: Path, *, kind: str) -> dict[str, Any]:
    payload, _digest = _read_frozen_file(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind} must be a JSON object")
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="ID=PATH")
    parser.add_argument(
        "--factorial", action="append", required=True, metavar="ID=PATH"
    )
    parser.add_argument(
        "--election", action="append", required=True, metavar="POOL=PATH"
    )
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-inputs", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--sts-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factorial_paths = _parse_pairs(args.factorial, kind="factorial")
    artifact = score_silver_oracle(
        annotation_root=args.annotations,
        arm_paths=_parse_pairs(args.arm, kind="arm"),
        factorial_paths=factorial_paths,
        election_paths=_parse_pairs(args.election, kind="election"),
        freeze_manifest_path=args.freeze_manifest,
        freeze_inputs=_load_mapping_file(args.freeze_inputs, kind="freeze inputs"),
        bert_model=args.bert_model,
        sts_cache_path=args.sts_cache,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"output": str(args.output), "gate_status": artifact["gates"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
