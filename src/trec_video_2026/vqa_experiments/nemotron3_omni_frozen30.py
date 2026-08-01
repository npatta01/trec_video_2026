"""Independent dual-freeze evaluation for the Nemotron frozen-30 run."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from trec_video_2026.vqa_evaluation import normalized_exact_match, token_f1
from trec_video_2026.vqa_experiments.nemotron3_omni_grading import (
    BlindGradingBundle,
    GradeResolution,
)
from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    CONDITIONS,
    FROZEN30_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    _validate_run_contract,
    build_chat_request,
)
from trec_video_2026.vqa_gate import canonical_request_sha256

QWEN_RUN_ID = "native-audio-benchmark-v2"
QWEN_FULL_FREEZE_SHA256 = (
    "8aa8c76e809402a97e32277e0846b1cdae033c9727535717724b1be0b0235879"
)
QWEN_CONDITION_ORDER = (
    "qwen35-4b-12f-frozen30",
    "qwen35-4b-24f-frozen30",
    "minicpmo45-native-video-audio-off-frozen30",
    "qwen35-4b-12f-whisper-large-v3-frozen30",
)
CANDIDATE_RUN_ID = "nemotron3-omni-frozen30-20260801-a1"
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN_RUNTIME_IMAGE = (
    "localhost/trec-vqa-gate-rocm@"
    "sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
)
_CANDIDATE_CONDITIONS = ("audio-off", "audio-on")
_ALL_CONDITIONS = ("qwen12", *_CANDIDATE_CONDITIONS)
_GRADES = frozenset({"C", "P", "W"})
_GRADE_RANK = {"W": 0, "P": 1, "C": 2}
_RUBRIC = {"C": "Correct", "P": "Partially correct", "W": "Wrong"}
_LABEL_KEYS = frozenset(
    {"annotation", "correct_answer", "distractor", "distractors", "reference"}
)
_NORMALIZED_MEDIA_ROOT = Path("/dstack/input/normalized")


@dataclass(frozen=True)
class OperationalEvidence:
    completed: int
    expected: int
    duration_minutes: float
    compute_cost_usd: float
    artifacts_verified: bool
    privacy_or_audit_gap: bool


@dataclass(frozen=True, repr=False)
class Frozen30GradingBundle(BlindGradingBundle):
    """Opaque grader packets plus private verified rows for final scoring."""

    _rows: tuple[dict[str, Any], ...]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _parse_jsonl(data: bytes, *, description: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be UTF-8 JSONL") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{description} must contain only object rows")
    return rows


def _contains_label(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and key.lower() in _LABEL_KEYS)
            or _contains_label(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_label(item) for item in value)
    return False


def verify_qwen_freeze(predictions_path: Path) -> tuple[dict[str, Any], ...]:
    """Recompute and validate the pinned original 120-row Qwen v2 freeze."""

    if not predictions_path.is_file():
        raise ValueError("Qwen frozen predictions must be a file")
    frozen_bytes = predictions_path.read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != QWEN_FULL_FREEZE_SHA256:
        raise ValueError("Qwen pinned full-file SHA-256 mismatch")
    rows = _parse_jsonl(frozen_bytes, description="Qwen frozen predictions")
    if len(rows) != 120 or _canonical_jsonl(rows) != frozen_bytes:
        raise ValueError("Qwen freeze must contain 120 canonical rows")

    expected_keys = [
        (condition, video_id)
        for condition in QWEN_CONDITION_ORDER
        for video_id, _category, _role in FROZEN30_ROWS
    ]
    actual_keys = [(row.get("condition_id"), row.get("video_id")) for row in rows]
    if actual_keys != expected_keys:
        raise ValueError("Qwen canonical condition/order mismatch")
    for row, (condition, video_id) in zip(rows, expected_keys, strict=True):
        request = row.get("request")
        if (
            row.get("run_id") != QWEN_RUN_ID
            or row.get("status") != "passed"
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or not isinstance(request, Mapping)
            or request.get("run_id") != QWEN_RUN_ID
            or request.get("condition_id") != condition
            or request.get("video_id") != video_id
            or row.get("request_sha256") != canonical_request_sha256(request)
            or not all(
                _is_sha256(row.get(field))
                for field in (
                    "evidence_identity_sha256",
                    "evidence_manifest_sha256",
                    "launch_decision_sha256",
                )
            )
            or _contains_label(request)
        ):
            raise ValueError("Qwen embedded request provenance mismatch")
    for row in rows[:30]:
        if (
            row.get("model_id") != QWEN_MODEL_ID
            or row.get("model_revision") != QWEN_MODEL_REVISION
            or row.get("runtime_image") != QWEN_RUNTIME_IMAGE
        ):
            raise ValueError("Qwen12 model provenance mismatch")
    return tuple(rows[:30])


def _checksum_entries(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError("candidate SHA256SUMS must be a file")
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("candidate SHA256SUMS is malformed") from error
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or not _is_sha256(digest)
            or relative in entries
            or not relative.startswith("./")
        ):
            raise ValueError("candidate SHA256SUMS is malformed")
        entries[relative] = digest
    if not {"./manifest.json", "./predictions.jsonl"}.issubset(entries):
        raise ValueError("candidate SHA256SUMS omits frozen artifacts")
    return entries


def verify_candidate_freeze(
    *, predictions_path: Path, manifest_path: Path, checksums_path: Path
) -> tuple[dict[str, Any], ...]:
    """Recompute all candidate hashes, bindings, identities, and pair order."""

    if not predictions_path.is_file() or not manifest_path.is_file():
        raise ValueError("candidate predictions and manifest must be files")
    prediction_bytes = predictions_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    checksums = _checksum_entries(checksums_path)
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if checksums["./predictions.jsonl"] != prediction_sha:
        raise ValueError("candidate prediction SHA-256 mismatch")
    if checksums["./manifest.json"] != manifest_sha:
        raise ValueError("candidate manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("candidate manifest must be valid JSON") from error
    expected_manifest_fields = {
        "schema_version",
        "run_id",
        "mode",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_image_digest",
        "contract_sha256",
        "contract_file_sha256",
        "predictions_sha256",
        "frozen",
        "completion",
        "reference_answers_available",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("candidate manifest schema mismatch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != CANDIDATE_RUN_ID
        or manifest.get("mode") != "frozen30-paired"
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("runtime_image") != RUNTIME_IMAGE
        or manifest.get("runtime_image_digest") != RUNTIME_IMAGE_DIGEST
        or manifest.get("frozen") is not True
        or manifest.get("reference_answers_available") is not False
    ):
        raise ValueError("candidate manifest identity mismatch")
    if manifest.get("completion") != {"completed": 60, "expected": 60}:
        raise ValueError("candidate freeze must be 60/60 complete")
    if manifest.get("predictions_sha256") != prediction_sha:
        raise ValueError("candidate manifest prediction SHA-256 binding mismatch")

    contract_path = manifest_path.with_name("contract.json")
    if not contract_path.is_file():
        raise ValueError("candidate contract must be a file")
    contract_bytes = contract_path.read_bytes()
    if (
        manifest.get("contract_file_sha256")
        != hashlib.sha256(contract_bytes).hexdigest()
    ):
        raise ValueError("candidate contract file SHA-256 binding mismatch")
    try:
        contract = json.loads(contract_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("candidate contract must be valid JSON") from error
    if not isinstance(contract, dict):
        raise ValueError("candidate contract must be an object")
    if (
        manifest.get("contract_sha256")
        != hashlib.sha256(_canonical_json(contract)).hexdigest()
    ):
        raise ValueError("candidate contract canonical SHA-256 binding mismatch")
    try:
        mode, examples, conditions = _validate_run_contract(contract)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if (
        mode != "frozen30-paired"
        or conditions != [dict(x) for x in CONDITIONS]
        or contract.get("media_derivation") is None
    ):
        raise ValueError("candidate contract provenance mismatch")

    records = _parse_jsonl(prediction_bytes, description="candidate predictions")
    if len(records) != 60:
        raise ValueError("candidate freeze must be 60/60 complete")
    expected_fields = {
        "schema_version",
        "run_id",
        "mode",
        "pair_index",
        "video_id",
        "category",
        "role",
        "condition_id",
        "audio_enabled",
        "request_sha256",
        "model_id",
        "model_revision",
        "runtime_image_digest",
        "status",
        "answer",
        "answer_sha256",
        "error_type",
        "error_message",
        "elapsed_seconds",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair_index, ((video_id, category, role), example) in enumerate(
        zip(FROZEN30_ROWS, examples, strict=True)
    ):
        pair = records[pair_index * 2 : pair_index * 2 + 2]
        if video_id in seen:
            raise ValueError("candidate duplicate pair identity")
        seen.add(video_id)
        predictions: dict[str, dict[str, Any]] = {}
        for record, condition in zip(pair, CONDITIONS, strict=True):
            condition_id = condition["condition_id"]
            answer = record.get("answer")
            elapsed = record.get("elapsed_seconds")
            expected_request = build_chat_request(
                media_path=_NORMALIZED_MEDIA_ROOT / example["media_path"],
                question=example["question"],
                audio_enabled=condition["audio_enabled"],
            )
            if set(record) != expected_fields:
                raise ValueError("candidate prediction schema mismatch")
            if (
                record.get("schema_version") != 1
                or record.get("run_id") != CANDIDATE_RUN_ID
                or record.get("mode") != "frozen30-paired"
                or record.get("pair_index") != pair_index
                or record.get("video_id") != video_id
                or record.get("category") != category
                or record.get("role") != role
                or record.get("condition_id") != condition_id
                or record.get("audio_enabled") is not condition["audio_enabled"]
                or record.get("model_id") != MODEL_ID
                or record.get("model_revision") != MODEL_REVISION
                or record.get("runtime_image_digest") != RUNTIME_IMAGE_DIGEST
                or record.get("status") != "passed"
            ):
                raise ValueError(
                    "candidate prediction pair order or provenance mismatch"
                )
            if (
                record.get("request_sha256")
                != canonical_request_sha256(expected_request)
                or not isinstance(answer, str)
                or not answer.strip()
                or record.get("answer_sha256")
                != hashlib.sha256(answer.encode()).hexdigest()
                or record.get("error_type") is not None
                or record.get("error_message") is not None
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or elapsed < 0
            ):
                raise ValueError("candidate prediction request provenance mismatch")
            predictions[str(condition_id)] = {"completed": True, "answer": answer}
        result.append(
            {
                "row_key": video_id,
                "category": category,
                "role": role,
                "question": example["question"],
                "predictions": predictions,
            }
        )
    if len(seen) != 30:
        raise ValueError("candidate duplicate pair identity")
    return tuple(result)


def _validate_references(
    keys: tuple[str, ...], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    if len(rows) != 30:
        raise ValueError("reference loader must return exactly 30 rows")
    expected_categories = {
        video_id: category for video_id, category, _role in FROZEN30_ROWS
    }
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_key",
            "category",
            "question",
            "reference",
        }:
            raise ValueError("reference row schema mismatch")
        key = row.get("row_key")
        if not isinstance(key, str) or key in by_key:
            raise ValueError("reference row identities must be unique")
        if row.get("category") != expected_categories.get(key):
            raise ValueError("reference stratum mismatch")
        if not all(
            isinstance(row.get(field), str) and bool(row[field].strip())
            for field in ("question", "reference")
        ):
            raise ValueError("reference text must be non-empty")
        by_key[key] = row
    if tuple(by_key) != keys:
        raise ValueError("reference rows must match frozen order")
    return by_key


def build_blind_grading_bundle(
    *,
    qwen_predictions_path: Path,
    candidate_predictions_path: Path,
    candidate_manifest_path: Path,
    candidate_checksums_path: Path,
    reference_loader: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]],
    grader_ids: tuple[str, str],
    shuffle_seeds: tuple[int, int],
) -> Frozen30GradingBundle:
    """Verify both immutable freezes before making references accessible."""

    qwen_rows = verify_qwen_freeze(qwen_predictions_path)
    candidate_rows = verify_candidate_freeze(
        predictions_path=candidate_predictions_path,
        manifest_path=candidate_manifest_path,
        checksums_path=candidate_checksums_path,
    )
    keys = tuple(video_id for video_id, _category, _role in FROZEN30_ROWS)
    references = _validate_references(keys, reference_loader(keys))
    if (
        len(grader_ids) != 2
        or len(set(grader_ids)) != 2
        or not all(isinstance(value, str) and value for value in grader_ids)
    ):
        raise ValueError("exactly two independent grader identities are required")
    if len(shuffle_seeds) != 2 or shuffle_seeds[0] == shuffle_seeds[1]:
        raise ValueError("two distinct shuffle seeds are required")

    private_rows: list[dict[str, Any]] = []
    canonical_keys: list[tuple[str, str]] = []
    item_content: dict[tuple[str, str], dict[str, str]] = {}
    for index, (qwen_row, candidate_row) in enumerate(
        zip(qwen_rows, candidate_rows, strict=True)
    ):
        video_id, category, role = FROZEN30_ROWS[index]
        reference = references[video_id]
        predictions = {
            "qwen12": {"completed": True, "answer": qwen_row["answer"]},
            **candidate_row["predictions"],
        }
        row = {
            "row_key": video_id,
            "category": category,
            "role": role,
            "question": reference["question"],
            "reference": reference["reference"],
            "predictions": predictions,
        }
        private_rows.append(row)
        for condition in _ALL_CONDITIONS:
            key = (video_id, condition)
            canonical_keys.append(key)
            item_content[key] = {
                "question": row["question"],
                "candidate_answer": predictions[condition]["answer"],
                "reference_answer": row["reference"],
            }

    packets: dict[str, dict[str, Any]] = {}
    answer_keys: dict[str, dict[str, tuple[str, str]]] = {}
    prior: list[tuple[str, str]] | None = None
    for packet_index, (grader_id, seed) in enumerate(
        zip(grader_ids, shuffle_seeds, strict=True), start=1
    ):
        order = list(canonical_keys)
        random.Random(seed).shuffle(order)
        if order == prior:
            order = order[1:] + order[:1]
        prior = order
        packet_id = "blind-%d-%s" % (
            packet_index,
            hashlib.sha256(f"{packet_index}:{grader_id}:{seed}".encode()).hexdigest()[
                :12
            ],
        )
        answer_key = {}
        items = []
        for item_index, key in enumerate(order, start=1):
            item_id = f"item-{item_index:03d}"
            answer_key[item_id] = key
            items.append({"item_id": item_id, **item_content[key]})
        packets[grader_id] = {
            "packet_id": packet_id,
            "grader_id": grader_id,
            "rubric": dict(_RUBRIC),
            "items": items,
        }
        answer_keys[grader_id] = answer_key
    return Frozen30GradingBundle(
        packets=packets,
        _answer_keys=answer_keys,
        _canonical_keys=tuple(canonical_keys),
        _item_content=item_content,
        _rows=tuple(private_rows),
    )


def apply_decision_rule(
    *,
    improvements: int,
    regressions: int,
    category_regressions_exceed_improvements: bool,
    visual_control_regressions: int,
    exact_match_delta_rows: float,
    token_f1_delta: float,
    meteor_delta: float,
    operational: OperationalEvidence,
) -> dict[str, Any]:
    """Apply the preregistered semantic and operational boundaries verbatim."""

    semantic = (
        improvements >= 6
        and regressions <= 2
        and not category_regressions_exceed_improvements
        and visual_control_regressions == 0
        and exact_match_delta_rows >= -1
        and token_f1_delta >= -0.02 - 1e-12
        and meteor_delta >= -0.02 - 1e-12
    )
    operational_passed = (
        operational.completed == operational.expected == 60
        and operational.duration_minutes <= 60
        and operational.compute_cost_usd <= 1.25
        and operational.artifacts_verified
        and not operational.privacy_or_audit_gap
    )
    if semantic and operational_passed:
        disposition = "Current default"
    elif semantic:
        disposition = "Routed candidate"
    else:
        disposition = "Not promoted under the tested condition"
    return {
        "semantic_gate_passed": semantic,
        "operational_gate_passed": operational_passed,
        "disposition": disposition,
    }


def score_replacement(
    bundle: Frozen30GradingBundle,
    resolution: GradeResolution,
    *,
    meteor_fn: Callable[[str, str], float],
    operational: OperationalEvidence,
) -> dict[str, Any]:
    """Produce private per-row and aggregate metrics plus the disposition."""

    if resolution.status != "final" or resolution.final_grades is None:
        raise ValueError("finalized blind grades are required before scoring")
    if set(resolution.final_grades) != set(bundle._canonical_keys) or any(
        grade not in _GRADES for grade in resolution.final_grades.values()
    ):
        raise ValueError("final grades must cover all 90 frozen answers")
    examples = []
    condition_values: dict[str, dict[str, list[float]]] = {
        condition: {"exact_match": [], "token_f1": [], "meteor": []}
        for condition in _ALL_CONDITIONS
    }
    category_values: dict[str, dict[str, dict[str, list[float]]]] = {
        category: {
            condition: {"exact_match": [], "token_f1": [], "meteor": []}
            for condition in _ALL_CONDITIONS
        }
        for _video_id, category, _role in FROZEN30_ROWS
    }
    grade_counts = {
        condition: {"C": 0, "P": 0, "W": 0} for condition in _ALL_CONDITIONS
    }
    category_comparisons: dict[str, dict[str, int]] = {
        category: {"improvements": 0, "regressions": 0, "ties": 0}
        for _video_id, category, _role in FROZEN30_ROWS
    }
    audio_control_by_category = {
        category: {"improvements": 0, "regressions": 0, "ties": 0}
        for _video_id, category, _role in FROZEN30_ROWS
    }
    improvements = regressions = visual_regressions = 0
    for row in bundle._rows:
        metrics = {}
        for condition in _ALL_CONDITIONS:
            answer = row["predictions"][condition]["answer"]
            values = {
                "exact_match": normalized_exact_match(answer, row["reference"]),
                "token_f1": token_f1(answer, row["reference"]),
                "meteor": float(meteor_fn(answer, row["reference"])),
            }
            metrics[condition] = values
            for name, value in values.items():
                condition_values[condition][name].append(value)
                category_values[row["category"]][condition][name].append(value)
            grade_counts[condition][
                resolution.final_grades[(row["row_key"], condition)]
            ] += 1
        qwen_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "qwen12")]]
        on_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-on")]]
        off_rank = _GRADE_RANK[resolution.final_grades[(row["row_key"], "audio-off")]]
        comparison = "ties"
        if on_rank > qwen_rank:
            comparison = "improvements"
            improvements += 1
        elif on_rank < qwen_rank:
            comparison = "regressions"
            regressions += 1
            if row["role"] == "control":
                visual_regressions += 1
        category_comparisons[row["category"]][comparison] += 1
        audio_comparison = "ties"
        if on_rank > off_rank:
            audio_comparison = "improvements"
        elif on_rank < off_rank:
            audio_comparison = "regressions"
        audio_control_by_category[row["category"]][audio_comparison] += 1
        examples.append({**row, "automatic_metrics": metrics})
    aggregate = {
        condition: {name: fmean(values) for name, values in metrics.items()}
        for condition, metrics in condition_values.items()
    }
    category_metrics = {
        category: {
            condition: {name: fmean(values) for name, values in metrics.items()}
            for condition, metrics in conditions.items()
        }
        for category, conditions in category_values.items()
    }
    decision = apply_decision_rule(
        improvements=improvements,
        regressions=regressions,
        category_regressions_exceed_improvements=any(
            values["regressions"] > values["improvements"]
            for values in category_comparisons.values()
        ),
        visual_control_regressions=visual_regressions,
        exact_match_delta_rows=(
            sum(condition_values["audio-on"]["exact_match"])
            - sum(condition_values["qwen12"]["exact_match"])
        ),
        token_f1_delta=(
            aggregate["audio-on"]["token_f1"] - aggregate["qwen12"]["token_f1"]
        ),
        meteor_delta=(aggregate["audio-on"]["meteor"] - aggregate["qwen12"]["meteor"]),
        operational=operational,
    )
    return {
        "completion": {"completed": operational.completed, "expected": 60},
        "conditions": aggregate,
        "grade_counts": grade_counts,
        "category_metrics": category_metrics,
        "qwen12_vs_audio_on": {
            "improvements": improvements,
            "regressions": regressions,
            "ties": 30 - improvements - regressions,
            "by_category": category_comparisons,
        },
        "audio_off_vs_audio_on": {"by_category": audio_control_by_category},
        "examples": examples,
        **decision,
    }
