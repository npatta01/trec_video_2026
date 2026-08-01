"""Label-isolated three-condition TimeLens2 comparison contracts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .timelens2_selector import (
    _video_duration,
    build_hybrid_timestamps,
    canonical_sha256,
    normalize_intervals,
)

CONDITIONS = ("uniform-12", "hybrid-12", "combined-18")
CONDITION_FRAME_COUNTS = {
    "uniform-12": 12,
    "hybrid-12": 12,
    "combined-18": 18,
}
REQUEST_SCHEMA_VERSION = "timelens2-three-condition-requests-v1"
FREEZE_SCHEMA_VERSION = "timelens2-three-condition-inference-freeze-v1"
EXPECTED_ROWS = 30
EXPECTED_ANSWERS = EXPECTED_ROWS * len(CONDITIONS)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_LABEL_KEYS = {
    "annotation",
    "answer",
    "answers",
    "correct_answer",
    "incorrect_answer",
    "incorrect_answers",
    "reference",
    "references",
    "grade",
    "judgment",
    "prediction",
    "predictions",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reject_label_keys(value: object, *, location: str = "source") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label leakage in {location}: {normalized}")
            _reject_label_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_keys(child, location=f"{location}[{index}]")


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("media must be a non-empty relative path")
    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError("media must resolve beneath dataset_root")
    if not candidate.is_file():
        raise ValueError("media is not a file")
    return candidate


def prepare_frozen30_source(
    *,
    source_path: Path,
    dataset_root: Path,
    output_path: Path,
    container_media_root: Path,
    expected_source_sha256: str,
    duration_provider: Callable[[Path], float] = _video_duration,
) -> dict[str, object]:
    """Bind the exact label-free frozen-30 source to media bytes and duration."""

    source_path = source_path.resolve(strict=True)
    dataset_root = dataset_root.resolve(strict=True)
    if not _SHA256.fullmatch(expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256")
    source_sha256 = _file_sha256(source_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source does not match the frozen label-free source")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _reject_label_keys(source)
    if not isinstance(source, dict):
        raise ValueError("frozen source must be an object")
    examples = source.get("examples")
    if not isinstance(examples, list) or len(examples) != EXPECTED_ROWS:
        raise ValueError("frozen source must contain exactly 30 examples")

    requests: list[dict[str, object]] = []
    seen: set[str] = set()
    required_example_keys = {"category", "media", "question", "video_id"}
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict) or set(example) != required_example_keys:
            raise ValueError(f"example {index} must contain only label-free fields")
        video_id = example["video_id"]
        question = example["question"]
        category = example["category"]
        if not isinstance(video_id, str) or not video_id or video_id in seen:
            raise ValueError("frozen source requires 30 distinct video identities")
        seen.add(video_id)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"example {index} has invalid question")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"example {index} has invalid category")
        media = _inside(dataset_root, example["media"])
        duration = float(duration_provider(media))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"example {index} has invalid duration")
        identity: dict[str, object] = {
            "request_id": f"frozen30-{index:02d}",
            "video_id": video_id,
            "category": category,
            "query": question.strip(),
            "duration_seconds": duration,
            "media_sha256": _file_sha256(media),
            "media_path": str(container_media_root / media.name),
            "source_sha256": source_sha256,
        }
        requests.append({**identity, "request_sha256": canonical_sha256(identity)})
    prepared: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "requests": requests,
    }
    _write_json(output_path, prepared)
    return prepared


def _uniform_midpoints(duration: float, count: int) -> list[float]:
    return [duration * (index + 0.5) / count for index in range(count)]


def _global_slot(prefix: str, index: int, timestamp: float) -> dict[str, object]:
    return {
        "slot_id": f"{prefix}-{index:02d}",
        "role": "global",
        "requested_timestamp_seconds": timestamp,
    }


def build_three_condition_recipes(
    *,
    duration_seconds: float,
    normalized_intervals: Sequence[Sequence[float]],
) -> dict[str, dict[str, object]]:
    """Build uniform-12, hybrid-12, and uniform-plus-local combined-18."""

    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    intervals = normalize_intervals(normalized_intervals)
    hybrid = build_hybrid_timestamps(
        duration_seconds=duration,
        normalized_intervals=intervals,
    )
    uniform_slots = [
        _global_slot("uniform", index, timestamp)
        for index, timestamp in enumerate(_uniform_midpoints(duration, 12), start=1)
    ]
    hybrid_global_slots = [
        _global_slot("hybrid-global", index, float(timestamp))
        for index, timestamp in enumerate(hybrid["global"], start=1)
    ]
    local_slots: list[dict[str, object]] = []
    local_index = 0
    offset = 0
    for interval_index, ((start, end), count) in enumerate(
        zip(intervals, hybrid["local_counts"], strict=True)
    ):
        for timestamp in hybrid["local"][offset : offset + count]:
            local_index += 1
            local_slots.append(
                {
                    "slot_id": f"timelens-local-{local_index:02d}",
                    "role": "local",
                    "interval_index": interval_index,
                    "interval_seconds": [start, end],
                    "requested_timestamp_seconds": float(timestamp),
                }
            )
        offset += count
    if len(local_slots) != 6:
        raise ValueError("TimeLens local allocation must contain six slots")

    def ordered(slots: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        return [
            dict(slot)
            for slot in sorted(
                slots,
                key=lambda row: (
                    float(row["requested_timestamp_seconds"]),
                    str(row["slot_id"]),
                ),
            )
        ]

    return {
        "uniform-12": {"frame_count": 12, "slots": ordered(uniform_slots)},
        "hybrid-12": {
            "frame_count": 12,
            "slots": ordered(hybrid_global_slots + local_slots),
            "interval_padding_seconds": 0.0,
        },
        "combined-18": {
            "frame_count": 18,
            "slots": ordered(uniform_slots + local_slots),
            "interval_padding_seconds": 0.0,
        },
    }


def _default_frame_decoder(media: Path, timestamp: float, output: Path) -> None:
    raise RuntimeError("frame_decoder must be supplied by the runtime")


def extract_three_condition_frames(
    *,
    video_id: str,
    media_path: Path,
    recipes: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    frame_decoder: Callable[[Path, float, Path], None] = _default_frame_decoder,
    frame_timestamps_provider: Callable[[Path], Sequence[float]],
) -> dict[str, object]:
    """Decode shared slots once and enforce distinct images within conditions."""

    if set(recipes) != set(CONDITIONS):
        raise ValueError("three-condition extraction requires every condition")
    slot_definitions: dict[str, dict[str, object]] = {}
    slot_conditions: dict[str, set[str]] = {}
    condition_slot_ids: dict[str, list[str]] = {}
    for condition in CONDITIONS:
        recipe = recipes[condition]
        slots = recipe.get("slots")
        expected = CONDITION_FRAME_COUNTS[condition]
        if not isinstance(slots, list) or len(slots) != expected:
            raise ValueError(f"{condition} requires {expected} slots")
        ids: list[str] = []
        for raw_slot in slots:
            if not isinstance(raw_slot, Mapping):
                raise ValueError(f"{condition} slot must be an object")
            slot = dict(raw_slot)
            slot_id = slot.get("slot_id")
            if not isinstance(slot_id, str) or not slot_id:
                raise ValueError(f"{condition} slot_id is invalid")
            existing = slot_definitions.setdefault(slot_id, slot)
            if existing != slot:
                raise ValueError(f"shared slot {slot_id} has conflicting definitions")
            ids.append(slot_id)
            slot_conditions.setdefault(slot_id, set()).add(condition)
        if len(set(ids)) != expected:
            raise ValueError(f"{condition} slot identities must be distinct")
        condition_slot_ids[condition] = ids

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_hashes = {condition: set() for condition in CONDITIONS}
    resolved_rows: dict[str, dict[str, object]] = {}
    source_timestamps: list[float] | None = None

    def decode(slot_id: str, timestamp: float) -> dict[str, object]:
        output = output_dir / f"{slot_id}.png"
        frame_decoder(media_path, timestamp, output)
        if not output.is_file():
            raise RuntimeError(f"decoder did not create {slot_id}")
        slot = slot_definitions[slot_id]
        requested = float(slot["requested_timestamp_seconds"])
        return {
            "slot_id": slot_id,
            "role": slot["role"],
            "requested_timestamp_seconds": requested,
            "timestamp_seconds": timestamp,
            "resolution": (
                "requested"
                if timestamp == requested
                else "nearest-unused-source-pts"
            ),
            "path": str(output.resolve()),
            "sha256": _file_sha256(output),
        }

    processing_order = sorted(
        slot_definitions,
        key=lambda slot_id: (
            slot_definitions[slot_id].get("role") == "local",
            float(slot_definitions[slot_id]["requested_timestamp_seconds"]),
            slot_id,
        ),
    )
    for slot_id in processing_order:
        slot = slot_definitions[slot_id]
        requested = float(slot["requested_timestamp_seconds"])
        row = decode(slot_id, requested)
        conditions = slot_conditions[slot_id]

        def conflicts(candidate: Mapping[str, object]) -> bool:
            return any(
                candidate["sha256"] in selected_hashes[condition]
                for condition in conditions
            )

        if conflicts(row):
            if slot.get("role") != "local":
                raise ValueError("global frame collision within a condition")
            interval = slot.get("interval_seconds")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("local collision lacks interval provenance")
            start, end = float(interval[0]), float(interval[1])
            if source_timestamps is None:
                source_timestamps = sorted(
                    {
                        float(value)
                        for value in frame_timestamps_provider(media_path)
                        if math.isfinite(float(value)) and float(value) >= 0
                    }
                )
            candidates = sorted(
                (
                    value
                    for value in source_timestamps
                    if start < value < end and value != requested
                ),
                key=lambda value: (abs(value - requested), value),
            )
            for candidate in candidates:
                candidate_row = decode(slot_id, candidate)
                if not conflicts(candidate_row):
                    row = candidate_row
                    break
            else:
                raise ValueError(
                    "local frame collision has no unused source frame inside interval"
                )
        resolved_rows[slot_id] = row
        for condition in conditions:
            selected_hashes[condition].add(str(row["sha256"]))

    condition_rows = {
        condition: [dict(resolved_rows[slot_id]) for slot_id in condition_slot_ids[condition]]
        for condition in CONDITIONS
    }
    for condition, rows in condition_rows.items():
        if len({row["sha256"] for row in rows}) != CONDITION_FRAME_COUNTS[condition]:
            raise ValueError(f"{condition} decoded frames must be distinct")
    return {
        "schema_version": "timelens2-three-condition-frame-manifest-v1",
        "video_id": video_id,
        "media_sha256": _file_sha256(media_path),
        "frames": [dict(resolved_rows[slot_id]) for slot_id in processing_order],
        "conditions": condition_rows,
        "resolved_local_collisions": sum(
            row["resolution"] == "nearest-unused-source-pts"
            for row in resolved_rows.values()
        ),
    }


def freeze_three_condition_inference(
    *,
    requests: Sequence[Mapping[str, object]],
    raw_results: Sequence[Mapping[str, object]],
    normalized_results: Sequence[Mapping[str, object]],
    recipes: Sequence[Mapping[str, object]],
    frame_manifests: Sequence[Mapping[str, object]],
    answers: Sequence[Mapping[str, object]],
    output_path: Path,
) -> dict[str, object]:
    """Freeze a complete 30-row, 90-answer label-free inference result."""

    if len(requests) != EXPECTED_ROWS:
        raise ValueError("freeze requires exactly 30 requests")
    for name, rows in (
        ("raw selector", raw_results),
        ("normalized selector", normalized_results),
        ("recipe", recipes),
        ("frame manifest", frame_manifests),
    ):
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"freeze requires exactly 30 {name} rows")
    if len(answers) != EXPECTED_ANSWERS:
        raise ValueError("freeze requires 90 complete answers")

    request_index = {str(row.get("video_id")): row for row in requests}
    if len(request_index) != EXPECTED_ROWS or "" in request_index:
        raise ValueError("freeze requires 30 distinct request identities")
    expected_ids = set(request_index)
    for name, rows in (
        ("raw", raw_results),
        ("normalized", normalized_results),
        ("recipe", recipes),
        ("frame", frame_manifests),
    ):
        index = {str(row.get("video_id")): row for row in rows}
        if set(index) != expected_ids:
            raise ValueError(f"{name} identities do not match requests")
    recipe_index = {str(row["video_id"]): row for row in recipes}
    frame_index = {str(row["video_id"]): row for row in frame_manifests}
    for video_id in expected_ids:
        for owner, index in (("recipe", recipe_index), ("frame", frame_index)):
            conditions = index[video_id].get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
                raise ValueError(f"{video_id} {owner} conditions are incomplete")
            for condition, count in CONDITION_FRAME_COUNTS.items():
                value = conditions[condition]
                rows = value.get("slots") if owner == "recipe" and isinstance(value, Mapping) else value
                if not isinstance(rows, list) or len(rows) != count:
                    raise ValueError(f"{video_id} {condition} has wrong frame count")
                if owner == "frame":
                    hashes = [row.get("sha256") for row in rows if isinstance(row, Mapping)]
                    if len(hashes) != count or len(set(hashes)) != count:
                        raise ValueError(f"{video_id} {condition} frames are not distinct")

    answer_index: dict[tuple[str, str], Mapping[str, object]] = {}
    for answer in answers:
        key = (str(answer.get("video_id")), str(answer.get("condition_id")))
        if key in answer_index or key[0] not in expected_ids or key[1] not in CONDITIONS:
            raise ValueError(f"unexpected or duplicate answer pair: {key}")
        request = request_index[key[0]]
        if answer.get("request_sha256") != request.get("request_sha256"):
            raise ValueError(f"answer request hash mismatch for {key}")
        if answer.get("status") != "passed" or not str(answer.get("answer", "")).strip():
            raise ValueError(f"answer did not pass for {key}")
        answer_index[key] = answer
    expected_keys = {(video_id, condition) for video_id in expected_ids for condition in CONDITIONS}
    if set(answer_index) != expected_keys:
        raise ValueError("freeze requires 90 complete answers")
    freeze: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "frozen",
        "label_access": "permitted_after_verified_freeze",
        "conditions": list(CONDITIONS),
        "completion": {
            "videos": {"accepted": EXPECTED_ROWS, "required": EXPECTED_ROWS},
            "answers": {"accepted": EXPECTED_ANSWERS, "required": EXPECTED_ANSWERS},
        },
        "requests_sha256": canonical_sha256(list(requests)),
        "raw_results_sha256": canonical_sha256(list(raw_results)),
        "normalized_results_sha256": canonical_sha256(list(normalized_results)),
        "recipes_sha256": canonical_sha256(list(recipes)),
        "frame_manifests_sha256": canonical_sha256(list(frame_manifests)),
        "answers_sha256": canonical_sha256(list(answers)),
    }
    _write_json(output_path, freeze)
    return freeze


def build_blind_grading_packet(
    *,
    freeze: Mapping[str, object],
    answers: Sequence[Mapping[str, object]],
    references: Mapping[str, str],
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create answer candidates without condition identities and a separate key."""

    completion = freeze.get("completion")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("status") != "frozen"
        or not isinstance(completion, Mapping)
        or completion.get("answers") != {"accepted": 90, "required": 90}
    ):
        raise ValueError("blind grading requires a verified 30/90 freeze")
    answer_index = {
        (str(row.get("video_id")), str(row.get("condition_id"))): row
        for row in answers
    }
    video_ids = sorted(references)
    if len(video_ids) != EXPECTED_ROWS:
        raise ValueError("grading requires exactly 30 references")
    if set(answer_index) != {(video_id, condition) for video_id in video_ids for condition in CONDITIONS}:
        raise ValueError("grading answers do not match references and conditions")
    rng = random.Random(seed)
    packet_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, str]] = []
    for row_index, video_id in enumerate(video_ids, start=1):
        candidates = [
            {
                "condition_id": condition,
                "answer": str(answer_index[(video_id, condition)]["answer"]),
            }
            for condition in CONDITIONS
        ]
        rng.shuffle(candidates)
        public_candidates = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"row-{row_index:02d}-candidate-{candidate_index}"
            public_candidates.append(
                {"candidate_id": candidate_id, "answer": candidate["answer"]}
            )
            key_rows.append(
                {
                    "candidate_id": candidate_id,
                    "video_id": video_id,
                    "condition_id": str(candidate["condition_id"]),
                }
            )
        packet_rows.append(
            {
                "row_id": f"row-{row_index:02d}",
                "video_id": video_id,
                "reference": references[video_id],
                "candidates": public_candidates,
            }
        )
    return (
        {"schema_version": "timelens2-blind-grading-packet-v1", "rows": packet_rows},
        {"schema_version": "timelens2-blind-grading-key-v1", "rows": key_rows},
    )


def score_blind_judgments(
    *,
    packet: Mapping[str, object],
    key: Mapping[str, object],
    judgments: Mapping[str, str],
) -> dict[str, object]:
    """Reveal complete C/P/W judgments and summarize paired transitions."""

    packet_rows = packet.get("rows")
    key_rows = key.get("rows")
    if not isinstance(packet_rows, list) or len(packet_rows) != EXPECTED_ROWS:
        raise ValueError("judgment packet must contain 30 rows")
    if not isinstance(key_rows, list) or len(key_rows) != EXPECTED_ANSWERS:
        raise ValueError("judgment key must contain 90 rows")
    mapping: dict[str, tuple[str, str]] = {}
    for row in key_rows:
        if not isinstance(row, Mapping):
            raise ValueError("judgment key row must be an object")
        candidate_id = str(row.get("candidate_id"))
        mapping[candidate_id] = (str(row.get("video_id")), str(row.get("condition_id")))
    if set(judgments) != set(mapping) or any(value not in {"C", "P", "W"} for value in judgments.values()):
        raise ValueError("judgments must provide exactly 90 C/P/W grades")
    grade_by_pair = {mapping[candidate]: grade for candidate, grade in judgments.items()}
    counts = {
        condition: {
            grade: sum(
                value == grade
                for (video_id, candidate_condition), value in grade_by_pair.items()
                if candidate_condition == condition
            )
            for grade in ("C", "P", "W")
        }
        for condition in CONDITIONS
    }
    ordinal = {"W": 0, "P": 1, "C": 2}
    paired: dict[str, dict[str, int]] = {}
    video_ids = sorted({video_id for video_id, _ in grade_by_pair})
    for condition in CONDITIONS[1:]:
        improved = regressed = tied = 0
        for video_id in video_ids:
            baseline = ordinal[grade_by_pair[(video_id, "uniform-12")]]
            treatment = ordinal[grade_by_pair[(video_id, condition)]]
            if treatment > baseline:
                improved += 1
            elif treatment < baseline:
                regressed += 1
            else:
                tied += 1
        paired[condition] = {
            "improved": improved,
            "regressed": regressed,
            "tied": tied,
        }
    return {
        "schema_version": "timelens2-blind-grading-result-v1",
        "completion": {"accepted": len(judgments), "required": EXPECTED_ANSWERS},
        "conditions": counts,
        "paired": paired,
    }
