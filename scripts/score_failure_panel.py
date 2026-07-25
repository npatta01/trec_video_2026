#!/usr/bin/env python3
"""Score the failure panel across a label-isolated two-container boundary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import string
import subprocess
from statistics import fmean
from typing import Any, Callable, Literal, Mapping, NamedTuple, Sequence
from urllib.parse import parse_qs, urlparse


SCORER_DIGEST = (
    "sha256:effff2801d7677aa4894a641adc5793d6e5072a8cec71817bf925574862c88de"
)
SCORER_IMAGE = f"localhost/trec-vqa-gate-qwen-omni@{SCORER_DIGEST}"
INSIDE_SCRIPT = Path("/opt/score_failure_panel.py")
CANONICAL_ROOT = Path("/home/npatta01/data/competitions/trec_video_2026")
CANONICAL_DATA = CANONICAL_ROOT / "data"
Action = Literal["freeze", "evaluate"]
INFERENCE_MANIFEST_SHA256 = (
    "d2c5515b8c7398f326931e2f993f3fd6eac8e70d9ef01d1a1bd4afcf3f6792e8"
)
REFERENCE_MANIFEST_SHA256 = (
    "ad76ec57b964bf391f763e7f82b1671475cddaa90814727d9c40ee7c6310ccbd"
)
NLTK_VERSION = "3.9.2"
PUNKT_PATH = Path("/opt/nltk_data/tokenizers/punkt_tab/english")
PUNKT_SHA256 = "f1034c543e1f11ddc0928f743e15fd11958b369bca05df8917eba55e9cafcd29"
WORDNET_PATH = Path("/opt/nltk_data/corpora/wordnet.zip")
WORDNET_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_DIGEST_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")

FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "answer_key",
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
ALLOWED_CANDIDATE_KEYS = frozenset(
    {
        "answer",
        "condition_id",
        "model_id",
        "model_revision",
        "runtime_image",
        "source_sha256",
        "status",
        "video_id",
    }
)


class Layout(NamedTuple):
    repo_root: Path
    data_root: Path
    output_dir: Path
    candidate_path: Path
    inference_manifest: Path
    reference_manifest: Path
    script_path: Path


class Mount(NamedTuple):
    source: Path
    target: Path
    mode: Literal["ro", "rw"]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Stale atomic-write temporary exists: {temporary}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, indent: int | None = None
) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"Expected resource directory: {path}")
    payload = b"".join(
        relative.as_posix().encode("utf-8")
        + b"\0"
        + sha256_path(path / relative).encode("ascii")
        + b"\0"
        for relative in sorted(
            (item.relative_to(path) for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_hash(path: Path, expected: str, name: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(
            f"{name} manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _attest_scorer_runtime() -> dict[str, Any]:
    if Path(__file__).resolve() != INSIDE_SCRIPT:
        raise ValueError("Scoring can run only from the mounted frozen scorer path")
    if os.environ.get("VQA_RUNTIME_IMAGE") != SCORER_IMAGE:
        raise ValueError("Scorer runtime image identity is missing or incorrect")
    version = importlib.metadata.version("nltk")
    if version != NLTK_VERSION:
        raise ValueError(f"Expected NLTK {NLTK_VERSION}, got {version}")
    punkt = _directory_sha256(PUNKT_PATH)
    wordnet = sha256_path(WORDNET_PATH)
    if punkt != PUNKT_SHA256 or wordnet != WORDNET_SHA256:
        raise ValueError("Pinned NLTK resource bytes do not match")
    return {
        "nltk_version": version,
        "punkt_path": str(PUNKT_PATH),
        "punkt_sha256": punkt,
        "scorer_image": SCORER_IMAGE,
        "wordnet_path": str(WORDNET_PATH),
        "wordnet_sha256": wordnet,
    }


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {path}") from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL: {path}") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Prediction JSONL must contain object rows")
    return rows


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _load_jsonl(path)
    except OSError as error:
        raise ValueError(f"Invalid prediction source: {path}") from error
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(row, dict) for row in payload)
    ):
        raise ValueError(f"Prediction source must be a non-empty object list: {path}")
    return payload


def _manifest_examples(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    examples = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Manifest has no examples: {path}")
    if not all(isinstance(example, dict) for example in examples):
        raise ValueError("Manifest examples must be objects")
    ids = [example.get("video_id") for example in examples]
    if any(not isinstance(video_id, str) or not video_id for video_id in ids):
        raise ValueError("Manifest video IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest video IDs must be unique")
    return examples


def _reject_labels(value: Any, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_LABEL_KEYS:
                raise ValueError(f"label-bearing field {key!r} at {location}")
            _reject_labels(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_labels(child, f"{location}[{index}]")


def freeze_candidates(
    candidate_path: Path,
    inference_manifest: Path,
    output_dir: Path,
    *,
    runtime_provenance: Mapping[str, Any],
    scorer_script_path: Path,
) -> dict[str, Any]:
    """Validate and freeze label-free candidate predictions."""

    _validate_manifest_hash(
        inference_manifest,
        INFERENCE_MANIFEST_SHA256,
        "Inference",
    )
    examples = _manifest_examples(inference_manifest)
    _reject_labels(examples)
    panel_ids = [str(example["video_id"]) for example in examples]
    panel_set = set(panel_ids)
    rows = _load_jsonl(candidate_path)
    _reject_labels(rows)
    keys: set[tuple[str, str]] = set()
    condition_rows: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        extra = set(row) - ALLOWED_CANDIDATE_KEYS
        if extra:
            raise ValueError(
                f"Unsupported candidate fields at row {index}: {sorted(extra)}"
            )
        condition = row.get("condition_id")
        video_id = row.get("video_id")
        answer = row.get("answer")
        status = row.get("status")
        if not isinstance(condition, str) or not condition:
            raise ValueError("Every candidate needs a condition_id")
        if video_id not in panel_set:
            raise ValueError(f"Candidate video is outside frozen panel: {video_id}")
        if status != "passed" or not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Candidate is not a non-empty passed answer: {condition}/{video_id}"
            )
        for field in ("model_id", "model_revision", "runtime_image", "source_sha256"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(
                    f"Candidate provenance field is missing: {condition}/{video_id}/{field}"
                )
        if not _REVISION.fullmatch(str(row["model_revision"])):
            raise ValueError(f"Candidate model revision is not immutable: {condition}")
        if not _DIGEST_IMAGE.fullmatch(str(row["runtime_image"])):
            raise ValueError(
                f"Candidate runtime image is not digest-pinned: {condition}"
            )
        if not _SHA256.fullmatch(str(row["source_sha256"])):
            raise ValueError(f"Candidate source hash is invalid: {condition}")
        identity = (condition, str(video_id))
        if identity in keys:
            raise ValueError(f"Duplicate candidate: {condition}/{video_id}")
        keys.add(identity)
        condition_rows.setdefault(condition, []).append(str(video_id))
    for condition, video_ids in condition_rows.items():
        if video_ids != panel_ids:
            raise ValueError(
                f"Every aggregate condition must contain the exact ordered panel: {condition}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_script_path = output_dir / "frozen-scorer.py"
    script_bytes = scorer_script_path.read_bytes()
    _atomic_write_bytes(frozen_script_path, script_bytes)
    selected_path = output_dir / "selected-predictions.jsonl"
    _atomic_write_bytes(selected_path, canonical_jsonl(rows))
    record = {
        "candidate_prediction_sha256": sha256_path(candidate_path),
        "condition_counts": {
            condition: len(video_ids)
            for condition, video_ids in sorted(condition_rows.items())
        },
        "inference_manifest_sha256": sha256_path(inference_manifest),
        "runtime": dict(runtime_provenance),
        "scorer_script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "selected_prediction_sha256": sha256_path(selected_path),
        "selected_rows": len(rows),
    }
    freeze_path = output_dir / "prediction-freeze.json"
    _atomic_write_json(freeze_path, record)
    return record


def assemble_candidates(
    sources: Sequence[tuple[str, Path]],
    inference_manifest: Path,
    candidate_path: Path,
) -> list[dict[str, Any]]:
    """Project saved label-free model outputs into one minimal candidate file."""

    _validate_manifest_hash(
        inference_manifest,
        INFERENCE_MANIFEST_SHA256,
        "Inference",
    )
    panel_ids = [
        str(example["video_id"]) for example in _manifest_examples(inference_manifest)
    ]
    panel_set = set(panel_ids)
    panel_position = {video_id: index for index, video_id in enumerate(panel_ids)}
    assembled: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    for condition, source in sources:
        if not condition or condition in seen_conditions:
            raise ValueError(f"Duplicate or empty source condition: {condition!r}")
        seen_conditions.add(condition)
        source_rows = _load_prediction_rows(source)
        _reject_labels(source_rows)
        selected = [
            row
            for row in source_rows
            if row.get("condition_id") == condition and row.get("video_id") in panel_set
        ]
        selected.sort(key=lambda row: panel_position[str(row["video_id"])])
        if not selected:
            raise ValueError(f"No frozen-panel predictions found for {condition}")
        if len({str(row["video_id"]) for row in selected}) != len(selected):
            raise ValueError(f"Duplicate source predictions for {condition}")
        source_digest = sha256_path(source)
        for row in selected:
            assembled.append(
                {
                    "answer": row.get("answer"),
                    "condition_id": condition,
                    "model_id": row.get("model_id"),
                    "model_revision": row.get("model_revision"),
                    "runtime_image": row.get("runtime_image"),
                    "source_sha256": source_digest,
                    "status": row.get("status"),
                    "video_id": row.get("video_id"),
                }
            )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_temporary = candidate_path.with_name(candidate_path.name + ".tmp")
    if candidate_temporary.exists():
        raise ValueError(f"Stale candidate temporary exists: {candidate_temporary}")
    candidate_temporary.write_bytes(canonical_jsonl(assembled))
    # Reuse the freeze validator without creating a freeze: validate the exact
    # minimal schema, label absence, order, passed status, and answer presence.
    temporary_output = candidate_path.parent / ".candidate-validation"
    if temporary_output.exists():
        raise ValueError("Candidate validation directory already exists")
    try:
        freeze_candidates(
            candidate_temporary,
            inference_manifest,
            temporary_output,
            runtime_provenance={"assembly_validation": True},
            scorer_script_path=Path(__file__).resolve(),
        )
        os.replace(candidate_temporary, candidate_path)
    finally:
        if candidate_temporary.exists():
            candidate_temporary.unlink()
        for name in (
            "selected-predictions.jsonl",
            "prediction-freeze.json",
            "frozen-scorer.py",
        ):
            path = temporary_output / name
            if path.exists():
                path.unlink()
        if temporary_output.exists():
            temporary_output.rmdir()
    return assembled


def _validate_freeze(
    output_dir: Path,
    inference_manifest: Path,
    *,
    runtime_provenance: Mapping[str, Any],
    scorer_script_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    _validate_manifest_hash(
        inference_manifest,
        INFERENCE_MANIFEST_SHA256,
        "Inference",
    )
    selected_path = output_dir / "selected-predictions.jsonl"
    freeze_path = output_dir / "prediction-freeze.json"
    rows = _load_jsonl(selected_path)
    selected_bytes = selected_path.read_bytes()
    freeze_bytes = freeze_path.read_bytes()
    if canonical_jsonl(rows) != selected_bytes:
        raise ValueError("Prediction freeze selected bytes are not canonical")
    try:
        record = json.loads(freeze_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prediction freeze record is invalid") from error
    expected_counts = Counter(str(row.get("condition_id")) for row in rows)
    scorer_hash = sha256_path(scorer_script_path)
    if (
        record.get("selected_prediction_sha256") != sha256_path(selected_path)
        or record.get("selected_rows") != len(rows)
        or record.get("condition_counts") != dict(sorted(expected_counts.items()))
        or record.get("inference_manifest_sha256") != sha256_path(inference_manifest)
        or record.get("runtime") != dict(runtime_provenance)
        or record.get("scorer_script_sha256") != scorer_hash
    ):
        raise ValueError("Prediction freeze record does not match frozen bytes")
    expected_freeze = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    if freeze_bytes != expected_freeze:
        raise ValueError("Prediction freeze record bytes are not canonical")
    return rows, record, selected_bytes, freeze_bytes


def _normalize(answer: str) -> str:
    lowered = answer.lower()
    no_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    tokens = [
        token for token in no_punctuation.split() if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalize(prediction).split()
    expected = _normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _default_meteor(prediction: str, reference: str) -> float:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    return float(meteor_score([word_tokenize(reference)], word_tokenize(prediction)))


def evaluate_frozen(
    output_dir: Path,
    inference_manifest: Path,
    reference_manifest: Path,
    annotations: Sequence[Path],
    *,
    meteor_fn: Callable[[str, str], float] = _default_meteor,
    runtime_provenance: Mapping[str, Any],
    scorer_script_path: Path,
) -> dict[str, Any]:
    """Join immutable predictions to labels and compute panel diagnostics."""

    _validate_manifest_hash(
        reference_manifest,
        REFERENCE_MANIFEST_SHA256,
        "Reference",
    )
    rows, freeze, selected_before, freeze_before = _validate_freeze(
        output_dir,
        inference_manifest,
        runtime_provenance=runtime_provenance,
        scorer_script_path=scorer_script_path,
    )
    frozen_scorer_path = output_dir / "frozen-scorer.py"
    frozen_scorer_before = frozen_scorer_path.read_bytes()
    if hashlib.sha256(frozen_scorer_before).hexdigest() != freeze.get(
        "scorer_script_sha256"
    ):
        raise ValueError("Frozen scorer source differs from prediction freeze")
    inference = _manifest_examples(inference_manifest)
    reference = _manifest_examples(reference_manifest)
    inference_ids = [str(example["video_id"]) for example in inference]
    if [str(example["video_id"]) for example in reference] != inference_ids:
        raise ValueError("Reference order differs from frozen inference panel")
    annotations_by_name = {path.name: path for path in annotations}
    if len(annotations_by_name) != len(reference):
        raise ValueError("Evaluation requires one unique annotation per panel row")
    references: dict[str, str] = {}
    categories: dict[str, str] = {}
    annotation_hashes: list[dict[str, str]] = []
    for inference_example, reference_example in zip(inference, reference, strict=True):
        video_id = str(reference_example["video_id"])
        raw_annotation = Path(str(reference_example.get("annotation", "")))
        annotation = annotations_by_name.get(raw_annotation.name)
        if annotation is None or not annotation.is_file():
            raise ValueError(f"Missing declared annotation for {video_id}")
        payload = _load_json(annotation)
        if payload.get("question") != inference_example.get("question"):
            raise ValueError(f"Annotation question mismatch for {video_id}")
        video_url = payload.get("video_url")
        if not isinstance(video_url, str):
            raise ValueError(f"Annotation video URL is missing for {video_id}")
        parsed = urlparse(video_url)
        if parse_qs(parsed.query).get("v") != [video_id]:
            raise ValueError(f"Annotation video identity mismatch for {video_id}")
        correct = payload.get("correct_answer")
        if not isinstance(correct, str) or not correct.strip():
            raise ValueError(f"Missing correct answer for {video_id}")
        references[video_id] = correct
        categories[video_id] = str(inference_example.get("category", "unknown"))
        annotation_hashes.append(
            {
                "sha256": sha256_path(annotation),
                "video_id": video_id,
            }
        )

    scored: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row["video_id"])
        if video_id not in references:
            raise ValueError(
                f"Frozen prediction is outside reference panel: {video_id}"
            )
        prediction = str(row["answer"])
        reference_answer = references[video_id]
        scored.append(
            {
                **row,
                "category": categories[video_id],
                "reference": reference_answer,
                "normalized_exact_match": float(
                    _normalize(prediction) == _normalize(reference_answer)
                ),
                "token_f1": _token_f1(prediction, reference_answer),
                "meteor": meteor_fn(prediction, reference_answer),
            }
        )
    conditions: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition_id"]) for row in scored}):
        condition_rows = [row for row in scored if row["condition_id"] == condition]
        conditions[condition] = {
            "rows": len(condition_rows),
            "normalized_exact_match": fmean(
                float(row["normalized_exact_match"]) for row in condition_rows
            ),
            "token_f1": fmean(float(row["token_f1"]) for row in condition_rows),
            "meteor": fmean(float(row["meteor"]) for row in condition_rows),
        }
    scored_path = output_dir / "scored-predictions.jsonl"
    _atomic_write_bytes(scored_path, canonical_jsonl(scored))
    report = {
        "conditions": conditions,
        "examples": scored,
        "provenance": {
            **freeze,
            "annotation_sha256": annotation_hashes,
            "reference_manifest_sha256": sha256_path(reference_manifest),
            "runtime": dict(runtime_provenance),
            "scorer_script_sha256": sha256_path(scorer_script_path),
            "scored_prediction_sha256": sha256_path(scored_path),
            "references_loaded_only_after_prediction_freeze": True,
        },
    }
    _atomic_write_json(output_dir / "evaluation.json", report, indent=2)
    if (
        (output_dir / "selected-predictions.jsonl").read_bytes() != selected_before
        or (output_dir / "prediction-freeze.json").read_bytes() != freeze_before
        or frozen_scorer_path.read_bytes() != frozen_scorer_before
    ):
        raise ValueError("Prediction freeze or frozen scorer changed during evaluation")
    return report


def container_base_argv() -> list[str]:
    return [
        "podman",
        "run",
        "--pull=never",
        "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        f"VQA_RUNTIME_IMAGE={SCORER_IMAGE}",
    ]


def build_mounts(
    action: Action,
    layout: Layout,
    annotations: Sequence[Path],
) -> list[Mount]:
    scorer_source = (
        layout.script_path
        if action == "freeze"
        else layout.output_dir / "frozen-scorer.py"
    )
    mounts = [
        Mount(layout.output_dir, layout.output_dir, "rw"),
        Mount(
            layout.inference_manifest,
            CANONICAL_ROOT / "configs/vqa/failure_panel_12_inference.json",
            "ro",
        ),
        Mount(scorer_source, INSIDE_SCRIPT, "ro"),
    ]
    if action == "freeze":
        if annotations:
            raise ValueError("Freeze phase cannot receive annotations")
        mounts.append(Mount(layout.candidate_path, layout.candidate_path, "ro"))
    elif action == "evaluate":
        mounts.extend(
            [
                Mount(
                    layout.reference_manifest,
                    CANONICAL_ROOT / "configs/vqa/failure_panel_12.json",
                    "ro",
                ),
                Mount(
                    layout.output_dir / "selected-predictions.jsonl",
                    layout.output_dir / "selected-predictions.jsonl",
                    "ro",
                ),
                Mount(
                    layout.output_dir / "prediction-freeze.json",
                    layout.output_dir / "prediction-freeze.json",
                    "ro",
                ),
                *(Mount(path, path, "ro") for path in annotations),
            ]
        )
    else:
        raise ValueError(f"Unknown scoring action: {action}")
    return mounts


def discover_layout() -> Layout:
    script_path = Path(__file__).resolve()
    inside_container = script_path == INSIDE_SCRIPT
    repo_root = CANONICAL_ROOT if inside_container else script_path.parents[1]
    data_root = (
        CANONICAL_DATA
        if inside_container
        else (repo_root / "data").resolve(strict=True)
    )
    if data_root != CANONICAL_DATA:
        raise ValueError(f"data must resolve to {CANONICAL_DATA}")
    output = data_root / "vqa/benchmarks/followups/failure-panel-2026-07-24/scoring"
    return Layout(
        repo_root=repo_root,
        data_root=data_root,
        output_dir=output,
        candidate_path=output.parent / "candidate-predictions.jsonl",
        inference_manifest=repo_root / "configs/vqa/failure_panel_12_inference.json",
        reference_manifest=repo_root / "configs/vqa/failure_panel_12.json",
        script_path=script_path,
    )


def discover_annotations(layout: Layout) -> tuple[Path, ...]:
    reference = _manifest_examples(layout.reference_manifest)
    annotations = []
    root = layout.data_root / "vqa/2025/training/annotations"
    for example in reference:
        relative = Path(str(example.get("annotation", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Annotation path must be relative")
        path = layout.data_root / relative
        if path.parent != root or not path.is_file():
            raise ValueError(f"Annotation is outside exact root or missing: {path}")
        annotations.append(path)
    if len(annotations) != 12 or len(set(annotations)) != 12:
        raise ValueError("Failure panel requires exactly 12 annotations")
    return tuple(annotations)


def build_container_argv(
    action: Action,
    layout: Layout,
    mounts: Sequence[Mount],
) -> list[str]:
    argv = container_base_argv()
    for mount in mounts:
        argv.extend(["-v", f"{mount.source}:{mount.target}:{mount.mode}"])
    argv.extend(
        [
            "-w",
            str(CANONICAL_ROOT),
            SCORER_IMAGE,
            "python",
            str(INSIDE_SCRIPT),
            f"--inside={action}",
        ]
    )
    return argv


def _inspect_image() -> None:
    completed = subprocess.run(
        [
            "podman",
            "image",
            "inspect",
            "--format",
            "{{.Digest}}",
            SCORER_IMAGE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != SCORER_DIGEST:
        raise ValueError("Scorer image digest mismatch")


def _validate_mounts(mounts: Sequence[Mount], output_dir: Path) -> None:
    missing = [str(mount.source) for mount in mounts if not mount.source.exists()]
    if missing:
        raise FileNotFoundError(f"Missing mount sources: {missing}")
    writable = [mount for mount in mounts if mount.mode == "rw"]
    if writable != [Mount(output_dir, output_dir, "rw")]:
        raise ValueError("Scoring output directory must be the only writable mount")


def _validate_phase_state(action: Action, layout: Layout) -> None:
    entries = {path.name for path in layout.output_dir.iterdir()}
    expected = (
        set()
        if action == "freeze"
        else {
            "selected-predictions.jsonl",
            "prediction-freeze.json",
            "frozen-scorer.py",
        }
    )
    if entries != expected:
        raise ValueError(
            f"Unexpected {action} output-directory state: "
            f"expected {sorted(expected)}, got {sorted(entries)}"
        )


def _plan(
    action: Action, mounts: Sequence[Mount], argv: Sequence[str]
) -> dict[str, Any]:
    return {
        "phase": action,
        "scorer_image": SCORER_IMAGE,
        "container_argv": list(argv),
        "mounts": [
            {
                "source": str(mount.source),
                "target": str(mount.target),
                "mode": mount.mode,
            }
            for mount in mounts
        ],
    }


def _host(action: Action, dry_run: bool) -> None:
    layout = discover_layout()
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = () if action == "freeze" else discover_annotations(layout)
    mounts = build_mounts(action, layout, annotations)
    argv = build_container_argv(action, layout, mounts)
    if dry_run:
        print(json.dumps(_plan(action, mounts, argv), indent=2))
        return
    _validate_phase_state(action, layout)
    _validate_mounts(mounts, layout.output_dir)
    _inspect_image()
    subprocess.run(argv, check=True)


def _inside(action: Action) -> None:
    layout = discover_layout()
    runtime = _attest_scorer_runtime()
    if action == "freeze":
        if layout.reference_manifest.exists():
            raise ValueError("Freeze phase can see the reference manifest")
        annotation_root = layout.data_root / "vqa/2025/training/annotations"
        if annotation_root.exists():
            raise ValueError("Freeze phase can see the annotation directory")
        freeze_candidates(
            layout.candidate_path,
            layout.inference_manifest,
            layout.output_dir,
            runtime_provenance=runtime,
            scorer_script_path=Path(__file__).resolve(),
        )
    else:
        evaluate_frozen(
            layout.output_dir,
            layout.inference_manifest,
            layout.reference_manifest,
            discover_annotations(layout),
            runtime_provenance=runtime,
            scorer_script_path=Path(__file__).resolve(),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=("assemble", "freeze", "evaluate"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="CONDITION=PATH",
        help="Label-free prediction source used by assemble",
    )
    parser.add_argument(
        "--inside", choices=("freeze", "evaluate"), help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inside:
        if args.action or args.dry_run or args.source:
            raise ValueError("--inside cannot be combined with host arguments")
        _inside(args.inside)
    else:
        if not args.action:
            raise ValueError("Host mode requires freeze or evaluate")
        if args.action == "assemble":
            if args.dry_run or not args.source:
                raise ValueError("assemble requires one or more --source values")
            layout = discover_layout()
            sources = []
            for spec in args.source:
                condition, separator, raw_path = spec.partition("=")
                if not separator or not condition or not raw_path:
                    raise ValueError("--source must be CONDITION=PATH")
                sources.append((condition, Path(raw_path).resolve(strict=True)))
            assemble_candidates(
                sources,
                layout.inference_manifest,
                layout.candidate_path,
            )
        else:
            if args.source:
                raise ValueError("--source is valid only for assemble")
            _host(args.action, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
