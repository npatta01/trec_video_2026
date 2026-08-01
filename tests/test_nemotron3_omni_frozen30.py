from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from trec_video_2026.vqa_experiments import nemotron3_omni_frozen30 as frozen30
from trec_video_2026.vqa_experiments.nemotron3_omni_grading import (
    GradeResolution,
    reconcile_grades,
)
from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    CONDITIONS,
    FROZEN30_INFERENCE_MANIFEST_SHA256,
    FROZEN30_ROWS,
    FROZEN30_SOURCE_MEDIA_PATHS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    build_chat_request,
)

SOURCE_MANIFEST = (
    Path(__file__).parents[1] / "configs/vqa/native_audio_benchmark_30_inference.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_qwen(path: Path, *, mutation: str | None = None) -> str:
    rows = []
    for condition in frozen30.QWEN_CONDITION_ORDER:
        for index, (video_id, _category, _role) in enumerate(FROZEN30_ROWS):
            request = {
                "run_id": frozen30.QWEN_RUN_ID,
                "condition_id": condition,
                "video_id": video_id,
            }
            rows.append(
                {
                    "run_id": frozen30.QWEN_RUN_ID,
                    "condition_id": condition,
                    "video_id": video_id,
                    "status": "passed",
                    "answer": f"baseline answer {index}",
                    "model_id": "Qwen/Qwen3.5-4B",
                    "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "runtime_image": frozen30.QWEN_RUNTIME_IMAGE,
                    "evidence_identity_sha256": "2" * 64,
                    "evidence_manifest_sha256": "3" * 64,
                    "launch_decision_sha256": "4" * 64,
                    "request": request,
                    "request_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
                }
            )
    if mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "provenance":
        rows[0]["request"]["run_id"] = "other-run"
        rows[0]["request_sha256"] = hashlib.sha256(
            _canonical(rows[0]["request"])
        ).hexdigest()
    elif mutation == "runtime":
        rows[0]["runtime_image"] = "other-image@sha256:" + "1" * 64
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _media_roots(root: Path) -> dict[str, Path]:
    return {
        "source_media_root": root / "source-media",
        "normalized_media_root": root / "candidate-bundle" / "normalized",
    }


def _audit_root(root: Path) -> Path:
    return root / "retained-audit"


def _candidate_contract(root: Path) -> dict[str, object]:
    source_rows = json.loads(SOURCE_MANIFEST.read_text())["examples"]
    roots = _media_roots(root)
    source_examples = [
        {
            "video_id": video_id,
            "category": category,
            "role": role,
            "question": source_rows[index]["question"].strip(),
            "media_path": FROZEN30_SOURCE_MEDIA_PATHS[video_id],
            "media_sha256": hashlib.sha256(
                (
                    roots["source_media_root"] / FROZEN30_SOURCE_MEDIA_PATHS[video_id]
                ).read_bytes()
            ).hexdigest(),
        }
        for index, (video_id, category, role) in enumerate(FROZEN30_ROWS)
    ]
    source_contract = {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": source_examples,
    }
    return {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "source_contract_sha256": hashlib.sha256(
            _canonical(source_contract)
        ).hexdigest(),
        "media_derivation": {
            "container": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": [
            {
                "video_id": video_id,
                "category": category,
                "role": role,
                "question": source_rows[index]["question"].strip(),
                "source_media_path": FROZEN30_SOURCE_MEDIA_PATHS[video_id],
                "source_media_sha256": hashlib.sha256(
                    (
                        roots["source_media_root"]
                        / FROZEN30_SOURCE_MEDIA_PATHS[video_id]
                    ).read_bytes()
                ).hexdigest(),
                "media_path": f"{video_id}.mp4",
                "derived_media_sha256": hashlib.sha256(
                    (roots["normalized_media_root"] / f"{video_id}.mp4").read_bytes()
                ).hexdigest(),
                "source_duration_seconds": 10.0,
                "derived_duration_seconds": 10.0,
            }
            for index, (video_id, category, role) in enumerate(FROZEN30_ROWS)
        ],
    }


def _write_candidate(root: Path) -> tuple[Path, Path, Path]:
    roots = _media_roots(root)
    bundle_root = root / "candidate-bundle"
    for video_id, _category, _role in FROZEN30_ROWS:
        source = roots["source_media_root"] / FROZEN30_SOURCE_MEDIA_PATHS[video_id]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"canonical-source:{video_id}".encode())
        normalized = roots["normalized_media_root"] / f"{video_id}.mp4"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(f"normalized-h264-aac:{video_id}".encode())
    contract = _candidate_contract(root)
    contract_path = bundle_root / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    records = []
    for pair_index, (video_id, category, role) in enumerate(FROZEN30_ROWS):
        for condition in CONDITIONS:
            answer = f"candidate answer {pair_index} variant {int(condition['audio_enabled'])}"
            request = build_chat_request(
                media_path=Path("/dstack/input/normalized")
                / contract["examples"][pair_index]["media_path"],
                question=contract["examples"][pair_index]["question"],
                audio_enabled=condition["audio_enabled"],
            )
            records.append(
                {
                    "schema_version": 1,
                    "run_id": frozen30.CANDIDATE_RUN_ID,
                    "mode": "frozen30-paired",
                    "pair_index": pair_index,
                    "video_id": video_id,
                    "category": category,
                    "role": role,
                    "condition_id": condition["condition_id"],
                    "audio_enabled": condition["audio_enabled"],
                    "request_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
                    "status": "passed",
                    "answer": answer,
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    "error_type": None,
                    "error_message": None,
                    "elapsed_seconds": 1.0,
                }
            )
    predictions_path = bundle_root / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    prediction_sha = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "run_id": frozen30.CANDIDATE_RUN_ID,
        "mode": "frozen30-paired",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "contract_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "contract_file_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "predictions_sha256": prediction_sha,
        "frozen": True,
        "completion": {"completed": 60, "expected": 60},
        "reference_answers_available": False,
    }
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums_path = bundle_root / "SHA256SUMS"
    _refresh_checksums(predictions_path, manifest_path, checksums_path)
    _write_retained_audit(root)
    return predictions_path, manifest_path, checksums_path


def _refresh_checksums(predictions: Path, manifest: Path, checksums: Path) -> None:
    bundle_root = checksums.parent
    files = sorted(
        path for path in bundle_root.rglob("*") if path.is_file() and path != checksums
    )
    checksums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"./{path.relative_to(bundle_root).as_posix()}\n"
            for path in files
        )
    )


def _write_retained_audit(
    root: Path, *, elapsed_seconds: float = 3540.0, cost: float = 1.2
) -> Path:
    bundle_root = root / "candidate-bundle"
    audit_root = _audit_root(root)
    audit_root.mkdir(exist_ok=True)
    contract = json.loads((bundle_root / "contract.json").read_text())
    source_contract = {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": [
            {
                "video_id": example["video_id"],
                "category": example["category"],
                "role": example["role"],
                "question": example["question"],
                "media_path": example["source_media_path"],
                "media_sha256": example["source_media_sha256"],
            }
            for example in contract["examples"]
        ],
    }
    input_contract_path = audit_root / "input-contract.json"
    input_contract_path.write_text(json.dumps(source_contract, sort_keys=True) + "\n")
    input_objects = {
        "contract.json": hashlib.sha256(input_contract_path.read_bytes()).hexdigest(),
        **{
            example["media_path"]: example["media_sha256"]
            for example in source_contract["examples"]
        },
    }
    marker = {
        "schema_version": 1,
        "input_count": 31,
        "object_count": 32,
        "contract_sha256": input_objects["contract.json"],
        "objects": input_objects,
    }
    marker_path = audit_root / "INPUTS_COMPLETE.json"
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
    (audit_root / "preview.txt").write_text("exact declined preview\n")
    (audit_root / "collection-status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": frozen30.CANDIDATE_RUN_ID,
                "collection_kind": "completed-candidate",
                "scoreable": False,
                "reason": "pending-independent-freeze-verification",
            },
            sort_keys=True,
        )
        + "\n"
    )
    (audit_root / "dstack-run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": frozen30.CANDIDATE_RUN_ID,
                "id": "run-uuid",
                "status": "done",
                "status_message": "terminal",
                "termination_reason": None,
                "job_termination_reason": None,
                "submitted_at": "2026-08-01T12:00:00+00:00",
                "finished_at": "2026-08-01T12:59:00+00:00",
                "elapsed_seconds": elapsed_seconds,
                "cost": cost,
                "backend": "runpod",
                "region": "US-TX-3",
                "price": 2.0,
                "instance_type": "NVIDIA L40S",
                "cpu_count": 8,
                "gpu_count": 1,
                "gpu_name": "L40S",
                "gpu_vendor": "nvidia",
                "gpu_memory_mib": 49152,
                "host_memory_mib": 65536,
                "disk_mib": 102400,
                "spot": False,
                "exit_status": 0,
            },
            sort_keys=True,
        )
        + "\n"
    )
    candidate_files = {
        f"./{path.relative_to(bundle_root).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file()
    }
    audit_evidence_files = {
        f"./{name}": hashlib.sha256((audit_root / name).read_bytes()).hexdigest()
        for name in (
            "INPUTS_COMPLETE.json",
            "collection-status.json",
            "dstack-run.json",
            "input-contract.json",
            "preview.txt",
        )
    }
    anchor = {
        "schema_version": 1,
        "anchor_type": "nemotron3-frozen30-collected-workload",
        "run_id": frozen30.CANDIDATE_RUN_ID,
        "candidate_file_count": len(candidate_files),
        "candidate_files": candidate_files,
        "audit_evidence_file_count": len(audit_evidence_files),
        "audit_evidence_files": audit_evidence_files,
    }
    anchor_path = audit_root / "workload-anchor.json"
    anchor_path.write_text(json.dumps(anchor, sort_keys=True) + "\n")
    audit_files = sorted(
        path
        for path in audit_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (audit_root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.name}\n"
            for path in audit_files
        )
    )
    return audit_root


def _retained_candidate_args(root: Path) -> dict[str, Path]:
    return {
        **_media_roots(root),
        "audit_bundle_root": _audit_root(root),
    }


def _write_privacy_review(root: Path, *, gap: bool = False) -> Path:
    anchor_path = _audit_root(root) / "workload-anchor.json"
    path = root / "privacy-audit-review.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_type": "nemotron3-frozen30-privacy-audit",
                "run_id": frozen30.CANDIDATE_RUN_ID,
                "workload_anchor_sha256": hashlib.sha256(
                    anchor_path.read_bytes()
                ).hexdigest(),
                "reviewer_id": "reviewer@example.org",
                "reviewed_at": "2026-08-01T18:00:00+00:00",
                "determination": ("load-bearing-gap" if gap else "no-load-bearing-gap"),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _derived_operational(
    root: Path,
    *,
    elapsed_seconds: float = 3540.0,
    cost: float = 1.2,
    gap: bool = False,
) -> frozen30.OperationalEvidence:
    bundle_root = root / "candidate-bundle"
    if not (bundle_root / "manifest.json").is_file():
        predictions, manifest, _checksums = _write_candidate(root)
    else:
        predictions = bundle_root / "predictions.jsonl"
        manifest = bundle_root / "manifest.json"
    _write_retained_audit(root, elapsed_seconds=elapsed_seconds, cost=cost)
    return frozen30.derive_operational_evidence(
        candidate_predictions_path=predictions,
        candidate_manifest_path=manifest,
        audit_bundle_root=_audit_root(root),
        privacy_review_path=_write_privacy_review(root, gap=gap),
    )


def _refresh_candidate_bindings(
    predictions: Path, manifest: Path, checksums: Path
) -> None:
    contract = json.loads(manifest.with_name("contract.json").read_text())
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    for record in records:
        example = contract["examples"][record["pair_index"]]
        request = build_chat_request(
            media_path=Path("/dstack/input/normalized") / example["media_path"],
            question=example["question"],
            audio_enabled=record["audio_enabled"],
        )
        record["request_sha256"] = hashlib.sha256(_canonical(request)).hexdigest()
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    body = json.loads(manifest.read_text())
    body["contract_file_sha256"] = hashlib.sha256(
        manifest.with_name("contract.json").read_bytes()
    ).hexdigest()
    body["contract_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    body["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)


def _references(keys: tuple[str, ...]) -> list[dict[str, str]]:
    by_id = {video_id: category for video_id, category, _role in FROZEN30_ROWS}
    questions = {
        row["video_id"]: row["question"].strip()
        for row in json.loads(SOURCE_MANIFEST.read_text())["examples"]
    }
    return [
        {
            "row_key": key,
            "category": by_id[key],
            "question": questions[key],
            "reference": f"reference {index}",
        }
        for index, key in enumerate(keys)
    ]


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    qwen = tmp_path / "qwen.jsonl"
    qwen_sha = _write_qwen(qwen)
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", qwen_sha)
    predictions, manifest, checksums = _write_candidate(tmp_path)
    return frozen30.build_blind_grading_bundle(
        qwen_predictions_path=qwen,
        candidate_predictions_path=predictions,
        candidate_manifest_path=manifest,
        candidate_checksums_path=checksums,
        **_retained_candidate_args(tmp_path),
        reference_loader=_references,
        grader_ids=("grader-a", "grader-b"),
        shuffle_seeds=(1709, 2903),
    )


def test_qwen_full_file_hash_is_recomputed_before_structure_or_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    expected_sha = _write_qwen(qwen)
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", expected_sha)
    qwen.write_bytes(qwen.read_bytes() + b" ")
    predictions, manifest, checksums = _write_candidate(tmp_path)
    calls = []

    with pytest.raises(ValueError, match="pinned full-file SHA-256"):
        frozen30.build_blind_grading_bundle(
            qwen_predictions_path=qwen,
            candidate_predictions_path=predictions,
            candidate_manifest_path=manifest,
            candidate_checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
            reference_loader=lambda keys: calls.append(keys) or _references(keys),
            grader_ids=("grader-a", "grader-b"),
            shuffle_seeds=(1709, 2903),
        )
    assert calls == []


@pytest.mark.parametrize("mutation", ["order", "provenance", "runtime"])
def test_qwen_canonical_order_and_embedded_provenance_are_independently_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(
        frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen, mutation=mutation)
    )
    with pytest.raises(ValueError, match="order|provenance"):
        frozen30.verify_qwen_freeze(qwen)


@pytest.mark.parametrize("artifact", ["predictions", "manifest", "checksums"])
def test_candidate_prediction_manifest_and_checksums_mutation_fail_before_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)
    {"predictions": predictions, "manifest": manifest, "checksums": checksums}[
        artifact
    ].write_bytes(
        {"predictions": predictions, "manifest": manifest, "checksums": checksums}[
            artifact
        ].read_bytes()
        + b"tamper"
    )
    calls = []
    with pytest.raises(ValueError):
        frozen30.build_blind_grading_bundle(
            qwen_predictions_path=qwen,
            candidate_predictions_path=predictions,
            candidate_manifest_path=manifest,
            candidate_checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
            reference_loader=lambda keys: calls.append(keys) or _references(keys),
            grader_ids=("grader-a", "grader-b"),
            shuffle_seeds=(1709, 2903),
        )
    assert calls == []


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicate"])
def test_candidate_requires_exact_adjacent_unique_pairs(
    tmp_path: Path, mutation: str
) -> None:
    predictions, manifest, checksums = _write_candidate(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    if mutation == "missing":
        rows.pop()
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[2:4] = rows[0:2]
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    prediction_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
    body = json.loads(manifest.read_text())
    body["predictions_sha256"] = prediction_sha
    if mutation == "missing":
        body["completion"] = {"completed": 59, "expected": 60}
    manifest.write_text(json.dumps(body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)
    with pytest.raises(ValueError, match="60/60|order|duplicate"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


@pytest.mark.parametrize("elapsed", [math.nan, math.inf, -math.inf])
def test_candidate_rejects_nonfinite_elapsed_seconds(
    tmp_path: Path, elapsed: float
) -> None:
    """Catches NaN or infinity surviving JSON parsing into a scoreable freeze."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["elapsed_seconds"] = elapsed
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    body = json.loads(manifest.read_text())
    body["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)

    with pytest.raises(ValueError, match="request provenance"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


@pytest.mark.parametrize(
    "mutation", ["coherent-media", "declared-missing", "extra-file"]
)
def test_retained_anchor_rejects_coherent_or_exact_set_candidate_substitution(
    tmp_path: Path, mutation: str
) -> None:
    """Catches replacing a whole internally consistent candidate after collection."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    bundle_root = manifest.parent
    if mutation == "coherent-media":
        contract_path = bundle_root / "contract.json"
        contract = json.loads(contract_path.read_text())
        media_path = bundle_root / "normalized" / contract["examples"][0]["media_path"]
        media_path.write_bytes(b"coherently substituted normalized media")
        contract["examples"][0]["derived_media_sha256"] = hashlib.sha256(
            media_path.read_bytes()
        ).hexdigest()
        contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
        _refresh_candidate_bindings(predictions, manifest, checksums)
    elif mutation == "declared-missing":
        checksums.write_text(
            checksums.read_text() + f"{'a' * 64}  ./unlisted-private.log\n"
        )
    else:
        (bundle_root / "unlisted-private.log").write_text("private extra\n")

    with pytest.raises(ValueError, match="anchor|file set|missing|extra"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


def test_canonical_source_bytes_must_match_retained_input_marker(
    tmp_path: Path,
) -> None:
    """Catches coherently substituting source bytes and the mutable candidate contract."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = manifest.with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    example = contract["examples"][0]
    source_path = (
        _media_roots(tmp_path)["source_media_root"] / example["source_media_path"]
    )
    source_path.write_bytes(b"substituted canonical source")
    example["source_media_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    source_rows = json.loads(SOURCE_MANIFEST.read_text())["examples"]
    source_contract = {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "conditions": [dict(condition) for condition in CONDITIONS],
        "examples": [
            {
                "video_id": video_id,
                "category": category,
                "role": role,
                "question": source_rows[index]["question"].strip(),
                "media_path": FROZEN30_SOURCE_MEDIA_PATHS[video_id],
                "media_sha256": candidate_example["source_media_sha256"],
            }
            for index, ((video_id, category, role), candidate_example) in enumerate(
                zip(FROZEN30_ROWS, contract["examples"], strict=True)
            )
        ],
    }
    contract["source_contract_sha256"] = hashlib.sha256(
        _canonical(source_contract)
    ).hexdigest()
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    _refresh_candidate_bindings(predictions, manifest, checksums)

    with pytest.raises(
        ValueError, match="input marker|input contract|source media|anchor"
    ):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


def test_label_leakage_and_reference_stratum_mismatch_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = manifest.with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    contract["examples"][0]["reference"] = "leaked"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    manifest_body = json.loads(manifest.read_text())
    manifest_body["contract_file_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_body["contract_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    manifest.write_text(json.dumps(manifest_body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)
    with pytest.raises(ValueError, match="label-bearing"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


def test_candidate_recomputes_request_provenance_from_manifest_bound_contract(
    tmp_path: Path,
) -> None:
    predictions, manifest, checksums = _write_candidate(tmp_path)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["request_sha256"] = "9" * 64
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    body = json.loads(manifest.read_text())
    body["predictions_sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)

    with pytest.raises(ValueError, match="request provenance"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


@pytest.mark.parametrize("mutation", ["question", "ground_truth", "source_hash"])
def test_candidate_normalized_contract_is_anchored_to_pinned_source_manifest(
    tmp_path: Path, mutation: str
) -> None:
    """Catches coherent relabeling or question replacement across all inner hashes."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = manifest.with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    if mutation == "question":
        contract["examples"][0]["question"] = "A different grading question?"
        contract["source_contract_sha256"] = "a" * 64
    elif mutation == "ground_truth":
        contract["examples"][0]["ground_truth"] = "secret label"
    else:
        contract["source_contract_sha256"] = "a" * 64
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    _refresh_candidate_bindings(predictions, manifest, checksums)

    with pytest.raises(ValueError, match="source|question|label-bearing|schema"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


@pytest.mark.parametrize("field", ["source_media_sha256", "derived_media_sha256"])
def test_candidate_media_hashes_are_checked_against_independent_artifact_bytes(
    tmp_path: Path, field: str
) -> None:
    """Catches trusting a coherently refreshed contract instead of media bytes."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = manifest.with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    contract["examples"][0][field] = "a" * 64
    if field == "source_media_sha256":
        source_rows = json.loads(SOURCE_MANIFEST.read_text())["examples"]
        source_contract = {
            "schema_version": 1,
            "mode": "frozen30-paired",
            "panel_id": "nemotron3-omni-frozen30",
            "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "conditions": [dict(condition) for condition in CONDITIONS],
            "examples": [
                {
                    "video_id": video_id,
                    "category": category,
                    "role": role,
                    "question": source_rows[index]["question"].strip(),
                    "media_path": FROZEN30_SOURCE_MEDIA_PATHS[video_id],
                    "media_sha256": example["source_media_sha256"],
                }
                for index, ((video_id, category, role), example) in enumerate(
                    zip(FROZEN30_ROWS, contract["examples"], strict=True)
                )
            ],
        }
        contract["source_contract_sha256"] = hashlib.sha256(
            _canonical(source_contract)
        ).hexdigest()
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    _refresh_candidate_bindings(predictions, manifest, checksums)

    with pytest.raises(ValueError, match="media.*hash|hash.*media|input contract"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
        )


def test_candidate_and_reference_questions_must_match_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)

    def mismatched(keys: tuple[str, ...]):
        rows = _references(keys)
        rows[0]["question"] = "A different grading question?"
        return rows

    with pytest.raises(ValueError, match="question"):
        frozen30.build_blind_grading_bundle(
            qwen_predictions_path=qwen,
            candidate_predictions_path=predictions,
            candidate_manifest_path=manifest,
            candidate_checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
            reference_loader=mismatched,
            grader_ids=("grader-a", "grader-b"),
            shuffle_seeds=(1709, 2903),
        )


def test_blind_packets_hide_condition_and_row_identity_and_use_two_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    packets = list(bundle.packets.values())
    assert len(packets) == 2
    assert [x["candidate_answer"] for x in packets[0]["items"]] != [
        x["candidate_answer"] for x in packets[1]["items"]
    ]
    payload = json.dumps(bundle.packets, sort_keys=True)
    assert len(packets[0]["items"]) == 90
    for video_id, category, role in FROZEN30_ROWS:
        assert video_id not in payload
        assert category not in payload
        assert f'"{role}"' not in payload
    for forbidden in ("qwen", "nemotron", "audio-off", "audio-on", '"condition"'):
        assert forbidden not in payload.lower()


def test_packet_identity_does_not_reveal_seed_or_caller_grader_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)
    bundle = frozen30.build_blind_grading_bundle(
        qwen_predictions_path=qwen,
        candidate_predictions_path=predictions,
        candidate_manifest_path=manifest,
        candidate_checksums_path=checksums,
        **_retained_candidate_args(tmp_path),
        reference_loader=_references,
        grader_ids=("nemotron-reviewer", "audio-on-reviewer"),
        shuffle_seeds=(1709, 2903),
    )

    assert set(bundle.packets) == {"grader-1", "grader-2"}
    payload = json.dumps(bundle.packets, sort_keys=True).lower()
    assert "nemotron-reviewer" not in payload
    assert "audio-on-reviewer" not in payload
    observed = bundle.packets["grader-1"]["packet_id"]
    recovered = [
        seed
        for seed in range(10_000)
        if "blind-1-"
        + hashlib.sha256(f"1:nemotron-reviewer:{seed}".encode()).hexdigest()[:12]
        == observed
    ]
    assert recovered == []


def test_public_question_order_cannot_verify_low_entropy_seed_guesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    packet = bundle.packets["grader-1"]
    observed_questions = [item["question"] for item in packet["items"]]
    questions = {
        row["video_id"]: row["question"].strip()
        for row in json.loads(SOURCE_MANIFEST.read_text())["examples"]
    }
    canonical = [
        (video_id, condition)
        for video_id, _category, _role in FROZEN30_ROWS
        for condition in ("qwen12", "audio-off", "audio-on")
    ]
    recovered = []
    for seed in range(10_000):
        order = list(canonical)
        random.Random(seed).shuffle(order)
        if [
            questions[video_id] for video_id, _condition in order
        ] == observed_questions:
            recovered.append(seed)
    assert recovered == []


def test_reference_stratum_mismatch_fails_after_both_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)

    def wrong(keys: tuple[str, ...]):
        rows = _references(keys)
        rows[0]["category"] = "wrong stratum"
        return rows

    with pytest.raises(ValueError, match="stratum"):
        frozen30.build_blind_grading_bundle(
            qwen_predictions_path=qwen,
            candidate_predictions_path=predictions,
            candidate_manifest_path=manifest,
            candidate_checksums_path=checksums,
            **_retained_candidate_args(tmp_path),
            reference_loader=wrong,
            grader_ids=("grader-a", "grader-b"),
            shuffle_seeds=(1709, 2903),
        )


def test_adjudication_packet_contains_only_disagreements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    submissions = []
    for index, packet in enumerate(bundle.packets.values()):
        submissions.append(
            {
                "grader_id": packet["grader_id"],
                "grades": [
                    {"item_id": item["item_id"], "grade": "P"}
                    for item in packet["items"]
                ],
            }
        )
    submissions[1]["grades"][0]["grade"] = "C"
    result = reconcile_grades(bundle, submissions)
    assert result.status == "adjudication_required"
    assert len(result.adjudication_packet["items"]) == 1


def _packet_submission(
    packet: dict[str, object],
    grades: dict[str, str] | None = None,
    *,
    grader_id: str | None = None,
):
    grades = grades or {}
    return {
        "grader_id": grader_id or packet["grader_id"],
        "packet_id": packet["packet_id"],
        "grades": [
            {
                "item_id": item["item_id"],
                "grade": grades.get(item["candidate_answer"], "P"),
            }
            for item in packet["items"]
        ],
    }


def test_grader_submission_must_bind_to_exact_packet_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    resolver = getattr(frozen30, "reconcile_frozen30_grades", reconcile_grades)
    assignments = bundle._grader_assignment_artifact["assignments"]
    submissions = [
        _packet_submission(
            bundle.packets[assignment["grader_alias"]],
            grader_id=assignment["grader_id"],
        )
        for assignment in assignments
    ]
    assert resolver(bundle, submissions).status == "final"
    submissions[0]["packet_id"] = "blind-wrong"

    with pytest.raises(ValueError, match="packet ID"):
        resolver(bundle, submissions)


def test_adjudication_submission_must_bind_to_exact_disagreement_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    resolver = getattr(frozen30, "reconcile_frozen30_grades", reconcile_grades)
    packets = list(bundle.packets.values())
    assignments = bundle._grader_assignment_artifact["assignments"]
    disputed_answer = packets[0]["items"][0]["candidate_answer"]
    submissions = [
        _packet_submission(packets[0], grader_id=assignments[0]["grader_id"]),
        _packet_submission(
            packets[1],
            {disputed_answer: "C"},
            grader_id=assignments[1]["grader_id"],
        ),
    ]
    pending = resolver(bundle, submissions)
    assert pending.status == "adjudication_required"

    with pytest.raises(ValueError, match="packet ID"):
        resolver(
            bundle,
            submissions,
            adjudication={
                "grader_id": "grader-3",
                "packet_id": "adjudication-wrong",
                "grades": [
                    {"case_id": item["case_id"], "grade": "C"}
                    for item in pending.adjudication_packet["items"]
                ],
            },
        )
    final = resolver(
        bundle,
        submissions,
        adjudication={
            "grader_id": "grader-3",
            "packet_id": pending.adjudication_packet["packet_id"],
            "grades": [
                {"case_id": item["case_id"], "grade": "C"}
                for item in pending.adjudication_packet["items"]
            ],
        },
    )
    assert final.status == "final"


@pytest.mark.parametrize("assigned_grader_index", [0, 1])
def test_original_private_grader_cannot_adjudicate_a_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assigned_grader_index: int,
) -> None:
    """Catches losing first-round private identities during alias translation."""
    bundle = _bundle(tmp_path, monkeypatch)
    packets = list(bundle.packets.values())
    assignments = bundle._grader_assignment_artifact["assignments"]
    disputed_answer = packets[0]["items"][0]["candidate_answer"]
    submissions = [
        _packet_submission(packets[0], grader_id=assignments[0]["grader_id"]),
        _packet_submission(
            packets[1],
            {disputed_answer: "C"},
            grader_id=assignments[1]["grader_id"],
        ),
    ]
    pending = frozen30.reconcile_frozen30_grades(bundle, submissions)

    with pytest.raises(ValueError, match="distinct third grader"):
        frozen30.reconcile_frozen30_grades(
            bundle,
            submissions,
            adjudication={
                "grader_id": assignments[assigned_grader_index]["grader_id"],
                "packet_id": pending.adjudication_packet["packet_id"],
                "grades": [
                    {"case_id": item["case_id"], "grade": "C"}
                    for item in pending.adjudication_packet["items"]
                ],
            },
        )


def test_private_grader_assignment_binds_identity_alias_and_packet_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches discarding caller identities after generating public aliases."""
    bundle = _bundle(tmp_path, monkeypatch)
    artifact = bundle._grader_assignment_artifact

    assert artifact["schema_version"] == 1
    assert [row["grader_id"] for row in artifact["assignments"]] == [
        "grader-a",
        "grader-b",
    ]
    assert [row["grader_alias"] for row in artifact["assignments"]] == [
        "grader-1",
        "grader-2",
    ]
    for assignment in artifact["assignments"]:
        packet = bundle.packets[assignment["grader_alias"]]
        assert assignment["packet_id"] == packet["packet_id"]
        assert (
            assignment["packet_sha256"]
            == hashlib.sha256(_canonical(packet)).hexdigest()
        )
    assert "grader-a" not in json.dumps(bundle.packets)
    assert "grader-b" not in json.dumps(bundle.packets)

    alias_submissions = [
        _packet_submission(packet) for packet in bundle.packets.values()
    ]
    with pytest.raises(ValueError, match="assigned grader"):
        frozen30.reconcile_frozen30_grades(bundle, alias_submissions)


@pytest.mark.parametrize(
    ("changes", "semantic"),
    [
        ({}, True),
        ({"improvements": 5}, False),
        ({"regressions": 3}, False),
        ({"category_regressions_exceed_improvements": True}, False),
        ({"visual_control_regressions": 1}, False),
        ({"exact_match_delta_rows": -2}, False),
        ({"token_f1_delta": -0.0200001}, False),
        ({"meteor_delta": -0.0200001}, False),
        (
            {
                "exact_match_delta_rows": -1,
                "token_f1_delta": -0.02,
                "meteor_delta": -0.02,
            },
            True,
        ),
    ],
)
def test_predeclared_semantic_gate_boundaries(
    tmp_path: Path, changes: dict[str, object], semantic: bool
) -> None:
    values = {
        "improvements": 6,
        "regressions": 2,
        "category_regressions_exceed_improvements": False,
        "visual_control_regressions": 0,
        "exact_match_delta_rows": -1,
        "token_f1_delta": -0.02,
        "meteor_delta": -0.02,
    }
    values.update(changes)
    decision = frozen30.apply_decision_rule(
        **values,
        operational=_derived_operational(tmp_path, elapsed_seconds=3600.0, cost=1.25),
    )
    assert decision["semantic_gate_passed"] is semantic
    assert decision["disposition"] == (
        "Current default" if semantic else "Not promoted under the tested condition"
    )


def test_semantic_pass_without_operational_pass_is_routed_candidate(
    tmp_path: Path,
) -> None:
    decision = frozen30.apply_decision_rule(
        improvements=6,
        regressions=2,
        category_regressions_exceed_improvements=False,
        visual_control_regressions=0,
        exact_match_delta_rows=-1,
        token_f1_delta=-0.02,
        meteor_delta=-0.02,
        operational=_derived_operational(tmp_path, elapsed_seconds=3600.6, cost=1.25),
    )
    assert decision["disposition"] == "Routed candidate"


def test_metric_boundary_rejects_value_below_exact_negative_point_zero_two(
    tmp_path: Path,
) -> None:
    decision = frozen30.apply_decision_rule(
        improvements=6,
        regressions=2,
        category_regressions_exceed_improvements=False,
        visual_control_regressions=0,
        exact_match_delta_rows=-1,
        token_f1_delta=-0.020000000000000018,
        meteor_delta=-0.020000000000000018,
        operational=_derived_operational(tmp_path, elapsed_seconds=3600.0, cost=1.25),
    )
    assert decision["semantic_gate_passed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"completed": True},
        {"expected": 59},
        {"duration_minutes": -0.01},
        {"duration_minutes": math.nan},
        {"duration_minutes": math.inf},
        {"compute_cost_usd": -0.01},
        {"compute_cost_usd": math.nan},
        {"artifacts_verified": 1},
        {"privacy_or_audit_gap": 0},
    ],
)
def test_operational_evidence_rejects_malformed_or_inconsistent_values(
    changes: dict[str, object],
) -> None:
    values = {
        "completed": 60,
        "expected": 60,
        "duration_minutes": 60.0,
        "compute_cost_usd": 1.25,
        "artifacts_verified": True,
        "privacy_or_audit_gap": False,
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        frozen30.OperationalEvidence(**values)


def test_operational_evidence_is_derived_from_anchor_bound_run_and_review_artifacts(
    tmp_path: Path,
) -> None:
    """Catches caller metrics replacing collected status, duration, price, and cost."""
    predictions, manifest, _checksums = _write_candidate(tmp_path)
    operational = frozen30.derive_operational_evidence(
        candidate_predictions_path=predictions,
        candidate_manifest_path=manifest,
        audit_bundle_root=_audit_root(tmp_path),
        privacy_review_path=_write_privacy_review(tmp_path),
    )

    assert operational.completed == operational.expected == 60
    assert operational.run_status == "done"
    assert operational.duration_minutes == 59.0
    assert operational.hourly_price_usd == 2.0
    assert operational.compute_cost_usd == 1.2
    assert operational.artifacts_verified is True
    assert operational.privacy_or_audit_gap is False


@pytest.mark.parametrize(
    "mutation",
    ["generic-sidecar", "anchor", "reviewer", "timestamp", "run", "audit"],
)
def test_forged_or_unbound_review_and_audit_cannot_authorize_promotion(
    tmp_path: Path, mutation: str
) -> None:
    """Catches a generic Boolean sidecar or coherently rewritten audit authority."""
    predictions, manifest, _checksums = _write_candidate(tmp_path)
    review_path = _write_privacy_review(tmp_path)
    if mutation == "generic-sidecar":
        review_path.write_text(
            json.dumps(
                {
                    "completed": 60,
                    "expected": 60,
                    "duration_minutes": 1.0,
                    "compute_cost_usd": 0.0,
                    "artifacts_verified": True,
                    "privacy_or_audit_gap": False,
                }
            )
        )
    elif mutation in {"anchor", "reviewer", "timestamp", "run"}:
        review = json.loads(review_path.read_text())
        field, value = {
            "anchor": ("workload_anchor_sha256", "a" * 64),
            "reviewer": ("reviewer_id", ""),
            "timestamp": ("reviewed_at", "2026-08-01"),
            "run": ("run_id", "other-run"),
        }[mutation]
        review[field] = value
        review_path.write_text(json.dumps(review, sort_keys=True) + "\n")
    else:
        dstack_path = _audit_root(tmp_path) / "dstack-run.json"
        dstack = json.loads(dstack_path.read_text())
        dstack["cost"] = 0.0
        dstack_path.write_text(json.dumps(dstack, sort_keys=True) + "\n")
        audit_root = _audit_root(tmp_path)
        files = sorted(
            path
            for path in audit_root.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (audit_root / "SHA256SUMS").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.name}\n"
                for path in files
            )
        )

    with pytest.raises(ValueError, match="schema|anchor|reviewer|timestamp|run|audit"):
        frozen30.derive_operational_evidence(
            candidate_predictions_path=predictions,
            candidate_manifest_path=manifest,
            audit_bundle_root=_audit_root(tmp_path),
            privacy_review_path=review_path,
        )


def test_decision_rule_rejects_unvalidated_operational_objects() -> None:
    malformed = SimpleNamespace(
        completed=60,
        expected=60,
        duration_minutes=-1,
        compute_cost_usd=-1,
        artifacts_verified=1,
        privacy_or_audit_gap=0,
    )
    with pytest.raises(TypeError, match="OperationalEvidence"):
        frozen30.apply_decision_rule(
            improvements=6,
            regressions=2,
            category_regressions_exceed_improvements=False,
            visual_control_regressions=0,
            exact_match_delta_rows=-1,
            token_f1_delta=-0.02,
            meteor_delta=-0.02,
            operational=malformed,
        )


def test_private_score_reports_automatic_cpw_category_and_completion_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches omitting a preregistered output family from the private score."""
    bundle = _bundle(tmp_path, monkeypatch)
    for row in bundle._rows:
        for prediction in row["predictions"].values():
            prediction["answer"] = row["reference"]
    grades = {key: "P" for key in bundle._canonical_keys}
    for video_id, _category, _role in FROZEN30_ROWS[:6]:
        grades[(video_id, "audio-on")] = "C"
    resolution = GradeResolution("final", grades, None)

    report = frozen30.score_replacement(
        bundle,
        resolution,
        meteor_fn=lambda prediction, reference: float(prediction == reference),
        operational=_derived_operational(tmp_path),
    )

    assert report["completion"] == {"completed": 60, "expected": 60}
    assert set(report["conditions"]) == {"qwen12", "audio-off", "audio-on"}
    assert set(report["conditions"]["audio-on"]) == {
        "exact_match",
        "token_f1",
        "meteor",
    }
    assert report["grade_counts"]["audio-on"] == {"C": 6, "P": 24, "W": 0}
    assert set(report["category_metrics"]) == {
        "Dialogue and speech",
        "Non-speech sound and music",
        "Temporal and action",
        "Spatial, OCR, and color",
        "Visual controls",
    }
    assert set(report["category_metrics"]["Visual controls"]) == {
        "qwen12",
        "audio-off",
        "audio-on",
    }
    assert report["disposition"] == "Current default"
