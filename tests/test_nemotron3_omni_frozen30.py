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
    for video_id, _category, _role in FROZEN30_ROWS:
        source = roots["source_media_root"] / FROZEN30_SOURCE_MEDIA_PATHS[video_id]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"canonical-source:{video_id}".encode())
        normalized = roots["normalized_media_root"] / f"{video_id}.mp4"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(f"normalized-h264-aac:{video_id}".encode())
    contract = _candidate_contract(root)
    contract_path = root / "contract.json"
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
    predictions_path = root / "predictions.jsonl"
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
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    checksums_path = root / "SHA256SUMS"
    _refresh_checksums(predictions_path, manifest_path, checksums_path)
    return predictions_path, manifest_path, checksums_path


def _refresh_checksums(predictions: Path, manifest: Path, checksums: Path) -> None:
    checksums.write_text(
        f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  ./manifest.json\n"
        f"{hashlib.sha256(predictions.read_bytes()).hexdigest()}  ./predictions.jsonl\n"
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
        **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
        )


def test_label_leakage_and_reference_stratum_mismatch_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen.jsonl"
    monkeypatch.setattr(frozen30, "QWEN_FULL_FREEZE_SHA256", _write_qwen(qwen))
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract = json.loads((tmp_path / "contract.json").read_text())
    contract["examples"][0]["reference"] = "leaked"
    (tmp_path / "contract.json").write_text(json.dumps(contract, sort_keys=True) + "\n")
    manifest_body = json.loads(manifest.read_text())
    manifest_body["contract_file_sha256"] = hashlib.sha256(
        (tmp_path / "contract.json").read_bytes()
    ).hexdigest()
    manifest_body["contract_sha256"] = hashlib.sha256(_canonical(contract)).hexdigest()
    manifest.write_text(json.dumps(manifest_body, sort_keys=True) + "\n")
    _refresh_checksums(predictions, manifest, checksums)
    with pytest.raises(ValueError, match="label-bearing"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
        )


@pytest.mark.parametrize("mutation", ["question", "ground_truth", "source_hash"])
def test_candidate_normalized_contract_is_anchored_to_pinned_source_manifest(
    tmp_path: Path, mutation: str
) -> None:
    """Catches coherent relabeling or question replacement across all inner hashes."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = tmp_path / "contract.json"
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
            **_media_roots(tmp_path),
        )


@pytest.mark.parametrize("field", ["source_media_sha256", "derived_media_sha256"])
def test_candidate_media_hashes_are_checked_against_independent_artifact_bytes(
    tmp_path: Path, field: str
) -> None:
    """Catches trusting a coherently refreshed contract instead of media bytes."""
    predictions, manifest, checksums = _write_candidate(tmp_path)
    contract_path = tmp_path / "contract.json"
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

    with pytest.raises(ValueError, match="media.*hash|hash.*media"):
        frozen30.verify_candidate_freeze(
            predictions_path=predictions,
            manifest_path=manifest,
            checksums_path=checksums,
            **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
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
        **_media_roots(tmp_path),
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
            **_media_roots(tmp_path),
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
    changes: dict[str, object], semantic: bool
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
        operational=frozen30.OperationalEvidence(
            completed=60,
            expected=60,
            duration_minutes=60,
            compute_cost_usd=1.25,
            artifacts_verified=True,
            privacy_or_audit_gap=False,
        ),
    )
    assert decision["semantic_gate_passed"] is semantic
    assert decision["disposition"] == (
        "Current default" if semantic else "Not promoted under the tested condition"
    )


def test_semantic_pass_without_operational_pass_is_routed_candidate() -> None:
    decision = frozen30.apply_decision_rule(
        improvements=6,
        regressions=2,
        category_regressions_exceed_improvements=False,
        visual_control_regressions=0,
        exact_match_delta_rows=-1,
        token_f1_delta=-0.02,
        meteor_delta=-0.02,
        operational=frozen30.OperationalEvidence(
            completed=60,
            expected=60,
            duration_minutes=60.01,
            compute_cost_usd=1.25,
            artifacts_verified=True,
            privacy_or_audit_gap=False,
        ),
    )
    assert decision["disposition"] == "Routed candidate"


def test_metric_boundary_rejects_value_below_exact_negative_point_zero_two() -> None:
    decision = frozen30.apply_decision_rule(
        improvements=6,
        regressions=2,
        category_regressions_exceed_improvements=False,
        visual_control_regressions=0,
        exact_match_delta_rows=-1,
        token_f1_delta=-0.020000000000000018,
        meteor_delta=-0.020000000000000018,
        operational=frozen30.OperationalEvidence(
            completed=60,
            expected=60,
            duration_minutes=60,
            compute_cost_usd=1.25,
            artifacts_verified=True,
            privacy_or_audit_gap=False,
        ),
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
        operational=frozen30.OperationalEvidence(
            completed=60,
            expected=60,
            duration_minutes=59.0,
            compute_cost_usd=1.20,
            artifacts_verified=True,
            privacy_or_audit_gap=False,
        ),
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
