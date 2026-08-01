from __future__ import annotations

import hashlib
import json
from pathlib import Path

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


def _candidate_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "frozen30-paired",
        "panel_id": "nemotron3-omni-frozen30",
        "source_manifest_sha256": FROZEN30_INFERENCE_MANIFEST_SHA256,
        "source_contract_sha256": "7" * 64,
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
                "question": f"question {index}",
                "source_media_path": FROZEN30_SOURCE_MEDIA_PATHS[video_id],
                "source_media_sha256": "5" * 64,
                "media_path": f"{video_id}.mp4",
                "derived_media_sha256": "8" * 64,
                "source_duration_seconds": 10.0,
                "derived_duration_seconds": 10.0,
            }
            for index, (video_id, category, role) in enumerate(FROZEN30_ROWS)
        ],
    }


def _write_candidate(root: Path) -> tuple[Path, Path, Path]:
    contract = _candidate_contract()
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


def _references(keys: tuple[str, ...]) -> list[dict[str, str]]:
    by_id = {video_id: category for video_id, category, _role in FROZEN30_ROWS}
    return [
        {
            "row_key": key,
            "category": by_id[key],
            "question": f"question {index}",
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


def test_metric_boundary_tolerates_only_binary_floating_point_noise() -> None:
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
    assert decision["semantic_gate_passed"] is True


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
