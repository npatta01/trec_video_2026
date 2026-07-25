import copy
import hashlib
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.omni_contract import (
    DEFAULT_PANEL,
    build_compatibility_contract,
    contract_sha256,
    normalize_prediction_output,
    parse_prediction,
    promotion_decision,
    request_sha256,
    validate_contract,
    validate_inference_evidence,
    validate_prediction,
)


def _write_panel_fixture(data_root: Path) -> None:
    for selection in DEFAULT_PANEL:
        annotation = (
            data_root / "vqa/2025/training/annotations" / f"{selection.video_id}.json"
        )
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            json.dumps(
                {
                    "question": f"Question for {selection.operational_tag}?",
                    "correct_answer": "must never enter the contract",
                    "incorrect_answers": ["also private to scoring"],
                }
            ),
            encoding="utf-8",
        )

        media = data_root / selection.media_path
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"media:{selection.video_id}".encode())

        audio = data_root / selection.audio_path
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(f"audio:{selection.video_id}".encode())

        frame_dir = data_root / selection.frame_dir
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 13):
            (frame_dir / f"frame-{index:02d}.png").write_bytes(
                f"frame:{selection.video_id}:{index}".encode()
            )


def _valid_prediction(example: dict) -> dict:
    return {
        "schema_version": 1,
        "run_id": "test-run",
        "video_id": example["video_id"],
        "operational_tag": example["operational_tag"],
        "request_sha256": request_sha256(example),
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "runtime_image_digest": "sha256:" + "b" * 64,
        "device": "cuda:0 (ROCm; Radeon 8060S Graphics)",
        "precision": "bfloat16",
        "elapsed_seconds": 1.25,
        "peak_memory_bytes": 1234,
        "status": "passed",
        "raw_output": '{"answer":"example","evidence":"audio and frames"}',
        "parsed_output": {
            "answer": "example",
            "evidence": "audio and frames",
        },
        "output_parse_status": "passed",
        "input_contract_error": False,
        "modality_attestation": {
            "same_request": True,
            "same_forward_pass": True,
            "generation_call_count": 1,
            "generation_input_keys": [
                "input_ids",
                "pixel_values",
                "input_features",
            ],
            "first_forward_input_keys": [
                "input_ids",
                "pixel_values",
                "input_features",
            ],
            "vision_input_keys": ["pixel_values"],
            "audio_input_keys": ["input_features"],
            "vision_element_count": 12,
            "audio_element_count": 1,
            "vision_input_sha256": "c" * 64,
            "audio_input_sha256": "d" * 64,
        },
    }


def test_contract_binds_three_roles_to_frames_and_audio(tmp_path: Path):
    _write_panel_fixture(tmp_path)

    contract = build_compatibility_contract(tmp_path)

    assert [row["operational_tag"] for row in contract["examples"]] == [
        "lexical_speech",
        "prosody",
        "non_speech_sound",
    ]
    assert all(len(row["frame_sha256s"]) == 12 for row in contract["examples"])
    assert all(len(row["audio_sha256"]) == 64 for row in contract["examples"])
    assert all(len(row["media_sha256"]) == 64 for row in contract["examples"])
    encoded = json.dumps(contract).lower()
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded
    validate_contract(contract, tmp_path, verify_files=True)


def test_contract_hash_is_canonical(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    contract = build_compatibility_contract(tmp_path)
    expected = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()

    assert contract_sha256(contract) == expected


def test_contract_rejects_missing_modality_binding(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    contract = build_compatibility_contract(tmp_path)
    broken = copy.deepcopy(contract)
    broken["examples"][0]["audio_sha256"] = ""

    with pytest.raises(ValueError, match="audio_sha256"):
        validate_contract(broken, tmp_path)


def test_inference_evidence_does_not_require_unused_source_video(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    contract = build_compatibility_contract(tmp_path)
    for row in contract["examples"]:
        (tmp_path / row["media_path"]).unlink()

    validate_inference_evidence(contract, tmp_path, verify_files=True)

    missing_frame = tmp_path / contract["examples"][0]["frame_paths"][0]
    missing_frame.unlink()
    with pytest.raises(ValueError, match="missing frame_paths"):
        validate_inference_evidence(contract, tmp_path, verify_files=True)


def test_prediction_requires_joint_modality_forward(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    example = build_compatibility_contract(tmp_path)["examples"][0]
    prediction = _valid_prediction(example)
    prediction["modality_attestation"]["same_forward_pass"] = False

    with pytest.raises(ValueError, match="same forward pass"):
        validate_prediction(prediction, example)


def test_prediction_requires_runtime_attestation(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    example = build_compatibility_contract(tmp_path)["examples"][0]
    prediction = _valid_prediction(example)
    del prediction["runtime_image_digest"]

    with pytest.raises(ValueError, match="runtime_image_digest"):
        validate_prediction(prediction, example)


def test_prediction_rejects_unbound_request(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    example = build_compatibility_contract(tmp_path)["examples"][0]
    prediction = _valid_prediction(example)
    prediction["request_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="request_sha256"):
        validate_prediction(prediction, example)


def test_parse_prediction_accepts_fenced_json():
    parsed = parse_prediction(
        '```json\n{"answer": "a cat", "evidence": "heard a scream"}\n```'
    )

    assert parsed == {"answer": "a cat", "evidence": "heard a scream"}


def test_parse_prediction_rejects_plain_text():
    with pytest.raises(ValueError, match="JSON object"):
        parse_prediction("probably a cat")


def test_normalize_prediction_output_accepts_nonempty_plain_text():
    assert normalize_prediction_output("  answer: probably a cat  ") == {
        "answer": "probably a cat",
        "evidence": None,
        "output_format": "plain_text",
    }


def test_normalize_prediction_output_accepts_plain_text_fence():
    assert normalize_prediction_output("```text\nmeow\n```") == {
        "answer": "meow",
        "evidence": None,
        "output_format": "plain_text",
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        "",
        " \n\t ",
        "answer: ",
        '{"answer": "unterminated"',
        'answer: {"answer": "unterminated"',
        "```json\nnot valid json\n```",
        '```text\n{"answer": "unterminated"\n```',
        "```text\nmeow",
    ],
)
def test_normalize_prediction_output_fails_closed(raw_output: str):
    with pytest.raises(ValueError):
        normalize_prediction_output(raw_output)


def test_normalize_prediction_output_preserves_structured_json():
    assert normalize_prediction_output(
        '{"answer":"a cat","evidence":"heard a scream"}'
    ) == {
        "answer": "a cat",
        "evidence": "heard a scream",
        "output_format": "json",
    }


def test_promotion_requires_all_three_joint_rows(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    contract = build_compatibility_contract(tmp_path)
    predictions = [_valid_prediction(row) for row in contract["examples"]]

    assert promotion_decision(contract, predictions)["decision"] == "promote"

    predictions[1]["modality_attestation"]["audio_element_count"] = 0
    decision = promotion_decision(contract, predictions)
    assert decision["decision"] == "stop"
    assert "prosody" in decision["failed_roles"]


def test_prediction_rejects_failed_or_unknown_status(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    example = build_compatibility_contract(tmp_path)["examples"][0]
    for status in ("failed", "unknown"):
        prediction = _valid_prediction(example)
        prediction["status"] = status

        with pytest.raises(ValueError, match="status must be passed"):
            validate_prediction(prediction, example)


def test_prediction_requires_first_forward_key_evidence(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    example = build_compatibility_contract(tmp_path)["examples"][0]
    prediction = _valid_prediction(example)
    prediction["modality_attestation"]["first_forward_input_keys"] = ["input_ids"]

    with pytest.raises(ValueError, match="first forward"):
        validate_prediction(prediction, example)


def test_promotion_rejects_failed_rows_and_duplicate_roles(tmp_path: Path):
    _write_panel_fixture(tmp_path)
    contract = build_compatibility_contract(tmp_path)
    failed = [_valid_prediction(row) for row in contract["examples"]]
    for row in failed:
        row["status"] = "failed"
        row["output_parse_status"] = "not_attempted"
    assert promotion_decision(contract, failed)["decision"] == "stop"

    predictions = [_valid_prediction(row) for row in contract["examples"]]
    predictions[2] = copy.deepcopy(predictions[1])
    decision = promotion_decision(contract, predictions)
    assert decision["decision"] == "stop"
    assert "duplicate result" in decision["reasons"]["prosody"]
