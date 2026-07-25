import array

import pytest

from trec_video_2026.vqa_experiments.omni_runtime import (
    attest_generation_inputs,
    build_prediction_record,
    build_runtime_failure_record,
    gemma_messages,
    phi_prompt,
)


def _example() -> dict:
    return {
        "video_id": "example-video",
        "question": "What is happening and what is heard?",
        "frame_paths": [f"frames/frame-{index:02d}.png" for index in range(1, 13)],
        "audio_path": "audio/example.wav",
    }


def test_phi_prompt_places_frames_and_audio_in_one_user_turn():
    prompt = phi_prompt(_example())

    assert prompt.startswith("<|user|>")
    assert prompt.endswith("<|end|><|assistant|>")
    assert prompt.count("<|image_") == 12
    assert prompt.count("<|audio_") == 1
    assert prompt.index("<|image_12|>") < prompt.index("<|audio_1|>")
    assert "What is happening and what is heard?" in prompt
    assert "JSON object" in prompt


def test_gemma_messages_put_frames_audio_and_question_in_one_message():
    messages = gemma_messages(_example())

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content].count("image") == 12
    assert [item["type"] for item in content].count("audio") == 1
    assert [item["type"] for item in content].count("text") == 1
    assert "What is happening and what is heard?" in content[-1]["text"]


def test_attestation_requires_both_modalities_in_first_forward():
    processor_inputs = {
        "pixel_values": array.array("f", [1.0, 2.0]),
        "input_features": array.array("f", [3.0, 4.0]),
        "input_ids": array.array("l", [1, 2]),
    }

    attestation = attest_generation_inputs(
        processor_inputs,
        first_forward_input_keys={"pixel_values", "input_features", "input_ids"},
        vision_input_keys={"pixel_values"},
        audio_input_keys={"input_features"},
    )

    assert attestation["same_request"] is True
    assert attestation["same_forward_pass"] is True
    assert attestation["generation_call_count"] == 1
    assert attestation["vision_element_count"] == 2
    assert attestation["audio_element_count"] == 2
    assert attestation["vision_input_sha256"] != attestation["audio_input_sha256"]


def test_attestation_rejects_separate_modality_forwards():
    with pytest.raises(ValueError, match="same first forward"):
        attest_generation_inputs(
            {
                "pixel_values": array.array("f", [1.0]),
                "input_features": array.array("f", [2.0]),
            },
            first_forward_input_keys={"pixel_values"},
            vision_input_keys={"pixel_values"},
            audio_input_keys={"input_features"},
        )


def test_prediction_record_marks_unparseable_structured_output_failed():
    record = build_prediction_record(
        example={
            "video_id": "example-video",
            "operational_tag": "prosody",
            "question": "Question?",
        },
        request_digest="a" * 64,
        model_id="example/model",
        model_revision="b" * 40,
        runtime_image_digest="sha256:" + "c" * 64,
        run_id="run",
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        elapsed_seconds=1.0,
        peak_memory_bytes=100,
        raw_output='{"answer": "not terminated"',
        modality_attestation={"same_request": True},
    )

    assert record["status"] == "failed"
    assert record["output_parse_status"] == "failed"
    assert record["error_type"] == "output_parse_error"


def test_prediction_record_accepts_nonempty_plain_text_answer():
    record = build_prediction_record(
        example={
            "video_id": "example-video",
            "operational_tag": "non_speech_sound",
            "question": "Question?",
        },
        request_digest="a" * 64,
        model_id="example/model",
        model_revision="b" * 40,
        runtime_image_digest="sha256:" + "c" * 64,
        run_id="run",
        device="cuda:0 (NVIDIA CUDA)",
        precision="bfloat16",
        elapsed_seconds=1.0,
        peak_memory_bytes=100,
        raw_output="meow",
        modality_attestation={"same_request": True},
    )

    assert record["status"] == "passed"
    assert record["output_parse_status"] == "passed"
    assert record["parsed_output"] == {
        "answer": "meow",
        "evidence": None,
        "output_format": "plain_text",
    }


def test_prediction_record_rejects_empty_plain_text_answer():
    record = build_prediction_record(
        example={
            "video_id": "example-video",
            "operational_tag": "non_speech_sound",
            "question": "Question?",
        },
        request_digest="a" * 64,
        model_id="example/model",
        model_revision="b" * 40,
        runtime_image_digest="sha256:" + "c" * 64,
        run_id="run",
        device="cuda:0 (NVIDIA CUDA)",
        precision="bfloat16",
        elapsed_seconds=1.0,
        peak_memory_bytes=100,
        raw_output="answer: \n",
        modality_attestation={"same_request": True},
    )

    assert record["status"] == "failed"
    assert record["output_parse_status"] == "failed"


def test_runtime_failure_record_preserves_attempt_attestation():
    record = build_runtime_failure_record(
        example={
            "video_id": "example-video",
            "operational_tag": "lexical_speech",
        },
        request_digest="a" * 64,
        model_id="example/model",
        model_revision="b" * 40,
        runtime_image_digest="sha256:" + "c" * 64,
        run_id="run",
        device="cuda:0 (ROCm)",
        precision="bfloat16",
        elapsed_seconds=2.0,
        peak_memory_bytes=100,
        error=RuntimeError("verified ROCm failure"),
        input_contract_error=True,
        failure_stage="attestation",
    )

    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert record["failure_stage"] == "attestation"
    assert record["input_contract_error"] is True
    assert record["modality_attestation"]["same_request"] is False
    assert record["modality_attestation"]["generation_call_count"] == 0
