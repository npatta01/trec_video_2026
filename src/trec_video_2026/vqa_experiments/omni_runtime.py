"""Model-specific request formatting for joint image/audio generation."""

from __future__ import annotations

import hashlib
from typing import Any

from .omni_contract import normalize_prediction_output

OUTPUT_INSTRUCTION = (
    "Answer the question using the visual frames and the audio together. "
    'Return only a JSON object with non-empty string fields "answer" and '
    '"evidence". The evidence field must briefly say which visual and audio '
    "cues support the answer."
)

NEUTRAL_OUTPUT_INSTRUCTION = (
    "Answer the question using the provided evidence. "
    'Return only a JSON object with non-empty string fields "answer" and '
    '"evidence". The evidence field must briefly state the cues supporting '
    "the answer."
)

VISUAL_CONCISE_OUTPUT_INSTRUCTION = (
    "Answer the question using the provided visual evidence. "
    'Return only a JSON object with non-empty string fields "answer" and '
    '"evidence". The "answer" field must contain only the direct answer in '
    'one short phrase or sentence, without explanation. The "evidence" field '
    "must briefly state the visual cues supporting the answer."
)

INSTRUCTION_PROFILES = {
    "audio-expecting": OUTPUT_INSTRUCTION,
    "neutral": NEUTRAL_OUTPUT_INSTRUCTION,
    "visual-concise": VISUAL_CONCISE_OUTPUT_INSTRUCTION,
}


def instruction_for_profile(profile: str) -> str:
    """Return one immutable instruction profile selected by a run wrapper."""
    try:
        return INSTRUCTION_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unknown instruction profile: {profile}") from error


def phi_prompt(example: dict[str, Any]) -> str:
    image_tokens = "".join(
        f"<|image_{index}|>" for index, _ in enumerate(example["frame_paths"], start=1)
    )
    return (
        f"<|user|>{image_tokens}<|audio_1|>"
        f"{OUTPUT_INSTRUCTION}\nQuestion: {example['question']}"
        "<|end|><|assistant|>"
    )


def gemma_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": path} for path in example["frame_paths"]
    ]
    content.append({"type": "audio", "audio": example["audio_path"]})
    content.append(
        {
            "type": "text",
            "text": f"{OUTPUT_INSTRUCTION}\nQuestion: {example['question']}",
        }
    )
    return [{"role": "user", "content": content}]


def _input_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    candidate = value
    for method in ("detach", "cpu", "contiguous"):
        function = getattr(candidate, method, None)
        if function is not None:
            candidate = function()
    numpy = getattr(candidate, "numpy", None)
    if numpy is not None:
        candidate = numpy()
    tobytes = getattr(candidate, "tobytes", None)
    if tobytes is None:
        raise TypeError(f"cannot attest input of type {type(value).__name__}")
    return tobytes()


def _inputs_sha256(inputs: dict[str, Any], keys: set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_input_bytes(inputs[key]))
    return digest.hexdigest()


def _leading_element_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape):
        count = int(shape[0])
    else:
        try:
            count = len(value)
        except TypeError as error:
            raise TypeError(
                "modality input has no leading element dimension"
            ) from error
    if count < 1:
        raise ValueError("modality input must contain at least one element")
    return count


def attest_generation_inputs(
    processor_inputs: dict[str, Any],
    *,
    first_forward_input_keys: set[str],
    vision_input_keys: set[str],
    audio_input_keys: set[str],
) -> dict[str, Any]:
    """Attest that processed image and audio tensors reach one first forward."""

    available = {key for key, value in processor_inputs.items() if value is not None}
    used_vision = available & vision_input_keys
    used_audio = available & audio_input_keys
    if len(used_vision) != 1 or len(used_audio) != 1:
        raise ValueError("processor inputs must contain both vision and audio")
    if not used_vision.issubset(first_forward_input_keys) or not (
        used_audio.issubset(first_forward_input_keys)
    ):
        raise ValueError("vision and audio must be present in the same first forward")
    return {
        "same_request": True,
        "same_forward_pass": True,
        "generation_call_count": 1,
        "generation_input_keys": sorted(available),
        "first_forward_input_keys": sorted(first_forward_input_keys),
        "vision_input_keys": sorted(used_vision),
        "audio_input_keys": sorted(used_audio),
        "vision_element_count": _leading_element_count(
            processor_inputs[next(iter(used_vision))]
        ),
        "audio_element_count": _leading_element_count(
            processor_inputs[next(iter(used_audio))]
        ),
        "vision_input_sha256": _inputs_sha256(processor_inputs, used_vision),
        "audio_input_sha256": _inputs_sha256(processor_inputs, used_audio),
    }


def build_prediction_record(
    *,
    example: dict[str, Any],
    request_digest: str,
    model_id: str,
    model_revision: str,
    runtime_image_digest: str,
    run_id: str,
    device: str,
    precision: str,
    elapsed_seconds: float,
    peak_memory_bytes: int,
    raw_output: str,
    modality_attestation: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": example["video_id"],
        "operational_tag": example["operational_tag"],
        "request_sha256": request_digest,
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime_image_digest": runtime_image_digest,
        "device": device,
        "precision": precision,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "raw_output": raw_output,
        "input_contract_error": False,
        "modality_attestation": modality_attestation,
    }
    try:
        record["parsed_output"] = normalize_prediction_output(raw_output)
    except ValueError as error:
        record.update(
            {
                "status": "failed",
                "output_parse_status": "failed",
                "error_type": "output_parse_error",
                "error_message": str(error),
            }
        )
    else:
        record.update(
            {
                "status": "passed",
                "output_parse_status": "passed",
                "error_type": None,
                "error_message": None,
            }
        )
    return record


def build_runtime_failure_record(
    *,
    example: dict[str, Any],
    request_digest: str,
    model_id: str,
    model_revision: str,
    runtime_image_digest: str,
    run_id: str,
    device: str,
    precision: str,
    elapsed_seconds: float,
    peak_memory_bytes: int,
    error: Exception,
    input_contract_error: bool | None,
    failure_stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": example["video_id"],
        "operational_tag": example["operational_tag"],
        "request_sha256": request_digest,
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime_image_digest": runtime_image_digest,
        "device": device,
        "precision": precision,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "status": "failed",
        "raw_output": "",
        "output_parse_status": "not_attempted",
        "input_contract_error": input_contract_error,
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "modality_attestation": {
            "same_request": False,
            "same_forward_pass": False,
            "generation_call_count": 0,
            "generation_input_keys": [],
            "first_forward_input_keys": [],
            "vision_input_keys": [],
            "audio_input_keys": [],
            "vision_element_count": 0,
            "audio_element_count": 0,
            "vision_input_sha256": None,
            "audio_input_sha256": None,
        },
    }
