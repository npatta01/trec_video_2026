"""Dedicated SeamlessM4T speech-to-English translation runner."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/seamless-m4t-v2-large"
MODEL_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
DEVICE = "cuda:0"
PRECISION = "bfloat16"
TEXT_TRANSLATION_IMAGE = (
    "localhost/trec-vqa-gate-rocm@sha256:"
    "510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5"
)


def translation_request_sha256(value: Any) -> str:
    """Hash a translation protocol value with canonical JSON encoding."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def translation_execution_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Bind every source, prompt, runtime, and deterministic decoding field."""

    return {
        "request": request,
        "execution": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": TEXT_TRANSLATION_IMAGE,
            "translation_mode": "text_to_text",
            "processor_interface": "AutoProcessor(text,src_lang)",
            "target_language": "English",
            "target_language_code": "eng",
            "generate_speech": False,
            "do_sample": False,
            "device": "cuda:0 (ROCm)",
            "precision": PRECISION,
        },
    }


def validate_translation_attempt(
    attempt: dict[str, Any], request: dict[str, Any]
) -> bool:
    """Return whether an attempt exactly satisfies the frozen text protocol."""

    execution = translation_execution_identity(request)["execution"]
    translated_text = attempt.get("translated_text")
    runtime = attempt.get("runtime")
    expected = {
        "status": "passed",
        "request": request,
        "request_sha256": translation_request_sha256(request),
        "reuse_identity_sha256": translation_request_sha256(
            translation_execution_identity(request)
        ),
        "run_id": request.get("run_id"),
        "condition_id": request.get("condition_id"),
        "video_id": request.get("video_id"),
        "original_text": request.get("original_text"),
        "source_attempt_sha256": request.get("source_attempt_sha256"),
        "source_language": request.get("source_language"),
        "source_language_code": request.get("source_language_code"),
        **execution,
    }
    return (
        all(attempt.get(field) == value for field, value in expected.items())
        and isinstance(translated_text, str)
        and bool(translated_text.strip())
        and isinstance(runtime, str)
        and bool(runtime.strip())
        and attempt.get("error_type") is None
        and attempt.get("error_message") is None
    )


def select_translation_attempts(
    attempts: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select exactly one fully valid append-only attempt per expected request."""

    selected = []
    for request in requests:
        valid = [
            attempt
            for attempt in attempts
            if validate_translation_attempt(attempt, request)
        ]
        if len(valid) != 1:
            raise ValueError(
                "Expected exactly one valid translation for "
                f"{request.get('video_id')}; found {len(valid)}"
            )
        selected.append(valid[0])
    if len(selected) != len(requests):
        raise ValueError("Translation selection count mismatch")
    return selected


def request_identity(request: Any) -> dict[str, Any]:
    """Copy only the protocol identity fields from a runner request."""

    payload = request if isinstance(request, dict) else {}
    identity = {}
    for field in ("run_id", "condition_id", "video_id"):
        value = payload.get(field)
        identity[field] = value if isinstance(value, str) and value.strip() else None
    return identity


def validate_request(request: dict[str, Any]) -> None:
    for field in ("run_id", "condition_id", "video_id"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")

    audio_path = request.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise ValueError("audio_path must be a nonempty absolute path")
    if not Path(audio_path).is_absolute():
        raise ValueError("audio_path must be a nonempty absolute path")


def prepare_audio_inputs(processor: Any, audio: Any, *, device: Any, dtype: Any) -> Any:
    inputs = processor(audio=audio, sampling_rate=16_000, return_tensors="pt")
    return inputs.to(device=device, dtype=dtype)


def prepare_text_inputs(
    processor: Any,
    text: str,
    *,
    source_language: str,
    device: Any,
) -> Any:
    if not text.strip():
        raise ValueError("Translation source text cannot be empty")
    if not source_language.strip():
        raise ValueError("Translation source language cannot be empty")
    inputs = processor(text=text, src_lang=source_language, return_tensors="pt")
    return inputs.to(device=device)


def decode_translation(processor: Any, generated_tokens: Any) -> str:
    text = processor.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if not text:
        raise ValueError("SeamlessM4T returned an empty translation")
    return text


def decode_text_translation(processor: Any, generated_tokens: Any) -> str:
    sequence = generated_tokens[0]
    if hasattr(sequence, "tolist"):
        sequence = sequence.tolist()
    if sequence and isinstance(sequence[0], list):
        sequence = sequence[0]
    text = processor.decode(sequence, skip_special_tokens=True).strip()
    if not text:
        raise ValueError("SeamlessM4T returned an empty text translation")
    return text


def translate_audio(
    model: Any,
    processor: Any,
    audio: Any,
    *,
    device: Any,
    dtype: Any,
    inference_mode: Any,
) -> str:
    inputs = prepare_audio_inputs(processor, audio, device=device, dtype=dtype)
    with inference_mode():
        generated = model.generate(**inputs, tgt_lang="eng")
    return decode_translation(processor, generated)


def translate_text(
    model: Any,
    processor: Any,
    text: str,
    *,
    source_language: str,
    device: Any,
    inference_mode: Any,
) -> str:
    inputs = prepare_text_inputs(
        processor,
        text,
        source_language=source_language,
        device=device,
    )
    with inference_mode():
        generated = model.generate(
            **inputs,
            tgt_lang="eng",
            generate_speech=False,
            do_sample=False,
        )
    return decode_text_translation(processor, generated)


def validate_cli_args(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        raise ValueError("This runner uses a pinned model and revision")


def peak_memory_bytes(torch_module: Any) -> int | None:
    if torch_module is None:
        return None
    try:
        return torch_module.cuda.max_memory_allocated()
    except Exception:
        return None


def _result(
    *,
    request: Any,
    status: str,
    source_audio_sha256: str | None,
    text: str | None,
    quality: dict[str, Any] | None,
    started: float,
    runtime: str,
    peak_memory_bytes: int | None,
    error: Exception | None = None,
    device: str = "unverified",
    precision: str = "unverified",
) -> dict[str, Any]:
    message = str(error)[:4_000] if error else None
    return {
        **request_identity(request),
        "status": status,
        "source_audio_sha256": source_audio_sha256,
        "text": text,
        "quality": quality,
        "elapsed_seconds": time.monotonic() - started,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime": runtime,
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "device": device,
        "precision": precision,
        "peak_memory_bytes": peak_memory_bytes,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
    }


def main(argv: list[str] | None = None) -> int:
    """Read one JSON request and emit exactly one JSON result."""

    started = time.monotonic()
    request: Any = {}
    runtime = "unavailable"
    source_audio_sha256 = None
    torch_module = None
    device = "unverified"
    precision = "unverified"
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise TypeError("Runner request must be a JSON object")
        validate_request(request)
        validate_cli_args(argv)

        import soundfile
        import torch
        import transformers
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        from trec_video_2026.vqa_benchmark import source_fingerprint
        from trec_video_2026.vqa_whisper import (
            load_wav_audio,
            transcript_quality,
            validate_rocm_bfloat16,
        )

        torch_module = torch
        runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
        torch.cuda.reset_peak_memory_stats()
        audio_path = Path(request["audio_path"])
        source_audio_sha256 = source_fingerprint(audio_path).sha256
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        device, precision = validate_rocm_bfloat16(model, torch)
        audio = load_wav_audio(audio_path, soundfile)
        text = translate_audio(
            model,
            processor,
            audio,
            device=DEVICE,
            dtype=torch.bfloat16,
            inference_mode=torch.inference_mode,
        )
        quality = transcript_quality(text, duration_seconds=len(audio) / 16_000)
        result = _result(
            request=request,
            status="passed",
            source_audio_sha256=source_audio_sha256,
            text=text,
            quality=quality,
            started=started,
            runtime=runtime,
            peak_memory_bytes=peak_memory_bytes(torch_module),
            device=device,
            precision=precision,
        )
    except Exception as error:
        result = _result(
            request=request,
            status="failed",
            source_audio_sha256=source_audio_sha256,
            text=None,
            quality=None,
            started=started,
            runtime=runtime,
            peak_memory_bytes=peak_memory_bytes(torch_module),
            error=error,
            device=device,
            precision=precision,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
