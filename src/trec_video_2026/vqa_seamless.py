"""Dedicated SeamlessM4T speech-to-English translation runner."""

from __future__ import annotations

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


def decode_translation(processor: Any, generated_tokens: Any) -> str:
    text = processor.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if not text:
        raise ValueError("SeamlessM4T returned an empty translation")
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
