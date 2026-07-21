"""Dedicated SeamlessM4T speech-to-English translation runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODEL_ID = "facebook/seamless-m4t-v2-large"
MODEL_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
DEVICE = "cuda:0"
PRECISION = "float16"


def request_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Copy only the protocol identity fields from a runner request."""

    return {
        field: request[field]
        for field in ("run_id", "condition_id", "video_id")
    }


def prepare_audio_inputs(
    processor: Any, audio: Any, *, device: Any, dtype: Any
) -> Any:
    inputs = processor(audio=audio, sampling_rate=16_000, return_tensors="pt")
    return inputs.to(device=device, dtype=dtype)


def decode_translation(processor: Any, generated_tokens: Any) -> str:
    text = processor.decode(
        generated_tokens[0], skip_special_tokens=True
    ).strip()
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
    inputs = prepare_audio_inputs(
        processor, audio, device=device, dtype=dtype
    )
    with inference_mode():
        generated = model.generate(**inputs, tgt_lang="eng")
    return decode_translation(processor, generated)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    return parser.parse_args(argv)


def _result(
    *,
    request: dict[str, Any],
    status: str,
    source_audio_sha256: str | None,
    text: str | None,
    quality: dict[str, Any] | None,
    started: float,
    model_id: str,
    revision: str,
    runtime: str,
    peak_memory_bytes: int,
    error: Exception | None = None,
) -> dict[str, Any]:
    message = str(error)[:4_000] if error else None
    return {
        **request_identity(request),
        "status": status,
        "source_audio_sha256": source_audio_sha256,
        "text": text,
        "quality": quality,
        "elapsed_seconds": time.monotonic() - started,
        "model_id": model_id,
        "model_revision": revision,
        "runtime": runtime,
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "device": "cuda:0 (ROCm)",
        "precision": PRECISION,
        "peak_memory_bytes": peak_memory_bytes,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
    }


def main(argv: list[str] | None = None) -> int:
    """Read one JSON request and emit exactly one JSON result."""

    import soundfile
    import torch
    import transformers
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    from trec_video_2026.vqa_benchmark import source_fingerprint
    from trec_video_2026.vqa_whisper import load_wav_audio, transcript_quality

    args = parse_args(argv)
    request = json.load(sys.stdin)
    started = time.monotonic()
    runtime = (
        f"transformers {transformers.__version__}; torch {torch.__version__}"
    )
    source_audio_sha256 = None
    torch.cuda.reset_peak_memory_stats()
    try:
        audio_path = Path(request["audio_path"])
        source_audio_sha256 = source_fingerprint(audio_path).sha256
        processor = AutoProcessor.from_pretrained(
            args.model_id, revision=args.revision
        )
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        audio = load_wav_audio(audio_path, soundfile)
        text = translate_audio(
            model,
            processor,
            audio,
            device=DEVICE,
            dtype=torch.float16,
            inference_mode=torch.inference_mode,
        )
        quality = transcript_quality(
            text, duration_seconds=len(audio) / 16_000
        )
        result = _result(
            request=request,
            status="passed",
            source_audio_sha256=source_audio_sha256,
            text=text,
            quality=quality,
            started=started,
            model_id=args.model_id,
            revision=args.revision,
            runtime=runtime,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        result = _result(
            request=request,
            status="failed",
            source_audio_sha256=source_audio_sha256,
            text=None,
            quality=None,
            started=started,
            model_id=args.model_id,
            revision=args.revision,
            runtime=runtime,
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            error=error,
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
