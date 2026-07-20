"""Timestamped Whisper transcription for the frozen VQA evidence cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def load_wav_audio(audio_path: Path, soundfile_module: Any) -> Any:
    audio, sample_rate = soundfile_module.read(str(audio_path), dtype="float32")
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16000 Hz audio, found {sample_rate}")
    if getattr(audio, "ndim", 1) != 1:
        raise ValueError("Expected mono audio")
    return audio


def transcribe_audio(asr_pipeline: Any, audio_input: Any) -> dict[str, Any]:
    pipeline_input = str(audio_input) if isinstance(audio_input, Path) else audio_input
    output = asr_pipeline(
        pipeline_input,
        return_timestamps=True,
        generate_kwargs={"task": "translate"},
    )
    chunks = []
    rendered = []
    for chunk in output.get("chunks", []):
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        chunks.append({"start": start, "end": end, "text": text})
        rendered.append(f"[{format_timestamp(start)}] {text}")
    return {
        "text": output.get("text", "").strip(),
        "timestamped_text": "\n".join(rendered),
        "chunks": chunks,
    }


def transcript_quality(text: str, *, duration_seconds: float) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("Transcript duration must be positive")
    word_count = len(text.split())
    words_per_second = word_count / duration_seconds
    status = "rejected-hallucination" if words_per_second > 5.0 else "accepted"
    return {
        "status": status,
        "word_count": word_count,
        "words_per_second": words_per_second,
        "rule": "reject when whitespace-delimited words exceed 5 per second",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--revision", default="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch
    import transformers
    import soundfile
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )

    args = parse_args(argv)
    request = json.load(sys.stdin)
    audio_path = Path(request["audio_path"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        processor = AutoProcessor.from_pretrained(
            args.model_id, revision=args.revision
        )
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch.float16,
            device=0,
        )
        audio = load_wav_audio(audio_path, soundfile)
        transcript = transcribe_audio(asr, audio)
        quality = transcript_quality(
            transcript["text"], duration_seconds=len(audio) / 16_000
        )
        result = {
            "status": "passed",
            **transcript,
            "quality": quality,
            "elapsed_seconds": time.monotonic() - started,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "runtime": (
                f"transformers {transformers.__version__}; torch {torch.__version__}"
            ),
            "device": "cuda:0 (ROCm)",
            "precision": "float16",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "error_type": None,
            "error_message": None,
        }
    except Exception as error:
        message = str(error)
        result = {
            "status": "failed",
            "text": None,
            "timestamped_text": None,
            "chunks": [],
            "elapsed_seconds": time.monotonic() - started,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "runtime": (
                f"transformers {transformers.__version__}; torch {torch.__version__}"
            ),
            "device": "cuda:0 (ROCm)",
            "precision": "float16",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "error_type": type(error).__name__,
            "error_message": message[:4_000],
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
