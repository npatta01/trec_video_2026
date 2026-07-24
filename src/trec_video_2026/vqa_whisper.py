"""Timestamped Whisper transcription for the frozen VQA evidence cache."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

WHISPER_GENERATION_SAFEGUARDS = {
    "condition_on_prev_tokens": False,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}

# Whisper applies its 1.35 compression threshold to individual decoder
# segments during generation. This post-hoc gate measures the entire rendered
# transcript, where normal repeated syntax compresses more readily, so it uses
# a deliberately more conservative threshold.
EXTERNAL_TEXT_COMPRESSION_RATIO_THRESHOLD = 3.0

KNOWN_WHISPER_BOILERPLATE = (
    "subtitles by the amara org community",
    "subtitles by amara org community",
)


def validate_rocm_bfloat16(model: Any, torch_module: Any) -> tuple[str, str]:
    """Verify the loaded model's actual accelerator and parameter dtype."""

    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is unavailable")
    if not getattr(torch_module.version, "hip", None):
        raise RuntimeError("PyTorch runtime is not ROCm-enabled")
    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        raise RuntimeError("Loaded model is not on a CUDA-compatible accelerator")
    if parameter.dtype != torch_module.bfloat16:
        raise RuntimeError("Loaded model parameters are not bfloat16")
    return f"{parameter.device} (ROCm)", "bfloat16"


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


def transcribe_audio(
    asr_pipeline: Any,
    audio_input: Any,
    *,
    task: str = "translate",
) -> dict[str, Any]:
    if task not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported Whisper task: {task}")
    pipeline_input = str(audio_input) if isinstance(audio_input, Path) else audio_input
    output = asr_pipeline(
        pipeline_input,
        return_timestamps=True,
        return_language=True,
        generate_kwargs={"task": task, **WHISPER_GENERATION_SAFEGUARDS},
    )
    chunks = []
    rendered = []
    languages = []
    no_speech_probabilities = []
    for chunk in output.get("chunks", []):
        start, end = chunk["timestamp"]
        text = chunk["text"].strip()
        normalized_chunk = {"start": start, "end": end, "text": text}
        language = chunk.get("language")
        if isinstance(language, str) and language.strip():
            language = language.strip()
            normalized_chunk["language"] = language
            languages.append(language)
        no_speech_probability = chunk.get(
            "no_speech_probability", chunk.get("no_speech_prob")
        )
        if isinstance(no_speech_probability, int | float):
            normalized_chunk["no_speech_probability"] = float(no_speech_probability)
            no_speech_probabilities.append(float(no_speech_probability))
        chunks.append(normalized_chunk)
        if text:
            rendered.append(f"[{format_timestamp(start)}] {text}")
    output_language = output.get("language")
    if isinstance(output_language, str) and output_language.strip():
        language = output_language.strip()
    else:
        unique_languages = list(dict.fromkeys(languages))
        language = unique_languages[0] if len(unique_languages) == 1 else None
    output_no_speech = output.get("no_speech_probability", output.get("no_speech_prob"))
    if isinstance(output_no_speech, int | float):
        no_speech_probability = float(output_no_speech)
        no_speech_probability_source = "output"
    elif no_speech_probabilities:
        no_speech_probability = float(statistics.median(no_speech_probabilities))
        no_speech_probability_source = "chunk-median"
    else:
        no_speech_probability = None
        no_speech_probability_source = "unavailable"
    return {
        "text": output.get("text", "").strip(),
        "timestamped_text": "\n".join(rendered),
        "chunks": chunks,
        "language": language,
        "no_speech_probability": no_speech_probability,
        "diagnostics": {
            "chunk_count": len(chunks),
            "languages": list(dict.fromkeys(languages)),
            "max_no_speech_probability": (
                max(no_speech_probabilities) if no_speech_probabilities else None
            ),
            "no_speech_probability_source": no_speech_probability_source,
            "no_speech_metadata_available": no_speech_probability is not None,
            "no_speech_probabilities": no_speech_probabilities,
        },
    }


def transcribe_multilingual_audio(
    asr_pipeline: Any, audio_input: Any
) -> dict[str, Any]:
    """Retain the native-language transcript and a separate English translation."""

    original = transcribe_audio(asr_pipeline, audio_input, task="transcribe")
    translation = transcribe_audio(asr_pipeline, audio_input, task="translate")
    return {
        "original": original,
        "translation": translation,
        "source_language": original["language"] or translation["language"],
    }


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def _compression_ratio(text: str) -> float:
    encoded = text.encode("utf-8")
    if not encoded:
        return 0.0
    return len(encoded) / len(zlib.compress(encoded))


def _repeated_window_count(words: list[str]) -> int:
    if len(words) < 20:
        return 0
    window_size = 6
    windows = Counter(
        tuple(words[index : index + window_size])
        for index in range(len(words) - window_size + 1)
    )
    return max(windows.values(), default=0)


def transcript_quality(
    text: str,
    *,
    duration_seconds: float,
    chunks: list[dict[str, Any]] | None = None,
    mode: str = "translate",
    no_speech_probability: float | None = None,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("Transcript duration must be positive")
    if mode not in {"transcribe", "translate"}:
        raise ValueError(f"Unsupported transcript mode: {mode}")
    word_count = len(text.split())
    words_per_second = word_count / duration_seconds
    letters = [character for character in text if character.isalpha()]
    latin_letters = [character for character in letters if character.isascii()]
    latin_letter_ratio = len(latin_letters) / len(letters) if letters else 1.0
    normalized_words = _normalized_words(text)
    normalized_text = " ".join(normalized_words)
    compression_ratio = _compression_ratio(text)
    repeated_window_count = _repeated_window_count(normalized_words)
    max_timestamp_seconds = max(
        (
            float(chunk["end"])
            for chunk in chunks or []
            if isinstance(chunk.get("end"), int | float)
        ),
        default=0.0,
    )
    timestamp_overrun_seconds = max(0.0, max_timestamp_seconds - duration_seconds)
    contains_boilerplate = any(
        phrase in normalized_text for phrase in KNOWN_WHISPER_BOILERPLATE
    )

    if not text.strip():
        status = "rejected-empty"
    elif timestamp_overrun_seconds > 0.5:
        status = "rejected-timestamp-overrun"
    elif words_per_second > 5.0:
        status = "rejected-hallucination"
    elif no_speech_probability is not None and no_speech_probability >= 0.6:
        status = "rejected-no-speech"
    elif contains_boilerplate:
        status = "rejected-boilerplate"
    elif repeated_window_count >= 4:
        status = "rejected-repetition"
    elif (
        word_count >= 20
        and compression_ratio > EXTERNAL_TEXT_COMPRESSION_RATIO_THRESHOLD
    ):
        status = "rejected-compression"
    elif mode == "translate" and latin_letter_ratio < 0.8:
        status = "rejected-non-english-script"
    else:
        status = "accepted"
    return {
        "status": status,
        "word_count": word_count,
        "words_per_second": words_per_second,
        "latin_letter_ratio": latin_letter_ratio,
        "mode": mode,
        "compression_ratio": compression_ratio,
        "compression_ratio_threshold": EXTERNAL_TEXT_COMPRESSION_RATIO_THRESHOLD,
        "decoder_segment_compression_ratio_threshold": (
            WHISPER_GENERATION_SAFEGUARDS["compression_ratio_threshold"]
        ),
        "compression_ratio_definition": (
            "whole-transcript UTF-8 bytes divided by zlib-compressed bytes"
        ),
        "repeated_window_count": repeated_window_count,
        "contains_known_boilerplate": contains_boilerplate,
        "max_timestamp_seconds": max_timestamp_seconds,
        "timestamp_overrun_seconds": timestamp_overrun_seconds,
        "no_speech_probability": no_speech_probability,
        "rule": (
            "reject empty output, material timestamp overruns, more than 5 words "
            "per second, decoder no-speech probability >= 0.6, known ASR "
            "boilerplate, four repeated six-word windows, highly compressible "
            "long output, or non-Latin translation output"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--revision", default="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import soundfile
    import torch
    import transformers
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        pipeline,
    )

    args = parse_args(argv)
    request = json.load(sys.stdin)
    audio_path = Path(request["audio_path"])
    from trec_video_2026.vqa_benchmark import source_fingerprint

    audio_fingerprint = source_fingerprint(audio_path)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = "unverified"
    precision = "unverified"
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_id,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        device, precision = validate_rocm_bfloat16(model, torch)
        processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch.bfloat16,
            device=0,
        )
        audio = load_wav_audio(audio_path, soundfile)
        multilingual_transcript = transcribe_multilingual_audio(asr, audio)
        original = multilingual_transcript["original"]
        transcript = multilingual_transcript["translation"]
        quality = transcript_quality(
            transcript["text"],
            duration_seconds=len(audio) / 16_000,
            chunks=transcript["chunks"],
            mode="translate",
            no_speech_probability=transcript["no_speech_probability"],
        )
        original_quality = transcript_quality(
            original["text"],
            duration_seconds=len(audio) / 16_000,
            chunks=original["chunks"],
            mode="transcribe",
            no_speech_probability=original["no_speech_probability"],
        )
        result = {
            "status": "passed",
            "video_id": request.get("video_id"),
            "source_audio_sha256": audio_fingerprint.sha256,
            **transcript,
            "source_language": multilingual_transcript["source_language"],
            "original_language_transcript": original,
            "english_translation": transcript,
            "quality": quality,
            "original_quality": original_quality,
            "elapsed_seconds": time.monotonic() - started,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "runtime": (
                f"transformers {transformers.__version__}; torch {torch.__version__}"
            ),
            "device": device,
            "precision": precision,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "error_type": None,
            "error_message": None,
        }
    except Exception as error:  # noqa: BLE001 - runner emits structured failure JSON
        message = str(error)
        result = {
            "status": "failed",
            "video_id": request.get("video_id"),
            "source_audio_sha256": audio_fingerprint.sha256,
            "text": None,
            "timestamped_text": None,
            "chunks": [],
            "elapsed_seconds": time.monotonic() - started,
            "model_id": args.model_id,
            "model_revision": args.revision,
            "runtime": (
                f"transformers {transformers.__version__}; torch {torch.__version__}"
            ),
            "device": device,
            "precision": precision,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "error_type": type(error).__name__,
            "error_message": message[:4_000],
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
