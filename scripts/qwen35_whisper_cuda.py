#!/usr/bin/env python3
"""Run label-free Whisper large-v3 translation evidence on CUDA."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_whisper import (
    WHISPER_GENERATION_SAFEGUARDS,
    format_timestamp,
    transcript_quality,
)

YOUTUBE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def safe_audio_path(audio_root: Path | None, raw_path: object, video_id: object) -> Path:
    if not isinstance(video_id, str) or YOUTUBE_ID_PATTERN.fullmatch(video_id) is None:
        raise RuntimeError("evidence video_id is not a valid YouTube ID")
    if not isinstance(raw_path, str):
        raise RuntimeError(f"audio path is missing for {video_id}")
    path = Path(raw_path).resolve(strict=True)
    if audio_root is not None and not path.is_relative_to(audio_root.resolve(strict=True)):
        raise RuntimeError(f"audio path escapes the evidence root for {video_id}")
    return path


def normalize_pipeline_output(output: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    rendered = []
    languages = []
    no_speech_probabilities = []
    for chunk in output.get("chunks", []):
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            continue
        start, end = timestamp
        text = str(chunk.get("text", "")).strip()
        normalized = {
            "start": float(start) if isinstance(start, (int, float)) else None,
            "end": float(end) if isinstance(end, (int, float)) else None,
            "text": text,
        }
        language = chunk.get("language")
        if isinstance(language, str) and language.strip():
            normalized["language"] = language.strip()
            languages.append(language.strip())
        no_speech = chunk.get("no_speech_probability", chunk.get("no_speech_prob"))
        if isinstance(no_speech, (int, float)):
            normalized["no_speech_probability"] = float(no_speech)
            no_speech_probabilities.append(float(no_speech))
        chunks.append(normalized)
        if text and isinstance(start, (int, float)):
            rendered.append(f"[{format_timestamp(float(start))}] {text}")
    output_language = output.get("language")
    language = (
        output_language.strip()
        if isinstance(output_language, str) and output_language.strip()
        else (list(dict.fromkeys(languages))[0] if len(set(languages)) == 1 else None)
    )
    output_no_speech = output.get(
        "no_speech_probability", output.get("no_speech_prob")
    )
    if isinstance(output_no_speech, (int, float)):
        no_speech_probability = float(output_no_speech)
        no_speech_source = "output"
    elif no_speech_probabilities:
        no_speech_probability = max(no_speech_probabilities)
        no_speech_source = "chunk-max"
    else:
        no_speech_probability = None
        no_speech_source = "unavailable"
    return {
        "text": str(output.get("text", "")).strip(),
        "timestamped_text": "\n".join(rendered),
        "chunks": chunks,
        "language": language,
        "no_speech_probability": no_speech_probability,
        "diagnostics": {
            "chunk_count": len(chunks),
            "languages": list(dict.fromkeys(languages)),
            "no_speech_probability_source": no_speech_source,
            "no_speech_metadata_available": no_speech_probability is not None,
            "no_speech_probabilities": no_speech_probabilities,
        },
    }


def transcribe(
    asr: Any, audio: Any, *, task: str, duration_seconds: float
) -> dict[str, Any]:
    output = asr(
        audio,
        return_timestamps=True,
        return_language=True,
        generate_kwargs={"task": task, **WHISPER_GENERATION_SAFEGUARDS},
    )
    normalized = normalize_pipeline_output(output)
    normalized["quality"] = transcript_quality(
        normalized["text"],
        duration_seconds=duration_seconds,
        chunks=normalized["chunks"],
        mode=task,
        no_speech_probability=normalized["no_speech_probability"],
    )
    return normalized


def load_asr(model_id: str, revision: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip:
        raise RuntimeError("CUDA is unavailable or the runtime is not NVIDIA CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bfloat16 is unavailable")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.bfloat16,
        device=0,
    )
    return asr, {
        "model_load_seconds": time.monotonic() - started,
        "model_id": model_id,
        "model_revision": revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_bf16_supported": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/whisper-large-v3")
    parser.add_argument(
        "--revision", default="06f233fe06e710322aca913c1bc4249a0d71fce1"
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--audio-root", type=Path)
    return parser.parse_args()


def main() -> int:
    import soundfile

    args = parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    examples = evidence.get("examples")
    if not isinstance(examples, list) or not examples:
        raise SystemExit("evidence examples must be a non-empty list")
    count = args.count or len(examples)
    examples = examples[:count]
    asr, runtime = load_asr(args.model_id, args.revision)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        audio = example.get("audio", {})
        video_id = example["video_id"]
        if audio.get("status") == "no-audio-stream":
            results.append(
                {
                    "status": "no-audio-stream",
                    "video_id": video_id,
                    "source_audio_sha256": None,
                    "transcript_included": False,
                    "text": None,
                    "timestamped_text": None,
                    "original_language_transcript": None,
                    "english_translation": None,
                }
            )
            continue
        audio_path = safe_audio_path(args.audio_root, audio.get("path"), video_id)
        audio_array, sample_rate = soundfile.read(str(audio_path), dtype="float32")
        if sample_rate != 16_000:
            raise RuntimeError(f"{video_id}: expected 16000 Hz audio, got {sample_rate}")
        if getattr(audio_array, "ndim", 1) != 1:
            raise RuntimeError(f"{video_id}: expected mono audio")
        duration = float(len(audio_array) / sample_rate)
        row_started = time.monotonic()
        translation = transcribe(asr, audio_array, task="translate", duration_seconds=duration)
        accepted = translation["quality"]["status"] == "accepted"
        selected_text = translation["timestamped_text"] or translation["text"]
        results.append(
            {
                "status": "passed",
                "video_id": video_id,
                "source_audio_sha256": audio.get("sha256"),
                "source_audio_duration_seconds": duration,
                "transcript_included": accepted,
                "text": translation["text"],
                "timestamped_text": translation["timestamped_text"],
                "selected_transcript": selected_text if accepted else None,
                "quality": translation["quality"],
                "source_language": translation.get("language"),
                "original_language_transcript": None,
                "english_translation": translation,
                "elapsed_seconds": time.monotonic() - row_started,
            }
        )
        if index % 10 == 0:
            print(json.dumps({"processed": index, "total": len(examples)}), flush=True)
    payload = {
        "schema_version": 1,
        "label_policy": "label-free-transcript-evidence",
        "model": runtime,
        "transcription_mode": "translate-only",
        "example_count": len(results),
        "accepted_count": sum(row.get("transcript_included") is True for row in results),
        "no_audio_count": sum(row.get("status") == "no-audio-stream" for row in results),
        "wall_seconds": time.monotonic() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({k: payload[k] for k in ("schema_version", "label_policy", "model", "example_count", "accepted_count", "no_audio_count", "wall_seconds")}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"example_count": len(results), "accepted_count": payload["accepted_count"], "wall_seconds": payload["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
