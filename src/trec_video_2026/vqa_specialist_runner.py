"""Custom-code runners for the InternVL and MiniCPM compatibility probes.

Heavy model, image, audio, and tensor dependencies are intentionally imported
only by :func:`main`, which keeps the repository's host-side tooling lightweight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from trec_video_2026.vqa_models import normalize_generated_answer


FRAME_COUNT = 12
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_TILES = 2
MINICPM_AUDIO_SAMPLE_RATE = 16_000
MINICPM_MAX_SLICE_NUMS = 1


@dataclass(frozen=True)
class InternVLEvidence:
    frame_paths: tuple[Path, ...]
    timestamps: tuple[float, ...]


def order_evidence(request: dict[str, Any]) -> InternVLEvidence:
    """Validate and temporally order the controlled twelve-frame evidence."""

    paths = tuple(Path(path) for path in request["frame_paths"])
    timestamps = tuple(float(value) for value in request["frame_timestamps"])
    if len(paths) != FRAME_COUNT or len(timestamps) != FRAME_COUNT:
        raise ValueError(f"Specialist runners require exactly {FRAME_COUNT} frames")
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("Frame timestamps must be finite and non-negative")
    ordered = sorted(zip(timestamps, paths, strict=True))
    ordered_timestamps = tuple(item[0] for item in ordered)
    if len(set(ordered_timestamps)) != FRAME_COUNT:
        raise ValueError("Frame timestamps must be distinct")
    return InternVLEvidence(
        frame_paths=tuple(item[1] for item in ordered),
        timestamps=ordered_timestamps,
    )


def build_internvl_prompt(
    evidence: InternVLEvidence, question: str, transcript: str | None
) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    lines = [
        f"Frame{index} at {timestamp:.3f} seconds: <image>"
        for index, timestamp in enumerate(evidence.timestamps, start=1)
    ]
    if transcript is not None:
        if not transcript.strip():
            raise ValueError("Transcript must be non-empty when supplied")
        lines.extend(("Timestamped audio transcript:", transcript.strip()))
    lines.extend(
        (
            "Question: " + question.strip(),
            (
                "Answer with one short phrase or sentence. Return only the answer; "
                "do not explain your reasoning."
            ),
        )
    )
    return "\n".join(lines)


def dynamic_tiles(
    image: Any, *, image_size: int = INTERNVL_IMAGE_SIZE, max_tiles: int = INTERNVL_MAX_TILES
) -> list[Any]:
    """Apply InternVL-style aspect-ratio tiling with a strict total tile cap."""

    if image_size <= 0 or max_tiles <= 0:
        raise ValueError("Image size and tile cap must be positive")
    aspect_ratio = image.width / image.height
    candidates = {
        (columns, rows)
        for tiles in range(1, max_tiles + 1)
        for columns in range(1, tiles + 1)
        for rows in range(1, tiles + 1)
        if columns * rows == tiles
    }
    columns, rows = min(
        candidates,
        key=lambda grid: (
            abs(aspect_ratio - grid[0] / grid[1]),
            grid[0] * grid[1],
        ),
    )
    resized = image.resize((columns * image_size, rows * image_size))
    return [
        resized.crop(
            (
                column * image_size,
                row * image_size,
                (column + 1) * image_size,
                (row + 1) * image_size,
            )
        )
        for row in range(rows)
        for column in range(columns)
    ]


def load_internvl(
    auto_model: Any,
    auto_tokenizer: Any,
    torch_module: Any,
    model_id: str,
    revision: str,
) -> tuple[Any, Any]:
    model = auto_model.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch_module.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        # InternVL's custom model does not implement Transformers' SDPA path.
        attn_implementation="eager",
        use_flash_attn=False,
    ).eval()
    tokenizer = auto_tokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, tokenizer


def load_minicpm(
    auto_model: Any,
    torch_module: Any,
    model_id: str,
    revision: str,
    *,
    snapshot_resolver: Any = None,
) -> tuple[Any, Any]:
    model = auto_model.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch_module.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        init_vision=True,
        init_audio=True,
        init_tts=False,
    ).eval()
    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        snapshot_resolver = snapshot_download
    snapshot_path = snapshot_resolver(
        model_id, revision=revision, local_files_only=True
    )
    model.config._name_or_path = str(snapshot_path)
    model.prepare_processor()
    processor = model.processor
    return model, processor


def split_audio_at_frame_midpoints(
    audio: Any, *, timestamps: Sequence[float], sample_rate: int
) -> list[Any]:
    """Partition mono audio into temporal regions represented by each frame."""

    if sample_rate <= 0 or not timestamps:
        raise ValueError("Audio sample rate and frame timestamps must be positive")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Frame timestamps must be strictly increasing")
    duration = len(audio) / sample_rate
    if duration <= 0:
        raise ValueError("Audio is empty")
    boundaries = [0.0]
    boundaries.extend(
        (left + right) / 2 for left, right in zip(timestamps, timestamps[1:])
    )
    boundaries.append(duration)
    sample_boundaries = [
        min(len(audio), max(0, round(value * sample_rate))) for value in boundaries
    ]
    chunks = [
        audio[start:end]
        for start, end in zip(sample_boundaries, sample_boundaries[1:])
    ]
    if any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("Frame timestamps cannot be aligned to non-empty audio chunks")
    return chunks


def build_minicpm_content(
    *,
    evidence: InternVLEvidence,
    audio: Any,
    sample_rate: int,
    question: str,
    transcript: str | None,
    image_module: Any,
) -> list[Any]:
    """Build MiniCPM's documented native [image, audio, ...] input sequence."""

    if not question.strip():
        raise ValueError("Question cannot be empty")
    ordered = sorted(
        zip(evidence.timestamps, evidence.frame_paths, strict=True), key=lambda item: item[0]
    )
    timestamps = tuple(item[0] for item in ordered)
    chunks = split_audio_at_frame_midpoints(
        audio, timestamps=timestamps, sample_rate=sample_rate
    )
    content: list[Any] = []
    for (_, path), chunk in zip(ordered, chunks, strict=True):
        content.append(image_module.open(path).convert("RGB"))
        content.append(chunk)
    prompt = "Question: " + question.strip()
    if transcript is not None:
        if not transcript.strip():
            raise ValueError("Transcript must be non-empty when supplied")
        prompt += "\nTimestamped audio transcript:\n" + transcript.strip()
    prompt += (
        "\nAnswer with one short phrase or sentence. Return only the answer; "
        "do not explain your reasoning."
    )
    content.append(prompt)
    return content


def chat_internvl(
    model: Any,
    tokenizer: Any,
    pixel_values: Any,
    num_patches_list: list[int],
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    response = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config={"do_sample": False, "max_new_tokens": max_new_tokens},
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    answer = normalize_generated_answer(response)
    if not answer:
        raise ValueError("Model returned an empty answer")
    return answer


def chat_minicpm(
    model: Any, content: list[Any], *, processor: Any, max_new_tokens: int
) -> str:
    response = model.chat(
        msgs=[{"role": "user", "content": content}],
        processor=processor,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_image_id=False,
        max_slice_nums=MINICPM_MAX_SLICE_NUMS,
        use_tts_template=False,
        enable_thinking=False,
        omni_mode=True,
        generate_audio=False,
    )
    answer = normalize_generated_answer(response)
    if not answer:
        raise ValueError("Model returned an empty answer")
    return answer


def _prepare_internvl_pixels(
    evidence: InternVLEvidence, image_module: Any, torch_module: Any, transforms: Any
) -> tuple[Any, list[int]]:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    tensors = []
    tile_counts = []
    for path in evidence.frame_paths:
        image = image_module.open(path).convert("RGB")
        tiles = dynamic_tiles(image)
        tile_counts.append(len(tiles))
        tensors.extend(transform(tile) for tile in tiles)
    pixels = torch_module.stack(tensors).to(
        dtype=torch_module.bfloat16, device="cuda:0"
    )
    return pixels, tile_counts


def _load_audio(request: dict[str, Any], librosa_module: Any) -> tuple[Any, int]:
    source = request.get("audio_path") or request.get("media_path")
    if not source:
        raise ValueError("MiniCPM native audio requires media_path or audio_path")
    audio, sample_rate = librosa_module.load(
        source, sr=MINICPM_AUDIO_SAMPLE_RATE, mono=True
    )
    if sample_rate != MINICPM_AUDIO_SAMPLE_RATE:
        raise ValueError("Audio loader did not return 16 kHz audio")
    return audio, sample_rate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("internvl", "minicpm"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Read one request or an ordered request list and emit structured JSON."""

    # Trusted remote model code may print informational messages. Keep stdout
    # reserved for the runner's single JSON protocol response.
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    args = parse_args(argv)
    payload = json.load(sys.stdin)
    is_batch = isinstance(payload, list)
    requests = payload if is_batch else [payload]
    if not requests:
        raise ValueError("Runner request batch cannot be empty")

    loaded_at = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
    try:
        if args.kind == "internvl":
            from torchvision import transforms

            model, tokenizer = load_internvl(
                AutoModel, AutoTokenizer, torch, args.model_id, args.revision
            )
            librosa = None
        else:
            import librosa

            model, processor = load_minicpm(
                AutoModel, torch, args.model_id, args.revision
            )
            tokenizer = None
            transforms = None
        load_seconds = time.monotonic() - loaded_at
    except Exception as error:
        results = [
            _error_result(
                error,
                loaded_at,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
            )
            for _ in requests
        ]
        print(
            json.dumps(results if is_batch else results[0], ensure_ascii=False),
            file=protocol_stdout,
        )
        return 0

    results = []
    for request in requests:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            evidence = order_evidence(request)
            if args.kind == "internvl":
                pixels, tile_counts = _prepare_internvl_pixels(
                    evidence, Image, torch, transforms
                )
                prompt = build_internvl_prompt(
                    evidence, request["question"], request.get("transcript")
                )
                answer = chat_internvl(
                    model,
                    tokenizer,
                    pixels,
                    tile_counts,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                metadata = {
                    "evidence_mode": "native_dynamic_tiling",
                    "frame_count": FRAME_COUNT,
                    "tile_counts": tile_counts,
                    "max_tiles_per_frame": INTERNVL_MAX_TILES,
                    "tile_size_pixels": INTERNVL_IMAGE_SIZE,
                    "native_audio": False,
                    "transcript_included": request.get("transcript") is not None,
                }
            else:
                audio, sample_rate = _load_audio(request, librosa)
                content = build_minicpm_content(
                    evidence=evidence,
                    audio=audio,
                    sample_rate=sample_rate,
                    question=request["question"],
                    transcript=request.get("transcript"),
                    image_module=Image,
                )
                answer = chat_minicpm(
                    model,
                    content,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                )
                metadata = {
                    "evidence_mode": "selected_frames_native_audio",
                    "frame_count": FRAME_COUNT,
                    "native_audio": True,
                    "audio_sample_rate_hz": MINICPM_AUDIO_SAMPLE_RATE,
                    "audio_chunk_count": FRAME_COUNT,
                    "max_slice_nums": MINICPM_MAX_SLICE_NUMS,
                    "transcript_included": request.get("transcript") is not None,
                }
            result = _result(
                "passed",
                answer,
                started,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
                metadata=metadata,
            )
        except Exception as error:
            result = _error_result(
                error,
                started,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
            )
        result["model_load_seconds"] = load_seconds
        results.append(result)
    print(
        json.dumps(results if is_batch else results[0], ensure_ascii=False),
        file=protocol_stdout,
    )
    return 0


def _error_result(
    error: Exception,
    started: float,
    *,
    args: argparse.Namespace,
    runtime: str,
    peak_memory: int,
) -> dict[str, Any]:
    if args.kind == "internvl":
        metadata = {
            "evidence_mode": "native_dynamic_tiling",
            "frame_count": FRAME_COUNT,
            "max_tiles_per_frame": INTERNVL_MAX_TILES,
            "tile_size_pixels": INTERNVL_IMAGE_SIZE,
            "native_audio": False,
        }
    else:
        metadata = {
            "evidence_mode": "selected_frames_native_audio",
            "frame_count": FRAME_COUNT,
            "native_audio": True,
            "audio_sample_rate_hz": MINICPM_AUDIO_SAMPLE_RATE,
            "audio_chunk_count": FRAME_COUNT,
            "max_slice_nums": MINICPM_MAX_SLICE_NUMS,
        }
    return _result(
        classify_error_status(error),
        None,
        started,
        args=args,
        runtime=runtime,
        peak_memory=peak_memory,
        error=error,
        metadata=metadata,
    )


def classify_error_status(error: Exception) -> str:
    """Distinguish verified runtime incompatibility from ordinary failures."""

    error_name = type(error).__name__
    message = str(error).lower()
    unsupported = error_name in {"NotImplementedError", "UnsupportedError", "HIPError"}
    unsupported = unsupported or "not implemented for" in message or "hip error" in message
    return "unsupported" if unsupported else "failed"


def _result(
    status: str,
    answer: str | None,
    started: float,
    *,
    args: argparse.Namespace,
    runtime: str,
    peak_memory: int,
    metadata: dict[str, Any],
    error: Exception | None = None,
) -> dict[str, Any]:
    message = str(error) if error else None
    if message and len(message) > 4_000:
        message = message[:3_999] + "…"
    return {
        "status": status,
        "answer": answer,
        "elapsed_seconds": time.monotonic() - started,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "runtime": runtime,
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "precision": "bfloat16",
        "device": "cuda:0 (ROCm)",
        "peak_memory_bytes": peak_memory,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
        **metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
