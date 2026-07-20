"""Direct Transformers runner used inside the pinned ROCm container."""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_models import (
    build_controlled_messages,
    normalize_generated_answer,
)

MIN_PIXELS_PER_FRAME = 256 * 256
MAX_PIXELS_PER_FRAME = 512 * 512


def load_processor(auto_processor: Any, model_id: str, revision: str) -> Any:
    """Load a processor with a bounded and recorded per-frame pixel budget."""

    return auto_processor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS_PER_FRAME,
        max_pixels=MAX_PIXELS_PER_FRAME,
    )


def select_auto_model_class(
    kind: str, image_text_class: Any, multimodal_class: Any
) -> Any:
    if kind == "image-text":
        return image_text_class
    if kind == "multimodal":
        return multimodal_class
    raise ValueError(f"Unsupported model class: {kind}")


def generate_answer(
    *,
    processor: Any,
    model: Any,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    transcript: str | None,
    max_new_tokens: int,
) -> str:
    messages = build_controlled_messages(
        question=question,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        transcript=transcript,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_length = inputs["input_ids"].shape[-1]
    inputs = inputs.to(model.device)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    answer_ids = generated[0][input_length:]
    text = processor.decode(answer_ids, skip_special_tokens=True)
    answer = normalize_generated_answer(text)
    if not answer:
        raise ValueError("Model returned an empty answer")
    return answer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--model-class", choices=("image-text", "multimodal"), default="image-text"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read one runner request from stdin and emit one runner result as JSON."""

    import torch
    import transformers
    from transformers import (
        AutoModelForImageTextToText,
        AutoModelForMultimodalLM,
        AutoProcessor,
    )

    args = parse_args(argv)
    payload = json.load(sys.stdin)
    is_batch = isinstance(payload, list)
    requests = payload if is_batch else [payload]
    if not requests:
        raise ValueError("Runner request batch cannot be empty")
    model_id = args.model_id
    revision = args.revision
    load_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        processor = load_processor(AutoProcessor, model_id, revision)
        auto_model = select_auto_model_class(
            args.model_class, AutoModelForImageTextToText, AutoModelForMultimodalLM
        )
        model = auto_model.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        load_seconds = time.monotonic() - load_started
        results = []
        for request in requests:
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                answer = generate_answer(
                    processor=processor,
                    model=model,
                    question=request["question"],
                    frame_paths=tuple(Path(path) for path in request["frame_paths"]),
                    frame_timestamps=tuple(request["frame_timestamps"]),
                    transcript=request.get("transcript"),
                    max_new_tokens=args.max_new_tokens,
                )
                result = _result(
                    status="passed",
                    answer=answer,
                    started=started,
                    model_id=model_id,
                    revision=revision,
                    runtime=f"transformers {transformers.__version__}; torch {torch.__version__}",
                    peak_memory=torch.cuda.max_memory_allocated(),
                )
            except Exception as error:
                result = _error_result(
                    error=error,
                    started=started,
                    model_id=model_id,
                    revision=revision,
                    runtime=f"transformers {transformers.__version__}; torch {torch.__version__}",
                    peak_memory=torch.cuda.max_memory_allocated(),
                )
            result["model_load_seconds"] = load_seconds
            results.append(result)
    except Exception as error:  # Runner must return structured model/runtime errors.
        results = [
            _error_result(
                error=error,
                started=load_started,
                model_id=model_id,
                revision=revision,
                runtime=f"transformers {transformers.__version__}; torch {torch.__version__}",
                peak_memory=torch.cuda.max_memory_allocated(),
            )
            for _ in requests
        ]
    print(json.dumps(results if is_batch else results[0], ensure_ascii=False))
    return 0


def _error_result(
    *,
    error: Exception,
    started: float,
    model_id: str,
    revision: str,
    runtime: str,
    peak_memory: int,
) -> dict[str, Any]:
    error_name = type(error).__name__
    unsupported = error_name in {
        "NotImplementedError",
        "UnsupportedError",
        "HIPError",
    } or "not implemented for" in str(error).lower()
    return _result(
        status="unsupported" if unsupported else "failed",
        answer=None,
        started=started,
        model_id=model_id,
        revision=revision,
        runtime=runtime,
        peak_memory=peak_memory,
        error=error,
    )


def _result(
    *,
    status: str,
    answer: str | None,
    started: float,
    model_id: str,
    revision: str,
    runtime: str,
    peak_memory: int,
    error: Exception | None = None,
) -> dict[str, Any]:
    message = str(error) if error else None
    if message and len(message) > 4_000:
        message = message[:3_999] + "…"
    return {
        "status": status,
        "answer": answer,
        "elapsed_seconds": time.monotonic() - started,
        "model_id": model_id,
        "model_revision": revision,
        "runtime": runtime,
        "precision": "bfloat16",
        "device": "cuda:0 (ROCm)",
        "peak_memory_bytes": peak_memory,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
        "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
        "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
    }


if __name__ == "__main__":
    raise SystemExit(main())
