"""Qwen2.5-Omni matched native-video audio ablation runner.

Heavy model, media, and tensor dependencies are imported only by :func:`main`
so the repository's host-side test and orchestration environment stays small.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Mapping

from trec_video_2026.vqa_models import normalize_generated_answer


ALLOWED_REQUEST_FIELDS = frozenset(
    {
        "run_id",
        "condition_id",
        "video_id",
        "question",
        "frame_paths",
        "frame_timestamps",
        "frame_descriptions",
        "transcript",
        "media_path",
        "audio_path",
        "audio_mode",
    }
)


def validate_request(request: Mapping[str, Any]) -> None:
    """Fail closed on labels, alternate evidence, or malformed identities."""

    unknown = sorted(set(request) - ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"Runner request has unknown fields: {', '.join(unknown)}")
    missing = sorted(
        {"run_id", "condition_id", "video_id", "question", "media_path", "audio_mode"}
        - set(request)
    )
    if missing:
        raise ValueError(f"Runner request is missing: {', '.join(missing)}")
    for field in ("run_id", "condition_id", "video_id", "question", "media_path"):
        value = request[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Runner request {field} must be a non-empty string")
    if request.get("transcript") is not None:
        raise ValueError("Qwen Omni native-video requests cannot include a transcript")
    if request.get("audio_path") is not None:
        raise ValueError("Qwen Omni audio must come from the matched video container")
    omni_audio_enabled(request)


def omni_audio_enabled(request: Mapping[str, Any]) -> bool:
    mode = request.get("audio_mode")
    if mode not in {"audio-off", "audio-on"}:
        raise ValueError("audio_mode must be audio-off or audio-on")
    return mode == "audio-on"


def build_conversation(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build identical video evidence and question text for both audio modes."""

    validate_request(request)
    question = str(request["question"]).strip()
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You answer questions about video evidence. Respond with one "
                        "short phrase or sentence and no explanation."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(request["media_path"])},
                {
                    "type": "text",
                    "text": (
                        f"Question: {question}\nAnswer with one short phrase or "
                        "sentence. Return only the answer; do not explain your reasoning."
                    ),
                },
            ],
        },
    ]


def load_omni(
    auto_model: Any,
    auto_processor: Any,
    torch_module: Any,
    model_id: str,
    revision: str,
    *,
    snapshot_resolver: Any = None,
) -> tuple[Any, Any]:
    """Load only the exact cached revision, on ROCm BF16 with SDPA."""

    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        snapshot_resolver = snapshot_download
    snapshot_path = snapshot_resolver(
        model_id, revision=revision, local_files_only=True
    )
    model = auto_model.from_pretrained(
        str(snapshot_path),
        torch_dtype=torch_module.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    model.disable_talker()
    processor = auto_processor.from_pretrained(
        str(snapshot_path), local_files_only=True
    )
    return model, processor


def prepare_inputs(
    processor: Any,
    process_mm_info: Any,
    conversation: list[dict[str, Any]],
    *,
    use_audio_in_video: bool,
    model: Any,
) -> Any:
    """Apply one explicit audio switch at both preprocessing boundaries."""

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    return inputs.to(model.device).to(model.dtype)


def generate_text(
    model: Any,
    processor: Any,
    inputs: Mapping[str, Any],
    *,
    use_audio_in_video: bool,
    max_new_tokens: int,
) -> str:
    """Generate deterministic text only, with the same explicit audio switch."""

    text_ids = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        return_audio=False,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    prompt_ids = inputs.get("input_ids")
    if prompt_ids is None or len(prompt_ids) != len(text_ids):
        raise ValueError("Qwen Omni generation omitted matched prompt token IDs")
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(prompt_ids, text_ids, strict=True)
    ]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if len(decoded) != 1:
        raise ValueError("Qwen Omni returned an unexpected result count")
    answer = normalize_generated_answer(decoded[0])
    if not answer:
        raise ValueError("Qwen Omni returned an empty answer")
    return answer


def validate_rocm_bfloat16(model: Any, torch_module: Any) -> tuple[str, str]:
    """Fail closed instead of silently accepting CPU or non-ROCm execution."""

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Read one request or an ordered request list and emit structured JSON."""

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    import torch
    import transformers
    from qwen_omni_utils import process_mm_info
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    args = parse_args(argv)
    payload = json.load(sys.stdin)
    is_batch = isinstance(payload, list)
    requests = payload if is_batch else [payload]
    if not requests:
        raise ValueError("Runner request batch cannot be empty")

    runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
    loaded_at = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = "unverified"
    precision = "unverified"
    talker_disabled = False
    try:
        model, processor = load_omni(
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            torch,
            args.model_id,
            args.revision,
        )
        talker_disabled = True
        device, precision = validate_rocm_bfloat16(model, torch)
        load_seconds = time.monotonic() - loaded_at
    except Exception as error:
        results = [
            _error_result(
                error,
                loaded_at,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
                device=device,
                precision=precision,
                audio_enabled=None,
                talker_disabled=talker_disabled,
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
        audio_enabled = None
        try:
            validate_request(request)
            audio_enabled = omni_audio_enabled(request)
            conversation = build_conversation(request)
            inputs = prepare_inputs(
                processor,
                process_mm_info,
                conversation,
                use_audio_in_video=audio_enabled,
                model=model,
            )
            answer = generate_text(
                model,
                processor,
                inputs,
                use_audio_in_video=audio_enabled,
                max_new_tokens=args.max_new_tokens,
            )
            result = _result(
                "passed",
                answer,
                started,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
                device=device,
                precision=precision,
                audio_enabled=audio_enabled,
                talker_disabled=talker_disabled,
            )
        except Exception as error:
            result = _error_result(
                error,
                started,
                args=args,
                runtime=runtime,
                peak_memory=torch.cuda.max_memory_allocated(),
                device=device,
                precision=precision,
                audio_enabled=audio_enabled,
                talker_disabled=talker_disabled,
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
    device: str,
    precision: str,
    audio_enabled: bool | None,
    talker_disabled: bool,
) -> dict[str, Any]:
    message = str(error) or repr(error)
    if len(message) > 4_000:
        message = message[:3_999] + "…"
    unsupported = type(error).__name__ in {
        "NotImplementedError",
        "UnsupportedError",
        "HIPError",
    } or "hip error" in message.lower()
    return _result(
        "unsupported" if unsupported else "failed",
        None,
        started,
        args=args,
        runtime=runtime,
        peak_memory=peak_memory,
        device=device,
        precision=precision,
        audio_enabled=audio_enabled,
        talker_disabled=talker_disabled,
        error_type=type(error).__name__,
        error_message=message,
    )


def _result(
    status: str,
    answer: str | None,
    started: float,
    *,
    args: argparse.Namespace,
    runtime: str,
    peak_memory: int,
    device: str,
    precision: str,
    audio_enabled: bool | None,
    talker_disabled: bool,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "answer": answer,
        "elapsed_seconds": max(0.0, time.monotonic() - started),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "runtime": runtime,
        "precision": precision,
        "device": device,
        "peak_memory_bytes": peak_memory,
        "error_type": error_type,
        "error_message": error_message,
        "evidence_mode": (
            "native_video_audio_on"
            if audio_enabled
            else "native_video_audio_off"
            if audio_enabled is False
            else "native_video_audio_mode_unverified"
        ),
        "native_video": True,
        "native_audio": audio_enabled,
        "talker_disabled": talker_disabled,
        "return_audio": False,
        "transcript_included": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
