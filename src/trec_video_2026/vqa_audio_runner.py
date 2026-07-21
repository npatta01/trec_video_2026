"""Isolated ROCm BF16 runners for audio transcription and diagnostics.

Heavy model and audio dependencies are imported only by :func:`main`, keeping
the host-side request/result contract independently testable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_whisper import format_timestamp


def request_identity(request: Any) -> dict[str, str | None]:
    """Copy only label-free protocol identity fields from a request."""

    payload = request if isinstance(request, dict) else {}
    return {
        field: value if isinstance(value := payload.get(field), str) and value.strip() else None
        for field in ("run_id", "condition_id", "video_id")
    }


def validate_request(request: dict[str, Any], *, model_kind: str) -> None:
    for field in ("run_id", "condition_id", "video_id"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
    audio_path = request.get("audio_path")
    if (
        not isinstance(audio_path, str)
        or not audio_path.strip()
        or not Path(audio_path).is_absolute()
    ):
        raise ValueError("audio_path must be a nonempty absolute path")
    mode = request.get("mode")
    if model_kind == "qwen3_asr":
        if mode != "transcription":
            raise ValueError("Qwen3-ASR supports transcription only")
    elif model_kind == "voxtral":
        if mode not in {"transcription", "translation", "question_answer"}:
            raise ValueError("Unknown Voxtral audio mode")
    else:
        raise ValueError("Unknown audio model kind")
    if mode == "question_answer":
        question = request.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")


def build_voxtral_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    content = [{"type": "audio", "path": request["audio_path"]}]
    if request.get("mode") == "question_answer":
        content.append({"type": "text", "text": request["question"].strip()})
    elif request.get("mode") == "translation":
        content.append(
            {
                "type": "text",
                "text": (
                    "Translate the complete audio into English. "
                    "Return only the translation."
                ),
            }
        )
    return [{"role": "user", "content": content}]


def prepare_voxtral_inputs(
    processor: Any,
    request: dict[str, Any],
    *,
    model_id: str,
) -> Any:
    """Use Voxtral's dedicated ASR surface for pure transcription."""

    if request["mode"] == "transcription":
        return processor.apply_transcription_request(
            language=None,
            audio=request["audio_path"],
            model_id=model_id,
        )
    return processor.apply_chat_template(
        build_voxtral_messages(request),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def decode_voxtral(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    device: str,
    dtype: Any,
    max_new_tokens: int,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device=device, dtype=dtype)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    prompt_length = len(inputs["input_ids"][0])
    new_tokens = [row[prompt_length:] for row in generated]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    if not text:
        raise ValueError("Voxtral returned empty text")
    return text


def normalize_qwen_asr_result(output: Any) -> dict[str, Any]:
    text = getattr(output, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Qwen3-ASR returned empty text")
    language = getattr(output, "language", None)
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Qwen3-ASR omitted language metadata")
    chunks = []
    rendered = []
    for segment in getattr(output, "time_stamps", None) or []:
        chunk_text = str(segment.text).strip()
        start = float(segment.start_time)
        end = float(segment.end_time)
        chunks.append({"start": start, "end": end, "text": chunk_text})
        rendered.append(f"[{format_timestamp(start)}] {chunk_text}")
    return {
        "text": text.strip(),
        "language": language.strip(),
        "timestamped_text": "\n".join(rendered),
        "chunks": chunks,
    }


def transcribe_qwen(model: Any, audio_path: str) -> dict[str, Any]:
    outputs = model.transcribe(audio=audio_path, language=None)
    if len(outputs) != 1:
        raise ValueError("Qwen3-ASR returned an unexpected result count")
    return normalize_qwen_asr_result(outputs[0])


def validate_rocm_bfloat16(model: Any, torch_module: Any) -> tuple[str, str]:
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


def classify_error_status(error: Exception) -> str:
    message = str(error).lower()
    unsupported = type(error).__name__ in {
        "NotImplementedError",
        "UnsupportedError",
        "HIPError",
    }
    unsupported = unsupported or "not implemented for" in message or "hip error" in message
    return "unsupported" if unsupported else "failed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("voxtral", "qwen3_asr"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


def _model_with_parameters(model: Any) -> Any:
    candidate = model
    for _ in range(3):
        if callable(getattr(candidate, "parameters", None)):
            return candidate
        candidate = getattr(candidate, "model", None)
        if candidate is None:
            break
    raise RuntimeError("Loaded model does not expose parameters for BF16 validation")


def _voxtral_from_inputs(
    model: Any,
    processor: Any,
    inputs: Any,
    *,
    max_new_tokens: int,
) -> str:
    generated = model.generate(
        **inputs, do_sample=False, max_new_tokens=max_new_tokens
    )
    prompt_length = inputs.input_ids.shape[1]
    decoded = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    if not decoded:
        raise ValueError("Voxtral returned empty text")
    return decoded


def _base_result(
    request: Any,
    *,
    args: argparse.Namespace,
    status: str,
    started: float,
    runtime: str,
    device: str,
    precision: str,
    peak_memory: int | None,
    source_audio_sha256: str | None,
    error: Exception | None = None,
) -> dict[str, Any]:
    message = str(error)[:4_000] if error else None
    return {
        **request_identity(request),
        "status": status,
        "mode": request.get("mode") if isinstance(request, dict) else None,
        "source_audio_sha256": source_audio_sha256,
        "text": None,
        "translated_text": None,
        "answer": None,
        "language": None,
        "timestamped_text": None,
        "chunks": [],
        "quality": None,
        "elapsed_seconds": time.monotonic() - started,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "runtime": runtime,
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "device": device,
        "precision": precision,
        "peak_memory_bytes": peak_memory,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
    }


def main(argv: list[str] | None = None) -> int:
    """Read one request or ordered request batch and emit structured JSON."""

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    import soundfile
    import torch
    import transformers

    from trec_video_2026.vqa_benchmark import source_fingerprint
    from trec_video_2026.vqa_whisper import transcript_quality

    args = parse_args(argv)
    payload = json.load(sys.stdin)
    is_batch = isinstance(payload, list)
    requests = payload if is_batch else [payload]
    if not requests:
        raise ValueError("Audio runner request batch cannot be empty")

    loaded_at = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    runtime = f"transformers {transformers.__version__}; torch {torch.__version__}"
    device = "unverified"
    precision = "unverified"
    try:
        if args.kind == "voxtral":
            from transformers import AutoProcessor, VoxtralForConditionalGeneration

            processor = AutoProcessor.from_pretrained(
                args.model_id, revision=args.revision
            )
            model = VoxtralForConditionalGeneration.from_pretrained(
                args.model_id,
                revision=args.revision,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
            ).eval()
        else:
            from qwen_asr import Qwen3ASRModel

            model = Qwen3ASRModel.from_pretrained(
                args.model_id,
                revision=args.revision,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                max_inference_batch_size=1,
                max_new_tokens=args.max_new_tokens,
            )
            processor = None
        device, precision = validate_rocm_bfloat16(
            _model_with_parameters(model), torch
        )
        load_seconds = time.monotonic() - loaded_at
    except Exception as error:
        results = [
            {
                **_base_result(
                    request,
                    args=args,
                    status=classify_error_status(error),
                    started=loaded_at,
                    runtime=runtime,
                    device=device,
                    precision=precision,
                    peak_memory=torch.cuda.max_memory_allocated(),
                    source_audio_sha256=None,
                    error=error,
                ),
                "model_load_seconds": None,
            }
            for request in requests
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
        audio_hash = None
        try:
            validate_request(request, model_kind=args.kind)
            audio_path = Path(request["audio_path"])
            audio_hash = source_fingerprint(audio_path).sha256
            duration = soundfile.info(str(audio_path)).duration
            result = _base_result(
                request,
                args=args,
                status="passed",
                started=started,
                runtime=runtime,
                device=device,
                precision=precision,
                peak_memory=None,
                source_audio_sha256=audio_hash,
            )
            if args.kind == "qwen3_asr":
                output = transcribe_qwen(model, str(audio_path))
                result.update(output)
                result["quality"] = transcript_quality(
                    output["text"], duration_seconds=duration
                )
            else:
                inputs = prepare_voxtral_inputs(
                    processor, request, model_id=args.model_id
                ).to(device="cuda:0", dtype=torch.bfloat16)
                text = _voxtral_from_inputs(
                    model,
                    processor,
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                )
                if request["mode"] == "question_answer":
                    result["answer"] = text
                else:
                    result["text"] = text
                    result["translated_text"] = (
                        text if request["mode"] == "translation" else None
                    )
                    result["quality"] = transcript_quality(
                        text, duration_seconds=duration
                    )
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        except Exception as error:
            result = _base_result(
                request,
                args=args,
                status=classify_error_status(error),
                started=started,
                runtime=runtime,
                device=device,
                precision=precision,
                peak_memory=torch.cuda.max_memory_allocated(),
                source_audio_sha256=audio_hash,
                error=error,
            )
        result["model_load_seconds"] = load_seconds
        results.append(result)

    print(
        json.dumps(results if is_batch else results[0], ensure_ascii=False),
        file=protocol_stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
