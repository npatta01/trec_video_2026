"""Direct Transformers runner used inside the pinned ROCm container."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_models import (
    build_controlled_messages,
    normalize_generated_answer,
)

MIN_PIXELS_PER_FRAME = 256 * 256
MAX_PIXELS_PER_FRAME = 512 * 512


def attach_request_identity(
    result: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Return a result with the request identity embedded for auditability."""

    return {
        **result,
        **{field: request[field] for field in ("run_id", "condition_id", "video_id")},
    }


def load_processor(auto_processor: Any, model_id: str, revision: str) -> Any:
    """Load a processor with a bounded and recorded per-frame pixel budget."""

    kwargs = {
        "min_pixels": MIN_PIXELS_PER_FRAME,
        "max_pixels": MAX_PIXELS_PER_FRAME,
    }
    if model_id.startswith("google/gemma-4-"):
        kwargs = {}
    if model_id.startswith("allenai/Molmo2-"):
        from transformers.processing_utils import ProcessorMixin

        return load_legacy_optional_processor(
            auto_processor,
            ProcessorMixin,
            model_id,
            revision,
            **kwargs,
        )
    return auto_processor.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, **kwargs
    )


def load_legacy_optional_processor(
    auto_processor: Any,
    processor_mixin: Any,
    model_id: str,
    revision: str,
    **processor_kwargs: Any,
) -> Any:
    """Load pinned Hub processors that still use ``optional_attributes``.

    Transformers 5 removed that legacy constructor path.  Molmo2's pinned
    remote processor still relies on it, so scope the compatibility shim to
    processor construction and always restore the shared base class.
    """

    original_init = processor_mixin.__init__

    def compatible_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        optional = set(getattr(instance, "optional_attributes", ()))
        legacy = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in optional and key != "chat_template"
        }
        original_init(instance, *args, **kwargs)
        for key, value in legacy.items():
            setattr(instance, key, value)

    processor_mixin.__init__ = compatible_init
    try:
        return auto_processor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            **processor_kwargs,
        )
    finally:
        processor_mixin.__init__ = original_init


def register_legacy_default_rope(rope_functions: dict[str, Any]) -> None:
    """Restore the removed legacy default RoPE entry for pinned Molmo2 code."""

    if "default" in rope_functions:
        return

    def compute_default_rope(
        config: Any,
        device: Any = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[Any, float]:
        del seq_len, layer_type
        import torch

        base = getattr(config, "rope_theta", 10_000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        positions = torch.arange(0, dim, 2, dtype=torch.int64).to(
            device=device, dtype=torch.float
        )
        return 1.0 / (base ** (positions / dim)), 1.0

    rope_functions["default"] = compute_default_rope


def generation_compat_kwargs(
    model: Any,
    input_length: int,
    *,
    arange: Any = None,
) -> dict[str, Any]:
    """Return narrowly scoped generation arguments for pinned remote code."""

    if getattr(getattr(model, "config", None), "model_type", None) != "molmo2":
        return {}
    if arange is None:
        import torch

        arange = torch.arange
    return {"cache_position": arange(input_length, device=model.device)}


def select_auto_model_class(
    kind: str, image_text_class: Any, multimodal_class: Any
) -> Any:
    if kind == "image-text":
        return image_text_class
    if kind == "multimodal":
        return multimodal_class
    raise ValueError(f"Unsupported model class: {kind}")


def _tensor_counts(tensors: list[Any], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in tensors:
        value = str(getattr(tensor, attribute))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def validate_rocm_bfloat16(
    model: Any, torch_module: Any
) -> tuple[str, str, dict[str, Any]]:
    """Attest BF16 parameters and accelerator-resident model buffers."""

    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is unavailable")
    torch_hip = getattr(torch_module.version, "hip", None)
    if not torch_hip:
        raise RuntimeError("PyTorch runtime is not ROCm-enabled")
    parameters = list(model.parameters())
    buffers = list(getattr(model, "buffers", lambda: ())())
    if not parameters:
        raise RuntimeError("Loaded model has no parameters to attest")
    if any(str(parameter.device) != "cuda:0" for parameter in parameters):
        raise RuntimeError("Loaded model parameters are not all on cuda:0")
    if any(parameter.dtype != torch_module.bfloat16 for parameter in parameters):
        raise RuntimeError("Loaded model parameters are not bfloat16")
    if any(str(buffer.device) != "cuda:0" for buffer in buffers):
        raise RuntimeError("Loaded model buffers are not all on cuda:0")
    attestation = {
        "parameter_tensor_count": len(parameters),
        "parameter_devices": _tensor_counts(parameters, "device"),
        "parameter_dtypes": _tensor_counts(parameters, "dtype"),
        "buffer_tensor_count": len(buffers),
        "buffer_devices": _tensor_counts(buffers, "device"),
        "buffer_dtypes": _tensor_counts(buffers, "dtype"),
        "torch_hip": str(torch_hip),
    }
    return "cuda:0 (ROCm)", "bfloat16", attestation


def validate_bfloat16_accelerator(
    model: Any, torch_module: Any
) -> tuple[str, str, dict[str, Any]]:
    """Attest BF16 placement on either a ROCm or NVIDIA CUDA accelerator."""

    torch_hip = getattr(torch_module.version, "hip", None)
    if torch_hip:
        return validate_rocm_bfloat16(model, torch_module)
    torch_cuda = getattr(torch_module.version, "cuda", None)
    if not torch_cuda:
        raise RuntimeError("PyTorch runtime is neither ROCm nor NVIDIA CUDA-enabled")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is unavailable")
    parameters = list(model.parameters())
    buffers = list(getattr(model, "buffers", lambda: ())())
    if not parameters:
        raise RuntimeError("Loaded model has no parameters to attest")
    if any(str(parameter.device) != "cuda:0" for parameter in parameters):
        raise RuntimeError("Loaded model parameters are not all on cuda:0")
    if any(parameter.dtype != torch_module.bfloat16 for parameter in parameters):
        raise RuntimeError("Loaded model parameters are not bfloat16")
    if any(str(buffer.device) != "cuda:0" for buffer in buffers):
        raise RuntimeError("Loaded model buffers are not all on cuda:0")
    attestation = {
        "parameter_tensor_count": len(parameters),
        "parameter_devices": _tensor_counts(parameters, "device"),
        "parameter_dtypes": _tensor_counts(parameters, "dtype"),
        "buffer_tensor_count": len(buffers),
        "buffer_devices": _tensor_counts(buffers, "device"),
        "buffer_dtypes": _tensor_counts(buffers, "dtype"),
        "accelerator_backend": "nvidia-cuda",
        "torch_cuda": str(torch_cuda),
        "torch_hip": None,
    }
    return "cuda:0 (NVIDIA CUDA)", "bfloat16", attestation


def generate_answer(
    *,
    processor: Any,
    model: Any,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    frame_descriptions: tuple[str, ...] | None = None,
    transcript: str | None,
    max_new_tokens: int,
) -> str:
    messages = build_controlled_messages(
        question=question,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        frame_descriptions=frame_descriptions,
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
        **generation_compat_kwargs(model, input_length),
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
    from transformers import AutoProcessor

    args = parse_args(argv)
    payload = json.load(sys.stdin)
    is_batch = isinstance(payload, list)
    requests = payload if is_batch else [payload]
    if not requests:
        raise ValueError("Runner request batch cannot be empty")
    model_id = args.model_id
    revision = args.revision
    if model_id.startswith("allenai/Molmo2-"):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        register_legacy_default_rope(ROPE_INIT_FUNCTIONS)
    load_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    device = "unverified"
    precision = "unverified"
    runtime_attestation = None
    try:
        processor = load_processor(AutoProcessor, model_id, revision)
        if args.model_class == "image-text":
            from transformers import AutoModelForImageTextToText

            auto_model = AutoModelForImageTextToText
        else:
            from transformers import AutoModelForMultimodalLM

            auto_model = AutoModelForMultimodalLM
        model = auto_model.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        device, precision, runtime_attestation = validate_bfloat16_accelerator(
            model, torch
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
                    frame_descriptions=(
                        tuple(request["frame_descriptions"])
                        if request.get("frame_descriptions") is not None
                        else None
                    ),
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
                    device=device,
                    precision=precision,
                    runtime_attestation=runtime_attestation,
                )
            except Exception as error:  # noqa: BLE001 - preserve per-request failures
                result = _error_result(
                    error=error,
                    started=started,
                    model_id=model_id,
                    revision=revision,
                    runtime=f"transformers {transformers.__version__}; torch {torch.__version__}",
                    peak_memory=torch.cuda.max_memory_allocated(),
                    device=device,
                    precision=precision,
                    runtime_attestation=runtime_attestation,
                )
            result["model_load_seconds"] = load_seconds
            results.append(attach_request_identity(result, request))
    except Exception as error:  # noqa: BLE001 - structured runtime failures
        results = [
            attach_request_identity(
                _error_result(
                    error=error,
                    started=load_started,
                    model_id=model_id,
                    revision=revision,
                    runtime=f"transformers {transformers.__version__}; torch {torch.__version__}",
                    peak_memory=torch.cuda.max_memory_allocated(),
                    device=device,
                    precision=precision,
                    runtime_attestation=runtime_attestation,
                ),
                request,
            )
            for request in requests
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
    device: str,
    precision: str,
    runtime_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    error_name = type(error).__name__
    unsupported = (
        error_name
        in {
            "NotImplementedError",
            "UnsupportedError",
            "HIPError",
        }
        or "not implemented for" in str(error).lower()
    )
    return _result(
        status="unsupported" if unsupported else "failed",
        answer=None,
        started=started,
        model_id=model_id,
        revision=revision,
        runtime=runtime,
        peak_memory=peak_memory,
        device=device,
        precision=precision,
        runtime_attestation=runtime_attestation,
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
    device: str,
    precision: str,
    runtime_attestation: Mapping[str, Any] | None,
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
        "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
        "precision": precision,
        "device": device,
        "runtime_attestation": (
            dict(runtime_attestation) if runtime_attestation is not None else None
        ),
        "peak_memory_bytes": peak_memory,
        "error_type": type(error).__name__ if error else None,
        "error_message": message,
        "min_pixels_per_frame": MIN_PIXELS_PER_FRAME,
        "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
    }


if __name__ == "__main__":
    raise SystemExit(main())
