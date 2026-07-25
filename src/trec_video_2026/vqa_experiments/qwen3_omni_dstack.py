"""Pinned Qwen3-Omni compatibility probe for a detached dstack task."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import time
import wave
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from .omni_contract import (
    contract_sha256,
    parse_prediction,
    promotion_decision,
    request_sha256,
    validate_contract,
)
from .omni_runtime import (
    OUTPUT_INSTRUCTION,
    build_prediction_record,
    build_runtime_failure_record,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
RUNTIME_IMAGE_DIGEST = (
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
RUNTIME_IMAGE = f"docker.io/qwenllm/qwen3-omni@{RUNTIME_IMAGE_DIGEST}"
CONTRACT_SHA256 = "b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
CONTRACT_FILE_SHA256 = (
    "dfd2d765e0dc0947e81e0387e09b88a0b75221a91571f43ae9812fdd66151309"
)
VIDEO_IDS = ("NN7G28sBOlY", "NUmb2P3Bzu8", "NkXPPJoZ-h8")
VISION_INPUT_KEYS = frozenset({"pixel_values", "image_grid_thw"})
AUDIO_INPUT_KEYS = frozenset({"input_features", "feature_attention_mask"})
MIN_GPU_MEMORY_BYTES = 80 * 1024**3
MAX_NEW_TOKENS = 2048
GENERATION_SEED = 20260724
EXPECTED_EOS_TOKEN_ID = 151645
EXPECTED_PAD_TOKEN_ID = 151643
EXPECTED_EOS_TOKEN = "<|im_end|>"
EXPECTED_PAD_TOKEN = "<|endoftext|>"


class ProbeStageError(RuntimeError):
    """Preserve the boundary at which a compatibility attempt failed."""

    def __init__(
        self,
        stage: str,
        original_error: Exception,
        *,
        modality_attestation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{stage}: {original_error}")
        self.stage = stage
        self.original_error = original_error
        self.modality_attestation = modality_attestation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_input_path(data_root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe contract input path: {relative}")
    return data_root.joinpath(*path.parts)


def validate_panel_identity(contract: dict[str, Any], actual_digest: str) -> None:
    """Fail closed unless this is the exact frozen three-video contract."""

    if actual_digest != CONTRACT_SHA256:
        raise ValueError(
            f"contract digest mismatch: expected {CONTRACT_SHA256}, got {actual_digest}"
        )
    examples = contract.get("examples")
    if not isinstance(examples, list):
        raise TypeError("contract examples must be a list")
    video_ids = tuple(row.get("video_id") for row in examples)
    if video_ids != VIDEO_IDS:
        raise ValueError(f"contract video IDs must be {VIDEO_IDS}, got {video_ids}")


def verify_uploaded_evidence(contract: dict[str, Any], data_root: Path) -> None:
    """Verify every uploaded frame and audio file against the frozen contract."""

    validate_contract(contract, data_root, verify_files=False)
    for example in contract["examples"]:
        pairs = [
            (example["audio_path"], example["audio_sha256"]),
            *zip(example["frame_paths"], example["frame_sha256s"], strict=True),
        ]
        for relative, expected in pairs:
            path = _safe_input_path(data_root, relative)
            if not path.is_file():
                raise ValueError(f"missing frozen input: {relative}")
            actual = _sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"frozen input hash mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )


def build_messages(example: dict[str, Any], data_root: Path) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {
            "type": "image",
            "image": str(_safe_input_path(data_root, relative)),
        }
        for relative in example["frame_paths"]
    ]
    content.append(
        {
            "type": "audio",
            "audio": str(_safe_input_path(data_root, example["audio_path"])),
        }
    )
    content.append(
        {
            "type": "text",
            "text": f"{OUTPUT_INSTRUCTION}\nQuestion: {example['question']}",
        }
    )
    return [{"role": "user", "content": content}]


def generation_special_token_ids(processor: Any) -> dict[str, int]:
    """Resolve explicit termination IDs from the checkpoint tokenizer."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor has no tokenizer")
    expected = {
        "eos_token_id": (EXPECTED_EOS_TOKEN_ID, EXPECTED_EOS_TOKEN),
        "pad_token_id": (EXPECTED_PAD_TOKEN_ID, EXPECTED_PAD_TOKEN),
    }
    result: dict[str, int] = {}
    for attribute, (expected_id, expected_token) in expected.items():
        value = getattr(tokenizer, attribute, None)
        if type(value) is not int:
            raise ValueError(f"tokenizer {attribute} must be an integer")
        if value != expected_id:
            raise ValueError(
                f"tokenizer {attribute} must be {expected_id}, got {value}"
            )
        token = tokenizer.convert_ids_to_tokens(value)
        if token != expected_token:
            raise ValueError(
                f"tokenizer {attribute} must resolve to {expected_token!r}, "
                f"got {token!r}"
            )
        result[attribute] = value
    return result


def normalize_generation_output(raw_output: str) -> str:
    """Extract only a valid terminal answer object after Thinking output."""

    text = raw_output.strip()
    try:
        parsed = parse_prediction(text)
    except ValueError:
        pass
    else:
        return json.dumps(parsed, ensure_ascii=False)

    for start in reversed([index for index, char in enumerate(text) if char == "{"]):
        candidate = text[start:]
        try:
            parsed = parse_prediction(candidate)
        except ValueError:
            continue
        return json.dumps(parsed, ensure_ascii=False)
    return text


def _input_bytes(value: Any) -> bytes:
    candidate = value
    for method in ("detach", "cpu", "contiguous"):
        function = getattr(candidate, method, None)
        if function is not None:
            candidate = function()
    numpy = getattr(candidate, "numpy", None)
    if numpy is not None:
        candidate = numpy()
    tobytes = getattr(candidate, "tobytes", None)
    if tobytes is None:
        raise TypeError(f"cannot attest input of type {type(value).__name__}")
    return tobytes()


def _inputs_sha256(inputs: dict[str, Any], keys: frozenset[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_input_bytes(inputs[key]))
    return digest.hexdigest()


def attest_qwen_inputs(
    processor_inputs: dict[str, Any],
    *,
    first_forward_input_keys: set[str],
) -> dict[str, Any]:
    """Attest processed visual and audio tensors in one first model forward."""

    available = {key for key, value in processor_inputs.items() if value is not None}
    missing = (VISION_INPUT_KEYS | AUDIO_INPUT_KEYS) - available
    if missing:
        raise ValueError(f"processor omitted joint modality inputs: {sorted(missing)}")
    required = VISION_INPUT_KEYS | AUDIO_INPUT_KEYS
    missing_forward = required - first_forward_input_keys
    if missing_forward:
        raise ValueError(
            f"first forward omitted joint modality inputs: {sorted(missing_forward)}"
        )
    image_shape = getattr(processor_inputs["image_grid_thw"], "shape", ())
    audio_shape = getattr(processor_inputs["input_features"], "shape", ())
    if not image_shape or int(image_shape[0]) != 12:
        raise ValueError("processed request must contain exactly 12 visual elements")
    if not audio_shape or int(audio_shape[0]) != 1:
        raise ValueError("processed request must contain exactly one audio element")
    return {
        "same_request": True,
        "same_forward_pass": True,
        "generation_call_count": 1,
        "generation_input_keys": sorted(available),
        "first_forward_input_keys": sorted(first_forward_input_keys),
        "vision_input_keys": sorted(VISION_INPUT_KEYS),
        "audio_input_keys": sorted(AUDIO_INPUT_KEYS),
        "vision_element_count": 12,
        "audio_element_count": 1,
        "vision_input_sha256": _inputs_sha256(processor_inputs, VISION_INPUT_KEYS),
        "audio_input_sha256": _inputs_sha256(processor_inputs, AUDIO_INPUT_KEYS),
    }


@contextlib.contextmanager
def _capture_first_forward(model: Any) -> Iterator[list[set[str]]]:
    captures: list[set[str]] = []

    def hook(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        if not captures:
            captures.append({key for key, value in kwargs.items() if value is not None})

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captures
    finally:
        handle.remove()


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _runtime_versions(torch: Any, transformers: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("accelerate", "flash-attn", "qwen-omni-utils"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "packages": packages,
    }


def _gpu_attestation(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA accelerator is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bfloat16")
    properties = torch.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    if total_memory <= MIN_GPU_MEMORY_BYTES:
        raise RuntimeError(
            "GPU memory headroom is insufficient: "
            f"required more than {MIN_GPU_MEMORY_BYTES}, got {total_memory}"
        )
    return {
        "device": f"cuda:0 (NVIDIA CUDA; {properties.name})",
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": total_memory,
        "cuda_version": torch.version.cuda,
        "bf16_supported": True,
    }


def preflight(contract_path: Path, data_root: Path) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import (
        Qwen3OmniMoeProcessor,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    contract_file_digest = _sha256_file(contract_path)
    if contract_file_digest != CONTRACT_FILE_SHA256:
        raise ValueError(
            "contract file digest mismatch: "
            f"expected {CONTRACT_FILE_SHA256}, got {contract_file_digest}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    canonical_digest = contract_sha256(contract)
    validate_panel_identity(contract, canonical_digest)
    verify_uploaded_evidence(contract, data_root)
    gpu = _gpu_attestation(torch)
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "contract_sha256": canonical_digest,
        "contract_file_sha256": contract_file_digest,
        "video_ids": list(VIDEO_IDS),
        "runtime_versions": _runtime_versions(torch, transformers),
        "processor_class": Qwen3OmniMoeProcessor.__name__,
        "model_class": Qwen3OmniMoeThinkerForConditionalGeneration.__name__,
        **gpu,
    }


def _load_model() -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from transformers import (
        Qwen3OmniMoeProcessor,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    gpu = _gpu_attestation(torch)
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return (
        torch,
        processor,
        model,
        {
            **gpu,
            "runtime_versions": _runtime_versions(torch, transformers),
        },
    )


def _generate_one(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    example: dict[str, Any],
    data_root: Path,
) -> tuple[str, dict[str, Any]]:
    try:
        inputs = processor.apply_chat_template(
            build_messages(example, data_root),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        processor_inputs = dict(inputs)
        input_length = int(inputs["input_ids"].shape[1])
        special_token_ids = generation_special_token_ids(processor)
    except Exception as error:
        raise ProbeStageError("processor", error) from error
    try:
        device = next(model.parameters()).device
        inputs = inputs.to(device).to(model.dtype)
    except Exception as error:
        raise ProbeStageError("device_transfer", error) from error
    try:
        torch.manual_seed(GENERATION_SEED)
        torch.cuda.manual_seed_all(GENERATION_SEED)
        with torch.inference_mode(), _capture_first_forward(model) as captures:
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                use_audio_in_video=False,
                **special_token_ids,
            )
    except Exception as error:
        raise ProbeStageError("generation", error) from error
    if not captures:
        error = RuntimeError("model.generate did not invoke the model forward")
        raise ProbeStageError("attestation", error) from error
    try:
        attestation = attest_qwen_inputs(
            processor_inputs,
            first_forward_input_keys=captures[0],
        )
    except Exception as error:
        raise ProbeStageError("attestation", error) from error
    try:
        response_tokens = generated[:, input_length:]
        generated_new_token_count = int(response_tokens.shape[1])
        final_generated_token_id = int(response_tokens[0, -1].item())
        ended_with_eos = final_generated_token_id == special_token_ids["eos_token_id"]
        raw_output = processor.batch_decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as error:
        raise ProbeStageError("decode", error) from error
    generation_attestation = {
        "checkpoint_generation_config": True,
        "do_sample": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": GENERATION_SEED,
        **special_token_ids,
        "eos_token": EXPECTED_EOS_TOKEN,
        "pad_token": EXPECTED_PAD_TOKEN,
        "generated_new_token_count": generated_new_token_count,
        "final_generated_token_id": final_generated_token_id,
        "ended_with_eos": ended_with_eos,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
    }
    if not ended_with_eos:
        error = RuntimeError(
            "generation did not terminate with EOS: "
            f"generated_new_token_count={generated_new_token_count}, "
            f"final_generated_token_id={final_generated_token_id}"
        )
        attestation["generation"] = generation_attestation
        raise ProbeStageError(
            "termination",
            error,
            modality_attestation=attestation,
        ) from error
    attestation["generation"] = generation_attestation
    return normalize_generation_output(raw_output), attestation


def _failure_details(error: Exception) -> tuple[Exception, str, bool | None]:
    if not isinstance(error, ProbeStageError):
        return error, "internal", None
    is_contract = error.stage in {"processor", "attestation"}
    return error.original_error, error.stage, True if is_contract else None


def _preserve_failure_attestation(
    record: dict[str, Any], error: Exception
) -> dict[str, Any]:
    """Retain proven modality and generation facts on a failed row."""

    if (
        isinstance(error, ProbeStageError)
        and error.modality_attestation is not None
    ):
        record["modality_attestation"] = error.modality_attestation
    return record


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_probe(
    *,
    contract_path: Path,
    data_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    preflight_record = preflight(contract_path, data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    for reserved in ("predictions.jsonl", "summary.json", "manifest.json"):
        if (output_dir / reserved).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / reserved}")
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    predictions: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    torch = None
    try:
        torch, processor, model, runtime = _load_model()
        model_load_seconds = time.perf_counter() - load_started
    except Exception as load_error:  # noqa: BLE001
        model_load_seconds = time.perf_counter() - load_started
        runtime = preflight_record
        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch is not None and torch.cuda.is_available()
            else 0
        )
        for example in contract["examples"]:
            record = build_runtime_failure_record(
                example=example,
                request_digest=request_sha256(example),
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                runtime_image_digest=RUNTIME_IMAGE_DIGEST,
                run_id=run_id,
                device=runtime["device"],
                precision="bfloat16",
                elapsed_seconds=model_load_seconds,
                peak_memory_bytes=peak,
                error=load_error,
                input_contract_error=False,
                failure_stage="model_load",
            )
            record.update(
                {
                    "runtime_image": RUNTIME_IMAGE,
                    "runtime_versions": runtime["runtime_versions"],
                    "gpu_name": runtime["gpu_name"],
                    "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
                    "model_load_seconds": model_load_seconds,
                }
            )
            predictions.append(record)
    else:
        for example in contract["examples"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                raw_output, attestation = _generate_one(
                    torch=torch,
                    processor=processor,
                    model=model,
                    example=example,
                    data_root=data_root,
                )
                elapsed = time.perf_counter() - started
                record = build_prediction_record(
                    example=example,
                    request_digest=request_sha256(example),
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    runtime_image_digest=RUNTIME_IMAGE_DIGEST,
                    run_id=run_id,
                    device=runtime["device"],
                    precision="bfloat16",
                    elapsed_seconds=elapsed,
                    peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                    raw_output=raw_output,
                    modality_attestation=attestation,
                )
            except Exception as error:  # noqa: BLE001
                elapsed = time.perf_counter() - started
                underlying, stage, input_contract_error = _failure_details(error)
                record = build_runtime_failure_record(
                    example=example,
                    request_digest=request_sha256(example),
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    runtime_image_digest=RUNTIME_IMAGE_DIGEST,
                    run_id=run_id,
                    device=runtime["device"],
                    precision="bfloat16",
                    elapsed_seconds=elapsed,
                    peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                    error=underlying,
                    input_contract_error=input_contract_error,
                    failure_stage=stage,
                )
                _preserve_failure_attestation(record, error)
            audio_path = _safe_input_path(data_root, example["audio_path"])
            record.update(
                {
                    "runtime_image": RUNTIME_IMAGE,
                    "runtime_versions": runtime["runtime_versions"],
                    "gpu_name": runtime["gpu_name"],
                    "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
                    "model_load_seconds": model_load_seconds,
                    "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "audio_duration_seconds": _audio_duration(audio_path),
                    "visual_input_contract": "12 frozen sampled frames",
                    "native_temporal_video": False,
                }
            )
            predictions.append(record)
            _write_predictions(output_dir / "predictions.jsonl", predictions)

    predictions_path = output_dir / "predictions.jsonl"
    _write_predictions(predictions_path, predictions)
    verify_uploaded_evidence(contract, data_root)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "contract_sha256": CONTRACT_SHA256,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "predictions_sha256": _sha256_file(predictions_path),
        "runner_sha256": _sha256_file(Path(__file__)),
        "rows": len(predictions),
        "compatibility": promotion_decision(contract, predictions),
        "semantic_promotion_gate": {
            "decision": "not_evaluated",
            "reason": "label-isolated adjudication is required",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "inputs": {
            "contract_sha256": CONTRACT_SHA256,
            "contract_file_sha256": CONTRACT_FILE_SHA256,
            "video_ids": list(VIDEO_IDS),
        },
        "outputs": {
            "predictions.jsonl": _sha256_file(predictions_path),
            "summary.json": _sha256_file(summary_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned Qwen3-Omni three-video compatibility probe."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(preflight(args.contract, args.data_root), sort_keys=True))
            return 0
        if args.output_dir is None or args.run_id is None:
            parser.error(
                "--output-dir and --run-id are required unless --preflight is used"
            )
        summary = run_probe(
            contract_path=args.contract,
            data_root=args.data_root,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failure_stage": "preflight" if args.preflight else "fatal",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
