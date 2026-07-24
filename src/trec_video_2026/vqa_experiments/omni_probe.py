"""Run pinned Phi-4 MM or Gemma 3n joint vision/audio probes.

This module intentionally imports the heavyweight inference dependencies only
inside runtime functions. Repository unit tests therefore remain lightweight;
the probe itself runs inside the pinned ROCm Podman image.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .omni_contract import (
    contract_sha256,
    promotion_decision,
    request_sha256,
    validate_inference_evidence,
)
from .omni_runtime import (
    attest_generation_inputs,
    build_prediction_record,
    build_runtime_failure_record,
    gemma_messages,
    phi_prompt,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    vision_input_keys: frozenset[str]
    audio_input_keys: frozenset[str]


MODEL_SPECS = {
    "phi4mm": ModelSpec(
        model_id="microsoft/Phi-4-multimodal-instruct",
        revision="93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        vision_input_keys=frozenset({"input_image_embeds"}),
        audio_input_keys=frozenset({"input_audio_embeds"}),
    ),
    "gemma3n": ModelSpec(
        model_id="google/gemma-3n-E4B-it",
        revision="c1221e9c62e34a43ab7ffacd1be0ea71f126ef10",
        vision_input_keys=frozenset({"pixel_values"}),
        audio_input_keys=frozenset({"input_features"}),
    ),
}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ProbeStageError(RuntimeError):
    def __init__(self, stage: str, original_error: Exception) -> None:
        super().__init__(f"{stage}: {original_error}")
        self.stage = stage
        self.original_error = original_error


def classify_probe_failure(
    error: Exception,
) -> tuple[Exception, str, bool | None]:
    if not isinstance(error, ProbeStageError):
        return error, "internal", None
    input_contract_error = True if error.stage in {"processor", "attestation"} else None
    return error.original_error, error.stage, input_contract_error


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run-id must contain only ASCII letters, digits, dots, underscores, "
            "or hyphens and must start with a letter or digit"
        )
    return run_id


def resolve_example_paths(example: dict[str, Any], data_root: Path) -> dict[str, Any]:
    resolved = dict(example)
    resolved["frame_paths"] = [str(data_root / path) for path in example["frame_paths"]]
    resolved["audio_path"] = str(data_root / example["audio_path"])
    resolved["media_path"] = str(data_root / example["media_path"])
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_description(torch: Any | None) -> str:
    if torch is None:
        return "unknown (PyTorch import failed)"
    if not torch.cuda.is_available():
        return "cpu (ROCm accelerator unavailable)"
    hip = torch.version.hip
    if hip:
        runtime = f"ROCm {hip}"
    else:
        runtime = f"NVIDIA CUDA {torch.version.cuda or 'unknown'}"
    return f"cuda:0 ({runtime}; {torch.cuda.get_device_name(0)})"


def _runtime_versions(
    torch: Any | None, transformers: Any | None
) -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unavailable"),
        "transformers": getattr(transformers, "__version__", "unavailable"),
        "rocm": getattr(getattr(torch, "version", None), "hip", None),
        "cuda": getattr(getattr(torch, "version", None), "cuda", None),
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


def _load_backend(model_key: str, model_path: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible accelerator is unavailable to PyTorch")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=model_key == "phi4mm",
        local_files_only=True,
    )
    common = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "local_files_only": True,
        "_attn_implementation": "sdpa" if model_key == "phi4mm" else "eager",
    }
    if model_key == "phi4mm":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **common,
        )
    else:
        from transformers import Gemma3nForConditionalGeneration

        model = Gemma3nForConditionalGeneration.from_pretrained(model_path, **common)
    model.eval()
    return torch, processor, model


def _processor_inputs_phi(processor: Any, example: dict[str, Any]) -> dict[str, Any]:
    import soundfile
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in example["frame_paths"]]
    audio = soundfile.read(example["audio_path"])
    return processor(
        text=phi_prompt(example),
        images=images,
        audios=[audio],
        return_tensors="pt",
    )


def _processor_inputs_gemma(processor: Any, example: dict[str, Any]) -> dict[str, Any]:
    return processor.apply_chat_template(
        gemma_messages(example),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _generate_one(
    *,
    model_key: str,
    spec: ModelSpec,
    processor: Any,
    model: Any,
    torch: Any,
    example: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    try:
        inputs = (
            _processor_inputs_phi(processor, example)
            if model_key == "phi4mm"
            else _processor_inputs_gemma(processor, example)
        )
        processor_inputs = dict(inputs)
        input_length = inputs["input_ids"].shape[1]
    except Exception as error:
        raise ProbeStageError("processor", error) from error

    try:
        if model_key == "gemma3n":
            inputs = inputs.to("cuda:0", dtype=torch.bfloat16)
        else:
            inputs = inputs.to("cuda:0")
    except Exception as error:
        raise ProbeStageError("device_transfer", error) from error

    try:
        with torch.inference_mode(), _capture_first_forward(model) as captures:
            generated = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )
    except Exception as error:
        raise ProbeStageError("generation", error) from error
    if not captures:
        error = RuntimeError("model.generate did not invoke the model forward")
        raise ProbeStageError("attestation", error) from error
    try:
        attestation = attest_generation_inputs(
            processor_inputs,
            first_forward_input_keys=captures[0],
            vision_input_keys=set(spec.vision_input_keys),
            audio_input_keys=set(spec.audio_input_keys),
        )
    except Exception as error:
        raise ProbeStageError("attestation", error) from error
    try:
        response_tokens = generated[:, input_length:]
        raw_output = processor.batch_decode(
            response_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    except Exception as error:
        raise ProbeStageError("decode", error) from error
    return raw_output, attestation


def run_probe(
    *,
    model_key: str,
    model_path: Path,
    contract_path: Path,
    data_root: Path,
    output_dir: Path,
    runtime_image: str,
    runtime_image_digest: str,
    run_id: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    validate_run_id(run_id)
    spec = MODEL_SPECS[model_key]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_inference_evidence(contract, data_root, verify_files=True)
    actual_contract_sha256 = contract_sha256(contract)
    if actual_contract_sha256 != expected_contract_sha256:
        raise ValueError(
            "frozen contract digest mismatch: "
            f"expected {expected_contract_sha256}, got {actual_contract_sha256}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    predictions_path = output_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []

    load_started = time.perf_counter()
    try:
        torch, processor, model = _load_backend(model_key, model_path)
        model_load_seconds = time.perf_counter() - load_started
        device = _device_description(torch)
        versions = _runtime_versions(torch, __import__("transformers"))
        runner_hash = _file_sha256(Path(__file__))
    # Every dependency, device, and model-load failure must become an attested
    # result row rather than aborting before the three-row gate is recorded.
    except Exception as load_error:  # noqa: BLE001
        model_load_seconds = time.perf_counter() - load_started
        torch = sys.modules.get("torch")
        transformers = sys.modules.get("transformers")
        device = _device_description(torch)
        versions = _runtime_versions(torch, transformers)
        runner_hash = _file_sha256(Path(__file__))
        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch is not None and torch.cuda.is_available()
            else 0
        )
        for example in contract["examples"]:
            record = build_runtime_failure_record(
                example=example,
                request_digest=request_sha256(example),
                model_id=spec.model_id,
                model_revision=spec.revision,
                runtime_image_digest=runtime_image_digest,
                run_id=run_id,
                device=device,
                precision="bfloat16",
                elapsed_seconds=model_load_seconds,
                peak_memory_bytes=peak,
                error=load_error,
                input_contract_error=False,
                failure_stage="model_load",
            )
            record.update(
                {
                    "model_load_seconds": model_load_seconds,
                    "runtime_image": runtime_image,
                    "runtime_versions": versions,
                    "runner_sha256": runner_hash,
                }
            )
            predictions.append(record)
    else:
        for contract_example in contract["examples"]:
            example = resolve_example_paths(contract_example, data_root)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                raw_output, attestation = _generate_one(
                    model_key=model_key,
                    spec=spec,
                    processor=processor,
                    model=model,
                    torch=torch,
                    example=example,
                )
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_allocated())
                record = build_prediction_record(
                    example=contract_example,
                    request_digest=request_sha256(contract_example),
                    model_id=spec.model_id,
                    model_revision=spec.revision,
                    runtime_image_digest=runtime_image_digest,
                    run_id=run_id,
                    device=device,
                    precision="bfloat16",
                    elapsed_seconds=elapsed,
                    peak_memory_bytes=peak,
                    raw_output=raw_output,
                    modality_attestation=attestation,
                )
            # A compatibility probe intentionally records arbitrary backend
            # failures so a verified ROCm/input blocker survives in the artifact.
            except Exception as error:  # noqa: BLE001
                elapsed = time.perf_counter() - started
                peak = int(torch.cuda.max_memory_allocated())
                underlying_error, failure_stage, input_contract_error = (
                    classify_probe_failure(error)
                )
                record = build_runtime_failure_record(
                    example=contract_example,
                    request_digest=request_sha256(contract_example),
                    model_id=spec.model_id,
                    model_revision=spec.revision,
                    runtime_image_digest=runtime_image_digest,
                    run_id=run_id,
                    device=device,
                    precision="bfloat16",
                    elapsed_seconds=elapsed,
                    peak_memory_bytes=peak,
                    error=underlying_error,
                    input_contract_error=input_contract_error,
                    failure_stage=failure_stage,
                )
                underlying_error.__traceback__ = None
            record.update(
                {
                    "model_load_seconds": model_load_seconds,
                    "runtime_image": runtime_image,
                    "runtime_versions": versions,
                    "runner_sha256": runner_hash,
                }
            )
            predictions.append(record)
            predictions_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
                encoding="utf-8",
            )

    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "runtime_image": runtime_image,
        "runtime_image_digest": runtime_image_digest,
        "contract_sha256": actual_contract_sha256,
        "predictions_sha256": _file_sha256(predictions_path),
        "promotion": promotion_decision(contract, predictions),
        "rows": len(predictions),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a pinned non-Qwen joint vision/audio compatibility probe."
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()
    summary = run_probe(
        model_key=args.model,
        model_path=args.model_path,
        contract_path=args.contract,
        data_root=args.data_root,
        output_dir=args.output_dir,
        runtime_image=args.runtime_image,
        runtime_image_digest=args.runtime_image_digest,
        run_id=args.run_id,
        expected_contract_sha256=args.expected_contract_sha256,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["promotion"]["decision"] == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
