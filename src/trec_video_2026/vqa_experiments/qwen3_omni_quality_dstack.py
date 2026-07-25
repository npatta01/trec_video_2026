"""Run the paired frames-only and native-audio Qwen3-Omni quality probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .omni_contract import parse_prediction
from .qwen3_omni_dstack import (
    AUDIO_INPUT_KEYS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    _failure_details,
    _generate_one,
    _load_model,
    _preserve_failure_attestation,
    _sha256_file,
    _write_predictions,
)
from .qwen3_omni_quality_contract import (
    quality_contract_sha256,
    validate_uploaded_quality_contract,
)


def _condition_request_sha256(
    example: dict[str, Any], condition: dict[str, Any]
) -> str:
    request = {"example": example, "condition": condition}
    encoded = json.dumps(
        request, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _peak_memory(torch: Any, method: str) -> int:
    cuda = getattr(torch, "cuda", None)
    function = getattr(cuda, method, None)
    return int(function()) if function is not None else 0


def _base_record(
    *,
    example: dict[str, Any],
    condition: dict[str, Any],
    pair_index: int,
    run_id: str,
    runtime: dict[str, Any],
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "video_id": example["video_id"],
        "request_sha256": _condition_request_sha256(example, condition),
        "condition_id": condition["condition_id"],
        "audio_enabled": condition["audio_enabled"],
        "pair_index": pair_index,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "device": runtime["device"],
        "precision": "bfloat16",
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _prediction_record(
    *, raw_output: str, modality_attestation: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    record = _base_record(**kwargs)
    record.update(
        {
            "raw_output": raw_output,
            "input_contract_error": False,
            "modality_attestation": modality_attestation,
        }
    )
    try:
        record["parsed_output"] = parse_prediction(raw_output)
    except ValueError as error:
        record.update(
            {
                "status": "failed",
                "output_parse_status": "failed",
                "error_type": "output_parse_error",
                "error_message": str(error),
                "failure_stage": "parsing",
            }
        )
    else:
        record.update(
            {
                "status": "passed",
                "output_parse_status": "passed",
                "error_type": None,
                "error_message": None,
            }
        )
    return record


def _failure_record(
    *,
    error: Exception,
    input_contract_error: bool | None,
    failure_stage: str,
    **kwargs: Any,
) -> dict[str, Any]:
    record = _base_record(**kwargs)
    record.update(
        {
            "status": "failed",
            "raw_output": "",
            "output_parse_status": "not_attempted",
            "input_contract_error": input_contract_error,
            "failure_stage": failure_stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "modality_attestation": {
                "same_request": False,
                "same_forward_pass": False,
                "generation_call_count": 0,
                "generation_input_keys": [],
                "first_forward_input_keys": [],
                "vision_input_keys": [],
                "audio_input_keys": [],
                "vision_element_count": 0,
                "audio_element_count": 0,
                "vision_input_sha256": None,
                "audio_input_sha256": None,
            },
        }
    )
    return _preserve_failure_attestation(record, error)


def _condition_passed(record: dict[str, Any]) -> bool:
    attestation = record.get("modality_attestation")
    if not isinstance(attestation, dict):
        return False
    audio_enabled = record["audio_enabled"]
    expected_audio_keys = sorted(AUDIO_INPUT_KEYS) if audio_enabled else []
    return (
        record.get("status") == "passed"
        and record.get("output_parse_status") == "passed"
        and attestation.get("same_request") is True
        and attestation.get("same_forward_pass") is True
        and attestation.get("generation_call_count") == 1
        and attestation.get("vision_element_count") == 12
        and attestation.get("audio_element_count") == (1 if audio_enabled else 0)
        and attestation.get("audio_input_keys") == expected_audio_keys
        and attestation.get("generation", {}).get("ended_with_eos") is True
    )


def _add_runtime_metadata(
    record: dict[str, Any], runtime: dict[str, Any], model_load_seconds: float
) -> None:
    record.update(
        {
            "runtime_image": RUNTIME_IMAGE,
            "runtime_versions": runtime["runtime_versions"],
            "gpu_name": runtime["gpu_name"],
            "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
            "model_load_seconds": model_load_seconds,
        }
    )


def _reserved_output_paths(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        output_dir / name
        for name in (
            "predictions.jsonl",
            "summary.json",
            "manifest.json",
            "preflight.json",
        )
    )
    if any(path.exists() for path in paths):
        existing = next(path for path in paths if path.exists())
        raise FileExistsError(f"refusing to overwrite {existing}")
    return paths


def _preflight_record(contract: dict[str, Any], run_id: str) -> dict[str, Any]:
    examples = contract["examples"]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "quality_contract_sha256": quality_contract_sha256(contract),
        "video_count": len(examples),
        "frame_files": sum(len(example["frame_paths"]) for example in examples),
        "audio_files": len(examples),
        "condition_ids": [condition["condition_id"] for condition in contract["conditions"]],
    }


def run_quality_probe(
    *, contract_path: Path, data_root: Path, output_dir: Path, run_id: str
) -> dict[str, Any]:
    """Run adjacent condition pairs and stop after a failed first-pair canary."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_uploaded_quality_contract(contract, data_root, verify_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path, summary_path, manifest_path, preflight_path = _reserved_output_paths(
        output_dir
    )
    preflight = _preflight_record(contract, run_id)
    contract_digest = preflight["quality_contract_sha256"]
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    conditions = contract["conditions"]
    predictions: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    torch = None
    try:
        torch, processor, model, runtime = _load_model()
        model_load_seconds = time.perf_counter() - load_started
    except Exception as load_error:  # noqa: BLE001
        model_load_seconds = time.perf_counter() - load_started
        runtime = {
            "device": "unavailable",
            "runtime_versions": {},
            "gpu_name": "unavailable",
            "gpu_total_memory_bytes": 0,
        }
        example = contract["examples"][0]
        for condition in conditions:
            record = _failure_record(
                example=example,
                condition=condition,
                pair_index=0,
                run_id=run_id,
                runtime=runtime,
                elapsed_seconds=model_load_seconds,
                peak_memory_bytes=_peak_memory(torch, "max_memory_allocated"),
                error=load_error,
                input_contract_error=False,
                failure_stage="model_load",
            )
            _add_runtime_metadata(record, runtime, model_load_seconds)
            predictions.append(record)
            _write_predictions(predictions_path, predictions)
    else:
        for pair_index, example in enumerate(contract["examples"]):
            pair_passed = True
            for condition in conditions:
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
                        audio_enabled=condition["audio_enabled"],
                    )
                    record = _prediction_record(
                        example=example,
                        condition=condition,
                        pair_index=pair_index,
                        run_id=run_id,
                        runtime=runtime,
                        elapsed_seconds=time.perf_counter() - started,
                        peak_memory_bytes=_peak_memory(torch, "max_memory_allocated"),
                        raw_output=raw_output,
                        modality_attestation=attestation,
                    )
                except Exception as error:  # noqa: BLE001
                    underlying, stage, input_contract_error = _failure_details(error)
                    record = _failure_record(
                        example=example,
                        condition=condition,
                        pair_index=pair_index,
                        run_id=run_id,
                        runtime=runtime,
                        elapsed_seconds=time.perf_counter() - started,
                        peak_memory_bytes=_peak_memory(torch, "max_memory_allocated"),
                        error=underlying,
                        input_contract_error=input_contract_error,
                        failure_stage=stage,
                    )
                _add_runtime_metadata(record, runtime, model_load_seconds)
                record["peak_memory_reserved_bytes"] = _peak_memory(
                    torch, "max_memory_reserved"
                )
                predictions.append(record)
                _write_predictions(predictions_path, predictions)
                pair_passed = pair_passed and _condition_passed(record)
            if pair_index == 0 and not pair_passed:
                break

    _write_predictions(predictions_path, predictions)
    canary_passed = len(predictions) >= len(conditions) and all(
        _condition_passed(record) for record in predictions[: len(conditions)]
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "quality_contract_sha256": contract_digest,
        "predictions_sha256": _sha256_file(predictions_path),
        "runner_sha256": _sha256_file(Path(__file__)),
        "rows": len(predictions),
        "canary_passed": canary_passed,
        "succeeded": len(predictions) == len(contract["examples"]) * len(conditions)
        and all(_condition_passed(record) for record in predictions),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "inputs": {"quality_contract_sha256": contract_digest},
        "outputs": {
            "predictions.jsonl": _sha256_file(predictions_path),
            "summary.json": _sha256_file(summary_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = run_quality_probe(
        contract_path=args.contract,
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
