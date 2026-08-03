"""Run the canary-gated paired Qwen3-Omni frozen-30 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .omni_contract import parse_prediction
from .omni_runtime import instruction_for_profile
from .qwen3_omni_dstack import (
    AUDIO_INPUT_KEYS,
    EXPECTED_EOS_TOKEN_ID,
    EXPECTED_PAD_TOKEN_ID,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    ProbeStageError,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    VISION_INPUT_KEYS,
    _failure_details,
    _generate_one,
    _load_model,
    _preserve_failure_attestation,
    _sha256_file,
)
from .qwen3_omni_frozen30_contract import (
    contract_sha256,
    validate_uploaded_contract,
)
from .qwen3_omni_frozen30_profiles import (
    A4_PROFILE,
    PROFILES,
    Frozen30ExperimentProfile,
    profile_for_key,
)

OUTPUT_NAMES = (
    "preflight.json",
    "canary.json",
    "predictions.jsonl",
    "summary.json",
    "manifest.json",
)
INSTRUCTION_PROFILE = "neutral"
SCHEMA_VERSION = 1

_SOURCE_DIRECTORY = Path(__file__).resolve().parent
_SOURCE_FILES = (
    "omni_contract.py",
    "omni_runtime.py",
    "qwen3_omni_dstack.py",
    "qwen3_omni_frozen30_contract.py",
    "qwen3_omni_frozen30_dstack.py",
    "qwen3_omni_frozen30_profiles.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(_canonical_bytes(row) + b"\n" for row in rows))


def _condition_request_sha256(
    example: dict[str, Any], condition: dict[str, Any]
) -> str:
    return hashlib.sha256(
        _canonical_bytes({"condition": condition, "example": example})
    ).hexdigest()


def _peak_memory(torch: Any, method: str) -> int:
    cuda = getattr(torch, "cuda", None)
    function = getattr(cuda, method, None)
    return int(function()) if function is not None else 0


def _base_record(
    *,
    example: dict[str, Any],
    condition: dict[str, Any],
    pair_index: int | None,
    scored: bool,
    run_id: str,
    contract_digest: str,
    instruction_sha256: str,
    profile: Frozen30ExperimentProfile,
    runtime: dict[str, Any],
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "contract_sha256": contract_digest,
        "video_id": example["video_id"],
        "condition_id": condition["condition_id"],
        "condition_request_sha256": _condition_request_sha256(example, condition),
        "audio_enabled": condition["audio_enabled"],
        "pair_index": pair_index,
        "scored": scored,
        "instruction_profile": profile.instruction_profile,
        "instruction_sha256": instruction_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "device": runtime["device"],
        "precision": "bfloat16",
        "runtime_versions": runtime["runtime_versions"],
        "gpu_name": runtime["gpu_name"],
        "gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _prediction_record(
    *,
    raw_output: str,
    modality_attestation: dict[str, Any],
    **kwargs: Any,
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
        parsed_output = parse_prediction(raw_output)
    except ValueError as error:
        record.update(
            {
                "status": "failed",
                "output_parse_status": "failed",
                "failure_stage": "parsing",
                "error_type": "output_parse_error",
                "error_message": str(error),
            }
        )
    else:
        record.update(
            {
                "status": "passed",
                "output_parse_status": "passed",
                "parsed_output": parsed_output,
                "failure_stage": None,
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


def _condition_passed(
    record: dict[str, Any], profile: Frozen30ExperimentProfile = A4_PROFILE
) -> bool:
    attestation = record.get("modality_attestation")
    if not isinstance(attestation, dict):
        return False
    audio_enabled = record.get("audio_enabled")
    if type(audio_enabled) is not bool:
        return False
    expected_audio_keys = sorted(AUDIO_INPUT_KEYS) if audio_enabled else []
    generation = attestation.get("generation")
    if type(generation) is not dict:
        return False
    instruction_sha256 = hashlib.sha256(
        instruction_for_profile(profile.instruction_profile).encode("utf-8")
    ).hexdigest()

    def exact_int(value: Any, expected: int) -> bool:
        return type(value) is int and value == expected

    def keys_prove_modalities(value: Any) -> bool:
        if (
            type(value) is not list
            or any(type(key) is not str for key in value)
            or len(value) != len(set(value))
        ):
            return False
        return (
            VISION_INPUT_KEYS.issubset(value)
            and sorted(key for key in value if key in AUDIO_INPUT_KEYS)
            == expected_audio_keys
        )

    return (
        record.get("status") == "passed"
        and record.get("output_parse_status") == "passed"
        and record.get("instruction_profile") == profile.instruction_profile
        and record.get("instruction_sha256") == instruction_sha256
        and attestation.get("same_request") is True
        and attestation.get("same_forward_pass") is True
        and exact_int(attestation.get("generation_call_count"), 1)
        and exact_int(attestation.get("vision_element_count"), 12)
        and exact_int(attestation.get("audio_element_count"), 1 if audio_enabled else 0)
        and type(attestation.get("vision_input_keys")) is list
        and attestation.get("vision_input_keys") == sorted(VISION_INPUT_KEYS)
        and type(attestation.get("audio_input_keys")) is list
        and attestation.get("audio_input_keys") == expected_audio_keys
        and keys_prove_modalities(attestation.get("generation_input_keys"))
        and keys_prove_modalities(attestation.get("first_forward_input_keys"))
        and exact_int(generation.get("eos_token_id"), EXPECTED_EOS_TOKEN_ID)
        and exact_int(generation.get("pad_token_id"), EXPECTED_PAD_TOKEN_ID)
        and exact_int(generation.get("seed"), profile.generation_seed)
        and exact_int(generation.get("max_new_tokens"), MAX_NEW_TOKENS)
        and generation.get("do_sample") is profile.do_sample
        and generation.get("enable_thinking") is profile.enable_thinking
        and generation.get("ended_with_eos") is True
    )


def _reserved_output_paths(output_dir: Path) -> dict[str, Path]:
    paths = {name: output_dir / name for name in OUTPUT_NAMES}
    reserved: list[Path] = []
    try:
        for path in paths.values():
            with path.open("xb"):
                pass
            reserved.append(path)
    except FileExistsError as error:
        for created in reserved:
            created.unlink()
        raise FileExistsError(f"refusing to overwrite {path}") from error
    except Exception:
        for created in reserved:
            created.unlink()
        raise
    return paths


def _unavailable_runtime() -> dict[str, Any]:
    return {
        "device": "unavailable",
        "runtime_versions": {},
        "gpu_name": "unavailable",
        "gpu_total_memory_bytes": 0,
    }


def _execute_condition(
    *,
    torch: Any,
    processor: Any,
    model: Any,
    runtime: dict[str, Any],
    example: dict[str, Any],
    condition: dict[str, Any],
    pair_index: int | None,
    scored: bool,
    data_root: Path,
    run_id: str,
    contract_digest: str,
    instruction: str,
    instruction_sha256: str,
    profile: Frozen30ExperimentProfile,
    model_load_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    common = {
        "example": example,
        "condition": condition,
        "pair_index": pair_index,
        "scored": scored,
        "run_id": run_id,
        "contract_digest": contract_digest,
        "instruction_sha256": instruction_sha256,
        "profile": profile,
        "runtime": runtime,
    }
    peak_memory_bytes = 0
    peak_memory_reserved_bytes = 0
    try:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as error:
            raise ProbeStageError("cuda_cleanup", error) from error
        raw_output, attestation = _generate_one(
            torch=torch,
            processor=processor,
            model=model,
            example=example,
            data_root=data_root,
            audio_enabled=condition["audio_enabled"],
            output_instruction=instruction,
            do_sample=profile.do_sample,
            enable_thinking=profile.enable_thinking,
            seed=profile.generation_seed,
        )
        try:
            peak_memory_bytes = _peak_memory(torch, "max_memory_allocated")
            peak_memory_reserved_bytes = _peak_memory(torch, "max_memory_reserved")
        except Exception as error:
            raise ProbeStageError(
                "memory_attestation",
                error,
                modality_attestation=attestation,
            ) from error
        record = _prediction_record(
            **common,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory_bytes=peak_memory_bytes,
            raw_output=raw_output,
            modality_attestation=attestation,
        )
    except Exception as error:  # noqa: BLE001
        underlying, stage, input_contract_error = _failure_details(error)
        try:
            peak_memory_bytes = _peak_memory(torch, "max_memory_allocated")
        except Exception:  # noqa: BLE001
            peak_memory_bytes = 0
        try:
            peak_memory_reserved_bytes = _peak_memory(torch, "max_memory_reserved")
        except Exception:  # noqa: BLE001
            peak_memory_reserved_bytes = 0
        record = _failure_record(
            **common,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory_bytes=peak_memory_bytes,
            error=underlying,
            input_contract_error=input_contract_error,
            failure_stage=stage,
        )
        _preserve_failure_attestation(record, error)
    record["peak_memory_reserved_bytes"] = peak_memory_reserved_bytes
    record["model_load_seconds"] = model_load_seconds
    return record


def _model_load_failure_canary(
    *,
    error: Exception,
    example: dict[str, Any],
    condition: dict[str, Any],
    run_id: str,
    contract_digest: str,
    instruction_sha256: str,
    profile: Frozen30ExperimentProfile,
    elapsed_seconds: float,
) -> dict[str, Any]:
    record = _failure_record(
        example=example,
        condition=condition,
        pair_index=None,
        scored=False,
        run_id=run_id,
        contract_digest=contract_digest,
        instruction_sha256=instruction_sha256,
        profile=profile,
        runtime=_unavailable_runtime(),
        elapsed_seconds=elapsed_seconds,
        peak_memory_bytes=0,
        error=error,
        input_contract_error=False,
        failure_stage="model_load",
    )
    record["peak_memory_reserved_bytes"] = 0
    record["model_load_seconds"] = elapsed_seconds
    return record


def _expected_scored_order(
    contract: dict[str, Any],
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> list[tuple[str, str, bool, int]]:
    return [
        (example["video_id"], condition_id, audio_enabled, pair_index)
        for pair_index, example in enumerate(contract["examples"])
        for condition_id, audio_enabled in profile.condition_specs
    ]


def _actual_scored_order(
    rows: list[dict[str, Any]],
) -> list[tuple[Any, Any, Any, Any]]:
    return [
        (
            row.get("video_id"),
            row.get("condition_id"),
            row.get("audio_enabled"),
            row.get("pair_index"),
        )
        for row in rows
    ]


def _validated_probe_inputs(
    *,
    contract_path: Path,
    data_root: Path,
    run_id: str,
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if profile_for_key(profile.key) != profile:
        raise ValueError("experiment profile does not match a sealed profile")
    if run_id != profile.run_id:
        raise ValueError(f"run_id must be {profile.run_id}")
    instruction = instruction_for_profile(profile.instruction_profile)
    instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_uploaded_contract(
        contract, data_root, verify_files=True, profile=profile
    )
    contract_digest = contract_sha256(contract)
    generation = {
        "seed": profile.generation_seed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": profile.do_sample,
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "pad_token_id": EXPECTED_PAD_TOKEN_ID,
    }
    if profile.enable_thinking is not None:
        generation["enable_thinking"] = profile.enable_thinking
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "contract_sha256": contract_digest,
        "instruction_profile": profile.instruction_profile,
        "instruction_sha256": instruction_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "video_count": len(contract["examples"]),
        "frame_files": sum(len(row["frame_paths"]) for row in contract["examples"]),
        "audio_files": len(contract["examples"]),
        "condition_ids": [
            condition["condition_id"] for condition in contract["conditions"]
        ],
        "canary_video_id": profile.canary_video_id,
        "generation": generation,
    }
    return contract, instruction, preflight


def preflight_frozen30_probe(
    *,
    contract_path: Path,
    data_root: Path,
    run_id: str,
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> dict[str, Any]:
    """Validate the real runner inputs and runtime pins without reserving output."""
    _contract, _instruction, preflight = _validated_probe_inputs(
        contract_path=contract_path,
        data_root=data_root,
        run_id=run_id,
        profile=profile,
    )
    return preflight


def run_frozen30_probe(
    *,
    contract_path: Path,
    data_root: Path,
    output_dir: Path,
    run_id: str,
    profile: Frozen30ExperimentProfile = A4_PROFILE,
) -> dict[str, Any]:
    """Run the profile's canary, then its complete frozen diagnostic panel."""
    contract, instruction, preflight = _validated_probe_inputs(
        contract_path=contract_path,
        data_root=data_root,
        run_id=run_id,
        profile=profile,
    )
    instruction_sha256 = preflight["instruction_sha256"]
    contract_digest = preflight["contract_sha256"]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _reserved_output_paths(output_dir)
    _write_json(paths["preflight.json"], preflight)
    _write_predictions(paths["predictions.jsonl"], [])

    canary_example = next(
        example
        for example in contract["examples"]
        if example["video_id"] == profile.canary_video_id
    )
    canary_condition = next(
        condition
        for condition in contract["conditions"]
        if condition["condition_id"] == profile.canary_condition_id
    )
    load_started = time.perf_counter()
    try:
        torch, processor, model, runtime = _load_model()
    except Exception as load_error:  # noqa: BLE001
        model_load_seconds = time.perf_counter() - load_started
        canary = _model_load_failure_canary(
            error=load_error,
            example=canary_example,
            condition=canary_condition,
            run_id=run_id,
            contract_digest=contract_digest,
            instruction_sha256=instruction_sha256,
            profile=profile,
            elapsed_seconds=model_load_seconds,
        )
        scored_rows: list[dict[str, Any]] = []
    else:
        model_load_seconds = time.perf_counter() - load_started
        canary = _execute_condition(
            torch=torch,
            processor=processor,
            model=model,
            runtime=runtime,
            example=canary_example,
            condition=canary_condition,
            pair_index=None,
            scored=False,
            data_root=data_root,
            run_id=run_id,
            contract_digest=contract_digest,
            instruction=instruction,
            instruction_sha256=instruction_sha256,
            profile=profile,
            model_load_seconds=model_load_seconds,
        )
        scored_rows = []
        if _condition_passed(canary, profile):
            for pair_index, example in enumerate(contract["examples"]):
                for condition in contract["conditions"]:
                    record = _execute_condition(
                        torch=torch,
                        processor=processor,
                        model=model,
                        runtime=runtime,
                        example=example,
                        condition=condition,
                        pair_index=pair_index,
                        scored=True,
                        data_root=data_root,
                        run_id=run_id,
                        contract_digest=contract_digest,
                        instruction=instruction,
                        instruction_sha256=instruction_sha256,
                        profile=profile,
                        model_load_seconds=model_load_seconds,
                    )
                    scored_rows.append(record)
                    _write_predictions(paths["predictions.jsonl"], scored_rows)

    _write_json(paths["canary.json"], canary)
    _write_predictions(paths["predictions.jsonl"], scored_rows)
    canary_passed = _condition_passed(canary, profile)
    scored_passed = sum(_condition_passed(record, profile) for record in scored_rows)
    expected_order = _expected_scored_order(contract, profile)
    order_passed = _actual_scored_order(scored_rows) == expected_order
    succeeded = (
        canary_passed
        and len(scored_rows) == profile.expected_scored_rows
        and scored_passed == profile.expected_scored_rows
        and order_passed
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "contract_sha256": contract_digest,
        "instruction_profile": profile.instruction_profile,
        "instruction_sha256": instruction_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "canary_passed": canary_passed,
        "scored_rows": len(scored_rows),
        "scored_passed": scored_passed,
        "expected_order_passed": order_passed,
        "predictions_sha256": _sha256_file(paths["predictions.jsonl"]),
        "succeeded": succeeded,
    }
    _write_json(paths["summary.json"], summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "inputs": {
            "contract_sha256": contract_digest,
            "instruction_profile": profile.instruction_profile,
            "instruction_sha256": instruction_sha256,
        },
        "sources": {
            name: _sha256_file(_SOURCE_DIRECTORY / name) for name in _SOURCE_FILES
        },
        "outputs": {
            name: _sha256_file(paths[name])
            for name in OUTPUT_NAMES
            if name != "manifest.json"
        },
    }
    _write_json(paths["manifest.json"], manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", choices=tuple(PROFILES), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    profile = profile_for_key(args.experiment)
    if args.preflight:
        if args.output_dir is not None:
            parser.error("--output-dir is forbidden with --preflight")
        preflight = preflight_frozen30_probe(
            contract_path=args.contract,
            data_root=args.data_root,
            run_id=args.run_id,
            profile=profile,
        )
        print(json.dumps(preflight, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --preflight is used")
    summary = run_frozen30_probe(
        contract_path=args.contract,
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        profile=profile,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["succeeded"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
