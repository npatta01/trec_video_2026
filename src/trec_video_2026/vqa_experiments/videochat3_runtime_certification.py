"""Fail-closed local contracts for VideoChat3 runtime certification."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from trec_video_2026.vqa_experiments.videochat3_pilot import (
    prepare_videochat3_inputs,
)


Backend = Literal["cpu", "rocm", "nvidia-cuda"]

_LANE_BACKENDS: dict[str, Backend] = {
    "cpu-smoke": "cpu",
    "rocm-smoke": "rocm",
    "cuda-certification": "nvidia-cuda",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CERTIFICATE_PIN_KEYS = frozenset(
    {
        "runtime_image",
        "requirements_sha256",
        "model_id",
        "model_revision",
        "transformers_revision",
        "source_snapshot_sha256",
    }
)
_RUNTIME_ATTESTATION_KEYS = frozenset(
    {"backend", "precision", "device", "torch_version", "transformers_version"}
)
_LOCATION_KEYS = frozenset({"provider", "region", "gpu", "cost_usd"})
_OUTCOME_KEYS = frozenset(
    {
        "input_shape",
        "timestamps",
        "decoding",
        "completed",
        "answer_length",
        "seed",
        "remote_detail_sha256",
    }
)
_FORBIDDEN_KEY_PARTS = ("token", "secret", "answer", "prompt", "media", "row", "reference")
_ALLOWED_SENSITIVE_SUBSTRING_KEYS = frozenset({"answer_length", "max_new_tokens"})
_EXPECTED_STATIC_PINS = {
    "runtime_image": (
        "docker.io/pytorch/pytorch@sha256:"
        "b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
    ),
    "requirements_sha256": (
        "9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6"
    ),
    "model_id": "MCG-NJU/VideoChat3-4B",
    "model_revision": "37fa901ec5913f84bc31108ebc1e60ad1903634c",
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
}
_SYNTHETIC_SHAPE = (12, 3, 32, 32)
_SYNTHETIC_CHECKSUM = 4700160


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{name} keys must be exactly {sorted(expected)}")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_static_pins(value: Mapping[str, Any], name: str) -> None:
    for key, expected_value in _EXPECTED_STATIC_PINS.items():
        if value[key] != expected_value:
            raise ValueError(f"{name} {key} pin mismatch")


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_cuda_runtime_metadata(
    attestation: Mapping[str, Any], location: Mapping[str, Any]
) -> None:
    if attestation["device"] != "cuda:0":
        raise ValueError("device must be cuda:0")
    for key in ("torch_version", "transformers_version"):
        _require_nonempty_string(attestation[key], key)
    for key in ("provider", "region", "gpu"):
        _require_nonempty_string(location[key], key)
    cost = location["cost_usd"]
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("cost_usd must be finite and non-negative")


def _reject_forbidden_keys(value: Any, *, location: str = "certificate") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text not in _ALLOWED_SENSITIVE_SUBSTRING_KEYS and any(
                part in key_text for part in _FORBIDDEN_KEY_PARTS
            ):
                raise ValueError(f"forbidden certificate key at {location}.{key}")
            _reject_forbidden_keys(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, location=f"{location}[{index}]")


def classify_torch_backend(torch_module: Any) -> Backend:
    """Identify HIP before interpreting PyTorch's shared CUDA namespace."""

    version = torch_module.version
    if version.hip:
        return "rocm"
    if version.cuda and torch_module.cuda.is_available():
        return "nvidia-cuda"
    return "cpu"


def validate_lane(lane: str, backend: str) -> None:
    """Require an attestation to match its explicitly requested lane."""

    expected_backend = _LANE_BACKENDS.get(lane)
    if expected_backend is None:
        raise ValueError(f"unknown lane: {lane}")
    if backend != expected_backend:
        raise ValueError(
            f"lane {lane} requires {expected_backend}; observed backend: {backend}"
        )


def synthetic_timestamps() -> tuple[float, ...]:
    """Return the fixed chronological timestamps for the synthetic video."""

    return (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)


def build_synthetic_video(torch_module: Any) -> Any:
    """Build a deterministic twelve-frame uint8 video without external files."""

    return torch_module.arange(12 * 3 * 32 * 32, dtype=torch_module.uint8).reshape(
        *_SYNTHETIC_SHAPE
    )


def _validate_synthetic_video(video: Any, torch_module: Any) -> int:
    if tuple(video.shape) != _SYNTHETIC_SHAPE:
        raise ValueError("synthetic video shape mismatch")
    if video.dtype != torch_module.uint8:
        raise ValueError("synthetic video dtype mismatch")
    checksum = int(video.sum().item())
    if checksum != _SYNTHETIC_CHECKSUM:
        raise ValueError("synthetic video checksum mismatch")
    return checksum


class _SyntheticMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _SyntheticInputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=type("InputIds", (), {"shape": (1, 1)})())
        self.moves: list[Any] = []

    def to(self, value: Any) -> "_SyntheticInputs":
        self.moves.append(value)
        return self


class _SyntheticVideoKwargs:
    __annotations__ = {
        "do_convert_rgb": bool | None,
        "do_resize": bool | None,
        "size": dict[str, int] | None,
    }


class _SyntheticVideoProcessor:
    valid_kwargs = _SyntheticVideoKwargs
    do_convert_rgb = True
    do_resize = True
    size = {"shortest_edge": 32, "longest_edge": 1024}


class _StrictSyntheticProcessor:
    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self.inputs: _SyntheticInputs | None = None
        self.video_processor = _SyntheticVideoProcessor()

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        if kwargs != {"tokenize": False, "add_generation_prompt": True}:
            raise ValueError("synthetic processor chat template mismatch")
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError("synthetic processor requires one message")
        return "synthetic-videochat3-prompt"

    def __call__(self, **kwargs: Any) -> _SyntheticInputs:
        video = kwargs.get("videos", [None])[0]
        metadata = kwargs.get("video_metadata", [None])[0]
        if kwargs.get("text") != "synthetic-videochat3-prompt":
            raise ValueError("synthetic processor prompt mismatch")
        if kwargs.get("do_sample_frames") is not False:
            raise ValueError("synthetic processor must not resample frames")
        if kwargs.get("do_convert_rgb") is not True or kwargs.get("do_resize") is not True:
            raise ValueError("synthetic processor defaults were not forwarded")
        if kwargs.get("return_metadata") is not True or kwargs.get("return_tensors") != "pt":
            raise ValueError("synthetic processor output contract mismatch")
        _validate_synthetic_video(video, self._torch)
        if (
            metadata.frames_indices != list(range(12))
            or metadata.timestamps != list(synthetic_timestamps())
        ):
            raise ValueError("synthetic processor metadata mismatch")
        self.inputs = _SyntheticInputs()
        return self.inputs


def _display_dtype(value: Any) -> str:
    return str(value).removeprefix("torch.")


def run_smoke(lane: str, torch_module: Any) -> dict[str, Any]:
    """Exercise synthetic tensor and adapter seams without model weights."""

    backend = classify_torch_backend(torch_module)
    validate_lane(lane, backend)
    video = build_synthetic_video(torch_module)
    checksum = _validate_synthetic_video(video, torch_module)
    processor = _StrictSyntheticProcessor(torch_module)
    device = "cpu" if backend == "cpu" else "cuda:0"
    dtype = torch_module.float32 if backend == "cpu" else torch_module.bfloat16
    inputs, _ = prepare_videochat3_inputs(
        processor=processor,
        metadata_base=_SyntheticMetadata,
        question="What is in the synthetic video?",
        video_tensor=video,
        timestamps=synthetic_timestamps(),
        width=32,
        height=32,
        model_device=device,
        model_dtype=dtype,
    )
    record: dict[str, Any] = {
        "lane": lane,
        "backend": backend,
        "synthetic_shape": list(_SYNTHETIC_SHAPE),
        "synthetic_checksum": checksum,
        "timestamps": list(synthetic_timestamps()),
        "preprocessing_moves": [_display_dtype(value) for value in inputs.moves],
    }
    if backend == "rocm" and torch_module.cuda.is_bf16_supported():
        probe = video.to("cuda:0", dtype=torch_module.bfloat16)
        doubled = probe + probe
        if tuple(doubled.shape) != _SYNTHETIC_SHAPE:
            raise ValueError("ROCm bfloat16 arithmetic shape mismatch")
        if str(doubled.device) != "cuda:0":
            raise ValueError("ROCm bfloat16 arithmetic device mismatch")
        if doubled.dtype != torch_module.bfloat16:
            raise ValueError("ROCm bfloat16 arithmetic dtype mismatch")
        if not bool(torch_module.isfinite(doubled).all().item()):
            raise ValueError("ROCm bfloat16 arithmetic is not finite")
        if int(doubled.float().sum().item()) != checksum * 2:
            raise ValueError("ROCm bfloat16 arithmetic mismatch")
        record["bf16_arithmetic"] = "passed"
    return record


def build_certificate(
    runtime: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a sanitized, hash-bound NVIDIA certification record."""

    _reject_forbidden_keys(runtime, location="runtime")
    _reject_forbidden_keys(outcome, location="outcome")
    _require_exact_keys(runtime, _CERTIFICATE_PIN_KEYS | _RUNTIME_ATTESTATION_KEYS | _LOCATION_KEYS, "runtime")
    _require_exact_keys(outcome, _OUTCOME_KEYS, "outcome")
    _validate_static_pins(runtime, "runtime")
    if runtime["backend"] != "nvidia-cuda":
        raise ValueError(f"certificate observed backend: {runtime['backend']}")
    if runtime["precision"] != "bfloat16":
        raise ValueError("certificate precision must be bfloat16")
    _validate_cuda_runtime_metadata(runtime, runtime)
    _require_sha256(runtime["requirements_sha256"], "requirements_sha256")
    _require_sha256(runtime["source_snapshot_sha256"], "source_snapshot_sha256")
    _require_sha256(outcome["remote_detail_sha256"], "remote_detail_sha256")
    if outcome["completed"] is not True:
        raise ValueError("outcome completed must be true")
    if not isinstance(outcome["answer_length"], int) or outcome["answer_length"] <= 0:
        raise ValueError("outcome answer_length must be a positive integer")
    if outcome["seed"] != 20260724:
        raise ValueError("outcome seed must be 20260724")

    return {
        "schema_version": "videochat3-runtime-certification-v1",
        "pins": {key: runtime[key] for key in sorted(_CERTIFICATE_PIN_KEYS)},
        "attestation": {
            key: runtime[key] for key in sorted(_RUNTIME_ATTESTATION_KEYS)
        },
        "location": {key: runtime[key] for key in sorted(_LOCATION_KEYS)},
        "synthetic_input": {
            "shape": list(outcome["input_shape"]),
            "timestamps": list(outcome["timestamps"]),
            "decoding": dict(_require_mapping(outcome["decoding"], "decoding")),
            "seed": outcome["seed"],
        },
        "outcome": {
            "completed": outcome["completed"],
            "answer_length": outcome["answer_length"],
            "remote_detail_sha256": outcome["remote_detail_sha256"],
        },
    }


def validate_certificate(
    certificate: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    """Fail closed unless a sanitized, complete NVIDIA certificate matches pins."""

    _reject_forbidden_keys(certificate)
    _require_exact_keys(
        certificate,
        frozenset({"schema_version", "pins", "attestation", "location", "synthetic_input", "outcome"}),
        "certificate",
    )
    if certificate["schema_version"] != "videochat3-runtime-certification-v1":
        raise ValueError("unsupported certificate schema_version")
    pins = _require_mapping(certificate["pins"], "pins")
    attestation = _require_mapping(certificate["attestation"], "attestation")
    location = _require_mapping(certificate["location"], "location")
    synthetic_input = _require_mapping(certificate["synthetic_input"], "synthetic_input")
    outcome = _require_mapping(certificate["outcome"], "outcome")
    _require_exact_keys(pins, _CERTIFICATE_PIN_KEYS, "pins")
    _require_exact_keys(attestation, _RUNTIME_ATTESTATION_KEYS, "attestation")
    _require_exact_keys(location, _LOCATION_KEYS, "location")
    _require_exact_keys(synthetic_input, frozenset({"shape", "timestamps", "decoding", "seed"}), "synthetic_input")
    _require_exact_keys(outcome, frozenset({"completed", "answer_length", "remote_detail_sha256"}), "outcome")
    _require_sha256(pins["requirements_sha256"], "requirements_sha256")
    _require_sha256(pins["source_snapshot_sha256"], "source_snapshot_sha256")
    _require_sha256(outcome["remote_detail_sha256"], "remote_detail_sha256")
    _validate_static_pins(pins, "certificate")
    _require_exact_keys(expected, _CERTIFICATE_PIN_KEYS, "expected")
    _validate_static_pins(expected, "expected")
    _require_sha256(expected["source_snapshot_sha256"], "expected source_snapshot_sha256")
    if attestation["backend"] != "nvidia-cuda":
        raise ValueError(f"certificate observed backend: {attestation['backend']}")
    if attestation["precision"] != "bfloat16":
        raise ValueError("certificate precision must be bfloat16")
    _validate_cuda_runtime_metadata(attestation, location)
    if outcome["completed"] is not True:
        raise ValueError("certificate outcome completed must be true")
    if not isinstance(outcome["answer_length"], int) or outcome["answer_length"] <= 0:
        raise ValueError("certificate outcome answer_length must be a positive integer")
    if synthetic_input["shape"] != [12, 3, 32, 32]:
        raise ValueError("certificate synthetic input shape mismatch")
    if synthetic_input["timestamps"] != list(synthetic_timestamps()):
        raise ValueError("certificate synthetic timestamps mismatch")
    decoding = _require_mapping(synthetic_input["decoding"], "decoding")
    if decoding != {"do_sample": False, "max_new_tokens": 32}:
        raise ValueError("certificate decoding settings mismatch")
    if synthetic_input["seed"] != 20260724:
        raise ValueError("certificate synthetic seed mismatch")
    for key, value in expected.items():
        if pins[key] != value:
            raise ValueError(f"certificate {key} pin mismatch")


def main(argv: list[str] | None = None, *, torch_module: Any | None = None) -> int:
    """Run one explicit local runtime attestation lane."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lane", choices=("cpu-smoke", "rocm-smoke", "cuda-certification")
    )
    args = parser.parse_args(argv)
    if torch_module is None:
        import torch

        torch_module = torch
    print(json.dumps(run_smoke(args.lane, torch_module), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
