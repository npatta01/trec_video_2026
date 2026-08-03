from __future__ import annotations

import os
import hashlib
import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.videochat3_runtime_certification import (
    build_certificate,
    build_synthetic_video,
    classify_torch_backend,
    main,
    run_smoke,
    synthetic_timestamps,
    validate_lane,
    validate_certificate,
)


def _fake_torch(*, hip: str | None, cuda: str | None, available: bool) -> object:
    """Supply the complete Torch version and CUDA surface this contract reads."""

    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip=hip, cuda=cuda),
        cuda=types.SimpleNamespace(is_available=lambda: available),
    )


@pytest.mark.parametrize(
    ("torch_module", "expected"),
    [
        (_fake_torch(hip="6.4", cuda="12.8", available=True), "rocm"),
        (_fake_torch(hip=None, cuda="12.8", available=True), "nvidia-cuda"),
        (_fake_torch(hip=None, cuda=None, available=False), "cpu"),
    ],
)
def test_classify_torch_backend_identifies_the_observed_runtime(
    torch_module: object, expected: str
) -> None:
    """Catches HIP being mistaken for NVIDIA CUDA or CPU."""

    assert classify_torch_backend(torch_module) == expected


@pytest.mark.parametrize(
    ("lane", "backend"),
    [
        ("cpu-smoke", "cpu"),
        ("rocm-smoke", "rocm"),
        ("cuda-certification", "nvidia-cuda"),
    ],
)
def test_validate_lane_accepts_only_its_declared_backend(
    lane: str, backend: str
) -> None:
    """Catches a lane accepting a backend other than the one it attests."""

    validate_lane(lane, backend)


@pytest.mark.parametrize("observed", ("cpu", "rocm"))
def test_cuda_certification_rejects_non_nvidia_backends_by_name(observed: str) -> None:
    """Catches the CUDA gate silently accepting CPU or HIP fallback."""

    with pytest.raises(ValueError, match=observed):
        validate_lane("cuda-certification", observed)


def test_validate_lane_rejects_unknown_lane() -> None:
    """Catches a typo creating an un-attested runtime lane."""

    with pytest.raises(ValueError, match="unknown lane"):
        validate_lane("unrecognized", "cpu")


@pytest.mark.parametrize(
    "lane",
    [
        "cpu-smoke",
        "rocm-smoke",
        "cuda-certification",
    ],
)
def test_cli_accepts_each_declared_lane_with_its_observed_backend(
    lane: str,
) -> None:
    """Catches a CLI mode bypassing the same backend-to-lane validation."""

    torch_module = {
        "cpu-smoke": _SmokeTorch(
            hip=None, cuda=None, available=False, bf16_supported=False
        ),
        "rocm-smoke": _SmokeTorch(
            hip="6.4", cuda="12.8", available=True, bf16_supported=True
        ),
        "cuda-certification": _SmokeTorch(
            hip=None, cuda="12.8", available=True, bf16_supported=True
        ),
    }[lane]
    assert main([lane], torch_module=torch_module) == 0


class _Tensor:
    def __init__(self, values: list[int], dtype: object) -> None:
        self.values = values
        self.dtype = dtype
        self.shape: tuple[int, ...] = ()
        self.device = "cpu"
        self.to_calls: list[tuple[object, object | None]] = []
        self.add_count = 0

    def reshape(self, *shape: int) -> "_Tensor":
        self.shape = shape
        return self

    def __getitem__(self, index: tuple[int, int, int, int]) -> int:
        frame, channel, height, width = index
        offset = (((frame * 3 + channel) * 32 + height) * 32) + width
        return self.values[offset]

    def sum(self) -> object:
        return types.SimpleNamespace(item=lambda: sum(self.values))

    def to(self, device: object, dtype: object | None = None) -> "_Tensor":
        self.to_calls.append((device, dtype))
        self.device = str(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __add__(self, other: object) -> "_Tensor":
        assert other is self
        self.add_count += 1
        doubled = _Tensor([value * 2 for value in self.values], self.dtype)
        doubled.shape = self.shape
        doubled.device = self.device
        return doubled

    def float(self) -> "_Tensor":
        converted = _Tensor(self.values, "float32")
        converted.shape = self.shape
        converted.device = self.device
        return converted


class _SyntheticTorch:
    uint8 = "uint8"

    @staticmethod
    def arange(length: int, *, dtype: object) -> _Tensor:
        return _Tensor([index % 256 for index in range(length)], dtype)


class _SmokeCuda:
    def __init__(self, *, available: bool, bf16_supported: bool) -> None:
        self.available = available
        self.bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16_supported


class _SmokeTorch(_SyntheticTorch):
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(
        self, *, hip: str | None, cuda: str | None, available: bool, bf16_supported: bool
    ) -> None:
        self.version = types.SimpleNamespace(hip=hip, cuda=cuda)
        self.cuda = _SmokeCuda(
            available=available, bf16_supported=bf16_supported
        )
        self.last_tensor: _Tensor | None = None

    def arange(self, length: int, *, dtype: object) -> _Tensor:
        self.last_tensor = _Tensor([index % 256 for index in range(length)], dtype)
        return self.last_tensor

    @staticmethod
    def isfinite(value: object) -> object:
        return types.SimpleNamespace(
            all=lambda: types.SimpleNamespace(item=lambda: True)
        )


class _RoundedBfloat16Tensor(_Tensor):
    device = "cuda:0"

    def sum(self) -> object:
        return types.SimpleNamespace(item=lambda: 9371648.0)

    def float(self) -> _Tensor:
        exact = _Tensor([value * 2 for value in self.values], "float32")
        exact.shape = self.shape
        exact.device = "cuda:0"  # type: ignore[attr-defined]
        return exact


class _RoundedReductionTensor(_Tensor):
    device = "cpu"

    def to(self, device: object, dtype: object | None = None) -> "_RoundedReductionTensor":
        super().to(device, dtype)
        self.device = str(device)
        return self

    def __add__(self, other: object) -> _RoundedBfloat16Tensor:
        assert other is self
        doubled = _RoundedBfloat16Tensor(self.values, self.dtype)
        doubled.shape = self.shape
        doubled.device = self.device
        return doubled


class _RoundedReductionTorch(_SmokeTorch):
    def arange(self, length: int, *, dtype: object) -> _RoundedReductionTensor:
        self.last_tensor = _RoundedReductionTensor(
            [index % 256 for index in range(length)], dtype
        )
        return self.last_tensor


def _runtime() -> dict[str, object]:
    return {
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
        "source_snapshot_sha256": "c" * 64,
        "backend": "nvidia-cuda",
        "precision": "bfloat16",
        "device": "cuda:0",
        "torch_version": "2.10.0",
        "transformers_version": "5.0.0.dev0",
        "provider": "runpod",
        "region": "us-east",
        "gpu": "A40",
        "cost_usd": 0.25,
    }


def _expected_pins() -> dict[str, str]:
    return {
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
        "source_snapshot_sha256": "c" * 64,
    }


def _outcome() -> dict[str, object]:
    return {
        "input_shape": (12, 3, 32, 32),
        "timestamps": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0),
        "decoding": {"do_sample": False, "max_new_tokens": 32},
        "completed": True,
        "answer_length": 6,
        "seed": 20260724,
        "remote_detail_sha256": "d" * 64,
    }


def test_synthetic_video_uses_twelve_deterministic_byte_frames() -> None:
    """Catches a synthetic input changing shape, dtype, or frame values."""

    video = build_synthetic_video(_SyntheticTorch())

    assert video.shape == (12, 3, 32, 32)
    assert video.dtype == "uint8"
    assert video[0, 0, 0, 0] == 0
    assert video[0, 0, 0, 31] == 31
    assert video[0, 0, 1, 0] == 32
    assert video[11, 2, 31, 31] == 255


def test_synthetic_timestamps_are_the_literal_chronological_seconds() -> None:
    """Catches timestamps drifting from the contract's twelve exact seconds."""

    assert synthetic_timestamps() == (
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
    )


def test_cpu_smoke_exercises_the_synthetic_video_preprocessing_seam() -> None:
    """Catches the CPU lane merely classifying Torch without testing the adapter."""

    torch_module = _SmokeTorch(
        hip=None, cuda=None, available=False, bf16_supported=False
    )

    record = run_smoke("cpu-smoke", torch_module)

    assert record["backend"] == "cpu"
    assert record["synthetic_checksum"] == 4700160
    assert record["synthetic_shape"] == [12, 3, 32, 32]
    assert record["timestamps"] == list(synthetic_timestamps())
    assert record["preprocessing_moves"] == ["cpu", "float32"]


def test_rocm_smoke_places_synthetic_frames_and_exercises_bfloat16_math() -> None:
    """Catches the HIP lane skipping available GPU BF16 tensor work."""

    torch_module = _SmokeTorch(
        hip="6.4", cuda="12.8", available=True, bf16_supported=True
    )

    record = run_smoke("rocm-smoke", torch_module)

    assert record["backend"] == "rocm"
    assert record["bf16_arithmetic"] == "passed"
    assert torch_module.last_tensor is not None
    assert torch_module.last_tensor.to_calls == [("cuda:0", "bfloat16")]


def test_rocm_smoke_uses_an_exact_float32_reduction_after_bfloat16_math() -> None:
    """Catches valid BF16 tensor arithmetic being rejected by rounded reduction."""

    torch_module = _RoundedReductionTorch(
        hip="6.4", cuda="12.8", available=True, bf16_supported=True
    )

    record = run_smoke("rocm-smoke", torch_module)

    assert record["bf16_arithmetic"] == "passed"


def test_certificate_is_sanitized_and_validates_matching_pins() -> None:
    """Catches a certificate leaking raw output or losing its bound pins."""

    certificate = build_certificate(_runtime(), _outcome())

    assert "raw_answer" not in str(certificate).lower()
    assert "secret" not in str(certificate).lower()
    assert certificate["outcome"]["answer_length"] == 6
    assert certificate["attestation"]["precision"] == "bfloat16"
    assert certificate["synthetic_input"]["seed"] == 20260724
    validate_certificate(certificate, _expected_pins())


@pytest.mark.parametrize(
    ("runtime_key", "changed_value", "message"),
    (
        ("device", "cpu", "device must be cuda:0"),
        ("torch_version", "", "torch_version must be a non-empty string"),
        (
            "transformers_version",
            " ",
            "transformers_version must be a non-empty string",
        ),
        ("provider", "", "provider must be a non-empty string"),
        ("region", " ", "region must be a non-empty string"),
        ("gpu", "", "gpu must be a non-empty string"),
        ("cost_usd", -0.01, "cost_usd must be finite and non-negative"),
        ("cost_usd", float("nan"), "cost_usd must be finite and non-negative"),
        ("cost_usd", False, "cost_usd must be finite and non-negative"),
    ),
)
def test_certificate_builder_rejects_contradictory_cuda_runtime_metadata(
    runtime_key: str, changed_value: object, message: str
) -> None:
    """Catches a CPU or malformed location being certified as an NVIDIA run."""

    runtime = _runtime()
    runtime[runtime_key] = changed_value

    with pytest.raises(ValueError, match=message):
        build_certificate(runtime, _outcome())


@pytest.mark.parametrize(
    ("section", "key", "changed_value", "message"),
    (
        ("attestation", "device", "cpu", "device must be cuda:0"),
        ("attestation", "torch_version", "", "torch_version must be a non-empty string"),
        ("location", "provider", "", "provider must be a non-empty string"),
        ("location", "region", " ", "region must be a non-empty string"),
        ("location", "gpu", "", "gpu must be a non-empty string"),
        ("location", "cost_usd", -1, "cost_usd must be finite and non-negative"),
    ),
)
def test_certificate_validator_rejects_tampered_cuda_runtime_metadata(
    section: str, key: str, changed_value: object, message: str
) -> None:
    """Catches a valid certificate being changed into a contradictory record."""

    certificate = build_certificate(_runtime(), _outcome())
    certificate[section][key] = changed_value

    with pytest.raises(ValueError, match=message):
        validate_certificate(certificate, _expected_pins())


@pytest.mark.parametrize("digest_key", ("requirements_sha256", "source_snapshot_sha256"))
def test_certificate_rejects_non_lowercase_sha256_pins(digest_key: str) -> None:
    """Catches malformed or uppercase digests being treated as immutable pins."""

    runtime = _runtime()
    runtime[digest_key] = "A" * 64

    with pytest.raises(ValueError, match=digest_key):
        build_certificate(runtime, _outcome())


def test_certificate_rejects_incomplete_outcome() -> None:
    """Catches a failed or incomplete synthetic run being certified as complete."""

    outcome = _outcome()
    outcome["completed"] = False

    with pytest.raises(ValueError, match="completed"):
        build_certificate(_runtime(), outcome)


def test_certificate_rejects_pin_and_source_snapshot_mismatches() -> None:
    """Catches certificates being reused after any bound input changes."""

    certificate = build_certificate(_runtime(), _outcome())

    with pytest.raises(ValueError, match="requirements_sha256"):
        validate_certificate(
            certificate, {**_expected_pins(), "requirements_sha256": "e" * 64}
        )
    with pytest.raises(ValueError, match="source_snapshot_sha256"):
        validate_certificate(
            certificate,
            {**_expected_pins(), "source_snapshot_sha256": "e" * 64},
        )


@pytest.mark.parametrize(
    ("runtime_key", "changed_value"),
    [
        ("runtime_image", "docker.io/pytorch/pytorch@sha256:" + "f" * 64),
        ("model_id", "MCG-NJU/VideoChat3-8B"),
        ("model_revision", "f" * 40),
        ("transformers_revision", "f" * 40),
    ],
)
def test_certificate_rejects_every_static_pin_drift(
    runtime_key: str, changed_value: str
) -> None:
    """Catches a certificate claiming an unapproved model or runtime pin."""

    runtime = _runtime()
    runtime[runtime_key] = changed_value

    with pytest.raises(ValueError, match=runtime_key):
        build_certificate(runtime, _outcome())


def test_certificate_requires_a_complete_expected_pin_mapping() -> None:
    """Catches a downstream verifier validating a certificate with partial pins."""

    certificate = build_certificate(_runtime(), _outcome())

    with pytest.raises(ValueError, match="expected"):
        validate_certificate(certificate, {})


def test_certificate_rejects_non_bfloat16_nvidia_attestation() -> None:
    """Catches a CUDA FP32 result being accepted as NVIDIA BF16 certification."""

    certificate = build_certificate(_runtime(), _outcome())
    certificate["attestation"]["precision"] = "float32"

    with pytest.raises(ValueError, match="precision"):
        validate_certificate(certificate, _expected_pins())


def test_certificate_builder_rejects_non_bfloat16_nvidia_attestation() -> None:
    """Catches the builder emitting a reusable CUDA FP32 certificate."""

    runtime = _runtime()
    runtime["precision"] = "float32"

    with pytest.raises(ValueError, match="precision"):
        build_certificate(runtime, _outcome())


def test_certificate_rejects_raw_answers_and_unknown_expected_pins() -> None:
    """Catches secret-bearing records and unchecked expected inputs."""

    outcome = _outcome()
    outcome["raw_answer"] = "private model text"

    with pytest.raises(ValueError, match="forbidden"):
        build_certificate(_runtime(), outcome)
    with pytest.raises(ValueError, match="expected"):
        validate_certificate(
            build_certificate(_runtime(), _outcome()),
            {**_expected_pins(), "unknown": "x"},
        )


RUNTIME_SMOKE_SCRIPT = Path("scripts/run-videochat3-runtime-smoke.sh")
CUDA_SMOKE_BOOTSTRAP = Path("scripts/bootstrap-videochat3-cuda-smoke.sh")
CERTIFICATE_VERIFIER = Path("scripts/verify-videochat3-runtime-certificate.py")
CUDA_IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)
CUDA_SMOKE_IMAGE = "localhost/trec-videochat3-cuda-smoke:runtime-certification"
ROCM_IMAGE = (
    "docker.io/rocm/pytorch@"
    "sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e"
)


def _fake_podman_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Provide an external-process boundary that records Podman invocations."""

    executable = tmp_path / "podman"
    log = tmp_path / "podman.log"
    executable.write_text(
        """#!/usr/bin/env bash
printf '%s ' "$@" >>"$PODMAN_LOG"
printf '\\n' >>"$PODMAN_LOG"
if [[ "$1" == "image" && "$2" == "exists" ]]; then
  exit "${PODMAN_IMAGE_EXISTS:-0}"
fi
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PODMAN_LOG": str(log),
    }
    return environment, log


def test_cuda_smoke_bootstrap_marks_every_system_pip_mutation_as_pep668_safe(
    tmp_path: Path,
) -> None:
    """Catches a base-image PEP 668 failure before the smoke image can build."""

    fake_python = tmp_path / "python"
    command_log = tmp_path / "python.log"
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.10.0\n", encoding="utf-8")
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$FAKE_PYTHON_LOG"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(CUDA_SMOKE_BOOTSTRAP), str(requirements)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "FAKE_PYTHON_LOG": str(command_log),
        },
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands == [
        "-m pip uninstall --break-system-packages --yes spin",
        "-m pip install --break-system-packages --no-cache-dir --no-deps uv==0.11.3",
        f"-m uv pip install --system --break-system-packages --no-managed-python --requirements {requirements}",
        "-m uv pip check --system",
    ]


def test_podman_cpu_smoke_uses_the_exact_cuda_image_without_devices(
    tmp_path: Path,
) -> None:
    """Catches the required CPU lane drifting from its pinned CUDA image."""

    environment, log = _fake_podman_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(RUNTIME_SMOKE_SCRIPT), "cpu"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    command = log.read_text(encoding="utf-8")
    assert "build" in command
    assert "containers/videochat3-cuda-smoke.Containerfile" in command
    assert CUDA_IMAGE in command
    assert CUDA_SMOKE_IMAGE in command
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "/dev/kfd" not in command and "/dev/dri" not in command
    assert "/workspace/src:ro" in command
    assert "/data" not in command


def test_podman_rocm_smoke_uses_cached_image_and_advisory_devices(
    tmp_path: Path,
) -> None:
    """Catches the advisory HIP lane omitting its cached runtime or GPU devices."""

    environment, log = _fake_podman_environment(tmp_path)
    device_root = tmp_path / "controlled-devices"
    (device_root / "dev").mkdir(parents=True)
    (device_root / "dev" / "kfd").touch()
    (device_root / "dev" / "dri").touch()
    environment["ROCM_DEVICE_CHECK_ROOT"] = str(device_root)
    result = subprocess.run(
        ["bash", str(RUNTIME_SMOKE_SCRIPT), "rocm"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    command = log.read_text(encoding="utf-8")
    assert ROCM_IMAGE in command
    assert "--pull=never" in command
    assert "--device=/dev/kfd" in command
    assert "--device=/dev/dri" in command


def test_podman_rocm_reports_an_advisory_skip_when_controlled_devices_are_missing(
    tmp_path: Path,
) -> None:
    """Catches an unavailable ROCm device being treated as a required failure."""

    environment, _ = _fake_podman_environment(tmp_path)
    environment["ROCM_DEVICE_CHECK_ROOT"] = str(tmp_path / "no-devices")
    result = subprocess.run(
        ["bash", str(RUNTIME_SMOKE_SCRIPT), "rocm"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "advisory skip" in result.stderr.lower()


def test_podman_cpu_smoke_commands_never_request_a_model_download(
    tmp_path: Path,
) -> None:
    """Catches the local smoke artifacts adding a model-weight download operation."""

    environment, log = _fake_podman_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(RUNTIME_SMOKE_SCRIPT), "cpu"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "download" not in log.read_text(encoding="utf-8").lower()


def _source_snapshot(source_root: Path) -> str:
    """Hand-compute the verifier's content-addressed source fixture."""

    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_certificate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create a complete hash-bound certificate without external runtime inputs."""

    source_root = tmp_path / "source"
    shutil.copytree("src/trec_video_2026", source_root)
    requirements = Path("configs/vqa/videochat3_cuda_requirements.txt")
    runtime = _runtime()
    runtime["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
    runtime["source_snapshot_sha256"] = _source_snapshot(source_root)
    certificate = tmp_path / "certificate.json"
    certificate.write_text(
        json.dumps(build_certificate(runtime, _outcome())), encoding="utf-8"
    )
    return certificate, source_root, requirements


def _run_certificate_cli(
    certificate: Path, source_root: Path, requirements: Path
) -> subprocess.CompletedProcess[str]:
    """Invoke the real fail-closed CLI over a filesystem fixture."""

    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(CERTIFICATE_VERIFIER),
            "--certificate",
            str(certificate),
            "--source-root",
            str(source_root),
            "--requirements",
            str(requirements),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_certificate_cli_fails_closed_for_a_missing_certificate(tmp_path: Path) -> None:
    """Catches an absent certificate being treated as an eligible runtime."""

    _, source_root, requirements = _runtime_certificate_fixture(tmp_path)

    result = _run_certificate_cli(tmp_path / "missing.json", source_root, requirements)

    assert result.returncode == 2
    assert "certificate not found" in result.stderr.lower()


def test_certificate_cli_fails_closed_for_malformed_certificate(tmp_path: Path) -> None:
    """Catches malformed certificate JSON being accepted or crashing open."""

    certificate, source_root, requirements = _runtime_certificate_fixture(tmp_path)
    certificate.write_text("{", encoding="utf-8")

    result = _run_certificate_cli(certificate, source_root, requirements)

    assert result.returncode == 2
    assert "certificate is not valid json" in result.stderr.lower()


def test_certificate_cli_fails_closed_for_a_hash_bound_mismatch(tmp_path: Path) -> None:
    """Catches a certificate being reused with a source snapshot it did not attest."""

    certificate, source_root, requirements = _runtime_certificate_fixture(tmp_path)
    source_file = source_root / "vqa_experiments" / "videochat3_runtime_certification.py"
    source_file.write_text(source_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = _run_certificate_cli(certificate, source_root, requirements)

    assert result.returncode == 2
    assert "source_snapshot_sha256" in result.stderr


def test_certificate_cli_accepts_a_complete_matching_nvidia_certificate(
    tmp_path: Path,
) -> None:
    """Catches the eligibility gate rejecting the complete synthetic CUDA certificate."""

    certificate, source_root, requirements = _runtime_certificate_fixture(tmp_path)

    result = _run_certificate_cli(certificate, source_root, requirements)

    assert result.returncode == 0, result.stderr
    assert "eligible" in result.stdout.lower()
