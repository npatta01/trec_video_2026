from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "dstack/qwen36-failure-panel.dstack.yml"
RUNTIME_PREFLIGHT_CONFIG = ROOT / "dstack/qwen36-runtime-preflight.dstack.yml"
WRAPPER = ROOT / "scripts/run_failure_panel_dstack.sh"
PANEL = ROOT / "configs/vqa/failure_panel_12_inference.json"
STAGED_REQUESTS = (
    ROOT / "data/vqa/benchmarks/followups/failure-panel-2026-07-24/"
    "dstack-input/requests.json"
)

EXPECTED_PANEL_SHA256 = (
    "d2c5515b8c7398f326931e2f993f3fd6eac8e70d9ef01d1a1bd4afcf3f6792e8"
)
EXPECTED_IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)
EXPECTED_RUNTIME_IMAGE = EXPECTED_IMAGE
EXPECTED_MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
EXPECTED_TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
EXPECTED_TRANSFORMERS_URL = (
    "https://github.com/huggingface/transformers/archive/"
    f"{EXPECTED_TRANSFORMERS_REVISION}.tar.gz"
)


def _label_free_requests(tmp_path: Path) -> Path:
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    evidence = tmp_path / "evidence"
    requests = []
    for example in panel["examples"]:
        frame_dir = evidence / example["video_id"]
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        frame_records = []
        for index in range(1, 13):
            frame = frame_dir / f"frame-{index:02d}.png"
            frame.write_bytes(f"{example['video_id']}:{index}".encode())
            frame_paths.append(str(frame))
            frame_records.append(
                {
                    "path": frame.name,
                    "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                }
            )
        (frame_dir / "evidence.json").write_text(
            json.dumps({"frames": frame_records}),
            encoding="utf-8",
        )
        requests.append(
            {
                "run_id": "qwen36-failure-panel-20260724-a10",
                "condition_id": "qwen36-35b-a3b-12f-failure-panel",
                "video_id": example["video_id"],
                "question": example["question"],
                "frame_paths": frame_paths,
                "frame_timestamps": [float(index) for index in range(12)],
                "frame_descriptions": None,
                "transcript": None,
            }
        )
    request_path = tmp_path / "requests.json"
    request_path.write_text(
        json.dumps(requests, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return request_path


def _run_wrapper(
    tmp_path: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    extra_env = extra_env or {}
    requests = (
        Path(extra_env["INPUT_REQUESTS"])
        if "INPUT_REQUESTS" in extra_env
        else (
            Path(extra_env["SOURCE_REQUESTS"])
            if "SOURCE_REQUESTS" in extra_env
            else _label_free_requests(tmp_path)
        )
    )
    fake_runtime = tmp_path / "fake-runtime"
    fake_runtime.mkdir(exist_ok=True)
    (fake_runtime / "torch.py").write_text("", encoding="utf-8")
    fake_transformers = fake_runtime / "transformers"
    fake_transformers.mkdir(exist_ok=True)
    (fake_transformers / "__init__.py").write_text(
        "__version__ = 'preflight-fake'\nAutoProcessor = object\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHON_BIN": str(Path(os.sys.executable).resolve()),
        "PYTHONPATH": f"{fake_runtime}:{ROOT / 'src'}",
        "INPUT_REQUESTS": str(requests),
        "PANEL_MANIFEST": str(PANEL),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "EXPERIMENT_RUN_NAME": "qwen36-failure-panel-20260724-a10",
        "EXPECTED_PANEL_SHA256": EXPECTED_PANEL_SHA256,
        "MODEL_ID": "Qwen/Qwen3.6-35B-A3B",
        "MODEL_REVISION": EXPECTED_MODEL_REVISION,
        "HF_REMOTE_ROOT": "hf://buckets/Npatta01/trec_mlm_2026/"
        "trec_video_2026/experiments",
        "EXPECTED_EVIDENCE_ROOT": str(tmp_path / "evidence"),
        **extra_env,
    }
    return subprocess.run(
        ["bash", str(WRAPPER), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_dstack_config_is_bounded_digest_pinned_and_label_free() -> None:
    text = CONFIG.read_text(encoding="utf-8")

    assert EXPECTED_IMAGE in text
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert (
        "HF_REMOTE_ROOT=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments"
    ) in text
    assert (
        "EVIDENCE_ARTIFACT_URI=hf://buckets/Npatta01/trec_mlm_2026/"
        "trec_video_2026/artifacts/failure-panel-12-frames-d2c5515b8c73"
    ) in text
    assert "../data/vqa/benchmarks/evidence/" not in text
    assert "backends: [runpod, vastai]" in text
    assert "gpu: RTXPRO6000WS,A100,H100,H200,B200:1:80GB.." in text
    assert "memory: 192GB.." in text
    assert f"VQA_RUNTIME_IMAGE={EXPECTED_RUNTIME_IMAGE}" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 4.00" in text
    assert "max_duration: 4h" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 30m" in text
    assert "training/videos_original" not in text
    assert "annotation" not in text.lower()
    assert "reference" not in text.lower()
    assert "vqa-training-one.dstack.yml" not in text
    assert "name: qwen36-failure-panel-20260724-a10" in text
    assert "EXPERIMENT_RUN_NAME=qwen36-failure-panel-20260724-a10" in text
    assert "qwen36-failure-panel-20260724-a9" not in text
    assert "qwen36-failure-panel-20260724-a8" not in text
    assert "qwen36-failure-panel-20260724-a7" not in text
    assert "qwen36-failure-panel-20260724-a6" not in text
    assert "qwen36-failure-panel-20260724-a5" not in text
    assert "qwen36-failure-panel-20260724-a4" not in text
    assert "docker.io/vllm/" not in text
    assert "PYTHON_BIN=/usr/bin/python3" in text
    assert "/usr/local/bin/python3" not in text


def test_dstack_renders_all_phases_as_one_bash_command_argument() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    entrypoint = next(
        line.removeprefix("entrypoint: ")
        for line in text.splitlines()
        if line.startswith("entrypoint: ")
    )
    commands_block = text.split("\ncommands:\n", 1)[1].split("\nresources:\n", 1)[0]
    assert sum(line.startswith("  - ") for line in commands_block.splitlines()) == 1
    assert commands_block.startswith("  - |\n")
    command = "\n".join(
        line.removeprefix("    ") for line in commands_block.splitlines()[1:]
    )
    rendered_job_argv = [*shlex.split(entrypoint), command]
    assert rendered_job_argv == ["/bin/bash", "-c", command]
    assert command.index("fetch-evidence") < command.index("--preflight")
    assert command.index("--preflight") < command.index(" run")
    assert "set -euo pipefail" in command


def test_runtime_bootstrap_is_first_exactly_pinned_and_capability_checked() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    commands_block = text.split("\ncommands:\n", 1)[1].split("\nresources:\n", 1)[0]
    command = "\n".join(
        line.removeprefix("    ") for line in commands_block.splitlines()[1:]
    )
    assert command.index("bootstrap-transformers") < command.index("fetch-evidence")
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert EXPECTED_TRANSFORMERS_REVISION in wrapper
    assert EXPECTED_TRANSFORMERS_URL in wrapper
    assert "uv pip install" in wrapper
    assert "--system" in wrapper
    assert "--no-managed-python" in wrapper
    assert "--no-python-downloads" in wrapper
    assert 'AutoConfig.for_model("qwen3_5_moe")' in wrapper
    assert '"5.15.0.dev0"' in wrapper
    assert "uv==0.11.3" in wrapper
    assert 'importlib.metadata.version("uv")' in wrapper
    assert "--break-system-packages" in wrapper
    assert "uv pip check" in wrapper
    assert "torch.cuda.is_available()" in wrapper
    assert "torch.cuda.is_bf16_supported()" in wrapper
    assert "cuda_device_name" in wrapper
    assert "AutoProcessor" in wrapper
    assert "AutoModelForImageTextToText" in wrapper
    requirements = (ROOT / "configs/vqa/qwen36_a4_runtime_requirements.txt").read_text(
        encoding="utf-8"
    )
    lock = (ROOT / "configs/vqa/qwen36_a4_runtime_lock.txt").read_text(encoding="utf-8")
    for requirement in (
        "huggingface-hub==1.24.0",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
        "accelerate==1.12.0",
        "av==16.0.1",
        "numpy==2.4.6",
        "packaging==26.2",
        "PyYAML==6.0.3",
        "regex==2026.7.19",
        "requests==2.34.2",
        "tqdm==4.67.3",
        "pillow==12.2.0",
        "sentencepiece==0.2.1",
        "filelock==3.29.0",
    ):
        assert requirement in requirements
        assert requirement.lower() in lock.lower()
    assert "qwen36_a4_runtime_lock.txt" in text
    assert "click==8.4.2" in requirements
    assert "click==8.4.2" in lock
    assert "pip uninstall" in wrapper
    assert "spin" in wrapper
    assert 'importlib.metadata.version("spin")' in wrapper


def test_a2_transformers_4576_runtime_fails_qwen35_moe_capability_check(
    tmp_path: Path,
) -> None:
    fake_runtime = tmp_path / "a2-runtime"
    fake_transformers = fake_runtime / "transformers"
    fake_transformers.mkdir(parents=True)
    (fake_transformers / "__init__.py").write_text(
        "__version__ = '4.57.6'\n"
        "class AutoConfig:\n"
        "    @staticmethod\n"
        "    def for_model(name):\n"
        "        raise ValueError(f'unrecognized model {name}')\n",
        encoding="utf-8",
    )

    completed = _run_wrapper(
        tmp_path,
        "verify-transformers-runtime",
        extra_env={"PYTHONPATH": str(fake_runtime)},
    )

    assert completed.returncode != 0
    assert "qwen3_5_moe" in completed.stderr


def test_a4_missing_interpreter_regression_and_runtime_preflight_config() -> None:
    main = CONFIG.read_text(encoding="utf-8")
    preflight = RUNTIME_PREFLIGHT_CONFIG.read_text(encoding="utf-8")
    assert "/usr/local/bin/python3" not in main
    assert "/usr/local/bin/python3" not in preflight
    assert "PYTHON_BIN=/usr/bin/python3" in main
    assert "PYTHON_BIN=/usr/bin/python3" in preflight
    assert EXPECTED_IMAGE in preflight
    assert "qwen36_a4_runtime_lock.txt" in preflight
    assert "bootstrap-transformers" in preflight
    assert "fetch-evidence" not in preflight
    assert " run" not in preflight
    assert "max_duration: 30m" in preflight


def test_staged_request_identity_was_bumped_to_a5() -> None:
    requests = json.loads(STAGED_REQUESTS.read_text(encoding="utf-8"))
    assert len(requests) == 12
    assert {request["run_id"] for request in requests} == {
        "qwen36-failure-panel-20260724-a10"
    }


def test_wrapper_preflight_exercises_real_child_argv(tmp_path: Path) -> None:
    completed = _run_wrapper(tmp_path, "--preflight")

    assert completed.returncode == 0, completed.stderr
    record = json.loads((tmp_path / "output/preflight.json").read_text())
    assert record["panel_manifest_sha256"] == EXPECTED_PANEL_SHA256
    assert record["request_count"] == 12
    assert record["model_revision"] == EXPECTED_MODEL_REVISION
    assert record["child_argv"] == [
        str(Path(os.sys.executable).resolve()),
        "-m",
        "trec_video_2026.vqa_transformers_runner",
        "--model-id",
        "Qwen/Qwen3.6-35B-A3B",
        "--revision",
        EXPECTED_MODEL_REVISION,
        "--model-class",
        "image-text",
        "--max-new-tokens",
        "64",
    ]
    assert record["child_help_returncode"] == 0


def test_prepare_local_projects_only_label_free_derived_frame_requests(
    tmp_path: Path,
) -> None:
    source = _label_free_requests(tmp_path)
    staged = tmp_path / "staged/requests.json"
    completed = _run_wrapper(
        tmp_path,
        "prepare-local",
        extra_env={
            "SOURCE_REQUESTS": str(source),
            "STAGED_REQUESTS": str(staged),
        },
    )

    assert completed.returncode == 0, completed.stderr
    requests = json.loads(staged.read_text(encoding="utf-8"))
    assert len(requests) == 12
    assert list(requests[0]) == [
        "condition_id",
        "frame_descriptions",
        "frame_paths",
        "frame_timestamps",
        "question",
        "run_id",
        "transcript",
        "video_id",
    ]
    assert requests[0]["frame_paths"][0] == (
        "/dstack/input/evidence/NN7G28sBOlY/frame-01.png"
    )
    assert all(request["transcript"] is None for request in requests)


def test_prepare_local_evidence_hardlinks_only_frozen_derived_frames(
    tmp_path: Path,
) -> None:
    source = _label_free_requests(tmp_path)
    staged = tmp_path / "staged-evidence"
    completed = _run_wrapper(
        tmp_path,
        "prepare-evidence-local",
        extra_env={
            "SOURCE_REQUESTS": str(source),
            "SOURCE_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "STAGED_EVIDENCE_ROOT": str(staged),
        },
    )

    assert completed.returncode == 0, completed.stderr
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    assert sorted(path.name for path in staged.iterdir()) == sorted(
        example["video_id"] for example in panel["examples"]
    )
    for example in panel["examples"]:
        source_frame = tmp_path / "evidence" / example["video_id"] / "frame-01.png"
        staged_frame = staged / example["video_id"] / "frame-01.png"
        assert staged_frame.samefile(source_frame)
        assert len(list((staged / example["video_id"]).glob("frame-*.png"))) == 12
        assert (staged / example["video_id"] / "evidence.json").is_file()
    assert len([path for path in staged.rglob("*") if path.is_file()]) == 156

    repeated = _run_wrapper(
        tmp_path,
        "prepare-evidence-local",
        extra_env={
            "SOURCE_REQUESTS": str(source),
            "SOURCE_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "STAGED_EVIDENCE_ROOT": str(staged),
        },
    )
    assert repeated.returncode == 0, repeated.stderr


@pytest.mark.parametrize("corruption", ["extra", "symlink", "label", "partial"])
def test_prepare_local_evidence_rejects_unsafe_or_stale_destination(
    tmp_path: Path, corruption: str
) -> None:
    source = _label_free_requests(tmp_path)
    staged = tmp_path / "staged-evidence"
    first = _run_wrapper(
        tmp_path,
        "prepare-evidence-local",
        extra_env={
            "SOURCE_REQUESTS": str(source),
            "SOURCE_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "STAGED_EVIDENCE_ROOT": str(staged),
        },
    )
    assert first.returncode == 0, first.stderr
    first_id = json.loads(PANEL.read_text())["examples"][0]["video_id"]
    if corruption == "extra":
        (staged / "stale.txt").write_text("stale", encoding="utf-8")
    elif corruption == "symlink":
        (staged / first_id / "link").symlink_to("frame-01.png")
    elif corruption == "label":
        metadata = tmp_path / "evidence" / first_id / "evidence.json"
        payload = json.loads(metadata.read_text())
        payload["annotation"] = "leak"
        metadata.write_text(json.dumps(payload), encoding="utf-8")
    else:
        (staged / first_id / "frame-12.png").unlink()

    repeated = _run_wrapper(
        tmp_path,
        "prepare-evidence-local",
        extra_env={
            "SOURCE_REQUESTS": str(source),
            "SOURCE_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "STAGED_EVIDENCE_ROOT": str(staged),
        },
    )

    assert repeated.returncode != 0


def test_wrapper_rejects_zero_exit_child_with_failed_rows(tmp_path: Path) -> None:
    requests = json.loads(_label_free_requests(tmp_path).read_text())
    results = [
        {
            "video_id": request["video_id"],
            "status": "failed",
            "answer": None,
            "model_id": "Qwen/Qwen3.6-35B-A3B",
            "model_revision": EXPECTED_MODEL_REVISION,
            "runtime_image": EXPECTED_RUNTIME_IMAGE,
            "precision": "bfloat16",
            "device": "cuda:0 (NVIDIA CUDA)",
            "runtime_attestation": {
                "accelerator_backend": "nvidia-cuda",
                "parameter_devices": {"cuda:0": 1},
                "parameter_dtypes": {"torch.bfloat16": 1},
                "buffer_devices": {"cuda:0": 1},
            },
        }
        for request in requests
    ]
    result_path = tmp_path / "failed-results.json"
    result_path.write_text(json.dumps(results), encoding="utf-8")
    fake_child = tmp_path / "child"
    fake_child.write_text(
        f"#!/bin/sh\ncat '{result_path}'\nexit 0\n",
        encoding="utf-8",
    )
    fake_child.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_uvx.chmod(0o755)

    completed = _run_wrapper(
        tmp_path,
        "run",
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CHILD_EXECUTABLE": str(fake_child),
            "VQA_RUNTIME_IMAGE": EXPECTED_RUNTIME_IMAGE,
        },
    )

    assert completed.returncode != 0
    assert (
        "prediction validation failed"
        in (tmp_path / "output/workload.stderr").read_text()
    )
    terminal = json.loads((tmp_path / "output/terminal-status.json").read_text())
    assert terminal["workload_returncode"] != 0


def test_local_upload_preflight_rejects_existing_immutable_prefix(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "hf buckets list" ]; then '
        "echo existing-object; exit 0; fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_uvx.chmod(0o755)
    staged = tmp_path / "staged-evidence"
    staged.mkdir()

    completed = _run_wrapper(
        tmp_path,
        "preflight-evidence-upload-local",
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "STAGED_EVIDENCE_ROOT": str(staged),
            "EVIDENCE_ARTIFACT_URI": "hf://buckets/Npatta01/trec_mlm_2026/"
            "trec_video_2026/artifacts/failure-panel-12-frames-d2c5515b8c73",
        },
    )

    assert completed.returncode != 0
    assert "evidence artifact prefix already exists" in completed.stderr


@pytest.mark.parametrize("forbidden", ["reference", "answer", "annotation"])
def test_wrapper_rejects_label_fields_recursively(
    tmp_path: Path, forbidden: str
) -> None:
    requests = _label_free_requests(tmp_path)
    payload = json.loads(requests.read_text())
    payload[0][forbidden] = "leak"
    requests.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run_wrapper(
        tmp_path,
        "--preflight",
        extra_env={"INPUT_REQUESTS": str(requests)},
    )

    assert completed.returncode != 0
    assert "forbidden label-bearing field" in completed.stderr


def test_wrapper_rejects_wrong_panel_hash_and_mutable_run_name(tmp_path: Path) -> None:
    wrong_hash = _run_wrapper(
        tmp_path / "wrong-hash",
        "--preflight",
        extra_env={"EXPECTED_PANEL_SHA256": "0" * 64},
    )
    mutable = _run_wrapper(
        tmp_path / "mutable",
        "--preflight",
        extra_env={"EXPERIMENT_RUN_NAME": "latest"},
    )

    assert wrong_hash.returncode != 0
    assert "panel manifest SHA-256 mismatch" in wrong_hash.stderr
    assert mutable.returncode != 0
    assert "immutable experiment run name" in mutable.stderr


def test_wrapper_preserves_workload_rc_when_sync_also_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "hf buckets list" ]; then exit 0; fi\n'
        'if [ "$1 $2 $3" = "hf buckets sync" ]; then exit 23; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_uvx.chmod(0o755)
    fake_child = tmp_path / "child"
    fake_child.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
    fake_child.chmod(0o755)

    completed = _run_wrapper(
        tmp_path,
        "run",
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CHILD_EXECUTABLE": str(fake_child),
        },
    )

    assert completed.returncode == 17
    terminal = json.loads((tmp_path / "output/terminal-status.json").read_text())
    assert terminal == {"sync_returncode": 23, "workload_returncode": 17}


def test_wrapper_rejects_existing_immutable_remote_prefix(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "hf buckets list" ]; then '
        "echo existing-object; exit 0; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_uvx.chmod(0o755)

    completed = _run_wrapper(
        tmp_path,
        "run",
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert completed.returncode != 0
    assert "remote run prefix already exists" in completed.stderr


def test_failure_panel_manifest_identity_is_still_frozen() -> None:
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == EXPECTED_PANEL_SHA256
