"""Executable safety contracts for TimeLens2 three-condition dstack tasks."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
SELECTOR_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
ANSWER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
TRANSFORMERS_VERSION = "5.15.0.dev0"
TRANSFORMERS_SOURCE_URL = (
    "https://github.com/huggingface/transformers/archive/"
    f"{TRANSFORMERS_REVISION}.tar.gz"
)
SOURCE_SHA256 = "12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499"
GPU = "A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:48GB.."
CASES = (
    (
        "tl2-3cond-canary-20260801-a3",
        "timelens2-three-condition-canary-20260801-a3",
        Path("dstack/timelens2-three-condition-canary.dstack.yml"),
        Path("scripts/run-timelens2-three-condition-canary-dstack.sh"),
        "30m",
    ),
    (
        "tl2-3cond-frozen30-20260801-a2",
        "timelens2-three-condition-frozen30-20260801-a2",
        Path("dstack/timelens2-three-condition-frozen30.dstack.yml"),
        Path("scripts/run-timelens2-three-condition-dstack.sh"),
        "2h",
    ),
)


def _env_list(config: dict[str, object]) -> dict[str, str]:
    values = config["env"]
    assert isinstance(values, list)
    result = {}
    for value in values:
        key, separator, item = str(value).partition("=")
        assert separator
        result[key] = item
    return result


@pytest.mark.parametrize(
    "task_name,run_id,config_path,runner_path,duration", CASES
)
def test_dstack_config_is_exact_cuda_and_budget_bounded(
    task_name: str,
    run_id: str,
    config_path: Path,
    runner_path: Path,
    duration: str,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env = _env_list(config)

    assert config["type"] == "task"
    assert config["name"] == task_name
    assert len(task_name) <= 41
    assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", task_name)
    assert config["image"] == IMAGE
    assert env["HF_TOKEN"] == "${{ secrets.hf_token }}"
    assert env["EXPERIMENT_RUN_ID"] == run_id
    assert env["HF_REMOTE_URI"].endswith(f"/experiments/{run_id}")
    assert env["RUNTIME_IMAGE"] == IMAGE
    assert env["SELECTOR_MODEL_REVISION"] == SELECTOR_REVISION
    assert env["ANSWER_MODEL_REVISION"] == ANSWER_REVISION
    assert config["resources"] == {
        "gpu": GPU,
        "cpu": "8..",
        "memory": "96GB..",
        "disk": "100GB..",
    }
    assert config["fleets"] == ["gpu-marketplace"]
    assert config["backends"] == ["runpod", "vastai"]
    assert config["spot_policy"] == "on-demand"
    assert config["max_price"] == 1.0
    assert config["max_duration"] == duration
    assert config["retry"] == {"on_events": ["no-capacity"], "duration": "20m"}
    serialized = config_path.read_text(encoding="utf-8").lower()
    assert "rocm" not in serialized
    assert "amd" not in serialized
    assert "adjudicated" not in serialized
    assert "annotation" not in serialized
    assert ".timelens2-private" not in serialized
    assert runner_path.name in serialized


@pytest.mark.parametrize(
    "_task_name,run_id,config_path,runner_path,_duration", CASES
)
def test_runner_validate_executes_without_network_or_secret_output(
    _task_name: str,
    run_id: str,
    config_path: Path,
    runner_path: Path,
    _duration: str,
) -> None:
    remote = (
        "hf://buckets/Npatta01/trec_video_2026/experiments/"
        f"{run_id}"
    )
    env = {
        **os.environ,
        "HF_TOKEN": "test-only-placeholder",
        "PYTHON_BIN": sys.executable,
        "EXPERIMENT_RUN_ID": run_id,
        "HF_REMOTE_URI": remote,
        "RUNTIME_IMAGE": IMAGE,
        "SELECTOR_MODEL_ID": "MCG-NJU/TimeLens2-2B",
        "SELECTOR_MODEL_REVISION": SELECTOR_REVISION,
        "ANSWER_MODEL_ID": "Qwen/Qwen3.5-4B",
        "ANSWER_MODEL_REVISION": ANSWER_REVISION,
    }
    if "frozen30" in run_id:
        env["SOURCE_SHA256"] = SOURCE_SHA256
        env["HF_INPUT_URI"] = (
            "hf://buckets/Npatta01/trec_video_2026/experiments/"
            "timelens2-three-condition-frozen30-20260801-a1/input"
        )
    completed = subprocess.run(
        ["bash", str(runner_path), "validate"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "test-only-placeholder" not in completed.stdout
    assert "test-only-placeholder" not in completed.stderr


@pytest.mark.parametrize("runner_path", [case[3] for case in CASES])
def test_runner_rejects_runtime_without_dense_qwen35_capability(
    tmp_path: Path, runner_path: Path
) -> None:
    fake_runtime = tmp_path / "legacy-runtime"
    transformers = fake_runtime / "transformers"
    transformers.mkdir(parents=True)
    (transformers / "__init__.py").write_text(
        "__version__ = '4.57.6'\n"
        "class AutoConfig:\n"
        "    @staticmethod\n"
        "    def for_model(name):\n"
        "        raise ValueError(f'unrecognized model {name}')\n"
        "AutoModelForImageTextToText = object\n"
        "AutoProcessor = object\n",
        encoding="utf-8",
    )
    (fake_runtime / "torch.py").write_text(
        "class version:\n"
        "    cuda = '12.4'\n"
        "    hip = None\n"
        "class cuda:\n"
        "    is_available = staticmethod(lambda: True)\n"
        "    is_bf16_supported = staticmethod(lambda: True)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(runner_path), "verify-runtime"],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(fake_runtime)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "qwen3_5" in completed.stderr


@pytest.mark.parametrize("runner_path", [case[3] for case in CASES])
def test_runner_accepts_exact_dense_qwen35_runtime_capability(
    tmp_path: Path, runner_path: Path
) -> None:
    fake_runtime = tmp_path / "supported-runtime"
    transformers = fake_runtime / "transformers"
    transformers.mkdir(parents=True)
    (transformers / "__init__.py").write_text(
        f"__version__ = '{TRANSFORMERS_VERSION}'\n"
        "class Config:\n"
        "    model_type = 'qwen3_5'\n"
        "class AutoConfig:\n"
        "    @staticmethod\n"
        "    def for_model(name):\n"
        "        if name != 'qwen3_5':\n"
        "            raise ValueError(name)\n"
        "        return Config()\n"
        "AutoModelForImageTextToText = object\n"
        "AutoProcessor = object\n",
        encoding="utf-8",
    )
    (fake_runtime / "torch.py").write_text(
        "class version:\n"
        "    cuda = '12.4'\n"
        "    hip = None\n"
        "class cuda:\n"
        "    is_available = staticmethod(lambda: True)\n"
        "    is_bf16_supported = staticmethod(lambda: True)\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", str(runner_path), "verify-runtime"],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHON_BIN": sys.executable, "PYTHONPATH": str(fake_runtime)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = __import__("json").loads(completed.stdout)
    assert record == {
        "cuda_available": True,
        "cuda_bf16_supported": True,
        "qwen3_5_model_type": "qwen3_5",
        "transformers_revision": TRANSFORMERS_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
    }


@pytest.mark.parametrize("runner_path", [case[3] for case in CASES])
def test_runner_bootstrap_uses_exact_qwen35_transformers_source(
    runner_path: Path,
) -> None:
    wrapper = runner_path.read_text(encoding="utf-8")

    assert f'readonly TRANSFORMERS_REVISION="{TRANSFORMERS_REVISION}"' in wrapper
    assert f'readonly TRANSFORMERS_VERSION="{TRANSFORMERS_VERSION}"' in wrapper
    assert TRANSFORMERS_SOURCE_URL in wrapper
    assert '"transformers @ $TRANSFORMERS_SOURCE_URL"' in wrapper
    assert 'verify_transformers_runtime "$OUTPUT_DIR/runtime-capability.json"' in wrapper


def test_frozen30_task_mounts_code_only_and_uses_private_input_uri() -> None:
    config = yaml.safe_load(CASES[1][2].read_text(encoding="utf-8"))
    env = _env_list(config)
    files = config["files"]

    assert env["SOURCE_SHA256"] == SOURCE_SHA256
    assert env["HF_INPUT_URI"].endswith(
        "/experiments/timelens2-three-condition-frozen30-20260801-a1/input"
    )
    assert files == [
        "../scripts/run-timelens2-three-condition-dstack.sh:/dstack/run/scripts/run-timelens2-three-condition-dstack.sh",
        "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
    ]
