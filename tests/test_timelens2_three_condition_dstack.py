"""Executable safety contracts for TimeLens2 three-condition dstack tasks."""

from __future__ import annotations

import os
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
SOURCE_SHA256 = "12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499"
GPU = "A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200:1:48GB.."
CASES = (
    (
        "timelens2-three-condition-canary-20260801-a1",
        Path("dstack/timelens2-three-condition-canary.dstack.yml"),
        Path("scripts/run-timelens2-three-condition-canary-dstack.sh"),
        "30m",
    ),
    (
        "timelens2-three-condition-frozen30-20260801-a1",
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


@pytest.mark.parametrize("run_id,config_path,runner_path,duration", CASES)
def test_dstack_config_is_exact_cuda_and_budget_bounded(
    run_id: str, config_path: Path, runner_path: Path, duration: str
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    env = _env_list(config)

    assert config["type"] == "task"
    assert config["name"] == run_id
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


@pytest.mark.parametrize("run_id,config_path,runner_path,_duration", CASES)
def test_runner_validate_executes_without_network_or_secret_output(
    run_id: str, config_path: Path, runner_path: Path, _duration: str
) -> None:
    remote = (
        "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
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
        env["HF_INPUT_URI"] = f"{remote}/input"
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


def test_frozen30_task_mounts_code_only_and_uses_private_input_uri() -> None:
    config = yaml.safe_load(CASES[1][1].read_text(encoding="utf-8"))
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
