"""Safety and identity checks for the Nemotron 3 Omni dstack pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
    EXPANDED_ROWS,
    FROZEN30_ROWS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    TARGET_ROWS,
)

CANARY_ID = "nemotron3-omni-runner-canary-20260730-a1"
CANARY_A2_ID = "nemotron3-omni-runner-canary-20260730-a2"
PILOT_ID = "nemotron3-omni-pilot-20260730-a1"
RETIRED_PILOT_A2_ID = "nemotron3-omni-pilot-20260730-a2"
CODEC_SENTINEL_B1_ID = "nemotron3-omni-codec-sentinel-20260730-b1"
PILOT_B1_ID = "nemotron3-omni-pilot-20260730-b1"
EXPANDED12_CANARY_C1_ID = "nemotron3-omni-expanded12-canary-20260730-c1"
EXPANDED12_CANARY_C1_SCHEDULER_ID = "nemotron3-expanded12-canary-20260730-c1"
EXPANDED12_C1_ID = "nemotron3-omni-expanded12-20260730-c1"
EXPANDED12_CANARY_D1_ID = "nemotron3-omni-expanded12-canary-20260731-d1"
EXPANDED12_CANARY_D1_SCHEDULER_ID = "nemotron3-expanded12-canary-20260731-d1"
EXPANDED12_D1_ID = "nemotron3-omni-expanded12-20260731-d1"
FROZEN30_CANARY_ID = "nemotron3-omni-frozen30-canary-20260801-a1"
FROZEN30_CANARY_SCHEDULER_ID = "nemotron3-f30-canary-20260801-a1"
FROZEN30_ID = "nemotron3-omni-frozen30-20260801-a1"
REMOTE_ROOT = "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments"
CANARY_CONFIG = Path("dstack/nemotron3-omni-runner-canary-a2.dstack.yml")
FAILED_CANARY_CONFIG = Path("dstack/nemotron3-omni-runner-canary.dstack.yml")
PILOT_CONFIG = Path("dstack/nemotron3-omni-pilot.dstack.yml")
CODEC_SENTINEL_B1_CONFIG = Path("dstack/nemotron3-omni-codec-sentinel-b1.dstack.yml")
PILOT_B1_CONFIG = Path("dstack/nemotron3-omni-pilot-b1.dstack.yml")
EXPANDED12_CANARY_C1_CONFIG = Path(
    "dstack/nemotron3-omni-expanded12-canary-c1.dstack.yml"
)
EXPANDED12_C1_CONFIG = Path("dstack/nemotron3-omni-expanded12-c1.dstack.yml")
EXPANDED12_CANARY_D1_CONFIG = Path(
    "dstack/nemotron3-omni-expanded12-canary-d1.dstack.yml"
)
EXPANDED12_D1_CONFIG = Path("dstack/nemotron3-omni-expanded12-d1.dstack.yml")
FROZEN30_CANARY_CONFIG = Path("dstack/nemotron3-omni-frozen30-canary-a1.dstack.yml")
FROZEN30_CONFIG = Path("dstack/nemotron3-omni-frozen30-a1.dstack.yml")
FLEET_ID = "gpu-marketplace"
RUNNER = Path("scripts/run-nemotron3-omni-dstack.sh")
UPLOADER = Path("scripts/upload-nemotron3-omni-pilot-inputs.sh")
COLLECTOR = Path("scripts/collect-nemotron3-omni-dstack.sh")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _env_mapping(config: dict[str, object]) -> dict[str, str]:
    values = config["env"]
    assert isinstance(values, list)
    return dict(str(value).split("=", 1) for value in values)


def _yaml(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def _write_checksums(root: Path, relative_paths: list[str]) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((root / relative).read_bytes()).hexdigest()}  ./{relative}\n"
            for relative in relative_paths
        ),
        encoding="utf-8",
    )


def _collector_harness(
    tmp_path: Path, *, technical_error: bool, extra_location: str | None = None
) -> tuple[Path, dict[str, str], Path, Path, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    collector = scripts / COLLECTOR.name
    shutil.copy2(COLLECTOR, collector)
    module_dir = repo / "src/trec_video_2026/vqa_experiments"
    module_dir.mkdir(parents=True)
    (repo / "src/trec_video_2026/__init__.py").write_text("")
    (module_dir / "__init__.py").write_text("")
    shutil.copy2(
        "src/trec_video_2026/vqa_experiments/dstack_metadata.py",
        module_dir / "dstack_metadata.py",
    )
    data_target = tmp_path / "data-target"
    data_target.mkdir()
    (repo / "data").symlink_to(data_target, target_is_directory=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_root = tmp_path / "remote"
    fake_log = tmp_path / "calls.log"
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ "$1" != hf ]]; do shift; done
shift
printf '%s\t' "$@" >>"$FAKE_CALL_LOG"
printf '\n' >>"$FAKE_CALL_LOG"
command="$2"
shift 2
remote_path() {
  printf '%s/%s' "$FAKE_HF_ROOT" "${1#hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/}"
}
case "$command" in
  list)
    target="$(remote_path "$1")"
    if [[ "$*" == *"--format quiet"* ]]; then
      [[ ! -d "$target" ]] || find "$target" -type f
    else
      printf '[]\n'
    fi
    ;;
  sync)
    if [[ "$1" == hf://* ]]; then
      source="$(remote_path "$1")"
      if [[ "${INJECT_AUDIT_EXTRA:-0}" == 1 && "$1" == */audit ]]; then
        printf 'unlisted audit object\n' >"$source/unlisted-audit.txt"
      fi
      destination="$2"
    else
      source="$1"
      destination="$(remote_path "$2")"
    fi
    mkdir -p "$destination"
    cp -a "$source/." "$destination/"
    ;;
  *) exit 97 ;;
esac
""",
        encoding="utf-8",
    )
    fake_uvx.chmod(0o755)
    fake_readlink = fake_bin / "readlink"
    fake_readlink.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == -f && "$2" == "$FAKE_DATA_LINK" ]]; then
  printf '%s\n' /home/npatta01/data/competitions/trec_video_2026/data
else
  exec /usr/bin/readlink "$@"
fi
""",
        encoding="utf-8",
    )
    fake_readlink.chmod(0o755)
    fake_dstack = fake_bin / "dstack"
    fake_dstack.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\t' "$@" >>"$FAKE_CALL_LOG"
printf '\n' >>"$FAKE_CALL_LOG"
cat "$FAKE_DSTACK_JSON"
""",
        encoding="utf-8",
    )
    fake_dstack.chmod(0o755)

    run_id = FROZEN30_ID
    outputs = fake_root / run_id / "outputs"
    outputs.mkdir(parents=True)
    checksum_paths: list[str] = []
    if technical_error:
        technical = {
            "schema_version": 1,
            "run_id": run_id,
            "failure_stage": "media-normalization",
            "exit_status": 2,
            "credentials_recorded": False,
        }
        (outputs / "technical-error.json").write_text(
            json.dumps(technical, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksum_paths.append("technical-error.json")
    else:
        normalized = outputs / "normalized"
        normalized.mkdir()
        for index in range(30):
            relative = f"normalized/media-{index:02d}.mp4"
            (outputs / relative).write_bytes(f"normalized-{index}".encode())
            checksum_paths.append(relative)
        (outputs / "manifest.json").write_text(
            '{"completion":{"completed":60,"expected":60}}\n'
        )
        checksum_paths.append("manifest.json")
    _write_checksums(outputs, checksum_paths)
    if extra_location == "workload":
        (outputs / "unlisted-workload.txt").write_text("private unlisted object\n")

    preview = tmp_path / "preview.txt"
    preview.write_text("exact declined preview\n", encoding="utf-8")
    raw_metadata = {
        "id": "run-uuid",
        "run_spec": {"run_name": run_id},
        "status": "failed" if technical_error else "done",
        "status_message": "terminal",
        "termination_reason": "job-failed" if technical_error else None,
        "submitted_at": "2026-08-01T12:00:00+00:00",
        "cost": 0.5,
        "latest_job_submission": {
            "termination_reason": "job-failed" if technical_error else None,
            "finished_at": "2026-08-01T12:10:00+00:00",
            "exit_status": 2 if technical_error else 0,
            "job_provisioning_data": {
                "backend": "runpod",
                "region": "US-TX-3",
                "price": 2.0,
                "instance_type": {
                    "name": "NVIDIA L40S",
                    "resources": {
                        "cpus": 8,
                        "memory_mib": 65536,
                        "gpus": [
                            {
                                "name": "L40S",
                                "memory_mib": 49152,
                                "vendor": "nvidia",
                            }
                        ],
                        "disk": {"size_mib": 102400},
                        "spot": False,
                    },
                },
            },
            "job_runtime_data": {
                "offer": {"backend": "runpod", "region": "US-TX-3", "price": 2.0}
            },
        },
    }
    metadata_path = tmp_path / "dstack.json"
    metadata_path.write_text(json.dumps(raw_metadata), encoding="utf-8")
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_HF_ROOT": str(fake_root),
            "FAKE_CALL_LOG": str(fake_log),
            "FAKE_DATA_LINK": str(repo / "data"),
            "FAKE_DSTACK_JSON": str(metadata_path),
            "INJECT_AUDIT_EXTRA": "1" if extra_location == "audit" else "0",
        }
    )
    return collector, env, preview, data_target, fake_root


def test_configs_pin_identity_use_cuda_and_fit_the_full_retry_budget() -> None:
    """Catches mutable identities, non-CUDA pools, or a worst case over $8."""
    configs = {
        CANARY_A2_ID: _text(CANARY_CONFIG),
        PILOT_ID: _text(PILOT_CONFIG),
    }
    worst_case = 0.0
    for run_id, config in configs.items():
        assert "type: task" in config
        assert f"name: {run_id}" in config
        assert f"image: {RUNTIME_IMAGE}" in config
        assert f"EXPERIMENT_RUN_ID={run_id}" in config
        assert f"HF_REMOTE_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"MODEL_ID={MODEL_ID}" in config
        assert f"MODEL_REVISION={MODEL_REVISION}" in config
        assert "gpu: L40S:1:48GB.." in config
        assert "backends: [runpod, vastai]" in config
        assert "spot_policy: on-demand" in config
        assert "  on_events: [no-capacity]" in config
        assert "max_duration: 45m" in config
        assert "ports:" not in config
        assert "../data/" not in config
        assert "rocm" not in config.lower()
        assert f"fleets: [{FLEET_ID}]" in config
        max_price = float(re.search(r"max_price: ([0-9.]+)", config).group(1))
        worst_case += max_price * 0.75
    assert worst_case * 1.5 <= 8.0  # both a1 runs plus at most one a2 equivalent
    assert f"name: {CANARY_ID}" in _text(FAILED_CANARY_CONFIG)


def test_b1_configs_are_sentinel_first_exact_image_cuda_tasks_with_hard_caps() -> None:
    """Catches identity drift, wrong input prefix, or an over-budget b1 retry."""
    configs = {
        CODEC_SENTINEL_B1_ID: (_text(CODEC_SENTINEL_B1_CONFIG), "30m"),
        PILOT_B1_ID: (_text(PILOT_B1_CONFIG), "45m"),
    }
    worst_case = 0.0
    for run_id, (config, duration) in configs.items():
        assert "type: task" in config
        assert f"name: {run_id}" in config
        assert f"image: {RUNTIME_IMAGE}" in config
        assert f"EXPERIMENT_RUN_ID={run_id}" in config
        assert f"HF_REMOTE_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"HF_INPUT_URI={REMOTE_ROOT}/{PILOT_B1_ID}" in config
        assert f"MODEL_ID={MODEL_ID}" in config
        assert f"MODEL_REVISION={MODEL_REVISION}" in config
        assert "gpu: L40S:1:48GB.." in config
        assert f"fleets: [{FLEET_ID}]" in config
        assert "backends: [runpod, vastai]" in config
        assert "spot_policy: on-demand" in config
        assert "max_price: 2.00" in config
        assert f"max_duration: {duration}" in config
        assert "ports:" not in config
        assert "../data/" not in config
        assert "rocm" not in config.lower()
        worst_case += 2.0 * (0.5 if duration == "30m" else 0.75)
    assert worst_case == 2.5
    assert "EXPERIMENT_MODE=codec-sentinel" in configs[CODEC_SENTINEL_B1_ID][0]
    assert "EXPERIMENT_MODE=pilot" in configs[PILOT_B1_ID][0]


def test_expanded_c1_configs_pin_the_single_canary_and_pilot_envelope() -> None:
    """Catches a mutable c1 submission, an over-cap pool, or wrong pilot mode."""
    configs = {
        EXPANDED12_CANARY_C1_ID: (
            _text(EXPANDED12_CANARY_C1_CONFIG),
            EXPANDED12_CANARY_C1_SCHEDULER_ID,
            "canary",
            "30m",
        ),
        EXPANDED12_C1_ID: (
            _text(EXPANDED12_C1_CONFIG),
            EXPANDED12_C1_ID,
            "expanded-pilot",
            "45m",
        ),
    }
    for run_id, (config, scheduler_id, mode, duration) in configs.items():
        assert "type: task" in config
        assert f"name: {scheduler_id}" in config
        assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", scheduler_id)
        assert f"EXPERIMENT_MODE={mode}" in config
        assert f"EXPERIMENT_RUN_ID={run_id}" in config
        assert f"HF_REMOTE_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"HF_INPUT_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"image: {RUNTIME_IMAGE}" in config
        assert f"RUNTIME_IMAGE={RUNTIME_IMAGE}" in config
        assert f"RUNTIME_IMAGE_DIGEST={RUNTIME_IMAGE_DIGEST}" in config
        assert f"MODEL_ID={MODEL_ID}" in config
        assert f"MODEL_REVISION={MODEL_REVISION}" in config
        assert "gpu: L40S:1:48GB.." in config
        assert f"fleets: [{FLEET_ID}]" in config
        assert "backends: [runpod, vastai]" in config
        assert "spot_policy: on-demand" in config
        assert "max_price: 2.00" in config
        assert f"max_duration: {duration}" in config
        assert "ports:" not in config
        assert "../data/" not in config
        assert "rocm" not in config.lower()
        assert "HF_TOKEN=${{ secrets.hf_token }}" in config


def test_expanded_d1_configs_pin_the_authorized_canary_and_pilot_envelope() -> None:
    """Catches drift from the exact d1 identity, prefix, or bounded c1 envelope."""
    configs = {
        EXPANDED12_CANARY_D1_ID: (
            _text(EXPANDED12_CANARY_D1_CONFIG),
            EXPANDED12_CANARY_D1_SCHEDULER_ID,
            "canary",
            "30m",
        ),
        EXPANDED12_D1_ID: (
            _text(EXPANDED12_D1_CONFIG),
            EXPANDED12_D1_ID,
            "expanded-pilot",
            "45m",
        ),
    }
    for run_id, (config, scheduler_id, mode, duration) in configs.items():
        assert "type: task" in config
        assert f"name: {scheduler_id}" in config
        assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", scheduler_id)
        assert f"EXPERIMENT_MODE={mode}" in config
        assert f"EXPERIMENT_RUN_ID={run_id}" in config
        assert f"HF_REMOTE_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"HF_INPUT_URI={REMOTE_ROOT}/{run_id}" in config
        assert f"image: {RUNTIME_IMAGE}" in config
        assert f"RUNTIME_IMAGE={RUNTIME_IMAGE}" in config
        assert f"RUNTIME_IMAGE_DIGEST={RUNTIME_IMAGE_DIGEST}" in config
        assert f"MODEL_ID={MODEL_ID}" in config
        assert f"MODEL_REVISION={MODEL_REVISION}" in config
        assert "gpu: L40S:1:48GB.." in config
        assert f"fleets: [{FLEET_ID}]" in config
        assert "backends: [runpod, vastai]" in config
        assert "spot_policy: on-demand" in config
        assert "max_price: 2.00" in config
        assert f"max_duration: {duration}" in config
        assert "  on_events: [no-capacity]" in config
        assert "  duration: 20m" in config
        assert "ports:" not in config
        assert "../data/" not in config
        assert "rocm" not in config.lower()
        assert "HF_TOKEN=${{ secrets.hf_token }}" in config

    assert len(EXPANDED12_CANARY_D1_SCHEDULER_ID) == 39

    operational = _text(RUNNER) + _text(UPLOADER) + _text(COLLECTOR)
    assert f"canary:{EXPANDED12_CANARY_D1_ID}" in operational
    assert f"expanded-pilot:{EXPANDED12_D1_ID}" in operational
    assert EXPANDED12_CANARY_D1_SCHEDULER_ID in operational
    assert "expanded12-20260731-d2" not in operational


def test_frozen30_configs_pin_exact_a1_foreground_cuda_budget_envelopes() -> None:
    """Catches an unpinned runtime, broadened GPU pool, or detached-incompatible wrapper."""
    cases = {
        FROZEN30_CANARY_ID: (
            FROZEN30_CANARY_CONFIG,
            FROZEN30_CANARY_SCHEDULER_ID,
            "canary",
            "30m",
        ),
        FROZEN30_ID: (FROZEN30_CONFIG, FROZEN30_ID, "frozen30-paired", "60m"),
    }
    for run_id, (path, scheduler_id, mode, duration) in cases.items():
        config = _yaml(path)
        assert config["type"] == "task"
        assert config["name"] == scheduler_id
        assert re.fullmatch(r"[a-z][a-z0-9-]{1,40}", scheduler_id)
        assert config["image"] == RUNTIME_IMAGE
        assert config["entrypoint"] == "/bin/bash -c"
        assert config["shell"] == "bash"
        assert config["working_dir"] == "/dstack/run"
        env = _env_mapping(config)
        assert env == {
            "HF_TOKEN": "${{ secrets.hf_token }}",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHON_BIN": "/usr/bin/python3",
            "PYTHONPATH": "/dstack/run/src",
            "EXPERIMENT_MODE": mode,
            "EXPERIMENT_RUN_ID": run_id,
            "HF_REMOTE_URI": f"{REMOTE_ROOT}/{run_id}",
            "HF_INPUT_URI": f"{REMOTE_ROOT}/{run_id}",
            "RUNTIME_IMAGE": RUNTIME_IMAGE,
            "RUNTIME_IMAGE_DIGEST": RUNTIME_IMAGE_DIGEST,
            "MODEL_ID": MODEL_ID,
            "MODEL_REVISION": MODEL_REVISION,
        }
        assert config["commands"] == [
            'set -euo pipefail\nexec bash /dstack/run/scripts/run-nemotron3-omni-dstack.sh "$EXPERIMENT_MODE"\n'
        ]
        assert config["fleets"] == [FLEET_ID]
        resources = config["resources"]
        assert resources["gpu"] == "L40S:1:48GB.."
        assert isinstance(resources["gpu"], str)
        assert config["backends"] == ["runpod", "vastai"]
        assert config["spot_policy"] == "on-demand"
        assert config["max_price"] == 2.0
        assert config["max_duration"] == duration
        assert config["retry"] == {
            "on_events": ["no-capacity"],
            "duration": "20m",
        }
        assert "ports" not in config
    assert len(FROZEN30_CANARY_SCHEDULER_ID) <= 41


def test_frozen30_scripts_reject_all_uncreated_a2_identities_before_external_work() -> (
    None
):
    """Catches accidental pre-authorization of either hypothetical a2 retry."""
    poison_token = "test-only-token-that-must-not-reach-an-external-command"
    hypothetical_ids = (
        ("canary", "nemotron3-omni-frozen30-canary-20260801-a2"),
        ("frozen30-paired", "nemotron3-omni-frozen30-20260801-a2"),
    )
    for mode, run_id in hypothetical_ids:
        runner_env = os.environ.copy()
        runner_env.update(
            {
                "EXPERIMENT_RUN_ID": run_id,
                "HF_REMOTE_URI": f"{REMOTE_ROOT}/{run_id}",
                "RUNTIME_IMAGE": RUNTIME_IMAGE,
                "RUNTIME_IMAGE_DIGEST": RUNTIME_IMAGE_DIGEST,
                "MODEL_ID": MODEL_ID,
                "MODEL_REVISION": MODEL_REVISION,
                "HF_TOKEN": poison_token,
            }
        )
        cases = (
            (RUNNER, [mode], runner_env),
            (UPLOADER, [run_id], {**os.environ, "HF_TOKEN": poison_token}),
            (COLLECTOR, [run_id], {**os.environ, "HF_TOKEN": poison_token}),
        )
        for script, args, env in cases:
            result = subprocess.run(
                ["bash", str(script), *args],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                timeout=5,
            )
            assert result.returncode == 2, (script, result.stdout, result.stderr)
            assert "unexpected immutable" in result.stderr

    operational = _text(RUNNER) + _text(UPLOADER) + _text(COLLECTOR)
    assert "frozen30-canary-20260801-a2" not in operational
    assert "frozen30-20260801-a2" not in operational


def test_frozen30_uploader_emits_contract_derived_checksummed_32_object_bundle(
    tmp_path: Path,
) -> None:
    """Catches a second media list, a missing MP4, or an unverifiable remote bundle."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_root = tmp_path / "remote"
    fake_log = tmp_path / "hf.log"
    fake_uvx = fake_bin / "uvx"
    fake_uvx.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
while [[ \"$1\" != hf ]]; do shift; done
shift
printf '%s\\t' \"$@\" >>\"$FAKE_HF_LOG\"
printf '\\n' >>\"$FAKE_HF_LOG\"
[[ \"$1:$2\" == buckets:list || \"$1:$2\" == buckets:cp || \"$1:$2\" == buckets:sync ]]
command=\"$2\"
shift 2
remote_path() {
  printf '%s/%s' \"$FAKE_HF_ROOT\" \"${1#hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/}\"
}
case \"$command\" in
  list)
    target=\"$(remote_path \"$1\")\"
    if [[ \"$*\" == *\"--format quiet\"* ]]; then
      [[ ! -d \"$target\" ]] || find \"$target\" -type f -o -type l
    else
      printf '[]\\n'
    fi
    ;;
  cp)
    source=\"$1\"
    target=\"$(remote_path \"$2\")\"
    mkdir -p \"$(dirname \"$target\")\"
    if [[ \"$(basename \"$source\")\" == contract.json || \"$(basename \"$source\")\" == INPUTS_COMPLETE.json ]]; then
      cp -- \"$source\" \"$target\"
    else
      ln -s \"$(readlink -f \"$source\")\" \"$target\"
    fi
    ;;
  sync)
    source=\"$(remote_path \"$1\")\"
    destination=\"$2\"
    mkdir -p \"$destination\"
    cp -a \"$source/.\" \"$destination/\"
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_uvx.chmod(0o755)
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_HF_ROOT": str(fake_root),
            "FAKE_HF_LOG": str(fake_log),
        }
    )

    result = subprocess.run(
        ["bash", str(UPLOADER), FROZEN30_ID],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    uploaded = fake_root / FROZEN30_ID / "inputs"
    remote_files = sorted(
        path.relative_to(uploaded).as_posix()
        for path in uploaded.rglob("*")
        if path.is_file()
    )
    assert len(remote_files) == 32
    assert remote_files[0:2] == ["INPUTS_COMPLETE.json", "contract.json"]
    contract = json.loads((uploaded / "contract.json").read_text())
    expected_media = sorted(example["media_path"] for example in contract["examples"])
    assert len(expected_media) == len(set(expected_media)) == 30
    assert "vqa/2025/training/videos_original/QFljXc4WpFA.mp4" in expected_media
    assert remote_files[2:] == expected_media
    marker = json.loads((uploaded / "INPUTS_COMPLETE.json").read_text())
    assert marker["object_count"] == 32
    assert marker["input_count"] == 31
    assert sorted(marker["objects"]) == ["contract.json", *expected_media]

    calls = fake_log.read_text().splitlines()
    copy_calls = [line for line in calls if line.startswith("buckets\tcp\t")]
    assert len(copy_calls) == 32
    assert any(line.startswith("buckets\tsync\t") for line in calls)
    assert any("buckets\tlist\t" in line and "--recursive" in line for line in calls)


def test_frozen30_runner_and_collector_preserve_independently_hashable_media() -> None:
    """Catches duplicated media identities or a predictions-only collected bundle."""
    runner = _text(RUNNER)
    uploader = _text(UPLOADER)
    collector = _text(COLLECTOR)
    operational = runner + uploader + collector

    assert f"canary:{FROZEN30_CANARY_ID}" in runner
    assert f"frozen30-paired:{FROZEN30_ID}" in runner
    assert "build-frozen30-contract" in uploader
    assert "native_audio_benchmark_30_inference.json" in uploader
    historical_ids = {row[0] for row in TARGET_ROWS} | {row[0] for row in EXPANDED_ROWS}
    for video_id, _category, _role in FROZEN30_ROWS:
        if video_id in historical_ids:
            continue
        assert video_id not in uploader
        assert video_id not in runner
    assert "contract_media_records" in runner
    assert 'preserve_normalized_media "$normalized_root"' in runner
    assert "OUTPUT_DIR/normalized" in runner
    assert runner.index("preflight_inputs") < runner.index("normalize_pilot_inputs")
    assert runner.index("normalize_pilot_inputs") < runner.index("start_server")
    assert "find . -type f ! -name SHA256SUMS" in runner
    assert "sha256sum -c SHA256SUMS" in runner

    assert FROZEN30_CANARY_ID in collector
    assert FROZEN30_CANARY_SCHEDULER_ID in collector
    assert FROZEN30_ID in collector
    assert "preview output is required for frozen30 a1" in collector
    assert 'cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"' in collector
    assert 'dstack run get "$DSTACK_RUN_NAME" --json' in collector
    assert 'hf_cli buckets sync "${REMOTE_URI}/outputs" "$TEMP_OUTPUT_DIR"' in collector
    assert "sha256sum -c SHA256SUMS" in collector
    assert "--diagnose" not in operational


def test_frozen30_collector_preserves_terminal_technical_error_without_normalized_media(
    tmp_path: Path,
) -> None:
    """Catches stranding a valid pre-normalization failure before audit collection."""
    collector, env, preview, data_target, fake_root = _collector_harness(
        tmp_path, technical_error=True
    )
    result = subprocess.run(
        ["bash", str(collector), FROZEN30_ID, str(preview)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    collected = data_target / "vqa/benchmarks/nemotron3-omni" / FROZEN30_ID
    technical = json.loads((collected / "technical-error.json").read_text())
    assert technical["failure_stage"] == "media-normalization"
    assert technical["exit_status"] == 2
    assert not (collected / "normalized").exists()
    assert (collected / "dstack-run.json").is_file()
    assert (collected / "artifact-hashes.sha256").is_file()
    status = json.loads((collected / "collection-status.json").read_text())
    assert status == {
        "schema_version": 1,
        "run_id": FROZEN30_ID,
        "collection_kind": "terminal-non-scoreable",
        "scoreable": False,
        "reason": "technical-error",
    }
    audit = fake_root / FROZEN30_ID / "audit"
    assert (audit / "preview.txt").read_text() == preview.read_text()
    assert (audit / "dstack-run.json").is_file()
    assert json.loads((audit / "collection-status.json").read_text()) == status
    assert (audit / "SHA256SUMS").is_file()


@pytest.mark.parametrize("extra_location", ["workload", "audit"])
def test_frozen30_collector_rejects_unlisted_recursive_bundle_objects(
    tmp_path: Path, extra_location: str
) -> None:
    """Catches accepting privacy-sensitive extras absent from the producer checksum."""
    collector, env, preview, data_target, _fake_root = _collector_harness(
        tmp_path, technical_error=False, extra_location=extra_location
    )
    result = subprocess.run(
        ["bash", str(collector), FROZEN30_ID, str(preview)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert "checksum file set mismatch" in result.stderr
    collected = data_target / "vqa/benchmarks/nemotron3-omni" / FROZEN30_ID
    assert not collected.exists()


def test_frozen30_collector_requires_30_normalized_files_for_non_failure_bundle(
    tmp_path: Path,
) -> None:
    """Catches treating a 29-media non-failure bundle as scoreable."""
    collector, env, preview, data_target, fake_root = _collector_harness(
        tmp_path, technical_error=False
    )
    missing = fake_root / FROZEN30_ID / "outputs/normalized/media-29.mp4"
    missing.unlink()
    _write_checksums(
        fake_root / FROZEN30_ID / "outputs",
        [
            "manifest.json",
            *(f"normalized/media-{index:02d}.mp4" for index in range(29)),
        ],
    )
    result = subprocess.run(
        ["bash", str(collector), FROZEN30_ID, str(preview)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )
    assert result.returncode != 0
    assert "exactly 30 normalized artifacts" in result.stderr
    collected = data_target / "vqa/benchmarks/nemotron3-omni" / FROZEN30_ID
    assert not collected.exists()


def test_frozen30_collector_keeps_completed_candidate_pending_independent_verification(
    tmp_path: Path,
) -> None:
    """Catches either blocking 60/60 collection or prematurely declaring it scoreable."""
    collector, env, preview, data_target, _fake_root = _collector_harness(
        tmp_path, technical_error=False
    )
    result = subprocess.run(
        ["bash", str(collector), FROZEN30_ID, str(preview)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    collected = data_target / "vqa/benchmarks/nemotron3-omni" / FROZEN30_ID
    assert len(list((collected / "normalized").glob("*.mp4"))) == 30
    status = json.loads((collected / "collection-status.json").read_text())
    assert status == {
        "schema_version": 1,
        "run_id": FROZEN30_ID,
        "collection_kind": "completed-candidate",
        "scoreable": False,
        "reason": "pending-independent-freeze-verification",
    }


def test_expanded_c1_scripts_keep_preview_audit_and_expanded_input_boundaries() -> None:
    """Catches skipped preview evidence, incomplete inputs, or unnormalized expanded media."""
    runner = _text(RUNNER)
    uploader = _text(UPLOADER)
    collector = _text(COLLECTOR)
    operational = runner + uploader + collector

    assert f"canary:{EXPANDED12_CANARY_C1_ID}" in runner
    assert f"expanded-pilot:{EXPANDED12_C1_ID}" in runner
    assert "build-expanded-contract" in uploader
    assert "input_count=13" in uploader
    expanded_uploader = uploader.split(
        'if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" || "$RUN_ID" == "$EXPECTED_RUN_ID_D1" ]]',
        1,
    )[1]
    assert expanded_uploader.split("fi", 1)[0].count(".mkv:") == 12
    assert "expanded-pilot" in runner
    assert "normalize_pilot_inputs" in runner
    assert runner.count("ids=(NN7G28sBOlY NUmb2P3Bzu8") == 2
    assert runner.index("normalize_pilot_inputs") < runner.index("start_server")
    assert EXPANDED12_CANARY_C1_ID in collector
    assert EXPANDED12_C1_ID in collector
    assert EXPANDED12_CANARY_C1_SCHEDULER_ID in collector
    assert 'DSTACK_RUN_NAME="$RUN_ID"' in collector
    assert 'dstack run get "$DSTACK_RUN_NAME" --json' in collector
    assert '--run-id "$DSTACK_RUN_NAME"' in collector
    assert "preview output is required for c1" in collector
    assert 'cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"' in collector
    assert 'hf_cli buckets sync "$TEMP_AUDIT_DIR" "${REMOTE_URI}/audit"' in collector
    assert "dstack run get" in collector
    assert "dstack_metadata" in collector
    assert "--diagnose" not in operational
    assert RETIRED_PILOT_A2_ID not in operational
    c1_config_filenames = {
        path.name for path in Path("dstack").glob("nemotron3-omni-*.dstack.yml")
    }
    assert "20260730-c2" not in operational
    assert c1_config_filenames == {
        "nemotron3-omni-codec-sentinel-b1.dstack.yml",
        "nemotron3-omni-expanded12-c1.dstack.yml",
        "nemotron3-omni-expanded12-canary-c1.dstack.yml",
        "nemotron3-omni-expanded12-canary-d1.dstack.yml",
        "nemotron3-omni-expanded12-d1.dstack.yml",
        "nemotron3-omni-frozen30-a1.dstack.yml",
        "nemotron3-omni-frozen30-canary-a1.dstack.yml",
        "nemotron3-omni-pilot-b1.dstack.yml",
        "nemotron3-omni-pilot.dstack.yml",
        "nemotron3-omni-runner-canary-a2.dstack.yml",
        "nemotron3-omni-runner-canary.dstack.yml",
    }
    assert all(
        filename not in c1_config_filenames
        for filename in (
            "nemotron3-omni-pilot-c2.dstack.yml",
            "nemotron3-omni-expanded12-c2.dstack.yml",
            "nemotron3-omni-c2.dstack.yml",
        )
    )


def test_runner_uses_real_foreground_wrapper_and_native_server_seam() -> None:
    """Catches a synthetic smoke test, hidden retry loop, or secret-unsafe debug path."""
    runner = _text(RUNNER)
    assert RUNTIME_IMAGE in runner
    assert MODEL_ID in runner
    assert MODEL_REVISION in runner
    assert 'readonly PYTHON_BIN="/usr/bin/python3"' in runner
    assert 'vllm_version != "0.20.0"' in runner
    assert "torch.cuda.is_available()" in runner
    assert "torch.version.hip is None" in runner
    assert "uname -m" in runner
    assert '"$VLLM_BIN" serve "$MODEL_ID"' in runner
    assert '--revision "$MODEL_REVISION"' in runner
    assert "--allowed-local-media-path /dstack/input" in runner
    assert "--reasoning-parser nemotron_v3" in runner
    assert 'server_pid="$!"' in runner
    assert (
        '"$PYTHON_BIN" -m trec_video_2026.vqa_experiments.nemotron3_omni_pilot run'
        in runner
    )
    assert "workload_rc=$?" in runner
    assert "finalize_and_sync" in runner
    assert "--diagnose" not in runner
    assert "logs -d" not in runner
    assert "dstack apply" not in runner
    assert "hf download" not in runner
    assert 'HF_TOKEN="$HF_ACCESS_TOKEN" "$VLLM_BIN"' not in runner
    assert "ffprobe" in runner
    assert 'stream_types != ["audio", "video"]' in runner
    assert 'media_root="$INPUT_DIR/media"' in runner
    assert "validate-inputs" in runner
    assert runner.index("validate-inputs") < runner.index("start_server")
    assert "technical-error.json" in runner
    assert "failure_stage" in runner


def test_b1_runner_normalizes_all_six_rows_and_validates_both_streams() -> None:
    """Catches treatment-only transcoding or normalization without decode checks."""
    runner = _text(RUNNER)
    assert "normalize_pilot_inputs" in runner
    assert "-c:v libx264" in runner
    assert "-pix_fmt yuv420p" in runner
    assert "-c:a aac" in runner
    assert "-map_metadata -1" in runner
    assert "-threads 1" in runner
    assert 'video_codecs != ["h264"]' in runner
    assert 'audio_codecs != ["aac"]' in runner
    assert "ffmpeg" in runner
    assert "-map 0:v:0" in runner
    assert "-map 0:a:0" in runner
    assert "build-normalized-contract" in runner
    assert runner.index("validate-inputs") < runner.index("normalize_pilot_inputs")
    assert runner.index("normalize_pilot_inputs") < runner.index("start_server")
    for video_id, _, _ in TARGET_ROWS:
        assert video_id in runner
    assert 'output="$normalized_root/$id.mp4"' in runner


def test_private_uploader_selects_only_six_original_media_files_without_token_env() -> (
    None
):
    """Catches label/media over-upload or placing HF_TOKEN outside dstack secrets."""
    uploader = _text(UPLOADER)
    assert "HF_TOKEN must remain unset" in uploader
    assert "build-contract" in uploader
    assert "configs/vqa/failure_panel_12.json" in uploader
    assert "annotation" not in uploader.lower()
    assert "--delete" not in uploader
    assert "videos_original" in uploader
    assert "hf_cli buckets cp" in uploader
    for video_id, _, _ in TARGET_ROWS:
        assert video_id in uploader
    historical_uploader = uploader.split(
        'if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" || "$RUN_ID" == "$EXPECTED_RUN_ID_D1" ]]',
        1,
    )[0]
    assert historical_uploader.count(".mkv:") == 6
    assert PILOT_B1_ID in uploader
    assert CODEC_SENTINEL_B1_ID not in uploader


def test_collector_downloads_only_outputs_and_checksum_verifies_before_publish() -> (
    None
):
    """Catches exposure of input media or publication of unverified artifacts."""
    collector = _text(COLLECTOR)
    assert "${REMOTE_URI}/outputs" in collector
    assert "${REMOTE_URI}/inputs" not in collector
    assert "sha256sum -c SHA256SUMS" in collector
    assert "dstack run get" in collector
    assert "dstack_metadata" in collector
    assert "refusing to overwrite" in collector
    assert "--diagnose" not in collector
    assert "--delete" not in collector


def test_b1_collector_durably_verifies_exact_preview_and_sanitized_run_metadata() -> (
    None
):
    """Catches a local-only audit trail or unchecked immutable audit upload."""
    collector = _text(COLLECTOR)
    assert CODEC_SENTINEL_B1_ID in collector
    assert PILOT_B1_ID in collector
    assert "preview output is required for b1" in collector
    assert 'cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"' in collector
    assert "dstack run get" in collector
    assert "dstack_metadata" in collector
    assert '"${REMOTE_URI}/audit"' in collector
    assert "immutable audit prefix already exists" in collector
    assert 'hf_cli buckets sync "$TEMP_AUDIT_DIR" "${REMOTE_URI}/audit"' in collector
    assert 'hf_cli buckets list "${REMOTE_URI}/audit" --format json' in collector
    assert 'hf_cli buckets sync "${REMOTE_URI}/audit" "$VERIFY_AUDIT_DIR"' in collector
    assert collector.count("verify_checksum_bundle") >= 3
    assert collector.count("sha256sum -c SHA256SUMS") >= 2


def test_model_shell_scripts_parse() -> None:
    """Catches shell syntax errors before any paid workload is submitted."""
    for script in (RUNNER, UPLOADER, COLLECTOR):
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


def test_consumed_retry_boundary_rejects_pilot_a2_before_external_work() -> None:
    """Catches re-enabling the forbidden pilot retry in any operational script."""
    poison_token = "test-only-token-that-must-not-reach-an-external-command"
    cases = (
        (
            RUNNER,
            ["pilot"],
            {
                "EXPERIMENT_RUN_ID": RETIRED_PILOT_A2_ID,
                "HF_REMOTE_URI": f"{REMOTE_ROOT}/{RETIRED_PILOT_A2_ID}",
                "RUNTIME_IMAGE": RUNTIME_IMAGE,
                "RUNTIME_IMAGE_DIGEST": RUNTIME_IMAGE_DIGEST,
                "MODEL_ID": MODEL_ID,
                "MODEL_REVISION": MODEL_REVISION,
                "HF_TOKEN": poison_token,
            },
        ),
        (UPLOADER, [RETIRED_PILOT_A2_ID], {"HF_TOKEN": poison_token}),
        (COLLECTOR, [RETIRED_PILOT_A2_ID], {"HF_TOKEN": poison_token}),
    )

    for script, args, overrides in cases:
        env = os.environ.copy()
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(script), *args],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=5,
        )
        assert result.returncode == 2, (script, result.stdout, result.stderr)
        assert "unexpected immutable" in result.stderr, (
            script,
            result.stdout,
            result.stderr,
        )


def test_b1_scripts_allow_only_the_two_new_identities_before_external_work() -> None:
    """Catches accidental b2/generalized retry paths while allowing b1 collection."""
    runner = _text(RUNNER)
    uploader = _text(UPLOADER)
    collector = _text(COLLECTOR)
    assert f"codec-sentinel:{CODEC_SENTINEL_B1_ID}" in runner
    assert f"pilot:{PILOT_B1_ID}" in runner
    assert "20260730-b2" not in runner + uploader + collector
    assert RETIRED_PILOT_A2_ID not in runner + uploader + collector
