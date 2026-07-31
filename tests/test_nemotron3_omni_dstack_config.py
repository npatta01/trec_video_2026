"""Safety and identity checks for the Nemotron 3 Omni dstack pilot."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from trec_video_2026.vqa_experiments.nemotron3_omni_pilot import (
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
REMOTE_ROOT = "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments"
CANARY_CONFIG = Path("dstack/nemotron3-omni-runner-canary-a2.dstack.yml")
FAILED_CANARY_CONFIG = Path("dstack/nemotron3-omni-runner-canary.dstack.yml")
PILOT_CONFIG = Path("dstack/nemotron3-omni-pilot.dstack.yml")
FLEET_ID = "gpu-marketplace"
RUNNER = Path("scripts/run-nemotron3-omni-dstack.sh")
UPLOADER = Path("scripts/upload-nemotron3-omni-pilot-inputs.sh")
COLLECTOR = Path("scripts/collect-nemotron3-omni-dstack.sh")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    assert uploader.count(".mkv:") == 6


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
