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
CODEC_SENTINEL_B1_ID = "nemotron3-omni-codec-sentinel-20260730-b1"
PILOT_B1_ID = "nemotron3-omni-pilot-20260730-b1"
EXPANDED12_CANARY_C1_ID = "nemotron3-omni-expanded12-canary-20260730-c1"
EXPANDED12_C1_ID = "nemotron3-omni-expanded12-20260730-c1"
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
        EXPANDED12_CANARY_C1_ID: (_text(EXPANDED12_CANARY_C1_CONFIG), "canary", "30m"),
        EXPANDED12_C1_ID: (_text(EXPANDED12_C1_CONFIG), "expanded-pilot", "45m"),
    }
    for run_id, (config, mode, duration) in configs.items():
        assert "type: task" in config
        assert f"name: {run_id}" in config
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
        'if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" ]]', 1
    )[1]
    assert expanded_uploader.split("fi", 1)[0].count(".mkv:") == 12
    assert "expanded-pilot" in runner
    assert "normalize_pilot_inputs" in runner
    assert runner.count("ids=(NN7G28sBOlY NUmb2P3Bzu8") == 2
    assert runner.index("normalize_pilot_inputs") < runner.index("start_server")
    assert EXPANDED12_CANARY_C1_ID in collector
    assert EXPANDED12_C1_ID in collector
    assert "preview output is required for c1" in collector
    assert 'cp -- "$PREVIEW_PATH" "$TEMP_AUDIT_DIR/preview.txt"' in collector
    assert 'hf_cli buckets sync "$TEMP_AUDIT_DIR" "${REMOTE_URI}/audit"' in collector
    assert "dstack run get" in collector
    assert "dstack_metadata" in collector
    assert "--diagnose" not in operational
    assert RETIRED_PILOT_A2_ID not in operational
    assert "20260730-c2" not in operational


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
        'if [[ "$RUN_ID" == "$EXPECTED_RUN_ID_C1" ]]', 1
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
    assert collector.count("sha256sum -c SHA256SUMS") >= 3


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
