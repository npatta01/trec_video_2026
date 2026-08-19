from pathlib import Path
import subprocess


CONFIG = Path("dstack/phi4mm-quality.dstack.yml")
RUNNER = Path("scripts/run-phi4mm-cuda-dstack.sh")

MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
CONTRACT_SHA256 = "b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
RUN_ID = "phi4mm-cuda-quality-3-20260724-a4"


def _run_prefix_check(mock_body: str) -> subprocess.CompletedProcess[str]:
    script = RUNNER.resolve()
    command = f"""
set -euo pipefail
source {script}
hf_cli() {{ {mock_body}; }}
set +e
remote_prefix_is_empty
prefix_check_rc=$?
set -e
printf '%s' "$prefix_check_rc"
"""
    return subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )


def test_phi4mm_dstack_runtime_is_immutable_and_bounded():
    text = CONFIG.read_text(encoding="utf-8")

    assert f"image: {IMAGE}" in text
    assert "MODEL_KEY=phi4mm" in text
    assert MODEL_REVISION in text
    assert CONTRACT_SHA256 in text
    assert ":latest" not in text
    assert "type: task" in text
    assert "ports:" not in text
    assert "vendor: nvidia" in text
    assert "name: [A100, H100, H200]" in text
    assert "memory: 80GB.." in text
    assert "disk: 100GB" in text
    assert "backends: [runpod, vastai]" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 2.00" in text
    assert "max_duration: 1h" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 20m" in text


def test_phi4mm_dstack_uploads_only_the_frozen_panel():
    text = CONFIG.read_text(encoding="utf-8")

    assert text.count(".png:/dstack/input/vqa/benchmarks/evidence/") == 36
    assert text.count(".wav:/dstack/input/vqa/benchmarks/runs/") == 3
    assert "/dstack/input/contract.json" in text
    assert "videos_original" not in text
    assert "annotations" not in text


def test_phi4mm_wrapper_pins_dependencies_and_preserves_failures():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'TRANSFORMERS_VERSION="4.48.2"' in text
    assert 'SCIPY_VERSION="1.15.1"' in text
    assert 'HF_CLI_VERSION="1.24.0"' in text
    assert 'UV_VERSION="0.11.3"' in text
    assert 'PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"' in text
    assert 'hf_cli download "$MODEL_ID"' in text
    assert '--revision "$MODEL_REVISION"' in text
    assert '--local-dir "$MODEL_DIR"' in text
    assert "hf_cli auth whoami" in text
    assert "workload_rc=$?" in text
    assert "sync_rc=$?" in text
    assert "sha256sum >SHA256SUMS" in text
    assert "hf_cli buckets sync" in text
    assert "--delete" not in text


def test_phi4mm_wrapper_preflights_remote_model_import_before_weight_load():
    text = RUNNER.read_text(encoding="utf-8")

    assert "preflight_model_import" in text
    assert "get_class_from_dynamic_module" in text
    assert '"modeling_phi4mm.Phi4MMForCausalLM"' in text
    assert "AutoProcessor.from_pretrained" in text
    assert "local_files_only=True" in text
    assert text.index("preflight_model_import") < text.index(
        '"$PYTHON_BIN" -m trec_video_2026.vqa_experiments.omni_probe'
    )


def test_phi4mm_wrapper_finalizes_captured_early_failures():
    text = RUNNER.read_text(encoding="utf-8")

    for stage in (
        "artifact_cli_bootstrap",
        "authentication",
        "remote_prefix_check",
        "runtime_bootstrap",
        "input_preflight",
        "model_download",
        "model_import_preflight",
        "inference",
    ):
        assert stage in text
    assert "failure_stage" in text
    assert 'finalize_and_sync "$workload_rc" "$failure_stage"' in text


def test_phi4mm_wrapper_constrains_recursive_mutation_and_runtime_provenance():
    text = RUNNER.read_text(encoding="utf-8")

    assert '[[ "$DATA_ROOT" == "/dstack/input" ]]' in text
    assert '[[ "$RUNTIME_IMAGE_DIGEST" == "$EXPECTED_RUNTIME_IMAGE_DIGEST" ]]' in text


def test_phi4mm_token_is_scoped_away_from_remote_model_code():
    config = CONFIG.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "HF_TOKEN=${{ secrets.hf_token }}" in config
    assert 'HF_ACCESS_TOKEN="${HF_TOKEN:-}"' in runner
    assert "unset HF_TOKEN" in runner
    assert 'HF_TOKEN="$HF_ACCESS_TOKEN" uvx' in runner
    assert 'env -u HF_TOKEN "$PYTHON_BIN" -m trec_video_2026' in runner


def test_phi4mm_results_use_an_immutable_private_bucket_prefix():
    combined = CONFIG.read_text(encoding="utf-8") + RUNNER.read_text(encoding="utf-8")
    expected = (
        f"hf://buckets/Npatta01/trec_video_2026/experiments/{RUN_ID}"
    )

    assert expected in combined
    assert RUN_ID in combined
    assert "HF_TOKEN=${{ secrets.hf_token }}" in combined


def test_phi4mm_empty_prefix_check_uses_quiet_machine_output():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'buckets list "$HF_REMOTE_URI" --format quiet' in text
    assert "prefix_rc=$?" in text
    assert "(( prefix_rc == 0 ))" in text


def test_phi4mm_each_captured_stage_is_fail_fast():
    text = RUNNER.read_text(encoding="utf-8")

    assert '(\n    set -e\n    "$@"\n  )' in text


def test_phi4mm_prefix_check_accepts_empty_listing():
    result = _run_prefix_check("return 0")

    assert result.returncode == 0
    assert result.stdout == "0"


def test_phi4mm_prefix_check_preserves_cli_failure():
    result = _run_prefix_check("return 17")

    assert result.returncode == 0
    assert result.stdout == "17"
    assert "could not verify immutable remote prefix (rc=17)" in result.stderr
