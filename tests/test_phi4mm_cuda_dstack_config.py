from pathlib import Path


CONFIG = Path("dstack/phi4mm-quality.dstack.yml")
RUNNER = Path("scripts/run-phi4mm-cuda-dstack.sh")

MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
CONTRACT_SHA256 = "b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
RUN_ID = "phi4mm-cuda-quality-3-20260724-a3"


def test_phi4mm_dstack_runtime_is_immutable_and_bounded():
    text = CONFIG.read_text(encoding="utf-8")

    assert "HISTORICAL FAILED-RUN RECORD" in text
    assert f"image: {IMAGE}" in text
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

    assert "HISTORICAL FAILED-RUN RECORD" in text
    assert 'TRANSFORMERS_VERSION="4.48.2"' in text
    assert 'HF_CLI_VERSION="1.24.0"' in text
    assert 'UV_VERSION="0.11.3"' in text
    assert 'PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"' in text
    assert 'hf_cli download "$MODEL_ID"' in text
    assert '--revision "$MODEL_REVISION"' in text
    assert '--local-dir "$MODEL_DIR"' in text
    assert "hf_cli auth whoami" in text
    assert "preflight_rc=$?" in text
    assert "workload_rc=$?" in text
    assert "sync_rc=$?" in text
    assert "sha256sum >SHA256SUMS" in text
    assert "hf_cli buckets sync" in text
    assert "--delete" not in text


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
        f"hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/{RUN_ID}"
    )

    assert expected in combined
    assert RUN_ID in combined
    assert "HF_TOKEN=${{ secrets.hf_token }}" in combined
