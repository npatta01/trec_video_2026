"""Safety tests for immutable TimeLens2 canary and compatibility tasks."""

from pathlib import Path

CANARY = Path("dstack/timelens2-runner-canary.dstack.yml")
PILOT = Path("dstack/timelens2-selector-pilot.dstack.yml")
RUNNER = Path("scripts/run-timelens2-selector-dstack.sh")

IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"


def test_canary_is_exact_image_cuda_only_and_has_no_private_media() -> None:
    text = CANARY.read_text()

    assert "name: timelens2-runner-canary-20260730-a1" in text
    assert f"image: {IMAGE}" in text
    assert f"MODEL_REVISION={REVISION}" in text
    assert "RUN_MODE=canary" in text
    assert "gpu: A40:1:24GB.." in text
    assert "name: [" not in text
    assert "amd" not in text.lower()
    assert "rocm" not in text.lower()
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert ".mkv:" not in text
    assert "compatibility-requests.json" not in text
    assert "max_price: 0.55" in text
    assert "max_duration: 15m" in text
    assert "backends: [runpod, vastai]" in text
    assert "on_events: [no-capacity]" in text


def test_compatibility_task_uploads_only_three_videos_and_label_free_requests() -> None:
    text = PILOT.read_text()
    files = text.split("files:\n", 1)[1].split("\ncommands:", 1)[0]

    assert "name: timelens2-selector-pilot-20260730-a1" in text
    assert f"image: {IMAGE}" in text
    assert f"MODEL_REVISION={REVISION}" in text
    assert "RUN_MODE=compatibility" in text
    assert files.count(".mkv:/dstack/input/media/") == 3
    assert files.count("compatibility-requests.json") == 2
    assert "annotations/" not in files
    assert "correct_answer" not in text
    assert "temporal-relevant-scene-oracle-10" not in text
    assert "temporal-required-moment-oracle-10" not in text
    assert "max_price: 0.65" in text
    assert "max_duration: 1h" in text
    assert "gpu: A40:1:24GB.." in text
    assert "backends: [runpod, vastai]" in text
    assert "name: [" not in text


def test_wrapper_pins_runtime_and_preserves_failure_and_artifact_contracts() -> None:
    text = RUNNER.read_text()

    assert REVISION in text
    assert IMAGE in text
    assert 'readonly TRANSFORMERS_VERSION="4.57.6"' in text
    assert 'readonly QWEN_VL_UTILS_VERSION="0.0.14"' in text
    assert 'uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"' in text
    assert "buckets sync" in text
    assert "SHA256SUMS" in text
    assert "--delete" not in text
    assert "diagnose" not in text
    assert "timelens2-runner-canary-20260730-a1" in text
    assert "timelens2-selector-pilot-20260730-a1" in text
    assert "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/" in text
    assert (
        'exec "$PYTHON_BIN" -m trec_video_2026.vqa_experiments.timelens2_selector'
        in text
    )
    assert "remote_prefix_is_empty" in text
    assert "workload_rc" in text
    assert "sync_rc" in text
