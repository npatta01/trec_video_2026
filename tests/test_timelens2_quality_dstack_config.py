"""Safety tests for the immutable TimeLens2 selector quality task."""

from pathlib import Path

CONFIG = Path("dstack/timelens2-selector-quality-10.dstack.yml")
RUNNER = Path("scripts/run-timelens2-selector-quality-dstack.sh")
RUN_ID = "timelens2-selector-quality-20260801-a2"
IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
GPU_POOL = "A40,A6000,L40,L40S,RTX6000Ada,A100,H100,H200"
SELECTOR_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
ANSWER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_SHA256 = "d2cfec431652bcf7c6c83be59a4932fc0ed75016d6a3dc973e49cebdd9ed0241"
REMOTE = (
    "hf://buckets/Npatta01/trec_video_2026/experiments/"
    f"{RUN_ID}"
)


def test_quality_task_is_exact_image_cuda_and_budget_bounded() -> None:
    text = CONFIG.read_text(encoding="utf-8")

    assert f"name: {RUN_ID}" in text
    assert f"image: {IMAGE}" in text
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert f"EXPERIMENT_RUN_ID={RUN_ID}" in text
    assert f"HF_REMOTE_URI={REMOTE}" in text
    assert f"SELECTOR_MODEL_REVISION={SELECTOR_REVISION}" in text
    assert f"ANSWER_MODEL_REVISION={ANSWER_REVISION}" in text
    assert f"SOURCE_SHA256={SOURCE_SHA256}" in text
    assert f"gpu: {GPU_POOL}:1:48GB.." in text
    assert "cpu: 8.." in text
    assert "memory: 96GB.." in text
    assert "disk: 100GB.." in text
    assert "fleets: [gpu-marketplace]" in text
    assert "backends: [runpod, vastai]" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 1.00" in text
    assert "max_duration: 2h" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 20m" in text
    assert "amd" not in text.lower()
    assert "rocm" not in text.lower()


def test_quality_task_mounts_code_only_and_never_mounts_labels_or_media() -> None:
    text = CONFIG.read_text(encoding="utf-8")

    assert "../scripts/run-timelens2-selector-quality-dstack.sh:/dstack/run/" in text
    assert "../src/trec_video_2026:/dstack/run/src/trec_video_2026" in text
    assert "adjudicated.json" not in text
    assert "temporal-required-moment" not in text
    assert "temporal-relevant-scene-oracle-10-r2" not in text
    assert ".mkv:" not in text
    assert ".timelens2-private" not in text


def test_quality_runner_pins_both_models_runtime_and_hf_cli() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert f'EXPECTED_RUNTIME_IMAGE="{IMAGE}"' in text
    assert f'EXPECTED_RUN_ID="{RUN_ID}"' in text
    assert f'EXPECTED_REMOTE_URI="{REMOTE}"' in text
    assert f'EXPECTED_SELECTOR_REVISION="{SELECTOR_REVISION}"' in text
    assert f'EXPECTED_ANSWER_REVISION="{ANSWER_REVISION}"' in text
    assert f'EXPECTED_SOURCE_SHA256="{SOURCE_SHA256}"' in text
    assert 'TRANSFORMERS_VERSION="4.57.6"' in text
    assert 'QWEN_VL_UTILS_VERSION="0.0.14"' in text
    assert 'PYAV_VERSION="16.0.1"' in text
    assert 'IMAGEIO_FFMPEG_VERSION="0.6.0"' in text
    assert 'HF_CLI_VERSION="1.26.0"' in text
    assert 'unset HF_TOKEN' in text
    assert 'torch.version.hip is None' in text
    assert 'hf buckets sync' not in text
    assert 'hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR"' in text
    assert 'hf_cli buckets sync "$OUTPUT_DIR" "$HF_OUTPUT_URI"' in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "timelens2_quality prepare" in text
    assert "timelens2_quality run" in text
    assert "adjudicated.json" not in text
    assert "temporal-required-moment" not in text
    assert "diagnose" not in text.lower()


def test_quality_runner_preserves_workload_and_sync_failures() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'workload_rc=$?' in text
    assert 'sync_rc=$?' in text
    assert '(( workload_rc == 0 )) || return "$workload_rc"' in text
    assert 'return "$sync_rc"' in text
    assert 'find . -type f ! -name SHA256SUMS -print0' in text
