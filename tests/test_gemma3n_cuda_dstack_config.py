from pathlib import Path


CONFIG = Path("dstack/gemma3n-quality.dstack.yml")
RUNNER = Path("scripts/run-phi4mm-cuda-dstack.sh")

MODEL_REVISION = "c1221e9c62e34a43ab7ffacd1be0ea71f126ef10"
IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
CONTRACT_SHA256 = "b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b"
RUN_ID = "gemma3n-cuda-quality-3-20260724-a1"


def test_gemma3n_dstack_runtime_is_immutable_and_bounded():
    text = CONFIG.read_text(encoding="utf-8")

    assert f"image: {IMAGE}" in text
    assert "MODEL_KEY=gemma3n" in text
    assert MODEL_REVISION in text
    assert CONTRACT_SHA256 in text
    assert ":latest" not in text
    assert "type: task" in text
    assert "ports:" not in text
    assert "vendor: nvidia" in text
    assert "name: [A40, L40, L40S, RTX6000Ada, A100, H100, H200]" in text
    assert "memory: 40GB.." in text
    assert "disk: 100GB" in text
    assert "backends: [runpod, vastai]" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 1.50" in text
    assert "max_duration: 1h" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 20m" in text


def test_gemma3n_dstack_uploads_only_the_frozen_panel():
    text = CONFIG.read_text(encoding="utf-8")

    assert text.count(".png:/dstack/input/vqa/benchmarks/evidence/") == 36
    assert text.count(".wav:/dstack/input/vqa/benchmarks/runs/") == 3
    assert "/dstack/input/contract.json" in text
    assert "videos_original" not in text
    assert "annotations" not in text


def test_gemma3n_runner_pins_native_dependencies_and_preflights_import():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'GEMMA_TRANSFORMERS_VERSION="5.14.1"' in text
    assert 'TIMM_VERSION="1.0.28"' in text
    assert 'LIBROSA_VERSION="0.11.0"' in text
    assert '"timm==$TIMM_VERSION"' in text
    assert '"librosa==$LIBROSA_VERSION"' in text
    assert 'timm.is_model("mobilenetv5_300m_enc")' in text
    assert "Gemma3nForConditionalGeneration" in text
    assert "AutoProcessor.from_pretrained" in text
    assert "local_files_only=True" in text
    assert "gemma_messages" in text
    assert "processor.apply_chat_template" in text
    assert '"input_features"' in text
    assert '"pixel_values"' in text
    assert '--model "$MODEL_KEY"' in text


def test_gemma3n_results_use_an_immutable_private_bucket_prefix():
    combined = CONFIG.read_text(encoding="utf-8") + RUNNER.read_text(encoding="utf-8")
    expected = (
        f"hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/{RUN_ID}"
    )

    assert expected in combined
    assert RUN_ID in combined
    assert "HF_TOKEN=${{ secrets.hf_token }}" in combined
