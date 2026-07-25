from pathlib import Path

SCRIPT = Path("scripts/run-omni-compatibility.sh")
CONTAINERFILE = Path("containers/omni-rocm.Containerfile")
PHI_CONTAINERFILE = Path("containers/phi4mm-rocm.Containerfile")


def test_runner_pins_image_and_disables_network():
    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        "localhost/trec-vqa-omni-rocm@"
        "sha256:b1f97c47769a8a2f32a5ac9a18cc62d2aab57ed32c66720775407f6213ebf96f"
    ) in text
    assert (
        "localhost/trec-vqa-phi4mm-rocm@"
        "sha256:17275d29326529d0f0275d0466311f380b623a62d965ba372f931cae34c6d6f5"
    ) in text
    assert "--network=none" in text
    assert "--device=/dev/kfd" in text
    assert "--device=/dev/dri" in text
    assert "--group-add=keep-groups" in text
    assert "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1" in text
    assert "--publish" not in text


def test_runner_mounts_only_inference_source_inputs_and_cache():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "${DATA_ROOT}:/data:ro" not in text
    assert "${DATA_ROOT}/${CONTRACT_REL}:/contract.json:ro" in text
    assert (
        "${DATA_ROOT}/vqa/2025/training/videos_original:"
        "/data/vqa/2025/training/videos_original:ro"
    ) in text
    assert (
        "${DATA_ROOT}/vqa/benchmarks/runs/native-audio-benchmark-v2/audio:"
        "/data/vqa/benchmarks/runs/native-audio-benchmark-v2/audio:ro"
    ) in text
    assert (
        "${DATA_ROOT}/vqa/benchmarks/evidence:"
        "/data/vqa/benchmarks/evidence:ro"
    ) in text
    assert "${HF_CACHE_ROOT}:/cache:ro" in text
    assert "${REPO_ROOT}/src:/workspace/src:ro" in text
    assert "${REPO_ROOT}:/workspace:ro" not in text
    assert "external/nist-vqa" not in text
    assert "${OUTPUT_DIR}:/output:rw" in text


def test_runner_enforces_frozen_contract_and_new_safe_output_dir():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b" in text
    assert "${CONTRACT_REL}.sha256" in text
    assert "--expected-contract-sha256" in text
    assert "RUN_ID" in text and "must contain only" in text
    assert 'mkdir "${OUTPUT_DIR}"' in text
    assert 'mkdir -p "${OUTPUT_DIR}"' not in text


def test_runtime_image_derives_from_pinned_rocm_base():
    text = CONTAINERFILE.read_text(encoding="utf-8")

    assert (
        "FROM localhost/trec-vqa-gate-gemma@"
        "sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa"
    ) in text
    assert "backoff==2.2.1" in text
    assert "peft==0.13.2" in text
    assert "timm==1.0.28" in text


def test_phi_runtime_pins_model_declared_transformers_version():
    text = PHI_CONTAINERFILE.read_text(encoding="utf-8")

    assert (
        "FROM localhost/trec-vqa-gate-gemma@"
        "sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa"
    ) in text
    assert "transformers==4.48.2" in text
    assert "accelerate==1.3.0" in text
    assert "backoff==2.2.1" in text
    assert "peft==0.13.2" in text
