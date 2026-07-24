from pathlib import Path

SCRIPT = Path("scripts/run-omni-compatibility.sh")
CONTAINERFILE = Path("containers/omni-rocm.Containerfile")


def test_runner_pins_image_and_disables_network():
    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        "localhost/trec-vqa-omni-rocm@"
        "sha256:cbda16214c953bcd2bd2a7bf65d26561ac406b0f1fbab2f719b25baab1d757f4"
    ) in text
    assert "--network=none" in text
    assert "--device=/dev/kfd" in text
    assert "--device=/dev/dri" in text
    assert "--group-add=keep-groups" in text
    assert "--publish" not in text


def test_runner_mounts_inputs_and_cache_read_only():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "${DATA_ROOT}:/data:ro" in text
    assert "${HF_CACHE_ROOT}:/cache:ro" in text
    assert "${REPO_ROOT}:/workspace:ro" in text
    assert "${OUTPUT_DIR}:/output:rw" in text


def test_runtime_image_derives_from_pinned_rocm_base():
    text = CONTAINERFILE.read_text(encoding="utf-8")

    assert (
        "FROM localhost/trec-vqa-gate-gemma@"
        "sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa"
    ) in text
    assert "backoff==2.2.1" in text
    assert "peft==0.13.2" in text
