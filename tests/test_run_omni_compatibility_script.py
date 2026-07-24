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
