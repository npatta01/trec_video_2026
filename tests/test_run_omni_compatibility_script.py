from pathlib import Path

SCRIPT = Path("scripts/run-omni-compatibility.sh")


def test_runner_pins_image_and_disables_network():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "localhost/trec-vqa-gate-gemma@sha256:" in text
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
