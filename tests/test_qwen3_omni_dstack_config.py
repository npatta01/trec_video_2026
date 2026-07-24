from pathlib import Path
import re

import pytest

from trec_video_2026.vqa_experiments.dstack_metadata import sanitize_dstack_run
from trec_video_2026.vqa_experiments.qwen3_omni_dstack import (
    CONTRACT_SHA256,
    GENERATION_SEED,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
    RUNTIME_IMAGE_DIGEST,
    attest_qwen_inputs,
    build_messages,
    normalize_generation_output,
    validate_panel_identity,
)

CONFIG = Path("dstack/qwen3-omni-compatibility.dstack.yml")


def _example() -> dict:
    return {
        "video_id": "NN7G28sBOlY",
        "operational_tag": "lexical_speech",
        "question": "What is said?",
        "frame_paths": [f"frames/frame-{index:02d}.png" for index in range(1, 13)],
        "audio_path": "audio/NN7G28sBOlY.wav",
    }


def test_qwen_runtime_uses_immutable_model_and_image_identifiers():
    text = CONFIG.read_text(encoding="utf-8")
    top_level_images = [line for line in text.splitlines() if line.startswith("image:")]

    assert MODEL_ID == "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    assert MODEL_REVISION == "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
    assert RUNTIME_IMAGE_DIGEST == (
        "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
    )
    assert RUNTIME_IMAGE.endswith(f"@{RUNTIME_IMAGE_DIGEST}")
    assert MODEL_REVISION in text
    assert top_level_images == [f"image: {RUNTIME_IMAGE}"]
    assert ":latest" not in text


def test_dstack_task_has_fixed_resource_and_cost_envelope():
    text = CONFIG.read_text(encoding="utf-8")

    assert "type: task" in text
    assert "ports:" not in text
    assert "memory: 90GB.." in text
    assert "name: [H100, H200, B200, RTXPRO6000WS]" in text
    assert "backends: [runpod, vastai]" in text
    assert "spot_policy: on-demand" in text
    assert "max_price: 4.50" in text
    assert "max_duration: 2h" in text
    assert "on_events: [no-capacity]" in text
    assert "duration: 30m" in text


def test_dstack_task_uploads_only_frozen_inputs_and_syncs_private_results():
    text = CONFIG.read_text(encoding="utf-8")
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")
    combined = text + runner
    files_section = text.split("files:\n", 1)[1].split("\ncommands:", 1)[0]
    mappings = [
        line.removeprefix("  - ")
        for line in files_section.splitlines()
        if line.startswith("  - ")
    ]

    assert CONTRACT_SHA256 in text
    assert "/dstack/input/contract.json" in text
    assert "/dstack/input/vqa/benchmarks/evidence/" in text
    assert text.count(".png:/dstack/input/vqa/benchmarks/evidence/") == 36
    assert len(mappings) == 42
    assert mappings[:3] == [
        "../scripts/run-qwen3-omni-dstack.sh:"
        "/dstack/run/scripts/run-qwen3-omni-dstack.sh",
        "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
        "../data/vqa/benchmarks/omni-compatibility-3/contract.json:"
        "/dstack/input/contract.json",
    ]
    frame_pattern = re.compile(
        r"\.\./data/vqa/benchmarks/evidence/[0-9a-f]{64}/frame-\d{2}\.png:"
        r"/dstack/input/vqa/benchmarks/evidence/[0-9a-f]{64}/frame-\d{2}\.png"
    )
    assert all(frame_pattern.fullmatch(mapping) for mapping in mappings[3:39])
    assert all(
        mapping.startswith(
            "../data/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        )
        and ":/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        in mapping
        and mapping.endswith(".wav")
        for mapping in mappings[39:]
    )
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NN7G28sBOlY.wav"
    ) in text
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NUmb2P3Bzu8.wav"
    ) in text
    assert (
        "/dstack/input/vqa/benchmarks/runs/native-audio-benchmark-v2/audio/"
        "NkXPPJoZ-h8.wav"
    ) in text
    assert 'chmod -R a-w "$DATA_ROOT"' in combined
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert (
        "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
        "experiments/qwen3-omni-compatibility-3-20260724-a3"
    ) in combined
    assert "name: qwen3-omni-compatibility-3-20260724-a3" in text
    assert 'readonly EXPECTED_RUN_ID="qwen3-omni-compatibility-3-20260724-a3"' in runner
    assert 'uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"' in runner
    assert "--break-system-packages" not in runner
    assert "hf_cli buckets sync" in runner
    assert "--delete" not in combined


def test_dstack_wrapper_syncs_sanitized_artifacts_after_preflight_failure():
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert "preflight_rc=$?" in runner
    assert "finalize_and_sync" in runner
    assert 'return "$preflight_rc"' in runner
    assert "preflight.stdout.json" in runner
    assert "preflight.stderr" not in runner


def test_dstack_wrapper_fails_closed_on_all_absolute_io_paths():
    runner = Path("scripts/run-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert '[[ "$CONTRACT_PATH" == /dstack/input/contract.json ]]' in runner
    assert '[[ "$DATA_ROOT" == /dstack/input ]]' in runner
    assert '[[ "$OUTPUT_DIR" == /dstack/output ]]' in runner
    assert 'chmod -R a-w "$DATA_ROOT"' in runner
    assert "chmod -R a-w /dstack/input" not in runner


def test_qwen_message_contains_all_frozen_modalities_in_one_turn():
    messages = build_messages(_example(), Path("/dstack/input"))

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [item["type"] for item in content].count("image") == 12
    assert [item["type"] for item in content].count("audio") == 1
    assert [item["type"] for item in content].count("text") == 1
    assert "JSON object" in content[-1]["text"]


def test_thinking_output_normalizes_only_a_terminal_json_object():
    raw = (
        "<think>I should inspect both modalities.</think>\n"
        '{"answer":"speaker raises her voice","evidence":"loud speech and gesture"}'
    )

    normalized = normalize_generation_output(raw)

    assert normalized == (
        '{"answer": "speaker raises her voice", "evidence": "loud speech and gesture"}'
    )
    assert normalize_generation_output("reasoning without a final object") == (
        "reasoning without a final object"
    )
    assert normalize_generation_output('{"answer":"x","evidence":"y"} trailing') == (
        '{"answer":"x","evidence":"y"} trailing'
    )


def test_generation_policy_uses_checkpoint_config_with_bounded_seeded_override():
    source = Path("src/trec_video_2026/vqa_experiments/qwen3_omni_dstack.py").read_text(
        encoding="utf-8"
    )

    assert MAX_NEW_TOKENS == 2048
    assert GENERATION_SEED == 20260724
    assert "max_new_tokens=MAX_NEW_TOKENS" in source
    assert "torch.manual_seed(GENERATION_SEED)" in source
    assert "do_sample=True" in source
    assert "do_sample=False" not in source
    assert "inputs.to(device).to(model.dtype)" in source


def test_panel_identity_fails_closed_on_hash_or_video_mismatch():
    contract = {
        "examples": [
            {"video_id": "NN7G28sBOlY"},
            {"video_id": "NUmb2P3Bzu8"},
            {"video_id": "NkXPPJoZ-h8"},
        ]
    }
    validate_panel_identity(contract, CONTRACT_SHA256)

    with pytest.raises(ValueError, match="contract digest"):
        validate_panel_identity(contract, "0" * 64)

    contract["examples"][2]["video_id"] = "unexpected"
    with pytest.raises(ValueError, match="video IDs"):
        validate_panel_identity(contract, CONTRACT_SHA256)


def test_qwen_attestation_uses_cardinality_tensors_and_first_forward():
    import array

    class Tensor:
        def __init__(self, values, shape):
            self.values = array.array("f", values)
            self.shape = shape

        def tobytes(self):
            return self.values.tobytes()

    inputs = {
        "pixel_values": Tensor([1.0, 2.0], (2,)),
        "image_grid_thw": Tensor(range(36), (12, 3)),
        "input_features": Tensor([3.0, 4.0], (1, 2)),
        "feature_attention_mask": Tensor([1.0, 1.0], (1, 2)),
        "input_ids": Tensor([1.0, 2.0], (1, 2)),
    }

    attestation = attest_qwen_inputs(
        inputs,
        first_forward_input_keys=set(inputs),
    )

    assert attestation["same_request"] is True
    assert attestation["same_forward_pass"] is True
    assert attestation["generation_call_count"] == 1
    assert attestation["vision_element_count"] == 12
    assert attestation["audio_element_count"] == 1
    assert set(attestation["vision_input_keys"]) == {
        "image_grid_thw",
        "pixel_values",
    }
    assert set(attestation["audio_input_keys"]) == {
        "feature_attention_mask",
        "input_features",
    }


def test_collector_targets_only_ignored_canonical_directory_and_allowlists_metadata():
    collector = Path("scripts/collect-qwen3-omni-dstack.sh").read_text(encoding="utf-8")

    assert "data/vqa/benchmarks/qwen3-omni-compatibility-3/$RUN_ID" in collector
    assert "dstack run get" in collector
    assert "sha256sum -c SHA256SUMS" in collector
    assert "artifact-hashes.sha256" in collector
    assert "ssh_key" not in collector
    assert "--delete" not in collector


def test_dstack_metadata_uses_exact_provisioned_offer_shape():
    raw = {
        "id": "run-uuid",
        "run_spec": {"run_name": "qwen3-omni-a1"},
        "status": "terminated",
        "status_message": "terminated",
        "termination_reason": "stopped_by_user",
        "submitted_at": "2026-07-24T20:03:52.493209+00:00",
        "cost": 0.1844,
        "latest_job_submission": {
            "status": "terminated",
            "termination_reason": "terminated_by_user",
            "submitted_at": "2026-07-24T20:03:52.495154+00:00",
            "finished_at": "2026-07-24T20:06:23.738912+00:00",
            "exit_status": None,
            "job_provisioning_data": {
                "backend": "runpod",
                "region": "CA-MTL-4",
                "price": 4.39,
                "instance_type": {
                    "name": "NVIDIA H200",
                    "resources": {
                        "cpus": 28,
                        "memory_mib": 257024,
                        "gpus": [
                            {
                                "name": "H200",
                                "memory_mib": 144384,
                                "vendor": "nvidia",
                            }
                        ],
                        "disk": {"size_mib": 204800},
                        "spot": False,
                    },
                },
            },
            "job_runtime_data": {
                "offer": {
                    "backend": "runpod",
                    "region": "CA-MTL-4",
                    "price": 4.39,
                }
            },
        },
    }

    sanitized = sanitize_dstack_run(raw, run_id="qwen3-omni-a1")

    assert sanitized["backend"] == "runpod"
    assert sanitized["region"] == "CA-MTL-4"
    assert sanitized["price"] == 4.39
    assert sanitized["gpu_name"] == "H200"
    assert sanitized["gpu_vendor"] == "nvidia"
    assert sanitized["gpu_count"] == 1
    assert sanitized["cpu_count"] == 28
    assert sanitized["gpu_memory_mib"] == 144384
    assert sanitized["host_memory_mib"] == 257024
    assert sanitized["disk_mib"] == 204800
    assert sanitized["elapsed_seconds"] == pytest.approx(151.245703)
    assert set(sanitized).isdisjoint({"user", "backend_data", "public_ip"})

    incomplete = {**raw, "cost": None}
    with pytest.raises(ValueError, match="run.cost"):
        sanitize_dstack_run(incomplete, run_id="qwen3-omni-a1")

    mismatched = {
        **raw,
        "latest_job_submission": {
            **raw["latest_job_submission"],
            "job_runtime_data": {
                "offer": {
                    "backend": "runpod",
                    "region": "US-TX-3",
                    "price": 4.39,
                }
            },
        },
    }
    with pytest.raises(ValueError, match="region differs"):
        sanitize_dstack_run(mismatched, run_id="qwen3-omni-a1")
