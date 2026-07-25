"""Static safety checks for the immutable Qwen3-Omni quality dstack task."""

from __future__ import annotations

import json
from pathlib import Path

RUN_ID = "qwen3-omni-quality-10-20260725-a1"
REMOTE_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/" + RUN_ID
)
IMAGE = (
    "docker.io/qwenllm/qwen3-omni@"
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
CONTRACT = Path("data/vqa/benchmarks/qwen3-omni-quality-10/contract.json")
CONFIG = Path("dstack/qwen3-omni-quality-10.dstack.yml")
RUNNER = Path("scripts/run-qwen3-omni-quality-dstack.sh")
COLLECTOR = Path("scripts/collect-qwen3-omni-quality-dstack.sh")


def _config() -> str:
    return CONFIG.read_text(encoding="utf-8")


def _file_mappings(config: str) -> list[str]:
    return [
        line.strip()[2:]
        for line in config.splitlines()
        if line.startswith("  - ../")
    ]


def _contract_paths() -> set[str]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = {
        path
        for example in contract["examples"]
        for path in [*example["frame_paths"], example["audio_path"]]
    }
    assert len(paths) == 130
    return paths


def test_quality_config_is_immutable_and_bounded() -> None:
    config = _config()
    assert "type: task" in config
    assert f"name: {RUN_ID}" in config
    assert f"image: {IMAGE}" in config
    assert "ports:" not in config
    assert "backends: [runpod, vastai]" in config
    assert "spot_policy: on-demand" in config
    assert "max_price: 4.50" in config
    assert "max_duration: 1h" in config
    assert "  on_events: [no-capacity]" in config
    assert "  duration: 30m" in config
    assert "    memory: 90GB.." in config
    assert "  memory: 192GB.." in config

    assert f"EXPERIMENT_RUN_ID={RUN_ID}" in config
    assert f"HF_REMOTE_URI={REMOTE_URI}" in config
    assert f"RUNTIME_IMAGE={IMAGE}" in config
    assert f"MODEL_ID={MODEL_ID}" in config
    assert f"MODEL_REVISION={MODEL_REVISION}" in config
    assert "PYTHON_BIN=/usr/bin/python3" in config


def test_quality_config_maps_only_exact_contract_evidence() -> None:
    config = _config()
    mappings = _file_mappings(config)
    evidence_mappings = [mapping for mapping in mappings if ".png:" in mapping or ".wav:" in mapping]
    assert len([mapping for mapping in evidence_mappings if ".png:" in mapping]) == 120
    assert len([mapping for mapping in evidence_mappings if ".wav:" in mapping]) == 10
    expected = {
        f"../data/{relative_path}:/dstack/input/{relative_path}"
        for relative_path in _contract_paths()
    }
    assert set(evidence_mappings) == expected
    contract_mapping = (
        "../data/vqa/benchmarks/qwen3-omni-quality-10/contract.json:"
        "/dstack/input/contract.json"
    )
    assert all(
        "../data/" not in mapping or mapping in expected | {contract_mapping}
        for mapping in mappings
    )
    assert contract_mapping in mappings
    assert "../src/trec_video_2026:/dstack/run/src/trec_video_2026" in mappings
    assert "../scripts/run-qwen3-omni-quality-dstack.sh:/dstack/run/scripts/run-qwen3-omni-quality-dstack.sh" in mappings


def test_quality_runner_binds_identity_and_preserves_workload_failure() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert f'readonly EXPECTED_RUN_ID="{RUN_ID}"' in runner
    assert f'readonly EXPECTED_REMOTE_URI="{REMOTE_URI}"' in runner
    assert IMAGE in runner
    assert MODEL_REVISION in runner
    assert 'readonly PYTHON_BIN="/usr/bin/python3"' in runner
    assert '${PYTHON_BIN:-' not in runner
    assert 'uv=="$UV_VERSION"' not in runner
    assert '"uv==$UV_VERSION"' in runner
    assert 'huggingface-hub==$HF_CLI_VERSION' in runner
    assert 'import importlib.metadata' in runner
    assert 'sysconfig.get_path("scripts")' in runner
    assert 'UVX_BIN="$("$PYTHON_BIN" -' in runner
    assert '"$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION"' in runner
    assert 'command -v uvx' not in runner
    assert '\n  uvx --from ' not in runner
    assert "remote_prefix_is_empty" in runner
    assert '[[ -z "$listing" || "$listing" == "[]" ]]' in runner
    assert "actual_contract_sha256" in runner
    assert '"$actual_contract_sha256" == "$EXPECTED_CONTRACT_SHA256"' in runner
    assert "finalize_and_sync \"$workload_rc\"" in runner
    assert "(( workload_rc == 0 )) || return \"$workload_rc\"" in runner
    assert "-m trec_video_2026.vqa_experiments.qwen3_omni_quality_dstack" in runner
    assert "exec bash" not in runner
    assert "--delete" not in runner


def test_quality_collector_is_immutable_checksum_verified_and_sanitized() -> None:
    collector = COLLECTOR.read_text(encoding="utf-8")
    assert f'readonly EXPECTED_RUN_ID="{RUN_ID}"' in collector
    assert 'readonly PYTHON_BIN="/usr/bin/python3"' in collector
    assert '"uv==$UV_VERSION"' in collector
    assert 'import importlib.metadata' in collector
    assert 'sysconfig.get_path("scripts")' in collector
    assert '"$UVX_BIN" --from "huggingface-hub==$HF_CLI_VERSION"' in collector
    assert 'command -v uvx' not in collector
    assert '\n  uvx --from ' not in collector
    assert "qwen3-omni-quality-10/$RUN_ID" in collector
    assert "experiments/$RUN_ID" in collector
    assert '[[ "$RUN_ID" == "$EXPECTED_RUN_ID" ]]' in collector
    assert 'huggingface-hub==$HF_CLI_VERSION' in collector
    assert "sha256sum -c SHA256SUMS" in collector
    assert "dstack run get" in collector
    assert "trec_video_2026.vqa_experiments.dstack_metadata" in collector
    assert "artifact-hashes.sha256" in collector
    assert 'mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX"' in collector
    assert 'mv "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"' in collector
    assert 'trap \'rm -rf "$TEMP_OUTPUT_DIR"\' EXIT' in collector
    assert "--delete" not in collector
