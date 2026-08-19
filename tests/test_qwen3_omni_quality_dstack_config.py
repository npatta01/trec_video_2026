"""Static safety checks for the immutable Qwen3-Omni quality dstack task."""

from __future__ import annotations

import json
import io
import os
import subprocess
import tarfile
from pathlib import Path

RUN_ID = "qwen3-omni-quality-10-20260725-a1"
NEUTRAL_RUN_ID = "qwen3-omni-quality-10-neutral-20260725-a2"
REMOTE_URI = (
    "hf://buckets/Npatta01/trec_video_2026/experiments/" + RUN_ID
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
COLLECTOR_LIB = Path("scripts/qwen3-omni-quality-collector-lib.sh")
NEUTRAL_CONFIG = Path("dstack/qwen3-omni-quality-10-neutral.dstack.yml")
NEUTRAL_RUNNER = Path("scripts/run-qwen3-omni-quality-10-neutral-dstack.sh")
NEUTRAL_COLLECTOR = Path("scripts/collect-qwen3-omni-quality-10-neutral-dstack.sh")
DSTACKIGNORE = Path(".dstackignore")
MAX_MAPPED_SOURCE_ARCHIVE_BYTES = 2 * 1024 * 1024
OVERSIZED_FRAME = (
    "vqa/benchmarks/evidence/"
    "0b20cdcb075f8a2fcd3640a0a40356146796859051c4c203b9acb8c051ed68a6/"
    "frame-11.png"
)
CHUNK_DIRECTORY = "vqa/benchmarks/qwen3-omni-quality-10/transport/frame-11"
CHUNK_NAMES = ("part-0000.bin", "part-0001.bin", "part-0002.bin")


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


def _source_archive_size(path: Path) -> int:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        tar.add(path, arcname=path.name)
    return archive.tell()


def test_dstack_archive_excludes_only_safe_non_runtime_directories() -> None:
    """Catches rendered reports inflating the archive or runtime inputs being ignored."""
    ignored_roots = {
        line.removeprefix("/").removesuffix("/")
        for line in DSTACKIGNORE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert ignored_roots == {
        ".agents",
        ".ruff_cache",
        ".superpowers",
        "docs",
        "plans",
        "reports",
        "tests",
    }

    required_paths = {
        DSTACKIGNORE,
        CONFIG,
        RUNNER,
        *Path("src/trec_video_2026").rglob("*"),
    }
    for mapping in _file_mappings(_config()):
        source = mapping.split(":", 1)[0]
        required_paths.add(Path(os.path.normpath(CONFIG.parent / source)))
    required_files = {path for path in required_paths if path.is_file()}

    assert CONFIG in required_files
    assert RUNNER in required_files
    assert CONTRACT in required_files
    assert len({path for path in required_files if path.suffix == ".png"}) == 119
    assert len({path for path in required_files if path.suffix == ".bin"}) == 3
    assert len({path for path in required_files if path.suffix == ".wav"}) == 10
    assert all(path.parts[0] not in ignored_roots for path in required_files)


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
    direct_evidence_mappings = [
        mapping for mapping in mappings if ".png:" in mapping or ".wav:" in mapping
    ]
    chunk_mappings = [mapping for mapping in mappings if ".bin:" in mapping]
    assert len([mapping for mapping in direct_evidence_mappings if ".png:" in mapping]) == 119
    assert len([mapping for mapping in direct_evidence_mappings if ".wav:" in mapping]) == 10
    assert len(chunk_mappings) == len(CHUNK_NAMES)
    expected_direct = {
        f"../data/{relative_path}:/dstack/input/{relative_path}"
        for relative_path in _contract_paths() - {OVERSIZED_FRAME}
    }
    expected_chunks = {
        f"../data/{CHUNK_DIRECTORY}/{name}:/dstack/input/.transport/frame-11/{name}"
        for name in CHUNK_NAMES
    }
    assert set(direct_evidence_mappings) == expected_direct
    assert set(chunk_mappings) == expected_chunks
    contract_mapping = (
        "../data/vqa/benchmarks/qwen3-omni-quality-10/contract.json:"
        "/dstack/input/contract.json"
    )
    assert all(
        "../data/" not in mapping
        or mapping in expected_direct | expected_chunks | {contract_mapping}
        for mapping in mappings
    )
    assert contract_mapping in mappings
    assert "../src/trec_video_2026:/dstack/run/src/trec_video_2026" in mappings
    assert "../scripts/run-qwen3-omni-quality-dstack.sh:/dstack/run/scripts/run-qwen3-omni-quality-dstack.sh" in mappings

    assert OVERSIZED_FRAME in _contract_paths()
    assert len({path for path in _contract_paths() if path.endswith(".png")}) == 120
    assert len({path for path in _contract_paths() if path.endswith(".wav")}) == 10


def test_every_quality_mapping_source_archive_is_below_server_limit() -> None:
    """Catches any single mapped source exceeding the server's 2 MiB cap."""
    for mapping in _file_mappings(_config()):
        source = Path(os.path.normpath(CONFIG.parent / mapping.split(":", 1)[0]))
        assert source.exists(), source
        assert _source_archive_size(source) < MAX_MAPPED_SOURCE_ARCHIVE_BYTES, source


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
    assert "qwen3_omni_quality_transport reconstruct" in runner
    assert '--contract "$CONTRACT_PATH"' in runner
    assert runner.index("qwen3_omni_quality_transport reconstruct") < runner.index(
        'chmod -R a-w "$DATA_ROOT"'
    )
    assert "--delete" not in runner


def test_quality_collector_is_immutable_checksum_verified_and_sanitized() -> None:
    collector = COLLECTOR.read_text(encoding="utf-8")
    collector_lib = COLLECTOR_LIB.read_text(encoding="utf-8")
    combined = collector + collector_lib
    assert f'readonly EXPECTED_RUN_ID="{RUN_ID}"' in collector
    assert 'readonly PYTHON_BIN="/usr/bin/python3"' in collector
    assert '"$PYTHON_BIN" -m pip' not in combined
    assert "--break-system-packages" not in combined
    assert "resolve_pinned_uvx" in collector
    assert "qwen3-omni-quality-10/$RUN_ID" in collector
    assert "experiments/$RUN_ID" in collector
    assert '[[ "$RUN_ID" == "$EXPECTED_RUN_ID" ]]' in collector
    assert 'huggingface-hub==$HF_CLI_VERSION' in combined
    assert '"$UVX_LAUNCHER" --from "uv==$UV_VERSION" uvx "$@"' in collector_lib
    assert "command -v uvx" in collector_lib
    assert "sha256sum -c SHA256SUMS" in collector
    assert "dstack run get" in collector
    assert "trec_video_2026.vqa_experiments.dstack_metadata" in collector
    assert "artifact-hashes.sha256" in collector
    assert 'mktemp -d "${OUTPUT_DIR}.tmp.XXXXXX"' in collector
    assert 'publish_verified_output "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"' in collector
    assert 'trap \'rm -rf "$TEMP_OUTPUT_DIR"\' EXIT' in collector
    assert "--delete" not in collector


def test_quality_collector_bootstrap_uses_pinned_inner_uvx_without_system_python(
    tmp_path: Path,
) -> None:
    """Catches PEP 668-unsafe installation or launching HF with the outer uvx."""
    launcher = tmp_path / "uvx"
    log = tmp_path / "uvx.log"
    launcher.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >>"$UVX_TEST_LOG"
case "$*" in
  "--from uv==0.11.3 uvx --version")
    echo "uvx 0.11.3 (test-platform)"
    ;;
  "--from uv==0.11.3 uvx --from huggingface-hub==1.24.0 hf version")
    echo "hf 1.24.0"
    ;;
  *)
    exit 9
    ;;
esac
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -euo pipefail
source "$1"
UV_VERSION="0.11.3"
HF_CLI_VERSION="1.24.0"
UVX_LAUNCHER=""
resolve_pinned_uvx
hf_cli version
''',
            "collector-bootstrap-test",
            str(COLLECTOR_LIB.resolve()),
        ],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "UVX_TEST_LOG": str(log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "--from uv==0.11.3 uvx --version",
        "--from uv==0.11.3 uvx --from huggingface-hub==1.24.0 hf version",
    ]


def test_quality_collector_publish_race_fails_without_nesting_or_leaking_temp(
    tmp_path: Path,
) -> None:
    """Catches a concurrent destination creation being treated as publication."""
    temporary_output = tmp_path / "run.tmp.ABC123"
    output = tmp_path / "run"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r'''
set -euo pipefail
source "$1"
TEMP_OUTPUT_DIR="$2"
OUTPUT_DIR="$3"
mkdir "$TEMP_OUTPUT_DIR"
touch "$TEMP_OUTPUT_DIR/verified-artifact"
trap 'rm -rf "$TEMP_OUTPUT_DIR"' EXIT
mv() {
  mkdir "$OUTPUT_DIR"
  command mv "$@"
}
publish_verified_output "$TEMP_OUTPUT_DIR" "$OUTPUT_DIR"
trap - EXIT
''',
            "collector-race-test",
            str(COLLECTOR_LIB.resolve()),
            str(temporary_output),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to publish over existing output" in result.stderr
    assert output.is_dir()
    assert not (output / temporary_output.name).exists()
    assert not temporary_output.exists()


def test_neutral_a2_scaffolding_keeps_a1_immutable_and_binds_no_repo_submission():
    """Catches an a2 configuration that can reuse a1 identity or its prompt profile."""
    neutral_config = NEUTRAL_CONFIG.read_text(encoding="utf-8")
    neutral_runner = NEUTRAL_RUNNER.read_text(encoding="utf-8")
    neutral_collector = NEUTRAL_COLLECTOR.read_text(encoding="utf-8")
    a1_config = CONFIG.read_text(encoding="utf-8")

    neutral_uri = (
        "hf://buckets/Npatta01/trec_video_2026/experiments/"
        + NEUTRAL_RUN_ID
    )
    assert f"name: {NEUTRAL_RUN_ID}" in neutral_config
    assert f"EXPERIMENT_RUN_ID={NEUTRAL_RUN_ID}" in neutral_config
    assert f"HF_REMOTE_URI={neutral_uri}" in neutral_config
    assert f'readonly EXPECTED_RUN_ID="{NEUTRAL_RUN_ID}"' in neutral_runner
    assert f'readonly EXPECTED_REMOTE_URI="{neutral_uri}"' in neutral_runner
    assert f'readonly EXPECTED_RUN_ID="{NEUTRAL_RUN_ID}"' in neutral_collector
    assert "--instruction-profile neutral" in neutral_runner
    assert "max_price: 4.50" in neutral_config
    assert "max_duration: 1h" in neutral_config
    assert "qwen3_omni_quality_transport reconstruct" in neutral_runner
    assert "sha256sum -c SHA256SUMS" in neutral_collector
    assert "publish_verified_output" in neutral_collector
    assert RUN_ID not in neutral_config + neutral_runner + neutral_collector
    assert NEUTRAL_RUN_ID not in a1_config

    normalized_a2 = neutral_config.replace(NEUTRAL_RUN_ID, RUN_ID).replace(
        "run-qwen3-omni-quality-10-neutral-dstack.sh",
        "run-qwen3-omni-quality-dstack.sh",
    )
    assert normalized_a2 == a1_config
