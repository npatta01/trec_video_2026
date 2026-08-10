from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-vqa-event-map-dstack.sh"
UPLOADER = ROOT / "scripts/upload-vqa-event-map-inputs.sh"
IMAGE = (
    "docker.io/vllm/vllm-openai:v0.20.0@"
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
REVISION = "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
SOURCE_BUNDLE_SHA256 = "3dbfa1e31b4550a9ed1c77a2abdbad55c15d1594c7e2a153ad26dcc2101b138b"
CONTROLLER_SHA256 = "91815765952ede4e7986b09eec64ce2f831536259a1b3e1f12d0127b244be5a8"
PROTOCOL_SHA256 = "bc140b5ace4f1cfce9feb98434ca5276466602ed7e2487261788fc163c9fc4f2"
MODULE_SHA256 = "0deaf7eccafc9225ba533982c19ea894fdc8571c9081db4a52a3ddf11d5ca366"
REMOTE_ROOT = "hf://buckets/Npatta01/trec-video-2026-event-map-20260809"
FAILED_CANARY_ATTEMPT_UUID = "4baaf48c-b690-45f0-b6bf-065b9dc1e527"
CURRENT_CANARY_RUN_ID = "vqa-event-map-canary-a2"
CURRENT_CANARY_ATTEMPT_UUID = "d6d3a33e-84c5-4067-bcea-fe82cdd10e2f"
CONFIGS = [
    ROOT / "dstack/vqa-event-map-canary.dstack.yml",
    *(ROOT / "dstack" / f"vqa-event-map-v-map-chunk-{i:02d}.dstack.yml" for i in range(4)),
    *(ROOT / "dstack" / f"vqa-event-map-av-map-chunk-{i:02d}.dstack.yml" for i in range(4)),
]


def _env(config: dict[str, object]) -> dict[str, str]:
    return dict(str(value).split("=", 1) for value in config["env"])


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_preflight_fixture(root: Path) -> dict[str, str]:
    media_root = root / "media"
    media_root.mkdir(parents=True)
    rows = []
    canary_ids = [f"fixture-{index}" for index in range(4)]
    for index, video_id in enumerate(canary_ids):
        media = media_root / f"{video_id}.mkv"
        media.write_bytes(f"fixture media {index}".encode())
        rows.append(
            {
                "duration_seconds": float(index + 1),
                "media_path": f"media/{video_id}.mkv",
                "media_sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                "question": f"What happens in fixture {index}?",
                "question_stratum": "temporal",
                "video_id": video_id,
            }
        )
    ids_sha256 = hashlib.sha256(("\n".join(canary_ids) + "\n").encode()).hexdigest()
    controller = {
        "schema_version": 1,
        "run_id": "fixture-input",
        "label_policy": "label-free-controller",
        "panel": {
            "row_count": 4,
            "canary_ids": canary_ids,
            "ids_sha256": ids_sha256,
        },
        "rows": rows,
    }
    controller_path = root / "event-map-controller.json"
    controller_path.write_text(
        json.dumps(controller, sort_keys=True) + "\n", encoding="utf-8"
    )
    media_manifest_path = root / "event-map-media-manifest.json"
    media_manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": []}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    request_rows = []
    for row in rows:
        for condition in ("v-map", "av-map"):
            request = {
                "schema_version": 1,
                "run_id": "fixture-input",
                "condition_id": condition,
                "slice_id": "canary",
                **row,
            }
            request["request_sha256"] = _canonical_sha256(request)
            request_rows.append(request)
    request_path = root / "event-map-canary.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in request_rows),
        encoding="utf-8",
    )
    controller_sha256 = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    media_manifest_sha256 = hashlib.sha256(
        media_manifest_path.read_bytes()
    ).hexdigest()
    source_bundle_sha256 = _canonical_sha256(
        {
            "controller_sha256": controller_sha256,
            "media_manifest_sha256": media_manifest_sha256,
        }
    )
    module_sha256 = "1" * 64
    protocol_without_hash = {
        "schema_version": 1,
        "controller_sha256": controller_sha256,
        "media_manifest_sha256": media_manifest_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "inference_module_sha256": module_sha256,
        "request_files": {
            request_path.name: {
                "file_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                "row_count": 8,
            }
        },
    }
    protocol = {
        **protocol_without_hash,
        "protocol_sha256": _canonical_sha256(protocol_without_hash),
    }
    (root / "event-map-protocol.json").write_text(
        json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "source_bundle_sha256": source_bundle_sha256,
        "controller_sha256": controller_sha256,
        "protocol_sha256": protocol["protocol_sha256"],
        "request_sha256": protocol["request_files"][request_path.name][
            "file_sha256"
        ],
        "module_sha256": module_sha256,
        "ids_sha256": ids_sha256,
    }


def _embedded_preflight_source() -> str:
    runner = RUNNER.read_text(encoding="utf-8")
    command = '"$EXPECTED_REQUEST_FILE_SHA256" "$EXPECTED_INFERENCE_MODULE_SHA256" "$EXPECTED_PANEL_IDS_SHA256" <<\'PY\''
    command_index = runner.index(command)
    source_start = runner.index("import hashlib\n", command_index)
    source_end = runner.index("\nPY\n", source_start)
    return runner[source_start:source_end] + "\n"


def test_all_configs_pin_model_runtime_identity_and_exact_slice_caps() -> None:
    assert len(CONFIGS) == 9
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = _env(config)
        assert config["type"] == "task"
        assert config["image"] == IMAGE
        assert config["resources"]["gpu"] == "L40S:1:48GB.."
        assert config["backends"] == ["runpod", "vastai"]
        assert config["spot_policy"] == "on-demand"
        assert config["max_price"] <= 1.0
        assert config["max_duration"] in {"45m", "4h"}
        assert config["retry"]["on_events"] == ["no-capacity"]
        assert "ports" not in config
        assert env["MODEL_ID"] == MODEL
        assert env["MODEL_REVISION"] == REVISION
        assert env["RUNTIME_IMAGE"] == IMAGE
        assert env["RUNTIME_IMAGE_DIGEST"] == IMAGE.split("@", 1)[1]
        assert env["HF_REMOTE_URI"].startswith(REMOTE_ROOT + "-")
        assert env["HF_INPUT_URI"].endswith("/inputs")


def test_canary_is_paired_and_full_configs_are_single_condition_slices() -> None:
    canary = yaml.safe_load(CONFIGS[0].read_text(encoding="utf-8"))
    canary_env = _env(canary)
    assert canary["name"] == CURRENT_CANARY_RUN_ID
    assert canary_env["EXPERIMENT_RUN_ID"] == CURRENT_CANARY_RUN_ID
    assert canary_env["HF_REMOTE_URI"] == f"{REMOTE_ROOT}-{CURRENT_CANARY_RUN_ID}"
    assert canary_env["ATTEMPT_UUID"] == CURRENT_CANARY_ATTEMPT_UUID
    assert canary_env["ATTEMPT_UUID"] != FAILED_CANARY_ATTEMPT_UUID
    assert canary_env["EXPERIMENT_MODE"] == "canary"
    assert canary_env["EVENT_MAP_CONDITIONS"] == "v-map,av-map"
    assert canary["max_duration"] == "45m"
    assert canary["max_price"] <= 1.0

    for path in CONFIGS[1:]:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = _env(config)
        assert env["EXPERIMENT_MODE"] == "full"
        assert env["EVENT_MAP_CONDITIONS"] in {"v-map", "av-map"}
        assert env["EVENT_MAP_SLICE"] in {"00", "01", "02", "03"}
        assert config["max_duration"] == "4h"
        assert config["max_price"] <= 1.0


def test_all_attempts_and_input_artifact_hashes_are_exactly_pinned() -> None:
    attempts: list[str] = []
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = _env(config)
        attempts.append(str(uuid.UUID(env["ATTEMPT_UUID"])))
        assert env["EXPECTED_SOURCE_BUNDLE_SHA256"] == SOURCE_BUNDLE_SHA256
        assert env["EXPECTED_CONTROLLER_SHA256"] == CONTROLLER_SHA256
        assert env["EXPECTED_PROTOCOL_SHA256"] == PROTOCOL_SHA256
        assert env["EXPECTED_INFERENCE_MODULE_SHA256"] == MODULE_SHA256
        assert len(env["EXPECTED_REQUEST_FILE_SHA256"]) == 64
        int(env["EXPECTED_REQUEST_FILE_SHA256"], 16)
    assert len(set(attempts)) == len(CONFIGS)


def test_runner_fails_closed_on_identity_leakage_gpu_and_prefix_races() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "buckets create" in text
    assert "--private" in text
    assert "--exist-ok" not in text
    assert "reserve_remote_prefix" in text
    assert "no-clobber" in text or "atomic" in text
    assert "L40S" in text
    assert "45000" in text
    assert "H100" not in text
    assert "HF_TOKEN" in text
    assert "unset HF_TOKEN" in text
    assert "label" in text.lower()
    assert "SHA256SUMS" in text
    assert "terminal-status.json" in text
    assert "payload_checksum" in text
    assert "attempt_uuid" in text
    assert "source_bundle_sha256" in text
    assert "controller_sha256" in text
    assert "protocol_sha256" in text
    assert "nvidia-smi" in text
    assert "--allowed-local-media-path" in text
    assert "127.0.0.1" in text
    assert "public" in text.lower()
    assert "MAX_OUTPUT_TOKENS=768" in text
    assert "--video-pruning-rate 0.5" in text
    assert "--kv-cache-dtype fp8" in text
    assert "--reasoning-parser nemotron_v3" in text
    assert "uuid.uuid4" not in text
    for name in (
        "EXPECTED_SOURCE_BUNDLE_SHA256",
        "EXPECTED_CONTROLLER_SHA256",
        "EXPECTED_PROTOCOL_SHA256",
        "EXPECTED_REQUEST_FILE_SHA256",
        "EXPECTED_INFERENCE_MODULE_SHA256",
        "EXPECTED_METADATA_BUNDLE_SHA256",
    ):
        assert name in text


def test_runner_bootstraps_before_hf_use_and_normalizes_only_requested_media() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    main = text[text.index("main() {") :]
    assert main.index("bootstrap_runtime") < main.index("download_inputs")
    assert "request_rows =" in text
    assert "requested_ids =" in text
    assert "for row in controller[\"rows\"]:" not in text[
        text.index("normalize_inputs() {") : text.index("reserve_remote_prefix() {")
    ]


def test_embedded_preflight_writes_artifact_for_a_valid_bundle(tmp_path: Path) -> None:
    pins = _write_preflight_fixture(tmp_path)
    attempt_uuid = "e9f7dd2c-1d03-49cb-99e6-72d7289c36b7"
    output_path = tmp_path / "preflight.json"

    result = subprocess.run(
        [
            "python3",
            "-",
            str(tmp_path),
            str(tmp_path / "event-map-canary.jsonl"),
            "canary",
            "v-map,av-map",
            "",
            attempt_uuid,
            str(output_path),
            pins["source_bundle_sha256"],
            pins["controller_sha256"],
            pins["protocol_sha256"],
            pins["request_sha256"],
            pins["module_sha256"],
            pins["ids_sha256"],
        ],
        input=_embedded_preflight_source(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "run_id": "fixture-input",
        "attempt_uuid": attempt_uuid,
        "source_bundle_sha256": pins["source_bundle_sha256"],
        "controller_sha256": pins["controller_sha256"],
        "protocol_sha256": pins["protocol_sha256"],
        "request_file_sha256": pins["request_sha256"],
        "row_count": 8,
        "condition_id": "v-map,av-map",
        "slice_id": "",
    }


def test_runner_publishes_verified_payload_before_terminal_markers() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "failure-status.json" in text
    publisher = text[
        text.index("publish_and_verify_payload() {") : text.index(
            "finalize_failure() {"
        )
    ]
    assert 'buckets sync "$OUTPUT_DIR"' in publisher
    assert "sha256sum -c SHA256SUMS" in publisher
    success = text[text.index("finalize_success() {") : text.index("on_exit() {")]
    assert success.index("publish_and_verify_payload") < success.index(
        'buckets cp "$OUTPUT_DIR/terminal-status.json"'
    )
    failure = text[text.index("finalize_failure() {") : text.index("finalize_success() {")]
    assert failure.index("publish_and_verify_payload") < failure.index(
        'buckets cp "$OUTPUT_DIR/failure-status.json"'
    )


def test_shell_syntax_and_dstack_preview_commands_are_available_without_submission() -> None:
    result = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["commands"][0].strip().startswith("set -euo pipefail")
        assert "dstack apply" not in RUNNER.read_text(encoding="utf-8")


def test_input_uploader_builds_metadata_and_streams_exact_media_without_copying_dataset() -> None:
    text = UPLOADER.read_text(encoding="utf-8")
    assert "build_vqa_event_map_bundle.py" in text
    assert "qwen35_vqa2025_475_label_free.json" in text
    assert "frame-evidence.json" in text
    assert "EXPECTED_DATA_TARGET" in text
    assert "readlink -f" in text
    assert "buckets sync" in text
    assert "--include" in text
    assert 'MEDIA_FILTERS+=(--exclude "*")' not in text
    assert "REMOTE_MEDIA_COUNT" in text
    assert '[[ "$REMOTE_MEDIA_COUNT" -eq 475 ]]' in text
    assert "SHA256SUMS" in text
    assert "cp --" not in text
    assert "HF_TOKEN must remain unset" in text
    result = subprocess.run(
        ["bash", "-n", str(UPLOADER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
