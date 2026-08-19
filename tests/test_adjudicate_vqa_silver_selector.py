from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATOR_PATH = ROOT / "scripts/adjudicate_vqa_silver_selector.py"
CANARY_CONFIG = ROOT / "dstack/vqa-silver-oracle-selector-canary.dstack.yml"
RUN_ID = "vqa-silver-oracle-election-20260809-b4"
RUN_NAME = "vqa-silver-selector-canary-20260809-b4"
SLICE_ID = "canary-b4"
EXECUTION_RUN_ID = f"{RUN_ID}-{SLICE_ID}"
EXECUTION_ATTEMPT_ID = "2d1cb945-31a7-4ad1-8462-b0acbce30ee1"
DSTACK_RUN_UUID = "5368b9a3-a9aa-4d0c-b73d-533ba33b0d41"
JOB_SUBMISSION_UUID = "a530be1c-2af7-42f9-ac29-785efcfcad69"
PROTOCOL_SHA256 = "1" * 64
SOURCE_BUNDLE_SHA256 = "2" * 64
SUBMITTED_CODE_REVISION = "3" * 40
REQUEST_FILE_SHA256 = "4" * 64
PANEL_IDS_SHA256 = "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@"
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
RUNTIME_IMAGE_DIGEST = RUNTIME_IMAGE.rsplit("@", 1)[1]
OUTPUT_BASE = (
    f"hf://buckets/Npatta01/trec_video_2026/experiments/{RUN_ID}"
)
REMOTE_OUTPUT_URI = f"{OUTPUT_BASE}/selector/{SLICE_ID}/{EXECUTION_ATTEMPT_ID}"
MEDIA_URI = (
    "hf://buckets/Npatta01/trec_video_2026/artifacts/"
    "vqa2025-training-media-475-20260809-a1/media"
)
CONTROLLER_URI = f"{OUTPUT_BASE}/input-controller"
GPU_NAME = "NVIDIA H100 80GB HBM3"
GPU_TOTAL_BYTES = 80 * 1024**3
ARTIFACT_FILENAMES = frozenset(
    {
        "nvidia-smi.csv",
        "preflight.json",
        "runtime-preflight-marker.json",
        "selector-results.jsonl",
        "source-bundle-identity.json",
        "summary.json",
        "terminal-status.json",
        "workload.stdout.json",
    }
)


def load_adjudicator():
    spec = importlib.util.spec_from_file_location(
        "vqa_selector_adjudicator", ADJUDICATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load selector adjudicator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_checksums(artifact_dir: Path) -> None:
    files = sorted(path for path in artifact_dir.iterdir() if path.name != "SHA256SUMS")
    (artifact_dir / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.name}\n"
            for path in files
        ),
        encoding="utf-8",
    )


def pinned_canary_config(tmp_path: Path) -> Path:
    text = CANARY_CONFIG.read_text(encoding="utf-8")
    replacements = {
        "EXPECTED_PROTOCOL_SHA256": PROTOCOL_SHA256,
        "EXPECTED_SOURCE_BUNDLE_SHA256": SOURCE_BUNDLE_SHA256,
        "SUBMITTED_CODE_REVISION": SUBMITTED_CODE_REVISION,
        "EXECUTION_ATTEMPT_ID": EXECUTION_ATTEMPT_ID,
    }
    for key, value in replacements.items():
        text, count = __import__("re").subn(
            rf"(?m)^(  - {key}=)[^\n]+$", rf"\g<1>{value}", text
        )
        assert count == 1, key
    path = tmp_path / "canary.dstack.yml"
    path.write_text(text, encoding="utf-8")
    return path


def valid_generation(*, elapsed_seconds: float = 80.0) -> dict[str, object]:
    return {
        "elapsed_seconds": elapsed_seconds,
        "failure_stage": None,
        "generation": {
            "ended_with_eos": True,
            "final_generated_token_id": 151645,
            "gpu_total_memory_bytes": GPU_TOTAL_BYTES,
            "peak_memory_allocated_bytes": 70 * 1024**3,
            "peak_memory_reserved_bytes": 72 * 1024**3,
            "modality_attestation": {
                "same_first_forward": True,
                "native_video_element_count": 1,
                "native_audio_element_count": 1,
            },
        },
    }


def expected_safe_env() -> dict[str, str]:
    return {
        "CONTROLLER_URI": CONTROLLER_URI,
        "EXECUTION_ATTEMPT_ID": EXECUTION_ATTEMPT_ID,
        "EXPECTED_PROTOCOL_SHA256": PROTOCOL_SHA256,
        "EXPECTED_SOURCE_BUNDLE_SHA256": SOURCE_BUNDLE_SHA256,
        "EXPERIMENT_RUN_ID": RUN_ID,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEDIA_URI": MEDIA_URI,
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "OUTPUT_BASE": OUTPUT_BASE,
        "PYTHONPATH": "/dstack/run/src",
        "RUNTIME_IMAGE": RUNTIME_IMAGE,
        "SELECTOR_SLICE": SLICE_ID,
        "SUBMITTED_CODE_REVISION": SUBMITTED_CODE_REVISION,
    }


def expected_resources() -> dict[str, object]:
    return {
        "cpu": {"min": 16},
        "memory": {"min": 192.0},
        "gpu": {
            "vendor": "nvidia",
            "name": ["H100"],
            "count": {"min": 1, "max": 1},
            "memory": {"min": 80.0},
        },
        "disk": {"size": {"min": 220.0, "max": 220.0}},
    }


def billing_evidence(*, cost: float = 1.35) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "dstack-ps-a-json-v0.19",
        "project": "private-project",
        "run": {
            "id": DSTACK_RUN_UUID,
            "cost": cost,
            "status": "done",
            "submitted_at": "2026-08-09T12:00:00+00:00",
            "run_spec": {
                "run_name": RUN_NAME,
                "configuration": {
                    "type": "task",
                    "name": RUN_NAME,
                    "image": RUNTIME_IMAGE,
                    "entrypoint": "/bin/bash -c",
                    "shell": "bash",
                    "working_dir": "/dstack/run",
                    "env": expected_safe_env(),
                    "secret_env_keys": ["HF_TOKEN"],
                    "commands": [
                        "exec bash /dstack/run/scripts/"
                        "run-vqa-silver-oracle-selector-dstack.sh"
                    ],
                    "resources": expected_resources(),
                    "fleets": ["gpu-marketplace"],
                    "backends": ["runpod", "vastai"],
                    "spot_policy": "on-demand",
                    "max_price": 3.35,
                    "max_duration": 2700,
                },
            },
            "jobs": [
                {
                    "job_index": 0,
                    "job_submissions": [
                        {
                            "id": JOB_SUBMISSION_UUID,
                            "status": "done",
                            "submitted_at": "2026-08-09T12:00:01+00:00",
                            "finished_at": "2026-08-09T12:20:01+00:00",
                            "exit_status": 0,
                            "job_provisioning_data": {
                                "backend": "runpod",
                                "region": "US-KS-2",
                                "price": 3.35,
                                "instance_type": {
                                    "name": "NVIDIA H100",
                                    "resources": {
                                        "gpus": [
                                            {
                                                "name": "H100",
                                                "vendor": "nvidia",
                                                "memory_mib": 81920,
                                            }
                                        ]
                                    },
                                },
                            },
                        }
                    ],
                }
            ],
        },
    }


def raw_dstack_ps() -> dict[str, object]:
    evidence = billing_evidence()
    run = copy.deepcopy(evidence["run"])
    assert isinstance(run, dict)
    run_spec = run["run_spec"]
    assert isinstance(run_spec, dict)
    configuration = run_spec["configuration"]
    assert isinstance(configuration, dict)
    safe_env = configuration.pop("env")
    configuration.pop("secret_env_keys")
    assert isinstance(safe_env, dict)
    configuration["env"] = {"HF_TOKEN": "PRIVATE_TOKEN_MUST_NOT_SURVIVE", **safe_env}
    configuration["files"] = [{"local": "private", "remote": "/dstack/run"}]
    run["user"] = "private-user"
    run["ssh_key_pub"] = "private-key-material"
    return {"project": evidence["project"], "runs": [run], "server_extra": "ignored"}


def canary_artifacts(
    tmp_path: Path,
    *,
    original_elapsed_seconds: float = 80.0,
    sentinel_elapsed_seconds: float = 1.0,
    repair_count: int = 2,
    model_load_seconds: float = 120.0,
    sampled_free_bytes: int = 6 * 1024**3,
    gpu_name: str = GPU_NAME,
) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    sentinel_hash = hashlib.sha256(b"resource-sentinel").hexdigest()
    video_ids = [f"video-{index:02d}" for index in range(10)] + ["resource-sentinel"]
    request_ids_sha256 = hashlib.sha256(
        ("\n".join(video_ids) + "\n").encode("utf-8")
    ).hexdigest()
    preflight = {
        "schema_version": 1,
        "run_id": EXECUTION_RUN_ID,
        "controller_run_id": RUN_ID,
        "execution_attempt_id": EXECUTION_ATTEMPT_ID,
        "remote_output_uri": REMOTE_OUTPUT_URI,
        "protocol_sha256": PROTOCOL_SHA256,
        "requests_filename": "selector-canary.jsonl",
        "request_file_sha256": REQUEST_FILE_SHA256,
        "request_ids_sha256": request_ids_sha256,
        "request_count": 11,
        "panel_ids_sha256": PANEL_IDS_SHA256,
        "resource_sentinel_id_sha256": sentinel_hash,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "submitted_code_revision": SUBMITTED_CODE_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "runtime_image_digest": RUNTIME_IMAGE_DIGEST,
        "native_video_requested": True,
        "embedded_audio_requested": True,
        "video_fps": 2.0,
    }
    write_json(artifact_dir / "preflight.json", preflight)
    marker_without_hash = {**preflight, "status": "passed"}
    marker = {
        **marker_without_hash,
        "marker_sha256": canonical_sha256(marker_without_hash),
    }
    write_json(artifact_dir / "runtime-preflight-marker.json", marker)

    result_identity = {
        "execution_run_id": EXECUTION_RUN_ID,
        "controller_run_id": RUN_ID,
        "execution_attempt_id": EXECUTION_ATTEMPT_ID,
        "remote_output_uri": REMOTE_OUTPUT_URI,
        "protocol_sha256": PROTOCOL_SHA256,
        "requests_filename": "selector-canary.jsonl",
        "request_file_sha256": REQUEST_FILE_SHA256,
        "request_ids_sha256": request_ids_sha256,
        "panel_ids_sha256": PANEL_IDS_SHA256,
        "resource_sentinel_id_sha256": sentinel_hash,
        "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
        "submitted_code_revision": SUBMITTED_CODE_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": GPU_TOTAL_BYTES,
    }
    results: list[dict[str, object]] = []
    for index in range(10):
        attempts = [
            valid_generation(
                elapsed_seconds=original_elapsed_seconds / (1 + (repair_count > index))
            )
        ]
        if index < repair_count:
            attempts.append(
                valid_generation(elapsed_seconds=original_elapsed_seconds / 2)
            )
        results.append(
            {
                "video_id": f"video-{index:02d}",
                "status": "passed",
                "failure_stage": None,
                "repair_used": index < repair_count,
                "attempts": attempts,
                **result_identity,
            }
        )
    results.append(
        {
            "video_id": "resource-sentinel",
            "status": "passed",
            "failure_stage": None,
            "repair_used": False,
            "attempts": [valid_generation(elapsed_seconds=sentinel_elapsed_seconds)],
            **result_identity,
        }
    )
    results_path = artifact_dir / "selector-results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in results
        ),
        encoding="utf-8",
    )
    generation_seconds = sum(
        attempt["elapsed_seconds"] for row in results for attempt in row["attempts"]
    )
    summary = {
        **preflight,
        "device": f"cuda:0 (NVIDIA CUDA; {gpu_name})",
        "precision": "bfloat16",
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": GPU_TOTAL_BYTES,
        "runtime_versions": {"torch": "test"},
        "model_load_seconds": model_load_seconds,
        "rows": 11,
        "passed": 11,
        "failed": 0,
        "schema_failures": repair_count,
        "repair_count": repair_count,
        "elapsed_generation_seconds": generation_seconds,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "status": "passed",
    }
    write_json(artifact_dir / "summary.json", summary)
    write_json(artifact_dir / "workload.stdout.json", summary)
    write_json(
        artifact_dir / "source-bundle-identity.json",
        {
            "schema_version": 1,
            "execution_run_id": EXECUTION_RUN_ID,
            "execution_attempt_id": EXECUTION_ATTEMPT_ID,
            "remote_output_uri": REMOTE_OUTPUT_URI,
            "source_bundle_sha256": SOURCE_BUNDLE_SHA256,
            "submitted_code_revision": SUBMITTED_CODE_REVISION,
        },
    )
    write_json(
        artifact_dir / "terminal-status.json",
        {
            "artifact_sync_attempted": True,
            "workload_returncode": 0,
            "execution_attempt_id": EXECUTION_ATTEMPT_ID,
            "remote_output_uri": REMOTE_OUTPUT_URI,
        },
    )
    sampled_free_mib = sampled_free_bytes / 1024**2
    (artifact_dir / "nvidia-smi.csv").write_text(
        "timestamp, name, memory.total [MiB], memory.used [MiB], "
        "memory.free [MiB], utilization.gpu [%], power.draw [W]\n"
        f"2026/08/09 12:05:00.000, {gpu_name}, 81920, 4096, "
        f"{sampled_free_mib}, 80, 500\n",
        encoding="utf-8",
    )
    assert {path.name for path in artifact_dir.iterdir()} == ARTIFACT_FILENAMES
    write_checksums(artifact_dir)
    return artifact_dir


def adjudicate(adjudicator, artifact_dir: Path, evidence: object, config: Path):
    return adjudicator.adjudicate_canary(
        artifact_dir,
        evidence,
        expected_config_path=config,
    )


def test_sanitizer_extracts_one_exact_run_and_never_persists_raw_secrets(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)

    sanitized = adjudicator.sanitize_dstack_billing_evidence(
        raw_dstack_ps(), expected_config_path=config
    )

    assert sanitized == billing_evidence()
    serialized = json.dumps(sanitized, sort_keys=True)
    assert "PRIVATE_TOKEN_MUST_NOT_SURVIVE" not in serialized
    assert "private-user" not in serialized
    assert "private-key-material" not in serialized
    assert sanitized["run"]["run_spec"]["configuration"]["secret_env_keys"] == [
        "HF_TOKEN"
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw["runs"].append(copy.deepcopy(raw["runs"][0])), "exactly one"),
        (
            lambda raw: raw["runs"][0]["run_spec"]["configuration"]["env"].update(
                {"UNEXPECTED_SECRET": "do-not-copy"}
            ),
            "environment",
        ),
        (
            lambda raw: raw["runs"][0]["run_spec"]["configuration"]["env"].pop(
                "EXECUTION_ATTEMPT_ID"
            ),
            "environment",
        ),
    ],
)
def test_sanitizer_rejects_ambiguous_or_nonexact_raw_records(
    tmp_path: Path, mutation, match: str
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    raw = raw_dstack_ps()
    mutation(raw)

    with pytest.raises(ValueError, match=match):
        adjudicator.sanitize_dstack_billing_evidence(raw, expected_config_path=config)


def test_adjudication_accepts_exact_b4_artifacts_and_billing_path_or_object(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    evidence = billing_evidence()

    decision = adjudicate(adjudicator, artifacts, evidence, config)
    assert decision["status"] == "go"
    assert decision["gates"]["billing_evidence"] is True
    assert decision["gates"]["artifact_integrity"] is True
    assert decision["gates"]["evidence_identity"] is True
    assert decision["gates"]["h100_identity"] is True
    assert decision["projection"]["shard_row_counts"] == [119, 119, 119, 118]

    evidence_path = tmp_path / "billing.json"
    write_json(evidence_path, evidence)
    assert adjudicate(adjudicator, artifacts, evidence_path, config)["status"] == "go"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["run"].update({"raw_user": "must-fail"}),
        lambda value: value["run"].update({"status": "running"}),
        lambda value: value["run"].update({"id": "not-a-uuid"}),
        lambda value: value["run"]["run_spec"].update({"run_name": "a2-run"}),
        lambda value: value["run"]["run_spec"]["configuration"]["env"].update(
            {"EXECUTION_ATTEMPT_ID": "511c5f82-68c1-4df8-a19e-ac91e28f9232"}
        ),
        lambda value: value["run"]["jobs"][0]["job_submissions"][0][
            "job_provisioning_data"
        ]["instance_type"].update({"name": "NVIDIA A100"}),
        lambda value: value["run"]["jobs"][0]["job_submissions"][0][
            "job_provisioning_data"
        ].update({"price": 3.350001}),
        lambda value: value["run"]["jobs"][0]["job_submissions"][0].update(
            {"finished_at": "2026-08-09T11:59:00+00:00"}
        ),
    ],
)
def test_adjudication_rejects_missing_mixed_or_tampered_billing_fields(
    tmp_path: Path, mutation
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    evidence = billing_evidence()
    mutation(evidence)

    decision = adjudicate(adjudicator, artifacts, evidence, config)

    assert decision["status"] == "no-go"
    assert "billing_evidence" in decision["failure_reasons"]
    assert "must-fail" not in json.dumps(decision)


def test_adjudication_rejects_float_instead_of_bound_billing_evidence(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)

    decision = adjudicate(adjudicator, artifacts, 1.35, config)

    assert decision["status"] == "no-go"
    assert "billing_evidence" in decision["failure_reasons"]


@pytest.mark.parametrize(
    "filename",
    [
        "preflight.json",
        "source-bundle-identity.json",
        "summary.json",
        "selector-results.jsonl",
    ],
)
def test_adjudication_rejects_tampered_artifact_even_with_regenerated_sha256sums(
    tmp_path: Path, filename: str
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    path = artifacts / filename
    path.write_bytes(path.read_bytes() + b" ")
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert set(decision["failure_reasons"]) & {
        "artifact_integrity",
        "evidence_identity",
    }


def test_adjudication_rejects_result_checksum_mismatch_when_remote_hashes_regenerated(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    results = artifacts / "selector-results.jsonl"
    results.write_bytes(results.read_bytes() + b"\n")
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert set(decision["failure_reasons"]) & {
        "artifact_integrity",
        "evidence_identity",
    }


def test_adjudication_rejects_substituted_attempt_even_if_all_artifacts_are_rehashed(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    substitute = "511c5f82-68c1-4df8-a19e-ac91e28f9232"
    for path in artifacts.iterdir():
        if path.name == "SHA256SUMS":
            continue
        payload = path.read_bytes().replace(
            EXECUTION_ATTEMPT_ID.encode(), substitute.encode()
        )
        path.write_bytes(payload)
    evidence = billing_evidence()
    evidence["run"]["run_spec"]["configuration"]["env"]["EXECUTION_ATTEMPT_ID"] = (
        substitute
    )
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, evidence, config)

    assert decision["status"] == "no-go"
    assert "billing_evidence" in decision["failure_reasons"]


@pytest.mark.parametrize("gpu_name", ["NVIDIA A100-SXM4-80GB", "NVIDIA H200"])
def test_adjudication_rejects_non_h100_runtime_and_result_identity(
    tmp_path: Path, gpu_name: str
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path, gpu_name=gpu_name)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert "h100_identity" in decision["failure_reasons"]


def test_adjudication_rejects_mixed_nvidia_smi_gpu_names(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    telemetry = artifacts / "nvidia-smi.csv"
    telemetry.write_text(
        telemetry.read_text(encoding="utf-8")
        + "2026/08/09 12:05:01.000, NVIDIA A100-SXM4-80GB, 81920, 4096, "
        "6144, 80, 500\n",
        encoding="utf-8",
    )
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert "h100_identity" in decision["failure_reasons"]


def test_adjudication_rejects_eleven_row_boundary_failures(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    results_path = artifacts / "selector-results.jsonl"
    rows = results_path.read_text(encoding="utf-8").splitlines()[:-1]
    results_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    summary["rows"] = 10
    summary["passed"] = 10
    summary["results_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
    write_json(artifacts / "summary.json", summary)
    write_json(artifacts / "workload.stdout.json", summary)
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert "canary_row_count" in decision["failure_reasons"]


def test_adjudication_rejects_three_original_repairs(tmp_path: Path) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path, repair_count=3)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert "original_ten_repairs" in decision["failure_reasons"]


def test_adjudication_rejects_any_eos_modality_or_generation_failure(
    tmp_path: Path,
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path)
    path = artifacts / "selector-results.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()]
    results[0]["attempts"][0]["generation"]["ended_with_eos"] = False
    results[1]["attempts"][0]["generation"]["modality_attestation"][
        "same_first_forward"
    ] = False
    results[2]["attempts"][0]["failure_stage"] = "generation"
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in results
        ),
        encoding="utf-8",
    )
    summary = json.loads((artifacts / "summary.json").read_text())
    summary["results_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(artifacts / "summary.json", summary)
    write_json(artifacts / "workload.stdout.json", summary)
    write_checksums(artifacts)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(), config)

    assert decision["status"] == "no-go"
    assert "eos" in decision["failure_reasons"]
    assert "modality" in decision["failure_reasons"]
    assert "generation_failures" in decision["failure_reasons"]


@pytest.mark.parametrize(
    ("reason", "artifact_kwargs", "cost"),
    [
        (
            "original_ten_generation_seconds",
            {"original_elapsed_seconds": 895.000001},
            1.35,
        ),
        ("model_load_seconds", {"model_load_seconds": 120.000001}, 1.35),
        ("sampled_free_memory", {"sampled_free_bytes": 6 * 1024**3 - 1}, 1.35),
        ("actual_billed_cost", {}, 1.350001),
    ],
)
def test_adjudication_rejects_operational_gate_overages(
    tmp_path: Path, reason: str, artifact_kwargs: dict[str, object], cost: float
) -> None:
    adjudicator = load_adjudicator()
    config = pinned_canary_config(tmp_path)
    artifacts = canary_artifacts(tmp_path, **artifact_kwargs)

    decision = adjudicate(adjudicator, artifacts, billing_evidence(cost=cost), config)

    assert decision["status"] == "no-go"
    assert reason in decision["failure_reasons"]


def test_full_projection_rejects_aggregate_longest_and_compute_boundaries() -> None:
    adjudicator = load_adjudicator()
    safe = adjudicator.project_full_panel(
        original_ten_generation_seconds=800.0,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert safe["aggregate_gpu_hours"] < 12.0
    assert safe["longest_shard_hours"] < 3.25
    assert safe["compute_usd"] < 39.48
    assert safe["projected_billed_cost_usd"] < 45.0

    aggregate_boundary = adjudicator.project_full_panel(
        original_ten_generation_seconds=(12 * 3600 - 4 * 120) * 10 / 475,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert math.isclose(aggregate_boundary["aggregate_gpu_hours"], 12.0)
    assert aggregate_boundary["gates"]["aggregate_gpu_hours"] is True
    aggregate_over = adjudicator.project_full_panel(
        original_ten_generation_seconds=(12 * 3600 - 4 * 120) * 10 / 475 + 0.001,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert aggregate_over["gates"]["aggregate_gpu_hours"] is False

    longest_boundary = adjudicator.project_full_panel(
        original_ten_generation_seconds=(3.25 * 3600 - 120) * 10 / 119,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert math.isclose(longest_boundary["longest_shard_hours"], 3.25)
    assert longest_boundary["gates"]["longest_shard_hours"] is True
    longest_over = adjudicator.project_full_panel(
        original_ten_generation_seconds=(3.25 * 3600 - 120) * 10 / 119 + 0.001,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert longest_over["gates"]["longest_shard_hours"] is False

    compute_boundary = adjudicator.project_full_panel(
        original_ten_generation_seconds=(39.48 / 3.35 * 3600 - 4 * 120) * 10 / 475,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert math.isclose(compute_boundary["compute_usd"], 39.48)
    assert compute_boundary["gates"]["compute_usd"] is True
    compute_over = adjudicator.project_full_panel(
        original_ten_generation_seconds=(39.48 / 3.35 * 3600 - 4 * 120) * 10 / 475
        + 0.001,
        model_load_seconds=120.0,
        actual_billed_cost_usd=1.0,
    )
    assert compute_over["gates"]["compute_usd"] is False


def test_adjudication_marks_projected_billed_cost_over_cap() -> None:
    adjudicator = load_adjudicator()
    projection = adjudicator.project_full_panel(
        original_ten_generation_seconds=800.0,
        model_load_seconds=120.0,
        actual_billed_cost_usd=45.0,
    )

    assert projection["gates"]["projected_billed_cost_usd"] is False
