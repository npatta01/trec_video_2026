#!/usr/bin/env python3
"""Sanitize dstack billing and adjudicate the exact B4 selector canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

CANARY_ROW_COUNT = 11
ORIGINAL_CANARY_ROW_COUNT = 10
FULL_PANEL_COUNT = 475
SHARD_ROW_COUNTS = (119, 119, 119, 118)
CANONICAL_PANEL_IDS_SHA256 = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXPECTED_RUN_ID = "vqa-silver-oracle-election-20260809-b4"
EXPECTED_DSTACK_RUN_NAME = "vqa-silver-selector-canary-20260809-b4"
EXPECTED_SLICE = "canary-b4"
EXPECTED_REQUEST_FILENAME = "selector-canary.jsonl"
EXPECTED_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
EXPECTED_MODEL_REVISION = "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b"
EXPECTED_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@"
    "sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)
EXPECTED_RUNTIME_IMAGE_DIGEST = EXPECTED_RUNTIME_IMAGE.rsplit("@", 1)[1]
EXPECTED_MEDIA_URI = (
    "hf://buckets/Npatta01/trec_video_2026/artifacts/"
    "vqa2025-training-media-475-20260809-a1/media"
)
EXPECTED_OUTPUT_BASE = (
    f"hf://buckets/Npatta01/trec_video_2026/experiments/{EXPECTED_RUN_ID}"
)
EXPECTED_CONTROLLER_URI = f"{EXPECTED_OUTPUT_BASE}/input-controller"
EXPECTED_COMMAND = (
    "exec bash /dstack/run/scripts/run-vqa-silver-oracle-selector-dstack.sh"
)
EXPECTED_RESOURCES = {
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
SOURCE_CONTRACT = "dstack-ps-a-json-v0.19"
CANARY_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "dstack/vqa-silver-oracle-selector-canary.dstack.yml"
)
MIN_GPU_MEMORY_BYTES = 78_700_000_000
MIN_FREE_MEMORY_BYTES = 6 * 1024**3
MAX_PEAK_ALLOCATED_FRACTION = 0.90
EXPECTED_EOS_TOKEN_ID = 151645
MAX_ORIGINAL_GENERATION_SECONDS = 895.0
MAX_MODEL_LOAD_SECONDS = 120.0
MAX_CANARY_BILLED_COST_USD = 1.35
MAX_AGGREGATE_GPU_HOURS = 12.0
MAX_LONGEST_SHARD_HOURS = 3.25
MAX_COMPUTE_USD = 39.48
MAX_PROJECTED_BILLED_COST_USD = 45.0
H100_PRICE_USD_PER_HOUR = 3.35
EXPECTED_ARTIFACT_FILENAMES = frozenset(
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
PREFLIGHT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "controller_run_id",
        "execution_attempt_id",
        "remote_output_uri",
        "protocol_sha256",
        "requests_filename",
        "request_file_sha256",
        "request_ids_sha256",
        "request_count",
        "panel_ids_sha256",
        "resource_sentinel_id_sha256",
        "source_bundle_sha256",
        "submitted_code_revision",
        "model_id",
        "model_revision",
        "runtime_image",
        "runtime_image_digest",
        "native_video_requested",
        "embedded_audio_requested",
        "video_fps",
    }
)
SUMMARY_EXTRA_KEYS = frozenset(
    {
        "device",
        "precision",
        "gpu_name",
        "gpu_total_memory_bytes",
        "runtime_versions",
        "model_load_seconds",
        "rows",
        "passed",
        "failed",
        "schema_failures",
        "repair_count",
        "elapsed_generation_seconds",
        "results_sha256",
        "status",
    }
)
RESULT_IDENTITY_FIELDS = {
    "execution_run_id": "run_id",
    "controller_run_id": "controller_run_id",
    "execution_attempt_id": "execution_attempt_id",
    "remote_output_uri": "remote_output_uri",
    "protocol_sha256": "protocol_sha256",
    "requests_filename": "requests_filename",
    "request_file_sha256": "request_file_sha256",
    "request_ids_sha256": "request_ids_sha256",
    "panel_ids_sha256": "panel_ids_sha256",
    "resource_sentinel_id_sha256": "resource_sentinel_id_sha256",
    "source_bundle_sha256": "source_bundle_sha256",
    "submitted_code_revision": "submitted_code_revision",
    "model_id": "model_id",
    "model_revision": "model_revision",
    "runtime_image": "runtime_image",
}
DECISION_GATE_NAMES = (
    "billing_evidence",
    "artifact_integrity",
    "evidence_identity",
    "h100_identity",
    "request_filename",
    "panel_id_digest",
    "canary_row_count",
    "sentinel_identity",
    "valid_rows",
    "result_id_digest",
    "original_ten_repairs",
    "eos",
    "modality",
    "memory",
    "generation_failures",
    "generation_evidence",
    "top_level_generation_failures",
    "attempt_timings",
    "original_ten_generation_seconds",
    "summary_generation_seconds",
    "model_load_seconds",
    "sampled_free_memory",
    "actual_billed_cost",
    "aggregate_gpu_hours",
    "longest_shard_hours",
    "compute_usd",
    "projected_billed_cost_usd",
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_REVISION = re.compile(r"[0-9a-f]{40}")
_GPU_FAMILY = re.compile(r"(?<![A-Z0-9])(?:A|H|L|T|V)\d{2,4}(?![A-Z0-9])")
_ENV_LINE = re.compile(r"^  - ([A-Z][A-Z0-9_]*)=(.*)$")
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64}) [ *](\./[^\r\n]+)$")
_ZERO_UUID = uuid.UUID(int=0)


class EvidenceError(ValueError):
    """An expected safe evidence contract was absent or inconsistent."""


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvidenceError(f"{field} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{field} fields do not match the allowlist")


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} must be nonempty text")
    return value.strip()


def _canonical_uuid(value: Any, field: str) -> str:
    text = _nonempty_text(value, field)
    try:
        parsed = uuid.UUID(text)
    except ValueError as error:
        raise EvidenceError(f"{field} must be a canonical UUID") from error
    if parsed == _ZERO_UUID or str(parsed) != text:
        raise EvidenceError(f"{field} must be a nonzero canonical UUID")
    return text


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty_text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{field} must include a timezone")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ordered_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str | None:
    ids = [row.get("video_id") for row in rows]
    if any(not isinstance(video_id, str) or not video_id for video_id in ids):
        return None
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _is_h100_name(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    upper = value.strip().upper()
    if not (upper == "H100" or upper.startswith("NVIDIA H100")):
        return False
    return _GPU_FAMILY.findall(upper) == ["H100"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_expected_configuration(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    env: dict[str, str] = {}
    in_env = False
    for line in text.splitlines():
        if line == "env:":
            in_env = True
            continue
        if in_env and line and not line.startswith("  - "):
            break
        if not in_env or not line:
            continue
        match = _ENV_LINE.fullmatch(line)
        if match is None or match.group(1) in env:
            raise EvidenceError("tracked canary environment is not exact")
        env[match.group(1)] = match.group(2)

    safe_fixed = {
        "CONTROLLER_URI": EXPECTED_CONTROLLER_URI,
        "EXPERIMENT_RUN_ID": EXPECTED_RUN_ID,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MEDIA_URI": EXPECTED_MEDIA_URI,
        "MODEL_ID": EXPECTED_MODEL_ID,
        "MODEL_REVISION": EXPECTED_MODEL_REVISION,
        "OUTPUT_BASE": EXPECTED_OUTPUT_BASE,
        "PYTHONPATH": "/dstack/run/src",
        "RUNTIME_IMAGE": EXPECTED_RUNTIME_IMAGE,
        "SELECTOR_SLICE": EXPECTED_SLICE,
    }
    dynamic_keys = {
        "EXECUTION_ATTEMPT_ID",
        "EXPECTED_PROTOCOL_SHA256",
        "EXPECTED_SOURCE_BUNDLE_SHA256",
        "SUBMITTED_CODE_REVISION",
    }
    expected_keys = set(safe_fixed) | dynamic_keys | {"HF_TOKEN"}
    if set(env) != expected_keys or any(
        env.get(key) != value for key, value in safe_fixed.items()
    ):
        raise EvidenceError("tracked canary environment is not exact")
    if env["HF_TOKEN"] != "${{ secrets.hf_token }}":
        raise EvidenceError("tracked canary secret reference is not exact")
    protocol = env["EXPECTED_PROTOCOL_SHA256"]
    source = env["EXPECTED_SOURCE_BUNDLE_SHA256"]
    revision = env["SUBMITTED_CODE_REVISION"]
    if _HEX_SHA256.fullmatch(protocol) is None or protocol == "0" * 64:
        raise EvidenceError("tracked protocol pin is not finalized")
    if _HEX_SHA256.fullmatch(source) is None or source == "0" * 64:
        raise EvidenceError("tracked source pin is not finalized")
    if _HEX_REVISION.fullmatch(revision) is None or revision == "0" * 40:
        raise EvidenceError("tracked revision pin is not finalized")
    _canonical_uuid(env["EXECUTION_ATTEMPT_ID"], "tracked execution attempt ID")

    required_fragments = (
        f"name: {EXPECTED_DSTACK_RUN_NAME}",
        f"image: {EXPECTED_RUNTIME_IMAGE}",
        "entrypoint: /bin/bash -c",
        "shell: bash",
        "working_dir: /dstack/run",
        "vendor: nvidia",
        "name: H100",
        "count: 1",
        "memory: 80GB..",
        "cpu: 16..",
        "memory: 192GB..",
        "disk: 220GB",
        "fleets: [gpu-marketplace]",
        "backends: [runpod, vastai]",
        "spot_policy: on-demand",
        "max_price: 3.35",
        "max_duration: 45m",
        f"  - {EXPECTED_COMMAND}",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise EvidenceError("tracked canary configuration is not exact")
    safe_env = {key: value for key, value in env.items() if key != "HF_TOKEN"}
    return {
        "type": "task",
        "name": EXPECTED_DSTACK_RUN_NAME,
        "image": EXPECTED_RUNTIME_IMAGE,
        "entrypoint": "/bin/bash -c",
        "shell": "bash",
        "working_dir": "/dstack/run",
        "env": dict(sorted(safe_env.items())),
        "secret_env_keys": ["HF_TOKEN"],
        "commands": [EXPECTED_COMMAND],
        "resources": EXPECTED_RESOURCES,
        "fleets": ["gpu-marketplace"],
        "backends": ["runpod", "vastai"],
        "spot_policy": "on-demand",
        "max_price": 3.35,
        "max_duration": 2700,
    }


def _sanitize_configuration(
    raw: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    raw_env = _mapping(raw.get("env"), "run configuration environment")
    expected_env = _mapping(expected["env"], "expected environment")
    if set(raw_env) != set(expected_env) | {"HF_TOKEN"}:
        raise EvidenceError("run configuration environment is not exact")
    if any(raw_env.get(key) != value for key, value in expected_env.items()):
        raise EvidenceError("run configuration environment is not exact")
    if "HF_TOKEN" not in raw_env:
        raise EvidenceError("run configuration secret environment is missing")

    fields = {
        "type": raw.get("type"),
        "name": raw.get("name"),
        "image": raw.get("image"),
        "entrypoint": raw.get("entrypoint"),
        "shell": raw.get("shell"),
        "working_dir": raw.get("working_dir"),
        "env": dict(sorted(expected_env.items())),
        "secret_env_keys": ["HF_TOKEN"],
        "commands": list(_sequence(raw.get("commands"), "configuration commands")),
        "resources": raw.get("resources"),
        "fleets": [
            item.get("name") if isinstance(item, Mapping) else item
            for item in _sequence(raw.get("fleets"), "configuration fleets")
        ],
        "backends": list(_sequence(raw.get("backends"), "configuration backends")),
        "spot_policy": raw.get("spot_policy"),
        "max_price": _finite_number(raw.get("max_price")),
        "max_duration": raw.get("max_duration"),
    }
    if fields != expected:
        raise EvidenceError("run configuration does not match tracked canary")
    return fields


def _sanitize_jobs(raw_jobs: Any) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for job_index, raw_job_value in enumerate(_sequence(raw_jobs, "run jobs")):
        raw_job = _mapping(raw_job_value, "run job")
        submissions: list[dict[str, Any]] = []
        for raw_submission_value in _sequence(
            raw_job.get("job_submissions"), "job submissions"
        ):
            raw_submission = _mapping(raw_submission_value, "job submission")
            provisioning = _mapping(
                raw_submission.get("job_provisioning_data"),
                "job provisioning data",
            )
            instance = _mapping(
                provisioning.get("instance_type"), "provisioned instance type"
            )
            resources = _mapping(
                instance.get("resources"), "provisioned instance resources"
            )
            gpus = []
            for raw_gpu_value in _sequence(resources.get("gpus"), "provisioned GPUs"):
                raw_gpu = _mapping(raw_gpu_value, "provisioned GPU")
                gpus.append(
                    {
                        "name": raw_gpu.get("name"),
                        "vendor": raw_gpu.get("vendor"),
                        "memory_mib": raw_gpu.get("memory_mib"),
                    }
                )
            submissions.append(
                {
                    "id": raw_submission.get("id"),
                    "status": raw_submission.get("status"),
                    "submitted_at": raw_submission.get("submitted_at"),
                    "finished_at": raw_submission.get("finished_at"),
                    "exit_status": raw_submission.get("exit_status"),
                    "job_provisioning_data": {
                        "backend": provisioning.get("backend"),
                        "region": provisioning.get("region"),
                        "price": _finite_number(provisioning.get("price")),
                        "instance_type": {
                            "name": instance.get("name"),
                            "resources": {"gpus": gpus},
                        },
                    },
                }
            )
        jobs.append({"job_index": job_index, "job_submissions": submissions})
    return jobs


def sanitize_dstack_billing_evidence(
    raw_payload: Mapping[str, Any],
    *,
    expected_config_path: Path | None = None,
) -> dict[str, Any]:
    """Extract only the safe B4 fields from `dstack ps -a --json`."""

    raw = _mapping(raw_payload, "dstack ps payload")
    project = _nonempty_text(raw.get("project"), "dstack project")
    expected = _load_expected_configuration(expected_config_path or CANARY_CONFIG_PATH)
    candidates = []
    for value in _sequence(raw.get("runs"), "dstack runs"):
        run = _mapping(value, "dstack run")
        run_spec = run.get("run_spec")
        if isinstance(run_spec, Mapping) and (
            run_spec.get("run_name") == EXPECTED_DSTACK_RUN_NAME
        ):
            candidates.append(run)
    if len(candidates) != 1:
        raise EvidenceError("dstack payload must contain exactly one B4 canary run")
    selected = candidates[0]
    raw_run_spec = _mapping(selected.get("run_spec"), "run specification")
    configuration = _sanitize_configuration(
        _mapping(raw_run_spec.get("configuration"), "run configuration"), expected
    )
    evidence = {
        "schema_version": 1,
        "source": SOURCE_CONTRACT,
        "project": project,
        "run": {
            "id": selected.get("id"),
            "cost": _finite_number(selected.get("cost")),
            "status": selected.get("status"),
            "submitted_at": selected.get("submitted_at"),
            "run_spec": {
                "run_name": raw_run_spec.get("run_name"),
                "configuration": configuration,
            },
            "jobs": _sanitize_jobs(selected.get("jobs")),
        },
    }
    _validate_billing_evidence(evidence, expected)
    return evidence


def _validate_billing_evidence(
    evidence: Mapping[str, Any], expected_configuration: Mapping[str, Any]
) -> dict[str, Any]:
    root = _mapping(evidence, "billing evidence")
    _exact_keys(
        root, {"schema_version", "source", "project", "run"}, "billing evidence"
    )
    if root.get("schema_version") != 1 or root.get("source") != SOURCE_CONTRACT:
        raise EvidenceError("billing evidence version is not exact")
    _nonempty_text(root.get("project"), "billing project")
    run = _mapping(root.get("run"), "billing run")
    _exact_keys(
        run,
        {"id", "cost", "status", "submitted_at", "run_spec", "jobs"},
        "billing run",
    )
    run_uuid = _canonical_uuid(run.get("id"), "dstack run UUID")
    if run.get("status") != "done":
        raise EvidenceError("dstack run status must be done")
    run_submitted = _timestamp(run.get("submitted_at"), "dstack run submitted_at")
    cost = _finite_number(run.get("cost"))
    if cost is None or cost < 0:
        raise EvidenceError("dstack run cost must be nonnegative")
    run_spec = _mapping(run.get("run_spec"), "billing run specification")
    _exact_keys(run_spec, {"run_name", "configuration"}, "billing run specification")
    if run_spec.get("run_name") != EXPECTED_DSTACK_RUN_NAME:
        raise EvidenceError("dstack run name is not the exact B4 canary")
    configuration = _mapping(run_spec.get("configuration"), "billing configuration")
    if configuration != expected_configuration:
        raise EvidenceError("billing configuration does not match tracked canary")
    _exact_keys(configuration, set(expected_configuration), "billing configuration")

    jobs = _sequence(run.get("jobs"), "billing jobs")
    if len(jobs) != 1:
        raise EvidenceError("B4 canary billing must contain exactly one job")
    latest_finish: datetime | None = None
    submission_count = 0
    for job_index, job_value in enumerate(jobs):
        job = _mapping(job_value, "billing job")
        _exact_keys(job, {"job_index", "job_submissions"}, "billing job")
        if job.get("job_index") != job_index:
            raise EvidenceError("billing job index is not deterministic")
        submissions = _sequence(job.get("job_submissions"), "billing submissions")
        if not submissions:
            raise EvidenceError("billing job has no job submission")
        for submission_value in submissions:
            submission_count += 1
            submission = _mapping(submission_value, "billing job submission")
            _exact_keys(
                submission,
                {
                    "id",
                    "status",
                    "submitted_at",
                    "finished_at",
                    "exit_status",
                    "job_provisioning_data",
                },
                "billing job submission",
            )
            _canonical_uuid(submission.get("id"), "job submission UUID")
            if submission.get("status") != "done" or submission.get("exit_status") != 0:
                raise EvidenceError("job submission must be done with exit status zero")
            submitted = _timestamp(
                submission.get("submitted_at"), "job submission submitted_at"
            )
            finished = _timestamp(
                submission.get("finished_at"), "job submission finished_at"
            )
            if submitted < run_submitted or finished < submitted:
                raise EvidenceError("job submission timestamps are inconsistent")
            if latest_finish is not None and finished < latest_finish:
                raise EvidenceError("job submission timestamps are not ordered")
            latest_finish = finished
            provisioning = _mapping(
                submission.get("job_provisioning_data"), "billing provisioning"
            )
            _exact_keys(
                provisioning,
                {"backend", "region", "price", "instance_type"},
                "billing provisioning",
            )
            _nonempty_text(provisioning.get("backend"), "billing backend")
            _nonempty_text(provisioning.get("region"), "billing region")
            price = _finite_number(provisioning.get("price"))
            if price is None or price <= 0 or price > H100_PRICE_USD_PER_HOUR:
                raise EvidenceError("provisioned price exceeds the exact H100 cap")
            instance = _mapping(
                provisioning.get("instance_type"), "billing instance type"
            )
            _exact_keys(instance, {"name", "resources"}, "billing instance type")
            if not _is_h100_name(instance.get("name")):
                raise EvidenceError("provisioned instance is not H100")
            resources = _mapping(
                instance.get("resources"), "billing instance resources"
            )
            _exact_keys(resources, {"gpus"}, "billing instance resources")
            gpus = _sequence(resources.get("gpus"), "billing GPUs")
            if len(gpus) != 1:
                raise EvidenceError("billing job must provision exactly one H100")
            gpu = _mapping(gpus[0], "billing GPU")
            _exact_keys(gpu, {"name", "vendor", "memory_mib"}, "billing GPU")
            memory_mib = _finite_number(gpu.get("memory_mib"))
            if (
                not _is_h100_name(gpu.get("name"))
                or str(gpu.get("vendor", "")).casefold() != "nvidia"
                or memory_mib is None
                or memory_mib * 1024**2 < MIN_GPU_MEMORY_BYTES
            ):
                raise EvidenceError("billing GPU is not an 80GB-class NVIDIA H100")
    if submission_count == 0:
        raise EvidenceError("billing evidence has no exact job submission")
    return {
        "run_uuid": run_uuid,
        "cost": cost,
        "configuration": dict(configuration),
    }


def _load_billing_evidence(value: Mapping[str, Any] | Path) -> Mapping[str, Any]:
    if isinstance(value, Path):
        payload = _load_json(value)
        return _mapping(payload, "billing evidence file")
    if isinstance(value, Mapping):
        return value
    raise EvidenceError("billing evidence must be a JSON path or object")


def _parse_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(EXPECTED_ARTIFACT_FILENAMES):
        raise EvidenceError("SHA256SUMS does not list the exact canary artifacts")
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise EvidenceError("SHA256SUMS contains a malformed entry")
        relative = PurePosixPath(match.group(2))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise EvidenceError("SHA256SUMS contains an unsafe path")
        name = relative.name
        if name in checksums:
            raise EvidenceError("SHA256SUMS contains a duplicate path")
        checksums[name] = match.group(1)
    if set(checksums) != EXPECTED_ARTIFACT_FILENAMES:
        raise EvidenceError("SHA256SUMS does not list the exact canary artifacts")
    return checksums


def load_nvidia_smi_csv(path: Path) -> list[dict[str, object]]:
    """Load strict H100 name, total-memory, and free-memory telemetry samples."""

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=1):
            normalized = {
                str(key).strip(): str(value).strip()
                for key, value in raw.items()
                if key is not None and value is not None
            }
            name = normalized.get("name")
            total_key = next(
                (key for key in normalized if "memory.total" in key.casefold()), None
            )
            free_key = next(
                (key for key in normalized if "memory.free" in key.casefold()), None
            )
            if not name or total_key is None or free_key is None:
                raise EvidenceError(f"telemetry row {row_number} is incomplete")
            try:
                total = float(normalized[total_key].split()[0])
                free = float(normalized[free_key].split()[0])
            except ValueError as error:
                raise EvidenceError(
                    f"telemetry row {row_number} has invalid memory"
                ) from error
            total_multiplier = 1024**2 if "mib" in total_key.casefold() else 1
            free_multiplier = 1024**2 if "mib" in free_key.casefold() else 1
            rows.append(
                {
                    "gpu_name": name,
                    "memory_total_bytes": total * total_multiplier,
                    "memory_free_bytes": free * free_multiplier,
                }
            )
    if not rows:
        raise EvidenceError("telemetry contains no samples")
    return rows


def _load_artifacts(artifact_dir: Path) -> dict[str, Any]:
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise EvidenceError("artifact directory must be a regular directory")
    paths = list(artifact_dir.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise EvidenceError("artifact directory contains a non-regular entry")
    expected_names = set(EXPECTED_ARTIFACT_FILENAMES) | {"SHA256SUMS"}
    if {path.name for path in paths} != expected_names:
        raise EvidenceError("artifact directory does not contain the exact canary set")
    checksums = _parse_sha256sums(artifact_dir / "SHA256SUMS")
    for name, expected in checksums.items():
        if _sha256_file(artifact_dir / name) != expected:
            raise EvidenceError("an artifact does not match SHA256SUMS")

    summary = _mapping(_load_json(artifact_dir / "summary.json"), "summary")
    preflight = _mapping(_load_json(artifact_dir / "preflight.json"), "preflight")
    source = _mapping(
        _load_json(artifact_dir / "source-bundle-identity.json"), "source identity"
    )
    marker = _mapping(
        _load_json(artifact_dir / "runtime-preflight-marker.json"),
        "runtime preflight marker",
    )
    terminal = _mapping(
        _load_json(artifact_dir / "terminal-status.json"), "terminal status"
    )
    workload = _mapping(
        _load_json(artifact_dir / "workload.stdout.json"), "workload stdout"
    )
    json_artifacts = {
        "summary.json": summary,
        "preflight.json": preflight,
        "source-bundle-identity.json": source,
        "runtime-preflight-marker.json": marker,
        "terminal-status.json": terminal,
        "workload.stdout.json": workload,
    }
    for name, payload in json_artifacts.items():
        if (artifact_dir / name).read_bytes() != _pretty_json_bytes(payload):
            raise EvidenceError(f"{name} bytes are not in the frozen JSON form")
    results = []
    for line in (
        (artifact_dir / "selector-results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if not line.strip():
            continue
        results.append(_mapping(json.loads(line), "result row"))
    expected_results = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in results
    ).encode("utf-8")
    if (artifact_dir / "selector-results.jsonl").read_bytes() != expected_results:
        raise EvidenceError("result bytes are not in the frozen JSONL form")
    return {
        "checksums": checksums,
        "summary": summary,
        "preflight": preflight,
        "source": source,
        "marker": marker,
        "terminal": terminal,
        "workload": workload,
        "results": results,
        "telemetry": load_nvidia_smi_csv(artifact_dir / "nvidia-smi.csv"),
    }


def _validate_artifact_identity(
    artifacts: Mapping[str, Any], expected_configuration: Mapping[str, Any]
) -> None:
    preflight = _mapping(artifacts["preflight"], "preflight")
    summary = _mapping(artifacts["summary"], "summary")
    source = _mapping(artifacts["source"], "source identity")
    marker = _mapping(artifacts["marker"], "preflight marker")
    terminal = _mapping(artifacts["terminal"], "terminal status")
    workload = _mapping(artifacts["workload"], "workload stdout")
    result_rows = [
        _mapping(row, "result row")
        for row in _sequence(artifacts["results"], "results")
    ]
    checksums = _mapping(artifacts["checksums"], "checksums")
    env = _mapping(expected_configuration["env"], "expected environment")
    attempt_id = env["EXECUTION_ATTEMPT_ID"]
    execution_run_id = f"{EXPECTED_RUN_ID}-{EXPECTED_SLICE}"
    remote_output_uri = f"{EXPECTED_OUTPUT_BASE}/selector/{EXPECTED_SLICE}/{attempt_id}"
    expected_fixed = {
        "schema_version": 1,
        "run_id": execution_run_id,
        "controller_run_id": EXPECTED_RUN_ID,
        "execution_attempt_id": attempt_id,
        "remote_output_uri": remote_output_uri,
        "protocol_sha256": env["EXPECTED_PROTOCOL_SHA256"],
        "requests_filename": EXPECTED_REQUEST_FILENAME,
        "request_count": CANARY_ROW_COUNT,
        "panel_ids_sha256": CANONICAL_PANEL_IDS_SHA256,
        "source_bundle_sha256": env["EXPECTED_SOURCE_BUNDLE_SHA256"],
        "submitted_code_revision": env["SUBMITTED_CODE_REVISION"],
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "runtime_image": EXPECTED_RUNTIME_IMAGE,
        "runtime_image_digest": EXPECTED_RUNTIME_IMAGE_DIGEST,
        "native_video_requested": True,
        "embedded_audio_requested": True,
        "video_fps": 2.0,
    }
    _exact_keys(preflight, set(PREFLIGHT_KEYS), "preflight")
    if any(preflight.get(key) != value for key, value in expected_fixed.items()):
        raise EvidenceError("preflight identity does not match tracked B4 canary")
    for field in (
        "request_file_sha256",
        "request_ids_sha256",
        "resource_sentinel_id_sha256",
    ):
        if (
            not isinstance(preflight.get(field), str)
            or _HEX_SHA256.fullmatch(preflight[field]) is None
        ):
            raise EvidenceError(f"preflight {field} is invalid")
    _exact_keys(summary, set(PREFLIGHT_KEYS | SUMMARY_EXTRA_KEYS), "summary")
    if any(summary.get(key) != preflight.get(key) for key in PREFLIGHT_KEYS):
        raise EvidenceError("summary and preflight identities differ")
    if workload != summary:
        raise EvidenceError("workload stdout does not equal the exact summary")

    _exact_keys(
        source,
        {
            "schema_version",
            "execution_run_id",
            "execution_attempt_id",
            "remote_output_uri",
            "source_bundle_sha256",
            "submitted_code_revision",
        },
        "source identity",
    )
    expected_source = {
        "schema_version": 1,
        "execution_run_id": execution_run_id,
        "execution_attempt_id": attempt_id,
        "remote_output_uri": remote_output_uri,
        "source_bundle_sha256": env["EXPECTED_SOURCE_BUNDLE_SHA256"],
        "submitted_code_revision": env["SUBMITTED_CODE_REVISION"],
    }
    if source != expected_source:
        raise EvidenceError("source identity does not match tracked B4 canary")

    _exact_keys(
        terminal,
        {
            "artifact_sync_attempted",
            "workload_returncode",
            "execution_attempt_id",
            "remote_output_uri",
        },
        "terminal status",
    )
    if terminal != {
        "artifact_sync_attempted": True,
        "workload_returncode": 0,
        "execution_attempt_id": attempt_id,
        "remote_output_uri": remote_output_uri,
    }:
        raise EvidenceError("terminal status does not bind successful B4 artifacts")

    _exact_keys(marker, set(PREFLIGHT_KEYS) | {"status", "marker_sha256"}, "marker")
    marker_without_hash = {
        key: value for key, value in marker.items() if key != "marker_sha256"
    }
    if marker_without_hash != {**preflight, "status": "passed"} or marker.get(
        "marker_sha256"
    ) != _canonical_sha256(marker_without_hash):
        raise EvidenceError("runtime preflight marker is not exact")

    results_sha256 = checksums.get("selector-results.jsonl")
    if summary.get("results_sha256") != results_sha256:
        raise EvidenceError("summary result checksum does not match exact result bytes")
    if len(result_rows) != CANARY_ROW_COUNT:
        return
    if _ordered_ids_sha256(result_rows) != preflight["request_ids_sha256"]:
        raise EvidenceError("result IDs do not match the canary request identity")
    for row in result_rows:
        for result_field, preflight_field in RESULT_IDENTITY_FIELDS.items():
            if row.get(result_field) != preflight.get(preflight_field):
                raise EvidenceError("a result row has a mixed execution identity")


def _h100_identity_gate(artifacts: Mapping[str, Any]) -> bool:
    summary = _mapping(artifacts["summary"], "summary")
    results = _sequence(artifacts["results"], "results")
    telemetry = _sequence(artifacts["telemetry"], "telemetry")
    summary_name = summary.get("gpu_name")
    summary_total = _finite_number(summary.get("gpu_total_memory_bytes"))
    if (
        not _is_h100_name(summary_name)
        or summary_total is None
        or summary_total < MIN_GPU_MEMORY_BYTES
    ):
        return False
    for row_value in results:
        row = _mapping(row_value, "result")
        total = _finite_number(row.get("gpu_total_memory_bytes"))
        if (
            row.get("gpu_name") != summary_name
            or not _is_h100_name(row.get("gpu_name"))
            or total is None
            or total < MIN_GPU_MEMORY_BYTES
        ):
            return False
    for sample_value in telemetry:
        sample = _mapping(sample_value, "telemetry sample")
        total = _finite_number(sample.get("memory_total_bytes"))
        if (
            sample.get("gpu_name") != summary_name
            or not _is_h100_name(sample.get("gpu_name"))
            or total is None
            or total < MIN_GPU_MEMORY_BYTES
        ):
            return False
    return bool(results) and bool(telemetry)


def _attempts(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    collected: list[Mapping[str, Any]] = []
    for row in results:
        raw_attempts = row.get("attempts")
        if not isinstance(raw_attempts, list):
            continue
        collected.extend(
            attempt for attempt in raw_attempts if isinstance(attempt, Mapping)
        )
    return collected


def project_full_panel(
    *,
    original_ten_generation_seconds: float,
    model_load_seconds: float,
    actual_billed_cost_usd: float,
) -> dict[str, Any]:
    """Project four concurrent shards using exact frozen row counts."""

    per_row_seconds = original_ten_generation_seconds / ORIGINAL_CANARY_ROW_COUNT
    shard_seconds = [
        model_load_seconds + per_row_seconds * row_count
        for row_count in SHARD_ROW_COUNTS
    ]
    aggregate_seconds = sum(shard_seconds)
    aggregate_gpu_hours = aggregate_seconds / 3600.0
    longest_shard_hours = max(shard_seconds) / 3600.0
    compute_usd = aggregate_gpu_hours * H100_PRICE_USD_PER_HOUR
    projected_billed_cost_usd = actual_billed_cost_usd + compute_usd
    gates = {
        "aggregate_gpu_hours": aggregate_gpu_hours <= MAX_AGGREGATE_GPU_HOURS,
        "longest_shard_hours": longest_shard_hours <= MAX_LONGEST_SHARD_HOURS,
        "compute_usd": compute_usd <= MAX_COMPUTE_USD,
        "projected_billed_cost_usd": (
            projected_billed_cost_usd <= MAX_PROJECTED_BILLED_COST_USD
        ),
    }
    return {
        "panel_row_count": FULL_PANEL_COUNT,
        "shard_row_counts": list(SHARD_ROW_COUNTS),
        "per_row_generation_seconds": per_row_seconds,
        "shard_seconds": shard_seconds,
        "aggregate_seconds": aggregate_seconds,
        "aggregate_gpu_hours": aggregate_gpu_hours,
        "longest_shard_hours": longest_shard_hours,
        "compute_usd": compute_usd,
        "projected_billed_cost_usd": projected_billed_cost_usd,
        "gates": gates,
    }


def _generation_gates(attempts: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    eos = True
    modality = True
    memory = True
    generation_failures = False
    elapsed_valid = True
    for attempt in attempts:
        elapsed = _finite_number(attempt.get("elapsed_seconds"))
        elapsed_valid = elapsed_valid and elapsed is not None and elapsed >= 0
        if attempt.get("failure_stage") == "generation":
            generation_failures = True
        generation = attempt.get("generation")
        if not isinstance(generation, Mapping):
            eos = modality = memory = False
            continue
        eos = eos and generation.get("ended_with_eos") is True
        eos = (
            eos and generation.get("final_generated_token_id") == EXPECTED_EOS_TOKEN_ID
        )
        attestation = generation.get("modality_attestation")
        modality = modality and isinstance(attestation, Mapping)
        if isinstance(attestation, Mapping):
            modality = modality and (
                attestation.get("same_first_forward") is True
                and attestation.get("native_video_element_count") == 1
                and attestation.get("native_audio_element_count") == 1
            )
        total = _finite_number(generation.get("gpu_total_memory_bytes"))
        allocated = _finite_number(generation.get("peak_memory_allocated_bytes"))
        reserved = _finite_number(generation.get("peak_memory_reserved_bytes"))
        memory = memory and total is not None and total >= MIN_GPU_MEMORY_BYTES
        if total is None:
            memory = False
        else:
            memory = memory and allocated is not None
            if allocated is not None:
                memory = memory and allocated <= total * MAX_PEAK_ALLOCATED_FRACTION
            memory = memory and reserved is not None
            if reserved is not None:
                memory = memory and total - reserved >= MIN_FREE_MEMORY_BYTES
    return {
        "eos": eos,
        "modality": modality,
        "memory": memory,
        "generation_failures": not generation_failures,
        "attempt_timings": elapsed_valid,
    }


def _bounded_failure(reason: str) -> dict[str, Any]:
    gates = {name: False for name in DECISION_GATE_NAMES}
    return {
        "schema_version": 2,
        "status": "no-go",
        "gates": gates,
        "failure_reasons": [reason],
        "observed": {
            "dstack_run_uuid": None,
            "execution_attempt_id": None,
            "result_row_count": None,
            "original_row_count": None,
            "original_ten_repairs": None,
            "original_ten_generation_seconds": None,
            "model_load_seconds": None,
            "sampled_free_memory_bytes": None,
            "actual_billed_cost_usd": None,
        },
        "projection": None,
    }


def adjudicate_canary(
    artifact_dir: Path,
    billing_evidence: Mapping[str, Any] | Path,
    *,
    expected_config_path: Path | None = None,
) -> dict[str, Any]:
    """Return a bounded B4 decision tied to exact artifacts and dstack billing."""

    try:
        expected = _load_expected_configuration(
            expected_config_path or CANARY_CONFIG_PATH
        )
        billing = _validate_billing_evidence(
            _load_billing_evidence(billing_evidence), expected
        )
    except (EvidenceError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return _bounded_failure("billing_evidence")
    try:
        artifacts = _load_artifacts(Path(artifact_dir))
    except (EvidenceError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return _bounded_failure("artifact_integrity")
    try:
        _validate_artifact_identity(artifacts, expected)
    except (EvidenceError, TypeError, ValueError):
        return _bounded_failure("evidence_identity")

    summary = _mapping(artifacts["summary"], "summary")
    result_rows = [
        _mapping(row, "result") for row in _sequence(artifacts["results"], "results")
    ]
    telemetry = [
        _mapping(row, "telemetry")
        for row in _sequence(artifacts["telemetry"], "telemetry")
    ]
    sentinel_hash = summary.get("resource_sentinel_id_sha256")
    sentinel_rows = [
        row
        for row in result_rows
        if isinstance(row.get("video_id"), str)
        and isinstance(sentinel_hash, str)
        and hashlib.sha256(row["video_id"].encode("utf-8")).hexdigest() == sentinel_hash
    ]
    original_rows = [row for row in result_rows if row not in sentinel_rows]
    all_attempts = _attempts(result_rows)
    original_attempts = _attempts(original_rows)
    all_generation_seconds = sum(
        _finite_number(attempt.get("elapsed_seconds")) or 0.0
        for attempt in all_attempts
    )
    original_generation_seconds = sum(
        _finite_number(attempt.get("elapsed_seconds")) or 0.0
        for attempt in original_attempts
    )
    model_load_seconds = _finite_number(summary.get("model_load_seconds"))
    actual_cost = billing["cost"]
    free_samples = [
        value
        for row in telemetry
        for value in [_finite_number(row.get("memory_free_bytes"))]
        if value is not None
    ]
    min_free_bytes = min(free_samples) if free_samples else None
    generation_gates = _generation_gates(all_attempts)
    generation_evidence = len(all_attempts) >= CANARY_ROW_COUNT and all(
        isinstance(row.get("attempts"), list) and bool(row["attempts"])
        for row in result_rows
    )
    top_level_generation_failure = any(
        row.get("failure_stage") == "generation" for row in result_rows
    )
    result_ids_hash = _ordered_ids_sha256(result_rows)
    gates: dict[str, bool] = {
        "billing_evidence": True,
        "artifact_integrity": True,
        "evidence_identity": True,
        "h100_identity": _h100_identity_gate(artifacts),
        "request_filename": summary.get("requests_filename")
        == EXPECTED_REQUEST_FILENAME,
        "panel_id_digest": summary.get("panel_ids_sha256")
        == CANONICAL_PANEL_IDS_SHA256,
        "canary_row_count": len(result_rows) == CANARY_ROW_COUNT
        and summary.get("rows") == CANARY_ROW_COUNT,
        "sentinel_identity": len(sentinel_rows) == 1,
        "valid_rows": len(result_rows) == CANARY_ROW_COUNT
        and all(row.get("status") == "passed" for row in result_rows)
        and summary.get("passed") == CANARY_ROW_COUNT
        and summary.get("failed") == 0,
        "result_id_digest": result_ids_hash == summary.get("request_ids_sha256"),
        "original_ten_repairs": len(original_rows) == ORIGINAL_CANARY_ROW_COUNT
        and sum(row.get("repair_used") is True for row in original_rows) <= 2,
        "eos": generation_gates["eos"],
        "modality": generation_gates["modality"],
        "memory": generation_gates["memory"],
        "generation_failures": generation_gates["generation_failures"],
        "generation_evidence": generation_evidence,
        "top_level_generation_failures": not top_level_generation_failure,
        "attempt_timings": generation_gates["attempt_timings"],
        "original_ten_generation_seconds": original_generation_seconds
        <= MAX_ORIGINAL_GENERATION_SECONDS,
        "summary_generation_seconds": math.isclose(
            all_generation_seconds,
            _finite_number(summary.get("elapsed_generation_seconds")) or math.nan,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "model_load_seconds": model_load_seconds is not None
        and 0 <= model_load_seconds <= MAX_MODEL_LOAD_SECONDS,
        "sampled_free_memory": min_free_bytes is not None
        and min_free_bytes >= MIN_FREE_MEMORY_BYTES,
        "actual_billed_cost": 0 <= actual_cost <= MAX_CANARY_BILLED_COST_USD,
    }
    projection = project_full_panel(
        original_ten_generation_seconds=original_generation_seconds,
        model_load_seconds=model_load_seconds or 0.0,
        actual_billed_cost_usd=actual_cost,
    )
    gates.update(projection["gates"])
    reasons = [name for name, passed in gates.items() if not passed]
    env = _mapping(expected["env"], "expected environment")
    return {
        "schema_version": 2,
        "status": "go" if not reasons else "no-go",
        "gates": gates,
        "failure_reasons": reasons,
        "observed": {
            "dstack_run_uuid": billing["run_uuid"],
            "execution_attempt_id": env["EXECUTION_ATTEMPT_ID"],
            "result_row_count": len(result_rows),
            "original_row_count": len(original_rows),
            "original_ten_repairs": sum(
                row.get("repair_used") is True for row in original_rows
            ),
            "original_ten_generation_seconds": original_generation_seconds,
            "model_load_seconds": model_load_seconds,
            "sampled_free_memory_bytes": min_free_bytes,
            "actual_billed_cost_usd": actual_cost,
        },
        "projection": projection,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sanitize = commands.add_parser("sanitize-billing")
    sanitize.add_argument("--dstack-ps-json", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    decide = commands.add_parser("adjudicate")
    decide.add_argument("--artifacts", type=Path, required=True)
    decide.add_argument("--billing-evidence", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "sanitize-billing":
        try:
            sanitized = sanitize_dstack_billing_evidence(
                _mapping(_load_json(args.dstack_ps_json), "dstack ps payload")
            )
        except (EvidenceError, OSError, json.JSONDecodeError, TypeError, ValueError):
            print(
                json.dumps({"status": "failed", "failure_reason": "billing_evidence"})
            )
            return 2
        _write_json(args.output, sanitized)
        print(json.dumps({"status": "written", "schema_version": 1}))
        return 0
    decision = adjudicate_canary(args.artifacts, args.billing_evidence)
    _write_json(args.output, decision)
    print(json.dumps(decision, sort_keys=True))
    return 0 if decision["status"] == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())
