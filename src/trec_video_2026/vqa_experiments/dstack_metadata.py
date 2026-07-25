"""Allowlisted projection of dstack run metadata for experiment artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


def _required(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"dstack metadata omitted {context}.{key}")
    return value


def sanitize_dstack_run(raw: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    run_spec = _required(raw, "run_spec", context="run")
    actual_run_id = _required(run_spec, "run_name", context="run_spec")
    if actual_run_id != run_id:
        raise ValueError(f"dstack run identity mismatch: {actual_run_id} != {run_id}")
    latest = _required(raw, "latest_job_submission", context="run")
    provisioning = _required(
        latest, "job_provisioning_data", context="latest_job_submission"
    )
    runtime = _required(latest, "job_runtime_data", context="latest_job_submission")
    offer = _required(runtime, "offer", context="job_runtime_data")
    instance_type = _required(
        provisioning, "instance_type", context="job_provisioning_data"
    )
    resources = _required(instance_type, "resources", context="instance_type")
    gpus = _required(resources, "gpus", context="resources")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError("dstack metadata must attest exactly one GPU")
    gpu = gpus[0]

    for key in ("backend", "region", "price"):
        provisioned_value = _required(
            provisioning, key, context="job_provisioning_data"
        )
        offered_value = _required(offer, key, context="offer")
        if provisioned_value != offered_value:
            raise ValueError(f"dstack {key} differs between offer and provisioning")

    status = _required(raw, "status", context="run")
    if status not in {"done", "failed", "terminated"}:
        raise ValueError(f"dstack run is not terminal: {status}")
    submitted_at = _required(raw, "submitted_at", context="run")
    finished_at = _required(latest, "finished_at", context="latest_job_submission")
    elapsed_seconds = (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(submitted_at)
    ).total_seconds()
    disk = _required(resources, "disk", context="resources")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "id": _required(raw, "id", context="run"),
        "status": status,
        "status_message": raw.get("status_message"),
        "termination_reason": raw.get("termination_reason"),
        "job_termination_reason": latest.get("termination_reason"),
        "submitted_at": submitted_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "cost": _required(raw, "cost", context="run"),
        "backend": provisioning["backend"],
        "region": provisioning["region"],
        "price": provisioning["price"],
        "instance_type": _required(instance_type, "name", context="instance_type"),
        "cpu_count": _required(resources, "cpus", context="resources"),
        "gpu_count": len(gpus),
        "gpu_name": _required(gpu, "name", context="gpu"),
        "gpu_vendor": _required(gpu, "vendor", context="gpu"),
        "gpu_memory_mib": _required(gpu, "memory_mib", context="gpu"),
        "host_memory_mib": _required(resources, "memory_mib", context="resources"),
        "disk_mib": _required(disk, "size_mib", context="disk"),
        "spot": _required(resources, "spot", context="resources"),
        "exit_status": latest.get("exit_status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    raw = json.load(sys.stdin)
    sanitized = sanitize_dstack_run(raw, run_id=args.run_id)
    args.output.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
