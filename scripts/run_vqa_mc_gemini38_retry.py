#!/usr/bin/env python3
"""Retry only invalid rows from the frozen Gemini 3.8 V3 replication."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import run_vqa_deadline_gemini_full as deadline
import run_vqa_mc_gemini38_static_full as g38
import package_vqa_mc_submissions as packager


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.8-flash"
DEFAULT_SOURCE = (
    ROOT / "data/vqa/benchmarks/vqa-mc-gemini38-static-task2-v2-20260912-a1"
)
DEFAULT_OUTPUT = (
    ROOT / "data/vqa/benchmarks/vqa-mc-gemini38-static-task2-v2-20260913-retry-a2"
)
DEFAULT_TASK2 = ROOT / "datasets/trecvid-2026/vqa/task2/vqa.2026.testingDataset.Task2_v2.json"
DEFAULT_CANDIDATE = DEFAULT_OUTPUT / "submission-v3-rescued.json"
DEFAULT_PACKAGE = (
    ROOT
    / "submissions/trecvid-2026/vqa-mc-task2-v2/g38_mc_direct/submission.json"
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def select_retry_inputs(
    *, protocol: Mapping[str, Any], freeze: Mapping[str, Any], v1: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select exactly the source run's frozen invalid rows, never unavailable rows."""

    if protocol.get("model") != MODEL or freeze.get("model") != MODEL:
        raise ValueError("source model is not Gemini 3.8 Flash")
    if freeze.get("status") != "predictions_frozen":
        raise ValueError("source predictions are not frozen")
    protocol_rows = protocol.get("rows")
    prediction_rows = freeze.get("rows")
    weak_query_ids = protocol.get("weak_query_ids")
    if not isinstance(protocol_rows, list) or not isinstance(prediction_rows, list):
        raise ValueError("source rows are invalid")
    if not isinstance(weak_query_ids, list):
        raise ValueError("source weak Q_IDs are invalid")
    invalid_ids = [
        int(row["query_id"])
        for row in prediction_rows
        if isinstance(row, Mapping) and row.get("status") == "invalid"
    ]
    if not invalid_ids or len(invalid_ids) != len(set(invalid_ids)):
        raise ValueError("source invalid-row set is empty or duplicated")
    invalid_set = set(invalid_ids)
    rows = [dict(row) for row in protocol_rows if row.get("query_id") in invalid_set]
    v1_rows = [dict(row) for row in v1 if row.get("Q_ID") in invalid_set]
    if [row.get("query_id") for row in rows] != invalid_ids:
        raise ValueError("source invalid-row order differs from protocol")
    if [row.get("Q_ID") for row in v1_rows] != invalid_ids:
        raise ValueError("source invalid-row order differs from V1")
    return {
        "rows": rows,
        "v1": v1_rows,
        "weak_query_ids": [qid for qid in weak_query_ids if qid in invalid_set],
    }


def overlay_successes(
    *,
    source_submission: Sequence[Mapping[str, Any]],
    retry_submission: Sequence[Mapping[str, Any]],
    retry_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay only retry rows that passed strict parsing."""

    passed = {
        int(row["query_id"])
        for row in retry_rows
        if isinstance(row, Mapping) and row.get("status") == "passed"
    }
    retry_index = {int(row["Q_ID"]): dict(row) for row in retry_submission}
    if not passed <= retry_index.keys():
        raise ValueError("passed retry rows are absent from retry submission")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source_row in source_submission:
        query_id = source_row.get("Q_ID")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id in seen:
            raise ValueError("source submission Q_ID coverage is invalid")
        seen.add(query_id)
        replacement = retry_index.get(query_id) if query_id in passed else None
        if replacement is not None and replacement.get("Video_ID") != source_row.get("Video_ID"):
            raise ValueError("retry submission video identity differs")
        output.append(replacement or dict(source_row))
    return output


def build_retry_protocol(
    *,
    rows: Sequence[Mapping[str, Any]],
    weak_query_ids: Sequence[int],
    source_protocol: Mapping[str, Any],
    max_cost_usd: float,
    created_at: str,
) -> dict[str, Any]:
    """Build a subset protocol without regenerating or changing source media."""

    runner = g38.configured_runner()
    if source_protocol.get("model") != MODEL:
        raise ValueError("source model is not Gemini 3.8 Flash")
    if source_protocol.get("model_arm") != runner.MODEL_ARM:
        raise ValueError("source model arm differs from V3")
    settings = source_protocol.get("settings")
    source_values = source_protocol.get("source")
    if not isinstance(settings, Mapping) or not isinstance(source_values, Mapping):
        raise ValueError("source protocol provenance is invalid")
    frozen_rows = [dict(row) for row in rows]
    ready = [row for row in frozen_rows if row.get("status") == "ready"]
    estimate = runner.estimate_cost(
        ready_rows=len(ready),
        ready_video_seconds=sum(float(row["duration_seconds"]) for row in ready),
    )
    runner.assert_cost_gate(estimate, max_cost_usd=max_cost_usd)
    return {
        "schema_version": 1,
        "experiment_id": "vqa-mc-gemini38-static-invalid-retry-20260913",
        "status": "prepared",
        "created_at": created_at,
        "model": MODEL,
        "model_arm": runner.MODEL_ARM,
        "settings": dict(settings),
        "source": dict(source_values),
        "weak_query_ids": sorted(weak_query_ids),
        "estimate": estimate,
        "max_cost_usd": max_cost_usd,
        "rows": frozen_rows,
    }


def prepare_retry(*, source: Path, output: Path, max_cost_usd: float) -> dict[str, Any]:
    runner = g38.configured_runner()
    source = Path(source)
    protocol = runner._protocol(source)
    freeze = runner._read_value(source / "predictions-freeze.json")
    v1 = runner._read_value(source / "v1-submission.json")
    if not isinstance(freeze, Mapping) or not isinstance(v1, list):
        raise ValueError("source artifacts are invalid")
    selected = select_retry_inputs(protocol=protocol, freeze=freeze, v1=v1)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    retry_protocol = build_retry_protocol(
        rows=selected["rows"],
        weak_query_ids=selected["weak_query_ids"],
        source_protocol=protocol,
        max_cost_usd=max_cost_usd,
        created_at=_now(),
    )
    runner._v1_index(selected["v1"], retry_protocol["rows"])
    output.mkdir(parents=True)
    for name in ("uploads", "jobs", "status", "deliveries"):
        (output / name).mkdir()
    deadline._write_json(output / "protocol.json", retry_protocol)
    v1_artifact_sha256 = runner._write_value_once(
        output / "v1-submission.json", selected["v1"]
    )
    ready = [row for row in retry_protocol["rows"] if row.get("status") == "ready"]
    manifest = {
        "schema_version": 1,
        "experiment_id": retry_protocol["experiment_id"],
        "status": "prepared",
        "created_at": retry_protocol["created_at"],
        "protocol_sha256": deadline._sha256_file(output / "protocol.json"),
        "v1_artifact_sha256": v1_artifact_sha256,
        "row_count": len(retry_protocol["rows"]),
        "ready_count": len(ready),
        "unavailable_count": 0,
        "unsupported_count": 0,
        "unique_ready_video_count": len({str(row["video_id"]) for row in ready}),
        "estimate": retry_protocol["estimate"],
        "max_cost_usd": max_cost_usd,
    }
    deadline._write_json(output / "manifest.json", manifest)
    for row in selected["rows"]:
        video_id = str(row["video_id"])
        upload = deadline._read_json(source / "uploads" / f"{video_id}.json")
        if upload.get("media_sha256") != row.get("media_sha256"):
            raise ValueError("source upload media hash differs from frozen protocol")
        runner._write_value_once(output / "uploads" / f"{video_id}.json", upload)
    retry_manifest = {
        "schema_version": 1,
        "status": "prepared",
        "source_root_name": source.name,
        "source_protocol_sha256": deadline._sha256_file(source / "protocol.json"),
        "source_predictions_freeze_sha256": deadline._sha256_file(
            source / "predictions-freeze.json"
        ),
        "source_submission_v3_sha256": deadline._sha256_file(source / "submission-v3.json"),
        "retry_query_ids": [row["query_id"] for row in selected["rows"]],
        "retry_count": len(selected["rows"]),
        "max_cost_usd": max_cost_usd,
    }
    runner._write_value_once(output / "retry-manifest.json", retry_manifest)
    return {
        "status": "prepared",
        "row_count": len(retry_protocol["rows"]),
        "ready_count": len(ready),
        "unavailable_count": 0,
        "projected_cost_usd": retry_protocol["estimate"]["projected_cost_usd"],
        "retry_count": len(selected["rows"]),
    }


def freeze_retry(*, source: Path, output: Path) -> dict[str, Any]:
    runner = g38.configured_runner()
    result = runner.freeze_run(root=output, frozen_at=_now())
    retry_freeze = runner._read_value(Path(output) / "predictions-freeze.json")
    retry_submission = runner._read_value(Path(output) / "submission-v3.json")
    source_submission = runner._read_value(Path(source) / "submission-v3.json")
    if not isinstance(retry_freeze, Mapping):
        raise ValueError("retry freeze is invalid")
    if not isinstance(retry_submission, list) or not isinstance(source_submission, list):
        raise ValueError("submission artifacts are invalid")
    retry_rows = retry_freeze.get("rows")
    if not isinstance(retry_rows, list):
        raise ValueError("retry prediction rows are invalid")
    merged = overlay_successes(
        source_submission=source_submission,
        retry_submission=retry_submission,
        retry_rows=retry_rows,
    )
    merged_sha256 = runner._write_value_once(
        Path(output) / "submission-v3-rescued.json", merged
    )
    rescue_summary = {
        "schema_version": 1,
        "status": "complete",
        "retry_count": len(retry_rows),
        "recovered_count": sum(row.get("status") == "passed" for row in retry_rows),
        "remaining_invalid_count": sum(row.get("status") != "passed" for row in retry_rows),
        "observed_retry_cost_usd": retry_freeze.get("observed_cost_usd"),
        "usage_complete": retry_freeze.get("usage_complete"),
        "retry_predictions_freeze_sha256": deadline._sha256_file(
            Path(output) / "predictions-freeze.json"
        ),
        "rescued_submission_v3_sha256": merged_sha256,
    }
    runner._write_value_once(Path(output) / "rescue-summary.json", rescue_summary)
    return {**result, **rescue_summary}


def package_submission(
    *,
    candidate: Path,
    task2: Path,
    target: Path,
    expected_count: int = 2_443,
) -> dict[str, Any]:
    """Validate and package one Gemini 3.8 organizer upload artifact."""

    candidate = Path(candidate)
    target = Path(target)
    if target.name != "submission.json" or target.parent.name != "g38_mc_direct":
        raise ValueError("Gemini 3.8 upload target must use the approved run tag")
    candidate_bytes = candidate.read_bytes()
    submission = json.loads(candidate_bytes)
    task_rows = json.loads(Path(task2).read_bytes())
    validation = packager.validate_mc_submission(
        submission=submission,
        task_rows=task_rows,
        expected_count=expected_count,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or target.is_symlink() or target.read_bytes() != candidate_bytes:
            raise ValueError("immutable Gemini 3.8 upload artifact changed")
    else:
        target.write_bytes(candidate_bytes)
    return {
        "status": "packaged",
        "run_tag": target.parent.name,
        "path": str(target),
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "size_bytes": len(candidate_bytes),
        **validation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--max-cost-usd", type=float, required=True)
    for name in ("submit", "status", "freeze", "show"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    package = commands.add_parser("package")
    package.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    package.add_argument("--task2", type=Path, default=DEFAULT_TASK2)
    package.add_argument("--target", type=Path, default=DEFAULT_PACKAGE)
    return parser


def _emit(result: Mapping[str, Any]) -> None:
    allowed = (
        "status", "retry_count", "request_count", "request_bytes", "response_count",
        "passed_count", "fallback_count", "recovered_count", "remaining_invalid_count",
        "projected_cost_usd", "observed_cost_usd", "observed_retry_cost_usd",
        "run_tag", "path", "sha256", "size_bytes", "row_count",
    )
    print(json.dumps({key: result[key] for key in allowed if key in result}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        runner = g38.configured_runner()
        if args.command == "prepare":
            result = prepare_retry(
                source=args.source, output=args.output, max_cost_usd=args.max_cost_usd
            )
        elif args.command == "submit":
            result = runner.submit_batch(root=args.output, client=runner._client(), submitted_at=_now())
        elif args.command == "status":
            result = runner.record_status(root=args.output, client=runner._client(), checked_at=_now())
        elif args.command == "freeze":
            result = freeze_retry(source=args.source, output=args.output)
        elif args.command == "show":
            summary = args.output / "rescue-summary.json"
            result = runner._read_value(summary) if summary.exists() else runner._protocol(args.output)
        elif args.command == "package":
            result = package_submission(
                candidate=args.candidate,
                task2=args.task2,
                target=args.target,
            )
        else:  # pragma: no cover
            raise ValueError("invalid command")
        _emit(result)
        return 0
    except Exception:  # noqa: BLE001 - do not leak private competition evidence
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
