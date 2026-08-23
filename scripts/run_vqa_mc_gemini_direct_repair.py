#!/usr/bin/env python3
"""Retry only invalid direct-video MC outputs and derive V2b/V3b.

The parent V2/V3 freeze remains immutable. Provider-mutating phases operate on
a separate nine-row repair root; finalization overlays only valid repaired
rankings and retains the parent's explicit fallbacks otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_vqa_deadline_gemini_full as deadline
import run_vqa_mc_gemini_direct_full as direct


ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = (
    ROOT / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2-v3-20260822-a1"
)
DEFAULT_OUTPUT = (
    ROOT / "data/vqa/benchmarks/vqa-mc-gemini-direct-v2b-v3b-repair9-20260823-a1"
)
EXPERIMENT_ID = "vqa-mc-gemini-direct-v2b-v3b-repair9-20260823"
EXPECTED_INVALID_COUNT = 9


def _rows(value: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{label} rows are invalid")
    return rows


def select_repair_inputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    v1_submission: Sequence[Mapping[str, Any]],
    weak_query_ids: set[int],
    parent_freeze: Mapping[str, Any],
    expected_invalid_count: int,
) -> dict[str, Any]:
    """Select exactly the parent's invalid rows, never unavailable fallbacks."""

    if (
        isinstance(expected_invalid_count, bool)
        or not isinstance(expected_invalid_count, int)
        or expected_invalid_count <= 0
    ):
        raise ValueError("expected invalid count is invalid")
    parent_rows = _rows(parent_freeze, label="parent freeze")
    parent_by_id = {int(row["query_id"]): row for row in parent_rows}
    row_by_id = {int(row["query_id"]): row for row in rows}
    if len(parent_by_id) != len(parent_rows) or set(parent_by_id) != set(row_by_id):
        raise ValueError("parent freeze coverage differs from Task 2")
    invalid_ids = {
        query_id for query_id, row in parent_by_id.items() if row.get("status") == "invalid"
    }
    if len(invalid_ids) != expected_invalid_count:
        raise ValueError("parent invalid count differs from repair contract")
    selected_rows = [dict(row) for row in rows if int(row["query_id"]) in invalid_ids]
    if any(row.get("status") != "ready" for row in selected_rows):
        raise ValueError("an invalid parent row is not media-ready")
    v1_by_id = {int(row["Q_ID"]): row for row in v1_submission}
    if len(v1_by_id) != len(v1_submission) or not invalid_ids <= set(v1_by_id):
        raise ValueError("V1 coverage differs from repair rows")
    return {
        "rows": selected_rows,
        "v1_submission": [dict(v1_by_id[int(row["query_id"])]) for row in selected_rows],
        "weak_query_ids": weak_query_ids & invalid_ids,
        "invalid_query_ids": sorted(invalid_ids),
    }


def assemble_repaired_candidates(
    *,
    rows: Sequence[Mapping[str, Any]],
    v1_submission: Sequence[Mapping[str, Any]],
    weak_query_ids: set[int],
    parent_freeze: Mapping[str, Any],
    repair_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Overlay valid retries onto parent predictions and assemble V2b/V3b."""

    parent_rows = _rows(parent_freeze, label="parent freeze")
    repair_rows = _rows(repair_freeze, label="repair freeze")
    parent_by_id = {int(row["query_id"]): dict(row) for row in parent_rows}
    repair_by_id = {int(row["query_id"]): row for row in repair_rows}
    if len(parent_by_id) != len(parent_rows) or len(repair_by_id) != len(repair_rows):
        raise ValueError("prediction Q_IDs are duplicated")
    invalid_ids = {
        query_id for query_id, row in parent_by_id.items() if row.get("status") == "invalid"
    }
    if set(repair_by_id) != invalid_ids:
        raise ValueError("repair Q_ID coverage differs from parent invalid coverage")
    recovered: list[int] = []
    for query_id, repaired in repair_by_id.items():
        if repaired.get("status") == "passed":
            parent_by_id[query_id] = dict(repaired)
            recovered.append(query_id)
    assembled = direct.assemble_submissions(
        rows=rows,
        predictions=parent_by_id,
        v1_submission=v1_submission,
        weak_query_ids=weak_query_ids,
    )
    remaining = sorted(
        query_id
        for query_id in invalid_ids
        if parent_by_id[query_id].get("status") != "passed"
    )
    return {
        "v2b": assembled["v2"],
        "v3b": assembled["v3"],
        "v2b_direct_count": assembled["v2_direct_count"],
        "v3b_direct_count": assembled["v3_direct_count"],
        "recovered_query_ids": sorted(recovered),
        "remaining_invalid_query_ids": remaining,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_repair(*, output: Path, max_cost_usd: float) -> dict[str, Any]:
    task2 = direct._task2_bytes()
    rows = direct.load_panel(
        task2=task2,
        requests_path=direct.REQUESTS_PATH,
        media_root=direct.MEDIA_ROOT,
    )
    v1 = direct._json_source(direct.V1_PATH)
    parent_path = PARENT_ROOT / "predictions-freeze.json"
    parent_freeze = direct._read_value(parent_path)
    if not isinstance(v1, list) or not isinstance(parent_freeze, Mapping):
        raise ValueError("repair sources are invalid")
    selected = select_repair_inputs(
        rows=rows,
        v1_submission=v1,
        weak_query_ids=direct._weak_query_ids(),
        parent_freeze=parent_freeze,
        expected_invalid_count=EXPECTED_INVALID_COUNT,
    )
    result = direct.prepare_run(
        rows=selected["rows"],
        root=output,
        v1_submission=selected["v1_submission"],
        weak_query_ids=selected["weak_query_ids"],
        task2_sha256=hashlib.sha256(task2).hexdigest(),
        v1_sha256=deadline._sha256_file(direct.V1_PATH),
        weak_queue_sha256=deadline._sha256_file(direct.WEAK_QUEUE_PATH),
        max_cost_usd=max_cost_usd,
        created_at=direct._now(),
        experiment_id=EXPERIMENT_ID,
    )
    binding = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": deadline._sha256_file(output / "protocol.json"),
        "parent_predictions_freeze_sha256": _sha(parent_path),
        "parent_submission_v2_sha256": _sha(PARENT_ROOT / "submission-v2.json"),
        "parent_submission_v3_sha256": _sha(PARENT_ROOT / "submission-v3.json"),
        "invalid_query_ids": selected["invalid_query_ids"],
    }
    direct._write_value_once(output / "repair-parent.json", binding)
    return result


def finalize_repair(*, output: Path) -> dict[str, Any]:
    binding = direct._read_value(output / "repair-parent.json")
    if not isinstance(binding, Mapping) or binding.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("repair parent binding is invalid")
    parent_path = PARENT_ROOT / "predictions-freeze.json"
    expected_hashes = {
        "parent_predictions_freeze_sha256": _sha(parent_path),
        "parent_submission_v2_sha256": _sha(PARENT_ROOT / "submission-v2.json"),
        "parent_submission_v3_sha256": _sha(PARENT_ROOT / "submission-v3.json"),
    }
    if any(binding.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("parent artifact changed after repair preparation")
    task2 = direct._task2_bytes()
    rows = direct.load_panel(
        task2=task2,
        requests_path=direct.REQUESTS_PATH,
        media_root=direct.MEDIA_ROOT,
    )
    v1 = direct._json_source(direct.V1_PATH)
    parent_freeze = direct._read_value(parent_path)
    repair_freeze = direct._read_value(output / "predictions-freeze.json")
    repair_summary = deadline._read_json(output / "summary.json")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (v1, list),
            (parent_freeze, Mapping),
            (repair_freeze, Mapping),
        )
    ):
        raise ValueError("repair finalization sources are invalid")
    assembled = assemble_repaired_candidates(
        rows=rows,
        v1_submission=v1,
        weak_query_ids=direct._weak_query_ids(),
        parent_freeze=parent_freeze,
        repair_freeze=repair_freeze,
    )
    v2b_sha = direct._write_value_once(output / "submission-v2b.json", assembled["v2b"])
    v3b_sha = direct._write_value_once(output / "submission-v3b.json", assembled["v3b"])
    final = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "parent_predictions_freeze_sha256": expected_hashes[
            "parent_predictions_freeze_sha256"
        ],
        "repair_predictions_freeze_sha256": _sha(output / "predictions-freeze.json"),
        "requested_retry_count": len(binding["invalid_query_ids"]),
        "recovered_count": len(assembled["recovered_query_ids"]),
        "remaining_invalid_count": len(assembled["remaining_invalid_query_ids"]),
        "remaining_unavailable_count": 10,
        "v2b_direct_count": assembled["v2b_direct_count"],
        "v3b_direct_count": assembled["v3b_direct_count"],
        "v2b_sha256": v2b_sha,
        "v3b_sha256": v3b_sha,
        "retry_cost_usd": repair_summary.get("observed_cost_usd"),
    }
    deadline._write_json_once(output / "repair-summary.json", final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_mc_gemini_direct_repair")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--max-cost-usd", type=float, required=True)
    for name in ("upload", "submit", "status", "freeze", "finalize", "show"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _emit(result: Mapping[str, Any]) -> None:
    allowed = (
        "status",
        "row_count",
        "ready_count",
        "uploaded_count",
        "failed_count",
        "request_count",
        "response_count",
        "passed_count",
        "fallback_count",
        "projected_cost_usd",
        "observed_cost_usd",
        "requested_retry_count",
        "recovered_count",
        "remaining_invalid_count",
        "remaining_unavailable_count",
        "v2b_direct_count",
        "v3b_direct_count",
        "retry_cost_usd",
    )
    print(json.dumps({key: result[key] for key in allowed if key in result}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "prepare":
            result = prepare_repair(output=args.output, max_cost_usd=args.max_cost_usd)
        elif args.command == "upload":
            result = direct.upload_run(
                root=args.output, client=direct._client(), sleep=direct.time.sleep
            )
        elif args.command == "submit":
            result = direct.submit_batch(
                root=args.output, client=direct._client(), submitted_at=direct._now()
            )
        elif args.command == "status":
            result = direct.record_status(
                root=args.output, client=direct._client(), checked_at=direct._now()
            )
        elif args.command == "freeze":
            result = direct.freeze_run(root=args.output, frozen_at=direct._now())
        elif args.command == "finalize":
            result = finalize_repair(output=args.output)
        elif args.command == "show":
            final_path = args.output / "repair-summary.json"
            result = (
                deadline._read_json(final_path)
                if final_path.exists()
                else deadline._read_json(args.output / "manifest.json")
            )
        else:  # pragma: no cover
            raise ValueError("invalid command")
        _emit(result)
        return 0
    except Exception:  # noqa: BLE001 - CLI boundary must not leak private evidence
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
