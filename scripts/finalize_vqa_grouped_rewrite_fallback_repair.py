#!/usr/bin/env python3
"""Merge fresh and deterministic grouped repairs into a full VQA submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import prepare_vqa_grouped_rewrite_fallback_repair as prepare
import run_vqa_grouped_rewrite_full as frozen
import salvage_vqa_grouped_rewrite_full as salvage

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION_ROOT = prepare.DEFAULT_OUTPUT_ROOT
DEFAULT_FRESH_ROOT = _ROOT / "data/vqa/benchmarks/grouped-rewrite-fallback-repair-20260817/fresh-290"
DEFAULT_EXISTING_ROOT = prepare.DEFAULT_SALVAGE_ROOT
DEFAULT_OUTPUT_ROOT = _ROOT / "data/vqa/benchmarks/grouped-rewrite-fallback-repair-20260817/final"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _deterministic_row(row: Mapping[str, Any], repair: Mapping[str, Any]) -> dict[str, Any]:
    result = _copy(row["fallback"])
    answers = repair.get("answers")
    if not isinstance(answers, list) or len(answers) != 10:
        raise ValueError(f"deterministic repair answer count is invalid: {row['query_id']}")
    result["Answers"] = [
        {"Rank": index, "Answer": answer["answer"], "Time": 0.0}
        for index, answer in enumerate(answers, 1)
    ]
    return result


def merge_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing_by_qid: Mapping[int, Mapping[str, Any]],
    existing_status_by_qid: Mapping[int, str],
    deterministic_by_qid: Mapping[int, Mapping[str, Any]],
    fresh_by_qid: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply fresh > deterministic > existing grouped > fusion precedence."""
    final_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for row in rows:
        query_id = int(row["query_id"])
        if query_id in fresh_by_qid:
            final = _copy(fresh_by_qid[query_id])
            status = "fresh_repaired"
        elif query_id in deterministic_by_qid:
            final = _deterministic_row(row, deterministic_by_qid[query_id])
            status = "deterministic_repaired"
        else:
            final = _copy(existing_by_qid.get(query_id, row["fallback"]))
            status = "existing_grouped" if existing_status_by_qid.get(query_id) != "fallback_fusion" else "fallback_fusion"
        if final.get("Q_ID") != query_id or final.get("Video_ID") != row["video_id"]:
            raise ValueError(f"merged question/video binding drift: {query_id}")
        answers = final.get("Answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError(f"merged answer count is invalid: {query_id}")
        if answers[0].get("Answer") != row["rank1_answer"]:
            raise ValueError(f"merged rank one changed: {query_id}")
        final_rows.append(final)
        provenance.append(
            {
                "Q_ID": query_id,
                "Video_ID": row["video_id"],
                "status": status,
                "answer_sha256": [_sha_text(str(answer["Answer"])) for answer in answers],
            }
        )
    return final_rows, provenance


def _validated_terminal_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root).resolve()
    summary_path = root / "summary.json"
    final_path = root / "final-organizer.json"
    provenance_path = root / "final-provenance.json"
    if not summary_path.is_file() or not final_path.is_file() or not provenance_path.is_file():
        raise ValueError(f"terminal repair input root is incomplete: {root}")
    summary = frozen._json(summary_path)
    final_rows = frozen._json(final_path)
    provenance = frozen._json(provenance_path)
    if (
        not isinstance(summary, Mapping)
        or summary.get("status") != "complete"
        or summary.get("final_sha256") != frozen._sha_file(final_path)
        or summary.get("provenance_sha256") != frozen._sha_file(provenance_path)
        or not isinstance(final_rows, list)
        or not isinstance(provenance, list)
    ):
        raise ValueError(f"terminal repair input hash binding is invalid: {root}")
    return dict(summary), final_rows, provenance


def finalize_repair(
    *,
    preparation_root: Path = DEFAULT_PREPARATION_ROOT,
    fresh_root: Path = DEFAULT_FRESH_ROOT,
    existing_root: Path = DEFAULT_EXISTING_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Freeze the full repaired panel after validating every input binding."""
    preparation_root = Path(preparation_root).resolve()
    fresh_root = Path(fresh_root).resolve()
    existing_root = Path(existing_root).resolve()
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"final repair root must be absent: {output_root}")
    prep_summary_path = preparation_root / "summary.json"
    prep_manifest_path = preparation_root / "manifest.json"
    deterministic_path = preparation_root / "deterministic-repairs.json"
    legacy_ids_path = preparation_root / "legacy-query-ids.json"
    if not all(path.is_file() for path in (prep_summary_path, prep_manifest_path, deterministic_path, legacy_ids_path)):
        raise ValueError("repair preparation root is incomplete")
    prep_summary = frozen._json(prep_summary_path)
    prep_manifest = frozen._json(prep_manifest_path)
    deterministic_rows = frozen._json(deterministic_path)
    legacy_ids = frozen._json(legacy_ids_path)
    if (
        not isinstance(prep_summary, Mapping)
        or prep_summary.get("status") != "complete"
        or prep_summary.get("manifest_file_sha256") != frozen._sha_file(prep_manifest_path)
        or prep_summary.get("deterministic_repairs_file_sha256") != frozen._sha_file(deterministic_path)
        or prep_summary.get("legacy_query_ids_file_sha256") != frozen._sha_file(legacy_ids_path)
        or not isinstance(prep_manifest, Mapping)
        or not isinstance(deterministic_rows, list)
        or not isinstance(legacy_ids, list)
    ):
        raise ValueError("repair preparation hash binding is invalid")
    if prep_manifest.get("deterministic_repairs_sha256") != prepare._sha_bytes(prepare._canonical(deterministic_rows)):
        raise ValueError("deterministic repair payload drift")
    if prep_manifest.get("legacy_query_ids_sha256") != prepare._sha_bytes(prepare._canonical(legacy_ids)):
        raise ValueError("legacy query selection drift")
    input_paths = {
        name: preparation_root / "inputs" / f"{name}.json"
        for name in ("questions", "gemini", "qwen", "control", "fusion", "provenance")
    }
    fresh_summary, fresh_rows, fresh_provenance = _validated_terminal_root(fresh_root)
    fresh_manifest = frozen._json(fresh_root / "manifest.json")
    expected_fresh_artifacts = frozen.source_artifacts(**input_paths)
    expected_fresh_artifacts_sha256 = frozen._sha_bytes(frozen._canonical(expected_fresh_artifacts))
    if (
        not isinstance(fresh_manifest, Mapping)
        or fresh_manifest.get("query_count") != len(legacy_ids)
        or fresh_manifest.get("global_max_cost_usd") != 5.0
        or fresh_manifest.get("source_artifacts") != expected_fresh_artifacts
        or fresh_manifest.get("source_artifacts_sha256") != expected_fresh_artifacts_sha256
        or fresh_summary.get("source_artifacts_sha256") != expected_fresh_artifacts_sha256
    ):
        raise ValueError("fresh repair manifest/source binding is invalid")
    legacy_set = set(legacy_ids)
    fresh_by_all = {int(row["Q_ID"]): row for row in fresh_rows if isinstance(row, Mapping)}
    fresh_status = {
        int(item["Q_ID"]): item.get("status")
        for item in fresh_provenance
        if isinstance(item, Mapping) and isinstance(item.get("Q_ID"), int)
    }
    if set(fresh_by_all) != legacy_set or set(fresh_status) != legacy_set:
        raise ValueError("fresh repair query coverage drift")
    if any(status not in {"generated", "fallback_fusion"} for status in fresh_status.values()):
        raise ValueError("fresh repair has an invalid terminal status")
    from trec_video_2026.vqa_deadline_canary import validate_ranked_submission

    validate_ranked_submission(fresh_rows, expected_query_ids=legacy_set)
    existing_summary, existing_rows, existing_provenance = _validated_terminal_root(existing_root)
    paths = prepare._source_paths(source_paths)
    canonical_rows = frozen.load_rows(**paths, expected_query_count=frozen.EXPECTED_QUERY_COUNT)
    expected_ids = {int(row["query_id"]) for row in canonical_rows}
    validate_ranked_submission(existing_rows, expected_query_ids=expected_ids)
    existing_by_qid = {int(row["Q_ID"]): row for row in existing_rows}
    existing_status = {
        int(item["Q_ID"]): str(item.get("status"))
        for item in existing_provenance
        if isinstance(item, Mapping) and isinstance(item.get("Q_ID"), int)
    }
    deterministic_by_qid = {
        int(item["query_id"]): item
        for item in deterministic_rows
        if isinstance(item, Mapping) and isinstance(item.get("query_id"), int)
    }
    if len(deterministic_by_qid) != 320 or set(deterministic_by_qid) & legacy_set:
        raise ValueError("deterministic and fresh repair selections overlap or are incomplete")
    fresh_generated = {query_id: fresh_by_all[query_id] for query_id, status in fresh_status.items() if status == "generated"}
    final_rows, provenance = merge_rows(
        canonical_rows,
        existing_by_qid=existing_by_qid,
        existing_status_by_qid=existing_status,
        deterministic_by_qid=deterministic_by_qid,
        fresh_by_qid=fresh_generated,
    )
    validate_ranked_submission(final_rows, expected_query_ids=expected_ids)
    status_counts = dict(sorted(Counter(item["status"] for item in provenance).items()))
    historical_cost = float(prep_manifest.get("a3_accounted_cost_usd", 0.0))
    fresh_cost = float(fresh_summary.get("accounted_cost_usd", 0.0))
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "strategy_id": "grouped_rewrite_fallback_repair_final",
        "query_count": len(final_rows),
        "preparation_root": str(preparation_root),
        "preparation_inventory_sha256": prepare._sha_bytes(prepare._canonical(salvage.root_inventory(preparation_root))),
        "fresh_root": str(fresh_root),
        "fresh_inventory_sha256": prepare._sha_bytes(prepare._canonical(salvage.root_inventory(fresh_root))),
        "existing_root": str(existing_root),
        "existing_inventory_sha256": prepare._sha_bytes(prepare._canonical(salvage.root_inventory(existing_root))),
        "canonical_source_artifacts": frozen.source_artifacts(**paths),
        "canonical_source_artifacts_sha256": frozen._sha_bytes(frozen._canonical(frozen.source_artifacts(**paths))),
        "status_counts": status_counts,
        "historical_accounted_cost_usd": historical_cost,
        "fresh_accounted_cost_usd": fresh_cost,
        "cumulative_accounted_cost_usd": round(historical_cost + fresh_cost, 8),
        "existing_final_sha256": existing_summary["final_sha256"],
        "fresh_final_sha256": fresh_summary["final_sha256"],
        "preparation_summary_sha256": frozen._sha_file(prep_summary_path),
    }
    output_root.mkdir(parents=True)
    manifest_hash = prepare._write(output_root / "manifest.json", manifest)
    final_hash = prepare._write(output_root / "final-organizer.json", final_rows)
    provenance_hash = prepare._write(output_root / "final-provenance.json", provenance)
    summary = {
        "status": "complete",
        "query_count": len(final_rows),
        "status_counts": status_counts,
        "fallback_query_count": status_counts.get("fallback_fusion", 0),
        "historical_accounted_cost_usd": historical_cost,
        "fresh_accounted_cost_usd": fresh_cost,
        "cumulative_accounted_cost_usd": round(historical_cost + fresh_cost, 8),
        "manifest_sha256": manifest_hash,
        "final_sha256": final_hash,
        "provenance_sha256": provenance_hash,
    }
    prepare._write(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(prog="finalize_vqa_grouped_rewrite_fallback_repair")
    parser.add_argument("--preparation-root", type=Path, default=DEFAULT_PREPARATION_ROOT)
    parser.add_argument("--fresh-root", type=Path, default=DEFAULT_FRESH_ROOT)
    parser.add_argument("--existing-root", type=Path, default=DEFAULT_EXISTING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = finalize_repair(
        preparation_root=args.preparation_root,
        fresh_root=args.fresh_root,
        existing_root=args.existing_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
