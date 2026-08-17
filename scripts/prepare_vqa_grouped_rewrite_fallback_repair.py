#!/usr/bin/env python3
"""Prepare immutable offline and fresh-call inputs for grouped fallback repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_vqa_grouped_rewrite_full as frozen
import salvage_vqa_grouped_rewrite_full as salvage

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A3_ROOT = frozen.DEFAULT_OUTPUT_ROOT
DEFAULT_SALVAGE_ROOT = (
    _ROOT
    / "data/vqa/benchmarks/grouped-rewrite-full-20260817-grouped_unique_generator_only-a3-salvage-final"
)
DEFAULT_OUTPUT_ROOT = _ROOT / "data/vqa/benchmarks/grouped-rewrite-fallback-repair-20260817/preparation"


def _normalized_answer(value: str) -> str:
    return " ".join(value.split()).casefold()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable repair artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return _sha_bytes(data)


def _validated_groups(query: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, dict[str, Any]] | None:
    groups = query.get("groups")
    if not isinstance(groups, list):
        return None
    candidates = {item["source_id"]: item for item in row["source_candidates"]}
    group_by_id: dict[str, dict[str, Any]] = {}
    covered: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"group_id", "source_ids"}:
            return None
        group_id, source_ids = group.get("group_id"), group.get("source_ids")
        if (
            not isinstance(group_id, str)
            or group_id in group_by_id
            or not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(source_id, str) for source_id in source_ids)
            or len(source_ids) != len(set(source_ids))
            or any(source_id not in candidates or source_id in covered for source_id in source_ids)
        ):
            return None
        covered.extend(source_ids)
        group_by_id[group_id] = {"group_id": group_id, "source_ids": list(source_ids)}
    if len(covered) != len(candidates) or set(covered) != set(candidates):
        return None
    pair = frozen.MUST_SEPARATE_SOURCE_PAIRS.get(int(row["query_id"]))
    if pair and pair[0] in candidates and pair[1] in candidates:
        owner = {source_id: group_id for group_id, group in group_by_id.items() for source_id in group["source_ids"]}
        if owner[pair[0]] == owner[pair[1]]:
            return None
    return group_by_id


def repair_current_query(query: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Repair a current-schema query using only frozen generated/source/fusion text."""
    if query.get("query_id") != row.get("query_id"):
        return None
    group_by_id = _validated_groups(query, row)
    answers = query.get("answers")
    if group_by_id is None or not isinstance(answers, list) or len(answers) != frozen.ANSWERS_PER_QUERY:
        return None
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed_duplicate = False
    provider_ranks = [answer.get("rank") if isinstance(answer, Mapping) else None for answer in answers]
    repaired_rank_sequence = provider_ranks != list(range(1, frozen.ANSWERS_PER_QUERY + 1))
    for answer in answers:
        if (
            not isinstance(answer, Mapping)
            or set(answer) != {"rank", "answer", "source_group"}
            or isinstance(answer.get("rank"), bool)
            or not isinstance(answer.get("rank"), int)
            or not isinstance(answer.get("answer"), str)
            or not answer["answer"].strip()
            or answer.get("source_group") not in group_by_id
        ):
            return None
        normalized = _normalized_answer(answer["answer"])
        if normalized in seen:
            removed_duplicate = True
            continue
        seen.add(normalized)
        ordered.append(
            {
                "rank": len(ordered) + 1,
                "answer": answer["answer"],
                "source_group": answer["source_group"],
                "origin": "generated",
            }
        )
    if not ordered or ordered[0]["answer"] != row["rank1_answer"]:
        return None
    rank1_group = next(
        (group_id for group_id, group in group_by_id.items() if row["rank1_source"] in group["source_ids"]),
        None,
    )
    if rank1_group is None or ordered[0]["source_group"] != rank1_group:
        return None
    candidates = {item["source_id"]: item["answer"] for item in row["source_candidates"]}
    inserted_exact_hedge = False
    for group_id, group in group_by_id.items():
        exact = {candidates[source_id] for source_id in group["source_ids"]}
        if not any(answer["source_group"] == group_id and answer["answer"] in exact for answer in ordered):
            replacement = next(
                (candidates[source_id] for source_id in group["source_ids"] if _normalized_answer(candidates[source_id]) not in seen),
                None,
            )
            removable_index = next(
                (
                    index
                    for index in range(len(ordered) - 1, 0, -1)
                    if ordered[index]["answer"]
                    not in {
                        candidates[source_id]
                        for source in group_by_id.values()
                        for source_id in source["source_ids"]
                    }
                ),
                None,
            )
            if replacement is None or removable_index is None:
                return None
            removed = ordered.pop(removable_index)
            seen.remove(_normalized_answer(removed["answer"]))
            seen.add(_normalized_answer(replacement))
            ordered.append(
                {
                    "rank": len(ordered) + 1,
                    "answer": replacement,
                    "source_group": group_id,
                    "origin": "source_hedge",
                }
            )
            inserted_exact_hedge = True
    if set(answer["source_group"] for answer in ordered) != set(group_by_id):
        return None
    for fallback in row["fallback"]["Answers"]:
        if len(ordered) == frozen.ANSWERS_PER_QUERY:
            break
        text = fallback.get("Answer")
        if not isinstance(text, str) or not text.strip() or _normalized_answer(text) in seen:
            continue
        seen.add(_normalized_answer(text))
        ordered.append(
            {
                "rank": len(ordered) + 1,
                "answer": text,
                "source_group": "fusion_fallback",
                "origin": "fusion_fill",
            }
        )
    if len(ordered) != frozen.ANSWERS_PER_QUERY:
        return None
    for index, answer in enumerate(ordered, 1):
        answer["rank"] = index
    if inserted_exact_hedge:
        repair_reason = "exact_source_hedge_inserted"
    elif removed_duplicate:
        repair_reason = "deduplicated_with_fusion_fill"
    elif repaired_rank_sequence:
        repair_reason = "rank_sequence_repaired"
    else:
        repair_reason = "group_contract_repaired"
    return {
        "query_id": int(row["query_id"]),
        "groups": [dict(group) for group in group_by_id.values()],
        "answers": ordered,
        "repair_reason": repair_reason,
    }


def _validated_terminal_root(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = Path(root).resolve()
    summary_path = root / "summary.json"
    final_path = root / "final-organizer.json"
    provenance_path = root / "final-provenance.json"
    if not summary_path.is_file() or not final_path.is_file() or not provenance_path.is_file():
        raise ValueError(f"terminal root is incomplete: {root}")
    summary = frozen._json(summary_path)
    if (
        not isinstance(summary, Mapping)
        or summary.get("status") != "complete"
        or summary.get("final_sha256") != _sha_file(final_path)
        or summary.get("provenance_sha256") != _sha_file(provenance_path)
    ):
        raise ValueError(f"terminal root hash binding is invalid: {root}")
    return dict(summary), salvage.root_inventory(root)


def _source_paths(source_paths: Mapping[str, Path] | None) -> dict[str, Path]:
    if source_paths is not None:
        return {name: Path(path) for name, path in source_paths.items()}
    return {
        "questions": frozen.DEFAULT_QUESTIONS,
        "gemini": frozen.DEFAULT_GEMINI,
        "qwen": frozen.DEFAULT_QWEN,
        "control": frozen.DEFAULT_CONTROL,
        "fusion": frozen.DEFAULT_FUSION,
        "provenance": frozen.DEFAULT_PROVENANCE,
    }


def _query_ids(rows: list[dict[str, Any]], path: Path) -> set[int]:
    return {frozen._qid(row, path) for row in rows}


def prepare_repair(
    *,
    a3_root: Path = DEFAULT_A3_ROOT,
    salvage_root: Path = DEFAULT_SALVAGE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_paths: Mapping[str, Path] | None = None,
    run: bool = False,
) -> dict[str, Any]:
    """Reconcile all fallbacks and optionally freeze repair inputs."""
    output_root = Path(output_root)
    if run and (output_root.exists() or output_root.is_symlink()):
        raise FileExistsError(f"repair preparation root must be absent: {output_root}")
    a3_root = Path(a3_root).resolve()
    salvage_root = Path(salvage_root).resolve()
    a3_summary, a3_inventory = _validated_terminal_root(a3_root)
    salvage_summary, salvage_inventory = _validated_terminal_root(salvage_root)
    paths = _source_paths(source_paths)
    rows = frozen.load_rows(**paths, expected_query_count=frozen.EXPECTED_QUERY_COUNT)
    rows_by_qid = {int(row["query_id"]): row for row in rows}
    salvage_provenance = frozen._json(salvage_root / "final-provenance.json")
    if not isinstance(salvage_provenance, list):
        raise ValueError("salvage provenance must be an array")
    fallback_ids = {
        int(item["Q_ID"])
        for item in salvage_provenance
        if isinstance(item, Mapping) and item.get("status") == "fallback_fusion"
    }
    legacy_ids: set[int] = set()
    for marker_path in sorted(a3_root.glob("batch-*/exposed-fallback.json")):
        marker = frozen._json(marker_path)
        query_ids = marker.get("source_query_ids") if isinstance(marker, Mapping) else None
        if not isinstance(query_ids, list) or any(isinstance(qid, bool) or not isinstance(qid, int) for qid in query_ids):
            raise ValueError(f"invalid exposed fallback marker: {marker_path}")
        legacy_ids.update(query_ids)
    current_ids = fallback_ids - legacy_ids
    if len(fallback_ids) != 610 or len(legacy_ids) != 290 or len(current_ids) != 320:
        raise ValueError("fallback partition no longer matches the frozen 610 = 290 + 320 census")
    raw_by_qid: dict[int, Mapping[str, Any]] = {}
    for failed_path in sorted(a3_root.glob("batch-*/failed.json")):
        attempt = salvage._attempt_path(failed_path.parent)
        if attempt is None or not (attempt / "raw-response.json").is_file():
            continue
        for query in salvage._parse_raw_queries(attempt / "raw-response.json"):
            query_id = query.get("query_id")
            if isinstance(query_id, int) and not isinstance(query_id, bool):
                raw_by_qid[query_id] = query
    deterministic: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for query_id in sorted(current_ids):
        query = raw_by_qid.get(query_id)
        repaired = repair_current_query(query, rows_by_qid[query_id]) if query is not None else None
        if repaired is None:
            raise ValueError(f"current-schema fallback is not deterministically repairable: {query_id}")
        reasons[repaired["repair_reason"]] += 1
        deterministic.append(repaired)
    filtered_sources: dict[str, list[dict[str, Any]]] = {}
    filtered_counts: dict[str, int] = {}
    source_artifacts = frozen.source_artifacts(**paths)
    for name, path in paths.items():
        source_rows = frozen._rows(frozen._json(path), path)
        filtered = [row for row in source_rows if frozen._qid(row, path) in legacy_ids]
        if _query_ids(filtered, path) != legacy_ids:
            raise ValueError(f"filtered source does not cover the 290 legacy queries: {name}")
        filtered_sources[name] = filtered
        filtered_counts[name] = len(_query_ids(filtered, path))
    a3_inventory_sha256 = _sha_bytes(_canonical(a3_inventory))
    salvage_inventory_sha256 = _sha_bytes(_canonical(salvage_inventory))
    deterministic_sha256 = _sha_bytes(_canonical(deterministic))
    legacy_ids_payload = sorted(legacy_ids)
    legacy_ids_sha256 = _sha_bytes(_canonical(legacy_ids_payload))
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "strategy_id": "grouped_rewrite_fallback_repair_preparation",
        "query_count": len(rows),
        "existing_grouped_query_count": len(rows) - len(fallback_ids),
        "deterministic_repair_query_count": len(deterministic),
        "legacy_fresh_query_count": len(legacy_ids),
        "repair_reason_counts": dict(sorted(reasons.items())),
        "a3_root": str(a3_root),
        "a3_summary_sha256": _sha_file(a3_root / "summary.json"),
        "a3_inventory_sha256": a3_inventory_sha256,
        "salvage_root": str(salvage_root),
        "salvage_summary_sha256": _sha_file(salvage_root / "summary.json"),
        "salvage_inventory_sha256": salvage_inventory_sha256,
        "source_artifacts": source_artifacts,
        "source_artifacts_sha256": _sha_bytes(_canonical(source_artifacts)),
        "deterministic_repairs_sha256": deterministic_sha256,
        "legacy_query_ids_sha256": legacy_ids_sha256,
        "filtered_source_query_counts": dict(sorted(filtered_counts.items())),
        "filtered_source_sha256": {
            name: _sha_bytes(_canonical(source_rows)) for name, source_rows in sorted(filtered_sources.items())
        },
        "a3_accounted_cost_usd": float(a3_summary.get("accounted_cost_usd", 0.0)),
        "salvage_final_sha256": salvage_summary["final_sha256"],
    }
    result = {
        "status": "dry_run" if not run else "inputs_frozen",
        "existing_grouped_query_count": manifest["existing_grouped_query_count"],
        "deterministic_repair_query_count": len(deterministic),
        "legacy_fresh_query_count": len(legacy_ids),
        "repair_reason_counts": manifest["repair_reason_counts"],
        "filtered_source_query_counts": manifest["filtered_source_query_counts"],
        "manifest_sha256": _sha_bytes(_canonical(manifest)),
        "output_root_created": run,
    }
    if not run:
        return result
    _write(output_root / "manifest.json", manifest)
    _write(output_root / "deterministic-repairs.json", deterministic)
    _write(output_root / "legacy-query-ids.json", legacy_ids_payload)
    input_hashes: dict[str, str] = {}
    for name, source_rows in sorted(filtered_sources.items()):
        input_hashes[name] = _write(output_root / "inputs" / f"{name}.json", source_rows)
    empty_accounting = {
        "declared_terminal_cost_usd": 0.0,
        "actual_unterminal_raw_cost_usd": 0.0,
        "cumulative_prior_cost_usd": 0.0,
    }
    empty_prior_hash = _write(
        output_root / "empty-prior" / "manifest.json",
        {"status": "empty_prior", "prior_accounting": empty_accounting},
    )
    summary = dict(result)
    summary.update(
        status="complete",
        manifest_file_sha256=_sha_file(output_root / "manifest.json"),
        deterministic_repairs_file_sha256=_sha_file(output_root / "deterministic-repairs.json"),
        legacy_query_ids_file_sha256=_sha_file(output_root / "legacy-query-ids.json"),
        filtered_input_file_sha256=input_hashes,
        empty_prior_manifest_sha256=empty_prior_hash,
    )
    _write(output_root / "summary.json", summary)
    result["summary_sha256"] = _sha_file(output_root / "summary.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare_vqa_grouped_rewrite_fallback_repair")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare_repair(output_root=args.output_root, run=args.run), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
