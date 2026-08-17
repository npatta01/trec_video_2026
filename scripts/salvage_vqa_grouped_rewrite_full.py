#!/usr/bin/env python3
"""Offline salvage of independently valid answers from a frozen grouped run.

This module never constructs a provider client.  It reads a terminal A3 root,
validates each candidate query against the frozen source row, and writes a new
immutable salvage root with fusion fallback for anything missing or invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_vqa_grouped_rewrite_full as frozen

DEFAULT_SOURCE_ROOT = frozen.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = frozen._ROOT / "data/vqa/benchmarks/grouped-rewrite-full-20260817-grouped_unique_generator_only-a3-salvage-final"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read salvage artifact: {path}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable salvage artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return _sha_bytes(data)


def root_inventory(root: Path) -> list[dict[str, str]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")
    return [
        {"path": str(path.relative_to(root)), "sha256": _sha_file(path)}
        for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)))
        if path.is_file() or path.is_symlink()
    ]


def _query_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": int(row["query_id"]),
        "question": str(row.get("question", "")),
        "rank1_source": row.get("rank1_source"),
        "rank1_answer": row.get("rank1_answer"),
        "source_candidates": [
            {"source_id": item["source_id"], "original_rank": 1, "answer": item["answer"]}
            for item in row.get("source_candidates", [])
        ],
    }


def _request_binding(query: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query.get("query_id"),
        "question": query.get("question", ""),
        "rank1_source": query.get("rank1_source"),
        "rank1_answer": query.get("rank1_answer"),
        "source_candidates": query.get("source_candidates", []),
    }


def _request_matches(request: Mapping[str, Any], batch: Sequence[Mapping[str, Any]]) -> bool:
    queries = request.get("queries")
    if not isinstance(queries, list) or len(queries) != len(batch):
        return False
    return all(_request_binding(query) == _query_binding(row) for query, row in zip(queries, batch, strict=True))


def _raw_text(path: Path) -> str:
    payload = _json(path)
    provider = payload.get("provider_response", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(provider, Mapping):
        return ""
    if isinstance(provider.get("text"), str):
        return provider["text"]
    candidates = provider.get("candidates")
    if not isinstance(candidates, list):
        return ""
    parts = []
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, Mapping) else None
        candidate_parts = content.get("parts") if isinstance(content, Mapping) else None
        if not isinstance(candidate_parts, list):
            continue
        for part in candidate_parts:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


def _parse_raw_queries(path: Path) -> list[Mapping[str, Any]]:
    text = _raw_text(path)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    queries = payload.get("queries") if isinstance(payload, Mapping) else None
    return [query for query in queries if isinstance(query, Mapping)] if isinstance(queries, list) else []


def _attempt_path(batch_root: Path) -> Path | None:
    attempts = sorted(batch_root.glob("attempt-*/"))
    return attempts[0] if attempts else None


def _validate_query(query: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        result = frozen.validate_generator({"queries": [dict(query)]}, [row])
    except (TypeError, ValueError, KeyError):
        return None
    return result["queries"][0]


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


def salvage_run(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_paths: Mapping[str, Path] | None = None,
    expected_query_count: int | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"salvage output root must be absent: {output_root}")
    output_root = output_root.resolve()
    source_manifest_path = source_root / "manifest.json"
    source_manifest = _json(source_manifest_path)
    if not isinstance(source_manifest, Mapping):
        raise ValueError("source manifest must be an object")
    required_manifest_keys = {
        "batch_size",
        "batch_count",
        "query_count",
        "source_artifacts",
        "source_artifacts_sha256",
        "request_bundle_sha256",
        "generator_response_schema_sha256",
    }
    if not required_manifest_keys.issubset(source_manifest):
        raise ValueError("source manifest is incomplete")
    source_manifest_sha256 = _sha_file(source_manifest_path)
    source_summary_path = source_root / "summary.json"
    source_final_path = source_root / "final-organizer.json"
    source_provenance_path = source_root / "final-provenance.json"
    if not source_summary_path.exists() or not source_final_path.exists() or not source_provenance_path.exists():
        raise ValueError("complete source summary/final/provenance are required")
    source_summary = _json(source_summary_path)
    if not isinstance(source_summary, Mapping):
        raise ValueError("source summary must be an object")
    if source_summary.get("status") != "complete":
        raise ValueError("source summary is not complete")
    if source_summary.get("final_sha256") != _sha_file(source_final_path):
        raise ValueError("source summary final hash mismatch")
    if source_summary.get("provenance_sha256") != _sha_file(source_provenance_path):
        raise ValueError("source summary provenance hash mismatch")
    paths = _source_paths(source_paths)
    artifacts = frozen.source_artifacts(**paths)
    if source_manifest.get("source_artifacts") != artifacts or source_manifest.get("source_artifacts_sha256") != _sha_bytes(_canonical(artifacts)) or source_summary.get("source_artifacts_sha256") != _sha_bytes(_canonical(artifacts)):
        raise ValueError("source artifact hash drift")
    batch_size = int(source_manifest["batch_size"])
    query_count = int(source_manifest["query_count"])
    if expected_query_count is not None and expected_query_count != query_count:
        raise ValueError("source query count mismatch")
    rows = frozen.load_rows(**paths, expected_query_count=query_count)
    batches = frozen.batches(rows, batch_size=batch_size)
    if len(batches) != int(source_manifest["batch_count"]):
        raise ValueError("source batch count mismatch")
    schema_hash = _sha_bytes(_canonical(frozen._generator_response_schema(batch_size)))
    if source_manifest.get("generator_response_schema_sha256") != schema_hash:
        raise ValueError("source response schema hash drift")
    requests = [frozen.build_generator_request(batch, index + 1) for index, batch in enumerate(batches)]
    if source_manifest.get("request_bundle_sha256") != _sha_bytes(_canonical(requests)):
        raise ValueError("source request bundle hash drift")
    source_final_rows = _json(source_final_path)
    from trec_video_2026.vqa_deadline_canary import validate_ranked_submission

    validate_ranked_submission(source_final_rows, expected_query_ids={int(row["query_id"]) for row in rows})
    source_provenance = _json(source_provenance_path)
    if source_provenance != frozen._build_provenance(rows, source_final_rows):
        raise ValueError("source provenance binding drift")
    source_inventory = root_inventory(source_root)
    source_inventory_sha256 = _sha_bytes(_canonical(source_inventory))
    final_by_qid = {
        int(row["Q_ID"]): row
        for row in source_final_rows
        if isinstance(row, Mapping) and isinstance(row.get("Q_ID"), int)
    }
    generated: dict[int, tuple[Mapping[str, Any], float, str, int]] = {}
    salvageable = 0
    invalid = 0
    accepted_batches = 0
    for index, batch in enumerate(batches):
        batch_id = index + 1
        batch_root = source_root / f"batch-{batch_id:04d}"
        marker_path = next((batch_root / name for name in ("accepted.json", "failed.json") if (batch_root / name).exists()), None)
        if marker_path is None:
            invalid += len(batch)
            continue
        marker = _json(marker_path)
        attempt = _attempt_path(batch_root)
        request = _json(attempt / "request.json") if attempt is not None and (attempt / "request.json").exists() else None
        expected_request = dict(requests[index])
        expected_request["source_artifacts"] = artifacts
        expected_request["source_artifacts_sha256"] = _sha_bytes(_canonical(artifacts))
        if not isinstance(request, Mapping) or dict(request) != expected_request or not _request_matches(request, batch):
            invalid += len(batch)
            continue
        candidate_queries: list[Mapping[str, Any]] = []
        is_accepted = marker_path.name == "accepted.json" and marker.get("status") == "accepted"
        if is_accepted and isinstance(marker.get("queries"), list):
            parsed_path = attempt / "parsed-response.json" if attempt is not None else None
            parsed = _json(parsed_path) if parsed_path is not None and parsed_path.exists() else None
            if (
                not isinstance(parsed, Mapping)
                or parsed.get("queries") != marker["queries"]
                or len(marker["queries"]) != len(batch)
                or not all(isinstance(query, Mapping) for query in marker["queries"])
            ):
                invalid += len(batch)
                continue
            candidate_queries = [query for query in marker["queries"] if isinstance(query, Mapping)]
        elif marker_path.name == "failed.json" and marker.get("status") == "failed" and attempt is not None:
            raw_path = attempt / "raw-response.json"
            if raw_path.exists():
                candidate_queries = _parse_raw_queries(raw_path)
        by_qid = {int(row["query_id"]): row for row in batch}
        accepted_here = set()
        batch_generated: dict[int, tuple[Mapping[str, Any], float, str, int]] = {}
        accepted_invalid = False
        elapsed = float(marker.get("elapsed_seconds", 0.0))
        for query in candidate_queries:
            query_id = query.get("query_id")
            if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id not in by_qid or query_id in accepted_here:
                accepted_invalid = accepted_invalid or is_accepted
                continue
            normalized = _validate_query(query, by_qid[query_id])
            if normalized is None:
                accepted_invalid = accepted_invalid or is_accepted
                continue
            if is_accepted:
                source_row = final_by_qid.get(query_id)
                source_answers = source_row.get("Answers") if isinstance(source_row, Mapping) else None
                if (
                    not isinstance(source_row, Mapping)
                    or source_row.get("Q_ID") != query_id
                    or source_row.get("Video_ID") != by_qid[query_id].get("video_id")
                    or not isinstance(source_answers, list)
                    or [
                        {"Rank": answer.get("Rank"), "Answer": answer.get("Answer")}
                        for answer in source_answers
                    ]
                    != [
                        {"Rank": answer["rank"], "Answer": answer["answer"]}
                        for answer in normalized["answers"]
                    ]
                ):
                    accepted_invalid = True
                    continue
            batch_generated[query_id] = (normalized, elapsed, "accepted" if is_accepted else "raw_salvaged", batch_id)
            accepted_here.add(query_id)
        if accepted_invalid:
            invalid += len(batch)
            continue
        generated.update(batch_generated)
        if is_accepted:
            accepted_batches += 1
        salvageable += len(accepted_here)
        invalid += len(batch) - len(accepted_here)
    final_rows = []
    provenance = []
    for row in rows:
        query_id = int(row["query_id"])
        fallback = dict(row["fallback"])
        item = generated.get(query_id)
        if item is None:
            final_row = fallback
            status = "fallback_fusion"
            elapsed = 0.0
            batch_id = None
        else:
            query, elapsed, status, batch_id = item
            final_row = dict(fallback)
            final_row["Answers"] = [{"Rank": answer["rank"], "Answer": answer["answer"], "Time": round(elapsed, 6)} for answer in query["answers"]]
        final_rows.append(final_row)
        provenance.append({"Q_ID": query_id, "Video_ID": row["video_id"], "status": status, "batch_id": batch_id, "answer_sha256": [_sha_bytes(str(answer["Answer"]).encode()) for answer in final_row["Answers"]]})
    validate_ranked_submission(final_rows, expected_query_ids={int(row["query_id"]) for row in rows})
    if source_inventory_sha256 != _sha_bytes(_canonical(root_inventory(source_root))):
        raise ValueError("source root inventory drift")
    output_root = Path(output_root)
    manifest = {
        "status": "inputs_frozen",
        "strategy_id": "offline_salvage",
        "source_root": str(source_root),
        "source_manifest_sha256": source_manifest_sha256,
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_inventory_sha256,
        "source_artifacts": artifacts,
        "source_artifacts_sha256": _sha_bytes(_canonical(artifacts)),
        "query_count": query_count,
        "batch_count": len(batches),
        "salvaged_query_count": salvageable,
        "fallback_query_count": invalid,
    }
    manifest_hash = _write(output_root / "manifest.json", manifest)
    final_hash = _write(output_root / "final-organizer.json", final_rows)
    provenance_hash = _write(output_root / "final-provenance.json", provenance)
    summary = {
        "status": "complete",
        "query_count": query_count,
        "batch_count": len(batches),
        "accepted_batch_count": accepted_batches,
        "salvaged_query_count": salvageable,
        "fallback_query_count": invalid,
        "final_sha256": final_hash,
        "provenance_sha256": provenance_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "manifest_sha256": manifest_hash,
        "source_inventory_sha256": source_inventory_sha256,
        "source_artifacts_sha256": _sha_bytes(_canonical(artifacts)),
    }
    _write(output_root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="salvage_vqa_grouped_rewrite_full")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(salvage_run(source_root=args.source_root, output_root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
