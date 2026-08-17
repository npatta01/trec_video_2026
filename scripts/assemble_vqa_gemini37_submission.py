"""Assemble a deterministic Gemini-over-baseline TRECVID VQA submission.

This is an offline submission transformation.  It consumes only frozen model
predictions, an already validated ranked baseline, and an expected query set;
it never opens annotations and never calls a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trec_video_2026.vqa_deadline_canary import validate_ranked_submission


EXPECTED_QUERY_COUNT = 2_443
DEFAULT_BASELINE = Path(
    "data/vqa/submissions/qwen35-safety-20260810-a1/final/submission-top10.json"
)
_PAYLOAD_KEYS = ("rows", "predictions", "records", "submission")
_PASSED_STATUSES = frozenset({"passed", "succeeded", "success", "complete"})
_FALLBACK_STATUSES = frozenset(
    {
        "cost_cap_blocked",
        "failed",
        "upload_failed",
        "unavailable",
        "unsupported",
        "not_available",
        "error",
        "skipped",
        "not_ready",
    }
)
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {"correct_answer", "reference", "ground_truth", "annotation", "annotations"}
)
_GEMINI_MODEL = "gemini-3.7-flash"


class AssemblyError(ValueError):
    """Raised when frozen inputs cannot produce a valid official-shaped file."""


@dataclass(frozen=True)
class AssemblyResult:
    rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    output_path: Path | None = None
    provenance_path: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    body = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AssemblyError(f"cannot read input: {path}") from error
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssemblyError(f"invalid JSONL at {path}:{line_number}") from error
        return rows
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise AssemblyError(f"invalid JSON input: {path}") from error


def _rows_from_payload(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        artifact_hash = payload.get("_artifact_sha256")
        if artifact_hash is not None:
            body = dict(payload)
            body.pop("_artifact_sha256", None)
            if artifact_hash != _canonical_sha256(body):
                raise AssemblyError(f"artifact hash mismatch: {path}")
        rows = None
        for key in _PAYLOAD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            raise AssemblyError(f"input has no row array: {path}")
        row_count = payload.get("row_count")
        if row_count is not None and row_count != len(rows):
            raise AssemblyError(f"input row_count does not match rows: {path}")
    else:
        raise AssemblyError(f"input must be a JSON array or object: {path}")
    if any(not isinstance(row, Mapping) for row in rows):
        raise AssemblyError(f"input rows must be JSON objects: {path}")
    return [dict(row) for row in rows]


def _query_id(row: Mapping[str, Any], *, path: Path) -> int:
    value = row.get("Q_ID", row.get("query_id", row.get("q_id")))
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssemblyError(f"invalid query ID in {path}")
    return value


def _video_id(row: Mapping[str, Any], *, path: Path) -> str:
    value = row.get("Video_ID", row.get("video_id"))
    if not isinstance(value, str) or not value.strip():
        raise AssemblyError(f"invalid video ID in {path}")
    return value


def _expected_queries(path: Path) -> tuple[tuple[int, ...], dict[int, str]]:
    payload = _read_payload(path)
    if isinstance(payload, Mapping) and not any(
        isinstance(payload.get(key), list)
        for key in ("query_ids", "queries", "rows", "topics")
    ):
        # Also accept {"1": ..., "2": ...} as a compact query-ID map.
        try:
            ids = [int(key) for key in payload]
        except (TypeError, ValueError) as error:
            raise AssemblyError(f"expected query set has no recognized rows: {path}") from error
        videos = {
            query_id: value
            for query_id, value in ((int(key), value) for key, value in payload.items())
            if isinstance(value, str) and value.strip()
        }
    else:
        if isinstance(payload, Mapping):
            for key in ("query_ids", "queries", "rows", "topics"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    payload = candidate
                    break
        if not isinstance(payload, list):
            raise AssemblyError(f"expected query set must be an array: {path}")
        ids = []
        videos = {}
        for item in payload:
            if isinstance(item, Mapping):
                query_id = _query_id(item, path=path)
                ids.append(query_id)
                value = item.get("Video_ID", item.get("video_id"))
                if isinstance(value, str) and value.strip():
                    videos[query_id] = value
            elif isinstance(item, int) and not isinstance(item, bool):
                ids.append(item)
            else:
                raise AssemblyError(f"invalid expected query row in {path}")
    if len(ids) != len(set(ids)):
        raise AssemblyError(f"expected query set contains duplicate IDs: {path}")
    if not ids:
        raise AssemblyError(f"expected query set is empty: {path}")
    return tuple(sorted(ids)), videos


def _normalize_answer(text: str) -> str:
    return " ".join(text.casefold().split())


def _baseline_rows(
    path: Path, expected_ids: set[int], expected_videos: Mapping[int, str]
) -> dict[int, dict[str, Any]]:
    rows = _rows_from_payload(_read_payload(path), path=path)
    try:
        validate_ranked_submission(rows, expected_query_ids=expected_ids)
    except (TypeError, ValueError) as error:
        raise AssemblyError(f"baseline is not a valid ten-answer submission: {error}") from error
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        query_id = _query_id(row, path=path)
        video_id = _video_id(row, path=path)
        if expected_videos.get(query_id) not in (None, video_id):
            raise AssemblyError(f"baseline video ID disagrees with expected query set: {query_id}")
        if query_id in result:
            raise AssemblyError(f"duplicate baseline query ID: {query_id}")
        result[query_id] = dict(row)
    if set(result) != expected_ids:
        raise AssemblyError("baseline query IDs do not match expected query set")
    return result


def _gemini_rows(
    path: Path, expected_ids: set[int], expected_videos: Mapping[int, str]
) -> dict[int, dict[str, Any]]:
    payload = _read_payload(path)
    if not isinstance(payload, Mapping):
        raise AssemblyError("Gemini freeze must be a terminal envelope object")
    if payload.get("status") != "predictions_frozen":
        raise AssemblyError("Gemini freeze status must be predictions_frozen")
    if payload.get("model") != _GEMINI_MODEL:
        raise AssemblyError(f"Gemini freeze model must be {_GEMINI_MODEL}")
    rows = _rows_from_payload(payload, path=path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise AssemblyError(f"Gemini freeze contains a label field: {path}")
        query_id = _query_id(row, path=path)
        video_id = _video_id(row, path=path)
        if query_id in result:
            raise AssemblyError(f"duplicate Gemini query ID: {query_id}")
        if query_id not in expected_ids:
            raise AssemblyError(f"Gemini query ID is outside expected set: {query_id}")
        if expected_videos.get(query_id) not in (None, video_id):
            raise AssemblyError(f"Gemini video ID disagrees with expected query set: {query_id}")
        status = row.get("status")
        if not isinstance(status, str):
            raise AssemblyError(f"Gemini status is missing for query {query_id}")
        status = status.casefold()
        answer = row.get("answer")
        if status in _PASSED_STATUSES:
            if not isinstance(answer, str) or not answer.strip():
                raise AssemblyError(f"passed Gemini row has no answer: {query_id}")
        elif status not in _FALLBACK_STATUSES:
            raise AssemblyError(f"unsupported Gemini status {status!r}: {query_id}")
        result[query_id] = {
            "query_id": query_id,
            "video_id": video_id,
            "status": status,
            "answer": answer.strip() if isinstance(answer, str) else "",
            "time": row.get(
                "Time",
                row.get("time", row.get("elapsed", row.get("elapsed_seconds", 0.0))),
            ),
        }
    if set(result) != expected_ids:
        raise AssemblyError("Gemini query IDs do not match expected query set")
    return result


def _gemini_time(row: Mapping[str, Any]) -> float:
    value = row.get("time", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"Gemini answer time is not numeric: {row['query_id']}")
    if not math.isfinite(float(value)) or value < 0:
        raise AssemblyError(f"Gemini answer time is invalid: {row['query_id']}")
    return float(value)


def _assemble_row(
    baseline: Mapping[str, Any], gemini: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_id = int(baseline["Q_ID"])
    if gemini["status"] not in _PASSED_STATUSES:
        return dict(baseline), {
            "Q_ID": query_id,
            "Video_ID": baseline["Video_ID"],
            "gemini_status": gemini["status"],
            "answer_source": "baseline_fallback",
            "baseline_ranks_used": list(range(1, 11)),
        }
    answer = str(gemini["answer"])
    seen = {_normalize_answer(answer)}
    selected = [{"Rank": 1, "Answer": answer, "Time": _gemini_time(gemini)}]
    baseline_ranks: list[int] = []
    for candidate in sorted(baseline["Answers"], key=lambda item: item["Rank"]):
        normalized = _normalize_answer(str(candidate["Answer"]))
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            {
                "Rank": len(selected) + 1,
                "Answer": candidate["Answer"],
                "Time": candidate["Time"],
            }
        )
        baseline_ranks.append(int(candidate["Rank"]))
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise AssemblyError(f"not enough distinct baseline answers for query {query_id}")
    result = dict(baseline)
    result["Answers"] = selected
    return result, {
        "Q_ID": query_id,
        "Video_ID": baseline["Video_ID"],
        "gemini_status": gemini["status"],
        "answer_source": "gemini_rank_1_plus_baseline",
        "baseline_ranks_used": baseline_ranks,
        "gemini_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    }


def _write_new_json(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise AssemblyError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def assemble_submission(
    *,
    gemini_freeze: Path,
    baseline_submission: Path = DEFAULT_BASELINE,
    expected_queries: Path,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    output_path: Path | None = None,
    provenance_path: Path | None = None,
) -> AssemblyResult:
    """Build and validate the deterministic Gemini-over-baseline submission."""

    expected_ids_order, expected_videos = _expected_queries(expected_queries)
    if len(expected_ids_order) != expected_query_count:
        raise AssemblyError(
            f"expected query set must contain exactly {expected_query_count} IDs"
        )
    expected_ids = set(expected_ids_order)
    baseline = _baseline_rows(baseline_submission, expected_ids, expected_videos)
    gemini = _gemini_rows(gemini_freeze, expected_ids, expected_videos)
    output_rows: list[dict[str, Any]] = []
    row_provenance: list[dict[str, Any]] = []
    for query_id in expected_ids_order:
        row, provenance = _assemble_row(baseline[query_id], gemini[query_id])
        output_rows.append(row)
        row_provenance.append(provenance)
    try:
        validate_ranked_submission(output_rows, expected_query_ids=expected_ids)
    except (TypeError, ValueError) as error:
        raise AssemblyError(f"assembled submission failed repository validator: {error}") from error
    if [row["Q_ID"] for row in output_rows] != list(expected_ids_order):
        raise AssemblyError("assembled query ordering is not deterministic")
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "assembler": "assemble_vqa_gemini37_submission",
        "expected_query_count": expected_query_count,
        "row_count": len(output_rows),
        "expected_queries": {
            "path": str(expected_queries.resolve()),
            "sha256": _sha256_file(expected_queries),
        },
        "gemini_freeze": {
            "path": str(gemini_freeze.resolve()),
            "sha256": _sha256_file(gemini_freeze),
        },
        "baseline_submission": {
            "path": str(baseline_submission.resolve()),
            "sha256": _sha256_file(baseline_submission),
        },
        "passed_gemini_count": sum(
            item["gemini_status"] in _PASSED_STATUSES for item in row_provenance
        ),
        "baseline_fallback_count": sum(
            item["answer_source"] == "baseline_fallback" for item in row_provenance
        ),
        "rows": row_provenance,
    }
    written_output: Path | None = None
    written_provenance: Path | None = None
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        provenance["output_submission_sha256"] = _write_new_json(output_path, output_rows)
        written_output = output_path
        if provenance_path is None:
            provenance_path = output_path.with_name(output_path.stem + ".provenance.json")
        provenance_path = provenance_path.expanduser().resolve()
        _write_new_json(provenance_path, provenance)
        written_provenance = provenance_path
    return AssemblyResult(
        rows=tuple(output_rows),
        provenance=provenance,
        output_path=written_output,
        provenance_path=written_provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini-freeze", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--expected-queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--expected-query-count", type=int, default=EXPECTED_QUERY_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = assemble_submission(
            gemini_freeze=args.gemini_freeze,
            baseline_submission=args.baseline,
            expected_queries=args.expected_queries,
            expected_query_count=args.expected_query_count,
            output_path=args.output,
            provenance_path=args.provenance,
        )
    except (AssemblyError, OSError) as error:
        raise SystemExit(f"assemble_vqa_gemini37_submission: {error}") from error
    print(json.dumps(result.provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
