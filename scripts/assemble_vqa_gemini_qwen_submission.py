"""Assemble an offline Gemini/Qwen fusion TRECVID VQA submission.

The fusion policy is intentionally small and auditable: for each query it
keeps the first distinct successful Gemini answer, then the transcript and
native Qwen answers, then fills the remaining ranks with the validated
baseline.  Failed model rows are skipped.  This script never opens labels or
media and never calls a provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trec_video_2026.vqa_deadline_canary import (  # noqa: E402
    validate_ranked_submission,
)

from scripts.assemble_vqa_gemini37_submission import (  # noqa: E402
    AssemblyError as GeminiAssemblyError,
    _baseline_rows,
    _expected_queries,
    _gemini_rows,
    _normalize_answer,
    _sha256_file,
)
from scripts.assemble_vqa_qwen36_submission import (  # noqa: E402
    AssemblyError as QwenAssemblyError,
    _qwen_rows,
)


EXPECTED_QUERY_COUNT = 2_443
DEFAULT_BASELINE = Path(
    "data/vqa/submissions/qwen35-safety-20260810-a1/final/submission-top10.json"
)
TRANSCRIPT_CONDITION_ID = "video-plus-transcript"
NATIVE_CONDITION_ID = "native-video"


class AssemblyError(ValueError):
    """Raised when frozen inputs cannot produce an official-shaped submission."""


@dataclass(frozen=True)
class AssemblyResult:
    rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    output_path: Path | None = None
    provenance_path: Path | None = None


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _answer_time(candidate: Mapping[str, Any], *, query_id: int, source: str) -> float:
    value = candidate.get("time")
    source_label = {
        "gemini": "Gemini",
        "qwen_video_plus_transcript": "Qwen transcript",
        "qwen_native_video": "Qwen native",
    }.get(source, source)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(
            f"{source_label} answer time is not numeric for query {query_id}"
        )
    if not math.isfinite(float(value)) or value < 0:
        raise AssemblyError(
            f"{source_label} answer time is invalid for query {query_id}"
        )
    return float(value)


def _write_new_json(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise AssemblyError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _assemble_row(
    baseline: Mapping[str, Any],
    gemini: Mapping[str, Any],
    qwen: Mapping[tuple[int, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    query_id = int(baseline["Q_ID"])
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    if gemini["status"] in {"passed", "succeeded", "success", "complete"}:
        candidates.append(("gemini", gemini))
    candidates.extend(
        (
            source,
            qwen[(query_id, condition_id)],
        )
        for source, condition_id in (
            ("qwen_video_plus_transcript", TRANSCRIPT_CONDITION_ID),
            ("qwen_native_video", NATIVE_CONDITION_ID),
        )
        if qwen[(query_id, condition_id)]["status"]
        in {"passed", "succeeded", "success", "complete"}
    )

    selected: list[dict[str, Any]] = []
    selected_sources: list[str] = []
    seen: set[str] = set()
    source_counts = {
        "gemini": 0,
        "qwen_video_plus_transcript": 0,
        "qwen_native_video": 0,
        "baseline": 0,
    }
    for source, candidate in candidates:
        answer = str(candidate["answer"])
        normalized = _normalize_answer(answer)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            {
                "Rank": len(selected) + 1,
                "Answer": answer,
                "Time": _answer_time(
                    candidate, query_id=query_id, source=source
                ),
            }
        )
        selected_sources.append(source)
        source_counts[source] += 1

    baseline_ranks: list[int] = []
    for candidate in sorted(baseline["Answers"], key=lambda item: item["Rank"]):
        answer = str(candidate["Answer"])
        normalized = _normalize_answer(answer)
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
        selected_sources.append("baseline")
        baseline_ranks.append(int(candidate["Rank"]))
        source_counts["baseline"] += 1
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise AssemblyError(f"not enough distinct answers for query {query_id}")

    result = dict(baseline)
    result["Answers"] = selected
    provenance = {
        "Q_ID": query_id,
        "Video_ID": baseline["Video_ID"],
        "gemini_status": gemini["status"],
        "qwen_statuses": {
            TRANSCRIPT_CONDITION_ID: qwen[(query_id, TRANSCRIPT_CONDITION_ID)]["status"],
            NATIVE_CONDITION_ID: qwen[(query_id, NATIVE_CONDITION_ID)]["status"],
        },
        "selected_sources": selected_sources,
        "baseline_ranks_used": baseline_ranks,
        "answer_sha256": [_sha256_bytes(item["Answer"]) for item in selected],
    }
    return result, provenance, source_counts


def assemble_submission(
    *,
    gemini_freeze: Path,
    qwen_freeze: Path,
    baseline_submission: Path = DEFAULT_BASELINE,
    expected_queries: Path,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    output_path: Path | None = None,
    provenance_path: Path | None = None,
) -> AssemblyResult:
    """Build and validate the deterministic Gemini/Qwen fusion submission."""

    try:
        expected_ids_order, expected_videos = _expected_queries(expected_queries)
        if len(expected_ids_order) != expected_query_count:
            raise AssemblyError(
                f"expected query set must contain exactly {expected_query_count} IDs"
            )
        expected_ids = set(expected_ids_order)
        baseline = _baseline_rows(baseline_submission, expected_ids, expected_videos)
        gemini = _gemini_rows(gemini_freeze, expected_ids, expected_videos)
        qwen, model_binding = _qwen_rows(
            qwen_freeze,
            expected_ids,
            expected_videos,
            run_manifest=None,
        )
    except (GeminiAssemblyError, QwenAssemblyError) as error:
        raise AssemblyError(str(error)) from error

    output_rows: list[dict[str, Any]] = []
    row_provenance: list[dict[str, Any]] = []
    selected_counts = {
        "gemini": 0,
        "qwen_video_plus_transcript": 0,
        "qwen_native_video": 0,
        "baseline": 0,
    }
    for query_id in expected_ids_order:
        row, provenance, counts = _assemble_row(
            baseline[query_id], gemini[query_id], qwen
        )
        output_rows.append(row)
        row_provenance.append(provenance)
        for source, count in counts.items():
            selected_counts[source] += count

    try:
        validate_ranked_submission(output_rows, expected_query_ids=expected_ids)
    except (TypeError, ValueError) as error:
        raise AssemblyError(f"assembled submission failed repository validator: {error}") from error
    if [row["Q_ID"] for row in output_rows] != list(expected_ids_order):
        raise AssemblyError("assembled query ordering is not deterministic")

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "assembler": "assemble_vqa_gemini_qwen_submission",
        "policy": [
            "gemini",
            "qwen_video_plus_transcript",
            "qwen_native_video",
            "baseline",
        ],
        "expected_query_count": expected_query_count,
        "row_count": len(output_rows),
        "model": {
            "gemini": "gemini-3.7-flash",
            "qwen": model_binding,
        },
        "expected_queries": {
            "path": str(expected_queries.resolve()),
            "sha256": _sha256_file(expected_queries),
        },
        "gemini_freeze": {
            "path": str(gemini_freeze.resolve()),
            "sha256": _sha256_file(gemini_freeze),
        },
        "qwen_freeze": {
            "path": str(qwen_freeze.resolve()),
            "sha256": _sha256_file(qwen_freeze),
        },
        "baseline_submission": {
            "path": str(baseline_submission.resolve()),
            "sha256": _sha256_file(baseline_submission),
        },
        "selected_counts": selected_counts,
        "rows": row_provenance,
    }

    written_output = None
    written_provenance = None
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        if provenance_path is None:
            provenance_path = output_path.with_name(output_path.stem + ".provenance.json")
        provenance_path = provenance_path.expanduser().resolve()
        if output_path.exists() or output_path.is_symlink():
            raise AssemblyError(f"refusing to overwrite existing output: {output_path}")
        if provenance_path.exists() or provenance_path.is_symlink():
            raise AssemblyError(
                f"refusing to overwrite existing output: {provenance_path}"
            )
        provenance["output_submission_sha256"] = _write_new_json(output_path, output_rows)
        _write_new_json(provenance_path, provenance)
        written_output = output_path
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
    parser.add_argument("--qwen-freeze", type=Path, required=True)
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
            qwen_freeze=args.qwen_freeze,
            baseline_submission=args.baseline,
            expected_queries=args.expected_queries,
            expected_query_count=args.expected_query_count,
            output_path=args.output,
            provenance_path=args.provenance,
        )
    except (AssemblyError, OSError) as error:
        raise SystemExit(f"assemble_vqa_gemini_qwen_submission: {error}") from error
    print(json.dumps(result.provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
