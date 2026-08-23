#!/usr/bin/env python3
"""Join frozen MC predictions to private labels and score provider families."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    CONDITIONS,
    canonical_json,
    fuse_borda,
    score_rankings,
    sha256_file,
)


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not an object: {path.name}")
    return value


def _valid_ranking(row: Mapping[str, Any] | None) -> list[str] | None:
    if row is None or row.get("status") != "passed":
        return None
    ranking = row.get("ranking")
    if not isinstance(ranking, list):
        return None
    return list(ranking)


def _metric_by_stratum(
    predictions: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int | float]]:
    return {
        stratum: score_rankings(
            {query_id: predictions.get(query_id) for query_id in labels if strata[query_id] == stratum},
            {query_id: label for query_id, label in labels.items() if strata[query_id] == stratum},
        )
        for stratum in sorted(set(strata.values()))
    }


def score(
    *,
    requests_path: Path,
    labels_path: Path,
    gemini_path: Path,
    qwen_path: Path | None,
) -> dict[str, Any]:
    requests = _read(requests_path)
    labels_payload = _read(labels_path)
    request_rows = requests.get("rows")
    label_rows = labels_payload.get("rows")
    if not isinstance(request_rows, list) or not isinstance(label_rows, list):
        raise ValueError("request and label artifacts must contain rows")
    request_ids = [row.get("video_id") for row in request_rows if isinstance(row, Mapping)]
    label_ids = [row.get("query_id") for row in label_rows if isinstance(row, Mapping)]
    if request_ids != label_ids or len(request_ids) != len(request_rows):
        raise ValueError("request/label identity drift")
    labels = {row["query_id"]: row["correct_option_id"] for row in label_rows}
    strata = {row["video_id"]: row["stratum"] for row in request_rows}

    gemini_rows = _read(gemini_path).get("rows")
    if not isinstance(gemini_rows, list):
        raise ValueError("Gemini prediction artifact must contain rows")
    gemini_index: dict[str, Mapping[str, Any]] = {}
    for row in gemini_rows:
        if not isinstance(row, Mapping) or row.get("variant_kind") != "canonical":
            continue
        video_id = row.get("video_id")
        if video_id in gemini_index:
            raise ValueError(f"duplicate canonical Gemini row: {video_id}")
        gemini_index[video_id] = row
    gemini = {query_id: _valid_ranking(gemini_index.get(query_id)) for query_id in request_ids}

    qwen_views: dict[str, dict[str, list[str] | None]] = {
        condition: {query_id: None for query_id in request_ids} for condition in CONDITIONS
    }
    if qwen_path is not None:
        qwen_rows = _read(qwen_path).get("rows")
        if not isinstance(qwen_rows, list):
            raise ValueError("Qwen prediction artifact must contain rows")
        seen: set[tuple[str, str]] = set()
        for row in qwen_rows:
            if not isinstance(row, Mapping) or row.get("variant_kind") != "canonical":
                continue
            key = (str(row.get("video_id")), str(row.get("condition_id")))
            if key in seen:
                raise ValueError(f"duplicate canonical Qwen row: {key}")
            seen.add(key)
            if key[0] in labels and key[1] in qwen_views:
                qwen_views[key[1]][key[0]] = _valid_ranking(row)

    qwen_fused: dict[str, list[str] | None] = {}
    family_fused: dict[str, list[str] | None] = {}
    comparable = 0
    top1_agreement = 0
    for query_id in request_ids:
        transcript = qwen_views[CONDITIONS[0]][query_id]
        native = qwen_views[CONDITIONS[1]][query_id]
        if transcript is not None and native is not None:
            qwen_fused[query_id] = fuse_borda([transcript, native])
            comparable += 1
            top1_agreement += transcript[0] == native[0]
        else:
            qwen_fused[query_id] = None
        family_fused[query_id] = (
            fuse_borda([gemini[query_id], qwen_fused[query_id]])
            if gemini[query_id] is not None and qwen_fused[query_id] is not None
            else None
        )

    systems = {
        "gemini": gemini,
        "qwen_video_plus_transcript": qwen_views[CONDITIONS[0]],
        "qwen_native_video": qwen_views[CONDITIONS[1]],
        "qwen_dual_view": qwen_fused,
        "equal_family_borda": family_fused,
    }
    metrics = {name: score_rankings(predictions, labels) for name, predictions in systems.items()}
    by_stratum = {
        name: _metric_by_stratum(predictions, labels, strata)
        for name, predictions in systems.items()
    }
    rows = [
        {
            "query_id": query_id,
            "stratum": strata[query_id],
            "correct_option_id": labels[query_id],
            "gemini_ranking": gemini[query_id],
            "qwen_video_plus_transcript_ranking": qwen_views[CONDITIONS[0]][query_id],
            "qwen_native_video_ranking": qwen_views[CONDITIONS[1]][query_id],
            "qwen_dual_view_ranking": qwen_fused[query_id],
            "equal_family_borda_ranking": family_fused[query_id],
        }
        for query_id in request_ids
    ]
    return {
        "schema_version": 1,
        "status": "scored",
        "source_sha256s": {
            "requests": sha256_file(requests_path),
            "labels": sha256_file(labels_path),
            "gemini": sha256_file(gemini_path),
            "qwen": sha256_file(qwen_path) if qwen_path is not None else None,
        },
        "metrics": metrics,
        "by_stratum": by_stratum,
        "qwen_view_comparable_count": comparable,
        "qwen_view_top1_agreement_count": top1_agreement,
        "qwen_view_top1_agreement_rate": top1_agreement / comparable if comparable else None,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_vqa_mc_poc20")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = score(
        requests_path=args.requests,
        labels_path=args.labels,
        gemini_path=args.gemini,
        qwen_path=args.qwen,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json(result))
    print(json.dumps({"status": "scored", "metrics": result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
