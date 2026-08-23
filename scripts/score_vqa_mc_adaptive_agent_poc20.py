#!/usr/bin/env python3
"""Score frozen adaptive-agent MC rankings after the inference freeze."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    canonical_json,
    parse_ranking,
    score_rankings,
    sha256_file,
)


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not an object: {path.name}")
    return value


def _rows(path: Path) -> list[Mapping[str, Any]]:
    rows = _read(path).get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"artifact rows are invalid: {path.name}")
    return list(rows)


def _prediction_index(
    path: Path, request_ids: Sequence[str], *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _rows(path):
        if row.get("variant_kind") != "canonical":
            continue
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or video_id not in request_ids:
            raise ValueError(f"{label} prediction identity drift")
        if video_id in indexed:
            raise ValueError(f"duplicate {label} prediction row: {video_id}")
        indexed[video_id] = row
    if list(indexed) != list(request_ids):
        raise ValueError(f"{label} prediction identity drift")
    return indexed


def _inspection_index(
    path: Path, request_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in _rows(path):
        video_id = row.get("video_id")
        inspection = row.get("inspection")
        if (
            not isinstance(video_id, str)
            or video_id not in request_ids
            or not isinstance(inspection, Mapping)
        ):
            raise ValueError("inspection identity drift")
        if video_id in indexed:
            raise ValueError(f"duplicate inspection row: {video_id}")
        indexed[video_id] = inspection
    if list(indexed) != list(request_ids):
        raise ValueError("inspection identity drift")
    return indexed


def _valid_ranking(row: Mapping[str, Any]) -> list[str] | None:
    if row.get("status") != "passed" or not isinstance(row.get("ranking"), list):
        return None
    try:
        return parse_ranking({"ranking": row["ranking"]})
    except ValueError:
        return None


def _metric_by_stratum(
    predictions: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
    strata: Mapping[str, str],
) -> dict[str, dict[str, int | float]]:
    return {
        stratum: score_rankings(
            {
                query_id: predictions.get(query_id)
                for query_id in labels
                if strata[query_id] == stratum
            },
            {
                query_id: answer
                for query_id, answer in labels.items()
                if strata[query_id] == stratum
            },
        )
        for stratum in sorted(set(strata.values()))
    }


def _transitions(
    before: Mapping[str, Sequence[str] | None],
    after: Mapping[str, Sequence[str] | None],
    labels: Mapping[str, str],
) -> dict[str, int]:
    counts = Counter(
        {
            "repaired": 0,
            "regressed": 0,
            "unchanged_correct": 0,
            "unchanged_wrong": 0,
        }
    )
    for query_id, correct in labels.items():
        before_correct = before.get(query_id) is not None and before[query_id][0] == correct
        after_correct = after.get(query_id) is not None and after[query_id][0] == correct
        if not before_correct and after_correct:
            counts["repaired"] += 1
        elif before_correct and not after_correct:
            counts["regressed"] += 1
        elif before_correct:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return dict(counts)


def _interval_bucket(fraction: float) -> str:
    if fraction <= 0.25:
        return "0-25%"
    if fraction <= 0.50:
        return "25-50%"
    if fraction <= 0.75:
        return "50-75%"
    return "75-100%"


def score(
    *,
    requests_path: Path,
    labels_path: Path,
    control_path: Path,
    stage1_path: Path,
    final_path: Path,
    inspections_path: Path,
) -> dict[str, Any]:
    request_rows = _rows(requests_path)
    label_rows = _rows(labels_path)
    request_ids = [row.get("video_id") for row in request_rows]
    label_ids = [row.get("query_id") for row in label_rows]
    if (
        not all(isinstance(value, str) for value in request_ids)
        or request_ids != label_ids
        or len(set(request_ids)) != len(request_ids)
    ):
        raise ValueError("request/label identity drift")
    ids = [str(value) for value in request_ids]
    labels = {row["query_id"]: row["correct_option_id"] for row in label_rows}
    strata = {row["video_id"]: row["stratum"] for row in request_rows}
    durations = {
        row["video_id"]: float(row["duration_seconds"]) for row in request_rows
    }

    control_index = _prediction_index(control_path, ids, label="control")
    stage1_index = _prediction_index(stage1_path, ids, label="stage1")
    final_index = _prediction_index(final_path, ids, label="final")
    inspections = _inspection_index(inspections_path, ids)
    systems = {
        "frozen_control": {
            video_id: _valid_ranking(control_index[video_id]) for video_id in ids
        },
        "stage1": {
            video_id: _valid_ranking(stage1_index[video_id]) for video_id in ids
        },
        "final": {
            video_id: _valid_ranking(final_index[video_id]) for video_id in ids
        },
    }
    metrics = {
        name: score_rankings(predictions, labels)
        for name, predictions in systems.items()
    }
    by_stratum = {
        name: _metric_by_stratum(predictions, labels, strata)
        for name, predictions in systems.items()
    }
    transitions_stage1 = _transitions(systems["stage1"], systems["final"], labels)
    transitions_control = _transitions(
        systems["frozen_control"], systems["final"], labels
    )

    fps_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    gap_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    interval_counts: Counter[str] = Counter(
        {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
    )
    final_source_counts: Counter[str] = Counter()
    for video_id in ids:
        plan = inspections[video_id]
        fps_counts[str(float(plan["fps"]))] += 1
        resolution_counts[str(plan["resolution"])] += 1
        gap_counts[str(plan["evidence_gap"])] += 1
        source_counts[str(plan["source"])] += 1
        fraction = (
            float(plan["end_seconds"]) - float(plan["start_seconds"])
        ) / durations[video_id]
        if not 0 < fraction <= 1.000001:
            raise ValueError(f"inspection interval fraction is invalid for {video_id}")
        interval_counts[_interval_bucket(fraction)] += 1
        final_source_counts[str(final_index[video_id].get("final_source"))] += 1

    decision_gate = {
        "requires_final_valid_count": len(ids),
        "final_exceeds_stage1": (
            metrics["final"]["top1_correct"] > metrics["stage1"]["top1_correct"]
        ),
        "final_exceeds_frozen_control": (
            metrics["final"]["top1_correct"]
            > metrics["frozen_control"]["top1_correct"]
        ),
        "repairs_exceed_regressions": (
            transitions_stage1["repaired"] > transitions_stage1["regressed"]
        ),
    }
    decision_gate["promising_prototype"] = bool(
        metrics["final"]["valid_count"] == len(ids)
        and all(
            decision_gate[key]
            for key in (
                "final_exceeds_stage1",
                "final_exceeds_frozen_control",
                "repairs_exceed_regressions",
            )
        )
    )

    rows = [
        {
            "query_id": video_id,
            "stratum": strata[video_id],
            "correct_option_id": labels[video_id],
            "frozen_control_ranking": systems["frozen_control"][video_id],
            "stage1_ranking": systems["stage1"][video_id],
            "final_ranking": systems["final"][video_id],
            "final_source": final_index[video_id].get("final_source"),
            "inspection": dict(inspections[video_id]),
        }
        for video_id in ids
    ]
    return {
        "schema_version": 1,
        "status": "scored",
        "source_sha256s": {
            "requests": sha256_file(requests_path),
            "labels": sha256_file(labels_path),
            "frozen_control": sha256_file(control_path),
            "stage1": sha256_file(stage1_path),
            "final": sha256_file(final_path),
            "inspections": sha256_file(inspections_path),
        },
        "metrics": metrics,
        "by_stratum": by_stratum,
        "transitions_stage1_to_final": transitions_stage1,
        "transitions_control_to_final": transitions_control,
        "inspection_distribution": {
            "fps": dict(sorted(fps_counts.items())),
            "resolution": dict(sorted(resolution_counts.items())),
            "evidence_gap": dict(sorted(gap_counts.items())),
            "source": dict(sorted(source_counts.items())),
            "interval_fraction": dict(interval_counts),
        },
        "final_source_counts": dict(sorted(final_source_counts.items())),
        "decision_gate": decision_gate,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_vqa_mc_adaptive_agent_poc20")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--inspections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = score(
        requests_path=args.requests,
        labels_path=args.labels,
        control_path=args.control,
        stage1_path=args.stage1,
        final_path=args.final,
        inspections_path=args.inspections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json(result))
    print(
        json.dumps(
            {
                "status": "scored",
                "metrics": result["metrics"],
                "transitions_stage1_to_final": result[
                    "transitions_stage1_to_final"
                ],
                "decision_gate": result["decision_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
