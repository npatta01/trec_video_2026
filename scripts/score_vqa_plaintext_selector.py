#!/usr/bin/env python3
"""Score the label-free plaintext event-map selector against the 475-row panel.

This is deliberately a thin, production-independent adapter around the pinned
semantic metric implementations.  Labels are loaded only inside this process;
the emitted artifact contains video IDs and metrics, never questions or
reference/prediction text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score_vqa_factorial as factorial  # noqa: E402
import score_vqa_semantic_panel as task1  # noqa: E402


EXPECTED_ROWS = 475
EXPECTED_ANNOTATIONS = 481
EXPECTED_BERT_MODEL = "722cf37b1afa9454edce342e7895e588b6ff1d59"
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_ITERATIONS = 2000


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"row {line_number} is not an object")
        rows.append(value)
    return rows


def _mean(rows: list[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _aggregate(rows: list[Mapping[str, Any]], *, include_strata: bool = True) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty rows")
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row["stratum"])].append(row)
    result: dict[str, Any] = {
        "row_count": len(rows),
        "meteor": _mean(rows, "meteor"),
        "bertscore_f1": _mean(rows, "bertscore_f1"),
        "sts": _mean(rows, "sts"),
        "semantic_composite": _mean(rows, "semantic_composite"),
        "normalized_exact_match": _mean(rows, "normalized_exact_match"),
        "token_f1": _mean(rows, "token_f1"),
        # NDCG with one answer per video is always 1.0 and is not informative.
        "ndcg": 1.0,
        "ndcg_degenerate": True,
    }
    if include_strata:
        result["strata"] = {
            name: _aggregate(group, include_strata=False)
            for name, group in sorted(by_stratum.items())
        }
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _bootstrap_mean(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    return {
        "lower_95": _percentile(means, 0.025),
        "upper_95": _percentile(means, 0.975),
    }


def _score_pairs(
    pairs: list[tuple[str, str]],
    *,
    bert_model: Path,
    sts_workers: int,
) -> dict[tuple[str, str], dict[str, float]]:
    ordered_pairs = sorted(set(pairs))
    meteor = {
        pair: float(task1._default_meteor(pair[0], pair[1]))
        for pair in ordered_pairs
    }
    bert_values = task1._default_bertscore(
        [pair[0] for pair in ordered_pairs],
        [pair[1] for pair in ordered_pairs],
        bert_model,
    )
    if len(bert_values) != len(ordered_pairs):
        raise ValueError("BERTScore returned the wrong number of rows")

    def score_sts(pair: tuple[str, str]) -> tuple[tuple[str, str], float]:
        return pair, float(
            task1.umbc_sts_score(
                pair[0],
                pair[1],
                timeout=task1.DEFAULT_STS_TIMEOUT,
                retries=task1.DEFAULT_STS_RETRIES,
            )
        )

    sts: dict[tuple[str, str], float] = {}
    with ThreadPoolExecutor(max_workers=sts_workers) as executor:
        futures = [executor.submit(score_sts, pair) for pair in ordered_pairs]
        for future in as_completed(futures):
            pair, value = future.result()
            sts[pair] = value

    return {
        pair: {
            "meteor": meteor[pair],
            "bertscore_f1": float(bert_value),
            "sts": sts[pair],
            "semantic_composite": statistics.fmean(
                (meteor[pair], float(bert_value), sts[pair])
            ),
        }
        for pair, bert_value in zip(ordered_pairs, bert_values, strict=True)
    }


def _main(args: argparse.Namespace) -> dict[str, Any]:
    annotations = task1._load_annotations(args.annotations)
    if len(annotations) != EXPECTED_ANNOTATIONS:
        raise ValueError(f"expected {EXPECTED_ANNOTATIONS} annotation files, got {len(annotations)}")
    source_rows = _read_jsonl(args.results)
    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} result rows, got {len(source_rows)}")
    by_id = {str(row.get("video_id")): row for row in source_rows}
    if len(by_id) != EXPECTED_ROWS or set(by_id) - set(annotations):
        raise ValueError("result IDs do not match the declared annotation panel")

    passed = [row for row in source_rows if row.get("status") == "passed" and str(row.get("selector_answer", "")).strip()]
    failed = [row for row in source_rows if row not in passed]
    pairs = [
        (
            factorial.normalize_nist_text(str(row["selector_answer"])),
            factorial.normalize_nist_text(str(annotations[str(row["video_id"])]["reference"])),
        )
        for row in passed
    ]
    scored_pairs = _score_pairs(pairs, bert_model=args.bert_model, sts_workers=args.sts_workers)

    scored_rows: list[dict[str, Any]] = []
    for row, pair in zip(passed, pairs, strict=True):
        prediction, reference = pair
        metrics = scored_pairs[pair]
        annotation = annotations[str(row["video_id"])]
        scored_rows.append(
            {
                "video_id": str(row["video_id"]),
                "stratum": annotation["question_stratum"],
                "status": "passed",
                "normalized_exact_match": factorial.normalized_exact_match(prediction, reference),
                "token_f1": factorial.token_f1(prediction, reference),
                "answer_token_count": len(prediction.split()),
                **metrics,
                "ndcg": 1.0,
                "ndcg_degenerate": True,
            }
        )

    # Report the honest full-panel view too: an answer-generation failure gets
    # zero quality credit, while NDCG remains explicitly degenerate.
    full_rows = list(scored_rows)
    for row in failed:
        video_id = str(row["video_id"])
        annotation = annotations[video_id]
        full_rows.append(
            {
                "video_id": video_id,
                "stratum": annotation["question_stratum"],
                "status": "failed_zero_credit",
                "failure_type": row.get("error_type"),
                "normalized_exact_match": 0.0,
                "token_f1": 0.0,
                "answer_token_count": 0,
                "meteor": 0.0,
                "bertscore_f1": 0.0,
                "sts": 0.0,
                "semantic_composite": 0.0,
                "ndcg": 1.0,
                "ndcg_degenerate": True,
            }
        )
    full_rows.sort(key=lambda row: row["video_id"])
    scored_rows.sort(key=lambda row: row["video_id"])

    failure_counts = Counter(str(row.get("error_type") or "unknown") for row in failed)
    quality_rows = [
        {
            "video_id": row["video_id"],
            "status": row["status"],
            "stratum": row["stratum"],
            **{field: row[field] for field in ("meteor", "bertscore_f1", "sts", "semantic_composite")},
        }
        for row in scored_rows
    ]
    output = {
        "schema_version": 1,
        "experiment": "vqa-plaintext-event-map-full-20260810",
        "panel": {
            "annotation_files": len(annotations),
            "videos": EXPECTED_ROWS,
            "completed_selector_outputs": len(passed),
            "failed_selector_outputs": len(failed),
            "completion_rate": len(passed) / EXPECTED_ROWS,
        },
        "metrics": {
            "official_style": ["METEOR", "BERTScore-F1", "STS", "NDCG"],
            "selector_passed_only": _aggregate(scored_rows),
            "full_panel_failure_penalized": _aggregate(full_rows),
            "interpretation": {
                "ndcg": "1.0 is degenerate for one response per video; no ranking comparison was available",
                "failure_penalty": "failed selector rows receive zero credit for quality metrics",
            },
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "failures": [
            {
                "video_id": str(row["video_id"]),
                "status": row.get("status"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
            }
            for row in sorted(failed, key=lambda item: str(item["video_id"]))
        ],
        "rows": quality_rows,
        "integrity": {
            "labels_joined_after_prediction_freeze": True,
            "reference_text_emitted": False,
            "prediction_text_emitted": False,
            "bert_model_snapshot": args.bert_model.name,
            "bert_model_snapshot_expected": EXPECTED_BERT_MODEL,
            "sts_endpoint": task1.STS_ENDPOINT,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sts-workers", type=int, default=8)
    return parser


if __name__ == "__main__":
    result = _main(build_parser().parse_args())
    print(json.dumps({"output": str(result), "panel": result["panel"]}, sort_keys=True))
