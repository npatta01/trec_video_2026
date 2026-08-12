#!/usr/bin/env python3
"""Score frozen Gemini whole-video synthesis treatment against A and B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from score_direct_gemini_evidence_canary import token_f1  # noqa: E402
from score_vqa_semantic_panel import (  # noqa: E402
    _default_bertscore,
    _default_meteor,
    normalize_nist_text,
    sha256_path,
    umbc_sts_score,
)


MANUAL_ASSESSMENTS = {
    "FBD7bvFnshE": {
        "verdict": "ambiguous",
        "reason": "A coherent whole-video topic, while the reference is a generic visual description.",
    },
    "FKsBrmJ2xSg": {
        "verdict": "wrong",
        "reason": "The synthesis captures all three dives, but the answer selects the immediate splash instead of the repeated-action answer.",
    },
    "IRTTtnSRTyY": {
        "verdict": "good",
        "reason": "Correctly identifies the repeated popping sound.",
    },
    "IVqpg_1aWXU": {"verdict": "good", "reason": "Correctly reads SIGMA AIRLINES."},
    "JTolYjiErBw": {
        "verdict": "good",
        "reason": "Correctly infers harvesting for consumption or use.",
    },
    "LSz_xBDBPNc": {
        "verdict": "wrong",
        "reason": "Misses the spoken instruction to put it down and incorrectly claims no later instruction.",
    },
    "Mj3kccs07fg": {
        "verdict": "ambiguous",
        "reason": "Three is visually plausible, while the frozen reference says two.",
    },
}
SUPPORTED_BASELINE_ROWS = {"IRTTtnSRTyY", "IVqpg_1aWXU", "JTolYjiErBw", "LSz_xBDBPNc"}


def load_frozen_predictions(root: Path) -> list[dict[str, Any]]:
    freeze = json.loads((root / "predictions-freeze.json").read_text())
    if freeze.get("row_count") != 7:
        raise ValueError("Expected a frozen seven-row synthesis treatment")
    rows = []
    for item in freeze["rows"]:
        path = root / "rows" / item["video_id"] / "complete.json"
        if sha256_path(path) != item["complete_sha256"]:
            raise ValueError(f"Frozen prediction changed: {item['video_id']}")
        rows.append(json.loads(path.read_text()))
    return rows


def _valid_spans(spans: list[dict[str, Any]], duration: float) -> bool:
    return bool(spans) and all(
        0
        <= float(span["start_seconds"])
        < float(span["end_seconds"])
        <= duration + 0.001
        for span in spans
    )


def audit_grounding(synthesis: dict[str, Any], duration: float) -> list[str]:
    """Return deterministic grounding issues for observation-bearing fields."""

    issues = []
    if not _valid_spans(synthesis["overview"]["support"], duration):
        issues.append("overview.support")
    for index, phase in enumerate(synthesis["phases"]):
        if not (
            0
            <= float(phase["start_seconds"])
            < float(phase["end_seconds"])
            <= duration + 0.001
        ):
            issues.append(f"phases[{index}]")
    for field in ("patterns", "state_changes", "salient_av", "inferences"):
        for index, item in enumerate(synthesis[field]):
            spans = item["instances"] if field == "patterns" else item["support"]
            if not _valid_spans(spans, duration):
                issues.append(f"{field}[{index}]")
            if field == "patterns" and int(item["count"]) != len(spans):
                issues.append(f"patterns[{index}].count")
    return issues


def score(
    *,
    root: Path,
    baseline_metrics_path: Path,
    scope_metrics_path: Path,
    bert_model: Path,
) -> dict[str, Any]:
    predictions = load_frozen_predictions(root)
    baseline_metrics = json.loads(baseline_metrics_path.read_text())
    scope_metrics = json.loads(scope_metrics_path.read_text())
    baseline = {
        row["video_id"]: row
        for row in baseline_metrics["rows"]
        if row["condition"] == "direct_answer"
    }
    scope = {
        row["video_id"]: row
        for row in scope_metrics["rows"]
        if row["condition"] == "scope_control"
    }
    pairs = [
        {
            "video_id": row["video_id"],
            "category": row["category"],
            "reference": baseline[row["video_id"]]["reference"],
            "direct_av": baseline[row["video_id"]]["prediction"],
            "scope_control": scope[row["video_id"]]["prediction"],
            "synthesis_treatment": row["answer"],
        }
        for row in predictions
    ]
    conditions = ("direct_av", "scope_control", "synthesis_treatment")
    candidates = [pair[condition] for condition in conditions for pair in pairs]
    references = [pair["reference"] for _condition in conditions for pair in pairs]
    bert = _default_bertscore(candidates, references, bert_model)
    rows = []
    index = 0
    for condition in conditions:
        for pair in pairs:
            prediction, reference = pair[condition], pair["reference"]
            meteor = _default_meteor(prediction, reference)
            sts = umbc_sts_score(prediction, reference)
            bertscore = float(bert[index])
            index += 1
            rows.append(
                {
                    "video_id": pair["video_id"],
                    "category": pair["category"],
                    "condition": condition,
                    "prediction": prediction,
                    "reference": reference,
                    "metrics": {
                        "normalized_exact_match": float(
                            normalize_nist_text(prediction)
                            == normalize_nist_text(reference)
                        ),
                        "token_f1": token_f1(prediction, reference),
                        "meteor": meteor,
                        "bertscore_f1": bertscore,
                        "sts": sts,
                        "semantic_composite": statistics.fmean(
                            (meteor, bertscore, sts)
                        ),
                        "ndcg": 1.0,
                    },
                }
            )
    summaries = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            metric: statistics.fmean(row["metrics"][metric] for row in subset)
            for metric in subset[0]["metrics"]
        }

    protocol = json.loads((root / "protocol.json").read_text())
    durations = {row["video_id"]: row["duration_seconds"] for row in protocol["rows"]}
    grounding = {}
    for row in predictions:
        video_id = row["video_id"]
        synthesis = json.loads(
            (root / "rows" / video_id / "synthesis" / "response.json").read_text()
        )
        grounding[video_id] = audit_grounding(synthesis, durations[video_id])
    verdicts = {
        video_id: item["verdict"] for video_id, item in MANUAL_ASSESSMENTS.items()
    }
    regressions = sorted(
        video_id for video_id in SUPPORTED_BASELINE_ROWS if verdicts[video_id] != "good"
    )
    gate = {
        "passed": verdicts["FKsBrmJ2xSg"] == "good"
        and not regressions
        and not any(grounding.values()),
        "repetition_answer_repaired": verdicts["FKsBrmJ2xSg"] == "good",
        "supported_answer_regressions": regressions,
        "all_synthesis_observations_timestamp_grounded": not any(grounding.values()),
    }
    return {
        "schema_version": 1,
        "panel": {"row_count": 7, "descriptive_only": True},
        "conditions": summaries,
        "manual_assessments": MANUAL_ASSESSMENTS,
        "grounding_audit": grounding,
        "gate": gate,
        "rows": rows,
        "integrity": {
            "predictions_frozen_before_labels": True,
            "prediction_freeze_sha256": sha256_path(root / "predictions-freeze.json"),
            "protocol_sha256": sha256_path(root / "protocol.json"),
        },
        "metric_notes": {
            "semantic_composite": "mean(METEOR, BERTScore-F1, STS)",
            "ndcg": "Exactly 1.0 and non-informative for one answer.",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--scope-metrics", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = score(
        root=args.root,
        baseline_metrics_path=args.baseline_metrics,
        scope_metrics_path=args.scope_metrics,
        bert_model=args.bert_model,
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate": artifact["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
