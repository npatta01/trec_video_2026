#!/usr/bin/env python3
"""Score the frozen scope-aware Gemini control against its direct-AV baseline."""

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


MANUAL_VERDICTS = {
    "FBD7bvFnshE": {
        "verdict": "ambiguous",
        "reason": "Plausible topical synthesis, but the reference expects a generic visual description.",
    },
    "FKsBrmJ2xSg": {
        "verdict": "wrong",
        "reason": "Treats the clip as a simple sequence and misses the repeated-diving pattern.",
    },
    "IRTTtnSRTyY": {
        "verdict": "good",
        "reason": "Correctly identifies the repeated popping sound.",
    },
    "IVqpg_1aWXU": {
        "verdict": "good",
        "reason": "Correctly reads SIGMA AIRLINES from the native video.",
    },
    "JTolYjiErBw": {
        "verdict": "wrong",
        "reason": "Abstains despite the existing direct native-AV answer correctly inferring eating intent.",
    },
    "LSz_xBDBPNc": {
        "verdict": "wrong",
        "reason": "Hallucinates turn the lights out instead of the supported instruction to put it down.",
    },
    "Mj3kccs07fg": {
        "verdict": "ambiguous",
        "reason": "Three is visually plausible, while the frozen reference says two.",
    },
}
SUPPORTED_BASELINE_ROWS = {
    "IRTTtnSRTyY",
    "IVqpg_1aWXU",
    "JTolYjiErBw",
    "LSz_xBDBPNc",
}


def load_predictions(root: Path) -> list[dict[str, Any]]:
    """Load complete rows only after validating the immutable prediction freeze."""

    freeze = json.loads((root / "predictions-freeze.json").read_text())
    if freeze.get("row_count") != 7:
        raise ValueError("Expected a frozen seven-row scope control")
    rows = []
    for item in freeze["rows"]:
        path = root / "rows" / item["video_id"] / "complete.json"
        if sha256_path(path) != item["complete_sha256"]:
            raise ValueError(f"Frozen prediction changed: {item['video_id']}")
        rows.append(json.loads(path.read_text()))
    return rows


def manual_gate(verdicts: dict[str, str]) -> dict[str, Any]:
    """Apply the predeclared mechanism and no-regression gate."""

    repetition_repaired = verdicts.get("FKsBrmJ2xSg") == "good"
    regressions = sorted(
        video_id
        for video_id in SUPPORTED_BASELINE_ROWS
        if verdicts.get(video_id) != "good"
    )
    return {
        "passed": repetition_repaired and not regressions,
        "repetition_repaired": repetition_repaired,
        "supported_answer_regressions": regressions,
    }


def score(
    *,
    root: Path,
    baseline_root: Path,
    annotation_root: Path,
    bert_model: Path,
) -> dict[str, Any]:
    """Score scope-aware and existing direct-AV predictions after freeze."""

    scope_rows = load_predictions(root)
    baseline_metrics = json.loads(
        (baseline_root / "postfreeze-metrics.json").read_text()
    )
    baseline = {
        row["video_id"]: row
        for row in baseline_metrics["rows"]
        if row["condition"] == "direct_answer"
    }
    pairs = []
    for row in scope_rows:
        annotation = json.loads(
            (annotation_root / f"{row['video_id']}.json").read_text()
        )
        if annotation["question"] != row["question"]:
            raise ValueError(f"Question mismatch: {row['video_id']}")
        pairs.append(
            {
                "video_id": row["video_id"],
                "category": row["category"],
                "reference": annotation["correct_answer"],
                "baseline": baseline[row["video_id"]]["prediction"],
                "scope_control": row["answer"],
            }
        )
    predictions = [pair[condition] for condition in ("baseline", "scope_control") for pair in pairs]
    references = [pair["reference"] for _condition in ("baseline", "scope_control") for pair in pairs]
    bert = _default_bertscore(predictions, references, bert_model)
    scored = []
    index = 0
    for condition in ("baseline", "scope_control"):
        for pair in pairs:
            prediction = pair[condition]
            reference = pair["reference"]
            meteor = _default_meteor(prediction, reference)
            sts = umbc_sts_score(prediction, reference)
            bertscore = float(bert[index])
            index += 1
            scored.append(
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
    for condition in ("baseline", "scope_control"):
        rows = [row for row in scored if row["condition"] == condition]
        summaries[condition] = {
            metric: statistics.fmean(row["metrics"][metric] for row in rows)
            for metric in rows[0]["metrics"]
        }
    verdicts = {
        video_id: assessment["verdict"]
        for video_id, assessment in MANUAL_VERDICTS.items()
    }
    return {
        "schema_version": 1,
        "panel": {"row_count": 7, "descriptive_only": True},
        "conditions": summaries,
        "manual_assessments": MANUAL_VERDICTS,
        "gate": manual_gate(verdicts),
        "rows": scored,
        "integrity": {
            "predictions_frozen_before_labels": True,
            "prediction_freeze_sha256": sha256_path(
                root / "predictions-freeze.json"
            ),
            "protocol_sha256": sha256_path(root / "protocol.json"),
        },
        "metric_notes": {
            "semantic_composite": "mean(METEOR, BERTScore-F1, STS)",
            "ndcg": "Exactly 1.0 and non-informative for one answer.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = score(
        root=args.root,
        baseline_root=args.baseline_root,
        annotation_root=args.annotations,
        bert_model=args.bert_model,
    )
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "gate": artifact["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
