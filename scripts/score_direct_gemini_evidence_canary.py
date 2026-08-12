#!/usr/bin/env python3
"""Score the frozen seven-row direct Gemini evidence canary."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from score_vqa_semantic_panel import (  # noqa: E402
    _default_bertscore,
    _default_meteor,
    normalize_nist_text,
    sha256_path,
    umbc_sts_score,
)


CONDITIONS = ("direct_answer", "timeline_answer", "frame_answer")

MANUAL_ASSESSMENTS = {
    "FBD7bvFnshE": {
        "timeline": ("good", "Captures the visible career-milestone text and the woman on screen."),
        "direct_answer": ("ambiguous", "A plausible specific topic, but the reference expects a generic description of a woman talking."),
        "timeline_answer": ("ambiguous", "Well supported by the on-screen milestones, but mismatched with the generic reference."),
        "frame_answer": ("okay", "Broadly describes the montage, though it is less concrete than the visible text."),
        "overall": "Reference/task interpretation is questionable; do not count this as a clean model failure.",
    },
    "FKsBrmJ2xSg": {
        "timeline": ("good", "Correctly records all three dives and their intervening splashes."),
        "direct_answer": ("wrong", "Selects a later handstand instead of the repeated diving outcome."),
        "timeline_answer": ("wrong", "Returns the immediate splash and misses that the man repeats the action."),
        "frame_answer": ("wrong", "The selected frames contain the repeated dives, but the answer still chooses the splash."),
        "overall": "Clear question-interpretation and temporal-scope failure despite sufficient evidence.",
    },
    "IRTTtnSRTyY": {
        "timeline": ("good", "Correctly identifies repeated pop-it presses and popping sounds across the clip."),
        "direct_answer": ("good", "Concise semantic match to the reference sound."),
        "timeline_answer": ("good", "Correct concise description of the sound."),
        "frame_answer": ("good", "Correct and grounded, although more verbose than necessary."),
        "overall": "Solved by every path; direct native AV is the cleanest answer.",
    },
    "IVqpg_1aWXU": {
        "timeline": ("okay", "Locates the toy airplane scenes but does not transcribe the small text on the plane."),
        "direct_answer": ("good", "Correctly reads SIGMA AIRLINES."),
        "timeline_answer": ("wrong", "Honestly reports that the timeline lacks the text, but does not answer the labeled question."),
        "frame_answer": ("good", "The selected higher-detail frames restore the correct OCR answer."),
        "overall": "A clear case where native visual detail or selected frames add evidence missing from the timeline.",
    },
    "JTolYjiErBw": {
        "timeline": ("good", "Faithfully records digging, picking, and placing the items in a bowl."),
        "direct_answer": ("good", "Matches the reference intent to eat them, though it omits cooking."),
        "timeline_answer": ("wrong", "Answers where the mushrooms are placed, not why they are picked."),
        "frame_answer": ("wrong", "Also substitutes the immediate action for the requested motivation."),
        "overall": "Direct AV succeeds; derived paths confuse an observed action with causal intent.",
    },
    "LSz_xBDBPNc": {
        "timeline": ("okay", "Captures the visual sequence but records only that someone speaks, without the instruction."),
        "direct_answer": ("good", "Drop it is semantically equivalent to the reference, put it down."),
        "timeline_answer": ("wrong", "Invents an instruction to hit it with a hammer."),
        "frame_answer": ("okay", "Gets the core instruction right but is unnecessarily long and adds downstream events."),
        "overall": "Native audio resolves the question; the text timeline loses the decisive speech evidence.",
    },
    "Mj3kccs07fg": {
        "timeline": ("good", "Records the two named foods the woman handles, but does not settle how every visible food grouping should be counted."),
        "direct_answer": ("ambiguous", "Says three; selected frames visibly suggest three groupings, while the reference says two."),
        "timeline_answer": ("ambiguous", "Matches the reference value of two, but may inherit the timeline's narrower counting interpretation."),
        "frame_answer": ("ambiguous", "Says three consistently with the visible groupings but conflicts with the reference."),
        "overall": "Needs human adjudication of what counts as a distinct dish; not a clean correctness example.",
    },
}


def token_f1(prediction: str, reference: str) -> float:
    """Return bag-of-token F1 after NIST text normalization."""

    predicted = normalize_nist_text(prediction).split()
    expected = normalize_nist_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _load_rows(run_root: Path, annotation_root: Path) -> list[dict[str, Any]]:
    freeze = json.loads((run_root / "predictions-freeze.json").read_text())
    if freeze.get("row_count") != 7:
        raise ValueError("Expected a frozen seven-row canary")
    rows = []
    for item in freeze["rows"]:
        complete_path = run_root / "rows" / item["video_id"] / "complete.json"
        if sha256_path(complete_path) != item["complete_sha256"]:
            raise ValueError(f"Frozen prediction changed: {item['video_id']}")
        complete = json.loads(complete_path.read_text())
        annotation_path = annotation_root / f"{item['video_id']}.json"
        annotation = json.loads(annotation_path.read_text())
        if annotation["question"] != complete["question"]:
            raise ValueError(f"Question mismatch: {item['video_id']}")
        rows.append(
            {
                **complete,
                "reference": annotation["correct_answer"],
                "annotation_sha256": sha256_path(annotation_path),
            }
        )
    return rows


def score(
    *, run_root: Path, annotation_root: Path, bert_model: Path
) -> dict[str, Any]:
    """Score all three frozen conditions and return an audit artifact."""

    rows = _load_rows(run_root, annotation_root)
    predictions = [row[condition] for condition in CONDITIONS for row in rows]
    references = [row["reference"] for _condition in CONDITIONS for row in rows]
    bert_scores = _default_bertscore(predictions, references, bert_model)

    scored_rows: list[dict[str, Any]] = []
    index = 0
    for condition in CONDITIONS:
        for row in rows:
            prediction = row[condition]
            reference = row["reference"]
            meteor = _default_meteor(prediction, reference)
            sts = umbc_sts_score(prediction, reference)
            bertscore = float(bert_scores[index])
            index += 1
            metrics = {
                "normalized_exact_match": float(
                    normalize_nist_text(prediction) == normalize_nist_text(reference)
                ),
                "token_f1": token_f1(prediction, reference),
                "meteor": meteor,
                "bertscore_f1": bertscore,
                "sts": sts,
                "semantic_composite": statistics.fmean((meteor, bertscore, sts)),
                "ndcg": 1.0,
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Non-finite metric: {row['video_id']} {condition}")
            scored_rows.append(
                {
                    "video_id": row["video_id"],
                    "category": row["category"],
                    "condition": condition,
                    "prediction": prediction,
                    "reference": reference,
                    "metrics": metrics,
                }
            )

    conditions = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in scored_rows if row["condition"] == condition]
        conditions[condition] = {
            metric: statistics.fmean(row["metrics"][metric] for row in condition_rows)
            for metric in condition_rows[0]["metrics"]
        }

    return {
        "schema_version": 1,
        "panel": {"row_count": len(rows), "conditions": list(CONDITIONS)},
        "conditions": conditions,
        "rows": scored_rows,
        "metric_notes": {
            "semantic_composite": "mean(METEOR, BERTScore-F1, STS)",
            "ndcg": "Exactly 1.0 and non-informative for one submitted answer.",
            "inference": "Descriptive only; seven purposively sampled rows.",
        },
        "integrity": {
            "predictions_frozen_before_labels": True,
            "freeze_sha256": sha256_path(run_root / "predictions-freeze.json"),
            "bert_model_sha256": sha256_path(bert_model),
            "annotation_hashes": {
                row["video_id"]: row["annotation_sha256"] for row in rows
            },
        },
    }


def render_inspection(run_root: Path, artifact: dict[str, Any]) -> str:
    """Render a private, media-linked Markdown inspection index."""

    metrics = {
        (row["video_id"], row["condition"]): row
        for row in artifact["rows"]
    }
    lines = [
        "# Direct Gemini evidence canary",
        "",
        "Seven purposively sampled, labeled videos. Predictions were frozen before references were read.",
        "",
        "| Answer path | METEOR | BERTScore-F1 | STS | Composite |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = artifact["conditions"][condition]
        lines.append(
            f"| {condition} | {summary['meteor']:.3f} | "
            f"{summary['bertscore_f1']:.3f} | {summary['sts']:.3f} | "
            f"{summary['semantic_composite']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Human verdict summary",
            "",
            "`good` means supported and semantically correct; `okay` means useful but incomplete or overlong; `wrong` means it misses the requested answer; `ambiguous` means the reference or counting interpretation needs adjudication.",
            "",
            "| Artifact | Good | Okay | Wrong | Ambiguous |",
            "|---|---:|---:|---:|---:|",
            "| Question-blind timeline | 5 | 2 | 0 | 0 |",
            "| Direct native-AV answer | 4 | 0 | 1 | 2 |",
            "| Timeline-only answer | 1 | 0 | 4 | 2 |",
            "| Timeline + selected-frame answer | 2 | 2 | 2 | 1 |",
        ]
    )

    for complete_path in sorted((run_root / "rows").glob("*/complete.json")):
        row_root = complete_path.parent
        complete = json.loads(complete_path.read_text())
        video_id = complete["video_id"]
        assessment = MANUAL_ASSESSMENTS[video_id]
        direct = metrics[(video_id, "direct_answer")]
        events = json.loads((row_root / "timeline" / "events.json").read_text())
        frames = json.loads(
            (row_root / "selected-frames" / "manifest.json").read_text()
        )
        video = next((row_root / "delivery" / "delivery-media").glob("*.webm"))
        lines.extend(
            [
                "",
                f"## {complete['category']}: {video_id}",
                "",
                f"[Native AV delivery]({video.resolve()}) · "
                f"[selected frames]({(row_root / 'selected-frames').resolve()})",
                "",
                f"**Question:** {complete['question']}",
                "",
                f"**Reference (revealed after freeze):** {direct['reference']}",
                "",
                f"**Overall assessment:** {assessment['overall']}",
                "",
                "| Path | Verdict | Answer | Why | Composite |",
                "|---|---|---|---|---:|",
            ]
        )
        for condition in CONDITIONS:
            scored = metrics[(video_id, condition)]
            answer = scored["prediction"].replace("|", "\\|")
            verdict, reason = assessment[condition]
            lines.append(
                f"| {condition} | **{verdict}** | {answer} | {reason} | "
                f"{scored['metrics']['semantic_composite']:.3f} |"
            )
        timeline_verdict, timeline_reason = assessment["timeline"]
        lines.extend(
            [
                "",
                f"**Question-blind timeline verdict: {timeline_verdict}.** {timeline_reason}",
                "",
                "**Question-blind timeline**",
                "",
            ]
        )
        for event in events:
            lines.append(
                f"- {event['start_seconds']:.3f}–{event['end_seconds']:.3f}: "
                f"{event['description']}"
            )
        lines.extend(["", "**Question-selected spans**", ""])
        for span in complete["selected_spans"]:
            lines.append(
                f"- {span['start_seconds']:.3f}–{span['end_seconds']:.3f}: "
                f"{span['reason']}"
            )
        lines.extend(["", "**Extracted frame timestamps**", ""])
        lines.append(
            ", ".join(f"{frame['timestamp_seconds']:.3f}s" for frame in frames)
        )
        contact_sheet = row_root / "selected-frames" / "contact-sheet.jpg"
        if contact_sheet.exists():
            lines.extend(
                ["", f"![Selected frames]({contact_sheet.resolve()})"]
            )
    lines.extend(
        [
            "",
            "## Audit notes",
            "",
            "- Two text-only selector responses were regenerated after an initial timestamp-parser failure; native-video calls were not repeated. The pre-preservation gaps are recorded in the affected row directories.",
            "- The speech selector's first response hit its output ceiling. That attempt and the one text-only retry are both preserved.",
            "- NDCG is exactly 1.0 and non-informative because each path emits one answer.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspection-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = score(
        run_root=args.run_root,
        annotation_root=args.annotations,
        bert_model=args.bert_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    if args.inspection_output is not None:
        args.inspection_output.parent.mkdir(parents=True, exist_ok=True)
        args.inspection_output.write_text(render_inspection(args.run_root, artifact))
    print(json.dumps({"output": str(args.output), "rows": artifact["panel"]["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
