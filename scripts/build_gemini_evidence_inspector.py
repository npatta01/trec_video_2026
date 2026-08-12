#!/usr/bin/env python3
"""Build a private static inspector for the seven-row Gemini evidence canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from score_direct_gemini_evidence_canary import (  # noqa: E402
    CONDITIONS,
    MANUAL_ASSESSMENTS,
)


ANSWER_LABELS = {
    "direct_answer": "Direct native AV",
    "timeline_answer": "Timeline only",
    "frame_answer": "Timeline + selected frames",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _media_url(prefix: str, *parts: str) -> str:
    if not prefix.startswith("/") or "://" in prefix or ".." in PurePosixPath(prefix).parts:
        raise ValueError("media prefix must be a same-origin absolute URL path")
    return str(PurePosixPath(prefix, *parts))


def load_cases(run_root: Path, media_prefix: str) -> list[dict[str, Any]]:
    """Validate and project the frozen private artifacts for the inspector."""

    metric_artifact = _read_json(run_root / "postfreeze-metrics.json")
    metric_rows = {
        (row["video_id"], row["condition"]): row for row in metric_artifact["rows"]
    }
    row_ids = {path.parent.name for path in (run_root / "rows").glob("*/complete.json")}
    expected_ids = set(MANUAL_ASSESSMENTS)
    if row_ids != expected_ids:
        raise ValueError(
            f"Expected exactly seven frozen rows; missing={sorted(expected_ids-row_ids)}, "
            f"extra={sorted(row_ids-expected_ids)}"
        )

    cases = []
    for video_id in sorted(expected_ids):
        row_root = run_root / "rows" / video_id
        complete = _read_json(row_root / "complete.json")
        timeline = _read_json(row_root / "timeline" / "events.json")
        frame_manifest = _read_json(row_root / "selected-frames" / "manifest.json")
        videos = sorted((row_root / "delivery" / "delivery-media").glob("*.webm"))
        if len(videos) != 1:
            raise ValueError(f"Missing required artifact: one WebM for {video_id}")
        assessment = MANUAL_ASSESSMENTS[video_id]
        answers = []
        for condition in CONDITIONS:
            scored = metric_rows.get((video_id, condition))
            if scored is None:
                raise ValueError(f"Missing required artifact: metric {video_id}/{condition}")
            verdict, reason = assessment[condition]
            answers.append(
                {
                    "condition": condition,
                    "label": ANSWER_LABELS[condition],
                    "answer": scored["prediction"],
                    "verdict": verdict,
                    "reason": reason,
                    "semantic_composite": scored["metrics"]["semantic_composite"],
                }
            )
        frames = []
        for frame in frame_manifest:
            frame_path = row_root / "selected-frames" / frame["path"]
            if not frame_path.is_file():
                raise ValueError(f"Missing required artifact: {frame_path}")
            frames.append(
                {
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "url": _media_url(
                        media_prefix,
                        "rows",
                        video_id,
                        "selected-frames",
                        frame["path"],
                    ),
                }
            )
        cases.append(
            {
                "video_id": video_id,
                "category": complete["category"],
                "question": complete["question"],
                "reference": metric_rows[(video_id, CONDITIONS[0])]["reference"],
                "overall": assessment["overall"],
                "timeline_verdict": assessment["timeline"][0],
                "timeline_reason": assessment["timeline"][1],
                "answers": answers,
                "timeline": timeline,
                "selected_spans": complete["selected_spans"],
                "frames": frames,
                "video_url": _media_url(
                    media_prefix,
                    "rows",
                    video_id,
                    "delivery",
                    "delivery-media",
                    videos[0].name,
                ),
            }
        )
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--media-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_cases(args.run_root, args.media_prefix)
    raise NotImplementedError("HTML rendering is implemented in Task 2")


if __name__ == "__main__":
    raise SystemExit(main())
