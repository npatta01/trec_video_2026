#!/usr/bin/env python3
"""Run the fixed Stage 0.5g transcript-to-answer diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_qwen_transcript_answer import (
    QWEN_CONDITION,
    WHISPER_CONDITION,
    bind_answer_annotations,
    build_blind_review_packet,
    freeze_answer_predictions,
    prepare_answer_test,
    record_answer_attempt,
    score_answer_review,
    status_answer_test,
)


PRIVATE_ROOT = Path(
    "/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/"
    "model-verifier-canaries-20260814/stage05g-qwen-transcript-answer-v1"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed 12-row Qwen/Whisper transcript-to-answer diagnostic"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "run",
        "freeze",
        "bind-annotations",
        "review-packet",
        "score",
        "status",
    ):
        commands.add_parser(command)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    now: Callable[[], str] = _utc_now,
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        freeze = prepare_answer_test(PRIVATE_ROOT, prepared_at=now())
        _print({"status": freeze["status"]})
    elif args.command == "run":
        terminal = record_answer_attempt(PRIVATE_ROOT)
        status = status_answer_test(PRIVATE_ROOT)
        _print(
            {
                "status": terminal["status"],
                "exit_code": terminal["exit_code"],
                "completion_count": status.get("completion_count"),
                "required_count": status.get("required_count"),
            }
        )
    elif args.command == "freeze":
        freeze = freeze_answer_predictions(PRIVATE_ROOT)
        _print({"status": freeze["status"], "decision": freeze["decision"]})
    elif args.command == "bind-annotations":
        binding = bind_answer_annotations(PRIVATE_ROOT, bound_at=now())
        _print(
            {
                "status": binding["status"],
                "annotation_count": binding["annotation_count"],
            }
        )
    elif args.command == "review-packet":
        packet = build_blind_review_packet(PRIVATE_ROOT)
        _print({"status": packet["status"], "row_count": packet["row_count"]})
    elif args.command == "score":
        judgment_path = PRIVATE_ROOT / "review-judgments-input.json"
        judgments = json.loads(judgment_path.read_text(encoding="utf-8"))
        score = score_answer_review(PRIVATE_ROOT, judgments, scored_at=now())
        _print(
            {
                "status": score["status"],
                "decision": score["decision"],
                "reviewed_answer_count": score["reviewed_answer_count"],
                "correct": {
                    condition: score["conditions"][condition]["correct"]
                    for condition in (QWEN_CONDITION, WHISPER_CONDITION)
                },
            }
        )
    else:
        status = status_answer_test(PRIVATE_ROOT)
        sanitized = {
            key: status[key]
            for key in (
                "status",
                "decision",
                "completion_count",
                "required_count",
                "annotation_count",
                "review_row_count",
                "reviewed_answer_count",
                "conditions",
                "paired",
                "paid_compute_usd",
            )
            if key in status
        }
        _print(sanitized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
