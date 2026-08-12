#!/usr/bin/env python3
"""Run the seven-row scope-aware direct native-AV control."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from google import genai


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_direct_gemini_evidence_canary import (  # noqa: E402
    FPS,
    MODEL,
    _call,
    _sha256,
    _upload,
    _video_part,
    _write_once,
)


TEMPORAL_SCOPES = {
    "moment",
    "interval",
    "sequence",
    "repetition_pattern",
    "whole_video",
}
TARGETS = {
    "action",
    "state",
    "change",
    "intent_cause",
    "outcome",
    "count",
    "speech",
    "sound",
    "visible_text",
    "topic",
}
_SCOPE = re.compile(
    r"^SCOPE:\s*temporal=(?P<temporal>[^;]+);\s*target=(?P<target>.+)$",
    re.IGNORECASE,
)
_EVIDENCE = re.compile(
    r"^EVIDENCE:\s*(?P<timestamps>[^|]+)\|\s*(?P<support>.+)$",
    re.IGNORECASE,
)
_ANSWER = re.compile(r"^ANSWER:\s*(?P<answer>.+)$", re.IGNORECASE)
_UNCERTAINTY = re.compile(r"^UNCERTAINTY:\s*(?P<uncertainty>.+)$", re.IGNORECASE)


def answer_prompt(question: str) -> str:
    """Return the frozen, general scope-before-answer prompt."""

    return f"""Watch and listen to the complete video from beginning to end. Before answering, identify what the question quantifies over and what semantic target it asks for.

Temporal scope is non-exclusive: moment, interval, sequence, repetition_pattern, whole_video.
Semantic target is non-exclusive: action, state, change, intent_cause, outcome, count, speech, sound, visible_text, topic.

Do not assume that a temporal question asks only for the nearest event. Check whether the wording and complete video support a sequence, repeated pattern, or whole-video outcome. Distinguish an observed intermediate action from a requested motive or cause. For exact speech, sound, counts, or visible text, rely on the native audio and visual evidence. Use only evidence present in the video. Give one concise best-supported answer.

Return plain text only in exactly this form. Use comma-separated values when more than one scope or target applies, and one or more EVIDENCE lines:
SCOPE: temporal=value; target=value
EVIDENCE: timestamp_or_ranges | concise support
ANSWER: one concise answer
UNCERTAINTY: none, or one concise credible ambiguity

Question: {question}"""


def _values(raw: str) -> list[str]:
    return [value.strip().lower() for value in raw.split(",") if value.strip()]


def parse_answer(text: str) -> dict[str, Any]:
    """Parse and validate the scope-aware plain-text response."""

    temporal: list[str] | None = None
    targets: list[str] | None = None
    evidence: list[dict[str, str]] = []
    answer: str | None = None
    uncertainty: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        scope = _SCOPE.match(line)
        if scope:
            temporal = _values(scope.group("temporal"))
            targets = _values(scope.group("target"))
            continue
        support = _EVIDENCE.match(line)
        if support:
            evidence.append(
                {
                    "timestamps": support.group("timestamps").strip(),
                    "support": support.group("support").strip(),
                }
            )
            continue
        selected_answer = _ANSWER.match(line)
        if selected_answer:
            answer = selected_answer.group("answer").strip()
            continue
        selected_uncertainty = _UNCERTAINTY.match(line)
        if selected_uncertainty:
            uncertainty = selected_uncertainty.group("uncertainty").strip()
    if temporal is None or not temporal:
        raise ValueError("response has no temporal scope")
    unknown_temporal = set(temporal) - TEMPORAL_SCOPES
    if unknown_temporal:
        raise ValueError(f"unknown temporal scope: {sorted(unknown_temporal)}")
    if targets is None or not targets:
        raise ValueError("response has no semantic target")
    unknown_targets = set(targets) - TARGETS
    if unknown_targets:
        raise ValueError(f"unknown semantic target: {sorted(unknown_targets)}")
    if not evidence or not answer or not uncertainty:
        raise ValueError("response requires evidence, answer, and uncertainty")
    return {
        "temporal_scope": temporal,
        "targets": targets,
        "evidence": evidence,
        "answer": answer,
        "uncertainty": uncertainty,
    }


def prepare(source_root: Path, output: Path) -> None:
    """Freeze the seven-row panel and generic decision protocol."""

    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if len(source_manifest["rows"]) != 7:
        raise ValueError("scope control requires exactly seven source rows")
    rows = []
    for row in source_manifest["rows"]:
        video_id = row["video_id"]
        media_dir = source_root / "rows" / video_id / "delivery" / "delivery-media"
        media = sorted(media_dir.glob("*.webm"))
        if len(media) != 1:
            raise ValueError(f"expected one frozen WebM for {video_id}")
        rows.append(
            {
                "video_id": video_id,
                "category": row["category"],
                "question": row["question"],
                "duration_seconds": row["duration_seconds"],
                "media_path": str(media[0].resolve()),
                "media_sha256": _sha256(media[0]),
            }
        )
    protocol = {
        "schema_version": 1,
        "experiment_id": "direct-gemini-scope-control-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "mechanism control for scope-aware answering before whole-video synthesis",
        "model": MODEL,
        "video_fps": FPS,
        "media_resolution": "HIGH",
        "thinking_level": "HIGH",
        "temperature": 0.0,
        "seed": 42,
        "condition": "question plus complete native AV plus scope-before-answer prompt",
        "excluded_inputs": [
            "question-blind timeline",
            "whole-video synthesis",
            "selected frames",
            "reference answer",
        ],
        "validity": (
            "label-aware operator mechanism test on spent rows; provider receives no labels; "
            "not a generalization estimate"
        ),
        "decision_gate": {
            "required": [
                "capture repeated-action scope on the known temporal mechanism row",
                "preserve the four previously supported direct-native-AV answers",
            ],
            "reported_separately": [
                "two rows with unresolved reference or task ambiguity"
            ],
            "next_if_pass": "run the matched whole-video-synthesis treatment C",
            "next_if_fail": "whole-video synthesis C is directly justified",
        },
        "prompt_sha256": hashlib.sha256(answer_prompt("{QUESTION}").encode()).hexdigest(),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "rows": rows,
    }
    _write_once(output / "protocol.json", protocol)


def run_row(root: Path, video_id: str) -> None:
    """Run one resumable, one-attempt native-AV scope-aware answer call."""

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    row = next((item for item in protocol["rows"] if item["video_id"] == video_id), None)
    if row is None:
        raise ValueError("video is not in the frozen protocol")
    row_root = root / "rows" / video_id
    complete_path = row_root / "complete.json"
    if complete_path.is_file():
        print(json.dumps({"video_id": video_id, "status": "already-complete"}))
        return
    media = Path(row["media_path"])
    if _sha256(media) != row["media_sha256"]:
        raise ValueError("frozen delivery media identity changed")
    prompt = answer_prompt(row["question"])
    if hashlib.sha256(answer_prompt("{QUESTION}").encode()).hexdigest() != protocol["prompt_sha256"]:
        raise ValueError("prompt template identity changed")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = None
    cleanup = "not-uploaded"
    try:
        uploaded = _upload(client, media)
        response, metadata = _call(client, [_video_part(uploaded), prompt], 4096)
        _write_once(row_root / "request.json", {"question": row["question"], "prompt": prompt})
        _write_once(row_root / "response.txt", response)
        _write_once(row_root / "metadata.json", metadata)
        parsed = parse_answer(response)
        _write_once(row_root / "parsed.json", parsed)
        _write_once(
            complete_path,
            {
                "video_id": video_id,
                "category": row["category"],
                "question": row["question"],
                "answer": parsed["answer"],
                "temporal_scope": parsed["temporal_scope"],
                "targets": parsed["targets"],
                "uncertainty": parsed["uncertainty"],
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            },
        )
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
                cleanup = "succeeded"
            except Exception:
                cleanup = "failed"
            _write_once(row_root / "cleanup.json", {"uploaded_file_deletion": cleanup})
    print(json.dumps({"video_id": video_id, "status": "complete", "cleanup": cleanup}))


def freeze(root: Path) -> None:
    """Freeze all seven parsed predictions before scoring."""

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    for row in protocol["rows"]:
        complete = root / "rows" / row["video_id"] / "complete.json"
        if not complete.is_file():
            raise ValueError(f"incomplete row: {row['video_id']}")
        rows.append(
            {
                "video_id": row["video_id"],
                "complete_sha256": _sha256(complete),
            }
        )
    _write_once(
        root / "predictions-freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "row_count": len(rows),
            "protocol_sha256": _sha256(root / "protocol.json"),
            "rows": rows,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    row_parser = sub.add_parser("run-row")
    row_parser.add_argument("--root", type=Path, required=True)
    row_parser.add_argument("--video-id", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.source_root, args.output)
    elif args.command == "run-row":
        run_row(args.root, args.video_id)
    else:
        freeze(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
