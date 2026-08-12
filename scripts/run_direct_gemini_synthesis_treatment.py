#!/usr/bin/env python3
"""Run grounded whole-video synthesis followed by scope-aware native-AV QA."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from google import genai
from google.genai import types


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_direct_gemini_evidence_canary import (  # noqa: E402
    FPS,
    MODEL,
    _call,
    _metadata,
    _sha256,
    _upload,
    _video_part,
    _write_once,
    parse_timeline,
)
from run_direct_gemini_scope_control import parse_answer  # noqa: E402


SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
    },
    "required": ["start_seconds", "end_seconds"],
}
GROUNDED_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "support": {"type": "array", "items": SPAN_SCHEMA},
    },
    "required": ["description", "support"],
}
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": GROUNDED_CLAIM_SCHEMA,
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                },
                "required": [
                    "label",
                    "description",
                    "start_seconds",
                    "end_seconds",
                ],
            },
        },
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "count": {"type": "integer"},
                    "instances": {"type": "array", "items": SPAN_SCHEMA},
                    "variation": {"type": "string"},
                },
                "required": ["description", "count", "instances", "variation"],
            },
        },
        "state_changes": {"type": "array", "items": GROUNDED_CLAIM_SCHEMA},
        "salient_av": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["speech", "visible_text", "sound"],
                    },
                    "content": {"type": "string"},
                    "support": {"type": "array", "items": SPAN_SCHEMA},
                },
                "required": ["kind", "content", "support"],
            },
        },
        "inferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {"type": "string", "enum": ["goal", "cause", "outcome"]},
                    "support": {"type": "array", "items": SPAN_SCHEMA},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["claim", "kind", "support", "confidence"],
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overview",
        "phases",
        "patterns",
        "state_changes",
        "salient_av",
        "inferences",
        "uncertainties",
    ],
}


def synthesis_prompt(timeline: str) -> str:
    """Return the frozen question-blind whole-video synthesis prompt."""

    return f"""Watch and listen to the complete native video from beginning to end. Using the frozen question-blind event map below as navigation, create a concise grounded whole-video synthesis. Do not answer any question.

Represent the video's overall activity, beginning/middle/end phases, repeated or multi-scene patterns, important state changes, and salient speech, visible text, and sounds. Enumerate every claimed repeated pattern with all observed instance spans and a count. Separate observation from inference: goals, causes, and outcomes may appear only in the inferences field with confidence and timestamp support. If no pattern, state change, salient AV item, or inference is supported, return an empty list. Flag uncertainty instead of guessing. The native video is authoritative when it conflicts with the map. Keep the complete response concise.

Frozen question-blind event map:
{timeline}"""


def answer_prompt(question: str, timeline: str, synthesis_json: str) -> str:
    """Return the frozen C answer prompt over AV, map, and synthesis."""

    return f"""Answer the question using the complete native video/audio, the frozen event map, and the grounded whole-video synthesis. The native video is authoritative; the text artifacts are navigation and structure, not replacements for visual or audio evidence.

First identify the non-exclusive temporal scope: moment, interval, sequence, repetition_pattern, whole_video. Then identify the non-exclusive semantic target: action, state, change, intent_cause, outcome, count, speech, sound, visible_text, topic. Do not assume a temporal question asks only for the nearest event. Use the whole-video synthesis when the answer concerns repetition, overall activity, change across phases, intent, cause, or outcome. For exact speech, sound, counts, or visible text, verify against native AV. Give one concise best-supported answer.

Return plain text only in exactly this form, with one or more EVIDENCE lines:
SCOPE: temporal=value; target=value
EVIDENCE: timestamp_or_ranges | concise support
ANSWER: one concise answer
UNCERTAINTY: none, or one concise credible ambiguity

Question: {question}

Frozen event map:
{timeline}

Question-blind whole-video synthesis:
{synthesis_json}"""


def _call_json(
    client: Any, contents: Any
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    started = time.monotonic()
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            seed=42,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=SYNTHESIS_SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
                include_thoughts=False,
            ),
        ),
    )
    raw = response.text or ""
    payload = json.loads(raw)
    if sorted(payload) != sorted(SYNTHESIS_SCHEMA["required"]):
        raise ValueError("synthesis response does not match required top-level fields")
    return payload, raw, _metadata(response, time.monotonic() - started)


def prepare(source_root: Path, output: Path) -> None:
    """Freeze source media, timelines, prompts, and the C decision gate."""

    source_manifest = json.loads((source_root / "manifest.json").read_text())
    rows = []
    for row in source_manifest["rows"]:
        video_id = row["video_id"]
        row_root = source_root / "rows" / video_id
        media = sorted((row_root / "delivery" / "delivery-media").glob("*.webm"))
        timeline_path = row_root / "timeline" / "response.txt"
        if len(media) != 1 or not timeline_path.is_file():
            raise ValueError(f"missing frozen C input for {video_id}")
        parse_timeline(timeline_path.read_text(), float(row["duration_seconds"]))
        rows.append(
            {
                "video_id": video_id,
                "category": row["category"],
                "question": row["question"],
                "duration_seconds": row["duration_seconds"],
                "media_path": str(media[0].resolve()),
                "media_sha256": _sha256(media[0]),
                "timeline_path": str(timeline_path.resolve()),
                "timeline_sha256": _sha256(timeline_path),
            }
        )
    protocol = {
        "schema_version": 1,
        "experiment_id": "direct-gemini-whole-video-synthesis-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "settings": {
            "video_fps": FPS,
            "media_resolution": "HIGH",
            "thinking_level": "HIGH",
            "temperature": 0.0,
            "seed": 42,
        },
        "condition": "question plus native AV plus frozen map plus question-blind whole-video synthesis",
        "synthesis_question_blind": True,
        "label_isolation": "provider calls receive no labels or reference answers",
        "validity": "mechanism treatment on seven spent rows; not a generalization estimate",
        "decision_gate": {
            "required": [
                "repair repeated-action scope beyond B",
                "preserve the four supported direct-native-AV answers",
                "all synthesis observation claims timestamp-grounded",
            ]
        },
        "synthesis_prompt_sha256": hashlib.sha256(
            synthesis_prompt("{TIMELINE}").encode()
        ).hexdigest(),
        "answer_prompt_sha256": hashlib.sha256(
            answer_prompt("{QUESTION}", "{TIMELINE}", "{SYNTHESIS}").encode()
        ).hexdigest(),
        "schema_sha256": hashlib.sha256(
            json.dumps(SYNTHESIS_SCHEMA, sort_keys=True).encode()
        ).hexdigest(),
        "rows": rows,
    }
    _write_once(output / "protocol.json", protocol)


def run_row(root: Path, video_id: str) -> None:
    """Run one synthesis and one answer call, reusing one uploaded video."""

    protocol = json.loads((root / "protocol.json").read_text())
    row = next(
        (item for item in protocol["rows"] if item["video_id"] == video_id), None
    )
    if row is None:
        raise ValueError("video is not in the frozen protocol")
    row_root = root / "rows" / video_id
    if (row_root / "complete.json").is_file():
        print(json.dumps({"video_id": video_id, "status": "already-complete"}))
        return
    media = Path(row["media_path"])
    timeline_path = Path(row["timeline_path"])
    if (
        _sha256(media) != row["media_sha256"]
        or _sha256(timeline_path) != row["timeline_sha256"]
    ):
        raise ValueError("frozen C input identity changed")
    timeline = timeline_path.read_text()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded = None
    cleanup = "not-uploaded"
    try:
        uploaded = _upload(client, media)
        video = _video_part(uploaded)
        synth_prompt = synthesis_prompt(timeline)
        synthesis, synthesis_raw, synthesis_metadata = _call_json(
            client, [video, synth_prompt]
        )
        _write_once(row_root / "synthesis" / "request.json", {"prompt": synth_prompt})
        _write_once(row_root / "synthesis" / "response.json", synthesis)
        _write_once(row_root / "synthesis" / "raw-response.txt", synthesis_raw)
        _write_once(row_root / "synthesis" / "metadata.json", synthesis_metadata)

        synthesis_json = json.dumps(synthesis, sort_keys=True)
        qa_prompt = answer_prompt(row["question"], timeline, synthesis_json)
        response, answer_metadata = _call(client, [video, qa_prompt], 4096)
        _write_once(
            row_root / "answer" / "request.json",
            {"question": row["question"], "prompt": qa_prompt},
        )
        _write_once(row_root / "answer" / "response.txt", response)
        _write_once(row_root / "answer" / "metadata.json", answer_metadata)
        parsed = parse_answer(response)
        _write_once(row_root / "answer" / "parsed.json", parsed)
        _write_once(
            row_root / "complete.json",
            {
                "video_id": video_id,
                "category": row["category"],
                "question": row["question"],
                "answer": parsed["answer"],
                "temporal_scope": parsed["temporal_scope"],
                "targets": parsed["targets"],
                "uncertainty": parsed["uncertainty"],
                "synthesis_sha256": _sha256(row_root / "synthesis" / "response.json"),
                "answer_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            },
        )
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
                cleanup = "succeeded"
            except Exception:
                cleanup = "failed"
            cleanup_path = row_root / "cleanup.json"
            if cleanup_path.exists():
                attempt = 2
                while (row_root / f"cleanup-attempt-{attempt}.json").exists():
                    attempt += 1
                cleanup_path = row_root / f"cleanup-attempt-{attempt}.json"
            _write_once(cleanup_path, {"uploaded_file_deletion": cleanup})
    print(json.dumps({"video_id": video_id, "status": "complete", "cleanup": cleanup}))


def freeze(root: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for row in protocol["rows"]:
        complete = root / "rows" / row["video_id"] / "complete.json"
        if not complete.is_file():
            raise ValueError(f"incomplete row: {row['video_id']}")
        rows.append({"video_id": row["video_id"], "complete_sha256": _sha256(complete)})
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
