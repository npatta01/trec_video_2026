"""Private, question-conditioned Nemotron event-map inference core.

The module deliberately keeps the model response boundary small.  Labels,
references, transcripts, metrics, and model reasoning are not part of the
request contract; event summaries are retained only in the private workload
payload and are never exposed through a downstream answer API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
MODEL_REVISION = "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
RUNTIME_IMAGE_DIGEST = (
    "sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
)
RUNTIME_IMAGE = f"docker.io/vllm/vllm-openai:v0.20.0@{RUNTIME_IMAGE_DIGEST}"
CANONICAL_PANEL_IDS_SHA256 = (
    "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
)
EXPECTED_ROW_COUNT = 475
SHARD_COUNTS = (119, 119, 119, 118)
MAX_OUTPUT_TOKENS = 768
PLAINTEXT_MAP_MAX_TOKENS = 256
PLAINTEXT_SELECTOR_MAX_TOKENS = 64
PLAINTEXT_STOP = ["\nEND"]
MAX_EVENTS = 6
MAX_SELECTED_INTERVALS = 2
MAX_SELECTED_SECONDS = 6.0
COVERAGE_TOLERANCE_SECONDS = 0.25
VIDEO_FPS = 1
VIDEO_MAX_FRAMES = 64
VIDEO_PRUNING_RATE = 0.5
KV_CACHE_DTYPE = "fp8"
REASONING_PARSER = "nemotron_v3"
CONDITION_ORDER = ("v-map", "av-map")
CANARY_STRATA = ("speech", "sound", "ocr_detail", "temporal")
EVENT_MAP_PROMPT = (
    "Watch the complete local video and answer the question. Return only the "
    "literal JSON object required by the schema. Do not think aloud, explain, "
    "write markdown, include a transcript, or include references, labels, "
    "metrics, or reasoning. Your answer must be at most 20 words. Build one "
    "to six chronological non-overlapping events that cover the measured clip. "
    "Select one or two short intervals inside event boundaries; their total "
    "duration must not exceed 6 seconds. Event summaries are private evidence."
)
EVENT_MAP_RETRY_PROMPT = (
    "Return exactly one JSON object and no other bytes. Use exactly these keys: "
    "answer, abstain, confidence, events, selected_intervals. Do not use a "
    "markdown fence, thinking tags, analysis, reasoning, transcript, label, "
    "reference, score, or metric. The answer is at most 20 words. Events must "
    "cover the clip, be chronological and non-overlapping, and selections must "
    "be inside their event with total duration at most 6 seconds."
)

EVENT_MAP_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "abstain", "confidence", "events", "selected_intervals"],
    "properties": {
        "answer": {"type": "string", "minLength": 1, "maxLength": 240},
        "abstain": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EVENTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "event_id",
                    "start",
                    "end",
                    "visual_summary",
                    "audio_summary",
                ],
                "properties": {
                    "event_id": {"type": "string", "pattern": r"^event-[A-Za-z0-9_-]+$"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "visual_summary": {"type": "string", "minLength": 1},
                    "audio_summary": {"type": ["string", "null"]},
                },
            },
        },
        "selected_intervals": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SELECTED_INTERVALS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event_id", "start", "end"],
                "properties": {
                    "event_id": {"type": "string", "pattern": r"^event-[A-Za-z0-9_-]+$"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
            },
        },
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVENT_ID = re.compile(r"^event-[A-Za-z0-9_-]{1,63}$")
_MARKDOWN = re.compile(r"```|<\s*/?\s*think\b", re.IGNORECASE)
_FORBIDDEN_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "answerkey",
        "correctanswer",
        "distractor",
        "expectedanswer",
        "groundtruth",
        "label",
        "labels",
        "reference",
        "referenceanswer",
        "metric",
        "metrics",
        "score",
        "scores",
        "f1",
        "meteor",
        "bertscore",
        "sts",
        "ndcg",
        "reasoning",
        "analysis",
        "rationale",
        "explanation",
        "chainofthought",
        "thinking",
        "transcript",
        "whisper",
        "prediction",
        "predictions",
    }
)
_SPEECH_WORDS = re.compile(r"\b(say|says|said|speaks?|spoken|ask|asks|question|tell|tells)\b")
_SOUND_WORDS = re.compile(r"\b(sound|hear|heard|music|song|noise|tone|instrument|sing)\w*\b")
_OCR_WORDS = re.compile(r"\b(written|text|sign|color|colour|brand|label|read)\w*\b")
_TEMPORAL_WORDS = re.compile(
    r"\b(after|before|then|first|last|when|end|begin|next|while|change|happen)\w*\b"
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ids = [row.get("video_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("video IDs must be nonempty text")
    if len(set(ids)) != len(ids):
        raise ValueError("video IDs must be unique")
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def question_stratum(question: str) -> str:
    """Classify a question with deterministic lexical rules only."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    text = question.casefold()
    if _OCR_WORDS.search(text):
        return "ocr_detail"
    explicit_temporal = re.search(
        r"\b(after|before|then|first|last|when|next|while|change|happen)\w*\b", text
    )
    if _SPEECH_WORDS.search(text) and not explicit_temporal:
        return "speech"
    if _SOUND_WORDS.search(text) and not explicit_temporal:
        return "sound"
    if _TEMPORAL_WORDS.search(text):
        return "temporal"
    if re.search(r"\b(how many|count|number of)\b", text):
        return "counting"
    if re.search(r"\b(why|cause|causes|reason)\b", text):
        return "causal"
    return "other"


def select_canary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select one row from each predeclared question-only canary stratum."""

    selected: list[dict[str, Any]] = []
    for stratum in CANARY_STRATA:
        for row in rows:
            question = row.get("question")
            if question_stratum(question) == stratum:
                selected.append(
                    {
                        key: row[key]
                        for key in (
                            "video_id",
                            "question",
                            "question_stratum",
                            "media_path",
                            "media_sha256",
                            "duration_seconds",
                        )
                        if key in row
                    }
                    | {"question_stratum": stratum}
                )
                break
        else:
            raise ValueError(f"canary stratum {stratum} has no row")
    return selected


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _reject_forbidden_fields(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if normalized in _FORBIDDEN_KEYS or any(
                token in normalized for token in ("reasoning", "groundtruth", "reference")
            ):
                raise ValueError(f"forbidden field at {location}.{key}")
            _reject_forbidden_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_forbidden_fields(nested, location=f"{location}[{index}]")


def _reject_markdown(value: Any, *, location: str = "$") -> None:
    if isinstance(value, str) and _MARKDOWN.search(value):
        raise ValueError(f"markdown or thinking content at {location}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_markdown(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_markdown(nested, location=f"{location}[{index}]")


def validate_private_response(
    value: Mapping[str, Any], *, duration_seconds: float
) -> dict[str, Any]:
    """Validate and return the exact private event-map JSON shape."""

    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(value, Mapping):
        raise ValueError("response must be an object")
    _reject_forbidden_fields(value)
    _reject_markdown(value)
    expected = {"answer", "abstain", "confidence", "events", "selected_intervals"}
    if set(value) != expected:
        raise ValueError("response has extra keys")
    answer = value.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be nonempty")
    if len(answer.split()) > 20:
        raise ValueError("answer exceeds 20 words")
    if type(value.get("abstain")) is not bool:
        raise ValueError("abstain must be boolean")
    confidence = _finite(value.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    events = value.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError("events must contain one to six events")
    seen_ids: set[str] = set()
    normalized_events: list[dict[str, Any]] = []
    previous_end = -math.inf
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {
            "event_id",
            "start",
            "end",
            "visual_summary",
            "audio_summary",
        }:
            raise ValueError(f"event {index} has extra keys")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or _EVENT_ID.fullmatch(event_id) is None:
            raise ValueError("event_id must be opaque")
        if event_id in seen_ids:
            raise ValueError("event_id must be unique")
        seen_ids.add(event_id)
        start = _finite(event.get("start"), f"events[{index}].start")
        end = _finite(event.get("end"), f"events[{index}].end")
        if start >= end:
            raise ValueError("event start must precede end")
        if start < -COVERAGE_TOLERANCE_SECONDS or end > duration + COVERAGE_TOLERANCE_SECONDS:
            raise ValueError("event is outside measured clip")
        if start < previous_end - 1e-9:
            raise ValueError("events must be chronological and non-overlapping")
        if index > 0 and start > previous_end + COVERAGE_TOLERANCE_SECONDS:
            raise ValueError("events contain a gap and do not cover the clip")
        visual = event.get("visual_summary")
        audio = event.get("audio_summary")
        if not isinstance(visual, str) or not visual.strip():
            raise ValueError("visual_summary must be nonempty")
        if audio is not None and (not isinstance(audio, str) or not audio.strip()):
            raise ValueError("audio_summary must be text or null")
        normalized_events.append(
            {
                "event_id": event_id,
                "start": start,
                "end": end,
                "visual_summary": visual.strip(),
                "audio_summary": audio.strip() if isinstance(audio, str) else None,
            }
        )
        previous_end = end
    if normalized_events[0]["start"] > COVERAGE_TOLERANCE_SECONDS:
        raise ValueError("events do not cover clip start")
    if normalized_events[-1]["end"] < duration - COVERAGE_TOLERANCE_SECONDS:
        raise ValueError("events do not cover clip end")

    intervals = value.get("selected_intervals")
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= MAX_SELECTED_INTERVALS:
        raise ValueError("selected_intervals must contain one or two intervals")
    events_by_id = {event["event_id"]: event for event in normalized_events}
    normalized_intervals: list[dict[str, Any]] = []
    total = 0.0
    prior_end = -math.inf
    for index, interval in enumerate(intervals):
        if not isinstance(interval, Mapping) or set(interval) != {"event_id", "start", "end"}:
            raise ValueError(f"selected interval {index} has extra keys")
        event_id = interval.get("event_id")
        if event_id not in events_by_id:
            raise ValueError("selected interval event_id is unknown")
        start = _finite(interval.get("start"), f"selected_intervals[{index}].start")
        end = _finite(interval.get("end"), f"selected_intervals[{index}].end")
        if not 0.0 <= start < end <= duration:
            raise ValueError("selected interval is outside clip")
        event = events_by_id[event_id]
        if start < event["start"] or end > event["end"]:
            raise ValueError("selected interval is not inside event")
        if start < prior_end:
            raise ValueError(
                "selected intervals must be chronological and non-overlapping"
            )
        prior_end = end
        total += end - start
        normalized_intervals.append({"event_id": event_id, "start": start, "end": end})
    if total > MAX_SELECTED_SECONDS + 1e-9:
        raise ValueError("selected interval duration exceeds 6 seconds")
    return {
        "answer": answer.strip(),
        "abstain": value["abstain"],
        "confidence": confidence,
        "events": normalized_events,
        "selected_intervals": normalized_intervals,
    }


def _response_content(payload: Mapping[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("completion choices are incomplete")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("completion finish reason is incomplete")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("completion content is incomplete")
    return message["content"], "stop"


def _plaintext_lines(payload: Mapping[str, Any], *, kind: str) -> list[str]:
    content, _finish_reason = _response_content(payload)
    if _MARKDOWN.search(content):
        raise ValueError(f"{kind} contains markdown or thinking content")
    lines = content.strip().splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{kind} protocol is empty or has blank lines")
    if lines[-1].strip() == "END":
        lines.pop()
    if not lines or any(line.strip() == "END" for line in lines):
        raise ValueError(f"{kind} has an invalid END marker")
    return lines


def build_plaintext_map_request(
    *,
    media_path: Path,
    question: str,
    duration_seconds: float,
    audio_enabled: bool,
) -> dict[str, Any]:
    """Build the no-grammar full-video timeline request."""

    if not media_path.is_absolute():
        raise ValueError("media_path must be absolute")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    text = (
        "Watch the complete video, including audio, from beginning to end.\n\n"
        "For the question below, produce a compact timeline and a direct answer.\n"
        "Return only this line format:\n\n"
        "E1|<representative timestamp in seconds>|<factual description, at most 12 words>\n"
        "E2|<representative timestamp in seconds>|<factual description, at most 12 words>\n"
        "...\n"
        "ANSWER|<answer, at most 20 words>\n"
        "END\n\n"
        "Requirements:\n"
        "- Produce 2–6 event lines.\n"
        "- IDs must be E1, E2, ... in order.\n"
        "- Timestamps must be strictly increasing and between 0 and D.\n"
        "- Include the opening and ending portions of the video.\n"
        "- Timestamps are representative evidence points, not intervals.\n"
        "- No JSON, markdown, explanation, or additional lines.\n\n"
        f"Duration: {duration:.3f} seconds\n"
        f"Question: {question.strip()}"
    )
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": media_path.resolve().as_uri()}},
                    {"type": "text", "text": text},
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": PLAINTEXT_MAP_MAX_TOKENS,
        "stop": list(PLAINTEXT_STOP),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": bool(audio_enabled)},
    }


def build_plaintext_selector_request(
    *, question: str, frozen_map: str
) -> dict[str, Any]:
    """Build the text-only selector request over a frozen parsed map."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if not isinstance(frozen_map, str) or not frozen_map.strip():
        raise ValueError("frozen_map must be nonempty")
    text = (
        "Select the one or two mapped events that contain the strongest evidence.\n\n"
        "Return only:\n\n"
        "SELECT|E#\n"
        "ANSWER|<answer, at most 20 words>\n"
        "END\n\n"
        "For two events use SELECT|E#,E#. Use only IDs in the supplied map. "
        "No explanation.\n\n"
        f"Question: {question.strip()}\n"
        f"Frozen map:\n{frozen_map.strip()}"
    )
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": PLAINTEXT_SELECTOR_MAX_TOKENS,
        "stop": list(PLAINTEXT_STOP),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_plaintext_map(
    payload: Mapping[str, Any], *, duration_seconds: float
) -> dict[str, Any]:
    """Parse a bounded timeline without reconstructing omitted model content."""

    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    lines = _plaintext_lines(payload, kind="map")
    answer_line_index = next(
        (index for index, line in enumerate(lines) if line.startswith("ANSWER|")),
        None,
    )
    if answer_line_index is None or answer_line_index != len(lines) - 1:
        raise ValueError("map must end with ANSWER")
    answer = lines[-1][len("ANSWER|") :].strip()
    if not answer or len(answer.split()) > 20:
        raise ValueError("map answer must be at most 20 words")

    event_lines = lines[:-1]
    if not 2 <= len(event_lines) <= MAX_EVENTS:
        raise ValueError("map must contain 2–6 event lines")
    events: list[dict[str, Any]] = []
    previous_timestamp = -math.inf
    for index, line in enumerate(event_lines, start=1):
        parts = line.split("|", 2)
        if len(parts) != 3 or parts[0] != f"E{index}":
            raise ValueError("map event line IDs or fields are invalid")
        try:
            timestamp = float(parts[1])
        except ValueError as error:
            raise ValueError("map timestamp is not finite") from error
        if not math.isfinite(timestamp) or not 0.0 <= timestamp <= duration:
            raise ValueError("map timestamp is outside clip")
        if timestamp <= previous_timestamp:
            raise ValueError("map timestamps must be strictly increasing")
        description = parts[2].strip()
        if not description or "|" in description:
            raise ValueError("map event description is invalid")
        if len(description.split()) > 12:
            raise ValueError("map event description must be at most 12 words")
        events.append(
            {"event_id": f"E{index}", "timestamp": timestamp, "description": description}
        )
        previous_timestamp = timestamp
    if events[0]["timestamp"] > duration * 0.25:
        raise ValueError("map does not include the opening portion")
    if events[-1]["timestamp"] < duration * 0.75:
        raise ValueError("map does not include the ending portion")
    return {
        "answer": answer,
        "events": events,
        "map_text": "\n".join(
            [
                *(f'{event["event_id"]}|{event["timestamp"]:.6f}|{event["description"]}' for event in events),
                f"ANSWER|{answer}",
            ]
        ),
    }


def parse_plaintext_selector(
    payload: Mapping[str, Any], *, event_ids: set[str]
) -> dict[str, Any]:
    """Parse a selector over already validated event IDs."""

    if not event_ids:
        raise ValueError("event_ids must not be empty")
    lines = _plaintext_lines(payload, kind="selector")
    if len(lines) != 2 or not lines[0].startswith("SELECT|") or not lines[1].startswith("ANSWER|"):
        raise ValueError("selector protocol must contain SELECT and ANSWER lines")
    selected = [value.strip() for value in lines[0][len("SELECT|") :].split(",")]
    if len(selected) not in (1, 2) or len(set(selected)) != len(selected):
        raise ValueError("selector must contain one or two unique IDs")
    if any(value not in event_ids for value in selected):
        raise ValueError("selector references unknown event")
    answer = lines[1][len("ANSWER|") :].strip()
    if not answer or len(answer.split()) > 20:
        raise ValueError("selector answer must be at most 20 words")
    return {"selected_event_ids": selected, "answer": answer}


def derive_anchor_cells(
    events: Sequence[Mapping[str, Any]], *, duration_seconds: float
) -> list[dict[str, Any]]:
    """Turn representative anchors into an exact, non-overlapping timeline."""

    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0 or not events:
        raise ValueError("duration_seconds and events must be nonempty")
    timestamps = [_finite(event.get("timestamp"), "event timestamp") for event in events]
    if any(timestamp < 0 or timestamp > duration for timestamp in timestamps):
        raise ValueError("event timestamp is outside clip")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("event timestamps must be strictly increasing")
    boundaries = [0.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(timestamps, timestamps[1:]))
    boundaries.append(duration)
    return [
        {"event_id": event["event_id"], "start": boundaries[index], "end": boundaries[index + 1]}
        for index, event in enumerate(events)
    ]


def expand_selected_anchors(
    events: Sequence[Mapping[str, Any]],
    selected_event_ids: Sequence[str],
    *,
    duration_seconds: float,
    window_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """Expand anchors into centered windows with deterministic edge handling."""

    duration = _finite(duration_seconds, "duration_seconds")
    window = _finite(window_seconds, "window_seconds")
    if duration <= 0 or window <= 0:
        raise ValueError("duration and window must be positive")
    if not 1 <= len(selected_event_ids) <= MAX_SELECTED_INTERVALS:
        raise ValueError("selected event count exceeds budget")
    by_id = {event.get("event_id"): event for event in events}
    if len(by_id) != len(events) or any(event_id not in by_id for event_id in selected_event_ids):
        raise ValueError("selected event ID is unknown")
    if len(set(selected_event_ids)) != len(selected_event_ids):
        raise ValueError("selected event IDs must be unique")
    selected = sorted(
        (by_id[event_id] for event_id in selected_event_ids),
        key=lambda event: _finite(event.get("timestamp"), "event timestamp"),
    )
    actual_window = min(window, duration)
    windows: list[dict[str, Any]] = []
    prior_end = -math.inf
    for event in selected:
        timestamp = _finite(event.get("timestamp"), "event timestamp")
        start = min(max(0.0, timestamp - actual_window / 2.0), duration - actual_window)
        end = start + actual_window
        if start < prior_end:
            raise ValueError("selected evidence windows overlap")
        windows.append({"event_id": event["event_id"], "start": start, "end": end})
        prior_end = end
    if sum(window["end"] - window["start"] for window in windows) > MAX_SELECTED_SECONDS:
        raise ValueError("selected evidence exceeds 6 seconds")
    return windows


def parse_completion(payload: Mapping[str, Any], *, duration_seconds: float) -> dict[str, Any]:
    content, _finish_reason = _response_content(payload)
    if _MARKDOWN.search(content):
        raise ValueError("markdown or thinking content")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("completion is not JSON") from error
    if not isinstance(decoded, Mapping):
        raise ValueError("completion JSON must be an object")
    return validate_private_response(decoded, duration_seconds=duration_seconds)


def build_chat_request(
    *,
    media_path: Path,
    question: str,
    duration_seconds: float,
    audio_enabled: bool,
    response_format_supported: bool,
    retry: bool = False,
) -> dict[str, Any]:
    if not media_path.is_absolute():
        raise ValueError("media_path must be absolute")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    request: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": media_path.resolve().as_uri()}},
                    {
                        "type": "text",
                        "text": (EVENT_MAP_RETRY_PROMPT if retry else EVENT_MAP_PROMPT)
                        + f"\nMeasured clip duration: {duration:.3f} seconds. "
                        + f"Cover continuously from 0.000 through {duration:.3f} seconds."
                        + "\nQuestion: "
                        + question.strip(),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {"use_audio_in_video": bool(audio_enabled)},
    }
    if response_format_supported:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "nemotron_event_map",
                "strict": True,
                "schema": EVENT_MAP_RESPONSE_SCHEMA,
            },
        }
    return request


def post_json(url: str, payload: Mapping[str, Any], *, timeout_seconds: float = 300.0) -> dict[str, Any]:
    body = canonical_json(payload)
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("vLLM response must be an object")
    return decoded


def run_one_request(
    *,
    request: Mapping[str, Any],
    duration_seconds: float,
    post_json: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    media_path: Path,
    audio_enabled: bool,
    response_format_supported: bool,
    elapsed_clock: Callable[[], float] = time.monotonic,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = elapsed_clock()
    base_request = build_chat_request(
        media_path=media_path,
        question=str(request.get("question", "")),
        duration_seconds=duration_seconds,
        audio_enabled=audio_enabled,
        response_format_supported=response_format_supported,
    )
    primary_request_sha256 = canonical_sha256(base_request)
    result: dict[str, Any] = {
        "schema_version": 1,
        "video_id": request.get("video_id"),
        "condition_id": "av-map" if audio_enabled else "v-map",
        "request_sha256": primary_request_sha256,
        "primary_request_sha256": primary_request_sha256,
        "retry_request_sha256": None,
        "primary_valid": False,
        "retry_valid": None,
        "retry_used": False,
        "status": "failed",
        "error_type": "schema_invalid",
        "response_sha256": None,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "memory_telemetry": {
            "path": "telemetry/nvidia-smi.csv",
            "sampling_interval_seconds": 5,
        },
    }
    if runtime_identity is not None:
        result["runtime_identity"] = dict(runtime_identity)
    try:
        primary_payload = post_json(base_request)
        parsed = parse_completion(primary_payload, duration_seconds=duration_seconds)
    except ValueError:
        result["retry_used"] = True
        result["retry_valid"] = False
        retry_request = build_chat_request(
            media_path=media_path,
            question=str(request.get("question", "")),
            duration_seconds=duration_seconds,
            audio_enabled=audio_enabled,
            response_format_supported=response_format_supported,
            retry=True,
        )
        retry_request_sha256 = canonical_sha256(retry_request)
        result["retry_request_sha256"] = retry_request_sha256
        result["request_sha256"] = retry_request_sha256
        try:
            retry_payload = post_json(retry_request)
            parsed = parse_completion(retry_payload, duration_seconds=duration_seconds)
        except ValueError:
            finished = elapsed_clock()
            result["elapsed_seconds"] = max(0.0, finished - started)
            return result
        except Exception:
            finished = elapsed_clock()
            result["error_type"] = "request_failed"
            result["elapsed_seconds"] = max(0.0, finished - started)
            return result
        result["retry_valid"] = True
        response_payload = retry_payload
    except Exception:
        finished = elapsed_clock()
        result["error_type"] = "request_failed"
        result["elapsed_seconds"] = max(0.0, finished - started)
        return result
    else:
        result["primary_valid"] = True
        result["retry_valid"] = None
        response_payload = primary_payload

    finished = elapsed_clock()
    result.update(parsed)
    result.update(
        {
            "status": "passed",
            "error_type": None,
            "response_sha256": canonical_sha256(response_payload),
            "elapsed_seconds": max(0.0, finished - started),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_image": RUNTIME_IMAGE,
        }
    )
    return result


def validate_gpu_identity(name: str, memory_mib: int) -> dict[str, Any]:
    if name not in {"L40S", "NVIDIA L40S"}:
        raise ValueError("GPU must be L40S")
    if isinstance(memory_mib, bool) or not isinstance(memory_mib, int) or memory_mib < 45_000:
        raise ValueError("L40S GPU reported memory is below the safe floor")
    return {"gpu_name": "L40S", "gpu_memory_mib": memory_mib}


def probe_media(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("ffprobe output must be an object")
    return value


def _duration_from_probe(probe: Mapping[str, Any], field: str) -> float:
    form = probe.get("format")
    if not isinstance(form, Mapping):
        raise ValueError(f"{field} format metadata is missing")
    value = form.get("duration")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise ValueError(f"{field} duration must be finite") from error
    return _finite(value, f"{field} duration")


def validate_normalized_media(
    *,
    source_path: Path,
    derived_path: Path,
    expected_source_sha256: str,
    source_probe: Mapping[str, Any] | None = None,
    derived_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not source_path.is_file() or source_path.is_symlink():
        raise ValueError("source media is missing")
    if not derived_path.is_file() or derived_path.is_symlink():
        raise ValueError("normalized media is missing")
    if _SHA256.fullmatch(expected_source_sha256) is None:
        raise ValueError("source media SHA-256 is invalid")
    source_sha256 = sha256_file(source_path)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source media hash mismatch")
    source_meta = source_probe if source_probe is not None else probe_media(source_path)
    derived_meta = derived_probe if derived_probe is not None else probe_media(derived_path)
    streams = derived_meta.get("streams")
    if not isinstance(streams, list):
        raise ValueError("normalized streams are missing")
    source_codecs = sorted(
        (stream.get("codec_type"), stream.get("codec_name"))
        for stream in (source_meta.get("streams") or [])
        if isinstance(stream, Mapping)
    )
    derived_codecs = sorted(
        (stream.get("codec_type"), stream.get("codec_name"))
        for stream in streams
        if isinstance(stream, Mapping)
    )
    if not {codec_type for codec_type, _codec_name in source_codecs} >= {"audio", "video"}:
        raise ValueError("source media must contain video and audio streams")
    if derived_codecs != [("audio", "aac"), ("video", "h264")]:
        raise ValueError("normalized media must contain h264 video and aac audio")
    source_duration = _duration_from_probe(source_meta, "source")
    derived_duration = _duration_from_probe(derived_meta, "derived")
    if abs(source_duration - derived_duration) > COVERAGE_TOLERANCE_SECONDS:
        raise ValueError("normalized duration drift")
    return {
        "source_media_sha256": source_sha256,
        "derived_media_sha256": sha256_file(derived_path),
        "source_duration_seconds": source_duration,
        "derived_duration_seconds": derived_duration,
        "source_streams": source_codecs,
        "derived_streams": derived_codecs,
    }


def write_payload_checksums(root: Path, payload_paths: Sequence[Path]) -> Path:
    root = root.resolve()
    entries: list[tuple[str, str]] = []
    for path in payload_paths:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("payload must be inside output root") from error
        if relative == "SHA256SUMS" or not resolved.is_file():
            raise ValueError("payload paths must be regular files other than SHA256SUMS")
        entries.append((relative, sha256_file(resolved)))
    if len({relative for relative, _digest in entries}) != len(entries):
        raise ValueError("payload paths must be unique")
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{digest}  ./{relative}\n" for relative, digest in sorted(entries)),
        encoding="utf-8",
    )
    return checksums


def write_terminal_marker(
    root: Path,
    *,
    run_id: str,
    attempt_uuid: str,
    output_identity: str,
    payload_checksum: str,
    row_count: int,
) -> Path:
    if _SHA256.fullmatch(payload_checksum) is None:
        raise ValueError("payload checksum must be SHA-256")
    marker = root / "terminal-status.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": run_id,
                "attempt_uuid": attempt_uuid,
                "output_identity": output_identity,
                "payload_checksum": payload_checksum,
                "row_count": row_count,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def validate_controller_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("label_policy") != "label-free-controller":
        raise ValueError("controller identity is not label-free")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("controller rows are missing")
    _reject_forbidden_fields(payload)
    if ordered_ids_sha256(rows) != payload.get("panel", {}).get("ids_sha256"):
        raise ValueError("controller ID digest mismatch")
    return dict(payload)


def normalize_media_file(
    *,
    source_path: Path,
    derived_path: Path,
    expected_source_sha256: str,
) -> dict[str, Any]:
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    if not derived_path.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-n",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                "-shortest",
                str(derived_path),
            ],
            check=True,
        )
    return validate_normalized_media(
        source_path=source_path,
        derived_path=derived_path,
        expected_source_sha256=expected_source_sha256,
    )


def run_request_file(
    *,
    requests_path: Path,
    normalized_root: Path,
    normalized_controller_path: Path,
    output_path: Path,
    server_url: str,
    response_format_supported: bool,
    runtime_identity: Mapping[str, Any] | None = None,
    post: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run one private request file while retaining no raw model response."""

    controller = json.loads(normalized_controller_path.read_text(encoding="utf-8"))
    normalized_rows = controller.get("rows") if isinstance(controller, Mapping) else None
    if not isinstance(normalized_rows, list):
        raise ValueError("normalized controller rows are missing")
    normalized_by_id = {row.get("video_id"): row for row in normalized_rows if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    for line in requests_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("request file must contain object rows")
            rows.append(value)
    if not rows:
        raise ValueError("request file must not be empty")
    sender = post or (lambda url, payload: post_json(url, payload))
    results: list[dict[str, Any]] = []
    for row in rows:
        video_id = row.get("video_id")
        normalized = normalized_by_id.get(video_id)
        if not isinstance(video_id, str) or not isinstance(normalized, Mapping):
            raise ValueError("request row is not bound to normalized controller")
        media_path = normalized_root / f"{video_id}.mp4"
        result = run_one_request(
            request=row,
            duration_seconds=float(normalized["derived_duration_seconds"]),
            post_json=lambda payload, _url=server_url: sender(_url, payload),
            media_path=media_path,
            audio_enabled=row.get("condition_id") == "av-map",
            response_format_supported=response_format_supported,
            runtime_identity=runtime_identity,
        )
        result.update(
            {
                "run_id": row.get("run_id"),
                "slice_id": row.get("slice_id"),
                "question": row.get("question"),
                "question_stratum": row.get("question_stratum"),
                "controller_request_sha256": row.get("request_sha256"),
                "media_path": normalized.get("media_path"),
                "source_media_sha256": normalized.get("source_media_sha256"),
                "normalized_media_sha256": normalized.get("derived_media_sha256"),
                "source_duration_seconds": normalized.get("source_duration_seconds"),
                "normalized_duration_seconds": normalized.get("derived_duration_seconds"),
            }
        )
        results.append(result)
    output_path.write_bytes(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results).encode("utf-8")
    )
    return results


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-file")
    run_parser.add_argument("--requests", type=Path, required=True)
    run_parser.add_argument("--normalized-root", type=Path, required=True)
    run_parser.add_argument("--normalized-controller", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--server-url", required=True)
    run_parser.add_argument("--response-format-supported", action="store_true")
    run_parser.add_argument("--runtime-json", type=Path)
    args = parser.parse_args(argv)
    runtime = None
    if args.runtime_json is not None:
        runtime = json.loads(args.runtime_json.read_text(encoding="utf-8"))
    rows = run_request_file(
        requests_path=args.requests,
        normalized_root=args.normalized_root,
        normalized_controller_path=args.normalized_controller,
        output_path=args.output,
        server_url=args.server_url,
        response_format_supported=args.response_format_supported,
        runtime_identity=runtime,
    )
    print(json.dumps({"row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
