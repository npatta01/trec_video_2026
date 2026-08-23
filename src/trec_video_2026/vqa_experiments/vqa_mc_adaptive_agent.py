"""Provider-independent contracts for the bounded VQA MC inspection agent."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import (
    OPTION_IDS,
    parse_ranking,
)


ALLOWED_FPS = frozenset((1.0, 4.0, 8.0))
ALLOWED_RESOLUTIONS = frozenset(("low", "high"))
EVIDENCE_GAPS = frozenset(
    (
        "text_detail",
        "rapid_or_repeated_action",
        "speech_or_sound",
        "global_composition",
        "temporal_order",
        "other",
    )
)
PLAN_KEYS = frozenset(
    (
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
        "evidence_gap",
        "evidence_question",
    )
)
FALLBACK_EVIDENCE_QUESTION = (
    "Re-examine the complete video and verify which answer option is best supported."
)


RANKING_PROPERTY = {
    "type": "array",
    "items": {"type": "string", "enum": list(OPTION_IDS)},
    "minItems": 4,
    "maxItems": 4,
}
INSPECTION_PROPERTY = {
    "type": "object",
    "properties": {
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "fps": {"type": "number", "enum": [1.0, 4.0, 8.0]},
        "resolution": {"type": "string", "enum": ["low", "high"]},
        "evidence_gap": {"type": "string", "enum": sorted(EVIDENCE_GAPS)},
        "evidence_question": {
            "type": "string",
            "minLength": 1,
            "maxLength": 400,
        },
    },
    "required": [
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
        "evidence_gap",
        "evidence_question",
    ],
    "additionalProperties": False,
}
PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "inspection": INSPECTION_PROPERTY,
    },
    "required": ["ranking", "inspection"],
    "additionalProperties": False,
}
EVIDENCE_PROPERTY = {
    "type": "object",
    "properties": {
        "timestamp_seconds": {"type": "number"},
        "observation": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["timestamp_seconds", "observation"],
    "additionalProperties": False,
}
FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "evidence": {
            "type": "array",
            "items": EVIDENCE_PROPERTY,
            "minItems": 1,
            "maxItems": 3,
        },
    },
    "required": ["ranking", "evidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class InspectionPlan:
    start_seconds: float
    end_seconds: float
    fps: float
    resolution: str
    evidence_gap: str
    evidence_question: str
    source: str


def _object(raw: object) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("response is not JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError("response must be an object")
    return raw


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_plan(value: object, *, duration_seconds: float) -> InspectionPlan:
    duration = _finite_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    if not isinstance(value, Mapping) or set(value) != PLAN_KEYS:
        raise ValueError("inspection plan has invalid fields")
    start = _finite_number(value["start_seconds"], "start_seconds")
    end = _finite_number(value["end_seconds"], "end_seconds")
    fps = _finite_number(value["fps"], "fps")
    resolution = value["resolution"]
    evidence_gap = value["evidence_gap"]
    evidence_question = value["evidence_question"]
    if start < 0 or end > duration or end - start < 0.001:
        raise ValueError("inspection interval is outside the video")
    if fps not in ALLOWED_FPS:
        raise ValueError("inspection FPS is unsupported")
    if resolution not in ALLOWED_RESOLUTIONS:
        raise ValueError("inspection resolution is unsupported")
    if evidence_gap not in EVIDENCE_GAPS:
        raise ValueError("inspection evidence gap is unsupported")
    if not isinstance(evidence_question, str):
        raise ValueError("evidence question must be text")
    question = evidence_question.strip()
    if not question or len(question) > 400:
        raise ValueError("evidence question length is invalid")
    return InspectionPlan(
        start_seconds=start,
        end_seconds=end,
        fps=fps,
        resolution=str(resolution),
        evidence_gap=str(evidence_gap),
        evidence_question=question,
        source="model",
    )


def fallback_plan(duration_seconds: float) -> InspectionPlan:
    duration = _finite_number(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    return InspectionPlan(
        start_seconds=0.0,
        end_seconds=duration,
        fps=4.0,
        resolution="high",
        evidence_gap="other",
        evidence_question=FALLBACK_EVIDENCE_QUESTION,
        source="fallback",
    )


def plan_dict(plan: InspectionPlan) -> dict[str, Any]:
    return asdict(plan)


def parse_stage1(raw: object, *, duration_seconds: float) -> dict[str, Any]:
    payload = _object(raw)
    if set(payload) != {"ranking", "inspection"}:
        raise ValueError("stage-one response has invalid fields")
    ranking = parse_ranking({"ranking": payload["ranking"]})
    inspection = validate_plan(
        payload["inspection"], duration_seconds=duration_seconds
    )
    return {"ranking": ranking, "inspection": inspection}


def parse_stage2(raw: object, *, duration_seconds: float) -> dict[str, Any]:
    payload = _object(raw)
    if set(payload) != {"ranking", "evidence"}:
        raise ValueError("stage-two response has invalid fields")
    ranking = parse_ranking({"ranking": payload["ranking"]})
    evidence = payload["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        raise ValueError("stage-two evidence must contain one to three entries")
    duration = _finite_number(duration_seconds, "duration_seconds")
    parsed_evidence: list[dict[str, Any]] = []
    for entry in evidence:
        if not isinstance(entry, Mapping) or set(entry) != {
            "timestamp_seconds",
            "observation",
        }:
            raise ValueError("stage-two evidence entry has invalid fields")
        timestamp = _finite_number(entry["timestamp_seconds"], "timestamp_seconds")
        observation = entry["observation"]
        if timestamp < 0 or timestamp > duration:
            raise ValueError("evidence timestamp is outside the video")
        if not isinstance(observation, str):
            raise ValueError("evidence observation must be text")
        text = observation.strip()
        if not text or len(text) > 500:
            raise ValueError("evidence observation length is invalid")
        parsed_evidence.append(
            {"timestamp_seconds": timestamp, "observation": text}
        )
    return {"ranking": ranking, "evidence": parsed_evidence}


def _option_lines(row: Mapping[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("row options are invalid")
    lines: list[str] = []
    for option in options:
        if not isinstance(option, Mapping):
            raise ValueError("row option is invalid")
        option_id = option.get("option_id")
        text = option.get("text")
        if option_id not in OPTION_IDS or not isinstance(text, str) or not text.strip():
            raise ValueError("row option ID or text is invalid")
        lines.append(f"{option_id}. {text}")
    if [line[0] for line in lines] != list(OPTION_IDS):
        raise ValueError("row options must be in canonical A-D order")
    return "\n".join(lines)


def stage1_prompt(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    duration = _finite_number(row.get("duration_seconds"), "duration_seconds")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question is invalid")
    return (
        "Watch and listen to the complete video. Rank all four options from most "
        "likely to least likely. Then always request exactly one follow-up inspection "
        "that would best verify or overturn the leading answer. Choose any valid time "
        "interval, FPS 1/4/8, and low/high resolution. Use high resolution for small "
        "text or details, denser FPS for rapid or repeated actions, and a focused "
        "interval for speech, sound, temporal order, or scene composition. Return "
        "only JSON matching the supplied schema. Do not claim certainty instead of "
        "requesting an inspection.\n\n"
        f"Video duration: {duration:.3f} seconds\n"
        f"Question: {question.strip()}\nOptions:\n{_option_lines(row)}"
    )


def stage2_prompt(
    row: Mapping[str, Any],
    plan: InspectionPlan,
    stage1_ranking: Sequence[str] | None,
) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question is invalid")
    if stage1_ranking is None:
        top_two_line = "Stage-one top-two set: unavailable due to an invalid first pass."
    else:
        parsed = parse_ranking({"ranking": list(stage1_ranking)})
        top_two_line = (
            "Stage-one top-two set (unordered): " + ", ".join(sorted(parsed[:2]))
        )
    return (
        "Inspect this requested native audio-video view and make the final decision. "
        "Rank every option A-D, using the inspection to verify or overturn the first "
        "pass. Evidence timestamps must be original-video seconds, not seconds "
        "relative to the clipped view. Return only JSON matching the supplied schema.\n\n"
        f"Inspection question: {plan.evidence_question}\n"
        f"{top_two_line}\n"
        f"Question: {question.strip()}\nOptions:\n{_option_lines(row)}"
    )


def choose_final_ranking(
    stage1_ranking: Sequence[str] | None,
    stage2_ranking: Sequence[str] | None,
) -> tuple[list[str] | None, str]:
    if stage2_ranking is not None:
        return parse_ranking({"ranking": list(stage2_ranking)}), "stage2"
    if stage1_ranking is not None:
        return parse_ranking({"ranking": list(stage1_ranking)}), "stage1_fallback"
    return None, "invalid"
