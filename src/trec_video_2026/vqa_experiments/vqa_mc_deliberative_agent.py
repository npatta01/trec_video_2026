"""Provider-independent contracts for the single-model deliberative MC agent."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_common import (
    OPTION_IDS,
    parse_ranking,
)


ALLOWED_FPS = frozenset((1.0, 4.0, 8.0))
ALLOWED_RESOLUTIONS = frozenset(("low", "high"))
ACTION_KINDS = frozenset(("rewatch_full", "inspect_interval"))
FOCUS_RESOLUTIONS = frozenset(("absent", "present", "not_resolvable"))

RANKING_PROPERTY = {
    "type": "array",
    "items": {"type": "string", "enum": list(OPTION_IDS)},
    "minItems": 4,
    "maxItems": 4,
}
HYPOTHESIS_PROPERTY = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string", "enum": list(OPTION_IDS)},
        "case_for": {"type": "string", "minLength": 1, "maxLength": 600},
        "case_against": {"type": "string", "minLength": 1, "maxLength": 600},
    },
    "required": ["option_id", "case_for", "case_against"],
    "additionalProperties": False,
}
OBSERVATION_PROPERTY = {
    "type": "object",
    "properties": {
        "timestamp_seconds": {"type": "number"},
        "literal_observation": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "interpretation": {"type": "string", "minLength": 1, "maxLength": 600},
    },
    "required": ["timestamp_seconds", "literal_observation", "interpretation"],
    "additionalProperties": False,
}
ASSESSMENT_PROPERTY = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string", "enum": list(OPTION_IDS)},
        "case_summary": {"type": "string", "minLength": 1, "maxLength": 600},
        "supporting_observation_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 12,
        },
        "opposing_observation_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 12,
        },
    },
    "required": [
        "option_id",
        "case_summary",
        "supporting_observation_ids",
        "opposing_observation_ids",
    ],
    "additionalProperties": False,
}
EXPECTATION_PROPERTY = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string", "enum": list(OPTION_IDS)},
        "expected_observation": {
            "type": "string",
            "minLength": 1,
            "maxLength": 400,
        },
    },
    "required": ["option_id", "expected_observation"],
    "additionalProperties": False,
}
ACTION_PROPERTY = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": sorted(ACTION_KINDS)},
        "start_seconds": {"type": "number"},
        "end_seconds": {"type": "number"},
        "fps": {"type": "number", "enum": sorted(ALLOWED_FPS)},
        "resolution": {"type": "string", "enum": sorted(ALLOWED_RESOLUTIONS)},
        "investigative_focus": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "expectations": {
            "type": "array",
            "items": EXPECTATION_PROPERTY,
            "minItems": 1,
            "maxItems": 4,
        },
    },
    "required": [
        "kind",
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
        "investigative_focus",
        "expectations",
    ],
    "additionalProperties": False,
}

SOLVER_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "hypotheses": {
            "type": "array",
            "items": HYPOTHESIS_PROPERTY,
            "minItems": 4,
            "maxItems": 4,
        },
        "observations": {
            "type": "array",
            "items": OBSERVATION_PROPERTY,
            "minItems": 1,
            "maxItems": 8,
        },
        "strongest_challenge": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "unresolved_questions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 400},
            "maxItems": 6,
        },
        "candidate_checks": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 400},
            "maxItems": 2,
        },
    },
    "required": [
        "ranking",
        "hypotheses",
        "observations",
        "strongest_challenge",
        "unresolved_questions",
        "candidate_checks",
    ],
    "additionalProperties": False,
}
PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "option_assessments": {
            "type": "array",
            "items": ASSESSMENT_PROPERTY,
            "minItems": 4,
            "maxItems": 4,
        },
        "remaining_uncertainty": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "decision": {
            "type": "string",
            "enum": ["finish_search", "inspect"],
        },
        "action": {"anyOf": [ACTION_PROPERTY, {"type": "null"}]},
    },
    "required": [
        "ranking",
        "option_assessments",
        "remaining_uncertainty",
        "decision",
        "action",
    ],
    "additionalProperties": False,
}
OBSERVER_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": OBSERVATION_PROPERTY,
            "minItems": 1,
            "maxItems": 8,
        },
        "view_limitations": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "focus_resolution": {
            "type": "string",
            "enum": sorted(FOCUS_RESOLUTIONS),
        },
    },
    "required": ["observations", "view_limitations", "focus_resolution"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "decisive_observation_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 8,
        },
        "strongest_counterargument": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "counterargument_resolution": {
            "type": "string",
            "minLength": 1,
            "maxLength": 600,
        },
        "ambiguity_present": {"type": "boolean"},
        "ambiguity_note": {"type": "string", "minLength": 1, "maxLength": 600},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": [
        "ranking",
        "decisive_observation_ids",
        "strongest_counterargument",
        "counterargument_resolution",
        "ambiguity_present",
        "ambiguity_note",
        "rationale",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class MediaAction:
    kind: str
    start_seconds: float
    end_seconds: float
    fps: float
    resolution: str
    investigative_focus: str
    expectations: tuple[dict[str, str], ...]


def _object(value: object, *, keys: set[str], name: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has invalid fields")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    result = value.strip()
    if not result or len(result) > limit:
        raise ValueError(f"{name} length is invalid")
    return result


def _text_list(value: object, name: str, *, maximum: int, limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded list")
    return [_text(item, name, limit) for item in value]


def _option_rows(
    value: object,
    *,
    name: str,
    keys: set[str],
    text_fields: Sequence[tuple[str, int]],
    list_fields: Sequence[tuple[str, int]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(OPTION_IDS):
        raise ValueError(f"{name} must contain all options")
    rows: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != keys:
            raise ValueError(f"{name} row has invalid fields")
        option_id = entry.get("option_id")
        if option_id not in OPTION_IDS:
            raise ValueError(f"{name} option ID is invalid")
        row: dict[str, Any] = {"option_id": option_id}
        for field, limit in text_fields:
            row[field] = _text(entry.get(field), f"{name}.{field}", limit)
        for field, maximum in list_fields:
            row[field] = _text_list(
                entry.get(field), f"{name}.{field}", maximum=maximum, limit=80
            )
        rows.append(row)
    if {row["option_id"] for row in rows} != set(OPTION_IDS):
        raise ValueError(f"{name} must contain each option exactly once")
    return sorted(rows, key=lambda row: str(row["option_id"]))


def _observations(
    value: object,
    *,
    duration_seconds: float,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    duration = _finite(duration_seconds, "duration_seconds")
    end = duration if end_seconds is None else end_seconds
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("observations must contain one to eight rows")
    parsed: list[dict[str, Any]] = []
    required = {"timestamp_seconds", "literal_observation", "interpretation"}
    for index, entry in enumerate(value, start=1):
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError("observation has invalid fields")
        timestamp = _finite(entry["timestamp_seconds"], "timestamp_seconds")
        if timestamp < 0 or timestamp > duration:
            raise ValueError("observation timestamp is outside the video")
        if timestamp < start_seconds or timestamp > end:
            raise ValueError("observation timestamp is outside the realized view")
        row = {
            "timestamp_seconds": timestamp,
            "literal_observation": _text(
                entry["literal_observation"], "literal_observation", 600
            ),
            "interpretation": _text(entry["interpretation"], "interpretation", 600),
        }
        if prefix is not None:
            row["observation_id"] = f"{prefix}-o{index}"
        parsed.append(row)
    return parsed


def _expectations(value: object) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise ValueError("expectations must contain one to four rows")
    parsed: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != {
            "option_id",
            "expected_observation",
        }:
            raise ValueError("expectation has invalid fields")
        option_id = entry.get("option_id")
        if option_id not in OPTION_IDS:
            raise ValueError("expectation option ID is invalid")
        parsed.append(
            {
                "option_id": str(option_id),
                "expected_observation": _text(
                    entry.get("expected_observation"), "expected_observation", 400
                ),
            }
        )
    if len({entry["option_id"] for entry in parsed}) != len(parsed):
        raise ValueError("expectation option IDs must be unique")
    return tuple(parsed)


def validate_action(value: object, *, duration_seconds: float) -> MediaAction:
    keys = {
        "kind",
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
        "investigative_focus",
        "expectations",
    }
    payload = _object(value, keys=keys, name="action")
    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    kind = payload["kind"]
    if kind not in ACTION_KINDS:
        raise ValueError("action kind is unsupported")
    start = _finite(payload["start_seconds"], "start_seconds")
    end = _finite(payload["end_seconds"], "end_seconds")
    if start < 0 or end > duration or end <= start:
        raise ValueError("action interval is outside the video")
    if kind == "rewatch_full" and (start != 0.0 or end != duration):
        raise ValueError("rewatch_full must cover the complete video")
    fps = _finite(payload["fps"], "fps")
    if fps not in ALLOWED_FPS:
        raise ValueError("action FPS is unsupported")
    resolution = payload["resolution"]
    if resolution not in ALLOWED_RESOLUTIONS:
        raise ValueError("action resolution is unsupported")
    return MediaAction(
        kind=str(kind),
        start_seconds=start,
        end_seconds=end,
        fps=fps,
        resolution=str(resolution),
        investigative_focus=_text(
            payload["investigative_focus"], "investigative_focus", 500
        ),
        expectations=_expectations(payload["expectations"]),
    )


def parse_solver(
    raw: object, *, duration_seconds: float, solver_id: str
) -> dict[str, Any]:
    keys = {
        "ranking",
        "hypotheses",
        "observations",
        "strongest_challenge",
        "unresolved_questions",
        "candidate_checks",
    }
    payload = _object(raw, keys=keys, name="solver response")
    identifier = _text(solver_id, "solver_id", 40)
    prefix = identifier.replace("_", "-")
    return {
        "solver_id": identifier,
        "ranking": parse_ranking({"ranking": payload["ranking"]}),
        "hypotheses": _option_rows(
            payload["hypotheses"],
            name="hypotheses",
            keys={"option_id", "case_for", "case_against"},
            text_fields=(("case_for", 600), ("case_against", 600)),
            list_fields=(),
        ),
        "observations": _observations(
            payload["observations"],
            duration_seconds=duration_seconds,
            prefix=prefix,
        ),
        "strongest_challenge": _text(
            payload["strongest_challenge"], "strongest_challenge", 600
        ),
        "unresolved_questions": _text_list(
            payload["unresolved_questions"],
            "unresolved_questions",
            maximum=6,
            limit=400,
        ),
        "candidate_checks": _text_list(
            payload["candidate_checks"],
            "candidate_checks",
            maximum=2,
            limit=400,
        ),
    }


def parse_planner(raw: object, *, duration_seconds: float) -> dict[str, Any]:
    keys = {
        "ranking",
        "option_assessments",
        "remaining_uncertainty",
        "decision",
        "action",
    }
    payload = _object(raw, keys=keys, name="planner response")
    decision = payload["decision"]
    if decision not in {"finish_search", "inspect"}:
        raise ValueError("planner decision is invalid")
    action = payload["action"]
    if decision == "finish_search":
        if action is not None:
            raise ValueError("finish_search action must be null")
        parsed_action = None
    else:
        if action is None:
            raise ValueError("inspect decision requires an action")
        parsed_action = validate_action(action, duration_seconds=duration_seconds)
    return {
        "ranking": parse_ranking({"ranking": payload["ranking"]}),
        "option_assessments": _option_rows(
            payload["option_assessments"],
            name="option_assessments",
            keys={
                "option_id",
                "case_summary",
                "supporting_observation_ids",
                "opposing_observation_ids",
            },
            text_fields=(("case_summary", 600),),
            list_fields=(
                ("supporting_observation_ids", 12),
                ("opposing_observation_ids", 12),
            ),
        ),
        "remaining_uncertainty": _text(
            payload["remaining_uncertainty"], "remaining_uncertainty", 600
        ),
        "decision": str(decision),
        "action": parsed_action,
    }


def parse_observer(
    raw: object,
    *,
    action: MediaAction,
    duration_seconds: float,
    view_id: str,
) -> dict[str, Any]:
    payload = _object(
        raw,
        keys={"observations", "view_limitations", "focus_resolution"},
        name="observer response",
    )
    resolution = payload["focus_resolution"]
    if resolution not in FOCUS_RESOLUTIONS:
        raise ValueError("focus_resolution is invalid")
    return {
        "view_id": _text(view_id, "view_id", 80),
        "observations": _observations(
            payload["observations"],
            duration_seconds=duration_seconds,
            start_seconds=action.start_seconds,
            end_seconds=action.end_seconds,
        ),
        "view_limitations": _text(
            payload["view_limitations"], "view_limitations", 600
        ),
        "focus_resolution": str(resolution),
    }


def parse_judge(
    raw: object, *, observation_ids: set[str] | frozenset[str]
) -> dict[str, Any]:
    keys = {
        "ranking",
        "decisive_observation_ids",
        "strongest_counterargument",
        "counterargument_resolution",
        "ambiguity_present",
        "ambiguity_note",
        "rationale",
    }
    payload = _object(raw, keys=keys, name="judge response")
    citations = _text_list(
        payload["decisive_observation_ids"],
        "decisive_observation_ids",
        maximum=8,
        limit=80,
    )
    unknown = sorted(set(citations) - set(observation_ids))
    if unknown:
        raise ValueError("judge cites an unknown observation ID")
    ambiguity = payload["ambiguity_present"]
    if not isinstance(ambiguity, bool):
        raise ValueError("ambiguity_present must be boolean")
    return {
        "ranking": parse_ranking({"ranking": payload["ranking"]}),
        "decisive_observation_ids": citations,
        "strongest_counterargument": _text(
            payload["strongest_counterargument"], "strongest_counterargument", 600
        ),
        "counterargument_resolution": _text(
            payload["counterargument_resolution"],
            "counterargument_resolution",
            600,
        ),
        "ambiguity_present": ambiguity,
        "ambiguity_note": _text(payload["ambiguity_note"], "ambiguity_note", 600),
        "rationale": _text(payload["rationale"], "rationale", 800),
    }


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
        lines.append(f"{option_id}. {text.strip()}")
    if [line[0] for line in lines] != list(OPTION_IDS):
        raise ValueError("row options must use canonical A-D order")
    return "\n".join(lines)


def _question(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question is invalid")
    return question.strip()


def solver_prompt(row: Mapping[str, Any]) -> str:
    duration = _finite(row.get("duration_seconds"), "duration_seconds")
    return (
        "Watch and listen to the complete video. Independently evaluate every answer "
        "option. Record concise literal timestamped observations separately from "
        "their interpretation. Build a case for and against each option, rank all "
        "four, identify the strongest challenge to the leader, and name at most two "
        "observations that could materially change the ranking. Return only JSON "
        "matching the supplied schema.\n\n"
        f"Video duration: {duration:.3f} seconds\n"
        f"Question: {_question(row)}\nOptions:\n{_option_lines(row)}"
    )


def planner_prompt(
    row: Mapping[str, Any],
    notebook: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> str:
    return (
        "Act as an evidence-search planner. Compare every option using the two "
        "anonymized analyses and append-only notebook. Identify the uncertainty most "
        "likely to change the ranking. Either finish, or request exactly one native-AV "
        "view and preregister what each relevant interpretation predicts before the "
        "view is observed. Write a neutral investigative focus suitable for an observer "
        "that will not see the options or current rankings. Return only JSON matching "
        "the supplied schema.\n\n"
        f"Question: {_question(row)}\nOptions:\n{_option_lines(row)}\n\n"
        f"Remaining mechanical budget:\n{json.dumps(dict(budget), sort_keys=True)}\n\n"
        f"Notebook:\n{json.dumps(dict(notebook), ensure_ascii=False, sort_keys=True)}"
    )


def observer_prompt(action: MediaAction) -> str:
    return (
        "Observe only the supplied native audio-video view. Do not choose an answer "
        "and do not infer events outside the view. Record literal visible or audible "
        "facts first, then put any interpretation in its separate field. Use original-"
        "video seconds and state the limitations of this view. Return only JSON "
        "matching the supplied schema.\n\n"
        f"Investigative focus: {action.investigative_focus}"
    )


def _all_observations(notebook: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations: list[Mapping[str, Any]] = []
    solvers = notebook.get("solvers")
    rounds = notebook.get("rounds")
    if isinstance(solvers, list):
        for solver in solvers:
            if isinstance(solver, Mapping) and isinstance(solver.get("observations"), list):
                observations.extend(
                    item
                    for item in solver["observations"]
                    if isinstance(item, Mapping)
                )
    if isinstance(rounds, list):
        for round_entry in rounds:
            if not isinstance(round_entry, Mapping):
                continue
            observer = round_entry.get("observer")
            if isinstance(observer, Mapping) and isinstance(
                observer.get("observations"), list
            ):
                observations.extend(
                    item
                    for item in observer["observations"]
                    if isinstance(item, Mapping)
                )
    return observations


def judge_prompt(row: Mapping[str, Any], notebook: Mapping[str, Any]) -> str:
    observations = _all_observations(notebook)
    literal_lines = [
        f"{item.get('observation_id')}: t={item.get('timestamp_seconds')} — "
        f"{item.get('literal_observation')}"
        for item in observations
    ]
    interpretation_lines = [
        f"{item.get('observation_id')}: {item.get('interpretation')}"
        for item in observations
    ]
    return (
        "Solve the multiple-choice item afresh using the complete native audio-video "
        "and the audit notebook. Literal observations are listed before interpretations. "
        "Evaluate every option, address the strongest counterargument, cite only existing "
        "observation IDs, record unresolved ambiguity, and return a complete ranking. "
        "Do not preserve an earlier answer merely because it came first or last. Return "
        "only JSON matching the supplied schema.\n\n"
        f"Question: {_question(row)}\nOptions:\n{_option_lines(row)}\n\n"
        "Literal observations\n"
        + ("\n".join(literal_lines) or "None recorded.")
        + "\n\nInterpretations\n"
        + ("\n".join(interpretation_lines) or "None recorded.")
        + "\n\nAnonymized notebook\n"
        + json.dumps(dict(notebook), ensure_ascii=False, sort_keys=True)
    )


def initialize_notebook(
    video_id: str, solver_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identifier = _text(video_id, "video_id", 200)
    records = [copy.deepcopy(dict(record)) for record in solver_records]
    solver_ids = [record.get("solver_id") for record in records]
    if len(records) > 2 or len(set(solver_ids)) != len(solver_ids):
        raise ValueError("solver records are invalid")
    return {"schema_version": 1, "video_id": identifier, "solvers": records, "rounds": []}


def append_planner(
    notebook: Mapping[str, Any], *, round_index: int, planner: Mapping[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(notebook))
    rounds = updated.get("rounds")
    if not isinstance(rounds, list) or round_index != len(rounds) + 1:
        raise ValueError("planner round order is invalid")
    planner_entry = copy.deepcopy(dict(planner))
    if isinstance(planner_entry.get("action"), MediaAction):
        planner_entry["action"] = asdict(planner_entry["action"])
    rounds.append({"round_index": round_index, "planner": planner_entry})
    return updated


def append_observer(
    notebook: Mapping[str, Any],
    *,
    round_index: int,
    action: MediaAction,
    observer: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(notebook))
    rounds = updated.get("rounds")
    if not isinstance(rounds, list) or not 1 <= round_index <= len(rounds):
        raise ValueError("observer round does not exist")
    entry = rounds[round_index - 1]
    if not isinstance(entry, dict) or entry.get("round_index") != round_index:
        raise ValueError("observer round order is invalid")
    if "observer" in entry:
        raise ValueError("observer entry already exists")
    parsed_observer = copy.deepcopy(dict(observer))
    observations = parsed_observer.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observer observations are invalid")
    for index, observation in enumerate(observations, start=1):
        if not isinstance(observation, dict):
            raise ValueError("observer observation is invalid")
        observation["observation_id"] = f"round-{round_index}-o{index}"
    entry["action"] = asdict(action)
    entry["provenance"] = copy.deepcopy(dict(provenance))
    entry["observer"] = parsed_observer
    return updated


def notebook_observation_ids(notebook: Mapping[str, Any]) -> frozenset[str]:
    identifiers = {
        str(item["observation_id"])
        for item in _all_observations(notebook)
        if isinstance(item.get("observation_id"), str)
    }
    return frozenset(identifiers)


def choose_final_ranking(
    judge_record: Mapping[str, Any] | None,
    planner_records: Sequence[Mapping[str, Any]],
    solver_records: Sequence[Mapping[str, Any]],
) -> tuple[list[str] | None, str]:
    if judge_record is not None and isinstance(judge_record.get("ranking"), list):
        return parse_ranking({"ranking": judge_record["ranking"]}), "judge"
    for planner in reversed(planner_records):
        if isinstance(planner.get("ranking"), list):
            return parse_ranking({"ranking": planner["ranking"]}), "planner_fallback"
    for solver in solver_records:
        if isinstance(solver.get("ranking"), list):
            return parse_ranking({"ranking": solver["ranking"]}), "solver_fallback"
    return None, "invalid"
