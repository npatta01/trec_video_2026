"""Provider-independent contracts for the four-option MC evidence agent."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trec_video_2026.vqa_experiments.vqa_mc_common import OPTION_IDS, parse_ranking


ALLOWED_FPS = frozenset((1.0, 4.0, 8.0))
ALLOWED_RESOLUTIONS = frozenset(("low", "high"))
ACTION_KINDS = frozenset(("rewatch_full", "inspect_interval"))
OPTION_STATUSES = frozenset(("viable", "disfavored", "inconsistent"))
FOCUS_RESOLUTIONS = frozenset(("present", "absent", "unresolved"))
STOP_DECISIONS = frozenset(("continue", "accept_stop"))
STOP_STATUSES = frozenset(("evidence_certified", "forced_stop", "budget_stop"))
MAX_OBSERVATIONS = 3


RANKING_PROPERTY = {
    "type": "array",
    "items": {"type": "string", "enum": list(OPTION_IDS)},
    "minItems": 4,
    "maxItems": 4,
}
EVIDENCE_IDS_PROPERTY = {
    "type": "array",
    "items": {"type": "string", "minLength": 1, "maxLength": 80},
    "maxItems": 24,
}
LEDGER_ROW_PROPERTY = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string", "enum": list(OPTION_IDS)},
        "observable_claim": {"type": "string", "minLength": 1, "maxLength": 800},
        "supporting_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "contradicting_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "untested_claims": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "maxItems": 8,
        },
        "status": {"type": "string", "enum": sorted(OPTION_STATUSES)},
        "interpretation": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": [
        "option_id",
        "observable_claim",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "untested_claims",
        "status",
        "interpretation",
    ],
    "additionalProperties": False,
}
OBSERVATION_PROPERTY = {
    "type": "object",
    "properties": {
        "timestamp_seconds": {"type": "number"},
        "literal_observation": {"type": "string", "minLength": 1, "maxLength": 800},
        "interpretation": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": ["timestamp_seconds", "literal_observation", "interpretation"],
    "additionalProperties": False,
}
EXPECTED_OUTCOME_PROPERTY = {
    "type": "object",
    "properties": {
        "option_id": {"type": "string", "enum": list(OPTION_IDS)},
        "literal_outcome": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["option_id", "literal_outcome"],
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
        "neutral_request": {"type": "string", "minLength": 1, "maxLength": 800},
        "tested_option_ids": {
            "type": "array",
            "items": {"type": "string", "enum": list(OPTION_IDS)},
            "minItems": 1,
            "maxItems": 4,
        },
        "expected_outcomes": {
            "type": "array",
            "items": EXPECTED_OUTCOME_PROPERTY,
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
        "neutral_request",
        "tested_option_ids",
        "expected_outcomes",
    ],
    "additionalProperties": False,
}

INITIALIZER_SCHEMA = {
    "type": "object",
    "properties": {
        "provisional_ranking": RANKING_PROPERTY,
        "ledger": {
            "type": "array",
            "items": LEDGER_ROW_PROPERTY,
            "minItems": 4,
            "maxItems": 4,
        },
        "literal_observations": {
            "type": "array",
            "items": OBSERVATION_PROPERTY,
            "minItems": 1,
            "maxItems": 10,
        },
        "unresolved_distinctions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "maxItems": 8,
        },
    },
    "required": [
        "provisional_ranking",
        "ledger",
        "literal_observations",
        "unresolved_distinctions",
    ],
    "additionalProperties": False,
}
PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "provisional_ranking": RANKING_PROPERTY,
        "unresolved_distinction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
        "action": ACTION_PROPERTY,
        "why_discriminating": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
    },
    "required": [
        "provisional_ranking",
        "unresolved_distinction",
        "action",
        "why_discriminating",
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
            "maxItems": 10,
        },
        "exact_transcription": {"type": "string", "minLength": 1, "maxLength": 1600},
        "focus_resolution": {"type": "string", "enum": sorted(FOCUS_RESOLUTIONS)},
        "view_limitations": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": [
        "observations",
        "exact_transcription",
        "focus_resolution",
        "view_limitations",
    ],
    "additionalProperties": False,
}
UPDATER_SCHEMA = {
    "type": "object",
    "properties": {
        "provisional_ranking": RANKING_PROPERTY,
        "ledger": {
            "type": "array",
            "items": LEDGER_ROW_PROPERTY,
            "minItems": 4,
            "maxItems": 4,
        },
        "mapped_observation_ids": EVIDENCE_IDS_PROPERTY,
        "unresolved_distinctions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
            "maxItems": 8,
        },
    },
    "required": [
        "provisional_ranking",
        "ledger",
        "mapped_observation_ids",
        "unresolved_distinctions",
    ],
    "additionalProperties": False,
}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": sorted(STOP_DECISIONS)},
        "tested_viable_option_ids": {
            "type": "array",
            "items": {"type": "string", "enum": list(OPTION_IDS)},
            "maxItems": 4,
        },
        "leader_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "unexplained_counterevidence_ids": EVIDENCE_IDS_PROPERTY,
        "next_unresolved_distinction": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
        "direct_replacement_supported": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": [
        "decision",
        "tested_viable_option_ids",
        "leader_evidence_ids",
        "unexplained_counterevidence_ids",
        "next_unresolved_distinction",
        "direct_replacement_supported",
        "reason",
    ],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": RANKING_PROPERTY,
        "decisive_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "strongest_counterevidence": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
        "counterevidence_resolution": {
            "type": "string",
            "minLength": 1,
            "maxLength": 800,
        },
        "protected_answer_action": {
            "type": "string",
            "enum": ["retained", "replaced"],
        },
        "direct_contradiction_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "replacement_support_evidence_ids": EVIDENCE_IDS_PROPERTY,
        "stop_status": {"type": "string", "enum": sorted(STOP_STATUSES)},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": [
        "ranking",
        "decisive_evidence_ids",
        "strongest_counterevidence",
        "counterevidence_resolution",
        "protected_answer_action",
        "direct_contradiction_evidence_ids",
        "replacement_support_evidence_ids",
        "stop_status",
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
    neutral_request: str
    tested_option_ids: tuple[str, ...]
    expected_outcomes: tuple[dict[str, str], ...]


def _object(value: object, *, keys: set[str], name: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has invalid fields")
    return value


def _text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    result = value.strip()
    if not result or len(result) > maximum:
        raise ValueError(f"{name} length is invalid")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text_list(value: object, name: str, *, maximum: int, item_limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded list")
    return [_text(item, name, item_limit) for item in value]


def _id_list(value: object, name: str, *, maximum: int = 24) -> list[str]:
    values = _text_list(value, name, maximum=maximum, item_limit=80)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _ranking(value: object) -> list[str]:
    return parse_ranking({"ranking": value})


def _ledger(value: object, *, known_evidence_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(OPTION_IDS):
        raise ValueError("ledger must contain each option exactly once")
    required = {
        "option_id",
        "observable_claim",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "untested_claims",
        "status",
        "interpretation",
    }
    rows: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError("ledger row has invalid fields")
        option_id = entry.get("option_id")
        if option_id not in OPTION_IDS:
            raise ValueError("ledger option ID is invalid")
        support = _id_list(entry.get("supporting_evidence_ids"), "supporting_evidence_ids")
        contradiction = _id_list(
            entry.get("contradicting_evidence_ids"), "contradicting_evidence_ids"
        )
        if known_evidence_ids is not None and (
            set(support) | set(contradiction)
        ) - known_evidence_ids:
            raise ValueError("ledger cites an unknown observation")
        status = entry.get("status")
        if status not in OPTION_STATUSES:
            raise ValueError("ledger option status is invalid")
        rows.append(
            {
                "option_id": str(option_id),
                "observable_claim": _text(
                    entry.get("observable_claim"), "observable_claim", 800
                ),
                "supporting_evidence_ids": support,
                "contradicting_evidence_ids": contradiction,
                "untested_claims": _text_list(
                    entry.get("untested_claims"),
                    "untested_claims",
                    maximum=8,
                    item_limit=500,
                ),
                "status": str(status),
                "interpretation": _text(
                    entry.get("interpretation"), "interpretation", 800
                ),
            }
        )
    if {row["option_id"] for row in rows} != set(OPTION_IDS):
        raise ValueError("ledger must contain each option exactly once")
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
    upper = duration if end_seconds is None else end_seconds
    if not isinstance(value, list) or not 1 <= len(value) <= 10:
        raise ValueError("observations must contain one to ten rows")
    required = {"timestamp_seconds", "literal_observation", "interpretation"}
    parsed: list[dict[str, Any]] = []
    for index, entry in enumerate(value, start=1):
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError("observation has invalid fields")
        timestamp = _finite(entry.get("timestamp_seconds"), "timestamp_seconds")
        if timestamp < 0 or timestamp > duration:
            raise ValueError("observation timestamp is outside the video")
        if timestamp < start_seconds or timestamp > upper:
            raise ValueError("observation timestamp is outside the realized view")
        row = {
            "timestamp_seconds": timestamp,
            "literal_observation": _text(
                entry.get("literal_observation"), "literal_observation", 800
            ),
            "interpretation": _text(
                entry.get("interpretation"), "interpretation", 800
            ),
        }
        if prefix is not None:
            row["observation_id"] = f"{prefix}-o{index}"
        parsed.append(row)
    return parsed


def validate_action(value: object, *, duration_seconds: float) -> MediaAction:
    required = {
        "kind",
        "start_seconds",
        "end_seconds",
        "fps",
        "resolution",
        "neutral_request",
        "tested_option_ids",
        "expected_outcomes",
    }
    payload = _object(value, keys=required, name="media action")
    duration = _finite(duration_seconds, "duration_seconds")
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    kind = payload.get("kind")
    if kind not in ACTION_KINDS:
        raise ValueError("media action kind is unsupported")
    start = _finite(payload.get("start_seconds"), "start_seconds")
    end = _finite(payload.get("end_seconds"), "end_seconds")
    if start < 0 or end <= start or end > duration:
        raise ValueError("media action interval is outside the video")
    if kind == "rewatch_full" and (start != 0.0 or end != duration):
        raise ValueError("rewatch_full must cover the complete video")
    fps = _finite(payload.get("fps"), "fps")
    if fps not in ALLOWED_FPS:
        raise ValueError("media action FPS is unsupported")
    resolution = payload.get("resolution")
    if resolution not in ALLOWED_RESOLUTIONS:
        raise ValueError("media action resolution is unsupported")
    tested = _id_list(payload.get("tested_option_ids"), "tested_option_ids", maximum=4)
    if not tested or not set(tested) <= set(OPTION_IDS):
        raise ValueError("tested option IDs are invalid")
    outcomes = payload.get("expected_outcomes")
    if not isinstance(outcomes, list) or not 1 <= len(outcomes) <= 4:
        raise ValueError("expected outcomes are invalid")
    parsed_outcomes: list[dict[str, str]] = []
    for entry in outcomes:
        if not isinstance(entry, Mapping) or set(entry) != {"option_id", "literal_outcome"}:
            raise ValueError("expected outcome has invalid fields")
        option_id = entry.get("option_id")
        if option_id not in OPTION_IDS:
            raise ValueError("expected outcome option ID is invalid")
        parsed_outcomes.append(
            {
                "option_id": str(option_id),
                "literal_outcome": _text(
                    entry.get("literal_outcome"), "literal_outcome", 500
                ),
            }
        )
    if {row["option_id"] for row in parsed_outcomes} != set(tested):
        raise ValueError("expected outcomes must cover every tested option")
    return MediaAction(
        kind=str(kind),
        start_seconds=start,
        end_seconds=end,
        fps=fps,
        resolution=str(resolution),
        neutral_request=_text(payload.get("neutral_request"), "neutral_request", 800),
        tested_option_ids=tuple(tested),
        expected_outcomes=tuple(
            sorted(parsed_outcomes, key=lambda row: str(row["option_id"]))
        ),
    )


def parse_initializer(
    raw: object, *, duration_seconds: float, protected_ranking: Sequence[str]
) -> dict[str, Any]:
    payload = _object(
        raw,
        keys={
            "provisional_ranking",
            "ledger",
            "literal_observations",
            "unresolved_distinctions",
        },
        name="initializer response",
    )
    protected = _ranking(list(protected_ranking))
    observations = _observations(
        payload.get("literal_observations"),
        duration_seconds=duration_seconds,
        prefix="initializer",
    )
    return {
        "protected_ranking": protected,
        "provisional_ranking": _ranking(payload.get("provisional_ranking")),
        "ledger": _ledger(
            payload.get("ledger"),
            known_evidence_ids={str(row["observation_id"]) for row in observations},
        ),
        "literal_observations": observations,
        "unresolved_distinctions": _text_list(
            payload.get("unresolved_distinctions"),
            "unresolved_distinctions",
            maximum=8,
            item_limit=500,
        ),
    }


def parse_planner(
    raw: object, *, duration_seconds: float, round_index: int
) -> dict[str, Any]:
    if round_index not in {1, 2, 3}:
        raise ValueError("planner round is invalid")
    payload = _object(
        raw,
        keys={
            "provisional_ranking",
            "unresolved_distinction",
            "action",
            "why_discriminating",
        },
        name="planner response",
    )
    return {
        "round_index": round_index,
        "provisional_ranking": _ranking(payload.get("provisional_ranking")),
        "unresolved_distinction": _text(
            payload.get("unresolved_distinction"), "unresolved_distinction", 800
        ),
        "action": validate_action(
            payload.get("action"), duration_seconds=duration_seconds
        ),
        "why_discriminating": _text(
            payload.get("why_discriminating"), "why_discriminating", 800
        ),
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
        keys={
            "observations",
            "exact_transcription",
            "focus_resolution",
            "view_limitations",
        },
        name="observer response",
    )
    resolution = payload.get("focus_resolution")
    if resolution not in FOCUS_RESOLUTIONS:
        raise ValueError("focus resolution is invalid")
    return {
        "view_id": _text(view_id, "view_id", 80),
        "observations": _observations(
            payload.get("observations"),
            duration_seconds=duration_seconds,
            start_seconds=action.start_seconds,
            end_seconds=action.end_seconds,
        ),
        "exact_transcription": _text(
            payload.get("exact_transcription"), "exact_transcription", 1600
        ),
        "focus_resolution": str(resolution),
        "view_limitations": _text(
            payload.get("view_limitations"), "view_limitations", 800
        ),
    }


def notebook_observation_ids(notebook: Mapping[str, Any]) -> frozenset[str]:
    identifiers: set[str] = set()
    initializer = notebook.get("initializer")
    if isinstance(initializer, Mapping):
        for row in initializer.get("literal_observations", []):
            if isinstance(row, Mapping) and isinstance(row.get("observation_id"), str):
                identifiers.add(str(row["observation_id"]))
    rounds = notebook.get("rounds")
    if isinstance(rounds, list):
        for entry in rounds:
            if not isinstance(entry, Mapping):
                continue
            observer = entry.get("observer")
            if not isinstance(observer, Mapping):
                continue
            for row in observer.get("observations", []):
                if isinstance(row, Mapping) and isinstance(row.get("observation_id"), str):
                    identifiers.add(str(row["observation_id"]))
    return frozenset(identifiers)


def observer_observation_ids(notebook: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        identifier
        for identifier in notebook_observation_ids(notebook)
        if identifier.startswith("round-")
    )


def _current_ledger(notebook: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rounds = notebook.get("rounds")
    if isinstance(rounds, list):
        for entry in reversed(rounds):
            if isinstance(entry, Mapping):
                updater = entry.get("updater")
                if isinstance(updater, Mapping) and isinstance(updater.get("ledger"), list):
                    return list(updater["ledger"])
    initializer = notebook.get("initializer")
    if isinstance(initializer, Mapping) and isinstance(initializer.get("ledger"), list):
        return list(initializer["ledger"])
    raise ValueError("notebook has no ledger")


def live_option_ids(notebook: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(row["option_id"])
        for row in _current_ledger(notebook)
        if row.get("status") == "viable"
    )


def parse_update(raw: object, *, notebook: Mapping[str, Any]) -> dict[str, Any]:
    payload = _object(
        raw,
        keys={
            "provisional_ranking",
            "ledger",
            "mapped_observation_ids",
            "unresolved_distinctions",
        },
        name="updater response",
    )
    known = set(notebook_observation_ids(notebook))
    mapped = _id_list(payload.get("mapped_observation_ids"), "mapped_observation_ids")
    if set(mapped) - known:
        raise ValueError("updater cites an unknown observation")
    return {
        "provisional_ranking": _ranking(payload.get("provisional_ranking")),
        "ledger": _ledger(payload.get("ledger"), known_evidence_ids=known),
        "mapped_observation_ids": mapped,
        "unresolved_distinctions": _text_list(
            payload.get("unresolved_distinctions"),
            "unresolved_distinctions",
            maximum=8,
            item_limit=500,
        ),
    }


def parse_critic(
    raw: object, *, notebook: Mapping[str, Any], observation_count: int
) -> dict[str, Any]:
    payload = _object(
        raw,
        keys={
            "decision",
            "tested_viable_option_ids",
            "leader_evidence_ids",
            "unexplained_counterevidence_ids",
            "next_unresolved_distinction",
            "direct_replacement_supported",
            "reason",
        },
        name="critic response",
    )
    decision = payload.get("decision")
    if decision not in STOP_DECISIONS:
        raise ValueError("critic decision is invalid")
    if decision == "accept_stop" and observation_count < 1:
        raise ValueError("critic cannot stop before a valid blinded observation")
    tested = _id_list(
        payload.get("tested_viable_option_ids"),
        "tested_viable_option_ids",
        maximum=4,
    )
    if set(tested) != set(live_option_ids(notebook)):
        raise ValueError("critic must test all viable options")
    known = set(notebook_observation_ids(notebook))
    leader = _id_list(payload.get("leader_evidence_ids"), "leader_evidence_ids")
    counter = _id_list(
        payload.get("unexplained_counterevidence_ids"),
        "unexplained_counterevidence_ids",
    )
    if (set(leader) | set(counter)) - known:
        raise ValueError("critic cites an unknown observation")
    replacement = payload.get("direct_replacement_supported")
    if not isinstance(replacement, bool):
        raise ValueError("direct_replacement_supported must be boolean")
    return {
        "decision": str(decision),
        "tested_viable_option_ids": tested,
        "leader_evidence_ids": leader,
        "unexplained_counterevidence_ids": counter,
        "next_unresolved_distinction": _text(
            payload.get("next_unresolved_distinction"),
            "next_unresolved_distinction",
            800,
        ),
        "direct_replacement_supported": replacement,
        "reason": _text(payload.get("reason"), "reason", 800),
    }


def parse_judge(
    raw: object,
    *,
    notebook: Mapping[str, Any],
    protected_ranking: Sequence[str],
) -> dict[str, Any]:
    payload = _object(
        raw,
        keys={
            "ranking",
            "decisive_evidence_ids",
            "strongest_counterevidence",
            "counterevidence_resolution",
            "protected_answer_action",
            "direct_contradiction_evidence_ids",
            "replacement_support_evidence_ids",
            "stop_status",
            "rationale",
        },
        name="judge response",
    )
    ranking = _ranking(payload.get("ranking"))
    protected = _ranking(list(protected_ranking))
    known = set(notebook_observation_ids(notebook))
    observer_ids = set(observer_observation_ids(notebook))
    decisive = _id_list(payload.get("decisive_evidence_ids"), "decisive_evidence_ids")
    contradiction = _id_list(
        payload.get("direct_contradiction_evidence_ids"),
        "direct_contradiction_evidence_ids",
    )
    support = _id_list(
        payload.get("replacement_support_evidence_ids"),
        "replacement_support_evidence_ids",
    )
    if (set(decisive) | set(contradiction) | set(support)) - known:
        raise ValueError("judge cites unknown evidence")
    action = payload.get("protected_answer_action")
    replaced = ranking[0] != protected[0]
    if replaced:
        if action != "replaced":
            raise ValueError("changed protected answer must be marked replaced")
        if not contradiction or not set(contradiction) <= observer_ids:
            raise ValueError("replacement requires new contradiction evidence")
        if not support or not set(support) <= observer_ids:
            raise ValueError("replacement requires new positive support evidence")
    elif action != "retained":
        raise ValueError("unchanged protected answer must be marked retained")
    status = payload.get("stop_status")
    if status not in STOP_STATUSES:
        raise ValueError("judge stop status is invalid")
    return {
        "ranking": ranking,
        "decisive_evidence_ids": decisive,
        "strongest_counterevidence": _text(
            payload.get("strongest_counterevidence"),
            "strongest_counterevidence",
            800,
        ),
        "counterevidence_resolution": _text(
            payload.get("counterevidence_resolution"),
            "counterevidence_resolution",
            800,
        ),
        "protected_answer_action": str(action),
        "direct_contradiction_evidence_ids": contradiction,
        "replacement_support_evidence_ids": support,
        "stop_status": str(status),
        "rationale": _text(payload.get("rationale"), "rationale", 1200),
    }


def initialize_notebook(
    video_id: str,
    protected_ranking: Sequence[str],
    initializer: Mapping[str, Any],
) -> dict[str, Any]:
    identifier = _text(video_id, "video_id", 160)
    protected = _ranking(list(protected_ranking))
    if list(initializer.get("protected_ranking", [])) != protected:
        raise ValueError("initializer protected ranking does not match")
    return {
        "schema_version": 1,
        "video_id": identifier,
        "protected_ranking": protected,
        "initializer": copy.deepcopy(dict(initializer)),
        "rounds": [],
    }


def append_round_stage(
    notebook: Mapping[str, Any],
    *,
    round_index: int,
    stage: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if round_index not in {1, 2, 3}:
        raise ValueError("round index is invalid")
    if stage not in {"planner", "observer", "updater", "critic", "tool_result"}:
        raise ValueError("round stage is invalid")
    updated = copy.deepcopy(dict(notebook))
    rounds = updated.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("notebook rounds are invalid")
    while len(rounds) < round_index:
        rounds.append({"round_index": len(rounds) + 1})
    entry = rounds[round_index - 1]
    if not isinstance(entry, dict) or entry.get("round_index") != round_index:
        raise ValueError("notebook round is invalid")
    if stage in entry:
        raise ValueError("notebook stage is append-only")
    parsed = copy.deepcopy(dict(record))
    if isinstance(parsed.get("action"), MediaAction):
        parsed["action"] = asdict(parsed["action"])
    if stage == "observer":
        observations = parsed.get("observations")
        if not isinstance(observations, list):
            raise ValueError("observer record has no observations")
        for index, observation in enumerate(observations, start=1):
            if not isinstance(observation, dict):
                raise ValueError("observer observation is invalid")
            observation["observation_id"] = f"round-{round_index}-o{index}"
    entry[stage] = parsed
    return updated


def next_search_state(*, decision: str, observation_count: int) -> str:
    if decision not in STOP_DECISIONS:
        raise ValueError("critic decision is invalid")
    if observation_count < 0 or observation_count > MAX_OBSERVATIONS:
        raise ValueError("observation count is invalid")
    if decision == "accept_stop":
        if observation_count < 1:
            raise ValueError("search cannot stop before an observation")
        return "evidence_certified"
    return "forced_stop" if observation_count == MAX_OBSERVATIONS else "continue"


def _option_lines(row: Mapping[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("row options are invalid")
    lines: list[str] = []
    for entry in options:
        if not isinstance(entry, Mapping):
            raise ValueError("row option is invalid")
        option_id = entry.get("option_id")
        text = entry.get("text")
        if option_id not in OPTION_IDS or not isinstance(text, str) or not text.strip():
            raise ValueError("row option is invalid")
        lines.append(f"{option_id}. {text.strip()}")
    if [line[0] for line in lines] != list(OPTION_IDS):
        raise ValueError("row options must be in A-D order")
    return "\n".join(lines)


def initializer_prompt(row: Mapping[str, Any], protected_ranking: Sequence[str]) -> str:
    return "\n".join(
        [
            "Inspect the complete native video and audio.",
            "Build a literal evidence ledger for every answer A-D before deciding.",
            "Separate visible/heard observations from interpretation.",
            "Do not discard an option merely because it is initially unlikely.",
            f"Protected direct ranking (fallback only): {' > '.join(protected_ranking)}",
            f"Question: {_text(row.get('question'), 'question', 2000)}",
            "Options:",
            _option_lines(row),
        ]
    )


def planner_prompt(
    row: Mapping[str, Any], notebook: Mapping[str, Any], round_index: int
) -> str:
    if round_index not in {1, 2, 3}:
        raise ValueError("planner round is invalid")
    return "\n".join(
        [
            "Choose one native-video observation that best distinguishes the live A-D claims.",
            "The neutral_request will be shown to an observer with no question or options.",
            "Make neutral_request descriptive and do not include option IDs or answer text.",
            f"Round: {round_index}",
            f"Question: {_text(row.get('question'), 'question', 2000)}",
            "Options:",
            _option_lines(row),
            "Notebook:",
            json.dumps(notebook, ensure_ascii=False, sort_keys=True),
        ]
    )


def observer_prompt(action: MediaAction, *, original_time_origin_seconds: float) -> str:
    origin = _finite(original_time_origin_seconds, "original_time_origin_seconds")
    return "\n".join(
        [
            "Observe only the supplied native video/audio view.",
            "Report literal timestamped facts before interpretation.",
            "Transcribe visible text or speech exactly when the request calls for it.",
            f"Original timeline origin: {origin:.3f} seconds.",
            f"Neutral investigative request: {action.neutral_request}",
        ]
    )


def updater_prompt(row: Mapping[str, Any], notebook: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Map the newest blinded literal observations back onto every A-D claim.",
            "Preserve contrary evidence and do not delete earlier observations.",
            f"Question: {_text(row.get('question'), 'question', 2000)}",
            "Options:",
            _option_lines(row),
            "Notebook:",
            json.dumps(notebook, ensure_ascii=False, sort_keys=True),
        ]
    )


def critic_prompt(row: Mapping[str, Any], notebook: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Try to reject stopping. Accept only when literal evidence tests every viable option.",
            "Repeated model agreement is not evidence.",
            "Identify unexplained counterevidence and a feasible view that could change the ranking.",
            f"Question: {_text(row.get('question'), 'question', 2000)}",
            "Options:",
            _option_lines(row),
            "Notebook:",
            json.dumps(notebook, ensure_ascii=False, sort_keys=True),
        ]
    )


def judge_prompt(row: Mapping[str, Any], notebook: Mapping[str, Any]) -> str:
    literal: list[dict[str, Any]] = []
    interpretations: list[dict[str, Any]] = []
    initializer = notebook.get("initializer")
    if isinstance(initializer, Mapping):
        for item in initializer.get("literal_observations", []):
            if isinstance(item, Mapping):
                literal.append(
                    {
                        "observation_id": item.get("observation_id"),
                        "timestamp_seconds": item.get("timestamp_seconds"),
                        "literal_observation": item.get("literal_observation"),
                    }
                )
                interpretations.append(
                    {
                        "observation_id": item.get("observation_id"),
                        "interpretation": item.get("interpretation"),
                    }
                )
    rounds = notebook.get("rounds")
    if isinstance(rounds, list):
        for entry in rounds:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("observer"), Mapping):
                continue
            for item in entry["observer"].get("observations", []):
                if isinstance(item, Mapping):
                    literal.append(
                        {
                            "observation_id": item.get("observation_id"),
                            "timestamp_seconds": item.get("timestamp_seconds"),
                            "literal_observation": item.get("literal_observation"),
                        }
                    )
                    interpretations.append(
                        {
                            "observation_id": item.get("observation_id"),
                            "interpretation": item.get("interpretation"),
                        }
                    )
    return "\n".join(
        [
            "Solve the item afresh using the complete native video/audio and evidence notebook.",
            "Retain the protected top answer unless new blinded observer evidence both contradicts it and positively supports the replacement.",
            f"Question: {_text(row.get('question'), 'question', 2000)}",
            "Options:",
            _option_lines(row),
            f"Protected ranking: {' > '.join(notebook['protected_ranking'])}",
            "Literal observations:",
            json.dumps(literal, ensure_ascii=False, sort_keys=True),
            "Interpretations:",
            json.dumps(interpretations, ensure_ascii=False, sort_keys=True),
            "Complete notebook:",
            json.dumps(notebook, ensure_ascii=False, sort_keys=True),
        ]
    )


def media_action_dict(action: MediaAction) -> dict[str, Any]:
    return asdict(action)


def choose_final_ranking(
    judge: Mapping[str, Any] | None, protected_ranking: Sequence[str]
) -> tuple[list[str], str]:
    protected = _ranking(list(protected_ranking))
    if isinstance(judge, Mapping):
        return _ranking(judge.get("ranking")), "judge"
    return protected, "protected_direct_fallback"
