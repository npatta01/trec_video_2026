from __future__ import annotations

import math

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_deliberative_agent import (
    JUDGE_SCHEMA,
    OBSERVER_SCHEMA,
    PLANNER_SCHEMA,
    SOLVER_SCHEMA,
    MediaAction,
    append_observer,
    append_planner,
    choose_final_ranking,
    initialize_notebook,
    judge_prompt,
    notebook_observation_ids,
    observer_prompt,
    parse_judge,
    parse_observer,
    parse_planner,
    parse_solver,
    planner_prompt,
    solver_prompt,
    validate_action,
)
from trec_video_2026.vqa_experiments.vqa_mc_common import canonical_json


def _row() -> dict[str, object]:
    return {
        "video_id": "video-a",
        "question": "How many times does the person jump?",
        "duration_seconds": 6.0,
        "options": [
            {"option_id": option_id, "text": text}
            for option_id, text in zip(
                "ABCD",
                ("once", "twice", "three times", "four times"),
                strict=True,
            )
        ],
    }


def _hypotheses() -> list[dict[str, str]]:
    return [
        {
            "option_id": option_id,
            "case_for": f"Evidence for {option_id}.",
            "case_against": f"Evidence against {option_id}.",
        }
        for option_id in "ABCD"
    ]


def _solver_payload(ranking: str = "ABCD") -> dict[str, object]:
    return {
        "ranking": list(ranking),
        "hypotheses": _hypotheses(),
        "observations": [
            {
                "timestamp_seconds": 2.0,
                "literal_observation": "The hand touches the ball.",
                "interpretation": "This may be a release.",
            }
        ],
        "strongest_challenge": "The contact may occur outside this moment.",
        "unresolved_questions": ["Was there another release later?"],
        "candidate_checks": ["Review the complete sequence."],
    }


def _solver_record(solver_id: str, ranking: str = "ABCD") -> dict[str, object]:
    return parse_solver(
        _solver_payload(ranking), duration_seconds=6.0, solver_id=solver_id
    )


def _action_payload(kind: str = "inspect_interval") -> dict[str, object]:
    return {
        "kind": kind,
        "start_seconds": 1.0 if kind == "inspect_interval" else 0.0,
        "end_seconds": 4.0 if kind == "inspect_interval" else 6.0,
        "fps": 8.0,
        "resolution": "high",
        "investigative_focus": "Describe every contact and release literally.",
        "expectations": [
            {"option_id": "A", "expected_observation": "One release is visible."},
            {"option_id": "B", "expected_observation": "Two releases are visible."},
        ],
    }


def _action() -> MediaAction:
    return validate_action(_action_payload(), duration_seconds=6.0)


def _planner_payload(
    *, decision: str = "inspect", ranking: str = "BACD"
) -> dict[str, object]:
    return {
        "ranking": list(ranking),
        "option_assessments": [
            {
                "option_id": option_id,
                "case_summary": f"Current case for {option_id}.",
                "supporting_observation_ids": ["solver-1-o1"],
                "opposing_observation_ids": [],
            }
            for option_id in "ABCD"
        ],
        "remaining_uncertainty": "Whether another release occurs later.",
        "decision": decision,
        "action": _action_payload() if decision == "inspect" else None,
    }


def _observer_payload(timestamp_seconds: float = 2.5) -> dict[str, object]:
    return {
        "observations": [
            {
                "timestamp_seconds": timestamp_seconds,
                "literal_observation": "The ball leaves the hand once.",
                "interpretation": "This is one completed release.",
            }
        ],
        "view_limitations": "Only the requested interval was visible.",
        "focus_resolution": "present",
    }


def _judge_payload(
    *, ranking: str = "BACD", citations: list[str] | None = None
) -> dict[str, object]:
    return {
        "ranking": list(ranking),
        "decisive_observation_ids": (
            ["round-1-o1"] if citations is None else citations
        ),
        "strongest_counterargument": "The earlier contact may not be a release.",
        "counterargument_resolution": "The full sequence confirms separation.",
        "ambiguity_present": False,
        "ambiguity_note": "No unresolved ambiguity.",
        "rationale": "The cited event best supports option B.",
    }


def _budget() -> dict[str, object]:
    return {
        "calls_used": 2,
        "calls_remaining": 5,
        "observer_input_tokens_remaining": 40_000,
        "final_call_reserved": True,
        "cost_cap_usd": 0.75,
    }


def test_schemas_are_strict_and_role_specific() -> None:
    assert SOLVER_SCHEMA["required"] == [
        "ranking",
        "hypotheses",
        "observations",
        "strongest_challenge",
        "unresolved_questions",
        "candidate_checks",
    ]
    assert PLANNER_SCHEMA["required"] == [
        "ranking",
        "option_assessments",
        "remaining_uncertainty",
        "decision",
        "action",
    ]
    assert OBSERVER_SCHEMA["required"] == [
        "observations",
        "view_limitations",
        "focus_resolution",
    ]
    assert JUDGE_SCHEMA["required"] == [
        "ranking",
        "decisive_observation_ids",
        "strongest_counterargument",
        "counterargument_resolution",
        "ambiguity_present",
        "ambiguity_note",
        "rationale",
    ]
    assert all(
        schema["additionalProperties"] is False
        for schema in (SOLVER_SCHEMA, PLANNER_SCHEMA, OBSERVER_SCHEMA, JUDGE_SCHEMA)
    )


def test_solver_requires_all_hypotheses_and_separates_observation() -> None:
    parsed = parse_solver(
        _solver_payload(), duration_seconds=6.0, solver_id="solver_1"
    )

    assert parsed["solver_id"] == "solver_1"
    assert [row["option_id"] for row in parsed["hypotheses"]] == list("ABCD")
    assert parsed["observations"][0]["observation_id"] == "solver-1-o1"
    assert (
        parsed["observations"][0]["literal_observation"]
        == "The hand touches the ball."
    )
    assert parsed["observations"][0]["interpretation"] == "This may be a release."


def test_solver_rejects_duplicate_or_missing_hypothesis() -> None:
    payload = _solver_payload()
    payload["hypotheses"] = [*_hypotheses()[:3], _hypotheses()[0]]

    with pytest.raises(ValueError, match="hypotheses"):
        parse_solver(payload, duration_seconds=6.0, solver_id="solver_1")


def test_option_keyed_rows_are_normalized_from_model_order() -> None:
    solver_payload = _solver_payload()
    solver_payload["hypotheses"] = list(reversed(solver_payload["hypotheses"]))
    planner_payload = _planner_payload(decision="finish_search")
    planner_payload["option_assessments"] = list(
        reversed(planner_payload["option_assessments"])
    )

    solver = parse_solver(
        solver_payload, duration_seconds=6.0, solver_id="solver_1"
    )
    planner = parse_planner(planner_payload, duration_seconds=6.0)

    assert [row["option_id"] for row in solver["hypotheses"]] == list("ABCD")
    assert [row["option_id"] for row in planner["option_assessments"]] == list(
        "ABCD"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"start_seconds": math.nan},
        {"start_seconds": -0.1},
        {"end_seconds": 6.1},
        {"start_seconds": 4.0, "end_seconds": 4.0},
        {"fps": 2.0},
        {"resolution": "ultra"},
        {"investigative_focus": "   "},
    ],
)
def test_validate_action_rejects_invalid_mechanics(updates: dict[str, object]) -> None:
    payload = _action_payload()
    payload.update(updates)

    with pytest.raises(ValueError):
        validate_action(payload, duration_seconds=6.0)


def test_rewatch_full_must_use_exact_complete_interval() -> None:
    payload = _action_payload("rewatch_full")
    payload["end_seconds"] = 5.0

    with pytest.raises(ValueError, match="complete"):
        validate_action(payload, duration_seconds=6.0)


def test_planner_finish_has_no_action_and_inspect_requires_action() -> None:
    finished = parse_planner(_planner_payload(decision="finish_search"), duration_seconds=6.0)
    assert finished["decision"] == "finish_search"
    assert finished["action"] is None

    invalid = _planner_payload()
    invalid["action"] = None
    with pytest.raises(ValueError, match="action"):
        parse_planner(invalid, duration_seconds=6.0)


def test_observer_timestamps_must_be_inside_realized_view() -> None:
    with pytest.raises(ValueError, match="view"):
        parse_observer(
            _observer_payload(timestamp_seconds=4.5),
            action=_action(),
            duration_seconds=6.0,
            view_id="view-1",
        )


def test_judge_citations_must_exist() -> None:
    with pytest.raises(ValueError, match="observation ID"):
        parse_judge(
            _judge_payload(citations=["missing"]),
            observation_ids={"solver-1-o1"},
        )


def test_blind_solver_prompt_has_no_anchor_state() -> None:
    prompt = solver_prompt(_row())

    assert "anchor" not in prompt.casefold()
    assert "A. once" in prompt
    assert "D. four times" in prompt


def test_observer_prompt_contains_focus_but_no_question_options_or_rankings() -> None:
    action = _action()
    prompt = observer_prompt(action)

    assert action.investigative_focus in prompt
    assert str(_row()["question"]) not in prompt
    assert "A. once" not in prompt
    assert "ranking" not in prompt.casefold()


def test_planner_prompt_contains_notebook_and_budget() -> None:
    notebook = initialize_notebook(
        "video-a", [_solver_record("solver_1"), _solver_record("solver_2", "BACD")]
    )

    prompt = planner_prompt(_row(), notebook, _budget())

    assert "solver_1" in prompt
    assert "observer_input_tokens_remaining" in prompt
    assert "40,000" not in prompt


def test_append_observer_does_not_mutate_prior_notebook() -> None:
    original = initialize_notebook(
        "video-a", [_solver_record("solver_1"), _solver_record("solver_2")]
    )
    with_planner = append_planner(
        original,
        round_index=1,
        planner=parse_planner(_planner_payload(), duration_seconds=6.0),
    )
    before = canonical_json(with_planner)
    observer = parse_observer(
        _observer_payload(),
        action=_action(),
        duration_seconds=6.0,
        view_id="view-1",
    )

    updated = append_observer(
        with_planner,
        round_index=1,
        action=_action(),
        observer=observer,
        provenance={"view_id": "view-1", "fps": 8.0, "resolution": "high"},
    )

    assert canonical_json(with_planner) == before
    assert updated["rounds"][0]["observer"]["observations"][0]["observation_id"] == "round-1-o1"
    assert "round-1-o1" in notebook_observation_ids(updated)


def test_judge_prompt_renders_literal_observations_before_interpretations() -> None:
    notebook = initialize_notebook(
        "video-a", [_solver_record("solver_1"), _solver_record("solver_2")]
    )

    prompt = judge_prompt(_row(), notebook)

    assert prompt.index("Literal observations") < prompt.index("Interpretations")


def test_final_selection_prefers_judge_then_latest_planner_then_solver() -> None:
    solver = _solver_record("solver_1", "ABCD")
    planner = parse_planner(
        _planner_payload(decision="finish_search", ranking="BACD"),
        duration_seconds=6.0,
    )
    judge = parse_judge(_judge_payload(ranking="CABD", citations=[]), observation_ids=set())

    assert choose_final_ranking(judge, [planner], [solver]) == (list("CABD"), "judge")
    assert choose_final_ranking(None, [planner], [solver]) == (list("BACD"), "planner_fallback")
    assert choose_final_ranking(None, [], [solver]) == (list("ABCD"), "solver_fallback")
    assert choose_final_ranking(None, [], []) == (None, "invalid")
