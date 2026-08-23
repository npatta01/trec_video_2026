from __future__ import annotations

import copy

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_four_option_agent import (
    MediaAction,
    append_round_stage,
    choose_final_ranking,
    initialize_notebook,
    judge_prompt,
    next_search_state,
    notebook_observation_ids,
    observer_prompt,
    parse_critic,
    parse_initializer,
    parse_judge,
    parse_observer,
    parse_planner,
    parse_update,
)


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


def _ledger() -> list[dict[str, object]]:
    return [
        {
            "option_id": option_id,
            "observable_claim": f"The clip contains the count represented by {option_id}.",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "untested_claims": ["A gapless count is still needed."],
            "status": "viable",
            "interpretation": f"Option {option_id} remains possible.",
        }
        for option_id in "ABCD"
    ]


def _initializer_payload() -> dict[str, object]:
    return {
        "provisional_ranking": list("BACD"),
        "ledger": _ledger(),
        "literal_observations": [
            {
                "timestamp_seconds": 2.0,
                "literal_observation": "The person leaves the floor once.",
                "interpretation": "This may be one completed jump.",
            }
        ],
        "unresolved_distinctions": ["Whether another jump occurs later."],
    }


def _initializer() -> dict[str, object]:
    return parse_initializer(
        _initializer_payload(),
        duration_seconds=6.0,
        protected_ranking=list("BACD"),
    )


def _action_payload() -> dict[str, object]:
    return {
        "kind": "inspect_interval",
        "start_seconds": 1.0,
        "end_seconds": 4.0,
        "fps": 8.0,
        "resolution": "high",
        "neutral_request": (
            "Describe every completed departure from and return to the ground."
        ),
        "tested_option_ids": list("ABCD"),
        "expected_outcomes": [
            {
                "option_id": option_id,
                "literal_outcome": f"The corresponding number of cycles is visible for {option_id}.",
            }
            for option_id in "ABCD"
        ],
    }


def _action() -> MediaAction:
    return parse_planner(
        {
            "provisional_ranking": list("BACD"),
            "unresolved_distinction": "The number of completed jump cycles.",
            "action": _action_payload(),
            "why_discriminating": "A gapless count separates all four choices.",
        },
        duration_seconds=6.0,
        round_index=1,
    )["action"]


def _observer_payload() -> dict[str, object]:
    return {
        "observations": [
            {
                "timestamp_seconds": 2.5,
                "literal_observation": "One departure and landing are visible.",
                "interpretation": "This is one completed cycle.",
            }
        ],
        "exact_transcription": "No speech or text was requested.",
        "focus_resolution": "present",
        "view_limitations": "The interval excludes the first and last second.",
    }


def _notebook_with_observation(
    *, protected_ranking: list[str] | None = None
) -> dict[str, object]:
    notebook = initialize_notebook(
        "video-a", protected_ranking or list("BACD"), _initializer()
    )
    planner = parse_planner(
        {
            "provisional_ranking": list("BACD"),
            "unresolved_distinction": "The number of completed jump cycles.",
            "action": _action_payload(),
            "why_discriminating": "A gapless count separates all four choices.",
        },
        duration_seconds=6.0,
        round_index=1,
    )
    notebook = append_round_stage(
        notebook, round_index=1, stage="planner", record=planner
    )
    observer = parse_observer(
        _observer_payload(),
        action=planner["action"],
        duration_seconds=6.0,
        view_id="view-1",
    )
    return append_round_stage(
        notebook, round_index=1, stage="observer", record=observer
    )


def _critic_payload(decision: str) -> dict[str, object]:
    return {
        "decision": decision,
        "tested_viable_option_ids": list("ABCD"),
        "leader_evidence_ids": ["round-1-o1"],
        "unexplained_counterevidence_ids": [],
        "next_unresolved_distinction": (
            "No feasible observation remains."
            if decision == "accept_stop"
            else "Inspect the complete clip for additional cycles."
        ),
        "direct_replacement_supported": False,
        "reason": "The literal evidence now distinguishes the choices.",
    }


def _judge_payload(ranking: str = "BACD") -> dict[str, object]:
    return {
        "ranking": list(ranking),
        "decisive_evidence_ids": ["round-1-o1"],
        "strongest_counterevidence": "The view excludes part of the clip.",
        "counterevidence_resolution": "The complete judge view resolves the gap.",
        "protected_answer_action": "retained" if ranking[0] == "B" else "replaced",
        "direct_contradiction_evidence_ids": [],
        "replacement_support_evidence_ids": [],
        "stop_status": "evidence_certified",
        "rationale": "The cited literal evidence supports the final ranking.",
    }


def test_initializer_requires_exactly_one_ledger_row_for_each_option() -> None:
    parsed = _initializer()
    assert [row["option_id"] for row in parsed["ledger"]] == list("ABCD")

    duplicate = _initializer_payload()
    duplicate["ledger"][3]["option_id"] = "A"
    with pytest.raises(ValueError, match="each option exactly once"):
        parse_initializer(
            duplicate,
            duration_seconds=6.0,
            protected_ranking=list("BACD"),
        )


def test_planner_action_requires_expected_outcomes_for_every_tested_option() -> None:
    payload = {
        "provisional_ranking": list("BACD"),
        "unresolved_distinction": "The number of completed jump cycles.",
        "action": _action_payload(),
        "why_discriminating": "A gapless count separates all four choices.",
    }
    payload["action"]["expected_outcomes"].pop()

    with pytest.raises(ValueError, match="tested option"):
        parse_planner(payload, duration_seconds=6.0, round_index=1)


def test_critic_cannot_accept_stop_before_a_valid_blinded_observation() -> None:
    notebook = initialize_notebook("video-a", list("BACD"), _initializer())
    with pytest.raises(ValueError, match="before a valid blinded observation"):
        parse_critic(
            _critic_payload("accept_stop"),
            notebook=notebook,
            observation_count=0,
        )


def test_critic_accepts_stop_only_after_testing_every_viable_option() -> None:
    notebook = _notebook_with_observation()
    payload = _critic_payload("accept_stop")
    payload["tested_viable_option_ids"] = ["A", "B"]
    with pytest.raises(ValueError, match="all viable options"):
        parse_critic(payload, notebook=notebook, observation_count=1)


def test_third_rejected_stop_records_forced_stop() -> None:
    assert next_search_state(decision="continue", observation_count=3) == "forced_stop"
    assert next_search_state(decision="continue", observation_count=2) == "continue"
    assert (
        next_search_state(decision="accept_stop", observation_count=1)
        == "evidence_certified"
    )


def test_observer_prompt_contains_only_neutral_request_and_timeline() -> None:
    prompt = observer_prompt(_action(), original_time_origin_seconds=0.0)
    prohibited = [
        str(_row()["question"]),
        "A. once",
        "B. twice",
        "protected answer",
        "ranking",
        "expected_outcomes",
        "option_id",
    ]
    assert all(value.casefold() not in prompt.casefold() for value in prohibited)
    assert "Describe every completed departure" in prompt


def test_observer_rejects_timestamps_outside_realized_view() -> None:
    payload = _observer_payload()
    payload["observations"][0]["timestamp_seconds"] = 5.0
    with pytest.raises(ValueError, match="realized view"):
        parse_observer(
            payload,
            action=_action(),
            duration_seconds=6.0,
            view_id="view-1",
        )


def test_update_is_append_only_and_requires_known_observation_ids() -> None:
    notebook = _notebook_with_observation()
    before = copy.deepcopy(notebook)
    payload = {
        "provisional_ranking": list("ABCD"),
        "ledger": [
            {
                **row,
                "supporting_evidence_ids": ["round-1-o1"]
                if row["option_id"] == "A"
                else [],
            }
            for row in _ledger()
        ],
        "mapped_observation_ids": ["round-1-o1"],
        "unresolved_distinctions": [],
    }
    update = parse_update(payload, notebook=notebook)
    updated = append_round_stage(
        notebook, round_index=1, stage="updater", record=update
    )
    assert notebook == before
    assert updated["rounds"][0]["updater"]["provisional_ranking"] == list("ABCD")

    payload["mapped_observation_ids"] = ["missing-o1"]
    with pytest.raises(ValueError, match="unknown observation"):
        parse_update(payload, notebook=notebook)


def test_judge_replacement_requires_new_contradiction_and_positive_support() -> None:
    notebook = _notebook_with_observation(protected_ranking=list("BACD"))
    payload = _judge_payload(ranking="ABCD")
    with pytest.raises(ValueError, match="contradiction evidence"):
        parse_judge(payload, notebook=notebook, protected_ranking=list("BACD"))

    payload["direct_contradiction_evidence_ids"] = ["round-1-o1"]
    with pytest.raises(ValueError, match="positive support evidence"):
        parse_judge(payload, notebook=notebook, protected_ranking=list("BACD"))

    payload["replacement_support_evidence_ids"] = ["round-1-o1"]
    parsed = parse_judge(
        payload, notebook=notebook, protected_ranking=list("BACD")
    )
    assert parsed["protected_answer_action"] == "replaced"


def test_judge_citations_must_exist_and_observer_evidence_is_identifiable() -> None:
    notebook = _notebook_with_observation()
    assert "round-1-o1" in notebook_observation_ids(notebook)
    payload = _judge_payload()
    payload["decisive_evidence_ids"] = ["missing-o1"]
    with pytest.raises(ValueError, match="unknown evidence"):
        parse_judge(payload, notebook=notebook, protected_ranking=list("BACD"))


def test_judge_prompt_places_literal_observations_before_interpretations() -> None:
    prompt = judge_prompt(_row(), _notebook_with_observation())
    assert prompt.index("Literal observations") < prompt.index("Interpretations")


def test_final_selection_uses_only_valid_judge_or_protected_direct_fallback() -> None:
    notebook = _notebook_with_observation()
    judge = parse_judge(
        _judge_payload(), notebook=notebook, protected_ranking=list("BACD")
    )
    assert choose_final_ranking(judge, list("BACD")) == (list("BACD"), "judge")
    assert choose_final_ranking(None, list("BACD")) == (
        list("BACD"),
        "protected_direct_fallback",
    )
