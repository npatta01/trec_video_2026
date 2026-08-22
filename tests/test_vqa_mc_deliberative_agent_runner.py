from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_deliberative_agent import (
    append_planner,
    initialize_notebook,
    parse_planner,
    parse_solver,
    validate_action,
)


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_deliberative_runner",
    ROOT / "scripts/run_vqa_mc_gemini_deliberative_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _row() -> dict[str, Any]:
    return {
        "query_id": "video-a",
        "video_id": "video-a",
        "stratum": "counting",
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


def _upload() -> dict[str, str]:
    return {
        "file_uri": "https://provider.invalid/video-a",
        "mime_type": "video/webm",
    }


def _solver_payload(ranking: str = "ABCD") -> dict[str, object]:
    return {
        "ranking": list(ranking),
        "hypotheses": [
            {
                "option_id": option_id,
                "case_for": f"Evidence for {option_id}.",
                "case_against": f"Evidence against {option_id}.",
            }
            for option_id in "ABCD"
        ],
        "observations": [
            {
                "timestamp_seconds": 2.0,
                "literal_observation": "The person leaves the floor.",
                "interpretation": "This is one jump.",
            }
        ],
        "strongest_challenge": "Another jump may occur later.",
        "unresolved_questions": ["Does another jump occur?"],
        "candidate_checks": ["Review the remainder."],
    }


def _solver_record(solver_id: str, ranking: str = "ABCD") -> dict[str, object]:
    return parse_solver(
        _solver_payload(ranking), duration_seconds=6.0, solver_id=solver_id
    )


def _planner_payload(decision: str = "finish_search") -> dict[str, object]:
    action = None
    if decision == "inspect":
        action = {
            "kind": "inspect_interval",
            "start_seconds": 1.0,
            "end_seconds": 4.0,
            "fps": 8.0,
            "resolution": "high",
            "investigative_focus": "Describe each time the person leaves the floor.",
            "expectations": [
                {"option_id": "A", "expected_observation": "One jump."},
                {"option_id": "B", "expected_observation": "Two jumps."},
            ],
        }
    return {
        "ranking": list("ABCD"),
        "option_assessments": [
            {
                "option_id": option_id,
                "case_summary": f"Case for {option_id}.",
                "supporting_observation_ids": ["solver-1-o1"],
                "opposing_observation_ids": [],
            }
            for option_id in "ABCD"
        ],
        "remaining_uncertainty": "Whether another jump occurs.",
        "decision": decision,
        "action": action,
    }


def _notebook() -> dict[str, object]:
    return initialize_notebook(
        "video-a", [_solver_record("solver_1"), _solver_record("solver_2", "BACD")]
    )


def _budget() -> dict[str, object]:
    return {
        "calls_used": 2,
        "calls_remaining": 5,
        "observer_input_tokens_remaining": 40_000,
        "final_call_reserved": True,
        "cost_cap_usd": 0.75,
    }


def _request_payload(tmp_path: Path, *, include_label: bool = False) -> Path:
    row = _row()
    if include_label:
        row["correct_option_id"] = "A"
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "test-deliberative-agent",
                "rows": [row],
                "variants": [
                    {
                        "variant_id": "video-a:canonical",
                        "video_id": "video-a",
                        "variant_kind": "canonical",
                        "display_order": list("ABCD"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _historical_root(tmp_path: Path) -> Path:
    root = tmp_path / "historical"
    delivery = root / "deliveries/video-a/delivery-media/video-a.webm"
    delivery.parent.mkdir(parents=True)
    delivery.write_bytes(b"fake video")
    return root


def test_solver_and_judge_receive_complete_low_4fps_video() -> None:
    solver = runner.build_solver_request(
        row=_row(), upload=_upload(), stage="anchor"
    )
    judge = runner.build_judge_request(
        row=_row(), upload=_upload(), notebook=_notebook()
    )

    for request in (solver, judge):
        video = request.contents[0].parts[0]
        assert video.video_metadata.fps == 4.0
        assert video.video_metadata.start_offset is None
        assert video.video_metadata.end_offset is None
        assert video.media_resolution.level.name.endswith("LOW")
        assert request.config.thinking_config.thinking_level.name.endswith("HIGH")


def test_planner_is_text_only_and_observer_is_blinded() -> None:
    planner = runner.build_planner_request(
        row=_row(), notebook=_notebook(), budget=_budget(), round_index=1
    )
    action = validate_action(
        _planner_payload("inspect")["action"], duration_seconds=6.0
    )
    observer = runner.build_observer_request(upload=_upload(), action=action)

    assert all(
        getattr(part, "file_data", None) is None
        for part in planner.contents[0].parts
    )
    rendered = "\n".join(part.text or "" for part in observer.contents[0].parts)
    assert _row()["question"] not in rendered
    assert "A. once" not in rendered
    assert "ranking" not in rendered.casefold()


def test_observer_uses_requested_interval_fps_and_resolution() -> None:
    action = validate_action(
        _planner_payload("inspect")["action"], duration_seconds=6.0
    )
    request = runner.build_observer_request(upload=_upload(), action=action)
    video = request.contents[0].parts[0]

    assert video.video_metadata.start_offset == "1.000s"
    assert video.video_metadata.end_offset == "4.000s"
    assert video.video_metadata.fps == 8.0
    assert video.media_resolution.level.name.endswith("HIGH")


def test_preflight_rejects_labels_and_cost_above_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label"):
        runner.load_requests(
            _request_payload(tmp_path, include_label=True),
            expected_rows=None,
            expected_variants=None,
        )

    with pytest.raises(ValueError, match="cap"):
        runner.preflight(
            requests_path=_request_payload(tmp_path),
            historical_root=_historical_root(tmp_path),
            max_cost_usd=0.000001,
            expected_rows=None,
            expected_variants=None,
        )


def test_cost_estimate_reserves_seven_calls_and_final_judge(tmp_path: Path) -> None:
    payload = runner.load_requests(
        _request_payload(tmp_path), expected_rows=None, expected_variants=None
    )
    estimate = runner.estimate_worst_case_cost(payload)

    assert estimate["row_count"] == 1
    assert estimate["request_count_ceiling"] == 7
    assert estimate["max_calls_per_row"] == 7
    assert estimate["final_call_reserved"] is True
    assert 0 < estimate["estimated_cost_usd"] < 0.75


def test_action_authorization_consumes_shared_observer_budget() -> None:
    action = validate_action(
        _planner_payload("inspect")["action"], duration_seconds=6.0
    )
    estimated = runner.estimate_action_input_tokens(action)

    assert estimated == 3.0 * runner.MEDIA_TOKENS_PER_SECOND[(8.0, "high")]
    assert runner.authorize_action(
        action,
        calls_used=3,
        observer_input_tokens_remaining=int(estimated),
    ) == {
        "authorized": True,
        "estimated_input_tokens": int(estimated),
        "remaining_input_tokens": 0,
    }
    denied = runner.authorize_action(
        action,
        calls_used=6,
        observer_input_tokens_remaining=40_000,
    )
    assert denied["authorized"] is False
    assert denied["reason"] == "final_call_reserve"


def test_notebook_with_planner_is_json_serializable() -> None:
    notebook = append_planner(
        _notebook(),
        round_index=1,
        planner=parse_planner(_planner_payload("inspect"), duration_seconds=6.0),
    )

    json.dumps(notebook)
