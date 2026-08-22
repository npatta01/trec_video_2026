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
from trec_video_2026.vqa_experiments.vqa_mc_poc20 import sha256_file


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
            "end_seconds": 3.0,
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
    assert video.video_metadata.end_offset == "3.000s"
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
    assert runner.ROLE_OUTPUT_LIMITS == {
        "solver": 2048,
        "planner": 1536,
        "observer": 1536,
        "judge": 2048,
    }
    assert estimate["prior_failed_run_cost_usd"] == pytest.approx(0.0952995)
    assert 0 < estimate["estimated_cost_usd"] < 0.75
    assert estimate["combined_estimated_cost_usd"] < 0.75


def test_action_authorization_consumes_shared_observer_budget() -> None:
    action = validate_action(
        _planner_payload("inspect")["action"], duration_seconds=6.0
    )
    estimated = runner.estimate_action_input_tokens(action)

    assert estimated == 2.0 * runner.MEDIA_TOKENS_PER_SECOND[(8.0, "high")]
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


class _Models:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.requests.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.models = _Models(responses)


def _provider_response(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": json.dumps(value),
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "totalTokenCount": 110,
        },
    }


def _observer_payload(timestamp_seconds: float = 2.5) -> dict[str, object]:
    return {
        "observations": [
            {
                "timestamp_seconds": timestamp_seconds,
                "literal_observation": "The person leaves the floor once.",
                "interpretation": "This is a completed jump.",
            }
        ],
        "view_limitations": "Only the requested view was available.",
        "focus_resolution": "present",
    }


def _judge_payload(citation: str) -> dict[str, object]:
    return {
        "ranking": list("ABCD"),
        "decisive_observation_ids": [citation],
        "strongest_counterargument": "Another jump might be hidden.",
        "counterargument_resolution": "The complete view shows no other jump.",
        "ambiguity_present": False,
        "ambiguity_note": "No unresolved ambiguity.",
        "rationale": "The cited observation and complete view support A.",
    }


def _path_responses(action_count: int) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = [
        _solver_payload("ABCD"),
        _solver_payload("BACD"),
    ]
    if action_count == 0:
        values.extend([_planner_payload("finish_search"), _judge_payload("solver-1-o1")])
    elif action_count == 1:
        values.extend(
            [
                _planner_payload("inspect"),
                _observer_payload(),
                _planner_payload("finish_search"),
                _judge_payload("round-1-o1"),
            ]
        )
    else:
        values.extend(
            [
                _planner_payload("inspect"),
                _observer_payload(),
                _planner_payload("inspect"),
                _observer_payload(),
                _judge_payload("round-2-o1"),
            ]
        )
    return [_provider_response(value) for value in values]


@pytest.mark.parametrize(
    ("action_count", "expected_calls"),
    [(0, 4), (1, 6), (2, 7)],
)
def test_run_supports_all_bounded_paths(
    tmp_path: Path, action_count: int, expected_calls: int
) -> None:
    client = _Client(_path_responses(action_count))
    output = tmp_path / f"output-{action_count}"
    uploads: list[Path] = []

    result = runner.run(
        requests_path=_request_payload(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=output,
        client=client,
        upload_file=lambda _client, path: uploads.append(path) or _upload(),
        expected_rows=None,
        expected_variants=None,
    )

    assert result["final_valid_count"] == 1
    assert result["observer_call_count"] == action_count
    assert result["request_count"] == expected_calls
    assert result["usage_complete"] is True
    assert result["prompt_tokens"] == expected_calls * 100
    assert result["output_tokens"] == expected_calls * 10
    assert len(client.models.requests) == expected_calls
    assert len(uploads) == 1
    assert (output / "anchor/raw-freeze.json").is_file()
    assert (output / "blind/raw-freeze.json").is_file()
    assert (output / "judge/raw-freeze.json").is_file()
    final = json.loads((output / "final-predictions-freeze.json").read_text())
    assert final["rows"][0]["ranking"] == list("ABCD")
    assert final["rows"][0]["final_source"] == "judge"
    notebook_path = output / final["rows"][0]["notebook_path"]
    assert final["rows"][0]["notebook_sha256"] == sha256_file(notebook_path)
    assert runner.verify_output(output)["status"] == "verified"


def test_invalid_judge_uses_explicit_planner_fallback(tmp_path: Path) -> None:
    responses = _path_responses(0)
    responses[-1] = _provider_response(_judge_payload("missing-observation"))
    output = tmp_path / "fallback-output"

    result = runner.run(
        requests_path=_request_payload(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=output,
        client=_Client(responses),
        upload_file=lambda _client, _path: _upload(),
        expected_rows=None,
        expected_variants=None,
    )

    assert result["final_valid_count"] == 1
    final = json.loads((output / "final-predictions-freeze.json").read_text())
    assert final["rows"][0]["final_source"] == "planner_fallback"
    assert final["rows"][0]["judge_status"] == "failed"


def test_invalid_planner_gets_one_correction_and_removes_second_observer(
    tmp_path: Path,
) -> None:
    responses = [
        _provider_response(_solver_payload("ABCD")),
        _provider_response(_solver_payload("BACD")),
        _provider_response({"invalid": "planner"}),
        _provider_response(_planner_payload("inspect")),
        _provider_response(_observer_payload()),
        _provider_response(_planner_payload("inspect")),
        _provider_response(_judge_payload("round-1-o1")),
    ]
    output = tmp_path / "corrected-planner-output"

    result = runner.run(
        requests_path=_request_payload(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=output,
        client=_Client(responses),
        upload_file=lambda _client, _path: _upload(),
        expected_rows=None,
        expected_variants=None,
    )

    assert result["request_count"] == 7
    assert result["observer_call_count"] == 1
    assert result["call_counts_by_role"]["planner_correction"] == 1
    notebook = json.loads((output / "notebooks/final/00.json").read_text())
    assert notebook["rounds"][1]["tool_result"]["authorized"] is False
    assert notebook["rounds"][1]["tool_result"]["reason"] == "final_call_reserve"


def test_verify_output_rejects_tampered_notebook(tmp_path: Path) -> None:
    output = tmp_path / "tampered-output"
    runner.run(
        requests_path=_request_payload(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=output,
        client=_Client(_path_responses(0)),
        upload_file=lambda _client, _path: _upload(),
        expected_rows=None,
        expected_variants=None,
    )
    notebook_path = output / "notebooks/final/00.json"
    notebook_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="notebook hash mismatch"):
        runner.verify_output(output)
