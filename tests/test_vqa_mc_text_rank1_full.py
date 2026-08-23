from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_vqa_mc_text_rank1_full as runner  # noqa: E402


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    task2 = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Question": "What happens?",
            "Options": [{"option": value} for value in ("Stops", "Fades", "Moves", "Waits")],
        },
        {
            "Q_ID": 2,
            "Video_ID": "v2",
            "Question": " How many? ",
            "Options": [{"option": value} for value in ("Zero", "One", "Two", "Three", "Four")],
        },
    ]
    fusion = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Answers": [
                {"Rank": rank, "Answer": "It fades" if rank == 1 else f"unused-{rank}", "Time": 1.0}
                for rank in range(1, 11)
            ],
        },
        {
            "Q_ID": 2,
            "Video_ID": "v2",
            "Answers": [
                {"Rank": rank, "Answer": "Two" if rank == 1 else f"unused-{rank}", "Time": 1.0}
                for rank in range(1, 11)
            ],
        },
    ]
    return {
        "task2": _write(tmp_path / "task2.json", task2),
        "fusion": _write(tmp_path / "fusion.json", fusion),
    }


def test_load_rows_uses_only_fusion_rank_one_and_supports_variable_options(
    inputs: dict[str, Path],
) -> None:
    rows = runner.load_rows(inputs["task2"], inputs["fusion"], expected_query_count=2)

    assert [len(row["options"]) for row in rows] == [4, 5]
    assert [row["rank1_answer"] for row in rows] == ["It fades", "Two"]
    assert "unused-2" not in json.dumps(rows)
    assert rows[1]["question"] == "How many?"


def test_load_rows_accepts_one_shot_task2_bytes(inputs: dict[str, Path]) -> None:
    task2_bytes = inputs["task2"].read_bytes()

    rows = runner.load_rows(task2_bytes, inputs["fusion"], expected_query_count=2)

    assert [row["query_id"] for row in rows] == [1, 2]


def test_query_limit_creates_a_matched_canary_subset(inputs: dict[str, Path]) -> None:
    rows = runner.load_rows(
        inputs["task2"], inputs["fusion"], expected_query_count=1, query_limit=1
    )

    assert [row["query_id"] for row in rows] == [1]


def test_build_request_pins_discounted_model_and_complete_option_indices(
    inputs: dict[str, Path],
) -> None:
    rows = runner.load_rows(inputs["task2"], inputs["fusion"], expected_query_count=2)
    request = runner.build_request(rows, batch_id=1)

    assert request["model"] == "google/gemini-3.7-flash"
    assert runner.BATCH_MODEL == "google/gemini-3.7-flash:batch"
    assert request["reasoning"]["effort"] == "high"
    assert "models" not in request
    assert "fallbacks" not in request
    body = json.loads(request["messages"][1]["content"])
    assert body["queries"][0]["prior_video_answer"] == "It fades"
    assert body["queries"][0]["options"] == [
        {"option_id": 1, "option": "Stops"},
        {"option_id": 2, "option": "Fades"},
        {"option_id": 3, "option": "Moves"},
        {"option_id": 4, "option": "Waits"},
    ]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True


def test_batch_envelope_uses_async_api_and_unique_request_ids(inputs: dict[str, Path]) -> None:
    rows = runner.load_rows(inputs["task2"], inputs["fusion"], expected_query_count=2)
    requests = [runner.build_request(rows, batch_id=1)]

    envelope = runner.build_batch_envelope(requests)

    assert envelope["endpoint"] == "/v1/chat/completions"
    assert envelope["model"] == "google/gemini-3.7-flash"
    assert envelope["requests"] == [{"custom_id": "batch-0001", "body": requests[0]}]
    assert ":batch" not in json.dumps(envelope["requests"])


def test_openrouter_success_payload_may_include_null_error() -> None:
    payload = {"id": "batch-1", "status": "validating", "error": None}

    assert runner.validate_openrouter_payload(payload) is payload


@pytest.mark.parametrize(
    "prompt, completion",
    [(float("nan"), 0.9375), (-1.0, 0.9375), (0.1875, float("inf")), (0.1875, -1.0)],
)
def test_discounted_pricing_rejects_nonfinite_or_negative_values(
    prompt: float, completion: float
) -> None:
    with pytest.raises(RuntimeError, match="pricing"):
        runner.validate_discounted_pricing(prompt, completion)


def test_terminal_batch_rejects_wrong_model_or_over_cap_cost() -> None:
    terminal = {
        "status": "completed",
        "model": runner.MODEL,
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "cost": 0.00013125,
        },
        "results": [
            {
                "custom_id": "batch-0001",
                "response": {
                    "status_code": 200,
                    "body": {"model": runner.MODEL, "choices": []},
                },
                "error": None,
            }
        ],
    }
    wrong_model = json.loads(json.dumps(terminal))
    wrong_model["results"][0]["response"]["body"]["model"] = "other/model"
    with pytest.raises(ValueError, match="model"):
        runner.validate_terminal_batch(wrong_model, expected_request_count=1, max_cost_usd=1.0)

    over_cap = json.loads(json.dumps(terminal))
    over_cap["usage"]["cost"] = 2.0
    with pytest.raises(ValueError, match="cost"):
        runner.validate_terminal_batch(over_cap, expected_request_count=1, max_cost_usd=1.0)

    discounted = json.loads(json.dumps(terminal))
    discounted["usage"]["cost"] /= 2
    assert runner.validate_terminal_batch(
        discounted, expected_request_count=1, max_cost_usd=1.0
    )["cost"] == pytest.approx(0.000065625)


def test_validate_and_organize_preserve_exact_released_option_strings(
    inputs: dict[str, Path],
) -> None:
    rows = runner.load_rows(inputs["task2"], inputs["fusion"], expected_query_count=2)
    payload = {
        "queries": [
            {"query_id": 1, "ranking": [2, 1, 4, 3]},
            {"query_id": 2, "ranking": [3, 2, 4, 1, 5]},
        ]
    }

    validated = runner.validate_response(payload, rows)
    submission = runner.organize_submission(rows, validated, elapsed_seconds={1: 0.4, 2: 0.4})

    assert submission[0] == {
        "Q_ID": 1,
        "Video_ID": "v1.webm",
        "Answers": [
            {"Rank": 1, "Answer": "Fades"},
            {"Rank": 2, "Answer": "Stops"},
            {"Rank": 3, "Answer": "Waits"},
            {"Rank": 4, "Answer": "Moves"},
        ],
    }
    released = {option for row in rows for option in row["options"]}
    assert all(answer["Answer"] in released for row in submission for answer in row["Answers"])
    assert "It fades" not in json.dumps(submission)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"queries": [{"query_id": 1, "ranking": [2, 2, 1, 3]}]}, "permutation"),
        ({"queries": [{"query_id": 1, "ranking": [2, 1, 3]}]}, "permutation"),
        ({"queries": [{"query_id": 9, "ranking": [2, 1, 4, 3]}]}, "identity"),
    ],
)
def test_validate_response_rejects_incomplete_or_wrong_rankings(
    inputs: dict[str, Path], payload: dict[str, Any], message: str
) -> None:
    row = runner.load_rows(inputs["task2"], inputs["fusion"], expected_query_count=2)[0]
    with pytest.raises(ValueError, match=message):
        runner.validate_response(payload, [row])


def test_dry_run_is_provider_free_and_cost_gated(inputs: dict[str, Path], tmp_path: Path) -> None:
    class Client:
        calls = 0

        def complete(self, _request: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("dry run must not call provider")

    client = Client()
    result = runner.run_full(
        task2=inputs["task2"],
        fusion=inputs["fusion"],
        output_root=tmp_path / "output",
        expected_query_count=2,
        max_cost_usd=1.0,
        client=client,
    )

    assert result["status"] == "dry_run"
    assert result["model"] == "google/gemini-3.7-flash:batch"
    assert result["estimated_cost_usd"] < 1.0
    assert client.calls == 0
    assert not (tmp_path / "output").exists()


def test_live_run_uses_validated_provider_payload(inputs: dict[str, Path], tmp_path: Path) -> None:
    class Client:
        def __init__(self) -> None:
            self.envelopes: list[dict[str, Any]] = []

        def complete_many(
            self, requests: list[dict[str, Any]], *, on_created: Any = None
        ) -> dict[str, Any]:
            self.envelopes.append(runner.build_batch_envelope(requests))
            if on_created is not None:
                on_created({"id": "batch-1", "status": "validating", "error": None})
            return {
                "id": "batch-1",
                "status": "completed",
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                    "cost": 0.00013125,
                },
                "results": [
                    {
                        "custom_id": "batch-0001",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": "generation-1",
                                "model": runner.MODEL,
                                "choices": [
                                    {
                                        "message": {
                                            "content": json.dumps(
                                                {
                                                    "queries": [
                                                        {"query_id": 1, "ranking": [2, 1, 4, 3]},
                                                        {"query_id": 2, "ranking": [3, 2, 4, 1, 5]},
                                                    ]
                                                }
                                            )
                                        }
                                    }
                                ],
                            },
                        },
                        "error": None,
                    }
                ],
            }

    client = Client()
    result = runner.run_full(
        task2=inputs["task2"],
        fusion=inputs["fusion"],
        output_root=tmp_path / "output",
        expected_query_count=2,
        max_cost_usd=1.0,
        run=True,
        client=client,
    )

    assert result["status"] == "complete"
    assert len(client.envelopes) == 1
    created = json.loads((tmp_path / "output" / "openrouter-batch-created.json").read_text())
    assert created["id"] == "batch-1"
    submission = json.loads((tmp_path / "output" / "submission.json").read_text())
    assert submission[0]["Answers"][0]["Answer"] == "Fades"


def test_resume_uses_frozen_batch_id_without_second_dispatch(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    terminal = {
        "id": "batch-resume",
        "model": runner.MODEL,
        "status": "completed",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "cost": 0.00013125,
        },
        "results": [
            {
                "custom_id": "batch-0001",
                "response": {
                    "status_code": 200,
                    "body": {
                        "model": runner.MODEL,
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "queries": [
                                                {"query_id": 1, "ranking": [2, 1, 4, 3]},
                                                {"query_id": 2, "ranking": [3, 2, 4, 1, 5]},
                                            ]
                                        }
                                    )
                                }
                            }
                        ],
                    },
                },
                "error": None,
            }
        ],
    }

    class InterruptedClient:
        def complete_many(self, requests: list[dict[str, Any]], *, on_created: Any) -> Any:
            on_created({"id": "batch-resume", "status": "validating", "error": None})
            raise RuntimeError("transient polling failure")

    root = tmp_path / "resume-output"
    with pytest.raises(RuntimeError, match="polling"):
        runner.run_full(
            task2=inputs["task2"], fusion=inputs["fusion"], output_root=root,
            expected_query_count=2, max_cost_usd=1.0, run=True, client=InterruptedClient(),
        )

    class ResumeClient(runner.OpenRouterClient):
        creates = 0
        waits: list[str] = []

        def __init__(self) -> None:
            pass

        def assert_discounted_price(self) -> dict[str, float]:
            raise AssertionError("resume must use the pricing frozen before dispatch")

        def complete_many(self, *_args: Any, **_kwargs: Any) -> Any:
            self.creates += 1
            raise AssertionError("resume must not create another batch")

        def wait_batch(self, batch_id: str) -> dict[str, Any]:
            self.waits.append(batch_id)
            return terminal

    client = ResumeClient()
    result = runner.run_full(
        task2=inputs["task2"], fusion=inputs["fusion"], output_root=root,
        expected_query_count=2, max_cost_usd=1.0, run=True, resume=True, client=client,
    )

    assert result["status"] == "complete"
    assert client.creates == 0
    assert client.waits == ["batch-resume"]

    # A crash after some or all accepted files were written remains recoverable;
    # the new invocation's wall time is not part of ranking identity.
    repeated = runner.run_full(
        task2=inputs["task2"], fusion=inputs["fusion"], output_root=root,
        expected_query_count=2, max_cost_usd=1.0, run=True, resume=True, client=client,
    )
    assert repeated["submission_sha256"] == result["submission_sha256"]
    assert client.creates == 0


def test_private_api_key_uses_approved_0600_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_env = tmp_path / "openrouter.env"
    private_env.write_text("OPENROUTER_API_KEY='private-test-key'\n", encoding="utf-8")
    private_env.chmod(0o600)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(runner, "PRIVATE_ENV_FILE", private_env)

    assert runner.private_api_key() == "private-test-key"
