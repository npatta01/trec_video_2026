from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_vqa_mc_text_audit_full as audit  # noqa: E402
import run_vqa_mc_text_rank1_full as baseline  # noqa: E402


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
            "Video_ID": "v2.webm",
            "Question": "How many?",
            "Options": [{"option": value} for value in ("One", "Two", "Three", "Four")],
        },
    ]
    fusion = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Answers": [{"Rank": 1, "Answer": "It fades"}],
        },
        {
            "Q_ID": 2,
            "Video_ID": "v2.webm",
            "Answers": [{"Rank": 1, "Answer": "None (0)"}],
        },
    ]
    submission = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Answers": [
                {"Rank": 1, "Answer": "Fades"},
                {"Rank": 2, "Answer": "Stops"},
                {"Rank": 3, "Answer": "Waits"},
                {"Rank": 4, "Answer": "Moves"},
            ],
        },
        {
            "Q_ID": 2,
            "Video_ID": "v2.webm",
            "Answers": [
                {"Rank": 1, "Answer": "One"},
                {"Rank": 2, "Answer": "Two"},
                {"Rank": 3, "Answer": "Three"},
                {"Rank": 4, "Answer": "Four"},
            ],
        },
    ]
    return {
        "task2": _write(tmp_path / "task2.json", task2),
        "fusion": _write(tmp_path / "fusion.json", fusion),
        "submission": _write(tmp_path / "submission.json", submission),
    }


def test_load_rows_recovers_current_top_id_without_changing_released_options(
    inputs: dict[str, Path],
) -> None:
    rows = audit.load_audit_rows(
        inputs["task2"], inputs["fusion"], inputs["submission"], expected_query_count=2
    )

    assert [row["current_top_option_id"] for row in rows] == [2, 1]
    assert rows[0]["options"] == ["Stops", "Fades", "Moves", "Waits"]


def test_audit_request_is_blind_to_the_existing_submission(inputs: dict[str, Path]) -> None:
    rows = audit.load_audit_rows(
        inputs["task2"], inputs["fusion"], inputs["submission"], expected_query_count=2
    )

    request = audit.build_request(rows, batch_id=1)
    user_payload = json.loads(request["messages"][1]["content"])
    serialized = json.dumps(user_payload)

    assert request["model"] == baseline.MODEL
    assert user_payload["queries"][0]["prior_video_answer"] == "It fades"
    assert "current_top_option_id" not in serialized
    assert "current ranking" not in request["messages"][0]["content"].lower()
    query_schema = request["response_format"]["json_schema"]["schema"]["properties"]["queries"]["items"]
    assert set(query_schema["required"]) == {
        "query_id", "match_quality_code", "best_option_ids", "issue_codes",
        "requires_video_review",
    }
    assert "string" not in json.dumps(query_schema)


def test_validate_response_requires_review_for_partial_or_no_match(
    inputs: dict[str, Path],
) -> None:
    rows = audit.load_audit_rows(
        inputs["task2"], inputs["fusion"], inputs["submission"], expected_query_count=2
    )
    payload = {
        "queries": [
                {
                    "query_id": 1,
                    "match_quality_code": 2,
                    "best_option_ids": [2],
                    "issue_codes": [],
                    "requires_video_review": False,
                },
                {
                    "query_id": 2,
                    "match_quality_code": 4,
                    "best_option_ids": [1],
                    "issue_codes": [1],
                    "requires_video_review": True,
                },
        ]
    }

    accepted = audit.validate_response(payload, rows)
    records = audit.organize_audit(rows, accepted)

    assert records[0]["selected_top_is_best"] is True
    assert records[0]["review_required"] is False
    assert records[1]["selected_top_is_best"] is True
    assert records[1]["selected_top_supported"] is False
    assert records[1]["review_required"] is True

    invalid = json.loads(json.dumps(payload))
    invalid["queries"][1]["requires_video_review"] = False
    with pytest.raises(ValueError, match="requires video review"):
        audit.validate_response(invalid, rows)


def test_current_top_outside_blind_best_options_enters_review_queue(
    inputs: dict[str, Path],
) -> None:
    rows = audit.load_audit_rows(
        inputs["task2"], inputs["fusion"], inputs["submission"], expected_query_count=2
    )
    assessments = {
        1: {
            "match_quality": "exact",
            "best_option_ids": [1],
            "issue_types": ["semantic_mismatch"],
            "requires_video_review": False,
            "reason": "Option 1 is the direct answer.",
        },
        2: {
            "match_quality": "none",
            "best_option_ids": [1],
            "issue_types": ["numeric_answer_absent"],
            "requires_video_review": True,
            "reason": "No supported option.",
        },
    }

    records = audit.organize_audit(rows, assessments)
    queue = audit.review_queue(records)

    assert [row["Q_ID"] for row in queue] == [1, 2]
    assert records[0]["selected_top_is_best"] is False
    assert records[0]["review_required"] is True


def test_live_canary_writes_complete_sidecar_without_modifying_submission(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    response_payload = {
        "queries": [
            {
                "query_id": 1,
                "match_quality_code": 2,
                "best_option_ids": [2],
                "issue_codes": [],
                "requires_video_review": False,
            },
            {
                "query_id": 2,
                "match_quality_code": 4,
                "best_option_ids": [1],
                "issue_codes": [1],
                "requires_video_review": True,
            },
        ]
    }

    class Client:
        def complete_many(self, requests: list[dict[str, Any]], *, on_created: Any) -> Any:
            on_created({"id": "audit-1", "status": "validating", "error": None})
            return {
                "id": "audit-1",
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
                                "model": baseline.MODEL,
                                "choices": [{"message": {"content": json.dumps(response_payload)}}],
                            },
                        },
                        "error": None,
                    }
                ],
            }

    original = inputs["submission"].read_bytes()
    root = tmp_path / "audit"
    result = audit.run_full(
        task2=inputs["task2"],
        fusion=inputs["fusion"],
        submission=inputs["submission"],
        output_root=root,
        expected_query_count=2,
        max_cost_usd=1.0,
        run=True,
        client=Client(),
    )

    assert result["status"] == "complete"
    assert result["review_required_count"] == 1
    assert inputs["submission"].read_bytes() == original
    assert len((root / "submission-audit.jsonl").read_text().splitlines()) == 2
    assert len((root / "video-review-queue.jsonl").read_text().splitlines()) == 1


def test_resume_reuses_frozen_audit_batch_without_second_dispatch(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    payload = {
        "queries": [
            {
                "query_id": 1,
                "match_quality_code": 2,
                "best_option_ids": [2],
                "issue_codes": [],
                "requires_video_review": False,
            },
            {
                "query_id": 2,
                "match_quality_code": 4,
                "best_option_ids": [1],
                "issue_codes": [1],
                "requires_video_review": True,
            },
        ]
    }
    terminal = {
        "id": "audit-resume",
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
                        "model": baseline.MODEL,
                        "choices": [{"message": {"content": json.dumps(payload)}}],
                    },
                },
                "error": None,
            }
        ],
    }

    class InterruptedClient:
        def complete_many(self, requests: list[dict[str, Any]], *, on_created: Any) -> Any:
            on_created({"id": "audit-resume", "status": "validating", "error": None})
            raise RuntimeError("polling interrupted")

    root = tmp_path / "audit-resume"
    with pytest.raises(RuntimeError, match="polling"):
        audit.run_full(
            task2=inputs["task2"], fusion=inputs["fusion"], submission=inputs["submission"],
            output_root=root, expected_query_count=2, max_cost_usd=1.0,
            run=True, client=InterruptedClient(),
        )

    class ResumeClient:
        creates = 0
        waits: list[str] = []

        def complete_many(self, *_args: Any, **_kwargs: Any) -> Any:
            self.creates += 1
            raise AssertionError("resume must not create another audit batch")

        def wait_batch(self, batch_id: str) -> dict[str, Any]:
            self.waits.append(batch_id)
            return terminal

    client = ResumeClient()
    result = audit.run_full(
        task2=inputs["task2"], fusion=inputs["fusion"], submission=inputs["submission"],
        output_root=root, expected_query_count=2, max_cost_usd=1.0,
        run=True, resume=True, client=client,
    )

    assert result["status"] == "complete"
    assert client.creates == 0
    assert client.waits == ["audit-resume"]


def test_accepted_audit_resume_normalizes_json_object_query_ids(tmp_path: Path) -> None:
    assessments = {
        query_id: {
            "match_quality": "exact",
            "best_option_ids": [1],
            "issue_types": [],
            "requires_video_review": False,
            "reason": "Exact.",
        }
        for query_id in (1, 2, 10)
    }
    path = tmp_path / "accepted.json"

    first_hash = audit._write_or_verify_accepted(path, assessments, duration=1.0)
    resumed_hash = audit._write_or_verify_accepted(path, assessments, duration=2.0)

    assert resumed_hash == first_hash
