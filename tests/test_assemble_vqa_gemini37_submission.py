from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.assemble_vqa_gemini37_submission import (  # noqa: E402
    AssemblyError,
    assemble_submission,
)


def _baseline_row(q_id: int, video_id: str) -> dict[str, object]:
    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "Answers": [
            {"Rank": rank, "Answer": f"baseline {q_id}-{rank}", "Time": 0.1}
            for rank in range(1, 11)
        ],
    }


def test_assemble_promotes_passed_gemini_and_falls_back_on_failed(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([_baseline_row(1, "video-1"), _baseline_row(2, "video-2")]),
        encoding="utf-8",
    )
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 2,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": "passed",
                        "answer": "Gemini answer",
                        "time": 1.5,
                    },
                    {
                        "query_id": 2,
                        "video_id": "video-2",
                        "status": "failed",
                        "answer": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1, 2]), encoding="utf-8")

    result = assemble_submission(
        gemini_freeze=gemini_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=2,
    )

    assert [row["Q_ID"] for row in result.rows] == [1, 2]
    assert result.rows[0]["Answers"][0] == {
        "Rank": 1,
        "Answer": "Gemini answer",
        "Time": 1.5,
    }
    assert [answer["Answer"] for answer in result.rows[0]["Answers"][1:]] == [
        f"baseline 1-{rank}" for rank in range(1, 10)
    ]
    assert result.rows[1] == _baseline_row(2, "video-2")


@pytest.mark.parametrize(
    "fallback_status", ["unsupported", "cost_cap_blocked", "upload_failed"]
)
def test_assemble_preserves_baseline_for_preflight_or_cost_fallback(
    tmp_path: Path, fallback_status: str
) -> None:
    baseline = [_baseline_row(1, "video-1")]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 1,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": fallback_status,
                        "answer": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1]), encoding="utf-8")

    result = assemble_submission(
        gemini_freeze=gemini_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert list(result.rows) == baseline


def test_passed_gemini_excludes_normalized_duplicate_baseline_answer(
    tmp_path: Path,
) -> None:
    baseline = _baseline_row(1, "video-1")
    baseline["Answers"][0]["Answer"] = "  Gemini   answer  "
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([baseline]), encoding="utf-8")
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 1,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": "passed",
                        "answer": "Gemini answer",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1]), encoding="utf-8")

    result = assemble_submission(
        gemini_freeze=gemini_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert [answer["Answer"] for answer in result.rows[0]["Answers"]] == [
        "Gemini answer",
        *[f"baseline 1-{rank}" for rank in range(2, 11)],
    ]


def test_assemble_writes_submission_and_hash_bound_provenance(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 1,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": "unavailable",
                        "answer": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1]), encoding="utf-8")
    output = tmp_path / "submission.json"

    result = assemble_submission(
        gemini_freeze=gemini_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
        output_path=output,
    )

    assert result.output_path == output
    assert result.provenance_path == tmp_path / "submission.provenance.json"
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["output_submission_sha256"] == result.provenance["output_submission_sha256"]
    assert provenance["baseline_fallback_count"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("status", "not_terminal", "status"), ("model", "forged-model", "model")),
)
def test_gemini_freeze_requires_terminal_status_and_exact_model(
    field: str,
    value: str,
    message: str,
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 1,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": "passed",
                        "answer": "answer",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1]), encoding="utf-8")

    envelope = json.loads(gemini_path.read_text(encoding="utf-8"))
    envelope[field] = value
    gemini_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(AssemblyError, match=message):
        assemble_submission(
            gemini_freeze=gemini_path,
            baseline_submission=baseline_path,
            expected_queries=expected_path,
            expected_query_count=1,
        )


def test_gemini_elapsed_seconds_is_used_for_rank_one_time(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    gemini_path = tmp_path / "gemini.json"
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": 1,
                "rows": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "status": "passed",
                        "answer": "answer",
                        "elapsed_seconds": 2.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([1]), encoding="utf-8")

    result = assemble_submission(
        gemini_freeze=gemini_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert result.rows[0]["Answers"][0]["Time"] == 2.5
