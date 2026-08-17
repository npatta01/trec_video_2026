from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.assemble_vqa_qwen36_submission import (  # noqa: E402
    AssemblyError,
    assemble_submission,
    main,
)


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def _baseline_row(q_id: int, video_id: str) -> dict[str, object]:
    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "Answers": [
            {"Rank": rank, "Answer": f"baseline {q_id}-{rank}", "Time": 0.1}
            for rank in range(1, 11)
        ],
    }


def _freeze(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "predictions_frozen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "row_count": len(rows),
        "rows": rows,
    }


def _prediction(
    q_id: int,
    video_id: str,
    condition_id: str,
    *,
    answer: str = "",
    status: str = "passed",
    elapsed_seconds: float = 1.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": q_id,
        "video_id": video_id,
        "condition_id": condition_id,
        "status": status,
        "error_type": None if status == "passed" else "provider_error",
        "elapsed_seconds": elapsed_seconds,
    }
    if answer:
        row["answer"] = answer
    return row


def test_assemble_native_then_transcript_and_fill_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps([_baseline_row(1, "video-1"), _baseline_row(2, "video-2")]),
        encoding="utf-8",
    )
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(
        json.dumps(
            _freeze(
                [
                    _prediction(1, "video-1", "video-plus-transcript", answer="Transcript one"),
                    _prediction(1, "video-1", "native-video", answer="Native one", elapsed_seconds=2.5),
                    _prediction(2, "video-2", "video-plus-transcript", answer="Transcript two"),
                    _prediction(2, "video-2", "native-video", status="failed"),
                ]
            )
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        json.dumps(
            [
                {"Q_ID": 1, "Video_ID": "video-1"},
                {"Q_ID": 2, "Video_ID": "video-2"},
            ]
        ),
        encoding="utf-8",
    )

    result = assemble_submission(
        qwen_freeze=freeze_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=2,
    )

    assert [row["Q_ID"] for row in result.rows] == [1, 2]
    assert [answer["Answer"] for answer in result.rows[0]["Answers"]] == [
        "Native one",
        "Transcript one",
        *[f"baseline 1-{rank}" for rank in range(1, 9)],
    ]
    assert result.rows[0]["Answers"][0]["Time"] == 2.5
    assert [answer["Answer"] for answer in result.rows[1]["Answers"]] == [
        "Transcript two",
        *[f"baseline 2-{rank}" for rank in range(1, 10)],
    ]


def test_transcript_first_swaps_the_two_qwen_candidates_globally(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(
        json.dumps(
            _freeze(
                [
                    _prediction(1, "video-1", "video-plus-transcript", answer="Transcript"),
                    _prediction(1, "video-1", "native-video", answer="Native"),
                ]
            )
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    result = assemble_submission(
        qwen_freeze=freeze_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
        transcript_first=True,
    )

    assert [answer["Answer"] for answer in result.rows[0]["Answers"][:2]] == [
        "Transcript",
        "Native",
    ]
    assert result.provenance["transcript_first"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda rows: rows.pop(),
            id="missing-condition-pair",
        ),
        pytest.param(
            lambda rows: rows.__setitem__(1, rows[0].copy()),
            id="duplicate-condition-pair",
        ),
    ],
)
def test_condition_pairs_must_be_exact(tmp_path: Path, mutate) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    rows = [
        _prediction(1, "video-1", "video-plus-transcript", answer="Transcript"),
        _prediction(1, "video-1", "native-video", answer="Native"),
    ]
    mutate(rows)
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(json.dumps(_freeze(rows)), encoding="utf-8")
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    with pytest.raises(AssemblyError, match="condition"):
        assemble_submission(
            qwen_freeze=freeze_path,
            baseline_submission=baseline_path,
            expected_queries=expected_path,
            expected_query_count=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "not_terminal", "status"),
        ("model_id", "Qwen/forged", "model"),
        ("model_revision", "forged-revision", "revision"),
    ],
)
def test_freeze_requires_terminal_status_and_pinned_model(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    freeze = _freeze(
        [
            _prediction(1, "video-1", "video-plus-transcript", answer="Transcript"),
            _prediction(1, "video-1", "native-video", answer="Native"),
        ]
    )
    freeze[field] = value
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    with pytest.raises(AssemblyError, match=message):
        assemble_submission(
            qwen_freeze=freeze_path,
            baseline_submission=baseline_path,
            expected_queries=expected_path,
            expected_query_count=1,
        )


def test_run_manifest_can_bind_current_qwen_freeze_schema(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    freeze = _freeze(
        [
            _prediction(1, "video-1", "video-plus-transcript", answer="Transcript"),
            _prediction(1, "video-1", "native-video", answer="Native"),
        ]
    )
    freeze.pop("model_id")
    freeze.pop("model_revision")
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    result = assemble_submission(
        qwen_freeze=freeze_path,
        run_manifest=manifest_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert result.provenance["model"]["model_revision"] == MODEL_REVISION
    assert result.provenance["run_manifest"]["path"] == str(manifest_path.resolve())


def test_failed_conditions_preserve_all_baseline_answers(tmp_path: Path) -> None:
    baseline = _baseline_row(1, "video-1")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([baseline]), encoding="utf-8")
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(
        json.dumps(
            _freeze(
                [
                    _prediction(1, "video-1", "video-plus-transcript", status="unavailable"),
                    _prediction(1, "video-1", "native-video", status="failed"),
                ]
            )
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    result = assemble_submission(
        qwen_freeze=freeze_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert result.rows[0] == baseline
    assert result.provenance["baseline_fallback_count"] == 1


def test_normalized_duplicate_qwen_answers_are_not_repeated(tmp_path: Path) -> None:
    baseline = _baseline_row(1, "video-1")
    baseline["Answers"][0]["Answer"] = "Native answer"
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([baseline]), encoding="utf-8")
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(
        json.dumps(
            _freeze(
                [
                    _prediction(1, "video-1", "video-plus-transcript", answer=" native   answer "),
                    _prediction(1, "video-1", "native-video", answer="Native answer"),
                ]
            )
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")

    result = assemble_submission(
        qwen_freeze=freeze_path,
        baseline_submission=baseline_path,
        expected_queries=expected_path,
        expected_query_count=1,
    )

    assert [answer["Answer"] for answer in result.rows[0]["Answers"]] == [
        "Native answer",
        "baseline 1-2",
        *[f"baseline 1-{rank}" for rank in range(3, 11)],
    ]


def test_cli_transcript_first_writes_valid_submission(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps([_baseline_row(1, "video-1")]), encoding="utf-8")
    freeze_path = tmp_path / "qwen.json"
    freeze_path.write_text(
        json.dumps(
            _freeze(
                [
                    _prediction(1, "video-1", "video-plus-transcript", answer="Transcript"),
                    _prediction(1, "video-1", "native-video", answer="Native"),
                ]
            )
        ),
        encoding="utf-8",
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(json.dumps([{"Q_ID": 1, "Video_ID": "video-1"}]), encoding="utf-8")
    output_path = tmp_path / "submission.json"

    main(
        [
            "--qwen-freeze",
            str(freeze_path),
            "--baseline",
            str(baseline_path),
            "--expected-queries",
            str(expected_path),
            "--expected-query-count",
            "1",
            "--output",
            str(output_path),
            "--transcript-first",
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert [answer["Answer"] for answer in output[0]["Answers"][:2]] == [
        "Transcript",
        "Native",
    ]
