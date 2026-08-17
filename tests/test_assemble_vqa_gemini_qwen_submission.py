from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.assemble_vqa_gemini_qwen_submission import (  # noqa: E402
    AssemblyError,
    assemble_submission,
)


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def test_fusion_cli_help_runs_without_installed_project(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(repository / "scripts/assemble_vqa_gemini_qwen_submission.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--gemini-freeze" in completed.stdout


def _baseline_row(q_id: int, video_id: str) -> dict[str, object]:
    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "Answers": [
            {"Rank": rank, "Answer": f"baseline {q_id}-{rank}", "Time": 0.1}
            for rank in range(1, 11)
        ],
    }


def _gemini_row(
    q_id: int,
    video_id: str,
    *,
    status: str = "succeeded",
    answer: str = "Gemini answer",
    elapsed_seconds: float = 1.25,
) -> dict[str, object]:
    return {
        "query_id": q_id,
        "video_id": video_id,
        "status": status,
        "answer": answer if status == "succeeded" else "",
        "elapsed_seconds": elapsed_seconds,
    }


def _qwen_row(
    q_id: int,
    video_id: str,
    condition_id: str,
    *,
    status: str = "succeeded",
    answer: str = "Qwen answer",
    elapsed_seconds: float = 2.5,
) -> dict[str, object]:
    return {
        "query_id": q_id,
        "video_id": video_id,
        "condition_id": condition_id,
        "status": status,
        "answer": answer if status == "succeeded" else "",
        "elapsed_seconds": elapsed_seconds,
    }


def _write_inputs(
    tmp_path: Path,
    *,
    gemini_rows: list[dict[str, object]],
    qwen_rows: list[dict[str, object]],
    baseline: list[dict[str, object]] | None = None,
    expected: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    baseline_path = tmp_path / "baseline.json"
    expected_path = tmp_path / "expected.json"
    gemini_path = tmp_path / "gemini.json"
    qwen_path = tmp_path / "qwen.json"
    if baseline is None:
        baseline = [_baseline_row(1, "video-1"), _baseline_row(2, "video-2")]
    if expected is None:
        expected = [
            {"Q_ID": row["Q_ID"], "Video_ID": row["Video_ID"]}
            for row in baseline
        ]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    expected_path.write_text(json.dumps(expected), encoding="utf-8")
    gemini_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model": "gemini-3.7-flash",
                "row_count": len(gemini_rows),
                "rows": gemini_rows,
            }
        ),
        encoding="utf-8",
    )
    qwen_path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "row_count": len(qwen_rows),
                "rows": qwen_rows,
            }
        ),
        encoding="utf-8",
    )
    return gemini_path, qwen_path, baseline_path, expected_path


def test_fusion_orders_gemini_then_transcript_then_native_then_baseline(
    tmp_path: Path,
) -> None:
    gemini, qwen, baseline, expected = _write_inputs(
        tmp_path,
        gemini_rows=[
            _gemini_row(1, "video-1", answer="Gemini one"),
            _gemini_row(2, "video-2", answer="Gemini two"),
        ],
        qwen_rows=[
            _qwen_row(1, "video-1", "video-plus-transcript", answer="Transcript one"),
            _qwen_row(1, "video-1", "native-video", answer="Native one", elapsed_seconds=3.5),
            _qwen_row(2, "video-2", "video-plus-transcript", answer="Transcript two"),
            _qwen_row(2, "video-2", "native-video", answer="Native two"),
        ],
    )

    result = assemble_submission(
        gemini_freeze=gemini,
        qwen_freeze=qwen,
        baseline_submission=baseline,
        expected_queries=expected,
        expected_query_count=2,
    )

    assert [row["Q_ID"] for row in result.rows] == [1, 2]
    assert [item["Answer"] for item in result.rows[0]["Answers"][:3]] == [
        "Gemini one",
        "Transcript one",
        "Native one",
    ]
    assert result.rows[0]["Answers"][0]["Time"] == 1.25
    assert result.rows[0]["Answers"][2]["Time"] == 3.5
    assert result.provenance["selected_counts"] == {
        "gemini": 2,
        "qwen_video_plus_transcript": 2,
        "qwen_native_video": 2,
        "baseline": 14,
    }


def test_fusion_skips_failed_rows_and_deduplicates_before_baseline_fill(
    tmp_path: Path,
) -> None:
    baseline = _baseline_row(1, "video-1")
    baseline["Answers"][0]["Answer"] = "same answer"
    gemini, qwen, baseline_path, expected = _write_inputs(
        tmp_path,
        gemini_rows=[_gemini_row(1, "video-1", answer=" Same   answer ")],
        qwen_rows=[
            _qwen_row(1, "video-1", "video-plus-transcript", answer="same answer"),
            _qwen_row(1, "video-1", "native-video", status="unavailable"),
        ],
        baseline=[baseline],
        expected=[{"Q_ID": 1, "Video_ID": "video-1"}],
    )

    result = assemble_submission(
        gemini_freeze=gemini,
        qwen_freeze=qwen,
        baseline_submission=baseline_path,
        expected_queries=expected,
        expected_query_count=1,
    )

    answers = result.rows[0]["Answers"]
    assert [item["Answer"] for item in answers] == [
        "Same   answer",
        *[f"baseline 1-{rank}" for rank in range(2, 11)],
    ]
    assert result.provenance["rows"][0]["selected_sources"] == [
        "gemini",
        *["baseline" for _ in range(9)],
    ]
    assert result.provenance["rows"][0]["qwen_statuses"] == {
        "video-plus-transcript": "succeeded",
        "native-video": "unavailable",
    }


@pytest.mark.parametrize("bad_time", [True, "1.5", float("inf"), -0.1])
def test_fusion_rejects_invalid_gemini_answer_time(
    tmp_path: Path, bad_time: object
) -> None:
    gemini, qwen, baseline, expected = _write_inputs(
        tmp_path,
        gemini_rows=[_gemini_row(1, "video-1", elapsed_seconds=bad_time)],
        qwen_rows=[
            _qwen_row(1, "video-1", "video-plus-transcript"),
            _qwen_row(1, "video-1", "native-video"),
        ],
        baseline=[_baseline_row(1, "video-1")],
        expected=[{"Q_ID": 1, "Video_ID": "video-1"}],
    )

    with pytest.raises(AssemblyError, match="Gemini answer time"):
        assemble_submission(
            gemini_freeze=gemini,
            qwen_freeze=qwen,
            baseline_submission=baseline,
            expected_queries=expected,
            expected_query_count=1,
        )


def test_fusion_writes_provenance_and_refuses_overwrite(tmp_path: Path) -> None:
    gemini, qwen, baseline, expected = _write_inputs(
        tmp_path,
        gemini_rows=[_gemini_row(1, "video-1"), _gemini_row(2, "video-2")],
        qwen_rows=[
            _qwen_row(1, "video-1", "video-plus-transcript"),
            _qwen_row(1, "video-1", "native-video"),
            _qwen_row(2, "video-2", "video-plus-transcript"),
            _qwen_row(2, "video-2", "native-video"),
        ],
    )
    output = tmp_path / "submission.json"
    result = assemble_submission(
        gemini_freeze=gemini,
        qwen_freeze=qwen,
        baseline_submission=baseline,
        expected_queries=expected,
        expected_query_count=2,
        output_path=output,
    )
    assert result.provenance_path == tmp_path / "submission.provenance.json"
    assert json.loads(result.provenance_path.read_text(encoding="utf-8"))["output_submission_sha256"]
    with pytest.raises(AssemblyError, match="overwrite"):
        assemble_submission(
            gemini_freeze=gemini,
            qwen_freeze=qwen,
            baseline_submission=baseline,
            expected_queries=expected,
            expected_query_count=2,
            output_path=output,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.__setitem__("status", "running"), "Gemini freeze status"),
        (lambda payload: payload.__setitem__("model_id", "Qwen/forged"), "model"),
        (lambda payload: payload["rows"].pop(), "row_count"),
    ],
)
def test_fusion_rejects_invalid_pinned_or_incomplete_inputs(
    tmp_path: Path, mutate, message: str
) -> None:
    gemini, qwen, baseline, expected = _write_inputs(
        tmp_path,
        gemini_rows=[_gemini_row(1, "video-1"), _gemini_row(2, "video-2")],
        qwen_rows=[
            _qwen_row(1, "video-1", "video-plus-transcript"),
            _qwen_row(1, "video-1", "native-video"),
            _qwen_row(2, "video-2", "video-plus-transcript"),
            _qwen_row(2, "video-2", "native-video"),
        ],
    )
    target = gemini if "Gemini" in message else qwen
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssemblyError, match=message):
        assemble_submission(
            gemini_freeze=gemini,
            qwen_freeze=qwen,
            baseline_submission=baseline,
            expected_queries=expected,
            expected_query_count=2,
        )
