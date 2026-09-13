from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_vqa_mc_gemini38_retry as retry  # noqa: E402


def _row(query_id: int, status: str, ranking: list[int] | None = None) -> dict:
    return {
        "query_id": query_id,
        "video_id": f"v{query_id}.webm",
        "status": status,
        "ranking": ranking,
    }


def test_select_retry_inputs_uses_only_frozen_invalid_rows() -> None:
    protocol = {
        "model": "gemini-3.8-flash",
        "rows": [
            {"query_id": 1, "video_id": "v1.webm"},
            {"query_id": 2, "video_id": "v2.webm"},
            {"query_id": 3, "video_id": "v3.webm"},
        ],
        "weak_query_ids": [2, 3],
    }
    freeze = {
        "status": "predictions_frozen",
        "model": "gemini-3.8-flash",
        "rows": [_row(1, "passed", [1, 2]), _row(2, "invalid"), _row(3, "unavailable")],
    }
    v1 = [
        {"Q_ID": 1, "Video_ID": "v1.webm"},
        {"Q_ID": 2, "Video_ID": "v2.webm"},
        {"Q_ID": 3, "Video_ID": "v3.webm"},
    ]

    selected = retry.select_retry_inputs(protocol=protocol, freeze=freeze, v1=v1)

    assert [row["query_id"] for row in selected["rows"]] == [2]
    assert [row["Q_ID"] for row in selected["v1"]] == [2]
    assert selected["weak_query_ids"] == [2]


def test_select_retry_inputs_rejects_wrong_model() -> None:
    with pytest.raises(ValueError, match="source model"):
        retry.select_retry_inputs(
            protocol={"model": "gemini-3.7-flash", "rows": [], "weak_query_ids": []},
            freeze={"status": "predictions_frozen", "model": "gemini-3.7-flash", "rows": []},
            v1=[],
        )


def test_overlay_replaces_only_successfully_retried_rows() -> None:
    source = [
        {"Q_ID": 1, "Video_ID": "v1.webm", "Answers": ["source-1"]},
        {"Q_ID": 2, "Video_ID": "v2.webm", "Answers": ["source-2"]},
        {"Q_ID": 3, "Video_ID": "v3.webm", "Answers": ["source-3"]},
    ]
    retry_submission = [
        {"Q_ID": 2, "Video_ID": "v2.webm", "Answers": ["retry-2"]},
        {"Q_ID": 3, "Video_ID": "v3.webm", "Answers": ["retry-3"]},
    ]
    retry_rows = [_row(2, "passed", [1, 2]), _row(3, "invalid")]

    merged = retry.overlay_successes(
        source_submission=source,
        retry_submission=retry_submission,
        retry_rows=retry_rows,
    )

    assert merged == [source[0], retry_submission[0], source[2]]


def test_build_retry_protocol_preserves_source_row_hash_bindings() -> None:
    source_row = {
        "query_id": 2,
        "video_id": "v2.webm",
        "status": "ready",
        "duration_seconds": 1.0,
        "input_sha256": "1" * 64,
        "media_sha256": "2" * 64,
        "delivery_sha256": "3" * 64,
    }
    source_protocol = {
        "model": "gemini-3.8-flash",
        "model_arm": "native-av-lowres-4fps-high-thinking-direct-options",
        "settings": {"video_fps": 4.0},
        "source": {
            "task2_sha256": "4" * 64,
            "v1_sha256": "5" * 64,
            "weak_queue_sha256": "6" * 64,
        },
    }

    protocol = retry.build_retry_protocol(
        rows=[source_row],
        weak_query_ids=[],
        source_protocol=source_protocol,
        max_cost_usd=0.10,
        created_at="2026-09-13T00:00:00Z",
    )

    assert protocol["rows"] == [source_row]
    assert protocol["rows"][0]["input_sha256"] == "1" * 64
    assert protocol["rows"][0]["media_sha256"] == "2" * 64
    assert protocol["rows"][0]["delivery_sha256"] == "3" * 64


def test_package_submission_writes_one_normal_upload_file(tmp_path: Path) -> None:
    task = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Options": [{"option": "A"}, {"option": "B"}],
        }
    ]
    submission = [
        {
            "Q_ID": 1,
            "Video_ID": "v1.webm",
            "Answers": [{"Rank": 1, "Answer": "B"}, {"Rank": 2, "Answer": "A"}],
        }
    ]
    task_path = tmp_path / "task.json"
    candidate_path = tmp_path / "private-rescue.json"
    target = tmp_path / "g38_mc_direct" / "submission.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    candidate_path.write_text(json.dumps(submission), encoding="utf-8")

    result = retry.package_submission(
        candidate=candidate_path,
        task2=task_path,
        target=target,
        expected_count=1,
    )

    assert json.loads(target.read_text(encoding="utf-8")) == submission
    assert [path.name for path in target.parent.iterdir()] == ["submission.json"]
    assert result["run_tag"] == "g38_mc_direct"
    assert result["row_count"] == 1
