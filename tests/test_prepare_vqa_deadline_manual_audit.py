from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import scripts.prepare_vqa_deadline_manual_audit as cli  # noqa: E402
from scripts.prepare_vqa_deadline_manual_audit import (  # noqa: E402
    FreezeValidationError,
    assemble_private_packet,
    validate_official_submission,
    validate_prediction_freeze,
)


def _submission_row(q_id: int, video_id: str) -> dict[str, object]:
    return {
        "Q_ID": q_id,
        "Video_ID": video_id,
        "Answers": [
            {"Rank": rank, "Answer": f"answer {rank}", "Time": 0.1}
            for rank in range(1, 11)
        ],
    }


def test_validate_prediction_freeze_checks_single_answer_raw_freeze(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions-freeze.json"
    path.write_text(
        json.dumps(
            {
                "status": "predictions_frozen",
                "decision": "ready_for_scoring",
                "rows": [{"video_id": "A" * 11, "answer": "one answer"}],
            }
        ),
        encoding="utf-8",
    )

    result = validate_prediction_freeze(
        path,
        expected_video_ids={"A" * 11},
        model="gemini",
    )

    assert result.answers["A" * 11]["gemini"] == "one answer"
    assert result.sha256
    assert result.decision == "ready_for_scoring"


def test_validate_prediction_freeze_rejects_incomplete_freeze_before_references(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions-freeze.json"
    path.write_text(
        json.dumps({"rows": []}),
        encoding="utf-8",
    )

    with pytest.raises(FreezeValidationError, match="exactly 1 rows"):
        validate_prediction_freeze(
            path,
            expected_video_ids={"A" * 11},
            model="gemini",
        )


def test_validate_official_submission_checks_ten_ranked_answers_separately(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canary.json"
    path.write_text(json.dumps([_submission_row(1, "A" * 11)]), encoding="utf-8")

    result = validate_official_submission(
        path,
        expected_query_ids={1},
        expected_video_ids={"A" * 11},
    )

    assert result.rows[0]["Answers"][0]["Rank"] == 1


def test_validate_prediction_freeze_checks_both_qwen_condition_answers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions-freeze.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "video_id": "A" * 11,
                        "condition_id": "video-plus-transcript",
                        "status": "passed",
                        "answer": "with transcript",
                    },
                    {
                        "video_id": "A" * 11,
                        "condition_id": "native-video",
                        "status": "passed",
                        "answer": "without transcript",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = validate_prediction_freeze(
        path,
        expected_video_ids={"A" * 11},
        model="qwen",
    )

    assert result.answers["A" * 11] == {
        "qwen_transcript": "with transcript",
        "qwen_no_transcript": "without transcript",
    }


def test_validate_prediction_freeze_rejects_failed_qwen_condition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions-freeze.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "video_id": "A" * 11,
                        "condition_id": "video-plus-transcript",
                        "status": "passed",
                        "answer": "with transcript",
                    },
                    {
                        "video_id": "A" * 11,
                        "condition_id": "native-video",
                        "status": "failed",
                        "answer": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FreezeValidationError, match="failed|non-empty|condition"):
        validate_prediction_freeze(
            path,
            expected_video_ids={"A" * 11},
            model="qwen",
        )


def test_validate_prediction_freeze_rejects_duplicate_qwen_condition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions-freeze.json"
    row = {
        "video_id": "A" * 11,
        "condition_id": "video-plus-transcript",
        "status": "passed",
        "answer": "answer",
    }
    path.write_text(json.dumps({"rows": [row, row]}), encoding="utf-8")

    with pytest.raises(FreezeValidationError, match="duplicate Qwen condition"):
        validate_prediction_freeze(
            path,
            expected_video_ids={"A" * 11},
            model="qwen",
        )


def test_assemble_private_packet_records_media_before_references(tmp_path: Path) -> None:
    source = tmp_path / "video.mkv"
    source.write_bytes(b"media-placeholder")
    destination = tmp_path / "audit"

    def fake_sheet(source_path: Path, destination_path: Path, **_: object) -> None:
        assert source_path == source
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"jpeg-placeholder")

    manifest = assemble_private_packet(
        [
            {
                "video_id": "A" * 11,
                "stratum": "speech",
                "question": "What is said?",
                "duration_seconds": 1.0,
                "source_path": str(source),
                "source_sha256": "source-hash",
            }
        ],
        destination,
        contact_sheet_builder=fake_sheet,
    )

    assert manifest["references_loaded"] is False
    assert (destination / "contact-sheets" / ("A" * 11 + ".jpg")).is_file()


def test_prepare_refuses_incomplete_pair_before_reference_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel_rows = [
        {
            "video_id": "A" * 11,
            "stratum": "speech",
            "question": "What is said?",
            "duration_seconds": 1.0,
            "source_path": str(tmp_path / "video.mkv"),
            "source_sha256": "source-hash",
        }
    ]
    monkeypatch.setattr(cli, "load_panel", lambda *_args, **_kwargs: panel_rows)
    monkeypatch.setattr(
        cli,
        "_load_references",
        lambda *_args, **_kwargs: pytest.fail("references opened before both raw freezes passed"),
    )
    gemini = tmp_path / "gemini.json"
    gemini.write_text(json.dumps({"rows": [{"video_id": "A" * 11, "answer": "ok"}]}))
    qwen = tmp_path / "qwen.json"
    qwen.write_text(json.dumps({"rows": []}))

    with pytest.raises(FreezeValidationError, match="exactly 2 rows"):
        cli.prepare_manual_audit(
            panel_path=tmp_path / "panel.json",
            data_root=tmp_path,
            gemini_freeze=gemini,
            qwen_freeze=qwen,
            reference_root=tmp_path / "references",
            output_root=tmp_path / "audit",
        )
