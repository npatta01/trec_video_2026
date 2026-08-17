from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_vqa_grouped_rewrite_fallback_repair as prepare  # noqa: E402


def _row() -> dict[str, Any]:
    return {
        "query_id": 307,
        "video_id": "video.webm",
        "question": "What happens?",
        "rank1_source": "gemini_overlay",
        "rank1_answer": "The person waves.",
        "source_candidates": [
            {"source_id": "gemini_overlay", "answer": "The person waves.", "time": 1.0},
            {"source_id": "qwen_native_video", "answer": "Someone waves.", "time": 2.0},
        ],
        "fallback": {
            "Q_ID": 307,
            "Video_ID": "video.webm",
            "Answers": [
                {"Rank": rank, "Answer": answer, "Time": float(rank)}
                for rank, answer in enumerate(
                    [
                        "The person waves.",
                        "Someone waves.",
                        "A person is waving.",
                        "They wave.",
                        "The person gestures.",
                        "A wave is shown.",
                        "The individual waves.",
                        "Someone makes a waving gesture.",
                        "A person makes a hand gesture.",
                        "The person raises a hand.",
                    ],
                    1,
                )
            ],
        },
    }


def _duplicate_query() -> dict[str, Any]:
    return {
        "query_id": 307,
        "groups": [
            {"group_id": "g1", "source_ids": ["gemini_overlay"]},
            {"group_id": "g2", "source_ids": ["qwen_native_video"]},
        ],
        "answers": [
            {"rank": 1, "answer": "The person waves.", "source_group": "g1"},
            {"rank": 2, "answer": "Someone waves.", "source_group": "g2"},
            {"rank": 3, "answer": "They wave.", "source_group": "g1"},
            {"rank": 4, "answer": "A person is waving.", "source_group": "g1"},
            {"rank": 5, "answer": "A wave is shown.", "source_group": "g1"},
            {"rank": 6, "answer": "The individual waves.", "source_group": "g1"},
            {"rank": 7, "answer": "Someone makes a waving gesture.", "source_group": "g2"},
            {"rank": 8, "answer": "The person gestures.", "source_group": "g1"},
            {"rank": 9, "answer": "They wave.", "source_group": "g1"},
            {"rank": 10, "answer": "The person raises a hand.", "source_group": "g1"},
        ],
    }


def test_duplicate_repair_preserves_rank_one_and_fills_from_fusion() -> None:
    repaired = prepare.repair_current_query(_duplicate_query(), _row())

    assert repaired is not None
    assert repaired["repair_reason"] == "deduplicated_with_fusion_fill"
    assert repaired["answers"][0] == {
        "rank": 1,
        "answer": "The person waves.",
        "source_group": "g1",
        "origin": "generated",
    }
    assert len(repaired["answers"]) == 10
    assert len({" ".join(answer["answer"].split()).casefold() for answer in repaired["answers"]}) == 10
    assert repaired["answers"][-1] == {
        "rank": 10,
        "answer": "A person makes a hand gesture.",
        "source_group": "fusion_fallback",
        "origin": "fusion_fill",
    }


def test_missing_exact_group_answer_is_inserted_without_new_text() -> None:
    query = _duplicate_query()
    query["answers"][1] = {"rank": 2, "answer": "A waving individual.", "source_group": "g2"}
    query["answers"][8] = {"rank": 9, "answer": "A raised hand is visible.", "source_group": "g1"}

    repaired = prepare.repair_current_query(query, _row())

    assert repaired is not None
    assert repaired["repair_reason"] == "exact_source_hedge_inserted"
    assert repaired["answers"][-1] == {
        "rank": 10,
        "answer": "Someone waves.",
        "source_group": "g2",
        "origin": "source_hedge",
    }
    assert all(answer["answer"] != "The person raises a hand." for answer in repaired["answers"])


def test_invalid_provider_rank_sequence_is_reassigned() -> None:
    query = _duplicate_query()
    query["answers"][8] = {"rank": 6, "answer": "A raised hand is visible.", "source_group": "g1"}

    repaired = prepare.repair_current_query(query, _row())

    assert repaired is not None
    assert repaired["repair_reason"] == "rank_sequence_repaired"
    assert [answer["rank"] for answer in repaired["answers"]] == list(range(1, 11))


def test_real_panel_preparation_reconciles_all_fallbacks_without_writing(tmp_path: Path) -> None:
    output_root = tmp_path / "preparation"

    result = prepare.prepare_repair(output_root=output_root, run=False)

    assert result["status"] == "dry_run"
    assert result["existing_grouped_query_count"] == 1_833
    assert result["deterministic_repair_query_count"] == 320
    assert result["legacy_fresh_query_count"] == 290
    assert result["repair_reason_counts"] == {
        "deduplicated_with_fusion_fill": 314,
        "exact_source_hedge_inserted": 5,
        "rank_sequence_repaired": 1,
    }
    assert result["filtered_source_query_counts"] == {
        "control": 290,
        "fusion": 290,
        "gemini": 290,
        "provenance": 290,
        "questions": 290,
        "qwen": 290,
    }
    assert result["output_root_created"] is False
    assert not output_root.exists()


def test_real_panel_preparation_freezes_once_into_complete_root(tmp_path: Path) -> None:
    output_root = tmp_path / "preparation"

    result = prepare.prepare_repair(output_root=output_root, run=True)

    assert result["status"] == "inputs_frozen"
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "summary.json").is_file()
    assert (output_root / "deterministic-repairs.json").is_file()
    assert (output_root / "legacy-query-ids.json").is_file()
    assert sorted(path.name for path in (output_root / "inputs").glob("*.json")) == [
        "control.json",
        "fusion.json",
        "gemini.json",
        "provenance.json",
        "questions.json",
        "qwen.json",
    ]
    with pytest.raises(FileExistsError, match="preparation root must be absent"):
        prepare.prepare_repair(output_root=output_root, run=True)
