from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_vqa_grouped_rewrite_fallback_repair as prepare  # noqa: E402
import finalize_vqa_grouped_rewrite_fallback_repair as finalize  # noqa: E402
import run_vqa_grouped_rewrite_full as runner  # noqa: E402


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


def test_merge_precedence_preserves_existing_and_uses_fallback_last() -> None:
    rows = []
    existing = {}
    existing_status = {}
    deterministic = {}
    fresh = {}
    for query_id in range(1, 5):
        row = _row()
        row["query_id"] = query_id
        row["video_id"] = f"v{query_id}"
        row["fallback"]["Q_ID"] = query_id
        row["fallback"]["Video_ID"] = f"v{query_id}"
        rows.append(row)
        existing[query_id] = row["fallback"]
        existing_status[query_id] = "fallback_fusion"
    grouped = {**existing[3], "Answers": [dict(answer) for answer in existing[3]["Answers"]]}
    grouped["Answers"][1]["Answer"] = "existing grouped answer"
    existing[3] = grouped
    existing_status[3] = "raw_salvaged"
    deterministic[1] = prepare.repair_current_query(_duplicate_query(), _row())
    deterministic[2] = prepare.repair_current_query(_duplicate_query(), _row())
    assert deterministic[1] is not None and deterministic[2] is not None
    fresh[1] = {**existing[1], "Answers": [dict(answer) for answer in existing[1]["Answers"]]}
    fresh[1]["Answers"][1]["Answer"] = "fresh grouped answer"

    merged, provenance = finalize.merge_rows(
        rows,
        existing_by_qid=existing,
        existing_status_by_qid=existing_status,
        deterministic_by_qid=deterministic,
        fresh_by_qid=fresh,
    )

    assert merged[0]["Answers"][1]["Answer"] == "fresh grouped answer"
    assert merged[1]["Answers"][1]["Answer"] == "Someone waves."
    assert merged[2]["Answers"][1]["Answer"] == "existing grouped answer"
    assert merged[3] == rows[3]["fallback"]
    assert [item["status"] for item in provenance] == [
        "fresh_repaired",
        "deterministic_repaired",
        "existing_grouped",
        "fallback_fusion",
    ]


def test_finalize_real_panel_merges_fresh_deterministic_and_existing(
    tmp_path: Path,
) -> None:
    preparation_root = tmp_path / "preparation"
    prepare.prepare_repair(output_root=preparation_root, run=True)
    input_paths = {
        name: preparation_root / "inputs" / f"{name}.json"
        for name in ("questions", "gemini", "qwen", "control", "fusion", "provenance")
    }
    fresh_root = tmp_path / "fresh"
    fresh_rows = json.loads(input_paths["fusion"].read_text(encoding="utf-8"))
    first_qid = fresh_rows[0]["Q_ID"]
    fresh_rows[0]["Answers"][1]["Answer"] = "Fresh repaired wording."
    fresh_provenance = [
        {
            "Q_ID": row["Q_ID"],
            "Video_ID": row["Video_ID"],
            "status": "generated" if row["Q_ID"] == first_qid else "fallback_fusion",
            "answer_sha256": [runner._sha_bytes(answer["Answer"].encode()) for answer in row["Answers"]],
        }
        for row in fresh_rows
    ]
    fresh_root.mkdir()
    final_path = prepare._write(fresh_root / "final-organizer.json", fresh_rows)
    provenance_path = prepare._write(fresh_root / "final-provenance.json", fresh_provenance)
    artifacts = runner.source_artifacts(**input_paths)
    prepare._write(
        fresh_root / "manifest.json",
        {
            "query_count": 290,
            "global_max_cost_usd": 5.0,
            "source_artifacts": artifacts,
            "source_artifacts_sha256": runner._sha_bytes(runner._canonical(artifacts)),
        },
    )
    prepare._write(
        fresh_root / "summary.json",
        {
            "status": "complete",
            "query_count": 290,
            "accounted_cost_usd": 0.5,
            "final_sha256": final_path,
            "provenance_sha256": provenance_path,
            "source_artifacts_sha256": runner._sha_bytes(runner._canonical(artifacts)),
        },
    )

    output_root = tmp_path / "final"
    result = finalize.finalize_repair(
        preparation_root=preparation_root,
        fresh_root=fresh_root,
        output_root=output_root,
    )

    assert result["status"] == "complete"
    assert result["status_counts"] == {
        "deterministic_repaired": 319,
        "existing_grouped": 1_833,
        "fallback_fusion": 290,
        "fresh_repaired": 1,
    }
    final_rows = json.loads((output_root / "final-organizer.json").read_text(encoding="utf-8"))
    assert len(final_rows) == 2_443
    assert final_rows[0]["Answers"][1]["Answer"] == "Fresh repaired wording."


def test_finalize_completed_repair_recovers_every_fresh_raw_query(tmp_path: Path) -> None:
    output_root = tmp_path / "completed-final"

    result = finalize.finalize_repair(output_root=output_root)

    assert result["fallback_query_count"] == 1
    assert result["strict_redundant_answer_slots"] == 585
    assert result["status_counts"] == {
        "deterministic_repaired": 319,
        "existing_grouped": 1_833,
        "fallback_fusion": 1,
        "fresh_raw_repaired": 30,
        "fresh_raw_salvaged": 75,
        "fresh_repaired": 185,
    }


def test_strict_postprocess_removes_punctuation_duplicates_with_fusion_fill() -> None:
    canonical = _row()
    repaired = {**canonical["fallback"], "Answers": [dict(answer) for answer in canonical["fallback"]["Answers"]]}
    repaired["Answers"][2]["Answer"] = "Someone waves"
    provenance = [
        {
            "Q_ID": 307,
            "Video_ID": "video.webm",
            "status": "deterministic_repaired",
            "answer_sha256": [],
        }
    ]

    rows, updated = finalize.strict_postprocess_repaired_rows(
        [repaired],
        provenance,
        existing_by_qid={307: canonical["fallback"]},
    )

    assert updated[0]["status"] == "deterministic_repaired"
    assert updated[0]["strict_deduplicated"] is True
    assert updated[0]["strict_fusion_fill_count"] == 1
    assert len({finalize.strict_answer_key(answer["Answer"]) for answer in rows[0]["Answers"]}) == 10
