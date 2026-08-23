from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_mc_gemini_direct_repair as repair  # noqa: E402


def _submission(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Q_ID": row["query_id"],
            "Video_ID": row["video_id"],
            "Answers": [
                {"Rank": index, "Answer": option}
                for index, option in enumerate(row["options"], start=1)
            ],
        }
        for row in rows
    ]


@pytest.fixture
def repair_panel() -> dict[str, object]:
    rows = [
        {
            "query_id": 1,
            "video_id": "one.webm",
            "question": "What happens?",
            "options": ["A", "B", "C"],
            "status": "ready",
        },
        {
            "query_id": 2,
            "video_id": "two.webm",
            "question": "Who arrives?",
            "options": ["D", "E", "F"],
            "status": "ready",
        },
        {
            "query_id": 3,
            "video_id": "missing.webm",
            "question": "What is missing?",
            "options": ["G", "H"],
            "status": "unavailable",
        },
    ]
    parent = {
        "rows": [
            {"query_id": 1, "status": "passed", "ranking": [2, 1, 3]},
            {"query_id": 2, "status": "invalid", "ranking": None},
            {"query_id": 3, "status": "unavailable", "ranking": None},
        ]
    }
    repaired = {
        "rows": [
            {"query_id": 2, "status": "passed", "ranking": [3, 1, 2]},
        ]
    }
    return {
        "rows": rows,
        "v1": _submission(rows),
        "parent": parent,
        "repaired": repaired,
    }


def test_select_repair_inputs_retries_only_parent_invalid_rows(
    repair_panel: dict[str, object],
) -> None:
    selected = repair.select_repair_inputs(
        rows=repair_panel["rows"],
        v1_submission=repair_panel["v1"],
        weak_query_ids={2},
        parent_freeze=repair_panel["parent"],
        expected_invalid_count=1,
    )

    assert [row["query_id"] for row in selected["rows"]] == [2]
    assert [row["Q_ID"] for row in selected["v1_submission"]] == [2]
    assert selected["weak_query_ids"] == {2}


def test_assemble_repaired_candidates_preserves_parent_and_repairs_both_candidates(
    repair_panel: dict[str, object],
) -> None:
    original_parent = deepcopy(repair_panel["parent"])

    result = repair.assemble_repaired_candidates(
        rows=repair_panel["rows"],
        v1_submission=repair_panel["v1"],
        weak_query_ids={2},
        parent_freeze=repair_panel["parent"],
        repair_freeze=repair_panel["repaired"],
    )

    assert repair_panel["parent"] == original_parent
    assert result["recovered_query_ids"] == [2]
    assert result["remaining_invalid_query_ids"] == []
    assert result["v2b_direct_count"] == 1
    assert result["v3b_direct_count"] == 2
    assert result["v2b"][1]["Answers"][0]["Answer"] == "F"
    assert result["v3b"][0]["Answers"][0]["Answer"] == "B"
    assert result["v3b"][1]["Answers"][0]["Answer"] == "F"
    assert result["v3b"][2] == repair_panel["v1"][2]


def test_assemble_repaired_candidates_rejects_nonfailed_repair_row(
    repair_panel: dict[str, object],
) -> None:
    repair_panel["repaired"]["rows"] = [
        {"query_id": 1, "status": "passed", "ranking": [1, 2, 3]}
    ]

    with pytest.raises(ValueError, match="coverage"):
        repair.assemble_repaired_candidates(
            rows=repair_panel["rows"],
            v1_submission=repair_panel["v1"],
            weak_query_ids={2},
            parent_freeze=repair_panel["parent"],
            repair_freeze=repair_panel["repaired"],
        )
