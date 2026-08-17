from __future__ import annotations

import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_deadline_canary import (
    load_panel,
    validate_ranked_submission,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "configs/vqa/deadline_canary_5.json"
DATA_ROOT = REPO_ROOT / "data"


def test_frozen_panel_has_five_unique_labeled_rows_across_required_strata() -> None:
    rows = load_panel(PANEL_PATH, data_root=DATA_ROOT)

    assert len(rows) == 5
    assert len({row["video_id"] for row in rows}) == 5
    assert {row["stratum"] for row in rows} == {
        "temporal",
        "causal",
        "ocr_detail",
        "counting",
        "speech",
    }
    assert all("/2025/training/videos_original/" in row["source_path"] for row in rows)


def test_ranked_submission_requires_ten_distinct_ordered_answers() -> None:
    row = {
        "Q_ID": 7,
        "Video_ID": "example.webm",
        "Answers": [
            {"Rank": rank, "Answer": f"candidate {rank}", "Time": 1.5}
            for rank in range(1, 11)
        ],
    }

    validate_ranked_submission([row], expected_query_ids={7})

    duplicate = json.loads(json.dumps(row))
    duplicate["Answers"][9]["Answer"] = duplicate["Answers"][0]["Answer"]
    with pytest.raises(ValueError, match="distinct"):
        validate_ranked_submission([duplicate], expected_query_ids={7})


def test_ranked_submission_rejects_missing_query_or_nonconsecutive_ranks() -> None:
    row = {
        "Q_ID": 7,
        "Video_ID": "example.webm",
        "Answers": [
            {"Rank": rank, "Answer": f"candidate {rank}", "Time": 1.5}
            for rank in range(1, 11)
        ],
    }
    row["Answers"][-1]["Rank"] = 11

    with pytest.raises(ValueError, match="ranks"):
        validate_ranked_submission([row], expected_query_ids={7})

    with pytest.raises(ValueError, match="query IDs"):
        validate_ranked_submission([], expected_query_ids={7})
