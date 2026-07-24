import hashlib
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.oracle_panel import (
    DEFAULT_SELECTED_IDS,
    build_oracle_panel,
)

SELECTED_IDS = [f"video-{index:02d}" for index in range(10)]


@pytest.fixture
def panel_inputs(tmp_path: Path) -> tuple[Path, Path]:
    annotation_dir = tmp_path / "annotations"
    video_dir = tmp_path / "videos"
    annotation_dir.mkdir()
    video_dir.mkdir()

    for video_id in SELECTED_IDS:
        (annotation_dir / f"{video_id}.json").write_text(
            json.dumps(
                {
                    "question": f"What did the speaker say in {video_id}?",
                    "correct_answer": "secret reference answer",
                    "incorrect_answers": ["secret distractor"],
                }
            ),
            encoding="utf-8",
        )
        (video_dir / f"{video_id}.mkv").write_bytes(video_id.encode())

    return annotation_dir, video_dir


def test_oracle_panel_contains_no_reference_answers(
    panel_inputs: tuple[Path, Path],
) -> None:
    panel = build_oracle_panel(*panel_inputs, SELECTED_IDS)

    encoded = json.dumps(panel).lower()
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded
    assert "secret reference answer" not in encoded
    assert len(panel["examples"]) == 10


def test_oracle_panel_binds_media_hashes(
    panel_inputs: tuple[Path, Path],
) -> None:
    panel = build_oracle_panel(*panel_inputs, SELECTED_IDS)

    for row in panel["examples"]:
        expected = hashlib.sha256(row["video_id"].encode()).hexdigest()
        assert row["media_sha256"] == expected
        assert len(row["media_sha256"]) == 64
        assert row["operational_tag"] == "speech_content"


@pytest.mark.parametrize(
    ("selected_ids", "message"),
    [
        (SELECTED_IDS[:-1] + [SELECTED_IDS[0]], "duplicate"),
        (SELECTED_IDS[:-1] + ["missing-video"], "missing media"),
    ],
)
def test_oracle_panel_rejects_invalid_selection(
    panel_inputs: tuple[Path, Path],
    selected_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_oracle_panel(*panel_inputs, selected_ids)


def test_oracle_panel_rejects_annotation_without_question(
    panel_inputs: tuple[Path, Path],
) -> None:
    annotation_dir, video_dir = panel_inputs
    (annotation_dir / f"{SELECTED_IDS[0]}.json").write_text(
        json.dumps({"correct_answer": "still private"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="question"):
        build_oracle_panel(annotation_dir, video_dir, SELECTED_IDS)


def test_default_panel_is_lexical_and_includes_horse_age_example() -> None:
    assert "NN7G28sBOlY" in DEFAULT_SELECTED_IDS
    assert "NUmb2P3Bzu8" not in DEFAULT_SELECTED_IDS  # prosody
    assert "Gp4LVTn3YbY" not in DEFAULT_SELECTED_IDS  # temporal action
    assert len(DEFAULT_SELECTED_IDS) == len(set(DEFAULT_SELECTED_IDS)) == 10
