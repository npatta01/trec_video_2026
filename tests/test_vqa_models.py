from pathlib import Path

import pytest

from trec_video_2026.vqa_models import (
    build_controlled_messages,
    normalize_generated_answer,
)


def test_controlled_messages_keep_frames_in_timestamp_order(tmp_path: Path) -> None:
    frames = tuple(tmp_path / name for name in ("late.jpg", "early.jpg"))
    messages = build_controlled_messages(
        question="What happens?",
        frame_paths=frames,
        frame_timestamps=(8.5, 1.5),
        transcript=None,
    )

    content = messages[0]["content"]
    assert [item["type"] for item in content] == [
        "text",
        "image",
        "text",
        "image",
        "text",
        "text",
    ]
    assert content[0]["text"] == "Frame at 1.500 seconds:"
    assert content[1]["url"].endswith("early.jpg")
    assert content[2]["text"] == "Frame at 8.500 seconds:"
    assert content[3]["url"].endswith("late.jpg")
    assert "Transcript" not in str(messages)


def test_controlled_messages_label_detail_crop(tmp_path: Path) -> None:
    frame = tmp_path / "crop.png"
    frame.write_bytes(b"png")
    messages = build_controlled_messages(
        question="What color is the image?",
        frame_paths=(frame,),
        frame_timestamps=(0.61,),
        frame_descriptions=("Detail crop from frame at 0.610 seconds:",),
        transcript=None,
    )
    assert messages[0]["content"][0]["text"] == (
        "Detail crop from frame at 0.610 seconds:"
    )


def test_controlled_messages_keep_custom_descriptions_with_sorted_frames(
    tmp_path: Path,
) -> None:
    frames = tuple(tmp_path / name for name in ("late.jpg", "early.jpg"))

    messages = build_controlled_messages(
        question="What happens?",
        frame_paths=frames,
        frame_timestamps=(8.5, 1.5),
        frame_descriptions=("Late detail:", "Early detail:"),
        transcript=None,
    )

    content = messages[0]["content"]
    assert (content[0]["text"], content[1]["url"].endswith("early.jpg")) == (
        "Early detail:",
        True,
    )
    assert (content[2]["text"], content[3]["url"].endswith("late.jpg")) == (
        "Late detail:",
        True,
    )


def test_controlled_messages_fall_back_for_empty_description(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"

    messages = build_controlled_messages(
        question="What happens?",
        frame_paths=(frame,),
        frame_timestamps=(2.5,),
        frame_descriptions=("",),
        transcript=None,
    )

    assert messages[0]["content"][0]["text"] == "Frame at 2.500 seconds:"


def test_controlled_messages_reject_description_count_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="descriptions and paths"):
        build_controlled_messages(
            question="What happens?",
            frame_paths=(tmp_path / "frame.jpg",),
            frame_timestamps=(2.5,),
            frame_descriptions=(),
            transcript=None,
        )


def test_controlled_messages_include_frozen_transcript_once(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    messages = build_controlled_messages(
        question="What do they discuss?",
        frame_paths=(frame,),
        frame_timestamps=(0.5,),
        transcript="[00:01.20] The missing bag.",
    )

    serialized = str(messages)
    assert serialized.count("[00:01.20] The missing bag.") == 1
    assert "Answer with one short phrase or sentence" in serialized


def test_normalize_generated_answer_removes_thinking_and_answer_prefix() -> None:
    assert (
        normalize_generated_answer("<think>reasoning</think> Answer: The red bag.\n")
        == "The red bag."
    )
