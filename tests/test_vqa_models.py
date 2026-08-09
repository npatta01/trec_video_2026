from pathlib import Path

import pytest

from trec_video_2026.vqa_models import (
    build_controlled_messages,
    classify_question_type,
    normalize_generated_answer,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many cups are on the table?", "count"),
        ("What words are written on the sign?", "visible-text"),
        ("What does she say after opening the door?", "speech"),
        ("What happens before the dog runs away?", "temporal"),
        ("Why does the pan catch fire?", "causal-procedural"),
        ("What color is the bicycle?", "general"),
    ],
)
def test_question_type_is_deterministic(question: str, expected: str) -> None:
    assert classify_question_type(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What words does she say are written on the sign?", "visible-text"),
        ("How many cups are there after the table is cleared?", "count"),
        ("What does she say before the dog runs away?", "speech"),
        ("What happens before the pan catches fire and why?", "temporal"),
    ],
)
def test_question_type_obeys_declared_precedence(
    question: str, expected: str
) -> None:
    assert classify_question_type(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How is the woman dressed?", "general"),
        ("What is the first letter on the sign?", "visible-text"),
        ("What is the first word on the poster?", "visible-text"),
        ("What is the man holding?", "general"),
        ("How does the woman open the door?", "causal-procedural"),
        ("How is the door opened?", "causal-procedural"),
        ("What is the first event in the video?", "temporal"),
        ("How does the video start?", "temporal"),
        ("How does the woman appear?", "general"),
        ("How does the open door look?", "general"),
        ("How is the cooked chicken colored?", "general"),
        ("How does the closed box appear?", "general"),
        ("How does the closed window appear?", "general"),
        ("How is the prepared meal presented?", "general"),
        ("How do you open the door?", "causal-procedural"),
        ("How is the chicken cooked?", "causal-procedural"),
        ("How is the chicken cooked in the video?", "causal-procedural"),
        ("How is the meal prepared?", "causal-procedural"),
        ("How does Alice open the door?", "causal-procedural"),
        ("How do workers prepare the meal?", "causal-procedural"),
    ],
)
def test_question_type_avoids_broad_vqa_false_matches(
    question: str, expected: str
) -> None:
    assert classify_question_type(question) == expected


def test_current_answer_policy_preserves_existing_instruction_byte_for_byte(
    tmp_path: Path,
) -> None:
    messages = build_controlled_messages(
        question="What color is the bicycle?",
        frame_paths=(tmp_path / "frame.jpg",),
        frame_timestamps=(0.5,),
        transcript=None,
        answer_policy="current",
    )

    assert messages[0]["content"][-1] == {
        "type": "text",
        "text": (
            "Answer with one short phrase or sentence. Return only the answer; "
            "do not explain your reasoning."
        ),
    }


@pytest.mark.parametrize(
    ("question", "contract"),
    [
        ("What words are written on the sign?", "visible words"),
        ("What does she say after opening the door?", "spoken content"),
        ("How many cups are on the table?", "count or quantity"),
        ("What happens before the dog runs away?", "temporal relation"),
        ("Why does the pan catch fire?", "cause or procedure"),
        ("What color is the bicycle?", "short phrase or sentence"),
    ],
)
def test_question_aware_policy_is_type_specific_and_label_free(
    tmp_path: Path, question: str, contract: str
) -> None:
    messages = build_controlled_messages(
        question=question,
        frame_paths=(tmp_path / "frame.jpg",),
        frame_timestamps=(0.5,),
        transcript=None,
        answer_policy="question-aware",
    )

    instruction = messages[0]["content"][-1]["text"]
    assert contract in instruction
    assert "Return only the answer" in instruction
    assert "explain your reasoning" in instruction
    serialized = repr(messages).lower()
    assert "reference answer" not in serialized
    assert "ground truth" not in serialized
    assert not any(item.get("type") in {"reference", "label"} for item in messages[0]["content"])


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
