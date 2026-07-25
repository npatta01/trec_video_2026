"""Shared, model-independent prompt behavior for VQA model runners."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def build_controlled_messages(
    *,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    transcript: str | None,
    frame_descriptions: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Build one ordered multimodal message without any evaluation label."""

    if len(frame_paths) != len(frame_timestamps):
        raise ValueError("Frame paths and timestamps must have equal length")
    if frame_descriptions is not None and len(frame_descriptions) != len(frame_paths):
        raise ValueError("Frame descriptions and paths must have equal length")
    if not question.strip():
        raise ValueError("Question cannot be empty")
    default_descriptions = tuple(
        f"Frame at {timestamp:.3f} seconds:" for timestamp in frame_timestamps
    )
    descriptions = (
        default_descriptions
        if frame_descriptions is None
        else tuple(
            description or default
            for description, default in zip(
                frame_descriptions, default_descriptions, strict=True
            )
        )
    )
    ordered = sorted(
        zip(frame_timestamps, frame_paths, descriptions, strict=True)
    )
    content: list[dict[str, str]] = []
    for _, frame_path, description in ordered:
        content.append({"type": "text", "text": description})
        content.append({"type": "image", "url": str(frame_path.resolve())})
    if transcript is not None:
        if not transcript.strip():
            raise ValueError("Transcript must be non-empty when supplied")
        content.append(
            {
                "type": "text",
                "text": "Timestamped audio transcript:\n" + transcript.strip(),
            }
        )
    content.append({"type": "text", "text": "Question: " + question.strip()})
    content.append(
        {
            "type": "text",
            "text": (
                "Answer with one short phrase or sentence. Return only the answer; "
                "do not explain your reasoning."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def normalize_generated_answer(text: str) -> str:
    """Normalize presentation wrappers without changing answer semantics."""

    text = re.sub(r"(?is)^.*?</think>\s*", "", text.strip())
    text = re.sub(r"(?i)^\s*(?:final\s+)?answer\s*:\s*", "", text)
    return " ".join(text.split())
