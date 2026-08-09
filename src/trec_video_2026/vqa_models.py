"""Shared, model-independent prompt behavior for VQA model runners."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_CURRENT_ANSWER_INSTRUCTION = (
    "Answer with one short phrase or sentence. Return only the answer; "
    "do not explain your reasoning."
)
_QUESTION_AWARE_INSTRUCTIONS = {
    "visible-text": (
        "Answer with the visible words relevant to the question, preserving their "
        "order. Return only the answer; do not explain your reasoning."
    ),
    "speech": (
        "Answer with the spoken content relevant to the question. Return only the "
        "answer; do not explain your reasoning."
    ),
    "count": (
        "Answer with the requested count or quantity. Return only the answer; "
        "do not explain your reasoning."
    ),
    "temporal": (
        "Answer with the requested temporal relation or event. Return only the "
        "answer; do not explain your reasoning."
    ),
    "causal-procedural": (
        "Answer with the requested cause or procedure. Return only the answer; "
        "do not explain your reasoning."
    ),
    "general": (
        "Answer with one short phrase or sentence that directly addresses the "
        "question. Return only the answer; do not explain your reasoning."
    ),
}

_ACTIVE_METHOD_VERB_PATTERN = re.compile(
    r"\b(?:work|happen|make|use|open|close|operate|cook|prepare|"
    r"cause|start|begin|end|change|catch\s+fire)\b",
    re.IGNORECASE,
)
_ACTIVE_METHOD_PREFIX_PATTERN = re.compile(
    r"^\s*how\s+(?:do|does|did|can|should|could)\s+"
    r"(?P<body>.+?)\s*[?.!]?$",
    re.IGNORECASE,
)
_PASSIVE_METHOD_PATTERN = re.compile(
    r"^\s*how\s+(?:is|are|was|were)\s+"
    r".+?\s+(?:made|used|opened|closed|operated|cooked|prepared)\b"
    r"(?:\s+(?:in|with|using|by)\b[^?.!]*)?\s*[?.!]?$",
    re.IGNORECASE,
)
_COMMON_NON_METHOD_PREDICATE_PATTERN = re.compile(
    r"\b(?:look|looks|looked|appear|appears|appeared|seem|seems|seemed|"
    r"color|colors|colored|colour|colours|coloured|hold|holds|held|"
    r"holding|wear|wears|wore|wearing|dress|dresses|dressed|show|"
    r"shows|showed|showing|stand|stands|stood|standing|sit|sits|sat|"
    r"sitting)\b",
    re.IGNORECASE,
)
_COORDINATION_MARKER_PATTERN = re.compile(r"\b(?:and|then|while)\s*$", re.IGNORECASE)

_QUESTION_TYPE_RULES = (
    (
        "visible-text",
        (
            re.compile(r"^\s*(?:what|which)\s+words?\b", re.IGNORECASE),
            re.compile(
                r"^\s*(?:what|which)\s+(?:text|letters?|writing)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:what|which)\s+(?:is|are)\s+(?:the\s+)?"
                r"(?:(?:first|last|next)\s+)?(?:letters?|words?|text|writing)\b",
                re.IGNORECASE,
            ),
            re.compile(r"^\s*(?:read|transcribe)\b", re.IGNORECASE),
            re.compile(r"^\s*what\s+is\s+written\b", re.IGNORECASE),
            re.compile(
                r"^\s*what\s+(?:does|do)\s+the\s+"
                r"(?:sign|poster|label|banner|screen)\s+say\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "speech",
        (
            re.compile(
                r"^\s*(?:what|which)\s+(?:does|did|do|is|are)\s+"
                r"(?:he|she|they|someone|(?:the|this|that|a|an)\s+"
                r"(?:person|man|woman|boy|girl|speaker|narrator))\s+"
                r"say(?:ing)?\b",
                re.IGNORECASE,
            ),
            re.compile(r"^\s*(?:what|which)\s+is\s+being\s+said\b", re.IGNORECASE),
            re.compile(r"^\s*what\s+are\s+they\s+saying\b", re.IGNORECASE),
        ),
    ),
    (
        "count",
        (
            re.compile(r"^\s*how\s+many\b", re.IGNORECASE),
            re.compile(r"^\s*how\s+much\b", re.IGNORECASE),
            re.compile(
                r"^\s*(?:what|which)\s+(?:is|are)\s+the\s+"
                r"(?:number|count)\b",
                re.IGNORECASE,
            ),
            re.compile(r"^\s*count\b", re.IGNORECASE),
        ),
    ),
    (
        "temporal",
        (
            re.compile(
                r"^\s*(?:what|which)\s+(?:happens?|happened|occurs?)\s+"
                r"(?:before|after|when|first|next|then)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:what|which)\s+(?:is|was)\s+the\s+"
                r"(?:first|next|last)\s+(?:event|step|action|scene|"
                r"thing\s+that\s+happens?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*(?:before|after|when|while|during|once|until)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"^\s*how\s+(?:does|do|did|is|are|was|were)\b"
                r".*\b(?:start|starts|begin|begins|end|ends|change|changes)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "causal-procedural",
        (
            re.compile(r"^\s*why\b", re.IGNORECASE),
            re.compile(
                r"^\s*(?:what|which)\s+(?:causes?|caused|leads?)\b",
                re.IGNORECASE,
            ),
            re.compile(r"^\s*how\s+to\b", re.IGNORECASE),
        ),
    ),
)


def _is_active_method_question(question: str) -> bool:
    match = _ACTIVE_METHOD_PREFIX_PATTERN.fullmatch(question)
    if match is None:
        return False

    body = match.group("body").strip()
    for predicate_match in _ACTIVE_METHOD_VERB_PATTERN.finditer(body):
        subject = body[: predicate_match.start()].strip()
        suffix = body[predicate_match.end() :].strip()
        if _is_adjectival_method_candidate(subject, suffix):
            continue
        return True
    return False


def _is_adjectival_method_candidate(subject: str, suffix: str) -> bool:
    if not subject or subject.split()[-1].lower() in {"the", "a", "an"}:
        return True
    if _COMMON_NON_METHOD_PREDICATE_PATTERN.search(subject):
        return True

    suffix_predicate = _COMMON_NON_METHOD_PREDICATE_PATTERN.search(suffix)
    if suffix_predicate is None:
        return False
    return not _COORDINATION_MARKER_PATTERN.search(suffix[: suffix_predicate.start()])


def _is_passive_method_question(question: str) -> bool:
    return _PASSIVE_METHOD_PATTERN.fullmatch(question) is not None


def classify_question_type(question: str) -> str:
    """Classify a question using deterministic, precedence-ordered rules."""

    normalized = " ".join(question.split())
    for question_type, patterns in _QUESTION_TYPE_RULES:
        if any(pattern.search(normalized) for pattern in patterns):
            return question_type
        if question_type == "causal-procedural" and (
            _is_active_method_question(normalized)
            or _is_passive_method_question(normalized)
        ):
            return question_type
    return "general"


def build_controlled_messages(
    *,
    question: str,
    frame_paths: tuple[Path, ...],
    frame_timestamps: tuple[float, ...],
    transcript: str | None,
    frame_descriptions: tuple[str, ...] | None = None,
    answer_policy: str = "current",
) -> list[dict[str, Any]]:
    """Build one ordered multimodal message without any evaluation label."""

    if answer_policy not in {"current", "question-aware"}:
        raise ValueError(f"Unsupported answer policy: {answer_policy}")
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
    ordered = sorted(zip(frame_timestamps, frame_paths, descriptions, strict=True))
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
    instruction = _CURRENT_ANSWER_INSTRUCTION
    if answer_policy == "question-aware":
        instruction = _QUESTION_AWARE_INSTRUCTIONS[classify_question_type(question)]
    content.append(
        {
            "type": "text",
            "text": instruction,
        }
    )
    return [{"role": "user", "content": content}]


def normalize_generated_answer(text: str) -> str:
    """Normalize presentation wrappers without changing answer semantics."""

    text = re.sub(r"(?is)^.*?</think>\s*", "", text.strip())
    text = re.sub(r"(?i)^\s*(?:final\s+)?answer\s*:\s*", "", text)
    return " ".join(text.split())
