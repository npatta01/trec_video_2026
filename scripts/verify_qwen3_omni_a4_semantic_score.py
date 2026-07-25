#!/usr/bin/env python3
"""Reproduce the tracked Qwen3-Omni a4 scores from ignored annotations."""

from __future__ import annotations

import hashlib
import json
import string
from collections import Counter
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = (
    ROOT
    / "docs/experiments/artifacts/qwen3-omni-a4-semantic-score.json"
)


def _normalize(answer: str) -> str:
    lowered = answer.lower()
    without_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    tokens = [
        token
        for token in without_punctuation.split()
        if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def _exact_match(prediction: str, reference: str) -> float:
    return float(_normalize(prediction) == _normalize(reference))


def _token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _normalize(prediction).split()
    reference_tokens = _normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum(
        (Counter(prediction_tokens) & Counter(reference_tokens)).values()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    score = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    exact_values = []
    f1_values = []
    for example in score["examples"]:
        annotation_path = ROOT / example["annotation_path"]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        reference = annotation["correct_answer"]
        reference_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
        assert reference_hash == example["reference_sha256"]

        exact = _exact_match(example["answer"], reference)
        f1 = _token_f1(example["answer"], reference)
        assert exact == example["normalized_exact_match"]
        assert f1 == example["token_f1"]
        exact_values.append(exact)
        f1_values.append(f1)

    assert fmean(exact_values) == score["aggregate"]["normalized_exact_match"]
    assert fmean(f1_values) == score["aggregate"]["token_f1"]
    print("Qwen3-Omni a4 semantic score reproduces exactly.")


if __name__ == "__main__":
    main()
