"""Privacy-safe shared primitives for multiple-choice VQA experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


OPTION_IDS = tuple("ABCD")


def canonical_json(value: Any) -> bytes:
    """Encode a value as stable UTF-8 JSON with a trailing newline."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ranking(
    raw: object, *, option_ids: Sequence[str] = OPTION_IDS
) -> list[str]:
    """Parse only a JSON object containing one complete option permutation."""

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("ranking is not canonical JSON") from error
    else:
        payload = raw
    if not isinstance(payload, Mapping) or set(payload) != {"ranking"}:
        raise ValueError("ranking object must contain only the ranking field")
    ranking = payload.get("ranking")
    expected = list(option_ids)
    if not isinstance(ranking, list) or len(ranking) != len(expected):
        raise ValueError("ranking must contain every option exactly once")
    if not all(isinstance(value, str) for value in ranking) or sorted(
        ranking
    ) != sorted(expected):
        raise ValueError("ranking must be a permutation of the option IDs")
    return list(ranking)


def score_rankings(
    predictions: Mapping[str, Sequence[str] | None], labels: Mapping[str, str]
) -> dict[str, int | float]:
    """Score rankings against frozen option IDs, penalizing invalid rows as zero."""

    valid = 0
    top1 = 0
    reciprocal_rank = 0.0
    for query_id, correct_option_id in labels.items():
        ranking = predictions.get(query_id)
        if ranking is None:
            continue
        parsed = parse_ranking({"ranking": list(ranking)})
        valid += 1
        if parsed[0] == correct_option_id:
            top1 += 1
        reciprocal_rank += 1.0 / (parsed.index(correct_option_id) + 1)
    row_count = len(labels)
    return {
        "row_count": row_count,
        "valid_count": valid,
        "invalid_count": row_count - valid,
        "top1_correct": top1,
        "top1_accuracy": top1 / row_count if row_count else 0.0,
        "mrr": reciprocal_rank / row_count if row_count else 0.0,
    }
