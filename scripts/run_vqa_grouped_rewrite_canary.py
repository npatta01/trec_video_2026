#!/usr/bin/env python3
"""Run the fixed, text-only five-query grouped-rewrite canary.

The default path is provider-free.  It reads the frozen submission artifacts,
builds one bounded request, and validates the estimated cost without creating
the private output root.  A provider call is possible only with ``--run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

QUERY_IDS = (1, 310, 1563, 2404, 2421)
MODEL = "gemini-3.7-flash"
SEED = 42
MAX_OUTPUT_TOKENS = 8192
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75
MAX_COST_USD = 0.10
ANSWERS_PER_QUERY = 10
STRATEGY_SEMANTIC_REWRITE_A = "semantic_rewrite_a"
STRATEGY_EXACT_GROUP_HEDGE_B = "exact_group_hedge_b"
STRATEGY_VERIFIED_PARAPHRASE_C = "verified_paraphrase_c"
STRATEGIES = (
    STRATEGY_SEMANTIC_REWRITE_A,
    STRATEGY_EXACT_GROUP_HEDGE_B,
    STRATEGY_VERIFIED_PARAPHRASE_C,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTING_QUESTIONS = _REPO_ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
DEFAULT_GEMINI_OVERLAY = _REPO_ROOT / (
    "data/vqa/benchmarks/deadline-submission-rescue-20260817/"
    "gemini-27-mp4/overlay/predictions-freeze.json"
)
DEFAULT_QWEN_AGGREGATE = _REPO_ROOT / (
    "data/vqa/benchmarks/deadline-submission-full-20260816/"
    "qwen36/aggregate/predictions-freeze.json"
)
DEFAULT_QWEN35_CONTROL = _REPO_ROOT / (
    "data/vqa/submissions/deadline-20260817-ready/"
    "03-qwen35-baseline-control-top10.json"
)
DEFAULT_FUSION_SUBMISSION = _REPO_ROOT / (
    "data/vqa/submissions/deadline-20260817-ready/"
    "05-gemini37-qwen36-fusion-top10.json"
)
DEFAULT_FUSION_PROVENANCE = _REPO_ROOT / (
    "data/vqa/submissions/deadline-20260817-ready/"
    "05-gemini37-qwen36-fusion-top10.provenance.json"
)
PRIVATE_RUN_ROOTS = {
    STRATEGY_SEMANTIC_REWRITE_A: _REPO_ROOT
    / "data/vqa/benchmarks/grouped-rewrite-canary-20260817-semantic_rewrite_a",
    STRATEGY_EXACT_GROUP_HEDGE_B: _REPO_ROOT
    / "data/vqa/benchmarks/grouped-rewrite-canary-20260817-exact_group_hedge_b",
    STRATEGY_VERIFIED_PARAPHRASE_C: _REPO_ROOT
    / "data/vqa/benchmarks/grouped-rewrite-canary-20260817-verified_paraphrase_c",
}
PRIVATE_RUN_ROOT = PRIVATE_RUN_ROOTS[STRATEGY_SEMANTIC_REWRITE_A]

_SUCCESS_STATUSES = frozenset({"succeeded", "success", "passed", "complete"})
_SOURCE_NAMES = (
    "gemini_overlay",
    "qwen_video_plus_transcript",
    "qwen_native_video",
    "qwen35_control",
)
# Predeclared conflict boundaries are frozen into each request and manifest.
MUST_SEPARATE_SOURCE_PAIRS: dict[int, tuple[str, str]] = {
    1563: ("qwen_video_plus_transcript", "qwen_native_video"),
    2404: ("qwen_video_plus_transcript", "qwen_native_video"),
    2421: ("gemini_overlay", "qwen35_control"),
}
_SAFE_RAW_RESPONSE_KEYS = frozenset(
    {"candidates", "modelVersion", "promptFeedback", "responseId", "usageMetadata"}
)
VERIFIER_CLAIM_CATEGORIES = (
    "action",
    "entity",
    "role",
    "object",
    "count",
    "degree",
    "manner",
    "emotion",
    "time",
    "location",
    "negation",
    "modality",
    "causality",
    "other",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "labels",
        "annotations",
        "references",
        "metrics",
        "media",
        "frame_paths",
        "transcript",
        "raw_response_sha256",
        "baseline_ranks_used",
    }
)
_SPACE_RE = re.compile(r"\s+")

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "queries": {
            "type": "array",
            "minItems": len(QUERY_IDS),
            "maxItems": len(QUERY_IDS),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query_id": {"type": "integer"},
                    "groups": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "group_id": {"type": "string"},
                                "source_ids": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["group_id", "source_ids"],
                        },
                    },
                    "answers": {
                        "type": "array",
                        "minItems": ANSWERS_PER_QUERY,
                        "maxItems": ANSWERS_PER_QUERY,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "rank": {"type": "integer"},
                                "answer": {"type": "string"},
                                "source_group": {"type": "string"},
                            },
                            "required": ["rank", "answer", "source_group"],
                        },
                    },
                },
                "required": ["query_id", "groups", "answers"],
            },
        }
    },
    "required": ["queries"],
}

VERIFIER_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verifications": {
            "type": "array",
            "minItems": len(QUERY_IDS),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query_id": {"type": "integer"},
                    "group_id": {"type": "string"},
                    "answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "answer_id": {"type": "string"},
                                "answer_sha256": {"type": "string"},
                                "entailed": {"type": "boolean"},
                                "unsupported_claims": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "category": {
                                                "type": "string",
                                                "enum": list(VERIFIER_CLAIM_CATEGORIES),
                                            },
                                            "claim": {"type": "string"},
                                        },
                                        "required": ["category", "claim"],
                                    },
                                },
                                "supporting_source_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "supporting_quotes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "source_id": {"type": "string"},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["source_id", "quote"],
                                    },
                                },
                            },
                            "required": [
                                "answer_id",
                                "answer_sha256",
                                "entailed",
                                "unsupported_claims",
                                "supporting_source_ids",
                                "supporting_quotes",
                            ],
                        },
                    },
                },
                "required": ["query_id", "group_id", "answers"],
            },
        }
    },
    "required": ["verifications"],
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error


def _rows(value: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = value.get("rows")
        if rows is None:
            rows = value.get("predictions")
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact rows are not an object list: {path}")
    return [dict(row) for row in rows]


def _query_id(row: Mapping[str, Any], *, path: Path) -> int:
    value = row.get("query_id", row.get("Q_ID"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"artifact query ID is invalid: {path}")
    return value


def _video_id(row: Mapping[str, Any], *, path: Path) -> str:
    value = row.get("video_id", row.get("Video_ID"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact video ID is invalid: {path}")
    return value


def _answer(value: Any, *, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact answer is empty: {path}")
    return value


def _status_available(row: Mapping[str, Any]) -> bool:
    return row.get("status") in _SUCCESS_STATUSES


def _by_query(rows: Sequence[Mapping[str, Any]], *, path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        query_id = _query_id(row, path=path)
        if query_id in result:
            raise ValueError(f"duplicate query ID in artifact: {path}")
        result[query_id] = dict(row)
    return result


def _source_candidate(
    *, source_id: str, answer: str, query_id: int
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "original_rank": 1,
        "answer": answer,
        "query_id": query_id,
    }


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in _FORBIDDEN_KEYS or _contains_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip()).casefold()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    body["artifact_sha256"] = _sha256_bytes(_canonical_json(body))
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return _sha256_bytes(serialized.encode("utf-8"))


def _first_source(value: str) -> str:
    return {
        "gemini": "gemini_overlay",
        "qwen_video_plus_transcript": "qwen_video_plus_transcript",
        "qwen_native_video": "qwen_native_video",
        "baseline": "qwen35_control",
    }.get(value, "")


def _validate_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of the two fixed canary strategies")
    return strategy


def _must_separate_source_pairs() -> list[dict[str, Any]]:
    return [
        {"query_id": query_id, "source_ids": list(source_pair)}
        for query_id, source_pair in MUST_SEPARATE_SOURCE_PAIRS.items()
    ]


def _source_artifacts(
    *,
    questions_path: Path,
    gemini_overlay_path: Path,
    qwen_aggregate_path: Path,
    qwen35_control_path: Path,
    fusion_submission_path: Path,
    fusion_provenance_path: Path,
) -> list[dict[str, str]]:
    paths = (
        ("questions", questions_path),
        ("gemini_overlay", gemini_overlay_path),
        ("qwen_aggregate", qwen_aggregate_path),
        ("qwen35_control", qwen35_control_path),
        ("fusion_submission", fusion_submission_path),
        ("fusion_provenance", fusion_provenance_path),
    )
    result: list[dict[str, str]] = []
    for name, raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"source artifact is not a regular file: {name}")
        result.append({"name": name, "path": str(path), "sha256": _sha256_file(path)})
    return result


def _source_artifacts_sha256(artifacts: Sequence[Mapping[str, str]]) -> str:
    return _sha256_bytes(_canonical_json(list(artifacts)))


def _assert_source_artifacts_unchanged(
    artifacts: Sequence[Mapping[str, str]],
    *,
    questions_path: Path,
    gemini_overlay_path: Path,
    qwen_aggregate_path: Path,
    qwen35_control_path: Path,
    fusion_submission_path: Path,
    fusion_provenance_path: Path,
) -> None:
    current = _source_artifacts(
        questions_path=questions_path,
        gemini_overlay_path=gemini_overlay_path,
        qwen_aggregate_path=qwen_aggregate_path,
        qwen35_control_path=qwen35_control_path,
        fusion_submission_path=fusion_submission_path,
        fusion_provenance_path=fusion_provenance_path,
    )
    if list(artifacts) != current:
        raise ValueError("source artifact changed after input freeze")


def select_rows(
    questions_path: Path = DEFAULT_TESTING_QUESTIONS,
    gemini_overlay_path: Path = DEFAULT_GEMINI_OVERLAY,
    qwen_aggregate_path: Path = DEFAULT_QWEN_AGGREGATE,
    qwen35_control_path: Path = DEFAULT_QWEN35_CONTROL,
    fusion_submission_path: Path = DEFAULT_FUSION_SUBMISSION,
    fusion_provenance_path: Path = DEFAULT_FUSION_PROVENANCE,
    query_ids: Sequence[int] = QUERY_IDS,
) -> list[dict[str, Any]]:
    """Select only the five released questions and genuine rank-1 sources."""

    expected_ids = tuple(query_ids)
    if expected_ids != QUERY_IDS:
        raise ValueError("the grouped-rewrite canary panel is fixed")
    questions = _by_query(_rows(_read_json(Path(questions_path)), path=Path(questions_path)), path=Path(questions_path))
    gemini = _by_query(_rows(_read_json(Path(gemini_overlay_path)), path=Path(gemini_overlay_path)), path=Path(gemini_overlay_path))
    qwen_rows = _rows(_read_json(Path(qwen_aggregate_path)), path=Path(qwen_aggregate_path))
    control = _by_query(_rows(_read_json(Path(qwen35_control_path)), path=Path(qwen35_control_path)), path=Path(qwen35_control_path))
    fusion = _by_query(_rows(_read_json(Path(fusion_submission_path)), path=Path(fusion_submission_path)), path=Path(fusion_submission_path))
    provenance_payload = _read_json(Path(fusion_provenance_path))
    provenance = _by_query(
        _rows(provenance_payload, path=Path(fusion_provenance_path)),
        path=Path(fusion_provenance_path),
    )

    qwen: dict[tuple[int, str], dict[str, Any]] = {}
    for row in qwen_rows:
        query_id = _query_id(row, path=Path(qwen_aggregate_path))
        condition = row.get("condition_id")
        if isinstance(condition, str) and condition in {"video-plus-transcript", "native-video"}:
            key = (
                query_id,
                "qwen_video_plus_transcript"
                if condition == "video-plus-transcript"
                else "qwen_native_video",
            )
            if key not in qwen and _status_available(row):
                qwen[key] = dict(row)

    selected: list[dict[str, Any]] = []
    for query_id in expected_ids:
        source_question = questions.get(query_id)
        source_fusion = fusion.get(query_id)
        source_provenance = provenance.get(query_id)
        source_control = control.get(query_id)
        if not all((source_question, source_fusion, source_provenance, source_control)):
            raise ValueError(f"panel artifact is missing query {query_id}")
        video_id = _video_id(source_question, path=Path(questions_path))
        if _video_id(source_fusion, path=Path(fusion_submission_path)) != video_id:
            raise ValueError(f"video ID mismatch for query {query_id}")
        question = source_question.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question is empty for query {query_id}")
        fusion_answers = source_fusion.get("Answers")
        if not isinstance(fusion_answers, list) or not fusion_answers:
            raise ValueError(f"fusion rank-1 is missing for query {query_id}")
        fusion_rank_one = next((item for item in fusion_answers if item.get("Rank") == 1), None)
        if not isinstance(fusion_rank_one, Mapping):
            raise ValueError(f"fusion rank-1 is missing for query {query_id}")
        rank1_answer = _answer(fusion_rank_one.get("Answer"), path=Path(fusion_submission_path))
        selected_sources = source_provenance.get("selected_sources")
        if not isinstance(selected_sources, list) or not selected_sources:
            raise ValueError(f"fusion rank-1 provenance is missing for query {query_id}")
        rank1_source = _first_source(selected_sources[0]) if isinstance(selected_sources[0], str) else ""
        if rank1_source not in _SOURCE_NAMES:
            raise ValueError(f"fusion rank-1 source is invalid for query {query_id}")

        candidates: list[dict[str, Any]] = []
        overlay = gemini.get(query_id)
        if overlay and _status_available(overlay):
            candidates.append(
                _source_candidate(
                    source_id="gemini_overlay",
                    answer=_answer(overlay.get("answer"), path=Path(gemini_overlay_path)),
                    query_id=query_id,
                )
            )
        for source_id in ("qwen_video_plus_transcript", "qwen_native_video"):
            qwen_row = qwen.get((query_id, source_id))
            if qwen_row:
                candidates.append(
                    _source_candidate(
                        source_id=source_id,
                        answer=_answer(qwen_row.get("answer"), path=Path(qwen_aggregate_path)),
                        query_id=query_id,
                    )
                )
        control_answers = source_control.get("Answers")
        if not isinstance(control_answers, list):
            raise ValueError(f"control answers are missing for query {query_id}")
        control_rank_one = next((item for item in control_answers if item.get("Rank") == 1), None)
        if not isinstance(control_rank_one, Mapping):
            raise ValueError(f"control rank-1 is missing for query {query_id}")
        candidates.append(
            _source_candidate(
                source_id="qwen35_control",
                answer=_answer(control_rank_one.get("Answer"), path=Path(qwen35_control_path)),
                query_id=query_id,
            )
        )
        candidate_sources = {candidate["source_id"] for candidate in candidates}
        if rank1_source not in candidate_sources:
            raise ValueError(f"rank-1 source is unavailable for query {query_id}")
        rank1_candidate = next(item for item in candidates if item["source_id"] == rank1_source)
        if rank1_candidate["answer"] != rank1_answer:
            raise ValueError(f"fusion rank-1 answer disagrees with source for query {query_id}")
        row = {
            "query_id": query_id,
            "video_id": video_id,
            "question": question.strip(),
            "rank1_source": rank1_source,
            "rank1_answer": rank1_answer,
            "source_candidates": candidates,
        }
        if _contains_forbidden_key(row):
            raise ValueError(f"forbidden field entered selected row for query {query_id}")
        selected.append(row)
    return selected


def _estimate_cost(request: Mapping[str, Any]) -> tuple[int, int, float]:
    prompt_tokens = max(1, math.ceil(len(_canonical_json(request)) / 4))
    output_tokens = MAX_OUTPUT_TOKENS
    cost = prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    return prompt_tokens, output_tokens, round(cost, 8)


def build_request(
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str = STRATEGY_SEMANTIC_REWRITE_A,
) -> dict[str, Any]:
    """Build one text-only JSON request for all five queries."""

    _validate_strategy(strategy)
    if len(rows) != len(QUERY_IDS) or [row.get("query_id") for row in rows] != list(QUERY_IDS):
        raise ValueError("request rows must be the fixed five-query panel in order")
    queries: list[dict[str, Any]] = []
    for row in rows:
        candidates = row.get("source_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("each request row needs source candidates")
        queries.append(
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "rank1_source": row["rank1_source"],
                "rank1_answer": row["rank1_answer"],
                "source_candidates": [
                    {
                        "source_id": candidate["source_id"],
                        "original_rank": candidate["original_rank"],
                        "answer": candidate["answer"],
                    }
                    for candidate in candidates
                ],
                "requested_answer_count": ANSWERS_PER_QUERY,
            }
        )
    instructions = (
        "For each query, group every source candidate exactly once. Keep incompatible "
        "meanings in separate groups. Return exactly ten distinct answers; rank 1 must "
        "copy the frozen rank1_answer byte-for-byte. Every answer must name its group. "
        "Do not add facts, metadata, or fields outside the requested JSON schema. "
        "Never place both members of a predeclared source pair in one group: "
        f"{json.dumps(_must_separate_source_pairs(), sort_keys=True)}."
    )
    if strategy == STRATEGY_EXACT_GROUP_HEDGE_B:
        instructions += (
            " Before adding paraphrases, represent every semantic group with at least one "
            "answer copied exactly from one of that group's source candidates."
        )
    if strategy == STRATEGY_VERIFIED_PARAPHRASE_C:
        instructions += (
            " This is stage one of a two-call verified paraphrase strategy. A separate "
            "isolated verifier will check every generated answer for entailment. The "
            "verifier must reject added manner, emotion, object, count, timing, location, "
            "role, or degree claims; synonyms are allowed."
        )
    config = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "seed": SEED,
        "response_json_schema": RESPONSE_JSON_SCHEMA,
    }
    request: dict[str, Any] = {
        "model": MODEL,
        "seed": SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_json_schema": RESPONSE_JSON_SCHEMA,
        "config": config,
        "strategy_id": strategy,
        "request_count": 2 if strategy == STRATEGY_VERIFIED_PARAPHRASE_C else 1,
        "must_separate_source_pairs": _must_separate_source_pairs(),
        "instructions": instructions,
        "queries": queries,
    }
    prompt_tokens, output_tokens, estimated_cost = _estimate_cost(request)
    request["estimated_prompt_tokens"] = prompt_tokens
    request["estimated_output_tokens"] = output_tokens
    request["estimated_cost_usd"] = estimated_cost
    if strategy == STRATEGY_VERIFIED_PARAPHRASE_C:
        verifier_prompt_tokens = max(
            1,
            math.ceil(
                len(
                    _canonical_json(
                        {
                            "questions": [row["question"] for row in rows],
                            "groups": [row["source_candidates"] for row in rows],
                            "answer_count": len(rows) * ANSWERS_PER_QUERY,
                        }
                    )
                )
                / 4
            ),
        )
        verifier_cost = verifier_prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
        request["estimated_verifier_prompt_tokens"] = verifier_prompt_tokens
        request["estimated_verifier_output_tokens"] = output_tokens
        request["estimated_total_cost_usd"] = round(estimated_cost + verifier_cost, 8)
        request["verifier_response_json_schema"] = VERIFIER_RESPONSE_JSON_SCHEMA
        if request["estimated_total_cost_usd"] > MAX_COST_USD:
            raise ValueError("estimated combined cost exceeds $0.10 cap")
    if estimated_cost > MAX_COST_USD:
        raise ValueError("estimated cost exceeds $0.10 cap")
    return request


def _usage_value(usage: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _normalize_usage(usage: Any) -> dict[str, int]:
    def value(*names: str) -> int | None:
        if isinstance(usage, Mapping):
            return _usage_value(usage, *names)
        for name in names:
            candidate = getattr(usage, name, None)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                return candidate
        return None

    prompt = value("prompt_tokens", "promptTokenCount", "prompt_token_count")
    response = value(
        "response_tokens",
        "responseTokenCount",
        "response_token_count",
        "output_tokens",
        "candidatesTokenCount",
        "candidates_token_count",
    )
    total = value("total_tokens", "totalTokenCount", "total_token_count")
    thoughts = value("thoughts_tokens", "thoughtsTokenCount", "thoughts_token_count")
    if prompt is None:
        return {}
    if response is None and total is None:
        return {}
    conservative_output = max(response or 0, (total - prompt) if total is not None else 0)
    normalized = {"prompt_tokens": prompt, "output_tokens": conservative_output}
    if response is not None:
        normalized["response_tokens"] = response
    if total is not None:
        normalized["total_tokens"] = total
    if thoughts is not None:
        normalized["thoughts_tokens"] = thoughts
    return normalized


def _observed_usage(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = _usage_value(usage, "prompt_tokens")
    output = _usage_value(usage, "output_tokens")
    if prompt is None or output is None:
        return None
    return prompt, output


def _observed_cost(payload: Mapping[str, Any]) -> float | None:
    usage = _observed_usage(payload)
    if usage is None:
        return None
    prompt_tokens, output_tokens = usage
    if output_tokens > MAX_OUTPUT_TOKENS:
        raise ValueError("observed output tokens exceed cap")
    return round(
        prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION,
        8,
    )


def validate_response(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    strategy: str = STRATEGY_SEMANTIC_REWRITE_A,
) -> dict[str, Any]:
    """Validate one parsed provider response against the frozen request rows."""

    _validate_strategy(strategy)
    if not isinstance(payload, Mapping):
        raise ValueError("response must be an object")
    if set(payload) - {"queries", "usage"}:
        raise ValueError("response contains fields outside the frozen schema")
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != len(rows):
        raise ValueError("response query count is invalid")
    expected = {int(row["query_id"]): row for row in rows}
    seen_queries: set[int] = set()
    validated_queries: list[dict[str, Any]] = []
    for query in queries:
        if not isinstance(query, Mapping):
            raise ValueError("response query must be an object")
        query_id = query.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id not in expected:
            raise ValueError("response query ID crosses the frozen panel")
        if query_id in seen_queries:
            raise ValueError("response contains duplicate query IDs")
        seen_queries.add(query_id)
        if set(query) != {"query_id", "groups", "answers"}:
            raise ValueError("response query schema is not exact")
        row = expected[query_id]
        groups = query.get("groups")
        answers = query.get("answers")
        if not isinstance(groups, list) or not isinstance(answers, list):
            raise ValueError("response query groups and answers are required")
        candidate_by_source = {
            candidate.get("source_id"): candidate
            for candidate in row.get("source_candidates", [])
            if isinstance(candidate, Mapping)
        }
        if len(candidate_by_source) != len(row.get("source_candidates", [])):
            raise ValueError("source candidates are not unique")
        group_by_id: dict[str, Mapping[str, Any]] = {}
        covered: list[str] = []
        for group in groups:
            if not isinstance(group, Mapping):
                raise ValueError("response group must be an object")
            if set(group) != {"group_id", "source_ids"}:
                raise ValueError("response group schema is not exact")
            group_id = group.get("group_id")
            source_ids = group.get("source_ids")
            if not isinstance(group_id, str) or not group_id or group_id in group_by_id:
                raise ValueError("response group ID is invalid")
            if not isinstance(source_ids, list) or not source_ids or not all(isinstance(item, str) for item in source_ids):
                raise ValueError("response group source IDs are invalid")
            if len(set(source_ids)) != len(source_ids):
                raise ValueError("response group repeats a source candidate")
            if any(source_id not in candidate_by_source for source_id in source_ids):
                raise ValueError("response group references an unknown source candidate")
            if any(source_id in covered for source_id in source_ids):
                raise ValueError("source candidate is grouped more than once")
            covered.extend(source_ids)
            group_by_id[group_id] = group
        if set(covered) != set(candidate_by_source):
            raise ValueError("source candidate coverage is incomplete")
        for pair_query_id, (left_source, right_source) in MUST_SEPARATE_SOURCE_PAIRS.items():
            if pair_query_id != query_id:
                continue
            if left_source not in candidate_by_source or right_source not in candidate_by_source:
                continue
            left_group = next(
                group_id
                for group_id, group in group_by_id.items()
                if left_source in group["source_ids"]
            )
            right_group = next(
                group_id
                for group_id, group in group_by_id.items()
                if right_source in group["source_ids"]
            )
            if left_group == right_group:
                raise ValueError(
                    f"predeclared source pair for query {query_id} must remain separate"
                )
        if len(answers) != ANSWERS_PER_QUERY:
            raise ValueError("each query must contain exactly ten answers")
        rank_values: list[int] = []
        normalized_answers: set[str] = set()
        for answer in answers:
            if not isinstance(answer, Mapping):
                raise ValueError("response answer must be an object")
            if set(answer) != {"rank", "answer", "source_group"}:
                raise ValueError("response answer schema is not exact")
            rank = answer.get("rank")
            text = answer.get("answer")
            source_group = answer.get("source_group")
            if isinstance(rank, bool) or not isinstance(rank, int) or not isinstance(text, str) or not text.strip():
                raise ValueError("response answer fields are invalid")
            if not isinstance(source_group, str) or source_group not in group_by_id:
                raise ValueError("response answer references an unknown source group")
            rank_values.append(rank)
            normalized = _normalize(text)
            if normalized in normalized_answers:
                raise ValueError("answers must be ten distinct normalized strings")
            normalized_answers.add(normalized)
        if rank_values != list(range(1, ANSWERS_PER_QUERY + 1)):
            raise ValueError("response ranks must be exactly 1 through 10")
        represented_groups = {answer["source_group"] for answer in answers}
        if represented_groups != set(group_by_id):
            raise ValueError("every group must be represented by an answer")
        if strategy in {
            STRATEGY_EXACT_GROUP_HEDGE_B,
            STRATEGY_VERIFIED_PARAPHRASE_C,
        }:
            for group_id, group in group_by_id.items():
                exact_sources = {
                    str(candidate_by_source[source_id]["answer"])
                    for source_id in group["source_ids"]
                }
                if not any(
                    answer["source_group"] == group_id and answer["answer"] in exact_sources
                    for answer in answers
                ):
                    raise ValueError("every group needs an exact source answer")
        if answers[0]["answer"] != row["rank1_answer"]:
            raise ValueError("rank 1 must preserve the frozen answer byte-for-byte")
        rank1_source = row["rank1_source"]
        rank1_group = next(
            group_id for group_id, group in group_by_id.items() if rank1_source in group["source_ids"]
        )
        if answers[0]["source_group"] != rank1_group:
            raise ValueError("rank 1 source group does not preserve the frozen source")
        validated_queries.append(
            {
                "query_id": query_id,
                "groups": [dict(group) for group in groups],
                "answers": [dict(answer) for answer in answers],
            }
        )
    if seen_queries != set(expected):
        raise ValueError("response is missing a frozen query")
    usage = _observed_usage(payload)
    observed_cost: float | None = None
    if usage is not None:
        prompt_tokens, output_tokens = usage
        if output_tokens > MAX_OUTPUT_TOKENS:
            raise ValueError("observed output tokens exceed cap")
        observed_cost = round(
            prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
            + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION,
            8,
        )
        if observed_cost > MAX_COST_USD:
            raise ValueError("observed cost exceeds $0.10 cap")
    result: dict[str, Any] = {
        "status": "validated",
        "query_count": len(validated_queries),
        "answer_count": len(validated_queries) * ANSWERS_PER_QUERY,
        "source_group_coverage": "complete",
        "queries": validated_queries,
    }
    if usage is not None:
        result["prompt_tokens"] = usage[0]
        result["output_tokens"] = usage[1]
        result["observed_cost_usd"] = observed_cost
    return result


def build_verifier_request(
    rows: Sequence[Mapping[str, Any]],
    generated: Mapping[str, Any],
    *,
    strategy: str = STRATEGY_VERIFIED_PARAPHRASE_C,
) -> dict[str, Any]:
    """Build the isolated second-stage request for verified paraphrase C."""

    if strategy != STRATEGY_VERIFIED_PARAPHRASE_C:
        raise ValueError("verifier request is only defined for verified paraphrase C")
    validated = (
        generated
        if generated.get("status") == "validated"
        else validate_response(generated, rows, strategy=strategy)
    )
    rows_by_id = {int(row["query_id"]): row for row in rows}
    candidates_by_id = {
        query_id: {
            candidate["source_id"]: candidate
            for candidate in row["source_candidates"]
        }
        for query_id, row in rows_by_id.items()
    }
    items: list[dict[str, Any]] = []
    for query in validated["queries"]:
        query_id = int(query["query_id"])
        row = rows_by_id[query_id]
        source_candidates = candidates_by_id[query_id]
        answers_by_group: dict[str, list[Mapping[str, Any]]] = {}
        for answer in query["answers"]:
            answers_by_group.setdefault(answer["source_group"], []).append(answer)
        for group in query["groups"]:
            group_id = group["group_id"]
            group_sources = [source_candidates[source_id] for source_id in group["source_ids"]]
            verifier_answers = []
            for answer in answers_by_group[group_id]:
                text = answer["answer"]
                answer_id = _sha256_bytes(
                    f"{query_id}\0{group_id}\0{text}".encode("utf-8")
                )
                verifier_answers.append(
                    {
                        "answer_id": answer_id,
                        "answer_sha256": _sha256_bytes(text.encode("utf-8")),
                        "text": text,
                    }
                )
            items.append(
                {
                    "query_id": query_id,
                    "group_id": group_id,
                    "question": row["question"],
                    "source_candidates": [
                        {"source_id": candidate["source_id"], "answer": candidate["answer"]}
                        for candidate in group_sources
                    ],
                    "answers": verifier_answers,
                }
            )
    random.Random(SEED).shuffle(items)
    request: dict[str, Any] = {
        "model": MODEL,
        "seed": SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_json_schema": VERIFIER_RESPONSE_JSON_SCHEMA,
        "strategy_id": strategy,
        "stage": "verifier",
        "request_count": 2,
        "instructions": (
            "Verify each cited answer independently against only its question and the "
            "one cited group's source candidates. Synonyms are allowed, but reject any "
            "unsupported atomic action, entity, role, object, count, degree, manner, "
            "emotion, time, location, negation, modality, or causality claim. Return "
            "exact answer IDs and hashes, supporting source IDs, and exact supporting "
            "quotes. Do not use other groups or generator rationale."
        ),
        "items": items,
    }
    prompt_tokens, output_tokens, estimated_cost = _estimate_cost(request)
    request["estimated_prompt_tokens"] = prompt_tokens
    request["estimated_output_tokens"] = output_tokens
    request["estimated_cost_usd"] = estimated_cost
    return request


def validate_verifier_response(
    payload: Mapping[str, Any], verifier_request: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize all fifty verdicts and fail closed on semantic rejection."""

    if set(payload) - {"verifications", "usage"}:
        raise ValueError("verifier response contains fields outside the frozen schema")
    verifications = payload.get("verifications")
    if not isinstance(verifications, list):
        raise ValueError("verifier response verifications are required")
    expected: dict[tuple[int, str, str], dict[str, Any]] = {}
    for item in verifier_request["items"]:
        source_by_id = {candidate["source_id"]: candidate for candidate in item["source_candidates"]}
        for answer in item["answers"]:
            expected[(int(item["query_id"]), item["group_id"], answer["answer_id"])] = {
                "item": item,
                "answer": answer,
                "source_by_id": source_by_id,
                }
    if len(expected) != len(QUERY_IDS) * ANSWERS_PER_QUERY:
        raise ValueError("verifier request must contain exactly fifty answers")
    expected_item_keys = {(key[0], key[1]) for key in expected}
    seen: set[tuple[int, str, str]] = set()
    seen_items: set[tuple[int, str]] = set()
    normalized_verifications: list[dict[str, Any]] = []
    semantic_rejected = False
    category_counts: dict[str, int] = {}
    quote_count = 0
    for verification in verifications:
        if set(verification) != {"query_id", "group_id", "answers"}:
            raise ValueError("verifier item schema is not exact")
        query_id = verification.get("query_id")
        group_id = verification.get("group_id")
        if not isinstance(query_id, int) or not isinstance(group_id, str):
            raise ValueError("verifier item identity is invalid")
        item_key = (query_id, group_id)
        if item_key not in expected_item_keys or item_key in seen_items:
            raise ValueError("verifier group coverage is invalid")
        seen_items.add(item_key)
        answers = verification.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError("verifier group answers are required")
        normalized_answers: list[dict[str, Any]] = []
        for answer in answers:
            required = {
                "answer_id",
                "answer_sha256",
                "entailed",
                "unsupported_claims",
                "supporting_source_ids",
                "supporting_quotes",
            }
            if set(answer) != required:
                raise ValueError("verifier answer schema is not exact")
            key = (query_id, group_id, answer["answer_id"])
            expected_answer = expected.get(key)
            if expected_answer is None or key in seen:
                raise ValueError("verifier answer coverage is invalid")
            seen.add(key)
            if answer["answer_sha256"] != expected_answer["answer"]["answer_sha256"]:
                raise ValueError("verifier answer hash does not match frozen text")
            if not isinstance(answer["entailed"], bool):
                raise ValueError("verifier entailment flag is invalid")
            unsupported = answer["unsupported_claims"]
            if not isinstance(unsupported, list):
                raise ValueError("verifier unsupported claims are invalid")
            normalized_claims: list[dict[str, str]] = []
            for claim in unsupported:
                if (
                    not isinstance(claim, Mapping)
                    or set(claim) != {"category", "claim"}
                    or claim["category"] not in VERIFIER_CLAIM_CATEGORIES
                    or not isinstance(claim["claim"], str)
                    or not claim["claim"].strip()
                ):
                    raise ValueError("verifier unsupported claim schema is invalid")
                category = str(claim["category"])
                normalized_claims.append(
                    {"category": category, "claim": str(claim["claim"])}
                )
                category_counts[category] = category_counts.get(category, 0) + 1
            if answer["entailed"] is not True or normalized_claims:
                semantic_rejected = True
            source_by_id = expected_answer["source_by_id"]
            source_ids = answer["supporting_source_ids"]
            if (
                not isinstance(source_ids, list)
                or not source_ids
                or len(set(source_ids)) != len(source_ids)
                or not set(source_ids) <= set(source_by_id)
            ):
                raise ValueError("verifier supporting source IDs are invalid")
            quotes = answer["supporting_quotes"]
            if not isinstance(quotes, list) or not quotes:
                raise ValueError("verifier supporting quotes are required")
            normalized_quotes: list[dict[str, str]] = []
            for quote in quotes:
                if not isinstance(quote, Mapping) or set(quote) != {"source_id", "quote"}:
                    raise ValueError("verifier supporting quote schema is not exact")
                if quote["source_id"] not in source_ids:
                    raise ValueError("verifier quote source is not cited")
                source = source_by_id.get(quote["source_id"])
                if (
                    source is None
                    or not isinstance(quote["quote"], str)
                    or not quote["quote"].strip()
                    or quote["quote"] not in source["answer"]
                ):
                    semantic_rejected = True
                normalized_quotes.append(
                    {"source_id": str(quote["source_id"]), "quote": str(quote["quote"])}
                )
                quote_count += 1
            normalized_answers.append(
                {
                    "answer_id": answer["answer_id"],
                    "answer_sha256": answer["answer_sha256"],
                    "entailed": answer["entailed"],
                    "unsupported_claims": normalized_claims,
                    "supporting_source_ids": list(source_ids),
                    "supporting_quotes": normalized_quotes,
                }
            )
        normalized_verifications.append(
            {"query_id": query_id, "group_id": group_id, "answers": normalized_answers}
        )
    if seen != set(expected) or seen_items != {(key[0], key[1]) for key in expected}:
        raise ValueError("verifier response coverage is incomplete")
    result: dict[str, Any] = {
        "status": "verification_rejected" if semantic_rejected else "verified",
        "answer_count": len(seen),
        "verification_coverage": "complete",
        "verifications": normalized_verifications,
        "verification_counts": {
            "total": len(seen),
            "entailed": sum(
                answer["entailed"]
                for verification in normalized_verifications
                for answer in verification["answers"]
            ),
            "rejected": sum(
                not answer["entailed"] or bool(answer["unsupported_claims"])
                for verification in normalized_verifications
                for answer in verification["answers"]
            ),
            "supporting_quotes": quote_count,
        },
        "unsupported_claim_categories": category_counts,
    }
    usage = _observed_usage(payload)
    if usage is not None:
        result["prompt_tokens"] = usage[0]
        result["output_tokens"] = usage[1]
    return result


def _response_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        text = raw.get("text")
        usage = raw.get("usage") or raw.get("usage_metadata")
        if isinstance(text, str):
            payload_text = text.strip()
        elif isinstance(raw.get("queries"), list):
            payload = dict(raw)
            normalized_usage = _normalize_usage(usage) if usage is not None else {}
            if normalized_usage:
                payload["usage"] = normalized_usage
            return payload
        else:
            raise ValueError("provider response has no JSON text")
    else:
        text = getattr(raw, "text", None)
        usage = getattr(raw, "usage_metadata", None)
        if not isinstance(text, str):
            raise ValueError("provider response has no JSON text")
        payload_text = text.strip()
    normalized_usage = _normalize_usage(usage) if usage is not None else {}
    if payload_text.startswith("```"):
        payload_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload_text, flags=re.IGNORECASE)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("provider response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("provider response JSON must be an object")
    if normalized_usage:
        payload["usage"] = normalized_usage
    return payload


def _safe_raw(raw: Any) -> dict[str, Any]:
    try:
        from google.genai import types
    except ImportError:
        types = None
    if types is not None and isinstance(raw, types.GenerateContentResponse):
        dumped = raw.model_dump(mode="json", by_alias=True, exclude_none=True)
        result = {
            key: dumped[key] for key in _SAFE_RAW_RESPONSE_KEYS if key in dumped
        }
        usage = getattr(raw, "usage_metadata", None)
        normalized_usage = _normalize_usage(usage) if usage is not None else {}
        if normalized_usage:
            result["usageMetadata"] = normalized_usage
        return result
    if isinstance(raw, Mapping):
        result = json.loads(json.dumps(dict(raw), default=str))
        usage = raw.get("usage") or raw.get("usage_metadata")
        normalized_usage = _normalize_usage(usage) if usage is not None else {}
        if normalized_usage:
            result["usage_metadata"] = normalized_usage
        return result
    result: dict[str, Any] = {}
    for name in ("text", "model_version", "response_id", "finish_reason"):
        value = getattr(raw, name, None)
        if value is not None:
            result[name] = json.loads(json.dumps(value, default=str))
    usage = getattr(raw, "usage_metadata", None)
    normalized_usage = _normalize_usage(usage) if usage is not None else {}
    if normalized_usage:
        result["usage_metadata"] = normalized_usage
    return result


def _live_call(
    client: Any,
    request: Mapping[str, Any],
    *,
    response_schema: Mapping[str, Any] | None = None,
) -> Any:
    schema = response_schema or RESPONSE_JSON_SCHEMA
    try:
        from google.genai import types

        config: Any = types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=schema,
            seed=SEED,
        )
    except ImportError:
        config = {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "seed": SEED,
        }
    body = {"instructions": request["instructions"]}
    if "queries" in request:
        body["queries"] = request["queries"]
    elif "items" in request:
        body["items"] = request["items"]
    else:
        raise ValueError("provider request has no query or verifier items")
    return client.models.generate_content(
        model=MODEL,
        contents=json.dumps(body, ensure_ascii=False),
        config=config,
    )


def run_canary(
    questions_path: Path = DEFAULT_TESTING_QUESTIONS,
    gemini_overlay_path: Path = DEFAULT_GEMINI_OVERLAY,
    qwen_aggregate_path: Path = DEFAULT_QWEN_AGGREGATE,
    qwen35_control_path: Path = DEFAULT_QWEN35_CONTROL,
    fusion_submission_path: Path = DEFAULT_FUSION_SUBMISSION,
    fusion_provenance_path: Path = DEFAULT_FUSION_PROVENANCE,
    output_root: Path | None = None,
    *,
    client: Any | None = None,
    strategy: str = STRATEGY_SEMANTIC_REWRITE_A,
    run: bool = False,
    dry_run: bool | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    _validate_strategy(strategy)
    if dry_run is not None:
        run = not dry_run
    source_artifacts = _source_artifacts(
        questions_path=Path(questions_path),
        gemini_overlay_path=Path(gemini_overlay_path),
        qwen_aggregate_path=Path(qwen_aggregate_path),
        qwen35_control_path=Path(qwen35_control_path),
        fusion_submission_path=Path(fusion_submission_path),
        fusion_provenance_path=Path(fusion_provenance_path),
    )
    rows = select_rows(
        questions_path,
        gemini_overlay_path,
        qwen_aggregate_path,
        qwen35_control_path,
        fusion_submission_path,
        fusion_provenance_path,
    )
    request = build_request(rows, strategy=strategy)
    request["source_artifacts"] = source_artifacts
    request["source_artifacts_sha256"] = _source_artifacts_sha256(source_artifacts)
    if not run:
        result = {
            "status": "dry_run",
            "strategy_id": strategy,
            "query_count": len(rows),
            "request_count": request["request_count"],
            "requested_answer_count": len(rows) * ANSWERS_PER_QUERY,
            "estimated_prompt_tokens": request["estimated_prompt_tokens"],
            "estimated_output_tokens": request["estimated_output_tokens"],
            "estimated_cost_usd": request["estimated_cost_usd"],
            "source_artifacts": source_artifacts,
            "source_artifacts_sha256": request["source_artifacts_sha256"],
            "must_separate_source_pairs": request["must_separate_source_pairs"],
            "output_root_created": False,
        }
        if strategy == STRATEGY_VERIFIED_PARAPHRASE_C:
            result["estimated_verifier_prompt_tokens"] = request[
                "estimated_verifier_prompt_tokens"
            ]
            result["estimated_verifier_output_tokens"] = request[
                "estimated_verifier_output_tokens"
            ]
            result["estimated_total_cost_usd"] = request["estimated_total_cost_usd"]
            result["verifier_response_json_schema"] = request[
                "verifier_response_json_schema"
            ]
        return result
    if client is None:
        raise ValueError("provider client is required only for --run")
    root = Path(output_root) if output_root is not None else PRIVATE_RUN_ROOTS[strategy]
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"private output root already exists: {root}")
    root.mkdir(parents=True)
    created = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        input_hash = _write_json_exclusive(
            root / "inputs-freeze.json",
            {
                "schema_version": 1,
                "status": "inputs_frozen",
                "created_at": created,
                "strategy_id": strategy,
                "source_artifacts": source_artifacts,
                "source_artifacts_sha256": request["source_artifacts_sha256"],
                "must_separate_source_pairs": request["must_separate_source_pairs"],
                "rows": rows,
            },
        )
        request_hash = _write_json_exclusive(root / "request.json", request)
        _assert_source_artifacts_unchanged(
            source_artifacts,
            questions_path=Path(questions_path),
            gemini_overlay_path=Path(gemini_overlay_path),
            qwen_aggregate_path=Path(qwen_aggregate_path),
            qwen35_control_path=Path(qwen35_control_path),
            fusion_submission_path=Path(fusion_submission_path),
            fusion_provenance_path=Path(fusion_provenance_path),
        )
        raw = _live_call(client, request)
        raw_hash = _write_json_exclusive(
            root / "raw-response.json",
            {"strategy_id": strategy, "provider_response": _safe_raw(raw)},
        )
        parsed_payload = _response_payload(raw)
        parsed = validate_response(parsed_payload, rows, strategy=strategy)
        if _observed_usage(parsed_payload) is None:
            raise ValueError("provider usage metadata is required to freeze live evidence")
        parsed_hash = _write_json_exclusive(
            root / "parsed-response.json", {"strategy_id": strategy, "result": parsed}
        )
        verifier_result: dict[str, Any] | None = None
        verifier_request_hash: str | None = None
        verifier_raw_hash: str | None = None
        verifier_parsed_hash: str | None = None
        verifier_observed_cost: float | None = None
        if strategy == STRATEGY_VERIFIED_PARAPHRASE_C:
            verifier_request = build_verifier_request(
                rows, parsed, strategy=STRATEGY_VERIFIED_PARAPHRASE_C
            )
            verifier_request["source_artifacts"] = source_artifacts
            verifier_request["source_artifacts_sha256"] = request["source_artifacts_sha256"]
            verifier_request["must_separate_source_pairs"] = request[
                "must_separate_source_pairs"
            ]
            verifier_request_hash = _write_json_exclusive(
                root / "verifier-request.json", verifier_request
            )
            _assert_source_artifacts_unchanged(
                source_artifacts,
                questions_path=Path(questions_path),
                gemini_overlay_path=Path(gemini_overlay_path),
                qwen_aggregate_path=Path(qwen_aggregate_path),
                qwen35_control_path=Path(qwen35_control_path),
                fusion_submission_path=Path(fusion_submission_path),
                fusion_provenance_path=Path(fusion_provenance_path),
            )
            verifier_raw = _live_call(
                client,
                verifier_request,
                response_schema=VERIFIER_RESPONSE_JSON_SCHEMA,
            )
            verifier_raw_hash = _write_json_exclusive(
                root / "verifier-raw-response.json",
                {"strategy_id": strategy, "provider_response": _safe_raw(verifier_raw)},
            )
            verifier_payload = _response_payload(verifier_raw)
            verifier_result = validate_verifier_response(verifier_payload, verifier_request)
            verifier_observed_cost = _observed_cost(verifier_payload)
            if verifier_observed_cost is None:
                raise ValueError("verifier usage metadata is required to freeze live evidence")
            verifier_parsed_hash = _write_json_exclusive(
                root / "verifier-parsed-response.json",
                {"strategy_id": strategy, "result": verifier_result},
            )
        observed_cost = _observed_cost(parsed_payload)
        if observed_cost is None:
            raise ValueError("provider usage metadata is required to freeze live evidence")
        combined_observed_cost = observed_cost + (verifier_observed_cost or 0.0)
        if combined_observed_cost > MAX_COST_USD:
            raise ValueError("combined observed cost exceeds $0.10 cap")
        final_status = (
            "verification_rejected"
            if verifier_result is not None and verifier_result["status"] == "verification_rejected"
            else "predictions_frozen"
        )
        summary = {
            "schema_version": 1,
            "status": final_status,
            "created_at": created,
            "strategy_id": strategy,
            "model": MODEL,
            "request_count": request["request_count"],
            "query_count": len(rows),
            "answer_count": len(rows) * ANSWERS_PER_QUERY,
            "input_freeze_sha256": input_hash,
            "request_sha256": request_hash,
            "raw_response_sha256": raw_hash,
            "parsed_response_sha256": parsed_hash,
            "source_artifacts": source_artifacts,
            "source_artifacts_sha256": request["source_artifacts_sha256"],
            "must_separate_source_pairs": request["must_separate_source_pairs"],
            "estimated_cost_usd": request.get(
                "estimated_total_cost_usd", request["estimated_cost_usd"]
            ),
            "observed_cost_usd": combined_observed_cost,
        }
        if strategy == STRATEGY_VERIFIED_PARAPHRASE_C:
            summary.update(
                {
                    "generator_raw_response_sha256": raw_hash,
                    "generator_parsed_response_sha256": parsed_hash,
                    "verifier_request_sha256": verifier_request_hash,
                    "verifier_raw_response_sha256": verifier_raw_hash,
                    "verifier_parsed_response_sha256": verifier_parsed_hash,
                    "verifier_answer_count": verifier_result["answer_count"]
                    if verifier_result is not None
                    else 0,
                }
            )
        summary_hash = _write_json_exclusive(root / "summary.json", summary)
        _write_json_exclusive(
            root / "terminal.json",
            {
                "schema_version": 1,
                "status": final_status,
                "strategy_id": strategy,
                "summary_sha256": summary_hash,
                "source_artifacts_sha256": request["source_artifacts_sha256"],
            },
        )
        return summary
    except Exception as error:  # noqa: BLE001 - freeze terminal failure evidence
        if not (root / "terminal.json").exists():
            _write_json_exclusive(
                root / "terminal.json",
                {
                    "schema_version": 1,
                    "status": "failed_terminal",
                    "strategy_id": strategy,
                    "source_artifacts_sha256": request["source_artifacts_sha256"],
                    "must_separate_source_pairs": request["must_separate_source_pairs"],
                    "error_type": type(error).__name__,
                },
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_grouped_rewrite_canary")
    parser.add_argument("--run", action="store_true", help="explicitly issue the one provider call")
    parser.add_argument("--strategy", choices=STRATEGIES, default=STRATEGY_SEMANTIC_REWRITE_A)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--questions", type=Path, default=DEFAULT_TESTING_QUESTIONS)
    parser.add_argument("--gemini-overlay", type=Path, default=DEFAULT_GEMINI_OVERLAY)
    parser.add_argument("--qwen-aggregate", type=Path, default=DEFAULT_QWEN_AGGREGATE)
    parser.add_argument("--qwen35-control", type=Path, default=DEFAULT_QWEN35_CONTROL)
    parser.add_argument("--fusion-submission", type=Path, default=DEFAULT_FUSION_SUBMISSION)
    parser.add_argument("--fusion-provenance", type=Path, default=DEFAULT_FUSION_PROVENANCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        client = None
        if args.run:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not configured")
            client = genai.Client(api_key=api_key)
        result = run_canary(
            args.questions,
            args.gemini_overlay,
            args.qwen_aggregate,
            args.qwen35_control,
            args.fusion_submission,
            args.fusion_provenance,
            args.output_root,
            client=client,
            strategy=args.strategy,
            run=args.run,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"status": "error", "error": "command failed"}, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
