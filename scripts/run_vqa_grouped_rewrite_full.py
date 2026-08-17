#!/usr/bin/env python3
"""Resumable full-panel grouped-rewrite generator.

The default command only freezes source hashes, batches, requests, and cost
estimates.  Provider calls require ``--run``.  Every batch has an immutable
attempt directory; a completed batch is never retried or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_QUERY_COUNT = 2_443
BATCH_SIZE = 10
MAX_BATCH_CONCURRENCY = 4
ANSWERS_PER_QUERY = 10
MODEL = "gemini-3.7-flash"
SEED = 42
GENERATOR_MAX_OUTPUT_TOKENS = 16_384
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75
GLOBAL_MAX_COST_USD = 25.0
STRATEGY = "grouped_unique_generator_only"

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = _ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
DEFAULT_GEMINI = _ROOT / (
    "data/vqa/benchmarks/deadline-submission-rescue-20260817/"
    "gemini-27-mp4/overlay/predictions-freeze.json"
)
DEFAULT_QWEN = _ROOT / (
    "data/vqa/benchmarks/deadline-submission-full-20260816/"
    "qwen36/aggregate/predictions-freeze.json"
)
DEFAULT_CONTROL = _ROOT / "data/vqa/submissions/deadline-20260817-ready/03-qwen35-baseline-control-top10.json"
DEFAULT_FUSION = _ROOT / "data/vqa/submissions/deadline-20260817-ready/05-gemini37-qwen36-fusion-top10.json"
DEFAULT_PROVENANCE = _ROOT / "data/vqa/submissions/deadline-20260817-ready/05-gemini37-qwen36-fusion-top10.provenance.json"
DEFAULT_OUTPUT_ROOT = _ROOT / "data/vqa/benchmarks/grouped-rewrite-full-20260817-grouped_unique_generator_only"

SUCCESS = frozenset({"succeeded", "success", "passed", "complete"})
SOURCE_IDS = (
    "gemini_overlay",
    "qwen_video_plus_transcript",
    "qwen_native_video",
    "qwen35_control",
)
MUST_SEPARATE_SOURCE_PAIRS = {
    1563: ("qwen_video_plus_transcript", "qwen_native_video"),
    2404: ("qwen_video_plus_transcript", "qwen_native_video"),
    2421: ("gemini_overlay", "qwen35_control"),
}

GENERATOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query_id": {"type": "integer"},
                    "groups": {"type": "array"},
                    "answers": {"type": "array", "minItems": 10, "maxItems": 10},
                },
                "required": ["query_id", "groups", "answers"],
            },
        }
    },
    "required": ["queries"],
}
def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc


def _rows(value: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result = value
    elif isinstance(value, dict) and isinstance(value.get("rows"), list):
        result = value["rows"]
    elif isinstance(value, dict) and isinstance(value.get("predictions"), list):
        result = value["predictions"]
    else:
        raise ValueError(f"rows are not an object list: {path}")
    if not all(isinstance(row, dict) for row in result):
        raise ValueError(f"rows contain a non-object: {path}")
    return [dict(row) for row in result]


def _qid(row: Mapping[str, Any], path: Path) -> int:
    value = row.get("query_id", row.get("Q_ID"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid query ID in {path}")
    return value


def _video(row: Mapping[str, Any], path: Path) -> str:
    value = row.get("video_id", row.get("Video_ID"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid video ID in {path}")
    return value


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def source_artifacts(
    questions: Path = DEFAULT_QUESTIONS,
    gemini: Path = DEFAULT_GEMINI,
    qwen: Path = DEFAULT_QWEN,
    control: Path = DEFAULT_CONTROL,
    fusion: Path = DEFAULT_FUSION,
    provenance: Path = DEFAULT_PROVENANCE,
) -> list[dict[str, str]]:
    paths = (("questions", questions), ("gemini_overlay", gemini), ("qwen_aggregate", qwen),
             ("qwen35_control", control), ("fusion_submission", fusion), ("fusion_provenance", provenance))
    result = []
    for name, raw in paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise ValueError(f"source artifact is not a file: {path}")
        result.append({"name": name, "path": str(path), "sha256": _sha_file(path)})
    return result


def _by_qid(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for row in rows:
        query_id = _qid(row, path)
        if query_id in result:
            raise ValueError(f"duplicate query ID in {path}: {query_id}")
        result[query_id] = dict(row)
    return result


def _answer(row: Mapping[str, Any], path: Path) -> tuple[str, float]:
    value = row.get("answer", row.get("Answer"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty answer in {path}")
    time_value = row.get("time", row.get("Time", 0.0))
    if isinstance(time_value, bool) or not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)):
        raise ValueError(f"invalid answer time in {path}")
    return value, float(time_value)


def load_rows(
    questions: Path = DEFAULT_QUESTIONS,
    gemini: Path = DEFAULT_GEMINI,
    qwen: Path = DEFAULT_QWEN,
    control: Path = DEFAULT_CONTROL,
    fusion: Path = DEFAULT_FUSION,
    provenance: Path = DEFAULT_PROVENANCE,
    *,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
) -> list[dict[str, Any]]:
    q_rows = _by_qid(_rows(_json(Path(questions)), Path(questions)), Path(questions))
    g_rows = _by_qid(_rows(_json(Path(gemini)), Path(gemini)), Path(gemini))
    c_rows = _by_qid(_rows(_json(Path(control)), Path(control)), Path(control))
    f_rows = _by_qid(_rows(_json(Path(fusion)), Path(fusion)), Path(fusion))
    p_rows = _by_qid(_rows(_json(Path(provenance)), Path(provenance)), Path(provenance))
    q_by_condition: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _rows(_json(Path(qwen)), Path(qwen)):
        query_id = _qid(row, Path(qwen))
        condition = row.get("condition_id")
        if condition in {"video-plus-transcript", "native-video"} and row.get("status") in SUCCESS:
            source = "qwen_video_plus_transcript" if condition == "video-plus-transcript" else "qwen_native_video"
            q_by_condition.setdefault((query_id, source), dict(row))
    query_ids = sorted(q_rows)
    if len(query_ids) != expected_query_count:
        raise ValueError(f"expected {expected_query_count} questions, found {len(query_ids)}")
    output = []
    for query_id in query_ids:
        question_row, fusion_row, provenance_row = q_rows[query_id], f_rows.get(query_id), p_rows.get(query_id)
        control_row = c_rows.get(query_id)
        if not fusion_row or not provenance_row or not control_row:
            raise ValueError(f"released panel missing query {query_id}")
        candidates = []
        overlay = g_rows.get(query_id)
        if overlay and overlay.get("status") in SUCCESS:
            text, timestamp = _answer(overlay, Path(gemini))
            candidates.append({"source_id": "gemini_overlay", "original_rank": 1, "answer": text, "time": timestamp})
        for source in ("qwen_video_plus_transcript", "qwen_native_video"):
            row = q_by_condition.get((query_id, source))
            if row:
                text, timestamp = _answer(row, Path(qwen))
                candidates.append({"source_id": source, "original_rank": 1, "answer": text, "time": timestamp})
        control_answers = control_row.get("Answers")
        if not isinstance(control_answers, list):
            raise ValueError(f"control answers missing for query {query_id}")
        control_rank1 = next((item for item in control_answers if item.get("Rank") == 1), None)
        if not isinstance(control_rank1, Mapping):
            raise ValueError(f"control rank 1 missing for query {query_id}")
        text, timestamp = _answer(control_rank1, Path(control))
        candidates.append({"source_id": "qwen35_control", "original_rank": 1, "answer": text, "time": timestamp})
        fusion_answers = fusion_row.get("Answers")
        if not isinstance(fusion_answers, list) or len(fusion_answers) != ANSWERS_PER_QUERY:
            raise ValueError(f"fusion must contain ten answers for query {query_id}")
        rank1 = next((item for item in fusion_answers if item.get("Rank") == 1), None)
        selected_sources = provenance_row.get("selected_sources")
        if not isinstance(rank1, Mapping) or not isinstance(selected_sources, list) or not selected_sources:
            raise ValueError(f"fusion rank one provenance missing for query {query_id}")
        rank1_answer, _ = _answer(rank1, Path(fusion))
        source_name = {"gemini": "gemini_overlay", "baseline": "qwen35_control"}.get(selected_sources[0], selected_sources[0])
        if source_name not in {item["source_id"] for item in candidates}:
            raise ValueError(f"fusion rank-one source unavailable for query {query_id}")
        source_candidate = next(item for item in candidates if item["source_id"] == source_name)
        if source_candidate["answer"] != rank1_answer:
            raise ValueError(f"fusion rank-one answer mismatch for query {query_id}")
        output.append({
            "query_id": query_id,
            "video_id": _video(question_row, Path(questions)),
            "question": str(question_row.get("question", "")).strip(),
            "rank1_source": source_name,
            "rank1_answer": rank1_answer,
            "source_candidates": candidates,
            "fallback": {"Q_ID": query_id, "Video_ID": _video(fusion_row, Path(fusion)), "Answers": fusion_answers},
        })
    return output


def batches(rows: Sequence[Mapping[str, Any]], batch_size: int = BATCH_SIZE) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [list(map(dict, rows[index:index + batch_size])) for index in range(0, len(rows), batch_size)]


def _estimate(request: Mapping[str, Any], output_tokens: int) -> tuple[int, int, float]:
    # A wire byte is conservatively treated as one input token.  This is an
    # intentional upper bound for dispatch reservations; provider usage can
    # be lower, but cannot make the reservation understate the request size.
    prompt = max(1, len(_canonical(request)))
    cost = prompt / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    return prompt, output_tokens, round(cost, 8)


def build_generator_request(batch: Sequence[Mapping[str, Any]], batch_id: int) -> dict[str, Any]:
    queries = []
    for row in batch:
        queries.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "rank1_source": row["rank1_source"],
            "rank1_answer": row["rank1_answer"],
            "source_candidates": [
                {"source_id": item["source_id"], "original_rank": 1, "answer": item["answer"]}
                for item in row["source_candidates"]
            ],
            "requested_answer_count": ANSWERS_PER_QUERY,
        })
    request = {
        "strategy_id": STRATEGY,
        "stage": "generator",
        "batch_id": batch_id,
        "model": MODEL,
        "seed": SEED,
        "max_output_tokens": GENERATOR_MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_json_schema": GENERATOR_SCHEMA,
        "instructions": "Group every source exactly once, preserve conflicts, preserve rank 1 byte-for-byte, and produce ten distinct grounded answers per query. Every group needs an exact source answer; do not add facts.",
        "queries": queries,
    }
    prompt, output, cost = _estimate(request, GENERATOR_MAX_OUTPUT_TOKENS)
    request.update(estimated_prompt_tokens=prompt, estimated_output_tokens=output, estimated_cost_usd=cost)
    return request


def validate_generator(payload: Mapping[str, Any], batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != len(batch):
        raise ValueError("generator query count is invalid")
    expected = {int(row["query_id"]): row for row in batch}
    seen = set()
    normalized_queries = []
    for query in queries:
        if not isinstance(query, Mapping) or set(query) != {"query_id", "groups", "answers"}:
            raise ValueError("generator query schema is invalid")
        query_id = query.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id not in expected or query_id in seen:
            raise ValueError("generator query identity is invalid")
        seen.add(query_id)
        row = expected[query_id]
        candidates = {item["source_id"]: item for item in row["source_candidates"]}
        groups = query["groups"]
        if not isinstance(groups, list):
            raise ValueError("generator groups are required")
        group_by_id = {}
        covered = []
        for group in groups:
            if not isinstance(group, Mapping) or set(group) != {"group_id", "source_ids"}:
                raise ValueError("generator group schema is invalid")
            gid, source_ids = group.get("group_id"), group.get("source_ids")
            if not isinstance(gid, str) or gid in group_by_id or not isinstance(source_ids, list) or not source_ids:
                raise ValueError("generator group identity is invalid")
            if any(not isinstance(source_id, str) for source_id in source_ids) or len(source_ids) != len(set(source_ids)) or any(source_id not in candidates or source_id in covered for source_id in source_ids):
                raise ValueError("generator source coverage is invalid")
            covered.extend(source_ids)
            group_by_id[gid] = dict(group)
        if len(covered) != len(candidates) or set(covered) != set(candidates):
            raise ValueError("generator source coverage is incomplete")
        pair = MUST_SEPARATE_SOURCE_PAIRS.get(query_id)
        if pair and pair[0] in candidates and pair[1] in candidates:
            left_group = next(gid for gid, group in group_by_id.items() if pair[0] in group["source_ids"])
            right_group = next(gid for gid, group in group_by_id.items() if pair[1] in group["source_ids"])
            if left_group == right_group:
                raise ValueError("predeclared conflict pair was merged")
        answers = query["answers"]
        if not isinstance(answers, list) or len(answers) != ANSWERS_PER_QUERY:
            raise ValueError("generator must return ten answers")
        ranks = []
        distinct = set()
        represented = set()
        for answer in answers:
            if not isinstance(answer, Mapping) or set(answer) != {"rank", "answer", "source_group"}:
                raise ValueError("generator answer schema is invalid")
            if isinstance(answer["rank"], bool) or not isinstance(answer["rank"], int) or not isinstance(answer["answer"], str) or not answer["answer"].strip() or answer["source_group"] not in group_by_id:
                raise ValueError("generator answer fields are invalid")
            normalized = " ".join(answer["answer"].split()).casefold()
            if normalized in distinct:
                raise ValueError("generator answers are not distinct")
            distinct.add(normalized)
            ranks.append(answer["rank"])
            represented.add(answer["source_group"])
        if ranks != list(range(1, ANSWERS_PER_QUERY + 1)) or represented != set(group_by_id):
            raise ValueError("generator rank or group representation is invalid")
        if answers[0]["answer"] != row["rank1_answer"]:
            raise ValueError("generator rank one changed")
        rank1_group = next(gid for gid, group in group_by_id.items() if row["rank1_source"] in group["source_ids"])
        if answers[0]["source_group"] != rank1_group:
            raise ValueError("generator rank one group changed")
        for gid, group in group_by_id.items():
            exact = {candidates[source_id]["answer"] for source_id in group["source_ids"]}
            if not any(answer["source_group"] == gid and answer["answer"] in exact for answer in answers):
                raise ValueError("generator group lacks exact source answer")
        normalized_queries.append({"query_id": query_id, "groups": [dict(g) for g in groups], "answers": [dict(a) for a in answers]})
    if seen != set(expected):
        raise ValueError("generator panel coverage is incomplete")
    return {"status": "validated", "query_count": len(batch), "answer_count": len(batch) * ANSWERS_PER_QUERY, "queries": normalized_queries}


def _usage(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt, output = usage.get("prompt_tokens"), usage.get("output_tokens")
    if not isinstance(prompt, int) or not isinstance(output, int):
        return None
    return prompt, output


def _cost(payload: Mapping[str, Any]) -> float:
    usage = _usage(payload)
    if usage is None:
        raise ValueError("provider usage metadata is required")
    prompt, output = usage
    return round(prompt / 1_000_000 * INPUT_USD_PER_MILLION + output / 1_000_000 * OUTPUT_USD_PER_MILLION, 8)


def _normalized_usage(raw: Any) -> dict[str, int]:
    if isinstance(raw, Mapping):
        usage = raw.get("usage") or raw.get("usage_metadata")
        if isinstance(usage, Mapping):
            prompt = usage.get("prompt_tokens", usage.get("prompt_token_count"))
            output = usage.get("output_tokens", usage.get("response_token_count", usage.get("candidates_token_count")))
            total = usage.get("total_tokens", usage.get("total_token_count"))
        else:
            return {}
    else:
        usage = getattr(raw, "usage_metadata", None)
        prompt = getattr(usage, "prompt_token_count", None) if usage is not None else None
        output = getattr(usage, "response_token_count", None) if usage is not None else None
        if output is None and usage is not None:
            output = getattr(usage, "candidates_token_count", None)
        total = getattr(usage, "total_token_count", None) if usage is not None else None
    if not isinstance(prompt, int) or not isinstance(output, int):
        return {}
    if isinstance(total, int):
        output = max(output, total - prompt)
    result = {"prompt_tokens": prompt, "output_tokens": output}
    if isinstance(total, int):
        result["total_tokens"] = total
    return result


def _raw_cost(raw: Any) -> float | None:
    usage = _normalized_usage(raw)
    if not usage:
        return None
    return round(
        usage["prompt_tokens"] / 1_000_000 * INPUT_USD_PER_MILLION
        + usage["output_tokens"] / 1_000_000 * OUTPUT_USD_PER_MILLION,
        8,
    )


def _batch_reservation(batch: Sequence[Mapping[str, Any]], batch_id: int) -> float:
    # The generator request is the only live provider call.  Its wire-byte
    # upper bound plus the configured output ceiling is the full reservation.
    return build_generator_request(batch, batch_id)["estimated_cost_usd"]


def _write(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return _sha_bytes(data)


def _safe_raw(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return json.loads(json.dumps(dict(raw), default=str))
    if hasattr(raw, "model_dump"):
        dumped = raw.model_dump(mode="json", by_alias=True, exclude_none=True)
        return {
            key: value
            for key, value in dumped.items()
            if key in {"candidates", "modelVersion", "responseId", "promptFeedback", "usageMetadata"}
        }
    return {"text": str(getattr(raw, "text", ""))}


def _call(client: Any, request: Mapping[str, Any]) -> Any:
    from google.genai import types
    config = types.GenerateContentConfig(max_output_tokens=request["max_output_tokens"], response_mime_type="application/json", response_json_schema=request["response_json_schema"], seed=SEED)
    body = {"instructions": request["instructions"], "queries": request["queries"]}
    return client.models.generate_content(model=MODEL, contents=json.dumps(body, ensure_ascii=False), config=config)


def _text_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        text = raw.get("text")
        usage = raw.get("usage") or raw.get("usage_metadata")
    else:
        text = getattr(raw, "text", None)
        usage = getattr(raw, "usage_metadata", None)
    if not isinstance(text, str):
        raise ValueError("provider response has no JSON text")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("provider response must be an object")
    if usage is not None:
        if isinstance(usage, Mapping):
            normalized = dict(usage)
            if "response_tokens" in normalized and "output_tokens" not in normalized:
                normalized["output_tokens"] = normalized["response_tokens"]
            if "candidates_token_count" in normalized and "output_tokens" not in normalized:
                normalized["output_tokens"] = normalized["candidates_token_count"]
        else:
            prompt = getattr(usage, "prompt_token_count", None)
            output = getattr(usage, "response_token_count", None)
            if output is None:
                output = getattr(usage, "candidates_token_count", None)
            total = getattr(usage, "total_token_count", None)
            if isinstance(total, int) and isinstance(output, int) and isinstance(prompt, int):
                output = max(output, total - prompt)
            normalized = {"prompt_tokens": prompt, "output_tokens": output}
            if total is not None:
                normalized["total_tokens"] = total
        if isinstance(normalized.get("prompt_tokens"), int) and isinstance(normalized.get("output_tokens"), int):
            payload["usage"] = normalized
    return payload


def _organize(rows: Sequence[Mapping[str, Any]], generated: Mapping[int, Mapping[str, Any]], elapsed: Mapping[int, float]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        query_id = int(row["query_id"])
        fallback = row["fallback"]
        result = dict(fallback)
        if query_id not in generated:
            output.append(result)
            continue
        answers = []
        elapsed_seconds = float(elapsed[query_id])
        for index, answer in enumerate(generated[query_id]["answers"], start=1):
            answers.append({"Rank": index, "Answer": answer["answer"], "Time": round(elapsed_seconds, 6)})
        result["Answers"] = answers
        output.append(result)
    return output


def _build_provenance(rows: Sequence[Mapping[str, Any]], organized: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Q_ID": int(row["query_id"]),
            "Video_ID": row["video_id"],
            "status": "fallback_fusion" if organized[index]["Answers"] == row["fallback"]["Answers"] else "generated",
            "answer_sha256": [_sha_bytes(str(answer["Answer"]).encode()) for answer in organized[index]["Answers"]],
        }
        for index, row in enumerate(rows)
    ]


def _accounted_markers(root: Path, batch_count: int, reservations: Sequence[float]) -> tuple[float, int]:
    accounted = 0.0
    accepted = 0
    for index in range(batch_count):
        batch_root = root / f"batch-{index + 1:04d}"
        marker = next(
            (batch_root / name for name in ("accepted.json", "rejected.json", "failed.json") if (batch_root / name).exists()),
            None,
        )
        if marker is None:
            continue
        marker_payload = _json(marker)
        accounted += float(marker_payload.get("accounted_cost_usd", reservations[index]))
        accepted += int(marker.name == "accepted.json")
    return round(accounted, 8), accepted


def _recover_final(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    batch_list: Sequence[Sequence[Mapping[str, Any]]],
    reservations: Sequence[float],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    write_summary: bool,
) -> dict[str, Any] | None:
    final_path = root / "final-organizer.json"
    summary_path = root / "summary.json"
    provenance_path = root / "final-provenance.json"
    if not final_path.exists():
        if summary_path.exists():
            raise ValueError("summary exists without final organizer")
        return None
    from trec_video_2026.vqa_deadline_canary import validate_ranked_submission

    final_rows = _json(final_path)
    validate_ranked_submission(final_rows, expected_query_ids={int(row["query_id"]) for row in rows})
    final_hash = _sha_file(final_path)
    if summary_path.exists():
        summary = _json(summary_path)
        if summary.get("status") != "complete":
            raise ValueError("existing final summary is not complete")
        if summary.get("final_sha256") != final_hash:
            raise ValueError("summary hash mismatch for final organizer")
        if not provenance_path.exists() or summary.get("provenance_sha256") != _sha_file(provenance_path):
            raise ValueError("summary hash mismatch for final provenance")
        if summary.get("source_artifacts_sha256") != _sha_bytes(_canonical(artifacts)):
            raise ValueError("summary source hash mismatch")
        return summary
    if not write_summary:
        return None
    if provenance_path.exists():
        provenance = _json(provenance_path)
    else:
        provenance = _build_provenance(rows, final_rows)
        _write(provenance_path, provenance)
    provenance_hash = _sha_file(provenance_path)
    accounted, accepted_batch_count = _accounted_markers(root, len(batch_list), reservations)
    summary = {
        "status": "complete",
        "query_count": len(rows),
        "batch_count": len(batch_list),
        "accepted_batch_count": accepted_batch_count,
        "fallback_query_count": sum(item.get("status") == "fallback_fusion" for item in provenance),
        "accounted_cost_usd": accounted,
        "final_sha256": final_hash,
        "provenance_sha256": provenance_hash,
        "source_artifacts": list(artifacts),
        "source_artifacts_sha256": _sha_bytes(_canonical(artifacts)),
        "recovered_finalization": True,
    }
    _write(summary_path, summary)
    return summary


def run_full(
    *,
    questions: Path = DEFAULT_QUESTIONS,
    gemini: Path = DEFAULT_GEMINI,
    qwen: Path = DEFAULT_QWEN,
    control: Path = DEFAULT_CONTROL,
    fusion: Path = DEFAULT_FUSION,
    provenance: Path = DEFAULT_PROVENANCE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run: bool = False,
    client: Any | None = None,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
) -> dict[str, Any]:
    artifacts = source_artifacts(questions, gemini, qwen, control, fusion, provenance)
    rows = load_rows(questions, gemini, qwen, control, fusion, provenance, expected_query_count=expected_query_count)
    batch_list = batches(rows)
    requests = [build_generator_request(batch, index + 1) for index, batch in enumerate(batch_list)]
    reservations = [_batch_reservation(batch, index + 1) for index, batch in enumerate(batch_list)]
    estimated = sum(reservations)
    if estimated > GLOBAL_MAX_COST_USD:
        raise ValueError("estimated full-panel cost exceeds global cap")
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "strategy_id": STRATEGY,
        "query_count": len(rows),
        "batch_count": len(batch_list),
        "batch_size": BATCH_SIZE,
        "model": MODEL,
        "seed": SEED,
        "generator_max_output_tokens": GENERATOR_MAX_OUTPUT_TOKENS,
        "max_concurrency": MAX_BATCH_CONCURRENCY,
        "global_max_cost_usd": GLOBAL_MAX_COST_USD,
        "source_artifacts": artifacts,
        "source_artifacts_sha256": _sha_bytes(_canonical(artifacts)),
    }
    existing_manifest = Path(output_root) / "manifest.json"
    if existing_manifest.exists() and _json(existing_manifest) != manifest:
        raise ValueError("full-run manifest/config/source hash drift")
    recovered = _recover_final(
        Path(output_root), rows, batch_list, reservations, artifacts, write_summary=run
    )
    if recovered is not None:
        return recovered
    if not run:
        return {"status": "dry_run", "query_count": len(rows), "batch_count": len(batch_list), "batch_size": BATCH_SIZE, "request_count": len(batch_list), "estimated_cost_usd": round(estimated, 8), "max_concurrency": MAX_BATCH_CONCURRENCY, "source_artifacts": artifacts, "source_artifacts_sha256": _sha_bytes(_canonical(artifacts)), "output_root_created": False}
    if client is None:
        raise ValueError("client is required only with --run")
    from trec_video_2026.vqa_deadline_canary import validate_ranked_submission

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        if _json(root / "manifest.json") != manifest:
            raise ValueError("full-run manifest/config/source hash drift")
    else:
        _write(root / "manifest.json", manifest)
    completed: dict[int, Mapping[str, Any]] = {}
    elapsed: dict[int, float] = {}
    terminal_present: list[bool] = []
    prior_accounted = 0.0
    for index in range(len(batch_list)):
        batch_root = root / f"batch-{index + 1:04d}"
        marker = next(
            (batch_root / name for name in ("accepted.json", "rejected.json", "failed.json") if (batch_root / name).exists()),
            None,
        )
        terminal_present.append(marker is not None)
        if marker is not None:
            marker_payload = _json(marker)
            prior_accounted += float(marker_payload.get("accounted_cost_usd", reservations[index]))
    if prior_accounted > GLOBAL_MAX_COST_USD:
        raise ValueError("prior accounted cost exceeds global cap")
    # Live execution is intentionally explicit and resumable.  Each worker owns
    # one batch directory, so no batch can be called twice by this process.
    def process(index: int) -> tuple[int, Mapping[str, Any] | None, float, float]:
        batch_id = index + 1
        batch = batch_list[index]
        batch_root = root / f"batch-{batch_id:04d}"
        if (batch_root / "accepted.json").exists():
            accepted = _json(batch_root / "accepted.json")
            return batch_id, accepted, float(accepted.get("elapsed_seconds", 0.0)), 0.0
        if (batch_root / "rejected.json").exists():
            rejected = _json(batch_root / "rejected.json")
            return batch_id, None, float(rejected.get("elapsed_seconds", 0.0)), 0.0
        if (batch_root / "failed.json").exists():
            failed = _json(batch_root / "failed.json")
            return batch_id, None, float(failed.get("elapsed_seconds", 0.0)), 0.0
        attempt = batch_root / "attempt-0001"
        if attempt.exists():
            _write(
                batch_root / "failed.json",
                {
                    "status": "ambiguous_attempt",
                    "error_type": "InterruptedAttempt",
                    "elapsed_seconds": 0.0,
                    "accounted_cost_usd": reservations[index],
                },
            )
            return batch_id, None, 0.0, reservations[index]
        attempt.mkdir(parents=True, exist_ok=False)
        request = requests[index]
        request["source_artifacts"] = artifacts
        request["source_artifacts_sha256"] = _sha_bytes(_canonical(artifacts))
        _write(attempt / "request.json", request)
        start = time.monotonic()
        raw = _call(client, request)
        _write(attempt / "raw-response.json", {"provider_response": _safe_raw(raw)})
        _write(attempt / "usage.json", {"usage": _normalized_usage(raw)})
        generator_payload = _text_payload(raw)
        generator_result = validate_generator(generator_payload, batch)
        _write(attempt / "parsed-response.json", generator_result)
        elapsed_seconds = time.monotonic() - start
        accounted_cost = _cost(generator_payload)
        accepted = {"status": "accepted", "elapsed_seconds": elapsed_seconds, "accounted_cost_usd": accounted_cost, "queries": generator_result["queries"], "attempt": str(attempt)}
        _write(batch_root / "accepted.json", accepted)
        return batch_id, accepted, elapsed_seconds, accounted_cost
    total_accounted = prior_accounted
    for wave_start in range(0, len(batch_list), MAX_BATCH_CONCURRENCY):
        wave_indices = list(range(wave_start, min(wave_start + MAX_BATCH_CONCURRENCY, len(batch_list))))
        if total_accounted + sum(reservations[index] for index in wave_indices if not terminal_present[index]) > GLOBAL_MAX_COST_USD:
            for index in wave_indices:
                batch_root = root / f"batch-{index + 1:04d}"
                if not any((batch_root / marker).exists() for marker in ("accepted.json", "rejected.json", "failed.json")):
                    _write(batch_root / "failed.json", {"status": "budget_gate", "elapsed_seconds": 0.0, "accounted_cost_usd": 0.0})
            continue
        with ThreadPoolExecutor(max_workers=MAX_BATCH_CONCURRENCY) as pool:
            future_indices = {pool.submit(process, index): index for index in wave_indices}
            for future in as_completed(future_indices):
                index = future_indices[future]
                try:
                    batch_id, accepted, elapsed_seconds, accounted = future.result()
                except Exception as error:  # preserve fallback, evidence, and reservation
                    batch_id = index + 1
                    batch_root = root / f"batch-{batch_id:04d}"
                    attempt = batch_root / "attempt-0001"
                    if attempt.exists() and not (attempt / "error.json").exists():
                        _write(attempt / "error.json", {"error_type": type(error).__name__, "message": str(error)})
                    _write(
                        batch_root / "failed.json",
                        {"status": "failed", "error_type": type(error).__name__, "elapsed_seconds": 0.0, "accounted_cost_usd": reservations[index]},
                    )
                    elapsed_seconds, accepted, accounted = 0.0, None, reservations[index]
                total_accounted += accounted
                if total_accounted > GLOBAL_MAX_COST_USD:
                    raise ValueError("accounted full-panel cost exceeds global cap")
                elapsed.update({int(row["query_id"]): elapsed_seconds for row in batch_list[batch_id - 1]})
                if accepted:
                    for query in accepted["queries"]:
                        completed[int(query["query_id"])] = query
    organized = _organize(rows, completed, elapsed)
    validate_ranked_submission(organized, expected_query_ids={int(row["query_id"]) for row in rows})
    final_hash = _write(root / "final-organizer.json", organized)
    provenance = _build_provenance(rows, organized)
    provenance_hash = _write(root / "final-provenance.json", provenance)
    accepted_batch_count = sum(
        all(int(row["query_id"]) in completed for row in batch) for batch in batch_list
    )
    summary = {"status": "complete", "query_count": len(rows), "batch_count": len(batch_list), "accepted_batch_count": accepted_batch_count, "fallback_query_count": len(rows) - len(completed), "accounted_cost_usd": round(total_accounted, 8), "final_sha256": final_hash, "provenance_sha256": provenance_hash, "source_artifacts": artifacts, "source_artifacts_sha256": _sha_bytes(_canonical(artifacts))}
    _write(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_vqa_grouped_rewrite_full")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--gemini", type=Path, default=DEFAULT_GEMINI)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args(argv)
    client = None
    if args.run:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=key)
    print(json.dumps(run_full(questions=args.questions, gemini=args.gemini, qwen=args.qwen, control=args.control, fusion=args.fusion, provenance=args.provenance, output_root=args.output_root, run=args.run, client=client), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
