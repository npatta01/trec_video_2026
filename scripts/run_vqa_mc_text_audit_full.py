#!/usr/bin/env python3
"""Blindly audit whether the frozen Task 1 answer is represented by MC options.

The auditor never receives the existing MC ranking. It emits private sidecars
for review while leaving the official submission byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_vqa_mc_text_rank1_full as baseline


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = "fusion_rank1_blind_mc_option_support_audit"
MATCH_QUALITIES = ("exact", "paraphrase", "partial", "none")
ISSUE_TYPES = (
    "numeric_answer_absent",
    "polarity_absent",
    "entity_absent",
    "answer_more_specific_than_options",
    "answer_less_specific_than_options",
    "malformed_options",
    "ambiguous_best_option",
    "semantic_mismatch",
)
QUALITY_CODE_TO_LABEL = {index: label for index, label in enumerate(MATCH_QUALITIES, start=1)}
ISSUE_CODE_TO_LABEL = {index: label for index, label in enumerate(ISSUE_TYPES, start=1)}
DEFAULT_SUBMISSION = (
    ROOT / "data/vqa/benchmarks/vqa-mc-text-rank1-20260822-a1/submission.json"
)


def load_audit_rows(
    task2: Path | bytes,
    fusion: Path,
    submission: Path,
    *,
    expected_query_count: int | None = baseline.EXPECTED_QUERY_COUNT,
    query_limit: int | None = None,
) -> list[dict[str, Any]]:
    rows = baseline.load_rows(
        task2,
        fusion,
        expected_query_count=expected_query_count,
        query_limit=query_limit,
    )
    raw_submission = baseline._json(Path(submission))
    if not isinstance(raw_submission, list) or not all(
        isinstance(row, Mapping) for row in raw_submission
    ):
        raise ValueError("submission rows are invalid")
    if query_limit is not None:
        raw_submission = raw_submission[:query_limit]
    if len(raw_submission) != len(rows):
        raise ValueError("submission and audit row counts differ")

    submission_by_id: dict[int, Mapping[str, Any]] = {}
    for row in raw_submission:
        query_id = row.get("Q_ID")
        if (
            isinstance(query_id, bool)
            or not isinstance(query_id, int)
            or query_id in submission_by_id
        ):
            raise ValueError("submission Q_ID coverage is invalid")
        submission_by_id[query_id] = row

    output: list[dict[str, Any]] = []
    for row in rows:
        query_id = row["query_id"]
        submitted = submission_by_id.get(query_id)
        if not isinstance(submitted, Mapping) or submitted.get("Video_ID") != row["video_id"]:
            raise ValueError(f"submission identity mismatch for Q_ID {query_id}")
        answers = submitted.get("Answers")
        if not isinstance(answers, list) or len(answers) != len(row["options"]):
            raise ValueError(f"submission answers are invalid for Q_ID {query_id}")
        ordered = sorted(
            answers,
            key=lambda answer: answer.get("Rank") if isinstance(answer, Mapping) else -1,
        )
        if [answer.get("Rank") for answer in ordered if isinstance(answer, Mapping)] != list(
            range(1, len(row["options"]) + 1)
        ):
            raise ValueError(f"submission ranks are invalid for Q_ID {query_id}")
        ranked_text = [answer.get("Answer") for answer in ordered]
        if sorted(ranked_text) != sorted(row["options"]):
            raise ValueError(f"submission options differ for Q_ID {query_id}")
        current_top = ranked_text[0]
        current_top_option_id = row["options"].index(current_top) + 1
        output.append(
            {
                **row,
                "current_top_answer": current_top,
                "current_top_option_id": current_top_option_id,
            }
        )
    if set(submission_by_id) != {row["query_id"] for row in rows}:
        raise ValueError("submission Q_ID coverage differs from audit rows")
    return output


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": baseline.BATCH_SIZE,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query_id": {"type": "integer"},
                        "match_quality_code": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": len(QUALITY_CODE_TO_LABEL),
                        },
                        "best_option_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 11,
                            "uniqueItems": True,
                            "items": {"type": "integer"},
                        },
                        "issue_codes": {
                            "type": "array",
                            "maxItems": len(ISSUE_TYPES),
                            "uniqueItems": True,
                            "items": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": len(ISSUE_CODE_TO_LABEL),
                            },
                        },
                        "requires_video_review": {"type": "boolean"},
                    },
                    "required": [
                        "query_id",
                        "match_quality_code",
                        "best_option_ids",
                        "issue_codes",
                        "requires_video_review",
                    ],
                },
            }
        },
        "required": ["queries"],
    }


def build_request(batch: Sequence[Mapping[str, Any]], batch_id: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("audit batch is empty")
    queries = [
        {
            "query_id": int(row["query_id"]),
            "question": row["question"],
            "prior_video_answer": row["rank1_answer"],
            "options": [
                {"option_id": index, "option": option}
                for index, option in enumerate(row["options"], start=1)
            ],
        }
        for row in batch
    ]
    return {
        "model": baseline.MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Independently audit whether a prior answer generated from a video is represented "
                    "by the supplied multiple-choice options. You are not selecting a new video answer. "
                    "Use exact when an option preserves the answer without meaningful loss; paraphrase "
                    "for semantic equivalence; partial when only an incomplete, overbroad, or overspecific "
                    "option is available; and none when every option contradicts or fails to express the "
                    "prior answer. Return the closest best_option_ids even for partial or none. Mark partial "
                    "and none as requiring video review. Encode match quality as 1=exact, 2=paraphrase, "
                    "3=partial, 4=none. Encode issues as 1=numeric answer absent, 2=polarity absent, "
                    "3=entity absent, 4=answer more specific than options, 5=answer less specific than "
                    "options, 6=malformed options, 7=ambiguous best option, 8=semantic mismatch. Treat "
                    "numbers, polarity, entities, temporal scope, and malformed fragments carefully."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"strategy": STRATEGY, "batch_id": batch_id, "queries": queries},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vqa_mc_option_support_audit",
                "strict": True,
                "schema": _response_schema(),
            },
        },
        "reasoning": {"effort": "high", "exclude": True},
        "max_tokens": baseline.MAX_OUTPUT_TOKENS,
        "seed": baseline.SEED,
        "temperature": 0,
        "usage": {"include": True},
    }


def validate_response(
    payload: Mapping[str, Any], batch: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != len(batch):
        raise ValueError("audit response query count is invalid")
    expected = {int(row["query_id"]): row for row in batch}
    required = {
        "query_id",
        "match_quality_code",
        "best_option_ids",
        "issue_codes",
        "requires_video_review",
    }
    output: dict[int, dict[str, Any]] = {}
    for query in queries:
        if not isinstance(query, Mapping) or set(query) != required:
            raise ValueError("audit response schema is invalid")
        query_id = query.get("query_id")
        if (
            isinstance(query_id, bool)
            or not isinstance(query_id, int)
            or query_id not in expected
            or query_id in output
        ):
            raise ValueError("audit response identity is invalid")
        quality_code = query.get("match_quality_code")
        best_ids = query.get("best_option_ids")
        issue_codes = query.get("issue_codes")
        review = query.get("requires_video_review")
        option_count = len(expected[query_id]["options"])
        if (
            isinstance(quality_code, bool)
            or not isinstance(quality_code, int)
            or quality_code not in QUALITY_CODE_TO_LABEL
        ):
            raise ValueError("audit match quality is invalid")
        if (
            not isinstance(best_ids, list)
            or not best_ids
            or len(best_ids) != len(set(best_ids))
            or any(
                isinstance(option_id, bool)
                or not isinstance(option_id, int)
                or option_id < 1
                or option_id > option_count
                for option_id in best_ids
            )
        ):
            raise ValueError("audit best option IDs are invalid")
        if (
            not isinstance(issue_codes, list)
            or len(issue_codes) != len(set(issue_codes))
            or any(
                isinstance(issue_code, bool)
                or not isinstance(issue_code, int)
                or issue_code not in ISSUE_CODE_TO_LABEL
                for issue_code in issue_codes
            )
        ):
            raise ValueError("audit issue types are invalid")
        if not isinstance(review, bool):
            raise ValueError("audit video review flag is invalid")
        quality = QUALITY_CODE_TO_LABEL[quality_code]
        issues = [ISSUE_CODE_TO_LABEL[issue_code] for issue_code in issue_codes]
        if quality in {"partial", "none"} and not review:
            raise ValueError("partial or no match requires video review")
        if quality == "none" and not issues:
            raise ValueError("no-match audit requires an issue type")
        reason = f"Auditor classified option support as {quality}"
        if issues:
            reason += "; issues: " + ", ".join(issue.replace("_", " ") for issue in issues)
        reason += "."
        output[query_id] = {
            "match_quality": quality,
            "best_option_ids": list(best_ids),
            "issue_types": list(issues),
            "requires_video_review": review,
            "reason": reason.strip(),
        }
    if set(output) != set(expected):
        raise ValueError("audit response coverage is incomplete")
    return output


def organize_audit(
    rows: Sequence[Mapping[str, Any]], assessments: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        query_id = int(row["query_id"])
        assessment = assessments.get(query_id)
        if not isinstance(assessment, Mapping):
            raise ValueError(f"missing audit assessment for Q_ID {query_id}")
        best_ids = list(assessment["best_option_ids"])
        current_id = int(row["current_top_option_id"])
        quality = str(assessment["match_quality"])
        selected_is_best = current_id in best_ids
        selected_supported = selected_is_best and quality in {"exact", "paraphrase"}
        review_required = (
            bool(assessment["requires_video_review"])
            or quality in {"partial", "none"}
            or not selected_is_best
        )
        output.append(
            {
                "Q_ID": query_id,
                "Video_ID": row["video_id"],
                "question": row["question"],
                "prior_answer": row["rank1_answer"],
                "current_top_answer": row["current_top_answer"],
                "current_top_option_id": current_id,
                "match_quality": quality,
                "best_option_ids": best_ids,
                "best_options": [row["options"][option_id - 1] for option_id in best_ids],
                "issue_types": list(assessment["issue_types"]),
                "selected_top_is_best": selected_is_best,
                "selected_top_supported": selected_supported,
                "auditor_requires_video_review": bool(assessment["requires_video_review"]),
                "review_required": review_required,
                "reason": assessment["reason"],
            }
        )
    return output


def review_queue(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in records if record.get("review_required") is True]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        for row in rows
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _jsonl_bytes(rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    data = _jsonl_bytes(rows)
    if not path.exists() and not path.is_symlink():
        return _write_jsonl(path, rows)
    if path.read_bytes() != data:
        raise ValueError(f"existing immutable JSONL artifact differs: {path}")
    return hashlib.sha256(data).hexdigest()


def _write_or_verify_accepted(
    path: Path, assessments: Mapping[int, Mapping[str, Any]], duration: float
) -> str:
    normalized_assessments = {str(query_id): value for query_id, value in assessments.items()}
    value = {"status": "accepted", "elapsed_seconds": duration, "assessments": assessments}
    if not path.exists() and not path.is_symlink():
        return baseline._write(path, value)
    existing = baseline._json(path)
    if (
        not isinstance(existing, Mapping)
        or existing.get("status") != "accepted"
        or baseline._canonical(existing.get("assessments"))
        != baseline._canonical(normalized_assessments)
    ):
        raise ValueError(f"existing accepted audit artifact differs: {path}")
    return baseline._sha_file(path)


def run_full(
    *,
    task2: Path | bytes,
    fusion: Path,
    submission: Path = DEFAULT_SUBMISSION,
    output_root: Path,
    expected_query_count: int | None = baseline.EXPECTED_QUERY_COUNT,
    query_limit: int | None = None,
    max_cost_usd: float = baseline.GLOBAL_MAX_COST_USD,
    run: bool = False,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    baseline._validate_cap(max_cost_usd)
    rows = load_audit_rows(
        task2,
        fusion,
        submission,
        expected_query_count=expected_query_count,
        query_limit=query_limit,
    )
    batch_list = baseline.batches(rows)
    requests = [build_request(batch, index) for index, batch in enumerate(batch_list, start=1)]
    estimates = [baseline._estimate(request) for request in requests]
    estimated_cost = round(sum(item["estimated_cost_usd"] for item in estimates), 8)
    if estimated_cost > max_cost_usd:
        raise ValueError("estimated audit cost exceeds run cap")
    base = {
        "status": "dry_run",
        "strategy": STRATEGY,
        "model": baseline.BATCH_MODEL,
        "dispatched_model": baseline.MODEL,
        "api_mode": "openrouter_async_batch",
        "query_count": len(rows),
        "batch_count": len(batch_list),
        "batch_size": baseline.BATCH_SIZE,
        "max_output_tokens_per_batch": baseline.MAX_OUTPUT_TOKENS,
        "estimated_cost_usd": estimated_cost,
        "max_cost_usd": max_cost_usd,
        "task2_sha256": baseline._source_sha256(task2),
        "fusion_sha256": baseline._sha_file(Path(fusion)),
        "submission_sha256": baseline._sha_file(Path(submission)),
        "request_set_sha256": baseline._sha_value(requests),
        "blind_to_existing_ranking": True,
    }
    if not run:
        if resume:
            raise ValueError("--resume requires --run")
        return base

    root = Path(output_root)
    resume_manifest: Mapping[str, Any] | None = None
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"resume audit root does not exist: {root}")
        loaded_manifest = baseline._json(root / "manifest.json")
        baseline._validate_resume_manifest(loaded_manifest, base)
        if (
            not isinstance(loaded_manifest, Mapping)
            or loaded_manifest.get("submission_sha256") != base["submission_sha256"]
            or loaded_manifest.get("blind_to_existing_ranking") is not True
        ):
            raise ValueError("resume audit inputs differ from the frozen manifest")
        resume_manifest = loaded_manifest
    elif root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit root: {root}")
    if client is None:
        client = baseline.OpenRouterClient(baseline.private_api_key())
    live_pricing = resume_manifest.get("live_pricing") if resume_manifest is not None else None
    if not resume and isinstance(client, baseline.OpenRouterClient):
        live_pricing = client.assert_discounted_price()
    if not resume:
        root.mkdir(parents=True)
        baseline._write(
            root / "manifest.json", {**base, "status": "running", "live_pricing": live_pricing}
        )
    for index, request in enumerate(requests, start=1):
        baseline._write_or_verify(root / f"batch-{index:04d}" / "request.json", request)

    started = time.monotonic()
    terminal_path = root / "openrouter-batch-terminal.json"
    if resume:
        created = baseline._json(root / "openrouter-batch-created.json")
        batch_id = created.get("id") if isinstance(created, Mapping) else None
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("frozen OpenRouter audit batch ID is unavailable")
        terminal = baseline._json(terminal_path) if terminal_path.exists() else client.wait_batch(batch_id)
    else:
        terminal = client.complete_many(
            requests,
            on_created=lambda created: baseline._write(
                root / "openrouter-batch-created.json", created
            ),
        )
    duration = time.monotonic() - started
    baseline._write_or_verify(terminal_path, terminal)
    usage = baseline.validate_terminal_batch(
        terminal, expected_request_count=len(batch_list), max_cost_usd=max_cost_usd
    )
    results = terminal.get("results")
    assert isinstance(results, list)
    by_custom_id = {
        result.get("custom_id"): result
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("custom_id"), str)
    }
    assessments: dict[int, dict[str, Any]] = {}
    for index, batch in enumerate(batch_list, start=1):
        custom_id = f"batch-{index:04d}"
        result = by_custom_id.get(custom_id)
        if not isinstance(result, Mapping) or result.get("error") is not None:
            raise ValueError(f"OpenRouter audit result failed for {custom_id}")
        response = result.get("response")
        body = (
            response.get("body")
            if isinstance(response, Mapping) and response.get("status_code") == 200
            else None
        )
        if not isinstance(body, Mapping):
            raise ValueError(f"OpenRouter audit body is invalid for {custom_id}")
        batch_root = root / custom_id
        baseline._write_or_verify(batch_root / "raw-response.json", baseline._safe_raw(body))
        payload, _ = baseline._provider_payload(body)
        accepted = validate_response(payload, batch)
        _write_or_verify_accepted(batch_root / "accepted.json", accepted, duration)
        assessments.update(accepted)

    records = organize_audit(rows, assessments)
    queue = review_queue(records)
    audit_sha = _write_or_verify_jsonl(root / "submission-audit.jsonl", records)
    queue_sha = _write_or_verify_jsonl(root / "video-review-queue.jsonl", queue)
    quality_counts = Counter(record["match_quality"] for record in records)
    issue_counts = Counter(
        issue for record in records for issue in record["issue_types"]
    )
    summary = {
        **base,
        "status": "complete",
        "live_pricing": live_pricing,
        "completed_query_count": len(records),
        "match_quality_counts": dict(sorted(quality_counts.items())),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "selected_top_is_blind_best_count": sum(
            bool(record["selected_top_is_best"]) for record in records
        ),
        "selected_top_supported_count": sum(
            bool(record["selected_top_supported"]) for record in records
        ),
        "review_required_count": len(queue),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "accounted_cost_usd": round(usage["cost"], 8),
        "submission_audit_sha256": audit_sha,
        "video_review_queue_sha256": queue_sha,
    }
    baseline._write_or_verify(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2", type=Path, required=True, help="Task 2 JSON path, or -")
    parser.add_argument("--fusion", type=Path, default=baseline.DEFAULT_FUSION)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=baseline.EXPECTED_QUERY_COUNT)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--max-cost-usd", type=float, default=baseline.GLOBAL_MAX_COST_USD)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    task2_source: Path | bytes = sys.stdin.buffer.read() if args.task2 == Path("-") else args.task2
    result = run_full(
        task2=task2_source,
        fusion=args.fusion,
        submission=args.submission,
        output_root=args.output_root,
        expected_query_count=args.expected_query_count,
        query_limit=args.query_limit,
        max_cost_usd=args.max_cost_usd,
        run=args.run,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
