#!/usr/bin/env python3
"""Rank released MC options from the frozen fusion rank-one AG answer.

The default mode is provider-free.  ``--run`` dispatches text-only requests
through OpenRouter's explicitly discounted Gemini batch model, and ``--resume``
continues a frozen provider batch without dispatching a second paid job.  The
final submission is assembled only from exact option strings in the released
Task 2 file; model-produced free text can never enter the submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemini-3.7-flash"
BATCH_MODEL = f"{MODEL}:batch"
STRATEGY = "fusion_rank1_text_llm_option_ranking"
BATCH_SIZE = 10
MAX_CONCURRENCY = 4
MAX_OUTPUT_TOKENS = 16_384
INPUT_USD_PER_MILLION = 0.1875
OUTPUT_USD_PER_MILLION = 0.9375
GLOBAL_MAX_COST_USD = 5.0
EXPECTED_QUERY_COUNT = 2_443
SEED = 42
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/openrouter.env")
DEFAULT_FUSION = ROOT / "submissions/trecvid-2026/vqa-ag/g37_q36_fusion/submission.json"
BATCHES_URL = "https://openrouter.ai/api/beta/batches"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc


def _source_json(source: Path | bytes, label: str) -> Any:
    if isinstance(source, bytes):
        try:
            return json.loads(source)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read JSON artifact: {label}") from exc
    return _json(Path(source))


def _source_sha256(source: Path | bytes) -> str:
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _rows(value: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"rows are not an object list: {path}")
    return [dict(row) for row in value]


def _qid(row: Mapping[str, Any], path: Path) -> int:
    value = row.get("Q_ID")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid Q_ID in {path}")
    return value


def load_rows(
    task2: Path | bytes,
    fusion: Path = DEFAULT_FUSION,
    *,
    expected_query_count: int | None = EXPECTED_QUERY_COUNT,
    query_limit: int | None = None,
) -> list[dict[str, Any]]:
    task2_label = Path(task2) if not isinstance(task2, bytes) else Path("<task2-stdin>")
    fusion_path = Path(fusion)
    task_rows = _rows(_source_json(task2, str(task2_label)), task2_label)
    fusion_rows = _rows(_json(fusion_path), fusion_path)
    if query_limit is not None:
        if isinstance(query_limit, bool) or not isinstance(query_limit, int) or query_limit <= 0:
            raise ValueError("query limit must be a positive integer")
        task_rows = task_rows[:query_limit]
        fusion_rows = fusion_rows[:query_limit]
    if expected_query_count is not None and len(task_rows) != expected_query_count:
        raise ValueError(f"Task 2 must contain {expected_query_count} rows")
    if len(fusion_rows) != len(task_rows):
        raise ValueError("Task 2 and fusion row counts differ")

    fusion_by_id: dict[int, Mapping[str, Any]] = {}
    for row in fusion_rows:
        query_id = _qid(row, fusion_path)
        if query_id in fusion_by_id:
            raise ValueError("duplicate fusion Q_ID")
        fusion_by_id[query_id] = row

    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for task_row in task_rows:
        query_id = _qid(task_row, task2_label)
        if query_id in seen or query_id not in fusion_by_id:
            raise ValueError("Task 2 Q_ID coverage is invalid")
        seen.add(query_id)
        fusion_row = fusion_by_id[query_id]
        video_id = task_row.get("Video_ID")
        question = task_row.get("Question")
        if not isinstance(video_id, str) or not video_id or fusion_row.get("Video_ID") != video_id:
            raise ValueError(f"video identity mismatch for Q_ID {query_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"invalid question for Q_ID {query_id}")
        raw_options = task_row.get("Options")
        if not isinstance(raw_options, list) or len(raw_options) < 2:
            raise ValueError(f"invalid options for Q_ID {query_id}")
        options: list[str] = []
        for option in raw_options:
            text = option.get("option") if isinstance(option, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"invalid option for Q_ID {query_id}")
            options.append(text)
        if len(options) != len(set(options)):
            raise ValueError(f"duplicate option text for Q_ID {query_id}")

        answers = fusion_row.get("Answers")
        if not isinstance(answers, list):
            raise ValueError(f"fusion answers missing for Q_ID {query_id}")
        rank_one = [answer for answer in answers if isinstance(answer, Mapping) and answer.get("Rank") == 1]
        if len(rank_one) != 1 or not isinstance(rank_one[0].get("Answer"), str) or not rank_one[0]["Answer"].strip():
            raise ValueError(f"fusion rank one is invalid for Q_ID {query_id}")
        output.append(
            {
                "query_id": query_id,
                "video_id": video_id,
                "question": question.strip(),
                "rank1_answer": rank_one[0]["Answer"],
                "options": options,
            }
        )
    if set(fusion_by_id) != seen:
        raise ValueError("fusion Q_ID coverage differs from Task 2")
    return output


def batches(
    rows: Sequence[Mapping[str, Any]], batch_size: int = BATCH_SIZE
) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [list(map(dict, rows[index : index + batch_size])) for index in range(0, len(rows), batch_size)]


def _response_schema(_query_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "maxItems": BATCH_SIZE,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "query_id": {"type": "integer"},
                        "ranking": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 11,
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["query_id", "ranking"],
                },
            }
        },
        "required": ["queries"],
    }


def build_request(batch: Sequence[Mapping[str, Any]], batch_id: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("request batch is empty")
    queries = []
    for row in batch:
        queries.append(
            {
                "query_id": int(row["query_id"]),
                "question": row["question"],
                "prior_video_answer": row["rank1_answer"],
                "options": [
                    {"option_id": index, "option": option}
                    for index, option in enumerate(row["options"], start=1)
                ],
            }
        )
    schema = _response_schema(len(batch))
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rank only the supplied multiple-choice options using the prior answer, "
                    "which was generated earlier from the video. Interpret paraphrases, numbers, "
                    "and negation semantically. For every query, return every supplied option_id "
                    "exactly once from most compatible to least compatible. If no option is exact, "
                    "rank the closest meanings. Do not invent an option or a new video answer."
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
                "name": "vqa_mc_rankings",
                "strict": True,
                "schema": schema,
            },
        },
        "reasoning": {"effort": "high", "exclude": True},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": SEED,
        "temperature": 0,
        "usage": {"include": True},
    }


def build_batch_envelope(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not requests:
        raise ValueError("OpenRouter batch must contain requests")
    for request in requests:
        if request.get("model") != MODEL:
            raise ValueError("OpenRouter batch request model drift")
    # Field order is material to OpenRouter's streaming batch parser.
    return {
        "endpoint": "/v1/chat/completions",
        "model": MODEL,
        "requests": [
            {"custom_id": f"batch-{index:04d}", "body": dict(request)}
            for index, request in enumerate(requests, start=1)
        ],
    }


def validate_response(
    payload: Mapping[str, Any], batch: Sequence[Mapping[str, Any]]
) -> dict[int, list[int]]:
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != len(batch):
        raise ValueError("response query count is invalid")
    expected = {int(row["query_id"]): row for row in batch}
    result: dict[int, list[int]] = {}
    for query in queries:
        if not isinstance(query, Mapping) or set(query) != {"query_id", "ranking"}:
            raise ValueError("response query schema is invalid")
        query_id = query.get("query_id")
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id not in expected or query_id in result:
            raise ValueError("response query identity is invalid")
        ranking = query.get("ranking")
        option_count = len(expected[query_id]["options"])
        if (
            not isinstance(ranking, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ranking)
            or sorted(ranking) != list(range(1, option_count + 1))
        ):
            raise ValueError(f"response ranking is not a complete permutation for Q_ID {query_id}")
        result[query_id] = list(ranking)
    if set(result) != set(expected):
        raise ValueError("response query coverage is incomplete")
    return result


def organize_submission(
    rows: Sequence[Mapping[str, Any]],
    rankings: Mapping[int, Sequence[int]],
    *,
    elapsed_seconds: Mapping[int, float] | None = None,
) -> list[dict[str, Any]]:
    del elapsed_seconds  # MC submissions do not contain generation time.
    output = []
    for row in rows:
        query_id = int(row["query_id"])
        ranking = rankings.get(query_id)
        if ranking is None:
            raise ValueError(f"missing ranking for Q_ID {query_id}")
        options = row["options"]
        output.append(
            {
                "Q_ID": query_id,
                "Video_ID": row["video_id"],
                "Answers": [
                    {"Rank": rank, "Answer": options[option_id - 1]}
                    for rank, option_id in enumerate(ranking, start=1)
                ],
            }
        )
    return output


def _estimate(request: Mapping[str, Any]) -> dict[str, Any]:
    # One UTF-8 wire byte per input token is a conservative reservation.
    prompt_tokens = max(1, len(_canonical(request)))
    output_tokens = MAX_OUTPUT_TOKENS
    cost = (
        prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    return {
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def _validate_cap(max_cost_usd: float) -> None:
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
        or max_cost_usd > GLOBAL_MAX_COST_USD
    ):
        raise ValueError(f"max-cost-usd must be finite, positive, and <= {GLOBAL_MAX_COST_USD}")


def private_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    path = PRIVATE_ENV_FILE
    try:
        if path.stat().st_mode & 0o077:
            raise ValueError(f"private env file permissions must be 0600: {path}")
        values: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, raw_value = line.split("=", 1)
            parts = shlex.split(raw_value, posix=True)
            if len(parts) == 1:
                values[name.strip()] = parts[0]
    except OSError as exc:
        raise ValueError(f"cannot read private OpenRouter env file: {path}") from exc
    key = values.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is unavailable")
    return key


def validate_openrouter_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter returned a non-object response")
    if payload.get("error") is not None:
        raise RuntimeError(f"OpenRouter error: {payload['error']}")
    return payload


def validate_discounted_pricing(prompt: Any, completion: Any) -> dict[str, float]:
    values = {"input_usd_per_million": prompt, "output_usd_per_million": completion}
    normalized: dict[str, float] = {}
    for label, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("OpenRouter model pricing is invalid")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise RuntimeError("OpenRouter model pricing is invalid")
        normalized[label] = number
    if (
        normalized["input_usd_per_million"] > INPUT_USD_PER_MILLION
        or normalized["output_usd_per_million"] > OUTPUT_USD_PER_MILLION
    ):
        raise RuntimeError(
            "discount expired: live price is "
            f"${normalized['input_usd_per_million']}/M input and "
            f"${normalized['output_usd_per_million']}/M output"
        )
    return normalized


def _valid_model_identity(value: Any) -> bool:
    return isinstance(value, str) and (value == MODEL or value.startswith(f"{MODEL}-"))


def validate_terminal_batch(
    terminal: Mapping[str, Any], *, expected_request_count: int, max_cost_usd: float
) -> dict[str, Any]:
    _validate_cap(max_cost_usd)
    if terminal.get("status") != "completed":
        raise ValueError(f"OpenRouter batch status is not completed: {terminal.get('status')}")
    terminal_model = terminal.get("model")
    if terminal_model is not None and not _valid_model_identity(terminal_model):
        raise ValueError(f"OpenRouter batch model is invalid: {terminal_model}")

    counts = terminal.get("request_counts")
    if not isinstance(counts, Mapping) or any(
        isinstance(counts.get(key), bool) or not isinstance(counts.get(key), int)
        for key in ("total", "completed", "failed")
    ):
        raise ValueError("OpenRouter batch request counts are invalid")
    if (
        counts["total"] != expected_request_count
        or counts["completed"] != expected_request_count
        or counts["failed"] != 0
    ):
        raise ValueError("OpenRouter batch request coverage is invalid")

    results = terminal.get("results")
    if not isinstance(results, list) or len(results) != expected_request_count:
        raise ValueError("OpenRouter batch result coverage is invalid")
    for result in results:
        response = result.get("response") if isinstance(result, Mapping) else None
        body = response.get("body") if isinstance(response, Mapping) else None
        if not isinstance(body, Mapping) or not _valid_model_identity(body.get("model")):
            raise ValueError("OpenRouter result model is invalid")

    usage = terminal.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("OpenRouter batch usage metadata is required")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (
        isinstance(prompt, bool)
        or not isinstance(prompt, int)
        or prompt < 0
        or isinstance(completion, bool)
        or not isinstance(completion, int)
        or completion < 0
    ):
        raise ValueError("OpenRouter batch token usage is invalid")
    cost = usage.get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise ValueError("OpenRouter batch cost is invalid")
    cost = float(cost)
    if not math.isfinite(cost) or cost < 0 or cost > max_cost_usd:
        raise ValueError("OpenRouter batch cost is invalid or exceeds the run cap")
    calculated = (
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + completion / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
    tolerance = max(1e-8, calculated * 1e-6)
    if cost > calculated + tolerance:
        raise ValueError("OpenRouter batch cost exceeds the token-price ceiling")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost}


class OpenRouterClient:
    def __init__(self, api_key: str, *, timeout_seconds: float = 900.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _request(self, url: str, *, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = _canonical(body) if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method="POST" if body is not None else "GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "TRECVID 2026 VQA MC rank1 baseline",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        return validate_openrouter_payload(payload)

    def complete_many(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        poll_interval_seconds: float = 10.0,
        on_created: Any = None,
    ) -> dict[str, Any]:
        envelope = build_batch_envelope(requests)
        created = self._request(BATCHES_URL, body=envelope)
        batch_id = created.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise RuntimeError("OpenRouter batch creation returned no batch ID")
        if on_created is not None:
            on_created(created)
        return self.wait_batch(batch_id, poll_interval_seconds=poll_interval_seconds)

    def wait_batch(
        self,
        batch_id: str,
        *,
        poll_interval_seconds: float = 10.0,
        max_transient_errors: int = 6,
    ) -> dict[str, Any]:
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("OpenRouter batch ID is invalid")
        terminal = frozenset({"completed", "failed", "cancelled", "expired"})
        status = None
        created: dict[str, Any] = {}
        transient_errors = 0
        while status not in terminal:
            if status is not None:
                time.sleep(poll_interval_seconds)
            try:
                created = self._request(f"{BATCHES_URL}/{batch_id}")
                transient_errors = 0
            except RuntimeError:
                transient_errors += 1
                if transient_errors > max_transient_errors:
                    raise
                time.sleep(min(poll_interval_seconds * (2 ** (transient_errors - 1)), 30.0))
                continue
            status = created.get("status")
        if status != "completed":
            raise RuntimeError(f"OpenRouter batch ended with status {status}: {created.get('error')}")
        return created

    def assert_discounted_price(self) -> dict[str, float]:
        payload = self._request(MODELS_URL)
        models = payload.get("data")
        if not isinstance(models, list):
            raise RuntimeError("OpenRouter models response is invalid")
        match = next((item for item in models if isinstance(item, Mapping) and item.get("id") == BATCH_MODEL), None)
        if not isinstance(match, Mapping) or not isinstance(match.get("pricing"), Mapping):
            raise RuntimeError(f"discounted model is unavailable: {BATCH_MODEL}")
        try:
            prompt = float(match["pricing"]["prompt"]) * 1_000_000
            completion = float(match["pricing"]["completion"]) * 1_000_000
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("OpenRouter model pricing is invalid") from exc
        return validate_discounted_pricing(prompt, completion)


def _provider_payload(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    choices = raw.get("choices")
    content = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("provider response has no JSON content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("provider JSON content is not an object")
    usage = raw.get("usage")
    if not isinstance(usage, Mapping):
        return payload, {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        raise ValueError("provider token usage is invalid")
    normalized = {"prompt_tokens": prompt, "completion_tokens": completion}
    if isinstance(total, int):
        normalized["total_tokens"] = total
    return payload, normalized


def _safe_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key in {"id", "model", "provider", "choices", "usage", "openrouter_metadata"}
    }


def _write(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_or_verify(path: Path, value: Any) -> str:
    if not path.exists() and not path.is_symlink():
        return _write(path, value)
    existing = _json(path)
    if _canonical(existing) != _canonical(value):
        raise ValueError(f"existing immutable artifact differs: {path}")
    return _sha_file(path)


def _write_or_verify_accepted(path: Path, rankings: Mapping[int, Sequence[int]], duration: float) -> str:
    value = {"status": "accepted", "elapsed_seconds": duration, "rankings": rankings}
    if not path.exists() and not path.is_symlink():
        return _write(path, value)
    existing = _json(path)
    if (
        not isinstance(existing, Mapping)
        or existing.get("status") != "accepted"
        or _canonical(existing.get("rankings")) != _canonical(rankings)
    ):
        raise ValueError(f"existing accepted artifact differs: {path}")
    return _sha_file(path)


def _validate_resume_manifest(manifest: Any, base: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("resume manifest is invalid")
    immutable = (
        "strategy",
        "model",
        "dispatched_model",
        "api_mode",
        "query_count",
        "batch_count",
        "batch_size",
        "max_output_tokens_per_batch",
        "max_cost_usd",
        "task2_sha256",
        "fusion_sha256",
        "request_set_sha256",
    )
    if any(manifest.get(key) != base.get(key) for key in immutable):
        raise ValueError("resume inputs or run configuration differ from the frozen manifest")


def run_full(
    *,
    task2: Path | bytes,
    fusion: Path = DEFAULT_FUSION,
    output_root: Path,
    expected_query_count: int | None = EXPECTED_QUERY_COUNT,
    query_limit: int | None = None,
    max_cost_usd: float = GLOBAL_MAX_COST_USD,
    run: bool = False,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    _validate_cap(max_cost_usd)
    rows = load_rows(
        task2,
        fusion,
        expected_query_count=expected_query_count,
        query_limit=query_limit,
    )
    batch_list = batches(rows)
    requests = [build_request(batch, index) for index, batch in enumerate(batch_list, start=1)]
    estimates = [_estimate(request) for request in requests]
    estimated_cost = round(sum(item["estimated_cost_usd"] for item in estimates), 8)
    if estimated_cost > max_cost_usd:
        raise ValueError("estimated full-panel cost exceeds run cap")
    base = {
        "status": "dry_run",
        "strategy": STRATEGY,
        "model": BATCH_MODEL,
        "dispatched_model": MODEL,
        "api_mode": "openrouter_async_batch",
        "query_count": len(rows),
        "batch_count": len(batch_list),
        "batch_size": BATCH_SIZE,
        "max_output_tokens_per_batch": MAX_OUTPUT_TOKENS,
        "estimated_cost_usd": estimated_cost,
        "max_cost_usd": max_cost_usd,
        "task2_sha256": _source_sha256(task2),
        "fusion_sha256": _sha_file(Path(fusion)),
        "request_set_sha256": _sha_value(requests),
    }
    if not run:
        if resume:
            raise ValueError("--resume requires --run")
        return base

    root = Path(output_root)
    resume_manifest: Mapping[str, Any] | None = None
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"resume run root does not exist: {root}")
        loaded_manifest = _json(root / "manifest.json")
        _validate_resume_manifest(loaded_manifest, base)
        assert isinstance(loaded_manifest, Mapping)
        resume_manifest = loaded_manifest
    elif root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing to overwrite run root: {root}")
    if client is None:
        client = OpenRouterClient(private_api_key())
    live_pricing = resume_manifest.get("live_pricing") if resume_manifest is not None else None
    if not resume and isinstance(client, OpenRouterClient):
        live_pricing = client.assert_discounted_price()
    if not resume:
        root.mkdir(parents=True)
        _write(root / "manifest.json", {**base, "status": "running", "live_pricing": live_pricing})

    for index, request in enumerate(requests, start=1):
        _write_or_verify(root / f"batch-{index:04d}" / "request.json", request)

    started = time.monotonic()
    terminal_path = root / "openrouter-batch-terminal.json"
    if resume:
        created = _json(root / "openrouter-batch-created.json")
        batch_id = created.get("id") if isinstance(created, Mapping) else None
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("frozen OpenRouter batch ID is unavailable")
        terminal_batch = _json(terminal_path) if terminal_path.exists() else client.wait_batch(batch_id)
    else:
        terminal_batch = client.complete_many(
            requests,
            on_created=lambda created: _write(root / "openrouter-batch-created.json", created),
        )
    duration = time.monotonic() - started
    _write_or_verify(terminal_path, terminal_batch)
    normalized_usage = validate_terminal_batch(
        terminal_batch,
        expected_request_count=len(batch_list),
        max_cost_usd=max_cost_usd,
    )
    results = terminal_batch.get("results")
    assert isinstance(results, list)  # Established by validate_terminal_batch.
    by_custom_id = {
        result.get("custom_id"): result
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("custom_id"), str)
    }
    rankings: dict[int, list[int]] = {}
    elapsed: dict[int, float] = {}
    for index, batch in enumerate(batch_list, start=1):
        custom_id = f"batch-{index:04d}"
        result = by_custom_id.get(custom_id)
        if not isinstance(result, Mapping) or result.get("error") is not None:
            raise ValueError(f"OpenRouter result failed for {custom_id}")
        response = result.get("response")
        body = response.get("body") if isinstance(response, Mapping) and response.get("status_code") == 200 else None
        if not isinstance(body, Mapping):
            raise ValueError(f"OpenRouter result body is invalid for {custom_id}")
        batch_root = root / custom_id
        _write_or_verify(batch_root / "raw-response.json", _safe_raw(body))
        payload, _ = _provider_payload(body)
        accepted = validate_response(payload, batch)
        _write_or_verify_accepted(batch_root / "accepted.json", accepted, duration)
        rankings.update(accepted)
        elapsed.update({int(row["query_id"]): duration for row in batch})

    actual_prompt_tokens = normalized_usage["prompt_tokens"]
    actual_completion_tokens = normalized_usage["completion_tokens"]

    submission = organize_submission(rows, rankings, elapsed_seconds=elapsed)
    submission_hash = _write_or_verify(root / "submission.json", submission)
    actual_cost = round(normalized_usage["cost"], 8)
    summary = {
        **base,
        "status": "complete",
        "live_pricing": live_pricing,
        "completed_query_count": len(rankings),
        "prompt_tokens": actual_prompt_tokens,
        "completion_tokens": actual_completion_tokens,
        "accounted_cost_usd": actual_cost,
        "submission_sha256": submission_hash,
    }
    _write_or_verify(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2", type=Path, required=True, help="Task 2 JSON path, or - for stdin")
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=EXPECTED_QUERY_COUNT)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--max-cost-usd", type=float, default=GLOBAL_MAX_COST_USD)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    task2_source: Path | bytes = sys.stdin.buffer.read() if args.task2 == Path("-") else args.task2
    result = run_full(
        task2=task2_source,
        fusion=args.fusion,
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
