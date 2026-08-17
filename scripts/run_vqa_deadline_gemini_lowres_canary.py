#!/usr/bin/env python3
"""Run the five-row Gemini low-resolution/high-thinking canary.

This intentionally has its own output root and freeze identity.  It reuses the
same frozen panel, preserved WebM deliveries, upload checks, prompt, and
one-attempt provider protocol as the other deadline canaries, while changing
only the video sampling, media resolution, thinking level, and output ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from google import genai
from google.genai import types

from trec_video_2026 import vqa_gemini37_flash_smoke as smoke

import run_vqa_deadline_gemini_canary as high_cost


_REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = _REPO_ROOT / "configs/vqa/deadline_canary_5.json"
DATA_ROOT = _REPO_ROOT / "data"
HISTORICAL_ROOT = DATA_ROOT / "vqa/benchmarks/direct-gemini-batch-475-v1"
PRIVATE_RUN_ROOT = DATA_ROOT / (
    "vqa/benchmarks/deadline-submission-canary-5-20260816/"
    "gemini-lowres-4fps-highthink"
)
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")

MODEL = "gemini-3.7-flash"
MIME_TYPE = "video/webm"
VIDEO_FPS = 4.0
MEDIA_RESOLUTION = types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
THINKING_LEVEL = types.ThinkingLevel.HIGH
MAX_OUTPUT_TOKENS = 2048
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75
MAX_ESTIMATED_COST_USD = 0.15
MAX_OBSERVED_COST_USD = 0.15
# This is deliberately conservative for LOW-resolution 4-fps input while
# leaving room for the 2,048-token output/thinking ceiling.
ESTIMATED_INPUT_TOKENS_PER_SECOND = 1_200
ESTIMATED_QUESTION_TOKENS = 100


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["_artifact_sha256"] = _sha256_bytes(_canonical_json(body))
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return _sha256_bytes(serialized.encode("utf-8"))


def _estimated_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = sum(float(row["duration_seconds"]) for row in rows)
    prompt_tokens = math.ceil(seconds * ESTIMATED_INPUT_TOKENS_PER_SECOND) + (
        len(rows) * ESTIMATED_QUESTION_TOKENS
    )
    output_tokens = len(rows) * MAX_OUTPUT_TOKENS
    cost = prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    return {
        "duration_seconds": round(seconds, 3),
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def build_request(*, row: Mapping[str, Any], upload: Mapping[str, Any]) -> types.InlinedRequest:
    question = row.get("question", row.get("question_text", row.get("questionText")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question must be a non-empty string")
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type", MIME_TYPE)
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload file_uri must be a non-empty string")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload mime_type must be a video MIME type")
    video = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(fps=VIDEO_FPS),
        media_resolution=types.PartMediaResolution(level=MEDIA_RESOLUTION),
    )
    prompt = types.Part(text=f"{smoke.DIRECT_ANSWER_PROMPT}\n\nQuestion: {question.strip()}")
    metadata = {
        "arm": "candidate_37flash_lowres_highthink",
        "row_id": str(row.get("row_id", row.get("video_id", ""))),
        "video_id": str(row.get("video_id", "")),
        "question": question.strip(),
        "file_uri": file_uri,
        "delivery_sha256": str(row.get("delivery_sha256", "")),
        "mime_type": mime_type,
        "protocol_sha256": str(row.get("protocol_sha256", "")),
        "video_fps": str(VIDEO_FPS),
        "media_resolution": MEDIA_RESOLUTION.value,
        "thinking_level": THINKING_LEVEL.value,
    }
    config = types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="text/plain",
        seed=42,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_level=THINKING_LEVEL,
        ),
    )
    return types.InlinedRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[video, prompt])],
        metadata=metadata,
        config=config,
    )


def _parse_response(*, raw: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = smoke._response_payload(raw)
    parsed = high_cost._parse_response(raw=payload, row=row)
    finish_reason = parsed.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
            finish_reason = candidates[0].get("finish_reason", candidates[0].get("finishReason"))
    if isinstance(finish_reason, str):
        parsed["finish_reason"] = finish_reason
    if finish_reason in {"MAX_TOKENS", "MAX_TOKENS_REACHED"}:
        parsed["status"] = "failed"
        parsed["failure_reason"] = "max_tokens"
    return parsed


def _usage_cost(usage: Mapping[str, Any] | None) -> tuple[int, int, float] | None:
    """Return prompt and billed output, including hidden thinking tokens."""

    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    candidate_output = usage.get("output_tokens")
    total = usage.get("total_tokens")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (prompt, candidate_output, total)
    ) or total < prompt:
        return None
    billed_output = max(candidate_output, total - prompt)
    return (
        prompt,
        billed_output,
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + billed_output / 1_000_000 * OUTPUT_USD_PER_MILLION,
    )


def _observed_cost(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, float | None]:
    prompt = 0
    output = 0
    complete = True
    for row in rows:
        cost = _usage_cost(row.get("usage") if isinstance(row, Mapping) else None)
        if cost is None:
            complete = False
            continue
        prompt_value, output_value, _ = cost
        prompt += prompt_value
        output += output_value
    if not complete:
        return prompt, output, None
    return prompt, output, round(
        prompt / 1_000_000 * INPUT_USD_PER_MILLION
        + output / 1_000_000 * OUTPUT_USD_PER_MILLION,
        8,
    )


def _run_live(
    *,
    rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    client: Any,
    created_at: str,
    sleep: Callable[[float], None],
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    uploads_dir = output_root / "uploads"
    requests_dir = output_root / "requests"
    raw_dir = output_root / "raw"
    uploads_dir.mkdir()
    requests_dir.mkdir()
    raw_dir.mkdir()
    experiment_id = "deadline-submission-canary-5-20260816-gemini-lowres-4fps-highthink"
    protocol_sha256 = _sha256_bytes(_canonical_json({"rows": list(frozen_rows)}))
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "inputs_frozen",
        "created_at": created_at,
        "model": MODEL,
        "model_settings": {
            "video_fps": VIDEO_FPS,
            "media_resolution": MEDIA_RESOLUTION.value,
            "thinking_level": THINKING_LEVEL.value,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "protocol_sha256": protocol_sha256,
        "rows": list(frozen_rows),
        "cost": dict(estimate),
        "max_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        "max_observed_cost_usd": MAX_OBSERVED_COST_USD,
        "pricing": {
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_usd_per_million": OUTPUT_USD_PER_MILLION,
            "hidden_thinking_billed_as_output": True,
        },
    }
    _write_json_exclusive(output_root / "manifest.json", manifest)

    upload_records: list[dict[str, Any]] = []
    for row, frozen in zip(rows, frozen_rows, strict=True):
        video_id = str(row["video_id"])
        delivery = Path(str(frozen["delivery_path"]))
        attempt = {
            "schema_version": 1,
            "status": "upload_attempt_reserved",
            "video_id": video_id,
            "delivery_sha256": frozen["delivery_sha256"],
            "delivery_size_bytes": frozen["delivery_size_bytes"],
            "mime_type": MIME_TYPE,
        }
        attempt_sha256 = _write_json_exclusive(uploads_dir / f"{video_id}.attempt.json", attempt)
        provider = high_cost._upload_once(client=client, row=row, delivery=delivery, sleep=sleep)
        record = {
            "schema_version": 1,
            "status": "uploaded",
            "video_id": video_id,
            "delivery_sha256": frozen["delivery_sha256"],
            "delivery_size_bytes": frozen["delivery_size_bytes"],
            "mime_type": MIME_TYPE,
            "attempt_sha256": attempt_sha256,
            "file_uri": provider["uri"],
            "file_name": provider["name"],
            "provider_snapshot": provider,
        }
        _write_json_exclusive(uploads_dir / f"{video_id}.json", record)
        upload_records.append(record)

    parsed_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    running_cost = 0.0
    for row, frozen, upload in zip(rows, frozen_rows, upload_records, strict=True):
        video_id = str(row["video_id"])
        bound = dict(row)
        bound["protocol_sha256"] = protocol_sha256
        bound["delivery_sha256"] = frozen["delivery_sha256"]
        request = build_request(row=bound, upload=upload)
        request_dump = smoke._plain(request.model_dump(mode="json", by_alias=True, exclude_none=True))
        request_record = {
            "video_id": video_id,
            "model": MODEL,
            "settings": {
                "video_fps": VIDEO_FPS,
                "media_resolution": MEDIA_RESOLUTION.value,
                "thinking_level": THINKING_LEVEL.value,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
            "delivery_sha256": frozen["delivery_sha256"],
            "request_sha256": _sha256_bytes(_canonical_json(request_dump)),
        }
        _write_json_exclusive(requests_dir / f"{video_id}.json", request_record)
        # The declared worst-case per-row estimate keeps the five-call run
        # below the hard cap before any paid request is issued.
        row_estimate = (
            math.ceil(float(row["duration_seconds"]) * ESTIMATED_INPUT_TOKENS_PER_SECOND)
            + ESTIMATED_QUESTION_TOKENS
        ) / 1_000_000 * INPUT_USD_PER_MILLION + MAX_OUTPUT_TOKENS / 1_000_000 * OUTPUT_USD_PER_MILLION
        if running_cost + row_estimate > MAX_OBSERVED_COST_USD:
            raise ValueError("next request estimate exceeds hard observed-cost cap")
        try:
            raw = client.models.generate_content(
                model=MODEL,
                contents=request.contents,
                config=request.config,
            )
            safe_raw = smoke._safe_response_object(raw)
            parsed = _parse_response(raw=safe_raw, row=row)
            raw_record = {
                "schema_version": 1,
                "status": "response_received",
                "video_id": video_id,
                "request_sha256": _sha256_file(requests_dir / f"{video_id}.json"),
                "response": safe_raw,
            }
            usage_cost = _usage_cost(parsed.get("usage"))
            if usage_cost is not None:
                running_cost += usage_cost[2]
        except Exception as error:  # noqa: BLE001 - preserve terminal failure
            safe_error = smoke._safe_error(error)
            raw_record = {
                "schema_version": 1,
                "status": "request_failed",
                "video_id": video_id,
                "request_sha256": _sha256_file(requests_dir / f"{video_id}.json"),
                "error": safe_error,
            }
            parsed = {
                "video_id": video_id,
                "stratum": row["stratum"],
                "status": "failed",
                "answer": "",
                "response_id": None,
                "model_version": None,
                "finish_reason": None,
                "usage": None,
                "error": safe_error,
                "failure_reason": "request_failed",
            }
        _write_json_exclusive(raw_dir / f"{video_id}.json", raw_record)
        raw_rows.append(
            {
                "video_id": video_id,
                "status": raw_record["status"],
                "raw_sha256": _sha256_file(raw_dir / f"{video_id}.json"),
            }
        )
        parsed_rows.append(parsed)

    raw_freeze = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "raw_frozen",
        "frozen_at": created_at,
        "model": MODEL,
        "manifest_sha256": _sha256_file(output_root / "manifest.json"),
        "rows": raw_rows,
    }
    raw_freeze_sha256 = _write_json_exclusive(output_root / "raw-freeze.json", raw_freeze)
    prompt_tokens, output_tokens, observed_cost = _observed_cost(parsed_rows)
    if observed_cost is not None and observed_cost > MAX_OBSERVED_COST_USD:
        raise ValueError("observed cost exceeds hard cap")
    predictions = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "predictions_frozen",
        "frozen_at": created_at,
        "model": MODEL,
        "manifest_sha256": _sha256_file(output_root / "manifest.json"),
        "raw_freeze_sha256": raw_freeze_sha256,
        "row_count": len(parsed_rows),
        "completed_count": sum(row["status"] == "succeeded" for row in parsed_rows),
        "rows": parsed_rows,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "billed_output_includes_thinking_tokens": True,
        "observed_cost_usd": observed_cost,
    }
    predictions_sha256 = _write_json_exclusive(output_root / "predictions-freeze.json", predictions)
    summary = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "predictions_frozen",
        "row_count": len(parsed_rows),
        "completed_count": predictions["completed_count"],
        "upload_count": len(upload_records),
        "request_count": len(parsed_rows),
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "observed_cost_usd": observed_cost,
        "prompt_tokens": prompt_tokens,
        "billed_output_tokens": output_tokens,
        "manifest_sha256": _sha256_file(output_root / "manifest.json"),
        "raw_freeze_sha256": raw_freeze_sha256,
        "predictions_freeze_sha256": predictions_sha256,
    }
    _write_json_exclusive(output_root / "summary.json", summary)
    return summary


def run_canary(
    *,
    panel_path: Path = PANEL_PATH,
    data_root: Path = DATA_ROOT,
    historical_root: Path = HISTORICAL_ROOT,
    output_root: Path = PRIVATE_RUN_ROOT,
    client: Any | None = None,
    dry_run: bool = False,
    created_at: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    rows, frozen_rows = high_cost._frozen_rows(
        panel_path=Path(panel_path),
        data_root=Path(data_root),
        historical_root=Path(historical_root),
    )
    estimate = _estimated_cost(rows)
    if estimate["estimated_cost_usd"] > MAX_ESTIMATED_COST_USD:
        raise ValueError("estimated cost exceeds authorized cap")
    if dry_run:
        return {
            "status": "dry_run",
            "row_count": len(rows),
            "upload_count": len(rows),
            "request_count": len(rows),
            **estimate,
        }
    if client is None:
        raise ValueError("provider client is required for a live run")
    return _run_live(
        rows=rows,
        frozen_rows=frozen_rows,
        output_root=Path(output_root),
        client=client,
        created_at=created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        sleep=sleep,
        estimate=estimate,
    )


def _private_api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    mode = PRIVATE_ENV_FILE.stat().st_mode & 0o777
    if mode != 0o600:
        raise ValueError("approved Gemini credential file permissions are not 0600")
    for line in PRIVATE_ENV_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != "GEMINI_API_KEY":
            continue
        parsed = shlex.split(raw_value.strip(), comments=True)
        if parsed and parsed[0].strip():
            return parsed[0].strip()
    raise ValueError("GEMINI_API_KEY is not configured")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_gemini_lowres_canary")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run_canary(
            dry_run=args.dry_run,
            client=None if args.dry_run else genai.Client(api_key=_private_api_key()),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
