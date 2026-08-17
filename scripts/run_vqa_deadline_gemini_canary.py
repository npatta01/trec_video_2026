#!/usr/bin/env python3
"""Run exactly one synchronous Gemini 3.7 Flash native-AV request per panel row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from google import genai
from google.genai import types

from trec_video_2026 import vqa_deadline_canary as contract
from trec_video_2026 import vqa_gemini37_flash_smoke as smoke


_REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = _REPO_ROOT / "configs/vqa/deadline_canary_5.json"
DATA_ROOT = _REPO_ROOT / "data"
HISTORICAL_ROOT = DATA_ROOT / "vqa/benchmarks/direct-gemini-batch-475-v1"
PRIVATE_RUN_ROOT = DATA_ROOT / "vqa/benchmarks/deadline-submission-canary-5-20260816/gemini"
PRIVATE_ENV_FILE = Path("/home/npatta01/.config/trec_video_2026/gemini.env")

MODEL = "gemini-3.7-flash"
MIME_TYPE = "video/webm"
MAX_OUTPUT_TOKENS = 2048
MAX_ESTIMATED_COST_USD = 0.15
INPUT_USD_PER_MILLION = 0.375
OUTPUT_USD_PER_MILLION = 1.875
# The smoke's measured native-AV prompt range is about 864--1,555 tokens/sec;
# this ceiling intentionally uses the high end plus a small question allowance.
ESTIMATED_INPUT_TOKENS_PER_SECOND = 1_600
ESTIMATED_QUESTION_TOKENS = 100
MAX_UPLOAD_POLLS = 30
UPLOAD_POLL_SECONDS = 1.0


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


def _is_regular_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return not path.is_symlink() and stat.S_ISREG(mode)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["_artifact_sha256"] = _sha256_bytes(_canonical_json(body))
    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return _sha256_bytes(serialized.encode("utf-8"))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object: {path.name}")
    expected = value.get("_artifact_sha256")
    body = dict(value)
    body.pop("_artifact_sha256", None)
    if expected != _sha256_bytes(_canonical_json(body)):
        raise ValueError(f"artifact hash mismatch: {path.name}")
    return value


def _delivery_path(historical_root: Path, video_id: str) -> Path:
    root = Path(historical_root).resolve(strict=True)
    candidate = root / "deliveries" / video_id / "delivery-media" / f"{video_id}.webm"
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"delivery is outside the historical root for {video_id}") from error
    if not _is_regular_file(candidate):
        raise ValueError(f"preserved WebM delivery is missing for {video_id}")
    return candidate


def _frozen_rows(
    *, panel_path: Path, data_root: Path, historical_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = contract.load_panel(panel_path, data_root=data_root)
    frozen: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row["video_id"])
        delivery = _delivery_path(historical_root, video_id)
        frozen.append(
            {
                "video_id": video_id,
                "stratum": row["stratum"],
                "question": row["question"],
                "duration_seconds": row["duration_seconds"],
                "delivery_path": str(delivery),
                "delivery_sha256": _sha256_file(delivery),
                "delivery_size_bytes": delivery.stat().st_size,
                "delivery_mime_type": MIME_TYPE,
                "source_sha256": row["source_sha256"],
            }
        )
    return rows, frozen


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
    """Use the smoke-tested native video+audio request primitive for one row."""

    bound = dict(row)
    bound["row_id"] = str(row["video_id"])
    bound["delivery_sha256"] = str(row.get("delivery_sha256", ""))
    bound["protocol_sha256"] = str(row.get("protocol_sha256", ""))
    upload_bound = {
        "file_uri": upload["file_uri"],
        "mime_type": upload.get("mime_type", MIME_TYPE),
    }
    return smoke.build_inlined_request(
        row=bound,
        upload=upload_bound,
        arm="candidate_37flash",
    )


def _upload_once(
    *,
    client: Any,
    row: Mapping[str, Any],
    delivery: Path,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    before_hash = _sha256_file(delivery)
    before_size = delivery.stat().st_size
    uploaded = client.files.upload(
        file=delivery,
        config=types.UploadFileConfig(mime_type=MIME_TYPE),
    )
    current = smoke._provider_file_metadata(uploaded, expected_mime=MIME_TYPE)
    if current.get("size_bytes") is not None and current["size_bytes"] != before_size:
        raise ValueError("provider upload size does not match preserved delivery")
    if current.get("sha256_hash") is not None and current["sha256_hash"] != before_hash:
        raise ValueError("provider upload hash does not match preserved delivery")
    for poll_index in range(MAX_UPLOAD_POLLS):
        if current["file_state"] == "ACTIVE":
            break
        if current["file_state"] == "failed":
            raise ValueError("provider upload entered FAILED state")
        if poll_index == MAX_UPLOAD_POLLS - 1:
            raise ValueError("provider upload timed out before ACTIVE")
        current = smoke._provider_file_metadata(
            client.files.get(name=current["name"]), expected_mime=MIME_TYPE
        )
        sleep(UPLOAD_POLL_SECONDS)
    if _sha256_file(delivery) != before_hash or delivery.stat().st_size != before_size:
        raise ValueError("preserved delivery changed during upload")
    current["delivery_sha256"] = before_hash
    current["delivery_size_bytes"] = before_size
    return current


def _parse_response(*, raw: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = smoke._response_payload(raw)
    text = smoke._response_text(payload)
    answer = text.strip() if isinstance(text, str) else ""
    usage = smoke._usage(payload)
    failure: str | None = None
    if not answer:
        failure = "empty_text"
    if payload.get("error"):
        failure = "provider_error"
    response_id = payload.get("response_id", payload.get("responseId", payload.get("id")))
    finish_reason = payload.get("finish_reason", payload.get("finishReason"))
    return {
        "video_id": row["video_id"],
        "stratum": row["stratum"],
        "status": "succeeded" if failure is None else "failed",
        "answer": answer,
        "response_id": response_id if isinstance(response_id, str) else None,
        "model_version": payload.get("model_version", payload.get("modelVersion")),
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "usage": usage,
        "error": smoke._safe_error(payload.get("error")) if failure else None,
        "failure_reason": failure,
    }


def _observed_cost(parsed_rows: Sequence[Mapping[str, Any]]) -> tuple[int, int, float | None]:
    prompt = 0
    output = 0
    complete = True
    for row in parsed_rows:
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            complete = False
            continue
        prompt_value = usage.get("prompt_tokens")
        output_value = usage.get("output_tokens")
        if not isinstance(prompt_value, int) or not isinstance(output_value, int):
            complete = False
            continue
        prompt += prompt_value
        output += output_value
    if not complete:
        return prompt, output, None
    cost = prompt / 1_000_000 * INPUT_USD_PER_MILLION + output / 1_000_000 * OUTPUT_USD_PER_MILLION
    return prompt, output, round(cost, 8)


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
    protocol_sha256 = _sha256_bytes(_canonical_json({"rows": list(frozen_rows)}))
    manifest = {
        "schema_version": 1,
        "experiment_id": "deadline-submission-canary-5-20260816-gemini",
        "status": "inputs_frozen",
        "created_at": created_at,
        "model": MODEL,
        "protocol_sha256": protocol_sha256,
        "rows": list(frozen_rows),
        "cost": dict(estimate),
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
        attempt_path = uploads_dir / f"{video_id}.attempt.json"
        attempt_sha256 = _write_json_exclusive(attempt_path, attempt)
        try:
            provider = _upload_once(client=client, row=row, delivery=delivery, sleep=sleep)
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
        except Exception as error:  # noqa: BLE001 - preserve terminal upload failure
            record = {
                "schema_version": 1,
                "status": "failed",
                "video_id": video_id,
                "delivery_sha256": frozen["delivery_sha256"],
                "delivery_size_bytes": frozen["delivery_size_bytes"],
                "mime_type": MIME_TYPE,
                "attempt_sha256": attempt_sha256,
                "provider_error": smoke._safe_error(error),
            }
            _write_json_exclusive(uploads_dir / f"{video_id}.json", record)
            raise
        _write_json_exclusive(uploads_dir / f"{video_id}.json", record)
        upload_records.append(record)

    parsed_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row, frozen, upload in zip(rows, frozen_rows, upload_records, strict=True):
        video_id = str(row["video_id"])
        request = build_request(row=row, upload=upload)
        request_dump = smoke._plain(request.model_dump(mode="json", by_alias=True, exclude_none=True))
        request_record = {
            "video_id": video_id,
            "model": MODEL,
            "delivery_sha256": frozen["delivery_sha256"],
            "request_sha256": _sha256_bytes(_canonical_json(request_dump)),
        }
        _write_json_exclusive(requests_dir / f"{video_id}.json", request_record)
        try:
            # One direct, synchronous native-AV call. There is deliberately no retry.
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
        except Exception as error:  # noqa: BLE001 - terminalize this one request
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

    # These two terminal artifacts are the freeze boundary. No reference or
    # annotation path is accepted by this runner or opened before this point.
    raw_freeze = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "raw_frozen",
        "frozen_at": created_at,
        "model": MODEL,
        "manifest_sha256": _sha256_file(output_root / "manifest.json"),
        "rows": raw_rows,
    }
    raw_freeze_sha256 = _write_json_exclusive(output_root / "raw-freeze.json", raw_freeze)
    prompt_tokens, output_tokens, observed_cost = _observed_cost(parsed_rows)
    predictions = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
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
        "observed_cost_usd": observed_cost,
    }
    predictions_sha256 = _write_json_exclusive(
        output_root / "predictions-freeze.json", predictions
    )
    summary = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "predictions_frozen",
        "row_count": len(parsed_rows),
        "completed_count": predictions["completed_count"],
        "upload_count": len(upload_records),
        "request_count": len(parsed_rows),
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "observed_cost_usd": observed_cost,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
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
    rows, frozen_rows = _frozen_rows(
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
        created_at=created_at or _now(),
        sleep=sleep,
        estimate=estimate,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def client_factory(*, api_key: str) -> Any:
    return genai.Client(api_key=api_key)


def _private_api_key() -> str:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    if value:
        return value
    try:
        mode = PRIVATE_ENV_FILE.stat().st_mode & 0o777
    except OSError as error:
        raise ValueError("approved Gemini credential file is unavailable") from error
    if mode != 0o600:
        raise ValueError("approved Gemini credential file permissions are not 0600")
    try:
        lines = PRIVATE_ENV_FILE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("approved Gemini credential file cannot be read") from error
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != "GEMINI_API_KEY":
            continue
        try:
            parsed = shlex.split(raw_value.strip(), comments=True)
        except ValueError as error:
            raise ValueError("approved Gemini credential file is malformed") from error
        if parsed and parsed[0].strip():
            return parsed[0].strip()
    raise ValueError("GEMINI_API_KEY is not configured")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_gemini_canary")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit(result: Mapping[str, Any]) -> None:
    safe_keys = (
        "status",
        "row_count",
        "completed_count",
        "upload_count",
        "request_count",
        "estimated_cost_usd",
        "observed_cost_usd",
        "prompt_tokens",
        "output_tokens",
    )
    print(json.dumps({key: result[key] for key in safe_keys if key in result}, sort_keys=True))


def main(
    argv: Sequence[str] | None = None,
    *,
    client_factory_override: Callable[..., Any] | None = None,
) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.dry_run:
            result = run_canary(dry_run=True)
        else:
            factory = client_factory_override or client_factory
            result = run_canary(client=factory(api_key=_private_api_key()))
        _emit(result)
        return 0
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - do not leak private paths/provider text
        print(json.dumps({"status": "error", "error": "command failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
