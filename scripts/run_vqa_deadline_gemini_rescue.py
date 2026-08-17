#!/usr/bin/env python3
"""Bounded, additive rescue for the 27 failed Gemini full-panel rows.

The original full-panel freeze is the only selector input.  The command is
dry-run by default; ``--run`` is required before any provider call.  Provider
artifacts live below a new immutable root and the final freeze is an overlay
envelope that can be consumed without changing the original run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT / "scripts"), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_vqa_deadline_gemini_full as full  # noqa: E402


BASE_RUN_ROOT = _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-full-20260816/gemini"
BASE_PREDICTIONS_PATH = BASE_RUN_ROOT / "predictions-freeze.json"
BASE_MANIFEST_PATH = BASE_RUN_ROOT / "manifest.json"
REQUESTS_PATH = _REPO_ROOT / "data/vqa/submissions/qwen35-safety-20260810-a1/requests.json"
MEDIA_ROOT = _REPO_ROOT / "data/vqa/2026/testing"
PRIVATE_RUN_ROOT = (
    _REPO_ROOT / "data/vqa/benchmarks/deadline-submission-rescue-20260817/gemini-27"
)

BASE_PREDICTIONS_SHA256 = (
    "e3f0e4cf6d9dd65986f328f3834e96b753b180b3b6123a646418ec1424078cb0"
)
EXPECTED_QUERY_IDS = (
    116,
    139,
    175,
    310,
    647,
    886,
    932,
    963,
    995,
    1013,
    1022,
    1055,
    1063,
    1084,
    1090,
    1127,
    1140,
    1168,
    1173,
    1190,
    1462,
    1563,
    1771,
    2279,
    2385,
    2404,
    2421,
)
EXPECTED_REASON_BY_QUERY = {
    116: "max_tokens",
    139: "max_tokens",
    175: "empty_text",
    310: "empty_text",
    647: "max_tokens",
    886: "empty_text",
    932: "max_tokens",
    963: "max_tokens",
    995: "provider_upload_failed",
    1013: "empty_text",
    1022: "provider_upload_failed",
    1055: "provider_upload_failed",
    1063: "provider_upload_failed",
    1084: "empty_text",
    1090: "provider_upload_failed",
    1127: "max_tokens",
    1140: "max_tokens",
    1168: "provider_upload_failed",
    1173: "provider_upload_failed",
    1190: "max_tokens",
    1462: "max_tokens",
    1563: "empty_text",
    1771: "empty_text",
    2279: "empty_text",
    2385: "max_tokens",
    2404: "empty_text",
    2421: "max_tokens",
}
EXPECTED_REASON_COUNTS = {
    reason: list(EXPECTED_REASON_BY_QUERY.values()).count(reason)
    for reason in {"max_tokens", "empty_text", "provider_upload_failed"}
}
EXPECTED_PANEL_COUNT = 2_443
MAX_OUTPUT_TOKENS = 8_192
MAX_ESTIMATED_COST_USD = 2.0
MAX_OBSERVED_COST_USD = 2.0
UPLOAD_WORKERS = 1
PROVIDER_RETRY_POLICY = {
    "max_retries": 1,
    "code": 503,
    "backoff_seconds": 10,
    "workers": UPLOAD_WORKERS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _artifact_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("_artifact_sha256", None)
    return body


def _write_json_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        existing = full._read_json(path)
        if _artifact_body(existing) != dict(value):
            raise ValueError(f"immutable artifact changed: {path.name}")
        return _sha256_file(path)
    return full._write_json(path, value)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _base_freeze(
    *, freeze_path: Path, manifest_path: Path, requests_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = Path(freeze_path).resolve(strict=True)
    if _sha256_file(freeze_path) != BASE_PREDICTIONS_SHA256:
        raise ValueError("original predictions freeze hash does not match approved source")
    freeze = full._read_json(freeze_path)
    if freeze.get("status") != "predictions_frozen":
        raise ValueError("original predictions freeze is not terminal")
    if freeze.get("model") != full.MODEL:
        raise ValueError("original predictions freeze model changed")
    manifest = full._read_json(Path(manifest_path).resolve(strict=True))
    manifest_sha256 = _sha256_file(Path(manifest_path))
    if freeze.get("manifest_sha256") != manifest_sha256:
        raise ValueError("original freeze is not bound to the supplied manifest")
    if manifest.get("input", {}).get("sha256") != _sha256_file(Path(requests_path)):
        raise ValueError("original manifest is not bound to the canonical requests input")
    return freeze, manifest


def selection_reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {reason: 0 for reason in EXPECTED_REASON_COUNTS}
    for row in rows:
        reason = row.get("rescue_source_failure_reason")
        if reason in counts:
            counts[reason] += 1
    return counts


def select_rescue_rows(
    *,
    freeze_path: Path = BASE_PREDICTIONS_PATH,
    manifest_path: Path = BASE_MANIFEST_PATH,
    requests_path: Path = REQUESTS_PATH,
    media_root: Path = MEDIA_ROOT,
) -> list[dict[str, Any]]:
    """Select failures from the freeze and bind them to label-free inputs."""

    freeze, manifest = _base_freeze(
        freeze_path=freeze_path, manifest_path=manifest_path, requests_path=requests_path
    )
    if freeze.get("row_count") != EXPECTED_PANEL_COUNT:
        raise ValueError("original panel row count changed")
    frozen_rows = freeze.get("rows")
    if not isinstance(frozen_rows, list):
        raise ValueError("original freeze rows are missing")
    selected_frozen = [
        row
        for row in frozen_rows
        if isinstance(row, Mapping)
        and row.get("failure_reason") in EXPECTED_REASON_COUNTS
    ]
    selected_ids = [row.get("query_id") for row in selected_frozen]
    if selected_ids != list(EXPECTED_QUERY_IDS):
        raise ValueError("failure selector did not reproduce the approved query IDs")
    if {
        reason: sum(row.get("failure_reason") == reason for row in selected_frozen)
        for reason in EXPECTED_REASON_COUNTS
    } != EXPECTED_REASON_COUNTS:
        raise ValueError("failure selector reason counts changed")
    if {
        int(row["query_id"]): row.get("failure_reason") for row in selected_frozen
    } != EXPECTED_REASON_BY_QUERY:
        raise ValueError("failure selector per-query reasons changed")
    if any(
        not isinstance(row.get("query_id"), int)
        or not isinstance(row.get("video_id"), str)
        for row in selected_frozen
    ):
        raise ValueError("selected freeze row binding is malformed")

    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list):
        raise ValueError("original manifest rows are missing")
    manifest_by_id = {
        row["query_id"]: row
        for row in manifest_rows
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), int)
    }
    loaded = full.load_input_rows(
        Path(requests_path).resolve(strict=True),
        media_root=Path(media_root),
        expected_count=EXPECTED_PANEL_COUNT,
    )
    loaded_by_id = {int(row["query_id"]): row for row in loaded}
    request_sha256 = _sha256_file(Path(requests_path))
    if manifest.get("input", {}).get("sha256") != request_sha256:
        raise ValueError("source requests input changed since the original freeze")

    selected: list[dict[str, Any]] = []
    for frozen in selected_frozen:
        query_id = int(frozen["query_id"])
        source = loaded_by_id.get(query_id)
        bound = manifest_by_id.get(query_id)
        if source is None or bound is None:
            raise ValueError(f"missing source binding for query {query_id}")
        for key in (
            "query_id",
            "video_id",
            "question",
            "availability",
            "source_media_sha256",
            "source_media_size_bytes",
        ):
            if source.get(key) != bound.get(key):
                raise ValueError(f"source binding changed for query {query_id}: {key}")
        if frozen["video_id"] != source["video_id"]:
            raise ValueError(f"freeze video binding changed for query {query_id}")
        if source.get("status") != "ready":
            raise ValueError(f"selected query {query_id} is not media-ready")
        row = dict(source)
        row["rescue_source_failure_reason"] = frozen["failure_reason"]
        selected.append(row)
    return selected


def _estimate_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row.get("status") == "ready"]
    seconds = sum(float(row.get("duration_seconds", 0.0)) for row in ready)
    prompt_tokens = math.ceil(seconds * full.ESTIMATED_INPUT_TOKENS_PER_SECOND) + (
        len(ready) * full.ESTIMATED_QUESTION_TOKENS
    )
    output_tokens = len(ready) * MAX_OUTPUT_TOKENS
    cost = (
        prompt_tokens / 1_000_000 * full.INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * full.OUTPUT_USD_PER_MILLION
    )
    return {
        "duration_seconds": round(seconds, 3),
        "prompt_tokens_ceiling": prompt_tokens,
        "output_tokens_ceiling": output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }


def build_rescue_request(
    *, row: Mapping[str, Any], upload: Mapping[str, Any]
) -> Any:
    """Build the held-fixed native-AV protocol with the enlarged output cap."""

    question = row.get("question")
    file_uri = upload.get("file_uri")
    mime_type = upload.get("mime_type")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("row question must be non-empty")
    if not isinstance(file_uri, str) or not file_uri:
        raise ValueError("upload URI must be non-empty")
    if not isinstance(mime_type, str) or not mime_type.startswith("video/"):
        raise ValueError("upload MIME must be video")
    video = full.types.Part(
        file_data=full.types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=full.types.VideoMetadata(fps=full.VIDEO_FPS),
        media_resolution=full.types.PartMediaResolution(level=full.MEDIA_RESOLUTION),
    )
    prompt = full.types.Part(text=f"{full.PROMPT}\n\nQuestion: {question.strip()}")
    return full.types.InlinedRequest(
        model=full.MODEL,
        contents=[full.types.Content(role="user", parts=[video, prompt])],
        metadata={
            "arm": "rescue_37flash_lowres_4fps_highthink_8192",
            "row_id": str(row["query_id"]),
            "video_id": str(row["video_id"]),
            "question": question.strip(),
            "file_uri": file_uri,
            "delivery_sha256": str(row.get("media_sha256", "")),
            "mime_type": mime_type,
            "video_fps": str(full.VIDEO_FPS),
            "media_resolution": full.MEDIA_RESOLUTION.value,
            "thinking_level": full.THINKING_LEVEL.value,
            "max_output_tokens": str(MAX_OUTPUT_TOKENS),
        },
        config=full.types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
            seed=42,
            thinking_config=full.types.ThinkingConfig(
                include_thoughts=False,
                thinking_level=full.THINKING_LEVEL,
            ),
        ),
    )


@contextmanager
def _rescue_protocol() -> Iterator[None]:
    """Temporarily adapt the proven engine, restoring every global afterward."""

    saved = {
        "MAX_OUTPUT_TOKENS": full.MAX_OUTPUT_TOKENS,
        "UPLOAD_WORKERS": full.UPLOAD_WORKERS,
        "PROVIDER_RETRY_POLICY": full.PROVIDER_RETRY_POLICY,
        "build_request": full.build_request,
    }
    full.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
    full.UPLOAD_WORKERS = UPLOAD_WORKERS
    full.PROVIDER_RETRY_POLICY = dict(PROVIDER_RETRY_POLICY)
    full.build_request = build_rescue_request
    try:
        yield
    finally:
        full.MAX_OUTPUT_TOKENS = saved["MAX_OUTPUT_TOKENS"]
        full.UPLOAD_WORKERS = saved["UPLOAD_WORKERS"]
        full.PROVIDER_RETRY_POLICY = saved["PROVIDER_RETRY_POLICY"]
        full.build_request = saved["build_request"]


def _write_root_manifest(
    *, output_root: Path, rows: Sequence[Mapping[str, Any]], estimate: Mapping[str, Any],
    requests_path: Path, freeze_path: Path, manifest_path: Path, created_at: str,
) -> str:
    body = {
        "schema_version": 1,
        "experiment_id": "deadline-submission-rescue-20260817-gemini-27",
        "status": "inputs_frozen",
        "created_at": created_at,
        "base_predictions_freeze_sha256": BASE_PREDICTIONS_SHA256,
        "base_predictions_freeze_path": str(Path(freeze_path).resolve()),
        "base_manifest_sha256": _sha256_file(Path(manifest_path)),
        "input": {"path": str(Path(requests_path).resolve()), "sha256": _sha256_file(Path(requests_path))},
        "model": full.MODEL,
        "prompt": full.PROMPT,
        "model_settings": {
            "video_fps": full.VIDEO_FPS,
            "media_resolution": full.MEDIA_RESOLUTION.value,
            "thinking_level": full.THINKING_LEVEL.value,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "provider_retry_policy": dict(PROVIDER_RETRY_POLICY),
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "selected_reason_counts": selection_reason_counts(rows),
        "row_count": len(rows),
        "unique_video_count": len({str(row["video_id"]) for row in rows}),
        "cost": dict(estimate),
        "max_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        "max_observed_cost_usd": MAX_OBSERVED_COST_USD,
        "provider_artifacts_root": str((output_root / "provider-artifacts").resolve()),
        "rows": [
            dict(bound)
            | {
                "input_sha256": _sha256_file(Path(requests_path)),
                "rescue_source_failure_reason": EXPECTED_REASON_BY_QUERY[
                    int(bound["query_id"])
                ],
            }
            for bound in _selected_manifest_bindings(
                manifest_path=manifest_path, selected_ids=EXPECTED_QUERY_IDS
            )
        ],
    }
    return _write_json_once(output_root / "manifest.json", body)


def _selected_manifest_bindings(
    *, manifest_path: Path, selected_ids: Sequence[int]
) -> list[Mapping[str, Any]]:
    manifest = full._read_json(Path(manifest_path).resolve(strict=True))
    by_id = {
        int(row["query_id"]): row
        for row in manifest.get("rows", [])
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), int)
    }
    try:
        return [by_id[query_id] for query_id in selected_ids]
    except KeyError as error:
        raise ValueError("rescue manifest is missing a selected source binding") from error


def _finalize(
    *, output_root: Path, rows: Sequence[Mapping[str, Any]], estimate: Mapping[str, Any],
    rescue_manifest_sha256: str, original_base_manifest_sha256: str,
) -> dict[str, Any]:
    core_root = output_root / "provider-artifacts"
    core_predictions = full._read_json(core_root / "predictions-freeze.json")
    core_summary = full._read_json(core_root / "summary.json")
    if core_predictions.get("status") != "predictions_frozen":
        raise ValueError("rescue provider artifacts did not reach predictions_frozen")
    allowed_statuses = {
        "succeeded",
        "failed",
        "upload_failed",
        "unavailable",
        "unsupported",
        "cost_cap_blocked",
    }
    if any(row.get("status") not in allowed_statuses for row in core_predictions["rows"]):
        raise ValueError("rescue predictions contain a non-terminal row status")
    predictions = {
        **_artifact_body(core_predictions),
        "experiment_id": "deadline-submission-rescue-20260817-gemini-27",
        "manifest_sha256": rescue_manifest_sha256,
        "base_predictions_freeze_sha256": BASE_PREDICTIONS_SHA256,
        "base_manifest_sha256": original_base_manifest_sha256,
        "source_manifest_sha256": full._sha256_file(core_root / "manifest.json"),
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "provider_retry_policy": dict(PROVIDER_RETRY_POLICY),
        "unique_video_count": len({str(row["video_id"]) for row in rows}),
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "estimated_cost_cap_usd": MAX_ESTIMATED_COST_USD,
    }
    predictions_sha256 = _write_json_once(output_root / "predictions-freeze.json", predictions)
    summary = {
        "schema_version": 1,
        "experiment_id": predictions["experiment_id"],
        "status": predictions["status"],
        "row_count": predictions["row_count"],
        "completed_count": predictions["completed_count"],
        "upload_count": core_summary.get("upload_count", 0),
        "request_count": len(rows),
        "unique_video_count": predictions["unique_video_count"],
        "failed_count": sum(row.get("status") == "failed" for row in predictions["rows"]),
        "estimated_cost_usd": estimate["estimated_cost_usd"],
        "estimated_cost_cap_usd": MAX_ESTIMATED_COST_USD,
        "observed_cost_usd": predictions.get("observed_cost_usd"),
        "observed_cost_cap_usd": MAX_OBSERVED_COST_USD,
        "predictions_freeze_sha256": predictions_sha256,
        "provider_predictions_freeze_sha256": full._sha256_file(core_root / "predictions-freeze.json"),
    }
    _write_json_once(output_root / "summary.json", summary)
    return summary


def run_rescue(
    *,
    freeze_path: Path = BASE_PREDICTIONS_PATH,
    manifest_path: Path = BASE_MANIFEST_PATH,
    requests_path: Path = REQUESTS_PATH,
    media_root: Path = MEDIA_ROOT,
    output_root: Path = PRIVATE_RUN_ROOT,
    client: Any | None = None,
    dry_run: bool | None = None,
    run: bool = False,
    max_estimated_cost_usd: float = MAX_ESTIMATED_COST_USD,
    max_observed_cost_usd: float = MAX_OBSERVED_COST_USD,
    created_at: str | None = None,
    sleep: Any = full.time.sleep,
) -> dict[str, Any]:
    if dry_run is None:
        dry_run = not run
    if dry_run and run:
        raise ValueError("dry_run and run are mutually exclusive")
    if not dry_run and not run:
        raise ValueError("live rescue requires run=True")
    if max_estimated_cost_usd > MAX_ESTIMATED_COST_USD:
        raise ValueError("estimated cap cannot exceed approved $2.00 cap")
    if max_observed_cost_usd > MAX_OBSERVED_COST_USD:
        raise ValueError("observed cap cannot exceed approved $2.00 cap")
    rows = select_rescue_rows(
        freeze_path=freeze_path,
        manifest_path=manifest_path,
        requests_path=requests_path,
        media_root=media_root,
    )
    estimate = _estimate_cost(rows)
    if estimate["estimated_cost_usd"] > max_estimated_cost_usd:
        raise ValueError(
            f"estimated cost {estimate['estimated_cost_usd']:.8f} exceeds cost cap "
            f"{max_estimated_cost_usd:.8f}"
        )
    result = {
        "status": "dry_run",
        "row_count": len(rows),
        "selected_query_ids": list(EXPECTED_QUERY_IDS),
        "unique_video_count": len({str(row["video_id"]) for row in rows}),
        "native_request_count": len(rows),
        "upload_count": len({str(row["video_id"]) for row in rows}),
        "request_count": len(rows),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "estimated_cost_cap_usd": max_estimated_cost_usd,
        "observed_cost_cap_usd": max_observed_cost_usd,
        **estimate,
    }
    if dry_run:
        return result
    if client is None:
        raise ValueError("provider client is required for a live rescue")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest_path = output_root / "manifest.json"
    if root_manifest_path.exists():
        existing_manifest = full._read_json(root_manifest_path)
        existing_created_at = existing_manifest.get("created_at")
        if not isinstance(existing_created_at, str) or not existing_created_at:
            raise ValueError("existing rescue manifest has no creation timestamp")
        created = existing_created_at
    else:
        created = created_at or _now()
    source_manifest_sha256 = _write_root_manifest(
        output_root=output_root,
        rows=rows,
        estimate=estimate,
        requests_path=Path(requests_path),
        freeze_path=Path(freeze_path),
        manifest_path=Path(manifest_path),
        created_at=created,
    )
    core_root = output_root / "provider-artifacts"
    with _rescue_protocol():
        full_result = full._run_live(
            rows=rows,
            requests_path=Path(requests_path).resolve(),
            output_root=core_root,
            client=client,
            estimate=estimate,
            max_estimated_cost_usd=max_estimated_cost_usd,
            max_observed_cost_usd=max_observed_cost_usd,
            created_at=created,
            sleep=sleep,
            parser=full.parse_response,
        )
    del full_result
    return _finalize(
        output_root=output_root,
        rows=rows,
        estimate=estimate,
        rescue_manifest_sha256=source_manifest_sha256,
        original_base_manifest_sha256=_sha256_file(Path(manifest_path)),
    )


def client_factory(*, api_key: str) -> Any:
    return full.client_factory(api_key=api_key)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_vqa_deadline_gemini_rescue")
    parser.add_argument("--run", action="store_true", help="perform provider calls")
    parser.add_argument("--dry-run", action="store_true", help="validate and estimate only")
    parser.add_argument("--freeze", type=Path, default=BASE_PREDICTIONS_PATH)
    parser.add_argument("--manifest", type=Path, default=BASE_MANIFEST_PATH)
    parser.add_argument("--requests", type=Path, default=REQUESTS_PATH)
    parser.add_argument("--media-root", type=Path, default=MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=PRIVATE_RUN_ROOT)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=MAX_ESTIMATED_COST_USD)
    parser.add_argument("--max-observed-cost-usd", type=float, default=MAX_OBSERVED_COST_USD)
    return parser


def main(argv: Sequence[str] | None = None, *, client_factory_override: Any = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.run and args.dry_run:
            raise ValueError("--run and --dry-run are mutually exclusive")
        is_live = bool(args.run)
        client = None if not is_live else (client_factory_override or client_factory)(api_key=full._private_api_key())
        result = run_rescue(
            freeze_path=args.freeze,
            manifest_path=args.manifest,
            requests_path=args.requests,
            media_root=args.media_root,
            output_root=args.output,
            client=client,
            dry_run=not is_live,
            run=is_live,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            max_observed_cost_usd=args.max_observed_cost_usd,
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
